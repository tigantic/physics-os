#!/usr/bin/env python3
"""Stress test: decaying turbulence with band-limited random vorticity.

Uses seeded random Fourier modes to produce a multi-mode IC that
immediately saturates the QTT rank budget through nonlinear advection
(Hadamard products).  The K=5 default places the IC at ~15% of the
rank cap, leaving enough headroom for Poisson CG to converge while
stressing the full advection → Poisson → Helmholtz pipeline.

Why K=5:
  - IC rank ~10 after rSVD truncation with k⁻² spectrum
  - Poisson CG intermediates: 5 × rank(ψ) ≈ 50 < 64 cap
  - Advection Hadamard products immediately fill rank to 64
  - Conservation preserved despite CG truncation noise floor
  - Higher K (≥8) causes CG divergence: Poisson residual > 1.0

Monitored quantities:
  - chi (max bond dim) per step vs rank cap
  - saturation rate (fraction of steps at rank cap)
  - wall-clock time per step (detect drift / blowup)
  - total circulation conservation (invariant)
  - divergence proxy: Poisson residual ∝ ||∇·u||
  - Poisson residual trend (decreasing = healthy)

Verdict levels:
  PASS — conservation OK, all CG converges, step time stable
  WARN — conservation OK, CG or saturation issues (truncation floor)
  FAIL — conservation blown (> 1e-4) or step time blowup (> 3×)

Usage:
    python3 run_stress_turbulence.py --n-bits 9              # 512²
    python3 run_stress_turbulence.py --n-bits 10             # 1024²
    python3 run_stress_turbulence.py --n-bits 9 --t-final 0.05
    python3 run_stress_turbulence.py --n-bits 9 --n-modes 8  # harder

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time

# Physical constants
VISCOSITY = 0.01
IC_SEED = 42
IC_TYPE = "stress_decaying_turbulence_seeded"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stress test: decaying turbulence (IMEX-CNAB2)",
    )
    p.add_argument("--n-bits", type=int, default=9,
                   help="Bits per dim (default: 9 → 512²)")
    p.add_argument("--t-final", type=float, default=0.05,
                   help="Physical end time (default: 0.05)")
    p.add_argument("--n-modes", type=int, default=5,
                   help="Fourier modes K per dim (default: 5, "
                        "rank ~10 IC → CG works, advection saturates)")
    p.add_argument("--max-rank", type=int, default=64,
                   help="Max rank ceiling (default: 64)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for IC (default: 42)")
    p.add_argument("--json", action="store_true",
                   help="Write JSON results file")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n_bits = args.n_bits
    t_final = args.t_final
    n_modes = args.n_modes
    max_rank = args.max_rank
    seed = args.seed
    grid_size = 2 ** n_bits

    # ── Enable logging ──────────────────────────────────────────────
    for logger_name in [
        "ontic.engine.vm.gpu_runtime",
        "ontic.engine.vm.gpu_operators",
    ]:
        _logger = logging.getLogger(logger_name)
        if not _logger.handlers:
            _handler = logging.StreamHandler(sys.stdout)
            _handler.setFormatter(logging.Formatter("  %(message)s"))
            _logger.addHandler(_handler)
            _logger.setLevel(logging.INFO)

    print("=" * 72)
    print("  STRESS TEST: Decaying Turbulence (seeded)")
    print(f"  Grid: {grid_size}² | K={n_modes} modes | seed={seed}")
    print(f"  Target: t={t_final} | ν={VISCOSITY} | rank cap={max_rank}")
    print("=" * 72)
    print()

    # ── Compile ─────────────────────────────────────────────────────
    print("[1/5] Compiling IMEX-CNAB2 (stress IC)...")
    t0 = time.perf_counter()

    from ontic.engine.vm.compilers.navier_stokes_2d_imex import (
        NavierStokes2DImexCompiler,
    )

    # First pass: get dt to compute n_steps
    compiler_probe = NavierStokes2DImexCompiler(
        n_bits=n_bits,
        n_steps=1,
        viscosity=VISCOSITY,
        ic_type=IC_TYPE,
        ic_n_modes=n_modes,
        ic_seed=seed,
    )
    probe_prog = compiler_probe.compile()
    dt = probe_prog.dt
    n_steps = max(1, math.ceil(t_final / dt))

    # Compile with correct n_steps
    compiler = NavierStokes2DImexCompiler(
        n_bits=n_bits,
        n_steps=n_steps,
        viscosity=VISCOSITY,
        ic_type=IC_TYPE,
        ic_n_modes=n_modes,
        ic_seed=seed,
    )
    program = compiler.compile()
    t_compile = time.perf_counter() - t0

    dt_explicit = 0.25 * (1.0 / grid_size) ** 2 / (2.0 * VISCOSITY)
    speedup = dt / dt_explicit

    print(f"  ✓ dt = {dt:.6e} ({speedup:.0f}× vs explicit)")
    print(f"  ✓ n_steps = {n_steps} → T_final = {dt * n_steps:.6e}")
    print(f"  ✓ K={n_modes} → {n_modes**2} Fourier modes "
          f"(rank ≤{n_modes**2} before truncation)")
    print(f"  ✓ {len(program.instructions)} instructions, "
          f"{program.n_registers} registers")
    print(f"  ✓ Compile: {t_compile:.3f}s")
    print()

    # ── Execute ─────────────────────────────────────────────────────
    print("[2/5] Executing on GPU...")

    import torch
    if not torch.cuda.is_available():
        print("  ✗ CUDA not available")
        return 1

    from ontic.engine.vm.gpu_runtime import GPURuntime, GPURankGovernor

    governor = GPURankGovernor(
        max_rank=max_rank,
        adaptive=True,
        base_rank=max_rank,
        min_rank=4,
        rel_tol=1e-10,
    )
    runtime = GPURuntime(governor=governor)

    t0 = time.perf_counter()
    result = runtime.execute(program)
    t_exec = time.perf_counter() - t0

    torch.cuda.synchronize()
    print()
    print(f"  ✓ Execution: {t_exec:.1f}s ({t_exec/n_steps:.2f}s/step)")
    if not result.success:
        print(f"  ✗ Error: {result.error}")
        return 1
    print()

    # ── Final field ranks ───────────────────────────────────────────
    omega_final_rank = -1
    psi_final_rank = -1
    if hasattr(result, "fields") and result.fields:
        for fname, ftensor in result.fields.items():
            if fname == "omega":
                omega_final_rank = ftensor.max_rank
            elif fname == "psi":
                psi_final_rank = ftensor.max_rank

    # ── Per-step analysis ───────────────────────────────────────────
    print("[3/5] Per-step telemetry...")

    tel = result.telemetry
    steps = tel.steps

    # Extract per-step data
    chi_per_step = [s.chi_max for s in steps]
    time_per_step = [s.wall_time_s for s in steps]
    inv_per_step = []
    for s in steps:
        iv = s.invariant_values
        if iv:
            inv_per_step.append(next(iter(iv.values())))
        else:
            inv_per_step.append(float("nan"))

    # Probes (CG diagnostics)
    probes = result.probes if hasattr(result, "probes") else {}
    p_iters = probes.get("poisson_cg_iters", [])
    p_resid = probes.get("poisson_relative_residual", [])
    p_conv = probes.get("poisson_converged", [])
    h_iters = probes.get("helmholtz_cg_iters", [])
    h_resid = probes.get("helmholtz_relative_residual", [])
    h_conv = probes.get("helmholtz_converged", [])

    # ── Chi growth analysis ─────────────────────────────────────────
    print()
    print("  ── Chi (max bond dim) per step ──")
    print(f"  {'Step':>6s}  {'chi':>4s}  {'cap':>4s}  "
          f"{'sat?':>5s}  {'t/step':>8s}  {'P_iters':>7s}  "
          f"{'P_resid':>9s}  {'H_iters':>7s}  {'H_resid':>9s}")
    print(f"  {'─' * 6}  {'─' * 4}  {'─' * 4}  "
          f"{'─' * 5}  {'─' * 8}  {'─' * 7}  "
          f"{'─' * 9}  {'─' * 7}  {'─' * 9}")

    n_saturated = 0
    for i, chi in enumerate(chi_per_step):
        saturated = chi >= max_rank
        if saturated:
            n_saturated += 1
        sat_mark = "  YES" if saturated else "   no"

        step_t = time_per_step[i] if i < len(time_per_step) else 0.0
        pi = int(p_iters[i]) if i < len(p_iters) else -1
        pr = p_resid[i] if i < len(p_resid) else float("nan")
        hi = int(h_iters[i]) if i < len(h_iters) else -1
        hr = h_resid[i] if i < len(h_resid) else float("nan")

        print(f"  {i + 1:6d}  {chi:4d}  {max_rank:4d}  "
              f"{sat_mark}  {step_t:8.3f}  {pi:7d}  "
              f"{pr:9.2e}  {hi:7d}  {hr:9.2e}")

    # ── Summary statistics ──────────────────────────────────────────
    print()
    print("[4/5] Summary statistics...")
    print()

    sat_rate = n_saturated / max(len(chi_per_step), 1)
    chi_max_overall = max(chi_per_step) if chi_per_step else 0
    chi_mean = sum(chi_per_step) / len(chi_per_step) if chi_per_step else 0

    # Conservation
    inv_initial = getattr(tel, "invariant_initial", None)
    inv_final = getattr(tel, "invariant_final", None)
    inv_error = getattr(tel, "invariant_error", None)
    if inv_error is not None:
        conservation = inv_error
    elif inv_initial is not None and inv_final is not None:
        conservation = abs(inv_final - inv_initial)
    else:
        conservation = float("nan")

    conservation_ok = conservation < 1e-4

    # Step time drift
    if len(time_per_step) > 5:
        first5 = sum(time_per_step[:5]) / 5
        last5 = sum(time_per_step[-5:]) / 5
        time_drift = last5 / first5 if first5 > 0 else 1.0
    else:
        first5 = (sum(time_per_step) / len(time_per_step)
                  if time_per_step else 0)
        last5 = first5
        time_drift = 1.0

    # Divergence proxy: max Poisson residual
    div_proxy = max(p_resid) if p_resid else float("nan")

    # Poisson residual trend: first half vs second half
    if len(p_resid) >= 4:
        half = len(p_resid) // 2
        p_resid_first_half = sum(p_resid[:half]) / half
        p_resid_second_half = (sum(p_resid[half:])
                               / (len(p_resid) - half))
        p_trend = (p_resid_second_half / p_resid_first_half
                   if p_resid_first_half > 0 else 1.0)
        p_trend_label = ("improving" if p_trend < 0.9
                         else "stable" if p_trend < 1.1
                         else "degrading")
    else:
        p_trend = 1.0
        p_trend_label = "N/A"

    # All converged?
    all_poisson_conv = all(c > 0.5 for c in p_conv) if p_conv else True
    all_helmholtz_conv = all(c > 0.5 for c in h_conv) if h_conv else True
    helmholtz_conv_rate = (
        sum(1 for c in h_conv if c > 0.5) / len(h_conv)
        if h_conv else 1.0
    )

    print(f"  Chi growth:")
    print(f"    peak chi:        {chi_max_overall} / {max_rank} cap")
    print(f"    mean chi:        {chi_mean:.1f}")
    print(f"    saturation rate: {sat_rate:.1%}"
          f" ({n_saturated}/{len(chi_per_step)} steps)")
    print(f"    scaling class:   {tel.classify_scaling()}")
    if omega_final_rank >= 0:
        print(f"    omega final rank: {omega_final_rank}")
    if psi_final_rank >= 0:
        print(f"    psi final rank:   {psi_final_rank}")
    print()
    print(f"  Conservation:")
    print(f"    ΔΓ:              {conservation:.2e}")
    print(f"    OK (< 1e-4):     {conservation_ok}")
    print()
    print(f"  Divergence proxy (max Poisson ||r||/||b||):")
    print(f"    {div_proxy:.2e}")
    print(f"    Poisson trend:   {p_trend:.2f}× ({p_trend_label})")
    print()
    mean_time = (sum(time_per_step) / len(time_per_step)
                 if time_per_step else 0)
    print(f"  Step time:")
    print(f"    mean:            {mean_time:.3f}s")
    print(f"    first 5 avg:     {first5:.3f}s")
    print(f"    last 5 avg:      {last5:.3f}s")
    print(f"    drift ratio:     {time_drift:.2f}× "
          f"({'stable' if time_drift < 1.5 else 'DRIFTING'})")
    print()
    print(f"  Solvers:")
    if p_iters:
        print(f"    Poisson:  mean {sum(p_iters) / len(p_iters):.1f} iters, "
              f"all converged: {all_poisson_conv}")
    else:
        print("    Poisson:  no data")
    if h_iters:
        print(f"    Helmholtz: mean {sum(h_iters) / len(h_iters):.1f} iters, "
              f"all converged: {all_helmholtz_conv} "
              f"({helmholtz_conv_rate:.0%})")
    else:
        print("    Helmholtz: no data")
    print()

    # ── Verdict ─────────────────────────────────────────────────────
    # Three levels:
    #   PASS — conservation OK + all CG converges + step time stable
    #   WARN — conservation OK but CG truncation floor or saturation
    #   FAIL — conservation blown OR step time blowup (primary failure)
    print("[5/5] Verdict")
    print("=" * 72)

    primary_fail: list[str] = []
    secondary_warn: list[str] = []

    # Primary failure indicators (any one → FAIL)
    if not conservation_ok:
        primary_fail.append(
            f"conservation drift {conservation:.2e} > 1e-4")
    if time_drift > 3.0:
        primary_fail.append(
            f"step time blowup {time_drift:.1f}× > 3×")
    if p_resid and max(p_resid) > 1.0:
        primary_fail.append(
            f"Poisson CG divergent (max resid {max(p_resid):.2e} > 1.0)")

    # Secondary indicators (CG convergence, saturation — expected
    # under stress due to QTT truncation noise floor)
    if not all_poisson_conv and not primary_fail:
        secondary_warn.append(
            f"Poisson CG at truncation floor "
            f"(resid {min(p_resid):.2e}–{max(p_resid):.2e}, "
            f"trend: {p_trend_label})")
    if not all_helmholtz_conv:
        secondary_warn.append(
            f"Helmholtz CG: {helmholtz_conv_rate:.0%} converged")
    if sat_rate > 0.5:
        secondary_warn.append(
            f"saturation rate {sat_rate:.0%} "
            f"(expected for multi-mode IC)")
    if time_drift > 1.5:
        secondary_warn.append(
            f"step time drift {time_drift:.2f}×")

    if primary_fail:
        status = "FAIL"
        print(f"  Status: FAIL")
        for msg in primary_fail:
            print(f"    ✗ {msg}")
        for msg in secondary_warn:
            print(f"    ⚠ {msg}")
    elif secondary_warn:
        status = "WARN"
        print(f"  Status: WARN (physics OK, CG at truncation floor)")
        for msg in secondary_warn:
            print(f"    ⚠ {msg}")
        print(f"    ✓ Conservation: {conservation:.2e}")
    else:
        status = "PASS"
        print("  Status: PASS")
        print("    ✓ Conservation, solver convergence, timing — all OK")

    print(f"\n  Grid:        {grid_size}² ({grid_size ** 2:,} points)")
    print(f"  IC:          {IC_TYPE} (K={n_modes}, seed={seed})")
    print(f"  Steps:       {n_steps}")
    print(f"  T_final:     {dt * n_steps:.6e}")
    print(f"  Wall clock:  {t_exec:.1f}s")
    print("=" * 72)

    # ── JSON ────────────────────────────────────────────────────────
    if args.json:
        results = {
            "test": "stress_decaying_turbulence_seeded",
            "grid": f"{grid_size}x{grid_size}",
            "n_bits": n_bits,
            "n_steps": n_steps,
            "n_modes": n_modes,
            "seed": seed,
            "dt": dt,
            "t_final": dt * n_steps,
            "wall_clock_s": round(t_exec, 2),
            "conservation": conservation,
            "conservation_ok": conservation_ok,
            "chi_per_step": chi_per_step,
            "chi_max": chi_max_overall,
            "chi_mean": round(chi_mean, 1),
            "saturation_rate": round(sat_rate, 4),
            "scaling_class": tel.scaling_class,
            "time_per_step": [round(t, 4) for t in time_per_step],
            "time_drift_ratio": round(time_drift, 3),
            "divergence_proxy": div_proxy,
            "poisson_iters": [int(x) for x in p_iters],
            "poisson_residuals": [float(x) for x in p_resid],
            "poisson_trend": round(p_trend, 3),
            "poisson_trend_label": p_trend_label,
            "helmholtz_iters": [int(x) for x in h_iters],
            "helmholtz_residuals": [float(x) for x in h_resid],
            "helmholtz_conv_rate": round(helmholtz_conv_rate, 4),
            "omega_final_rank": omega_final_rank,
            "psi_final_rank": psi_final_rank,
            "status": status,
        }
        out_path = f"stress_turb_{grid_size}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2,
                      default=lambda o: None if math.isnan(o) else o)
        print(f"\n  JSON → {out_path}")

    return 0 if status in ("PASS", "WARN") else 1


if __name__ == "__main__":
    sys.exit(main())
