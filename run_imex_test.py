#!/usr/bin/env python3
"""IMEX-CNAB2 test run: 4096² × 100 steps.

Tests the IMEX (Crank–Nicolson + Adams–Bashforth 2) time integration
for 2D Navier–Stokes.  Compares timestep size and physics quality
against the explicit Euler baseline.

IMEX advantage: dt is CFL-limited by advection (dt ∝ h), not by
diffusion (dt ∝ h²).  At 4096², this is ~1000× larger timestep.

Usage:
    python3 run_imex_test.py                     # Default: 4096², 100 steps
    python3 run_imex_test.py --n-bits 9          # Quick: 512², 100 steps
    python3 run_imex_test.py --n-bits 12 --n-steps 200

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time

# ── Constants ────────────────────────────────────────────────────────
VISCOSITY = 0.01


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IMEX-CNAB2 test run for 2D Navier–Stokes",
    )
    p.add_argument("--n-bits", type=int, default=12,
                   help="Bits per dim (default: 12 → 4096²)")
    p.add_argument("--n-steps", type=int, default=100,
                   help="Time steps (default: 100)")
    p.add_argument("--max-rank", type=int, default=64,
                   help="Max rank ceiling (default: 64)")
    p.add_argument("--json", action="store_true",
                   help="Output JSON results file")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n_bits = args.n_bits
    n_steps = args.n_steps
    max_rank = args.max_rank
    grid_size = 2 ** n_bits

    print("=" * 72)
    print("  IMEX-CNAB2 TEST RUN")
    print(f"  2D Navier–Stokes (vorticity-stream) — {grid_size}² × {n_steps} steps")
    print("=" * 72)
    print()

    # ── Enable runtime logging ──────────────────────────────────────
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

    # ── Compile IMEX program ────────────────────────────────────────
    print("[1/4] Compiling IMEX-CNAB2 program...")
    t0 = time.perf_counter()

    from ontic.engine.vm.compilers.navier_stokes_2d_imex import (
        NavierStokes2DImexCompiler,
    )

    compiler = NavierStokes2DImexCompiler(
        n_bits=n_bits,
        n_steps=n_steps,
        viscosity=VISCOSITY,
    )
    program = compiler.compile()
    t_compile = time.perf_counter() - t0

    N = 2 ** n_bits
    h = 1.0 / N
    dt_explicit = 0.25 * h * h / (2.0 * VISCOSITY)
    dt_imex = program.dt
    speedup = dt_imex / dt_explicit

    print(f"  ✓ Compiled: {len(program.instructions)} instructions, "
          f"{program.n_registers} registers")
    print(f"  ✓ dt (IMEX):     {dt_imex:.6e}")
    print(f"  ✓ dt (explicit): {dt_explicit:.6e}")
    print(f"  ✓ IMEX timestep advantage: {speedup:.0f}×")
    print(f"  ✓ Helmholtz α = dt·ν/2 = {dt_imex * VISCOSITY / 2.0:.6e}")
    print(f"  ✓ Compile time: {t_compile:.3f}s")
    print()

    # ── Execute on GPU ──────────────────────────────────────────────
    print("[2/4] Executing on GPU...")
    t0 = time.perf_counter()

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
    result = runtime.execute(program)
    t_exec = time.perf_counter() - t0

    torch.cuda.synchronize()
    print()
    print(f"  ✓ Execution time: {t_exec:.1f}s")
    print(f"  ✓ Success: {result.success}")
    if not result.success:
        print(f"  ✗ Error: {result.error}")
        return 1
    print()

    # ── Evaluate physics quality ────────────────────────────────────
    print("[3/4] Evaluating physics quality...")

    tel = result.telemetry

    # Conservation: invariant_error from runtime (|ΔΓ/Γ₀|)
    inv_initial = getattr(tel, "invariant_initial", None)
    inv_final = getattr(tel, "invariant_final", None)
    inv_error = getattr(tel, "invariant_error", None)

    if inv_initial is not None and inv_final is not None:
        abs_drift = abs(inv_final - inv_initial)
        if inv_error is not None and inv_error > 0:
            max_deviation = inv_error
        else:
            max_deviation = abs_drift / max(abs(inv_initial), 1e-30)
        conservation_ok = max_deviation < 1e-4
    else:
        max_deviation = float("nan")
        conservation_ok = False

    # Compression ratio
    total_grid = (2 ** n_bits) ** 2
    peak_rank = getattr(tel, "peak_rank", max_rank) or max_rank
    n_cores = 2 * n_bits
    storage_tt = n_cores * peak_rank * peak_rank * 2  # approx
    compression = total_grid / max(storage_tt, 1)

    print(f"  ✓ Conservation (ΔΓ): {max_deviation:.2e}")
    print(f"  ✓ Conservation OK:   {conservation_ok}")
    print(f"  ✓ Compression ratio: {compression:,.0f}×")
    print(f"  ✓ Peak rank:         {peak_rank}")
    print()

    # ── Helmholtz solver stats ──────────────────────────────────────
    probes = result.probes if hasattr(result, "probes") else {}
    h_iters = probes.get("helmholtz_cg_iters", [])
    h_conv = probes.get("helmholtz_converged", [])
    p_iters = probes.get("poisson_cg_iters", [])
    p_conv = probes.get("poisson_converged", [])

    if h_iters:
        print("  Helmholtz CG stats:")
        print(f"    Mean iters:  {sum(h_iters)/len(h_iters):.1f}")
        print(f"    Max iters:   {max(h_iters):.0f}")
        print(f"    All converged: {all(c > 0.5 for c in h_conv)}")
    if p_iters:
        print("  Poisson CG stats:")
        print(f"    Mean iters:  {sum(p_iters)/len(p_iters):.1f}")
        print(f"    Max iters:   {max(p_iters):.0f}")
        print(f"    All converged: {all(c > 0.5 for c in p_conv)}")
    print()

    # ── Summary ─────────────────────────────────────────────────────
    print("[4/4] Summary")
    print("=" * 72)
    all_ok = result.success and conservation_ok
    status = "PASS" if all_ok else "FAIL"
    print(f"  Grid:           {grid_size}² ({grid_size**2:,} points)")
    print(f"  Steps:          {n_steps}")
    print(f"  Time scheme:    IMEX-CNAB2")
    print(f"  dt:             {dt_imex:.6e} ({speedup:.0f}× vs explicit)")
    print(f"  T_final:        {dt_imex * n_steps:.6e}")
    print(f"  Wall clock:     {t_exec:.1f}s")
    print(f"  Conservation:   {max_deviation:.2e}")
    print(f"  Compression:    {compression:,.0f}×")
    print(f"  Status:         {status}")
    print("=" * 72)

    # ── JSON output ─────────────────────────────────────────────────
    if args.json:
        results = {
            "test": "imex_cnab2",
            "grid": f"{grid_size}x{grid_size}",
            "n_bits": n_bits,
            "n_steps": n_steps,
            "time_scheme": "IMEX-CNAB2",
            "dt_imex": dt_imex,
            "dt_explicit": dt_explicit,
            "dt_speedup": speedup,
            "t_final": dt_imex * n_steps,
            "wall_clock_s": round(t_exec, 2),
            "conservation_deviation": max_deviation,
            "conservation_ok": conservation_ok,
            "compression_ratio": round(compression),
            "peak_rank": peak_rank,
            "helmholtz_mean_iters": (
                round(sum(h_iters) / len(h_iters), 1) if h_iters else None
            ),
            "poisson_mean_iters": (
                round(sum(p_iters) / len(p_iters), 1) if p_iters else None
            ),
            "status": status,
        }
        out_path = f"imex_test_{grid_size}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  JSON results → {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
