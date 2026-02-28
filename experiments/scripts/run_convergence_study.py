#!/usr/bin/env python3
"""QTT Physics VM — Time-Step Convergence Study.

Determines the minimum number of time steps required for Maxwell 3D
antenna metrics (S₁₁, gain, bandwidth) to converge.

Runs a single dipole design at 64³ (cheap per-step) with geometrically
increasing step counts to empirically find the convergence threshold.

Physics rationale
-----------------
CFL time step:  dt = 0.3 · h / (c · √3),  h = 1/N
Source pulse:    Gaussian envelope peaking at t_peak = 3/(2π·BW) ≈ 0.955
Minimum steps to reach pulse peak: t_peak / dt ≈ 5.5 · N

At N=64 :  ~353 steps to reach pulse peak
At N=4096: ~22,577 steps to reach pulse peak

This study sweeps step counts from 50 → 3000 at N=64 to find the
step-count-to-grid-size ratio (steps/N) where metrics converge.
That ratio then predicts the required steps at any grid size.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from typing import Any


def main() -> None:
    t_global = time.perf_counter()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── Configuration ────────────────────────────────────────────────
    N_BITS = 6              # 64³ — fast per-step for convergence sweep
    MAX_RANK = 48
    SEED = 42

    # Step counts to test (geometrically spaced)
    STEP_COUNTS = [50, 100, 200, 350, 500, 750, 1000, 1500, 2000, 3000]

    N = 2 ** N_BITS
    h = 1.0 / N
    c = 1.0
    dt_cfl = 0.3 * h / (c * math.sqrt(3.0))
    freq_center = 1.0
    freq_bw = 0.5
    tau = 1.0 / (2.0 * math.pi * freq_bw)
    t_peak = 3.0 * tau
    steps_to_peak = int(math.ceil(t_peak / dt_cfl))

    OUTPUT = f"convergence_study_{ts}.json"

    print("=" * 80)
    print("  HYPERTENSOR-VM  —  TIME-STEP CONVERGENCE STUDY")
    print("=" * 80)
    print(f"  Grid:            {N}³  ({N**3:,} DOFs)")
    print(f"  CFL dt:          {dt_cfl:.6e}")
    print(f"  Source τ:        {tau:.4f}")
    print(f"  Pulse peak at:   t = {t_peak:.4f}  ({steps_to_peak} steps)")
    print(f"  Steps to test:   {STEP_COUNTS}")
    print(f"  Max rank:        {MAX_RANK}")
    print(f"  Output:          {OUTPUT}")
    print("=" * 80)
    print()
    sys.stdout.flush()

    # ── Imports ──────────────────────────────────────────────────────
    print("Loading QTT antenna pipeline...", flush=True)
    import torch
    from ontic.engine.vm.antenna import (
        DipoleAntennaDesign,
        SweepOrchestrator,
    )

    # Use a fixed dipole design
    design = DipoleAntennaDesign(
        freq_center=freq_center,
        freq_bandwidth=freq_bw,
    )
    space = design.design_space
    param_list = space.random_points(n=1, seed=SEED)
    params = param_list[0]
    pstr = ", ".join(f"{k}={v:.4f}" for k, v in params.items())
    print(f"  Fixed dipole:    {pstr}")
    print()
    sys.stdout.flush()

    # ── Run convergence sweep ────────────────────────────────────────
    results: list[dict[str, Any]] = []

    for i, n_steps in enumerate(STEP_COUNTS):
        t_sim = n_steps * dt_cfl
        t_ratio = t_sim / t_peak
        steps_per_N = n_steps / N

        print(f"\n{'─'*72}")
        print(f"  [{i+1}/{len(STEP_COUNTS)}]  n_steps = {n_steps}")
        print(f"     t_sim = {t_sim:.4f}  "
              f"({t_ratio:.2f}× pulse peak, "
              f"{steps_per_N:.1f} steps/N)")
        print(f"{'─'*72}")
        sys.stdout.flush()

        orchestrator = SweepOrchestrator(
            n_bits=N_BITS,
            n_steps=n_steps,
            max_rank=MAX_RANK,
            extract_far_field=True,
            n_surface_samples=8,
            n_theta=37,
            n_phi=18,
            verbose=True,
        )

        t0 = time.perf_counter()
        sweep_result = orchestrator.sweep(design, [params])
        wall_s = time.perf_counter() - t0

        point = sweep_result.points[0]
        entry: dict[str, Any] = {
            "n_steps": n_steps,
            "t_sim": round(t_sim, 6),
            "t_ratio_vs_peak": round(t_ratio, 4),
            "steps_per_N": round(steps_per_N, 2),
            "wall_time_s": round(wall_s, 2),
            "success": point.success,
            "chi_max": point.chi_max,
            "s11_min_db": round(point.s11_min_db, 4),
            "peak_gain_dbi": round(point.peak_gain_dbi, 2),
            "fractional_bandwidth": round(point.fractional_bandwidth, 6),
            "radiation_efficiency": round(point.radiation_efficiency, 6),
            "vswr_min": round(point.vswr_min, 4) if point.vswr_min < 1e6 else None,
            "n_freq_bins": point.n_freq_bins,
            "f_resonance": point.f_resonance,
            "dft_norms": {k: round(v, 8) for k, v in point.dft_norms.items()},
        }
        results.append(entry)

        # Print summary line
        print(f"\n  ► S₁₁ = {entry['s11_min_db']:.2f} dB  |  "
              f"Gain = {entry['peak_gain_dbi']:.1f} dBi  |  "
              f"BW = {entry['fractional_bandwidth']*100:.2f}%  |  "
              f"η = {entry['radiation_efficiency']:.4f}  |  "
              f"χ = {entry['chi_max']}  |  "
              f"{wall_s:.1f}s")
        sys.stdout.flush()

        # Free GPU memory
        del orchestrator, sweep_result
        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.perf_counter() - t_global

    # ── Analysis ─────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  CONVERGENCE ANALYSIS")
    print(f"{'='*80}")
    print(f"  Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print()

    # Print table
    hdr = (f"  {'Steps':>6s}  {'t/t_peak':>8s}  {'s/N':>5s}  "
           f"{'S₁₁(dB)':>9s}  {'Gain(dBi)':>10s}  {'BW(%)':>7s}  "
           f"{'η':>7s}  {'Time(s)':>8s}")
    print(hdr)
    print(f"  {'─'*6}  {'─'*8}  {'─'*5}  {'─'*9}  {'─'*10}  {'─'*7}  "
          f"{'─'*7}  {'─'*8}")
    for r in results:
        print(f"  {r['n_steps']:6d}  "
              f"{r['t_ratio_vs_peak']:8.2f}  "
              f"{r['steps_per_N']:5.1f}  "
              f"{r['s11_min_db']:9.2f}  "
              f"{r['peak_gain_dbi']:10.1f}  "
              f"{r['fractional_bandwidth']*100:7.2f}  "
              f"{r['radiation_efficiency']:7.4f}  "
              f"{r['wall_time_s']:8.1f}")

    # Identify convergence point (where S₁₁ first drops below -10 dB)
    s11_threshold = -10.0
    converged_step = None
    for r in results:
        if r["s11_min_db"] < s11_threshold:
            converged_step = r
            break

    print()
    if converged_step:
        ratio = converged_step["steps_per_N"]
        n_conv = converged_step["n_steps"]
        print(f"  ✓ S₁₁ < {s11_threshold} dB first reached at "
              f"{n_conv} steps ({ratio:.1f} steps/N)")
        print()
        print(f"  Extrapolation to larger grids:")
        for test_bits in [9, 10, 12, 14]:
            test_N = 2 ** test_bits
            est_steps = int(math.ceil(ratio * test_N))
            print(f"    {test_N:>7,}³:  ~{est_steps:,} steps")
    else:
        # Find the best S₁₁ achieved
        best = min(results, key=lambda r: r["s11_min_db"])
        print(f"  ✗ S₁₁ never reached {s11_threshold} dB")
        print(f"    Best S₁₁ = {best['s11_min_db']:.2f} dB "
              f"at {best['n_steps']} steps")
        print(f"    May need more steps or design refinement")

    # Look for DFT energy convergence
    print()
    print("  DFT energy convergence:")
    for r in results:
        dft_total = sum(abs(v) for v in r["dft_norms"].values())
        print(f"    {r['n_steps']:6d} steps:  Σ|DFT| = {dft_total:.6e}")

    # ── Save results ─────────────────────────────────────────────────
    output_data = {
        "protocol": "TIME_STEP_CONVERGENCE_STUDY",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_bits": N_BITS,
            "grid": f"{N}³",
            "total_dofs": N ** 3,
            "max_rank": MAX_RANK,
            "dt_cfl": dt_cfl,
            "freq_center": freq_center,
            "freq_bandwidth": freq_bw,
            "tau": tau,
            "t_peak": t_peak,
            "steps_to_peak": steps_to_peak,
            "dipole_params": params,
            "step_counts_tested": STEP_COUNTS,
        },
        "results": results,
        "analysis": {
            "total_wall_time_s": round(total_time, 2),
            "s11_threshold_db": s11_threshold,
            "converged_at_steps": (
                converged_step["n_steps"] if converged_step else None
            ),
            "converged_steps_per_N": (
                converged_step["steps_per_N"] if converged_step else None
            ),
        },
    }

    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT}")
    print()


if __name__ == "__main__":
    main()
