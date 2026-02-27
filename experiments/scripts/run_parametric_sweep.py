#!/usr/bin/env python3
"""QTT Physics VM — Parametric antenna sweep & Pareto analysis demo.

End-to-end demonstration of the automated antenna IP discovery pipeline:

    1.  Select antenna family (dipole — fastest for demo)
    2.  Sample design space (6 random candidates)
    3.  Run GPU-accelerated sweep at 512³ (n_bits=9, 200 steps)
    4.  Pareto-rank across gain / bandwidth / efficiency
    5.  Score candidates on 6 IP dimensions
    6.  Triage: FILE_NOW / HOLD / DISCARD
    7.  Save results to JSON

Usage:
    python run_parametric_sweep.py [--n-bits 9] [--n-steps 200] [--n-candidates 6] [--seed 42]

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parametric antenna sweep with Pareto optimisation"
    )
    parser.add_argument("--n-bits", type=int, default=9,
                        help="Grid resolution bits (9 → 512³, 10 → 1024³)")
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Time steps per candidate")
    parser.add_argument("--n-candidates", type=int, default=6,
                        help="Number of random candidates")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    parser.add_argument("--family", type=str, default="dipole",
                        choices=["dipole", "patch", "e_shaped", "u_slot"],
                        help="Antenna family to sweep")
    parser.add_argument("--output", type=str, default="sweep_pareto_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    # ── Imports ──────────────────────────────────────────────────────
    print("Loading QTT antenna pipeline...")
    from tensornet.engine.vm.antenna import (
        DipoleAntennaDesign,
        PatchAntennaDesign,
        EShapedPatchDesign,
        USlotsDesign,
        MaterialLibrary,
        SweepOrchestrator,
        ParetoOptimizer,
    )

    # ── 1. Select antenna family ─────────────────────────────────────
    mat_lib = MaterialLibrary()

    if args.family == "dipole":
        design = DipoleAntennaDesign(freq_center=1.0, freq_bandwidth=0.5)
    elif args.family == "patch":
        substrate = mat_lib.get("FR-4")
        design = PatchAntennaDesign(
            freq_center=1.0, freq_bandwidth=0.5, substrate=substrate
        )
    elif args.family == "e_shaped":
        substrate = mat_lib.get("Rogers RO4003C")
        design = EShapedPatchDesign(
            freq_center=1.0, freq_bandwidth=0.5, substrate=substrate
        )
    elif args.family == "u_slot":
        substrate = mat_lib.get("Rogers RT5880")
        design = USlotsDesign(
            freq_center=1.0, freq_bandwidth=0.5, substrate=substrate
        )
    else:
        print(f"Unknown family: {args.family}")
        sys.exit(1)

    space = design.design_space
    print(f"\nAntenna family: {design.family_name}")
    print(f"Design space: {space.n_dims} dimensions")
    for v in space.variables:
        print(f"  {v.name:20s}  [{v.low:.4f}, {v.high:.4f}]  "
              f"default={v.default:.4f}  ({v.description})")

    # ── 2. Sample parameter space ────────────────────────────────────
    param_list = space.random_points(n=args.n_candidates, seed=args.seed)
    print(f"\nSampled {len(param_list)} valid candidates "
          f"(seed={args.seed})")

    for i, params in enumerate(param_list):
        param_str = ", ".join(f"{k}={v:.4f}" for k, v in params.items())
        print(f"  [{i}] {param_str}")

    # ── 3. Run parametric sweep ──────────────────────────────────────
    N = 2 ** args.n_bits
    print(f"\nLaunching sweep: {N}³ grid, {args.n_steps} steps, "
          f"{len(param_list)} candidates")

    orchestrator = SweepOrchestrator(
        n_bits=args.n_bits,
        n_steps=args.n_steps,
        max_rank=48,
        extract_far_field=True,
        verbose=True,
    )

    sweep_start = time.perf_counter()
    sweep_result = orchestrator.sweep(design, param_list)
    sweep_time = time.perf_counter() - sweep_start

    print(f"\n{'─'*72}")
    print(f"SWEEP COMPLETE: {sweep_result.n_successful}/{sweep_result.n_points} "
          f"succeeded in {sweep_time:.1f}s")

    # Print per-candidate summary
    for pt in sweep_result.points:
        status = "✓" if pt.success else "✗"
        print(f"  {status} {pt.candidate_id[:12]}  "
              f"S₁₁={pt.s11_min_db:+.1f} dB  "
              f"Gain={pt.peak_gain_dbi:.1f} dBi  "
              f"BW={pt.fractional_bandwidth:.3f}  "
              f"η={pt.radiation_efficiency:.2f}  "
              f"χ_max={pt.chi_max}  "
              f"t={pt.wall_time_s:.1f}s")

    # ── 4. Pareto analysis ───────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("PARETO ANALYSIS & CLAIM SCORING")

    # Define target bands for market relevance scoring
    target_bands = [
        (2.4e9, 2.5e9),    # Wi-Fi 2.4 GHz
        (3.5e9, 3.7e9),    # 5G n78
        (5.15e9, 5.85e9),  # Wi-Fi 5 GHz
        (24.25e9, 27.5e9), # 5G mmWave n258
        (0.5, 1.5),        # Normalised band (for unit-cell sims)
    ]

    optimizer = ParetoOptimizer(
        objectives=[
            "peak_gain_dbi",
            "fractional_bandwidth",
            "radiation_efficiency",
        ],
        maximize=[True, True, True],
        target_bands=target_bands,
    )

    pareto_result = optimizer.optimize(sweep_result.successful_points())

    # ── 5. Print results ─────────────────────────────────────────────
    summary = pareto_result.summary()
    print(f"\nAnalysed: {summary['n_candidates']} candidates  |  "
          f"Fronts: {summary['n_fronts']}")
    print(f"Triage:  FILE_NOW={summary['n_file_now']}  "
          f"HOLD={summary['n_hold']}  "
          f"DISCARD={summary['n_discard']}")

    print(f"\n{'─'*72}")
    print("RANKED CANDIDATES (by composite score)")
    print(f"{'─'*72}")

    for rank, sc in enumerate(pareto_result.candidates, 1):
        print(f"\n  #{rank}  {sc.point.candidate_id[:16]}")
        print(f"      Composite: {sc.composite_score:.3f}  "
              f"  Front: {sc.pareto_front}  "
              f"  Crowding: {sc.crowding_distance:.2f}")
        print(f"      RF={sc.rf_performance:.2f}  "
              f"Mfg={sc.manufacturability:.2f}  "
              f"Nov={sc.novelty_proxy:.2f}  "
              f"Brd={sc.claim_breadth:.2f}  "
              f"Mkt={sc.market_relevance:.2f}  "
              f"Ver={sc.verification_quality:.2f}")
        print(f"      Triage: {sc.triage}  — {sc.triage_reason}")
        print(f"      Metrics: S₁₁={sc.point.s11_min_db:+.1f} dB  "
              f"Gain={sc.point.peak_gain_dbi:.1f} dBi  "
              f"BW={sc.point.fractional_bandwidth:.3f}")

    # ── 6. FILE_NOW candidates ───────────────────────────────────────
    file_now = pareto_result.file_now()
    if file_now:
        print(f"\n{'='*72}")
        print(f"  IP ACTION: {len(file_now)} candidate(s) ready for filing")
        print(f"{'='*72}")
        for sc in file_now:
            print(f"  → {sc.point.candidate_id}")
            print(f"    Score: {sc.composite_score:.3f}  "
                  f"Novelty: {sc.novelty_proxy:.2f}")
            params_str = ", ".join(
                f"{k}={v:.4f}" for k, v in sc.point.params.items()
            )
            print(f"    Params: {params_str}")
    else:
        print(f"\n  No FILE_NOW candidates — consider expanding the "
              "design space or increasing N")

    # ── 7. Save results to JSON ──────────────────────────────────────
    output_data = {
        "config": {
            "family": design.family_name,
            "n_bits": args.n_bits,
            "n_steps": args.n_steps,
            "n_candidates": len(param_list),
            "seed": args.seed,
            "grid_size": f"{N}³",
            "total_sweep_time_s": sweep_time,
        },
        "sweep_summary": {
            "n_points": sweep_result.n_points,
            "n_successful": sweep_result.n_successful,
            "n_failed": sweep_result.n_failed,
        },
        "pareto_summary": summary,
        "candidates": [sc.to_dict() for sc in pareto_result.candidates],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  PIPELINE SUMMARY")
    print(f"  Family:     {design.family_name}")
    print(f"  Grid:       {N}³ ({N**3:,.0f} cells)")
    print(f"  Candidates: {sweep_result.n_successful}/{sweep_result.n_points} "
          f"succeeded")
    print(f"  Best score: {summary['top_score']:.3f}")
    print(f"  FILE_NOW:   {summary['n_file_now']}")
    print(f"  Total time: {sweep_time:.1f}s")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
