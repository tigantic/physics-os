#!/usr/bin/env python3
"""QTT Physics VM — Exascale Multi-Family Antenna Invention Sweep.

Runs ALL 4 antenna families at 16,384³ (4.4 trillion DOFs) through the
parametric sweep + Pareto IP scoring pipeline.  This is the real deal:
no 512³ toy runs, no 50-step shortcuts.

Families:
    1. Dipole          — baseline reference (3 candidates)
    2. Rectangular Patch — standard planar (3 candidates)
    3. E-Shaped Patch   — wideband slots (3 candidates)
    4. U-Slot Patch     — multi-band (3 candidates)

Total: 12 candidates × 16,384³ × 200 steps = 4.4T DOFs per candidate,
       combined Pareto across families for cross-topology IP discovery.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone


def main() -> None:
    t_global = time.perf_counter()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── Configuration ────────────────────────────────────────────────
    N_BITS = 14            # 16,384³ = 4.4 trillion DOFs
    N_STEPS = 200          # full EM pulse propagation
    N_CANDIDATES = 3       # per family
    SEED = 42
    MAX_RANK = 48
    OUTPUT = f"exascale_invention_results_{ts}.json"

    N = 2 ** N_BITS
    DOFS = N ** 3

    print("=" * 80)
    print("  HYPERTENSOR-VM  —  EXASCALE ANTENNA INVENTION SWEEP")
    print("=" * 80)
    print(f"  Grid:       {N:,}³  ({DOFS:,.0f} DOFs = {DOFS/1e12:.1f}T)")
    print(f"  Steps:      {N_STEPS}")
    print(f"  Families:   4  (dipole, patch, e-shaped, u-slot)")
    print(f"  Candidates: {N_CANDIDATES} per family = {4*N_CANDIDATES} total")
    print(f"  Max rank:   {MAX_RANK}")
    print(f"  Seed:       {SEED}")
    print(f"  Output:     {OUTPUT}")
    print("=" * 80)
    print()

    # ── Imports ──────────────────────────────────────────────────────
    print("Loading QTT antenna pipeline...", flush=True)
    from ontic.engine.vm.antenna import (
        DipoleAntennaDesign,
        PatchAntennaDesign,
        EShapedPatchDesign,
        USlotsDesign,
        MaterialLibrary,
        SweepOrchestrator,
        ParetoOptimizer,
    )

    mat_lib = MaterialLibrary()

    # ── Define all 4 families ────────────────────────────────────────
    families = {
        "dipole": DipoleAntennaDesign(
            freq_center=1.0, freq_bandwidth=0.5,
        ),
        "patch": PatchAntennaDesign(
            freq_center=1.0, freq_bandwidth=0.5,
            substrate=mat_lib.get("FR-4"),
        ),
        "e_shaped": EShapedPatchDesign(
            freq_center=1.0, freq_bandwidth=0.5,
            substrate=mat_lib.get("Rogers RO4003C"),
        ),
        "u_slot": USlotsDesign(
            freq_center=1.0, freq_bandwidth=0.5,
            substrate=mat_lib.get("Rogers RT5880"),
        ),
    }

    # ── Create orchestrator at 16,384³ ───────────────────────────────
    orchestrator = SweepOrchestrator(
        n_bits=N_BITS,
        n_steps=N_STEPS,
        max_rank=MAX_RANK,
        extract_far_field=True,
        n_surface_samples=8,
        n_theta=37,
        n_phi=18,
        verbose=True,
    )

    # ── Sweep all families ───────────────────────────────────────────
    all_points = []
    family_results = {}

    for fname, design in families.items():
        print(f"\n{'#'*80}")
        print(f"  FAMILY: {design.family_name}  "
              f"({N_CANDIDATES} candidates at {N:,}³)")
        print(f"{'#'*80}\n")
        sys.stdout.flush()

        space = design.design_space
        print(f"  Design space: {space.n_dims}D")
        for v in space.variables:
            print(f"    {v.name:20s}  [{v.low:.4f}, {v.high:.4f}]  "
                  f"default={v.default:.4f}")
        sys.stdout.flush()

        # Sample candidates
        param_list = space.random_points(n=N_CANDIDATES, seed=SEED)
        print(f"\n  Sampled {len(param_list)} candidates (seed={SEED}):")
        for i, params in enumerate(param_list):
            pstr = ", ".join(f"{k}={v:.4f}" for k, v in params.items())
            print(f"    [{i}] {pstr}")
        sys.stdout.flush()

        # Execute sweep
        sweep_result = orchestrator.sweep(design, param_list)
        family_results[fname] = sweep_result

        # Collect successful points
        successes = sweep_result.successful_points()
        all_points.extend(successes)

        print(f"\n  {fname} complete: "
              f"{sweep_result.n_successful}/{sweep_result.n_points} "
              f"succeeded, {sweep_result.total_wall_time_s:.1f}s total")
        sys.stdout.flush()

        gc.collect()

    # ── Combined Pareto across all families ──────────────────────────
    print(f"\n{'='*80}")
    print(f"  CROSS-FAMILY PARETO ANALYSIS — {len(all_points)} candidates")
    print(f"{'='*80}\n")
    sys.stdout.flush()

    target_bands = [
        (2.4e9, 2.5e9),    # Wi-Fi 2.4 GHz
        (3.5e9, 3.7e9),    # 5G n78
        (5.15e9, 5.85e9),  # Wi-Fi 5 GHz
        (24.25e9, 27.5e9), # 5G mmWave n258
        (0.5, 1.5),        # Normalised band (unit-cell sims)
    ]

    optimizer = ParetoOptimizer(
        objectives=["peak_gain_dbi", "fractional_bandwidth",
                     "radiation_efficiency"],
        maximize=[True, True, True],
        target_bands=target_bands,
    )

    if len(all_points) == 0:
        print("  ERROR: No successful candidates. Cannot run Pareto.")
        sys.exit(1)

    pareto_result = optimizer.optimize(all_points)

    # ── Print ranked results ─────────────────────────────────────────
    summary = pareto_result.summary()
    print(f"  Analysed: {summary['n_candidates']} candidates  |  "
          f"Fronts: {summary['n_fronts']}")
    print(f"  Triage:   FILE_NOW={summary['n_file_now']}  "
          f"HOLD={summary['n_hold']}  "
          f"DISCARD={summary['n_discard']}")

    print(f"\n{'─'*80}")
    print("  RANKED CANDIDATES (by composite score)")
    print(f"{'─'*80}")

    for rank, sc in enumerate(pareto_result.candidates, 1):
        print(f"\n  #{rank}  {sc.point.candidate_id}  "
              f"[{sc.point.family}]")
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
        print(f"      Family: {sc.point.family}  |  "
              f"Grid: {sc.point.grid_size}  "
              f"({sc.point.n_bits}-bit QTT)")
        print(f"      S₁₁={sc.point.s11_min_db:+.1f} dB  "
              f"Gain={sc.point.peak_gain_dbi:.1f} dBi  "
              f"BW={sc.point.fractional_bandwidth:.3f}  "
              f"η={sc.point.radiation_efficiency:.2f}")
        pstr = ", ".join(
            f"{k}={v:.4f}" for k, v in sc.point.params.items()
        )
        print(f"      Params: {pstr}")

    # ── FILE_NOW candidates ──────────────────────────────────────────
    file_now = pareto_result.file_now()
    if file_now:
        print(f"\n{'='*80}")
        print(f"  ★ IP ACTION: {len(file_now)} CANDIDATE(S) → FILE_NOW")
        print(f"{'='*80}")
        for sc in file_now:
            print(f"\n  → {sc.point.candidate_id}  [{sc.point.family}]")
            print(f"    Composite: {sc.composite_score:.3f}  "
                  f"Novelty: {sc.novelty_proxy:.2f}")
            pstr = ", ".join(
                f"{k}={v:.4f}" for k, v in sc.point.params.items()
            )
            print(f"    Params: {pstr}")
            print(f"    S₁₁={sc.point.s11_min_db:+.1f} dB  "
                  f"Gain={sc.point.peak_gain_dbi:.1f} dBi  "
                  f"BW={sc.point.fractional_bandwidth:.3f}")
    else:
        print(f"\n  No FILE_NOW candidates at current thresholds "
              "(composite ≥ 0.75, novelty ≥ 0.6).")
        print("  Top candidates are HOLD — review for tweak potential.")

    # ── Save results ─────────────────────────────────────────────────
    total_time = time.perf_counter() - t_global

    output_data = {
        "attestation": {
            "protocol": "EXASCALE_ANTENNA_INVENTION_SWEEP",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project": "HyperTensor-VM",
        },
        "config": {
            "n_bits": N_BITS,
            "grid": f"{N:,}³",
            "total_dofs": DOFS,
            "dofs_human": f"{DOFS/1e12:.1f}T",
            "n_steps": N_STEPS,
            "max_rank": MAX_RANK,
            "seed": SEED,
            "families": list(families.keys()),
            "n_candidates_per_family": N_CANDIDATES,
            "total_candidates": 4 * N_CANDIDATES,
            "total_wall_time_s": total_time,
        },
        "per_family_summary": {},
        "pareto_summary": summary,
        "ranked_candidates": [
            sc.to_dict() for sc in pareto_result.candidates
        ],
    }

    for fname, sr in family_results.items():
        fsum = sr.summary()
        output_data["per_family_summary"][fname] = fsum

    # Compute attestation hash
    payload = json.dumps(output_data, sort_keys=True, default=str)
    output_data["self_hash"] = hashlib.sha256(
        payload.encode()
    ).hexdigest()

    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT}")

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  EXASCALE INVENTION SWEEP — FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Grid:         {N:,}³  ({DOFS/1e12:.1f}T DOFs)")
    print(f"  Families:     {len(families)}")
    print(f"  Candidates:   {len(all_points)}"
          f"/{4*N_CANDIDATES} succeeded")
    print(f"  Top score:    {summary['top_score']:.3f}")
    print(f"  FILE_NOW:     {summary['n_file_now']}")
    print(f"  HOLD:         {summary['n_hold']}")
    print(f"  DISCARD:      {summary['n_discard']}")
    print(f"  Total time:   {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*80}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
