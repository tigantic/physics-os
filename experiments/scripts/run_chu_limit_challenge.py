#!/usr/bin/env python3
"""Run the Chu Limit Antenna Challenge.

Designs a 3D antenna within an electrically small sphere (ka=0.3)
at 1 GHz using QTT frequency-domain topology optimization.
Compares the achieved Q-factor against the theoretical Chu limit.

Usage
-----
    python3 run_chu_limit_challenge.py [--n-bits N] [--max-iter M]

The 3D scalar Helmholtz equation is solved entirely in
frequency domain using the QTT-compressed DMRG solver
(no time-stepping).

    H·E = -J,    H = ∇²_s + k²ε

where ∇²_s is the UPML stretched Laplacian, ε encodes the
conductor/air topology, and J is the antenna feed current.
The MPO operators live on 3n QTT sites (N³ grid, N = 2^n).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ontic.em.chu_limit import (
    C0,
    ChuOptConfig,
    ChuProblemConfig,
    chu_limit_q,
    optimize_chu_antenna,
    print_q_limits,
)


def main() -> int:
    """Run the Chu limit challenge."""
    parser = argparse.ArgumentParser(
        description="Chu Limit Antenna Challenge — 3D QTT Topology Optimization",
    )
    parser.add_argument(
        "--n-bits", type=int, default=6,
        help="QTT bits per dimension (N = 2^n per axis). Default: 6 (64³)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=60,
        help="Maximum optimization iterations. Default: 60",
    )
    parser.add_argument(
        "--max-rank", type=int, default=128,
        help="Maximum QTT bond dimension. Default: 128",
    )
    parser.add_argument(
        "--frequency-ghz", type=float, default=1.0,
        help="Operating frequency in GHz. Default: 1.0",
    )
    parser.add_argument(
        "--ka", type=float, default=0.3,
        help="Electrical size ka. Default: 0.3",
    )
    parser.add_argument(
        "--no-q-extract", action="store_true",
        help="Skip Q extraction frequency sweep",
    )
    parser.add_argument(
        "--output", type=str, default="chu_limit_result.json",
        help="Output JSON file. Default: chu_limit_result.json",
    )
    args = parser.parse_args()

    # ============================================================
    # Problem Configuration
    # ============================================================
    config = ChuProblemConfig(
        frequency_hz=args.frequency_ghz * 1e9,
        ka=args.ka,
        n_bits=args.n_bits,
        domain_wavelengths=0.0,  # auto-scale based on grid resolution
        pml_depth=8,
        max_rank=args.max_rank,
        n_sweeps=40,
        solver_tol=1e-4,
        damping=0.01,
        pec_penalty=1e8,
    )

    opt_config = ChuOptConfig(
        max_iterations=args.max_iter,
        learning_rate=0.2,
        beta_init=1.0,
        beta_max=32.0,
        beta_increase_every=15,
        beta_factor=2.0,
        eta=0.5,
        filter_radius=1,
        regularisation_weight=0.005,
        convergence_tol=1e-4,
        target_s11_db=-15.0,
    )

    # ============================================================
    # Print analytical limits
    # ============================================================
    print("\n" + "=" * 70)
    print("  CHU LIMIT ANTENNA CHALLENGE")
    print("  3D Frequency-Domain QTT Topology Optimization")
    print("  No time-stepping — pure Helmholtz inversion via DMRG")
    print("=" * 70)

    print(config.summary())
    limits = print_q_limits(config.ka)

    # ============================================================
    # Run optimization
    # ============================================================
    t_start = time.perf_counter()

    result = optimize_chu_antenna(
        config=config,
        opt_config=opt_config,
        verbose=True,
        extract_q=not args.no_q_extract,
    )

    t_total = time.perf_counter() - t_start

    # ============================================================
    # Report
    # ============================================================
    print(result.summary())

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  ka = {config.ka:.4f}")
    print(f"  Q_Chu = {config.q_chu:.2f}")
    if result.q_result is not None:
        Q_achieved = result.q_result.Q
        ratio = Q_achieved / config.q_chu
        print(f"  Q_achieved = {Q_achieved:.1f}")
        print(f"  Q / Q_Chu = {ratio:.2f}")
        if ratio < 1.5:
            print(f"  *** NEAR THEORETICAL LIMIT! ***")
        elif ratio < 3.0:
            print(f"  ** Good performance **")
    print(f"  |S₁₁| = {result.s11_db:.1f} dB")
    print(f"  Binarisation = {result.binarisation:.4f}")
    print(f"  Volume fraction = {result.volume_fraction:.1%}")
    print(f"  Total time = {t_total:.1f} s")
    print(f"{'='*70}")

    # ============================================================
    # Save results
    # ============================================================
    output_data = {
        "problem": {
            "frequency_hz": config.frequency_hz,
            "ka": config.ka,
            "wavelength_mm": config.wavelength * 1e3,
            "sphere_radius_mm": config.sphere_radius * 1e3,
            "grid": f"{config.N}³",
            "qtt_sites": 3 * config.n_bits,
            "max_rank": config.max_rank,
        },
        "analytical_limits": {
            k: float(v) for k, v in limits.items()
        },
        "optimization": {
            "iterations": result.n_iterations,
            "converged": result.converged,
            "s11_db": result.s11_db,
            "binarisation": result.binarisation,
            "volume_fraction": result.volume_fraction,
            "time_s": t_total,
        },
        "objective_history": result.objective_history,
        "s11_db_history": [
            20.0 * np.log10(max(abs(s), 1e-30))
            for s in result.s11_history
        ],
    }

    if result.q_result is not None:
        output_data["q_extraction"] = {
            "Q": result.q_result.Q,
            "Q_Chu": config.q_chu,
            "Q_ratio": result.q_result.Q / config.q_chu,
            "f_center_ghz": result.q_result.f_center / 1e9,
            "bandwidth_mhz": result.q_result.bandwidth_hz / 1e6,
            "s11_min_db": result.q_result.s11_min_db,
        }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=float)
    print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
