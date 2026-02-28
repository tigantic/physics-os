"""PWA Engine CLI — run experiments from the command line.

Usage::

    pwa-engine                       # Run full 10-experiment suite
    pwa-engine --convention-only     # Run convention reduction test only
    pwa-engine --help                # Show usage

When installed via ``pip install -e pwa_engine/``, the ``pwa-engine``
command becomes available on PATH.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    """Entry point for the ``pwa-engine`` console script."""
    parser = argparse.ArgumentParser(
        prog="pwa-engine",
        description=(
            "PWA Compute Engine V3.0.0 — Eq. 5.48 (Badui 2020) "
            "with Gram-accelerated likelihood"
        ),
    )
    parser.add_argument(
        "--convention-only",
        action="store_true",
        help="Run only the convention reduction test (fast, ~0.5s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON metadata output",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"PWA Engine V3.0.0 | device={device}")
    print("=" * 60)

    from pwa_engine import convention_reduction_test

    if args.convention_only:
        t0 = time.perf_counter()
        result = convention_reduction_test(n_events=500, seed=42)
        elapsed = time.perf_counter() - t0
        print(f"Convention Reduction Test: {'PASS' if result['all_pass'] else 'FAIL'}")
        print(f"  Test 1 (full vs single ε):     {result['test_1_full_vs_single_eps']:.2e}")
        print(f"  Test 2 (model vs manual):       {result['test_2_model_vs_manual']:.2e}")
        print(f"  Test 3 (full ρ vs diagonal):    {result['test_3_full_rho_vs_diagonal']:.2e}")
        print(f"  Wall time: {elapsed:.2f}s")

        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, default=str))
            print(f"  Output: {args.output}")

        sys.exit(0 if result["all_pass"] else 1)

    # Full suite: delegate to experiments/run_pwa_engine.py main()
    try:
        from experiments.run_pwa_engine import main as run_full_suite
        run_full_suite()
    except ImportError:
        print(
            "ERROR: Full experiment suite requires the physics-os repo.\n"
            "  Use --convention-only for standalone validation, or run from\n"
            "  the repo root: python3 experiments/run_pwa_engine.py",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
