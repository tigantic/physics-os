#!/usr/bin/env python3
"""Rank Atlas Benchmark v1.0 — Standalone Runner.

Executes the Rank Atlas Benchmark using the standardized configuration,
produces results in the benchmark schema, and runs validation.

Usage:
    # Full benchmark (all 20 packs, all resolutions, 3 trials)
    python run_benchmark.py --output results.json

    # Single pack test
    python run_benchmark.py --packs II --n-bits 6 7 --output test.json

    # Custom config
    python run_benchmark.py --config configs/benchmark_v1.json --output results.json

    # Validate only (no measurement)
    python run_benchmark.py --validate-only results.json

Exit codes:
    0 — Benchmark completed (or validation passed)
    1 — Benchmark completed with validation warnings
    2 — Runtime error or validation failure
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "benchmark_v1.json"
DEFAULT_SVD_TOLERANCE = 1e-6
DEFAULT_MAX_RANK_CEILING = 2048
DEFAULT_N_BITS = [6, 7, 8, 9, 10]
DEFAULT_SEEDS = [42, 137, 2026]
DEFAULT_N_TRIALS = 3

ALL_PACK_IDS = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
]


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark execution
# ─────────────────────────────────────────────────────────────────────────────


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate benchmark configuration."""
    with open(config_path) as f:
        config = json.load(f)

    if "packs" not in config:
        raise ValueError(f"Config file {config_path} missing 'packs' key.")

    return config


def run_single_measurement(
    pack_id: str,
    pack_config: dict[str, Any],
    n_bits: int,
    complexity_value: float,
    seed: int,
    svd_tolerance: float,
    max_rank_ceiling: int,
) -> dict[str, Any] | None:
    """Run a single benchmark measurement.

    Delegates to the rank_atlas_campaign.py infrastructure for actual
    measurement execution.

    Returns:
        Measurement dict conforming to the benchmark schema, or None on failure.
    """
    try:
        # Import the campaign infrastructure
        from scripts.research.rank_atlas_campaign import (
            run_single_measurement as campaign_measure,
        )

        taxonomy_id = pack_config["taxonomy_id"]
        complexity_param_name = pack_config["complexity_param_name"]

        result = campaign_measure(
            pack_id=pack_id,
            taxonomy_id=taxonomy_id,
            complexity_param_name=complexity_param_name,
            complexity_param_value=complexity_value,
            n_bits=n_bits,
            svd_tolerance=svd_tolerance,
            max_rank_ceiling=max_rank_ceiling,
            seed=seed,
        )

        if result is not None:
            # Ensure seed is in the output
            if isinstance(result, dict):
                result["seed"] = seed
            elif hasattr(result, "__dict__"):
                result_dict = result.__dict__
                result_dict["seed"] = seed
                return result_dict

        return result if isinstance(result, dict) else None

    except ImportError:
        # Fallback: try direct import from scripts path
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))
        try:
            from rank_atlas_campaign import (  # type: ignore[import-not-found]
                run_single_measurement as campaign_measure,
            )

            taxonomy_id = pack_config["taxonomy_id"]
            complexity_param_name = pack_config["complexity_param_name"]

            result = campaign_measure(
                pack_id=pack_id,
                taxonomy_id=taxonomy_id,
                complexity_param_name=complexity_param_name,
                complexity_param_value=complexity_value,
                n_bits=n_bits,
                svd_tolerance=svd_tolerance,
                max_rank_ceiling=max_rank_ceiling,
                seed=seed,
            )

            if result is not None:
                if isinstance(result, dict):
                    result["seed"] = seed
                elif hasattr(result, "__dict__"):
                    result_dict = result.__dict__
                    result_dict["seed"] = seed
                    return result_dict

            return result if isinstance(result, dict) else None

        except ImportError as exc:
            print(
                f"  ERROR: Cannot import campaign infrastructure: {exc}",
                file=sys.stderr,
            )
            return None

    except Exception as exc:
        print(
            f"  ERROR: Measurement failed for Pack {pack_id}, "
            f"n_bits={n_bits}, complexity={complexity_value}, "
            f"seed={seed}: {exc}",
            file=sys.stderr,
        )
        return None


def run_benchmark(
    pack_ids: list[str],
    n_bits_list: list[int],
    n_trials: int,
    config: dict[str, Any],
    svd_tolerance: float,
    max_rank_ceiling: int,
) -> list[dict[str, Any]]:
    """Run the full benchmark suite.

    Args:
        pack_ids: List of pack IDs to measure.
        n_bits_list: List of resolutions to sweep.
        n_trials: Number of trials per configuration.
        config: Benchmark configuration dict.
        svd_tolerance: SVD truncation tolerance.
        max_rank_ceiling: Maximum bond dimension.

    Returns:
        List of measurement dicts.
    """
    seeds = DEFAULT_SEEDS[:n_trials]
    packs_config = config["packs"]
    results: list[dict[str, Any]] = []

    total_packs = len(pack_ids)

    for pack_idx, pack_id in enumerate(pack_ids, 1):
        if pack_id not in packs_config:
            print(f"WARNING: Pack {pack_id} not in config, skipping.")
            continue

        pack_cfg = packs_config[pack_id]
        complexity_values = pack_cfg.get("complexity_values", [1.0])

        total_configs = len(complexity_values) * len(n_bits_list) * len(seeds)
        print(
            f"\n[{pack_idx}/{total_packs}] Pack {pack_id} "
            f"({pack_cfg.get('domain_name', '?')}) — "
            f"{total_configs} measurements"
        )

        config_count = 0
        for cpv in complexity_values:
            for n_bits in n_bits_list:
                for seed in seeds:
                    config_count += 1
                    t0 = time.monotonic()

                    measurement = run_single_measurement(
                        pack_id=pack_id,
                        pack_config=pack_cfg,
                        n_bits=n_bits,
                        complexity_value=cpv,
                        seed=seed,
                        svd_tolerance=svd_tolerance,
                        max_rank_ceiling=max_rank_ceiling,
                    )

                    elapsed = time.monotonic() - t0

                    if measurement is not None:
                        results.append(measurement)
                        chi = measurement.get("max_rank", "?")
                        print(
                            f"  [{config_count}/{total_configs}] "
                            f"n_bits={n_bits}, ξ={cpv:.4g}, "
                            f"seed={seed} → χ_max={chi} "
                            f"({elapsed:.1f}s)"
                        )
                    else:
                        print(
                            f"  [{config_count}/{total_configs}] "
                            f"n_bits={n_bits}, ξ={cpv:.4g}, "
                            f"seed={seed} → FAILED "
                            f"({elapsed:.1f}s)"
                        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank Atlas Benchmark v1.0 — Runner"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rank_atlas_benchmark_results.json",
        help="Output JSON file for results (default: rank_atlas_benchmark_results.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"Benchmark configuration file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--packs",
        nargs="*",
        default=None,
        help="Pack IDs to run (default: all 20). Use ALL for all packs.",
    )
    parser.add_argument(
        "--n-bits",
        nargs="*",
        type=int,
        default=None,
        help=f"Resolution values (default: {DEFAULT_N_BITS})",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Number of trials per config (default: {DEFAULT_N_TRIALS})",
    )
    parser.add_argument(
        "--svd-tolerance",
        type=float,
        default=DEFAULT_SVD_TOLERANCE,
        help=f"SVD truncation tolerance (default: {DEFAULT_SVD_TOLERANCE})",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=DEFAULT_MAX_RANK_CEILING,
        help=f"Maximum rank ceiling (default: {DEFAULT_MAX_RANK_CEILING})",
    )
    parser.add_argument(
        "--validate-only",
        type=str,
        default=None,
        help="Validate an existing results file without running measurements",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-measurement validation",
    )
    args = parser.parse_args()

    # Validate-only mode
    if args.validate_only:
        from benchmarks.rank_atlas.validate import validate_results

        report = validate_results(args.validate_only)
        return 0 if report.passed else 1

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 2

    config = load_config(config_path)

    # Determine packs
    pack_ids = args.packs
    if pack_ids is None or (len(pack_ids) == 1 and pack_ids[0].upper() == "ALL"):
        pack_ids = ALL_PACK_IDS
    pack_ids = [p.upper() for p in pack_ids]

    # Determine resolutions
    n_bits_list = args.n_bits if args.n_bits else DEFAULT_N_BITS

    # Print banner
    print("=" * 60)
    print("Rank Atlas Benchmark v1.0")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Packs: {', '.join(pack_ids)} ({len(pack_ids)} total)")
    print(f"Resolutions: n_bits = {n_bits_list}")
    print(f"Trials: {args.n_trials}")
    print(f"SVD tolerance: {args.svd_tolerance}")
    print(f"Max rank ceiling: {args.max_rank}")
    print(f"Output: {args.output}")
    print("=" * 60)

    t_start = time.monotonic()

    # Run benchmark
    results = run_benchmark(
        pack_ids=pack_ids,
        n_bits_list=n_bits_list,
        n_trials=args.n_trials,
        config=config,
        svd_tolerance=args.svd_tolerance,
        max_rank_ceiling=args.max_rank,
    )

    elapsed_total = time.monotonic() - t_start

    # Write results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Benchmark complete: {len(results)} measurements in {elapsed_total:.1f}s")
    print(f"Results written to: {output_path}")

    # Run validation
    if not args.skip_validation and results:
        print(f"\nRunning validation...")
        try:
            from benchmarks.rank_atlas.validate import validate_results

            report = validate_results(str(output_path))
            print(f"Verdict: {report.benchmark_verdict}")

            if report.errors:
                print(f"Schema errors: {len(report.errors)}")
                return 1
        except ImportError:
            # Try relative import
            validate_path = Path(__file__).parent / "validate.py"
            if validate_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "validate", str(validate_path)
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    report = mod.validate_results(str(output_path))
                    print(f"Verdict: {report.benchmark_verdict}")
                    if report.errors:
                        return 1

    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
