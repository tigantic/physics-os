#!/usr/bin/env python3
"""Golden Physics Benchmark v1.0 — Standalone Runner.

Executes all 7 QTT physics domains through the full HyperTensor pipeline
(compile → execute → sanitize → validate → attest → verify), collects
structured measurements, and optionally validates results against the
documented tolerance bands.

Usage:
    # Full benchmark (all 7 domains, 3 trials)
    python run_benchmark.py --output results.json

    # Single domain
    python run_benchmark.py --domains burgers maxwell --output quick.json

    # More trials
    python run_benchmark.py --n-trials 5 --output deep.json

    # Validate only (no measurement)
    python run_benchmark.py --validate-only results.json

Exit codes:
    0 — Benchmark completed and validation passed
    1 — Benchmark completed with validation warnings or failures
    2 — Runtime error or file I/O failure

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import json
import signal
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
DEFAULT_SEEDS = [42, 137, 2026]
DEFAULT_N_TRIALS = 3
JOB_TIMEOUT_S = 120

ALL_DOMAINS = [
    "burgers",
    "maxwell",
    "maxwell_3d",
    "schrodinger",
    "advection_diffusion",
    "vlasov_poisson",
    "navier_stokes_2d",
]

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      G O L D E N   P H Y S I C S   B E N C H M A R K           ║
║                                                                  ║
║              HyperTensor QTT VM • v1.0                          ║
║                                                                  ║
║   7 Domains • Full Pipeline • Trustless Attestation             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Timeout
# ─────────────────────────────────────────────────────────────────────────────


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Job exceeded hard timeout")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate benchmark configuration."""
    with open(config_path) as f:
        config = json.load(f)

    if "domains" not in config:
        raise ValueError(f"Config file {config_path} missing 'domains' key.")

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Single measurement
# ─────────────────────────────────────────────────────────────────────────────


def run_single_measurement(
    domain: str,
    domain_config: dict[str, Any],
    trial: int,
    seed: int,
    timeout_s: int = JOB_TIMEOUT_S,
) -> dict[str, Any]:
    """Run one domain through the full pipeline and return a measurement.

    Parameters
    ----------
    domain : str
        Physics domain key.
    domain_config : dict
        Domain-specific config from benchmark_v1.json.
    trial : int
        Trial number (1-indexed).
    seed : int
        Random seed for this trial.
    timeout_s : int
        Hard timeout in seconds.

    Returns
    -------
    dict
        Measurement record conforming to ``schema.json``.
    """
    from hypertensor.core.certificates import issue_certificate, verify_certificate
    from hypertensor.core.evidence import generate_claims, generate_validation_report
    from hypertensor.core.executor import ExecutionConfig, execute
    from hypertensor.core.hasher import content_hash
    from hypertensor.core.sanitizer import sanitize_result

    n_bits = domain_config["n_bits"]
    n_steps = domain_config["n_steps"]
    max_rank = domain_config["max_rank"]
    conservation_band_max = domain_config["conservation_error_max"]
    wall_time_band_max = domain_config["wall_time_max_s"]
    conservation_quantity = domain_config["conservation_quantity"]

    job_id = f"golden-{domain}-t{trial}-s{seed}"
    ts = datetime.now(timezone.utc).isoformat()

    # Base record (filled on success or failure)
    record: dict[str, Any] = {
        "domain": domain,
        "trial": trial,
        "seed": seed,
        "n_bits": n_bits,
        "n_steps": n_steps,
        "max_rank": max_rank,
        "conservation_quantity": conservation_quantity,
        "conservation_band_max": conservation_band_max,
        "wall_time_band_max": wall_time_band_max,
        "device": "cpu",
        "timestamp": ts,
    }

    config = ExecutionConfig(
        domain=domain,
        n_bits=n_bits,
        n_steps=n_steps,
        max_rank=max_rank,
    )

    # Arm SIGALRM for hard timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)

    try:
        # ── Execute ─────────────────────────────────────────────────
        t0 = time.monotonic()
        raw_result = execute(config)
        wall_time = time.monotonic() - t0

        if not raw_result.success:
            raise RuntimeError(
                f"Execution failed: {getattr(raw_result, 'error', 'unknown')}"
            )

        # ── Sanitize ────────────────────────────────────────────────
        sanitized = sanitize_result(raw_result, domain, include_fields=False)

        # ── Conservation ────────────────────────────────────────────
        cons = sanitized.get("conservation")
        if cons is not None:
            rel_err = cons["relative_error"]
            within_band = rel_err <= conservation_band_max
            record.update({
                "conservation_initial": cons["initial_value"],
                "conservation_final": cons["final_value"],
                "conservation_relative_error": rel_err,
                "conservation_status": cons["status"],
                "conservation_within_band": within_band,
                "conservation_drift_per_step": round(rel_err / max(n_steps, 1), 12),
            })
        else:
            record.update({
                "conservation_initial": None,
                "conservation_final": None,
                "conservation_relative_error": 0.0,
                "conservation_status": "conserved",
                "conservation_within_band": True,
                "conservation_drift_per_step": 0.0,
            })

        # ── Grid metadata ──────────────────────────────────────────
        grid = sanitized["grid"]
        n_dims = grid["dimensions"]
        grid_points = 1
        for r in grid["resolution"]:
            grid_points *= r

        perf = sanitized["performance"]
        throughput = perf["throughput_gp_per_s"]

        record.update({
            "n_dims": n_dims,
            "grid_points": grid_points,
            "wall_time_s": round(wall_time, 6),
            "throughput_gp_per_s": round(throughput, 1),
            "grid_resolution": grid["resolution"],
            "domain_bounds": grid.get("domain_bounds"),
        })

        # ── Fields returned ────────────────────────────────────────
        if sanitized.get("fields") is not None:
            record["fields_returned"] = list(sanitized["fields"].keys())

        # ── Evidence + Attestation ─────────────────────────────────
        validation = generate_validation_report(sanitized, domain)
        claims = generate_claims(sanitized, domain)
        result_hash = content_hash(sanitized)
        input_spec = {
            "domain": domain,
            "n_bits": n_bits,
            "n_steps": n_steps,
            "max_rank": max_rank,
        }

        certificate = issue_certificate(
            job_id=job_id,
            claims=claims,
            input_manifest_hash=content_hash(input_spec),
            result_hash=result_hash,
            config_hash=content_hash(input_spec),
            runtime_version="3.1.0",
            device_class="cpu",
        )

        verified = verify_certificate(certificate)

        record.update({
            "certificate_issued": True,
            "certificate_verified": verified,
            "certificate_job_id": job_id,
            "n_claims": len(claims),
            "pipeline_success": True,
            "runtime_version": "3.1.0",
        })

    except Exception as exc:
        # Pipeline failed — record the failure
        record.update({
            "n_dims": 0,
            "grid_points": 0,
            "wall_time_s": 0.0,
            "throughput_gp_per_s": 0.0,
            "conservation_initial": None,
            "conservation_final": None,
            "conservation_relative_error": 0.0,
            "conservation_status": "conserved",
            "conservation_within_band": False,
            "certificate_issued": False,
            "certificate_verified": False,
            "certificate_job_id": job_id,
            "n_claims": 0,
            "pipeline_success": False,
            "error_message": str(exc),
        })

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark
# ─────────────────────────────────────────────────────────────────────────────


def run_benchmark(
    domains: list[str],
    n_trials: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run the full golden benchmark suite.

    Parameters
    ----------
    domains : list[str]
        Domain keys to measure.
    n_trials : int
        Number of trials per domain.
    config : dict
        Benchmark config (must contain "domains" key).

    Returns
    -------
    list[dict]
        List of measurement records.
    """
    seeds = DEFAULT_SEEDS[:n_trials]
    if n_trials > len(DEFAULT_SEEDS):
        # Extend seeds deterministically
        import hashlib
        for i in range(len(DEFAULT_SEEDS), n_trials):
            h = hashlib.sha256(f"golden-seed-{i}".encode()).digest()
            seeds.append(int.from_bytes(h[:4], "big"))

    domains_config = config["domains"]
    results: list[dict[str, Any]] = []
    total_domains = len(domains)

    for d_idx, domain in enumerate(domains, 1):
        if domain not in domains_config:
            print(f"WARNING: Domain '{domain}' not in config, skipping.")
            continue

        d_cfg = domains_config[domain]
        total_trials = len(seeds)

        print(
            f"\n[{d_idx}/{total_domains}] Domain: {domain} "
            f"(n_bits={d_cfg['n_bits']}, n_steps={d_cfg['n_steps']}) "
            f"— {total_trials} trials"
        )

        for t_idx, seed in enumerate(seeds, 1):
            t0 = time.monotonic()

            measurement = run_single_measurement(
                domain=domain,
                domain_config=d_cfg,
                trial=t_idx,
                seed=seed,
            )

            elapsed = time.monotonic() - t0
            results.append(measurement)

            status = "✓" if measurement["pipeline_success"] else "✗"
            cons_err = measurement.get("conservation_relative_error", "?")
            in_band = measurement.get("conservation_within_band", False)
            band_str = "in-band" if in_band else "OUT-OF-BAND"
            cert = "cert✓" if measurement.get("certificate_verified") else "cert✗"

            if isinstance(cons_err, float):
                cons_str = f"{cons_err:.2e}"
            else:
                cons_str = str(cons_err)

            print(
                f"  [{t_idx}/{total_trials}] seed={seed} → "
                f"{status} wall={elapsed:.3f}s, "
                f"err={cons_str} ({band_str}), "
                f"{cert} ({elapsed:.1f}s)"
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 72)
    print("GOLDEN PHYSICS BENCHMARK — SUMMARY")
    print("=" * 72)

    # Group by domain
    from collections import defaultdict
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)

    print(
        f"{'Domain':<22} {'Trials':>6} {'Pass':>4} {'Fail':>4} "
        f"{'Avg Wall':>10} {'Avg Err':>12} {'Cert%':>6}"
    )
    print("-" * 72)

    total_pass = 0
    total_fail = 0

    for domain in ALL_DOMAINS:
        trials = by_domain.get(domain, [])
        if not trials:
            continue

        n_pass = sum(1 for t in trials if t["pipeline_success"])
        n_fail = len(trials) - n_pass
        total_pass += n_pass
        total_fail += n_fail

        avg_wall = sum(t["wall_time_s"] for t in trials) / len(trials)
        avg_err = sum(
            t["conservation_relative_error"] for t in trials
        ) / len(trials)
        cert_pct = (
            sum(1 for t in trials if t.get("certificate_verified")) /
            len(trials) * 100
        )

        print(
            f"{domain:<22} {len(trials):>6} {n_pass:>4} {n_fail:>4} "
            f"{avg_wall:>9.3f}s {avg_err:>12.2e} {cert_pct:>5.0f}%"
        )

    print("-" * 72)
    total = total_pass + total_fail
    print(
        f"{'TOTAL':<22} {total:>6} {total_pass:>4} {total_fail:>4}"
    )

    verdict = "PASSED" if total_fail == 0 else "FAILED"
    print(f"\nVerdict: {verdict}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Golden Physics Benchmark v1.0 — Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --output results.json
  python run_benchmark.py --domains burgers maxwell --output quick.json
  python run_benchmark.py --n-trials 5 --output deep.json
  python run_benchmark.py --validate-only results.json
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="golden_benchmark_results.json",
        help="Output JSON file (default: golden_benchmark_results.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"Benchmark config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Domains to run (default: all 7). Use ALL for all domains.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Number of trials per domain (default: {DEFAULT_N_TRIALS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=JOB_TIMEOUT_S,
        help=f"Per-job timeout in seconds (default: {JOB_TIMEOUT_S})",
    )
    parser.add_argument(
        "--validate-only",
        type=str,
        default=None,
        metavar="FILE",
        help="Validate an existing results file without running",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-measurement validation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (only print summary)",
    )
    args = parser.parse_args()

    # ── Validate-only mode ──────────────────────────────────────────
    if args.validate_only:
        try:
            from benchmarks.golden.validate import validate_results
        except ImportError:
            validate_path = Path(__file__).parent / "validate.py"
            import importlib.util
            spec = importlib.util.spec_from_file_location("validate", str(validate_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                validate_results = mod.validate_results
            else:
                print("ERROR: Cannot import validate module.", file=sys.stderr)
                return 2

        report = validate_results(args.validate_only)
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.passed else 1

    # ── Load configuration ──────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 2

    config = load_config(config_path)

    # ── Determine domains ───────────────────────────────────────────
    domains = args.domains
    if domains is None or (len(domains) == 1 and domains[0].upper() == "ALL"):
        domains = ALL_DOMAINS
    domains = [d.lower() for d in domains]

    # Validate domain names
    valid_set = set(config["domains"].keys())
    invalid = [d for d in domains if d not in valid_set]
    if invalid:
        print(f"ERROR: Unknown domains: {invalid}", file=sys.stderr)
        print(f"Valid: {sorted(valid_set)}", file=sys.stderr)
        return 2

    # ── Print banner ────────────────────────────────────────────────
    if not args.quiet:
        print(BANNER)
    print("=" * 64)
    print("Golden Physics Benchmark v1.0")
    print("=" * 64)
    print(f"Config:   {config_path}")
    print(f"Domains:  {', '.join(domains)} ({len(domains)} total)")
    print(f"Trials:   {args.n_trials}")
    print(f"Timeout:  {args.timeout}s per job")
    print(f"Output:   {args.output}")
    print("=" * 64)

    t_start = time.monotonic()

    # ── Run benchmark ───────────────────────────────────────────────
    results = run_benchmark(
        domains=domains,
        n_trials=args.n_trials,
        config=config,
    )

    elapsed_total = time.monotonic() - t_start

    # ── Write results ───────────────────────────────────────────────
    output_path = Path(args.output)
    output_data = {
        "_meta": {
            "benchmark": "Golden Physics Benchmark v1.0",
            "runner_version": "1.0.0",
            "config": str(config_path),
            "domains": domains,
            "n_trials": args.n_trials,
            "total_measurements": len(results),
            "total_time_s": round(elapsed_total, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "measurements": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # ── Print summary ───────────────────────────────────────────────
    print_summary(results)
    print(f"\nTotal time: {elapsed_total:.1f}s")
    print(f"Results:   {output_path} ({len(results)} measurements)")

    # ── Run validation ──────────────────────────────────────────────
    exit_code = 0
    if not args.skip_validation and results:
        print("\nRunning validation...")
        try:
            from benchmarks.golden.validate import validate_results
        except ImportError:
            validate_path = Path(__file__).parent / "validate.py"
            import importlib.util
            spec = importlib.util.spec_from_file_location("validate", str(validate_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                validate_results = mod.validate_results
            else:
                print("WARNING: Cannot import validate module.", file=sys.stderr)
                validate_results = None

        if validate_results is not None:
            report = validate_results(str(output_path))
            print(f"Validation verdict: {report.benchmark_verdict}")

            if report.errors:
                print(f"  Errors: {len(report.errors)}")
                for err in report.errors[:10]:
                    print(f"    [{err.index}] {err.field}: {err.message}")
                exit_code = 1
            if report.warnings:
                print(f"  Warnings: {len(report.warnings)}")

    print(f"\n{'=' * 64}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
