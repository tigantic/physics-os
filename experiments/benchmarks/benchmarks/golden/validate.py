#!/usr/bin/env python3
"""Golden Physics Benchmark v1.0 — Result Validation Checker.

Validates submitted benchmark results against the golden schema and
acceptance criteria. Produces a structured report of schema violations,
consistency errors, and benchmark pass/fail verdict.

Usage:
    python validate.py <results_file.json> [--strict] [--report <out.json>]

Exit codes:
    0 — All checks passed
    1 — Validation errors found
    2 — File not found or unparseable

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_PATH = Path(__file__).parent / "schema.json"

VALID_DOMAINS = {
    "burgers",
    "maxwell",
    "maxwell_3d",
    "schrodinger",
    "advection_diffusion",
    "vlasov_poisson",
    "navier_stokes_2d",
}

VALID_DEVICES = {"cpu", "cuda", "gpu"}

VALID_CONSERVATION_STATUS = {"conserved", "drift"}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationError:
    """A single validation error or warning."""

    index: int  # Measurement index (-1 for global)
    field: str
    message: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class ValidationReport:
    """Complete validation report."""

    file_path: str
    total_measurements: int = 0
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    domain_coverage: dict[str, int] = field(default_factory=dict)
    domain_pass_rate: dict[str, float] = field(default_factory=dict)
    domain_cert_rate: dict[str, float] = field(default_factory=dict)
    domain_conservation: dict[str, dict[str, Any]] = field(default_factory=dict)
    benchmark_verdict: str = "NOT_EVALUATED"

    def add_error(
        self, index: int, fld: str, msg: str, *, severity: str = "error"
    ) -> None:
        entry = ValidationError(index=index, field=fld, message=msg, severity=severity)
        if severity == "warning":
            self.warnings.append(entry)
        else:
            self.errors.append(entry)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "total_measurements": self.total_measurements,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [
                {"index": e.index, "field": e.field, "message": e.message}
                for e in self.errors
            ],
            "warnings": [
                {"index": w.index, "field": w.field, "message": w.message}
                for w in self.warnings
            ],
            "domain_coverage": self.domain_coverage,
            "domain_pass_rate": self.domain_pass_rate,
            "domain_cert_rate": self.domain_cert_rate,
            "domain_conservation": self.domain_conservation,
            "benchmark_verdict": self.benchmark_verdict,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "domain": str,
    "trial": int,
    "seed": int,
    "n_bits": int,
    "n_steps": int,
    "max_rank": int,
    "n_dims": int,
    "grid_points": int,
    "wall_time_s": (int, float),
    "throughput_gp_per_s": (int, float),
    "conservation_quantity": str,
    "conservation_initial": (int, float, type(None)),
    "conservation_final": (int, float, type(None)),
    "conservation_relative_error": (int, float),
    "conservation_status": str,
    "conservation_within_band": bool,
    "conservation_band_max": (int, float),
    "certificate_issued": bool,
    "certificate_verified": bool,
    "certificate_job_id": str,
    "n_claims": int,
    "pipeline_success": bool,
    "device": str,
    "timestamp": str,
}


def _validate_schema(
    record: dict[str, Any], index: int, report: ValidationReport
) -> bool:
    """Validate a single measurement against required schema fields."""
    valid = True

    for fld, expected_type in REQUIRED_FIELDS.items():
        if fld not in record:
            report.add_error(index, fld, f"Missing required field '{fld}'.")
            valid = False
            continue

        val = record[fld]
        if not isinstance(val, expected_type):
            report.add_error(
                index,
                fld,
                f"Field '{fld}' has type {type(val).__name__}, "
                f"expected {expected_type}.",
            )
            valid = False

    return valid


def _validate_consistency(
    record: dict[str, Any], index: int, report: ValidationReport
) -> None:
    """Check internal consistency of a measurement record."""

    # Domain validity
    domain = record.get("domain", "")
    if domain not in VALID_DOMAINS:
        report.add_error(index, "domain", f"Unknown domain '{domain}'.")

    # Device validity
    device = record.get("device", "")
    if device not in VALID_DEVICES:
        report.add_error(index, "device", f"Invalid device '{device}'.")

    # Conservation status validity
    status = record.get("conservation_status", "")
    if status not in VALID_CONSERVATION_STATUS:
        report.add_error(
            index, "conservation_status",
            f"Invalid status '{status}', expected {VALID_CONSERVATION_STATUS}.",
        )

    # n_bits range
    n_bits = record.get("n_bits", 0)
    if not 2 <= n_bits <= 20:
        report.add_error(
            index, "n_bits", f"n_bits={n_bits} outside valid range [2, 20]."
        )

    # n_steps positive
    n_steps = record.get("n_steps", 0)
    if n_steps < 1:
        report.add_error(index, "n_steps", f"n_steps={n_steps} must be >= 1.")

    # max_rank positive
    max_rank = record.get("max_rank", 0)
    if max_rank < 1:
        report.add_error(index, "max_rank", f"max_rank={max_rank} must be >= 1.")

    # Wall time non-negative
    wall = record.get("wall_time_s", -1)
    if wall < 0:
        report.add_error(index, "wall_time_s", f"wall_time_s={wall} must be >= 0.")

    # Conservation error non-negative
    cons_err = record.get("conservation_relative_error", -1)
    if cons_err < 0:
        report.add_error(
            index, "conservation_relative_error",
            f"conservation_relative_error={cons_err} must be >= 0.",
        )

    # Conservation band consistency
    band_max = record.get("conservation_band_max", 0)
    within = record.get("conservation_within_band", True)
    if cons_err >= 0 and band_max > 0:
        expected_within = cons_err <= band_max
        if within != expected_within:
            report.add_error(
                index, "conservation_within_band",
                f"conservation_within_band={within} inconsistent with "
                f"error={cons_err:.2e} vs band_max={band_max:.2e}.",
            )

    # n_claims non-negative
    n_claims = record.get("n_claims", -1)
    if n_claims < 0:
        report.add_error(index, "n_claims", f"n_claims={n_claims} must be >= 0.")

    # Trial positive
    trial = record.get("trial", 0)
    if trial < 1:
        report.add_error(index, "trial", f"trial={trial} must be >= 1.")

    # Grid points consistency
    grid_points = record.get("grid_points", 0)
    n_dims = record.get("n_dims", 0)
    if record.get("pipeline_success") and grid_points < 1:
        report.add_error(
            index, "grid_points",
            f"grid_points={grid_points} must be >= 1 for successful pipeline.",
        )

    # Certificate consistency
    if record.get("pipeline_success"):
        if not record.get("certificate_issued"):
            report.add_error(
                index, "certificate_issued",
                "Pipeline succeeded but no certificate was issued.",
            )
        if not record.get("certificate_verified"):
            report.add_error(
                index, "certificate_verified",
                "Pipeline succeeded but certificate did not verify.",
            )


def _validate_acceptance(
    records: list[dict[str, Any]], report: ValidationReport
) -> None:
    """Check cross-domain acceptance criteria."""
    from collections import defaultdict

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_domain[r.get("domain", "unknown")].append(r)

    # Domain coverage
    for domain in VALID_DOMAINS:
        trials = by_domain.get(domain, [])
        report.domain_coverage[domain] = len(trials)

        if not trials:
            report.add_error(
                -1, "coverage",
                f"Domain '{domain}' has no measurements.",
                severity="warning",
            )
            continue

        # Pass rate
        n_pass = sum(1 for t in trials if t.get("pipeline_success"))
        pass_rate = n_pass / len(trials)
        report.domain_pass_rate[domain] = round(pass_rate, 4)

        if pass_rate < 1.0:
            report.add_error(
                -1, "pass_rate",
                f"Domain '{domain}' pass rate {pass_rate:.0%} < 100%.",
            )

        # Certificate rate
        n_cert = sum(1 for t in trials if t.get("certificate_verified"))
        cert_rate = n_cert / len(trials)
        report.domain_cert_rate[domain] = round(cert_rate, 4)

        if cert_rate < 1.0:
            report.add_error(
                -1, "cert_rate",
                f"Domain '{domain}' cert rate {cert_rate:.0%} < 100%.",
            )

        # Conservation band
        n_in_band = sum(1 for t in trials if t.get("conservation_within_band"))
        band_rate = n_in_band / len(trials)

        errors = [
            t.get("conservation_relative_error", 0) for t in trials
            if t.get("pipeline_success")
        ]
        avg_err = sum(errors) / max(len(errors), 1)
        max_err = max(errors) if errors else 0

        report.domain_conservation[domain] = {
            "in_band_rate": round(band_rate, 4),
            "avg_error": avg_err,
            "max_error": max_err,
        }

        if band_rate < 1.0:
            report.add_error(
                -1, "conservation",
                f"Domain '{domain}' conservation in-band rate {band_rate:.0%} < 100%.",
            )

    # Wall time band checks
    for domain, trials in by_domain.items():
        for i, t in enumerate(trials):
            wt = t.get("wall_time_s", 0)
            wt_max = t.get("wall_time_band_max", float("inf"))
            if wt > wt_max:
                report.add_error(
                    -1, "wall_time",
                    f"Domain '{domain}' trial {t.get('trial', '?')} "
                    f"wall_time={wt:.2f}s exceeds band {wt_max:.1f}s.",
                    severity="warning",
                )

    # Overall verdict
    total_measured = sum(report.domain_coverage.values())
    total_pass = sum(
        v for v in report.domain_pass_rate.values()
    )
    all_domains_covered = all(
        report.domain_coverage.get(d, 0) > 0 for d in VALID_DOMAINS
    )
    all_pass = all(
        report.domain_pass_rate.get(d, 0) >= 1.0 for d in VALID_DOMAINS
        if report.domain_coverage.get(d, 0) > 0
    )
    all_cert = all(
        report.domain_cert_rate.get(d, 0) >= 1.0 for d in VALID_DOMAINS
        if report.domain_coverage.get(d, 0) > 0
    )
    all_conserved = all(
        report.domain_conservation.get(d, {}).get("in_band_rate", 0) >= 1.0
        for d in VALID_DOMAINS
        if report.domain_coverage.get(d, 0) > 0
    )

    if report.errors:
        report.benchmark_verdict = "FAILED"
    elif all_domains_covered and all_pass and all_cert and all_conserved:
        report.benchmark_verdict = "PASSED"
    elif not all_domains_covered:
        report.benchmark_verdict = "INCOMPLETE"
    else:
        report.benchmark_verdict = "DEGRADED"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def validate_results(file_path: str, strict: bool = False) -> ValidationReport:
    """Validate a golden benchmark results file.

    Parameters
    ----------
    file_path : str
        Path to the JSON results file.
    strict : bool
        If True, treat warnings as errors.

    Returns
    -------
    ValidationReport
        Structured validation report.
    """
    report = ValidationReport(file_path=file_path)

    # Load file
    path = Path(file_path)
    if not path.exists():
        report.add_error(-1, "file", f"File not found: {path}")
        report.benchmark_verdict = "FAILED"
        return report

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        report.add_error(-1, "file", f"JSON parse error: {exc}")
        report.benchmark_verdict = "FAILED"
        return report

    # Handle wrapped format (with _meta + measurements) or raw array
    if isinstance(data, dict) and "measurements" in data:
        measurements = data["measurements"]
    elif isinstance(data, list):
        measurements = data
    else:
        report.add_error(
            -1, "format",
            "Expected JSON array or object with 'measurements' key.",
        )
        report.benchmark_verdict = "FAILED"
        return report

    if not isinstance(measurements, list):
        report.add_error(-1, "format", "'measurements' must be an array.")
        report.benchmark_verdict = "FAILED"
        return report

    report.total_measurements = len(measurements)

    if len(measurements) == 0:
        report.add_error(-1, "measurements", "No measurements found.")
        report.benchmark_verdict = "FAILED"
        return report

    # Per-measurement validation
    for idx, record in enumerate(measurements):
        if not isinstance(record, dict):
            report.add_error(idx, "type", "Measurement must be a JSON object.")
            continue

        schema_ok = _validate_schema(record, idx, report)
        if schema_ok:
            _validate_consistency(record, idx, report)

    # Cross-domain acceptance
    _validate_acceptance(measurements, report)

    # Strict mode: promote warnings
    if strict:
        for w in report.warnings:
            w.severity = "error"
            report.errors.append(w)
        report.warnings.clear()
        if report.errors and report.benchmark_verdict == "PASSED":
            report.benchmark_verdict = "FAILED"

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Golden Physics Benchmark v1.0 — Validator"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the benchmark results JSON file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write validation report to JSON file.",
    )
    args = parser.parse_args()

    report = validate_results(args.results_file, strict=args.strict)

    # Print summary
    print(f"File: {report.file_path}")
    print(f"Measurements: {report.total_measurements}")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")

    if report.errors:
        print("\nErrors:")
        for err in report.errors:
            idx_str = f"[{err.index}]" if err.index >= 0 else "[global]"
            print(f"  {idx_str} {err.field}: {err.message}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            idx_str = f"[{w.index}]" if w.index >= 0 else "[global]"
            print(f"  {idx_str} {w.field}: {w.message}")

    print(f"\nDomain coverage: {report.domain_coverage}")
    print(f"Pass rates: {report.domain_pass_rate}")
    print(f"Cert rates: {report.domain_cert_rate}")
    print(f"\nVerdict: {report.benchmark_verdict}")

    # Write report
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"\nReport written to: {args.report}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
