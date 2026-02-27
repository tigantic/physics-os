#!/usr/bin/env python3
"""Rank Atlas Benchmark v1.0 — Result Validation Checker.

Validates submitted benchmark results against the Rank Atlas schema and
acceptance criteria. Produces a structured report of schema violations,
consistency errors, and benchmark pass/fail verdict.

Usage:
    python validate.py <results_file.json> [--strict] [--report <out.json>]

Exit codes:
    0 — All checks passed
    1 — Validation errors found
    2 — File not found or unparseable
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_PATH = Path(__file__).parent / "schema.json"

VALID_PACK_IDS = {
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
}

REQUIRED_N_BITS = {6, 7, 8, 9, 10}

REQUIRED_SEEDS = {42, 137, 2026}

GRID_INDEP_THRESHOLD = 0.05  # |b|/a < 5%

CONJECTURE_B_MAX_EXPONENT = 2.0

MIN_PACKS_GRID_INDEP = 18

VALID_DEVICES = {"cuda", "cpu"}

VALID_AREA_LAW_TYPES = {"area", "volume", "log_corrected"}

MIN_MAX_RANK_CEILING = 2048


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationError:
    """A single validation error."""

    index: int  # Measurement index in the array (-1 for global)
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
    pack_coverage: dict[str, int] = field(default_factory=dict)
    resolution_coverage: dict[str, set[int]] = field(default_factory=dict)
    grid_independence: dict[str, str] = field(default_factory=dict)
    scaling_classes: dict[str, str] = field(default_factory=dict)
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
            "pack_coverage": self.pack_coverage,
            "resolution_coverage": {
                k: sorted(v) for k, v in self.resolution_coverage.items()
            },
            "grid_independence": self.grid_independence,
            "scaling_classes": self.scaling_classes,
            "benchmark_verdict": self.benchmark_verdict,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "pack_id": str,
    "domain_name": str,
    "problem_name": str,
    "n_bits": int,
    "n_cells": int,
    "complexity_param_name": str,
    "complexity_param_value": (int, float),
    "svd_tolerance": (int, float),
    "max_rank_ceiling": int,
    "n_sites": int,
    "bond_dimensions": list,
    "singular_value_spectra": list,
    "max_rank": int,
    "mean_rank": (int, float),
    "median_rank": int,
    "rank_std": (int, float),
    "peak_rank_site": int,
    "rank_utilization": (int, float),
    "dense_bytes": int,
    "qtt_bytes": int,
    "compression_ratio": (int, float),
    "wall_time_s": (int, float),
    "peak_gpu_mem_mb": (int, float),
    "device": str,
    "timestamp": str,
    "seed": int,
}


def validate_measurement_schema(
    record: dict[str, Any], index: int, report: ValidationReport
) -> bool:
    """Validate a single measurement against the required schema fields.

    Returns True if all required fields are present and correctly typed.
    """
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


def validate_measurement_consistency(
    record: dict[str, Any], index: int, report: ValidationReport
) -> None:
    """Check internal consistency of a measurement record."""

    # Pack ID validity
    pack_id = record.get("pack_id", "")
    if pack_id not in VALID_PACK_IDS:
        report.add_error(index, "pack_id", f"Unknown pack_id '{pack_id}'.")

    # n_bits range
    n_bits = record.get("n_bits", 0)
    if not 4 <= n_bits <= 20:
        report.add_error(
            index, "n_bits", f"n_bits={n_bits} outside valid range [4, 20]."
        )

    # n_sites consistency
    n_sites = record.get("n_sites", 0)
    if n_sites < 1:
        report.add_error(index, "n_sites", f"n_sites={n_sites} must be >= 1.")

    # bond_dimensions length = n_sites - 1
    bond_dims = record.get("bond_dimensions", [])
    if n_sites > 0 and len(bond_dims) != n_sites - 1:
        report.add_error(
            index,
            "bond_dimensions",
            f"bond_dimensions has {len(bond_dims)} elements, "
            f"expected {n_sites - 1} (n_sites - 1).",
        )

    # singular_value_spectra length = n_sites - 1
    sv_spectra = record.get("singular_value_spectra", [])
    if n_sites > 0 and len(sv_spectra) != n_sites - 1:
        report.add_error(
            index,
            "singular_value_spectra",
            f"singular_value_spectra has {len(sv_spectra)} elements, "
            f"expected {n_sites - 1}.",
        )

    # SV spectra lengths must match bond dimensions
    for bond_idx, (bd, sv) in enumerate(zip(bond_dims, sv_spectra)):
        if isinstance(sv, list) and isinstance(bd, int):
            if len(sv) < bd:
                report.add_error(
                    index,
                    "singular_value_spectra",
                    f"Bond {bond_idx}: SV spectrum has {len(sv)} values "
                    f"but bond dimension is {bd}.",
                )

    # max_rank consistency
    max_rank = record.get("max_rank", 0)
    if bond_dims and max_rank != max(bond_dims):
        report.add_error(
            index,
            "max_rank",
            f"max_rank={max_rank} does not match max(bond_dimensions)="
            f"{max(bond_dims)}.",
        )

    # rank_utilization consistency
    mean_rank = record.get("mean_rank", 0)
    rank_util = record.get("rank_utilization", 0)
    if max_rank > 0:
        expected_util = mean_rank / max_rank
        if abs(rank_util - expected_util) > 0.01:
            report.add_error(
                index,
                "rank_utilization",
                f"rank_utilization={rank_util:.4f} inconsistent with "
                f"mean_rank/max_rank={expected_util:.4f}.",
            )

    # compression_ratio consistency
    dense_bytes = record.get("dense_bytes", 0)
    qtt_bytes = record.get("qtt_bytes", 0)
    comp_ratio = record.get("compression_ratio", 0)
    if qtt_bytes > 0 and dense_bytes > 0:
        expected_ratio = dense_bytes / qtt_bytes
        if abs(comp_ratio - expected_ratio) / max(expected_ratio, 1e-10) > 0.01:
            report.add_error(
                index,
                "compression_ratio",
                f"compression_ratio={comp_ratio:.4f} inconsistent with "
                f"dense_bytes/qtt_bytes={expected_ratio:.4f}.",
            )

    # max_rank_ceiling warning
    ceiling = record.get("max_rank_ceiling", 0)
    if ceiling < MIN_MAX_RANK_CEILING:
        report.add_error(
            index,
            "max_rank_ceiling",
            f"max_rank_ceiling={ceiling} < {MIN_MAX_RANK_CEILING} "
            f"(benchmark requires >= {MIN_MAX_RANK_CEILING}).",
            severity="warning",
        )

    # Rank ceiling saturation warning
    if max_rank > 0 and ceiling > 0 and max_rank >= ceiling:
        report.add_error(
            index,
            "max_rank",
            f"max_rank={max_rank} equals max_rank_ceiling={ceiling}. "
            f"Measurement may be ceiling-saturated.",
            severity="warning",
        )

    # Device check
    device = record.get("device", "")
    if device not in VALID_DEVICES:
        report.add_error(
            index, "device", f"device='{device}' not in {VALID_DEVICES}."
        )

    # SVD tolerance positivity
    svd_tol = record.get("svd_tolerance", 0)
    if svd_tol <= 0 or svd_tol >= 1:
        report.add_error(
            index,
            "svd_tolerance",
            f"svd_tolerance={svd_tol} must be in (0, 1).",
        )

    # Optional field type checks
    area_law_type = record.get("area_law_type")
    if area_law_type is not None and area_law_type not in VALID_AREA_LAW_TYPES:
        report.add_error(
            index,
            "area_law_type",
            f"area_law_type='{area_law_type}' not in {VALID_AREA_LAW_TYPES}.",
        )

    # Bond dimensions must be positive integers
    for bond_idx, bd in enumerate(bond_dims):
        if isinstance(bd, int) and bd < 1:
            report.add_error(
                index,
                "bond_dimensions",
                f"Bond {bond_idx}: bond dimension {bd} < 1.",
            )

    # SVs must be non-negative and descending
    for bond_idx, sv in enumerate(sv_spectra):
        if isinstance(sv, list) and len(sv) >= 2:
            for sv_idx in range(len(sv) - 1):
                if isinstance(sv[sv_idx], (int, float)) and isinstance(
                    sv[sv_idx + 1], (int, float)
                ):
                    if sv[sv_idx] < sv[sv_idx + 1] - 1e-15:
                        report.add_error(
                            index,
                            "singular_value_spectra",
                            f"Bond {bond_idx}: SVs not descending at "
                            f"positions {sv_idx},{sv_idx + 1} "
                            f"({sv[sv_idx]:.6e} < {sv[sv_idx + 1]:.6e}).",
                        )
                        break  # One error per bond is enough


# ─────────────────────────────────────────────────────────────────────────────
# Campaign-level analysis
# ─────────────────────────────────────────────────────────────────────────────


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Simple least-squares linear fit y = a + b*x.

    Returns (intercept, slope, r_squared).
    """
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0

    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        return sy / n, 0.0, 0.0

    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n

    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)

    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    return a, b, r_sq


def _power_fit(
    xs: list[float], ys: list[float]
) -> tuple[float, float, float]:
    """Power-law fit y = c * x^alpha via log-log linear regression.

    Returns (coefficient, exponent, r_squared).
    """
    log_xs = [math.log(x) for x in xs if x > 0]
    log_ys = [math.log(y) for y in ys if y > 0]

    if len(log_xs) < 2 or len(log_ys) < 2:
        return 1.0, 0.0, 0.0

    min_len = min(len(log_xs), len(log_ys))
    log_xs = log_xs[:min_len]
    log_ys = log_ys[:min_len]

    log_c, alpha, r_sq = _linear_fit(log_xs, log_ys)
    return math.exp(log_c), alpha, r_sq


def analyze_grid_independence(
    measurements: list[dict[str, Any]], report: ValidationReport
) -> None:
    """Test grid independence per pack: chi_max vs n_bits linear fit."""

    # Group measurements by (pack_id, complexity_param_value)
    groups: dict[tuple[str, float], list[tuple[int, int]]] = {}
    for m in measurements:
        key = (m["pack_id"], m.get("complexity_param_value", 0.0))
        n_bits = m.get("n_bits", 0)
        max_rank = m.get("max_rank", 0)
        groups.setdefault(key, []).append((n_bits, max_rank))

    pack_results: dict[str, list[bool]] = {}

    for (pack_id, cpv), points in groups.items():
        # Deduplicate by n_bits (take max rank across trials)
        by_nbits: dict[int, int] = {}
        for nb, mr in points:
            by_nbits[nb] = max(by_nbits.get(nb, 0), mr)

        if len(by_nbits) < 3:
            continue  # Need at least 3 resolution points

        xs = sorted(by_nbits.keys())
        ys = [by_nbits[x] for x in xs]

        a, b, r_sq = _linear_fit([float(x) for x in xs], [float(y) for y in ys])

        passed = abs(a) > 1e-10 and abs(b) / abs(a) < GRID_INDEP_THRESHOLD
        pack_results.setdefault(pack_id, []).append(passed)

    # Aggregate at pack level
    for pack_id, results in pack_results.items():
        all_pass = all(results)
        report.grid_independence[pack_id] = "PASS" if all_pass else "FAIL"


def analyze_scaling_classes(
    measurements: list[dict[str, Any]], report: ValidationReport
) -> None:
    """Classify each pack's complexity scaling behavior."""

    # Group by pack_id
    pack_data: dict[str, list[tuple[float, int]]] = {}
    for m in measurements:
        pack_id = m["pack_id"]
        cpv = m.get("complexity_param_value", 1.0)
        max_rank = m.get("max_rank", 0)
        cpn = m.get("complexity_param_name", "fixed")

        if cpn == "fixed":
            continue  # Can't fit scaling for fixed complexity

        pack_data.setdefault(pack_id, []).append((cpv, max_rank))

    for pack_id, points in pack_data.items():
        # Deduplicate by complexity value (take max rank)
        by_cpv: dict[float, int] = {}
        for cpv, mr in points:
            by_cpv[cpv] = max(by_cpv.get(cpv, 0), mr)

        if len(by_cpv) < 3:
            report.scaling_classes[pack_id] = "INSUFFICIENT_DATA"
            continue

        xs = sorted(by_cpv.keys())
        ys = [by_cpv[x] for x in xs]

        c, alpha, r_sq = _power_fit(
            [float(x) for x in xs], [float(y) for y in ys]
        )

        max_chi = max(ys)
        ceiling = 2048  # default

        if max_chi >= ceiling:
            cls = "D"
        elif abs(alpha) >= 0.5:
            cls = "C"
        elif abs(alpha) >= 0.1:
            cls = "B"
        else:
            cls = "A"

        report.scaling_classes[pack_id] = (
            f"{cls} (alpha={alpha:.4f}, R²={r_sq:.3f})"
        )


def evaluate_benchmark_verdict(report: ValidationReport) -> None:
    """Determine overall pass/fail based on acceptance criteria."""

    if report.errors:
        report.benchmark_verdict = "INVALID (schema errors)"
        return

    # Check pack coverage
    missing_packs = VALID_PACK_IDS - set(report.pack_coverage.keys())
    if missing_packs:
        report.add_error(
            -1,
            "pack_coverage",
            f"Missing packs: {sorted(missing_packs)}.",
            severity="warning",
        )

    # Check resolution coverage
    for pack_id, resolutions in report.resolution_coverage.items():
        missing_res = REQUIRED_N_BITS - resolutions
        if missing_res:
            report.add_error(
                -1,
                "resolution_coverage",
                f"Pack {pack_id} missing n_bits: {sorted(missing_res)}.",
                severity="warning",
            )

    # Grid independence check
    gi_pass_count = sum(
        1 for v in report.grid_independence.values() if v == "PASS"
    )
    gi_total = len(report.grid_independence)

    # Scaling class check
    class_d_count = sum(
        1 for v in report.scaling_classes.values() if v.startswith("D")
    )

    # Determine verdict
    if class_d_count >= 3:
        report.benchmark_verdict = "FAILED (>= 3 Class D packs)"
    elif gi_total > 0 and gi_pass_count < MIN_PACKS_GRID_INDEP:
        report.benchmark_verdict = (
            f"FAILED (grid independence: {gi_pass_count}/{gi_total}, "
            f"need >= {MIN_PACKS_GRID_INDEP})"
        )
    elif class_d_count > 0:
        report.benchmark_verdict = (
            f"CONDITIONAL (1-2 Class D packs, investigate)"
        )
    else:
        report.benchmark_verdict = "PASSED"


# ─────────────────────────────────────────────────────────────────────────────
# Main validation pipeline
# ─────────────────────────────────────────────────────────────────────────────


def validate_results(file_path: str, *, strict: bool = False) -> ValidationReport:
    """Run the full validation pipeline on a results file.

    Args:
        file_path: Path to a JSON file containing an array of measurements.
        strict: If True, treat warnings as errors.

    Returns:
        ValidationReport with all findings.
    """
    report = ValidationReport(file_path=file_path)

    # Load file
    path = Path(file_path)
    if not path.exists():
        report.add_error(-1, "file", f"File not found: {file_path}")
        return report

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        report.add_error(-1, "file", f"JSON parse error: {exc}")
        return report

    if not isinstance(data, list):
        report.add_error(
            -1, "file", f"Expected JSON array, got {type(data).__name__}."
        )
        return report

    report.total_measurements = len(data)

    if len(data) == 0:
        report.add_error(-1, "file", "Empty measurement array.")
        return report

    # Per-measurement validation
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            report.add_error(
                idx, "record", f"Expected object, got {type(record).__name__}."
            )
            continue

        schema_ok = validate_measurement_schema(record, idx, report)
        if schema_ok:
            validate_measurement_consistency(record, idx, report)

    # Build coverage maps from valid records
    valid_records: list[dict[str, Any]] = []
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            continue

        pack_id = record.get("pack_id", "")
        n_bits = record.get("n_bits", 0)

        if pack_id in VALID_PACK_IDS:
            report.pack_coverage[pack_id] = (
                report.pack_coverage.get(pack_id, 0) + 1
            )
            report.resolution_coverage.setdefault(pack_id, set()).add(n_bits)
            valid_records.append(record)

    # Campaign-level analysis (only if schema is clean enough)
    schema_error_count = sum(
        1 for e in report.errors if e.severity == "error"
    )
    if schema_error_count == 0 and valid_records:
        analyze_grid_independence(valid_records, report)
        analyze_scaling_classes(valid_records, report)
        evaluate_benchmark_verdict(report)

    # Strict mode: promote warnings to errors
    if strict:
        for w in report.warnings:
            w.severity = "error"
            report.errors.append(w)
        report.warnings.clear()

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank Atlas Benchmark v1.0 — Result Validator"
    )
    parser.add_argument(
        "results_file", help="Path to JSON results file"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write validation report to this JSON file",
    )
    args = parser.parse_args()

    report = validate_results(args.results_file, strict=args.strict)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Rank Atlas Benchmark v1.0 — Validation Report")
    print(f"{'=' * 60}")
    print(f"File: {report.file_path}")
    print(f"Measurements: {report.total_measurements}")
    print(f"Packs covered: {len(report.pack_coverage)}/20")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")
    print()

    if report.errors:
        print("ERRORS:")
        for e in report.errors[:20]:  # Cap display
            label = f"  [{e.index}]" if e.index >= 0 else "  [global]"
            print(f"{label} {e.field}: {e.message}")
        if len(report.errors) > 20:
            print(f"  ... and {len(report.errors) - 20} more errors")
        print()

    if report.warnings:
        print("WARNINGS:")
        for w in report.warnings[:10]:
            label = f"  [{w.index}]" if w.index >= 0 else "  [global]"
            print(f"{label} {w.field}: {w.message}")
        if len(report.warnings) > 10:
            print(f"  ... and {len(report.warnings) - 10} more warnings")
        print()

    if report.grid_independence:
        gi_pass = sum(1 for v in report.grid_independence.values() if v == "PASS")
        print(f"Grid Independence: {gi_pass}/{len(report.grid_independence)} pass")

    if report.scaling_classes:
        print("Scaling Classes:")
        for pack_id in sorted(report.scaling_classes.keys()):
            print(f"  Pack {pack_id}: {report.scaling_classes[pack_id]}")

    print(f"\nVerdict: {report.benchmark_verdict}")
    print(f"{'=' * 60}\n")

    # Write report file if requested
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"Report written to {args.report}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
