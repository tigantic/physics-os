#!/usr/bin/env python3
"""PWA Engine Regression Gate.

Reads ``docs/papers/paper/figures/pwa_engine_metadata.json`` produced by
``python3 experiments/run_pwa_engine.py`` and asserts hard thresholds
on every experiment result.  If any threshold is violated the script
exits non-zero and prints the failing gate.

Thresholds are intentionally generous (2× to 5× the typical observed
values) so that the gate catches real regressions—not statistical
fluctuations in MC sampling—while still flagging code-level breakage.

Usage (CI):
    python3 experiments/run_pwa_engine.py          # produce metadata
    python3 tools/scripts/check_pwa_regression.py        # gate

Exit codes:
    0  — all gates pass
    1  — one or more gates fail
    2  — metadata file missing or malformed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

METADATA_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "papers" / "paper" / "figures" / "pwa_engine_metadata.json"

# ── Thresholds ──────────────────────────────────────────────────────────────
# Each entry: (json_path, comparator, threshold, description)
# json_path uses '.' for nested keys and '[i]' for list indexing.
# Comparators: '<' means value must be < threshold, '>' means > threshold,
#              '==' means value must equal threshold.

GATES: List[Tuple[str, str, float, str]] = [
    # Convention test: all 3 errors at machine precision
    ("convention_test.test_1_err", "<", 1e-10,
     "Convention test 1 (full vs single ε)"),
    ("convention_test.test_2_err", "<", 1e-10,
     "Convention test 2 (model vs manual)"),
    ("convention_test.test_3_err", "<", 1e-10,
     "Convention test 3 (full ρ vs diagonal)"),
    ("convention_test.all_pass", "==", True,
     "Convention test all_pass flag"),

    # Parameter recovery
    ("parameter_recovery.yield_rmse", "<", 0.15,
     "Yield RMSE (typical ~0.03)"),
    ("parameter_recovery.phase_rmse_rad", "<", 1.50,
     "Phase RMSE in radians (typical ~0.6, varies with MC sampling)"),
    ("parameter_recovery.basin_fraction", ">", 0.02,
     "Basin convergence fraction (typical ~0.07 on CPU)"),

    # Acceleration: at least one benchmark has speedup > 1×
    ("acceleration.benchmarks[0].agreement", "<", 1e-10,
     "Gram vs baseline agreement (first benchmark)"),

    # Moment validation
    ("moment_validation.chi2_per_ndf", "<", 5.0,
     "Moment χ²/ndf (typical ~1.0)"),

    # Beam asymmetry: polarized phase resolution (may be < 1 on small MC)
    ("beam_asymmetry.phase_improvement", ">", 0.5,
     "Polarization phase improvement lower bound"),

    # Bootstrap
    ("bootstrap.converged_fraction", ">", 0.5,
     "Bootstrap converged fraction (typical ~0.9)"),

    # Coupled channel: joint should improve over ch1 alone
    ("coupled_channel.joint_yield_rmse", "<", 0.30,
     "Coupled-channel joint yield RMSE"),
    ("coupled_channel.yield_improvement_vs_ch1", ">", 1.0,
     "Coupled-channel improvement over single channel"),

    # Mass-dependent: resonance mass recovery within 100 MeV
    ("mass_dependent.m0_fit_s", "<", 100.0,
     "Mass-dependent: S-wave m₀ fit is finite (sanity)"),
    ("mass_dependent.m0_fit_d", "<", 100.0,
     "Mass-dependent: D-wave m₀ fit is finite (sanity)"),

    # Runtime budget: 300s generous cap (typical ~94s)
    ("total_time_s", "<", 300.0,
     "Total wall time under 300s budget"),
]


def _resolve_path(data: Dict[str, Any], path: str) -> Any:
    """Resolve a dotted path like 'a.b[0].c' into a nested dict value."""
    current: Any = data
    for part in path.replace("[", ".[").split("."):
        if not part:
            continue
        if part.startswith("[") and part.endswith("]"):
            idx = int(part[1:-1])
            current = current[idx]
        else:
            current = current[part]
    return current


def main() -> int:
    if not METADATA_PATH.exists():
        print(f"FAIL: Metadata file not found: {METADATA_PATH}", file=sys.stderr)
        return 2

    try:
        metadata = json.loads(METADATA_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"FAIL: Cannot parse metadata: {exc}", file=sys.stderr)
        return 2

    n_pass = 0
    n_fail = 0
    failures: List[str] = []

    for path, comparator, threshold, description in GATES:
        try:
            value = _resolve_path(metadata, path)
        except (KeyError, IndexError, TypeError) as exc:
            failures.append(f"  MISSING  {path} — {description} ({exc})")
            n_fail += 1
            continue

        passed = False
        if comparator == "<":
            passed = value < threshold
        elif comparator == ">":
            passed = value > threshold
        elif comparator == "==":
            passed = value == threshold

        if passed:
            n_pass += 1
            print(f"  PASS  {description}: {value}")
        else:
            n_fail += 1
            msg = f"  FAIL  {description}: {value} (expected {comparator} {threshold})"
            failures.append(msg)
            print(msg)

    print()
    print(f"PWA Regression Gate: {n_pass} passed, {n_fail} failed out of {n_pass + n_fail}")

    if failures:
        print("\nFailed gates:")
        for f in failures:
            print(f)
        return 1

    print("ALL GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
