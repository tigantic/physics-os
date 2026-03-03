#!/usr/bin/env python3
"""V&V Regression Gate — CI/CD release criterion enforcement.

Reads scenario JSON artifacts and enforces hard thresholds derived from
observed production runs.  Exit code 0 = pass (ship it), 1 = fail (block).

Thresholds are set with headroom above observed values so that real
regressions trip the gate while normal numeric noise does not.

Observed baselines (2026-03-03, commit b18b9a85, RTX 5070 Laptop GPU):

  Scenario 1 (TG_LH_512_QTT_PROD):
    enstrophy_error_rel:  ~5e-7   → gate at 1e-4
    omega_l2_error_rel:   ~2.5e-7 → gate at 1e-4
    div_relative_to_vel:  ~7e-5   → gate at 1e-3
    reproducibility_enst: ~1.5e-10 → gate at 1e-6
    reproducibility_ol2:  ~7.7e-11 → gate at 1e-6
    poisson_max_residual: ~1.0e-3 → gate at 1.05e-3
    cold_iters:           29      → gate at 40
    poisson_stability:    ~1.04   → gate at 2.0

  Scenario 2 (TG_XCHECK_256_QTT_vs_DENSEFDFFT):
    field_omega_rel_l2:   ~3.5e-8 → gate at 5e-3
    enstrophy_rel_diff:   ~4.4e-12 → gate at 1e-4
    omega_l2_rel_diff:    ~2.2e-12 → gate at 1e-4

Usage:
    python3 scripts/vv_regression_gate.py [--scenario-dir scenario_output/data]
    python3 scripts/vv_regression_gate.py --check-only scenario1
    python3 scripts/vv_regression_gate.py --check-only scenario2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

# ═══════════════════════════════════════════════════════════════════════
# Gate definitions
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GateCheck:
    """A single regression gate check."""

    name: str
    value: float
    threshold: float
    op: str = "<="  # comparison operator: "<=" or ">="
    severity: str = "error"  # "error" blocks release, "warn" is advisory
    passed: bool = False
    detail: str = ""

    def evaluate(self) -> bool:
        if math.isnan(self.value):
            self.passed = False
            self.detail = "value is NaN"
            return False
        if self.op == "<=":
            self.passed = self.value <= self.threshold
        elif self.op == ">=":
            self.passed = self.value >= self.threshold
        else:
            self.passed = False
            self.detail = f"unknown op: {self.op}"
        return self.passed


@dataclass
class GateResult:
    """Aggregated gate result for a scenario."""

    scenario: str
    passed: bool = True
    checks: list[GateCheck] = field(default_factory=list)
    errors: int = 0
    warnings: int = 0
    missing: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Scenario 1: TG_LH_512_QTT_PROD
# ═══════════════════════════════════════════════════════════════════════

def gate_scenario1(data: dict[str, Any]) -> GateResult:
    """Enforce regression gates on the long-horizon 512² production run."""
    result = GateResult(scenario="TG_LH_512_QTT_PROD")

    seeds = data.get("seeds", {})
    if not seeds:
        result.passed = False
        result.missing = "no seeds data found"
        return result

    # ── Per-seed accuracy & constraint gates ──────────────────────
    for sid, sdata in seeds.items():
        final = sdata.get("final", {})
        poisson = sdata.get("poisson", {})
        series = sdata.get("series", {})

        # Accuracy: enstrophy error vs analytical
        result.checks.append(GateCheck(
            name=f"enstrophy_error_rel_seed{sid}",
            value=final.get("enstrophy_error_rel", float("nan")),
            threshold=1e-4,
            severity="error",
        ))

        # Accuracy: omega L2 error vs analytical
        result.checks.append(GateCheck(
            name=f"omega_l2_error_rel_seed{sid}",
            value=final.get("omega_l2_error_rel", float("nan")),
            threshold=1e-4,
            severity="error",
        ))

        # Constraint: divergence
        result.checks.append(GateCheck(
            name=f"div_relative_to_vel_seed{sid}",
            value=final.get("div_relative_to_vel", float("nan")),
            threshold=1e-3,
            severity="error",
        ))

        # Solver: Poisson max residual bounded below tol
        result.checks.append(GateCheck(
            name=f"poisson_max_residual_seed{sid}",
            value=poisson.get("max_residual", float("nan")),
            threshold=1.05e-3,
            severity="error",
        ))

        # Solver: cold start iters bounded
        result.checks.append(GateCheck(
            name=f"cold_iters_seed{sid}",
            value=float(poisson.get("cold_iters", float("nan"))),
            threshold=40.0,
            severity="error",
        ))

        # Solver stability: last residual ≤ 2× first residual
        resid_summary = series.get("poisson_residual", {})
        if isinstance(resid_summary, dict) and resid_summary.get("n", 0) >= 2:
            first = resid_summary["first"]
            last = resid_summary["last"]
            ratio = last / (first + 1e-30)
        else:
            ratio = float("nan")
        result.checks.append(GateCheck(
            name=f"poisson_stability_seed{sid}",
            value=ratio,
            threshold=2.0,
            severity="error",
        ))

    # ── Reproducibility gates (across seeds) ──────────────────────
    seed_keys = sorted(seeds.keys())
    if len(seed_keys) >= 2:
        s0, s1 = seed_keys[0], seed_keys[1]
        e0 = seeds[s0]["final"]["enstrophy"]
        e1 = seeds[s1]["final"]["enstrophy"]
        e_avg = 0.5 * (abs(e0) + abs(e1))
        delta_e = abs(e0 - e1) / e_avg if e_avg > 1e-30 else float("nan")

        o0 = seeds[s0]["final"]["omega_l2"]
        o1 = seeds[s1]["final"]["omega_l2"]
        o_avg = 0.5 * (abs(o0) + abs(o1))
        delta_o = abs(o0 - o1) / o_avg if o_avg > 1e-30 else float("nan")

        result.checks.append(GateCheck(
            name="reproducibility_enstrophy",
            value=delta_e,
            threshold=1e-6,
            severity="error",
        ))
        result.checks.append(GateCheck(
            name="reproducibility_omega_l2",
            value=delta_o,
            threshold=1e-6,
            severity="error",
        ))

    # ── Performance gate (advisory) ───────────────────────────────
    for sid, sdata in seeds.items():
        wall = sdata.get("wall_time_s", float("nan"))
        # Baseline ~1200s per seed.  Gate at 1.25× = 1500s.
        # This is a warn, not error — GPU thermals and load vary.
        result.checks.append(GateCheck(
            name=f"wall_time_seed{sid}",
            value=wall,
            threshold=1500.0,
            severity="warn",
        ))

    # Evaluate all
    for c in result.checks:
        c.evaluate()
        if not c.passed:
            if c.severity == "error":
                result.errors += 1
            else:
                result.warnings += 1

    result.passed = result.errors == 0
    return result


# ═══════════════════════════════════════════════════════════════════════
# Scenario 2: TG_XCHECK_256_QTT_vs_DENSEFDFFT
# ═══════════════════════════════════════════════════════════════════════

def gate_scenario2(data: dict[str, Any]) -> GateResult:
    """Enforce regression gates on the dense FFT cross-check."""
    result = GateResult(scenario="TG_XCHECK_256_QTT_vs_DENSEFDFFT")

    comp = data.get("comparison", {})
    qtt = data.get("qtt", {})
    poisson = qtt.get("poisson", {})

    # Field agreement: rel_L2(omega)
    result.checks.append(GateCheck(
        name="field_omega_rel_l2",
        value=comp.get("omega_rel_l2", float("nan")),
        threshold=5e-3,
        severity="error",
    ))

    # QoI agreement: enstrophy
    result.checks.append(GateCheck(
        name="enstrophy_rel_diff",
        value=comp.get("enstrophy_rel_diff", float("nan")),
        threshold=1e-4,
        severity="error",
    ))

    # QoI agreement: omega L2
    result.checks.append(GateCheck(
        name="omega_l2_rel_diff",
        value=comp.get("omega_l2_rel_diff", float("nan")),
        threshold=1e-4,
        severity="error",
    ))

    # Poisson max residual
    result.checks.append(GateCheck(
        name="poisson_max_residual",
        value=poisson.get("max_residual", float("nan")),
        threshold=1.05e-3,
        severity="error",
    ))

    # Constraint: divergence
    result.checks.append(GateCheck(
        name="div_relative_to_vel",
        value=qtt.get("div_relative_to_vel", float("nan")),
        threshold=1e-3,
        severity="error",
    ))

    # Performance (advisory)
    result.checks.append(GateCheck(
        name="wall_time_total",
        value=data.get("total_wall_time_s", float("nan")),
        threshold=600.0,  # ~1.5× baseline of ~415s
        severity="warn",
    ))

    for c in result.checks:
        c.evaluate()
        if not c.passed:
            if c.severity == "error":
                result.errors += 1
            else:
                result.warnings += 1

    result.passed = result.errors == 0
    return result


# ═══════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════

def _format_result(gr: GateResult) -> str:
    """Human-readable gate report."""
    lines: list[str] = []
    status = "PASS" if gr.passed else "FAIL"
    lines.append(f"{'=' * 64}")
    lines.append(f"GATE: {gr.scenario}  [{status}]")
    lines.append(f"{'=' * 64}")

    if gr.missing:
        lines.append(f"  MISSING: {gr.missing}")
        return "\n".join(lines)

    for c in gr.checks:
        icon = "✓" if c.passed else ("✗" if c.severity == "error" else "⚠")
        sev = f"[{c.severity.upper()}]" if not c.passed else ""
        if math.isnan(c.value):
            val_str = "NaN"
        elif c.value >= 1e6 or (c.value != 0 and abs(c.value) < 1e-3):
            val_str = f"{c.value:.4e}"
        else:
            val_str = f"{c.value:.4f}"
        lines.append(f"  {icon} {c.name}: {val_str} {c.op} {c.threshold}  {sev}")

    lines.append(f"  errors={gr.errors}, warnings={gr.warnings}")
    return "\n".join(lines)


def _to_json(gr: GateResult) -> dict[str, Any]:
    """Serializable gate report."""
    return {
        "scenario": gr.scenario,
        "passed": gr.passed,
        "errors": gr.errors,
        "warnings": gr.warnings,
        "checks": [
            {
                "name": c.name,
                "value": c.value,
                "threshold": c.threshold,
                "op": c.op,
                "severity": c.severity,
                "passed": c.passed,
            }
            for c in gr.checks
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

SCENARIO_FILES = {
    "scenario1": "tg_lh_512_qtt_prod.json",
    "scenario2": "tg_xcheck_256_qtt_vs_densefdfft.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V&V Regression Gate — CI/CD release criterion enforcement",
    )
    parser.add_argument(
        "--scenario-dir",
        default="scenario_output/data",
        help="Directory containing scenario JSON artifacts",
    )
    parser.add_argument(
        "--check-only",
        choices=["scenario1", "scenario2"],
        default=None,
        help="Check only one scenario (default: both)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON gate report to this path",
    )
    args = parser.parse_args()

    scenarios_to_check = (
        [args.check_only] if args.check_only else list(SCENARIO_FILES.keys())
    )

    results: list[GateResult] = []
    all_pass = True

    for scenario_key in scenarios_to_check:
        filename = SCENARIO_FILES[scenario_key]
        path = os.path.join(args.scenario_dir, filename)

        if not os.path.exists(path):
            gr = GateResult(scenario=scenario_key, passed=False)
            gr.missing = f"artifact not found: {path}"
            gr.errors = 1
            results.append(gr)
            all_pass = False
            print(_format_result(gr))
            continue

        with open(path) as f:
            data = json.load(f)

        if scenario_key == "scenario1":
            gr = gate_scenario1(data)
        else:
            gr = gate_scenario2(data)

        results.append(gr)
        print(_format_result(gr))
        print()

        if not gr.passed:
            all_pass = False

    # Summary
    print("=" * 64)
    if all_pass:
        print("RELEASE GATE: PASS — all scenarios within thresholds")
    else:
        print("RELEASE GATE: FAIL — regressions detected")
    print("=" * 64)

    # Optional JSON output
    if args.output:
        report = {
            "passed": all_pass,
            "scenarios": [_to_json(gr) for gr in results],
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written: {args.output}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
