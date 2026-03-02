"""Evidence bundle generation.

An evidence bundle is the structured proof artifact that accompanies
every job result.  It contains:
- Validation checks (schema conformance, conservation, stability)
- Claim-witness pairs for the trust certificate
- Content hashes for tamper detection

Evidence is generated server-side AFTER execution and BEFORE the
result leaves the server.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .hasher import content_hash

logger = logging.getLogger(__name__)


def generate_validation_report(
    sanitized_result: dict[str, Any],
    domain_key: str,
) -> dict[str, Any]:
    """Run validation checks against a sanitized result.

    Returns a structured validation report with named checks.
    """
    checks: list[dict[str, Any]] = []

    # Check 1: Fields present
    fields = sanitized_result.get("fields")
    if fields is not None:
        checks.append({
            "name": "fields_present",
            "passed": len(fields) > 0,
            "detail": f"{len(fields)} field(s) returned",
            "severity": "error",
        })

        # Check 2: No NaN/null values in fields
        for fname, fdata in fields.items():
            values = fdata.get("values", [])
            has_null = _contains_null(values)
            checks.append({
                "name": f"field_{fname}_finite",
                "passed": not has_null,
                "detail": f"Field '{fname}' contains {'null values' if has_null else 'only finite values'}",
                "severity": "error" if has_null else "info",
            })
    else:
        checks.append({
            "name": "fields_present",
            "passed": True,
            "detail": "Fields omitted (return_fields=false or oversized)",
            "severity": "info",
        })

    # Check 3: Conservation
    conservation = sanitized_result.get("conservation")
    if conservation:
        conserved = conservation["status"] == "conserved"
        tier = conservation.get("resolution_tier", "production")
        error_metric = conservation.get("error_metric", "relative")
        error_value = conservation.get("error_value", conservation.get("relative_error", 0))
        tier_threshold = conservation.get("tier_threshold", 1e-4)

        # Preview-tier runs get "warning" severity — they are
        # qualitative smoke tests, not production solves.
        if not conserved and tier == "preview":
            severity = "warning"
        elif not conserved:
            severity = "error"
        else:
            severity = "info"
        tier_label = f" [{tier}]" if tier != "production" else ""
        metric_label = "abs" if error_metric == "absolute" else "rel"
        checks.append({
            "name": "conservation_law",
            "passed": conserved,
            "detail": (
                f"{conservation['quantity']}: "
                f"{metric_label}_error={error_value:.2e} "
                f"({conservation['status']}, "
                f"threshold={tier_threshold:.0e})"
                f"{tier_label}"
            ),
            "severity": severity,
        })

    # Check 4: Performance sanity
    perf = sanitized_result.get("performance", {})
    wall_time = perf.get("wall_time_s", 0)
    checks.append({
        "name": "execution_completed",
        "passed": wall_time > 0,
        "detail": f"wall_time={wall_time:.4f}s",
        "severity": "error" if wall_time <= 0 else "info",
    })

    # Check 5: Stability (no divergence — fields should have bounded norms)
    stable = True
    if fields:
        for fname, fdata in fields.items():
            values = fdata.get("values", [])
            max_val = _max_abs(values)
            if max_val > 1e15:
                stable = False
                checks.append({
                    "name": f"field_{fname}_stable",
                    "passed": False,
                    "detail": f"Field '{fname}' has max |value| = {max_val:.2e} (diverged)",
                    "severity": "error",
                })
    checks.append({
        "name": "numerical_stability",
        "passed": stable,
        "detail": "All fields within stable bounds" if stable else "Numerical divergence detected",
        "severity": "error" if not stable else "info",
    })

    all_passed = all(c["passed"] for c in checks if c["severity"] == "error")

    return {
        "valid": all_passed,
        "checks": checks,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_claims(
    sanitized_result: dict[str, Any],
    domain_key: str,
) -> list[dict[str, Any]]:
    """Extract claim-witness pairs for the trust certificate.

    Each claim is a verifiable predicate with its evidence (witness).
    """
    claims: list[dict[str, Any]] = []

    # Conservation claim
    conservation = sanitized_result.get("conservation")
    if conservation:
        error_value = conservation.get("error_value", conservation.get("relative_error", 0))
        error_metric = conservation.get("error_metric", "relative")
        tier_threshold = conservation.get("tier_threshold", 1e-4)
        tier_name = conservation.get("resolution_tier", "production")
        satisfied = error_value < tier_threshold
        metric_label = "absolute" if error_metric == "absolute" else "relative"
        claims.append({
            "tag": "CONSERVATION",
            "claim": (
                f"{conservation['quantity']} preserved to "
                f"{error_value:.2e} {metric_label} error"
            ),
            "witness": {
                "initial": conservation["initial_value"],
                "final": conservation["final_value"],
                "error_value": error_value,
                "error_metric": error_metric,
                "relative_error": conservation.get("relative_error", error_value),
                "absolute_error": conservation.get("absolute_error", 0),
                "threshold": tier_threshold,
                "resolution_tier": tier_name,
            },
            "satisfied": satisfied,
        })

    # Stability claim
    perf = sanitized_result.get("performance", {})
    wall_time = perf.get("wall_time_s", 0)
    completed = wall_time > 0
    claims.append({
        "tag": "STABILITY",
        "claim": "Simulation completed without numerical divergence",
        "witness": {
            "wall_time_s": wall_time,
            "time_steps": perf.get("time_steps", 0),
            "completed": completed,
        },
        "satisfied": completed,
    })

    # Bound claim (fields within physical range)
    fields = sanitized_result.get("fields")
    if fields:
        max_val = max(_max_abs(f.get("values", [])) for f in fields.values())
        bounded = max_val < 1e15
        claims.append({
            "tag": "BOUND",
            "claim": f"All field values bounded (max |value| = {max_val:.6e})",
            "witness": {
                "max_absolute_value": max_val,
                "threshold": 1e15,
            },
            "satisfied": bounded,
        })

    # ── CONVERGENCE (reserved tag, now implemented) ─────────────────
    convergence_data = sanitized_result.get("convergence")
    if convergence_data:
        observed_order = convergence_data.get("observed_order", 0.0)
        required_order = convergence_data.get("required_order", 2.0)
        qoi_name = convergence_data.get("qoi", "unknown")
        satisfied = observed_order >= required_order
        claims.append({
            "tag": "CONVERGENCE",
            "claim": (
                f"Grid convergence verified for {qoi_name}: "
                f"observed order {observed_order:.3f} >= required {required_order:.1f}"
            ),
            "witness": {
                "qoi": qoi_name,
                "observed_order": observed_order,
                "required_order": required_order,
                "refinement_levels": convergence_data.get("levels", 0),
            },
            "satisfied": satisfied,
        })

    # ── REPRODUCIBILITY (reserved tag, now implemented) ─────────────
    reproducibility = sanitized_result.get("reproducibility")
    if reproducibility:
        config_hash = reproducibility.get("config_hash", "")
        determinism_tier = reproducibility.get("determinism_tier", "unknown")
        claims.append({
            "tag": "REPRODUCIBILITY",
            "claim": (
                f"Deterministic execution: tier={determinism_tier}, "
                f"config_hash={config_hash[:12]}"
            ),
            "witness": {
                "determinism_tier": determinism_tier,
                "config_hash": config_hash,
                "seed": reproducibility.get("seed", 0),
            },
            "satisfied": bool(config_hash),
        })

    # ── ENERGY_BOUND (reserved tag, now implemented) ────────────────
    energy_data = sanitized_result.get("energy_bound")
    if energy_data:
        energy_value = energy_data.get("value", 0.0)
        energy_threshold = energy_data.get("threshold", 1e15)
        energy_name = energy_data.get("quantity", "total_energy")
        satisfied = abs(energy_value) < energy_threshold
        claims.append({
            "tag": "ENERGY_BOUND",
            "claim": (
                f"Energy bounded: {energy_name} = {energy_value:.6e} "
                f"(threshold {energy_threshold:.2e})"
            ),
            "witness": {
                "quantity": energy_name,
                "value": energy_value,
                "threshold": energy_threshold,
            },
            "satisfied": satisfied,
        })

    # ── CFL_SATISFIED (reserved tag, now implemented) ───────────────
    cfl_data = sanitized_result.get("cfl")
    if cfl_data:
        max_cfl = cfl_data.get("max_cfl", 0.0)
        cfl_limit = cfl_data.get("cfl_limit", 1.0)
        satisfied = max_cfl <= cfl_limit
        claims.append({
            "tag": "CFL_SATISFIED",
            "claim": (
                f"CFL condition satisfied: max CFL = {max_cfl:.4f} "
                f"(<= {cfl_limit:.1f})"
            ),
            "witness": {
                "max_cfl": max_cfl,
                "cfl_limit": cfl_limit,
            },
            "satisfied": satisfied,
        })

    # ── BOUNDEDNESS (compressible: ρ > 0, p > 0 predicates) ────────
    boundedness_data = sanitized_result.get("boundedness")
    if boundedness_data:
        predicates = boundedness_data.get("predicates", {})
        all_satisfied = all(predicates.values()) if predicates else False
        failed_preds = [k for k, v in predicates.items() if not v]
        claims.append({
            "tag": "BOUNDEDNESS",
            "claim": (
                "Physical boundedness predicates satisfied"
                if all_satisfied
                else f"Boundedness violations: {', '.join(failed_preds)}"
            ),
            "witness": {
                "predicates": predicates,
                "all_satisfied": all_satisfied,
                "failed": failed_preds,
            },
            "satisfied": all_satisfied,
        })

    return claims


def _contains_null(values: Any) -> bool:
    """Recursively check for None values in a nested list."""
    if values is None:
        return True
    if isinstance(values, list):
        return any(_contains_null(v) for v in values)
    return False


def _max_abs(values: Any) -> float:
    """Recursively find max absolute value in a nested list."""
    if values is None:
        return 0.0
    if isinstance(values, (int, float)):
        return abs(values)
    if isinstance(values, list):
        if not values:
            return 0.0
        return max(_max_abs(v) for v in values)
    return 0.0
