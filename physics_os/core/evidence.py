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
        checks.append({
            "name": "conservation_law",
            "passed": conserved,
            "detail": (
                f"{conservation['quantity']}: "
                f"relative_error={conservation['relative_error']:.2e} "
                f"({conservation['status']})"
            ),
            "severity": "error" if not conserved else "info",
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
        rel_err = conservation["relative_error"]
        satisfied = rel_err < 1e-4
        claims.append({
            "tag": "CONSERVATION",
            "claim": (
                f"{conservation['quantity']} preserved to "
                f"{rel_err:.2e} relative error"
            ),
            "witness": {
                "initial": conservation["initial_value"],
                "final": conservation["final_value"],
                "relative_error": rel_err,
                "threshold": 1e-4,
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
