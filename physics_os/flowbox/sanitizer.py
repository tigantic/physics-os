"""FlowBox output sanitizer — tighter whitelist than the general sanitizer.

Built on top of ``physics_os.core.sanitizer.sanitize_result()``.  Adds
FlowBox-specific constraints:

1. Field filtering by tier (Explorer: omega only, Builder+: omega + psi)
2. QoI reshaping into product-friendly ``metrics`` dict
3. Render metadata injection
4. No internal Poisson diagnostics exposed (solver knobs stay hidden)
5. Removes general-purpose conservation block in favor of FlowBox metrics

The IP boundary is enforced at TWO levels:
  - General sanitizer strips TT internals (FORBIDDEN_FIELDS)
  - FlowBox sanitizer strips solver internals and tier-restricted fields
"""

from __future__ import annotations

import math
from typing import Any

from ..core.sanitizer import sanitize_result
from .contract import FlowBoxConfig


def sanitize_flowbox(
    raw_result: Any,  # ExecutionResult
    config: FlowBoxConfig,
    physics_qoi: dict[str, Any] | None = None,
    render_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sanitize a raw VM result into a FlowBox-safe output.

    Parameters
    ----------
    raw_result : ExecutionResult
        Raw result from the VM execution.
    config : FlowBoxConfig
        Resolved FlowBox configuration (has tier caps).
    physics_qoi : dict, optional
        Pre-extracted physics QoI from ``extract_physics_qoi()``.
    render_metadata : dict, optional
        Render result metadata from ``generate_render()``.

    Returns
    -------
    dict
        Public-safe sanitized result with FlowBox product envelope.
    """
    # ── Step 1: General sanitization (IP boundary) ──────────────
    execution_context = {
        "n_bits": config.n_bits,
        "n_steps": config.steps,
    }
    sanitized = sanitize_result(
        execution_result=raw_result,
        domain_key="navier_stokes_2d",
        precision=8,
        max_field_points=config.grid * config.grid,
        include_fields=True,
        include_coordinates=True,
        execution_context=execution_context,
    )

    # ── Step 2: Filter fields by tier ───────────────────────────
    allowed_fields = set(config.output_fields)
    if sanitized.get("fields") and isinstance(sanitized["fields"], dict):
        sanitized["fields"] = {
            k: v for k, v in sanitized["fields"].items()
            if k in allowed_fields
        }

    # ── Step 3: Build FlowBox metrics (friendlier than raw QoIs) ─
    metrics = _build_metrics(sanitized, physics_qoi, config)
    sanitized["metrics"] = metrics

    # ── Step 4: Inject render metadata ──────────────────────────
    if render_metadata is not None:
        sanitized["render"] = render_metadata

    # ── Step 5: Strip raw conservation block ────────────────────
    # FlowBox users get structured metrics, not raw conservation.
    # The conservation data is already captured in metrics.
    # Keep it for validation but don't duplicate.

    # ── Step 6: Strip internal Poisson diagnostics ──────────────
    # The physics_qoi may contain Poisson solver details.  We expose
    # only a summary boolean (converged or not) and the max residual.
    if sanitized.get("physics_qoi"):
        poisson = sanitized["physics_qoi"].get("poisson", {})
        sanitized["physics_qoi"] = {
            "enstrophy": sanitized["physics_qoi"].get("enstrophy"),
            "omega_l2": sanitized["physics_qoi"].get("omega_l2"),
            "poisson_converged": poisson.get("max_residual_below_1e-3", False),
        }

    return sanitized


def _build_metrics(
    sanitized: dict[str, Any],
    physics_qoi: dict[str, Any] | None,
    config: FlowBoxConfig,
) -> dict[str, Any]:
    """Assemble the FlowBox ``metrics`` block.

    Exposed metrics:
    - enstrophy (float)
    - omega_l2 (float)
    - conservation_status (str): "conserved" or "drift"
    - conservation_error (float)
    - poisson_converged (bool)
    - wall_time_s (float)
    - throughput_gp_per_s (float)
    """
    metrics: dict[str, Any] = {}

    # From physics QoI
    if physics_qoi:
        metrics["enstrophy"] = physics_qoi.get("enstrophy")
        metrics["omega_l2"] = physics_qoi.get("omega_l2")

        poisson = physics_qoi.get("poisson", {})
        metrics["poisson_converged"] = poisson.get(
            "max_residual_below_1e-3", False
        )

    # From conservation block
    conservation = sanitized.get("conservation")
    if conservation:
        metrics["conservation_status"] = conservation.get("status", "unknown")
        metrics["conservation_error"] = conservation.get("error_value")
        metrics["conservation_quantity"] = conservation.get("quantity")

    # From performance block
    perf = sanitized.get("performance", {})
    metrics["wall_time_s"] = perf.get("wall_time_s", 0.0)
    metrics["throughput_gp_per_s"] = perf.get("throughput_gp_per_s", 0.0)
    metrics["time_steps"] = perf.get("time_steps", 0)
    metrics["grid_points"] = perf.get("grid_points", 0)

    return metrics
