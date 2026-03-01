"""IP-safe output sanitizer.

This is the ONLY path from VM internals to the outside world.
It converts ``ExecutionResult`` → public-safe dictionaries,
stripping ALL tensor-train internal state.

Removed:
    • Bond dimensions (χ_max, χ_mean, χ_final)
    • Compression ratios
    • Singular value spectra
    • TT cores (raw tensor decomposition)
    • Rank evolution / saturation rate
    • Scaling classification (A/B/C/D)
    • IR opcodes and instruction details
    • Register counts
    • Internal class names and paths

Retained:
    • Reconstructed physical field values (dense arrays)
    • Conservation diagnostics (with resolution-aware grading)
    • Wall-clock time and throughput
    • Grid metadata
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ── Resolution tiers ────────────────────────────────────────────

# Conservation thresholds are resolution-dependent.  A 64×64
# preview run cannot be held to the same bar as a 1024×1024
# production solve.  The tiers below map n_bits (per dim)
# to a named tier and an appropriate relative-error threshold.

@dataclass(frozen=True)
class ResolutionTier:
    """Named resolution tier with conservation threshold."""

    name: str
    threshold: float
    description: str


# Tier boundaries: n_bits → tier
# The resolution advisor enforces tier-aware minimums:
#   quick=9 (512), standard=10 (1024), high=11 (2048), max=12 (4096)
# so tiers below "standard" should never appear in production.
# We still define "preview" for defensive backward-compatibility.
#
# preview   : n_bits ≤ 8   (grid ≤ 256)   — threshold 1e-1
# standard  : 9 ≤ n_bits ≤ 10 (512–1024)  — threshold 1e-3
# production: n_bits ≥ 11 (2048+)          — threshold 1e-5

_TIERS = {
    "preview": ResolutionTier(
        name="preview",
        threshold=1e-1,
        description="≤256 pts/dim — below QTT compression threshold",
    ),
    "standard": ResolutionTier(
        name="standard",
        threshold=1e-3,
        description="512–1024 pts/dim — engineering quality",
    ),
    "production": ResolutionTier(
        name="production",
        threshold=1e-5,
        description="≥2048 pts/dim — publication quality",
    ),
}


def classify_resolution(n_bits: int) -> ResolutionTier:
    """Return the resolution tier for a given bits-per-dim."""
    if n_bits <= 8:
        return _TIERS["preview"]
    if n_bits <= 10:
        return _TIERS["standard"]
    return _TIERS["production"]


# ── Field units (informational, not IP-sensitive) ───────────────────

_FIELD_UNITS: dict[str, dict[str, str]] = {
    "burgers": {"u": "m/s"},
    "maxwell": {"E": "V/m", "B": "T"},
    "maxwell_3d": {"Ex": "V/m", "Ey": "V/m", "Ez": "V/m",
                   "Bx": "T", "By": "T", "Bz": "T"},
    "schrodinger": {"psi_re": "", "psi_im": "", "V": "J"},
    "advection_diffusion": {"u": "kg/m³"},
    "vlasov_poisson": {"f": "s/m⁴"},
    "navier_stokes_2d": {"omega": "1/s", "psi": "m²/s"},
}


def _safe_float(val: float, precision: int = 8) -> Any:
    """Convert a float to a JSON-safe value.

    NaN → None, ±Inf → None.  Normal floats are rounded.
    """
    if math.isnan(val):
        return None
    if math.isinf(val):
        return None
    return round(val, precision)


def _dense_to_list(arr: NDArray, precision: int = 8) -> list[Any]:
    """Convert numpy array to rounded JSON-safe list."""
    rounded = np.round(arr.astype(np.float64), precision)
    # Replace NaN/Inf in the array
    rounded = np.where(np.isfinite(rounded), rounded, 0.0)
    return rounded.tolist()


def _build_coordinates(
    bits_per_dim: tuple[int, ...],
    domain_bounds: tuple[tuple[float, float], ...],
    precision: int = 8,
) -> dict[str, list[float]]:
    """Build named coordinate arrays from QTT dimension metadata."""
    dim_names = ["x", "y", "z", "v", "w"]
    coords: dict[str, list[float]] = {}
    for i, (bits, (lo, hi)) in enumerate(zip(bits_per_dim, domain_bounds)):
        n = 2 ** bits
        name = dim_names[i] if i < len(dim_names) else f"dim{i}"
        arr = np.linspace(lo, hi, n, endpoint=False)
        coords[name] = np.round(arr, precision).tolist()
    return coords


def sanitize_result(
    execution_result: Any,  # ontic.vm.runtime.ExecutionResult
    domain_key: str,
    precision: int = 8,
    max_field_points: int = 500_000,
    include_fields: bool = True,
    include_coordinates: bool = True,
    execution_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sanitize an ExecutionResult into a public-safe dictionary.

    This function is the critical IP boundary.  NOTHING from the
    internal telemetry or tensor representation passes through
    except reconstructed physical values and conservation metrics.

    Parameters
    ----------
    execution_result : ExecutionResult
        Raw result from the VM runtime.
    domain_key : str
        Physics domain identifier.
    precision : int
        Decimal places for rounding.
    max_field_points : int
        Skip field data if total grid exceeds this.
    include_fields : bool
        Whether to include dense field arrays.
    include_coordinates : bool
        Whether to include coordinate arrays.
    execution_context : dict, optional
        Resolution metadata for tier-aware conservation grading.
        Expected keys: ``n_bits`` (int), ``n_steps`` (int).
        If absent, defaults to production-tier thresholds.
    """
    telemetry = execution_result.telemetry
    fields_raw = execution_result.fields

    # ── Grid info ───────────────────────────────────────────────────
    first_field = next(iter(fields_raw.values()))
    bits_per_dim = first_field.bits_per_dim
    domain_bounds = first_field.domain
    n_dims = len(bits_per_dim)
    resolution = [2 ** b for b in bits_per_dim]
    total_points = math.prod(resolution)

    grid: dict[str, Any] = {
        "dimensions": n_dims,
        "resolution": resolution,
        "domain_bounds": [[lo, hi] for lo, hi in domain_bounds],
    }
    if include_coordinates:
        grid["coordinates"] = _build_coordinates(bits_per_dim, domain_bounds, precision)

    # ── Fields → dense (ONLY physical values) ───────────────────────
    fields: dict[str, Any] | None = None
    if include_fields and total_points <= max_field_points:
        unit_map = _FIELD_UNITS.get(domain_key, {})
        fields = {}
        for fname, qtt in fields_raw.items():
            dense = qtt.to_dense()
            shape = resolution if n_dims > 1 else [total_points]
            fields[fname] = {
                "name": fname,
                "shape": shape,
                "values": _dense_to_list(dense, precision),
                "unit": unit_map.get(fname, ""),
            }

    # ── Conservation (public-safe subset of telemetry) ──────────────
    conservation: dict[str, Any] | None = None
    if telemetry.invariant_name:
        initial = telemetry.invariant_initial
        final = telemetry.invariant_final

        # When the invariant is near-zero (e.g., ∫ω dA for symmetric
        # vortex IC), relative error is numerically meaningless.
        # Switch to absolute error in that regime.
        abs_err = abs(final - initial)
        abs_initial = abs(initial)

        # Threshold below which we consider the invariant "near-zero"
        # and use absolute error instead of relative error.
        _NEAR_ZERO = 1e-6

        if abs_initial > _NEAR_ZERO:
            # Normal case: meaningful initial value → relative error
            error_value = abs_err / abs_initial
            error_metric = "relative"
        else:
            # Near-zero invariant (e.g., ∫ω dA = 0 by symmetry)
            # Use absolute error directly.  The drift should stay
            # within machine-precision for truly conserved quantities.
            error_value = abs_err
            error_metric = "absolute"

        # Resolution-aware tier classification
        ctx_n_bits = (execution_context or {}).get("n_bits")
        if ctx_n_bits is not None:
            tier = classify_resolution(ctx_n_bits)
        else:
            # Infer from grid: bits_per_dim[0]
            tier = classify_resolution(bits_per_dim[0])

        # For absolute error on near-zero invariants, use a fixed
        # absolute threshold rather than the tier's relative one.
        if error_metric == "absolute":
            threshold = 1e-6  # absolute drift tolerance
        else:
            threshold = tier.threshold

        status = "conserved" if error_value < threshold else "drift"

        conservation = {
            "quantity": telemetry.invariant_name,
            "initial_value": _safe_float(initial, precision),
            "final_value": _safe_float(final, precision),
            "error_value": float(f"{error_value:.2e}"),
            "error_metric": error_metric,
            "relative_error": float(f"{error_value:.2e}") if error_metric == "relative" else float(f"{abs_err / (abs_initial + 1e-30):.2e}"),
            "absolute_error": float(f"{abs_err:.2e}"),
            "status": status,
            "resolution_tier": tier.name,
            "tier_threshold": threshold,
            "tier_description": tier.description,
        }

    # ── Performance (only wall time + throughput — no rank info) ────
    wall = telemetry.total_wall_time_s
    throughput = (total_points * telemetry.n_steps) / (wall + 1e-30)
    performance = {
        "wall_time_s": round(wall, 4),
        "grid_points": total_points,
        "time_steps": telemetry.n_steps,
        "throughput_gp_per_s": round(throughput, 1),
    }

    return {
        "grid": grid,
        "fields": fields,
        "conservation": conservation,
        "performance": performance,
    }
