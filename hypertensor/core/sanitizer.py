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
    • Conservation diagnostics
    • Wall-clock time and throughput
    • Grid metadata
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


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
    execution_result: Any,  # tensornet.vm.runtime.ExecutionResult
    domain_key: str,
    precision: int = 8,
    max_field_points: int = 500_000,
    include_fields: bool = True,
    include_coordinates: bool = True,
) -> dict[str, Any]:
    """Sanitize an ExecutionResult into a public-safe dictionary.

    This function is the critical IP boundary.  NOTHING from the
    internal telemetry or tensor representation passes through
    except reconstructed physical values and conservation metrics.
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
        rel_err = abs(final - initial) / (abs(initial) + 1e-30)
        conservation = {
            "quantity": telemetry.invariant_name,
            "initial_value": _safe_float(initial, precision),
            "final_value": _safe_float(final, precision),
            "relative_error": float(f"{rel_err:.2e}"),
            "status": "conserved" if rel_err < 1e-4 else "drift",
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
