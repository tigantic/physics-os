"""Ontic API — IP-safe result serializer.

Converts ``ExecutionResult`` into API response models, stripping
ALL tensor-train internals:

  Removed:
    • Bond dimensions (χ_max, χ_mean, χ_final)
    • Compression ratios
    • Singular value spectra
    • TT cores (the raw tensor decomposition)
    • Rank evolution / saturation rate
    • Scaling classification (A/B/C/D)
    • IR opcodes and instruction details

  Retained:
    • Reconstructed physical field values (dense arrays)
    • Conservation diagnostics (quantity, initial, final, error)
    • Wall-clock time
    • Grid metadata (resolution, domain bounds)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import settings
from ..models import (
    ConservationDiagnostic,
    FieldData,
    GridInfo,
    PerformanceInfo,
    SimulationResponse,
)


# ── Field Units (informational) ────────────────────────────────────

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

_EQUATIONS: dict[str, str] = {
    "burgers": "∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²",
    "maxwell": "∂E/∂t = c·∂B/∂x,  ∂B/∂t = c·∂E/∂x",
    "maxwell_3d": "∂E/∂t = c·∇×B,  ∂B/∂t = −c·∇×E",
    "schrodinger": "i·∂ψ/∂t = −½·∂²ψ/∂x² + V(x)·ψ",
    "advection_diffusion": "∂u/∂t + v·∂u/∂x = κ·∂²u/∂x²",
    "vlasov_poisson": "∂f/∂t + v·∂f/∂x + E·∂f/∂v = 0,  ∂²φ/∂x² = −ρ",
    "navier_stokes_2d": "∂ω/∂t + (u·∇)ω = ν·∇²ω,  ∇²ψ = −ω",
}


def _round_list(arr: NDArray, precision: int) -> list[Any]:
    """Round and convert a numpy array to a JSON-serializable list.

    For 1-D arrays, returns a flat list.
    For N-D arrays, returns nested lists preserving structure.
    """
    rounded = np.round(arr, precision)
    if rounded.ndim == 1:
        return rounded.tolist()
    return rounded.tolist()


def _make_coordinates(
    bits_per_dim: tuple[int, ...],
    domain_bounds: tuple[tuple[float, float], ...],
    precision: int,
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


def serialize_result(
    *,
    job_id: str,
    result: Any,  # tensornet.vm.runtime.ExecutionResult
    domain: str,
    domain_label: str,
    params_echo: dict[str, Any],
    return_fields: bool,
    return_coordinates: bool,
) -> SimulationResponse:
    """Convert an ExecutionResult to an IP-safe SimulationResponse.

    This is the ONLY path from VM internals to the outside world.
    It deliberately discards all TT-specific information.
    """
    telemetry = result.telemetry
    precision = settings.field_precision

    # ── Grid info ───────────────────────────────────────────────────
    # Recover dimension metadata from the first field's QTTTensor
    first_field_name = next(iter(result.fields))
    first_field: Any = result.fields[first_field_name]  # QTTTensor

    bits_per_dim = first_field.bits_per_dim
    domain_bounds = first_field.domain
    n_dims = len(bits_per_dim)
    resolution_per_dim = [2 ** b for b in bits_per_dim]
    total_grid_points = math.prod(resolution_per_dim)

    coordinates = None
    if return_coordinates:
        coordinates = _make_coordinates(bits_per_dim, domain_bounds, precision)

    grid = GridInfo(
        dimensions=n_dims,
        resolution=resolution_per_dim,
        domain_bounds=[[lo, hi] for lo, hi in domain_bounds],
        coordinates=coordinates,
    )

    # ── Fields → dense arrays ───────────────────────────────────────
    fields_out: dict[str, FieldData] | None = None
    if return_fields:
        if total_grid_points > settings.max_field_points:
            # Too large for JSON response — skip field data
            fields_out = None
        else:
            fields_out = {}
            unit_map = _FIELD_UNITS.get(domain, {})
            for fname, qtt in result.fields.items():
                dense = qtt.to_dense()
                shape = resolution_per_dim if n_dims > 1 else [total_grid_points]
                values = _round_list(dense, precision)
                fields_out[fname] = FieldData(
                    name=fname,
                    shape=shape,
                    values=values,
                    unit=unit_map.get(fname, ""),
                )

    # ── Conservation diagnostic ─────────────────────────────────────
    conservation = None
    if telemetry.invariant_name:
        initial = telemetry.invariant_initial
        final = telemetry.invariant_final
        rel_err = abs(final - initial) / (abs(initial) + 1e-30)
        conservation = ConservationDiagnostic(
            quantity=telemetry.invariant_name,
            initial_value=round(initial, precision),
            final_value=round(final, precision),
            relative_error=float(f"{rel_err:.2e}"),
            status="conserved" if rel_err < 1e-4 else "drift",
        )

    # ── Performance ─────────────────────────────────────────────────
    wall = telemetry.total_wall_time_s
    throughput = (total_grid_points * telemetry.n_steps) / (wall + 1e-30)
    performance = PerformanceInfo(
        wall_time_s=round(wall, 4),
        grid_points=total_grid_points,
        time_steps=telemetry.n_steps,
        throughput_gp_per_s=round(throughput, 1),
    )

    return SimulationResponse(
        job_id=job_id,
        status="completed" if result.success else "failed",
        domain=domain,
        domain_label=domain_label,
        equation=_EQUATIONS.get(domain, ""),
        parameters=params_echo,
        grid=grid,
        fields=fields_out,
        conservation=conservation,
        performance=performance,
        error=result.error if not result.success else None,
    )
