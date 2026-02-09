"""
Post-Processing API — probe, slice, integrate, FFT, statistics on fields.

Built on the ``FieldData`` / ``SimulationState`` data model.  All operations
return new ``FieldData`` objects or plain tensors — no in-place mutation.

Operations
----------
* **probe** — Interpolate a field at arbitrary points (nearest-cell for now,
  bilinear/trilinear in future).
* **slice_field** — Extract a hyper-plane slice from a structured field.
* **integrate** — Volume / surface / line integration of a field.
* **field_statistics** — Min, max, mean, std, percentiles.
* **fft_field** — FFT of a 1-D or N-D field (returns power spectrum or
  complex coefficients).
* **gradient_field** — Finite-difference gradient of a scalar field.
* **histogram** — Histogram of field values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from tensornet.platform.data_model import (
    FieldData,
    Mesh,
    SimulationState,
    StructuredMesh,
)

__all__ = [
    "probe",
    "slice_field",
    "integrate",
    "field_statistics",
    "fft_field",
    "gradient_field",
    "histogram",
    "FieldStats",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Probe — point interpolation
# ═══════════════════════════════════════════════════════════════════════════════


def probe(
    field: FieldData,
    points: Tensor,
) -> Tensor:
    """
    Sample a field at arbitrary spatial locations.

    Uses **nearest-cell** interpolation.  For structured meshes, maps the
    query point to the nearest cell index via floor division.

    Parameters
    ----------
    field : FieldData
    points : (n_query, ndim) — query coordinates.

    Returns
    -------
    Tensor of shape (n_query,) for scalar fields, (n_query, components) for
    vector fields.
    """
    mesh = field.mesh
    pts = points.to(torch.float64)
    if pts.ndim == 1:
        pts = pts.unsqueeze(0)

    if isinstance(mesh, StructuredMesh):
        return _probe_structured(field, mesh, pts)

    # Generic fallback: compute distances to all cell centers
    centers = mesh.cell_centers().to(torch.float64)
    dists = torch.cdist(pts, centers)  # (n_query, n_cells)
    nearest = dists.argmin(dim=1)  # (n_query,)
    data = field.data.to(torch.float64)
    return data[nearest]


def _probe_structured(
    field: FieldData,
    mesh: StructuredMesh,
    pts: Tensor,
) -> Tensor:
    """Nearest-cell probe for structured meshes."""
    ndim = mesh.ndim
    shape = mesh.shape
    data = field.data.to(torch.float64)

    # Map each query point to a flat cell index
    indices = torch.zeros(pts.shape[0], dtype=torch.long)
    for d in range(ndim):
        lo, hi = mesh.domain[d]
        dx = mesh.dx[d]
        # Cell index in dimension d
        coord = pts[:, d].clamp(lo, hi - 1e-15)
        idx_d = ((coord - lo) / dx).long().clamp(0, shape[d] - 1)
        # Multiply by stride for flat indexing
        stride = 1
        for dd in range(d + 1, ndim):
            stride *= shape[dd]
        indices = indices + idx_d * stride

    return data[indices]


# ═══════════════════════════════════════════════════════════════════════════════
# Slice — hyper-plane extraction
# ═══════════════════════════════════════════════════════════════════════════════


def slice_field(
    field: FieldData,
    axis: int,
    index: int,
) -> Tuple[Tensor, Tensor]:
    """
    Extract a slice from a structured field along a given axis.

    Parameters
    ----------
    field : FieldData on a StructuredMesh
    axis : dimension to slice (0, 1, 2)
    index : cell index along that axis

    Returns
    -------
    (coordinates, values) — the remaining-dimension coordinates and the
    field values on the slice.

    Raises
    ------
    TypeError
        If the field is not on a StructuredMesh.
    IndexError
        If axis or index is out of range.
    """
    mesh = field.mesh
    if not isinstance(mesh, StructuredMesh):
        raise TypeError("slice_field requires a StructuredMesh")
    if axis < 0 or axis >= mesh.ndim:
        raise IndexError(f"axis {axis} out of range for {mesh.ndim}-D mesh")
    if index < 0 or index >= mesh.shape[axis]:
        raise IndexError(
            f"index {index} out of range for axis {axis} (shape={mesh.shape[axis]})"
        )

    shape = mesh.shape
    data = field.data.to(torch.float64)

    # Reshape flat data to multi-dim
    data_nd = data.reshape(shape)
    sliced = data_nd.select(axis, index)

    # Build coordinates for the remaining dimensions
    remaining_dims = [d for d in range(mesh.ndim) if d != axis]
    coord_arrays = []
    for d in remaining_dims:
        lo, hi = mesh.domain[d]
        dx = mesh.dx[d]
        coord_arrays.append(
            torch.linspace(lo + 0.5 * dx, hi - 0.5 * dx, shape[d], dtype=torch.float64)
        )

    if len(coord_arrays) == 0:
        # 1-D field, slice gives a scalar
        coords = torch.tensor(
            [mesh.domain[axis][0] + (index + 0.5) * mesh.dx[axis]],
            dtype=torch.float64,
        )
        return coords, sliced.unsqueeze(0) if sliced.ndim == 0 else sliced

    if len(coord_arrays) == 1:
        return coord_arrays[0], sliced

    coords = torch.stack(
        torch.meshgrid(*coord_arrays, indexing="ij"), dim=-1
    )
    return coords, sliced


# ═══════════════════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════════════════


def integrate(
    field: FieldData,
    *,
    region: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute the volume integral of a field: ∫ f dV.

    For structured meshes, uses cell_volumes * field_values.
    For unstructured meshes, uses the mesh-provided cell_volumes().

    Parameters
    ----------
    field : FieldData
    region : optional boolean mask (n_cells,) to restrict integration.

    Returns
    -------
    Scalar tensor (or vector for vector fields).
    """
    mesh = field.mesh
    data = field.data.to(torch.float64)
    vols = mesh.cell_volumes().to(torch.float64)

    if region is not None:
        mask = region.to(torch.bool)
        data = data[mask]
        vols = vols[mask]

    if data.ndim == 1:
        return (data * vols).sum()
    else:
        # Vector field: integrate component-wise
        return (data * vols.unsqueeze(-1)).sum(dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Field Statistics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FieldStats:
    """Summary statistics for a field."""

    name: str
    min: float
    max: float
    mean: float
    std: float
    l2_norm: float
    percentiles: Dict[str, float]  # e.g. {"p05": ..., "p50": ..., "p95": ...}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "l2_norm": self.l2_norm,
            "percentiles": dict(self.percentiles),
        }


def field_statistics(
    field: FieldData,
    *,
    percentiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> FieldStats:
    """
    Compute descriptive statistics over a field.

    Parameters
    ----------
    field : FieldData
    percentiles : quantiles to compute (values in [0, 1]).

    Returns
    -------
    FieldStats
    """
    data = field.data.to(torch.float64).flatten()
    pctl_dict: Dict[str, float] = {}
    for p in percentiles:
        idx = int(round(p * (len(data) - 1)))
        sorted_data = data.sort().values
        pctl_dict[f"p{int(p * 100):02d}"] = sorted_data[idx].item()

    return FieldStats(
        name=field.name,
        min=data.min().item(),
        max=data.max().item(),
        mean=data.mean().item(),
        std=data.std().item(),
        l2_norm=data.norm(p=2).item(),
        percentiles=pctl_dict,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FFT
# ═══════════════════════════════════════════════════════════════════════════════


def fft_field(
    field: FieldData,
    *,
    return_power: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Compute the FFT of a field.

    Parameters
    ----------
    field : FieldData on a StructuredMesh (1-D or 2-D or 3-D).
    return_power : if True, return power spectrum |F(k)|^2; else complex coeffs.

    Returns
    -------
    (frequencies, spectrum) — wavenumber tensor and spectrum values.

    For 1-D: frequencies shape (N//2+1,), spectrum shape (N//2+1,).
    For N-D: frequencies is a list-tensor of per-axis frequencies.
    """
    mesh = field.mesh
    if not isinstance(mesh, StructuredMesh):
        raise TypeError("fft_field requires a StructuredMesh")

    data = field.data.to(torch.float64)
    shape = mesh.shape

    if mesh.ndim == 1:
        data_1d = data.reshape(shape[0])
        F = torch.fft.rfft(data_1d)
        n = shape[0]
        dx = mesh.dx[0]
        freqs = torch.fft.rfftfreq(n, d=dx)
        if return_power:
            return freqs, (F.abs() ** 2)
        return freqs, F

    # Multi-dimensional
    data_nd = data.reshape(shape)
    F = torch.fft.rfftn(data_nd)
    freq_axes: List[Tensor] = []
    for d in range(mesh.ndim):
        n = shape[d]
        dx = mesh.dx[d]
        if d == mesh.ndim - 1:
            freq_axes.append(torch.fft.rfftfreq(n, d=dx))
        else:
            freq_axes.append(torch.fft.fftfreq(n, d=dx))

    if return_power:
        return torch.stack(
            torch.meshgrid(*freq_axes, indexing="ij"), dim=-1
        ).reshape(-1, mesh.ndim), (F.abs() ** 2).reshape(-1)
    return torch.stack(
        torch.meshgrid(*freq_axes, indexing="ij"), dim=-1
    ).reshape(-1, mesh.ndim), F.reshape(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Gradient
# ═══════════════════════════════════════════════════════════════════════════════


def gradient_field(field: FieldData) -> FieldData:
    """
    Compute the finite-difference gradient of a scalar field on a
    ``StructuredMesh``.

    Uses second-order central differences in the interior and first-order
    one-sided differences at boundaries.

    Parameters
    ----------
    field : scalar FieldData on a StructuredMesh.

    Returns
    -------
    FieldData with ``components=ndim``, shape ``(n_cells, ndim)``.
    """
    mesh = field.mesh
    if not isinstance(mesh, StructuredMesh):
        raise TypeError("gradient_field requires a StructuredMesh")
    if field.components != 1:
        raise ValueError("gradient_field requires a scalar field")

    data = field.data.to(torch.float64).reshape(mesh.shape)
    ndim = mesh.ndim
    grad_components: List[Tensor] = []

    for d in range(ndim):
        dx = mesh.dx[d]
        # Use torch.gradient for central differences with boundary handling
        grads = torch.gradient(data, spacing=(dx,), dim=d)[0]
        grad_components.append(grads)

    grad = torch.stack(grad_components, dim=-1).reshape(-1, ndim)
    return FieldData(
        name=f"grad_{field.name}",
        data=grad,
        mesh=mesh,
        components=ndim,
        units=f"{field.units}/m",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Histogram
# ═══════════════════════════════════════════════════════════════════════════════


def histogram(
    field: FieldData,
    n_bins: int = 50,
    *,
    range_: Optional[Tuple[float, float]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute a histogram of field values.

    Parameters
    ----------
    field : FieldData
    n_bins : number of bins
    range_ : (min, max) range; default uses field min/max.

    Returns
    -------
    (bin_edges, counts) — bin edges shape (n_bins+1,), counts shape (n_bins,).
    """
    data = field.data.to(torch.float64).flatten()
    if range_ is not None:
        lo, hi = range_
    else:
        lo = data.min().item()
        hi = data.max().item()

    if lo == hi:
        hi = lo + 1.0

    bin_edges = torch.linspace(lo, hi, n_bins + 1, dtype=torch.float64)
    counts = torch.histc(data, bins=n_bins, min=lo, max=hi)
    return bin_edges, counts
