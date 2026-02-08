"""
Unified Data Model — Mesh, Field, Boundary Conditions, Simulation State.

These are *concrete* dataclasses / lightweight containers, not protocols.
They are backend-agnostic: the ``data`` payloads are ``torch.Tensor`` today
and can be swapped for QTT cores once the substrate bridge is complete.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════════
# Mesh
# ═══════════════════════════════════════════════════════════════════════════════


class Mesh:
    """
    Base mesh interface.  Concrete subclasses provide grid-specific storage.

    Attributes
    ----------
    ndim : int
        Spatial dimensionality (1, 2, 3).
    n_cells : int
        Total number of cells / elements.
    """

    def __init__(self, ndim: int, n_cells: int) -> None:
        self._ndim = ndim
        self._n_cells = n_cells

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def n_cells(self) -> int:
        return self._n_cells

    def cell_volumes(self) -> Tensor:
        """Return a 1-D tensor of cell volumes / areas / lengths."""
        raise NotImplementedError

    def cell_centers(self) -> Tensor:
        """Return ``(n_cells, ndim)`` tensor of cell centroids."""
        raise NotImplementedError


class StructuredMesh(Mesh):
    """
    Logically-rectangular grid (1-D, 2-D, or 3-D).

    Parameters
    ----------
    shape : tuple of int
        Grid cells per dimension, e.g. ``(128, 128)`` for a 2-D grid.
    domain : tuple of (float, float)
        Per-dimension physical extent, e.g. ``((0., 1.), (0., 1.))``.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        domain: Tuple[Tuple[float, float], ...],
    ) -> None:
        if len(shape) != len(domain):
            raise ValueError(
                f"shape dims ({len(shape)}) != domain dims ({len(domain)})"
            )
        super().__init__(ndim=len(shape), n_cells=int(torch.prod(torch.tensor(shape)).item()))
        self.shape = shape
        self.domain = domain
        # Precompute dx per dimension
        self.dx: Tuple[float, ...] = tuple(
            (hi - lo) / n for (lo, hi), n in zip(domain, shape)
        )

    def cell_volumes(self) -> Tensor:
        vol = 1.0
        for d in self.dx:
            vol *= d
        return torch.full((self.n_cells,), vol, dtype=torch.float64)

    def cell_centers(self) -> Tensor:
        grids = [
            torch.linspace(lo + 0.5 * d, hi - 0.5 * d, n, dtype=torch.float64)
            for (lo, hi), n, d in zip(self.domain, self.shape, self.dx)
        ]
        coords = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1)
        return coords.reshape(-1, self.ndim)

    def __repr__(self) -> str:
        return (
            f"StructuredMesh(shape={self.shape}, domain={self.domain}, "
            f"dx={tuple(round(d, 8) for d in self.dx)})"
        )


class UnstructuredMesh(Mesh):
    """
    Unstructured mesh stored as nodes + connectivity.

    Parameters
    ----------
    nodes : Tensor
        ``(n_nodes, ndim)`` coordinates.
    elements : Tensor
        ``(n_elements, nodes_per_element)`` connectivity (0-based indices).
    """

    def __init__(self, nodes: Tensor, elements: Tensor) -> None:
        if nodes.ndim != 2:
            raise ValueError(f"nodes must be 2-D, got shape {nodes.shape}")
        if elements.ndim != 2:
            raise ValueError(f"elements must be 2-D, got shape {elements.shape}")
        super().__init__(ndim=nodes.shape[1], n_cells=elements.shape[0])
        self.nodes = nodes
        self.elements = elements

    def cell_centers(self) -> Tensor:
        verts = self.nodes[self.elements]  # (n_elem, vpn, ndim)
        return verts.mean(dim=1)

    def cell_volumes(self) -> Tensor:
        # Triangle 2-D fallback; subclass for higher-order
        if self.ndim == 2 and self.elements.shape[1] == 3:
            v = self.nodes[self.elements]  # (n, 3, 2)
            a = v[:, 1] - v[:, 0]
            b = v[:, 2] - v[:, 0]
            return 0.5 * torch.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        raise NotImplementedError(
            f"cell_volumes not implemented for ndim={self.ndim}, "
            f"nodes_per_element={self.elements.shape[1]}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FieldData
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FieldData:
    """
    A named field defined on a mesh.

    Parameters
    ----------
    name : str
        Field name (``'velocity'``, ``'pressure'``, …).
    data : Tensor
        Values — shape depends on field type:
        - scalar:  ``(n_cells,)``
        - vector:  ``(n_cells, ndim)``
        - tensor:  ``(n_cells, ndim, ndim)``
    mesh : Mesh
        The mesh this field lives on.
    components : int
        Number of components (1 = scalar, ndim = vector, …).
    units : str
        SI-compatible unit string (e.g. ``'m/s'``).
    """

    name: str
    data: Tensor
    mesh: Mesh
    components: int = 1
    units: str = "1"

    def __post_init__(self) -> None:
        if self.data.shape[0] != self.mesh.n_cells:
            raise ValueError(
                f"Field '{self.name}' length ({self.data.shape[0]}) "
                f"!= mesh n_cells ({self.mesh.n_cells})"
            )

    def norm(self, p: int = 2) -> Tensor:
        """L-p norm over the domain."""
        if self.components == 1:
            return torch.norm(self.data, p=p)
        return torch.norm(self.data.reshape(-1), p=p)

    def max(self) -> Tensor:
        return self.data.max()

    def min(self) -> Tensor:
        return self.data.min()

    def clone(self) -> "FieldData":
        return FieldData(
            name=self.name,
            data=self.data.clone(),
            mesh=self.mesh,
            components=self.components,
            units=self.units,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Boundary & Initial Conditions
# ═══════════════════════════════════════════════════════════════════════════════


class BCType(enum.Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    PERIODIC = "periodic"
    ABSORBING = "absorbing"
    SYMMETRY = "symmetry"


@dataclass(frozen=True)
class BoundaryCondition:
    """
    A boundary condition on a named field, applied to a named boundary region.
    """

    field_name: str
    region: str
    bc_type: BCType
    value: Union[float, Tensor, None] = None
    coefficients: Optional[Dict[str, float]] = None

    def apply(self, field: FieldData, mesh: Mesh) -> FieldData:
        """
        Return a new FieldData with the BC enforced.

        For production code at V0.4+, this dispatches to the discretization's
        BC applicator.  The default implementation handles 1-D structured grids.
        """
        if not isinstance(mesh, StructuredMesh) or mesh.ndim != 1:
            raise NotImplementedError(
                "Generic BC.apply only implemented for 1-D StructuredMesh. "
                "Use the discretization's BC applicator for higher dims."
            )
        data = field.data.clone()
        cells = mesh.shape[0]
        if self.bc_type == BCType.DIRICHLET:
            val = self.value if self.value is not None else 0.0
            if isinstance(val, (int, float)):
                val = torch.tensor(val, dtype=data.dtype)
            if self.region == "left":
                data[0] = val
            elif self.region == "right":
                data[cells - 1] = val
        elif self.bc_type == BCType.NEUMANN:
            grad = self.value if self.value is not None else 0.0
            dx = mesh.dx[0]
            if isinstance(grad, (int, float)):
                grad = torch.tensor(grad, dtype=data.dtype)
            if self.region == "left":
                data[0] = data[1] - grad * dx
            elif self.region == "right":
                data[cells - 1] = data[cells - 2] + grad * dx
        elif self.bc_type == BCType.PERIODIC:
            if self.region in ("left", "right"):
                data[0] = data[cells - 2]
                data[cells - 1] = data[1]
        return FieldData(
            name=field.name,
            data=data,
            mesh=mesh,
            components=field.components,
            units=field.units,
        )


@dataclass(frozen=True)
class InitialCondition:
    """
    Describes how to initialize a field.
    """

    field_name: str
    ic_type: str  # 'uniform', 'function', 'file', 'gaussian', …
    value: Union[float, Tensor, None] = None
    function: Optional[Any] = None  # Callable[[Tensor], Tensor]
    metadata: Dict[str, Any] = dc_field(default_factory=dict)

    def generate(self, mesh: Mesh) -> FieldData:
        """Create a FieldData on the given mesh."""
        centers = mesh.cell_centers()
        if self.ic_type == "uniform":
            val = self.value if self.value is not None else 0.0
            if isinstance(val, (int, float)):
                data = torch.full(
                    (mesh.n_cells,), val, dtype=torch.float64
                )
            else:
                data = val.expand(mesh.n_cells)
        elif self.ic_type == "function":
            if self.function is None:
                raise ValueError("ic_type='function' requires a callable")
            data = self.function(centers)
        elif self.ic_type == "gaussian":
            mu = self.metadata.get("mu", 0.0)
            sigma = self.metadata.get("sigma", 0.1)
            r2 = ((centers - mu) ** 2).sum(dim=-1)
            data = torch.exp(-r2 / (2.0 * sigma ** 2))
        else:
            raise ValueError(f"Unknown ic_type: {self.ic_type!r}")
        return FieldData(
            name=self.field_name,
            data=data,
            mesh=mesh,
            units=self.metadata.get("units", "1"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SimulationState
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SimulationState:
    """
    Immutable-ish snapshot of the simulation at a particular time.

    The Solver reads this, produces a new SimulationState at ``t + dt``.
    """

    t: float
    fields: Dict[str, FieldData]
    mesh: Mesh
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    step_index: int = 0

    def get_field(self, name: str) -> FieldData:
        if name not in self.fields:
            raise KeyError(
                f"Field '{name}' not in state. Available: {list(self.fields)}"
            )
        return self.fields[name]

    def with_fields(self, **updates: FieldData) -> "SimulationState":
        """Return a new state with some fields replaced."""
        new_fields = dict(self.fields)
        new_fields.update(updates)
        return SimulationState(
            t=self.t,
            fields=new_fields,
            mesh=self.mesh,
            metadata=dict(self.metadata),
            step_index=self.step_index,
        )

    def advance(
        self,
        dt: float,
        new_fields: Dict[str, FieldData],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> "SimulationState":
        """Return a new state at ``t + dt``."""
        md = dict(self.metadata)
        if extra_metadata:
            md.update(extra_metadata)
        return SimulationState(
            t=self.t + dt,
            fields=new_fields,
            mesh=self.mesh,
            metadata=md,
            step_index=self.step_index + 1,
        )

    def clone(self) -> "SimulationState":
        return SimulationState(
            t=self.t,
            fields={k: v.clone() for k, v in self.fields.items()},
            mesh=self.mesh,
            metadata=dict(self.metadata),
            step_index=self.step_index,
        )
