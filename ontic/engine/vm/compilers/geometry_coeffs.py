"""QTT Physics VM — Geometry Coefficient Compiler.

Compiles geometry descriptions into QTT-native coefficient fields that
replace traditional meshing.  The resulting fields live entirely in
TT-core format and are consumed by the VM via MASK_MULTIPLY and
Hadamard operations.

Produced coefficient fields
---------------------------
- **Solid mask** ``χ_solid(x)`` — indicator field (1 inside solid, 0 in fluid).
- **Penalization strength** ``β(x)`` — penalization coefficient for immersed
  boundary methods (large in solid, zero in fluid).
- **Distance proxy** ``φ(x)`` — signed-distance-like field for wall models.
  Exact signed distance is typically not TT-friendly, so we use a smooth
  approximation that preserves the key property: ``|∇φ| ≈ 1`` near walls.
- **Material coefficient** ``a(x)`` — spatially varying material property
  (e.g., conductivity, permeability).

IP Boundary Compliance
----------------------
All geometry fields are QTT tensors — they are **internal state**.
The TT cores, bond dimensions, and rank statistics of geometry fields
are subject to the same IP boundary rules as any other QTT tensor:
they NEVER leak through the sanitizer (§20.4).

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..qtt_tensor import QTTTensor
from ..ir import (
    FieldSpec,
    BCKind,
    Instruction,
    OpCode,
    load_field,
    store_field,
    mask_multiply,
    hadamard,
    scale,
)


# ══════════════════════════════════════════════════════════════════════
# Geometry primitives
# ══════════════════════════════════════════════════════════════════════

class GeometryPrimitive(enum.Enum):
    """Supported procedural geometry types."""
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    SPHERE = "sphere"
    CHANNEL = "channel"
    STEP = "backward_facing_step"
    CUSTOM = "custom"


@dataclass(frozen=True)
class GeometrySpec:
    """Specification for a single geometry object.

    Parameters
    ----------
    primitive : GeometryPrimitive
        Shape type.
    params : dict[str, Any]
        Shape-specific parameters (center, radius, extents, etc.).
    is_solid : bool
        True = solid (mask = 1 inside), False = void/fluid.
    material_id : int
        Material identifier for CHT / multi-material cases.
    """
    primitive: GeometryPrimitive
    params: dict[str, Any] = field(default_factory=dict)
    is_solid: bool = True
    material_id: int = 0


@dataclass(frozen=True)
class GeometryScene:
    """A collection of geometry objects defining the computational domain.

    Parameters
    ----------
    objects : list[GeometrySpec]
        Geometry primitives in the scene.
    bits_per_dim : tuple[int, ...]
        QTT resolution per dimension.
    domain : tuple[tuple[float, float], ...]
        Physical domain bounds.
    penalization_strength : float
        Default β value inside solid regions.
    wall_band_width : float
        Width of the near-wall band (in physical units) for the
        distance proxy field.
    """
    objects: list[GeometrySpec]
    bits_per_dim: tuple[int, ...]
    domain: tuple[tuple[float, float], ...]
    penalization_strength: float = 1e6
    wall_band_width: float = 0.05


# ══════════════════════════════════════════════════════════════════════
# Indicator functions for each primitive
# ══════════════════════════════════════════════════════════════════════

def _rectangle_indicator(
    params: dict[str, Any],
) -> Callable[..., NDArray]:
    """Rectangle/box indicator: 1 inside, 0 outside.

    Parameters: center, half_extents
    """
    center = np.array(params["center"], dtype=np.float64)
    half = np.array(params["half_extents"], dtype=np.float64)

    def indicator(*coords: NDArray) -> NDArray:
        inside = np.ones_like(coords[0], dtype=np.float64)
        for d in range(len(coords)):
            inside *= (
                (coords[d] >= center[d] - half[d]) &
                (coords[d] <= center[d] + half[d])
            ).astype(np.float64)
        return inside

    return indicator


def _circle_indicator(
    params: dict[str, Any],
) -> Callable[..., NDArray]:
    """Circle/sphere indicator.

    Parameters: center, radius
    """
    center = np.array(params["center"], dtype=np.float64)
    radius = float(params["radius"])

    def indicator(*coords: NDArray) -> NDArray:
        r_sq = sum(
            (coords[d] - center[d]) ** 2
            for d in range(len(coords))
        )
        return (r_sq <= radius ** 2).astype(np.float64)

    return indicator


def _channel_indicator(
    params: dict[str, Any],
) -> Callable[..., NDArray]:
    """Channel flow indicator: fluid between two walls.

    Parameters: wall_y_lo, wall_y_hi (solid below/above these y-values)
    """
    y_lo = float(params["wall_y_lo"])
    y_hi = float(params["wall_y_hi"])

    def indicator(*coords: NDArray) -> NDArray:
        # Solid outside [y_lo, y_hi]
        y = coords[1] if len(coords) > 1 else coords[0]
        return ((y < y_lo) | (y > y_hi)).astype(np.float64)

    return indicator


def _step_indicator(
    params: dict[str, Any],
) -> Callable[..., NDArray]:
    """Backward-facing step indicator.

    Parameters: step_x, step_y (step corner position)
    """
    step_x = float(params["step_x"])
    step_y = float(params["step_y"])

    def indicator(*coords: NDArray) -> NDArray:
        x = coords[0]
        y = coords[1] if len(coords) > 1 else np.zeros_like(x)
        return ((x <= step_x) & (y <= step_y)).astype(np.float64)

    return indicator


_INDICATOR_REGISTRY: dict[
    GeometryPrimitive,
    Callable[[dict[str, Any]], Callable[..., NDArray]],
] = {
    GeometryPrimitive.RECTANGLE: _rectangle_indicator,
    GeometryPrimitive.CIRCLE: _circle_indicator,
    GeometryPrimitive.SPHERE: _circle_indicator,  # same math
    GeometryPrimitive.CHANNEL: _channel_indicator,
    GeometryPrimitive.STEP: _step_indicator,
}


# ══════════════════════════════════════════════════════════════════════
# Smooth distance proxy
# ══════════════════════════════════════════════════════════════════════

def _smooth_distance_proxy(
    indicator_fn: Callable[..., NDArray],
    band_width: float,
) -> Callable[..., NDArray]:
    """Build a smooth approximation of the signed distance field.

    Uses a tanh-smoothed version of the indicator to produce
    φ(x) ∈ [-1, 1] where:
      φ < 0 inside solid
      φ > 0 outside solid
      |∇φ| ≈ 1/band_width near the interface

    This is TT-friendly because tanh of a smooth function stays smooth.
    """
    eps = band_width / 4.0  # smoothing length

    def distance_proxy(*coords: NDArray) -> NDArray:
        chi = indicator_fn(*coords)
        # Convert indicator to signed: 1 → -1 (inside), 0 → +1 (outside)
        raw = 1.0 - 2.0 * chi
        # Smooth with tanh
        return np.tanh(raw / max(eps, 1e-12))

    return distance_proxy


# ══════════════════════════════════════════════════════════════════════
# GeometryCompiler: scene → QTT coefficient fields
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CompiledGeometry:
    """Result of geometry compilation: QTT-native coefficient fields.

    All fields are QTTTensor instances — fully compressed in TT format.
    Their TT cores are internal state and MUST NOT leak through the
    sanitizer.

    Attributes
    ----------
    solid_mask : QTTTensor
        χ_solid(x): 1 inside solid, 0 in fluid.
    penalization : QTTTensor
        β(x): penalization strength field (β × χ_solid).
    distance_proxy : QTTTensor
        φ(x): smooth signed-distance-like field.
    material_coeff : QTTTensor | None
        a(x): material coefficient field (if multi-material).
    field_specs : dict[str, FieldSpec]
        IR-compatible field specifications for the compiled fields.
    rank_stats : dict[str, int]
        Internal rank statistics (PRIVATE — forbidden externally).
    """
    solid_mask: QTTTensor
    penalization: QTTTensor
    distance_proxy: QTTTensor
    material_coeff: QTTTensor | None = None
    field_specs: dict[str, FieldSpec] = field(default_factory=dict)
    rank_stats: dict[str, int] = field(default_factory=dict)


class GeometryCompiler:
    """Compiles a GeometryScene into QTT coefficient fields.

    The compiler:
    1. Builds indicator functions for each geometry primitive
    2. Combines them into a unified solid mask
    3. Derives penalization, distance, and material fields
    4. Compresses all fields to QTT format

    No meshing involved — pure QTT-native geometry compilation.
    """

    def __init__(
        self,
        max_rank: int = 64,
        cutoff: float = 1e-12,
    ) -> None:
        self._max_rank = max_rank
        self._cutoff = cutoff

    def compile(self, scene: GeometryScene) -> CompiledGeometry:
        """Compile a geometry scene into QTT coefficient fields.

        Parameters
        ----------
        scene : GeometryScene
            The geometry description to compile.

        Returns
        -------
        CompiledGeometry
            QTT-native coefficient fields ready for VM consumption.
        """
        bpd = scene.bits_per_dim
        dom = scene.domain
        n_dims = len(bpd)

        # Step 1: Build combined indicator function
        indicators: list[Callable[..., NDArray]] = []
        for obj in scene.objects:
            if obj.primitive == GeometryPrimitive.CUSTOM:
                if "indicator_fn" not in obj.params:
                    raise ValueError(
                        "CUSTOM geometry requires 'indicator_fn' in params"
                    )
                indicators.append(obj.params["indicator_fn"])
            else:
                factory = _INDICATOR_REGISTRY.get(obj.primitive)
                if factory is None:
                    raise ValueError(
                        f"Unsupported geometry primitive: {obj.primitive}"
                    )
                indicators.append(factory(obj.params))

        def combined_indicator(*coords: NDArray) -> NDArray:
            result = np.zeros_like(coords[0], dtype=np.float64)
            for ind_fn in indicators:
                result = np.maximum(result, ind_fn(*coords))
            return result

        # Step 2: Compile to QTT
        solid_mask = QTTTensor.from_function(
            combined_indicator,
            bits_per_dim=bpd,
            domain=dom,
            max_rank=self._max_rank,
            cutoff=self._cutoff,
        )

        # Step 3: Penalization field = β × χ_solid
        beta = scene.penalization_strength

        def penalization_fn(*coords: NDArray) -> NDArray:
            return beta * combined_indicator(*coords)

        penalization = QTTTensor.from_function(
            penalization_fn,
            bits_per_dim=bpd,
            domain=dom,
            max_rank=self._max_rank,
            cutoff=self._cutoff,
        )

        # Step 4: Distance proxy
        dist_fn = _smooth_distance_proxy(
            combined_indicator, scene.wall_band_width,
        )
        distance_proxy = QTTTensor.from_function(
            dist_fn,
            bits_per_dim=bpd,
            domain=dom,
            max_rank=self._max_rank,
            cutoff=self._cutoff,
        )

        # Step 5: Material coefficient (if distinct materials exist)
        material_ids = {obj.material_id for obj in scene.objects}
        material_coeff: QTTTensor | None = None
        if len(material_ids) > 1:
            # Build material coefficient as weighted sum of indicators
            def material_fn(*coords: NDArray) -> NDArray:
                result = np.ones_like(coords[0], dtype=np.float64)
                for obj in scene.objects:
                    factory = _INDICATOR_REGISTRY.get(obj.primitive)
                    if factory is not None:
                        ind = factory(obj.params)(*coords)
                        result += float(obj.material_id) * ind
                return result

            material_coeff = QTTTensor.from_function(
                material_fn,
                bits_per_dim=bpd,
                domain=dom,
                max_rank=self._max_rank,
                cutoff=self._cutoff,
            )

        # Step 6: Collect rank statistics (PRIVATE — never leaves VM)
        rank_stats = {
            "solid_mask_max_rank": solid_mask.max_rank,
            "penalization_max_rank": penalization.max_rank,
            "distance_proxy_max_rank": distance_proxy.max_rank,
        }
        if material_coeff is not None:
            rank_stats["material_coeff_max_rank"] = material_coeff.max_rank

        # Step 7: Build IR-compatible field specs
        field_specs = {
            "chi_solid": FieldSpec(
                name="chi_solid",
                n_dims=n_dims,
                bits_per_dim=bpd,
                bc=BCKind.DIRICHLET,
                bc_params={"value": 0.0},
                conserved_quantity=None,
            ),
            "beta_penalization": FieldSpec(
                name="beta_penalization",
                n_dims=n_dims,
                bits_per_dim=bpd,
                bc=BCKind.DIRICHLET,
                bc_params={"value": 0.0},
                conserved_quantity=None,
            ),
            "phi_distance": FieldSpec(
                name="phi_distance",
                n_dims=n_dims,
                bits_per_dim=bpd,
                bc=BCKind.NEUMANN,
                bc_params={},
                conserved_quantity=None,
            ),
        }

        return CompiledGeometry(
            solid_mask=solid_mask,
            penalization=penalization,
            distance_proxy=distance_proxy,
            material_coeff=material_coeff,
            field_specs=field_specs,
            rank_stats=rank_stats,
        )


# ══════════════════════════════════════════════════════════════════════
# Convenience: compile common geometry patterns
# ══════════════════════════════════════════════════════════════════════

def compile_cylinder_in_channel(
    bits_per_dim: tuple[int, int] = (8, 8),
    domain: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 2.2), (0.0, 0.41),
    ),
    center: tuple[float, float] = (0.2, 0.2),
    radius: float = 0.05,
    penalization_strength: float = 1e6,
    max_rank: int = 64,
) -> CompiledGeometry:
    """Compile a cylinder-in-channel geometry (2D flow around cylinder).

    This is a standard CFD benchmark geometry (DFG benchmark 2D-1/2D-2).
    """
    scene = GeometryScene(
        objects=[
            GeometrySpec(
                primitive=GeometryPrimitive.CIRCLE,
                params={"center": list(center), "radius": radius},
                is_solid=True,
            ),
        ],
        bits_per_dim=bits_per_dim,
        domain=domain,
        penalization_strength=penalization_strength,
    )
    compiler = GeometryCompiler(max_rank=max_rank)
    return compiler.compile(scene)


def compile_backward_facing_step(
    bits_per_dim: tuple[int, int] = (8, 8),
    domain: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 10.0), (0.0, 2.0),
    ),
    step_x: float = 2.0,
    step_y: float = 1.0,
    penalization_strength: float = 1e6,
    max_rank: int = 64,
) -> CompiledGeometry:
    """Compile a backward-facing step geometry.

    Standard test case for separated flow and reattachment.
    """
    scene = GeometryScene(
        objects=[
            GeometrySpec(
                primitive=GeometryPrimitive.STEP,
                params={"step_x": step_x, "step_y": step_y},
                is_solid=True,
            ),
        ],
        bits_per_dim=bits_per_dim,
        domain=domain,
        penalization_strength=penalization_strength,
    )
    compiler = GeometryCompiler(max_rank=max_rank)
    return compiler.compile(scene)


def compile_lid_driven_cavity(
    bits_per_dim: tuple[int, int] = (8, 8),
    domain: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 1.0), (0.0, 1.0),
    ),
    max_rank: int = 64,
) -> CompiledGeometry:
    """Compile a lid-driven cavity geometry (no internal obstacles).

    The cavity itself is pure fluid — the geometry compiler produces
    a trivially zero solid mask, but the penalization and distance
    fields capture the boundary layer near the walls.
    """
    # For lid-driven cavity, the walls are the domain boundary itself.
    # We compile a thin wall band at the boundary for wall-model support.
    wall_thickness = 0.01 * (domain[0][1] - domain[0][0])

    def wall_indicator(x: NDArray, y: NDArray) -> NDArray:
        near_left = (x < domain[0][0] + wall_thickness).astype(np.float64)
        near_right = (x > domain[0][1] - wall_thickness).astype(np.float64)
        near_bottom = (y < domain[1][0] + wall_thickness).astype(np.float64)
        near_top = (y > domain[1][1] - wall_thickness).astype(np.float64)
        return np.maximum(
            np.maximum(near_left, near_right),
            np.maximum(near_bottom, near_top),
        )

    scene = GeometryScene(
        objects=[
            GeometrySpec(
                primitive=GeometryPrimitive.CUSTOM,
                params={"indicator_fn": wall_indicator},
                is_solid=True,
            ),
        ],
        bits_per_dim=bits_per_dim,
        domain=domain,
        penalization_strength=1e6,
        wall_band_width=wall_thickness * 2,
    )
    compiler = GeometryCompiler(max_rank=max_rank)
    return compiler.compile(scene)
