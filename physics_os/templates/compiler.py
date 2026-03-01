"""Problem compiler — translates ProblemSpec into ProblemResult.

The compiler is the central orchestrator:

1. Validates the spec against the registry constraints
2. Looks up fluid/material properties
3. Computes dimensionless numbers (Re, Ma, Pr, …)
4. Invokes the resolution advisor
5. Maps the problem class to the correct QTT domain + parameters
6. Builds the ``ProblemResult`` ready for the executor
"""

from __future__ import annotations

import math
from typing import Any

from physics_os.core.dimensionless import (
    classify_flow,
    mach_number,
    reynolds_number,
)
from physics_os.core.fluids import FluidProperties
from physics_os.core.resolution import QualityTier, advise
from physics_os.templates.models import (
    GeometryType,
    ProblemClass,
    ProblemResult,
    ProblemSpec,
)
from physics_os.templates.registry import TemplateRegistry


# ═══════════════════════════════════════════════════════════════════
# Geometry → characteristic length
# ═══════════════════════════════════════════════════════════════════

_CHAR_LENGTH_EXTRACTORS: dict[GeometryType, str] = {
    GeometryType.CIRCLE: "radius",           # L = 2r (diameter)
    GeometryType.ELLIPSE: "semi_b",          # L = 2b (frontal)
    GeometryType.RECTANGLE: "height",        # L = height
    GeometryType.ROUNDED_RECTANGLE: "height",
    GeometryType.WEDGE: "length",
    GeometryType.NACA_AIRFOIL: "chord",
    GeometryType.FLAT_PLATE: "length",
    GeometryType.FIN_ARRAY: "fin_height",
    GeometryType.PIPE_BEND: "gap_width",
    GeometryType.ANNULUS: "gap_width",
    GeometryType.BACKWARD_STEP: "step_height",
    GeometryType.NONE: "_fallback",
}


def _extract_char_length(spec: ProblemSpec) -> float:
    """Extract characteristic length from geometry params."""
    shape = spec.geometry.shape
    params = spec.geometry.params

    if shape == GeometryType.NONE:
        # Use domain_multiplier reciprocal as fallback
        return 0.1

    key = _CHAR_LENGTH_EXTRACTORS.get(shape, "_fallback")
    if key == "_fallback":
        return 0.1

    if shape == GeometryType.CIRCLE:
        return 2.0 * params.get("radius", 0.01)
    if shape == GeometryType.ELLIPSE:
        return 2.0 * params.get("semi_b", 0.01)
    if shape in (GeometryType.PIPE_BEND, GeometryType.ANNULUS):
        return params.get("gap_width", params.get("outer_radius", 0.02) - params.get("inner_radius", 0.01))

    return params.get(key, 0.1)


# ═══════════════════════════════════════════════════════════════════
# Problem class → domain mapping
# ═══════════════════════════════════════════════════════════════════

def _map_domain(problem_class: ProblemClass) -> str:
    """Map a problem class to the QTT solver domain key."""
    mapping = {
        ProblemClass.EXTERNAL_FLOW:     "navier_stokes_2d",
        ProblemClass.INTERNAL_FLOW:     "navier_stokes_2d",
        ProblemClass.HEAT_TRANSFER:     "advection_diffusion",
        ProblemClass.WAVE_PROPAGATION:  "maxwell",
        ProblemClass.NATURAL_CONVECTION: "navier_stokes_2d",
        ProblemClass.BOUNDARY_LAYER:    "navier_stokes_2d",
        ProblemClass.VORTEX_DYNAMICS:   "navier_stokes_2d",
        ProblemClass.CHANNEL_FLOW:      "navier_stokes_2d",
    }
    return mapping[problem_class]


# ═══════════════════════════════════════════════════════════════════
# Domain-parameter builders
# ═══════════════════════════════════════════════════════════════════

def _build_ns2d_params(
    spec: ProblemSpec,
    re: float,
    char_len: float,
    fluid: FluidProperties,
) -> dict[str, Any]:
    """Build parameters for the navier_stokes_2d domain."""
    return {
        "reynolds_number": re,
        "viscosity": fluid.kinematic_viscosity,
    }


def _build_advdiff_params(
    spec: ProblemSpec,
    re: float,
    char_len: float,
    fluid: FluidProperties,
) -> dict[str, Any]:
    """Build parameters for advection_diffusion domain (heat transfer)."""
    return {
        "velocity": spec.flow.velocity,
        "diffusivity": fluid.thermal_conductivity / (fluid.density * fluid.specific_heat_cp),
    }


def _build_maxwell_params(
    spec: ProblemSpec,
    re: float,
    char_len: float,
    fluid: FluidProperties,
) -> dict[str, Any]:
    """Build parameters for maxwell domain (wave propagation)."""
    return {
        "c": spec.flow.velocity,  # wave speed
    }


_PARAM_BUILDERS = {
    "navier_stokes_2d": _build_ns2d_params,
    "advection_diffusion": _build_advdiff_params,
    "maxwell": _build_maxwell_params,
}


# ═══════════════════════════════════════════════════════════════════
# Main compiler entry-point
# ═══════════════════════════════════════════════════════════════════

def compile_problem(spec: ProblemSpec) -> ProblemResult:
    """Compile a ``ProblemSpec`` into a ``ProblemResult``.

    Parameters
    ----------
    spec : ProblemSpec
        Validated client input.

    Returns
    -------
    ProblemResult
        Ready-to-execute simulation configuration with physics context.

    Raises
    ------
    KeyError
        If the fluid is not in the registry.
    ValueError
        If geometry type is not supported for the problem class.
    """
    registry = TemplateRegistry()
    template = registry.get(spec.problem_class)
    warnings: list[str] = []

    # ── Validate geometry ─────────────────────────────────────────
    if (
        spec.geometry.shape != GeometryType.NONE
        and spec.geometry.shape not in template.supported_geometries
    ):
        raise ValueError(
            f"Geometry '{spec.geometry.shape.value}' is not supported for "
            f"'{spec.problem_class.value}'. Supported: "
            f"{[g.value for g in template.supported_geometries]}"
        )

    # ── Look up fluid ─────────────────────────────────────────────
    fluid = FluidProperties.get(spec.flow.fluid)

    # ── Characteristic length ─────────────────────────────────────
    char_len = _extract_char_length(spec)

    # ── Dimensionless numbers ─────────────────────────────────────
    re = reynolds_number(spec.flow.velocity, char_len, fluid.kinematic_viscosity)
    ma = mach_number(spec.flow.velocity, fluid.speed_of_sound)
    regime = classify_flow(re, ma)

    if "turbulent" in regime.label and re > 1e6:
        warnings.append(
            f"High Reynolds number (Re={re:.2e}). "
            f"QTT solution represents a filtered (LES-like) result."
        )

    # ── Quality tier ──────────────────────────────────────────────
    tier = QualityTier(spec.quality)

    # ── Resolution advisor ────────────────────────────────────────
    domain_length = spec.domain_multiplier * char_len
    if spec.t_end is not None:
        t_end = spec.t_end
    else:
        # Auto: 5 flow-through times
        t_end = 5.0 * domain_length / spec.flow.velocity

    resolution = advise(
        re=re,
        characteristic_length=char_len,
        velocity=spec.flow.velocity,
        domain_length=domain_length,
        t_end=t_end,
        spatial_dims=2,
        speed_of_sound=fluid.speed_of_sound,
        tier=tier,
    )
    warnings.extend(resolution.warnings)

    # ── Domain + parameters ───────────────────────────────────────
    domain_key = _map_domain(spec.problem_class)
    builder = _PARAM_BUILDERS.get(domain_key, _build_ns2d_params)
    parameters = builder(spec, re, char_len, fluid)

    return ProblemResult(
        domain=domain_key,
        n_bits=resolution.n_bits,
        n_steps=resolution.n_steps,
        dt=resolution.dt_recommended,
        max_rank=spec.max_rank,
        parameters=parameters,
        reynolds_number=re,
        mach_number=ma,
        characteristic_length=char_len,
        fluid_name=fluid.name,
        geometry_type=spec.geometry.shape.value,
        problem_class=spec.problem_class.value,
        quality_tier=spec.quality,
        warnings=warnings,
        resolution_grid_1d=resolution.grid_points_1d,
        boundary_layer_thickness=resolution.boundary_layer_thickness,
        domain_extent=domain_length,
    )
