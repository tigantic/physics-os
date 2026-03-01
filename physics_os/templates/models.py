"""Problem template data models.

Pydantic models that define the client-facing schema for
problem specification.  These models are validated at the API
boundary and compiled into low-level QTT execution configs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════


class ProblemClass(str, Enum):
    """Supported problem classes (maps to a problem compiler)."""

    EXTERNAL_FLOW = "external_flow"
    INTERNAL_FLOW = "internal_flow"
    HEAT_TRANSFER = "heat_transfer"
    WAVE_PROPAGATION = "wave_propagation"
    NATURAL_CONVECTION = "natural_convection"
    BOUNDARY_LAYER = "boundary_layer"
    VORTEX_DYNAMICS = "vortex_dynamics"
    CHANNEL_FLOW = "channel_flow"


class GeometryType(str, Enum):
    """Supported parameterised geometry types."""

    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    WEDGE = "wedge"
    NACA_AIRFOIL = "naca_airfoil"
    FLAT_PLATE = "flat_plate"
    FIN_ARRAY = "fin_array"
    PIPE_BEND = "pipe_bend"
    ANNULUS = "annulus"
    BACKWARD_STEP = "backward_step"
    NONE = "none"


# ═══════════════════════════════════════════════════════════════════
# Sub-models
# ═══════════════════════════════════════════════════════════════════


class GeometrySpec(BaseModel):
    """Parameterised geometry specification.

    The ``params`` dict is type-checked by the problem compiler
    against the required parameters for the chosen ``shape``.
    """

    shape: GeometryType = GeometryType.NONE
    params: dict[str, float] = Field(default_factory=dict)


class FlowConditions(BaseModel):
    """Free-stream / reference flow conditions."""

    velocity: float = Field(
        ..., gt=0, description="Free-stream velocity [m/s]."
    )
    fluid: str = Field(
        default="air",
        description="Fluid name (lookup key in FluidProperties registry).",
    )
    temperature: Optional[float] = Field(
        default=None, description="Reference temperature [K] (overrides fluid default)."
    )
    pressure: Optional[float] = Field(
        default=None, description="Reference pressure [Pa] (defaults to 101325)."
    )


class BoundarySpec(BaseModel):
    """Template-level boundary specification.

    The problem compiler maps these human-readable names to the
    platform's BCType enum and builds BoundaryCondition objects.
    """

    inlet: str = Field(default="uniform", description="Inlet profile type.")
    outlet: str = Field(default="zero_gradient", description="Outlet BC type.")
    walls: str = Field(default="no_slip", description="Wall BC type.")
    top: str = Field(default="symmetry", description="Top boundary type.")
    bottom: str = Field(default="symmetry", description="Bottom boundary type.")


# ═══════════════════════════════════════════════════════════════════
# Top-level spec
# ═══════════════════════════════════════════════════════════════════


class ProblemSpec(BaseModel):
    """Complete problem specification from the client.

    This is the single input object the Problem Compiler receives.
    """

    problem_class: ProblemClass
    geometry: GeometrySpec = Field(default_factory=GeometrySpec)
    flow: FlowConditions
    boundaries: BoundarySpec = Field(default_factory=BoundarySpec)
    quality: str = Field(default="standard", description="Quality tier: quick|standard|high|maximum.")
    t_end: Optional[float] = Field(default=None, gt=0, description="Simulation end time [s]. Auto-computed if omitted.")
    domain_multiplier: float = Field(default=10.0, gt=1, description="Domain = multiplier × characteristic_length.")
    max_rank: int = Field(default=64, ge=2, le=128, description="Max QTT tensor rank.")

    @field_validator("quality")
    @classmethod
    def _validate_quality(cls, v: str) -> str:
        allowed = {"quick", "standard", "high", "maximum"}
        if v not in allowed:
            raise ValueError(f"quality must be one of {allowed}, got {v!r}")
        return v


# ═══════════════════════════════════════════════════════════════════
# Compiled result (output of problem compiler)
# ═══════════════════════════════════════════════════════════════════


class ProblemResult(BaseModel):
    """Output from the problem compiler — ready for executor.

    Contains both the low-level QTT execution parameters and the
    physics context needed for post-processing.
    """

    # ── Execution parameters (sent to executor) ───────────────────
    domain: str = Field(..., description="QTT domain key (e.g. 'navier_stokes_2d').")
    n_bits: int = Field(..., ge=4, le=14)
    n_steps: int = Field(..., ge=1, le=10_000)
    dt: float = Field(..., gt=0)
    max_rank: int = Field(..., ge=2, le=128)
    parameters: dict[str, Any] = Field(default_factory=dict)

    # ── Physics context (for post-processing) ─────────────────────
    reynolds_number: float = Field(default=0.0)
    mach_number: float = Field(default=0.0)
    characteristic_length: float = Field(default=1.0)
    fluid_name: str = Field(default="air")
    geometry_type: str = Field(default="none")
    problem_class: str = Field(default="external_flow")
    quality_tier: str = Field(default="standard")

    # ── Diagnostics ───────────────────────────────────────────────
    warnings: list[str] = Field(default_factory=list)
    resolution_grid_1d: int = Field(default=0)
    boundary_layer_thickness: float = Field(default=0.0)
    domain_extent: float = Field(default=0.0)
