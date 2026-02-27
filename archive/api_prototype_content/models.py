"""HyperTensor API — Request and response schemas.

Every response strips TT-internal data (bond dimensions, singular
values, compression ratios, TT cores, rank evolution).  Clients
receive physical observables and conservation diagnostics only.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════
# Domain Enum
# ═══════════════════════════════════════════════════════════════════


class PhysicsDomain(str, Enum):
    """Available physics domains."""

    BURGERS = "burgers"
    MAXWELL_1D = "maxwell"
    MAXWELL_3D = "maxwell_3d"
    SCHRODINGER = "schrodinger"
    DIFFUSION = "advection_diffusion"
    VLASOV_POISSON = "vlasov_poisson"
    NAVIER_STOKES_2D = "navier_stokes_2d"


# ═══════════════════════════════════════════════════════════════════
# Request Models
# ═══════════════════════════════════════════════════════════════════


class SimulationRequest(BaseModel):
    """Top-level simulation request.

    The ``domain`` field selects the physics compiler.  ``parameters``
    holds domain-specific configuration (boundary conditions, physical
    constants, etc.).  ``resolution`` controls grid quality.
    """

    domain: PhysicsDomain = Field(
        ..., description="Physics domain to simulate."
    )
    resolution: ResolutionConfig = Field(
        default_factory=lambda: ResolutionConfig(),
        description="Grid and time-stepping configuration.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Domain-specific physical parameters.  "
            "See GET /v1/domains/{domain} for available parameters."
        ),
    )
    return_fields: bool = Field(
        default=True,
        description="Include dense field arrays in the response.",
    )
    return_coordinates: bool = Field(
        default=True,
        description="Include coordinate grid arrays.",
    )


class ResolutionConfig(BaseModel):
    """Grid and time-stepping configuration."""

    n_bits: int = Field(
        default=8,
        ge=4,
        le=14,
        description=(
            "Grid resolution in bits per dimension.  "
            "Grid size = 2^n_bits per dimension.  "
            "Range: 4 (16 points) to 14 (16 384 points)."
        ),
    )
    n_steps: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Number of time steps to simulate.",
    )
    dt: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Override time step.  Leave null for automatic CFL-based "
            "time step selection (recommended)."
        ),
    )
    max_rank: int = Field(
        default=64,
        ge=2,
        le=128,
        description="Maximum bond dimension for truncation policy.",
    )


# Forward reference resolution
SimulationRequest.model_rebuild()


# ═══════════════════════════════════════════════════════════════════
# Response Models — IP-Safe
# ═══════════════════════════════════════════════════════════════════


class GridInfo(BaseModel):
    """Spatial grid metadata."""

    dimensions: int = Field(..., description="Number of spatial dimensions.")
    resolution: list[int] = Field(
        ..., description="Grid points per dimension (e.g. [256] or [64, 64])."
    )
    domain_bounds: list[list[float]] = Field(
        ...,
        description="Physical domain bounds per dimension: [[lo, hi], ...].",
    )
    coordinates: dict[str, list[float]] | None = Field(
        default=None,
        description="Named coordinate arrays (x, y, z, v).  Omitted if return_coordinates=false.",
    )


class FieldData(BaseModel):
    """Dense physical field values.

    Contains ONLY the reconstructed physical quantities — no TT
    cores, bond dimensions, or singular value decomposition data.
    """

    name: str = Field(..., description="Field name (e.g. 'u', 'E', 'omega').")
    shape: list[int] = Field(..., description="Array shape.")
    values: list[Any] = Field(
        ...,
        description=(
            "Flattened field values (row-major).  "
            "Reshape with the provided 'shape' to recover N-D array."
        ),
    )
    unit: str = Field(default="", description="Physical unit (informational).")


class ConservationDiagnostic(BaseModel):
    """Conservation law verification.

    Reports how well the simulation preserved the expected invariant,
    *without* revealing the internal compression mechanism.
    """

    quantity: str = Field(
        ..., description="Conserved quantity name (e.g. 'total_mass', 'EM_energy')."
    )
    initial_value: float = Field(..., description="Value at t=0.")
    final_value: float = Field(..., description="Value at t_final.")
    relative_error: float = Field(
        ..., description="|(final - initial) / initial|."
    )
    status: str = Field(
        ..., description="'conserved' if relative_error < 1e-4, else 'drift'."
    )


class PerformanceInfo(BaseModel):
    """Execution performance summary — observable metrics only."""

    wall_time_s: float = Field(..., description="Total wall-clock time in seconds.")
    grid_points: int = Field(..., description="Total grid points (product of resolution).")
    time_steps: int = Field(..., description="Number of time steps executed.")
    throughput_gp_per_s: float = Field(
        ..., description="Grid-points × time-steps per second."
    )


class SimulationResponse(BaseModel):
    """Complete simulation result.

    Contains physical observables, conservation checks, and
    performance metrics.  All tensor-train internals are stripped.
    """

    job_id: str = Field(..., description="Unique job identifier.")
    status: str = Field(..., description="'completed' or 'failed'.")
    domain: str = Field(..., description="Physics domain that was solved.")
    domain_label: str = Field(..., description="Human-readable domain description.")
    equation: str = Field(default="", description="Governing equation (LaTeX).")

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Physical parameters used (echoed from request + defaults).",
    )
    grid: GridInfo = Field(..., description="Spatial grid information.")
    fields: dict[str, FieldData] | None = Field(
        default=None,
        description=(
            "Computed physical fields.  Omitted if return_fields=false "
            "or the simulation failed."
        ),
    )
    conservation: ConservationDiagnostic | None = Field(
        default=None, description="Conservation law diagnostic."
    )
    performance: PerformanceInfo = Field(..., description="Execution metrics.")
    error: str | None = Field(
        default=None, description="Error message if status='failed'."
    )


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str


# ═══════════════════════════════════════════════════════════════════
# Domain Catalog Models
# ═══════════════════════════════════════════════════════════════════


class DomainParameter(BaseModel):
    """Describes a tunable physical parameter for a domain."""

    name: str
    type: str = "float"
    default: Any = None
    description: str = ""
    min: float | None = None
    max: float | None = None


class DomainInfo(BaseModel):
    """Full description of a physics domain."""

    domain: str
    domain_label: str
    equation: str
    spatial_dimensions: int
    fields: list[str]
    conserved_quantity: str
    parameters: list[DomainParameter]
    example_request: dict[str, Any]


class DomainListResponse(BaseModel):
    """List of available physics domains."""

    domains: list[DomainInfo]
    count: int


# ═══════════════════════════════════════════════════════════════════
# Health Models
# ═══════════════════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    """Server health status."""

    status: str
    version: str
    device: str
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None
    uptime_s: float
    domains_available: int
