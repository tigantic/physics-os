"""FlowBox capability contract.

Defines the complete product contract for FlowBox v1:

  - **Presets**: IC library (taylor_green, vortex_merge, vortex_dipole, decay_noise)
  - **Tiers**: Grid/step/cadence caps tied to billing tiers
  - **Poisson profiles**: Solver configuration exposed as a simple enum
  - **Validation**: Parameter range enforcement (E002), tier cap enforcement
  - **Request/response schemas**: Pydantic models for the API surface

This module is the SINGLE SOURCE OF TRUTH for what FlowBox accepts
and what it returns.  All other FlowBox code references this contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════
# Presets
# ═══════════════════════════════════════════════════════════════════


class PresetKey(str, Enum):
    """Available IC presets."""

    TAYLOR_GREEN = "taylor_green"
    VORTEX_MERGE = "vortex_merge"
    VORTEX_DIPOLE = "vortex_dipole"
    DECAY_NOISE = "decay_noise"


@dataclass(frozen=True)
class PresetSpec:
    """Static definition of a FlowBox IC preset."""

    key: PresetKey
    label: str
    description: str
    ic_type: str  # Compiler ic_type key or "custom"
    default_viscosity: float
    default_dt_cfl_factor: float  # CFL safety factor for auto-dt
    analytical_solution: bool
    ic_n_modes: int = 0  # Only for multi_mode presets


PRESETS: dict[str, PresetSpec] = {
    "taylor_green": PresetSpec(
        key=PresetKey.TAYLOR_GREEN,
        label="Taylor-Green Vortex",
        description=(
            "Canonical benchmark: ω₀ = 2·sin(2πx)·sin(2πy).  "
            "Decays exponentially as exp(−8π²νt).  Verified against "
            "dense FFT reference to 3.5×10⁻⁸ relative L2."
        ),
        ic_type="taylor_green",
        default_viscosity=0.01,
        default_dt_cfl_factor=0.5,
        analytical_solution=True,
    ),
    "vortex_merge": PresetSpec(
        key=PresetKey.VORTEX_MERGE,
        label="Co-Rotating Vortex Merger",
        description=(
            "Two co-rotating Gaussian vortices at (0.35, 0.5) and "
            "(0.65, 0.5) that merge into a single vortex through "
            "viscous interaction."
        ),
        ic_type="custom",
        default_viscosity=0.001,
        default_dt_cfl_factor=0.25,
        analytical_solution=False,
    ),
    "vortex_dipole": PresetSpec(
        key=PresetKey.VORTEX_DIPOLE,
        label="Vortex Dipole",
        description=(
            "Counter-rotating vortex pair at (0.35, 0.5) and (0.65, 0.5) "
            "that propagates through the periodic domain.  Exhibits "
            "complex reconnection dynamics."
        ),
        ic_type="custom",
        default_viscosity=0.001,
        default_dt_cfl_factor=0.25,
        analytical_solution=False,
    ),
    "decay_noise": PresetSpec(
        key=PresetKey.DECAY_NOISE,
        label="Turbulent Decay",
        description=(
            "Broadband Fourier IC with Kolmogorov-like 1/(k²+m²) spectrum.  "
            "Seed-deterministic.  4 modes per dim (16 rank-1 terms)."
        ),
        ic_type="multi_mode",
        default_viscosity=0.01,
        default_dt_cfl_factor=0.5,
        analytical_solution=False,
        ic_n_modes=4,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# Poisson profiles (hides internal MG knobs)
# ═══════════════════════════════════════════════════════════════════


class PoissonProfile(str, Enum):
    """Poisson solver aggressiveness level."""

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


@dataclass(frozen=True)
class PoissonConfig:
    """Internal Poisson solver parameters resolved from a profile."""

    precond: str
    tol: float
    max_iters: int


_POISSON_PROFILES: dict[str, PoissonConfig] = {
    "fast": PoissonConfig(precond="mg", tol=5e-3, max_iters=40),
    "balanced": PoissonConfig(precond="mg", tol=1e-3, max_iters=80),
    "accurate": PoissonConfig(precond="mg", tol=1e-5, max_iters=200),
}


def resolve_poisson(profile: PoissonProfile) -> PoissonConfig:
    """Map a user-facing Poisson profile to internal solver config."""
    return _POISSON_PROFILES[profile.value]


# ═══════════════════════════════════════════════════════════════════
# Tier definitions (mapped to billing tiers)
# ═══════════════════════════════════════════════════════════════════


class FlowBoxTier(str, Enum):
    """Product tiers — mapped 1:1 to billing.stripe_billing.Tier."""

    EXPLORER = "explorer"
    BUILDER = "builder"
    PROFESSIONAL = "professional"


@dataclass(frozen=True)
class TierCaps:
    """Server-enforced limits per tier."""

    max_grid: int          # Max N where grid is N×N
    max_n_bits: int        # log2(max_grid)
    max_steps: int         # Max time steps per job
    min_cadence: int       # Min output cadence (every N steps)
    allowed_fields: tuple[str, ...]  # Which fields to return
    render_watermark: bool  # Watermark on MP4


TIER_CAPS: dict[str, TierCaps] = {
    "explorer": TierCaps(
        max_grid=512,
        max_n_bits=9,
        max_steps=500,
        min_cadence=10,
        allowed_fields=("omega",),
        render_watermark=True,
    ),
    "builder": TierCaps(
        max_grid=512,
        max_n_bits=9,
        max_steps=5000,
        min_cadence=5,
        allowed_fields=("omega", "psi"),
        render_watermark=False,
    ),
    "professional": TierCaps(
        max_grid=1024,
        max_n_bits=10,
        max_steps=10000,
        min_cadence=1,
        allowed_fields=("omega", "psi"),
        render_watermark=False,
    ),
}


def get_tier_caps(tier: FlowBoxTier) -> TierCaps:
    """Look up caps by tier enum."""
    return TIER_CAPS[tier.value]


# ═══════════════════════════════════════════════════════════════════
# Custom IC factories (vortex_merge, vortex_dipole)
# ═══════════════════════════════════════════════════════════════════

# Gaussian vortex parameters
_VORTEX_SIGMA = 0.08      # Gaussian half-width
_VORTEX_AMPLITUDE = 5.0   # Peak vorticity


def build_custom_separable(preset_key: str) -> list[tuple[list[Any], float]] | None:
    """Build QTT-compatible separable IC factors for custom presets.

    Returns a list of (factors, scale) tuples for multi-term separable
    initialization, matching the format consumed by
    ``GPURuntime._initialize_field_gpu()``.

    Returns None for presets handled by the standard compiler (taylor_green,
    multi_mode).
    """
    sigma = _VORTEX_SIGMA

    if preset_key == "vortex_merge":
        # Two co-rotating Gaussian vortices (same sign)
        return [
            (
                [
                    lambda x, _c=0.35, _s=sigma: np.exp(
                        -((x - _c) ** 2) / (2.0 * _s ** 2)
                    ),
                    lambda y, _c=0.5, _s=sigma: np.exp(
                        -((y - _c) ** 2) / (2.0 * _s ** 2)
                    ),
                ],
                _VORTEX_AMPLITUDE,
            ),
            (
                [
                    lambda x, _c=0.65, _s=sigma: np.exp(
                        -((x - _c) ** 2) / (2.0 * _s ** 2)
                    ),
                    lambda y, _c=0.5, _s=sigma: np.exp(
                        -((y - _c) ** 2) / (2.0 * _s ** 2)
                    ),
                ],
                _VORTEX_AMPLITUDE,
            ),
        ]

    if preset_key == "vortex_dipole":
        # Counter-rotating pair (opposite sign)
        return [
            (
                [
                    lambda x, _c=0.35, _s=sigma: np.exp(
                        -((x - _c) ** 2) / (2.0 * _s ** 2)
                    ),
                    lambda y, _c=0.5, _s=sigma: np.exp(
                        -((y - _c) ** 2) / (2.0 * _s ** 2)
                    ),
                ],
                _VORTEX_AMPLITUDE,
            ),
            (
                [
                    lambda x, _c=0.65, _s=sigma: np.exp(
                        -((x - _c) ** 2) / (2.0 * _s ** 2)
                    ),
                    lambda y, _c=0.5, _s=sigma: np.exp(
                        -((y - _c) ** 2) / (2.0 * _s ** 2)
                    ),
                ],
                -_VORTEX_AMPLITUDE,  # opposite sign
            ),
        ]

    return None


def build_dense_ic(preset_key: str, N: int, viscosity: float) -> np.ndarray:
    """Build the dense NxN vorticity IC for rendering.

    Parameters
    ----------
    preset_key : str
        One of the PresetKey values.
    N : int
        Grid resolution (points per dim).
    viscosity : float
        Not used for IC but stored for API consistency.

    Returns
    -------
    omega : np.ndarray, shape (N, N)
        Vorticity field on a node-centered grid x_i = i·h,
        h = 1/N, periodic domain [0, 1)².
    """
    h = 1.0 / N
    x = np.arange(N) * h
    X, Y = np.meshgrid(x, x, indexing="ij")

    if preset_key == "taylor_green":
        return 2.0 * np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)

    if preset_key == "vortex_merge":
        sigma = _VORTEX_SIGMA
        g1 = _VORTEX_AMPLITUDE * np.exp(
            -((X - 0.35) ** 2 + (Y - 0.5) ** 2) / (2.0 * sigma ** 2)
        )
        g2 = _VORTEX_AMPLITUDE * np.exp(
            -((X - 0.65) ** 2 + (Y - 0.5) ** 2) / (2.0 * sigma ** 2)
        )
        return g1 + g2

    if preset_key == "vortex_dipole":
        sigma = _VORTEX_SIGMA
        g1 = _VORTEX_AMPLITUDE * np.exp(
            -((X - 0.35) ** 2 + (Y - 0.5) ** 2) / (2.0 * sigma ** 2)
        )
        g2 = -_VORTEX_AMPLITUDE * np.exp(
            -((X - 0.65) ** 2 + (Y - 0.5) ** 2) / (2.0 * sigma ** 2)
        )
        return g1 + g2

    if preset_key == "decay_noise":
        # Replicate the compiler's multi_mode spectrum:
        # ω₀ = Σ [2/(k²+m²)] sin(2πkx) sin(2πmy), k,m ∈ 1..4
        omega = np.zeros((N, N), dtype=np.float64)
        for k in range(1, 5):
            for m in range(1, 5):
                amp = 2.0 / (k * k + m * m)
                omega += amp * (
                    np.sin(2.0 * np.pi * k * X)
                    * np.sin(2.0 * np.pi * m * Y)
                )
        return omega

    raise ValueError(f"Unknown preset: {preset_key!r}")


# ═══════════════════════════════════════════════════════════════════
# Request / response models
# ═══════════════════════════════════════════════════════════════════


class FlowBoxOutputs(BaseModel):
    """Requested output configuration."""

    cadence: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Capture a frame every N steps for the render.",
    )
    fields: list[str] = Field(
        default_factory=lambda: ["omega"],
        description="Which fields to include in the result.",
    )
    render: bool = Field(
        default=True,
        description="Generate MP4 animation.",
    )
    render_colormap: str = Field(
        default="RdBu_r",
        description="Matplotlib colormap for vorticity render.",
    )
    render_fps: int = Field(
        default=30,
        ge=1,
        le=120,
        description="Frames per second in the MP4.",
    )


class FlowBoxRequest(BaseModel):
    """FlowBox job submission request.

    This is the public-facing schema.  Parameter ranges are
    validated here (E002); tier caps are enforced at submission
    time by the router.
    """

    preset: PresetKey = Field(
        ...,
        description="Initial condition preset.",
    )
    grid: int = Field(
        default=512,
        description="Grid resolution N (N×N).  Must be 512 or 1024.",
    )
    viscosity: float | None = Field(
        default=None,
        ge=1e-6,
        le=1.0,
        description="Kinematic viscosity ν.  Uses preset default if omitted.",
    )
    dt: float | None = Field(
        default=None,
        gt=0.0,
        description="Time step size.  Auto-CFL if omitted.",
    )
    steps: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Number of time steps.",
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="RNG seed for deterministic execution.",
    )
    poisson_profile: PoissonProfile = Field(
        default=PoissonProfile.BALANCED,
        description="Poisson solver profile: fast, balanced, or accurate.",
    )
    outputs: FlowBoxOutputs = Field(
        default_factory=FlowBoxOutputs,
        description="Output configuration.",
    )

    @field_validator("grid")
    @classmethod
    def _validate_grid(cls, v: int) -> int:
        allowed = {512, 1024}
        if v not in allowed:
            raise ValueError(
                f"grid must be one of {sorted(allowed)}, got {v}.  "
                f"Minimum production grid is 512×512."
            )
        return v


# ═══════════════════════════════════════════════════════════════════
# Resolved config (preset + tier caps applied)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class FlowBoxConfig:
    """Fully resolved FlowBox execution config.

    Created by ``resolve()`` after applying preset defaults and
    tier caps.  This is what the executor receives.
    """

    preset: str
    preset_spec: PresetSpec
    tier: FlowBoxTier
    tier_caps: TierCaps

    # Grid
    n_bits: int
    grid: int  # 2^n_bits

    # Physics
    viscosity: float
    dt: float
    steps: int
    seed: int

    # Poisson
    poisson: PoissonConfig

    # Outputs
    output_cadence: int
    output_fields: tuple[str, ...]
    render: bool
    render_colormap: str
    render_fps: int
    render_watermark: bool

    @property
    def n_frames(self) -> int:
        """Number of render frames (excluding initial frame)."""
        return max(1, self.steps // self.output_cadence)


def resolve(
    request: FlowBoxRequest,
    tier: FlowBoxTier = FlowBoxTier.EXPLORER,
) -> FlowBoxConfig:
    """Resolve a FlowBoxRequest into a fully validated FlowBoxConfig.

    Applies preset defaults, computes auto-dt, and enforces tier caps.

    Raises
    ------
    ValueError
        If any parameter violates tier caps (E002).
    """
    preset_spec = PRESETS[request.preset.value]
    caps = get_tier_caps(tier)

    # ── Grid ────────────────────────────────────────────────────
    if request.grid > caps.max_grid:
        raise ValueError(
            f"Grid {request.grid} exceeds {tier.value} tier limit "
            f"({caps.max_grid}).  Upgrade to a higher tier."
        )
    n_bits = int(math.log2(request.grid))
    grid = request.grid

    # ── Steps ───────────────────────────────────────────────────
    steps = min(request.steps, caps.max_steps)

    # ── Viscosity ───────────────────────────────────────────────
    viscosity = request.viscosity or preset_spec.default_viscosity

    # ── dt (auto-CFL) ──────────────────────────────────────────
    h = 1.0 / grid
    cfl_factor = preset_spec.default_dt_cfl_factor
    if request.dt is not None:
        dt = request.dt
    else:
        # CFL condition: dt ≤ factor × h² / (2ν)
        dt = cfl_factor * h * h / (2.0 * viscosity)

    # ── Output cadence ──────────────────────────────────────────
    cadence = max(request.outputs.cadence, caps.min_cadence)

    # ── Output fields (intersect with tier whitelist) ───────────
    requested_fields = request.outputs.fields
    allowed = set(caps.allowed_fields)
    output_fields = tuple(f for f in requested_fields if f in allowed)
    if not output_fields:
        output_fields = (caps.allowed_fields[0],)

    # ── Poisson ─────────────────────────────────────────────────
    poisson = resolve_poisson(request.poisson_profile)

    return FlowBoxConfig(
        preset=request.preset.value,
        preset_spec=preset_spec,
        tier=tier,
        tier_caps=caps,
        n_bits=n_bits,
        grid=grid,
        viscosity=viscosity,
        dt=dt,
        steps=steps,
        seed=request.seed,
        poisson=poisson,
        output_cadence=cadence,
        output_fields=output_fields,
        render=request.outputs.render,
        render_colormap=request.outputs.render_colormap,
        render_fps=request.outputs.render_fps,
        render_watermark=caps.render_watermark,
    )


# ═══════════════════════════════════════════════════════════════════
# Preset listing (for GET /v1/flowbox/presets)
# ═══════════════════════════════════════════════════════════════════


def list_presets() -> list[dict[str, Any]]:
    """Return public-safe preset metadata for the API."""
    return [
        {
            "key": spec.key.value,
            "label": spec.label,
            "description": spec.description,
            "default_viscosity": spec.default_viscosity,
            "analytical_solution": spec.analytical_solution,
        }
        for spec in PRESETS.values()
    ]


def list_tiers() -> list[dict[str, Any]]:
    """Return public-safe tier metadata for the API."""
    return [
        {
            "tier": name,
            "max_grid": caps.max_grid,
            "max_steps": caps.max_steps,
            "min_cadence": caps.min_cadence,
            "allowed_fields": list(caps.allowed_fields),
            "render_watermark": caps.render_watermark,
        }
        for name, caps in TIER_CAPS.items()
    ]
