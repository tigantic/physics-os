"""Automatic resolution advisor for problem templates.

Translates physical parameters (Re, Ma, geometry size) into the
appropriate QTT grid resolution (n_bits) and time-stepping
parameters.  The advisor considers:

* Boundary layer thickness  (δ ~ L / √Re)
* Kolmogorov micro-scale    (η ~ L Re^{-3/4})   for turbulent flows
* CFL stability constraint  (Δt ≤ CFL · Δx / (U + a))
* Acoustic time-scale       for compressible flows

Four quality tiers map to increasing computational cost:

    quick      →  exploratory / real-time preview
    standard   →  engineering design (default)
    high       →  publication quality
    maximum    →  platform ceiling (14-bit)

References
----------
- Pope, *Turbulent Flows*, Cambridge, 2000 (§6, §9)
- Colonius & Lele, J. Comput. Phys. 2004 (resolution requirements)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QualityTier(str, Enum):
    """Named quality tiers for simulation resolution."""

    QUICK = "quick"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


# Tier → target points-per-boundary-layer-thickness
_PPBL: dict[QualityTier, int] = {
    QualityTier.QUICK: 4,
    QualityTier.STANDARD: 8,
    QualityTier.HIGH: 16,
    QualityTier.MAXIMUM: 32,
}

# Tier → minimum n_bits.  QTT compression only becomes effective
# above ~256 points/axis (n_bits ≥ 8).  Below that the TT cores
# are smaller than dense storage and every Poisson solve is pure
# overhead.  These floors guarantee every run exercises QTT at a
# resolution where it provides real value.
_MIN_N_BITS_TIER: dict[QualityTier, int] = {
    QualityTier.QUICK: 9,       # 512×512   — QTT compression ≈ 3–5×
    QualityTier.STANDARD: 10,   # 1024×1024 — production baseline
    QualityTier.HIGH: 11,       # 2048×2048 — publication quality
    QualityTier.MAXIMUM: 12,    # 4096×4096 — platform near-ceiling
}

# Minimum points across the body geometry.  Even when the BL is
# thick (low Re), the geometry itself must be resolved to capture
# wake structure, pressure distribution, and correct drag.
_MIN_POINTS_PER_BODY: int = 32

# Platform hard limits  (mirrors physics_os/api/config.py)
_MAX_N_BITS = 14
_MAX_FIELD_POINTS = 500_000


@dataclass(frozen=True, slots=True)
class ResolutionAdvice:
    """Recommended simulation parameters from the resolution advisor.

    Attributes
    ----------
    n_bits : int
        Grid exponent: N = 2^n_bits points per axis.
    grid_points_1d : int
        Number of grid points in one dimension.
    total_points : int
        Total grid points (Nˢᵖᵃᵗⁱᵃˡ_ᵈⁱᵐˢ).
    dx : float
        Grid spacing [m].
    dt_cfl : float
        CFL-limited time step [s].
    dt_recommended : float
        Recommended Δt (CFL × safety factor) [s].
    n_steps : int
        Recommended number of time steps to reach *t_end*.
    tier : QualityTier
        Quality tier used.
    boundary_layer_thickness : float
        Estimated boundary layer δ [m].
    points_per_bl : float
        Grid points across the boundary layer.
    kolmogorov_scale : float | None
        Estimated Kolmogorov η [m] (None for laminar).
    cfl_number : float
        CFL number used.
    warnings : list[str]
        Human-readable warnings (e.g., "under-resolved turbulence").
    """

    n_bits: int
    grid_points_1d: int
    total_points: int
    dx: float
    dt_cfl: float
    dt_recommended: float
    n_steps: int
    tier: QualityTier
    boundary_layer_thickness: float
    points_per_bl: float
    kolmogorov_scale: Optional[float]
    cfl_number: float
    warnings: list[str]


def advise(
    *,
    re: float,
    characteristic_length: float,
    velocity: float,
    domain_length: float,
    t_end: float,
    spatial_dims: int = 2,
    speed_of_sound: float = 343.2,
    tier: QualityTier = QualityTier.STANDARD,
    cfl_safety: float = 0.5,
) -> ResolutionAdvice:
    """Compute recommended grid and time-step parameters.

    Parameters
    ----------
    re : float
        Reynolds number.
    characteristic_length : float
        Reference body length [m].
    velocity : float
        Free-stream velocity [m/s].
    domain_length : float
        Physical domain extent in longest dimension [m].
    t_end : float
        Simulation end time [s].
    spatial_dims : int
        Number of spatial dimensions (1 or 2).
    speed_of_sound : float
        Speed of sound [m/s] (for CFL with acoustic waves).
    tier : QualityTier
        Desired quality level.
    cfl_safety : float
        Safety factor applied to the CFL time step.

    Returns
    -------
    ResolutionAdvice
        Complete recommendation with diagnostics.
    """
    if re <= 0:
        raise ValueError(f"Reynolds number must be > 0, got {re}")
    if characteristic_length <= 0:
        raise ValueError(f"characteristic_length must be > 0, got {characteristic_length}")
    if velocity <= 0:
        raise ValueError(f"velocity must be > 0, got {velocity}")
    if domain_length <= 0:
        raise ValueError(f"domain_length must be > 0, got {domain_length}")
    if t_end <= 0:
        raise ValueError(f"t_end must be > 0, got {t_end}")

    warnings: list[str] = []

    # ── Boundary-layer estimate ───────────────────────────────────
    # Laminar: δ/L ≈ 5 / √Re   (Blasius)
    # Turbulent: δ/L ≈ 0.37 / Re^{1/5}
    is_turbulent = re > 4000
    if is_turbulent:
        bl_thickness = 0.37 * characteristic_length / re**0.2
    else:
        bl_thickness = 5.0 * characteristic_length / math.sqrt(re)

    # ── Kolmogorov scale (turbulent only) ─────────────────────────
    if is_turbulent:
        eta = characteristic_length * re ** (-0.75)
    else:
        eta = None

    # ── Required Δx to resolve boundary layer ─────────────────────
    ppbl_target = _PPBL[tier]
    dx_bl = bl_thickness / ppbl_target

    # ── Required Δx to resolve body geometry ──────────────────────
    # Even for creeping flow where δ >> body, we need enough points
    # across the characteristic length to capture pressure field,
    # wake structure, and geometry curvature.
    dx_body = characteristic_length / _MIN_POINTS_PER_BODY

    # Take the finer of the two requirements
    dx_required = min(dx_bl, dx_body)

    # ── n_bits from required Δx ───────────────────────────────────
    n_points_needed = math.ceil(domain_length / dx_required)
    n_bits_raw = math.ceil(math.log2(max(n_points_needed, 16)))
    min_n_bits = _MIN_N_BITS_TIER[tier]
    n_bits = max(min_n_bits, min(_MAX_N_BITS, n_bits_raw))

    # Informational: note when QTT tier floor raises resolution
    if n_bits_raw < min_n_bits:
        warnings.append(
            f"Physics requires only 2^{n_bits_raw} points/axis; "
            f"QTT tier floor raised to 2^{min_n_bits} ({tier.value}) "
            f"for effective tensor compression."
        )

    grid_1d = 2**n_bits
    dx_actual = domain_length / grid_1d
    total_points = grid_1d**spatial_dims

    # ── Points-per-boundary-layer achieved ────────────────────────
    ppbl_actual = bl_thickness / dx_actual if dx_actual > 0 else 0.0

    if ppbl_actual < 3:
        warnings.append(
            f"Boundary layer severely under-resolved ({ppbl_actual:.1f} pts/δ). "
            f"Consider increasing tier or reducing domain."
        )
    elif ppbl_actual < ppbl_target * 0.5:
        warnings.append(
            f"Boundary layer under-resolved ({ppbl_actual:.1f} pts/δ vs "
            f"{ppbl_target} target). May affect accuracy."
        )

    if eta is not None and dx_actual > 4 * eta:
        warnings.append(
            f"Grid does not resolve Kolmogorov scale "
            f"(Δx={dx_actual:.2e} vs η={eta:.2e}). "
            f"Results represent LES-like filtered solution."
        )

    if total_points > _MAX_FIELD_POINTS:
        warnings.append(
            f"Total points ({total_points:,}) exceeds field export limit "
            f"({_MAX_FIELD_POINTS:,}). Full field data will be omitted from API response."
        )

    # ── n_bits was clamped – warn ─────────────────────────────────
    if n_bits_raw > _MAX_N_BITS:
        warnings.append(
            f"Requested resolution (2^{n_bits_raw}) exceeds platform "
            f"ceiling (2^{_MAX_N_BITS}). Clamped to {_MAX_N_BITS} bits."
        )

    # ── CFL time step ─────────────────────────────────────────────
    wave_speed = velocity + speed_of_sound
    dt_cfl = dx_actual / wave_speed if wave_speed > 0 else dx_actual / max(velocity, 1e-12)
    dt_rec = cfl_safety * dt_cfl

    n_steps = max(1, math.ceil(t_end / dt_rec))
    # Cap at platform limit
    if n_steps > 10_000:
        warnings.append(
            f"Computed {n_steps} steps exceeds platform limit (10,000). "
            f"Clamped; effective end time = {10_000 * dt_rec:.4e} s."
        )
        n_steps = 10_000

    return ResolutionAdvice(
        n_bits=n_bits,
        grid_points_1d=grid_1d,
        total_points=total_points,
        dx=dx_actual,
        dt_cfl=dt_cfl,
        dt_recommended=dt_rec,
        n_steps=n_steps,
        tier=tier,
        boundary_layer_thickness=bl_thickness,
        points_per_bl=ppbl_actual,
        kolmogorov_scale=eta,
        cfl_number=cfl_safety,
        warnings=warnings,
    )


def advise_from_physics(
    *,
    velocity: float,
    characteristic_length: float,
    kinematic_viscosity: float,
    domain_multiplier: float = 10.0,
    flow_through_times: float = 5.0,
    spatial_dims: int = 2,
    speed_of_sound: float = 343.2,
    tier: QualityTier = QualityTier.STANDARD,
) -> ResolutionAdvice:
    """Higher-level advisor that computes Re and t_end automatically.

    Parameters
    ----------
    velocity : float
        Free-stream velocity [m/s].
    characteristic_length : float
        Reference body length [m].
    kinematic_viscosity : float
        ν [m²/s].
    domain_multiplier : float
        Domain length = multiplier × characteristic_length.
    flow_through_times : float
        t_end = FTT × domain_length / velocity.
    spatial_dims : int
        1 or 2.
    speed_of_sound : float
        Speed of sound [m/s].
    tier : QualityTier
        Quality level.

    Returns
    -------
    ResolutionAdvice
    """
    re = velocity * characteristic_length / kinematic_viscosity
    domain_length = domain_multiplier * characteristic_length
    t_end = flow_through_times * domain_length / velocity

    return advise(
        re=re,
        characteristic_length=characteristic_length,
        velocity=velocity,
        domain_length=domain_length,
        t_end=t_end,
        spatial_dims=spatial_dims,
        speed_of_sound=speed_of_sound,
        tier=tier,
    )
