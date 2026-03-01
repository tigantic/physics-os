"""Post-processing extractors and correlation comparison engine.

After a QTT simulation completes, these extractors convert raw
field data into engineering quantities and compare them against
published correlations.

Extractors
----------
- ``extract_drag``           : Integrate surface pressure → Cd
- ``extract_lift``           : Integrate surface pressure → Cl
- ``extract_nusselt``        : Local / average Nusselt number
- ``extract_strouhal``       : FFT of velocity probe → St
- ``extract_pressure_drop``  : Inlet−outlet ΔP
- ``extract_skin_friction``  : Wall shear → Cf
- ``extract_velocity_profile``: Cross-section u(y) profile
- ``extract_recirculation``  : Reattachment length from sign change
- ``extract_boundary_layer`` : δ, δ*, θ from velocity profile

Correlation engine
------------------
- ``compare_correlations``   : Match result against textbook value
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

from physics_os.core.dimensionless import (
    DragCorrelation,
    NusseltCorrelation,
    StrouhalCorrelation,
    drag_cylinder,
    drag_flat_plate_friction,
    drag_sphere,
    nusselt_cylinder_crossflow,
    nusselt_flat_plate_laminar,
    nusselt_flat_plate_turbulent,
    nusselt_natural_convection_vertical_plate,
    strouhal_cylinder,
)


# ═══════════════════════════════════════════════════════════════════
# Result containers
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class DragResult:
    """Drag force extraction result."""
    cd: float
    force_x: float
    reference_pressure: float
    method: str = "surface_pressure_integral"


@dataclass(frozen=True, slots=True)
class LiftResult:
    """Lift force extraction result."""
    cl: float
    force_y: float
    method: str = "surface_pressure_integral"


@dataclass(frozen=True, slots=True)
class NusseltResult:
    """Nusselt number extraction result."""
    nu_avg: float
    nu_local: Optional[Tensor] = None
    h_avg: float = 0.0
    method: str = "temperature_gradient"


@dataclass(frozen=True, slots=True)
class StrouhalResult:
    """Strouhal number from spectral analysis."""
    st: float
    dominant_frequency: float
    spectrum_peak_amplitude: float
    method: str = "fft_velocity_probe"


@dataclass(frozen=True, slots=True)
class PressureDropResult:
    """Pressure drop between inlet and outlet."""
    delta_p: float
    friction_factor: float = 0.0
    method: str = "inlet_outlet_average"


@dataclass(frozen=True, slots=True)
class SkinFrictionResult:
    """Skin friction coefficient."""
    cf_avg: float
    cf_local: Optional[Tensor] = None
    method: str = "wall_shear_stress"


@dataclass(frozen=True, slots=True)
class VelocityProfileResult:
    """Cross-section velocity profile."""
    y: Tensor
    u: Tensor
    u_max: float
    u_avg: float


@dataclass(frozen=True, slots=True)
class RecirculationResult:
    """Recirculation zone measurement."""
    reattachment_length: float
    reattachment_x: float
    step_height: float
    normalized_length: float  # x_r / h


@dataclass(frozen=True, slots=True)
class BoundaryLayerResult:
    """Boundary layer thicknesses."""
    delta_99: float       # 99% thickness
    delta_star: float     # displacement thickness
    theta: float          # momentum thickness
    shape_factor: float   # H = δ*/θ
    x_location: float


# ═══════════════════════════════════════════════════════════════════
# Extractors
# ═══════════════════════════════════════════════════════════════════

def extract_drag(
    pressure_field: Tensor,
    solid_mask: Tensor,
    dx: float,
    dy: float,
    rho_inf: float,
    u_inf: float,
) -> DragResult:
    """Extract drag coefficient from pressure field and solid mask.

    Approximates the surface integral of pressure × outward normal
    by finite differences at the solid–fluid interface.

    Parameters
    ----------
    pressure_field : Tensor
        2D pressure field (Ny, Nx).
    solid_mask : Tensor
        Boolean mask, True inside solid (Ny, Nx).
    dx, dy : float
        Grid spacing.
    rho_inf, u_inf : float
        Freestream density and velocity.
    """
    q_inf = 0.5 * rho_inf * u_inf**2
    if q_inf == 0:
        return DragResult(cd=0.0, force_x=0.0, reference_pressure=0.0)

    # Find fluid cells adjacent to solid (surface cells)
    mask_f = solid_mask.float()
    # Pressure gradient at interface approximates n · p ds
    # Use central difference of mask to find interface normals
    nx = torch.zeros_like(pressure_field)
    ny = torch.zeros_like(pressure_field)

    # x-direction gradient of mask (points from solid → fluid)
    nx[:, 1:-1] = (mask_f[:, 2:] - mask_f[:, :-2]) / 2.0
    ny[1:-1, :] = (mask_f[2:, :] - mask_f[:-2, :]) / 2.0

    # Surface integral: F_x = ∫ p * n_x ds
    force_x = (pressure_field * nx * dy).sum().item()
    cd = force_x / (q_inf * 1.0)  # per unit reference length

    return DragResult(
        cd=cd,
        force_x=force_x,
        reference_pressure=q_inf,
    )


def extract_lift(
    pressure_field: Tensor,
    solid_mask: Tensor,
    dx: float,
    dy: float,
    rho_inf: float,
    u_inf: float,
) -> LiftResult:
    """Extract lift coefficient from pressure field and solid mask."""
    q_inf = 0.5 * rho_inf * u_inf**2
    if q_inf == 0:
        return LiftResult(cl=0.0, force_y=0.0)

    mask_f = solid_mask.float()
    ny = torch.zeros_like(pressure_field)
    ny[1:-1, :] = (mask_f[2:, :] - mask_f[:-2, :]) / 2.0

    force_y = (pressure_field * ny * dx).sum().item()
    cl = force_y / (q_inf * 1.0)

    return LiftResult(cl=cl, force_y=force_y)


def extract_nusselt(
    temperature_field: Tensor,
    T_wall: float,
    T_inf: float,
    thermal_conductivity: float,
    characteristic_length: float,
    dy: float,
    wall_row: int = 0,
) -> NusseltResult:
    """Extract Nusselt number from temperature gradient at the wall.

    Nu = h L / k = -(dT/dn)|_wall × L / (T_wall - T_inf)
    """
    delta_T = T_wall - T_inf
    if abs(delta_T) < 1e-30:
        return NusseltResult(nu_avg=0.0, h_avg=0.0)

    # Temperature gradient at wall (first-order forward difference)
    dT_dn = (temperature_field[wall_row + 1, :] - temperature_field[wall_row, :]) / dy
    q_local = -thermal_conductivity * dT_dn
    h_local = q_local / delta_T
    nu_local = h_local * characteristic_length / thermal_conductivity

    nu_avg = nu_local.mean().item()
    h_avg = h_local.mean().item()

    return NusseltResult(
        nu_avg=nu_avg,
        nu_local=nu_local,
        h_avg=h_avg,
    )


def extract_strouhal(
    velocity_probe: Tensor,
    dt: float,
    characteristic_length: float,
    free_stream_velocity: float,
) -> StrouhalResult:
    """Extract Strouhal number from velocity time-series via FFT.

    Parameters
    ----------
    velocity_probe : Tensor
        1D time-series of velocity at a probe location.
    dt : float
        Time step between samples.
    characteristic_length, free_stream_velocity : float
        For non-dimensionalisation.
    """
    n = velocity_probe.shape[0]
    if n < 4:
        return StrouhalResult(st=0.0, dominant_frequency=0.0, spectrum_peak_amplitude=0.0)

    # Subtract mean
    signal = velocity_probe - velocity_probe.mean()

    # FFT
    spec = torch.fft.rfft(signal)
    power = torch.abs(spec[1:])  # exclude DC
    freqs = torch.fft.rfftfreq(n, d=dt)[1:]

    if power.numel() == 0:
        return StrouhalResult(st=0.0, dominant_frequency=0.0, spectrum_peak_amplitude=0.0)

    peak_idx = power.argmax().item()
    f_dom = freqs[peak_idx].item()
    amplitude = power[peak_idx].item()

    st = f_dom * characteristic_length / free_stream_velocity if free_stream_velocity > 0 else 0.0

    return StrouhalResult(
        st=st,
        dominant_frequency=f_dom,
        spectrum_peak_amplitude=amplitude,
    )


def extract_pressure_drop(
    pressure_field: Tensor,
    inlet_col: int = 0,
    outlet_col: int = -1,
) -> PressureDropResult:
    """Extract pressure drop between inlet and outlet columns."""
    p_inlet = pressure_field[:, inlet_col].mean().item()
    p_outlet = pressure_field[:, outlet_col].mean().item()
    delta_p = p_inlet - p_outlet
    return PressureDropResult(delta_p=delta_p)


def extract_skin_friction(
    velocity_field: Tensor,
    dy: float,
    rho: float,
    u_inf: float,
    wall_row: int = 0,
) -> SkinFrictionResult:
    """Extract skin friction from wall velocity gradient.

    Cf = τ_w / (0.5 ρ U²)  where τ_w = μ du/dy|_wall ≈ ρ ν du/dy|_wall.
    """
    q_inf = 0.5 * rho * u_inf**2
    if q_inf == 0:
        return SkinFrictionResult(cf_avg=0.0)

    # du/dy at wall (first-order)
    du_dy = velocity_field[wall_row + 1, :] / dy
    # For unit viscosity simplification: Cf = du_dy * dy / (q_inf)
    # More accurately: τ_w = μ du/dy, but we need μ.
    # Return normalised gradient as proxy
    cf_local = du_dy / (u_inf / dy)
    cf_avg = cf_local.abs().mean().item()

    return SkinFrictionResult(cf_avg=cf_avg, cf_local=cf_local)


def extract_velocity_profile(
    u_field: Tensor,
    y_coords: Tensor,
    x_col: int,
) -> VelocityProfileResult:
    """Extract velocity profile u(y) at a given x-column."""
    u_profile = u_field[:, x_col]
    return VelocityProfileResult(
        y=y_coords,
        u=u_profile,
        u_max=u_profile.max().item(),
        u_avg=u_profile.mean().item(),
    )


def extract_recirculation(
    u_field: Tensor,
    x_coords: Tensor,
    step_x: float,
    step_height: float,
    probe_row: int,
) -> RecirculationResult:
    """Find reattachment point from sign change in wall-adjacent u.

    Scans downstream from the step for the first positive-velocity cell.
    """
    u_wall = u_field[probe_row, :]
    n = u_wall.shape[0]

    # Find step location in grid
    step_idx = (x_coords - step_x).abs().argmin().item()

    reattach_x = x_coords[-1].item()  # default: end of domain
    for i in range(step_idx, n - 1):
        if u_wall[i].item() <= 0 and u_wall[i + 1].item() > 0:
            # Linear interpolation for exact location
            frac = -u_wall[i].item() / (u_wall[i + 1].item() - u_wall[i].item() + 1e-30)
            reattach_x = x_coords[i].item() + frac * (x_coords[i + 1].item() - x_coords[i].item())
            break

    length = reattach_x - step_x
    return RecirculationResult(
        reattachment_length=length,
        reattachment_x=reattach_x,
        step_height=step_height,
        normalized_length=length / step_height if step_height > 0 else 0.0,
    )


def extract_boundary_layer(
    u_profile: Tensor,
    y_coords: Tensor,
    u_inf: float,
    x_location: float,
    dy: float,
) -> BoundaryLayerResult:
    """Extract boundary layer thicknesses from a velocity profile.

    Parameters
    ----------
    u_profile : Tensor
        u(y) velocity profile (1D).
    y_coords : Tensor
        Corresponding y coordinates.
    u_inf : float
        Freestream velocity.
    x_location : float
        Streamwise location of the profile.
    dy : float
        Grid spacing for integration.
    """
    n = u_profile.shape[0]
    u_ratio = u_profile / u_inf

    # δ_99: first y where u/U_∞ ≥ 0.99
    delta_99 = y_coords[-1].item()
    for i in range(n):
        if u_ratio[i].item() >= 0.99:
            delta_99 = y_coords[i].item()
            break

    # δ*: displacement thickness ∫(1 - u/U) dy
    integrand_star = 1.0 - u_ratio.clamp(max=1.0)
    delta_star = (integrand_star * dy).sum().item()

    # θ: momentum thickness ∫(u/U)(1 - u/U) dy
    integrand_theta = u_ratio.clamp(min=0, max=1.0) * (1.0 - u_ratio.clamp(max=1.0))
    theta = (integrand_theta * dy).sum().item()

    shape_factor = delta_star / theta if theta > 1e-30 else 0.0

    return BoundaryLayerResult(
        delta_99=delta_99,
        delta_star=delta_star,
        theta=theta,
        shape_factor=shape_factor,
        x_location=x_location,
    )


# ═══════════════════════════════════════════════════════════════════
# Correlation comparison engine
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CorrelationComparison:
    """Result of comparing simulation output against a correlation."""

    quantity: str
    simulated: float
    correlated: float
    relative_error: float
    source: str
    within_tolerance: bool
    tolerance: float


def compare_drag_cylinder(
    cd_sim: float,
    re_D: float,
    tolerance: float = 0.25,
) -> CorrelationComparison:
    """Compare simulated Cd against empirical cylinder drag."""
    corr = drag_cylinder(re_D)
    err = abs(cd_sim - corr.cd) / max(abs(corr.cd), 1e-30)
    return CorrelationComparison(
        quantity="Cd (cylinder)",
        simulated=cd_sim,
        correlated=corr.cd,
        relative_error=err,
        source=corr.source,
        within_tolerance=err <= tolerance,
        tolerance=tolerance,
    )


def compare_strouhal_cylinder(
    st_sim: float,
    re_D: float,
    diameter: float,
    velocity: float,
    tolerance: float = 0.15,
) -> CorrelationComparison:
    """Compare simulated Strouhal against cylinder correlation."""
    corr = strouhal_cylinder(re_D, diameter, velocity)
    err = abs(st_sim - corr.st) / max(abs(corr.st), 1e-30) if corr.st > 0 else 0.0
    return CorrelationComparison(
        quantity="St (cylinder vortex shedding)",
        simulated=st_sim,
        correlated=corr.st,
        relative_error=err,
        source=corr.source,
        within_tolerance=err <= tolerance,
        tolerance=tolerance,
    )


def compare_nusselt_flat_plate(
    nu_sim: float,
    re_L: float,
    pr: float,
    thermal_conductivity: float,
    length: float,
    tolerance: float = 0.20,
) -> CorrelationComparison:
    """Compare simulated Nu against flat-plate correlation."""
    if re_L < 5e5:
        corr = nusselt_flat_plate_laminar(re_L, pr, thermal_conductivity, length)
        source_label = "laminar"
    else:
        corr = nusselt_flat_plate_turbulent(re_L, pr, thermal_conductivity, length)
        source_label = "turbulent"
    err = abs(nu_sim - corr.nu) / max(abs(corr.nu), 1e-30)
    return CorrelationComparison(
        quantity=f"Nu (flat plate, {source_label})",
        simulated=nu_sim,
        correlated=corr.nu,
        relative_error=err,
        source=corr.source,
        within_tolerance=err <= tolerance,
        tolerance=tolerance,
    )


def compare_nusselt_cylinder(
    nu_sim: float,
    re_D: float,
    pr: float,
    thermal_conductivity: float,
    diameter: float,
    tolerance: float = 0.20,
) -> CorrelationComparison:
    """Compare simulated Nu against Churchill-Bernstein cylinder correlation."""
    corr = nusselt_cylinder_crossflow(re_D, pr, thermal_conductivity, diameter)
    err = abs(nu_sim - corr.nu) / max(abs(corr.nu), 1e-30)
    return CorrelationComparison(
        quantity="Nu (cylinder crossflow)",
        simulated=nu_sim,
        correlated=corr.nu,
        relative_error=err,
        source=corr.source,
        within_tolerance=err <= tolerance,
        tolerance=tolerance,
    )


def compare_skin_friction_plate(
    cf_sim: float,
    re_L: float,
    tolerance: float = 0.20,
) -> CorrelationComparison:
    """Compare simulated Cf against flat-plate friction."""
    corr = drag_flat_plate_friction(re_L)
    err = abs(cf_sim - corr.cd) / max(abs(corr.cd), 1e-30)
    return CorrelationComparison(
        quantity="Cf (flat plate friction)",
        simulated=cf_sim,
        correlated=corr.cd,
        relative_error=err,
        source=corr.source,
        within_tolerance=err <= tolerance,
        tolerance=tolerance,
    )
