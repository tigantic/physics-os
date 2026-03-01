"""Dimensionless number computations and engineering correlations.

The Physics OS needs to translate physical specifications (velocity,
temperature, geometry) into the dimensionless parameters that govern
the PDE behaviour, and to translate raw outputs back into engineering
quantities (Nusselt, Strouhal, drag coefficient correlations).

Every function is pure — no side effects, no I/O.

References
----------
- Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*, 7th ed.
- White, *Viscous Fluid Flow*, 3rd ed.
- Schlichting & Gersten, *Boundary-Layer Theory*, 9th ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════
# Core dimensionless groups
# ═══════════════════════════════════════════════════════════════════


def reynolds_number(
    velocity: float,
    length: float,
    kinematic_viscosity: float,
) -> float:
    """Reynolds number: Re = U L / ν.

    Parameters
    ----------
    velocity : float   Free-stream velocity [m/s].
    length : float     Characteristic length [m].
    kinematic_viscosity : float   ν [m²/s].
    """
    if kinematic_viscosity <= 0:
        raise ValueError(f"kinematic_viscosity must be > 0, got {kinematic_viscosity}")
    return velocity * length / kinematic_viscosity


def mach_number(velocity: float, speed_of_sound: float) -> float:
    """Mach number: Ma = U / a."""
    if speed_of_sound <= 0:
        raise ValueError(f"speed_of_sound must be > 0, got {speed_of_sound}")
    return velocity / speed_of_sound


def prandtl_number(
    dynamic_viscosity: float,
    specific_heat_cp: float,
    thermal_conductivity: float,
) -> float:
    """Prandtl number: Pr = μ c_p / k."""
    if thermal_conductivity <= 0:
        raise ValueError(f"thermal_conductivity must be > 0, got {thermal_conductivity}")
    return dynamic_viscosity * specific_heat_cp / thermal_conductivity


def rayleigh_number(
    gravity: float,
    beta: float,
    delta_T: float,
    length: float,
    kinematic_viscosity: float,
    thermal_diffusivity: float,
) -> float:
    """Rayleigh number: Ra = g β ΔT L³ / (ν α).

    Parameters
    ----------
    gravity : float   Gravitational acceleration [m/s²].
    beta : float      Volumetric expansion coefficient [1/K].
    delta_T : float   Temperature difference [K].
    length : float    Characteristic length [m].
    kinematic_viscosity : float   ν [m²/s].
    thermal_diffusivity : float   α [m²/s].
    """
    return gravity * beta * delta_T * length**3 / (kinematic_viscosity * thermal_diffusivity)


def grashof_number(
    gravity: float,
    beta: float,
    delta_T: float,
    length: float,
    kinematic_viscosity: float,
) -> float:
    """Grashof number: Gr = g β ΔT L³ / ν²."""
    return gravity * beta * delta_T * length**3 / kinematic_viscosity**2


def peclet_number(velocity: float, length: float, diffusivity: float) -> float:
    """Péclet number: Pe = U L / α."""
    if diffusivity <= 0:
        raise ValueError(f"diffusivity must be > 0, got {diffusivity}")
    return velocity * length / diffusivity


def strouhal_number(frequency: float, length: float, velocity: float) -> float:
    """Strouhal number: St = f L / U."""
    if velocity <= 0:
        raise ValueError(f"velocity must be > 0, got {velocity}")
    return frequency * length / velocity


def richardson_number(
    gravity: float,
    beta: float,
    delta_T: float,
    length: float,
    velocity: float,
) -> float:
    """Richardson number: Ri = Gr / Re² = g β ΔT L / U²."""
    return gravity * beta * delta_T * length / velocity**2


def dean_number(re: float, d_h: float, r_c: float) -> float:
    """Dean number for curved-pipe flow: De = Re √(d_h / (2 R_c))."""
    return re * math.sqrt(d_h / (2.0 * r_c))


# ═══════════════════════════════════════════════════════════════════
# Engineering correlations
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class NusseltCorrelation:
    """Result of a Nusselt number correlation.

    Attributes
    ----------
    nu : float         Nusselt number.
    h : float          Convective heat-transfer coefficient [W/(m²·K)].
    source : str       Citation for the correlation used.
    validity : str     Range of validity (Re / Ra / Pr).
    """

    nu: float
    h: float
    source: str
    validity: str


def nusselt_flat_plate_laminar(
    re_L: float,
    pr: float,
    thermal_conductivity: float,
    length: float,
) -> NusseltCorrelation:
    """Flat-plate laminar boundary layer (Blasius + Pohlhausen).

    Nu_L = 0.664 Re_L^{1/2} Pr^{1/3}   for  Re_L < 5×10⁵,  Pr ≥ 0.6.
    """
    nu = 0.664 * re_L**0.5 * pr ** (1.0 / 3.0)
    h = nu * thermal_conductivity / length
    return NusseltCorrelation(
        nu=nu,
        h=h,
        source="Incropera & DeWitt, Eq. 7.30",
        validity="Re_L < 5e5, Pr >= 0.6",
    )


def nusselt_flat_plate_turbulent(
    re_L: float,
    pr: float,
    thermal_conductivity: float,
    length: float,
) -> NusseltCorrelation:
    """Flat-plate turbulent boundary layer (mixed, Whitaker).

    Nu_L = (0.037 Re_L^{4/5} - 871) Pr^{1/3}   for  5×10⁵ < Re_L < 10⁸.
    """
    nu = (0.037 * re_L**0.8 - 871.0) * pr ** (1.0 / 3.0)
    h = nu * thermal_conductivity / length
    return NusseltCorrelation(
        nu=nu,
        h=h,
        source="Incropera & DeWitt, Eq. 7.38",
        validity="5e5 < Re_L < 1e8, 0.6 <= Pr <= 60",
    )


def nusselt_cylinder_crossflow(
    re_D: float,
    pr: float,
    thermal_conductivity: float,
    diameter: float,
) -> NusseltCorrelation:
    """Churchill-Bernstein correlation for cylinder in cross-flow.

    Nu_D = 0.3 + [0.62 Re^{1/2} Pr^{1/3}] / [1 + (0.4/Pr)^{2/3}]^{1/4}
           × [1 + (Re/282000)^{5/8}]^{4/5}

    Valid for Re·Pr > 0.2.
    """
    term1 = 0.62 * re_D**0.5 * pr ** (1.0 / 3.0)
    term2 = (1.0 + (0.4 / pr) ** (2.0 / 3.0)) ** 0.25
    term3 = (1.0 + (re_D / 282_000.0) ** (5.0 / 8.0)) ** 0.8
    nu = 0.3 + (term1 / term2) * term3
    h = nu * thermal_conductivity / diameter
    return NusseltCorrelation(
        nu=nu,
        h=h,
        source="Churchill & Bernstein, ASME J. Heat Transfer 99:300, 1977",
        validity="Re*Pr > 0.2",
    )


def nusselt_natural_convection_vertical_plate(
    ra_L: float,
    pr: float,
    thermal_conductivity: float,
    length: float,
) -> NusseltCorrelation:
    """Churchill-Chu correlation for natural convection on a vertical plate.

    Nu_L = {0.825 + 0.387 Ra^{1/6} / [1 + (0.492/Pr)^{9/16}]^{8/27}}²
    """
    denom = (1.0 + (0.492 / pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    nu = (0.825 + 0.387 * ra_L ** (1.0 / 6.0) / denom) ** 2
    h = nu * thermal_conductivity / length
    return NusseltCorrelation(
        nu=nu,
        h=h,
        source="Churchill & Chu, Int. J. Heat Mass Transfer 18:1323, 1975",
        validity="All Ra (laminar + turbulent)",
    )


# ── Strouhal correlations ────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class StrouhalCorrelation:
    """Predicted vortex shedding Strouhal number."""

    st: float
    frequency_hz: float
    source: str
    validity: str


def strouhal_cylinder(
    re_D: float,
    diameter: float,
    velocity: float,
) -> StrouhalCorrelation:
    """Vortex shedding Strouhal number for a circular cylinder.

    Uses the Fey–König–Eckelmann correlation for
    47 < Re < 2×10⁵ (laminar vortex street → turbulent wake).
    """
    if re_D < 47:
        st = 0.0  # no periodic shedding
        source = "Below onset (Re < 47)"
    elif re_D < 180:
        # Laminar vortex street  (Williamson, 1996)
        st = 0.2120 - 3.81 / re_D
        source = "Williamson, Annu. Rev. Fluid Mech. 28, 1996"
    elif re_D < 2e5:
        st = 0.21  # broadly constant in subcritical regime
        source = "Roshko, NACA TN-3169, 1954"
    else:
        st = 0.27  # supercritical
        source = "Roshko, J. Fluid Mech. 10, 1961"

    f = st * velocity / diameter if diameter > 0 else 0.0
    return StrouhalCorrelation(
        st=st,
        frequency_hz=f,
        source=source,
        validity="47 < Re_D < 2e5 (primary range)",
    )


# ── Drag coefficient correlations ────────────────────────────────

@dataclass(frozen=True, slots=True)
class DragCorrelation:
    """Predicted drag coefficient from an empirical correlation."""

    cd: float
    source: str
    validity: str


def drag_cylinder(re_D: float) -> DragCorrelation:
    """Drag coefficient for an infinite circular cylinder in cross-flow.

    Uses piecewise fits from Schlichting & Gersten.
    """
    if re_D < 1:
        cd = 24.0 / max(re_D, 1e-12)
        source = "Stokes (creeping flow)"
    elif re_D < 1000:
        cd = 10.0 / re_D**0.5
        source = "Schlichting fit (intermediate)"
    elif re_D < 2e5:
        cd = 1.2
        source = "Subcritical (flat Cd)"
    else:
        cd = 0.3
        source = "Supercritical drag crisis"
    return DragCorrelation(cd=cd, source=source, validity=f"Re_D ≈ {re_D:.0f}")


def drag_flat_plate_friction(re_L: float) -> DragCorrelation:
    """Skin-friction drag on a flat plate aligned with the flow.

    Laminar: C_f = 1.328 / Re_L^{1/2}   (Blasius)
    Turbulent: C_f ≈ 0.074 / Re_L^{1/5} - 1742 / Re_L   (Schlichting)
    """
    if re_L < 5e5:
        cd = 1.328 / re_L**0.5
        source = "Blasius (laminar)"
    else:
        cd = 0.074 / re_L**0.2 - 1742.0 / re_L
        source = "Schlichting (turbulent, mixed)"
    return DragCorrelation(cd=cd, source=source, validity=f"Re_L ≈ {re_L:.2e}")


def drag_sphere(re_D: float) -> DragCorrelation:
    """Drag coefficient for a sphere (Schiller-Naumann + subcritical)."""
    if re_D < 0.1:
        cd = 24.0 / max(re_D, 1e-12)
        source = "Stokes (creeping flow)"
    elif re_D < 1000:
        cd = 24.0 / re_D * (1.0 + 0.15 * re_D**0.687)
        source = "Schiller-Naumann"
    elif re_D < 2e5:
        cd = 0.44
        source = "Newton (subcritical)"
    else:
        cd = 0.1
        source = "Supercritical drag crisis"
    return DragCorrelation(cd=cd, source=source, validity=f"Re_D ≈ {re_D:.0f}")


# ── Flow regime classification ────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FlowRegime:
    """Classification of the flow regime."""

    label: str
    description: str
    re: float
    ma: float


def classify_flow(
    re: float,
    ma: float = 0.0,
) -> FlowRegime:
    """Classify the flow regime from Re and Ma.

    Returns a ``FlowRegime`` with a short label and description.
    """
    # Compressibility classification
    if ma < 0.3:
        comp = "incompressible"
    elif ma < 0.8:
        comp = "subsonic-compressible"
    elif ma < 1.2:
        comp = "transonic"
    elif ma < 5.0:
        comp = "supersonic"
    else:
        comp = "hypersonic"

    # Viscous classification
    if re < 1:
        visc = "Stokes (creeping)"
    elif re < 2300:
        visc = "laminar"
    elif re < 4000:
        visc = "transitional"
    else:
        visc = "turbulent"

    label = f"{visc}, {comp}"
    description = (
        f"Re = {re:.2e} → {visc}; Ma = {ma:.3f} → {comp}. "
        f"Turbulence modelling {'recommended' if re > 4000 else 'not required'}."
    )
    return FlowRegime(label=label, description=description, re=re, ma=ma)
