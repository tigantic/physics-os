"""
Coupled MHD: Hartmann channel flow, MHD-coupled crystal growth (Czochralski),
electromagnetic pump, magnetoconvection.

Upgrades domain XVIII.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Constants
# ---------------------------------------------------------------------------

MU_0: float = 4.0 * math.pi * 1e-7   # H/m


# ---------------------------------------------------------------------------
#  Hartmann Channel Flow
# ---------------------------------------------------------------------------

class HartmannFlow:
    r"""
    Exact Hartmann solution for MHD channel flow.

    Conducting fluid between two plates at y = ±a with transverse B₀.

    $$u(y) = \frac{U_0}{\tanh(Ha)}\left[\tanh(Ha)
             - \frac{\cosh(Ha\,y/a)}{\cosh(Ha)}\right] + U_0\frac{1 - \cosh(Ha\,y/a)/\cosh(Ha)}{\tanh(Ha)}$$

    Simplified pressure-driven:
    $$u(y) = -\frac{a^2}{\mu}\frac{dp}{dx}\frac{1}{Ha}
             \left[1 - \frac{\cosh(Ha\,y/a)}{\cosh(Ha)}\right]$$

    Hartmann number: $Ha = B_0 a\sqrt{\sigma/(\rho\nu)}$
    """

    def __init__(self, a: float, B0: float,
                 rho: float, nu: float, sigma: float,
                 dp_dx: float = -1.0) -> None:
        """
        Parameters
        ----------
        a : Half-channel width (m).
        B0 : Applied magnetic field (T).
        rho : Fluid density (kg/m³).
        nu : Kinematic viscosity (m²/s).
        sigma : Electrical conductivity (S/m).
        dp_dx : Pressure gradient (Pa/m), negative for flow in +x.
        """
        self.a = a
        self.B0 = B0
        self.rho = rho
        self.nu = nu
        self.sigma = sigma
        self.dp_dx = dp_dx
        self.mu = rho * nu

    @property
    def hartmann_number(self) -> float:
        """Ha = B₀ a √(σ/(ρν))."""
        return self.B0 * self.a * math.sqrt(self.sigma / (self.rho * self.nu))

    @property
    def stuart_number(self) -> float:
        """N = Ha²/Re (interaction parameter)."""
        # For channel flow, we define based on Ha and typical Re
        return self.hartmann_number**2

    def velocity_profile(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Exact Hartmann velocity profile u(y).

        Parameters
        ----------
        y : Position array, -a ≤ y ≤ a.
        """
        Ha = self.hartmann_number
        eta = y / self.a

        if Ha < 1e-6:
            # Poiseuille limit
            return (-self.a**2 / (2 * self.mu)) * self.dp_dx * (1 - eta**2)

        return (-self.a**2 / self.mu * self.dp_dx / Ha
                * (1.0 - np.cosh(Ha * eta) / math.cosh(Ha)))

    def induced_current(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Induced current density j_z(y) = σ(E_z - u B₀)."""
        u = self.velocity_profile(y)
        Ha = self.hartmann_number

        # E_z from insulating wall BC → j integral = 0
        # For insulating walls: E_z = <u>B₀
        u_mean = self.flow_rate() / (2 * self.a)
        return self.sigma * (u_mean - u) * self.B0

    def flow_rate(self) -> float:
        """Volume flow rate per unit depth Q = ∫u dy."""
        Ha = self.hartmann_number
        if Ha < 1e-6:
            return -2 * self.a**3 / (3 * self.mu) * self.dp_dx

        return (-2 * self.a**3 / (self.mu * Ha) * self.dp_dx
                * (1.0 - math.tanh(Ha) / Ha))

    def wall_shear_stress(self) -> float:
        """τ_w = μ du/dy|_{y=a}."""
        Ha = self.hartmann_number
        if Ha < 1e-6:
            return -self.a * self.dp_dx

        return (-self.a * self.dp_dx / Ha) * Ha * math.tanh(Ha) / math.cosh(Ha)

    def hartmann_layer_thickness(self) -> float:
        """δ_Ha = a/Ha."""
        Ha = self.hartmann_number
        return self.a / Ha if Ha > 1e-6 else float('inf')


# ---------------------------------------------------------------------------
#  MHD Crystal Growth (Czochralski)
# ---------------------------------------------------------------------------

class CzochralskiMHD:
    r"""
    MHD effects in Czochralski crystal growth.

    Combines natural convection with applied magnetic damping of melt flow.

    Key dimensionless groups:
    - Grashof: $Gr = g\beta\Delta T L^3/\nu^2$
    - Hartmann: $Ha = B_0 L\sqrt{\sigma/(\rho\nu)}$
    - MHD suppression: $u \propto u_0 / (1 + Ha^2/Gr^{1/2})$

    Solves simplified axisymmetric Stokes + Lorentz force model.
    """

    def __init__(self, L: float, rho: float, nu: float, sigma: float,
                 alpha_th: float, kappa: float,
                 g: float = 9.81) -> None:
        """
        Parameters
        ----------
        L : Crucible radius (m).
        rho : Melt density (kg/m³).
        nu : Kinematic viscosity (m²/s).
        sigma : Electrical conductivity (S/m).
        alpha_th : Thermal expansion coefficient (1/K).
        kappa : Thermal diffusivity (m²/s).
        g : Gravity (m/s²).
        """
        self.L = L
        self.rho = rho
        self.nu = nu
        self.sigma = sigma
        self.alpha_th = alpha_th
        self.kappa = kappa
        self.g = g

    def grashof(self, delta_T: float) -> float:
        return self.g * self.alpha_th * delta_T * self.L**3 / self.nu**2

    def hartmann(self, B0: float) -> float:
        return B0 * self.L * math.sqrt(self.sigma / (self.rho * self.nu))

    def prandtl(self) -> float:
        return self.nu / self.kappa

    def convective_velocity_scale(self, delta_T: float,
                                     B0: float = 0.0) -> float:
        """Estimate of melt velocity with MHD damping.

        u ~ u_buoy / (1 + Ha²/√Gr) where u_buoy = √(g β ΔT L).
        """
        Gr = self.grashof(delta_T)
        Ha = self.hartmann(B0)
        u_buoy = math.sqrt(self.g * self.alpha_th * delta_T * self.L)

        if Gr > 0:
            suppression = 1.0 + Ha**2 / math.sqrt(Gr)
        else:
            suppression = 1.0

        return u_buoy / suppression

    def oxygen_transport_peclet(self, delta_T: float,
                                  B0: float, D_O: float) -> float:
        """Péclet number for oxygen transport: Pe = uL/D_O."""
        u = self.convective_velocity_scale(delta_T, B0)
        return u * self.L / D_O

    def interface_deflection(self, delta_T: float, B0: float,
                               Tm: float, k_s: float) -> float:
        """Estimate crystal-melt interface deflection (m).

        δ ~ (k_m ΔT_melt) / (k_s G_s) with MHD reduction.
        """
        k_m = self.rho * self.kappa * self.nu / self.L  # crude thermal cond.
        u = self.convective_velocity_scale(delta_T, B0)
        # Interface deflection scales with convective heat flux
        return k_m * delta_T / (k_s * delta_T / self.L) * (u * self.L / self.kappa)


# ---------------------------------------------------------------------------
#  Electromagnetic Pump
# ---------------------------------------------------------------------------

class EMPump:
    r"""
    Electromagnetic induction pump for liquid metals.

    Travelling magnetic field induces eddy currents → Lorentz force.

    $$F = \sigma(\mathbf{v} \times \mathbf{B} + \mathbf{E})\times\mathbf{B}$$

    Equivalent circuit model:
    - Slip: $s = (v_s - v)/v_s$ where $v_s = 2f\tau$ (synchronous velocity)
    - Thrust: $F \propto s B^2 \sigma / (1 + s^2\sigma^2/(...)$)
    """

    def __init__(self, B_peak: float, frequency: float,
                 pole_pitch: float, channel_width: float,
                 channel_depth: float, sigma: float,
                 rho: float) -> None:
        """
        Parameters
        ----------
        B_peak : Peak magnetic field (T).
        frequency : Supply frequency (Hz).
        pole_pitch : Pole pitch τ (m).
        channel_width : w (m).
        channel_depth : d (m).
        sigma : Fluid conductivity (S/m).
        rho : Fluid density (kg/m³).
        """
        self.B_peak = B_peak
        self.f = frequency
        self.tau = pole_pitch
        self.w = channel_width
        self.d = channel_depth
        self.sigma = sigma
        self.rho = rho

    @property
    def synchronous_velocity(self) -> float:
        """v_s = 2fτ (m/s)."""
        return 2.0 * self.f * self.tau

    def slip(self, v_fluid: float) -> float:
        """s = (v_s - v)/v_s."""
        vs = self.synchronous_velocity
        return (vs - v_fluid) / vs if abs(vs) > 1e-30 else 0.0

    def thrust_density(self, v_fluid: float) -> float:
        """Volumetric Lorentz force density (N/m³).

        F_v = σ s B² v_s / (1 + (s·σ·B·d)² correction).
        """
        s = self.slip(v_fluid)
        vs = self.synchronous_velocity

        # Simplified: F ~ σ s ω_s B² d / 2 where ω_s = 2πf
        # More accurately: induced current ~ σ·s·v_s·B, force ~ j×B
        j_induced = self.sigma * s * vs * self.B_peak
        return j_induced * self.B_peak

    def pressure_rise(self, v_fluid: float, length: float) -> float:
        """Pressure developed (Pa) = F·L × active length."""
        return self.thrust_density(v_fluid) * length

    def efficiency(self, v_fluid: float) -> float:
        """η = (1-s) = v/v_s for ideal pump."""
        s = self.slip(v_fluid)
        return max(1.0 - s, 0.0)

    def operating_point(self, pipe_loss_coeff: float,
                          length: float) -> Tuple[float, float]:
        """Find steady-state operating point: ΔP_pump = ΔP_loss.

        pipe_loss_coeff: k in ΔP_loss = k·v².

        Returns (v_operating, ΔP_operating).
        """
        vs = self.synchronous_velocity
        # Bisection
        v_lo, v_hi = 0.0, vs

        for _ in range(100):
            v_mid = 0.5 * (v_lo + v_hi)
            dp_pump = self.pressure_rise(v_mid, length)
            dp_loss = pipe_loss_coeff * v_mid**2

            if dp_pump > dp_loss:
                v_lo = v_mid
            else:
                v_hi = v_mid

            if abs(v_hi - v_lo) < 1e-8:
                break

        v_op = 0.5 * (v_lo + v_hi)
        return v_op, self.pressure_rise(v_op, length)


# ---------------------------------------------------------------------------
#  Magnetoconvection
# ---------------------------------------------------------------------------

class Magnetoconvection:
    r"""
    Rayleigh-Bénard convection with vertical magnetic field.

    Critical Rayleigh number modified by Chandrasekhar number:
    $$Ra_c \approx \pi^4(1 + Q\pi^{-2})$$

    where Chandrasekhar number $Q = B_0^2 d^2 \sigma / (\rho\nu)$.

    Nusselt number correlation:
    $$Nu \approx 1 + (Ra/Ra_c - 1)^{0.3}$$ for $Ra > Ra_c$
    """

    def __init__(self, d: float, rho: float, nu: float,
                 sigma: float, kappa: float,
                 alpha_th: float, g: float = 9.81) -> None:
        self.d = d
        self.rho = rho
        self.nu = nu
        self.sigma = sigma
        self.kappa = kappa
        self.alpha_th = alpha_th
        self.g = g

    def rayleigh(self, delta_T: float) -> float:
        return self.g * self.alpha_th * delta_T * self.d**3 / (self.nu * self.kappa)

    def chandrasekhar(self, B0: float) -> float:
        return B0**2 * self.d**2 * self.sigma / (self.rho * self.nu)

    def critical_rayleigh(self, B0: float) -> float:
        """Ra_c with magnetic suppression."""
        Q = self.chandrasekhar(B0)
        # Chandrasekhar exact: Ra_c = π²(π² + a²)²/a² + π²Q
        # Minimise over wavenumber a → for large Q: Ra_c ≈ π²Q
        Ra_c_hydro = 27.0 * math.pi**4 / 4.0  # ≈ 657.5 for rigid-rigid
        return Ra_c_hydro + math.pi**2 * Q

    def nusselt(self, delta_T: float, B0: float) -> float:
        """Nusselt number."""
        Ra = self.rayleigh(delta_T)
        Ra_c = self.critical_rayleigh(B0)
        if Ra <= Ra_c:
            return 1.0
        return 1.0 + (Ra / Ra_c - 1.0)**0.3

    def onset_temperature_diff(self, B0: float) -> float:
        """ΔT at onset of convection."""
        Ra_c = self.critical_rayleigh(B0)
        return Ra_c * self.nu * self.kappa / (self.g * self.alpha_th * self.d**3)
