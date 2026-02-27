"""
Computational Glaciology — shallow ice approximation, Glen's flow law,
ice sheet thermodynamics, isostatic adjustment, calving.

Domain XIII.6 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Constants (Ice Sheet)
# ---------------------------------------------------------------------------

RHO_ICE: float = 917.0        # kg/m³
RHO_WATER: float = 1028.0     # kg/m³
RHO_MANTLE: float = 3300.0    # kg/m³
G_ACC: float = 9.81           # m/s²
K_ICE: float = 2.1            # W/(m·K) thermal conductivity
CP_ICE: float = 2009.0        # J/(kg·K) heat capacity
KAPPA_ICE: float = K_ICE / (RHO_ICE * CP_ICE)  # m²/s thermal diffusivity
LATENT_HEAT: float = 3.34e5   # J/kg (latent heat of fusion)
T_MELT: float = 273.15        # K


# ---------------------------------------------------------------------------
#  Glen's Flow Law
# ---------------------------------------------------------------------------

class GlenFlowLaw:
    r"""
    Glen's flow law for polycrystalline ice.

    $$\dot{\varepsilon}_{ij} = A\,\tau_e^{n-1}\sigma'_{ij}$$

    where $\tau_e = \sqrt{\tfrac{1}{2}\sigma'_{ij}\sigma'_{ij}}$
    is the effective stress, $n \approx 3$, and

    $$A(T) = A_0\exp\left(-\frac{Q}{R T}\right)$$

    $Q = 60$ kJ/mol (T < 263 K), $Q = 115$ kJ/mol (T ≥ 263 K).
    """

    R_GAS: float = 8.314

    def __init__(self, n: float = 3.0) -> None:
        self.n = n
        # Standard parameters (Cuffey & Paterson, 2010)
        self.A0_cold: float = 3.985e-13  # Pa⁻ⁿ s⁻¹ (T < 263 K)
        self.A0_warm: float = 1.916e3    # Pa⁻ⁿ s⁻¹ (T ≥ 263 K)
        self.Q_cold: float = 60e3        # J/mol
        self.Q_warm: float = 115e3       # J/mol (enhanced by recrystallisation)

    def rate_factor(self, T: float) -> float:
        """A(T) — Arrhenius rate factor (Pa⁻ⁿ s⁻¹)."""
        if T < 263.0:
            return self.A0_cold * math.exp(-self.Q_cold / (self.R_GAS * T))
        return self.A0_warm * math.exp(-self.Q_warm / (self.R_GAS * T))

    def effective_viscosity(self, T: float, strain_rate_eff: float) -> float:
        """η_eff = (1/2) A⁻¹/ⁿ |ε̇|^{(1-n)/n}."""
        A = self.rate_factor(T)
        if strain_rate_eff < 1e-30:
            strain_rate_eff = 1e-30
        return 0.5 * A**(-1 / self.n) * strain_rate_eff**((1 - self.n) / self.n)

    def deformation_velocity(self, H: float, alpha: float, T: float) -> float:
        """Depth-averaged deformation velocity (SIA).

        v_d = (2A/(n+2)) (ρ g sin α)ⁿ Hⁿ⁺¹
        """
        A = self.rate_factor(T)
        tau_b = RHO_ICE * G_ACC * H * math.sin(alpha)
        return 2 * A / (self.n + 2) * tau_b**self.n * H


# ---------------------------------------------------------------------------
#  Shallow Ice Approximation (SIA)
# ---------------------------------------------------------------------------

class ShallowIceApproximation:
    r"""
    1D shallow ice approximation for ice sheet evolution.

    $$\frac{\partial H}{\partial t} = M - \nabla\cdot\mathbf{q}$$

    Ice flux (Glen's law, n=3):
    $$q = -\frac{2A}{n+2}(\rho g)^n H^{n+2}|\nabla s|^{n-1}\nabla s$$

    where $s = b + H$ is surface elevation, $b$ is bedrock.
    """

    def __init__(self, nx: int = 200, dx: float = 5000.0,
                 glen: Optional[GlenFlowLaw] = None) -> None:
        self.nx = nx
        self.dx = dx
        self.glen = glen or GlenFlowLaw()

        self.H = np.zeros(nx)  # ice thickness
        self.b = np.zeros(nx)  # bedrock elevation

    def surface(self) -> NDArray:
        """s = b + H."""
        return self.b + self.H

    def ice_flux(self, T: float = 263.0) -> NDArray:
        """Compute ice flux q at staggered grid."""
        s = self.surface()
        n = self.glen.n
        A = self.glen.rate_factor(T)
        factor = -2 * A / (n + 2) * (RHO_ICE * G_ACC)**n

        q = np.zeros(self.nx - 1)
        for i in range(self.nx - 1):
            H_stag = 0.5 * (self.H[i] + self.H[i + 1])
            dsdx = (s[i + 1] - s[i]) / self.dx
            q[i] = factor * H_stag**(n + 2) * abs(dsdx)**(n - 1) * dsdx

        return q

    def step(self, dt: float, M: NDArray, T: float = 263.0) -> NDArray:
        """Advance ice thickness by dt.

        M: surface mass balance (m/s, positive = accumulation).
        """
        q = self.ice_flux(T)
        dqdx = np.zeros(self.nx)
        dqdx[1:-1] = (q[1:] - q[:-1]) / self.dx

        self.H += dt * (M - dqdx)
        self.H = np.maximum(self.H, 0.0)
        return self.H

    def volume(self) -> float:
        """Total ice volume (m³ per unit width)."""
        return float(np.sum(self.H) * self.dx)


# ---------------------------------------------------------------------------
#  Glacial Isostatic Adjustment (GIA)
# ---------------------------------------------------------------------------

class GlacialIsostaticAdjustment:
    r"""
    Glacial isostatic adjustment (ELRA model).

    Elastic lithosphere, relaxing asthenosphere:
    $$D\nabla^4 w + \rho_m g w = q(x,t)$$

    where $w$ = bedrock deflection, $D$ = flexural rigidity,
    $q = \rho_i g H$ = ice load.

    Relaxation:
    $$\frac{\partial b}{\partial t} = -\frac{1}{\tau_a}(b - b_{\text{eq}})$$

    τ_a ≈ 3000 yr (asthenospheric relaxation time).
    """

    def __init__(self, nx: int = 200, dx: float = 5000.0,
                 D: float = 1e25, tau_a: float = 3000 * 3.156e7) -> None:
        """
        D: flexural rigidity (N·m).
        tau_a: relaxation time (seconds).
        """
        self.nx = nx
        self.dx = dx
        self.D = D
        self.tau_a = tau_a
        self.b = np.zeros(nx)

    def equilibrium_deflection(self, H: NDArray) -> NDArray:
        """b_eq = −ρ_i/ρ_m × H (local isostasy)."""
        return -RHO_ICE / RHO_MANTLE * H

    def step(self, H: NDArray, dt: float) -> NDArray:
        """ELRA relaxation step.

        db/dt = −(b − b_eq)/τ_a
        """
        b_eq = self.equilibrium_deflection(H)
        self.b += dt / self.tau_a * (b_eq - self.b)
        return self.b

    def sea_level_equivalent(self, H: NDArray, ocean_area: float = 3.625e14) -> float:
        """Sea level equivalent of ice volume.

        SLE = ρ_i V_ice / (ρ_w × A_ocean)  [metres]
        """
        V = float(np.sum(H) * self.dx**2)
        return RHO_ICE * V / (RHO_WATER * ocean_area)


# ---------------------------------------------------------------------------
#  Ice Sheet Thermodynamics
# ---------------------------------------------------------------------------

class IceThermodynamics1D:
    r"""
    1D vertical temperature profile in an ice sheet.

    $$\frac{\partial T}{\partial t} = \kappa\frac{\partial^2 T}{\partial z^2}
      - w\frac{\partial T}{\partial z} + \frac{\Phi}{\rho c_p}$$

    where $w$ = vertical velocity, $\Phi$ = strain heating.

    Boundary conditions:
    - Surface: $T(z=H) = T_s$
    - Base: $-k ∂T/∂z|_{z=0} = G$ (geothermal heat flux)

    Robin (1955) analytical solution:
    $$T(z) = T_s + \frac{G H}{k}\sqrt{\frac{\pi}{2Pe}}
      \left[\text{erf}\left(\sqrt{\frac{Pe}{2}}\right)
      - \text{erf}\left(\sqrt{\frac{Pe}{2}}\frac{z}{H}\right)\right]$$

    where $Pe = wH/\kappa$ is the Peclet number.
    """

    def __init__(self, nz: int = 50, H: float = 3000.0) -> None:
        self.nz = nz
        self.H = H
        self.dz = H / (nz - 1)
        self.z = np.linspace(0, H, nz)
        self.T = np.full(nz, 263.0)

    def robin_solution(self, T_s: float = 243.0, G: float = 0.05,
                          w: float = 0.1) -> NDArray:
        """Robin (1955) analytical steady-state profile.

        G: geothermal heat flux (W/m²).
        w: vertical velocity at surface (m/yr → m/s internally).
        """
        from scipy.special import erf

        w_s = w / 3.156e7  # m/yr → m/s
        Pe = abs(w_s) * self.H / KAPPA_ICE if abs(w_s) > 1e-20 else 0.0

        T = np.zeros(self.nz)
        for i in range(self.nz):
            zeta = self.z[i] / self.H
            if Pe > 1e-6:
                T[i] = T_s + (G * self.H / K_ICE) * math.sqrt(math.pi / (2 * Pe)) * (
                    erf(math.sqrt(Pe / 2)) - erf(math.sqrt(Pe / 2) * zeta))
            else:
                T[i] = T_s + G / K_ICE * (self.H - self.z[i])

        return np.minimum(T, T_MELT)

    def step(self, dt: float, T_s: float = 243.0, G: float = 0.05,
                w_profile: Optional[NDArray] = None) -> NDArray:
        """Explicit FD timestep for temperature."""
        if w_profile is None:
            w_profile = np.zeros(self.nz)

        T_new = self.T.copy()
        for i in range(1, self.nz - 1):
            d2Tdz2 = (self.T[i + 1] - 2 * self.T[i] + self.T[i - 1]) / self.dz**2
            dTdz = (self.T[i + 1] - self.T[i - 1]) / (2 * self.dz)
            T_new[i] = self.T[i] + dt * (KAPPA_ICE * d2Tdz2 - w_profile[i] * dTdz)

        # Boundary conditions
        T_new[-1] = T_s  # surface
        T_new[0] = T_new[1] + G * self.dz / K_ICE  # basal geothermal

        self.T = np.minimum(T_new, T_MELT)
        return self.T

    def basal_melt_rate(self, G: float = 0.05) -> float:
        """Basal melt rate if T_base = T_melt.

        ṁ = (G − k ∂T/∂z|_base) / (ρ L)
        """
        dTdz_base = (self.T[1] - self.T[0]) / self.dz
        Q_net = G - K_ICE * dTdz_base
        if Q_net <= 0:
            return 0.0
        return Q_net / (RHO_ICE * LATENT_HEAT)  # m/s
