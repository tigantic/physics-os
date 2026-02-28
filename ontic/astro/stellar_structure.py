"""
Stellar Structure & Evolution — hydrostatic equilibrium, nuclear burning,
mixing-length convection, opacity, stellar evolution tracks.

Domain XII.1 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Constants (CGS)
# ---------------------------------------------------------------------------

G_CGS: float = 6.674e-8        # cm³ g⁻¹ s⁻²
C_CGS: float = 2.998e10        # cm/s
K_B: float = 1.381e-16         # erg/K
M_P: float = 1.673e-24         # g (proton mass)
SIGMA_SB: float = 5.670e-5     # erg cm⁻² s⁻¹ K⁻⁴
A_RAD: float = 4 * SIGMA_SB / C_CGS  # radiation constant
M_SUN: float = 1.989e33        # g
R_SUN: float = 6.957e10        # cm
L_SUN: float = 3.828e33        # erg/s


# ---------------------------------------------------------------------------
#  Equations of State
# ---------------------------------------------------------------------------

class StellarEOS:
    r"""
    Equation of state for stellar interiors.

    Ideal gas + radiation:
    $$P = P_{\text{gas}} + P_{\text{rad}} = \frac{\rho k_B T}{\mu m_p}
      + \frac{1}{3}aT^4$$

    Mean molecular weight for fully ionised gas:
    $$\frac{1}{\mu} = 2X + \frac{3}{4}Y + \frac{1}{2}Z$$
    where X, Y, Z = hydrogen, helium, metal mass fractions.

    Electron degeneracy pressure (non-relativistic):
    $$P_{\text{deg}} = K_{\text{NR}}\left(\frac{\rho}{\mu_e}\right)^{5/3}$$
    $K_{\text{NR}} = 1.004\times10^{13}$ (CGS), $\mu_e = 2/(1+X)$.
    """

    K_NR: float = 1.004e13  # non-relativistic degeneracy constant

    def __init__(self, X: float = 0.7, Y: float = 0.28, Z: float = 0.02) -> None:
        self.X = X
        self.Y = Y
        self.Z = Z
        self.mu = 1.0 / (2 * X + 0.75 * Y + 0.5 * Z)
        self.mu_e = 2.0 / (1 + X)

    def gas_pressure(self, rho: float, T: float) -> float:
        """Ideal gas pressure."""
        return rho * K_B * T / (self.mu * M_P)

    def radiation_pressure(self, T: float) -> float:
        """Radiation pressure: P_rad = (1/3) a T⁴."""
        return A_RAD * T**4 / 3

    def total_pressure(self, rho: float, T: float) -> float:
        """P = P_gas + P_rad."""
        return self.gas_pressure(rho, T) + self.radiation_pressure(T)

    def degeneracy_pressure(self, rho: float) -> float:
        """Non-relativistic electron degeneracy pressure."""
        return self.K_NR * (rho / self.mu_e)**(5.0 / 3)

    def beta(self, rho: float, T: float) -> float:
        """Gas pressure fraction: β = P_gas / P_total."""
        P_gas = self.gas_pressure(rho, T)
        P_total = self.total_pressure(rho, T)
        return P_gas / (P_total + 1e-30)

    def adiabatic_gradient(self, rho: float, T: float) -> float:
        """∇_ad = (∂ ln T / ∂ ln P)_s ≈ 0.4 for ideal gas."""
        b = self.beta(rho, T)
        return (32 - 24 * b - 3 * b**2) / (8 * (4 - 3 * b)) if b < 0.9999 else 0.4


# ---------------------------------------------------------------------------
#  Stellar Structure Equations
# ---------------------------------------------------------------------------

class StellarStructure:
    r"""
    Four equations of stellar structure (1D, spherically symmetric).

    1. Mass continuity:
    $$\frac{dm}{dr} = 4\pi r^2 \rho$$

    2. Hydrostatic equilibrium:
    $$\frac{dP}{dr} = -\frac{Gm\rho}{r^2}$$

    3. Energy generation:
    $$\frac{dL}{dr} = 4\pi r^2 \rho \varepsilon$$

    4. Energy transport:
    $$\frac{dT}{dr} = -\frac{3\kappa\rho L}{64\pi\sigma_B T^3 r^2}
      \quad\text{(radiative)}$$
    $$\frac{dT}{dr} = \left(1 - \frac{1}{\gamma}\right)\frac{T}{P}\frac{dP}{dr}
      \quad\text{(convective, adiabatic)}$$
    """

    def __init__(self, M_star: float = M_SUN, eos: Optional[StellarEOS] = None) -> None:
        self.M = M_star
        self.eos = eos or StellarEOS()

    def dm_dr(self, r: float, rho: float) -> float:
        """Mass continuity."""
        return 4 * math.pi * r**2 * rho

    def dP_dr(self, r: float, m: float, rho: float) -> float:
        """Hydrostatic equilibrium."""
        if r < 1e5:
            return 0.0
        return -G_CGS * m * rho / r**2

    def dL_dr(self, r: float, rho: float, epsilon: float) -> float:
        """Luminosity gradient."""
        return 4 * math.pi * r**2 * rho * epsilon

    def dT_dr_radiative(self, r: float, rho: float, T: float,
                           L: float, kappa: float) -> float:
        """Radiative temperature gradient."""
        if r < 1e5 or T < 1.0:
            return 0.0
        return -3 * kappa * rho * L / (64 * math.pi * SIGMA_SB * T**3 * r**2)

    def dT_dr_convective(self, r: float, m: float, rho: float,
                            T: float, P: float) -> float:
        """Adiabatic convective temperature gradient."""
        grad_ad = self.eos.adiabatic_gradient(rho, T)
        if P < 1e-10 or r < 1e5:
            return 0.0
        dPdr = self.dP_dr(r, m, rho)
        return grad_ad * T / P * dPdr

    def integrate(self, n_shells: int = 1000, r_max: Optional[float] = None,
                     rho_c: float = 1.5e2, T_c: float = 1.5e7,
                     kappa: float = 0.4, epsilon_func=None) -> Dict[str, NDArray]:
        """Integrate stellar structure from centre outward.

        Returns profiles of r, m, P, T, L, rho.
        """
        if r_max is None:
            r_max = 2 * R_SUN

        dr = r_max / n_shells
        r = np.zeros(n_shells)
        m = np.zeros(n_shells)
        P = np.zeros(n_shells)
        T = np.zeros(n_shells)
        L = np.zeros(n_shells)
        rho = np.zeros(n_shells)

        r[0] = dr
        rho[0] = rho_c
        T[0] = T_c
        P[0] = self.eos.total_pressure(rho_c, T_c)
        m[0] = 4 / 3 * math.pi * dr**3 * rho_c
        eps = epsilon_func(rho_c, T_c) if epsilon_func else self._pp_chain_rate(rho_c, T_c)
        L[0] = 4 / 3 * math.pi * dr**3 * rho_c * eps

        for i in range(1, n_shells):
            r[i] = r[i - 1] + dr
            ri = r[i]

            eps = epsilon_func(rho[i - 1], T[i - 1]) if epsilon_func else self._pp_chain_rate(rho[i - 1], T[i - 1])

            m[i] = m[i - 1] + self.dm_dr(ri, rho[i - 1]) * dr
            P[i] = P[i - 1] + self.dP_dr(ri, m[i], rho[i - 1]) * dr
            L[i] = L[i - 1] + self.dL_dr(ri, rho[i - 1], eps) * dr
            T[i] = T[i - 1] + self.dT_dr_radiative(ri, rho[i - 1], T[i - 1], L[i], kappa) * dr

            if P[i] <= 0 or T[i] <= 0:
                r = r[:i]
                m = m[:i]
                P = P[:i]
                T = T[:i]
                L = L[:i]
                rho = rho[:i]
                break

            rho[i] = P[i] * self.eos.mu * M_P / (K_B * T[i]) if T[i] > 0 else 0.0

        return {'r': r, 'm': m, 'P': P, 'T': T, 'L': L, 'rho': rho}

    @staticmethod
    def _pp_chain_rate(rho: float, T: float) -> float:
        """pp-chain energy generation rate (erg/g/s).

        ε_pp ≈ 1.08×10⁻¹² ρ X² T₆⁻²/³ exp(−33.80 T₆⁻¹/³) erg/g/s
        """
        X = 0.7
        T6 = T / 1e6
        if T6 < 1e-3:
            return 0.0
        return 1.08e-12 * rho * X**2 * T6**(-2 / 3) * math.exp(-33.80 * T6**(-1 / 3))


# ---------------------------------------------------------------------------
#  Mixing-Length Theory for Convection
# ---------------------------------------------------------------------------

class MixingLengthConvection:
    r"""
    Mixing-length theory (MLT) for stellar convection (Böhm-Vitense, 1958).

    Convective flux:
    $$F_{\text{conv}} = \rho c_p T \sqrt{\frac{g\delta}{8H_P}}
      \left(\frac{\alpha_{\text{MLT}} H_P}{2}\right)^2
      (\nabla - \nabla_{\text{ad}})^{3/2}$$

    Schwarzschild criterion: convective if $\nabla_{\text{rad}} > \nabla_{\text{ad}}$.

    $\alpha_{\text{MLT}} \approx 1.5$–$2.0$ (mixing length / pressure scale height).
    """

    def __init__(self, alpha_mlt: float = 1.8) -> None:
        self.alpha = alpha_mlt

    def pressure_scale_height(self, P: float, rho: float, g: float) -> float:
        """H_P = P / (ρg)."""
        return P / (rho * g + 1e-30)

    def radiative_gradient(self, kappa: float, rho: float, L: float,
                              r: float, m: float, T: float) -> float:
        """∇_rad = (3 κ ρ L P) / (64π σ G m T⁴)."""
        P = rho * K_B * T / (0.6 * M_P)
        return (3 * kappa * rho * L * P
                / (64 * math.pi * SIGMA_SB * G_CGS * m * T**4 + 1e-30))

    def is_convective(self, grad_rad: float, grad_ad: float = 0.4) -> bool:
        """Schwarzschild criterion."""
        return grad_rad > grad_ad

    def convective_velocity(self, delta_grad: float, g: float,
                               H_P: float) -> float:
        """v_conv = (α H_P / 2) √(g δ∇ / (2 H_P))."""
        if delta_grad <= 0:
            return 0.0
        l_m = self.alpha * H_P / 2
        return l_m * math.sqrt(g * delta_grad / (2 * H_P))

    def convective_flux(self, rho: float, T: float, cp: float,
                           delta_grad: float, g: float, H_P: float) -> float:
        """F_conv = ρ c_p T v_conv Δ∇."""
        v = self.convective_velocity(delta_grad, g, H_P)
        return rho * cp * T * v * delta_grad


# ---------------------------------------------------------------------------
#  Opacity Models
# ---------------------------------------------------------------------------

class StellarOpacity:
    r"""
    Rosseland mean opacity models for stellar interiors.

    Kramers' opacity laws:
    - Free-free (bremsstrahlung): $\kappa_{\text{ff}} \propto \rho T^{-7/2}$
    - Bound-free: $\kappa_{\text{bf}} \propto Z(1+X)\rho T^{-7/2}$
    - Electron scattering: $\kappa_{\text{es}} = 0.2(1+X)$ cm²/g

    $$\kappa_{\text{Kramers}} = 4.34\times10^{25} \frac{Z(1+X)}{T^{3.5}} \rho
      \quad\text{cm}^2\text{/g}$$
    """

    def __init__(self, X: float = 0.7, Z: float = 0.02) -> None:
        self.X = X
        self.Z = Z

    def electron_scattering(self) -> float:
        """Thomson electron scattering opacity."""
        return 0.2 * (1 + self.X)

    def kramers(self, rho: float, T: float) -> float:
        """Kramers' free-free + bound-free opacity."""
        if T < 1.0:
            return 1e4
        return 4.34e25 * self.Z * (1 + self.X) * rho / T**3.5

    def total(self, rho: float, T: float) -> float:
        """Total Rosseland mean opacity (harmonic combination)."""
        k_es = self.electron_scattering()
        k_kr = self.kramers(rho, T)
        return k_es + k_kr  # simplified sum (proper: harmonic mean)

    def eddington_luminosity(self, M: float) -> float:
        """L_Edd = 4πGMc / κ_es."""
        kappa_es = self.electron_scattering()
        return 4 * math.pi * G_CGS * M * C_CGS / kappa_es


# ---------------------------------------------------------------------------
#  Nuclear Burning Rates
# ---------------------------------------------------------------------------

class NuclearBurning:
    r"""
    Thermonuclear energy generation rates for stellar cores.

    pp chain (T < 15 MK):
    $$\varepsilon_{\text{pp}} \approx 1.08\times10^{-12}\rho X^2
      T_6^{-2/3}\exp(-33.80\,T_6^{-1/3})$$

    CNO cycle (T > 15 MK):
    $$\varepsilon_{\text{CNO}} \approx 8.24\times10^{-31}\rho X X_{\text{CNO}}
      T_6^{-2/3}\exp(-152.28\,T_6^{-1/3})$$

    Triple-alpha (He burning, T > 10⁸ K):
    $$\varepsilon_{3\alpha} \approx 3.86\times10^{11}\rho^2 Y^3
      T_8^{-3}\exp(-44.027/T_8)$$
    """

    def __init__(self, X: float = 0.7, Y: float = 0.28, Z: float = 0.02) -> None:
        self.X = X
        self.Y = Y
        self.Z = Z

    def pp_chain(self, rho: float, T: float) -> float:
        """pp chain rate (erg/g/s)."""
        T6 = T / 1e6
        if T6 < 1e-3:
            return 0.0
        return 1.08e-12 * rho * self.X**2 * T6**(-2 / 3) * math.exp(-33.80 * T6**(-1 / 3))

    def cno_cycle(self, rho: float, T: float) -> float:
        """CNO cycle rate (erg/g/s)."""
        T6 = T / 1e6
        if T6 < 1e-2:
            return 0.0
        X_CNO = 0.7 * self.Z
        return 8.24e-31 * rho * self.X * X_CNO * T6**(-2 / 3) * math.exp(-152.28 * T6**(-1 / 3))

    def triple_alpha(self, rho: float, T: float) -> float:
        """Triple-alpha rate (erg/g/s)."""
        T8 = T / 1e8
        if T8 < 1e-3:
            return 0.0
        return 3.86e11 * rho**2 * self.Y**3 * T8**(-3) * math.exp(-44.027 / T8)

    def total_rate(self, rho: float, T: float) -> float:
        """Total nuclear energy generation."""
        return self.pp_chain(rho, T) + self.cno_cycle(rho, T) + self.triple_alpha(rho, T)
