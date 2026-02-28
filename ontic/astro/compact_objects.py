"""
Compact Objects: TOV neutron star, Kerr ISCO, Shakura-Sunyaev accretion disk.

Upgrades domain XII.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants (CGS)
# ---------------------------------------------------------------------------

G: float = 6.674e-8          # dyn cm² g⁻²
C: float = 2.998e10          # cm/s
M_SUN: float = 1.989e33      # g
HBAR: float = 1.055e-27      # erg·s
K_BOLT: float = 1.381e-16    # erg/K
SIGMA_SB: float = 5.670e-5   # erg cm⁻² s⁻¹ K⁻⁴
M_NEUTRON: float = 1.675e-24 # g
M_PROTON: float = 1.673e-24  # g


# ---------------------------------------------------------------------------
#  Equation of State
# ---------------------------------------------------------------------------

class NeutronStarEOS:
    """
    Equation of state for neutron star matter.

    Implements:
    - Polytropic EOS: P = K ρ^Γ
    - SLy parametrised EOS (Douchin & Haensel 2001)
    - Simple free-neutron Fermi gas
    """

    @staticmethod
    def polytropic(rho: float, K: float = 5.38e9,
                     Gamma: float = 2.34) -> float:
        """P = K ρ^Γ (cgs)."""
        return K * rho**Gamma

    @staticmethod
    def polytropic_energy_density(rho: float, K: float = 5.38e9,
                                    Gamma: float = 2.34) -> float:
        """Energy density ε = ρc² + P/(Γ-1)."""
        P = NeutronStarEOS.polytropic(rho, K, Gamma)
        return rho * C**2 + P / (Gamma - 1.0)

    @staticmethod
    def free_neutron_fermi(rho: float) -> float:
        """Free degenerate neutron gas EOS.

        P = (3π²)^{2/3} ℏ² / (5 m_n) × n^{5/3}
        where n = ρ / m_n.
        """
        n = rho / M_NEUTRON
        pf = (3.0 * math.pi**2 * n)**(1.0 / 3.0) * HBAR
        return (3.0 * math.pi**2)**(2.0 / 3.0) * HBAR**2 / (5.0 * M_NEUTRON) * n**(5.0 / 3.0)

    @staticmethod
    def sly_eos(rho: float) -> float:
        """SLy parametrised EOS (piecewise polytrope approximation).

        Uses 4-piece polytrope fit from Read et al. (2009).
        """
        log_rho = math.log10(rho + 1e-30)

        # Piecewise: (log ρ range, K, Γ) in CGS
        pieces = [
            (0.0, 14.165, 6.22e12, 1.58),     # low density crust
            (14.165, 14.7, 6.178e15, 1.28),    # inner crust
            (14.7, 15.0, 5.38e9, 2.34),        # nuclear
            (15.0, 16.0, 3.01e8, 2.87),        # high density
        ]

        for rho_lo, rho_hi, K, Gamma in pieces:
            if log_rho < rho_hi:
                return K * rho**Gamma

        # Ultra-high density: stiffest EOS
        return pieces[-1][2] * rho**pieces[-1][3]


# ---------------------------------------------------------------------------
#  TOV (Tolman-Oppenheimer-Volkoff) Equation
# ---------------------------------------------------------------------------

class TOVSolver:
    r"""
    Tolman-Oppenheimer-Volkoff equation for static spherically
    symmetric neutron stars.

    $$\frac{dp}{dr} = -\frac{G}{r^2}\left(\rho + \frac{p}{c^2}\right)
        \left(m + \frac{4\pi r^3 p}{c^2}\right)
        \left(1 - \frac{2Gm}{rc^2}\right)^{-1}$$

    $$\frac{dm}{dr} = 4\pi r^2 \rho$$

    Integrates from centre (ρ_c) outward until P → 0 (surface).
    """

    def __init__(self, eos_func=None, K: float = 5.38e9,
                 Gamma: float = 2.34) -> None:
        """
        Parameters
        ----------
        eos_func : Callable(rho) → P. If None, uses polytropic.
        """
        if eos_func is not None:
            self.eos = eos_func
        else:
            self.eos = lambda rho: NeutronStarEOS.polytropic(rho, K, Gamma)
        self.K = K
        self.Gamma = Gamma

    def _energy_density(self, rho: float) -> float:
        """ε = ρ c² + P/(Γ-1) for polytropic."""
        P = self.eos(rho)
        return rho * C**2 + P / (self.Gamma - 1.0 + 1e-30)

    def _rho_from_P(self, P: float) -> float:
        """Invert EOS: ρ(P). Newton iteration on P(ρ) - P_target = 0."""
        if P <= 0:
            return 0.0
        # Initial guess from polytropic inversion
        rho = (P / self.K)**(1.0 / self.Gamma) if self.K > 0 else 1e14
        for _ in range(50):
            P_trial = self.eos(rho)
            dP = self.eos(rho * 1.001) - P_trial
            drho = 0.001 * rho
            if abs(dP) < 1e-30:
                break
            rho -= (P_trial - P) / (dP / drho)
            rho = max(rho, 1e-30)
        return rho

    def integrate(self, rho_c: float, dr: float = 100.0,
                    r_max: float = 2e6) -> Dict[str, float]:
        """
        Integrate TOV from centre to surface.

        Parameters
        ----------
        rho_c : Central density (g/cm³).
        dr : Radial step (cm).
        r_max : Maximum radius cutoff (cm).

        Returns
        -------
        Dict with keys: M_solar, R_km, rho_c, P_c.
        """
        P = self.eos(rho_c)
        rho = rho_c
        m = 0.0
        r = dr  # start slightly off centre

        # Initial mass from Taylor expansion
        m = 4.0 * math.pi / 3.0 * rho * r**3

        r_values = [r]
        m_values = [m]
        P_values = [P]

        while r < r_max:
            if P <= 0:
                break

            # TOV RHS
            eps = self._energy_density(rho)
            factor1 = eps + P / C**2
            factor2 = m + 4.0 * math.pi * r**3 * P / C**2
            factor3 = 1.0 - 2.0 * G * m / (r * C**2)

            if factor3 <= 0:
                break  # Inside Schwarzschild radius (shouldn't happen for physical EOS)

            dP_dr = -G * factor1 * factor2 / (r**2 * C**2 * factor3)
            dm_dr = 4.0 * math.pi * r**2 * rho

            # RK4 step (simplified: Euler for mass, full for pressure)
            P += dP_dr * dr
            m += dm_dr * dr
            r += dr

            if P <= 0:
                P = 0.0
                break

            rho = self._rho_from_P(P)

            r_values.append(r)
            m_values.append(m)
            P_values.append(P)

        return {
            "M_solar": m / M_SUN,
            "R_km": r / 1e5,
            "rho_c": rho_c,
            "P_c": self.eos(rho_c),
            "r_values": np.array(r_values),
            "m_values": np.array(m_values),
            "P_values": np.array(P_values),
        }

    def mass_radius_curve(self, rho_min: float = 1e14,
                            rho_max: float = 3e15,
                            n_points: int = 50) -> Tuple[NDArray, NDArray]:
        """Compute M-R curve by varying central density."""
        rho_c_arr = np.logspace(math.log10(rho_min), math.log10(rho_max), n_points)
        M_arr = np.zeros(n_points)
        R_arr = np.zeros(n_points)

        for i, rho_c in enumerate(rho_c_arr):
            result = self.integrate(rho_c)
            M_arr[i] = result["M_solar"]
            R_arr[i] = result["R_km"]

        return M_arr, R_arr


# ---------------------------------------------------------------------------
#  Kerr Black Hole Orbits
# ---------------------------------------------------------------------------

class KerrBlackHole:
    r"""
    Kerr black hole geodesics and ISCO (innermost stable circular orbit).

    Kerr metric in Boyer-Lindquist:
    $$ds^2 = -\left(1-\frac{2Mr}{\Sigma}\right)dt^2 + \frac{\Sigma}{\Delta}dr^2
             + \Sigma d\theta^2 + \frac{A}{\Sigma}\sin^2\theta\,d\phi^2
             - \frac{4Mar\sin^2\theta}{\Sigma}dt\,d\phi$$

    ISCO radius: $r_{\text{ISCO}} = M(3 + Z_2 \mp \sqrt{(3-Z_1)(3+Z_1+2Z_2)})$
    """

    def __init__(self, M: float = 1.0, a_star: float = 0.0) -> None:
        """
        Parameters
        ----------
        M : Mass in geometric units (G = c = 1), or solar masses.
        a_star : Spin parameter a/M, -1 < a* < 1.
        """
        self.M = M
        self.a = a_star * M  # dimensionful spin

    @property
    def r_horizon_plus(self) -> float:
        """Outer event horizon r+ = M + √(M² - a²)."""
        return self.M + math.sqrt(max(self.M**2 - self.a**2, 0.0))

    @property
    def r_horizon_minus(self) -> float:
        """Inner (Cauchy) horizon r- = M - √(M² - a²)."""
        return self.M - math.sqrt(max(self.M**2 - self.a**2, 0.0))

    @property
    def r_ergosphere(self) -> float:
        """Ergosphere outer boundary (equatorial): r_ergo = 2M."""
        return 2.0 * self.M

    def isco_radius(self, prograde: bool = True) -> float:
        """ISCO radius for prograde (+) or retrograde (-) orbits."""
        M = self.M
        a = self.a / M  # dimensionless

        Z1 = 1.0 + (1.0 - a**2)**(1.0 / 3.0) * ((1.0 + a)**(1.0 / 3.0) + (1.0 - a)**(1.0 / 3.0))
        Z2 = math.sqrt(3.0 * a**2 + Z1**2)

        if prograde:
            return M * (3.0 + Z2 - math.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))
        else:
            return M * (3.0 + Z2 + math.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))

    def isco_energy(self, prograde: bool = True) -> float:
        """Specific energy at ISCO: E/μ."""
        r = self.isco_radius(prograde)
        M = self.M
        return math.sqrt(1.0 - 2.0 * M / (3.0 * r))

    def radiative_efficiency(self, prograde: bool = True) -> float:
        """η = 1 - E_ISCO. Maximum: ~42% for a* → 1."""
        return 1.0 - self.isco_energy(prograde)

    def orbital_angular_momentum(self, r: float) -> float:
        """Specific angular momentum L for circular orbit at r."""
        M = self.M
        a = self.a
        sign = 1.0  # prograde
        num = M**0.5 * (r**2 - 2.0 * a * M**0.5 * r**0.5 + a**2)
        den = r**0.75 * math.sqrt(r**1.5 - 3.0 * M * r**0.5 + 2.0 * a * M**0.5)
        return sign * num / (den + 1e-30)

    def orbital_frequency(self, r: float) -> float:
        """Keplerian orbital frequency Ω = M^{1/2} / (r^{3/2} + a M^{1/2})."""
        return math.sqrt(self.M) / (r**1.5 + self.a * math.sqrt(self.M))


# ---------------------------------------------------------------------------
#  Shakura-Sunyaev Accretion Disk
# ---------------------------------------------------------------------------

class ShakuraSunyaevDisk:
    r"""
    Standard thin accretion disk (Shakura & Sunyaev 1973).

    α-prescription: viscous stress $w_{r\phi} = \alpha P$.

    Radial structure:
    $$T_{\text{eff}}(r) = \left[\frac{3GM\dot{M}}{8\pi\sigma r^3}
                           \left(1-\sqrt{\frac{r_{\text{in}}}{r}}\right)\right]^{1/4}$$

    Three zones:
    - (a) Inner: radiation pressure, electron scattering
    - (b) Middle: gas pressure, electron scattering
    - (c) Outer: gas pressure, free-free opacity
    """

    def __init__(self, M: float = 10.0, Mdot_edd_fraction: float = 0.1,
                 alpha: float = 0.1) -> None:
        """
        Parameters
        ----------
        M : BH mass in solar masses.
        Mdot_edd_fraction : Ṁ / Ṁ_Edd.
        alpha : Viscosity parameter.
        """
        self.M = M * M_SUN               # g
        self.alpha = alpha
        self.r_g = G * self.M / C**2     # gravitational radius

        # Eddington accretion rate (assuming η = 0.1)
        L_edd = 4.0 * math.pi * G * self.M * M_PROTON * C / 6.65e-25
        self.Mdot = Mdot_edd_fraction * L_edd / (0.1 * C**2)
        self.r_in = 6.0 * self.r_g       # Schwarzschild ISCO

    def effective_temperature(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """T_eff(r)."""
        factor = 3.0 * G * self.M * self.Mdot / (8.0 * math.pi * SIGMA_SB * r**3)
        inner = np.maximum(1.0 - np.sqrt(self.r_in / r), 0.0)
        return (factor * inner)**0.25

    def surface_density(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """Σ(r) for zone (b): gas pressure, electron scattering.

        Σ = 5.2 α^{-4/5} ṁ^{3/5} m^{1/4} r₁₀^{-3/4} f^{3/5} g/cm²
        """
        m = self.M / M_SUN
        mdot_16 = self.Mdot / 1e16
        r10 = r / (1e10)
        f = np.maximum(1.0 - np.sqrt(self.r_in / r), 1e-10)

        return (5.2 * self.alpha**(-0.8) * mdot_16**0.6
                * m**0.25 * r10**(-0.75) * f**0.6)

    def disk_height(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """H/r aspect ratio.

        Zone (b): H = 1.7e8 α^{-1/10} ṁ^{3/20} m^{9/8} r₁₀^{21/20} f^{3/20}
        """
        m = self.M / M_SUN
        mdot_16 = self.Mdot / 1e16
        r10 = r / 1e10
        f = np.maximum(1.0 - np.sqrt(self.r_in / r), 1e-10)

        H = (1.7e8 * self.alpha**(-0.1) * mdot_16**0.15
             * m**1.125 * r10**1.05 * f**0.15)
        return H / r

    def luminosity(self) -> float:
        """Total disk luminosity L = η Ṁ c²."""
        eta = 1.0 - math.sqrt(1.0 - 2.0 / 6.0)  # Schwarzschild efficiency
        return eta * self.Mdot * C**2

    def spectrum(self, nu: NDArray[np.float64],
                  n_annuli: int = 200) -> NDArray[np.float64]:
        """
        Multi-colour blackbody spectrum F_ν.

        F_ν = (4π cos i / D²) Σ B_ν(T_eff) r dr
        (normalised per unit area, i = 0, D = 1).
        """
        r_out = 1000.0 * self.r_in
        r_arr = np.logspace(np.log10(self.r_in * 1.01), np.log10(r_out), n_annuli)
        T_arr = self.effective_temperature(r_arr)

        F_nu = np.zeros_like(nu)
        for i in range(len(r_arr) - 1):
            dr_ann = r_arr[i + 1] - r_arr[i]
            T = T_arr[i]
            if T < 1.0:
                continue
            # Planck function B_ν
            h = 6.626e-27
            x = h * nu / (K_BOLT * T)
            x = np.clip(x, 0, 500)
            B_nu = 2.0 * h * nu**3 / C**2 / (np.exp(x) - 1.0 + 1e-30)
            F_nu += B_nu * 2.0 * math.pi * r_arr[i] * dr_ann

        return F_nu

    def peak_temperature(self) -> float:
        """Peak temperature occurs at r = (49/36) r_in."""
        r_peak = (49.0 / 36.0) * self.r_in
        return float(self.effective_temperature(np.array([r_peak]))[0])
