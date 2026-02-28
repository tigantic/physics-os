"""
Photochemistry: internal conversion / intersystem crossing rates,
photodissociation, fluorescence lifetime.

Upgrades domain XV.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

K_BOLT: float = 1.381e-23      # J/K
H_PLANCK: float = 6.626e-34    # J·s
HBAR: float = 1.055e-34        # J·s
C_LIGHT: float = 2.998e8       # m/s
EV_TO_J: float = 1.602e-19
CM1_TO_J: float = H_PLANCK * C_LIGHT * 100  # 1 cm⁻¹ in J
EPSILON_0: float = 8.854e-12   # F/m


# ---------------------------------------------------------------------------
#  Franck-Condon Factors
# ---------------------------------------------------------------------------

class FranckCondonFactors:
    r"""
    Franck-Condon overlap integrals for displaced harmonic oscillators.

    For a single mode with shift Δ (dimensionless):
    $$|\langle v'|v\rangle|^2 = e^{-S}\frac{S^{|v'-v|}}{|v'-v|!}
      \times\left[\min(v',v)! / \max(v',v)!\right] L_{min}^{|v'-v|}(S)^2$$

    Simplified for 0→v': $FC_{0v'} = e^{-S} S^{v'} / v'!$ (Poisson).
    Huang-Rhys factor: $S = \Delta^2/2$.
    """

    def __init__(self, S: float) -> None:
        """
        Parameters
        ----------
        S : Huang-Rhys factor.
        """
        self.S = S

    def overlap_0_to_v(self, v: int) -> float:
        """FC factor |⟨0|v'⟩|² (Poisson distribution)."""
        return math.exp(-self.S) * self.S**v / math.factorial(v)

    def spectrum(self, v_max: int = 20) -> NDArray[np.float64]:
        """FC progression from v=0."""
        return np.array([self.overlap_0_to_v(v) for v in range(v_max)])

    @staticmethod
    def from_displacement(delta_Q: float, omega: float,
                            mu: float) -> "FranckCondonFactors":
        """Construct from physical displacement.

        S = μω ΔQ² / (2ℏ).
        """
        S = mu * omega * delta_Q**2 / (2.0 * HBAR)
        return FranckCondonFactors(S)


# ---------------------------------------------------------------------------
#  Internal Conversion (IC) — Energy Gap Law
# ---------------------------------------------------------------------------

class InternalConversion:
    r"""
    Non-radiative internal conversion rate (energy gap law).

    Englman-Jortner:
    $$k_{IC} = \frac{2\pi}{\hbar}|V_{el}|^2\frac{1}{\sqrt{2\pi\hbar\omega_M E_{00}}}
               \exp\left(-\gamma\frac{E_{00}}{\hbar\omega_M}\right)$$

    where $\gamma = \ln(E_{00}/(S\hbar\omega_M)) - 1$,
    $\omega_M$ = accepting mode frequency, $V_{el}$ = electronic coupling,
    $E_{00}$ = energy gap.
    """

    def __init__(self, E_gap: float, V_el: float,
                 omega_M: float, S: float) -> None:
        """
        Parameters
        ----------
        E_gap : Energy gap (eV).
        V_el : Electronic coupling matrix element (eV).
        omega_M : Accepting mode frequency (cm⁻¹).
        S : Huang-Rhys factor for accepting mode.
        """
        self.E_gap = E_gap * EV_TO_J
        self.V_el = V_el * EV_TO_J
        self.omega_M = omega_M * CM1_TO_J / HBAR  # rad/s
        self.S = S

    def rate(self, T: float = 300.0) -> float:
        """k_IC (s⁻¹) at temperature T."""
        E00 = self.E_gap
        hw_M = HBAR * self.omega_M

        if hw_M < 1e-30 or self.S < 1e-30:
            return 0.0

        p = E00 / hw_M  # effective number of quanta
        gamma = math.log(max(p / self.S, 1e-10)) - 1.0

        prefactor = 2.0 * math.pi / HBAR * self.V_el**2
        density = 1.0 / math.sqrt(2.0 * math.pi * hw_M * E00 + 1e-60)
        exponential = math.exp(-gamma * p)

        # Thermal factor: (1 + n_B) where n_B = Bose occupation
        n_B = 1.0 / (math.exp(hw_M / (K_BOLT * T)) - 1.0) if K_BOLT * T > 0 else 0
        thermal = (1.0 + n_B)**p

        return abs(prefactor * density * exponential * thermal)


# ---------------------------------------------------------------------------
#  Intersystem Crossing (ISC) — Spin-Orbit Coupling
# ---------------------------------------------------------------------------

class IntersystemCrossing:
    r"""
    Intersystem crossing rate via spin-orbit coupling.

    $$k_{ISC} = \frac{2\pi}{\hbar}|H_{SO}|^2 \rho_{FC}$$

    where $H_{SO}$ = spin-orbit coupling matrix element,
    ρ_FC = Franck-Condon weighted density of states.

    El-Sayed rules:
    - S(n,π*) → T(π,π*): allowed, large HSO
    - S(π,π*) → T(n,π*): allowed, large HSO
    - S(π,π*) → T(π,π*): forbidden, small HSO
    """

    def __init__(self, H_SO: float, E_gap: float,
                 FC_density: float = 1e10) -> None:
        """
        Parameters
        ----------
        H_SO : Spin-orbit coupling (cm⁻¹).
        E_gap : Singlet-triplet gap (eV).
        FC_density : FC-weighted density of states (1/J).
        """
        self.H_SO = H_SO * CM1_TO_J
        self.E_gap = E_gap * EV_TO_J
        self.FC_density = FC_density

    def rate(self) -> float:
        """k_ISC (s⁻¹)."""
        return 2.0 * math.pi / HBAR * self.H_SO**2 * self.FC_density

    @staticmethod
    def el_sayed_allowed(transition_type: str) -> bool:
        """Check El-Sayed selection rule."""
        allowed = {
            "n_pi_star_to_pi_pi_star": True,
            "pi_pi_star_to_n_pi_star": True,
            "n_pi_star_to_n_pi_star": False,
            "pi_pi_star_to_pi_pi_star": False,
        }
        return allowed.get(transition_type, False)


# ---------------------------------------------------------------------------
#  Photodissociation
# ---------------------------------------------------------------------------

class Photodissociation:
    r"""
    Direct photodissociation rates.

    Cross-section model (Gaussian):
    $$\sigma(\lambda) = \sigma_{\max}\exp\left(-\frac{(\lambda-\lambda_0)^2}{2w^2}\right)$$

    J-value (photolysis rate):
    $$J = \int \sigma(\lambda)\phi(\lambda)F(\lambda)\,d\lambda$$

    where φ = quantum yield, F = actinic flux.
    """

    def __init__(self, sigma_max: float = 1e-20,
                 lambda_0: float = 300e-9,
                 width: float = 20e-9,
                 quantum_yield: float = 1.0) -> None:
        """
        Parameters
        ----------
        sigma_max : Peak absorption cross-section (cm²).
        lambda_0 : Peak wavelength (m).
        width : Gaussian width (m).
        quantum_yield : φ (0-1).
        """
        self.sigma_max = sigma_max
        self.lambda_0 = lambda_0
        self.width = width
        self.phi = quantum_yield

    def cross_section(self, wavelength: NDArray[np.float64]) -> NDArray[np.float64]:
        """σ(λ) in cm²."""
        return self.sigma_max * np.exp(-0.5 * ((wavelength - self.lambda_0) / self.width)**2)

    def j_value(self, wavelength: NDArray[np.float64],
                  actinic_flux: NDArray[np.float64]) -> float:
        """Photolysis rate J (s⁻¹).

        Parameters
        ----------
        wavelength : λ array (m).
        actinic_flux : F(λ) (photons cm⁻² s⁻¹ nm⁻¹).
        """
        sigma = self.cross_section(wavelength)
        dlambda = np.gradient(wavelength) * 1e9  # m → nm
        return float(np.sum(sigma * self.phi * actinic_flux * dlambda))

    def bond_dissociation_energy(self, dissociation_wavelength: float) -> float:
        """BDE from threshold wavelength: E = hc/λ."""
        return H_PLANCK * C_LIGHT / dissociation_wavelength / EV_TO_J  # eV


# ---------------------------------------------------------------------------
#  Fluorescence Lifetime
# ---------------------------------------------------------------------------

class FluorescenceLifetime:
    r"""
    Fluorescence lifetime and quantum yield.

    Radiative rate (Einstein A coefficient):
    $$k_r = \frac{1}{\tau_r} = \frac{8\pi n^2 \nu^2}{c^2}\frac{g_l}{g_u}
            \int\varepsilon(\nu)\,d\nu$$

    Strickler-Berg equation:
    $$\frac{1}{\tau_r} = 2.88\times 10^{-9} n^2
      \langle\tilde\nu^{-3}\rangle^{-1}\int\varepsilon(\tilde\nu)\,d\tilde\nu$$

    Observed lifetime: $\tau_{obs} = 1/(k_r + k_{nr})$

    Quantum yield: $\Phi_F = k_r / (k_r + k_{nr})$
    """

    def __init__(self, k_r: float, k_nr: float = 0.0) -> None:
        """
        Parameters
        ----------
        k_r : Radiative rate constant (s⁻¹).
        k_nr : Non-radiative rate constant (s⁻¹).
        """
        self.k_r = k_r
        self.k_nr = k_nr

    @property
    def radiative_lifetime(self) -> float:
        """τ_r = 1/k_r (s)."""
        return 1.0 / self.k_r if self.k_r > 0 else float('inf')

    @property
    def observed_lifetime(self) -> float:
        """τ_obs = 1/(k_r + k_nr) (s)."""
        total = self.k_r + self.k_nr
        return 1.0 / total if total > 0 else float('inf')

    @property
    def quantum_yield(self) -> float:
        """Φ_F = k_r / (k_r + k_nr)."""
        total = self.k_r + self.k_nr
        return self.k_r / total if total > 0 else 0.0

    @staticmethod
    def strickler_berg(n_refract: float,
                        epsilon: NDArray[np.float64],
                        nu_tilde: NDArray[np.float64]) -> float:
        """Radiative rate from Strickler-Berg equation.

        Parameters
        ----------
        n_refract : Refractive index.
        epsilon : Molar absorption coefficient (L mol⁻¹ cm⁻¹).
        nu_tilde : Wavenumber (cm⁻¹).
        """
        dnu = np.gradient(nu_tilde)

        # ∫ε dν̃
        integral_eps = float(np.trapz(epsilon, nu_tilde))

        # ⟨ν̃⁻³⟩ from emission (approximate from absorption blue-shifted)
        avg_nu3_inv = float(np.trapz(epsilon / nu_tilde**3, nu_tilde) / (integral_eps + 1e-30))

        return 2.88e-9 * n_refract**2 * integral_eps / (avg_nu3_inv + 1e-60)

    def stern_volmer(self, k_q: float, Q: float) -> float:
        """Stern-Volmer quenching.

        Φ₀/Φ = 1 + K_SV[Q] where K_SV = k_q τ₀.
        τ₀ = observed_lifetime.
        """
        K_SV = k_q * self.observed_lifetime
        return 1.0 + K_SV * Q

    def forster_rate(self, R: float, R0: float) -> float:
        """Förster (FRET) energy transfer rate.

        k_FRET = (1/τ_D)(R₀/R)⁶.
        """
        tau_D = self.observed_lifetime
        if tau_D <= 0 or R <= 0:
            return 0.0
        return (1.0 / tau_D) * (R0 / R)**6
