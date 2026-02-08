"""
Reaction Rate Theory: Transition State Theory (harmonic, variational),
RRKM unimolecular decomposition, Kramers escape rate.

Upgrades domain XV.2.
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
R_GAS: float = 8.314           # J/(mol·K)
N_AV: float = 6.022e23


# ---------------------------------------------------------------------------
#  Eyring Transition State Theory
# ---------------------------------------------------------------------------

class TransitionStateTheory:
    r"""
    Conventional Transition State Theory (TST).

    Eyring equation:
    $$k_{TST} = \frac{k_B T}{h}\frac{Q^\ddagger}{Q_R}\exp\left(-\frac{E_a}{k_B T}\right)$$

    With Wigner tunnelling correction:
    $$\kappa_W = 1 + \frac{1}{24}\left(\frac{h\nu^\ddagger}{k_B T}\right)^2$$

    Partition functions: translational, rotational, vibrational (harmonic).
    """

    def __init__(self, E_a: float, frequencies_reactant: List[float],
                 frequencies_ts: List[float],
                 imaginary_freq: float = 0.0,
                 sigma_rot_r: int = 1, sigma_rot_ts: int = 1) -> None:
        """
        Parameters
        ----------
        E_a : Activation energy (kJ/mol).
        frequencies_reactant : Vibrational frequencies of reactant (cm⁻¹).
        frequencies_ts : Real vibrational frequencies of TS (cm⁻¹).
        imaginary_freq : Imaginary frequency at TS (cm⁻¹, positive value).
        sigma_rot_r, sigma_rot_ts : Rotational symmetry numbers.
        """
        self.E_a = E_a * 1e3 / N_AV  # Convert kJ/mol → J/molecule
        self.freq_R = frequencies_reactant
        self.freq_TS = frequencies_ts
        self.imag_freq = imaginary_freq
        self.sigma_R = sigma_rot_r
        self.sigma_TS = sigma_rot_ts

    def _vibrational_partition(self, freqs: List[float], T: float) -> float:
        """Quantum harmonic oscillator partition function.

        q_vib = Π (1 - exp(-hcν/kT))⁻¹
        """
        q = 1.0
        for nu in freqs:
            if nu <= 0:
                continue
            x = H_PLANCK * 2.998e10 * nu / (K_BOLT * T)
            q *= 1.0 / (1.0 - math.exp(-x)) if x < 500 else 1.0
        return q

    def rate_constant(self, T: float) -> float:
        """k_TST(T) in s⁻¹ (unimolecular) or cm³/s (bimolecular)."""
        q_R = self._vibrational_partition(self.freq_R, T) / self.sigma_R
        q_TS = self._vibrational_partition(self.freq_TS, T) / self.sigma_TS

        return (K_BOLT * T / H_PLANCK) * (q_TS / q_R) * math.exp(-self.E_a / (K_BOLT * T))

    def wigner_correction(self, T: float) -> float:
        """Wigner tunnelling correction κ_W."""
        if self.imag_freq <= 0:
            return 1.0
        u = H_PLANCK * 2.998e10 * self.imag_freq / (K_BOLT * T)
        return 1.0 + u**2 / 24.0

    def rate_with_tunnelling(self, T: float) -> float:
        """k = κ_W × k_TST."""
        return self.wigner_correction(T) * self.rate_constant(T)

    def arrhenius_parameters(self, T_range: Tuple[float, float] = (200, 2000),
                               n_pts: int = 50) -> Tuple[float, float]:
        """Fit Arrhenius A and Ea from k(T) data.

        Returns (A, Ea_kJ_mol).
        """
        T_arr = np.linspace(*T_range, n_pts)
        k_arr = np.array([self.rate_constant(T) for T in T_arr])

        # ln k = ln A - Ea/(R T)
        inv_T = 1.0 / T_arr
        ln_k = np.log(k_arr + 1e-300)

        # Linear fit
        coeffs = np.polyfit(inv_T, ln_k, 1)
        Ea = -coeffs[0] * R_GAS / 1e3  # kJ/mol
        A = math.exp(coeffs[1])

        return A, Ea


# ---------------------------------------------------------------------------
#  Variational TST
# ---------------------------------------------------------------------------

class VariationalTST:
    r"""
    Variational Transition State Theory (VTST).

    Minimise rate over dividing surface location:
    $$k_{VTST} = \min_s k_{TST}(s, T)$$

    where s parameterises the dividing surface along the reaction path.
    """

    def __init__(self, energy_profile: NDArray[np.float64],
                 freq_profiles: List[List[float]],
                 s_values: NDArray[np.float64]) -> None:
        """
        Parameters
        ----------
        energy_profile : E(s) along reaction coordinate (kJ/mol).
        freq_profiles : Vibrational frequencies at each s value.
        s_values : Reaction coordinate values.
        """
        self.E = energy_profile
        self.freqs = freq_profiles
        self.s = s_values

    def rate_at_dividing_surface(self, idx: int, T: float) -> float:
        """k_TST at dividing surface location idx."""
        E_a = self.E[idx] * 1e3 / N_AV  # kJ/mol → J
        q_vib = 1.0
        for nu in self.freqs[idx]:
            if nu <= 0:
                continue
            x = H_PLANCK * 2.998e10 * nu / (K_BOLT * T)
            q_vib *= 1.0 / (1.0 - math.exp(-x)) if x < 500 else 1.0

        # Reference: first point
        q_ref = 1.0
        for nu in self.freqs[0]:
            if nu <= 0:
                continue
            x = H_PLANCK * 2.998e10 * nu / (K_BOLT * T)
            q_ref *= 1.0 / (1.0 - math.exp(-x)) if x < 500 else 1.0

        return (K_BOLT * T / H_PLANCK) * (q_vib / q_ref) * math.exp(-E_a / (K_BOLT * T))

    def variational_rate(self, T: float) -> Tuple[float, float]:
        """Find minimum rate over all dividing surfaces.

        Returns (k_VTST, optimal s).
        """
        k_min = float('inf')
        s_opt = self.s[0]

        for i in range(len(self.s)):
            k = self.rate_at_dividing_surface(i, T)
            if k < k_min:
                k_min = k
                s_opt = self.s[i]

        return k_min, s_opt


# ---------------------------------------------------------------------------
#  RRKM Theory
# ---------------------------------------------------------------------------

class RRKMTheory:
    r"""
    Rice-Ramsperger-Kassel-Marcus unimolecular rate theory.

    Microcanonical rate:
    $$k(E) = \frac{W^\ddagger(E - E_0)}{h\rho(E)}$$

    where $W^\ddagger$ = sum of states at TS, $\rho$ = density of states of reactant.

    Thermal (high-pressure) rate:
    $$k_\infty(T) = \frac{k_B T}{h}\frac{Q^\ddagger}{Q}\exp(-E_0/k_B T)$$

    Falloff: Lindemann-Hinshelwood with strong-collision approximation.
    """

    def __init__(self, E0: float, freq_reactant: List[float],
                 freq_ts: List[float]) -> None:
        """
        Parameters
        ----------
        E0 : Barrier height (cm⁻¹).
        freq_reactant : Reactant vibrational frequencies (cm⁻¹).
        freq_ts : TS vibrational frequencies (cm⁻¹), one fewer than reactant.
        """
        self.E0 = E0
        self.freq_R = [f for f in freq_reactant if f > 0]
        self.freq_TS = [f for f in freq_ts if f > 0]

    def density_of_states(self, E: float, freqs: List[float]) -> float:
        """Beyer-Swinehart direct count ρ(E).

        Discrete count at energy bins of 1 cm⁻¹.
        """
        n_bins = int(E) + 1
        if n_bins <= 0:
            return 0.0
        rho = np.zeros(n_bins)
        rho[0] = 1.0

        for freq in freqs:
            freq_int = max(int(round(freq)), 1)
            for e in range(freq_int, n_bins):
                rho[e] += rho[e - freq_int]

        return float(rho[-1])

    def sum_of_states(self, E: float) -> float:
        """W‡(E - E0) = Σ ρ_TS(ε) for ε = 0 to E-E0."""
        excess = E - self.E0
        if excess <= 0:
            return 0.0
        # Integrate density of states
        n_bins = int(excess) + 1
        rho = np.zeros(n_bins)
        rho[0] = 1.0

        for freq in self.freq_TS:
            freq_int = max(int(round(freq)), 1)
            for e in range(freq_int, n_bins):
                rho[e] += rho[e - freq_int]

        return float(np.sum(rho))

    def microcanonical_rate(self, E: float) -> float:
        """k(E) = W‡(E-E0) / (h ρ(E))."""
        W = self.sum_of_states(E)
        rho = self.density_of_states(E, self.freq_R)
        if rho < 1e-30:
            return 0.0
        return W / (H_PLANCK * 2.998e10 * rho)  # Convert cm⁻¹ → s⁻¹

    def thermal_rate_high_pressure(self, T: float) -> float:
        """k_∞(T) from canonical TST."""
        q_R = 1.0
        for nu in self.freq_R:
            x = H_PLANCK * 2.998e10 * nu / (K_BOLT * T)
            q_R *= 1.0 / (1.0 - math.exp(-x)) if x < 500 else 1.0

        q_TS = 1.0
        for nu in self.freq_TS:
            x = H_PLANCK * 2.998e10 * nu / (K_BOLT * T)
            q_TS *= 1.0 / (1.0 - math.exp(-x)) if x < 500 else 1.0

        E0_J = self.E0 * H_PLANCK * 2.998e10
        return (K_BOLT * T / H_PLANCK) * (q_TS / q_R) * math.exp(-E0_J / (K_BOLT * T))


# ---------------------------------------------------------------------------
#  Kramers Escape Rate
# ---------------------------------------------------------------------------

class KramersRate:
    r"""
    Kramers' theory: thermally activated escape from a potential well
    in the presence of friction.

    Overdamped (high friction):
    $$k = \frac{\omega_0 \omega_b}{2\pi\gamma}\exp(-E_b/k_B T)$$

    Underdamped (low friction):
    $$k = \frac{\gamma\omega_0 E_b}{2\pi\omega_0 k_B T}\exp(-E_b/k_B T)$$

    Intermediate (Grote-Hynes):
    $$k = \frac{\lambda_r}{\omega_b}k_{TST}$$
    where $\lambda_r$ is reactive frequency solving $\lambda_r = \omega_b^2/(\lambda_r + \gamma)$.
    """

    def __init__(self, omega_0: float, omega_b: float,
                 E_b: float, gamma: float) -> None:
        """
        Parameters
        ----------
        omega_0 : Well frequency (s⁻¹).
        omega_b : Barrier frequency (s⁻¹).
        E_b : Barrier height (J or eV → converted).
        gamma : Friction coefficient (s⁻¹).
        """
        self.omega_0 = omega_0
        self.omega_b = omega_b
        self.E_b = E_b
        self.gamma = gamma

    def tst_rate(self, T: float) -> float:
        """TST rate (no friction): k = (ω₀/2π) exp(-Eb/kBT)."""
        return self.omega_0 / (2.0 * math.pi) * math.exp(-self.E_b / (K_BOLT * T))

    def overdamped_rate(self, T: float) -> float:
        """Smoluchowski limit (γ >> ωb)."""
        return (self.omega_0 * self.omega_b / (2.0 * math.pi * self.gamma)
                * math.exp(-self.E_b / (K_BOLT * T)))

    def underdamped_rate(self, T: float) -> float:
        """Energy diffusion limit (γ << ωb)."""
        return (self.gamma * self.E_b / (K_BOLT * T)
                * self.omega_0 / (2.0 * math.pi)
                * math.exp(-self.E_b / (K_BOLT * T)))

    def grote_hynes_rate(self, T: float) -> float:
        """Grote-Hynes: interpolates between limits.

        λ_r² + γ λ_r - ωb² = 0 → λ_r = (-γ + √(γ² + 4ωb²))/2.
        """
        lambda_r = (-self.gamma + math.sqrt(self.gamma**2 + 4 * self.omega_b**2)) / 2.0
        transmission = lambda_r / self.omega_b if self.omega_b > 0 else 1.0
        return transmission * self.tst_rate(T)

    def turnover_regime(self) -> str:
        """Identify friction regime."""
        if self.gamma > 10 * self.omega_b:
            return "overdamped"
        elif self.gamma < 0.1 * self.omega_b:
            return "underdamped"
        else:
            return "intermediate"
