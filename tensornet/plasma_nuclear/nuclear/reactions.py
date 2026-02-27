"""
Nuclear Reactions — optical model, R-matrix theory, statistical (Hauser-Feshbach),
compound nucleus, direct reactions.

Domain X.2 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Optical Model Potential
# ---------------------------------------------------------------------------

class OpticalModelPotential:
    r"""
    Optical model for nucleon-nucleus scattering.

    $$U(r) = -V_R f(r, R_V, a_V) -iW_V f(r, R_W, a_W)
      + 4ia_D W_D\frac{d}{dr}f(r, R_D, a_D)
      + V_{so}\frac{1}{r}\frac{df}{dr}\hat{\mathbf{l}}\cdot\hat{\mathbf{s}}
      + V_C(r)$$

    where $f(r,R,a) = 1/(1+\exp((r-R)/a))$ is the Woods-Saxon form factor.

    Parameters: energy-dependent global parametrisations (e.g., Koning-Delaroche).
    """

    def __init__(self, A_target: int = 40, Z_target: int = 20,
                 E_lab: float = 10.0) -> None:
        self.A = A_target
        self.Z = Z_target
        self.E = E_lab  # MeV
        self.R0 = 1.25 * A_target**(1 / 3)

        # Koning-Delaroche-like parametrisation
        self.V_R = 52.9 - 0.299 * E_lab  # Real volume depth (MeV)
        self.W_V = max(0.0, 0.4 * (E_lab - 10))  # Volume imaginary
        self.W_D = 11.5 * math.exp(-0.01 * E_lab)  # Surface imaginary
        self.V_so = 6.2  # Spin-orbit (MeV)

        self.a_V = 0.65
        self.a_W = 0.65
        self.a_D = 0.65
        self.a_so = 0.59

        self.R_V = 1.25 * A_target**(1 / 3)
        self.R_W = 1.25 * A_target**(1 / 3)
        self.R_D = 1.25 * A_target**(1 / 3)
        self.R_so = 1.10 * A_target**(1 / 3)
        self.R_C = 1.25 * A_target**(1 / 3)

    @staticmethod
    def woods_saxon(r: NDArray, R: float, a: float) -> NDArray:
        """Woods-Saxon form factor."""
        return 1.0 / (1.0 + np.exp((r - R) / a))

    @staticmethod
    def ws_derivative(r: NDArray, R: float, a: float) -> NDArray:
        """Derivative of Woods-Saxon."""
        e = np.exp((r - R) / a)
        return -e / (a * (1 + e)**2)

    def coulomb_potential(self, r: NDArray) -> NDArray:
        """Coulomb potential (uniform sphere)."""
        V_C = np.zeros_like(r)
        inside = r <= self.R_C
        outside = ~inside
        V_C[inside] = (self.Z * 1.44 / (2 * self.R_C)
                      * (3 - (r[inside] / self.R_C)**2))
        V_C[outside] = self.Z * 1.44 / r[outside]
        return V_C

    def total_potential(self, r: NDArray, l: int = 0,
                          j: float = 0.5) -> NDArray:
        """Full optical potential U(r) + V_C(r)."""
        # Real volume
        U_real = -self.V_R * self.woods_saxon(r, self.R_V, self.a_V)

        # Imaginary volume
        U_imag_vol = -self.W_V * self.woods_saxon(r, self.R_W, self.a_W)

        # Surface imaginary
        U_imag_surf = (-4 * self.a_D * self.W_D
                      * self.ws_derivative(r, self.R_D, self.a_D))

        # Spin-orbit
        ls = 0.5 * (j * (j + 1) - l * (l + 1) - 0.75)
        U_so = (self.V_so * self.ws_derivative(r, self.R_so, self.a_so)
                * ls / np.maximum(r, 1e-6))

        return U_real + U_imag_vol + U_imag_surf + U_so + self.coulomb_potential(r)


# ---------------------------------------------------------------------------
#  R-Matrix Theory
# ---------------------------------------------------------------------------

class RMatrixSolver:
    r"""
    R-matrix theory for nuclear reactions (Wigner & Eisenbud, 1947).

    $$R(\ell, E) = \sum_\lambda \frac{\gamma_{\lambda c}^2}{E_\lambda - E}$$

    where $E_\lambda$ = resonance energy, $\gamma_{\lambda c}$ = reduced width amplitude.

    Cross section:
    $$\sigma_{\alpha\beta} = \frac{\pi}{k^2}\sum_J (2J+1)
      |S^J_{\alpha\beta} - \delta_{\alpha\beta}|^2$$
    """

    @dataclass
    class Resonance:
        energy: float          # E_λ (MeV)
        width: float           # Γ = 2Pγ² (MeV)
        reduced_width: float   # γ² (MeV)
        spin: float            # J
        parity: int            # π = ±1

    def __init__(self, channel_radius: float = 5.0) -> None:
        self.a = channel_radius  # fm
        self.resonances: List[RMatrixSolver.Resonance] = []

    def add_resonance(self, E: float, gamma2: float, J: float = 0.5,
                         parity: int = 1, width: Optional[float] = None) -> None:
        """Add a resonance level."""
        Gamma = width if width is not None else 2 * gamma2
        self.resonances.append(self.Resonance(E, Gamma, gamma2, J, parity))

    def r_matrix(self, E: float) -> complex:
        """R(E) = Σ_λ γ²/(E_λ − E)."""
        R = 0.0 + 0.0j
        for res in self.resonances:
            R += res.reduced_width / (res.energy - E + 1e-30)
        return R

    def collision_matrix(self, E: float, l: int = 0,
                            penetrability: Optional[float] = None) -> complex:
        """Collision (S) matrix from R-matrix.

        S = exp(2iφ)(1 − 2iP R)/(1 + iP R) (single-channel, single-level).
        """
        if penetrability is None:
            k = math.sqrt(2 * 939.0 * abs(E)) / 197.3 if E > 0 else 0.01
            penetrability = k * self.a  # s-wave P_l ≈ ka

        R = self.r_matrix(E)
        S = (1 - 1j * penetrability * R) / (1 + 1j * penetrability * R)
        return S

    def cross_section(self, energies: NDArray, l: int = 0) -> NDArray:
        """Elastic scattering cross section σ(E).

        σ = (π/k²) |1 − S|²
        """
        sigma = np.zeros(len(energies))
        for ie, E in enumerate(energies):
            if E <= 0:
                continue
            k = math.sqrt(2 * 939.0 * E) / 197.3
            S = self.collision_matrix(E, l)
            sigma[ie] = math.pi / k**2 * abs(1 - S)**2 * 10  # fm² → mb
        return sigma

    def breit_wigner(self, E: float, Er: float, Gamma: float,
                        J: float, k: float) -> float:
        """Breit-Wigner single-resonance cross section.

        σ = (π/k²)(2J+1) Γ²/4 / ((E−E_r)² + Γ²/4)
        """
        return (math.pi / k**2 * (2 * J + 1) * Gamma**2 / 4
                / ((E - Er)**2 + Gamma**2 / 4)) * 10


# ---------------------------------------------------------------------------
#  Hauser-Feshbach Statistical Model
# ---------------------------------------------------------------------------

class HauserFeshbach:
    r"""
    Hauser-Feshbach statistical model for compound nucleus reactions.

    $$\sigma_{\alpha\beta}(E) = \frac{\pi}{k_\alpha^2}
      \sum_J(2J+1)\frac{T_\alpha^J T_\beta^J}{\sum_\gamma T_\gamma^J}$$

    where $T^J_c$ = transmission coefficient for channel c at spin J.

    Level density (back-shifted Fermi gas):
    $$\rho(E^*) = \frac{\sqrt{\pi}}{12}\frac{\exp(2\sqrt{aU})}{a^{1/4}U^{5/4}}$$
    where $U = E^* - \Delta$, $a \approx A/8$ MeV⁻¹.
    """

    def __init__(self, A_compound: int = 41, E_excitation: float = 10.0) -> None:
        self.A = A_compound
        self.Ex = E_excitation
        self.a = A_compound / 8.0  # level density parameter (MeV⁻¹)
        self.delta = 12.0 / math.sqrt(A_compound)  # pairing
        self.channels: Dict[str, NDArray] = {}

    def level_density(self, Ex: float) -> float:
        """Back-shifted Fermi gas level density ρ(E*)."""
        U = Ex - self.delta
        if U <= 0:
            return 1e-10
        return (math.sqrt(math.pi) / 12
                * math.exp(2 * math.sqrt(self.a * U))
                / (self.a**0.25 * U**1.25))

    def transmission_coefficient(self, E: float, l: int = 0,
                                     V_barrier: float = 0.0) -> float:
        """Hill-Wheeler transmission through barrier.

        T_l(E) = 1 / (1 + exp(2π(V_l − E)/ℏω))
        """
        hw = 1.0  # MeV, curvature
        return 1.0 / (1.0 + math.exp(2 * math.pi * (V_barrier - E) / hw))

    def set_channel(self, name: str, T_values: NDArray) -> None:
        """Set transmission coefficients for a decay channel."""
        self.channels[name] = T_values

    def partial_cross_section(self, T_entrance: NDArray,
                                 T_exit: NDArray,
                                 T_total: NDArray,
                                 k: float, J_max: int = 10) -> float:
        """σ_αβ = (π/k²) Σ_J (2J+1) T_α T_β / T_tot."""
        sigma = 0.0
        for J in range(J_max):
            if J < len(T_entrance) and J < len(T_exit) and J < len(T_total):
                if T_total[J] < 1e-30:
                    continue
                sigma += (2 * J + 1) * T_entrance[J] * T_exit[J] / T_total[J]
        return math.pi / k**2 * sigma * 10  # mb

    def evaporation_spectrum(self, E_ex: float, channel_mass: float = 1.0,
                                n_bins: int = 100) -> Tuple[NDArray, NDArray]:
        """Weisskopf evaporation spectrum.

        dσ/dE ∝ E σ_inv(E) ρ(E* − S − E)
        where S = separation energy.
        """
        S = 8.0  # MeV (typical separation energy)
        E_max = E_ex - S
        if E_max <= 0:
            return np.zeros(1), np.zeros(1)

        E_out = np.linspace(0.1, E_max, n_bins)
        spectrum = np.zeros(n_bins)

        for i, E in enumerate(E_out):
            rho = self.level_density(E_ex - S - E)
            sigma_inv = E  # geometric ∝ E for neutrons
            spectrum[i] = E * sigma_inv * rho

        if np.max(spectrum) > 0:
            spectrum /= np.max(spectrum)

        return E_out, spectrum


# ---------------------------------------------------------------------------
#  Direct Nuclear Reactions (DWBA)
# ---------------------------------------------------------------------------

class DWBATransfer:
    r"""
    Distorted-Wave Born Approximation for direct nuclear reactions.

    Transfer amplitude:
    $$T_{fi} = \langle\chi_f^{(-)}|V|A\rangle \otimes |\chi_i^{(+)}\rangle$$

    Angular distribution for (d,p) stripping:
    $$\frac{d\sigma}{d\Omega} = \frac{\mu_i\mu_f}{(2\pi\hbar^2)^2}
      \frac{k_f}{k_i}S_{lj}|T_{fi}|^2$$

    where S_{lj} = spectroscopic factor.
    """

    def __init__(self, mass_projectile: float = 2.0,
                 mass_target: float = 40.0,
                 E_lab: float = 20.0) -> None:
        self.m_p = mass_projectile  # amu
        self.m_t = mass_target
        self.E = E_lab  # MeV
        self.mu = mass_projectile * mass_target / (mass_projectile + mass_target) * 931.5

    def momentum_transfer(self, theta: float, Q: float = 0.0) -> float:
        """Momentum transfer q at angle θ."""
        E_cm = self.E * self.m_t / (self.m_p + self.m_t)
        ki = math.sqrt(2 * self.mu * E_cm) / 197.3
        kf = math.sqrt(2 * self.mu * (E_cm + Q)) / 197.3 if E_cm + Q > 0 else 0.0
        q = math.sqrt(ki**2 + kf**2 - 2 * ki * kf * math.cos(theta))
        return q

    def form_factor(self, q: float, R: float = 5.0) -> float:
        """Nuclear form factor |F(q)|² ≈ [3j₁(qR)/(qR)]²."""
        x = q * R
        if x < 1e-6:
            return 1.0
        j1 = math.sin(x) / x**2 - math.cos(x) / x
        return (3 * j1 / x)**2

    def angular_distribution(self, l_transfer: int,
                                theta_array: NDArray,
                                spectroscopic_factor: float = 1.0,
                                Q: float = 0.0) -> NDArray:
        """dσ/dΩ for a single-nucleon transfer with angular momentum l."""
        dsigma = np.zeros(len(theta_array))
        for it, theta in enumerate(theta_array):
            q = self.momentum_transfer(theta, Q)
            ff = self.form_factor(q)
            dsigma[it] = spectroscopic_factor * ff * (2 * l_transfer + 1)
        return dsigma
