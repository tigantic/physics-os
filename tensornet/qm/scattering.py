"""
Quantum Scattering Theory — Partial-wave analysis, Born approximation,
T-matrix, R-matrix, Breit-Wigner resonance.

Domain VI.3 — NEW.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Partial-Wave T-Matrix
# ---------------------------------------------------------------------------

class PartialWaveScattering:
    r"""
    Partial-wave decomposition of scattering amplitude.

    $$f(\theta) = \sum_l (2l+1) f_l P_l(\cos\theta)$$

    Phase shift $\delta_l$ from T-matrix:
    $$T_l = \frac{e^{2i\delta_l}-1}{2ik} = \frac{e^{i\delta_l}\sin\delta_l}{k}$$

    Total cross section (optical theorem):
    $$\sigma_{\text{tot}} = \frac{4\pi}{k^2}\sum_l(2l+1)\sin^2\delta_l$$
    """

    def __init__(self, k: float = 1.0, l_max: int = 10) -> None:
        self.k = k
        self.l_max = l_max
        self.delta = np.zeros(l_max + 1)  # phase shifts

    def set_phase_shifts(self, deltas: NDArray) -> None:
        n = min(len(deltas), self.l_max + 1)
        self.delta[:n] = deltas[:n]

    def t_matrix_element(self, l: int) -> complex:
        """T_l = e^{iδ_l}sin(δ_l)/k."""
        d = self.delta[l]
        return np.exp(1j * d) * math.sin(d) / self.k

    def s_matrix_element(self, l: int) -> complex:
        """S_l = e^{2iδ_l}."""
        return np.exp(2j * self.delta[l])

    def differential_cross_section(self, theta: NDArray) -> NDArray:
        """dσ/dΩ = |f(θ)|²."""
        f = np.zeros(len(theta), dtype=complex)
        cos_theta = np.cos(theta)
        for l in range(self.l_max + 1):
            Pl = self._legendre(l, cos_theta)
            f += (2 * l + 1) * self.t_matrix_element(l) * Pl
        return np.abs(f)**2

    def total_cross_section(self) -> float:
        """σ_tot = (4π/k²) Σ (2l+1) sin²δ_l."""
        sigma = 0.0
        for l in range(self.l_max + 1):
            sigma += (2 * l + 1) * math.sin(self.delta[l])**2
        return 4 * math.pi / self.k**2 * sigma

    def transport_cross_section(self) -> float:
        """σ_tr = (4π/k²) Σ (l+1) sin²(δ_l − δ_{l+1})."""
        sigma = 0.0
        for l in range(self.l_max):
            sigma += (l + 1) * math.sin(self.delta[l] - self.delta[l + 1])**2
        return 4 * math.pi / self.k**2 * sigma

    @staticmethod
    def _legendre(l: int, x: NDArray) -> NDArray:
        """Legendre polynomial P_l(x) via recursion."""
        if l == 0:
            return np.ones_like(x)
        elif l == 1:
            return x.copy()
        P_prev = np.ones_like(x)
        P_curr = x.copy()
        for n in range(2, l + 1):
            P_next = ((2 * n - 1) * x * P_curr - (n - 1) * P_prev) / n
            P_prev = P_curr
            P_curr = P_next
        return P_curr


# ---------------------------------------------------------------------------
#  Born Approximation
# ---------------------------------------------------------------------------

class BornApproximation:
    r"""
    First Born approximation for scattering:

    $$f^{(1)}(\mathbf{q}) = -\frac{m}{2\pi\hbar^2}\int V(\mathbf{r})e^{-i\mathbf{q}\cdot\mathbf{r}}d^3r$$

    For central potential: f(θ) depends only on $q = 2k\sin(\theta/2)$.

    Yukawa: $V(r) = V_0 e^{-\mu r}/r$
    → $f^{(1)} = -\frac{2mV_0}{\hbar^2(q^2+\mu^2)}$

    Coulomb: $V(r) = Zz e^2/(4πε_0 r)$
    → Rutherford formula.
    """

    def __init__(self, mass: float = 1.0, hbar: float = 1.0) -> None:
        self.m = mass
        self.hbar = hbar

    def yukawa_amplitude(self, k: float, theta: float,
                           V0: float = 1.0, mu: float = 1.0) -> complex:
        """Born amplitude for Yukawa potential."""
        q = 2 * k * math.sin(theta / 2)
        return -2 * self.m * V0 / (self.hbar**2 * (q**2 + mu**2))

    def coulomb_rutherford(self, k: float, theta: float,
                              Z1: float = 1.0, Z2: float = 1.0) -> float:
        """Rutherford cross section (exact for pure Coulomb).

        dσ/dΩ = (a/(4k²sin²(θ/2)))²
        where a = 2mZ₁Z₂e²/(4πε₀ℏ²).
        """
        # a = Z1*Z2*e²m / (2ε₀ℏ²) — in natural units e²/4πε₀ = 1
        a = Z1 * Z2 * self.m / self.hbar**2
        sin_half = math.sin(theta / 2)
        if sin_half < 1e-10:
            return float('inf')
        return (a / (4 * k**2 * sin_half**2))**2

    def born_series_convergence(self, k: float, V0: float,
                                   a: float) -> float:
        """Born parameter: β = 2mV₀a²/ℏ² (must be ≪ 1 for validity)."""
        return 2 * self.m * abs(V0) * a**2 / self.hbar**2


# ---------------------------------------------------------------------------
#  R-Matrix Method
# ---------------------------------------------------------------------------

class RMatrixScattering:
    r"""
    R-matrix method: solve Schrödinger equation inside a box r < a,
    then match to asymptotic.

    $$R_l(E) = \sum_\lambda \frac{\gamma_\lambda^2}{E_\lambda - E}$$

    $R$ → $S$: $S_l = e^{-2ik_l a}\frac{1+ik_l a R_l}{1-ik_l a R_l}$

    Useful for nuclear reactions and electron-atom scattering.
    """

    def __init__(self, a_boundary: float = 10.0, n_basis: int = 20) -> None:
        self.a = a_boundary
        self.n_basis = n_basis

        self.E_lambda = np.zeros(n_basis)  # pole energies
        self.gamma2 = np.zeros(n_basis)    # reduced widths squared

    def set_poles(self, energies: NDArray, widths: NDArray) -> None:
        n = min(len(energies), self.n_basis)
        self.E_lambda[:n] = energies[:n]
        self.gamma2[:n] = widths[:n]

    def r_matrix(self, E: float, l: int = 0) -> float:
        """R_l(E) = Σ γ²_λ / (E_λ − E)."""
        R = 0.0
        for i in range(self.n_basis):
            if abs(self.E_lambda[i] - E) > 1e-15:
                R += self.gamma2[i] / (self.E_lambda[i] - E)
        return R

    def s_matrix(self, E: float, k: float, l: int = 0) -> complex:
        """S_l from R-matrix."""
        R = self.r_matrix(E, l)
        ka = k * self.a
        return np.exp(-2j * ka) * (1 + 1j * ka * R) / (1 - 1j * ka * R)

    def phase_shift(self, E: float, k: float, l: int = 0) -> float:
        """Extract phase shift from S-matrix."""
        S = self.s_matrix(E, k, l)
        return float(np.angle(S)) / 2

    def cross_section(self, E: float, k: float, l: int = 0) -> float:
        """Partial cross section from R-matrix."""
        S = self.s_matrix(E, k, l)
        return math.pi / k**2 * (2 * l + 1) * abs(1 - S)**2


# ---------------------------------------------------------------------------
#  Breit-Wigner Resonance
# ---------------------------------------------------------------------------

class BreitWignerResonance:
    r"""
    Breit-Wigner resonance cross section:

    $$\sigma_l(E) = \frac{\pi}{k^2}\frac{(2l+1)\Gamma_l^2}{(E-E_r)^2+\Gamma^2/4}$$

    Peak cross section: $\sigma_{\max} = \frac{4\pi}{k_r^2}(2l+1)$ (unitary limit).

    Multi-channel: $\sigma_{ab} \propto \Gamma_a\Gamma_b / [(E-E_r)^2 + Γ²/4]$.
    """

    def __init__(self, Er: float = 1.0, Gamma: float = 0.1,
                 l: int = 0, mass: float = 1.0) -> None:
        self.Er = Er
        self.Gamma = Gamma
        self.l = l
        self.m = mass

    def cross_section(self, E: NDArray) -> NDArray:
        """Breit-Wigner cross section σ(E)."""
        k = np.sqrt(2 * self.m * E + 1e-30)
        return (math.pi / k**2 * (2 * self.l + 1)
                * self.Gamma**2 / ((E - self.Er)**2 + self.Gamma**2 / 4))

    def phase_shift(self, E: NDArray) -> NDArray:
        """Phase shift through resonance: δ_l = arctan(Γ/(2(E_r−E)))."""
        return np.arctan2(self.Gamma / 2, self.Er - E)

    def time_delay(self, E: NDArray) -> NDArray:
        """Wigner time delay: Δt = 2ℏ dδ/dE."""
        return self.Gamma / ((E - self.Er)**2 + self.Gamma**2 / 4)

    def lifetime(self) -> float:
        """Resonance lifetime τ = ℏ/Γ."""
        return 1.0 / self.Gamma  # ℏ = 1

    def multichannel(self, E: NDArray, Gamma_a: float,
                       Gamma_b: float) -> NDArray:
        """Multi-channel: σ_{a→b} = π/k² g Γ_a Γ_b / [(E−E_r)² + Γ²/4]."""
        k = np.sqrt(2 * self.m * E + 1e-30)
        g = (2 * self.l + 1)  # statistical factor
        return (math.pi / k**2 * g * Gamma_a * Gamma_b
                / ((E - self.Er)**2 + self.Gamma**2 / 4))

    def fano_profile(self, E: NDArray, q: float = 2.0) -> NDArray:
        """Fano resonance profile: σ ∝ (q+ε)²/(1+ε²), ε=(E−Er)/(Γ/2)."""
        eps = (E - self.Er) / (self.Gamma / 2)
        return (q + eps)**2 / (1 + eps**2)
