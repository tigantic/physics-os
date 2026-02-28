"""
Quantum Optics — Jaynes-Cummings model, photon statistics, squeezed states,
Hong-Ou-Mandel effect, quantum state tomography.

Domain IV.2 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

HBAR: float = 1.055e-34   # J·s
C_LIGHT: float = 2.998e8  # m/s


# ---------------------------------------------------------------------------
#  Jaynes-Cummings Model
# ---------------------------------------------------------------------------

class JaynesCummingsModel:
    r"""
    Jaynes-Cummings model: single two-level atom + single cavity mode.

    $$\hat{H}_{\text{JC}} = \hbar\omega_c\hat{a}^\dagger\hat{a}
      + \frac{\hbar\omega_a}{2}\hat{\sigma}_z
      + \hbar g(\hat{a}\hat{\sigma}_+ + \hat{a}^\dagger\hat{\sigma}_-)$$

    Vacuum Rabi splitting: $\Omega_n = 2g\sqrt{n+1}$

    Dressed states:
    $$|+,n\rangle = \cos\theta_n|e,n\rangle + \sin\theta_n|g,n+1\rangle$$
    $$|-,n\rangle = -\sin\theta_n|e,n\rangle + \cos\theta_n|g,n+1\rangle$$

    $\tan(2\theta_n) = 2g\sqrt{n+1}/\Delta$, $\Delta = \omega_a - \omega_c$.
    """

    def __init__(self, omega_a: float = 1.0, omega_c: float = 1.0,
                 g: float = 0.1, n_max: int = 20) -> None:
        """
        omega_a, omega_c: atom/cavity frequency (normalised units).
        g: coupling strength.
        n_max: Fock space truncation.
        """
        self.omega_a = omega_a
        self.omega_c = omega_c
        self.g = g
        self.n_max = n_max
        self.detuning = omega_a - omega_c

    def dressed_energies(self, n: int) -> Tuple[float, float]:
        """Dressed-state energies E±,n.

        E±,n = ℏω_c(n+1/2) ± (1/2)ℏΩ_n
        """
        Omega_n = math.sqrt(self.detuning**2 + 4 * self.g**2 * (n + 1))
        base = self.omega_c * (n + 0.5)
        return base + 0.5 * Omega_n, base - 0.5 * Omega_n

    def rabi_frequency(self, n: int) -> float:
        """Generalised Rabi frequency Ω_n."""
        return math.sqrt(self.detuning**2 + 4 * self.g**2 * (n + 1))

    def hamiltonian(self) -> NDArray:
        """Build full Hamiltonian in |e,n⟩, |g,n+1⟩ basis.

        Size: 2(n_max + 1).
        """
        dim = 2 * (self.n_max + 1)
        H = np.zeros((dim, dim))

        for n in range(self.n_max + 1):
            i_e = 2 * n
            i_g = 2 * n + 1

            # Diagonal: |e,n⟩
            H[i_e, i_e] = self.omega_c * n + 0.5 * self.omega_a
            # Diagonal: |g,n+1⟩
            if n < self.n_max:
                H[i_g, i_g] = self.omega_c * (n + 1) - 0.5 * self.omega_a

            # Coupling
            coupling = self.g * math.sqrt(n + 1)
            if i_g < dim:
                H[i_e, i_g] = coupling
                H[i_g, i_e] = coupling

        return H

    def collapse_revival(self, n_bar: float, t: NDArray) -> NDArray:
        """Atomic inversion ⟨σ_z(t)⟩ for coherent state |α⟩ with ⟨n⟩ = n_bar.

        W(t) = Σ_n P(n) cos(2g√(n+1) t)
        """
        W = np.zeros_like(t)
        for n in range(self.n_max + 1):
            P_n = math.exp(-n_bar) * n_bar**n / math.factorial(n)
            W += P_n * np.cos(2 * self.g * math.sqrt(n + 1) * t)
        return W


# ---------------------------------------------------------------------------
#  Photon Statistics
# ---------------------------------------------------------------------------

class PhotonStatistics:
    r"""
    Photon number statistics for quantum light.

    Coherent state: $P(n) = e^{-|\alpha|^2}\frac{|\alpha|^{2n}}{n!}$, $g^{(2)} = 1$

    Thermal state: $P(n) = \frac{\bar{n}^n}{(\bar{n}+1)^{n+1}}$, $g^{(2)} = 2$

    Fock state: $P(n) = \delta_{n,N}$, $g^{(2)} = 1 - 1/N$

    Second-order correlation: $g^{(2)}(0) = \frac{\langle n(n-1)\rangle}{\langle n\rangle^2}$
    """

    def coherent_distribution(self, alpha: complex,
                                 n_max: int = 30) -> NDArray:
        """P(n) for coherent state |α⟩."""
        n_bar = abs(alpha)**2
        ns = np.arange(n_max)
        P = np.array([math.exp(-n_bar) * n_bar**n / math.factorial(n)
                      for n in ns])
        return P

    def thermal_distribution(self, n_bar: float,
                                n_max: int = 30) -> NDArray:
        """P(n) for thermal state."""
        ns = np.arange(n_max)
        P = np.array([n_bar**n / (n_bar + 1)**(n + 1) for n in ns])
        return P

    def fock_distribution(self, N: int, n_max: int = 30) -> NDArray:
        """P(n) for Fock state |N⟩."""
        P = np.zeros(n_max)
        if N < n_max:
            P[N] = 1.0
        return P

    def g2(self, P: NDArray) -> float:
        """g⁽²⁾(0) from photon number distribution."""
        ns = np.arange(len(P))
        n_mean = float(np.sum(ns * P))
        n_n1_mean = float(np.sum(ns * (ns - 1) * P))
        if n_mean < 1e-30:
            return 0.0
        return n_n1_mean / n_mean**2

    def mandel_Q(self, P: NDArray) -> float:
        """Mandel Q parameter: Q = (⟨Δn²⟩ − ⟨n⟩)/⟨n⟩.

        Q < 0: sub-Poissonian (non-classical).
        Q = 0: Poissonian (coherent).
        Q > 0: super-Poissonian (thermal, bunched).
        """
        ns = np.arange(len(P))
        n_mean = float(np.sum(ns * P))
        n2_mean = float(np.sum(ns**2 * P))
        var = n2_mean - n_mean**2
        if n_mean < 1e-30:
            return 0.0
        return (var - n_mean) / n_mean


# ---------------------------------------------------------------------------
#  Squeezed States
# ---------------------------------------------------------------------------

class SqueezedState:
    r"""
    Squeezed vacuum and squeezed coherent states.

    Squeeze operator: $\hat{S}(r,\phi) = \exp[\frac{r}{2}(e^{-2i\phi}\hat{a}^2 - e^{2i\phi}\hat{a}^{\dagger 2})]$

    Quadrature variances:
    $$(\Delta X_1)^2 = \frac{1}{4}e^{-2r}, \quad
      (\Delta X_2)^2 = \frac{1}{4}e^{2r}$$

    Photon number: $\langle n\rangle = \sinh^2 r + |\alpha|^2$
    """

    def __init__(self, r: float = 1.0, phi: float = 0.0,
                 alpha: complex = 0.0) -> None:
        """
        r: squeezing parameter.
        phi: squeezing angle.
        alpha: coherent displacement.
        """
        self.r = r
        self.phi = phi
        self.alpha = alpha

    def quadrature_variance(self) -> Tuple[float, float]:
        """(ΔX₁², ΔX₂²) — squeezed and anti-squeezed."""
        return 0.25 * math.exp(-2 * self.r), 0.25 * math.exp(2 * self.r)

    def mean_photon_number(self) -> float:
        """⟨n⟩ = sinh²(r) + |α|²."""
        return math.sinh(self.r)**2 + abs(self.alpha)**2

    def squeezing_dB(self) -> float:
        """Squeezing in dB: −10 log₁₀(e^{−2r})."""
        return 10 * 2 * self.r / math.log(10)

    def wigner_function(self, x: NDArray, p: NDArray) -> NDArray:
        """Wigner function W(x, p) for squeezed coherent state.

        W = (2/π) exp(−2e^{2r}(x−x₀)² − 2e^{−2r}(p−p₀)²)
        """
        x0 = math.sqrt(2) * self.alpha.real
        p0 = math.sqrt(2) * self.alpha.imag
        X, P = np.meshgrid(x, p, indexing='ij')
        return (2 / math.pi * np.exp(-2 * math.exp(2 * self.r) * (X - x0)**2
                                     - 2 * math.exp(-2 * self.r) * (P - p0)**2))


# ---------------------------------------------------------------------------
#  Hong-Ou-Mandel Effect
# ---------------------------------------------------------------------------

class HongOuMandel:
    r"""
    Hong-Ou-Mandel two-photon interference at a beam splitter.

    For identical photons at a 50:50 beam splitter:
    $$P_{\text{coinc}} = \frac{1}{2}(1 - |\langle\xi_1|\xi_2\rangle|^2)$$

    HOM dip visibility: $\mathcal{V} = 1 - P_{\text{coinc,min}}/P_{\text{coinc,max}}$

    For Gaussian wavepackets with temporal offset τ:
    $$P_{\text{coinc}}(\tau) = \frac{1}{2}\left(1 - e^{-\tau^2/(4\sigma_t^2)}\right)$$
    """

    def __init__(self, sigma_t: float = 100e-15) -> None:
        """
        sigma_t: pulse duration (seconds, FWHM/2.35).
        """
        self.sigma_t = sigma_t

    def coincidence_probability(self, tau: float) -> float:
        """P_coinc(τ) for identical Gaussian photons.

        tau: time delay (seconds).
        """
        return 0.5 * (1 - math.exp(-tau**2 / (4 * self.sigma_t**2)))

    def coincidence_curve(self, tau_max: float = 500e-15,
                             n_pts: int = 200) -> Tuple[NDArray, NDArray]:
        """Full HOM dip curve."""
        tau = np.linspace(-tau_max, tau_max, n_pts)
        P = 0.5 * (1 - np.exp(-tau**2 / (4 * self.sigma_t**2)))
        return tau * 1e15, P  # convert to fs

    def visibility(self, distinguishability: float = 0.0) -> float:
        """HOM visibility V = 1 − ε where ε is distinguishability.

        Perfect indistinguishability: V = 1 (complete dip).
        """
        return 1.0 - distinguishability

    def beam_splitter_transform(self, a_in: complex,
                                   b_in: complex) -> Tuple[complex, complex]:
        """50:50 beam splitter: c = (a + ib)/√2, d = (ia + b)/√2."""
        c = (a_in + 1j * b_in) / math.sqrt(2)
        d = (1j * a_in + b_in) / math.sqrt(2)
        return c, d
