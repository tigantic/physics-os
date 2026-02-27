"""
Kondo & Impurity Physics — Anderson impurity model, NRG, CT-QMC,
Kondo temperature extraction.

Domain VII.9 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Anderson Impurity Model
# ---------------------------------------------------------------------------

class AndersonImpurityModel:
    r"""
    Single-impurity Anderson model (SIAM):

    $$H = \sum_{k\sigma}\varepsilon_k c_{k\sigma}^\dagger c_{k\sigma}
      + \sum_\sigma \varepsilon_d d_\sigma^\dagger d_\sigma
      + U n_{d\uparrow}n_{d\downarrow}
      + \sum_{k\sigma}(V_k c_{k\sigma}^\dagger d_\sigma + \text{h.c.})$$

    Hybridisation function:
    $$\Delta(\omega) = \pi\sum_k |V_k|^2\delta(\omega-\varepsilon_k)$$

    For flat band: $\Delta(\omega) = \pi\rho_0 V^2 \approx \text{const}$ (wide-band limit).

    Atomic limit (V=0): 4 states with energies 0, ε_d, ε_d, 2ε_d + U.
    """

    def __init__(self, eps_d: float = -2.0, U: float = 4.0,
                 V: float = 0.5, half_bandwidth: float = 5.0,
                 n_bath: int = 20) -> None:
        self.eps_d = eps_d
        self.U = U
        self.V = V
        self.D = half_bandwidth
        self.n_bath = n_bath

        # Discretised bath levels
        self.eps_k = np.linspace(-self.D, self.D, n_bath)
        self.V_k = np.ones(n_bath) * V / math.sqrt(n_bath)

    @property
    def hybridisation_width(self) -> float:
        """Γ = πρ₀V² (half-width at half-max of impurity spectral function)."""
        rho0 = 1.0 / (2 * self.D)
        return math.pi * rho0 * self.V**2

    def kondo_temperature(self) -> float:
        r"""Kondo temperature (Haldane formula):

        $$T_K = \sqrt{\Gamma U/2}\exp\!\left(\frac{\pi\varepsilon_d(\varepsilon_d+U)}{2\Gamma U}\right)$$

        Valid in Kondo regime: $-U < \varepsilon_d < 0$, $\Gamma \ll U$.
        """
        Gamma = self.hybridisation_width
        ed = self.eps_d
        U = self.U
        if Gamma <= 0 or U <= 0:
            return 0.0
        prefactor = math.sqrt(Gamma * U / 2)
        exponent = math.pi * ed * (ed + U) / (2 * Gamma * U)
        return prefactor * math.exp(exponent)

    def schrieffer_wolff_J(self) -> float:
        """Effective Kondo exchange coupling from Schrieffer-Wolff:

        J = 2V²[1/|ε_d| + 1/(ε_d + U)].
        """
        ed = self.eps_d
        U = self.U
        return 2 * self.V**2 * (1.0 / abs(ed) + 1.0 / (ed + U))

    def atomic_states(self) -> dict:
        """Atomic limit (V=0) energies and occupancies."""
        return {
            'empty': {'energy': 0.0, 'n': 0},
            'up': {'energy': self.eps_d, 'n': 1},
            'down': {'energy': self.eps_d, 'n': 1},
            'double': {'energy': 2 * self.eps_d + self.U, 'n': 2},
        }

    def mean_field_occupation(self, T: float = 0.01) -> float:
        """Hartree-Fock mean-field impurity occupation."""
        Gamma = self.hybridisation_width
        # Self-consistent: ε_eff = ε_d + U<n_σ>
        n_sigma = 0.5  # start from half-filling
        for _ in range(100):
            eps_eff = self.eps_d + self.U * n_sigma
            # Lorentzian DOS: n = ½ − (1/π)arctan(ε_eff/Γ)
            n_new = 0.5 - math.atan(eps_eff / Gamma) / math.pi
            if abs(n_new - n_sigma) < 1e-8:
                break
            n_sigma = 0.5 * n_sigma + 0.5 * n_new
        return 2 * n_sigma  # total n = 2*n_sigma


# ---------------------------------------------------------------------------
#  Numerical Renormalization Group (NRG) — Wilson Chain
# ---------------------------------------------------------------------------

class WilsonChainNRG:
    r"""
    Wilson's Numerical Renormalization Group for SIAM.

    Logarithmic discretisation with parameter Λ > 1.
    Bath energies form a geometric series:
    $$\varepsilon_n \sim \Lambda^{-n/2}$$

    Hopping integrals:
    $$t_n \approx \frac{(1+\Lambda^{-1})D}{2\sqrt{\ln\Lambda}}
      \frac{\Lambda^{-n/2}(1-\Lambda^{-(n+1)})}
           {\sqrt{(1-\Lambda^{-(2n+1)})(1-\Lambda^{-(2n+3)})}}$$
    """

    def __init__(self, Lambda: float = 2.0, n_sites: int = 30,
                 half_bandwidth: float = 1.0) -> None:
        self.Lambda = Lambda
        self.n_sites = n_sites
        self.D = half_bandwidth

        self.t_n = self._compute_hoppings()

    def _compute_hoppings(self) -> NDArray:
        """Wilson chain hopping parameters."""
        L = self.Lambda
        t = np.zeros(self.n_sites)

        for n in range(self.n_sites):
            num = L**(-(n) / 2) * (1 - L**(-(n + 1)))
            den = math.sqrt((1 - L**(-(2 * n + 1))) * (1 - L**(-(2 * n + 3))))
            t[n] = (1 + 1 / L) * self.D / (2 * math.sqrt(math.log(L))) * num / (den + 1e-30)

        return t

    def energy_scale(self, n: int) -> float:
        """Energy scale at iteration n: ω_n ~ Λ^{−(n−1)/2}."""
        return self.D * self.Lambda**(-(n - 1) / 2)

    def iterative_diagonalise(self, eps_d: float, U: float,
                                 Gamma: float, n_keep: int = 300) -> NDArray:
        """Simplified NRG iteration (spinless for demonstration).

        Real NRG keeps full spin sectors and truncates.
        Returns approximate spectrum at each step.
        """
        # Initial: impurity 2 states |0>, |1>
        H = np.array([[0, 0], [0, eps_d]])

        spectra = []
        dim = 2

        for n in range(min(self.n_sites, 15)):
            # Add site n: double the Hilbert space
            dim_new = dim * 2
            H_new = np.zeros((dim_new, dim_new))

            # Old block
            H_new[:dim, :dim] = H * self.Lambda**0.5  # rescale

            # New site energy
            for i in range(dim):
                H_new[dim + i, dim + i] = 0  # new site on-site = 0

            # Hopping
            # Simplified: connect last site to new site
            for i in range(dim):
                H_new[i, dim + i] = self.t_n[n]
                H_new[dim + i, i] = self.t_n[n]

            # Truncate if too large
            if dim_new > n_keep:
                evals, evecs = np.linalg.eigh(H_new)
                H_new = np.diag(evals[:n_keep])
                dim_new = n_keep

            H = H_new
            dim = dim_new

            evals = np.linalg.eigvalsh(H)
            spectra.append(evals[:min(20, len(evals))])

        return np.array([s for s in spectra], dtype=object)


# ---------------------------------------------------------------------------
#  CT-QMC (Continuous-Time Quantum Monte Carlo) — Hybridisation Expansion
# ---------------------------------------------------------------------------

class CTQMC_HybridisationExpansion:
    r"""
    CT-QMC in the hybridisation expansion (CT-HYB).

    Partition function:
    $$Z = \sum_{k=0}^{\infty}\int d\tau_1\cdots d\tau_{2k}\,
      \det[\Delta(\tau_i-\tau_j')]\,\mathrm{Tr}[T_\tau\prod e^{-\Delta\tau H_{\text{loc}}}]$$

    Monte Carlo: propose insert/remove hybridisation segments.
    """

    def __init__(self, beta: float = 10.0, eps_d: float = -2.0,
                 U: float = 4.0, Gamma: float = 0.5) -> None:
        self.beta = beta
        self.eps_d = eps_d
        self.U = U
        self.Gamma = Gamma

    def hybridisation_function(self, tau: float) -> float:
        """Δ(τ) for flat DOS (wide-band limit).

        Δ(τ) = −(Γ/2) / sin(πτ/β) × (π/β) for 0 < τ < β.
        """
        if abs(tau) < 1e-10 or abs(tau - self.beta) < 1e-10:
            return -self.Gamma / 2 * math.pi / self.beta * 100  # regularised
        return -self.Gamma / 2 * math.pi / (self.beta * math.sin(math.pi * tau / self.beta))

    def local_weight(self, segments: list, sigma: int = 0) -> float:
        """Local trace weight for a segment configuration.

        segments: list of (tau_start, tau_end) pairs where impurity is occupied.
        """
        weight = 1.0
        for start, end in segments:
            length = end - start
            if length < 0:
                length += self.beta  # wrap around
            weight *= math.exp(-self.eps_d * length)
        return weight

    def run_sampling(self, n_mc: int = 10000, n_warmup: int = 2000,
                       seed: int = 42) -> dict:
        """Simple MC sampling of segment configurations.

        Returns average occupation and sign.
        """
        rng = np.random.default_rng(seed)
        segments: list = []
        occupation_sum = 0.0
        sign_sum = 0.0
        n_samples = 0

        for step in range(n_mc + n_warmup):
            # Propose: insert or remove a segment
            if len(segments) == 0 or rng.random() < 0.5:
                # Insert
                tau_start = rng.random() * self.beta
                length = rng.exponential(1.0 / (abs(self.eps_d) + 0.1))
                length = min(length, self.beta * 0.5)
                tau_end = tau_start + length

                new_segments = segments + [(tau_start, tau_end)]
                w_old = self.local_weight(segments)
                w_new = self.local_weight(new_segments)
                det_ratio = abs(self.hybridisation_function(length))

                ratio = w_new * det_ratio * self.beta / (w_old * (len(segments) + 1) + 1e-30)
                if rng.random() < min(1.0, abs(ratio)):
                    segments = new_segments
            else:
                # Remove random segment
                idx = rng.integers(len(segments))
                new_segments = segments[:idx] + segments[idx + 1:]
                w_old = self.local_weight(segments)
                w_new = self.local_weight(new_segments)

                ratio = w_new * len(segments) / (w_old * self.beta + 1e-30)
                if rng.random() < min(1.0, abs(ratio)):
                    segments = new_segments

            if step >= n_warmup:
                total_occ = sum(((e - s) % self.beta) for s, e in segments) / self.beta
                occupation_sum += total_occ
                sign_sum += 1.0
                n_samples += 1

        avg_n = occupation_sum / (n_samples + 1e-30)
        return {
            'occupation': avg_n,
            'average_sign': sign_sum / (n_samples + 1e-30),
            'expansion_order': len(segments),
            'n_samples': n_samples,
        }


# ---------------------------------------------------------------------------
#  Kondo Temperature Extraction
# ---------------------------------------------------------------------------

class KondoTemperatureExtractor:
    r"""
    Methods to extract T_K from spectral/thermodynamic data.

    1. From impurity susceptibility: $\chi_{\text{imp}}(T=T_K) = 0.0701/(T_K)$
    2. From Wilson ratio: $R_W = \frac{4\pi^2}{3}\frac{\chi_{\text{imp}}}{(\gamma_{\text{imp}})} = 2$
    3. From spectral function: half-width of Kondo peak = T_K
    """

    @staticmethod
    def from_susceptibility(chi_data: NDArray, T_data: NDArray) -> float:
        """Extract T_K from χ_imp(T) = 1/(4T_K) at T ≫ T_K.

        Fit Tχ(T) and find where it reaches universal value ≈ 0.25.
        """
        T_chi = T_data * chi_data
        # T_K ~ T where Tχ ≈ 0.0701
        target = 0.0701
        idx = np.argmin(np.abs(T_chi - target))
        return float(T_data[idx])

    @staticmethod
    def from_spectral_width(omega: NDArray, A_omega: NDArray) -> float:
        """Extract T_K from HWHM of Kondo resonance peak at ω=0."""
        # Find peak near ω=0
        idx_zero = np.argmin(np.abs(omega))
        peak = A_omega[idx_zero]
        half_max = peak / 2

        # Search for HWHM on positive ω side
        for i in range(idx_zero, len(omega)):
            if A_omega[i] < half_max:
                return float(abs(omega[i]))
        return float(abs(omega[-1]))

    @staticmethod
    def wilson_ratio(chi_imp: float, gamma_imp: float) -> float:
        """Wilson ratio R_W = (4π²/3)(χ_imp/γ_imp).

        R_W = 2 for Kondo impurity (universal).
        R_W = 1 for free electron gas.
        """
        return (4 * math.pi**2 / 3) * chi_imp / (gamma_imp + 1e-30)

    @staticmethod
    def bethe_ansatz_susceptibility(T: float, T_K: float) -> float:
        """Exact Bethe ansatz susceptibility for T ≪ T_K.

        χ(T) ≈ 1/(4T_K)[1 − (πT/T_K)² + ...]
        """
        x = math.pi * T / T_K
        return 1.0 / (4 * T_K) * (1 - x**2 + 0.5 * x**4)
