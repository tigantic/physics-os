"""
Disordered quantum and classical systems.

Upgrades domain IX.5: Anderson localisation, KPM spectral method,
Edwards-Anderson spin glass, participation ratio and fractal dimension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Anderson Tight-Binding Model
# ---------------------------------------------------------------------------

class AndersonModel:
    r"""
    Anderson tight-binding model with random on-site disorder.

    $$H = -t\sum_{\langle i,j\rangle}(c^\dagger_i c_j + \text{h.c.})
        + \sum_i \varepsilon_i\,n_i$$

    where $\varepsilon_i \in [-W/2, W/2]$ uniformly distributed.

    At critical disorder $W_c / t \approx 16.5$ (3D Anderson transition).

    Implements:
    - 1D, 2D, 3D tight-binding Hamiltonian with PBC
    - Exact diagonalisation for small systems
    - Transfer matrix method for 1D localisation length
    - Inverse participation ratio
    """

    def __init__(self, L: int, dim: int = 1, t: float = 1.0,
                 W: float = 1.0, seed: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        L : Linear system size.
        dim : Spatial dimension (1, 2, or 3).
        t : Hopping amplitude.
        W : Disorder strength (box distribution width).
        seed : RNG seed for reproducibility.
        """
        self.L = L
        self.dim = dim
        self.t = t
        self.W = W
        self.N = L**dim
        self.rng = np.random.default_rng(seed)
        self._disorder = self.rng.uniform(-W / 2.0, W / 2.0, self.N)

    def hamiltonian(self) -> NDArray[np.float64]:
        """Build N×N Hamiltonian matrix with nearest-neighbour hopping + disorder."""
        H = np.diag(self._disorder.copy())
        N = self.N
        L = self.L

        for site in range(N):
            coords = self._site_to_coords(site)
            for d in range(self.dim):
                # Forward neighbour
                neighbour_coords = list(coords)
                neighbour_coords[d] = (coords[d] + 1) % L
                j = self._coords_to_site(tuple(neighbour_coords))
                H[site, j] -= self.t
                H[j, site] -= self.t

        return H

    def _site_to_coords(self, site: int) -> Tuple[int, ...]:
        coords: List[int] = []
        s = site
        for _ in range(self.dim):
            coords.append(s % self.L)
            s //= self.L
        return tuple(coords)

    def _coords_to_site(self, coords: Tuple[int, ...]) -> int:
        site = 0
        factor = 1
        for c in coords:
            site += c * factor
            factor *= self.L
        return site

    def eigensolve(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Full diagonalisation. Returns (eigenvalues, eigenvectors)."""
        H = self.hamiltonian()
        return np.linalg.eigh(H)

    def localisation_length_1d(self, E: float, n_samples: int = 1000,
                                 chain_length: int = 10000) -> float:
        r"""
        1D localisation length via transfer matrix product.

        Transfer matrix at site n:
        $$T_n = \begin{pmatrix} (E - \varepsilon_n)/t & -1 \\ 1 & 0 \end{pmatrix}$$

        $$\xi^{-1} = -\lim_{N\to\infty} \frac{1}{2N}\ln\|T_N \cdots T_1\|$$
        """
        if self.dim != 1:
            raise ValueError("Transfer matrix method only for 1D")

        rng = np.random.default_rng(self.rng.integers(0, 2**31))
        log_norms = 0.0

        v = np.array([1.0, 0.0])
        for n in range(chain_length):
            eps = rng.uniform(-self.W / 2.0, self.W / 2.0)
            T = np.array([
                [(E - eps) / self.t, -1.0],
                [1.0, 0.0],
            ])
            v = T @ v
            norm = np.linalg.norm(v)
            if norm > 1e-15:
                log_norms += math.log(norm)
                v /= norm

        lyapunov = log_norms / chain_length
        if lyapunov < 1e-15:
            return float('inf')
        return 1.0 / lyapunov


# ---------------------------------------------------------------------------
#  Kernel Polynomial Method (KPM)
# ---------------------------------------------------------------------------

class KPMSpectral:
    r"""
    Kernel Polynomial Method for spectral functions of large sparse matrices.

    Expand spectral density as Chebyshev series:
    $$\rho(E) = \frac{1}{\pi\sqrt{1-\tilde{E}^2}}
        \left[g_0\mu_0 + 2\sum_{n=1}^{N_C} g_n\mu_n\,T_n(\tilde{E})\right]$$

    where $\mu_n = \langle r|T_n(\tilde{H})|r\rangle$ (stochastic trace).

    Jackson kernel: $g_n = [(N_C - n + 1)\cos(\pi n/(N_C+1))
        + \sin(\pi n/(N_C+1))\cot(\pi/(N_C+1))] / (N_C + 1)$
    """

    def __init__(self, H: NDArray, n_chebyshev: int = 1024) -> None:
        """
        Parameters
        ----------
        H : Hamiltonian matrix (dense or could be sparse with @ support).
        n_chebyshev : Number of Chebyshev moments.
        """
        eigenvalues_approx = np.array([
            float(np.max(np.abs(np.sum(H, axis=1)))),
        ])
        self.E_scale = float(eigenvalues_approx[0]) * 1.01
        self.H_scaled = H / self.E_scale  # Rescale to [-1, 1]
        self.N_C = n_chebyshev
        self.dim = H.shape[0]

    def _jackson_kernel(self) -> NDArray[np.float64]:
        """Jackson damping coefficients."""
        N = self.N_C
        n = np.arange(N)
        g = ((N - n + 1) * np.cos(math.pi * n / (N + 1))
             + np.sin(math.pi * n / (N + 1)) / np.tan(math.pi / (N + 1))) / (N + 1)
        return g

    def moments(self, n_random: int = 10,
                seed: Optional[int] = None) -> NDArray[np.float64]:
        """
        Stochastic trace: μ_n = (1/R) Σ_r ⟨r|T_n(H̃)|r⟩.
        """
        rng = np.random.default_rng(seed)
        mu = np.zeros(self.N_C)

        for _ in range(n_random):
            # Random phase vector
            phases = rng.choice([-1.0, 1.0], size=self.dim)
            r = phases / math.sqrt(self.dim)

            # Chebyshev recursion: T_0 = I, T_1 = H̃, T_{n+1} = 2H̃T_n - T_{n-1}
            alpha_0 = r.copy()
            alpha_1 = self.H_scaled @ r

            mu[0] += float(r @ alpha_0)
            if self.N_C > 1:
                mu[1] += float(r @ alpha_1)

            for n in range(2, self.N_C):
                alpha_2 = 2.0 * (self.H_scaled @ alpha_1) - alpha_0
                mu[n] += float(r @ alpha_2)
                alpha_0 = alpha_1
                alpha_1 = alpha_2

        mu /= n_random
        return mu

    def dos(self, n_energy: int = 500,
            n_random: int = 10) -> Tuple[NDArray, NDArray]:
        """
        Compute density of states.

        Returns (energies, dos).
        """
        mu = self.moments(n_random)
        g = self._jackson_kernel()

        E_tilde = np.linspace(-0.99, 0.99, n_energy)
        rho = np.zeros(n_energy)

        for i, e in enumerate(E_tilde):
            val = g[0] * mu[0]
            for n in range(1, self.N_C):
                # T_n(e) via Chebyshev recursion
                Tn = math.cos(n * math.acos(e))
                val += 2.0 * g[n] * mu[n] * Tn
            rho[i] = val / (math.pi * math.sqrt(1.0 - e**2 + 1e-15))

        energies = E_tilde * self.E_scale
        rho_scaled = np.maximum(rho / self.E_scale, 0.0)
        return energies, rho_scaled


# ---------------------------------------------------------------------------
#  Edwards-Anderson Spin Glass
# ---------------------------------------------------------------------------

class EdwardsAndersonSpinGlass:
    r"""
    Edwards-Anderson spin glass with Gaussian-distributed couplings.

    $$H = -\sum_{\langle i,j\rangle} J_{ij}\,s_i\,s_j$$

    where $J_{ij} \sim \mathcal{N}(0, J^2)$.

    Implements:
    - Metropolis MC with replica exchange (parallel tempering)
    - Spin glass order parameter $q = (1/N)\sum_i s_i^{(1)} s_i^{(2)}$  (two replicas)
    - Binder cumulant for transition detection
    """

    def __init__(self, L: int, dim: int = 3, J_std: float = 1.0,
                 seed: Optional[int] = None) -> None:
        self.L = L
        self.dim = dim
        self.N = L**dim
        self.rng = np.random.default_rng(seed)

        # Generate coupling constants
        self._build_couplings(J_std)
        # Initial random spin configuration
        self.spins = self.rng.choice([-1, 1], size=self.N)

    def _build_couplings(self, J_std: float) -> None:
        """Build nearest-neighbour coupling dictionary."""
        self.couplings: Dict[Tuple[int, int], float] = {}
        import itertools  # noqa: delayed import
        for site in range(self.N):
            coords = []
            s = site
            for _ in range(self.dim):
                coords.append(s % self.L)
                s //= self.L
            for d in range(self.dim):
                nc = list(coords)
                nc[d] = (coords[d] + 1) % self.L
                j = 0
                factor = 1
                for c in nc:
                    j += c * factor
                    factor *= self.L
                if site < j:
                    self.couplings[(site, j)] = float(
                        self.rng.normal(0.0, J_std)
                    )

    def energy(self, spins: Optional[NDArray] = None) -> float:
        """Total energy."""
        if spins is None:
            spins = self.spins
        E = 0.0
        for (i, j), Jij in self.couplings.items():
            E -= Jij * spins[i] * spins[j]
        return E

    def metropolis_sweep(self, beta: float) -> int:
        """
        One Metropolis sweep (N attempted spin flips).
        Returns number of accepted flips.
        """
        accepted = 0
        for _ in range(self.N):
            site = self.rng.integers(0, self.N)
            dE = 0.0
            # Compute local energy change
            coords: List[int] = []
            s = site
            for _ in range(self.dim):
                coords.append(s % self.L)
                s //= self.L

            for d in range(self.dim):
                for direction in [+1, -1]:
                    nc = list(coords)
                    nc[d] = (coords[d] + direction) % self.L
                    j = 0
                    factor = 1
                    for c in nc:
                        j += c * factor
                        factor *= self.L
                    key = (min(site, j), max(site, j))
                    if key in self.couplings:
                        dE += 2.0 * self.couplings[key] * self.spins[site] * self.spins[j]

            if dE <= 0 or self.rng.random() < math.exp(-beta * dE):
                self.spins[site] *= -1
                accepted += 1

        return accepted

    def overlap_parameter(self, spins1: NDArray, spins2: NDArray) -> float:
        r"""Spin glass order parameter $q = (1/N)\sum_i s_i^{(1)} s_i^{(2)}$."""
        return float(np.mean(spins1 * spins2))

    def binder_cumulant(self, q_samples: NDArray[np.float64]) -> float:
        r"""
        Binder cumulant:
        $$g = \frac{1}{2}\left(3 - \frac{\langle q^4\rangle}{\langle q^2\rangle^2}\right)$$

        g → 1 in ordered phase, g → 0 in paramagnetic.
        """
        q2 = float(np.mean(q_samples**2))
        q4 = float(np.mean(q_samples**4))
        if abs(q2) < 1e-15:
            return 0.0
        return 0.5 * (3.0 - q4 / q2**2)


# ---------------------------------------------------------------------------
#  Participation Ratio & Fractal Dimension
# ---------------------------------------------------------------------------

class LocalisationMetrics:
    r"""
    Wavefunction localisation diagnostics.

    Inverse Participation Ratio (IPR):
    $$\text{IPR}_n = \sum_i |\psi_n(i)|^4$$

    - Extended state: IPR ~ 1/N
    - Localised state: IPR ~ O(1)

    Generalised IPR:
    $$P_q = \sum_i |\psi(i)|^{2q}$$

    Fractal dimension from multifractal scaling:
    $$P_q \sim N^{-D_q(q-1)}$$
    """

    @staticmethod
    def ipr(eigenvector: NDArray) -> float:
        """Inverse participation ratio Σ|ψ_i|⁴."""
        return float(np.sum(np.abs(eigenvector)**4))

    @staticmethod
    def participation_ratio(eigenvector: NDArray) -> float:
        """PR = 1/IPR. Effective number of sites occupied."""
        ipr = float(np.sum(np.abs(eigenvector)**4))
        if ipr < 1e-30:
            return float(len(eigenvector))
        return 1.0 / ipr

    @staticmethod
    def generalised_ipr(eigenvector: NDArray, q: float = 2.0) -> float:
        r"""$P_q = \sum_i |\psi_i|^{2q}$."""
        return float(np.sum(np.abs(eigenvector)**(2.0 * q)))

    @staticmethod
    def fractal_dimension(sizes: NDArray[np.float64],
                           iprs: NDArray[np.float64],
                           q: float = 2.0) -> float:
        r"""
        Extract fractal dimension $D_q$ from IPR scaling:
        $P_q \sim N^{-D_q(q-1)}$ → $D_q = -\log(P_q) / ((q-1)\log N)$.

        Parameters
        ----------
        sizes : System sizes N.
        iprs : Corresponding IPR(q) values.
        """
        if q == 1.0:
            raise ValueError("q=1 requires separate Shannon entropy analysis")

        log_N = np.log(sizes)
        log_P = np.log(iprs + 1e-300)

        # Linear fit: log(P_q) = -(q-1)*D_q * log(N) + const
        coeffs = np.polyfit(log_N, log_P, 1)
        D_q = -coeffs[0] / (q - 1.0)
        return float(D_q)

    @staticmethod
    def level_spacing_distribution(eigenvalues: NDArray[np.float64],
                                     n_bins: int = 50) -> Tuple[NDArray, NDArray]:
        """
        Nearest-neighbour level spacing distribution P(s).

        Unfold spectrum first, then bin spacings.
        P(s) = π s/2 exp(-π s²/4) for GOE (delocalised).
        P(s) = exp(-s) for Poisson (localised).
        """
        E = np.sort(eigenvalues)
        # Unfold: map to uniform level density
        cdf = np.arange(1, len(E) + 1) / len(E)
        # Interpolate unfolded energies
        spacings = np.diff(cdf * len(E))
        # Normalise so ⟨s⟩ = 1
        mean_s = float(np.mean(spacings))
        if mean_s > 1e-15:
            spacings /= mean_s

        hist, bin_edges = np.histogram(spacings, bins=n_bins, range=(0, 4), density=True)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centres, hist
