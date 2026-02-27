"""
Advanced Monte Carlo Methods — cluster algorithms, parallel tempering, entropic
sampling, histogram reweighting, multicanonical ensemble.

Domain V.4 — NEW.
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

K_B: float = 1.381e-23  # J/K


# ---------------------------------------------------------------------------
#  Swendsen-Wang Cluster Algorithm
# ---------------------------------------------------------------------------

class SwendsenWangCluster:
    r"""
    Swendsen-Wang cluster algorithm for the Ising model.

    Bond activation probability: $p = 1 - e^{-2\beta J}$ for aligned spins.

    Steps:
    1. Between all aligned nearest-neighbour pairs, activate bond with prob $p$.
    2. Identify Fortuin-Kasteleyn clusters via BFS/union-find.
    3. Flip each cluster independently with probability 1/2.

    Reduces critical slowing down: $z \approx 0.2$ vs $z \approx 2$ for Metropolis.
    """

    def __init__(self, L: int = 32, J: float = 1.0) -> None:
        """L×L 2D Ising model."""
        self.L = L
        self.J = J
        self.spins = np.random.choice([-1, 1], size=(L, L))

    def _find(self, parent: NDArray, i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(self, parent: NDArray, rank: NDArray,
                  i: int, j: int) -> None:
        ri, rj = self._find(parent, i), self._find(parent, j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1

    def step(self, T: float) -> None:
        """One Swendsen-Wang sweep."""
        L = self.L
        N = L * L
        beta = 1 / T
        p = 1 - math.exp(-2 * beta * self.J)

        parent = np.arange(N)
        rank = np.zeros(N, dtype=int)

        # Activate bonds
        for i in range(L):
            for j in range(L):
                idx = i * L + j
                # Right neighbour
                ni = i
                nj = (j + 1) % L
                n_idx = ni * L + nj
                if self.spins[i, j] == self.spins[ni, nj]:
                    if np.random.random() < p:
                        self._union(parent, rank, idx, n_idx)
                # Down neighbour
                ni = (i + 1) % L
                nj = j
                n_idx = ni * L + nj
                if self.spins[i, j] == self.spins[ni, nj]:
                    if np.random.random() < p:
                        self._union(parent, rank, idx, n_idx)

        # Find clusters and flip
        cluster_flip: Dict[int, int] = {}
        for i in range(L):
            for j in range(L):
                idx = i * L + j
                root = self._find(parent, idx)
                if root not in cluster_flip:
                    cluster_flip[root] = 1 if np.random.random() < 0.5 else -1
                self.spins[i, j] = cluster_flip[root] * abs(self.spins[i, j])

    def magnetisation(self) -> float:
        """⟨|M|⟩/N."""
        return abs(float(np.mean(self.spins)))

    def energy(self) -> float:
        """Total energy per spin."""
        L = self.L
        E = 0.0
        for i in range(L):
            for j in range(L):
                S = self.spins[i, j]
                nn = self.spins[(i + 1) % L, j] + self.spins[i, (j + 1) % L]
                E -= self.J * S * nn
        return E / (L * L)


# ---------------------------------------------------------------------------
#  Parallel Tempering (Replica Exchange)
# ---------------------------------------------------------------------------

class ParallelTempering:
    r"""
    Parallel tempering (replica exchange Monte Carlo).

    Run $M$ replicas at temperatures $T_1 < T_2 < \ldots < T_M$.

    After independent MC sweeps, propose swap between adjacent replicas:
    $$P_{\text{swap}} = \min\left(1, \exp\left[(\beta_i-\beta_j)(E_i-E_j)\right]\right)$$

    Temperature selection: geometric progression $T_k = T_{\min}(T_{\max}/T_{\min})^{k/(M-1)}$
    """

    def __init__(self, energy_func: Callable[[NDArray], float],
                 propose_func: Callable[[NDArray], NDArray],
                 n_replicas: int = 8,
                 T_min: float = 1.0, T_max: float = 10.0) -> None:
        """
        energy_func: E(state) → float.
        propose_func: state → new_state (single MC move).
        """
        self.energy = energy_func
        self.propose = propose_func
        self.n_rep = n_replicas
        self.temperatures = np.geomspace(T_min, T_max, n_replicas)
        self.states: List[NDArray] = []
        self.energies = np.zeros(n_replicas)

    def initialise(self, init_state: NDArray) -> None:
        """Initialise all replicas with the same state."""
        self.states = [init_state.copy() for _ in range(self.n_rep)]
        self.energies = np.array([self.energy(s) for s in self.states])

    def local_sweep(self, n_local: int = 100) -> None:
        """Perform n_local Metropolis moves per replica."""
        for r in range(self.n_rep):
            T = self.temperatures[r]
            beta = 1 / T
            for _ in range(n_local):
                new_state = self.propose(self.states[r])
                new_E = self.energy(new_state)
                dE = new_E - self.energies[r]
                if dE <= 0 or np.random.random() < math.exp(-beta * dE):
                    self.states[r] = new_state
                    self.energies[r] = new_E

    def replica_exchange(self) -> int:
        """Attempt swaps between adjacent replicas. Returns n_swaps."""
        n_swaps = 0
        for r in range(self.n_rep - 1):
            beta_i = 1 / self.temperatures[r]
            beta_j = 1 / self.temperatures[r + 1]
            delta = (beta_i - beta_j) * (self.energies[r] - self.energies[r + 1])
            if delta <= 0 or np.random.random() < math.exp(delta):
                self.states[r], self.states[r + 1] = self.states[r + 1], self.states[r]
                self.energies[r], self.energies[r + 1] = self.energies[r + 1], self.energies[r]
                n_swaps += 1
        return n_swaps

    def run(self, n_sweeps: int = 1000,
               n_local: int = 100) -> Dict[str, NDArray]:
        """Run PT simulation."""
        E_history = np.zeros((n_sweeps, self.n_rep))
        swap_rates = np.zeros(n_sweeps)

        for s in range(n_sweeps):
            self.local_sweep(n_local)
            n_swaps = self.replica_exchange()
            E_history[s] = self.energies
            swap_rates[s] = n_swaps / max(self.n_rep - 1, 1)

        return {
            'E_history': E_history,
            'swap_rates': swap_rates,
            'temperatures': self.temperatures,
        }


# ---------------------------------------------------------------------------
#  Histogram Reweighting (Ferrenberg-Swendsen)
# ---------------------------------------------------------------------------

class HistogramReweighting:
    r"""
    Single-histogram and multi-histogram reweighting.

    Single histogram at $\beta_0$:
    $$\langle A\rangle_\beta = \frac{\sum_E A(E)\,H(E)\,e^{-(\beta-\beta_0)E}}{\sum_E H(E)\,e^{-(\beta-\beta_0)E}}$$

    Multi-histogram (WHAM):
    $$P(E) = \frac{\sum_i H_i(E)}{\sum_i n_i e^{f_i - \beta_i E}}$$
    $$e^{-f_i} = \sum_E P(E) e^{-\beta_i E}$$

    Iterate until self-consistent.
    """

    def __init__(self) -> None:
        self.histograms: List[Tuple[float, NDArray, NDArray]] = []

    def add_histogram(self, beta: float, energies: NDArray,
                         bins: NDArray) -> None:
        """Store histogram H(E) collected at inverse temperature β.

        energies: bin centres.
        bins: counts.
        """
        self.histograms.append((beta, energies.copy(), bins.copy()))

    def single_reweight(self, beta_target: float,
                           idx: int = 0) -> Tuple[float, float]:
        """Reweight histogram idx to get ⟨E⟩ and ⟨E²⟩ at β_target.

        Returns (mean_E, var_E).
        """
        beta_0, E, H = self.histograms[idx]
        d_beta = beta_target - beta_0
        weights = H * np.exp(-d_beta * E)
        Z = float(np.sum(weights))
        if Z < 1e-30:
            return 0.0, 0.0
        mean_E = float(np.sum(E * weights)) / Z
        mean_E2 = float(np.sum(E**2 * weights)) / Z
        return mean_E, mean_E2 - mean_E**2

    def specific_heat(self, beta_target: float, N: int,
                         idx: int = 0) -> float:
        """C_V(β) = β² (⟨E²⟩ − ⟨E⟩²) / N."""
        _, var_E = self.single_reweight(beta_target, idx)
        return beta_target**2 * var_E / N

    def wham(self, E_bins: NDArray, n_iter: int = 100) -> NDArray:
        """Weighted histogram analysis method.

        Returns density of states Ω(E) ∝ P(E).
        """
        n_hists = len(self.histograms)
        n_E = len(E_bins)

        # Collect histograms on common energy grid
        H_all = np.zeros((n_hists, n_E))
        betas = np.zeros(n_hists)
        n_samples = np.zeros(n_hists)

        for i, (beta, E, H) in enumerate(self.histograms):
            betas[i] = beta
            n_samples[i] = float(np.sum(H))
            for j in range(n_E):
                idx = np.argmin(np.abs(E - E_bins[j]))
                if idx < len(H):
                    H_all[i, j] = H[idx]

        f = np.zeros(n_hists)  # free energies
        log_omega = np.zeros(n_E)

        for _ in range(n_iter):
            # Update log_omega
            num = np.sum(H_all, axis=0)
            den = np.zeros(n_E)
            for i in range(n_hists):
                den += n_samples[i] * np.exp(f[i] - betas[i] * E_bins)
            log_omega = np.log(num + 1e-30) - np.log(den + 1e-30)

            # Update f
            for i in range(n_hists):
                f[i] = -np.log(np.sum(np.exp(log_omega - betas[i] * E_bins)) + 1e-30)

        return np.exp(log_omega)


# ---------------------------------------------------------------------------
#  Multicanonical Ensemble (MUCA)
# ---------------------------------------------------------------------------

class MulticanonicalMC:
    r"""
    Multicanonical ensemble (flat-histogram) Monte Carlo.

    Modified acceptance:
    $$P_{\text{acc}} = \min\left(1, \frac{w(E_{\text{old}})}{w(E_{\text{new}})}\right)$$
    where $w(E) = 1/\Omega(E)$ to produce flat energy histogram.

    Iterative weight estimation:
    1. Start with w(E) = 1.
    2. Run MC, collect H(E).
    3. Update: $w^{(k+1)}(E) = w^{(k)}(E) \times H(E)$.
    4. Repeat until H(E) is flat.
    """

    def __init__(self, energy_func: Callable[[NDArray], float],
                 propose_func: Callable[[NDArray], NDArray],
                 E_min: float = -2.0, E_max: float = 0.0,
                 n_bins: int = 100) -> None:
        self.energy = energy_func
        self.propose = propose_func
        self.E_bins = np.linspace(E_min, E_max, n_bins)
        self.dE = self.E_bins[1] - self.E_bins[0]
        self.n_bins = n_bins
        self.log_w = np.zeros(n_bins)
        self.histogram = np.zeros(n_bins)

    def _bin_index(self, E: float) -> int:
        idx = int((E - self.E_bins[0]) / self.dE)
        return max(0, min(self.n_bins - 1, idx))

    def sweep(self, state: NDArray, n_steps: int = 1000) -> NDArray:
        """One multicanonical sweep."""
        E_curr = self.energy(state)
        i_curr = self._bin_index(E_curr)

        for _ in range(n_steps):
            new_state = self.propose(state)
            E_new = self.energy(new_state)
            i_new = self._bin_index(E_new)

            d_log_w = self.log_w[i_curr] - self.log_w[i_new]
            if d_log_w >= 0 or np.random.random() < math.exp(d_log_w):
                state = new_state
                E_curr = E_new
                i_curr = i_new

            self.histogram[i_curr] += 1

        return state

    def update_weights(self) -> None:
        """Update weights from histogram: log_w += log(H)."""
        mask = self.histogram > 0
        self.log_w[mask] += np.log(self.histogram[mask])
        self.log_w -= np.min(self.log_w)
        self.histogram[:] = 0

    def is_flat(self, threshold: float = 0.8) -> bool:
        """Check if histogram is flat (min/mean > threshold)."""
        if np.min(self.histogram) == 0:
            return False
        return float(np.min(self.histogram)) / float(np.mean(self.histogram)) > threshold

    def density_of_states(self) -> NDArray:
        """Ω(E) ∝ exp(−log_w)."""
        return np.exp(-self.log_w + np.max(-self.log_w))
