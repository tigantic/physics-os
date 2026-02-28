"""
Equilibrium statistical mechanics — Monte Carlo, partition functions, phase transitions.

Upgrades domain V.1 from spin-model MPO (ontic/mps/hamiltonians.py) to
full classical statistical mechanics:
  - Metropolis-Hastings MC for classical Ising / Potts / XY
  - Wolff single-cluster algorithm (critical slowing-down reduction)
  - Wang-Landau flat-histogram (density of states g(E))
  - Canonical partition function engine (exact enumeration / transfer matrix)
  - Landau mean-field theory (phase transitions, order parameter)

Physical constants: k_B = 1.380649×10⁻²³ J/K.
Reduced temperature T* = k_B T / J used internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

K_B: float = 1.380649e-23  # J/K


# ===================================================================
#  Lattice spin models
# ===================================================================

class IsingModel:
    r"""
    2D Ising model on a square lattice with periodic BCs.

    $$H = -J \sum_{\langle ij \rangle} s_i s_j - h \sum_i s_i, \quad s_i \in \{-1, +1\}$$

    Exact 2D critical temperature (Onsager): $T_c = 2J / \ln(1+\sqrt{2}) \approx 2.269J/k_B$.
    """

    def __init__(self, L: int, J: float = 1.0, h: float = 0.0,
                 temperature: float = 2.269, seed: Optional[int] = None) -> None:
        self.L = L
        self.J = J
        self.h = h
        self.T = temperature
        self.beta = 1.0 / temperature if temperature > 0 else np.inf
        self.rng = np.random.default_rng(seed)

        # Spin lattice: random ±1
        self.spins = self.rng.choice([-1, 1], size=(L, L)).astype(np.int8)

    @property
    def critical_temperature(self) -> float:
        """Onsager exact Tc for 2D square lattice."""
        return 2.0 * self.J / math.log(1.0 + math.sqrt(2.0))

    def energy(self) -> float:
        """Total energy H = -J Σ s_i s_j - h Σ s_i."""
        s = self.spins.astype(np.float64)
        E_nn = -self.J * np.sum(
            s * (np.roll(s, 1, axis=0) + np.roll(s, 1, axis=1)))
        E_field = -self.h * np.sum(s)
        return float(E_nn + E_field)

    def magnetisation(self) -> float:
        """m = (1/N) Σ s_i."""
        return float(np.mean(self.spins))

    def magnetisation_abs(self) -> float:
        """<|m|>."""
        return float(abs(np.mean(self.spins)))

    def _delta_energy(self, i: int, j: int) -> float:
        """Energy change from flipping spin at (i, j)."""
        L = self.L
        s = self.spins
        si = s[i, j]
        nn_sum = (s[(i + 1) % L, j] + s[(i - 1) % L, j] +
                  s[i, (j + 1) % L] + s[i, (j - 1) % L])
        return 2.0 * si * (self.J * nn_sum + self.h)


class PottsModel:
    r"""
    q-state Potts model on a square lattice.

    $$H = -J \sum_{\langle ij\rangle} \delta_{s_i, s_j}, \quad s_i \in \{0,1,\dots,q-1\}$$

    Critical temperature (2D): $T_c = J / \ln(1+\sqrt{q})$.
    """

    def __init__(self, L: int, q: int = 3, J: float = 1.0,
                 temperature: float = 1.0, seed: Optional[int] = None) -> None:
        self.L = L
        self.q = q
        self.J = J
        self.T = temperature
        self.beta = 1.0 / temperature if temperature > 0 else np.inf
        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.integers(0, q, size=(L, L))

    @property
    def critical_temperature(self) -> float:
        return self.J / math.log(1.0 + math.sqrt(self.q))

    def energy(self) -> float:
        s = self.spins
        E = -(self.J * np.sum(
            (s == np.roll(s, 1, axis=0)).astype(np.float64) +
            (s == np.roll(s, 1, axis=1)).astype(np.float64)))
        return float(E)

    def _delta_energy(self, i: int, j: int, new_state: int) -> float:
        L = self.L
        s = self.spins
        old_state = s[i, j]
        if old_state == new_state:
            return 0.0
        nn = [s[(i + 1) % L, j], s[(i - 1) % L, j],
              s[i, (j + 1) % L], s[i, (j - 1) % L]]
        old_bonds = sum(1 for n in nn if n == old_state)
        new_bonds = sum(1 for n in nn if n == new_state)
        return -self.J * (new_bonds - old_bonds)


class XYModel:
    r"""
    Classical XY model on a square lattice.

    $$H = -J \sum_{\langle ij\rangle} \cos(\theta_i - \theta_j)
          - h \sum_i \cos(\theta_i)$$

    Undergoes Berezinskii-Kosterlitz-Thouless (BKT) transition at
    $T_{\text{BKT}} \approx 0.893 J / k_B$ (2D).
    """

    def __init__(self, L: int, J: float = 1.0, h: float = 0.0,
                 temperature: float = 1.0, seed: Optional[int] = None) -> None:
        self.L = L
        self.J = J
        self.h = h
        self.T = temperature
        self.beta = 1.0 / temperature if temperature > 0 else np.inf
        self.rng = np.random.default_rng(seed)
        self.angles = self.rng.uniform(0, 2 * np.pi, size=(L, L))

    @property
    def bkt_temperature(self) -> float:
        return 0.893 * self.J

    def energy(self) -> float:
        theta = self.angles
        E_nn = -self.J * np.sum(
            np.cos(theta - np.roll(theta, 1, axis=0)) +
            np.cos(theta - np.roll(theta, 1, axis=1)))
        E_field = -self.h * np.sum(np.cos(theta))
        return float(E_nn + E_field)

    def magnetisation_vector(self) -> Tuple[float, float]:
        """(m_x, m_y) = (1/N)(Σ cos θ_i, Σ sin θ_i)."""
        N = self.L ** 2
        mx = float(np.sum(np.cos(self.angles))) / N
        my = float(np.sum(np.sin(self.angles))) / N
        return mx, my

    def magnetisation_abs(self) -> float:
        mx, my = self.magnetisation_vector()
        return math.sqrt(mx**2 + my**2)

    def vorticity(self) -> NDArray[np.float64]:
        r"""
        Compute vortex charge at each plaquette (detects BKT vortices).

        $n_v = \frac{1}{2\pi} \oint d\theta$ around plaquette.
        """
        theta = self.angles
        L = self.L
        # Differences around plaquette (wrapped to [-π, π])
        def wrap(d: NDArray) -> NDArray:
            return np.arctan2(np.sin(d), np.cos(d))

        d1 = wrap(theta - np.roll(theta, 1, axis=1))
        d2 = wrap(np.roll(theta, 1, axis=1) -
                  np.roll(np.roll(theta, 1, axis=1), 1, axis=0))
        d3 = wrap(np.roll(np.roll(theta, 1, axis=1), 1, axis=0) -
                  np.roll(theta, 1, axis=0))
        d4 = wrap(np.roll(theta, 1, axis=0) - theta)
        circulation = d1 + d2 + d3 + d4
        return np.round(circulation / (2.0 * np.pi)).astype(np.float64)


# ===================================================================
#  Metropolis-Hastings Monte Carlo
# ===================================================================

@dataclass
class MCResult:
    """Monte Carlo simulation result."""
    energies: NDArray[np.float64]
    magnetisations: NDArray[np.float64]
    acceptance_rate: float
    specific_heat: float = 0.0
    susceptibility: float = 0.0


class MetropolisMC:
    r"""
    Metropolis-Hastings Monte Carlo for classical spin models.

    Acceptance probability:
    $$
    P_{\text{accept}} = \min\!\left(1, \exp(-\beta\,\Delta E)\right)
    $$

    Single spin-flip dynamics. Supports Ising, Potts, and XY models.
    """

    def __init__(self, model: Union[IsingModel, PottsModel, XYModel],
                 seed: Optional[int] = None) -> None:
        self.model = model
        self.rng = np.random.default_rng(seed)

    def sweep(self) -> int:
        """
        One Monte Carlo sweep (N attempted spin flips).

        Returns number of accepted moves.
        """
        m = self.model
        L = m.L
        N = L * L
        accepted = 0

        if isinstance(m, IsingModel):
            for _ in range(N):
                i = self.rng.integers(0, L)
                j = self.rng.integers(0, L)
                dE = m._delta_energy(i, j)
                if dE <= 0 or self.rng.random() < math.exp(-m.beta * dE):
                    m.spins[i, j] *= -1
                    accepted += 1

        elif isinstance(m, PottsModel):
            for _ in range(N):
                i = self.rng.integers(0, L)
                j = self.rng.integers(0, L)
                new_state = self.rng.integers(0, m.q)
                dE = m._delta_energy(i, j, new_state)
                if dE <= 0 or self.rng.random() < math.exp(-m.beta * dE):
                    m.spins[i, j] = new_state
                    accepted += 1

        elif isinstance(m, XYModel):
            max_angle = np.pi / 3.0  # Tunable step size
            for _ in range(N):
                i = self.rng.integers(0, L)
                j = self.rng.integers(0, L)
                old_theta = m.angles[i, j]

                # Compute old local energy
                nn_sum_cos_old = self._xy_nn_cos(m, i, j, old_theta)
                E_old = -m.J * nn_sum_cos_old - m.h * math.cos(old_theta)

                new_theta = old_theta + self.rng.uniform(-max_angle, max_angle)
                nn_sum_cos_new = self._xy_nn_cos(m, i, j, new_theta)
                E_new = -m.J * nn_sum_cos_new - m.h * math.cos(new_theta)

                dE = E_new - E_old
                if dE <= 0 or self.rng.random() < math.exp(-m.beta * dE):
                    m.angles[i, j] = new_theta % (2.0 * np.pi)
                    accepted += 1

        return accepted

    @staticmethod
    def _xy_nn_cos(model: XYModel, i: int, j: int, theta: float) -> float:
        L = model.L
        a = model.angles
        return (math.cos(theta - a[(i + 1) % L, j]) +
                math.cos(theta - a[(i - 1) % L, j]) +
                math.cos(theta - a[i, (j + 1) % L]) +
                math.cos(theta - a[i, (j - 1) % L]))

    def run(self, n_sweeps: int, n_warmup: int = 100) -> MCResult:
        """
        Run MC simulation collecting energy and magnetisation.

        Parameters
        ----------
        n_sweeps : Number of measurement sweeps.
        n_warmup : Thermalisation sweeps (discarded).
        """
        # Warmup
        for _ in range(n_warmup):
            self.sweep()

        N = self.model.L ** 2
        energies = np.empty(n_sweeps)
        mags = np.empty(n_sweeps)
        total_accepted = 0

        for s in range(n_sweeps):
            total_accepted += self.sweep()
            energies[s] = self.model.energy() / N
            if isinstance(self.model, XYModel):
                mags[s] = self.model.magnetisation_abs()
            elif isinstance(self.model, PottsModel):
                # Use max state fraction as order parameter
                counts = np.bincount(self.model.spins.ravel(), minlength=self.model.q)
                mags[s] = float(np.max(counts)) / N
            else:
                mags[s] = self.model.magnetisation_abs()

        acceptance = total_accepted / (n_sweeps * N)

        # Thermodynamic quantities
        beta = self.model.beta
        E_mean = np.mean(energies)
        E2_mean = np.mean(energies**2)
        cv = beta**2 * N * (E2_mean - E_mean**2)

        m_mean = np.mean(mags)
        m2_mean = np.mean(mags**2)
        chi = beta * N * (m2_mean - m_mean**2)

        return MCResult(
            energies=energies,
            magnetisations=mags,
            acceptance_rate=acceptance,
            specific_heat=float(cv),
            susceptibility=float(chi),
        )


# ===================================================================
#  Wolff Cluster Monte Carlo
# ===================================================================

class WolffClusterMC:
    r"""
    Wolff single-cluster algorithm for Ising model.

    Dramatically reduces critical slowing-down near $T_c$.

    Algorithm:
    1. Pick random seed spin s_0
    2. For each unvisited neighbour with same spin:
       add to cluster with probability $p = 1 - \exp(-2\beta J)$
    3. Flip entire cluster

    Dynamic critical exponent $z \approx 0.25$ (vs $z \approx 2.17$ single-flip).
    """

    def __init__(self, model: IsingModel, seed: Optional[int] = None) -> None:
        if not isinstance(model, IsingModel):
            raise TypeError("Wolff algorithm requires IsingModel")
        self.model = model
        self.rng = np.random.default_rng(seed)

    def cluster_flip(self) -> int:
        """
        Build and flip one Wolff cluster.

        Returns cluster size.
        """
        m = self.model
        L = m.L
        p_add = 1.0 - math.exp(-2.0 * m.beta * m.J)

        # Pick random seed
        i0 = self.rng.integers(0, L)
        j0 = self.rng.integers(0, L)
        seed_spin = m.spins[i0, j0]

        cluster = set()
        stack = [(i0, j0)]
        cluster.add((i0, j0))

        while stack:
            ci, cj = stack.pop()
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = (ci + di) % L, (cj + dj) % L
                if (ni, nj) not in cluster and m.spins[ni, nj] == seed_spin:
                    if self.rng.random() < p_add:
                        cluster.add((ni, nj))
                        stack.append((ni, nj))

        # Flip cluster
        for ci, cj in cluster:
            m.spins[ci, cj] *= -1

        return len(cluster)

    def run(self, n_clusters: int, n_warmup: int = 100) -> MCResult:
        """
        Run Wolff simulation for n_clusters cluster flips.
        """
        N = self.model.L ** 2

        # Warmup
        for _ in range(n_warmup):
            self.cluster_flip()

        energies = np.empty(n_clusters)
        mags = np.empty(n_clusters)

        for s in range(n_clusters):
            self.cluster_flip()
            energies[s] = self.model.energy() / N
            mags[s] = self.model.magnetisation_abs()

        beta = self.model.beta
        E_mean = np.mean(energies)
        E2_mean = np.mean(energies**2)
        cv = beta**2 * N * (E2_mean - E_mean**2)

        m_mean = np.mean(mags)
        m2_mean = np.mean(mags**2)
        chi = beta * N * (m2_mean - m_mean**2)

        return MCResult(
            energies=energies,
            magnetisations=mags,
            acceptance_rate=1.0,  # cluster algorithms always accept
            specific_heat=float(cv),
            susceptibility=float(chi),
        )


# ===================================================================
#  Wang-Landau Flat-Histogram Monte Carlo
# ===================================================================

class WangLandauMC:
    r"""
    Wang-Landau algorithm for computing the density of states g(E).

    $$g(E) \to g(E) \cdot f \text{ when energy E is visited}$$
    $$H(E) \text{ histogram tracked; when flat, } f \to \sqrt{f}$$

    Converges to the microcanonical density of states from which
    canonical thermodynamics at any temperature can be extracted:
    $$Z(\beta) = \sum_E g(E) e^{-\beta E}$$
    """

    def __init__(self, model: IsingModel, seed: Optional[int] = None,
                 f_init: float = math.e, f_min: float = 1e-8,
                 flatness_criterion: float = 0.8) -> None:
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.f = f_init
        self.f_min = f_min
        self.flatness = flatness_criterion

        # Determine energy range for Ising model
        L = model.L
        N = L * L
        # Max energy per spin = 2J * (num_bonds) / N, discrete in steps of 4J
        E_max = 2 * model.J * 2 * N  # all antiparallel
        self.E_min = -E_max
        self.E_max = E_max
        self.dE = 4.0 * model.J  # Ising energy steps

        n_bins = int((self.E_max - self.E_min) / self.dE) + 1
        self.n_bins = n_bins
        self.ln_g = np.zeros(n_bins)  # ln(g(E))
        self.histogram = np.zeros(n_bins, dtype=np.int64)

    def _energy_to_bin(self, E: float) -> int:
        idx = int(round((E - self.E_min) / self.dE))
        return max(0, min(idx, self.n_bins - 1))

    def _is_flat(self) -> bool:
        """Check if histogram is flat (all bins > flatness * mean)."""
        nonzero = self.histogram[self.histogram > 0]
        if len(nonzero) < 2:
            return False
        return float(np.min(nonzero)) >= self.flatness * float(np.mean(nonzero))

    def run(self, max_iterations: int = 100_000_000,
            check_interval: int = 10_000) -> None:
        """
        Run Wang-Landau simulation until modification factor < f_min.
        """
        m = self.model
        L = m.L
        E_current = m.energy()
        ln_f = math.log(self.f)

        iteration = 0
        while ln_f > math.log(self.f_min) and iteration < max_iterations:
            # Propose single spin flip
            i = self.rng.integers(0, L)
            j = self.rng.integers(0, L)
            dE = m._delta_energy(i, j)
            E_new = E_current + dE

            bin_old = self._energy_to_bin(E_current)
            bin_new = self._energy_to_bin(E_new)

            # Wang-Landau acceptance
            if self.ln_g[bin_new] <= self.ln_g[bin_old] or \
               self.rng.random() < math.exp(self.ln_g[bin_old] - self.ln_g[bin_new]):
                m.spins[i, j] *= -1
                E_current = E_new
                self.ln_g[bin_new] += ln_f
                self.histogram[bin_new] += 1
            else:
                self.ln_g[bin_old] += ln_f
                self.histogram[bin_old] += 1

            iteration += 1

            # Check flatness periodically
            if iteration % check_interval == 0 and self._is_flat():
                ln_f /= 2.0
                self.histogram[:] = 0

        # Normalise ln(g) so that minimum is 0
        self.ln_g -= np.min(self.ln_g[self.ln_g > -np.inf])

    def canonical_average(self, beta: float) -> Tuple[float, float, float]:
        """
        Compute canonical thermodynamic quantities at inverse temperature β.

        Returns (mean_energy_per_spin, specific_heat_per_spin, free_energy_per_spin).
        """
        N = self.model.L ** 2
        E_values = np.arange(self.n_bins) * self.dE + self.E_min

        # Shift for numerical stability
        ln_weights = self.ln_g - beta * E_values
        ln_weights -= np.max(ln_weights)
        weights = np.exp(ln_weights)
        Z = np.sum(weights)

        if Z < 1e-300:
            return 0.0, 0.0, 0.0

        probs = weights / Z
        E_mean = np.sum(probs * E_values) / N
        E2_mean = np.sum(probs * E_values**2) / N**2
        cv = beta**2 * N * (E2_mean - E_mean**2)

        F = -math.log(Z) / (beta * N) if beta > 0 else 0.0

        return float(E_mean), float(cv), float(F)

    def entropy(self) -> NDArray[np.float64]:
        """Microcanonical entropy S(E) = k_B ln(g(E))."""
        return K_B * self.ln_g


# ===================================================================
#  Partition Function Engine
# ===================================================================

class PartitionFunction:
    r"""
    Canonical partition function computation.

    Exact enumeration (small systems):
    $$Z(\beta) = \sum_{\text{states}} e^{-\beta E}$$

    Transfer matrix (1D/quasi-1D):
    $$Z = \text{Tr}(\mathbf{T}^N), \quad T_{s s'} = e^{-\beta H(s, s')}$$

    Thermodynamic quantities:
    $$\langle E\rangle = -\frac{\partial \ln Z}{\partial \beta}, \quad
      C_V = \beta^2 \left(\langle E^2\rangle - \langle E\rangle^2\right)$$
    """

    @staticmethod
    def exact_enumeration_ising_1d(N: int, J: float, h: float,
                                   beta: float) -> Dict[str, float]:
        r"""
        Exact partition function for 1D Ising chain via transfer matrix.

        Transfer matrix:
        $$T = \begin{pmatrix}
            e^{\beta(J+h)} & e^{-\beta J} \\
            e^{-\beta J} & e^{\beta(J-h)}
        \end{pmatrix}$$

        $$Z = \lambda_+^N + \lambda_-^N$$
        """
        T = np.array([
            [math.exp(beta * (J + h)), math.exp(-beta * J)],
            [math.exp(-beta * J), math.exp(beta * (J - h))],
        ])
        eigenvalues = np.linalg.eigvalsh(T)
        lam_plus = float(max(eigenvalues))
        lam_minus = float(min(eigenvalues))

        Z = lam_plus ** N + lam_minus ** N
        ln_Z = N * math.log(lam_plus) + math.log(1.0 + (lam_minus / lam_plus)**N)

        # Free energy per spin
        F_per_spin = -ln_Z / (beta * N) if beta > 0 else 0.0

        # Mean energy from eigenvalue derivatives (numerical)
        dbeta = 1e-8
        beta_p = beta + dbeta
        T_p = np.array([
            [math.exp(beta_p * (J + h)), math.exp(-beta_p * J)],
            [math.exp(-beta_p * J), math.exp(beta_p * (J - h))],
        ])
        eig_p = np.linalg.eigvalsh(T_p)
        lam_plus_p = float(max(eig_p))
        ln_Z_p = N * math.log(lam_plus_p) + math.log(
            1.0 + (float(min(eig_p)) / lam_plus_p)**N)
        E_mean = -(ln_Z_p - ln_Z) / dbeta / N

        return {
            "Z": Z,
            "ln_Z": ln_Z,
            "free_energy_per_spin": F_per_spin,
            "energy_per_spin": E_mean,
            "lambda_plus": lam_plus,
            "lambda_minus": lam_minus,
        }

    @staticmethod
    def transfer_matrix_2d_strip(Ly: int, Lx: int, J: float,
                                 beta: float) -> Dict[str, float]:
        """
        Transfer matrix for 2D Ising strip of width Ly, length Lx.

        Transfer matrix is 2^Ly × 2^Ly; exact for small Ly.
        """
        n_states = 2 ** Ly

        def spin_config(idx: int) -> List[int]:
            return [2 * ((idx >> k) & 1) - 1 for k in range(Ly)]

        # Build transfer matrix
        T = np.zeros((n_states, n_states))
        for si in range(n_states):
            s_left = spin_config(si)
            for sj in range(n_states):
                s_right = spin_config(sj)
                # Intra-column energy
                E_col = -J * sum(s_right[k] * s_right[(k + 1) % Ly]
                                 for k in range(Ly))
                # Inter-column energy
                E_inter = -J * sum(s_left[k] * s_right[k] for k in range(Ly))
                T[si, sj] = math.exp(-beta * (E_col + E_inter))

        eigenvalues = np.linalg.eigvalsh(T)
        lam_max = float(np.max(eigenvalues))

        ln_Z = Lx * math.log(lam_max) if lam_max > 0 else -np.inf
        F_per_spin = -ln_Z / (beta * Lx * Ly) if beta > 0 else 0.0

        return {
            "ln_Z": ln_Z,
            "free_energy_per_spin": F_per_spin,
            "lambda_max": lam_max,
            "correlation_length": (-1.0 / math.log(
                float(sorted(eigenvalues)[-2]) / lam_max)
                if len(eigenvalues) > 1 and sorted(eigenvalues)[-2] > 0
                else np.inf),
        }


# ===================================================================
#  Landau Mean-Field Theory
# ===================================================================

class LandauMeanField:
    r"""
    Landau mean-field theory for phase transitions.

    Ising mean-field free energy:
    $$f(m, T) = -\frac{zJ}{2}m^2 - \frac{T}{2}\left[
        (1+m)\ln\frac{1+m}{2} + (1-m)\ln\frac{1-m}{2}
    \right] - hm$$

    Mean-field equation: $m = \tanh(\beta(zJm + h))$

    General Landau expansion:
    $$F = F_0 + \frac{a}{2}\phi^2 + \frac{b}{4}\phi^4 + \frac{c}{6}\phi^6 - h\phi$$

    where $a = a_0(T - T_c)$, giving continuous (b > 0) or first-order (b < 0) transitions.
    """

    @staticmethod
    def ising_mean_field(T: float, J: float = 1.0, h: float = 0.0,
                         z: int = 4) -> Tuple[float, float]:
        """
        Solve Ising mean-field self-consistency equation.

        Parameters
        ----------
        T : Temperature [reduced units, k_B = 1].
        J : Coupling constant.
        h : External field.
        z : Coordination number (4 for square, 6 for cubic).

        Returns
        -------
        m : Spontaneous magnetisation.
        f : Free energy per spin.
        """
        beta = 1.0 / T if T > 0 else np.inf

        # Iterative solution of m = tanh(β(zJm + h))
        m = 0.5  # Initial guess
        for _ in range(1000):
            arg = beta * (z * J * m + h)
            arg = np.clip(arg, -500.0, 500.0)
            m_new = math.tanh(arg)
            if abs(m_new - m) < 1e-12:
                break
            m = 0.5 * m_new + 0.5 * m  # Damped iteration

        # Free energy
        if abs(m) < 1.0 - 1e-15:
            f = (-0.5 * z * J * m**2
                 - T * 0.5 * ((1 + m) * math.log((1 + m) / 2)
                               + (1 - m) * math.log((1 - m) / 2))
                 - h * m)
        else:
            f = -0.5 * z * J * m**2 - h * m

        return m, f

    @staticmethod
    def critical_temperature_mf(J: float = 1.0, z: int = 4) -> float:
        """Mean-field critical temperature T_c = zJ/k_B."""
        return z * J

    @staticmethod
    def landau_free_energy(phi: NDArray[np.float64],
                           a: float, b: float, c: float = 0.0,
                           h: float = 0.0) -> NDArray[np.float64]:
        r"""
        Landau free energy density.

        $$f(\phi) = \frac{a}{2}\phi^2 + \frac{b}{4}\phi^4 + \frac{c}{6}\phi^6 - h\phi$$
        """
        return a / 2 * phi**2 + b / 4 * phi**4 + c / 6 * phi**6 - h * phi

    @staticmethod
    def landau_equilibrium(a: float, b: float, c: float = 0.0,
                           h: float = 0.0) -> List[float]:
        """
        Find equilibrium order parameter(s) from dF/dφ = 0.

        Returns list of real solutions.
        """
        # dF/dφ = aφ + bφ³ + cφ⁵ - h = 0
        # For h = 0: φ = 0 or φ² = (-b ± √(b²-4ac)) / (2c) [if c > 0]
        #           or φ² = -a/b [if c = 0 and b > 0]
        solutions: List[float] = [0.0]

        if abs(h) < 1e-30:
            if abs(c) < 1e-30:
                # φ(aφ + bφ²) = 0 → φ = 0 or φ² = -a/b
                if b != 0 and -a / b > 0:
                    val = math.sqrt(-a / b)
                    solutions.extend([val, -val])
            else:
                # φ(a + bφ² + cφ⁴) = 0
                disc = b**2 - 4 * a * c
                if disc >= 0:
                    sq = math.sqrt(disc)
                    for root in [(-b + sq) / (2 * c), (-b - sq) / (2 * c)]:
                        if root > 0:
                            val = math.sqrt(root)
                            solutions.extend([val, -val])
        else:
            # Numerical solution with Newton's method from multiple initial guesses
            for phi0 in np.linspace(-3.0, 3.0, 20):
                phi = phi0
                for _ in range(100):
                    f = a * phi + b * phi**3 + c * phi**5 - h
                    fp = a + 3 * b * phi**2 + 5 * c * phi**4
                    if abs(fp) < 1e-30:
                        break
                    phi -= f / fp
                    if abs(f) < 1e-12:
                        # Check it's not a duplicate
                        is_dup = any(abs(phi - s) < 1e-6 for s in solutions)
                        if not is_dup:
                            solutions.append(phi)
                        break

        return sorted(solutions)

    @staticmethod
    def critical_exponents_mf() -> Dict[str, float]:
        """Mean-field critical exponents (Landau theory)."""
        return {
            "alpha": 0.0,    # Specific heat: C ~ |t|^{-α}
            "beta": 0.5,     # Magnetisation: m ~ |t|^β
            "gamma": 1.0,    # Susceptibility: χ ~ |t|^{-γ}
            "delta": 3.0,    # Critical isotherm: m ~ h^{1/δ}
            "nu": 0.5,       # Correlation length: ξ ~ |t|^{-ν}
            "eta": 0.0,      # Anomalous dimension
        }
