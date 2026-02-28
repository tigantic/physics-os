"""
Many-Body Localization & Disorder — Random-field XXZ, level statistics,
participation ratio, entanglement growth.

Domain VII.5 — NEW.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Random-Field XXZ Spin Chain
# ---------------------------------------------------------------------------

class RandomFieldXXZ:
    r"""
    Random-field XXZ model:

    $$H = J\sum_i(S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + \Delta S_i^z S_{i+1}^z)
      + \sum_i h_i S_i^z$$

    $h_i \in [-W, W]$ uniformly distributed.

    MBL transition at $W_c \approx 3.5J$ for $\Delta = 1$ (Heisenberg).

    Exact diagonalisation for small chains (L ≤ 16).
    """

    def __init__(self, L: int = 10, J: float = 1.0,
                 Delta: float = 1.0, W: float = 5.0,
                 seed: int = 42) -> None:
        self.L = L
        self.J = J
        self.Delta = Delta
        self.W = W

        rng = np.random.default_rng(seed)
        self.h_fields = W * (2 * rng.random(L) - 1)

        self.dim = 2**L
        self._H: Optional[NDArray] = None
        self._evals: Optional[NDArray] = None
        self._evecs: Optional[NDArray] = None

    def _pauli_z(self, site: int) -> NDArray:
        """σ^z on site i, identity elsewhere."""
        op = np.eye(1)
        for s in range(self.L):
            sz = np.array([[1, 0], [0, -1]], dtype=float)
            I2 = np.eye(2)
            op = np.kron(op, sz if s == site else I2)
        return op

    def _pauli_plus(self, site: int) -> NDArray:
        """σ^+ on site i."""
        op = np.eye(1)
        for s in range(self.L):
            sp = np.array([[0, 1], [0, 0]], dtype=float)
            I2 = np.eye(2)
            op = np.kron(op, sp if s == site else I2)
        return op

    def _pauli_minus(self, site: int) -> NDArray:
        """σ^− on site i."""
        op = np.eye(1)
        for s in range(self.L):
            sm = np.array([[0, 0], [1, 0]], dtype=float)
            I2 = np.eye(2)
            op = np.kron(op, sm if s == site else I2)
        return op

    def build_hamiltonian(self) -> NDArray:
        """Build full Hamiltonian matrix."""
        if self._H is not None:
            return self._H

        H = np.zeros((self.dim, self.dim))

        for i in range(self.L - 1):
            # XX + YY = 2(S+S- + S-S+) (using 1/2 factor for spin-1/2)
            Sp_i = self._pauli_plus(i)
            Sm_i = self._pauli_minus(i)
            Sp_j = self._pauli_plus(i + 1)
            Sm_j = self._pauli_minus(i + 1)
            Sz_i = 0.5 * self._pauli_z(i)
            Sz_j = 0.5 * self._pauli_z(i + 1)

            H += 0.5 * self.J * (Sp_i @ Sm_j + Sm_i @ Sp_j)
            H += self.J * self.Delta * Sz_i @ Sz_j

        # Random field
        for i in range(self.L):
            H += self.h_fields[i] * 0.5 * self._pauli_z(i)

        self._H = H
        return H

    def diagonalize(self) -> Tuple[NDArray, NDArray]:
        """Full exact diagonalization."""
        if self._evals is not None:
            return self._evals, self._evecs
        H = self.build_hamiltonian()
        self._evals, self._evecs = np.linalg.eigh(H)
        return self._evals, self._evecs


# ---------------------------------------------------------------------------
#  Level Statistics
# ---------------------------------------------------------------------------

class LevelStatistics:
    r"""
    Energy level statistics diagnostics for MBL transition.

    Gap ratio:
    $$r_n = \frac{\min(\delta_n, \delta_{n+1})}{\max(\delta_n, \delta_{n+1})}$$

    GOE (thermal): $\langle r\rangle \approx 0.5307$.
    Poisson (MBL): $\langle r\rangle \approx 0.3863$.
    """

    def __init__(self, energies: NDArray) -> None:
        self.energies = np.sort(energies)
        self.gaps = np.diff(self.energies)

    def gap_ratios(self) -> NDArray:
        """Compute r_n = min(δ_n, δ_{n+1}) / max(δ_n, δ_{n+1})."""
        g = self.gaps
        r = np.minimum(g[:-1], g[1:]) / (np.maximum(g[:-1], g[1:]) + 1e-30)
        return r

    def mean_gap_ratio(self) -> float:
        """<r> averaged over bulk (middle 50% of spectrum)."""
        r = self.gap_ratios()
        n = len(r)
        start = n // 4
        end = 3 * n // 4
        return float(np.mean(r[start:end]))

    @staticmethod
    def goe_prediction() -> float:
        return 0.5307

    @staticmethod
    def poisson_prediction() -> float:
        return 0.3863

    def is_thermal(self, threshold: float = 0.48) -> bool:
        """True if <r> is closer to GOE than Poisson."""
        return self.mean_gap_ratio() > threshold


# ---------------------------------------------------------------------------
#  Participation Ratio & IPR
# ---------------------------------------------------------------------------

class ParticipationRatio:
    r"""
    Inverse participation ratio (IPR) for wave function localisation.

    $$\text{IPR}_2 = \sum_i |\psi_i|^4$$

    Generalised: $\text{IPR}_q = \sum_i |\psi_i|^{2q}$.

    Extended state: IPR ~ 1/D.
    Localised state: IPR ~ O(1).

    Fractal dimension: $D_q = \frac{\ln \text{IPR}_q}{(1-q)\ln D}$.
    """

    def __init__(self, eigenstates: NDArray) -> None:
        """eigenstates: shape (dim, n_states) — columns are states."""
        self.states = eigenstates
        self.dim = eigenstates.shape[0]

    def ipr(self, state_idx: int, q: int = 2) -> float:
        """IPR_q for a specific eigenstate."""
        psi = self.states[:, state_idx]
        return float(np.sum(np.abs(psi)**(2 * q)))

    def mean_ipr(self, q: int = 2) -> float:
        """Mean IPR over all eigenstates."""
        n_states = self.states.shape[1]
        return float(np.mean([self.ipr(i, q) for i in range(n_states)]))

    def fractal_dimension(self, state_idx: int, q: int = 2) -> float:
        """Multifractal dimension D_q."""
        ipr_val = self.ipr(state_idx, q)
        if q == 1 or ipr_val <= 0:
            return 0.0
        return math.log(ipr_val) / ((1 - q) * math.log(self.dim))


# ---------------------------------------------------------------------------
#  Entanglement Dynamics
# ---------------------------------------------------------------------------

class EntanglementDynamics:
    r"""
    Entanglement entropy growth after a quench.

    Thermal phase: $S(t) \sim vt$ (ballistic, until saturation $S \sim L$).
    MBL phase: $S(t) \sim \ln t$ (logarithmic growth).

    Computes half-chain von Neumann entanglement entropy.
    """

    def __init__(self, L: int, H: NDArray) -> None:
        self.L = L
        self.dim = 2**L
        self.H = H
        self.dim_A = 2**(L // 2)
        self.dim_B = 2**(L - L // 2)

    def evolve_and_measure(self, psi0: NDArray, times: NDArray) -> NDArray:
        """Time-evolve ψ₀ and measure entanglement at each time.

        Uses full diagonalization for exact evolution.
        Returns array of entanglement entropies.
        """
        evals, evecs = np.linalg.eigh(self.H)
        coeffs = evecs.T @ psi0  # project onto eigenbasis

        S = np.zeros(len(times))
        for ti, t in enumerate(times):
            phases = np.exp(-1j * evals * t)
            psi_t = evecs @ (coeffs * phases)
            S[ti] = self._entanglement_entropy(psi_t)

        return S

    def _entanglement_entropy(self, psi: NDArray) -> float:
        """Half-chain von Neumann entropy via SVD."""
        rho = psi.reshape(self.dim_A, self.dim_B)
        s = np.linalg.svd(rho, compute_uv=False)
        s = s[s > 1e-15]
        probs = s**2
        return float(-np.sum(probs * np.log(probs + 1e-30)))

    def page_entropy(self) -> float:
        """Page value for random state: S ≈ ln(d_A) − d_A/(2d_B)."""
        d_A = min(self.dim_A, self.dim_B)
        d_B = max(self.dim_A, self.dim_B)
        return math.log(d_A) - d_A / (2 * d_B)
