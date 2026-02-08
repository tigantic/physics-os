"""
Topological Phases of Matter — Toric code, Chern number, topological
entanglement entropy, anyonic braiding.

Domain VII.4 — NEW.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Toric Code (Kitaev)
# ---------------------------------------------------------------------------

class ToricCode:
    r"""
    Kitaev toric code on an L×L lattice (periodic).

    Star operators: $A_s = \prod_{j\in\text{star}(s)} \sigma_j^x$
    Plaquette operators: $B_p = \prod_{j\in\text{plaq}(p)} \sigma_j^z$

    Hamiltonian: $H = -J_s\sum_s A_s - J_p\sum_p B_p$

    Ground-state degeneracy on torus = 4 (non-trivial topology).
    Excitations: e-particles (A_s = −1) and m-particles (B_p = −1).
    """

    def __init__(self, L: int = 4, Js: float = 1.0,
                 Jp: float = 1.0) -> None:
        self.L = L
        self.Js = Js
        self.Jp = Jp

        # 2L² edges: L² horizontal + L² vertical
        self.n_qubits = 2 * L**2
        self.n_stars = L**2
        self.n_plaquettes = L**2

        # Classical spin state (σ^z basis): +1 or -1
        self.spins = np.ones(self.n_qubits, dtype=int)

    def _edge_index(self, x: int, y: int, direction: str) -> int:
        """Map (x, y, direction) to edge index.

        direction: 'h' (horizontal, right) or 'v' (vertical, up).
        """
        x = x % self.L
        y = y % self.L
        if direction == 'h':
            return y * self.L + x
        else:
            return self.L**2 + y * self.L + x

    def star_operator(self, x: int, y: int) -> List[int]:
        """Edges touching vertex (x, y)."""
        return [
            self._edge_index(x, y, 'h'),          # right
            self._edge_index(x - 1, y, 'h'),      # left
            self._edge_index(x, y, 'v'),           # up
            self._edge_index(x, y - 1, 'v'),       # down
        ]

    def plaquette_operator(self, x: int, y: int) -> List[int]:
        """Edges around plaquette with lower-left corner (x, y)."""
        return [
            self._edge_index(x, y, 'h'),           # bottom
            self._edge_index(x, y + 1, 'h'),       # top
            self._edge_index(x, y, 'v'),            # left
            self._edge_index(x + 1, y, 'v'),        # right
        ]

    def star_eigenvalue(self, x: int, y: int) -> int:
        """A_s = product of σ^x (in σ^z basis, flip detection)."""
        edges = self.star_operator(x, y)
        return int(np.prod(self.spins[edges]))

    def plaquette_eigenvalue(self, x: int, y: int) -> int:
        """B_p = product of σ^z."""
        edges = self.plaquette_operator(x, y)
        return int(np.prod(self.spins[edges]))

    def energy(self) -> float:
        """H = −Js ΣAs − Jp ΣBp."""
        E = 0.0
        for y in range(self.L):
            for x in range(self.L):
                E -= self.Js * self.star_eigenvalue(x, y)
                E -= self.Jp * self.plaquette_eigenvalue(x, y)
        return E

    def create_e_pair(self, path: List[Tuple[int, int, str]]) -> None:
        """Create e-particle pair by flipping σ^z along a path."""
        for x, y, d in path:
            idx = self._edge_index(x, y, d)
            self.spins[idx] *= -1

    def count_excitations(self) -> Tuple[int, int]:
        """Count e-particles and m-particles."""
        n_e = 0
        n_m = 0
        for y in range(self.L):
            for x in range(self.L):
                if self.star_eigenvalue(x, y) == -1:
                    n_e += 1
                if self.plaquette_eigenvalue(x, y) == -1:
                    n_m += 1
        return n_e, n_m

    def ground_state_degeneracy(self) -> int:
        """On a torus: GSD = 4 (2^{2g}, genus g=1)."""
        return 4


# ---------------------------------------------------------------------------
#  Chern Number Computation
# ---------------------------------------------------------------------------

class ChernNumberCalculator:
    r"""
    Chern number from Berry curvature on a 2D Brillouin zone.

    $$C = \frac{1}{2\pi}\int_{\text{BZ}} F_{xy}\,dk_x\,dk_y$$

    For a 2-band model $H(\mathbf{k}) = \mathbf{d}(\mathbf{k})\cdot\boldsymbol{\sigma}$:
    $$C = \frac{1}{4\pi}\int \hat{d}\cdot(\partial_{k_x}\hat{d}\times\partial_{k_y}\hat{d})\,d^2k$$

    Implements:
    - Haldane model
    - Qi-Wu-Zhang (QWZ) model
    - Lattice Chern number via U(1) link method (Fukui-Hatsugai-Suzuki)
    """

    def __init__(self, nk: int = 50) -> None:
        self.nk = nk
        self.kx = np.linspace(-np.pi, np.pi, nk, endpoint=False)
        self.ky = np.linspace(-np.pi, np.pi, nk, endpoint=False)

    def qwz_hamiltonian(self, kx: float, ky: float,
                          m: float = 1.5) -> NDArray:
        """QWZ model: H = sin(kx)σx + sin(ky)σy + (m−cos(kx)−cos(ky))σz."""
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)

        dx = math.sin(kx)
        dy = math.sin(ky)
        dz = m - math.cos(kx) - math.cos(ky)

        return dx * sx + dy * sy + dz * sz

    def _ground_state(self, H: NDArray) -> NDArray:
        """Ground-state eigenvector of 2×2 Hamiltonian."""
        w, v = np.linalg.eigh(H)
        return v[:, 0]

    def chern_number_fhs(self, hamiltonian_func, **kwargs) -> float:
        """Fukui-Hatsugai-Suzuki lattice Chern number.

        Uses U(1) link variables on discretised BZ.
        Exact integer for any mesh.
        """
        nk = self.nk
        dkx = 2 * np.pi / nk
        dky = 2 * np.pi / nk

        states = np.zeros((nk, nk, 2), dtype=complex)
        for i in range(nk):
            for j in range(nk):
                H = hamiltonian_func(self.kx[i], self.ky[j], **kwargs)
                states[i, j] = self._ground_state(H)

        # U(1) link variables
        total_F = 0.0
        for i in range(nk):
            for j in range(nk):
                i1 = (i + 1) % nk
                j1 = (j + 1) % nk

                U1 = np.vdot(states[i, j], states[i1, j])
                U1 /= abs(U1) + 1e-30
                U2 = np.vdot(states[i1, j], states[i1, j1])
                U2 /= abs(U2) + 1e-30
                U3 = np.vdot(states[i1, j1], states[i, j1])
                U3 /= abs(U3) + 1e-30
                U4 = np.vdot(states[i, j1], states[i, j])
                U4 /= abs(U4) + 1e-30

                F = np.log(U1 * U2 * U3 * U4)
                total_F += F.imag

        return total_F / (2 * np.pi)

    def haldane_model(self, kx: float, ky: float, M: float = 0.0,
                        t1: float = 1.0, t2: float = 0.2,
                        phi: float = np.pi / 2) -> NDArray:
        """Haldane model on honeycomb lattice.

        H = ε(k)I + d(k)·σ with:
        - d_z = M − 2t₂sinφ Σ sin(k·b_i)
        - d_{x,y} from NN hopping.
        """
        # Honeycomb reciprocal vectors
        a1 = np.array([1, 0])
        a2 = np.array([0.5, math.sqrt(3) / 2])
        b1 = a2 - a1
        b2 = -(a1 + a2)
        b3 = a1

        k = np.array([kx, ky])
        # NN
        f = sum(np.exp(1j * np.dot(k, bi)) for bi in [b1, b2, b3])
        # NNN
        nnn_vecs = [a1, a2, a2 - a1, -a1, -a2, a1 - a2]
        g = sum(np.sin(np.dot(k, ai)) for ai in nnn_vecs[:3])

        dx = t1 * f.real
        dy = t1 * f.imag
        dz = M - 2 * t2 * math.sin(phi) * g

        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)

        return dx * sx + dy * sy + dz * sz


# ---------------------------------------------------------------------------
#  Topological Entanglement Entropy
# ---------------------------------------------------------------------------

class TopologicalEntanglementEntropy:
    r"""
    Topological entanglement entropy (TEE) from entanglement scaling.

    Kitaev-Preskill / Levin-Wen construction:

    $$S_A = \alpha|\partial A| - \gamma$$

    $\gamma = \ln\mathcal{D}$ where $\mathcal{D} = \sqrt{\sum_a d_a^2}$ is
    the total quantum dimension.

    For toric code: $\gamma = \ln 2$ (two anyon types with $d=1$).
    For Fibonacci anyons: $\gamma = \ln\phi$ where $\phi = (1+\sqrt{5})/2$.
    """

    @staticmethod
    def tee_toric_code() -> float:
        """γ = ln 2 for Z₂ toric code."""
        return math.log(2)

    @staticmethod
    def tee_fibonacci() -> float:
        """γ = ln φ for Fibonacci anyons."""
        phi = (1 + math.sqrt(5)) / 2
        return math.log(phi)

    @staticmethod
    def total_quantum_dimension(quantum_dims: List[float]) -> float:
        """D = √(Σ d_a²)."""
        return math.sqrt(sum(d**2 for d in quantum_dims))

    @staticmethod
    def kitaev_preskill(S_A: float, S_B: float, S_C: float,
                          S_AB: float, S_BC: float, S_AC: float,
                          S_ABC: float) -> float:
        """Kitaev-Preskill combination to extract γ.

        γ = −(S_A + S_B + S_C − S_AB − S_BC − S_AC + S_ABC)
        """
        return -(S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC)


# ---------------------------------------------------------------------------
#  Anyonic Braiding Statistics
# ---------------------------------------------------------------------------

class AnyonicBraiding:
    r"""
    Anyonic braiding matrices for Ising and Fibonacci anyons.

    Ising anyons (σ particles):
    - Fusion: σ×σ = 1 + ψ
    - Braiding matrix: R = e^{−iπ/8} on the 2D fusion space.

    Fibonacci anyons (τ particles):
    - Fusion: τ×τ = 1 + τ
    - F-matrix (recoupling):
      $F = \begin{pmatrix}\phi^{-1} & \phi^{-1/2}\\\phi^{-1/2} & -\phi^{-1}\end{pmatrix}$
    """

    @staticmethod
    def ising_R_matrix() -> NDArray:
        """Ising anyon R-matrix in the {1, ψ} fusion basis."""
        R = np.zeros((2, 2), dtype=complex)
        R[0, 0] = np.exp(-1j * np.pi / 8)
        R[1, 1] = np.exp(3j * np.pi / 8)
        return R

    @staticmethod
    def fibonacci_F_matrix() -> NDArray:
        """Fibonacci anyon F-matrix."""
        phi = (1 + math.sqrt(5)) / 2
        return np.array([
            [phi**(-1), phi**(-0.5)],
            [phi**(-0.5), -phi**(-1)]
        ])

    @staticmethod
    def fibonacci_R_matrix() -> NDArray:
        """Fibonacci anyon R-matrix."""
        return np.array([
            [np.exp(4j * np.pi / 5), 0],
            [0, np.exp(-3j * np.pi / 5)]
        ], dtype=complex)

    @staticmethod
    def braid_word(generators: List[Tuple[int, int]],
                     R: NDArray, dim: int) -> NDArray:
        """Compute braid group representation from a word of generators.

        generators: list of (strand_i, strand_j) crossings.
        Returns unitary matrix.
        """
        result = np.eye(dim, dtype=complex)
        for _i, _j in generators:
            result = R @ result  # simplified: same R for all crossings
        return result

    @staticmethod
    def topological_spin(theta: float) -> complex:
        """Topological spin: T_a = e^{2πi h_a}."""
        return np.exp(2j * np.pi * theta)
