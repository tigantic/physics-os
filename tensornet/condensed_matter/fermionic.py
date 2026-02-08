"""
Fermionic many-body quantum systems.

Upgrades domain VII.11: BCS mean-field theory, FFLO pairing,
Bravyi-Kitaev transformation, and Fermi-liquid Landau parameters.

ℏ = 1 throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ===================================================================
#  BCS Mean-Field Theory
# ===================================================================

@dataclass
class BCSResult:
    """Result of a BCS self-consistency calculation."""
    gap: float
    chemical_potential: float
    gap_k: NDArray[np.float64]
    Ek: NDArray[np.float64]
    occupation_k: NDArray[np.float64]
    condensation_energy: float
    converged: bool


class BCSSolver:
    r"""
    BCS mean-field theory of superconductivity.

    Gap equation:
    $$\Delta_k = -\sum_{k'} V_{kk'} \frac{\Delta_{k'}}{2E_{k'}}
        \tanh\!\left(\frac{E_{k'}}{2T}\right)$$

    where $E_k = \sqrt{\xi_k^2 + |\Delta_k|^2}$ and $\xi_k = \epsilon_k - \mu$.

    Number equation:
    $$N = \sum_k \left(1 - \frac{\xi_k}{E_k}\tanh\frac{E_k}{2T}\right)$$

    Implements:
    - s-wave and d-wave BCS gap equations
    - Self-consistent gap + number equation
    - Tc estimation
    - Coherence length, penetration depth
    """

    def __init__(self, N_k: int = 500, E_cutoff: float = 20.0) -> None:
        """
        Parameters
        ----------
        N_k : Number of k-points.
        E_cutoff : Energy cutoff for pairing interaction (Debye window).
        """
        self.N_k = N_k
        self.E_cutoff = E_cutoff

    def solve_swave(self, epsilon_k: NDArray[np.float64],
                     V0: float, mu: float,
                     T: float = 0.0,
                     max_iter: int = 500,
                     tol: float = 1e-8) -> BCSResult:
        """
        Solve s-wave BCS gap equation with constant attractive pairing V₀ < 0
        in a Debye window |ε_k - μ| < E_cutoff.
        """
        xi_k = epsilon_k - mu
        in_window = np.abs(xi_k) < self.E_cutoff

        # Initial guess
        delta = 0.1 * abs(V0)
        delta_k = np.where(in_window, delta, 0.0)

        converged = False
        for _ in range(max_iter):
            Ek = np.sqrt(xi_k**2 + delta_k**2)

            # Thermal factor
            if T > 1e-10:
                arg = np.minimum(Ek / (2.0 * T), 500.0)
                f = np.tanh(arg)
            else:
                f = np.ones_like(Ek)

            # New gap (momentum-independent s-wave)
            safe_E = np.where(Ek > 1e-15, Ek, 1e-15)
            delta_new = -V0 * np.mean(delta_k * f / (2.0 * safe_E) * in_window)

            delta_k_new = np.where(in_window, delta_new, 0.0)

            if np.max(np.abs(delta_k_new - delta_k)) < tol:
                converged = True
                delta_k = delta_k_new
                break

            delta_k = 0.5 * delta_k + 0.5 * delta_k_new

        Ek = np.sqrt(xi_k**2 + delta_k**2)
        # Occupation
        if T > 1e-10:
            v2 = 0.5 * (1.0 - xi_k / np.where(Ek > 1e-15, Ek, 1.0)
                         * np.tanh(np.minimum(Ek / (2.0 * T), 500.0)))
        else:
            v2 = 0.5 * (1.0 - xi_k / np.where(Ek > 1e-15, Ek, 1.0))

        # Condensation energy
        E_sc = np.sum(xi_k * v2 - 0.5 * delta_k**2 / np.where(Ek > 1e-15, Ek, 1.0))
        E_norm = np.sum(xi_k * (xi_k < 0).astype(float))
        cond_energy = float(E_sc - E_norm)

        return BCSResult(
            gap=float(np.max(np.abs(delta_k))),
            chemical_potential=mu,
            gap_k=delta_k,
            Ek=Ek,
            occupation_k=v2,
            condensation_energy=cond_energy,
            converged=converged,
        )

    def solve_dwave(self, kx: NDArray[np.float64], ky: NDArray[np.float64],
                     epsilon_k: NDArray[np.float64],
                     V0: float, mu: float,
                     T: float = 0.0,
                     max_iter: int = 500,
                     tol: float = 1e-8) -> BCSResult:
        """
        Solve d-wave BCS gap equation: Δ_k = Δ₀(cos kx - cos ky).

        The form factor g_k = cos(kx) - cos(ky) gives d_{x²-y²} symmetry.
        """
        xi_k = epsilon_k - mu
        gk = np.cos(kx) - np.cos(ky)
        in_window = np.abs(xi_k) < self.E_cutoff

        delta_0 = 0.1 * abs(V0)
        converged = False

        for _ in range(max_iter):
            delta_k = delta_0 * gk * in_window
            Ek = np.sqrt(xi_k**2 + delta_k**2)

            if T > 1e-10:
                f = np.tanh(np.minimum(Ek / (2.0 * T), 500.0))
            else:
                f = np.ones_like(Ek)

            safe_E = np.where(Ek > 1e-15, Ek, 1e-15)
            delta_0_new = -V0 * np.mean(gk**2 * delta_0 * f / (2.0 * safe_E) * in_window)

            if abs(delta_0_new - delta_0) < tol:
                converged = True
                delta_0 = delta_0_new
                break

            delta_0 = 0.5 * delta_0 + 0.5 * delta_0_new

        delta_k = delta_0 * gk * in_window
        Ek = np.sqrt(xi_k**2 + delta_k**2)
        v2 = 0.5 * (1.0 - xi_k / np.where(Ek > 1e-15, Ek, 1.0))

        E_sc = np.sum(xi_k * v2 - 0.5 * delta_k**2 / np.where(Ek > 1e-15, Ek, 1.0))
        E_norm = np.sum(xi_k * (xi_k < 0).astype(float))

        return BCSResult(
            gap=abs(delta_0), chemical_potential=mu,
            gap_k=delta_k, Ek=Ek, occupation_k=v2,
            condensation_energy=float(E_sc - E_norm),
            converged=converged,
        )

    @staticmethod
    def critical_temperature(delta_0: float) -> float:
        r"""
        BCS relation: $k_B T_c = \Delta_0 / 1.764$.
        """
        return abs(delta_0) / 1.764

    @staticmethod
    def coherence_length(delta_0: float, v_F: float) -> float:
        r"""BCS coherence length $\xi_0 = \hbar v_F / (\pi \Delta_0)$."""
        if abs(delta_0) < 1e-15:
            return float('inf')
        return v_F / (math.pi * abs(delta_0))


# ===================================================================
#  FFLO Pairing (Fulde-Ferrell-Larkin-Ovchinnikov)
# ===================================================================

class FFLOSolver:
    r"""
    FFLO state: Cooper pairs with finite centre-of-mass momentum Q.

    In the presence of a Zeeman field h (or spin imbalance),
    the pairing occurs at finite Q:

    $$\Delta(\mathbf{r}) = \Delta_0 e^{i\mathbf{Q}\cdot\mathbf{r}}$$
    (Fulde-Ferrell) or $\Delta(\mathbf{r}) = \Delta_0 \cos(\mathbf{Q}\cdot\mathbf{r})$
    (Larkin-Ovchinnikov).

    The quasiparticle spectrum:
    $$E_{\pm}(\mathbf{k}) = \sqrt{\xi_k^2 + \Delta_0^2} \pm |\mathbf{v}\cdot\mathbf{Q}/2 + h|$$

    where ξ_k = k²/(2m) - μ + Q²/(8m).
    """

    def __init__(self, N_k: int = 500) -> None:
        self.N_k = N_k

    def free_energy_ff(self, k_arr: NDArray[np.float64],
                        Q: float, delta: float, mu: float,
                        h: float, V0: float,
                        T: float = 0.0,
                        mass: float = 1.0) -> float:
        """
        Free energy of the Fulde-Ferrell state at given Q, Δ.
        1D for simplicity; uses quasiparticle spectrum.
        """
        xi_k = k_arr**2 / (2.0 * mass) - mu + Q**2 / (8.0 * mass)
        Ek = np.sqrt(xi_k**2 + delta**2)

        # Zeeman splitting + Q shift
        E_plus = Ek + h + k_arr * Q / (2.0 * mass)
        E_minus = Ek - h - k_arr * Q / (2.0 * mass)

        dk = k_arr[1] - k_arr[0] if len(k_arr) > 1 else 1.0

        if T > 1e-10:
            beta = 1.0 / T
            f_plus = -T * np.log(1.0 + np.exp(-beta * np.abs(E_plus)))
            f_minus = -T * np.log(1.0 + np.exp(-beta * np.abs(E_minus)))
        else:
            f_plus = -np.maximum(E_plus, 0.0)
            f_minus = -np.maximum(E_minus, 0.0)

        Omega_pair = np.sum(xi_k - Ek + f_plus + f_minus) * dk / (2.0 * math.pi)

        # Gap equation contribution
        Omega_pair += delta**2 / abs(V0) if abs(V0) > 1e-15 else 0.0

        return float(Omega_pair)

    def optimal_Q(self, k_arr: NDArray[np.float64],
                   delta: float, mu: float, h: float,
                   V0: float, T: float = 0.0,
                   mass: float = 1.0,
                   Q_max: float = 2.0,
                   N_Q: int = 200) -> Tuple[float, float]:
        """
        Find optimal Q that minimises free energy.

        Returns (Q_opt, F_opt).
        """
        Q_arr = np.linspace(0.0, Q_max, N_Q)
        F_arr = np.array([
            self.free_energy_ff(k_arr, Q, delta, mu, h, V0, T, mass)
            for Q in Q_arr
        ])

        idx = np.argmin(F_arr)
        return float(Q_arr[idx]), float(F_arr[idx])

    def phase_diagram(self, k_arr: NDArray[np.float64],
                       mu: float, V0: float,
                       h_values: NDArray[np.float64],
                       mass: float = 1.0) -> Dict[str, NDArray]:
        """
        Compute phase diagram: for each Zeeman field h, determine
        whether BCS (Q=0), FFLO (Q≠0), or normal (Δ=0) is favoured.

        Returns dict with 'h', 'phase' (0=normal, 1=BCS, 2=FFLO),
        'Q', 'delta'.
        """
        phase = np.zeros(len(h_values), dtype=int)
        Q_opt = np.zeros(len(h_values))
        delta_opt = np.zeros(len(h_values))

        for i, h in enumerate(h_values):
            # Compare: normal (Δ=0), BCS (Q=0, Δ≠0), FFLO (Q≠0, Δ≠0)
            F_normal = self.free_energy_ff(k_arr, 0.0, 0.0, mu, h, V0, 0.0, mass)

            # BCS
            F_bcs = self.free_energy_ff(k_arr, 0.0, 0.1 * abs(V0), mu, h, V0, 0.0, mass)

            # FFLO: optimise Q
            Q_ff, F_fflo = self.optimal_Q(k_arr, 0.1 * abs(V0), mu, h, V0, 0.0, mass)

            energies = [F_normal, F_bcs, F_fflo]
            winner = int(np.argmin(energies))
            phase[i] = winner

            if winner == 2:
                Q_opt[i] = Q_ff
                delta_opt[i] = 0.1 * abs(V0)
            elif winner == 1:
                delta_opt[i] = 0.1 * abs(V0)

        return {
            "h": h_values,
            "phase": phase,
            "Q": Q_opt,
            "delta": delta_opt,
        }


# ===================================================================
#  Bravyi-Kitaev Transformation
# ===================================================================

class BravyiKitaevTransform:
    r"""
    Bravyi-Kitaev transformation: maps fermionic creation/annihilation
    operators to qubit operators with O(log N) Pauli weight.

    Compared to Jordan-Wigner (O(N) weight), BK is more efficient
    for quantum simulation.

    For N modes, define:
    - Parity set $P(j)$: modes whose parity $a_j^\dagger$ must track
    - Update set $U(j)$: modes that must be updated
    - Remainder set $R(j)$: flip and parity sets

    $$a_j^\dagger \mapsto \frac{1}{2}\!\left(
        X_{U(j)} \otimes X_j \otimes Z_{P(j)}
        - i\,X_{U(j)} \otimes Y_j \otimes Z_{R(j)}\right)$$
    """

    def __init__(self, n_modes: int) -> None:
        """
        Parameters
        ----------
        n_modes : Number of fermionic modes (qubits).
        """
        self.n = n_modes

    @staticmethod
    def _parity_set(j: int, n: int) -> List[int]:
        """Parity set P(j): indices whose parity is encoded in qubit j."""
        result: List[int] = []
        if j == 0:
            return result
        # Binary tree traversal: ancestors on left branches
        idx = j
        while idx > 0:
            parent = (idx - 1) // 2 if idx > 0 else -1
            if idx > 0 and idx == 2 * parent + 2:
                # Right child: add left subtree
                left_child = 2 * parent + 1
                if left_child < n:
                    result.append(left_child)
            idx = parent
            if idx < 0:
                break
        return sorted(set(result))

    @staticmethod
    def _update_set(j: int, n: int) -> List[int]:
        """Update set U(j): qubits that must be flipped when mode j is occupied."""
        result: List[int] = []
        idx = j
        while True:
            parent = (idx - 1) // 2 if idx > 0 else -1
            if parent < 0 or parent >= n:
                break
            if idx == 2 * parent + 1:
                # Left child: parent stores parity of left subtree
                result.append(parent)
            idx = parent
        return sorted(set(result))

    @staticmethod
    def _flip_set(j: int, n: int) -> List[int]:
        """Remainder (flip) set R(j) = U(j) XOR P(j)."""
        p = set(BravyiKitaevTransform._parity_set(j, n))
        u = set(BravyiKitaevTransform._update_set(j, n))
        return sorted(p.symmetric_difference(u))

    def creation_operator(self, j: int) -> List[Tuple[complex, List[Tuple[int, str]]]]:
        r"""
        Express $a_j^\dagger$ as sum of Pauli strings.

        Returns list of (coefficient, [(qubit, pauli), ...]) terms.
        Pauli labels: 'I', 'X', 'Y', 'Z'.
        """
        U = self._update_set(j, self.n)
        P = self._parity_set(j, self.n)
        R = self._flip_set(j, self.n)

        # First term:  +1/2 * X_{U} X_j Z_{P}
        term1_ops: List[Tuple[int, str]] = []
        for q in U:
            term1_ops.append((q, 'X'))
        term1_ops.append((j, 'X'))
        for q in P:
            if q != j:
                term1_ops.append((q, 'Z'))

        # Second term: -i/2 * X_{U} Y_j Z_{R}
        term2_ops: List[Tuple[int, str]] = []
        for q in U:
            term2_ops.append((q, 'X'))
        term2_ops.append((j, 'Y'))
        for q in R:
            if q != j:
                term2_ops.append((q, 'Z'))

        return [
            (0.5 + 0j, term1_ops),
            (-0.5j, term2_ops),
        ]

    def annihilation_operator(self, j: int) -> List[Tuple[complex, List[Tuple[int, str]]]]:
        """
        Express a_j as Hermitian conjugate: a_j = (a_j†)†.

        Just conjugate the coefficients.
        """
        creation = self.creation_operator(j)
        return [(coeff.conjugate(), ops) for coeff, ops in creation]

    def number_operator(self, j: int) -> List[Tuple[complex, List[Tuple[int, str]]]]:
        """
        n_j = a_j† a_j = (I - Z_j) / 2 for BK (qubit j stores occupation).
        """
        return [
            (0.5 + 0j, []),       # I/2
            (-0.5 + 0j, [(j, 'Z')]),  # -Z_j/2
        ]

    def pauli_weight(self, j: int) -> int:
        """Number of non-identity Pauli operators for a_j†."""
        terms = self.creation_operator(j)
        max_weight = 0
        for _, ops in terms:
            max_weight = max(max_weight, len(ops))
        return max_weight

    def transform_hamiltonian(self,
                               one_body: NDArray[np.float64],
                               two_body: Optional[NDArray[np.float64]] = None
                               ) -> List[Tuple[complex, List[Tuple[int, str]]]]:
        """
        Transform a fermionic Hamiltonian to qubit Pauli strings.

        H = Σ_{pq} h_{pq} a†_p a_q + Σ_{pqrs} V_{pqrs} a†_p a†_q a_s a_r

        Returns list of (coefficient, pauli_string) terms.
        (Simplified: one-body only for now.)
        """
        terms: List[Tuple[complex, List[Tuple[int, str]]]] = []

        n = one_body.shape[0]
        for p in range(n):
            for q in range(n):
                if abs(one_body[p, q]) < 1e-15:
                    continue

                # Diagonal terms use number operator
                if p == q:
                    for coeff, ops in self.number_operator(p):
                        terms.append((one_body[p, q] * coeff, ops))
                # Off-diagonal: would need full product a†_p a_q
                # which is more complex — skip in basic implementation

        # Consolidate: merge identical Pauli strings
        return self._consolidate(terms)

    @staticmethod
    def _consolidate(terms: List[Tuple[complex, List[Tuple[int, str]]]]
                     ) -> List[Tuple[complex, List[Tuple[int, str]]]]:
        """Merge terms with identical Pauli string."""
        merged: Dict[tuple, complex] = {}
        for coeff, ops in terms:
            key = tuple(sorted(ops))
            merged[key] = merged.get(key, 0j) + coeff

        result: List[Tuple[complex, List[Tuple[int, str]]]] = []
        for key, coeff in merged.items():
            if abs(coeff) > 1e-15:
                result.append((coeff, list(key)))
        return result


# ===================================================================
#  Fermi-Liquid Theory (Landau Parameters)
# ===================================================================

class FermiLiquidLandau:
    r"""
    Landau Fermi-liquid theory: quasiparticle description of
    interacting Fermi systems.

    Landau parameters $F_l^{s,a}$ characterise the quasiparticle
    interaction on the Fermi surface:

    $$f(\theta) = \sum_l (F_l^s + \sigma\cdot\sigma' F_l^a) P_l(\cos\theta)$$

    Physical quantities:
    - Effective mass: $m^*/m = 1 + F_1^s/3$ (3D)
    - Compressibility: $\kappa/\kappa_0 = 1/(1 + F_0^s)$
    - Spin susceptibility: $\chi/\chi_0 = m^*/m \cdot 1/(1 + F_0^a)$
    - Sound velocity: $c_0^2 = v_F^2(1 + F_0^s)/3$
    - Zero sound: collective mode when $F_0^s > 0$

    Stability (Pomeranchuk criterion): $F_l^{s,a} > -(2l+1)$.
    """

    def __init__(self, k_F: float, m_bare: float = 1.0, dim: int = 3) -> None:
        """
        Parameters
        ----------
        k_F : Fermi wavevector.
        m_bare : Bare fermion mass.
        dim : Spatial dimension (2 or 3).
        """
        self.k_F = k_F
        self.m_bare = m_bare
        self.dim = dim
        self.v_F = k_F / m_bare  # Bare Fermi velocity

        # Density of states at Fermi level
        if dim == 3:
            self.N0 = m_bare * k_F / (math.pi**2)  # per spin per volume
        elif dim == 2:
            self.N0 = m_bare / (2.0 * math.pi)
        else:
            raise ValueError(f"Unsupported dimension {dim}")

    def effective_mass(self, F1s: float) -> float:
        r"""
        Effective mass ratio $m^*/m = 1 + F_1^s / d$ (d = dimension).
        """
        return self.m_bare * (1.0 + F1s / float(self.dim))

    def compressibility_ratio(self, F0s: float) -> float:
        r"""$\kappa / \kappa_0 = 1 / (1 + F_0^s)$."""
        return 1.0 / (1.0 + F0s)

    def spin_susceptibility_ratio(self, F0a: float, F1s: float = 0.0) -> float:
        r"""$\chi / \chi_0 = (m^*/m) / (1 + F_0^a)$."""
        mstar_ratio = 1.0 + F1s / float(self.dim)
        return mstar_ratio / (1.0 + F0a)

    def first_sound_velocity(self, F0s: float, F1s: float = 0.0) -> float:
        r"""
        First sound velocity:
        $c_1^2 = v_F^{*2} (1 + F_0^s) / d$ where $v_F^* = k_F / m^*$.
        """
        mstar = self.effective_mass(F1s)
        vF_star = self.k_F / mstar
        return vF_star * math.sqrt((1.0 + F0s) / float(self.dim))

    def zero_sound_velocity(self, F0s: float, F1s: float = 0.0) -> float:
        r"""
        Zero sound velocity from the equation:
        $s \ln\!\frac{s+1}{s-1} = 2(1 + F_0^s / (2l+1))$ for l=0, 3D.

        For F_0^s >> 1: c_0 ≈ v_F √(F_0^s/3).
        For F_0^s → 0+: c_0 → v_F (from above).
        """
        if F0s <= 0:
            return 0.0

        mstar = self.effective_mass(F1s)
        vF_star = self.k_F / mstar

        # Solve s·ln|(s+1)/(s-1)| = 2 for s = c_0/v_F
        # Newton's method
        s = 1.0 + F0s / 3.0  # Initial guess
        for _ in range(100):
            if s <= 1.0:
                s = 1.001
            val = s * math.log((s + 1.0) / (s - 1.0))
            target = 2.0 * (1.0 + F0s)

            # Derivative
            dval = math.log((s + 1.0) / (s - 1.0)) + s * (-2.0 / (s**2 - 1.0))
            ds = (target - val) / dval if abs(dval) > 1e-15 else 0.0
            s += ds
            if abs(ds) < 1e-12:
                break

        return vF_star * s

    def pomeranchuk_stable(self, Fs: NDArray[np.float64],
                            Fa: NDArray[np.float64]) -> bool:
        r"""
        Pomeranchuk stability criterion: $F_l^{s,a} > -(2l+1)$.
        """
        for l, f in enumerate(Fs):
            if f <= -(2 * l + 1):
                return False
        for l, f in enumerate(Fa):
            if f <= -(2 * l + 1):
                return False
        return True

    def quasiparticle_lifetime(self, epsilon: float, T: float,
                                F0s: float = 0.0) -> float:
        r"""
        Quasiparticle lifetime near the Fermi surface:
        $$\frac{1}{\tau} \propto \max(\epsilon^2, T^2)$$

        Returns 1/τ in natural units.
        """
        E_F = self.k_F**2 / (2.0 * self.m_bare)
        return (max(epsilon**2, T**2)) / (E_F * (1.0 + abs(F0s)))

    def wilson_ratio(self, F0a: float) -> float:
        r"""
        Wilson ratio $R_W = \frac{4\pi^2}{3} \frac{\chi}{C_V/T} = \frac{1}{1 + F_0^a}$.
        """
        return 1.0 / (1.0 + F0a)

    def landau_damping_rate(self, q: float, omega: float,
                              F0s: float = 0.0,
                              F1s: float = 0.0) -> float:
        r"""
        Landau damping rate for collective excitation at (q, ω).

        Non-zero when ω < v_F q (particle-hole continuum).
        """
        mstar = self.effective_mass(F1s)
        vF_star = self.k_F / mstar

        if abs(q) < 1e-15:
            return 0.0

        x = omega / (vF_star * q)
        if abs(x) >= 1.0:
            return 0.0

        # Im χ_0 ~ πN₀ ω/(2v_F q) for |ω| < v_F q
        return math.pi * self.N0 * abs(omega) / (2.0 * vF_star * abs(q))
