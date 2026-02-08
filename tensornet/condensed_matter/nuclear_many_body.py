"""
Nuclear Many-Body Physics — Nuclear shell model CI, Richardson-Gaudin pairing,
chiral EFT interactions.

Domain VII.12 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Nuclear Shell Model (Configuration Interaction)
# ---------------------------------------------------------------------------

class NuclearShellModel:
    r"""
    Nuclear shell model: exact diagonalisation in a valence space.

    $$H = \sum_{\alpha}e_\alpha a_\alpha^\dagger a_\alpha
      + \frac{1}{4}\sum_{\alpha\beta\gamma\delta}\bar{V}_{\alpha\beta\gamma\delta}
      a_\alpha^\dagger a_\beta^\dagger a_\delta a_\gamma$$

    where $\bar{V}$ = antisymmetrised two-body matrix elements (TBME).

    Typical valence space: p-shell (0p_{3/2}, 0p_{1/2}), sd-shell, pf-shell.
    """

    def __init__(self, n_orbits: int = 4, n_particles: int = 2) -> None:
        self.n_orbits = n_orbits
        self.n_particles = n_particles

        self.spe = np.zeros(n_orbits)  # single-particle energies
        self.tbme: Dict[Tuple[int, int, int, int], float] = {}

        self.dim = 0
        self.basis: List[Tuple[int, ...]] = []
        self._H_matrix: Optional[NDArray] = None
        self._evals: Optional[NDArray] = None
        self._evecs: Optional[NDArray] = None

    def set_single_particle_energies(self, spe: NDArray) -> None:
        self.spe = spe.copy()

    def set_tbme(self, a: int, b: int, c: int, d: int, V: float) -> None:
        """Set antisymmetrised TBME ⟨ab|V|cd⟩_AS."""
        self.tbme[(a, b, c, d)] = V
        self.tbme[(b, a, d, c)] = V
        self.tbme[(a, b, d, c)] = -V
        self.tbme[(b, a, c, d)] = -V

    def get_tbme(self, a: int, b: int, c: int, d: int) -> float:
        return self.tbme.get((a, b, c, d), 0.0)

    def build_basis(self) -> None:
        """Build Slater determinant basis: all ways to put n_particles into n_orbits."""
        from itertools import combinations
        self.basis = list(combinations(range(self.n_orbits), self.n_particles))
        self.dim = len(self.basis)

    def _occupation(self, state: Tuple[int, ...]) -> NDArray:
        """Occupation number representation."""
        occ = np.zeros(self.n_orbits, dtype=int)
        for s in state:
            occ[s] = 1
        return occ

    def build_hamiltonian(self) -> NDArray:
        """Build H matrix in Slater determinant basis."""
        if self._H_matrix is not None:
            return self._H_matrix

        if not self.basis:
            self.build_basis()

        H = np.zeros((self.dim, self.dim))

        for I, bra in enumerate(self.basis):
            # Diagonal: one-body + diagonal two-body
            occ = self._occupation(bra)
            E1 = float(np.sum(self.spe * occ))

            E2 = 0.0
            orbs = list(bra)
            for i in range(len(orbs)):
                for j in range(i + 1, len(orbs)):
                    a, b = orbs[i], orbs[j]
                    E2 += self.get_tbme(a, b, a, b)

            H[I, I] = E1 + E2

            # Off-diagonal: one-particle–one-hole and two-particle–two-hole
            for J in range(I + 1, self.dim):
                ket = self.basis[J]
                diff_bra = [o for o in bra if o not in ket]
                diff_ket = [o for o in ket if o not in bra]

                if len(diff_bra) == 2 and len(diff_ket) == 2:
                    a, b = diff_bra
                    c, d = diff_ket
                    # Phase from reordering
                    phase = self._phase(bra, ket, diff_bra, diff_ket)
                    H[I, J] = phase * self.get_tbme(a, b, c, d)
                    H[J, I] = H[I, J]

        self._H_matrix = H
        return H

    def _phase(self, bra: tuple, ket: tuple,
                 diff_bra: list, diff_ket: list) -> int:
        """Compute fermionic sign from permuting creation operators."""
        # Count transpositions needed
        bra_list = list(bra)
        n_perm = 0
        for orb in diff_bra:
            idx = bra_list.index(orb)
            n_perm += idx
            bra_list.pop(idx)

        ket_list = list(ket)
        for orb in diff_ket:
            idx = ket_list.index(orb)
            n_perm += idx
            ket_list.pop(idx)

        return (-1)**n_perm

    def diagonalize(self) -> Tuple[NDArray, NDArray]:
        if self._evals is not None:
            return self._evals, self._evecs
        H = self.build_hamiltonian()
        self._evals, self._evecs = np.linalg.eigh(H)
        return self._evals, self._evecs

    def ground_state_energy(self) -> float:
        evals, _ = self.diagonalize()
        return float(evals[0])

    def excitation_spectrum(self, n: int = 10) -> NDArray:
        evals, _ = self.diagonalize()
        return evals[:n] - evals[0]


# ---------------------------------------------------------------------------
#  Richardson-Gaudin Pairing Model
# ---------------------------------------------------------------------------

class RichardsonGaudinPairing:
    r"""
    Richardson-Gaudin exactly-solvable pairing model.

    $$H = \sum_j \varepsilon_j \hat{N}_j - G \sum_{jj'} P_j^\dagger P_{j'}$$

    $P_j^\dagger = c_{j+}^\dagger c_{j-}^\dagger$ (pair creation).

    Richardson's equations for pair energies $E_\alpha$:
    $$\frac{1}{G} = \sum_j \frac{\Omega_j}{2\varepsilon_j - E_\alpha}
      - \sum_{\beta\neq\alpha}\frac{2}{E_\beta - E_\alpha}$$

    where $\Omega_j$ = degeneracy of level j.
    """

    def __init__(self, levels: NDArray, degeneracies: Optional[NDArray] = None,
                 G: float = 0.5) -> None:
        self.eps = levels.copy()
        self.n_levels = len(levels)
        self.Omega = (degeneracies.copy() if degeneracies is not None
                       else np.ones(self.n_levels) * 2)
        self.G = G

    def richardson_equations(self, E_pairs: NDArray) -> NDArray:
        """Evaluate Richardson's equations (should be zero at solution).

        Returns residual vector.
        """
        n_pairs = len(E_pairs)
        residuals = np.zeros(n_pairs)

        for alpha in range(n_pairs):
            Ea = E_pairs[alpha]
            sum1 = sum(self.Omega[j] / (2 * self.eps[j] - Ea)
                        for j in range(self.n_levels))
            sum2 = sum(2.0 / (E_pairs[beta] - Ea)
                        for beta in range(n_pairs) if beta != alpha)
            residuals[alpha] = 1.0 / self.G - sum1 + sum2

        return residuals

    def solve_richardson(self, n_pairs: int,
                           max_iter: int = 500,
                           tol: float = 1e-10) -> NDArray:
        """Solve Richardson's equations via Newton-Raphson.

        Initial guess: pair energies at 2*ε_j.
        Returns pair energies E_α.
        """
        # Initial guess: slightly below 2*eps for lowest levels
        E = 2 * self.eps[:n_pairs] - 0.1 * self.G

        for iteration in range(max_iter):
            F = self.richardson_equations(E)
            if np.max(np.abs(F)) < tol:
                break

            # Numerical Jacobian
            n = len(E)
            J = np.zeros((n, n))
            dE = 1e-6
            for i in range(n):
                E_p = E.copy()
                E_p[i] += dE
                J[:, i] = (self.richardson_equations(E_p) - F) / dE

            try:
                delta = np.linalg.solve(J, -F)
                E += delta
            except np.linalg.LinAlgError:
                E += 0.01 * np.random.randn(n)

        return E

    def total_energy(self, E_pairs: NDArray) -> float:
        """Total ground-state energy = Σ E_α."""
        return float(np.sum(E_pairs))

    def condensation_energy(self, E_pairs: NDArray) -> float:
        """Energy gain from pairing: E_cond = E_paired − E_unpaired."""
        E_paired = self.total_energy(E_pairs)
        E_unpaired = float(np.sum(2 * self.eps[:len(E_pairs)]))
        return E_paired - E_unpaired


# ---------------------------------------------------------------------------
#  Chiral Effective Field Theory Interaction
# ---------------------------------------------------------------------------

class ChiralEFTInteraction:
    r"""
    Chiral effective field theory (χEFT) nucleon-nucleon potential.

    Leading order (LO): one-pion exchange + contact terms.
    $$V_{\text{OPE}}(r) = -\frac{g_A^2}{4f_\pi^2}
      \frac{e^{-m_\pi r}}{4\pi r}(\boldsymbol{\tau}_1\cdot\boldsymbol{\tau}_2)
      \left[\sigma_1\cdot\sigma_2 + S_{12}\left(1+\frac{3}{m_\pi r}+\frac{3}{(m_\pi r)^2}\right)\right]$$

    Contact terms (LO): $C_S + C_T \sigma_1\cdot\sigma_2$.
    NLO: two-pion exchange + higher contacts.
    """

    def __init__(self, Lambda_UV: float = 500.0) -> None:  # MeV
        # Physical constants
        self.m_pi = 138.0       # MeV, pion mass
        self.f_pi = 92.4        # MeV, pion decay constant
        self.g_A = 1.29         # axial coupling
        self.hbar_c = 197.3     # MeV·fm

        # Cutoff
        self.Lambda = Lambda_UV

        # LO contact LECs (typical values in fm²)
        self.C_S = -0.15  # fm²
        self.C_T = 0.05   # fm²

    def one_pion_exchange(self, r: float) -> Tuple[float, float]:
        """OPE central and tensor components (fm^-1).

        Returns (V_central, V_tensor) in MeV.
        """
        mu = self.m_pi / self.hbar_c  # fm^-1
        x = mu * r
        if x < 0.01:
            x = 0.01

        prefactor = -(self.g_A / (2 * self.f_pi))**2 * self.m_pi**3 / (12 * math.pi)
        yukawa = math.exp(-x) / x

        V_c = prefactor * yukawa
        S12_factor = (1 + 3.0 / x + 3.0 / x**2) * yukawa
        V_t = prefactor * S12_factor

        return V_c, V_t

    def contact_potential(self, p: float, p_prime: float) -> float:
        """LO contact terms in momentum space (MeV fm³).

        V_contact = C_S + C_T σ₁·σ₂ (σ₁·σ₂ depends on channel).
        """
        # Regulator
        f_reg = math.exp(-(p / self.Lambda)**4 - (p_prime / self.Lambda)**4)
        return (self.C_S + self.C_T) * f_reg  # singlet: σ₁·σ₂ = −3

    def nuclear_matter_energy(self, kF: float = 1.33) -> float:
        """Crude Hartree-Fock nuclear matter energy per nucleon.

        kF in fm^-1 (saturation: kF ≈ 1.33 fm^-1).
        Returns E/A in MeV.
        """
        # Kinetic
        m_N = 939  # MeV
        E_kin = 3 * (self.hbar_c * kF)**2 / (10 * m_N)

        # Potential (very crude: just contact)
        rho = 2 * kF**3 / (3 * math.pi**2)  # fm^-3
        E_pot = 0.5 * rho * (self.C_S - 3 * self.C_T) * self.hbar_c**2

        return E_kin + E_pot

    def deuteron_binding_energy(self) -> float:
        """Approximate deuteron binding energy from OPE + contact.

        Exact: B_d = 2.2246 MeV.
        Solve radial Schrödinger (simplified variational).
        """
        # Variational with exponential: ψ(r) = e^{−αr}
        m_N = 939 / self.hbar_c  # fm^-1
        m_pi = self.m_pi / self.hbar_c

        best_E = 0.0
        for alpha_val in np.linspace(0.1, 2.0, 100):
            # Kinetic: ⟨T⟩ = ℏ²α²/m_N (3D, l=0)
            T = self.hbar_c**2 * alpha_val**2 / m_N

            # OPE potential expectation: ∫ e^{−2αr} V(r) 4πr² dr / ∫ e^{−2αr} 4πr² dr
            # With Yukawa: ∫ e^{−(2α+μ)r} r dr / ∫ e^{−2αr} r² dr
            mu = m_pi
            V_prefactor = -(self.g_A / (2 * self.f_pi))**2 * self.m_pi**3 / (12 * math.pi) * self.hbar_c
            ratio = (2 * alpha_val)**3 / ((2 * alpha_val + mu)**2) / 2

            V_exp = V_prefactor * ratio

            E = T + V_exp
            if E < best_E:
                best_E = E

        return abs(best_E)


# ---------------------------------------------------------------------------
#  Nuclear Binding Energy (Bethe-Weizsäcker)
# ---------------------------------------------------------------------------

class BetheWeizsacker:
    r"""
    Semi-empirical mass formula:

    $$B(A,Z) = a_V A - a_S A^{2/3} - a_C\frac{Z(Z-1)}{A^{1/3}}
      - a_A\frac{(A-2Z)^2}{4A} + \delta(A,Z)$$

    Pairing term: $\delta = a_P A^{-1/2}$ if even-even, $-a_P A^{-1/2}$ if odd-odd, 0 otherwise.
    """

    def __init__(self) -> None:
        self.aV = 15.67   # MeV
        self.aS = 17.23
        self.aC = 0.714
        self.aA = 23.285
        self.aP = 12.0

    def binding_energy(self, A: int, Z: int) -> float:
        """Total binding energy B(A,Z) in MeV."""
        N = A - Z
        B = (self.aV * A
              - self.aS * A**(2 / 3)
              - self.aC * Z * (Z - 1) / A**(1 / 3)
              - self.aA * (A - 2 * Z)**2 / (4 * A))

        # Pairing
        if Z % 2 == 0 and N % 2 == 0:
            B += self.aP / A**0.5
        elif Z % 2 == 1 and N % 2 == 1:
            B -= self.aP / A**0.5

        return B

    def binding_per_nucleon(self, A: int, Z: int) -> float:
        return self.binding_energy(A, Z) / A

    def most_stable_Z(self, A: int) -> int:
        """Find most stable isobar for given A."""
        best_Z = 1
        best_B = 0.0
        for Z in range(1, A):
            B = self.binding_energy(A, Z)
            if B > best_B:
                best_B = B
                best_Z = Z
        return best_Z

    def drip_line_neutron(self, Z: int) -> int:
        """Approximate neutron drip line: find A where S_n → 0."""
        for A in range(Z + 1, 4 * Z):
            Sn = self.binding_energy(A, Z) - self.binding_energy(A - 1, Z)
            if Sn < 0:
                return A - 1
        return 3 * Z
