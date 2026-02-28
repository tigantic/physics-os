"""
Beyond-DFT Methods — Hartree-Fock, MP2, CCSD, CASSCF/DMRG.

Domain VIII.2 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Hartree-Fock (Restricted, Closed-Shell)
# ---------------------------------------------------------------------------

class RestrictedHartreeFock:
    r"""
    Restricted Hartree-Fock (RHF) in an orthonormal basis.

    Fock matrix:
    $$F_{\mu\nu} = h_{\mu\nu} + \sum_{\lambda\sigma}P_{\lambda\sigma}
      \left[(\mu\nu|\lambda\sigma) - \tfrac{1}{2}(\mu\lambda|\nu\sigma)\right]$$

    Density matrix: $P_{\mu\nu} = 2\sum_{i}^{N/2}C_{\mu i}C_{\nu i}$

    Roothaan-Hall: $\mathbf{FC} = \mathbf{SC}\boldsymbol{\varepsilon}$
    """

    def __init__(self, n_basis: int = 10, n_electrons: int = 2) -> None:
        self.n_basis = n_basis
        self.n_occ = n_electrons // 2
        self.h_core: Optional[NDArray] = None    # (n, n)
        self.eri: Optional[NDArray] = None         # (n, n, n, n) two-electron integrals
        self.S: Optional[NDArray] = None           # overlap
        self.C: Optional[NDArray] = None           # MO coefficients
        self.P: Optional[NDArray] = None           # density matrix
        self.eigenvalues: Optional[NDArray] = None

    def set_integrals(self, h_core: NDArray, eri: NDArray,
                        S: Optional[NDArray] = None) -> None:
        """Set one-electron (h_core), two-electron (eri), overlap (S) integrals."""
        self.h_core = h_core
        self.eri = eri
        self.S = S if S is not None else np.eye(self.n_basis)

    def _build_fock(self) -> NDArray:
        """Build Fock matrix from density and integrals."""
        n = self.n_basis
        F = self.h_core.copy()
        for mu in range(n):
            for nu in range(n):
                for lam in range(n):
                    for sig in range(n):
                        F[mu, nu] += self.P[lam, sig] * (
                            self.eri[mu, nu, lam, sig]
                            - 0.5 * self.eri[mu, lam, nu, sig])
        return F

    def scf(self, max_iter: int = 100, tol: float = 1e-8) -> Dict[str, float]:
        """Self-consistent field iteration."""
        n = self.n_basis

        # Symmetric orthogonalisation: X = S^{-1/2}
        evals_S, U_S = np.linalg.eigh(self.S)
        X = U_S @ np.diag(1 / np.sqrt(evals_S)) @ U_S.T

        # Initial guess: core Hamiltonian
        F0 = X.T @ self.h_core @ X
        evals, evecs = np.linalg.eigh(F0)
        self.C = X @ evecs
        self.P = 2 * self.C[:, :self.n_occ] @ self.C[:, :self.n_occ].T

        E_prev = 0.0
        for iteration in range(max_iter):
            F = self._build_fock()

            # Electronic energy
            E_elec = 0.5 * float(np.sum(self.P * (self.h_core + F)))

            # Convergence
            if abs(E_elec - E_prev) < tol:
                self.eigenvalues = evals
                return {'E_electronic': E_elec, 'iterations': iteration + 1,
                        'converged': True}
            E_prev = E_elec

            # Diagonalise in orthogonal basis
            F_prime = X.T @ F @ X
            evals, evecs = np.linalg.eigh(F_prime)
            self.C = X @ evecs
            self.P = 2 * self.C[:, :self.n_occ] @ self.C[:, :self.n_occ].T

        self.eigenvalues = evals
        return {'E_electronic': E_elec, 'iterations': max_iter, 'converged': False}


# ---------------------------------------------------------------------------
#  MP2 Perturbation Theory
# ---------------------------------------------------------------------------

class MP2Correlation:
    r"""
    Møller-Plesset second-order perturbation theory.

    $$E_{\text{corr}}^{(2)} = -\sum_{i<j}^{\text{occ}}\sum_{a<b}^{\text{virt}}
      \frac{|\langle ij\|ab\rangle|^2}{\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j}$$

    where $\langle ij\|ab\rangle = \langle ij|ab\rangle - \langle ij|ba\rangle$ (antisymmetrised).
    """

    def __init__(self, C: NDArray, eigenvalues: NDArray, eri: NDArray,
                 n_occ: int) -> None:
        self.C = C
        self.eps = eigenvalues
        self.eri_ao = eri
        self.n_occ = n_occ
        self.n_basis = C.shape[0]
        self.n_virt = self.n_basis - n_occ

    def transform_eri(self) -> NDArray:
        """AO → MO integral transformation (O(N^5) naive)."""
        n = self.n_basis
        C = self.C

        # 4-index transform
        tmp1 = np.einsum('pi,pqrs->iqrs', C, self.eri_ao)
        tmp2 = np.einsum('qj,iqrs->ijrs', C, tmp1)
        tmp3 = np.einsum('ra,ijrs->ijas', C, tmp2)
        eri_mo = np.einsum('sb,ijas->ijab', C, tmp3)
        return eri_mo

    def correlation_energy(self) -> float:
        """Compute MP2 correlation energy."""
        eri_mo = self.transform_eri()
        nocc = self.n_occ
        E2 = 0.0

        for i in range(nocc):
            for j in range(nocc):
                for a in range(nocc, self.n_basis):
                    for b in range(nocc, self.n_basis):
                        denom = (self.eps[a] + self.eps[b]
                                 - self.eps[i] - self.eps[j])
                        if abs(denom) < 1e-12:
                            continue
                        direct = eri_mo[i, j, a - nocc, b - nocc]
                        exchange = eri_mo[i, j, b - nocc, a - nocc]
                        E2 += (direct * (2 * direct - exchange)) / denom

        return E2


# ---------------------------------------------------------------------------
#  CCSD (Coupled Cluster Singles and Doubles)
# ---------------------------------------------------------------------------

class CCSDSolver:
    r"""
    Coupled Cluster with Singles and Doubles.

    $$|\Psi\rangle = e^{\hat{T}_1 + \hat{T}_2}|\Phi_0\rangle$$

    Amplitude equations (projected):
    $$\langle\Phi_i^a|e^{-\hat{T}}He^{\hat{T}}|\Phi_0\rangle = 0 \quad\text{(singles)}$$
    $$\langle\Phi_{ij}^{ab}|e^{-\hat{T}}He^{\hat{T}}|\Phi_0\rangle = 0 \quad\text{(doubles)}$$

    Iteratively solved for $t_i^a, t_{ij}^{ab}$ amplitudes.
    """

    def __init__(self, n_occ: int, n_virt: int, fock_diag: NDArray,
                 eri_mo: NDArray) -> None:
        self.nocc = n_occ
        self.nvirt = n_virt
        self.fock = fock_diag
        self.eri = eri_mo

        self.t1 = np.zeros((n_occ, n_virt))
        self.t2 = np.zeros((n_occ, n_occ, n_virt, n_virt))

    def _denom_1(self, i: int, a: int) -> float:
        return self.fock[i] - self.fock[self.nocc + a]

    def _denom_2(self, i: int, j: int, a: int, b: int) -> float:
        return (self.fock[i] + self.fock[j]
                - self.fock[self.nocc + a] - self.fock[self.nocc + b])

    def update_amplitudes(self) -> None:
        """One iteration of the CCSD amplitude equations (CCD-level approx)."""
        nocc, nvirt = self.nocc, self.nvirt

        t1_new = np.zeros_like(self.t1)
        t2_new = np.zeros_like(self.t2)

        for i in range(nocc):
            for a in range(nvirt):
                r1 = self.fock[self.nocc + a] - self.fock[i]
                # Add doubles contribution to singles
                for j in range(nocc):
                    for b in range(nvirt):
                        r1 += self.eri[i, j, a, b] * self.t1[j, b]
                t1_new[i, a] = -self.t1[i, a] * r1 / (self._denom_1(i, a) + 1e-15)

        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvirt):
                    for b in range(nvirt):
                        r2 = self.eri[i, j, a, b]
                        d = self._denom_2(i, j, a, b)
                        if abs(d) > 1e-15:
                            t2_new[i, j, a, b] = r2 / d

        self.t1 = t1_new
        self.t2 = t2_new

    def correlation_energy(self) -> float:
        """CCSD correlation energy."""
        E = 0.0
        for i in range(self.nocc):
            for j in range(self.nocc):
                for a in range(self.nvirt):
                    for b in range(self.nvirt):
                        E += self.eri[i, j, a, b] * (
                            self.t2[i, j, a, b]
                            + self.t1[i, a] * self.t1[j, b])
        return E

    def solve(self, max_iter: int = 50, tol: float = 1e-8) -> float:
        """Iterate to convergence."""
        E_prev = 0.0
        for it in range(max_iter):
            self.update_amplitudes()
            E_corr = self.correlation_energy()
            if abs(E_corr - E_prev) < tol:
                return E_corr
            E_prev = E_corr
        return E_corr


# ---------------------------------------------------------------------------
#  CASSCF / Multi-Reference
# ---------------------------------------------------------------------------

class CASSCFSolver:
    r"""
    Complete Active Space SCF (CASSCF).

    Active space: CAS(n_el, n_orb) — full CI within active window,
    orbital optimisation outside.

    $$|\Psi\rangle = \sum_I c_I |\Phi_I\rangle$$

    where {Φ_I} are all Slater determinants in the active space.

    For large active spaces: use DMRG as the CI solver (DMRG-CASSCF).
    """

    def __init__(self, n_active_el: int = 4, n_active_orb: int = 4) -> None:
        self.n_el = n_active_el
        self.n_orb = n_active_orb
        self.n_det = math.comb(n_active_orb, n_active_el // 2)**2
        self.ci_coeffs: Optional[NDArray] = None
        self.energy: float = 0.0

    def generate_determinants(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Generate all alpha/beta occupations for CAS(n_el, n_orb)."""
        from itertools import combinations
        n_alpha = self.n_el // 2
        n_beta = self.n_el - n_alpha

        alpha_configs = list(combinations(range(self.n_orb), n_alpha))
        beta_configs = list(combinations(range(self.n_orb), n_beta))

        return [(a, b) for a in alpha_configs for b in beta_configs]

    def solve_ci(self, h1e: NDArray, h2e: NDArray) -> float:
        """Full CI in active space.

        h1e: (n_orb, n_orb) one-electron integrals.
        h2e: (n_orb, n_orb, n_orb, n_orb) two-electron integrals.
        """
        dets = self.generate_determinants()
        n_det = len(dets)

        H_ci = np.zeros((n_det, n_det))
        for I in range(n_det):
            occ_a, occ_b = dets[I]
            # Diagonal
            for i in occ_a:
                H_ci[I, I] += h1e[i, i]
            for i in occ_b:
                H_ci[I, I] += h1e[i, i]
            for i in occ_a:
                for j in occ_a:
                    if i < j:
                        H_ci[I, I] += h2e[i, j, i, j] - h2e[i, j, j, i]
            for i in occ_b:
                for j in occ_b:
                    if i < j:
                        H_ci[I, I] += h2e[i, j, i, j] - h2e[i, j, j, i]
            for i in occ_a:
                for j in occ_b:
                    H_ci[I, I] += h2e[i, j, i, j]

        evals, evecs = np.linalg.eigh(H_ci)
        self.energy = float(evals[0])
        self.ci_coeffs = evecs[:, 0]
        return self.energy
