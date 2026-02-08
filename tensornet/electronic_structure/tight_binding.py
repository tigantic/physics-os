"""
Semi-Empirical & Tight-Binding Methods — DFTB (Slater-Koster),
Extended Hückel, SCC-DFTB.

Domain VIII.3 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Slater-Koster Tight-Binding
# ---------------------------------------------------------------------------

class SlaterKosterTB:
    r"""
    Two-centre Slater-Koster tight-binding model.

    Hamiltonian matrix elements between atom-centred orbitals:
    $$H_{\mu\nu} = \langle\mu|H|\nu\rangle = \sum_{\alpha}
      t_{\alpha}(|\mathbf{R}|)\,SK_{\alpha}(\hat{\mathbf{R}})$$

    where SK are direction cosine angular factors and
    $t_\alpha$ are the Slater-Koster parameters ($ss\sigma$, $sp\sigma$,
    $pp\sigma$, $pp\pi$, etc.) as functions of bond length.

    Parametrised form: $t_\alpha(r) = t_\alpha^0 \exp(-\lambda(r/r_0 - 1))$.
    """

    def __init__(self, n_atoms: int = 2,
                 orbitals_per_atom: int = 4) -> None:
        self.n_atoms = n_atoms
        self.n_orb = orbitals_per_atom  # s, px, py, pz
        self.n_basis = n_atoms * self.n_orb

        # Default SK parameters (sp3 like Si)
        self.params = {
            'ss_sigma': -1.82,
            'sp_sigma': 1.96,
            'pp_sigma': 3.17,
            'pp_pi': -0.84,
            'r0': 2.35,       # Å (equilibrium bond)
            'lambda': 2.0,    # decay rate
        }

    def hopping(self, r: float, param_key: str) -> float:
        """Distance-dependent hopping: t(r) = t₀ exp(−λ(r/r₀ − 1))."""
        t0 = self.params[param_key]
        return t0 * math.exp(-self.params['lambda'] * (r / self.params['r0'] - 1))

    def direction_cosines(self, Rij: NDArray) -> Tuple[float, float, float]:
        """Direction cosines l, m, n = R_x/|R|, R_y/|R|, R_z/|R|."""
        r = float(np.linalg.norm(Rij))
        if r < 1e-12:
            return (0.0, 0.0, 1.0)
        return (Rij[0] / r, Rij[1] / r, Rij[2] / r)

    def sk_matrix_element(self, orb_i: str, orb_j: str,
                             Rij: NDArray) -> float:
        """Slater-Koster matrix element between two orbitals.

        orb_i, orb_j in {'s', 'px', 'py', 'pz'}.
        """
        r = float(np.linalg.norm(Rij))
        l, m, n = self.direction_cosines(Rij)

        ss = self.hopping(r, 'ss_sigma')
        sp = self.hopping(r, 'sp_sigma')
        pps = self.hopping(r, 'pp_sigma')
        ppp = self.hopping(r, 'pp_pi')

        key = (orb_i, orb_j)
        if key == ('s', 's'):
            return ss
        elif key == ('s', 'px'):
            return l * sp
        elif key == ('s', 'py'):
            return m * sp
        elif key == ('s', 'pz'):
            return n * sp
        elif key == ('px', 's'):
            return -l * sp
        elif key == ('py', 's'):
            return -m * sp
        elif key == ('pz', 's'):
            return -n * sp
        elif key == ('px', 'px'):
            return l**2 * pps + (1 - l**2) * ppp
        elif key == ('py', 'py'):
            return m**2 * pps + (1 - m**2) * ppp
        elif key == ('pz', 'pz'):
            return n**2 * pps + (1 - n**2) * ppp
        elif key in [('px', 'py'), ('py', 'px')]:
            return l * m * (pps - ppp)
        elif key in [('px', 'pz'), ('pz', 'px')]:
            return l * n * (pps - ppp)
        elif key in [('py', 'pz'), ('pz', 'py')]:
            return m * n * (pps - ppp)
        return 0.0

    def build_hamiltonian(self, positions: NDArray,
                             on_site: Optional[NDArray] = None) -> NDArray:
        """Build full tight-binding Hamiltonian.

        positions: (n_atoms, 3).
        on_site: (n_atoms, n_orb) on-site energies.
        """
        N = self.n_basis
        H = np.zeros((N, N))
        orb_labels = ['s', 'px', 'py', 'pz'][:self.n_orb]

        # On-site energies
        if on_site is None:
            on_site = np.zeros((self.n_atoms, self.n_orb))
            for a in range(self.n_atoms):
                on_site[a] = [-5.0, -1.0, -1.0, -1.0][:self.n_orb]

        for a in range(self.n_atoms):
            for mu in range(self.n_orb):
                H[a * self.n_orb + mu, a * self.n_orb + mu] = on_site[a, mu]

        # Hopping
        for a in range(self.n_atoms):
            for b in range(a + 1, self.n_atoms):
                Rij = positions[b] - positions[a]
                r = float(np.linalg.norm(Rij))
                if r > 2 * self.params['r0']:
                    continue
                for mu, orb_mu in enumerate(orb_labels):
                    for nu, orb_nu in enumerate(orb_labels):
                        h = self.sk_matrix_element(orb_mu, orb_nu, Rij)
                        i = a * self.n_orb + mu
                        j = b * self.n_orb + nu
                        H[i, j] = h
                        H[j, i] = h

        return H

    def band_structure(self, positions: NDArray, k_path: NDArray) -> NDArray:
        """Compute band structure along k-path for a periodic system.

        Simple 1D chain version.
        """
        n_k = len(k_path)
        bands = np.zeros((n_k, self.n_basis))

        for ik, k in enumerate(k_path):
            H_k = self.build_hamiltonian(positions)
            # Add Bloch phase for periodic
            for a in range(self.n_atoms):
                for b in range(self.n_atoms):
                    phase = np.exp(1j * k * (positions[b, 0] - positions[a, 0]))
                    for mu in range(self.n_orb):
                        for nu in range(self.n_orb):
                            i = a * self.n_orb + mu
                            j = b * self.n_orb + nu
                            H_k[i, j] *= phase.real  # simplified

            evals = np.linalg.eigvalsh(H_k)
            bands[ik] = evals

        return bands


# ---------------------------------------------------------------------------
#  SCC-DFTB (Self-Consistent-Charge DFTB)
# ---------------------------------------------------------------------------

class SCCDFTB:
    r"""
    Self-Consistent-Charge Density-Functional Tight-Binding.

    Total energy:
    $$E = \sum_i^{\text{occ}} \langle\psi_i|H_0|\psi_i\rangle
      + \frac{1}{2}\sum_{AB}\gamma_{AB}\Delta q_A\Delta q_B + E_{\text{rep}}$$

    Charge transfer: $\Delta q_A = q_A^0 - q_A$ via Mulliken analysis.

    $\gamma_{AB}$ interpolation (Elstner):
    $$\gamma_{AB}(R) = \frac{1}{R} - S(R, U_A, U_B)$$

    where U = Hubbard parameter = chemical hardness ≈ d²E/dN².
    """

    def __init__(self, n_atoms: int = 2, hubbard_U: Optional[NDArray] = None,
                 tb: Optional[SlaterKosterTB] = None) -> None:
        self.n_atoms = n_atoms
        self.U = hubbard_U if hubbard_U is not None else np.ones(n_atoms) * 0.3
        self.tb = tb or SlaterKosterTB(n_atoms)

        self.delta_q = np.zeros(n_atoms)
        self.total_energy: float = 0.0

    def gamma_function(self, R: float, U_A: float, U_B: float) -> float:
        """Short-range correction to 1/R Coulomb."""
        if R < 1e-10:
            return 0.5 * (U_A + U_B)
        tau_A = 3.2 * U_A
        tau_B = 3.2 * U_B
        tau_avg = 0.5 * (tau_A + tau_B)
        S = math.exp(-tau_avg * R) * (1 + tau_avg * R / 2)
        return 1.0 / R - S / R

    def mulliken_charges(self, C: NDArray, S: NDArray, n_occ: int) -> NDArray:
        """Mulliken population analysis: q_A = Σ_{μ∈A} (PS)_{μμ}."""
        P = 2 * C[:, :n_occ] @ C[:, :n_occ].T
        PS = P @ S
        charges = np.zeros(self.n_atoms)
        n_orb = self.tb.n_orb
        for a in range(self.n_atoms):
            for mu in range(n_orb):
                charges[a] += PS[a * n_orb + mu, a * n_orb + mu]
        return charges

    def scf(self, positions: NDArray, n_electrons: int,
              max_iter: int = 50, tol: float = 1e-6) -> Dict[str, float]:
        """SCC-DFTB self-consistent cycle."""
        n_occ = n_electrons // 2
        H0 = self.tb.build_hamiltonian(positions)
        S = np.eye(self.tb.n_basis)

        q0 = np.ones(self.n_atoms) * (n_electrons / self.n_atoms)

        for iteration in range(max_iter):
            # Build charge-corrected Hamiltonian
            H = H0.copy()
            n_orb = self.tb.n_orb
            for a in range(self.n_atoms):
                for b in range(self.n_atoms):
                    gab = self.gamma_function(
                        float(np.linalg.norm(positions[b] - positions[a])),
                        self.U[a], self.U[b])
                    shift = 0.5 * gab * (self.delta_q[a] + self.delta_q[b])
                    for mu in range(n_orb):
                        i = a * n_orb + mu
                        for nu in range(n_orb):
                            j = b * n_orb + nu
                            H[i, j] += 0.5 * S[i, j] * shift

            evals, evecs = np.linalg.eigh(H)
            q_new = self.mulliken_charges(evecs, S, n_occ)
            delta_q_new = q0 - q_new

            converged = float(np.max(np.abs(delta_q_new - self.delta_q))) < tol
            self.delta_q = delta_q_new

            if converged:
                break

        E_band = 2 * sum(evals[:n_occ])
        E_coulomb = 0.0
        for a in range(self.n_atoms):
            for b in range(self.n_atoms):
                R = float(np.linalg.norm(positions[b] - positions[a]))
                gab = self.gamma_function(R, self.U[a], self.U[b])
                E_coulomb += 0.5 * gab * self.delta_q[a] * self.delta_q[b]

        self.total_energy = E_band + E_coulomb
        return {
            'E_total': self.total_energy,
            'E_band': E_band,
            'E_coulomb': E_coulomb,
            'iterations': iteration + 1,
        }


# ---------------------------------------------------------------------------
#  Extended Hückel Theory
# ---------------------------------------------------------------------------

class ExtendedHuckel:
    r"""
    Extended Hückel Theory (EHT) — Wolfsberg-Helmholz approximation.

    $$H_{\mu\nu} = \frac{K}{2}(H_{\mu\mu} + H_{\nu\nu})S_{\mu\nu}$$

    where $K = 1.75$ (Wolfsberg-Helmholz constant),
    $H_{\mu\mu} = -\text{VSIE}$ (valence state ionisation energy),
    $S_{\mu\nu}$ = overlap integral.

    No self-consistency — single diagonalisation.
    Good for: orbital symmetry, Walsh diagrams, band structures.
    """

    K_WH: float = 1.75

    # VSIE values (eV) — negative of ionisation energy
    VSIE = {
        'H_1s': -13.6,
        'C_2s': -21.4, 'C_2p': -11.4,
        'N_2s': -26.0, 'N_2p': -13.4,
        'O_2s': -32.3, 'O_2p': -14.8,
        'Si_3s': -17.3, 'Si_3p': -9.2,
    }

    def __init__(self) -> None:
        pass

    def hamiltonian(self, H_diag: NDArray, S: NDArray) -> NDArray:
        """Build EHT Hamiltonian from diagonal energies and overlap.

        H_diag: (n,) on-site energies.
        S: (n, n) overlap matrix.
        """
        n = len(H_diag)
        H = np.zeros((n, n))
        for i in range(n):
            H[i, i] = H_diag[i]
            for j in range(i + 1, n):
                H[i, j] = self.K_WH / 2 * (H_diag[i] + H_diag[j]) * S[i, j]
                H[j, i] = H[i, j]
        return H

    def solve(self, H_diag: NDArray, S: NDArray) -> Tuple[NDArray, NDArray]:
        """Solve generalised eigenvalue problem HC = SCε."""
        H = self.hamiltonian(H_diag, S)
        evals_S, U_S = np.linalg.eigh(S)
        X = U_S @ np.diag(1 / np.sqrt(np.maximum(evals_S, 1e-10))) @ U_S.T
        H_prime = X.T @ H @ X
        evals, evecs = np.linalg.eigh(H_prime)
        C = X @ evecs
        return evals, C
