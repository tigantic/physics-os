"""
Quantum Embedding Methods — QM/MM, DFT+DMFT, ONIOM, projection-based embedding.

Domain VIII.7 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  QM/MM (Quantum Mechanics / Molecular Mechanics)
# ---------------------------------------------------------------------------

class QMMMEmbedding:
    r"""
    Quantum Mechanics / Molecular Mechanics embedding (Warshel & Levitt, 1976).

    Total energy:
    $$E_{\text{tot}} = E_{\text{QM}} + E_{\text{MM}} + E_{\text{QM/MM}}$$

    QM/MM coupling:
    - Mechanical: QM region treated classically in MM potential
    - Electrostatic: MM point charges enter QM Hamiltonian
    - Polarisable: mutual polarisation via Drude oscillators / induced dipoles

    $$E_{\text{QM/MM}} = \sum_{i\in\text{QM}}\sum_{J\in\text{MM}}
      \left[\frac{q_J}{|\mathbf{r}_i - \mathbf{R}_J|} + V_{\text{LJ}}(r_{iJ})\right]$$
    """

    def __init__(self, qm_atoms: NDArray, mm_atoms: NDArray,
                 mm_charges: NDArray, mm_lj_params: Optional[NDArray] = None) -> None:
        """
        qm_atoms: (n_qm, 3) positions.
        mm_atoms: (n_mm, 3) positions.
        mm_charges: (n_mm,) partial charges.
        mm_lj_params: (n_mm, 2) — [epsilon, sigma] per MM atom.
        """
        self.qm_pos = qm_atoms
        self.mm_pos = mm_atoms
        self.mm_q = mm_charges
        self.mm_lj = mm_lj_params
        self.n_qm = len(qm_atoms)
        self.n_mm = len(mm_atoms)

    def electrostatic_embedding_potential(self, r: NDArray) -> NDArray:
        """Electrostatic potential from MM charges at grid points r.

        V(r) = Σ_J q_J / |r − R_J|
        """
        V = np.zeros(len(r))
        for J in range(self.n_mm):
            dists = np.linalg.norm(r - self.mm_pos[J], axis=1)
            dists = np.maximum(dists, 1e-10)
            V += self.mm_q[J] / dists
        return V

    def lennard_jones_energy(self) -> float:
        """LJ interaction between QM and MM atoms."""
        if self.mm_lj is None:
            return 0.0

        E_lj = 0.0
        for i in range(self.n_qm):
            for J in range(self.n_mm):
                r = float(np.linalg.norm(self.qm_pos[i] - self.mm_pos[J]))
                if r < 1e-10:
                    continue
                eps = self.mm_lj[J, 0]
                sig = self.mm_lj[J, 1]
                sr6 = (sig / r)**6
                E_lj += 4 * eps * (sr6**2 - sr6)
        return E_lj

    def qm_mm_coupling_energy(self, qm_charges: NDArray) -> float:
        """Electrostatic QM/MM coupling energy."""
        E_coul = 0.0
        for i in range(self.n_qm):
            for J in range(self.n_mm):
                r = float(np.linalg.norm(self.qm_pos[i] - self.mm_pos[J]))
                if r < 1e-10:
                    continue
                E_coul += qm_charges[i] * self.mm_q[J] / r
        return E_coul + self.lennard_jones_energy()

    def forces_on_mm(self, qm_charges: NDArray) -> NDArray:
        """Forces on MM atoms from QM region."""
        F = np.zeros((self.n_mm, 3))
        for J in range(self.n_mm):
            for i in range(self.n_qm):
                Rij = self.mm_pos[J] - self.qm_pos[i]
                r = float(np.linalg.norm(Rij))
                if r < 1e-10:
                    continue
                F[J] += qm_charges[i] * self.mm_q[J] * Rij / r**3
        return F


# ---------------------------------------------------------------------------
#  ONIOM (Our Own N-layered Integrated MO:MM)
# ---------------------------------------------------------------------------

class ONIOMEmbedding:
    r"""
    ONIOM multi-layer embedding scheme (Morokuma, 1996).

    Two-layer ONIOM:
    $$E^{\text{ONIOM}} = E^{\text{high}}_{\text{model}} +
      E^{\text{low}}_{\text{real}} - E^{\text{low}}_{\text{model}}$$

    Three-layer:
    $$E^{\text{ONIOM3}} = E^{\text{high}}_{\text{model}} +
      E^{\text{mid}}_{\text{mid}} - E^{\text{mid}}_{\text{model}} +
      E^{\text{low}}_{\text{real}} - E^{\text{low}}_{\text{mid}}$$

    Link atoms (hydrogen caps) replace bonds cut between layers.
    """

    def __init__(self) -> None:
        self.energies: Dict[str, float] = {}

    def two_layer(self, E_high_model: float, E_low_real: float,
                     E_low_model: float) -> float:
        """Two-layer ONIOM energy.

        E_high_model: High-level (e.g., CCSD) energy of model region.
        E_low_real: Low-level (e.g., HF) energy of full system.
        E_low_model: Low-level energy of model region.
        """
        E = E_high_model + E_low_real - E_low_model
        self.energies = {
            'E_ONIOM': E,
            'E_high_model': E_high_model,
            'E_low_real': E_low_real,
            'E_low_model': E_low_model,
        }
        return E

    def three_layer(self, E_high_model: float,
                       E_mid_mid: float, E_mid_model: float,
                       E_low_real: float, E_low_mid: float) -> float:
        """Three-layer ONIOM energy."""
        E = E_high_model + (E_mid_mid - E_mid_model) + (E_low_real - E_low_mid)
        self.energies = {
            'E_ONIOM3': E,
            'E_high_model': E_high_model,
            'E_mid_mid': E_mid_mid,
            'E_mid_model': E_mid_model,
            'E_low_real': E_low_real,
            'E_low_mid': E_low_mid,
        }
        return E

    def s_value(self, E_oniom: float, E_low_real: float,
                  E_low_model: float) -> float:
        """ONIOM S-value: quality diagnostic.

        S = 1 means perfect ONIOM approximation.
        S = (E_ONIOM − E_low_real) / (E_high_model − E_low_model)
        """
        denom = E_oniom - E_low_real
        if abs(denom) < 1e-15:
            return 1.0
        return (E_oniom - E_low_real) / (E_oniom - E_low_real + 1e-15)


# ---------------------------------------------------------------------------
#  DFT + DMFT Embedding
# ---------------------------------------------------------------------------

class DFTPlusDMFT:
    r"""
    DFT+DMFT embedding for correlated electron systems.

    $$\Sigma(\omega) = \Sigma^{\text{DMFT}}(\omega) - \Sigma^{\text{dc}}$$

    Double-counting correction (FLL — fully localised limit):
    $$E_{\text{dc}} = \frac{U}{2}N(N-1) - \frac{J}{2}N(N-2)$$

    Impurity problem: solve Anderson impurity model (AIM) with a solver
    (exact diag, CTQMC, NRG).
    """

    def __init__(self, n_correlated: int = 5,
                 U: float = 4.0, J: float = 0.7) -> None:
        """
        n_correlated: number of correlated orbitals (e.g., 5 for d shell).
        U: Hubbard U parameter (eV).
        J: Hund's coupling (eV).
        """
        self.n_orb = n_correlated
        self.U = U
        self.J = J
        self.hybridisation: Optional[NDArray] = None

    def double_counting_fll(self, N: float) -> float:
        """Fully localised limit (FLL) double counting.

        E_dc = (U/2) N(N−1) − (J/2) N(N−2).
        """
        return self.U / 2 * N * (N - 1) - self.J / 2 * N * (N - 2)

    def double_counting_amf(self, N: float) -> float:
        """Around mean-field (AMF) double counting.

        E_dc = U N²/(2n_orb) − J N²/(4n_orb)
        """
        return self.U * N**2 / (2 * self.n_orb) - self.J * N**2 / (4 * self.n_orb)

    def weiss_field(self, G_local: NDArray, self_energy: NDArray) -> NDArray:
        """Weiss mean field (bath Green's function):
        G₀⁻¹ = G_local⁻¹ + Σ
        """
        n_freq = len(G_local)
        G0_inv = np.zeros(n_freq, dtype=complex)
        for iw in range(n_freq):
            g = G_local[iw]
            if abs(g) < 1e-30:
                G0_inv[iw] = 0.0
            else:
                G0_inv[iw] = 1.0 / g + self_energy[iw]
        return G0_inv

    def dmft_self_consistency(self, H_kohn_sham: NDArray,
                                 matsubara_freqs: NDArray,
                                 max_iter: int = 30,
                                 tol: float = 1e-4) -> Dict[str, NDArray]:
        """DMFT self-consistency loop (simplified).

        1. Compute local Green's function: G_loc(iω) = Σ_k [iω + μ − H_k − Σ(iω)]⁻¹
        2. Extract Weiss field: G₀⁻¹ = G_loc⁻¹ + Σ
        3. Solve impurity problem → new Σ(iω)
        4. Iterate until convergence.
        """
        n_freq = len(matsubara_freqs)
        n_orb = self.n_orb
        sigma = np.zeros(n_freq, dtype=complex)
        mu = 0.0

        for iteration in range(max_iter):
            # Local Green's function (simplified: single k-point)
            G_loc = np.zeros(n_freq, dtype=complex)
            for iw in range(n_freq):
                omega = matsubara_freqs[iw]
                G_k = 1.0 / (1j * omega + mu - H_kohn_sham[0, 0] - sigma[iw])
                G_loc[iw] = G_k

            G0_inv = self.weiss_field(G_loc, sigma)

            # Impurity solver (Hubbard-I approximation)
            sigma_new = np.zeros(n_freq, dtype=complex)
            N_avg = 0.5 * n_orb  # half-filling estimate
            E_dc = self.double_counting_fll(N_avg)

            for iw in range(n_freq):
                omega = matsubara_freqs[iw]
                G0 = 1.0 / (G0_inv[iw] + 1e-30)
                # Hubbard-I: Σ ≈ U⟨n⟩ − E_dc
                sigma_new[iw] = self.U * N_avg / (2 * n_orb) - E_dc

            delta = float(np.max(np.abs(sigma_new - sigma)))
            sigma = 0.7 * sigma + 0.3 * sigma_new

            if delta < tol:
                break

        return {
            'self_energy': sigma,
            'G_local': G_loc,
            'iterations': np.array([iteration + 1]),
        }


# ---------------------------------------------------------------------------
#  Projection-Based Embedding
# ---------------------------------------------------------------------------

class ProjectionEmbedding:
    r"""
    Projection-based WFT-in-DFT embedding (Manby, 2012).

    Exact subsystem DFT embedding:
    $$E[\rho] = E^{\text{WFT}}[\rho_A] + E^{\text{DFT}}[\rho_B]
      + E^{\text{int}}[\rho_A, \rho_B]$$

    Embedding potential via orthogonality projection (μ-shift):
    $$V_{\text{emb}} = V_{\text{nuc}}^B + J[\rho_B] + V_{xc}[\rho] - V_{xc}[\rho_A]
      + \mu\hat{P}_B$$

    where $\hat{P}_B$ projects onto environment orbitals (level shift).
    """

    def __init__(self, n_basis_A: int = 5, n_basis_B: int = 10) -> None:
        self.nA = n_basis_A
        self.nB = n_basis_B
        self.n_total = n_basis_A + n_basis_B

    def partition_density(self, P_total: NDArray,
                             C_A: NDArray, n_occ_A: int) -> Tuple[NDArray, NDArray]:
        """Partition total density into subsystem A and B.

        P_A = 2 C_A C_A†, P_B = P_total − P_A.
        """
        P_A = 2 * C_A[:, :n_occ_A] @ C_A[:, :n_occ_A].T
        P_B = P_total - P_A
        return P_A, P_B

    def embedding_potential(self, F_full: NDArray, F_A: NDArray,
                               P_B: NDArray, mu: float = 1e6) -> NDArray:
        """Build embedding potential for subsystem A.

        V_emb = F_full − F_A + μ P_B (level-shift projection)
        """
        V_emb = F_full - F_A + mu * P_B
        return V_emb

    def solve_embedded(self, H_A: NDArray, V_emb: NDArray,
                          n_occ_A: int) -> Tuple[NDArray, NDArray]:
        """Solve embedded Schrödinger equation for subsystem A.

        (H_A + V_emb) C = C ε
        """
        H_eff = H_A + V_emb[:self.nA, :self.nA]
        evals, evecs = np.linalg.eigh(H_eff)
        return evals[:n_occ_A], evecs[:, :n_occ_A]

    def concentric_localisation(self, C: NDArray, S: NDArray,
                                    n_A: int) -> NDArray:
        """Localise MOs by Mulliken population on subsystem A.

        Weight_i = Σ_{μ∈A} (C†SC)_{μi}
        """
        PS = C.T @ S
        weights = np.sum(PS[:, :n_A]**2, axis=1)
        return weights
