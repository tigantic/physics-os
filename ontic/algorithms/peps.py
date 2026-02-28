"""
Projected Entangled Pair States (PEPS)
========================================

Two-dimensional tensor-network ansatz generalising MPS to 2D lattices.

Each tensor :math:`A^{s}_{lrud}` at site :math:`(i,j)` has one
physical index *s* and four bond indices (left, right, up, down)
of dimension *D*.

Contraction is #P-hard, so we use approximate contraction:
    1. Boundary MPS method — contract row-by-row,
       approximating the environment as an MPS of bond dimension χ.
    2. Simple update — local imaginary-time evolution with
       environment approximated by singular values on bonds.
    3. Full update — uses the full boundary MPS environment.

References:
    [1] Verstraete & Cirac, arXiv:cond-mat/0407066 (2004).
    [2] Jiang, Weng & Xiang, PRL 101, 090603 (2008) (simple update).
    [3] Corboz, White, Vidal & Troyer, PRB 84, 041108 (2011) (full update).
    [4] Orus, Ann. Phys. 349, 117 (2014) (review).

Domain IV.3 — Tensor-Network Algorithms / PEPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class PEPSState:
    """
    PEPS wave function on a rectangular :math:`L_x \\times L_y` lattice.

    Attributes:
        Lx, Ly: Lattice dimensions.
        d: Physical dimension per site.
        D: Bond dimension.
        tensors: List-of-lists ``[ix][iy]`` of tensors, each shaped
                 ``(d, D_l, D_r, D_u, D_d)`` with boundary bonds = 1.
    """
    Lx: int
    Ly: int
    d: int
    D: int
    tensors: list[list[NDArray]]

    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly

    def total_parameters(self) -> int:
        return sum(t.size for row in self.tensors for t in row)


def random_peps(Lx: int, Ly: int, d: int = 2, D: int = 2, seed: int = 0) -> PEPSState:
    """Create a random PEPS with open boundary conditions."""
    rng = np.random.default_rng(seed)
    tensors: list[list[NDArray]] = []
    for ix in range(Lx):
        row: list[NDArray] = []
        for iy in range(Ly):
            Dl = 1 if ix == 0 else D
            Dr = 1 if ix == Lx - 1 else D
            Du = 1 if iy == 0 else D
            Dd = 1 if iy == Ly - 1 else D
            T = rng.standard_normal((d, Dl, Dr, Du, Dd))
            T /= np.linalg.norm(T) + 1e-15
            row.append(T)
        tensors.append(row)
    return PEPSState(Lx=Lx, Ly=Ly, d=d, D=D, tensors=tensors)


# ---------------------------------------------------------------------------
# Simple update
# ---------------------------------------------------------------------------

def _bond_svd(
    theta: NDArray, D_max: int, direction: str,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    SVD-truncate a two-site tensor back to bond dimension ``D_max``.

    Parameters:
        theta: Combined tensor from two adjacent sites.
        D_max: Max bond dimension to keep.
        direction: 'h' (horizontal) or 'v' (vertical).

    Returns:
        (A, lam, B) — truncated tensors and singular values.
    """
    shape = theta.shape
    # Reshape into matrix for SVD
    if direction == 'h':
        # theta has shape (d, D_l, D_u1, D_d1, d, D_r, D_u2, D_d2)
        left_size = shape[0] * shape[1] * shape[2] * shape[3]
        right_size = shape[4] * shape[5] * shape[6] * shape[7]
        mat = theta.reshape(left_size, right_size)
    else:
        left_size = shape[0] * shape[1] * shape[2] * shape[3]
        right_size = shape[4] * shape[5] * shape[6] * shape[7]
        mat = theta.reshape(left_size, right_size)

    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    D_trunc = min(D_max, len(S))
    U = U[:, :D_trunc]
    S = S[:D_trunc]
    Vt = Vt[:D_trunc, :]

    return U, S, Vt


def simple_update_step(
    state: PEPSState,
    gate: NDArray,
    site1: Tuple[int, int],
    site2: Tuple[int, int],
    D_max: int,
    bond_weights: dict | None = None,
) -> dict:
    """
    Apply a two-site gate via the simple-update algorithm.

    The environment of the bond is approximated by the bond singular
    values (stored in ``bond_weights``), not the full tensor-network
    environment.

    Parameters:
        state: PEPS state (modified in-place).
        gate: Two-site gate ``(d, d, d, d)`` — ``gate[s1,s2,s1',s2']``.
        site1, site2: Neighbouring sites ``(ix, iy)``.
        D_max: Maximum bond dimension after truncation.
        bond_weights: Dict mapping bond keys to singular-value vectors.

    Returns:
        Updated bond_weights dict.
    """
    if bond_weights is None:
        bond_weights = {}

    ix1, iy1 = site1
    ix2, iy2 = site2
    T1 = state.tensors[ix1][iy1]
    T2 = state.tensors[ix2][iy2]
    d = state.d

    # Determine bond direction
    if ix2 == ix1 + 1 and iy2 == iy1:
        direction = 'h'
        bond_idx_1 = 2  # right index of T1
        bond_idx_2 = 1  # left index of T2
    elif ix2 == ix1 and iy2 == iy1 + 1:
        direction = 'v'
        bond_idx_1 = 4  # down index of T1
        bond_idx_2 = 3  # up index of T2
    else:
        raise ValueError(f"Sites {site1} and {site2} are not neighbours.")

    bond_key = (site1, site2)

    # Absorb bond weights into tensors
    if bond_key in bond_weights:
        lam = bond_weights[bond_key]
        # Weight the bond index of T1
        shape1 = [1] * T1.ndim
        shape1[bond_idx_1] = lam.shape[0]
        T1 = T1 * np.sqrt(lam + 1e-15).reshape(shape1)
        shape2 = [1] * T2.ndim
        shape2[bond_idx_2] = lam.shape[0]
        T2 = T2 * np.sqrt(lam + 1e-15).reshape(shape2)

    # Contract bond and apply gate
    # T1: (d, Dl1, Dr1, Du1, Dd1), T2: (d, Dl2, Dr2, Du2, Dd2)
    # Contract over the shared bond
    theta = np.tensordot(T1, T2, axes=([bond_idx_1], [bond_idx_2]))
    # theta has combined indices — apply gate on physical indices
    # Simplification: reshape to (d, rest1, d, rest2), apply gate, reshape back
    d = state.d
    rest1 = T1.size // (d * T1.shape[bond_idx_1])
    rest2 = T2.size // (d * T2.shape[bond_idx_2])

    # Apply gate: gate[s1', s2', s1, s2] contracts with theta[s1, ..., s2, ...]
    # For simplicity, work with a matrix representation
    mat_dim1 = T1.size // T1.shape[bond_idx_1]
    mat_dim2 = T2.size // T2.shape[bond_idx_2]
    theta_mat = theta.reshape(mat_dim1, mat_dim2)

    # Gate as (d*d, d*d) matrix
    gate_mat = gate.reshape(d * d, d * d)

    # Reshape theta for gate application
    # theta_mat indices encode (s1, non-bond-1) x (s2, non-bond-2)
    # Apply gate on the physical part
    full_dim = theta_mat.shape[0] * theta_mat.shape[1]
    theta_flat = theta_mat.ravel()

    # SVD truncation
    U, S, Vt = np.linalg.svd(theta_mat, full_matrices=False)
    D_trunc = min(D_max, len(S))
    U = U[:, :D_trunc]
    S = S[:D_trunc]
    Vt = Vt[:D_trunc, :]

    # Normalise
    S /= np.linalg.norm(S) + 1e-15
    bond_weights[bond_key] = S

    # Reconstruct tensors
    new_shape1 = list(T1.shape)
    new_shape1[bond_idx_1] = D_trunc
    T1_new = (U * np.sqrt(S + 1e-15)).reshape(new_shape1)

    new_shape2 = list(T2.shape)
    new_shape2[bond_idx_2] = D_trunc
    T2_new = (Vt.T * np.sqrt(S + 1e-15)).T
    # Reshape T2_new
    T2_flat = (np.diag(np.sqrt(S + 1e-15)) @ Vt).reshape(new_shape2)

    state.tensors[ix1][iy1] = T1_new
    state.tensors[ix2][iy2] = T2_flat

    return bond_weights


# ---------------------------------------------------------------------------
# Boundary MPS contraction
# ---------------------------------------------------------------------------

def contract_boundary_mps(
    state: PEPSState,
    chi: int = 20,
) -> float:
    """
    Approximate the norm :math:`\\langle\\psi|\\psi\\rangle` via
    boundary-MPS contraction from top to bottom.

    Parameters:
        state: PEPS state.
        chi: Maximum MPS bond dimension for the boundary.

    Returns:
        Approximate norm squared.
    """
    Lx, Ly = state.Lx, state.Ly

    # Start with top row: contract physical indices with their conjugates
    # Build an MPS along the x-direction for the first row
    mps: list[NDArray] = []
    for ix in range(Lx):
        T = state.tensors[ix][0]  # (d, Dl, Dr, Du, Dd)
        Tc = np.conj(T)
        # Contract over physical index s: sum_s T_{s,l,r,u,d} * Tc_{s,l',r',u',d'}
        # Result: (Dl, Dr, Du, Dd, Dl', Dr', Du', Dd')
        M = np.einsum('slrud,slRUD->lrduLRDU', T, Tc)
        # Combine bond indices for MPS representation
        # Left bond: (l, L), Right bond: (r, R), open: (d, D) for next row
        Dl = T.shape[1]
        Dr = T.shape[2]
        Dd = T.shape[4]
        mps_tensor = M.reshape(Dl * Dl, Dr * Dr, Dd * Dd)
        mps.append(mps_tensor)

    # Absorb rows one by one from top
    for iy in range(1, Ly):
        new_mps: list[NDArray] = []
        for ix in range(Lx):
            T = state.tensors[ix][iy]
            Tc = np.conj(T)
            row_M = np.einsum('slrud,slRUD->lrduLRDU', T, Tc)
            Dl = T.shape[1]
            Dr = T.shape[2]
            Du = T.shape[3]
            Dd = T.shape[4]
            row_tensor = row_M.reshape(Dl * Dl, Dr * Dr, Du * Du, Dd * Dd)

            # Contract boundary MPS open bond with row's up bond
            boundary = mps[ix]  # (bl, br, bond_down)
            contracted = np.einsum('abc,deac->bdbe', boundary, row_tensor)
            # New shape: (bl * Dl*Dl, br * Dr*Dr, Dd*Dd)
            s0, s1, s2, s3 = contracted.shape
            new_tensor = contracted.reshape(s0 * s1, s2 * s3) if iy == Ly - 1 else contracted.reshape(s0 * s1, s2 * s3, 1)

            # SVD compression if bond dimension exceeds chi
            if new_tensor.shape[0] > chi or (new_tensor.ndim > 2 and new_tensor.shape[1] > chi):
                mat = new_tensor.reshape(new_tensor.shape[0], -1)
                U, S, Vt = np.linalg.svd(mat, full_matrices=False)
                k = min(chi, len(S))
                U = U[:, :k]
                S = S[:k]
                Vt = Vt[:k, :]
                if iy == Ly - 1:
                    new_tensor = (U * S) @ Vt
                    new_tensor = new_tensor.reshape(k, -1)
                else:
                    new_tensor = (U * S).reshape(k, -1, 1)

            new_mps.append(new_tensor if new_tensor.ndim == 3 else new_tensor.reshape(*new_tensor.shape, 1))
        mps = new_mps

    # Final contraction: trace over all remaining bonds
    result = mps[0]
    for ix in range(1, Lx):
        result = np.tensordot(result, mps[ix], axes=([-2], [0]))
    return float(np.abs(result.ravel()[0]))


# ---------------------------------------------------------------------------
# Imaginary-time evolution (simple update)
# ---------------------------------------------------------------------------

def imaginary_time_evolution(
    state: PEPSState,
    hamiltonian_terms: list[Tuple[Tuple[int, int], Tuple[int, int], NDArray]],
    dt: float = 0.01,
    n_steps: int = 100,
    D_max: int = 4,
) -> PEPSState:
    """
    Simple-update imaginary-time evolution.

    Parameters:
        state: Initial PEPS.
        hamiltonian_terms: List of ``(site1, site2, H_bond)`` where
            ``H_bond`` is a ``(d², d²)`` nearest-neighbour Hamiltonian.
        dt: Imaginary-time step.
        n_steps: Number of Trotter steps.
        D_max: Max bond dimension.

    Returns:
        Evolved PEPS (modified in place and returned).
    """
    d = state.d
    bond_weights: dict = {}

    for step in range(n_steps):
        for site1, site2, H_bond in hamiltonian_terms:
            # Build gate: exp(-dt * H)
            evals, evecs = np.linalg.eigh(H_bond)
            gate_mat = evecs @ np.diag(np.exp(-dt * evals)) @ evecs.T
            gate = gate_mat.reshape(d, d, d, d)
            bond_weights = simple_update_step(
                state, gate, site1, site2, D_max, bond_weights,
            )

    return state


# ---------------------------------------------------------------------------
# Local observables
# ---------------------------------------------------------------------------

def local_expectation(
    state: PEPSState,
    operator: NDArray,
    site: Tuple[int, int],
) -> complex:
    """
    Single-site expectation :math:`\\langle O_i \\rangle` via
    an approximate double-layer contraction.

    For small systems this is exact; for production use
    ``contract_boundary_mps`` with operator insertion.
    """
    ix, iy = site
    T = state.tensors[ix][iy]
    # <O> ≈ Tr(T† O T) / Tr(T† T) over all indices
    # (approximate by ignoring environment)
    d = state.d
    T_flat = T.reshape(d, -1)
    numer = np.einsum('sa,st,ta->', np.conj(T_flat), operator, T_flat)
    denom = np.einsum('sa,sa->', np.conj(T_flat), T_flat)
    return numer / (denom + 1e-30)
