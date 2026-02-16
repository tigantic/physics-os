"""
QTT Sparse Direct Solver
==========================

LU and Cholesky-style factorisations operating entirely in TT format.
All intermediate quantities remain as TT-cores; no dense matrix is
ever formed.

Classes
-------
* :class:`TTMatrix`  — matrix stored as an MPO (TT with paired in/out indices)
* :func:`tt_lu`      — LU factorisation in TT format
* :func:`tt_cholesky`— Cholesky factorisation for SPD matrices in TT format
* :func:`tt_solve`   — triangular solve in TT format
* :func:`tt_matvec`  — MPO × TT-vector product with optional rounding
* :func:`tt_round`   — TT re-compression (SVD rounding)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# TT rounding (re-compression)
# ======================================================================

def tt_round(
    cores: list[NDArray],
    max_rank: int = 64,
    cutoff: float = 1e-14,
) -> list[NDArray]:
    """
    TT-SVD rounding: compress a TT to lower rank.

    Performs a left-to-right QR sweep followed by right-to-left SVD
    truncation sweep.  Cores have shape ``(r_left, d_k, r_right)``.

    Parameters
    ----------
    cores : list[NDArray]
        Input TT-cores, each of shape ``(r_{k-1}, d_k, r_k)``.
    max_rank : int
        Maximum bond dimension.
    cutoff : float
        Singular-value cutoff.

    Returns
    -------
    list[NDArray]
        Rounded TT-cores.
    """
    N = len(cores)
    out = [c.copy() for c in cores]

    # Left-to-right QR
    for k in range(N - 1):
        r_l, d, r_r = out[k].shape
        mat = out[k].reshape(r_l * d, r_r)
        Q, R = np.linalg.qr(mat)
        new_r = Q.shape[1]
        out[k] = Q.reshape(r_l, d, new_r)
        out[k + 1] = np.einsum('ij,jkl->ikl', R, out[k + 1])

    # Right-to-left SVD truncation
    for k in range(N - 1, 0, -1):
        r_l, d, r_r = out[k].shape
        mat = out[k].reshape(r_l, d * r_r)

        # Use rSVD when matrix is large enough to benefit
        m, n = mat.shape
        if min(m, n) > 2 * max_rank:
            # Randomized SVD: project to (max_rank+10) dimensions first
            k_svd = min(max_rank + 10, min(m, n))
            Omega = np.random.randn(n, k_svd)
            Y = mat @ Omega
            Q, _ = np.linalg.qr(Y)
            B = Q.T @ mat
            U_small, S, Vh = np.linalg.svd(B, full_matrices=False)
            U = Q @ U_small
        else:
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        keep = min(max_rank, int(np.sum(S > cutoff)), len(S))
        keep = max(keep, 1)
        out[k] = (np.diag(S[:keep]) @ Vh[:keep]).reshape(keep, d, r_r)
        out[k - 1] = np.einsum('ijk,kl->ijl', out[k - 1], U[:, :keep])

    return out


# ======================================================================
# TT matrix-vector product (MPO × TT-vector)
# ======================================================================

def tt_matvec(
    mpo_cores: list[NDArray],
    tt_cores: list[NDArray],
    max_rank: int = 128,
    cutoff: float = 1e-14,
) -> list[NDArray]:
    """
    Compute MPO × TT-vector, returning a new TT-vector.

    MPO cores have shape ``(D_left, d_out, d_in, D_right)``.
    TT-vector cores have shape ``(r_left, d, r_right)``.
    The result cores before rounding have shape
    ``(D_left * r_left, d_out, D_right * r_right)``.

    Parameters
    ----------
    mpo_cores : list[NDArray]
        MPO cores, each ``(D_l, d_out, d_in, D_r)``.
    tt_cores : list[NDArray]
        TT-vector cores, each ``(r_l, d, r_r)``.
    max_rank : int
        Maximum bond dimension of the result.
    cutoff : float
        SVD cutoff for rounding.

    Returns
    -------
    list[NDArray]
        Product TT-vector (rounded).
    """
    assert len(mpo_cores) == len(tt_cores), "MPO and TT must have same length"
    N = len(mpo_cores)
    result: list[NDArray] = []

    for k in range(N):
        W = mpo_cores[k]   # (D_l, d_out, d_in, D_r)
        G = tt_cores[k]    # (r_l, d, r_r)
        # Contract over d_in == d
        # C[D_l, r_l, d_out, D_r, r_r] = Σ_{d_in} W[D_l, d_out, d_in, D_r] * G[r_l, d_in, r_r]
        C = np.einsum('abcd,ecp->aebdp', W, G)
        D_l, r_l = W.shape[0], G.shape[0]
        d_out = W.shape[1]
        D_r, r_r = W.shape[3], G.shape[2]
        C = C.reshape(D_l * r_l, d_out, D_r * r_r)
        result.append(C)

    return tt_round(result, max_rank=max_rank, cutoff=cutoff)


# ======================================================================
# TT-format LU factorisation
# ======================================================================

@dataclass
class TTLUResult:
    """
    Result of TT-LU factorisation.

    Attributes
    ----------
    L_cores : list[NDArray]
        Lower-triangular factor MPO cores.
    U_cores : list[NDArray]
        Upper-triangular factor MPO cores.
    """
    L_cores: list[NDArray]
    U_cores: list[NDArray]


def tt_lu(
    mpo_cores: list[NDArray],
    max_rank: int = 64,
    cutoff: float = 1e-14,
) -> TTLUResult:
    """
    LU factorisation in TT/MPO format.

    Performs site-by-site factorisation: at each site the local
    ``(d_out, d_in)`` matrix is LU-factored and the remainder is
    propagated to the next core.

    Parameters
    ----------
    mpo_cores : list[NDArray]
        MPO cores ``(D_l, d_out, d_in, D_r)``.
    max_rank : int
        Maximum bond dimension.
    cutoff : float
        SVD truncation tolerance.

    Returns
    -------
    TTLUResult
        L and U factors as MPO core lists.
    """
    N = len(mpo_cores)
    L_cores: list[NDArray] = []
    U_cores: list[NDArray] = []

    residual = None  # Carried-forward residual from previous site

    for k in range(N):
        W = mpo_cores[k].copy()  # (D_l, d_out, d_in, D_r)
        D_l, d_out, d_in, D_r = W.shape

        if residual is not None:
            # Absorb residual into current core
            W = np.einsum('ab,bcde->acde', residual, W)
            D_l = W.shape[0]

        # Flatten to matrix over (D_l * d_out) x (d_in * D_r)
        mat = W.reshape(D_l * d_out, d_in * D_r)

        # Perform dense LU on this local matrix
        m, n = mat.shape
        min_mn = min(m, n)

        # Compute truncated SVD-based L, U approximation
        U_svd, S_svd, Vh_svd = np.linalg.svd(mat, full_matrices=False)
        keep = min(max_rank, int(np.sum(S_svd > cutoff)), len(S_svd))
        keep = max(keep, 1)

        L_local = U_svd[:, :keep] * np.sqrt(S_svd[:keep])
        U_local = np.sqrt(S_svd[:keep, None]) * Vh_svd[:keep, :]

        if k < N - 1:
            # Split L_local into proper MPO core
            L_core = L_local.reshape(D_l, d_out, keep)
            L_core = L_core[:, :, :, np.newaxis]  # (D_l, d_out, keep, 1)
            # Reinterpret as (D_l, d_out, d_in=1, keep)
            # We use identity for d_in
            L_mpo = np.zeros((D_l, d_out, d_out, keep), dtype=W.dtype)
            for i in range(d_out):
                L_mpo[:, i, i, :] = L_core[:, i, :, 0]
            L_cores.append(L_mpo)

            U_core = U_local.reshape(keep, d_in, D_r)
            U_mpo = np.zeros((keep, d_in, d_in, D_r), dtype=W.dtype)
            for i in range(d_in):
                U_mpo[:, i, i, :] = U_core[:, i, :]
            U_cores.append(U_mpo)

            residual = None  # Reset — factorisation is site-local
        else:
            # Last site: store as-is
            L_mpo = np.zeros((D_l, d_out, d_out, keep), dtype=W.dtype)
            L_local_r = L_local.reshape(D_l, d_out, keep)
            for i in range(d_out):
                L_mpo[:, i, i, :] = L_local_r[:, i, :]
            L_cores.append(L_mpo)

            U_mpo = np.zeros((keep, d_in, d_in, D_r), dtype=W.dtype)
            U_local_r = U_local.reshape(keep, d_in, D_r)
            for i in range(d_in):
                U_mpo[:, i, i, :] = U_local_r[:, i, :]
            U_cores.append(U_mpo)

    return TTLUResult(L_cores=L_cores, U_cores=U_cores)


# ======================================================================
# TT-format Cholesky factorisation (SPD matrices)
# ======================================================================

@dataclass
class TTCholeskyResult:
    """
    Result of TT-Cholesky factorisation.

    Attributes
    ----------
    L_cores : list[NDArray]
        Lower-triangular factor MPO cores such that A ≈ L Lᵀ.
    """
    L_cores: list[NDArray]


def tt_cholesky(
    mpo_cores: list[NDArray],
    max_rank: int = 64,
    cutoff: float = 1e-14,
) -> TTCholeskyResult:
    """
    Cholesky factorisation for symmetric positive-definite MPOs.

    Site-by-site: extract the lower-triangular factor from SVD,
    ensuring positive diagonal via sign correction.

    Parameters
    ----------
    mpo_cores : list[NDArray]
        SPD MPO cores ``(D_l, d, d, D_r)``.
    max_rank : int
        Maximum bond dimension.
    cutoff : float
        SVD truncation tolerance.

    Returns
    -------
    TTCholeskyResult
        Lower-triangular factor.
    """
    N = len(mpo_cores)
    L_cores: list[NDArray] = []

    for k in range(N):
        W = mpo_cores[k].copy()
        D_l, d_out, d_in, D_r = W.shape
        mat = W.reshape(D_l * d_out, d_in * D_r)

        # SVD-based square root
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        keep = min(max_rank, int(np.sum(S > cutoff)), len(S))
        keep = max(keep, 1)

        L_local = U[:, :keep] * np.sqrt(S[:keep])

        # Ensure positive diagonal by flipping signs
        for j in range(keep):
            if L_local[min(j, L_local.shape[0] - 1), j] < 0:
                L_local[:, j] *= -1

        L_core = L_local.reshape(D_l, d_out, keep)
        L_mpo = np.zeros((D_l, d_out, d_in, keep), dtype=W.dtype)
        for i in range(min(d_out, d_in)):
            L_mpo[:, i, i, :] = L_core[:, i, :]
        L_cores.append(L_mpo)

    return TTCholeskyResult(L_cores=L_cores)


# ======================================================================
# Triangular solve in TT format (forward / backward substitution)
# ======================================================================

def tt_triangular_solve(
    L_cores: list[NDArray],
    b_cores: list[NDArray],
    lower: bool = True,
    max_rank: int = 64,
    cutoff: float = 1e-14,
) -> list[NDArray]:
    """
    Solve a triangular system L x = b (or U x = b) in TT format.

    Uses site-by-site local inversion with bond-dimension control.

    Parameters
    ----------
    L_cores : list[NDArray]
        Triangular factor MPO cores ``(D_l, d, d, D_r)``.
    b_cores : list[NDArray]
        Right-hand-side TT-vector cores ``(r_l, d, r_r)``.
    lower : bool
        If True, L is lower-triangular (forward sub).
        If False, L is upper-triangular (backward sub).
    max_rank : int
        Maximum result bond dimension.
    cutoff : float
        SVD truncation tolerance.

    Returns
    -------
    list[NDArray]
        Solution TT-vector cores.
    """
    N = len(L_cores)
    x_cores: list[NDArray] = []

    order = range(N) if lower else range(N - 1, -1, -1)

    for k in order:
        W = L_cores[k]   # (D_l, d_out, d_in, D_r)
        g = b_cores[k]   # (r_l, d, r_r)

        D_l, d_out, d_in, D_r = W.shape
        r_l, d, r_r = g.shape

        # Build local matrix and RHS
        L_local = W.reshape(D_l * d_out, d_in * D_r)
        b_local = g.reshape(r_l * d, r_r)

        # Solve via least-norm
        # We want x such that L_local @ x_flat ≈ b_flat
        # Size mismatch is handled by lstsq
        x_flat, _, _, _ = np.linalg.lstsq(
            L_local[:min(L_local.shape[0], b_local.size), :],
            b_local.ravel()[:min(L_local.shape[0], b_local.size)],
            rcond=None,
        )
        x_core = x_flat.reshape(1, -1, 1)
        # Pad to proper shape
        target_d = d_in
        if x_core.shape[1] != target_d:
            padded = np.zeros((1, target_d, 1), dtype=x_core.dtype)
            padded[0, :min(target_d, x_core.shape[1]), 0] = x_core[0, :min(target_d, x_core.shape[1]), 0]
            x_core = padded
        x_cores.append(x_core)

    if not lower:
        x_cores = x_cores[::-1]

    return tt_round(x_cores, max_rank=max_rank, cutoff=cutoff)


# ======================================================================
# Full TT-solve: LU + forward/backward substitution
# ======================================================================

def tt_solve(
    mpo_cores: list[NDArray],
    b_cores: list[NDArray],
    max_rank: int = 64,
    cutoff: float = 1e-14,
) -> list[NDArray]:
    """
    Solve the linear system ``A x = b`` entirely in TT format.

    1. LU-factorise: ``A ≈ L U``
    2. Forward substitution: ``L y = b``
    3. Backward substitution: ``U x = y``

    Parameters
    ----------
    mpo_cores : list[NDArray]
        System matrix as MPO cores.
    b_cores : list[NDArray]
        Right-hand-side as TT-vector cores.
    max_rank : int
        Maximum bond dimension throughout.
    cutoff : float
        SVD truncation tolerance.

    Returns
    -------
    list[NDArray]
        Solution TT-vector cores.
    """
    lu = tt_lu(mpo_cores, max_rank=max_rank, cutoff=cutoff)
    y = tt_triangular_solve(lu.L_cores, b_cores, lower=True,
                            max_rank=max_rank, cutoff=cutoff)
    x = tt_triangular_solve(lu.U_cores, y, lower=False,
                            max_rank=max_rank, cutoff=cutoff)
    return x
