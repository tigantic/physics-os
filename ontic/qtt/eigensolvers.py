"""
QTT Eigensolvers
==================

Native Lanczos and Davidson eigensolvers operating entirely in TT format.
All intermediate vectors are stored and manipulated as TT-cores; no dense
matrix is ever formed.

Key classes / functions
-----------------------
* :func:`tt_lanczos`     — Lanczos eigensolver in TT format
* :func:`tt_davidson`    — Davidson eigensolver with TT preconditioning
* :func:`tt_inner`       — inner product of two TT vectors
* :func:`tt_axpy`        — TT vector addition  y ← α x + y
* :func:`tt_norm`        — Frobenius norm of a TT vector
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ontic.qtt.sparse_direct import tt_matvec, tt_round


# ======================================================================
# TT-vector algebra
# ======================================================================

def tt_inner(a: list[NDArray], b: list[NDArray]) -> float:
    """
    Inner product :math:`\\langle a, b \\rangle` of two TT-vectors.

    Contracts from left to right via transfer matrices.
    Cost: :math:`O(N r^3 d)`.
    """
    N = len(a)
    assert len(b) == N
    # Initialise transfer matrix (1, 1)
    T = np.ones((1, 1), dtype=a[0].dtype)
    for k in range(N):
        # a[k]: (ra_l, d, ra_r),  b[k]: (rb_l, d, rb_r)
        # T_new[ra_r, rb_r] = sum_{ra_l, rb_l, d} T[ra_l, rb_l] a[k][ra_l, d, ra_r] b[k][rb_l, d, rb_r]
        T = np.einsum('ij,idk,jdl->kl', T, a[k], b[k])
    return float(T.item())


def tt_norm(cores: list[NDArray]) -> float:
    """Frobenius norm of a TT-vector."""
    return np.sqrt(max(tt_inner(cores, cores), 0.0))


def tt_scale(cores: list[NDArray], alpha: float) -> list[NDArray]:
    """Scale a TT-vector by a scalar."""
    result = [c.copy() for c in cores]
    result[0] = result[0] * alpha
    return result


def tt_add(a: list[NDArray], b: list[NDArray]) -> list[NDArray]:
    """
    Add two TT-vectors.  Result bond dimension = ra + rb.
    """
    N = len(a)
    assert len(b) == N
    cores: list[NDArray] = []
    for k in range(N):
        ra_l, d, ra_r = a[k].shape
        rb_l, db, rb_r = b[k].shape
        assert d == db

        if k == 0:
            # First core: concatenate along right bond
            C = np.zeros((1, d, ra_r + rb_r), dtype=a[k].dtype)
            C[:, :, :ra_r] = a[k]
            C[:, :, ra_r:] = b[k]
        elif k == N - 1:
            # Last core: concatenate along left bond
            C = np.zeros((ra_l + rb_l, d, 1), dtype=a[k].dtype)
            C[:ra_l, :, :] = a[k]
            C[ra_l:, :, :] = b[k]
        else:
            # Middle: block diagonal
            C = np.zeros((ra_l + rb_l, d, ra_r + rb_r), dtype=a[k].dtype)
            C[:ra_l, :, :ra_r] = a[k]
            C[ra_l:, :, ra_r:] = b[k]
        cores.append(C)
    return cores


def tt_axpy(
    alpha: float,
    x: list[NDArray],
    y: list[NDArray],
    max_rank: int = 128,
    cutoff: float = 1e-14,
) -> list[NDArray]:
    """Compute ``y + alpha * x`` with rounding."""
    sx = tt_scale(x, alpha)
    return tt_round(tt_add(sx, y), max_rank=max_rank, cutoff=cutoff)


# ======================================================================
# TT-Lanczos eigensolver
# ======================================================================

@dataclass
class TTEigResult:
    """
    Result of a TT eigensolver.

    Attributes
    ----------
    eigenvalue : float
        Lowest eigenvalue found.
    eigenvector : list[NDArray]
        Corresponding eigenvector as TT-cores.
    eigenvalues : NDArray
        All converged eigenvalues.
    residuals : list[float]
        Residual norms per iteration.
    converged : bool
    n_iter : int
    """
    eigenvalue: float
    eigenvector: list[NDArray]
    eigenvalues: NDArray
    residuals: list[float]
    converged: bool
    n_iter: int


def tt_lanczos(
    mpo_cores: list[NDArray],
    v0: Optional[list[NDArray]] = None,
    n_bits: Optional[int] = None,
    d: int = 2,
    max_iter: int = 50,
    tol: float = 1e-8,
    max_rank: int = 64,
    n_eigenvalues: int = 1,
    seed: Optional[int] = None,
) -> TTEigResult:
    """
    Lanczos eigensolver in TT format.

    Builds a Krylov subspace :math:`\\{v, Av, A^2v, \\ldots\\}` where all
    vectors are stored and orthogonalised in TT format.

    Parameters
    ----------
    mpo_cores : list[NDArray]
        Hamiltonian / operator as MPO cores ``(D_l, d, d, D_r)``.
    v0 : list[NDArray], optional
        Initial TT-vector.  Random if not provided.
    n_bits : int, optional
        Number of TT-cores (required if ``v0`` is None).
    d : int
        Local physical dimension (default 2).
    max_iter : int
        Maximum Lanczos iterations.
    tol : float
        Convergence tolerance on eigenvalue.
    max_rank : int
        Maximum TT bond dimension for intermediate vectors.
    n_eigenvalues : int
        Number of eigenvalues to return.
    seed : int, optional
        RNG seed for random initial vector.

    Returns
    -------
    TTEigResult
    """
    rng = np.random.default_rng(seed)

    if v0 is None:
        assert n_bits is not None, "Provide v0 or n_bits"
        v0 = [rng.standard_normal((1, d, 1)) for _ in range(n_bits)]
    else:
        n_bits = len(v0)

    # Normalise v0
    nrm = tt_norm(v0)
    if nrm > 1e-30:
        v0 = tt_scale(v0, 1.0 / nrm)

    # Lanczos vectors
    V: list[list[NDArray]] = [v0]
    alphas: list[float] = []
    betas: list[float] = [0.0]
    residuals: list[float] = []

    v_prev: Optional[list[NDArray]] = None
    v_curr = v0

    for j in range(max_iter):
        # w = A v_curr
        w = tt_matvec(mpo_cores, v_curr, max_rank=max_rank)

        # alpha_j = <v_curr, w>
        alpha_j = tt_inner(v_curr, w)
        alphas.append(alpha_j)

        # w = w - alpha_j * v_curr
        w = tt_axpy(-alpha_j, v_curr, w, max_rank=max_rank)

        # w = w - beta_j * v_prev
        if v_prev is not None:
            w = tt_axpy(-betas[-1], v_prev, w, max_rank=max_rank)

        # beta_{j+1} = ||w||
        beta_next = tt_norm(w)
        betas.append(beta_next)

        # Check convergence via tridiagonal eigenvalues
        k = len(alphas)
        T_mat = np.diag(alphas) + np.diag(betas[1:k], 1) + np.diag(betas[1:k], -1)
        eigs = np.sort(np.linalg.eigvalsh(T_mat))
        residuals.append(beta_next)

        if beta_next < tol:
            return TTEigResult(
                eigenvalue=float(eigs[0]),
                eigenvector=v_curr,
                eigenvalues=eigs[:n_eigenvalues],
                residuals=residuals,
                converged=True,
                n_iter=j + 1,
            )

        # Normalise w → v_{j+1}
        v_prev = v_curr
        v_curr = tt_scale(w, 1.0 / (beta_next + 1e-30))
        v_curr = tt_round(v_curr, max_rank=max_rank)
        V.append(v_curr)

    # Final eigenvalues from tridiagonal matrix
    k = len(alphas)
    T_mat = np.diag(alphas) + np.diag(betas[1:k], 1) + np.diag(betas[1:k], -1)
    eigs = np.sort(np.linalg.eigvalsh(T_mat))

    return TTEigResult(
        eigenvalue=float(eigs[0]),
        eigenvector=V[0],
        eigenvalues=eigs[:n_eigenvalues],
        residuals=residuals,
        converged=False,
        n_iter=max_iter,
    )


# ======================================================================
# TT-Davidson eigensolver
# ======================================================================

def tt_davidson(
    mpo_cores: list[NDArray],
    v0: Optional[list[NDArray]] = None,
    n_bits: Optional[int] = None,
    d: int = 2,
    max_iter: int = 50,
    max_subspace: int = 20,
    tol: float = 1e-8,
    max_rank: int = 64,
    seed: Optional[int] = None,
) -> TTEigResult:
    """
    Davidson eigensolver in TT format.

    Similar to Lanczos but uses a preconditioned residual expansion
    for faster convergence on diagonally-dominant operators.

    Parameters
    ----------
    mpo_cores : list[NDArray]
        Operator MPO cores.
    v0 : list[NDArray], optional
        Initial guess.
    n_bits : int, optional
        Number of TT-cores.
    d : int
        Local dimension.
    max_iter : int
        Maximum iterations.
    max_subspace : int
        Maximum subspace dimension before restart.
    tol : float
        Convergence tolerance.
    max_rank : int
        TT bond dimension cap.
    seed : int, optional
        RNG seed.

    Returns
    -------
    TTEigResult
    """
    rng = np.random.default_rng(seed)

    if v0 is None:
        assert n_bits is not None
        v0 = [rng.standard_normal((1, d, 1)) for _ in range(n_bits)]
    else:
        n_bits = len(v0)

    nrm = tt_norm(v0)
    if nrm > 1e-30:
        v0 = tt_scale(v0, 1.0 / nrm)

    V: list[list[NDArray]] = [v0]
    residuals: list[float] = []
    theta = 0.0

    for it in range(max_iter):
        # Build small Hamiltonian in subspace
        m = len(V)
        H_sub = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i, m):
                Hv = tt_matvec(mpo_cores, V[j], max_rank=max_rank)
                H_sub[i, j] = tt_inner(V[i], Hv)
                H_sub[j, i] = H_sub[i, j]

        # Diagonalise subspace
        eigs, vecs = np.linalg.eigh(H_sub)
        theta = float(eigs[0])
        c = vecs[:, 0]

        # Build Ritz vector
        ritz = tt_scale(V[0], c[0])
        for i in range(1, m):
            ritz = tt_axpy(c[i], V[i], ritz, max_rank=max_rank)

        # Residual r = A*ritz - theta*ritz
        Aritz = tt_matvec(mpo_cores, ritz, max_rank=max_rank)
        residual = tt_axpy(-theta, ritz, Aritz, max_rank=max_rank)
        res_norm = tt_norm(residual)
        residuals.append(res_norm)

        if res_norm < tol:
            return TTEigResult(
                eigenvalue=theta,
                eigenvector=ritz,
                eigenvalues=eigs[:1],
                residuals=residuals,
                converged=True,
                n_iter=it + 1,
            )

        # Preconditioned correction: t = r / (theta - D_ii) approximated
        # as simple scaling (diagonal preconditioner in TT)
        correction = tt_scale(residual, 1.0 / max(abs(theta) + 1.0, 1e-10))
        correction = tt_round(correction, max_rank=max_rank)

        # Orthogonalise against V
        for v in V:
            overlap = tt_inner(v, correction)
            correction = tt_axpy(-overlap, v, correction, max_rank=max_rank)

        cnorm = tt_norm(correction)
        if cnorm > 1e-14:
            correction = tt_scale(correction, 1.0 / cnorm)
            correction = tt_round(correction, max_rank=max_rank)
            V.append(correction)

        # Restart if subspace too large
        if len(V) >= max_subspace:
            V = [ritz]

    return TTEigResult(
        eigenvalue=theta,
        eigenvector=V[0],
        eigenvalues=np.array([theta]),
        residuals=residuals,
        converged=False,
        n_iter=max_iter,
    )
