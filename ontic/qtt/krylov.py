"""
QTT-Krylov Methods
====================

CG and GMRES operating entirely in TT format with rank control.
All vectors live in TT representation; matvec uses MPO-TT contraction.

Key functions
-------------
* :func:`tt_cg`    — Conjugate Gradient in TT format (SPD operators)
* :func:`tt_gmres` — GMRES in TT format (general operators)

Both methods re-compress intermediate TT vectors after every
rank-increasing operation (addition, matvec) to keep rank bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ontic.qtt.sparse_direct import tt_matvec, tt_round
from ontic.qtt.eigensolvers import (
    tt_inner,
    tt_norm,
    tt_scale,
    tt_add,
    tt_axpy,
)


# ======================================================================
# Result container
# ======================================================================

@dataclass
class TTKrylovResult:
    """
    Result of a TT-Krylov solve.

    Attributes
    ----------
    x : list[NDArray]
        Solution TT-vector cores.
    residual_norms : list[float]
        Residual norm per iteration.
    converged : bool
    n_iter : int
    """
    x: list[NDArray]
    residual_norms: list[float]
    converged: bool
    n_iter: int


# ======================================================================
# TT-CG (Conjugate Gradient)
# ======================================================================

def tt_cg(
    mpo_cores: list[NDArray],
    b_cores: list[NDArray],
    x0: Optional[list[NDArray]] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    max_rank: int = 64,
) -> TTKrylovResult:
    """
    Conjugate Gradient in TT format for SPD systems ``A x = b``.

    All vectors and the residual remain in TT format with rank bounded
    by ``max_rank``.

    Parameters
    ----------
    mpo_cores : list[NDArray]
        SPD operator A as MPO cores ``(D_l, d, d, D_r)``.
    b_cores : list[NDArray]
        Right-hand-side b as TT-vector cores ``(r_l, d, r_r)``.
    x0 : list[NDArray], optional
        Initial guess (zero if None).
    max_iter : int
        Maximum CG iterations.
    tol : float
        Residual-norm tolerance.
    max_rank : int
        Maximum TT rank for intermediates.

    Returns
    -------
    TTKrylovResult
    """
    N = len(b_cores)
    d = b_cores[0].shape[1]

    if x0 is None:
        x = [np.zeros((1, d, 1), dtype=b_cores[0].dtype) for _ in range(N)]
    else:
        x = [c.copy() for c in x0]

    # r = b - A x
    Ax = tt_matvec(mpo_cores, x, max_rank=max_rank)
    r = tt_axpy(-1.0, Ax, [c.copy() for c in b_cores], max_rank=max_rank)

    p = [c.copy() for c in r]
    rr = tt_inner(r, r)
    b_norm = tt_norm(b_cores)
    if b_norm < 1e-30:
        b_norm = 1.0

    residuals: list[float] = [np.sqrt(max(rr, 0.0))]

    for it in range(max_iter):
        # Ap = A p
        Ap = tt_matvec(mpo_cores, p, max_rank=max_rank)
        pAp = tt_inner(p, Ap)

        if abs(pAp) < 1e-30:
            break

        alpha = rr / pAp

        # x = x + alpha * p
        x = tt_axpy(alpha, p, x, max_rank=max_rank)

        # r = r - alpha * Ap
        r = tt_axpy(-alpha, Ap, r, max_rank=max_rank)

        rr_new = tt_inner(r, r)
        res_norm = np.sqrt(max(rr_new, 0.0))
        residuals.append(res_norm)

        if res_norm / b_norm < tol:
            return TTKrylovResult(
                x=x,
                residual_norms=residuals,
                converged=True,
                n_iter=it + 1,
            )

        beta = rr_new / max(rr, 1e-30)
        p = tt_axpy(beta, p, [c.copy() for c in r], max_rank=max_rank)
        rr = rr_new

    return TTKrylovResult(
        x=x,
        residual_norms=residuals,
        converged=False,
        n_iter=max_iter,
    )


# ======================================================================
# TT-GMRES
# ======================================================================

def tt_gmres(
    mpo_cores: list[NDArray],
    b_cores: list[NDArray],
    x0: Optional[list[NDArray]] = None,
    max_iter: int = 50,
    restart: int = 20,
    tol: float = 1e-8,
    max_rank: int = 64,
) -> TTKrylovResult:
    """
    GMRES in TT format for general (non-symmetric) systems ``A x = b``.

    Uses Arnoldi iteration with TT inner products and modified
    Gram-Schmidt.  Restarts every *restart* iterations.

    Parameters
    ----------
    mpo_cores : list[NDArray]
        Operator A as MPO cores.
    b_cores : list[NDArray]
        Right-hand-side b as TT-vector cores.
    x0 : list[NDArray], optional
        Initial guess.
    max_iter : int
        Maximum total iterations.
    restart : int
        Restart period (inner GMRES iterations per cycle).
    tol : float
        Residual tolerance.
    max_rank : int
        Maximum TT rank.

    Returns
    -------
    TTKrylovResult
    """
    N = len(b_cores)
    d = b_cores[0].shape[1]

    if x0 is None:
        x = [np.zeros((1, d, 1), dtype=b_cores[0].dtype) for _ in range(N)]
    else:
        x = [c.copy() for c in x0]

    b_norm = tt_norm(b_cores)
    if b_norm < 1e-30:
        b_norm = 1.0

    all_residuals: list[float] = []
    total_iter = 0

    for _ in range(max_iter // max(restart, 1) + 1):
        # r = b - A x
        Ax = tt_matvec(mpo_cores, x, max_rank=max_rank)
        r = tt_axpy(-1.0, Ax, [c.copy() for c in b_cores], max_rank=max_rank)
        beta = tt_norm(r)
        all_residuals.append(beta)

        if beta / b_norm < tol:
            return TTKrylovResult(
                x=x, residual_norms=all_residuals,
                converged=True, n_iter=total_iter,
            )

        # Arnoldi basis
        V: list[list[NDArray]] = [tt_scale(r, 1.0 / (beta + 1e-30))]
        H = np.zeros((restart + 1, restart), dtype=np.float64)
        g = np.zeros(restart + 1, dtype=np.float64)
        g[0] = beta

        # Givens rotations
        cs = np.zeros(restart, dtype=np.float64)
        sn = np.zeros(restart, dtype=np.float64)

        for j in range(restart):
            total_iter += 1
            # w = A v_j
            w = tt_matvec(mpo_cores, V[j], max_rank=max_rank)

            # Modified Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = tt_inner(V[i], w)
                w = tt_axpy(-H[i, j], V[i], w, max_rank=max_rank)

            H[j + 1, j] = tt_norm(w)

            if H[j + 1, j] > 1e-14:
                V.append(tt_scale(w, 1.0 / H[j + 1, j]))
                V[-1] = tt_round(V[-1], max_rank=max_rank)
            else:
                # Breakdown — lucky convergence
                V.append(w)

            # Apply previous Givens rotations
            for i in range(j):
                tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp

            # Current Givens rotation
            rho = np.hypot(H[j, j], H[j + 1, j])
            if rho > 1e-30:
                cs[j] = H[j, j] / rho
                sn[j] = H[j + 1, j] / rho
            else:
                cs[j] = 1.0
                sn[j] = 0.0

            H[j, j] = rho
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]

            res_est = abs(g[j + 1])
            all_residuals.append(res_est)

            if res_est / b_norm < tol:
                # Solve small upper-triangular system
                y = np.linalg.solve(H[:j + 1, :j + 1], g[:j + 1])
                for i in range(j + 1):
                    x = tt_axpy(y[i], V[i], x, max_rank=max_rank)
                return TTKrylovResult(
                    x=x, residual_norms=all_residuals,
                    converged=True, n_iter=total_iter,
                )

        # End of restart cycle: update x
        m = restart
        y = np.linalg.solve(H[:m, :m], g[:m])
        for i in range(m):
            x = tt_axpy(y[i], V[i], x, max_rank=max_rank)

    return TTKrylovResult(
        x=x,
        residual_norms=all_residuals,
        converged=False,
        n_iter=total_iter,
    )
