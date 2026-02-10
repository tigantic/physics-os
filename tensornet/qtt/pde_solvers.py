"""
QTT-PDE: Implicit Time-Stepping in TT Format
==============================================

Full implicit integrators (backward Euler, Crank–Nicolson, BDF-2)
that solve the resulting linear systems entirely in TT format using
the Krylov solvers from :mod:`tensornet.qtt.krylov`.

The user supplies an MPO representation of the spatial operator *L* and
a TT-vector initial condition.  The module assembles the implicit system
``(I - dt * L) u_{n+1} = rhs`` and solves it with TT-CG or TT-GMRES.

Key classes / functions
-----------------------
* :class:`PDEConfig`         — integrator parameters
* :class:`PDEResult`         — full solution trajectory
* :func:`backward_euler`     — first-order A-stable
* :func:`crank_nicolson`     — second-order A-stable
* :func:`bdf2`               — second-order L-stable
* :func:`identity_mpo`       — build I in MPO format
* :func:`shifted_operator`   — build (I − α L) MPO
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from tensornet.qtt.sparse_direct import tt_round, tt_matvec
from tensornet.qtt.eigensolvers import tt_axpy, tt_scale, tt_add
from tensornet.qtt.krylov import tt_cg, tt_gmres


# ======================================================================
# Helper: identity MPO
# ======================================================================

def identity_mpo(n_sites: int, d: int = 2) -> list[NDArray]:
    """
    Build the identity operator as an MPO with bond dimension 1.

    Parameters
    ----------
    n_sites : int
        Number of sites.
    d : int
        Physical dimension at each site.

    Returns
    -------
    list[NDArray]
        MPO cores, each of shape (1, d, d, 1).
    """
    cores: list[NDArray] = []
    for _ in range(n_sites):
        core = np.zeros((1, d, d, 1))
        for i in range(d):
            core[0, i, i, 0] = 1.0
        cores.append(core)
    return cores


def shifted_operator(
    L_cores: list[NDArray],
    alpha: float,
    d: int = 2,
) -> list[NDArray]:
    """
    Build MPO for (I − α L).

    Uses block-diagonal stacking of identity and L MPO cores.

    Parameters
    ----------
    L_cores : list[NDArray]
        MPO cores for the spatial operator L.
    alpha : float
        Coefficient (typically dt or dt/2).
    d : int
        Physical dimension.

    Returns
    -------
    list[NDArray]
        MPO cores for (I − α L), with bond dimension = 1 + r_L.
    """
    N = len(L_cores)
    I_cores = identity_mpo(N, d)
    result: list[NDArray] = []

    for k in range(N):
        ri_l, di, do, ri_r = I_cores[k].shape
        rl_l, dl, dlo, rl_r = L_cores[k].shape

        if di != dl or do != dlo:
            raise ValueError(
                f"Site {k}: physical dimension mismatch "
                f"({di},{do}) vs ({dl},{dlo})"
            )

        if k == 0:
            # First site: stack horizontally [I_core | -alpha * L_core]
            new_r = ri_r + rl_r
            core = np.zeros((1, di, do, new_r))
            core[:, :, :, :ri_r] = I_cores[k]
            core[:, :, :, ri_r:] = -alpha * L_cores[k]
            result.append(core)
        elif k == N - 1:
            # Last site: stack vertically [I_core; L_core]
            new_l = ri_l + rl_l
            core = np.zeros((new_l, di, do, 1))
            core[:ri_l, :, :, :] = I_cores[k]
            core[ri_l:, :, :, :] = L_cores[k]
            result.append(core)
        else:
            # Middle sites: block diagonal
            new_l = ri_l + rl_l
            new_r = ri_r + rl_r
            core = np.zeros((new_l, di, do, new_r))
            core[:ri_l, :, :, :ri_r] = I_cores[k]
            core[ri_l:, :, :, ri_r:] = L_cores[k]
            result.append(core)

    return result


# ======================================================================
# Configuration and result
# ======================================================================

@dataclass
class PDEConfig:
    """
    Configuration for QTT-PDE integrator.

    Attributes
    ----------
    dt : float
        Time step.
    n_steps : int
        Number of time steps.
    solver : str
        Linear solver: 'cg' for symmetric positive-definite operators,
        'gmres' for general.
    max_rank : int
        Maximum TT bond dimension.
    krylov_tol : float
        Convergence tolerance for the linear solver.
    krylov_maxiter : int
        Maximum Krylov iterations per solve.
    save_every : int
        Store solution snapshot every *save_every* steps.
    """
    dt: float = 0.01
    n_steps: int = 100
    solver: str = 'cg'
    max_rank: int = 64
    krylov_tol: float = 1e-8
    krylov_maxiter: int = 200
    save_every: int = 10


@dataclass
class PDEResult:
    """
    Result from a QTT-PDE integration.

    Attributes
    ----------
    snapshots : list[list[NDArray]]
        Solution TT-cores at saved time steps.
    times : list[float]
        Physical times of snapshots.
    residuals : list[float]
        Final residual of each linear solve.
    converged_steps : int
        Number of steps where the linear solver converged.
    total_steps : int
        Total number of steps attempted.
    """
    snapshots: list[list[NDArray]] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    converged_steps: int = 0
    total_steps: int = 0


# ======================================================================
# Solvers
# ======================================================================

def _solve(
    A_cores: list[NDArray],
    rhs_cores: list[NDArray],
    x0_cores: list[NDArray],
    config: PDEConfig,
) -> tuple[list[NDArray], float, bool]:
    """Dispatch to CG or GMRES."""
    if config.solver == 'cg':
        result = tt_cg(
            A_cores, rhs_cores, x0_cores,
            max_iter=config.krylov_maxiter,
            tol=config.krylov_tol,
            max_rank=config.max_rank,
        )
    elif config.solver == 'gmres':
        result = tt_gmres(
            A_cores, rhs_cores, x0_cores,
            max_iter=config.krylov_maxiter,
            tol=config.krylov_tol,
            max_rank=config.max_rank,
        )
    else:
        raise ValueError(f"Unknown solver: {config.solver!r}")

    final_res = result.residual_norms[-1] if result.residual_norms else float('inf')
    return result.x, final_res, result.converged


def backward_euler(
    L_cores: list[NDArray],
    u0_cores: list[NDArray],
    config: PDEConfig,
    source_fn: Optional[callable] = None,
    d: int = 2,
) -> PDEResult:
    """
    Backward Euler: solve ``(I − dt L) u_{n+1} = u_n [+ dt f_n]``
    at each step.

    Parameters
    ----------
    L_cores : list[NDArray]
        MPO cores for spatial operator L.
    u0_cores : list[NDArray]
        Initial condition TT-vector.
    config : PDEConfig
        Integration parameters.
    source_fn : callable, optional
        ``f(t) → list[NDArray]``: source term in TT format.
    d : int
        Physical dimension.

    Returns
    -------
    PDEResult
    """
    A_cores = shifted_operator(L_cores, config.dt, d=d)
    u = [c.copy() for c in u0_cores]
    result = PDEResult()
    result.snapshots.append([c.copy() for c in u])
    result.times.append(0.0)

    for step in range(1, config.n_steps + 1):
        t = step * config.dt
        rhs = [c.copy() for c in u]

        if source_fn is not None:
            src = source_fn(t)
            rhs = tt_axpy(config.dt, src, rhs, max_rank=config.max_rank)

        u_new, res, conv = _solve(A_cores, rhs, u, config)
        u = u_new
        result.residuals.append(res)
        result.total_steps += 1
        if conv:
            result.converged_steps += 1

        if step % config.save_every == 0 or step == config.n_steps:
            result.snapshots.append([c.copy() for c in u])
            result.times.append(t)

    return result


def crank_nicolson(
    L_cores: list[NDArray],
    u0_cores: list[NDArray],
    config: PDEConfig,
    source_fn: Optional[callable] = None,
    d: int = 2,
) -> PDEResult:
    """
    Crank–Nicolson: solve
    ``(I − dt/2 L) u_{n+1} = (I + dt/2 L) u_n [+ dt f_{n+1/2}]``.

    Parameters
    ----------
    L_cores, u0_cores, config, source_fn, d
        Same as :func:`backward_euler`.

    Returns
    -------
    PDEResult
    """
    A_cores = shifted_operator(L_cores, config.dt / 2.0, d=d)

    # Explicit half: (I + dt/2 L) — but we compute it as matvec with
    # the "explicit" shifted operator
    B_cores = shifted_operator(L_cores, -config.dt / 2.0, d=d)

    u = [c.copy() for c in u0_cores]
    result = PDEResult()
    result.snapshots.append([c.copy() for c in u])
    result.times.append(0.0)

    for step in range(1, config.n_steps + 1):
        t = step * config.dt
        rhs = tt_matvec(B_cores, u, max_rank=config.max_rank)

        if source_fn is not None:
            src = source_fn(t - config.dt / 2.0)
            rhs = tt_axpy(config.dt, src, rhs, max_rank=config.max_rank)

        u_new, res, conv = _solve(A_cores, rhs, u, config)
        u = u_new
        result.residuals.append(res)
        result.total_steps += 1
        if conv:
            result.converged_steps += 1

        if step % config.save_every == 0 or step == config.n_steps:
            result.snapshots.append([c.copy() for c in u])
            result.times.append(t)

    return result


def bdf2(
    L_cores: list[NDArray],
    u0_cores: list[NDArray],
    config: PDEConfig,
    source_fn: Optional[callable] = None,
    d: int = 2,
) -> PDEResult:
    """
    BDF-2: second-order L-stable integrator.

    ``(I − 2dt/3 L) u_{n+1} = 4/3 u_n − 1/3 u_{n-1} [+ 2dt/3 f_{n+1}]``

    Uses one backward Euler step for initialization.

    Parameters
    ----------
    L_cores, u0_cores, config, source_fn, d
        Same as :func:`backward_euler`.

    Returns
    -------
    PDEResult
    """
    dt = config.dt
    A_cores = shifted_operator(L_cores, 2.0 * dt / 3.0, d=d)

    u_prev = [c.copy() for c in u0_cores]
    result = PDEResult()
    result.snapshots.append([c.copy() for c in u_prev])
    result.times.append(0.0)

    # Step 1: backward Euler for initialization
    A1_cores = shifted_operator(L_cores, dt, d=d)
    rhs = [c.copy() for c in u_prev]
    if source_fn is not None:
        src = source_fn(dt)
        rhs = tt_axpy(dt, src, rhs, max_rank=config.max_rank)

    u_curr, res, conv = _solve(A1_cores, rhs, u_prev, config)
    result.residuals.append(res)
    result.total_steps += 1
    if conv:
        result.converged_steps += 1

    if config.save_every == 1:
        result.snapshots.append([c.copy() for c in u_curr])
        result.times.append(dt)

    # Steps 2+: BDF-2
    for step in range(2, config.n_steps + 1):
        t = step * dt

        # rhs = 4/3 u_n - 1/3 u_{n-1}
        rhs = tt_axpy(-1.0 / 3.0, u_prev, tt_scale(u_curr, 4.0 / 3.0),
                       max_rank=config.max_rank)

        if source_fn is not None:
            src = source_fn(t)
            rhs = tt_axpy(2.0 * dt / 3.0, src, rhs,
                          max_rank=config.max_rank)

        u_new, res, conv = _solve(A_cores, rhs, u_curr, config)
        u_prev = u_curr
        u_curr = u_new
        result.residuals.append(res)
        result.total_steps += 1
        if conv:
            result.converged_steps += 1

        if step % config.save_every == 0 or step == config.n_steps:
            result.snapshots.append([c.copy() for c in u_curr])
            result.times.append(t)

    return result
