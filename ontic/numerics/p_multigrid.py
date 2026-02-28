"""
p-Multigrid
============

Multigrid acceleration for high-order finite-element / spectral-element
methods by coarsening in **polynomial degree** while keeping the mesh
fixed.

Key idea: at each level, reduce the polynomial order p → p-1 → … → 1
and apply standard smoothing/coarsening. The coarsest level (p = 1)
may be solved directly or handed to a geometric/algebraic multigrid.

Cycle structure mirrors classical multigrid (V-cycle, W-cycle):

.. math::
    \\underbrace{S^{\\nu_1}(p)}_\\text{pre-smooth}
    \\to R_{p \\to p-1} r
    \\to \\text{coarse solve}
    \\to P_{p-1 \\to p} e
    \\to \\underbrace{S^{\\nu_2}(p)}_\\text{post-smooth}

References:
    [1] Rønquist & Patera, "Spectral element multigrid", J. Sci. Comp. 1987.
    [2] Fidkowski, Oliver & Lu, JCP 2005.

Domain I.3.4 — Numerics / Solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def _gauss_lobatto_points(p: int) -> NDArray:
    """
    Gauss-Lobatto-Legendre nodes on [-1, 1] for polynomial order p.

    Uses Chebyshev initial guess + Newton iteration on :math:`(1-x^2) P'_p(x) = 0`.
    """
    if p == 0:
        return np.array([0.0])
    if p == 1:
        return np.array([-1.0, 1.0])

    n = p + 1
    x = -np.cos(np.pi * np.arange(n) / p)

    for _ in range(50):
        Lp = np.zeros(n)
        Lp_prev = np.zeros(n)
        Lp[:] = 1.0
        Lp_prev[:] = 0.0
        L = np.zeros(n)
        L[:] = x

        for k in range(1, p):
            Lp_prev[:] = Lp
            Lp[:] = L
            L[:] = ((2 * k + 1) * x * Lp - k * Lp_prev) / (k + 1)

        dL = p * (Lp - x * L) / (1.0 - x**2 + 1e-30)
        dx = -L / (dL + 1e-30)
        x += dx
        if np.max(np.abs(dx)) < 1e-15:
            break

    x[0] = -1.0
    x[-1] = 1.0
    return x


def _build_interpolation_1d(x_fine: NDArray, x_coarse: NDArray) -> NDArray:
    """
    Build Lagrange interpolation matrix from coarse nodes to fine nodes.

    Returns shape ``(len(x_fine), len(x_coarse))`` matrix.
    """
    nc = len(x_coarse)
    nf = len(x_fine)
    I_mat = np.zeros((nf, nc))

    for j in range(nc):
        for i in range(nf):
            val = 1.0
            for k in range(nc):
                if k != j:
                    val *= (x_fine[i] - x_coarse[k]) / (x_coarse[j] - x_coarse[k] + 1e-30)
            I_mat[i, j] = val

    return I_mat


@dataclass
class PMultigridLevel:
    """Single level in the p-multigrid hierarchy."""
    p: int
    A: NDArray                    # system matrix at this polynomial order
    smoother: str = "jacobi"
    P: Optional[NDArray] = None   # prolongation to next finer
    R: Optional[NDArray] = None   # restriction to next coarser


class PMultigridSolver:
    """
    p-Multigrid solver for spectral / high-order FEM systems.

    The user supplies a function ``assemble(p) -> NDArray`` that returns the
    system matrix for polynomial order p (assembled on the *same* mesh).
    The solver automatically builds the polynomial coarsening hierarchy.

    Example::

        def assemble(p):
            n = p + 1
            # 1-D Laplacian on GLL nodes of order p
            x = gauss_lobatto_points(p)
            D = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        D[i, j] = 1.0
                        for k in range(n):
                            if k != i and k != j:
                                D[i, j] *= (x[i] - x[k]) / (x[j] - x[k])
            return D.T @ D

        solver = PMultigridSolver(p_max=8, assemble_fn=assemble)
        x = solver.solve(b)
    """

    def __init__(
        self,
        p_max: int,
        assemble_fn: Callable[[int], NDArray],
        p_min: int = 1,
        smoother: str = "jacobi",
        omega: float = 2.0 / 3.0,
    ) -> None:
        if p_max < p_min:
            raise ValueError(f"p_max ({p_max}) must be >= p_min ({p_min})")

        self.p_max = p_max
        self.p_min = p_min
        self.omega = omega
        self.smoother = smoother
        self.levels: List[PMultigridLevel] = []
        self._build_hierarchy(assemble_fn)

    def _build_hierarchy(self, assemble_fn: Callable[[int], NDArray]) -> None:
        """Build levels from p_max down to p_min."""
        orders = list(range(self.p_max, self.p_min - 1, -1))

        for p in orders:
            A = assemble_fn(p)
            self.levels.append(PMultigridLevel(p=p, A=A, smoother=self.smoother))

        # Build prolongation / restriction between consecutive levels
        for k in range(len(self.levels) - 1):
            p_fine = self.levels[k].p
            p_coarse = self.levels[k + 1].p

            x_fine = _gauss_lobatto_points(p_fine)
            x_coarse = _gauss_lobatto_points(p_coarse)

            P = _build_interpolation_1d(x_fine, x_coarse)
            R = P.T.copy()

            self.levels[k].P = P
            self.levels[k].R = R

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def _smooth(self, A: NDArray, x: NDArray, b: NDArray, n_iter: int) -> NDArray:
        """Weighted Jacobi smoothing with stability guard."""
        diag = np.diag(A).copy()
        diag[np.abs(diag) < 1e-30] = 1e-30
        for _ in range(n_iter):
            x = x + self.omega * (b - A @ x) / diag
            # Guard against overflow / NaN
            if not np.all(np.isfinite(x)):
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def v_cycle(
        self,
        b: NDArray,
        x: Optional[NDArray] = None,
        level: int = 0,
        pre_smooth: int = 3,
        post_smooth: int = 3,
    ) -> NDArray:
        """
        One p-multigrid V-cycle.

        Parameters:
            b: Right-hand side at this level.
            x: Initial guess (default: zero).
            level: Current level (0 = finest).
            pre_smooth: Number of pre-smoothing steps.
            post_smooth: Number of post-smoothing steps.

        Returns:
            Updated solution vector.
        """
        lev = self.levels[level]
        n = lev.A.shape[0]

        if x is None:
            x = np.zeros(n)

        # Coarsest level: direct solve
        if level == self.n_levels - 1 or lev.P is None:
            return np.linalg.solve(lev.A, b)

        # Pre-smooth
        x = self._smooth(lev.A, x, b, pre_smooth)

        # Restrict residual
        r = b - lev.A @ x
        r_coarse = lev.R @ r

        # Recurse
        e_coarse = self.v_cycle(
            r_coarse, level=level + 1,
            pre_smooth=pre_smooth, post_smooth=post_smooth,
        )

        # Prolongate correction
        x = x + lev.P @ e_coarse

        # Post-smooth
        x = self._smooth(lev.A, x, b, post_smooth)

        return x

    def solve(
        self,
        b: NDArray,
        x0: Optional[NDArray] = None,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> NDArray:
        """
        Solve Ax = b using p-multigrid V-cycles.

        Returns:
            Solution vector x.
        """
        A = self.levels[0].A
        x = x0 if x0 is not None else np.zeros(A.shape[0])

        r0 = np.linalg.norm(b - A @ x)
        if r0 < 1e-30:
            return x

        for _ in range(max_iter):
            x = self.v_cycle(b, x)
            r = np.linalg.norm(b - A @ x)
            if r / r0 < tol:
                break

        return x
