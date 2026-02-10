"""
Isogeometric Analysis (IGA)
============================

Uses NURBS (Non-Uniform Rational B-Splines) as both the geometry
representation and the analysis basis, providing exact geometry
and high-order smoothness.

Key idea: the same NURBS basis that defines CAD geometry is used
directly for the finite-element approximation, eliminating mesh
generation and geometry approximation errors.

B-spline basis (univariate, Cox–de Boor recursion):
    N_{i,0}(ξ) = 1 if ξ_i ≤ ξ < ξ_{i+1}, else 0
    N_{i,p}(ξ) = (ξ - ξ_i)/(ξ_{i+p} - ξ_i) N_{i,p-1}(ξ)
               + (ξ_{i+p+1} - ξ)/(ξ_{i+p+1} - ξ_{i+1}) N_{i+1,p-1}(ξ)

NURBS: R_i(ξ) = w_i N_{i,p}(ξ) / Σ_j w_j N_{j,p}(ξ)

References:
    [1] Hughes, Cottrell & Bazilevs, "Isogeometric Analysis: CAD,
        Finite Elements, NURBS, Exact Geometry and Mesh Refinement",
        Comput. Methods Appl. Mech. Eng. 194, 2005.
    [2] Cottrell, Hughes & Bazilevs, "Isogeometric Analysis: Toward
        Integration of CAD and FEA", Wiley, 2009.
    [3] Piegl & Tiller, "The NURBS Book", Springer, 1997.

Domain III.1 — Solid Mechanics / Isogeometric Analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# B-spline basis
# ---------------------------------------------------------------------------

def find_span(n: int, p: int, u: float, U: NDArray) -> int:
    """Find knot span index for parameter *u* in knot vector *U*.

    Equivalent to Algorithm A2.1 from The NURBS Book.
    """
    if u >= U[n + 1]:
        return n
    if u <= U[p]:
        return p

    lo, hi = p, n + 1
    mid = (lo + hi) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


def basis_funs(i: int, u: float, p: int, U: NDArray) -> NDArray:
    """
    Evaluate non-zero B-spline basis functions at *u*.

    Returns array of length ``p + 1``: N[i-p], ..., N[i].
    Algorithm A2.2 from The NURBS Book.
    """
    N = np.zeros(p + 1, dtype=np.float64)
    N[0] = 1.0
    left = np.zeros(p + 1, dtype=np.float64)
    right = np.zeros(p + 1, dtype=np.float64)

    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            if abs(denom) < 1e-30:
                temp = 0.0
            else:
                temp = N[r] / denom
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N


def basis_funs_derivs(
    i: int, u: float, p: int, U: NDArray, n_deriv: int = 1,
) -> NDArray:
    """
    Evaluate B-spline basis functions and derivatives.

    Returns array ``(n_deriv + 1, p + 1)`` — ders[k][j] = d^k N_{i-p+j}/du^k.
    Algorithm A2.3 from The NURBS Book.
    """
    ndu = np.zeros((p + 1, p + 1), dtype=np.float64)
    ndu[0, 0] = 1.0
    left = np.zeros(p + 1, dtype=np.float64)
    right = np.zeros(p + 1, dtype=np.float64)

    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / (ndu[j, r] + 1e-30)
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    ders = np.zeros((n_deriv + 1, p + 1), dtype=np.float64)
    for j in range(p + 1):
        ders[0, j] = ndu[j, p]

    a = np.zeros((2, p + 1), dtype=np.float64)
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, n_deriv + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / (ndu[pk + 1, rk] + 1e-30)
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / (ndu[pk + 1, rk + j] + 1e-30)
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / (ndu[pk + 1, r] + 1e-30)
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1

    r_factor = float(p)
    for k in range(1, n_deriv + 1):
        ders[k] *= r_factor
        r_factor *= (p - k)

    return ders


# ---------------------------------------------------------------------------
# NURBS helpers
# ---------------------------------------------------------------------------

def nurbs_basis(
    i: int, u: float, p: int, U: NDArray, weights: NDArray,
) -> Tuple[NDArray, NDArray]:
    """
    Evaluate NURBS basis R and its derivative dR/du.

    R_j = w_j N_j / Σ_k w_k N_k.
    """
    ders = basis_funs_derivs(i, u, p, U, n_deriv=1)  # (2, p+1)
    N = ders[0]
    dN = ders[1]

    w_local = weights[i - p: i + 1]

    W = np.dot(N, w_local)
    dW = np.dot(dN, w_local)

    R = w_local * N / (W + 1e-30)
    dR = w_local * (dN * W - N * dW) / (W ** 2 + 1e-30)

    return R, dR


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class IGAState:
    """IGA solution state.

    Attributes:
        u: Control-point solution values ``(n_cp,)`` or ``(n_cp, dim)``.
    """
    u: NDArray

    @property
    def n_cp(self) -> int:
        return self.u.shape[0]


# ---------------------------------------------------------------------------
# 1D IGA Solver
# ---------------------------------------------------------------------------

class IGASolver1D:
    r"""
    1D Isogeometric Analysis solver for elliptic and parabolic problems.

    Uses B-spline (or NURBS) basis with exact geometry.

    Solves:
        :math:`-\frac{d}{dx}\left(a(x)\frac{du}{dx}\right) = f(x)`

    with Dirichlet boundary conditions.

    Parameters:
        knot_vector: Open knot vector ``(n + p + 2,)``.
        degree: Polynomial degree.
        control_points: Physical coordinates ``(n + 1,)``.
        weights: NURBS weights ``(n + 1,)``; all-ones for B-splines.
        n_gauss: Quadrature points per non-zero knot span.

    Example::

        U = np.array([0,0,0, 0.5, 1,1,1], dtype=np.float64)
        cp = np.array([0.0, 0.25, 0.75, 1.0])
        solver = IGASolver1D(knot_vector=U, degree=2, control_points=cp)
        state = solver.solve_poisson(lambda x: np.sin(np.pi * x))
    """

    def __init__(
        self,
        knot_vector: NDArray,
        degree: int,
        control_points: NDArray,
        weights: Optional[NDArray] = None,
        n_gauss: int = 4,
    ) -> None:
        self.U = np.asarray(knot_vector, dtype=np.float64)
        self.p = degree
        self.cp = np.asarray(control_points, dtype=np.float64)
        self.n = len(self.cp) - 1  # Number of basis functions = n + 1
        self.weights = (
            np.asarray(weights, dtype=np.float64)
            if weights is not None
            else np.ones(self.n + 1, dtype=np.float64)
        )
        self.n_gauss = n_gauss

        # Find unique knot spans (non-zero measure)
        unique_knots = np.unique(self.U)
        self.spans = []
        for k in range(len(unique_knots) - 1):
            if unique_knots[k + 1] - unique_knots[k] > 1e-14:
                self.spans.append((unique_knots[k], unique_knots[k + 1]))

        # Gauss quadrature on [-1,1]
        self.gp, self.gw = np.polynomial.legendre.leggauss(n_gauss)

    def _map_to_physical(self, u_param: float) -> float:
        """Map parameter u to physical coordinate x via NURBS."""
        i = find_span(self.n, self.p, u_param, self.U)
        R, _ = nurbs_basis(i, u_param, self.p, self.U, self.weights)
        return float(R @ self.cp[i - self.p: i + 1])

    def _jacobian(self, u_param: float) -> float:
        """Compute dx/du at parameter value u."""
        i = find_span(self.n, self.p, u_param, self.U)
        _, dR = nurbs_basis(i, u_param, self.p, self.U, self.weights)
        return float(dR @ self.cp[i - self.p: i + 1])

    def assemble_stiffness(
        self,
        a_func: Optional[Callable] = None,
    ) -> NDArray:
        """
        Assemble stiffness matrix for -d/dx(a du/dx).

        K_IJ = ∫ a(x) dR_I/dx dR_J/dx dx.
        """
        from typing import Callable as _TCallable
        n_dof = self.n + 1
        K = np.zeros((n_dof, n_dof), dtype=np.float64)

        for ua, ub in self.spans:
            for gq in range(self.n_gauss):
                u_local = 0.5 * ((ub - ua) * self.gp[gq] + (ua + ub))
                w_local = 0.5 * (ub - ua) * self.gw[gq]

                i = find_span(self.n, self.p, u_local, self.U)
                R, dR = nurbs_basis(i, u_local, self.p, self.U, self.weights)

                J = float(dR @ self.cp[i - self.p: i + 1])
                if abs(J) < 1e-30:
                    continue
                dRdx = dR / J
                x = float(R @ self.cp[i - self.p: i + 1])

                a_val = 1.0 if a_func is None else float(a_func(x))

                local_ids = list(range(i - self.p, i + 1))
                for li, gi in enumerate(local_ids):
                    for lj, gj in enumerate(local_ids):
                        K[gi, gj] += a_val * dRdx[li] * dRdx[lj] * abs(J) * w_local
        return K

    def assemble_mass(self) -> NDArray:
        """Mass matrix M_IJ = ∫ R_I R_J dx."""
        n_dof = self.n + 1
        M = np.zeros((n_dof, n_dof), dtype=np.float64)

        for ua, ub in self.spans:
            for gq in range(self.n_gauss):
                u_local = 0.5 * ((ub - ua) * self.gp[gq] + (ua + ub))
                w_local = 0.5 * (ub - ua) * self.gw[gq]

                i = find_span(self.n, self.p, u_local, self.U)
                R, dR = nurbs_basis(i, u_local, self.p, self.U, self.weights)
                J = abs(float(dR @ self.cp[i - self.p: i + 1]))
                if J < 1e-30:
                    continue

                local_ids = list(range(i - self.p, i + 1))
                for li, gi in enumerate(local_ids):
                    for lj, gj in enumerate(local_ids):
                        M[gi, gj] += R[li] * R[lj] * J * w_local
        return M

    def assemble_load(
        self,
        f_func: Callable,
    ) -> NDArray:
        """Load vector F_I = ∫ R_I f(x) dx."""
        from typing import Callable
        n_dof = self.n + 1
        F = np.zeros(n_dof, dtype=np.float64)

        for ua, ub in self.spans:
            for gq in range(self.n_gauss):
                u_local = 0.5 * ((ub - ua) * self.gp[gq] + (ua + ub))
                w_local = 0.5 * (ub - ua) * self.gw[gq]

                i = find_span(self.n, self.p, u_local, self.U)
                R, dR = nurbs_basis(i, u_local, self.p, self.U, self.weights)
                J = abs(float(dR @ self.cp[i - self.p: i + 1]))
                if J < 1e-30:
                    continue
                x = float(R @ self.cp[i - self.p: i + 1])

                local_ids = list(range(i - self.p, i + 1))
                for li, gi in enumerate(local_ids):
                    F[gi] += R[li] * f_func(x) * J * w_local
        return F

    def solve_poisson(
        self,
        f_func: Callable,
        u_left: float = 0.0,
        u_right: float = 0.0,
        a_func: Optional[Callable] = None,
    ) -> IGAState:
        """Solve Poisson equation with Dirichlet BCs."""
        K = self.assemble_stiffness(a_func)
        F = self.assemble_load(f_func)

        # Dirichlet BCs at first and last control point
        K[0, :] = 0.0
        K[0, 0] = 1.0
        F[0] = u_left
        K[-1, :] = 0.0
        K[-1, -1] = 1.0
        F[-1] = u_right

        u = np.linalg.solve(K, F)
        return IGAState(u=u)

    def evaluate(self, state: IGAState, n_pts: int = 100) -> Tuple[NDArray, NDArray]:
        """Evaluate the IGA solution on a fine parameter grid."""
        u_params = np.linspace(self.U[self.p], self.U[self.n + 1], n_pts)
        x_out = np.zeros(n_pts, dtype=np.float64)
        u_out = np.zeros(n_pts, dtype=np.float64)

        for k, u_param in enumerate(u_params):
            i = find_span(self.n, self.p, min(u_param, self.U[self.n + 1] - 1e-14), self.U)
            R, _ = nurbs_basis(i, u_param, self.p, self.U, self.weights)
            local_u = state.u[i - self.p: i + 1]
            local_cp = self.cp[i - self.p: i + 1]
            x_out[k] = float(R @ local_cp)
            u_out[k] = float(R @ local_u)
        return x_out, u_out

    def l2_error(
        self,
        state: IGAState,
        exact: Callable,
    ) -> float:
        """L² error against exact solution."""
        err2 = 0.0
        for ua, ub in self.spans:
            for gq in range(self.n_gauss):
                u_local = 0.5 * ((ub - ua) * self.gp[gq] + (ua + ub))
                w_local = 0.5 * (ub - ua) * self.gw[gq]

                i = find_span(self.n, self.p, u_local, self.U)
                R, dR = nurbs_basis(i, u_local, self.p, self.U, self.weights)
                J = abs(float(dR @ self.cp[i - self.p: i + 1]))
                x = float(R @ self.cp[i - self.p: i + 1])
                u_h = float(R @ state.u[i - self.p: i + 1])
                err2 += (u_h - exact(x)) ** 2 * J * w_local
        return float(np.sqrt(err2))
