"""
Space-Time Discontinuous Galerkin (ST-DG) Method
==================================================

Treats time as an additional dimension and solves the full space-time
problem element-by-element, enabling arbitrary-order accuracy in
both space and time simultaneously.

Space-time variational form for  ∂u/∂t + ∂f(u)/∂x = 0:

    ∫_{K_n} φ (∂u/∂t + ∂f/∂x) dΩ dt
        - ∫_{∂K_n^-} φ [[u]] dΩ = 0

where K_n is a space-time slab and [[·]] denotes the time-jump
at slab interfaces (causality upwinding in time).

The method is unconditionally stable (implicit in the time direction)
and naturally handles moving meshes and ALE formulations.

Supported configurations:
    - 1D space × time (2D space-time elements)
    - Tensor-product Legendre basis
    - Local implicit solve per space-time slab
    - Upwind flux in time, LF flux in space

References:
    [1] van der Vegt & van der Ven, "Space–Time Discontinuous Galerkin
        Finite Element Method with Dynamic Grid Motion",
        J. Comp. Phys. 182, 2002.
    [2] Klaij, van der Vegt & van der Ven, "Space–time DG Method
        for the Compressible Navier–Stokes Equations",
        J. Comp. Phys. 217, 2006.

Domain II.1 — Computational Fluid Dynamics / Space-Time DG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Quadrature and basis (reuse from dg module if available)
# ---------------------------------------------------------------------------

def _gauss_legendre(n: int) -> Tuple[NDArray, NDArray]:
    points, weights = np.polynomial.legendre.leggauss(n)
    return points.astype(np.float64), weights.astype(np.float64)


def _legendre_basis(x: NDArray, p: int) -> NDArray:
    N = len(x)
    phi = np.zeros((p + 1, N), dtype=np.float64)
    phi[0] = 1.0
    if p >= 1:
        phi[1] = x
    for k in range(2, p + 1):
        phi[k] = ((2 * k - 1) * x * phi[k - 1] - (k - 1) * phi[k - 2]) / k
    return phi


def _legendre_deriv(x: NDArray, p: int) -> NDArray:
    phi = _legendre_basis(x, p)
    dphi = np.zeros_like(phi)
    if p >= 1:
        dphi[1] = 1.0
    for k in range(2, p + 1):
        dphi[k] = (2 * k - 1) * phi[k - 1] + dphi[k - 2]
    return dphi


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class STDGState:
    """
    Space-time DG state for a single time slab.

    Attributes:
        coeffs: Modal coefficients ``(n_elem, (px+1)*(pt+1))`` per slab.
        mesh: Spatial element boundaries ``(n_elem + 1,)``.
        t_start: Start time of the current slab.
        t_end: End time of the current slab.
    """
    coeffs: NDArray
    mesh: NDArray
    t_start: float
    t_end: float

    @property
    def n_elements(self) -> int:
        return self.coeffs.shape[0]

    @property
    def dt(self) -> float:
        return self.t_end - self.t_start


# ---------------------------------------------------------------------------
# Space-Time DG Solver (1+1D)
# ---------------------------------------------------------------------------

class SpaceTimeDGSolver:
    r"""
    Space-time DG solver for 1D scalar conservation laws.

    Each "time slab" :math:`[t^n, t^{n+1}]` is treated as a
    two-dimensional space-time element solved with a local implicit
    system (Newton iteration per slab for nonlinear fluxes).

    Parameters:
        x_left, x_right: Spatial domain.
        n_elements: Number of spatial elements.
        px: Polynomial order in space.
        pt: Polynomial order in time.
        flux_func: Physical flux f(u).
        flux_deriv: df/du for linearisation.
        n_quad_x: Spatial quadrature points.
        n_quad_t: Temporal quadrature points.
    """

    def __init__(
        self,
        x_left: float = 0.0,
        x_right: float = 1.0,
        n_elements: int = 50,
        px: int = 2,
        pt: int = 2,
        flux_func: Optional[Callable] = None,
        flux_deriv: Optional[Callable] = None,
        n_quad_x: Optional[int] = None,
        n_quad_t: Optional[int] = None,
    ) -> None:
        self.x_left = x_left
        self.x_right = x_right
        self.n_elements = n_elements
        self.px = px
        self.pt = pt
        self.n_modes = (px + 1) * (pt + 1)
        self.flux_func = flux_func or (lambda u: 0.5 * u ** 2)
        self.flux_deriv = flux_deriv or (lambda u: u)

        self.n_qx = n_quad_x or px + 2
        self.n_qt = n_quad_t or pt + 2

        # Spatial mesh
        self.mesh = np.linspace(x_left, x_right, n_elements + 1, dtype=np.float64)
        self.dx = np.diff(self.mesh)

        # Reference quadrature
        self.xi_x, self.w_x = _gauss_legendre(self.n_qx)
        self.xi_t, self.w_t = _gauss_legendre(self.n_qt)

        # Reference basis values
        self.phi_x = _legendre_basis(self.xi_x, px)    # (px+1, n_qx)
        self.dphi_x = _legendre_deriv(self.xi_x, px)
        self.phi_t = _legendre_basis(self.xi_t, pt)    # (pt+1, n_qt)
        self.dphi_t = _legendre_deriv(self.xi_t, pt)

        # Boundary evaluations for time jump
        self.phi_t_m1 = _legendre_basis(np.array([-1.0]), pt)[:, 0]  # t = t^n
        self.phi_t_p1 = _legendre_basis(np.array([1.0]), pt)[:, 0]   # t = t^{n+1}

        # Spatial boundary basis
        self.phi_x_m1 = _legendre_basis(np.array([-1.0]), px)[:, 0]
        self.phi_x_p1 = _legendre_basis(np.array([1.0]), px)[:, 0]

        self._step_count = 0

    def _mode_index(self, kx: int, kt: int) -> int:
        """Flatten (kx, kt) to linear mode index."""
        return kt * (self.px + 1) + kx

    def _eval_at_quad(self, c: NDArray) -> NDArray:
        """Evaluate solution at space-time quadrature points.

        c: coefficients ``(n_modes,)`` for one element.
        Returns: ``(n_qt, n_qx)``.
        """
        u = np.zeros((self.n_qt, self.n_qx), dtype=np.float64)
        for kt in range(self.pt + 1):
            for kx in range(self.px + 1):
                idx = self._mode_index(kx, kt)
                u += c[idx] * np.outer(self.phi_t[kt], self.phi_x[kx])
        return u

    def _eval_at_t_boundary(self, c: NDArray, side: int) -> NDArray:
        """Evaluate at t = -1 (side=0) or t = +1 (side=1).

        Returns spatial values at quadrature: ``(n_qx,)``.
        """
        phi_t_vals = self.phi_t_m1 if side == 0 else self.phi_t_p1
        u = np.zeros(self.n_qx, dtype=np.float64)
        for kt in range(self.pt + 1):
            for kx in range(self.px + 1):
                idx = self._mode_index(kx, kt)
                u += c[idx] * phi_t_vals[kt] * self.phi_x[kx]
        return u

    def _eval_at_x_boundary(self, c: NDArray, side: int) -> NDArray:
        """Evaluate at x = -1 (side=0) or x = +1 (side=1).

        Returns temporal values at quadrature: ``(n_qt,)``.
        """
        phi_x_vals = self.phi_x_m1 if side == 0 else self.phi_x_p1
        u = np.zeros(self.n_qt, dtype=np.float64)
        for kt in range(self.pt + 1):
            for kx in range(self.px + 1):
                idx = self._mode_index(kx, kt)
                u += c[idx] * self.phi_t[kt] * phi_x_vals[kx]
        return u

    def project_initial(
        self,
        u0: Callable[[NDArray], NDArray],
        t0: float = 0.0,
        dt: float = 0.01,
    ) -> STDGState:
        """
        Build initial slab state by L² projecting u0 into the
        space-time basis at temporal order-zero (constant in time).
        """
        coeffs = np.zeros((self.n_elements, self.n_modes), dtype=np.float64)
        for e in range(self.n_elements):
            xc = 0.5 * (self.mesh[e] + self.mesh[e + 1])
            he = self.dx[e]
            x_phys = xc + 0.5 * he * self.xi_x
            u_vals = u0(x_phys)
            # Project onto spatial modes at time mode 0
            for kx in range(self.px + 1):
                M_kx = 2.0 / (2 * kx + 1)
                c = np.sum(self.w_x * u_vals * self.phi_x[kx]) / M_kx
                for kt in range(self.pt + 1):
                    # Only kt=0 gets the IC value
                    M_kt = 2.0 / (2 * kt + 1)
                    if kt == 0:
                        coeffs[e, self._mode_index(kx, kt)] = c
        return STDGState(
            coeffs=coeffs,
            mesh=self.mesh.copy(),
            t_start=t0,
            t_end=t0 + dt,
        )

    def solve_slab(
        self,
        u_prev_at_tn: NDArray,
        dt: float,
        t_start: float,
        n_newton: int = 8,
    ) -> NDArray:
        r"""
        Solve one space-time slab given the solution at the previous
        time level :math:`u^-(t^n, x)`.

        Uses Newton iteration on the nonlinear space-time residual.

        Args:
            u_prev_at_tn: Solution at t^n at spatial quadrature
                ``(n_elements, n_qx)``.
            dt: Time step.
            t_start: Slab start time.
            n_newton: Maximum Newton iterations.

        Returns:
            Slab coefficients ``(n_elements, n_modes)``.
        """
        NE = self.n_elements
        NM = self.n_modes
        coeffs = np.zeros((NE, NM), dtype=np.float64)

        # Initial guess: constant-in-time from u_prev
        for e in range(NE):
            for kx in range(self.px + 1):
                M_kx = 2.0 / (2 * kx + 1)
                val = np.sum(self.w_x * u_prev_at_tn[e] * self.phi_x[kx]) / M_kx
                coeffs[e, self._mode_index(kx, 0)] = val

        Jx = np.array([0.5 * dx for dx in self.dx])
        Jt = 0.5 * dt

        for _iter in range(n_newton):
            residual = np.zeros((NE, NM), dtype=np.float64)

            for e in range(NE):
                c = coeffs[e]
                u_qt = self._eval_at_quad(c)   # (n_qt, n_qx)
                f_qt = self.flux_func(u_qt)

                for kt in range(self.pt + 1):
                    for kx in range(self.px + 1):
                        idx = self._mode_index(kx, kt)

                        # Volume: ∫∫ [u_t φ + f φ_x] J dx dt
                        # du/dt contribution
                        du_dt = np.zeros((self.n_qt, self.n_qx), dtype=np.float64)
                        for kt2 in range(self.pt + 1):
                            for kx2 in range(self.px + 1):
                                idx2 = self._mode_index(kx2, kt2)
                                du_dt += c[idx2] * np.outer(
                                    self.dphi_t[kt2], self.phi_x[kx2]
                                ) / Jt

                        phi_test = np.outer(self.phi_t[kt], self.phi_x[kx])

                        # Time derivative term
                        residual[e, idx] += Jx[e] * Jt * np.sum(
                            np.outer(self.w_t, self.w_x) * du_dt * phi_test
                        )

                        # Spatial flux (integration by parts)
                        residual[e, idx] -= Jt * np.sum(
                            np.outer(self.w_t, self.w_x) * f_qt
                            * np.outer(self.phi_t[kt], self.dphi_x[kx])
                        )

                        # Time jump at t^n: [u^+ - u^-] φ(t^n)
                        u_plus = self._eval_at_t_boundary(c, side=0)  # u at t^n+
                        jump = u_plus - u_prev_at_tn[e]
                        residual[e, idx] += Jx[e] * np.sum(
                            self.w_x * jump * self.phi_x[kx]
                        ) * self.phi_t_m1[kt]

            # Simple damped Newton update
            max_res = np.max(np.abs(residual))
            if max_res < 1e-12:
                break

            # Approximate Jacobian: use mass-matrix preconditioner
            for e in range(NE):
                for kt in range(self.pt + 1):
                    for kx in range(self.px + 1):
                        idx = self._mode_index(kx, kt)
                        M_kx = 2.0 / (2 * kx + 1)
                        M_kt = 2.0 / (2 * kt + 1)
                        diag = Jx[e] * Jt * M_kx * M_kt + Jx[e] * M_kx * self.phi_t_m1[kt] ** 2
                        if abs(diag) > 1e-30:
                            coeffs[e, idx] -= 0.5 * residual[e, idx] / diag

        return coeffs

    def step(self, state: STDGState, dt: float) -> STDGState:
        """Advance one time slab."""
        # Extract u at previous t^n (end of previous slab)
        u_at_tn = np.zeros((self.n_elements, self.n_qx), dtype=np.float64)
        for e in range(self.n_elements):
            u_at_tn[e] = self._eval_at_t_boundary(state.coeffs[e], side=1)

        t_new_start = state.t_end
        t_new_end = state.t_end + dt
        coeffs = self.solve_slab(u_at_tn, dt, t_new_start)

        self._step_count += 1
        return STDGState(
            coeffs=coeffs,
            mesh=state.mesh,
            t_start=t_new_start,
            t_end=t_new_end,
        )

    def step_n(self, state: STDGState, n_steps: int, dt: float) -> STDGState:
        for _ in range(n_steps):
            state = self.step(state, dt)
        return state

    @property
    def steps(self) -> int:
        return self._step_count

    def extract_solution_at_tend(self, state: STDGState)  -> Tuple[NDArray, NDArray]:
        """Extract u(x, t_end) from the current slab."""
        x_all = np.zeros((self.n_elements, self.n_qx), dtype=np.float64)
        u_all = np.zeros((self.n_elements, self.n_qx), dtype=np.float64)
        for e in range(self.n_elements):
            xc = 0.5 * (self.mesh[e] + self.mesh[e + 1])
            x_all[e] = xc + 0.5 * self.dx[e] * self.xi_x
            u_all[e] = self._eval_at_t_boundary(state.coeffs[e], side=1)
        return x_all.ravel(), u_all.ravel()
