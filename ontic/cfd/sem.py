"""
Spectral Element Method (SEM)
==============================

High-order continuous Galerkin method using Gauss-Lobatto-Legendre
(GLL) nodal basis on each element with C⁰ continuity.

Weak form for elliptic problem -∇²u = f:
    ∫_Ω ∇φ·∇u dΩ = ∫_Ω φ f dΩ

    discretised element-by-element with GLL quadrature so that the
    mass matrix is diagonal (spectral lumping).

Key features:
    - GLL nodal basis → diagonal mass matrix
    - C⁰ continuity via direct stiffness summation
    - Spectral convergence for smooth solutions
    - Efficient tensor-product structure in 2D/3D

References:
    [1] Patera, "A Spectral Element Method for Fluid Dynamics",
        J. Comp. Phys. 54, 1984.
    [2] Karniadakis & Sherwin, "Spectral/hp Element Methods for
        Computational Fluid Dynamics", Oxford, 2005.
    [3] Deville, Fischer & Mund, "High-Order Methods for
        Incompressible Fluid Flow", Cambridge, 2002.

Domain II.1 — Computational Fluid Dynamics / Spectral Element Method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# GLL Quadrature
# ---------------------------------------------------------------------------

def gll_points(N: int) -> Tuple[NDArray, NDArray]:
    """
    Gauss-Lobatto-Legendre nodes and weights on [-1, 1].

    Uses Newton iteration on the derivative of the Legendre polynomial
    P'_N(x) for the interior points, plus the boundary points ±1.

    Args:
        N: Number of points (polynomial order p = N - 1).

    Returns:
        (xi, w) of length N.
    """
    if N < 2:
        raise ValueError("Need N >= 2 for GLL points")
    if N == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    x = np.zeros(N, dtype=np.float64)
    x[0] = -1.0
    x[-1] = 1.0

    # Initial guess: Chebyshev-Gauss-Lobatto
    for i in range(1, N - 1):
        x[i] = -np.cos(np.pi * i / (N - 1))

    # Newton iteration
    for _ in range(100):
        converged = True
        for i in range(1, N - 1):
            # Evaluate P_{N-1}(x_i) and P'_{N-1}(x_i) via recurrence
            P_prev = 1.0
            P_curr = x[i]
            for k in range(2, N):
                P_next = ((2 * k - 1) * x[i] * P_curr - (k - 1) * P_prev) / k
                P_prev = P_curr
                P_curr = P_next
            # P'_{N-1} = (N-1)(x P_{N-1} - P_{N-2}) / (x² - 1)
            dP = (N - 1) * (x[i] * P_curr - P_prev) / (x[i] ** 2 - 1.0 + 1e-30)
            # P''_{N-1} for Newton on P'
            d2P = (2.0 * x[i] * dP - (N - 1) * N * P_curr) / (1.0 - x[i] ** 2 + 1e-30)
            dx = -dP / (d2P + 1e-30)
            x[i] += dx
            if abs(dx) > 1e-15:
                converged = False
        if converged:
            break

    x.sort()

    # Weights: w_i = 2 / (N(N-1) [P_{N-1}(x_i)]²)
    P = np.ones(N, dtype=np.float64)
    P_curr_arr = x.copy()
    for k in range(2, N):
        P_prev_arr = P.copy()
        P = P_curr_arr.copy()
        P_curr_arr = ((2 * k - 1) * x * P - (k - 1) * P_prev_arr) / k

    w = 2.0 / (N * (N - 1) * P_curr_arr ** 2 + 1e-30)
    return x, w


# ---------------------------------------------------------------------------
# Derivative matrix
# ---------------------------------------------------------------------------

def derivative_matrix(xi: NDArray) -> NDArray:
    """
    Lagrange interpolant derivative matrix D_{ij} = ℓ'_j(ξ_i).

    For a function u represented at nodes ξ_j, du/dξ at ξ_i is:
        (du/dξ)_i = Σ_j D_{ij} u_j
    """
    N = len(xi)
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = 1.0
                for m in range(N):
                    if m != j:
                        if m == i:
                            D[i, j] *= 1.0 / (xi[j] - xi[m])
                        else:
                            D[i, j] *= (xi[i] - xi[m]) / (xi[j] - xi[m])
    # Diagonal from row-sum property
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])
    return D


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SEMState:
    """
    State for the Spectral Element Method.

    Attributes:
        u: Solution values at GLL nodes per element — ``(n_elem, N)``.
        mesh: Element boundaries ``(n_elem + 1,)``.
    """
    u: NDArray
    mesh: NDArray

    @property
    def n_elements(self) -> int:
        return self.u.shape[0]

    @property
    def N(self) -> int:
        """Number of GLL points per element."""
        return self.u.shape[1]

    @property
    def order(self) -> int:
        return self.N - 1


# ---------------------------------------------------------------------------
# 1D SEM Solver
# ---------------------------------------------------------------------------

class SEMSolver1D:
    r"""
    1D Spectral Element solver for elliptic and parabolic problems.

    Supports:
        - Steady Poisson: :math:`-u'' = f`
        - Unsteady diffusion: :math:`u_t = \nu u_{xx}`
        - Unsteady advection-diffusion: :math:`u_t + a u_x = \nu u_{xx}`

    Uses GLL nodal basis with diagonal mass matrix and explicit
    or implicit time stepping.

    Parameters:
        x_left: Left domain boundary.
        x_right: Right domain boundary.
        n_elements: Number of spectral elements.
        order: Polynomial order per element (N = order + 1 GLL points).
        nu: Diffusion coefficient.
        a: Advection speed.

    Example::

        solver = SEMSolver1D(x_left=0, x_right=1, n_elements=8, order=8, nu=0.01)
        state = solver.project(lambda x: np.sin(np.pi * x))
        for _ in range(1000):
            state = solver.step(state, dt=1e-5)
    """

    def __init__(
        self,
        x_left: float = 0.0,
        x_right: float = 1.0,
        n_elements: int = 8,
        order: int = 8,
        nu: float = 0.01,
        a: float = 0.0,
    ) -> None:
        self.x_left = x_left
        self.x_right = x_right
        self.n_elements = n_elements
        self.order = order
        self.N = order + 1  # GLL points per element
        self.nu = nu
        self.a = a

        # Reference element
        self.xi, self.w = gll_points(self.N)
        self.D = derivative_matrix(self.xi)

        # Mesh
        self.mesh = np.linspace(x_left, x_right, n_elements + 1, dtype=np.float64)
        self.dx = np.diff(self.mesh)

        # Total DOF (including shared nodes)
        self.n_global = n_elements * order + 1

        # Build global ↔ local connectivity
        self.loc2glob = np.zeros((n_elements, self.N), dtype=np.intp)
        for e in range(n_elements):
            self.loc2glob[e] = np.arange(e * order, e * order + self.N)

        # Physical coordinates at each GLL node
        self.x_nodes = np.zeros(self.n_global, dtype=np.float64)
        for e in range(n_elements):
            x_c = 0.5 * (self.mesh[e] + self.mesh[e + 1])
            h_e = self.dx[e]
            x_local = x_c + 0.5 * h_e * self.xi
            self.x_nodes[self.loc2glob[e]] = x_local

        self._step_count = 0

    # -----------------------------------------------------------------------
    # Assembly
    # -----------------------------------------------------------------------

    def _assemble_mass(self) -> NDArray:
        """Assemble global diagonal mass matrix (GLL lumping)."""
        M = np.zeros(self.n_global, dtype=np.float64)
        for e in range(self.n_elements):
            J_e = 0.5 * self.dx[e]  # Jacobian
            M[self.loc2glob[e]] += J_e * self.w
        return M

    def _assemble_stiffness(self) -> NDArray:
        """Assemble global stiffness matrix K."""
        K = np.zeros((self.n_global, self.n_global), dtype=np.float64)
        for e in range(self.n_elements):
            J_e = 0.5 * self.dx[e]
            # Local stiffness: K_e = (1/J) D^T W D
            W_diag = np.diag(self.w)
            K_local = (1.0 / J_e) * (self.D.T @ W_diag @ self.D)
            g = self.loc2glob[e]
            for i in range(self.N):
                for j in range(self.N):
                    K[g[i], g[j]] += K_local[i, j]
        return K

    def _assemble_advection(self) -> NDArray:
        """Assemble global advection matrix C = a ∫ φ_i φ'_j."""
        C = np.zeros((self.n_global, self.n_global), dtype=np.float64)
        for e in range(self.n_elements):
            # C_local = W D (in reference coords, Jacobians cancel)
            C_local = np.diag(self.w) @ self.D
            g = self.loc2glob[e]
            for i in range(self.N):
                for j in range(self.N):
                    C[g[i], g[j]] += C_local[i, j]
        return self.a * C

    # -----------------------------------------------------------------------
    # Projection
    # -----------------------------------------------------------------------

    def project(self, u0: Callable[[NDArray], NDArray]) -> SEMState:
        """L² project an initial condition onto the SEM space."""
        u_global = u0(self.x_nodes)
        u_local = np.zeros((self.n_elements, self.N), dtype=np.float64)
        for e in range(self.n_elements):
            u_local[e] = u_global[self.loc2glob[e]]
        return SEMState(u=u_local, mesh=self.mesh.copy())

    def to_global(self, state: SEMState) -> NDArray:
        """Scatter element-local values to global DOF vector."""
        u = np.zeros(self.n_global, dtype=np.float64)
        for e in range(self.n_elements):
            u[self.loc2glob[e]] = state.u[e]
        return u

    def from_global(self, u_global: NDArray) -> SEMState:
        """Gather global vector into element-local state."""
        u_local = np.zeros((self.n_elements, self.N), dtype=np.float64)
        for e in range(self.n_elements):
            u_local[e] = u_global[self.loc2glob[e]]
        return SEMState(u=u_local, mesh=self.mesh.copy())

    # -----------------------------------------------------------------------
    # Poisson solver
    # -----------------------------------------------------------------------

    def solve_poisson(
        self,
        f_func: Callable[[NDArray], NDArray],
        u_left: float = 0.0,
        u_right: float = 0.0,
    ) -> SEMState:
        """
        Solve the Poisson equation -u'' = f with Dirichlet BCs.

        Uses direct solve (dense linear system).
        """
        K = self._assemble_stiffness()
        M_diag = self._assemble_mass()
        rhs = M_diag * f_func(self.x_nodes)

        # Apply Dirichlet BCs
        K[0, :] = 0.0
        K[0, 0] = 1.0
        rhs[0] = u_left
        K[-1, :] = 0.0
        K[-1, -1] = 1.0
        rhs[-1] = u_right

        u_global = np.linalg.solve(K, rhs)
        return self.from_global(u_global)

    # -----------------------------------------------------------------------
    # Time stepping
    # -----------------------------------------------------------------------

    def step(
        self,
        state: SEMState,
        dt: float,
        bc_left: float = 0.0,
        bc_right: float = 0.0,
    ) -> SEMState:
        """
        Advance one time step (explicit Euler) for advection-diffusion.

        u_t = ν u_xx - a u_x
        """
        M_diag = self._assemble_mass()
        K = self._assemble_stiffness()
        C = self._assemble_advection()
        u = self.to_global(state)

        rhs = -self.nu * K @ u - C @ u
        u_new = u + dt * rhs / (M_diag + 1e-30)

        # Enforce BCs
        u_new[0] = bc_left
        u_new[-1] = bc_right

        self._step_count += 1
        return self.from_global(u_new)

    def step_rk4(
        self,
        state: SEMState,
        dt: float,
        bc_left: float = 0.0,
        bc_right: float = 0.0,
    ) -> SEMState:
        """RK4 time stepper for advection-diffusion."""
        M_inv = 1.0 / (self._assemble_mass() + 1e-30)
        K = self._assemble_stiffness()
        C = self._assemble_advection()

        def rhs_func(u: NDArray) -> NDArray:
            r = -self.nu * K @ u - C @ u
            r *= M_inv
            r[0] = 0.0
            r[-1] = 0.0
            return r

        u = self.to_global(state)
        k1 = rhs_func(u)
        k2 = rhs_func(u + 0.5 * dt * k1)
        k3 = rhs_func(u + 0.5 * dt * k2)
        k4 = rhs_func(u + dt * k3)
        u_new = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        u_new[0] = bc_left
        u_new[-1] = bc_right

        self._step_count += 1
        return self.from_global(u_new)

    def step_n(self, state: SEMState, n_steps: int, dt: float, **kwargs) -> SEMState:
        for _ in range(n_steps):
            state = self.step_rk4(state, dt, **kwargs)
        return state

    @property
    def steps(self) -> int:
        return self._step_count

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def l2_norm(self, state: SEMState) -> float:
        """L² norm via GLL quadrature."""
        total = 0.0
        for e in range(self.n_elements):
            J_e = 0.5 * self.dx[e]
            total += J_e * np.sum(self.w * state.u[e] ** 2)
        return float(np.sqrt(total))

    def l2_error(self, state: SEMState, exact: Callable[[NDArray], NDArray]) -> float:
        """L² error against an exact solution."""
        total = 0.0
        for e in range(self.n_elements):
            J_e = 0.5 * self.dx[e]
            x_e = 0.5 * (self.mesh[e] + self.mesh[e + 1])
            x_local = x_e + 0.5 * self.dx[e] * self.xi
            u_exact = exact(x_local)
            total += J_e * np.sum(self.w * (state.u[e] - u_exact) ** 2)
        return float(np.sqrt(total))
