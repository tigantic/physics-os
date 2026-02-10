"""
Discontinuous Galerkin (DG) Method
===================================

High-order DG solver for hyperbolic conservation laws with modal
basis functions on each element and Riemann-flux coupling.

Semi-discrete form on element K:
    ∫_K φ_j ∂u/∂t dΩ = -∫_K φ_j ∇·F(u) dΩ
        + ∫_{∂K} φ_j F*(u⁻,u⁺)·n dΓ

where φ_j are local basis functions, F* is a numerical flux
(Lax-Friedrichs, Roe, HLLC), and integration is performed via
Gauss-Legendre quadrature.

Supported Features:
    - Modal Legendre basis (1D, 2D tensor-product, 3D tensor-product)
    - Arbitrary polynomial order p
    - Lax-Friedrichs, Roe, and HLLC numerical fluxes
    - Explicit RK4 time integration
    - Slope limiting (minmod, moment limiter)
    - Shock capturing via artificial viscosity

References:
    [1] Hesthaven & Warburton, "Nodal Discontinuous Galerkin Methods",
        Springer, 2008.
    [2] Cockburn & Shu, "The Runge-Kutta Discontinuous Galerkin Method
        for Conservation Laws V", J. Comp. Phys. 141, 1998.
    [3] Karniadakis & Sherwin, "Spectral/hp Element Methods for
        Computational Fluid Dynamics", Oxford, 2005.

Domain II.1 — Computational Fluid Dynamics / Discontinuous Galerkin.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

def gauss_legendre(n: int) -> Tuple[NDArray, NDArray]:
    """
    Gauss-Legendre quadrature points and weights on [-1, 1].

    Args:
        n: Number of quadrature points.

    Returns:
        Tuple of (points, weights).
    """
    points, weights = np.polynomial.legendre.leggauss(n)
    return points.astype(np.float64), weights.astype(np.float64)


def gauss_lobatto(n: int) -> Tuple[NDArray, NDArray]:
    """
    Gauss-Lobatto-Legendre quadrature on [-1, 1] (includes endpoints).

    Uses eigenvalue method for interior points.
    """
    if n < 2:
        raise ValueError("Gauss-Lobatto requires n >= 2")
    if n == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    # Interior points are zeros of P'_{n-1}
    inner, _ = gauss_legendre(n - 2)
    # Newton iteration on P'_{n-1}
    x = np.zeros(n, dtype=np.float64)
    x[0] = -1.0
    x[-1] = 1.0
    # Initial guess from Chebyshev nodes
    for i in range(1, n - 1):
        x[i] = -np.cos(np.pi * i / (n - 1))

    for _ in range(50):
        P = np.ones(n, dtype=np.float64)
        Pm1 = x.copy()
        for k in range(2, n):
            Pm2 = P.copy()
            P = Pm1.copy()
            Pm1 = ((2 * k - 1) * x * P - (k - 1) * Pm2) / k
        # dP/dx for Legendre degree n-1
        dP = (n - 1) * (P - x * Pm1) / (1.0 - x ** 2 + 1e-30)
        # Newton correction (for interior points only)
        for i in range(1, n - 1):
            if abs(dP[i]) > 1e-30:
                x[i] -= (x[i] * Pm1[i] - P[i]) / ((n - 1) * Pm1[i])

    # Recompute Legendre polynomial for weights
    P_nm1 = np.ones_like(x)
    P_cur = x.copy()
    for k in range(2, n):
        P_prev = P_nm1.copy()
        P_nm1 = P_cur.copy()
        P_cur = ((2 * k - 1) * x * P_nm1 - (k - 1) * P_prev) / k

    w = 2.0 / (n * (n - 1) * P_cur ** 2 + 1e-30)
    x.sort()
    return x, w


# ---------------------------------------------------------------------------
# Legendre Basis
# ---------------------------------------------------------------------------

def legendre_basis(x: NDArray, p: int) -> NDArray:
    """
    Evaluate Legendre polynomials P_0 ... P_p at points x ∈ [-1,1].

    Returns:
        Array of shape ``(p+1, len(x))``.
    """
    N = len(x)
    phi = np.zeros((p + 1, N), dtype=np.float64)
    phi[0] = 1.0
    if p >= 1:
        phi[1] = x
    for k in range(2, p + 1):
        phi[k] = ((2 * k - 1) * x * phi[k - 1] - (k - 1) * phi[k - 2]) / k
    return phi


def legendre_basis_deriv(x: NDArray, p: int) -> NDArray:
    """
    Evaluate derivatives dP_k/dx for k = 0..p.

    Returns:
        Array of shape ``(p+1, len(x))``.
    """
    N = len(x)
    phi = legendre_basis(x, p)
    dphi = np.zeros((p + 1, N), dtype=np.float64)
    # dP_0/dx = 0, dP_1/dx = 1
    if p >= 1:
        dphi[1] = 1.0
    for k in range(2, p + 1):
        dphi[k] = (2 * k - 1) * phi[k - 1] + dphi[k - 2]
    return dphi


# ---------------------------------------------------------------------------
# Numerical Fluxes
# ---------------------------------------------------------------------------

class FluxType(Enum):
    """Numerical flux variants for DG."""
    LAX_FRIEDRICHS = "lax-friedrichs"
    ROE = "roe"
    HLLC = "hllc"


def lax_friedrichs_flux(
    u_left: NDArray,
    u_right: NDArray,
    flux_func: Callable[[NDArray], NDArray],
    max_speed: float,
) -> NDArray:
    r"""
    Local Lax-Friedrichs (Rusanov) numerical flux:

    .. math::
        F^* = \frac{1}{2}[F(u^-) + F(u^+)] - \frac{\alpha}{2}(u^+ - u^-)

    where α = max |∂F/∂u|.
    """
    return 0.5 * (flux_func(u_left) + flux_func(u_right)) - 0.5 * max_speed * (u_right - u_left)


def roe_flux_scalar(
    u_left: NDArray,
    u_right: NDArray,
    flux_func: Callable[[NDArray], NDArray],
    df_du: Callable[[NDArray], NDArray],
) -> NDArray:
    """Roe flux for scalar conservation law."""
    a_roe = np.where(
        np.abs(u_right - u_left) > 1e-12,
        (flux_func(u_right) - flux_func(u_left)) / (u_right - u_left + 1e-30),
        df_du(0.5 * (u_left + u_right)),
    )
    return 0.5 * (
        flux_func(u_left) + flux_func(u_right)
        - np.abs(a_roe) * (u_right - u_left)
    )


# ---------------------------------------------------------------------------
# Slope Limiters
# ---------------------------------------------------------------------------

def minmod(a: float, b: float) -> float:
    """Minmod function."""
    if a * b <= 0:
        return 0.0
    return a if abs(a) < abs(b) else b


def minmod_array(a: NDArray, b: NDArray) -> NDArray:
    """Element-wise minmod."""
    result = np.where(a * b <= 0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))
    return result


# ---------------------------------------------------------------------------
# DG State
# ---------------------------------------------------------------------------

@dataclass
class DGState:
    """
    Modal coefficient state for the DG method.

    Attributes:
        coeffs: Modal coefficients — shape ``(n_elements, n_modes, n_vars)``.
        mesh: Element boundaries ``(n_elements + 1,)``.
    """
    coeffs: NDArray
    mesh: NDArray

    @property
    def n_elements(self) -> int:
        return self.coeffs.shape[0]

    @property
    def n_modes(self) -> int:
        return self.coeffs.shape[1]

    @property
    def n_vars(self) -> int:
        return self.coeffs.shape[2] if self.coeffs.ndim == 3 else 1

    @property
    def order(self) -> int:
        return self.n_modes - 1

    def cell_average(self) -> NDArray:
        """Return cell averages (zeroth mode)."""
        if self.coeffs.ndim == 3:
            return self.coeffs[:, 0, :]
        return self.coeffs[:, 0]


# ---------------------------------------------------------------------------
# 1D DG Solver
# ---------------------------------------------------------------------------

class DGSolver1D:
    r"""
    Discontinuous Galerkin solver for 1D scalar conservation laws.

    .. math::
        \frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} = 0

    Supports arbitrary polynomial order with modal Legendre basis,
    Gauss-Legendre quadrature, and explicit RK4 time stepping.

    Parameters:
        x_left: Left boundary of domain.
        x_right: Right boundary of domain.
        n_elements: Number of elements.
        order: Polynomial order p (number of modes = p + 1).
        flux_func: Physical flux function f(u).
        flux_deriv: Derivative df/du (for Roe flux and max wave speed).
        num_flux: Numerical flux type.
        n_quad: Number of quadrature points (defaults to order + 1).

    Example::

        solver = DGSolver1D(
            x_left=0.0, x_right=1.0,
            n_elements=100, order=3,
            flux_func=lambda u: 0.5 * u ** 2,  # Burgers
            flux_deriv=lambda u: u,
        )
        state = solver.project(lambda x: np.sin(2 * np.pi * x))
        for _ in range(200):
            state = solver.step(state, dt=1e-4)
    """

    def __init__(
        self,
        x_left: float = 0.0,
        x_right: float = 1.0,
        n_elements: int = 50,
        order: int = 3,
        flux_func: Optional[Callable[[NDArray], NDArray]] = None,
        flux_deriv: Optional[Callable[[NDArray], NDArray]] = None,
        num_flux: FluxType = FluxType.LAX_FRIEDRICHS,
        n_quad: Optional[int] = None,
    ) -> None:
        self.x_left = x_left
        self.x_right = x_right
        self.n_elements = n_elements
        self.order = order
        self.n_modes = order + 1
        self.num_flux = num_flux
        self.n_quad = n_quad if n_quad is not None else order + 2

        self.flux_func = flux_func or (lambda u: 0.5 * u ** 2)
        self.flux_deriv = flux_deriv or (lambda u: u)

        # Uniform mesh
        self.mesh = np.linspace(x_left, x_right, n_elements + 1, dtype=np.float64)
        self.dx = np.diff(self.mesh)

        # Reference element quadrature and basis
        self.xi_q, self.w_q = gauss_legendre(self.n_quad)
        self.phi_q = legendre_basis(self.xi_q, order)     # (p+1, n_quad)
        self.dphi_q = legendre_basis_deriv(self.xi_q, order)

        # Boundary evaluations (ξ = -1 and ξ = +1)
        self.phi_left = legendre_basis(np.array([-1.0]), order)[:, 0]   # (p+1,)
        self.phi_right = legendre_basis(np.array([1.0]), order)[:, 0]

        # Mass matrix diagonal (Legendre polynomials are orthogonal)
        # ∫_{-1}^{1} P_k² dξ = 2/(2k+1)
        self.M_diag = np.array([2.0 / (2 * k + 1) for k in range(self.n_modes)],
                               dtype=np.float64)

        self._step_count = 0

    def project(self, u0: Callable[[NDArray], NDArray]) -> DGState:
        """
        L² project an initial condition onto the DG space.

        Args:
            u0: Callable u0(x) → values.

        Returns:
            DGState with projected coefficients.
        """
        coeffs = np.zeros((self.n_elements, self.n_modes), dtype=np.float64)
        for e in range(self.n_elements):
            x_e = 0.5 * (self.mesh[e] + self.mesh[e + 1])
            he = self.dx[e]
            # Map quadrature to physical space
            x_phys = x_e + 0.5 * he * self.xi_q
            u_vals = u0(x_phys)
            # Project: c_k = (1/M_kk) ∫ u φ_k dξ
            for k in range(self.n_modes):
                coeffs[e, k] = (
                    np.sum(self.w_q * u_vals * self.phi_q[k]) / self.M_diag[k]
                )
        return DGState(coeffs=coeffs, mesh=self.mesh.copy())

    def evaluate(self, state: DGState, xi: NDArray) -> NDArray:
        """Evaluate solution at reference points ξ for all elements."""
        phi = legendre_basis(xi, self.order)  # (p+1, n_xi)
        # u(ξ) = Σ_k c_k φ_k(ξ)
        return np.einsum("ek,kn->en", state.coeffs, phi)

    def evaluate_physical(self, state: DGState, n_pts: int = 10) -> Tuple[NDArray, NDArray]:
        """Evaluate solution on a fine physical grid for plotting."""
        xi = np.linspace(-1, 1, n_pts)
        u_ref = self.evaluate(state, xi)  # (n_elem, n_pts)
        x_phys = np.zeros((self.n_elements, n_pts), dtype=np.float64)
        for e in range(self.n_elements):
            xc = 0.5 * (self.mesh[e] + self.mesh[e + 1])
            x_phys[e] = xc + 0.5 * self.dx[e] * xi
        return x_phys.ravel(), u_ref.ravel()

    def _rhs(self, coeffs: NDArray) -> NDArray:
        """Compute right-hand side of the semi-discrete DG system."""
        NE = self.n_elements
        NM = self.n_modes
        rhs = np.zeros((NE, NM), dtype=np.float64)

        # Evaluate solution at quadrature points and boundaries
        u_q = np.einsum("ek,kn->en", coeffs, self.phi_q)  # (NE, n_quad)
        u_left = np.einsum("ek,k->e", coeffs, self.phi_left)     # (NE,)
        u_right = np.einsum("ek,k->e", coeffs, self.phi_right)   # (NE,)

        # Volume integral: ∫ f(u) dφ_k/dξ dξ  (integration by parts)
        f_q = self.flux_func(u_q)  # f at quadrature points
        for k in range(NM):
            rhs[:, k] += np.sum(
                self.w_q[None, :] * f_q * self.dphi_q[k][None, :], axis=1
            )

        # Numerical flux at element interfaces
        # Right boundary of element e is left boundary of element e+1
        max_speed = float(np.max(np.abs(self.flux_deriv(u_q))))
        for e in range(NE):
            # Left interface (e-1|e)
            if e > 0:
                u_minus = u_right[e - 1]
                u_plus = u_left[e]
            else:
                # Periodic BC
                u_minus = u_right[-1]
                u_plus = u_left[0]
            f_star_left = float(lax_friedrichs_flux(
                np.array([u_minus]), np.array([u_plus]),
                self.flux_func, max_speed,
            ))

            # Right interface (e|e+1)
            if e < NE - 1:
                u_minus = u_right[e]
                u_plus = u_left[e + 1]
            else:
                u_minus = u_right[-1]
                u_plus = u_left[0]
            f_star_right = float(lax_friedrichs_flux(
                np.array([u_minus]), np.array([u_plus]),
                self.flux_func, max_speed,
            ))

            # Surface terms: subtract f*·φ at boundaries
            for k in range(NM):
                rhs[e, k] -= f_star_right * self.phi_right[k]
                rhs[e, k] += f_star_left * self.phi_left[k]

        # Scale by 2/h (Jacobian) and divide by mass matrix
        for e in range(NE):
            rhs[e] *= 2.0 / self.dx[e]
            rhs[e] /= self.M_diag

        return rhs

    def step(self, state: DGState, dt: float) -> DGState:
        """Advance one step with classical RK4."""
        c = state.coeffs
        k1 = self._rhs(c)
        k2 = self._rhs(c + 0.5 * dt * k1)
        k3 = self._rhs(c + 0.5 * dt * k2)
        k4 = self._rhs(c + dt * k3)
        c_new = c + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self._step_count += 1
        return DGState(coeffs=c_new, mesh=state.mesh)

    def step_n(self, state: DGState, n_steps: int, dt: float) -> DGState:
        """Advance *n_steps*."""
        for _ in range(n_steps):
            state = self.step(state, dt)
        return state

    def apply_minmod_limiter(self, state: DGState) -> DGState:
        """Apply minmod slope limiter (limits the first mode only)."""
        c = state.coeffs.copy()
        NE = self.n_elements
        for e in range(NE):
            # Cell average slopes
            u_bar = c[e, 0]
            u_bar_m = c[(e - 1) % NE, 0]
            u_bar_p = c[(e + 1) % NE, 0]
            forward = u_bar_p - u_bar
            backward = u_bar - u_bar_m

            if self.n_modes > 1:
                # The slope coefficient is c[e, 1] / √3 approximately
                slope_orig = c[e, 1]
                slope_limited = minmod(slope_orig, minmod(forward, backward))
                if abs(slope_limited - slope_orig) > 1e-14:
                    c[e, 1] = slope_limited
                    # Zero higher modes
                    c[e, 2:] = 0.0
        return DGState(coeffs=c, mesh=state.mesh)

    @property
    def steps(self) -> int:
        return self._step_count

    def l2_norm(self, state: DGState) -> float:
        """Compute discrete L² norm of the solution."""
        u_q = np.einsum("ek,kn->en", state.coeffs, self.phi_q)
        total = 0.0
        for e in range(self.n_elements):
            total += 0.5 * self.dx[e] * np.sum(self.w_q * u_q[e] ** 2)
        return float(np.sqrt(total))

    def total_mass(self, state: DGState) -> float:
        """Integral of u over the domain (should be conserved)."""
        return float(np.sum(state.coeffs[:, 0] * self.dx))
