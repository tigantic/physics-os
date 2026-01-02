"""
Method of Manufactured Solutions (MMS) for 2D Euler Equations
==============================================================

This module implements formal MMS verification for the 2D Euler solver.
MMS is the gold standard for verifying PDE discretizations (ASME V&V 10-2019).

Procedure:
1. CHOOSE a smooth analytical solution u_exact(x, y, t)
2. COMPUTE the source term f = ∂U/∂t + ∇·F(U) via symbolic differentiation
3. SOLVE with the code including the source term
4. MEASURE error ||u_numerical - u_exact||
5. REFINE mesh and verify error decreases at expected order

Reference:
    Roache, P.J. (2002). "Code Verification by the Method of Manufactured Solutions"
    ASME Journal of Fluids Engineering, 124(1), 4-10.

Constitution Compliance: Article IV.1 (Verification), SV-2 Requirement
Tags: [V&V] [MMS] [EULER-2D]
"""

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import pytest
import torch

# Manufactured solution parameters
PI = math.pi


@dataclass
class MMSSolution:
    """
    Manufactured solution for 2D Euler equations.

    The solution is chosen to be:
    - Smooth (infinitely differentiable)
    - Bounded away from unphysical values (ρ > 0, p > 0)
    - Satisfies periodic boundary conditions

    Form:
        ρ(x,y,t) = ρ₀ + ε_ρ · sin(kx·x) · sin(ky·y) · cos(ω·t)
        u(x,y,t) = u₀ + ε_u · cos(kx·x) · sin(ky·y) · cos(ω·t)
        v(x,y,t) = v₀ + ε_v · sin(kx·x) · cos(ky·y) · cos(ω·t)
        p(x,y,t) = p₀ + ε_p · sin(kx·x) · sin(ky·y) · cos(ω·t)
    """

    # Base state
    rho_0: float = 1.0
    u_0: float = 0.3
    v_0: float = 0.2
    p_0: float = 1.0

    # Perturbation amplitudes (small for linearization validity)
    eps_rho: float = 0.1
    eps_u: float = 0.1
    eps_v: float = 0.1
    eps_p: float = 0.1

    # Wave numbers
    kx: float = 2.0 * PI
    ky: float = 2.0 * PI
    omega: float = 1.0

    # Gas properties
    gamma: float = 1.4

    def rho(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """Manufactured density field."""
        return self.rho_0 + self.eps_rho * torch.sin(self.kx * x) * torch.sin(
            self.ky * y
        ) * math.cos(self.omega * t)

    def u(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """Manufactured x-velocity field."""
        return self.u_0 + self.eps_u * torch.cos(self.kx * x) * torch.sin(
            self.ky * y
        ) * math.cos(self.omega * t)

    def v(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """Manufactured y-velocity field."""
        return self.v_0 + self.eps_v * torch.sin(self.kx * x) * torch.cos(
            self.ky * y
        ) * math.cos(self.omega * t)

    def p(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """Manufactured pressure field."""
        return self.p_0 + self.eps_p * torch.sin(self.kx * x) * torch.sin(
            self.ky * y
        ) * math.cos(self.omega * t)

    def E(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """Total energy from equation of state."""
        rho = self.rho(x, y, t)
        u = self.u(x, y, t)
        v = self.v(x, y, t)
        p = self.p(x, y, t)
        return p / (self.gamma - 1) + 0.5 * rho * (u**2 + v**2)

    def source_continuity(
        self, x: torch.Tensor, y: torch.Tensor, t: float
    ) -> torch.Tensor:
        """
        Source term for continuity equation: S_ρ = ∂ρ/∂t + ∂(ρu)/∂x + ∂(ρv)/∂y

        Hand-derived symbolically for verification.
        """
        kx, ky, omega = self.kx, self.ky, self.omega
        cos_t = math.cos(omega * t)
        sin_t = math.sin(omega * t)

        sin_kx = torch.sin(kx * x)
        cos_kx = torch.cos(kx * x)
        sin_ky = torch.sin(ky * y)
        cos_ky = torch.cos(ky * y)

        # ∂ρ/∂t = -ω·ε_ρ·sin(kx·x)·sin(ky·y)·sin(ω·t)
        drho_dt = -omega * self.eps_rho * sin_kx * sin_ky * sin_t

        # ρu = (ρ₀ + ε_ρ·sin·sin·cos_t)(u₀ + ε_u·cos·sin·cos_t)
        # ∂(ρu)/∂x requires product rule expansion
        rho = self.rho(x, y, t)
        u = self.u(x, y, t)
        v = self.v(x, y, t)

        # Derivative terms
        drho_dx = self.eps_rho * kx * cos_kx * sin_ky * cos_t
        drho_dy = self.eps_rho * ky * sin_kx * cos_ky * cos_t
        du_dx = -self.eps_u * kx * sin_kx * sin_ky * cos_t
        dv_dy = -self.eps_v * ky * sin_kx * sin_ky * cos_t

        # ∂(ρu)/∂x = u·∂ρ/∂x + ρ·∂u/∂x
        d_rhou_dx = u * drho_dx + rho * du_dx

        # ∂(ρv)/∂y = v·∂ρ/∂y + ρ·∂v/∂y
        d_rhov_dy = v * drho_dy + rho * dv_dy

        return drho_dt + d_rhou_dx + d_rhov_dy

    def source_x_momentum(
        self, x: torch.Tensor, y: torch.Tensor, t: float
    ) -> torch.Tensor:
        """
        Source term for x-momentum: S_ρu = ∂(ρu)/∂t + ∂(ρu² + p)/∂x + ∂(ρuv)/∂y
        """
        kx, ky, omega = self.kx, self.ky, self.omega
        cos_t = math.cos(omega * t)
        sin_t = math.sin(omega * t)

        sin_kx = torch.sin(kx * x)
        cos_kx = torch.cos(kx * x)
        sin_ky = torch.sin(ky * y)
        cos_ky = torch.cos(ky * y)

        rho = self.rho(x, y, t)
        u = self.u(x, y, t)
        v = self.v(x, y, t)
        p = self.p(x, y, t)

        # Time derivatives
        drho_dt = -omega * self.eps_rho * sin_kx * sin_ky * sin_t
        du_dt = -omega * self.eps_u * cos_kx * sin_ky * sin_t

        # ∂(ρu)/∂t = u·∂ρ/∂t + ρ·∂u/∂t
        d_rhou_dt = u * drho_dt + rho * du_dt

        # Spatial derivatives
        drho_dx = self.eps_rho * kx * cos_kx * sin_ky * cos_t
        du_dx = -self.eps_u * kx * sin_kx * sin_ky * cos_t
        dp_dx = self.eps_p * kx * cos_kx * sin_ky * cos_t

        drho_dy = self.eps_rho * ky * sin_kx * cos_ky * cos_t
        du_dy = self.eps_u * ky * cos_kx * cos_ky * cos_t
        dv_dy = -self.eps_v * ky * sin_kx * sin_ky * cos_t

        # ∂(ρu² + p)/∂x = 2ρu·∂u/∂x + u²·∂ρ/∂x + ∂p/∂x
        d_flux_x = 2 * rho * u * du_dx + u**2 * drho_dx + dp_dx

        # ∂(ρuv)/∂y = ρu·∂v/∂y + ρv·∂u/∂y + uv·∂ρ/∂y
        d_flux_y = rho * u * dv_dy + rho * v * du_dy + u * v * drho_dy

        return d_rhou_dt + d_flux_x + d_flux_y

    def source_y_momentum(
        self, x: torch.Tensor, y: torch.Tensor, t: float
    ) -> torch.Tensor:
        """
        Source term for y-momentum: S_ρv = ∂(ρv)/∂t + ∂(ρuv)/∂x + ∂(ρv² + p)/∂y
        """
        kx, ky, omega = self.kx, self.ky, self.omega
        cos_t = math.cos(omega * t)
        sin_t = math.sin(omega * t)

        sin_kx = torch.sin(kx * x)
        cos_kx = torch.cos(kx * x)
        sin_ky = torch.sin(ky * y)
        cos_ky = torch.cos(ky * y)

        rho = self.rho(x, y, t)
        u = self.u(x, y, t)
        v = self.v(x, y, t)
        p = self.p(x, y, t)

        # Time derivatives
        drho_dt = -omega * self.eps_rho * sin_kx * sin_ky * sin_t
        dv_dt = -omega * self.eps_v * sin_kx * cos_ky * sin_t

        # ∂(ρv)/∂t = v·∂ρ/∂t + ρ·∂v/∂t
        d_rhov_dt = v * drho_dt + rho * dv_dt

        # Spatial derivatives
        drho_dx = self.eps_rho * kx * cos_kx * sin_ky * cos_t
        du_dx = -self.eps_u * kx * sin_kx * sin_ky * cos_t
        dv_dx = self.eps_v * kx * cos_kx * cos_ky * cos_t

        drho_dy = self.eps_rho * ky * sin_kx * cos_ky * cos_t
        dv_dy = -self.eps_v * ky * sin_kx * sin_ky * cos_t
        dp_dy = self.eps_p * ky * sin_kx * cos_ky * cos_t

        # ∂(ρuv)/∂x = ρv·∂u/∂x + ρu·∂v/∂x + uv·∂ρ/∂x
        d_flux_x = rho * v * du_dx + rho * u * dv_dx + u * v * drho_dx

        # ∂(ρv² + p)/∂y = 2ρv·∂v/∂y + v²·∂ρ/∂y + ∂p/∂y
        d_flux_y = 2 * rho * v * dv_dy + v**2 * drho_dy + dp_dy

        return d_rhov_dt + d_flux_x + d_flux_y

    def source_energy(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """
        Source term for energy: S_E = ∂E/∂t + ∂((E+p)u)/∂x + ∂((E+p)v)/∂y
        """
        kx, ky, omega = self.kx, self.ky, self.omega
        gamma = self.gamma
        cos_t = math.cos(omega * t)
        sin_t = math.sin(omega * t)

        sin_kx = torch.sin(kx * x)
        cos_kx = torch.cos(kx * x)
        sin_ky = torch.sin(ky * y)
        cos_ky = torch.cos(ky * y)

        rho = self.rho(x, y, t)
        u = self.u(x, y, t)
        v = self.v(x, y, t)
        p = self.p(x, y, t)
        E = self.E(x, y, t)

        # Time derivatives of primitives
        drho_dt = -omega * self.eps_rho * sin_kx * sin_ky * sin_t
        du_dt = -omega * self.eps_u * cos_kx * sin_ky * sin_t
        dv_dt = -omega * self.eps_v * sin_kx * cos_ky * sin_t
        dp_dt = -omega * self.eps_p * sin_kx * sin_ky * sin_t

        # ∂E/∂t = ∂p/∂t / (γ-1) + 0.5·(∂ρ/∂t·(u²+v²) + 2ρ·(u·∂u/∂t + v·∂v/∂t))
        dE_dt = dp_dt / (gamma - 1) + 0.5 * (
            drho_dt * (u**2 + v**2) + 2 * rho * (u * du_dt + v * dv_dt)
        )

        # Spatial derivatives
        drho_dx = self.eps_rho * kx * cos_kx * sin_ky * cos_t
        du_dx = -self.eps_u * kx * sin_kx * sin_ky * cos_t
        dv_dx = self.eps_v * kx * cos_kx * cos_ky * cos_t
        dp_dx = self.eps_p * kx * cos_kx * sin_ky * cos_t

        drho_dy = self.eps_rho * ky * sin_kx * cos_ky * cos_t
        du_dy = self.eps_u * ky * cos_kx * cos_ky * cos_t
        dv_dy = -self.eps_v * ky * sin_kx * sin_ky * cos_t
        dp_dy = self.eps_p * ky * sin_kx * cos_ky * cos_t

        # ∂E/∂x
        dE_dx = dp_dx / (gamma - 1) + 0.5 * (
            drho_dx * (u**2 + v**2) + 2 * rho * (u * du_dx + v * dv_dx)
        )

        # ∂E/∂y
        dE_dy = dp_dy / (gamma - 1) + 0.5 * (
            drho_dy * (u**2 + v**2) + 2 * rho * (u * du_dy + v * dv_dy)
        )

        # H = E + p
        H = E + p

        # ∂(Hu)/∂x = H·∂u/∂x + u·∂H/∂x = H·∂u/∂x + u·(∂E/∂x + ∂p/∂x)
        d_flux_x = H * du_dx + u * (dE_dx + dp_dx)

        # ∂(Hv)/∂y = H·∂v/∂y + v·∂H/∂y = H·∂v/∂y + v·(∂E/∂y + ∂p/∂y)
        d_flux_y = H * dv_dy + v * (dE_dy + dp_dy)

        return dE_dt + d_flux_x + d_flux_y


class Euler2DMMS:
    """
    2D Euler solver with MMS source term injection.

    Uses simple finite volume discretization for clarity.
    Production solvers (HLLC, etc.) can be verified similarly.
    """

    def __init__(self, nx: int, ny: int, gamma: float = 1.4):
        self.nx = nx
        self.ny = ny
        self.gamma = gamma
        self.dx = 1.0 / nx
        self.dy = 1.0 / ny

        # Grid coordinates (cell centers)
        self.x = torch.linspace(0.5 * self.dx, 1.0 - 0.5 * self.dx, nx)
        self.y = torch.linspace(0.5 * self.dy, 1.0 - 0.5 * self.dy, ny)
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing="ij")

    def initialize(self, mms: MMSSolution, t: float = 0.0) -> torch.Tensor:
        """Initialize conserved variables from MMS solution."""
        rho = mms.rho(self.X, self.Y, t)
        u = mms.u(self.X, self.Y, t)
        v = mms.v(self.X, self.Y, t)
        E = mms.E(self.X, self.Y, t)

        # Conserved: [rho, rho*u, rho*v, E]
        U = torch.zeros(4, self.nx, self.ny)
        U[0] = rho
        U[1] = rho * u
        U[2] = rho * v
        U[3] = E
        return U

    def flux_x(self, U: torch.Tensor) -> torch.Tensor:
        """Compute x-direction flux."""
        rho = U[0]
        rho_u = U[1]
        rho_v = U[2]
        E = U[3]

        u = rho_u / rho
        p = (self.gamma - 1) * (E - 0.5 * (rho_u**2 + rho_v**2) / rho)

        F = torch.zeros_like(U)
        F[0] = rho_u
        F[1] = rho_u * u + p
        F[2] = rho_v * u
        F[3] = (E + p) * u
        return F

    def flux_y(self, U: torch.Tensor) -> torch.Tensor:
        """Compute y-direction flux."""
        rho = U[0]
        rho_u = U[1]
        rho_v = U[2]
        E = U[3]

        v = rho_v / rho
        p = (self.gamma - 1) * (E - 0.5 * (rho_u**2 + rho_v**2) / rho)

        G = torch.zeros_like(U)
        G[0] = rho_v
        G[1] = rho_u * v
        G[2] = rho_v * v + p
        G[3] = (E + p) * v
        return G

    def compute_source(self, mms: MMSSolution, t: float) -> torch.Tensor:
        """Compute MMS source terms."""
        S = torch.zeros(4, self.nx, self.ny)
        S[0] = mms.source_continuity(self.X, self.Y, t)
        S[1] = mms.source_x_momentum(self.X, self.Y, t)
        S[2] = mms.source_y_momentum(self.X, self.Y, t)
        S[3] = mms.source_energy(self.X, self.Y, t)
        return S

    def rhs(self, U: torch.Tensor, mms: MMSSolution, t: float) -> torch.Tensor:
        """Compute RHS = -∂F/∂x - ∂G/∂y + S."""
        # Central differences for fluxes (2nd order)
        Fx = self.flux_x(U)
        Gy = self.flux_y(U)

        # Periodic BCs
        dFdx = (torch.roll(Fx, -1, dims=1) - torch.roll(Fx, 1, dims=1)) / (2 * self.dx)
        dGdy = (torch.roll(Gy, -1, dims=2) - torch.roll(Gy, 1, dims=2)) / (2 * self.dy)

        # MMS source term (crucial for verification)
        S = self.compute_source(mms, t)

        return -dFdx - dGdy + S

    def step_rk4(
        self, U: torch.Tensor, dt: float, mms: MMSSolution, t: float
    ) -> torch.Tensor:
        """RK4 time integration."""
        k1 = self.rhs(U, mms, t)
        k2 = self.rhs(U + 0.5 * dt * k1, mms, t + 0.5 * dt)
        k3 = self.rhs(U + 0.5 * dt * k2, mms, t + 0.5 * dt)
        k4 = self.rhs(U + dt * k3, mms, t + dt)
        return U + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(self, mms: MMSSolution, t_final: float, cfl: float = 0.4) -> torch.Tensor:
        """Integrate to t_final."""
        U = self.initialize(mms, 0.0)
        t = 0.0

        # Estimate stable dt
        u_max = max(abs(mms.u_0) + mms.eps_u, abs(mms.v_0) + mms.eps_v)
        a_max = math.sqrt(mms.gamma * (mms.p_0 + mms.eps_p) / (mms.rho_0 - mms.eps_rho))
        dt = cfl * min(self.dx, self.dy) / (u_max + a_max)

        while t < t_final:
            if t + dt > t_final:
                dt = t_final - t
            U = self.step_rk4(U, dt, mms, t)
            t += dt

        return U

    def compute_error(self, U: torch.Tensor, mms: MMSSolution, t: float) -> dict:
        """Compute L1, L2, Linf errors vs exact solution."""
        U_exact = self.initialize(mms, t)
        diff = U - U_exact

        # Norms per variable
        vars = ["rho", "rho_u", "rho_v", "E"]
        errors = {}

        for i, var in enumerate(vars):
            errors[f"L1_{var}"] = torch.abs(diff[i]).mean().item()
            errors[f"L2_{var}"] = torch.sqrt((diff[i] ** 2).mean()).item()
            errors[f"Linf_{var}"] = torch.abs(diff[i]).max().item()

        # Combined L2 norm
        errors["L2_total"] = torch.sqrt((diff**2).mean()).item()

        return errors


# =============================================================================
# MMS VERIFICATION TESTS
# =============================================================================


@pytest.fixture
def mms_solution():
    """Standard MMS solution."""
    return MMSSolution()


class TestEuler2DMMS:
    """
    Method of Manufactured Solutions tests for 2D Euler.

    These tests verify the spatial discretization order of accuracy
    by running grid refinement studies and checking convergence rates.
    """

    @pytest.mark.mms
    @pytest.mark.benchmark
    def test_mms_solution_positive(self, mms_solution: MMSSolution):
        """Verify MMS solution maintains positive density and pressure."""
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        for t in [0.0, 0.5, 1.0, 2.0]:
            rho = mms_solution.rho(X, Y, t)
            p = mms_solution.p(X, Y, t)

            assert rho.min() > 0, f"Negative density at t={t}"
            assert p.min() > 0, f"Negative pressure at t={t}"

    @pytest.mark.mms
    def test_source_term_nonzero(self, mms_solution: MMSSolution):
        """Verify source terms are non-trivial (actually testing something)."""
        x = torch.linspace(0, 1, 32)
        y = torch.linspace(0, 1, 32)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        S_rho = mms_solution.source_continuity(X, Y, 0.5)
        S_rhou = mms_solution.source_x_momentum(X, Y, 0.5)
        S_rhov = mms_solution.source_y_momentum(X, Y, 0.5)
        S_E = mms_solution.source_energy(X, Y, 0.5)

        # Source terms should be non-zero for non-trivial test
        assert S_rho.abs().max() > 1e-10, "Continuity source is trivially zero"
        assert S_rhou.abs().max() > 1e-10, "X-momentum source is trivially zero"
        assert S_rhov.abs().max() > 1e-10, "Y-momentum source is trivially zero"
        assert S_E.abs().max() > 1e-10, "Energy source is trivially zero"

    @pytest.mark.mms
    @pytest.mark.benchmark
    def test_short_time_accuracy(self, mms_solution: MMSSolution):
        """Verify solution matches exact solution at short times."""
        solver = Euler2DMMS(nx=64, ny=64, gamma=mms_solution.gamma)
        t_final = 0.01  # Very short time

        U = solver.solve(mms_solution, t_final)
        errors = solver.compute_error(U, mms_solution, t_final)

        # At short times, error should be small
        assert (
            errors["L2_total"] < 1e-3
        ), f"Short-time error too large: {errors['L2_total']}"

    @pytest.mark.mms
    @pytest.mark.convergence
    def test_spatial_convergence_order(self, mms_solution: MMSSolution):
        """
        Verify 2nd order spatial convergence.

        This is the key MMS test: run at multiple resolutions and verify
        the error decreases at the expected rate (h² for 2nd order).
        """
        t_final = 0.05  # Short time to minimize temporal error
        grids = [16, 32, 64]
        errors = []

        for n in grids:
            solver = Euler2DMMS(nx=n, ny=n, gamma=mms_solution.gamma)
            U = solver.solve(mms_solution, t_final)
            err = solver.compute_error(U, mms_solution, t_final)
            errors.append(err["L2_total"])

        # Compute convergence rates
        rates = []
        for i in range(1, len(errors)):
            rate = math.log(errors[i - 1] / errors[i]) / math.log(2)
            rates.append(rate)

        # Average rate should be ~2.0 for 2nd order scheme
        avg_rate = sum(rates) / len(rates)

        assert (
            avg_rate > 1.5
        ), f"Convergence rate {avg_rate:.2f} is below 2nd order threshold"
        assert avg_rate < 3.0, f"Convergence rate {avg_rate:.2f} is suspiciously high"

    @pytest.mark.mms
    @pytest.mark.convergence
    def test_density_convergence(self, mms_solution: MMSSolution):
        """Verify density field converges at 2nd order."""
        t_final = 0.05
        grids = [16, 32, 64]
        errors = []

        for n in grids:
            solver = Euler2DMMS(nx=n, ny=n, gamma=mms_solution.gamma)
            U = solver.solve(mms_solution, t_final)
            err = solver.compute_error(U, mms_solution, t_final)
            errors.append(err["L2_rho"])

        # Convergence rate for density
        rate = math.log(errors[0] / errors[-1]) / math.log(len(grids))

        assert rate > 1.5, f"Density convergence rate {rate:.2f} is below threshold"

    @pytest.mark.mms
    @pytest.mark.convergence
    def test_energy_convergence(self, mms_solution: MMSSolution):
        """Verify energy field converges at 2nd order."""
        t_final = 0.05
        grids = [16, 32, 64]
        errors = []

        for n in grids:
            solver = Euler2DMMS(nx=n, ny=n, gamma=mms_solution.gamma)
            U = solver.solve(mms_solution, t_final)
            err = solver.compute_error(U, mms_solution, t_final)
            errors.append(err["L2_E"])

        # Convergence rate for energy
        rate = math.log(errors[0] / errors[-1]) / math.log(len(grids))

        assert rate > 1.5, f"Energy convergence rate {rate:.2f} is below threshold"

    @pytest.mark.mms
    @pytest.mark.slow
    def test_high_resolution_convergence(self, mms_solution: MMSSolution):
        """
        Extended convergence study with finer grids.

        Only run in full V&V suite (marked slow).
        """
        t_final = 0.02
        grids = [16, 32, 64, 128]
        errors = []

        for n in grids:
            solver = Euler2DMMS(nx=n, ny=n, gamma=mms_solution.gamma)
            U = solver.solve(mms_solution, t_final)
            err = solver.compute_error(U, mms_solution, t_final)
            errors.append(err["L2_total"])

        # Compute all convergence rates
        rates = []
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║       MMS CONVERGENCE STUDY: 2D EULER EQUATIONS              ║")
        print("╠═════════╤═══════════╤═══════════╤═══════════╤═══════════════╣")
        print("║  Grid   │    Δx     │   L2 Err  │   Ratio   │   Order (p)   ║")
        print("╠═════════╪═══════════╪═══════════╪═══════════╪═══════════════╣")

        for i, n in enumerate(grids):
            dx = 1.0 / n
            if i == 0:
                print(
                    f"║  {n:3d}    │ {dx:.2e} │ {errors[i]:.2e} │    —      │      —        ║"
                )
            else:
                ratio = errors[i - 1] / errors[i]
                order = math.log(ratio) / math.log(2)
                rates.append(order)
                print(
                    f"║  {n:3d}    │ {dx:.2e} │ {errors[i]:.2e} │   {ratio:.2f}    │     {order:.2f}       ║"
                )

        avg_rate = sum(rates) / len(rates)
        print("╠═════════╧═══════════╧═══════════╧═══════════╧═══════════════╣")
        print(
            f"║  Average Convergence Order: {avg_rate:.2f} (expected: 2.00)           ║"
        )
        print(
            f"║  RESULT: {'✅ PASS' if avg_rate > 1.8 else '❌ FAIL'}                                          ║"
        )
        print("╚══════════════════════════════════════════════════════════════╝")

        assert (
            avg_rate > 1.8
        ), f"Average convergence rate {avg_rate:.2f} is below 2nd order"


if __name__ == "__main__":
    # Quick standalone test
    mms = MMSSolution()

    print("Testing MMS solution positivity...")
    x = torch.linspace(0, 1, 64)
    y = torch.linspace(0, 1, 64)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    for t in [0.0, 0.5, 1.0]:
        rho = mms.rho(X, Y, t)
        p = mms.p(X, Y, t)
        print(
            f"  t={t}: rho ∈ [{rho.min():.3f}, {rho.max():.3f}], p ∈ [{p.min():.3f}, {p.max():.3f}]"
        )

    print("\nRunning convergence study...")
    grids = [16, 32, 64]
    errors = []

    for n in grids:
        solver = Euler2DMMS(nx=n, ny=n, gamma=mms.gamma)
        U = solver.solve(mms, 0.05)
        err = solver.compute_error(U, mms, 0.05)
        errors.append(err["L2_total"])
        print(f"  {n}x{n}: L2 error = {err['L2_total']:.4e}")

    rates = [
        math.log(errors[i - 1] / errors[i]) / math.log(2) for i in range(1, len(errors))
    ]
    print(f"\nConvergence rates: {rates}")
    print(f"Average order: {sum(rates)/len(rates):.2f}")
