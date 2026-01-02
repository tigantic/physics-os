"""
Method of Manufactured Solutions (MMS) for Advection-Diffusion Equation
========================================================================

Verifies the advection solver using MMS.

The advection-diffusion equation:
    ∂u/∂t + v·∇u = D∇²u

For pure advection (D=0):
    ∂u/∂t + v·∇u = 0

MMS Procedure:
1. Choose smooth manufactured solution u_exact(x, y, t)
2. Compute source term S = ∂u/∂t + v·∇u - D∇²u
3. Solve modified equation: ∂u/∂t + v·∇u - D∇²u = S
4. Verify ||u_numerical - u_exact|| decreases at expected order

Reference:
    Roache, P.J. (2002). "Code Verification by the Method of Manufactured Solutions"

Constitution Compliance: Article IV.1 (Verification), SV-2 Requirement
Tags: [V&V] [MMS] [ADVECTION]
"""

import math
from dataclasses import dataclass
from typing import Tuple

import pytest
import torch

PI = math.pi


@dataclass
class AdvectionMMSSolution:
    """
    Manufactured solution for 2D advection-diffusion.

    Form:
        u(x,y,t) = u₀ + ε·sin(kx·x + ky·y - ω·t)

    This is a traveling wave solution for pure advection.
    """

    # Base state
    u_0: float = 1.0

    # Wave parameters
    epsilon: float = 0.3
    kx: float = 2.0 * PI
    ky: float = 2.0 * PI
    omega: float = 2.0 * PI  # ω = |k|·|v| for exact advection

    # Velocity field (constant)
    vx: float = 0.5
    vy: float = 0.5

    # Diffusion coefficient
    D: float = 0.01

    def u_exact(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """Manufactured solution."""
        phase = self.kx * x + self.ky * y - self.omega * t
        return self.u_0 + self.epsilon * torch.sin(phase)

    def source_term(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """
        Source term S = ∂u/∂t + v·∇u - D∇²u

        Hand-derived:
            ∂u/∂t = -ω·ε·cos(phase)
            ∂u/∂x = kx·ε·cos(phase)
            ∂u/∂y = ky·ε·cos(phase)
            ∂²u/∂x² = -kx²·ε·sin(phase)
            ∂²u/∂y² = -ky²·ε·sin(phase)
        """
        phase = self.kx * x + self.ky * y - self.omega * t
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Time derivative
        du_dt = -self.omega * self.epsilon * cos_phase

        # Advection: v·∇u
        du_dx = self.kx * self.epsilon * cos_phase
        du_dy = self.ky * self.epsilon * cos_phase
        advection = self.vx * du_dx + self.vy * du_dy

        # Diffusion: D∇²u
        d2u_dx2 = -self.kx**2 * self.epsilon * sin_phase
        d2u_dy2 = -self.ky**2 * self.epsilon * sin_phase
        diffusion = self.D * (d2u_dx2 + d2u_dy2)

        # Source = ∂u/∂t + v·∇u - D∇²u
        return du_dt + advection - diffusion


class AdvectionDiffusionSolver:
    """
    Simple advection-diffusion solver for MMS verification.

    Uses upwind advection + central diffusion with RK4 time stepping.
    """

    def __init__(self, nx: int, ny: int, vx: float, vy: float, D: float):
        self.nx = nx
        self.ny = ny
        self.vx = vx
        self.vy = vy
        self.D = D
        self.dx = 1.0 / nx
        self.dy = 1.0 / ny

        # Grid coordinates
        self.x = torch.linspace(0.5 * self.dx, 1.0 - 0.5 * self.dx, nx)
        self.y = torch.linspace(0.5 * self.dy, 1.0 - 0.5 * self.dy, ny)
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing="ij")

    def advection_rhs(self, u: torch.Tensor) -> torch.Tensor:
        """Compute advection term: -v·∇u using upwind."""
        # Upwind differencing for advection
        if self.vx > 0:
            du_dx = (u - torch.roll(u, 1, dims=0)) / self.dx
        else:
            du_dx = (torch.roll(u, -1, dims=0) - u) / self.dx

        if self.vy > 0:
            du_dy = (u - torch.roll(u, 1, dims=1)) / self.dy
        else:
            du_dy = (torch.roll(u, -1, dims=1) - u) / self.dy

        return -(self.vx * du_dx + self.vy * du_dy)

    def diffusion_rhs(self, u: torch.Tensor) -> torch.Tensor:
        """Compute diffusion term: D∇²u using central differences."""
        # Central differences for Laplacian
        d2u_dx2 = (
            torch.roll(u, -1, dims=0) - 2 * u + torch.roll(u, 1, dims=0)
        ) / self.dx**2
        d2u_dy2 = (
            torch.roll(u, -1, dims=1) - 2 * u + torch.roll(u, 1, dims=1)
        ) / self.dy**2

        return self.D * (d2u_dx2 + d2u_dy2)

    def rhs(self, u: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Total RHS: -v·∇u + D∇²u + S."""
        return self.advection_rhs(u) + self.diffusion_rhs(u) + source

    def step_rk4(
        self, u: torch.Tensor, dt: float, mms: AdvectionMMSSolution, t: float
    ) -> torch.Tensor:
        """RK4 time step."""
        S = mms.source_term(self.X, self.Y, t)
        k1 = self.rhs(u, S)

        S = mms.source_term(self.X, self.Y, t + 0.5 * dt)
        k2 = self.rhs(u + 0.5 * dt * k1, S)
        k3 = self.rhs(u + 0.5 * dt * k2, S)

        S = mms.source_term(self.X, self.Y, t + dt)
        k4 = self.rhs(u + dt * k3, S)

        return u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(
        self, mms: AdvectionMMSSolution, t_final: float, cfl: float = 0.4
    ) -> torch.Tensor:
        """Integrate to t_final."""
        u = mms.u_exact(self.X, self.Y, 0.0)
        t = 0.0

        # Stable dt
        v_max = max(abs(self.vx), abs(self.vy))
        dt_adv = cfl * min(self.dx, self.dy) / v_max if v_max > 0 else float("inf")
        dt_diff = (
            cfl * min(self.dx, self.dy) ** 2 / (4 * self.D)
            if self.D > 0
            else float("inf")
        )
        dt = min(dt_adv, dt_diff)

        while t < t_final:
            if t + dt > t_final:
                dt = t_final - t
            u = self.step_rk4(u, dt, mms, t)
            t += dt

        return u

    def compute_error(
        self, u: torch.Tensor, mms: AdvectionMMSSolution, t: float
    ) -> dict:
        """Compute L1, L2, Linf errors."""
        u_exact = mms.u_exact(self.X, self.Y, t)
        diff = u - u_exact

        return {
            "L1": torch.abs(diff).mean().item(),
            "L2": torch.sqrt((diff**2).mean()).item(),
            "Linf": torch.abs(diff).max().item(),
        }


# =============================================================================
# MMS VERIFICATION TESTS
# =============================================================================


@pytest.fixture
def advection_mms():
    """Standard advection-diffusion MMS solution."""
    return AdvectionMMSSolution()


class TestAdvectionDiffusionMMS:
    """MMS tests for advection-diffusion solver."""

    @pytest.mark.mms
    def test_solution_smooth(self, advection_mms: AdvectionMMSSolution):
        """Verify MMS solution is smooth and bounded."""
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        for t in [0.0, 0.5, 1.0]:
            u = advection_mms.u_exact(X, Y, t)
            assert u.min() > 0, f"Solution negative at t={t}"
            assert u.max() < 2, f"Solution unbounded at t={t}"

    @pytest.mark.mms
    def test_source_nonzero(self, advection_mms: AdvectionMMSSolution):
        """Verify source term is non-trivial."""
        x = torch.linspace(0, 1, 32)
        y = torch.linspace(0, 1, 32)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        S = advection_mms.source_term(X, Y, 0.5)
        assert S.abs().max() > 1e-10, "Source term is trivially zero"

    @pytest.mark.mms
    @pytest.mark.benchmark
    def test_short_time_accuracy(self, advection_mms: AdvectionMMSSolution):
        """Verify solution matches exact at short times."""
        solver = AdvectionDiffusionSolver(
            nx=64, ny=64, vx=advection_mms.vx, vy=advection_mms.vy, D=advection_mms.D
        )
        t_final = 0.01

        u = solver.solve(advection_mms, t_final)
        errors = solver.compute_error(u, advection_mms, t_final)

        assert errors["L2"] < 1e-3, f"Short-time error too large: {errors['L2']}"

    @pytest.mark.mms
    @pytest.mark.convergence
    def test_spatial_convergence_order(self, advection_mms: AdvectionMMSSolution):
        """Verify 1st order spatial convergence (upwind)."""
        t_final = 0.05
        grids = [16, 32, 64]
        errors = []

        for n in grids:
            solver = AdvectionDiffusionSolver(
                nx=n, ny=n, vx=advection_mms.vx, vy=advection_mms.vy, D=advection_mms.D
            )
            u = solver.solve(advection_mms, t_final)
            err = solver.compute_error(u, advection_mms, t_final)
            errors.append(err["L2"])

        # Compute convergence rates
        rates = []
        for i in range(1, len(errors)):
            rate = math.log(errors[i - 1] / errors[i]) / math.log(2)
            rates.append(rate)

        avg_rate = sum(rates) / len(rates)

        # Upwind is 1st order for advection, central is 2nd for diffusion
        # Mixed scheme should show ~1st order dominated behavior
        assert avg_rate > 0.8, f"Convergence rate {avg_rate:.2f} is below threshold"

    @pytest.mark.mms
    @pytest.mark.convergence
    def test_pure_diffusion_convergence(self):
        """Test pure diffusion (v=0) for 2nd order convergence."""
        mms = AdvectionMMSSolution(vx=0.0, vy=0.0, D=0.1, omega=0.0)
        t_final = 0.02
        grids = [16, 32, 64]
        errors = []

        for n in grids:
            solver = AdvectionDiffusionSolver(nx=n, ny=n, vx=0.0, vy=0.0, D=0.1)
            u = solver.solve(mms, t_final)
            err = solver.compute_error(u, mms, t_final)
            errors.append(err["L2"])

        # Central differences for diffusion is 2nd order
        rate = math.log(errors[0] / errors[-1]) / math.log(len(grids))
        assert rate > 1.5, f"Diffusion convergence rate {rate:.2f} below 2nd order"


if __name__ == "__main__":
    mms = AdvectionMMSSolution()

    print("Testing Advection-Diffusion MMS...")
    print(f"  Velocity: ({mms.vx}, {mms.vy})")
    print(f"  Diffusion: D = {mms.D}")

    grids = [16, 32, 64]
    errors = []

    for n in grids:
        solver = AdvectionDiffusionSolver(nx=n, ny=n, vx=mms.vx, vy=mms.vy, D=mms.D)
        u = solver.solve(mms, 0.05)
        err = solver.compute_error(u, mms, 0.05)
        errors.append(err["L2"])
        print(f"  {n}x{n}: L2 error = {err['L2']:.4e}")

    rates = [
        math.log(errors[i - 1] / errors[i]) / math.log(2) for i in range(1, len(errors))
    ]
    print(f"\nConvergence rates: {rates}")
    print(f"Average order: {sum(rates)/len(rates):.2f}")
