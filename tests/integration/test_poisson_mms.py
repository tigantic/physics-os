"""
Method of Manufactured Solutions (MMS) for Poisson Equation
============================================================

Verifies the Poisson solver using MMS.

The Poisson equation:
    ∇²φ = f

arises from the projection step in incompressible flow:
    ∇²φ = ∇·u*

MMS Procedure:
1. Choose smooth manufactured solution φ_exact(x, y)
2. Compute source term f = ∇²φ_exact
3. Solve ∇²φ_numerical = f
4. Verify ||φ_numerical - φ_exact|| decreases at expected order

Reference:
    Roache, P.J. (2002). "Code Verification by the Method of Manufactured Solutions"

Constitution Compliance: Article IV.1 (Verification), SV-2 Requirement
Tags: [V&V] [MMS] [POISSON]
"""

import math
from dataclasses import dataclass
from typing import Tuple

import pytest
import torch

PI = math.pi


@dataclass
class PoissonMMSSolution:
    """
    Manufactured solution for 2D Poisson equation.

    Form (satisfies homogeneous Dirichlet BCs on [0,1]²):
        φ(x,y) = sin(n·π·x) · sin(m·π·y)

    Then:
        ∇²φ = -(n²+m²)π² · sin(n·π·x) · sin(m·π·y)
    """

    # Wave numbers (must be integers for Dirichlet BCs)
    n: int = 2
    m: int = 2

    # Amplitude
    A: float = 1.0

    def phi_exact(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Manufactured solution."""
        return self.A * torch.sin(self.n * PI * x) * torch.sin(self.m * PI * y)

    def source_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Source term f = ∇²φ = -(n² + m²)π²φ."""
        return -(self.n**2 + self.m**2) * PI**2 * self.phi_exact(x, y)

    def dphi_dx(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """∂φ/∂x for gradient verification."""
        return (
            self.A
            * self.n
            * PI
            * torch.cos(self.n * PI * x)
            * torch.sin(self.m * PI * y)
        )

    def dphi_dy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """∂φ/∂y for gradient verification."""
        return (
            self.A
            * self.m
            * PI
            * torch.sin(self.n * PI * x)
            * torch.cos(self.m * PI * y)
        )


class PoissonSolver:
    """
    2D Poisson solver using iterative Jacobi method.

    For MMS verification of the discretization.
    Production code uses TT-based ALS (tt_poisson.py).
    """

    def __init__(self, nx: int, ny: int, bc: str = "dirichlet"):
        self.nx = nx
        self.ny = ny
        self.bc = bc
        self.dx = 1.0 / (nx + 1)  # Grid spacing for Dirichlet
        self.dy = 1.0 / (ny + 1)

        # Interior grid points (exclude boundaries for Dirichlet)
        self.x = torch.linspace(self.dx, 1.0 - self.dx, nx)
        self.y = torch.linspace(self.dy, 1.0 - self.dy, ny)
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing="ij")

    def solve_jacobi(
        self, rhs: torch.Tensor, tol: float = 1e-8, max_iter: int = 10000
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Solve ∇²φ = f using Jacobi iteration.

        Returns:
            (solution, iterations, final_residual)
        """
        phi = torch.zeros_like(rhs)
        dx2 = self.dx**2
        dy2 = self.dy**2

        # Jacobi iteration coefficient
        denom = 2.0 / dx2 + 2.0 / dy2

        for iteration in range(max_iter):
            phi_old = phi.clone()

            # Update interior points
            phi_new = torch.zeros_like(phi)

            # Left neighbor (x-1)
            phi_new[1:, :] += phi_old[:-1, :] / dx2
            # Right neighbor (x+1)
            phi_new[:-1, :] += phi_old[1:, :] / dx2
            # Bottom neighbor (y-1)
            phi_new[:, 1:] += phi_old[:, :-1] / dy2
            # Top neighbor (y+1)
            phi_new[:, :-1] += phi_old[:, 1:] / dy2

            # Homogeneous Dirichlet: boundary values are 0 (already in phi_new)

            # Complete update
            phi = (phi_new - rhs) / denom

            # Check convergence
            residual = torch.abs(phi - phi_old).max().item()
            if residual < tol:
                return phi, iteration + 1, residual

        return phi, max_iter, residual

    def compute_laplacian(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute discrete Laplacian for verification."""
        laplacian = torch.zeros_like(phi)
        dx2 = self.dx**2
        dy2 = self.dy**2

        # Interior
        laplacian[1:-1, 1:-1] = (
            phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]
        ) / dx2 + (phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2]) / dy2

        # Boundaries (using ghost cells = 0 for Dirichlet)
        # Left boundary
        laplacian[0, 1:-1] = (phi[1, 1:-1] - 2 * phi[0, 1:-1]) / dx2 + (
            phi[0, 2:] - 2 * phi[0, 1:-1] + phi[0, :-2]
        ) / dy2
        # Right boundary
        laplacian[-1, 1:-1] = (-2 * phi[-1, 1:-1] + phi[-2, 1:-1]) / dx2 + (
            phi[-1, 2:] - 2 * phi[-1, 1:-1] + phi[-1, :-2]
        ) / dy2
        # etc. for corners...

        return laplacian

    def compute_error(self, phi: torch.Tensor, mms: PoissonMMSSolution) -> dict:
        """Compute L1, L2, Linf errors."""
        phi_exact = mms.phi_exact(self.X, self.Y)
        diff = phi - phi_exact

        return {
            "L1": torch.abs(diff).mean().item(),
            "L2": torch.sqrt((diff**2).mean()).item(),
            "Linf": torch.abs(diff).max().item(),
        }


# =============================================================================
# MMS VERIFICATION TESTS
# =============================================================================


@pytest.fixture
def poisson_mms():
    """Standard Poisson MMS solution."""
    return PoissonMMSSolution()


class TestPoissonMMS:
    """MMS tests for Poisson solver."""

    @pytest.mark.mms
    def test_source_correct(self, poisson_mms: PoissonMMSSolution):
        """Verify source term is consistent with Laplacian of exact solution."""
        x = torch.linspace(0.1, 0.9, 32)
        y = torch.linspace(0.1, 0.9, 32)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Numerically compute Laplacian
        dx = x[1] - x[0]
        phi = poisson_mms.phi_exact(X, Y)

        # Central difference Laplacian
        laplacian_num = (
            torch.roll(phi, -1, dims=0) - 2 * phi + torch.roll(phi, 1, dims=0)
        ) / dx**2 + (
            torch.roll(phi, -1, dims=1) - 2 * phi + torch.roll(phi, 1, dims=1)
        ) / dx**2

        # Analytical source
        source = poisson_mms.source_term(X, Y)

        # Should match on interior (avoiding boundaries)
        # Note: Discretization error scales with dx² for 5-point stencil
        diff = (laplacian_num[5:-5, 5:-5] - source[5:-5, 5:-5]).abs().max().item()
        assert diff < 0.5, f"Source term mismatch: {diff}"

    @pytest.mark.mms
    def test_solution_smooth(self, poisson_mms: PoissonMMSSolution):
        """Verify MMS solution is smooth and bounded."""
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        phi = poisson_mms.phi_exact(X, Y)

        assert (
            phi.abs().max() <= poisson_mms.A + 1e-6
        ), "Solution exceeds amplitude bound"
        # Check it satisfies BCs (approximately zero at boundaries)
        # Note: linspace doesn't hit exact boundary points, so use relaxed tolerance
        assert phi[0, :].abs().max() < 1e-6, "BC violated at x=0"
        assert phi[-1, :].abs().max() < 1e-6, "BC violated at x=1"
        assert phi[:, 0].abs().max() < 1e-6, "BC violated at y=0"
        assert phi[:, -1].abs().max() < 1e-6, "BC violated at y=1"

    @pytest.mark.mms
    @pytest.mark.benchmark
    def test_solver_converges(self, poisson_mms: PoissonMMSSolution):
        """Verify solver converges to expected solution."""
        solver = PoissonSolver(nx=32, ny=32)
        rhs = poisson_mms.source_term(solver.X, solver.Y)

        phi, iters, residual = solver.solve_jacobi(rhs)

        assert residual < 1e-6, f"Solver did not converge: residual={residual}"

        errors = solver.compute_error(phi, poisson_mms)
        assert errors["L2"] < 0.1, f"Solution error too large: {errors['L2']}"

    @pytest.mark.mms
    @pytest.mark.convergence
    def test_spatial_convergence_order(self, poisson_mms: PoissonMMSSolution):
        """Verify 2nd order spatial convergence."""
        grids = [16, 32, 64]
        errors = []

        for n in grids:
            solver = PoissonSolver(nx=n, ny=n)
            rhs = poisson_mms.source_term(solver.X, solver.Y)
            phi, _, _ = solver.solve_jacobi(rhs, tol=1e-10, max_iter=20000)
            err = solver.compute_error(phi, poisson_mms)
            errors.append(err["L2"])

        # Compute convergence rate
        rates = []
        for i in range(1, len(errors)):
            rate = math.log(errors[i - 1] / errors[i]) / math.log(2)
            rates.append(rate)

        avg_rate = sum(rates) / len(rates)

        # 5-point Laplacian stencil is 2nd order
        assert avg_rate > 1.8, f"Convergence rate {avg_rate:.2f} below 2nd order"

    @pytest.mark.mms
    def test_higher_modes(self):
        """Test with higher frequency manufactured solution."""
        mms = PoissonMMSSolution(n=4, m=4)
        solver = PoissonSolver(nx=64, ny=64)
        rhs = mms.source_term(solver.X, solver.Y)

        phi, iters, residual = solver.solve_jacobi(rhs, tol=1e-8, max_iter=50000)

        errors = solver.compute_error(phi, mms)
        assert errors["L2"] < 0.05, f"High-mode error too large: {errors['L2']}"


if __name__ == "__main__":
    mms = PoissonMMSSolution()

    print("Testing Poisson MMS...")
    print(f"  Mode: n={mms.n}, m={mms.m}")

    grids = [16, 32, 64]
    errors = []

    for n in grids:
        solver = PoissonSolver(nx=n, ny=n)
        rhs = mms.source_term(solver.X, solver.Y)
        phi, iters, _ = solver.solve_jacobi(rhs, tol=1e-10, max_iter=20000)
        err = solver.compute_error(phi, mms)
        errors.append(err["L2"])
        print(f"  {n}x{n}: L2 error = {err['L2']:.4e} ({iters} iterations)")

    rates = [
        math.log(errors[i - 1] / errors[i]) / math.log(2) for i in range(1, len(errors))
    ]
    print(f"\nConvergence rates: {rates}")
    print(f"Average order: {sum(rates)/len(rates):.2f}")
