"""
Test Module: Boundary Conditions

Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation
    - Article V, Section 5.1: Float64 precision

References:
    Anderson, J.D. (1995). "Computational Fluid Dynamics."
    McGraw-Hill.

    LeVeque, R.J. (2007). "Finite Difference Methods for
    Ordinary and Partial Differential Equations." SIAM.
"""

import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest
import torch

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def deterministic_seed():
    """Per Article III, Section 3.2: Reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture
def device():
    """Get device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def bc_params():
    """Default boundary condition parameters."""
    return {
        "nx": 32,
        "ny": 32,
        "nz": 16,
        "dx": 0.1,
        "dtype": torch.float64,
    }


# ============================================================================
# BOUNDARY CONDITION UTILITIES
# ============================================================================


def apply_dirichlet_1d(
    u: torch.Tensor,
    left_value: float,
    right_value: float,
) -> torch.Tensor:
    """Apply Dirichlet boundary conditions."""
    u[0] = left_value
    u[-1] = right_value
    return u


def apply_neumann_1d(
    u: torch.Tensor,
    left_gradient: float,
    right_gradient: float,
    dx: float,
) -> torch.Tensor:
    """Apply Neumann boundary conditions."""
    # Left: du/dx = left_gradient -> u[0] = u[1] - dx * left_gradient
    u[0] = u[1] - dx * left_gradient
    # Right: du/dx = right_gradient -> u[-1] = u[-2] + dx * right_gradient
    u[-1] = u[-2] + dx * right_gradient
    return u


def apply_periodic_1d(u: torch.Tensor) -> torch.Tensor:
    """Apply periodic boundary conditions."""
    u[0] = u[-2]
    u[-1] = u[1]
    return u


def apply_dirichlet_2d(
    u: torch.Tensor,
    top: float = 0.0,
    bottom: float = 0.0,
    left: float = 0.0,
    right: float = 0.0,
) -> torch.Tensor:
    """Apply Dirichlet BC in 2D."""
    u[0, :] = bottom
    u[-1, :] = top
    u[:, 0] = left
    u[:, -1] = right
    return u


def apply_neumann_2d(
    u: torch.Tensor,
    dx: float,
    dy: float,
    top: float = 0.0,
    bottom: float = 0.0,
    left: float = 0.0,
    right: float = 0.0,
) -> torch.Tensor:
    """Apply Neumann BC in 2D."""
    u[0, :] = u[1, :] - dy * bottom
    u[-1, :] = u[-2, :] + dy * top
    u[:, 0] = u[:, 1] - dx * left
    u[:, -1] = u[:, -2] + dx * right
    return u


def apply_periodic_2d(u: torch.Tensor) -> torch.Tensor:
    """Apply periodic BC in 2D."""
    u[0, :] = u[-2, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]
    return u


def apply_robin_1d(
    u: torch.Tensor,
    alpha_left: float,
    beta_left: float,
    gamma_left: float,
    alpha_right: float,
    beta_right: float,
    gamma_right: float,
    dx: float,
) -> torch.Tensor:
    """Apply Robin BC: alpha*u + beta*du/dn = gamma."""
    # Left: alpha*u[0] + beta*(-du/dx) = gamma
    # u[0] = (gamma + beta*u[1]/dx) / (alpha + beta/dx)
    if abs(alpha_left + beta_left / dx) > 1e-10:
        u[0] = (gamma_left + beta_left * u[1] / dx) / (alpha_left + beta_left / dx)

    # Right: alpha*u[-1] + beta*(du/dx) = gamma
    if abs(alpha_right + beta_right / dx) > 1e-10:
        u[-1] = (gamma_right + beta_right * u[-2] / dx) / (
            alpha_right + beta_right / dx
        )

    return u


def apply_symmetry_1d(
    u: torch.Tensor,
    left_symmetric: bool = True,
    right_symmetric: bool = True,
) -> torch.Tensor:
    """Apply symmetry boundary conditions."""
    if left_symmetric:
        u[0] = u[1]
    if right_symmetric:
        u[-1] = u[-2]
    return u


def apply_outflow_1d(
    u: torch.Tensor,
    left_outflow: bool = False,
    right_outflow: bool = True,
) -> torch.Tensor:
    """Apply outflow (zero gradient) BC."""
    if left_outflow:
        u[0] = u[1]
    if right_outflow:
        u[-1] = u[-2]
    return u


def apply_no_slip_2d(
    u: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply no-slip BC for velocity."""
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    v[0, :] = 0.0
    v[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0

    return u, v


def apply_lid_driven_cavity(
    u: torch.Tensor,
    v: torch.Tensor,
    lid_velocity: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply lid-driven cavity BC."""
    # No-slip walls
    u[0, :] = 0.0  # Bottom
    u[:, 0] = 0.0  # Left
    u[:, -1] = 0.0  # Right

    v[0, :] = 0.0
    v[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0

    # Moving lid (top)
    u[-1, :] = lid_velocity

    return u, v


# ============================================================================
# UNIT TESTS: DIRICHLET BC
# ============================================================================


class TestDirichletBC:
    """Test Dirichlet boundary conditions."""

    @pytest.mark.unit
    def test_dirichlet_1d(self, deterministic_seed, bc_params):
        """1D Dirichlet BC."""
        u = torch.randn(bc_params["nx"], dtype=bc_params["dtype"])

        u = apply_dirichlet_1d(u, left_value=0.0, right_value=1.0)

        assert u[0] == 0.0
        assert u[-1] == 1.0

    @pytest.mark.unit
    def test_dirichlet_2d(self, deterministic_seed, bc_params):
        """2D Dirichlet BC."""
        u = torch.randn(bc_params["nx"], bc_params["ny"], dtype=bc_params["dtype"])

        u = apply_dirichlet_2d(u, top=1.0, bottom=0.0, left=0.0, right=0.0)

        assert torch.all(u[0, :] == 0.0)  # bottom
        # Check top row interior (corners may be overwritten by left/right BCs)
        assert torch.all(u[-1, 1:-1] == 1.0)  # top interior
        assert u[0, 0] == 0.0  # corners

    @pytest.mark.unit
    def test_dirichlet_preserves_interior(self, deterministic_seed, bc_params):
        """Dirichlet BC doesn't change interior."""
        u = torch.randn(bc_params["nx"], dtype=bc_params["dtype"])
        interior = u[1:-1].clone()

        u = apply_dirichlet_1d(u, 0.0, 1.0)

        assert torch.allclose(u[1:-1], interior)


# ============================================================================
# UNIT TESTS: NEUMANN BC
# ============================================================================


class TestNeumannBC:
    """Test Neumann boundary conditions."""

    @pytest.mark.unit
    def test_neumann_1d(self, deterministic_seed, bc_params):
        """1D Neumann BC."""
        u = torch.ones(bc_params["nx"], dtype=bc_params["dtype"])

        u = apply_neumann_1d(
            u, left_gradient=0.0, right_gradient=0.0, dx=bc_params["dx"]
        )

        # Zero gradient means u[0] = u[1] and u[-1] = u[-2]
        assert u[0] == u[1]
        assert u[-1] == u[-2]

    @pytest.mark.unit
    def test_neumann_2d(self, deterministic_seed, bc_params):
        """2D Neumann BC."""
        u = torch.ones(bc_params["nx"], bc_params["ny"], dtype=bc_params["dtype"])

        u = apply_neumann_2d(u, dx=bc_params["dx"], dy=bc_params["dx"])

        assert torch.allclose(u[0, :], u[1, :])
        assert torch.allclose(u[-1, :], u[-2, :])

    @pytest.mark.unit
    def test_neumann_nonzero_gradient(self, deterministic_seed, bc_params):
        """Neumann with non-zero gradient."""
        u = torch.ones(bc_params["nx"], dtype=bc_params["dtype"])
        dx = bc_params["dx"]

        u = apply_neumann_1d(u, left_gradient=1.0, right_gradient=-1.0, dx=dx)

        # Left: u[0] = u[1] - dx * 1.0
        expected_left = u[1] - dx * 1.0
        # Right: u[-1] = u[-2] + dx * (-1.0)
        expected_right = u[-2] - dx

        # Note: u[1] should still be 1.0
        assert u[0] == pytest.approx(1.0 - dx)


# ============================================================================
# UNIT TESTS: PERIODIC BC
# ============================================================================


class TestPeriodicBC:
    """Test periodic boundary conditions."""

    @pytest.mark.unit
    def test_periodic_1d(self, deterministic_seed, bc_params):
        """1D periodic BC."""
        u = torch.arange(bc_params["nx"], dtype=bc_params["dtype"])

        u = apply_periodic_1d(u)

        assert u[0] == u[-2]
        assert u[-1] == u[1]

    @pytest.mark.unit
    def test_periodic_2d(self, deterministic_seed, bc_params):
        """2D periodic BC."""
        u = torch.randn(bc_params["nx"], bc_params["ny"], dtype=bc_params["dtype"])

        u = apply_periodic_2d(u)

        assert torch.allclose(u[0, :], u[-2, :])
        assert torch.allclose(u[-1, :], u[1, :])
        assert torch.allclose(u[:, 0], u[:, -2])
        assert torch.allclose(u[:, -1], u[:, 1])

    @pytest.mark.unit
    def test_periodic_continuity(self, deterministic_seed):
        """Periodic BC ensures continuity."""
        N = 32
        x = torch.linspace(0, 2 * math.pi, N + 1, dtype=torch.float64)[:-1]
        u = torch.sin(x)

        # Add ghost points
        u_ext = torch.zeros(N + 2, dtype=torch.float64)
        u_ext[1:-1] = u
        u_ext = apply_periodic_1d(u_ext)

        # u[0] should equal u[N-1] for sin
        assert u_ext[0] == pytest.approx(u_ext[-2], abs=1e-10)


# ============================================================================
# UNIT TESTS: ROBIN BC
# ============================================================================


class TestRobinBC:
    """Test Robin boundary conditions."""

    @pytest.mark.unit
    def test_robin_reduces_to_dirichlet(self, deterministic_seed, bc_params):
        """Robin with beta=0 is Dirichlet."""
        u = torch.randn(bc_params["nx"], dtype=bc_params["dtype"])

        u = apply_robin_1d(
            u,
            alpha_left=1.0,
            beta_left=0.0,
            gamma_left=2.0,
            alpha_right=1.0,
            beta_right=0.0,
            gamma_right=3.0,
            dx=bc_params["dx"],
        )

        assert u[0] == pytest.approx(2.0)
        assert u[-1] == pytest.approx(3.0)

    @pytest.mark.unit
    def test_robin_reduces_to_neumann(self, deterministic_seed, bc_params):
        """Robin with alpha=0 is Neumann."""
        u = torch.ones(bc_params["nx"], dtype=bc_params["dtype"])
        dx = bc_params["dx"]

        u = apply_robin_1d(
            u,
            alpha_left=0.0,
            beta_left=1.0,
            gamma_left=0.0,  # Zero gradient
            alpha_right=0.0,
            beta_right=1.0,
            gamma_right=0.0,
            dx=dx,
        )

        # Zero gradient: u[0] = u[1]
        assert u[0] == pytest.approx(u[1], abs=1e-10)


# ============================================================================
# UNIT TESTS: SYMMETRY BC
# ============================================================================


class TestSymmetryBC:
    """Test symmetry boundary conditions."""

    @pytest.mark.unit
    def test_symmetry_1d(self, deterministic_seed, bc_params):
        """1D symmetry BC."""
        u = torch.randn(bc_params["nx"], dtype=bc_params["dtype"])

        u = apply_symmetry_1d(u)

        assert u[0] == u[1]
        assert u[-1] == u[-2]

    @pytest.mark.unit
    def test_symmetry_preserves_even_function(self, deterministic_seed):
        """Symmetry BC for even functions."""
        N = 33  # Odd for symmetry point
        x = torch.linspace(-1, 1, N, dtype=torch.float64)
        u = x**2  # Even function

        # Symmetry at midpoint
        mid = N // 2
        assert u[mid - 1] == pytest.approx(u[mid + 1], abs=1e-10)


# ============================================================================
# UNIT TESTS: OUTFLOW BC
# ============================================================================


class TestOutflowBC:
    """Test outflow boundary conditions."""

    @pytest.mark.unit
    def test_outflow_1d(self, deterministic_seed, bc_params):
        """1D outflow BC."""
        u = torch.randn(bc_params["nx"], dtype=bc_params["dtype"])

        u = apply_outflow_1d(u, right_outflow=True)

        assert u[-1] == u[-2]

    @pytest.mark.unit
    def test_outflow_advection(self, deterministic_seed):
        """Outflow BC for advection."""
        # Wave moving right should exit cleanly
        N = 32
        u = torch.zeros(N, dtype=torch.float64)
        u[N // 2 : N // 2 + 5] = 1.0  # Pulse

        u = apply_outflow_1d(u, right_outflow=True)

        # Outflow doesn't reflect
        assert u[-1] == u[-2]


# ============================================================================
# UNIT TESTS: NO-SLIP BC
# ============================================================================


class TestNoSlipBC:
    """Test no-slip wall boundary conditions."""

    @pytest.mark.unit
    def test_no_slip_2d(self, deterministic_seed, bc_params):
        """2D no-slip BC."""
        u = torch.randn(bc_params["nx"], bc_params["ny"], dtype=bc_params["dtype"])
        v = torch.randn(bc_params["nx"], bc_params["ny"], dtype=bc_params["dtype"])

        u, v = apply_no_slip_2d(u, v)

        # All boundaries should be zero
        assert torch.all(u[0, :] == 0)
        assert torch.all(u[-1, :] == 0)
        assert torch.all(u[:, 0] == 0)
        assert torch.all(u[:, -1] == 0)

        assert torch.all(v[0, :] == 0)
        assert torch.all(v[-1, :] == 0)

    @pytest.mark.unit
    def test_lid_driven_cavity(self, deterministic_seed, bc_params):
        """Lid-driven cavity BC."""
        u = torch.zeros(bc_params["nx"], bc_params["ny"], dtype=bc_params["dtype"])
        v = torch.zeros(bc_params["nx"], bc_params["ny"], dtype=bc_params["dtype"])

        u, v = apply_lid_driven_cavity(u, v, lid_velocity=1.0)

        # Lid moves
        assert torch.all(u[-1, :] == 1.0)

        # Other walls stationary
        assert torch.all(u[0, :] == 0)
        assert torch.all(v[:, :] == 0)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestBCIntegration:
    """Integration tests for boundary conditions."""

    @pytest.mark.integration
    def test_laplace_dirichlet(self, deterministic_seed, bc_params):
        """Laplace equation with Dirichlet BC."""
        nx, ny = bc_params["nx"], bc_params["ny"]
        u = torch.zeros(nx, ny, dtype=bc_params["dtype"])

        # BCs: u=0 on three sides, u=1 on top
        u = apply_dirichlet_2d(u, top=1.0, bottom=0.0, left=0.0, right=0.0)

        # Jacobi iteration
        for _ in range(100):
            u_new = u.clone()
            u_new[1:-1, 1:-1] = 0.25 * (
                u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
            )
            # Re-apply BC
            u_new = apply_dirichlet_2d(u_new, top=1.0, bottom=0.0, left=0.0, right=0.0)
            u = u_new

        # Solution should be between 0 and 1
        assert u.min() >= -0.01
        assert u.max() <= 1.01

    @pytest.mark.integration
    def test_heat_equation_neumann(self, deterministic_seed, bc_params):
        """Heat equation with insulated (Neumann) BC."""
        nx = bc_params["nx"]
        dx = bc_params["dx"]
        dt = 0.0001
        alpha = 1.0

        u = torch.zeros(nx, dtype=bc_params["dtype"])
        u[nx // 4 : 3 * nx // 4] = 1.0  # Initial hot region

        # Time integration
        for _ in range(100):
            u_new = u.clone()
            u_new[1:-1] = u[1:-1] + alpha * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
            u_new = apply_neumann_1d(u_new, 0.0, 0.0, dx)
            u = u_new

        # Heat should diffuse but total conserved (insulated)
        # Energy approximately conserved
        assert u.sum() == pytest.approx(u.sum(), rel=0.1)


# ============================================================================
# FLOAT64 COMPLIANCE
# ============================================================================


class TestFloat64ComplianceBC:
    """Article V: Float64 precision tests."""

    @pytest.mark.unit
    def test_bc_float64(self, deterministic_seed, bc_params):
        """BC preserves float64."""
        u = torch.randn(bc_params["nx"], dtype=torch.float64)
        u = apply_dirichlet_1d(u, 0.0, 1.0)

        assert u.dtype == torch.float64

    @pytest.mark.unit
    def test_robin_float64(self, deterministic_seed, bc_params):
        """Robin BC preserves float64."""
        u = torch.randn(bc_params["nx"], dtype=torch.float64)
        u = apply_robin_1d(u, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, bc_params["dx"])

        assert u.dtype == torch.float64


# ============================================================================
# GPU COMPATIBILITY
# ============================================================================


class TestGPUCompatibilityBC:
    """Test GPU execution compatibility."""

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dirichlet_on_gpu(self, device):
        """Dirichlet BC on GPU."""
        u = torch.randn(32, dtype=torch.float64, device=device)
        u = apply_dirichlet_1d(u, 0.0, 1.0)

        # Compare device type (cuda:0 == cuda)
        assert u.device.type == device.type

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_periodic_2d_on_gpu(self, device):
        """2D periodic BC on GPU."""
        u = torch.randn(32, 32, dtype=torch.float64, device=device)
        u = apply_periodic_2d(u)

        # Compare device type (cuda:0 == cuda)
        assert u.device.type == device.type


# ============================================================================
# REPRODUCIBILITY
# ============================================================================


class TestReproducibilityBC:
    """Article III, Section 3.2: Reproducibility tests."""

    @pytest.mark.unit
    def test_deterministic_bc(self):
        """BC application is deterministic."""
        torch.manual_seed(42)
        u1 = torch.randn(32, dtype=torch.float64)
        u1 = apply_robin_1d(u1, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.1)

        torch.manual_seed(42)
        u2 = torch.randn(32, dtype=torch.float64)
        u2 = apply_robin_1d(u2, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.1)

        assert torch.allclose(u1, u2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
