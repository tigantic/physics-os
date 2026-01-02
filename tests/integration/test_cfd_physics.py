"""
Integration tests for CFD solvers - verifying physical bounds and conservation.

These tests ensure that Riemann solvers and Euler equation solvers produce
physically valid solutions that respect fundamental conservation laws.

Physics Verified:
    1. Positivity of density (ρ > 0)
    2. Positivity of pressure (p > 0)
    3. Conservation of mass, momentum, and energy
    4. Rankine-Hugoniot jump conditions at shocks
"""

import math

import pytest
import torch

# Mark entire module with V&V taxonomy markers
# Per HYPERTENSOR_VV_FRAMEWORK.md Section 7.1
pytestmark = [
    pytest.mark.integration,
    pytest.mark.physics,
    pytest.mark.benchmark,  # Tier 1 benchmarks (Sod, Lax, etc.)
]

from tensornet.cfd.godunov import (conserved_to_primitive, euler_flux,
                                   exact_riemann, hll_flux, hllc_flux,
                                   primitive_to_conserved, roe_flux)

# ============================================================================
# Test Data: Standard Riemann Problems (Toro's Test Suite)
# ============================================================================

# Sod shock tube (Test 1)
SOD_LEFT = {"rho": 1.0, "u": 0.0, "p": 1.0}
SOD_RIGHT = {"rho": 0.125, "u": 0.0, "p": 0.1}

# Lax shock tube (Test 2) - stronger shock
LAX_LEFT = {"rho": 0.445, "u": 0.698, "p": 3.528}
LAX_RIGHT = {"rho": 0.5, "u": 0.0, "p": 0.571}

# Double rarefaction (Test 3) - tests vacuum handling
DOUBLE_RARE_LEFT = {"rho": 1.0, "u": -2.0, "p": 0.4}
DOUBLE_RARE_RIGHT = {"rho": 1.0, "u": 2.0, "p": 0.4}

# Strong shock (Test 4)
STRONG_LEFT = {"rho": 5.99924, "u": 19.5975, "p": 460.894}
STRONG_RIGHT = {"rho": 5.99242, "u": -6.19633, "p": 46.0950}

# Stationary contact (Test 5)
STATIONARY_LEFT = {"rho": 1.0, "u": -19.59745, "p": 1000.0}
STATIONARY_RIGHT = {"rho": 1.0, "u": -19.59745, "p": 0.01}


class TestExactRiemannPhysicalBounds:
    """Test that exact Riemann solver produces physically valid solutions."""

    @pytest.fixture
    def x_grid(self):
        """Standard spatial grid for testing."""
        return torch.linspace(-0.5, 0.5, 201)

    def test_sod_positive_density(self, x_grid):
        """Verify ρ > 0 everywhere for Sod shock tube."""
        rho, u, p = exact_riemann(
            rho_L=SOD_LEFT["rho"],
            u_L=SOD_LEFT["u"],
            p_L=SOD_LEFT["p"],
            rho_R=SOD_RIGHT["rho"],
            u_R=SOD_RIGHT["u"],
            p_R=SOD_RIGHT["p"],
            x=x_grid,
            t=0.2,
            gamma=1.4,
        )
        assert torch.all(rho > 0), f"Negative density found: min(ρ) = {rho.min()}"

    def test_sod_positive_pressure(self, x_grid):
        """Verify p > 0 everywhere for Sod shock tube."""
        rho, u, p = exact_riemann(
            rho_L=SOD_LEFT["rho"],
            u_L=SOD_LEFT["u"],
            p_L=SOD_LEFT["p"],
            rho_R=SOD_RIGHT["rho"],
            u_R=SOD_RIGHT["u"],
            p_R=SOD_RIGHT["p"],
            x=x_grid,
            t=0.2,
            gamma=1.4,
        )
        assert torch.all(p > 0), f"Negative pressure found: min(p) = {p.min()}"

    def test_lax_positive_density(self, x_grid):
        """Verify ρ > 0 for Lax shock tube (stronger shock)."""
        rho, u, p = exact_riemann(
            rho_L=LAX_LEFT["rho"],
            u_L=LAX_LEFT["u"],
            p_L=LAX_LEFT["p"],
            rho_R=LAX_RIGHT["rho"],
            u_R=LAX_RIGHT["u"],
            p_R=LAX_RIGHT["p"],
            x=x_grid,
            t=0.14,
            gamma=1.4,
        )
        assert torch.all(rho > 0), f"Negative density found: min(ρ) = {rho.min()}"

    def test_lax_positive_pressure(self, x_grid):
        """Verify p > 0 for Lax shock tube."""
        rho, u, p = exact_riemann(
            rho_L=LAX_LEFT["rho"],
            u_L=LAX_LEFT["u"],
            p_L=LAX_LEFT["p"],
            rho_R=LAX_RIGHT["rho"],
            u_R=LAX_RIGHT["u"],
            p_R=LAX_RIGHT["p"],
            x=x_grid,
            t=0.14,
            gamma=1.4,
        )
        assert torch.all(p > 0), f"Negative pressure found: min(p) = {p.min()}"

    def test_double_rarefaction_physical_bounds(self, x_grid):
        """Test double rarefaction wave maintains ρ > 0, p > 0."""
        rho, u, p = exact_riemann(
            rho_L=DOUBLE_RARE_LEFT["rho"],
            u_L=DOUBLE_RARE_LEFT["u"],
            p_L=DOUBLE_RARE_LEFT["p"],
            rho_R=DOUBLE_RARE_RIGHT["rho"],
            u_R=DOUBLE_RARE_RIGHT["u"],
            p_R=DOUBLE_RARE_RIGHT["p"],
            x=x_grid,
            t=0.15,
            gamma=1.4,
        )
        assert torch.all(
            rho > 0
        ), f"Negative density in double rarefaction: min(ρ) = {rho.min()}"
        assert torch.all(
            p > 0
        ), f"Negative pressure in double rarefaction: min(p) = {p.min()}"

    def test_solution_bounded_by_initial_data(self, x_grid):
        """Verify solution values are bounded by extrema of initial conditions."""
        rho, u, p = exact_riemann(
            rho_L=SOD_LEFT["rho"],
            u_L=SOD_LEFT["u"],
            p_L=SOD_LEFT["p"],
            rho_R=SOD_RIGHT["rho"],
            u_R=SOD_RIGHT["u"],
            p_R=SOD_RIGHT["p"],
            x=x_grid,
            t=0.2,
            gamma=1.4,
        )
        # Density should be within reasonable bounds (no vacuum or extreme compression)
        assert rho.max() <= 10 * max(SOD_LEFT["rho"], SOD_RIGHT["rho"])
        assert rho.min() >= 0.01 * min(SOD_LEFT["rho"], SOD_RIGHT["rho"])


class TestApproximateRiemannSolvers:
    """Test HLL, HLLC, and Roe solvers maintain conservation."""

    @pytest.fixture
    def sod_conserved_states(self):
        """Create conserved variable states for Sod shock tube."""
        U_L = primitive_to_conserved(
            torch.tensor([SOD_LEFT["rho"]]),
            torch.tensor([SOD_LEFT["u"]]),
            torch.tensor([SOD_LEFT["p"]]),
            gamma=1.4,
        )
        U_R = primitive_to_conserved(
            torch.tensor([SOD_RIGHT["rho"]]),
            torch.tensor([SOD_RIGHT["u"]]),
            torch.tensor([SOD_RIGHT["p"]]),
            gamma=1.4,
        )
        return U_L, U_R

    def test_hll_produces_valid_flux(self, sod_conserved_states):
        """Verify HLL flux is finite and reasonably bounded."""
        U_L, U_R = sod_conserved_states
        flux = hll_flux(U_L, U_R, gamma=1.4)

        assert torch.all(torch.isfinite(flux)), "HLL flux contains NaN or Inf"
        assert torch.all(
            torch.abs(flux) < 1e6
        ), "HLL flux has unreasonably large values"

    def test_hllc_produces_valid_flux(self, sod_conserved_states):
        """Verify HLLC flux is finite and reasonably bounded."""
        U_L, U_R = sod_conserved_states
        flux = hllc_flux(U_L, U_R, gamma=1.4)

        assert torch.all(torch.isfinite(flux)), "HLLC flux contains NaN or Inf"
        assert torch.all(
            torch.abs(flux) < 1e6
        ), "HLLC flux has unreasonably large values"

    def test_roe_produces_valid_flux(self, sod_conserved_states):
        """Verify Roe flux is finite and reasonably bounded."""
        U_L, U_R = sod_conserved_states
        flux = roe_flux(U_L, U_R, gamma=1.4)

        assert torch.all(torch.isfinite(flux)), "Roe flux contains NaN or Inf"
        assert torch.all(
            torch.abs(flux) < 1e6
        ), "Roe flux has unreasonably large values"

    def test_flux_consistency_for_uniform_state(self):
        """If U_L = U_R, all solvers should give the same flux."""
        U = primitive_to_conserved(
            torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([1.0]), gamma=1.4
        )

        F_exact = euler_flux(U, gamma=1.4)
        F_hll = hll_flux(U, U, gamma=1.4)
        F_hllc = hllc_flux(U, U, gamma=1.4)
        F_roe = roe_flux(U, U, gamma=1.4)

        assert torch.allclose(
            F_hll, F_exact, atol=1e-10
        ), "HLL differs for uniform state"
        assert torch.allclose(
            F_hllc, F_exact, atol=1e-10
        ), "HLLC differs for uniform state"
        assert torch.allclose(
            F_roe, F_exact, atol=1e-10
        ), "Roe differs for uniform state"


class TestPrimitiveConservedConversion:
    """Test conversion between primitive and conserved variables."""

    def test_roundtrip_conversion(self):
        """Verify primitive → conserved → primitive is identity."""
        rho = torch.tensor([1.0, 0.5, 2.0])
        u = torch.tensor([0.0, 1.5, -0.5])
        p = torch.tensor([1.0, 0.4, 3.0])

        U = primitive_to_conserved(rho, u, p, gamma=1.4)
        rho2, u2, p2 = conserved_to_primitive(U, gamma=1.4)

        assert torch.allclose(rho, rho2, atol=1e-12)
        assert torch.allclose(u, u2, atol=1e-12)
        assert torch.allclose(p, p2, atol=1e-12)

    def test_conserved_energy_formula(self):
        """Verify E = p/(γ-1) + ρu²/2."""
        rho = torch.tensor([1.0])
        u = torch.tensor([2.0])
        p = torch.tensor([1.0])
        gamma = 1.4

        U = primitive_to_conserved(rho, u, p, gamma=gamma)

        expected_E = p / (gamma - 1) + 0.5 * rho * u**2
        actual_E = U[..., 2]

        assert torch.allclose(actual_E, expected_E, atol=1e-12)


class TestEulerFluxProperties:
    """Test Euler flux satisfies expected mathematical properties."""

    def test_mass_flux_equals_momentum(self):
        """Verify F[0] = ρu (mass flux equals momentum)."""
        rho = torch.tensor([1.5])
        u = torch.tensor([0.8])
        p = torch.tensor([1.2])

        U = primitive_to_conserved(rho, u, p, gamma=1.4)
        F = euler_flux(U, gamma=1.4)

        assert torch.allclose(F[..., 0], rho * u, atol=1e-12)

    def test_momentum_flux(self):
        """Verify F[1] = ρu² + p."""
        rho = torch.tensor([1.5])
        u = torch.tensor([0.8])
        p = torch.tensor([1.2])

        U = primitive_to_conserved(rho, u, p, gamma=1.4)
        F = euler_flux(U, gamma=1.4)

        expected = rho * u**2 + p
        assert torch.allclose(F[..., 1], expected, atol=1e-12)


class TestRankineHugoniotConditions:
    """Test that shock solutions satisfy Rankine-Hugoniot jump conditions."""

    def test_shock_jump_mass_conservation(self):
        """Verify [ρ(u-S)] = 0 across shock (mass flux continuity)."""
        x_grid = torch.linspace(-0.5, 0.5, 1001)
        rho, u, p = exact_riemann(
            rho_L=SOD_LEFT["rho"],
            u_L=SOD_LEFT["u"],
            p_L=SOD_LEFT["p"],
            rho_R=SOD_RIGHT["rho"],
            u_R=SOD_RIGHT["u"],
            p_R=SOD_RIGHT["p"],
            x=x_grid,
            t=0.2,
            gamma=1.4,
        )

        # Find the shock location (maximum density gradient on right side)
        drho = torch.diff(rho)
        right_half = drho[len(drho) // 2 :]
        shock_idx = len(drho) // 2 + torch.argmax(torch.abs(right_half))

        # Get states across shock
        rho_pre = rho[shock_idx - 5]  # Pre-shock (star state)
        rho_post = rho[shock_idx + 5]  # Post-shock (right state)
        u_pre = u[shock_idx - 5]
        u_post = u[shock_idx + 5]

        # Estimate shock speed from position
        # Shock should be near x = 0.2 * S where S is shock speed
        shock_x = x_grid[shock_idx]
        S_est = shock_x / 0.2

        # Mass flux should be conserved
        mass_flux_pre = rho_pre * (u_pre - S_est)
        mass_flux_post = rho_post * (u_post - S_est)

        # Relative error should be small
        rel_err = torch.abs(mass_flux_pre - mass_flux_post) / (
            torch.abs(mass_flux_pre) + 1e-10
        )
        assert rel_err < 0.05, f"Mass flux jump error: {rel_err:.2%}"


class TestEuler1DBoundaryConditions:
    """Test boundary condition implementations for 1D Euler solver."""

    def test_reflective_bc_reverses_momentum(self):
        """Verify reflective BC creates ghost cell with reversed momentum."""
        from tensornet.cfd import BCType1D, Euler1D, sod_shock_tube_ic

        solver = Euler1D(N=100, x_min=0.0, x_max=1.0)
        solver.set_boundary_conditions(
            left=BCType1D.REFLECTIVE, right=BCType1D.REFLECTIVE
        )

        # Create a state with non-zero velocity
        state = sod_shock_tube_ic(N=100, x_min=0.0, x_max=1.0, gamma=1.4)
        solver.set_initial_condition(state)

        # Manually set a velocity to test reflection
        solver.state.rho_u[0] = 1.0  # Positive momentum at left
        solver.state.rho_u[-1] = -1.0  # Negative momentum at right

        U = solver.state.to_conserved()

        # Check ghost cells
        left_ghost = solver._apply_left_bc(U)
        right_ghost = solver._apply_right_bc(U)

        # Momentum should be reversed
        assert (
            left_ghost[0, 1] == -U[0, 1]
        ), "Left reflective BC should reverse momentum"
        assert (
            right_ghost[0, 1] == -U[-1, 1]
        ), "Right reflective BC should reverse momentum"

        # Density and energy should be copied
        assert left_ghost[0, 0] == U[0, 0], "Left BC should copy density"
        assert left_ghost[0, 2] == U[0, 2], "Left BC should copy energy"

    def test_transmissive_bc_copies_state(self):
        """Verify transmissive BC creates zero-gradient ghost cell."""
        from tensornet.cfd import BCType1D, Euler1D, sod_shock_tube_ic

        solver = Euler1D(N=100, x_min=0.0, x_max=1.0)
        solver.set_boundary_conditions(
            left=BCType1D.TRANSMISSIVE, right=BCType1D.TRANSMISSIVE
        )

        state = sod_shock_tube_ic(N=100, x_min=0.0, x_max=1.0, gamma=1.4)
        solver.set_initial_condition(state)

        U = solver.state.to_conserved()

        left_ghost = solver._apply_left_bc(U)
        right_ghost = solver._apply_right_bc(U)

        assert torch.allclose(
            left_ghost, U[0:1]
        ), "Left transmissive should copy first cell"
        assert torch.allclose(
            right_ghost, U[-1:]
        ), "Right transmissive should copy last cell"

    def test_periodic_bc_wraps_state(self):
        """Verify periodic BC wraps state from opposite boundary."""
        from tensornet.cfd import BCType1D, Euler1D, sod_shock_tube_ic

        solver = Euler1D(N=100, x_min=0.0, x_max=1.0)
        solver.set_boundary_conditions(left=BCType1D.PERIODIC, right=BCType1D.PERIODIC)

        state = sod_shock_tube_ic(N=100, x_min=0.0, x_max=1.0, gamma=1.4)
        solver.set_initial_condition(state)

        U = solver.state.to_conserved()

        left_ghost = solver._apply_left_bc(U)
        right_ghost = solver._apply_right_bc(U)

        assert torch.allclose(
            left_ghost, U[-1:]
        ), "Left periodic should wrap from right"
        assert torch.allclose(
            right_ghost, U[0:1]
        ), "Right periodic should wrap from left"

    def test_reflective_bc_conserves_mass(self):
        """Verify simulation with reflective walls conserves total mass."""
        from tensornet.cfd import BCType1D, Euler1D
        from tensornet.cfd.euler_1d import EulerState

        solver = Euler1D(N=100, x_min=0.0, x_max=1.0, cfl=0.4)
        solver.set_boundary_conditions(
            left=BCType1D.REFLECTIVE, right=BCType1D.REFLECTIVE
        )

        # Uniform density with velocity pointing right
        rho = torch.ones(100, dtype=torch.float64)
        rho_u = torch.ones(100, dtype=torch.float64) * 0.5  # Rightward velocity
        p = torch.ones(100, dtype=torch.float64)
        E = p / 0.4 + 0.5 * rho_u**2 / rho

        state = EulerState(rho=rho, rho_u=rho_u, E=E, gamma=1.4)
        solver.set_initial_condition(state)

        initial_mass = solver.state.rho.sum().item()

        # Run for several steps
        for _ in range(50):
            solver.step()

        final_mass = solver.state.rho.sum().item()

        # Mass should be conserved (within numerical tolerance)
        rel_error = abs(final_mass - initial_mass) / initial_mass
        assert rel_error < 0.01, f"Mass not conserved: {rel_error:.2%} error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
