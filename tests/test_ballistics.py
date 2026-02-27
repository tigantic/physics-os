"""
Test Module: tensornet/defense/ballistics.py

Phase 13: 6-DOF Ballistic Trajectory Solver
Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation
    - Article V, Section 5.1: Float64 precision

References:
    McCoy, R.L. (1999). "Modern Exterior Ballistics."
    Schiffer Publishing. ISBN: 0764307207

    Litz, B. (2015). "Applied Ballistics for Long Range Shooting."
    Applied Ballistics LLC. ISBN: 978-0990920618
"""

import numpy as np
import pytest
import torch

from tensornet.aerospace.defense.ballistics import BallisticSolution, BallisticSolver

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
def solver_338():
    """.338 Lapua Magnum solver (long range precision)."""
    return BallisticSolver(
        bullet_mass_grains=250.0, muzzle_velocity_fps=2950.0, bc_g7=0.310
    )


@pytest.fixture
def solver_308():
    """.308 Winchester solver (medium range)."""
    return BallisticSolver(
        bullet_mass_grains=175.0, muzzle_velocity_fps=2650.0, bc_g7=0.243
    )


# ============================================================================
# UNIT TESTS
# ============================================================================


class TestBallisticSolverInit:
    """Test BallisticSolver initialization."""

    @pytest.mark.unit
    def test_init_338(self, deterministic_seed):
        """Test .338 Lapua initialization."""
        solver = BallisticSolver(
            bullet_mass_grains=250.0, muzzle_velocity_fps=2950.0, bc_g7=0.310
        )

        assert solver.bc == 0.310
        # Check SI conversion (2950 fps ≈ 899 m/s)
        assert solver.muzzle_velocity == pytest.approx(899.16, rel=0.01)

    @pytest.mark.unit
    def test_mass_conversion(self, solver_338, deterministic_seed):
        """Test grains to kg conversion."""
        # 250 grains = 0.0162 kg
        expected_kg = 250 * 0.0000648
        assert solver_338.mass == pytest.approx(expected_kg, rel=0.01)

    @pytest.mark.unit
    def test_float64_gravity(self, solver_338, deterministic_seed):
        """Gravity tensor should be float64."""
        assert solver_338.gravity.dtype == torch.float64
        assert solver_338.gravity[1].item() == pytest.approx(-9.81, rel=0.01)


class TestDragComputation:
    """Test aerodynamic drag calculations."""

    @pytest.mark.unit
    @pytest.mark.physics
    def test_drag_opposes_motion(self, solver_338, deterministic_seed):
        """Drag force should oppose velocity."""
        velocity = torch.tensor([0.0, 0.0, 500.0], dtype=torch.float64)
        wind = torch.zeros(3, dtype=torch.float64)

        drag = solver_338.compute_drag(velocity, wind)

        # Drag should be in -Z direction (opposing motion)
        assert drag[2].item() < 0

    @pytest.mark.unit
    @pytest.mark.physics
    def test_drag_increases_with_speed(self, solver_338, deterministic_seed):
        """Drag force should increase with velocity squared."""
        slow = torch.tensor([0.0, 0.0, 300.0], dtype=torch.float64)
        fast = torch.tensor([0.0, 0.0, 600.0], dtype=torch.float64)
        wind = torch.zeros(3, dtype=torch.float64)

        drag_slow = solver_338.compute_drag(slow, wind)
        drag_fast = solver_338.compute_drag(fast, wind)

        # 2x velocity → 4x drag
        ratio = abs(drag_fast[2].item() / drag_slow[2].item())
        assert ratio > 3.0  # Should be ~4x

    @pytest.mark.unit
    @pytest.mark.physics
    def test_higher_bc_less_drag(self, deterministic_seed):
        """Higher BC should result in less drag."""
        low_bc = BallisticSolver(bc_g7=0.200)
        high_bc = BallisticSolver(bc_g7=0.400)

        velocity = torch.tensor([0.0, 0.0, 500.0], dtype=torch.float64)
        wind = torch.zeros(3, dtype=torch.float64)

        drag_low = low_bc.compute_drag(velocity, wind)
        drag_high = high_bc.compute_drag(velocity, wind)

        assert abs(drag_high[2].item()) < abs(drag_low[2].item())


class TestWindField:
    """Test wind field sampling."""

    @pytest.mark.unit
    def test_default_wind_shear(self, solver_338, deterministic_seed):
        """Default wind model should have shear."""
        pos_near = torch.tensor([0.0, 0.0, 100.0], dtype=torch.float64)
        pos_far = torch.tensor([0.0, 0.0, 600.0], dtype=torch.float64)

        wind_near = solver_338.sample_wind_field(pos_near)
        wind_far = solver_338.sample_wind_field(pos_far)

        # Wind direction should differ at 500m boundary
        assert not torch.allclose(wind_near, wind_far)


class TestTrajectory:
    """Test trajectory solving."""

    @pytest.mark.unit
    def test_solve_returns_solution(self, solver_338, deterministic_seed):
        """Solver should return BallisticSolution."""
        solution = solver_338.solve_trajectory(target_distance=1000.0, verbose=False)

        assert isinstance(solution, BallisticSolution)

    @pytest.mark.unit
    @pytest.mark.physics
    def test_bullet_drops(self, solver_338, deterministic_seed):
        """Bullet should drop due to gravity."""
        solution = solver_338.solve_trajectory(
            target_distance=500.0, target_elevation=0.0, verbose=False
        )

        # Drop should be negative (below aim point)
        assert solution.drop_meters < 0

    @pytest.mark.unit
    @pytest.mark.physics
    def test_longer_range_more_drop(self, solver_338, deterministic_seed):
        """Longer range should have more drop."""
        sol_500 = solver_338.solve_trajectory(500.0, verbose=False)
        sol_1000 = solver_338.solve_trajectory(1000.0, verbose=False)

        assert abs(sol_1000.drop_meters) > abs(sol_500.drop_meters)

    @pytest.mark.unit
    @pytest.mark.physics
    def test_time_of_flight_positive(self, solver_338, deterministic_seed):
        """Time of flight should be positive."""
        solution = solver_338.solve_trajectory(1000.0, verbose=False)

        assert solution.time_of_flight > 0

    @pytest.mark.unit
    @pytest.mark.physics
    def test_impact_velocity_less_than_muzzle(self, solver_338, deterministic_seed):
        """Impact velocity should be less than muzzle velocity (drag)."""
        solution = solver_338.solve_trajectory(1000.0, verbose=False)

        assert solution.impact_velocity < solver_338.muzzle_velocity

    @pytest.mark.unit
    def test_solution_has_corrections(self, solver_338, deterministic_seed):
        """Solution should have elevation and windage corrections."""
        solution = solver_338.solve_trajectory(1000.0, verbose=False)

        assert solution.elevation_moa != 0  # Some correction needed
        assert solution.elevation_mils != 0


class TestBallisticSolution:
    """Test BallisticSolution dataclass."""

    @pytest.mark.unit
    def test_solution_string(self, solver_338, deterministic_seed):
        """Solution should have string representation."""
        solution = solver_338.solve_trajectory(1000.0, verbose=False)

        sol_str = str(solution)
        assert "Target" in sol_str or "MOA" in sol_str

    @pytest.mark.unit
    def test_moa_mils_conversion(self, solver_338, deterministic_seed):
        """MOA and Mils should be consistent."""
        solution = solver_338.solve_trajectory(1000.0, verbose=False)

        # 1 MOA ≈ 0.2909 Mils
        expected_mils = solution.elevation_moa * 0.2909
        assert solution.elevation_mils == pytest.approx(expected_mils, rel=0.05)


class TestDeterminism:
    """Test reproducibility requirements."""

    @pytest.mark.unit
    def test_deterministic_trajectory(self, deterministic_seed):
        """Same inputs should give same trajectory."""
        s1 = BallisticSolver(bc_g7=0.310)
        sol1 = s1.solve_trajectory(1000.0, verbose=False)

        s2 = BallisticSolver(bc_g7=0.310)
        sol2 = s2.solve_trajectory(1000.0, verbose=False)

        assert sol1.drop_meters == sol2.drop_meters
        assert sol1.time_of_flight == sol2.time_of_flight


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestBallisticsIntegration:
    """Integration tests for ballistics solver."""

    @pytest.mark.integration
    def test_full_long_range_solution(self, deterministic_seed):
        """Test complete long-range solution workflow."""
        # .338 Lapua at 1500m
        solver = BallisticSolver(
            bullet_mass_grains=250.0, muzzle_velocity_fps=2950.0, bc_g7=0.310
        )

        solution = solver.solve_trajectory(
            target_distance=1500.0, target_elevation=0.0, verbose=False
        )

        # Verify reasonable results for 1500m
        assert solution.time_of_flight > 2.0  # Should take > 2 seconds
        assert abs(solution.drop_meters) > 10  # Significant drop
        assert solution.impact_velocity > 400  # Still supersonic

    @pytest.mark.integration
    def test_uphill_vs_downhill(self, deterministic_seed):
        """Uphill and downhill shots have different solutions."""
        solver = BallisticSolver(bc_g7=0.310)

        sol_flat = solver.solve_trajectory(1000.0, target_elevation=0.0, verbose=False)
        sol_uphill = solver.solve_trajectory(
            1000.0, target_elevation=100.0, verbose=False
        )
        sol_downhill = solver.solve_trajectory(
            1000.0, target_elevation=-100.0, verbose=False
        )

        # Elevation changes should affect solution
        # (Rifleman's rule: shoot high uphill/downhill)
        assert sol_flat.elevation_moa != sol_uphill.elevation_moa
        assert sol_flat.elevation_moa != sol_downhill.elevation_moa


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
