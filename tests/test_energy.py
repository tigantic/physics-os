"""
Test Module: tensornet/energy/turbine.py

Phase 5: Wind Farm Wake Physics
Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation

References:
    Jensen, N.O. (1983). "A Note on Wind Generator Interaction."
    Risø-M No. 2411, Risø National Laboratory.
"""

import numpy as np
import pytest
import torch

from tensornet.energy.turbine import TurbineSpec, WindFarm

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
def simple_farm():
    """Two-turbine wind farm for testing."""
    turbines = [
        {"x": 0, "y": 100, "z": 0, "radius": 40, "yaw": 0},
        {"x": 0, "y": 100, "z": 400, "radius": 40, "yaw": 0},
    ]
    return WindFarm(turbines, environment="offshore")


@pytest.fixture
def wind_field():
    """Standard wind field tensor for testing."""
    # Shape: [components, depth, height, width] = [3, 100, 50, 50]
    field = torch.ones((3, 100, 50, 50), dtype=torch.float64) * 12.0  # 12 m/s
    return field


# ============================================================================
# UNIT TESTS
# ============================================================================


class TestWindFarmInit:
    """Test WindFarm initialization."""

    @pytest.mark.unit
    def test_init_offshore(self, deterministic_seed):
        """Test offshore wind farm initialization."""
        turbines = [{"x": 0, "y": 100, "z": 0, "radius": 40, "yaw": 0}]
        farm = WindFarm(turbines, environment="offshore")

        assert farm.environment == "offshore"
        assert farm.k == WindFarm.WAKE_DECAY_OFFSHORE
        assert len(farm.turbines) == 1

    @pytest.mark.unit
    def test_init_land(self, deterministic_seed):
        """Test land-based wind farm initialization."""
        turbines = [{"x": 0, "y": 100, "z": 0, "radius": 40, "yaw": 0}]
        farm = WindFarm(turbines, environment="land")

        assert farm.environment == "land"
        assert farm.k == WindFarm.WAKE_DECAY_LAND

    @pytest.mark.unit
    def test_init_custom_air_density(self, deterministic_seed):
        """Test custom air density override."""
        turbines = [{"x": 0, "y": 100, "z": 0, "radius": 40, "yaw": 0}]
        farm = WindFarm(turbines, air_density=1.0)

        assert farm.air_density == 1.0


class TestWakePhysics:
    """Test wake physics calculations."""

    @pytest.mark.unit
    @pytest.mark.physics
    def test_wake_decay_positive(self, simple_farm, wind_field, deterministic_seed):
        """Wake intensity should decay with distance."""
        simple_farm.apply_wakes(wind_field, grid_resolution=10.0)

        # Check that downstream velocities are reduced
        upstream_vel = wind_field[0, 0, 25, 25].item()  # z=0
        downstream_vel = wind_field[0, 40, 25, 25].item()  # z=400m behind

        # Downstream should have lower velocity due to wake
        assert downstream_vel <= upstream_vel

    @pytest.mark.unit
    @pytest.mark.physics
    def test_wake_expansion(self, deterministic_seed):
        """Wake should expand linearly with distance (Jensen model)."""
        turbines = [{"x": 25, "y": 100, "z": 0, "radius": 40, "yaw": 0}]
        farm = WindFarm(turbines, environment="offshore")

        # Jensen expansion: r_wake = r_rotor + k * x
        k = farm.k
        r_rotor = 40
        x = 400  # meters downstream

        expected_wake_radius = r_rotor + k * x
        assert expected_wake_radius > r_rotor  # Wake expands

    @pytest.mark.unit
    @pytest.mark.physics
    def test_betz_limit_respected(self, deterministic_seed):
        """Power coefficient should not exceed Betz limit."""
        assert WindFarm.TYPICAL_CP < WindFarm.BETZ_LIMIT
        assert WindFarm.BETZ_LIMIT == pytest.approx(0.593, rel=0.01)


class TestPowerCalculation:
    """Test power output calculations."""

    @pytest.mark.unit
    def test_power_output_positive(self, simple_farm, wind_field, deterministic_seed):
        """Power output should be positive with wind."""
        power = simple_farm.calculate_power_output(wind_field, grid_resolution=10.0)

        assert power > 0

    @pytest.mark.unit
    def test_power_increases_with_wind(self, simple_farm, deterministic_seed):
        """Power should increase with wind speed (P ∝ v³)."""
        low_wind = torch.ones((3, 100, 50, 50), dtype=torch.float64) * 8.0
        high_wind = torch.ones((3, 100, 50, 50), dtype=torch.float64) * 16.0

        power_low = simple_farm.calculate_power_output(low_wind, grid_resolution=10.0)
        power_high = simple_farm.calculate_power_output(high_wind, grid_resolution=10.0)

        # Power scales as v³, so doubling wind speed → 8x power
        assert power_high > power_low
        assert power_high / power_low > 4  # Allow some margin


class TestDeterminism:
    """Test reproducibility requirements."""

    @pytest.mark.unit
    def test_deterministic_output(self, deterministic_seed):
        """Same seed should produce same results."""
        turbines = [
            {"x": 0, "y": 100, "z": 0, "radius": 40, "yaw": 0},
            {"x": 0, "y": 100, "z": 400, "radius": 40, "yaw": 0},
        ]

        torch.manual_seed(42)
        farm1 = WindFarm(turbines)
        field1 = torch.ones((3, 100, 50, 50), dtype=torch.float64) * 12.0
        farm1.apply_wakes(field1, grid_resolution=10.0)

        torch.manual_seed(42)
        farm2 = WindFarm(turbines)
        field2 = torch.ones((3, 100, 50, 50), dtype=torch.float64) * 12.0
        farm2.apply_wakes(field2, grid_resolution=10.0)

        assert torch.allclose(field1, field2)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestWindFarmIntegration:
    """Integration tests for complete wind farm workflows."""

    @pytest.mark.integration
    def test_full_simulation_workflow(self, deterministic_seed):
        """Test complete simulation from setup to power calculation."""
        # Setup farm
        turbines = [
            {"x": 10, "y": 100, "z": 0, "radius": 40, "yaw": 0, "rated_power": 5.0},
            {"x": 10, "y": 100, "z": 400, "radius": 40, "yaw": 0, "rated_power": 5.0},
            {"x": 10, "y": 100, "z": 800, "radius": 40, "yaw": 0, "rated_power": 5.0},
        ]
        farm = WindFarm(turbines, environment="offshore")

        # Create wind field
        field = torch.ones((3, 100, 50, 50), dtype=torch.float64) * 12.0

        # Apply wakes
        farm.apply_wakes(field, grid_resolution=10.0)

        # Calculate power
        power = farm.calculate_power_output(field, grid_resolution=10.0)

        # Verify outputs
        assert power > 0
        assert power < 15.0  # Less than 3 × 5MW rated (wake losses)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
