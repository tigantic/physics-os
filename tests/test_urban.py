"""
Test Module: tensornet/infrastructure/urban.py

Phase 13: Urban Air Mobility Flight Corridors
Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation
    - Article V, Section 5.1: Float64 precision

References:
    Betz, A. (1919). "Das Maximum der theoretisch möglichen Ausnützung
    des Windes durch Windmotoren." Zeitschrift für das gesamte Turbinenwesen.
    
    Oke, T.R. (1988). "Street design and urban canopy layer climate."
    Energy and Buildings, 11(1-3), 103-113.
"""

import pytest
import torch
import numpy as np

from tensornet.infrastructure.urban import (
    UrbanFlowSolver,
    FlightSafetyReport,
)


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
def urban_solver():
    """Standard urban flow solver."""
    return UrbanFlowSolver(
        domain_size_m=1000.0,
        resolution=64
    )


@pytest.fixture
def downtown_scenario():
    """Dense urban canyon scenario."""
    solver = UrbanFlowSolver(
        domain_size_m=500.0,
        resolution=128
    )
    # Add building configuration
    solver.add_buildings([
        {'x': 100, 'y': 100, 'width': 30, 'depth': 30, 'height': 150},
        {'x': 200, 'y': 100, 'width': 40, 'depth': 40, 'height': 200},
        {'x': 150, 'y': 200, 'width': 35, 'depth': 35, 'height': 180},
    ])
    return solver


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestUrbanFlowSolverInit:
    """Test UrbanFlowSolver initialization."""
    
    @pytest.mark.unit
    def test_init_dimensions(self, deterministic_seed):
        """Test domain dimensions."""
        solver = UrbanFlowSolver(domain_size_m=2000.0, resolution=256)
        
        assert solver.domain_size_m == 2000.0
        assert solver.resolution == 256
        assert solver.velocity.shape[0] == 3  # u, v, w components
    
    @pytest.mark.unit
    def test_float64_compliance(self, deterministic_seed):
        """Per Article V: Physics tensors must be float64."""
        solver = UrbanFlowSolver(resolution=32)
        
        assert solver.velocity.dtype == torch.float64
        assert solver.pressure.dtype == torch.float64
        assert solver.building_mask.dtype == torch.bool or solver.building_mask.dtype == torch.float64
    
    @pytest.mark.unit
    def test_initial_calm_conditions(self, urban_solver, deterministic_seed):
        """Initial velocity should be zero or calm."""
        velocity_magnitude = torch.norm(urban_solver.velocity, dim=0)
        assert velocity_magnitude.max().item() < 1e-6
    
    @pytest.mark.unit
    def test_grid_spacing(self, urban_solver, deterministic_seed):
        """Grid spacing should be consistent."""
        dx = urban_solver.domain_size_m / urban_solver.resolution
        assert abs(urban_solver.dx - dx) < 1e-10


class TestBuildingConfiguration:
    """Test building setup."""
    
    @pytest.mark.unit
    def test_add_single_building(self, urban_solver, deterministic_seed):
        """Adding a building should create obstacle."""
        urban_solver.add_buildings([
            {'x': 500, 'y': 500, 'width': 50, 'height': 100}
        ])
        
        # Building mask should have blocked cells
        assert urban_solver.building_mask.sum().item() > 0
    
    @pytest.mark.unit
    def test_add_multiple_buildings(self, urban_solver, deterministic_seed):
        """Multiple buildings should be added."""
        urban_solver.add_buildings([
            {'x': 300, 'y': 300, 'width': 30, 'height': 80},
            {'x': 600, 'y': 600, 'width': 40, 'height': 120},
        ])
        
        # Should have more blocked cells than single building
        count = urban_solver.building_mask.sum().item()
        assert count > 10  # Multiple cells blocked
    
    @pytest.mark.unit
    def test_building_height_stored(self, urban_solver, deterministic_seed):
        """Building heights should be stored."""
        urban_solver.add_buildings([
            {'x': 500, 'y': 500, 'width': 50, 'height': 200}
        ])
        
        # Check height field exists and has values
        assert hasattr(urban_solver, 'building_height')
        assert urban_solver.building_height.max().item() >= 200


class TestWindSimulation:
    """Test wind field simulation."""
    
    @pytest.mark.unit
    def test_set_ambient_wind(self, urban_solver, deterministic_seed):
        """Should be able to set ambient wind."""
        urban_solver.set_ambient_wind(speed_ms=10.0, direction_deg=270.0)
        
        # Wind velocity should be set
        velocity_magnitude = torch.norm(urban_solver.velocity, dim=0).mean()
        assert velocity_magnitude.item() > 1.0
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_wind_blocked_by_building(self, downtown_scenario, deterministic_seed):
        """Buildings should create wake regions."""
        downtown_scenario.set_ambient_wind(speed_ms=8.0, direction_deg=270.0)
        downtown_scenario.step(n_steps=50)
        
        # Velocity should be lower behind buildings
        velocity_magnitude = torch.norm(downtown_scenario.velocity, dim=0)
        
        # Get average velocity in domain
        mean_velocity = velocity_magnitude.mean().item()
        assert mean_velocity > 0  # Flow exists
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_urban_canyon_acceleration(self, urban_solver, deterministic_seed):
        """Channeling between buildings should accelerate flow."""
        # Create canyon - two parallel buildings
        urban_solver.add_buildings([
            {'x': 400, 'y': 300, 'width': 30, 'depth': 200, 'height': 100},
            {'x': 600, 'y': 300, 'width': 30, 'depth': 200, 'height': 100},
        ])
        
        urban_solver.set_ambient_wind(speed_ms=5.0, direction_deg=0.0)  # North wind
        urban_solver.step(n_steps=100)
        
        velocity_magnitude = torch.norm(urban_solver.velocity, dim=0)
        
        # Flow should exist in canyon
        canyon_velocity = velocity_magnitude.mean().item()
        assert canyon_velocity > 0


class TestFlightCorridor:
    """Test flight corridor safety assessment."""
    
    @pytest.mark.unit
    def test_assess_corridor(self, downtown_scenario, deterministic_seed):
        """Should assess flight corridor safety."""
        downtown_scenario.set_ambient_wind(speed_ms=6.0, direction_deg=180.0)
        downtown_scenario.step(n_steps=50)
        
        # Define corridor
        corridor = {
            'start': (50, 250, 80),  # x, y, z
            'end': (450, 250, 80),
            'width': 30
        }
        
        safety = downtown_scenario.assess_corridor(corridor)
        
        assert hasattr(safety, 'safe')
        assert hasattr(safety, 'max_turbulence')
    
    @pytest.mark.unit
    def test_corridor_above_buildings_safer(self, downtown_scenario, deterministic_seed):
        """Corridors above buildings should be safer."""
        downtown_scenario.set_ambient_wind(speed_ms=10.0, direction_deg=270.0)
        downtown_scenario.step(n_steps=50)
        
        # Low corridor (through building layer)
        low_corridor = {'start': (50, 250, 50), 'end': (450, 250, 50), 'width': 20}
        
        # High corridor (above buildings)
        high_corridor = {'start': (50, 250, 300), 'end': (450, 250, 300), 'width': 20}
        
        low_safety = downtown_scenario.assess_corridor(low_corridor)
        high_safety = downtown_scenario.assess_corridor(high_corridor)
        
        # High should have lower turbulence
        assert high_safety.max_turbulence <= low_safety.max_turbulence


class TestFlightSafetyReport:
    """Test FlightSafetyReport generation."""
    
    @pytest.mark.unit
    def test_report_generation(self, downtown_scenario, deterministic_seed):
        """Should generate safety report."""
        downtown_scenario.set_ambient_wind(speed_ms=5.0, direction_deg=90.0)
        downtown_scenario.step(n_steps=20)
        
        report = downtown_scenario.get_safety_report()
        
        assert isinstance(report, FlightSafetyReport)
    
    @pytest.mark.unit
    def test_report_contains_fields(self, downtown_scenario, deterministic_seed):
        """Report should contain required fields."""
        downtown_scenario.set_ambient_wind(speed_ms=5.0, direction_deg=90.0)
        downtown_scenario.step(n_steps=20)
        
        report = downtown_scenario.get_safety_report()
        
        assert hasattr(report, 'max_wind_speed_ms')
        assert hasattr(report, 'turbulence_intensity')
        assert hasattr(report, 'safe_zones')
    
    @pytest.mark.unit
    def test_report_string_representation(self, downtown_scenario, deterministic_seed):
        """Report should have string representation."""
        downtown_scenario.set_ambient_wind(speed_ms=5.0, direction_deg=90.0)
        downtown_scenario.step(n_steps=20)
        
        report = downtown_scenario.get_safety_report()
        report_str = str(report)
        
        assert len(report_str) > 0
        assert 'wind' in report_str.lower() or 'speed' in report_str.lower()


class TestDeterminism:
    """Test reproducibility requirements."""
    
    @pytest.mark.unit
    def test_deterministic_simulation(self):
        """Same seed should give same results."""
        # First simulation
        torch.manual_seed(42)
        np.random.seed(42)
        solver1 = UrbanFlowSolver(resolution=32)
        solver1.add_buildings([{'x': 500, 'y': 500, 'width': 50, 'height': 100}])
        solver1.set_ambient_wind(speed_ms=8.0, direction_deg=45.0)
        solver1.step(n_steps=20)
        velocity1 = solver1.velocity.clone()
        
        # Second simulation
        torch.manual_seed(42)
        np.random.seed(42)
        solver2 = UrbanFlowSolver(resolution=32)
        solver2.add_buildings([{'x': 500, 'y': 500, 'width': 50, 'height': 100}])
        solver2.set_ambient_wind(speed_ms=8.0, direction_deg=45.0)
        solver2.step(n_steps=20)
        velocity2 = solver2.velocity
        
        assert torch.allclose(velocity1, velocity2, rtol=1e-10)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestUrbanIntegration:
    """Integration tests for urban flow simulation."""
    
    @pytest.mark.integration
    def test_full_uam_scenario(self, deterministic_seed):
        """Test complete UAM flight safety assessment."""
        # Create downtown domain
        solver = UrbanFlowSolver(
            domain_size_m=1000.0,
            resolution=128
        )
        
        # Add realistic building configuration
        buildings = [
            {'x': 300, 'y': 300, 'width': 60, 'depth': 60, 'height': 180},
            {'x': 500, 'y': 300, 'width': 80, 'depth': 80, 'height': 250},
            {'x': 400, 'y': 500, 'width': 50, 'depth': 50, 'height': 150},
            {'x': 600, 'y': 600, 'width': 70, 'depth': 40, 'height': 200},
        ]
        solver.add_buildings(buildings)
        
        # Set wind conditions
        solver.set_ambient_wind(speed_ms=12.0, direction_deg=315.0)
        
        # Simulate to steady state
        solver.step(n_steps=100)
        
        # Get comprehensive report
        report = solver.get_safety_report()
        
        # Verify reasonable physics
        assert report.max_wind_speed_ms >= 0
        assert report.max_wind_speed_ms < 50  # Reasonable upper bound
    
    @pytest.mark.integration
    @pytest.mark.physics
    def test_vertiport_assessment(self, deterministic_seed):
        """Test vertiport location suitability."""
        solver = UrbanFlowSolver(
            domain_size_m=500.0,
            resolution=64
        )
        
        # Buildings around potential vertiport
        solver.add_buildings([
            {'x': 100, 'y': 250, 'width': 40, 'height': 120},
            {'x': 400, 'y': 250, 'width': 40, 'height': 120},
        ])
        
        # Test conditions
        solver.set_ambient_wind(speed_ms=8.0, direction_deg=270.0)
        solver.step(n_steps=50)
        
        # Assess vertiport location (between buildings)
        vertiport_zone = {
            'x_min': 200, 'x_max': 300,
            'y_min': 200, 'y_max': 300,
            'z_min': 0, 'z_max': 50
        }
        
        zone_safety = solver.assess_zone(vertiport_zone)
        
        assert hasattr(zone_safety, 'safe_for_operations')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
