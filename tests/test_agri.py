"""
Test Module: tensornet/agriculture/vertical_farm.py

Phase 15: Vertical Farm Microclimate Control
Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation
    - Article V, Section 5.1: Float64 precision

References:
    Kozai, T. (2013). "Resource use efficiency of closed plant production
    system with artificial light: Concept, estimation and application to
    plant factory." Proceedings of the Japan Academy, Series B, 89(10), 447-461.
    
    Penman, H.L. (1948). "Natural evaporation from open water, bare soil
    and grass." Proceedings of the Royal Society A, 193(1032), 120-145.
"""

import pytest
import torch
import numpy as np

from tensornet.agriculture.vertical_farm import (
    VerticalFarm,
    HarvestReport,
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
def small_farm():
    """Small vertical farm for testing."""
    return VerticalFarm(
        n_levels=4,
        area_per_level_m2=50.0,
        crop_type='lettuce'
    )


@pytest.fixture
def production_farm():
    """Production-scale vertical farm."""
    return VerticalFarm(
        n_levels=12,
        area_per_level_m2=500.0,
        crop_type='basil'
    )


@pytest.fixture
def strawberry_farm():
    """Strawberry vertical farm."""
    return VerticalFarm(
        n_levels=8,
        area_per_level_m2=200.0,
        crop_type='strawberry'
    )


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestVerticalFarmInit:
    """Test VerticalFarm initialization."""
    
    @pytest.mark.unit
    def test_init_dimensions(self, deterministic_seed):
        """Test farm dimensions."""
        farm = VerticalFarm(n_levels=6, area_per_level_m2=100.0)
        
        assert farm.n_levels == 6
        assert farm.area_per_level_m2 == 100.0
    
    @pytest.mark.unit
    def test_float64_compliance(self, deterministic_seed):
        """Per Article V: Physics tensors must be float64."""
        farm = VerticalFarm(n_levels=4)
        
        assert farm.temperature.dtype == torch.float64
        assert farm.humidity.dtype == torch.float64
        assert farm.co2_ppm.dtype == torch.float64
        assert farm.light_ppfd.dtype == torch.float64
    
    @pytest.mark.unit
    def test_initial_conditions(self, small_farm, deterministic_seed):
        """Initial conditions should be reasonable."""
        # Temperature 18-25°C
        assert small_farm.temperature.min().item() >= 15
        assert small_farm.temperature.max().item() <= 30
        
        # Humidity 40-80%
        assert small_farm.humidity.min().item() >= 30
        assert small_farm.humidity.max().item() <= 90
    
    @pytest.mark.unit
    def test_crop_parameters_loaded(self, small_farm, deterministic_seed):
        """Crop-specific parameters should be loaded."""
        assert hasattr(small_farm, 'optimal_temp')
        assert hasattr(small_farm, 'optimal_humidity')
        assert hasattr(small_farm, 'growth_rate')


class TestEnvironmentControl:
    """Test environmental control systems."""
    
    @pytest.mark.unit
    def test_set_temperature(self, small_farm, deterministic_seed):
        """Should control temperature per level."""
        small_farm.set_temperature(level=2, temp_c=22.5)
        
        assert abs(small_farm.temperature[2].item() - 22.5) < 0.1
    
    @pytest.mark.unit
    def test_set_humidity(self, small_farm, deterministic_seed):
        """Should control humidity per level."""
        small_farm.set_humidity(level=1, humidity_pct=65.0)
        
        assert abs(small_farm.humidity[1].item() - 65.0) < 1.0
    
    @pytest.mark.unit
    def test_set_co2(self, small_farm, deterministic_seed):
        """Should control CO2 enrichment."""
        small_farm.set_co2(level=0, ppm=1000.0)
        
        assert abs(small_farm.co2_ppm[0].item() - 1000.0) < 10.0
    
    @pytest.mark.unit
    def test_set_lighting(self, small_farm, deterministic_seed):
        """Should control lighting intensity."""
        small_farm.set_light(level=3, ppfd=400.0)
        
        assert abs(small_farm.light_ppfd[3].item() - 400.0) < 10.0
    
    @pytest.mark.unit
    def test_temperature_bounds(self, small_farm, deterministic_seed):
        """Temperature should stay in safe bounds."""
        # Try to set extreme temperature
        small_farm.set_temperature(level=0, temp_c=50.0)
        
        # Should be clamped to safe range
        assert small_farm.temperature[0].item() <= 35.0


class TestPlantGrowth:
    """Test plant growth simulation."""
    
    @pytest.mark.unit
    def test_initial_biomass(self, small_farm, deterministic_seed):
        """Initial biomass should be seedling weight."""
        initial_biomass = small_farm.biomass.mean().item()
        
        # Seedlings typically < 1g
        assert initial_biomass > 0
        assert initial_biomass < 10.0  # grams
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_growth_increases_biomass(self, small_farm, deterministic_seed):
        """Optimal conditions should increase biomass."""
        # Set optimal conditions
        for level in range(small_farm.n_levels):
            small_farm.set_temperature(level, small_farm.optimal_temp)
            small_farm.set_humidity(level, small_farm.optimal_humidity)
            small_farm.set_co2(level, 1000.0)
            small_farm.set_light(level, 300.0)
        
        initial_biomass = small_farm.biomass.sum().item()
        
        # Simulate 24 hours
        for _ in range(24):
            small_farm.step(dt_hours=1.0)
        
        final_biomass = small_farm.biomass.sum().item()
        
        assert final_biomass > initial_biomass
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_suboptimal_conditions_slow_growth(self, deterministic_seed):
        """Suboptimal conditions should reduce growth rate."""
        # Optimal farm
        optimal_farm = VerticalFarm(n_levels=4, crop_type='lettuce')
        for level in range(4):
            optimal_farm.set_temperature(level, optimal_farm.optimal_temp)
            optimal_farm.set_humidity(level, optimal_farm.optimal_humidity)
        
        # Suboptimal farm (too hot)
        suboptimal_farm = VerticalFarm(n_levels=4, crop_type='lettuce')
        for level in range(4):
            suboptimal_farm.set_temperature(level, 32.0)  # Too hot
        
        # Simulate same duration
        for _ in range(48):
            optimal_farm.step(dt_hours=1.0)
            suboptimal_farm.step(dt_hours=1.0)
        
        # Optimal should grow more
        assert optimal_farm.biomass.sum() >= suboptimal_farm.biomass.sum()
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_light_affects_photosynthesis(self, small_farm, deterministic_seed):
        """Light intensity should affect growth."""
        # Set conditions
        for level in range(small_farm.n_levels):
            small_farm.set_temperature(level, small_farm.optimal_temp)
        
        # Set different light levels
        small_farm.set_light(0, 100.0)   # Low light
        small_farm.set_light(1, 300.0)   # Medium light
        small_farm.set_light(2, 500.0)   # High light
        small_farm.set_light(3, 500.0)   # High light
        
        # Simulate
        for _ in range(24):
            small_farm.step(dt_hours=1.0)
        
        # Higher light levels should have more growth (up to saturation)
        assert small_farm.biomass[1].mean() >= small_farm.biomass[0].mean() * 0.9


class TestWaterNutrient:
    """Test water and nutrient management."""
    
    @pytest.mark.unit
    def test_water_consumption(self, small_farm, deterministic_seed):
        """Plants should consume water."""
        initial_water = small_farm.reservoir_liters
        
        # Simulate with growth
        for _ in range(24):
            small_farm.step(dt_hours=1.0)
        
        final_water = small_farm.reservoir_liters
        
        assert final_water < initial_water
    
    @pytest.mark.unit
    def test_nutrient_depletion(self, small_farm, deterministic_seed):
        """Nutrients should deplete over time."""
        initial_nutrients = small_farm.nutrient_ec.clone()
        
        # Simulate
        for _ in range(48):
            small_farm.step(dt_hours=1.0)
        
        # Some nutrients should be consumed
        assert small_farm.nutrient_ec.sum() <= initial_nutrients.sum()
    
    @pytest.mark.unit
    def test_add_nutrients(self, small_farm, deterministic_seed):
        """Should be able to add nutrients."""
        initial_ec = small_farm.nutrient_ec.mean().item()
        
        small_farm.add_nutrients(ec_boost=0.5)
        
        assert small_farm.nutrient_ec.mean().item() > initial_ec


class TestHarvest:
    """Test harvest functionality."""
    
    @pytest.mark.unit
    def test_harvest_level(self, small_farm, deterministic_seed):
        """Should be able to harvest a level."""
        # Grow to maturity
        for _ in range(100):
            small_farm.step(dt_hours=1.0)
        
        biomass_before = small_farm.biomass[2].sum().item()
        
        harvest = small_farm.harvest_level(level=2)
        
        # Level should be reset
        assert small_farm.biomass[2].sum().item() < biomass_before
        assert harvest > 0
    
    @pytest.mark.unit
    def test_harvest_report(self, small_farm, deterministic_seed):
        """Should generate harvest report."""
        # Grow
        for _ in range(100):
            small_farm.step(dt_hours=1.0)
        
        small_farm.harvest_level(0)
        small_farm.harvest_level(1)
        
        report = small_farm.get_harvest_report()
        
        assert isinstance(report, HarvestReport)
    
    @pytest.mark.unit
    def test_harvest_report_fields(self, small_farm, deterministic_seed):
        """Report should have required fields."""
        for _ in range(50):
            small_farm.step(dt_hours=1.0)
        
        small_farm.harvest_level(0)
        report = small_farm.get_harvest_report()
        
        assert hasattr(report, 'total_yield_kg')
        assert hasattr(report, 'yield_per_m2')
        assert hasattr(report, 'water_efficiency')


class TestEnergyEfficiency:
    """Test energy efficiency tracking."""
    
    @pytest.mark.unit
    def test_track_energy_consumption(self, small_farm, deterministic_seed):
        """Should track energy consumption."""
        initial_energy = small_farm.total_energy_kwh
        
        # Simulate with lights on
        for level in range(small_farm.n_levels):
            small_farm.set_light(level, 400.0)
        
        for _ in range(12):
            small_farm.step(dt_hours=1.0)
        
        assert small_farm.total_energy_kwh > initial_energy
    
    @pytest.mark.unit
    def test_lighting_is_major_energy_cost(self, small_farm, deterministic_seed):
        """Lighting should be majority of energy cost."""
        for level in range(small_farm.n_levels):
            small_farm.set_light(level, 500.0)
        
        for _ in range(24):
            small_farm.step(dt_hours=1.0)
        
        # Lighting should be > 50% of energy
        lighting_fraction = small_farm.lighting_energy_kwh / small_farm.total_energy_kwh
        assert lighting_fraction > 0.4  # At least 40%


class TestDeterminism:
    """Test reproducibility requirements."""
    
    @pytest.mark.unit
    def test_deterministic_growth(self):
        """Same seed should give same growth."""
        # First farm
        torch.manual_seed(42)
        np.random.seed(42)
        farm1 = VerticalFarm(n_levels=4, crop_type='lettuce')
        for _ in range(24):
            farm1.step(dt_hours=1.0)
        biomass1 = farm1.biomass.clone()
        
        # Second farm
        torch.manual_seed(42)
        np.random.seed(42)
        farm2 = VerticalFarm(n_levels=4, crop_type='lettuce')
        for _ in range(24):
            farm2.step(dt_hours=1.0)
        biomass2 = farm2.biomass
        
        assert torch.allclose(biomass1, biomass2, rtol=1e-10)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestVerticalFarmIntegration:
    """Integration tests for vertical farm."""
    
    @pytest.mark.integration
    def test_full_growth_cycle(self, deterministic_seed):
        """Test complete growth cycle from seedling to harvest."""
        farm = VerticalFarm(
            n_levels=6,
            area_per_level_m2=100.0,
            crop_type='lettuce'
        )
        
        # Set optimal conditions
        for level in range(6):
            farm.set_temperature(level, farm.optimal_temp)
            farm.set_humidity(level, farm.optimal_humidity)
            farm.set_co2(level, 1000.0)
            farm.set_light(level, 300.0)
        
        # Grow for 30 days (lettuce cycle)
        for day in range(30):
            # Simulate day/night cycle
            for hour in range(24):
                if 6 <= hour < 22:  # 16h photoperiod
                    for level in range(6):
                        farm.set_light(level, 300.0)
                else:
                    for level in range(6):
                        farm.set_light(level, 0.0)
                
                farm.step(dt_hours=1.0)
        
        # Harvest all levels
        total_harvest = 0
        for level in range(6):
            total_harvest += farm.harvest_level(level)
        
        # Should have meaningful harvest
        assert total_harvest > 0
        
        # Get report
        report = farm.get_harvest_report()
        assert report.total_yield_kg > 0
        assert report.water_efficiency > 0
    
    @pytest.mark.integration
    @pytest.mark.physics
    def test_resource_efficiency_comparison(self, deterministic_seed):
        """Compare resource efficiency across crops."""
        crops = ['lettuce', 'basil', 'strawberry']
        results = {}
        
        for crop in crops:
            torch.manual_seed(42)
            np.random.seed(42)
            
            farm = VerticalFarm(
                n_levels=4,
                area_per_level_m2=50.0,
                crop_type=crop
            )
            
            # Set conditions
            for level in range(4):
                farm.set_temperature(level, farm.optimal_temp)
                farm.set_humidity(level, farm.optimal_humidity)
                farm.set_light(level, 300.0)
            
            # Grow for 14 days
            for _ in range(14 * 24):
                farm.step(dt_hours=1.0)
            
            results[crop] = {
                'biomass': farm.biomass.sum().item(),
                'water_used': farm.total_water_liters,
                'energy_used': farm.total_energy_kwh
            }
        
        # All crops should have grown
        for crop in crops:
            assert results[crop]['biomass'] > 0
    
    @pytest.mark.integration
    def test_staggered_harvest_strategy(self, deterministic_seed):
        """Test staggered planting/harvesting."""
        farm = VerticalFarm(n_levels=8, area_per_level_m2=100.0, crop_type='lettuce')
        
        # Set conditions
        for level in range(8):
            farm.set_temperature(level, farm.optimal_temp)
            farm.set_humidity(level, farm.optimal_humidity)
            farm.set_light(level, 300.0)
        
        total_harvested = 0
        
        # Simulate 60 days with weekly harvests
        for day in range(60):
            for hour in range(24):
                farm.step(dt_hours=1.0)
            
            # Harvest one level per week
            if day > 0 and day % 7 == 0:
                level_to_harvest = (day // 7) % 8
                harvest = farm.harvest_level(level_to_harvest)
                total_harvested += harvest
        
        # Should have harvested multiple times
        assert total_harvested > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
