"""
HyperFOAM Intake - Submitter Unit Tests
=======================================

Article II Compliance:
- Section 2.1: No code merged without passing tests
- Section 2.2: Coverage target 80%+

Tests the SimulationSubmitter which converts Human Units → SI Physics Units.
"""

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from staging.submitter import (
    SimulationSubmitter,
    UnitConverter,
    SolverConstants,
)


class TestUnitConverter:
    """Test unit conversion accuracy."""
    
    @pytest.fixture
    def converter(self):
        return UnitConverter()
    
    # =========================================================================
    # Length conversions
    # =========================================================================
    
    def test_feet_to_meters(self, converter):
        """1 foot = 0.3048 meters."""
        assert converter.feet_to_meters(1.0) == pytest.approx(0.3048, rel=1e-6)
        assert converter.feet_to_meters(10.0) == pytest.approx(3.048, rel=1e-6)
        assert converter.feet_to_meters(0.0) == 0.0
    
    def test_inches_to_meters(self, converter):
        """1 inch = 0.0254 meters."""
        assert converter.inches_to_meters(1.0) == pytest.approx(0.0254, rel=1e-6)
        assert converter.inches_to_meters(24.0) == pytest.approx(0.6096, rel=1e-6)
    
    # =========================================================================
    # Temperature conversions
    # =========================================================================
    
    def test_fahrenheit_to_kelvin_freezing(self, converter):
        """32°F = 273.15 K (freezing point)."""
        assert converter.fahrenheit_to_kelvin(32.0) == pytest.approx(273.15, rel=1e-4)
    
    def test_fahrenheit_to_kelvin_boiling(self, converter):
        """212°F = 373.15 K (boiling point)."""
        assert converter.fahrenheit_to_kelvin(212.0) == pytest.approx(373.15, rel=1e-4)
    
    def test_fahrenheit_to_kelvin_hvac_supply(self, converter):
        """55°F is typical HVAC supply temperature."""
        result = converter.fahrenheit_to_kelvin(55.0)
        # 55°F = 12.78°C = 285.93 K
        assert result == pytest.approx(285.93, rel=1e-3)
    
    def test_celsius_to_kelvin(self, converter):
        """0°C = 273.15 K."""
        assert converter.celsius_to_kelvin(0.0) == pytest.approx(273.15, rel=1e-6)
        assert converter.celsius_to_kelvin(100.0) == pytest.approx(373.15, rel=1e-6)
    
    # =========================================================================
    # Flow conversions
    # =========================================================================
    
    def test_cfm_to_m3s(self, converter):
        """CFM to cubic meters per second."""
        # 1 CFM = 0.000471947 m³/s
        assert converter.cfm_to_m3s(1.0) == pytest.approx(0.000471947, rel=1e-4)
        assert converter.cfm_to_m3s(1000.0) == pytest.approx(0.471947, rel=1e-4)
    
    def test_fpm_to_ms(self, converter):
        """FPM (feet per minute) to m/s."""
        # 1 fpm = 0.00508 m/s
        assert converter.fpm_to_ms(1.0) == pytest.approx(0.00508, rel=1e-4)
        assert converter.fpm_to_ms(500.0) == pytest.approx(2.54, rel=1e-4)
    
    # =========================================================================
    # Energy conversions
    # =========================================================================
    
    def test_btu_hr_to_watts(self, converter):
        """BTU/hr to Watts."""
        # 1 BTU/hr ≈ 0.293 W
        assert converter.btu_hr_to_watts(3412.0) == pytest.approx(1000.0, rel=0.01)
        assert converter.btu_hr_to_watts(0.0) == 0.0
    
    def test_watts_to_btu_hr(self, converter):
        """Watts to BTU/hr."""
        assert converter.watts_to_btu_hr(1000.0) == pytest.approx(3412.0, rel=0.01)


class TestSolverConstants:
    """Test physics constants are correct."""
    
    def test_gravity_vector(self):
        """Gravity is -9.81 m/s² in Y direction."""
        constants = SolverConstants()
        assert constants.GRAVITY == (0, -9.81, 0)
    
    def test_air_density(self):
        """Air density at standard conditions."""
        constants = SolverConstants()
        assert constants.AIR_DENSITY == pytest.approx(1.225, rel=0.01)
    
    def test_atmospheric_pressure(self):
        """Standard atmospheric pressure in Pascals."""
        constants = SolverConstants()
        assert constants.P_ATM == 101325


class TestSimulationSubmitter:
    """Test the main submitter class."""
    
    @pytest.fixture
    def submitter(self):
        return SimulationSubmitter()
    
    @pytest.fixture
    def valid_input(self):
        """Standard valid input for testing."""
        return {
            "project_name": "Test Project",
            "room_name": "Test Room",
            "room_width": 20.0,
            "room_length": 30.0,
            "room_height": 10.0,
            "inlet_cfm": 500.0,
            "supply_temp": 55.0,
            "diffuser_width": 24,
            "diffuser_height": 24,
            "vent_count": 1,
            "heat_load": 1000,
        }
    
    # =========================================================================
    # submit_job() tests
    # =========================================================================
    
    def test_submit_job_returns_dict(self, submitter, valid_input):
        """submit_job returns a dictionary."""
        result = submitter.submit_job(valid_input)
        assert isinstance(result, dict)
    
    def test_submit_job_has_case_id(self, submitter, valid_input):
        """Payload has unique case ID."""
        result = submitter.submit_job(valid_input)
        assert "case_id" in result
        assert result["case_id"].startswith("HF_")
    
    def test_submit_job_unique_case_ids(self, submitter, valid_input):
        """Each submission gets unique case ID."""
        result1 = submitter.submit_job(valid_input)
        result2 = submitter.submit_job(valid_input)
        assert result1["case_id"] != result2["case_id"]
    
    def test_submit_job_has_domain(self, submitter, valid_input):
        """Payload has domain section with SI units."""
        result = submitter.submit_job(valid_input)
        
        assert "domain" in result
        domain = result["domain"]
        assert "width_x_m" in domain
        assert "length_z_m" in domain
        assert "height_y_m" in domain
    
    def test_submit_job_domain_converted_to_meters(self, submitter, valid_input):
        """Domain dimensions converted from feet to meters."""
        result = submitter.submit_job(valid_input)
        domain = result["domain"]
        
        # 20 ft = 6.096 m
        assert domain["width_x_m"] == pytest.approx(6.096, rel=1e-3)
        # 30 ft = 9.144 m
        assert domain["length_z_m"] == pytest.approx(9.144, rel=1e-3)
        # 10 ft = 3.048 m
        assert domain["height_y_m"] == pytest.approx(3.048, rel=1e-3)
    
    def test_submit_job_has_boundary_conditions(self, submitter, valid_input):
        """Payload has boundary conditions."""
        result = submitter.submit_job(valid_input)
        
        assert "boundary_conditions" in result
        bc = result["boundary_conditions"]
        assert "inlet" in bc
        assert "outlet" in bc
        assert "walls" in bc
    
    def test_submit_job_inlet_has_velocity(self, submitter, valid_input):
        """Inlet has velocity vector in m/s."""
        result = submitter.submit_job(valid_input)
        inlet = result["boundary_conditions"]["inlet"]
        
        assert "velocity_vector_ms" in inlet
        assert len(inlet["velocity_vector_ms"]) == 3
        # Y component should be negative (downward from ceiling)
        assert inlet["velocity_vector_ms"][1] < 0
    
    def test_submit_job_inlet_has_temperature_kelvin(self, submitter, valid_input):
        """Inlet temperature is in Kelvin."""
        result = submitter.submit_job(valid_input)
        inlet = result["boundary_conditions"]["inlet"]
        
        assert "temperature_k" in inlet
        # 55°F = 285.93 K
        assert inlet["temperature_k"] == pytest.approx(285.93, rel=1e-3)
    
    def test_submit_job_units_are_si(self, submitter, valid_input):
        """Payload declares SI units."""
        result = submitter.submit_job(valid_input)
        assert result["units"] == "SI"
    
    def test_submit_job_has_solver_settings(self, submitter, valid_input):
        """Payload has solver settings."""
        result = submitter.submit_job(valid_input)
        assert "solver_settings" in result
    
    def test_submit_job_has_heat_sources(self, submitter, valid_input):
        """Heat sources are converted to Watts."""
        result = submitter.submit_job(valid_input)
        
        assert "heat_sources" in result
        assert len(result["heat_sources"]) > 0
        
        heat_source = result["heat_sources"][0]
        assert "power_w" in heat_source
        # 1000 BTU/hr ≈ 293 W
        assert heat_source["power_w"] == pytest.approx(293, rel=0.05)
    
    def test_submit_job_no_heat_sources_when_zero(self, submitter):
        """No heat sources added when heat_load is 0."""
        input_data = {
            "room_width": 20.0,
            "room_length": 30.0,
            "room_height": 10.0,
            "heat_load": 0,
        }
        result = submitter.submit_job(input_data)
        assert len(result["heat_sources"]) == 0
    
    # =========================================================================
    # Input validation tests (Article III, Section 3.4)
    # =========================================================================
    
    def test_submit_job_handles_none_input(self, submitter):
        """None input doesn't crash, uses defaults."""
        result = submitter.submit_job(None)
        
        assert isinstance(result, dict)
        assert "case_id" in result
        assert "domain" in result
    
    def test_submit_job_handles_empty_dict(self, submitter):
        """Empty dict uses defaults."""
        result = submitter.submit_job({})
        
        assert isinstance(result, dict)
        assert "domain" in result
        # Should have reasonable defaults
        assert result["domain"]["width_x_m"] > 0
    
    def test_submit_job_handles_missing_fields(self, submitter):
        """Missing fields use defaults, don't crash."""
        partial_input = {
            "room_width": 15.0,
            # Missing: room_length, room_height, inlet_cfm, etc.
        }
        result = submitter.submit_job(partial_input)
        
        assert isinstance(result, dict)
        assert result["domain"]["width_x_m"] == pytest.approx(4.572, rel=1e-3)  # 15 ft
    
    def test_submit_job_handles_none_values(self, submitter):
        """None values in dict use defaults."""
        input_with_nones = {
            "room_width": 20.0,
            "room_length": None,
            "inlet_cfm": None,
        }
        result = submitter.submit_job(input_with_nones)
        
        assert isinstance(result, dict)
        # Should not crash, should use defaults for None values
    
    def test_submit_job_velocity_from_cfm(self, submitter):
        """Velocity calculated from CFM."""
        input_data = {
            "room_width": 20.0,
            "room_length": 30.0,
            "room_height": 10.0,
            "inlet_cfm": 1000.0,
            "diffuser_width": 24,
            "diffuser_height": 24,
        }
        result = submitter.submit_job(input_data)
        
        inlet = result["boundary_conditions"]["inlet"]
        # Velocity should be calculated from CFM / area
        vel_y = abs(inlet["velocity_vector_ms"][1])
        assert vel_y > 0
    
    def test_submit_job_velocity_from_fpm_when_no_cfm(self, submitter):
        """Use inlet_velocity when inlet_cfm not provided."""
        input_data = {
            "room_width": 20.0,
            "room_length": 30.0,
            "room_height": 10.0,
            "inlet_cfm": None,
            "inlet_velocity": 500.0,  # FPM
        }
        result = submitter.submit_job(input_data)
        
        inlet = result["boundary_conditions"]["inlet"]
        vel_y = abs(inlet["velocity_vector_ms"][1])
        # 500 fpm = 2.54 m/s
        assert vel_y == pytest.approx(2.54, rel=0.1)
    
    # =========================================================================
    # validate_payload() tests
    # =========================================================================
    
    def test_validate_payload_valid(self, submitter, valid_input):
        """Valid payload passes validation."""
        payload = submitter.submit_job(valid_input)
        validation = submitter.validate_payload(payload)
        
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
    
    def test_validate_payload_too_small_room(self, submitter):
        """Very small room triggers error."""
        input_data = {
            "room_width": 1.0,   # 1 ft = 0.3m < 0.5m minimum
            "room_length": 1.0,
            "room_height": 3.0,
        }
        payload = submitter.submit_job(input_data)
        validation = submitter.validate_payload(payload)
        
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
    
    def test_validate_payload_warns_high_velocity(self, submitter):
        """Very high velocity triggers warning."""
        input_data = {
            "room_width": 10.0,
            "room_length": 10.0,
            "room_height": 8.0,
            "inlet_cfm": 10000.0,  # Very high for small room
        }
        payload = submitter.submit_job(input_data)
        validation = submitter.validate_payload(payload)
        
        # Should have warning about high velocity
        assert len(validation["warnings"]) > 0
    
    def test_validate_payload_returns_structure(self, submitter, valid_input):
        """Validation result has correct structure."""
        payload = submitter.submit_job(valid_input)
        validation = submitter.validate_payload(payload)
        
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
        assert isinstance(validation["errors"], list)
        assert isinstance(validation["warnings"], list)
    
    # =========================================================================
    # to_json() tests
    # =========================================================================
    
    def test_to_json_serializable(self, submitter, valid_input):
        """Payload serializes to valid JSON."""
        payload = submitter.submit_job(valid_input)
        json_str = submitter.to_json(payload)
        
        assert isinstance(json_str, str)
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["case_id"] == payload["case_id"]
    
    def test_to_json_pretty_print(self, submitter, valid_input):
        """to_json with indent produces readable output."""
        payload = submitter.submit_job(valid_input)
        json_str = submitter.to_json(payload, indent=2)
        
        assert "\n" in json_str
        assert "  " in json_str


class TestGridCalculation:
    """Test mesh grid calculations."""
    
    @pytest.fixture
    def submitter(self):
        return SimulationSubmitter()
    
    def test_grid_resolution_calculated(self, submitter):
        """Grid resolution is calculated based on room size."""
        input_data = {
            "room_width": 20.0,
            "room_length": 30.0,
            "room_height": 10.0,
        }
        result = submitter.submit_job(input_data)
        
        domain = result["domain"]
        assert "grid_resolution" in domain
        assert len(domain["grid_resolution"]) == 3
    
    def test_total_cells_calculated(self, submitter):
        """Total cell count is calculated."""
        input_data = {
            "room_width": 20.0,
            "room_length": 30.0,
            "room_height": 10.0,
        }
        result = submitter.submit_job(input_data)
        
        domain = result["domain"]
        assert "total_cells" in domain
        
        # Total cells = nx * ny * nz
        grid = domain["grid_resolution"]
        expected_cells = grid[0] * grid[1] * grid[2]
        assert domain["total_cells"] == expected_cells


class TestEndToEnd:
    """End-to-end workflow tests."""
    
    def test_realistic_office_simulation(self):
        """Complete realistic office scenario."""
        submitter = SimulationSubmitter()
        
        # Typical small office parameters
        office_input = {
            "project_name": "Small Office CFD",
            "room_name": "Private Office",
            "room_width": 12.0,    # 12 ft
            "room_length": 15.0,   # 15 ft
            "room_height": 9.0,    # 9 ft ceiling
            "inlet_cfm": 150.0,    # Typical for 1-2 person office
            "supply_temp": 55.0,   # Standard supply temp
            "diffuser_width": 24,  # 24" diffuser
            "diffuser_height": 24,
            "vent_count": 1,
            "heat_load": 500,      # ~1 person + computer
        }
        
        payload = submitter.submit_job(office_input)
        validation = submitter.validate_payload(payload)
        
        # Should be completely valid
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        
        # Verify physics make sense
        inlet = payload["boundary_conditions"]["inlet"]
        vel_y = abs(inlet["velocity_vector_ms"][1])
        
        # Velocity should be reasonable (1-10 m/s typical)
        assert 0.5 < vel_y < 15.0
        
        # Temperature should be ~286 K
        assert 280 < inlet["temperature_k"] < 300


# =============================================================================
# BATCH PAYLOAD TESTS
# =============================================================================

class TestBatchPayloadGeneration:
    """
    Tests for multi-room batch payload generation.
    
    Article II §2.1: Coverage for new features.
    """
    
    @pytest.fixture
    def submitter(self):
        return SimulationSubmitter()
    
    @pytest.fixture
    def multi_room_result(self):
        """Simulated multi-room parse result."""
        return {
            "success": True,
            "fields": {
                "project_name": {"value": "Test Building"},
                "supply_temp": {"value": 55.0},
                "diffuser_width": {"value": 24},
                "diffuser_height": {"value": 24},
            },
            "multi_room": {
                "enabled": True,
                "room_count": 3,
                "rooms": [
                    {
                        "room_name": "Conference A",
                        "width_ft": 30.0,
                        "length_ft": 25.0,
                        "height_ft": 10.0,
                        "airflow_cfm": 1500.0,
                    },
                    {
                        "room_name": "Conference B",
                        "width_ft": 25.0,
                        "length_ft": 20.0,
                        "height_ft": 10.0,
                        "airflow_cfm": 1200.0,
                    },
                    {
                        "room_name": "Executive Office",
                        "width_ft": 15.0,
                        "length_ft": 12.0,
                        "height_ft": 9.0,
                        "airflow_cfm": 600.0,
                    },
                ],
                "selected_index": 0,
            },
        }
    
    def test_submit_batch_all_rooms(self, submitter, multi_room_result):
        """Batch submit generates payload for all rooms."""
        result = submitter.submit_batch(multi_room_result)
        
        assert result["success"] is True
        assert result["room_count"] == 3
        assert len(result["payloads"]) == 3
        assert "batch_id" in result
    
    def test_submit_batch_specific_rooms(self, submitter, multi_room_result):
        """Batch submit can process specific rooms by index."""
        result = submitter.submit_batch(multi_room_result, room_indices=[0, 2])
        
        assert result["success"] is True
        assert result["room_count"] == 2
        room_names = [p["room_name"] for p in result["payloads"]]
        assert "Conference A" in room_names
        assert "Executive Office" in room_names
        assert "Conference B" not in room_names
    
    def test_submit_batch_payload_structure(self, submitter, multi_room_result):
        """Each batch payload has correct structure."""
        result = submitter.submit_batch(multi_room_result)
        
        for payload_info in result["payloads"]:
            assert "room_name" in payload_info
            assert "case_id" in payload_info
            assert "payload" in payload_info
            assert "valid" in payload_info
            
            # The payload itself should have domain, boundary_conditions, etc.
            payload = payload_info["payload"]
            assert "domain" in payload
            assert "boundary_conditions" in payload
            assert "units" in payload
            assert payload["units"] == "SI"
    
    def test_submit_batch_unique_case_ids(self, submitter, multi_room_result):
        """Each room in batch gets unique case_id."""
        result = submitter.submit_batch(multi_room_result)
        
        case_ids = [p["case_id"] for p in result["payloads"]]
        assert len(case_ids) == len(set(case_ids))  # All unique
    
    def test_submit_batch_empty_rooms(self, submitter):
        """Batch with no rooms returns error."""
        empty_result = {"multi_room": {"rooms": []}}
        result = submitter.submit_batch(empty_result)
        
        assert result["success"] is False
        assert len(result["errors"]) > 0
    
    def test_submit_batch_invalid_indices_filtered(self, submitter, multi_room_result):
        """Invalid room indices are filtered out gracefully."""
        result = submitter.submit_batch(multi_room_result, room_indices=[0, 999, -1])
        
        # Should only process index 0 (valid)
        assert result["room_count"] == 1
        assert result["payloads"][0]["room_name"] == "Conference A"
