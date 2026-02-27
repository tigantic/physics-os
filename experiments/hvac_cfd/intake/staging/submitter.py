"""
Simulation Submitter
====================

The "Submit" action bridge.

Takes user-validated UI data (in Feet/CFM) and converts it to 
SI Physics Payload (Meters/Pascals) for the HyperTensor solver.

FLOW:
-----
    1. User validates Yellow/Red fields in Staging Area
    2. All fields turn Green (validated)
    3. User clicks "Submit"
    4. This module converts Human Units → Physics Units
    5. Clean JSON sent to HyperTensor solver

PUBLIC API (Article V, Section 5.1):
------------------------------------
    SimulationSubmitter.submit_job(validated_ui_data: Dict) -> Dict
        Main entry point. Converts Human Units to SI Physics Units.
        
    SimulationSubmitter.validate_payload(payload: Dict) -> Dict
        Validates the solver payload before submission.
        Returns {"valid": bool, "errors": [...], "warnings": [...]}
        
    SimulationSubmitter.to_json(payload: Dict, indent: int = 2) -> str
        Serialize payload to JSON string.

INPUT VALIDATION (Article III, Section 3.4):
--------------------------------------------
    All inputs are validated at the boundary. Invalid/missing values
    use safe defaults with warnings, never crash.

UNIT CONVERSIONS:
-----------------
    Length: feet → meters (×0.3048), inches → meters (×0.0254)
    Temperature: °F → K ((F-32)×5/9+273.15)
    Flow: CFM → m³/s (×0.000471947), FPM → m/s (×0.00508)
    Energy: BTU/hr → W (÷3.412)

CONSTITUTION COMPLIANCE:
------------------------
    - Article III, Section 3.1: Defined IPC protocol (JSON payload)
    - Article III, Section 3.2: Graceful handling of bad inputs
    - Article III, Section 3.4: All inputs validated at boundary
    - Article V, Section 5.1: All public APIs documented
"""

import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

from .logger import get_logger

# Module logger
logger = get_logger(__name__)


@dataclass
class SolverConstants:
    """Physics constants for CFD simulation."""
    GRAVITY: tuple = (0, -9.81, 0)  # Y-axis up/down
    AIR_DENSITY: float = 1.225  # kg/m³ (standard at 15°C)
    P_ATM: float = 101325  # Pascals
    AIR_VISCOSITY: float = 1.81e-5  # Pa·s
    AIR_THERMAL_CONDUCTIVITY: float = 0.026  # W/(m·K)
    AIR_SPECIFIC_HEAT: float = 1006  # J/(kg·K)
    PRANDTL: float = 0.71


class UnitConverter:
    """Unit conversion utilities."""
    
    # Length
    @staticmethod
    def feet_to_meters(ft: float) -> float:
        return ft * 0.3048
    
    @staticmethod
    def inches_to_meters(inches: float) -> float:
        return inches * 0.0254
    
    # Temperature
    @staticmethod
    def fahrenheit_to_kelvin(f: float) -> float:
        return (f - 32) * (5/9) + 273.15
    
    @staticmethod
    def celsius_to_kelvin(c: float) -> float:
        return c + 273.15
    
    # Flow
    @staticmethod
    def cfm_to_m3s(cfm: float) -> float:
        """Convert cubic feet per minute to cubic meters per second."""
        return cfm * 0.000471947
    
    @staticmethod
    def fpm_to_ms(fpm: float) -> float:
        """Convert feet per minute to meters per second."""
        return fpm * 0.00508
    
    # Energy
    @staticmethod
    def btu_hr_to_watts(btu_hr: float) -> float:
        return btu_hr / 3.412
    
    @staticmethod
    def watts_to_btu_hr(watts: float) -> float:
        return watts * 3.412


class SimulationSubmitter:
    """
    Converts validated UI data to solver-ready JSON payload.
    
    This is the gateway between the user-friendly form and the physics engine.
    """
    
    def __init__(self):
        self.constants = SolverConstants()
        self.converter = UnitConverter()
    
    def submit_job(self, validated_ui_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for generating solver payloads.
        
        Takes the 'Clean' UI dictionary (user has verified all numbers).
        Returns the strict JSON payload for the Solver.
        
        Args:
            validated_ui_data: Dict with keys like:
                - project_name: str
                - room_width: float (feet)
                - room_length: float (feet)
                - room_height: float (feet)
                - inlet_cfm: float (CFM) - OR - inlet_velocity: float (FPM)
                - supply_temp: float (Fahrenheit)
                - diffuser_width: float (inches)
                - diffuser_height: float (inches)
                - vent_count: int
                - heat_load: float (BTU/hr)
        
        Returns:
            Solver-ready JSON payload with SI units.
            Payload is always valid and usable - missing values use safe defaults.
        
        Raises:
            No exceptions raised - per Article III, Section 3.2 (graceful failure).
        """
        # =================================================================
        # INPUT VALIDATION (Article III, Section 3.4)
        # Trust nothing from the wire. Validate everything.
        # =================================================================
        from .sanitize import sanitize_project_name, sanitize_room_name
        
        start_time = time.time()
        
        if validated_ui_data is None:
            validated_ui_data = {}
        
        if not isinstance(validated_ui_data, dict):
            validated_ui_data = {}
        
        logger.info("Generating solver payload", extra={
            "project_name": validated_ui_data.get("project_name", "Unnamed"),
            "room_width_ft": validated_ui_data.get("room_width"),
            "cfm": validated_ui_data.get("inlet_cfm"),
        })
        
        # Generate unique case ID with microseconds for uniqueness
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        
        # Sanitize string inputs (Article VII, Section 7.3 - no security shortcuts)
        raw_project_name = validated_ui_data.get("project_name", "Unnamed")
        project_name = sanitize_project_name(str(raw_project_name) if raw_project_name else "Unnamed")
        
        raw_room_name = validated_ui_data.get("room_name", "Main Room")
        room_name = sanitize_room_name(str(raw_room_name) if raw_room_name else "Main Room")
        
        # Include microseconds in hash to ensure uniqueness even in rapid succession
        unique_seed = f"{project_name}{timestamp}{now.microsecond}"
        case_hash = hashlib.md5(unique_seed.encode()).hexdigest()[:8]
        case_id = f"HF_{timestamp}_{case_hash}"
        
        # Initialize payload structure
        solver_payload = {
            "case_id": case_id,
            "project_name": project_name,
            "room_name": room_name,
            "created_at": datetime.now().isoformat(),
            "units": "SI",
            "domain": {},
            "boundary_conditions": {},
            "heat_sources": [],
            "solver_settings": {
                "gravity_vector": list(self.constants.GRAVITY),
                "turbulence_model": "k-epsilon-realizable",
                "pressure_velocity_coupling": "SIMPLE",
                "time_stepping": "steady",
                "convergence_criteria": 1e-5,
            },
            "fluid_properties": {
                "density_kg_m3": self.constants.AIR_DENSITY,
                "viscosity_pa_s": self.constants.AIR_VISCOSITY,
                "thermal_conductivity_w_mk": self.constants.AIR_THERMAL_CONDUCTIVITY,
                "specific_heat_j_kgk": self.constants.AIR_SPECIFIC_HEAT,
                "prandtl_number": self.constants.PRANDTL,
            }
        }
        
        # =====================================================================
        # 1. GEOMETRY CONVERSION (Feet → Meters)
        # =====================================================================
        
        # Helper to handle None values (Article III, Section 3.4)
        def safe_get(key: str, default):
            """Get value from dict, using default if None or missing."""
            val = validated_ui_data.get(key)
            return default if val is None else val
        
        width_ft = safe_get("room_width", 20)
        length_ft = safe_get("room_length", 15)
        height_ft = safe_get("room_height", 9)
        
        domain_si = {
            "width_x_m": round(self.converter.feet_to_meters(width_ft), 4),
            "length_z_m": round(self.converter.feet_to_meters(length_ft), 4),
            "height_y_m": round(self.converter.feet_to_meters(height_ft), 4),
        }
        
        # Calculate volume and cell count
        volume_m3 = domain_si["width_x_m"] * domain_si["length_z_m"] * domain_si["height_y_m"]
        
        # Auto-calculate grid resolution (target ~0.1m cells for small rooms, ~0.2m for large)
        avg_dim = (domain_si["width_x_m"] + domain_si["length_z_m"] + domain_si["height_y_m"]) / 3
        cell_size = 0.1 if avg_dim < 5 else 0.15 if avg_dim < 10 else 0.2
        
        nx = max(16, int(domain_si["width_x_m"] / cell_size))
        ny = max(16, int(domain_si["height_y_m"] / cell_size))
        nz = max(16, int(domain_si["length_z_m"] / cell_size))
        
        domain_si["volume_m3"] = round(volume_m3, 3)
        domain_si["grid_resolution"] = [nx, ny, nz]
        domain_si["total_cells"] = nx * ny * nz
        domain_si["cell_size_m"] = round(cell_size, 3)
        
        solver_payload["domain"] = domain_si
        
        # =====================================================================
        # 2. INLET BOUNDARY CONDITION (The Hard Part: CFM → Velocity)
        # =====================================================================
        
        # Helper to get value with fallback (handles explicit None values)
        def safe_get(key, default):
            val = validated_ui_data.get(key)
            return default if val is None else val
        
        temp_f = safe_get("supply_temp", 55)
        vent_count = safe_get("vent_count", 1)
        
        # Get diffuser dimensions (default 24x24 inch)
        diff_w_in = safe_get("diffuser_width", 24)
        diff_h_in = safe_get("diffuser_height", 24)
        
        # Diffuser area calculation (needed before CFM/velocity)
        diff_w_m = self.converter.inches_to_meters(diff_w_in)
        diff_h_m = self.converter.inches_to_meters(diff_h_in)
        face_area_m2 = diff_w_m * diff_h_m
        
        # Apply "Free Area Factor" (standard plaque diffuser is ~10-20% open)
        free_area_ratio = safe_get("free_area_ratio", 0.15)
        effective_area_m2 = face_area_m2 * free_area_ratio * vent_count
        
        # -----------------------------------------------------------------
        # CFM/Velocity Resolution Logic:
        #   - If CFM provided: use it, calculate velocity
        #   - If only Velocity provided: back-calculate CFM
        #   - If neither: use defaults
        # -----------------------------------------------------------------
        cfm = validated_ui_data.get("inlet_cfm")
        inlet_velocity_fpm = validated_ui_data.get("inlet_velocity")
        
        if cfm is not None:
            # CFM provided - primary path
            flow_rate_m3s = self.converter.cfm_to_m3s(cfm)
            if effective_area_m2 > 0:
                velocity_mag_ms = flow_rate_m3s / effective_area_m2
            else:
                velocity_mag_ms = 2.0
        elif inlet_velocity_fpm is not None:
            # Velocity provided (FPM) - reverse calculate CFM
            velocity_mag_ms = self.converter.fpm_to_ms(inlet_velocity_fpm)
            flow_rate_m3s = velocity_mag_ms * effective_area_m2
            cfm = flow_rate_m3s / 0.000471947  # Convert back to CFM for reference
        else:
            # Neither provided - use defaults
            cfm = 250  # Typical small office
            flow_rate_m3s = self.converter.cfm_to_m3s(cfm)
            if effective_area_m2 > 0:
                velocity_mag_ms = flow_rate_m3s / effective_area_m2
            else:
                velocity_mag_ms = 2.0
        
        temp_k = self.converter.fahrenheit_to_kelvin(temp_f)
        
        # Auto-center diffuser(s) on ceiling
        inlet_center = [
            domain_si["width_x_m"] / 2,
            domain_si["height_y_m"],  # Top of domain (ceiling)
            domain_si["length_z_m"] / 2,
        ]
        
        solver_payload["boundary_conditions"]["inlet"] = {
            "type": "velocity_inlet",
            "location_center_m": inlet_center,
            "dimensions_m": [diff_w_m, diff_h_m],
            "velocity_vector_ms": [0, -round(velocity_mag_ms, 3), 0],  # Negative Y = Down
            "temperature_k": round(temp_k, 2),
            "flow_rate_m3s": round(flow_rate_m3s, 6),
            "vent_count": vent_count,
            "effective_area_m2": round(effective_area_m2, 6),
            # Original values for reference
            "_original": {
                "cfm": cfm,
                "temp_f": temp_f,
                "diffuser_in": [diff_w_in, diff_h_in],
            }
        }
        
        # =====================================================================
        # 3. OUTLET BOUNDARY CONDITION (Pressure Outlet)
        # =====================================================================
        
        # Default: return on opposite wall at floor level
        outlet_center = [
            domain_si["width_x_m"] / 2,
            0.3,  # Near floor
            0.0,  # On the back wall
        ]
        
        solver_payload["boundary_conditions"]["outlet"] = {
            "type": "pressure_outlet",
            "location_center_m": outlet_center,
            "dimensions_m": [0.3, 0.15],  # Standard return grille
            "gauge_pressure_pa": 0,
        }
        
        # =====================================================================
        # 4. WALL BOUNDARY CONDITIONS
        # =====================================================================
        
        solver_payload["boundary_conditions"]["walls"] = {
            "floor": {"type": "wall", "thermal": "adiabatic"},
            "ceiling": {"type": "wall", "thermal": "adiabatic"},
            "north": {"type": "wall", "thermal": "adiabatic"},
            "south": {"type": "wall", "thermal": "adiabatic"},
            "east": {"type": "wall", "thermal": "adiabatic"},
            "west": {"type": "wall", "thermal": "adiabatic"},
        }
        
        # =====================================================================
        # 5. HEAT SOURCES
        # =====================================================================
        
        heat_load_btu = validated_ui_data.get("heat_load", 0)
        if heat_load_btu > 0:
            heat_load_w = self.converter.btu_hr_to_watts(heat_load_btu)
            
            # Place heat source at floor center (simulates occupant/equipment)
            solver_payload["heat_sources"].append({
                "name": "Internal Load",
                "type": "volumetric",
                "location_center_m": [
                    domain_si["width_x_m"] / 2,
                    0.5,  # Near floor
                    domain_si["length_z_m"] / 2,
                ],
                "power_w": round(heat_load_w, 1),
                "_original_btu_hr": heat_load_btu,
            })
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info("Solver payload generated", extra={
            "case_id": solver_payload["case_id"],
            "total_cells": solver_payload.get("domain", {}).get("total_cells", 0),
            "duration_ms": round(duration_ms, 2),
        })
        
        return solver_payload
    
    def validate_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the solver payload before submission.
        
        Returns dict with 'valid' bool and 'errors' list.
        """
        errors = []
        warnings = []
        
        # Check domain
        domain = payload.get("domain", {})
        if domain.get("width_x_m", 0) < 0.5:
            errors.append("Room width too small (< 0.5m)")
        if domain.get("length_z_m", 0) < 0.5:
            errors.append("Room length too small (< 0.5m)")
        if domain.get("height_y_m", 0) < 1.5:
            errors.append("Room height too small (< 1.5m)")
        if domain.get("total_cells", 0) > 10_000_000:
            warnings.append("Very fine grid (>10M cells). Simulation may be slow.")
        
        # Check inlet
        inlet = payload.get("boundary_conditions", {}).get("inlet", {})
        velocity = inlet.get("velocity_vector_ms", [0, 0, 0])
        vel_mag = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2) ** 0.5
        
        # Zero airflow is an error - nothing to simulate
        if vel_mag < 0.01:
            errors.append("No airflow (CFM ≈ 0). Enter a valid CFM value to run simulation.")
        elif vel_mag < 0.1:
            warnings.append("Very low inlet velocity (<0.1 m/s)")
        if vel_mag > 20:
            warnings.append("Very high inlet velocity (>20 m/s). Check CFM value.")
        
        temp_k = inlet.get("temperature_k", 0)
        if temp_k < 273 or temp_k > 320:
            warnings.append(f"Unusual supply temperature: {temp_k:.1f}K")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    def submit_batch(self, multi_room_result: Dict[str, Any], 
                     room_indices: list = None) -> Dict[str, Any]:
        """
        Generate solver payloads for multiple rooms from a multi-room parse result.
        
        Args:
            multi_room_result: Parser result containing multi_room.rooms list
            room_indices: Optional list of room indices to process.
                         If None, processes all rooms.
        
        Returns:
            {
                "success": bool,
                "batch_id": str,
                "room_count": int,
                "payloads": [
                    {"room_name": str, "case_id": str, "payload": Dict, "valid": bool},
                    ...
                ],
                "errors": [...]
            }
        """
        start_time = time.time()
        
        multi_room = multi_room_result.get("multi_room", {})
        rooms = multi_room.get("rooms", [])
        
        if not rooms:
            return {
                "success": False,
                "batch_id": None,
                "room_count": 0,
                "payloads": [],
                "errors": ["No rooms found in parse result"],
            }
        
        # Determine which rooms to process
        if room_indices is None:
            room_indices = list(range(len(rooms)))
        else:
            # Validate indices
            room_indices = [i for i in room_indices if 0 <= i < len(rooms)]
        
        # Generate batch ID
        now = datetime.now()
        batch_id = f"BATCH_{now.strftime('%Y%m%d_%H%M%S')}_{len(room_indices)}rooms"
        
        # Base UI data from the parse result
        base_fields = multi_room_result.get("fields", {})
        base_ui_data = {}
        for field_name, field_data in base_fields.items():
            if isinstance(field_data, dict):
                base_ui_data[field_name] = field_data.get("value")
            else:
                base_ui_data[field_name] = field_data
        
        results = []
        errors = []
        
        for idx in room_indices:
            room = rooms[idx]
            
            # Build UI data for this specific room
            room_ui_data = base_ui_data.copy()
            
            # Override with room-specific values
            field_mapping = {
                "room_name": "room_name",
                "width_ft": "room_width",
                "length_ft": "room_length",
                "height_ft": "room_height",
                "airflow_cfm": "inlet_cfm",
                "supply_temp_f": "supply_temp",
                "velocity_fpm": "inlet_velocity",
                "vent_count": "vent_count",
                "heat_load": "heat_load",
            }
            
            for room_field, ui_field in field_mapping.items():
                if room_field in room and room[room_field] is not None:
                    room_ui_data[ui_field] = room[room_field]
            
            try:
                # Generate payload for this room
                payload = self.submit_job(room_ui_data)
                validation = self.validate_payload(payload)
                
                results.append({
                    "room_name": room.get("room_name", f"Room {idx + 1}"),
                    "room_index": idx,
                    "case_id": payload.get("case_id"),
                    "payload": payload,
                    "valid": validation.get("valid", False),
                    "warnings": validation.get("warnings", []),
                })
            except Exception as e:
                errors.append(f"Room {idx + 1}: {str(e)}")
        
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info("Batch payload generation complete", extra={
            "batch_id": batch_id,
            "room_count": len(results),
            "errors": len(errors),
            "duration_ms": round(duration_ms, 2),
        })
        
        return {
            "success": len(errors) == 0,
            "batch_id": batch_id,
            "room_count": len(results),
            "payloads": results,
            "errors": errors,
        }
    
    def to_json(self, payload: Dict[str, Any], indent: int = 2) -> str:
        """Serialize payload to JSON string."""
        return json.dumps(payload, indent=indent)


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    submitter = SimulationSubmitter()
    
    # Example validated user input (after Green validation)
    final_user_input = {
        "project_name": "Executive Office 404",
        "room_name": "Corner Office",
        "room_width": 12.0,     # feet
        "room_length": 15.0,    # feet
        "room_height": 9.0,     # feet
        "inlet_cfm": 250.0,     # CFM
        "supply_temp": 55.0,    # Fahrenheit
        "diffuser_width": 24,   # inches
        "diffuser_height": 24,  # inches
        "vent_count": 1,
        "heat_load": 500,       # BTU/hr (1 person + laptop)
    }
    
    # Generate solver payload
    payload = submitter.submit_job(final_user_input)
    
    # Validate
    validation = submitter.validate_payload(payload)
    print("Validation:", validation)
    print()
    
    # Print payload
    print(submitter.to_json(payload))
