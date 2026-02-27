"""
Job Spec Generator
==================

Converts validated intake form data to job_spec.json format
for the HyperFOAM CFD solver.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

try:
    from .schema import IntakeSchema, FieldCategory
    from .units import (
        UnitConverter, UnitSystem, LengthUnit, TemperatureUnit, 
        AirflowUnit, ProjectUnits
    )
except ImportError:
    from schema import IntakeSchema, FieldCategory
    from units import (
        UnitConverter, UnitSystem, LengthUnit, TemperatureUnit, 
        AirflowUnit, ProjectUnits
    )


class JobSpecGenerator:
    """
    Generates job_spec.json from intake form data.
    
    Handles:
    - Unit conversion to solver's internal units (metric SI)
    - Default value injection
    - Geometry file references
    - Validation and error reporting
    """
    
    def __init__(self, schema: Optional[IntakeSchema] = None):
        self.schema = schema or IntakeSchema()
    
    def generate(self, form_data: Dict[str, Any], 
                 output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate job_spec.json from form data.
        
        Args:
            form_data: Dictionary of field values from intake form
            output_path: Optional path to write JSON file
            
        Returns:
            Complete job_spec dictionary
        """
        # Determine unit system
        unit_system = UnitSystem.IMPERIAL
        if form_data.get("unit_system") == "metric":
            unit_system = UnitSystem.METRIC
        
        project_units = (ProjectUnits.metric() if unit_system == UnitSystem.METRIC 
                        else ProjectUnits.imperial())
        
        # Build job spec structure
        job_spec = {
            "version": "2.0",
            "generated_by": "HyperFOAM Universal Intake",
            "generated_at": datetime.now().isoformat(),
            "checksum": None,  # Will be computed at end
            
            "project": self._build_project_section(form_data),
            "geometry": self._build_geometry_section(form_data, project_units),
            "hvac": self._build_hvac_section(form_data, project_units),
            "sources": self._build_sources_section(form_data, project_units),
            "loads": self._build_loads_section(form_data, project_units),
            "grid": self._build_grid_section(form_data),
            "solver": self._build_solver_section(form_data),
            "targets": self._build_targets_section(form_data, project_units),
            "compliance": self._build_compliance_section(form_data),
        }
        
        # Compute checksum
        spec_str = json.dumps(job_spec, sort_keys=True, default=str)
        job_spec["checksum"] = hashlib.sha256(spec_str.encode()).hexdigest()[:16]
        
        # Write to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(job_spec, f, indent=2, default=str)
        
        return job_spec
    
    def _build_project_section(self, data: Dict) -> Dict:
        """Build project metadata section."""
        return {
            "name": data.get("project_name", "Unnamed Project"),
            "description": data.get("project_description", ""),
            "client": data.get("client_name", ""),
            "number": data.get("project_number", ""),
            "room_name": data.get("room_name", "Main Zone"),
            "created": datetime.now().isoformat(),
        }
    
    def _build_geometry_section(self, data: Dict, units: ProjectUnits) -> Dict:
        """Build geometry section with unit conversion."""
        # Get dimensions in user's units
        length = data.get("room_length", 0)
        width = data.get("room_width", 0)
        height = data.get("room_height", 0)
        
        # Convert to meters for solver
        length_m = self._convert_length_to_meters(length, data.get("length_unit", "ft"))
        width_m = self._convert_length_to_meters(width, data.get("length_unit", "ft"))
        height_m = self._convert_length_to_meters(height, data.get("length_unit", "ft"))
        
        geometry = {
            "type": "box",  # Default to box geometry
            "dimensions": {
                "length": length_m,
                "width": width_m,
                "height": height_m,
            },
            "units": "meters",
            "original_units": data.get("length_unit", "ft"),
            "original_dimensions": {
                "length": length,
                "width": width,
                "height": height,
            },
        }
        
        # Add floor area and volume
        geometry["floor_area_m2"] = length_m * width_m
        geometry["volume_m3"] = length_m * width_m * height_m
        
        # Add geometry file if provided
        if data.get("geometry_file"):
            geometry["type"] = "file"
            geometry["file_path"] = str(data["geometry_file"])
        
        # Add blueprint scale if provided
        if data.get("blueprint_scale"):
            geometry["blueprint_scale"] = data["blueprint_scale"]
        
        return geometry
    
    def _build_hvac_section(self, data: Dict, units: ProjectUnits) -> Dict:
        """Build HVAC system configuration."""
        # Convert supply airflow to m³/s
        supply_cfm = data.get("supply_airflow", 0)
        supply_cms = self._convert_airflow_to_cms(supply_cfm, data.get("unit_system"))
        
        # Convert temperature to Celsius
        supply_temp = data.get("supply_temperature", 55)
        supply_temp_c = self._convert_temp_to_celsius(
            supply_temp, 
            "F" if data.get("unit_system") != "metric" else "C"
        )
        
        hvac = {
            "system_type": data.get("hvac_system_type", "vav"),
            "supply": {
                "diffuser_count": data.get("vent_count", 1),
                "diffuser_type": data.get("diffuser_type", "ceiling_4way"),
                "total_airflow_m3s": supply_cms,
                "total_airflow_cfm": supply_cfm,
                "temperature_c": supply_temp_c,
                "temperature_f": supply_temp if data.get("unit_system") != "metric" else supply_temp_c * 9/5 + 32,
            },
            "return": {
                "grille_count": data.get("return_count", 1),
            },
        }
        
        # Per-diffuser airflow
        if data.get("vent_count", 0) > 0:
            hvac["supply"]["per_diffuser_m3s"] = supply_cms / data["vent_count"]
            hvac["supply"]["per_diffuser_cfm"] = supply_cfm / data["vent_count"]
        
        return hvac
    
    def _build_sources_section(self, data: Dict, units: ProjectUnits) -> Dict:
        """Build heat sources section."""
        sources = {
            "occupants": {
                "count": data.get("occupancy", 0),
                "activity_level": data.get("occupant_activity", "office"),
                "heat_per_person_w": self._get_metabolic_rate(data.get("occupant_activity", "office")),
            },
            "lighting": {
                "power_density_w_m2": data.get("lighting_load", 0) * (10.764 if data.get("unit_system") != "metric" else 1),
            },
            "equipment": {
                "power_density_w_m2": data.get("equipment_load", 0) * (10.764 if data.get("unit_system") != "metric" else 1),
            },
        }
        
        if data.get("solar_gain"):
            sources["solar"] = {
                "peak_gain_w": data["solar_gain"],
            }
        
        return sources
    
    def _build_loads_section(self, data: Dict, units: ProjectUnits) -> Dict:
        """Calculate total thermal loads."""
        # Get geometry for calculations
        length_m = self._convert_length_to_meters(
            data.get("room_length", 0), 
            data.get("length_unit", "ft")
        )
        width_m = self._convert_length_to_meters(
            data.get("room_width", 0),
            data.get("length_unit", "ft")
        )
        floor_area = length_m * width_m
        
        # Calculate individual loads
        occupant_load = (data.get("occupancy", 0) * 
                        self._get_metabolic_rate(data.get("occupant_activity", "office")))
        
        lighting_density = data.get("lighting_load", 0)
        if data.get("unit_system") != "metric":
            lighting_density *= 10.764  # W/ft² to W/m²
        lighting_load = lighting_density * floor_area
        
        equipment_density = data.get("equipment_load", 0)
        if data.get("unit_system") != "metric":
            equipment_density *= 10.764
        equipment_load = equipment_density * floor_area
        
        solar_load = data.get("solar_gain", 0)
        
        total_load = occupant_load + lighting_load + equipment_load + solar_load
        
        return {
            "occupant_w": occupant_load,
            "lighting_w": lighting_load,
            "equipment_w": equipment_load,
            "solar_w": solar_load,
            "total_sensible_w": total_load,
            "total_sensible_btu_hr": total_load * 3.412,
        }
    
    def _build_grid_section(self, data: Dict) -> Dict:
        """Build computational grid settings."""
        resolution = data.get("grid_resolution", "medium")
        
        cell_counts = {
            "coarse": 100000,
            "medium": 500000,
            "fine": 2000000,
            "very_fine": 10000000,
        }
        
        return {
            "resolution": resolution,
            "target_cells": cell_counts.get(resolution, 500000),
            "auto_refine": True,
            "boundary_layers": 5,
        }
    
    def _build_solver_section(self, data: Dict) -> Dict:
        """Build CFD solver settings."""
        return {
            "turbulence_model": data.get("turbulence_model", "k_epsilon"),
            "steady_state": data.get("steady_state", True),
            "max_iterations": 5000 if data.get("steady_state", True) else None,
            "time_step": 0.1 if not data.get("steady_state", True) else None,
            "end_time": data.get("simulation_time", 300) if not data.get("steady_state", True) else None,
            "convergence_criteria": {
                "residual_target": 1e-6,
                "monitor_interval": 100,
            },
            "schemes": {
                "gradient": "Gauss linear",
                "divergence": "bounded Gauss linearUpwind grad",
                "laplacian": "Gauss linear corrected",
            },
        }
    
    def _build_targets_section(self, data: Dict, units: ProjectUnits) -> Dict:
        """Build target conditions and setpoints."""
        # Convert temperatures
        cooling_setpoint = data.get("indoor_setpoint_cooling", 75)
        heating_setpoint = data.get("indoor_setpoint_heating", 70)
        
        cooling_c = self._convert_temp_to_celsius(
            cooling_setpoint,
            "F" if data.get("unit_system") != "metric" else "C"
        )
        heating_c = self._convert_temp_to_celsius(
            heating_setpoint,
            "F" if data.get("unit_system") != "metric" else "C"
        )
        
        return {
            "temperature": {
                "cooling_setpoint_c": cooling_c,
                "heating_setpoint_c": heating_c,
                "tolerance_k": 1.0,
            },
            "humidity": {
                "target_rh_percent": data.get("relative_humidity_target", 50),
            },
            "outdoor_design": {
                "summer_db_c": self._convert_temp_to_celsius(
                    data.get("outdoor_temp_summer", 95),
                    "F" if data.get("unit_system") != "metric" else "C"
                ) if data.get("outdoor_temp_summer") else None,
                "winter_db_c": self._convert_temp_to_celsius(
                    data.get("outdoor_temp_winter", 10),
                    "F" if data.get("unit_system") != "metric" else "C"
                ) if data.get("outdoor_temp_winter") else None,
            },
        }
    
    def _build_compliance_section(self, data: Dict) -> Dict:
        """Build compliance and standards requirements."""
        # Convert max velocity to m/s
        max_velocity = data.get("max_air_velocity", 50)  # Default 50 ft/min
        if data.get("unit_system") != "metric":
            max_velocity_ms = max_velocity * 0.00508  # ft/min to m/s
        else:
            max_velocity_ms = max_velocity
        
        return {
            "standard": data.get("ventilation_standard", "ashrae_62_1"),
            "space_type": data.get("space_type_ashrae", "office"),
            "targets": {
                "adpi_minimum": data.get("adpi_target", 80),
                "ppd_maximum": data.get("ppd_limit", 10),
                "max_velocity_ms": max_velocity_ms,
                "co2_limit_ppm": data.get("co2_limit", 1000),
            },
            "ventilation": {
                "per_person_lps": self._get_ventilation_rate(
                    data.get("space_type_ashrae", "office"), "person"
                ),
                "per_area_lps_m2": self._get_ventilation_rate(
                    data.get("space_type_ashrae", "office"), "area"
                ),
            },
        }
    
    # ========== Helper Methods ==========
    
    def _convert_length_to_meters(self, value: float, unit: str) -> float:
        """Convert length to meters."""
        conversions = {
            "ft": 0.3048,
            "in": 0.0254,
            "m": 1.0,
            "cm": 0.01,
            "mm": 0.001,
        }
        return value * conversions.get(unit, 0.3048)
    
    def _convert_temp_to_celsius(self, value: float, unit: str) -> float:
        """Convert temperature to Celsius."""
        if unit.upper() == "F":
            return (value - 32) * 5/9
        elif unit.upper() == "K":
            return value - 273.15
        return value  # Already Celsius
    
    def _convert_airflow_to_cms(self, cfm: float, unit_system: str) -> float:
        """Convert airflow to m³/s."""
        if unit_system == "metric":
            return cfm  # Assume input is already m³/s
        return cfm * 0.000471947  # CFM to m³/s
    
    def _get_metabolic_rate(self, activity: str) -> float:
        """Get metabolic rate in Watts per person."""
        rates = {
            "sleeping": 40,
            "seated_quiet": 60,
            "office": 75,
            "standing_light": 90,
            "walking": 110,
            "light_machine": 140,
            "heavy_work": 235,
        }
        return rates.get(activity, 75)
    
    def _get_ventilation_rate(self, space_type: str, rate_type: str) -> float:
        """Get ASHRAE 62.1 ventilation rate."""
        # Per person rates (L/s)
        person_rates = {
            "office": 2.5,
            "conference": 2.5,
            "classroom": 5.0,
            "retail": 3.8,
            "restaurant": 3.8,
            "hotel_lobby": 2.5,
            "hotel_room": 2.5,
            "hospital_patient": 2.5,
            "laboratory": 5.0,
            "gymnasium": 10.0,
            "warehouse": 2.5,
            "data_center": 2.5,
        }
        
        # Per area rates (L/s/m²)
        area_rates = {
            "office": 0.3,
            "conference": 0.3,
            "classroom": 0.3,
            "retail": 0.6,
            "restaurant": 0.9,
            "hotel_lobby": 0.3,
            "hotel_room": 0.3,
            "hospital_patient": 0.6,
            "laboratory": 0.9,
            "gymnasium": 0.3,
            "warehouse": 0.06,
            "data_center": 0.3,
        }
        
        if rate_type == "person":
            return person_rates.get(space_type, 2.5)
        return area_rates.get(space_type, 0.3)
    
    def validate_for_generation(self, data: Dict) -> tuple:
        """
        Validate form data before generation.
        Returns (is_valid, errors_dict).
        """
        errors = self.schema.validate_data(data)
        return len(errors) == 0, errors
    
    def generate_gui_format(self, form_data: Dict[str, Any], 
                            project_dir: Path,
                            blueprint_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate job_spec.json in HyperFOAM GUI's native format.
        
        Args:
            form_data: Dictionary of field values from intake form
            project_dir: Project directory path
            blueprint_path: Optional path to blueprint image
            
        Returns:
            Complete job_spec dictionary in GUI format
        """
        # Determine unit system
        is_metric = form_data.get("unit_system") == "metric"
        
        # Get dimensions in user's units
        length = float(form_data.get("room_length", 10))
        width = float(form_data.get("room_width", 10))
        height = float(form_data.get("room_height", 3))
        
        # Convert to meters for GUI (it uses meters internally)
        length_unit = form_data.get("length_unit", "ft")
        length_m = self._convert_length_to_meters(length, length_unit)
        width_m = self._convert_length_to_meters(width, length_unit)
        height_m = self._convert_length_to_meters(height, length_unit)
        
        # Convert temperatures to Celsius
        supply_temp = float(form_data.get("supply_temperature", 55))
        temp_unit = "C" if is_metric else "F"
        supply_temp_c = self._convert_temp_to_celsius(supply_temp, temp_unit)
        
        ambient_temp = float(form_data.get("indoor_setpoint_cooling", 22 if is_metric else 72))
        ambient_temp_c = self._convert_temp_to_celsius(ambient_temp, temp_unit)
        
        # Convert airflow to m³/s
        supply_cfm = float(form_data.get("supply_airflow", 100))
        supply_m3s = self._convert_airflow_to_cms(supply_cfm, form_data.get("unit_system", "imperial"))
        
        # Get counts
        vent_count = int(form_data.get("vent_count", 1))
        return_count = int(form_data.get("return_count", 1))
        occupant_count = int(form_data.get("occupancy", 1))
        
        # Build vents list
        vents = []
        per_vent_flow = supply_m3s / vent_count if vent_count > 0 else supply_m3s
        
        # Supply vents - distributed across ceiling
        for i in range(vent_count):
            x_pos = length_m * (i + 1) / (vent_count + 1)
            vents.append({
                "id": i + 1,
                "name": f"Supply {i + 1}",
                "type": 0,  # 0 = supply
                "position": [x_pos, height_m - 0.2, width_m / 2],
                "dimensions": [0.6, 0.15, 0.6],
                "direction": [0, -1, 0],  # Blowing downward
                "flowRate": per_vent_flow,
                "velocity": 2.5,
                "temperature": supply_temp_c,
                "diffuserPattern": "4-way",
            })
        
        # Return vents - near floor
        for i in range(return_count):
            x_pos = length_m * (i + 1) / (return_count + 1)
            vents.append({
                "id": vent_count + i + 1,
                "name": f"Return {i + 1}",
                "type": 1,  # 1 = return
                "position": [x_pos, 0.3, width_m / 2],
                "dimensions": [0.6, 0.15, 0.6],
                "direction": [0, 1, 0],
                "flowRate": per_vent_flow,
                "velocity": 2.5,
                "temperature": ambient_temp_c,
                "diffuserPattern": "4-way",
            })
        
        # Build occupants list
        occupants = []
        for i in range(occupant_count):
            x_pos = length_m * (i + 1) / (occupant_count + 1)
            occupants.append({
                "id": i + 1,
                "name": f"Person {i + 1}",
                "position": [x_pos, 0, width_m / 2],
                "heatOutput": self._get_metabolic_rate(form_data.get("occupant_activity", "office")),
                "co2Output": 0.005,  # L/s typical for office work
            })
        
        # Build room name
        room_name = form_data.get("room_name", "Main Room")
        project_name = form_data.get("project_name", "HyperFOAM Project")
        
        # Detected rooms from extraction
        detected_rooms = form_data.get("detected_rooms", [])
        
        # Build rooms list - one room per detected room, or single main room
        rooms = []
        if detected_rooms and len(detected_rooms) > 1:
            # Multiple rooms detected - create each one
            for i, room in enumerate(detected_rooms):
                rooms.append({
                    "id": i + 1,
                    "name": room if isinstance(room, str) else str(room),
                    "dimensions": [length_m, height_m, width_m],
                    "position": [0, 0, 0],
                    "wallThickness": 0.15,
                })
        else:
            # Single room
            rooms.append({
                "id": 1,
                "name": room_name,
                "dimensions": [length_m, height_m, width_m],
                "position": [0, 0, 0],
                "wallThickness": 0.15,
            })
        
        # Build complete job_spec in GUI format
        job_spec = {
            "schema_version": "1.0.0",
            "schema_min_reader_version": "1.0.0",
            
            "coordinate_system": {
                "origin": [0, 0, 0],
                "upAxis": "Y",
            },
            
            "deliverables": {
                "projectName": project_name.replace(" ", "_"),
                "author": form_data.get("client_name", ""),
                "templateName": "default",
                "includeFlow": True,
                "includeThermal": True,
                "includeComfort": True,
                "includeCO2": True,
                "includeOptimization": False,
            },
            
            "geometry": {
                "rooms": rooms,
                "obstacles": [],
                "openings": [],
                "assets": [],
                "blueprintPath": str(blueprint_path) if blueprint_path else "",
            },
            
            "hvac": {
                "vents": vents,
                "ambientTemperature": ambient_temp_c,
                "ambientPressure": 101325,
                "ambientCO2": 400,
            },
            
            "sources": {
                "occupants": occupants,
                "lightingLoad": float(form_data.get("lighting_load", 0)),
                "equipmentLoad": float(form_data.get("equipment_load", 0)),
            },
            
            "grid": {
                "nx": 64,
                "ny": 32,
                "nz": 48,
                "uniformGrid": True,
            },
            
            "solver": {
                "maxSteps": 1000,
                "maxSimTime": 10,
                "dt": 0.01,
                "convergenceTol": 1e-6,
                "enableTurbulence": form_data.get("turbulence_model", "none") != "none",
                "turbulenceModel": form_data.get("turbulence_model", "none"),
                "useMixedPrecision": True,
                "isValidated": False,
            },
            
            "targets": {
                "velocityMax": 0.25,
                "pmvMin": -0.5,
                "pmvMax": 0.5,
                "ppdMax": float(form_data.get("ppd_limit", 10)),
                "edtMax": 1.5,
                "adpiThreshold": float(form_data.get("adpi_target", 80)),
                "co2Max": float(form_data.get("co2_limit", 1000)),
            },
            
            "optimization": {
                "objective": "maximize_adpi",
                "maxIterations": 50,
                "populationSize": 20,
                "multiObjective": False,
                "decisionVars": [],
            },
            
            "units": {
                "length": 0 if is_metric else 0,  # 0 = meters (GUI always uses meters internally)
                "temperature": 0,  # 0 = Celsius
                "flow": 0,  # 0 = m³/s
            },
        }
        
        return job_spec
