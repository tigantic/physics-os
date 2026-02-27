"""
Intake Field Schema
===================

Defines all fields for the Universal Intake System, categorized as:
- MANDATORY: Required for simulation to run
- RECOMMENDED: Strongly suggested for accurate results
- OPTIONAL: Advanced/fine-tuning parameters
- COMPLIANCE: Regulatory and standard requirements
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
import re


class FieldCategory(Enum):
    """Field importance categories."""
    MANDATORY = "mandatory"      # Simulation cannot run without
    RECOMMENDED = "recommended"  # Significantly affects accuracy
    OPTIONAL = "optional"        # Fine-tuning / advanced
    COMPLIANCE = "compliance"    # Standards and regulations


class FieldType(Enum):
    """Data types for fields."""
    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    SELECT = "select"           # Dropdown selection
    MULTISELECT = "multiselect" # Multiple selection
    MEASUREMENT = "measurement" # Value with unit
    TEMPERATURE = "temperature"
    AIRFLOW = "airflow"
    FILE = "file"               # File upload
    JSON = "json"               # Structured data
    GEOMETRY = "geometry"       # 3D coordinates/mesh


@dataclass
class FieldValidation:
    """Validation rules for a field."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None  # Regex pattern for text
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    error_message: Optional[str] = None
    
    def validate(self, value: Any) -> tuple:
        """
        Validate a value against rules.
        Returns (is_valid, error_message).
        """
        if value is None:
            return True, None
        
        if self.min_value is not None and isinstance(value, (int, float)):
            if value < self.min_value:
                return False, self.error_message or f"Value must be >= {self.min_value}"
        
        if self.max_value is not None and isinstance(value, (int, float)):
            if value > self.max_value:
                return False, self.error_message or f"Value must be <= {self.max_value}"
        
        if self.pattern and isinstance(value, str):
            if not re.match(self.pattern, value):
                return False, self.error_message or f"Value must match pattern: {self.pattern}"
        
        if self.allowed_values and value not in self.allowed_values:
            return False, self.error_message or f"Value must be one of: {self.allowed_values}"
        
        if self.custom_validator:
            try:
                result = self.custom_validator(value)
                if not result:
                    return False, self.error_message or "Custom validation failed"
            except Exception as e:
                return False, str(e)
        
        return True, None


@dataclass
class IntakeField:
    """
    Definition of a single intake field.
    
    Attributes:
        name: Internal field name (snake_case)
        label: Display label for UI
        category: Importance category
        field_type: Data type
        description: Help text for user
        default: Default value
        validation: Validation rules
        unit_type: For measurements, the unit category
        depends_on: Fields this depends on (for conditional display)
        group: UI grouping
        order: Display order within group
    """
    name: str
    label: str
    category: FieldCategory
    field_type: FieldType
    description: str = ""
    default: Any = None
    validation: Optional[FieldValidation] = None
    unit_type: Optional[str] = None  # "length", "area", "temperature", etc.
    depends_on: Optional[Dict[str, Any]] = None
    group: str = "general"
    order: int = 0
    options: Optional[List[Dict[str, Any]]] = None  # For select fields
    
    def to_dict(self) -> Dict[str, Any]:
        """Export field definition as dictionary."""
        return {
            "name": self.name,
            "label": self.label,
            "category": self.category.value,
            "field_type": self.field_type.value,
            "description": self.description,
            "default": self.default,
            "unit_type": self.unit_type,
            "group": self.group,
            "order": self.order,
            "options": self.options,
        }


class IntakeSchema:
    """
    Complete schema for the Universal Intake System.
    
    Organizes all fields by category and group, provides
    validation, and generates form structure.
    """
    
    def __init__(self):
        self.fields: Dict[str, IntakeField] = {}
        self._build_schema()
    
    def _build_schema(self):
        """Define all intake fields."""
        
        # ============================================================
        # PROJECT INFORMATION
        # ============================================================
        self._add_field(IntakeField(
            name="project_name",
            label="Project Name",
            category=FieldCategory.MANDATORY,
            field_type=FieldType.TEXT,
            description="Unique identifier for this project",
            group="project",
            order=1,
            validation=FieldValidation(
                pattern=r"^[\w\s\-_]{1,100}$",
                error_message="Project name must be 1-100 characters (letters, numbers, spaces, hyphens)"
            ),
        ))
        
        self._add_field(IntakeField(
            name="project_description",
            label="Project Description",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.TEXT,
            description="Brief description of the project scope",
            group="project",
            order=2,
        ))
        
        self._add_field(IntakeField(
            name="client_name",
            label="Client Name",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.TEXT,
            group="project",
            order=3,
        ))
        
        self._add_field(IntakeField(
            name="project_number",
            label="Project Number",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.TEXT,
            description="Internal project tracking number",
            group="project",
            order=4,
        ))
        
        # ============================================================
        # UNIT SYSTEM
        # ============================================================
        self._add_field(IntakeField(
            name="unit_system",
            label="Unit System",
            category=FieldCategory.MANDATORY,
            field_type=FieldType.SELECT,
            description="Primary measurement system for this project",
            default="imperial",
            group="units",
            order=1,
            options=[
                {"value": "imperial", "label": "Imperial (ft, °F, CFM)"},
                {"value": "metric", "label": "Metric (m, °C, m³/s)"},
            ],
        ))
        
        self._add_field(IntakeField(
            name="length_unit",
            label="Length Unit",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.SELECT,
            description="Unit for length measurements",
            default="ft",
            group="units",
            order=2,
            options=[
                {"value": "ft", "label": "Feet (ft)"},
                {"value": "in", "label": "Inches (in)"},
                {"value": "m", "label": "Meters (m)"},
                {"value": "cm", "label": "Centimeters (cm)"},
                {"value": "mm", "label": "Millimeters (mm)"},
            ],
        ))
        
        self._add_field(IntakeField(
            name="blueprint_scale",
            label="Blueprint Scale",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.TEXT,
            description="Scale notation (e.g., 1/4\" = 1', 1:100)",
            group="units",
            order=3,
        ))
        
        # ============================================================
        # ROOM GEOMETRY
        # ============================================================
        self._add_field(IntakeField(
            name="room_name",
            label="Room/Zone Name",
            category=FieldCategory.MANDATORY,
            field_type=FieldType.TEXT,
            description="Name of the room or zone being analyzed",
            group="geometry",
            order=1,
        ))
        
        self._add_field(IntakeField(
            name="room_length",
            label="Room Length",
            category=FieldCategory.MANDATORY,
            field_type=FieldType.MEASUREMENT,
            description="Length of the room (longest dimension)",
            unit_type="length",
            group="geometry",
            order=2,
            default=30.0,
            validation=FieldValidation(min_value=1.0, max_value=1000),
        ))
        
        self._add_field(IntakeField(
            name="room_width",
            label="Room Width",
            category=FieldCategory.MANDATORY,
            field_type=FieldType.MEASUREMENT,
            description="Width of the room",
            unit_type="length",
            group="geometry",
            order=3,
            default=20.0,
            validation=FieldValidation(min_value=1.0, max_value=1000),
        ))
        
        self._add_field(IntakeField(
            name="room_height",
            label="Ceiling Height",
            category=FieldCategory.MANDATORY,
            field_type=FieldType.MEASUREMENT,
            description="Floor to ceiling height",
            unit_type="length",
            group="geometry",
            order=4,
            default=10.0,
            validation=FieldValidation(min_value=1.0, max_value=100),
        ))
        
        self._add_field(IntakeField(
            name="floor_area",
            label="Floor Area",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.MEASUREMENT,
            description="Total floor area (auto-calculated if not provided)",
            unit_type="area",
            group="geometry",
            order=5,
        ))
        
        self._add_field(IntakeField(
            name="room_volume",
            label="Room Volume",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.MEASUREMENT,
            description="Total room volume (auto-calculated if not provided)",
            unit_type="volume",
            group="geometry",
            order=6,
        ))
        
        self._add_field(IntakeField(
            name="geometry_file",
            label="Geometry File",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.FILE,
            description="IFC, OBJ, or STL file for complex geometry",
            group="geometry",
            order=7,
        ))
        
        # ============================================================
        # HVAC SYSTEM
        # ============================================================
        self._add_field(IntakeField(
            name="hvac_system_type",
            label="HVAC System Type",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.SELECT,
            description="Type of HVAC system serving this space",
            default="vav",
            group="hvac",
            order=1,
            options=[
                {"value": "vav", "label": "Variable Air Volume (VAV)"},
                {"value": "cav", "label": "Constant Air Volume (CAV)"},
                {"value": "doas", "label": "Dedicated Outdoor Air (DOAS)"},
                {"value": "vrf", "label": "Variable Refrigerant Flow (VRF)"},
                {"value": "split", "label": "Split System"},
                {"value": "ptac", "label": "PTAC/PTHP"},
                {"value": "radiant", "label": "Radiant Floor/Ceiling"},
                {"value": "chilled_beam", "label": "Chilled Beam"},
                {"value": "other", "label": "Other"},
            ],
        ))
        
        self._add_field(IntakeField(
            name="vent_count",
            label="Number of Supply Diffusers",
            category=FieldCategory.MANDATORY,
            field_type=FieldType.INTEGER,
            description="Total number of supply air diffusers/registers",
            group="hvac",
            order=2,
            validation=FieldValidation(min_value=1, max_value=500),
        ))
        
        self._add_field(IntakeField(
            name="return_count",
            label="Number of Return Grilles",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.INTEGER,
            description="Total number of return air grilles",
            default=1,
            group="hvac",
            order=3,
            validation=FieldValidation(min_value=0, max_value=100),
        ))
        
        self._add_field(IntakeField(
            name="diffuser_type",
            label="Diffuser Type",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.SELECT,
            description="Type of supply air diffuser",
            default="ceiling_4way",
            group="hvac",
            order=4,
            options=[
                {"value": "ceiling_4way", "label": "Ceiling - 4-Way"},
                {"value": "ceiling_2way", "label": "Ceiling - 2-Way"},
                {"value": "ceiling_1way", "label": "Ceiling - 1-Way (Linear)"},
                {"value": "ceiling_round", "label": "Ceiling - Round"},
                {"value": "ceiling_perforated", "label": "Ceiling - Perforated"},
                {"value": "sidewall", "label": "Sidewall Register"},
                {"value": "floor", "label": "Floor Diffuser"},
                {"value": "displacement", "label": "Displacement Diffuser"},
                {"value": "slot", "label": "Slot Diffuser"},
                {"value": "jet", "label": "Jet Nozzle"},
            ],
        ))
        
        self._add_field(IntakeField(
            name="supply_airflow",
            label="Total Supply Airflow",
            category=FieldCategory.MANDATORY,
            field_type=FieldType.AIRFLOW,
            description="Total supply air volume flow rate",
            unit_type="airflow",
            group="hvac",
            order=5,
            validation=FieldValidation(min_value=1, max_value=1000000),
        ))
        
        self._add_field(IntakeField(
            name="supply_temperature",
            label="Supply Air Temperature",
            category=FieldCategory.MANDATORY,
            field_type=FieldType.TEMPERATURE,
            description="Temperature of air leaving supply diffusers",
            unit_type="temperature",
            group="hvac",
            order=6,
            default=55,  # °F typical cooling
        ))
        
        self._add_field(IntakeField(
            name="return_airflow",
            label="Return Airflow",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.AIRFLOW,
            description="Return air volume (defaults to supply if not specified)",
            unit_type="airflow",
            group="hvac",
            order=7,
        ))
        
        # Critical CFD parameters for accurate simulation
        self._add_field(IntakeField(
            name="diffuser_effective_area",
            label="Diffuser Effective Area",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.NUMBER,
            description="Open area where air actually exits (not face size). Critical for correct throw velocity. Typical: 40-60% of face area.",
            default=0.05,  # 0.05 m² ≈ 0.5 ft² per diffuser typical
            unit_type="area",
            group="hvac",
            order=8,
            validation=FieldValidation(min_value=0.001, max_value=10.0),
        ))
        
        self._add_field(IntakeField(
            name="diffuser_neck_size",
            label="Diffuser Neck Size",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.MEASUREMENT,
            description="Duct connection size (e.g., 8 inch, 10 inch diameter)",
            unit_type="length",
            group="hvac",
            order=9,
        ))
        
        # ============================================================
        # WALL THERMAL BOUNDARIES (Critical for accuracy)
        # ============================================================
        self._add_field(IntakeField(
            name="wall_boundary_type",
            label="Wall Thermal Boundary",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.SELECT,
            description="How walls interact thermally. Adiabatic=no heat transfer (airflow only). Fixed Temp=set surface temps. U-Value=realistic heat flux.",
            default="adiabatic",
            group="envelope",
            order=1,
            options=[
                {"value": "adiabatic", "label": "Adiabatic (No Heat Transfer - Airflow Only)"},
                {"value": "fixed_temp", "label": "Fixed Surface Temperature"},
                {"value": "u_value", "label": "U-Value / Heat Flux (Most Accurate)"},
            ],
        ))
        
        self._add_field(IntakeField(
            name="wall_temperature",
            label="Wall Surface Temperature",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.TEMPERATURE,
            description="Fixed wall surface temperature (only if Fixed Temp selected)",
            default=72,  # °F
            unit_type="temperature",
            group="envelope",
            order=2,
        ))
        
        self._add_field(IntakeField(
            name="wall_u_value",
            label="Wall U-Value",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.NUMBER,
            description="Wall thermal transmittance (BTU/hr·ft²·°F or W/m²·K). Typical: 0.05-0.1 BTU/hr·ft²·°F for insulated walls.",
            default=0.08,
            group="envelope",
            order=3,
            validation=FieldValidation(min_value=0.01, max_value=2.0),
        ))
        
        self._add_field(IntakeField(
            name="window_area",
            label="Window/Glazing Area",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.NUMBER,
            description="Total window area (sq ft or m²). Windows often hottest surface.",
            default=0,
            unit_type="area",
            group="envelope",
            order=4,
        ))
        
        self._add_field(IntakeField(
            name="window_u_value",
            label="Window U-Value",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.NUMBER,
            description="Window thermal transmittance. Typical: 0.25-0.5 for double-pane.",
            default=0.35,
            group="envelope",
            order=5,
            validation=FieldValidation(min_value=0.1, max_value=1.5),
        ))
        
        self._add_field(IntakeField(
            name="exterior_wall_area",
            label="Exterior Wall Area",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.NUMBER,
            description="Total exterior wall area (sq ft or m²). Zero for interior rooms.",
            default=0,
            unit_type="area",
            group="envelope",
            order=6,
        ))
        
        # ============================================================
        # THERMAL LOADS
        # ============================================================
        self._add_field(IntakeField(
            name="occupancy",
            label="Design Occupancy",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.INTEGER,
            description="Number of people in the space at design conditions",
            default=1,
            group="loads",
            order=1,
            validation=FieldValidation(min_value=0, max_value=10000),
        ))
        
        self._add_field(IntakeField(
            name="occupant_activity",
            label="Occupant Activity Level",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.SELECT,
            description="Metabolic activity level",
            default="office",
            group="loads",
            order=2,
            options=[
                {"value": "sleeping", "label": "Sleeping (40 W/person)"},
                {"value": "seated_quiet", "label": "Seated, Quiet (60 W/person)"},
                {"value": "office", "label": "Office Work (75 W/person)"},
                {"value": "standing_light", "label": "Standing, Light Work (90 W/person)"},
                {"value": "walking", "label": "Walking (110 W/person)"},
                {"value": "light_machine", "label": "Light Machine Work (140 W/person)"},
                {"value": "heavy_work", "label": "Heavy Work (235 W/person)"},
            ],
        ))
        
        self._add_field(IntakeField(
            name="lighting_load",
            label="Lighting Load",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.NUMBER,
            description="Total lighting power (Watts)",
            default=0,
            unit_type="power",
            group="loads",
            order=3,
            validation=FieldValidation(min_value=0, max_value=100000),
        ))
        
        self._add_field(IntakeField(
            name="equipment_load",
            label="Equipment Load",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.NUMBER,
            description="Total equipment/plug load power (Watts)",
            default=0,
            unit_type="power",
            group="loads",
            order=4,
            validation=FieldValidation(min_value=0, max_value=500000),
        ))
        
        self._add_field(IntakeField(
            name="solar_gain",
            label="Solar Heat Gain",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.NUMBER,
            description="Peak solar heat gain (BTU/hr or W)",
            unit_type="power",
            group="loads",
            order=5,
        ))
        
        # ============================================================
        # DESIGN CONDITIONS
        # ============================================================
        self._add_field(IntakeField(
            name="indoor_setpoint_cooling",
            label="Cooling Setpoint",
            category=FieldCategory.RECOMMENDED,
            field_type=FieldType.TEMPERATURE,
            description="Indoor temperature setpoint for cooling",
            default=75,  # °F
            unit_type="temperature",
            group="conditions",
            order=1,
        ))
        
        self._add_field(IntakeField(
            name="indoor_setpoint_heating",
            label="Heating Setpoint",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.TEMPERATURE,
            description="Indoor temperature setpoint for heating",
            default=70,  # °F
            unit_type="temperature",
            group="conditions",
            order=2,
        ))
        
        self._add_field(IntakeField(
            name="relative_humidity_target",
            label="Target Relative Humidity",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.NUMBER,
            description="Target relative humidity percentage",
            default=50,
            group="conditions",
            order=3,
            validation=FieldValidation(min_value=0, max_value=100),
        ))
        
        self._add_field(IntakeField(
            name="outdoor_temp_summer",
            label="Summer Outdoor Design Temp",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.TEMPERATURE,
            description="Outdoor design temperature for cooling (1% design day)",
            unit_type="temperature",
            group="conditions",
            order=4,
        ))
        
        self._add_field(IntakeField(
            name="outdoor_temp_winter",
            label="Winter Outdoor Design Temp",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.TEMPERATURE,
            description="Outdoor design temperature for heating (99% design day)",
            unit_type="temperature",
            group="conditions",
            order=5,
        ))
        
        # ============================================================
        # SOLVER SETTINGS
        # ============================================================
        self._add_field(IntakeField(
            name="turbulence_model",
            label="Turbulence Model",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.SELECT,
            description="CFD turbulence model selection",
            default="k_epsilon",
            group="solver",
            order=1,
            options=[
                {"value": "k_epsilon", "label": "k-ε (Standard)"},
                {"value": "k_epsilon_rng", "label": "k-ε RNG"},
                {"value": "k_omega", "label": "k-ω"},
                {"value": "k_omega_sst", "label": "k-ω SST"},
                {"value": "spalart_allmaras", "label": "Spalart-Allmaras"},
                {"value": "les", "label": "LES (Large Eddy Simulation)"},
                {"value": "laminar", "label": "Laminar"},
            ],
        ))
        
        self._add_field(IntakeField(
            name="grid_resolution",
            label="Grid Resolution",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.SELECT,
            description="Computational grid density",
            default="medium",
            group="solver",
            order=2,
            options=[
                {"value": "coarse", "label": "Coarse (Fast, ~100k cells)"},
                {"value": "medium", "label": "Medium (Balanced, ~500k cells)"},
                {"value": "fine", "label": "Fine (Accurate, ~2M cells)"},
                {"value": "very_fine", "label": "Very Fine (~10M cells)"},
                {"value": "custom", "label": "Custom"},
            ],
        ))
        
        self._add_field(IntakeField(
            name="simulation_time",
            label="Simulation Duration",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.NUMBER,
            description="Total simulation time in seconds (for transient)",
            default=300,
            group="solver",
            order=3,
            validation=FieldValidation(min_value=1, max_value=86400),
        ))
        
        self._add_field(IntakeField(
            name="steady_state",
            label="Steady State Simulation",
            category=FieldCategory.OPTIONAL,
            field_type=FieldType.BOOLEAN,
            description="Run steady-state instead of transient simulation",
            default=True,
            group="solver",
            order=4,
        ))
        
        # ============================================================
        # COMPLIANCE & STANDARDS
        # ============================================================
        self._add_field(IntakeField(
            name="adpi_target",
            label="ADPI Target",
            category=FieldCategory.COMPLIANCE,
            field_type=FieldType.NUMBER,
            description="Air Distribution Performance Index target (ASHRAE 55)",
            default=80,
            group="compliance",
            order=1,
            validation=FieldValidation(min_value=0, max_value=100),
        ))
        
        self._add_field(IntakeField(
            name="ppd_limit",
            label="PPD Limit",
            category=FieldCategory.COMPLIANCE,
            field_type=FieldType.NUMBER,
            description="Predicted Percentage Dissatisfied limit (ISO 7730)",
            default=10,
            group="compliance",
            order=2,
            validation=FieldValidation(min_value=0, max_value=100),
        ))
        
        self._add_field(IntakeField(
            name="max_air_velocity",
            label="Maximum Air Velocity",
            category=FieldCategory.COMPLIANCE,
            field_type=FieldType.NUMBER,
            description="Maximum allowable air velocity in occupied zone (ft/min or m/s)",
            default=50,  # ft/min
            unit_type="velocity",
            group="compliance",
            order=3,
        ))
        
        self._add_field(IntakeField(
            name="co2_limit",
            label="CO₂ Limit",
            category=FieldCategory.COMPLIANCE,
            field_type=FieldType.NUMBER,
            description="Maximum CO₂ concentration (ppm) per ASHRAE 62.1",
            default=1000,
            group="compliance",
            order=4,
            validation=FieldValidation(min_value=400, max_value=5000),
        ))
        
        self._add_field(IntakeField(
            name="ventilation_standard",
            label="Ventilation Standard",
            category=FieldCategory.COMPLIANCE,
            field_type=FieldType.SELECT,
            description="Applicable ventilation standard",
            default="ashrae_62_1",
            group="compliance",
            order=5,
            options=[
                {"value": "ashrae_62_1", "label": "ASHRAE 62.1"},
                {"value": "ashrae_62_2", "label": "ASHRAE 62.2 (Residential)"},
                {"value": "en_16798", "label": "EN 16798 (Europe)"},
                {"value": "gb_50736", "label": "GB 50736 (China)"},
                {"value": "custom", "label": "Custom Requirements"},
            ],
        ))
        
        self._add_field(IntakeField(
            name="space_type_ashrae",
            label="Space Type (ASHRAE)",
            category=FieldCategory.COMPLIANCE,
            field_type=FieldType.SELECT,
            description="Space category for ventilation rate calculation",
            default="office",
            group="compliance",
            order=6,
            options=[
                {"value": "office", "label": "Office Space"},
                {"value": "conference", "label": "Conference/Meeting Room"},
                {"value": "classroom", "label": "Classroom"},
                {"value": "retail", "label": "Retail Sales"},
                {"value": "restaurant", "label": "Restaurant Dining"},
                {"value": "hotel_lobby", "label": "Hotel Lobby"},
                {"value": "hotel_room", "label": "Hotel Guest Room"},
                {"value": "hospital_patient", "label": "Hospital Patient Room"},
                {"value": "laboratory", "label": "Laboratory"},
                {"value": "gymnasium", "label": "Gymnasium"},
                {"value": "warehouse", "label": "Warehouse"},
                {"value": "data_center", "label": "Data Center"},
            ],
        ))
    
    def _add_field(self, field: IntakeField):
        """Add a field to the schema."""
        self.fields[field.name] = field
    
    def get_field(self, name: str) -> Optional[IntakeField]:
        """Get a field by name."""
        return self.fields.get(name)
    
    def get_fields_by_category(self, category: FieldCategory) -> List[IntakeField]:
        """Get all fields in a category."""
        return [f for f in self.fields.values() if f.category == category]
    
    def get_fields_by_group(self, group: str) -> List[IntakeField]:
        """Get all fields in a group, sorted by order."""
        fields = [f for f in self.fields.values() if f.group == group]
        return sorted(fields, key=lambda f: f.order)
    
    def get_groups(self) -> List[str]:
        """Get all unique group names."""
        groups = list(set(f.group for f in self.fields.values()))
        # Define preferred order
        order = ["project", "units", "geometry", "hvac", "loads", 
                 "conditions", "solver", "compliance"]
        return sorted(groups, key=lambda g: order.index(g) if g in order else 999)
    
    def get_mandatory_fields(self) -> List[IntakeField]:
        """Get all mandatory fields."""
        return self.get_fields_by_category(FieldCategory.MANDATORY)
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate form data against schema.
        Returns dict of field_name -> list of error messages.
        """
        errors = {}
        
        # Check mandatory fields
        for field in self.get_mandatory_fields():
            if field.name not in data or data[field.name] is None:
                errors[field.name] = [f"{field.label} is required"]
            elif field.validation:
                is_valid, error = field.validation.validate(data[field.name])
                if not is_valid:
                    errors[field.name] = [error]
        
        # Validate all provided fields
        for name, value in data.items():
            if name in self.fields and value is not None:
                field = self.fields[name]
                if field.validation:
                    is_valid, error = field.validation.validate(value)
                    if not is_valid:
                        if name not in errors:
                            errors[name] = []
                        errors[name].append(error)
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Export schema as dictionary."""
        return {
            "fields": {name: f.to_dict() for name, f in self.fields.items()},
            "groups": self.get_groups(),
            "mandatory_count": len(self.get_mandatory_fields()),
        }


# Group labels and descriptions for UI
GROUP_METADATA = {
    "project": {
        "label": "📋 Project Information",
        "description": "Basic project identification and metadata",
        "icon": "folder",
    },
    "units": {
        "label": "📏 Measurement Units",
        "description": "Define the unit system for all measurements",
        "icon": "ruler",
    },
    "geometry": {
        "label": "🏗️ Room Geometry",
        "description": "Physical dimensions of the space",
        "icon": "cube",
    },
    "hvac": {
        "label": "🌀 HVAC System",
        "description": "Air distribution system configuration",
        "icon": "wind",
    },
    "loads": {
        "label": "⚡ Thermal Loads",
        "description": "Heat sources and occupancy",
        "icon": "flame",
    },
    "conditions": {
        "label": "🌡️ Design Conditions",
        "description": "Temperature and humidity targets",
        "icon": "thermometer",
    },
    "solver": {
        "label": "⚙️ Solver Settings",
        "description": "Advanced CFD simulation parameters",
        "icon": "settings",
    },
    "compliance": {
        "label": "✅ Compliance & Standards",
        "description": "Regulatory requirements and performance targets",
        "icon": "check-circle",
    },
}
