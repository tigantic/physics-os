"""
Unit Conversion System for HyperFOAM Universal Intake
=====================================================

Implements the "Sandwich Method" for physics engine unit handling:

    INPUT (Imperial) → SOLVER (SI/Meters/Kelvin) → OUTPUT (Imperial)

WHY SI FOR PHYSICS CALCULATIONS:
--------------------------------
1. **Solver Nativity**: Navier-Stokes equations assume SI units (kg, m, s).
   In Imperial, you need gc factors: F = ma/gc where gc ≈ 32.2.
   In SI: F = ma (clean, no conversion factors).

2. **Scientific Constants**: Boltzmann, Stefan-Boltzmann, Ideal Gas constant
   are standard in SI. Imperial versions are error-prone.

3. **Reynolds Number Trap**: If you input 10 ft but solver thinks 10 m,
   Re is off by ~3.28x, potentially changing flow regime (laminar↔turbulent).

4. **Ecosystem Compatibility**: PyTorch, NumPy, SciPy, OpenFOAM all use SI.

SUPPORTED UNITS:
----------------
- Length: meters, centimeters, millimeters, feet, inches, feet+inches
- Area: m², ft², in²
- Volume: m³, ft³, liters, gallons
- Temperature: Celsius, Fahrenheit, Kelvin
- Airflow: CFM, m³/s, m³/h, L/s
- Pressure: Pa, inWG, PSI
- Velocity: m/s, ft/min (FPM)

KEY CONVERSION CONSTANTS (NIST SP 811):
---------------------------------------
- 1 ft = 0.3048 m (exact, by definition)
- 1 CFM = 4.71947×10⁻⁴ m³/s
- K = °C + 273.15
- °F = °C × 9/5 + 32
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any
import re
from decimal import Decimal, ROUND_HALF_UP



class UnitSystem(Enum):
    """Primary unit system selection."""
    METRIC = "metric"
    IMPERIAL = "imperial"
    
    @classmethod
    def from_string(cls, value: str) -> "UnitSystem":
        """Parse unit system from string."""
        value = value.lower().strip()
        if value in ("metric", "si", "m", "meters", "metres"):
            return cls.METRIC
        elif value in ("imperial", "us", "ft", "feet", "inches"):
            return cls.IMPERIAL
        raise ValueError(f"Unknown unit system: {value}")


class LengthUnit(Enum):
    """Length measurement units."""
    METERS = ("m", "meters", "metres", "meter", "metre")
    CENTIMETERS = ("cm", "centimeters", "centimetres", "centimeter")
    MILLIMETERS = ("mm", "millimeters", "millimetres", "millimeter")
    FEET = ("ft", "feet", "'", "foot")
    INCHES = ("in", "inches", "\"", "inch")
    FEET_INCHES = ("ft-in", "feet-inches", "ft+in")
    
    @classmethod
    def from_string(cls, value: str) -> "LengthUnit":
        """Parse length unit from string."""
        value = value.lower().strip().replace(".", "")
        for unit in cls:
            if value in unit.value:
                return unit
        raise ValueError(f"Unknown length unit: {value}")
    
    @property
    def symbol(self) -> str:
        """Get standard symbol for unit."""
        return self.value[0]


class AreaUnit(Enum):
    """Area measurement units."""
    SQ_METERS = ("m²", "m2", "sq m", "sqm", "square meters")
    SQ_FEET = ("ft²", "ft2", "sq ft", "sqft", "square feet")
    SQ_INCHES = ("in²", "in2", "sq in", "sqin", "square inches")
    
    @classmethod
    def from_string(cls, value: str) -> "AreaUnit":
        value = value.lower().strip()
        for unit in cls:
            if value in unit.value:
                return unit
        raise ValueError(f"Unknown area unit: {value}")


class VolumeUnit(Enum):
    """Volume measurement units."""
    CUBIC_METERS = ("m³", "m3", "cu m", "cubic meters")
    CUBIC_FEET = ("ft³", "ft3", "cu ft", "cubic feet")
    LITERS = ("L", "l", "liters", "litres")
    GALLONS = ("gal", "gallons")
    
    @classmethod
    def from_string(cls, value: str) -> "VolumeUnit":
        value = value.lower().strip()
        for unit in cls:
            if value.lower() in [v.lower() for v in unit.value]:
                return unit
        raise ValueError(f"Unknown volume unit: {value}")


class TemperatureUnit(Enum):
    """Temperature units."""
    CELSIUS = ("°C", "C", "celsius", "deg C")
    FAHRENHEIT = ("°F", "F", "fahrenheit", "deg F")
    KELVIN = ("K", "kelvin")
    
    @classmethod
    def from_string(cls, value: str) -> "TemperatureUnit":
        value = value.strip()
        for unit in cls:
            if value in unit.value or value.upper() in [v.upper() for v in unit.value]:
                return unit
        raise ValueError(f"Unknown temperature unit: {value}")


class AirflowUnit(Enum):
    """Airflow rate units."""
    CFM = ("CFM", "cfm", "ft³/min", "cubic feet per minute")
    CMS = ("m³/s", "cms", "cubic meters per second")
    CMH = ("m³/h", "cmh", "cubic meters per hour")
    LPS = ("L/s", "lps", "liters per second")
    
    @classmethod
    def from_string(cls, value: str) -> "AirflowUnit":
        value = value.strip()
        for unit in cls:
            if value in unit.value or value.lower() in [v.lower() for v in unit.value]:
                return unit
        raise ValueError(f"Unknown airflow unit: {value}")


class VelocityUnit(Enum):
    """Air velocity units."""
    MPS = ("m/s", "mps", "meters per second")
    FPM = ("ft/min", "fpm", "feet per minute")
    
    @classmethod
    def from_string(cls, value: str) -> "VelocityUnit":
        value = value.lower().strip()
        for unit in cls:
            if value in unit.value:
                return unit
        raise ValueError(f"Unknown velocity unit: {value}")


class PressureUnit(Enum):
    """Pressure units."""
    PASCAL = ("Pa", "pascal", "pascals")
    INWG = ("inWG", "in WG", "inches water gauge", "in. w.g.")
    PSI = ("psi", "PSI", "pounds per square inch")
    
    @classmethod
    def from_string(cls, value: str) -> "PressureUnit":
        value = value.strip()
        for unit in cls:
            if value in unit.value or value.lower() in [v.lower() for v in unit.value]:
                return unit
        raise ValueError(f"Unknown pressure unit: {value}")


@dataclass
class Measurement:
    """
    A measurement with value and unit, supporting automatic conversion.
    
    Examples:
        >>> m = Measurement(10, LengthUnit.FEET)
        >>> m.to(LengthUnit.METERS)
        Measurement(value=3.048, unit=LengthUnit.METERS)
        
        >>> m = Measurement.parse("10'-6\"")  # Feet and inches
        >>> m.to_meters()
        3.2004
    """
    value: float
    unit: Union[LengthUnit, AreaUnit, VolumeUnit, TemperatureUnit, 
                AirflowUnit, VelocityUnit, PressureUnit]
    original_text: Optional[str] = None
    confidence: float = 1.0  # Extraction confidence (0-1)
    
    def __post_init__(self):
        """Validate measurement after initialization."""
        if self.value < 0 and not isinstance(self.unit, TemperatureUnit):
            raise ValueError(f"Negative value not allowed for {type(self.unit).__name__}")
    
    @classmethod
    def parse(cls, text: str, expected_type: str = "length") -> "Measurement":
        """
        Parse a measurement from free-form text.
        
        Supports formats:
        - "10 ft"
        - "10'-6\""
        - "10 feet 6 inches"
        - "3.5m"
        - "120 x 80 cm" (returns first dimension)
        """
        text = text.strip()
        original = text
        confidence = 1.0
        
        # Handle feet-inches notation: 10'-6", 10' 6", 10ft 6in
        ft_in_patterns = [
            r"(\d+(?:\.\d+)?)['\s]*[-]?\s*(\d+(?:\.\d+)?)[\"″]",  # 10'-6"
            r"(\d+(?:\.\d+)?)\s*(?:ft|feet|')\s*(\d+(?:\.\d+)?)\s*(?:in|inches|\")",  # 10 ft 6 in
        ]
        
        for pattern in ft_in_patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                feet = float(match.group(1))
                inches = float(match.group(2))
                total_feet = feet + inches / 12.0
                return cls(
                    value=total_feet,
                    unit=LengthUnit.FEET,
                    original_text=original,
                    confidence=confidence
                )
        
        # Handle simple unit patterns
        patterns = {
            "length": [
                (r"(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b", LengthUnit.METERS),
                (r"(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)\b", LengthUnit.CENTIMETERS),
                (r"(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)\b", LengthUnit.MILLIMETERS),
                (r"(\d+(?:\.\d+)?)\s*(?:ft|feet|')\b", LengthUnit.FEET),
                (r"(\d+(?:\.\d+)?)\s*(?:in|inches?|\")\b", LengthUnit.INCHES),
            ],
            "area": [
                (r"(\d+(?:\.\d+)?)\s*(?:m²|m2|sq\s*m)\b", AreaUnit.SQ_METERS),
                (r"(\d+(?:\.\d+)?)\s*(?:ft²|ft2|sq\s*ft)\b", AreaUnit.SQ_FEET),
            ],
            "volume": [
                (r"(\d+(?:\.\d+)?)\s*(?:m³|m3|cu\s*m)\b", VolumeUnit.CUBIC_METERS),
                (r"(\d+(?:\.\d+)?)\s*(?:ft³|ft3|cu\s*ft)\b", VolumeUnit.CUBIC_FEET),
                (r"(\d+(?:\.\d+)?)\s*(?:L|liters?|litres?)\b", VolumeUnit.LITERS),
            ],
            "temperature": [
                (r"(\d+(?:\.\d+)?)\s*(?:°?[Cc]|celsius)\b", TemperatureUnit.CELSIUS),
                (r"(\d+(?:\.\d+)?)\s*(?:°?[Ff]|fahrenheit)\b", TemperatureUnit.FAHRENHEIT),
                (r"(\d+(?:\.\d+)?)\s*(?:K|kelvin)\b", TemperatureUnit.KELVIN),
            ],
            "airflow": [
                (r"(\d+(?:\.\d+)?)\s*(?:CFM|cfm)\b", AirflowUnit.CFM),
                (r"(\d+(?:\.\d+)?)\s*(?:m³/s|cms)\b", AirflowUnit.CMS),
                (r"(\d+(?:\.\d+)?)\s*(?:m³/h|cmh)\b", AirflowUnit.CMH),
                (r"(\d+(?:\.\d+)?)\s*(?:L/s|lps)\b", AirflowUnit.LPS),
            ],
        }
        
        if expected_type in patterns:
            for pattern, unit in patterns[expected_type]:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return cls(
                        value=float(match.group(1)),
                        unit=unit,
                        original_text=original,
                        confidence=confidence
                    )
        
        # Try to extract just a number (lower confidence)
        number_match = re.search(r"(\d+(?:\.\d+)?)", text)
        if number_match:
            # Guess unit based on value magnitude for length
            value = float(number_match.group(1))
            if expected_type == "length":
                # Heuristic: values > 100 are likely mm/cm, < 100 likely ft/m
                unit = LengthUnit.FEET if value < 100 else LengthUnit.CENTIMETERS
                confidence = 0.5
            else:
                raise ValueError(f"Cannot determine unit from: {text}")
            
            return cls(
                value=value,
                unit=unit,
                original_text=original,
                confidence=confidence
            )
        
        raise ValueError(f"Cannot parse measurement from: {text}")
    
    def to_si(self) -> float:
        """Convert to SI base unit (meters, m², m³, °C, m³/s, m/s, Pa)."""
        return UnitConverter.to_si(self.value, self.unit)
    
    def to(self, target_unit) -> "Measurement":
        """Convert to another unit of the same type."""
        si_value = self.to_si()
        new_value = UnitConverter.from_si(si_value, target_unit)
        return Measurement(
            value=new_value,
            unit=target_unit,
            original_text=self.original_text,
            confidence=self.confidence
        )
    
    def __str__(self) -> str:
        """Format measurement for display."""
        if isinstance(self.unit, LengthUnit) and self.unit == LengthUnit.FEET:
            # Display as feet and inches if fractional
            feet = int(self.value)
            inches = (self.value - feet) * 12
            if inches > 0.01:
                return f"{feet}'-{inches:.1f}\""
            return f"{feet}'"
        
        # Standard format
        symbol = self.unit.value[0] if hasattr(self.unit, 'value') else str(self.unit)
        return f"{self.value:.3f} {symbol}"
    
    def __repr__(self) -> str:
        return f"Measurement({self.value}, {self.unit.name})"


class UnitConverter:
    """
    Static unit conversion methods.
    
    All conversions go through SI units as intermediate:
    - Length: meters
    - Area: square meters
    - Volume: cubic meters
    - Temperature: Celsius (with offset handling)
    - Airflow: m³/s
    - Velocity: m/s
    - Pressure: Pa
    """
    
    # Length conversions to meters
    LENGTH_TO_METERS = {
        LengthUnit.METERS: 1.0,
        LengthUnit.CENTIMETERS: 0.01,
        LengthUnit.MILLIMETERS: 0.001,
        LengthUnit.FEET: 0.3048,
        LengthUnit.INCHES: 0.0254,
        LengthUnit.FEET_INCHES: 0.3048,  # Base is feet
    }
    
    # Area conversions to m²
    AREA_TO_SQ_METERS = {
        AreaUnit.SQ_METERS: 1.0,
        AreaUnit.SQ_FEET: 0.092903,
        AreaUnit.SQ_INCHES: 0.00064516,
    }
    
    # Volume conversions to m³
    VOLUME_TO_CUBIC_METERS = {
        VolumeUnit.CUBIC_METERS: 1.0,
        VolumeUnit.CUBIC_FEET: 0.0283168,
        VolumeUnit.LITERS: 0.001,
        VolumeUnit.GALLONS: 0.00378541,
    }
    
    # Airflow conversions to m³/s
    AIRFLOW_TO_CMS = {
        AirflowUnit.CMS: 1.0,
        AirflowUnit.CFM: 0.000471947,
        AirflowUnit.CMH: 1/3600,
        AirflowUnit.LPS: 0.001,
    }
    
    # Velocity conversions to m/s
    VELOCITY_TO_MPS = {
        VelocityUnit.MPS: 1.0,
        VelocityUnit.FPM: 0.00508,
    }
    
    # Pressure conversions to Pa
    PRESSURE_TO_PA = {
        PressureUnit.PASCAL: 1.0,
        PressureUnit.INWG: 249.089,
        PressureUnit.PSI: 6894.76,
    }
    
    @classmethod
    def to_si(cls, value: float, unit) -> float:
        """Convert any supported unit to its SI base."""
        if isinstance(unit, LengthUnit):
            return value * cls.LENGTH_TO_METERS[unit]
        elif isinstance(unit, AreaUnit):
            return value * cls.AREA_TO_SQ_METERS[unit]
        elif isinstance(unit, VolumeUnit):
            return value * cls.VOLUME_TO_CUBIC_METERS[unit]
        elif isinstance(unit, TemperatureUnit):
            if unit == TemperatureUnit.CELSIUS:
                return value
            elif unit == TemperatureUnit.FAHRENHEIT:
                return (value - 32) * 5/9
            elif unit == TemperatureUnit.KELVIN:
                return value - 273.15
        elif isinstance(unit, AirflowUnit):
            return value * cls.AIRFLOW_TO_CMS[unit]
        elif isinstance(unit, VelocityUnit):
            return value * cls.VELOCITY_TO_MPS[unit]
        elif isinstance(unit, PressureUnit):
            return value * cls.PRESSURE_TO_PA[unit]
        
        raise ValueError(f"Unsupported unit type: {type(unit)}")
    
    @classmethod
    def from_si(cls, si_value: float, target_unit) -> float:
        """Convert from SI base unit to target unit."""
        if isinstance(target_unit, LengthUnit):
            return si_value / cls.LENGTH_TO_METERS[target_unit]
        elif isinstance(target_unit, AreaUnit):
            return si_value / cls.AREA_TO_SQ_METERS[target_unit]
        elif isinstance(target_unit, VolumeUnit):
            return si_value / cls.VOLUME_TO_CUBIC_METERS[target_unit]
        elif isinstance(target_unit, TemperatureUnit):
            if target_unit == TemperatureUnit.CELSIUS:
                return si_value
            elif target_unit == TemperatureUnit.FAHRENHEIT:
                return si_value * 9/5 + 32
            elif target_unit == TemperatureUnit.KELVIN:
                return si_value + 273.15
        elif isinstance(target_unit, AirflowUnit):
            return si_value / cls.AIRFLOW_TO_CMS[target_unit]
        elif isinstance(target_unit, VelocityUnit):
            return si_value / cls.VELOCITY_TO_MPS[target_unit]
        elif isinstance(target_unit, PressureUnit):
            return si_value / cls.PRESSURE_TO_PA[target_unit]
        
        raise ValueError(f"Unsupported unit type: {type(target_unit)}")
    
    @classmethod
    def convert(cls, value: float, from_unit, to_unit) -> float:
        """
        Convert between any two compatible units.
        
        Example:
            >>> UnitConverter.convert(10, LengthUnit.FEET, LengthUnit.METERS)
            3.048
        """
        if type(from_unit) != type(to_unit):
            raise ValueError(f"Cannot convert between {type(from_unit)} and {type(to_unit)}")
        
        si_value = cls.to_si(value, from_unit)
        return cls.from_si(si_value, to_unit)
    
    @classmethod
    def format_dual(cls, value: float, unit, precision: int = 2) -> str:
        """
        Format a value with both metric and imperial equivalents.
        
        Example:
            >>> UnitConverter.format_dual(10, LengthUnit.FEET)
            "10.00 ft (3.05 m)"
        """
        if isinstance(unit, LengthUnit):
            si_value = cls.to_si(value, unit)
            if unit in (LengthUnit.METERS, LengthUnit.CENTIMETERS, LengthUnit.MILLIMETERS):
                imperial = cls.from_si(si_value, LengthUnit.FEET)
                return f"{value:.{precision}f} {unit.symbol} ({imperial:.{precision}f} ft)"
            else:
                metric = si_value
                return f"{value:.{precision}f} {unit.symbol} ({metric:.{precision}f} m)"
        
        return f"{value:.{precision}f} {unit.value[0]}"


@dataclass 
class ProjectUnits:
    """
    Project-wide unit configuration.
    
    Stores the preferred unit system and specific overrides for the project.
    """
    system: UnitSystem = UnitSystem.IMPERIAL
    length: LengthUnit = LengthUnit.FEET
    area: AreaUnit = AreaUnit.SQ_FEET
    volume: VolumeUnit = VolumeUnit.CUBIC_FEET
    temperature: TemperatureUnit = TemperatureUnit.FAHRENHEIT
    airflow: AirflowUnit = AirflowUnit.CFM
    velocity: VelocityUnit = VelocityUnit.FPM
    pressure: PressureUnit = PressureUnit.INWG
    
    @classmethod
    def metric(cls) -> "ProjectUnits":
        """Create metric project units configuration."""
        return cls(
            system=UnitSystem.METRIC,
            length=LengthUnit.METERS,
            area=AreaUnit.SQ_METERS,
            volume=VolumeUnit.CUBIC_METERS,
            temperature=TemperatureUnit.CELSIUS,
            airflow=AirflowUnit.CMS,
            velocity=VelocityUnit.MPS,
            pressure=PressureUnit.PASCAL,
        )
    
    @classmethod
    def imperial(cls) -> "ProjectUnits":
        """Create imperial project units configuration."""
        return cls(
            system=UnitSystem.IMPERIAL,
            length=LengthUnit.FEET,
            area=AreaUnit.SQ_FEET,
            volume=VolumeUnit.CUBIC_FEET,
            temperature=TemperatureUnit.FAHRENHEIT,
            airflow=AirflowUnit.CFM,
            velocity=VelocityUnit.FPM,
            pressure=PressureUnit.INWG,
        )
    
    def to_dict(self) -> Dict[str, str]:
        """Export units configuration as dictionary."""
        return {
            "system": self.system.value,
            "length": self.length.symbol,
            "area": self.area.value[0],
            "volume": self.volume.value[0],
            "temperature": self.temperature.value[0],
            "airflow": self.airflow.value[0],
            "velocity": self.velocity.value[0],
            "pressure": self.pressure.value[0],
        }


def detect_unit_system(text: str) -> Tuple[UnitSystem, float]:
    """
    Detect the primary unit system used in a text document.
    
    Returns (system, confidence) tuple.
    """
    metric_patterns = [
        r"\b\d+\s*(?:m|meters?|metres?)\b",
        r"\b\d+\s*(?:cm|centimeters?)\b",
        r"\b\d+\s*(?:mm|millimeters?)\b",
        r"\b\d+\s*(?:m²|m2|sqm)\b",
        r"\b\d+\s*(?:°C|celsius)\b",
        r"\b\d+\s*(?:m³/s|L/s)\b",
    ]
    
    imperial_patterns = [
        r"\b\d+\s*(?:ft|feet|')\b",
        r"\b\d+\s*(?:in|inches?|\")\b",
        r"\b\d+['\s]*-?\s*\d+\"",  # 10'-6"
        r"\b\d+\s*(?:ft²|sqft)\b",
        r"\b\d+\s*(?:°F|fahrenheit)\b",
        r"\b\d+\s*(?:CFM|cfm)\b",
    ]
    
    metric_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in metric_patterns)
    imperial_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in imperial_patterns)
    
    total = metric_count + imperial_count
    if total == 0:
        return UnitSystem.IMPERIAL, 0.5  # Default with low confidence
    
    if metric_count > imperial_count:
        return UnitSystem.METRIC, metric_count / total
    else:
        return UnitSystem.IMPERIAL, imperial_count / total


# Convenience functions
def feet_inches(feet: float, inches: float = 0) -> Measurement:
    """Create a measurement from feet and inches."""
    total = feet + inches / 12.0
    return Measurement(total, LengthUnit.FEET)


def meters(value: float) -> Measurement:
    """Create a measurement in meters."""
    return Measurement(value, LengthUnit.METERS)


def centimeters(value: float) -> Measurement:
    """Create a measurement in centimeters."""
    return Measurement(value, LengthUnit.CENTIMETERS)


def cfm(value: float) -> Measurement:
    """Create an airflow measurement in CFM."""
    return Measurement(value, AirflowUnit.CFM)


def celsius(value: float) -> Measurement:
    """Create a temperature measurement in Celsius."""
    return Measurement(value, TemperatureUnit.CELSIUS)


def fahrenheit(value: float) -> Measurement:
    """Create a temperature measurement in Fahrenheit."""
    return Measurement(value, TemperatureUnit.FAHRENHEIT)
