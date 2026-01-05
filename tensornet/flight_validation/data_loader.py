"""
Flight data loading and parsing utilities.

This module provides tools for loading flight data from various
sources and formats for validation against CFD simulations.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np


class FlightDataSource(Enum):
    """Sources of flight data."""

    WIND_TUNNEL = auto()
    FLIGHT_TEST = auto()
    TRAJECTORY_RECONSTRUCTION = auto()
    TELEMETRY = auto()
    PUBLISHED_DATA = auto()
    SIMULATION = auto()


class FlightDataFormat(Enum):
    """Flight data file formats."""

    CSV = "csv"
    JSON = "json"
    HDF5 = "hdf5"
    MATLAB = "mat"
    TECPLOT = "dat"
    BINARY = "bin"


@dataclass
class SensorReading:
    """Single sensor reading."""

    timestamp: float
    value: float
    sensor_id: str
    unit: str = ""
    uncertainty: float = 0.0
    quality_flag: int = 0  # 0 = good, 1 = suspect, 2 = bad


@dataclass
class FlightCondition:
    """Flight condition at a specific time."""

    timestamp: float

    # Atmospheric conditions
    altitude_m: float = 0.0
    mach_number: float = 0.0
    velocity_m_s: float = 0.0

    # Ambient conditions
    pressure_pa: float = 101325.0
    temperature_k: float = 288.15
    density_kg_m3: float = 1.225

    # Vehicle state
    angle_of_attack_deg: float = 0.0
    sideslip_angle_deg: float = 0.0
    roll_angle_deg: float = 0.0

    # Position (geodetic)
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "altitude_m": self.altitude_m,
            "mach_number": self.mach_number,
            "velocity_m_s": self.velocity_m_s,
            "pressure_pa": self.pressure_pa,
            "temperature_k": self.temperature_k,
            "density_kg_m3": self.density_kg_m3,
            "angle_of_attack_deg": self.angle_of_attack_deg,
            "sideslip_angle_deg": self.sideslip_angle_deg,
        }

    def dynamic_pressure_pa(self) -> float:
        """Calculate dynamic pressure."""
        return 0.5 * self.density_kg_m3 * self.velocity_m_s**2


@dataclass
class AerodynamicData:
    """Aerodynamic forces and moments."""

    timestamp: float

    # Force coefficients
    cl: float = 0.0  # Lift coefficient
    cd: float = 0.0  # Drag coefficient
    cy: float = 0.0  # Side force coefficient

    # Moment coefficients
    cm: float = 0.0  # Pitching moment coefficient
    cn: float = 0.0  # Yawing moment coefficient
    croll: float = 0.0  # Rolling moment coefficient

    # Surface data
    cp_distribution: np.ndarray | None = None
    heat_flux_distribution: np.ndarray | None = None

    # Uncertainties
    cl_uncertainty: float = 0.0
    cd_uncertainty: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cl": self.cl,
            "cd": self.cd,
            "cy": self.cy,
            "cm": self.cm,
            "cn": self.cn,
            "croll": self.croll,
        }


@dataclass
class FlightRecord:
    """Complete flight data record."""

    record_id: str
    source: FlightDataSource

    # Time range
    start_time: float = 0.0
    end_time: float = 0.0

    # Data arrays
    conditions: list[FlightCondition] = field(default_factory=list)
    aero_data: list[AerodynamicData] = field(default_factory=list)
    sensor_readings: dict[str, list[SensorReading]] = field(default_factory=dict)

    # Metadata
    vehicle_name: str = ""
    test_name: str = ""
    date: str = ""
    notes: str = ""

    # Data quality
    quality_score: float = 1.0
    calibration_status: str = "unknown"

    def __post_init__(self):
        """Initialize time range from data."""
        if self.conditions and not self.start_time:
            timestamps = [c.timestamp for c in self.conditions]
            self.start_time = min(timestamps)
            self.end_time = max(timestamps)

    def get_condition_at_time(self, t: float) -> FlightCondition | None:
        """Get flight condition at specific time (interpolated)."""
        if not self.conditions:
            return None

        # Find bracketing conditions
        for i in range(len(self.conditions) - 1):
            c1 = self.conditions[i]
            c2 = self.conditions[i + 1]

            if c1.timestamp <= t <= c2.timestamp:
                # Linear interpolation
                alpha = (t - c1.timestamp) / (c2.timestamp - c1.timestamp)

                return FlightCondition(
                    timestamp=t,
                    altitude_m=c1.altitude_m + alpha * (c2.altitude_m - c1.altitude_m),
                    mach_number=c1.mach_number
                    + alpha * (c2.mach_number - c1.mach_number),
                    velocity_m_s=c1.velocity_m_s
                    + alpha * (c2.velocity_m_s - c1.velocity_m_s),
                    pressure_pa=c1.pressure_pa
                    + alpha * (c2.pressure_pa - c1.pressure_pa),
                    temperature_k=c1.temperature_k
                    + alpha * (c2.temperature_k - c1.temperature_k),
                    density_kg_m3=c1.density_kg_m3
                    + alpha * (c2.density_kg_m3 - c1.density_kg_m3),
                    angle_of_attack_deg=c1.angle_of_attack_deg
                    + alpha * (c2.angle_of_attack_deg - c1.angle_of_attack_deg),
                    sideslip_angle_deg=c1.sideslip_angle_deg
                    + alpha * (c2.sideslip_angle_deg - c1.sideslip_angle_deg),
                )

        return None

    def get_aero_at_time(self, t: float) -> AerodynamicData | None:
        """Get aerodynamic data at specific time."""
        if not self.aero_data:
            return None

        # Find closest aero data point
        closest = min(self.aero_data, key=lambda a: abs(a.timestamp - t))
        return closest

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "source": self.source.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_conditions": len(self.conditions),
            "num_aero_points": len(self.aero_data),
            "vehicle_name": self.vehicle_name,
            "test_name": self.test_name,
            "quality_score": self.quality_score,
        }


class FlightDataLoader:
    """
    Loader for flight data from various sources.
    """

    def __init__(self):
        """Initialize loader."""
        self._parsers = {
            FlightDataFormat.CSV: self._parse_csv,
            FlightDataFormat.JSON: self._parse_json,
        }

    def load(
        self,
        path: str | Path,
        format: FlightDataFormat | None = None,
        source: FlightDataSource = FlightDataSource.FLIGHT_TEST,
    ) -> FlightRecord:
        """
        Load flight data from file.

        Args:
            path: Path to data file
            format: Data format (auto-detected if not specified)
            source: Data source type

        Returns:
            FlightRecord with loaded data
        """
        path = Path(path)

        if format is None:
            format = self._detect_format(path)

        parser = self._parsers.get(format)
        if parser is None:
            raise ValueError(f"Unsupported format: {format}")

        record = parser(path)
        record.source = source
        record.record_id = path.stem

        return record

    def _detect_format(self, path: Path) -> FlightDataFormat:
        """Detect file format from extension."""
        suffix = path.suffix.lower()

        format_map = {
            ".csv": FlightDataFormat.CSV,
            ".json": FlightDataFormat.JSON,
            ".h5": FlightDataFormat.HDF5,
            ".hdf5": FlightDataFormat.HDF5,
            ".mat": FlightDataFormat.MATLAB,
            ".dat": FlightDataFormat.TECPLOT,
        }

        return format_map.get(suffix, FlightDataFormat.CSV)

    def _parse_csv(self, path: Path) -> FlightRecord:
        """Parse CSV flight data."""
        record = FlightRecord(
            record_id=path.stem,
            source=FlightDataSource.FLIGHT_TEST,
        )

        # Read CSV file
        with open(path) as f:
            lines = f.readlines()

        if not lines:
            return record

        # Parse header
        headers = [h.strip().lower() for h in lines[0].split(",")]

        # Column mapping
        col_map = {
            "time": "timestamp",
            "t": "timestamp",
            "alt": "altitude_m",
            "altitude": "altitude_m",
            "mach": "mach_number",
            "velocity": "velocity_m_s",
            "v": "velocity_m_s",
            "pressure": "pressure_pa",
            "p": "pressure_pa",
            "temperature": "temperature_k",
            "temp": "temperature_k",
            "density": "density_kg_m3",
            "rho": "density_kg_m3",
            "aoa": "angle_of_attack_deg",
            "alpha": "angle_of_attack_deg",
            "cl": "cl",
            "cd": "cd",
        }

        # Map headers
        header_indices = {}
        for i, h in enumerate(headers):
            if h in col_map:
                header_indices[col_map[h]] = i
            elif h in ["timestamp", "altitude_m", "mach_number", "cl", "cd"]:
                header_indices[h] = i

        # Parse data rows
        for line in lines[1:]:
            values = line.strip().split(",")
            if len(values) < len(headers):
                continue

            try:
                # Extract values
                data = {}
                for key, idx in header_indices.items():
                    data[key] = float(values[idx])

                # Create flight condition
                if "timestamp" in data:
                    condition = FlightCondition(
                        timestamp=data.get("timestamp", 0),
                        altitude_m=data.get("altitude_m", 0),
                        mach_number=data.get("mach_number", 0),
                        velocity_m_s=data.get("velocity_m_s", 0),
                        pressure_pa=data.get("pressure_pa", 101325),
                        temperature_k=data.get("temperature_k", 288.15),
                        density_kg_m3=data.get("density_kg_m3", 1.225),
                        angle_of_attack_deg=data.get("angle_of_attack_deg", 0),
                    )
                    record.conditions.append(condition)

                # Create aero data if available
                if "cl" in data or "cd" in data:
                    aero = AerodynamicData(
                        timestamp=data.get("timestamp", 0),
                        cl=data.get("cl", 0),
                        cd=data.get("cd", 0),
                    )
                    record.aero_data.append(aero)

            except (ValueError, IndexError):
                continue

        return record

    def _parse_json(self, path: Path) -> FlightRecord:
        """Parse JSON flight data."""
        with open(path) as f:
            data = json.load(f)

        record = FlightRecord(
            record_id=data.get("record_id", path.stem),
            source=FlightDataSource.FLIGHT_TEST,
            vehicle_name=data.get("vehicle_name", ""),
            test_name=data.get("test_name", ""),
            date=data.get("date", ""),
            notes=data.get("notes", ""),
        )

        # Parse conditions
        for cond_data in data.get("conditions", []):
            condition = FlightCondition(
                timestamp=cond_data.get("timestamp", 0),
                altitude_m=cond_data.get("altitude_m", 0),
                mach_number=cond_data.get("mach_number", 0),
                velocity_m_s=cond_data.get("velocity_m_s", 0),
                pressure_pa=cond_data.get("pressure_pa", 101325),
                temperature_k=cond_data.get("temperature_k", 288.15),
                density_kg_m3=cond_data.get("density_kg_m3", 1.225),
                angle_of_attack_deg=cond_data.get("angle_of_attack_deg", 0),
                sideslip_angle_deg=cond_data.get("sideslip_angle_deg", 0),
            )
            record.conditions.append(condition)

        # Parse aerodynamic data
        for aero_data in data.get("aero_data", []):
            aero = AerodynamicData(
                timestamp=aero_data.get("timestamp", 0),
                cl=aero_data.get("cl", 0),
                cd=aero_data.get("cd", 0),
                cy=aero_data.get("cy", 0),
                cm=aero_data.get("cm", 0),
                cl_uncertainty=aero_data.get("cl_uncertainty", 0),
                cd_uncertainty=aero_data.get("cd_uncertainty", 0),
            )
            record.aero_data.append(aero)

        return record


def load_flight_data(
    path: str | Path,
    source: FlightDataSource = FlightDataSource.FLIGHT_TEST,
) -> FlightRecord:
    """
    Load flight data from file.

    Args:
        path: Path to data file
        source: Data source type

    Returns:
        FlightRecord with loaded data
    """
    loader = FlightDataLoader()
    return loader.load(path, source=source)


def parse_telemetry(
    data: str | bytes | dict,
    format: str = "json",
) -> FlightRecord:
    """
    Parse telemetry data from string/bytes.

    Args:
        data: Telemetry data
        format: Data format ("json", "binary", etc.)

    Returns:
        FlightRecord with parsed data
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")

    if format == "json" and isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, dict):
        raise ValueError("Data must be dictionary or JSON string")

    record = FlightRecord(
        record_id=data.get("id", f"telemetry_{int(time.time())}"),
        source=FlightDataSource.TELEMETRY,
    )

    # Parse based on structure
    if "frames" in data:
        # Multi-frame telemetry
        for frame in data["frames"]:
            t = frame.get("time", 0)

            condition = FlightCondition(
                timestamp=t,
                altitude_m=frame.get("altitude", 0),
                mach_number=frame.get("mach", 0),
                velocity_m_s=frame.get("velocity", 0),
                angle_of_attack_deg=frame.get("aoa", 0),
            )
            record.conditions.append(condition)

    elif "state" in data:
        # Single state telemetry
        state = data["state"]
        condition = FlightCondition(
            timestamp=data.get("time", time.time()),
            altitude_m=state.get("altitude", 0),
            mach_number=state.get("mach", 0),
            velocity_m_s=state.get("velocity", 0),
        )
        record.conditions.append(condition)

    return record
