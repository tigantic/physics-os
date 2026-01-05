"""
Simulation Module
=================

End-to-end simulation capabilities for hypersonic vehicle analysis,
including hardware-in-the-loop testing, flight data integration,
and mission simulation with Monte Carlo analysis.

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                    Simulation Module                              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
    │  │     HIL     │  │ Flight Data │  │ Real-Time   │  │ Mission │ │
    │  │  Interface  │  │  Interface  │  │    CFD      │  │   Sim   │ │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
    └──────────────────────────────────────────────────────────────────┘

Submodules:
    hil: Hardware-in-the-loop simulation interface
    flight_data: Telemetry parsing and trajectory reconstruction
    realtime_cfd: CFD-guidance coupling for real-time aero lookup
    mission: End-to-end mission simulation with uncertainty
"""

from tensornet.simulation.flight_data import (  # Data structures; Functions
    DataQuality,
    FlightRecord,
    TelemetryFormat,
    TelemetryFrame,
    TrajectoryReconstruction,
    compute_reconstruction_error,
    create_synthetic_flight_record,
    parse_telemetry,
    validate_against_flight,
)
from tensornet.simulation.hil import (  # Configuration; Sensor models; Actuator models; Interface; Factory functions
    ActuatorModel,
    ActuatorType,
    AirDataSensor,
    ControlSurface,
    GPSSensor,
    HILConfig,
    HILInterface,
    HILMode,
    IMUSensor,
    SensorModel,
    SensorNoise,
    SensorType,
    ThrustActuator,
    create_vehicle_actuators,
    create_vehicle_sensors,
)
from tensornet.simulation.mission import (  # Enums; Configuration; Results; Simulator; Analysis
    FailureMode,
    MissionConfig,
    MissionPhase,
    MissionResult,
    MissionSimulator,
    MonteCarloConfig,
    UncertaintyModel,
    analyze_dispersion,
    compute_sensitivity,
    run_monte_carlo,
)
from tensornet.simulation.realtime_cfd import (  # Configuration; Core classes; Functions
    AeroCoefficient,
    AeroPoint,
    AeroTable,
    AeroTableConfig,
    InterpolationMethod,
    RealTimeCFD,
    TableDimension,
    build_aero_table,
    create_hypersonic_waverider_model,
    interpolate_coefficients,
    validate_aero_table,
)

__all__ = [
    # HIL
    "HILConfig",
    "HILMode",
    "SensorType",
    "ActuatorType",
    "SensorModel",
    "SensorNoise",
    "IMUSensor",
    "GPSSensor",
    "AirDataSensor",
    "ActuatorModel",
    "ControlSurface",
    "ThrustActuator",
    "HILInterface",
    "create_vehicle_sensors",
    "create_vehicle_actuators",
    # Flight Data
    "TelemetryFormat",
    "DataQuality",
    "TelemetryFrame",
    "FlightRecord",
    "TrajectoryReconstruction",
    "parse_telemetry",
    "validate_against_flight",
    "compute_reconstruction_error",
    "create_synthetic_flight_record",
    # Real-Time CFD
    "InterpolationMethod",
    "TableDimension",
    "AeroTableConfig",
    "AeroPoint",
    "AeroCoefficient",
    "AeroTable",
    "RealTimeCFD",
    "build_aero_table",
    "interpolate_coefficients",
    "validate_aero_table",
    "create_hypersonic_waverider_model",
    # Mission
    "MissionPhase",
    "FailureMode",
    "MissionConfig",
    "UncertaintyModel",
    "MonteCarloConfig",
    "MissionResult",
    "MissionSimulator",
    "run_monte_carlo",
    "analyze_dispersion",
    "compute_sensitivity",
]
