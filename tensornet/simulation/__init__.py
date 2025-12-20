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

from tensornet.simulation.hil import (
    # Configuration
    HILConfig,
    HILMode,
    SensorType,
    ActuatorType,
    # Sensor models
    SensorModel,
    SensorNoise,
    IMUSensor,
    GPSSensor,
    AirDataSensor,
    # Actuator models
    ActuatorModel,
    ControlSurface,
    ThrustActuator,
    # Interface
    HILInterface,
    # Factory functions
    create_vehicle_sensors,
    create_vehicle_actuators,
)

from tensornet.simulation.flight_data import (
    # Data structures
    TelemetryFormat,
    DataQuality,
    TelemetryFrame,
    FlightRecord,
    TrajectoryReconstruction,
    # Functions
    parse_telemetry,
    validate_against_flight,
    compute_reconstruction_error,
    create_synthetic_flight_record,
)

from tensornet.simulation.realtime_cfd import (
    # Configuration
    InterpolationMethod,
    TableDimension,
    AeroTableConfig,
    AeroPoint,
    AeroCoefficient,
    # Core classes
    AeroTable,
    RealTimeCFD,
    # Functions
    build_aero_table,
    interpolate_coefficients,
    validate_aero_table,
    create_hypersonic_waverider_model,
)

from tensornet.simulation.mission import (
    # Enums
    MissionPhase,
    FailureMode,
    # Configuration
    MissionConfig,
    UncertaintyModel,
    MonteCarloConfig,
    # Results
    MissionResult,
    # Simulator
    MissionSimulator,
    # Analysis
    run_monte_carlo,
    analyze_dispersion,
    compute_sensitivity,
)

__all__ = [
    # HIL
    'HILConfig', 'HILMode', 'SensorType', 'ActuatorType',
    'SensorModel', 'SensorNoise', 'IMUSensor', 'GPSSensor', 'AirDataSensor',
    'ActuatorModel', 'ControlSurface', 'ThrustActuator',
    'HILInterface', 'create_vehicle_sensors', 'create_vehicle_actuators',
    # Flight Data
    'TelemetryFormat', 'DataQuality', 'TelemetryFrame', 'FlightRecord',
    'TrajectoryReconstruction', 'parse_telemetry', 'validate_against_flight',
    'compute_reconstruction_error', 'create_synthetic_flight_record',
    # Real-Time CFD
    'InterpolationMethod', 'TableDimension', 'AeroTableConfig', 'AeroPoint',
    'AeroCoefficient', 'AeroTable', 'RealTimeCFD', 'build_aero_table',
    'interpolate_coefficients', 'validate_aero_table', 'create_hypersonic_waverider_model',
    # Mission
    'MissionPhase', 'FailureMode', 'MissionConfig', 'UncertaintyModel',
    'MonteCarloConfig', 'MissionResult', 'MissionSimulator',
    'run_monte_carlo', 'analyze_dispersion', 'compute_sensitivity',
]
