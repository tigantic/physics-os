"""
Digital Twin Framework for HyperTensor.

This module provides real-time digital twin capabilities for hypersonic
vehicles, enabling live state synchronization, predictive analytics,
and operational decision support.

Components:
    - StateSync: Real-time state synchronization between physical and digital systems
    - ReducedOrderModel: Fast physics-based surrogates for real-time prediction
    - HealthMonitor: Structural and thermal health monitoring with anomaly detection
    - PredictiveMaintenance: Remaining useful life estimation and maintenance scheduling
    - DigitalTwin: Complete digital twin orchestrator

Example:
    >>> from tensornet.digital_twin import DigitalTwin, DigitalTwinConfig
    >>> config = DigitalTwinConfig(
    ...     vehicle_id="HGV-001",
    ...     sync_rate=100.0,  # Hz
    ...     enable_prediction=True,
    ...     health_monitoring=True
    ... )
    >>> twin = DigitalTwin(config)
    >>> twin.connect_to_vehicle()
    >>> twin.start_sync()
"""

from .health_monitor import (
                             AnomalyDetector,
                             AnomalyType,
                             HealthConfig,
                             HealthMetric,
                             HealthMonitor,
                             HealthStatus,
                             StructuralHealth,
                             ThermalHealth,
                             compute_damage_index,
                             estimate_thermal_margin,
)
from .predictive import (
                             FailureMode,
                             MaintenanceConfig,
                             MaintenanceScheduler,
                             PredictiveMaintenance,
                             RULEstimator,
                             compute_reliability,
                             estimate_remaining_life,
                             optimize_maintenance_schedule,
)
from .reduced_order import (
                             AutoencoderROM,
                             DMDModel,
                             PODModel,
                             ReducedOrderModel,
                             ROMConfig,
                             ROMType,
                             compute_projection_error,
                             create_rom_from_snapshots,
                             validate_rom_accuracy,
)
from .state_sync import (
                             StateSync,
                             StateSynchronizer,
                             StateVector,
                             SyncConfig,
                             SyncMode,
                             SyncStatus,
                             compute_state_divergence,
                             extrapolate_state,
                             interpolate_state,
)
from .twin import (
                             DigitalTwin,
                             DigitalTwinConfig,
                             TwinMode,
                             TwinStatus,
                             connect_twins,
                             create_vehicle_twin,
                             validate_twin_fidelity,
)

__all__ = [
    # State Synchronization
    "StateVector",
    "SyncConfig",
    "StateSync",
    "StateSynchronizer",
    "SyncMode",
    "SyncStatus",
    "compute_state_divergence",
    "interpolate_state",
    "extrapolate_state",
    # Reduced-Order Models
    "ROMConfig",
    "ROMType",
    "ReducedOrderModel",
    "PODModel",
    "DMDModel",
    "AutoencoderROM",
    "create_rom_from_snapshots",
    "validate_rom_accuracy",
    "compute_projection_error",
    # Health Monitoring
    "HealthMetric",
    "HealthStatus",
    "AnomalyType",
    "HealthConfig",
    "HealthMonitor",
    "StructuralHealth",
    "ThermalHealth",
    "AnomalyDetector",
    "compute_damage_index",
    "estimate_thermal_margin",
    # Predictive Maintenance
    "MaintenanceConfig",
    "FailureMode",
    "RULEstimator",
    "MaintenanceScheduler",
    "PredictiveMaintenance",
    "estimate_remaining_life",
    "compute_reliability",
    "optimize_maintenance_schedule",
    # Digital Twin Core
    "DigitalTwinConfig",
    "TwinMode",
    "TwinStatus",
    "DigitalTwin",
    "create_vehicle_twin",
    "connect_twins",
    "validate_twin_fidelity",
]
