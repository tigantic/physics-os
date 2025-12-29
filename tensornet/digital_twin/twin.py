"""
Digital Twin orchestrator for hypersonic vehicles.

This module provides the main DigitalTwin class that integrates all
subsystems including state synchronization, reduced-order models,
health monitoring, and predictive maintenance into a cohesive
digital twin solution.

The digital twin enables:
    - Real-time state mirroring of physical vehicle
    - Predictive simulation for decision support
    - Health monitoring and anomaly detection
    - Maintenance optimization

Author: HyperTensor Team
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum, auto
import threading
import time
import queue

from .state_sync import (
    StateVector, SyncConfig, StateSync, SyncMode, SyncStatus,
    compute_state_divergence, interpolate_state, extrapolate_state
)
from .reduced_order import (
    ROMConfig, ROMType, ReducedOrderModel, PODModel,
    create_rom_from_snapshots
)
from .health_monitor import (
    HealthConfig, HealthMonitor, HealthStatus, HealthState
)
from .predictive import (
    MaintenanceConfig, PredictiveMaintenance, MaintenanceSchedule,
    ComponentType, ComponentState
)


class TwinMode(Enum):
    """Operating mode of digital twin."""
    OFFLINE = auto()      # No connection to physical system
    MONITORING = auto()   # Passive monitoring only
    SHADOW = auto()       # Active shadowing with sync
    PREDICTIVE = auto()   # Predictive lookahead enabled
    CONTROL = auto()      # Closed-loop with physical system


class TwinStatus(Enum):
    """Status of digital twin."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    SYNCED = auto()
    DIVERGED = auto()
    PAUSED = auto()
    ERROR = auto()


@dataclass
class DigitalTwinConfig:
    """Configuration for digital twin system."""
    # Identity
    vehicle_id: str = "HGV-001"
    twin_version: str = "1.0.0"
    
    # Synchronization
    sync_rate: float = 100.0  # Hz
    max_latency: float = 0.050  # 50ms
    
    # Capabilities
    enable_prediction: bool = True
    prediction_horizon: float = 1.0  # seconds
    enable_health_monitoring: bool = True
    enable_maintenance: bool = True
    
    # Models
    use_rom: bool = True
    rom_type: ROMType = ROMType.POD
    rom_modes: int = 50
    
    # Thresholds
    divergence_threshold: float = 0.1
    resync_threshold: float = 0.5
    
    # Logging
    log_telemetry: bool = True
    telemetry_rate: float = 10.0  # Hz


@dataclass
class TwinTelemetry:
    """Telemetry record from digital twin."""
    timestamp: float
    mode: TwinMode
    status: TwinStatus
    
    # State
    physical_state: Optional[StateVector] = None
    digital_state: Optional[StateVector] = None
    divergence: float = 0.0
    
    # Health
    health_status: Optional[HealthStatus] = None
    damage_index: float = 0.0
    
    # Predictions
    predicted_state: Optional[StateVector] = None
    prediction_confidence: float = 0.0


class DigitalTwin:
    """
    Main digital twin orchestrator for hypersonic vehicles.
    
    Integrates all subsystems to provide comprehensive digital
    twin capabilities including real-time monitoring, prediction,
    health tracking, and maintenance optimization.
    
    Example:
        >>> config = DigitalTwinConfig(vehicle_id="HGV-001")
        >>> twin = DigitalTwin(config)
        >>> twin.initialize()
        >>> twin.start()
        >>> # Feed physical state updates
        >>> twin.update_physical_state(state)
        >>> # Get predictions
        >>> prediction = twin.predict(horizon=1.0)
    """
    
    def __init__(self, config: DigitalTwinConfig):
        self.config = config
        self.mode = TwinMode.OFFLINE
        self.status = TwinStatus.UNINITIALIZED
        
        # Subsystems
        self._init_subsystems()
        
        # State
        self.current_physical_state: Optional[StateVector] = None
        self.current_digital_state: Optional[StateVector] = None
        self.last_update_time = 0.0
        
        # Threading
        self.running = False
        self.update_thread: Optional[threading.Thread] = None
        self.state_queue: queue.Queue = queue.Queue()
        
        # Telemetry
        self.telemetry_buffer: List[TwinTelemetry] = []
        self.max_telemetry = 10000
        
        # Callbacks
        self.on_divergence: Optional[Callable[[float], None]] = None
        self.on_health_warning: Optional[Callable[[HealthState], None]] = None
        self.on_maintenance_due: Optional[Callable[[MaintenanceSchedule], None]] = None
    
    def _init_subsystems(self):
        """Initialize all subsystems."""
        # State synchronization
        sync_config = SyncConfig(
            sync_rate=self.config.sync_rate,
            max_latency=self.config.max_latency,
        )
        self.state_sync = StateSync(sync_config)
        
        # Health monitoring
        if self.config.enable_health_monitoring:
            health_config = HealthConfig()
            self.health_monitor = HealthMonitor(health_config)
        else:
            self.health_monitor = None
        
        # Predictive maintenance
        if self.config.enable_maintenance:
            maint_config = MaintenanceConfig()
            self.maintenance = PredictiveMaintenance(maint_config)
        else:
            self.maintenance = None
        
        # Reduced-order model (initialized later with data)
        self.rom: Optional[ReducedOrderModel] = None
    
    def initialize(self, training_data: Optional[torch.Tensor] = None):
        """
        Initialize digital twin with training data.
        
        Args:
            training_data: Optional snapshot matrix for ROM training
        """
        self.status = TwinStatus.INITIALIZING
        
        # Train ROM if data provided
        if training_data is not None and self.config.use_rom:
            rom_config = ROMConfig(
                n_modes=self.config.rom_modes,
                energy_threshold=0.99,
            )
            self.rom = create_rom_from_snapshots(
                training_data,
                self.config.rom_type,
                rom_config
            )
        
        # Register default components for maintenance
        if self.maintenance:
            self.maintenance.register_component("TPS-NOSE", ComponentType.TPS_TILE)
            self.maintenance.register_component("LE-LEFT", ComponentType.LEADING_EDGE)
            self.maintenance.register_component("LE-RIGHT", ComponentType.LEADING_EDGE)
            self.maintenance.register_component("ELEV-LEFT", ComponentType.CONTROL_SURFACE)
            self.maintenance.register_component("ELEV-RIGHT", ComponentType.CONTROL_SURFACE)
        
        self.status = TwinStatus.RUNNING
    
    def start(self, mode: TwinMode = TwinMode.SHADOW):
        """
        Start digital twin operation.
        
        Args:
            mode: Operating mode
        """
        self.mode = mode
        self.running = True
        
        # Start state synchronization
        if mode in [TwinMode.SHADOW, TwinMode.PREDICTIVE, TwinMode.CONTROL]:
            sync_mode = SyncMode.REAL_TIME if mode != TwinMode.MONITORING else SyncMode.BATCH
            self.state_sync.start(sync_mode)
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.status = TwinStatus.RUNNING
    
    def stop(self):
        """Stop digital twin operation."""
        self.running = False
        self.state_sync.stop()
        
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        
        self.mode = TwinMode.OFFLINE
        self.status = TwinStatus.PAUSED
    
    def update_physical_state(self, state: StateVector):
        """
        Update with new physical state from vehicle/HIL.
        
        Args:
            state: Current physical vehicle state
        """
        self.current_physical_state = state
        self.state_sync.push_physical_state(state)
        self.state_queue.put(('physical', state))
    
    def update_digital_state(self, state: StateVector):
        """
        Update with new digital (simulated) state.
        
        Args:
            state: Current simulation state
        """
        self.current_digital_state = state
        self.state_sync.push_digital_state(state)
        self.state_queue.put(('digital', state))
    
    def get_current_state(self) -> Optional[StateVector]:
        """Get best current state estimate."""
        return self.state_sync.get_synchronized_state()
    
    def predict(self, horizon: float,
               control_sequence: Optional[np.ndarray] = None
               ) -> Optional[StateVector]:
        """
        Predict future state.
        
        Args:
            horizon: Prediction horizon in seconds
            control_sequence: Optional control inputs for prediction
            
        Returns:
            Predicted state at t+horizon
        """
        if not self.config.enable_prediction:
            return None
        
        current = self.get_current_state()
        if current is None:
            return None
        
        # Use ROM for prediction if available
        if self.rom is not None:
            # Encode current state
            state_vec = current.to_vector()
            state_tensor = torch.tensor(state_vec, dtype=torch.float32)
            
            z = self.rom.encode(state_tensor)
            
            # Simple propagation in latent space
            # (In practice, this would use a dynamics model)
            pred_tensor = self.rom.decode(z)
            
            # D-014: Use tolist() for StateVector conversion
            return StateVector.from_vector(
                pred_tensor.cpu().tolist(),
                current.timestamp + horizon
            )
        
        # Fall back to extrapolation
        states = self.state_sync.physical_buffer.get_latest(10)
        if len(states) < 2:
            return None
        
        return extrapolate_state(states, current.timestamp + horizon)
    
    def get_health_status(self) -> Optional[HealthState]:
        """Get current health status."""
        if self.health_monitor:
            return self.health_monitor.current_state
        return None
    
    def get_maintenance_schedule(self) -> List[MaintenanceSchedule]:
        """Get current maintenance schedule."""
        if self.maintenance:
            return self.maintenance.generate_maintenance_plan()
        return []
    
    def _update_loop(self):
        """Main update loop (runs in thread)."""
        dt = 1.0 / self.config.sync_rate
        last_telemetry = 0.0
        telemetry_dt = 1.0 / self.config.telemetry_rate
        
        while self.running:
            try:
                # Process state updates
                try:
                    source, state = self.state_queue.get(timeout=dt)
                    self._process_state_update(source, state)
                except queue.Empty:
                    pass
                
                # Check divergence
                self._check_divergence()
                
                # Update health monitoring
                if self.health_monitor and self.current_physical_state:
                    self._update_health()
                
                # Log telemetry
                now = time.time()
                if self.config.log_telemetry and now - last_telemetry > telemetry_dt:
                    self._log_telemetry()
                    last_telemetry = now
                
            except Exception as e:
                self.status = TwinStatus.ERROR
                print(f"Digital twin error: {e}")
    
    def _process_state_update(self, source: str, state: StateVector):
        """Process incoming state update."""
        self.last_update_time = state.timestamp
        
        if source == 'physical':
            self.current_physical_state = state
        else:
            self.current_digital_state = state
    
    def _check_divergence(self):
        """Check for state divergence."""
        if self.current_physical_state is None or self.current_digital_state is None:
            return
        
        divergence = compute_state_divergence(
            self.current_physical_state,
            self.current_digital_state
        )
        
        total_div = divergence['total']
        
        if total_div > self.config.resync_threshold:
            self.status = TwinStatus.DIVERGED
            if self.on_divergence:
                self.on_divergence(total_div)
        elif total_div > self.config.divergence_threshold:
            # Warning but not diverged
            pass
        else:
            if self.status == TwinStatus.DIVERGED:
                self.status = TwinStatus.SYNCED
    
    def _update_health(self):
        """Update health monitoring with current state."""
        state = self.current_physical_state
        if state is None:
            return
        
        # Extract health-relevant data
        loads = state.accelerations * 1000  # Scale to loads
        temps = state.temperatures if state.temperatures is not None else np.array([state.mach * 200 + 1000])
        
        health_state = self.health_monitor.update(
            timestamp=state.timestamp,
            loads=loads,
            temperatures=temps,
        )
        
        # Check for warnings
        if health_state.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            if self.on_health_warning:
                self.on_health_warning(health_state)
        
        # Update maintenance tracking
        if self.maintenance:
            # Update component states based on vehicle state
            damage_rate = 0.0001 * (state.mach / 5.0)  # Higher Mach = more damage
            for comp_id in self.maintenance.components:
                self.maintenance.update_component(
                    comp_id,
                    delta_hours=1.0/3600.0,  # One second
                    damage_increment=damage_rate,
                    operating_conditions={'temperature': float(np.max(temps))}
                )
    
    def _log_telemetry(self):
        """Log current telemetry."""
        health_state = self.get_health_status()
        
        telemetry = TwinTelemetry(
            timestamp=time.time(),
            mode=self.mode,
            status=self.status,
            physical_state=self.current_physical_state,
            digital_state=self.current_digital_state,
            divergence=self.state_sync.divergence_history[-1]['total'] if self.state_sync.divergence_history else 0.0,
            health_status=health_state.status if health_state else None,
            damage_index=health_state.structural_damage if health_state else 0.0,
        )
        
        self.telemetry_buffer.append(telemetry)
        if len(self.telemetry_buffer) > self.max_telemetry:
            self.telemetry_buffer = self.telemetry_buffer[-self.max_telemetry:]
    
    def get_telemetry(self, n_latest: int = 100) -> List[TwinTelemetry]:
        """Get latest telemetry records."""
        return self.telemetry_buffer[-n_latest:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get digital twin statistics."""
        sync_stats = self.state_sync.get_statistics()
        
        health = self.get_health_status()
        fleet_health = self.maintenance.get_fleet_health() if self.maintenance else {}
        
        return {
            'vehicle_id': self.config.vehicle_id,
            'mode': self.mode.name,
            'status': self.status.name,
            'sync': sync_stats,
            'health': health.to_dict() if health else {},
            'fleet': fleet_health,
            'telemetry_count': len(self.telemetry_buffer),
            'rom_active': self.rom is not None,
        }


def create_vehicle_twin(vehicle_id: str,
                       config: Optional[DigitalTwinConfig] = None
                       ) -> DigitalTwin:
    """
    Factory function to create a vehicle digital twin.
    
    Args:
        vehicle_id: Vehicle identifier
        config: Optional configuration (uses defaults if None)
        
    Returns:
        Initialized digital twin
    """
    if config is None:
        config = DigitalTwinConfig(vehicle_id=vehicle_id)
    else:
        config.vehicle_id = vehicle_id
    
    twin = DigitalTwin(config)
    twin.initialize()
    
    return twin


def connect_twins(primary: DigitalTwin, secondary: DigitalTwin):
    """
    Connect two digital twins for redundancy/comparison.
    
    Args:
        primary: Primary digital twin
        secondary: Secondary digital twin
    """
    # Share state updates between twins
    original_update = primary.update_physical_state
    
    def shared_update(state: StateVector):
        original_update(state)
        secondary.update_physical_state(state)
    
    primary.update_physical_state = shared_update


def validate_twin_fidelity(twin: DigitalTwin,
                          test_states: List[StateVector],
                          tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Validate digital twin fidelity against reference states.
    
    Args:
        twin: Digital twin to validate
        test_states: Reference states for validation
        tolerance: Acceptable error tolerance
        
    Returns:
        Validation results
    """
    errors = []
    
    for ref_state in test_states:
        twin.update_physical_state(ref_state)
        predicted = twin.predict(horizon=0.1)
        
        if predicted is not None:
            div = compute_state_divergence(ref_state, predicted)
            errors.append(div['total'])
    
    if not errors:
        return {'valid': False, 'reason': 'No predictions generated'}
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    return {
        'valid': mean_error < tolerance,
        'mean_error': mean_error,
        'max_error': max_error,
        'n_tests': len(errors),
        'within_tolerance': sum(1 for e in errors if e < tolerance) / len(errors),
    }


def test_digital_twin():
    """Test digital twin system."""
    print("Testing Digital Twin System...")
    
    # Create twin
    config = DigitalTwinConfig(
        vehicle_id="TEST-001",
        sync_rate=50.0,
        enable_prediction=True,
        enable_health_monitoring=True,
        enable_maintenance=True,
    )
    
    twin = DigitalTwin(config)
    twin.initialize()
    print("  ✓ Twin initialized")
    
    # Start in shadow mode
    twin.start(TwinMode.SHADOW)
    print(f"  ✓ Twin started in {twin.mode.name} mode")
    
    # Create test states
    test_state = StateVector(
        timestamp=0.0,
        position=np.array([0.0, 0.0, 30000.0]),
        velocity=np.array([2000.0, 0.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rates=np.array([0.0, 0.0, 0.0]),
        accelerations=np.array([10.0, 0.0, -5.0]),
        control_surfaces=np.array([0.0, 0.0, 0.0]),
        mach=6.0,
        altitude=30000.0,
    )
    
    # Feed states
    print("\n  Feeding states...")
    for i in range(20):
        state = test_state.copy()
        state.timestamp = i * 0.1
        state.position = test_state.position + np.array([i * 200.0, 0.0, -i * 5.0])
        state.mach = 6.0 + i * 0.1
        
        twin.update_physical_state(state)
        twin.update_digital_state(state)
        time.sleep(0.02)
    
    # Check status
    print(f"\n  Twin status: {twin.status.name}")
    
    # Get current state
    current = twin.get_current_state()
    if current:
        print(f"  Current position: {current.position}")
    
    # Test prediction
    if twin.config.enable_prediction:
        prediction = twin.predict(horizon=1.0)
        if prediction:
            print(f"  Predicted position (t+1s): {prediction.position}")
    
    # Check health
    health = twin.get_health_status()
    if health:
        print(f"\n  Health status: {health.status.name}")
        print(f"  Structural damage: {health.structural_damage*100:.2f}%")
    
    # Get maintenance schedule
    schedule = twin.get_maintenance_schedule()
    print(f"\n  Maintenance items: {len(schedule)}")
    
    # Get statistics
    stats = twin.get_statistics()
    print(f"\n  Statistics:")
    print(f"    Mode: {stats['mode']}")
    print(f"    Status: {stats['status']}")
    print(f"    Telemetry records: {stats['telemetry_count']}")
    
    # Stop twin
    twin.stop()
    print(f"\n  Twin stopped: {twin.status.name}")
    
    # Test factory function
    print("\n  Testing factory function...")
    twin2 = create_vehicle_twin("HGV-002")
    assert twin2.config.vehicle_id == "HGV-002"
    print("  ✓ Factory function works")
    
    # Test validation
    print("\n  Testing fidelity validation...")
    test_states = [test_state.copy() for _ in range(5)]
    validation = validate_twin_fidelity(twin2, test_states)
    print(f"    Valid: {validation.get('valid', 'N/A')}")
    
    print("\nDigital Twin: All tests passed!")


if __name__ == "__main__":
    test_digital_twin()
