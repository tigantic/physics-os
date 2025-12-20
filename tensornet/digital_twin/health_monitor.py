"""
Health monitoring for hypersonic vehicle digital twins.

This module provides real-time structural and thermal health monitoring
with anomaly detection capabilities. It enables predictive maintenance
by tracking cumulative damage and detecting off-nominal conditions.

Key capabilities:
    - Structural health monitoring with damage accumulation
    - Thermal protection system integrity assessment
    - Anomaly detection using statistical and ML methods
    - Fatigue life estimation and monitoring

Author: HyperTensor Team
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum, auto
from collections import deque
import warnings


class HealthStatus(Enum):
    """Overall health status of vehicle system."""
    NOMINAL = auto()      # All systems within limits
    CAUTION = auto()      # Approaching limits
    WARNING = auto()       # Limits exceeded, recoverable
    CRITICAL = auto()      # Immediate action required
    FAILED = auto()        # System failure


class AnomalyType(Enum):
    """Type of detected anomaly."""
    NONE = auto()
    SENSOR_DRIFT = auto()
    SENSOR_FAILURE = auto()
    STRUCTURAL_DAMAGE = auto()
    THERMAL_EXCEEDANCE = auto()
    VIBRATION_ANOMALY = auto()
    CONTROL_ANOMALY = auto()
    UNKNOWN = auto()


class HealthMetric(Enum):
    """Specific health metrics tracked."""
    STRUCTURAL_INTEGRITY = auto()
    THERMAL_MARGIN = auto()
    FATIGUE_LIFE = auto()
    TPS_INTEGRITY = auto()
    VIBRATION_LEVEL = auto()
    CONTROL_AUTHORITY = auto()


@dataclass
class HealthConfig:
    """Configuration for health monitoring."""
    # Thresholds
    structural_warning: float = 0.8  # Damage index
    structural_critical: float = 0.95
    thermal_margin_warning: float = 0.1  # 10% margin
    thermal_margin_critical: float = 0.02
    
    # Fatigue parameters
    fatigue_exponent: float = 3.0  # S-N curve exponent
    fatigue_limit_cycles: float = 1e7  # Endurance limit cycles
    
    # Anomaly detection
    anomaly_window: int = 100  # Samples for baseline
    anomaly_threshold: float = 3.0  # Standard deviations
    
    # Update rates
    update_interval: float = 0.1  # seconds
    
    # TPS limits (K)
    tps_design_temp: float = 2000.0  # Design temperature
    tps_max_temp: float = 2200.0     # Maximum allowable
    
    # Vibration limits (g)
    vibration_warning: float = 5.0
    vibration_critical: float = 10.0


@dataclass
class HealthState:
    """Current health state of vehicle."""
    timestamp: float
    status: HealthStatus
    
    # Damage indices (0-1, 1=failure)
    structural_damage: float = 0.0
    tps_damage: float = 0.0
    fatigue_fraction: float = 0.0
    
    # Current values
    max_temperature: float = 0.0
    max_vibration: float = 0.0
    thermal_margin: float = 1.0
    
    # Anomaly state
    anomaly_detected: bool = False
    anomaly_type: AnomalyType = AnomalyType.NONE
    anomaly_confidence: float = 0.0
    
    # Predictions
    remaining_life_hours: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'status': self.status.name,
            'structural_damage': self.structural_damage,
            'tps_damage': self.tps_damage,
            'fatigue_fraction': self.fatigue_fraction,
            'max_temperature': self.max_temperature,
            'max_vibration': self.max_vibration,
            'thermal_margin': self.thermal_margin,
            'anomaly_detected': self.anomaly_detected,
            'anomaly_type': self.anomaly_type.name,
            'remaining_life_hours': self.remaining_life_hours,
        }


class HealthMonitor:
    """
    Comprehensive health monitoring system for hypersonic vehicles.
    
    Tracks structural integrity, thermal state, fatigue accumulation,
    and performs anomaly detection across multiple subsystems.
    """
    
    def __init__(self, config: HealthConfig):
        self.config = config
        
        # Current state
        self.current_state = HealthState(
            timestamp=0.0,
            status=HealthStatus.NOMINAL
        )
        
        # History
        self.state_history: List[HealthState] = []
        self.max_history_size = 10000
        
        # Cumulative damage tracking
        self.cumulative_fatigue = 0.0
        self.cumulative_thermal_cycles = 0.0
        self.peak_load_history: deque = deque(maxlen=1000)
        self.peak_temp_history: deque = deque(maxlen=1000)
        
        # Anomaly detection baselines
        self.baseline_established = False
        self.baseline_mean: Dict[str, float] = {}
        self.baseline_std: Dict[str, float] = {}
        self.measurement_buffer: Dict[str, deque] = {}
        
        # Subsystem monitors
        self.structural = StructuralHealth(config)
        self.thermal = ThermalHealth(config)
        self.anomaly_detector = AnomalyDetector(config)
    
    def update(self, timestamp: float,
              loads: Optional[np.ndarray] = None,
              temperatures: Optional[np.ndarray] = None,
              vibrations: Optional[np.ndarray] = None,
              control_state: Optional[np.ndarray] = None) -> HealthState:
        """
        Update health state with new measurements.
        
        Args:
            timestamp: Current time
            loads: Structural loads array
            temperatures: Temperature distribution
            vibrations: Vibration measurements
            control_state: Control surface states
            
        Returns:
            Updated health state
        """
        # Update structural health
        if loads is not None:
            self.structural.update(loads)
            self.peak_load_history.append(np.max(np.abs(loads)))
        
        # Update thermal health
        if temperatures is not None:
            self.thermal.update(temperatures)
            self.peak_temp_history.append(np.max(temperatures))
        
        # Update anomaly detection
        measurements = {}
        if loads is not None:
            measurements['load_max'] = np.max(np.abs(loads))
            measurements['load_mean'] = np.mean(np.abs(loads))
        if temperatures is not None:
            measurements['temp_max'] = np.max(temperatures)
            measurements['temp_mean'] = np.mean(temperatures)
        if vibrations is not None:
            measurements['vib_rms'] = np.sqrt(np.mean(vibrations**2))
        
        anomaly_result = self.anomaly_detector.check(measurements)
        
        # Compute cumulative fatigue
        if loads is not None:
            self._update_fatigue(loads)
        
        # Determine overall status
        status = self._compute_status()
        
        # Create new state
        self.current_state = HealthState(
            timestamp=timestamp,
            status=status,
            structural_damage=self.structural.damage_index,
            tps_damage=self.thermal.tps_damage,
            fatigue_fraction=self.cumulative_fatigue,
            max_temperature=self.thermal.max_temp,
            max_vibration=measurements.get('vib_rms', 0.0),
            thermal_margin=self.thermal.thermal_margin,
            anomaly_detected=anomaly_result['detected'],
            anomaly_type=anomaly_result['type'],
            anomaly_confidence=anomaly_result['confidence'],
            remaining_life_hours=self._estimate_remaining_life(),
        )
        
        # Store history
        self.state_history.append(self.current_state)
        if len(self.state_history) > self.max_history_size:
            self.state_history = self.state_history[-self.max_history_size:]
        
        return self.current_state
    
    def _update_fatigue(self, loads: np.ndarray):
        """Update cumulative fatigue damage."""
        # Rainflow counting approximation
        load_range = np.max(loads) - np.min(loads)
        
        if load_range > 0:
            # Miner's rule: D = sum(n_i / N_i)
            # N = K / S^m where S is stress range
            cycles = 1  # One cycle per update
            N = self.config.fatigue_limit_cycles / (load_range / 100.0) ** self.config.fatigue_exponent
            self.cumulative_fatigue += cycles / max(N, 1.0)
            self.cumulative_fatigue = min(self.cumulative_fatigue, 1.0)
    
    def _compute_status(self) -> HealthStatus:
        """Compute overall health status."""
        # Check structural
        if self.structural.damage_index > self.config.structural_critical:
            return HealthStatus.CRITICAL
        if self.structural.damage_index > self.config.structural_warning:
            return HealthStatus.WARNING
        
        # Check thermal
        if self.thermal.thermal_margin < self.config.thermal_margin_critical:
            return HealthStatus.CRITICAL
        if self.thermal.thermal_margin < self.config.thermal_margin_warning:
            return HealthStatus.WARNING
        
        # Check fatigue
        if self.cumulative_fatigue > 0.95:
            return HealthStatus.CRITICAL
        if self.cumulative_fatigue > 0.8:
            return HealthStatus.WARNING
        
        # Check for anomalies
        if self.anomaly_detector.last_result.get('detected', False):
            confidence = self.anomaly_detector.last_result.get('confidence', 0.0)
            if confidence > 0.9:
                return HealthStatus.WARNING
            elif confidence > 0.7:
                return HealthStatus.CAUTION
        
        return HealthStatus.NOMINAL
    
    def _estimate_remaining_life(self) -> Optional[float]:
        """Estimate remaining useful life in hours."""
        if self.cumulative_fatigue >= 1.0:
            return 0.0
        
        if len(self.state_history) < 10:
            return None
        
        # Estimate fatigue rate from history
        recent = self.state_history[-100:]
        if len(recent) < 2:
            return None
        
        dt = recent[-1].timestamp - recent[0].timestamp
        df = recent[-1].fatigue_fraction - recent[0].fatigue_fraction
        
        if dt <= 0 or df <= 0:
            return None
        
        fatigue_rate = df / dt  # per second
        remaining = 1.0 - self.cumulative_fatigue
        
        remaining_seconds = remaining / fatigue_rate
        return remaining_seconds / 3600.0  # Convert to hours
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current health state."""
        return {
            'status': self.current_state.status.name,
            'structural_damage': f"{self.current_state.structural_damage*100:.1f}%",
            'tps_damage': f"{self.current_state.tps_damage*100:.1f}%",
            'fatigue_consumed': f"{self.current_state.fatigue_fraction*100:.1f}%",
            'thermal_margin': f"{self.current_state.thermal_margin*100:.1f}%",
            'max_temperature': f"{self.current_state.max_temperature:.0f} K",
            'remaining_life': (f"{self.current_state.remaining_life_hours:.1f} hrs"
                              if self.current_state.remaining_life_hours else "N/A"),
            'anomaly': self.current_state.anomaly_type.name if self.current_state.anomaly_detected else "None",
        }


class StructuralHealth:
    """
    Structural health monitoring subsystem.
    
    Tracks load history, estimates damage accumulation,
    and monitors for structural anomalies.
    """
    
    def __init__(self, config: HealthConfig):
        self.config = config
        
        # State
        self.damage_index = 0.0
        self.peak_load = 0.0
        self.load_history: deque = deque(maxlen=1000)
        
        # Damage accumulation parameters
        self.design_load = 1000.0  # Reference design load
        self.damage_rate = 1e-6   # Damage per unit load
    
    def update(self, loads: np.ndarray):
        """Update with new load measurements."""
        max_load = np.max(np.abs(loads))
        self.peak_load = max(self.peak_load, max_load)
        self.load_history.append(max_load)
        
        # Accumulate damage (simplified model)
        if max_load > self.design_load * 0.5:
            # Nonlinear damage above threshold
            overload_ratio = max_load / self.design_load
            damage_increment = self.damage_rate * overload_ratio ** 2
            self.damage_index += damage_increment
            self.damage_index = min(self.damage_index, 1.0)
    
    def reset(self):
        """Reset damage tracking (e.g., after repair)."""
        self.damage_index = 0.0
        self.peak_load = 0.0
        self.load_history.clear()


class ThermalHealth:
    """
    Thermal protection system health monitoring.
    
    Tracks temperature history, thermal cycling, and
    estimates TPS degradation.
    """
    
    def __init__(self, config: HealthConfig):
        self.config = config
        
        # State
        self.tps_damage = 0.0
        self.max_temp = 0.0
        self.thermal_margin = 1.0
        self.temp_history: deque = deque(maxlen=1000)
        
        # Thermal cycling
        self.last_temp = None
        self.thermal_cycles = 0
    
    def update(self, temperatures: np.ndarray):
        """Update with new temperature measurements."""
        current_max = np.max(temperatures)
        self.max_temp = max(self.max_temp, current_max)
        self.temp_history.append(current_max)
        
        # Thermal margin (distance to limit)
        self.thermal_margin = max(0.0, 1.0 - current_max / self.config.tps_max_temp)
        
        # Detect thermal cycles
        if self.last_temp is not None:
            if current_max > self.last_temp + 100:  # Heating
                self.thermal_cycles += 0.5
            elif current_max < self.last_temp - 100:  # Cooling
                self.thermal_cycles += 0.5
        self.last_temp = current_max
        
        # TPS damage accumulation
        if current_max > self.config.tps_design_temp:
            overtemp_ratio = current_max / self.config.tps_design_temp
            damage_increment = 1e-5 * (overtemp_ratio - 1.0) ** 2
            self.tps_damage += damage_increment
            self.tps_damage = min(self.tps_damage, 1.0)


class AnomalyDetector:
    """
    Statistical anomaly detection system.
    
    Uses baseline statistics and deviation thresholds to
    detect off-nominal behavior across multiple channels.
    """
    
    def __init__(self, config: HealthConfig):
        self.config = config
        
        # Baseline statistics
        self.baseline_mean: Dict[str, float] = {}
        self.baseline_std: Dict[str, float] = {}
        self.baseline_established = False
        
        # Data buffers
        self.buffers: Dict[str, deque] = {}
        
        # Last result
        self.last_result: Dict[str, Any] = {
            'detected': False,
            'type': AnomalyType.NONE,
            'confidence': 0.0,
        }
    
    def check(self, measurements: Dict[str, float]) -> Dict[str, Any]:
        """
        Check measurements for anomalies.
        
        Args:
            measurements: Dictionary of current measurements
            
        Returns:
            Anomaly detection result
        """
        # Update buffers
        for key, value in measurements.items():
            if key not in self.buffers:
                self.buffers[key] = deque(maxlen=self.config.anomaly_window)
            self.buffers[key].append(value)
        
        # Establish baseline if not done
        if not self.baseline_established:
            if all(len(buf) >= self.config.anomaly_window 
                   for buf in self.buffers.values()):
                self._establish_baseline()
            return {'detected': False, 'type': AnomalyType.NONE, 'confidence': 0.0}
        
        # Check for anomalies
        max_deviation = 0.0
        anomaly_channel = None
        
        for key, value in measurements.items():
            if key in self.baseline_mean and key in self.baseline_std:
                mean = self.baseline_mean[key]
                std = self.baseline_std[key] + 1e-8
                deviation = abs(value - mean) / std
                
                if deviation > max_deviation:
                    max_deviation = deviation
                    anomaly_channel = key
        
        # Determine anomaly type
        detected = max_deviation > self.config.anomaly_threshold
        anomaly_type = AnomalyType.NONE
        
        if detected and anomaly_channel:
            if 'temp' in anomaly_channel:
                anomaly_type = AnomalyType.THERMAL_EXCEEDANCE
            elif 'load' in anomaly_channel:
                anomaly_type = AnomalyType.STRUCTURAL_DAMAGE
            elif 'vib' in anomaly_channel:
                anomaly_type = AnomalyType.VIBRATION_ANOMALY
            else:
                anomaly_type = AnomalyType.UNKNOWN
        
        # Compute confidence
        if max_deviation > 0:
            confidence = min(1.0, (max_deviation - self.config.anomaly_threshold) / 
                           self.config.anomaly_threshold + 0.5) if detected else 0.0
        else:
            confidence = 0.0
        
        self.last_result = {
            'detected': detected,
            'type': anomaly_type,
            'confidence': confidence,
            'max_deviation': max_deviation,
            'channel': anomaly_channel,
        }
        
        return self.last_result
    
    def _establish_baseline(self):
        """Compute baseline statistics from buffer."""
        for key, buffer in self.buffers.items():
            values = np.array(list(buffer))
            self.baseline_mean[key] = np.mean(values)
            self.baseline_std[key] = np.std(values)
        
        self.baseline_established = True
    
    def reset_baseline(self):
        """Reset baseline (e.g., after system change)."""
        self.baseline_established = False
        self.baseline_mean.clear()
        self.baseline_std.clear()
        for buffer in self.buffers.values():
            buffer.clear()


def compute_damage_index(load_history: np.ndarray,
                        design_load: float,
                        fatigue_exponent: float = 3.0) -> float:
    """
    Compute damage index from load history using Miner's rule.
    
    Args:
        load_history: Array of load values
        design_load: Reference design load
        fatigue_exponent: S-N curve exponent
        
    Returns:
        Cumulative damage index (0-1)
    """
    damage = 0.0
    
    for load in load_history:
        if load > 0:
            load_ratio = load / design_load
            N = 1e7 / (load_ratio ** fatigue_exponent)
            damage += 1.0 / max(N, 1.0)
    
    return min(damage, 1.0)


def estimate_thermal_margin(temperatures: np.ndarray,
                           limit_temp: float,
                           safety_factor: float = 1.1) -> float:
    """
    Estimate thermal margin from temperature distribution.
    
    Args:
        temperatures: Current temperature array
        limit_temp: Temperature limit
        safety_factor: Safety factor for margin calculation
        
    Returns:
        Thermal margin (0-1, 0=at limit)
    """
    max_temp = np.max(temperatures)
    effective_limit = limit_temp / safety_factor
    
    margin = 1.0 - max_temp / effective_limit
    return max(0.0, min(1.0, margin))


def test_health_monitor():
    """Test health monitoring module."""
    print("Testing Health Monitoring Module...")
    
    # Create monitor
    config = HealthConfig()
    monitor = HealthMonitor(config)
    
    # Simulate nominal operation
    print("\n  Simulating nominal operation...")
    for i in range(50):
        loads = np.random.randn(10) * 100 + 500  # Nominal loads
        temps = np.random.randn(10) * 50 + 1500   # Nominal temps
        vibs = np.random.randn(3) * 1.0           # Nominal vibs
        
        state = monitor.update(
            timestamp=i * 0.1,
            loads=loads,
            temperatures=temps,
            vibrations=vibs,
        )
    
    assert state.status in [HealthStatus.NOMINAL, HealthStatus.CAUTION]
    print(f"    Status: {state.status.name}")
    print(f"    Structural damage: {state.structural_damage*100:.2f}%")
    
    # Simulate high load event
    print("\n  Simulating high load event...")
    for i in range(10):
        loads = np.random.randn(10) * 100 + 1500  # High loads
        temps = np.random.randn(10) * 50 + 2100   # High temps
        
        state = monitor.update(
            timestamp=50 * 0.1 + i * 0.1,
            loads=loads,
            temperatures=temps,
        )
    
    print(f"    Status after event: {state.status.name}")
    print(f"    Thermal margin: {state.thermal_margin*100:.1f}%")
    
    # Test damage index computation
    load_history = np.array([500, 800, 1200, 600, 1500])
    damage = compute_damage_index(load_history, design_load=1000.0)
    assert 0 <= damage <= 1
    print(f"\n  Damage index: {damage:.6f}")
    
    # Test thermal margin
    temps = np.array([1800, 1900, 1850, 1700])
    margin = estimate_thermal_margin(temps, limit_temp=2200.0)
    assert 0 <= margin <= 1
    print(f"  Thermal margin: {margin*100:.1f}%")
    
    # Test anomaly detection
    detector = AnomalyDetector(config)
    
    # Build baseline
    for i in range(100):
        detector.check({'test': np.random.randn() * 10 + 100})
    
    # Check normal value
    result = detector.check({'test': 105})
    assert not result['detected']
    
    # Check anomalous value
    result = detector.check({'test': 200})  # Far from baseline
    print(f"\n  Anomaly detection test: deviation={result.get('max_deviation', 0):.1f}")
    
    # Test health summary
    summary = monitor.get_summary()
    print(f"\n  Health Summary:")
    for key, value in summary.items():
        print(f"    {key}: {value}")
    
    print("\nHealth Monitoring: All tests passed!")


if __name__ == "__main__":
    test_health_monitor()
