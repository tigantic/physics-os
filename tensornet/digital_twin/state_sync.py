"""
Real-time state synchronization for digital twin systems.

This module provides the infrastructure for synchronizing state between
physical vehicles and their digital counterparts, enabling real-time
monitoring, prediction, and control optimization.

Key features:
    - Low-latency state transfer with configurable update rates
    - Automatic state interpolation and extrapolation for timing mismatches
    - Divergence detection and correction mechanisms
    - Network resilience with buffering and recovery
    - Multi-fidelity state representation

Author: HyperTensor Team
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from enum import Enum, auto
import time
import threading
import queue
from collections import deque


class SyncMode(Enum):
    """Synchronization mode for digital twin."""
    REAL_TIME = auto()      # Continuous real-time sync
    BATCH = auto()          # Periodic batch updates
    EVENT_DRIVEN = auto()   # Sync on state changes
    PREDICTIVE = auto()     # Prediction with correction
    RECORD = auto()         # Record for post-analysis


class SyncStatus(Enum):
    """Status of synchronization process."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    SYNCED = auto()
    DIVERGED = auto()
    RECOVERING = auto()
    ERROR = auto()


@dataclass
class StateVector:
    """
    Complete state vector for hypersonic vehicle.
    
    Represents the full state including position, velocity, attitude,
    rates, and internal states for both physical and digital systems.
    """
    timestamp: float
    
    # Position (Earth-fixed, m)
    position: np.ndarray  # [x, y, z]
    
    # Velocity (body-fixed, m/s)
    velocity: np.ndarray  # [u, v, w]
    
    # Attitude (quaternion)
    quaternion: np.ndarray  # [q0, q1, q2, q3]
    
    # Angular rates (body-fixed, rad/s)
    angular_rates: np.ndarray  # [p, q, r]
    
    # Accelerations (body-fixed, m/s^2)
    accelerations: np.ndarray  # [ax, ay, az]
    
    # Control surfaces (rad)
    control_surfaces: np.ndarray  # [elevator, aileron, rudder, ...]
    
    # Propulsion state
    thrust: float = 0.0
    fuel_mass: float = 0.0
    
    # Thermal state (K)
    temperatures: Optional[np.ndarray] = None
    
    # Structural loads
    loads: Optional[np.ndarray] = None
    
    # Atmospheric state
    mach: float = 0.0
    dynamic_pressure: float = 0.0
    altitude: float = 0.0
    
    # Confidence/uncertainty
    covariance: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Ensure numpy arrays."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=np.float64)
        if not isinstance(self.quaternion, np.ndarray):
            self.quaternion = np.array(self.quaternion, dtype=np.float64)
        if not isinstance(self.angular_rates, np.ndarray):
            self.angular_rates = np.array(self.angular_rates, dtype=np.float64)
        if not isinstance(self.accelerations, np.ndarray):
            self.accelerations = np.array(self.accelerations, dtype=np.float64)
        if not isinstance(self.control_surfaces, np.ndarray):
            self.control_surfaces = np.array(self.control_surfaces, dtype=np.float64)
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat state vector."""
        components = [
            self.position,
            self.velocity,
            self.quaternion,
            self.angular_rates,
            self.accelerations,
            self.control_surfaces,
            np.array([self.thrust, self.fuel_mass, self.mach,
                     self.dynamic_pressure, self.altitude]),
        ]
        if self.temperatures is not None:
            components.append(self.temperatures)
        if self.loads is not None:
            components.append(self.loads)
        return np.concatenate(components)
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, timestamp: float,
                   n_control: int = 3, n_temps: int = 0,
                   n_loads: int = 0) -> 'StateVector':
        """Reconstruct from flat vector."""
        idx = 0
        
        position = vector[idx:idx+3]; idx += 3
        velocity = vector[idx:idx+3]; idx += 3
        quaternion = vector[idx:idx+4]; idx += 4
        angular_rates = vector[idx:idx+3]; idx += 3
        accelerations = vector[idx:idx+3]; idx += 3
        control_surfaces = vector[idx:idx+n_control]; idx += n_control
        
        thrust = vector[idx]; idx += 1
        fuel_mass = vector[idx]; idx += 1
        mach = vector[idx]; idx += 1
        dynamic_pressure = vector[idx]; idx += 1
        altitude = vector[idx]; idx += 1
        
        temperatures = None
        loads = None
        
        if n_temps > 0:
            temperatures = vector[idx:idx+n_temps]; idx += n_temps
        if n_loads > 0:
            loads = vector[idx:idx+n_loads]
        
        return cls(
            timestamp=timestamp,
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            angular_rates=angular_rates,
            accelerations=accelerations,
            control_surfaces=control_surfaces,
            thrust=thrust,
            fuel_mass=fuel_mass,
            mach=mach,
            dynamic_pressure=dynamic_pressure,
            altitude=altitude,
            temperatures=temperatures,
            loads=loads,
        )
    
    def copy(self) -> 'StateVector':
        """Create deep copy."""
        return StateVector(
            timestamp=self.timestamp,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            angular_rates=self.angular_rates.copy(),
            accelerations=self.accelerations.copy(),
            control_surfaces=self.control_surfaces.copy(),
            thrust=self.thrust,
            fuel_mass=self.fuel_mass,
            temperatures=self.temperatures.copy() if self.temperatures is not None else None,
            loads=self.loads.copy() if self.loads is not None else None,
            mach=self.mach,
            dynamic_pressure=self.dynamic_pressure,
            altitude=self.altitude,
            covariance=self.covariance.copy() if self.covariance is not None else None,
        )


@dataclass
class SyncConfig:
    """Configuration for state synchronization."""
    # Timing
    sync_rate: float = 100.0  # Hz
    max_latency: float = 0.050  # 50ms max
    timeout: float = 1.0  # Connection timeout
    
    # Buffer settings
    buffer_size: int = 1000  # State history size
    interpolation_window: int = 5  # States for interpolation
    
    # Divergence thresholds
    position_threshold: float = 10.0  # m
    velocity_threshold: float = 5.0  # m/s
    attitude_threshold: float = 0.1  # rad
    
    # Recovery
    max_divergence_time: float = 0.5  # s
    recovery_gain: float = 0.1
    
    # Network
    retry_attempts: int = 3
    heartbeat_interval: float = 0.1  # s
    
    # Prediction
    enable_prediction: bool = True
    prediction_horizon: float = 0.1  # s


class StateBuffer:
    """
    Thread-safe circular buffer for state history.
    
    Maintains temporal history of states for interpolation,
    extrapolation, and analysis purposes.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def push(self, state: StateVector):
        """Add state to buffer."""
        with self.lock:
            self.buffer.append(state)
    
    def get_latest(self, n: int = 1) -> List[StateVector]:
        """Get n most recent states."""
        with self.lock:
            if len(self.buffer) == 0:
                return []
            n = min(n, len(self.buffer))
            return list(self.buffer)[-n:]
    
    def get_at_time(self, t: float) -> Optional[StateVector]:
        """Get state closest to time t."""
        with self.lock:
            if len(self.buffer) == 0:
                return None
            
            # Binary search for closest time
            times = [s.timestamp for s in self.buffer]
            idx = np.searchsorted(times, t)
            
            if idx == 0:
                return self.buffer[0]
            elif idx >= len(self.buffer):
                return self.buffer[-1]
            else:
                # Return closer of two neighbors
                if t - times[idx-1] < times[idx] - t:
                    return self.buffer[idx-1]
                else:
                    return self.buffer[idx]
    
    def get_range(self, t_start: float, t_end: float) -> List[StateVector]:
        """Get states in time range."""
        with self.lock:
            return [s for s in self.buffer
                   if t_start <= s.timestamp <= t_end]
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()
    
    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)


def interpolate_state(state1: StateVector, state2: StateVector,
                     t: float) -> StateVector:
    """
    Interpolate state between two time points.
    
    Uses linear interpolation for positions and velocities,
    SLERP for quaternions, and linear for other quantities.
    
    Args:
        state1: Earlier state
        state2: Later state
        t: Target time
        
    Returns:
        Interpolated state at time t
    """
    if state1.timestamp > state2.timestamp:
        state1, state2 = state2, state1
    
    # Compute interpolation factor
    dt = state2.timestamp - state1.timestamp
    if dt < 1e-10:
        return state1.copy()
    
    alpha = (t - state1.timestamp) / dt
    alpha = np.clip(alpha, 0.0, 1.0)
    
    # Linear interpolation helper
    def lerp(a, b, alpha):
        return a + alpha * (b - a)
    
    # SLERP for quaternions
    def slerp(q1, q2, alpha):
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        if dot > 0.9995:
            # Linear interpolation for nearly identical quaternions
            result = q1 + alpha * (q2 - q1)
            return result / np.linalg.norm(result)
        
        theta_0 = np.arccos(dot)
        theta = theta_0 * alpha
        
        q_perp = q2 - q1 * dot
        q_perp = q_perp / np.linalg.norm(q_perp)
        
        return q1 * np.cos(theta) + q_perp * np.sin(theta)
    
    return StateVector(
        timestamp=t,
        position=lerp(state1.position, state2.position, alpha),
        velocity=lerp(state1.velocity, state2.velocity, alpha),
        quaternion=slerp(state1.quaternion, state2.quaternion, alpha),
        angular_rates=lerp(state1.angular_rates, state2.angular_rates, alpha),
        accelerations=lerp(state1.accelerations, state2.accelerations, alpha),
        control_surfaces=lerp(state1.control_surfaces, state2.control_surfaces, alpha),
        thrust=lerp(state1.thrust, state2.thrust, alpha),
        fuel_mass=lerp(state1.fuel_mass, state2.fuel_mass, alpha),
        mach=lerp(state1.mach, state2.mach, alpha),
        dynamic_pressure=lerp(state1.dynamic_pressure, state2.dynamic_pressure, alpha),
        altitude=lerp(state1.altitude, state2.altitude, alpha),
        temperatures=(lerp(state1.temperatures, state2.temperatures, alpha)
                     if state1.temperatures is not None and state2.temperatures is not None else None),
        loads=(lerp(state1.loads, state2.loads, alpha)
               if state1.loads is not None and state2.loads is not None else None),
    )


def extrapolate_state(states: List[StateVector], t: float,
                     order: int = 2) -> StateVector:
    """
    Extrapolate state forward in time.
    
    Uses polynomial extrapolation based on recent state history.
    
    Args:
        states: Recent state history (newest last)
        t: Target time
        order: Polynomial order (1=linear, 2=quadratic)
        
    Returns:
        Extrapolated state at time t
    """
    if len(states) == 0:
        raise ValueError("Need at least one state for extrapolation")
    
    if len(states) == 1:
        # Just return copy with updated timestamp
        result = states[0].copy()
        result.timestamp = t
        return result
    
    # Use last few states for extrapolation
    n_use = min(len(states), order + 1)
    recent = states[-n_use:]
    
    times = np.array([s.timestamp for s in recent])
    dt = t - times[-1]
    
    if order == 1 or len(recent) < 3:
        # Linear extrapolation
        state_last = recent[-1]
        if len(recent) >= 2:
            state_prev = recent[-2]
            dt_prev = times[-1] - times[-2]
            
            if dt_prev > 1e-10:
                rate = (state_last.position - state_prev.position) / dt_prev
                pos = state_last.position + rate * dt
                
                vel_rate = (state_last.velocity - state_prev.velocity) / dt_prev
                vel = state_last.velocity + vel_rate * dt
            else:
                pos = state_last.position
                vel = state_last.velocity
        else:
            pos = state_last.position
            vel = state_last.velocity
        
        # For quaternion, integrate angular velocity
        omega = state_last.angular_rates
        q = state_last.quaternion.copy()
        
        # Small angle quaternion update
        dq = np.array([1.0, 0.5*omega[0]*dt, 0.5*omega[1]*dt, 0.5*omega[2]*dt])
        q_new = np.array([
            q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2] - q[3]*dq[3],
            q[0]*dq[1] + q[1]*dq[0] + q[2]*dq[3] - q[3]*dq[2],
            q[0]*dq[2] - q[1]*dq[3] + q[2]*dq[0] + q[3]*dq[1],
            q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1] + q[3]*dq[0],
        ])
        q_new = q_new / np.linalg.norm(q_new)
        
        return StateVector(
            timestamp=t,
            position=pos,
            velocity=vel,
            quaternion=q_new,
            angular_rates=state_last.angular_rates.copy(),
            accelerations=state_last.accelerations.copy(),
            control_surfaces=state_last.control_surfaces.copy(),
            thrust=state_last.thrust,
            fuel_mass=state_last.fuel_mass,
            mach=state_last.mach,
            dynamic_pressure=state_last.dynamic_pressure,
            altitude=state_last.altitude,
        )
    
    else:
        # Quadratic extrapolation
        # Fit parabola to positions
        positions = np.array([s.position for s in recent])
        velocities = np.array([s.velocity for s in recent])
        
        # Polynomial coefficients for each dimension
        pos_extrap = np.zeros(3)
        vel_extrap = np.zeros(3)
        
        for i in range(3):
            coeffs_pos = np.polyfit(times, positions[:, i], 2)
            coeffs_vel = np.polyfit(times, velocities[:, i], 2)
            pos_extrap[i] = np.polyval(coeffs_pos, t)
            vel_extrap[i] = np.polyval(coeffs_vel, t)
        
        # Use linear for other quantities
        state_last = recent[-1]
        
        return StateVector(
            timestamp=t,
            position=pos_extrap,
            velocity=vel_extrap,
            quaternion=state_last.quaternion.copy(),
            angular_rates=state_last.angular_rates.copy(),
            accelerations=state_last.accelerations.copy(),
            control_surfaces=state_last.control_surfaces.copy(),
            thrust=state_last.thrust,
            fuel_mass=state_last.fuel_mass,
            mach=state_last.mach,
            dynamic_pressure=state_last.dynamic_pressure,
            altitude=state_last.altitude,
        )


def compute_state_divergence(physical: StateVector,
                            digital: StateVector) -> Dict[str, float]:
    """
    Compute divergence metrics between physical and digital states.
    
    Returns:
        Dictionary of divergence metrics for each state component
    """
    divergence = {}
    
    # Position divergence (m)
    divergence['position'] = np.linalg.norm(physical.position - digital.position)
    
    # Velocity divergence (m/s)
    divergence['velocity'] = np.linalg.norm(physical.velocity - digital.velocity)
    
    # Attitude divergence (quaternion angle, rad)
    q1 = physical.quaternion / np.linalg.norm(physical.quaternion)
    q2 = digital.quaternion / np.linalg.norm(digital.quaternion)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    divergence['attitude'] = 2.0 * np.arccos(dot)
    
    # Angular rate divergence (rad/s)
    divergence['angular_rates'] = np.linalg.norm(
        physical.angular_rates - digital.angular_rates)
    
    # Mach divergence
    divergence['mach'] = np.abs(physical.mach - digital.mach)
    
    # Altitude divergence (m)
    divergence['altitude'] = np.abs(physical.altitude - digital.altitude)
    
    # Time synchronization error (s)
    divergence['time_offset'] = np.abs(physical.timestamp - digital.timestamp)
    
    # Overall divergence metric (weighted RMS)
    weights = {'position': 1/100, 'velocity': 1/10, 'attitude': 1.0,
               'angular_rates': 1.0, 'mach': 10.0, 'altitude': 1/1000}
    
    weighted_sum = sum(weights.get(k, 1.0) * v**2 for k, v in divergence.items()
                       if k in weights)
    divergence['total'] = np.sqrt(weighted_sum / len(weights))
    
    return divergence


class StateSync:
    """
    State synchronization manager for digital twin.
    
    Handles real-time state transfer, divergence detection,
    and automatic correction between physical and digital systems.
    """
    
    def __init__(self, config: SyncConfig):
        self.config = config
        self.mode = SyncMode.REAL_TIME
        self.status = SyncStatus.DISCONNECTED
        
        # State buffers
        self.physical_buffer = StateBuffer(config.buffer_size)
        self.digital_buffer = StateBuffer(config.buffer_size)
        
        # Synchronization state
        self.last_sync_time = 0.0
        self.divergence_start_time: Optional[float] = None
        self.total_synced_states = 0
        
        # Threading
        self.sync_thread: Optional[threading.Thread] = None
        self.running = False
        self.state_queue: queue.Queue = queue.Queue()
        
        # Callbacks
        self.on_state_received: Optional[Callable[[StateVector], None]] = None
        self.on_divergence: Optional[Callable[[Dict[str, float]], None]] = None
        self.on_recovery: Optional[Callable[[], None]] = None
        
        # Statistics
        self.latency_history: deque = deque(maxlen=1000)
        self.divergence_history: deque = deque(maxlen=1000)
    
    def start(self, mode: SyncMode = SyncMode.REAL_TIME):
        """Start synchronization process."""
        self.mode = mode
        self.running = True
        self.status = SyncStatus.CONNECTING
        
        if mode == SyncMode.REAL_TIME:
            self.sync_thread = threading.Thread(target=self._sync_loop)
            self.sync_thread.daemon = True
            self.sync_thread.start()
        
        self.status = SyncStatus.SYNCED
    
    def stop(self):
        """Stop synchronization."""
        self.running = False
        if self.sync_thread is not None:
            self.sync_thread.join(timeout=2.0)
        self.status = SyncStatus.DISCONNECTED
    
    def push_physical_state(self, state: StateVector):
        """
        Push new physical state (from vehicle/HIL).
        
        Args:
            state: Current physical vehicle state
        """
        self.physical_buffer.push(state)
        self.state_queue.put(('physical', state))
        
        # Update latency
        receive_time = time.time()
        self.latency_history.append(receive_time - state.timestamp)
    
    def push_digital_state(self, state: StateVector):
        """
        Push new digital state (from simulation).
        
        Args:
            state: Current digital twin state
        """
        self.digital_buffer.push(state)
        self.state_queue.put(('digital', state))
    
    def get_synchronized_state(self, t: Optional[float] = None) -> Optional[StateVector]:
        """
        Get best estimate of vehicle state at time t.
        
        Combines physical and digital states with appropriate weighting.
        
        Args:
            t: Target time (None = latest)
            
        Returns:
            Best state estimate
        """
        if t is None:
            physical_states = self.physical_buffer.get_latest(1)
            digital_states = self.digital_buffer.get_latest(1)
        else:
            physical_state = self.physical_buffer.get_at_time(t)
            digital_state = self.digital_buffer.get_at_time(t)
            physical_states = [physical_state] if physical_state else []
            digital_states = [digital_state] if digital_state else []
        
        if len(physical_states) == 0 and len(digital_states) == 0:
            return None
        
        if len(physical_states) == 0:
            return digital_states[0]
        
        if len(digital_states) == 0:
            return physical_states[0]
        
        # Blend physical and digital based on divergence
        physical = physical_states[0]
        digital = digital_states[0]
        
        divergence = compute_state_divergence(physical, digital)
        
        # If divergence is low, trust physical
        if divergence['total'] < 0.1:
            return physical
        
        # If divergence is high, blend towards digital (which has physics)
        blend_factor = min(1.0, divergence['total'])
        
        return self._blend_states(physical, digital, blend_factor)
    
    def _blend_states(self, physical: StateVector, digital: StateVector,
                     alpha: float) -> StateVector:
        """Blend two states with factor alpha (0=physical, 1=digital)."""
        return interpolate_state(physical, digital, 
                                physical.timestamp + alpha * 
                                (digital.timestamp - physical.timestamp))
    
    def _sync_loop(self):
        """Main synchronization loop (runs in thread)."""
        dt = 1.0 / self.config.sync_rate
        
        while self.running:
            try:
                # Process incoming states
                try:
                    source, state = self.state_queue.get(timeout=dt)
                    self._process_state(source, state)
                except queue.Empty:
                    pass
                
                # Check for divergence
                self._check_divergence()
                
                # Recovery if needed
                if self.status == SyncStatus.DIVERGED:
                    self._attempt_recovery()
                
            except Exception as e:
                self.status = SyncStatus.ERROR
                print(f"Sync error: {e}")
    
    def _process_state(self, source: str, state: StateVector):
        """Process received state."""
        self.total_synced_states += 1
        self.last_sync_time = state.timestamp
        
        if self.on_state_received:
            self.on_state_received(state)
    
    def _check_divergence(self):
        """Check for state divergence."""
        physical_states = self.physical_buffer.get_latest(1)
        digital_states = self.digital_buffer.get_latest(1)
        
        if len(physical_states) == 0 or len(digital_states) == 0:
            return
        
        divergence = compute_state_divergence(physical_states[0], digital_states[0])
        self.divergence_history.append(divergence)
        
        # Check thresholds
        exceeded = (
            divergence['position'] > self.config.position_threshold or
            divergence['velocity'] > self.config.velocity_threshold or
            divergence['attitude'] > self.config.attitude_threshold
        )
        
        if exceeded:
            if self.divergence_start_time is None:
                self.divergence_start_time = time.time()
            
            if time.time() - self.divergence_start_time > self.config.max_divergence_time:
                self.status = SyncStatus.DIVERGED
                if self.on_divergence:
                    self.on_divergence(divergence)
        else:
            self.divergence_start_time = None
            if self.status == SyncStatus.DIVERGED:
                self.status = SyncStatus.SYNCED
    
    def _attempt_recovery(self):
        """Attempt to recover from divergence."""
        self.status = SyncStatus.RECOVERING
        
        # Get latest physical state
        physical_states = self.physical_buffer.get_latest(1)
        if len(physical_states) == 0:
            return
        
        # Apply correction to digital state
        physical = physical_states[0]
        digital_states = self.digital_buffer.get_latest(1)
        
        if len(digital_states) > 0:
            digital = digital_states[0]
            
            # Gradual correction
            gain = self.config.recovery_gain
            corrected = StateVector(
                timestamp=physical.timestamp,
                position=digital.position + gain * (physical.position - digital.position),
                velocity=digital.velocity + gain * (physical.velocity - digital.velocity),
                quaternion=physical.quaternion.copy(),  # Trust physical attitude
                angular_rates=physical.angular_rates.copy(),
                accelerations=physical.accelerations.copy(),
                control_surfaces=physical.control_surfaces.copy(),
                thrust=physical.thrust,
                fuel_mass=physical.fuel_mass,
                mach=physical.mach,
                dynamic_pressure=physical.dynamic_pressure,
                altitude=physical.altitude,
            )
            
            self.digital_buffer.push(corrected)
        
        self.status = SyncStatus.SYNCED
        if self.on_recovery:
            self.on_recovery()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        latencies = list(self.latency_history)
        divergences = [d['total'] for d in self.divergence_history]
        
        return {
            'status': self.status.name,
            'mode': self.mode.name,
            'total_states': self.total_synced_states,
            'physical_buffer_size': len(self.physical_buffer),
            'digital_buffer_size': len(self.digital_buffer),
            'mean_latency': np.mean(latencies) if latencies else 0.0,
            'max_latency': np.max(latencies) if latencies else 0.0,
            'mean_divergence': np.mean(divergences) if divergences else 0.0,
            'max_divergence': np.max(divergences) if divergences else 0.0,
        }


class StateSynchronizer:
    """
    High-level state synchronizer with automatic mode selection.
    
    Provides simplified interface for common synchronization patterns.
    """
    
    def __init__(self, sync_rate: float = 100.0):
        config = SyncConfig(sync_rate=sync_rate)
        self.sync = StateSync(config)
        
    def connect(self, mode: SyncMode = SyncMode.REAL_TIME):
        """Connect and start synchronization."""
        self.sync.start(mode)
    
    def disconnect(self):
        """Disconnect and stop synchronization."""
        self.sync.stop()
    
    def update_physical(self, state: StateVector):
        """Update with physical state."""
        self.sync.push_physical_state(state)
    
    def update_digital(self, state: StateVector):
        """Update with digital state."""
        self.sync.push_digital_state(state)
    
    def get_state(self, t: Optional[float] = None) -> Optional[StateVector]:
        """Get synchronized state."""
        return self.sync.get_synchronized_state(t)
    
    @property
    def is_synced(self) -> bool:
        """Check if currently synchronized."""
        return self.sync.status == SyncStatus.SYNCED


def test_state_sync():
    """Test state synchronization module."""
    print("Testing State Synchronization Module...")
    
    # Create test states
    state1 = StateVector(
        timestamp=0.0,
        position=np.array([0.0, 0.0, 10000.0]),
        velocity=np.array([1000.0, 0.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rates=np.array([0.0, 0.0, 0.0]),
        accelerations=np.array([0.0, 0.0, -9.81]),
        control_surfaces=np.array([0.0, 0.0, 0.0]),
        mach=3.0,
        altitude=10000.0,
    )
    
    state2 = StateVector(
        timestamp=0.1,
        position=np.array([100.0, 0.0, 9990.0]),
        velocity=np.array([1000.0, 0.0, -10.0]),
        quaternion=np.array([0.9999, 0.01, 0.0, 0.0]),
        angular_rates=np.array([0.1, 0.0, 0.0]),
        accelerations=np.array([0.0, 0.0, -9.81]),
        control_surfaces=np.array([0.01, 0.0, 0.0]),
        mach=3.0,
        altitude=9990.0,
    )
    
    # Test interpolation
    interp = interpolate_state(state1, state2, 0.05)
    assert np.allclose(interp.position, [50.0, 0.0, 9995.0], rtol=0.01)
    print("  ✓ State interpolation")
    
    # Test extrapolation
    extrap = extrapolate_state([state1, state2], 0.2)
    assert extrap.timestamp == 0.2
    assert extrap.position[0] > state2.position[0]  # Moving forward
    print("  ✓ State extrapolation")
    
    # Test divergence computation
    state2_diverged = state2.copy()
    state2_diverged.position += np.array([50.0, 0.0, 0.0])
    div = compute_state_divergence(state2, state2_diverged)
    assert div['position'] == 50.0
    print("  ✓ Divergence computation")
    
    # Test state buffer
    buffer = StateBuffer(100)
    buffer.push(state1)
    buffer.push(state2)
    assert len(buffer) == 2
    latest = buffer.get_latest(1)
    assert len(latest) == 1
    assert latest[0].timestamp == 0.1
    print("  ✓ State buffer")
    
    # Test synchronizer
    sync = StateSynchronizer(sync_rate=100.0)
    sync.connect(SyncMode.BATCH)  # Use batch mode for testing
    sync.update_physical(state1)
    sync.update_digital(state1)
    result = sync.get_state()
    assert result is not None
    sync.disconnect()
    print("  ✓ State synchronizer")
    
    # Test to_vector and from_vector
    vec = state1.to_vector()
    reconstructed = StateVector.from_vector(vec, state1.timestamp)
    assert np.allclose(reconstructed.position, state1.position)
    assert np.allclose(reconstructed.velocity, state1.velocity)
    print("  ✓ State serialization")
    
    print("State Synchronization: All tests passed!")


if __name__ == "__main__":
    test_state_sync()
