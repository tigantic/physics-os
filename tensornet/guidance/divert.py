"""
Divert Guidance System
======================

Phase 22: Kill vehicle divert and attitude control.

Implements guidance algorithms for exoatmospheric kill vehicles (EKV)
with divert thrusters for terminal homing.

Key Features:
- Proportional navigation guidance variants
- Thruster selection and pulse modulation
- State estimation and prediction
- Miss distance minimization

References:
    - Zarchan, "Tactical and Strategic Missile Guidance" (2012)
    - Shneydor, "Missile Guidance and Pursuit" (1998)
    - Gutman & Leitner, "Optimal Control for Kill Vehicle" AIAA J Guidance (1998)

Constitution Compliance: Article II.1 (Guidance Systems)
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, NamedTuple, Callable
from dataclasses import dataclass, field
import math


# =============================================================================
# Constants
# =============================================================================

GRAVITY = 9.81  # m/s² (for reference, typically negligible in exo)
SPEED_OF_LIGHT = 299792458.0  # m/s


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ThrusterConfig:
    """
    Divert thruster configuration.
    
    Attributes:
        thrust_level: Thrust magnitude (N)
        position: (x, y, z) position on vehicle (m)
        direction: (nx, ny, nz) thrust direction (unit vector)
        min_pulse_width: Minimum on-time (s)
        max_pulse_width: Maximum on-time (s)
        response_time: Valve response time (s)
        specific_impulse: Isp (s)
    """
    thrust_level: float
    position: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    min_pulse_width: float = 0.010
    max_pulse_width: float = 1.0
    response_time: float = 0.005
    specific_impulse: float = 220.0


@dataclass
class VehicleState:
    """
    Kill vehicle state vector.
    
    Attributes:
        position: (x, y, z) position in inertial frame (m)
        velocity: (vx, vy, vz) velocity (m/s)
        mass: Current mass (kg)
        attitude: Quaternion [q0, q1, q2, q3] or Euler angles
        angular_rate: (p, q, r) body rates (rad/s)
    """
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    mass: float
    attitude: Optional[Tuple[float, float, float, float]] = None
    angular_rate: Optional[Tuple[float, float, float]] = None


@dataclass
class TargetState:
    """
    Target state estimate.
    
    Attributes:
        position: (x, y, z) position (m)
        velocity: (vx, vy, vz) velocity (m/s)
        acceleration: (ax, ay, az) estimated acceleration (m/s²)
        covariance: State error covariance (6x6 or 9x9)
    """
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    covariance: Optional[Tensor] = None


@dataclass
class GuidanceCommand:
    """
    Guidance output command.
    
    Attributes:
        acceleration_command: Commanded acceleration (m/s²)
        thruster_commands: Dict of thruster_id -> pulse_width
        predicted_miss: Predicted miss distance (m)
        time_to_go: Estimated time to intercept (s)
    """
    acceleration_command: Tuple[float, float, float]
    thruster_commands: Dict[int, float]
    predicted_miss: float
    time_to_go: float


# =============================================================================
# Geometry Utilities
# =============================================================================

def closing_velocity(
    kv_velocity: Tuple[float, float, float],
    target_velocity: Tuple[float, float, float]
) -> float:
    """Compute closing velocity (approach speed)."""
    dvx = kv_velocity[0] - target_velocity[0]
    dvy = kv_velocity[1] - target_velocity[1]
    dvz = kv_velocity[2] - target_velocity[2]
    return math.sqrt(dvx**2 + dvy**2 + dvz**2)


def line_of_sight_rate(
    kv_state: VehicleState,
    target_state: TargetState
) -> Tuple[float, float, float]:
    """
    Compute line-of-sight rate vector.
    
    LOS rate = (R × V_rel) / |R|²
    
    Args:
        kv_state: Kill vehicle state
        target_state: Target state
        
    Returns:
        (λ_dot_x, λ_dot_y, λ_dot_z) LOS rate (rad/s)
    """
    # Relative position (target - KV)
    rx = target_state.position[0] - kv_state.position[0]
    ry = target_state.position[1] - kv_state.position[1]
    rz = target_state.position[2] - kv_state.position[2]
    
    # Relative velocity
    vx = target_state.velocity[0] - kv_state.velocity[0]
    vy = target_state.velocity[1] - kv_state.velocity[1]
    vz = target_state.velocity[2] - kv_state.velocity[2]
    
    # Range
    r = math.sqrt(rx**2 + ry**2 + rz**2)
    if r < 1.0:
        return (0.0, 0.0, 0.0)
    
    # R × V
    cross_x = ry * vz - rz * vy
    cross_y = rz * vx - rx * vz
    cross_z = rx * vy - ry * vx
    
    # LOS rate
    r_sq = r**2
    lambda_dot_x = cross_x / r_sq
    lambda_dot_y = cross_y / r_sq
    lambda_dot_z = cross_z / r_sq
    
    return (lambda_dot_x, lambda_dot_y, lambda_dot_z)


def time_to_go(
    kv_state: VehicleState,
    target_state: TargetState
) -> float:
    """
    Estimate time to intercept.
    
    Uses simple range / closing velocity approximation.
    """
    rx = target_state.position[0] - kv_state.position[0]
    ry = target_state.position[1] - kv_state.position[1]
    rz = target_state.position[2] - kv_state.position[2]
    
    vx = target_state.velocity[0] - kv_state.velocity[0]
    vy = target_state.velocity[1] - kv_state.velocity[1]
    vz = target_state.velocity[2] - kv_state.velocity[2]
    
    r = math.sqrt(rx**2 + ry**2 + rz**2)
    v_close = math.sqrt(vx**2 + vy**2 + vz**2)
    
    if v_close < 1.0:
        return float('inf')
    
    # Use dot product for closing rate
    r_dot = -(rx * vx + ry * vy + rz * vz) / r
    
    if r_dot <= 0:
        return float('inf')  # Opening, not closing
    
    return r / r_dot


def zero_effort_miss(
    kv_state: VehicleState,
    target_state: TargetState,
    tgo: float
) -> Tuple[float, float, float]:
    """
    Compute zero-effort miss (ZEM) vector.
    
    Miss distance if no further maneuvers are made.
    
    Args:
        kv_state: Kill vehicle state
        target_state: Target state
        tgo: Time to go (s)
        
    Returns:
        (zem_x, zem_y, zem_z) miss vector at closest approach
    """
    # Relative position at current time
    rx = target_state.position[0] - kv_state.position[0]
    ry = target_state.position[1] - kv_state.position[1]
    rz = target_state.position[2] - kv_state.position[2]
    
    # Relative velocity
    vx = target_state.velocity[0] - kv_state.velocity[0]
    vy = target_state.velocity[1] - kv_state.velocity[1]
    vz = target_state.velocity[2] - kv_state.velocity[2]
    
    # Relative acceleration (target only, assuming ballistic KV)
    ax = target_state.acceleration[0]
    ay = target_state.acceleration[1]
    az = target_state.acceleration[2]
    
    # Predicted relative position at tgo
    zem_x = rx + vx * tgo + 0.5 * ax * tgo**2
    zem_y = ry + vy * tgo + 0.5 * ay * tgo**2
    zem_z = rz + vz * tgo + 0.5 * az * tgo**2
    
    return (zem_x, zem_y, zem_z)


# =============================================================================
# Guidance Laws
# =============================================================================

def proportional_navigation(
    kv_state: VehicleState,
    target_state: TargetState,
    N: float = 3.0
) -> Tuple[float, float, float]:
    """
    Proportional Navigation guidance law.
    
    a_cmd = N * V_c * λ_dot
    
    Args:
        kv_state: Kill vehicle state
        target_state: Target state
        N: Navigation constant (typically 3-5)
        
    Returns:
        (ax, ay, az) commanded acceleration (m/s²)
    """
    # Closing velocity
    Vc = closing_velocity(kv_state.velocity, target_state.velocity)
    
    # LOS rate
    lambda_dot = line_of_sight_rate(kv_state, target_state)
    
    # PN command
    ax = N * Vc * lambda_dot[0]
    ay = N * Vc * lambda_dot[1]
    az = N * Vc * lambda_dot[2]
    
    return (ax, ay, az)


def augmented_proportional_navigation(
    kv_state: VehicleState,
    target_state: TargetState,
    N: float = 3.0
) -> Tuple[float, float, float]:
    """
    Augmented Proportional Navigation (APN).
    
    Accounts for target acceleration:
    a_cmd = N * V_c * λ_dot + 0.5 * N * a_target
    
    Args:
        kv_state: Kill vehicle state
        target_state: Target state
        N: Navigation constant
        
    Returns:
        Commanded acceleration (m/s²)
    """
    # Base PN
    ax_pn, ay_pn, az_pn = proportional_navigation(kv_state, target_state, N)
    
    # Target acceleration compensation
    at = target_state.acceleration
    ax = ax_pn + 0.5 * N * at[0]
    ay = ay_pn + 0.5 * N * at[1]
    az = az_pn + 0.5 * N * at[2]
    
    return (ax, ay, az)


def optimal_guidance_law(
    kv_state: VehicleState,
    target_state: TargetState,
    tgo_estimate: Optional[float] = None
) -> Tuple[float, float, float]:
    """
    Optimal (ZEM-based) guidance law.
    
    a_cmd = 3 * ZEM / t_go²
    
    Minimizes fuel consumption for zero miss distance.
    
    Args:
        kv_state: Kill vehicle state
        target_state: Target state
        tgo_estimate: Time to go (computed if None)
        
    Returns:
        Commanded acceleration (m/s²)
    """
    tgo = tgo_estimate or time_to_go(kv_state, target_state)
    
    if tgo < 0.1:
        return (0.0, 0.0, 0.0)  # Too close to intercept
    
    # Zero-effort miss
    zem = zero_effort_miss(kv_state, target_state, tgo)
    
    # Optimal acceleration
    scale = 3.0 / (tgo ** 2)
    ax = scale * zem[0]
    ay = scale * zem[1]
    az = scale * zem[2]
    
    return (ax, ay, az)


# =============================================================================
# Thruster Allocation
# =============================================================================

class DivertThruster:
    """
    Single divert thruster model.
    
    Handles pulse modulation and response characteristics.
    """
    
    def __init__(self, config: ThrusterConfig, id: int = 0):
        self.config = config
        self.id = id
        self._accumulated_impulse = 0.0
        self._last_command_time = 0.0
        self._on = False
    
    @property
    def thrust_direction(self) -> Tuple[float, float, float]:
        """Get thrust direction (normalized)."""
        d = self.config.direction
        mag = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
        return (d[0]/mag, d[1]/mag, d[2]/mag)
    
    def compute_pulse_width(
        self,
        acceleration_component: float,
        vehicle_mass: float,
        dt: float
    ) -> float:
        """
        Compute required pulse width for desired acceleration.
        
        Args:
            acceleration_component: Desired acceleration in thrust direction (m/s²)
            vehicle_mass: Current vehicle mass (kg)
            dt: Control cycle time (s)
            
        Returns:
            Pulse width (s), 0 if no thrust needed
        """
        # Required force
        F_required = abs(acceleration_component) * vehicle_mass
        
        if F_required < 1.0:  # Dead band
            return 0.0
        
        # Pulse width for duty cycle
        duty = F_required / self.config.thrust_level
        pulse = duty * dt
        
        # Enforce limits
        if pulse < self.config.min_pulse_width:
            return 0.0  # Below minimum, don't fire
        
        pulse = min(pulse, self.config.max_pulse_width)
        pulse = min(pulse, dt)  # Can't exceed cycle time
        
        return pulse
    
    def compute_delta_v(self, pulse_width: float, vehicle_mass: float) -> float:
        """Compute delta-V from pulse."""
        impulse = self.config.thrust_level * pulse_width
        return impulse / vehicle_mass


class DivertGuidance:
    """
    Complete divert guidance system.
    
    Integrates guidance law, thruster allocation, and state estimation.
    
    Attributes:
        thrusters: List of divert thrusters
        guidance_law: 'pn', 'apn', or 'optimal'
        nav_constant: Navigation constant N
        max_acceleration: Maximum commanded acceleration (m/s²)
    """
    
    def __init__(
        self,
        thrusters: List[DivertThruster],
        guidance_law: str = 'optimal',
        nav_constant: float = 3.5,
        max_acceleration: float = 100.0
    ):
        self.thrusters = thrusters
        self.guidance_law = guidance_law
        self.nav_constant = nav_constant
        self.max_acceleration = max_acceleration
        
        # Build thruster matrix
        self._build_thruster_matrix()
    
    def _build_thruster_matrix(self):
        """Build thruster direction matrix for allocation."""
        n = len(self.thrusters)
        self.thruster_matrix = torch.zeros(3, n, dtype=torch.float64)
        
        for i, t in enumerate(self.thrusters):
            d = t.thrust_direction
            self.thruster_matrix[0, i] = d[0]
            self.thruster_matrix[1, i] = d[1]
            self.thruster_matrix[2, i] = d[2]
    
    def compute_guidance(
        self,
        kv_state: VehicleState,
        target_state: TargetState,
        dt: float
    ) -> GuidanceCommand:
        """
        Compute complete guidance command.
        
        Args:
            kv_state: Kill vehicle state
            target_state: Target state estimate
            dt: Control cycle time (s)
            
        Returns:
            GuidanceCommand with thruster commands
        """
        # Compute time to go
        tgo = time_to_go(kv_state, target_state)
        
        # Compute acceleration command
        if self.guidance_law == 'pn':
            a_cmd = proportional_navigation(kv_state, target_state, self.nav_constant)
        elif self.guidance_law == 'apn':
            a_cmd = augmented_proportional_navigation(kv_state, target_state, self.nav_constant)
        else:  # optimal
            a_cmd = optimal_guidance_law(kv_state, target_state, tgo)
        
        # Limit acceleration
        a_mag = math.sqrt(a_cmd[0]**2 + a_cmd[1]**2 + a_cmd[2]**2)
        if a_mag > self.max_acceleration:
            scale = self.max_acceleration / a_mag
            a_cmd = (a_cmd[0] * scale, a_cmd[1] * scale, a_cmd[2] * scale)
        
        # Thruster allocation
        thruster_commands = self._allocate_thrusters(a_cmd, kv_state.mass, dt)
        
        # Predicted miss
        zem = zero_effort_miss(kv_state, target_state, tgo)
        miss = math.sqrt(zem[0]**2 + zem[1]**2 + zem[2]**2)
        
        return GuidanceCommand(
            acceleration_command=a_cmd,
            thruster_commands=thruster_commands,
            predicted_miss=miss,
            time_to_go=tgo
        )
    
    def _allocate_thrusters(
        self,
        a_cmd: Tuple[float, float, float],
        mass: float,
        dt: float
    ) -> Dict[int, float]:
        """
        Allocate acceleration command to thrusters.
        
        Uses pseudo-inverse for optimal allocation.
        
        Args:
            a_cmd: Commanded acceleration (m/s²)
            mass: Vehicle mass (kg)
            dt: Control cycle time (s)
            
        Returns:
            Dict mapping thruster ID to pulse width
        """
        # Required force
        F_cmd = torch.tensor([a_cmd[0] * mass, a_cmd[1] * mass, a_cmd[2] * mass],
                            dtype=torch.float64)
        
        # Thruster force magnitudes via pseudo-inverse
        # F = B * u where B is thruster matrix, u is force magnitudes
        B_pinv = torch.linalg.pinv(self.thruster_matrix)
        u = B_pinv @ F_cmd
        
        # Convert to pulse widths
        commands = {}
        for i, thruster in enumerate(self.thrusters):
            if u[i] > 0:  # Only positive thrust (push, not pull)
                a_component = u[i].item() / mass
                pulse = thruster.compute_pulse_width(a_component, mass, dt)
                if pulse > 0:
                    commands[thruster.id] = pulse
        
        return commands
    
    def predict_intercept(
        self,
        kv_state: VehicleState,
        target_state: TargetState,
        dt: float = 0.1,
        max_steps: int = 1000
    ) -> Tuple[float, float, float]:
        """
        Predict intercept by simulating forward.
        
        Returns:
            (time_to_intercept, miss_distance, delta_v_used)
        """
        # Simple propagation
        kv_pos = list(kv_state.position)
        kv_vel = list(kv_state.velocity)
        tgt_pos = list(target_state.position)
        tgt_vel = list(target_state.velocity)
        
        total_dv = 0.0
        t = 0.0
        
        for _ in range(max_steps):
            # Range
            r = math.sqrt(sum((kv_pos[i] - tgt_pos[i])**2 for i in range(3)))
            
            if r < 1.0:  # Hit
                return t, r, total_dv
            
            # Update positions
            for i in range(3):
                kv_pos[i] += kv_vel[i] * dt
                tgt_pos[i] += tgt_vel[i] * dt
            
            t += dt
            
            # Check if opening
            r_new = math.sqrt(sum((kv_pos[i] - tgt_pos[i])**2 for i in range(3)))
            if r_new > r * 1.01:  # Opening
                return t, r_new, total_dv
        
        # Max time reached
        r_final = math.sqrt(sum((kv_pos[i] - tgt_pos[i])**2 for i in range(3)))
        return t, r_final, total_dv


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data structures
    'ThrusterConfig',
    'VehicleState',
    'TargetState',
    'GuidanceCommand',
    # Geometry
    'closing_velocity',
    'line_of_sight_rate',
    'time_to_go',
    'zero_effort_miss',
    # Guidance laws
    'proportional_navigation',
    'augmented_proportional_navigation',
    'optimal_guidance_law',
    # Classes
    'DivertThruster',
    'DivertGuidance',
]
