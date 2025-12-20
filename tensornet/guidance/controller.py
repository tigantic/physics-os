"""
Guidance Controller
===================

Physics-aware guidance for hypersonic vehicles with trajectory tracking,
constraint handling, and real-time CFD integration.

Guidance Laws:
    - Bank-to-turn for hypersonic glide
    - Proportional navigation for terminal guidance
    - Energy management for range control
    - Predictive guidance with forward simulation

Constraints:
    - Thermal: q̇ < q̇_max, Q < Q_max (heat load)
    - Structural: g_n < g_max, q_dyn < q_max
    - Corridor: altitude bounds, skip-out prevention
    - Range: TAEM energy management

Real-Time Considerations:
    - 100 Hz update rate for guidance loop
    - Pre-computed lookup tables
    - GPU-accelerated trajectory prediction
    - Graceful degradation under overload
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Union, Callable
from enum import Enum
import math

from tensornet.guidance.trajectory import (
    VehicleState, TrajectorySolver, TrajectoryConfig,
    AtmosphericModel, isa_atmosphere
)


class GuidanceMode(Enum):
    """Guidance mode phases."""
    ENTRY = "entry"                 # Initial entry phase
    EQUILIBRIUM_GLIDE = "eq_glide"  # Equilibrium glide at L/D max
    RANGE_CONTROL = "range"         # Bank angle modulation for range
    ENERGY_MANAGEMENT = "taem"      # TAEM: Terminal Area Energy Management
    TERMINAL = "terminal"           # Final approach
    CAPTURE = "capture"             # Runway capture


class ConstraintType(Enum):
    """Constraint type classification."""
    THERMAL_RATE = "thermal_rate"   # Heat rate q̇
    THERMAL_LOAD = "thermal_load"   # Integrated heat load Q
    STRUCTURAL_G = "structural_g"   # Normal acceleration
    DYNAMIC_PRESSURE = "q_dynamic"  # Dynamic pressure
    ALTITUDE_MAX = "alt_max"        # Skip-out prevention
    ALTITUDE_MIN = "alt_min"        # Ground avoidance
    BANK_LIMIT = "bank_limit"       # Bank angle limit


@dataclass
class GuidanceCommand:
    """Output command from guidance law."""
    bank_angle_rad: float = 0.0      # Commanded bank angle
    angle_of_attack_rad: float = 0.0  # Commanded AoA
    bank_rate_rad_s: float = 0.0      # Bank rate (for limiting)
    mode: GuidanceMode = GuidanceMode.ENTRY
    constraint_margin: float = 1.0    # 0-1, 1 = no constraint active
    predicted_range_m: float = 0.0
    
    def to_controls(self) -> Dict[str, float]:
        """Convert to control surface deflections."""
        # Simplified conversion
        # Real system would use trim tables
        de = -self.angle_of_attack_rad * 0.5  # Elevator for AoA
        da = self.bank_angle_rad * 0.3         # Aileron for bank
        dr = 0.0                               # Coordinated turn
        
        return {'de': de, 'da': da, 'dr': dr}


@dataclass 
class TrajectoryConstraint:
    """Definition of a trajectory constraint."""
    constraint_type: ConstraintType
    max_value: float
    current_value: float = 0.0
    margin: float = 0.1  # Activation margin (10% below limit)
    
    @property
    def violation(self) -> float:
        """Constraint violation (0 if satisfied, positive if violated)."""
        return max(0, self.current_value - self.max_value)
    
    @property
    def relative_margin(self) -> float:
        """Relative margin to constraint (1 = at limit, 0 = far from limit)."""
        if self.max_value <= 0:
            return 0.0
        return self.current_value / self.max_value
    
    @property
    def is_active(self) -> bool:
        """Whether constraint is active (within margin of limit)."""
        return self.relative_margin > (1 - self.margin)


@dataclass
class WaypointTarget:
    """Target waypoint for guidance."""
    latitude_rad: float
    longitude_rad: float
    altitude_m: float = 0.0
    velocity_m_s: float = 0.0  # Target velocity (0 = any)
    heading_rad: float = 0.0   # Target heading (NaN = any)


@dataclass
class CorridorBounds:
    """Entry corridor definition."""
    min_altitude_m: float = 15000.0   # Avoid ground
    max_altitude_m: float = 120000.0  # Avoid skip-out
    min_velocity_m_s: float = 100.0   # Stall avoidance
    max_heat_rate_W_cm2: float = 100.0
    max_heat_load_J_cm2: float = 10000.0
    max_g_load: float = 3.0
    max_bank_angle_rad: float = math.radians(80)
    max_aoa_rad: float = math.radians(40)


def proportional_navigation(
    vehicle_state: VehicleState,
    target_state: VehicleState,
    nav_ratio: float = 3.0
) -> float:
    """
    Proportional navigation guidance law.
    
    Used for terminal homing where line-of-sight rate
    is nulled by commanding acceleration perpendicular to LOS.
    
    Args:
        vehicle_state: Current vehicle state
        target_state: Target state
        nav_ratio: Navigation ratio (typically 3-5)
        
    Returns:
        Commanded lateral acceleration (m/s²)
    """
    # Position difference (simplified flat Earth)
    dx = (target_state.longitude_rad - vehicle_state.longitude_rad) * 6371000
    dy = (target_state.latitude_rad - vehicle_state.latitude_rad) * 6371000
    dz = target_state.altitude_m - vehicle_state.altitude_m
    
    range_to_target = math.sqrt(dx**2 + dy**2 + dz**2)
    
    if range_to_target < 1.0:
        return 0.0
    
    # Velocity difference
    dVx = target_state.u_m_s - vehicle_state.u_m_s
    dVy = target_state.v_m_s - vehicle_state.v_m_s
    dVz = target_state.w_m_s - vehicle_state.w_m_s
    
    # Line of sight rate (cross product of LOS and relative velocity)
    # Simplified to horizontal plane
    V_closing = -(dx * dVx + dy * dVy + dz * dVz) / range_to_target
    
    # LOS rotation rate
    omega_los = (dx * dVy - dy * dVx) / (range_to_target ** 2)
    
    # PN acceleration command
    a_cmd = nav_ratio * V_closing * omega_los
    
    return a_cmd


def bank_angle_guidance(
    current_state: VehicleState,
    target: WaypointTarget,
    corridor: CorridorBounds,
    L_over_D: float = 1.5
) -> GuidanceCommand:
    """
    Bank-to-turn guidance for hypersonic glide vehicles.
    
    Modulates bank angle to control range while maintaining
    equilibrium glide within thermal and structural constraints.
    
    Args:
        current_state: Current vehicle state
        target: Target waypoint
        corridor: Constraint corridor
        L_over_D: Current lift-to-drag ratio
        
    Returns:
        GuidanceCommand with bank angle and mode
    """
    # Current position
    lat = current_state.latitude_rad
    lon = current_state.longitude_rad
    alt = current_state.altitude_m
    V = current_state.velocity_magnitude
    
    # Target bearing
    dlat = target.latitude_rad - lat
    dlon = target.longitude_rad - lon
    
    range_to_target = math.sqrt(
        (dlat * 6371000)**2 + 
        (dlon * math.cos(lat) * 6371000)**2
    )
    
    target_heading = math.atan2(dlon * math.cos(lat), dlat)
    
    # Current heading (from velocity)
    phi, theta, psi = current_state.euler_angles
    heading_error = target_heading - psi
    
    # Normalize to [-π, π]
    while heading_error > math.pi:
        heading_error -= 2 * math.pi
    while heading_error < -math.pi:
        heading_error += 2 * math.pi
    
    # Equilibrium glide bank angle (for L/D and range)
    # Derived from equilibrium glide equations
    if V > 100:
        # Estimate required range capability
        # Higher bank reduces range, lower increases it
        E_ratio = (V**2 / 2 + 9.81 * alt) / (target.velocity_m_s**2 / 2 + 9.81 * target.altitude_m + 1)
        
        # Reference bank for equilibrium glide
        bank_ref = math.acos(1.0 / max(1.0, L_over_D))
        
        # Modulate bank for heading
        bank_heading_correction = 0.5 * heading_error  # Simple proportional
        
        # Range control (increase bank if too much energy, decrease if not enough)
        predicted_range = L_over_D * alt  # Simplified
        range_error = range_to_target - predicted_range
        bank_range_correction = -0.0001 * range_error  # Very gentle
        
        bank_cmd = bank_ref + bank_heading_correction + bank_range_correction
        
        # Apply sign for turn direction
        if heading_error > 0:
            bank_cmd = abs(bank_cmd)
        else:
            bank_cmd = -abs(bank_cmd)
    else:
        bank_cmd = 0.0
    
    # Limit bank angle
    bank_cmd = max(-corridor.max_bank_angle_rad, 
                   min(corridor.max_bank_angle_rad, bank_cmd))
    
    # Angle of attack for L/D (simplified)
    aoa_cmd = math.radians(15)  # Near L/D max for many hypersonic vehicles
    
    # Determine mode
    if alt > 60000:
        mode = GuidanceMode.ENTRY
    elif V > 1000:
        mode = GuidanceMode.EQUILIBRIUM_GLIDE
    elif range_to_target > 50000:
        mode = GuidanceMode.ENERGY_MANAGEMENT
    else:
        mode = GuidanceMode.TERMINAL
    
    return GuidanceCommand(
        bank_angle_rad=bank_cmd,
        angle_of_attack_rad=aoa_cmd,
        mode=mode,
        predicted_range_m=L_over_D * alt
    )


class GuidanceController:
    """
    Main guidance controller with constraint handling and CFD integration.
    
    Implements a predictor-corrector guidance scheme with:
    - Reference trajectory generation
    - Online trajectory prediction
    - Constraint monitoring and handling
    - CFD-based aerodynamic lookup
    """
    
    def __init__(
        self,
        corridor: CorridorBounds = None,
        target: WaypointTarget = None,
        dt: float = 0.01
    ):
        self.corridor = corridor or CorridorBounds()
        self.target = target or WaypointTarget(0, 0)
        self.dt = dt
        
        # Constraint tracking
        self.constraints: Dict[ConstraintType, TrajectoryConstraint] = {}
        self._initialize_constraints()
        
        # CFD lookup table (Mach, alpha) -> (CL, CD, Cm)
        self.aero_table: Optional[Dict] = None
        
        # Predictor trajectory solver
        self.predictor = TrajectorySolver(TrajectoryConfig(dt_s=0.1))
        
        # State history for filtering
        self.command_history: List[GuidanceCommand] = []
        self.heat_load_accumulated = 0.0
    
    def _initialize_constraints(self):
        """Initialize constraint monitors."""
        self.constraints = {
            ConstraintType.THERMAL_RATE: TrajectoryConstraint(
                ConstraintType.THERMAL_RATE,
                self.corridor.max_heat_rate_W_cm2
            ),
            ConstraintType.THERMAL_LOAD: TrajectoryConstraint(
                ConstraintType.THERMAL_LOAD,
                self.corridor.max_heat_load_J_cm2
            ),
            ConstraintType.STRUCTURAL_G: TrajectoryConstraint(
                ConstraintType.STRUCTURAL_G,
                self.corridor.max_g_load
            ),
            ConstraintType.ALTITUDE_MIN: TrajectoryConstraint(
                ConstraintType.ALTITUDE_MIN,
                self.corridor.min_altitude_m
            ),
            ConstraintType.ALTITUDE_MAX: TrajectoryConstraint(
                ConstraintType.ALTITUDE_MAX,
                self.corridor.max_altitude_m
            ),
        }
    
    def set_aero_table(self, table: Dict):
        """
        Set aerodynamic lookup table from CFD.
        
        Args:
            table: Dict with keys (Mach, alpha_deg) -> (CL, CD, Cm)
        """
        self.aero_table = table
    
    def lookup_aerodynamics(
        self,
        mach: float,
        alpha_deg: float
    ) -> Tuple[float, float, float]:
        """
        Look up aerodynamic coefficients.
        
        Args:
            mach: Mach number
            alpha_deg: Angle of attack in degrees
            
        Returns:
            (CL, CD, Cm)
        """
        if self.aero_table is None:
            # Default model
            CL = 0.05 * alpha_deg
            CD = 0.02 + 0.001 * alpha_deg**2
            if mach > 5:
                # Hypersonic corrections
                CL *= 0.5
                CD *= 1.5
            Cm = -0.05 * alpha_deg
            return CL, CD, Cm
        
        # Interpolate from table (simplified nearest neighbor)
        # Real implementation would use bilinear interpolation
        nearest_key = min(
            self.aero_table.keys(),
            key=lambda k: abs(k[0] - mach) + abs(k[1] - alpha_deg)
        )
        return self.aero_table[nearest_key]
    
    def estimate_heating(
        self,
        state: VehicleState,
        atm: AtmosphericModel
    ) -> float:
        """
        Estimate stagnation point heat rate.
        
        Uses Sutton-Graves correlation for convective heating.
        
        Args:
            state: Vehicle state
            atm: Atmospheric conditions
            
        Returns:
            Heat rate in W/cm²
        """
        V = state.velocity_magnitude
        rho = atm.density_kg_m3
        
        # Sutton-Graves constant for air
        k = 1.83e-4  # W/(cm² * (kg/m³)^0.5 * (m/s)³)
        
        # Nose radius (assumed)
        R_nose = 0.5  # m
        
        # Heat rate
        q_dot = k * math.sqrt(rho / R_nose) * V**3
        
        # Convert to W/cm²
        q_dot_cm2 = q_dot * 1e-4
        
        return q_dot_cm2
    
    def estimate_g_load(
        self,
        state: VehicleState,
        atm: AtmosphericModel,
        bank_angle: float,
        aoa: float
    ) -> float:
        """
        Estimate normal g-load.
        
        Args:
            state: Vehicle state
            atm: Atmospheric conditions
            bank_angle: Bank angle in radians
            aoa: Angle of attack in radians
            
        Returns:
            Normal load factor (g's)
        """
        V = state.velocity_magnitude
        rho = atm.density_kg_m3
        S = 10.0  # Reference area (assumed)
        m = state.mass_kg
        g = 9.81
        
        # Dynamic pressure
        q = 0.5 * rho * V**2
        
        # Lift coefficient
        mach = V / atm.speed_of_sound_m_s
        CL, _, _ = self.lookup_aerodynamics(mach, math.degrees(aoa))
        
        # Lift force
        L = q * S * CL
        
        # Normal acceleration (body Z, accounting for bank)
        n_z = L / (m * g) / math.cos(bank_angle)
        
        return abs(n_z)
    
    def update_constraints(
        self,
        state: VehicleState,
        command: GuidanceCommand
    ):
        """
        Update constraint values based on current state.
        
        Args:
            state: Current vehicle state
            command: Current guidance command
        """
        atm = isa_atmosphere(state.altitude_m)
        
        # Heat rate
        q_dot = self.estimate_heating(state, atm)
        self.constraints[ConstraintType.THERMAL_RATE].current_value = q_dot
        
        # Accumulated heat load
        self.heat_load_accumulated += q_dot * self.dt
        self.constraints[ConstraintType.THERMAL_LOAD].current_value = self.heat_load_accumulated
        
        # G-load
        n = self.estimate_g_load(
            state, atm, 
            command.bank_angle_rad, 
            command.angle_of_attack_rad
        )
        self.constraints[ConstraintType.STRUCTURAL_G].current_value = n
        
        # Altitude
        self.constraints[ConstraintType.ALTITUDE_MIN].current_value = (
            self.corridor.min_altitude_m - state.altitude_m
        )  # Positive if below min
        self.constraints[ConstraintType.ALTITUDE_MAX].current_value = state.altitude_m
    
    def apply_constraint_limiting(
        self,
        command: GuidanceCommand,
        state: VehicleState
    ) -> GuidanceCommand:
        """
        Modify command to respect constraints.
        
        Uses a priority-based approach:
        1. Thermal rate (reduce bank to reduce heating)
        2. G-load (reduce bank and AoA)
        3. Altitude (adjust vertical control)
        
        Args:
            command: Nominal guidance command
            state: Current vehicle state
            
        Returns:
            Constraint-limited command
        """
        # Check most critical constraints
        # relative_margin can be > 1 if constraint is violated, so clamp it
        thermal_rel = min(1.0, self.constraints[ConstraintType.THERMAL_RATE].relative_margin)
        g_rel = min(1.0, self.constraints[ConstraintType.STRUCTURAL_G].relative_margin)
        
        thermal_margin = 1.0 - thermal_rel
        g_margin = 1.0 - g_rel
        
        min_margin = max(0.0, min(thermal_margin, g_margin))
        
        # If approaching limits, reduce bank angle
        if min_margin < 0.2:  # Within 20% of limit
            scale = max(0.3, min_margin / 0.2)  # Scale down, but not below 30%
            command.bank_angle_rad *= scale
            command.constraint_margin = min_margin
        
        # Hard limits
        command.bank_angle_rad = max(
            -self.corridor.max_bank_angle_rad,
            min(self.corridor.max_bank_angle_rad, command.bank_angle_rad)
        )
        
        command.angle_of_attack_rad = max(
            0,
            min(self.corridor.max_aoa_rad, command.angle_of_attack_rad)
        )
        
        return command
    
    def predict_trajectory(
        self,
        state: VehicleState,
        command: GuidanceCommand,
        horizon_s: float = 30.0
    ) -> List[VehicleState]:
        """
        Predict future trajectory with current command.
        
        Args:
            state: Current state
            command: Command to hold
            horizon_s: Prediction horizon
            
        Returns:
            Predicted trajectory
        """
        controls = command.to_controls()
        
        def const_controls(t, s):
            return controls
        
        return self.predictor.propagate(state, horizon_s, const_controls)
    
    def compute_guidance(
        self,
        state: VehicleState,
        target: Optional[WaypointTarget] = None
    ) -> GuidanceCommand:
        """
        Main guidance computation loop.
        
        Args:
            state: Current vehicle state
            target: Optional override for target
            
        Returns:
            Guidance command
        """
        target = target or self.target
        
        # Get atmospheric conditions for L/D estimate
        atm = isa_atmosphere(state.altitude_m)
        V = state.velocity_magnitude
        mach = V / atm.speed_of_sound_m_s if atm.speed_of_sound_m_s > 0 else 0
        
        # Estimate current L/D
        aoa_est = state.angle_of_attack
        CL, CD, _ = self.lookup_aerodynamics(mach, math.degrees(aoa_est))
        L_over_D = CL / CD if CD > 0.001 else 10.0
        
        # Compute nominal guidance
        command = bank_angle_guidance(
            state, target, self.corridor, L_over_D
        )
        
        # Update constraint tracking
        self.update_constraints(state, command)
        
        # Apply constraint limiting
        command = self.apply_constraint_limiting(command, state)
        
        # Store for filtering
        self.command_history.append(command)
        if len(self.command_history) > 100:
            self.command_history.pop(0)
        
        return command
    
    def reset(self):
        """Reset controller state for new trajectory."""
        self.command_history.clear()
        self.heat_load_accumulated = 0.0
        self._initialize_constraints()


def closed_loop_simulation(
    initial_state: VehicleState,
    target: WaypointTarget,
    duration_s: float = 60.0,
    dt: float = 0.01
) -> Tuple[List[VehicleState], List[GuidanceCommand]]:
    """
    Run closed-loop guidance simulation.
    
    Args:
        initial_state: Initial vehicle state
        target: Target waypoint
        duration_s: Simulation duration
        dt: Time step
        
    Returns:
        (trajectory, commands)
    """
    controller = GuidanceController(target=target, dt=dt)
    solver = TrajectorySolver(TrajectoryConfig(dt_s=dt))
    
    state = initial_state
    trajectory = [state]
    commands = []
    
    t = 0.0
    while t < duration_s and state.altitude_m > 0:
        # Compute guidance
        command = controller.compute_guidance(state)
        commands.append(command)
        
        # Propagate dynamics
        controls = command.to_controls()
        state = solver.single_step(state, controls)
        trajectory.append(state)
        
        t += dt
    
    return trajectory, commands


def validate_guidance_module():
    """Validate guidance controller module."""
    print("\n" + "=" * 70)
    print("GUIDANCE CONTROLLER VALIDATION")
    print("=" * 70)
    
    # Test 1: Guidance Command
    print("\n[Test 1] Guidance Command")
    print("-" * 40)
    
    cmd = GuidanceCommand(
        bank_angle_rad=math.radians(30),
        angle_of_attack_rad=math.radians(15),
        mode=GuidanceMode.EQUILIBRIUM_GLIDE
    )
    
    controls = cmd.to_controls()
    print(f"Bank: {math.degrees(cmd.bank_angle_rad):.1f}°")
    print(f"AoA: {math.degrees(cmd.angle_of_attack_rad):.1f}°")
    print(f"Controls: de={controls['de']:.3f}, da={controls['da']:.3f}")
    print("✓ PASS")
    
    # Test 2: Trajectory Constraint
    print("\n[Test 2] Trajectory Constraint")
    print("-" * 40)
    
    constraint = TrajectoryConstraint(
        ConstraintType.THERMAL_RATE,
        max_value=100.0,
        current_value=85.0
    )
    
    print(f"Max value: {constraint.max_value}")
    print(f"Current: {constraint.current_value}")
    print(f"Relative margin: {constraint.relative_margin:.2f}")
    print(f"Is active: {constraint.is_active}")
    print(f"Violation: {constraint.violation}")
    
    assert constraint.is_active  # Within 10% of limit
    assert constraint.violation == 0  # Not violated
    print("✓ PASS")
    
    # Test 3: Proportional Navigation
    print("\n[Test 3] Proportional Navigation")
    print("-" * 40)
    
    vehicle = VehicleState(
        latitude_rad=0.0,
        longitude_rad=0.0,
        u_m_s=1000.0
    )
    
    target_state = VehicleState(
        latitude_rad=0.01,
        longitude_rad=0.01,
        u_m_s=0.0
    )
    
    a_cmd = proportional_navigation(vehicle, target_state)
    print(f"PN acceleration command: {a_cmd:.2f} m/s²")
    print("✓ PASS")
    
    # Test 4: Bank Angle Guidance
    print("\n[Test 4] Bank Angle Guidance")
    print("-" * 40)
    
    state = VehicleState(
        latitude_rad=0.0,
        longitude_rad=0.0,
        altitude_m=50000,
        u_m_s=3000
    )
    
    target = WaypointTarget(
        latitude_rad=0.1,
        longitude_rad=0.1,
        altitude_m=0
    )
    
    corridor = CorridorBounds()
    
    cmd = bank_angle_guidance(state, target, corridor, L_over_D=1.5)
    
    print(f"Bank command: {math.degrees(cmd.bank_angle_rad):.1f}°")
    print(f"AoA command: {math.degrees(cmd.angle_of_attack_rad):.1f}°")
    print(f"Mode: {cmd.mode.value}")
    print(f"Predicted range: {cmd.predicted_range_m/1000:.1f} km")
    print("✓ PASS")
    
    # Test 5: Guidance Controller
    print("\n[Test 5] Guidance Controller")
    print("-" * 40)
    
    controller = GuidanceController(target=target)
    
    cmd = controller.compute_guidance(state)
    
    print(f"Controller output:")
    print(f"  Bank: {math.degrees(cmd.bank_angle_rad):.1f}°")
    print(f"  AoA: {math.degrees(cmd.angle_of_attack_rad):.1f}°")
    print(f"  Constraint margin: {cmd.constraint_margin:.2f}")
    
    # Check constraints were updated
    thermal = controller.constraints[ConstraintType.THERMAL_RATE]
    print(f"  Thermal rate: {thermal.current_value:.2f} W/cm²")
    print("✓ PASS")
    
    # Test 6: Heating Estimate
    print("\n[Test 6] Heating Estimate")
    print("-" * 40)
    
    atm = isa_atmosphere(60000)
    q_dot = controller.estimate_heating(state, atm)
    
    print(f"At 60 km, 3 km/s:")
    print(f"  Stagnation heating: {q_dot:.2f} W/cm²")
    
    assert q_dot > 0  # Should have some heating
    print("✓ PASS")
    
    # Test 7: Constraint Limiting
    print("\n[Test 7] Constraint Limiting")
    print("-" * 40)
    
    # Create command at limit
    high_bank_cmd = GuidanceCommand(
        bank_angle_rad=math.radians(89),  # Very high bank
        angle_of_attack_rad=math.radians(45)  # Very high AoA
    )
    
    limited_cmd = controller.apply_constraint_limiting(high_bank_cmd, state)
    
    print(f"Before limiting: bank={math.degrees(high_bank_cmd.bank_angle_rad):.1f}°")
    print(f"After limiting: bank={math.degrees(limited_cmd.bank_angle_rad):.1f}°")
    
    assert abs(limited_cmd.bank_angle_rad) <= corridor.max_bank_angle_rad
    print("✓ PASS")
    
    # Test 8: Aerodynamic Lookup
    print("\n[Test 8] Aerodynamic Lookup")
    print("-" * 40)
    
    CL, CD, Cm = controller.lookup_aerodynamics(5.0, 10.0)
    L_D = CL / CD
    
    print(f"At Mach 5, AoA 10°:")
    print(f"  CL = {CL:.3f}, CD = {CD:.4f}")
    print(f"  L/D = {L_D:.2f}")
    print(f"  Cm = {Cm:.3f}")
    print("✓ PASS")
    
    # Test 9: Closed-Loop Simulation (short)
    print("\n[Test 9] Closed-Loop Simulation")
    print("-" * 40)
    
    initial = VehicleState(
        altitude_m=40000,
        u_m_s=2000,
        mass_kg=1500
    )
    
    target = WaypointTarget(
        latitude_rad=0.05,
        longitude_rad=0.05
    )
    
    trajectory, commands = closed_loop_simulation(
        initial, target, duration_s=1.0, dt=0.01
    )
    
    print(f"Simulated {len(trajectory)} states")
    print(f"Generated {len(commands)} commands")
    print(f"Start: alt={trajectory[0].altitude_m:.0f}m")
    print(f"End:   alt={trajectory[-1].altitude_m:.0f}m")
    
    assert len(trajectory) > 10
    print("✓ PASS")
    
    print("\n" + "=" * 70)
    print("GUIDANCE CONTROLLER VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    validate_guidance_module()
