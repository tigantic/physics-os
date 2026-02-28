"""
Hypersonic Flight Environment
==============================

Gymnasium-compatible RL environment for training autonomous
hypersonic flight agents.

The agent must navigate a Mach 10 glider through a "Safety Tube"
(optimal trajectory from Phase 3) while managing:
- Heat flux (TPS limits)
- Dynamic pressure (structural limits)
- G-forces (control authority)

Reward Function:
    R = (Velocity * 0.1) - (Heat_Flux * 2.0) - (Distance_From_Tube * 5.0)

The agent that survives is the one that learns to hug the tube.
"""

from __future__ import annotations

import os

# Phase 3 imports
import sys
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ontic.applied.physics.hypersonic import AtmosphericModel, VehicleConfig

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class HypersonicEnvConfig:
    """Configuration for hypersonic flight environment."""

    # Vehicle
    mach: float = 10.0
    nose_radius: float = 0.15  # meters
    q_limit: float = 50000.0  # Pa (dynamic pressure limit)
    tps_limit: float = 2000.0  # K (thermal protection limit)
    max_g: float = 9.0  # G-force limit

    # Flight envelope
    min_altitude: float = 20000.0  # meters (20 km)
    max_altitude: float = 60000.0  # meters (60 km)
    min_velocity: float = 2000.0  # m/s
    max_velocity: float = 3500.0  # m/s (Mach 10 @ altitude)

    # Grid (observation space)
    grid_size: int = 32  # 32x32 local hazard field

    # Episode
    max_steps: int = 1000
    dt: float = 0.1  # seconds per step

    # Control
    max_pitch_rate: float = 5.0  # deg/s
    max_roll_rate: float = 10.0  # deg/s
    max_thrust_change: float = 0.1  # fraction/step

    # Reward weights
    velocity_weight: float = 0.1
    heat_penalty: float = 2.0
    tube_penalty: float = 5.0
    survival_bonus: float = 1.0
    crash_penalty: float = -100.0

    # Trajectory tube
    tube_radius: float = 500.0  # meters

    # Turbulence
    turbulence_level: str = "high"  # "none", "low", "medium", "high"

    # Seed
    seed: int | None = None


# =============================================================================
# AIRCRAFT STATE
# =============================================================================


@dataclass
class AircraftState:
    """State of the hypersonic vehicle."""

    # Position (meters, in local tangent plane)
    x: float = 0.0
    y: float = 0.0
    z: float = 30000.0  # altitude

    # Velocity (m/s)
    vx: float = 3000.0
    vy: float = 0.0
    vz: float = 0.0

    # Orientation (degrees)
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0

    # Thrust (0-1)
    thrust: float = 0.5

    # Thermal state
    heat_accumulated: float = 0.0
    current_heat_flux: float = 0.0

    # Status
    alive: bool = True
    crash_reason: str = ""

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for observation."""
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.vx,
                self.vy,
                self.vz,
                self.pitch,
                self.roll,
                self.yaw,
                self.thrust,
                self.heat_accumulated,
                self.current_heat_flux,
            ],
            dtype=np.float32,
        )

    @property
    def speed(self) -> float:
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])


# =============================================================================
# TRAJECTORY TUBE
# =============================================================================


@dataclass
class TrajectoryTube:
    """
    The "Safety Tube" - optimal path from Phase 3.

    Agents are rewarded for staying inside this tube.
    """

    waypoints: np.ndarray  # Shape: (N, 3) - x, y, z
    radius: float = 500.0  # meters

    @classmethod
    def generate_test_trajectory(cls, num_waypoints: int = 100) -> TrajectoryTube:
        """Generate a test trajectory for training."""
        t = np.linspace(0, 1, num_waypoints)

        # Gentle descent with curves - starting higher for Mach 10 safety
        x = t * 100000  # 100 km horizontal
        y = 2000 * np.sin(2 * np.pi * t)  # Lateral deviation (reduced)
        z = 50000 - 15000 * t  # Descend from 50km to 35km (higher altitude for Mach 10)

        waypoints = np.stack([x, y, z], axis=1)
        return cls(waypoints=waypoints)

    def distance_to_tube(self, position: np.ndarray) -> tuple[float, int]:
        """
        Calculate distance from position to tube centerline.

        Returns:
            distance: Distance to nearest point on centerline
            segment_idx: Index of nearest waypoint
        """
        # Find nearest waypoint
        diffs = self.waypoints - position
        distances = np.linalg.norm(diffs, axis=1)
        nearest_idx = np.argmin(distances)

        return distances[nearest_idx], nearest_idx

    def is_inside(self, position: np.ndarray) -> bool:
        """Check if position is inside the tube."""
        dist, _ = self.distance_to_tube(position)
        return dist <= self.radius

    def get_tube_direction(self, segment_idx: int) -> np.ndarray:
        """Get the direction of the tube at a segment."""
        if segment_idx >= len(self.waypoints) - 1:
            segment_idx = len(self.waypoints) - 2

        direction = self.waypoints[segment_idx + 1] - self.waypoints[segment_idx]
        return direction / (np.linalg.norm(direction) + 1e-8)


# =============================================================================
# HYPERSONIC ENVIRONMENT
# =============================================================================


class HypersonicEnv(gym.Env):
    """
    Gymnasium environment for hypersonic flight training.

    Observation Space:
        - Aircraft state (12 values)
        - Distance to tube (1 value)
        - Tube direction (3 values)
        - Local hazard field (32x32 grid)

    Action Space:
        - Pitch rate (-1 to 1)
        - Roll rate (-1 to 1)
        - Thrust change (-1 to 1)

    Reward:
        R = velocity_bonus - heat_penalty - tube_penalty + survival_bonus
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        # Parse config
        if config is None:
            config = {}
        self.config = HypersonicEnvConfig(**config)
        self.render_mode = render_mode

        # Initialize random state
        self._np_random = None

        # Initialize vehicle config for physics
        self.vehicle = VehicleConfig(
            mach_cruise=self.config.mach,
            q_limit_Pa=self.config.q_limit,
            tps_limit_K=self.config.tps_limit,
            nose_radius_m=self.config.nose_radius,
        )

        # Initialize atmospheric model
        self.atmosphere = AtmosphericModel()

        # Generate trajectory tube (from Phase 3)
        self.tube = TrajectoryTube.generate_test_trajectory()
        self.tube.radius = self.config.tube_radius

        # Aircraft state
        self.aircraft = AircraftState()

        # Episode tracking
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_stats = {
            "length": 0,
            "return": 0.0,
            "survival_time": 0.0,
            "max_heat_flux": 0.0,
            "min_tube_distance": float("inf"),
            "crash_reason": "",
        }

        # Define action space: [pitch_rate, roll_rate, thrust_change]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # Define observation space
        # State vector (12) + tube info (4) + local hazard (32x32)
        state_dim = 12 + 4 + self.config.grid_size * self.config.grid_size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        # Reset aircraft to start of tube
        start_pos = self.tube.waypoints[0]
        start_dir = self.tube.get_tube_direction(0)

        # Add some randomization for robustness
        position_noise = self._np_random.normal(0, 50, size=3)  # 50m std
        velocity_noise = self._np_random.normal(0, 10, size=3)  # 10 m/s std

        initial_speed = self.config.mach * 300  # Approximate Mach 10 at altitude

        self.aircraft = AircraftState(
            x=float(start_pos[0] + position_noise[0]),
            y=float(start_pos[1] + position_noise[1]),
            z=float(start_pos[2] + position_noise[2]),
            vx=float(start_dir[0] * initial_speed + velocity_noise[0]),
            vy=float(start_dir[1] * initial_speed + velocity_noise[1]),
            vz=float(start_dir[2] * initial_speed + velocity_noise[2]),
            thrust=0.5,
            heat_accumulated=0.0,
            alive=True,
        )

        # Reset episode tracking
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_stats = {
            "length": 0,
            "return": 0.0,
            "survival_time": 0.0,
            "max_heat_flux": 0.0,
            "min_tube_distance": float("inf"),
            "crash_reason": "",
        }

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: [pitch_rate, roll_rate, thrust_change] in [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Apply action
        self._apply_action(action)

        # Physics step
        self._physics_step()

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.step_count >= self.config.max_steps

        # Calculate reward
        reward = self._calculate_reward(terminated)

        # Update tracking
        self.step_count += 1
        self.total_reward += reward
        self.episode_stats["length"] = self.step_count
        self.episode_stats["return"] = self.total_reward
        self.episode_stats["survival_time"] = self.step_count * self.config.dt

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray):
        """Apply control action to aircraft."""
        pitch_rate = action[0] * self.config.max_pitch_rate
        roll_rate = action[1] * self.config.max_roll_rate
        thrust_change = action[2] * self.config.max_thrust_change

        # Update orientation
        self.aircraft.pitch += pitch_rate * self.config.dt
        self.aircraft.roll += roll_rate * self.config.dt
        self.aircraft.thrust = np.clip(self.aircraft.thrust + thrust_change, 0.0, 1.0)

        # Clamp pitch/roll to reasonable values
        self.aircraft.pitch = np.clip(self.aircraft.pitch, -30, 30)
        self.aircraft.roll = np.clip(self.aircraft.roll, -60, 60)

    def _physics_step(self):
        """Advance physics simulation one timestep."""
        dt = self.config.dt

        # Get atmospheric conditions at current altitude
        altitude = self.aircraft.z
        rho = self.atmosphere.density(altitude)
        temperature = self.atmosphere.temperature(altitude)

        # Calculate forces
        speed = self.aircraft.speed

        # Drag (simplified)
        drag_coeff = 0.05
        drag = 0.5 * rho * speed**2 * drag_coeff

        # Thrust (scramjet approximation)
        max_thrust = 50000  # N
        thrust_force = self.aircraft.thrust * max_thrust

        # Lift (pitch controls vertical acceleration)
        pitch_rad = np.radians(self.aircraft.pitch)
        lift_coeff = 0.1 * np.sin(pitch_rad)
        lift = 0.5 * rho * speed**2 * lift_coeff

        # Gravity
        gravity = 9.81 * (6371000 / (6371000 + altitude)) ** 2

        # Update velocity
        velocity_dir = self.aircraft.velocity / (speed + 1e-8)

        # Axial acceleration (thrust - drag) / mass
        mass = 5000  # kg
        axial_accel = (thrust_force - drag) / mass

        # Normal acceleration (lift - gravity component)
        normal_accel = lift / mass - gravity * np.cos(pitch_rad)

        # Apply accelerations
        self.aircraft.vx += velocity_dir[0] * axial_accel * dt
        self.aircraft.vz += normal_accel * dt

        # Roll affects lateral movement
        roll_rad = np.radians(self.aircraft.roll)
        lateral_accel = lift / mass * np.sin(roll_rad)
        self.aircraft.vy += lateral_accel * dt

        # Add turbulence
        if self.config.turbulence_level != "none":
            turbulence_std = {
                "low": 5.0,
                "medium": 15.0,
                "high": 30.0,
            }.get(self.config.turbulence_level, 15.0)

            turbulence = self._np_random.normal(0, turbulence_std, size=3)
            self.aircraft.vx += turbulence[0] * dt
            self.aircraft.vy += turbulence[1] * dt
            self.aircraft.vz += turbulence[2] * dt

        # Update position
        self.aircraft.x += self.aircraft.vx * dt
        self.aircraft.y += self.aircraft.vy * dt
        self.aircraft.z += self.aircraft.vz * dt

        # Calculate heat flux (Sutton-Graves)
        stag_rho = rho  # Simplified
        stag_v = speed
        k_sutton = 1.74153e-4  # Sutton-Graves constant
        heat_flux = (
            k_sutton * np.sqrt(stag_rho / self.vehicle.nose_radius_m) * stag_v**3
        )

        self.aircraft.current_heat_flux = heat_flux
        self.aircraft.heat_accumulated += heat_flux * dt

        # Track max heat flux
        if heat_flux > self.episode_stats["max_heat_flux"]:
            self.episode_stats["max_heat_flux"] = heat_flux

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Altitude limits
        if self.aircraft.z < self.config.min_altitude:
            self.aircraft.alive = False
            self.aircraft.crash_reason = "altitude_low"
            self.episode_stats["crash_reason"] = "Crashed: Altitude too low"
            return True

        if self.aircraft.z > self.config.max_altitude:
            self.aircraft.alive = False
            self.aircraft.crash_reason = "altitude_high"
            self.episode_stats["crash_reason"] = "Crashed: Left atmosphere"
            return True

        # Velocity limits
        if self.aircraft.speed < self.config.min_velocity:
            self.aircraft.alive = False
            self.aircraft.crash_reason = "stall"
            self.episode_stats["crash_reason"] = "Crashed: Stall"
            return True

        # Heat limit
        tps_temperature = 300 + self.aircraft.heat_accumulated / 1e6  # Simplified
        if tps_temperature > self.vehicle.tps_limit_K:
            self.aircraft.alive = False
            self.aircraft.crash_reason = "thermal"
            self.episode_stats["crash_reason"] = "Crashed: Thermal failure"
            return True

        # Dynamic pressure limit
        rho = self.atmosphere.density(self.aircraft.z)
        q = 0.5 * rho * self.aircraft.speed**2
        if q > self.vehicle.q_limit_Pa:
            self.aircraft.alive = False
            self.aircraft.crash_reason = "structural"
            self.episode_stats["crash_reason"] = "Crashed: Structural failure (Q limit)"
            return True

        return False

    def _calculate_reward(self, terminated: bool) -> float:
        """
        Calculate reward for current state.

        R = (Velocity * 0.1) - (Heat_Flux * 2.0) - (Distance_From_Tube * 5.0)
        """
        # Velocity bonus (normalized)
        speed_normalized = self.aircraft.speed / self.config.max_velocity
        velocity_bonus = speed_normalized * self.config.velocity_weight

        # Heat penalty (normalized to 0-1 range)
        heat_normalized = self.aircraft.current_heat_flux / 1e6  # Normalize by 1 MW/m²
        heat_penalty = heat_normalized * self.config.heat_penalty

        # Tube distance penalty
        tube_dist, _ = self.tube.distance_to_tube(self.aircraft.position)
        tube_normalized = tube_dist / self.config.tube_radius  # 1.0 at edge of tube
        tube_penalty = tube_normalized * self.config.tube_penalty

        # Track min distance
        if tube_dist < self.episode_stats["min_tube_distance"]:
            self.episode_stats["min_tube_distance"] = tube_dist

        # Survival bonus
        survival = self.config.survival_bonus if self.aircraft.alive else 0.0

        # Crash penalty
        crash_penalty = (
            self.config.crash_penalty if terminated and not self.aircraft.alive else 0.0
        )

        reward = velocity_bonus - heat_penalty - tube_penalty + survival + crash_penalty

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        # Aircraft state (12 values)
        state = self.aircraft.to_array()

        # Tube info (4 values)
        tube_dist, segment_idx = self.tube.distance_to_tube(self.aircraft.position)
        tube_dir = self.tube.get_tube_direction(segment_idx)
        tube_info = np.array([tube_dist] + list(tube_dir), dtype=np.float32)

        # Local hazard field (32x32)
        hazard_grid = self._get_local_hazard_field()

        # Concatenate
        obs = np.concatenate(
            [
                state,
                tube_info,
                hazard_grid.flatten(),
            ]
        )

        return obs.astype(np.float32)

    def _get_local_hazard_field(self) -> np.ndarray:
        """
        Get a local 32x32 hazard field centered on aircraft.

        This is a simplified 2D slice showing heat/pressure hazards
        in the immediate vicinity.
        """
        grid_size = self.config.grid_size

        # Create grid centered on aircraft
        half_extent = 5000  # 5km in each direction
        x = np.linspace(-half_extent, half_extent, grid_size) + self.aircraft.x
        z = np.linspace(-2000, 2000, grid_size) + self.aircraft.z  # ±2km altitude

        X, Z = np.meshgrid(x, z)

        # Calculate hazard at each point
        hazard = np.zeros((grid_size, grid_size), dtype=np.float32)

        for i in range(grid_size):
            for j in range(grid_size):
                alt = Z[i, j]
                if alt < self.config.min_altitude or alt > self.config.max_altitude:
                    hazard[i, j] = 1.0  # Max hazard outside envelope
                else:
                    # Simplified hazard based on altitude
                    rho = self.atmosphere.density(alt)
                    speed = self.aircraft.speed
                    q = 0.5 * rho * speed**2
                    q_hazard = q / self.vehicle.q_limit_Pa

                    # Heat hazard increases at low altitude
                    heat_hazard = (self.config.max_altitude - alt) / (
                        self.config.max_altitude - self.config.min_altitude
                    )

                    hazard[i, j] = np.clip(max(q_hazard, heat_hazard * 0.5), 0, 1)

        return hazard

    def _get_info(self) -> dict[str, Any]:
        """Get info dict."""
        return {
            "step": self.step_count,
            "aircraft": {
                "position": self.aircraft.position.tolist(),
                "velocity": self.aircraft.velocity.tolist(),
                "speed": self.aircraft.speed,
                "altitude": self.aircraft.z,
                "heat_flux": self.aircraft.current_heat_flux,
                "alive": self.aircraft.alive,
            },
            "episode": self.episode_stats.copy(),
        }

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        # Create a simple visualization
        import numpy as np

        # 256x256 RGB image
        img = np.zeros((256, 256, 3), dtype=np.uint8)

        # Draw tube (green centerline)
        for i, wp in enumerate(self.tube.waypoints[::5]):  # Every 5th waypoint
            x_norm = int((wp[0] / 100000) * 255)  # Normalize x
            y_norm = int(((wp[2] - 20000) / 40000) * 255)  # Normalize altitude
            if 0 <= x_norm < 256 and 0 <= y_norm < 256:
                img[255 - y_norm, x_norm] = [0, 255, 0]  # Green

        # Draw aircraft (red dot)
        x_norm = int((self.aircraft.x / 100000) * 255)
        y_norm = int(((self.aircraft.z - 20000) / 40000) * 255)
        if 0 <= x_norm < 256 and 0 <= y_norm < 256:
            # Draw 3x3 red dot
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    px, py = x_norm + dx, 255 - y_norm + dy
                    if 0 <= px < 256 and 0 <= py < 256:
                        img[py, px] = [255, 0, 0]  # Red

        if self.render_mode == "human":
            # Would use cv2 or matplotlib here
            pass

        return img

    def close(self):
        """Clean up resources."""
        pass


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def make_hypersonic_env(
    mach: float = 10.0,
    turbulence: str = "high",
    **kwargs,
) -> HypersonicEnv:
    """
    Factory function to create HypersonicEnv.

    Args:
        mach: Target Mach number
        turbulence: "none", "low", "medium", "high"
        **kwargs: Additional config options

    Returns:
        Configured HypersonicEnv instance
    """
    config = {
        "mach": mach,
        "turbulence_level": turbulence,
        **kwargs,
    }
    return HypersonicEnv(config=config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HYPERSONIC ENVIRONMENT TEST")
    print("=" * 60)

    # Create environment
    env = make_hypersonic_env(mach=10.0, turbulence="high")

    print(f"\nObservation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Run random agent
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial altitude: {info['aircraft']['altitude']:.0f} m")
    print(f"Initial speed: {info['aircraft']['speed']:.0f} m/s")

    total_reward = 0
    steps = 0

    print("\n[RANDOM AGENT] Running 100 steps...")

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            print(f"\nEpisode ended at step {steps}")
            print(f"Reason: {info['episode'].get('crash_reason', 'timeout')}")
            break

    print(f"\n{'=' * 40}")
    print("RESULTS")
    print(f"{'=' * 40}")
    print(f"Steps survived: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Mean reward: {total_reward / steps:.4f}")
    print(f"Final altitude: {info['aircraft']['altitude']:.0f} m")
    print(f"Final speed: {info['aircraft']['speed']:.0f} m/s")
    print(f"Max heat flux: {info['episode']['max_heat_flux']:.2e} W/m²")
    print(f"Min tube distance: {info['episode']['min_tube_distance']:.0f} m")

    print("\n✓ HypersonicEnv operational")
    env.close()
