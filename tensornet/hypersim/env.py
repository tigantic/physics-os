"""
Fluid Environment
=================

Gymnasium-compatible environment for fluid dynamics control.
Uses QTT fields as state - agents never see decompressed data.

Features:
    - Standard gym interface (reset, step, render)
    - Deterministic seeding and checkpointing
    - Configurable physics (viscosity, forcing, BCs)
    - Multi-agent support
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class FluidEnvConfig:
    """Configuration for fluid environment."""

    # Grid
    dims: int = 2
    bits_per_dim: int = 8  # 256x256 grid
    rank: int = 16

    # Physics
    viscosity: float = 0.01
    dt: float = 0.01

    # Episode
    max_steps: int = 1000
    steps_per_action: int = 1  # Physics steps per agent step

    # Observation
    obs_resolution: int = 64  # Downsampled observation
    obs_channels: int = 3  # velocity_x, velocity_y, vorticity

    # Action
    action_type: str = "continuous"  # 'continuous' or 'discrete'
    n_actuators: int = 4  # Number of control points
    max_force: float = 1.0

    # Rewards
    reward_scale: float = 1.0

    # Rendering
    render_mode: str = "rgb_array"
    render_resolution: int = 256

    # Reproducibility
    seed: int | None = None

    def __post_init__(self):
        self.grid_size = 2**self.bits_per_dim


@dataclass
class EnvState:
    """Serializable environment state for checkpointing."""

    # Field state (QTT cores as numpy arrays)
    cores: list[np.ndarray]

    # Episode state
    step_count: int
    total_reward: float

    # Random state
    np_random_state: dict[str, Any]

    # Metadata
    timestamp: float = 0.0
    hash: str = ""

    def compute_hash(self) -> str:
        """Compute deterministic hash of state."""
        data = b""
        for core in self.cores:
            data += core.tobytes()
        data += str(self.step_count).encode()
        return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class StepResult:
    """Result from environment step."""

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


# =============================================================================
# FLUID ENVIRONMENT
# =============================================================================


class FluidEnv:
    """
    Gymnasium-compatible fluid dynamics environment.

    State: QTT-compressed velocity field
    Observation: Downsampled field slice
    Action: Control forces at actuator locations
    Reward: Task-specific (target matching, vorticity control, etc.)

    Usage:
        env = FluidEnv(config)
        obs, info = env.reset(seed=42)

        for _ in range(1000):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()
    """

    # Gym interface
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: FluidEnvConfig | None = None,
        reward_fn: RewardFunction | None = None,
    ):
        self.config = config or FluidEnvConfig()
        self.reward_fn = reward_fn

        # Lazy imports to avoid circular dependencies
        self._field_class = None
        self._ops_class = None

        # State
        self.field = None
        self.step_count = 0
        self.total_reward = 0.0
        self._np_random = None

        # Actuators
        self._actuator_positions = self._init_actuators()

        # Caches
        self._obs_cache = None
        self._render_cache = None

        # Stats
        self.episode_stats = {
            "length": 0,
            "return": 0.0,
            "physics_time_ms": 0.0,
        }

    def _init_actuators(self) -> np.ndarray:
        """Initialize actuator positions."""
        n = self.config.n_actuators

        if n == 4:
            # Corners
            return np.array(
                [
                    [0.25, 0.25],
                    [0.75, 0.25],
                    [0.25, 0.75],
                    [0.75, 0.75],
                ]
            )
        else:
            # Uniform grid
            side = int(np.ceil(np.sqrt(n)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < n:
                        x = (i + 0.5) / side
                        y = (j + 0.5) / side
                        positions.append([x, y])
            return np.array(positions)

    @property
    def observation_space(self):
        """Gymnasium observation space."""
        res = self.config.obs_resolution
        channels = self.config.obs_channels

        # Box space for image-like observations
        return {
            "type": "Box",
            "shape": (channels, res, res),
            "dtype": "float32",
            "low": -10.0,
            "high": 10.0,
        }

    @property
    def action_space(self):
        """Gymnasium action space."""
        n = self.config.n_actuators
        max_f = self.config.max_force

        if self.config.action_type == "continuous":
            # 2D force per actuator
            return {
                "type": "Box",
                "shape": (n, 2),
                "dtype": "float32",
                "low": -max_f,
                "high": max_f,
            }
        else:
            # Discrete: 5 actions per actuator (none, up, down, left, right)
            return {
                "type": "MultiDiscrete",
                "nvec": [5] * n,
            }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
                - 'init': Initial condition ('taylor_green', 'random', 'vortex')
                - 'state': EnvState to restore from

        Returns:
            (observation, info) tuple
        """
        # Seed random state
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        elif self._np_random is None:
            self._np_random = np.random.RandomState()

        # Handle options
        options = options or {}

        # Restore from checkpoint?
        if "state" in options:
            self._restore_state(options["state"])
        else:
            # Create new field
            init = options.get("init", "taylor_green")
            self._create_field(init)

        # Reset counters
        self.step_count = 0
        self.total_reward = 0.0
        self._obs_cache = None
        self._render_cache = None

        # Reset stats
        self.episode_stats = {
            "length": 0,
            "return": 0.0,
            "physics_time_ms": 0.0,
        }

        # Get observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Agent action (forces at actuators)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        t_start = time.perf_counter()

        # Convert action to forces
        forces = self._action_to_forces(action)

        # Apply forces and step physics
        for _ in range(self.config.steps_per_action):
            self._apply_forces(forces)
            self._step_physics()

        self.step_count += 1

        # Compute reward
        reward = self._compute_reward()
        self.total_reward += reward

        # Check termination
        terminated = self._check_terminated()
        truncated = self.step_count >= self.config.max_steps

        # Get observation
        obs = self._get_observation()
        info = self._get_info()

        # Update stats
        physics_time = (time.perf_counter() - t_start) * 1000
        self.episode_stats["length"] = self.step_count
        self.episode_stats["return"] = self.total_reward
        self.episode_stats["physics_time_ms"] += physics_time

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render environment state."""
        if self.field is None:
            return None

        if self._render_cache is None:
            from tensornet.hypervisual import VIRIDIS, SliceEngine, apply_colormap

            slicer = SliceEngine(self.field)
            result = slicer.slice(
                plane="xy",
                depth=0.5,
                resolution=self.config.render_resolution,
            )

            # Apply colormap
            self._render_cache = apply_colormap(result.data, VIRIDIS)

        return self._render_cache

    def close(self):
        """Clean up resources."""
        self.field = None
        self._obs_cache = None
        self._render_cache = None

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def get_state(self) -> EnvState:
        """Get serializable state for checkpointing."""
        cores = [c.cpu().numpy() for c in self.field.cores]

        state = EnvState(
            cores=cores,
            step_count=self.step_count,
            total_reward=self.total_reward,
            np_random_state=self._np_random.get_state(),
            timestamp=time.time(),
        )
        state.hash = state.compute_hash()

        return state

    def set_state(self, state: EnvState):
        """Restore from serialized state."""
        self._restore_state(state)

    def _restore_state(self, state: EnvState):
        """Restore environment from state."""
        from tensornet.substrate import Field

        cores = [torch.from_numpy(c) for c in state.cores]

        self.field = Field(
            cores=cores,
            dims=self.config.dims,
            bits_per_dim=self.config.bits_per_dim,
        )

        self.step_count = state.step_count
        self.total_reward = state.total_reward
        self._np_random.set_state(state.np_random_state)

        self._obs_cache = None
        self._render_cache = None

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _create_field(self, init: str = "taylor_green"):
        """Create initial field."""
        from tensornet.substrate import Field

        self.field = Field.create(
            dims=self.config.dims,
            bits_per_dim=self.config.bits_per_dim,
            rank=self.config.rank,
            init=init,
        )

    def _get_observation(self) -> np.ndarray:
        """Get observation from current field state."""
        if self._obs_cache is not None:
            return self._obs_cache

        if self.field is None:
            res = self.config.obs_resolution
            channels = self.config.obs_channels
            return np.zeros((channels, res, res), dtype=np.float32)

        from tensornet.hypervisual import SliceEngine

        slicer = SliceEngine(self.field)
        result = slicer.slice(
            plane="xy",
            depth=0.5,
            resolution=self.config.obs_resolution,
        )

        # Stack channels (for now just replicate)
        data = result.data.astype(np.float32)
        obs = np.stack([data] * self.config.obs_channels, axis=0)

        self._obs_cache = obs
        return obs

    def _get_info(self) -> dict[str, Any]:
        """Get info dict."""
        info = {
            "step": self.step_count,
            "total_reward": self.total_reward,
        }

        if self.field is not None:
            info["rank"] = self.field.rank
            info["energy"] = float(sum(c.norm().item() for c in self.field.cores))

        return info

    def _action_to_forces(
        self, action: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Convert action array to force specifications."""
        action = np.asarray(action, dtype=np.float32)

        if self.config.action_type == "discrete":
            # Convert discrete to continuous
            directions = {
                0: np.array([0, 0]),
                1: np.array([0, 1]),
                2: np.array([0, -1]),
                3: np.array([1, 0]),
                4: np.array([-1, 0]),
            }
            forces = []
            for i, a in enumerate(action.flatten()):
                pos = self._actuator_positions[i]
                force = directions.get(int(a), np.array([0, 0])) * self.config.max_force
                forces.append((pos, force))
        else:
            # Continuous actions
            action = action.reshape(-1, 2)
            action = np.clip(action, -self.config.max_force, self.config.max_force)

            forces = []
            for i, force in enumerate(action):
                pos = self._actuator_positions[i]
                forces.append((pos, force))

        return forces

    def _apply_forces(self, forces: list[tuple[np.ndarray, np.ndarray]]):
        """Apply forces to field."""
        from tensornet.fieldops import Impulse

        for pos, force in forces:
            if np.linalg.norm(force) > 1e-6:
                impulse = Impulse(
                    center=tuple(pos),
                    direction=tuple(force),
                    strength=float(np.linalg.norm(force)),
                    radius=0.1,
                )
                self.field = impulse.apply(self.field, dt=self.config.dt)

        self._obs_cache = None
        self._render_cache = None

    def _step_physics(self):
        """Advance physics simulation."""
        from tensornet.fieldops import fluid_graph

        graph = fluid_graph(viscosity=self.config.viscosity)
        self.field = graph.execute(self.field, dt=self.config.dt)

        self._obs_cache = None
        self._render_cache = None

    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        if self.reward_fn is not None:
            return self.reward_fn(self.field, self.step_count)

        # Default: negative energy (encourage dissipation)
        if self.field is None:
            return 0.0

        energy = sum(c.norm().item() for c in self.field.cores)
        return -energy * self.config.reward_scale * 0.01

    def _check_terminated(self) -> bool:
        """Check if episode should terminate (not truncate)."""
        if self.field is None:
            return False

        # Terminate on numerical instability
        for core in self.field.cores:
            if torch.isnan(core).any() or torch.isinf(core).any():
                return True

        return False

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def seed(self, seed: int) -> list[int]:
        """Set random seed (legacy interface)."""
        self._np_random = np.random.RandomState(seed)
        return [seed]

    @property
    def unwrapped(self):
        """Return base environment."""
        return self

    def __repr__(self) -> str:
        return (
            f"FluidEnv(dims={self.config.dims}, "
            f"grid={self.config.grid_size}x{self.config.grid_size}, "
            f"rank={self.config.rank})"
        )
