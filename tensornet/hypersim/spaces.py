"""
Observation and Action Spaces
=============================

Gymnasium-compatible spaces for QTT field environments.

Key concepts:
    - FieldObservation: Downsampled slices of field state
    - FieldAction: Force/control inputs to actuators
    - ActionMask: Valid action constraints
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Union
from enum import Enum


# =============================================================================
# OBSERVATION SPACES
# =============================================================================

class ObservationType(Enum):
    """Type of observation encoding."""
    SLICE_2D = "slice_2d"          # 2D slice at fixed position
    MULTI_SLICE = "multi_slice"    # Multiple 2D slices
    VOLUME_3D = "volume_3d"        # Full 3D volume (downsampled)
    PROBE_POINTS = "probe_points"  # Discrete sample points
    STATISTICS = "statistics"      # Global statistics only
    HYBRID = "hybrid"              # Combination of above


@dataclass
class ObservationConfig:
    """Configuration for observation extraction."""
    obs_type: ObservationType = ObservationType.SLICE_2D
    resolution: Tuple[int, ...] = (64, 64)
    slice_axis: int = 2
    slice_positions: List[float] = field(default_factory=lambda: [0.5])
    probe_locations: Optional[np.ndarray] = None
    include_stats: bool = True
    include_velocity: bool = True
    include_pressure: bool = True
    include_vorticity: bool = False
    stack_frames: int = 1


class FieldObservation:
    """
    Extract observations from QTT fields.
    
    Converts high-dimensional QTT state to fixed-size numpy array
    suitable for neural network input.
    """
    
    def __init__(self, config: ObservationConfig):
        self.config = config
        self._frame_buffer: List[np.ndarray] = []
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of observation array."""
        if self.config.obs_type == ObservationType.SLICE_2D:
            channels = self._num_channels()
            frames = self.config.stack_frames
            return (channels * frames, *self.config.resolution)
        
        elif self.config.obs_type == ObservationType.MULTI_SLICE:
            channels = self._num_channels() * len(self.config.slice_positions)
            frames = self.config.stack_frames
            return (channels * frames, *self.config.resolution)
        
        elif self.config.obs_type == ObservationType.VOLUME_3D:
            channels = self._num_channels()
            return (channels, *self.config.resolution)
        
        elif self.config.obs_type == ObservationType.PROBE_POINTS:
            n_probes = len(self.config.probe_locations) if self.config.probe_locations is not None else 16
            return (n_probes, self._num_channels())
        
        elif self.config.obs_type == ObservationType.STATISTICS:
            return (self._num_stats(),)
        
        else:
            # Hybrid: flattened combination
            return (self._hybrid_size(),)
    
    def _num_channels(self) -> int:
        """Number of field channels in observation."""
        c = 0
        if self.config.include_velocity:
            c += 3  # vx, vy, vz
        if self.config.include_pressure:
            c += 1
        if self.config.include_vorticity:
            c += 3
        return max(1, c)
    
    def _num_stats(self) -> int:
        """Number of statistics features."""
        return 8  # mean, std, min, max, energy, enstrophy, divergence, vorticity
    
    def _hybrid_size(self) -> int:
        """Size of hybrid observation."""
        slice_size = np.prod(self.config.resolution) * self._num_channels()
        stats_size = self._num_stats()
        return int(slice_size + stats_size)
    
    def extract(self, field: 'Field') -> np.ndarray:
        """
        Extract observation from field.
        
        Args:
            field: QTT field to observe
            
        Returns:
            Observation as numpy array
        """
        if self.config.obs_type == ObservationType.SLICE_2D:
            obs = self._extract_slice(field, self.config.slice_positions[0])
        
        elif self.config.obs_type == ObservationType.MULTI_SLICE:
            slices = [self._extract_slice(field, pos) for pos in self.config.slice_positions]
            obs = np.concatenate(slices, axis=0)
        
        elif self.config.obs_type == ObservationType.VOLUME_3D:
            obs = self._extract_volume(field)
        
        elif self.config.obs_type == ObservationType.PROBE_POINTS:
            obs = self._extract_probes(field)
        
        elif self.config.obs_type == ObservationType.STATISTICS:
            obs = self._extract_stats(field)
        
        else:
            # Hybrid
            slice_obs = self._extract_slice(field, 0.5).flatten()
            stats_obs = self._extract_stats(field)
            obs = np.concatenate([slice_obs, stats_obs])
        
        # Frame stacking
        if self.config.stack_frames > 1:
            obs = self._stack_frame(obs)
        
        return obs.astype(np.float32)
    
    def _extract_slice(self, field: 'Field', position: float) -> np.ndarray:
        """Extract 2D slice from field."""
        from ..hypervisual import SliceEngine
        
        engine = SliceEngine(resolution=self.config.resolution)
        slice_data = engine.slice(
            field,
            axis=self.config.slice_axis,
            position=position,
        )
        
        # Ensure correct channel count
        if slice_data.ndim == 2:
            slice_data = slice_data[np.newaxis, ...]
        
        return slice_data
    
    def _extract_volume(self, field: 'Field') -> np.ndarray:
        """Extract downsampled 3D volume."""
        # Sample at regular grid points
        res = self.config.resolution
        coords = np.stack(np.meshgrid(
            np.linspace(0, 1, res[0]),
            np.linspace(0, 1, res[1]),
            np.linspace(0, 1, res[2] if len(res) > 2 else 8),
            indexing='ij',
        ), axis=-1).reshape(-1, 3)
        
        values = field.sample(coords)
        return values.reshape(res[0], res[1], -1).transpose(2, 0, 1)
    
    def _extract_probes(self, field: 'Field') -> np.ndarray:
        """Extract values at probe points."""
        locations = self.config.probe_locations
        if locations is None:
            # Default grid of probes
            locations = np.random.rand(16, 3)
        
        values = field.sample(locations)
        return values
    
    def _extract_stats(self, field: 'Field') -> np.ndarray:
        """Extract global statistics."""
        stats = field.stats()
        return np.array([
            stats.mean,
            stats.std,
            stats.min_val,
            stats.max_val,
            stats.energy,
            stats.enstrophy,
            stats.divergence,
            stats.vorticity_magnitude,
        ], dtype=np.float32)
    
    def _stack_frame(self, obs: np.ndarray) -> np.ndarray:
        """Stack current frame with previous frames."""
        self._frame_buffer.append(obs)
        
        # Keep only last N frames
        if len(self._frame_buffer) > self.config.stack_frames:
            self._frame_buffer.pop(0)
        
        # Pad with zeros if not enough frames
        while len(self._frame_buffer) < self.config.stack_frames:
            self._frame_buffer.insert(0, np.zeros_like(obs))
        
        return np.concatenate(self._frame_buffer, axis=0)
    
    def reset(self):
        """Reset frame buffer."""
        self._frame_buffer.clear()


# =============================================================================
# ACTION SPACES
# =============================================================================

class ActionType(Enum):
    """Type of action encoding."""
    CONTINUOUS = "continuous"      # Box space
    DISCRETE = "discrete"          # Discrete space
    MULTI_DISCRETE = "multi_discrete"  # Multi-discrete
    HYBRID = "hybrid"              # Mixed continuous/discrete


@dataclass
class ActuatorConfig:
    """Configuration for a single actuator."""
    name: str
    position: np.ndarray  # [x, y, z] normalized coordinates
    radius: float = 0.05  # Influence radius
    max_force: float = 1.0
    force_dims: int = 3  # Number of force components
    discrete_bins: int = 5  # For discrete actions


@dataclass
class ActionConfig:
    """Configuration for action space."""
    action_type: ActionType = ActionType.CONTINUOUS
    actuators: List[ActuatorConfig] = field(default_factory=list)
    global_actions: bool = False  # Allow global (non-local) actions
    action_repeat: int = 1


class FieldAction:
    """
    Convert actions to field forces.
    
    Maps low-dimensional action vectors to high-dimensional
    force fields applied to the simulation.
    """
    
    def __init__(self, config: ActionConfig):
        self.config = config
        
        if not config.actuators:
            # Default: 4 corner actuators
            self.config.actuators = [
                ActuatorConfig("nw", np.array([0.25, 0.25, 0.5])),
                ActuatorConfig("ne", np.array([0.75, 0.25, 0.5])),
                ActuatorConfig("sw", np.array([0.25, 0.75, 0.5])),
                ActuatorConfig("se", np.array([0.75, 0.75, 0.5])),
            ]
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of action array."""
        if self.config.action_type == ActionType.CONTINUOUS:
            n_actions = sum(a.force_dims for a in self.config.actuators)
            if self.config.global_actions:
                n_actions += 6  # Global force + torque
            return (n_actions,)
        
        elif self.config.action_type == ActionType.DISCRETE:
            # Single discrete action
            return ()
        
        elif self.config.action_type == ActionType.MULTI_DISCRETE:
            return (len(self.config.actuators),)
        
        else:
            # Hybrid
            return (len(self.config.actuators) * 4,)
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Low and high bounds for continuous actions."""
        n = self.shape[0] if self.shape else 1
        low = -np.ones(n, dtype=np.float32)
        high = np.ones(n, dtype=np.float32)
        return low, high
    
    @property
    def n_discrete(self) -> Union[int, List[int]]:
        """Number of discrete actions."""
        if self.config.action_type == ActionType.DISCRETE:
            return np.prod([a.discrete_bins ** a.force_dims 
                           for a in self.config.actuators])
        elif self.config.action_type == ActionType.MULTI_DISCRETE:
            return [a.discrete_bins ** a.force_dims 
                    for a in self.config.actuators]
        return 1
    
    def to_forces(
        self,
        action: np.ndarray,
        grid_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Convert action to force field.
        
        Args:
            action: Action array from policy
            grid_shape: Shape of the simulation grid
            
        Returns:
            Force field tensor
        """
        forces = torch.zeros(*grid_shape, 3, dtype=torch.float32)
        
        if self.config.action_type == ActionType.CONTINUOUS:
            idx = 0
            for actuator in self.config.actuators:
                force_vec = action[idx:idx + actuator.force_dims]
                idx += actuator.force_dims
                
                self._apply_actuator(forces, actuator, force_vec, grid_shape)
        
        elif self.config.action_type == ActionType.DISCRETE:
            # Decode discrete action to per-actuator forces
            for i, actuator in enumerate(self.config.actuators):
                n = actuator.discrete_bins ** actuator.force_dims
                act_idx = (action // n) % n
                force_vec = self._decode_discrete(act_idx, actuator)
                self._apply_actuator(forces, actuator, force_vec, grid_shape)
        
        elif self.config.action_type == ActionType.MULTI_DISCRETE:
            for i, actuator in enumerate(self.config.actuators):
                force_vec = self._decode_discrete(action[i], actuator)
                self._apply_actuator(forces, actuator, force_vec, grid_shape)
        
        return forces
    
    def _apply_actuator(
        self,
        forces: torch.Tensor,
        actuator: ActuatorConfig,
        force_vec: np.ndarray,
        grid_shape: Tuple[int, ...],
    ):
        """Apply actuator force to field."""
        # Grid coordinates
        coords = [torch.linspace(0, 1, s) for s in grid_shape[:3]]
        grid = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
        
        # Distance from actuator
        pos = torch.tensor(actuator.position, dtype=torch.float32)
        dist = ((grid - pos) ** 2).sum(dim=-1).sqrt()
        
        # Gaussian influence
        mask = torch.exp(-0.5 * (dist / actuator.radius) ** 2)
        mask = mask / (mask.sum() + 1e-8)
        
        # Apply force
        force_tensor = torch.tensor(force_vec, dtype=torch.float32)
        force_tensor = force_tensor * actuator.max_force
        
        for d in range(min(3, len(force_vec))):
            forces[..., d] += mask * force_tensor[d]
    
    def _decode_discrete(
        self,
        action_idx: int,
        actuator: ActuatorConfig,
    ) -> np.ndarray:
        """Decode discrete action to force vector."""
        n = actuator.discrete_bins
        force_vec = np.zeros(actuator.force_dims)
        
        for d in range(actuator.force_dims):
            bin_idx = (action_idx // (n ** d)) % n
            # Map bin to [-1, 1]
            force_vec[d] = (2 * bin_idx / (n - 1) - 1) if n > 1 else 0
        
        return force_vec
    
    def reset(self):
        """Reset any internal state."""
        pass


# =============================================================================
# ACTION MASKS
# =============================================================================

class ActionMask:
    """
    Mask for valid actions.
    
    Used to constrain action space based on current state
    (e.g., physical limits, safety constraints).
    """
    
    def __init__(
        self,
        action_config: ActionConfig,
        constraints: Optional[List[Dict[str, Any]]] = None,
    ):
        self.action_config = action_config
        self.constraints = constraints or []
    
    def compute(self, field: 'Field') -> np.ndarray:
        """
        Compute valid action mask.
        
        Args:
            field: Current field state
            
        Returns:
            Boolean mask for valid actions (for discrete)
            or scale factors (for continuous)
        """
        if self.action_config.action_type == ActionType.CONTINUOUS:
            # Scale factors for continuous actions
            mask = np.ones(self.action_config.actuators[0].force_dims * 
                          len(self.action_config.actuators), dtype=np.float32)
        else:
            # Boolean mask for discrete actions
            if isinstance(self.action_config.action_type, list):
                mask = np.ones(int(np.prod(self.action_config.n_discrete)), dtype=bool)
            else:
                mask = np.ones(int(self.action_config.n_discrete), dtype=bool)
        
        # Apply constraints
        for constraint in self.constraints:
            mask = self._apply_constraint(mask, field, constraint)
        
        return mask
    
    def _apply_constraint(
        self,
        mask: np.ndarray,
        field: 'Field',
        constraint: Dict[str, Any],
    ) -> np.ndarray:
        """Apply a single constraint to mask."""
        constraint_type = constraint.get('type', 'none')
        
        if constraint_type == 'energy_limit':
            # Limit actions when energy is high
            stats = field.stats()
            if stats.energy > constraint['threshold']:
                mask *= 0.5
        
        elif constraint_type == 'boundary_safety':
            # Disable actions near boundaries
            pass  # Implementation depends on specific setup
        
        return mask


# =============================================================================
# GYMNASIUM SPACE BUILDERS
# =============================================================================

def build_observation_space(config: ObservationConfig):
    """Build Gymnasium observation space from config."""
    obs = FieldObservation(config)
    shape = obs.shape
    
    try:
        import gymnasium
        return gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=shape,
            dtype=np.float32,
        )
    except ImportError:
        # Return a simple object with shape info
        class BoxSpace:
            def __init__(self):
                self.shape = shape
                self.low = -np.inf
                self.high = np.inf
                self.dtype = np.float32
        return BoxSpace()


def build_action_space(config: ActionConfig):
    """Build Gymnasium action space from config."""
    action = FieldAction(config)
    
    try:
        import gymnasium
        
        if config.action_type == ActionType.CONTINUOUS:
            low, high = action.bounds
            return gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        
        elif config.action_type == ActionType.DISCRETE:
            return gymnasium.spaces.Discrete(int(action.n_discrete))
        
        elif config.action_type == ActionType.MULTI_DISCRETE:
            return gymnasium.spaces.MultiDiscrete(action.n_discrete)
        
        else:
            # Hybrid: use Dict space
            return gymnasium.spaces.Dict({
                'continuous': gymnasium.spaces.Box(
                    low=-1, high=1,
                    shape=(len(config.actuators) * 3,),
                    dtype=np.float32,
                ),
                'discrete': gymnasium.spaces.MultiDiscrete(
                    [5] * len(config.actuators),
                ),
            })
    
    except ImportError:
        # Return simple space info
        low, high = action.bounds
        class BoxSpace:
            def __init__(self):
                self.shape = low.shape
                self.low = low
                self.high = high
                self.dtype = np.float32
        return BoxSpace()
