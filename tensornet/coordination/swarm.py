# Copyright (c) 2025 Tigantic
# Phase 18: Swarm Coordination
"""
Swarm coordination for multi-vehicle systems.

Provides vehicle state management, swarm topology, and coordination
algorithms for distributed autonomous vehicle networks.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

import torch
import numpy as np


class TopologyType(Enum):
    """Swarm communication topology types."""
    
    FULLY_CONNECTED = auto()   # All-to-all communication
    RING = auto()              # Circular neighbor graph
    STAR = auto()              # Central leader topology
    MESH = auto()              # Grid-based topology
    TREE = auto()              # Hierarchical tree
    RANDOM = auto()            # Random k-regular graph
    DYNAMIC = auto()           # Proximity-based dynamic


@dataclass
class VehicleState:
    """State of an individual vehicle.
    
    Attributes:
        vehicle_id: Unique vehicle identifier
        position: 3D position [x, y, z]
        velocity: 3D velocity [vx, vy, vz]
        orientation: Quaternion [w, x, y, z]
        timestamp: State timestamp
        status: Vehicle status string
        battery: Battery level (0-1)
        payload: Optional payload data
    """
    
    vehicle_id: str
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    timestamp: float = field(default_factory=time.time)
    status: str = "active"
    battery: float = 1.0
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate and convert arrays."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        
        if self.position.shape != (3,):
            raise ValueError("position must be 3D")
        if self.velocity.shape != (3,):
            raise ValueError("velocity must be 3D")
        if self.orientation.shape != (4,):
            raise ValueError("orientation must be quaternion [w, x, y, z]")
    
    @property
    def speed(self) -> float:
        """Get scalar speed."""
        return float(np.linalg.norm(self.velocity))
    
    @property
    def heading(self) -> float:
        """Get heading angle in radians (yaw from quaternion)."""
        w, x, y, z = self.orientation
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    def distance_to(self, other: "VehicleState") -> float:
        """Compute Euclidean distance to another vehicle.
        
        Args:
            other: Other vehicle state
            
        Returns:
            Distance in meters
        """
        return float(np.linalg.norm(self.position - other.position))
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor representation.
        
        Returns:
            Tensor [position(3), velocity(3), orientation(4)]
        """
        state = np.concatenate([
            self.position,
            self.velocity,
            self.orientation,
        ])
        return torch.from_numpy(state).float()
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, vehicle_id: str) -> "VehicleState":
        """Create from tensor representation.
        
        Args:
            tensor: State tensor [10]
            vehicle_id: Vehicle ID
            
        Returns:
            VehicleState instance
        """
        # D-006 FIX: Use torch slicing directly, numpy only at boundary
        t = tensor.detach().cpu()
        return cls(
            vehicle_id=vehicle_id,
            position=t[:3].numpy(),
            velocity=t[3:6].numpy(),
            orientation=t[6:10].numpy(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vehicle_id": self.vehicle_id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "orientation": self.orientation.tolist(),
            "timestamp": self.timestamp,
            "status": self.status,
            "battery": self.battery,
            "payload": self.payload,
        }


@dataclass
class SwarmConfig:
    """Swarm coordination configuration.
    
    Attributes:
        topology: Communication topology type
        communication_range: Maximum communication distance (m)
        update_rate: State update rate (Hz)
        collision_radius: Minimum separation distance (m)
        max_speed: Maximum vehicle speed (m/s)
        max_acceleration: Maximum acceleration (m/s²)
        convergence_threshold: Formation convergence threshold (m)
    """
    
    topology: TopologyType = TopologyType.FULLY_CONNECTED
    communication_range: float = 100.0
    update_rate: float = 10.0
    collision_radius: float = 2.0
    max_speed: float = 20.0
    max_acceleration: float = 5.0
    convergence_threshold: float = 0.5
    
    def validate(self) -> bool:
        """Validate configuration.
        
        Returns:
            True if valid
        """
        if self.communication_range <= 0:
            return False
        if self.update_rate <= 0:
            return False
        if self.collision_radius <= 0:
            return False
        if self.max_speed <= 0:
            return False
        if self.max_acceleration <= 0:
            return False
        return True


class SwarmTopology:
    """Communication topology for swarm.
    
    Manages the communication graph between vehicles,
    determining which vehicles can exchange information.
    
    Attributes:
        topology_type: Type of topology
        config: Swarm configuration
    """
    
    def __init__(
        self,
        topology_type: TopologyType = TopologyType.FULLY_CONNECTED,
        config: Optional[SwarmConfig] = None,
    ) -> None:
        """Initialize topology.
        
        Args:
            topology_type: Type of topology
            config: Swarm configuration
        """
        self.topology_type = topology_type
        self.config = config or SwarmConfig(topology=topology_type)
        
        self._adjacency: Dict[str, Set[str]] = {}
        self._vehicle_ids: List[str] = []
    
    def build(self, vehicle_ids: List[str]) -> None:
        """Build topology for given vehicles.
        
        Args:
            vehicle_ids: List of vehicle IDs
        """
        self._vehicle_ids = list(vehicle_ids)
        n = len(vehicle_ids)
        
        self._adjacency = {vid: set() for vid in vehicle_ids}
        
        if self.topology_type == TopologyType.FULLY_CONNECTED:
            for i, vid in enumerate(vehicle_ids):
                self._adjacency[vid] = set(vehicle_ids) - {vid}
        
        elif self.topology_type == TopologyType.RING:
            for i, vid in enumerate(vehicle_ids):
                prev_idx = (i - 1) % n
                next_idx = (i + 1) % n
                self._adjacency[vid] = {
                    vehicle_ids[prev_idx],
                    vehicle_ids[next_idx],
                }
        
        elif self.topology_type == TopologyType.STAR:
            if n > 0:
                leader = vehicle_ids[0]
                self._adjacency[leader] = set(vehicle_ids[1:])
                for vid in vehicle_ids[1:]:
                    self._adjacency[vid] = {leader}
        
        elif self.topology_type == TopologyType.MESH:
            # Create grid-like connections
            side = int(math.ceil(math.sqrt(n)))
            for i, vid in enumerate(vehicle_ids):
                row, col = i // side, i % side
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    ni = nr * side + nc
                    if 0 <= nr < side and 0 <= nc < side and 0 <= ni < n:
                        neighbors.append(vehicle_ids[ni])
                self._adjacency[vid] = set(neighbors)
    
    def update_dynamic(self, states: Dict[str, VehicleState]) -> None:
        """Update dynamic topology based on proximity.
        
        Args:
            states: Current vehicle states
        """
        if self.topology_type != TopologyType.DYNAMIC:
            return
        
        vehicle_ids = list(states.keys())
        self._vehicle_ids = vehicle_ids
        self._adjacency = {vid: set() for vid in vehicle_ids}
        
        for i, vid1 in enumerate(vehicle_ids):
            for vid2 in vehicle_ids[i+1:]:
                dist = states[vid1].distance_to(states[vid2])
                if dist <= self.config.communication_range:
                    self._adjacency[vid1].add(vid2)
                    self._adjacency[vid2].add(vid1)
    
    def get_neighbors(self, vehicle_id: str) -> Set[str]:
        """Get neighbors of a vehicle.
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            Set of neighbor vehicle IDs
        """
        return self._adjacency.get(vehicle_id, set())
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix.
        
        Returns:
            NxN adjacency matrix
        """
        n = len(self._vehicle_ids)
        matrix = np.zeros((n, n))
        
        vid_to_idx = {vid: i for i, vid in enumerate(self._vehicle_ids)}
        
        for vid, neighbors in self._adjacency.items():
            i = vid_to_idx[vid]
            for neighbor in neighbors:
                j = vid_to_idx[neighbor]
                matrix[i, j] = 1
        
        return matrix
    
    def get_laplacian_matrix(self) -> np.ndarray:
        """Get graph Laplacian matrix.
        
        Returns:
            NxN Laplacian matrix (D - A)
        """
        adj = self.get_adjacency_matrix()
        degree = np.diag(adj.sum(axis=1))
        return degree - adj
    
    def is_connected(self) -> bool:
        """Check if graph is connected.
        
        Returns:
            True if all vehicles can communicate
        """
        if not self._vehicle_ids:
            return True
        
        visited = set()
        stack = [self._vehicle_ids[0]]
        
        while stack:
            vid = stack.pop()
            if vid in visited:
                continue
            visited.add(vid)
            stack.extend(self._adjacency.get(vid, set()) - visited)
        
        return len(visited) == len(self._vehicle_ids)


class SwarmCoordinator:
    """Main swarm coordination engine.
    
    Coordinates multiple vehicles using distributed algorithms
    for formation control, collision avoidance, and consensus.
    
    Attributes:
        config: Swarm configuration
        topology: Communication topology
    """
    
    def __init__(
        self,
        config: Optional[SwarmConfig] = None,
        topology: Optional[SwarmTopology] = None,
    ) -> None:
        """Initialize swarm coordinator.
        
        Args:
            config: Swarm configuration
            topology: Communication topology
        """
        self.config = config or SwarmConfig()
        self.topology = topology or SwarmTopology(
            self.config.topology,
            self.config,
        )
        
        self._states: Dict[str, VehicleState] = {}
        self._history: List[Dict[str, VehicleState]] = []
        self._commands: Dict[str, np.ndarray] = {}
    
    @property
    def num_vehicles(self) -> int:
        """Get number of vehicles."""
        return len(self._states)
    
    @property
    def vehicle_ids(self) -> List[str]:
        """Get list of vehicle IDs."""
        return list(self._states.keys())
    
    def register_vehicle(self, state: VehicleState) -> None:
        """Register a vehicle with the swarm.
        
        Args:
            state: Initial vehicle state
        """
        self._states[state.vehicle_id] = state
        self.topology.build(list(self._states.keys()))
    
    def unregister_vehicle(self, vehicle_id: str) -> None:
        """Remove a vehicle from the swarm.
        
        Args:
            vehicle_id: Vehicle to remove
        """
        if vehicle_id in self._states:
            del self._states[vehicle_id]
            self.topology.build(list(self._states.keys()))
    
    def update_state(self, state: VehicleState) -> None:
        """Update vehicle state.
        
        Args:
            state: New vehicle state
        """
        if state.vehicle_id not in self._states:
            raise ValueError(f"Unknown vehicle: {state.vehicle_id}")
        
        self._states[state.vehicle_id] = state
        
        if self.config.topology == TopologyType.DYNAMIC:
            self.topology.update_dynamic(self._states)
    
    def get_state(self, vehicle_id: str) -> Optional[VehicleState]:
        """Get vehicle state.
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            VehicleState or None
        """
        return self._states.get(vehicle_id)
    
    def get_all_states(self) -> Dict[str, VehicleState]:
        """Get all vehicle states.
        
        Returns:
            Dict mapping vehicle IDs to states
        """
        return dict(self._states)
    
    def compute_control(
        self,
        target_positions: Dict[str, np.ndarray],
        gain_p: float = 2.0,
        gain_d: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """Compute control commands to reach targets.
        
        Uses PD control with collision avoidance.
        
        Args:
            target_positions: Target positions per vehicle
            gain_p: Proportional gain
            gain_d: Derivative gain
            
        Returns:
            Acceleration commands per vehicle
        """
        commands = {}
        
        for vid, state in self._states.items():
            if vid not in target_positions:
                commands[vid] = np.zeros(3)
                continue
            
            target = target_positions[vid]
            
            # PD control
            pos_error = target - state.position
            vel_desired = gain_p * pos_error
            
            # Limit velocity
            speed = np.linalg.norm(vel_desired)
            if speed > self.config.max_speed:
                vel_desired = vel_desired * self.config.max_speed / speed
            
            vel_error = vel_desired - state.velocity
            acceleration = gain_d * vel_error
            
            # Add collision avoidance
            avoidance = self._compute_collision_avoidance(vid)
            acceleration = acceleration + avoidance
            
            # Limit acceleration
            acc_mag = np.linalg.norm(acceleration)
            if acc_mag > self.config.max_acceleration:
                acceleration = acceleration * self.config.max_acceleration / acc_mag
            
            commands[vid] = acceleration
        
        self._commands = commands
        return commands
    
    def _compute_collision_avoidance(self, vehicle_id: str) -> np.ndarray:
        """Compute collision avoidance acceleration.
        
        Args:
            vehicle_id: Vehicle to compute avoidance for
            
        Returns:
            Avoidance acceleration vector
        """
        state = self._states[vehicle_id]
        avoidance = np.zeros(3)
        
        for vid, other_state in self._states.items():
            if vid == vehicle_id:
                continue
            
            diff = state.position - other_state.position
            dist = np.linalg.norm(diff)
            
            if dist < self.config.collision_radius * 2:
                # Repulsive force inversely proportional to distance
                if dist > 0:
                    strength = (self.config.collision_radius * 2 - dist) / dist
                    avoidance += strength * diff / dist
        
        return avoidance
    
    def step(self, dt: float) -> None:
        """Advance simulation by dt seconds.
        
        Args:
            dt: Time step in seconds
        """
        # Record history
        self._history.append({
            vid: VehicleState(
                vehicle_id=s.vehicle_id,
                position=s.position.copy(),
                velocity=s.velocity.copy(),
                orientation=s.orientation.copy(),
                timestamp=s.timestamp,
                status=s.status,
                battery=s.battery,
            )
            for vid, s in self._states.items()
        })
        
        # Integrate dynamics
        for vid, state in self._states.items():
            acc = self._commands.get(vid, np.zeros(3))
            
            # Simple Euler integration
            new_velocity = state.velocity + acc * dt
            
            # Limit velocity
            speed = np.linalg.norm(new_velocity)
            if speed > self.config.max_speed:
                new_velocity = new_velocity * self.config.max_speed / speed
            
            new_position = state.position + new_velocity * dt
            
            state.velocity = new_velocity
            state.position = new_position
            state.timestamp = time.time()
    
    def check_convergence(
        self,
        target_positions: Dict[str, np.ndarray],
    ) -> bool:
        """Check if swarm has converged to targets.
        
        Args:
            target_positions: Target positions
            
        Returns:
            True if all vehicles within threshold
        """
        for vid, state in self._states.items():
            if vid not in target_positions:
                continue
            
            error = np.linalg.norm(state.position - target_positions[vid])
            if error > self.config.convergence_threshold:
                return False
        
        return True
    
    def get_metrics(self) -> Dict[str, float]:
        """Get swarm metrics.
        
        Returns:
            Dict of metric values
        """
        if not self._states:
            return {}
        
        positions = np.array([s.position for s in self._states.values()])
        velocities = np.array([s.velocity for s in self._states.values()])
        
        centroid = positions.mean(axis=0)
        spread = np.sqrt(np.mean(np.sum((positions - centroid) ** 2, axis=1)))
        
        mean_speed = np.mean(np.linalg.norm(velocities, axis=1))
        
        # Compute minimum pairwise distance
        min_dist = float('inf')
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, dist)
        
        return {
            "num_vehicles": len(self._states),
            "centroid_x": centroid[0],
            "centroid_y": centroid[1],
            "centroid_z": centroid[2],
            "spread": spread,
            "mean_speed": mean_speed,
            "min_pairwise_distance": min_dist if min_dist != float('inf') else 0.0,
            "topology_connected": self.topology.is_connected(),
        }


def compute_swarm_centroid(states: Dict[str, VehicleState]) -> np.ndarray:
    """Compute swarm centroid position.
    
    Args:
        states: Vehicle states
        
    Returns:
        3D centroid position
    """
    if not states:
        return np.zeros(3)
    
    positions = np.array([s.position for s in states.values()])
    return positions.mean(axis=0)


def compute_swarm_spread(states: Dict[str, VehicleState]) -> float:
    """Compute swarm spread (RMS distance from centroid).
    
    Args:
        states: Vehicle states
        
    Returns:
        Spread in meters
    """
    if not states:
        return 0.0
    
    positions = np.array([s.position for s in states.values()])
    centroid = positions.mean(axis=0)
    
    return float(np.sqrt(np.mean(np.sum((positions - centroid) ** 2, axis=1))))
