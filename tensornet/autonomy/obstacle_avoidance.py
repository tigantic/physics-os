"""
Obstacle Avoidance Module
=========================

Dynamic obstacle detection and avoidance for
autonomous tensor network operations.

Features:
- Multiple obstacle types
- Avoidance strategies
- Real-time replanning
- Safety zones
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
    List,
    Optional,
    Tuple,
)

import numpy as np


class ObstacleType(Enum):
    """Types of obstacles."""
    
    STATIC = auto()       # Fixed obstacle
    DYNAMIC = auto()      # Moving obstacle
    EXPANDING = auto()    # Growing obstacle
    TEMPORARY = auto()    # Time-limited
    VIRTUAL = auto()      # Soft constraint


class AvoidanceStrategy(Enum):
    """Avoidance strategies."""
    
    POTENTIAL_FIELD = auto()  # Repulsive field
    VELOCITY_OBSTACLE = auto()  # VO method
    REACTIVE = auto()          # Simple reactive
    PREDICTIVE = auto()        # Predict and plan
    HYBRID = auto()            # Combination


@dataclass
class Obstacle:
    """An obstacle to avoid.
    
    Attributes:
        obstacle_id: Unique identifier
        obstacle_type: Type of obstacle
        position: Center position
        radius: Obstacle radius
        velocity: Velocity if dynamic
        expansion_rate: Rate of expansion
        time_limit: Time until disappears
        priority: Avoidance priority
    """
    
    obstacle_id: int
    obstacle_type: ObstacleType = ObstacleType.STATIC
    position: Tuple[float, ...] = (0.0, 0.0)
    radius: float = 1.0
    velocity: Tuple[float, ...] = (0.0, 0.0)
    expansion_rate: float = 0.0
    time_limit: Optional[float] = None
    priority: int = 1
    created_time: float = field(default_factory=time.perf_counter)
    
    @property
    def age(self) -> float:
        """Age of obstacle."""
        return time.perf_counter() - self.created_time
    
    @property
    def is_expired(self) -> bool:
        """Check if obstacle has expired."""
        if self.time_limit is None:
            return False
        return self.age > self.time_limit
    
    def effective_radius(self, dt: float = 0.0) -> float:
        """Get effective radius at time offset.
        
        Args:
            dt: Time offset
            
        Returns:
            Effective radius
        """
        return self.radius + self.expansion_rate * dt
    
    def predicted_position(self, dt: float) -> Tuple[float, ...]:
        """Predict position at time offset.
        
        Args:
            dt: Time offset
            
        Returns:
            Predicted position
        """
        return tuple(
            p + v * dt for p, v in zip(self.position, self.velocity)
        )
    
    def distance_to(
        self,
        point: Tuple[float, ...],
        dt: float = 0.0,
    ) -> float:
        """Distance from point to obstacle surface.
        
        Args:
            point: Query point
            dt: Time offset
            
        Returns:
            Distance (negative if inside)
        """
        pred_pos = self.predicted_position(dt)
        center_dist = math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(point, pred_pos)
        ))
        return center_dist - self.effective_radius(dt)


@dataclass
class AvoidanceResult:
    """Result of avoidance computation.
    
    Attributes:
        success: Whether avoidance is possible
        avoidance_vector: Recommended direction
        avoidance_magnitude: How much to avoid
        closest_obstacle: Nearest obstacle
        safety_margin: Remaining safety margin
        computation_time: Time to compute
    """
    
    success: bool
    avoidance_vector: Tuple[float, ...] = (0.0, 0.0)
    avoidance_magnitude: float = 0.0
    closest_obstacle: Optional[Obstacle] = None
    safety_margin: float = float('inf')
    computation_time: float = 0.0


@dataclass
class ObstacleAvoidanceConfig:
    """Configuration for obstacle avoidance.
    
    Attributes:
        strategy: Avoidance strategy
        safety_margin: Minimum distance
        prediction_horizon: Time to predict
        repulsion_gain: Potential field gain
        max_avoidance: Maximum avoidance force
        update_rate: Update frequency
    """
    
    strategy: AvoidanceStrategy = AvoidanceStrategy.POTENTIAL_FIELD
    safety_margin: float = 1.0
    prediction_horizon: float = 2.0
    repulsion_gain: float = 10.0
    max_avoidance: float = 5.0
    update_rate: float = 10.0


class ObstacleAvoidance:
    """Obstacle avoidance system.
    
    Detects and computes avoidance maneuvers
    for dynamic obstacles.
    """
    
    def __init__(
        self,
        config: Optional[ObstacleAvoidanceConfig] = None,
    ) -> None:
        """Initialize avoidance system.
        
        Args:
            config: Configuration
        """
        self.config = config or ObstacleAvoidanceConfig()
        
        self.obstacles: Dict[int, Obstacle] = {}
        self._obstacle_counter = 0
        self._last_update = time.perf_counter()
    
    def add_obstacle(
        self,
        position: Tuple[float, ...],
        radius: float = 1.0,
        obstacle_type: ObstacleType = ObstacleType.STATIC,
        velocity: Optional[Tuple[float, ...]] = None,
        **kwargs,
    ) -> int:
        """Add obstacle to track.
        
        Args:
            position: Obstacle center
            radius: Obstacle radius
            obstacle_type: Type
            velocity: Velocity if dynamic
            **kwargs: Additional properties
            
        Returns:
            Obstacle ID
        """
        obstacle_id = self._obstacle_counter
        self._obstacle_counter += 1
        
        dim = len(position)
        if velocity is None:
            velocity = tuple(0.0 for _ in range(dim))
        
        obstacle = Obstacle(
            obstacle_id=obstacle_id,
            obstacle_type=obstacle_type,
            position=position,
            radius=radius,
            velocity=velocity,
            **kwargs,
        )
        
        self.obstacles[obstacle_id] = obstacle
        return obstacle_id
    
    def remove_obstacle(self, obstacle_id: int) -> bool:
        """Remove obstacle.
        
        Args:
            obstacle_id: Obstacle to remove
            
        Returns:
            Success
        """
        if obstacle_id in self.obstacles:
            del self.obstacles[obstacle_id]
            return True
        return False
    
    def update_obstacle(
        self,
        obstacle_id: int,
        position: Optional[Tuple[float, ...]] = None,
        velocity: Optional[Tuple[float, ...]] = None,
        radius: Optional[float] = None,
    ) -> bool:
        """Update obstacle state.
        
        Args:
            obstacle_id: Obstacle to update
            position: New position
            velocity: New velocity
            radius: New radius
            
        Returns:
            Success
        """
        if obstacle_id not in self.obstacles:
            return False
        
        obs = self.obstacles[obstacle_id]
        
        if position is not None:
            obs.position = position
        if velocity is not None:
            obs.velocity = velocity
        if radius is not None:
            obs.radius = radius
        
        return True
    
    def prune_expired(self) -> int:
        """Remove expired obstacles.
        
        Returns:
            Number removed
        """
        to_remove = [
            oid for oid, obs in self.obstacles.items()
            if obs.is_expired
        ]
        
        for oid in to_remove:
            del self.obstacles[oid]
        
        return len(to_remove)
    
    def compute_avoidance(
        self,
        position: Tuple[float, ...],
        velocity: Tuple[float, ...],
    ) -> AvoidanceResult:
        """Compute avoidance vector.
        
        Args:
            position: Current position
            velocity: Current velocity
            
        Returns:
            AvoidanceResult
        """
        start_time = time.perf_counter()
        
        strategy = self.config.strategy
        
        if strategy == AvoidanceStrategy.POTENTIAL_FIELD:
            result = self._potential_field_avoidance(position, velocity)
        elif strategy == AvoidanceStrategy.REACTIVE:
            result = self._reactive_avoidance(position, velocity)
        elif strategy == AvoidanceStrategy.PREDICTIVE:
            result = self._predictive_avoidance(position, velocity)
        else:
            result = self._potential_field_avoidance(position, velocity)
        
        result.computation_time = time.perf_counter() - start_time
        return result
    
    def _potential_field_avoidance(
        self,
        position: Tuple[float, ...],
        velocity: Tuple[float, ...],
    ) -> AvoidanceResult:
        """Potential field method.
        
        Args:
            position: Current position
            velocity: Current velocity
            
        Returns:
            AvoidanceResult
        """
        dim = len(position)
        total_force = [0.0] * dim
        closest_dist = float('inf')
        closest_obs = None
        
        for obs in self.obstacles.values():
            # Distance to obstacle
            diff = tuple(p - o for p, o in zip(position, obs.position))
            dist = math.sqrt(sum(d ** 2 for d in diff))
            
            surface_dist = dist - obs.radius - self.config.safety_margin
            
            if surface_dist < closest_dist:
                closest_dist = surface_dist
                closest_obs = obs
            
            if surface_dist < 0:
                # Inside danger zone - max repulsion
                if dist > 0:
                    direction = tuple(d / dist for d in diff)
                else:
                    direction = tuple(1.0 if i == 0 else 0.0 for i in range(dim))
                
                force_mag = self.config.max_avoidance
                for i in range(dim):
                    total_force[i] += direction[i] * force_mag
            
            elif surface_dist < self.config.safety_margin * 2:
                # Repulsion zone
                if dist > 0:
                    direction = tuple(d / dist for d in diff)
                    force_mag = self.config.repulsion_gain / (surface_dist ** 2 + 0.01)
                    force_mag = min(force_mag, self.config.max_avoidance)
                    
                    for i in range(dim):
                        total_force[i] += direction[i] * force_mag
        
        magnitude = math.sqrt(sum(f ** 2 for f in total_force))
        
        if magnitude > self.config.max_avoidance:
            total_force = tuple(f / magnitude * self.config.max_avoidance for f in total_force)
            magnitude = self.config.max_avoidance
        
        return AvoidanceResult(
            success=True,
            avoidance_vector=tuple(total_force),
            avoidance_magnitude=magnitude,
            closest_obstacle=closest_obs,
            safety_margin=closest_dist,
        )
    
    def _reactive_avoidance(
        self,
        position: Tuple[float, ...],
        velocity: Tuple[float, ...],
    ) -> AvoidanceResult:
        """Simple reactive avoidance.
        
        Args:
            position: Current position
            velocity: Current velocity
            
        Returns:
            AvoidanceResult
        """
        dim = len(position)
        
        # Find closest obstacle
        closest_dist = float('inf')
        closest_obs = None
        
        for obs in self.obstacles.values():
            dist = obs.distance_to(position)
            if dist < closest_dist:
                closest_dist = dist
                closest_obs = obs
        
        if closest_obs is None or closest_dist > self.config.safety_margin * 2:
            return AvoidanceResult(
                success=True,
                avoidance_vector=tuple(0.0 for _ in range(dim)),
                safety_margin=closest_dist,
            )
        
        # Compute avoidance direction
        diff = tuple(p - o for p, o in zip(position, closest_obs.position))
        dist = math.sqrt(sum(d ** 2 for d in diff)) or 1.0
        direction = tuple(d / dist for d in diff)
        
        # Scale by urgency
        urgency = max(0.0, 1.0 - closest_dist / (self.config.safety_margin * 2))
        magnitude = urgency * self.config.max_avoidance
        
        avoidance = tuple(d * magnitude for d in direction)
        
        return AvoidanceResult(
            success=True,
            avoidance_vector=avoidance,
            avoidance_magnitude=magnitude,
            closest_obstacle=closest_obs,
            safety_margin=closest_dist,
        )
    
    def _predictive_avoidance(
        self,
        position: Tuple[float, ...],
        velocity: Tuple[float, ...],
    ) -> AvoidanceResult:
        """Predictive avoidance.
        
        Args:
            position: Current position
            velocity: Current velocity
            
        Returns:
            AvoidanceResult
        """
        dim = len(position)
        dt = self.config.prediction_horizon
        
        # Predict future position
        future_pos = tuple(p + v * dt for p, v in zip(position, velocity))
        
        # Find potential collisions
        min_dist = float('inf')
        threat_obs = None
        
        for obs in self.obstacles.values():
            # Check along trajectory
            for t in np.linspace(0, dt, 10):
                test_pos = tuple(p + v * t for p, v in zip(position, velocity))
                dist = obs.distance_to(test_pos, t)
                
                if dist < min_dist:
                    min_dist = dist
                    threat_obs = obs
        
        if threat_obs is None or min_dist > self.config.safety_margin:
            return AvoidanceResult(
                success=True,
                avoidance_vector=tuple(0.0 for _ in range(dim)),
                safety_margin=min_dist,
            )
        
        # Compute perpendicular avoidance
        diff = tuple(p - o for p, o in zip(position, threat_obs.position))
        
        # Project perpendicular to velocity
        vel_mag = math.sqrt(sum(v ** 2 for v in velocity)) or 1.0
        vel_norm = tuple(v / vel_mag for v in velocity)
        
        # dot product
        dot = sum(d * v for d, v in zip(diff, vel_norm))
        
        # Perpendicular component
        perp = tuple(d - dot * v for d, v in zip(diff, vel_norm))
        perp_mag = math.sqrt(sum(p ** 2 for p in perp)) or 1.0
        perp_norm = tuple(p / perp_mag for p in perp)
        
        # Scale
        urgency = max(0.0, 1.0 - min_dist / self.config.safety_margin)
        magnitude = urgency * self.config.max_avoidance
        
        avoidance = tuple(p * magnitude for p in perp_norm)
        
        return AvoidanceResult(
            success=True,
            avoidance_vector=avoidance,
            avoidance_magnitude=magnitude,
            closest_obstacle=threat_obs,
            safety_margin=min_dist,
        )
    
    def is_path_clear(
        self,
        start: Tuple[float, ...],
        end: Tuple[float, ...],
        check_radius: float = 0.0,
    ) -> bool:
        """Check if path is clear of obstacles.
        
        Args:
            start: Start point
            end: End point
            check_radius: Radius to check
            
        Returns:
            Whether path is clear
        """
        # Sample points along path
        num_samples = max(10, int(math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(start, end)
        ))))
        
        for i in range(num_samples + 1):
            t = i / num_samples
            point = tuple(s + t * (e - s) for s, e in zip(start, end))
            
            for obs in self.obstacles.values():
                if obs.distance_to(point) < check_radius:
                    return False
        
        return True
    
    def get_safe_directions(
        self,
        position: Tuple[float, ...],
        num_directions: int = 8,
    ) -> List[Tuple[Tuple[float, ...], float]]:
        """Get safe movement directions.
        
        Args:
            position: Current position
            num_directions: Number of directions to check
            
        Returns:
            List of (direction, safety_score)
        """
        dim = len(position)
        directions = []
        
        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            
            if dim == 2:
                direction = (math.cos(angle), math.sin(angle))
            else:
                direction = tuple(
                    math.cos(angle) if j == 0 else (math.sin(angle) if j == 1 else 0.0)
                    for j in range(dim)
                )
            
            # Check safety along direction
            test_point = tuple(p + d * 5.0 for p, d in zip(position, direction))
            
            min_dist = float('inf')
            for obs in self.obstacles.values():
                dist = obs.distance_to(test_point)
                min_dist = min(min_dist, dist)
            
            safety = min_dist / (self.config.safety_margin + 1.0)
            directions.append((direction, safety))
        
        return sorted(directions, key=lambda x: -x[1])


def detect_obstacles(
    sensor_readings: List[Tuple[float, float]],
    position: Tuple[float, ...],
    default_radius: float = 1.0,
) -> List[Obstacle]:
    """Detect obstacles from sensor readings.
    
    Args:
        sensor_readings: List of (angle, distance)
        position: Current position
        default_radius: Default obstacle radius
        
    Returns:
        List of detected obstacles
    """
    obstacles = []
    
    for i, (angle, distance) in enumerate(sensor_readings):
        if distance < 0 or distance > 100:
            continue
        
        # Convert to position
        x = position[0] + distance * math.cos(angle)
        y = position[1] + distance * math.sin(angle)
        
        obstacles.append(Obstacle(
            obstacle_id=i,
            obstacle_type=ObstacleType.STATIC,
            position=(x, y),
            radius=default_radius,
        ))
    
    return obstacles


def compute_avoidance_vector(
    position: Tuple[float, ...],
    velocity: Tuple[float, ...],
    obstacles: List[Obstacle],
    strategy: AvoidanceStrategy = AvoidanceStrategy.POTENTIAL_FIELD,
) -> Tuple[float, ...]:
    """Compute avoidance vector.
    
    Args:
        position: Current position
        velocity: Current velocity
        obstacles: List of obstacles
        strategy: Avoidance strategy
        
    Returns:
        Avoidance vector
    """
    config = ObstacleAvoidanceConfig(strategy=strategy)
    avoider = ObstacleAvoidance(config)
    
    for obs in obstacles:
        avoider.obstacles[obs.obstacle_id] = obs
    
    result = avoider.compute_avoidance(position, velocity)
    return result.avoidance_vector
