"""
Path Planning Module
====================

Optimal path planning for tensor network operations
and swarm navigation.

Algorithms:
- A* search
- RRT (Rapidly-exploring Random Trees)
- Dijkstra's algorithm
- Potential field methods
"""

from __future__ import annotations

import math
import heapq
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Set,
)

import numpy as np


class PlanningAlgorithm(Enum):
    """Path planning algorithm."""
    
    A_STAR = auto()
    DIJKSTRA = auto()
    RRT = auto()
    RRT_STAR = auto()
    POTENTIAL_FIELD = auto()
    GREEDY = auto()


@dataclass
class Waypoint:
    """A waypoint in a path.
    
    Attributes:
        position: (x, y, z) or (x, y) coordinates
        velocity: Optional velocity at waypoint
        heading: Optional heading angle
        time: Optional time to reach
        properties: Additional properties
    """
    
    position: Tuple[float, ...]
    velocity: Optional[Tuple[float, ...]] = None
    heading: Optional[float] = None
    time: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def x(self) -> float:
        return self.position[0]
    
    @property
    def y(self) -> float:
        return self.position[1] if len(self.position) > 1 else 0.0
    
    @property
    def z(self) -> float:
        return self.position[2] if len(self.position) > 2 else 0.0
    
    def distance_to(self, other: Waypoint) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum(
            (a - b) ** 2 
            for a, b in zip(self.position, other.position)
        ))


@dataclass
class Path:
    """A complete path.
    
    Attributes:
        waypoints: List of waypoints
        total_distance: Total path length
        total_time: Estimated total time
        algorithm: Algorithm used
        is_valid: Whether path is valid
        cost: Path cost metric
    """
    
    waypoints: List[Waypoint] = field(default_factory=list)
    total_distance: float = 0.0
    total_time: float = 0.0
    algorithm: PlanningAlgorithm = PlanningAlgorithm.A_STAR
    is_valid: bool = True
    cost: float = 0.0
    
    def __len__(self) -> int:
        return len(self.waypoints)
    
    def add_waypoint(self, waypoint: Waypoint) -> None:
        """Add waypoint and update distance."""
        if self.waypoints:
            self.total_distance += self.waypoints[-1].distance_to(waypoint)
        self.waypoints.append(waypoint)
    
    def reverse(self) -> Path:
        """Return reversed path."""
        return Path(
            waypoints=list(reversed(self.waypoints)),
            total_distance=self.total_distance,
            total_time=self.total_time,
            algorithm=self.algorithm,
            is_valid=self.is_valid,
            cost=self.cost,
        )
    
    def subsection(self, start: int, end: int) -> Path:
        """Get subsection of path."""
        sub_waypoints = self.waypoints[start:end]
        
        # Calculate distance
        distance = 0.0
        for i in range(1, len(sub_waypoints)):
            distance += sub_waypoints[i-1].distance_to(sub_waypoints[i])
        
        return Path(
            waypoints=sub_waypoints,
            total_distance=distance,
            algorithm=self.algorithm,
            is_valid=self.is_valid,
        )


@dataclass
class PathPlannerConfig:
    """Configuration for path planner.
    
    Attributes:
        algorithm: Planning algorithm
        grid_resolution: Resolution for grid-based
        max_iterations: Max iterations
        goal_threshold: Distance to consider reached
        step_size: Step size for RRT
        neighbor_radius: Radius for neighbors
    """
    
    algorithm: PlanningAlgorithm = PlanningAlgorithm.A_STAR
    grid_resolution: float = 1.0
    max_iterations: int = 10000
    goal_threshold: float = 0.5
    step_size: float = 1.0
    neighbor_radius: float = 2.0
    heuristic_weight: float = 1.0


class GridNode:
    """Node for grid-based planning."""
    
    def __init__(
        self,
        x: int,
        y: int,
        g: float = float('inf'),
        h: float = 0.0,
        parent: Optional[GridNode] = None,
    ) -> None:
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.parent = parent
    
    @property
    def f(self) -> float:
        return self.g + self.h
    
    def __lt__(self, other: GridNode) -> bool:
        return self.f < other.f
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridNode):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))


class PathPlanner:
    """Path planner for navigation.
    
    Supports multiple algorithms for finding
    optimal paths through environments.
    """
    
    def __init__(
        self,
        config: Optional[PathPlannerConfig] = None,
    ) -> None:
        """Initialize planner.
        
        Args:
            config: Configuration
        """
        self.config = config or PathPlannerConfig()
        
        # Obstacle map (set of blocked coordinates)
        self.obstacles: Set[Tuple[int, int]] = set()
        self.bounds: Tuple[int, int, int, int] = (0, 0, 100, 100)
    
    def set_bounds(
        self,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
    ) -> None:
        """Set planning bounds.
        
        Args:
            x_min, y_min: Minimum coordinates
            x_max, y_max: Maximum coordinates
        """
        self.bounds = (x_min, y_min, x_max, y_max)
    
    def add_obstacle(self, x: int, y: int) -> None:
        """Add obstacle at coordinate."""
        self.obstacles.add((x, y))
    
    def add_obstacles(self, obstacles: List[Tuple[int, int]]) -> None:
        """Add multiple obstacles."""
        self.obstacles.update(obstacles)
    
    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self.obstacles.clear()
    
    def is_valid(self, x: int, y: int) -> bool:
        """Check if coordinate is valid.
        
        Args:
            x, y: Coordinates
            
        Returns:
            Whether valid
        """
        x_min, y_min, x_max, y_max = self.bounds
        
        if x < x_min or x >= x_max or y < y_min or y >= y_max:
            return False
        
        return (x, y) not in self.obstacles
    
    def heuristic(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> float:
        """Calculate heuristic distance.
        
        Args:
            x1, y1: Start point
            x2, y2: End point
            
        Returns:
            Estimated distance
        """
        # Euclidean distance
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def get_neighbors(
        self,
        x: int,
        y: int,
    ) -> List[Tuple[int, int, float]]:
        """Get valid neighbors.
        
        Args:
            x, y: Current position
            
        Returns:
            List of (x, y, cost)
        """
        neighbors = []
        
        # 8-connected grid
        directions = [
            (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
        ]
        
        for dx, dy, cost in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny, cost))
        
        return neighbors
    
    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Path:
        """Plan path from start to goal.
        
        Args:
            start: Start position
            goal: Goal position
            
        Returns:
            Planned path
        """
        algo = self.config.algorithm
        
        if algo == PlanningAlgorithm.A_STAR:
            return self._plan_astar(start, goal)
        elif algo == PlanningAlgorithm.DIJKSTRA:
            return self._plan_dijkstra(start, goal)
        elif algo == PlanningAlgorithm.RRT:
            return self._plan_rrt(start, goal)
        elif algo == PlanningAlgorithm.GREEDY:
            return self._plan_greedy(start, goal)
        else:
            return self._plan_astar(start, goal)
    
    def _plan_astar(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Path:
        """A* pathfinding.
        
        Args:
            start, goal: Start and goal
            
        Returns:
            Path
        """
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        
        if not self.is_valid(sx, sy) or not self.is_valid(gx, gy):
            return Path(is_valid=False)
        
        start_node = GridNode(sx, sy, g=0.0, h=self.heuristic(sx, sy, gx, gy))
        
        open_set: List[GridNode] = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        node_map: Dict[Tuple[int, int], GridNode] = {(sx, sy): start_node}
        
        iterations = 0
        
        while open_set and iterations < self.config.max_iterations:
            iterations += 1
            
            current = heapq.heappop(open_set)
            
            if current.x == gx and current.y == gy:
                return self._reconstruct_path(current, PlanningAlgorithm.A_STAR)
            
            closed_set.add((current.x, current.y))
            
            for nx, ny, cost in self.get_neighbors(current.x, current.y):
                if (nx, ny) in closed_set:
                    continue
                
                tentative_g = current.g + cost
                
                if (nx, ny) in node_map:
                    neighbor = node_map[(nx, ny)]
                    if tentative_g < neighbor.g:
                        neighbor.g = tentative_g
                        neighbor.parent = current
                        heapq.heapify(open_set)
                else:
                    h = self.heuristic(nx, ny, gx, gy) * self.config.heuristic_weight
                    neighbor = GridNode(nx, ny, g=tentative_g, h=h, parent=current)
                    node_map[(nx, ny)] = neighbor
                    heapq.heappush(open_set, neighbor)
        
        return Path(is_valid=False)
    
    def _plan_dijkstra(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Path:
        """Dijkstra's algorithm (A* with h=0)."""
        original_weight = self.config.heuristic_weight
        self.config.heuristic_weight = 0.0
        path = self._plan_astar(start, goal)
        self.config.heuristic_weight = original_weight
        path.algorithm = PlanningAlgorithm.DIJKSTRA
        return path
    
    def _plan_rrt(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Path:
        """RRT planning.
        
        Args:
            start, goal: Start and goal
            
        Returns:
            Path
        """
        class RRTNode:
            def __init__(self, x: float, y: float, parent=None):
                self.x = x
                self.y = y
                self.parent = parent
        
        x_min, y_min, x_max, y_max = self.bounds
        
        nodes = [RRTNode(start[0], start[1])]
        
        for _ in range(self.config.max_iterations):
            # Sample random point
            if random.random() < 0.1:
                rx, ry = goal[0], goal[1]
            else:
                rx = random.uniform(x_min, x_max)
                ry = random.uniform(y_min, y_max)
            
            # Find nearest node
            nearest = min(
                nodes,
                key=lambda n: (n.x - rx) ** 2 + (n.y - ry) ** 2
            )
            
            # Steer towards sample
            dist = math.sqrt((rx - nearest.x) ** 2 + (ry - nearest.y) ** 2)
            if dist > self.config.step_size:
                theta = math.atan2(ry - nearest.y, rx - nearest.x)
                rx = nearest.x + self.config.step_size * math.cos(theta)
                ry = nearest.y + self.config.step_size * math.sin(theta)
            
            # Check validity
            if not self.is_valid(int(rx), int(ry)):
                continue
            
            # Add new node
            new_node = RRTNode(rx, ry, parent=nearest)
            nodes.append(new_node)
            
            # Check if goal reached
            if math.sqrt((rx - goal[0]) ** 2 + (ry - goal[1]) ** 2) < self.config.goal_threshold:
                # Reconstruct path
                path = Path(algorithm=PlanningAlgorithm.RRT, is_valid=True)
                current = new_node
                
                waypoints = []
                while current:
                    waypoints.append(Waypoint(position=(current.x, current.y)))
                    current = current.parent
                
                path.waypoints = list(reversed(waypoints))
                
                # Calculate distance
                for i in range(1, len(path.waypoints)):
                    path.total_distance += path.waypoints[i-1].distance_to(
                        path.waypoints[i]
                    )
                
                return path
        
        return Path(is_valid=False, algorithm=PlanningAlgorithm.RRT)
    
    def _plan_greedy(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Path:
        """Greedy best-first search."""
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        
        path = Path(algorithm=PlanningAlgorithm.GREEDY)
        path.add_waypoint(Waypoint(position=start))
        
        x, y = sx, sy
        visited = {(x, y)}
        
        for _ in range(self.config.max_iterations):
            if x == gx and y == gy:
                path.is_valid = True
                return path
            
            # Find best neighbor
            neighbors = self.get_neighbors(x, y)
            best = None
            best_dist = float('inf')
            
            for nx, ny, _ in neighbors:
                if (nx, ny) in visited:
                    continue
                
                dist = self.heuristic(nx, ny, gx, gy)
                if dist < best_dist:
                    best_dist = dist
                    best = (nx, ny)
            
            if best is None:
                path.is_valid = False
                return path
            
            x, y = best
            visited.add((x, y))
            path.add_waypoint(Waypoint(position=(float(x), float(y))))
        
        path.is_valid = False
        return path
    
    def _reconstruct_path(
        self,
        goal_node: GridNode,
        algorithm: PlanningAlgorithm,
    ) -> Path:
        """Reconstruct path from goal node.
        
        Args:
            goal_node: Goal node with parent chain
            algorithm: Algorithm used
            
        Returns:
            Path
        """
        path = Path(algorithm=algorithm, is_valid=True, cost=goal_node.g)
        
        waypoints = []
        current: Optional[GridNode] = goal_node
        
        while current:
            waypoints.append(Waypoint(
                position=(float(current.x), float(current.y))
            ))
            current = current.parent
        
        path.waypoints = list(reversed(waypoints))
        path.total_distance = goal_node.g
        
        return path


def plan_path(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    obstacles: Optional[List[Tuple[int, int]]] = None,
    algorithm: PlanningAlgorithm = PlanningAlgorithm.A_STAR,
) -> Path:
    """Convenience function for path planning.
    
    Args:
        start: Start position
        goal: Goal position
        obstacles: List of obstacle coordinates
        algorithm: Planning algorithm
        
    Returns:
        Planned path
    """
    config = PathPlannerConfig(algorithm=algorithm)
    planner = PathPlanner(config)
    
    if obstacles:
        planner.add_obstacles(obstacles)
    
    return planner.plan(start, goal)


def smooth_path(
    path: Path,
    smoothing_factor: float = 0.5,
    iterations: int = 10,
) -> Path:
    """Smooth a path using gradient descent.
    
    Args:
        path: Input path
        smoothing_factor: How much to smooth
        iterations: Number of iterations
        
    Returns:
        Smoothed path
    """
    if len(path.waypoints) < 3:
        return path
    
    # Convert to numpy
    positions = np.array([w.position for w in path.waypoints])
    smoothed = positions.copy()
    
    for _ in range(iterations):
        for i in range(1, len(smoothed) - 1):
            # Move towards midpoint of neighbors
            mid = (smoothed[i - 1] + smoothed[i + 1]) / 2
            smoothed[i] = (
                smoothed[i] * (1 - smoothing_factor) +
                mid * smoothing_factor
            )
    
    # Create new path
    new_path = Path(algorithm=path.algorithm, is_valid=path.is_valid)
    for pos in smoothed:
        new_path.add_waypoint(Waypoint(position=tuple(pos)))
    
    return new_path
