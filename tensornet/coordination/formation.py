# Copyright (c) 2025 Tigantic
# Phase 18: Formation Control
"""
Formation control for multi-vehicle systems.

Provides formation types, controllers, and algorithms for
maintaining geometric formations in autonomous vehicle swarms.
"""

from __future__ import annotations

import math
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

import torch
import numpy as np

from tensornet.coordination.swarm import VehicleState


class FormationType(Enum):
    """Standard formation patterns."""
    
    LINE = auto()        # Single file line
    COLUMN = auto()      # Side-by-side columns
    WEDGE = auto()       # V-formation
    DIAMOND = auto()     # Diamond pattern
    CIRCLE = auto()      # Circular formation
    GRID = auto()        # Regular grid
    CUSTOM = auto()      # User-defined positions


@dataclass
class FormationConfig:
    """Formation configuration.
    
    Attributes:
        formation_type: Type of formation
        spacing: Inter-vehicle spacing (m)
        heading: Formation heading (radians)
        altitude: Formation altitude (m)
        scale: Formation scale factor
        leader_id: ID of formation leader
        custom_offsets: Custom position offsets for CUSTOM type
    """
    
    formation_type: FormationType = FormationType.WEDGE
    spacing: float = 10.0
    heading: float = 0.0
    altitude: float = 100.0
    scale: float = 1.0
    leader_id: Optional[str] = None
    custom_offsets: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.spacing <= 0:
            raise ValueError("spacing must be positive")
        if self.scale <= 0:
            raise ValueError("scale must be positive")


@dataclass
class FormationState:
    """Current formation state.
    
    Attributes:
        target_positions: Target positions per vehicle
        current_positions: Current positions per vehicle
        errors: Position errors per vehicle
        converged: Whether formation has converged
        metrics: Formation quality metrics
    """
    
    target_positions: Dict[str, np.ndarray]
    current_positions: Dict[str, np.ndarray]
    errors: Dict[str, float] = field(default_factory=dict)
    converged: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def from_states(
        cls,
        states: Dict[str, VehicleState],
        targets: Dict[str, np.ndarray],
        threshold: float = 0.5,
    ) -> "FormationState":
        """Create FormationState from vehicle states.
        
        Args:
            states: Current vehicle states
            targets: Target positions
            threshold: Convergence threshold
            
        Returns:
            FormationState instance
        """
        current = {vid: s.position.copy() for vid, s in states.items()}
        
        errors = {}
        for vid, target in targets.items():
            if vid in current:
                errors[vid] = float(np.linalg.norm(current[vid] - target))
        
        converged = all(e <= threshold for e in errors.values())
        
        # Compute metrics
        if errors:
            mean_error = np.mean(list(errors.values()))
            max_error = max(errors.values())
        else:
            mean_error = 0.0
            max_error = 0.0
        
        return cls(
            target_positions=targets,
            current_positions=current,
            errors=errors,
            converged=converged,
            metrics={
                "mean_error": mean_error,
                "max_error": max_error,
                "convergence_fraction": sum(1 for e in errors.values() if e <= threshold) / max(len(errors), 1),
            },
        )


class FormationController:
    """Formation controller for vehicle swarms.
    
    Computes and maintains geometric formations by generating
    target positions and control commands.
    
    Attributes:
        config: Formation configuration
    """
    
    def __init__(self, config: Optional[FormationConfig] = None) -> None:
        """Initialize formation controller.
        
        Args:
            config: Formation configuration
        """
        self.config = config or FormationConfig()
        self._slot_assignments: Dict[str, int] = {}
    
    def assign_slots(self, vehicle_ids: List[str]) -> Dict[str, int]:
        """Assign vehicles to formation slots.
        
        Args:
            vehicle_ids: List of vehicle IDs
            
        Returns:
            Mapping from vehicle ID to slot index
        """
        # Simple sequential assignment
        # Could use Hungarian algorithm for optimal assignment
        self._slot_assignments = {vid: i for i, vid in enumerate(vehicle_ids)}
        
        if self.config.leader_id and self.config.leader_id in vehicle_ids:
            # Move leader to slot 0
            leader_slot = self._slot_assignments[self.config.leader_id]
            for vid, slot in self._slot_assignments.items():
                if slot == 0:
                    self._slot_assignments[vid] = leader_slot
                    break
            self._slot_assignments[self.config.leader_id] = 0
        
        return self._slot_assignments
    
    def compute_formation_positions(
        self,
        center: np.ndarray,
        num_vehicles: int,
    ) -> List[np.ndarray]:
        """Compute formation positions relative to center.
        
        Args:
            center: Formation center position
            num_vehicles: Number of vehicles
            
        Returns:
            List of target positions
        """
        positions = []
        spacing = self.config.spacing * self.config.scale
        heading = self.config.heading
        
        # Rotation matrix for heading
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        
        def rotate(x: float, y: float) -> Tuple[float, float]:
            return x * cos_h - y * sin_h, x * sin_h + y * cos_h
        
        if self.config.formation_type == FormationType.LINE:
            # Single file line along heading
            for i in range(num_vehicles):
                offset = (i - num_vehicles // 2) * spacing
                dx, dy = rotate(offset, 0)
                positions.append(center + np.array([dx, dy, 0]))
        
        elif self.config.formation_type == FormationType.COLUMN:
            # Side by side
            for i in range(num_vehicles):
                offset = (i - num_vehicles // 2) * spacing
                dx, dy = rotate(0, offset)
                positions.append(center + np.array([dx, dy, 0]))
        
        elif self.config.formation_type == FormationType.WEDGE:
            # V-formation with leader at front
            positions.append(center.copy())  # Leader at center
            
            for i in range(1, num_vehicles):
                side = 1 if i % 2 == 1 else -1
                row = (i + 1) // 2
                
                x_offset = -row * spacing * 0.7
                y_offset = side * row * spacing * 0.7
                
                dx, dy = rotate(x_offset, y_offset)
                positions.append(center + np.array([dx, dy, 0]))
        
        elif self.config.formation_type == FormationType.DIAMOND:
            # Diamond pattern
            if num_vehicles >= 1:
                positions.append(center + np.array([spacing, 0, 0]))  # Front
            if num_vehicles >= 2:
                positions.append(center + np.array([-spacing, 0, 0]))  # Back
            if num_vehicles >= 3:
                positions.append(center + np.array([0, spacing, 0]))  # Right
            if num_vehicles >= 4:
                positions.append(center + np.array([0, -spacing, 0]))  # Left
            
            # Additional vehicles in concentric diamonds
            for i in range(4, num_vehicles):
                ring = (i - 4) // 4 + 2
                angle = ((i - 4) % 4 + 0.5) * math.pi / 2
                dx, dy = rotate(ring * spacing * math.cos(angle), ring * spacing * math.sin(angle))
                positions.append(center + np.array([dx, dy, 0]))
        
        elif self.config.formation_type == FormationType.CIRCLE:
            # Circular formation
            for i in range(num_vehicles):
                angle = 2 * math.pi * i / num_vehicles
                radius = spacing * num_vehicles / (2 * math.pi)
                dx, dy = rotate(radius * math.cos(angle), radius * math.sin(angle))
                positions.append(center + np.array([dx, dy, 0]))
        
        elif self.config.formation_type == FormationType.GRID:
            # Regular grid
            side = int(math.ceil(math.sqrt(num_vehicles)))
            for i in range(num_vehicles):
                row = i // side
                col = i % side
                x_offset = (col - side / 2) * spacing
                y_offset = (row - side / 2) * spacing
                dx, dy = rotate(x_offset, y_offset)
                positions.append(center + np.array([dx, dy, 0]))
        
        elif self.config.formation_type == FormationType.CUSTOM:
            # Use custom offsets
            for i in range(num_vehicles):
                offset_key = str(i)
                if offset_key in self.config.custom_offsets:
                    offset = self.config.custom_offsets[offset_key]
                    dx, dy = rotate(offset[0] * self.config.scale, offset[1] * self.config.scale)
                    positions.append(center + np.array([dx, dy, offset[2] if len(offset) > 2 else 0]))
                else:
                    positions.append(center.copy())
        
        return positions
    
    def get_target_positions(
        self,
        states: Dict[str, VehicleState],
        center: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Get target positions for all vehicles.
        
        Args:
            states: Current vehicle states
            center: Formation center (default: centroid)
            
        Returns:
            Target positions per vehicle
        """
        vehicle_ids = list(states.keys())
        
        if not self._slot_assignments or set(vehicle_ids) != set(self._slot_assignments.keys()):
            self.assign_slots(vehicle_ids)
        
        if center is None:
            positions = np.array([s.position for s in states.values()])
            center = positions.mean(axis=0)
        
        formation_positions = self.compute_formation_positions(center, len(vehicle_ids))
        
        targets = {}
        for vid, slot in self._slot_assignments.items():
            if slot < len(formation_positions):
                targets[vid] = formation_positions[slot]
                # Set altitude
                targets[vid][2] = self.config.altitude
        
        return targets
    
    def compute_control(
        self,
        states: Dict[str, VehicleState],
        targets: Dict[str, np.ndarray],
        gain_p: float = 2.0,
        gain_d: float = 1.0,
        max_speed: float = 20.0,
        max_accel: float = 5.0,
    ) -> Dict[str, np.ndarray]:
        """Compute formation control commands.
        
        Args:
            states: Current vehicle states
            targets: Target positions
            gain_p: Proportional gain
            gain_d: Derivative gain
            max_speed: Maximum speed (m/s)
            max_accel: Maximum acceleration (m/s²)
            
        Returns:
            Acceleration commands per vehicle
        """
        commands = {}
        
        for vid, state in states.items():
            if vid not in targets:
                commands[vid] = np.zeros(3)
                continue
            
            target = targets[vid]
            
            # Position error
            pos_error = target - state.position
            
            # Desired velocity
            vel_desired = gain_p * pos_error
            speed = np.linalg.norm(vel_desired)
            if speed > max_speed:
                vel_desired = vel_desired * max_speed / speed
            
            # Velocity error
            vel_error = vel_desired - state.velocity
            
            # Acceleration command
            accel = gain_d * vel_error
            accel_mag = np.linalg.norm(accel)
            if accel_mag > max_accel:
                accel = accel * max_accel / accel_mag
            
            commands[vid] = accel
        
        return commands
    
    def evaluate(
        self,
        states: Dict[str, VehicleState],
        targets: Dict[str, np.ndarray],
    ) -> FormationState:
        """Evaluate formation quality.
        
        Args:
            states: Current vehicle states
            targets: Target positions
            
        Returns:
            FormationState with metrics
        """
        return FormationState.from_states(states, targets)


def compute_formation_positions(
    formation_type: FormationType,
    center: np.ndarray,
    num_vehicles: int,
    spacing: float = 10.0,
    heading: float = 0.0,
) -> List[np.ndarray]:
    """Compute formation positions.
    
    Args:
        formation_type: Type of formation
        center: Formation center
        num_vehicles: Number of vehicles
        spacing: Inter-vehicle spacing
        heading: Formation heading (radians)
        
    Returns:
        List of target positions
    """
    config = FormationConfig(
        formation_type=formation_type,
        spacing=spacing,
        heading=heading,
    )
    controller = FormationController(config)
    return controller.compute_formation_positions(center, num_vehicles)


def validate_formation(
    states: Dict[str, VehicleState],
    targets: Dict[str, np.ndarray],
    threshold: float = 0.5,
) -> bool:
    """Check if vehicles are in formation.
    
    Args:
        states: Current vehicle states
        targets: Target positions
        threshold: Maximum position error
        
    Returns:
        True if all vehicles within threshold of targets
    """
    for vid, target in targets.items():
        if vid not in states:
            continue
        error = np.linalg.norm(states[vid].position - target)
        if error > threshold:
            return False
    return True
