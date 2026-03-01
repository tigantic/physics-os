"""
Ontic Engine Physics Module
========================

Hypersonic flight physics and trajectory optimization.

Modules:
    - hypersonic: Flight hazard field calculation for trajectory optimization
    - trajectory_optimizer: Find optimal paths through hazard fields
"""

from .hypersonic import (
                         HazardField,
                         VehicleConfig,
                         calculate_dynamic_pressure,
                         calculate_equilibrium_wall_temperature,
                         calculate_hazard_field,
                         calculate_stagnation_temperature,
                         calculate_wind_shear,
                         find_safe_corridors,
                         hazard_to_traversability,
)
from .trajectory_optimizer import (
                         Trajectory,
                         Waypoint,
                         find_optimal_trajectory,
                         optimize_trajectory_fast_marching,
                         optimize_trajectory_gradient,
)

__all__ = [
    # Hazard field
    "calculate_hazard_field",
    "calculate_dynamic_pressure",
    "calculate_stagnation_temperature",
    "calculate_equilibrium_wall_temperature",
    "calculate_wind_shear",
    "hazard_to_traversability",
    "find_safe_corridors",
    "HazardField",
    "VehicleConfig",
    # Trajectory
    "find_optimal_trajectory",
    "optimize_trajectory_gradient",
    "optimize_trajectory_fast_marching",
    "Trajectory",
    "Waypoint",
]
