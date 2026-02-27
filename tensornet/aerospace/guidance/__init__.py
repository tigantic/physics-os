"""
Guidance Module
===============

Physics-aware trajectory and guidance for hypersonic vehicles.

Submodules:
    trajectory: 6-DOF dynamics, atmospheric model, integration
    controller: Guidance laws, constraint handling, CFD coupling
"""

from tensornet.aerospace.guidance.controller import (
                                           ConstraintType,
                                           GuidanceCommand,
                                           GuidanceController,
                                           GuidanceMode,
                                           TrajectoryConstraint,
                                           bank_angle_guidance,
                                           proportional_navigation,
)
from tensornet.aerospace.guidance.trajectory import (
                                           AeroCoefficients,
                                           AtmosphericModel,
                                           TrajectoryConfig,
                                           TrajectorySolver,
                                           VehicleState,
                                           exponential_atmosphere,
                                           isa_atmosphere,
)

__all__ = [
    # Trajectory
    "AtmosphericModel",
    "VehicleState",
    "AeroCoefficients",
    "TrajectoryConfig",
    "TrajectorySolver",
    "isa_atmosphere",
    "exponential_atmosphere",
    # Controller
    "GuidanceMode",
    "ConstraintType",
    "GuidanceCommand",
    "TrajectoryConstraint",
    "GuidanceController",
    "proportional_navigation",
    "bank_angle_guidance",
]
