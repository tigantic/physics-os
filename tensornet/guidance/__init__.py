"""
Guidance Module
===============

Physics-aware trajectory and guidance for hypersonic vehicles.

Submodules:
    trajectory: 6-DOF dynamics, atmospheric model, integration
    controller: Guidance laws, constraint handling, CFD coupling
"""

from tensornet.guidance.trajectory import (
    AtmosphericModel,
    VehicleState,
    AeroCoefficients,
    TrajectoryConfig,
    TrajectorySolver,
    isa_atmosphere,
    exponential_atmosphere,
)

from tensornet.guidance.controller import (
    GuidanceMode,
    ConstraintType,
    GuidanceCommand,
    TrajectoryConstraint,
    GuidanceController,
    proportional_navigation,
    bank_angle_guidance,
)

__all__ = [
    # Trajectory
    'AtmosphericModel',
    'VehicleState',
    'AeroCoefficients',
    'TrajectoryConfig',
    'TrajectorySolver',
    'isa_atmosphere',
    'exponential_atmosphere',
    # Controller
    'GuidanceMode',
    'ConstraintType',
    'GuidanceCommand',
    'TrajectoryConstraint',
    'GuidanceController',
    'proportional_navigation',
    'bank_angle_guidance',
]
