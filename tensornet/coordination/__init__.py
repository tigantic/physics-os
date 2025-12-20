# Copyright (c) 2025 Tigantic
# Phase 18: Multi-Vehicle Coordination
"""
Multi-vehicle coordination module.

Provides swarm coordination, formation control, task allocation,
and consensus protocols for distributed autonomous vehicle systems.
"""

from tensornet.coordination.swarm import (
    VehicleState,
    SwarmConfig,
    SwarmTopology,
    SwarmCoordinator,
    compute_swarm_centroid,
    compute_swarm_spread,
)
from tensornet.coordination.formation import (
    FormationType,
    FormationConfig,
    FormationState,
    FormationController,
    compute_formation_positions,
    validate_formation,
)
from tensornet.coordination.task_allocation import (
    TaskPriority,
    TaskStatus,
    Task,
    Assignment,
    TaskAllocator,
    AuctionProtocol,
    allocate_tasks,
)
from tensornet.coordination.consensus import (
    ConsensusState,
    ConsensusConfig,
    ConsensusProtocol,
    AverageConsensus,
    MaxConsensus,
    run_consensus,
)

__all__ = [
    # Swarm coordination
    "VehicleState",
    "SwarmConfig",
    "SwarmTopology",
    "SwarmCoordinator",
    "compute_swarm_centroid",
    "compute_swarm_spread",
    # Formation control
    "FormationType",
    "FormationConfig",
    "FormationState",
    "FormationController",
    "compute_formation_positions",
    "validate_formation",
    # Task allocation
    "TaskPriority",
    "TaskStatus",
    "Task",
    "Assignment",
    "TaskAllocator",
    "AuctionProtocol",
    "allocate_tasks",
    # Consensus protocols
    "ConsensusState",
    "ConsensusConfig",
    "ConsensusProtocol",
    "AverageConsensus",
    "MaxConsensus",
    "run_consensus",
]
