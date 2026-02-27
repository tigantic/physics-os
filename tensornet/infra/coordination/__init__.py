# Copyright (c) 2025 Tigantic
# Phase 18: Multi-Vehicle Coordination
"""
Multi-vehicle coordination module.

Provides swarm coordination, formation control, task allocation,
and consensus protocols for distributed autonomous vehicle systems.
"""

from tensornet.infra.coordination.consensus import (
                                              AverageConsensus,
                                              ConsensusConfig,
                                              ConsensusProtocol,
                                              ConsensusState,
                                              MaxConsensus,
                                              run_consensus,
)
from tensornet.infra.coordination.formation import (
                                              FormationConfig,
                                              FormationController,
                                              FormationState,
                                              FormationType,
                                              compute_formation_positions,
                                              validate_formation,
)
from tensornet.infra.coordination.swarm import (
                                              SwarmConfig,
                                              SwarmCoordinator,
                                              SwarmTopology,
                                              VehicleState,
                                              compute_swarm_centroid,
                                              compute_swarm_spread,
)
from tensornet.infra.coordination.task_allocation import (
                                              Assignment,
                                              AuctionProtocol,
                                              Task,
                                              TaskAllocator,
                                              TaskPriority,
                                              TaskStatus,
                                              allocate_tasks,
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
