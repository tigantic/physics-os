"""
Distributed Computing Module for The Ontic Engine.

This module provides multi-GPU and multi-node support for
large-scale tensor network CFD simulations.

Components:
    - domain_decomp: Domain decomposition for parallel CFD
    - gpu_manager: Multi-GPU resource management
    - communication: MPI-style communication patterns
    - scheduler: Distributed task scheduling
    - parallel_solver: Parallel iterative solvers
"""

from .communication import (
                            AllReduceOp,
                            CommPattern,
                            Communicator,
                            all_reduce,
                            async_recv,
                            async_send,
                            barrier,
                            broadcast,
                            gather,
                            scatter,
)
from .domain_decomp import (
                            DomainConfig,
                            DomainDecomposition,
                            SubdomainInfo,
                            compute_ghost_zones,
                            decompose_domain,
                            exchange_ghost_data,
)
from .gpu_manager import (
                            GPUConfig,
                            GPUDevice,
                            GPUManager,
                            MemoryPool,
                            distribute_workload,
                            get_available_gpus,
                            select_optimal_device,
)
from .parallel_solver import (
                            DomainSolver,
                            ParallelCGSolver,
                            ParallelConfig,
                            ParallelGMRESSolver,
                            SchwarzPreconditioner,
                            parallel_solve,
)
from .scheduler import (
                            DistributedScheduler,
                            Task,
                            TaskConfig,
                            TaskGraph,
                            execute_parallel,
                            schedule_dependency_graph,
)

__all__ = [
    # Domain Decomposition
    "DomainConfig",
    "DomainDecomposition",
    "SubdomainInfo",
    "decompose_domain",
    "compute_ghost_zones",
    "exchange_ghost_data",
    # GPU Management
    "GPUConfig",
    "GPUDevice",
    "GPUManager",
    "MemoryPool",
    "get_available_gpus",
    "select_optimal_device",
    "distribute_workload",
    # Communication
    "CommPattern",
    "Communicator",
    "AllReduceOp",
    "async_send",
    "async_recv",
    "barrier",
    "all_reduce",
    "broadcast",
    "scatter",
    "gather",
    # Scheduler
    "TaskConfig",
    "Task",
    "TaskGraph",
    "DistributedScheduler",
    "schedule_dependency_graph",
    "execute_parallel",
    # Parallel Solvers
    "ParallelConfig",
    "DomainSolver",
    "ParallelCGSolver",
    "ParallelGMRESSolver",
    "SchwarzPreconditioner",
    "parallel_solve",
]
