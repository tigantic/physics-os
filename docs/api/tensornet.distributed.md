# Module `ontic.distributed`

Distributed Computing Module for The Physics OS.

This module provides multi-GPU and multi-node support for
large-scale tensor network CFD simulations.

Components:
    - domain_decomp: Domain decomposition for parallel CFD
    - gpu_manager: Multi-GPU resource management
    - communication: MPI-style communication patterns
    - scheduler: Distributed task scheduling
    - parallel_solver: Parallel iterative solvers

**Contents:**

- [Submodules](#submodules)

## Submodules

- [`distributed.communication`](#distributed-communication)
- [`distributed.domain_decomp`](#distributed-domain_decomp)
- [`distributed.gpu_manager`](#distributed-gpu_manager)
- [`distributed.scheduler`](#distributed-scheduler)
