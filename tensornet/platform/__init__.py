"""
HyperTensor Platform — Phases 1 + 2 + 3
=========================================

Canonical interfaces and infrastructure for the unified simulation + inference
platform.  Every domain pack, solver, and workflow must conform to these
protocols.  See ADR-0001 through ADR-0007 for rationale.

Submodules
----------
protocols   – ProblemSpec, Discretization, Operator, Solver, Observable, Workflow
domain_pack – DomainPack ABC + registry / discovery
data_model  – Mesh, Field, BoundaryCondition, InitialCondition containers
solvers     – Time integrators, linear solvers, nonlinear solvers (orchestration)
reproduce   – Deterministic-run context, seed management, artifact hashing
checkpoint  – Serialization / deserialization to HDF5-like Zarr stores
vv          – Verification & Validation harness (MMS, convergence, conservation,
              stability, performance, benchmarks)  [Phase 2]

Platform version: 1.0.0  (Phase 3 — Domain-Pack Framework & Anchor Vertical Slices)
"""

__version__ = "1.0.0"

from tensornet.platform.protocols import (
    ProblemSpec,
    Discretization,
    OperatorProto,
    Solver,
    Observable,
    Workflow,
)
from tensornet.platform.domain_pack import (
    DomainPack,
    DomainRegistry,
    get_registry,
)
from tensornet.platform.data_model import (
    Mesh,
    StructuredMesh,
    UnstructuredMesh,
    FieldData,
    BoundaryCondition,
    InitialCondition,
    BCType,
    SimulationState,
)
from tensornet.platform.solvers import (
    TimeIntegrator,
    ForwardEuler,
    RK4,
    IMEX_Euler,
    SymplecticEuler,
    StormerVerlet,
    LinearSolverProto,
    ConjugateGradient,
    GMRES,
    NonlinearSolverProto,
    NewtonSolver,
    PicardSolver,
)
from tensornet.platform.reproduce import (
    ReproducibilityContext,
    ArtifactHash,
    hash_tensor,
    capture_environment,
)
from tensornet.platform.checkpoint import (
    CheckpointStore,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Protocols
    "ProblemSpec",
    "Discretization",
    "OperatorProto",
    "Solver",
    "Observable",
    "Workflow",
    # Domain packs
    "DomainPack",
    "DomainRegistry",
    "get_registry",
    # Data model
    "Mesh",
    "StructuredMesh",
    "UnstructuredMesh",
    "FieldData",
    "BoundaryCondition",
    "InitialCondition",
    "BCType",
    "SimulationState",
    # Solvers
    "TimeIntegrator",
    "ForwardEuler",
    "RK4",
    "IMEX_Euler",
    "SymplecticEuler",
    "StormerVerlet",
    "LinearSolverProto",
    "ConjugateGradient",
    "GMRES",
    "NonlinearSolverProto",
    "NewtonSolver",
    "PicardSolver",
    # Reproducibility
    "ReproducibilityContext",
    "ArtifactHash",
    "hash_tensor",
    "capture_environment",
    # Checkpointing
    "CheckpointStore",
    "save_checkpoint",
    "load_checkpoint",
]
