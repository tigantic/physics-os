"""
Ontic Platform — V2.0.0 (Phases 1–7)
=============================================

Canonical interfaces and infrastructure for the unified simulation + inference
platform.  Every domain pack, solver, and workflow must conform to these
protocols.  See ADR-0001 through ADR-0011 for rationale.

Submodules
----------
protocols    – ProblemSpec, Discretization, Operator, Solver, Observable, Workflow
domain_pack  – DomainPack ABC + registry / discovery
data_model   – Mesh, Field, BoundaryCondition, InitialCondition containers
solvers      – Time integrators, linear solvers, nonlinear solvers (orchestration)
reproduce    – Deterministic-run context, seed management, artifact hashing
checkpoint   – Serialization / deserialization to HDF5-like Zarr stores
vv           – Verification & Validation harness (MMS, convergence, conservation,
               stability, performance, benchmarks)  [Phase 2]
lineage      – DAG-based provenance tracking  [Phase 4]

Phase 5–6:
qtt_accel    – QTT-format acceleration: contraction, rounding, arithmetic, ALS
coupled      – Coupled-physics infrastructure: multi-field solvers, UQ sampling,
               inverse problems, topology/shape optimization

Phase 7 (Productization):
export       – VTK/VTU, XDMF+HDF5, CSV, JSON export for external tools
mesh_import  – GMSH v2/v4, raw-array mesh import
postprocess  – probe, slice, integrate, FFT, gradient, histogram, statistics
visualize    – matplotlib-based field / convergence / spectrum plots
deprecation  – SemVer enforcement, @deprecated/@since decorators, version gates
security     – SBOM generation, dependency audit, license compliance

Platform version: 2.0.0  (Phase 7 — Productization & Ecosystem Hardening)
"""

# NOTE: This is the *platform API* version (SemVer for the substrate).
# The *package* version (pyproject.toml / ontic.__version__) is 40.0.0,
# which tracks the full inventory release.  Two separate version namespaces.
__version__ = "2.0.0"

from ontic.platform.protocols import (
    ProblemSpec,
    Discretization,
    OperatorProto,
    Solver,
    Observable,
    Workflow,
)
from ontic.platform.domain_pack import (
    DomainPack,
    DomainRegistry,
    get_registry,
)
from ontic.platform.data_model import (
    Mesh,
    StructuredMesh,
    UnstructuredMesh,
    FieldData,
    BoundaryCondition,
    InitialCondition,
    BCType,
    SimulationState,
)
from ontic.platform.solvers import (
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
from ontic.platform.reproduce import (
    ReproducibilityContext,
    ArtifactHash,
    hash_tensor,
    capture_environment,
)
from ontic.platform.checkpoint import (
    CheckpointStore,
    save_checkpoint,
    load_checkpoint,
)

# Phase 7 — Productization modules
from ontic.platform.export import (
    export_vtu,
    export_xdmf_hdf5,
    export_csv,
    export_json,
    ExportBundle,
)
from ontic.platform.mesh_import import (
    import_gmsh,
    import_raw,
    detect_mesh_format,
    MeshImportError,
)
from ontic.platform.postprocess import (
    probe,
    slice_field,
    integrate,
    field_statistics,
    FieldStats,
    fft_field,
    gradient_field,
    histogram,
)
from ontic.platform.visualize import (
    plot_field_1d,
    plot_field_2d,
    plot_convergence,
    plot_observable_history,
    plot_spectrum,
    ensure_matplotlib,
)
from ontic.platform.deprecation import (
    PLATFORM_VERSION,
    VersionInfo,
    deprecated,
    since,
    check_version_gate,
)
from ontic.platform.security import (
    generate_sbom,
    audit_dependencies,
    license_audit,
)

# §6 Data Infrastructure modules
from ontic.platform.timeseries_db import (
    AggregatedPoint,
    InMemoryTSDB,
    RetentionPolicy,
    TSDBBackend,
    TimeSeriesPoint,
    TimeSeriesQuery,
)
from ontic.platform.lakehouse import (
    ArtifactType,
    ArtifactMetadata,
    LakehouseCatalog,
    LakehouseQuery,
    LakehouseStore,
    PartitionScheme,
)
from ontic.platform.arrow_export import (
    ArrowBatch,
    ColumnSchema,
    ColumnType,
    ParquetReader,
    ParquetWriter,
    TableSchema,
    export_to_parquet,
    import_from_parquet,
    simulation_state_to_batch,
)
from ontic.platform.experiment_tracker import (
    Experiment,
    ExperimentTracker,
    MetricEntry,
    Run,
    RunStatus,
)
from ontic.platform.federation import (
    ChunkStatus,
    DataChunk,
    FederationManager,
    FederationNode,
    FederationProtocol,
    FederationRegistry,
    NodeRole,
)
from ontic.platform.replay import (
    EventType,
    ReplayEngine,
    ReplayEvent,
    ReplayLog,
)
from ontic.platform.data_versioning import (
    ContentStore,
    DataVersioning,
    DatasetSnapshot,
    DiffEntry,
    VersionGraph,
    diff_snapshots,
    merge_snapshots,
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
    # Export
    "export_vtu",
    "export_xdmf_hdf5",
    "export_csv",
    "export_json",
    "ExportBundle",
    # Import
    "import_gmsh",
    "import_raw",
    "detect_mesh_format",
    "MeshImportError",
    # Post-processing
    "probe",
    "slice_field",
    "integrate",
    "field_statistics",
    "FieldStats",
    "fft_field",
    "gradient_field",
    "histogram",
    # Visualization
    "plot_field_1d",
    "plot_field_2d",
    "plot_convergence",
    "plot_observable_history",
    "plot_spectrum",
    "ensure_matplotlib",
    # Deprecation / versioning
    "PLATFORM_VERSION",
    "VersionInfo",
    "deprecated",
    "since",
    "check_version_gate",
    # Security
    "generate_sbom",
    "audit_dependencies",
    "license_audit",
    # §6 — Time-series DB
    "TimeSeriesPoint",
    "TimeSeriesQuery",
    "AggregatedPoint",
    "RetentionPolicy",
    "TSDBBackend",
    "InMemoryTSDB",
    # §6 — Lakehouse
    "ArtifactType",
    "ArtifactMetadata",
    "PartitionScheme",
    "LakehouseCatalog",
    "LakehouseStore",
    "LakehouseQuery",
    # §6 — Arrow / Parquet
    "ColumnType",
    "ColumnSchema",
    "TableSchema",
    "ArrowBatch",
    "ParquetWriter",
    "ParquetReader",
    "simulation_state_to_batch",
    "export_to_parquet",
    "import_from_parquet",
    # §6 — Experiment tracker
    "RunStatus",
    "MetricEntry",
    "Run",
    "Experiment",
    "ExperimentTracker",
    # §6 — Federation
    "NodeRole",
    "FederationNode",
    "ChunkStatus",
    "DataChunk",
    "FederationRegistry",
    "FederationProtocol",
    "FederationManager",
    # §6 — Replay
    "EventType",
    "ReplayEvent",
    "ReplayLog",
    "ReplayEngine",
    # §6 — Data versioning
    "ContentStore",
    "DatasetSnapshot",
    "DiffEntry",
    "diff_snapshots",
    "VersionGraph",
    "merge_snapshots",
    "DataVersioning",
]
