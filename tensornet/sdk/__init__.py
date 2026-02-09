"""
HyperTensor SDK — Stable Public API Surface
=============================================

This module is the **primary entry point** for external teams building on
the HyperTensor platform.  It re-exports a curated, stable subset of the
internal platform modules so that consumers are shielded from internal
reorgs.

Versioning follows `Semantic Versioning 2.0.0 <https://semver.org/>`_.

Quick-start
-----------
::

    from tensornet.sdk import WorkflowBuilder, FieldData, StructuredMesh

    wf = (
        WorkflowBuilder("heat_1d")
        .domain(shape=(128,), extent=((0.0, 1.0),))
        .field("T", ic="uniform", value=300.0)
        .solver("PHY-I.1")
        .time(0.0, 0.5, dt=1e-3)
        .export("vtu", path="out")
        .build()
    )
    result = wf.run()
    print(result.solve_result.t_final)

Modules
-------
workflow    — ``WorkflowBuilder`` composable pipeline builder
recipes     — Pre-built per-domain recipes (CFD, heat, structural, …)

Re-exports from platform
~~~~~~~~~~~~~~~~~~~~~~~~
data_model  — ``Mesh``, ``StructuredMesh``, ``UnstructuredMesh``, ``FieldData``,
              ``BoundaryCondition``, ``InitialCondition``, ``SimulationState``
protocols   — ``ProblemSpec``, ``Solver``, ``Observable``, ``SolveResult``
export      — ``export_vtu``, ``export_csv``, ``export_json``
postprocess — ``probe``, ``slice_field``, ``integrate``, ``fft_field``,
              ``gradient_field``, ``histogram``, ``field_statistics``
visualize   — ``plot_field_1d``, ``plot_field_2d``, ``plot_convergence``
"""

from __future__ import annotations

__sdk_version__ = "2.0.0"

# ── Workflow builder ────────────────────────────────────────────────────────
from tensornet.sdk.workflow import (
    WorkflowBuilder,
    WorkflowConfig,
    ExecutedWorkflow,
    RunResult,
    FieldSpec,
    BCSpec,
    ExportSpec,
)

# ── Data model ──────────────────────────────────────────────────────────────
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

# ── Protocols ───────────────────────────────────────────────────────────────
from tensornet.platform.protocols import (
    ProblemSpec,
    Discretization,
    OperatorProto,
    Solver,
    Observable,
    Workflow,
    SolveResult,
    WorkflowResult,
)

# ── Domain packs ────────────────────────────────────────────────────────────
from tensornet.platform.domain_pack import (
    DomainPack,
    DomainRegistry,
    get_registry,
)

# ── Solvers ─────────────────────────────────────────────────────────────────
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

# ── Reproducibility ────────────────────────────────────────────────────────
from tensornet.platform.reproduce import (
    ReproducibilityContext,
    ArtifactHash,
    hash_tensor,
    capture_environment,
)

# ── Checkpointing ──────────────────────────────────────────────────────────
from tensornet.platform.checkpoint import (
    CheckpointStore,
    save_checkpoint,
    load_checkpoint,
)

# ── Export / Import ─────────────────────────────────────────────────────────
from tensornet.platform.export import (
    export_vtu,
    export_csv,
    export_json,
    ExportBundle,
)

from tensornet.platform.mesh_import import (
    import_gmsh,
    import_raw,
    detect_mesh_format,
    MeshImportError,
)

# ── Post-processing ────────────────────────────────────────────────────────
from tensornet.platform.postprocess import (
    probe,
    slice_field,
    integrate,
    field_statistics,
    FieldStats,
    fft_field,
    gradient_field,
    histogram,
)

# ── Visualization (optional — requires matplotlib) ─────────────────────────
from tensornet.platform.visualize import (
    plot_field_1d,
    plot_field_2d,
    plot_convergence,
    plot_observable_history,
    plot_spectrum,
    ensure_matplotlib,
)

# ── Deprecation / versioning ───────────────────────────────────────────────
from tensornet.platform.deprecation import (
    PLATFORM_VERSION,
    VersionInfo,
    deprecated,
    since,
    check_version_gate,
)

# ── Security ────────────────────────────────────────────────────────────────
from tensornet.platform.security import (
    generate_sbom,
    audit_dependencies,
    license_audit,
)

__all__ = [
    # SDK-native
    "WorkflowBuilder",
    "WorkflowConfig",
    "ExecutedWorkflow",
    "RunResult",
    "FieldSpec",
    "BCSpec",
    "ExportSpec",
    # Data model
    "Mesh",
    "StructuredMesh",
    "UnstructuredMesh",
    "FieldData",
    "BoundaryCondition",
    "InitialCondition",
    "BCType",
    "SimulationState",
    # Protocols
    "ProblemSpec",
    "Discretization",
    "OperatorProto",
    "Solver",
    "Observable",
    "Workflow",
    "SolveResult",
    "WorkflowResult",
    # Domain packs
    "DomainPack",
    "DomainRegistry",
    "get_registry",
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
    # Deprecation
    "PLATFORM_VERSION",
    "VersionInfo",
    "deprecated",
    "since",
    "check_version_gate",
    # Security
    "generate_sbom",
    "audit_dependencies",
    "license_audit",
]
