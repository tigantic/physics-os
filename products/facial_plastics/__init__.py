"""
HyperTensor Facial Plastics Platform
======================================

Physics-grounded, data-driven surgical simulation and planning system
for facial plastic and reconstructive surgery.

Built on the HyperTensor-VM multi-physics engine.

Sub-packages
------------
core        – Types, config, provenance, CaseBundle
data        – DICOM/photo/surface ingestion, case library, synthetic augment
twin        – Digital twin builder (segment, register, mesh, assign materials)
plan        – Surgical plan DSL, compiler, rhinoplasty operators
sim         – Multi-physics simulation (FEM, CFD, cartilage, sutures, healing)
metrics     – Aesthetic/functional/safety metrics, UQ, multi-objective optimizer
governance  – Audit trail, informed consent, RBAC
postop      – Outcome ingest, alignment, calibration, validation
"""

__version__ = "0.1.0"

# ── Core ──────────────────────────────────────────────────────────
from products.facial_plastics.core.types import (
    BoundingBox,
    ClinicalMeasurement,
    DicomMetadata,
    Landmark,
    LandmarkType,
    MaterialModel,
    MeshElementType,
    MeshQualityReport,
    Modality,
    ProcedureType,
    QualityLevel,
    RegistrationResult,
    SegmentationMask,
    SolverType,
    StructureType,
    SurfaceMesh,
    TissueProperties,
    Vec3,
    VolumeMesh,
    generate_case_id,
)
from products.facial_plastics.core.case_bundle import CaseBundle
from products.facial_plastics.core.config import PlatformConfig
from products.facial_plastics.core.provenance import Provenance

# ── Data ingestion ────────────────────────────────────────────────
from products.facial_plastics.data import (
    CaseLibrary,
    DicomIngester,
    PhotoIngester,
    SurfaceIngester,
    SyntheticAugmenter,
)

# ── Digital twin ──────────────────────────────────────────────────
from products.facial_plastics.twin import (
    LandmarkDetector,
    MaterialAssigner,
    MultiModalRegistrar,
    MultiStructureSegmenter,
    TwinBuilder,
    VolumetricMesher,
)

# ── Plan ──────────────────────────────────────────────────────────
from products.facial_plastics.plan import (
    PlanCompiler,
    SequenceNode,
    SurgicalOp,
    SurgicalPlan,
)
from products.facial_plastics.plan.operators import (
    RhinoplastyPlanBuilder,
)

# ── Simulation ────────────────────────────────────────────────────
from products.facial_plastics.sim import (
    AirwayCFDResult,
    AirwayCFDSolver,
    CartilageSolver,
    FEMResult,
    HealingModel,
    HealingState,
    SimOrchestrator,
    SimulationResult,
    SoftTissueFEM,
    SutureElement,
    SutureSystem,
)

# ── Metrics / UQ / Optimizer ─────────────────────────────────────
from products.facial_plastics.metrics import (
    AestheticMetrics,
    AestheticReport,
    FunctionalMetrics,
    FunctionalReport,
    OptimizationResult,
    ParetoFront,
    PlanOptimizer,
    SafetyMetrics,
    SafetyReport,
    SobolIndices,
    UncertaintyQuantifier,
    UQResult,
)

# ── Governance ────────────────────────────────────────────────────
from products.facial_plastics.governance import (
    AccessControl,
    AuditEvent,
    AuditLog,
    ConsentManager,
    ConsentRecord,
    Permission,
    Role,
)

# ── Reports ───────────────────────────────────────────────────────
from products.facial_plastics.reports import ReportBuilder

# ── Post-op ──────────────────────────────────────────────────────
from products.facial_plastics.postop import (
    AlignmentResult,
    CalibrationResult,
    ModelCalibrator,
    OutcomeAligner,
    OutcomeIngester,
    OutcomeRecord,
    PredictionValidator,
    ValidationReport,
)

__all__ = [
    # Core
    "BoundingBox",
    "CaseBundle",
    "ClinicalMeasurement",
    "DicomMetadata",
    "Landmark",
    "LandmarkType",
    "MaterialModel",
    "MeshElementType",
    "MeshQualityReport",
    "Modality",
    "PlatformConfig",
    "ProcedureType",
    "Provenance",
    "QualityLevel",
    "RegistrationResult",
    "SegmentationMask",
    "SolverType",
    "StructureType",
    "SurfaceMesh",
    "TissueProperties",
    "Vec3",
    "VolumeMesh",
    "generate_case_id",
    # Data
    "CaseLibrary",
    "DicomIngester",
    "PhotoIngester",
    "SurfaceIngester",
    "SyntheticAugmenter",
    # Twin
    "LandmarkDetector",
    "MaterialAssigner",
    "MultiModalRegistrar",
    "MultiStructureSegmenter",
    "TwinBuilder",
    "VolumetricMesher",
    # Plan
    "PlanCompiler",
    "RhinoplastyPlanBuilder",
    "SequenceNode",
    "SurgicalOp",
    "SurgicalPlan",
    # Sim
    "AirwayCFDResult",
    "AirwayCFDSolver",
    "CartilageSolver",
    "FEMResult",
    "HealingModel",
    "HealingState",
    "SimOrchestrator",
    "SimulationResult",
    "SoftTissueFEM",
    "SutureElement",
    "SutureSystem",
    # Metrics
    "AestheticMetrics",
    "AestheticReport",
    "FunctionalMetrics",
    "FunctionalReport",
    "OptimizationResult",
    "ParetoFront",
    "PlanOptimizer",
    "SafetyMetrics",
    "SafetyReport",
    "SobolIndices",
    "UncertaintyQuantifier",
    "UQResult",
    # Governance
    "AccessControl",
    "AuditEvent",
    "AuditLog",
    "ConsentManager",
    "ConsentRecord",
    "Permission",
    "Role",
    # Reports
    "ReportBuilder",
    # Postop
    "AlignmentResult",
    "CalibrationResult",
    "ModelCalibrator",
    "OutcomeAligner",
    "OutcomeIngester",
    "OutcomeRecord",
    "PredictionValidator",
    "ValidationReport",
]
