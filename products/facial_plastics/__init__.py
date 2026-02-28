"""
Ontic Facial Plastics Platform
======================================

Physics-grounded, data-driven surgical simulation and planning system
for facial plastic and reconstructive surgery.

Built on the physics-os multi-physics engine.

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

__version__ = "1.0.0"

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
    AnatomyGenerator,
    AnthropometricProfile,
    CaseLibrary,
    CaseLibraryCurator,
    DicomIngester,
    PairedDatasetBuilder,
    PairedDatasetReport,
    PairedQCThresholds,
    PairedSample,
    PhotoIngester,
    PopulationSampler,
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
    BLEPHAROPLASTY_OPERATORS,
    BlepharoplastyPlanBuilder,
    FACELIFT_OPERATORS,
    FaceliftPlanBuilder,
    FILLER_OPERATORS,
    FillerPlanBuilder,
    RHINOPLASTY_OPERATORS,
    RhinoplastyPlanBuilder,
)

# ── UI ────────────────────────────────────────────────────────────
from products.facial_plastics.ui.api import UIApplication
from products.facial_plastics.ui.auth import (
    APIKeyRecord,
    AuthMiddleware,
    KeyStore,
    RateLimitMiddleware,
)
from products.facial_plastics.ui.server import start_server
from products.facial_plastics.ui.wsgi import WSGIApplication, create_app

# ── Simulation ────────────────────────────────────────────────────
from products.facial_plastics.sim import (
    AgingRiskProfile,
    AgingTrajectory,
    AgingTrajectoryResult,
    AirwayCFDResult,
    AirwayCFDSolver,
    AnisotropicModel,
    BeamProperties,
    CartilageSolver,
    FEMResult,
    FSIResult,
    FSIValveSolver,
    FiberArchitecture,
    FiberFamily,
    FiberField,
    GraftType,
    HealingModel,
    HealingState,
    SimOrchestrator,
    SimulationResult,
    SoftTissueFEM,
    SutureElement,
    SutureSystem,
    TissueSnapshot,
    ValveGeometry,
    compute_hgo_stress,
    evaluate_anisotropic_stress,
    extract_valve_geometry,
)

# ── Metrics / UQ / Optimizer ─────────────────────────────────────
from products.facial_plastics.metrics import (
    AestheticMetrics,
    AestheticReport,
    CohortAnalytics,
    CohortReport,
    DistributedOptimizationResult,
    DistributedOptimizer,
    DistributionStats,
    EffectSize,
    FunctionalMetrics,
    FunctionalReport,
    IslandConfig,
    OptimizationResult,
    ParetoFront,
    PlanOptimizer,
    PoolBackend,
    RiskFactor,
    SafetyMetrics,
    SafetyReport,
    SobolIndices,
    SubgroupAnalysis,
    SurgeonProfile,
    TrendPoint,
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
    CrossTenantAccessError,
    Permission,
    ResourceQuota,
    Role,
    TenantConfig,
    TenantContext,
    TenantError,
    TenantManager,
    TenantNotFoundError,
    TenantQuotaExceededError,
    TenantRecord,
    TenantStatus,
    TenantSuspendedError,
    TenantTier,
)

# ── Reports ───────────────────────────────────────────────────────
from products.facial_plastics.reports import ReportBuilder

# ── Post-op ──────────────────────────────────────────────────────
from products.facial_plastics.postop import (
    AccuracyPanel,
    AlignmentResult,
    CalibrationPanel,
    CalibrationResult,
    CohortPanel,
    DashboardPayload,
    ModelCalibrator,
    OutcomeAligner,
    OutcomeIngester,
    OutcomeRecord,
    OutlierCase,
    OutlierPanel,
    PredictionValidator,
    RiskPanel,
    SurgeonPanel,
    TrendPanel,
    ValidationDashboard,
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
    "AnatomyGenerator",
    "AnthropometricProfile",
    "CaseLibrary",
    "CaseLibraryCurator",
    "DicomIngester",
    "PairedDatasetBuilder",
    "PairedDatasetReport",
    "PairedQCThresholds",
    "PairedSample",
    "PhotoIngester",
    "PopulationSampler",
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
    "BLEPHAROPLASTY_OPERATORS",
    "BlepharoplastyPlanBuilder",
    "FACELIFT_OPERATORS",
    "FaceliftPlanBuilder",
    "FILLER_OPERATORS",
    "FillerPlanBuilder",
    "PlanCompiler",
    "RHINOPLASTY_OPERATORS",
    "RhinoplastyPlanBuilder",
    "SequenceNode",
    "SurgicalOp",
    "SurgicalPlan",
    # UI
    "APIKeyRecord",
    "AuthMiddleware",
    "KeyStore",
    "RateLimitMiddleware",
    "UIApplication",
    "WSGIApplication",
    "create_app",
    "start_server",
    # Sim
    "AgingRiskProfile",
    "AgingTrajectory",
    "AgingTrajectoryResult",
    "AirwayCFDResult",
    "AirwayCFDSolver",
    "AnisotropicModel",
    "BeamProperties",
    "CartilageSolver",
    "FEMResult",
    "FSIResult",
    "FSIValveSolver",
    "FiberArchitecture",
    "FiberFamily",
    "FiberField",
    "GraftType",
    "HealingModel",
    "HealingState",
    "SimOrchestrator",
    "SimulationResult",
    "SoftTissueFEM",
    "SutureElement",
    "SutureSystem",
    "TissueSnapshot",
    "ValveGeometry",
    "compute_hgo_stress",
    "evaluate_anisotropic_stress",
    "extract_valve_geometry",
    # Metrics
    "AestheticMetrics",
    "AestheticReport",
    "CohortAnalytics",
    "CohortReport",
    "DistributedOptimizationResult",
    "DistributedOptimizer",
    "DistributionStats",
    "EffectSize",
    "FunctionalMetrics",
    "FunctionalReport",
    "IslandConfig",
    "OptimizationResult",
    "ParetoFront",
    "PlanOptimizer",
    "PoolBackend",
    "RiskFactor",
    "SafetyMetrics",
    "SafetyReport",
    "SobolIndices",
    "SubgroupAnalysis",
    "SurgeonProfile",
    "TrendPoint",
    "UncertaintyQuantifier",
    "UQResult",
    # Governance
    "AccessControl",
    "AuditEvent",
    "AuditLog",
    "ConsentManager",
    "ConsentRecord",
    "CrossTenantAccessError",
    "Permission",
    "ResourceQuota",
    "Role",
    "TenantConfig",
    "TenantContext",
    "TenantError",
    "TenantManager",
    "TenantNotFoundError",
    "TenantQuotaExceededError",
    "TenantRecord",
    "TenantStatus",
    "TenantSuspendedError",
    "TenantTier",
    # Reports
    "ReportBuilder",
    # Postop
    "AccuracyPanel",
    "AlignmentResult",
    "CalibrationPanel",
    "CalibrationResult",
    "CohortPanel",
    "DashboardPayload",
    "ModelCalibrator",
    "OutcomeAligner",
    "OutcomeIngester",
    "OutcomeRecord",
    "OutlierCase",
    "OutlierPanel",
    "PredictionValidator",
    "RiskPanel",
    "SurgeonPanel",
    "TrendPanel",
    "ValidationDashboard",
    "ValidationReport",
]
