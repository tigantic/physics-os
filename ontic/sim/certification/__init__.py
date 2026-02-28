"""
Certification Module for HyperTensor
====================================

Provides DO-178C compliance infrastructure and hardware deployment
capabilities for safety-critical tensor network applications.

Modules:
    - do178c: DO-178C certification framework
    - hardware: Hardware deployment and optimization

Usage:
    from ontic.sim.certification import (
        DAL, VerificationMethod, RequirementsDatabase,
        HardwareSpec, deploy_to_hardware
    )
"""

from .do178c import (  # Design Assurance Levels; Requirements Management; Test Evidence; Safety Assessment; Configuration Management; Verification; Convenience Functions
    DAL,
    ConfigurationItem,
    ConfigurationManagement,
    CoverageAnalyzer,
    CoverageReport,
    CoverageType,
    Hazard,
    HazardProbability,
    HazardSeverity,
    Requirement,
    RequirementsDatabase,
    RequirementStatus,
    RequirementType,
    SafetyAssessment,
    TestCase,
    TestResult,
    VerificationEvidence,
    VerificationMethod,
    VerificationPackage,
    create_hypertensor_requirements,
    create_sample_safety_assessment,
)
from .hardware import (  # Hardware Types; Quantization; Memory Optimization; Real-Time Scheduling; WCET Analysis; HIL Validation; Deployment; Convenience Functions
    HARDWARE_PRESETS,
    DeploymentArtifact,
    DeploymentPackage,
    HardwareSpec,
    HardwareType,
    HILTestResult,
    HILValidator,
    MemoryOptimizer,
    MemoryProfile,
    ModelQuantizer,
    Precision,
    QuantizationConfig,
    RealTimeScheduler,
    TaskSpec,
    WCETAnalyzer,
    deploy_to_hardware,
    estimate_inference_time,
)

__all__ = [
    # DO-178C
    "DAL",
    "VerificationMethod",
    "CoverageType",
    "RequirementType",
    "RequirementStatus",
    "Requirement",
    "RequirementsDatabase",
    "TestResult",
    "TestCase",
    "CoverageReport",
    "CoverageAnalyzer",
    "HazardSeverity",
    "HazardProbability",
    "Hazard",
    "SafetyAssessment",
    "ConfigurationItem",
    "ConfigurationManagement",
    "VerificationEvidence",
    "VerificationPackage",
    "create_hypertensor_requirements",
    "create_sample_safety_assessment",
    # Hardware
    "HardwareType",
    "Precision",
    "HardwareSpec",
    "HARDWARE_PRESETS",
    "QuantizationConfig",
    "ModelQuantizer",
    "MemoryProfile",
    "MemoryOptimizer",
    "TaskSpec",
    "RealTimeScheduler",
    "WCETAnalyzer",
    "HILTestResult",
    "HILValidator",
    "DeploymentArtifact",
    "DeploymentPackage",
    "deploy_to_hardware",
    "estimate_inference_time",
]
