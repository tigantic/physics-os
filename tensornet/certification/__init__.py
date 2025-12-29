"""
Certification Module for HyperTensor
====================================

Provides DO-178C compliance infrastructure and hardware deployment
capabilities for safety-critical tensor network applications.

Modules:
    - do178c: DO-178C certification framework
    - hardware: Hardware deployment and optimization

Usage:
    from tensornet.certification import (
        DAL, VerificationMethod, RequirementsDatabase,
        HardwareSpec, deploy_to_hardware
    )
"""

from .do178c import (
    # Design Assurance Levels
    DAL,
    VerificationMethod,
    CoverageType,
    
    # Requirements Management
    RequirementType,
    RequirementStatus,
    Requirement,
    RequirementsDatabase,
    
    # Test Evidence
    TestResult,
    TestCase,
    CoverageReport,
    CoverageAnalyzer,
    
    # Safety Assessment
    HazardSeverity,
    HazardProbability,
    Hazard,
    SafetyAssessment,
    
    # Configuration Management
    ConfigurationItem,
    ConfigurationManagement,
    
    # Verification
    VerificationEvidence,
    VerificationPackage,
    
    # Convenience Functions
    create_hypertensor_requirements,
    create_sample_safety_assessment,
)

from .hardware import (
    # Hardware Types
    HardwareType,
    Precision,
    HardwareSpec,
    HARDWARE_PRESETS,
    
    # Quantization
    QuantizationConfig,
    ModelQuantizer,
    
    # Memory Optimization
    MemoryProfile,
    MemoryOptimizer,
    
    # Real-Time Scheduling
    TaskSpec,
    RealTimeScheduler,
    
    # WCET Analysis
    WCETAnalyzer,
    
    # HIL Validation
    HILTestResult,
    HILValidator,
    
    # Deployment
    DeploymentArtifact,
    DeploymentPackage,
    
    # Convenience Functions
    deploy_to_hardware,
    estimate_inference_time,
)

__all__ = [
    # DO-178C
    'DAL',
    'VerificationMethod',
    'CoverageType',
    'RequirementType',
    'RequirementStatus',
    'Requirement',
    'RequirementsDatabase',
    'TestResult',
    'TestCase',
    'CoverageReport',
    'CoverageAnalyzer',
    'HazardSeverity',
    'HazardProbability',
    'Hazard',
    'SafetyAssessment',
    'ConfigurationItem',
    'ConfigurationManagement',
    'VerificationEvidence',
    'VerificationPackage',
    'create_hypertensor_requirements',
    'create_sample_safety_assessment',
    
    # Hardware
    'HardwareType',
    'Precision',
    'HardwareSpec',
    'HARDWARE_PRESETS',
    'QuantizationConfig',
    'ModelQuantizer',
    'MemoryProfile',
    'MemoryOptimizer',
    'TaskSpec',
    'RealTimeScheduler',
    'WCETAnalyzer',
    'HILTestResult',
    'HILValidator',
    'DeploymentArtifact',
    'DeploymentPackage',
    'deploy_to_hardware',
    'estimate_inference_time',
]
