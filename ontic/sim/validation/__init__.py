"""
Validation Module for Project HyperTensor.

Phase 15: Comprehensive validation and testing framework including:
- Physical validation tests (conservation laws, analytical solutions)
- Performance benchmarking utilities
- Regression testing framework
- Verification and validation (V&V) infrastructure

Components:
    - physical: Physics-based validation tests
    - benchmarks: Performance benchmarking utilities
    - regression: Automated regression testing
    - vv: Verification & Validation framework
"""

from .benchmarks import (  # Benchmark configuration; Timer utilities; Memory tracking; Scalability tests; Benchmark runners
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
    MemorySnapshot,
    MemoryTracker,
    PerformanceTimer,
    ScalabilityTest,
    StrongScalingTest,
    TimerContext,
    WeakScalingTest,
    compare_benchmarks,
    run_benchmark,
    run_benchmark_suite,
)
from .physical import (  # Conservation law validators; Analytical solution validators; Test result structures
    AnalyticalValidator,
    BlasiusValidator,
    ConservationValidator,
    EnergyConservationTest,
    IsentropicVortexValidator,
    MassConservationTest,
    MomentumConservationTest,
    ObliqueShockValidator,
    SodShockValidator,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    run_physical_validation,
)
from .regression import (  # Regression test framework; Golden value management; Comparison utilities; Test runners
    ArrayComparator,
    GoldenValue,
    GoldenValueStore,
    RegressionResult,
    RegressionSuite,
    RegressionTest,
    StateComparator,
    TensorComparator,
    run_full_regression,
    run_regression_tests,
    update_golden_values,
)
from .vv import (  # V&V Framework; Verification utilities; Validation utilities; Uncertainty quantification; Execution
    AnalyticalValidation,
    CodeVerification,
    ExperimentalValidation,
    IntegrationVerification,
    UncertaintyBand,
    UnitVerification,
    ValidationCase,
    ValidationUncertainty,
    VVCategory,
    VVLevel,
    VVPlan,
    VVReport,
    VVTest,
    VVTestResult,
    generate_vv_report,
    run_vv_plan,
)

__all__ = [
    # Physical validation
    "ConservationValidator",
    "MassConservationTest",
    "MomentumConservationTest",
    "EnergyConservationTest",
    "AnalyticalValidator",
    "SodShockValidator",
    "BlasiusValidator",
    "ObliqueShockValidator",
    "IsentropicVortexValidator",
    "ValidationResult",
    "ValidationSeverity",
    "ValidationReport",
    "run_physical_validation",
    # Benchmarks
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkSuite",
    "TimerContext",
    "PerformanceTimer",
    "MemoryTracker",
    "MemorySnapshot",
    "ScalabilityTest",
    "WeakScalingTest",
    "StrongScalingTest",
    "run_benchmark",
    "run_benchmark_suite",
    "compare_benchmarks",
    # Regression
    "RegressionTest",
    "RegressionSuite",
    "RegressionResult",
    "GoldenValue",
    "GoldenValueStore",
    "update_golden_values",
    "ArrayComparator",
    "TensorComparator",
    "StateComparator",
    "run_regression_tests",
    "run_full_regression",
    # V&V
    "VVLevel",
    "VVCategory",
    "VVTest",
    "VVTestResult",
    "VVPlan",
    "VVReport",
    "CodeVerification",
    "UnitVerification",
    "IntegrationVerification",
    "ValidationCase",
    "ExperimentalValidation",
    "AnalyticalValidation",
    "UncertaintyBand",
    "ValidationUncertainty",
    "run_vv_plan",
    "generate_vv_report",
]
