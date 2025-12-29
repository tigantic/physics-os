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

from .physical import (
    # Conservation law validators
    ConservationValidator,
    MassConservationTest,
    MomentumConservationTest,
    EnergyConservationTest,
    # Analytical solution validators
    AnalyticalValidator,
    SodShockValidator,
    BlasiusValidator,
    ObliqueShockValidator,
    IsentropicVortexValidator,
    # Test result structures
    ValidationResult,
    ValidationSeverity,
    ValidationReport,
    run_physical_validation,
)

from .benchmarks import (
    # Benchmark configuration
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
    # Timer utilities
    TimerContext,
    PerformanceTimer,
    # Memory tracking
    MemoryTracker,
    MemorySnapshot,
    # Scalability tests
    ScalabilityTest,
    WeakScalingTest,
    StrongScalingTest,
    # Benchmark runners
    run_benchmark,
    run_benchmark_suite,
    compare_benchmarks,
)

from .regression import (
    # Regression test framework
    RegressionTest,
    RegressionSuite,
    RegressionResult,
    # Golden value management
    GoldenValue,
    GoldenValueStore,
    update_golden_values,
    # Comparison utilities
    ArrayComparator,
    TensorComparator,
    StateComparator,
    # Test runners
    run_regression_tests,
    run_full_regression,
)

from .vv import (
    # V&V Framework
    VVLevel,
    VVCategory,
    VVTest,
    VVTestResult,
    VVPlan,
    VVReport,
    # Verification utilities
    CodeVerification,
    UnitVerification,
    IntegrationVerification,
    # Validation utilities  
    ValidationCase,
    ExperimentalValidation,
    AnalyticalValidation,
    # Uncertainty quantification
    UncertaintyBand,
    ValidationUncertainty,
    # Execution
    run_vv_plan,
    generate_vv_report,
)

__all__ = [
    # Physical validation
    'ConservationValidator',
    'MassConservationTest',
    'MomentumConservationTest', 
    'EnergyConservationTest',
    'AnalyticalValidator',
    'SodShockValidator',
    'BlasiusValidator',
    'ObliqueShockValidator',
    'IsentropicVortexValidator',
    'ValidationResult',
    'ValidationSeverity',
    'ValidationReport',
    'run_physical_validation',
    # Benchmarks
    'BenchmarkConfig',
    'BenchmarkResult',
    'BenchmarkSuite',
    'TimerContext',
    'PerformanceTimer',
    'MemoryTracker',
    'MemorySnapshot',
    'ScalabilityTest',
    'WeakScalingTest',
    'StrongScalingTest',
    'run_benchmark',
    'run_benchmark_suite',
    'compare_benchmarks',
    # Regression
    'RegressionTest',
    'RegressionSuite',
    'RegressionResult',
    'GoldenValue',
    'GoldenValueStore',
    'update_golden_values',
    'ArrayComparator',
    'TensorComparator',
    'StateComparator',
    'run_regression_tests',
    'run_full_regression',
    # V&V
    'VVLevel',
    'VVCategory',
    'VVTest',
    'VVTestResult',
    'VVPlan',
    'VVReport',
    'CodeVerification',
    'UnitVerification',
    'IntegrationVerification',
    'ValidationCase',
    'ExperimentalValidation',
    'AnalyticalValidation',
    'UncertaintyBand',
    'ValidationUncertainty',
    'run_vv_plan',
    'generate_vv_report',
]
