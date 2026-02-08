"""
Verification & Validation (V&V) Harness — Phase 2
===================================================

Provides the standard toolbox that any solver must pass through to advance
from V0.4 → V0.5 → V0.6 in the capability ledger.

Submodules
----------
mms           – Method of Manufactured Solutions framework
convergence   – Grid / timestep refinement studies
conservation  – Conserved-quantity monitors (mass, energy, divergence-free)
stability     – CFL checker, stiffness detector, blow-up sentinel
performance   – Profiling harness, memory tracker, roofline estimates
benchmarks    – Benchmark registry with golden outputs + acceptance thresholds

Phase 2 exit gate:  *Any solver at V0.4 must be able to reach V0.5 using
standard harness outputs — no bespoke validation.*
"""

__version__ = "0.1.0"

from tensornet.platform.vv.mms import (
    ManufacturedSolution,
    MMSProblem,
    ConvergencePoint,
    MMSConvergenceResult,
    mms_convergence_study,
)
from tensornet.platform.vv.convergence import (
    RefinementStudy,
    RefinementPoint,
    RefinementResult,
    grid_refinement_study,
    timestep_refinement_study,
    compute_order,
)
from tensornet.platform.vv.conservation import (
    ConservedQuantity,
    MassIntegral,
    EnergyIntegral,
    ConservationMonitor,
    ConservationReport,
)
from tensornet.platform.vv.stability import (
    StabilityCheck,
    CFLChecker,
    BlowupDetector,
    StiffnessEstimator,
    StabilityReport,
)
from tensornet.platform.vv.performance import (
    TimingResult,
    MemorySnapshot,
    PerformanceHarness,
    PerformanceReport,
    ScalingStudy,
    ScalingResult,
)
from tensornet.platform.vv.benchmarks import (
    BenchmarkProblem,
    BenchmarkRegistry,
    BenchmarkResult,
    GoldenOutput,
    get_benchmark_registry,
)

__all__ = [
    # MMS
    "ManufacturedSolution",
    "MMSProblem",
    "ConvergencePoint",
    "MMSConvergenceResult",
    "mms_convergence_study",
    # Convergence
    "RefinementStudy",
    "RefinementPoint",
    "RefinementResult",
    "grid_refinement_study",
    "timestep_refinement_study",
    "compute_order",
    # Conservation
    "ConservedQuantity",
    "MassIntegral",
    "EnergyIntegral",
    "ConservationMonitor",
    "ConservationReport",
    # Stability
    "StabilityCheck",
    "CFLChecker",
    "BlowupDetector",
    "StiffnessEstimator",
    "StabilityReport",
    # Performance
    "TimingResult",
    "MemorySnapshot",
    "PerformanceHarness",
    "PerformanceReport",
    "ScalingStudy",
    "ScalingResult",
    # Benchmarks
    "BenchmarkProblem",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "GoldenOutput",
    "get_benchmark_registry",
]
