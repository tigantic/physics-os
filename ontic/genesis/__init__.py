"""
TENSOR GENESIS — QTT Meta-Primitive Expansion Protocol

This module extends Quantized Tensor Train (QTT) compression into seven
unexploited mathematical domains plus applied science layers, creating
capabilities that are IMPOSSIBLE without QTT compression.

Constitutional Reference: TENSOR_GENESIS.md

Layers 20-26 (Meta-Primitives):
    - ot: Optimal Transport (Layer 20)
    - sgw: Spectral Graph Wavelets (Layer 21)
    - rmt: Random Matrix Theory (Layer 22)
    - tropical: Tropical Geometry (Layer 23)
    - rkhs: Kernel Methods (Layer 24)
    - topology: Persistent Homology (Layer 25)
    - ga: Geometric Algebra (Layer 26)

Layer 27+ (Applied Science):
    - aging: Biological Aging as Tensor Rank Dynamics (Layer 27)

Core Infrastructure:
    - core: Logging, exceptions, profiling, validation

Example:
    >>> from ontic.genesis import QTTSinkhorn, wasserstein_distance
    >>> from ontic.genesis.ot import QTTDistribution
    >>> 
    >>> mu = QTTDistribution.gaussian(mean=0.0, std=1.0, grid_size=2**40)
    >>> nu = QTTDistribution.gaussian(mean=1.0, std=1.0, grid_size=2**40)
    >>> W2 = wasserstein_distance(mu, nu, p=2)

    >>> from ontic.genesis import young_cell, YamanakaOperator
    >>> cell = young_cell(seed=42)
    >>> yamanaka = YamanakaOperator(target_rank=4)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

__version__ = "1.1.0"
__author__ = "Bradly Biron Baker Adams"
__constitutional_ref__ = "TENSOR_GENESIS.md"

# Core Infrastructure (Production Hardening)
from ontic.genesis.core import (
    # Logging
    get_logger,
    configure_logging,
    GenesisLogger,
    LogLevel,
    # Exceptions
    GenesisError,
    QTTRankError,
    ConvergenceError,
    DimensionMismatchError,
    NumericalInstabilityError,
    MemoryBudgetExceededError,
    InvalidInputError,
    CompressionError,
    # Profiling
    profile,
    profile_memory,
    timed,
    traced,
    ProfileResult,
    PerformanceTracker,
    # Validation
    validate_qtt_cores,
    validate_tensor_shape,
    validate_positive,
    validate_probability,
    check_numerical_stability,
)

# Layer 20: Optimal Transport
from ontic.genesis.ot import (
    QTTSinkhorn,
    QTTDistribution,
    wasserstein_distance,
    transport_plan,
    barycenter,
)

# Layer 21: Spectral Graph Wavelets
from ontic.genesis.sgw import (
    QTTLaplacian,
    QTTSignal,
    QTTGraphWavelet,
    ChebyshevApproximator,
    LowPassFilter,
    HighPassFilter,
    BandPassFilter,
)

# Layer 22: Random Matrix Theory
from ontic.genesis.rmt import (
    QTTEnsemble,
    QTTResolvent,
    SpectralDensity,
    WignerSemicircle,
    MarchenkoPastur,
    FreeConvolution,
    spectral_density,
    stieltjes_transform,
    resolvent_trace,
)

# Layer 23: Tropical Geometry
from ontic.genesis.tropical import (
    TropicalSemiring,
    MinPlusSemiring,
    MaxPlusSemiring,
    TropicalMatrix,
    tropical_matmul,
    tropical_power,
    tropical_kleene_star,
    all_pairs_shortest_path,
    floyd_warshall_tropical,
    bellman_ford_tropical,
    tropical_eigenvalue,
)

# Layer 24: Kernel Methods (RKHS)
from ontic.genesis.rkhs import (
    Kernel,
    RBFKernel,
    MaternKernel,
    PolynomialKernel,
    LinearKernel,
    PeriodicKernel,
    QTTKernelMatrix,
    kernel_matrix,
    GPPrior,
    GPPosterior,
    GPRegressor,
    SparseGP,
    kernel_ridge_regression,
    KernelRidgeRegressor,
    maximum_mean_discrepancy,
    mmd_test,
)

# Layer 25: Persistent Homology
from ontic.genesis.topology import (
    Simplex,
    SimplicialComplex,
    RipsComplex,
    VietorisRips,
    CechComplex,
    boundary_matrix,
    coboundary_matrix,
    QTTBoundaryOperator,
    PersistencePair,
    PersistenceDiagram,
    compute_persistence,
    bottleneck_distance,
    wasserstein_distance_diagram,
    persistence_landscape,
)

# Layer 26: Geometric Algebra
from ontic.genesis.ga import (
    CliffordAlgebra,
    Multivector,
    scalar,
    vector,
    bivector,
    pseudoscalar,
    geometric_product,
    inner_product,
    outer_product,
    reverse,
    magnitude,
    normalize,
    inverse,
    rotor_from_bivector,
    rotor_from_angle_plane,
    apply_rotor,
    ConformalGA,
    point_to_cga,
    cga_to_point,
    QTTMultivector,
    qtt_geometric_product,
)

# Layer 27: Biological Aging (QTT-Aging)
from ontic.genesis.aging import (
    # Cell state
    CellStateTensor,
    ModeSpec,
    AgingSignature,
    CellType,
    AgingHallmark,
    BiologicalMode,
    young_cell,
    aged_cell,
    embryonic_stem_cell,
    YAMANAKA_FACTORS,
    THOMSON_FACTORS,
    NUM_PROTEIN_CODING_GENES,
    NUM_PROTEINS,
    NUM_CPG_SITES,
    # Dynamics
    AgingOperator,
    AgingRateModel,
    AgingTrajectory,
    ModePerturbation,
    # Epigenetic clocks
    HorvathClock,
    GrimAgeClock,
    MethylationState,
    extract_methylation,
    young_methylation,
    aged_methylation,
    HORVATH_TOP_SITES,
    # Interventions
    InterventionResult,
    YamanakaOperator,
    PartialReprogrammingOperator,
    SenolyticOperator,
    CalorieRestrictionOperator,
    CombinationIntervention,
    find_optimal_intervention,
    # Topology
    AgingTopologyAnalyzer,
    AgingTopology,
    AgingPhase,
    TopologicalBarrier,
    RejuvenationPath,
    compute_rejuvenation_path,
)

__all__ = [
    # Core Infrastructure
    "get_logger",
    "configure_logging",
    "GenesisLogger",
    "LogLevel",
    "GenesisError",
    "QTTRankError",
    "ConvergenceError",
    "DimensionMismatchError",
    "NumericalInstabilityError",
    "MemoryBudgetExceededError",
    "InvalidInputError",
    "CompressionError",
    "profile",
    "profile_memory",
    "timed",
    "traced",
    "ProfileResult",
    "PerformanceTracker",
    "validate_qtt_cores",
    "validate_tensor_shape",
    "validate_positive",
    "validate_probability",
    "check_numerical_stability",
    # Layer 20: Optimal Transport
    "QTTSinkhorn",
    "QTTDistribution", 
    "wasserstein_distance",
    "transport_plan",
    "barycenter",
    # Layer 21: Spectral Graph Wavelets
    "QTTLaplacian",
    "QTTSignal",
    "QTTGraphWavelet",
    "ChebyshevApproximator",
    "LowPassFilter",
    "HighPassFilter",
    "BandPassFilter",
    # Layer 22: Random Matrix Theory
    "QTTEnsemble",
    "QTTResolvent",
    "SpectralDensity",
    "WignerSemicircle",
    "MarchenkoPastur",
    "FreeConvolution",
    "spectral_density",
    "stieltjes_transform",
    "resolvent_trace",
    # Layer 23: Tropical Geometry
    "TropicalSemiring",
    "MinPlusSemiring",
    "MaxPlusSemiring",
    "TropicalMatrix",
    "tropical_matmul",
    "tropical_power",
    "tropical_kleene_star",
    "all_pairs_shortest_path",
    "floyd_warshall_tropical",
    "bellman_ford_tropical",
    "tropical_eigenvalue",
    # Layer 24: Kernel Methods
    "Kernel",
    "RBFKernel",
    "MaternKernel",
    "PolynomialKernel",
    "LinearKernel",
    "PeriodicKernel",
    "QTTKernelMatrix",
    "kernel_matrix",
    "GPPrior",
    "GPPosterior",
    "GPRegressor",
    "SparseGP",
    "kernel_ridge_regression",
    "KernelRidgeRegressor",
    "maximum_mean_discrepancy",
    "mmd_test",
    # Layer 25: Persistent Homology
    "Simplex",
    "SimplicialComplex",
    "RipsComplex",
    "VietorisRips",
    "CechComplex",
    "boundary_matrix",
    "coboundary_matrix",
    "QTTBoundaryOperator",
    "PersistencePair",
    "PersistenceDiagram",
    "compute_persistence",
    "bottleneck_distance",
    "wasserstein_distance_diagram",
    "persistence_landscape",
    # Layer 26: Geometric Algebra
    "CliffordAlgebra",
    "Multivector",
    "scalar",
    "vector",
    "bivector",
    "pseudoscalar",
    "geometric_product",
    "inner_product",
    "outer_product",
    "reverse",
    "magnitude",
    "normalize",
    "inverse",
    "rotor_from_bivector",
    "rotor_from_angle_plane",
    "apply_rotor",
    "ConformalGA",
    "point_to_cga",
    "cga_to_point",
    "QTTMultivector",
    "qtt_geometric_product",
    # Layer 27: Biological Aging
    "CellStateTensor",
    "ModeSpec",
    "AgingSignature",
    "CellType",
    "AgingHallmark",
    "BiologicalMode",
    "young_cell",
    "aged_cell",
    "embryonic_stem_cell",
    "YAMANAKA_FACTORS",
    "THOMSON_FACTORS",
    "NUM_PROTEIN_CODING_GENES",
    "NUM_PROTEINS",
    "NUM_CPG_SITES",
    "AgingOperator",
    "AgingRateModel",
    "AgingTrajectory",
    "ModePerturbation",
    "HorvathClock",
    "GrimAgeClock",
    "MethylationState",
    "extract_methylation",
    "young_methylation",
    "aged_methylation",
    "HORVATH_TOP_SITES",
    "InterventionResult",
    "YamanakaOperator",
    "PartialReprogrammingOperator",
    "SenolyticOperator",
    "CalorieRestrictionOperator",
    "CombinationIntervention",
    "find_optimal_intervention",
    "AgingTopologyAnalyzer",
    "AgingTopology",
    "AgingPhase",
    "TopologicalBarrier",
    "RejuvenationPath",
    "compute_rejuvenation_path",
]
