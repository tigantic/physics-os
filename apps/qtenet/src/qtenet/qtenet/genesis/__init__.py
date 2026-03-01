"""
QTeneT Genesis — QTT Meta-Primitive Expansion Protocol

Genesis extends QTT compression into seven mathematical domains that are
IMPOSSIBLE without QTT — each layer enables new curse-breaking capabilities.

The 7 Genesis Layers:
    Layer 20: Optimal Transport (OT)     — Trillion-point Wasserstein distances
    Layer 21: Spectral Graph Wavelets    — Graph signal processing at scale
    Layer 22: Random Matrix Theory       — Spectral analysis beyond eigensolvers
    Layer 23: Tropical Geometry          — Shortest paths without iteration
    Layer 24: RKHS / Kernel Methods      — Kernel matrices that don't fit in memory
    Layer 25: Persistent Homology        — Topological data analysis at scale
    Layer 26: Geometric Algebra          — Physics-native representations

Each layer demonstrates curse-breaking on a NEW mathematical primitive.

Example:
    >>> from qtenet.genesis import ot, sgw, rmt
    >>> 
    >>> # Optimal Transport on 2^40 points (trillion-scale)
    >>> mu = ot.QTTDistribution.gaussian(mean=0.0, std=1.0, grid_size=2**40)
    >>> nu = ot.QTTDistribution.gaussian(mean=1.0, std=1.5, grid_size=2**40)
    >>> W2 = ot.wasserstein_distance(mu, nu, p=2)
    >>> 
    >>> # Random Matrix spectral density
    >>> density = rmt.spectral_density(matrix_size=2**20, sample_points=1000)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

# Re-export all Genesis layers from upstream ontic
from ontic.genesis import (
    # Core Infrastructure
    get_logger,
    configure_logging,
    GenesisLogger,
    LogLevel,
    GenesisError,
    QTTRankError,
    ConvergenceError,
    DimensionMismatchError,
    NumericalInstabilityError,
    MemoryBudgetExceededError,
    InvalidInputError,
    CompressionError,
    profile,
    profile_memory,
    timed,
    traced,
    ProfileResult,
    PerformanceTracker,
    validate_qtt_cores,
    validate_tensor_shape,
    validate_positive,
    validate_probability,
    check_numerical_stability,
)

# Layer 20: Optimal Transport
from ontic.genesis import ot

# Layer 21: Spectral Graph Wavelets
from ontic.genesis import sgw

# Layer 22: Random Matrix Theory
from ontic.genesis import rmt

# Layer 23: Tropical Geometry
from ontic.genesis import tropical

# Layer 24: Kernel Methods (RKHS)
from ontic.genesis import rkhs

# Layer 25: Persistent Homology
from ontic.genesis import topology

# Layer 26: Geometric Algebra
from ontic.genesis import ga


# Direct symbol exports for convenience
from ontic.genesis import (
    # Layer 20: Optimal Transport
    QTTSinkhorn,
    QTTDistribution,
    wasserstein_distance,
    transport_plan,
    barycenter,
    # Layer 21: Spectral Graph Wavelets
    QTTLaplacian,
    QTTSignal,
    QTTGraphWavelet,
    ChebyshevApproximator,
    LowPassFilter,
    HighPassFilter,
    BandPassFilter,
    # Layer 22: Random Matrix Theory
    QTTEnsemble,
    QTTResolvent,
    SpectralDensity,
    WignerSemicircle,
    MarchenkoPastur,
    FreeConvolution,
    spectral_density,
    stieltjes_transform,
    resolvent_trace,
    # Layer 23: Tropical Geometry
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
    # Layer 24: Kernel Methods
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
    # Layer 25: Persistent Homology
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
    # Layer 26: Geometric Algebra
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


__all__ = [
    # Submodules (recommended access pattern)
    "ot",
    "sgw",
    "rmt",
    "tropical",
    "rkhs",
    "topology",
    "ga",
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
]
