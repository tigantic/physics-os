"""
TENSOR GENESIS — QTT Meta-Primitive Expansion Protocol

This module extends Quantized Tensor Train (QTT) compression into seven
unexploited mathematical domains, creating capabilities that are IMPOSSIBLE
without QTT compression.

Constitutional Reference: TENSOR_GENESIS.md

Layers 20-26:
    - ot: Optimal Transport (Layer 20)
    - sgw: Spectral Graph Wavelets (Layer 21)
    - rmt: Random Matrix Theory (Layer 22)
    - tropical: Tropical Geometry (Layer 23)
    - rkhs: Kernel Methods (Layer 24)
    - topology: Persistent Homology (Layer 25)
    - ga: Geometric Algebra (Layer 26)

Example:
    >>> from tensornet.genesis import QTTSinkhorn, wasserstein_distance
    >>> from tensornet.genesis.ot import QTTDistribution
    >>> 
    >>> mu = QTTDistribution.gaussian(mean=0.0, std=1.0, grid_size=2**40)
    >>> nu = QTTDistribution.gaussian(mean=1.0, std=1.0, grid_size=2**40)
    >>> W2 = wasserstein_distance(mu, nu, p=2)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

__version__ = "1.0.0"
__author__ = "Bradly Biron Baker Adams"
__constitutional_ref__ = "TENSOR_GENESIS.md"

# Layer 20: Optimal Transport
from tensornet.genesis.ot import (
    QTTSinkhorn,
    QTTDistribution,
    wasserstein_distance,
    transport_plan,
    barycenter,
)

# Layer 21: Spectral Graph Wavelets
from tensornet.genesis.sgw import (
    QTTLaplacian,
    QTTSignal,
    QTTGraphWavelet,
    ChebyshevApproximator,
    LowPassFilter,
    HighPassFilter,
    BandPassFilter,
)

# Layer 22: Random Matrix Theory
from tensornet.genesis.rmt import (
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
from tensornet.genesis.tropical import (
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
from tensornet.genesis.rkhs import (
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
from tensornet.genesis.topology import (
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
from tensornet.genesis.ga import (
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
