"""
QTT-Optimal Transport (QTT-OT) — Layer 20 of the Ontic Engine Capability Stack

This module implements Optimal Transport algorithms in Quantized Tensor Train
format, enabling trillion-point distribution matching on commodity hardware.

Constitutional Reference: TENSOR_GENESIS.md, Part II, Primitive 1

Key Insight:
    The cost matrix C for grid-based distributions has Toeplitz structure,
    giving it TT rank O(1). The Gibbs kernel K = exp(-C/ε) inherits this
    low-rank structure. Sinkhorn iterations become MPO×MPS operations
    with O(r³ log N) complexity instead of O(N²).

Key Components:
    - QTTSinkhorn: Sinkhorn algorithm in QTT format
    - SinkhornResult: Result container for Sinkhorn solutions
    - QTTDistribution: Probability distributions in QTT format
    - QTTMatrix: Matrix Product Operators for cost matrices
    - QTTTransportPlan: Transport plan extraction and analysis
    - wasserstein_distance: High-level distance computation
    - transport_plan: Extract optimal transport plans
    - barycenter: Wasserstein barycenters

Cost Matrix Constructors:
    - euclidean_cost_mpo: Squared Euclidean distance
    - gaussian_kernel_mpo: Gibbs kernel exp(-C/ε)

Example:
    >>> from ontic.genesis.ot import QTTSinkhorn, QTTDistribution
    >>> mu = QTTDistribution.gaussian(mean=0.0, std=1.0, grid_size=2**40)
    >>> nu = QTTDistribution.gaussian(mean=1.0, std=1.0, grid_size=2**40)
    >>> solver = QTTSinkhorn(epsilon=0.01)
    >>> result = solver.solve(mu, nu)
    >>> print(f"Wasserstein distance: {result.wasserstein_distance:.6f}")

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

# Core Sinkhorn solver
from ontic.genesis.ot.sinkhorn_qtt import QTTSinkhorn, SinkhornResult, sinkhorn_distance

# Distribution representations
from ontic.genesis.ot.distributions import QTTDistribution

# Cost matrix representations
from ontic.genesis.ot.cost_matrices import (
    QTTMatrix,
    euclidean_cost_mpo,
    gaussian_kernel_mpo,
    toeplitz_cost_mpo,
    custom_cost_mpo,
)

# High-level distance API
from ontic.genesis.ot.wasserstein import wasserstein_distance, wasserstein_barycenter

# Transport plan extraction
from ontic.genesis.ot.transport_plan import QTTTransportPlan, transport_plan, monge_map

# Barycenter computation
from ontic.genesis.ot.barycenters import (
    barycenter,
    BarycenterResult,
    interpolate,
    geodesic,
)

__all__ = [
    # Core classes
    "QTTSinkhorn",
    "SinkhornResult",
    "QTTDistribution",
    "QTTMatrix",
    "QTTTransportPlan",
    "BarycenterResult",
    # Solver functions
    "sinkhorn_distance",
    "wasserstein_distance",
    "wasserstein_barycenter",
    # Cost matrices
    "euclidean_cost_mpo",
    "gaussian_kernel_mpo",
    "toeplitz_cost_mpo",
    "custom_cost_mpo",
    # Transport analysis
    "transport_plan",
    "monge_map",
    # Barycenters
    "barycenter",
    "interpolate",
    "geodesic",
]

__version__ = "1.0.0"
__layer__ = 20
__constitutional_ref__ = "TENSOR_GENESIS.md"
