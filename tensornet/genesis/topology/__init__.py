"""
QTT-PH: Persistent Homology in Quantized Tensor Train Format

TENSOR GENESIS Protocol — Layer 25

Persistent Homology captures topological features (connected components,
loops, voids) across multiple scales via filtration of simplicial complexes.

QTT Insight: The boundary operator ∂_k has sparse structure that admits
low-rank TT representation. For structured point clouds (lattices, manifolds),
the persistence diagram can be computed in O(r³ log N) time.

Key operations:
    - Simplicial complexes in TT format
    - Boundary operators ∂_k
    - Persistent homology via reduction
    - Wasserstein distance between persistence diagrams

Mathematical Foundation:
    - Betti numbers: β_k = dim(H_k) = dim(ker ∂_k / im ∂_{k+1})
    - Persistence pairs: (birth, death) for each topological feature
    - Bottleneck/Wasserstein distances on persistence diagrams

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from tensornet.genesis.topology.simplicial import (
    Simplex,
    SimplicialComplex,
    RipsComplex,
    VietorisRips,
    CechComplex,
)

from tensornet.genesis.topology.boundary import (
    boundary_matrix,
    QTTBoundaryOperator,
    coboundary_matrix,
)

from tensornet.genesis.topology.persistence import (
    PersistencePair,
    PersistenceDiagram,
    compute_persistence,
    reduce_boundary_matrix,
    persistence_pairs,
)

from tensornet.genesis.topology.distances import (
    bottleneck_distance,
    wasserstein_distance_diagram,
    persistence_landscape,
)

# QTT-Native implementation (true trillion-scale without densification)
from tensornet.genesis.topology.qtt_native import (
    QTTVector,
    QTTMatrix,
    QTTBoundaryMatrix,
    QTTPersistenceResult,
    QTTRipsComplex,
    qtt_persistence_grid_1d,
    qtt_betti_numbers_grid,
)

__all__ = [
    # Simplicial complexes
    "Simplex",
    "SimplicialComplex",
    "RipsComplex",
    "VietorisRips",
    "CechComplex",
    # Boundary operators
    "boundary_matrix",
    "QTTBoundaryOperator",
    # QTT-Native (true trillion-scale)
    "QTTVector",
    "QTTMatrix",
    "QTTBoundaryMatrix",
    "QTTPersistenceResult",
    "QTTRipsComplex",
    "qtt_persistence_grid_1d",
    "qtt_betti_numbers_grid",
    "coboundary_matrix",
    # Persistence
    "PersistencePair",
    "PersistenceDiagram",
    "compute_persistence",
    "reduce_boundary_matrix",
    "persistence_pairs",
    # Distances
    "bottleneck_distance",
    "wasserstein_distance_diagram",
    "persistence_landscape",
]
