"""
QTT-TG: Tropical Geometry in Quantized Tensor Train Format

TENSOR GENESIS Protocol — Layer 23

Tropical geometry replaces classical arithmetic with tropical semirings:
    Min-plus: (⊕, ⊗) = (min, +)
    Max-plus: (⊕, ⊗) = (max, +)

This transforms shortest-path problems into matrix operations:
    (A ⊗ B)_ij = min_k (A_ik + B_kj)

QTT Insight: Distance matrices have low TT rank due to smoothness.
The tropical product approximated via softmin preserves low rank:
    min(a, b) ≈ -(1/β) log(e^{-βa} + e^{-βb})

Complexity:
    - Standard all-pairs shortest path: O(N³)
    - QTT-TG tropical power: O(r³ log² N)

Applications:
    - Shortest paths on trillion-node graphs
    - Global optimization without gradients
    - Tropical linear programming
    - Algebraic geometry at scale

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from ontic.genesis.tropical.semiring import (
    TropicalSemiring,
    MinPlusSemiring,
    MaxPlusSemiring,
    tropical_min,
    tropical_max,
    softmin,
    softmax,
    tropical_add,
    tropical_mul,
)

from ontic.genesis.tropical.matrix import (
    TropicalMatrix,
    tropical_matmul,
    tropical_power,
    tropical_kleene_star,
)

from ontic.genesis.tropical.shortest_path import (
    all_pairs_shortest_path,
    single_source_shortest_path,
    floyd_warshall_tropical,
    bellman_ford_tropical,
    shortest_path_tree,
)

from ontic.genesis.tropical.convexity import (
    TropicalPolyhedron,
    TropicalHalfspace,
    tropical_convex_hull,
    is_tropically_convex,
)

from ontic.genesis.tropical.optimization import (
    tropical_linear_program,
    tropical_eigenvector,
    tropical_eigenvalue,
)

# QTT-Native implementation (true trillion-scale without densification)
from ontic.genesis.tropical.qtt_native import (
    QTTCore,
    QTTTropicalMatrix,
    qtt_tropical_matmul,
    qtt_floyd_warshall,
)

__all__ = [
    # Semiring operations
    "TropicalSemiring",
    "MinPlusSemiring", 
    "MaxPlusSemiring",
    "tropical_min",
    "tropical_max",
    "softmin",
    "softmax",
    "tropical_add",
    "tropical_mul",
    # Matrix operations
    "TropicalMatrix",
    "tropical_matmul",
    "tropical_power",
    "tropical_kleene_star",
    # Shortest paths
    "all_pairs_shortest_path",
    "single_source_shortest_path",
    "floyd_warshall_tropical",
    "bellman_ford_tropical",
    "shortest_path_tree",
    # Convexity
    "TropicalPolyhedron",
    "TropicalHalfspace",
    "tropical_convex_hull",
    "is_tropically_convex",
    # Optimization
    "tropical_linear_program",
    "tropical_eigenvector",
    "tropical_eigenvalue",
    # QTT-Native (true trillion-scale)
    "QTTCore",
    "QTTTropicalMatrix",
    "qtt_tropical_matmul",
    "qtt_floyd_warshall",
]
