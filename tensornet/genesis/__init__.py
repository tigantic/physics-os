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

__all__ = [
    # Layer 20: Optimal Transport
    "QTTSinkhorn",
    "QTTDistribution", 
    "wasserstein_distance",
    "transport_plan",
    "barycenter",
]
