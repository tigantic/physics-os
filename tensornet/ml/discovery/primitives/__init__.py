"""
PRIMITIVES — Genesis Primitive Wrappers for Autonomous Discovery

This module provides discovery-aware wrappers around the raw Genesis primitives.
Each wrapper implements the GenesisPrimitive protocol with:
    - process(): Main computation using underlying QTT implementation
    - detect_anomalies(): Primitive-specific anomaly detection
    - detect_invariants(): Conservation law discovery
    - detect_bottlenecks(): Resource constraint identification
    - predict(): Pattern-based prediction

Primitives:
    - OptimalTransportPrimitive: Distribution matching (Layer 20)
    - SpectralWaveletPrimitive: Multi-scale analysis (Layer 21)
    - RandomMatrixPrimitive: Spectral statistics (Layer 22)
    - KernelPrimitive: RKHS embeddings (Layer 24)
    - TopologyPrimitive: Persistent homology (Layer 25)
    - GeometricAlgebraPrimitive: Geometric transformations (Layer 26)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from tensornet.ml.discovery.primitives.optimal_transport import OptimalTransportPrimitive
from tensornet.ml.discovery.primitives.spectral_wavelets import SpectralWaveletPrimitive
from tensornet.ml.discovery.primitives.random_matrix import RandomMatrixPrimitive
from tensornet.ml.discovery.primitives.kernel import KernelPrimitive
from tensornet.ml.discovery.primitives.topology import TopologyPrimitive
from tensornet.ml.discovery.primitives.geometric_algebra import GeometricAlgebraPrimitive

__all__ = [
    "OptimalTransportPrimitive",
    "SpectralWaveletPrimitive",
    "RandomMatrixPrimitive",
    "KernelPrimitive",
    "TopologyPrimitive",
    "GeometricAlgebraPrimitive",
]
