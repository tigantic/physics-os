# Copyright (c) 2025 Tigantic
# Phase 18: Adaptive Bond Dimension Optimization
"""
Adaptive bond dimension management for tensor network simulations.

This module provides intelligent adaptive truncation strategies that dynamically
adjust bond dimensions during time evolution to balance accuracy and computational
cost. Key features include:

- Real-time entanglement entropy monitoring
- Area law validation and scaling analysis
- Multiple compression strategies (SVD, randomized, variational)
- Automatic bond dimension adaptation based on truncation error targets
"""

from .bond_optimizer import (
                             AdaptiveBondConfig,
                             AdaptiveTruncator,
                             BondDimensionTracker,
                             EntropyMonitor,
                             TruncationScheduler,
                             TruncationStrategy,
                             adapt_during_evolution,
                             estimate_optimal_chi,
)
from .compression import (
                             CompressionMethod,
                             CompressionResult,
                             CompressionStrategy,
                             RandomizedSVD,
                             SVDCompression,
                             TensorCrossInterpolation,
                             VariationalCompression,
                             compress_adaptively,
                             select_compression_strategy,
)
from .entanglement import (
                             AreaLawAnalyzer,
                             AreaLawScaling,
                             EntanglementEntropy,
                             EntanglementSpectrum,
                             MutualInformation,
                             analyze_area_law,
                             compute_entanglement_entropy,
                             compute_mutual_information,
                             compute_schmidt_spectrum,
)

__all__ = [
    # Bond optimizer
    "AdaptiveBondConfig",
    "TruncationStrategy",
    "BondDimensionTracker",
    "TruncationScheduler",
    "EntropyMonitor",
    "AdaptiveTruncator",
    "estimate_optimal_chi",
    "adapt_during_evolution",
    # Entanglement analysis
    "EntanglementSpectrum",
    "AreaLawScaling",
    "AreaLawAnalyzer",
    "EntanglementEntropy",
    "MutualInformation",
    "compute_entanglement_entropy",
    "compute_mutual_information",
    "analyze_area_law",
    "compute_schmidt_spectrum",
    # Compression
    "CompressionMethod",
    "CompressionResult",
    "CompressionStrategy",
    "SVDCompression",
    "RandomizedSVD",
    "VariationalCompression",
    "TensorCrossInterpolation",
    "compress_adaptively",
    "select_compression_strategy",
]
