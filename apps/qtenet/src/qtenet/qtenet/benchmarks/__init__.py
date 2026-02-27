"""
QTeneT Benchmarks — Quantify Curse-Breaking

Benchmarks that PROVE O(log N) scaling advantage.

The curse_of_dimensionality benchmark is the flagship — it demonstrates
that QTT achieves exponential compression while maintaining accuracy.

Example:
    >>> from qtenet.benchmarks import curse_of_dimensionality
    >>> 
    >>> # Run scaling analysis
    >>> results = curse_of_dimensionality(
    ...     dims=[3, 4, 5, 6],
    ...     qubits_per_dim=5,
    ...     max_rank=64,
    ... )
    >>> 
    >>> # Show compression ratios
    >>> for r in results:
    ...     print(f"{r.dims}D: {r.compression_ratio:,.0f}× compression")

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from qtenet.benchmarks.curse_scaling import (
    curse_of_dimensionality,
    dimension_scaling,
    rank_scaling,
    CurseScalingResult,
    BenchmarkSuite,
)

__all__ = [
    "curse_of_dimensionality",
    "dimension_scaling",
    "rank_scaling",
    "CurseScalingResult",
    "BenchmarkSuite",
]
