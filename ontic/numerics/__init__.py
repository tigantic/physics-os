"""
Numerics module for The Ontic Engine.

Provides rigorous numerical methods for Computer-Assisted Proofs:
- Interval arithmetic for error tracking
- High-precision operations
"""

from ontic.numerics.interval import Interval, IntervalTensor, validate_interval_arithmetic

__all__ = [
    "Interval",
    "IntervalTensor",
    "validate_interval_arithmetic",
]
