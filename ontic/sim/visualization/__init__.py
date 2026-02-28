"""
Ontic Visualization Module
================================

Decompression-free rendering and visualization for Tensor Trains.
"""

from .tensor_slicer import TensorSlicer, create_sine_qtt, create_slicer_from_qtt, create_test_qtt

__all__ = [
    "TensorSlicer",
    "create_slicer_from_qtt",
    "create_test_qtt",
    "create_sine_qtt",
]
