"""
TensorNet GPU Module
====================

OPERATION VALHALLA - Phase 2: THE MUSCLE

GPU-accelerated tensor operations using PyTorch CUDA.
All physics computation executes on RTX 5070 Tensor Cores.

Modules:
    - tensor_field: GPU-resident field storage and operations
    - fluid_dynamics: Navier-Stokes solver on CUDA
    - memory: VRAM management and allocation pool
"""

from .tensor_field import TensorField

__all__ = ['TensorField']
