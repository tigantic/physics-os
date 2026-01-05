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
    - advection: CUDA-accelerated Semi-Lagrangian advection (30x speedup)
"""

from .advection import advect_2d, advect_3d, advect_velocity_2d, is_cuda_available, print_gpu_status
from .tensor_field import TensorField

__all__ = [
    "TensorField",
    "advect_2d",
    "advect_velocity_2d",
    "advect_3d",
    "is_cuda_available",
    "print_gpu_status",
]
