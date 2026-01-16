"""
FluidElite Core Module
======================

Contains MPS, MPO, decompositions, and fast operations.

Elite Engineering Complete: 5/5 Optimizations
    1. Fused Laplacian     - mps_sum with single truncation (~5×)
    2. CG Fusion           - mps_linear_combination (~2×)
    3. Jacobi Preconditioner - pcg_solve with M⁻¹=h²/4 (~2× fewer iters)
    4. Multigrid V-cycle   - O(1) vs O(√N) (~8× for large grids)
    5. CUDA Hybrid         - .cuda()/.cpu() methods (seamless GPU)
"""

from fluidelite.core.mps import MPS
from fluidelite.core.mpo import MPO
from fluidelite.core.decompositions import svd_truncated, qr_positive, rsvd_truncated, SafeSVD
from fluidelite.core.fast_ops import vectorized_mpo_apply, vectorized_mps_add
from fluidelite.core.cross import ProjectedActivation, gelu_mps
from fluidelite.core.elite_ops import (
    mps_sum,
    mps_linear_combination,
    pcg_solve,
    multigrid_preconditioner,
    multigrid_vcycle,
    patch_mps_cuda,
    batched_truncate_,
    batched_norm,
    fused_canonicalize_truncate_,
)

__all__ = [
    "MPS",
    "MPO",
    "svd_truncated",
    "qr_positive",
    "rsvd_truncated",
    "SafeSVD",
    "vectorized_mpo_apply",
    "vectorized_mps_add",
    "ProjectedActivation",
    "gelu_mps",
    # Elite optimizations
    "mps_sum",
    "mps_linear_combination",
    "pcg_solve",
    "multigrid_preconditioner",
    "multigrid_vcycle",
    "patch_mps_cuda",
    # Batched GPU optimizations
    "batched_truncate_",
    "batched_norm",
    "fused_canonicalize_truncate_",
]
