"""
Projected Activations for Tensor Networks
==========================================

Applies non-linearities without full decompression.

This is the SIMPLIFIED version using ProjectedActivation.
The complex FunctionApproximator with TT-Cross was deemed "too slow"
and replaced with this direct core transformation approach.

Mathematical Note:
    ProjectedActivation applies f(core) for each core tensor.
    This is NOT equivalent to element-wise f(full_tensor), but
    works as a valid non-linearity in compressed feature space.

Constitutional Compliance:
    - Article V.5.1: All public classes/functions documented
    - Article VII.7.2: Definition of Done = USER-OBSERVABLE BEHAVIOR works
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fluidelite.core.mps import MPS


class ProjectedActivation(nn.Module):
    """
    Applies a non-linearity to an MPS without full decompression.
    
    Method: Applies activation function to core tensors directly,
    treating bond indices as feature channels.
    
    After activation, optionally re-compresses via truncation since non-linearities
    increase the effective rank (entropy of singular values increases).
    
    Args:
        activation_fn: Activation function (default: F.gelu)
        max_rank: Maximum bond dimension after re-compression
        skip_truncation: If True, caller handles truncation (for fused ops)
        
    Example:
        >>> act = ProjectedActivation(F.gelu, max_rank=64)
        >>> mps_out = act(mps_in)
    """
    def __init__(self, activation_fn=F.gelu, max_rank: int = 64, skip_truncation: bool = False):
        super().__init__()
        self.fn = activation_fn
        self.max_rank = max_rank
        self.skip_truncation = skip_truncation

    def forward(self, mps: MPS) -> MPS:
        """
        Apply activation to MPS.
        
        Args:
            mps: Input MPS |x⟩
            
        Returns:
            Output MPS |x'⟩ ≈ f(|x⟩)
        """
        new_cores = []
        
        # Apply function to each core tensor
        # This increases the "virtual" rank complexity because f(A*B) != f(A)*f(B)
        # But locally, it transforms the features in a learnable way
        for core in mps.tensors:
            new_cores.append(self.fn(core))
            
        res = MPS(new_cores)
        
        # Re-Compress: Non-linearities increase singular values' entropy
        # so we need to truncate back to target rank
        # Skip if caller will handle (for fused operations)
        if not self.skip_truncation:
            # Use batched truncation with STE for efficiency and gradient stability
            res.truncate_batched_ste_(chi_max=self.max_rank)
            res.normalize_()
        
        return res


def gelu_mps(mps: MPS, rank: int | None = None) -> MPS:
    """
    Helper for GELU activation on MPS.
    
    Args:
        mps: Input MPS
        rank: Maximum rank after activation (default: mps.chi)
        
    Returns:
        MPS with GELU applied
        
    Example:
        >>> mps = MPS.random(L=10, d=2, chi=32)
        >>> mps_gelu = gelu_mps(mps, rank=32)
    """
    op = ProjectedActivation(F.gelu, max_rank=rank if rank else mps.chi)
    return op(mps)


def relu_mps(mps: MPS, rank: int | None = None) -> MPS:
    """
    Helper for ReLU activation on MPS.
    
    Args:
        mps: Input MPS
        rank: Maximum rank after activation (default: mps.chi)
        
    Returns:
        MPS with ReLU applied
    """
    op = ProjectedActivation(F.relu, max_rank=rank if rank else mps.chi)
    return op(mps)


def tanh_mps(mps: MPS, rank: int | None = None) -> MPS:
    """
    Helper for tanh activation on MPS.
    
    Args:
        mps: Input MPS
        rank: Maximum rank after activation (default: mps.chi)
        
    Returns:
        MPS with tanh applied
    """
    op = ProjectedActivation(torch.tanh, max_rank=rank if rank else mps.chi)
    return op(mps)
