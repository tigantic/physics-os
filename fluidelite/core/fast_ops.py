"""
Vectorized Tensor Operations
============================

Eliminates Python loops for GPU acceleration.
Processes entire MPS/MPO chains in single kernel calls.

Constitutional Compliance:
    - Article V.5.1: All public functions documented
    - Article VII.7.2: Definition of Done = USER-OBSERVABLE BEHAVIOR works
    
BUG FIX APPLIED: Original einsum had space in output string.
"""

import torch
from torch import Tensor


def vectorized_mpo_apply(mps_cores: Tensor, mpo_cores: Tensor) -> Tensor:
    """
    Applies an MPO to an MPS in a single fused kernel.
    
    This eliminates the Python loop over sites, processing all L sites
    simultaneously on GPU for maximum throughput.
    
    Args:
        mps_cores: Stacked MPS tensors (L, Chi_l, d, Chi_r)
        mpo_cores: Stacked MPO tensors (L, D_l, d_out, d_in, D_r)
        
    Returns:
        new_cores: Stacked tensors (L, Chi*D_l, d_out, Chi*D_r)
        
    Note:
        Assumes uniform bond dimensions (padded if necessary).
        For non-uniform bonds, use the loop-based MPO.apply() method.
        
    Example:
        >>> mps = torch.randn(12, 32, 2, 32)  # 12 sites, chi=32, d=2
        >>> mpo = torch.randn(12, 4, 2, 2, 4)  # 12 sites, D=4, d=2
        >>> result = vectorized_mpo_apply(mps, mpo)
        >>> assert result.shape == (12, 128, 2, 128)  # chi*D = 32*4 = 128
    """
    # Dimensions:
    # L: Sites
    # C_l, C_r: MPS Bond Dims (using c for MPS bonds)
    # D_l, D_r: MPO Bond Dims (using d for MPO bonds, but avoiding conflict)
    # i: Physical In
    # o: Physical Out
    
    # Contract physical index 'i' simultaneously across all L sites.
    # mpo_cores: (L, D_l, d_out, d_in, D_r) -> indices: l, a, o, i, b
    # mps_cores: (L, C_l, d_in, C_r) -> indices: l, c, i, d
    # Output: (L, D_l, d_out, D_r, C_l, C_r) -> indices: l, a, o, b, c, d
    # BUG FIX: Original had 'ldoid, lcic -> ldocd c' (space in output)
    # Corrected einsum: contract over 'i' (physical in dimension)
    T = torch.einsum('laoib,lcid->laobcd', mpo_cores, mps_cores)
    
    # Get dimensions
    L, D_l, o, D_r, C_l, C_r = T.shape
    
    # Permute to (L, D_l, C_l, o, D_r, C_r) then reshape to (L, D_l*C_l, o, D_r*C_r)
    # This groups the bond dimensions correctly
    T = T.permute(0, 1, 4, 2, 3, 5).reshape(L, D_l * C_l, o, D_r * C_r)
    
    return T


def vectorized_mps_add(mps_a: Tensor, mps_b: Tensor) -> Tensor:
    """
    Adds two MPS stacks (Direct Sum) without Python loops.
    
    The direct sum of two MPS is a block-diagonal structure where
    the resulting bond dimension is R_a + R_b.
    
    Args:
        mps_a: First MPS stack (L, Chi_a_l, d, Chi_a_r)
        mps_b: Second MPS stack (L, Chi_b_l, d, Chi_b_r)
        
    Returns:
        Combined MPS stack (L, Chi_a_l + Chi_b_l, d, Chi_a_r + Chi_b_r)
        
    Note:
        Boundary cores (first/last sites) need special handling in caller
        to properly connect the left/right boundary conditions.
        
    Example:
        >>> mps_a = torch.randn(10, 16, 2, 16)
        >>> mps_b = torch.randn(10, 8, 2, 8)
        >>> result = vectorized_mps_add(mps_a, mps_b)
        >>> assert result.shape == (10, 24, 2, 24)  # 16+8 = 24
    """
    La, Cai, d_a, Caj = mps_a.shape
    Lb, Cbi, d_b, Cbj = mps_b.shape
    
    assert La == Lb, f"MPS must have same number of sites: {La} != {Lb}"
    assert d_a == d_b, f"MPS must have same physical dimension: {d_a} != {d_b}"
    
    L = La
    d = d_a
    
    C_new_l = Cai + Cbi
    C_new_r = Caj + Cbj
    
    # Allocate output tensor
    res = torch.zeros(L, C_new_l, d, C_new_r, dtype=mps_a.dtype, device=mps_a.device)
    
    # Block Diagonal Scatter: place A in top-left, B in bottom-right
    res[:, :Cai, :, :Caj] = mps_a
    res[:, Cai:, :, Caj:] = mps_b
    
    return res


def pad_mps_to_uniform(tensors: list[Tensor], target_chi: int) -> Tensor:
    """
    Pad a list of MPS tensors to uniform bond dimension and stack.
    
    This is required for vectorized operations which need uniform shapes.
    
    Args:
        tensors: List of MPS tensors with potentially varying shapes
        target_chi: Target bond dimension (pads with zeros)
        
    Returns:
        Stacked tensor (L, target_chi, d, target_chi)
        
    Example:
        >>> tensors = [torch.randn(1, 2, 4), torch.randn(4, 2, 4), torch.randn(4, 2, 1)]
        >>> stacked = pad_mps_to_uniform(tensors, target_chi=8)
        >>> assert stacked.shape == (3, 8, 2, 8)
    """
    L = len(tensors)
    d = tensors[0].shape[1]
    device = tensors[0].device
    dtype = tensors[0].dtype
    
    stacked = torch.zeros(L, target_chi, d, target_chi, dtype=dtype, device=device)
    
    for i, t in enumerate(tensors):
        chi_l, _, chi_r = t.shape
        stacked[i, :chi_l, :, :chi_r] = t
        
    return stacked


def unpad_mps_from_uniform(stacked: Tensor, original_shapes: list[tuple]) -> list[Tensor]:
    """
    Extract original non-uniform MPS tensors from padded stack.
    
    Args:
        stacked: Stacked tensor (L, chi_max, d, chi_max)
        original_shapes: List of original (chi_l, d, chi_r) shapes
        
    Returns:
        List of tensors with original shapes
        
    Example:
        >>> shapes = [(1, 2, 4), (4, 2, 4), (4, 2, 1)]
        >>> stacked = torch.randn(3, 8, 2, 8)
        >>> tensors = unpad_mps_from_uniform(stacked, shapes)
        >>> assert tensors[0].shape == (1, 2, 4)
    """
    tensors = []
    for i, (chi_l, d, chi_r) in enumerate(original_shapes):
        tensors.append(stacked[i, :chi_l, :, :chi_r].clone())
    return tensors
