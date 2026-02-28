"""
QTT Reciprocal: Element-wise Inversion for Nonlinear Terms
===========================================================

Provides GPU-accelerated element-wise reciprocal computation in QTT format
using Newton-Schulz iteration with rSVD truncation.

The Newton-Schulz iteration for 1/x:
    y_{n+1} = y_n * (2 - x * y_n)
    
converges quadratically when 0 < x * y_0 < 2.

This is essential for Euler equation fluxes where division by density is required:
    F_rhou = rhou^2 / rho + P
    F_rhov = rhou * rhov / rho
    F_E = (E + P) * rhou / rho

Author: HyperTensor Team
Date: January 2026
Tag: [PHYSICS-TOOLBOX] [NONLINEAR]
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from ontic.cfd.qtt_hadamard import qtt_hadamard, qtt_hadamard_inplace_scale
from ontic.cfd.nd_shift_mpo import truncate_cores


def qtt_reciprocal(
    cores: List[Tensor],
    max_rank: int = 64,
    newton_iters: int = 5,
    eps: float = 1e-10,
    tol: float = 1e-8,
) -> List[Tensor]:
    """
    Compute element-wise reciprocal 1/x in QTT format.
    
    Uses Newton-Schulz iteration with GPU-accelerated rSVD truncation.
    
    Newton-Schulz: y_{n+1} = y_n * (2 - x * y_n)
    Converges quadratically when 0 < x * y_0 < 2.
    
    For stability:
    1. Estimate scale from QTT norm
    2. Normalize x to O(1)
    3. Iterate in normalized space
    4. Scale result back
    
    Args:
        cores: QTT cores of the input (should be positive)
        max_rank: Maximum TT rank for rSVD truncation
        newton_iters: Number of Newton-Schulz iterations
        eps: Regularization to avoid division by zero
        tol: rSVD truncation tolerance
        
    Returns:
        QTT cores approximating 1/x
        
    Example:
        >>> rho_cores = [...]  # QTT density field
        >>> rho_inv = qtt_reciprocal(rho_cores, max_rank=64)
        >>> # Now rho_inv ≈ 1/rho element-wise
    """
    n_cores = len(cores)
    device = cores[0].device
    dtype = cores[0].dtype
    
    # Estimate scale from Frobenius norm product
    scale = _estimate_qtt_scale(cores)
    scale = max(scale, eps)
    
    # Regularize: add small epsilon for stability
    x_reg = _add_constant(cores, eps)
    
    # Scale x to have elements roughly O(1)
    x_scaled = qtt_hadamard_inplace_scale(x_reg, 1.0 / scale)
    x_scaled = truncate_cores(x_scaled, max_rank, tol=tol)
    
    # Initial guess: y_0 = 1 (constant QTT)
    y = _constant_qtt_cores(1.0, n_cores, device, dtype)
    two = _constant_qtt_cores(2.0, n_cores, device, dtype)
    
    # Newton-Schulz iterations
    for _ in range(newton_iters):
        # xy = x_scaled * y (Hadamard product)
        xy = qtt_hadamard(x_scaled, y, max_rank=max_rank, tol=tol)
        
        # 2 - xy
        neg_xy = qtt_hadamard_inplace_scale(xy, -1.0)
        two_minus_xy = _add_qtt(two, neg_xy, max_rank=max_rank, tol=tol)
        
        # y = y * (2 - xy)
        y = qtt_hadamard(y, two_minus_xy, max_rank=max_rank, tol=tol)
        
        # Clamp cores to prevent numerical explosion
        y = _clamp_cores(y, max_val=1e6)
    
    # Scale back: 1/x = (1/x_scaled) / scale
    y = qtt_hadamard_inplace_scale(y, 1.0 / scale)
    
    return y


def _estimate_qtt_scale(cores: List[Tensor]) -> float:
    """Estimate typical magnitude of QTT vector from core norms."""
    n = len(cores)
    scale = 1.0
    for core in cores[:min(3, n)]:
        scale *= core.norm().item() ** (1.0 / n)
    return scale


def _constant_qtt_cores(value: float, n_cores: int, device: torch.device, dtype: torch.dtype) -> List[Tensor]:
    """Create QTT cores for a constant function f(i) = value."""
    cores = []
    # Product of all cores should give 'value'
    # Use value^(1/n) on diagonal
    val_per_core = abs(value) ** (1.0 / n_cores)
    sign = 1.0 if value >= 0 else -1.0
    
    for k in range(n_cores):
        if k == 0:
            # First core: (1, 2, 1) - sum over physical index gives value^(1/n)
            core = torch.ones(1, 2, 1, device=device, dtype=dtype) * val_per_core
            if k == 0:
                core *= sign  # Apply sign to first core only
        elif k == n_cores - 1:
            # Last core: (1, 2, 1)
            core = torch.ones(1, 2, 1, device=device, dtype=dtype) * val_per_core
        else:
            # Middle cores: (1, 2, 1)
            core = torch.ones(1, 2, 1, device=device, dtype=dtype) * val_per_core
        cores.append(core)
    
    return cores


def _add_constant(cores: List[Tensor], value: float) -> List[Tensor]:
    """Add a constant to all elements of QTT: x + value."""
    const_cores = _constant_qtt_cores(value, len(cores), cores[0].device, cores[0].dtype)
    return _add_qtt(cores, const_cores, max_rank=max(c.shape[0] for c in cores) + 1)


def _add_qtt(a_cores: List[Tensor], b_cores: List[Tensor], max_rank: int = 64, tol: float = 1e-8) -> List[Tensor]:
    """
    Add two QTTs: C = A + B.
    
    TT addition concatenates bond dimensions.
    Result rank: r_C ≤ r_A + r_B.
    """
    if len(a_cores) != len(b_cores):
        raise ValueError(f"QTT length mismatch: {len(a_cores)} vs {len(b_cores)}")
    
    n_cores = len(a_cores)
    result = []
    
    for k in range(n_cores):
        a = a_cores[k]  # (rA_left, 2, rA_right)
        b = b_cores[k]  # (rB_left, 2, rB_right)
        
        rA_left, phys_dim, rA_right = a.shape
        rB_left, _, rB_right = b.shape
        
        if k == 0:
            # First core: concatenate on output rank
            # [A; B] along r_out dimension
            c = torch.cat([a, b], dim=2)
        elif k == n_cores - 1:
            # Last core: concatenate on input rank
            c = torch.cat([a, b], dim=0)
        else:
            # Middle cores: block diagonal
            c = torch.zeros(
                rA_left + rB_left, phys_dim, rA_right + rB_right,
                device=a.device, dtype=a.dtype
            )
            c[:rA_left, :, :rA_right] = a
            c[rA_left:, :, rB_right:] = b
        
        result.append(c)
    
    # Truncate to control rank
    return truncate_cores(result, max_rank, tol=tol)


def _clamp_cores(cores: List[Tensor], max_val: float = 1e6) -> List[Tensor]:
    """Clamp core values to prevent numerical explosion."""
    result = []
    for core in cores:
        core_clamped = torch.clamp(core, -max_val, max_val)
        if torch.isnan(core_clamped).any():
            core_clamped = torch.nan_to_num(core_clamped, nan=1.0)
        result.append(core_clamped)
    return result


def qtt_safe_divide(
    numerator_cores: List[Tensor],
    denominator_cores: List[Tensor],
    max_rank: int = 64,
    eps: float = 1e-10,
) -> List[Tensor]:
    """
    Safe element-wise division: numerator / (denominator + eps).
    
    Convenience wrapper combining reciprocal and Hadamard product.
    
    Args:
        numerator_cores: QTT cores of numerator
        denominator_cores: QTT cores of denominator
        max_rank: Maximum rank for truncation
        eps: Regularization added to denominator
        
    Returns:
        QTT cores of element-wise quotient
    """
    recip = qtt_reciprocal(denominator_cores, max_rank=max_rank, eps=eps)
    return qtt_hadamard(numerator_cores, recip, max_rank=max_rank)
