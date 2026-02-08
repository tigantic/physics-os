"""
N-Dimensional Gradient Operator

First derivative via central differences:
  ∂f/∂xᵢ = (S⁺ᵢf - S⁻ᵢf) / (2 dxᵢ)

Example:
    >>> from qtenet.operators import gradient_nd, apply_gradient
    >>> 
    >>> # Gradient along x-axis for 3D field
    >>> grad_x = gradient_nd(total_qubits=18, num_dims=3, axis=0, dx=0.01)
    >>> dfx = apply_gradient(grad_x, qtt_state, max_rank=64)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tensornet.cfd.nd_shift_mpo import (
    make_nd_shift_mpo,
    apply_nd_shift_mpo,
    truncate_cores,
)
from tensornet.cfd.pure_qtt_ops import mpo_add, mpo_scale, mpo_negate


def gradient_nd(
    total_qubits: int,
    num_dims: int,
    axis: int,
    dx: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    """
    Create gradient operator along specified axis.
    
    Uses central differences: ∂f/∂x = (f[x+1] - f[x-1]) / (2dx)
    
    Args:
        total_qubits: Total qubits in QTT
        num_dims: Number of dimensions
        axis: Axis to differentiate along (0=X, 1=Y, etc.)
        dx: Grid spacing
        device: Torch device
        dtype: Tensor dtype
    
    Returns:
        List of gradient MPO cores
    
    Example:
        # Velocity gradient in 6D phase space
        grad_vx = gradient_nd(30, num_dims=6, axis=3, dx=0.1)
    """
    if total_qubits % num_dims != 0:
        raise ValueError(
            f"total_qubits ({total_qubits}) must be divisible by num_dims ({num_dims})"
        )
    if axis < 0 or axis >= num_dims:
        raise ValueError(f"axis ({axis}) must be in [0, {num_dims-1}]")
    
    dev = torch.device(device)
    
    # Forward shift S⁺
    shift_plus = make_nd_shift_mpo(
        num_qubits_total=total_qubits,
        num_dims=num_dims,
        axis_idx=axis,
        direction=+1,
        device=dev,
        dtype=dtype,
    )
    
    # Backward shift S⁻
    shift_minus = make_nd_shift_mpo(
        num_qubits_total=total_qubits,
        num_dims=num_dims,
        axis_idx=axis,
        direction=-1,
        device=dev,
        dtype=dtype,
    )
    
    # Gradient = (S⁺ - S⁻) / (2dx)
    neg_shift_minus = mpo_negate(shift_minus)
    diff = mpo_add(shift_plus, neg_shift_minus)
    gradient = mpo_scale(diff, 1.0 / (2.0 * dx))
    
    return gradient


def apply_gradient(
    mpo_cores: list[Tensor],
    qtt_cores: list[Tensor],
    max_rank: int = 64,
    eps: float = 1e-10,
) -> list[Tensor]:
    """
    Apply gradient MPO to QTT state and truncate.
    
    Args:
        mpo_cores: Gradient MPO from gradient_nd()
        qtt_cores: QTT state cores
        max_rank: Maximum bond dimension
        eps: SVD truncation tolerance
    
    Returns:
        New QTT cores representing ∂f/∂x
    """
    result = apply_nd_shift_mpo(mpo_cores, qtt_cores)
    return truncate_cores(result, max_rank=max_rank, eps=eps)


__all__ = ["gradient_nd", "apply_gradient"]
