"""
N-Dimensional Laplacian Operator

The Laplacian is constructed from shift operators:
  ∇²f = Σᵢ (S⁺ᵢ - 2I + S⁻ᵢ) / dxᵢ²

where S⁺ᵢ and S⁻ᵢ are forward/backward shifts along axis i.

For QTT representation, the Laplacian MPO has rank at most 4×num_dims
(from combining 2×num_dims shift MPOs).

Example:
    >>> from qtenet.operators import laplacian_nd, apply_laplacian
    >>> 
    >>> # 3D Laplacian for 64^3 grid
    >>> lap = laplacian_nd(total_qubits=18, num_dims=3, dx=0.01)
    >>> result = apply_laplacian(lap, qtt_state, max_rank=64)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tensornet.cfd.nd_shift_mpo import make_laplacian_mpo as _make_laplacian_mpo
from tensornet.cfd.nd_shift_mpo import apply_laplacian_mpo as _apply_laplacian_mpo
from tensornet.cfd.nd_shift_mpo import truncate_cores as _truncate_cores


def laplacian_nd(
    total_qubits: int,
    num_dims: int,
    dx: float | list[float] = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    """
    Create N-dimensional Laplacian operator as fused MPO.
    
    The Laplacian is:
      ∇²f = Σᵢ (f[xᵢ+1] - 2f[xᵢ] + f[xᵢ-1]) / dxᵢ²
    
    This fuses all 2×num_dims shift operations into a single MPO
    for efficient application.
    
    Args:
        total_qubits: Total qubits in QTT (must be divisible by num_dims)
        num_dims: Number of physical dimensions
        dx: Grid spacing (scalar for uniform, list for non-uniform)
        device: Torch device
        dtype: Tensor dtype
    
    Returns:
        List of fused Laplacian MPO cores
    
    Example:
        # 3D Laplacian for 64^3 grid (18 qubits)
        lap = laplacian_nd(18, num_dims=3, dx=0.1)
        
        # 6D phase-space Laplacian (30 qubits)
        lap_6d = laplacian_nd(30, num_dims=6, dx=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    
    Complexity:
        - Construction: O(n_qubits × num_dims)
        - Application: O(n_qubits × r³)
    """
    if total_qubits % num_dims != 0:
        raise ValueError(
            f"total_qubits ({total_qubits}) must be divisible by num_dims ({num_dims})"
        )
    
    qubits_per_dim = total_qubits // num_dims
    
    # Handle dx as list
    if isinstance(dx, (int, float)):
        dx_list = [float(dx)] * num_dims
    else:
        dx_list = [float(d) for d in dx]
        if len(dx_list) != num_dims:
            raise ValueError(f"dx list length ({len(dx_list)}) must match num_dims ({num_dims})")
    
    dev = torch.device(device)
    
    return _make_laplacian_mpo(
        qubits_per_dim=qubits_per_dim,
        num_dims=num_dims,
        dx=dx_list[0] if all(d == dx_list[0] for d in dx_list) else dx_list,
        device=dev,
        dtype=dtype,
    )


def apply_laplacian(
    mpo_cores: list[Tensor],
    qtt_cores: list[Tensor],
    max_rank: int = 64,
    eps: float = 1e-10,
) -> list[Tensor]:
    """
    Apply Laplacian MPO to QTT state and truncate.
    
    Args:
        mpo_cores: Laplacian MPO from laplacian_nd()
        qtt_cores: QTT state cores
        max_rank: Maximum bond dimension after truncation
        eps: SVD truncation tolerance
    
    Returns:
        New QTT cores representing ∇²f
    """
    result = _apply_laplacian_mpo(mpo_cores, qtt_cores)
    return _truncate_cores(result, max_rank=max_rank, eps=eps)


__all__ = ["laplacian_nd", "apply_laplacian"]
