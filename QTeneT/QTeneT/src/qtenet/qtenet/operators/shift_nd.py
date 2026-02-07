"""
N-Dimensional Shift Operator — The Master Key for Breaking the Curse

This module wraps the core nd_shift_mpo implementation, providing a clean
enterprise API for arbitrary-dimensional QTT shifts.

The shift operator is fundamental because:
1. All finite-difference schemes reduce to shifts
2. Morton interleaving makes dimension-agnostic shifting possible
3. Rank-2 MPO structure keeps complexity O(log N × r³)

Key Insight (Morton Interleaving):
- 2D: bits cycle as x0, y0, x1, y1, ... (period 2)
- 3D: bits cycle as x0, y0, z0, x1, y1, z1, ... (period 3)
- 5D: bits cycle as x0, y0, z0, vx0, vy0, x1, ... (period 5)
- 6D: bits cycle as x0, y0, z0, vx0, vy0, vz0, x1, ... (period 6)

The shift MPO activates only on qubits belonging to the target axis,
passing through all other dimensions unchanged while propagating carry/borrow.

Example:
    >>> from qtenet.operators import shift_nd, apply_shift
    >>> 
    >>> # Create shift operator for 6D (64^6 = 68 billion points)
    >>> # Only needs 36 qubits!
    >>> shift_x = shift_nd(
    ...     total_qubits=36,  # 6 qubits/dim × 6 dims
    ...     num_dims=6,
    ...     axis=0,           # X direction
    ...     direction=+1      # Forward shift
    ... )
    >>> 
    >>> # Apply to QTT state
    >>> shifted_state = apply_shift(shift_x, qtt_state, max_rank=64)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

# Import from upstream tensornet
from tensornet.cfd.nd_shift_mpo import (
    make_nd_shift_mpo as _make_nd_shift_mpo,
    apply_nd_shift_mpo as _apply_nd_shift_mpo,
    truncate_cores as _truncate_cores,
    cuda_shift_available,
    enable_cuda_shifts,
    disable_cuda_shifts,
    NDShiftConfig as _NDShiftConfig,
)


@dataclass(frozen=True)
class ShiftConfig:
    """Configuration for N-dimensional shift operations.
    
    Attributes:
        total_qubits: Total qubits in the QTT (must be divisible by num_dims)
        num_dims: Number of dimensions (2, 3, 5, 6, etc.)
        axis: Which axis to shift (0=X, 1=Y, 2=Z, 3=Vx, 4=Vy, 5=Vz)
        direction: +1 for forward, -1 for backward
        device: Torch device
        dtype: Torch dtype
    """
    total_qubits: int
    num_dims: int
    axis: int
    direction: Literal[-1, 1] = 1
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        if self.total_qubits % self.num_dims != 0:
            raise ValueError(
                f"total_qubits ({self.total_qubits}) must be divisible by "
                f"num_dims ({self.num_dims})"
            )
        if self.axis < 0 or self.axis >= self.num_dims:
            raise ValueError(
                f"axis ({self.axis}) must be in [0, {self.num_dims-1}]"
            )
        if self.direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
    
    @property
    def qubits_per_dim(self) -> int:
        """Qubits per dimension."""
        return self.total_qubits // self.num_dims
    
    @property
    def grid_size_per_dim(self) -> int:
        """Grid points per dimension (2^qubits_per_dim)."""
        return 2 ** self.qubits_per_dim
    
    @property
    def total_points(self) -> int:
        """Total grid points (2^total_qubits)."""
        return 2 ** self.total_qubits


def shift_nd(
    total_qubits: int,
    num_dims: int,
    axis: int,
    direction: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    """
    Create N-dimensional shift operator as MPO cores.
    
    This is THE "Master Key" for breaking the curse of dimensionality.
    It enables O(log N) shifts in arbitrary dimensions using Morton interleaving.
    
    Args:
        total_qubits: Total qubits in QTT representation
                      (e.g., 30 for 32^6 grid with 5 qubits/dim × 6 dims)
        num_dims: Number of physical dimensions (2, 3, 5, 6, etc.)
        axis: Which axis to shift along
              (0=X, 1=Y, 2=Z, 3=Vx, 4=Vy, 5=Vz for phase space)
        direction: +1 for forward shift (x → x+1), -1 for backward shift
        device: Torch device ("cpu" or "cuda")
        dtype: Tensor dtype (default float32)
    
    Returns:
        List of MPO cores (Rank-2), one per qubit
    
    Examples:
        # 2D (128×128): 7 qubits/dim × 2 dims = 14 total
        shift_x = shift_nd(14, num_dims=2, axis=0, direction=+1)
        shift_y = shift_nd(14, num_dims=2, axis=1, direction=-1)
        
        # 3D (64^3): 6 qubits/dim × 3 dims = 18 total
        shift_z = shift_nd(18, num_dims=3, axis=2, direction=+1)
        
        # 5D phase space (32^5): 5 qubits/dim × 5 dims = 25 total
        shift_vx = shift_nd(25, num_dims=5, axis=3, direction=+1)
        
        # 6D Vlasov-Maxwell (32^6): 5 qubits/dim × 6 dims = 30 total
        shift_vz = shift_nd(30, num_dims=6, axis=5, direction=-1)
    
    Complexity:
        - MPO construction: O(n_qubits)
        - MPO application: O(n_qubits × r³) where r is max rank
        - Total per shift: O(log N × r³) where N = 2^total_qubits
    """
    config = ShiftConfig(
        total_qubits=total_qubits,
        num_dims=num_dims,
        axis=axis,
        direction=direction,
        device=device,
        dtype=dtype,
    )
    
    dev = torch.device(device)
    
    return _make_nd_shift_mpo(
        num_qubits_total=total_qubits,
        num_dims=num_dims,
        axis_idx=axis,
        direction=direction,
        device=dev,
        dtype=dtype,
    )


def apply_shift(
    qtt_cores: list[Tensor],
    mpo_cores: list[Tensor],
    max_rank: int = 64,
    eps: float = 1e-10,
) -> list[Tensor]:
    """
    Apply shift MPO to QTT state and truncate.
    
    Args:
        qtt_cores: List of QTT state cores
        mpo_cores: List of MPO cores from shift_nd()
        max_rank: Maximum bond dimension after truncation
        eps: SVD truncation tolerance
    
    Returns:
        New QTT cores (shifted and truncated)
    
    Complexity: O(n_qubits × r³) where r = max_rank
    """
    # Apply MPO - upstream API is (state_cores, mpo_cores)
    result_cores = _apply_nd_shift_mpo(qtt_cores, mpo_cores, max_rank=max_rank)
    
    return result_cores


def make_shift_operators(
    total_qubits: int,
    num_dims: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, list[Tensor]]:
    """
    Create all shift operators for an N-dimensional simulation.
    
    Returns a dictionary with keys like "shift_x_plus", "shift_x_minus", etc.
    
    Args:
        total_qubits: Total qubits in QTT
        num_dims: Number of dimensions
        device: Torch device
        dtype: Tensor dtype
    
    Returns:
        Dictionary mapping shift names to MPO cores
    
    Example:
        >>> ops = make_shift_operators(30, num_dims=6)
        >>> ops["shift_x_plus"]   # Forward X shift
        >>> ops["shift_vz_minus"] # Backward Vz shift
    """
    axis_names = ["x", "y", "z", "vx", "vy", "vz", "w", "u"][:num_dims]
    
    operators = {}
    for axis_idx, axis_name in enumerate(axis_names):
        for direction, suffix in [(1, "plus"), (-1, "minus")]:
            key = f"shift_{axis_name}_{suffix}"
            operators[key] = shift_nd(
                total_qubits=total_qubits,
                num_dims=num_dims,
                axis=axis_idx,
                direction=direction,
                device=device,
                dtype=dtype,
            )
    
    return operators


# Re-export CUDA controls
__all__ = [
    "shift_nd",
    "apply_shift",
    "make_shift_operators",
    "ShiftConfig",
    "cuda_shift_available",
    "enable_cuda_shifts",
    "disable_cuda_shifts",
]
