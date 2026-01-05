"""
Core QTT data structures and conversion functions.

This module provides the fundamental QTTState and MPO classes along with
functions for converting between dense tensors and QTT format.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

import torch
from torch import Tensor


@dataclass
class QTTState:
    """
    Quantized Tensor-Train representation of a 1D vector.
    
    A vector of length 2^n is represented as a chain of n cores,
    where each core is a 3-index tensor of shape (r_left, 2, r_right).
    
    Attributes:
        cores: List of 3-index tensors [r_left, 2, r_right]
        num_qubits: Number of "qubits" = log2(grid_size)
    
    Example:
        >>> qtt = QTTState(cores=[...], num_qubits=20)
        >>> print(f"Grid size: {qtt.grid_size}")  # 1,048,576
        >>> print(f"Memory: {qtt.memory_bytes / 1e3:.1f} KB")
    """
    
    cores: List[Tensor]
    num_qubits: int
    
    @property
    def grid_size(self) -> int:
        """Number of points represented: 2^num_qubits."""
        return 2 ** self.num_qubits
    
    @property
    def ranks(self) -> List[int]:
        """Bond dimensions between cores."""
        return [c.shape[2] for c in self.cores[:-1]]
    
    @property
    def max_rank(self) -> int:
        """Maximum bond dimension."""
        return max(self.ranks) if self.ranks else 1
    
    @property
    def memory_bytes(self) -> int:
        """Total memory used by cores in bytes."""
        return sum(c.numel() * c.element_size() for c in self.cores)
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of the cores."""
        return self.cores[0].dtype
    
    @property
    def device(self) -> torch.device:
        """Device where cores are stored."""
        return self.cores[0].device
    
    def to(self, device: torch.device) -> QTTState:
        """Move all cores to the specified device."""
        return QTTState(
            cores=[c.to(device) for c in self.cores],
            num_qubits=self.num_qubits
        )
    
    def clone(self) -> QTTState:
        """Create a deep copy."""
        return QTTState(
            cores=[c.clone() for c in self.cores],
            num_qubits=self.num_qubits
        )
    
    def __repr__(self) -> str:
        ranks_str = f"max_rank={self.max_rank}"
        mem_str = f"memory={self.memory_bytes / 1e3:.1f}KB"
        return f"QTTState(n={self.num_qubits}, grid={self.grid_size}, {ranks_str}, {mem_str})"


@dataclass
class MPO:
    """
    Matrix Product Operator for applying linear operators in QTT format.
    
    An MPO represents a linear operator on a QTT state. Each core is a
    4-index tensor of shape (r_left, d_phys, d_phys, r_right) where
    d_phys=2 for QTT.
    
    Attributes:
        cores: List of 4-index tensors [r_left, 2, 2, r_right]
        num_sites: Number of sites (same as num_qubits for QTT)
    """
    
    cores: List[Tensor]
    num_sites: int
    
    @property
    def ranks(self) -> List[int]:
        """Bond dimensions between cores."""
        return [c.shape[3] for c in self.cores[:-1]]
    
    @property
    def max_rank(self) -> int:
        """Maximum bond dimension."""
        return max(self.ranks) if self.ranks else 1
    
    @property
    def memory_bytes(self) -> int:
        """Total memory used by cores in bytes."""
        return sum(c.numel() * c.element_size() for c in self.cores)
    
    def __repr__(self) -> str:
        return f"MPO(sites={self.num_sites}, max_rank={self.max_rank})"


def dense_to_qtt(
    tensor: Tensor,
    max_bond: int = 64,
    tol: float = 1e-12,
    normalize: bool = False
) -> QTTState:
    """
    Convert a 1D dense tensor to QTT format.
    
    The tensor length must be a power of 2. The tensor is reshaped to
    (2, 2, ..., 2) and decomposed via sequential SVD.
    
    Args:
        tensor: 1D tensor of length 2^n
        max_bond: Maximum bond dimension (truncation parameter)
        tol: Relative tolerance for SVD truncation
        normalize: If True, normalize the result
    
    Returns:
        QTTState representation
    
    Raises:
        ValueError: If tensor length is not a power of 2
    
    Example:
        >>> x = torch.linspace(0, 1, 2**20)
        >>> qtt = dense_to_qtt(torch.sin(x), max_bond=32)
        >>> print(qtt)
    """
    if tensor.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got {tensor.dim()}D")
    
    N = tensor.shape[0]
    if N & (N - 1) != 0:
        raise ValueError(f"Tensor length must be power of 2, got {N}")
    
    num_qubits = int(math.log2(N))
    
    # TT-SVD algorithm: sequential decomposition from left to right
    # Reshape to (2, 2, ..., 2) tensor
    current = tensor.reshape([2] * num_qubits)
    
    cores: List[Tensor] = []
    r_left = 1  # Left bond dimension starts at 1
    
    for i in range(num_qubits - 1):
        # Reshape: (r_left * 2, remaining dimensions)
        # current has shape [r_left, 2, 2, ..., 2] with (num_qubits - i) indices of size 2
        # plus possibly a leading r_left dimension
        if i == 0:
            # current: [2, 2, 2, ..., 2] with num_qubits indices
            mat = current.reshape(2, -1)  # (2, 2^{n-1})
        else:
            # current: [r_left, 2, 2, ..., 2] with 1 + (num_qubits - i) indices
            mat = current.reshape(r_left * 2, -1)  # (r_left * 2, 2^{remaining})
        
        # SVD
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        
        # Determine truncation rank
        if tol > 0 and S.numel() > 0:
            total_sq = (S ** 2).sum()
            cumsum_sq = torch.cumsum(S ** 2, dim=0)
            # Keep enough singular values to capture (1-tol^2) of Frobenius norm
            rank_tol = torch.searchsorted(cumsum_sq, total_sq * (1 - tol**2)).item() + 1
        else:
            rank_tol = len(S)
        
        rank = min(max_bond, rank_tol, len(S))
        
        # Truncate
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        # Form core: reshape U to (r_left, 2, rank)
        if i == 0:
            core = U.reshape(1, 2, rank)
        else:
            core = U.reshape(r_left, 2, rank)
        
        cores.append(core)
        
        # Prepare for next iteration: absorb S into Vh
        # Next current = diag(S) @ Vh, then reshape for next site
        current = torch.diag(S) @ Vh
        remaining_qubits = num_qubits - i - 1
        current = current.reshape([rank] + [2] * remaining_qubits)
        r_left = rank
    
    # Last core: current has shape [r_left, 2]
    # Reshape to (r_left, 2, 1)
    last_core = current.reshape(r_left, 2, 1)
    cores.append(last_core)
    
    result = QTTState(cores=cores, num_qubits=num_qubits)
    
    if normalize:
        from qtt_sdk.operations import qtt_norm, qtt_scale
        norm = qtt_norm(result)
        if norm > 1e-15:
            result = qtt_scale(result, 1.0 / norm)
    
    return result


def qtt_to_dense(qtt: QTTState) -> Tensor:
    """
    Reconstruct dense tensor from QTT format.
    
    Warning: This allocates O(2^n) memory. Only use for small grids
    (n <= 25 or so, depending on available RAM).
    
    Args:
        qtt: QTT state to reconstruct
    
    Returns:
        Dense 1D tensor of length 2^n
    
    Example:
        >>> qtt = dense_to_qtt(original, max_bond=32)
        >>> reconstructed = qtt_to_dense(qtt)
        >>> error = torch.norm(original - reconstructed) / torch.norm(original)
    """
    if qtt.num_qubits > 30:
        raise MemoryError(
            f"Reconstruction of 2^{qtt.num_qubits} points would require "
            f"{2**qtt.num_qubits * 8 / 1e9:.1f} GB. Use compressed operations instead."
        )
    
    result = qtt.cores[0]  # (1, 2, r1)
    
    for core in qtt.cores[1:]:
        # result: (1, 2^k, r_k)
        # core: (r_k, 2, r_{k+1})
        # Contract on the bond dimension
        result = torch.einsum('...i,ijk->...jk', result, core)
    
    # result: (1, 2^n, 1) -> (2^n,)
    return result.squeeze(0).squeeze(-1).reshape(-1)


def random_qtt(
    num_qubits: int,
    bond_dim: int = 16,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None
) -> QTTState:
    """
    Create a random QTT state (useful for testing).
    
    Args:
        num_qubits: Number of qubits (grid size = 2^num_qubits)
        bond_dim: Bond dimension for all cores
        dtype: Data type
        device: Device for tensors
    
    Returns:
        Random QTT state
    """
    cores = []
    
    for i in range(num_qubits):
        r_left = 1 if i == 0 else bond_dim
        r_right = 1 if i == num_qubits - 1 else bond_dim
        core = torch.randn(r_left, 2, r_right, dtype=dtype, device=device)
        cores.append(core)
    
    return QTTState(cores=cores, num_qubits=num_qubits)
