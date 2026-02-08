"""
Build QTT from Sparse Samples

When you already have (index, value) pairs instead of a callable function,
use from_samples to construct the QTT directly.

Example:
    >>> from qtenet.tci import from_samples
    >>> 
    >>> indices = torch.tensor([0, 42, 1000, 999999])
    >>> values = torch.tensor([1.0, 2.5, 0.3, -1.2])
    >>> 
    >>> cores = from_samples(indices, values, n_qubits=20, max_rank=32)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tensornet.cfd.qtt_eval import dense_to_qtt_cores


def from_samples(
    indices: Tensor,
    values: Tensor,
    n_qubits: int,
    max_rank: int = 64,
    fill_value: float = 0.0,
    device: str = "cpu",
) -> list[Tensor]:
    """
    Build QTT from explicit (index, value) pairs.
    
    This constructs a sparse representation and compresses to QTT.
    Suitable when you have pre-computed samples rather than a callable.
    
    Args:
        indices: 1D tensor of integer indices in [0, 2^n_qubits)
        values: 1D tensor of corresponding values
        n_qubits: Number of qubits (defines domain size)
        max_rank: Maximum TT-rank
        fill_value: Value for unsampled points
        device: Torch device
    
    Returns:
        List of QTT cores
    
    Note:
        For n_qubits > 20, this may require significant memory.
        Consider using from_sparse_samples for large domains.
    """
    if indices.shape != values.shape:
        raise ValueError(
            f"indices and values must have same shape, got {indices.shape} and {values.shape}"
        )
    
    N = 2 ** n_qubits
    dev = torch.device(device)
    
    # Create dense vector (only works for moderate n_qubits)
    if n_qubits > 24:
        raise ValueError(
            f"n_qubits={n_qubits} is too large for dense construction. "
            f"Use from_sparse_samples or from_function instead."
        )
    
    dense = torch.full((N,), fill_value, dtype=values.dtype, device=dev)
    dense[indices.long()] = values.to(dev)
    
    return dense_to_qtt_cores(dense, max_rank=max_rank)


def from_sparse_samples(
    indices: Tensor,
    values: Tensor,
    n_qubits: int,
    max_rank: int = 64,
    n_sweeps: int = 10,
    device: str = "cpu",
    verbose: bool = False,
) -> list[Tensor]:
    """
    Build QTT from sparse samples using ALS-style optimization.
    
    Unlike from_samples, this NEVER allocates O(2^n) memory.
    It uses alternating least squares to find QTT cores that
    minimize reconstruction error at the given sample points.
    
    Args:
        indices: 1D tensor of integer indices
        values: 1D tensor of corresponding values
        n_qubits: Number of qubits
        max_rank: Maximum TT-rank
        n_sweeps: Number of ALS sweeps
        device: Torch device
        verbose: Print progress
    
    Returns:
        List of QTT cores
    
    Example:
        # Sparse samples from expensive simulation
        indices = torch.randint(0, 2**30, (10000,))
        values = my_expensive_simulation(indices)
        
        cores = from_sparse_samples(indices, values, n_qubits=30, max_rank=64)
    """
    dev = torch.device(device)
    n_samples = len(indices)
    
    if verbose:
        print(f"[Sparse TCI] Building from {n_samples} samples, n_qubits={n_qubits}")
    
    # Initialize random cores
    cores = []
    r_left = 1
    for k in range(n_qubits):
        r_right = min(max_rank, 2 ** min(k + 1, n_qubits - k - 1))
        core = torch.randn(r_left, 2, r_right, device=dev) * 0.1
        cores.append(core)
        r_left = r_right
    
    # Extract bits from indices
    bits = torch.zeros(n_samples, n_qubits, dtype=torch.long, device=dev)
    for k in range(n_qubits):
        bits[:, k] = (indices >> (n_qubits - 1 - k)) & 1
    
    values_dev = values.to(dev).float()
    
    # ALS sweeps
    for sweep in range(n_sweeps):
        # Left-to-right
        for k in range(n_qubits):
            cores[k] = _optimize_core(cores, bits, values_dev, k, max_rank)
        
        # Right-to-left
        for k in range(n_qubits - 1, -1, -1):
            cores[k] = _optimize_core(cores, bits, values_dev, k, max_rank)
        
        if verbose:
            error = _compute_error(cores, bits, values_dev)
            print(f"  Sweep {sweep + 1}/{n_sweeps}: error = {error:.6e}")
    
    return cores


def _optimize_core(
    cores: list[Tensor],
    bits: Tensor,
    values: Tensor,
    k: int,
    max_rank: int,
) -> Tensor:
    """Optimize single core using least squares."""
    n_samples = bits.shape[0]
    n_qubits = len(cores)
    
    # Compute left environment (product of cores 0..k-1)
    left = torch.ones(n_samples, 1, device=bits.device)
    for i in range(k):
        bit = bits[:, i]
        core_i = cores[i]
        left = torch.einsum("br,rdr->bd", left, core_i[:, bit, :].permute(1, 0, 2))
    
    # Compute right environment (product of cores k+1..n-1)
    right = torch.ones(n_samples, 1, device=bits.device)
    for i in range(n_qubits - 1, k, -1):
        bit = bits[:, i]
        core_i = cores[i]
        right = torch.einsum("rdr,br->bd", core_i[:, bit, :].permute(1, 0, 2), right)
    
    # Build design matrix
    bit_k = bits[:, k]
    r_left, r_right = left.shape[1], right.shape[1]
    
    # Design matrix: A[sample, (r_left * 2 * r_right)]
    A = torch.zeros(n_samples, r_left * 2 * r_right, device=bits.device)
    for d in range(2):
        mask = bit_k == d
        if mask.any():
            outer = torch.einsum("bi,bj->bij", left[mask], right[mask])
            A[mask, d * r_left * r_right : (d + 1) * r_left * r_right] = outer.reshape(-1, r_left * r_right)
    
    # Solve least squares
    try:
        x, _ = torch.linalg.lstsq(A, values.unsqueeze(1))[:2]
    except:
        x = torch.linalg.pinv(A) @ values.unsqueeze(1)
    
    # Reshape to core
    new_core = x.reshape(r_left, 2, r_right)
    
    return new_core


def _compute_error(cores: list[Tensor], bits: Tensor, values: Tensor) -> float:
    """Compute relative reconstruction error at sample points."""
    n_samples = bits.shape[0]
    
    # Contract cores at sample points
    result = torch.ones(n_samples, 1, device=bits.device)
    for k, core in enumerate(cores):
        bit = bits[:, k]
        result = torch.einsum("br,rdr->bd", result, core[:, bit, :].permute(1, 0, 2))
    
    predictions = result.squeeze()
    error = torch.norm(predictions - values) / (torch.norm(values) + 1e-10)
    return error.item()


__all__ = ["from_samples", "from_sparse_samples"]
