"""
Streaming QTT Construction

This module provides a production-grade QTT construction method that:
1. Works correctly (verified against dense_to_qtt)
2. Uses O(N) memory for initial construction (unavoidable for general functions)
3. Provides O(log N) storage after construction

For true O(r² log N) construction (TCI), see tci_true.py which requires
more sophisticated pivot selection algorithms.

This file exists because tci_true.py has bugs that need to be fixed.
"""

import torch
from typing import Callable, List, Optional, Tuple
from tensornet.cfd.pure_qtt_ops import QTTState, qtt_to_dense


def build_qtt_from_function(
    func: Callable[[torch.Tensor], torch.Tensor],
    n_qubits: int,
    max_rank: int = 64,
    tol: float = 1e-10,
    batch_size: int = 65536,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> Tuple[QTTState, dict]:
    """
    Build QTT from function by streaming evaluation.
    
    This is NOT matrix-free TCI - it evaluates the function at all N points.
    But it does so in batches to manage memory, and the resulting QTT is O(log N).
    
    For problems where N is small enough to fit in memory (N < 10^8),
    this is the most reliable approach.
    
    Args:
        func: Function that takes tensor of indices and returns values
        n_qubits: Number of qubits (grid has 2^n_qubits points)
        max_rank: Maximum bond dimension
        tol: SVD truncation tolerance
        batch_size: Batch size for function evaluation
        device: Torch device
        dtype: Data type for computation
        verbose: Print progress
        
    Returns:
        qtt: QTTState with the compressed representation
        info: Dictionary with statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = 2 ** n_qubits
    
    if verbose:
        print(f"[StreamQTT] Building QTT: {n_qubits} qubits, N={N:,}")
    
    # Evaluate function in batches
    values_list = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        indices = torch.arange(start, end, device=device, dtype=torch.long)
        batch_values = func(indices).to(dtype=dtype)
        values_list.append(batch_values)
    
    values = torch.cat(values_list)
    
    if verbose:
        print(f"[StreamQTT] Function evaluated at {N:,} points")
    
    # Now do TT-SVD
    # Reshape to (2, 2, ..., 2) and do sequential SVD
    tensor = values.reshape([2] * n_qubits)
    
    cores = []
    current = tensor
    
    for k in range(n_qubits - 1):
        # Current shape: (r_left, 2, 2, ..., 2) with n-k dimensions
        # Reshape to (r_left * 2, rest)
        shape = current.shape
        r_left = shape[0] if k > 0 else 1
        if k == 0:
            mat = current.reshape(2, -1)
        else:
            mat = current.reshape(shape[0] * 2, -1)
        
        # SVD
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        
        # Truncate
        rank = min(max_rank, len(S))
        if tol > 0:
            cutoff = (S > tol * S[0]).sum().item()
            rank = min(rank, max(1, cutoff))
        
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        # Form core
        if k == 0:
            core = U.reshape(1, 2, rank)
        else:
            core = U.reshape(shape[0], 2, rank)
        cores.append(core)
        
        # Prepare for next iteration
        current = (torch.diag(S) @ Vh).reshape(rank, 2, *([2] * (n_qubits - k - 2)) if k < n_qubits - 2 else [])
        if k == n_qubits - 2:
            current = current.reshape(rank, 2)
    
    # Last core
    cores.append(current.reshape(current.shape[0], 2, 1))
    
    qtt = QTTState(cores=cores, num_qubits=n_qubits)
    
    info = {
        'n_evals': N,
        'params': sum(c.numel() for c in cores),
        'max_rank': qtt.max_rank,
        'compression': N / sum(c.numel() for c in cores),
    }
    
    if verbose:
        print(f"[StreamQTT] Done: {info['params']:,} params, {info['compression']:.1f}x compression")
    
    return qtt, info


def verify_qtt_accuracy(
    qtt: QTTState,
    func: Callable[[torch.Tensor], torch.Tensor],
    n_samples: int = 1000,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Verify QTT accuracy by sampling random points.
    
    Args:
        qtt: QTT state to verify
        func: Original function
        n_samples: Number of random samples
        device: Torch device
        
    Returns:
        Dictionary with error statistics
    """
    if device is None:
        device = qtt.cores[0].device
    
    N = qtt.grid_size
    n = qtt.num_qubits
    
    # Sample random indices
    indices = torch.randint(0, N, (n_samples,), device=device)
    
    # Get exact values
    exact = func(indices)
    
    # Get QTT approximation by evaluating at sample points
    # This is O(n_samples * n_qubits * max_rank^2)
    approx = torch.zeros(n_samples, device=device, dtype=qtt.cores[0].dtype)
    
    for i, idx in enumerate(indices):
        idx_val = idx.item()
        # Extract bits in C-order (first core = most significant bit)
        bits = [(idx_val >> (n - 1 - k)) & 1 for k in range(n)]
        
        # Contract cores
        result = qtt.cores[0][:, bits[0], :]  # Shape: (1, r1)
        for k in range(1, n):
            core_slice = qtt.cores[k][:, bits[k], :]  # Shape: (r_k, r_{k+1})
            result = result @ core_slice
        approx[i] = result.squeeze()
    
    max_error = (approx - exact).abs().max().item()
    mean_error = (approx - exact).abs().mean().item()
    rel_error = max_error / (exact.abs().max().item() + 1e-10)
    
    return {
        'max_error': max_error,
        'mean_error': mean_error,
        'rel_error': rel_error,
        'n_samples': n_samples,
    }
