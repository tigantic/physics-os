"""
QTT (Quantized Tensor Train) Construction and Evaluation.

This module provides:
1. Dense TT-SVD construction: qtt_from_function_dense
2. Batch evaluation: qtt_eval_batch
3. Single-point evaluation: qtt_eval_at_index
4. Conversion utilities: dense_to_qtt_cores

Complexity:
- Construction: O(N × r²) for dense, O(r² × log N) for TCI
- Evaluation: O(log N × r²) per point
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Callable, List


def dense_to_qtt_cores(
    values: Tensor,
    max_rank: int = 64,
) -> List[Tensor]:
    """Convert dense vector to QTT cores via TT-SVD.
    
    Args:
        values: (N,) tensor where N = 2^n for some n
        max_rank: Maximum TT rank for truncation
        
    Returns:
        List of QTT cores, each shape (r_left, 2, r_right)
    """
    N = values.shape[0]
    n_qubits = int(torch.log2(torch.tensor(float(N))).ceil().item())
    
    if 2**n_qubits != N:
        # Pad to next power of 2
        padded = torch.zeros(2**n_qubits, device=values.device, dtype=values.dtype)
        padded[:N] = values
        values = padded
        N = 2**n_qubits
    
    device = values.device
    dtype = values.dtype
    
    cores = []
    remaining = values.clone()
    r_left = 1
    
    for i in range(n_qubits - 1):
        # Current remaining shape: (r_left * 2^(n-i),)
        # Reshape to (r_left * 2, 2^(n-i-1))
        remaining_size = remaining.numel()
        cols = remaining_size // (r_left * 2)
        mat = remaining.reshape(r_left * 2, cols)
        
        # SVD
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        
        # Truncate
        tol = S[0] * 1e-10 if len(S) > 0 else 0
        rank = min(max_rank, len(S), max(1, int((S > tol).sum().item())))
        rank = max(1, rank)
        
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        # Core: (r_left, 2, rank)
        core = U.reshape(r_left, 2, rank)
        cores.append(core.to(dtype))
        
        # Remaining: S @ Vh → (rank, cols) → flatten for next iteration
        remaining = (torch.diag(S) @ Vh).flatten()
        r_left = rank
    
    # Last core: (r_left, 2, 1)
    last_core = remaining.reshape(r_left, 2, 1)
    cores.append(last_core.to(dtype))
    
    return cores


def qtt_from_function_dense(
    f: Callable[[Tensor], Tensor],
    n_qubits: int,
    max_rank: int = 64,
    device: str = "cpu",
) -> List[Tensor]:
    """Build QTT from function by dense sampling + TT-SVD.
    
    This is the fallback method with O(N) complexity.
    For O(r² × log N), use TCI when available.
    
    Args:
        f: Function taking indices (batch,) and returning values (batch,)
        n_qubits: Number of qubits (N = 2^n_qubits)
        max_rank: Maximum TT rank
        device: Torch device
        
    Returns:
        List of QTT cores
    """
    N = 2**n_qubits
    indices = torch.arange(N, device=device)
    values = f(indices)
    return dense_to_qtt_cores(values, max_rank=max_rank)


def index_to_bits(index: Tensor, n_qubits: int) -> Tensor:
    """Convert flat indices to binary representation.
    
    Args:
        index: (batch,) tensor of indices
        n_qubits: Number of qubits
        
    Returns:
        (batch, n_qubits) tensor of bits, MSB first
    """
    bit_positions = 2 ** torch.arange(n_qubits - 1, -1, -1, device=index.device)
    bits = (index.unsqueeze(1) // bit_positions.unsqueeze(0)) % 2
    return bits.long()


def qtt_eval_at_index(
    cores: List[Tensor],
    index: int,
) -> Tensor:
    """Evaluate QTT at a single index.
    
    Complexity: O(n_qubits × r²)
    
    Args:
        cores: List of QTT cores, each shape (r_left, 2, r_right)
        index: Integer index in [0, 2^n_qubits)
        
    Returns:
        Scalar tensor
    """
    n_qubits = len(cores)
    
    # Decompose index to bits (MSB first, matching batch version)
    bits = []
    for k in range(n_qubits - 1, -1, -1):
        bits.append((index >> k) & 1)
    # bits is now MSB first: [b_{n-1}, b_{n-2}, ..., b_0]
    
    # Contract through cores
    v = cores[0][:, bits[0], :]
    for i in range(1, n_qubits):
        G_slice = cores[i][:, bits[i], :]
        v = v @ G_slice
    
    return v.squeeze()


def qtt_eval_batch(
    cores: List[Tensor],
    indices: Tensor,
) -> Tensor:
    """Evaluate QTT at a batch of indices.
    
    Complexity: O(batch × n_qubits × r²)
    
    Args:
        cores: List of QTT cores
        indices: (batch,) tensor of indices
        
    Returns:
        (batch,) tensor of values
    """
    n_qubits = len(cores)
    batch_size = indices.shape[0]
    
    bits = index_to_bits(indices, n_qubits)
    
    # First core
    v = cores[0][0, bits[:, 0], :]
    
    # Contract through remaining cores
    for i in range(1, n_qubits):
        core = cores[i]
        b = bits[:, i]
        r_left, _, r_right = core.shape
        
        # Gather correct slices
        core_expanded = core.permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1)
        b_idx = b.view(batch_size, 1, 1, 1).expand(-1, 1, r_left, r_right)
        selected = torch.gather(core_expanded, dim=1, index=b_idx).squeeze(1)
        
        v = torch.bmm(v.unsqueeze(1), selected).squeeze(1)
    
    return v.squeeze(-1)


def extract_lookup_table(
    cores: List[Tensor],
    n_contexts: int,
) -> Tensor:
    """Extract dense lookup table from QTT cores.
    
    For O(1) inference, precompute all outputs.
    
    Args:
        cores: QTT cores
        n_contexts: Number of valid contexts
        
    Returns:
        (n_contexts,) uint8 tensor of argmax values
    """
    indices = torch.arange(n_contexts, device=cores[0].device)
    values = qtt_eval_batch(cores, indices)
    return torch.round(values).clamp(0, 255).to(torch.uint8)
