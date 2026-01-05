"""
Tensor-Train Compression
========================

The core data structure and compression algorithm.

The Patent: After each physics step, we re-compress the state to bounded rank.
This keeps memory O(N·d·r²) instead of O(N^d).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class TTTensor:
    """
    Tensor-Train representation of high-dimensional state.
    
    A tensor A[i₁, i₂, ..., iₐ] is represented as:
        A[i₁, i₂, ..., iₐ] = G₁[i₁] · G₂[i₂] · ... · Gₐ[iₐ]
    
    where each Gₖ[iₖ] is an (rₖ₋₁ × rₖ) matrix.
    
    Attributes:
        cores: List of 3D arrays, each of shape (r_{k-1}, n_k, r_k)
        shape: Original tensor shape (n₁, n₂, ..., nₐ)
        ranks: TT-ranks (r₀, r₁, ..., rₐ) where r₀ = rₐ = 1
    """
    cores: List[np.ndarray]
    shape: Tuple[int, ...]
    ranks: Tuple[int, ...]
    
    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        return len(self.shape)
    
    @property
    def size(self) -> int:
        """Total number of elements in full tensor"""
        return int(np.prod(self.shape))
    
    @property
    def tt_size(self) -> int:
        """Number of elements stored in TT format"""
        return sum(c.size for c in self.cores)
    
    @property
    def compression_ratio(self) -> float:
        """How much smaller than full tensor"""
        return self.size / self.tt_size if self.tt_size > 0 else 1.0
    
    @property
    def max_rank(self) -> int:
        """Maximum bond dimension"""
        return max(self.ranks)
    
    def __repr__(self) -> str:
        return f"TTTensor(shape={self.shape}, ranks={self.ranks}, compression={self.compression_ratio:.1f}×)"


def tt_round(tensor: np.ndarray, max_rank: int = 12, cutoff: float = 1e-14) -> TTTensor:
    """
    Compress tensor to TT format with bounded rank.
    
    This is the core algorithm - "compress the universe".
    
    Uses sequential SVD (TT-SVD) to decompose a full tensor into
    a train of low-rank cores.
    
    Args:
        tensor: Full tensor of shape (n₁, n₂, ..., nₐ)
        max_rank: Maximum bond dimension (default: 12)
        cutoff: Truncation threshold for singular values
        
    Returns:
        TTTensor with bounded ranks
        
    Complexity:
        O(n^d · r²) where r = max_rank, vs O(n^d) storage for full
    """
    if tensor.ndim == 1:
        # 1D: trivial TT
        return TTTensor(
            cores=[tensor.reshape(1, -1, 1)],
            shape=tensor.shape,
            ranks=(1, 1)
        )
    
    # TT-SVD algorithm
    shape = tensor.shape
    d = len(shape)
    cores = []
    ranks = [1]
    
    # Reshape to matrix for first SVD
    current = tensor.reshape(shape[0], -1)
    
    for k in range(d - 1):
        # SVD of current matrix
        U, S, Vh = np.linalg.svd(current, full_matrices=False)
        
        # Determine rank: min of max_rank, available rank, significant singular values
        r = min(max_rank, len(S))
        if cutoff > 0:
            significant = np.sum(S > cutoff * S[0])
            r = min(r, max(significant, 1))
        r = max(r, 1)  # At least rank 1
        
        # Truncate
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]
        
        # Store core: reshape U into (r_{k-1}, n_k, r_k)
        core = U.reshape(ranks[-1], shape[k], r)
        cores.append(core)
        ranks.append(r)
        
        # Prepare next iteration: absorb S into Vh
        current = np.diag(S) @ Vh
        
        # Reshape for next mode
        if k < d - 2:
            remaining_size = int(np.prod(shape[k+2:]))
            current = current.reshape(r * shape[k+1], remaining_size)
    
    # Final core
    cores.append(current.reshape(ranks[-1], shape[-1], 1))
    ranks.append(1)
    
    return TTTensor(cores=cores, shape=shape, ranks=tuple(ranks))


def tt_to_full(tt: TTTensor) -> np.ndarray:
    """
    Reconstruct full tensor from TT format.
    
    Warning: This may be expensive for high-dimensional tensors!
    Use only for validation or small tensors.
    
    Args:
        tt: TTTensor to reconstruct
        
    Returns:
        Full numpy array of shape tt.shape
    """
    result = tt.cores[0]
    for core in tt.cores[1:]:
        # Contract last index of result with first index of core
        result = np.tensordot(result, core, axes=([-1], [0]))
    
    return result.reshape(tt.shape)


def tt_add(a: TTTensor, b: TTTensor, max_rank: Optional[int] = None) -> TTTensor:
    """
    Add two TT tensors: C = A + B
    
    The result has ranks r_k(C) ≤ r_k(A) + r_k(B).
    Use tt_round afterwards to reduce rank.
    
    Args:
        a, b: TTTensors of same shape
        max_rank: If provided, round result to this rank
        
    Returns:
        TTTensor representing a + b
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    
    d = a.ndim
    cores = []
    
    for k in range(d):
        ca = a.cores[k]
        cb = b.cores[k]
        
        # Direct sum of cores
        ra_left, n, ra_right = ca.shape
        rb_left, _, rb_right = cb.shape
        
        if k == 0:
            # First core: concatenate along right index
            core = np.concatenate([ca, cb], axis=2)
        elif k == d - 1:
            # Last core: concatenate along left index
            core = np.concatenate([ca, cb], axis=0)
        else:
            # Middle cores: block diagonal
            core = np.zeros((ra_left + rb_left, n, ra_right + rb_right))
            core[:ra_left, :, :ra_right] = ca
            core[ra_left:, :, ra_right:] = cb
        
        cores.append(core)
    
    ranks = tuple([1] + [c.shape[2] for c in cores[:-1]] + [1])
    result = TTTensor(cores=cores, shape=a.shape, ranks=ranks)
    
    # Optionally compress
    if max_rank is not None:
        full = tt_to_full(result)
        result = tt_round(full, max_rank=max_rank)
    
    return result


def tt_dot(a: TTTensor, b: TTTensor) -> float:
    """
    Inner product of two TT tensors: <A, B> = Σᵢ A[i]·B[i]
    
    Computed efficiently in TT format without reconstructing full tensors.
    
    Args:
        a, b: TTTensors of same shape
        
    Returns:
        Scalar inner product
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    
    # Start with identity
    result = np.ones((1, 1))
    
    for ca, cb in zip(a.cores, b.cores):
        # Contract: result[i,j] × ca[i,k,m] × cb[j,k,n] → new_result[m,n]
        # First contract over physical index k
        temp = np.einsum('ikm,jkn->ijmn', ca, cb)
        # Then contract with result
        result = np.einsum('ij,ijmn->mn', result, temp)
    
    return float(result[0, 0])


def tt_norm(tt: TTTensor) -> float:
    """Frobenius norm of TT tensor"""
    return np.sqrt(tt_dot(tt, tt))
