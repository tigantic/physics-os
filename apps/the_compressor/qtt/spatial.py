"""
QTT Spatial Compression Module
==============================

TT-SVD (Tensor Train Singular Value Decomposition) for N-dimensional data.

Features:
- O(N·r²) storage vs O(N^d) for raw tensors
- O(d·r²) random access per element
- Automatic rank selection via singular value thresholding
- GPU acceleration support (optional)

Mathematical Foundation:
    A tensor T ∈ ℝ^{n₁×n₂×...×n_d} is decomposed as:
    
    T[i₁,i₂,...,i_d] = G₁[:,i₁,:] · G₂[:,i₂,:] · ... · G_d[:,i_d,:]
    
    where G_k ∈ ℝ^{r_{k-1}×n_k×r_k} are the TT cores.

Usage:
    >>> from qtt.spatial import SpatialCompressor, tt_svd
    
    # Compress a 3D field
    >>> compressor = SpatialCompressor(max_rank=64)
    >>> cores = compressor.compress(temperature_field)
    
    # Random access
    >>> value = compressor.reconstruct_element((x, y, z))
    
    # Full reconstruction
    >>> reconstructed = compressor.reconstruct()
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class CompressionStats:
    """Statistics from TT-SVD compression."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    ranks: List[int]
    max_rank_used: int
    relative_error: float


def tt_svd(tensor: np.ndarray, 
           max_rank: int = 64,
           rel_eps: float = 1e-10) -> List[np.ndarray]:
    """
    Tensor Train SVD decomposition.
    
    Args:
        tensor: N-dimensional numpy array
        max_rank: Maximum TT rank (default 64)
        rel_eps: Relative epsilon for rank truncation
        
    Returns:
        List of TT cores [G₁, G₂, ..., G_d]
        where G_k has shape (r_{k-1}, n_k, r_k)
    """
    shape = tensor.shape
    n_modes = len(shape)
    
    if n_modes < 2:
        raise ValueError("Tensor must have at least 2 dimensions")
    
    cores = []
    remaining = tensor.astype(np.float64)
    r_prev = 1
    
    for i in range(n_modes - 1):
        # Reshape to (r_prev * n_i) x (remaining dimensions)
        remaining = remaining.reshape(r_prev * shape[i], -1)
        
        # SVD
        U, S, Vt = np.linalg.svd(remaining, full_matrices=False)
        
        # Determine rank via thresholding
        rank = min(max_rank, len(S))
        if S[0] > 0:
            threshold = rel_eps * S[0]
            sig_mask = S > threshold
            rank = min(rank, max(1, int(np.sum(sig_mask))))
        rank = max(1, rank)
        
        # Truncate
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        # Form core
        core = U.reshape(r_prev, shape[i], rank)
        cores.append(core.astype(np.float32))
        
        # Remaining tensor
        remaining = np.diag(S) @ Vt
        r_prev = rank
    
    # Final core
    cores.append(remaining.reshape(r_prev, shape[-1], 1).astype(np.float32))
    
    return cores


def tt_reconstruct(cores: List[np.ndarray]) -> np.ndarray:
    """
    Reconstruct full tensor from TT cores.
    
    Args:
        cores: List of TT cores from tt_svd
        
    Returns:
        Reconstructed N-dimensional array
    """
    n_cores = len(cores)
    shape = tuple(core.shape[1] for core in cores)
    
    # Start with first core
    result = cores[0]  # (1, n₁, r₁)
    
    for i in range(1, n_cores):
        # result: (1, n₁×...×n_i, r_i)
        # cores[i]: (r_i, n_{i+1}, r_{i+1})
        r_left = result.shape[0]
        n_left = result.shape[1]
        r_mid = result.shape[2]
        
        n_next = cores[i].shape[1]
        r_right = cores[i].shape[2]
        
        # Contract: (r_left, n_left, r_mid) × (r_mid, n_next, r_right)
        # -> (r_left, n_left, n_next, r_right)
        result = np.einsum('ijk,klm->ijlm', result, cores[i])
        result = result.reshape(r_left, n_left * n_next, r_right)
    
    return result.reshape(shape)


def tt_reconstruct_element(cores: List[np.ndarray], 
                           indices: Tuple[int, ...]) -> float:
    """
    Reconstruct a single element via TT contraction.
    
    O(d·r²) complexity where d = number of modes, r = max rank.
    
    Args:
        cores: List of TT cores
        indices: Tuple of indices (i₁, i₂, ..., i_d)
        
    Returns:
        Reconstructed scalar value
    """
    if len(indices) != len(cores):
        raise ValueError(f"Expected {len(cores)} indices, got {len(indices)}")
    
    # Extract slices and contract
    result = cores[0][:, indices[0], :]  # (1, r₁)
    
    for i in range(1, len(cores)):
        slice_i = cores[i][:, indices[i], :]  # (r_i, r_{i+1})
        result = result @ slice_i
    
    return float(result.flatten()[0])


class SpatialCompressor:
    """
    High-level interface for spatial data compression.
    
    Example:
        >>> compressor = SpatialCompressor(max_rank=32)
        >>> compressor.compress(data)
        >>> print(compressor.stats)
        >>> value = compressor.reconstruct_element((10, 20, 30))
    """
    
    def __init__(self, 
                 max_rank: int = 64,
                 rel_eps: float = 1e-10,
                 dtype: np.dtype = np.float32):
        """
        Initialize compressor.
        
        Args:
            max_rank: Maximum TT rank
            rel_eps: Relative epsilon for rank truncation
            dtype: Output dtype for cores
        """
        self.max_rank = max_rank
        self.rel_eps = rel_eps
        self.dtype = dtype
        
        self._cores: Optional[List[np.ndarray]] = None
        self._shape: Optional[Tuple[int, ...]] = None
        self._original_dtype: Optional[np.dtype] = None
        self._stats: Optional[CompressionStats] = None
    
    @property
    def cores(self) -> List[np.ndarray]:
        """Get TT cores."""
        if self._cores is None:
            raise ValueError("No data compressed yet. Call compress() first.")
        return self._cores
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get original tensor shape."""
        if self._shape is None:
            raise ValueError("No data compressed yet. Call compress() first.")
        return self._shape
    
    @property
    def stats(self) -> CompressionStats:
        """Get compression statistics."""
        if self._stats is None:
            raise ValueError("No data compressed yet. Call compress() first.")
        return self._stats
    
    def compress(self, 
                 tensor: np.ndarray,
                 compute_error: bool = True) -> List[np.ndarray]:
        """
        Compress tensor using TT-SVD.
        
        Args:
            tensor: N-dimensional numpy array
            compute_error: Whether to compute reconstruction error
            
        Returns:
            List of TT cores
        """
        self._shape = tensor.shape
        self._original_dtype = tensor.dtype
        
        # Compute TT-SVD
        self._cores = tt_svd(tensor, self.max_rank, self.rel_eps)
        
        # Compute statistics
        original_size = tensor.nbytes
        compressed_size = sum(core.nbytes for core in self._cores)
        ranks = [1] + [core.shape[2] for core in self._cores[:-1]] + [1]
        
        # Compute error if requested
        if compute_error:
            reconstructed = tt_reconstruct(self._cores)
            rel_error = np.linalg.norm(reconstructed - tensor) / np.linalg.norm(tensor)
        else:
            rel_error = float('nan')
        
        self._stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size,
            ranks=ranks,
            max_rank_used=max(ranks),
            relative_error=rel_error
        )
        
        return self._cores
    
    def reconstruct(self) -> np.ndarray:
        """Reconstruct full tensor from cores."""
        return tt_reconstruct(self.cores).astype(self._original_dtype)
    
    def reconstruct_element(self, indices: Tuple[int, ...]) -> float:
        """Reconstruct single element at given indices."""
        return tt_reconstruct_element(self.cores, indices)
    
    def save(self, path: str):
        """Save cores to .npz file."""
        np.savez_compressed(
            path,
            *self.cores,
            shape=np.array(self._shape),
            dtype=str(self._original_dtype)
        )
    
    @classmethod
    def load(cls, path: str) -> 'SpatialCompressor':
        """Load compressor from .npz file."""
        data = np.load(path, allow_pickle=True)
        
        compressor = cls()
        compressor._shape = tuple(data['shape'])
        compressor._original_dtype = np.dtype(str(data['dtype']))
        
        # Load cores (arr_0, arr_1, ...)
        cores = []
        i = 0
        while f'arr_{i}' in data:
            cores.append(data[f'arr_{i}'])
            i += 1
        compressor._cores = cores
        
        return compressor


# =============================================================================
# GPU Acceleration (Optional)
# =============================================================================

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def tt_svd_gpu(tensor: np.ndarray,
               max_rank: int = 64,
               rel_eps: float = 1e-10) -> List[np.ndarray]:
    """
    GPU-accelerated TT-SVD using CuPy.
    
    Falls back to CPU if CuPy not available.
    """
    if not HAS_GPU:
        return tt_svd(tensor, max_rank, rel_eps)
    
    shape = tensor.shape
    n_modes = len(shape)
    
    cores = []
    remaining = cp.asarray(tensor.astype(np.float64))
    r_prev = 1
    
    for i in range(n_modes - 1):
        remaining = remaining.reshape(r_prev * shape[i], -1)
        
        U, S, Vt = cp.linalg.svd(remaining, full_matrices=False)
        
        rank = min(max_rank, len(S))
        if S[0] > 0:
            threshold = rel_eps * S[0]
            sig_mask = S > threshold
            rank = min(rank, max(1, int(cp.sum(sig_mask))))
        rank = max(1, rank)
        
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        core = U.reshape(r_prev, shape[i], rank)
        cores.append(cp.asnumpy(core).astype(np.float32))
        
        remaining = cp.diag(S) @ Vt
        r_prev = rank
    
    cores.append(cp.asnumpy(remaining.reshape(r_prev, shape[-1], 1)).astype(np.float32))
    
    return cores


# =============================================================================
# Utilities
# =============================================================================

def estimate_compression_ratio(shape: Tuple[int, ...], 
                               max_rank: int) -> float:
    """
    Estimate compression ratio for given shape and rank.
    
    Args:
        shape: Tensor shape
        max_rank: Maximum TT rank
        
    Returns:
        Estimated compression ratio
    """
    n_elements = np.prod(shape)
    
    # TT storage: sum over modes of r_{k-1} * n_k * r_k
    tt_storage = 0
    ranks = [1] + [max_rank] * (len(shape) - 1) + [1]
    for i, n in enumerate(shape):
        tt_storage += ranks[i] * n * ranks[i + 1]
    
    return n_elements / tt_storage


def recommend_rank(tensor: np.ndarray,
                   target_ratio: float = 10.0,
                   target_error: float = 1e-3) -> int:
    """
    Recommend max_rank for given tensor and targets.
    
    Args:
        tensor: Input tensor
        target_ratio: Target compression ratio
        target_error: Target relative error
        
    Returns:
        Recommended max_rank
    """
    shape = tensor.shape
    
    # Binary search for rank
    low, high = 1, min(tensor.shape)
    best_rank = high
    
    for rank in [2, 4, 8, 16, 32, 64, 128, 256]:
        if rank > high:
            break
            
        ratio = estimate_compression_ratio(shape, rank)
        
        if ratio >= target_ratio:
            # Test actual error
            cores = tt_svd(tensor, rank)
            recon = tt_reconstruct(cores)
            error = np.linalg.norm(recon - tensor) / np.linalg.norm(tensor)
            
            if error <= target_error:
                best_rank = rank
                break
    
    return best_rank
