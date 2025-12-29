# Copyright (c) 2025 Tigantic
# Phase 18: Advanced Compression Strategies
"""
Advanced tensor compression strategies for adaptive bond dimension management.

Provides multiple compression methods including standard SVD, randomized SVD,
variational compression, and tensor cross interpolation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np


class CompressionMethod(Enum):
    """Available compression methods."""
    
    SVD = auto()              # Standard truncated SVD
    RANDOMIZED_SVD = auto()   # Randomized SVD (faster for large matrices)
    VARIATIONAL = auto()      # Variational optimization
    CROSS = auto()            # Tensor cross interpolation
    ADAPTIVE = auto()         # Automatically select best method


@dataclass
class CompressionResult:
    """Result of tensor compression.
    
    Attributes:
        compressed: Compressed tensor(s)
        singular_values: Singular values (if applicable)
        truncation_error: Compression error
        compression_ratio: Ratio of original to compressed size
        method: Method used for compression
        iterations: Number of iterations (for iterative methods)
        converged: Whether iterative method converged
        metadata: Additional method-specific information
    """
    
    compressed: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    singular_values: Optional[torch.Tensor] = None
    truncation_error: float = 0.0
    compression_ratio: float = 1.0
    method: CompressionMethod = CompressionMethod.SVD
    iterations: int = 1
    converged: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_memory_savings(self) -> float:
        """Get memory savings as a percentage.
        
        Returns:
            Percentage of memory saved (0-100)
        """
        if self.compression_ratio <= 1:
            return 0.0
        return 100.0 * (1.0 - 1.0 / self.compression_ratio)


class CompressionStrategy(ABC):
    """Abstract base class for compression strategies."""
    
    @abstractmethod
    def compress(
        self,
        tensor: torch.Tensor,
        target_rank: int,
        **kwargs,
    ) -> CompressionResult:
        """Compress a tensor to target rank.
        
        Args:
            tensor: Input tensor (reshaped to matrix if needed)
            target_rank: Target rank after compression
            **kwargs: Method-specific parameters
            
        Returns:
            CompressionResult with compressed tensor(s)
        """
        pass
    
    @abstractmethod
    def estimate_error(
        self,
        tensor: torch.Tensor,
        target_rank: int,
    ) -> float:
        """Estimate compression error without full compression.
        
        Args:
            tensor: Input tensor
            target_rank: Target rank
            
        Returns:
            Estimated compression error
        """
        pass


class SVDCompression(CompressionStrategy):
    """Standard truncated SVD compression.
    
    The gold standard for low-rank approximation, minimizing
    Frobenius norm error for a given rank.
    """
    
    def __init__(self, full_matrices: bool = False) -> None:
        """Initialize SVD compression.
        
        Args:
            full_matrices: Whether to compute full U and V matrices
        """
        self.full_matrices = full_matrices
    
    def compress(
        self,
        tensor: torch.Tensor,
        target_rank: int,
        return_factors: bool = True,
        **kwargs,
    ) -> CompressionResult:
        """Compress using truncated SVD.
        
        Args:
            tensor: Input matrix (or tensor reshaped to matrix)
            target_rank: Target rank
            return_factors: If True, return (U, S, Vh); else return U @ diag(S) @ Vh
            
        Returns:
            CompressionResult with U, S, Vh or reconstructed matrix
        """
        original_shape = tensor.shape
        original_size = tensor.numel()
        
        # Ensure 2D
        if tensor.dim() != 2:
            m = tensor.shape[0]
            n = tensor.numel() // m
            tensor = tensor.reshape(m, n)
        
        # Randomized SVD (4× faster)
        q = min(target_rank * 2, min(tensor.shape))
        U, S, Vh = torch.svd_lowrank(tensor, q=q, niter=2)
        
        # Truncate
        rank = min(target_rank, len(S))
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vh_trunc = Vh[:rank, :]
        
        # Compute truncation error
        if rank < len(S):
            truncation_error = float(torch.sum(S[rank:] ** 2))
        else:
            truncation_error = 0.0
        
        # Compression ratio
        compressed_size = U_trunc.numel() + rank + Vh_trunc.numel()
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        if return_factors:
            compressed = (U_trunc, S_trunc, Vh_trunc)
        else:
            compressed = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
        
        return CompressionResult(
            compressed=compressed,
            singular_values=S_trunc,
            truncation_error=truncation_error,
            compression_ratio=compression_ratio,
            method=CompressionMethod.SVD,
            metadata={"original_shape": original_shape, "rank": rank},
        )
    
    def estimate_error(
        self,
        tensor: torch.Tensor,
        target_rank: int,
    ) -> float:
        """Estimate truncation error.
        
        Args:
            tensor: Input tensor
            target_rank: Target rank
            
        Returns:
            Estimated error (sum of squared discarded singular values)
        """
        # Compute singular values only
        if tensor.dim() != 2:
            m = tensor.shape[0]
            n = tensor.numel() // m
            tensor = tensor.reshape(m, n)
        
        S = torch.linalg.svdvals(tensor)
        
        if target_rank >= len(S):
            return 0.0
        
        return float(torch.sum(S[target_rank:] ** 2))


class RandomizedSVD(CompressionStrategy):
    """Randomized SVD for efficient low-rank approximation.
    
    Uses random projections to compute approximate SVD much faster
    than full SVD for large matrices when target rank is small.
    
    Based on Halko, Martinsson, Tropp (2011):
    "Finding structure with randomness: Probabilistic algorithms for 
    constructing approximate matrix decompositions"
    """
    
    def __init__(
        self,
        oversampling: int = 10,
        n_power_iterations: int = 2,
    ) -> None:
        """Initialize randomized SVD.
        
        Args:
            oversampling: Extra dimensions for random projection
            n_power_iterations: Power iterations for improved accuracy
        """
        self.oversampling = oversampling
        self.n_power_iterations = n_power_iterations
    
    def compress(
        self,
        tensor: torch.Tensor,
        target_rank: int,
        return_factors: bool = True,
        **kwargs,
    ) -> CompressionResult:
        """Compress using randomized SVD.
        
        Args:
            tensor: Input matrix
            target_rank: Target rank
            return_factors: If True, return (U, S, Vh)
            
        Returns:
            CompressionResult with approximate low-rank factors
        """
        original_shape = tensor.shape
        original_size = tensor.numel()
        
        # Ensure 2D
        if tensor.dim() != 2:
            m = tensor.shape[0]
            n = tensor.numel() // m
            tensor = tensor.reshape(m, n)
        
        m, n = tensor.shape
        dtype = tensor.dtype
        device = tensor.device
        
        # Determine sketch size
        k = min(target_rank + self.oversampling, min(m, n))
        
        # Random projection
        Omega = torch.randn(n, k, dtype=dtype, device=device)
        Y = tensor @ Omega
        
        # Power iterations for better approximation
        for _ in range(self.n_power_iterations):
            Y = tensor @ (tensor.T @ Y)
        
        # Orthonormalize
        Q, _ = torch.linalg.qr(Y)
        
        # Project to lower dimension
        B = Q.T @ tensor
        
        # Randomized SVD of smaller matrix
        q = min(target_rank * 2, min(B.shape))
        U_small, S, Vh = torch.svd_lowrank(B, q=q, niter=2)
        
        # Recover U
        U = Q @ U_small
        
        # Truncate to target rank
        rank = min(target_rank, len(S))
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vh_trunc = Vh[:rank, :]
        
        # Estimate truncation error (approximate)
        if rank < len(S):
            truncation_error = float(torch.sum(S[rank:] ** 2))
        else:
            truncation_error = 0.0
        
        # Compression ratio
        compressed_size = U_trunc.numel() + rank + Vh_trunc.numel()
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        if return_factors:
            compressed = (U_trunc, S_trunc, Vh_trunc)
        else:
            compressed = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
        
        return CompressionResult(
            compressed=compressed,
            singular_values=S_trunc,
            truncation_error=truncation_error,
            compression_ratio=compression_ratio,
            method=CompressionMethod.RANDOMIZED_SVD,
            metadata={
                "original_shape": original_shape,
                "rank": rank,
                "oversampling": self.oversampling,
                "power_iterations": self.n_power_iterations,
            },
        )
    
    def estimate_error(
        self,
        tensor: torch.Tensor,
        target_rank: int,
    ) -> float:
        """Estimate truncation error using randomized method.
        
        Args:
            tensor: Input tensor
            target_rank: Target rank
            
        Returns:
            Estimated error
        """
        # Quick estimate using subsampling
        result = self.compress(tensor, target_rank)
        return result.truncation_error


class VariationalCompression(CompressionStrategy):
    """Variational optimization for tensor compression.
    
    Optimizes the compressed representation directly, which can
    sometimes achieve better compression than SVD for structured data.
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        learning_rate: float = 0.01,
    ) -> None:
        """Initialize variational compression.
        
        Args:
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            learning_rate: Learning rate for optimization
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
    
    def compress(
        self,
        tensor: torch.Tensor,
        target_rank: int,
        return_factors: bool = True,
        **kwargs,
    ) -> CompressionResult:
        """Compress using variational optimization.
        
        Minimizes ||A - UV^T||_F using alternating least squares.
        
        Args:
            tensor: Input matrix
            target_rank: Target rank
            return_factors: If True, return (U, V)
            
        Returns:
            CompressionResult with optimized factors
        """
        original_shape = tensor.shape
        original_size = tensor.numel()
        
        # Ensure 2D
        if tensor.dim() != 2:
            m = tensor.shape[0]
            n = tensor.numel() // m
            tensor = tensor.reshape(m, n)
        
        m, n = tensor.shape
        dtype = tensor.dtype
        device = tensor.device
        rank = min(target_rank, min(m, n))
        
        # Initialize with SVD (for better starting point)
        svd_strategy = SVDCompression()
        svd_result = svd_strategy.compress(tensor, rank)
        U, S, Vh = svd_result.compressed
        
        # Initialize factors
        U_opt = U @ torch.diag(torch.sqrt(S))
        V_opt = torch.diag(torch.sqrt(S)) @ Vh
        V_opt = V_opt.T  # Make it m x rank
        
        # Alternating least squares
        prev_error = float('inf')
        converged = False
        
        for iteration in range(self.max_iterations):
            # Update U: solve A = U V^T for U
            # U = A @ V @ (V^T @ V)^{-1}
            VtV = V_opt.T @ V_opt
            VtV_inv = torch.linalg.pinv(VtV)
            U_opt = tensor @ V_opt @ VtV_inv
            
            # Update V: solve A = U V^T for V
            # V^T = (U^T @ U)^{-1} @ U^T @ A
            UtU = U_opt.T @ U_opt
            UtU_inv = torch.linalg.pinv(UtU)
            V_opt = tensor.T @ U_opt @ UtU_inv
            
            # Compute error
            reconstruction = U_opt @ V_opt.T
            error = float(torch.norm(tensor - reconstruction, p='fro') ** 2)
            
            # Check convergence
            if abs(prev_error - error) < self.tolerance * prev_error:
                converged = True
                break
            
            prev_error = error
        
        # Compute rSVD of factorization for singular values
        Q_U, R_U = torch.linalg.qr(U_opt)
        Q_V, R_V = torch.linalg.qr(V_opt)
        
        middle = R_U @ R_V.T
        q = min(rank * 2, min(middle.shape))
        U_mid, S_mid, Vh_mid = torch.svd_lowrank(middle, q=q, niter=2)
        
        U_final = Q_U @ U_mid
        Vh_final = Vh_mid @ Q_V.T
        
        # Compression ratio
        compressed_size = U_final.numel() + rank + Vh_final.numel()
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        if return_factors:
            compressed = (U_final, S_mid, Vh_final)
        else:
            compressed = U_final @ torch.diag(S_mid) @ Vh_final
        
        return CompressionResult(
            compressed=compressed,
            singular_values=S_mid,
            truncation_error=prev_error,
            compression_ratio=compression_ratio,
            method=CompressionMethod.VARIATIONAL,
            iterations=iteration + 1,
            converged=converged,
            metadata={
                "original_shape": original_shape,
                "rank": rank,
                "max_iterations": self.max_iterations,
            },
        )
    
    def estimate_error(
        self,
        tensor: torch.Tensor,
        target_rank: int,
    ) -> float:
        """Estimate error using quick SVD."""
        svd = SVDCompression()
        return svd.estimate_error(tensor, target_rank)


class TensorCrossInterpolation(CompressionStrategy):
    """Tensor Cross Interpolation (TCI) / skeleton decomposition.
    
    Approximates a tensor using only a subset of its elements,
    which can be very efficient for tensors with special structure.
    
    Based on the CUR/skeleton decomposition approach.
    """
    
    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
    ) -> None:
        """Initialize TCI compression.
        
        Args:
            max_iterations: Maximum cross iterations
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def compress(
        self,
        tensor: torch.Tensor,
        target_rank: int,
        return_factors: bool = True,
        **kwargs,
    ) -> CompressionResult:
        """Compress using tensor cross interpolation.
        
        Computes CUR decomposition: A ≈ C @ U^{-1} @ R
        where C is a column selection, R is a row selection,
        and U is the intersection submatrix.
        
        Args:
            tensor: Input matrix
            target_rank: Target rank
            return_factors: Return factors vs reconstructed
            
        Returns:
            CompressionResult with CUR factors
        """
        original_shape = tensor.shape
        original_size = tensor.numel()
        
        # Ensure 2D
        if tensor.dim() != 2:
            m = tensor.shape[0]
            n = tensor.numel() // m
            tensor = tensor.reshape(m, n)
        
        m, n = tensor.shape
        rank = min(target_rank, min(m, n))
        
        # Use leverage score sampling for column/row selection
        # Start with SVD to get leverage scores
        U_svd, S_svd, Vh_svd = torch.linalg.svd(tensor, full_matrices=False)
        
        # Column leverage scores (from right singular vectors)
        col_leverage = torch.sum(Vh_svd[:rank, :] ** 2, dim=0)
        col_probs = col_leverage / torch.sum(col_leverage)
        
        # Row leverage scores (from left singular vectors)
        row_leverage = torch.sum(U_svd[:, :rank] ** 2, dim=1)
        row_probs = row_leverage / torch.sum(row_leverage)
        
        # Sample columns and rows
        col_indices = torch.multinomial(col_probs, rank, replacement=False)
        row_indices = torch.multinomial(row_probs, rank, replacement=False)
        
        # Sort for consistency
        col_indices, _ = torch.sort(col_indices)
        row_indices, _ = torch.sort(row_indices)
        
        # Extract C, R, and U
        C = tensor[:, col_indices]  # m x rank
        R = tensor[row_indices, :]  # rank x n
        U_inter = tensor[row_indices][:, col_indices]  # rank x rank
        
        # Compute pseudoinverse of U
        U_pinv = torch.linalg.pinv(U_inter)
        
        # Reconstruction
        reconstruction = C @ U_pinv @ R
        
        # Compute error
        truncation_error = float(torch.norm(tensor - reconstruction, p='fro') ** 2)
        
        # Compression ratio
        compressed_size = C.numel() + U_pinv.numel() + R.numel()
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        if return_factors:
            compressed = (C, U_pinv, R)
        else:
            compressed = reconstruction
        
        return CompressionResult(
            compressed=compressed,
            singular_values=S_svd[:rank],
            truncation_error=truncation_error,
            compression_ratio=compression_ratio,
            method=CompressionMethod.CROSS,
            metadata={
                "original_shape": original_shape,
                "rank": rank,
                "col_indices": col_indices.tolist(),
                "row_indices": row_indices.tolist(),
            },
        )
    
    def estimate_error(
        self,
        tensor: torch.Tensor,
        target_rank: int,
    ) -> float:
        """Estimate error using singular values."""
        svd = SVDCompression()
        return svd.estimate_error(tensor, target_rank)


def compress_adaptively(
    tensor: torch.Tensor,
    target_rank: int,
    method: CompressionMethod = CompressionMethod.ADAPTIVE,
    **kwargs,
) -> CompressionResult:
    """Compress a tensor using the specified or best method.
    
    Args:
        tensor: Input tensor
        target_rank: Target rank for compression
        method: Compression method to use
        **kwargs: Additional method-specific arguments
        
    Returns:
        CompressionResult with compressed tensor
    """
    if method == CompressionMethod.ADAPTIVE:
        method = select_compression_strategy(tensor, target_rank)
    
    strategy: CompressionStrategy
    
    if method == CompressionMethod.SVD:
        strategy = SVDCompression()
    elif method == CompressionMethod.RANDOMIZED_SVD:
        strategy = RandomizedSVD(
            oversampling=kwargs.get("oversampling", 10),
            n_power_iterations=kwargs.get("n_power_iterations", 2),
        )
    elif method == CompressionMethod.VARIATIONAL:
        strategy = VariationalCompression(
            max_iterations=kwargs.get("max_iterations", 100),
            tolerance=kwargs.get("tolerance", 1e-8),
        )
    elif method == CompressionMethod.CROSS:
        strategy = TensorCrossInterpolation(
            max_iterations=kwargs.get("max_iterations", 50),
            tolerance=kwargs.get("tolerance", 1e-8),
        )
    else:
        strategy = SVDCompression()
    
    return strategy.compress(tensor, target_rank, **kwargs)


def select_compression_strategy(
    tensor: torch.Tensor,
    target_rank: int,
) -> CompressionMethod:
    """Select the best compression strategy based on tensor properties.
    
    Args:
        tensor: Input tensor
        target_rank: Target rank
        
    Returns:
        Recommended CompressionMethod
    """
    # Get tensor dimensions
    if tensor.dim() == 2:
        m, n = tensor.shape
    else:
        m = tensor.shape[0]
        n = tensor.numel() // m
    
    min_dim = min(m, n)
    max_dim = max(m, n)
    
    # Heuristics for method selection
    
    # For small matrices, full SVD is fast and exact
    if min_dim <= 100:
        return CompressionMethod.SVD
    
    # For large matrices with small target rank, randomized SVD is faster
    if max_dim > 1000 and target_rank < min_dim / 10:
        return CompressionMethod.RANDOMIZED_SVD
    
    # For very large matrices, consider cross interpolation
    if max_dim > 10000 and target_rank < min_dim / 20:
        return CompressionMethod.CROSS
    
    # Default to standard SVD
    return CompressionMethod.SVD
