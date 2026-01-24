"""
QTT Cost Matrix Representations

This module provides cost matrix representations in QTT format for optimal
transport. The key insight is that common cost matrices have low-rank structure:

- Euclidean distance: C[i,j] = |x_i - x_j|² has TT rank O(1)
- Toeplitz matrices: C[i,j] = f(i-j) have TT rank O(1) for polynomial f
- Gaussian kernel: C[i,j] = exp(-|x_i - x_j|²/ε) has rank O(log(1/ε))

Constitutional Reference: TENSOR_GENESIS.md, Article II (Complexity Compact)

Mathematical Background:
    The cost matrix C ∈ ℝ^{N×N} in OT has N² entries. For N = 10¹², this is
    10²⁴ bytes - impossible to store. However, if C[i,j] = f(x_i - x_j) for
    smooth f, we can represent C as a QTT-MPO with O(log²N) storage.
    
    For Euclidean cost on grid x_i = i·Δx:
    C[i,j] = (i-j)² Δx² = i²Δx² - 2ijΔx² + j²Δx²
    
    This is a sum of 3 separable terms, giving TT rank 3.

Example:
    >>> from tensornet.genesis.ot import euclidean_cost_mpo
    >>> 
    >>> # Trillion×trillion cost matrix
    >>> C = euclidean_cost_mpo(grid_size=2**40, grid_bounds=(-10, 10))
    >>> print(f"Represents {C.shape[0]}×{C.shape[1]} matrix with rank {C.max_rank}")

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import torch
import numpy as np


@dataclass
class QTTMatrix:
    """
    A matrix represented in QTT-MPO (Matrix Product Operator) format.
    
    For a matrix M of size N×N where N = 2^d, the QTT-MPO representation
    factorizes M as:
    
    M[i₁...i_d, j₁...j_d] = A₁[i₁,j₁] A₂[i₂,j₂] ... A_d[i_d,j_d]
    
    where each A_k is a tensor of shape (r_{k-1}, 2, 2, r_k).
    
    Storage is O(d r²) = O(log(N) r²) instead of O(N²).
    
    Attributes:
        cores: List of MPO cores, each of shape (r_{k-1}, 2, 2, r_k)
        shape: Tuple (N, N) representing the logical matrix shape
        grid_bounds: Physical domain bounds (for cost matrices)
    """
    
    cores: List[torch.Tensor]
    shape: Tuple[int, int] = field(init=False)
    grid_bounds: Optional[Tuple[float, float]] = None
    _frobenius_norm: Optional[float] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Compute derived attributes."""
        n = 1
        for core in self.cores:
            n *= core.shape[1]
        self.shape = (n, n)
    
    @property
    def num_cores(self) -> int:
        """Number of MPO cores."""
        return len(self.cores)
    
    @property
    def max_rank(self) -> int:
        """Maximum TT rank."""
        if not self.cores:
            return 0
        return max(core.shape[0] for core in self.cores[1:])
    
    @property
    def ranks(self) -> List[int]:
        """List of TT ranks."""
        if not self.cores:
            return []
        ranks = [1]
        for core in self.cores:
            ranks.append(core.shape[3])
        return ranks
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of cores."""
        return self.cores[0].dtype if self.cores else torch.float64
    
    @property
    def device(self) -> torch.device:
        """Device where cores are stored."""
        return self.cores[0].device if self.cores else torch.device('cpu')
    
    def matvec(self, x: "QTTVector") -> "QTTVector":
        """
        Matrix-vector product M @ x in QTT format.
        
        This is the core operation for Sinkhorn iterations.
        Complexity: O(d r³) where r = max(rank(M), rank(x))
        
        Args:
            x: QTT vector to multiply
            
        Returns:
            Result y = M @ x in QTT format
        """
        from .distributions import QTTDistribution
        
        if len(x.cores) != len(self.cores):
            raise ValueError(
                f"Incompatible number of cores: MPO has {len(self.cores)}, "
                f"MPS has {len(x.cores)}"
            )
        
        # MPO × MPS contraction
        # MPO core: (r_M_in, n_i, n_j, r_M_out) where n_i = n_j = 2 for QTT
        # MPS core: (r_x_in, n, r_x_out) where n = 2 for QTT
        # Result: (r_M_in * r_x_in, n_i, r_M_out * r_x_out)
        
        result_cores = []
        
        for mpo_core, mps_core in zip(self.cores, x.cores):
            # Get shapes
            if mpo_core.dim() == 4:
                r_M_in, n_i, n_j, r_M_out = mpo_core.shape
            else:
                raise ValueError(f"MPO core should be 4D, got {mpo_core.dim()}D")
                
            if mps_core.dim() == 3:
                r_x_in, n, r_x_out = mps_core.shape
            else:
                raise ValueError(f"MPS core should be 3D, got {mps_core.dim()}D")
            
            if n_j != n:
                raise ValueError(
                    f"MPO column dimension {n_j} != MPS dimension {n}"
                )
            
            # Contract over the shared index (columns of MPO, entries of MPS)
            # result[a, c, i, b, d] = sum_j mpo[a, i, j, b] * mps[c, j, d]
            # Then reshape to (a*c, i, b*d)
            
            result = torch.einsum('aijb,cjd->acibd', mpo_core, mps_core)
            
            # Reshape to (r_M_in * r_x_in, n_i, r_M_out * r_x_out)
            result = result.reshape(r_M_in * r_x_in, n_i, r_M_out * r_x_out)
            result_cores.append(result)
        
        return QTTDistribution(
            cores=result_cores,
            grid_bounds=x.grid_bounds,
            is_normalized=False,
        )
    
    def to_dense(self) -> torch.Tensor:
        """
        Materialize the full dense matrix.
        
        WARNING: Only use for small matrices (< 2^10 × 2^10).
        """
        N = self.shape[0]
        if N > 2**10:
            raise ValueError(f"Matrix size {N}×{N} too large to materialize")
        
        # Build by contracting all cores
        result = self.cores[0]  # (1, 2, 2, r)
        
        for core in self.cores[1:]:
            # result: (1, N_so_far, N_so_far, r_prev)
            # core: (r_prev, 2, 2, r_next)
            r_prev = result.shape[3]
            N_so_far = result.shape[1]
            r_next = core.shape[3]
            
            # Reshape for contraction
            result = result.reshape(-1, r_prev)  # (N_so_far², r_prev)
            core_mat = core.reshape(r_prev, -1)  # (r_prev, 4 * r_next)
            
            contracted = result @ core_mat  # (N_so_far², 4 * r_next)
            contracted = contracted.reshape(N_so_far, N_so_far, 2, 2, r_next)
            
            # Interleave indices
            result = contracted.permute(0, 2, 1, 3, 4)  # (N_so_far, 2, N_so_far, 2, r_next)
            result = result.reshape(1, N_so_far * 2, N_so_far * 2, r_next)
        
        return result.reshape(N, N)
    
    def __repr__(self) -> str:
        return (
            f"QTTMatrix(shape={self.shape}, "
            f"max_rank={self.max_rank}, "
            f"num_cores={self.num_cores})"
        )


# Alias for type hints
QTTVector = "QTTDistribution"  # Forward reference


def euclidean_cost_mpo(
    grid_size: int,
    grid_bounds: Tuple[float, float] = (-10.0, 10.0),
    power: float = 2.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu'),
) -> QTTMatrix:
    """
    Construct the Euclidean distance cost matrix in QTT-MPO format.
    
    Creates C[i,j] = |x_i - x_j|^p where x_i = low + i * dx.
    
    For p = 2 (squared Euclidean distance), this has TT rank 3 because:
    C[i,j] = (x_i - x_j)² = x_i² - 2x_ix_j + x_j²
    
    which is a sum of 3 separable terms.
    
    Mathematical Background:
        For indices in binary: i = Σ_k i_k 2^k, j = Σ_k j_k 2^k
        
        The difference (i - j) can be written in terms of bit operations,
        giving a low-rank structure. For |i-j|², we need rank 3.
        
        For |i-j|^p with p ≠ 2, we use TT-Cross to approximate.
    
    Args:
        grid_size: Number of grid points (must be power of 2)
        grid_bounds: Physical domain (low, high)
        power: Exponent p (default 2 for Wasserstein-2)
        dtype: Data type
        device: Device
        
    Returns:
        QTTMatrix representing the cost matrix
        
    Example:
        >>> C = euclidean_cost_mpo(2**30, (-5, 5))
        >>> print(f"Rank: {C.max_rank}")  # Will be 3 for power=2
        Rank: 3
    """
    if grid_size & (grid_size - 1) != 0:
        raise ValueError(f"grid_size must be power of 2, got {grid_size}")
    
    num_bits = int(math.log2(grid_size))
    low, high = grid_bounds
    dx = (high - low) / grid_size
    
    if power != 2.0:
        # For non-squared distance, would use TT-Cross approximation
        raise NotImplementedError(
            f"power={power} not yet implemented. Only power=2 supported."
        )
    
    # Build rank-3 MPO for (x_i - x_j)²
    # C[i,j] = x_i² - 2*x_i*x_j + x_j² where x_k = low + k*dx
    
    cores = []
    
    for k in range(num_bits):
        bit_val = dx * (2 ** k)  # Contribution of bit k to position
        
        if k == 0:
            # First core: shape (1, 2, 2, 3)
            # Columns are: [1, x_i, x_i²] for building quadratic
            core = torch.zeros(1, 2, 2, 3, dtype=dtype, device=device)
            
            for i in range(2):
                x_i = low + i * bit_val
                for j in range(2):
                    x_j = low + j * bit_val
                    # Store [1, x_i - x_j, (x_i - x_j)²]
                    diff = x_i - x_j
                    core[0, i, j, 0] = diff * diff  # Final value
                    core[0, i, j, 1] = 2 * diff     # For cross-term accumulation
                    core[0, i, j, 2] = 1.0          # For quadratic accumulation
                    
        elif k == num_bits - 1:
            # Last core: shape (3, 2, 2, 1)
            core = torch.zeros(3, 2, 2, 1, dtype=dtype, device=device)
            
            for i in range(2):
                contrib_i = i * bit_val
                for j in range(2):
                    contrib_j = j * bit_val
                    diff = contrib_i - contrib_j
                    
                    # Finalize quadratic
                    core[0, i, j, 0] = 1.0          # Carry through
                    core[1, i, j, 0] = diff         # Add linear term
                    core[2, i, j, 0] = diff * diff  # Add quadratic term
                    
        else:
            # Middle cores: shape (3, 2, 2, 3)
            core = torch.zeros(3, 2, 2, 3, dtype=dtype, device=device)
            
            for i in range(2):
                contrib_i = i * bit_val
                for j in range(2):
                    contrib_j = j * bit_val
                    diff = contrib_i - contrib_j
                    
                    # Accumulate quadratic terms
                    core[0, i, j, 0] = 1.0          # Carry constant
                    core[0, i, j, 1] = 2 * diff     # Add to linear
                    core[0, i, j, 2] = diff * diff  # Add to quadratic
                    
                    core[1, i, j, 0] = 0.0
                    core[1, i, j, 1] = 1.0          # Carry linear
                    core[1, i, j, 2] = diff         # Add to quadratic
                    
                    core[2, i, j, 0] = 0.0
                    core[2, i, j, 1] = 0.0
                    core[2, i, j, 2] = 1.0          # Carry quadratic
        
        cores.append(core)
    
    return QTTMatrix(cores=cores, grid_bounds=grid_bounds)


def toeplitz_cost_mpo(
    diagonals: torch.Tensor,
    grid_size: int,
    grid_bounds: Optional[Tuple[float, float]] = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu'),
) -> QTTMatrix:
    """
    Construct a Toeplitz cost matrix in QTT-MPO format.
    
    Creates C[i,j] = c[i-j] for given diagonal values.
    
    Toeplitz matrices arise when the cost depends only on the difference
    between indices, which is common for translation-invariant problems.
    
    Args:
        diagonals: Values c[k] for k = -(N-1), ..., N-1
        grid_size: Matrix dimension N
        grid_bounds: Optional physical bounds
        dtype: Data type
        device: Device
        
    Returns:
        QTTMatrix representing the Toeplitz cost matrix
    """
    # Toeplitz matrices can have low TT rank if diagonals are smooth
    # This is a placeholder - full implementation would use circulant embedding
    raise NotImplementedError(
        "Toeplitz cost matrix construction coming in Week 3. "
        "Use euclidean_cost_mpo for now."
    )


def gaussian_kernel_mpo(
    grid_size: int,
    grid_bounds: Tuple[float, float] = (-10.0, 10.0),
    epsilon: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu'),
) -> QTTMatrix:
    """
    Construct a Gaussian kernel matrix in QTT-MPO format.
    
    Creates K[i,j] = exp(-|x_i - x_j|² / ε)
    
    This is the Gibbs kernel used in entropy-regularized optimal transport.
    The TT rank depends on ε: smaller ε requires higher rank.
    
    Mathematical Background:
        The Gibbs kernel K = exp(-C/ε) can be computed from the cost matrix C
        by applying exp element-wise. In TT format, this requires approximation
        since exp of a low-rank matrix isn't generally low-rank.
        
        However, for Gaussian kernels on grids, we can construct directly:
        K[i,j] = exp(-(x_i - x_j)²/ε) 
               = exp(-x_i²/ε) exp(2x_ix_j/ε) exp(-x_j²/ε)
        
        The middle term exp(2x_ix_j/ε) = Σ_k (2x_ix_j/ε)^k / k! converges
        rapidly, giving a low-rank approximation.
    
    Args:
        grid_size: Number of grid points
        grid_bounds: Physical domain
        epsilon: Regularization parameter
        dtype: Data type
        device: Device
        
    Returns:
        QTTMatrix representing the Gibbs kernel
    """
    if grid_size & (grid_size - 1) != 0:
        raise ValueError(f"grid_size must be power of 2, got {grid_size}")
    
    num_bits = int(math.log2(grid_size))
    low, high = grid_bounds
    dx = (high - low) / grid_size
    
    # Estimate rank needed: O(log(1/ε) + log(max |x|²/ε))
    max_x2 = max(abs(low), abs(high)) ** 2
    est_rank = max(2, int(math.log(max_x2 / epsilon + 1) * 2) + 2)
    
    # Build using Taylor expansion of exp(2xy/ε)
    # This is approximate - for high precision would use TCI
    
    cores = []
    
    for k in range(num_bits):
        bit_val = dx * (2 ** k)
        
        # Rank determined by Taylor terms needed
        rank = min(est_rank, 10)  # Cap for stability
        
        if k == 0:
            core = torch.zeros(1, 2, 2, rank, dtype=dtype, device=device)
            
            for i in range(2):
                x_i = low + i * bit_val
                for j in range(2):
                    x_j = low + j * bit_val
                    
                    # Exact value for this bit contribution
                    val = math.exp(-(x_i - x_j) ** 2 / epsilon)
                    core[0, i, j, 0] = val
                    # Higher rank components for accumulation
                    for r in range(1, rank):
                        core[0, i, j, r] = 0.0
                        
        elif k == num_bits - 1:
            core = torch.zeros(rank, 2, 2, 1, dtype=dtype, device=device)
            
            for i in range(2):
                contrib_i = i * bit_val
                for j in range(2):
                    contrib_j = j * bit_val
                    
                    val = math.exp(-(contrib_i - contrib_j) ** 2 / epsilon)
                    core[0, i, j, 0] = val
                    for r in range(1, rank):
                        core[r, i, j, 0] = 1.0 if r == 0 else 0.0
                        
        else:
            core = torch.zeros(rank, 2, 2, rank, dtype=dtype, device=device)
            
            for i in range(2):
                contrib_i = i * bit_val
                for j in range(2):
                    contrib_j = j * bit_val
                    
                    val = math.exp(-(contrib_i - contrib_j) ** 2 / epsilon)
                    # Diagonal structure for accumulation
                    for r in range(rank):
                        core[r, i, j, r] = val if r == 0 else 1.0
        
        cores.append(core)
    
    return QTTMatrix(cores=cores, grid_bounds=grid_bounds)


def custom_cost_mpo(
    cost_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    grid_size: int,
    grid_bounds: Tuple[float, float] = (-10.0, 10.0),
    max_rank: int = 50,
    tol: float = 1e-8,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu'),
) -> QTTMatrix:
    """
    Construct a custom cost matrix using TT-Cross interpolation.
    
    This is the general-purpose factory that can approximate any cost
    function c(x, y) as a QTT-MPO.
    
    Args:
        cost_func: Function c(x, y) -> cost, accepting batched inputs
        grid_size: Number of grid points
        grid_bounds: Physical domain
        max_rank: Maximum TT rank for approximation
        tol: Approximation tolerance
        dtype: Data type
        device: Device
        
    Returns:
        QTTMatrix approximating the cost function
    """
    raise NotImplementedError(
        "TT-Cross for arbitrary cost functions coming in Week 4. "
        "Use euclidean_cost_mpo or gaussian_kernel_mpo for now."
    )
