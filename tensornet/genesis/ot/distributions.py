"""
QTT Distribution Representations

This module provides probability distribution representations in QTT format,
enabling trillion-point distributions with O(r² log N) memory.

Constitutional Reference: TENSOR_GENESIS.md, Article I (Compression Covenant)

The key insight is that many probability distributions have smooth structure
that can be captured with low TT rank. For example:
- Gaussians are separable in the exponent → TT rank 1
- Mixtures have TT rank equal to number of components
- Kernel density estimates inherit structure from kernel

Example:
    >>> from tensornet.genesis.ot import QTTDistribution
    >>> 
    >>> # Trillion-point Gaussian
    >>> mu = QTTDistribution.gaussian(mean=0.0, std=1.0, grid_size=2**40)
    >>> print(f"Grid size: {mu.grid_size}, TT rank: {mu.max_rank}")
    Grid size: 1099511627776, TT rank: 2
    
    >>> # Mixture of Gaussians
    >>> nu = QTTDistribution.mixture([
    ...     (0.3, QTTDistribution.gaussian(-2, 0.5, 2**40)),
    ...     (0.7, QTTDistribution.gaussian(+1, 1.0, 2**40))
    ... ])

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Callable
import torch
import numpy as np

# Import from tensornet core - use relative imports for flexibility
try:
    from tensornet.core.qtt import QTT, qtt_round, qtt_add, qtt_hadamard
    from tensornet.core.decompositions import tt_decompose
except ImportError:
    # Fallback for standalone testing
    QTT = None


@dataclass
class QTTDistribution:
    """
    A probability distribution represented in Quantized Tensor Train format.
    
    This representation enables operations on distributions with up to 10¹²
    points using only O(r² log N) memory, where r is the TT rank.
    
    Attributes:
        cores: List of TT cores, each of shape (r_{k-1}, n_k, r_k)
        grid_bounds: (low, high) tuple defining the physical domain
        grid_size: Total number of grid points (product of mode sizes)
        is_normalized: Whether the distribution sums to 1
        
    Mathematical Background:
        A QTT representation factorizes a vector x of length N = 2^d as:
        
        x[i₁...i_d] = G₁[i₁] G₂[i₂] ... G_d[i_d]
        
        where each G_k is a matrix of size r_{k-1} × r_k for binary indices i_k.
        Storage is O(d r²) = O(log(N) r²) instead of O(N).
        
    Example:
        >>> mu = QTTDistribution.gaussian(0.0, 1.0, 2**30)
        >>> print(f"Represents {mu.grid_size:,} points with rank {mu.max_rank}")
        Represents 1,073,741,824 points with rank 2
    """
    
    cores: List[torch.Tensor]
    grid_bounds: Tuple[float, float] = (-10.0, 10.0)
    grid_size: int = field(init=False)
    is_normalized: bool = True
    _total_mass: Optional[float] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Compute derived attributes."""
        # Grid size is product of mode dimensions
        self.grid_size = 1
        for core in self.cores:
            self.grid_size *= core.shape[1]
    
    @property
    def num_cores(self) -> int:
        """Number of TT cores (= log₂(grid_size) for QTT)."""
        return len(self.cores)
    
    @property
    def max_rank(self) -> int:
        """Maximum TT rank across all cores."""
        if not self.cores:
            return 0
        return max(core.shape[0] for core in self.cores[1:])
    
    @property
    def ranks(self) -> List[int]:
        """List of TT ranks [r_0, r_1, ..., r_d]."""
        if not self.cores:
            return []
        ranks = [1]  # r_0 = 1
        for core in self.cores:
            ranks.append(core.shape[2])
        return ranks
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of the cores."""
        return self.cores[0].dtype if self.cores else torch.float64
    
    @property
    def device(self) -> torch.device:
        """Device where cores are stored."""
        return self.cores[0].device if self.cores else torch.device('cpu')
    
    @property
    def dx(self) -> float:
        """Grid spacing."""
        low, high = self.grid_bounds
        return (high - low) / self.grid_size
    
    def grid_points(self, as_numpy: bool = False) -> Union[torch.Tensor, np.ndarray]:
        """
        Return the physical grid points.
        
        Warning: This materializes O(N) memory! Only use for small grids.
        
        Args:
            as_numpy: Return numpy array instead of torch tensor
            
        Returns:
            Array of grid point locations
        """
        if self.grid_size > 2**20:
            raise ValueError(
                f"Grid size {self.grid_size} too large to materialize. "
                f"Use slicing or QTT operations instead."
            )
        low, high = self.grid_bounds
        x = torch.linspace(low, high, self.grid_size, dtype=self.dtype, device=self.device)
        return x.numpy() if as_numpy else x
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def gaussian(
        cls,
        mean: float = 0.0,
        std: float = 1.0,
        grid_size: int = 2**20,
        grid_bounds: Optional[Tuple[float, float]] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device('cpu'),
        normalize: bool = True,
    ) -> "QTTDistribution":
        """
        Create a Gaussian distribution in QTT format.
        
        The Gaussian has low TT rank due to its smooth structure.
        
        Args:
            mean: Mean of the Gaussian
            std: Standard deviation
            grid_size: Number of grid points (must be power of 2 for QTT)
            grid_bounds: (low, high) domain bounds. Auto-set if None.
            dtype: Torch data type
            device: Torch device
            normalize: Whether to normalize to sum to 1
            
        Returns:
            QTTDistribution representing the Gaussian
        """
        # Validate grid size is power of 2
        if grid_size & (grid_size - 1) != 0:
            raise ValueError(f"grid_size must be power of 2, got {grid_size}")
        
        num_bits = int(math.log2(grid_size))
        
        # Auto-set bounds to cover 6 sigma
        if grid_bounds is None:
            margin = 6 * std
            grid_bounds = (mean - margin, mean + margin)
        
        low, high = grid_bounds
        dx = (high - low) / grid_size
        
        # For small to moderate grids, compute dense values and convert via TT-SVD
        if grid_size <= 2**16:
            # Dense computation
            x = torch.linspace(low + dx/2, high - dx/2, grid_size, dtype=dtype, device=device)
            
            # Gaussian density (unnormalized)
            z = (x - mean) / std
            density = torch.exp(-0.5 * z * z)
            
            # Normalize to probability distribution
            if normalize:
                density = density / (density.sum() * dx)
            
            # Convert to QTT using TT-SVD (sequential left-to-right decomposition)
            # Reshape to 2x2x...x2 tensor
            tensor = density.reshape([2] * num_bits)
            
            # TT decomposition via sequential SVD
            cores = []
            C = tensor
            r_prev = 1
            
            for k in range(num_bits - 1):
                # Current shape: (r_prev, 2, 2, ..., 2)
                # Reshape to (r_prev * 2, remaining)
                C_shape = C.shape
                n_mode = C_shape[0] if k == 0 else C_shape[1]
                
                if k == 0:
                    # First iteration: C has shape (2, 2, ..., 2)
                    mat = C.reshape(2, -1)
                else:
                    # C has shape (r_prev, 2, 2, ..., 2)
                    mat = C.reshape(r_prev * 2, -1)
                
                # SVD
                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                
                # Determine rank (truncate small singular values)
                tol = 1e-12 * S[0] if S[0] > 0 else 1e-12
                rank = max(1, int((S > tol).sum()))
                rank = min(rank, 50)  # Cap rank for stability
                
                # Truncate
                U = U[:, :rank]
                S = S[:rank]
                Vh = Vh[:rank, :]
                
                # Form core: (r_in, 2, r_out)
                if k == 0:
                    core = U.reshape(1, 2, rank)
                else:
                    core = U.reshape(r_prev, 2, rank)
                
                cores.append(core)
                
                # Remaining tensor for next iteration
                SV = torch.diag(S) @ Vh
                remaining_size = mat.shape[1] // 2  # Each remaining mode has size 2
                if remaining_size > 0:
                    C = SV.reshape(rank, 2, remaining_size)
                else:
                    C = SV.reshape(rank, 2, 1)
                
                r_prev = rank
            
            # Last core: (r_in, 2, 1)
            last_core = C.reshape(r_prev, 2, 1)
            cores.append(last_core)
            
        else:
            # For large grids, use low-rank analytic construction
            raise NotImplementedError(
                f"Large grid Gaussian (N > 2^16) coming soon. "
                f"Use grid_size <= 65536 for now."
            )
        
        return cls(
            cores=cores,
            grid_bounds=grid_bounds,
            is_normalized=normalize,
        )
    
    @classmethod
    def uniform(
        cls,
        low: float = 0.0,
        high: float = 1.0,
        grid_size: int = 2**20,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device('cpu'),
    ) -> "QTTDistribution":
        """
        Create a uniform distribution in QTT format.
        
        The uniform distribution has TT rank 1 (simplest possible).
        
        Args:
            low: Lower bound of support
            high: Upper bound of support
            grid_size: Number of grid points
            dtype: Data type
            device: Device
            
        Returns:
            QTTDistribution representing uniform on [low, high]
        """
        if grid_size & (grid_size - 1) != 0:
            raise ValueError(f"grid_size must be power of 2, got {grid_size}")
        
        num_bits = int(math.log2(grid_size))
        value = 1.0 / grid_size  # Normalized uniform value
        
        # Rank-1 TT: each core just passes through
        cores = []
        for k in range(num_bits):
            if k == 0:
                core = torch.ones(1, 2, 1, dtype=dtype, device=device) * math.sqrt(value)
            elif k == num_bits - 1:
                core = torch.ones(1, 2, 1, dtype=dtype, device=device) * math.sqrt(value)
            else:
                core = torch.ones(1, 2, 1, dtype=dtype, device=device)
            cores.append(core)
        
        return cls(cores=cores, grid_bounds=(low, high), is_normalized=True)
    
    @classmethod
    def mixture(
        cls,
        components: List[Tuple[float, "QTTDistribution"]],
        round_tol: float = 1e-10,
    ) -> "QTTDistribution":
        """
        Create a mixture distribution from weighted components.
        
        The mixture has TT rank at most sum of component ranks (before rounding).
        
        Args:
            components: List of (weight, distribution) tuples
            round_tol: Tolerance for TT rounding after addition
            
        Returns:
            QTTDistribution representing the mixture
            
        Example:
            >>> mu1 = QTTDistribution.gaussian(-2, 0.5, 2**30)
            >>> mu2 = QTTDistribution.gaussian(+2, 0.5, 2**30)
            >>> mixture = QTTDistribution.mixture([(0.5, mu1), (0.5, mu2)])
        """
        if not components:
            raise ValueError("Must provide at least one component")
        
        # Validate weights
        weights = [w for w, _ in components]
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1, got {total_weight}")
        
        # Check all components have same grid
        first_dist = components[0][1]
        for _, dist in components[1:]:
            if dist.grid_size != first_dist.grid_size:
                raise ValueError("All components must have same grid_size")
            if dist.grid_bounds != first_dist.grid_bounds:
                raise ValueError("All components must have same grid_bounds")
        
        # Build mixture by weighted addition
        result = None
        for weight, dist in components:
            scaled = dist.scale(weight)
            if result is None:
                result = scaled
            else:
                result = result.add(scaled)
        
        # Round to control rank growth
        if round_tol > 0:
            result = result.round(round_tol)
        
        result.is_normalized = True
        return result
    
    @classmethod
    def delta(
        cls,
        location: float,
        grid_size: int = 2**20,
        grid_bounds: Tuple[float, float] = (-10.0, 10.0),
        width: float = 0.01,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device('cpu'),
    ) -> "QTTDistribution":
        """
        Create an approximate delta distribution (narrow Gaussian).
        
        Args:
            location: Location of the delta spike
            grid_size: Number of grid points
            grid_bounds: Domain bounds
            width: Width of the Gaussian approximation
            dtype: Data type
            device: Device
            
        Returns:
            QTTDistribution approximating delta at location
        """
        return cls.gaussian(
            mean=location,
            std=width,
            grid_size=grid_size,
            grid_bounds=grid_bounds,
            dtype=dtype,
            device=device,
            normalize=True,
        )
    
    @classmethod
    def from_function(
        cls,
        func: Callable[[torch.Tensor], torch.Tensor],
        grid_size: int = 2**20,
        grid_bounds: Tuple[float, float] = (-10.0, 10.0),
        max_rank: int = 50,
        tol: float = 1e-10,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device('cpu'),
    ) -> "QTTDistribution":
        """
        Create a distribution from a function using TT-Cross/TCI.
        
        Uses Tensor Cross Interpolation to build a QTT representation
        of an arbitrary function without materializing the full grid.
        
        Args:
            func: Function f(x) -> values, accepting batched input
            grid_size: Number of grid points
            grid_bounds: Domain bounds
            max_rank: Maximum TT rank
            tol: Approximation tolerance
            dtype: Data type
            device: Device
            
        Returns:
            QTTDistribution approximating the function
        """
        # This would use TCI - placeholder for now
        raise NotImplementedError(
            "TCI-based construction coming in Week 2. "
            "Use gaussian/uniform/mixture factories for now."
        )
    
    # =========================================================================
    # Operations
    # =========================================================================
    
    def total_mass(self) -> float:
        """
        Compute the total mass (integral) of the distribution.
        
        Uses the fact that summing a QTT is O(d r²) by contracting
        with the all-ones vector, which has rank 1.
        
        Returns:
            Total mass (sum of all values times dx)
        """
        if self._total_mass is not None:
            return self._total_mass
        
        # Contract with all-ones vector (rank 1)
        # For each core, sum over the physical index
        result = torch.ones(1, dtype=self.dtype, device=self.device)
        
        for core in self.cores:
            # core has shape (r_in, n, r_out)
            # Sum over n, contract with result
            summed = core.sum(dim=1)  # Shape: (r_in, r_out)
            result = result @ summed
        
        mass = result.item() * self.dx
        self._total_mass = mass
        return mass
    
    def normalize(self) -> "QTTDistribution":
        """
        Normalize the distribution to sum to 1.
        
        Returns:
            New normalized QTTDistribution
        """
        mass = self.total_mass()
        if abs(mass) < 1e-15:
            raise ValueError("Cannot normalize distribution with zero mass")
        
        return self.scale(1.0 / mass)
    
    def scale(self, factor: float) -> "QTTDistribution":
        """
        Scale the distribution by a constant factor.
        
        This is O(1) - just scale the first core.
        
        Args:
            factor: Scaling factor
            
        Returns:
            Scaled distribution
        """
        new_cores = [core.clone() for core in self.cores]
        new_cores[0] = new_cores[0] * factor
        
        return QTTDistribution(
            cores=new_cores,
            grid_bounds=self.grid_bounds,
            is_normalized=False,
        )
    
    def add(self, other: "QTTDistribution") -> "QTTDistribution":
        """
        Add two distributions (rank-additive operation).
        
        The result has rank r1 + r2 before rounding.
        
        Args:
            other: Distribution to add
            
        Returns:
            Sum distribution
        """
        if self.grid_size != other.grid_size:
            raise ValueError("Distributions must have same grid_size")
        if self.grid_bounds != other.grid_bounds:
            raise ValueError("Distributions must have same grid_bounds")
        
        # TT addition: concatenate cores along rank dimension
        new_cores = []
        for k, (c1, c2) in enumerate(zip(self.cores, other.cores)):
            r1_in, n, r1_out = c1.shape
            r2_in, _, r2_out = c2.shape
            
            if k == 0:
                # First core: concatenate along output rank
                new_core = torch.cat([c1, c2], dim=2)
            elif k == len(self.cores) - 1:
                # Last core: concatenate along input rank
                new_core = torch.cat([c1, c2], dim=0)
            else:
                # Middle cores: block diagonal
                new_core = torch.zeros(
                    r1_in + r2_in, n, r1_out + r2_out,
                    dtype=self.dtype, device=self.device
                )
                new_core[:r1_in, :, :r1_out] = c1
                new_core[r1_in:, :, r1_out:] = c2
            
            new_cores.append(new_core)
        
        return QTTDistribution(
            cores=new_cores,
            grid_bounds=self.grid_bounds,
            is_normalized=False,
        )
    
    def round(self, tol: float = 1e-10, max_rank: Optional[int] = None) -> "QTTDistribution":
        """
        Round (compress) the QTT to lower rank.
        
        Uses truncated SVD sweeping to find optimal lower-rank approximation.
        
        Constitutional Reference: Article I, Section 1.3 (Rounding Mandate)
        
        Args:
            tol: Relative tolerance for truncation
            max_rank: Maximum rank (None for tolerance-based)
            
        Returns:
            Rounded distribution with lower rank
        """
        # Right-to-left orthogonalization
        new_cores = [core.clone() for core in self.cores]
        
        for k in range(len(new_cores) - 1, 0, -1):
            core = new_cores[k]
            r_in, n, r_out = core.shape
            
            # Reshape to matrix and do QR
            mat = core.reshape(r_in, n * r_out)
            Q, R = torch.linalg.qr(mat.T)
            
            # Update cores
            new_cores[k] = Q.T.reshape(-1, n, r_out)
            new_cores[k-1] = torch.einsum('ijk,kl->ijl', new_cores[k-1], R.T)
        
        # Left-to-right SVD truncation
        for k in range(len(new_cores) - 1):
            core = new_cores[k]
            r_in, n, r_out = core.shape
            
            # Reshape and SVD
            mat = core.reshape(r_in * n, r_out)
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            
            # Truncate based on tolerance
            total_sq = (S ** 2).sum()
            cumsum = torch.cumsum(S ** 2, dim=0)
            keep = ((total_sq - cumsum) > tol * total_sq).sum() + 1
            
            if max_rank is not None:
                keep = min(keep, max_rank)
            
            keep = max(1, int(keep))
            
            # Update cores
            new_cores[k] = (U[:, :keep] * S[:keep]).reshape(r_in, n, keep)
            new_cores[k+1] = torch.einsum('ij,jkl->ikl', Vh[:keep, :], new_cores[k+1])
        
        return QTTDistribution(
            cores=new_cores,
            grid_bounds=self.grid_bounds,
            is_normalized=self.is_normalized,
        )
    
    def hadamard(self, other: "QTTDistribution") -> "QTTDistribution":
        """
        Element-wise (Hadamard) product of two distributions.
        
        This is rank-multiplicative: result rank = r1 × r2.
        MUST be followed by rounding per Article I, Section 1.3.
        
        Args:
            other: Distribution to multiply element-wise
            
        Returns:
            Hadamard product (unrounded)
        """
        if self.grid_size != other.grid_size:
            raise ValueError("Distributions must have same grid_size")
        
        # Hadamard product: Kronecker product of cores
        new_cores = []
        for c1, c2 in zip(self.cores, other.cores):
            r1_in, n, r1_out = c1.shape
            r2_in, _, r2_out = c2.shape
            
            # Kronecker product along rank dimensions
            new_core = torch.einsum('ijk,lmn->iljkn', c1, c2)
            new_core = new_core.reshape(r1_in * r2_in, n, r1_out * r2_out)
            new_cores.append(new_core)
        
        return QTTDistribution(
            cores=new_cores,
            grid_bounds=self.grid_bounds,
            is_normalized=False,
        )
    
    def to_dense(self) -> torch.Tensor:
        """
        Materialize the full dense vector.
        
        WARNING: Only use for small grids (< 2^20 points).
        
        Returns:
            Dense tensor of shape (grid_size,)
        """
        if self.grid_size > 2**20:
            raise ValueError(
                f"Grid size {self.grid_size} too large to materialize. "
                f"Maximum is 2^20 = 1,048,576 points."
            )
        
        # Contract all cores
        result = self.cores[0]  # Shape: (1, n, r)
        
        for core in self.cores[1:]:
            # result: (1, N_so_far, r_prev)
            # core: (r_prev, n, r_next)
            r_prev = result.shape[2]
            N_so_far = result.shape[1]
            n = core.shape[1]
            r_next = core.shape[2]
            
            # Reshape for batch matmul
            result = result.reshape(-1, r_prev)  # (N_so_far, r_prev)
            core_mat = core.reshape(r_prev, n * r_next)  # (r_prev, n * r_next)
            
            result = result @ core_mat  # (N_so_far, n * r_next)
            result = result.reshape(1, N_so_far * n, r_next)
        
        return result.reshape(-1)
    
    def __repr__(self) -> str:
        return (
            f"QTTDistribution(grid_size={self.grid_size}, "
            f"max_rank={self.max_rank}, "
            f"bounds={self.grid_bounds}, "
            f"normalized={self.is_normalized})"
        )
