"""
TCI: Build QTT from Black-Box Function

The from_function API is THE interface for compressing arbitrary functions
to QTT format without ever materializing the full N-dimensional tensor.

Algorithm (TT-Cross Interpolation):
    1. Initialize pivot indices for left/right contexts
    2. For each mode k (left to right):
       a. Sample "fiber": (r_left × 2 × r_right) function evaluations
       b. SVD to get core and truncate rank
       c. MaxVol to select new pivot indices
    3. Return cores directly

Complexity: O(r² × n_qubits) function evaluations
Memory: O(r² × n_qubits) — NEVER O(2^n)

This is fundamental for curse-breaking because:
1. Users can define ANY function (physics simulation, ML model, etc.)
2. QTeneT compresses it with logarithmic cost
3. All subsequent operations stay in QTT format

Example:
    >>> from qtenet.tci import from_function
    >>> 
    >>> # Your expensive function
    >>> def plasma_distribution(idx):
    ...     # Some physics simulation
    ...     return compute_plasma_state(idx)
    >>> 
    >>> # Compress to QTT (30 qubits = 1 billion points)
    >>> cores = from_function(plasma_distribution, n_qubits=30, max_rank=64)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

# Import from upstream ontic
from ontic.cfd.qtt_tci import (
    qtt_from_function_tci_python as _tci_python,
    qtt_from_function_dense as _tci_dense,
    RUST_AVAILABLE as _RUST_TCI_AVAILABLE,
)

# Try Rust TCI if available
try:
    from ontic.cfd.qtt_tci import qtt_from_function_tci_rust as _tci_rust
except ImportError:
    _tci_rust = None


@dataclass
class TCIConfig:
    """Configuration for TCI construction.
    
    Attributes:
        max_rank: Maximum TT-rank (bond dimension)
        tolerance: Convergence tolerance for SVD truncation
        max_iterations: Maximum TCI sweeps (if iterative)
        batch_size: Batch size for function evaluations
        use_rust: Use Rust TCI core if available
        device: Torch device
        dtype: Tensor dtype
    """
    max_rank: int = 64
    tolerance: float = 1e-6
    max_iterations: int = 50
    batch_size: int = 10000
    use_rust: bool = True
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    @property
    def rust_available(self) -> bool:
        """Check if Rust TCI core is available."""
        return _RUST_TCI_AVAILABLE and _tci_rust is not None


@dataclass
class TCIResult:
    """Result from TCI construction.
    
    Attributes:
        cores: List of QTT cores
        n_qubits: Number of qubits
        max_rank_achieved: Actual maximum rank in result
        n_function_evals: Number of function evaluations
        compression_ratio: Theoretical compression vs dense
        method: Which TCI method was used
        metadata: Additional metadata from construction
    """
    cores: list[Tensor]
    n_qubits: int
    max_rank_achieved: int
    n_function_evals: int
    compression_ratio: float
    method: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def total_parameters(self) -> int:
        """Total parameters in QTT representation."""
        return sum(c.numel() for c in self.cores)
    
    @property
    def dense_size(self) -> int:
        """Size of equivalent dense representation."""
        return 2 ** self.n_qubits


def from_function(
    f: Callable[[Tensor], Tensor],
    n_qubits: int,
    max_rank: int = 64,
    tolerance: float = 1e-6,
    device: str = "cpu",
    use_rust: bool = True,
    verbose: bool = False,
) -> list[Tensor]:
    """
    Build QTT from black-box function using TT-Cross Interpolation.
    
    This is THE curse-breaking interface. It compresses any function
    to QTT format using only O(r² × log N) function evaluations.
    
    Args:
        f: Function f(indices) -> values where:
           - indices: Tensor of shape (batch,) with integer indices in [0, 2^n_qubits)
           - values: Tensor of shape (batch,) with function values
        n_qubits: Number of qubits (function domain is [0, 2^n_qubits))
        max_rank: Maximum TT-rank (controls accuracy vs compression)
        tolerance: SVD truncation tolerance
        device: Torch device for computation
        use_rust: Use Rust TCI core if available (faster)
        verbose: Print progress information
    
    Returns:
        List of QTT cores
    
    Examples:
        # Simple 1D function
        def sine(idx):
            return torch.sin(idx.float() / 1000)
        
        cores = from_function(sine, n_qubits=20, max_rank=16)
        
        # Expensive physics simulation
        def plasma(idx):
            return run_plasma_sim(idx)  # Your expensive code
        
        cores = from_function(plasma, n_qubits=30, max_rank=64)
    
    Complexity:
        - Function evaluations: O(r² × n_qubits) where r = max_rank
        - For n_qubits=30, max_rank=64: ~120K evaluations vs 1 billion
    """
    config = TCIConfig(
        max_rank=max_rank,
        tolerance=tolerance,
        use_rust=use_rust,
        device=device,
    )
    
    # Choose best available method
    if config.rust_available and use_rust and _tci_rust is not None:
        if verbose:
            print(f"[TCI] Using Rust TCI core (n_qubits={n_qubits}, max_rank={max_rank})")
        cores, metadata = _tci_rust(
            f=f,
            n_qubits=n_qubits,
            max_rank=max_rank,
            tolerance=tolerance,
            device=device,
            verbose=verbose,
        )
    elif n_qubits <= 12:
        # Small problem: use dense (acceptable overhead)
        if verbose:
            print(f"[TCI] Small problem (2^{n_qubits}={2**n_qubits}), using dense TT-SVD")
        cores = _tci_dense(f, n_qubits, max_rank, device)
    else:
        # Use smart TCI with seeded pivots for better accuracy
        if verbose:
            print(f"[TCI] Using Smart TCI (n_qubits={n_qubits}, max_rank={max_rank})")
        cores = _tci_smart(
            f=f,
            n_qubits=n_qubits,
            max_rank=max_rank,
            tolerance=tolerance,
            device=device,
            verbose=verbose,
        )
    
    return cores


def _tci_smart(
    f: Callable[[Tensor], Tensor],
    n_qubits: int,
    max_rank: int,
    tolerance: float,
    device: str,
    verbose: bool,
) -> list[Tensor]:
    """
    Smart TCI with value-guided pivot initialization.
    
    Instead of random pivots, we:
    1. Sample function at diverse points
    2. Find high-value regions  
    3. Seed pivots from those high-value indices
    4. Run TCI sweep
    
    This ensures TCI captures function peaks, not just zero regions.
    """
    import torch
    from ontic.cfd.qtt_tci import _maxvol_simple
    
    dev = torch.device(device)
    N = 2 ** n_qubits
    
    # Step 1: Sample function at diverse points to find important regions
    # Use Latin hypercube-style sampling for better coverage
    n_probe = min(2000, N // 4)
    
    # Create stratified samples - divide range into strata
    strata = torch.linspace(0, N - 1, n_probe, dtype=torch.long, device=dev)
    
    # Add some noise within strata for diversity (but keep valid indices)
    if n_probe > 10:
        noise_range = max(1, N // n_probe // 2)
        noise = torch.randint(-noise_range, noise_range + 1, (n_probe,), device=dev)
        probe_indices = (strata + noise).clamp(0, N - 1)
    else:
        probe_indices = strata
    
    probe_indices = probe_indices.unique()
    
    # Evaluate function at probe points
    probe_values = f(probe_indices)
    
    # Find indices with highest absolute values
    abs_vals = probe_values.abs()
    n_important = min(max_rank * 2, len(probe_indices))
    _, top_k_idx = torch.topk(abs_vals, n_important)
    important_indices = probe_indices[top_k_idx]
    
    if verbose:
        max_val = abs_vals.max().item()
        mean_val = abs_vals.mean().item()
        print(f"  Probed {len(probe_indices)} points: max|f|={max_val:.4f}, mean|f|={mean_val:.4f}")
    
    # Step 2: Build QTT using seeded pivots from important indices
    cores = []
    total_evals = len(probe_indices)
    
    # Initialize
    accumulated_left = torch.zeros(1, dtype=torch.long, device=dev)
    
    # Build right pivots seeded from important indices
    right_pivots = []
    for k in range(n_qubits):
        n_right = n_qubits - k - 1
        if n_right > 0:
            max_right = 2 ** n_right
            
            # Extract right-context bits from important indices
            right_bits = (important_indices >> (k + 1)) % max_right
            unique_rights = right_bits.unique()
            
            # Ensure we have enough pivots - mix in stratified samples
            if len(unique_rights) < max_rank:
                n_extra = max_rank - len(unique_rights)
                # Stratified random for right context
                extra = torch.linspace(0, max_right - 1, n_extra, dtype=torch.long, device=dev)
                all_pivots = torch.cat([unique_rights, extra]).unique()
            else:
                all_pivots = unique_rights
            
            right_pivots.append(all_pivots[:max_rank])
        else:
            right_pivots.append(torch.zeros(1, dtype=torch.long, device=dev))
    
    # Left-to-right sweep
    for k in range(n_qubits):
        r_left = len(accumulated_left)
        right_indices = right_pivots[k][:max_rank]
        r_right = len(right_indices)
        
        # Build sample indices
        left_expanded = accumulated_left.view(-1, 1, 1).expand(r_left, 2, r_right)
        bits = torch.arange(2, device=dev).view(1, -1, 1).expand(r_left, 2, r_right)
        right_expanded = right_indices.view(1, 1, -1).expand(r_left, 2, r_right)
        
        sample_indices = left_expanded + (bits << k) + (right_expanded << (k + 1))
        sample_indices = sample_indices.reshape(-1)
        
        # Evaluate
        values = f(sample_indices)
        total_evals += len(sample_indices)
        
        fiber = values.reshape(r_left, 2, r_right)
        
        if k < n_qubits - 1:
            # SVD
            fiber_mat = fiber.reshape(r_left * 2, r_right)
            U, S, Vh = torch.linalg.svd(fiber_mat, full_matrices=False)
            
            # Truncate
            rank = min(max_rank, len(S))
            if tolerance > 0 and S[0] > 1e-15:
                rel_cutoff = tolerance * S[0]
                rank = min(rank, max(1, (S > rel_cutoff).sum().item()))
            
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            
            core = U.reshape(r_left, 2, rank).to(values.dtype)
            cores.append(core)
            
            # MaxVol for next pivots
            if U.shape[0] > rank:
                pivot_rows = _maxvol_simple(U, tol=1.05, max_iters=50)
                left_indices = pivot_rows // 2
                bit_vals = pivot_rows % 2
                accumulated_left = accumulated_left[left_indices] + (bit_vals << k)
            else:
                new_accumulated = accumulated_left.view(-1, 1) + (torch.arange(2, device=dev) << k).view(1, -1)
                accumulated_left = new_accumulated.reshape(-1)
        else:
            core = fiber.reshape(r_left, 2, 1).to(values.dtype)
            cores.append(core)
    
    if verbose:
        params = sum(c.numel() for c in cores)
        max_r = max(c.shape[-1] for c in cores)
        compression = N / params if params > 0 else 0
        print(f"  Built: {len(cores)} cores, max_rank={max_r}, params={params:,}, compression={compression:.0f}×")
    
    return cores


def from_function_2d(
    f: Callable[[Tensor, Tensor], Tensor],
    n_qubits_x: int,
    n_qubits_y: int,
    max_rank: int = 64,
    tolerance: float = 1e-6,
    device: str = "cpu",
    verbose: bool = False,
) -> list[Tensor]:
    """
    Build QTT from 2D function using Morton-order TCI.
    
    The 2D function is flattened to 1D using Morton (Z-curve) ordering,
    which preserves locality and enables efficient QTT compression.
    
    Args:
        f: Function f(x_indices, y_indices) -> values
        n_qubits_x: Qubits for x dimension
        n_qubits_y: Qubits for y dimension
        max_rank: Maximum TT-rank
        tolerance: SVD truncation tolerance
        device: Torch device
        verbose: Print progress
    
    Returns:
        List of QTT cores with interleaved (Morton) ordering
    
    Example:
        def temperature_field(x, y):
            return torch.exp(-(x**2 + y**2) / 100)
        
        cores = from_function_2d(temperature_field, n_qubits_x=10, n_qubits_y=10)
    """
    from ontic.cfd.qtt_2d import morton_encode_batch
    
    total_qubits = n_qubits_x + n_qubits_y
    
    def morton_wrapped(indices: Tensor) -> Tensor:
        # Decode Morton to (x, y)
        # Morton interleaves: x bits at even positions, y at odd
        n_x = 2 ** n_qubits_x
        n_y = 2 ** n_qubits_y
        
        x = torch.zeros_like(indices)
        y = torch.zeros_like(indices)
        
        for b in range(max(n_qubits_x, n_qubits_y)):
            if b < n_qubits_x:
                x |= ((indices >> (2 * b)) & 1) << b
            if b < n_qubits_y:
                y |= ((indices >> (2 * b + 1)) & 1) << b
        
        return f(x, y)
    
    return from_function(
        morton_wrapped,
        n_qubits=total_qubits,
        max_rank=max_rank,
        tolerance=tolerance,
        device=device,
        verbose=verbose,
    )


def from_function_nd(
    f: Callable[[list[Tensor]], Tensor],
    qubits_per_dim: list[int],
    max_rank: int = 64,
    tolerance: float = 1e-6,
    device: str = "cpu",
    verbose: bool = False,
) -> list[Tensor]:
    """
    Build QTT from N-dimensional function using Morton-order TCI.
    
    Generalizes to arbitrary dimensions with Morton interleaving.
    
    Args:
        f: Function f([x0, x1, ..., xn]) -> values
        qubits_per_dim: List of qubits per dimension
        max_rank: Maximum TT-rank
        tolerance: SVD truncation tolerance
        device: Torch device
        verbose: Print progress
    
    Returns:
        List of QTT cores with Morton ordering
    
    Example:
        # 5D phase space function
        def distribution(coords):
            x, y, z, vx, vy = coords
            return torch.exp(-(x**2 + y**2 + z**2 + vx**2 + vy**2))
        
        cores = from_function_nd(distribution, qubits_per_dim=[5, 5, 5, 5, 5])
    """
    num_dims = len(qubits_per_dim)
    total_qubits = sum(qubits_per_dim)
    max_bits = max(qubits_per_dim)
    
    def morton_decode_nd(indices: Tensor) -> list[Tensor]:
        """Decode Morton index to N-dimensional coordinates."""
        coords = [torch.zeros_like(indices) for _ in range(num_dims)]
        
        for b in range(max_bits):
            for dim in range(num_dims):
                if b < qubits_per_dim[dim]:
                    bit_pos = num_dims * b + dim
                    coords[dim] |= ((indices >> bit_pos) & 1) << b
        
        return coords
    
    def morton_wrapped(indices: Tensor) -> Tensor:
        coords = morton_decode_nd(indices)
        return f(coords)
    
    return from_function(
        morton_wrapped,
        n_qubits=total_qubits,
        max_rank=max_rank,
        tolerance=tolerance,
        device=device,
        verbose=verbose,
    )


__all__ = [
    "from_function",
    "from_function_2d",
    "from_function_nd",
    "TCIConfig",
    "TCIResult",
]
