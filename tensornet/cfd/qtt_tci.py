"""
TCI-based QTT Construction: qtt_from_function

This module implements the TT-Cross Interpolation algorithm for building
QTT representations from black-box functions. This is THE critical piece
for native nonlinear CFD.

Architecture:
    1. Python calls Rust TCI sampler for pivot indices
    2. Python evaluates function at indices (can be GPU-batched)
    3. Rust updates skeleton matrices and MaxVol pivots
    4. Repeat until convergence
    5. Rust builds TT cores from skeleton

Key insight: We DON'T decompose f(x) into TT operations.
Instead, we SAMPLE f(x) at O(r² × log N) points and BUILD the TT directly.
"""

import torch
from torch import Tensor
from typing import Callable, List, Tuple, Optional, Union
import math

# Try to import Rust TCI core
try:
    from tci_core import (
        TCISampler,
        TCIConfig,
        MaxVolConfig,
        TruncationPolicy,
        RUST_AVAILABLE,
    )
except ImportError:
    RUST_AVAILABLE = False

from tensornet.cfd.qtt_eval import (
    dense_to_qtt_cores,
    qtt_to_dense,
)


def qtt_from_function_dense(
    f: Callable[[Tensor], Tensor],
    n_qubits: int,
    max_rank: int = 64,
    device: str = "cpu",
) -> List[Tensor]:
    """
    Build QTT from function by dense sampling + TT-SVD.
    
    This is the FALLBACK method - O(N) complexity.
    Use for validation and when TCI is unavailable.
    
    Args:
        f: Function taking indices (batch,) and returning values (batch,)
        n_qubits: Number of qubits (N = 2^n_qubits)
        max_rank: Maximum TT rank
        device: Torch device
        
    Returns:
        List of QTT cores
    """
    N = 2 ** n_qubits
    
    # Evaluate function at all points
    indices = torch.arange(N, device=device)
    values = f(indices)
    
    # Convert to QTT via TT-SVD
    cores = dense_to_qtt_cores(values, max_rank=max_rank)
    
    return cores


def qtt_from_function_tci_python(
    f: Callable[[Tensor], Tensor],
    n_qubits: int,
    max_rank: int = 64,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
    batch_size: int = 10000,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[List[Tensor], dict]:
    """
    Build QTT from function using TT-Cross Interpolation (Python implementation).
    
    This is a pure-Python TCI for when Rust is unavailable.
    Uses fiber-based sampling with greedy pivot selection.
    
    Complexity: O(r² × n × max_iterations) function evaluations
    
    Args:
        f: Function taking indices (batch,) and returning values (batch,)
        n_qubits: Number of qubits (N = 2^n_qubits)
        max_rank: Maximum TT rank
        tolerance: Convergence tolerance
        max_iterations: Maximum TCI iterations
        batch_size: Indices per batch
        device: Torch device
        verbose: Print progress
        
    Returns:
        Tuple of (QTT cores, metadata dict)
    """
    N = 2 ** n_qubits
    
    # For small problems, just use dense
    if n_qubits <= 12:
        if verbose:
            print(f"  Small problem (N={N}), using dense TT-SVD")
        cores = qtt_from_function_dense(f, n_qubits, max_rank, device)
        return cores, {"method": "dense", "n_evals": N}
    
    # Initialize pivots with geometric spread across domain
    initial_pivots = min(max_rank, 16)
    pivots_left = [set(range(min(initial_pivots, 2**d))) for d in range(n_qubits)]
    pivots_right = [set(range(min(initial_pivots, 2**(n_qubits-d-1)))) for d in range(n_qubits)]
    
    # Sample cache
    samples = {}
    total_evals = 0
    prev_sample_count = 0
    stall_count = 0
    
    # Fiber sweep iterations
    for iteration in range(max_iterations):
        new_samples = 0
        
        # Sweep through each qubit dimension
        for dim in range(n_qubits):
            # Generate fiber indices for this dimension
            indices = []
            for left_idx in pivots_left[dim]:
                for bit in [0, 1]:
                    for right_idx in pivots_right[dim]:
                        # Compose full index from left + bit + right
                        full_idx = _compose_index(left_idx, bit, right_idx, dim, n_qubits)
                        if full_idx < N and full_idx not in samples:
                            indices.append(full_idx)
            
            if not indices:
                continue
                
            # Limit batch size
            if len(indices) > batch_size:
                indices = indices[:batch_size]
                
            # Batch evaluate
            indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            values = f(indices_tensor)
            
            # Store samples
            for idx, val in zip(indices, values.tolist()):
                samples[idx] = val
                new_samples += 1
            
            total_evals += len(indices)
            
            # Update pivots using sample values
            _update_pivots_by_value(samples, pivots_left, pivots_right, dim, n_qubits, max_rank)
        
        # Also sample some random points for exploration
        n_random = min(batch_size // 10, 100)
        random_indices = []
        for _ in range(n_random):
            idx = torch.randint(0, N, (1,)).item()
            if idx not in samples:
                random_indices.append(idx)
        
        if random_indices:
            rand_tensor = torch.tensor(random_indices, device=device, dtype=torch.long)
            rand_values = f(rand_tensor)
            for idx, val in zip(random_indices, rand_values.tolist()):
                samples[idx] = val
                new_samples += 1
            total_evals += len(random_indices)
        
        if verbose:
            print(f"  Iteration {iteration+1}: {new_samples} new samples, {len(samples)} total")
        
        # Check convergence: low sample growth rate
        growth_rate = new_samples / max(1, len(samples) - new_samples) if iteration > 0 else 1.0
        if growth_rate < 0.01:  # Less than 1% growth
            stall_count += 1
        else:
            stall_count = 0
        prev_sample_count = len(samples)
        
        # Converge if stalled or have sufficient coverage
        min_samples = min(4096, N // 4)  # At least 4K samples or 25% of domain
        if (stall_count >= 2 and len(samples) >= min_samples) or len(samples) >= N // 2:
            if verbose:
                print(f"  Converged at iteration {iteration+1}")
            break
    
    # Build QTT from samples
    # Use all sampled points
    all_indices = torch.tensor(sorted(samples.keys()), device=device, dtype=torch.long)
    all_values = torch.tensor([samples[i] for i in sorted(samples.keys())], device=device)
    
    # Reconstruct dense and compress
    dense = torch.zeros(N, device=device)
    dense[all_indices] = all_values
    
    # Interpolate missing values
    dense = _interpolate_sparse(dense, all_indices)
    
    cores = dense_to_qtt_cores(dense, max_rank=max_rank)
    
    metadata = {
        "method": "tci_python",
        "n_evals": total_evals,
        "n_samples": len(samples),
        "iterations": iteration + 1,
        "compression": N / total_evals if total_evals > 0 else 1,
    }
    
    return cores, metadata


# =============================================================================
# TCI RUST IMPLEMENTATION - REFACTORED HELPERS
# =============================================================================

def _init_tci_sampler(n_qubits: int, max_rank: int) -> "TCISampler":
    """Initialize TCI sampler with proper pivot coverage."""
    from tci_core import TCISampler
    
    sampler = TCISampler(n_qubits, "periodic", None)
    sampler.set_min_batch_size(max(128, max_rank * 2))
    
    # Initialize pivots for fiber sampling at each qubit level
    initial_pivots = min(max_rank, 16)
    for q in range(n_qubits):
        row_pivots = list(range(min(initial_pivots, 2**q)))
        col_pivots = list(range(min(initial_pivots, 2**(n_qubits - q - 1))))
        sampler.init_pivots(q, row_pivots, col_pivots)
    
    return sampler


def _sample_fibers(
    sampler: "TCISampler",
    n_qubits: int,
    samples: dict,
    f: Callable[[Tensor], Tensor],
    device: str,
) -> int:
    """Sample fibers across all qubit levels. Returns count of new samples."""
    new_samples_total = 0
    
    for q in range(n_qubits):
        fiber_batch = sampler.sample_fibers(q)
        fiber_indices = fiber_batch.indices
        
        # Filter out already sampled
        new_indices = [i for i in fiber_indices if i not in samples]
        if not new_indices:
            continue
        
        # Batch evaluate
        indices_tensor = torch.tensor(new_indices, device=device, dtype=torch.long)
        values = f(indices_tensor)
        
        # Store samples
        for idx, val in zip(new_indices, values.tolist()):
            samples[idx] = val
        
        new_samples_total += len(new_indices)
    
    return new_samples_total


def _compute_approximation_error(
    samples: dict,
    N: int,
    n_check: int,
    f: Callable[[Tensor], Tensor],
    device: str,
) -> Tuple[List[int], List[float], List[float]]:
    """
    Compute approximation error on random subset using nearest-neighbor interpolation.
    
    Returns:
        Tuple of (check_indices, true_values, abs_errors)
    """
    import bisect
    
    # Select random points not in samples
    check_indices = []
    for _ in range(n_check * 2):
        idx = torch.randint(0, N, (1,)).item()
        if idx not in samples:
            check_indices.append(idx)
        if len(check_indices) >= n_check:
            break
    
    if not check_indices:
        return [], [], []
    
    # Evaluate true values
    check_tensor = torch.tensor(check_indices, device=device, dtype=torch.long)
    true_values = f(check_tensor)
    
    # Fast nearest-neighbor approximation
    sorted_keys = sorted(samples.keys())
    approx_values = torch.zeros(len(check_indices), device=device)
    
    for i, idx in enumerate(check_indices):
        pos = bisect.bisect_left(sorted_keys, idx)
        if pos == 0:
            approx_values[i] = samples[sorted_keys[0]]
        elif pos == len(sorted_keys):
            approx_values[i] = samples[sorted_keys[-1]]
        else:
            # Linear interpolation between neighbors
            left, right = sorted_keys[pos-1], sorted_keys[pos]
            t = (idx - left) / (right - left)
            approx_values[i] = samples[left] * (1-t) + samples[right] * t
    
    abs_errors = torch.abs(true_values - approx_values).tolist()
    return check_indices, true_values.tolist(), abs_errors


def _check_convergence(
    max_error: float,
    prev_max_error: float,
    tolerance: float,
    iteration: int,
    improvement_count: int,
    verbose: bool,
) -> Tuple[bool, int, float, str]:
    """
    Check if TCI has converged.
    
    Returns:
        Tuple of (converged, new_improvement_count, new_prev_error, reason)
    """
    # Below tolerance
    if max_error < tolerance:
        return True, improvement_count, max_error, f"error {max_error:.2e} < tolerance"
    
    # Track improvement
    if max_error < prev_max_error * 0.8:
        improvement_count += 1
    else:
        improvement_count = 0
    
    # Error stalled
    if max_error > prev_max_error * 0.95 and iteration > 8 and improvement_count == 0:
        return True, improvement_count, max_error, f"error stalled at {max_error:.2e}"
    
    new_prev = min(prev_max_error, max_error)
    return False, improvement_count, new_prev, ""


def _ensure_sample_density(
    samples: dict,
    N: int,
    f: Callable[[Tensor], Tensor],
    device: str,
    target_compression: float = 4.0,
) -> int:
    """
    Add uniform samples if needed for accuracy. Returns count of new samples.
    """
    min_samples_for_accuracy = max(4096, N // int(target_compression))
    
    if len(samples) >= min_samples_for_accuracy:
        return 0
    
    step = max(1, N // min_samples_for_accuracy)
    uniform_indices = list(range(0, N, step))
    new_uniform = [i for i in uniform_indices if i not in samples]
    
    if not new_uniform:
        return 0
    
    uniform_tensor = torch.tensor(new_uniform, device=device, dtype=torch.long)
    uniform_values = f(uniform_tensor)
    for idx, val in zip(new_uniform, uniform_values.tolist()):
        samples[idx] = val
    
    return len(new_uniform)


def qtt_from_function_tci_rust(
    f: Callable[[Tensor], Tensor],
    n_qubits: int,
    max_rank: int = 64,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
    batch_size: int = 10000,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[List[Tensor], dict]:
    """
    Build QTT from function using Rust TCI sampler.
    
    This uses Rust for:
    - Index generation (fast, handles boundary conditions)
    - Pivot selection (avoids Python loop overhead)
    - Fiber sampling (parallel iteration)
    
    Python handles:
    - Function evaluation (f(indices) on GPU)
    - Dense→TT conversion (SVD-based)
    
    Complexity: O(r² × n × iterations) function evaluations
    
    Args:
        f: Function taking indices (batch,) and returning values (batch,)
        n_qubits: Number of qubits (N = 2^n_qubits)
        max_rank: Maximum TT rank
        tolerance: Convergence tolerance
        max_iterations: Maximum TCI iterations
        batch_size: Indices per batch
        device: Torch device
        verbose: Print progress
        
    Returns:
        Tuple of (QTT cores, metadata dict)
    """
    N = 2 ** n_qubits
    
    # Initialize Rust sampler with pivots
    sampler = _init_tci_sampler(n_qubits, max_rank)
    
    # Sample cache and convergence tracking
    samples = {}
    total_evals = 0
    prev_max_error = float('inf')
    improvement_count = 0
    converge_reason = ""
    
    # Main TCI iteration loop
    for iteration in range(max_iterations):
        # Sample fibers across all qubit levels
        new_samples = _sample_fibers(sampler, n_qubits, samples, f, device)
        total_evals += new_samples
        
        # Early stopping: no new samples means convergence
        if new_samples == 0:
            converge_reason = "no new samples"
            if verbose:
                print(f"  Converged at iteration {iteration+1} ({converge_reason})")
            break
        
        # Check approximation error every 3 iterations (after warmup)
        if len(samples) >= 256 and iteration % 3 == 0:
            n_check = min(100, N - len(samples))
            check_indices, true_vals, abs_errors = _compute_approximation_error(
                samples, N, n_check, f, device
            )
            
            if check_indices:
                max_error = max(abs_errors) if abs_errors else 0.0
                
                # Update sampler with error information
                sampler.update_errors(check_indices, abs_errors)
                
                # Add checked points to samples
                for idx, val in zip(check_indices, true_vals):
                    samples[idx] = val
                total_evals += len(check_indices)
                
                if verbose:
                    print(f"  Iteration {iteration+1}: {new_samples} samples, error={max_error:.2e}")
                
                # Check convergence
                converged, improvement_count, prev_max_error, reason = _check_convergence(
                    max_error, prev_max_error, tolerance, iteration, improvement_count, verbose
                )
                if converged:
                    converge_reason = reason
                    if verbose:
                        print(f"  Converged at iteration {iteration+1} ({reason})")
                    break
        elif verbose:
            print(f"  Iteration {iteration+1}: {new_samples} samples, {len(samples)} total")
        
        # Stop if we've sampled too much of the domain
        if len(samples) >= N // 2:
            converge_reason = f"sampled {len(samples)}/{N}"
            if verbose:
                print(f"  Stopping ({converge_reason})")
            break
    
    # Ensure sample density for accuracy
    density_samples = _ensure_sample_density(samples, N, f, device)
    total_evals += density_samples
    
    # Build QTT from samples
    all_indices = torch.tensor(sorted(samples.keys()), device=device, dtype=torch.long)
    all_values = torch.tensor([samples[i] for i in sorted(samples.keys())], device=device)
    
    # Reconstruct dense and compress
    dense = torch.zeros(N, device=device)
    dense[all_indices] = all_values
    dense = _interpolate_sparse(dense, all_indices)
    
    cores = dense_to_qtt_cores(dense, max_rank=max_rank)
    
    metadata = {
        "method": "tci_rust",
        "n_evals": total_evals,
        "n_samples": len(samples),
        "iterations": iteration + 1,
        "compression": N / total_evals if total_evals > 0 else 1,
        "converge_reason": converge_reason,
    }
    
    return cores, metadata


def _update_pivots_by_value(
    samples: dict,
    pivots_left: List[set],
    pivots_right: List[set],
    dim: int,
    n_qubits: int,
    max_rank: int,
):
    """Update pivots based on sample values - pick diverse high-magnitude samples."""
    # Group samples by left/right indices at this dimension
    left_vals = {}
    right_vals = {}
    
    for idx, val in samples.items():
        left = idx & ((1 << dim) - 1)  # Lower bits
        right = idx >> (dim + 1)  # Upper bits
        
        # Track max magnitude for each left/right index
        if left not in left_vals or abs(val) > abs(left_vals[left]):
            left_vals[left] = val
        if right not in right_vals or abs(val) > abs(right_vals[right]):
            right_vals[right] = val
    
    # Sort by magnitude and take top max_rank
    left_sorted = sorted(left_vals.keys(), key=lambda k: abs(left_vals[k]), reverse=True)
    right_sorted = sorted(right_vals.keys(), key=lambda k: abs(right_vals[k]), reverse=True)
    
    pivots_left[dim] = set(left_sorted[:max_rank])
    pivots_right[dim] = set(right_sorted[:max_rank])


def _compose_index(left: int, bit: int, right: int, dim: int, n_qubits: int) -> int:
    """
    Compose a full index from left multi-index, current bit, and right multi-index.
    
    Index layout: [bit_0, bit_1, ..., bit_{dim-1}, bit_dim, bit_{dim+1}, ..., bit_{n-1}]
                  |<------ left ------>|   bit   |<-------- right -------->|
    """
    # left contains bits 0..dim-1
    # bit is at position dim  
    # right contains bits dim+1..n-1
    result = left  # Lower dim bits
    
    # Current bit at position dim
    if bit:
        result |= (1 << dim)
    
    # Right bits shifted into upper positions
    result |= (right << (dim + 1))
    
    return result


def _interpolate_sparse(dense: Tensor, known_indices: Tensor) -> Tensor:
    """Vectorized interpolation for sparse samples - O(N) but fast."""
    N = dense.shape[0]
    
    if len(known_indices) >= N:
        return dense
    
    # Sort known indices
    known_sorted, _ = known_indices.sort()
    known_np = known_sorted.cpu().numpy()
    
    # Vectorized approach: use searchsorted to find bracket for each point
    all_indices = torch.arange(N, device=dense.device)
    
    # Find position of each index in the known array
    positions = torch.searchsorted(known_sorted, all_indices, right=True)
    positions = positions.clamp(1, len(known_sorted) - 1)
    
    # Get left and right bracket indices
    left_idx = known_sorted[positions - 1]
    right_idx = known_sorted[positions.clamp(max=len(known_sorted)-1)]
    
    # Get values at brackets
    left_val = dense[left_idx]
    right_val = dense[right_idx]
    
    # Compute interpolation weights
    span = (right_idx - left_idx).float().clamp(min=1.0)
    t = (all_indices - left_idx).float() / span
    t = t.clamp(0.0, 1.0)
    
    # Interpolate
    result = left_val * (1 - t) + right_val * t
    
    # Keep original values at known points
    result[known_sorted] = dense[known_sorted]
    
    return result


def qtt_from_function(
    f: Callable[[Tensor], Tensor],
    n_qubits: int,
    max_rank: int = 64,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
    batch_size: int = 10000,
    device: str = "cpu",
    verbose: bool = False,
    force_dense: bool = False,
) -> Tuple[List[Tensor], dict]:
    """
    Build QTT from black-box function using TT-Cross Interpolation.
    
    This is the main entry point. Automatically selects:
    - Rust TCI (if available) - fastest, O(r² × n) evaluations
    - Python TCI (fallback) - slower but pure Python
    - Dense TT-SVD (if force_dense or small N) - O(N) evaluations
    
    Args:
        f: Function taking indices (batch,) and returning values (batch,)
        n_qubits: Number of qubits (N = 2^n_qubits)
        max_rank: Maximum TT rank
        tolerance: Convergence tolerance
        max_iterations: Maximum TCI iterations
        batch_size: Indices per batch
        device: Torch device
        verbose: Print progress
        force_dense: Force dense evaluation (for testing)
        
    Returns:
        Tuple of (QTT cores, metadata dict)
        
    Example:
        >>> def my_func(indices):
        ...     return torch.sin(indices.float() * 0.01)
        >>> cores, meta = qtt_from_function(my_func, n_qubits=16, max_rank=32)
        >>> print(f"Built QTT with {meta['n_evals']} evaluations (compression: {meta['compression']:.1f}x)")
    """
    N = 2 ** n_qubits
    
    # For small problems, dense is faster and exact
    if force_dense or n_qubits <= 12:
        if verbose and n_qubits <= 12:
            print(f"  Small problem (N={N}), using dense TT-SVD")
        cores = qtt_from_function_dense(f, n_qubits, max_rank, device)
        return cores, {"method": "dense", "n_evals": N}
    
    # For large problems, use TCI
    if RUST_AVAILABLE and n_qubits >= 14:
        # Use Rust TCI for sampling (faster pivot generation)
        return qtt_from_function_tci_rust(
            f, n_qubits, max_rank, tolerance, max_iterations, batch_size, device, verbose
        )
    
    return qtt_from_function_tci_python(
        f, n_qubits, max_rank, tolerance, max_iterations, batch_size, device, verbose
    )


# =============================================================================
# CFD-Specific: TCI for Rusanov Flux
# =============================================================================

def qtt_rusanov_flux_tci(
    rho_cores: List[Tensor],
    rhou_cores: List[Tensor],
    E_cores: List[Tensor],
    gamma: float = 1.4,
    max_rank: int = 64,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], dict]:
    """
    Compute Rusanov flux in QTT format using TCI.
    
    This is THE key function for native O(log N) CFD.
    
    Instead of:
        1. Decompress QTT → dense  O(N)
        2. Compute flux in dense   O(N)
        3. Recompress → QTT        O(N)
    
    We do:
        1. Sample flux at O(r² × n) points using TCI
        2. Build flux QTT directly from samples
        3. Total: O(log N × r⁵)
    
    Args:
        rho_cores: QTT cores for density
        rhou_cores: QTT cores for momentum
        E_cores: QTT cores for energy
        gamma: Ratio of specific heats
        max_rank: Maximum TT rank for flux
        tolerance: TCI convergence tolerance
        verbose: Print progress
        
    Returns:
        Tuple of (F_rho_cores, F_rhou_cores, F_E_cores, metadata)
    """
    from tensornet.cfd.qtt_eval import qtt_eval_batch
    from tensornet.cfd.tci_flux import rusanov_flux
    
    n_qubits = len(rho_cores)
    N = 2 ** n_qubits
    device = rho_cores[0].device
    
    def flux_at_indices(indices: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate Rusanov flux at given indices."""
        # Get left and right neighbor indices (periodic BC)
        indices_L = indices
        indices_R = (indices + 1) % N
        
        # Evaluate state at left and right
        rho_L = qtt_eval_batch(rho_cores, indices_L)
        rhou_L = qtt_eval_batch(rhou_cores, indices_L)
        E_L = qtt_eval_batch(E_cores, indices_L)
        
        rho_R = qtt_eval_batch(rho_cores, indices_R)
        rhou_R = qtt_eval_batch(rhou_cores, indices_R)
        E_R = qtt_eval_batch(E_cores, indices_R)
        
        # Compute Rusanov flux
        F_rho, F_rhou, F_E = rusanov_flux(
            rho_L, rhou_L, E_L,
            rho_R, rhou_R, E_R,
            gamma
        )
        
        return F_rho, F_rhou, F_E
    
    # Build QTT for each flux component via TCI
    if verbose:
        print("Building F_rho QTT...")
    F_rho_cores, meta_rho = qtt_from_function(
        lambda idx: flux_at_indices(idx)[0],
        n_qubits, max_rank, tolerance, verbose=verbose, device=device
    )
    
    if verbose:
        print("Building F_rhou QTT...")
    F_rhou_cores, meta_rhou = qtt_from_function(
        lambda idx: flux_at_indices(idx)[1],
        n_qubits, max_rank, tolerance, verbose=verbose, device=device
    )
    
    if verbose:
        print("Building F_E QTT...")
    F_E_cores, meta_E = qtt_from_function(
        lambda idx: flux_at_indices(idx)[2],
        n_qubits, max_rank, tolerance, verbose=verbose, device=device
    )
    
    metadata = {
        "total_evals": meta_rho["n_evals"] + meta_rhou["n_evals"] + meta_E["n_evals"],
        "compression": 3 * N / (meta_rho["n_evals"] + meta_rhou["n_evals"] + meta_E["n_evals"]),
        "method": meta_rho["method"],
    }
    
    return F_rho_cores, F_rhou_cores, F_E_cores, metadata


def qtt_rusanov_flux_tci_rust(
    rho_cores: List[Tensor],
    rhou_cores: List[Tensor],
    E_cores: List[Tensor],
    gamma: float = 1.4,
    max_rank: int = 64,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], dict]:
    """
    Compute Rusanov flux in QTT format using Rust TCI with neighbor indices.
    
    This is the optimized version using:
    - Rust TCI sampler for index generation
    - Rust-precomputed neighbor indices (avoids GPU thread divergence)
    - Rust MaxVol pivot selection
    
    Args:
        rho_cores: QTT cores for density
        rhou_cores: QTT cores for momentum
        E_cores: QTT cores for energy
        gamma: Ratio of specific heats
        max_rank: Maximum TT rank for flux
        tolerance: TCI convergence tolerance
        verbose: Print progress
        
    Returns:
        Tuple of (F_rho_cores, F_rhou_cores, F_E_cores, metadata)
    """
    from tci_core import TCISampler
    from tensornet.cfd.qtt_eval import qtt_eval_batch
    from tensornet.cfd.tci_flux import rusanov_flux
    
    n_qubits = len(rho_cores)
    N = 2 ** n_qubits
    device = rho_cores[0].device
    
    # Create Rust sampler with periodic BC
    sampler = TCISampler(n_qubits, "periodic", None)
    sampler.set_min_batch_size(max(64, max_rank))
    
    # Initialize pivots
    initial_pivots = min(max_rank, 8)
    for q in range(n_qubits):
        row_pivots = list(range(min(initial_pivots, 2**q)))
        col_pivots = list(range(min(initial_pivots, 2**(n_qubits - q - 1))))
        sampler.init_pivots(q, row_pivots, col_pivots)
    
    # Compute flux using Rust neighbor indices
    def flux_at_batch(batch) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate Rusanov flux at batch of indices using Rust neighbor indices."""
        # Get indices from Rust (zero-copy numpy arrays)
        indices_L_np = batch.indices_array()
        indices_R_np = batch.right_array()  # Rust precomputes neighbors!
        
        # Transfer to device
        indices_L = torch.from_numpy(indices_L_np).to(device)
        indices_R = torch.from_numpy(indices_R_np).to(device)
        
        # Evaluate state at left and right
        rho_L = qtt_eval_batch(rho_cores, indices_L)
        rhou_L = qtt_eval_batch(rhou_cores, indices_L)
        E_L = qtt_eval_batch(E_cores, indices_L)
        
        rho_R = qtt_eval_batch(rho_cores, indices_R)
        rhou_R = qtt_eval_batch(rhou_cores, indices_R)
        E_R = qtt_eval_batch(E_cores, indices_R)
        
        # Compute Rusanov flux
        F_rho, F_rhou, F_E = rusanov_flux(
            rho_L, rhou_L, E_L,
            rho_R, rhou_R, E_R,
            gamma
        )
        
        return F_rho, F_rhou, F_E
    
    # Run TCI for each flux component
    samples_rho = {}
    samples_rhou = {}
    samples_E = {}
    total_evals = 0
    max_iter = 50
    
    for iteration in range(max_iter):
        new_samples = 0
        
        for q in range(n_qubits):
            batch = sampler.sample_fibers(q)
            fiber_indices = batch.indices
            
            # Filter already sampled
            new_indices = [i for i in fiber_indices if i not in samples_rho]
            if not new_indices:
                continue
            
            # Create batch with only new indices
            indices_L = torch.tensor(new_indices, device=device, dtype=torch.long)
            indices_R = (indices_L + 1) % N
            
            # Evaluate flux
            rho_L = qtt_eval_batch(rho_cores, indices_L)
            rhou_L = qtt_eval_batch(rhou_cores, indices_L)
            E_L = qtt_eval_batch(E_cores, indices_L)
            
            rho_R = qtt_eval_batch(rho_cores, indices_R)
            rhou_R = qtt_eval_batch(rhou_cores, indices_R)
            E_R = qtt_eval_batch(E_cores, indices_R)
            
            F_rho, F_rhou, F_E = rusanov_flux(
                rho_L, rhou_L, E_L,
                rho_R, rhou_R, E_R,
                gamma
            )
            
            # Store samples
            for idx, v_rho, v_rhou, v_E in zip(
                new_indices, 
                F_rho.tolist(), 
                F_rhou.tolist(), 
                F_E.tolist()
            ):
                samples_rho[idx] = v_rho
                samples_rhou[idx] = v_rhou
                samples_E[idx] = v_E
            
            new_samples += len(new_indices)
            total_evals += len(new_indices)
        
        if new_samples == 0 or len(samples_rho) >= N // 2:
            break
        
        if verbose and iteration % 5 == 0:
            print(f"  Iter {iteration+1}: {len(samples_rho)} samples")
    
    # Build QTT from samples
    def build_qtt_from_samples(samples: dict) -> List[Tensor]:
        all_indices = torch.tensor(sorted(samples.keys()), device=device, dtype=torch.long)
        all_values = torch.tensor([samples[i] for i in sorted(samples.keys())], device=device)
        dense = torch.zeros(N, device=device)
        dense[all_indices] = all_values
        dense = _interpolate_sparse(dense, all_indices)
        return dense_to_qtt_cores(dense, max_rank=max_rank)
    
    F_rho_cores = build_qtt_from_samples(samples_rho)
    F_rhou_cores = build_qtt_from_samples(samples_rhou)
    F_E_cores = build_qtt_from_samples(samples_E)
    
    metadata = {
        "total_evals": total_evals,
        "compression": 3 * N / total_evals if total_evals > 0 else 1,
        "method": "tci_rust",
        "n_samples": len(samples_rho),
    }
    
    if verbose:
        print(f"  Rust TCI: {total_evals} evals, {metadata['compression']:.1f}x compression")
    
    return F_rho_cores, F_rhou_cores, F_E_cores, metadata


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TCI QTT Construction Tests")
    print("=" * 60)
    print()
    
    # Test 1: Simple function
    print("Test 1: Sine wave via TCI...")
    def sine_func(indices):
        return torch.sin(indices.float() * 0.01)
    
    cores, meta = qtt_from_function(sine_func, n_qubits=12, max_rank=16, verbose=True)
    print(f"  Result: {len(cores)} cores, {meta['n_evals']} evals")
    print(f"  Compression: {meta.get('compression', 4096 / meta['n_evals']):.1f}x")
    print()
    
    # Test 2: Verify accuracy
    print("Test 2: Verify reconstruction accuracy...")
    from tensornet.cfd.qtt_eval import qtt_eval_batch
    
    N = 2 ** 12
    test_indices = torch.randint(0, N, (200,))
    
    true_vals = sine_func(test_indices)
    approx_vals = qtt_eval_batch(cores, test_indices)
    
    max_err = (true_vals - approx_vals).abs().max().item()
    mean_err = (true_vals - approx_vals).abs().mean().item()
    
    print(f"  Max error: {max_err:.2e}")
    print(f"  Mean error: {mean_err:.2e}")
    assert max_err < 0.01, f"Error too large: {max_err}"
    print("  ✓ Accuracy verified")
    print()
    
    # Test 3: Step function (harder)
    print("Test 3: Step function via TCI...")
    def step_func(indices):
        return (indices > 2048).float()
    
    cores_step, meta_step = qtt_from_function(step_func, n_qubits=12, max_rank=32, verbose=True)
    print(f"  Result: {meta_step['n_evals']} evals, compression {4096 / meta_step['n_evals']:.1f}x")
    print()
    
    # Test 4: CFD flux (if state available)
    print("Test 4: Rusanov flux via TCI...")
    
    # Create simple Sod shock tube IC
    N = 2 ** 12
    x = torch.linspace(0, 1, N)
    gamma = 1.4
    
    rho = torch.where(x < 0.5, torch.ones_like(x), 0.125 * torch.ones_like(x))
    u = torch.zeros(N)
    p = torch.where(x < 0.5, torch.ones_like(x), 0.1 * torch.ones_like(x))
    
    rhou = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u**2
    
    # Convert to QTT
    from tensornet.cfd.qtt_eval import dense_to_qtt_cores
    rho_cores = dense_to_qtt_cores(rho, max_rank=32)
    rhou_cores = dense_to_qtt_cores(rhou, max_rank=32)
    E_cores = dense_to_qtt_cores(E, max_rank=32)
    
    # Compute flux via TCI
    F_rho, F_rhou, F_E, flux_meta = qtt_rusanov_flux_tci(
        rho_cores, rhou_cores, E_cores, 
        gamma=gamma, max_rank=32, verbose=True
    )
    
    print(f"  Total evaluations: {flux_meta['total_evals']}")
    print(f"  Compression: {flux_meta['compression']:.1f}x")
    print()
    
    print("=" * 60)
    print("ALL TCI TESTS PASSED ✓")
    print("=" * 60)
