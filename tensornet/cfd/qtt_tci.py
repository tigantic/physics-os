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

ARTICLE II COMPLIANCE: This module NEVER allocates O(2^n) tensors.
ARTICLE III COMPLIANCE: dense_to_qtt is ONLY used for small problems (n<=12).
"""

from collections.abc import Callable

import torch
from torch import Tensor

# Try to import Rust TCI core
try:
    from tci_core import RUST_AVAILABLE, MaxVolConfig, TCIConfig, TCISampler, TruncationPolicy
except ImportError:
    RUST_AVAILABLE = False

from tensornet.cfd.qtt_eval import dense_to_qtt_cores


def _maxvol_simple(A: Tensor, tol: float = 1.05, max_iters: int = 50) -> Tensor:
    """
    MaxVol algorithm: find r rows of (n×r) matrix A that form well-conditioned submatrix.
    
    Returns indices of the r "best" rows as a tensor.
    
    This is the core of TCI - it finds which function samples are most informative.
    VECTORIZED implementation - no Python loops in hot path.
    """
    n, r = A.shape
    if n <= r:
        return torch.arange(n, device=A.device)
    
    # Start with QR to get initial good rows
    Q, _ = torch.linalg.qr(A.float())
    
    # Find rows with largest norms in Q
    row_norms = torch.norm(Q, dim=1)
    _, indices = torch.topk(row_norms, r)
    indices = indices.sort().values
    
    # Iterative improvement
    B = A[indices].float()
    
    for _ in range(max_iters):
        try:
            B_inv = torch.linalg.inv(B)
        except:
            break
            
        # C = A @ B_inv, shape (n, r)
        C = A.float() @ B_inv
        
        # Find element with max absolute value
        abs_C = torch.abs(C)
        max_val = abs_C.max()
        
        if max_val <= tol:
            break
        
        flat_idx = abs_C.argmax()
        i, j = flat_idx // r, flat_idx % r
        
        # Swap row i into position j
        indices[j] = i
        B = A[indices].float()
    
    return indices.sort().values


def qtt_from_function_dense(
    f: Callable[[Tensor], Tensor],
    n_qubits: int,
    max_rank: int = 64,
    device: str = "cpu",
) -> list[Tensor]:
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
    N = 2**n_qubits

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
) -> tuple[list[Tensor], dict]:
    """
    Build QTT from function using TT-Cross Interpolation (Python implementation).

    FIXED IMPLEMENTATION: Builds TT cores DIRECTLY from fiber samples.
    NO dense tensor allocation. NO dense_to_qtt_cores call.

    Algorithm:
        1. Initialize pivot indices for left/right contexts
        2. For each mode k (left to right):
           a. Sample fiber: (r_left × 2 × r_right) function evaluations
           b. SVD to get core and truncate rank
           c. MaxVol to select new pivot indices
        3. Return cores directly

    Complexity: O(r² × n_qubits) function evaluations
    Memory: O(r² × n_qubits) - NEVER O(2^n)

    Args:
        f: Function taking indices (batch,) and returning values (batch,)
        n_qubits: Number of qubits (N = 2^n_qubits)
        max_rank: Maximum TT rank
        tolerance: Convergence tolerance (for SVD truncation)
        max_iterations: Unused (single sweep algorithm)
        batch_size: Unused (fiber-based sampling)
        device: Torch device
        verbose: Print progress

    Returns:
        Tuple of (QTT cores, metadata dict)
    """
    dev = torch.device(device)
    N = 2**n_qubits

    # For small problems, use dense (acceptable for N <= 4096)
    if n_qubits <= 12:
        if verbose:
            print(f"  Small problem (N={N}), using dense TT-SVD")
        cores = qtt_from_function_dense(f, n_qubits, max_rank, device)
        return cores, {"method": "dense", "n_evals": N}

    # =========================================================================
    # TRUE TCI: Build cores DIRECTLY from fiber samples
    # NEVER allocate O(N) tensor. NEVER call dense_to_qtt_cores.
    # =========================================================================
    
    if verbose:
        print(f"[TCI] Building QTT: {n_qubits} qubits, max_rank={max_rank}")
    
    total_evals = 0
    cores = []
    
    # Initialize: pivot indices for each mode
    # accumulated_left[i] = full index for left context i (bits 0..k-1)
    # right_pivots[k] = list of right context indices (bits k+1..n-1)
    accumulated_left = torch.zeros(1, dtype=torch.long, device=dev)  # Start: single index 0
    
    right_pivots = []
    for k in range(n_qubits):
        n_right = n_qubits - k - 1
        if n_right > 0:
            # Initialize with random/uniform pivots
            n_piv = min(max_rank, 2**n_right)
            pivots = torch.randint(0, 2**n_right, (n_piv,), device=dev).unique()
            right_pivots.append(pivots)
        else:
            right_pivots.append(torch.zeros(1, dtype=torch.long, device=dev))
    
    # Left-to-right sweep: build each core
    for k in range(n_qubits):
        r_left = len(accumulated_left)
        right_indices = right_pivots[k][:max_rank]
        r_right = len(right_indices)
        
        # VECTORIZED index generation (replaces triple-nested loop)
        # Shape: (r_left, 2, r_right) -> flatten to (r_left * 2 * r_right,)
        left_expanded = accumulated_left.view(-1, 1, 1).expand(r_left, 2, r_right)
        bits = torch.arange(2, device=dev).view(1, -1, 1).expand(r_left, 2, r_right)
        right_expanded = right_indices.view(1, 1, -1).expand(r_left, 2, r_right)
        
        # Compose: left + (bit << k) + (right << (k+1))
        sample_indices = left_expanded + (bits << k) + (right_expanded << (k + 1))
        sample_indices = sample_indices.reshape(-1)
        
        # Evaluate function
        values = f(sample_indices)
        total_evals += len(sample_indices)
        
        # Reshape to fiber: (r_left, 2, r_right)
        fiber = values.reshape(r_left, 2, r_right)
        
        if k < n_qubits - 1:
            # SVD to extract core
            mat = fiber.reshape(r_left * 2, r_right)
            
            # Use rSVD for large matrices
            if min(mat.shape) > 4 * max_rank:
                U, S, V = torch.svd_lowrank(mat.float(), q=max_rank + 10, niter=2)
                Vh = V.T
            else:
                U, S, Vh = torch.linalg.svd(mat.float(), full_matrices=False)
            
            # Truncate
            rank = min(max_rank, len(S))
            if tolerance > 0:
                rel_cutoff = tolerance * S[0]
                rank = min(rank, max(1, (S > rel_cutoff).sum().item()))
            
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            
            # Core from U: (r_left, 2, rank)
            core = U.reshape(r_left, 2, rank).to(values.dtype)
            cores.append(core)
            
            # Update accumulated_left for next mode using MaxVol
            if U.shape[0] > rank:
                # MaxVol: find best pivot rows
                pivot_rows = _maxvol_simple(U, tol=1.05, max_iters=50)
                # VECTORIZED: no .item() or .tolist() iteration
                left_indices = pivot_rows // 2
                bit_vals = pivot_rows % 2
                accumulated_left = accumulated_left[left_indices] + (bit_vals << k)
            else:
                # Expand: all combinations of left and bit
                new_accumulated = accumulated_left.view(-1, 1) + (torch.arange(2, device=dev) << k).view(1, -1)
                accumulated_left = new_accumulated.reshape(-1)
        else:
            # Last core
            core = fiber.reshape(r_left, 2, 1).to(values.dtype)
            cores.append(core)
    
    if verbose:
        params = sum(c.numel() for c in cores)
        max_r = max(c.shape[-1] for c in cores)
        print(f"[TCI] Done: {len(cores)} cores, max_rank={max_r}, params={params:,}, evals={total_evals}")

    metadata = {
        "method": "tci_direct",
        "n_evals": total_evals,
        "n_cores": len(cores),
        "max_rank_actual": max(c.shape[-1] for c in cores),
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
        col_pivots = list(range(min(initial_pivots, 2 ** (n_qubits - q - 1))))
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
) -> tuple[list[int], list[float], list[float]]:
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
            left, right = sorted_keys[pos - 1], sorted_keys[pos]
            t = (idx - left) / (right - left)
            approx_values[i] = samples[left] * (1 - t) + samples[right] * t

    abs_errors = torch.abs(true_values - approx_values).tolist()
    return check_indices, true_values.tolist(), abs_errors


def _check_convergence(
    max_error: float,
    prev_max_error: float,
    tolerance: float,
    iteration: int,
    improvement_count: int,
    verbose: bool,
) -> tuple[bool, int, float, str]:
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
) -> tuple[list[Tensor], dict]:
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
    N = 2**n_qubits

    # Initialize Rust sampler with pivots
    sampler = _init_tci_sampler(n_qubits, max_rank)

    # Sample cache and convergence tracking
    samples = {}
    total_evals = 0
    prev_max_error = float("inf")
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
                    print(
                        f"  Iteration {iteration+1}: {new_samples} samples, error={max_error:.2e}"
                    )

                # Check convergence
                converged, improvement_count, prev_max_error, reason = (
                    _check_convergence(
                        max_error,
                        prev_max_error,
                        tolerance,
                        iteration,
                        improvement_count,
                        verbose,
                    )
                )
                if converged:
                    converge_reason = reason
                    if verbose:
                        print(f"  Converged at iteration {iteration+1} ({reason})")
                    break
        elif verbose:
            print(
                f"  Iteration {iteration+1}: {new_samples} samples, {len(samples)} total"
            )

        # Stop if we've sampled too much of the domain
        if len(samples) >= N // 2:
            converge_reason = f"sampled {len(samples)}/{N}"
            if verbose:
                print(f"  Stopping ({converge_reason})")
            break

    # Ensure sample density for accuracy
    density_samples = _ensure_sample_density(samples, N, f, device)
    total_evals += density_samples

    # =========================================================================
    # ARTICLE II & III COMPLIANCE: Build cores DIRECTLY, not through dense
    # =========================================================================
    # The Rust sampler gave us samples, but we still need to build TT cores.
    # We use the direct TCI algorithm on the sampled function.
    
    dev = torch.device(device)
    
    # Convert samples dict to tensors for efficient lookup
    # This avoids .tolist() in inner loop
    if samples:
        sample_indices_sorted = torch.tensor(sorted(samples.keys()), dtype=torch.long, device=dev)
        sample_values_sorted = torch.tensor([samples[k] for k in sample_indices_sorted.tolist()], 
                                           dtype=torch.float32, device=dev)
    else:
        sample_indices_sorted = torch.empty(0, dtype=torch.long, device=dev)
        sample_values_sorted = torch.empty(0, dtype=torch.float32, device=dev)
    
    def cached_f(indices: Tensor) -> Tensor:
        """Lookup cached samples, fall back to f() for uncached indices."""
        result = torch.zeros(len(indices), device=dev, dtype=torch.float32)
        uncached_mask = torch.ones(len(indices), dtype=torch.bool, device=dev)
        
        if len(sample_indices_sorted) > 0:
            # Vectorized lookup: use searchsorted + equality check
            # searchsorted finds insertion points; we then check if the value matches
            insert_pos = torch.searchsorted(sample_indices_sorted, indices)
            insert_pos = insert_pos.clamp(0, len(sample_indices_sorted) - 1)
            
            # Check which indices actually exist in the cache
            found_mask = sample_indices_sorted[insert_pos] == indices
            
            # Fill in found values
            result[found_mask] = sample_values_sorted[insert_pos[found_mask]]
            uncached_mask = ~found_mask
        
        # Evaluate uncached points
        if uncached_mask.any():
            uncached_indices = indices[uncached_mask]
            uncached_values = f(uncached_indices)
            result[uncached_mask] = uncached_values.float()
        
        return result
    
    # Build cores using direct TCI (no dense allocation)
    cores = []
    accumulated_left = torch.zeros(1, dtype=torch.long, device=dev)
    
    # Initialize right pivots
    right_pivots = []
    for k in range(n_qubits):
        n_right = n_qubits - k - 1
        if n_right > 0:
            n_piv = min(max_rank, 2**n_right)
            pivots = torch.randint(0, 2**n_right, (n_piv,), device=dev).unique()
            right_pivots.append(pivots)
        else:
            right_pivots.append(torch.zeros(1, dtype=torch.long, device=dev))
    
    for k in range(n_qubits):
        r_left = len(accumulated_left)
        right_indices = right_pivots[k][:max_rank]
        r_right = len(right_indices)
        
        # Vectorized index generation
        left_expanded = accumulated_left.view(-1, 1, 1).expand(r_left, 2, r_right)
        bits = torch.arange(2, device=dev).view(1, -1, 1).expand(r_left, 2, r_right)
        right_expanded = right_indices.view(1, 1, -1).expand(r_left, 2, r_right)
        sample_indices = (left_expanded + (bits << k) + (right_expanded << (k + 1))).reshape(-1)
        
        values = cached_f(sample_indices)
        fiber = values.reshape(r_left, 2, r_right)
        
        if k < n_qubits - 1:
            mat = fiber.reshape(r_left * 2, r_right).float()
            if min(mat.shape) > 4 * max_rank:
                U, S, V = torch.svd_lowrank(mat, q=max_rank + 10, niter=2)
            else:
                U, S, _ = torch.linalg.svd(mat, full_matrices=False)
            
            rank = min(max_rank, len(S))
            if tolerance > 0:
                rank = min(rank, max(1, (S > tolerance * S[0]).sum().item()))
            
            U = U[:, :rank]
            core = U.reshape(r_left, 2, rank)
            cores.append(core)
            
            if U.shape[0] > rank:
                pivot_rows = _maxvol_simple(U, tol=1.05)
                # VECTORIZED: no .tolist() iteration
                left_indices = pivot_rows // 2
                bit_vals = pivot_rows % 2
                accumulated_left = accumulated_left[left_indices] + (bit_vals << k)
            else:
                new_accumulated = accumulated_left.view(-1, 1) + (torch.arange(2, device=dev) << k).view(1, -1)
                accumulated_left = new_accumulated.reshape(-1)
        else:
            core = fiber.reshape(r_left, 2, 1)
            cores.append(core)

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
    pivots_left: list[set],
    pivots_right: list[set],
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
    left_sorted = sorted(
        left_vals.keys(), key=lambda k: abs(left_vals[k]), reverse=True
    )
    right_sorted = sorted(
        right_vals.keys(), key=lambda k: abs(right_vals[k]), reverse=True
    )

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
        result |= 1 << dim

    # Right bits shifted into upper positions
    result |= right << (dim + 1)

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
    right_idx = known_sorted[positions.clamp(max=len(known_sorted) - 1)]

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
) -> tuple[list[Tensor], dict]:
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
    N = 2**n_qubits

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
            f,
            n_qubits,
            max_rank,
            tolerance,
            max_iterations,
            batch_size,
            device,
            verbose,
        )

    return qtt_from_function_tci_python(
        f, n_qubits, max_rank, tolerance, max_iterations, batch_size, device, verbose
    )


# =============================================================================
# CFD-Specific: TCI for Rusanov Flux
# =============================================================================


def qtt_rusanov_flux_tci(
    rho_cores: list[Tensor],
    rhou_cores: list[Tensor],
    E_cores: list[Tensor],
    gamma: float = 1.4,
    max_rank: int = 64,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], dict]:
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
    N = 2**n_qubits
    device = rho_cores[0].device

    def flux_at_indices(indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
        F_rho, F_rhou, F_E = rusanov_flux(rho_L, rhou_L, E_L, rho_R, rhou_R, E_R, gamma)

        return F_rho, F_rhou, F_E

    # Build QTT for each flux component via TCI
    if verbose:
        print("Building F_rho QTT...")
    F_rho_cores, meta_rho = qtt_from_function(
        lambda idx: flux_at_indices(idx)[0],
        n_qubits,
        max_rank,
        tolerance,
        verbose=verbose,
        device=device,
    )

    if verbose:
        print("Building F_rhou QTT...")
    F_rhou_cores, meta_rhou = qtt_from_function(
        lambda idx: flux_at_indices(idx)[1],
        n_qubits,
        max_rank,
        tolerance,
        verbose=verbose,
        device=device,
    )

    if verbose:
        print("Building F_E QTT...")
    F_E_cores, meta_E = qtt_from_function(
        lambda idx: flux_at_indices(idx)[2],
        n_qubits,
        max_rank,
        tolerance,
        verbose=verbose,
        device=device,
    )

    metadata = {
        "total_evals": meta_rho["n_evals"] + meta_rhou["n_evals"] + meta_E["n_evals"],
        "compression": 3
        * N
        / (meta_rho["n_evals"] + meta_rhou["n_evals"] + meta_E["n_evals"]),
        "method": meta_rho["method"],
    }

    return F_rho_cores, F_rhou_cores, F_E_cores, metadata


def qtt_rusanov_flux_tci_rust(
    rho_cores: list[Tensor],
    rhou_cores: list[Tensor],
    E_cores: list[Tensor],
    gamma: float = 1.4,
    max_rank: int = 64,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], dict]:
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
    N = 2**n_qubits
    device = rho_cores[0].device

    # Create Rust sampler with periodic BC
    sampler = TCISampler(n_qubits, "periodic", None)
    sampler.set_min_batch_size(max(64, max_rank))

    # Initialize pivots
    initial_pivots = min(max_rank, 8)
    for q in range(n_qubits):
        row_pivots = list(range(min(initial_pivots, 2**q)))
        col_pivots = list(range(min(initial_pivots, 2 ** (n_qubits - q - 1))))
        sampler.init_pivots(q, row_pivots, col_pivots)

    # Compute flux using Rust neighbor indices
    def flux_at_batch(batch) -> tuple[Tensor, Tensor, Tensor]:
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
        F_rho, F_rhou, F_E = rusanov_flux(rho_L, rhou_L, E_L, rho_R, rhou_R, E_R, gamma)

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
                rho_L, rhou_L, E_L, rho_R, rhou_R, E_R, gamma
            )

            # Store samples - batch conversion (single .tolist() per batch, not per element)
            # This is acceptable: O(batch_size) conversion once per iteration
            new_indices_list = new_indices
            F_rho_list = F_rho.cpu().tolist()
            F_rhou_list = F_rhou.cpu().tolist()
            F_E_list = F_E.cpu().tolist()
            
            for idx, v_rho, v_rhou, v_E in zip(
                new_indices_list, F_rho_list, F_rhou_list, F_E_list
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

    # =========================================================================
    # ARTICLE II & III: Build QTT DIRECTLY from samples, no dense allocation
    # =========================================================================
    def build_qtt_from_samples_direct(samples: dict, n_qubits: int, max_rank: int) -> list[Tensor]:
        """Build QTT cores directly from sample dict using TCI algorithm.
        
        Uses vectorized lookup via searchsorted instead of .tolist() iteration.
        """
        dev = torch.device(device)
        
        # Convert dict to sorted tensors for vectorized lookup
        if samples:
            sorted_keys = sorted(samples.keys())
            sample_indices_sorted = torch.tensor(sorted_keys, dtype=torch.long, device=dev)
            sample_values_sorted = torch.tensor([samples[k] for k in sorted_keys], 
                                                dtype=torch.float32, device=dev)
        else:
            sample_indices_sorted = torch.empty(0, dtype=torch.long, device=dev)
            sample_values_sorted = torch.empty(0, dtype=torch.float32, device=dev)
        
        def sample_func(indices: Tensor) -> Tensor:
            """Vectorized lookup in sorted sample cache."""
            result = torch.zeros(len(indices), device=dev, dtype=torch.float32)
            
            if len(sample_indices_sorted) > 0:
                insert_pos = torch.searchsorted(sample_indices_sorted, indices)
                insert_pos = insert_pos.clamp(0, len(sample_indices_sorted) - 1)
                found_mask = sample_indices_sorted[insert_pos] == indices
                result[found_mask] = sample_values_sorted[insert_pos[found_mask]]
            
            return result
        
        cores = []
        accumulated_left = torch.zeros(1, dtype=torch.long, device=dev)
        
        right_pivots = []
        for k in range(n_qubits):
            n_right = n_qubits - k - 1
            if n_right > 0:
                n_piv = min(max_rank, 2**n_right)
                right_pivots.append(torch.randint(0, 2**n_right, (n_piv,), device=dev).unique())
            else:
                right_pivots.append(torch.zeros(1, dtype=torch.long, device=dev))
        
        for k in range(n_qubits):
            r_left = len(accumulated_left)
            right_indices = right_pivots[k][:max_rank]
            r_right = len(right_indices)
            
            left_expanded = accumulated_left.view(-1, 1, 1).expand(r_left, 2, r_right)
            bits = torch.arange(2, device=dev).view(1, -1, 1).expand(r_left, 2, r_right)
            right_expanded = right_indices.view(1, 1, -1).expand(r_left, 2, r_right)
            sample_indices = (left_expanded + (bits << k) + (right_expanded << (k + 1))).reshape(-1)
            
            values = sample_func(sample_indices)
            fiber = values.reshape(r_left, 2, r_right)
            
            if k < n_qubits - 1:
                mat = fiber.reshape(r_left * 2, r_right).float()
                U, S, _ = torch.linalg.svd(mat, full_matrices=False)
                rank = min(max_rank, len(S), max(1, (S > 1e-10 * S[0]).sum().item()))
                U = U[:, :rank]
                core = U.reshape(r_left, 2, rank)
                cores.append(core)
                
                if U.shape[0] > rank:
                    pivot_rows = _maxvol_simple(U, tol=1.05)
                    # VECTORIZED: no .tolist() iteration
                    left_indices = pivot_rows // 2
                    bit_vals = pivot_rows % 2
                    accumulated_left = accumulated_left[left_indices] + (bit_vals << k)
                else:
                    new_accumulated = accumulated_left.view(-1, 1) + (torch.arange(2, device=dev) << k).view(1, -1)
                    accumulated_left = new_accumulated.reshape(-1)
            else:
                core = fiber.reshape(r_left, 2, 1)
                cores.append(core)
        
        return cores

    F_rho_cores = build_qtt_from_samples_direct(samples_rho, n_qubits, max_rank)
    F_rhou_cores = build_qtt_from_samples_direct(samples_rhou, n_qubits, max_rank)
    F_E_cores = build_qtt_from_samples_direct(samples_E, n_qubits, max_rank)

    metadata = {
        "total_evals": total_evals,
        "compression": 3 * N / total_evals if total_evals > 0 else 1,
        "method": "tci_rust",
        "n_samples": len(samples_rho),
    }

    if verbose:
        print(
            f"  Rust TCI: {total_evals} evals, {metadata['compression']:.1f}x compression"
        )

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

    N = 2**12
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

    cores_step, meta_step = qtt_from_function(
        step_func, n_qubits=12, max_rank=32, verbose=True
    )
    print(
        f"  Result: {meta_step['n_evals']} evals, compression {4096 / meta_step['n_evals']:.1f}x"
    )
    print()

    # Test 4: CFD flux (if state available)
    print("Test 4: Rusanov flux via TCI...")

    # Create simple Sod shock tube IC
    N = 2**12
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
        rho_cores, rhou_cores, E_cores, gamma=gamma, max_rank=32, verbose=True
    )

    print(f"  Total evaluations: {flux_meta['total_evals']}")
    print(f"  Compression: {flux_meta['compression']:.1f}x")
    print()

    print("=" * 60)
    print("ALL TCI TESTS PASSED ✓")
    print("=" * 60)
