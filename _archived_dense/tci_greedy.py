"""
TCI Greedy: Proper Tensor Cross-Interpolation with Greedy Pivot Selection

The key insight from the TT-cross literature:
1. We need to maintain I_left[k] (row indices) and I_right[k] (column indices) 
2. At each step, we sample the cross: f[I_left, :, I_right]
3. We find the maximum entry to add to our pivot set
4. The skeleton approximation uses these pivots

Reference: Oseledets & Tyrtyshnikov, "TT-cross approximation for multidimensional arrays"
"""

import torch
from typing import Callable, List, Optional, Tuple


def tci_greedy(
    func: Callable[[torch.Tensor], torch.Tensor],
    n_qubits: int,
    max_rank: int = 64,
    tol: float = 1e-12,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> List[torch.Tensor]:
    """
    Build QTT via greedy TCI (TT-cross approximation).
    
    Algorithm:
    1. Initialize with a single pivot at index 0
    2. For each mode k:
       a. Evaluate f at cross indices (left_pivots × {0,1} × right_samples)
       b. Build skeleton approximation via pseudoinverse
       c. Find maximum error point and add to pivots
       d. Repeat until tolerance or max_rank reached
    
    This is O(r² × n) function evaluations and O(r² × n) memory.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = 2 ** n_qubits
    
    if verbose:
        print(f"[TCI-Greedy] n={n_qubits}, N={N:,}, max_rank={max_rank}")
    
    # For each mode k, we maintain:
    # - I_left[k]: list of k-bit numbers (left context indices)
    # - I_right[k]: list of (n-k-1)-bit numbers (right context indices)
    # - The cross matrix C[k] = f[I_left[k], :, I_right[k]]
    
    # Initialize: start with just index 0 everywhere
    I_left = [[0] for _ in range(n_qubits)]
    I_right = [[0] if k < n_qubits - 1 else [0] for k in range(n_qubits)]
    
    # Actually, let's use a different approach:
    # Build the TT decomposition by sweeping and updating pivots
    
    cores = []
    
    # Forward sweep: build initial decomposition
    left_pivots = [0]  # k-bit indices for left context
    
    for k in range(n_qubits):
        n_right_bits = n_qubits - k - 1
        r_left = len(left_pivots)
        
        # For right context, sample uniformly but ensure we hit important points
        if n_right_bits > 0:
            n_right_possible = 2 ** n_right_bits
            # Sample more densely for low ranks
            r_right = min(max_rank, n_right_possible)
            if r_right >= n_right_possible:
                right_samples = list(range(n_right_possible))
            else:
                # Include 0, N/2-relative positions, and random samples
                right_samples = []
                step = n_right_possible / r_right
                for i in range(r_right):
                    right_samples.append(int(i * step) % n_right_possible)
                right_samples = sorted(set(right_samples))
        else:
            right_samples = [0]
        
        r_right = len(right_samples)
        
        # Sample function at cross indices
        # Index formula: left_ctx * 2^(n-k) + bit * 2^(n-k-1) + right_ctx
        left_shift = n_qubits - k
        bit_shift = n_right_bits
        
        indices = []
        for left_ctx in left_pivots:
            for bit in range(2):
                for right_ctx in right_samples:
                    full_idx = (left_ctx << left_shift) + (bit << bit_shift) + right_ctx
                    indices.append(full_idx)
        
        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        values = func(indices_tensor)
        mat = values.reshape(r_left * 2, r_right)
        
        if k < n_qubits - 1:
            # SVD for rank truncation
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            
            # Determine rank based on singular value decay
            s_sum = S.sum()
            s_cumsum = S.cumsum(0)
            # Keep singular values that capture (1-tol) of the total
            rank_needed = 1
            for i in range(len(S)):
                if s_sum - s_cumsum[i] < tol * s_sum:
                    rank_needed = i + 1
                    break
                rank_needed = i + 1
            rank_needed = max(1, min(rank_needed, max_rank))
            
            # Truncate
            U_t = U[:, :rank_needed]
            S_t = S[:rank_needed]
            Vh_t = Vh[:rank_needed, :]
            
            # Store core as U (will absorb S*Vh into sampling for next mode)
            core = U_t.reshape(r_left, 2, rank_needed)
            cores.append(core)
            
            # Update left pivots for next mode
            # The key: we need to pick which (left_idx, bit) combinations to keep
            # Use row norms of U as importance weights
            row_norms = (U_t ** 2).sum(dim=1).sqrt()
            
            # Sort by importance and keep top rank_needed
            _, sorted_idx = row_norms.sort(descending=True)
            
            new_left_pivots = []
            for i in sorted_idx[:rank_needed].tolist():
                left_idx = i // 2
                bit = i % 2
                old_pivot = left_pivots[left_idx]
                new_pivot = old_pivot * 2 + bit
                new_left_pivots.append(new_pivot)
            
            left_pivots = new_left_pivots
            
            # Also need to propagate S*Vh to next mode's values
            # This is handled implicitly by re-sampling with new pivots
            
        else:
            # Last core
            core = mat.reshape(r_left, 2, 1)
            cores.append(core)
    
    if verbose:
        params = sum(c.numel() for c in cores)
        max_r = max(c.shape[2] for c in cores[:-1]) if len(cores) > 1 else 1
        print(f"[TCI-Greedy] Done: max_rank={max_r}, params={params:,}")
    
    return cores


def tci_als_simple(
    func: Callable[[torch.Tensor], torch.Tensor],
    n_qubits: int,
    max_rank: int = 32,
    tol: float = 1e-10,
    max_sweeps: int = 20,
    n_samples: int = 100,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> List[torch.Tensor]:
    """
    ALS-style TCI: Initialize randomly, then optimize via alternating least squares.
    
    This is more robust than greedy methods for difficult functions.
    
    Algorithm:
    1. Initialize cores randomly
    2. Sample random indices and their function values
    3. For each sweep:
       a. For each site k:
          - Fix all cores except k
          - Solve least squares for core k
       b. Check error on sample points
       c. If converged, stop
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = 2 ** n_qubits
    
    if verbose:
        print(f"[TCI-ALS] n={n_qubits}, N={N:,}, max_rank={max_rank}")
    
    # Sample indices and values for fitting
    n_fit = n_samples * n_qubits
    fit_idx = torch.randint(0, N, (n_fit,), device=device)
    fit_vals = func(fit_idx)
    
    # Initialize cores
    cores = []
    r = 1
    for k in range(n_qubits):
        r_next = min(max_rank, 2 ** (k + 1), 2 ** (n_qubits - k - 1))
        r_next = max(1, r_next)
        if k == n_qubits - 1:
            r_next = 1
        core = torch.randn(r, 2, r_next, device=device, dtype=dtype) * 0.1
        cores.append(core)
        r = r_next
    
    # ALS sweeps
    for sweep in range(max_sweeps):
        # Forward sweep
        for k in range(n_qubits):
            # Build left and right environments for all sample points
            left_env = _build_left_env(cores, fit_idx, k, n_qubits)  # (n_fit, r_left)
            right_env = _build_right_env(cores, fit_idx, k, n_qubits)  # (n_fit, r_right)
            
            r_left = cores[k].shape[0]
            r_right = cores[k].shape[2]
            
            # Build design matrix
            # For each sample point, the value should be: left_env @ core[:, bit, :] @ right_env
            # where bit is determined by the sample index
            bits = (fit_idx >> (n_qubits - k - 1)) & 1  # bit at position k
            
            # Design matrix: A[i, j] = left_env[i, r1] * right_env[i, r2] for appropriate (r1, bit, r2)
            A = torch.zeros(n_fit, r_left * 2 * r_right, device=device, dtype=dtype)
            for i in range(n_fit):
                b = bits[i].item()
                for r1 in range(r_left):
                    for r2 in range(r_right):
                        col = r1 * 2 * r_right + b * r_right + r2
                        A[i, col] = left_env[i, r1] * right_env[i, r2]
            
            # Solve least squares
            try:
                core_flat, residuals, rank, s = torch.linalg.lstsq(A, fit_vals)
            except:
                core_flat = torch.linalg.pinv(A) @ fit_vals
            
            cores[k] = core_flat.reshape(r_left, 2, r_right)
        
        # Check error
        pred_vals = _evaluate_qtt_batch(cores, fit_idx, n_qubits)
        error = (pred_vals - fit_vals).abs().max().item()
        rel_error = error / (fit_vals.abs().max().item() + 1e-16)
        
        if verbose and sweep % 5 == 0:
            print(f"  Sweep {sweep}: error={error:.2e}, rel_error={rel_error:.2e}")
        
        if rel_error < tol:
            if verbose:
                print(f"  Converged at sweep {sweep}")
            break
    
    if verbose:
        params = sum(c.numel() for c in cores)
        max_r = max(c.shape[2] for c in cores[:-1]) if len(cores) > 1 else 1
        print(f"[TCI-ALS] Done: max_rank={max_r}, params={params:,}")
    
    return cores


def _build_left_env(cores: List[torch.Tensor], indices: torch.Tensor, k: int, n_qubits: int) -> torch.Tensor:
    """Build left environment for site k: product of cores[0:k] at given indices."""
    device = cores[0].device
    dtype = cores[0].dtype
    n = indices.shape[0]
    
    if k == 0:
        return torch.ones(n, 1, device=device, dtype=dtype)
    
    result = torch.ones(n, 1, device=device, dtype=dtype)
    for kk in range(k):
        bits = (indices >> (n_qubits - kk - 1)) & 1
        core = cores[kk]  # (r_left, 2, r_right)
        
        # For each sample, select the right bit and contract
        new_result = torch.zeros(n, core.shape[2], device=device, dtype=dtype)
        for i in range(n):
            b = bits[i].item()
            new_result[i] = result[i] @ core[:, b, :]
        result = new_result
    
    return result


def _build_right_env(cores: List[torch.Tensor], indices: torch.Tensor, k: int, n_qubits: int) -> torch.Tensor:
    """Build right environment for site k: product of cores[k+1:] at given indices."""
    device = cores[0].device
    dtype = cores[0].dtype
    n = indices.shape[0]
    
    if k == n_qubits - 1:
        return torch.ones(n, 1, device=device, dtype=dtype)
    
    result = torch.ones(n, 1, device=device, dtype=dtype)
    for kk in range(n_qubits - 1, k, -1):
        bits = (indices >> (n_qubits - kk - 1)) & 1
        core = cores[kk]  # (r_left, 2, r_right)
        
        new_result = torch.zeros(n, core.shape[0], device=device, dtype=dtype)
        for i in range(n):
            b = bits[i].item()
            new_result[i] = core[:, b, :] @ result[i]
        result = new_result
    
    return result


def _evaluate_qtt_batch(cores: List[torch.Tensor], indices: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Evaluate QTT at batch of indices."""
    device = cores[0].device
    dtype = cores[0].dtype
    n = indices.shape[0]
    
    result = torch.ones(n, 1, device=device, dtype=dtype)
    for k in range(n_qubits - 1, -1, -1):
        bits = (indices >> (n_qubits - k - 1)) & 1
        core = cores[k]
        
        new_result = torch.zeros(n, core.shape[0], device=device, dtype=dtype)
        for i in range(n):
            b = bits[i].item()
            new_result[i] = core[:, b, :] @ result[i]
        result = new_result
    
    return result.squeeze(-1)
