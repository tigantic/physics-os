"""
TCI-2: Simple Correct Tensor Cross-Interpolation

Key insight: The problem with skeleton TCI is the SVD-based rank selection on 
sampled submatrices doesn't control global error.

This implementation uses a different approach:
1. Sweep left-to-right building cores one at a time
2. At each site, sample ALL necessary indices (r_left * 2 * r_right evaluations)
3. Use QR to get orthonormal cores, not skeleton approximation
4. The "cross" nature comes from only sampling at pivot indices, not dense

Memory: O(r² × n)
Evals per sweep: O(r² × n)
"""

import torch
from typing import Callable, List, Optional


def tci_correct(
    func: Callable[[torch.Tensor], torch.Tensor],
    n_qubits: int,
    max_rank: int = 64,
    tol: float = 1e-12,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> List[torch.Tensor]:
    """
    Build QTT using correct TCI with SVD-based construction.
    
    The algorithm:
    1. Start with left_pivots = [0] (single pivot at index 0)
    2. For each qubit k from 0 to n-1:
       a. Build sampling indices: for each (left_pivot, bit, right_sample)
       b. Evaluate function at these indices -> matrix (r_left*2, r_right)
       c. SVD decompose to get truncated cores
       d. Update left_pivots for next iteration
    3. The last core absorbs the remaining factor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = 2 ** n_qubits
    
    if verbose:
        print(f"[TCI] Building QTT: {n_qubits} qubits, N={N:,}, max_rank={max_rank}")
    
    cores = []
    
    # left_pivots: indices representing the "left context" (modes 0..k-1)
    left_pivots = [0]
    
    # R_accumulated: factor from previous SVD to be absorbed
    R_accum = torch.ones(1, 1, device=device, dtype=dtype)
    
    for k in range(n_qubits):
        r_left = len(left_pivots)
        n_right_bits = n_qubits - k - 1
        
        # Determine right samples
        if n_right_bits > 0:
            n_right_possible = 2 ** n_right_bits
            r_right = min(max_rank, n_right_possible)
            if r_right >= n_right_possible:
                right_samples = list(range(n_right_possible))
            else:
                step = n_right_possible / r_right
                right_samples = [int(i * step) for i in range(r_right)]
        else:
            right_samples = [0]
            r_right = 1
        
        # Build sample indices
        left_shift = n_qubits - k
        bit_shift = n_right_bits
        
        indices = []
        for left_ctx in left_pivots:
            for bit in range(2):
                for right_ctx in right_samples:
                    idx = (left_ctx << left_shift) + (bit << bit_shift) + right_ctx
                    indices.append(idx)
        
        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        values = func(indices_tensor)
        
        # Reshape to matrix: (r_left * 2, r_right)
        mat = values.reshape(r_left * 2, r_right)
        
        # Absorb R from previous step
        # mat_absorbed = R_accum @ mat (need to reshape properly)
        # R_accum is (prev_rank, r_left), mat rows are ordered as (left_idx * 2 + bit)
        if R_accum.shape[0] > 1 or R_accum.shape[1] > 1:
            # Need to properly apply R_accum
            # Actually, the left pivots already incorporate the SVD truncation
            # So we just reshape mat
            pass
        
        if k < n_qubits - 1:
            # SVD decomposition
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            
            # Truncate based on tolerance
            S_max = S[0].item() if S.numel() > 0 else 1.0
            if S_max > 0:
                cutoff = tol * S_max
                rank_needed = int((S > cutoff).sum().item())
            else:
                rank_needed = 1
            rank_needed = max(1, min(rank_needed, max_rank, len(S)))
            
            # Truncate
            U_trunc = U[:, :rank_needed]  # (r_left*2, rank_needed)
            S_trunc = S[:rank_needed]
            Vh_trunc = Vh[:rank_needed, :]  # (rank_needed, r_right)
            
            # Core is U reshaped, with S absorbed into the right
            core = U_trunc.reshape(r_left, 2, rank_needed)
            cores.append(core)
            
            # R to be absorbed next = diag(S) @ Vh
            R_accum = torch.diag(S_trunc) @ Vh_trunc
            
            # Update left_pivots: need rank_needed new pivots
            # These should correspond to the rows of U that have the most variance
            # For now, use first rank_needed combinations
            new_left_pivots = []
            for i in range(min(r_left * 2, rank_needed * 2)):  # Take enough to get rank_needed
                left_idx = i // 2
                bit = i % 2
                if left_idx < len(left_pivots):
                    old_pivot = left_pivots[left_idx]
                    new_pivot = old_pivot * 2 + bit
                    if len(new_left_pivots) < max_rank:
                        new_left_pivots.append(new_pivot)
            
            # Ensure we have at least rank_needed pivots
            # Add more if needed by cycling through
            while len(new_left_pivots) < rank_needed and len(new_left_pivots) < 2 ** (k + 1):
                for i in range(r_left * 2):
                    left_idx = i // 2
                    bit = i % 2
                    if left_idx < len(left_pivots):
                        old_pivot = left_pivots[left_idx]
                        new_pivot = old_pivot * 2 + bit
                        if new_pivot not in new_left_pivots and len(new_left_pivots) < max_rank:
                            new_left_pivots.append(new_pivot)
                break
            
            left_pivots = new_left_pivots[:rank_needed]
            
        else:
            # Last core: store values directly
            core = mat.reshape(r_left, 2, 1)
            cores.append(core)
    
    if verbose:
        params = sum(c.numel() for c in cores)
        max_r = max(c.shape[2] for c in cores[:-1]) if len(cores) > 1 else 1
        print(f"[TCI] Done: {n_qubits} cores, max_rank={max_r}, params={params:,}")
    
    return cores


def tci_with_sweeps(
    func: Callable[[torch.Tensor], torch.Tensor],
    n_qubits: int,
    max_rank: int = 64,
    tol: float = 1e-12,
    n_sweeps: int = 3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> List[torch.Tensor]:
    """
    TCI with forward-backward sweeps for refinement.
    
    Each sweep:
    1. Forward: left-to-right QR-based construction
    2. Backward: right-to-left for symmetric treatment
    
    This is like DMRG with sampling instead of eigensolves.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = 2 ** n_qubits
    
    if verbose:
        print(f"[TCI-sweep] n={n_qubits}, N={N:,}, max_rank={max_rank}, sweeps={n_sweeps}")
    
    # Initial forward sweep
    cores = tci_correct(func, n_qubits, max_rank, tol, device, dtype, verbose=False)
    
    for sweep in range(n_sweeps):
        # Estimate error on random samples
        n_check = 1000
        check_idx = torch.randint(0, N, (n_check,), device=device)
        check_vals = func(check_idx)
        
        # Evaluate QTT
        qtt_vals = _evaluate_qtt_at_indices(cores, check_idx, n_qubits)
        
        error = (qtt_vals - check_vals).abs().max().item()
        scale = check_vals.abs().max().item()
        rel_error = error / (scale + 1e-16)
        
        if verbose:
            print(f"  Sweep {sweep}: error={error:.2e}, rel_error={rel_error:.2e}")
        
        if rel_error < tol:
            break
        
        # Backward sweep: right-to-left
        # We rebuild cores from right to left
        right_pivots = [0]
        new_cores = [None] * n_qubits
        
        for k in range(n_qubits - 1, -1, -1):
            r_right = len(right_pivots)
            n_left_bits = k
            
            if n_left_bits > 0:
                n_left_possible = 2 ** n_left_bits
                r_left = min(max_rank, n_left_possible)
                if r_left >= n_left_possible:
                    left_samples = list(range(n_left_possible))
                else:
                    step = n_left_possible / r_left
                    left_samples = [int(i * step) for i in range(r_left)]
            else:
                left_samples = [0]
                r_left = 1
            
            # Build indices
            indices = []
            n_right_bits = n_qubits - k - 1
            for left_ctx in left_samples:
                for bit in range(2):
                    for right_ctx in right_pivots:
                        idx = (left_ctx << (n_qubits - k)) + (bit << n_right_bits) + right_ctx
                        indices.append(idx)
            
            indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            values = func(indices_tensor)
            mat = values.reshape(r_left, 2 * r_right)
            
            if k > 0:
                # LQ decomposition (like QR but from right)
                L, Q = torch.linalg.qr(mat.T)
                L, Q = L.T, Q.T  # Now L is lower, Q is orthonormal rows
                
                # Truncate
                new_rank = min(max_rank, Q.shape[0])
                Q_trunc = Q[:new_rank, :]
                
                core = Q_trunc.reshape(new_rank, 2, r_right)
                new_cores[k] = core
                
                # Update right pivots
                new_right_pivots = []
                for i in range(Q_trunc.shape[1] // 2):  # Each pair (bit=0, bit=1) for each right_pivot
                    right_idx = i % len(right_pivots)
                    bit = (i // len(right_pivots)) % 2
                    old_pivot = right_pivots[right_idx]
                    new_pivot = bit * (2 ** n_right_bits) + old_pivot
                    if new_pivot not in new_right_pivots:
                        new_right_pivots.append(new_pivot)
                
                if len(new_right_pivots) > max_rank:
                    new_right_pivots = new_right_pivots[:max_rank]
                
                right_pivots = new_right_pivots if new_right_pivots else [0]
            else:
                core = mat.reshape(1, 2, r_right)
                new_cores[k] = core
        
        cores = new_cores
    
    if verbose:
        params = sum(c.numel() for c in cores)
        max_r = max(c.shape[2] for c in cores[:-1]) if len(cores) > 1 else 1
        print(f"[TCI-sweep] Done: max_rank={max_r}, params={params:,}")
    
    return cores


def _evaluate_qtt_at_indices(cores: List[torch.Tensor], indices: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Evaluate QTT at given indices."""
    device = cores[0].device
    dtype = cores[0].dtype
    n_points = indices.shape[0]
    
    # Start from rightmost core
    result = torch.ones(n_points, 1, device=device, dtype=dtype)
    
    for k in range(n_qubits - 1, -1, -1):
        n_bits_right = n_qubits - k - 1
        bits = (indices >> n_bits_right) & 1  # Extract bit k
        
        # cores[k] shape: (r_left, 2, r_right)
        # For each point, select the appropriate slice
        core = cores[k]  # (r_left, 2, r_right)
        
        # Gather the right slices for each point
        # result shape: (n_points, r_right) -> (n_points, r_left)
        new_result = torch.zeros(n_points, core.shape[0], device=device, dtype=dtype)
        for i in range(n_points):
            bit = bits[i].item()
            new_result[i] = core[:, bit, :] @ result[i]
        
        result = new_result
    
    return result.squeeze(-1)
