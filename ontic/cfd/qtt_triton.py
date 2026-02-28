"""
Triton-accelerated QTT operations for CFD.

Optimized kernels for:
1. Residual absorption: R @ core (3D contraction)
2. Batched truncation sweeps across multiple fields

Target: 300-400ms per NS step at 32³ rank-32 (vs 2000ms+ with Python loops)
"""

import torch
from torch import Tensor
from typing import List, Tuple, Optional
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _residual_absorb_3d_kernel(
        R_ptr, core_ptr, out_ptr,
        M, K, D, N,
        stride_r_m, stride_r_k,
        stride_c_k, stride_c_d, stride_c_n,
        stride_o_m, stride_o_d, stride_o_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Compute out[m, d, n] = Σ_k R[m, k] · core[k, d, n]
        
        3D grid: (ceil(M/BLOCK_M), D, ceil(N/BLOCK_N))
        """
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_n = tl.program_id(2)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)

            # Load R[rm, rk]
            r_ptrs = R_ptr + rm[:, None] * stride_r_m + rk[None, :] * stride_r_k
            r_mask = (rm[:, None] < M) & (rk[None, :] < K)
            r_block = tl.load(r_ptrs, mask=r_mask, other=0.0)

            # Load core[rk, pid_d, rn]
            c_ptrs = core_ptr + rk[:, None] * stride_c_k + pid_d * stride_c_d + rn[None, :] * stride_c_n
            c_mask = (rk[:, None] < K) & (rn[None, :] < N)
            c_block = tl.load(c_ptrs, mask=c_mask, other=0.0)

            acc += tl.dot(r_block, c_block)

        # Store out[rm, pid_d, rn]
        o_ptrs = out_ptr + rm[:, None] * stride_o_m + pid_d * stride_o_d + rn[None, :] * stride_o_n
        o_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(o_ptrs, acc, mask=o_mask)


def triton_residual_absorb(R: Tensor, core: Tensor) -> Tensor:
    """
    Compute out = R @ core where:
    - R: (M, K) matrix (residual from previous SVD)
    - core: (K, D, N) tensor (QTT core)
    - out: (M, D, N) tensor
    
    This is the hot operation in TT-SVD truncation.
    Uses Triton kernel for fused 3D contraction.
    """
    if not TRITON_AVAILABLE:
        # Fallback to einsum
        return torch.einsum('mk,kdn->mdn', R, core)
    
    M, K_r = R.shape
    K_c, D, N = core.shape
    assert K_r == K_c, f"Dimension mismatch: R is ({M}, {K_r}), core is ({K_c}, {D}, {N})"
    K = K_r
    
    # Ensure contiguous
    R = R.contiguous()
    core = core.contiguous()
    
    out = torch.empty((M, D, N), device=R.device, dtype=R.dtype)
    
    # Block sizes tuned for small matrices (rank 16-64)
    BLOCK_M = min(32, M)
    BLOCK_N = min(32, N)
    BLOCK_K = min(32, K)
    
    # Round up to power of 2 for Triton
    BLOCK_M = max(16, 1 << (BLOCK_M - 1).bit_length())
    BLOCK_N = max(16, 1 << (BLOCK_N - 1).bit_length())
    BLOCK_K = max(16, 1 << (BLOCK_K - 1).bit_length())
    
    grid = (
        triton.cdiv(M, BLOCK_M),
        D,  # Physical dimension (always 2 for QTT)
        triton.cdiv(N, BLOCK_N),
    )
    
    _residual_absorb_3d_kernel[grid](
        R, core, out,
        M, K, D, N,
        R.stride(0), R.stride(1),
        core.stride(0), core.stride(1), core.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return out


def batched_truncation_sweep(
    fields: List[List[Tensor]],
    max_rank: int,
    tol: float = 1e-10,
) -> List[List[Tensor]]:
    """
    Truncate multiple QTT fields simultaneously with batched SVD.
    
    Instead of 6×15 = 90 individual SVD calls per sweep,
    we do 15 batched SVD calls (one per site) with batch size 6.
    
    Args:
        fields: List of B QTT fields, each a list of n_sites cores
        max_rank: Maximum rank after truncation
        tol: Relative tolerance for adaptive rank
    
    Returns:
        Truncated fields (modified in place and returned)
    
    Speedup: ~6x from batching, plus reduced Python overhead
    """
    if not fields:
        return fields
    
    B = len(fields)
    n_sites = len(fields[0])
    device = fields[0][0].device
    dtype = fields[0][0].dtype
    
    # Validate all fields have same structure
    for i, f in enumerate(fields):
        assert len(f) == n_sites, f"Field {i} has {len(f)} sites, expected {n_sites}"
    
    # Residuals for each field (initially None)
    residuals: List[Optional[Tensor]] = [None] * B
    
    # Process each site
    for site in range(n_sites - 1):
        # Step 1: Absorb residuals into cores (Triton kernel)
        cores_at_site = []
        shapes = []
        
        for i in range(B):
            core = fields[i][site]
            
            if residuals[i] is not None:
                # Fused 3D contraction: out = R @ core
                if TRITON_AVAILABLE:
                    core = triton_residual_absorb(residuals[i], core)
                else:
                    core = torch.einsum('mk,kdn->mdn', residuals[i], core)
            
            r_l, d, r_r = core.shape
            shapes.append((r_l, d, r_r))
            cores_at_site.append(core)
        
        # Step 2: Reshape to matrices and find max dimensions
        matrices = []
        for i, core in enumerate(cores_at_site):
            r_l, d, r_r = shapes[i]
            mat = core.reshape(r_l * d, r_r)
            matrices.append(mat)
        
        max_m = max(mat.shape[0] for mat in matrices)
        max_n = max(mat.shape[1] for mat in matrices)
        
        # Step 3: Pad and stack into batch
        batch = torch.zeros(B, max_m, max_n, device=device, dtype=dtype)
        for i, mat in enumerate(matrices):
            m, n = mat.shape
            batch[i, :m, :n] = mat
        
        # Step 4: ONE batched SVD call
        # cuSOLVER handles all B matrices in parallel
        U_batch, S_batch, Vh_batch = torch.linalg.svd(batch, full_matrices=False)
        
        # Step 5: Extract per-field results
        for i in range(B):
            r_l, d, r_r = shapes[i]
            actual_m = r_l * d
            actual_n = r_r
            
            # Determine rank (adaptive or fixed)
            S = S_batch[i, :min(actual_m, actual_n)]
            if tol > 0 and S.numel() > 0:
                # Adaptive: find rank where truncation error < tol
                total_norm_sq = (S ** 2).sum()
                cumsum = torch.cumsum(S ** 2, dim=0)
                tail_norm_sq = total_norm_sq - cumsum
                rel_error = torch.sqrt(tail_norm_sq / (total_norm_sq + 1e-30))
                valid = rel_error < tol
                if valid.any():
                    r = int(valid.nonzero()[0].item()) + 1
                else:
                    r = S.numel()
            else:
                r = min(actual_m, actual_n)
            
            r = min(r, max_rank)
            r = max(r, 1)  # At least rank 1
            
            # Extract U, reshape to core
            Ui = U_batch[i, :actual_m, :r]
            new_core = Ui.reshape(r_l, d, r)
            fields[i][site] = new_core
            
            # Form residual for next site: diag(S) @ Vh
            Si = S_batch[i, :r]
            Vhi = Vh_batch[i, :r, :actual_n]
            residuals[i] = Si.unsqueeze(1) * Vhi  # Broadcasting: (r,1) * (r,n) = (r,n)
    
    # Step 6: Absorb final residuals into last core
    for i in range(B):
        if residuals[i] is not None:
            core = fields[i][-1]
            if TRITON_AVAILABLE:
                fields[i][-1] = triton_residual_absorb(residuals[i], core)
            else:
                fields[i][-1] = torch.einsum('mk,kdn->mdn', residuals[i], core)
    
    return fields


def batched_truncate_single(
    cores_list: List[List[Tensor]],
    max_rank: int,
    tol: float = 1e-10,
) -> List[List[Tensor]]:
    """
    Convenience wrapper for batched_truncation_sweep.
    
    Takes a list of QTT core lists and truncates them all.
    """
    return batched_truncation_sweep(cores_list, max_rank, tol)


# ═══════════════════════════════════════════════════════════════════════════════════════
# PHASE-LEVEL BATCHED TRUNCATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def truncate_vorticity_fields(
    omega: List[List[Tensor]],
    max_rank: int,
    tol: float = 1e-10,
) -> List[List[Tensor]]:
    """
    Truncate all 3 vorticity components in one batched sweep.
    
    omega = [ωx, ωy, ωz], each a list of QTT cores.
    
    Uses batched SVD: 3 fields × 15 sites = 15 batched calls instead of 45.
    """
    return batched_truncation_sweep(omega, max_rank, tol)


def truncate_velocity_fields(
    u: List[List[Tensor]],
    max_rank: int,
    tol: float = 1e-10,
) -> List[List[Tensor]]:
    """
    Truncate all 3 velocity components in one batched sweep.
    """
    return batched_truncation_sweep(u, max_rank, tol)


def truncate_all_fields(
    omega: List[List[Tensor]],
    u: List[List[Tensor]],
    max_rank: int,
    tol: float = 1e-10,
) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
    """
    Truncate all 6 fields (ω + u) in one batched sweep.
    
    Maximum batching: 6 fields processed together.
    """
    all_fields = omega + u  # [ωx, ωy, ωz, ux, uy, uz]
    truncated = batched_truncation_sweep(all_fields, max_rank, tol)
    return truncated[:3], truncated[3:]


# ═══════════════════════════════════════════════════════════════════════════════════════
# BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════════════════

def benchmark_batched_vs_individual(n_bits: int = 5, max_rank: int = 32, n_trials: int = 10):
    """
    Compare batched truncation vs individual truncation.
    """
    import time
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_sites = 3 * n_bits
    
    # Create 6 random QTT fields
    def make_field():
        cores = []
        r = 1
        for i in range(n_sites):
            r_next = min(max_rank, 2 * r) if i < n_sites - 1 else 1
            core = torch.randn(r, 2, r_next, device=device)
            cores.append(core)
            r = r_next
        return cores
    
    fields = [make_field() for _ in range(6)]
    
    # Warm up
    _ = batched_truncation_sweep([f.copy() for f in fields], max_rank)
    torch.cuda.synchronize()
    
    # Benchmark batched
    fields_copy = [[c.clone() for c in f] for f in fields]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_trials):
        batched_truncation_sweep(fields_copy, max_rank)
        torch.cuda.synchronize()
    batched_time = (time.perf_counter() - t0) / n_trials * 1000
    
    # Benchmark individual (using turbo_truncate)
    from ontic.cfd.qtt_turbo import turbo_truncate
    
    fields_copy = [[c.clone() for c in f] for f in fields]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_trials):
        for f in fields_copy:
            turbo_truncate(f, max_rank)
        torch.cuda.synchronize()
    individual_time = (time.perf_counter() - t0) / n_trials * 1000
    
    print(f"Batched (6 fields):   {batched_time:.1f}ms")
    print(f"Individual (6 calls): {individual_time:.1f}ms")
    print(f"Speedup: {individual_time / batched_time:.1f}x")
    
    return batched_time, individual_time


if __name__ == "__main__":
    print("Testing Triton QTT operations...")
    
    if not TRITON_AVAILABLE:
        print("Triton not available, using fallback")
    else:
        print("Triton available")
    
    # Test residual absorption
    R = torch.randn(32, 48, device='cuda')
    core = torch.randn(48, 2, 64, device='cuda')
    
    out_triton = triton_residual_absorb(R, core)
    out_einsum = torch.einsum('mk,kdn->mdn', R, core)
    
    diff = (out_triton - out_einsum).abs().max().item()
    print(f"Triton vs einsum max diff: {diff:.2e}")
    
    # Benchmark
    print("\nBenchmarking batched vs individual truncation...")
    benchmark_batched_vs_individual()
