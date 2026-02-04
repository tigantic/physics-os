"""
Triton 3D Kernels for QTT Tensor-Train Operations
===================================================

Fused GPU kernels that eliminate Python loop overhead in the truncation 
sweep hot path. All kernels operate on 3D QTT cores (r_left, d, r_right)
where d=2 is the physical (binary) dimension.

Kernel inventory:
    1. residual_absorb_3d   - R @ core contraction: out[i,s,k] = sum_j R[i,j] * core[j,s,k]
    2. residual_form        - S_diag @ Vh:  R_new[i,k] = S[i] * Vh[i,k]
    3. batched_pad_cores    - Pad variable-size cores to common shape for batched SVD
    4. batched_unpad_extract - Extract truncated U, S, Vh per field from padded batched SVD output

Author: Brad / TiganticLabz (HAE-generated)
Target: CUDA via Triton JIT
"""

import torch
import math
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Triton import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

def _next_pow2(n: int) -> int:
    """Round up to nearest power of 2 (Triton BLOCK_SIZE constraint)."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


# ===========================================================================
# Kernel 1: 3D Residual Absorption
# ===========================================================================
# Computes: out[i, s, k] = sum_j R[i, j] * core[j, s, k]
# for all s in [0, d) simultaneously.
#
# This is the dominant operation in the QR+SVD truncation sweep.
# Called once per site per sweep. With 15 sites x 56 sweeps = 840 calls/step.
#
# 3D grid: (ceil(M/BM), D, ceil(N/BN))
#   - axis 0: output row blocks 
#   - axis 1: physical index d (always 2 for QTT)
#   - axis 2: output col blocks
# ===========================================================================

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
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_n = tl.program_id(2)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_off in range(0, K, BLOCK_K):
            rk = k_off + tl.arange(0, BLOCK_K)          # [BLOCK_K]

            # Load R[rm, rk]  -- shape (BLOCK_M, BLOCK_K)
            r_ptrs = R_ptr + rm[:, None] * stride_r_m + rk[None, :] * stride_r_k
            r_mask = (rm[:, None] < M) & (rk[None, :] < K)
            r_block = tl.load(r_ptrs, mask=r_mask, other=0.0)

            # Load core[rk, pid_d, rn]  -- shape (BLOCK_K, BLOCK_N)
            c_ptrs = (core_ptr
                      + rk[:, None] * stride_c_k
                      + pid_d * stride_c_d
                      + rn[None, :] * stride_c_n)
            c_mask = (rk[:, None] < K) & (rn[None, :] < N)
            c_block = tl.load(c_ptrs, mask=c_mask, other=0.0)

            acc += tl.dot(r_block, c_block)

        # Store out[rm, pid_d, rn]
        o_ptrs = (out_ptr
                  + rm[:, None] * stride_o_m
                  + pid_d * stride_o_d
                  + rn[None, :] * stride_o_n)
        o_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(o_ptrs, acc, mask=o_mask)


def triton_residual_absorb_3d(
    R: torch.Tensor,       # (M, K)
    core: torch.Tensor,    # (K, D, N)
) -> torch.Tensor:         # (M, D, N)
    """
    Fused 3D contraction: out[i, s, k] = sum_j R[i,j] * core[j, s, k].
    
    Falls back to einsum for CPU tensors or when Triton unavailable.
    Uses Triton for GPU tensors with total elements > 512 (below that,
    einsum dispatch is faster than Triton kernel launch overhead).
    """
    M, K = R.shape
    K2, D, N = core.shape
    assert K == K2, f"Contraction dim mismatch: R ({M},{K}) vs core ({K2},{D},{N})"

    # Fallback for small tensors, small K (tl.dot needs K>=16), or no Triton
    total_flops = M * K * D * N
    if not TRITON_AVAILABLE or not R.is_cuda or total_flops < 512 or K < 16:
        return torch.einsum('ij,jsk->isk', R, core)

    out = torch.empty(M, D, N, device=R.device, dtype=R.dtype)

    # Block sizes: powers of 2, tuned for typical QTT dimensions
    BLOCK_M = _next_pow2(min(M, 32))
    BLOCK_N = _next_pow2(min(N, 32))
    BLOCK_K = _next_pow2(min(K, 32))

    grid = (
        math.ceil(M / BLOCK_M),
        D,
        math.ceil(N / BLOCK_N),
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


# ===========================================================================
# Kernel 2: Residual Formation
# ===========================================================================
# Computes: R_new[i, k] = S[i] * Vh[i, k]
# This is a row-wise scaling — trivially parallelizable.
# ===========================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _residual_form_kernel(
        S_ptr, Vh_ptr, out_ptr,
        M, N,
        stride_vh_m, stride_vh_n,
        stride_o_m, stride_o_n,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        rn = tl.arange(0, BLOCK_N)

        if pid_m < M:
            s_val = tl.load(S_ptr + pid_m)

            for n_off in range(0, N, BLOCK_N):
                cols = n_off + rn
                mask = cols < N
                vh_ptrs = Vh_ptr + pid_m * stride_vh_m + cols * stride_vh_n
                vh_vals = tl.load(vh_ptrs, mask=mask, other=0.0)
                out_ptrs = out_ptr + pid_m * stride_o_m + cols * stride_o_n
                tl.store(out_ptrs, s_val * vh_vals, mask=mask)


def triton_residual_form(
    S: torch.Tensor,       # (r,)
    Vh: torch.Tensor,      # (r, N)
) -> torch.Tensor:         # (r, N)
    """
    Form residual matrix R = diag(S) @ Vh.
    
    Falls back to S[:, None] * Vh for small tensors.
    """
    r = S.shape[0]
    N = Vh.shape[1]

    if not TRITON_AVAILABLE or not S.is_cuda or r * N < 256:
        return S[:, None] * Vh

    out = torch.empty(r, N, device=S.device, dtype=S.dtype)
    BLOCK_N = _next_pow2(min(N, 128))

    _residual_form_kernel[(r,)](
        S, Vh, out,
        r, N,
        Vh.stride(0), Vh.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return out


# ===========================================================================
# Kernel 3: Batched Pad + Unpad for SVD
# ===========================================================================
# These are pure-PyTorch helpers (padding/unpadding is memory-bound,
# not compute-bound — Triton wouldn't help).
# ===========================================================================

def pad_matrices_to_batch(
    mats: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Pad a list of (m_i, n_i) matrices to a single (B, M_max, N_max) batch.
    
    Returns:
        batch: (B, M_max, N_max) tensor
        shapes: list of (m_i, n_i) original shapes
    """
    shapes = [(m.shape[0], m.shape[1]) for m in mats]
    M_max = max(s[0] for s in shapes)
    N_max = max(s[1] for s in shapes)
    B = len(mats)

    batch = torch.zeros(B, M_max, N_max, device=mats[0].device, dtype=mats[0].dtype)
    for i, m in enumerate(mats):
        batch[i, :m.shape[0], :m.shape[1]] = m

    return batch, shapes


def unpad_svd_results(
    U: torch.Tensor,       # (B, M_max, K)
    S: torch.Tensor,       # (B, K)
    Vh: torch.Tensor,      # (B, K, N_max)
    shapes: List[Tuple[int, int]],
    max_rank: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Extract per-field truncated (U_i, S_i, Vh_i) from batched SVD output.
    
    Returns list of (U_i[:m, :r], S_i[:r], Vh_i[:r, :n]) tuples.
    """
    results = []
    for i, (m, n) in enumerate(shapes):
        r = min(max_rank, min(m, n))
        Ui = U[i, :m, :r]            # (m, r)
        Si = S[i, :r]                # (r,)
        Vi = Vh[i, :r, :n]           # (r, n)
        results.append((Ui, Si, Vi))
    return results


# ===========================================================================
# Kernel 4: Fused MPO Apply (3D)
# ===========================================================================
# Computes: out[i*a, j, k*b] = sum_s core[i, s, k] * mpo[a, s, j, b]
# This is the derivative operation. Each MPO core has shape (r_m_l, d_in, d_out, r_m_r).
# For shift operators, typically r_m = 1 or 2.
# ===========================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _mpo_apply_3d_kernel(
        core_ptr, mpo_ptr, out_ptr,
        RI, D_IN, RK, RA, D_OUT, RB,
        stride_c_i, stride_c_s, stride_c_k,
        stride_m_a, stride_m_s, stride_m_j, stride_m_b,
        stride_o_ia, stride_o_j, stride_o_kb,
        BLOCK_I: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Grid: (ceil(RI/BLOCK_I) * RA, D_OUT, ceil(RK/BLOCK_K) * RB)
        Encodes (i_block, a) in axis 0 and (k_block, b) in axis 2.
        """
        # Decode compound indices
        pid_ia = tl.program_id(0)
        pid_j = tl.program_id(1)
        pid_kb = tl.program_id(2)

        n_i_blocks = tl.cdiv(RI, BLOCK_I)
        n_k_blocks = tl.cdiv(RK, BLOCK_K)

        i_block = pid_ia % n_i_blocks
        a = pid_ia // n_i_blocks
        k_block = pid_kb % n_k_blocks
        b = pid_kb // n_k_blocks

        ri = i_block * BLOCK_I + tl.arange(0, BLOCK_I)
        rk = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

        # Accumulate over s (physical input dimension, d=2 typically)
        acc = tl.zeros((BLOCK_I, BLOCK_K), dtype=tl.float32)

        for s in range(D_IN):
            # core[ri, s, rk]
            c_ptrs = core_ptr + ri[:, None] * stride_c_i + s * stride_c_s + rk[None, :] * stride_c_k
            c_mask = (ri[:, None] < RI) & (rk[None, :] < RK)
            c_val = tl.load(c_ptrs, mask=c_mask, other=0.0)

            # mpo[a, s, pid_j, b] -- scalar
            m_ptr = mpo_ptr + a * stride_m_a + s * stride_m_s + pid_j * stride_m_j + b * stride_m_b
            m_val = tl.load(m_ptr)

            acc += c_val * m_val

        # Store out[ri*RA + a, pid_j, rk*RB + b]
        out_rows = ri * RA + a
        out_cols = rk * RB + b
        o_ptrs = out_ptr + out_rows[:, None] * stride_o_ia + pid_j * stride_o_j + out_cols[None, :] * stride_o_kb
        o_mask = (ri[:, None] < RI) & (rk[None, :] < RK)
        tl.store(o_ptrs, acc, mask=o_mask)


def triton_mpo_apply_3d(
    core: torch.Tensor,    # (r_i, d_in, r_k)
    mpo: torch.Tensor,     # (r_a, d_in, d_out, r_b)
) -> torch.Tensor:         # (r_i * r_a, d_out, r_k * r_b)
    """
    Apply MPO core to state core via tensor contraction over physical index.
    
    For shift operators: mpo is typically (1, 2, 2, 1) or (2, 2, 2, 2).
    Falls back to einsum for small tensors or non-CUDA.
    """
    RI, D_IN, RK = core.shape
    RA, D_IN2, D_OUT, RB = mpo.shape
    assert D_IN == D_IN2

    # Einsum fallback — faster for small tensors
    if not TRITON_AVAILABLE or not core.is_cuda or RI * RK * RA * RB < 128:
        out = torch.einsum('isk,asjb->iajkb', core, mpo)
        return out.reshape(RI * RA, D_OUT, RK * RB)

    out = torch.empty(RI * RA, D_OUT, RK * RB, device=core.device, dtype=core.dtype)

    BLOCK_I = _next_pow2(min(RI, 16))
    BLOCK_K = _next_pow2(min(RK, 16))

    grid = (
        math.ceil(RI / BLOCK_I) * RA,
        D_OUT,
        math.ceil(RK / BLOCK_K) * RB,
    )

    _mpo_apply_3d_kernel[grid](
        core, mpo, out,
        RI, D_IN, RK, RA, D_OUT, RB,
        core.stride(0), core.stride(1), core.stride(2),
        mpo.stride(0), mpo.stride(1), mpo.stride(2), mpo.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_I=BLOCK_I,
        BLOCK_K=BLOCK_K,
    )
    return out


# ===========================================================================
# Kernel 5: Batched Inner Product Step
# ===========================================================================
# For computing <a, b> in QTT format without dense conversion.
# Contracts environment tensor with two cores:
#   env_new[ra', rb'] = sum_{ra, rb, s} env[ra, rb] * a[ra, s, ra'] * b[rb, s, rb']
# ===========================================================================

def inner_product_step(
    env: torch.Tensor,         # (r_a, r_b)
    core_a: torch.Tensor,      # (r_a, d, r_a')
    core_b: torch.Tensor,      # (r_b, d, r_b')
) -> torch.Tensor:             # (r_a', r_b')
    """
    One step of QTT inner product contraction.
    
    env_new[i', j'] = sum_{i, j, s} env[i, j] * a[i, s, i'] * b[j, s, j']
    """
    # Contract env with core_a: tmp[j, s, i'] = sum_i env[i, j] * a[i, s, i']
    tmp = torch.einsum('ij,isk->jsk', env, core_a)
    # Contract with core_b: out[i', j'] = sum_{j, s} tmp[j, s, i'] * b[j, s, j']
    out = torch.einsum('jsi,jsj->ij', tmp, core_b)
    return out


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    'TRITON_AVAILABLE',
    'triton_residual_absorb_3d',
    'triton_residual_form',
    'triton_mpo_apply_3d',
    'pad_matrices_to_batch',
    'unpad_svd_results',
    'inner_product_step',
]
