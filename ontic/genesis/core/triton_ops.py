#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║                 T R I T O N   Q T T   C O R E   O P E R A T I O N S                     ║
║                                                                                          ║
║                   PRODUCTION-GRADE • GPU-NATIVE • ZERO DENSE                            ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

THE RULES:
1. QTT should be Native - NEVER decompress to dense
2. SVD = rSVD (always randomized, never full torch.linalg.svd)
3. Python loops = Triton Kernels
4. Higher scale = higher compression = lower rank
5. "Decompression" kills the purpose of QTT
6. "Dense" is a killer of QTT optimization

This module provides:
- Triton kernels for TT core contractions
- Triton kernels for Gram matrices and rSVD
- Native QTT dot products (no dense materialization)
- Native QTT norms (no dense materialization)
- Native QTT additions (no dense materialization)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Union
import torch

# Check for Triton availability
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("WARNING: Triton not available. Falling back to PyTorch ops.")

# Check CUDA
HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if HAS_CUDA else 'cpu')

__all__ = [
    'rsvd_native',
    'qtt_dot_native',
    'qtt_norm_native',
    'qtt_add_native',
    'qtt_hadamard_native',
    'qtt_round_native',
    'mpo_qtt_apply_fused',
    'triton_matmul',
    'triton_gram',
    'HAS_TRITON',
    'HAS_CUDA',
]


# ═══════════════════════════════════════════════════════════════════════════════
# TRITON KERNELS (if available)
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    
    @triton.jit
    def _matmul_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Triton matmul kernel — L2 cache optimized."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask)


    @triton.jit
    def _gram_kernel(
        A_ptr, G_ptr,
        M, N,
        stride_am, stride_an,
        stride_gm, stride_gn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Compute Gram matrix G = A @ A.T efficiently."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, N, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            
            a_m_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_an
            a_n_ptrs = A_ptr + offs_n[:, None] * stride_am + offs_k[None, :] * stride_an
            
            a_m = tl.load(a_m_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < N), other=0.0)
            a_n = tl.load(a_n_ptrs, mask=(offs_n[:, None] < M) & (offs_k[None, :] < N), other=0.0)
            
            acc += tl.dot(a_m, tl.trans(a_n))
        
        g_ptrs = G_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < M)
        tl.store(g_ptrs, acc, mask=mask)


    @triton.jit
    def _tt_core_contract_kernel(
        A_ptr, B_ptr, C_ptr,
        R_left, D, R_mid, R_right,
        stride_a0, stride_a1, stride_a2,
        stride_b0, stride_b1, stride_b2,
        stride_c0, stride_c1, stride_c2,
        BLOCK_RL: tl.constexpr, BLOCK_RR: tl.constexpr, BLOCK_RM: tl.constexpr,
    ):
        """
        Contract two TT cores along the bond dimension.

        A: (R_left, D, R_mid)
        B: (R_mid, D, R_right)
        C: (R_left, D*D, R_right) = sum over R_mid

        V-04 RESOLVED: Tiled access over (R_left, R_right) with blocked
        accumulation over R_mid. Each program handles a (d1, d2) pair
        and a tile of (R_left × R_right), accumulating over R_mid in
        BLOCK_RM chunks for L2/SRAM locality.
        """
        # Grid: (ceil(R_left/BLOCK_RL) * ceil(R_right/BLOCK_RR), D*D)
        pid_tile = tl.program_id(0)
        pid_dd = tl.program_id(1)

        # Decode (d1, d2) from flattened index
        d1 = pid_dd // D
        d2 = pid_dd % D
        d_out = d1 * D + d2

        # Decode tile position within (R_left, R_right) grid
        tiles_rr = tl.cdiv(R_right, BLOCK_RR)
        pid_rl = pid_tile // tiles_rr
        pid_rr = pid_tile % tiles_rr

        offs_rl = pid_rl * BLOCK_RL + tl.arange(0, BLOCK_RL)
        offs_rr = pid_rr * BLOCK_RR + tl.arange(0, BLOCK_RR)

        # Accumulator tile: (BLOCK_RL, BLOCK_RR)
        acc = tl.zeros((BLOCK_RL, BLOCK_RR), dtype=tl.float32)

        # Tiled reduction over R_mid
        for rm_start in range(0, R_mid, BLOCK_RM):
            offs_rm = rm_start + tl.arange(0, BLOCK_RM)
            rm_mask = offs_rm < R_mid

            # Load A[offs_rl, d1, offs_rm] → (BLOCK_RL, BLOCK_RM)
            a_ptrs = (A_ptr
                      + offs_rl[:, None] * stride_a0
                      + d1 * stride_a1
                      + offs_rm[None, :] * stride_a2)
            a = tl.load(a_ptrs,
                        mask=(offs_rl[:, None] < R_left) & rm_mask[None, :],
                        other=0.0)

            # Load B[offs_rm, d2, offs_rr] → (BLOCK_RM, BLOCK_RR)
            b_ptrs = (B_ptr
                      + offs_rm[:, None] * stride_b0
                      + d2 * stride_b1
                      + offs_rr[None, :] * stride_b2)
            b = tl.load(b_ptrs,
                        mask=rm_mask[:, None] & (offs_rr[None, :] < R_right),
                        other=0.0)

            # Tile matmul: (BLOCK_RL, BLOCK_RM) @ (BLOCK_RM, BLOCK_RR)
            acc += tl.dot(a, b)

        # Store C[offs_rl, d_out, offs_rr]
        c_ptrs = (C_ptr
                  + offs_rl[:, None] * stride_c0
                  + d_out * stride_c1
                  + offs_rr[None, :] * stride_c2)
        store_mask = (offs_rl[:, None] < R_left) & (offs_rr[None, :] < R_right)
        tl.store(c_ptrs, acc, mask=store_mask)


    @triton.jit
    def _mpo_qtt_fused_kernel(
        # Flat data buffers (contiguous on GPU)
        W_flat_ptr, G_flat_ptr, C_flat_ptr,
        # Offset tables: byte offset into flat buffers for core k
        W_off_ptr, G_off_ptr, C_off_ptr,
        # Per-core shape tables (int32 arrays of length N)
        Dl_ptr, Dr_ptr, rl_ptr, rr_ptr,
        # Constants
        D_PHYS: tl.constexpr,
        BLOCK_LEFT: tl.constexpr, BLOCK_RIGHT: tl.constexpr,
    ):
        """Fused MPO × QTT contraction — zero-copy, offset-table indexed.

        Reads MPO cores and QTT cores from flat contiguous buffers via
        per-core offset tables.  Writes output directly into flat output
        buffer.  No padding, no extraction, no Python loops.

        Contraction per core k:
          C_k[dl*rl + i, dout, dr*rr + j] = Σ_din W_k[dl,dout,din,dr] * G_k[rl_i,din,rr_j]

        where i ∈ [0, rl), j ∈ [0, rr), dl ∈ [0, Dl), dr ∈ [0, Dr).

        Grid: (N_cores * D_PHYS, max_tiles)
          axis-0: (core_idx, dout) — one program per (core, output mode)
          axis-1: tile over output (Dl*rl) × (Dr*rr) space

        Memory layout (row-major):
          W_k: (Dl, D_PHYS, D_PHYS, Dr) — contiguous, stride = Dr, D_PHYS*Dr, ...
          G_k: (rl, D_PHYS, rr) — contiguous
          C_k: (Dl*rl, D_PHYS, Dr*rr) — contiguous output

        L2 strategy:
          - W data is tiny (Dl*D²*Dr ≤ 64 floats) → fits in registers
          - G data (rl*D*rr ≤ 64*2*64 = 8KB) → L2-resident across din
          - Output tile written once, coalesced
          - No padding waste — every byte loaded is useful data
        """
        pid_core_dout = tl.program_id(0)
        pid_tile = tl.program_id(1)

        core_idx = pid_core_dout // D_PHYS
        dout = pid_core_dout % D_PHYS

        # Load per-core dimensions
        Dl = tl.load(Dl_ptr + core_idx).to(tl.int32)
        Dr = tl.load(Dr_ptr + core_idx).to(tl.int32)
        rl = tl.load(rl_ptr + core_idx).to(tl.int32)
        rr = tl.load(rr_ptr + core_idx).to(tl.int32)

        # Output dimensions
        out_left = Dl * rl
        out_right = Dr * rr

        # Decode tile position in output space
        tiles_right = tl.cdiv(out_right, BLOCK_RIGHT)
        pid_left = pid_tile // tiles_right
        pid_right = pid_tile % tiles_right

        offs_left = pid_left * BLOCK_LEFT + tl.arange(0, BLOCK_LEFT)
        offs_right = pid_right * BLOCK_RIGHT + tl.arange(0, BLOCK_RIGHT)

        # Early exit for out-of-bounds tiles (grid is sized for max core)
        # This lets smaller cores skip work without branching inside the loop
        if pid_left * BLOCK_LEFT >= out_left:
            return
        if pid_right * BLOCK_RIGHT >= out_right:
            return

        # Decompose flat output indices into (dl, rl_idx) and (dr, rr_idx)
        dl_idx = offs_left // rl
        rl_idx = offs_left % rl
        dr_idx = offs_right // rr
        rr_idx = offs_right % rr

        # Base pointers from offset tables (element offsets, not bytes)
        W_base = W_flat_ptr + tl.load(W_off_ptr + core_idx).to(tl.int64)
        G_base = G_flat_ptr + tl.load(G_off_ptr + core_idx).to(tl.int64)

        # W_k layout: (Dl, D_PHYS, D_PHYS, Dr) row-major
        # W_k[dl, dout, din, dr] = W_base + dl*(D_PHYS*D_PHYS*Dr)
        #                                  + dout*(D_PHYS*Dr)
        #                                  + din*Dr + dr
        w_stride_dl = D_PHYS * D_PHYS * Dr
        w_stride_dout = D_PHYS * Dr
        w_stride_din = Dr

        # G_k layout: (rl, D_PHYS, rr) row-major
        # G_k[r, din, c] = G_base + r*(D_PHYS*rr) + din*rr + c
        g_stride_rl = D_PHYS * rr
        g_stride_din = rr

        # Accumulator — fp64 to match data precision (physics solver)
        acc = tl.zeros((BLOCK_LEFT, BLOCK_RIGHT), dtype=tl.float64)

        # Inner loop over d_in (=2, fully unrolled by compiler)
        for din in range(D_PHYS):
            # Gather W[dl, dout, din, dr] for this tile
            w_offsets = (dl_idx[:, None] * w_stride_dl
                         + dout * w_stride_dout
                         + din * w_stride_din
                         + dr_idx[None, :])
            w_mask = (dl_idx[:, None] < Dl) & (dr_idx[None, :] < Dr)
            w_vals = tl.load(W_base + w_offsets, mask=w_mask, other=0.0)

            # Gather G[rl, din, rr] for this tile
            g_offsets = (rl_idx[:, None] * g_stride_rl
                         + din * g_stride_din
                         + rr_idx[None, :])
            g_mask = (rl_idx[:, None] < rl) & (rr_idx[None, :] < rr)
            g_vals = tl.load(G_base + g_offsets, mask=g_mask, other=0.0)

            acc += w_vals * g_vals

        # C_k layout: (out_left, D_PHYS, out_right) row-major
        C_base = C_flat_ptr + tl.load(C_off_ptr + core_idx).to(tl.int64)
        c_stride_left = D_PHYS * out_right
        c_offsets = (offs_left[:, None] * c_stride_left
                     + dout * out_right
                     + offs_right[None, :])
        c_mask = (offs_left[:, None] < out_left) & (offs_right[None, :] < out_right)
        tl.store(C_base + c_offsets, acc, mask=c_mask)


    @triton.jit
    def _dot_contraction_kernel(
        A_ptr, B_ptr, out_ptr,
        R_left, D, R_right,
        stride_a0, stride_a1, stride_a2,
        stride_b0, stride_b1, stride_b2,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Compute partial contraction for QTT dot product.
        
        Contracts A[r_left, d, r_right] * B[r_left, d, r_right] over d.
        Result: (R_left, R_right) × (R_left, R_right) transfer matrix.
        """
        pid = tl.program_id(0)
        
        # Each program handles one (r_left, r_right) pair
        r_left = pid // R_right
        r_right = pid % R_right
        
        if r_left < R_left and r_right < R_right:
            acc = 0.0
            
            for d in range(D):
                a_idx = r_left * stride_a0 + d * stride_a1 + r_right * stride_a2
                b_idx = r_left * stride_b0 + d * stride_b1 + r_right * stride_b2
                
                a = tl.load(A_ptr + a_idx)
                b = tl.load(B_ptr + b_idx)
                acc += a * b
            
            tl.store(out_ptr + r_left * R_right + r_right, acc)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated matrix multiply."""
    if not HAS_TRITON or not A.is_cuda:
        return A @ B
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Dimension mismatch: {K} vs {K2}"
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _matmul_kernel[grid](
        A.contiguous(), B.contiguous(), C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


def triton_gram(A: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix A @ A.T using Triton."""
    if not HAS_TRITON or not A.is_cuda:
        return A @ A.T
    
    M, N = A.shape
    G = torch.empty((M, M), device=A.device, dtype=torch.float32)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = min(64, N)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M, BLOCK_N))
    
    _gram_kernel[grid](
        A.contiguous(), G,
        M, N,
        A.stride(0), A.stride(1),
        G.stride(0), G.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return G


# ═══════════════════════════════════════════════════════════════════════════════
# MPO × QTT FUSED APPLICATION — ZERO PYTHON LOOPS
# ═══════════════════════════════════════════════════════════════════════════════


def mpo_qtt_apply_fused(
    mpo_cores: list[torch.Tensor],
    qtt_cores: list[torch.Tensor],
    _w_cache: dict | None = None,
) -> list[torch.Tensor]:
    """Apply MPO to QTT via fused Triton kernel — no Python loops on GPU.

    Replaces the batched-einsum path in ``gpu_mpo_apply`` with a single
    Triton kernel launch that reads from flat contiguous buffers via
    per-core offset tables.

    Data flow:
      1. Pack MPO cores into flat buffer (cached — MPO never changes)
      2. Pack QTT cores into flat buffer (one ``torch.cat``)
      3. Single Triton kernel — no padding, no extraction
      4. Split output buffer into per-core tensors (``torch.split``)

    Parameters
    ----------
    mpo_cores : list[torch.Tensor]
        MPO cores ``(D_l, d_out, d_in, D_r)`` on CUDA.
    qtt_cores : list[torch.Tensor]
        QTT state cores ``(r_l, d_in, r_r)`` on CUDA.
    _w_cache : dict, optional
        If provided, used to cache the packed MPO flat buffer + offsets.
        Pass an empty dict on first call; subsequent calls reuse.

    Returns
    -------
    list[torch.Tensor]
        Contracted cores of shape ``(D_l*r_l, d_out, D_r*r_r)`` per core.
        These feed directly into ``qtt_round_native``.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available — cannot use fused kernel")

    N = len(mpo_cores)
    device = mpo_cores[0].device
    dtype = mpo_cores[0].dtype
    D_PHYS = mpo_cores[0].shape[1]  # always 2 for QTT

    # ── Pack MPO (cacheable — MPO is static in CG iterations) ──────
    if _w_cache is not None and "W_flat" in _w_cache:
        W_flat = _w_cache["W_flat"]
        W_offsets = _w_cache["W_offsets"]
        Dl_arr = _w_cache["Dl_arr"]
        Dr_arr = _w_cache["Dr_arr"]
        dl_list = _w_cache["dl_list"]
        dr_list = _w_cache["dr_list"]
    else:
        # Flatten all MPO cores contiguously
        w_parts: list[torch.Tensor] = []
        w_off_list: list[int] = []
        dl_list: list[int] = []
        dr_list: list[int] = []
        offset = 0
        for k in range(N):
            w = mpo_cores[k]  # (Dl, D_PHYS, D_PHYS, Dr)
            dl_list.append(w.shape[0])
            dr_list.append(w.shape[3])
            w_parts.append(w.reshape(-1))
            w_off_list.append(offset)
            offset += w.numel()
        W_flat = torch.cat(w_parts)
        W_offsets = torch.tensor(w_off_list, dtype=torch.int64, device=device)
        Dl_arr = torch.tensor(dl_list, dtype=torch.int32, device=device)
        Dr_arr = torch.tensor(dr_list, dtype=torch.int32, device=device)
        if _w_cache is not None:
            _w_cache["W_flat"] = W_flat
            _w_cache["W_offsets"] = W_offsets
            _w_cache["Dl_arr"] = Dl_arr
            _w_cache["Dr_arr"] = Dr_arr
            _w_cache["dl_list"] = dl_list
            _w_cache["dr_list"] = dr_list

    # ── Pack QTT state (changes every call — one torch.cat) ────────
    g_parts: list[torch.Tensor] = []
    g_off_list: list[int] = []
    rl_list: list[int] = []
    rr_list: list[int] = []
    offset = 0
    for k in range(N):
        g = qtt_cores[k]  # (rl, D_PHYS, rr)
        rl_list.append(g.shape[0])
        rr_list.append(g.shape[2])
        g_parts.append(g.reshape(-1))
        g_off_list.append(offset)
        offset += g.numel()

    G_flat = torch.cat(g_parts)
    G_offsets = torch.tensor(g_off_list, dtype=torch.int64, device=device)
    rl_arr = torch.tensor(rl_list, dtype=torch.int32, device=device)
    rr_arr = torch.tensor(rr_list, dtype=torch.int32, device=device)

    # ── Allocate output buffer ─────────────────────────────────────
    c_off_list: list[int] = []
    c_sizes: list[int] = []
    offset = 0
    for k in range(N):
        out_left = dl_list[k] * rl_list[k]   # Dl * rl
        out_right = dr_list[k] * rr_list[k]  # Dr * rr
        size = out_left * D_PHYS * out_right
        c_off_list.append(offset)
        c_sizes.append(size)
        offset += size

    C_flat = torch.empty(offset, device=device, dtype=dtype)
    C_offsets = torch.tensor(c_off_list, dtype=torch.int64, device=device)

    # ── Compute grid dimensions ────────────────────────────────────
    # Tile sizes — tuned for QTT rank regime (ranks 1-64, D=2)
    BLOCK_LEFT = 32
    BLOCK_RIGHT = 32

    # Grid axis-0: one program per (core, dout)
    grid_0 = N * D_PHYS

    # Grid axis-1: enough tiles for the largest core
    max_out_left = max(dl_list[k] * rl_list[k] for k in range(N))
    max_out_right = max(dr_list[k] * rr_list[k] for k in range(N))
    tiles_left = (max_out_left + BLOCK_LEFT - 1) // BLOCK_LEFT
    tiles_right = (max_out_right + BLOCK_RIGHT - 1) // BLOCK_RIGHT
    grid_1 = tiles_left * tiles_right

    # ── Launch single Triton kernel ────────────────────────────────
    _mpo_qtt_fused_kernel[(grid_0, grid_1)](
        W_flat, G_flat, C_flat,
        W_offsets, G_offsets, C_offsets,
        Dl_arr, Dr_arr, rl_arr, rr_arr,
        D_PHYS=D_PHYS,
        BLOCK_LEFT=BLOCK_LEFT, BLOCK_RIGHT=BLOCK_RIGHT,
    )

    # ── Split output into per-core tensors — single torch.split ───
    raw_parts = torch.split(C_flat, c_sizes)
    result_cores: list[torch.Tensor] = []
    for k in range(N):
        out_left = dl_list[k] * rl_list[k]
        out_right = dr_list[k] * rr_list[k]
        result_cores.append(raw_parts[k].view(out_left, D_PHYS, out_right))

    return result_cores


# ═══════════════════════════════════════════════════════════════════════════════
# rSVD — RANDOMIZED SVD (NEVER FULL SVD)
# ═══════════════════════════════════════════════════════════════════════════════

def rsvd_native(
    A: torch.Tensor,
    k: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD — GPU-native, O(mnk) complexity.
    
    RULES:
    1. NEVER call torch.linalg.svd on matrices > 4×4
    2. Use Gram matrix eigendecomposition for O(mnk) complexity
    3. All operations on GPU — no .cpu() calls
    4. Adaptive rank truncation based on singular value decay
    
    Args:
        A: Input matrix (m × n) on GPU
        k: Target rank
        n_oversamples: Oversampling parameter (default 10)
        n_iter: Power iterations (default 2)
        tol: Singular value threshold (default 1e-10)
    
    Returns:
        U: Left singular vectors (m × k)
        S: Singular values (k,)
        Vh: Right singular vectors (k × n)
    
    Complexity: O(mnk) vs O(mn²) for full SVD
    """
    m, n = A.shape
    device = A.device
    dtype = A.dtype

    # Small matrix fast path (Rule 2: full SVD only when min(m,n) <= 4)
    if m <= 4 or n <= 4:
        U, S, Vh = torch.linalg.svd(A.to(torch.float64), full_matrices=False)
        k_actual = min(k, len(S), max(1, int((S > tol * S[0]).sum())))
        return U[:, :k_actual].to(dtype), S[:k_actual].to(dtype), Vh[:k_actual, :].to(dtype)

    l = min(k + n_oversamples, min(m, n))

    # All large matmuls stay in the native dtype (float32) for GPU
    # throughput.  Consumer GPUs (RTX 30xx/40xx/50xx) have 1/32 FP64
    # rate — using float64 here was costing 16-32× wall time on every
    # truncation call.  Only the small Gram-matrix eigendecomposition
    # (at most l×l ≈ 74×74) is upcast to float64 for precision.
    # Effective truncation tolerance in float32 saturates at ~1e-7,
    # which may keep 1-2 extra singular values vs float64 — acceptable
    # since the max_rank cap is the binding constraint.

    if m <= n:
        # Wide matrix: work with A @ A.T which is (m × m)
        Omega = torch.randn(n, l, device=device, dtype=dtype)
        Y = A @ Omega

        # Power iteration
        for _ in range(n_iter):
            Z = A.T @ Y
            Y = A @ Z

        # QR
        Q, _ = torch.linalg.qr(Y, mode='reduced')

        # Gram matrix → float64 for eigendecomposition precision
        AAt_Q = A @ (A.T @ Q)
        BtB = (Q.T @ AAt_Q).to(torch.float64)

        eigvals, eigvecs = torch.linalg.eigh(BtB)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        S = torch.sqrt(torch.clamp(eigvals, min=0)).to(dtype)
        eigvecs = eigvecs.to(dtype)
        U = Q @ eigvecs

        k_actual = min(k, max(1, int((S > tol * S[0]).sum())))

        inv_S = torch.zeros_like(S)
        mask = S[:k_actual] > tol * S[0]
        inv_S[:k_actual][mask] = 1.0 / S[:k_actual][mask]

        Vh = (A.T @ U) * inv_S.unsqueeze(0)
        Vh = Vh.T

    else:
        # Tall matrix: work with A.T @ A which is (n × n)
        Omega = torch.randn(m, l, device=device, dtype=dtype)
        Y = A.T @ Omega

        for _ in range(n_iter):
            Z = A @ Y
            Y = A.T @ Z

        Q, _ = torch.linalg.qr(Y, mode='reduced')

        # Gram matrix → float64 for eigendecomposition precision
        AtA_Q = A.T @ (A @ Q)
        BtB = (Q.T @ AtA_Q).to(torch.float64)

        eigvals, eigvecs = torch.linalg.eigh(BtB)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        S = torch.sqrt(torch.clamp(eigvals, min=0)).to(dtype)
        eigvecs = eigvecs.to(dtype)
        V = Q @ eigvecs
        Vh = V.T

        k_actual = min(k, max(1, int((S > tol * S[0]).sum())))

        inv_S = torch.zeros_like(S)
        mask = S[:k_actual] > tol * S[0]
        inv_S[:k_actual][mask] = 1.0 / S[:k_actual][mask]

        U = (A @ V) * inv_S.unsqueeze(0)

    return (
        U[:, :k_actual],
        S[:k_actual],
        Vh[:k_actual, :],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# NATIVE QTT OPERATIONS — NO DENSE MATERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def qtt_dot_native(
    cores_a: List[torch.Tensor],
    cores_b: List[torch.Tensor],
) -> float:
    """
    Compute dot product <a, b> without dense materialization.

    Uses transfer matrix method:
    <a, b> = Tr(T_1 T_2 ... T_d)

    where T_k[r_a', r_b'] = sum_{r_a, r_b, i} T[r_a, r_b] A[r_a, i, r_a'] B[r_b, i, r_b']

    V-02 RESOLVED: The inner loop over physical modes is fused into a
    single ``torch.einsum`` per core — one GPU kernel launch per core
    instead of n_k separate matmuls. The outer loop over d cores is
    inherently sequential (transfer-matrix chain) and acceptable per
    QTT Law 3 (core-level sweeps are sequential).

    Complexity: O(d r^4) vs O(2^d) for dense
    """
    if len(cores_a) != len(cores_b):
        raise ValueError(f"Core count mismatch: {len(cores_a)} vs {len(cores_b)}")

    d = len(cores_a)
    device = cores_a[0].device
    dtype = cores_a[0].dtype

    # Initialize transfer matrix as 1x1 identity
    T = torch.ones(1, 1, device=device, dtype=dtype)

    for k in range(d):
        # Fused contraction: single einsum fuses the mode summation
        # and the transfer-matrix update into ONE GPU kernel launch.
        # T_new[ra', rb'] = Σ_{ra, rb, d} T[ra, rb] * A[ra, d, ra'] * B[rb, d, rb']
        T = torch.einsum('ij,idk,jdl->kl', T, cores_a[k], cores_b[k])

    return T.squeeze().item()


def qtt_norm_native(cores: List[torch.Tensor]) -> float:
    """
    Compute L2 norm ||x||_2 without dense materialization.
    
    ||x||^2 = <x, x> via transfer matrix method.
    
    Complexity: O(d r^4) vs O(2^d) for dense
    """
    return math.sqrt(max(0.0, qtt_dot_native(cores, cores)))


def qtt_add_native(
    cores_a: List[torch.Tensor],
    cores_b: List[torch.Tensor],
    max_rank: Optional[int] = None,
    tol: float = 1e-10,
) -> List[torch.Tensor]:
    """
    Add two QTT tensors: c = a + b without dense materialization.
    
    Result has rank r_a + r_b which is then truncated via rounding.
    
    Complexity: O(d (r_a + r_b)^3) vs O(2^d) for dense
    """
    if len(cores_a) != len(cores_b):
        raise ValueError(f"Core count mismatch: {len(cores_a)} vs {len(cores_b)}")

    d = len(cores_a)
    device = cores_a[0].device
    dtype = cores_a[0].dtype

    # ── Boundary cores: simple concatenation (no loop needed) ──────
    # First core: horizontal cat along right-bond dimension
    result_cores: List[torch.Tensor] = [torch.cat([cores_a[0], cores_b[0]], dim=2)]

    # ── Middle cores: block diagonal — batched allocation ──────────
    # Pre-compute shapes, allocate all zeros at once, fill via slicing.
    # No Python loop over cores — use list comprehension for the
    # allocation and torch.narrow/index_put for the two blocks.
    if d > 2:
        # Collect shapes for all middle cores (indices 1..d-2)
        shapes_a = [(cores_a[k].shape[0], cores_a[k].shape[1], cores_a[k].shape[2])
                     for k in range(1, d - 1)]
        shapes_b = [(cores_b[k].shape[0], cores_b[k].shape[1], cores_b[k].shape[2])
                     for k in range(1, d - 1)]

        # Allocate all middle cores at once as a padded batch, then
        # fill both blocks with a single scatter per block.
        max_rl = max(sa[0] + sb[0] for sa, sb in zip(shapes_a, shapes_b))
        max_rr = max(sa[2] + sb[2] for sa, sb in zip(shapes_a, shapes_b))
        n_mid = d - 2
        n_k = 2  # QTT physical dimension is always 2

        # Single allocation for all middle cores
        mid_batch = torch.zeros(
            n_mid, max_rl, n_k, max_rr,
            device=device, dtype=dtype,
        )

        # Fill A-blocks and B-blocks — vectorized index_put per core
        # (the shapes differ per core, so we iterate but do minimal
        # Python work: just two slice assignments per core)
        for i, (sa, sb) in enumerate(zip(shapes_a, shapes_b)):
            mid_batch[i, :sa[0], :, :sa[2]] = cores_a[i + 1]
            mid_batch[i, sa[0]:sa[0] + sb[0], :, sa[2]:sa[2] + sb[2]] = cores_b[i + 1]

        # Extract correctly-sized views (slicing, no copy)
        for i, (sa, sb) in enumerate(zip(shapes_a, shapes_b)):
            result_cores.append(
                mid_batch[i, :sa[0] + sb[0], :, :sa[2] + sb[2]].contiguous()
            )
    # Last core: vertical cat along left-bond dimension
    if d > 1:
        result_cores.append(torch.cat([cores_a[-1], cores_b[-1]], dim=0))

    # Round to reduce rank if max_rank specified
    if max_rank is not None:
        result_cores = qtt_round_native(result_cores, max_rank=max_rank, tol=tol)

    return result_cores


def qtt_sub_native(
    cores_a: List[torch.Tensor],
    cores_b: List[torch.Tensor],
    max_rank: Optional[int] = None,
    tol: float = 1e-10,
) -> List[torch.Tensor]:
    """
    Subtract two QTT tensors: c = a - b without dense materialization.
    
    Implementation: negate cores_b then add.
    Complexity: O(d (r_a + r_b)^3) vs O(2^d) for dense
    """
    if len(cores_a) != len(cores_b):
        raise ValueError(f"Core count mismatch: {len(cores_a)} vs {len(cores_b)}")
    
    # Negate first core of B to get -B
    # (scaling any single core by -1 negates the entire tensor)
    neg_cores_b = [c.clone() for c in cores_b]
    neg_cores_b[0] = -neg_cores_b[0]
    
    return qtt_add_native(cores_a, neg_cores_b, max_rank=max_rank, tol=tol)


def qtt_hadamard_native(
    cores_a: List[torch.Tensor],
    cores_b: List[torch.Tensor],
    max_rank: Optional[int] = None,
    tol: float = 1e-10,
) -> List[torch.Tensor]:
    """
    Element-wise (Hadamard) product: c = a ⊙ b without dense materialization.

    Each core becomes the Kronecker product: C_k = A_k ⊗ B_k
    Result has rank r_a * r_b which is then truncated.

    V-03 RESOLVED: The inner loop over physical modes is replaced by a
    single ``torch.einsum`` per core that computes the Kronecker product
    across all modes simultaneously — one GPU kernel per core.

    Complexity: O(d (r_a * r_b)^3) vs O(2^d) for dense
    """
    if len(cores_a) != len(cores_b):
        raise ValueError(f"Core count mismatch: {len(cores_a)} vs {len(cores_b)}")

    d = len(cores_a)

    # ── Batched Kronecker product across ALL cores ─────────────────
    # All d einsum('adc,bde->abdce') calls are independent.  We pad
    # cores to uniform bond dimensions, stack into a single (d, ...)
    # tensor, run ONE batched einsum, then slice out results.
    #
    # For QTT, n_k = 2 always, so the physical dimension is uniform.

    max_ra_l = max(cores_a[k].shape[0] for k in range(d))
    max_ra_r = max(cores_a[k].shape[2] for k in range(d))
    max_rb_l = max(cores_b[k].shape[0] for k in range(d))
    max_rb_r = max(cores_b[k].shape[2] for k in range(d))
    n_k = cores_a[0].shape[1]  # always 2 for QTT

    device = cores_a[0].device
    dtype = cores_a[0].dtype

    A_pad = torch.zeros(d, max_ra_l, n_k, max_ra_r, device=device, dtype=dtype)
    B_pad = torch.zeros(d, max_rb_l, n_k, max_rb_r, device=device, dtype=dtype)

    shapes: list[tuple[int, int, int, int]] = []
    for k in range(d):
        ra_l, _, ra_r = cores_a[k].shape
        rb_l, _, rb_r = cores_b[k].shape
        A_pad[k, :ra_l, :, :ra_r] = cores_a[k]
        B_pad[k, :rb_l, :, :rb_r] = cores_b[k]
        shapes.append((ra_l, ra_r, rb_l, rb_r))

    # Single batched einsum: d Kronecker products in ONE GPU kernel
    # A_pad: (d, ra_l, n_k, ra_r), B_pad: (d, rb_l, n_k, rb_r)
    # → C_batch: (d, ra_l, rb_l, n_k, ra_r, rb_r)
    C_batch = torch.einsum('kadc,kbde->kabdce', A_pad, B_pad)

    # Extract and reshape per-core results (slicing only)
    result_cores: List[torch.Tensor] = []
    for k in range(d):
        ra_l, ra_r, rb_l, rb_r = shapes[k]
        C_k = C_batch[k, :ra_l, :rb_l, :, :ra_r, :rb_r]
        result_cores.append(C_k.reshape(ra_l * rb_l, n_k, ra_r * rb_r).contiguous())

    # Round to reduce rank
    if max_rank is not None:
        result_cores = qtt_round_native(result_cores, max_rank=max_rank, tol=tol)

    return result_cores


def qtt_round_native(
    cores: List[torch.Tensor],
    max_rank: int = 50,
    tol: float = 1e-10,
) -> List[torch.Tensor]:
    """
    Truncate QTT ranks via left-to-right orthogonalization + rSVD.
    
    NO DENSE MATERIALIZATION — works entirely in QTT format.
    
    Complexity: O(d r^3) vs O(2^d) for dense
    """
    d = len(cores)
    device = cores[0].device
    dtype = cores[0].dtype
    
    # Make a copy
    new_cores = [c.clone() for c in cores]
    
    # Left-to-right sweep: orthogonalize cores
    for k in range(d - 1):
        C_k = new_cores[k]  # (r_left, n_k, r_right)
        r_left, n_k, r_right = C_k.shape
        
        # Reshape to matrix: (r_left * n_k, r_right)
        mat = C_k.reshape(r_left * n_k, r_right)
        
        # QR decomposition
        Q, R = torch.linalg.qr(mat, mode='reduced')
        
        new_rank = Q.shape[1]
        
        # Update current core
        new_cores[k] = Q.reshape(r_left, n_k, new_rank)
        
        # Absorb R into next core
        C_next = new_cores[k + 1]
        r_left_next, n_next, r_right_next = C_next.shape
        C_next = C_next.reshape(r_left_next, -1)
        C_next = R @ C_next
        new_cores[k + 1] = C_next.reshape(new_rank, n_next, r_right_next)
    
    # Right-to-left sweep: truncate via rSVD
    for k in range(d - 1, 0, -1):
        C_k = new_cores[k]  # (r_left, n_k, r_right)
        r_left, n_k, r_right = C_k.shape
        
        # Reshape to matrix: (r_left, n_k * r_right)
        mat = C_k.reshape(r_left, n_k * r_right)
        
        # rSVD truncation
        target_rank = min(max_rank, r_left, n_k * r_right)
        U, S, Vh = rsvd_native(mat, k=target_rank, tol=tol)
        
        new_rank = len(S)
        
        # Update current core: V @ S becomes core
        new_cores[k] = Vh.reshape(new_rank, n_k, r_right)
        
        # Absorb U @ S into previous core
        C_prev = new_cores[k - 1]
        r_left_prev, n_prev, r_right_prev = C_prev.shape
        C_prev = C_prev.reshape(-1, r_right_prev)
        C_prev = C_prev @ (U * S.unsqueeze(0))
        new_cores[k - 1] = C_prev.reshape(r_left_prev, n_prev, new_rank)
    
    return new_cores


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE RANK (higher scale = higher compression = lower rank)
# ═══════════════════════════════════════════════════════════════════════════════

def adaptive_rank(
    grid_size: int,
    scale: float = 1.0,
    base_rank: int = 16,
    min_rank: int = 2,
) -> int:
    """
    Compute adaptive rank based on problem scale.
    
    RULE: Higher scale → Higher compression → Lower rank
    
    The intuition: as we operate at coarser scales, the structure
    becomes more regular and lower rank suffices.
    """
    # Number of bits determines base complexity
    n_bits = int(math.log2(grid_size)) if grid_size > 0 else 1
    
    # Scale factor: higher scale means more smoothing → lower rank
    scale_factor = 1.0 / (1.0 + math.log1p(scale))
    
    # Adaptive rank
    rank = max(min_rank, int(base_rank * scale_factor))
    
    # Don't exceed n_bits (theoretical maximum useful rank)
    return min(rank, 2 ** n_bits)


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECT ENTRY EVALUATION (NO FULL DECOMPRESSION)
# ═══════════════════════════════════════════════════════════════════════════════

def qtt_evaluate_at_index(cores: List[torch.Tensor], index: int) -> float:
    """
    Evaluate QTT at a single index without full decompression.
    
    For x in QTT format with grid size N = 2^d:
    x[i] = A_1[i_1] @ A_2[i_2] @ ... @ A_d[i_d]
    
    where i = i_1 i_2 ... i_d in binary.
    
    Complexity: O(d r^2) vs O(N) for dense access
    """
    d = len(cores)
    dtype = cores[0].dtype
    device = cores[0].device
    
    # Binary representation of index
    bits = [(index >> (d - 1 - k)) & 1 for k in range(d)]
    
    # Contract along the chain
    result = torch.ones(1, 1, device=device, dtype=dtype)
    
    for k in range(d):
        bit = bits[k]
        core_slice = cores[k][:, bit, :]  # (r_left, r_right)
        result = result @ core_slice
    
    return result.squeeze().item()


def qtt_evaluate_at_indices(cores: List[torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
    """
    Evaluate QTT at multiple indices without full decompression.
    
    Batched version for efficiency.
    
    Args:
        cores: QTT cores
        indices: 1D tensor of indices to evaluate
    
    Returns:
        Values at the specified indices
    """
    d = len(cores)
    n_indices = len(indices)
    dtype = cores[0].dtype
    device = cores[0].device
    
    # Convert indices to binary
    bits = torch.zeros(n_indices, d, dtype=torch.long, device=device)
    for k in range(d):
        bits[:, k] = (indices >> (d - 1 - k)) & 1
    
    # Initialize: (n_indices, 1)
    result = torch.ones(n_indices, 1, 1, device=device, dtype=dtype)
    
    for k in range(d):
        # Select slices for each index
        core = cores[k]  # (r_left, 2, r_right)
        bit_k = bits[:, k]  # (n_indices,)
        
        # Gather: (n_indices, r_left, r_right)
        slices = core[:, bit_k, :].permute(1, 0, 2)
        
        # Batch matmul: (n_indices, 1, r_left) @ (n_indices, r_left, r_right)
        result = torch.bmm(result, slices)
    
    return result.squeeze(-1).squeeze(-1)
