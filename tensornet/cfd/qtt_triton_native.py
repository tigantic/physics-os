"""
Triton-Native QTT Engine — L2 Cache Optimized, Adaptive Rank
=============================================================

Replaces Python for-loops in the QTT hot path with fused Triton kernels.
Every operation stays in QTT-compressed form. Zero dense intermediaries.

Core Principles:
    1. Adaptive rank — tolerance-driven, never fixed max_rank
    2. rSVD — randomized SVD for O(k·m·n) truncation
    3. Fused Triton kernels — one kernel launch per operation, not L launches
    4. PackedQTT — single contiguous GPU buffer per state, L2-friendly
    5. No decompression — all arithmetic in QTT form

Architecture:
    PackedQTT stores all L cores in a single GPU buffer with metadata arrays.
    Triton kernels access all sites via offsets — no Python iteration.
    rSVD adaptive truncation decides rank per bond from tolerance.

Kernel Map (replaces Python for-loops):
    block_scatter_kernel    → qtt_add  (fused block-diagonal assembly)
    kronecker_site_kernel   → qtt_hadamard (fused Kronecker product)
    mpo_contract_kernel     → mpo_apply (fused MPO contraction)
    env_contract_kernel     → qtt_inner (fused environment chain)

Author: HyperTensor Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

# ═══════════════════════════════════════════════════════════════════════════════
# PackedQTT: Contiguous GPU Storage
# ═══════════════════════════════════════════════════════════════════════════════


class PackedQTT:
    """
    QTT state stored as a single contiguous GPU buffer.

    Eliminates per-core allocation overhead and enables Triton kernels
    to access all L sites through offset arrays.

    Memory layout (row-major per core):
        data[offsets[k] : offsets[k+1]] = cores[k].flatten()
        where cores[k].shape = (rl[k], D, rr[k])

    Metadata arrays (int32, on GPU):
        rl[k]: left bond dimension at site k
        rr[k]: right bond dimension at site k
        offsets[k]: byte offset into data buffer (cumulative)
    """

    __slots__ = ("data", "offsets", "rl", "rr", "num_sites", "D", "device", "dtype")

    def __init__(self, cores: List[torch.Tensor]) -> None:
        if not cores:
            raise ValueError("Cannot create PackedQTT from empty core list")

        self.num_sites = len(cores)
        self.D = cores[0].shape[1]
        self.device = cores[0].device
        self.dtype = cores[0].dtype

        rl_list: List[int] = []
        rr_list: List[int] = []
        offsets_list: List[int] = [0]

        for c in cores:
            r_l, d, r_r = c.shape
            if d != self.D:
                raise ValueError(f"Physical dimension mismatch: {d} vs {self.D}")
            rl_list.append(r_l)
            rr_list.append(r_r)
            offsets_list.append(offsets_list[-1] + c.numel())

        self.rl = torch.tensor(rl_list, dtype=torch.int32, device=self.device)
        self.rr = torch.tensor(rr_list, dtype=torch.int32, device=self.device)
        self.offsets = torch.tensor(offsets_list, dtype=torch.int64, device=self.device)

        total = offsets_list[-1]
        self.data = torch.empty(total, dtype=self.dtype, device=self.device)
        for k, c in enumerate(cores):
            self.data[offsets_list[k] : offsets_list[k + 1]] = c.reshape(-1)

    def unpack(self) -> List[torch.Tensor]:
        """Unpack to list of core tensors (views into data buffer)."""
        offsets = self.offsets.tolist()
        rl = self.rl.tolist()
        rr = self.rr.tolist()
        cores: List[torch.Tensor] = []
        for k in range(self.num_sites):
            flat = self.data[offsets[k] : offsets[k + 1]]
            cores.append(flat.reshape(rl[k], self.D, rr[k]))
        return cores

    def clone(self) -> "PackedQTT":
        """Deep copy."""
        pq = object.__new__(PackedQTT)
        pq.data = self.data.clone()
        pq.offsets = self.offsets.clone()
        pq.rl = self.rl.clone()
        pq.rr = self.rr.clone()
        pq.num_sites = self.num_sites
        pq.D = self.D
        pq.device = self.device
        pq.dtype = self.dtype
        return pq

    @property
    def total_params(self) -> int:
        return int(self.data.numel())

    @property
    def max_rank(self) -> int:
        return int(max(self.rl.max().item(), self.rr.max().item()))

    @property
    def ranks(self) -> List[int]:
        """Bond dimensions: ranks[k] = rr[k] = rl[k+1]."""
        return self.rr[:-1].tolist()

    def memory_bytes(self) -> int:
        return self.data.numel() * self.data.element_size()


# ═══════════════════════════════════════════════════════════════════════════════
# Triton Kernel: Block-Diagonal Scatter (QTT Addition)
# ═══════════════════════════════════════════════════════════════════════════════
# Replaces: pure_qtt_ops.qtt_add for-loop (L iterations)
# Replaces: qtt_batched_ops.add_cores_raw for-loop (L iterations)


@triton.jit
def _block_scatter_kernel(
    # Source A
    a_data_ptr,
    a_off_ptr,
    a_rl_ptr,
    a_rr_ptr,
    # Source B
    b_data_ptr,
    b_off_ptr,
    b_rl_ptr,
    b_rr_ptr,
    # Output
    o_data_ptr,
    o_off_ptr,
    o_rl_ptr,
    o_rr_ptr,
    # Scalars
    alpha,
    beta,
    num_sites: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Scatter A and B cores into block-diagonal output cores.

    Grid: (num_sites, num_blocks_per_site)
    Each program copies elements from A or B into their position in the output.
    """
    site = tl.program_id(0)
    blk = tl.program_id(1)
    tid = blk * BLOCK + tl.arange(0, BLOCK)

    # Per-site shapes
    arl = tl.load(a_rl_ptr + site).to(tl.int32)
    arr = tl.load(a_rr_ptr + site).to(tl.int32)
    brl = tl.load(b_rl_ptr + site).to(tl.int32)
    brr = tl.load(b_rr_ptr + site).to(tl.int32)
    orl = tl.load(o_rl_ptr + site).to(tl.int32)
    orr = tl.load(o_rr_ptr + site).to(tl.int32)

    a_offset = tl.load(a_off_ptr + site)
    b_offset = tl.load(b_off_ptr + site)
    o_offset = tl.load(o_off_ptr + site)

    out_size = orl * D * orr
    mask = tid < out_size

    # Decode (i, d, j) from linear index in output
    j = tid % orr
    rem = tid // orr
    d = rem % D
    i = rem // D

    # Determine placement based on site position:
    # Site 0 (first): horizontal concat → A[:, :, :rA_r] | B[:, :, :rB_r]
    # Site L-1 (last): vertical concat → A[:rA_l, :, :] over B[:rB_l, :, :]
    # Middle: block diagonal → A in top-left, B in bottom-right
    is_first = site == 0
    is_last = site == num_sites - 1

    # ─── A contribution ───
    # Middle: A occupies rows [0, arl), cols [0, arr)
    a_valid_mid = (i < arl) & (j < arr) & (~is_first) & (~is_last)
    a_src_mid = i * D * arr + d * arr + j

    # First: A occupies full rows, cols [0, arr)
    a_valid_first = is_first & (j < arr)
    a_src_first = i * D * arr + d * arr + j

    # Last: A occupies rows [0, arl), full cols
    a_valid_last = is_last & (i < arl)
    a_src_last = i * D * 1 + d * 1 + j  # rr=1 for last A

    a_valid = a_valid_mid | a_valid_first | a_valid_last
    a_src = tl.where(is_first, a_src_first, tl.where(is_last, a_src_last, a_src_mid))
    a_val = tl.load(a_data_ptr + a_offset + a_src, mask=mask & a_valid, other=0.0)

    # ─── B contribution ───
    # Middle: B occupies rows [arl, arl+brl), cols [arr, arr+brr)
    bi = i - arl
    bj = j - arr
    b_valid_mid = (bi >= 0) & (bi < brl) & (bj >= 0) & (bj < brr) & (~is_first) & (~is_last)
    b_src_mid = bi * D * brr + d * brr + bj

    # First: B occupies full rows, cols [arr, arr+brr)
    bj_first = j - arr
    b_valid_first = is_first & (bj_first >= 0) & (bj_first < brr)
    b_src_first = i * D * brr + d * brr + bj_first

    # Last: B occupies rows [arl, arl+brl), full cols
    bi_last = i - arl
    b_valid_last = is_last & (bi_last >= 0) & (bi_last < brl)
    b_src_last = bi_last * D * 1 + d * 1 + j

    b_valid = b_valid_mid | b_valid_first | b_valid_last
    b_src = tl.where(is_first, b_src_first, tl.where(is_last, b_src_last, b_src_mid))
    b_val = tl.load(b_data_ptr + b_offset + b_src, mask=mask & b_valid, other=0.0)

    # Weighted sum
    result = alpha * a_val + beta * b_val
    tl.store(o_data_ptr + o_offset + tid, result, mask=mask)


def triton_qtt_add(
    a: PackedQTT,
    b: PackedQTT,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> PackedQTT:
    """
    QTT addition: alpha * |a⟩ + beta * |b⟩

    Fused Triton kernel — one launch for all L sites.
    No Python loop. Ranks add: r_out = r_a + r_b (before truncation).
    """
    L = a.num_sites
    assert b.num_sites == L
    D = a.D

    # Compute output shapes
    a_rl = a.rl.tolist()
    a_rr = a.rr.tolist()
    b_rl = b.rl.tolist()
    b_rr = b.rr.tolist()

    o_rl_list: List[int] = []
    o_rr_list: List[int] = []
    offsets: List[int] = [0]

    for k in range(L):
        if k == 0:
            orl = 1  # First site: horizontal concat
            orr = a_rr[k] + b_rr[k]
        elif k == L - 1:
            orl = a_rl[k] + b_rl[k]  # Last site: vertical concat
            orr = 1
        else:
            orl = a_rl[k] + b_rl[k]  # Middle: block diagonal
            orr = a_rr[k] + b_rr[k]
        o_rl_list.append(orl)
        o_rr_list.append(orr)
        offsets.append(offsets[-1] + orl * D * orr)

    o_rl = torch.tensor(o_rl_list, dtype=torch.int32, device=a.device)
    o_rr = torch.tensor(o_rr_list, dtype=torch.int32, device=a.device)
    o_offsets = torch.tensor(offsets, dtype=torch.int64, device=a.device)
    o_data = torch.zeros(offsets[-1], dtype=a.dtype, device=a.device)

    max_out_size = max(o_rl_list[k] * D * o_rr_list[k] for k in range(L))
    BLOCK = 256
    num_blocks = (max_out_size + BLOCK - 1) // BLOCK

    _block_scatter_kernel[(L, num_blocks)](
        a.data,
        a.offsets,
        a.rl,
        a.rr,
        b.data,
        b.offsets,
        b.rl,
        b.rr,
        o_data,
        o_offsets,
        o_rl,
        o_rr,
        alpha,
        beta,
        L,
        D,
        BLOCK,
    )

    out = object.__new__(PackedQTT)
    out.data = o_data
    out.offsets = o_offsets
    out.rl = o_rl
    out.rr = o_rr
    out.num_sites = L
    out.D = D
    out.device = a.device
    out.dtype = a.dtype
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Triton Kernel: Kronecker Product (QTT Hadamard / Element-wise Product)
# ═══════════════════════════════════════════════════════════════════════════════
# Replaces: pure_qtt_ops.qtt_hadamard for-loop (L iterations)
# Replaces: qtt_batched_ops.hadamard_cores_raw for-loop (L iterations)


@triton.jit
def _kronecker_site_kernel(
    a_data_ptr,
    a_off_ptr,
    a_rl_ptr,
    a_rr_ptr,
    b_data_ptr,
    b_off_ptr,
    b_rl_ptr,
    b_rr_ptr,
    o_data_ptr,
    o_off_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Kronecker product at each site: out[i*rb+p, d, j*rb+q] = A[i,d,j] * B[p,d,q]

    Grid: (num_sites, num_blocks_per_site)
    """
    site = tl.program_id(0)
    blk = tl.program_id(1)
    tid = blk * BLOCK + tl.arange(0, BLOCK)

    arl = tl.load(a_rl_ptr + site).to(tl.int32)
    arr = tl.load(a_rr_ptr + site).to(tl.int32)
    brl = tl.load(b_rl_ptr + site).to(tl.int32)
    brr = tl.load(b_rr_ptr + site).to(tl.int32)

    orl = arl * brl
    orr = arr * brr
    out_size = orl * D * orr
    mask = tid < out_size

    a_off = tl.load(a_off_ptr + site)
    b_off = tl.load(b_off_ptr + site)
    o_off = tl.load(o_off_ptr + site)

    # Decode output (io, d, jo) where io = i*brl + p, jo = j*brr + q
    jo = tid % orr
    rem = tid // orr
    d = rem % D
    io = rem // D

    # A indices
    i = io // brl
    j = jo // brr
    a_idx = i * D * arr + d * arr + j

    # B indices
    p = io % brl
    q = jo % brr
    b_idx = p * D * brr + d * brr + q

    a_val = tl.load(a_data_ptr + a_off + a_idx, mask=mask, other=0.0)
    b_val = tl.load(b_data_ptr + b_off + b_idx, mask=mask, other=0.0)

    tl.store(o_data_ptr + o_off + tid, a_val * b_val, mask=mask)


def triton_qtt_hadamard(a: PackedQTT, b: PackedQTT) -> PackedQTT:
    """
    Element-wise product in QTT form: |a ⊙ b⟩

    Fused Triton kernel — one launch for all sites.
    Rank multiplies: r_out = r_a × r_b (before truncation).
    """
    L = a.num_sites
    assert b.num_sites == L
    D = a.D

    a_rl, a_rr = a.rl.tolist(), a.rr.tolist()
    b_rl, b_rr = b.rl.tolist(), b.rr.tolist()

    o_rl_list = [a_rl[k] * b_rl[k] for k in range(L)]
    o_rr_list = [a_rr[k] * b_rr[k] for k in range(L)]
    offsets: List[int] = [0]
    for k in range(L):
        offsets.append(offsets[-1] + o_rl_list[k] * D * o_rr_list[k])

    o_rl = torch.tensor(o_rl_list, dtype=torch.int32, device=a.device)
    o_rr = torch.tensor(o_rr_list, dtype=torch.int32, device=a.device)
    o_offsets = torch.tensor(offsets, dtype=torch.int64, device=a.device)
    o_data = torch.empty(offsets[-1], dtype=a.dtype, device=a.device)

    max_out = max(o_rl_list[k] * D * o_rr_list[k] for k in range(L))
    BLOCK = 256
    nblk = (max_out + BLOCK - 1) // BLOCK

    _kronecker_site_kernel[(L, nblk)](
        a.data, a.offsets, a.rl, a.rr,
        b.data, b.offsets, b.rl, b.rr,
        o_data, o_offsets,
        D, BLOCK,
    )

    out = object.__new__(PackedQTT)
    out.data = o_data
    out.offsets = o_offsets
    out.rl = o_rl
    out.rr = o_rr
    out.num_sites = L
    out.D = D
    out.device = a.device
    out.dtype = a.dtype
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Triton Kernel: MPO Application
# ═══════════════════════════════════════════════════════════════════════════════
# Replaces: pure_qtt_ops.apply_mpo for-loop (L iterations)
# Each site: O[oL, d_out, d_in, oR] × P[pL, d_in, pR] → (oL*pL, d_out, oR*pR)


@triton.jit
def _mpo_contract_kernel(
    mpo_data_ptr,
    mpo_off_ptr,
    mpo_rl_ptr,
    mpo_rr_ptr,
    qtt_data_ptr,
    qtt_off_ptr,
    qtt_rl_ptr,
    qtt_rr_ptr,
    out_data_ptr,
    out_off_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Contract MPO[oL, d_out, d_in, oR] with QTT[pL, d_in, pR] per site.
    Output: (oL*pL, d_out, oR*pR)

    Grid: (num_sites, num_blocks_per_site)
    """
    site = tl.program_id(0)
    blk = tl.program_id(1)
    tid = blk * BLOCK + tl.arange(0, BLOCK)

    # MPO shapes: stored as (oL, D, D, oR) → total elements = oL*D*D*oR
    mol = tl.load(mpo_rl_ptr + site).to(tl.int32)
    mor = tl.load(mpo_rr_ptr + site).to(tl.int32)
    pql = tl.load(qtt_rl_ptr + site).to(tl.int32)
    pqr = tl.load(qtt_rr_ptr + site).to(tl.int32)

    orl = mol * pql
    orr = mor * pqr
    out_size = orl * D * orr
    mask = tid < out_size

    mpo_off = tl.load(mpo_off_ptr + site)
    qtt_off = tl.load(qtt_off_ptr + site)
    o_off = tl.load(out_off_ptr + site)

    # Decode output index (io, d_out, jo) where io = o*pql + p, jo = r*pqr + q
    jo = tid % orr
    rem = tid // orr
    d_out = rem % D
    io = rem // D

    o = io // pql   # MPO left index
    p = io % pql    # QTT left index
    r = jo // pqr   # MPO right index
    q = jo % pqr    # QTT right index

    # Contract over d_in: sum_{d_in} MPO[o, d_out, d_in, r] * QTT[p, d_in, q]
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for d_in in range(D):  # D=2, fully unrolled
        # MPO layout: (mol, D, D, mor) → idx = o*D*D*mor + d_out*D*mor + d_in*mor + r
        mpo_idx = o * D * D * mor + d_out * D * mor + d_in * mor + r
        mpo_val = tl.load(mpo_data_ptr + mpo_off + mpo_idx, mask=mask, other=0.0)

        # QTT layout: (pql, D, pqr) → idx = p*D*pqr + d_in*pqr + q
        qtt_idx = p * D * pqr + d_in * pqr + q
        qtt_val = tl.load(qtt_data_ptr + qtt_off + qtt_idx, mask=mask, other=0.0)

        acc += mpo_val * qtt_val

    tl.store(out_data_ptr + o_off + tid, acc, mask=mask)


class PackedMPO:
    """MPO stored as contiguous buffer. Cores have shape (rl, D, D, rr)."""

    __slots__ = ("data", "offsets", "rl", "rr", "num_sites", "D", "device", "dtype")

    def __init__(self, cores: List[torch.Tensor]) -> None:
        self.num_sites = len(cores)
        self.D = cores[0].shape[1]
        self.device = cores[0].device
        self.dtype = cores[0].dtype

        rl_list: List[int] = []
        rr_list: List[int] = []
        offsets_list: List[int] = [0]

        for c in cores:
            rl, d1, d2, rr = c.shape
            assert d1 == self.D and d2 == self.D
            rl_list.append(rl)
            rr_list.append(rr)
            offsets_list.append(offsets_list[-1] + c.numel())

        self.rl = torch.tensor(rl_list, dtype=torch.int32, device=self.device)
        self.rr = torch.tensor(rr_list, dtype=torch.int32, device=self.device)
        self.offsets = torch.tensor(offsets_list, dtype=torch.int64, device=self.device)

        total = offsets_list[-1]
        self.data = torch.empty(total, dtype=self.dtype, device=self.device)
        for k, c in enumerate(cores):
            self.data[offsets_list[k] : offsets_list[k + 1]] = c.reshape(-1)

    def unpack(self) -> List[torch.Tensor]:
        offsets = self.offsets.tolist()
        rl = self.rl.tolist()
        rr = self.rr.tolist()
        cores: List[torch.Tensor] = []
        for k in range(self.num_sites):
            flat = self.data[offsets[k] : offsets[k + 1]]
            cores.append(flat.reshape(rl[k], self.D, self.D, rr[k]))
        return cores


def triton_mpo_apply(mpo: PackedMPO, qtt: PackedQTT) -> PackedQTT:
    """
    Apply MPO to QTT: |out⟩ = M|qtt⟩

    Fused Triton kernel — one launch for all L sites. No Python loop.
    Ranks multiply: r_out = r_mpo × r_qtt (before truncation).
    """
    L = qtt.num_sites
    assert mpo.num_sites == L
    D = qtt.D

    mol, mor = mpo.rl.tolist(), mpo.rr.tolist()
    pql, pqr = qtt.rl.tolist(), qtt.rr.tolist()

    o_rl_list = [mol[k] * pql[k] for k in range(L)]
    o_rr_list = [mor[k] * pqr[k] for k in range(L)]
    offsets: List[int] = [0]
    for k in range(L):
        offsets.append(offsets[-1] + o_rl_list[k] * D * o_rr_list[k])

    o_rl = torch.tensor(o_rl_list, dtype=torch.int32, device=qtt.device)
    o_rr = torch.tensor(o_rr_list, dtype=torch.int32, device=qtt.device)
    o_offsets = torch.tensor(offsets, dtype=torch.int64, device=qtt.device)
    o_data = torch.empty(offsets[-1], dtype=qtt.dtype, device=qtt.device)

    max_out = max(o_rl_list[k] * D * o_rr_list[k] for k in range(L))
    BLOCK = 256
    nblk = (max_out + BLOCK - 1) // BLOCK

    _mpo_contract_kernel[(L, nblk)](
        mpo.data, mpo.offsets, mpo.rl, mpo.rr,
        qtt.data, qtt.offsets, qtt.rl, qtt.rr,
        o_data, o_offsets,
        D, BLOCK,
    )

    out = object.__new__(PackedQTT)
    out.data = o_data
    out.offsets = o_offsets
    out.rl = o_rl
    out.rr = o_rr
    out.num_sites = L
    out.D = D
    out.device = qtt.device
    out.dtype = qtt.dtype
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Triton Kernel: Inner Product (Environment Chain Contraction)
# ═══════════════════════════════════════════════════════════════════════════════
# Replaces: pure_qtt_ops.qtt_inner_product for-loop (L iterations)
# Replaces: qtt_batched_ops.qtt_inner for-loop (L iterations)
# Inherently serial across sites, but fuses the per-site einsum.


@triton.jit
def _env_site_kernel(
    env_ptr,
    a_ptr,
    b_ptr,
    out_ptr,
    # Shapes
    ra_in: tl.constexpr,
    ra_out: tl.constexpr,
    rb_in: tl.constexpr,
    rb_out: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Single-site environment contraction:
      out[ra_out, rb_out] = sum_{ra_in, rb_in, d} env[ra_in, rb_in] *
                            A[ra_in, d, ra_out] * B[rb_in, d, rb_out]

    Launched once per site (called L times from Python, but each call is
    a single fused kernel instead of separate einsum dispatches).
    """
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)

    out_size = ra_out * rb_out
    mask = tid < out_size

    # Decode output (ia, ib) from tid
    ib = tid % rb_out
    ia = tid // rb_out

    # Accumulate over (j, k, d) where j=ra_in, k=rb_in
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for j in range(ra_in):
        for k in range(rb_in):
            env_val = tl.load(env_ptr + j * rb_in + k)
            for d in range(D):
                a_val = tl.load(a_ptr + j * D * ra_out + d * ra_out + ia, mask=mask, other=0.0)
                b_val = tl.load(b_ptr + k * D * rb_out + d * rb_out + ib, mask=mask, other=0.0)
                acc += env_val * a_val * b_val

    tl.store(out_ptr + tid, acc, mask=mask)


def triton_qtt_inner(a: PackedQTT, b: PackedQTT) -> float:
    """
    Inner product ⟨a|b⟩ without decompression.

    Uses fused Triton per-site contraction. The chain is serial across sites
    but each site's contraction is a single fused kernel (not 2 separate einsums).

    Falls back to PyTorch einsum for sites with large bond dims where
    the Triton kernel would have too many loop iterations.
    """
    L = a.num_sites
    assert b.num_sites == L
    D = a.D

    a_cores = a.unpack()
    b_cores = b.unpack()

    env = torch.ones(1, 1, device=a.device, dtype=a.dtype)

    TRITON_RANK_THRESHOLD = 64  # Use Triton for small ranks, einsum for large

    for k in range(L):
        ca, cb = a_cores[k], b_cores[k]
        ra_in, _, ra_out = ca.shape
        rb_in, _, rb_out = cb.shape

        if ra_in <= TRITON_RANK_THRESHOLD and rb_in <= TRITON_RANK_THRESHOLD:
            out_size = ra_out * rb_out
            BLOCK = min(256, max(32, out_size))
            # Round BLOCK up to next power of 2
            BLOCK = 1 << (BLOCK - 1).bit_length()
            nblk = (out_size + BLOCK - 1) // BLOCK
            out = torch.empty(ra_out * rb_out, device=a.device, dtype=torch.float32)

            _env_site_kernel[(nblk,)](
                env.contiguous(), ca.contiguous(), cb.contiguous(), out,
                ra_in, ra_out, rb_in, rb_out, D, BLOCK,
            )
            env = out.reshape(ra_out, rb_out)
        else:
            # PyTorch einsum fallback for very large ranks
            tmp = torch.einsum("ij,idk->jdk", env, ca)
            env = torch.einsum("jdk,jdl->kl", tmp, cb)

    return env.squeeze().item()


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Rank Truncation (rSVD, Tolerance-Driven)
# ═══════════════════════════════════════════════════════════════════════════════
# Replaces: pure_qtt_ops.truncate_qtt for-loop (serial SVD sweep)
# Uses rSVD (torch.svd_lowrank) for O(k·m·n) instead of O(m·n·min(m,n))
# Rank is determined by tolerance, NOT by a fixed maximum.


def _rsvd(
    mat: torch.Tensor,
    rank_budget: int = 0,
    oversampling: int = 10,
    n_power_iter: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU-only Randomized SVD — the ONLY SVD path in the engine.

    Uses ``torch.svd_lowrank`` which implements Halko-Martinsson-Tropp:
        1. Random projection (Gaussian sketch)
        2. QR orthogonalization
        3. Power iteration for spectral gap amplification
        4. Small dense SVD on the projected (k × n) matrix

    This NEVER calls cuSOLVER's dgesdd — no error-48, no CPU fallback.
    Stays entirely on GPU in the input dtype.

    Args:
        mat:           Input matrix (m × n), any device/dtype.
        rank_budget:   Max singular values to compute.
                       0 → min(m, n) (full rank, still via rSVD).
        oversampling:  Extra sketch columns (default 10).
        n_power_iter:  Subspace iterations for spectral gap (default 2).

    Returns:
        (U, S, Vh) where U is m×k, S is k, Vh is k×n.
    """
    m, n = mat.shape
    k_max = min(m, n)

    # Sanitize NaN/Inf in-place — rSVD diverges on non-finite inputs
    if not torch.isfinite(mat).all():
        mat = mat.clone()
        mat[~torch.isfinite(mat)] = 0.0

    # Determine sketch rank
    if rank_budget <= 0 or rank_budget >= k_max:
        q = k_max
    else:
        q = rank_budget

    # torch.svd_lowrank requires q + oversampling <= min(m, n)
    q_eff = min(q + oversampling, k_max)
    if q_eff < 1:
        q_eff = 1

    U, S, V = torch.svd_lowrank(mat, q=q_eff, niter=n_power_iter)
    # torch.svd_lowrank returns V (not Vh), convert to Vh = V^T
    Vh = V.T

    # Trim to requested budget
    r = min(q, U.shape[1])
    return U[:, :r], S[:r], Vh[:r, :]


def _adaptive_rank(
    S: torch.Tensor, tol: float, rank_cap: int, abs_tol_sq: float = 0.0,
) -> int:
    """
    Determine truncation rank from singular values.

    Mode 1 (relative, default when abs_tol_sq <= 0):
        sum_{i>r} sigma_i^2  <=  tol^2 * sum_i sigma_i^2

    Mode 2 (absolute, when abs_tol_sq > 0):
        sum_{i>r} sigma_i^2  <=  abs_tol_sq

    Absolute mode is used for dynamics-aware truncation where the reference
    scale is the update norm, not the state norm.  This prevents small-but-
    physical updates from being swallowed by truncation.
    """
    if tol <= 0 and abs_tol_sq <= 0:
        return min(rank_cap, len(S))

    if abs_tol_sq > 0:
        threshold = abs_tol_sq
    else:
        total_sq = (S * S).sum()
        if total_sq < 1e-30:
            return 1
        threshold = tol * tol * total_sq

    # Tail cumulative sum from right
    tail_sq = torch.cumsum(S.flip(0) ** 2, dim=0).flip(0)

    # Smallest r where tail <= threshold
    mask = tail_sq <= threshold
    if mask.any():
        r = mask.long().argmax().item()
        r = max(1, r)
    else:
        r = len(S)

    return min(r, rank_cap)


def adaptive_truncate(
    qtt: PackedQTT,
    tol: float = 1e-6,
    rank_cap: int = 512,
    abs_tol_sq: float = 0.0,
) -> PackedQTT:
    """
    Adaptive-rank QTT truncation using rSVD.

    Algorithm:
        1. Left-to-right QR sweep (orthogonalize)
        2. Right-to-left rSVD sweep (truncate per-bond based on tolerance)

    Rank is NEVER fixed — it's determined by the singular value spectrum
    at each bond. Smooth fields → fast SV decay → low rank.
    Complex fields → slow decay → higher rank (up to rank_cap safety limit).

    Args:
        qtt:        Input PackedQTT state
        tol:        Relative Frobenius tolerance for truncation
        rank_cap:   Safety cap to prevent OOM (not the target rank)
        abs_tol_sq: When > 0, use absolute tolerance mode.  The threshold
                    for discarding singular values is abs_tol_sq directly,
                    ignoring the state norm.  Used for dynamics-aware
                    truncation where the reference scale is ||update||.

    Returns:
        New PackedQTT with adaptively chosen ranks
    """
    cores = qtt.unpack()
    L = len(cores)
    D = qtt.D

    # Clone to avoid modifying input
    cores = [c.clone() for c in cores]

    # Phase 1: Left-to-right QR sweep (no truncation, just orthogonalize)
    for k in range(L - 1):
        c = cores[k]
        r_l, d, r_r = c.shape
        mat = c.reshape(r_l * d, r_r)
        Q, R = torch.linalg.qr(mat)
        cores[k] = Q.reshape(r_l, d, Q.shape[1])
        # Absorb R into next core
        nc = cores[k + 1]
        nr_l, nd, nr_r = nc.shape
        cores[k + 1] = torch.einsum("ij,jdk->idk", R, nc)

    # Phase 2: Right-to-left rSVD sweep (adaptive truncation)
    for k in range(L - 1, 0, -1):
        c = cores[k]
        r_l, d, r_r = c.shape
        mat = c.reshape(r_l, d * r_r)

        # GPU-only rSVD — never calls cuSolver dgesdd, never touches CPU
        U, S, Vh = _rsvd(mat, rank_budget=rank_cap)

        # Adaptive rank selection from singular value spectrum
        r = _adaptive_rank(S, tol, rank_cap, abs_tol_sq=abs_tol_sq)

        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]

        cores[k] = Vh.reshape(r, d, r_r)

        # Absorb U @ diag(S) into left neighbor
        R_left = U * S.unsqueeze(0)
        prev = cores[k - 1]
        cores[k - 1] = torch.einsum("asj,jr->asr", prev, R_left)

    return PackedQTT(cores)


def adaptive_truncate_batched(
    packed_list: List[PackedQTT],
    tol: float = 1e-6,
    rank_cap: int = 512,
    abs_tol_sq: float = 0.0,
) -> List[PackedQTT]:
    """
    Batched adaptive truncation for multiple QTT states.

    Processes all states through QR sweep, then batches SVDs where matrix
    sizes match, then applies adaptive rank selection per bond per state.

    Args:
        abs_tol_sq: When > 0, use absolute tolerance mode (dynamics-aware).
    """
    if not packed_list:
        return packed_list

    B = len(packed_list)
    L = packed_list[0].num_sites
    D = packed_list[0].D

    # Unpack all
    all_cores = [p.unpack() for p in packed_list]
    # Clone
    all_cores = [[c.clone() for c in cores] for cores in all_cores]

    # Phase 1: Left-to-right QR sweep (per field, but can batch QR calls)
    for k in range(L - 1):
        mats = []
        shapes = []
        for i in range(B):
            c = all_cores[i][k]
            r_l, d, r_r = c.shape
            shapes.append((r_l, d, r_r))
            mats.append(c.reshape(r_l * d, r_r))

        # Pad and batch QR
        M_max = max(m.shape[0] for m in mats)
        N_max = max(m.shape[1] for m in mats)
        batch = torch.zeros(B, M_max, N_max, device=mats[0].device, dtype=mats[0].dtype)
        for i, m in enumerate(mats):
            batch[i, : m.shape[0], : m.shape[1]] = m

        Q, R = torch.linalg.qr(batch)

        for i in range(B):
            r_l, d, r_r = shapes[i]
            m = r_l * d
            n = r_r
            r = min(m, n)

            Qi = Q[i, :m, :r].reshape(r_l, d, r)
            Ri = R[i, :r, :n]
            all_cores[i][k] = Qi

            nc = all_cores[i][k + 1]
            nr_l, nd, nr_r = nc.shape
            # Absorb R into next core: (r, n) @ (n, d*nr_r) → (r, d*nr_r)
            nc_mat = nc.reshape(nr_l, nd * nr_r)
            all_cores[i][k + 1] = (Ri @ nc_mat).reshape(r, nd, nr_r)

    # Phase 2: Right-to-left rSVD sweep (adaptive rank per bond per field)
    for k in range(L - 1, 0, -1):
        for i in range(B):
            c = all_cores[i][k]
            r_l, d, r_r = c.shape
            mat = c.reshape(r_l, d * r_r)

            # GPU-only rSVD — no cuSolver, no CPU
            U, S, Vh = _rsvd(mat, rank_budget=rank_cap)

            r = _adaptive_rank(S, tol, rank_cap, abs_tol_sq=abs_tol_sq)
            U = U[:, :r]
            S = S[:r]
            Vh = Vh[:r, :]

            all_cores[i][k] = Vh.reshape(r, d, r_r)
            R_left = U * S.unsqueeze(0)
            prev = all_cores[i][k - 1]
            all_cores[i][k - 1] = torch.einsum("asj,jr->asr", prev, R_left)

    return [PackedQTT(cores) for cores in all_cores]


# ═══════════════════════════════════════════════════════════════════════════════
# Scale-Aware Operations (QTT Core)
# ═══════════════════════════════════════════════════════════════════════════════


def triton_qtt_scale(qtt: PackedQTT, alpha: float) -> PackedQTT:
    """Scale QTT by scalar: |alpha * qtt⟩. Modifies first core only."""
    out = qtt.clone()
    cores = out.unpack()
    cores[0] = cores[0] * alpha
    # Re-pack efficiently (only first core changed)
    offset_0 = out.offsets[0].item()
    offset_1 = out.offsets[1].item()
    out.data[offset_0:offset_1] = cores[0].reshape(-1)
    return out


def triton_qtt_norm(qtt: PackedQTT) -> float:
    """||qtt||_2 = sqrt(<qtt, qtt>)."""
    return math.sqrt(abs(triton_qtt_inner(qtt, qtt)))


# ═══════════════════════════════════════════════════════════════════════════════
# Combined Stencil MPO Builders (Eliminate Catastrophic Cancellation)
# ═══════════════════════════════════════════════════════════════════════════════
# KEY INSIGHT: The Laplacian stencil (f_{+1} + f_{-1} - 2f)/dx² suffers from
# catastrophic cancellation when each shifted field is truncated independently.
#   f_{+1} ≈ f  and  f_{-1} ≈ f,  so the numerator is O(dx²·f'') ≈ 4e-5·||f||.
#   Six intermediate truncations at tol=1e-6 introduce cumulative error
#   ≈ 6 × 1e-6 × (N/2π)² × ||f|| ≈ 0.16·||f|| on a 1024³ grid.
#   The Laplacian result (~3·||f|| for TG) has ~5% error per step → total noise.
#
# FIX: Fuse the stencil into a single combined MPO where the subtraction
# happens at the OPERATOR level (exact), not after data truncation.
#   Derivative   MPO  D_i  = (S+_i − S−_i) / (2·dx)        → rank ≈ 4
#   Laplacian    MPO  Δ    = Σ_i (S+_i + S−_i − 2I) / dx²  → rank ≈ 13
#   One MPO apply + one truncation.  No cancellation.


def identity_mpo_cores(
    n_sites: int,
    D: int = 2,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """Identity MPO: each core is eye(D) reshaped to (1, D, D, 1)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eye = torch.eye(D, device=device, dtype=dtype).reshape(1, D, D, 1)
    return [eye.clone() for _ in range(n_sites)]


def mpo_add_cores(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> List[torch.Tensor]:
    """
    MPO addition: alpha·A + beta·B via block-diagonal core assembly.

    Scaling is applied ONLY at the first core to avoid alpha^L / beta^L.
    Structure mirrors QTT addition but with 4-index cores (rl, D, D, rr):
        First:  (1, D, D, raR+rbR)       — horizontal concat
        Middle: (raL+rbL, D, D, raR+rbR) — block diagonal
        Last:   (raL+rbL, D, D, 1)       — vertical concat

    This is a one-time pre-computation (not in the hot loop).
    """
    L = len(A)
    assert len(B) == L, f"MPO length mismatch: {L} vs {len(B)}"
    out: List[torch.Tensor] = []

    for k in range(L):
        Ak = A[k]
        Bk = B[k]
        raL, D1, D2, raR = Ak.shape
        rbL, _, _, rbR = Bk.shape

        if k == 0:
            # Horizontal concat — scaling applied here only
            C = torch.zeros(1, D1, D2, raR + rbR, device=Ak.device, dtype=Ak.dtype)
            C[:, :, :, :raR] = alpha * Ak
            C[:, :, :, raR:] = beta * Bk
        elif k == L - 1:
            # Vertical concat — no scaling
            C = torch.zeros(raL + rbL, D1, D2, 1, device=Ak.device, dtype=Ak.dtype)
            C[:raL, :, :, :] = Ak
            C[raL:, :, :, :] = Bk
        else:
            # Block diagonal — no scaling
            C = torch.zeros(raL + rbL, D1, D2, raR + rbR, device=Ak.device, dtype=Ak.dtype)
            C[:raL, :, :, :raR] = Ak
            C[raL:, :, :, raR:] = Bk

        out.append(C)
    return out


def build_derivative_mpo(
    shift_plus_axis: List[torch.Tensor],
    shift_minus_axis: List[torch.Tensor],
    dx: float,
) -> List[torch.Tensor]:
    """
    Build combined derivative MPO:  D = (S+ − S−) / (2·dx)

    Single MPO application replaces:
        fp = apply_and_truncate(S+, f)   ← truncation error ε₁
        fm = apply_and_truncate(S−, f)   ← truncation error ε₂
        result = (fp − fm) / (2dx)       ← ε₁ + ε₂ amplified by 1/dx

    With combined MPO, the subtraction is exact at the operator level.
    Rank = rank(S+) + rank(S−) ≈ 4.
    """
    inv_2dx = 1.0 / (2.0 * dx)
    return mpo_add_cores(shift_plus_axis, shift_minus_axis, alpha=inv_2dx, beta=-inv_2dx)


def build_laplacian_mpo_3d(
    shift_plus: List[List[torch.Tensor]],
    shift_minus: List[List[torch.Tensor]],
    dx: float,
) -> List[torch.Tensor]:
    """
    Build full 3D Laplacian as a single combined MPO:

        Δ = Σ_{axis} (S+_axis + S−_axis − 2·I) / dx²
          = (S+_x + S−_x + S+_y + S−_y + S+_z + S−_z − 6I) / dx²

    Rank = 6 × rank(shift) + rank(I) = 6×2 + 1 = 13.

    One apply + one truncation.  Zero catastrophic cancellation.
    """
    inv_dx2 = 1.0 / (dx * dx)
    n_sites = len(shift_plus[0])
    D = shift_plus[0][0].shape[1]
    device = shift_plus[0][0].device
    dtype = shift_plus[0][0].dtype

    # Accumulate: (S+_x + S−_x) / dx²
    combined = mpo_add_cores(shift_plus[0], shift_minus[0], inv_dx2, inv_dx2)

    # + (S+_y + S−_y) / dx²
    sy = mpo_add_cores(shift_plus[1], shift_minus[1], inv_dx2, inv_dx2)
    combined = mpo_add_cores(combined, sy, 1.0, 1.0)

    # + (S+_z + S−_z) / dx²
    sz = mpo_add_cores(shift_plus[2], shift_minus[2], inv_dx2, inv_dx2)
    combined = mpo_add_cores(combined, sz, 1.0, 1.0)

    # − 6I / dx²
    I_cores = identity_mpo_cores(n_sites, D, device, dtype)
    combined = mpo_add_cores(combined, I_cores, 1.0, -6.0 * inv_dx2)

    return combined


def build_combined_stencil_operators(
    shift_plus: List[List[torch.Tensor]],
    shift_minus: List[List[torch.Tensor]],
    dx: float,
) -> Tuple[List[PackedMPO], PackedMPO]:
    """
    Pre-build all combined stencil operators as PackedMPOs.

    Returns:
        (derivative_mpos, laplacian_mpo)
        derivative_mpos: [D_x, D_y, D_z] — 3 PackedMPOs for ∂/∂x, ∂/∂y, ∂/∂z
        laplacian_mpo: single PackedMPO for the full 3D Laplacian Δ
    """
    derivative_mpos: List[PackedMPO] = []
    for axis in range(3):
        d_cores = build_derivative_mpo(shift_plus[axis], shift_minus[axis], dx)
        derivative_mpos.append(PackedMPO(d_cores))

    lap_cores = build_laplacian_mpo_3d(shift_plus, shift_minus, dx)
    laplacian_mpo = PackedMPO(lap_cores)

    return derivative_mpos, laplacian_mpo


# ═══════════════════════════════════════════════════════════════════════════════
# Physics Operations (Native QTT, No Dense, No Cancellation)
# ═══════════════════════════════════════════════════════════════════════════════


def native_mpo_apply_and_truncate(
    mpo_input,
    qtt: PackedQTT,
    tol: float = 1e-6,
    rank_cap: int = 512,
) -> PackedQTT:
    """
    Apply MPO to QTT and adaptively truncate.

    Accepts either a PackedMPO or a list of 4D core tensors.
    """
    if isinstance(mpo_input, PackedMPO):
        packed_mpo = mpo_input
    else:
        packed_mpo = PackedMPO(mpo_input)
    result = triton_mpo_apply(packed_mpo, qtt)
    return adaptive_truncate(result, tol=tol, rank_cap=rank_cap)


def native_laplacian_combined(
    field: PackedQTT,
    laplacian_mpo: PackedMPO,
    tol: float = 1e-6,
    rank_cap: int = 512,
) -> PackedQTT:
    """
    ∇²f via a single combined Laplacian MPO application.

    The stencil subtraction (f₊ + f₋ − 2f)/dx² is baked into the operator.
    One MPO apply + one truncation.  Zero catastrophic cancellation.
    """
    raw = triton_mpo_apply(laplacian_mpo, field)
    return adaptive_truncate(raw, tol=tol, rank_cap=rank_cap)


def native_cross_product(
    u: List[PackedQTT],
    omega: List[PackedQTT],
    tol: float = 1e-6,
    rank_cap: int = 512,
) -> List[PackedQTT]:
    """
    Cross product u × ω in QTT form.

    (u × ω)_x = u_y · ω_z − u_z · ω_y
    (u × ω)_y = u_z · ω_x − u_x · ω_z
    (u × ω)_z = u_x · ω_y − u_y · ω_x
    """
    uy_oz = triton_qtt_hadamard(u[1], omega[2])
    uz_oy = triton_qtt_hadamard(u[2], omega[1])
    uz_ox = triton_qtt_hadamard(u[2], omega[0])
    ux_oz = triton_qtt_hadamard(u[0], omega[2])
    ux_oy = triton_qtt_hadamard(u[0], omega[1])
    uy_ox = triton_qtt_hadamard(u[1], omega[0])

    cx = triton_qtt_add(uy_oz, uz_oy, alpha=1.0, beta=-1.0)
    cy = triton_qtt_add(uz_ox, ux_oz, alpha=1.0, beta=-1.0)
    cz = triton_qtt_add(ux_oy, uy_ox, alpha=1.0, beta=-1.0)

    return adaptive_truncate_batched([cx, cy, cz], tol=tol, rank_cap=rank_cap)


def native_curl_combined(
    field: List[PackedQTT],
    derivative_mpos: List[PackedMPO],
    tol: float = 1e-6,
    rank_cap: int = 512,
) -> List[PackedQTT]:
    """
    Curl ∇ × F using pre-built combined derivative MPOs.

    Each derivative is a SINGLE MPO application (no intermediate truncation
    between S+ and S−).  The subtraction (S+ − S−)/(2dx) is exact in the operator.

    (∇ × F)_x = ∂F_z/∂y − ∂F_y/∂z
    (∇ × F)_y = ∂F_x/∂z − ∂F_z/∂x
    (∇ × F)_z = ∂F_y/∂x − ∂F_x/∂y
    """
    ddx, ddy, ddz = derivative_mpos

    def deriv(f: PackedQTT, mpo: PackedMPO) -> PackedQTT:
        raw = triton_mpo_apply(mpo, f)
        return adaptive_truncate(raw, tol=tol, rank_cap=rank_cap)

    dFz_dy = deriv(field[2], ddy)
    dFy_dz = deriv(field[1], ddz)
    curl_x = triton_qtt_add(dFz_dy, dFy_dz, alpha=1.0, beta=-1.0)

    dFx_dz = deriv(field[0], ddz)
    dFz_dx = deriv(field[2], ddx)
    curl_y = triton_qtt_add(dFx_dz, dFz_dx, alpha=1.0, beta=-1.0)

    dFy_dx = deriv(field[1], ddx)
    dFx_dy = deriv(field[0], ddy)
    curl_z = triton_qtt_add(dFy_dx, dFx_dy, alpha=1.0, beta=-1.0)

    return adaptive_truncate_batched([curl_x, curl_y, curl_z], tol=tol, rank_cap=rank_cap)


def native_compute_rhs(
    u: List[PackedQTT],
    omega: List[PackedQTT],
    nu: float,
    dx: float,
    shift_plus: List[List[torch.Tensor]],
    shift_minus: List[List[torch.Tensor]],
    tol: float = 1e-6,
    rank_cap: int = 512,
    derivative_mpos: Optional[List[PackedMPO]] = None,
    laplacian_mpo: Optional[PackedMPO] = None,
) -> List[PackedQTT]:
    """
    Compute vorticity RHS: dω/dt = ∇×(u×ω) + ν∇²ω

    When derivative_mpos and laplacian_mpo are provided (pre-built combined
    stencil operators), uses them for cancellation-free derivatives.
    Otherwise falls back to per-shift apply-and-truncate (legacy path).
    """
    # Phase 1: Cross product u × ω
    u_cross_omega = native_cross_product(u, omega, tol, rank_cap)

    # Phase 2: Curl of cross product
    if derivative_mpos is not None:
        curl_term = native_curl_combined(u_cross_omega, derivative_mpos, tol, rank_cap)
    else:
        # Legacy fallback (per-shift truncation — has cancellation issues)
        inv_2dx = 1.0 / (2.0 * dx)

        def _legacy_deriv(f: PackedQTT, axis: int) -> PackedQTT:
            fp = native_mpo_apply_and_truncate(shift_plus[axis], f, tol, rank_cap)
            fm = native_mpo_apply_and_truncate(shift_minus[axis], f, tol, rank_cap)
            return triton_qtt_add(fp, fm, alpha=inv_2dx, beta=-inv_2dx)

        dFz_dy = _legacy_deriv(u_cross_omega[2], 1)
        dFy_dz = _legacy_deriv(u_cross_omega[1], 2)
        cx = triton_qtt_add(dFz_dy, dFy_dz, alpha=1.0, beta=-1.0)
        dFx_dz = _legacy_deriv(u_cross_omega[0], 2)
        dFz_dx = _legacy_deriv(u_cross_omega[2], 0)
        cy = triton_qtt_add(dFx_dz, dFz_dx, alpha=1.0, beta=-1.0)
        dFy_dx = _legacy_deriv(u_cross_omega[1], 0)
        dFx_dy = _legacy_deriv(u_cross_omega[0], 1)
        cz = triton_qtt_add(dFy_dx, dFx_dy, alpha=1.0, beta=-1.0)
        curl_term = adaptive_truncate_batched([cx, cy, cz], tol=tol, rank_cap=rank_cap)

    # Phase 3: Viscous Laplacian
    lap_terms: List[PackedQTT] = []
    if laplacian_mpo is not None:
        for comp in range(3):
            lap = native_laplacian_combined(omega[comp], laplacian_mpo, tol, rank_cap)
            lap_terms.append(lap)
    else:
        # Legacy fallback (catastrophic cancellation on fine grids)
        inv_dx2 = 1.0 / (dx * dx)
        for comp in range(3):
            current = None
            for axis in range(3):
                fp = native_mpo_apply_and_truncate(shift_plus[axis], omega[comp], tol, rank_cap)
                fm = native_mpo_apply_and_truncate(shift_minus[axis], omega[comp], tol, rank_cap)
                term = triton_qtt_add(fp, fm, alpha=inv_dx2, beta=inv_dx2)
                if current is None:
                    current = term
                else:
                    current = triton_qtt_add(current, term, alpha=1.0, beta=1.0)
            center = triton_qtt_scale(omega[comp], -6.0 * inv_dx2)
            current = triton_qtt_add(current, center, alpha=1.0, beta=1.0)
            lap_terms.append(adaptive_truncate(current, tol=tol, rank_cap=rank_cap))

    # Phase 4: Combine rhs = curl + ν·laplacian
    rhs: List[PackedQTT] = []
    for comp in range(3):
        combined = triton_qtt_add(curl_term[comp], lap_terms[comp], alpha=1.0, beta=nu)
        rhs.append(combined)

    return adaptive_truncate_batched(rhs, tol=tol, rank_cap=rank_cap)


def native_rk2_step(
    u: List[PackedQTT],
    omega: List[PackedQTT],
    nu: float,
    dt: float,
    dx: float,
    shift_plus: List[List[torch.Tensor]],
    shift_minus: List[List[torch.Tensor]],
    tol: float = 1e-6,
    rank_cap: int = 512,
    derivative_mpos: Optional[List[PackedMPO]] = None,
    laplacian_mpo: Optional[PackedMPO] = None,
    u_update_fn: Optional[Any] = None,
) -> Tuple[List[PackedQTT], List[PackedQTT]]:
    """
    RK2 (Heun) time step — dynamics-aware truncation.

    Stage 1: k1 = f(t, ω)
    Stage 2: ω* = ω + dt·k1,  k2 = f(t+dt, ω*)
    Result:  ω_{n+1} = ω + dt/2 · (k1 + k2)

    DYNAMICS-AWARE TRUNCATION:
        After computing the update δ = dt/2·(k1+k2), the truncation tolerance
        is set relative to ||δ||, not ||ω||.  This prevents small-but-physical
        updates from being swallowed when ||δ|| << ||ω||.

        abs_tol_sq = (tol · ||δ_comp||)²  per component

    U-UPDATE:
        If u_update_fn is provided, u is updated after omega changes:
            u_new = u_update_fn(u, omega_old, omega_new)
        For Taylor-Green validation this rescales u by the omega decay ratio.
        For general flows this would be a Biot-Savart / Poisson solve.
    """

    def _rhs(u_in, omega_in):
        return native_compute_rhs(
            u_in, omega_in, nu, dx, shift_plus, shift_minus, tol, rank_cap,
            derivative_mpos=derivative_mpos, laplacian_mpo=laplacian_mpo,
        )

    # Stage 1: k1 = f(t, ω)
    k1 = _rhs(u, omega)

    # Euler predictor: ω* = ω + dt·k1
    # Dynamics-relative truncation for predictor too
    omega_star: List[PackedQTT] = []
    for comp in range(3):
        k1_norm = triton_qtt_norm(k1[comp])
        pred_abs_tol_sq = (tol * dt * k1_norm) ** 2
        raw = triton_qtt_add(omega[comp], k1[comp], alpha=1.0, beta=dt)
        omega_star.append(adaptive_truncate(raw, tol=tol, rank_cap=rank_cap,
                                            abs_tol_sq=pred_abs_tol_sq))

    # Stage 2: k2 = f(t+dt, ω*)
    k2 = _rhs(u, omega_star)

    # Final: ω_new = ω + dt/2 · (k1 + k2)
    # Dynamics-relative truncation per component
    omega_new: List[PackedQTT] = []
    for comp in range(3):
        k_avg = triton_qtt_add(k1[comp], k2[comp], alpha=1.0, beta=1.0)
        update_norm = triton_qtt_norm(k_avg) * (dt / 2.0)
        update_abs_tol_sq = (tol * update_norm) ** 2
        raw = triton_qtt_add(omega[comp], k_avg, alpha=1.0, beta=dt / 2.0)
        omega_new.append(adaptive_truncate(raw, tol=tol, rank_cap=rank_cap,
                                           abs_tol_sq=update_abs_tol_sq))

    # Update velocity from vorticity
    if u_update_fn is not None:
        u = u_update_fn(u, omega, omega_new)

    return u, omega_new


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics (All Native QTT)
# ═══════════════════════════════════════════════════════════════════════════════


def native_diagnostics(
    u: List[PackedQTT],
    omega: List[PackedQTT],
) -> Dict[str, Any]:
    """
    Compute KE and enstrophy from native PackedQTT fields.

    KE = 0.5 * Σ_i ⟨u_i|u_i⟩
    Enstrophy = 0.5 * Σ_i ⟨ω_i|ω_i⟩
    """
    ke = sum(triton_qtt_inner(v, v) for v in u)
    enstrophy = sum(triton_qtt_inner(w, w) for w in omega)

    max_rank_u = max(p.max_rank for p in u)
    max_rank_omega = max(p.max_rank for p in omega)

    return {
        "kinetic_energy": 0.5 * ke,
        "enstrophy": 0.5 * enstrophy,
        "max_rank_u": max_rank_u,
        "max_rank_omega": max_rank_omega,
        "total_params_u": sum(p.total_params for p in u),
        "total_params_omega": sum(p.total_params for p in omega),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


__all__ = [
    # Storage
    "PackedQTT",
    "PackedMPO",
    # Core ops (Triton-fused)
    "triton_qtt_add",
    "triton_qtt_hadamard",
    "triton_mpo_apply",
    "triton_qtt_inner",
    "triton_qtt_scale",
    "triton_qtt_norm",
    # Adaptive truncation (rSVD)
    "adaptive_truncate",
    "adaptive_truncate_batched",
    # Combined stencil operators
    "identity_mpo_cores",
    "mpo_add_cores",
    "build_derivative_mpo",
    "build_laplacian_mpo_3d",
    "build_combined_stencil_operators",
    # Physics (native QTT)
    "native_mpo_apply_and_truncate",
    "native_laplacian_combined",
    "native_cross_product",
    "native_curl_combined",
    "native_compute_rhs",
    "native_rk2_step",
    "native_diagnostics",
]
