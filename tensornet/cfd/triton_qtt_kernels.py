"""
Triton Kernels for QTT Operations
=================================

L2 cache optimized kernels for QTT core contractions.

Key optimizations:
1. Fused operations (no intermediate tensors)
2. L2 cache blocking for small matrices
3. Coalesced memory access
4. Register-level accumulation

Kernels:
- qtt_mpo_apply: Apply MPO to QTT state
- qtt_core_add: Fused core addition and truncation setup
- qtt_svd_step: Single SVD step with truncation
- qtt_hadamard: Fused Hadamard product

Author: HyperTensor Team
Date: 2025
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import List, Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    def _next_pow2(x: int) -> int:
        """Round up to next power of 2 (Triton requires this)."""
        return 1 << (x - 1).bit_length() if x > 0 else 1

    # ═══════════════════════════════════════════════════════════════════════════════════
    # MPO APPLICATION KERNEL
    # ═══════════════════════════════════════════════════════════════════════════════════

    @triton.jit
    def _mpo_apply_kernel(
        # State core: (r_s_left, d_s, r_s_right)
        state_ptr,
        # MPO core: (r_m_left, d_s, d_out, r_m_right)  
        mpo_ptr,
        # Output: (r_s_left * r_m_left, d_out, r_s_right * r_m_right)
        out_ptr,
        # Dimensions
        r_s_l, d_s, r_s_r,
        r_m_l, d_out, r_m_r,
        # Strides for state
        s_str_l, s_str_d, s_str_r,
        # Strides for MPO
        m_str_l, m_str_ds, m_str_do, m_str_r,
        # Strides for output
        o_str_l, o_str_d, o_str_r,
        # Block sizes
        BLOCK_L: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """
        Fused MPO application to QTT core.
        
        out[i*r_m_l + a, j, k*r_m_r + b] = sum_s state[i,s,k] * mpo[a,s,j,b]
        
        Optimized for L2 cache: typical r < 128, d = 2.
        """
        # Block indices
        pid_l = tl.program_id(0)  # Combined (i, a)
        pid_r = tl.program_id(1)  # Combined (k, b)
        
        out_r_l = r_s_l * r_m_l
        out_r_r = r_s_r * r_m_r
        
        # Decode combined indices
        l_idx = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        r_idx = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        
        mask_l = l_idx < out_r_l
        mask_r = r_idx < out_r_r
        
        # Decompose into state/mpo indices
        i_idx = l_idx // r_m_l
        a_idx = l_idx % r_m_l
        k_idx = r_idx // r_m_r
        b_idx = r_idx % r_m_r
        
        # Loop over output physical dimension
        for j in range(d_out):
            acc = tl.zeros((BLOCK_L, BLOCK_R), dtype=tl.float32)
            
            # Sum over input physical dimension
            for s in range(d_s):
                # Load state[i, s, k]
                s_offset = i_idx[:, None] * s_str_l + s * s_str_d + k_idx[None, :] * s_str_r
                s_val = tl.load(state_ptr + s_offset, mask=mask_l[:, None] & mask_r[None, :], other=0.0)
                
                # Load mpo[a, s, j, b]
                m_offset = a_idx[:, None] * m_str_l + s * m_str_ds + j * m_str_do + b_idx[None, :] * m_str_r
                m_val = tl.load(mpo_ptr + m_offset, mask=mask_l[:, None] & mask_r[None, :], other=0.0)
                
                acc += s_val * m_val
            
            # Store output[l_idx, j, r_idx]
            o_offset = l_idx[:, None] * o_str_l + j * o_str_d + r_idx[None, :] * o_str_r
            tl.store(out_ptr + o_offset, acc, mask=mask_l[:, None] & mask_r[None, :])


    # Size threshold for Triton vs einsum (einsum faster for small tensors)
    _TRITON_SIZE_THRESHOLD = 4096  # Output elements

    def triton_mpo_apply(
        state_core: Tensor,  # (r_s_l, d_s, r_s_r)
        mpo_core: Tensor,    # (r_m_l, d_s, d_out, r_m_r)
    ) -> Tensor:
        """
        Apply MPO core to state core.
        
        Uses einsum for small tensors, Triton for large.
        Returns: (r_s_l * r_m_l, d_out, r_s_r * r_m_r)
        """
        r_s_l, d_s, r_s_r = state_core.shape
        r_m_l, d_s2, d_out, r_m_r = mpo_core.shape
        assert d_s == d_s2
        
        out_r_l = r_s_l * r_m_l
        out_r_r = r_s_r * r_m_r
        out_size = out_r_l * d_out * out_r_r
        
        # Use einsum for small tensors (faster due to lower overhead)
        if out_size < _TRITON_SIZE_THRESHOLD:
            out = torch.einsum('isk,asjb->iajkb', state_core, mpo_core)
            return out.reshape(out_r_l, d_out, out_r_r)
        
        out = torch.empty(out_r_l, d_out, out_r_r, device=state_core.device, dtype=state_core.dtype)
        
        # Triton requires power-of-2 block sizes
        BLOCK_L = _next_pow2(min(32, out_r_l))
        BLOCK_R = _next_pow2(min(32, out_r_r))
        grid = (triton.cdiv(out_r_l, BLOCK_L), triton.cdiv(out_r_r, BLOCK_R))
        
        _mpo_apply_kernel[grid](
            state_core, mpo_core, out,
            r_s_l, d_s, r_s_r,
            r_m_l, d_out, r_m_r,
            state_core.stride(0), state_core.stride(1), state_core.stride(2),
            mpo_core.stride(0), mpo_core.stride(1), mpo_core.stride(2), mpo_core.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_L=BLOCK_L, BLOCK_R=BLOCK_R,
        )
        
        return out


    # ═══════════════════════════════════════════════════════════════════════════════════
    # FUSED HADAMARD PRODUCT KERNEL
    # ═══════════════════════════════════════════════════════════════════════════════════

    @triton.jit
    def _hadamard_core_kernel(
        # Input cores
        a_ptr, b_ptr,
        # Output core (Kronecker product)
        out_ptr,
        # Dimensions
        ra_l, ra_r, rb_l, rb_r, d,
        # Strides
        a_str_l, a_str_d, a_str_r,
        b_str_l, b_str_d, b_str_r,
        o_str_l, o_str_d, o_str_r,
        # Block size
        BLOCK: tl.constexpr,
    ):
        """
        Fused Hadamard core: out[i*rb_l+j, s, k*rb_r+l] = a[i,s,k] * b[j,s,l]
        """
        pid = tl.program_id(0)
        
        out_l = ra_l * rb_l
        out_r = ra_r * rb_r
        
        # Linear index over output left dimension
        l_base = pid * BLOCK
        l_idx = l_base + tl.arange(0, BLOCK)
        mask = l_idx < out_l
        
        # Decompose
        i_idx = l_idx // rb_l
        j_idx = l_idx % rb_l
        
        for s in range(d):
            for k in range(ra_r):
                for l in range(rb_r):
                    # Load a[i, s, k]
                    a_offset = i_idx * a_str_l + s * a_str_d + k * a_str_r
                    a_val = tl.load(a_ptr + a_offset, mask=mask, other=0.0)
                    
                    # Load b[j, s, l]
                    b_offset = j_idx * b_str_l + s * b_str_d + l * b_str_r
                    b_val = tl.load(b_ptr + b_offset, mask=mask, other=0.0)
                    
                    # Store out[l_idx, s, k*rb_r + l]
                    out_r_idx = k * rb_r + l
                    o_offset = l_idx * o_str_l + s * o_str_d + out_r_idx * o_str_r
                    tl.store(out_ptr + o_offset, a_val * b_val, mask=mask)


    def triton_hadamard_core(
        a: Tensor,  # (ra_l, d, ra_r)
        b: Tensor,  # (rb_l, d, rb_r)
    ) -> Tensor:
        """
        Triton-accelerated Hadamard core product.
        
        Returns: (ra_l * rb_l, d, ra_r * rb_r)
        """
        ra_l, d, ra_r = a.shape
        rb_l, d2, rb_r = b.shape
        assert d == d2
        
        out_l = ra_l * rb_l
        out_r = ra_r * rb_r
        
        out = torch.empty(out_l, d, out_r, device=a.device, dtype=a.dtype)
        
        # Triton requires power-of-2 block sizes
        BLOCK = _next_pow2(min(128, out_l))
        grid = (triton.cdiv(out_l, BLOCK),)
        
        _hadamard_core_kernel[grid](
            a, b, out,
            ra_l, ra_r, rb_l, rb_r, d,
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK=BLOCK,
        )
        
        return out


    # ═══════════════════════════════════════════════════════════════════════════════════
    # FUSED INNER PRODUCT KERNEL
    # ═══════════════════════════════════════════════════════════════════════════════════

    @triton.jit
    def _inner_contract_kernel(
        # Environment: (r_a, r_b)
        env_ptr,
        # Cores: (r_a_l, d, r_a_r) and (r_b_l, d, r_b_r)
        a_ptr, b_ptr,
        # Output: (r_a_r, r_b_r)
        out_ptr,
        # Dimensions
        r_a_l, r_a_r, r_b_l, r_b_r, d,
        # Strides
        e_str_a, e_str_b,
        a_str_l, a_str_d, a_str_r,
        b_str_l, b_str_d, b_str_r,
        o_str_a, o_str_b,
        # Block sizes
        BLOCK_A: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        """
        Fused inner product contraction step.
        
        out[k, l] = sum_{i,j,s} env[i,j] * a[i,s,k] * b[j,s,l]
        """
        pid_a = tl.program_id(0)
        pid_b = tl.program_id(1)
        
        k_idx = pid_a * BLOCK_A + tl.arange(0, BLOCK_A)
        l_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        
        mask_k = k_idx < r_a_r
        mask_l = l_idx < r_b_r
        
        acc = tl.zeros((BLOCK_A, BLOCK_B), dtype=tl.float32)
        
        for i in range(r_a_l):
            for j in range(r_b_l):
                # Load env[i, j]
                env_val = tl.load(env_ptr + i * e_str_a + j * e_str_b)
                
                for s in range(d):
                    # Load a[i, s, k]
                    a_offset = i * a_str_l + s * a_str_d + k_idx * a_str_r
                    a_val = tl.load(a_ptr + a_offset[:, None], mask=mask_k[:, None], other=0.0)
                    
                    # Load b[j, s, l]
                    b_offset = j * b_str_l + s * b_str_d + l_idx * b_str_r
                    b_val = tl.load(b_ptr + b_offset[None, :], mask=mask_l[None, :], other=0.0)
                    
                    acc += env_val * a_val * b_val
        
        # Store result
        o_offset = k_idx[:, None] * o_str_a + l_idx[None, :] * o_str_b
        tl.store(out_ptr + o_offset, acc, mask=mask_k[:, None] & mask_l[None, :])


    def triton_inner_step(
        env: Tensor,    # (r_a_l, r_b_l)
        a: Tensor,      # (r_a_l, d, r_a_r)
        b: Tensor,      # (r_b_l, d, r_b_r)
    ) -> Tensor:
        """
        Single step of inner product contraction.
        
        Returns: (r_a_r, r_b_r)
        """
        r_a_l, d, r_a_r = a.shape
        r_b_l, d2, r_b_r = b.shape
        assert d == d2
        assert env.shape == (r_a_l, r_b_l)
        
        out = torch.empty(r_a_r, r_b_r, device=a.device, dtype=a.dtype)
        
        # Triton requires power-of-2 block sizes
        BLOCK_A = _next_pow2(min(32, r_a_r))
        BLOCK_B = _next_pow2(min(32, r_b_r))
        grid = (triton.cdiv(r_a_r, BLOCK_A), triton.cdiv(r_b_r, BLOCK_B))
        
        _inner_contract_kernel[grid](
            env, a, b, out,
            r_a_l, r_a_r, r_b_l, r_b_r, d,
            env.stride(0), env.stride(1),
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            out.stride(0), out.stride(1),
            BLOCK_A=BLOCK_A, BLOCK_B=BLOCK_B,
        )
        
        return out


    # ═══════════════════════════════════════════════════════════════════════════════════
    # BATCHED SHIFT KERNEL
    # ═══════════════════════════════════════════════════════════════════════════════════

    @triton.jit
    def _shift_mpo_batch_kernel(
        # Input cores (batch, L, max_r, 2, max_r)
        in_ptr,
        # Shift MPO (L, 2, 2, 2)  - simple increment/decrement
        shift_ptr,
        # Output (batch, L, max_r*2, 2, max_r*2)
        out_ptr,
        # Dimensions
        batch, L, max_r,
        # Strides for input
        in_str_b, in_str_l, in_str_rl, in_str_d, in_str_rr,
        # Strides for MPO
        s_str_l, s_str_di, s_str_do, s_str_r,
        # Strides for output
        o_str_b, o_str_l, o_str_rl, o_str_d, o_str_rr,
        # Block size
        BLOCK_B: tl.constexpr,
    ):
        """
        Batched shift MPO application across all sites.
        """
        pid = tl.program_id(0)
        site = tl.program_id(1)
        
        b_base = pid * BLOCK_B
        b_idx = b_base + tl.arange(0, BLOCK_B)
        mask_b = b_idx < batch
        
        # For each output element
        for rl_out in range(max_r * 2):
            for d_out in range(2):
                for rr_out in range(max_r * 2):
                    acc = tl.zeros((BLOCK_B,), dtype=tl.float32)
                    
                    # Sum over input dimensions
                    for rl_in in range(max_r):
                        for d_in in range(2):
                            for rr_in in range(max_r):
                                # Load input[b, site, rl_in, d_in, rr_in]
                                i_off = b_idx * in_str_b + site * in_str_l + rl_in * in_str_rl + d_in * in_str_d + rr_in * in_str_rr
                                i_val = tl.load(in_ptr + i_off, mask=mask_b, other=0.0)
                                
                                # Load shift[site, d_in, d_out, ?]
                                # Simplified: assume shift MPO has rank 1-2
                                s_off = site * s_str_l + d_in * s_str_di + d_out * s_str_do
                                s_val = tl.load(shift_ptr + s_off)
                                
                                # Accumulate where indices match
                                acc += i_val * s_val
                    
                    # Store
                    o_off = b_idx * o_str_b + site * o_str_l + rl_out * o_str_rl + d_out * o_str_d + rr_out * o_str_rr
                    tl.store(out_ptr + o_off, acc, mask=mask_b)


# Fallback implementations for non-Triton systems
else:
    def triton_mpo_apply(state_core: Tensor, mpo_core: Tensor) -> Tensor:
        """Fallback: einsum MPO application."""
        return torch.einsum('isk,asjo->iajok', state_core, mpo_core).reshape(
            state_core.shape[0] * mpo_core.shape[0],
            mpo_core.shape[2],
            state_core.shape[2] * mpo_core.shape[3],
        )
    
    def triton_hadamard_core(a: Tensor, b: Tensor) -> Tensor:
        """Fallback: einsum Hadamard."""
        prod = torch.einsum('isk,jsl->ijskl', a, b)
        return prod.reshape(a.shape[0] * b.shape[0], a.shape[1], a.shape[2] * b.shape[2])
    
    def triton_inner_step(env: Tensor, a: Tensor, b: Tensor) -> Tensor:
        """Fallback: einsum inner product step."""
        return torch.einsum('ij,isk,jsl->kl', env, a, b)


# ═══════════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    'TRITON_AVAILABLE',
    'triton_mpo_apply',
    'triton_hadamard_core', 
    'triton_inner_step',
]
