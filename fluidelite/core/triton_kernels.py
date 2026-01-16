"""
Triton Fused Kernels for FluidElite
===================================

Eliminates Python loop overhead by fusing MPS/MPO operations into single GPU kernels.
This is the Phase 2 optimization that enables real GPU utilization.

Key Kernels:
1. fused_mpo_contract: Contracts all L sites of MPO with MPS in one kernel
2. fused_direct_sum: Block-diagonal addition of two MPS in one kernel  

Constitutional Compliance:
    - Article V.5.1: All public functions documented
    - Article VII.7.2: Definition of Done = USER-OBSERVABLE BEHAVIOR works
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _mpo_contract_tiled_kernel(
    # Pointers
    mps_ptr, mpo_ptr, out_ptr,
    # Dimensions  
    L, chi_in, d, D, chi_out,
    # Strides for MPS (L, chi, d, chi)
    mps_stride_l, mps_stride_cl, mps_stride_d, mps_stride_cr,
    # Strides for MPO (L, D, d, d, D)
    mpo_stride_l, mpo_stride_dl, mpo_stride_do, mpo_stride_di, mpo_stride_dr,
    # Strides for output (L, chi*D, d, chi*D)
    out_stride_l, out_stride_cl, out_stride_d, out_stride_cr,
    # Block sizes
    BLOCK_CL: tl.constexpr,
    BLOCK_CR: tl.constexpr,
):
    """
    Tiled MPO-MPS contraction kernel.
    
    Each program computes a BLOCK_CL x BLOCK_CR tile of output.
    Grid: (L, ceil(chi_out/BLOCK_CL), ceil(chi_out/BLOCK_CR))
    
    Contraction: out[l, a*chi+c, o, b*chi+e] = sum_i mpo[l,a,o,i,b] * mps[l,c,i,e]
    """
    # Program IDs
    pid_l = tl.program_id(0)   # Which site
    pid_cl = tl.program_id(1)  # Which output left-chi tile
    pid_cr = tl.program_id(2)  # Which output right-chi tile
    
    # Tile start indices
    cl_start = pid_cl * BLOCK_CL
    cr_start = pid_cr * BLOCK_CR
    
    # Offsets within tile
    cl_offs = cl_start + tl.arange(0, BLOCK_CL)  # [BLOCK_CL]
    cr_offs = cr_start + tl.arange(0, BLOCK_CR)  # [BLOCK_CR]
    
    # Masks for boundary
    cl_mask = cl_offs < chi_out
    cr_mask = cr_offs < chi_out
    
    # Decompose output indices: cl = a * chi_in + c, cr = b * chi_in + e
    a = cl_offs // chi_in  # [BLOCK_CL]
    c = cl_offs % chi_in   # [BLOCK_CL]
    b = cr_offs // chi_in  # [BLOCK_CR]
    e = cr_offs % chi_in   # [BLOCK_CR]
    
    # Loop over output physical dimension o
    for o in range(d):
        # Accumulator for this tile: [BLOCK_CL, BLOCK_CR]
        acc = tl.zeros((BLOCK_CL, BLOCK_CR), dtype=tl.float32)
        
        # Contract over input physical dimension i
        for i in range(d):
            # Load MPS[l, c, i, e] -> need [BLOCK_CL] x [BLOCK_CR] values
            # But c depends on cl, e depends on cr
            # mps[pid_l, c[:], i, e[:]] shape: [BLOCK_CL, BLOCK_CR]
            
            # Compute offsets for all (c, e) pairs
            # mps_offs[cl_idx, cr_idx] = pid_l * stride_l + c[cl_idx] * stride_cl + i * stride_d + e[cr_idx] * stride_cr
            mps_offs = (pid_l * mps_stride_l + 
                       c[:, None] * mps_stride_cl + 
                       i * mps_stride_d + 
                       e[None, :] * mps_stride_cr)
            
            # Load MPS values with mask
            mps_vals = tl.load(mps_ptr + mps_offs, 
                              mask=cl_mask[:, None] & cr_mask[None, :], 
                              other=0.0)
            
            # Load MPO[l, a, o, i, b] -> need [BLOCK_CL] x [BLOCK_CR] values
            # mpo[pid_l, a[:], o, i, b[:]]
            mpo_offs = (pid_l * mpo_stride_l +
                       a[:, None] * mpo_stride_dl +
                       o * mpo_stride_do +
                       i * mpo_stride_di +
                       b[None, :] * mpo_stride_dr)
            
            mpo_vals = tl.load(mpo_ptr + mpo_offs,
                              mask=cl_mask[:, None] & cr_mask[None, :],
                              other=0.0)
            
            # Accumulate: out = sum_i mpo * mps
            acc += mpo_vals * mps_vals
        
        # Store output tile
        out_offs = (pid_l * out_stride_l +
                   cl_offs[:, None] * out_stride_cl +
                   o * out_stride_d +
                   cr_offs[None, :] * out_stride_cr)
        
        tl.store(out_ptr + out_offs, acc,
                mask=cl_mask[:, None] & cr_mask[None, :])


def triton_mpo_contract(mps: Tensor, mpo: Tensor) -> Tensor:
    """
    Fused MPO-MPS contraction using Triton.
    
    Uses tiled kernel - each program computes a 64x64 tile.
    
    Args:
        mps: MPS cores stacked (L, chi, d, chi)
        mpo: MPO cores stacked (L, D, d_out, d_in, D)
        
    Returns:
        Contracted MPS (L, chi*D, d_out, chi*D)
    """
    L, chi, d, _ = mps.shape
    _, D, d_out, d_in, _ = mpo.shape
    
    assert d == d_in, f"Physical dimension mismatch: MPS has d={d}, MPO expects d_in={d_in}"
    
    chi_out = chi * D
    
    # Allocate output
    out = torch.zeros(L, chi_out, d_out, chi_out, device=mps.device, dtype=mps.dtype)
    
    # Block sizes - 64x64 tiles
    BLOCK_CL = 64
    BLOCK_CR = 64
    
    # Grid: one program per tile
    # Total programs: L * ceil(chi_out/64) * ceil(chi_out/64)
    # For chi_out=4096: 16 * 64 * 64 = 65,536 programs (not 268M!)
    grid = (
        L,
        (chi_out + BLOCK_CL - 1) // BLOCK_CL,
        (chi_out + BLOCK_CR - 1) // BLOCK_CR,
    )
    
    _mpo_contract_tiled_kernel[grid](
        mps, mpo, out,
        L, chi, d, D, chi_out,
        mps.stride(0), mps.stride(1), mps.stride(2), mps.stride(3),
        mpo.stride(0), mpo.stride(1), mpo.stride(2), mpo.stride(3), mpo.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_CL=BLOCK_CL,
        BLOCK_CR=BLOCK_CR,
    )
    
    return out


@triton.jit
def _direct_sum_kernel(
    # Pointers
    a_ptr, b_ptr, out_ptr,
    # Dimensions
    L, chi_a, chi_b, d, chi_out,
    # Strides for A (L, chi_a, d, chi_a)
    a_stride_l, a_stride_cl, a_stride_d, a_stride_cr,
    # Strides for B (L, chi_b, d, chi_b)  
    b_stride_l, b_stride_cl, b_stride_d, b_stride_cr,
    # Strides for output (L, chi_out, d, chi_out)
    out_stride_l, out_stride_cl, out_stride_d, out_stride_cr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused direct sum kernel for MPS addition.
    
    Copies A to top-left block, B to bottom-right block.
    Each program handles BLOCK_SIZE elements.
    """
    pid = tl.program_id(0)
    
    # Process A tensor (copy to top-left of output)
    numel_a = L * chi_a * d * chi_a
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_a = offs < numel_a
    
    # Decompose linear index for A: offs -> (l, cl, d_idx, cr)
    # Layout: [l, cl, d_idx, cr] in row-major
    cr_a = offs % chi_a
    temp = offs // chi_a
    d_idx_a = temp % d  
    temp = temp // d
    cl_a = temp % chi_a
    l_a = temp // chi_a
    
    # Load from A
    a_off = l_a * a_stride_l + cl_a * a_stride_cl + d_idx_a * a_stride_d + cr_a * a_stride_cr
    a_val = tl.load(a_ptr + a_off, mask=mask_a, other=0.0)
    
    # Store to output (same position, top-left block)
    out_off_a = l_a * out_stride_l + cl_a * out_stride_cl + d_idx_a * out_stride_d + cr_a * out_stride_cr
    tl.store(out_ptr + out_off_a, a_val, mask=mask_a)
    
    # Process B tensor (copy to bottom-right of output)
    numel_b = L * chi_b * d * chi_b
    mask_b = offs < numel_b
    
    # Decompose linear index for B
    cr_b = offs % chi_b
    temp = offs // chi_b
    d_idx_b = temp % d
    temp = temp // d
    cl_b = temp % chi_b
    l_b = temp // chi_b
    
    # Load from B
    b_off = l_b * b_stride_l + cl_b * b_stride_cl + d_idx_b * b_stride_d + cr_b * b_stride_cr
    b_val = tl.load(b_ptr + b_off, mask=mask_b, other=0.0)
    
    # Store to output (offset by chi_a, bottom-right block)
    out_off_b = (l_b * out_stride_l + 
                (cl_b + chi_a) * out_stride_cl + 
                d_idx_b * out_stride_d + 
                (cr_b + chi_a) * out_stride_cr)
    tl.store(out_ptr + out_off_b, b_val, mask=mask_b)


def triton_direct_sum(a: Tensor, b: Tensor) -> Tensor:
    """
    Fused direct sum of two MPS using Triton.
    
    Args:
        a: First MPS (L, chi_a, d, chi_a)
        b: Second MPS (L, chi_b, d, chi_b)
        
    Returns:
        Block-diagonal sum (L, chi_a+chi_b, d, chi_a+chi_b)
    """
    L, chi_a, d, _ = a.shape
    _, chi_b, _, _ = b.shape
    
    chi_out = chi_a + chi_b
    
    # Allocate output (zeros for off-diagonal blocks)
    out = torch.zeros(L, chi_out, d, chi_out, device=a.device, dtype=a.dtype)
    
    # Total elements to process
    numel = max(L * chi_a * d * chi_a, L * chi_b * d * chi_b)
    BLOCK_SIZE = 1024
    grid = ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    _direct_sum_kernel[grid](
        a, b, out,
        L, chi_a, chi_b, d, chi_out,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class FusedMPOApply(torch.autograd.Function):
    """
    Autograd wrapper for fused MPO application.
    
    Forward: Uses Triton kernel for speed
    Backward: Falls back to PyTorch for correctness
    """
    @staticmethod
    def forward(ctx, mps: Tensor, mpo: Tensor) -> Tensor:
        ctx.save_for_backward(mps, mpo)
        return triton_mpo_contract(mps, mpo)
    
    @staticmethod
    def backward(ctx, grad_out: Tensor):
        mps, mpo = ctx.saved_tensors
        
        # Gradient w.r.t. MPS: contract grad_out with MPO transposed
        # grad_mps[l,c,i,d] = sum_{a,o,b} grad_out[l,a*chi+c,o,b*chi+d] * mpo[l,a,o,i,b]
        L, chi, d_in, _ = mps.shape
        _, D, d_out, _, _ = mpo.shape
        chi_out = chi * D
        
        # Reshape grad_out for contraction
        grad_out_r = grad_out.reshape(L, D, chi, d_out, D, chi)
        # grad_out_r: (L, D, chi, d_out, D, chi) indices: l, a, c, o, b, d
        # mpo: (L, D, d_out, d_in, D) indices: l, a, o, i, b
        # Contract over a, o, b to get (L, chi, d_in, chi)
        grad_mps = torch.einsum('lacobe,laoib->lcid', grad_out_r.float(), mpo.float())
        
        # Gradient w.r.t. MPO: contract grad_out with MPS
        # grad_mpo[l,a,o,i,b] = sum_{c,d} grad_out[l,a*chi+c,o,b*chi+d] * mps[l,c,i,d]
        grad_mpo = torch.einsum('lacobe,lcid->laoib', grad_out_r.float(), mps.float())
        
        return grad_mps.to(mps.dtype), grad_mpo.to(mpo.dtype)


def fused_mpo_apply(mps: Tensor, mpo: Tensor) -> Tensor:
    """
    Apply MPO to MPS using fused Triton kernel with autograd support.
    
    Args:
        mps: Stacked MPS cores (L, chi, d, chi)
        mpo: Stacked MPO cores (L, D, d, d, D)
        
    Returns:
        Result MPS cores (L, chi*D, d, chi*D)
    """
    return FusedMPOApply.apply(mps, mpo)


class FusedDirectSum(torch.autograd.Function):
    """
    Autograd wrapper for fused direct sum.
    """
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.chi_a = a.shape[1]
        ctx.chi_b = b.shape[1]
        return triton_direct_sum(a, b)
    
    @staticmethod
    def backward(ctx, grad_out: Tensor):
        chi_a = ctx.chi_a
        chi_b = ctx.chi_b
        
        # Extract gradients from block diagonal
        grad_a = grad_out[:, :chi_a, :, :chi_a].contiguous()
        grad_b = grad_out[:, chi_a:, :, chi_a:].contiguous()
        
        return grad_a, grad_b


def fused_direct_sum(a: Tensor, b: Tensor) -> Tensor:
    """
    Add two MPS via block-diagonal direct sum using fused Triton kernel.
    
    Args:
        a: First MPS (L, chi_a, d, chi_a)
        b: Second MPS (L, chi_b, d, chi_b)
        
    Returns:
        Sum MPS (L, chi_a+chi_b, d, chi_a+chi_b)
    """
    return FusedDirectSum.apply(a, b)
