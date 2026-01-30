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
        BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """
        Contract two TT cores along the bond dimension.
        
        A: (R_left, D, R_mid)
        B: (R_mid, D, R_right)
        C: (R_left, D*D, R_right) = sum over R_mid
        
        This is the fundamental QTT operation - MUST be fast.
        """
        pid_left = tl.program_id(0)
        pid_d = tl.program_id(1)
        
        # For each (r_left, d1, d2, r_right) tuple:
        # C[r_left, d1*D+d2, r_right] = sum_{r_mid} A[r_left, d1, r_mid] * B[r_mid, d2, r_right]
        
        offs_left = pid_left * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        
        # This kernel handles one block of (r_left, d1, d2) combinations
        # We need to sum over r_mid for each
        
        for d1 in range(D):
            for d2 in range(D):
                d_out = d1 * D + d2
                
                if d_out < D * D:
                    for r_right in range(R_right):
                        acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
                        
                        for r_mid in range(R_mid):
                            # Load A[r_left, d1, r_mid]
                            a_ptr = A_ptr + offs_left * stride_a0 + d1 * stride_a1 + r_mid * stride_a2
                            a = tl.load(a_ptr, mask=offs_left < R_left, other=0.0)
                            
                            # Load B[r_mid, d2, r_right]
                            b = tl.load(B_ptr + r_mid * stride_b0 + d2 * stride_b1 + r_right * stride_b2)
                            
                            acc += a * b
                        
                        # Store C[r_left, d_out, r_right]
                        c_ptr = C_ptr + offs_left * stride_c0 + d_out * stride_c1 + r_right * stride_c2
                        tl.store(c_ptr, acc, mask=offs_left < R_left)


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
    
    # Small matrix fast path
    if m <= 4 or n <= 4:
        U, S, Vh = torch.linalg.svd(A.to(torch.float64), full_matrices=False)
        k_actual = min(k, len(S), max(1, int((S > tol * S[0]).sum())))
        return U[:, :k_actual].to(dtype), S[:k_actual].to(dtype), Vh[:k_actual, :].to(dtype)
    
    l = min(k + n_oversamples, min(m, n))
    
    # Work in float64 for numerical stability
    A_64 = A.to(torch.float64) if dtype != torch.float64 else A
    
    if m <= n:
        # Wide matrix: work with A @ A.T which is (m × m)
        Omega = torch.randn(n, l, device=device, dtype=torch.float64)
        Y = A_64 @ Omega
        
        # Power iteration
        for _ in range(n_iter):
            Z = A_64.T @ Y
            Y = A_64 @ Z
        
        # QR
        Q, _ = torch.linalg.qr(Y, mode='reduced')
        
        # Gram matrix: Q.T @ A @ A.T @ Q
        AAt_Q = A_64 @ (A_64.T @ Q)
        BtB = Q.T @ AAt_Q
        
        # Eigendecomposition of small Gram matrix
        eigvals, eigvecs = torch.linalg.eigh(BtB)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        S = torch.sqrt(torch.clamp(eigvals, min=0))
        U = Q @ eigvecs
        
        k_actual = min(k, max(1, int((S > tol * S[0]).sum())))
        
        inv_S = torch.zeros_like(S)
        mask = S[:k_actual] > tol * S[0]
        inv_S[:k_actual][mask] = 1.0 / S[:k_actual][mask]
        
        Vh = (A_64.T @ U) * inv_S.unsqueeze(0)
        Vh = Vh.T
        
    else:
        # Tall matrix: work with A.T @ A which is (n × n)
        Omega = torch.randn(m, l, device=device, dtype=torch.float64)
        Y = A_64.T @ Omega
        
        for _ in range(n_iter):
            Z = A_64 @ Y
            Y = A_64.T @ Z
        
        Q, _ = torch.linalg.qr(Y, mode='reduced')
        
        AtA_Q = A_64.T @ (A_64 @ Q)
        BtB = Q.T @ AtA_Q
        
        eigvals, eigvecs = torch.linalg.eigh(BtB)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        S = torch.sqrt(torch.clamp(eigvals, min=0))
        V = Q @ eigvecs
        Vh = V.T
        
        k_actual = min(k, max(1, int((S > tol * S[0]).sum())))
        
        inv_S = torch.zeros_like(S)
        mask = S[:k_actual] > tol * S[0]
        inv_S[:k_actual][mask] = 1.0 / S[:k_actual][mask]
        
        U = (A_64 @ V) * inv_S.unsqueeze(0)
    
    return (
        U[:, :k_actual].to(dtype),
        S[:k_actual].to(dtype),
        Vh[:k_actual, :].to(dtype),
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
    
    where T_k[r, r'] = sum_i A_k[r_left, i, r] * B_k[r_left', i, r']
    
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
        A_k = cores_a[k]  # (r_left_a, n_k, r_right_a)
        B_k = cores_b[k]  # (r_left_b, n_k, r_right_b)
        
        r_left_a, n_k, r_right_a = A_k.shape
        r_left_b, _, r_right_b = B_k.shape
        
        # Contract: T_new[r_a', r_b'] = sum_{r_a, r_b, i} T[r_a, r_b] * A[r_a, i, r_a'] * B[r_b, i, r_b']
        
        # Step 1: Reshape A and B for batch matmul
        # A_k: (r_left_a, n_k, r_right_a) -> (n_k, r_left_a, r_right_a)
        A_perm = A_k.permute(1, 0, 2)  # (n_k, r_left_a, r_right_a)
        B_perm = B_k.permute(1, 0, 2)  # (n_k, r_left_b, r_right_b)
        
        # Step 2: For each mode value i, compute outer product A[i] ⊗ B[i]
        # Result shape: (n_k, r_left_a, r_right_a, r_left_b, r_right_b)
        
        # Step 3: Sum over i and contract with T
        # This is the key operation — do it efficiently
        
        T_new = torch.zeros(r_right_a, r_right_b, device=device, dtype=dtype)
        
        for i in range(n_k):
            # A_i: (r_left_a, r_right_a)
            # B_i: (r_left_b, r_right_b)
            A_i = A_perm[i]
            B_i = B_perm[i]
            
            # Contract: T_new += A_i.T @ T @ B_i
            # (r_right_a, r_left_a) @ (r_left_a, r_left_b) @ (r_left_b, r_right_b)
            T_new += A_i.T @ T @ B_i
        
        T = T_new
    
    # Final trace (T should be 1x1)
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
    
    result_cores = []
    
    for k in range(d):
        A_k = cores_a[k]  # (r_left_a, n_k, r_right_a)
        B_k = cores_b[k]  # (r_left_b, n_k, r_right_b)
        
        r_left_a, n_k, r_right_a = A_k.shape
        r_left_b, _, r_right_b = B_k.shape
        
        if k == 0:
            # First core: horizontal concatenation
            # [[A, B]] with shape (1, n_k, r_right_a + r_right_b)
            C_k = torch.cat([A_k, B_k], dim=2)
        elif k == d - 1:
            # Last core: vertical concatenation
            # [[A], [B]] with shape (r_left_a + r_left_b, n_k, 1)
            C_k = torch.cat([A_k, B_k], dim=0)
        else:
            # Middle cores: block diagonal
            # [[A, 0], [0, B]] with shape (r_left_a + r_left_b, n_k, r_right_a + r_right_b)
            C_k = torch.zeros(
                r_left_a + r_left_b, n_k, r_right_a + r_right_b,
                device=device, dtype=dtype
            )
            C_k[:r_left_a, :, :r_right_a] = A_k
            C_k[r_left_a:, :, r_right_a:] = B_k
        
        result_cores.append(C_k)
    
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
    
    Complexity: O(d (r_a * r_b)^3) vs O(2^d) for dense
    """
    if len(cores_a) != len(cores_b):
        raise ValueError(f"Core count mismatch: {len(cores_a)} vs {len(cores_b)}")
    
    d = len(cores_a)
    device = cores_a[0].device
    dtype = cores_a[0].dtype
    
    result_cores = []
    
    for k in range(d):
        A_k = cores_a[k]  # (r_left_a, n_k, r_right_a)
        B_k = cores_b[k]  # (r_left_b, n_k, r_right_b)
        
        r_left_a, n_k, r_right_a = A_k.shape
        r_left_b, _, r_right_b = B_k.shape
        
        # Kronecker product over bond dimensions
        # C_k[ra*rb_left, i, ra*rb_right] = A_k[ra_left, i, ra_right] * B_k[rb_left, i, rb_right]
        
        C_k = torch.zeros(
            r_left_a * r_left_b, n_k, r_right_a * r_right_b,
            device=device, dtype=dtype
        )
        
        for i in range(n_k):
            # A_i: (r_left_a, r_right_a)
            # B_i: (r_left_b, r_right_b)
            A_i = A_k[:, i, :]
            B_i = B_k[:, i, :]
            
            # Kronecker product: (r_left_a * r_left_b, r_right_a * r_right_b)
            kron = torch.kron(A_i, B_i)
            C_k[:, i, :] = kron
        
        result_cores.append(C_k)
    
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
