"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    rSVD - RANDOMIZED SINGULAR VALUE DECOMPOSITION            ║
║                                                                              ║
║                 GPU-NATIVE • O(mnk) • ZERO DENSE PHILOSOPHY                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

QTT Principles Enforced:
1. NEVER call torch.linalg.svd on matrices > 4x4
2. Use Gram matrix eigendecomposition for O(mnk) complexity
3. All operations on GPU - no .cpu() calls
4. Adaptive rank truncation based on singular value decay

Usage:
    from ontic.genesis.core.rsvd import rsvd_gpu
    
    U, S, Vh = rsvd_gpu(A, k=50)  # Always GPU, always fast
"""

import torch
from typing import Tuple, Optional

__all__ = ['rsvd_gpu', 'tt_decompose_rsvd']


def rsvd_gpu(
    A: torch.Tensor,
    k: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD using Gram matrix eigendecomposition.
    
    This is the ONLY SVD function to use in the QTT codebase.
    It handles both wide (m < n) and tall (m >= n) matrices correctly.
    
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
    Memory: O(ml + l²) where l = k + n_oversamples
    
    CRITICAL: For matrices smaller than 5x5, falls back to full SVD
    since randomized methods add overhead on tiny matrices.
    """
    m, n = A.shape
    device = A.device
    dtype = A.dtype
    
    # Small matrix fast path - overhead not worth it
    if m <= 4 or n <= 4:
        U, S, Vh = torch.linalg.svd(A.to(torch.float64), full_matrices=False)
        k_actual = min(k, len(S), max(1, int((S > tol * S[0]).sum())))
        return U[:, :k_actual].to(dtype), S[:k_actual].to(dtype), Vh[:k_actual, :].to(dtype)
    
    l = min(k + n_oversamples, min(m, n))
    
    # Work in float64 for numerical stability
    A_64 = A.to(torch.float64) if dtype != torch.float64 else A
    
    if m <= n:
        # Wide matrix (m ≤ n): work with A @ A.T which is (m × m)
        # Random projection
        Omega = torch.randn(n, l, device=device, dtype=torch.float64)
        
        # Y = A @ Omega
        Y = A_64 @ Omega
        
        # Power iteration for better approximation
        for _ in range(n_iter):
            Z = A_64.T @ Y
            Y = A_64 @ Z
        
        # QR factorization
        Q, _ = torch.linalg.qr(Y, mode='reduced')  # (m, l)
        
        # Form B @ B.T = Q.T @ A @ A.T @ Q via (A @ A.T) @ Q
        AAt_Q = A_64 @ (A_64.T @ Q)  # (m, l)
        BtB = Q.T @ AAt_Q  # (l, l) - small!
        
        # Eigendecomposition of small Gram matrix
        eigvals, eigvecs = torch.linalg.eigh(BtB)
        
        # Sort descending (eigh returns ascending)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Singular values from eigenvalues
        S = torch.sqrt(torch.clamp(eigvals, min=0))
        
        # Left singular vectors: U = Q @ eigvecs
        U = Q @ eigvecs
        
        # Right singular vectors: V = A.T @ U @ diag(1/S)
        k_actual = min(k, max(1, int((S > tol * S[0]).sum())))
        
        inv_S = torch.zeros_like(S)
        mask = S[:k_actual] > tol * S[0]
        inv_S[:k_actual][mask] = 1.0 / S[:k_actual][mask]
        
        Vh = (A_64.T @ U) * inv_S.unsqueeze(0)
        Vh = Vh.T
        
    else:
        # Tall matrix (m > n): work with A.T @ A which is (n × n)
        Omega = torch.randn(m, l, device=device, dtype=torch.float64)
        
        Y = A_64.T @ Omega  # (n, l)
        
        # Power iteration
        for _ in range(n_iter):
            Z = A_64 @ Y
            Y = A_64.T @ Z
        
        Q, _ = torch.linalg.qr(Y, mode='reduced')  # (n, l)
        
        # Form B.T @ B = Q.T @ A.T @ A @ Q
        AtA_Q = A_64.T @ (A_64 @ Q)  # (n, l)
        BtB = Q.T @ AtA_Q  # (l, l) - small!
        
        eigvals, eigvecs = torch.linalg.eigh(BtB)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        S = torch.sqrt(torch.clamp(eigvals, min=0))
        
        # Right singular vectors: V = Q @ eigvecs
        V = Q @ eigvecs
        Vh = V.T
        
        # Left singular vectors: U = A @ V @ diag(1/S)
        k_actual = min(k, max(1, int((S > tol * S[0]).sum())))
        
        inv_S = torch.zeros_like(S)
        mask = S[:k_actual] > tol * S[0]
        inv_S[:k_actual][mask] = 1.0 / S[:k_actual][mask]
        
        U = (A_64 @ V) * inv_S.unsqueeze(0)
    
    # Truncate to k_actual
    k_actual = min(k, len(S), k_actual)
    
    return (
        U[:, :k_actual].to(dtype),
        S[:k_actual].to(dtype),
        Vh[:k_actual, :].to(dtype),
    )


def tt_decompose_rsvd(
    tensor: torch.Tensor,
    max_rank: int = 50,
    tol: float = 1e-10,
) -> Tuple[list, list]:
    """
    TT/QTT decomposition using rSVD with adaptive rank.
    
    This replaces the standard TT-SVD with O(d * n * r²) rSVD.
    
    Args:
        tensor: 1D tensor of size 2^d (for QTT) or nD tensor
        max_rank: Maximum TT rank allowed
        tol: Truncation tolerance
    
    Returns:
        cores: List of TT cores
        ranks: List of TT ranks [1, r1, r2, ..., r_{d-1}, 1]
    """
    import math
    
    device = tensor.device
    dtype = tensor.dtype
    
    # Determine shape
    if tensor.ndim == 1:
        # QTT format: reshape to [2, 2, ..., 2]
        n = tensor.numel()
        d = int(math.log2(n))
        if 2**d != n:
            raise ValueError(f"QTT requires tensor size to be power of 2, got {n}")
        shape = [2] * d
        tensor = tensor.reshape(shape)
    else:
        shape = list(tensor.shape)
        d = len(shape)
    
    cores = []
    ranks = [1]
    
    C = tensor.reshape(shape[0], -1).to(torch.float64)
    r_prev = 1
    
    for k in range(d - 1):
        m, n = C.shape
        target_rank = min(max_rank, m, n)
        
        # rSVD - handles wide/tall matrices correctly
        U, S, Vh = rsvd_gpu(C, k=target_rank, tol=tol)
        
        # Adaptive rank based on singular value decay
        rank = min(max_rank, max(1, len(S)))
        
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        # Form core: (r_in, mode_size, r_out)
        core = U.reshape(r_prev, shape[k], rank).to(dtype)
        cores.append(core)
        ranks.append(rank)
        
        # Remaining tensor for next iteration
        SV = torch.diag(S) @ Vh
        if k + 1 < d - 1:
            remaining_cols = SV.shape[1] // shape[k + 1]
            C = SV.reshape(rank * shape[k + 1], remaining_cols)
        else:
            C = SV
        r_prev = rank
    
    # Last core
    last_core = C.reshape(r_prev, shape[-1], 1).to(dtype)
    cores.append(last_core)
    ranks.append(1)
    
    return cores, ranks


def svd_fallback(
    mat: torch.Tensor,
    max_rank: int = 50,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified SVD that always uses rSVD for large matrices.
    
    This is a drop-in replacement for torch.linalg.svd calls.
    Import this and replace:
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    with:
        U, S, Vh = svd_fallback(mat, max_rank=50)
    """
    m, n = mat.shape
    k = min(max_rank, m, n)
    return rsvd_gpu(mat, k=k, tol=tol)
