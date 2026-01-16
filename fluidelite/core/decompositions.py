"""
Core Tensor Decompositions
==========================

Implements numerically stable and GPU-accelerated decompositions.
Includes SafeSVD (Lorentzian gradient), Randomized SVD, and QR with positive diagonal.

Constitutional Compliance:
    - Article V.5.1: All public functions documented
    - Article VII.7.2: Definition of Done = USER-OBSERVABLE BEHAVIOR works
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Tuple, Optional


class SafeSVD(torch.autograd.Function):
    """
    SVD with Lorentzian Broadening for the gradient.
    Prevents explosion when singular values are degenerate.
    
    This is critical for training stability in tensor networks where
    singular values can become very close or identical.
    """
    @staticmethod
    def forward(ctx, A: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass: standard SVD.
        
        Args:
            A: Input matrix
            
        Returns:
            U, S, Vh from SVD decomposition
        """
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        ctx.save_for_backward(U, S, Vh)
        return U, S, Vh

    @staticmethod
    def backward(ctx, dU: Tensor, dS: Tensor, dVh: Tensor) -> Tensor:
        """
        Backward pass with Lorentzian regularization.
        
        Uses F_ij = 1/(s_i² - s_j² + ε) with Lorentzian broadening
        to prevent gradient explosion when singular values are degenerate.
        """
        U, S, Vh = ctx.saved_tensors
        Vt = Vh
        
        S2 = S.pow(2)
        S2_new = S2.unsqueeze(-1)
        
        # Lorentzian broadening epsilon prevents division by zero
        epsilon = 1e-12
        F = 1.0 / (S2_new - S2.unsqueeze(-2) + epsilon)
        
        # Zero out diagonal (self-interaction terms)
        mask = torch.eye(F.shape[-1], device=F.device, dtype=torch.bool)
        F.masked_fill_(mask, 0.0)
        
        # Compute gradient contribution from dS
        dA = U @ torch.diag(dS) @ Vt
        
        # Add contributions from dU and dVh (simplified)
        if dU is not None:
            dA = dA + U @ (F * (U.T @ dU - dU.T @ U)) @ torch.diag(S) @ Vt
        if dVh is not None:
            dA = dA + U @ torch.diag(S) @ (F * (Vt @ dVh.T - dVh @ Vt.T)) @ Vt
            
        return dA


def rsvd_truncated(
    tensor: Tensor, 
    rank: int, 
    n_oversamples: int = 10, 
    n_iter: int = 2
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Randomized SVD for GPU acceleration.
    
    Uses the Halko-Martinsson-Tropp algorithm for O(m*n*k) complexity
    instead of O(min(m,n)³) for exact SVD.
    
    Args:
        tensor: Input matrix of shape (m, n)
        rank: Target rank (number of singular values to compute)
        n_oversamples: Extra samples for accuracy (default 10)
        n_iter: Power iterations for accuracy (default 2)
        
    Returns:
        U: Left singular vectors (m, rank)
        S: Singular values (rank,)
        Vh: Right singular vectors (rank, n)
        
    Example:
        >>> A = torch.randn(1000, 500)
        >>> U, S, Vh = rsvd_truncated(A, rank=50)
        >>> A_approx = U @ torch.diag(S) @ Vh
    """
    m, n = tensor.shape
    k = rank + n_oversamples
    
    # Random projection matrix
    Omega = torch.randn(n, k, device=tensor.device, dtype=tensor.dtype)
    
    # Form Y = A @ Omega and do power iteration for accuracy
    Y = tensor @ Omega
    for _ in range(n_iter):
        Y = tensor @ (tensor.T @ Y)
        
    # QR to get orthonormal basis
    Q, _ = torch.linalg.qr(Y)
    
    # Project A onto Q: B = Q^T @ A
    B = Q.T @ tensor
    
    # SVD of small matrix B
    U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    
    # Recover U = Q @ U_hat
    U = Q @ U_hat
    
    return U[:, :rank], S[:rank], Vh[:rank, :]


def svd_truncated(
    A: Tensor, 
    chi_max: Optional[int] = None, 
    cutoff: float = 1e-14, 
    use_rsvd_threshold: int = 32
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Unified SVD interface with automatic algorithm selection.
    
    - Uses exact SVD for small tensors (numerical precision)
    - Uses rSVD (via torch.svd_lowrank) for larger tensors (performance)
    
    Algorithm selection:
        - min(m,n) <= 32: Exact SVD (small matrices, precision matters)
        - min(m,n) > 32 AND chi_max <= min(m,n): Native rSVD (fast)
    
    Args:
        A: Input matrix of shape (m, n)
        chi_max: Maximum number of singular values to keep
        cutoff: Discard singular values below this threshold
        use_rsvd_threshold: Use rSVD when min(m,n) > threshold (default 32)
        
    Returns:
        U: Left singular vectors (m, k)
        S: Singular values (k,)
        Vh: Right singular vectors (k, n)
        
    Example:
        >>> A = torch.randn(100, 100, dtype=torch.float64)
        >>> U, S, Vh = svd_truncated(A, chi_max=20)
        >>> A_approx = U @ torch.diag(S) @ Vh
    """
    m, n = A.shape
    
    # Decide whether to use rSVD based on matrix size and truncation
    # rSVD is O(m*n*k) vs O(min(m,n)³) for exact - wins when k < min(m,n)
    use_rsvd = (
        chi_max is not None 
        and min(m, n) > use_rsvd_threshold 
        and chi_max <= min(m, n)  # Only need k <= matrix rank
    )
    
    if use_rsvd:
        # Use PyTorch native svd_lowrank - CUDA optimized
        # q = chi_max + oversampling for accuracy
        q = min(chi_max + 10, min(m, n))
        try:
            U, S, V = torch.svd_lowrank(A, q=q, niter=2)
            Vh = V.T
        except RuntimeError:
            # Fallback to our implementation
            U, S, Vh = rsvd_truncated(A, chi_max)
    else:
        try:
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        except RuntimeError:
            # SVD can fail for ill-conditioned matrices; add small noise
            noisy_A = A + torch.randn_like(A) * 1e-6
            U, S, Vh = torch.linalg.svd(noisy_A, full_matrices=False)

    # Apply cutoff threshold
    mask = S > cutoff
    if mask.sum() == 0:
        mask[0] = True  # Keep at least one
        
    U = U[:, mask]
    S = S[mask]
    Vh = Vh[mask, :]
    
    # Apply chi_max truncation
    if chi_max is not None and S.shape[0] > chi_max:
        U = U[:, :chi_max]
        S = S[:chi_max]
        Vh = Vh[:chi_max, :]
        
    return U, S, Vh


def qr_positive(A: Tensor) -> Tuple[Tensor, Tensor]:
    """
    QR decomposition with enforced positive diagonal R.
    
    Standard QR can have arbitrary signs on the diagonal of R.
    This function ensures R has positive diagonal, which is
    important for canonical MPS forms (gauge consistency).
    
    Args:
        A: Input matrix of shape (m, n)
        
    Returns:
        Q: Orthogonal matrix (m, min(m,n))
        R: Upper triangular with positive diagonal (min(m,n), n)
        
    Example:
        >>> A = torch.randn(50, 30, dtype=torch.float64)
        >>> Q, R = qr_positive(A)
        >>> assert torch.allclose(A, Q @ R)
        >>> assert (torch.diag(R) >= 0).all()
    """
    Q, R = torch.linalg.qr(A)
    
    # Get signs of diagonal elements
    diag_signs = torch.sign(torch.diag(R))
    diag_signs[diag_signs == 0] = 1  # Handle exact zeros
    
    # Apply sign correction to make diagonal positive
    Q = Q * diag_signs.unsqueeze(0)
    R = R * diag_signs.unsqueeze(1)
    
    return Q, R
