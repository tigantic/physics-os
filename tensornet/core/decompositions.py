"""
Tensor Decompositions
=====================

SVD and QR decompositions with truncation for tensor networks.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor


def svd_truncated(
    A: Tensor,
    chi_max: Optional[int] = None,
    cutoff: float = 1e-14,
    return_info: bool = False,
) -> Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, dict]:
    """
    Truncated SVD with bond dimension control.
    
    Computes A ≈ U @ diag(S) @ Vh where rank is limited by chi_max
    and singular values below cutoff are discarded.
    
    This is the optimal rank-k approximation by the Eckart-Young-Mirsky theorem.
    
    Args:
        A: Input matrix of shape (m, n)
        chi_max: Maximum number of singular values to keep
        cutoff: Discard singular values below this threshold
        return_info: If True, return dictionary with truncation info
        
    Returns:
        U: Left singular vectors (m, k)
        S: Singular values (k,)
        Vh: Right singular vectors (k, n)
        info: (optional) Dictionary with truncation_error, rank, etc.
        
    Example:
        >>> A = torch.randn(100, 100, dtype=torch.float64)
        >>> U, S, Vh = svd_truncated(A, chi_max=20)
        >>> A_approx = U @ torch.diag(S) @ Vh
    """
    # Full SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Determine truncation rank
    rank = len(S)
    
    # Apply cutoff
    mask = S > cutoff
    if not mask.all():
        rank = mask.sum().item()
    
    # Apply chi_max
    if chi_max is not None and chi_max < rank:
        rank = chi_max
    
    # Ensure at least rank 1
    rank = max(rank, 1)
    
    # Truncate
    U = U[:, :rank]
    S_trunc = S[:rank]
    Vh = Vh[:rank, :]
    
    if return_info:
        # Truncation error: ||A - A_k||_F / ||A||_F
        if rank < len(S):
            trunc_error = torch.sqrt((S[rank:] ** 2).sum()).item()
        else:
            trunc_error = 0.0
        
        info = {
            "truncation_error": trunc_error,
            "rank": rank,
            "original_rank": len(S),
            "max_singular_value": S[0].item(),
            "min_singular_value": S_trunc[-1].item(),
        }
        return U, S_trunc, Vh, info
    
    return U, S_trunc, Vh


def qr_positive(A: Tensor) -> Tuple[Tensor, Tensor]:
    """
    QR decomposition with positive diagonal R.
    
    Standard QR can have arbitrary signs on the diagonal of R.
    This function ensures R has positive diagonal, which is
    important for canonical MPS forms.
    
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
    
    # Fix signs
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1  # Handle exact zeros
    
    # Apply sign correction
    Q = Q * signs.unsqueeze(0)
    R = R * signs.unsqueeze(1)
    
    return Q, R


def thin_svd(A: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Thin (economy) SVD without truncation.
    
    Args:
        A: Input matrix (m, n)
        
    Returns:
        U: (m, k) where k = min(m, n)
        S: (k,)
        Vh: (k, n)
    """
    return torch.linalg.svd(A, full_matrices=False)


def polar_decomposition(A: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Polar decomposition A = U @ P where U is unitary and P is positive semidefinite.
    
    Args:
        A: Input matrix (m, n) with m >= n
        
    Returns:
        U: Unitary matrix (m, n)
        P: Positive semidefinite (n, n)
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U @ Vh, Vh.T.conj() @ torch.diag(S) @ Vh
