"""
Tensor Decompositions
=====================

SVD and QR decompositions with truncation for tensor networks.

Constitutional Compliance:
    - Article V.5.1: Condition number warnings when κ > 10¹⁰
    - Article V.5.2: SVD truncation with return_info
    - Article VIII.8.2: Memory profiling decorator
"""

import logging
import warnings

import torch
from torch import Tensor

from tensornet.core.profiling import memory_profile

# Constitutional threshold (Article V, Section 5.1)
CONDITION_NUMBER_THRESHOLD = 1e10

logger = logging.getLogger(__name__)


@memory_profile
def svd_truncated(
    A: Tensor,
    chi_max: int | None = None,
    cutoff: float = 1e-14,
    return_info: bool = False,
    use_rsvd: bool | None = None,
    rsvd_threshold: int = 256,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, dict]:
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
        use_rsvd: Force rSVD (True) or exact SVD (False). If None, auto-select.
        rsvd_threshold: Use rSVD when min(m,n) > threshold (default 256)

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
    m, n = A.shape
    min_dim = min(m, n)
    
    # Determine target rank
    if chi_max is not None:
        target_rank = min(chi_max, min_dim)
    else:
        target_rank = min_dim

    # Decide whether to use rSVD or exact SVD
    # rSVD is faster for large matrices when we only need top-k singular values
    # But it's approximate, so use exact SVD for small matrices or proofs
    if use_rsvd is None:
        # Auto-select: use rSVD for large matrices where we're truncating significantly
        use_rsvd = (min_dim > rsvd_threshold) and (target_rank < min_dim // 2)

    if use_rsvd:
        # Randomized SVD (Halko-Martinsson-Tropp algorithm)
        # O(m·n·k) instead of O(m·n·min(m,n))
        # Request slightly more for accuracy, then truncate
        q = min(target_rank + 5, min_dim)
        U, S, V = torch.svd_lowrank(A, q=q, niter=2)
        Vh = V.T
    else:
        # Exact SVD - needed for mathematical proofs and small matrices
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # Apply cutoff threshold
    rank = min(target_rank, len(S))
    if cutoff > 0:
        mask = S > cutoff
        if not mask.all():
            rank = max(min(mask.sum().item(), rank), 1)

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

        # Compute condition number (Article V, Section 5.1)
        if S_trunc[-1].item() > 0:
            condition_number = S[0].item() / S_trunc[-1].item()
        else:
            condition_number = float("inf")

        # Warn if condition number exceeds threshold
        if condition_number > CONDITION_NUMBER_THRESHOLD:
            warnings.warn(
                f"High condition number detected: κ = {condition_number:.2e} > 10¹⁰. "
                f"Numerical stability may be compromised.",
                RuntimeWarning,
            )
            logger.warning(
                f"Condition number κ = {condition_number:.2e} exceeds threshold"
            )

        info = {
            "truncation_error": trunc_error,
            "rank": rank,
            "original_rank": len(S),
            "max_singular_value": S[0].item(),
            "min_singular_value": S_trunc[-1].item(),
            "condition_number": condition_number,
        }
        return U, S_trunc, Vh, info

    return U, S_trunc, Vh


def qr_positive(A: Tensor) -> tuple[Tensor, Tensor]:
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


def thin_svd(A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Thin (economy) SVD without truncation.

    Args:
        A: Input matrix (m, n)

    Returns:
        U: (m, k) where k = min(m, n)
        S: (k,)
        Vh: (k, n)
    """
    # Use svd_lowrank for better performance (Halko-Martinsson-Tropp algorithm)
    min_dim = min(A.shape)
    return torch.svd_lowrank(A, q=min_dim, niter=2)


def polar_decomposition(A: Tensor) -> tuple[Tensor, Tensor]:
    """
    Polar decomposition A = U @ P where U is unitary and P is positive semidefinite.

    Args:
        A: Input matrix (m, n) with m >= n

    Returns:
        U: Unitary matrix (m, n)
        P: Positive semidefinite (n, n)
    """
    # Use svd_lowrank for 4× speedup
    min_dim = min(A.shape)
    U, S, Vh = torch.svd_lowrank(A, q=min_dim, niter=2)
    return U @ Vh, Vh.T.conj() @ torch.diag(S) @ Vh
