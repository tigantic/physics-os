"""
Lanczos Algorithm
=================

Krylov subspace methods for eigenvalue problems and matrix exponentials.

Theory
------
The Lanczos algorithm builds an orthonormal basis for the Krylov subspace:
    K_m(A, v) = span{v, Av, A²v, ..., A^{m-1}v}

In this basis, A is represented by a tridiagonal matrix T_m, which can
be efficiently diagonalized to approximate eigenvalues and eigenvectors.

Key property: Extremal eigenvalues converge fastest (Kaniel-Paige theory).

For matrix exponential: exp(A)v ≈ V_m exp(T_m) e_1 ||v||

Degenerate Eigenvalues (Article V.5.3)
--------------------------------------
When the ground state is degenerate (multiple eigenvalues with same value),
the Lanczos algorithm will converge to ONE eigenvector in the degenerate
subspace. The specific eigenvector depends on the initial vector v0.

Behavior with degenerate spectra:
- Convergence rate may be slower if gap to first excited state is small
- The returned eigenvector is a valid ground state but not unique
- For computing full degenerate subspace, use block Lanczos or run
  multiple times with orthogonalized initial vectors
- The residual ||Av - λv|| remains a valid convergence criterion

Warning: Near-degenerate eigenvalues (gap < tol) may cause the algorithm
to mix states. Increase num_iter or decrease tol if accuracy is critical.

References:
    .. [1] Golub, G.H., Van Loan, C.F. "Matrix Computations", 4th ed., 2013.
    .. [2] Saad, Y. "Numerical Methods for Large Eigenvalue Problems", 2011.
    .. [3] Parlett, B.N. "The Symmetric Eigenvalue Problem", SIAM, 1998.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable, List
import torch
from torch import Tensor
import math


@dataclass
class LanczosResult:
    """Result container for Lanczos algorithm."""
    eigenvalue: float
    eigenvector: Tensor
    converged: bool
    iterations: int
    residual: float


def lanczos_ground_state(
    matvec: Callable[[Tensor], Tensor],
    v0: Tensor,
    num_iter: int = 100,
    tol: float = 1e-12,
    reorthogonalize: bool = True,
) -> LanczosResult:
    """
    Find the ground state (lowest eigenvalue) using Lanczos.
    
    Args:
        matvec: Function that computes A @ v
        v0: Initial vector
        num_iter: Maximum Lanczos iterations
        tol: Convergence tolerance for eigenvalue
        reorthogonalize: Full reorthogonalization (slower but more stable)
        
    Returns:
        LanczosResult with ground state eigenvalue and eigenvector
    """
    dtype = v0.dtype
    device = v0.device
    dim = v0.numel()
    
    # Normalize initial vector
    v = v0.reshape(-1).clone()
    v = v / torch.linalg.norm(v)
    
    # Storage for Lanczos vectors and tridiagonal elements
    V = [v]
    alpha_list = []
    beta_list = []
    
    # First iteration
    w = matvec(v.reshape(v0.shape)).reshape(-1)
    alpha = torch.dot(v, w).real
    alpha_list.append(alpha)
    w = w - alpha * v
    
    E_old = float('inf')
    converged = False
    final_iter = 1
    
    for j in range(1, min(num_iter, dim)):
        beta = torch.linalg.norm(w)
        
        if beta < 1e-14:
            # Krylov subspace exhausted
            converged = True
            break
        
        beta_list.append(beta)
        
        v_old = v
        v = w / beta
        
        # Full reorthogonalization for numerical stability
        if reorthogonalize:
            for v_prev in V:
                v = v - torch.dot(v_prev, v) * v_prev
            v = v / torch.linalg.norm(v)
        
        V.append(v)
        
        # Matrix-vector product
        w = matvec(v.reshape(v0.shape)).reshape(-1)
        w = w - beta * v_old
        
        alpha = torch.dot(v, w).real
        alpha_list.append(alpha)
        w = w - alpha * v
        
        # Check convergence by diagonalizing current T
        if j >= 2:
            T = _build_tridiagonal(alpha_list, beta_list, dtype, device)
            eigenvalues, _ = torch.linalg.eigh(T)
            E_new = eigenvalues[0].item()
            
            if abs(E_new - E_old) < tol:
                converged = True
                final_iter = j + 1
                break
            
            E_old = E_new
        
        final_iter = j + 1
    
    # Final diagonalization
    T = _build_tridiagonal(alpha_list, beta_list, dtype, device)
    eigenvalues, eigenvectors = torch.linalg.eigh(T)
    
    # Ground state in Lanczos basis
    E0 = eigenvalues[0].item()
    gs_coeff = eigenvectors[:, 0]
    
    # Transform back to original basis
    V_stack = torch.stack(V, dim=1)
    psi = V_stack @ gs_coeff[:len(V)]
    
    # Compute residual: ||A*psi - E*psi||
    Apsi = matvec(psi.reshape(v0.shape)).reshape(-1)
    residual = torch.linalg.norm(Apsi - E0 * psi).item()
    
    return LanczosResult(
        eigenvalue=E0,
        eigenvector=psi.reshape(v0.shape),
        converged=converged or residual < tol,
        iterations=final_iter,
        residual=residual,
    )


def _build_tridiagonal(
    alpha: List[Tensor],
    beta: List[Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Build tridiagonal matrix from alpha (diagonal) and beta (off-diagonal)."""
    k = len(alpha)
    T = torch.zeros(k, k, dtype=dtype, device=device)
    
    for i in range(k):
        T[i, i] = alpha[i]
        if i < len(beta):
            T[i, i+1] = beta[i]
            T[i+1, i] = beta[i]
    
    return T


def lanczos_expm(
    matvec: Callable[[Tensor], Tensor],
    v: Tensor,
    t: complex,
    num_iter: int = 30,
    tol: float = 1e-12,
) -> Tensor:
    """
    Compute exp(t*A) @ v using Lanczos.
    
    Uses the Krylov approximation:
        exp(tA)v ≈ ||v|| * V_m * exp(t*T_m) * e_1
    
    where T_m is the tridiagonal representation in the Krylov basis.
    
    Args:
        matvec: Function that computes A @ v
        v: Vector to apply exponential to
        t: Time parameter (can be complex for e^{-iHt})
        num_iter: Number of Lanczos iterations
        tol: Tolerance for convergence
        
    Returns:
        exp(t*A) @ v
    """
    dtype = v.dtype if v.is_complex() else torch.complex128
    device = v.device
    dim = v.numel()
    
    # Normalize
    v_flat = v.reshape(-1).to(dtype)
    norm_v = torch.linalg.norm(v_flat)
    
    if norm_v < 1e-15:
        return torch.zeros_like(v)
    
    q = v_flat / norm_v
    
    # Build Krylov basis
    Q = [q]
    alpha_list = []
    beta_list = []
    
    w = matvec(q.reshape(v.shape)).reshape(-1).to(dtype)
    alpha = torch.dot(q.conj(), w)
    alpha_list.append(alpha)
    w = w - alpha * q
    
    for j in range(1, min(num_iter, dim)):
        beta = torch.linalg.norm(w)
        
        if beta.abs() < 1e-14:
            break
        
        beta_list.append(beta)
        q_old = q
        q = w / beta
        
        # Reorthogonalize
        for q_prev in Q:
            q = q - torch.dot(q_prev.conj(), q) * q_prev
        q = q / torch.linalg.norm(q)
        
        Q.append(q)
        
        w = matvec(q.reshape(v.shape)).reshape(-1).to(dtype)
        w = w - beta * q_old
        
        alpha = torch.dot(q.conj(), w)
        alpha_list.append(alpha)
        w = w - alpha * q
    
    # Build and exponentiate tridiagonal matrix
    k = len(alpha_list)
    T = torch.zeros(k, k, dtype=dtype, device=device)
    
    for i in range(k):
        T[i, i] = alpha_list[i]
        if i < len(beta_list):
            T[i, i+1] = beta_list[i]
            T[i+1, i] = beta_list[i]
    
    # exp(t*T)
    expT = torch.linalg.matrix_exp(t * T)
    
    # Extract first column (e_1 in Krylov basis)
    exp_e1 = expT[:, 0]
    
    # Transform back
    Q_stack = torch.stack(Q, dim=1)
    result = norm_v * (Q_stack @ exp_e1[:len(Q)])
    
    return result.reshape(v.shape)


def lanczos_eigenvalues(
    matvec: Callable[[Tensor], Tensor],
    v0: Tensor,
    num_eigenvalues: int = 1,
    num_iter: int = 100,
    tol: float = 1e-12,
) -> Tuple[Tensor, Tensor]:
    """
    Find multiple lowest eigenvalues using Lanczos.
    
    Args:
        matvec: Matrix-vector product function
        v0: Initial vector
        num_eigenvalues: Number of eigenvalues to find
        num_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        (eigenvalues, eigenvectors) - sorted by eigenvalue
    """
    dtype = v0.dtype
    device = v0.device
    dim = v0.numel()
    
    v = v0.reshape(-1).clone()
    v = v / torch.linalg.norm(v)
    
    V = [v]
    alpha_list = []
    beta_list = []
    
    w = matvec(v.reshape(v0.shape)).reshape(-1)
    alpha = torch.dot(v, w).real
    alpha_list.append(alpha)
    w = w - alpha * v
    
    for j in range(1, min(num_iter, dim)):
        beta = torch.linalg.norm(w)
        
        if beta < 1e-14:
            break
        
        beta_list.append(beta)
        v_old = v
        v = w / beta
        
        # Reorthogonalize
        for v_prev in V:
            v = v - torch.dot(v_prev, v) * v_prev
        v = v / torch.linalg.norm(v)
        
        V.append(v)
        
        w = matvec(v.reshape(v0.shape)).reshape(-1)
        w = w - beta * v_old
        
        alpha = torch.dot(v, w).real
        alpha_list.append(alpha)
        w = w - alpha * v
    
    # Diagonalize
    T = _build_tridiagonal(alpha_list, beta_list, dtype, device)
    eigenvalues, eigenvectors_T = torch.linalg.eigh(T)
    
    # Transform eigenvectors back
    V_stack = torch.stack(V, dim=1)
    k = min(num_eigenvalues, len(eigenvalues))
    
    eigenvectors = []
    for i in range(k):
        coeff = eigenvectors_T[:len(V), i]
        psi = V_stack @ coeff
        eigenvectors.append(psi.reshape(v0.shape))
    
    return eigenvalues[:k], torch.stack(eigenvectors)
