"""
QTT arithmetic operations in compressed format.

All operations work directly on QTT cores without decompressing to dense format,
enabling billion-point computations with minimal memory.
"""

from __future__ import annotations
from typing import List, Optional
import torch
from torch import Tensor

from qtt_sdk.core import QTTState


def qtt_add(
    a: QTTState,
    b: QTTState,
    max_bond: Optional[int] = None,
    tol: float = 1e-12
) -> QTTState:
    """
    Add two QTT states: result = a + b.
    
    The result has bond dimension r_a + r_b before truncation.
    Use max_bond to control rank growth.
    
    Args:
        a: First QTT state
        b: Second QTT state (must have same num_qubits)
        max_bond: Maximum bond dimension (None = no truncation)
        tol: Relative tolerance for truncation
    
    Returns:
        QTT state representing a + b
    
    Complexity:
        O(n * (r_a + r_b)^3) with truncation, O(n * (r_a + r_b)^2) without
    
    Example:
        >>> sum_state = qtt_add(qtt_a, qtt_b, max_bond=64)
    """
    if a.num_qubits != b.num_qubits:
        raise ValueError(
            f"QTT states must have same num_qubits: {a.num_qubits} vs {b.num_qubits}"
        )
    
    n = a.num_qubits
    new_cores: List[Tensor] = []
    
    for i in range(n):
        ca, cb = a.cores[i], b.cores[i]
        ra_left, _, ra_right = ca.shape
        rb_left, _, rb_right = cb.shape
        
        if i == 0:
            # First core: concatenate along right bond
            # (1, 2, ra) and (1, 2, rb) -> (1, 2, ra+rb)
            new_core = torch.cat([ca, cb], dim=2)
        elif i == n - 1:
            # Last core: concatenate along left bond
            # (ra, 2, 1) and (rb, 2, 1) -> (ra+rb, 2, 1)
            new_core = torch.cat([ca, cb], dim=0)
        else:
            # Middle core: block diagonal
            # (ra_l, 2, ra_r) and (rb_l, 2, rb_r) -> (ra_l+rb_l, 2, ra_r+rb_r)
            new_core = torch.zeros(
                ra_left + rb_left, 2, ra_right + rb_right,
                dtype=ca.dtype, device=ca.device
            )
            new_core[:ra_left, :, :ra_right] = ca
            new_core[ra_left:, :, ra_right:] = cb
        
        new_cores.append(new_core)
    
    result = QTTState(cores=new_cores, num_qubits=n)
    
    if max_bond is not None:
        result = truncate_qtt(result, max_bond, tol)
    
    return result


def qtt_scale(qtt: QTTState, scalar: float) -> QTTState:
    """
    Scale a QTT state by a scalar: result = scalar * qtt.
    
    Only modifies the first core, preserving bond dimensions.
    
    Args:
        qtt: QTT state to scale
        scalar: Scalar multiplier
    
    Returns:
        Scaled QTT state
    
    Complexity:
        O(r^2) - constant time regardless of grid size
    
    Example:
        >>> scaled = qtt_scale(qtt, 2.5)
    """
    new_cores = [core.clone() for core in qtt.cores]
    new_cores[0] = new_cores[0] * scalar
    return QTTState(cores=new_cores, num_qubits=qtt.num_qubits)


def qtt_inner_product(a: QTTState, b: QTTState) -> float:
    """
    Compute inner product <a|b> of two QTT states.
    
    Uses the transfer matrix method for efficient contraction.
    
    Args:
        a: First QTT state
        b: Second QTT state (must have same num_qubits)
    
    Returns:
        Inner product as a scalar
    
    Complexity:
        O(n * r_a^2 * r_b^2)
    
    Example:
        >>> overlap = qtt_inner_product(qtt_a, qtt_b)
    """
    if a.num_qubits != b.num_qubits:
        raise ValueError(
            f"QTT states must have same num_qubits: {a.num_qubits} vs {b.num_qubits}"
        )
    
    # Transfer matrix approach
    # T[i,j] represents contraction of all sites to the left
    # where i is bond index of a, j is bond index of b
    
    T = torch.ones(1, 1, dtype=a.dtype, device=a.device)
    
    for ca, cb in zip(a.cores, b.cores):
        # ca: (ra_left, 2, ra_right)
        # cb: (rb_left, 2, rb_right)
        # T: (ra_left, rb_left)
        
        # Contract: T_new[k,l] = sum over (i,j,s) of 
        #           T[i, j] * ca[i, s, k] * cb[j, s, l]
        # Result: T_new[ra_right, rb_right]
        
        T = torch.einsum('ij,isk,jsl->kl', T, ca, cb)
    
    return T.squeeze().item()


def qtt_norm(qtt: QTTState) -> float:
    """
    Compute L2 norm of a QTT state.
    
    Args:
        qtt: QTT state
    
    Returns:
        L2 norm = sqrt(<qtt|qtt>)
    
    Complexity:
        O(n * r^4)
    
    Example:
        >>> norm = qtt_norm(qtt)
    """
    return (qtt_inner_product(qtt, qtt) ** 0.5)


def truncate_qtt(
    qtt: QTTState,
    max_bond: int,
    tol: float = 1e-12
) -> QTTState:
    """
    Truncate (recompress) a QTT state to lower bond dimension.
    
    Uses sequential SVD sweeping for optimal truncation.
    
    Args:
        qtt: QTT state to truncate
        max_bond: Maximum allowed bond dimension
        tol: Relative tolerance for SVD truncation
    
    Returns:
        Truncated QTT state with smaller bond dimensions
    
    Complexity:
        O(n * r^3)
    
    Example:
        >>> compressed = truncate_qtt(qtt, max_bond=32)
    """
    if qtt.max_rank <= max_bond:
        return qtt.clone()
    
    n = qtt.num_qubits
    cores = [c.clone() for c in qtt.cores]
    
    # Left-to-right sweep: orthogonalize
    for i in range(n - 1):
        core = cores[i]
        r_left, d_phys, r_right = core.shape
        
        # Reshape and QR
        mat = core.reshape(r_left * d_phys, r_right)
        Q, R = torch.linalg.qr(mat)
        
        new_r = min(Q.shape[1], max_bond)
        Q = Q[:, :new_r]
        R = R[:new_r, :]
        
        cores[i] = Q.reshape(r_left, d_phys, new_r)
        
        # Absorb R into next core
        next_core = cores[i + 1]
        cores[i + 1] = torch.einsum('ij,jkl->ikl', R, next_core)
    
    # Right-to-left sweep: truncate with SVD
    for i in range(n - 1, 0, -1):
        core = cores[i]
        r_left, d_phys, r_right = core.shape
        
        # Reshape and SVD
        mat = core.reshape(r_left, d_phys * r_right)
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        
        # Truncate
        if tol > 0 and S.numel() > 0:
            total = S.sum()
            if total > 1e-15:
                cumsum = torch.cumsum(S, dim=0)
                rank_tol = torch.searchsorted(cumsum, total * (1 - tol)).item() + 1
            else:
                rank_tol = 1
        else:
            rank_tol = len(S)
        
        rank = min(max_bond, rank_tol, len(S))
        
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        cores[i] = Vh.reshape(rank, d_phys, r_right)
        
        # Absorb U @ diag(S) into previous core
        prev_core = cores[i - 1]
        US = U @ torch.diag(S)
        cores[i - 1] = torch.einsum('ijk,kl->ijl', prev_core, US)
    
    return QTTState(cores=cores, num_qubits=n)


def qtt_elementwise_product(
    a: QTTState,
    b: QTTState,
    max_bond: Optional[int] = None,
    tol: float = 1e-12
) -> QTTState:
    """
    Compute element-wise (Hadamard) product of two QTT states.
    
    The result has bond dimension r_a * r_b before truncation.
    
    Args:
        a: First QTT state
        b: Second QTT state (must have same num_qubits)
        max_bond: Maximum bond dimension (None = no truncation)
        tol: Relative tolerance for truncation
    
    Returns:
        QTT state representing element-wise product a * b
    
    Complexity:
        O(n * r_a^2 * r_b^2) without truncation
    """
    if a.num_qubits != b.num_qubits:
        raise ValueError(
            f"QTT states must have same num_qubits: {a.num_qubits} vs {b.num_qubits}"
        )
    
    n = a.num_qubits
    new_cores: List[Tensor] = []
    
    for ca, cb in zip(a.cores, b.cores):
        # ca: (ra_left, 2, ra_right)
        # cb: (rb_left, 2, rb_right)
        # Result: ((ra_left * rb_left), 2, (ra_right * rb_right))
        
        ra_left, _, ra_right = ca.shape
        rb_left, _, rb_right = cb.shape
        
        # Kronecker product of left and right bonds
        new_core = torch.einsum('iaj,kak->ikajk', ca, cb)
        new_core = new_core.reshape(ra_left * rb_left, 2, ra_right * rb_right)
        new_cores.append(new_core)
    
    result = QTTState(cores=new_cores, num_qubits=n)
    
    if max_bond is not None:
        result = truncate_qtt(result, max_bond, tol)
    
    return result


def qtt_subtract(
    a: QTTState,
    b: QTTState,
    max_bond: Optional[int] = None,
    tol: float = 1e-12
) -> QTTState:
    """
    Subtract two QTT states: result = a - b.
    
    Args:
        a: First QTT state
        b: Second QTT state to subtract
        max_bond: Maximum bond dimension
        tol: Relative tolerance for truncation
    
    Returns:
        QTT state representing a - b
    """
    neg_b = qtt_scale(b, -1.0)
    return qtt_add(a, neg_b, max_bond, tol)
