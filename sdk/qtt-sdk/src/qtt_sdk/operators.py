"""
MPO (Matrix Product Operator) construction and application.

This module provides operators for derivatives, Laplacians, and other
linear operations that can be applied directly in QTT format.
"""

from __future__ import annotations
from typing import List, Optional
import torch
from torch import Tensor

from qtt_sdk.core import QTTState, MPO


def identity_mpo(num_sites: int, dtype: torch.dtype = torch.float64) -> MPO:
    """
    Create identity MPO.
    
    Args:
        num_sites: Number of sites (qubits)
        dtype: Data type for tensors
    
    Returns:
        Identity MPO
    """
    cores = []
    for i in range(num_sites):
        # Identity: |0><0| + |1><1|
        core = torch.zeros(1, 2, 2, 1, dtype=dtype)
        core[0, 0, 0, 0] = 1.0
        core[0, 1, 1, 0] = 1.0
        cores.append(core)
    
    return MPO(cores=cores, num_sites=num_sites)


def shift_mpo(
    num_sites: int,
    direction: int = 1,
    boundary: str = "periodic",
    dtype: torch.dtype = torch.float64
) -> MPO:
    """
    Create shift operator MPO.
    
    Shifts the grid by one position in the specified direction.
    In binary (QTT) representation, this is an increment/decrement.
    
    Args:
        num_sites: Number of sites (qubits)
        direction: +1 for right shift, -1 for left shift
        boundary: "periodic" or "zero"
        dtype: Data type
    
    Returns:
        Shift MPO
    """
    # Binary increment: carry propagation
    # Each site has states: (carry_in) x (bit) -> (bit') x (carry_out)
    # The MPO bond carries the carry bit
    
    cores = []
    
    for i in range(num_sites):
        if i == 0:
            # Least significant bit, always add 1
            core = torch.zeros(1, 2, 2, 2, dtype=dtype)
            if direction > 0:
                # 0 + 1 = 1, no carry
                core[0, 0, 1, 0] = 1.0
                # 1 + 1 = 0, carry
                core[0, 1, 0, 1] = 1.0
            else:
                # 0 - 1 = 1, borrow (handled at boundary)
                core[0, 0, 1, 1] = 1.0
                # 1 - 1 = 0, no borrow
                core[0, 1, 0, 0] = 1.0
        elif i == num_sites - 1:
            # Most significant bit
            core = torch.zeros(2, 2, 2, 1, dtype=dtype)
            # No carry: identity
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            # Carry in: flip and periodic wrap
            if boundary == "periodic":
                core[1, 0, 1, 0] = 1.0
                core[1, 1, 0, 0] = 1.0
            else:  # zero boundary
                core[1, 0, 1, 0] = 1.0
                core[1, 1, 0, 0] = 1.0  # Still wrap for internal consistency
        else:
            # Middle bits
            core = torch.zeros(2, 2, 2, 2, dtype=dtype)
            # No carry in: identity
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            # Carry in: add 1
            core[1, 0, 1, 0] = 1.0  # 0 + 1 = 1, no carry out
            core[1, 1, 0, 1] = 1.0  # 1 + 1 = 0, carry out
        
        cores.append(core)
    
    return MPO(cores=cores, num_sites=num_sites)


def derivative_mpo(
    num_sites: int,
    dx: float,
    order: int = 1,
    dtype: torch.dtype = torch.float64
) -> MPO:
    """
    Create central difference derivative MPO.
    
    First derivative: (f[i+1] - f[i-1]) / (2*dx)
    Second derivative: (f[i+1] - 2*f[i] + f[i-1]) / (dx^2)
    
    Args:
        num_sites: Number of sites
        dx: Grid spacing
        order: Derivative order (1 or 2)
        dtype: Data type
    
    Returns:
        Derivative MPO
    
    Note:
        This is a sum of shift operators. For production use,
        consider using the Fourier-based derivative instead.
    """
    if order == 1:
        # D = (S+ - S-) / (2*dx)
        S_plus = shift_mpo(num_sites, direction=1, dtype=dtype)
        S_minus = shift_mpo(num_sites, direction=-1, dtype=dtype)
        
        # Combine: coeffs * [S+, -S-]
        # This requires MPO addition, which we approximate here
        coeff = 1.0 / (2.0 * dx)
        
        # Scale the operators
        for i, core in enumerate(S_plus.cores):
            if i == 0:
                S_plus.cores[i] = core * coeff
        
        for i, core in enumerate(S_minus.cores):
            if i == 0:
                S_minus.cores[i] = core * (-coeff)
        
        return _mpo_add(S_plus, S_minus)
    
    elif order == 2:
        # D2 = (S+ - 2*I + S-) / dx^2
        S_plus = shift_mpo(num_sites, direction=1, dtype=dtype)
        S_minus = shift_mpo(num_sites, direction=-1, dtype=dtype)
        I = identity_mpo(num_sites, dtype=dtype)
        
        coeff = 1.0 / (dx * dx)
        
        for i, core in enumerate(S_plus.cores):
            if i == 0:
                S_plus.cores[i] = core * coeff
        
        for i, core in enumerate(S_minus.cores):
            if i == 0:
                S_minus.cores[i] = core * coeff
        
        for i, core in enumerate(I.cores):
            if i == 0:
                I.cores[i] = core * (-2.0 * coeff)
        
        result = _mpo_add(S_plus, S_minus)
        result = _mpo_add(result, I)
        return result
    
    else:
        raise ValueError(f"Derivative order must be 1 or 2, got {order}")


def laplacian_mpo(
    num_sites: int,
    dx: float,
    dtype: torch.dtype = torch.float64
) -> MPO:
    """
    Create Laplacian (second derivative) MPO.
    
    Equivalent to derivative_mpo(num_sites, dx, order=2).
    
    Args:
        num_sites: Number of sites
        dx: Grid spacing
        dtype: Data type
    
    Returns:
        Laplacian MPO
    """
    return derivative_mpo(num_sites, dx, order=2, dtype=dtype)


def _mpo_add(a: MPO, b: MPO) -> MPO:
    """
    Add two MPOs by direct sum of bond spaces.
    
    Internal function for combining operators.
    """
    if a.num_sites != b.num_sites:
        raise ValueError("MPOs must have same number of sites")
    
    n = a.num_sites
    new_cores = []
    
    for i in range(n):
        ca, cb = a.cores[i], b.cores[i]
        ra_left, d1, d2, ra_right = ca.shape
        rb_left, _, _, rb_right = cb.shape
        
        if i == 0:
            # First: concatenate on right bond
            new_core = torch.cat([ca, cb], dim=3)
        elif i == n - 1:
            # Last: concatenate on left bond
            new_core = torch.cat([ca, cb], dim=0)
        else:
            # Middle: block diagonal
            new_core = torch.zeros(
                ra_left + rb_left, d1, d2, ra_right + rb_right,
                dtype=ca.dtype, device=ca.device
            )
            new_core[:ra_left, :, :, :ra_right] = ca
            new_core[ra_left:, :, :, ra_right:] = cb
        
        new_cores.append(new_core)
    
    return MPO(cores=new_cores, num_sites=n)


def apply_mpo(
    mpo: MPO,
    qtt: QTTState,
    max_bond: Optional[int] = None,
    tol: float = 1e-12
) -> QTTState:
    """
    Apply an MPO to a QTT state.
    
    The result has bond dimension r_mpo * r_qtt before truncation.
    
    Args:
        mpo: Matrix Product Operator to apply
        qtt: QTT state
        max_bond: Maximum bond dimension for result
        tol: Relative tolerance for truncation
    
    Returns:
        QTT state = MPO @ qtt
    
    Complexity:
        O(n * r_mpo^2 * r_qtt^2 * d^2) without truncation
    
    Example:
        >>> laplacian = laplacian_mpo(20, dx=0.001)
        >>> d2f = apply_mpo(laplacian, f_qtt, max_bond=64)
    """
    if mpo.num_sites != qtt.num_qubits:
        raise ValueError(
            f"MPO sites ({mpo.num_sites}) must match QTT qubits ({qtt.num_qubits})"
        )
    
    n = qtt.num_qubits
    new_cores = []
    
    for i in range(n):
        mpo_core = mpo.cores[i]  # (rm_left, d, d, rm_right)
        qtt_core = qtt.cores[i]  # (rq_left, d, rq_right)
        
        rm_left, d_out, d_in, rm_right = mpo_core.shape
        rq_left, d_phys, rq_right = qtt_core.shape
        
        # Contract on physical index
        # new[rm_left, rq_left, d_out, rm_right, rq_right]
        # = sum_d_in mpo[rm_left, d_out, d_in, rm_right] * qtt[rq_left, d_in, rq_right]
        
        contracted = torch.einsum('abcd,edf->aebcf', mpo_core, qtt_core)
        
        # Reshape to (rm_left * rq_left, d_out, rm_right * rq_right)
        new_core = contracted.reshape(
            rm_left * rq_left,
            d_out,
            rm_right * rq_right
        )
        
        new_cores.append(new_core)
    
    result = QTTState(cores=new_cores, num_qubits=n)
    
    if max_bond is not None:
        from qtt_sdk.operations import truncate_qtt
        result = truncate_qtt(result, max_bond, tol)
    
    return result


def mpo_to_matrix(mpo: MPO) -> Tensor:
    """
    Convert MPO to dense matrix (for small systems only).
    
    Warning: This creates a 2^n x 2^n matrix.
    Only use for testing with n <= 12 or so.
    
    Args:
        mpo: MPO to convert
    
    Returns:
        Dense matrix of shape (2^n, 2^n)
    """
    n = mpo.num_sites
    if n > 15:
        raise MemoryError(f"Matrix would be 2^{n} x 2^{n} = too large")
    
    result = mpo.cores[0]  # (1, d, d, r)
    
    for core in mpo.cores[1:]:
        # result: (..., r)
        # core: (r, d, d, r')
        result = torch.einsum('...i,ijkl->...jkl', result, core)
    
    # result: (1, d, d, d, d, ..., 1)
    # Reshape to (2^n, 2^n)
    dim = 2 ** n
    return result.squeeze(0).squeeze(-1).reshape(dim, dim)
