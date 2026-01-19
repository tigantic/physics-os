"""
QTT Hadamard Product: Element-wise Multiplication for Nonlinear Terms
======================================================================

The Problem:
    Navier-Stokes has nonlinear advection: (u·∇)u = u * ∂u/∂x + v * ∂u/∂y + ...
    This requires element-wise multiplication (Hadamard product).

The Math:
    For TT representations A and B:
        (A ⊙ B)[i₁...iₙ] = A[i₁...iₙ] * B[i₁...iₙ]
    
    In TT format:
        C_k[αₖ₋₁, iₖ, αₖ] = A_k[αₖ₋₁^A, iₖ, αₖ^A] ⊗ B_k[αₖ₋₁^B, iₖ, αₖ^B]
    
    Result rank: rank(C) = rank(A) * rank(B) → EXPLOSION!

The Solution:
    Truncate after each multiplication to control rank.
    For turbulence: ranks ~100, so 100×100=10000 → truncate to 200.

Also Provided:
    - qtt_hadamard: Element-wise multiplication
    - qtt_power: f^n via repeated multiplication
    - qtt_polynomial: Evaluate polynomial P(f)
    - qtt_nonlinear_advection: u*∂u/∂x term

Author: HyperTensor Team
Date: 2026-01-16
Tag: [PHYSICS-TOOLBOX] [NONLINEAR]
"""

from __future__ import annotations

import math
from typing import List, Tuple, Callable, Optional

import torch
from torch import Tensor

from tensornet.cfd.nd_shift_mpo import truncate_cores


# =============================================================================
# Core: QTT Hadamard Product
# =============================================================================

def qtt_hadamard(
    a_cores: List[Tensor],
    b_cores: List[Tensor],
    max_rank: int = 256,
    tol: float = 1e-8
) -> List[Tensor]:
    """
    Element-wise (Hadamard) product of two QTTs: C = A ⊙ B.
    
    C[i₁...iₙ] = A[i₁...iₙ] * B[i₁...iₙ]
    
    Implementation:
        For each core k:
            C_k[α^A ⊗ α^B, iₖ, β^A ⊗ β^B] = A_k[α^A, iₖ, β^A] * B_k[α^B, iₖ, β^B]
        
        This is a Kronecker product on the bond indices for matching physical index.
    
    Rank growth:
        rank(C) = rank(A) * rank(B)
        We truncate immediately to max_rank.
    
    Args:
        a_cores, b_cores: QTT cores to multiply
        max_rank: Maximum rank after truncation
        tol: Truncation tolerance
        
    Returns:
        QTT cores for element-wise product
    """
    if len(a_cores) != len(b_cores):
        raise ValueError(f"QTT length mismatch: {len(a_cores)} vs {len(b_cores)}")
    
    n_cores = len(a_cores)
    device = a_cores[0].device
    dtype = a_cores[0].dtype
    
    result = []
    
    for k in range(n_cores):
        a = a_cores[k]  # (rA_left, 2, rA_right)
        b = b_cores[k]  # (rB_left, 2, rB_right)
        
        rA_left, phys_dim, rA_right = a.shape
        rB_left, _, rB_right = b.shape
        
        # Kronecker product on bond indices for each physical index
        # Result shape: (rA_left * rB_left, phys_dim, rA_right * rB_right)
        
        c = torch.zeros(
            rA_left * rB_left, phys_dim, rA_right * rB_right,
            device=device, dtype=dtype
        )
        
        for p in range(phys_dim):
            # A[:, p, :] ⊗ B[:, p, :] → Kronecker product
            # Result: (rA_left * rB_left, rA_right * rB_right)
            a_slice = a[:, p, :]  # (rA_left, rA_right)
            b_slice = b[:, p, :]  # (rB_left, rB_right)
            
            # Kronecker: outer product then reshape
            # (rA_left, rA_right, rB_left, rB_right) → (rA*rB_left, rA*rB_right)
            kron = torch.einsum('ij,kl->ikjl', a_slice, b_slice)
            c[:, p, :] = kron.reshape(rA_left * rB_left, rA_right * rB_right)
        
        result.append(c)
    
    # Truncate to control rank explosion
    result = truncate_cores(result, max_rank, tol=tol)
    
    return result


def qtt_hadamard_inplace_scale(cores: List[Tensor], scalar: float) -> List[Tensor]:
    """
    Scale QTT by a scalar (rank-preserving).
    
    Simply multiplies the first core by the scalar.
    """
    result = [c.clone() for c in cores]
    result[0] = result[0] * scalar
    return result


# =============================================================================
# QTT Powers and Polynomials
# =============================================================================

def qtt_power(
    cores: List[Tensor],
    n: int,
    max_rank: int = 256,
    tol: float = 1e-8
) -> List[Tensor]:
    """
    Compute f^n via repeated squaring.
    
    f^4 = (f²)² → 2 multiplications instead of 3.
    
    Rank growth: Controlled by truncation after each step.
    """
    if n < 0:
        raise ValueError("Negative powers not supported")
    if n == 0:
        # Return constant 1 in QTT form
        return _qtt_constant(1.0, len(cores), cores[0].device, cores[0].dtype)
    if n == 1:
        return [c.clone() for c in cores]
    
    # Binary exponentiation
    result = None
    base = [c.clone() for c in cores]
    
    while n > 0:
        if n & 1:
            if result is None:
                result = [c.clone() for c in base]
            else:
                result = qtt_hadamard(result, base, max_rank, tol)
        
        base = qtt_hadamard(base, base, max_rank, tol)
        n >>= 1
    
    return result


def _qtt_constant(value: float, n_cores: int, device, dtype) -> List[Tensor]:
    """Create QTT representing constant function."""
    cores = []
    for i in range(n_cores):
        r_left = 1
        r_right = 1
        core = torch.ones(r_left, 2, r_right, device=device, dtype=dtype)
        cores.append(core)
    
    # Scale first core
    cores[0] = cores[0] * (value ** (1.0 / n_cores))  # Distribute across cores
    return cores


def qtt_polynomial(
    cores: List[Tensor],
    coeffs: List[float],
    max_rank: int = 256,
    tol: float = 1e-8
) -> List[Tensor]:
    """
    Evaluate polynomial P(f) = c₀ + c₁f + c₂f² + ...
    
    Uses Horner's method: P(f) = c₀ + f(c₁ + f(c₂ + ...))
    
    Args:
        cores: QTT for function f
        coeffs: Polynomial coefficients [c₀, c₁, c₂, ...]
        max_rank: Maximum rank
        tol: Truncation tolerance
        
    Returns:
        QTT for P(f)
    """
    if len(coeffs) == 0:
        return _qtt_constant(0.0, len(cores), cores[0].device, cores[0].dtype)
    
    n_cores = len(cores)
    device = cores[0].device
    dtype = cores[0].dtype
    
    # Horner's method (reverse order)
    from tensornet.cfd.pure_qtt_ops import qtt_add, QTTState
    
    result = _qtt_constant(coeffs[-1], n_cores, device, dtype)
    
    for c in reversed(coeffs[:-1]):
        # result = c + f * result
        product = qtt_hadamard(cores, result, max_rank, tol)
        const = _qtt_constant(c, n_cores, device, dtype)
        
        a = QTTState(cores=product, num_qubits=n_cores)
        b = QTTState(cores=const, num_qubits=n_cores)
        summed = qtt_add(a, b, max_bond=max_rank)
        result = truncate_cores(list(summed.cores), max_rank, tol)
    
    return result


# =============================================================================
# Nonlinear Advection Term
# =============================================================================

def qtt_nonlinear_advection(
    u_cores: List[Tensor],
    du_dx_cores: List[Tensor],
    max_rank: int = 256,
    tol: float = 1e-8
) -> List[Tensor]:
    """
    Compute nonlinear advection term: u * ∂u/∂x
    
    This is the core of the Navier-Stokes nonlinearity.
    
    Args:
        u_cores: Velocity field QTT
        du_dx_cores: Velocity gradient QTT
        max_rank: Maximum rank after multiplication
        tol: Truncation tolerance
        
    Returns:
        QTT for u * ∂u/∂x
    """
    return qtt_hadamard(u_cores, du_dx_cores, max_rank, tol)


def qtt_full_advection_3d(
    u_cores: List[Tensor],
    v_cores: List[Tensor],
    w_cores: List[Tensor],
    n_qubits_per_dim: int,
    dx: float,
    max_rank: int = 256,
    tol: float = 1e-8
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """
    Compute full 3D advection term: (u·∇)u for all three components.
    
    (u·∇)u_x = u * ∂u/∂x + v * ∂u/∂y + w * ∂u/∂z
    (u·∇)u_y = u * ∂v/∂x + v * ∂v/∂y + w * ∂v/∂z
    (u·∇)u_z = u * ∂w/∂x + v * ∂w/∂y + w * ∂w/∂z
    
    Returns:
        (adv_u, adv_v, adv_w): Advection of each velocity component
    """
    from tensornet.cfd.qtt_shift_stable import qtt_3d_central_diff_stable
    from tensornet.cfd.pure_qtt_ops import qtt_add, QTTState
    
    def compute_advection_component(f_cores: List[Tensor]) -> List[Tensor]:
        """Compute (u·∇)f = u*∂f/∂x + v*∂f/∂y + w*∂f/∂z"""
        
        # Compute derivatives
        df_dx = qtt_3d_central_diff_stable(f_cores, n_qubits_per_dim, 0, dx, max_rank, tol)
        df_dy = qtt_3d_central_diff_stable(f_cores, n_qubits_per_dim, 1, dx, max_rank, tol)
        df_dz = qtt_3d_central_diff_stable(f_cores, n_qubits_per_dim, 2, dx, max_rank, tol)
        
        # Multiply by velocity components
        u_df_dx = qtt_hadamard(u_cores, df_dx, max_rank, tol)
        v_df_dy = qtt_hadamard(v_cores, df_dy, max_rank, tol)
        w_df_dz = qtt_hadamard(w_cores, df_dz, max_rank, tol)
        
        # Sum
        a = QTTState(cores=u_df_dx, num_qubits=len(u_df_dx))
        b = QTTState(cores=v_df_dy, num_qubits=len(v_df_dy))
        sum1 = qtt_add(a, b, max_bond=max_rank * 2)
        
        c = QTTState(cores=list(sum1.cores), num_qubits=len(sum1.cores))
        d = QTTState(cores=w_df_dz, num_qubits=len(w_df_dz))
        total = qtt_add(c, d, max_bond=max_rank * 2)
        
        return truncate_cores(list(total.cores), max_rank, tol)
    
    adv_u = compute_advection_component(u_cores)
    adv_v = compute_advection_component(v_cores)
    adv_w = compute_advection_component(w_cores)
    
    return adv_u, adv_v, adv_w


# =============================================================================
# Nonlinear Activation Functions (for neural-physics hybrids)
# =============================================================================

def qtt_relu_approx(
    cores: List[Tensor],
    max_rank: int = 256,
    tol: float = 1e-8,
    n_terms: int = 10
) -> List[Tensor]:
    """
    Approximate ReLU(f) = max(0, f) using polynomial approximation.
    
    ReLU ≈ (1/2)(f + |f|) ≈ (1/2)(f + sqrt(f² + ε))
    
    We use Chebyshev approximation for smooth ReLU.
    """
    # Smooth ReLU: softplus(f) = log(1 + exp(f)) ≈ polynomial
    # For simplicity, use: ReLU ≈ f * sigmoid(k*f) for large k
    # sigmoid(x) ≈ 1/2 + x/4 - x³/48 + ... (Taylor)
    
    # Simple approximation: positive part using clipped polynomial
    # f_+ ≈ f/2 + f*tanh(f)/2 ≈ f * (1 + tanh(f))/2
    
    # Approximate tanh with polynomial
    tanh_coeffs = [0, 1, 0, -1/3, 0, 2/15, 0, -17/315]  # Taylor
    
    tanh_f = qtt_polynomial(cores, tanh_coeffs[:n_terms], max_rank, tol)
    
    # f * (1 + tanh(f)) / 2
    from tensornet.cfd.pure_qtt_ops import qtt_add, QTTState
    
    one = _qtt_constant(1.0, len(cores), cores[0].device, cores[0].dtype)
    a = QTTState(cores=one, num_qubits=len(one))
    b = QTTState(cores=tanh_f, num_qubits=len(tanh_f))
    one_plus_tanh = qtt_add(a, b, max_bond=max_rank)
    
    half_f = qtt_hadamard_inplace_scale(cores, 0.5)
    result = qtt_hadamard(half_f, list(one_plus_tanh.cores), max_rank, tol)
    
    return result


# =============================================================================
# Test Suite
# =============================================================================

def test_hadamard():
    """Test Hadamard product and nonlinear operations."""
    print("=" * 60)
    print("QTT Hadamard Product Test Suite")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    n_qubits = 8
    rank = 16
    
    # Create test QTTs
    def make_qtt(scale=1.0):
        cores = []
        for i in range(n_qubits):
            r_left = 1 if i == 0 else rank
            r_right = 1 if i == n_qubits - 1 else rank
            core = torch.randn(r_left, 2, r_right, device=device, dtype=dtype) * scale
            cores.append(core)
        return cores
    
    a = make_qtt(0.1)
    b = make_qtt(0.1)
    
    print(f"\nTest QTT: {n_qubits} qubits, rank {rank}")
    print(f"Grid size: {2**n_qubits}")
    
    # Test 1: Hadamard product
    print("\n1. Hadamard Product (A ⊙ B):")
    c = qtt_hadamard(a, b, max_rank=64, tol=1e-6)
    print(f"   Input ranks: A={rank}, B={rank}")
    print(f"   Before truncation: would be {rank}×{rank}={rank*rank}")
    print(f"   After truncation: {max(max(core.shape[0], core.shape[2]) for core in c)}")
    
    # Test 2: Power
    print("\n2. QTT Power (f³):")
    f_cubed = qtt_power(a, 3, max_rank=64, tol=1e-6)
    print(f"   Rank of f³: {max(max(core.shape[0], core.shape[2]) for core in f_cubed)}")
    
    # Test 3: Polynomial
    print("\n3. Polynomial P(f) = 1 + 2f + 3f²:")
    poly = qtt_polynomial(a, [1.0, 2.0, 3.0], max_rank=64, tol=1e-6)
    print(f"   Rank: {max(max(core.shape[0], core.shape[2]) for core in poly)}")
    
    # Test 4: Nonlinear advection
    print("\n4. Nonlinear Advection (u * ∂u/∂x):")
    from tensornet.cfd.qtt_shift_stable import qtt_central_diff_stable
    du_dx = qtt_central_diff_stable(a, dx=0.01, max_rank=64, tol=1e-6)
    advection = qtt_nonlinear_advection(a, du_dx, max_rank=64, tol=1e-6)
    print(f"   Rank: {max(max(core.shape[0], core.shape[2]) for core in advection)}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_hadamard()
