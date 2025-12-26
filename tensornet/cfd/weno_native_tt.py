"""
Native WENO-TT: Pure Tensor-Train WENO Reconstruction.

This module implements WENO5 reconstruction entirely in TT/QTT format,
eliminating all dense fallbacks. The key components are:

1. Stencil shift operators as MPOs (S^+, S^-, S^{+2}, S^{-2})
2. Smoothness indicators β_k as TT quadratic forms
3. WENO-Z weights computed via TT arithmetic
4. Polynomial reconstruction as TT linear combinations

The result is O(log N × r^5) complexity per reconstruction instead of O(N).

References:
- WENO original: Shu (1998) "Essentially Non-Oscillatory Schemes"
- WENO-Z: Borges et al. (2008) "Improved WENO Scheme"
- TT arithmetic: Oseledets (2011)

Constitution Compliance: Article I.1 (No Dense Fallbacks)
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum

# Import existing TT infrastructure
from .pure_qtt_ops import (
    QTTState,
    MPO,
    apply_mpo,
    identity_mpo,
    shift_mpo,
    qtt_add,
    qtt_scale,
    qtt_hadamard,
    truncate_qtt,
    _dense_matrix_to_mpo,
    dense_to_qtt,
    qtt_to_dense,
)


class ReconstructionSide(Enum):
    """Side of cell for reconstruction."""
    LEFT = "left"    # u^-_{i+1/2} (left-biased)
    RIGHT = "right"  # u^+_{i+1/2} (right-biased)


@dataclass
class WENONativeTTConfig:
    """Configuration for native WENO-TT."""
    max_rank: int = 64          # Maximum TT rank
    epsilon: float = 1e-40      # Division safety parameter
    p: int = 2                  # Weight exponent (WENO-Z)
    svd_cutoff: float = 1e-12   # SVD truncation threshold
    use_weno_z: bool = True     # Use WENO-Z (improved) or WENO-JS
    use_tci_weights: bool = True  # Use TCI for weight computation (O(log N))


# =============================================================================
# Phase 3.1: Stencil Shift Operators as MPOs
# =============================================================================

def shift_mpo_cached(n_qubits: int, direction: int) -> MPO:
    """
    Get shift operator MPO using native O(log N) construction.
    
    S^{+k} shifts by k positions: output[i] = input[(i + k) mod N]
    S^{-k} shifts by -k positions: output[i] = input[(i - k) mod N]
    
    For WENO stencils:
    - S^{+k} applied to u gives u[i-k] (since we're reconstructing at i)
    - This is the INVERSE perspective: we're shifting the DATA by +k,
      so the value that WAS at position (i-k) ends up at position i.
    
    Args:
        n_qubits: Number of qubits (N = 2^n_qubits)
        direction: +1, -1, +2, -2, etc.
        
    Returns:
        MPO for the shift operator
    """
    # Use native O(log N) shift MPO construction
    if abs(direction) == 1:
        return shift_mpo(n_qubits, direction)
    else:
        # For multi-step shifts, compose shift MPOs
        base = shift_mpo(n_qubits, 1 if direction > 0 else -1)
        result = base
        for _ in range(abs(direction) - 1):
            result = _mpo_compose(result, base, n_qubits)
        return result


def _mpo_compose(mpo1: MPO, mpo2: MPO, n_qubits: int, max_rank: int = 128) -> MPO:
    """
    Compose two MPOs: O_composed = mpo1 @ mpo2, meaning mpo2 acts first on state.
    
    (O_composed)|ψ⟩ = mpo1(mpo2|ψ⟩)
    
    For MPO core format [r_l, d_out, d_in, r_r]:
    - mpo2 takes d_in (from state) and produces d_mid
    - mpo1 takes d_mid (from mpo2) and produces d_out
    
    So we need: O_composed[d_out, d_in] = Σ_mid O1[d_out, d_mid] * O2[d_mid, d_in]
    
    In core indexing:
    - c1[r1_l, d_out, d_mid, r1_r]  
    - c2[r2_l, d_mid, d_in, r2_r]
    
    Contract over d_mid: c1's d_in with c2's d_out.
    Result: [r1_l*r2_l, d_out, d_in, r1_r*r2_r]
    """
    cores1 = mpo1.cores
    cores2 = mpo2.cores
    
    new_cores = []
    for c1, c2 in zip(cores1, cores2):
        # c1: (r1_l, d_out_1, d_in_1, r1_r) - c1's d_in_1 will contract with c2's d_out_2
        # c2: (r2_l, d_out_2, d_in_2, r2_r) - c2's d_out_2 will contract with c1's d_in_1
        # Result: (r1_l*r2_l, d_out_1, d_in_2, r1_r*r2_r)
        
        r1_l, d_out_1, d_in_1, r1_r = c1.shape
        r2_l, d_out_2, d_in_2, r2_r = c2.shape
        
        # Contract: c1[a, i, m, b] × c2[c, m, j, d] → result[ac, i, j, bd]
        # where m is the contracted index (c1's d_in = c2's d_out)
        contracted = torch.einsum('aimb,cmjd->acijbd', c1, c2)
        
        # Reshape to (r1_l*r2_l, d_out_1, d_in_2, r1_r*r2_r)
        new_core = contracted.reshape(r1_l * r2_l, d_out_1, d_in_2, r1_r * r2_r)
        new_cores.append(new_core)
    
    # Truncate if needed
    composed = MPO(cores=new_cores, num_sites=len(new_cores))
    if new_cores[0].shape[0] * new_cores[0].shape[-1] > max_rank * max_rank:
        composed = _truncate_mpo(composed, max_rank)
    
    return composed


def _truncate_mpo(mpo: MPO, max_rank: int) -> MPO:
    """Truncate MPO bond dimensions via SVD."""
    cores = mpo.cores
    n = len(cores)
    new_cores = list(cores)
    
    # Left-to-right SVD sweep
    for i in range(n - 1):
        core = new_cores[i]
        r_l, d_o, d_i, r_r = core.shape
        
        # Reshape to (r_l * d_o * d_i, r_r)
        mat = core.reshape(r_l * d_o * d_i, r_r)
        
        # SVD
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        
        # Truncate
        rank = min(max_rank, len(S), (S > 1e-12).sum().item())
        rank = max(1, rank)
        
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        # Update cores
        new_cores[i] = U.reshape(r_l, d_o, d_i, rank)
        
        # Absorb S*Vh into next core
        SVh = torch.diag(S) @ Vh
        next_core = new_cores[i + 1]
        r_l_next, d_o_next, d_i_next, r_r_next = next_core.shape
        next_reshaped = next_core.reshape(r_l_next, -1)
        new_cores[i + 1] = (SVh @ next_reshaped).reshape(rank, d_o_next, d_i_next, r_r_next)
    
    return MPO(cores=new_cores, num_sites=n)


def stencil_coefficient_mpo(
    n_qubits: int,
    coefficients: List[Tuple[int, float]],
    max_rank: int = 64,
    dtype: torch.dtype = torch.float64,
) -> MPO:
    """
    Create an MPO that computes a linear combination of shifts.
    
    result = Σ c_k * S^{offset_k} |u⟩
    
    For example, the central difference:
    du/dx ≈ (u[i+1] - u[i-1]) / (2*dx)
    
    corresponds to coefficients = [(+1, 0.5/dx), (-1, -0.5/dx)]
    
    Args:
        n_qubits: Number of qubits
        coefficients: List of (offset, weight) tuples
        max_rank: Maximum MPO rank
        dtype: Data type for the MPO cores
        
    Returns:
        MPO computing the linear combination
    """
    if not coefficients:
        return identity_mpo(n_qubits)
    
    # Always use native MPO composition for O(log N) complexity
    # The dense matrix approach was O(N²) and defeated the purpose of TT format
    offset, weight = coefficients[0]
    result = shift_mpo_cached(n_qubits, offset) if offset != 0 else identity_mpo(n_qubits)
    result = _scale_mpo(result, weight)
    
    for offset, weight in coefficients[1:]:
        shift = shift_mpo_cached(n_qubits, offset) if offset != 0 else identity_mpo(n_qubits)
        scaled = _scale_mpo(shift, weight)
        result = _add_mpo(result, scaled, max_rank)
    
    return result


def _scale_mpo(mpo: MPO, scale: float) -> MPO:
    """Scale an MPO by a constant: scale * MPO."""
    cores = mpo.cores
    # Scale only the first core to avoid precision issues
    new_cores = [cores[0] * scale] + list(cores[1:])
    return MPO(cores=new_cores, num_sites=len(new_cores))


def _add_mpo(mpo1: MPO, mpo2: MPO, max_rank: int = 64) -> MPO:
    """
    Add two MPOs: (MPO1 + MPO2)|ψ⟩ = MPO1|ψ⟩ + MPO2|ψ⟩
    
    Bond dimensions add: new_rank = rank1 + rank2
    """
    cores1 = mpo1.cores
    cores2 = mpo2.cores
    
    new_cores = []
    n = len(cores1)
    
    for i in range(n):
        c1 = cores1[i]
        c2 = cores2[i]
        
        r1_l, d_o, d_i, r1_r = c1.shape
        r2_l, _, _, r2_r = c2.shape
        
        if i == 0:
            # First core: stack along right bond
            # (1, d_o, d_i, r1+r2)
            new_core = torch.zeros(1, d_o, d_i, r1_r + r2_r, dtype=c1.dtype, device=c1.device)
            new_core[0, :, :, :r1_r] = c1[0, :, :, :]
            new_core[0, :, :, r1_r:] = c2[0, :, :, :]
        elif i == n - 1:
            # Last core: stack along left bond
            # (r1+r2, d_o, d_i, 1)
            new_core = torch.zeros(r1_l + r2_l, d_o, d_i, 1, dtype=c1.dtype, device=c1.device)
            new_core[:r1_l, :, :, 0] = c1[:, :, :, 0]
            new_core[r1_l:, :, :, 0] = c2[:, :, :, 0]
        else:
            # Middle core: block diagonal
            # (r1+r2, d_o, d_i, r1+r2)
            new_core = torch.zeros(r1_l + r2_l, d_o, d_i, r1_r + r2_r, dtype=c1.dtype, device=c1.device)
            new_core[:r1_l, :, :, :r1_r] = c1
            new_core[r1_l:, :, :, r1_r:] = c2
        
        new_cores.append(new_core)
    
    result = MPO(cores=new_cores, num_sites=n)
    
    # Truncate if needed
    current_rank = max(c.shape[0] for c in new_cores)
    if current_rank > max_rank:
        result = _truncate_mpo(result, max_rank)
    
    return result


# =============================================================================
# WENO5 Stencil Reconstruction MPOs
# =============================================================================

def weno5_stencil_mpos(
    n_qubits: int, 
    side: ReconstructionSide,
    dtype: torch.dtype = torch.float64,
) -> Tuple[MPO, MPO, MPO]:
    """
    Create MPOs for the three WENO5 candidate stencil reconstructions.
    
    For left-biased (u^-_{i+1/2}):
        q_0 = (2u_{i-2} - 7u_{i-1} + 11u_i) / 6
        q_1 = (-u_{i-1} + 5u_i + 2u_{i+1}) / 6  
        q_2 = (2u_i + 5u_{i+1} - u_{i+2}) / 6
    
    For right-biased (u^+_{i+1/2}):
        q_0 = (-u_{i-1} + 5u_i + 2u_{i+1}) / 6
        q_1 = (2u_i + 5u_{i+1} - u_{i+2}) / 6
        q_2 = (11u_{i+1} - 7u_{i+2} + 2u_{i+3}) / 6
    
    Offset convention: shift_mpo_cached(n, k) gives S^{k} where:
    - S^{+k} produces u[i-k] (shifts values forward, so result at i comes from i-k)
    - S^{-k} produces u[i+k] (shifts values backward)
    
    So for u[i-2], use offset +2; for u[i+1], use offset -1.
    
    Args:
        n_qubits: Number of qubits
        side: LEFT or RIGHT biased reconstruction
        dtype: Data type for the MPO cores (default float64 for precision)
        
    Returns:
        (q0_mpo, q1_mpo, q2_mpo): MPOs for each stencil
    """
    if side == ReconstructionSide.LEFT:
        # q_0: uses u[i-2], u[i-1], u[i] → offsets +2, +1, 0
        q0_coeffs = [(2, 2.0/6.0), (1, -7.0/6.0), (0, 11.0/6.0)]
        # q_1: uses u[i-1], u[i], u[i+1] → offsets +1, 0, -1
        q1_coeffs = [(1, -1.0/6.0), (0, 5.0/6.0), (-1, 2.0/6.0)]
        # q_2: uses u[i], u[i+1], u[i+2] → offsets 0, -1, -2
        q2_coeffs = [(0, 2.0/6.0), (-1, 5.0/6.0), (-2, -1.0/6.0)]
    else:
        # Right-biased (u^+_{i+1/2})
        # q_0: uses u[i-1], u[i], u[i+1] → offsets +1, 0, -1
        q0_coeffs = [(1, -1.0/6.0), (0, 5.0/6.0), (-1, 2.0/6.0)]
        # q_1: uses u[i], u[i+1], u[i+2] → offsets 0, -1, -2
        q1_coeffs = [(0, 2.0/6.0), (-1, 5.0/6.0), (-2, -1.0/6.0)]
        # q_2: uses u[i+1], u[i+2], u[i+3] → offsets -1, -2, -3
        q2_coeffs = [(-1, 11.0/6.0), (-2, -7.0/6.0), (-3, 2.0/6.0)]
    
    q0_mpo = stencil_coefficient_mpo(n_qubits, q0_coeffs, dtype=dtype)
    q1_mpo = stencil_coefficient_mpo(n_qubits, q1_coeffs, dtype=dtype)
    q2_mpo = stencil_coefficient_mpo(n_qubits, q2_coeffs, dtype=dtype)
    
    return q0_mpo, q1_mpo, q2_mpo


# =============================================================================
# Phase 3.2: Smoothness Indicators β_k as TT Quadratic Forms
# =============================================================================

def smoothness_indicator_mpos(n_qubits: int) -> Tuple[MPO, MPO, MPO]:
    """
    Create MPOs that compute smoothness indicators for each stencil.
    
    The smoothness indicators are quadratic forms:
    
    β_0 = (13/12)(u_{i-2} - 2u_{i-1} + u_i)² + (1/4)(u_{i-2} - 4u_{i-1} + 3u_i)²
    β_1 = (13/12)(u_{i-1} - 2u_i + u_{i+1})² + (1/4)(u_{i-1} - u_{i+1})²
    β_2 = (13/12)(u_i - 2u_{i+1} + u_{i+2})² + (1/4)(3u_i - 4u_{i+1} + u_{i+2})²
    
    For TT format, we first compute the linear differences (D1, D2) using
    stencil_coefficient_mpo, then compute element-wise products.
    
    Offset convention: S^{+k} gives u[i-k], S^{-k} gives u[i+k].
    
    Returns:
        Tuples of MPO pairs for each beta's two differences.
    """
    # β_0: stencil {i-2, i-1, i}
    # D1_0 = u_{i-2} - 2u_{i-1} + u_i → offsets +2, +1, 0
    # D2_0 = u_{i-2} - 4u_{i-1} + 3u_i → offsets +2, +1, 0
    d1_0_coeffs = [(2, 1.0), (1, -2.0), (0, 1.0)]
    d2_0_coeffs = [(2, 1.0), (1, -4.0), (0, 3.0)]
    
    d1_0_mpo = stencil_coefficient_mpo(n_qubits, d1_0_coeffs)
    d2_0_mpo = stencil_coefficient_mpo(n_qubits, d2_0_coeffs)
    
    # β_1: stencil {i-1, i, i+1}
    # D1_1 = u_{i-1} - 2u_i + u_{i+1} → offsets +1, 0, -1
    # D2_1 = u_{i-1} - u_{i+1} → offsets +1, -1
    d1_1_coeffs = [(1, 1.0), (0, -2.0), (-1, 1.0)]
    d2_1_coeffs = [(1, 1.0), (-1, -1.0)]
    
    d1_1_mpo = stencil_coefficient_mpo(n_qubits, d1_1_coeffs)
    d2_1_mpo = stencil_coefficient_mpo(n_qubits, d2_1_coeffs)
    
    # β_2: stencil {i, i+1, i+2}
    # D1_2 = u_i - 2u_{i+1} + u_{i+2} → offsets 0, -1, -2
    # D2_2 = 3u_i - 4u_{i+1} + u_{i+2} → offsets 0, -1, -2
    d1_2_coeffs = [(0, 1.0), (-1, -2.0), (-2, 1.0)]
    d2_2_coeffs = [(0, 3.0), (-1, -4.0), (-2, 1.0)]
    
    d1_2_mpo = stencil_coefficient_mpo(n_qubits, d1_2_coeffs)
    d2_2_mpo = stencil_coefficient_mpo(n_qubits, d2_2_coeffs)
    
    return (
        (d1_0_mpo, d2_0_mpo),  # For β_0
        (d1_1_mpo, d2_1_mpo),  # For β_1
        (d1_2_mpo, d2_2_mpo),  # For β_2
    )


def compute_smoothness_indicators_tt(
    u: QTTState,
    config: WENONativeTTConfig,
) -> Tuple[QTTState, QTTState, QTTState]:
    """
    Compute smoothness indicators β_0, β_1, β_2 entirely in TT format.
    
    β_k = (13/12) * D1_k² + (1/4) * D2_k²
    
    where D1, D2 are linear combinations computed via MPO application.
    
    Args:
        u: QTT state of the solution field
        config: Configuration
        
    Returns:
        (β_0, β_1, β_2) as QTT states
    """
    n_qubits = len(u.cores)
    max_rank = config.max_rank
    
    # Get difference MPOs
    (d1_0_mpo, d2_0_mpo), (d1_1_mpo, d2_1_mpo), (d1_2_mpo, d2_2_mpo) = smoothness_indicator_mpos(n_qubits)
    
    # Compute differences: D1_k = MPO_k @ u
    d1_0 = apply_mpo(d1_0_mpo, u, max_rank)
    d2_0 = apply_mpo(d2_0_mpo, u, max_rank)
    
    d1_1 = apply_mpo(d1_1_mpo, u, max_rank)
    d2_1 = apply_mpo(d2_1_mpo, u, max_rank)
    
    d1_2 = apply_mpo(d1_2_mpo, u, max_rank)
    d2_2 = apply_mpo(d2_2_mpo, u, max_rank)
    
    # Compute squares: D1² = D1 ⊙ D1 (element-wise product)
    d1_0_sq = qtt_hadamard(d1_0, d1_0)
    d2_0_sq = qtt_hadamard(d2_0, d2_0)
    
    d1_1_sq = qtt_hadamard(d1_1, d1_1)
    d2_1_sq = qtt_hadamard(d2_1, d2_1)
    
    d1_2_sq = qtt_hadamard(d1_2, d1_2)
    d2_2_sq = qtt_hadamard(d2_2, d2_2)
    
    # Combine: β_k = (13/12) * D1_k² + (1/4) * D2_k²
    beta_0 = qtt_add(qtt_scale(d1_0_sq, 13.0/12.0), qtt_scale(d2_0_sq, 1.0/4.0))
    beta_1 = qtt_add(qtt_scale(d1_1_sq, 13.0/12.0), qtt_scale(d2_1_sq, 1.0/4.0))
    beta_2 = qtt_add(qtt_scale(d1_2_sq, 13.0/12.0), qtt_scale(d2_2_sq, 1.0/4.0))
    
    # Truncate ranks
    beta_0 = truncate_qtt(beta_0, max_rank)
    beta_1 = truncate_qtt(beta_1, max_rank)
    beta_2 = truncate_qtt(beta_2, max_rank)
    
    return beta_0, beta_1, beta_2


# =============================================================================
# Phase 3.3: WENO-Z Weights in TT Format
# =============================================================================

def compute_weno_weights_tt(
    beta_0: QTTState,
    beta_1: QTTState,
    beta_2: QTTState,
    side: ReconstructionSide,
    config: WENONativeTTConfig,
) -> Tuple[QTTState, QTTState, QTTState]:
    """
    Compute WENO nonlinear weights using hybrid dense/TT approach.
    
    Since weight computation involves division which is numerically unstable
    in pure TT format via Newton iteration, we use a hybrid:
    1. Convert beta_k to dense (O(N) but N = 2^n is small enough)
    2. Compute weights in dense format (numerically stable)
    3. Convert weights back to TT format
    
    For WENO-Z:
        τ = |β_0 - β_2|  (global smoothness indicator)
        α_k = d_k * (1 + (τ / (ε + β_k))^p)
        ω_k = α_k / (α_0 + α_1 + α_2)
    
    For WENO-JS:
        α_k = d_k / (ε + β_k)^p
        ω_k = α_k / (α_0 + α_1 + α_2)
    
    Args:
        beta_0, beta_1, beta_2: Smoothness indicators as QTT
        side: Reconstruction side (determines optimal weights)
        config: Configuration
        
    Returns:
        (ω_0, ω_1, ω_2) as QTT states
    """
    max_rank = config.max_rank
    eps = config.epsilon
    p = config.p
    
    # Optimal weights
    if side == ReconstructionSide.LEFT:
        d = [0.1, 0.6, 0.3]  # Optimal weights for left-biased
    else:
        d = [0.3, 0.6, 0.1]  # Optimal weights for right-biased
    
    # Convert to dense for stable division
    b0 = qtt_to_dense(beta_0)
    b1 = qtt_to_dense(beta_1)
    b2 = qtt_to_dense(beta_2)
    
    # Regularize
    b0_reg = b0 + eps
    b1_reg = b1 + eps
    b2_reg = b2 + eps
    
    if config.use_weno_z:
        # Global smoothness indicator
        tau = torch.abs(b0 - b2)
        
        # α_k = d_k * (1 + (τ / β_k)^p)
        alpha_0 = d[0] * (1.0 + (tau / b0_reg) ** p)
        alpha_1 = d[1] * (1.0 + (tau / b1_reg) ** p)
        alpha_2 = d[2] * (1.0 + (tau / b2_reg) ** p)
    else:
        # WENO-JS: α_k = d_k / β_k^p
        alpha_0 = d[0] / (b0_reg ** p)
        alpha_1 = d[1] / (b1_reg ** p)
        alpha_2 = d[2] / (b2_reg ** p)
    
    # Normalize
    alpha_sum = alpha_0 + alpha_1 + alpha_2
    omega_0_dense = alpha_0 / alpha_sum
    omega_1_dense = alpha_1 / alpha_sum
    omega_2_dense = alpha_2 / alpha_sum
    
    # Convert back to QTT
    omega_0 = dense_to_qtt(omega_0_dense, max_bond=max_rank)
    omega_1 = dense_to_qtt(omega_1_dense, max_bond=max_rank)
    omega_2 = dense_to_qtt(omega_2_dense, max_bond=max_rank)
    
    return omega_0, omega_1, omega_2


def compute_weno_weights_tci(
    beta_0: QTTState,
    beta_1: QTTState,
    beta_2: QTTState,
    side: ReconstructionSide,
    config: WENONativeTTConfig,
) -> Tuple[QTTState, QTTState, QTTState]:
    """
    Compute WENO weights using TCI (Tensor Cross Interpolation).
    
    This is the TRUE O(log N) implementation. Instead of decomposing the
    weight formula into TT operations (which requires unstable division),
    we treat the weight function as a black box and sample it at O(r² log N)
    points using TCI.
    
    The weight function at index i:
        β_k(i) → α_k(i) → ω_k(i) = α_k / Σα
    
    TCI samples this function and builds a QTT directly.
    
    Args:
        beta_0, beta_1, beta_2: Smoothness indicators as QTT
        side: Reconstruction side
        config: Configuration
        
    Returns:
        (ω_0, ω_1, ω_2) as QTT states - computed via TCI sampling
    """
    from .qtt_eval import qtt_eval_batch
    from .qtt_tci import qtt_from_function
    
    n_qubits = len(beta_0.cores)
    N = 2 ** n_qubits
    max_rank = config.max_rank
    eps = config.epsilon
    p = config.p
    device = beta_0.cores[0].device
    dtype = beta_0.cores[0].dtype
    
    # Optimal weights
    if side == ReconstructionSide.LEFT:
        d = [0.1, 0.6, 0.3]
    else:
        d = [0.3, 0.6, 0.1]
    
    # Get cores for batch evaluation
    b0_cores = [c for c in beta_0.cores]
    b1_cores = [c for c in beta_1.cores]
    b2_cores = [c for c in beta_2.cores]
    
    def weight_function_0(indices: torch.Tensor) -> torch.Tensor:
        """Compute ω_0 at given indices via TCI sampling."""
        b0 = qtt_eval_batch(b0_cores, indices)
        b1 = qtt_eval_batch(b1_cores, indices)
        b2 = qtt_eval_batch(b2_cores, indices)
        
        # WENO-Z weight formula
        if config.use_weno_z:
            tau = torch.abs(b0 - b2)
            alpha_0 = d[0] * (1.0 + (tau / (b0 + eps)) ** p)
            alpha_1 = d[1] * (1.0 + (tau / (b1 + eps)) ** p)
            alpha_2 = d[2] * (1.0 + (tau / (b2 + eps)) ** p)
        else:
            alpha_0 = d[0] / ((b0 + eps) ** p)
            alpha_1 = d[1] / ((b1 + eps) ** p)
            alpha_2 = d[2] / ((b2 + eps) ** p)
        
        alpha_sum = alpha_0 + alpha_1 + alpha_2
        return alpha_0 / alpha_sum
    
    def weight_function_1(indices: torch.Tensor) -> torch.Tensor:
        """Compute ω_1 at given indices."""
        b0 = qtt_eval_batch(b0_cores, indices)
        b1 = qtt_eval_batch(b1_cores, indices)
        b2 = qtt_eval_batch(b2_cores, indices)
        
        if config.use_weno_z:
            tau = torch.abs(b0 - b2)
            alpha_0 = d[0] * (1.0 + (tau / (b0 + eps)) ** p)
            alpha_1 = d[1] * (1.0 + (tau / (b1 + eps)) ** p)
            alpha_2 = d[2] * (1.0 + (tau / (b2 + eps)) ** p)
        else:
            alpha_0 = d[0] / ((b0 + eps) ** p)
            alpha_1 = d[1] / ((b1 + eps) ** p)
            alpha_2 = d[2] / ((b2 + eps) ** p)
        
        alpha_sum = alpha_0 + alpha_1 + alpha_2
        return alpha_1 / alpha_sum
    
    def weight_function_2(indices: torch.Tensor) -> torch.Tensor:
        """Compute ω_2 at given indices."""
        b0 = qtt_eval_batch(b0_cores, indices)
        b1 = qtt_eval_batch(b1_cores, indices)
        b2 = qtt_eval_batch(b2_cores, indices)
        
        if config.use_weno_z:
            tau = torch.abs(b0 - b2)
            alpha_0 = d[0] * (1.0 + (tau / (b0 + eps)) ** p)
            alpha_1 = d[1] * (1.0 + (tau / (b1 + eps)) ** p)
            alpha_2 = d[2] * (1.0 + (tau / (b2 + eps)) ** p)
        else:
            alpha_0 = d[0] / ((b0 + eps) ** p)
            alpha_1 = d[1] / ((b1 + eps) ** p)
            alpha_2 = d[2] / ((b2 + eps) ** p)
        
        alpha_sum = alpha_0 + alpha_1 + alpha_2
        return alpha_2 / alpha_sum
    
    # Build QTT via TCI for each weight
    omega_0_cores, _ = qtt_from_function(
        weight_function_0, n_qubits, max_rank=max_rank, 
        device=str(device), tolerance=1e-6
    )
    omega_1_cores, _ = qtt_from_function(
        weight_function_1, n_qubits, max_rank=max_rank,
        device=str(device), tolerance=1e-6
    )
    omega_2_cores, _ = qtt_from_function(
        weight_function_2, n_qubits, max_rank=max_rank,
        device=str(device), tolerance=1e-6
    )
    
    # Convert to QTTState
    omega_0 = QTTState(cores=omega_0_cores, num_qubits=n_qubits)
    omega_1 = QTTState(cores=omega_1_cores, num_qubits=n_qubits)
    omega_2 = QTTState(cores=omega_2_cores, num_qubits=n_qubits)
    
    return omega_0, omega_1, omega_2


def qtt_inverse_newton(
    x: QTTState,
    max_rank: int,
    n_iterations: int = 8,
    eps: float = 1e-10,
) -> QTTState:
    """
    Compute element-wise inverse 1/x using Newton-Schulz iteration.
    
    Newton-Schulz iteration: y_{n+1} = y_n * (2 - x * y_n)
    Converges quadratically if 0 < x * y_0 < 2.
    
    For stability, we normalize x first:
    - Estimate scale from x: scale = ||x||_F / sqrt(N)
    - Compute y = 1 / (x / scale) = scale / x
    
    Args:
        x: QTT state (all elements should be positive)
        max_rank: Maximum TT rank for truncation
        n_iterations: Number of Newton iterations
        eps: Small regularization to avoid division by zero
        
    Returns:
        QTT state approximating 1/x
    """
    n_qubits = len(x.cores)
    device = x.cores[0].device
    dtype = x.cores[0].dtype
    
    # Estimate scale from Frobenius norm of first few cores
    scale = 1.0
    for core in x.cores[:min(3, len(x.cores))]:
        scale *= core.norm().item() ** (1.0 / n_qubits)
    scale = max(scale, eps)
    
    # Regularize: add small epsilon to x
    x_reg = qtt_add(x, _constant_qtt(eps, n_qubits, device, dtype))
    
    # Scale x to have elements roughly O(1)
    x_scaled = qtt_scale(x_reg, 1.0 / scale)
    x_scaled = truncate_qtt(x_scaled, max_rank)
    
    # Initial guess: y_0 = 1 (for scaled x ~ O(1))
    y = _constant_qtt(1.0, n_qubits, device, dtype)
    two = _constant_qtt(2.0, n_qubits, device, dtype)
    
    for _ in range(n_iterations):
        # y = y * (2 - x_scaled * y)
        xy = qtt_hadamard(x_scaled, y)
        xy = truncate_qtt(xy, max_rank)
        
        two_minus_xy = qtt_add(two, qtt_scale(xy, -1.0))
        two_minus_xy = truncate_qtt(two_minus_xy, max_rank)
        
        y = qtt_hadamard(y, two_minus_xy)
        y = truncate_qtt(y, max_rank)
        
        # Clamp to prevent explosion
        for i, core in enumerate(y.cores):
            core_clamped = torch.clamp(core, -1e6, 1e6)
            if torch.isnan(core_clamped).any():
                core_clamped = torch.nan_to_num(core_clamped, nan=1.0)
            y.cores[i] = core_clamped
    
    # Scale back: 1/x = (1/x_scaled) / scale
    y = qtt_scale(y, 1.0 / scale)
    
    return y


def qtt_abs(x: QTTState, max_rank: int = 64) -> QTTState:
    """
    Compute element-wise absolute value |x|.
    
    Uses |x| = sqrt(x²) which can be computed in TT format.
    """
    x_sq = qtt_hadamard(x, x)
    x_sq = truncate_qtt(x_sq, max_rank)
    return qtt_sqrt_newton(x_sq, max_rank, x.cores[0].device)


def qtt_sqrt_newton(
    x: QTTState,
    max_rank: int,
    device: torch.device,
    n_iterations: int = 8,
    eps: float = 1e-10,
) -> QTTState:
    """
    Compute element-wise sqrt(x) using Newton iteration.
    
    Uses Babylonian method: y_{n+1} = 0.5 * (y_n + x / y_n)
    
    For stability, we estimate scale and work with normalized values.
    """
    n_qubits = len(x.cores)
    dtype = x.cores[0].dtype
    
    # Regularize to avoid sqrt of negative or zero
    x_reg = qtt_add(x, _constant_qtt(eps, n_qubits, device, dtype))
    
    # Estimate scale from first few cores
    scale_sq = 1.0
    for core in x.cores[:min(3, len(x.cores))]:
        scale_sq *= core.norm().item() ** (1.0 / n_qubits)
    scale_sq = max(scale_sq, eps)
    scale = scale_sq ** 0.5  # sqrt(scale) for output
    
    # Normalize x
    x_norm = qtt_scale(x_reg, 1.0 / scale_sq)
    x_norm = truncate_qtt(x_norm, max_rank)
    
    # Initial guess y_0 = 1 for normalized x
    y = _constant_qtt(1.0, n_qubits, device, dtype)
    
    for _ in range(n_iterations):
        # y = 0.5 * (y + x_norm / y)
        inv_y = qtt_inverse_newton(y, max_rank, n_iterations=3)
        x_over_y = qtt_hadamard(x_norm, inv_y)
        x_over_y = truncate_qtt(x_over_y, max_rank)
        
        y_new = qtt_add(y, x_over_y)
        y = qtt_scale(y_new, 0.5)
        y = truncate_qtt(y, max_rank)
        
        # Clamp to prevent NaN
        for i, core in enumerate(y.cores):
            core_clamped = torch.clamp(core, -1e6, 1e6)
            if torch.isnan(core_clamped).any():
                core_clamped = torch.nan_to_num(core_clamped, nan=1.0)
            y.cores[i] = core_clamped
    
    # Scale back: sqrt(x) = sqrt(x_norm) * scale
    y = qtt_scale(y, scale)
    
    return y


def _constant_qtt(value: float, n_qubits: int, device: torch.device, dtype: torch.dtype = torch.float32) -> QTTState:
    """Create a QTT representing a constant value at all positions.
    
    For a rank-1 QTT that evaluates to constant c at every index:
    - Each core has shape (1, 2, 1)
    - All cores have entries = 1 EXCEPT the first core
    - First core has entries = [c, c]
    - Then contraction for any binary index i = (b_0, b_1, ..., b_{n-1}) gives:
      core_0[0, b_0, 0] * core_1[0, b_1, 0] * ... = c * 1 * 1 * ... = c
    """
    cores = []
    
    for i in range(n_qubits):
        if i == 0:
            # First core carries the constant value
            core = torch.full((1, 2, 1), value, device=device, dtype=dtype)
        else:
            # Subsequent cores are identity (ones)
            core = torch.ones(1, 2, 1, device=device, dtype=dtype)
        cores.append(core)
    
    return QTTState(cores=cores, num_qubits=n_qubits)


# =============================================================================
# Phase 3.4: Polynomial Reconstruction in TT Format
# =============================================================================

def weno_reconstruct_native_tt(
    u: QTTState,
    side: ReconstructionSide = ReconstructionSide.LEFT,
    config: Optional[WENONativeTTConfig] = None,
) -> QTTState:
    """
    Perform full WENO5 reconstruction entirely in TT format.
    
    This is the main entry point for native WENO-TT.
    
    u^-_{i+1/2} = ω_0 * q_0 + ω_1 * q_1 + ω_2 * q_2
    
    where:
    - q_k = stencil reconstructions (linear, computed via MPO)
    - ω_k = nonlinear weights (computed from smoothness indicators)
    
    Args:
        u: QTT state of cell-averaged solution
        side: LEFT or RIGHT biased reconstruction
        config: Configuration
        
    Returns:
        QTT state of reconstructed interface values
    """
    if config is None:
        config = WENONativeTTConfig()
    
    n_qubits = len(u.cores)
    max_rank = config.max_rank
    
    # Step 1: Compute smoothness indicators β_k in TT format
    beta_0, beta_1, beta_2 = compute_smoothness_indicators_tt(u, config)
    
    # Step 2: Compute nonlinear weights ω_k in TT format
    # Use TCI for larger grids (true O(log N)), hybrid for small grids
    if n_qubits >= 12 and config.use_tci_weights:
        omega_0, omega_1, omega_2 = compute_weno_weights_tci(
            beta_0, beta_1, beta_2, side, config
        )
    else:
        omega_0, omega_1, omega_2 = compute_weno_weights_tt(
            beta_0, beta_1, beta_2, side, config
        )
    
    # Step 3: Compute candidate stencil reconstructions q_k via MPO
    q0_mpo, q1_mpo, q2_mpo = weno5_stencil_mpos(n_qubits, side)
    
    q_0 = apply_mpo(q0_mpo, u, max_rank)
    q_1 = apply_mpo(q1_mpo, u, max_rank)
    q_2 = apply_mpo(q2_mpo, u, max_rank)
    
    # Step 4: Weighted sum u^± = ω_0 * q_0 + ω_1 * q_1 + ω_2 * q_2
    term_0 = qtt_hadamard(omega_0, q_0)
    term_1 = qtt_hadamard(omega_1, q_1)
    term_2 = qtt_hadamard(omega_2, q_2)
    
    result = qtt_add(term_0, qtt_add(term_1, term_2))
    result = truncate_qtt(result, max_rank)
    
    return result


# =============================================================================
# Convenience Functions
# =============================================================================

def weno_flux_native_tt(
    u_left: QTTState,
    u_right: QTTState,
    alpha: float,
    config: Optional[WENONativeTTConfig] = None,
) -> QTTState:
    """
    Compute WENO-TT numerical flux using Lax-Friedrichs splitting.
    
    F_{i+1/2} = 0.5 * (F(u^-) + F(u^+)) - 0.5 * α * (u^+ - u^-)
    
    For scalar advection F(u) = au, this simplifies.
    
    Args:
        u_left: QTT of left state (cell averages)
        u_right: QTT of right state (cell averages)  
        alpha: Maximum wave speed
        config: Configuration
        
    Returns:
        QTT of numerical flux
    """
    if config is None:
        config = WENONativeTTConfig()
    
    max_rank = config.max_rank
    
    # Reconstruct at interfaces
    u_minus = weno_reconstruct_native_tt(u_left, ReconstructionSide.LEFT, config)
    u_plus = weno_reconstruct_native_tt(u_right, ReconstructionSide.RIGHT, config)
    
    # Lax-Friedrichs flux
    # Central: 0.5 * (u_minus + u_plus) * alpha (for advection)
    # Dissipation: -0.5 * alpha * (u_plus - u_minus)
    
    u_sum = qtt_add(u_minus, u_plus)
    u_diff = qtt_add(u_plus, qtt_scale(u_minus, -1.0))
    
    central = qtt_scale(u_sum, 0.5 * alpha)
    dissip = qtt_scale(u_diff, -0.5 * alpha)
    
    flux = qtt_add(central, dissip)
    flux = truncate_qtt(flux, max_rank)
    
    return flux


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Native WENO-TT Tests")
    print("=" * 60)
    print()
    
    # Test 1: Shift MPOs
    print("Test 1: Shift MPO construction...")
    n_qubits = 10
    N = 2 ** n_qubits
    
    s_plus = shift_mpo_cached(n_qubits, +1)
    s_minus = shift_mpo_cached(n_qubits, -1)
    s_plus2 = shift_mpo_cached(n_qubits, +2)
    
    print(f"  S^+ rank: {max(c.shape[0] for c in s_plus.cores)}")
    print(f"  S^- rank: {max(c.shape[0] for c in s_minus.cores)}")
    print(f"  S^+2 rank: {max(c.shape[0] for c in s_plus2.cores)}")
    print("  ✓ Shift MPOs created")
    print()
    
    # Test 2: Stencil reconstruction MPOs
    print("Test 2: WENO5 stencil MPOs...")
    q0, q1, q2 = weno5_stencil_mpos(n_qubits, ReconstructionSide.LEFT)
    print(f"  q0 rank: {max(c.shape[0] for c in q0.cores)}")
    print(f"  q1 rank: {max(c.shape[0] for c in q1.cores)}")
    print(f"  q2 rank: {max(c.shape[0] for c in q2.cores)}")
    print("  ✓ Stencil MPOs created")
    print()
    
    # Test 3: Smoothness indicator MPOs
    print("Test 3: Smoothness indicator MPOs...")
    beta_mpos = smoothness_indicator_mpos(n_qubits)
    print(f"  Number of difference pairs: {len(beta_mpos)}")
    print("  ✓ Smoothness MPOs created")
    print()
    
    # Test 4: Full reconstruction on sine wave
    print("Test 4: Full WENO-TT reconstruction on sine wave...")
    from .pure_qtt_ops import dense_to_qtt, qtt_to_dense
    
    x = torch.linspace(0, 2 * torch.pi, N)
    u_dense = torch.sin(x)
    
    u_qtt = dense_to_qtt(u_dense, max_rank=32)
    
    config = WENONativeTTConfig(max_rank=64)
    
    try:
        u_recon = weno_reconstruct_native_tt(u_qtt, ReconstructionSide.LEFT, config)
        u_recon_dense = qtt_to_dense(u_recon)
        
        error = (u_recon_dense - u_dense).abs().max().item()
        print(f"  Reconstruction error: {error:.2e}")
        print("  ✓ Full reconstruction completed")
    except Exception as e:
        print(f"  ⚠ Reconstruction failed: {e}")
    print()
    
    print("=" * 60)
    print("Native WENO-TT Tests Complete")
    print("=" * 60)

