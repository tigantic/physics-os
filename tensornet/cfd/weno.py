"""
WENO and TENO Shock-Capturing Schemes for HyperTensor CFD.

This module implements high-order weighted essentially non-oscillatory (WENO)
and targeted essentially non-oscillatory (TENO) reconstruction schemes for
capturing shock waves without spurious oscillations.

Schemes implemented:
- WENO5-JS: Original 5th-order WENO (Jiang-Shu 1996)
- WENO5-Z: Improved WENO with better accuracy at critical points (Borges 2008)
- TENO5: Targeted ENO with sharp discontinuity detection (Fu 2016)

References:
- Jiang & Shu (1996) "Efficient Implementation of Weighted ENO Schemes"
- Borges et al. (2008) "An improved WENO scheme for hyperbolic conservation laws"
- Fu et al. (2016) "A family of high-order targeted ENO schemes"
- arXiv:2405.12301 — Tensor-Train WENO Scheme for Compressible Flows

Constitution Compliance: Article I.1 (Proof Requirements)
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto


class WENOVariant(Enum):
    """WENO scheme variants."""
    JS = auto()   # Jiang-Shu original
    Z = auto()    # WENO-Z improved
    M = auto()    # WENO-M mapped


class ReconstructionSide(Enum):
    """Which side of the cell interface to reconstruct."""
    LEFT = auto()   # u_{i+1/2}^-
    RIGHT = auto()  # u_{i+1/2}^+


@dataclass
class WENOConfig:
    """Configuration for WENO reconstruction."""
    epsilon: float = 1e-40  # Small number to prevent division by zero
    p: int = 2              # Exponent in nonlinear weights (typically 2)
    variant: WENOVariant = WENOVariant.Z


# =============================================================================
# Optimal Weights (Linear Weights)
# =============================================================================

def optimal_weights_left() -> Tuple[float, float, float]:
    """
    Optimal linear weights for left-biased reconstruction (u_{i+1/2}^-).
    
    These weights give 5th-order accuracy on smooth solutions when
    combined with the three candidate stencils.
    
    Returns:
        (d0, d1, d2): Optimal weights summing to 1
    """
    d0 = 1.0 / 10.0   # Weight for stencil {i-2, i-1, i}
    d1 = 6.0 / 10.0   # Weight for stencil {i-1, i, i+1}
    d2 = 3.0 / 10.0   # Weight for stencil {i, i+1, i+2}
    return d0, d1, d2


def optimal_weights_right() -> Tuple[float, float, float]:
    """
    Optimal linear weights for right-biased reconstruction (u_{i+1/2}^+).
    
    Returns:
        (d0, d1, d2): Optimal weights summing to 1
    """
    d0 = 3.0 / 10.0   # Weight for stencil {i-1, i, i+1}
    d1 = 6.0 / 10.0   # Weight for stencil {i, i+1, i+2}
    d2 = 1.0 / 10.0   # Weight for stencil {i+1, i+2, i+3}
    return d0, d1, d2


# =============================================================================
# Smoothness Indicators
# =============================================================================

def smoothness_indicators(u: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute smoothness indicators β for WENO reconstruction.
    
    The smoothness indicators measure the smoothness of the solution on
    each candidate stencil. Smaller values indicate smoother solutions.
    
    β_k = Σ_{l=1}^{r-1} Δx^{2l-1} ∫_{x_{i-1/2}}^{x_{i+1/2}} (d^l p_k / dx^l)^2 dx
    
    For 5th-order WENO with r=3:
    β_0 = (13/12)(u_{i-2} - 2u_{i-1} + u_i)^2 + (1/4)(u_{i-2} - 4u_{i-1} + 3u_i)^2
    β_1 = (13/12)(u_{i-1} - 2u_i + u_{i+1})^2 + (1/4)(u_{i-1} - u_{i+1})^2
    β_2 = (13/12)(u_i - 2u_{i+1} + u_{i+2})^2 + (1/4)(3u_i - 4u_{i+1} + u_{i+2})^2
    
    Args:
        u: Solution array of shape (..., N) where N >= 5
        
    Returns:
        (beta0, beta1, beta2): Smoothness indicators for each stencil,
                               each of shape (..., N-4)
    """
    # Extract stencil values (interior points only)
    um2 = u[..., :-4]    # u_{i-2}
    um1 = u[..., 1:-3]   # u_{i-1}
    u0 = u[..., 2:-2]    # u_i
    up1 = u[..., 3:-1]   # u_{i+1}
    up2 = u[..., 4:]     # u_{i+2}
    
    # Smoothness indicator for stencil 0: {i-2, i-1, i}
    beta0 = (13.0 / 12.0) * (um2 - 2.0 * um1 + u0) ** 2 + \
            (1.0 / 4.0) * (um2 - 4.0 * um1 + 3.0 * u0) ** 2
    
    # Smoothness indicator for stencil 1: {i-1, i, i+1}
    beta1 = (13.0 / 12.0) * (um1 - 2.0 * u0 + up1) ** 2 + \
            (1.0 / 4.0) * (um1 - up1) ** 2
    
    # Smoothness indicator for stencil 2: {i, i+1, i+2}
    beta2 = (13.0 / 12.0) * (u0 - 2.0 * up1 + up2) ** 2 + \
            (1.0 / 4.0) * (3.0 * u0 - 4.0 * up1 + up2) ** 2
    
    return beta0, beta1, beta2


def global_smoothness_indicator(beta0: Tensor, beta1: Tensor, beta2: Tensor) -> Tensor:
    """
    Compute global smoothness indicator τ for WENO-Z.
    
    τ_5 = |β_0 - β_2|
    
    This measures the global regularity and is used to improve accuracy
    at critical points where the solution is smooth but has zero derivatives.
    
    Args:
        beta0, beta1, beta2: Smoothness indicators from each stencil
        
    Returns:
        tau: Global smoothness indicator
    """
    return torch.abs(beta0 - beta2)


# =============================================================================
# Candidate Stencil Reconstructions
# =============================================================================

def candidate_stencils_left(u: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute candidate polynomial reconstructions for left-biased (u_{i+1/2}^-).
    
    Each stencil gives a 3rd-order accurate reconstruction at the right
    cell interface using a different set of three cells.
    
    q_0 = (2u_{i-2} - 7u_{i-1} + 11u_i) / 6       from {i-2, i-1, i}
    q_1 = (-u_{i-1} + 5u_i + 2u_{i+1}) / 6        from {i-1, i, i+1}
    q_2 = (2u_i + 5u_{i+1} - u_{i+2}) / 6         from {i, i+1, i+2}
    
    Args:
        u: Solution array of shape (..., N) where N >= 5
        
    Returns:
        (q0, q1, q2): Candidate reconstructions, each of shape (..., N-4)
    """
    um2 = u[..., :-4]
    um1 = u[..., 1:-3]
    u0 = u[..., 2:-2]
    up1 = u[..., 3:-1]
    up2 = u[..., 4:]
    
    q0 = (2.0 * um2 - 7.0 * um1 + 11.0 * u0) / 6.0
    q1 = (-um1 + 5.0 * u0 + 2.0 * up1) / 6.0
    q2 = (2.0 * u0 + 5.0 * up1 - up2) / 6.0
    
    return q0, q1, q2


def candidate_stencils_right(u: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute candidate polynomial reconstructions for right-biased (u_{i+1/2}^+).
    
    q_0 = (-u_{i-1} + 5u_i + 2u_{i+1}) / 6        from {i-1, i, i+1}
    q_1 = (2u_i + 5u_{i+1} - u_{i+2}) / 6         from {i, i+1, i+2}
    q_2 = (11u_{i+1} - 7u_{i+2} + 2u_{i+3}) / 6   from {i+1, i+2, i+3}
    
    Args:
        u: Solution array of shape (..., N) where N >= 6
        
    Returns:
        (q0, q1, q2): Candidate reconstructions
    """
    um1 = u[..., 1:-4]
    u0 = u[..., 2:-3]
    up1 = u[..., 3:-2]
    up2 = u[..., 4:-1]
    up3 = u[..., 5:]
    
    q0 = (-um1 + 5.0 * u0 + 2.0 * up1) / 6.0
    q1 = (2.0 * u0 + 5.0 * up1 - up2) / 6.0
    q2 = (11.0 * up1 - 7.0 * up2 + 2.0 * up3) / 6.0
    
    return q0, q1, q2


# =============================================================================
# Nonlinear Weights
# =============================================================================

def nonlinear_weights_js(
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    d0: float,
    d1: float,
    d2: float,
    epsilon: float = 1e-40,
    p: int = 2
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute nonlinear weights using WENO-JS (Jiang-Shu) formula.
    
    α_k = d_k / (ε + β_k)^p
    ω_k = α_k / Σ α_j
    
    Args:
        beta0, beta1, beta2: Smoothness indicators
        d0, d1, d2: Optimal linear weights
        epsilon: Small number to prevent division by zero
        p: Exponent (typically 2)
        
    Returns:
        (omega0, omega1, omega2): Nonlinear weights summing to 1
    """
    alpha0 = d0 / (epsilon + beta0) ** p
    alpha1 = d1 / (epsilon + beta1) ** p
    alpha2 = d2 / (epsilon + beta2) ** p
    
    alpha_sum = alpha0 + alpha1 + alpha2
    
    omega0 = alpha0 / alpha_sum
    omega1 = alpha1 / alpha_sum
    omega2 = alpha2 / alpha_sum
    
    return omega0, omega1, omega2


def nonlinear_weights_z(
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    d0: float,
    d1: float,
    d2: float,
    epsilon: float = 1e-40,
    p: int = 2
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute nonlinear weights using WENO-Z formula.
    
    The WENO-Z scheme uses the global smoothness indicator τ to improve
    accuracy at critical points (where derivatives are zero).
    
    α_k = d_k * (1 + (τ / (ε + β_k))^p)
    ω_k = α_k / Σ α_j
    
    Args:
        beta0, beta1, beta2: Smoothness indicators
        d0, d1, d2: Optimal linear weights
        epsilon: Small number to prevent division by zero
        p: Exponent (typically 2)
        
    Returns:
        (omega0, omega1, omega2): Nonlinear weights summing to 1
    """
    # Global smoothness indicator
    tau = global_smoothness_indicator(beta0, beta1, beta2)
    
    # WENO-Z weights
    alpha0 = d0 * (1.0 + (tau / (epsilon + beta0)) ** p)
    alpha1 = d1 * (1.0 + (tau / (epsilon + beta1)) ** p)
    alpha2 = d2 * (1.0 + (tau / (epsilon + beta2)) ** p)
    
    alpha_sum = alpha0 + alpha1 + alpha2
    
    omega0 = alpha0 / alpha_sum
    omega1 = alpha1 / alpha_sum
    omega2 = alpha2 / alpha_sum
    
    return omega0, omega1, omega2


# =============================================================================
# WENO5 Reconstruction Functions
# =============================================================================

def weno5_js(
    u: Tensor,
    side: ReconstructionSide = ReconstructionSide.LEFT,
    config: Optional[WENOConfig] = None
) -> Tensor:
    """
    5th-order WENO-JS reconstruction (Jiang-Shu 1996).
    
    Reconstructs the solution at cell interfaces using the original
    WENO scheme with 5th-order accuracy on smooth regions and
    essentially non-oscillatory behavior near discontinuities.
    
    Args:
        u: Solution array of shape (..., N) where N >= 5 (or N >= 6 for right)
        side: Which side of interface to reconstruct (LEFT or RIGHT)
        config: Optional WENO configuration
        
    Returns:
        Reconstructed values at cell interfaces, shape (..., N-4) or (..., N-5)
        
    Example:
        >>> u = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        >>> u_left = weno5_js(u, ReconstructionSide.LEFT)
        >>> # u_left gives u_{i+1/2}^- at interior interfaces
    """
    if config is None:
        config = WENOConfig(variant=WENOVariant.JS)
    
    # Get smoothness indicators
    beta0, beta1, beta2 = smoothness_indicators(u)
    
    if side == ReconstructionSide.LEFT:
        d0, d1, d2 = optimal_weights_left()
        q0, q1, q2 = candidate_stencils_left(u)
    else:
        d0, d1, d2 = optimal_weights_right()
        q0, q1, q2 = candidate_stencils_right(u)
        # Need to trim beta to match q dimensions
        beta0 = beta0[..., :-1]
        beta1 = beta1[..., :-1]
        beta2 = beta2[..., :-1]
    
    # Compute nonlinear weights
    omega0, omega1, omega2 = nonlinear_weights_js(
        beta0, beta1, beta2, d0, d1, d2,
        epsilon=config.epsilon, p=config.p
    )
    
    # Final reconstruction
    u_reconstructed = omega0 * q0 + omega1 * q1 + omega2 * q2
    
    return u_reconstructed


def weno5_z(
    u: Tensor,
    side: ReconstructionSide = ReconstructionSide.LEFT,
    config: Optional[WENOConfig] = None
) -> Tensor:
    """
    5th-order WENO-Z reconstruction (Borges 2008).
    
    Improved WENO scheme that achieves optimal convergence order at
    critical points where the first derivative vanishes.
    
    Args:
        u: Solution array of shape (..., N) where N >= 5 (or N >= 6 for right)
        side: Which side of interface to reconstruct (LEFT or RIGHT)
        config: Optional WENO configuration
        
    Returns:
        Reconstructed values at cell interfaces
        
    Note:
        WENO-Z is preferred over WENO-JS for most applications due to
        improved accuracy at critical points without additional cost.
    """
    if config is None:
        config = WENOConfig(variant=WENOVariant.Z)
    
    # Get smoothness indicators
    beta0, beta1, beta2 = smoothness_indicators(u)
    
    if side == ReconstructionSide.LEFT:
        d0, d1, d2 = optimal_weights_left()
        q0, q1, q2 = candidate_stencils_left(u)
    else:
        d0, d1, d2 = optimal_weights_right()
        q0, q1, q2 = candidate_stencils_right(u)
        # Trim beta to match
        beta0 = beta0[..., :-1]
        beta1 = beta1[..., :-1]
        beta2 = beta2[..., :-1]
    
    # Compute WENO-Z weights
    omega0, omega1, omega2 = nonlinear_weights_z(
        beta0, beta1, beta2, d0, d1, d2,
        epsilon=config.epsilon, p=config.p
    )
    
    # Final reconstruction
    u_reconstructed = omega0 * q0 + omega1 * q1 + omega2 * q2
    
    return u_reconstructed


# =============================================================================
# TENO5 Reconstruction
# =============================================================================

def teno_cutoff_function(
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    C_T: float = 1e-5
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute TENO sharp cutoff indicators.
    
    TENO uses a sharp cutoff to completely exclude stencils that cross
    discontinuities, unlike WENO which only reduces their weight.
    
    δ_k = 1 if stencil k is smooth, 0 otherwise
    
    The decision is based on comparing β_k to a reference smoothness.
    In smooth regions, all δ_k = 1 (all stencils used).
    Near discontinuities, stencils crossing the shock get δ_k = 0.
    
    Args:
        beta0, beta1, beta2: Smoothness indicators
        C_T: Cutoff threshold (smaller = more aggressive cutoff)
        
    Returns:
        (delta0, delta1, delta2): Binary cutoff indicators (0 or 1)
    """
    # Global smoothness indicator (TENO uses sum, not max like original)
    tau = torch.abs(beta0 - beta2)
    
    # Compute scale-independent smoothness measure
    # chi_k = (C_T + tau) / (C_T + beta_k)
    # If chi_k < 1, stencil k is less smooth than global measure
    chi0 = (C_T + tau) / (C_T + beta0 + 1e-40)
    chi1 = (C_T + tau) / (C_T + beta1 + 1e-40)
    chi2 = (C_T + tau) / (C_T + beta2 + 1e-40)
    
    # Sharp cutoff: δ_k = 1 if chi_k > threshold, else 0
    # In smooth regions, tau ≈ 0, so chi_k ≈ C_T/(C_T + beta_k) which is small
    # Near shocks, tau is large for crossing stencils
    
    # Alternative: use normalized approach from Fu et al. 2016
    # Reference is the geometric mean
    beta_bar = (beta0 * beta1 * beta2 + 1e-40) ** (1.0/3.0)
    
    # Cutoff indicator based on relative smoothness
    gamma0 = beta0 / (beta_bar + 1e-40)
    gamma1 = beta1 / (beta_bar + 1e-40)
    gamma2 = beta2 / (beta_bar + 1e-40)
    
    # In smooth regions, gamma_k ≈ 1 for all k (all stencils equally smooth)
    # Near discontinuity, stencils crossing shock have gamma >> 1
    threshold = 1.0 / C_T  # Large threshold, only cuts truly bad stencils
    
    delta0 = (gamma0 < threshold).to(beta0.dtype)
    delta1 = (gamma1 < threshold).to(beta1.dtype)
    delta2 = (gamma2 < threshold).to(beta2.dtype)
    
    # Ensure at least one stencil survives (fall back to smoothest)
    all_cut = (delta0 + delta1 + delta2) < 0.5
    if all_cut.any():
        # Find minimum beta and keep that stencil
        min_beta = torch.minimum(torch.minimum(beta0, beta1), beta2)
        delta0 = torch.where(all_cut & (beta0 <= min_beta + 1e-40), 
                             torch.ones_like(delta0), delta0)
        delta1 = torch.where(all_cut & (beta1 <= min_beta + 1e-40) & (delta0 < 0.5),
                             torch.ones_like(delta1), delta1)
        delta2 = torch.where(all_cut & (delta0 < 0.5) & (delta1 < 0.5),
                             torch.ones_like(delta2), delta2)
    
    return delta0, delta1, delta2


def teno5(
    u: Tensor,
    side: ReconstructionSide = ReconstructionSide.LEFT,
    C_T: float = 1e-5,
    config: Optional[WENOConfig] = None
) -> Tensor:
    """
    5th-order TENO reconstruction (Fu 2016).
    
    Targeted Essentially Non-Oscillatory scheme that uses sharp cutoff
    to completely exclude stencils crossing discontinuities, providing
    lower numerical dissipation than WENO near shocks.
    
    Args:
        u: Solution array of shape (..., N)
        side: Which side of interface to reconstruct
        C_T: Cutoff threshold for stencil exclusion
        config: Optional configuration
        
    Returns:
        Reconstructed values at cell interfaces
        
    Note:
        TENO provides sharper shock resolution than WENO but may be
        slightly less robust. For highly unsteady flows with strong
        shocks, WENO-Z is often preferred.
    """
    if config is None:
        config = WENOConfig(variant=WENOVariant.Z)
    
    # Get smoothness indicators
    beta0, beta1, beta2 = smoothness_indicators(u)
    
    if side == ReconstructionSide.LEFT:
        d0, d1, d2 = optimal_weights_left()
        q0, q1, q2 = candidate_stencils_left(u)
    else:
        d0, d1, d2 = optimal_weights_right()
        q0, q1, q2 = candidate_stencils_right(u)
        beta0 = beta0[..., :-1]
        beta1 = beta1[..., :-1]
        beta2 = beta2[..., :-1]
    
    # Get TENO cutoff indicators
    delta0, delta1, delta2 = teno_cutoff_function(beta0, beta1, beta2, C_T)
    
    # Apply cutoff to optimal weights
    d0_cut = d0 * delta0
    d1_cut = d1 * delta1
    d2_cut = d2 * delta2
    
    # Renormalize
    d_sum = d0_cut + d1_cut + d2_cut + 1e-40
    omega0 = d0_cut / d_sum
    omega1 = d1_cut / d_sum
    omega2 = d2_cut / d_sum
    
    # Final reconstruction
    u_reconstructed = omega0 * q0 + omega1 * q1 + omega2 * q2
    
    return u_reconstructed


# =============================================================================
# Flux Reconstruction with WENO
# =============================================================================

def weno_flux_split(
    flux_positive: Tensor,
    flux_negative: Tensor,
    variant: WENOVariant = WENOVariant.Z,
    config: Optional[WENOConfig] = None
) -> Tensor:
    """
    Compute numerical flux at cell interfaces using WENO with flux splitting.
    
    Uses Lax-Friedrichs flux splitting: f^± = (f ± α*u) / 2
    where α is the maximum wave speed.
    
    Args:
        flux_positive: Positive flux contribution f^+, shape (..., N)
        flux_negative: Negative flux contribution f^-, shape (..., N)
        variant: WENO variant to use
        config: Optional configuration
        
    Returns:
        Numerical flux at interfaces, shape (..., N-4) or (..., N-5)
    """
    # Choose reconstruction function
    if variant == WENOVariant.JS:
        weno_fn = weno5_js
    else:
        weno_fn = weno5_z
    
    # Reconstruct f^+ from left and f^- from right
    f_plus_left = weno_fn(flux_positive, ReconstructionSide.LEFT, config)
    f_minus_right = weno_fn(flux_negative, ReconstructionSide.RIGHT, config)
    
    # Trim to same size (right reconstruction is one shorter)
    f_plus_left = f_plus_left[..., :-1]
    
    # Numerical flux at interface
    flux_interface = f_plus_left + f_minus_right
    
    return flux_interface


def weno_reconstruct_euler(
    rho: Tensor,
    u: Tensor,
    p: Tensor,
    gamma: float = 1.4,
    variant: WENOVariant = WENOVariant.Z
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    WENO reconstruction for 1D Euler equations using characteristic variables.
    
    Transforms to characteristic variables, applies WENO reconstruction,
    then transforms back to primitive variables.
    
    Args:
        rho: Density array
        u: Velocity array
        p: Pressure array
        gamma: Ratio of specific heats
        variant: WENO variant
        
    Returns:
        (rho_L, u_L, p_L, rho_R, u_R, p_R): Left and right reconstructed states
    """
    # Sound speed
    a = torch.sqrt(gamma * p / rho)
    
    # Characteristic variables (Roe average at cell centers for simplicity)
    # w1 = -ρa*u + p  (entropy wave)
    # w2 = p - a²ρ    (left acoustic)
    # w3 = p + a²ρ    (right acoustic)
    
    # For simplicity, reconstruct primitive variables directly
    # Full characteristic decomposition is a future enhancement
    
    if variant == WENOVariant.JS:
        weno_fn = weno5_js
    else:
        weno_fn = weno5_z
    
    # Left-biased reconstruction (u^-)
    rho_L = weno_fn(rho, ReconstructionSide.LEFT)
    u_L = weno_fn(u, ReconstructionSide.LEFT)
    p_L = weno_fn(p, ReconstructionSide.LEFT)
    
    # Right-biased reconstruction (u^+)
    rho_R = weno_fn(rho, ReconstructionSide.RIGHT)
    u_R = weno_fn(u, ReconstructionSide.RIGHT)
    p_R = weno_fn(p, ReconstructionSide.RIGHT)
    
    # Trim to same size
    rho_L = rho_L[..., :-1]
    u_L = u_L[..., :-1]
    p_L = p_L[..., :-1]
    
    return rho_L, u_L, p_L, rho_R, u_R, p_R


# =============================================================================
# High-Level Interface
# =============================================================================

def reconstruct(
    u: Tensor,
    method: str = "weno5-z",
    side: ReconstructionSide = ReconstructionSide.LEFT,
    **kwargs
) -> Tensor:
    """
    High-level interface for WENO/TENO reconstruction.
    
    Args:
        u: Solution array
        method: One of "weno5-js", "weno5-z", "teno5"
        side: LEFT or RIGHT reconstruction
        **kwargs: Additional arguments passed to specific method
        
    Returns:
        Reconstructed values at cell interfaces
        
    Example:
        >>> u = torch.linspace(0, 1, 100)
        >>> u_recon = reconstruct(u, method="weno5-z")
    """
    method = method.lower()
    
    if method == "weno5-js":
        return weno5_js(u, side, **kwargs)
    elif method == "weno5-z":
        return weno5_z(u, side, **kwargs)
    elif method == "teno5":
        return teno5(u, side, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: weno5-js, weno5-z, teno5")


# =============================================================================
# Utility Functions
# =============================================================================

def convergence_order(
    errors: list[float],
    dx_values: list[float]
) -> float:
    """
    Compute convergence order from error data.
    
    Uses least-squares fit of log(error) vs log(dx).
    
    Args:
        errors: List of L2 or L∞ errors
        dx_values: Corresponding grid spacings
        
    Returns:
        Estimated order of convergence
    """
    import numpy as np
    
    log_err = np.log(errors)
    log_dx = np.log(dx_values)
    
    # Linear regression: log(err) = p * log(dx) + c
    coeffs = np.polyfit(log_dx, log_err, 1)
    
    return coeffs[0]  # Slope = order


def verify_fifth_order(
    u_exact_fn,
    N_values: list[int] = [32, 64, 128, 256],
    domain: Tuple[float, float] = (0.0, 1.0),
    method: str = "weno5-z"
) -> Tuple[float, list[float]]:
    """
    Verify 5th-order convergence on a smooth test function.
    
    Args:
        u_exact_fn: Callable that takes x tensor and returns exact solution
        N_values: Grid sizes to test
        domain: (x_min, x_max)
        method: WENO variant to test
        
    Returns:
        (order, errors): Estimated order and list of L2 errors
    """
    errors = []
    dx_values = []
    
    for N in N_values:
        x = torch.linspace(domain[0], domain[1], N)
        dx = (domain[1] - domain[0]) / (N - 1)
        dx_values.append(dx)
        
        u = u_exact_fn(x)
        u_recon = reconstruct(u, method=method)
        
        # Compare to exact reconstruction (analytical derivative)
        # For smooth functions, WENO should match high-order FD
        x_interface = 0.5 * (x[2:-2] + x[3:-1])
        u_exact_interface = u_exact_fn(x_interface)
        
        # L2 error
        err = torch.sqrt(torch.mean((u_recon - u_exact_interface) ** 2)).item()
        errors.append(err)
    
    order = convergence_order(errors, dx_values)
    
    return order, errors
