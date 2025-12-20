"""
Slope Limiters for High-Resolution Schemes
==========================================

Limiters prevent spurious oscillations near discontinuities while
maintaining high-order accuracy in smooth regions.

TVD (Total Variation Diminishing) limiters satisfy:
    TV(u^{n+1}) ≤ TV(u^n)

where TV(u) = Σ|u_{i+1} - u_i|.

For a ratio r = (u_i - u_{i-1}) / (u_{i+1} - u_i):
- φ(r) = limiter function
- Limited slope: Δu = φ(r) * (u_{i+1} - u_i)

Sweby's TVD region: max(0, min(2r, 1)) ≤ φ(r) ≤ max(0, min(r, 2))
"""

from typing import Union
import torch
from torch import Tensor


def minmod(r: Tensor) -> Tensor:
    """
    Minmod limiter.
    
    φ(r) = max(0, min(1, r))
    
    Most diffusive TVD limiter. Very robust but only first-order
    at smooth extrema.
    
    Args:
        r: Ratio of consecutive gradients
        
    Returns:
        Limiter value
    """
    return torch.clamp(r, min=0, max=1)


def minmod_3(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """
    Three-argument minmod function.
    
    minmod(a, b, c) = sign(a) * min(|a|, |b|, |c|) if signs agree, else 0
    
    Used in higher-order reconstructions.
    """
    same_sign = (torch.sign(a) == torch.sign(b)) & (torch.sign(b) == torch.sign(c))
    
    min_abs = torch.minimum(torch.minimum(torch.abs(a), torch.abs(b)), torch.abs(c))
    
    return torch.where(same_sign, torch.sign(a) * min_abs, torch.zeros_like(a))


def superbee(r: Tensor) -> Tensor:
    """
    Superbee limiter (Roe, 1985).
    
    φ(r) = max(0, min(2r, 1), min(r, 2))
    
    Most compressive TVD limiter. Excellent for contact discontinuities
    but can steepen smooth waves too aggressively.
    """
    return torch.maximum(
        torch.zeros_like(r),
        torch.maximum(
            torch.minimum(2 * r, torch.ones_like(r)),
            torch.minimum(r, 2 * torch.ones_like(r))
        )
    )


def van_leer(r: Tensor) -> Tensor:
    """
    Van Leer limiter.
    
    φ(r) = (r + |r|) / (1 + |r|)
    
    Smooth limiter with good balance between accuracy and
    oscillation suppression.
    """
    return (r + torch.abs(r)) / (1 + torch.abs(r) + 1e-10)


def van_albada(r: Tensor, epsilon: float = 1e-6) -> Tensor:
    """
    Van Albada limiter.
    
    φ(r) = (r² + r) / (r² + 1)
    
    Differentiable limiter, useful for implicit schemes.
    """
    r2 = r * r
    return (r2 + r) / (r2 + 1 + epsilon)


def mc_limiter(r: Tensor) -> Tensor:
    """
    MC (Monotonized Central) limiter.
    
    φ(r) = max(0, min(2r, (1+r)/2, 2))
    
    Symmetric limiter that uses the central difference when possible.
    Good balance of accuracy and stability.
    """
    return torch.maximum(
        torch.zeros_like(r),
        torch.minimum(
            torch.minimum(2 * r, (1 + r) / 2),
            2 * torch.ones_like(r)
        )
    )


def koren(r: Tensor) -> Tensor:
    """
    Koren limiter (third-order accurate in smooth regions).
    
    φ(r) = max(0, min(2r, (2 + r)/3, 2))
    
    Optimized for third-order upstream-biased interpolation.
    """
    return torch.maximum(
        torch.zeros_like(r),
        torch.minimum(
            torch.minimum(2 * r, (2 + r) / 3),
            2 * torch.ones_like(r)
        )
    )


def ospre(r: Tensor) -> Tensor:
    """
    OSPRE limiter (Waterson & Deconinck, 2007).
    
    φ(r) = 1.5 * (r² + r) / (r² + r + 1)
    
    Smooth, symmetric limiter with good convergence properties.
    """
    r2 = r * r
    return 1.5 * (r2 + r) / (r2 + r + 1 + 1e-10)


def compute_slope_ratio(u: Tensor, i: int) -> Tensor:
    """
    Compute slope ratio at cell i for limiter application.
    
    r_i = (u_i - u_{i-1}) / (u_{i+1} - u_i)
    
    Handles boundary cases with one-sided differences.
    """
    N = u.shape[0]
    
    if i <= 0:
        return torch.ones_like(u[0])  # One-sided
    if i >= N - 1:
        return torch.ones_like(u[0])  # One-sided
    
    delta_plus = u[i + 1] - u[i]
    delta_minus = u[i] - u[i - 1]
    
    # Avoid division by zero
    denom = torch.where(
        torch.abs(delta_plus) < 1e-10,
        torch.sign(delta_plus) * 1e-10,
        delta_plus
    )
    
    return delta_minus / denom


def apply_limiter(
    u: Tensor,
    limiter: str = 'minmod',
) -> Tensor:
    """
    Apply slope limiter to get limited gradients.
    
    Args:
        u: Cell values (N,) or (N, n_vars)
        limiter: Limiter name ('minmod', 'superbee', 'van_leer', 'mc')
        
    Returns:
        Limited slopes for reconstruction
    """
    limiters = {
        'minmod': minmod,
        'superbee': superbee,
        'van_leer': van_leer,
        'van_albada': van_albada,
        'mc': mc_limiter,
        'koren': koren,
        'ospre': ospre,
    }
    
    if limiter not in limiters:
        raise ValueError(f"Unknown limiter: {limiter}. Options: {list(limiters.keys())}")
    
    phi_func = limiters[limiter]
    
    # Handle multi-variable case
    if u.dim() == 1:
        u = u.unsqueeze(-1)
    
    N, n_vars = u.shape
    slopes = torch.zeros_like(u)
    
    for i in range(1, N - 1):
        delta_plus = u[i + 1] - u[i]
        delta_minus = u[i] - u[i - 1]
        
        # Slope ratio
        denom = torch.where(
            torch.abs(delta_plus) < 1e-10,
            torch.sign(delta_plus + 1e-20) * 1e-10,
            delta_plus
        )
        r = delta_minus / denom
        
        # Limited slope
        slopes[i] = phi_func(r) * delta_plus
    
    # Boundary slopes (first-order)
    slopes[0] = torch.zeros_like(slopes[0])
    slopes[-1] = torch.zeros_like(slopes[-1])
    
    return slopes.squeeze(-1) if slopes.shape[-1] == 1 else slopes


class MUSCL:
    """
    MUSCL (Monotonic Upstream-centered Scheme for Conservation Laws).
    
    Provides second-order reconstruction with slope limiting.
    
    u_{i+1/2,L} = u_i + (1-κ)/4 * φ(r_i) * (u_{i+1} - u_i) + (1+κ)/4 * φ(1/r_i) * (u_i - u_{i-1})
    u_{i+1/2,R} = u_{i+1} - (1+κ)/4 * φ(r_{i+1}) * (u_{i+2} - u_{i+1}) - (1-κ)/4 * φ(1/r_{i+1}) * (u_{i+1} - u_i)
    
    κ = -1: Fully upwind (second-order upwind)
    κ = 0: Fromm's scheme
    κ = 1/3: Third-order upwind-biased
    κ = 1: Central
    """
    
    def __init__(
        self,
        limiter: str = 'van_leer',
        kappa: float = 1/3,
    ):
        """
        Initialize MUSCL reconstruction.
        
        Args:
            limiter: Slope limiter to use
            kappa: Interpolation parameter
        """
        self.limiter = limiter
        self.kappa = kappa
        
        self._limiters = {
            'minmod': minmod,
            'superbee': superbee,
            'van_leer': van_leer,
            'mc': mc_limiter,
        }
    
    def reconstruct(
        self,
        u: Tensor,
    ) -> tuple:
        """
        Perform MUSCL reconstruction.
        
        Args:
            u: Cell-centered values (N,) or (N, n_vars)
            
        Returns:
            (u_L, u_R): Left and right interface values at N-1 faces
        """
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        
        N, n_vars = u.shape
        phi_func = self._limiters.get(self.limiter, minmod)
        
        # Compute slopes with limiter
        slopes = torch.zeros_like(u)
        
        for i in range(1, N - 1):
            delta_plus = u[i + 1] - u[i]
            delta_minus = u[i] - u[i - 1]
            
            denom = delta_plus.clone()
            denom[torch.abs(denom) < 1e-10] = 1e-10
            r = delta_minus / denom
            
            slopes[i] = phi_func(r) * delta_plus
        
        # Reconstruct at faces
        u_L = u[:-1] + 0.5 * slopes[:-1]
        u_R = u[1:] - 0.5 * slopes[1:]
        
        return u_L.squeeze(-1), u_R.squeeze(-1)
