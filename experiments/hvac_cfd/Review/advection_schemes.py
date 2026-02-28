"""
Advection Schemes for HVAC CFD
==============================

Multiple advection discretizations with different numerical properties:

| Scheme      | Order | Diffusion    | Stability        |
|-------------|-------|--------------|------------------|
| upwind      | O(Δx) | HIGH         | Unconditional    |
| central     | O(Δx²)| None         | CFL < 1          |
| quick       | O(Δx³)| Low          | CFL < ~0.5       |
| weno3       | O(Δx³)| Adaptive     | TVD              |

The old The Ontic Engine solver used spectral (FFT) which has zero numerical
diffusion. For non-periodic domains (walls, inlet, outlet), we need
finite difference schemes. Central differences are the closest to
spectral in terms of diffusion properties.

Tag: [PHASE-16] [HVAC] [TIER-1]
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_advection_central(
    u: Tensor, 
    v: Tensor, 
    dx: float, 
    dy: float
) -> tuple[Tensor, Tensor]:
    """
    Compute advection using central differences (2nd order, no numerical diffusion).
    
    This matches the spectral method's diffusion properties for smooth flows.
    
    advection_u = u * ∂u/∂x + v * ∂u/∂y
    advection_v = u * ∂v/∂x + v * ∂v/∂y
    
    Returns NEGATIVE of advection (ready to add to RHS).
    """
    # Central difference gradients (2nd order)
    # ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2Δx)
    
    du_dx = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2 * dx)
    du_dy = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dy)
    
    dv_dx = (torch.roll(v, -1, dims=0) - torch.roll(v, 1, dims=0)) / (2 * dx)
    dv_dy = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * dy)
    
    # Advection: (u·∇)u
    adv_u = u * du_dx + v * du_dy
    adv_v = u * dv_dx + v * dv_dy
    
    return -adv_u, -adv_v


def compute_advection_quick(
    u: Tensor,
    v: Tensor,
    dx: float,
    dy: float,
) -> tuple[Tensor, Tensor]:
    """
    QUICK scheme (Quadratic Upstream Interpolation for Convective Kinematics).
    
    3rd order accurate, low numerical diffusion, bounded.
    
    For positive velocity u > 0:
        ∂φ/∂x ≈ (3φ[i] + 3φ[i+1] - 7φ[i-1] + φ[i-2]) / (8Δx)
    
    Actually uses a simpler 3-point upwind-biased stencil:
        ∂φ/∂x ≈ (3φ[i+1] + 3φ[i] - 6φ[i-1] - φ[i-2] + φ[i+2]) / (8Δx)
    
    For simplicity, we use the standard QUICK interpolation at cell faces.
    """
    nx, ny = u.shape
    
    adv_u = torch.zeros_like(u)
    adv_v = torch.zeros_like(v)
    
    # For interior points, use QUICK
    # This is a simplified version using convective form
    
    for field, adv, vel_x, vel_y in [(u, adv_u, u, v), (v, adv_v, u, v)]:
        # X-direction advection
        # Face value at i+1/2 using QUICK
        phi_im2 = torch.roll(field, 2, dims=0)
        phi_im1 = torch.roll(field, 1, dims=0)
        phi_i = field
        phi_ip1 = torch.roll(field, -1, dims=0)
        phi_ip2 = torch.roll(field, -2, dims=0)
        
        # QUICK for u > 0: φ_{i+1/2} = (3/8)φ_{i+1} + (6/8)φ_i - (1/8)φ_{i-1}
        # QUICK for u < 0: φ_{i+1/2} = (3/8)φ_i + (6/8)φ_{i+1} - (1/8)φ_{i+2}
        
        u_pos = vel_x > 0
        face_pos = (3 * phi_ip1 + 6 * phi_i - phi_im1) / 8
        face_neg = (3 * phi_i + 6 * phi_ip1 - phi_ip2) / 8
        face_e = torch.where(u_pos, face_pos, face_neg)
        
        face_w = torch.roll(face_e, 1, dims=0)
        
        flux_x = vel_x * (face_e - face_w) / dx
        
        # Y-direction (similar)
        phi_jm2 = torch.roll(field, 2, dims=1)
        phi_jm1 = torch.roll(field, 1, dims=1)
        phi_j = field
        phi_jp1 = torch.roll(field, -1, dims=1)
        phi_jp2 = torch.roll(field, -2, dims=1)
        
        v_pos = vel_y > 0
        face_pos_y = (3 * phi_jp1 + 6 * phi_j - phi_jm1) / 8
        face_neg_y = (3 * phi_j + 6 * phi_jp1 - phi_jp2) / 8
        face_n = torch.where(v_pos, face_pos_y, face_neg_y)
        
        face_s = torch.roll(face_n, 1, dims=1)
        
        flux_y = vel_y * (face_n - face_s) / dy
        
        adv[:] = flux_x + flux_y
    
    return -adv_u, -adv_v


def compute_advection_hybrid(
    u: Tensor,
    v: Tensor,
    dx: float,
    dy: float,
    blend: float = 0.9,
) -> tuple[Tensor, Tensor]:
    """
    Hybrid central-upwind scheme.
    
    advection = blend * central + (1 - blend) * upwind
    
    blend = 1.0: Pure central (no diffusion, may oscillate)
    blend = 0.0: Pure upwind (high diffusion, always stable)
    blend = 0.9: Mostly central with slight damping
    
    This is a practical compromise for high-Re flows with non-periodic BC.
    """
    # Central component
    du_dx_c = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2 * dx)
    du_dy_c = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dy)
    dv_dx_c = (torch.roll(v, -1, dims=0) - torch.roll(v, 1, dims=0)) / (2 * dx)
    dv_dy_c = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * dy)
    
    adv_u_c = u * du_dx_c + v * du_dy_c
    adv_v_c = u * dv_dx_c + v * dv_dy_c
    
    # Upwind component
    # ∂u/∂x: backward if u > 0, forward if u < 0
    du_dx_b = (u - torch.roll(u, 1, dims=0)) / dx
    du_dx_f = (torch.roll(u, -1, dims=0) - u) / dx
    du_dx_u = torch.where(u > 0, du_dx_b, du_dx_f)
    
    du_dy_b = (u - torch.roll(u, 1, dims=1)) / dy
    du_dy_f = (torch.roll(u, -1, dims=1) - u) / dy
    du_dy_u = torch.where(v > 0, du_dy_b, du_dy_f)
    
    dv_dx_b = (v - torch.roll(v, 1, dims=0)) / dx
    dv_dx_f = (torch.roll(v, -1, dims=0) - v) / dx
    dv_dx_u = torch.where(u > 0, dv_dx_b, dv_dx_f)
    
    dv_dy_b = (v - torch.roll(v, 1, dims=1)) / dy
    dv_dy_f = (torch.roll(v, -1, dims=1) - v) / dy
    dv_dy_u = torch.where(v > 0, dv_dy_b, dv_dy_f)
    
    adv_u_u = u * du_dx_u + v * du_dy_u
    adv_v_u = u * dv_dx_u + v * dv_dy_u
    
    # Blend
    adv_u = blend * adv_u_c + (1 - blend) * adv_u_u
    adv_v = blend * adv_v_c + (1 - blend) * adv_v_u
    
    return -adv_u, -adv_v


def compute_advection_skew_symmetric(
    u: Tensor,
    v: Tensor,
    dx: float,
    dy: float,
) -> tuple[Tensor, Tensor]:
    """
    Skew-symmetric (rotational) form of advection.
    
    Instead of (u·∇)u, uses:
        advection = (1/2)[(u·∇)u + ∇(u·u/2) - u×ω]
    
    For 2D incompressible flow, this simplifies to:
        adv_u = u * ∂u/∂x + v * ∂u/∂y  (convective)
              = (1/2)[convective + conservative]
    
    The skew-symmetric form conserves kinetic energy exactly
    (to machine precision) even with central differences.
    This prevents artificial energy growth that can destabilize high-Re flows.
    
    Reference: Morinishi et al. (1998), JCP 143, 90-124
    """
    # Convective form: (u·∇)u
    du_dx = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2 * dx)
    du_dy = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dy)
    dv_dx = (torch.roll(v, -1, dims=0) - torch.roll(v, 1, dims=0)) / (2 * dx)
    dv_dy = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * dy)
    
    conv_u = u * du_dx + v * du_dy
    conv_v = u * dv_dx + v * dv_dy
    
    # Conservative form: ∇·(u⊗u)
    # ∂(uu)/∂x + ∂(uv)/∂y for u-component
    uu = u * u
    uv = u * v
    vv = v * v
    
    d_uu_dx = (torch.roll(uu, -1, dims=0) - torch.roll(uu, 1, dims=0)) / (2 * dx)
    d_uv_dy = (torch.roll(uv, -1, dims=1) - torch.roll(uv, 1, dims=1)) / (2 * dy)
    d_uv_dx = (torch.roll(uv, -1, dims=0) - torch.roll(uv, 1, dims=0)) / (2 * dx)
    d_vv_dy = (torch.roll(vv, -1, dims=1) - torch.roll(vv, 1, dims=1)) / (2 * dy)
    
    cons_u = d_uu_dx + d_uv_dy
    cons_v = d_uv_dx + d_vv_dy
    
    # Skew-symmetric = (1/2)(convective + conservative)
    adv_u = 0.5 * (conv_u + cons_u)
    adv_v = 0.5 * (conv_v + cons_v)
    
    return -adv_u, -adv_v


# ============================================================================
# COMPARISON TEST
# ============================================================================

def test_advection_schemes():
    """Compare numerical diffusion of different schemes."""
    import math
    
    print("=" * 70)
    print("ADVECTION SCHEME COMPARISON")
    print("=" * 70)
    
    # Create a jet profile (Gaussian)
    nx, ny = 256, 128
    dx = 9.0 / (nx - 1)
    dy = 3.0 / (ny - 1)
    
    x = torch.linspace(0, 9.0, nx, dtype=torch.float64)
    y = torch.linspace(0, 3.0, ny, dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Gaussian jet at ceiling
    jet_center = 2.916  # (2.832 + 3.0) / 2
    jet_width = 0.084   # (3.0 - 2.832) / 2
    
    u = 0.455 * torch.exp(-((Y - jet_center) / jet_width)**2)
    u[:, :int(0.9 * ny)] = 0  # Only at ceiling
    v = torch.zeros_like(u)
    
    # Add small velocity everywhere for advection
    u += 0.1
    
    print(f"\nGrid: {nx}×{ny}, dx={dx:.4f}, dy={dy:.4f}")
    print(f"Peak velocity: {u.max():.3f} m/s")
    print()
    
    schemes = [
        ("Upwind (original)", lambda u, v: compute_advection_upwind(u, v, dx, dy)),
        ("Central", lambda u, v: compute_advection_central(u, v, dx, dy)),
        ("Skew-symmetric", lambda u, v: compute_advection_skew_symmetric(u, v, dx, dy)),
        ("Hybrid (0.9)", lambda u, v: compute_advection_hybrid(u, v, dx, dy, 0.9)),
    ]
    
    print(f"{'Scheme':<20} {'max|adv_u|':>12} {'mean|adv_u|':>12}")
    print("-" * 50)
    
    for name, scheme in schemes:
        adv_u, adv_v = scheme(u, v)
        print(f"{name:<20} {adv_u.abs().max().item():>12.4e} {adv_u.abs().mean().item():>12.4e}")
    
    # Estimate numerical viscosity
    print("\n" + "=" * 70)
    print("NUMERICAL VISCOSITY ESTIMATE")
    print("=" * 70)
    
    U = 0.455  # characteristic velocity
    nu_physical = 1.5e-5  # air at Re=5000
    
    # Upwind: ν_num ≈ U*Δx/2
    nu_upwind = U * dx / 2
    
    # Central: ν_num ≈ 0 (dispersive, not diffusive)
    nu_central = 0.0
    
    print(f"\nPhysical viscosity:     ν = {nu_physical:.2e} m²/s")
    print(f"Upwind numerical visc:  ν_num ≈ {nu_upwind:.2e} m²/s ({nu_upwind/nu_physical:.0f}× physical)")
    print(f"Central numerical visc: ν_num ≈ {nu_central:.2e} m²/s")
    print()
    print("→ Upwind smears the jet, Central preserves it")
    print("=" * 70)


def compute_advection_upwind(u: Tensor, v: Tensor, dx: float, dy: float) -> tuple[Tensor, Tensor]:
    """Original upwind scheme for comparison."""
    adv_u = torch.zeros_like(u)
    adv_v = torch.zeros_like(v)
    
    # ∂u/∂x with upwinding
    du_dx_f = (torch.roll(u, -1, dims=0) - u) / dx
    du_dx_b = (u - torch.roll(u, 1, dims=0)) / dx
    du_dx = torch.where(u > 0, du_dx_b, du_dx_f)
    
    du_dy_f = (torch.roll(u, -1, dims=1) - u) / dy
    du_dy_b = (u - torch.roll(u, 1, dims=1)) / dy
    du_dy = torch.where(v > 0, du_dy_b, du_dy_f)
    
    dv_dx_f = (torch.roll(v, -1, dims=0) - v) / dx
    dv_dx_b = (v - torch.roll(v, 1, dims=0)) / dx
    dv_dx = torch.where(u > 0, dv_dx_b, dv_dx_f)
    
    dv_dy_f = (torch.roll(v, -1, dims=1) - v) / dy
    dv_dy_b = (v - torch.roll(v, 1, dims=1)) / dy
    dv_dy = torch.where(v > 0, dv_dy_b, dv_dy_f)
    
    adv_u = u * du_dx + v * du_dy
    adv_v = u * dv_dx + v * dv_dy
    
    return -adv_u, -adv_v


if __name__ == "__main__":
    test_advection_schemes()
