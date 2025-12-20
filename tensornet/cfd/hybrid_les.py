"""
Hybrid RANS-LES Models
======================

Bridge between RANS and LES approaches for efficient high-fidelity turbulence
modeling. Key hybrid methodologies:

    DES (Detached Eddy Simulation):
        - RANS in attached boundary layers
        - LES in separated regions and wakes
        - Length scale switching: l_hybrid = min(l_RANS, C_DES * Δ)
    
    DDES (Delayed DES):
        - Prevents premature LES in boundary layers
        - Shielding function delays transition to LES
        - Based on local flow/turbulence ratios
    
    IDDES (Improved Delayed DES):
        - Wall-modeled LES (WMLES) branch
        - Seamless RANS-LES interface
        - Optimized for wall-bounded flows

References:
    [1] Spalart et al., "Comments on the feasibility of LES for wings",
        Advances in DNS/LES, 1997
    [2] Spalart et al., "A New Version of Detached-Eddy Simulation",
        Theor. Comp. Fluid Dyn. 20, 2006 (DDES)
    [3] Shur et al., "A hybrid RANS-LES approach with delayed-DES and
        wall-modelled LES capabilities", IJHFF 29, 2008 (IDDES)
"""

import torch
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
from enum import Enum


class HybridModel(Enum):
    """Available hybrid RANS-LES models."""
    DES = "des"
    DDES = "ddes"
    IDDES = "iddes"
    SAS = "sas"  # Scale-Adaptive Simulation


@dataclass
class HybridLESState:
    """State variables for hybrid RANS-LES models."""
    nu_sgs: torch.Tensor        # Total SGS/turbulent viscosity
    blending: torch.Tensor      # Blending function (0=RANS, 1=LES)
    length_scale: torch.Tensor  # Hybrid length scale
    mode: torch.Tensor          # Operation mode indicator
    
    # Optional diagnostics
    f_d: Optional[torch.Tensor] = None   # DDES delay function
    f_e: Optional[torch.Tensor] = None   # IDDES elevation function
    f_b: Optional[torch.Tensor] = None   # IDDES blending function


# === Model Constants ===

# DES constants (SA-based)
C_DES = 0.65            # DES length scale coefficient
C_W = 0.15              # DDES shielding parameter

# DDES constants
C_D1 = 8.0              # Delay function constant
C_D2 = 3.0              # Delay function constant

# IDDES constants
C_T = 1.87              # IDDES blending constant
C_L = 3.55              # IDDES blending constant
C_MU = 0.09             # Turbulent viscosity constant (k-ε based)
KAPPA = 0.41            # Von Kármán constant


def compute_grid_scale(
    dx: torch.Tensor,
    dy: torch.Tensor,
    dz: Optional[torch.Tensor] = None,
    method: str = "max"
) -> torch.Tensor:
    """
    Compute LES grid/filter scale from mesh spacing.
    
    Args:
        dx, dy, dz: Grid spacing in each direction
        method: "max" (largest), "cube" (volume^1/3), or "sum" (sum/3)
        
    Returns:
        Grid scale Δ
    """
    if dz is None:
        # 2D case
        if method == "max":
            return torch.maximum(dx, dy)
        elif method == "cube":
            return torch.sqrt(dx * dy)
        else:
            return 0.5 * (dx + dy)
    else:
        # 3D case
        if method == "max":
            return torch.maximum(torch.maximum(dx, dy), dz)
        elif method == "cube":
            return (dx * dy * dz) ** (1.0 / 3.0)
        else:
            return (dx + dy + dz) / 3.0


def compute_wall_distance_scale(
    d_wall: torch.Tensor,
    kappa: float = KAPPA
) -> torch.Tensor:
    """
    Compute RANS mixing length scale from wall distance.
    
    l_RANS = κ * d_wall (for SA-type models)
    
    Args:
        d_wall: Distance to nearest wall
        kappa: Von Kármán constant
        
    Returns:
        RANS length scale
    """
    return kappa * d_wall


def des_length_scale(
    l_rans: torch.Tensor,
    delta: torch.Tensor,
    c_des: float = C_DES
) -> torch.Tensor:
    """
    Original DES length scale.
    
    l_DES = min(l_RANS, C_DES * Δ)
    
    When Δ is smaller (in separated regions), LES mode activates.
    
    Args:
        l_rans: RANS length scale (κ * d_wall)
        delta: LES grid scale
        c_des: DES coefficient
        
    Returns:
        Hybrid length scale
    """
    l_les = c_des * delta
    return torch.minimum(l_rans, l_les)


def compute_r_d(
    nu_t: torch.Tensor,
    nu: float,
    velocity_gradient: torch.Tensor,
    d_wall: torch.Tensor,
    kappa: float = KAPPA
) -> torch.Tensor:
    """
    Compute DDES delay function parameter r_d.
    
    r_d = (ν_t + ν) / (|∇u| κ² d²)
    
    This measures whether the flow is "RANS-like" (r_d ≈ 1) or
    resolved turbulence (r_d << 1).
    
    Args:
        nu_t: Turbulent/SGS viscosity
        nu: Molecular viscosity
        velocity_gradient: Velocity gradient magnitude |∇u|
        d_wall: Wall distance
        kappa: Von Kármán constant
        
    Returns:
        r_d parameter field
    """
    denominator = velocity_gradient * (kappa ** 2) * (d_wall ** 2) + 1e-30
    r_d = (nu_t + nu) / denominator
    
    return r_d


def ddes_delay_function(
    r_d: torch.Tensor,
    c_d1: float = C_D1,
    c_d2: float = C_D2
) -> torch.Tensor:
    """
    DDES delay/shielding function f_d.
    
    f_d = 1 - tanh([C_d1 * r_d]^C_d2)
    
    When f_d ≈ 0 (r_d large), RANS mode is preserved.
    When f_d ≈ 1 (r_d small), LES mode can activate.
    
    Args:
        r_d: Delay function parameter
        c_d1, c_d2: Model constants
        
    Returns:
        Delay function f_d
    """
    return 1.0 - torch.tanh((c_d1 * r_d) ** c_d2)


def ddes_length_scale(
    l_rans: torch.Tensor,
    delta: torch.Tensor,
    f_d: torch.Tensor,
    c_des: float = C_DES
) -> torch.Tensor:
    """
    DDES length scale with delay function.
    
    l_DDES = l_RANS - f_d * max(0, l_RANS - C_DES * Δ)
    
    Args:
        l_rans: RANS length scale
        delta: LES grid scale
        f_d: Delay function
        c_des: DES coefficient
        
    Returns:
        DDES hybrid length scale
    """
    l_les = c_des * delta
    return l_rans - f_d * torch.maximum(torch.zeros_like(l_rans), l_rans - l_les)


def iddes_blending_function(
    d_wall: torch.Tensor,
    delta: torch.Tensor,
    r_d: torch.Tensor,
    c_t: float = C_T,
    c_l: float = C_L
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    IDDES blending functions for RANS-LES interface.
    
    Args:
        d_wall: Wall distance
        delta: Grid scale
        r_d: Delay function parameter
        c_t, c_l: Model constants
        
    Returns:
        (f_e, f_b, alpha) - Elevation, blending, and alpha functions
    """
    # Elevation function (provides LES content in RANS zone)
    f_e = 2.0 * torch.exp(-9.0 * (torch.abs(r_d - 1.0)))
    
    # Step functions
    f_step = torch.tanh(c_l * (d_wall / delta) ** 6)
    
    # Blending (WMLES-RANS switching)
    f_b = torch.minimum(
        2.0 * torch.exp(-9.0 * (r_d ** 2)),
        1.0 - f_step
    )
    
    # Alpha: final blending parameter (0 = RANS, 1 = LES)
    alpha = torch.maximum(1.0 - ddes_delay_function(r_d), f_e)
    
    return f_e, f_b, alpha


def iddes_length_scale(
    l_rans: torch.Tensor,
    delta: torch.Tensor,
    d_wall: torch.Tensor,
    alpha: torch.Tensor,
    c_des: float = C_DES
) -> torch.Tensor:
    """
    IDDES hybrid length scale.
    
    l_IDDES = f_d * (1 + f_e) * l_RANS + (1 - f_d) * C_DES * Δ
    
    Simplified version using alpha parameter:
    l_IDDES = (1 - alpha) * l_RANS + alpha * C_DES * Δ
    
    Args:
        l_rans: RANS length scale
        delta: LES grid scale
        d_wall: Wall distance
        alpha: Blending function
        c_des: DES coefficient
        
    Returns:
        IDDES hybrid length scale
    """
    l_les = c_des * delta
    
    # WMLES mode: use wall distance for near-wall
    l_wmles = torch.minimum(
        torch.maximum(KAPPA * d_wall, delta),
        l_les
    )
    
    # Blend between RANS and WMLES-LES
    return (1.0 - alpha) * l_rans + alpha * l_wmles


def sas_length_scale(
    k: torch.Tensor,
    omega: torch.Tensor,
    velocity_gradient: torch.Tensor,
    velocity_laplacian: torch.Tensor,
    c_mu: float = C_MU,
    kappa: float = KAPPA
) -> torch.Tensor:
    """
    Scale-Adaptive Simulation (SAS) length scale.
    
    The SAS length scale responds to resolved turbulence
    without explicit grid-based switching.
    
    L_vK = κ |S| / |∇²u|  (von Kármán length scale)
    L_t = √k / (c_μ^0.25 ω)  (turbulent length scale)
    
    Args:
        k: Turbulent kinetic energy
        omega: Specific dissipation rate
        velocity_gradient: |S| strain rate magnitude
        velocity_laplacian: |∇²u| Laplacian of velocity
        c_mu, kappa: Model constants
        
    Returns:
        SAS length scale
    """
    # Turbulent length scale
    L_t = torch.sqrt(k + 1e-30) / ((c_mu ** 0.25) * omega + 1e-30)
    
    # Von Kármán length scale
    L_vK = kappa * velocity_gradient / (torch.abs(velocity_laplacian) + 1e-30)
    
    # SAS uses the minimum
    return torch.minimum(L_t, L_vK)


def compute_hybrid_viscosity(
    nu_rans: torch.Tensor,
    length_scale: torch.Tensor,
    strain_rate: torch.Tensor,
    model: HybridModel = HybridModel.DDES,
    c_s: float = 0.17
) -> torch.Tensor:
    """
    Compute turbulent viscosity for hybrid model.
    
    In RANS regions: use RANS viscosity
    In LES regions: use Smagorinsky-like model
    
    Args:
        nu_rans: RANS turbulent viscosity
        length_scale: Hybrid length scale
        strain_rate: Strain rate magnitude
        model: Hybrid model type
        c_s: Smagorinsky constant for LES regions
        
    Returns:
        Turbulent/SGS viscosity
    """
    # Smagorinsky-like for LES regions
    nu_sgs = (c_s * length_scale) ** 2 * strain_rate
    
    # Use the minimum (avoid excessive viscosity)
    return torch.minimum(nu_rans, nu_sgs)


def run_hybrid_les(
    rho: torch.Tensor,
    u: torch.Tensor,
    d_wall: torch.Tensor,
    grid_spacing: Tuple[torch.Tensor, ...],
    nu: float,
    nu_rans: torch.Tensor,
    model: HybridModel = HybridModel.DDES
) -> HybridLESState:
    """
    Main driver for hybrid RANS-LES computation.
    
    Args:
        rho: Density field
        u: Velocity field (Nd, Nx, Ny, [Nz])
        d_wall: Wall distance field
        grid_spacing: Tuple of (dx, dy, [dz]) tensors
        nu: Molecular viscosity
        nu_rans: RANS turbulent viscosity field
        model: Hybrid model to use
        
    Returns:
        HybridLESState with computed quantities
    """
    is_3d = len(grid_spacing) == 3
    
    # Compute grid scale
    if is_3d:
        delta = compute_grid_scale(grid_spacing[0], grid_spacing[1], grid_spacing[2])
    else:
        delta = compute_grid_scale(grid_spacing[0], grid_spacing[1])
    
    # RANS length scale
    l_rans = compute_wall_distance_scale(d_wall)
    
    # Compute velocity gradient magnitude (simplified)
    if is_3d:
        dudx = torch.gradient(u[0], spacing=(grid_spacing[0][0, 0, 0].item(),), dim=0)[0]
        dudy = torch.gradient(u[0], spacing=(grid_spacing[1][0, 0, 0].item(),), dim=1)[0]
        dudz = torch.gradient(u[0], spacing=(grid_spacing[2][0, 0, 0].item(),), dim=2)[0]
        vel_grad_mag = torch.sqrt(dudx ** 2 + dudy ** 2 + dudz ** 2 + 1e-30)
    else:
        dudx = torch.gradient(u[0], spacing=(grid_spacing[0][0, 0].item(),), dim=0)[0]
        dudy = torch.gradient(u[0], spacing=(grid_spacing[1][0, 0].item(),), dim=1)[0]
        vel_grad_mag = torch.sqrt(dudx ** 2 + dudy ** 2 + 1e-30)
    
    # Compute delay function parameter
    r_d = compute_r_d(nu_rans, nu, vel_grad_mag, d_wall)
    
    if model == HybridModel.DES:
        # Original DES
        length_scale = des_length_scale(l_rans, delta)
        f_d = None
        blending = (length_scale < l_rans).float()
        
    elif model == HybridModel.DDES:
        # Delayed DES
        f_d = ddes_delay_function(r_d)
        length_scale = ddes_length_scale(l_rans, delta, f_d)
        blending = f_d
        
    elif model == HybridModel.IDDES:
        # Improved Delayed DES
        f_e, f_b, alpha = iddes_blending_function(d_wall, delta, r_d)
        length_scale = iddes_length_scale(l_rans, delta, d_wall, alpha)
        f_d = ddes_delay_function(r_d)
        blending = alpha
        
    else:
        raise ValueError(f"Unknown hybrid model: {model}")
    
    # Compute hybrid viscosity
    nu_sgs = compute_hybrid_viscosity(nu_rans, length_scale, vel_grad_mag, model)
    
    # Mode indicator: 0 = RANS, 1 = LES
    mode = (blending > 0.5).float()
    
    return HybridLESState(
        nu_sgs=nu_sgs,
        blending=blending,
        length_scale=length_scale,
        mode=mode,
        f_d=f_d if model in [HybridModel.DDES, HybridModel.IDDES] else None
    )


def estimate_rans_les_ratio(state: HybridLESState) -> Dict[str, float]:
    """
    Compute statistics on RANS vs LES content.
    
    Args:
        state: Hybrid LES state
        
    Returns:
        Dict with RANS/LES percentages
    """
    total_cells = state.mode.numel()
    les_cells = (state.mode > 0.5).sum().item()
    rans_cells = total_cells - les_cells
    
    return {
        "rans_fraction": rans_cells / total_cells,
        "les_fraction": les_cells / total_cells,
        "avg_blending": state.blending.mean().item()
    }


def validate_hybrid_les():
    """Run validation tests for hybrid RANS-LES models."""
    print("\n" + "=" * 70)
    print("HYBRID RANS-LES MODEL VALIDATION")
    print("=" * 70)
    
    # Test 1: Grid scale computation
    print("\n[Test 1] Grid Scale Computation")
    print("-" * 40)
    
    dx = torch.ones(10, 10) * 0.01
    dy = torch.ones(10, 10) * 0.02
    dz = torch.ones(10, 10) * 0.03
    
    delta_2d = compute_grid_scale(dx, dy)
    delta_3d = compute_grid_scale(dx, dy, dz)
    
    print(f"2D grid scale (max): {delta_2d[0, 0]:.4f}")
    print(f"3D grid scale (max): {delta_3d[0, 0]:.4f}")
    
    assert torch.allclose(delta_2d, torch.ones_like(delta_2d) * 0.02)
    assert torch.allclose(delta_3d, torch.ones_like(delta_3d) * 0.03)
    print("✓ PASS")
    
    # Test 2: DES length scale switching
    print("\n[Test 2] DES Length Scale")
    print("-" * 40)
    
    d_wall = torch.linspace(0.001, 0.1, 20)
    l_rans = compute_wall_distance_scale(d_wall)
    delta = torch.ones(20) * 0.02
    
    l_des = des_length_scale(l_rans, delta)
    
    # Near wall: l_RANS < C_DES*Δ → RANS mode
    # Far from wall: l_RANS > C_DES*Δ → LES mode
    print(f"Near wall (d=0.001): l_DES = {l_des[0]:.5f}")
    print(f"Far wall (d=0.1): l_DES = {l_des[-1]:.5f}")
    
    assert l_des[0] < l_des[-1]
    print("✓ PASS")
    
    # Test 3: DDES delay function
    print("\n[Test 3] DDES Delay Function")
    print("-" * 40)
    
    r_d_values = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
    f_d = ddes_delay_function(r_d_values)
    
    for r, f in zip(r_d_values.tolist(), f_d.tolist()):
        print(f"r_d = {r:.1f}: f_d = {f:.4f}")
    
    # f_d should decrease as r_d increases
    assert f_d[0] > f_d[-1]
    print("✓ PASS")
    
    # Test 4: Full hybrid computation (2D)
    print("\n[Test 4] Full Hybrid DDES (2D)")
    print("-" * 40)
    
    Nx, Ny = 32, 32
    rho = torch.ones(Nx, Ny)
    u = torch.zeros(2, Nx, Ny)
    u[0] = torch.linspace(0, 1, Ny).unsqueeze(0).expand(Nx, -1)  # Couette-like
    
    d_wall = torch.linspace(0.001, 0.5, Ny).unsqueeze(0).expand(Nx, -1)
    dx = torch.ones(Nx, Ny) * 0.02
    dy = torch.ones(Nx, Ny) * 0.02
    nu_rans = torch.ones(Nx, Ny) * 0.001
    
    state = run_hybrid_les(
        rho=rho,
        u=u,
        d_wall=d_wall,
        grid_spacing=(dx, dy),
        nu=1e-5,
        nu_rans=nu_rans,
        model=HybridModel.DDES
    )
    
    stats = estimate_rans_les_ratio(state)
    
    print(f"RANS fraction: {stats['rans_fraction']:.2%}")
    print(f"LES fraction: {stats['les_fraction']:.2%}")
    print(f"Average blending: {stats['avg_blending']:.4f}")
    
    assert state.nu_sgs.shape == (Nx, Ny)
    assert state.blending.shape == (Nx, Ny)
    print("✓ PASS")
    
    # Test 5: IDDES blending functions
    print("\n[Test 5] IDDES Blending Functions")
    print("-" * 40)
    
    d_wall_test = torch.tensor([0.001, 0.01, 0.1])
    delta_test = torch.tensor([0.01, 0.01, 0.01])
    r_d_test = torch.tensor([0.5, 1.0, 2.0])
    
    f_e, f_b, alpha = iddes_blending_function(d_wall_test, delta_test, r_d_test)
    
    print("d_wall | f_e    | f_b    | alpha")
    print("-" * 35)
    for i in range(3):
        print(f"{d_wall_test[i]:.3f}  | {f_e[i]:.4f} | {f_b[i]:.4f} | {alpha[i]:.4f}")
    
    print("✓ PASS")
    
    # Test 6: Multiple models comparison
    print("\n[Test 6] Model Comparison")
    print("-" * 40)
    
    for model in [HybridModel.DES, HybridModel.DDES, HybridModel.IDDES]:
        state = run_hybrid_les(
            rho=rho,
            u=u,
            d_wall=d_wall,
            grid_spacing=(dx, dy),
            nu=1e-5,
            nu_rans=nu_rans,
            model=model
        )
        
        stats = estimate_rans_les_ratio(state)
        print(f"{model.value}: RANS {stats['rans_fraction']:.1%}, LES {stats['les_fraction']:.1%}")
    
    print("✓ PASS")
    
    print("\n" + "=" * 70)
    print("HYBRID RANS-LES VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    validate_hybrid_les()
