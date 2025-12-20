"""
Navier-Stokes Viscous Terms
===========================

This module implements the viscous flux contributions for the
compressible Navier-Stokes equations, extending the inviscid Euler
solver to handle boundary layers, heat transfer, and viscous dissipation.

The compressible Navier-Stokes equations:

    ∂ρ/∂t + ∇·(ρv) = 0
    ∂(ρv)/∂t + ∇·(ρv⊗v + pI) = ∇·τ
    ∂E/∂t + ∇·((E+p)v) = ∇·(τ·v) - ∇·q

where:
    τ = μ(∇v + ∇vᵀ - 2/3(∇·v)I)    (Newtonian stress tensor)
    q = -k∇T                         (Fourier heat conduction)

Transport Properties:
    μ(T) = Sutherland's law for dynamic viscosity
    k(T) = μ·cₚ/Pr for thermal conductivity (Prandtl number)

For hypersonic flows, viscous effects create:
    - Boundary layers with significant heating
    - Shock-boundary layer interactions
    - Heat shield thermal loads

References:
    [1] Anderson, "Hypersonic and High-Temperature Gas Dynamics", 2006
    [2] White, "Viscous Fluid Flow", 3rd ed., 2006
    [3] Sutherland, Phil. Mag. 36:507-531, 1893
"""

from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional
import math


# Physical constants for air
GAMMA_AIR = 1.4
R_AIR = 287.058  # J/(kg·K) - specific gas constant
CP_AIR = GAMMA_AIR * R_AIR / (GAMMA_AIR - 1)  # J/(kg·K)
CV_AIR = R_AIR / (GAMMA_AIR - 1)  # J/(kg·K)
PR_AIR = 0.72  # Prandtl number for air

# Sutherland's law constants for air
MU_REF = 1.716e-5  # Pa·s at T_ref
T_REF = 273.15  # K
S_MU = 110.4  # K - Sutherland temperature


@dataclass 
class TransportProperties:
    """Container for transport properties at a point."""
    mu: Tensor  # Dynamic viscosity [Pa·s]
    k: Tensor   # Thermal conductivity [W/(m·K)]
    Pr: float = PR_AIR  # Prandtl number


def sutherland_viscosity(
    T: Tensor,
    mu_ref: float = MU_REF,
    T_ref: float = T_REF,
    S: float = S_MU
) -> Tensor:
    """
    Compute dynamic viscosity using Sutherland's law.
    
    μ(T) = μ_ref * (T/T_ref)^(3/2) * (T_ref + S) / (T + S)
    
    Valid for T > 100 K (breaks down at very low temperatures).
    
    Args:
        T: Temperature field [K]
        mu_ref: Reference viscosity [Pa·s]
        T_ref: Reference temperature [K]
        S: Sutherland temperature [K]
        
    Returns:
        Dynamic viscosity field [Pa·s]
        
    Example:
        >>> T = torch.tensor([300.0, 500.0, 1000.0])
        >>> mu = sutherland_viscosity(T)
        >>> print(f"μ(300K) = {mu[0]:.3e} Pa·s")
    """
    T_ratio = T / T_ref
    return mu_ref * T_ratio**1.5 * (T_ref + S) / (T + S)


def thermal_conductivity(
    mu: Tensor,
    cp: float = CP_AIR,
    Pr: float = PR_AIR
) -> Tensor:
    """
    Compute thermal conductivity from viscosity via Prandtl number.
    
    k = μ * cₚ / Pr
    
    Args:
        mu: Dynamic viscosity [Pa·s]
        cp: Specific heat at constant pressure [J/(kg·K)]
        Pr: Prandtl number
        
    Returns:
        Thermal conductivity [W/(m·K)]
    """
    return mu * cp / Pr


def compute_transport_properties(
    T: Tensor,
    cp: float = CP_AIR,
    Pr: float = PR_AIR
) -> TransportProperties:
    """
    Compute all transport properties from temperature.
    
    Args:
        T: Temperature field [K]
        cp: Specific heat at constant pressure
        Pr: Prandtl number
        
    Returns:
        TransportProperties with μ and k fields
    """
    mu = sutherland_viscosity(T)
    k = thermal_conductivity(mu, cp, Pr)
    return TransportProperties(mu=mu, k=k, Pr=Pr)


def velocity_gradients_2d(
    u: Tensor,
    v: Tensor,
    dx: float,
    dy: float
) -> dict[str, Tensor]:
    """
    Compute velocity gradients using central differences.
    
    Args:
        u, v: Velocity components [Ny, Nx]
        dx, dy: Grid spacing
        
    Returns:
        Dictionary with du/dx, du/dy, dv/dx, dv/dy
    """
    # Central differences in interior, one-sided at boundaries
    dudx = torch.zeros_like(u)
    dudy = torch.zeros_like(u)
    dvdx = torch.zeros_like(v)
    dvdy = torch.zeros_like(v)
    
    # Interior: central differences
    dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
    dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
    dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
    
    # Boundaries: one-sided differences
    dudx[:, 0] = (u[:, 1] - u[:, 0]) / dx
    dudx[:, -1] = (u[:, -1] - u[:, -2]) / dx
    dudy[0, :] = (u[1, :] - u[0, :]) / dy
    dudy[-1, :] = (u[-1, :] - u[-2, :]) / dy
    
    dvdx[:, 0] = (v[:, 1] - v[:, 0]) / dx
    dvdx[:, -1] = (v[:, -1] - v[:, -2]) / dx
    dvdy[0, :] = (v[1, :] - v[0, :]) / dy
    dvdy[-1, :] = (v[-1, :] - v[-2, :]) / dy
    
    return {
        'dudx': dudx,
        'dudy': dudy,
        'dvdx': dvdx,
        'dvdy': dvdy,
    }


def temperature_gradient_2d(
    T: Tensor,
    dx: float,
    dy: float
) -> tuple[Tensor, Tensor]:
    """
    Compute temperature gradients using central differences.
    
    Args:
        T: Temperature field [Ny, Nx]
        dx, dy: Grid spacing
        
    Returns:
        Tuple of (dT/dx, dT/dy)
    """
    dTdx = torch.zeros_like(T)
    dTdy = torch.zeros_like(T)
    
    # Interior: central differences
    dTdx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * dx)
    dTdy[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2 * dy)
    
    # Boundaries: one-sided
    dTdx[:, 0] = (T[:, 1] - T[:, 0]) / dx
    dTdx[:, -1] = (T[:, -1] - T[:, -2]) / dx
    dTdy[0, :] = (T[1, :] - T[0, :]) / dy
    dTdy[-1, :] = (T[-1, :] - T[-2, :]) / dy
    
    return dTdx, dTdy


def stress_tensor_2d(
    grads: dict[str, Tensor],
    mu: Tensor
) -> dict[str, Tensor]:
    """
    Compute viscous stress tensor components.
    
    τ = μ(∇v + ∇vᵀ - 2/3(∇·v)I)
    
    For 2D:
        τ_xx = μ(2∂u/∂x - 2/3(∂u/∂x + ∂v/∂y))
        τ_yy = μ(2∂v/∂y - 2/3(∂u/∂x + ∂v/∂y))
        τ_xy = μ(∂u/∂y + ∂v/∂x)
    
    Args:
        grads: Velocity gradients from velocity_gradients_2d
        mu: Dynamic viscosity field
        
    Returns:
        Dictionary with tau_xx, tau_yy, tau_xy
    """
    dudx = grads['dudx']
    dudy = grads['dudy']
    dvdx = grads['dvdx']
    dvdy = grads['dvdy']
    
    # Divergence of velocity
    div_v = dudx + dvdy
    
    # Stress components
    tau_xx = mu * (2 * dudx - (2/3) * div_v)
    tau_yy = mu * (2 * dvdy - (2/3) * div_v)
    tau_xy = mu * (dudy + dvdx)
    
    return {
        'tau_xx': tau_xx,
        'tau_yy': tau_yy,
        'tau_xy': tau_xy,
    }


def heat_flux_2d(
    dTdx: Tensor,
    dTdy: Tensor,
    k: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Compute heat flux via Fourier's law.
    
    q = -k∇T
    
    Args:
        dTdx, dTdy: Temperature gradients
        k: Thermal conductivity
        
    Returns:
        Tuple of (qx, qy) heat flux components
    """
    qx = -k * dTdx
    qy = -k * dTdy
    return qx, qy


def viscous_flux_x_2d(
    u: Tensor,
    v: Tensor,
    tau: dict[str, Tensor],
    qx: Tensor
) -> Tensor:
    """
    Compute x-direction viscous flux vector.
    
    F_v = [0, τ_xx, τ_xy, u·τ_xx + v·τ_xy - qx]ᵀ
    
    Args:
        u, v: Velocity components
        tau: Stress tensor components
        qx: Heat flux in x-direction
        
    Returns:
        Viscous flux tensor [4, Ny, Nx]
    """
    Ny, Nx = u.shape
    F = torch.zeros(4, Ny, Nx, dtype=u.dtype, device=u.device)
    
    # Mass: no viscous flux
    F[0] = 0
    
    # x-momentum: τ_xx
    F[1] = tau['tau_xx']
    
    # y-momentum: τ_xy
    F[2] = tau['tau_xy']
    
    # Energy: viscous work + heat conduction
    F[3] = u * tau['tau_xx'] + v * tau['tau_xy'] - qx
    
    return F


def viscous_flux_y_2d(
    u: Tensor,
    v: Tensor,
    tau: dict[str, Tensor],
    qy: Tensor
) -> Tensor:
    """
    Compute y-direction viscous flux vector.
    
    G_v = [0, τ_xy, τ_yy, u·τ_xy + v·τ_yy - qy]ᵀ
    
    Args:
        u, v: Velocity components
        tau: Stress tensor components
        qy: Heat flux in y-direction
        
    Returns:
        Viscous flux tensor [4, Ny, Nx]
    """
    Ny, Nx = u.shape
    G = torch.zeros(4, Ny, Nx, dtype=u.dtype, device=u.device)
    
    # Mass: no viscous flux
    G[0] = 0
    
    # x-momentum: τ_xy
    G[1] = tau['tau_xy']
    
    # y-momentum: τ_yy
    G[2] = tau['tau_yy']
    
    # Energy: viscous work + heat conduction
    G[3] = u * tau['tau_xy'] + v * tau['tau_yy'] - qy
    
    return G


def viscous_flux_divergence_2d(
    Fv: Tensor,
    Gv: Tensor,
    dx: float,
    dy: float
) -> Tensor:
    """
    Compute divergence of viscous fluxes.
    
    ∇·F_v = ∂F_v/∂x + ∂G_v/∂y
    
    Uses central differences for second-order accuracy.
    
    Args:
        Fv: x-direction viscous flux [4, Ny, Nx]
        Gv: y-direction viscous flux [4, Ny, Nx]
        dx, dy: Grid spacing
        
    Returns:
        Divergence tensor [4, Ny, Nx]
    """
    div = torch.zeros_like(Fv)
    
    # Central differences in interior
    div[:, :, 1:-1] += (Fv[:, :, 2:] - Fv[:, :, :-2]) / (2 * dx)
    div[:, 1:-1, :] += (Gv[:, 2:, :] - Gv[:, :-2, :]) / (2 * dy)
    
    # One-sided at boundaries
    div[:, :, 0] += (Fv[:, :, 1] - Fv[:, :, 0]) / dx
    div[:, :, -1] += (Fv[:, :, -1] - Fv[:, :, -2]) / dx
    div[:, 0, :] += (Gv[:, 1, :] - Gv[:, 0, :]) / dy
    div[:, -1, :] += (Gv[:, -1, :] - Gv[:, -2, :]) / dy
    
    return div


def compute_viscous_rhs_2d(
    rho: Tensor,
    u: Tensor,
    v: Tensor,
    p: Tensor,
    dx: float,
    dy: float,
    gamma: float = GAMMA_AIR,
    R: float = R_AIR,
    Pr: float = PR_AIR
) -> Tensor:
    """
    Compute complete viscous RHS for 2D Navier-Stokes.
    
    This computes ∇·F_v (viscous flux divergence) to be ADDED
    to the inviscid Euler RHS.
    
    Args:
        rho: Density [kg/m³]
        u, v: Velocity components [m/s]
        p: Pressure [Pa]
        dx, dy: Grid spacing [m]
        gamma: Ratio of specific heats
        R: Specific gas constant [J/(kg·K)]
        Pr: Prandtl number
        
    Returns:
        Viscous source term [4, Ny, Nx] to ADD to RHS
    """
    # Temperature from ideal gas law: T = p / (ρR)
    T = p / (rho * R)
    
    # Transport properties
    cp = gamma * R / (gamma - 1)
    mu = sutherland_viscosity(T)
    k = thermal_conductivity(mu, cp, Pr)
    
    # Velocity gradients
    grads = velocity_gradients_2d(u, v, dx, dy)
    
    # Stress tensor
    tau = stress_tensor_2d(grads, mu)
    
    # Temperature gradients and heat flux
    dTdx, dTdy = temperature_gradient_2d(T, dx, dy)
    qx, qy = heat_flux_2d(dTdx, dTdy, k)
    
    # Viscous fluxes
    Fv = viscous_flux_x_2d(u, v, tau, qx)
    Gv = viscous_flux_y_2d(u, v, tau, qy)
    
    # Divergence (this is the RHS contribution)
    return viscous_flux_divergence_2d(Fv, Gv, dx, dy)


def reynolds_number(
    rho: float,
    u: float,
    L: float,
    mu: float
) -> float:
    """
    Compute Reynolds number.
    
    Re = ρuL/μ
    
    Args:
        rho: Density [kg/m³]
        u: Velocity [m/s]
        L: Characteristic length [m]
        mu: Dynamic viscosity [Pa·s]
        
    Returns:
        Reynolds number (dimensionless)
    """
    return rho * u * L / mu


def viscous_timestep_limit(
    dx: float,
    dy: float,
    mu_max: float,
    rho_min: float,
    safety: float = 0.25
) -> float:
    """
    Compute viscous stability limit for timestep.
    
    For explicit viscous terms:
        Δt < safety * min(Δx², Δy²) * ρ / (2μ)
    
    Args:
        dx, dy: Grid spacing
        mu_max: Maximum viscosity in domain
        rho_min: Minimum density in domain
        safety: Safety factor
        
    Returns:
        Maximum stable timestep
    """
    dmin_sq = min(dx**2, dy**2)
    return safety * dmin_sq * rho_min / (2 * mu_max)


def prandtl_meyer_function(M: float, gamma: float = 1.4) -> float:
    """
    Prandtl-Meyer expansion function.
    
    ν(M) = sqrt((γ+1)/(γ-1)) * arctan(sqrt((γ-1)/(γ+1)*(M²-1))) - arctan(sqrt(M²-1))
    
    Used for expansion fans in supersonic flow.
    
    Args:
        M: Mach number (> 1)
        gamma: Ratio of specific heats
        
    Returns:
        Prandtl-Meyer angle [radians]
    """
    if M < 1:
        return 0.0
    
    gr = (gamma - 1) / (gamma + 1)
    term1 = math.sqrt(1/gr) * math.atan(math.sqrt(gr * (M**2 - 1)))
    term2 = math.atan(math.sqrt(M**2 - 1))
    return term1 - term2


def stagnation_temperature(T: float, M: float, gamma: float = 1.4) -> float:
    """
    Compute stagnation (total) temperature.
    
    T₀/T = 1 + (γ-1)/2 * M²
    
    Args:
        T: Static temperature [K]
        M: Mach number
        gamma: Ratio of specific heats
        
    Returns:
        Stagnation temperature [K]
    """
    return T * (1 + (gamma - 1) / 2 * M**2)


def recovery_temperature(T_inf: float, M: float, r: float = 0.85, gamma: float = 1.4) -> float:
    """
    Compute adiabatic wall (recovery) temperature.
    
    T_r/T_∞ = 1 + r * (γ-1)/2 * M²
    
    where r is the recovery factor:
        r ≈ Pr^(1/2) for laminar boundary layers
        r ≈ Pr^(1/3) for turbulent boundary layers
    
    Args:
        T_inf: Freestream temperature [K]
        M: Freestream Mach number
        r: Recovery factor (0.85 typical for Pr=0.72 laminar)
        gamma: Ratio of specific heats
        
    Returns:
        Recovery temperature [K]
    """
    return T_inf * (1 + r * (gamma - 1) / 2 * M**2)


def heat_transfer_coefficient(
    q_wall: float,
    T_wall: float,
    T_recovery: float
) -> float:
    """
    Compute heat transfer coefficient.
    
    h = q_w / (T_r - T_w)
    
    Args:
        q_wall: Wall heat flux [W/m²]
        T_wall: Wall temperature [K]
        T_recovery: Recovery temperature [K]
        
    Returns:
        Heat transfer coefficient [W/(m²·K)]
    """
    return q_wall / (T_recovery - T_wall) if abs(T_recovery - T_wall) > 1e-10 else 0.0


def stanton_number(h: float, rho: float, u: float, cp: float = CP_AIR) -> float:
    """
    Compute Stanton number.
    
    St = h / (ρ u cₚ)
    
    Args:
        h: Heat transfer coefficient [W/(m²·K)]
        rho: Density [kg/m³]
        u: Velocity [m/s]
        cp: Specific heat [J/(kg·K)]
        
    Returns:
        Stanton number (dimensionless)
    """
    return h / (rho * u * cp) if rho * u > 1e-10 else 0.0
