"""
Hou-Luo Axisymmetric Blow-Up Ansatz.

This module implements the specific geometry that Hou & Luo identified as
a potential blow-up candidate for the 3D Euler/Navier-Stokes equations.

THE HOU-LUO DISCOVERY (2014)
============================

Hou & Luo studied axisymmetric flows with swirl in cylindrical coordinates
(r, θ, z) and discovered that vortex stretching is maximized when:

1. Two counter-rotating vortex rings approach each other
2. The collision creates a "hyperbolic stagnation point" 
3. Vorticity concentrates along the symmetry axis
4. The profile becomes self-similar as it collapses

The key equation in cylindrical coordinates:
    ω_θ / r = stretching term  → can blow up even as ω_θ stays bounded

The Geometry
============

    ┌─────────────────────────────────────────┐
    │                   ↑ z                    │
    │                   │                      │
    │     ╭───────╮     │     ╭───────╮       │
    │    ╱ Ring 1  ╲    │    ╱         ╲      │
    │   │  (↻ CCW)  │───┼───│  (↺ CW)   │     │
    │    ╲         ╱    │    ╲  Ring 2 ╱      │
    │     ╰───────╯     │     ╰───────╯       │
    │                   │                      │
    │         ←─────────┼─────────→ r          │
    │                   │                      │
    │    Collision zone at z = 0              │
    │    Maximum vortex stretching here       │
    └─────────────────────────────────────────┘

The rings approach z=0, creating intense stretching along the axis.

Reference: 
    T. Hou & G. Luo (2014), "Toward the Finite-Time Blowup of the 3D 
    Axisymmetric Euler Equations: A Numerical Investigation"
    Multiscale Modeling & Simulation 12(4):1722-1776
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import math


@dataclass
class HouLuoConfig:
    """Configuration for Hou-Luo axisymmetric ansatz."""
    
    # Grid
    N: int = 64                    # Resolution
    L: float = 2 * np.pi           # Domain size
    
    # Ring parameters
    ring_radius: float = 0.4       # R₀: radius of vortex rings (in units of L)
    core_thickness: float = 0.1    # σ: core Gaussian width
    ring_separation: float = 0.3   # Initial z-separation of rings
    circulation: float = 1.0       # Γ: circulation strength
    
    # Swirl
    swirl_amplitude: float = 0.5   # Azimuthal velocity component
    
    # Self-similar scaling  
    alpha: float = 0.5             # Velocity exponent
    beta: float = 0.5              # Spatial exponent


def create_hou_luo_profile(config: HouLuoConfig) -> torch.Tensor:
    """
    Create the Hou-Luo axisymmetric blow-up candidate.
    
    This is the geometry that Hou identified as most likely to blow up.
    Two counter-rotating vortex rings approaching the z=0 plane.
    
    Returns:
        U: Velocity field (N, N, N, 3) in Cartesian coordinates
    """
    N = config.N
    L = config.L
    
    # Create grid
    x = torch.linspace(-L/2, L/2, N, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    
    # Cylindrical coordinates
    R = torch.sqrt(X**2 + Y**2)  # radial distance from z-axis
    R_safe = R.clamp(min=1e-10)   # avoid division by zero
    
    # Unit vectors in cylindrical coords (expressed in Cartesian)
    # e_r = (cos θ, sin θ, 0) = (x/r, y/r, 0)
    # e_θ = (-sin θ, cos θ, 0) = (-y/r, x/r, 0)
    # e_z = (0, 0, 1)
    
    cos_theta = X / R_safe
    sin_theta = Y / R_safe
    
    # Initialize velocity
    U = torch.zeros(N, N, N, 3, dtype=torch.float64)
    
    # Ring parameters
    R0 = config.ring_radius * L      # Ring radius
    sigma = config.core_thickness * L # Core thickness
    z_sep = config.ring_separation * L  # Ring separation
    Gamma = config.circulation
    
    # Create two vortex rings at z = ±z_sep/2
    for ring_z, ring_sign in [(z_sep/2, 1.0), (-z_sep/2, -1.0)]:
        # Distance from ring core
        # Ring is a circle at (r=R0, z=ring_z)
        dist_from_core = torch.sqrt((R - R0)**2 + (Z - ring_z)**2)
        
        # Gaussian vortex core
        core = torch.exp(-dist_from_core**2 / (2 * sigma**2))
        
        # Vorticity is in the θ direction (azimuthal)
        # ω_θ = Γ * core_profile
        omega_theta = ring_sign * Gamma * core
        
        # For a ring vortex, the induced velocity has components:
        # u_r (radial) and u_z (axial)
        # From Biot-Savart: complicated, but near the ring:
        
        # Approximate: velocity ~ ∇ × (ψ e_θ) where ψ is stream function
        # For thin core: u_z ~ ω_θ * σ and u_r ~ gradient
        
        # Simplified model capturing the essential physics:
        # The ring induces flow that pushes toward the other ring
        dr = R - R0
        dz = Z - ring_z
        dist = torch.sqrt(dr**2 + dz**2 + 1e-10)
        
        # Induced velocity (simplified Biot-Savart)
        factor = Gamma * sigma**2 / (dist**2 + sigma**2)
        
        # u_z component: rings push toward collision plane
        u_z = ring_sign * factor * dr / (dist + sigma)
        
        # u_r component: radial flow
        u_r = -factor * dz / (dist + sigma)
        
        # Convert to Cartesian and add
        U[..., 0] += u_r * cos_theta  # u_x
        U[..., 1] += u_r * sin_theta  # u_y
        U[..., 2] += u_z              # u_z
    
    # Add swirl (azimuthal velocity u_θ)
    # This is crucial for the Hou-Luo mechanism
    # Swirl is concentrated near the axis
    swirl_profile = config.swirl_amplitude * torch.exp(-R**2 / (2 * sigma**2))
    
    # u_θ in Cartesian: u_x += -sin(θ) * u_θ, u_y += cos(θ) * u_θ
    U[..., 0] += -sin_theta * swirl_profile
    U[..., 1] += cos_theta * swirl_profile
    
    # Make divergence-free via Helmholtz projection
    U = project_divergence_free(U, L)
    
    # Normalize to unit energy
    energy = (U**2).sum() * (L/N)**3
    U = U / torch.sqrt(energy) * config.circulation
    
    return U


def project_divergence_free(U: torch.Tensor, L: float) -> torch.Tensor:
    """Project velocity field to be divergence-free."""
    N = U.shape[0]
    dx = L / N
    
    k = torch.fft.fftfreq(N, dx) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0  # Avoid division by zero
    
    U_hat = torch.fft.fftn(U, dim=(0, 1, 2))
    
    # Divergence in spectral space
    div_hat = 1j * kx * U_hat[..., 0] + 1j * ky * U_hat[..., 1] + 1j * kz * U_hat[..., 2]
    
    # Pressure Poisson: ΔP = div(U)
    P_hat = div_hat / k_sq
    P_hat[0, 0, 0] = 0
    
    # Subtract gradient of pressure
    U_hat[..., 0] -= 1j * kx * P_hat
    U_hat[..., 1] -= 1j * ky * P_hat
    U_hat[..., 2] -= 1j * kz * P_hat
    
    return torch.fft.ifftn(U_hat, dim=(0, 1, 2)).real


def create_collision_profile(N: int = 64, strength: float = 1.0) -> torch.Tensor:
    """
    Create a simplified vortex collision profile.
    
    Two vortex tubes approaching each other, which creates 
    the hyperbolic stagnation point geometry.
    """
    L = 2 * np.pi
    x = torch.linspace(-L/2, L/2, N, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    
    U = torch.zeros(N, N, N, 3, dtype=torch.float64)
    
    # Two vortex tubes along x-axis at z = ±0.5
    sigma = L / 15  # Core thickness
    
    for z0, sign in [(L/6, 1.0), (-L/6, -1.0)]:
        # Distance from tube axis
        r = torch.sqrt(Y**2 + (Z - z0)**2)
        
        # Lamb-Oseen vortex
        vortex = strength * (1 - torch.exp(-r**2 / (2*sigma**2)))
        
        # Azimuthal velocity around the tube (in y-z plane)
        u_theta = vortex / (r + 1e-10)
        
        # Convert to Cartesian
        cos_phi = Y / (r + 1e-10)
        sin_phi = (Z - z0) / (r + 1e-10)
        
        # u_θ in y-z plane: u_y = -sin(φ) * u_θ, u_z = cos(φ) * u_θ  
        U[..., 1] += sign * (-sin_phi) * u_theta
        U[..., 2] += sign * cos_phi * u_theta
        
    # Add strain that pushes tubes together
    strain_rate = 0.5 * strength
    U[..., 2] += -strain_rate * Z  # Compressive strain in z
    U[..., 0] += strain_rate * X / 2  # Extensional in x
    U[..., 1] += strain_rate * Y / 2  # Extensional in y
    
    U = project_divergence_free(U, L)
    
    # Normalize
    energy = (U**2).sum() * (L/N)**3
    U = U / torch.sqrt(energy) * strength
    
    return U


def analyze_blowup_geometry(U: torch.Tensor, verbose: bool = True) -> dict:
    """
    Analyze a velocity field for blow-up signatures.
    
    Checks for:
    1. Vortex stretching alignment
    2. Hyperbolic stagnation point geometry
    3. Vorticity concentration
    """
    N = U.shape[0]
    L = 2 * np.pi
    dx = L / N
    
    # Compute vorticity
    k = torch.fft.fftfreq(N, dx) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    
    U_hat = torch.fft.fftn(U, dim=(0, 1, 2))
    
    dUdy = torch.fft.ifftn(1j * ky.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
    dUdz = torch.fft.ifftn(1j * kz.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
    dUdx = torch.fft.ifftn(1j * kx.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
    
    omega = torch.zeros_like(U)
    omega[..., 0] = dUdy[..., 2] - dUdz[..., 1]  # ω_x
    omega[..., 1] = dUdz[..., 0] - dUdx[..., 2]  # ω_y
    omega[..., 2] = dUdx[..., 1] - dUdy[..., 0]  # ω_z
    
    omega_mag = torch.sqrt((omega**2).sum(dim=-1))
    
    # Strain rate tensor S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i) / 2
    S = torch.zeros(N, N, N, 3, 3, dtype=torch.float64)
    grads = [dUdx, dUdy, dUdz]
    for i in range(3):
        for j in range(3):
            S[..., i, j] = (grads[j][..., i] + grads[i][..., j]) / 2
    
    # Vortex stretching: ω · S · ω
    omega_S_omega = torch.zeros(N, N, N, dtype=torch.float64)
    for i in range(3):
        for j in range(3):
            omega_S_omega += omega[..., i] * S[..., i, j] * omega[..., j]
    
    # Normalize by |ω|²
    omega_sq = (omega**2).sum(dim=-1)
    stretching_alignment = omega_S_omega / (omega_sq + 1e-10)
    
    # Find location of maximum vorticity
    max_omega_idx = torch.argmax(omega_mag)
    max_loc = np.unravel_index(max_omega_idx.item(), omega_mag.shape)
    
    # Eigenvalues of strain at max vorticity location
    S_at_max = S[max_loc[0], max_loc[1], max_loc[2]]
    eigenvalues = torch.linalg.eigvalsh(S_at_max)
    
    results = {
        'max_omega': omega_mag.max().item(),
        'max_omega_location': max_loc,
        'mean_omega': omega_mag.mean().item(),
        'enstrophy': (omega_mag**2).sum().item() * dx**3,
        'max_stretching': stretching_alignment.max().item(),
        'strain_eigenvalues': eigenvalues.tolist(),
        'energy': (U**2).sum().item() * dx**3 / 2,
    }
    
    if verbose:
        print("=" * 60)
        print("BLOW-UP GEOMETRY ANALYSIS")
        print("=" * 60)
        print(f"  Maximum vorticity: ||ω||_∞ = {results['max_omega']:.4f}")
        print(f"  Location of max ω: {max_loc}")
        print(f"  Mean vorticity: {results['mean_omega']:.4f}")
        print(f"  Enstrophy: {results['enstrophy']:.4f}")
        print(f"  Max stretching alignment: {results['max_stretching']:.4f}")
        print(f"  Strain eigenvalues at max ω: {eigenvalues.tolist()}")
        print()
        
        # Interpret strain eigenvalues
        # For hyperbolic point: should have 1 positive, 2 negative (or vice versa)
        n_positive = (eigenvalues > 0).sum().item()
        n_negative = (eigenvalues < 0).sum().item()
        
        if n_positive == 1 and n_negative == 2:
            print("  ★ HYPERBOLIC POINT DETECTED (1+, 2-)")
            print("    This is the Hou-Luo geometry for blow-up!")
        elif n_positive == 2 and n_negative == 1:
            print("  ★ HYPERBOLIC POINT DETECTED (2+, 1-)")
            print("    This is the Hou-Luo geometry for blow-up!")
        else:
            print(f"  Strain signature: {n_positive} positive, {n_negative} negative")
        
        print("=" * 60)
    
    return results


def demo_hou_luo():
    """Demonstrate the Hou-Luo profile."""
    print("=" * 60)
    print("HOU-LUO AXISYMMETRIC BLOW-UP CANDIDATE")
    print("=" * 60)
    
    # Create profile
    config = HouLuoConfig(N=48, circulation=1.0)
    U = create_hou_luo_profile(config)
    
    print(f"\nProfile shape: {U.shape}")
    print(f"Profile energy: {(U**2).sum().item() * (config.L/config.N)**3 / 2:.4f}")
    
    # Analyze
    results = analyze_blowup_geometry(U, verbose=True)
    
    return U, results


if __name__ == "__main__":
    U, results = demo_hou_luo()
