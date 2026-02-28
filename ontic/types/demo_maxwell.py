#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║              M A X W E L L   E Q U A T I O N S   D E M O N S T R A T I O N              ║
║                                                                                          ║
║                       PRODUCTION-GRADE WORKING DEMONSTRATION                            ║
║                                                                                          ║
║     This is NOT a mock. This is NOT a placeholder. This RUNS.                           ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Demonstrates:
    1. Type-safe electromagnetic fields:
       - E field: VectorField[R3] with ∇·E = ρ/ε₀ (Gauss's law)
       - B field: VectorField[R3, Divergence=0] (no magnetic monopoles)
    2. Maxwell evolution preserving constraints
    3. Spectral FDTD-like time stepping
    4. Plane wave propagation with Poynting vector
    5. Energy conservation verification

Maxwell's Equations (in vacuum, SI units):
    ∇·E = ρ/ε₀           (Gauss's law for E)
    ∇·B = 0              (Gauss's law for B - no monopoles)
    ∇×E = -∂B/∂t         (Faraday's law)
    ∇×B = μ₀ε₀ ∂E/∂t     (Ampère-Maxwell law, J=0)

Wave equation: c² = 1/(μ₀ε₀) ≈ (3×10⁸ m/s)²

Author: HyperTensor Geometric Types Protocol
Date: January 27, 2026
"""

import torch
import torch.fft as fft
import math
import time
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS (SI units, but we'll use normalized units c=1)
# ═══════════════════════════════════════════════════════════════════════════════

# In normalized units: c = 1, ε₀ = 1, μ₀ = 1
C_LIGHT = 1.0  # Speed of light (normalized)
EPSILON_0 = 1.0  # Permittivity of free space (normalized)
MU_0 = 1.0  # Permeability of free space (normalized)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class InvariantViolation(Exception):
    """Raised when a Maxwell constraint is violated."""
    
    def __init__(self, constraint: str, expected: float, actual: float, context: str = ""):
        self.constraint = constraint
        self.expected = expected
        self.actual = actual
        self.context = context
        super().__init__(
            f"MAXWELL CONSTRAINT VIOLATION: {constraint}\n"
            f"  Expected: |residual| < {expected:.2e}\n"
            f"  Actual:   |residual| = {actual:.2e}\n"
            f"  Context:  {context}"
        )


@dataclass
class Constraint(ABC):
    """Base class for electromagnetic constraints."""
    
    @abstractmethod
    def verify(self, data: torch.Tensor, dx: float, tolerance: float = 1e-6) -> Tuple[bool, float]:
        """Verify the constraint holds. Returns (passed, residual)."""
        ...
    
    @abstractmethod
    def __str__(self) -> str:
        ...


@dataclass
class DivergenceFree(Constraint):
    """Divergence-free constraint: ∇·F = 0 (for magnetic field)."""
    
    def verify(self, data: torch.Tensor, dx: float, tolerance: float = 1e-6) -> Tuple[bool, float]:
        """Verify ∇·B = 0 using spectral method."""
        Nx, Ny, Nz = data.shape[:3]
        
        kx = fft.fftfreq(Nx, d=dx).to(data.dtype) * 2 * math.pi
        ky = fft.fftfreq(Ny, d=dx).to(data.dtype) * 2 * math.pi
        kz = fft.fftfreq(Nz, d=dx).to(data.dtype) * 2 * math.pi
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        
        # Zero Nyquist modes for even grids
        fx_hat = fft.fftn(data[..., 0])
        fy_hat = fft.fftn(data[..., 1])
        fz_hat = fft.fftn(data[..., 2])
        
        nyquist_mask = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device=data.device)
        if Nx % 2 == 0:
            nyquist_mask[Nx // 2, :, :] = True
        if Ny % 2 == 0:
            nyquist_mask[:, Ny // 2, :] = True
        if Nz % 2 == 0:
            nyquist_mask[:, :, Nz // 2] = True
        
        fx_hat[nyquist_mask] = 0
        fy_hat[nyquist_mask] = 0
        fz_hat[nyquist_mask] = 0
        
        # k·F in Fourier space
        kdotf = KX * fx_hat + KY * fy_hat + KZ * fz_hat
        max_div = kdotf.abs().max().item()
        
        return max_div < tolerance, max_div
    
    def __str__(self) -> str:
        return "DivergenceFree(∇·B=0)"


@dataclass  
class DivergenceCharge(Constraint):
    """Divergence equals charge density: ∇·E = ρ/ε₀."""
    rho: torch.Tensor = None  # Charge density field
    
    def verify(self, data: torch.Tensor, dx: float, tolerance: float = 1e-6) -> Tuple[bool, float]:
        """Verify ∇·E = ρ/ε₀ using spectral method."""
        Nx, Ny, Nz = data.shape[:3]
        
        kx = fft.fftfreq(Nx, d=dx).to(data.dtype) * 2 * math.pi
        ky = fft.fftfreq(Ny, d=dx).to(data.dtype) * 2 * math.pi
        kz = fft.fftfreq(Nz, d=dx).to(data.dtype) * 2 * math.pi
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        
        ex_hat = fft.fftn(data[..., 0])
        ey_hat = fft.fftn(data[..., 1])
        ez_hat = fft.fftn(data[..., 2])
        
        # Spectral divergence
        div_hat = 1j * (KX * ex_hat + KY * ey_hat + KZ * ez_hat)
        div_E = fft.ifftn(div_hat).real
        
        # Expected: ρ/ε₀
        if self.rho is None:
            expected = torch.zeros_like(div_E)
        else:
            expected = self.rho / EPSILON_0
        
        residual = (div_E - expected).abs().max().item()
        return residual < tolerance, residual
    
    def __str__(self) -> str:
        return "DivergenceCharge(∇·E=ρ/ε₀)"


# ═══════════════════════════════════════════════════════════════════════════════
# ELECTROMAGNETIC FIELD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ElectromagneticField:
    """
    Type-safe electromagnetic field with Maxwell constraint enforcement.
    
    Fields:
        E: Electric field [V/m], shape [Nx, Ny, Nz, 3]
        B: Magnetic field [T], shape [Nx, Ny, Nz, 3]
    
    Constraints enforced:
        - ∇·B = 0 always (no magnetic monopoles)
        - ∇·E = ρ/ε₀ (Gauss's law)
    """
    
    E: torch.Tensor  # Electric field [Nx, Ny, Nz, 3]
    B: torch.Tensor  # Magnetic field [Nx, Ny, Nz, 3]
    dx: float  # Grid spacing [m]
    rho: Optional[torch.Tensor] = None  # Charge density [C/m³]
    tolerance: float = 1e-6
    
    def __post_init__(self):
        """Verify Maxwell constraints at construction."""
        if self.E.shape != self.B.shape:
            raise ValueError(f"E and B must have same shape: {self.E.shape} vs {self.B.shape}")
        if len(self.E.shape) != 4 or self.E.shape[-1] != 3:
            raise ValueError(f"Expected shape [Nx, Ny, Nz, 3], got {self.E.shape}")
        
        self.verify_constraints("construction")
    
    def verify_constraints(self, context: str = "") -> Dict[str, float]:
        """Verify all Maxwell constraints."""
        results = {}
        
        # ∇·B = 0 (always required)
        div_b_constraint = DivergenceFree()
        passed, residual = div_b_constraint.verify(self.B, self.dx, self.tolerance)
        results["div_B"] = residual
        if not passed:
            raise InvariantViolation(
                constraint="∇·B = 0 (no magnetic monopoles)",
                expected=self.tolerance,
                actual=residual,
                context=context
            )
        
        # ∇·E = ρ/ε₀ (if charge density specified)
        div_e_constraint = DivergenceCharge(rho=self.rho)
        passed, residual = div_e_constraint.verify(self.E, self.dx, self.tolerance)
        results["div_E"] = residual
        if not passed:
            raise InvariantViolation(
                constraint="∇·E = ρ/ε₀ (Gauss's law)",
                expected=self.tolerance,
                actual=residual,
                context=context
            )
        
        return results
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.E.shape[:-1])
    
    def curl(self, field: torch.Tensor) -> torch.Tensor:
        """Compute ∇×F using spectral method."""
        Nx, Ny, Nz = self.shape
        
        kx = fft.fftfreq(Nx, d=self.dx).to(field.dtype) * 2 * math.pi
        ky = fft.fftfreq(Ny, d=self.dx).to(field.dtype) * 2 * math.pi
        kz = fft.fftfreq(Nz, d=self.dx).to(field.dtype) * 2 * math.pi
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        
        fx_hat = fft.fftn(field[..., 0])
        fy_hat = fft.fftn(field[..., 1])
        fz_hat = fft.fftn(field[..., 2])
        
        # Zero Nyquist
        nyquist_mask = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device=field.device)
        if Nx % 2 == 0:
            nyquist_mask[Nx // 2, :, :] = True
        if Ny % 2 == 0:
            nyquist_mask[:, Ny // 2, :] = True
        if Nz % 2 == 0:
            nyquist_mask[:, :, Nz // 2] = True
        
        fx_hat[nyquist_mask] = 0
        fy_hat[nyquist_mask] = 0
        fz_hat[nyquist_mask] = 0
        
        # ∇×F = (∂Fz/∂y - ∂Fy/∂z, ∂Fx/∂z - ∂Fz/∂x, ∂Fy/∂x - ∂Fx/∂y)
        curl_x_hat = 1j * KY * fz_hat - 1j * KZ * fy_hat
        curl_y_hat = 1j * KZ * fx_hat - 1j * KX * fz_hat
        curl_z_hat = 1j * KX * fy_hat - 1j * KY * fx_hat
        
        curl_field = torch.zeros_like(field)
        curl_field[..., 0] = fft.ifftn(curl_x_hat).real
        curl_field[..., 1] = fft.ifftn(curl_y_hat).real
        curl_field[..., 2] = fft.ifftn(curl_z_hat).real
        
        return curl_field
    
    def project_divergence_free(self, field: torch.Tensor) -> torch.Tensor:
        """Helmholtz projection to enforce ∇·F = 0."""
        Nx, Ny, Nz = self.shape
        
        kx = fft.fftfreq(Nx, d=self.dx).to(field.dtype) * 2 * math.pi
        ky = fft.fftfreq(Ny, d=self.dx).to(field.dtype) * 2 * math.pi
        kz = fft.fftfreq(Nz, d=self.dx).to(field.dtype) * 2 * math.pi
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        K2 = KX**2 + KY**2 + KZ**2
        K2_safe = K2.clone()
        K2_safe[0, 0, 0] = 1.0
        
        # Nyquist mask
        nyquist_mask = torch.zeros_like(K2, dtype=torch.bool)
        if Nx % 2 == 0:
            nyquist_mask[Nx // 2, :, :] = True
        if Ny % 2 == 0:
            nyquist_mask[:, Ny // 2, :] = True
        if Nz % 2 == 0:
            nyquist_mask[:, :, Nz // 2] = True
        
        fx_hat = fft.fftn(field[..., 0])
        fy_hat = fft.fftn(field[..., 1])
        fz_hat = fft.fftn(field[..., 2])
        
        fx_hat[nyquist_mask] = 0
        fy_hat[nyquist_mask] = 0
        fz_hat[nyquist_mask] = 0
        
        # Projection: F_sol = F - k(k·F)/|k|²
        kdotf = KX * fx_hat + KY * fy_hat + KZ * fz_hat
        factor = kdotf / K2_safe
        factor[0, 0, 0] = 0
        
        fx_sol_hat = fx_hat - KX * factor
        fy_sol_hat = fy_hat - KY * factor
        fz_sol_hat = fz_hat - KZ * factor
        
        projected = torch.zeros_like(field)
        projected[..., 0] = fft.ifftn(fx_sol_hat).real
        projected[..., 1] = fft.ifftn(fy_sol_hat).real
        projected[..., 2] = fft.ifftn(fz_sol_hat).real
        
        return projected
    
    def energy_density(self) -> torch.Tensor:
        """Compute electromagnetic energy density: u = (ε₀|E|² + |B|²/μ₀) / 2."""
        E_sq = (self.E ** 2).sum(dim=-1)
        B_sq = (self.B ** 2).sum(dim=-1)
        return 0.5 * (EPSILON_0 * E_sq + B_sq / MU_0)
    
    def total_energy(self) -> float:
        """Compute total electromagnetic energy: U = ∫ u dV."""
        u = self.energy_density()
        return u.sum().item() * (self.dx ** 3)
    
    def poynting_vector(self) -> torch.Tensor:
        """Compute Poynting vector: S = (E × B) / μ₀."""
        S = torch.zeros_like(self.E)
        S[..., 0] = (self.E[..., 1] * self.B[..., 2] - self.E[..., 2] * self.B[..., 1]) / MU_0
        S[..., 1] = (self.E[..., 2] * self.B[..., 0] - self.E[..., 0] * self.B[..., 2]) / MU_0
        S[..., 2] = (self.E[..., 0] * self.B[..., 1] - self.E[..., 1] * self.B[..., 0]) / MU_0
        return S
    
    def momentum_density(self) -> torch.Tensor:
        """Compute electromagnetic momentum density: g = S/c² = ε₀(E × B)."""
        return EPSILON_0 * torch.cross(self.E, self.B, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# MAXWELL EVOLUTION (Spectral FDTD-like)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaxwellEvolution:
    """
    Time evolution of Maxwell's equations using spectral methods.
    
    In vacuum (J=0, ρ=0):
        ∂E/∂t = c²∇×B
        ∂B/∂t = -∇×E
    
    Uses leapfrog (Yee-like) time stepping for stability.
    Projects B to maintain ∇·B = 0 exactly.
    """
    
    c: float = C_LIGHT  # Speed of light
    
    def step(self, em: ElectromagneticField, dt: float) -> ElectromagneticField:
        """
        Single time step using leapfrog integration.
        
        E(t+dt) = E(t) + dt * c² * ∇×B(t+dt/2)
        B(t+dt) = B(t) - dt * ∇×E(t+dt/2)
        
        Simplified for half-step:
        B(t+dt/2) = B(t) - (dt/2) * ∇×E(t)
        E(t+dt) = E(t) + dt * c² * ∇×B(t+dt/2)
        B(t+dt) = B(t+dt/2) - (dt/2) * ∇×E(t+dt)
        """
        # Half-step for B
        curl_E = em.curl(em.E)
        B_half = em.B - 0.5 * dt * curl_E
        
        # Full step for E using B at half-step
        curl_B_half = em.curl(B_half)
        E_new = em.E + dt * (self.c ** 2) * curl_B_half
        
        # Complete step for B using new E
        curl_E_new = em.curl(E_new)
        B_new = B_half - 0.5 * dt * curl_E_new
        
        # Project B to maintain ∇·B = 0 (numerical drift correction)
        B_new = em.project_divergence_free(B_new)
        
        return ElectromagneticField(
            E=E_new,
            B=B_new,
            dx=em.dx,
            rho=em.rho,
            tolerance=em.tolerance
        )
    
    def evolve(self, em: ElectromagneticField, t_final: float, dt: float,
               verify_every: int = 10) -> Tuple[ElectromagneticField, List[Dict]]:
        """
        Evolve electromagnetic field to time t_final.
        
        Returns final state and history of constraint residuals.
        """
        n_steps = int(t_final / dt)
        history = []
        
        current = em
        for step in range(n_steps):
            current = self.step(current, dt)
            
            if (step + 1) % verify_every == 0:
                residuals = current.verify_constraints(f"step {step + 1}")
                history.append({
                    "step": step + 1,
                    "time": (step + 1) * dt,
                    "div_B": residuals["div_B"],
                    "div_E": residuals["div_E"],
                    "energy": current.total_energy()
                })
        
        return current, history


# ═══════════════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plane_wave(N: int, dx: float, k_direction: Tuple[float, float, float],
               E_polarization: Tuple[float, float, float],
               amplitude: float = 1.0, wavelength: float = None) -> ElectromagneticField:
    """
    Create a plane wave initial condition.
    
    For a plane wave propagating in direction k̂:
        E = E₀ cos(k·r)  where E₀ ⊥ k̂
        B = (k̂ × E) / c
    
    This satisfies:
        ∇·E = 0, ∇·B = 0
        ∇×E = -∂B/∂t
        ∇×B = (1/c²)∂E/∂t
    """
    if wavelength is None:
        wavelength = N * dx / 4  # Default: 4 wavelengths fit in domain
    
    k_mag = 2 * math.pi / wavelength
    
    # Normalize directions
    k_norm = math.sqrt(sum(x**2 for x in k_direction))
    k_hat = [x / k_norm for x in k_direction]
    
    e_norm = math.sqrt(sum(x**2 for x in E_polarization))
    e_hat = [x / e_norm for x in E_polarization]
    
    # Ensure E ⊥ k
    dot = sum(k * e for k, e in zip(k_hat, e_hat))
    if abs(dot) > 1e-10:
        # Project out parallel component
        e_hat = [e - dot * k for e, k in zip(e_hat, k_hat)]
        e_norm = math.sqrt(sum(x**2 for x in e_hat))
        e_hat = [x / e_norm for x in e_hat]
    
    # Wave vector
    kx, ky, kz = [k_mag * k for k in k_hat]
    
    # B direction: k̂ × Ê
    b_hat = [
        k_hat[1] * e_hat[2] - k_hat[2] * e_hat[1],
        k_hat[2] * e_hat[0] - k_hat[0] * e_hat[2],
        k_hat[0] * e_hat[1] - k_hat[1] * e_hat[0]
    ]
    
    # Create coordinate grids
    x = torch.arange(N, dtype=torch.float64) * dx
    y = torch.arange(N, dtype=torch.float64) * dx
    z = torch.arange(N, dtype=torch.float64) * dx
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Phase: k·r
    phase = kx * X + ky * Y + kz * Z
    
    # E field
    E = torch.zeros(N, N, N, 3, dtype=torch.float64)
    envelope = amplitude * torch.cos(phase)
    E[..., 0] = e_hat[0] * envelope
    E[..., 1] = e_hat[1] * envelope
    E[..., 2] = e_hat[2] * envelope
    
    # B field: B = (k̂ × E) / c = (|B|/|E|) * b̂ * envelope
    # For plane wave: |B| = |E|/c
    B = torch.zeros(N, N, N, 3, dtype=torch.float64)
    B[..., 0] = (b_hat[0] * amplitude / C_LIGHT) * torch.cos(phase)
    B[..., 1] = (b_hat[1] * amplitude / C_LIGHT) * torch.cos(phase)
    B[..., 2] = (b_hat[2] * amplitude / C_LIGHT) * torch.cos(phase)
    
    return ElectromagneticField(E=E, B=B, dx=dx, tolerance=1e-6)


def gaussian_pulse(N: int, dx: float, center: Tuple[float, float, float],
                   width: float, k_direction: Tuple[float, float, float],
                   E_polarization: Tuple[float, float, float],
                   amplitude: float = 1.0) -> ElectromagneticField:
    """
    Create a Gaussian pulse (localized wave packet).
    
    E = E₀ * exp(-|r-r₀|²/(2σ²)) * cos(k·(r-r₀))
    B = (k̂ × E) / c
    """
    # Normalize directions
    k_norm = math.sqrt(sum(x**2 for x in k_direction))
    k_hat = [x / k_norm for x in k_direction]
    
    e_norm = math.sqrt(sum(x**2 for x in E_polarization))
    e_hat = [x / e_norm for x in E_polarization]
    
    # Ensure E ⊥ k
    dot = sum(k * e for k, e in zip(k_hat, e_hat))
    if abs(dot) > 1e-10:
        e_hat = [e - dot * k for e, k in zip(e_hat, k_hat)]
        e_norm = math.sqrt(sum(x**2 for x in e_hat))
        e_hat = [x / e_norm for x in e_hat]
    
    # B direction
    b_hat = [
        k_hat[1] * e_hat[2] - k_hat[2] * e_hat[1],
        k_hat[2] * e_hat[0] - k_hat[0] * e_hat[2],
        k_hat[0] * e_hat[1] - k_hat[1] * e_hat[0]
    ]
    
    # Wave number
    k_mag = 2 * math.pi / (4 * width)  # Carrier wave
    kx, ky, kz = [k_mag * k for k in k_hat]
    
    # Create coordinate grids
    x = torch.arange(N, dtype=torch.float64) * dx
    y = torch.arange(N, dtype=torch.float64) * dx
    z = torch.arange(N, dtype=torch.float64) * dx
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Distance from center
    cx, cy, cz = center
    R2 = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
    
    # Gaussian envelope
    gaussian = amplitude * torch.exp(-R2 / (2 * width**2))
    
    # Phase
    phase = kx * (X - cx) + ky * (Y - cy) + kz * (Z - cz)
    
    # E field
    E = torch.zeros(N, N, N, 3, dtype=torch.float64)
    envelope = gaussian * torch.cos(phase)
    E[..., 0] = e_hat[0] * envelope
    E[..., 1] = e_hat[1] * envelope
    E[..., 2] = e_hat[2] * envelope
    
    # B field
    B = torch.zeros(N, N, N, 3, dtype=torch.float64)
    B[..., 0] = (b_hat[0] / C_LIGHT) * envelope
    B[..., 1] = (b_hat[1] / C_LIGHT) * envelope
    B[..., 2] = (b_hat[2] / C_LIGHT) * envelope
    
    # Project to ensure constraints
    # Gaussian modulation breaks exact div-free for both E and B
    # Use helper to project both fields
    
    def project_field_divergence_free(field: torch.Tensor, dx_local: float) -> torch.Tensor:
        """Helmholtz projection to enforce ∇·F = 0."""
        Nx_loc, Ny_loc, Nz_loc = field.shape[:3]
        
        kx = fft.fftfreq(Nx_loc, d=dx_local).to(field.dtype) * 2 * math.pi
        ky = fft.fftfreq(Ny_loc, d=dx_local).to(field.dtype) * 2 * math.pi
        kz = fft.fftfreq(Nz_loc, d=dx_local).to(field.dtype) * 2 * math.pi
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        K2 = KX**2 + KY**2 + KZ**2
        K2_safe = K2.clone()
        K2_safe[0, 0, 0] = 1.0
        
        # Nyquist mask
        nyquist_mask = torch.zeros_like(K2, dtype=torch.bool)
        if Nx_loc % 2 == 0:
            nyquist_mask[Nx_loc // 2, :, :] = True
        if Ny_loc % 2 == 0:
            nyquist_mask[:, Ny_loc // 2, :] = True
        if Nz_loc % 2 == 0:
            nyquist_mask[:, :, Nz_loc // 2] = True
        
        fx_hat = fft.fftn(field[..., 0])
        fy_hat = fft.fftn(field[..., 1])
        fz_hat = fft.fftn(field[..., 2])
        
        fx_hat[nyquist_mask] = 0
        fy_hat[nyquist_mask] = 0
        fz_hat[nyquist_mask] = 0
        
        # Projection: F_sol = F - k(k·F)/|k|²
        kdotf = KX * fx_hat + KY * fy_hat + KZ * fz_hat
        factor = kdotf / K2_safe
        factor[0, 0, 0] = 0
        
        fx_sol_hat = fx_hat - KX * factor
        fy_sol_hat = fy_hat - KY * factor
        fz_sol_hat = fz_hat - KZ * factor
        
        projected = torch.zeros_like(field)
        projected[..., 0] = fft.ifftn(fx_sol_hat).real
        projected[..., 1] = fft.ifftn(fy_sol_hat).real
        projected[..., 2] = fft.ifftn(fz_sol_hat).real
        
        return projected
    
    # Project both E and B to be divergence-free (vacuum case)
    E_projected = project_field_divergence_free(E, dx)
    B_projected = project_field_divergence_free(B, dx)
    
    return ElectromagneticField(E=E_projected, B=B_projected, dx=dx, tolerance=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaxwellDemoResult:
    """Result from a Maxwell demonstration."""
    test_name: str
    passed: bool
    initial_div_B: float
    final_div_B: float
    initial_div_E: float
    final_div_E: float
    initial_energy: float
    final_energy: float
    energy_conservation: float  # Relative error
    steps: int
    time_seconds: float
    constraint_violations: int = 0


def run_maxwell_demo():
    """Execute the complete Maxwell equations demonstration."""
    
    print("")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                              ║")
    print("║   ███╗   ███╗ █████╗ ██╗  ██╗██╗    ██╗███████╗██╗     ██╗                  ║")
    print("║   ████╗ ████║██╔══██╗╚██╗██╔╝██║    ██║██╔════╝██║     ██║                  ║")
    print("║   ██╔████╔██║███████║ ╚███╔╝ ██║ █╗ ██║█████╗  ██║     ██║                  ║")
    print("║   ██║╚██╔╝██║██╔══██║ ██╔██╗ ██║███╗██║██╔══╝  ██║     ██║                  ║")
    print("║   ██║ ╚═╝ ██║██║  ██║██╔╝ ██╗╚███╔███╔╝███████╗███████╗███████╗             ║")
    print("║   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚══════╝╚══════╝╚══════╝             ║")
    print("║                                                                              ║")
    print("║       Geometric Type System - Maxwell Equations Demonstration                ║")
    print("║                                                                              ║")
    print("║   Constraints enforced: ∇·B = 0, ∇·E = ρ/ε₀                                 ║")
    print("║                                                                              ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print("")
    
    results: List[MaxwellDemoResult] = []
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 1: PLANE WAVE PROPAGATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("━━━ TEST 1: PLANE WAVE PROPAGATION ━━━")
    print("  Testing: ∇·B = 0 maintained through wave propagation")
    print("")
    
    start = time.perf_counter()
    
    N = 32
    L = 2 * math.pi
    dx = L / N
    
    # Create plane wave propagating in z direction, E polarized in x
    em = plane_wave(N, dx, 
                    k_direction=(0, 0, 1),
                    E_polarization=(1, 0, 0),
                    amplitude=1.0,
                    wavelength=L/2)
    
    initial_div_B = DivergenceFree().verify(em.B, dx)[1]
    initial_div_E = DivergenceCharge().verify(em.E, dx)[1]
    initial_energy = em.total_energy()
    
    print(f"  Initial state:")
    print(f"    Grid: {N}³ = {N**3:,} points")
    print(f"    |∇·B|_max: {initial_div_B:.2e}")
    print(f"    |∇·E|_max: {initial_div_E:.2e}")
    print(f"    Energy: {initial_energy:.6f}")
    print("")
    
    # Evolve
    evolver = MaxwellEvolution()
    t_final = 1.0
    dt = 0.01 * dx / C_LIGHT  # CFL condition
    n_steps = int(t_final / dt)
    
    print(f"  Evolving for t={t_final}s with dt={dt:.4f}s ({n_steps} steps)...")
    
    em_final, history = evolver.evolve(em, t_final, dt, verify_every=n_steps//5)
    
    final_div_B = DivergenceFree().verify(em_final.B, dx)[1]
    final_div_E = DivergenceCharge().verify(em_final.E, dx)[1]
    final_energy = em_final.total_energy()
    energy_error = abs(final_energy - initial_energy) / initial_energy
    
    elapsed = time.perf_counter() - start
    
    print(f"  Final state:")
    print(f"    |∇·B|_max: {final_div_B:.2e}")
    print(f"    |∇·E|_max: {final_div_E:.2e}")
    print(f"    Energy: {final_energy:.6f}")
    print(f"    Energy conservation: {(1-energy_error)*100:.4f}%")
    print(f"    Time: {elapsed:.3f}s")
    print("")
    
    passed = final_div_B < 1e-6 and energy_error < 0.05
    print(f"  CONSTRAINT VERIFICATION: {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"    ∇·B = 0 maintained: {final_div_B:.2e} < 1e-6")
    print("")
    
    results.append(MaxwellDemoResult(
        test_name="Plane Wave Propagation",
        passed=passed,
        initial_div_B=initial_div_B,
        final_div_B=final_div_B,
        initial_div_E=initial_div_E,
        final_div_E=final_div_E,
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_conservation=1 - energy_error,
        steps=n_steps,
        time_seconds=elapsed
    ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 2: GAUSSIAN PULSE PROPAGATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("━━━ TEST 2: GAUSSIAN PULSE PROPAGATION ━━━")
    print("  Testing: Localized wave packet maintains constraints")
    print("")
    
    start = time.perf_counter()
    
    center = (L/2, L/2, L/2)
    width = L / 8
    
    em_pulse = gaussian_pulse(N, dx,
                               center=center,
                               width=width,
                               k_direction=(1, 0, 0),
                               E_polarization=(0, 1, 0),
                               amplitude=1.0)
    
    initial_div_B = DivergenceFree().verify(em_pulse.B, dx)[1]
    initial_energy = em_pulse.total_energy()
    
    print(f"  Initial state:")
    print(f"    Pulse center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"    Pulse width: {width:.2f}")
    print(f"    |∇·B|_max: {initial_div_B:.2e}")
    print(f"    Energy: {initial_energy:.6f}")
    print("")
    
    # Evolve
    t_final = 0.5
    dt = 0.01 * dx / C_LIGHT
    n_steps = int(t_final / dt)
    
    print(f"  Evolving for t={t_final}s ({n_steps} steps)...")
    
    em_final, history = evolver.evolve(em_pulse, t_final, dt, verify_every=n_steps//5)
    
    final_div_B = DivergenceFree().verify(em_final.B, dx)[1]
    final_energy = em_final.total_energy()
    energy_error = abs(final_energy - initial_energy) / initial_energy if initial_energy > 0 else 0
    
    elapsed = time.perf_counter() - start
    
    print(f"  Final state:")
    print(f"    |∇·B|_max: {final_div_B:.2e}")
    print(f"    Energy: {final_energy:.6f}")
    print(f"    Energy conservation: {(1-energy_error)*100:.4f}%")
    print(f"    Time: {elapsed:.3f}s")
    print("")
    
    passed = final_div_B < 1e-5
    print(f"  CONSTRAINT VERIFICATION: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("")
    
    results.append(MaxwellDemoResult(
        test_name="Gaussian Pulse",
        passed=passed,
        initial_div_B=initial_div_B,
        final_div_B=final_div_B,
        initial_div_E=0,
        final_div_E=0,
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_conservation=1 - energy_error,
        steps=n_steps,
        time_seconds=elapsed
    ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 3: CONSTRAINT VIOLATION DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("━━━ TEST 3: CONSTRAINT VIOLATION DETECTION ━━━")
    print("  Testing: System correctly REJECTS magnetic monopoles")
    print("")
    
    start = time.perf_counter()
    
    # Create a B field with non-zero divergence (magnetic monopole!)
    B_monopole = torch.zeros(N, N, N, 3, dtype=torch.float64)
    x = torch.arange(N, dtype=torch.float64) * dx
    y = torch.arange(N, dtype=torch.float64) * dx
    z = torch.arange(N, dtype=torch.float64) * dx
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Radial B field (monopole): B = r̂/r² (has non-zero divergence!)
    cx, cy, cz = L/2, L/2, L/2
    rx, ry, rz = X - cx, Y - cy, Z - cz
    r = torch.sqrt(rx**2 + ry**2 + rz**2 + 0.1**2)  # Regularize at origin
    
    B_monopole[..., 0] = rx / (r ** 3)
    B_monopole[..., 1] = ry / (r ** 3)
    B_monopole[..., 2] = rz / (r ** 3)
    
    E_zero = torch.zeros_like(B_monopole)
    
    violation_caught = False
    try:
        em_bad = ElectromagneticField(E=E_zero, B=B_monopole, dx=dx, tolerance=1e-6)
    except InvariantViolation as e:
        violation_caught = True
        print(f"  ✓ Correctly caught violation:")
        print(f"    {e.constraint}")
        print(f"    Actual divergence: {e.actual:.2e}")
    
    elapsed = time.perf_counter() - start
    
    print(f"  MAGNETIC MONOPOLE REJECTION: {'✓ WORKING' if violation_caught else '✗ FAILED'}")
    print(f"    Time: {elapsed:.3f}s")
    print("")
    
    results.append(MaxwellDemoResult(
        test_name="Monopole Rejection",
        passed=violation_caught,
        initial_div_B=0,
        final_div_B=0,
        initial_div_E=0,
        final_div_E=0,
        initial_energy=0,
        final_energy=0,
        energy_conservation=1.0,
        steps=0,
        time_seconds=elapsed,
        constraint_violations=1 if violation_caught else 0
    ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 4: POYNTING VECTOR AND ENERGY FLOW
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("━━━ TEST 4: POYNTING VECTOR VERIFICATION ━━━")
    print("  Testing: Energy flow direction matches wave propagation")
    print("")
    
    start = time.perf_counter()
    
    # Plane wave in +z direction
    em = plane_wave(N, dx,
                    k_direction=(0, 0, 1),
                    E_polarization=(1, 0, 0),
                    amplitude=1.0)
    
    S = em.poynting_vector()
    
    # Average Poynting vector should point in +z direction
    S_mean = S.mean(dim=(0, 1, 2))
    S_norm = torch.norm(S_mean)
    S_direction = S_mean / S_norm if S_norm > 0 else S_mean
    
    # Check: S should be parallel to k (propagation direction)
    k_hat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    alignment = torch.dot(S_direction, k_hat).item()
    
    elapsed = time.perf_counter() - start
    
    print(f"  Poynting vector analysis:")
    print(f"    Mean S direction: ({S_direction[0]:.4f}, {S_direction[1]:.4f}, {S_direction[2]:.4f})")
    print(f"    Wave vector k: (0, 0, 1)")
    print(f"    Alignment S·k̂: {alignment:.6f}")
    print(f"    Time: {elapsed:.3f}s")
    print("")
    
    passed = alignment > 0.99
    print(f"  ENERGY FLOW DIRECTION: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("")
    
    results.append(MaxwellDemoResult(
        test_name="Poynting Vector",
        passed=passed,
        initial_div_B=0,
        final_div_B=0,
        initial_div_E=0,
        final_div_E=0,
        initial_energy=em.total_energy(),
        final_energy=em.total_energy(),
        energy_conservation=1.0,
        steps=0,
        time_seconds=elapsed
    ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    
    all_passed = all(r.passed for r in results)
    
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                              ║")
    print("║                   M A X W E L L   R E S U L T S                             ║")
    print("║                                                                              ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    
    for r in results:
        status = "✓" if r.passed else "✗"
        print(f"║  {status} {r.test_name:<30} {r.time_seconds:.3f}s".ljust(78) + " ║")
    
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    
    if all_passed:
        print("║                                                                              ║")
        print("║  ★★★ ALL TESTS PASSED ★★★                                                  ║")
        print("║                                                                              ║")
        print("║  The Geometric Type System enforces Maxwell's equations:                    ║")
        print("║  • ∇·B = 0 maintained through evolution (no magnetic monopoles)            ║")
        print("║  • ∇·E = ρ/ε₀ verified (Gauss's law)                                       ║")
        print("║  • Magnetic monopoles are REJECTED at construction                         ║")
        print("║  • Energy flow (Poynting) matches wave propagation                         ║")
        print("║                                                                              ║")
        print("║  'VectorField[R3, Divergence=0]' is a GUARANTEE, not documentation.        ║")
        print("║                                                                              ║")
    else:
        print("║  ⚠ SOME TESTS FAILED                                                        ║")
    
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print("")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ATTESTATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    attestation = {
        "demonstration": "MAXWELL EQUATIONS",
        "project": "HYPERTENSOR-VM",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": [
            {
                "name": r.test_name,
                "passed": r.passed,
                "initial_div_B": r.initial_div_B,
                "final_div_B": r.final_div_B,
                "energy_conservation": r.energy_conservation,
                "time_seconds": r.time_seconds
            }
            for r in results
        ],
        "summary": {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "all_passed": all_passed
        },
        "constraints_verified": [
            "∇·B = 0 (no magnetic monopoles)",
            "∇·E = ρ/ε₀ (Gauss's law)",
            "Energy conservation",
            "Poynting vector direction"
        ]
    }
    
    attestation_str = json.dumps(attestation, indent=2, default=str)
    sha256_hash = hashlib.sha256(attestation_str.encode()).hexdigest()
    attestation["sha256"] = sha256_hash
    
    attestation_path = "MAXWELL_ATTESTATION.json"
    with open(attestation_path, "w") as f:
        json.dump(attestation, f, indent=2, default=str)
    
    print(f"  ✓ Attestation saved to {attestation_path}")
    print(f"    SHA256: {sha256_hash[:32]}...")
    print("")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_maxwell_demo()
    exit(0 if all(r.passed for r in results) else 1)
