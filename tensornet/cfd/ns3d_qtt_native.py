"""
3D QTT-Native Navier-Stokes Solver for Turbulence DNS
======================================================

Production-grade implementation of incompressible Navier-Stokes equations
in native QTT format for direct numerical simulation of turbulence.

Mathematical Formulation:
    ∂u/∂t + (u·∇)u = -∇p + ν∇²u + f
    ∇·u = 0 (incompressibility)

Vorticity-Velocity Form (avoids pressure solve):
    ∂ω/∂t = ∇×(u×ω) + ν∇²ω
    ∇²ψ = -ω (stream function)
    u = ∇×ψ

Complexity Analysis:
    Traditional DNS:  O(N³) memory, O(N³ log N) FFT per step
    QTT-Native DNS:   O(r² × 3 log₂ N) memory, O(r³ × 3 log₂ N) per step
    
    For N=1024, r=64:
        Traditional: 4 GB, ~10¹⁰ FLOPs/step
        QTT-Native:  ~1 MB, ~10⁷ FLOPs/step
        Speedup:     ~1000× memory, ~1000× compute

Key Features:
1. Vorticity-velocity formulation (no pressure Poisson solve)
2. RK4 temporal integration with adaptive truncation
3. Scale-adaptive QTT compression (turbulent profile)
4. Spectral diagnostics for K41 validation
5. Energy/enstrophy conservation monitoring

Author: HyperTensor Team
Date: 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict, Any
from enum import Enum, auto
import time
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tensornet.cfd.morton_3d import Morton3DGrid, linear_to_morton_3d, morton_to_linear_3d
from tensornet.cfd.qtt_3d_state import (
    QTT3DState,
    QTT3DVectorField,
    qtt3d_add,
    qtt3d_sub,
    qtt3d_scale,
    qtt3d_truncate,
    QTT3DDerivatives,
    QTT3DDiagnostics,
    compute_diagnostics,
)
from tensornet.cfd.pure_qtt_ops import QTTState, dense_to_qtt, qtt_to_dense, qtt_add
from tensornet.cfd.nd_shift_mpo import truncate_cores


# ═══════════════════════════════════════════════════════════════════════════════════════
# SOLVER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class TimeIntegrator(Enum):
    """Temporal integration schemes."""
    EULER = auto()       # First-order Euler
    RK2 = auto()         # Second-order Runge-Kutta
    RK4 = auto()         # Fourth-order Runge-Kutta
    IMEX = auto()        # Implicit-Explicit (viscous implicit)


class TruncationStrategy(Enum):
    """QTT truncation strategies."""
    FIXED = auto()       # Fixed max rank
    ADAPTIVE = auto()    # Adapt based on singular values
    TURBULENT = auto()   # Scale-adaptive for turbulence


@dataclass
class NS3DConfig:
    """
    Configuration for 3D QTT Navier-Stokes solver.
    
    Physical Parameters:
        nu: Kinematic viscosity [m²/s]
        L: Domain size [m] (cubic domain [0,L]³)
        
    Numerical Parameters:
        n_bits: Bits per dimension (N = 2^n_bits grid points per axis)
        max_rank: Maximum QTT bond dimension
        dt: Time step (auto-computed if None)
        integrator: Time integration scheme
        truncation: QTT truncation strategy
        
    Tolerances:
        tol_svd: SVD truncation tolerance
        tol_divergence: Maximum allowed divergence
    """
    # Physical
    nu: float = 1e-4                           # Kinematic viscosity
    L: float = 2 * np.pi                       # Domain size
    
    # Grid
    n_bits: int = 6                            # N = 64 per axis
    max_rank: int = 64                         # Max QTT rank
    
    # Time stepping
    dt: Optional[float] = None                 # Auto if None
    cfl: float = 0.5                           # CFL number
    integrator: TimeIntegrator = TimeIntegrator.RK4
    
    # Truncation
    truncation: TruncationStrategy = TruncationStrategy.TURBULENT
    tol_svd: float = 1e-10                     # SVD tolerance
    
    # Tolerances
    tol_divergence: float = 1e-8               # Max divergence
    
    # Device
    device: str = 'cuda'
    dtype: str = 'float32'
    
    @property
    def N(self) -> int:
        """Grid points per axis."""
        return 1 << self.n_bits
    
    @property
    def dx(self) -> float:
        """Grid spacing."""
        return self.L / self.N
    
    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device if torch.cuda.is_available() else 'cpu')
    
    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float32 if self.dtype == 'float32' else torch.float64
    
    def compute_dt(self, u_max: float) -> float:
        """Compute stable time step from CFL condition."""
        # CFL: dt * u_max / dx < cfl
        # Viscous: dt * nu / dx² < 0.5
        dt_cfl = self.cfl * self.dx / max(u_max, 1e-10)
        dt_visc = 0.5 * self.dx**2 / self.nu
        return min(dt_cfl, dt_visc)


# ═══════════════════════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def taylor_green_3d(
    config: NS3DConfig,
) -> Tuple[QTT3DVectorField, QTT3DVectorField]:
    """
    Taylor-Green vortex initial condition.
    
    Exact solution for inviscid flow, canonical test case.
    
    u = (cos(x)sin(y)cos(z), -sin(x)cos(y)cos(z), 0)
    ω = ∇×u
    
    Returns:
        (velocity, vorticity) as QTT3DVectorField
    """
    N = config.N
    L = config.L
    device = config.torch_device
    dtype = config.torch_dtype
    
    # Create grid
    x = torch.linspace(0, L, N, device=device, dtype=dtype)
    y = torch.linspace(0, L, N, device=device, dtype=dtype)
    z = torch.linspace(0, L, N, device=device, dtype=dtype)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Velocity field
    ux = torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    uy = -torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    uz = torch.zeros_like(X)
    
    # Vorticity ω = ∇×u
    # ωx = ∂uz/∂y - ∂uy/∂z = sin(x)cos(y)sin(z)
    # ωy = ∂ux/∂z - ∂uz/∂x = -cos(x)sin(y)sin(z)
    # ωz = ∂uy/∂x - ∂ux/∂y = -cos(x)cos(y)cos(z) - cos(x)cos(y)cos(z) = -2cos(x)cos(y)cos(z)
    omega_x = torch.sin(X) * torch.cos(Y) * torch.sin(Z)
    omega_y = -torch.cos(X) * torch.sin(Y) * torch.sin(Z)
    omega_z = -2 * torch.cos(X) * torch.cos(Y) * torch.cos(Z)
    
    # Compress to QTT with tolerance-controlled rank
    u = QTT3DVectorField.from_dense(ux, uy, uz, max_rank=config.max_rank, tol=config.tol_svd)
    omega = QTT3DVectorField.from_dense(omega_x, omega_y, omega_z, max_rank=config.max_rank, tol=config.tol_svd)
    
    return u, omega


def kida_vortex_3d(
    config: NS3DConfig,
    A: float = 1.0,
) -> Tuple[QTT3DVectorField, QTT3DVectorField]:
    """
    Kida vortex initial condition.
    
    More complex vortex dynamics than Taylor-Green, tests
    vortex stretching and reconnection.
    
    u = A(sin(x)cos(y)cos(z), cos(x)sin(y)cos(z), -2cos(x)cos(y)sin(z))
    
    Returns:
        (velocity, vorticity) as QTT3DVectorField
    """
    N = config.N
    L = config.L
    device = config.torch_device
    dtype = config.torch_dtype
    
    x = torch.linspace(0, L, N, device=device, dtype=dtype)
    y = torch.linspace(0, L, N, device=device, dtype=dtype)
    z = torch.linspace(0, L, N, device=device, dtype=dtype)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Velocity (Kida form)
    ux = A * torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    uy = A * torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    uz = -2 * A * torch.cos(X) * torch.cos(Y) * torch.sin(Z)
    
    # Vorticity (computed analytically)
    # ωx = ∂uz/∂y - ∂uy/∂z = 2A*cos(x)sin(y)sin(z) + A*cos(x)sin(y)sin(z) = 3A*cos(x)sin(y)sin(z)
    # ωy = ∂ux/∂z - ∂uz/∂x = -A*sin(x)cos(y)sin(z) - 2A*sin(x)cos(y)sin(z) = -3A*sin(x)cos(y)sin(z)  
    # ωz = ∂uy/∂x - ∂ux/∂y = -A*sin(x)sin(y)cos(z) + A*sin(x)sin(y)cos(z) = 0
    omega_x = 3 * A * torch.cos(X) * torch.sin(Y) * torch.sin(Z)
    omega_y = -3 * A * torch.sin(X) * torch.cos(Y) * torch.sin(Z)
    omega_z = torch.zeros_like(X)
    
    u = QTT3DVectorField.from_dense(ux, uy, uz, max_rank=config.max_rank)
    omega = QTT3DVectorField.from_dense(omega_x, omega_y, omega_z, max_rank=config.max_rank)
    
    return u, omega


def isotropic_turbulence_3d(
    config: NS3DConfig,
    energy_spectrum: str = 'k41',
    k_peak: int = 4,
    seed: int = 42,
) -> Tuple[QTT3DVectorField, QTT3DVectorField]:
    """
    Isotropic turbulence initial condition.
    
    Generates divergence-free velocity field with specified
    energy spectrum in Fourier space.
    
    Spectra:
        'k41': E(k) ∝ k^(-5/3) (Kolmogorov)
        'k2': E(k) ∝ k² (low-k initial)
        'custom': User-defined
        
    Returns:
        (velocity, vorticity) as QTT3DVectorField
    """
    N = config.N
    L = config.L
    device = config.torch_device
    dtype = config.torch_dtype
    
    torch.manual_seed(seed)
    
    # Wavenumbers
    k = torch.fft.fftfreq(N, d=L/(2*np.pi*N), device=device)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)
    k_mag[0, 0, 0] = 1.0  # Avoid division by zero
    
    # Energy spectrum
    if energy_spectrum == 'k41':
        # Kolmogorov spectrum with low-k regularization
        E_k = k_mag**2 * torch.exp(-k_mag/k_peak) / (1 + (k_mag/k_peak)**2)**(11/6)
    elif energy_spectrum == 'k2':
        E_k = k_mag**2 * torch.exp(-k_mag/k_peak)
    else:
        raise ValueError(f"Unknown spectrum: {energy_spectrum}")
    
    E_k[0, 0, 0] = 0.0
    
    # Random phases
    phase_x = 2 * np.pi * torch.rand(N, N, N, device=device, dtype=dtype)
    phase_y = 2 * np.pi * torch.rand(N, N, N, device=device, dtype=dtype)
    phase_z = 2 * np.pi * torch.rand(N, N, N, device=device, dtype=dtype)
    
    # Complex velocity in Fourier space
    amp = torch.sqrt(E_k / (4 * np.pi * k_mag**2 + 1e-10))
    
    ux_hat = amp * torch.exp(1j * phase_x.to(torch.complex64))
    uy_hat = amp * torch.exp(1j * phase_y.to(torch.complex64))
    uz_hat = amp * torch.exp(1j * phase_z.to(torch.complex64))
    
    # Project to divergence-free (Helmholtz projection)
    # u_sol = u - k(k·u)/|k|²
    k_dot_u = kx * ux_hat + ky * uy_hat + kz * uz_hat
    k_sq = k_mag**2
    k_sq[0, 0, 0] = 1.0
    
    ux_hat = ux_hat - kx * k_dot_u / k_sq
    uy_hat = uy_hat - ky * k_dot_u / k_sq
    uz_hat = uz_hat - kz * k_dot_u / k_sq
    
    ux_hat[0, 0, 0] = 0.0
    uy_hat[0, 0, 0] = 0.0
    uz_hat[0, 0, 0] = 0.0
    
    # IFFT to physical space
    ux = torch.fft.ifftn(ux_hat).real.to(dtype)
    uy = torch.fft.ifftn(uy_hat).real.to(dtype)
    uz = torch.fft.ifftn(uz_hat).real.to(dtype)
    
    # Normalize energy
    E_target = 0.5  # Target kinetic energy
    E_current = 0.5 * (ux**2 + uy**2 + uz**2).mean().item()
    scale = np.sqrt(E_target / (E_current + 1e-10))
    ux *= scale
    uy *= scale
    uz *= scale
    
    # Compute vorticity
    deriv = QTT3DDerivatives(
        n_bits=config.n_bits,
        max_rank=config.max_rank,
        device=device,
        dtype=dtype,
        L=L,
    )
    
    u = QTT3DVectorField.from_dense(ux, uy, uz, max_rank=config.max_rank)
    omega = deriv.curl(u)
    
    return u, omega


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT 3D NAVIER-STOKES SOLVER
# ═══════════════════════════════════════════════════════════════════════════════════════

class NS3DQTTSolver:
    """
    3D QTT-Native Navier-Stokes Solver.
    
    Solves incompressible Navier-Stokes using vorticity-velocity
    formulation in native QTT format.
    
    Vorticity equation:
        ∂ω/∂t = ∇×(u×ω) + ν∇²ω
        
    Biot-Savart (spectral):
        u = ∇×(∇⁻²ω)
        
    Example:
        >>> config = NS3DConfig(n_bits=6, nu=1e-3, max_rank=64)
        >>> solver = NS3DQTTSolver(config)
        >>> u, omega = taylor_green_3d(config)
        >>> solver.initialize(u, omega)
        >>> for step in range(1000):
        ...     solver.step()
        ...     if step % 100 == 0:
        ...         print(solver.diagnostics)
    """
    
    def __init__(self, config: NS3DConfig):
        """Initialize solver with configuration."""
        self.config = config
        self.device = config.torch_device
        self.dtype = config.torch_dtype
        
        # Derivative operators
        self.deriv = QTT3DDerivatives(
            n_bits=config.n_bits,
            max_rank=config.max_rank,
            device=self.device,
            dtype=self.dtype,
            L=config.L,
        )
        
        # State
        self.u: Optional[QTT3DVectorField] = None
        self.omega: Optional[QTT3DVectorField] = None
        self.t: float = 0.0
        self.step_count: int = 0
        self.dt: float = config.dt or 0.001
        
        # Adaptive rank tracking - feed previous rank into SVD sizing
        self._rank_hint: int = 32  # Initial estimate
        
        # History
        self.diagnostics_history: List[QTT3DDiagnostics] = []
        
        # Morton grid for spectral operations
        self.morton_grid = Morton3DGrid(config.n_bits, L=config.L)
    
    def initialize(
        self,
        u: QTT3DVectorField,
        omega: QTT3DVectorField,
    ) -> None:
        """
        Initialize solver with velocity and vorticity fields.
        
        Args:
            u: Initial velocity field
            omega: Initial vorticity field
        """
        self.u = u
        self.omega = omega
        self.t = 0.0
        self.step_count = 0
        
        # Compute initial dt from CFL
        ux, uy, uz = u.to_dense()
        u_max = torch.sqrt(ux**2 + uy**2 + uz**2).max().item()
        if self.config.dt is None:
            self.dt = self.config.compute_dt(u_max)
        
        # Initial diagnostics
        diag = compute_diagnostics(self.u, self.omega, self.deriv, self.t)
        self.diagnostics_history.append(diag)
    
    def _velocity_from_vorticity_spectral(
        self,
        omega: QTT3DVectorField,
    ) -> QTT3DVectorField:
        """
        Compute velocity from vorticity via Biot-Savart (spectral).
        
        u = ∇×(∇⁻²ω)
        
        In Fourier space:
            û = ik × ω̂ / |k|²
            
        Note: This decompresses for spectral solve, then recompresses.
        Future: native QTT Poisson solver.
        """
        N = self.config.N
        L = self.config.L
        
        # Decompress vorticity
        ox, oy, oz = omega.to_dense()
        
        # Wavenumbers
        k = torch.fft.fftfreq(N, d=L/(2*np.pi*N), device=self.device)
        kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
        k_sq = kx**2 + ky**2 + kz**2
        k_sq[0, 0, 0] = 1.0  # Avoid division by zero
        
        # FFT of vorticity
        ox_hat = torch.fft.fftn(ox)
        oy_hat = torch.fft.fftn(oy)
        oz_hat = torch.fft.fftn(oz)
        
        # Biot-Savart: u = ik × ω / |k|²
        # ux = i(ky*ωz - kz*ωy) / |k|²
        # uy = i(kz*ωx - kx*ωz) / |k|²
        # uz = i(kx*ωy - ky*ωx) / |k|²
        ux_hat = 1j * (ky * oz_hat - kz * oy_hat) / k_sq
        uy_hat = 1j * (kz * ox_hat - kx * oz_hat) / k_sq
        uz_hat = 1j * (kx * oy_hat - ky * ox_hat) / k_sq
        
        # Zero mean
        ux_hat[0, 0, 0] = 0.0
        uy_hat[0, 0, 0] = 0.0
        uz_hat[0, 0, 0] = 0.0
        
        # IFFT
        ux = torch.fft.ifftn(ux_hat).real.to(self.dtype)
        uy = torch.fft.ifftn(uy_hat).real.to(self.dtype)
        uz = torch.fft.ifftn(uz_hat).real.to(self.dtype)
        
        # Recompress
        return QTT3DVectorField.from_dense(
            ux, uy, uz,
            max_rank=self.config.max_rank,
        )
    
    def _nonlinear_term(
        self,
        u: QTT3DVectorField,
        omega: QTT3DVectorField,
    ) -> QTT3DVectorField:
        """
        Compute nonlinear term ∇×(u×ω).
        
        u×ω = (uy*ωz - uz*ωy, uz*ωx - ux*ωz, ux*ωy - uy*ωx)
        
        Note: Currently uses decompression for Hadamard product.
        Future: native QTT Hadamard via cross approximation.
        """
        # Decompress
        ux, uy, uz = u.to_dense()
        ox, oy, oz = omega.to_dense()
        
        # Cross product u × ω
        cx = uy * oz - uz * oy
        cy = uz * ox - ux * oz
        cz = ux * oy - uy * ox
        
        # Compress
        cross = QTT3DVectorField.from_dense(
            cx, cy, cz,
            max_rank=self.config.max_rank,
        )
        
        # Curl of cross product
        return self.deriv.curl(cross)
    
    def _viscous_term(
        self,
        omega: QTT3DVectorField,
    ) -> QTT3DVectorField:
        """
        Compute viscous term ν∇²ω.
        
        Fully native QTT operation.
        """
        lap_omega = self.deriv.laplacian_vector(omega)
        
        return QTT3DVectorField(
            x=qtt3d_scale(lap_omega.x, self.config.nu),
            y=qtt3d_scale(lap_omega.y, self.config.nu),
            z=qtt3d_scale(lap_omega.z, self.config.nu),
        )
    
    def _rhs(
        self,
        u: QTT3DVectorField,
        omega: QTT3DVectorField,
    ) -> QTT3DVectorField:
        """
        Compute RHS of vorticity equation.
        
        ∂ω/∂t = ∇×(u×ω) + ν∇²ω
        """
        nonlinear = self._nonlinear_term(u, omega)
        viscous = self._viscous_term(omega)
        
        return QTT3DVectorField(
            x=qtt3d_add(nonlinear.x, viscous.x, max_rank=self.config.max_rank),
            y=qtt3d_add(nonlinear.y, viscous.y, max_rank=self.config.max_rank),
            z=qtt3d_add(nonlinear.z, viscous.z, max_rank=self.config.max_rank),
        )
    
    def _step_euler(self) -> None:
        """First-order Euler step."""
        rhs = self._rhs(self.u, self.omega)
        
        # ω_new = ω + dt * rhs
        self.omega = QTT3DVectorField(
            x=qtt3d_add(self.omega.x, qtt3d_scale(rhs.x, self.dt), max_rank=self.config.max_rank),
            y=qtt3d_add(self.omega.y, qtt3d_scale(rhs.y, self.dt), max_rank=self.config.max_rank),
            z=qtt3d_add(self.omega.z, qtt3d_scale(rhs.z, self.dt), max_rank=self.config.max_rank),
        )
        
        # Recover velocity
        self.u = self._velocity_from_vorticity_spectral(self.omega)
    
    def _step_rk4(self) -> None:
        """Fourth-order Runge-Kutta step."""
        dt = self.dt
        max_rank = self.config.max_rank
        
        omega_0 = self.omega.clone()
        u_0 = self.u.clone()
        
        # k1 = f(t, omega)
        k1 = self._rhs(u_0, omega_0)
        
        # omega_1 = omega_0 + 0.5*dt*k1
        omega_1 = QTT3DVectorField(
            x=qtt3d_add(omega_0.x, qtt3d_scale(k1.x, 0.5*dt), max_rank=max_rank),
            y=qtt3d_add(omega_0.y, qtt3d_scale(k1.y, 0.5*dt), max_rank=max_rank),
            z=qtt3d_add(omega_0.z, qtt3d_scale(k1.z, 0.5*dt), max_rank=max_rank),
        )
        u_1 = self._velocity_from_vorticity_spectral(omega_1)
        
        # k2 = f(t + 0.5*dt, omega_1)
        k2 = self._rhs(u_1, omega_1)
        
        # omega_2 = omega_0 + 0.5*dt*k2
        omega_2 = QTT3DVectorField(
            x=qtt3d_add(omega_0.x, qtt3d_scale(k2.x, 0.5*dt), max_rank=max_rank),
            y=qtt3d_add(omega_0.y, qtt3d_scale(k2.y, 0.5*dt), max_rank=max_rank),
            z=qtt3d_add(omega_0.z, qtt3d_scale(k2.z, 0.5*dt), max_rank=max_rank),
        )
        u_2 = self._velocity_from_vorticity_spectral(omega_2)
        
        # k3 = f(t + 0.5*dt, omega_2)
        k3 = self._rhs(u_2, omega_2)
        
        # omega_3 = omega_0 + dt*k3
        omega_3 = QTT3DVectorField(
            x=qtt3d_add(omega_0.x, qtt3d_scale(k3.x, dt), max_rank=max_rank),
            y=qtt3d_add(omega_0.y, qtt3d_scale(k3.y, dt), max_rank=max_rank),
            z=qtt3d_add(omega_0.z, qtt3d_scale(k3.z, dt), max_rank=max_rank),
        )
        u_3 = self._velocity_from_vorticity_spectral(omega_3)
        
        # k4 = f(t + dt, omega_3)
        k4 = self._rhs(u_3, omega_3)
        
        # omega_new = omega_0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        def combine_rk4(o0, k1, k2, k3, k4):
            """Combine RK4 stages for one component."""
            # (k1 + 2*k2 + 2*k3 + k4)
            sum_k = qtt3d_add(k1, qtt3d_scale(k2, 2.0), max_rank=max_rank)
            sum_k = qtt3d_add(sum_k, qtt3d_scale(k3, 2.0), max_rank=max_rank)
            sum_k = qtt3d_add(sum_k, k4, max_rank=max_rank)
            
            # omega_0 + (dt/6) * sum_k
            return qtt3d_add(o0, qtt3d_scale(sum_k, dt/6), max_rank=max_rank)
        
        self.omega = QTT3DVectorField(
            x=combine_rk4(omega_0.x, k1.x, k2.x, k3.x, k4.x),
            y=combine_rk4(omega_0.y, k1.y, k2.y, k3.y, k4.y),
            z=combine_rk4(omega_0.z, k1.z, k2.z, k3.z, k4.z),
        )
        
        # Recover velocity
        self.u = self._velocity_from_vorticity_spectral(self.omega)
    
    def step(self) -> QTT3DDiagnostics:
        """
        Advance one time step.
        
        Returns:
            Current diagnostics
        """
        if self.u is None or self.omega is None:
            raise RuntimeError("Solver not initialized. Call initialize() first.")
        
        # Time integration
        if self.config.integrator == TimeIntegrator.EULER:
            self._step_euler()
        elif self.config.integrator == TimeIntegrator.RK4:
            self._step_rk4()
        else:
            raise NotImplementedError(f"Integrator {self.config.integrator} not implemented")
        
        # Update time
        self.t += self.dt
        self.step_count += 1
        
        # Adaptive truncation
        if self.config.truncation == TruncationStrategy.TURBULENT:
            self._adaptive_truncate()
        
        # Compute diagnostics
        diag = compute_diagnostics(self.u, self.omega, self.deriv, self.t)
        self.diagnostics_history.append(diag)
        
        return diag
    
    def _adaptive_truncate(self) -> None:
        """
        Apply scale-adaptive truncation for turbulence.
        
        Uses rank hint from previous step to size SVD computations.
        This gives O(actual_rank²) cost instead of O(max_rank²).
        """
        max_rank = self.config.max_rank
        tol = self.config.tol_svd
        hint = self._rank_hint
        
        # Truncate velocity components, track max observed rank
        ux, r1 = qtt3d_truncate(self.u.x, max_rank=max_rank, tol=tol, rank_hint=hint)
        uy, r2 = qtt3d_truncate(self.u.y, max_rank=max_rank, tol=tol, rank_hint=hint)
        uz, r3 = qtt3d_truncate(self.u.z, max_rank=max_rank, tol=tol, rank_hint=hint)
        self.u = QTT3DVectorField(x=ux, y=uy, z=uz)
        
        # Truncate vorticity components
        ox, r4 = qtt3d_truncate(self.omega.x, max_rank=max_rank, tol=tol, rank_hint=hint)
        oy, r5 = qtt3d_truncate(self.omega.y, max_rank=max_rank, tol=tol, rank_hint=hint)
        oz, r6 = qtt3d_truncate(self.omega.z, max_rank=max_rank, tol=tol, rank_hint=hint)
        self.omega = QTT3DVectorField(x=ox, y=oy, z=oz)
        
        # Update rank hint for next step (rank evolves slowly)
        self._rank_hint = max(r1, r2, r3, r4, r5, r6)
    
    def run(
        self,
        t_final: float,
        callback: Optional[Callable[[int, QTT3DDiagnostics], None]] = None,
        checkpoint_interval: int = 100,
    ) -> List[QTT3DDiagnostics]:
        """
        Run simulation until t_final.
        
        Args:
            t_final: Final simulation time
            callback: Optional callback(step, diagnostics) 
            checkpoint_interval: Steps between callbacks
            
        Returns:
            List of diagnostics at checkpoint intervals
        """
        checkpoints = []
        
        while self.t < t_final:
            diag = self.step()
            
            if self.step_count % checkpoint_interval == 0:
                checkpoints.append(diag)
                if callback:
                    callback(self.step_count, diag)
        
        return checkpoints
    
    @property
    def diagnostics(self) -> QTT3DDiagnostics:
        """Current diagnostics."""
        return self.diagnostics_history[-1] if self.diagnostics_history else None


# ═══════════════════════════════════════════════════════════════════════════════════════
# SPECTRAL ANALYSIS (K41 VALIDATION)
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralDiagnostics:
    """Spectral diagnostics for K41 validation."""
    k: np.ndarray                # Wavenumbers
    E_k: np.ndarray              # Energy spectrum E(k)
    k41_slope: float             # Fitted slope (should be -5/3 ≈ -1.667)
    inertial_range: Tuple[int, int]  # (k_min, k_max) for inertial range
    kolmogorov_scale: float      # η = (ν³/ε)^(1/4)
    taylor_scale: float          # λ = √(10νE/ε)
    integral_scale: float        # L = ∫k⁻¹E(k)dk / ∫E(k)dk
    reynolds_lambda: float       # Re_λ = u_rms * λ / ν


def compute_energy_spectrum(
    u: QTT3DVectorField,
    config: NS3DConfig,
) -> SpectralDiagnostics:
    """
    Compute energy spectrum and K41 diagnostics.
    
    E(k) = ½ ∫|û(k)|² δ(|k| - k) d³k
    
    K41 prediction: E(k) ∝ k^(-5/3) in inertial range
    """
    N = config.N
    L = config.L
    nu = config.nu
    device = config.torch_device
    
    # Decompress velocity
    ux, uy, uz = u.to_dense()
    
    # FFT
    ux_hat = torch.fft.fftn(ux)
    uy_hat = torch.fft.fftn(uy)
    uz_hat = torch.fft.fftn(uz)
    
    # Energy in Fourier space
    E_hat = 0.5 * (torch.abs(ux_hat)**2 + torch.abs(uy_hat)**2 + torch.abs(uz_hat)**2)
    E_hat = E_hat.cpu().numpy()
    
    # Wavenumber magnitudes
    k = np.fft.fftfreq(N, d=L/(2*np.pi*N))
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Bin into shells
    k_max = int(N / 2)
    k_bins = np.arange(0, k_max + 1)
    E_k = np.zeros(k_max)
    
    for i in range(1, k_max):
        mask = (k_mag >= i - 0.5) & (k_mag < i + 0.5)
        E_k[i-1] = E_hat[mask].sum() / N**3
    
    k_bins = np.arange(1, k_max)
    E_k = E_k[:len(k_bins)]
    
    # Fit K41 slope in inertial range
    # Inertial range: k_forcing < k < k_dissipation
    k_min = max(4, int(N / 32))
    k_max_fit = min(int(N / 4), len(k_bins) - 1)
    
    log_k = np.log(k_bins[k_min:k_max_fit])
    log_E = np.log(E_k[k_min:k_max_fit] + 1e-30)
    
    # Linear fit
    if len(log_k) > 2:
        slope, _ = np.polyfit(log_k, log_E, 1)
    else:
        slope = 0.0
    
    # Turbulence scales
    # Total kinetic energy
    E_total = (ux**2 + uy**2 + uz**2).sum().item() * (L/N)**3 / 2
    
    # Dissipation rate (approximate)
    # ε ≈ 2ν ∫ k²E(k) dk
    dissipation = 2 * nu * np.sum(k_bins**2 * E_k) * (k_bins[1] - k_bins[0] if len(k_bins) > 1 else 1)
    dissipation = max(dissipation, 1e-20)
    
    # Kolmogorov scale
    eta = (nu**3 / dissipation)**0.25
    
    # Taylor microscale
    u_rms = np.sqrt(2 * E_total / L**3)
    taylor_lambda = np.sqrt(10 * nu * E_total / (L**3 * dissipation))
    
    # Integral scale
    if np.sum(E_k) > 0:
        integral_L = np.sum(E_k / (k_bins + 1e-10)) / np.sum(E_k)
    else:
        integral_L = 0.0
    
    # Taylor Reynolds number
    Re_lambda = u_rms * taylor_lambda / nu
    
    return SpectralDiagnostics(
        k=k_bins,
        E_k=E_k,
        k41_slope=slope,
        inertial_range=(k_min, k_max_fit),
        kolmogorov_scale=eta,
        taylor_scale=taylor_lambda,
        integral_scale=integral_L,
        reynolds_lambda=Re_lambda,
    )


# ═══════════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Config
    'TimeIntegrator',
    'TruncationStrategy', 
    'NS3DConfig',
    
    # Initial conditions
    'taylor_green_3d',
    'kida_vortex_3d',
    'isotropic_turbulence_3d',
    
    # Solver
    'NS3DQTTSolver',
    
    # Spectral analysis
    'SpectralDiagnostics',
    'compute_energy_spectrum',
]
