"""
GPU Fluid Dynamics Kernels
===========================

OPERATION VALHALLA - Phase 2.3: Fluid Dynamics

Navier-Stokes solver optimized for RTX 5070 Tensor Cores.
All operations use PyTorch for GPU acceleration.

Equations:
    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
    ∇·u = 0  (incompressibility)

Author: OPERATION VALHALLA
Date: 2025-12-28
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from .tensor_field import TensorField


class FluidDynamicsSolver:
    """
    GPU-accelerated incompressible Navier-Stokes solver.
    
    Uses semi-Lagrangian advection and pressure projection.
    Optimized for real-time atmospheric simulation.
    """
    
    def __init__(
        self,
        shape: Tuple[int, int, int],
        viscosity: float = 1e-5,
        dt: float = 0.01,
        device: str = 'cuda:0'
    ):
        """
        Initialize fluid solver.
        
        Args:
            shape: Grid dimensions (X, Y, Z)
            viscosity: Kinematic viscosity (m²/s)
            dt: Time step (s)
            device: CUDA device
        """
        self.shape = shape
        self.viscosity = viscosity
        self.dt = dt
        self.device = device
        
        # Velocity fields (3 components)
        self.u = torch.zeros(shape, device=device)  # X velocity
        self.v = torch.zeros(shape, device=device)  # Y velocity
        self.w = torch.zeros(shape, device=device)  # Z velocity
        
        # Pressure field
        self.pressure = torch.zeros(shape, device=device)
        
        # Grid spacing (assume uniform)
        self.dx = 1.0 / max(shape)
        
        print(f"✓ FluidDynamicsSolver initialized: {shape}")
        
    def advect(self, field: torch.Tensor, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Semi-Lagrangian advection (backward tracing).
        
        Args:
            field: Scalar or vector field to advect
            u, v, w: Velocity components
            
        Returns:
            Advected field
        """
        # Create coordinate grid
        X, Y, Z = self.shape
        x = torch.arange(X, device=self.device, dtype=torch.float32)
        y = torch.arange(Y, device=self.device, dtype=torch.float32)
        z = torch.arange(Z, device=self.device, dtype=torch.float32)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Backward trace positions
        pos_x = grid_x - u * self.dt / self.dx
        pos_y = grid_y - v * self.dt / self.dx
        pos_z = grid_z - w * self.dt / self.dx
        
        # Clamp to valid range
        pos_x = torch.clamp(pos_x, 0, X - 1)
        pos_y = torch.clamp(pos_y, 0, Y - 1)
        pos_z = torch.clamp(pos_z, 0, Z - 1)
        
        # Normalize to [-1, 1] for grid_sample
        pos_x = 2.0 * pos_x / (X - 1) - 1.0
        pos_y = 2.0 * pos_y / (Y - 1) - 1.0
        pos_z = 2.0 * pos_z / (Z - 1) - 1.0
        
        # Stack into grid (batch, depth, height, width, 3)
        grid = torch.stack([pos_z, pos_y, pos_x], dim=-1).unsqueeze(0)
        
        # Interpolate
        field_5d = field.unsqueeze(0).unsqueeze(0)  # (1, 1, X, Y, Z)
        advected = F.grid_sample(
            field_5d,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return advected.squeeze(0).squeeze(0)
    
    def diffuse(self, field: torch.Tensor, iterations: int = 20) -> torch.Tensor:
        """
        Diffusion using Jacobi iteration.
        
        Args:
            field: Field to diffuse
            iterations: Number of Jacobi iterations
            
        Returns:
            Diffused field
        """
        alpha = self.dx * self.dx / (self.viscosity * self.dt)
        beta = 4.0 + alpha
        
        result = field.clone()
        
        for _ in range(iterations):
            # Compute Laplacian using convolution
            laplacian = (
                torch.roll(result, 1, dims=0) + torch.roll(result, -1, dims=0) +
                torch.roll(result, 1, dims=1) + torch.roll(result, -1, dims=1) +
                torch.roll(result, 1, dims=2) + torch.roll(result, -1, dims=2)
            )
            
            result = (field + alpha * laplacian) / beta
        
        return result
    
    def compute_divergence(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity divergence: ∇·u
        
        Uses central differences on GPU.
        """
        du_dx = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2 * self.dx)
        dv_dy = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * self.dx)
        dw_dz = (torch.roll(w, -1, dims=2) - torch.roll(w, 1, dims=2)) / (2 * self.dx)
        
        return du_dx + dv_dy + dw_dz
    
    def project(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, iterations: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pressure projection to enforce incompressibility.
        
        Solves: ∇²p = ∇·u
        Then updates: u = u - ∇p
        """
        # Compute divergence
        div = self.compute_divergence(u, v, w)
        
        # Solve Poisson equation for pressure (Jacobi iteration)
        p = torch.zeros_like(div)
        
        for _ in range(iterations):
            p_neighbors = (
                torch.roll(p, 1, dims=0) + torch.roll(p, -1, dims=0) +
                torch.roll(p, 1, dims=1) + torch.roll(p, -1, dims=1) +
                torch.roll(p, 1, dims=2) + torch.roll(p, -1, dims=2)
            )
            p = (p_neighbors - self.dx * self.dx * div) / 6.0
        
        # Compute pressure gradient
        dp_dx = (torch.roll(p, -1, dims=0) - torch.roll(p, 1, dims=0)) / (2 * self.dx)
        dp_dy = (torch.roll(p, -1, dims=1) - torch.roll(p, 1, dims=1)) / (2 * self.dx)
        dp_dz = (torch.roll(p, -1, dims=2) - torch.roll(p, 1, dims=2)) / (2 * self.dx)
        
        # Subtract gradient to make divergence-free
        u_new = u - dp_dx
        v_new = v - dp_dy
        w_new = w - dp_dz
        
        self.pressure = p
        
        return u_new, v_new, w_new
    
    def add_force(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, 
                  fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add external forces (e.g., buoyancy, wind)."""
        return u + fx * self.dt, v + fy * self.dt, w + fz * self.dt
    
    def step(self, external_force: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        """
        Single time step of Navier-Stokes.
        
        Args:
            external_force: Optional (fx, fy, fz) force fields
        """
        # Add external forces
        if external_force is not None:
            fx, fy, fz = external_force
            self.u, self.v, self.w = self.add_force(self.u, self.v, self.w, fx, fy, fz)
        
        # Diffusion step
        self.u = self.diffuse(self.u, iterations=10)
        self.v = self.diffuse(self.v, iterations=10)
        self.w = self.diffuse(self.w, iterations=10)
        
        # Project to enforce incompressibility
        self.u, self.v, self.w = self.project(self.u, self.v, self.w)
        
        # Advection step (self-advect velocity)
        self.u = self.advect(self.u, self.u, self.v, self.w)
        self.v = self.advect(self.v, self.u, self.v, self.w)
        self.w = self.advect(self.w, self.u, self.v, self.w)
        
        # Project again after advection
        self.u, self.v, self.w = self.project(self.u, self.v, self.w)
    
    def get_vorticity(self) -> torch.Tensor:
        """
        Compute vorticity magnitude: |∇×u|
        
        Useful for visualization of turbulence.
        """
        # Compute curl components
        dwdy = (torch.roll(self.w, -1, dims=1) - torch.roll(self.w, 1, dims=1)) / (2 * self.dx)
        dvdz = (torch.roll(self.v, -1, dims=2) - torch.roll(self.v, 1, dims=2)) / (2 * self.dx)
        omega_x = dwdy - dvdz
        
        dudz = (torch.roll(self.u, -1, dims=2) - torch.roll(self.u, 1, dims=2)) / (2 * self.dx)
        dwdx = (torch.roll(self.w, -1, dims=0) - torch.roll(self.w, 1, dims=0)) / (2 * self.dx)
        omega_y = dudz - dwdx
        
        dvdx = (torch.roll(self.v, -1, dims=0) - torch.roll(self.v, 1, dims=0)) / (2 * self.dx)
        dudy = (torch.roll(self.u, -1, dims=1) - torch.roll(self.u, 1, dims=1)) / (2 * self.dx)
        omega_z = dvdx - dudy
        
        # Magnitude
        vorticity = torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        return vorticity
    
    def get_kinetic_energy(self) -> float:
        """Compute total kinetic energy."""
        ke = 0.5 * (self.u**2 + self.v**2 + self.w**2).sum()
        return float(ke.item())


def benchmark_fluid_dynamics(size: int = 64, steps: int = 10):
    """Benchmark GPU fluid dynamics solver."""
    import time
    
    print(f"\n{'='*60}")
    print("FLUID DYNAMICS BENCHMARK - RTX 5070")
    print(f"{'='*60}\n")
    
    # Initialize solver
    solver = FluidDynamicsSolver(
        shape=(size, size, size),
        viscosity=1e-5,
        dt=0.01
    )
    
    # Add initial vortex
    X, Y, Z = size, size, size
    x = torch.linspace(-1, 1, X, device='cuda')
    y = torch.linspace(-1, 1, Y, device='cuda')
    z = torch.linspace(-1, 1, Z, device='cuda')
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    r = torch.sqrt(grid_x**2 + grid_y**2)
    
    # Vortex velocity field
    solver.u = -grid_y * torch.exp(-r**2)
    solver.v = grid_x * torch.exp(-r**2)
    
    print(f"Initial kinetic energy: {solver.get_kinetic_energy():.2e}")
    
    # Benchmark time steps
    times = []
    for i in range(steps):
        t0 = time.perf_counter()
        solver.step()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        
        if i % 5 == 0:
            ke = solver.get_kinetic_energy()
            vort = solver.get_vorticity().max()
            print(f"Step {i:3d}: {elapsed:6.2f} ms | KE={ke:.2e} | Vort_max={vort:.3f}")
    
    print(f"\nAverage step time: {torch.tensor(times).mean():.2f} ± {torch.tensor(times).std():.2f} ms")
    print(f"Effective FPS: {1000 / torch.tensor(times).mean():.1f}")
    
    print(f"\n{'='*60}")
    vram_used = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM Usage: {vram_used:.3f} GB / 7.96 GB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    benchmark_fluid_dynamics(size=64, steps=10)
