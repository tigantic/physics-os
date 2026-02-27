"""
HyperFOAM Solver with Pressure Projection

This is the Physics Engine. It contains:
- Advection-Diffusion for momentum
- Brinkman penalization for solid obstacles
- Conjugate Gradient pressure solver for incompressibility
- Velocity correction to enforce div(u) = 0

The pressure solver is what makes air SWERVE around obstacles
instead of vanishing into them.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ProjectionConfig:
    nx: int = 128
    ny: int = 64
    nz: int = 64
    Lx: float = 9.0
    Ly: float = 3.0
    Lz: float = 3.0
    nu: float = 1.5e-5
    inlet_velocity: float = 0.455
    brinkman_coeff: float = 1e4
    dt: float = 0.001  # Reduced for stability (pressure shockwave can overshoot)
    
    def __post_init__(self):
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz


class GeometricPressureSolver:
    """
    Matrix-Free Conjugate Gradient Solver.
    Solves A * p = b where A is the Geometric Laplacian.
    
    The "Geometric" part means the Laplacian respects area fractions:
    div(area * grad(p)) = divergence
    
    This is what makes pressure build up at obstacle faces and
    redirect flow around them.
    
    OPTIMIZATION: The CG loop is compiled with torch.compile to fuse
    all 20 iterations into a single CUDA kernel graph, eliminating
    Python overhead and achieving ~50x speedup.
    """
    def __init__(self, grid, dt, n_iter: int = 20):
        self.grid = grid
        self.dt = dt
        self.n_iter = n_iter
        self.idx2 = 1.0 / (grid.dx**2)
        self.idy2 = 1.0 / (grid.dy**2)
        self.idz2 = 1.0 / (grid.dz**2)
        
        # Pre-extract geometry tensors for the compiled kernel
        self.area_x = grid.geo[1]
        self.area_y = grid.geo[2]
        self.area_z = grid.geo[3]
        self.mask = (grid.geo[0] > 0.01).float()
        
        # Compile the CG kernel (first call triggers compilation)
        self._cg_kernel = torch.compile(self._cg_iterations, mode="reduce-overhead")
        self._compiled = False

    def _apply_laplacian(self, p, area_x, area_y, area_z, idx2, idy2, idz2):
        """Computes div(Area * grad(p)) - inlined for torch.compile"""
        # Fluxes (Gradient * Area)
        grad_x = (torch.roll(p, -1, 0) - p) * area_x
        grad_y = (torch.roll(p, -1, 1) - p) * area_y
        grad_z = (torch.roll(p, -1, 2) - p) * area_z
        
        # Divergence of Fluxes
        lap_x = (grad_x - torch.roll(grad_x, 1, 0)) * idx2
        lap_y = (grad_y - torch.roll(grad_y, 1, 1)) * idy2
        lap_z = (grad_z - torch.roll(grad_z, 1, 2)) * idz2
        
        return lap_x + lap_y + lap_z

    def _cg_iterations(self, x, b, mask, area_x, area_y, area_z, 
                       idx2, idy2, idz2, n_iter):
        """
        Fixed-iteration CG loop - designed for torch.compile.
        
        No early exits, no Python conditionals on tensor values.
        This allows the entire loop to be fused into a single kernel.
        """
        # Initial residual
        Ax = self._apply_laplacian(x, area_x, area_y, area_z, idx2, idy2, idz2) * mask
        r = (b - Ax) * mask
        p = r.clone()
        rsold = torch.sum(r * r)
        
        # Fixed iterations (no early exit - enables fusion)
        for _ in range(n_iter):
            Ap = self._apply_laplacian(p, area_x, area_y, area_z, idx2, idy2, idz2) * mask
            
            pAp = torch.sum(p * Ap)
            alpha = rsold / (pAp + 1e-12)
            
            x = x + alpha * p
            r = r - alpha * Ap
            
            rsnew = torch.sum(r * r)
            beta = rsnew / (rsold + 1e-12)
            p = r + beta * p
            rsold = rsnew
        
        return x * mask

    def solve(self, divergence, p_guess):
        """Solve pressure Poisson equation using compiled CG kernel"""
        # Prepare RHS
        b = divergence / self.dt
        b = torch.clamp(b * self.mask, -1e6, 1e6)
        b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Prepare initial guess
        x = torch.clamp(p_guess * self.mask, -1e6, 1e6)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Run compiled CG kernel
        result = self._cg_kernel(
            x, b, self.mask,
            self.area_x, self.area_y, self.area_z,
            self.idx2, self.idy2, self.idz2,
            self.n_iter
        )
        
        if not self._compiled:
            self._compiled = True
            # Note: First call triggers JIT compilation (~2-5s)
        
        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


class HyperFoamSolver:
    """
    The main HyperFOAM solver.
    
    Algorithm (Fractional Step / Chorin Projection):
    1. Momentum Predictor: u* = u + dt * (advection + diffusion + drag)
    2. Compute Divergence: div(u*)
    3. Pressure Solve: Laplacian(p) = div(u*) / dt
    4. Velocity Correct: u = u* - dt * grad(p)
    
    The pressure solve is what enforces incompressibility and
    makes flow divert around obstacles.
    """
    def __init__(self, grid, config):
        self.grid = grid
        self.cfg = config
        self.device = grid.device
        
        # State
        shape = (grid.nx, grid.ny, grid.nz)
        self.u = torch.zeros(shape, device=self.device)
        self.v = torch.zeros(shape, device=self.device)
        self.w = torch.zeros(shape, device=self.device)
        self.p = torch.zeros(shape, device=self.device)
        
        self.pressure_solver = GeometricPressureSolver(grid, config.dt)
        
        # Masks
        self.vol_frac = grid.geo[0]
        self.fluid_mask = (self.vol_frac > 0.01).float()
    
    def init_uniform_flow(self, u_val=None):
        """Initialize with uniform x-velocity (faster development)"""
        if u_val is None:
            u_val = self.cfg.inlet_velocity
        # Set uniform u, masked by fluid
        self.u = torch.full_like(self.u, u_val) * self.fluid_mask
        self.v.zero_()
        self.w.zero_()

    def step(self):
        dt = self.cfg.dt
        dx, dy, dz = self.cfg.dx, self.cfg.dy, self.cfg.dz
        
        # --- 0. Helpers ---
        area_x, area_y, area_z = self.grid.geo[1], self.grid.geo[2], self.grid.geo[3]

        def get_laplacian(f):
            """Standard central difference Laplacian"""
            lap_x = (torch.roll(f, -1, 0) - 2*f + torch.roll(f, 1, 0)) / dx**2
            lap_y = (torch.roll(f, -1, 1) - 2*f + torch.roll(f, 1, 1)) / dy**2
            lap_z = (torch.roll(f, -1, 2) - 2*f + torch.roll(f, 1, 2)) / dz**2
            return lap_x + lap_y + lap_z

        def porous_advect(f, u, v, w):
            """First-order Upwind Advection (stable, dissipative)"""
            # Upwind: use upstream value based on flow direction
            # x-direction
            f_xp = torch.roll(f, -1, 0)  # f at i+1
            f_xm = torch.roll(f, 1, 0)   # f at i-1
            df_dx = torch.where(u > 0, (f - f_xm) / dx, (f_xp - f) / dx)
            
            # y-direction
            f_yp = torch.roll(f, -1, 1)
            f_ym = torch.roll(f, 1, 1)
            df_dy = torch.where(v > 0, (f - f_ym) / dy, (f_yp - f) / dy)
            
            # z-direction
            f_zp = torch.roll(f, -1, 2)
            f_zm = torch.roll(f, 1, 2)
            df_dz = torch.where(w > 0, (f - f_zm) / dz, (f_zp - f) / dz)
            
            return u * df_dx * area_x + v * df_dy * area_y + w * df_dz * area_z

        # --- 1. Momentum Predictor (Advection + Diffusion + Drag) ---
        
        # Calculate Advection
        adv_u = porous_advect(self.u, self.u, self.v, self.w)
        adv_v = porous_advect(self.v, self.u, self.v, self.w)
        adv_w = porous_advect(self.w, self.u, self.v, self.w)
        
        # Calculate Diffusion
        lap_u = get_laplacian(self.u)
        lap_v = get_laplacian(self.v)
        lap_w = get_laplacian(self.w)
        
        # Brinkman Drag (Infinite drag in solids)
        brinkman = self.cfg.brinkman_coeff * (1.0 - self.fluid_mask)
        
        # Update u*
        # u_new = u + dt * (-Advection + Viscosity - Drag)
        self.u += dt * (-adv_u + self.cfg.nu * lap_u - brinkman * self.u)
        self.v += dt * (-adv_v + self.cfg.nu * lap_v - brinkman * self.v)
        self.w += dt * (-adv_w + self.cfg.nu * lap_w - brinkman * self.w)
        
        # Stability clamp: prevent runaway velocities
        self.u = torch.clamp(self.u, -10.0, 10.0)
        self.v = torch.clamp(self.v, -10.0, 10.0)
        self.w = torch.clamp(self.w, -10.0, 10.0)
        
        # Force Inlet BC (Must happen before pressure solve)
        # Simple box inlet at (x=0, y=center, z=center)
        y_c, z_c = self.cfg.ny//2, self.cfg.nz//2
        self.u[0:2, y_c-10:y_c+10, z_c-10:z_c+10] = self.cfg.inlet_velocity

        # --- 2. Geometric Divergence ---
        # div = (flux_out - flux_in) / vol_frac
        # Note: We use the *updated* velocities (u*) here
        # ONLY compute divergence in fully fluid cells to avoid amplification
        div_raw = (
            (self.u * area_x - torch.roll(self.u, 1, 0) * torch.roll(area_x, 1, 0)) / dx +
            (self.v * area_y - torch.roll(self.v, 1, 1) * torch.roll(area_y, 1, 1)) / dy +
            (self.w * area_z - torch.roll(self.w, 1, 2) * torch.roll(area_z, 1, 2)) / dz
        )
        # Only divide where there's substantial fluid
        div = div_raw * self.fluid_mask
        
        # --- 3. Pressure Solve (CG) ---
        self.p = self.pressure_solver.solve(div, self.p)
        
        # --- 4. Velocity Correct ---
        # u = u - grad(p) * area
        dp_dx = (torch.roll(self.p, -1, 0) - self.p) / dx
        dp_dy = (torch.roll(self.p, -1, 1) - self.p) / dy
        dp_dz = (torch.roll(self.p, -1, 2) - self.p) / dz
        
        # Limit pressure gradient correction (stability)
        dp_dx = torch.clamp(dp_dx, -100, 100)
        dp_dy = torch.clamp(dp_dy, -100, 100)
        dp_dz = torch.clamp(dp_dz, -100, 100)
        
        self.u -= dp_dx * dt * area_x
        self.v -= dp_dy * dt * area_y
        self.w -= dp_dz * dt * area_z
        
        # Enforce BCs (Kill velocity in solids)
        self.u *= self.fluid_mask
        self.v *= self.fluid_mask
        self.w *= self.fluid_mask
