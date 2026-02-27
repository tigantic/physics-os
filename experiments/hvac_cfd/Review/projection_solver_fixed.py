"""
Projection Method Solver — Velocity-Pressure Formulation (FIXED)
=================================================================

FIX APPLIED: Replaced first-order upwind advection with central differences.

Root Cause of Original Failure:
    At Re=5000, upwind numerical viscosity ≈ 8×10⁻³ m²/s
    Physical viscosity ≈ 1.5×10⁻⁵ m²/s
    Ratio: 535× → jet smeared out immediately

Solution:
    Central differences have ~0 numerical viscosity.
    Matches spectral methods used in working HyperTensor solvers.
    
Tradeoff:
    - Upwind: unconditionally stable, highly diffusive
    - Central: requires CFL < 1, non-diffusive (what we need)

Performance Mandates (Source_of_Truth.md):
    §2: NO Python loops — fully vectorized PyTorch

Tag: [PHASE-16] [HVAC] [TIER-1] [FIXED]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor


@dataclass
class ProjectionConfig:
    """Configuration for projection method solver."""
    
    # Geometry (meters)
    length: float = 9.0
    height: float = 3.0
    
    # Grid
    nx: int = 256
    ny: int = 128
    
    # Inlet (upper left)
    inlet_y_start: float = 2.832
    inlet_y_end: float = 3.0
    inlet_velocity: float = 0.455
    
    # Outlet (lower right)
    outlet_y_start: float = 0.0
    outlet_y_end: float = 0.480
    
    # Fluid
    nu: float = 1.5e-5
    Re: float | None = None
    
    # Solver
    max_iterations: int = 5000
    convergence_tol: float = 1e-6
    dt_safety: float = 0.25  # Reduced from 0.3 for central differences
    
    # Pressure solver
    pressure_iterations: int = 100
    pressure_tol: float = 1e-8
    
    # Advection scheme: 'central', 'skew_symmetric', 'hybrid'
    advection_scheme: str = 'skew_symmetric'
    
    verbose: bool = True
    diag_interval: int = 100
    
    def __post_init__(self):
        self.dx = self.length / (self.nx - 1)
        self.dy = self.height / (self.ny - 1)
        self.inlet_height = self.inlet_y_end - self.inlet_y_start
        
        if self.Re is not None:
            self.nu = self.inlet_velocity * self.inlet_height / self.Re
        else:
            self.Re = self.inlet_velocity * self.inlet_height / self.nu


@dataclass
class ProjectionState:
    """State for projection method."""
    u: Tensor
    v: Tensor
    p: Tensor
    iteration: int = 0
    converged: bool = False
    residual_history: list[float] = field(default_factory=list)
    
    @property
    def velocity_magnitude(self) -> Tensor:
        return torch.sqrt(self.u**2 + self.v**2)


class ProjectionSolver:
    """
    Projection Method Solver for 2D Incompressible Flow.
    
    Uses fractional step method with:
    - CENTRAL DIFFERENCE advection (non-diffusive)
    - Implicit pressure (Jacobi iteration)
    - Velocity correction for divergence-free field
    
    Key Change: Advection uses central/skew-symmetric form to preserve
    jet structure at high Reynolds numbers.
    """
    
    def __init__(
        self,
        config: ProjectionConfig,
        dtype: torch.dtype = torch.float64,
        device: str | None = None,
    ):
        self.config = config
        self.dtype = dtype
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self._setup_grid()
        self._setup_boundary_masks()
        
    def _setup_grid(self) -> None:
        cfg = self.config
        
        self.x = torch.linspace(0, cfg.length, cfg.nx, dtype=self.dtype, device=self.device)
        self.y = torch.linspace(0, cfg.height, cfg.ny, dtype=self.dtype, device=self.device)
        
        self.dx = cfg.dx
        self.dy = cfg.dy
        self.dx2 = self.dx ** 2
        self.dy2 = self.dy ** 2
        
    def _setup_boundary_masks(self) -> None:
        cfg = self.config
        nx, ny = cfg.nx, cfg.ny
        
        # Inlet mask (upper portion of left wall)
        inlet_j_start = int((cfg.inlet_y_start / cfg.height) * (ny - 1))
        self.inlet_j_start = inlet_j_start
        self.inlet_j_end = ny
        
        self.inlet_mask = torch.zeros(nx, ny, dtype=torch.bool, device=self.device)
        self.inlet_mask[0, inlet_j_start:ny] = True
        
        # Outlet mask (lower portion of right wall)
        outlet_j_end = int((cfg.outlet_y_end / cfg.height) * (ny - 1))
        self.outlet_j_end = outlet_j_end
        
        self.outlet_mask = torch.zeros(nx, ny, dtype=torch.bool, device=self.device)
        self.outlet_mask[-1, 0:outlet_j_end] = True
        
        # Interior mask (for applying advection/diffusion)
        self.interior_mask = torch.ones(nx, ny, dtype=torch.bool, device=self.device)
        self.interior_mask[0, :] = False   # Left
        self.interior_mask[-1, :] = False  # Right
        self.interior_mask[:, 0] = False   # Bottom
        self.interior_mask[:, -1] = False  # Top
        
    def create_initial_state(self) -> ProjectionState:
        cfg = self.config
        nx, ny = cfg.nx, cfg.ny
        
        u = torch.zeros(nx, ny, dtype=self.dtype, device=self.device)
        v = torch.zeros(nx, ny, dtype=self.dtype, device=self.device)
        p = torch.zeros(nx, ny, dtype=self.dtype, device=self.device)
        
        # Initialize with inlet velocity
        u[self.inlet_mask] = cfg.inlet_velocity
        
        return ProjectionState(u=u, v=v, p=p)
    
    def apply_velocity_bc(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Apply velocity boundary conditions."""
        cfg = self.config
        
        # ===== INLET (left wall, upper portion) =====
        # Horizontal jet entering room
        u[0, self.inlet_j_start:] = cfg.inlet_velocity
        v[0, self.inlet_j_start:] = 0.0
        
        # ===== LEFT WALL (below inlet) =====
        # No-slip
        u[0, :self.inlet_j_start] = 0.0
        v[0, :self.inlet_j_start] = 0.0
        
        # ===== RIGHT WALL (above outlet) =====
        # No-slip
        u[-1, self.outlet_j_end:] = 0.0
        v[-1, self.outlet_j_end:] = 0.0
        
        # ===== OUTLET (right wall, lower portion) =====
        # Zero gradient (convective outflow)
        u[-1, :self.outlet_j_end] = u[-2, :self.outlet_j_end]
        v[-1, :self.outlet_j_end] = v[-2, :self.outlet_j_end]
        
        # ===== BOTTOM WALL =====
        u[:, 0] = 0.0
        v[:, 0] = 0.0
        
        # ===== TOP WALL (ceiling, except inlet) =====
        u[1:, -1] = 0.0  # Skip inlet at i=0
        v[:, -1] = 0.0
        
        return u, v
    
    def apply_pressure_bc(self, p: Tensor) -> Tensor:
        """Apply pressure boundary conditions (Neumann on walls)."""
        # Neumann BC: ∂p/∂n = 0
        p[0, :] = p[1, :]    # Left
        p[-1, :] = p[-2, :]  # Right
        p[:, 0] = p[:, 1]    # Bottom
        p[:, -1] = p[:, -2]  # Top
        
        # Reference pressure at outlet
        p[-1, :self.outlet_j_end] = 0.0
        
        return p
    
    # =========================================================================
    # ADVECTION SCHEMES (THE KEY FIX)
    # =========================================================================
    
    def compute_advection_central(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Central difference advection (2nd order, NO numerical diffusion).
        
        This is the critical fix. Central differences have zero intrinsic
        numerical viscosity, unlike upwind which has ν_num ≈ U·Δx/2.
        
        At Re=5000:
            - Upwind ν_num ≈ 8e-3 m²/s (535× physical!)
            - Central ν_num ≈ 0
        """
        dx, dy = self.dx, self.dy
        
        # Central difference gradients
        du_dx = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2 * dx)
        du_dy = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dy)
        
        dv_dx = (torch.roll(v, -1, dims=0) - torch.roll(v, 1, dims=0)) / (2 * dx)
        dv_dy = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * dy)
        
        # Convective form: (u·∇)u
        adv_u = u * du_dx + v * du_dy
        adv_v = u * dv_dx + v * dv_dy
        
        return -adv_u, -adv_v
    
    def compute_advection_skew_symmetric(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Skew-symmetric advection (energy-conserving).
        
        Uses: advection = 0.5 * (convective + conservative form)
        
        This form conserves kinetic energy exactly (to machine precision)
        even with central differences, preventing artificial energy growth
        that can destabilize high-Re flows.
        
        Reference: Morinishi et al. (1998), JCP 143, 90-124
        """
        dx, dy = self.dx, self.dy
        
        # Convective form: (u·∇)u
        du_dx = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2 * dx)
        du_dy = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dy)
        dv_dx = (torch.roll(v, -1, dims=0) - torch.roll(v, 1, dims=0)) / (2 * dx)
        dv_dy = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * dy)
        
        conv_u = u * du_dx + v * du_dy
        conv_v = u * dv_dx + v * dv_dy
        
        # Conservative form: ∇·(u⊗u)
        uu = u * u
        uv = u * v
        vv = v * v
        
        d_uu_dx = (torch.roll(uu, -1, dims=0) - torch.roll(uu, 1, dims=0)) / (2 * dx)
        d_uv_dy = (torch.roll(uv, -1, dims=1) - torch.roll(uv, 1, dims=1)) / (2 * dy)
        d_uv_dx = (torch.roll(uv, -1, dims=0) - torch.roll(uv, 1, dims=0)) / (2 * dx)
        d_vv_dy = (torch.roll(vv, -1, dims=1) - torch.roll(vv, 1, dims=1)) / (2 * dy)
        
        cons_u = d_uu_dx + d_uv_dy
        cons_v = d_uv_dx + d_vv_dy
        
        # Skew-symmetric = average of convective and conservative
        adv_u = 0.5 * (conv_u + cons_u)
        adv_v = 0.5 * (conv_v + cons_v)
        
        return -adv_u, -adv_v
    
    def compute_advection(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Dispatch to configured advection scheme."""
        scheme = self.config.advection_scheme
        
        if scheme == 'central':
            return self.compute_advection_central(u, v)
        elif scheme == 'skew_symmetric':
            return self.compute_advection_skew_symmetric(u, v)
        else:
            raise ValueError(f"Unknown advection scheme: {scheme}")
    
    def compute_diffusion(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Compute diffusion terms ν∇²u, ν∇²v (5-point Laplacian)."""
        nu = self.config.nu
        dx2, dy2 = self.dx2, self.dy2
        
        # 5-point Laplacian stencil
        lap_u = (
            (torch.roll(u, -1, dims=0) - 2*u + torch.roll(u, 1, dims=0)) / dx2 +
            (torch.roll(u, -1, dims=1) - 2*u + torch.roll(u, 1, dims=1)) / dy2
        )
        
        lap_v = (
            (torch.roll(v, -1, dims=0) - 2*v + torch.roll(v, 1, dims=0)) / dx2 +
            (torch.roll(v, -1, dims=1) - 2*v + torch.roll(v, 1, dims=1)) / dy2
        )
        
        return nu * lap_u, nu * lap_v
    
    def compute_divergence(self, u: Tensor, v: Tensor) -> Tensor:
        """Compute divergence ∇·u (central differences)."""
        div = (
            (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2 * self.dx) +
            (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * self.dy)
        )
        return div
    
    def solve_pressure_poisson(self, rhs: Tensor, p_init: Tensor) -> Tensor:
        """
        Solve pressure Poisson equation: ∇²p = rhs
        
        Uses Jacobi iteration (vectorized, no Python loops).
        """
        p = p_init.clone()
        cfg = self.config
        
        coeff = 2.0 / self.dx2 + 2.0 / self.dy2
        
        for _ in range(cfg.pressure_iterations):
            p_old = p.clone()
            
            # Jacobi update (vectorized)
            p = (
                (torch.roll(p_old, -1, dims=0) + torch.roll(p_old, 1, dims=0)) / self.dx2 +
                (torch.roll(p_old, -1, dims=1) + torch.roll(p_old, 1, dims=1)) / self.dy2 -
                rhs
            ) / coeff
            
            # Apply BCs
            p = self.apply_pressure_bc(p)
            
            # Check convergence
            diff = torch.abs(p - p_old).max().item()
            if diff < cfg.pressure_tol:
                break
                
        return p
    
    def compute_pressure_gradient(self, p: Tensor) -> tuple[Tensor, Tensor]:
        """Compute pressure gradient ∇p (central differences)."""
        dpdx = (torch.roll(p, -1, dims=0) - torch.roll(p, 1, dims=0)) / (2 * self.dx)
        dpdy = (torch.roll(p, -1, dims=1) - torch.roll(p, 1, dims=1)) / (2 * self.dy)
        return dpdx, dpdy
    
    def compute_timestep(self, u: Tensor, v: Tensor) -> float:
        """
        Compute stable timestep from CFL and viscous limits.
        
        Central differences require CFL < 1 (unlike upwind which is unconditional).
        """
        cfg = self.config
        
        max_u = torch.abs(u).max().item() + 1e-10
        max_v = torch.abs(v).max().item() + 1e-10
        
        # CFL condition (stricter for central differences)
        dt_cfl = cfg.dt_safety * min(self.dx / max_u, self.dy / max_v)
        
        # Viscous stability
        dt_visc = cfg.dt_safety * min(self.dx2, self.dy2) / (4 * cfg.nu)
        
        return min(dt_cfl, dt_visc)
    
    def solve(self, initial_state: ProjectionState | None = None) -> ProjectionState:
        """Solve to steady state using projection method."""
        cfg = self.config
        
        if initial_state is None:
            state = self.create_initial_state()
        else:
            state = initial_state
            
        u, v, p = state.u.clone(), state.v.clone(), state.p.clone()
        u, v = self.apply_velocity_bc(u, v)
        
        residual_history = []
        start_time = time.perf_counter()
        
        for iteration in range(cfg.max_iterations):
            u_old, v_old = u.clone(), v.clone()
            
            # 1. Compute timestep
            dt = self.compute_timestep(u, v)
            
            # 2. Predict velocity (explicit advection-diffusion)
            adv_u, adv_v = self.compute_advection(u, v)
            diff_u, diff_v = self.compute_diffusion(u, v)
            
            u_star = u + dt * (adv_u + diff_u)
            v_star = v + dt * (adv_v + diff_v)
            
            # Apply BCs to predicted velocity
            u_star, v_star = self.apply_velocity_bc(u_star, v_star)
            
            # 3. Solve pressure Poisson: ∇²p = (1/dt) ∇·u*
            div_ustar = self.compute_divergence(u_star, v_star)
            rhs = div_ustar / dt
            p = self.solve_pressure_poisson(rhs, p)
            
            # 4. Correct velocity: u = u* - dt * ∇p
            dpdx, dpdy = self.compute_pressure_gradient(p)
            u = u_star - dt * dpdx
            v = v_star - dt * dpdy
            
            # 5. Apply BCs to corrected velocity
            u, v = self.apply_velocity_bc(u, v)
            
            # 6. Check convergence
            du = torch.abs(u - u_old).max().item()
            dv = torch.abs(v - v_old).max().item()
            residual = max(du, dv)
            residual_history.append(residual)
            
            if cfg.verbose and iteration % cfg.diag_interval == 0:
                elapsed = time.perf_counter() - start_time
                div_max = torch.abs(self.compute_divergence(u, v)).max().item()
                ke = 0.5 * (u**2 + v**2).sum().item() * self.dx * self.dy
                print(f"  Iter {iteration:5d}: residual = {residual:.2e}, "
                      f"div = {div_max:.2e}, KE = {ke:.4f}, dt = {dt:.2e}, time = {elapsed:.1f}s")
            
            if residual < cfg.convergence_tol:
                if cfg.verbose:
                    print(f"  ✓ Converged at iteration {iteration}")
                return ProjectionState(
                    u=u, v=v, p=p,
                    iteration=iteration, converged=True,
                    residual_history=residual_history
                )
        
        if cfg.verbose:
            elapsed = time.perf_counter() - start_time
            print(f"  ✗ Did not converge after {cfg.max_iterations} iterations ({elapsed:.1f}s)")
            
        return ProjectionState(
            u=u, v=v, p=p,
            iteration=cfg.max_iterations, converged=False,
            residual_history=residual_history
        )
    
    def extract_profile(
        self, 
        state: ProjectionState, 
        x_position: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Extract velocity profile at given x-position."""
        i = int(x_position / self.config.length * (self.config.nx - 1))
        i = max(0, min(i, self.config.nx - 1))
        
        return self.y.cpu(), state.u[i, :].cpu(), state.v[i, :].cpu()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FIXED Projection Method Solver Test")
    print("Using SKEW-SYMMETRIC advection (zero numerical diffusion)")
    print("=" * 70)
    
    config = ProjectionConfig(
        nx=128, ny=64,
        Re=5000,
        max_iterations=1000,
        advection_scheme='skew_symmetric',  # THE FIX
        verbose=True,
    )
    
    print(f"\nGrid: {config.nx}×{config.ny}")
    print(f"Re = {config.Re}")
    print(f"Physical ν = {config.nu:.2e} m²/s")
    print(f"Advection: {config.advection_scheme}")
    print()
    
    solver = ProjectionSolver(config)
    state = solver.solve()
    
    print(f"\nMax velocity: {state.velocity_magnitude.max().item():.3f} m/s")
    print(f"Inlet velocity: {config.inlet_velocity} m/s")
    print(f"Converged: {state.converged}")
    print(f"Iterations: {state.iteration}")
