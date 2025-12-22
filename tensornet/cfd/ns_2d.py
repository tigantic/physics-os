"""
2D Incompressible Navier-Stokes Solver [PHASE-1A]
==================================================

Solves the incompressible Navier-Stokes equations:
    ∂u/∂t + (u·∇)u = -∇p + ν∇²u
    ∇·u = 0

Using the Chorin-Temam projection method [DECISION-005]:
1. Predictor: u* = u^n - dt[(u·∇)u - ν∇²u]
2. Poisson: ∇²φ = ∇·u* / dt  
3. Corrector: u^{n+1} = u* - dt∇φ

The projection step ensures exact incompressibility (∇·u = 0).

This is the foundation for studying regularity and potential
singularity formation through the χ(t) diagnostic.

Constitution Compliance: Article IV.1 (Verification), Phase 1a
Tag: [PHASE-1A] [DECISION-005]
"""

from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
import math

from tensornet.cfd.tt_poisson import (
    compute_advection_2d,
    compute_diffusion_2d,
    compute_divergence_2d,
    compute_gradient_2d,
    compute_vorticity_2d,
    project_velocity_2d,
    ProjectionResult,
)


@dataclass
class NSState:
    """State of 2D Navier-Stokes simulation."""
    u: Tensor              # x-velocity, shape (Nx, Ny)
    v: Tensor              # y-velocity, shape (Nx, Ny)
    t: float = 0.0         # Current time
    step: int = 0          # Time step counter
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.u.shape
    
    @property
    def dtype(self) -> torch.dtype:
        return self.u.dtype
    
    @property
    def device(self):
        return self.u.device


@dataclass
class NSDiagnostics:
    """
    Diagnostics for NS simulation.
    
    Tracks quantities relevant for regularity analysis.
    """
    time: float
    kinetic_energy: float      # (1/2)∫|u|² dV
    enstrophy: float           # (1/2)∫|ω|² dV
    max_vorticity: float       # max|ω|
    max_divergence: float      # max|∇·u| (should be ~0)
    cfl: float                 # CFL number
    
    # χ-regularity proxy (max vorticity / initial vorticity)
    chi_proxy: Optional[float] = None


@dataclass
class NSResult:
    """Result container for NS simulation."""
    final_state: NSState
    diagnostics_history: List[NSDiagnostics]
    dt_used: float
    nu: float
    completed: bool
    reason: str = ""


class NS2DSolver:
    """
    2D Incompressible Navier-Stokes solver.
    
    Uses spectral discretization with projection method.
    """
    
    def __init__(
        self,
        Nx: int,
        Ny: int,
        Lx: float = 2 * math.pi,
        Ly: float = 2 * math.pi,
        nu: float = 0.01,
        dtype: torch.dtype = torch.float64,
        device: str = 'cpu',
    ):
        """
        Initialize solver.
        
        Args:
            Nx, Ny: Grid points in x, y
            Lx, Ly: Domain size
            nu: Kinematic viscosity
            dtype: Tensor dtype
            device: Device
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.nu = nu
        self.dtype = dtype
        self.device = device
        
        # Create grids (using indexing='ij')
        x = torch.linspace(0, Lx - self.dx, Nx, dtype=dtype, device=device)
        y = torch.linspace(0, Ly - self.dy, Ny, dtype=dtype, device=device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
    
    def create_taylor_green(self, A: float = 1.0) -> NSState:
        """
        Create Taylor-Green vortex initial condition.
        
        Exact solution for 2D incompressible NS:
            u = A cos(x) sin(y) exp(-2νt)
            v = -A sin(x) cos(y) exp(-2νt)
        
        This is a benchmark for numerical accuracy.
        """
        u = A * torch.cos(self.X) * torch.sin(self.Y)
        v = -A * torch.sin(self.X) * torch.cos(self.Y)
        
        return NSState(u=u, v=v, t=0.0, step=0)
    
    def compute_rhs(
        self,
        state: NSState,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute RHS of NS equations (before projection).
        
        RHS = -advection + viscous_diffusion
            = -(u·∇)u + ν∇²u
        """
        # Advection: (u·∇)u
        adv_u, adv_v = compute_advection_2d(
            state.u, state.v, self.dx, self.dy, method='spectral'
        )
        
        # Diffusion: ν∇²u
        diff_u, diff_v = compute_diffusion_2d(
            state.u, state.v, self.dx, self.dy, method='spectral'
        )
        
        # RHS = -advection + ν * diffusion
        rhs_u = -adv_u + self.nu * diff_u
        rhs_v = -adv_v + self.nu * diff_v
        
        return rhs_u, rhs_v
    
    def step_forward_euler(
        self,
        state: NSState,
        dt: float,
    ) -> Tuple[NSState, ProjectionResult]:
        """
        Take one time step using Forward Euler + Projection.
        
        Steps:
        1. u* = u^n + dt * RHS
        2. Project u* to divergence-free space
        """
        # Compute RHS
        rhs_u, rhs_v = self.compute_rhs(state)
        
        # Predictor step
        u_star = state.u + dt * rhs_u
        v_star = state.v + dt * rhs_v
        
        # Projection step
        proj = project_velocity_2d(
            u_star, v_star, self.dx, self.dy, 
            dt=1.0,  # Projection doesn't scale by dt
            bc='periodic',
            method='spectral'
        )
        
        # New state
        new_state = NSState(
            u=proj.u_projected,
            v=proj.v_projected,
            t=state.t + dt,
            step=state.step + 1,
        )
        
        return new_state, proj
    
    def compute_diagnostics(
        self,
        state: NSState,
        dt: float,
        initial_max_vorticity: Optional[float] = None,
    ) -> NSDiagnostics:
        """Compute diagnostic quantities."""
        # Kinetic energy: (1/2) ∫|u|² dV
        ke = 0.5 * (state.u**2 + state.v**2).mean().item() * self.Lx * self.Ly
        
        # Vorticity
        omega = compute_vorticity_2d(
            state.u, state.v, self.dx, self.dy, method='spectral'
        )
        max_vort = torch.abs(omega).max().item()
        
        # Enstrophy: (1/2) ∫|ω|² dV
        enstrophy = 0.5 * (omega**2).mean().item() * self.Lx * self.Ly
        
        # Divergence
        div = compute_divergence_2d(
            state.u, state.v, self.dx, self.dy, method='spectral'
        )
        max_div = torch.abs(div).max().item()
        
        # CFL number
        max_u = torch.abs(state.u).max().item()
        max_v = torch.abs(state.v).max().item()
        cfl = dt * (max_u / self.dx + max_v / self.dy)
        
        # χ-proxy
        chi = None
        if initial_max_vorticity is not None and initial_max_vorticity > 0:
            chi = max_vort / initial_max_vorticity
        
        return NSDiagnostics(
            time=state.t,
            kinetic_energy=ke,
            enstrophy=enstrophy,
            max_vorticity=max_vort,
            max_divergence=max_div,
            cfl=cfl,
            chi_proxy=chi,
        )
    
    def compute_stable_dt(
        self,
        state: NSState,
        cfl_target: float = 0.5,
    ) -> float:
        """Compute stable time step based on CFL condition."""
        max_u = torch.abs(state.u).max().item()
        max_v = torch.abs(state.v).max().item()
        
        # Advective CFL: dt < dx / max_u
        dt_adv = cfl_target * min(self.dx / (max_u + 1e-10), 
                                   self.dy / (max_v + 1e-10))
        
        # Viscous stability: dt < dx² / (2ν)
        dt_visc = cfl_target * min(self.dx**2, self.dy**2) / (4 * self.nu + 1e-10)
        
        return min(dt_adv, dt_visc)
    
    def solve(
        self,
        initial_state: NSState,
        t_final: float,
        dt: Optional[float] = None,
        cfl_target: float = 0.5,
        diag_interval: int = 10,
        max_steps: int = 100000,
        verbose: bool = True,
    ) -> NSResult:
        """
        Integrate NS equations from initial state to t_final.
        
        Args:
            initial_state: Initial velocity field
            t_final: Final time
            dt: Time step (if None, computed adaptively)
            cfl_target: Target CFL number
            diag_interval: Steps between diagnostics
            max_steps: Maximum number of steps
            verbose: Print progress
            
        Returns:
            NSResult with final state and diagnostics
        """
        state = initial_state
        diagnostics = []
        
        # Initial diagnostics
        if dt is None:
            dt = self.compute_stable_dt(state, cfl_target)
        
        initial_diag = self.compute_diagnostics(state, dt)
        initial_max_vort = initial_diag.max_vorticity
        diagnostics.append(initial_diag)
        
        if verbose:
            print(f"NS2D Solver: t_final={t_final:.4f}, dt={dt:.2e}, ν={self.nu:.2e}")
            print(f"  Initial: KE={initial_diag.kinetic_energy:.4e}, "
                  f"ω_max={initial_diag.max_vorticity:.4e}")
        
        # Time integration loop
        for step_idx in range(max_steps):
            if state.t >= t_final:
                break
            
            # Adjust dt for final step
            dt_step = min(dt, t_final - state.t)
            
            # Take step
            state, proj = self.step_forward_euler(state, dt_step)
            
            # Diagnostics
            if step_idx % diag_interval == 0 or state.t >= t_final:
                diag = self.compute_diagnostics(state, dt, initial_max_vort)
                diagnostics.append(diag)
                
                if verbose and step_idx % (diag_interval * 10) == 0:
                    print(f"  t={state.t:.4f}: KE={diag.kinetic_energy:.4e}, "
                          f"ω_max={diag.max_vorticity:.4e}, div={diag.max_divergence:.2e}")
        
        completed = state.t >= t_final - 1e-10
        reason = "Completed" if completed else f"Max steps ({max_steps}) reached"
        
        if verbose:
            final_diag = diagnostics[-1]
            print(f"  Final: KE={final_diag.kinetic_energy:.4e}, "
                  f"ω_max={final_diag.max_vorticity:.4e}")
        
        return NSResult(
            final_state=state,
            diagnostics_history=diagnostics,
            dt_used=dt,
            nu=self.nu,
            completed=completed,
            reason=reason,
        )


def taylor_green_exact(
    X: Tensor,
    Y: Tensor,
    t: float,
    nu: float,
    A: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """
    Exact Taylor-Green solution.
    
    u = A cos(x) sin(y) exp(-2νt)
    v = -A sin(x) cos(y) exp(-2νt)
    """
    decay = A * math.exp(-2 * nu * t)
    u = decay * torch.cos(X) * torch.sin(Y)
    v = -decay * torch.sin(X) * torch.cos(Y)
    return u, v


def test_taylor_green():
    """
    Test solver against Taylor-Green exact solution.
    
    Phase 1a gate: match analytic decay rate < 5% error.
    """
    print("\n" + "=" * 60)
    print("Taylor-Green Validation Test [PHASE-1A]")
    print("=" * 60)
    
    # Parameters
    N = 64
    L = 2 * math.pi
    nu = 0.1  # Higher viscosity for faster decay
    t_final = 1.0
    
    # Create solver
    solver = NS2DSolver(N, N, L, L, nu=nu)
    
    # Initial condition
    state0 = solver.create_taylor_green(A=1.0)
    
    # Solve with conservative CFL for stability
    result = solver.solve(state0, t_final, cfl_target=0.2, diag_interval=10, verbose=True)
    
    # Compare to exact solution
    u_exact, v_exact = taylor_green_exact(solver.X, solver.Y, t_final, nu)
    
    u_error = torch.abs(result.final_state.u - u_exact).max().item()
    v_error = torch.abs(result.final_state.v - v_exact).max().item()
    max_error = max(u_error, v_error)
    
    # Compute energy decay rate
    initial_ke = result.diagnostics_history[0].kinetic_energy
    final_ke = result.diagnostics_history[-1].kinetic_energy
    
    # Exact decay: KE(t) = KE(0) * exp(-4νt)
    exact_decay = math.exp(-4 * nu * t_final)
    computed_decay = final_ke / initial_ke
    decay_error = abs(computed_decay - exact_decay) / exact_decay
    
    print(f"\n  Exact energy decay factor: {exact_decay:.6f}")
    print(f"  Computed decay factor:     {computed_decay:.6f}")
    print(f"  Decay rate error:          {decay_error * 100:.2f}%")
    print(f"  Max velocity error:        {max_error:.4e}")
    print(f"  Max divergence:            {result.diagnostics_history[-1].max_divergence:.2e}")
    
    # Gate criteria
    decay_pass = decay_error < 0.05  # < 5% error
    div_pass = result.diagnostics_history[-1].max_divergence < 1e-6
    
    print(f"\n  Decay rate gate (<5% error): {'PASS' if decay_pass else 'FAIL'}")
    print(f"  Divergence gate (<1e-6):     {'PASS' if div_pass else 'FAIL'}")
    print("=" * 60)
    
    return {
        'decay_error': decay_error,
        'max_velocity_error': max_error,
        'decay_passed': decay_pass,
        'divergence_passed': div_pass,
        'passed': decay_pass and div_pass,
    }


if __name__ == '__main__':
    result = test_taylor_green()
    print(f"\nResult: {result}")
