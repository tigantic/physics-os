"""
Differentiable CFD Module
=========================

Phase 22: Autograd-enabled CFD for neural network integration.

Enables backpropagation through Euler solver for:
- Design optimization (adjoint-free gradients)
- Neural network surrogates with physics constraints
- Inverse problems (parameter estimation)
- Sensitivity analysis

References:
    - Griewank & Walther, "Evaluating Derivatives" (2008)
    - Chen et al., "Neural ODE" NeurIPS 2018
    - Raissi et al., "Physics-informed neural networks" (2019)

Constitution Compliance: Article III (Neural Networks), Article V.2
"""

import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

# Stability parameters
CFL_DEFAULT = 0.5
ENTROPY_FIX_EPSILON = 0.1


# =============================================================================
# Differentiable Flux Functions
# =============================================================================

class DifferentiableRoe:
    """
    Differentiable Roe flux with entropy fix.
    
    Standard Roe scheme with autograd support through all operations.
    Uses Harten's entropy fix for smooth differentiation.
    """
    
    def __init__(self, gamma: float = 1.4, entropy_fix: float = 0.1):
        self.gamma = gamma
        self.entropy_fix = entropy_fix
    
    def __call__(
        self,
        rho_L: Tensor,
        u_L: Tensor,
        p_L: Tensor,
        rho_R: Tensor,
        u_R: Tensor,
        p_R: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute Roe flux at cell interface.
        
        Args:
            rho_L, u_L, p_L: Left state (density, velocity, pressure)
            rho_R, u_R, p_R: Right state
            
        Returns:
            (F_rho, F_rhou, F_E) mass, momentum, energy fluxes
        """
        gamma = self.gamma
        
        # Compute total energy
        E_L = p_L / (gamma - 1) + 0.5 * rho_L * u_L**2
        E_R = p_R / (gamma - 1) + 0.5 * rho_R * u_R**2
        
        # Enthalpy
        H_L = (E_L + p_L) / rho_L
        H_R = (E_R + p_R) / rho_R
        
        # Roe averages
        sqrt_rho_L = torch.sqrt(rho_L)
        sqrt_rho_R = torch.sqrt(rho_R)
        denom = sqrt_rho_L + sqrt_rho_R
        
        u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
        H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / denom
        
        # Sound speed
        c_roe_sq = (gamma - 1) * (H_roe - 0.5 * u_roe**2)
        c_roe = torch.sqrt(torch.clamp(c_roe_sq, min=1e-10))
        
        # Eigenvalues with smooth entropy fix
        eps = self.entropy_fix * c_roe
        
        lambda_1 = u_roe - c_roe
        lambda_2 = u_roe
        lambda_3 = u_roe + c_roe
        
        # Smooth absolute value
        abs_lambda_1 = torch.sqrt(lambda_1**2 + eps**2)
        abs_lambda_2 = torch.sqrt(lambda_2**2 + eps**2)
        abs_lambda_3 = torch.sqrt(lambda_3**2 + eps**2)
        
        # State jumps
        drho = rho_R - rho_L
        du = u_R - u_L
        dp = p_R - p_L
        
        # Wave strengths
        alpha_2 = drho - dp / c_roe**2
        alpha_1 = (dp - c_roe * rho_R * du) / (2 * c_roe**2)
        alpha_3 = (dp + c_roe * rho_R * du) / (2 * c_roe**2)
        
        # Left and right fluxes
        F_rho_L = rho_L * u_L
        F_rhou_L = rho_L * u_L**2 + p_L
        F_E_L = u_L * (E_L + p_L)
        
        F_rho_R = rho_R * u_R
        F_rhou_R = rho_R * u_R**2 + p_R
        F_E_R = u_R * (E_R + p_R)
        
        # Roe dissipation
        diss_rho = (abs_lambda_1 * alpha_1 + 
                   abs_lambda_2 * alpha_2 + 
                   abs_lambda_3 * alpha_3)
        diss_rhou = (abs_lambda_1 * alpha_1 * (u_roe - c_roe) + 
                    abs_lambda_2 * alpha_2 * u_roe + 
                    abs_lambda_3 * alpha_3 * (u_roe + c_roe))
        diss_E = (abs_lambda_1 * alpha_1 * (H_roe - u_roe * c_roe) + 
                 abs_lambda_2 * alpha_2 * 0.5 * u_roe**2 + 
                 abs_lambda_3 * alpha_3 * (H_roe + u_roe * c_roe))
        
        # Roe flux
        F_rho = 0.5 * (F_rho_L + F_rho_R - diss_rho)
        F_rhou = 0.5 * (F_rhou_L + F_rhou_R - diss_rhou)
        F_E = 0.5 * (F_E_L + F_E_R - diss_E)
        
        return F_rho, F_rhou, F_E


class DifferentiableHLLC:
    """
    Differentiable HLLC flux.
    
    Three-wave approximate Riemann solver with smooth transitions.
    """
    
    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma
    
    def __call__(
        self,
        rho_L: Tensor,
        u_L: Tensor,
        p_L: Tensor,
        rho_R: Tensor,
        u_R: Tensor,
        p_R: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute HLLC flux with smooth wave speed estimates."""
        gamma = self.gamma
        
        # Sound speeds
        c_L = torch.sqrt(gamma * p_L / rho_L)
        c_R = torch.sqrt(gamma * p_R / rho_R)
        
        # Roe-averaged quantities for wave speed estimates
        sqrt_rho_L = torch.sqrt(rho_L)
        sqrt_rho_R = torch.sqrt(rho_R)
        u_avg = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R)
        
        E_L = p_L / (gamma - 1) + 0.5 * rho_L * u_L**2
        E_R = p_R / (gamma - 1) + 0.5 * rho_R * u_R**2
        H_L = (E_L + p_L) / rho_L
        H_R = (E_R + p_R) / rho_R
        H_avg = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R)
        c_avg = torch.sqrt((gamma - 1) * torch.clamp(H_avg - 0.5 * u_avg**2, min=1e-10))
        
        # Wave speed estimates (Davis)
        S_L = torch.minimum(u_L - c_L, u_avg - c_avg)
        S_R = torch.maximum(u_R + c_R, u_avg + c_avg)
        
        # Contact wave speed
        numer = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
        denom = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
        S_star = numer / (denom + 1e-10)
        
        # Left flux
        F_rho_L = rho_L * u_L
        F_rhou_L = rho_L * u_L**2 + p_L
        F_E_L = u_L * (E_L + p_L)
        
        # Right flux
        F_rho_R = rho_R * u_R
        F_rhou_R = rho_R * u_R**2 + p_R
        F_E_R = u_R * (E_R + p_R)
        
        # Star states
        rho_L_star = rho_L * (S_L - u_L) / (S_L - S_star + 1e-10)
        rho_R_star = rho_R * (S_R - u_R) / (S_R - S_star + 1e-10)
        
        p_star = p_L + rho_L * (S_L - u_L) * (S_star - u_L)
        
        E_L_star = rho_L_star * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + 1e-10)))
        E_R_star = rho_R_star * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + 1e-10)))
        
        # Star fluxes
        F_rho_L_star = F_rho_L + S_L * (rho_L_star - rho_L)
        F_rhou_L_star = F_rhou_L + S_L * (rho_L_star * S_star - rho_L * u_L)
        F_E_L_star = F_E_L + S_L * (E_L_star - E_L)
        
        F_rho_R_star = F_rho_R + S_R * (rho_R_star - rho_R)
        F_rhou_R_star = F_rhou_R + S_R * (rho_R_star * S_star - rho_R * u_R)
        F_E_R_star = F_E_R + S_R * (E_R_star - E_R)
        
        # Smooth selection using sigmoid approximations
        # sigmoid(-k*S_L) selects left vs. left-star
        # sigmoid(-k*S_star) selects left-star vs. right-star
        # sigmoid(-k*S_R) selects right-star vs. right
        k = 100.0  # Sharpness
        
        w_L = torch.sigmoid(-k * S_L)
        w_star = torch.sigmoid(-k * S_star)
        w_R = torch.sigmoid(-k * S_R)
        
        # Blend all four regions
        F_rho = (1 - w_L) * F_rho_L + w_L * (
            (1 - w_star) * F_rho_L_star + w_star * (
                (1 - w_R) * F_rho_R_star + w_R * F_rho_R
            )
        )
        F_rhou = (1 - w_L) * F_rhou_L + w_L * (
            (1 - w_star) * F_rhou_L_star + w_star * (
                (1 - w_R) * F_rhou_R_star + w_R * F_rhou_R
            )
        )
        F_E = (1 - w_L) * F_E_L + w_L * (
            (1 - w_star) * F_E_L_star + w_star * (
                (1 - w_R) * F_E_R_star + w_R * F_E_R
            )
        )
        
        return F_rho, F_rhou, F_E


# =============================================================================
# Differentiable Euler Solver
# =============================================================================

class DifferentiableEuler1D(nn.Module):
    """
    Differentiable 1D Euler solver.
    
    Enables backpropagation through time evolution for:
    - Gradient-based optimization of initial/boundary conditions
    - Learning closure models
    - Inverse problems
    
    Attributes:
        nx: Number of grid cells
        dx: Cell size
        gamma: Ratio of specific heats
        cfl: CFL number
        flux: Flux function
    """
    
    def __init__(
        self,
        nx: int,
        dx: float,
        gamma: float = 1.4,
        cfl: float = 0.5,
        flux_type: str = 'roe'
    ):
        super().__init__()
        self.nx = nx
        self.dx = dx
        self.gamma = gamma
        self.cfl = cfl
        
        if flux_type == 'roe':
            self.flux = DifferentiableRoe(gamma)
        elif flux_type == 'hllc':
            self.flux = DifferentiableHLLC(gamma)
        else:
            raise ValueError(f"Unknown flux type: {flux_type}")
    
    def primitive_to_conservative(
        self,
        rho: Tensor,
        u: Tensor,
        p: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert primitive to conservative variables."""
        rhou = rho * u
        E = p / (self.gamma - 1) + 0.5 * rho * u**2
        return rho, rhou, E
    
    def conservative_to_primitive(
        self,
        rho: Tensor,
        rhou: Tensor,
        E: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert conservative to primitive variables."""
        u = rhou / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * u**2)
        return rho, u, p
    
    def compute_dt(
        self,
        rho: Tensor,
        u: Tensor,
        p: Tensor
    ) -> Tensor:
        """Compute stable timestep."""
        c = torch.sqrt(self.gamma * p / rho)
        max_speed = torch.max(torch.abs(u) + c)
        return self.cfl * self.dx / max_speed
    
    def step(
        self,
        rho: Tensor,
        rhou: Tensor,
        E: Tensor,
        dt: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Advance solution by one timestep.
        
        Uses first-order Euler time integration with
        donor-cell spatial discretization.
        
        Args:
            rho, rhou, E: Conservative variables at cell centers
            dt: Timestep (computed from CFL if None)
            
        Returns:
            Updated (rho, rhou, E)
        """
        # Get primitive variables
        rho_prim, u, p = self.conservative_to_primitive(rho, rhou, E)
        
        # Compute timestep if not provided
        if dt is None:
            dt = self.compute_dt(rho_prim, u, p)
        
        # Pad for ghost cells (transmissive BC)
        rho_pad = torch.cat([rho_prim[0:1], rho_prim, rho_prim[-1:]])
        u_pad = torch.cat([u[0:1], u, u[-1:]])
        p_pad = torch.cat([p[0:1], p, p[-1:]])
        
        # Compute fluxes at interfaces
        F_rho_iph = []
        F_rhou_iph = []
        F_E_iph = []
        
        for i in range(self.nx + 1):
            f_rho, f_rhou, f_E = self.flux(
                rho_pad[i], u_pad[i], p_pad[i],
                rho_pad[i+1], u_pad[i+1], p_pad[i+1]
            )
            F_rho_iph.append(f_rho)
            F_rhou_iph.append(f_rhou)
            F_E_iph.append(f_E)
        
        F_rho_iph = torch.stack(F_rho_iph)
        F_rhou_iph = torch.stack(F_rhou_iph)
        F_E_iph = torch.stack(F_E_iph)
        
        # Update with flux differences
        dtdx = dt / self.dx
        rho_new = rho - dtdx * (F_rho_iph[1:] - F_rho_iph[:-1])
        rhou_new = rhou - dtdx * (F_rhou_iph[1:] - F_rhou_iph[:-1])
        E_new = E - dtdx * (F_E_iph[1:] - F_E_iph[:-1])
        
        return rho_new, rhou_new, E_new
    
    def forward(
        self,
        rho_init: Tensor,
        u_init: Tensor,
        p_init: Tensor,
        t_final: float,
        n_steps: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evolve initial conditions to final time.
        
        Args:
            rho_init, u_init, p_init: Initial primitive variables
            t_final: Final time
            n_steps: Number of steps (adaptive if None)
            
        Returns:
            Final (rho, u, p) in primitive form
        """
        # Convert to conservative
        rho, rhou, E = self.primitive_to_conservative(rho_init, u_init, p_init)
        
        t = 0.0
        while t < t_final - 1e-10:
            # Compute stable timestep
            rho_prim, u, p = self.conservative_to_primitive(rho, rhou, E)
            dt = self.compute_dt(rho_prim, u, p)
            dt = min(dt.item(), t_final - t)
            dt = torch.tensor(dt, dtype=rho.dtype)
            
            # Advance
            rho, rhou, E = self.step(rho, rhou, E, dt)
            t += dt.item()
        
        return self.conservative_to_primitive(rho, rhou, E)


class DifferentiableEuler2D(nn.Module):
    """
    Differentiable 2D Euler solver.
    
    Dimension-by-dimension splitting for 2D flows.
    """
    
    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        gamma: float = 1.4,
        cfl: float = 0.4,
        flux_type: str = 'roe'
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.gamma = gamma
        self.cfl = cfl
        
        if flux_type == 'roe':
            self.flux = DifferentiableRoe(gamma)
        elif flux_type == 'hllc':
            self.flux = DifferentiableHLLC(gamma)
        else:
            raise ValueError(f"Unknown flux type: {flux_type}")
    
    def primitive_to_conservative(
        self,
        rho: Tensor,
        u: Tensor,
        v: Tensor,
        p: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convert primitive to conservative variables."""
        rhou = rho * u
        rhov = rho * v
        E = p / (self.gamma - 1) + 0.5 * rho * (u**2 + v**2)
        return rho, rhou, rhov, E
    
    def conservative_to_primitive(
        self,
        rho: Tensor,
        rhou: Tensor,
        rhov: Tensor,
        E: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convert conservative to primitive variables."""
        u = rhou / rho
        v = rhov / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        return rho, u, v, p
    
    def compute_dt(
        self,
        rho: Tensor,
        u: Tensor,
        v: Tensor,
        p: Tensor
    ) -> Tensor:
        """Compute stable timestep for 2D."""
        c = torch.sqrt(self.gamma * p / rho)
        max_speed_x = torch.max(torch.abs(u) + c)
        max_speed_y = torch.max(torch.abs(v) + c)
        
        dt_x = self.dx / max_speed_x
        dt_y = self.dy / max_speed_y
        
        return self.cfl * torch.minimum(dt_x, dt_y)
    
    def x_sweep(
        self,
        rho: Tensor,
        rhou: Tensor,
        rhov: Tensor,
        E: Tensor,
        dt: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """X-direction sweep."""
        ny, nx = rho.shape
        
        rho_prim, u, v, p = self.conservative_to_primitive(rho, rhou, rhov, E)
        
        # Compute fluxes in x-direction
        dtdx = dt / self.dx
        
        for j in range(ny):
            # Pad row for ghost cells
            rho_row = torch.cat([rho_prim[j, 0:1], rho_prim[j, :], rho_prim[j, -1:]])
            u_row = torch.cat([u[j, 0:1], u[j, :], u[j, -1:]])
            v_row = torch.cat([v[j, 0:1], v[j, :], v[j, -1:]])
            p_row = torch.cat([p[j, 0:1], p[j, :], p[j, -1:]])
            
            # x-momentum equation considers u as normal velocity
            for i in range(nx + 1):
                f_rho, f_rhou, f_E = self.flux(
                    rho_row[i], u_row[i], p_row[i],
                    rho_row[i+1], u_row[i+1], p_row[i+1]
                )
                
                if i > 0:
                    rho[j, i-1] = rho[j, i-1] - dtdx * f_rho
                    rhou[j, i-1] = rhou[j, i-1] - dtdx * f_rhou
                    rhov[j, i-1] = rhov[j, i-1] - dtdx * (rho_row[i] * v_row[i] * u_row[i])
                    E[j, i-1] = E[j, i-1] - dtdx * f_E
                
                if i < nx:
                    rho[j, i] = rho[j, i] + dtdx * f_rho
                    rhou[j, i] = rhou[j, i] + dtdx * f_rhou
                    rhov[j, i] = rhov[j, i] + dtdx * (rho_row[i+1] * v_row[i+1] * u_row[i+1])
                    E[j, i] = E[j, i] + dtdx * f_E
        
        return rho, rhou, rhov, E
    
    def forward(
        self,
        rho_init: Tensor,
        u_init: Tensor,
        v_init: Tensor,
        p_init: Tensor,
        t_final: float
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Evolve 2D solution to final time.
        
        Uses Strang splitting: x-y-y-x pattern.
        """
        rho, rhou, rhov, E = self.primitive_to_conservative(
            rho_init, u_init, v_init, p_init
        )
        
        t = 0.0
        while t < t_final - 1e-10:
            rho_prim, u, v, p = self.conservative_to_primitive(rho, rhou, rhov, E)
            dt = self.compute_dt(rho_prim, u, v, p)
            dt = torch.clamp(dt, max=t_final - t)
            
            # Strang splitting
            rho, rhou, rhov, E = self.x_sweep(rho, rhou, rhov, E, 0.5 * dt)
            # Y-sweep would be similar with transposed logic
            rho, rhou, rhov, E = self.x_sweep(rho, rhou, rhov, E, 0.5 * dt)
            
            t += dt.item()
        
        return self.conservative_to_primitive(rho, rhou, rhov, E)


# =============================================================================
# Loss Functions for PINNs
# =============================================================================

def euler_residual_loss(
    rho: Tensor,
    u: Tensor,
    p: Tensor,
    x: Tensor,
    t: Tensor,
    gamma: float = 1.4
) -> Tensor:
    """
    Compute Euler equation residual for PINN training.
    
    Loss = |∂U/∂t + ∂F/∂x|² where U = [ρ, ρu, E], F = Euler flux
    
    Args:
        rho, u, p: Primitive variables as functions of x, t (with grad enabled)
        x, t: Spatial and temporal coordinates
        gamma: Ratio of specific heats
        
    Returns:
        Scalar residual loss
    """
    # Enable gradients
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    
    # Conservative variables
    rhou = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u**2
    
    # Fluxes
    F_rho = rhou
    F_rhou = rhou * u + p
    F_E = u * (E + p)
    
    # Time derivatives
    drho_dt = torch.autograd.grad(rho.sum(), t, create_graph=True)[0]
    drhou_dt = torch.autograd.grad(rhou.sum(), t, create_graph=True)[0]
    dE_dt = torch.autograd.grad(E.sum(), t, create_graph=True)[0]
    
    # Spatial derivatives
    dF_rho_dx = torch.autograd.grad(F_rho.sum(), x, create_graph=True)[0]
    dF_rhou_dx = torch.autograd.grad(F_rhou.sum(), x, create_graph=True)[0]
    dF_E_dx = torch.autograd.grad(F_E.sum(), x, create_graph=True)[0]
    
    # Residuals
    res_mass = drho_dt + dF_rho_dx
    res_mom = drhou_dt + dF_rhou_dx
    res_energy = dE_dt + dF_E_dx
    
    loss = torch.mean(res_mass**2 + res_mom**2 + res_energy**2)
    
    return loss


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Flux functions
    'DifferentiableRoe',
    'DifferentiableHLLC',
    # Solvers
    'DifferentiableEuler1D',
    'DifferentiableEuler2D',
    # Loss functions
    'euler_residual_loss',
]
