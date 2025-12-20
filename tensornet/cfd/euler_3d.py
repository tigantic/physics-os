"""
3D Euler Equations Solver
==========================

Extends the 2D Euler solver to three dimensions using
directional splitting (Strang splitting in x-y-z).

The 3D Euler equations in conservative form:

    ∂U/∂t + ∂F/∂x + ∂G/∂y + ∂H/∂z = 0

where:
    U = [ρ, ρu, ρv, ρw, E]ᵀ

    F = [ρu, ρu² + p, ρuv, ρuw, (E+p)u]ᵀ
    G = [ρv, ρuv, ρv² + p, ρvw, (E+p)v]ᵀ
    H = [ρw, ρuw, ρvw, ρw² + p, (E+p)w]ᵀ

Strang Splitting (3D):
    L(Δt) = Lx(Δt/2) Ly(Δt/2) Lz(Δt) Ly(Δt/2) Lx(Δt/2)

This achieves second-order temporal accuracy while allowing
each sweep to use the optimized 1D HLLC solver.

References:
    [1] Toro, "Riemann Solvers and Numerical Methods", Ch. 16
    [2] LeVeque, "Finite Volume Methods for Hyperbolic Problems"
"""

from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Callable
import math


GAMMA_DEFAULT = 1.4


@dataclass
class Euler3DState:
    """
    State container for 3D Euler equations.
    
    Stores primitive variables on a 3D grid.
    
    Attributes:
        rho: Density [Nz, Ny, Nx]
        u: x-velocity [Nz, Ny, Nx]
        v: y-velocity [Nz, Ny, Nx]
        w: z-velocity [Nz, Ny, Nx]
        p: Pressure [Nz, Ny, Nx]
        gamma: Ratio of specific heats
    """
    rho: Tensor
    u: Tensor
    v: Tensor
    w: Tensor
    p: Tensor
    gamma: float = GAMMA_DEFAULT
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid dimensions (Nz, Ny, Nx)."""
        return self.rho.shape
    
    @property
    def E(self) -> Tensor:
        """Total energy per unit volume."""
        ke = 0.5 * self.rho * (self.u**2 + self.v**2 + self.w**2)
        return self.p / (self.gamma - 1) + ke
    
    def sound_speed(self) -> Tensor:
        """Local speed of sound."""
        return torch.sqrt(self.gamma * self.p / self.rho)
    
    def mach_number(self) -> Tensor:
        """Local Mach number."""
        speed = torch.sqrt(self.u**2 + self.v**2 + self.w**2)
        return speed / self.sound_speed()
    
    def to_conservative(self) -> Tensor:
        """
        Convert to conservative variables.
        
        Returns:
            Tensor of shape [5, Nz, Ny, Nx]
        """
        Nz, Ny, Nx = self.shape
        U = torch.zeros(5, Nz, Ny, Nx, dtype=self.rho.dtype, device=self.rho.device)
        U[0] = self.rho
        U[1] = self.rho * self.u
        U[2] = self.rho * self.v
        U[3] = self.rho * self.w
        U[4] = self.E
        return U
    
    @classmethod
    def from_conservative(cls, U: Tensor, gamma: float = GAMMA_DEFAULT) -> Euler3DState:
        """
        Create state from conservative variables.
        
        Args:
            U: Conservative variables [5, Nz, Ny, Nx]
            gamma: Ratio of specific heats
            
        Returns:
            Euler3DState
        """
        rho = U[0]
        u = U[1] / rho
        v = U[2] / rho
        w = U[3] / rho
        E = U[4]
        ke = 0.5 * rho * (u**2 + v**2 + w**2)
        p = (gamma - 1) * (E - ke)
        return cls(rho=rho, u=u, v=v, w=w, p=p, gamma=gamma)
    
    def copy(self) -> Euler3DState:
        """Create a deep copy."""
        return Euler3DState(
            rho=self.rho.clone(),
            u=self.u.clone(),
            v=self.v.clone(),
            w=self.w.clone(),
            p=self.p.clone(),
            gamma=self.gamma
        )


def hllc_flux_3d(
    rhoL: Tensor, uL: Tensor, vL: Tensor, wL: Tensor, pL: Tensor,
    rhoR: Tensor, uR: Tensor, vR: Tensor, wR: Tensor, pR: Tensor,
    gamma: float,
    direction: str = 'x'
) -> Tensor:
    """
    HLLC approximate Riemann solver for 3D Euler equations.
    
    Computes flux in the specified direction at cell interfaces.
    
    Args:
        rhoL, uL, vL, wL, pL: Left state primitives
        rhoR, uR, vR, wR, pR: Right state primitives
        gamma: Ratio of specific heats
        direction: 'x', 'y', or 'z'
        
    Returns:
        Flux tensor [5, ...]
    """
    # Select normal velocity based on direction
    if direction == 'x':
        unL, unR = uL, uR
    elif direction == 'y':
        unL, unR = vL, vR
    else:  # z
        unL, unR = wL, wR
    
    # Sound speeds
    cL = torch.sqrt(gamma * pL / rhoL)
    cR = torch.sqrt(gamma * pR / rhoR)
    
    # Roe averages for wave speed estimates
    sqrtL = torch.sqrt(rhoL)
    sqrtR = torch.sqrt(rhoR)
    denom = sqrtL + sqrtR
    
    u_roe = (sqrtL * unL + sqrtR * unR) / denom
    H_L = (gamma * pL / (gamma - 1) + 0.5 * rhoL * (uL**2 + vL**2 + wL**2)) / rhoL
    H_R = (gamma * pR / (gamma - 1) + 0.5 * rhoR * (uR**2 + vR**2 + wR**2)) / rhoR
    H_roe = (sqrtL * H_L + sqrtR * H_R) / denom
    
    c_roe = torch.sqrt(torch.clamp((gamma - 1) * (H_roe - 0.5 * u_roe**2), min=1e-10))
    
    # Wave speed estimates
    SL = torch.minimum(unL - cL, u_roe - c_roe)
    SR = torch.maximum(unR + cR, u_roe + c_roe)
    
    # Contact wave speed
    num = pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR)
    den = rhoL * (SL - unL) - rhoR * (SR - unR)
    SM = num / (den + 1e-14)
    
    # Total energies
    EL = pL / (gamma - 1) + 0.5 * rhoL * (uL**2 + vL**2 + wL**2)
    ER = pR / (gamma - 1) + 0.5 * rhoR * (uR**2 + vR**2 + wR**2)
    
    # Build fluxes based on direction
    shape = rhoL.shape
    F = torch.zeros(5, *shape, dtype=rhoL.dtype, device=rhoL.device)
    
    # Left flux
    if direction == 'x':
        FL = torch.stack([
            rhoL * uL,
            rhoL * uL**2 + pL,
            rhoL * uL * vL,
            rhoL * uL * wL,
            (EL + pL) * uL
        ])
        # Left star state
        factor_L = rhoL * (SL - unL) / (SL - SM + 1e-14)
        rho_starL = factor_L
        u_starL = SM
        v_starL = vL
        w_starL = wL
    elif direction == 'y':
        FL = torch.stack([
            rhoL * vL,
            rhoL * vL * uL,
            rhoL * vL**2 + pL,
            rhoL * vL * wL,
            (EL + pL) * vL
        ])
        factor_L = rhoL * (SL - unL) / (SL - SM + 1e-14)
        rho_starL = factor_L
        u_starL = uL
        v_starL = SM
        w_starL = wL
    else:  # z
        FL = torch.stack([
            rhoL * wL,
            rhoL * wL * uL,
            rhoL * wL * vL,
            rhoL * wL**2 + pL,
            (EL + pL) * wL
        ])
        factor_L = rhoL * (SL - unL) / (SL - SM + 1e-14)
        rho_starL = factor_L
        u_starL = uL
        v_starL = vL
        w_starL = SM
    
    # Right flux (similar structure)
    if direction == 'x':
        FR = torch.stack([
            rhoR * uR,
            rhoR * uR**2 + pR,
            rhoR * uR * vR,
            rhoR * uR * wR,
            (ER + pR) * uR
        ])
        factor_R = rhoR * (SR - unR) / (SR - SM + 1e-14)
        rho_starR = factor_R
        u_starR = SM
        v_starR = vR
        w_starR = wR
    elif direction == 'y':
        FR = torch.stack([
            rhoR * vR,
            rhoR * vR * uR,
            rhoR * vR**2 + pR,
            rhoR * vR * wR,
            (ER + pR) * vR
        ])
        factor_R = rhoR * (SR - unR) / (SR - SM + 1e-14)
        rho_starR = factor_R
        u_starR = uR
        v_starR = SM
        w_starR = wR
    else:  # z
        FR = torch.stack([
            rhoR * wR,
            rhoR * wR * uR,
            rhoR * wR * vR,
            rhoR * wR**2 + pR,
            (ER + pR) * wR
        ])
        factor_R = rhoR * (SR - unR) / (SR - SM + 1e-14)
        rho_starR = factor_R
        u_starR = uR
        v_starR = vR
        w_starR = SM
    
    # Star region pressure
    p_star = pL + rhoL * (SL - unL) * (SM - unL)
    
    # Star state conserved variables
    E_starL = (EL * (SL - unL) + p_star * SM - pL * unL) / (SL - SM + 1e-14)
    E_starR = (ER * (SR - unR) + p_star * SM - pR * unR) / (SR - SM + 1e-14)
    
    if direction == 'x':
        U_starL = torch.stack([rho_starL, rho_starL * u_starL, rho_starL * v_starL, rho_starL * w_starL, E_starL])
        U_starR = torch.stack([rho_starR, rho_starR * u_starR, rho_starR * v_starR, rho_starR * w_starR, E_starR])
    elif direction == 'y':
        U_starL = torch.stack([rho_starL, rho_starL * u_starL, rho_starL * v_starL, rho_starL * w_starL, E_starL])
        U_starR = torch.stack([rho_starR, rho_starR * u_starR, rho_starR * v_starR, rho_starR * w_starR, E_starR])
    else:
        U_starL = torch.stack([rho_starL, rho_starL * u_starL, rho_starL * v_starL, rho_starL * w_starL, E_starL])
        U_starR = torch.stack([rho_starR, rho_starR * u_starR, rho_starR * v_starR, rho_starR * w_starR, E_starR])
    
    # Star fluxes
    F_starL = FL + SL * (U_starL - torch.stack([rhoL, rhoL*uL, rhoL*vL, rhoL*wL, EL]))
    F_starR = FR + SR * (U_starR - torch.stack([rhoR, rhoR*uR, rhoR*vR, rhoR*wR, ER]))
    
    # HLLC flux selection
    F = torch.where(SL >= 0, FL, F)
    F = torch.where((SL < 0) & (SM >= 0), F_starL, F)
    F = torch.where((SM < 0) & (SR >= 0), F_starR, F)
    F = torch.where(SR < 0, FR, F)
    
    return F


class Euler3D:
    """
    3D Euler equations solver with Strang splitting.
    
    Uses directional splitting to reduce 3D problem to
    sequence of 1D Riemann problems.
    
    Example:
        >>> solver = Euler3D(Nx=64, Ny=64, Nz=64, Lx=1.0, Ly=1.0, Lz=1.0)
        >>> state = uniform_flow_3d(solver, M_inf=2.0)
        >>> result = solver.step(state, dt=1e-4)
    """
    
    def __init__(
        self,
        Nx: int,
        Ny: int,
        Nz: int,
        Lx: float,
        Ly: float,
        Lz: float,
        gamma: float = GAMMA_DEFAULT,
        cfl: float = 0.4
    ):
        """
        Initialize 3D Euler solver.
        
        Args:
            Nx, Ny, Nz: Grid points in each direction
            Lx, Ly, Lz: Domain size in each direction
            gamma: Ratio of specific heats
            cfl: CFL number
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.gamma = gamma
        self.cfl = cfl
        
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        self.dz = Lz / (Nz - 1)
    
    def compute_timestep(self, state: Euler3DState) -> float:
        """
        Compute CFL-limited timestep.
        
        Args:
            state: Current state
            
        Returns:
            Stable timestep
        """
        c = state.sound_speed()
        speed = torch.sqrt(state.u**2 + state.v**2 + state.w**2) + c
        dmin = min(self.dx, self.dy, self.dz)
        return self.cfl * dmin / speed.max().item()
    
    def _sweep_x(self, state: Euler3DState, dt: float) -> Euler3DState:
        """X-direction sweep."""
        U = state.to_conservative()
        
        # For each (k, j) line, solve 1D Riemann problem
        for k in range(self.Nz):
            for j in range(self.Ny):
                # Extract 1D line
                rho_line = state.rho[k, j, :]
                u_line = state.u[k, j, :]
                v_line = state.v[k, j, :]
                w_line = state.w[k, j, :]
                p_line = state.p[k, j, :]
                
                # Compute fluxes at interfaces
                F = hllc_flux_3d(
                    rho_line[:-1], u_line[:-1], v_line[:-1], w_line[:-1], p_line[:-1],
                    rho_line[1:], u_line[1:], v_line[1:], w_line[1:], p_line[1:],
                    self.gamma, direction='x'
                )
                
                # Update conserved variables (interior)
                U[:, k, j, 1:-1] -= dt / self.dx * (F[:, 1:] - F[:, :-1])
        
        return Euler3DState.from_conservative(U, self.gamma)
    
    def _sweep_y(self, state: Euler3DState, dt: float) -> Euler3DState:
        """Y-direction sweep."""
        U = state.to_conservative()
        
        for k in range(self.Nz):
            for i in range(self.Nx):
                rho_line = state.rho[k, :, i]
                u_line = state.u[k, :, i]
                v_line = state.v[k, :, i]
                w_line = state.w[k, :, i]
                p_line = state.p[k, :, i]
                
                G = hllc_flux_3d(
                    rho_line[:-1], u_line[:-1], v_line[:-1], w_line[:-1], p_line[:-1],
                    rho_line[1:], u_line[1:], v_line[1:], w_line[1:], p_line[1:],
                    self.gamma, direction='y'
                )
                
                U[:, k, 1:-1, i] -= dt / self.dy * (G[:, 1:] - G[:, :-1])
        
        return Euler3DState.from_conservative(U, self.gamma)
    
    def _sweep_z(self, state: Euler3DState, dt: float) -> Euler3DState:
        """Z-direction sweep."""
        U = state.to_conservative()
        
        for j in range(self.Ny):
            for i in range(self.Nx):
                rho_line = state.rho[:, j, i]
                u_line = state.u[:, j, i]
                v_line = state.v[:, j, i]
                w_line = state.w[:, j, i]
                p_line = state.p[:, j, i]
                
                H = hllc_flux_3d(
                    rho_line[:-1], u_line[:-1], v_line[:-1], w_line[:-1], p_line[:-1],
                    rho_line[1:], u_line[1:], v_line[1:], w_line[1:], p_line[1:],
                    self.gamma, direction='z'
                )
                
                U[:, 1:-1, j, i] -= dt / self.dz * (H[:, 1:] - H[:, :-1])
        
        return Euler3DState.from_conservative(U, self.gamma)
    
    def step(self, state: Euler3DState, dt: float) -> Euler3DState:
        """
        Advance one timestep using Strang splitting.
        
        L(Δt) = Lx(Δt/2) Ly(Δt/2) Lz(Δt) Ly(Δt/2) Lx(Δt/2)
        
        Args:
            state: Current state
            dt: Timestep
            
        Returns:
            Updated state
        """
        state = self._sweep_x(state, dt / 2)
        state = self._sweep_y(state, dt / 2)
        state = self._sweep_z(state, dt)
        state = self._sweep_y(state, dt / 2)
        state = self._sweep_x(state, dt / 2)
        return state
    
    def solve(
        self,
        initial_state: Euler3DState,
        t_final: float,
        callback: Optional[Callable] = None,
        max_steps: int = 100000
    ) -> tuple[Euler3DState, float, int]:
        """
        Solve to final time.
        
        Args:
            initial_state: Initial condition
            t_final: Final time
            callback: Optional callback(state, t, step)
            max_steps: Maximum steps
            
        Returns:
            (final_state, final_time, num_steps)
        """
        state = initial_state
        t = 0.0
        step = 0
        
        while t < t_final and step < max_steps:
            dt = self.compute_timestep(state)
            dt = min(dt, t_final - t)
            
            state = self.step(state, dt)
            
            t += dt
            step += 1
            
            if callback and callback(state, t, step):
                break
        
        return state, t, step


def uniform_flow_3d(
    solver: Euler3D,
    M_inf: float = 2.0,
    rho_inf: float = 1.225,
    p_inf: float = 101325.0,
    flow_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
) -> Euler3DState:
    """
    Create uniform supersonic flow initial condition.
    
    Args:
        solver: Euler3D solver instance
        M_inf: Freestream Mach number
        rho_inf: Freestream density
        p_inf: Freestream pressure
        flow_direction: Normalized flow direction (dx, dy, dz)
        
    Returns:
        Euler3DState with uniform flow
    """
    c_inf = math.sqrt(solver.gamma * p_inf / rho_inf)
    speed = M_inf * c_inf
    
    # Normalize direction
    dx, dy, dz = flow_direction
    mag = math.sqrt(dx**2 + dy**2 + dz**2)
    dx, dy, dz = dx/mag, dy/mag, dz/mag
    
    rho = torch.ones(solver.Nz, solver.Ny, solver.Nx, dtype=torch.float64) * rho_inf
    u = torch.ones_like(rho) * speed * dx
    v = torch.ones_like(rho) * speed * dy
    w = torch.ones_like(rho) * speed * dz
    p = torch.ones_like(rho) * p_inf
    
    return Euler3DState(rho=rho, u=u, v=v, w=w, p=p, gamma=solver.gamma)


def sod_3d_ic(solver: Euler3D, split_fraction: float = 0.5) -> Euler3DState:
    """
    3D Sod shock tube initial condition (split in x-direction).
    
    Args:
        solver: Euler3D solver instance
        split_fraction: Location of initial discontinuity
        
    Returns:
        Euler3DState with Sod IC
    """
    rho = torch.ones(solver.Nz, solver.Ny, solver.Nx, dtype=torch.float64)
    u = torch.zeros_like(rho)
    v = torch.zeros_like(rho)
    w = torch.zeros_like(rho)
    p = torch.ones_like(rho)
    
    # High pressure/density on left
    split_i = int(solver.Nx * split_fraction)
    rho[:, :, :split_i] = 1.0
    p[:, :, :split_i] = 1.0
    
    # Low pressure/density on right
    rho[:, :, split_i:] = 0.125
    p[:, :, split_i:] = 0.1
    
    return Euler3DState(rho=rho, u=u, v=v, w=w, p=p, gamma=solver.gamma)
