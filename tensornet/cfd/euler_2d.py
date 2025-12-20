"""
Two-dimensional compressible Euler equations solver.

Implements finite volume method with dimensional splitting (Strang)
for solving the 2D Euler equations on structured grids.

The 2D Euler equations in conservative form:

    ∂U/∂t + ∂F/∂x + ∂G/∂y = 0

where:
    U = [ρ, ρu, ρv, E]ᵀ
    F = [ρu, ρu² + p, ρuv, u(E + p)]ᵀ
    G = [ρv, ρuv, ρv² + p, v(E + p)]ᵀ

References:
    [1] Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics", 3rd ed.
    [2] LeVeque, "Finite Volume Methods for Hyperbolic Problems", Cambridge (2002)
"""

from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional
from enum import Enum
import math

from .godunov import hllc_flux, roe_flux, hll_flux
from .limiters import MUSCL


# Default ratio of specific heats for diatomic gas (air)
gamma_default = 1.4


class BCType(Enum):
    """Boundary condition types."""
    PERIODIC = "periodic"
    OUTFLOW = "outflow"
    INFLOW = "inflow"
    REFLECTIVE = "reflective"
    SUPERSONIC_INFLOW = "supersonic_inflow"
    SUPERSONIC_OUTFLOW = "supersonic_outflow"


@dataclass
class Euler2DState:
    """
    State vector for 2D Euler equations.
    
    All arrays have shape (Ny, Nx) representing the 2D grid.
    
    Attributes:
        rho: Density field
        u: x-velocity field
        v: y-velocity field
        p: Pressure field
        gamma: Ratio of specific heats
    """
    rho: Tensor
    u: Tensor
    v: Tensor
    p: Tensor
    gamma: float = gamma_default
    
    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape (Ny, Nx)."""
        return self.rho.shape
    
    @property
    def E(self) -> Tensor:
        """Total energy per unit volume."""
        kinetic = 0.5 * self.rho * (self.u**2 + self.v**2)
        internal = self.p / (self.gamma - 1)
        return kinetic + internal
    
    @property
    def H(self) -> Tensor:
        """Total enthalpy per unit volume."""
        return self.E + self.p
    
    @property
    def a(self) -> Tensor:
        """Sound speed."""
        return torch.sqrt(self.gamma * self.p / self.rho)
    
    @property
    def M(self) -> Tensor:
        """Mach number (magnitude)."""
        velocity_mag = torch.sqrt(self.u**2 + self.v**2)
        return velocity_mag / self.a
    
    def to_conservative(self) -> Tensor:
        """
        Convert to conservative variables.
        
        Returns:
            Tensor of shape (4, Ny, Nx): [rho, rho*u, rho*v, E]
        """
        return torch.stack([
            self.rho,
            self.rho * self.u,
            self.rho * self.v,
            self.E
        ], dim=0)
    
    @classmethod
    def from_conservative(cls, U: Tensor, gamma: float = gamma_default) -> "Euler2DState":
        """
        Create state from conservative variables.
        
        Args:
            U: Conservative variables (4, Ny, Nx)
            gamma: Ratio of specific heats
            
        Returns:
            Euler2DState instance
        """
        rho = U[0]
        u = U[1] / rho
        v = U[2] / rho
        E = U[3]
        p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        return cls(rho=rho, u=u, v=v, p=p, gamma=gamma)
    
    def clone(self) -> "Euler2DState":
        """Create a deep copy."""
        return Euler2DState(
            rho=self.rho.clone(),
            u=self.u.clone(),
            v=self.v.clone(),
            p=self.p.clone(),
            gamma=self.gamma
        )


class Euler2D:
    """
    2D Euler equations solver using finite volume method.
    
    Uses dimensional splitting (Strang splitting) to extend 1D
    schemes to 2D with second-order accuracy in time.
    
    Strang splitting: L(Δt) = Lx(Δt/2) Ly(Δt) Lx(Δt/2)
    
    Attributes:
        Nx: Number of cells in x-direction
        Ny: Number of cells in y-direction
        x: Cell-center x coordinates
        y: Cell-center y coordinates
        dx: Grid spacing in x
        dy: Grid spacing in y
        gamma: Ratio of specific heats
        state: Current flow state
    """
    
    def __init__(
        self,
        Nx: int,
        Ny: int,
        Lx: float = 1.0,
        Ly: float = 1.0,
        gamma: float = gamma_default,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu"
    ):
        """
        Initialize 2D Euler solver.
        
        Args:
            Nx: Number of cells in x
            Ny: Number of cells in y
            Lx: Domain length in x
            Ly: Domain length in y
            gamma: Ratio of specific heats
            dtype: Tensor data type
            device: Compute device
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.gamma = gamma
        self.dtype = dtype
        self.device = torch.device(device)
        
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        
        # Cell-center coordinates
        self.x = torch.linspace(
            self.dx / 2, Lx - self.dx / 2, Nx,
            dtype=dtype, device=self.device
        )
        self.y = torch.linspace(
            self.dy / 2, Ly - self.dy / 2, Ny,
            dtype=dtype, device=self.device
        )
        
        # 2D meshgrid (indexing='ij' gives shape (Ny, Nx))
        self.Y, self.X = torch.meshgrid(self.y, self.x, indexing='ij')
        
        # State
        self.state: Optional[Euler2DState] = None
        
        # Boundary conditions (default: outflow)
        self.bc_left = BCType.OUTFLOW
        self.bc_right = BCType.OUTFLOW
        self.bc_bottom = BCType.OUTFLOW
        self.bc_top = BCType.OUTFLOW
        
        # Inflow state (for inflow BCs)
        self.inflow_state: Optional[Euler2DState] = None
        
        # Time tracking
        self.time = 0.0
        self.step_count = 0
        
        # Limiter
        self.limiter = MUSCL(limiter='minmod')
    
    def set_initial_condition(self, state: Euler2DState) -> None:
        """Set initial flow condition."""
        self.state = state.clone()
        self.time = 0.0
        self.step_count = 0
    
    def set_boundary_conditions(
        self,
        left: BCType = BCType.OUTFLOW,
        right: BCType = BCType.OUTFLOW,
        bottom: BCType = BCType.OUTFLOW,
        top: BCType = BCType.OUTFLOW,
        inflow_state: Optional[Euler2DState] = None
    ) -> None:
        """Configure boundary conditions."""
        self.bc_left = left
        self.bc_right = right
        self.bc_bottom = bottom
        self.bc_top = top
        self.inflow_state = inflow_state
    
    def compute_dt(self, cfl: float = 0.4) -> float:
        """
        Compute stable time step based on CFL condition.
        
        For 2D: dt ≤ CFL / (|u|/dx + |v|/dy + a*sqrt(1/dx² + 1/dy²))
        """
        if self.state is None:
            raise RuntimeError("State not initialized")
        
        a = self.state.a
        u_abs = torch.abs(self.state.u)
        v_abs = torch.abs(self.state.v)
        
        # Wave speed in each direction
        lambda_x = u_abs + a
        lambda_y = v_abs + a
        
        # Maximum signal speed
        max_speed = torch.max(lambda_x / self.dx + lambda_y / self.dy)
        
        dt = cfl / max_speed.item()
        return dt
    
    def _apply_bc_x(self, U: Tensor, n_ghost: int = 2) -> Tensor:
        """
        Apply boundary conditions in x-direction.
        
        Args:
            U: Conservative variables (4, Ny, Nx)
            n_ghost: Number of ghost cells
            
        Returns:
            Padded array (4, Ny, Nx + 2*n_ghost)
        """
        # Allocate with ghost cells
        _, Ny, Nx = U.shape
        U_ext = torch.zeros(4, Ny, Nx + 2*n_ghost, dtype=self.dtype, device=self.device)
        U_ext[:, :, n_ghost:Nx+n_ghost] = U
        
        # Left BC
        if self.bc_left == BCType.OUTFLOW:
            for i in range(n_ghost):
                U_ext[:, :, i] = U[:, :, 0]
        elif self.bc_left == BCType.REFLECTIVE:
            for i in range(n_ghost):
                U_ext[:, :, n_ghost-1-i] = U[:, :, i]
                U_ext[1, :, n_ghost-1-i] *= -1  # Reflect u
        elif self.bc_left == BCType.INFLOW and self.inflow_state is not None:
            inflow_U = self.inflow_state.to_conservative()
            for i in range(n_ghost):
                U_ext[:, :, i] = inflow_U[:, :, 0:1].expand(-1, Ny, 1).squeeze(-1)
        elif self.bc_left == BCType.PERIODIC:
            U_ext[:, :, :n_ghost] = U[:, :, -n_ghost:]
        
        # Right BC
        if self.bc_right == BCType.OUTFLOW:
            for i in range(n_ghost):
                U_ext[:, :, Nx+n_ghost+i] = U[:, :, -1]
        elif self.bc_right == BCType.REFLECTIVE:
            for i in range(n_ghost):
                U_ext[:, :, Nx+n_ghost+i] = U[:, :, -(i+1)]
                U_ext[1, :, Nx+n_ghost+i] *= -1  # Reflect u
        elif self.bc_right == BCType.PERIODIC:
            U_ext[:, :, -n_ghost:] = U[:, :, :n_ghost]
        
        return U_ext
    
    def _apply_bc_y(self, U: Tensor, n_ghost: int = 2) -> Tensor:
        """
        Apply boundary conditions in y-direction.
        
        Args:
            U: Conservative variables (4, Ny, Nx)
            n_ghost: Number of ghost cells
            
        Returns:
            Padded array (4, Ny + 2*n_ghost, Nx)
        """
        _, Ny, Nx = U.shape
        U_ext = torch.zeros(4, Ny + 2*n_ghost, Nx, dtype=self.dtype, device=self.device)
        U_ext[:, n_ghost:Ny+n_ghost, :] = U
        
        # Bottom BC
        if self.bc_bottom == BCType.OUTFLOW:
            for i in range(n_ghost):
                U_ext[:, i, :] = U[:, 0, :]
        elif self.bc_bottom == BCType.REFLECTIVE:
            for i in range(n_ghost):
                U_ext[:, n_ghost-1-i, :] = U[:, i, :]
                U_ext[2, n_ghost-1-i, :] *= -1  # Reflect v
        elif self.bc_bottom == BCType.PERIODIC:
            U_ext[:, :n_ghost, :] = U[:, -n_ghost:, :]
        
        # Top BC
        if self.bc_top == BCType.OUTFLOW:
            for i in range(n_ghost):
                U_ext[:, Ny+n_ghost+i, :] = U[:, -1, :]
        elif self.bc_top == BCType.REFLECTIVE:
            for i in range(n_ghost):
                U_ext[:, Ny+n_ghost+i, :] = U[:, -(i+1), :]
                U_ext[2, Ny+n_ghost+i, :] *= -1  # Reflect v
        elif self.bc_top == BCType.PERIODIC:
            U_ext[:, -n_ghost:, :] = U[:, :n_ghost, :]
        
        return U_ext
    
    def _compute_x_flux(self, U: Tensor, flux_fn: str = "hllc") -> Tensor:
        """
        Compute numerical flux in x-direction at cell interfaces.
        
        Args:
            U: Conservative variables (4, Ny, Nx+2*ng) with ghost cells
            flux_fn: Flux function ("hllc", "roe", "hll")
            
        Returns:
            Flux at interfaces (4, Ny, Nx+1)
        """
        ng = 2  # ghost cells
        _, Ny, Nx_ext = U.shape
        Nx = Nx_ext - 2*ng
        
        # Extract primitives
        rho = U[0]
        u = U[1] / rho
        v = U[2] / rho
        E = U[3]
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        
        # Fluxes at Nx+1 interfaces
        F = torch.zeros(4, Ny, Nx + 1, dtype=self.dtype, device=self.device)
        
        # Loop over interfaces
        for j in range(Ny):
            for i in range(Nx + 1):
                # Left and right states at interface i+1/2
                iL = i + ng - 1
                iR = i + ng
                
                rhoL, uL, vL, pL = rho[j, iL], u[j, iL], v[j, iL], p[j, iL]
                rhoR, uR, vR, pR = rho[j, iR], u[j, iR], v[j, iR], p[j, iR]
                
                # MUSCL reconstruction for higher order
                if i > 0 and i < Nx:
                    # Reconstruct at interface
                    pass  # First-order for now
                
                # 1D Riemann problem in x-direction
                # Package as 1D states: rho, u, p (v is passive)
                if flux_fn == "hllc":
                    F_1d = self._hllc_2d(rhoL, uL, vL, pL, rhoR, uR, vR, pR)
                else:
                    F_1d = self._hllc_2d(rhoL, uL, vL, pL, rhoR, uR, vR, pR)
                
                F[:, j, i] = F_1d
        
        return F
    
    def _hllc_2d(
        self,
        rhoL: Tensor, uL: Tensor, vL: Tensor, pL: Tensor,
        rhoR: Tensor, uR: Tensor, vR: Tensor, pR: Tensor
    ) -> Tensor:
        """
        HLLC flux for 2D Euler (x-direction).
        
        The transverse velocity v is advected passively.
        """
        gamma = self.gamma
        
        # Sound speeds
        aL = torch.sqrt(gamma * pL / rhoL)
        aR = torch.sqrt(gamma * pR / rhoR)
        
        # Roe averages for wave speed estimates
        sqrt_rhoL = torch.sqrt(rhoL)
        sqrt_rhoR = torch.sqrt(rhoR)
        u_roe = (sqrt_rhoL * uL + sqrt_rhoR * uR) / (sqrt_rhoL + sqrt_rhoR)
        HL = (gamma * pL / (gamma - 1) + 0.5 * rhoL * (uL**2 + vL**2)) / rhoL
        HR = (gamma * pR / (gamma - 1) + 0.5 * rhoR * (uR**2 + vR**2)) / rhoR
        H_roe = (sqrt_rhoL * HL + sqrt_rhoR * HR) / (sqrt_rhoL + sqrt_rhoR)
        v_roe = (sqrt_rhoL * vL + sqrt_rhoR * vR) / (sqrt_rhoL + sqrt_rhoR)
        a_roe = torch.sqrt((gamma - 1) * (H_roe - 0.5 * (u_roe**2 + v_roe**2)))
        
        # Wave speed estimates (Davis)
        SL = torch.minimum(uL - aL, u_roe - a_roe)
        SR = torch.maximum(uR + aR, u_roe + a_roe)
        
        # Contact wave speed
        pstar_num = pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)
        pstar_den = rhoL * (SL - uL) - rhoR * (SR - uR)
        SM = pstar_num / pstar_den
        
        # Star region pressure
        pstar = pL + rhoL * (SL - uL) * (SM - uL)
        
        # Conservative variables and fluxes
        EL = pL / (gamma - 1) + 0.5 * rhoL * (uL**2 + vL**2)
        ER = pR / (gamma - 1) + 0.5 * rhoR * (uR**2 + vR**2)
        
        # Physical flux
        FL = torch.stack([
            rhoL * uL,
            rhoL * uL**2 + pL,
            rhoL * uL * vL,
            uL * (EL + pL)
        ])
        FR = torch.stack([
            rhoR * uR,
            rhoR * uR**2 + pR,
            rhoR * uR * vR,
            uR * (ER + pR)
        ])
        
        # Star states
        if SL >= 0:
            return FL
        elif SR <= 0:
            return FR
        elif SM >= 0:
            # Left star state
            rhoL_star = rhoL * (SL - uL) / (SL - SM)
            UL = torch.stack([rhoL, rhoL * uL, rhoL * vL, EL])
            UL_star = torch.stack([
                rhoL_star,
                rhoL_star * SM,
                rhoL_star * vL,
                rhoL_star * (EL / rhoL + (SM - uL) * (SM + pL / (rhoL * (SL - uL))))
            ])
            return FL + SL * (UL_star - UL)
        else:
            # Right star state
            rhoR_star = rhoR * (SR - uR) / (SR - SM)
            UR = torch.stack([rhoR, rhoR * uR, rhoR * vR, ER])
            UR_star = torch.stack([
                rhoR_star,
                rhoR_star * SM,
                rhoR_star * vR,
                rhoR_star * (ER / rhoR + (SM - uR) * (SM + pR / (rhoR * (SR - uR))))
            ])
            return FR + SR * (UR_star - UR)
    
    def _compute_y_flux(self, U: Tensor, flux_fn: str = "hllc") -> Tensor:
        """
        Compute numerical flux in y-direction at cell interfaces.
        
        Uses rotation: swap u<->v, then apply x-flux, then rotate back.
        """
        ng = 2
        _, Ny_ext, Nx = U.shape
        Ny = Ny_ext - 2*ng
        
        # Extract primitives
        rho = U[0]
        u = U[1] / rho
        v = U[2] / rho
        E = U[3]
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        
        # Fluxes at Ny+1 interfaces
        G = torch.zeros(4, Ny + 1, Nx, dtype=self.dtype, device=self.device)
        
        for i in range(Nx):
            for j in range(Ny + 1):
                jL = j + ng - 1
                jR = j + ng
                
                # Bottom and top states (swap u<->v for y-flux)
                rhoB, uB, vB, pB = rho[jL, i], v[jL, i], u[jL, i], p[jL, i]
                rhoT, uT, vT, pT = rho[jR, i], v[jR, i], u[jR, i], p[jR, i]
                
                # Compute flux as if x-direction
                F_rot = self._hllc_2d(rhoB, uB, vB, pB, rhoT, uT, vT, pT)
                
                # Rotate back: swap momentum components
                G[0, j, i] = F_rot[0]
                G[1, j, i] = F_rot[2]  # y-flux of x-momentum
                G[2, j, i] = F_rot[1]  # y-flux of y-momentum
                G[3, j, i] = F_rot[3]
        
        return G
    
    def _x_sweep(self, U: Tensor, dt: float) -> Tensor:
        """
        Perform x-direction sweep (finite volume update).
        
        Args:
            U: Conservative variables (4, Ny, Nx)
            dt: Time step
            
        Returns:
            Updated conservative variables
        """
        # Apply BCs
        U_ext = self._apply_bc_x(U)
        
        # Compute fluxes
        F = self._compute_x_flux(U_ext)
        
        # Conservative update
        U_new = U - (dt / self.dx) * (F[:, :, 1:] - F[:, :, :-1])
        
        return U_new
    
    def _y_sweep(self, U: Tensor, dt: float) -> Tensor:
        """
        Perform y-direction sweep (finite volume update).
        
        Args:
            U: Conservative variables (4, Ny, Nx)
            dt: Time step
            
        Returns:
            Updated conservative variables
        """
        # Apply BCs
        U_ext = self._apply_bc_y(U)
        
        # Compute fluxes
        G = self._compute_y_flux(U_ext)
        
        # Conservative update
        U_new = U - (dt / self.dy) * (G[:, 1:, :] - G[:, :-1, :])
        
        return U_new
    
    def step(self, dt: Optional[float] = None, cfl: float = 0.4) -> float:
        """
        Advance solution by one time step using Strang splitting.
        
        Strang splitting achieves second-order accuracy:
            U^{n+1} = Lx(dt/2) Ly(dt) Lx(dt/2) U^n
        
        Args:
            dt: Time step (computed from CFL if None)
            cfl: CFL number for automatic dt
            
        Returns:
            Actual time step used
        """
        if self.state is None:
            raise RuntimeError("State not initialized")
        
        if dt is None:
            dt = self.compute_dt(cfl)
        
        U = self.state.to_conservative()
        
        # Strang splitting
        U = self._x_sweep(U, dt / 2)
        U = self._y_sweep(U, dt)
        U = self._x_sweep(U, dt / 2)
        
        # Update state
        self.state = Euler2DState.from_conservative(U, self.gamma)
        self.time += dt
        self.step_count += 1
        
        return dt
    
    def solve(
        self,
        t_final: float,
        cfl: float = 0.4,
        max_steps: int = 100000,
        verbose: bool = False
    ) -> dict:
        """
        Integrate to final time.
        
        Args:
            t_final: Target simulation time
            cfl: CFL number
            max_steps: Maximum number of steps
            verbose: Print progress
            
        Returns:
            Dictionary with simulation info
        """
        if self.state is None:
            raise RuntimeError("State not initialized")
        
        steps = 0
        while self.time < t_final and steps < max_steps:
            dt = self.compute_dt(cfl)
            dt = min(dt, t_final - self.time)
            self.step(dt)
            steps += 1
            
            if verbose and steps % 100 == 0:
                print(f"Step {steps}: t = {self.time:.6f}, dt = {dt:.6e}")
        
        return {
            'time': self.time,
            'steps': steps,
            'final_dt': dt
        }


def supersonic_wedge_ic(
    Nx: int,
    Ny: int,
    M_inf: float = 5.0,
    p_inf: float = 1.0,
    rho_inf: float = 1.0,
    gamma: float = gamma_default,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu"
) -> Euler2DState:
    """
    Create initial condition for supersonic flow (uniform freestream).
    
    Args:
        Nx, Ny: Grid dimensions
        M_inf: Freestream Mach number
        p_inf: Freestream pressure
        rho_inf: Freestream density
        gamma: Ratio of specific heats
        
    Returns:
        Euler2DState with uniform supersonic flow in +x direction
    """
    a_inf = torch.sqrt(torch.tensor(gamma * p_inf / rho_inf, dtype=dtype))
    u_inf = M_inf * a_inf
    v_inf = torch.tensor(0.0, dtype=dtype)
    
    rho = torch.full((Ny, Nx), rho_inf, dtype=dtype, device=device)
    u = torch.full((Ny, Nx), u_inf.item(), dtype=dtype, device=device)
    v = torch.full((Ny, Nx), v_inf.item(), dtype=dtype, device=device)
    p = torch.full((Ny, Nx), p_inf, dtype=dtype, device=device)
    
    return Euler2DState(rho=rho, u=u, v=v, p=p, gamma=gamma)


def oblique_shock_exact(
    M1: float,
    theta: float,
    gamma: float = gamma_default
) -> dict:
    """
    Compute exact oblique shock relations.
    
    Given upstream Mach M1 and deflection angle theta, computes
    shock angle beta and downstream properties using θ-β-M relation:
    
        tan(θ) = 2 cot(β) * (M1² sin²(β) - 1) / (M1² (γ + cos(2β)) + 2)
    
    Args:
        M1: Upstream Mach number
        theta: Flow deflection angle (radians)
        gamma: Ratio of specific heats
        
    Returns:
        Dictionary with shock properties:
            beta: Shock angle (radians)
            M2: Downstream Mach number
            p2_p1: Pressure ratio
            rho2_rho1: Density ratio
            T2_T1: Temperature ratio
    """
    import math
    
    # Solve for beta using Newton iteration on theta-beta-M relation
    def theta_beta_M(beta: float) -> float:
        """Theta as function of beta and M1."""
        if beta <= 0 or beta >= math.pi/2:
            return float('inf')
        sin_b = math.sin(beta)
        cos_b = math.cos(beta)
        num = 2 * cos_b / sin_b * (M1**2 * sin_b**2 - 1)
        den = M1**2 * (gamma + math.cos(2*beta)) + 2
        return math.atan(num / den)
    
    # Initial guess: weak shock solution
    beta = math.asin(1.0 / M1) + 0.1  # Slightly above Mach angle
    
    for _ in range(50):
        theta_calc = theta_beta_M(beta)
        if abs(theta_calc - theta) < 1e-10:
            break
        
        # Numerical derivative
        eps = 1e-8
        dtheta_dbeta = (theta_beta_M(beta + eps) - theta_calc) / eps
        if abs(dtheta_dbeta) < 1e-12:
            break
        
        beta = beta - (theta_calc - theta) / dtheta_dbeta
        beta = max(math.asin(1.0/M1), min(beta, math.pi/2 - 0.01))
    
    # Compute downstream properties
    sin_b = math.sin(beta)
    M1n = M1 * sin_b  # Normal Mach number
    
    # Normal shock relations for M1n
    p2_p1 = 1 + 2 * gamma / (gamma + 1) * (M1n**2 - 1)
    rho2_rho1 = (gamma + 1) * M1n**2 / ((gamma - 1) * M1n**2 + 2)
    T2_T1 = p2_p1 / rho2_rho1
    
    # Downstream normal Mach number
    M2n_sq = ((gamma - 1) * M1n**2 + 2) / (2 * gamma * M1n**2 - (gamma - 1))
    M2n = math.sqrt(M2n_sq)
    
    # Downstream Mach number
    M2 = M2n / math.sin(beta - theta)
    
    return {
        'beta': beta,
        'M2': M2,
        'p2_p1': p2_p1,
        'rho2_rho1': rho2_rho1,
        'T2_T1': T2_T1,
        'M1n': M1n,
        'theta': theta
    }


def double_mach_reflection_ic(
    Nx: int,
    Ny: int,
    x_shock: float = 1.0/6.0,
    gamma: float = gamma_default,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu"
) -> tuple[Euler2DState, float, float]:
    """
    Double Mach reflection initial condition (Woodward & Colella, 1984).
    
    A Mach 10 shock at 60° angle impacting a reflecting wall.
    
    Args:
        Nx, Ny: Grid dimensions
        x_shock: Initial x-position of shock foot
        gamma: Ratio of specific heats
        
    Returns:
        (state, Lx, Ly): Initial state and domain size
    """
    Lx = 4.0
    Ly = 1.0
    
    dx = Lx / Nx
    dy = Ly / Ny
    
    x = torch.linspace(dx/2, Lx - dx/2, Nx, dtype=dtype, device=device)
    y = torch.linspace(dy/2, Ly - dy/2, Ny, dtype=dtype, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Shock angle 60° from x-axis
    angle = math.pi / 3
    
    # Pre-shock state (right of shock)
    rho_R = 1.4
    p_R = 1.0
    u_R = 0.0
    v_R = 0.0
    
    # Post-shock state (left of shock) - Mach 10 moving to right
    M_shock = 10.0
    # Rankine-Hugoniot relations
    p_L = p_R * (2 * gamma * M_shock**2 - (gamma - 1)) / (gamma + 1)
    rho_L = rho_R * (gamma + 1) * M_shock**2 / ((gamma - 1) * M_shock**2 + 2)
    
    # Shock velocity
    a_R = math.sqrt(gamma * p_R / rho_R)
    W = M_shock * a_R
    
    # Post-shock velocity in lab frame
    u_L = W * (1 - rho_R / rho_L)
    v_L = 0.0
    
    # Rotate to account for shock angle
    import math
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    u_L_rot = u_L * cos_a
    v_L_rot = u_L * sin_a
    
    # Shock position: x - x_shock = y * tan(30°)
    shock_pos = x_shock + Y / math.tan(angle)
    
    # Initialize arrays
    rho = torch.where(X < shock_pos, rho_L, rho_R)
    u = torch.where(X < shock_pos, u_L_rot, u_R)
    v = torch.where(X < shock_pos, v_L_rot, v_R)
    p = torch.where(X < shock_pos, p_L, p_R)
    
    return Euler2DState(rho=rho, u=u, v=v, p=p, gamma=gamma), Lx, Ly
