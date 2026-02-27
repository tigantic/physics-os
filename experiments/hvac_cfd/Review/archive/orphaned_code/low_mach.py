"""
TigantiCFD Low-Mach Solver
==========================

Low-Mach number compressible flow solver for stratified flows.

Capabilities:
- T3.01: Variable density buoyancy-driven flows
- T3.02: Acoustic filtering (pressure splitting)
- T3.03: Temperature-dependent properties
- T3.04: Stratification modeling

Reference:
    Majda, A. & Sethian, J. (1985). "The derivation and numerical 
    solution of the equations for zero Mach number combustion."
    Combustion Science and Technology, 42(3-4), 185-205.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import numpy as np


@dataclass
class LowMachConfig:
    """Configuration for low-Mach solver."""
    # Domain
    nx: int = 32
    ny: int = 32
    nz: int = 32
    Lx: float = 4.0
    Ly: float = 3.0
    Lz: float = 4.0
    
    # Reference state
    T_ref: float = 293.15     # Reference temperature [K]
    p_ref: float = 101325.0   # Reference pressure [Pa]
    rho_ref: float = 1.2      # Reference density [kg/m³]
    
    # Properties
    R_gas: float = 287.0      # Specific gas constant [J/(kg·K)]
    cp: float = 1005.0        # Specific heat [J/(kg·K)]
    Pr: float = 0.71          # Prandtl number
    
    # Gravity
    g: float = 9.81           # Gravitational acceleration [m/s²]
    
    # Solver parameters
    dt: float = 0.001
    tolerance: float = 1e-6
    max_pressure_iters: int = 50


class LowMachSolver:
    """
    Low-Mach number compressible flow solver.
    
    Uses pressure splitting:
        p(x,t) = p₀(t) + p₂(x,t)
    
    where p₀ is the thermodynamic pressure (spatially uniform)
    and p₂ is the hydrodynamic pressure.
    
    The density follows the equation of state:
        ρ = p₀ / (R T)
    
    This filters acoustic waves while retaining buoyancy effects.
    """
    
    def __init__(
        self,
        config: LowMachConfig,
        device: str = "cpu"
    ):
        self.config = config
        self.device = device
        
        # Grid spacing
        self.dx = config.Lx / config.nx
        self.dy = config.Ly / config.ny
        self.dz = config.Lz / config.nz
        
        # Viscosity
        self.mu = 1.81e-5  # Dynamic viscosity [Pa·s]
        self.alpha = self.mu / (config.rho_ref * config.Pr)  # Thermal diffusivity
        
        # Initialize fields
        self._init_fields()
        
        # Tracking
        self.time = 0.0
        self.step_count = 0
    
    def _init_fields(self) -> None:
        """Initialize velocity, temperature, and pressure fields."""
        cfg = self.config
        shape = (cfg.nx, cfg.ny, cfg.nz)
        
        # Velocity
        self.u = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.v = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.w = torch.zeros(shape, device=self.device, dtype=torch.float32)
        
        # Temperature (uniform initially)
        self.T = torch.full(shape, cfg.T_ref, device=self.device, dtype=torch.float32)
        
        # Thermodynamic pressure (uniform in space)
        self.p0 = cfg.p_ref
        
        # Hydrodynamic pressure perturbation
        self.p2 = torch.zeros(shape, device=self.device, dtype=torch.float32)
        
        # Density (from equation of state)
        self.rho = self._compute_density()
        
        # Fluid mask
        self.fluid_mask = torch.ones(shape, device=self.device, dtype=torch.float32)
    
    def _compute_density(self) -> torch.Tensor:
        """Compute density from equation of state: ρ = p₀ / (R·T)."""
        return self.p0 / (self.config.R_gas * self.T)
    
    def set_temperature_bc(
        self,
        T_floor: Optional[float] = None,
        T_ceiling: Optional[float] = None,
        T_walls: Optional[float] = None
    ) -> None:
        """Set temperature boundary conditions."""
        if T_floor is not None:
            self.T[:, 0, :] = T_floor
        if T_ceiling is not None:
            self.T[:, -1, :] = T_ceiling
        if T_walls is not None:
            self.T[0, :, :] = T_walls
            self.T[-1, :, :] = T_walls
            self.T[:, :, 0] = T_walls
            self.T[:, :, -1] = T_walls
    
    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance solution one time step.
        
        Algorithm:
        1. Advect-diffuse temperature
        2. Update density from EOS
        3. Compute buoyancy force
        4. Advect-diffuse momentum
        5. Solve pressure Poisson equation
        6. Project velocity to divergence-free
        """
        if dt is None:
            dt = self.config.dt
        
        # Store old density for continuity
        rho_old = self.rho.clone()
        
        # 1. Temperature equation
        self._step_temperature(dt)
        
        # 2. Update density from equation of state
        self.rho = self._compute_density()
        
        # 3. Compute buoyancy force: f_b = (ρ - ρ_ref) g
        rho_ref = self.config.rho_ref
        buoyancy = -(self.rho - rho_ref) * self.config.g
        
        # 4. Momentum predictor step (explicit)
        self._step_momentum(dt, buoyancy, rho_old)
        
        # 5. Pressure correction
        self._solve_pressure(dt)
        
        # 6. Velocity correction
        self._correct_velocity(dt)
        
        # Update tracking
        self.time += dt
        self.step_count += 1
    
    def _step_temperature(self, dt: float) -> None:
        """Advect-diffuse temperature."""
        # Diffusion (Laplacian)
        lap_T = self._laplacian(self.T)
        
        # Advection
        adv_T = self._advection(self.T)
        
        # Update
        self.T = self.T + dt * (self.alpha * lap_T - adv_T)
        
        # Clamp to physical bounds
        self.T = torch.clamp(self.T, 263.15, 373.15)
    
    def _step_momentum(
        self, 
        dt: float, 
        buoyancy: torch.Tensor,
        rho_old: torch.Tensor
    ) -> None:
        """
        Momentum predictor step.
        
        ρ(u* - uⁿ)/dt = -∇p₂ + μ∇²u + ρg + advection
        """
        nu = self.mu / self.rho
        
        # Viscous diffusion
        lap_u = self._laplacian(self.u)
        lap_v = self._laplacian(self.v)
        lap_w = self._laplacian(self.w)
        
        # Advection
        adv_u = self._advection(self.u)
        adv_v = self._advection(self.v)
        adv_w = self._advection(self.w)
        
        # Pressure gradient (from previous step)
        dpx, dpy, dpz = self._gradient(self.p2)
        
        # Update velocity (predictor)
        nu_mean = float(nu.mean())
        
        self.u = self.u + dt * (
            nu_mean * lap_u - adv_u - dpx / self.rho
        )
        
        self.v = self.v + dt * (
            nu_mean * lap_v - adv_v - dpy / self.rho + buoyancy / self.rho
        )
        
        self.w = self.w + dt * (
            nu_mean * lap_w - adv_w - dpz / self.rho
        )
        
        # Apply mask
        self.u = self.u * self.fluid_mask
        self.v = self.v * self.fluid_mask
        self.w = self.w * self.fluid_mask
    
    def _solve_pressure(self, dt: float) -> None:
        """
        Solve pressure Poisson equation for low-Mach flow.
        
        For variable density:
        ∇·(1/ρ ∇p₂) = (1/dt) [∇·u* + (1/ρ)(∂ρ/∂t)]
        """
        # Divergence of predicted velocity
        div = self._divergence()
        
        # Density time derivative (approximation)
        drho_dt = (self.rho - self.config.rho_ref) / dt
        
        # RHS
        rhs = (div + drho_dt / self.rho) / dt
        
        # Jacobi iteration for pressure
        p = self.p2.clone()
        
        idx2 = 1.0 / (self.dx * self.dx)
        idy2 = 1.0 / (self.dy * self.dy)
        idz2 = 1.0 / (self.dz * self.dz)
        
        for _ in range(self.config.max_pressure_iters):
            p_new = (
                idx2 * (torch.roll(p, -1, 0) + torch.roll(p, 1, 0)) +
                idy2 * (torch.roll(p, -1, 1) + torch.roll(p, 1, 1)) +
                idz2 * (torch.roll(p, -1, 2) + torch.roll(p, 1, 2)) -
                rhs
            ) / (2 * (idx2 + idy2 + idz2))
            
            p_new = p_new * self.fluid_mask
            
            # Check convergence
            residual = float(torch.max(torch.abs(p_new - p)))
            p = p_new
            
            if residual < self.config.tolerance:
                break
        
        self.p2 = p
    
    def _correct_velocity(self, dt: float) -> None:
        """Correct velocity to be divergence-free."""
        dpx, dpy, dpz = self._gradient(self.p2)
        
        self.u = self.u - dt * dpx / self.rho
        self.v = self.v - dt * dpy / self.rho
        self.w = self.w - dt * dpz / self.rho
        
        # Apply mask and BCs
        self.u = self.u * self.fluid_mask
        self.v = self.v * self.fluid_mask
        self.w = self.w * self.fluid_mask
        
        # Wall BCs
        self.u[0, :, :] = 0
        self.u[-1, :, :] = 0
        self.v[:, 0, :] = 0
        self.v[:, -1, :] = 0
        self.w[:, :, 0] = 0
        self.w[:, :, -1] = 0
    
    def _laplacian(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using central differences."""
        lap = (
            (torch.roll(phi, -1, 0) - 2*phi + torch.roll(phi, 1, 0)) / self.dx**2 +
            (torch.roll(phi, -1, 1) - 2*phi + torch.roll(phi, 1, 1)) / self.dy**2 +
            (torch.roll(phi, -1, 2) - 2*phi + torch.roll(phi, 1, 2)) / self.dz**2
        )
        return lap
    
    def _advection(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute advection term u·∇φ."""
        dphidx = (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / (2 * self.dx)
        dphidy = (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / (2 * self.dy)
        dphidz = (torch.roll(phi, -1, 2) - torch.roll(phi, 1, 2)) / (2 * self.dz)
        
        return self.u * dphidx + self.v * dphidy + self.w * dphidz
    
    def _gradient(self, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gradient ∇φ."""
        dphidx = (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / (2 * self.dx)
        dphidy = (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / (2 * self.dy)
        dphidz = (torch.roll(phi, -1, 2) - torch.roll(phi, 1, 2)) / (2 * self.dz)
        
        return dphidx, dphidy, dphidz
    
    def _divergence(self) -> torch.Tensor:
        """Compute divergence ∇·u."""
        dudx = (torch.roll(self.u, -1, 0) - torch.roll(self.u, 1, 0)) / (2 * self.dx)
        dvdy = (torch.roll(self.v, -1, 1) - torch.roll(self.v, 1, 1)) / (2 * self.dy)
        dwdz = (torch.roll(self.w, -1, 2) - torch.roll(self.w, 1, 2)) / (2 * self.dz)
        
        return dudx + dvdy + dwdz
    
    def get_stratification(self) -> np.ndarray:
        """
        Compute vertical temperature profile (stratification).
        
        Returns mean temperature at each vertical level.
        """
        T_np = self.T.detach().cpu().numpy()
        return np.mean(T_np, axis=(0, 2))
    
    def get_richardson_number(self) -> torch.Tensor:
        """
        Compute gradient Richardson number for stratification stability.
        
        Ri = (g/T_ref)(∂T/∂y) / (∂u/∂y)²
        
        Ri > 0.25: stable stratification
        Ri < 0: unstable (convective)
        """
        g = self.config.g
        T_ref = self.config.T_ref
        
        # Temperature gradient
        dTdy = (torch.roll(self.T, -1, 1) - torch.roll(self.T, 1, 1)) / (2 * self.dy)
        
        # Velocity shear
        dudy = (torch.roll(self.u, -1, 1) - torch.roll(self.u, 1, 1)) / (2 * self.dy)
        dwdy = (torch.roll(self.w, -1, 1) - torch.roll(self.w, 1, 1)) / (2 * self.dy)
        
        shear_sq = dudy**2 + dwdy**2 + 1e-10
        
        Ri = (g / T_ref) * dTdy / shear_sq
        
        return Ri
    
    def get_metrics(self) -> dict:
        """Get solver metrics for monitoring."""
        T_np = self.T.detach().cpu().numpy()
        rho_np = self.rho.detach().cpu().numpy()
        
        u_mag = torch.sqrt(self.u**2 + self.v**2 + self.w**2)
        
        return {
            "time": self.time,
            "step": self.step_count,
            "T_min": float(T_np.min()),
            "T_max": float(T_np.max()),
            "T_mean": float(T_np.mean()),
            "rho_min": float(rho_np.min()),
            "rho_max": float(rho_np.max()),
            "u_max": float(u_mag.max()),
            "divergence_max": float(torch.abs(self._divergence()).max()),
        }


def create_stratified_case(
    Lx: float = 4.0,
    Ly: float = 3.0,
    Lz: float = 4.0,
    T_floor: float = 303.15,  # 30°C
    T_ceiling: float = 288.15,  # 15°C
    resolution: int = 32
) -> LowMachSolver:
    """
    Create a stratified flow case (warm floor, cool ceiling).
    
    This is typical for atrium or data center scenarios.
    """
    config = LowMachConfig(
        nx=resolution,
        ny=resolution,
        nz=resolution,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
    )
    
    solver = LowMachSolver(config)
    
    # Set temperature BCs for stratification
    solver.set_temperature_bc(
        T_floor=T_floor,
        T_ceiling=T_ceiling
    )
    
    return solver
