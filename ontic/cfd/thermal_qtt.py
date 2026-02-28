"""
Thermal QTT Extension: Temperature Transport for CFD-HVAC
==========================================================

Advection-Diffusion equation for temperature:
    ∂T/∂t + (u·∇)T = α∇²T + Q/(ρ·Cp)

Where:
    T = temperature [K or °C]
    u = (u, v) velocity from NS solver
    α = thermal diffusivity of air ≈ 2.2e-5 m²/s
    Q = volumetric heat source [W/m³]
    ρ = air density ≈ 1.2 kg/m³
    Cp = specific heat ≈ 1005 J/(kg·K)

Coupled with NS2D_QTT_Native for velocity field.

Author: HyperTensor Team
Date: January 2026
"""

from dataclasses import dataclass
from typing import Optional

import torch

from ontic.cfd.ns2d_qtt_native import (
    NS2D_QTT_Native,
    NS2DQTTConfig,
    QTT2DNativeState,
    dense_to_qtt_2d_native,
    qtt_2d_native_to_dense,
)


@dataclass
class ThermalConfig:
    """Configuration for thermal transport solver."""
    
    # Air properties at ~20°C, 1 atm
    alpha: float = 2.2e-5       # Thermal diffusivity [m²/s]
    rho: float = 1.2            # Density [kg/m³]
    Cp: float = 1005.0          # Specific heat [J/(kg·K)]
    
    # Boundary conditions
    T_supply: float = 289.15    # Supply air temp [K] (16°C)
    T_ambient: float = 297.15   # Ambient/initial temp [K] (24°C)
    
    # Comfort calculation defaults
    met: float = 1.2            # Metabolic rate (seated office work)
    clo: float = 0.5            # Clothing insulation (summer indoor)
    humidity_RH: float = 50.0   # Relative humidity [%]
    
    @property
    def T_supply_C(self) -> float:
        return self.T_supply - 273.15
    
    @property
    def T_ambient_C(self) -> float:
        return self.T_ambient - 273.15


@dataclass
class HeatSource:
    """Defines a rectangular heat source region."""
    
    x_min: float  # [m]
    x_max: float
    y_min: float
    y_max: float
    power_density: float  # [W/m³] or total power / area for 2D
    
    def contains(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return mask of points inside this source."""
        return (
            (x >= self.x_min) & (x < self.x_max) &
            (y >= self.y_min) & (y < self.y_max)
        )


@dataclass
class Diffuser:
    """Ceiling diffuser specification."""
    
    x_center: float  # [m]
    width: float     # [m]
    velocity: float  # [m/s] downward
    T_supply: float  # [K]


@dataclass
class OpenOfficeConfig:
    """Configuration for Tier 2 open office scenario."""
    
    # Geometry
    Lx: float = 30.0   # [m] length
    Ly: float = 3.0    # [m] height (analyzing x-z cross-section)
    
    # Grid (power of 2)
    nx_bits: int = 10  # 1024 in x
    ny_bits: int = 9   # 512 in y (height)
    
    # HVAC
    n_diffusers: int = 5
    diffuser_width: float = 0.6   # [m]
    supply_velocity: float = 1.0   # [m/s]
    supply_temp_K: float = 289.15  # 16°C
    
    n_returns: int = 2
    return_height: float = 0.3    # [m] from floor
    
    # Heat sources
    n_occupants: int = 50
    occupant_power_W: float = 75.0
    n_computers: int = 50
    computer_power_W: float = 150.0
    
    # Solar
    solar_peak_W_m2: float = 200.0
    glazing_height: float = 2.0    # [m]
    
    # Physics
    nu: float = 1.5e-5   # Kinematic viscosity
    max_rank: int = 24
    
    @property
    def Nx(self) -> int:
        return 2 ** self.nx_bits
    
    @property
    def Ny(self) -> int:
        return 2 ** self.ny_bits
    
    @property
    def dx(self) -> float:
        return self.Lx / self.Nx
    
    @property
    def dy(self) -> float:
        return self.Ly / self.Ny
    
    @property
    def total_internal_load_W(self) -> float:
        return (self.n_occupants * self.occupant_power_W + 
                self.n_computers * self.computer_power_W)
    
    @property
    def solar_load_W_per_m(self) -> float:
        """Solar load per meter depth (2D assumption)."""
        return self.solar_peak_W_m2 * self.glazing_height


class ThermalQTTSolver:
    """
    Temperature transport solver in QTT format.
    
    Coupled with NS2D_QTT_Native for velocity field.
    Solves: ∂T/∂t + (u·∇)T = α∇²T + Q/(ρ·Cp)
    """
    
    def __init__(
        self, 
        ns_solver: NS2D_QTT_Native,
        thermal_config: ThermalConfig,
        heat_sources: Optional[list[HeatSource]] = None
    ):
        self.ns = ns_solver
        self.config = thermal_config
        self.heat_sources = heat_sources or []
        
        # Build heat source field (QTT)
        self._Q_field: Optional[QTT2DNativeState] = None
        
    def _build_heat_source_field(self) -> QTT2DNativeState:
        """Build volumetric heat source field Q [W/m³]."""
        cfg = self.ns.config
        
        # Create coordinate grids
        x = torch.linspace(0, cfg.Lx, cfg.Nx, device=cfg.device, dtype=cfg.dtype)
        y = torch.linspace(0, cfg.Ly, cfg.Ny, device=cfg.device, dtype=cfg.dtype)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        Q = torch.zeros_like(X)
        
        for source in self.heat_sources:
            mask = source.contains(X, Y)
            Q[mask] += source.power_density
        
        # Compress to QTT
        return dense_to_qtt_2d_native(
            Q, cfg.nx_bits, cfg.ny_bits, max_bond=cfg.max_rank
        )
    
    @property
    def Q_field(self) -> QTT2DNativeState:
        """Lazy-build heat source field."""
        if self._Q_field is None:
            self._Q_field = self._build_heat_source_field()
        return self._Q_field
    
    def step(
        self,
        T: QTT2DNativeState,
        u: QTT2DNativeState,
        v: QTT2DNativeState,
        dt: float
    ) -> QTT2DNativeState:
        """
        Advance temperature field by one timestep.
        
        Uses forward Euler with operator splitting:
        T^{n+1} = T^n + dt * (-u·∇T + α∇²T + Q/(ρCp))
        """
        cfg = self.config
        
        # Advection: -u·∇T = -(u·∂T/∂x + v·∂T/∂y)
        dTdx = self.ns._ddx(T)
        dTdy = self.ns._ddy(T)
        
        u_dTdx = self.ns._hadamard(u, dTdx)
        v_dTdy = self.ns._hadamard(v, dTdy)
        
        advection = self.ns._add(u_dTdx, v_dTdy)
        advection = self.ns._scale(advection, -1.0)
        
        # Diffusion: α∇²T
        lap_T = self.ns._laplacian(T)
        diffusion = self.ns._scale(lap_T, cfg.alpha)
        
        # Source: Q / (ρ·Cp)
        if self.heat_sources:
            source_coeff = 1.0 / (cfg.rho * cfg.Cp)
            source = self.ns._scale(self.Q_field, source_coeff)
        else:
            source = None
        
        # Combine RHS
        rhs = self.ns._add(advection, diffusion)
        if source is not None:
            rhs = self.ns._add(rhs, source)
        
        # Forward Euler update
        dT = self.ns._scale(rhs, dt)
        T_new = self.ns._add(T, dT)
        
        return T_new
    
    def compute_thermal_dt(self, velocity_scale: float = 1.0) -> float:
        """
        Compute stable timestep for thermal transport.
        
        Must satisfy both advective and diffusive CFL:
        - dt_adv < CFL * dx / u_max
        - dt_diff < dx² / (2 * α * ndim)
        """
        cfg = self.ns.config
        dx = min(cfg.dx, cfg.dy)
        
        # Advective limit
        dt_adv = 0.3 * dx / max(velocity_scale, 0.01)
        
        # Diffusive limit (2D)
        dt_diff = dx**2 / (4 * self.config.alpha)
        
        return min(dt_adv, dt_diff)


def create_open_office_ic(
    ns_config: NS2DQTTConfig,
    office_config: OpenOfficeConfig,
    thermal_config: ThermalConfig
) -> tuple[QTT2DNativeState, QTT2DNativeState, QTT2DNativeState, list[HeatSource]]:
    """
    Create initial conditions for open office simulation.
    
    Returns:
        omega: Initial vorticity (with diffuser jets)
        psi: Initial streamfunction
        T: Initial temperature field
        heat_sources: List of heat source regions
    """
    cfg = ns_config
    ofc = office_config
    tcfg = thermal_config
    
    # Coordinate grids
    x = torch.linspace(0, cfg.Lx, cfg.Nx, device=cfg.device, dtype=cfg.dtype)
    y = torch.linspace(0, cfg.Ly, cfg.Ny, device=cfg.device, dtype=cfg.dtype)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # ========== VELOCITY (via vorticity) ==========
    # Ceiling diffusers create downward jets -> vorticity at edges
    omega = torch.zeros_like(X)
    
    diffuser_spacing = cfg.Lx / (ofc.n_diffusers + 1)
    for i in range(ofc.n_diffusers):
        x_center = (i + 1) * diffuser_spacing
        half_width = ofc.diffuser_width / 2
        
        # Positive vorticity on left edge, negative on right edge
        # This creates a downward jet between them
        left_mask = (
            (X >= x_center - half_width - cfg.dx) & 
            (X < x_center - half_width + cfg.dx) &
            (Y > cfg.Ly - 0.3)
        )
        right_mask = (
            (X >= x_center + half_width - cfg.dx) & 
            (X < x_center + half_width + cfg.dx) &
            (Y > cfg.Ly - 0.3)
        )
        
        jet_strength = ofc.supply_velocity * 100  # Vorticity magnitude
        omega[left_mask] = jet_strength
        omega[right_mask] = -jet_strength
    
    psi = torch.zeros_like(X)  # Will be solved by Poisson
    
    # ========== TEMPERATURE ==========
    # Start at ambient, with cool supply air at ceiling
    T = torch.full_like(X, tcfg.T_ambient)
    
    # Cool air at diffuser locations
    for i in range(ofc.n_diffusers):
        x_center = (i + 1) * diffuser_spacing
        half_width = ofc.diffuser_width / 2
        
        diffuser_mask = (
            (X >= x_center - half_width) & 
            (X < x_center + half_width) &
            (Y > cfg.Ly - 0.1)
        )
        T[diffuser_mask] = tcfg.T_supply
    
    # ========== HEAT SOURCES ==========
    heat_sources: list[HeatSource] = []
    
    # Distribute workstations along the room
    # Each workstation is ~1.5m wide, at desk height ~0.75m
    workstation_spacing = cfg.Lx / (ofc.n_occupants + 1)
    desk_height = 0.75
    desk_width = 1.5
    
    # Combined power per workstation (occupant + computer)
    power_per_station = ofc.occupant_power_W + ofc.computer_power_W
    
    # In 2D cross-section, distribute power density
    # Assume office depth = 20m, so power per meter depth
    depth = 20.0  # [m]
    stations_per_row = ofc.n_occupants  # Simplified: one row
    
    # Create distributed heat source band at workstation height
    heat_sources.append(HeatSource(
        x_min=1.0,
        x_max=cfg.Lx - 1.0,
        y_min=desk_height - 0.2,
        y_max=desk_height + 0.5,
        power_density=ofc.total_internal_load_W / (depth * (cfg.Lx - 2.0) * 0.7)
    ))
    
    # Solar gain at west wall (right side, x = Lx)
    solar_power_per_m = ofc.solar_peak_W_m2 * ofc.glazing_height
    heat_sources.append(HeatSource(
        x_min=cfg.Lx - 0.5,
        x_max=cfg.Lx,
        y_min=0.5,
        y_max=0.5 + ofc.glazing_height,
        power_density=solar_power_per_m / (depth * 0.5 * ofc.glazing_height)
    ))
    
    # Compress to QTT
    print("Compressing open office IC to QTT...")
    t0 = __import__('time').time()
    
    omega_qtt = dense_to_qtt_2d_native(omega, cfg.nx_bits, cfg.ny_bits, cfg.max_rank)
    psi_qtt = dense_to_qtt_2d_native(psi, cfg.nx_bits, cfg.ny_bits, cfg.max_rank)
    T_qtt = dense_to_qtt_2d_native(T, cfg.nx_bits, cfg.ny_bits, cfg.max_rank)
    
    print(f"  Compressed in {__import__('time').time() - t0:.2f}s")
    print(f"  omega rank: {omega_qtt.max_rank}, psi rank: {psi_qtt.max_rank}, T rank: {T_qtt.max_rank}")
    
    return omega_qtt, psi_qtt, T_qtt, heat_sources


class NS2D_Thermal_QTT:
    """
    Combined NS + Thermal solver for HVAC simulations.
    
    Solves:
    1. Vorticity transport: ∂ω/∂t + (u·∇)ω = ν∇²ω
    2. Poisson: ∇²ψ = -ω
    3. Velocity: u = ∂ψ/∂y, v = -∂ψ/∂x
    4. Temperature: ∂T/∂t + (u·∇)T = α∇²T + Q/(ρCp)
    """
    
    def __init__(
        self,
        ns_config: NS2DQTTConfig,
        thermal_config: ThermalConfig,
        heat_sources: Optional[list[HeatSource]] = None
    ):
        self.ns = NS2D_QTT_Native(ns_config)
        self.thermal = ThermalQTTSolver(self.ns, thermal_config, heat_sources)
        self.config = ns_config
        self.thermal_config = thermal_config
        
    def compute_velocities(
        self, 
        psi: QTT2DNativeState
    ) -> tuple[QTT2DNativeState, QTT2DNativeState]:
        """
        Recover velocity from streamfunction.
        u = ∂ψ/∂y, v = -∂ψ/∂x
        """
        u = self.ns._ddy(psi)
        v = self.ns._scale(self.ns._ddx(psi), -1.0)
        return u, v
    
    def step(
        self,
        omega: QTT2DNativeState,
        psi: QTT2DNativeState,
        T: QTT2DNativeState,
        dt: float
    ) -> tuple[QTT2DNativeState, QTT2DNativeState, QTT2DNativeState]:
        """
        Advance all fields by one timestep.
        
        Returns:
            omega_new, psi_new, T_new
        """
        # Step NS solver
        omega_new, psi_new = self.ns.step(omega, psi, dt)
        
        # Get velocities for temperature advection
        u, v = self.compute_velocities(psi)
        
        # Step thermal solver
        T_new = self.thermal.step(T, u, v, dt)
        
        return omega_new, psi_new, T_new
    
    def compute_dt(self, velocity_scale: float = 1.0) -> float:
        """Compute stable timestep for coupled system."""
        dt_ns = self.ns.compute_dt()
        dt_thermal = self.thermal.compute_thermal_dt(velocity_scale)
        return min(dt_ns, dt_thermal)
