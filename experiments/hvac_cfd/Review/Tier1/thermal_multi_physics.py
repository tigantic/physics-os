"""
HyperFOAM Phase 4: Thermal & Species Transport

Production-ready thermal solver with:
- Temperature field + buoyancy coupling
- CO2 concentration tracking
- Age of Air calculation (ventilation effectiveness)
- Smoke/contaminant transport (fire scenarios)
- Dynamic heat sources (occupancy schedules)

This is the complete "Digital Twin" physics layer.

Equations:
-----------
Energy:     ∂T/∂t + u·∇T = α∇²T + Q/(ρCp)
Species:    ∂C/∂t + u·∇C = D∇²C + S
Age of Air: ∂τ/∂t + u·∇τ = D∇²τ + 1   (source = 1 everywhere)

Buoyancy:   F_z = -ρgβ(T - T_ref)  [Boussinesq approximation]
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

from hyper_grid import HyperGrid
from hyperfoam_solver import HyperFoamSolver, ProjectionConfig


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AirProperties:
    """Thermophysical properties of air at standard conditions."""
    rho: float = 1.2           # Density [kg/m³]
    cp: float = 1005.0         # Specific heat [J/(kg·K)]
    k: float = 0.026           # Thermal conductivity [W/(m·K)]
    mu: float = 1.8e-5         # Dynamic viscosity [Pa·s]
    
    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m²/s]"""
        return self.k / (self.rho * self.cp)
    
    @property
    def nu(self) -> float:
        """Kinematic viscosity [m²/s]"""
        return self.mu / self.rho
    
    @property
    def Pr(self) -> float:
        """Prandtl number"""
        return self.mu * self.cp / self.k
    
    @property
    def rho_cp(self) -> float:
        """Volumetric heat capacity [J/(m³·K)]"""
        return self.rho * self.cp


@dataclass 
class BuoyancyConfig:
    """Boussinesq buoyancy parameters."""
    enabled: bool = True
    beta: float = 1.0 / 293.0  # Thermal expansion coeff [1/K]
    g: float = 9.81            # Gravity [m/s²]
    T_ref: float = 293.15      # Reference temperature [K] (20°C)


@dataclass
class SpeciesConfig:
    """Configuration for a transported species (CO2, smoke, etc.)"""
    name: str
    diffusivity: float          # Molecular diffusivity [m²/s]
    initial_value: float = 0.0  # Initial concentration
    inlet_value: float = 0.0    # Supply air concentration
    
    # Source parameters (for CO2 from breathing, etc.)
    source_rate: float = 0.0    # Base source rate [unit/s per cell]


# Pre-configured species
CO2_CONFIG = SpeciesConfig(
    name="CO2",
    diffusivity=1.6e-5,     # CO2 in air [m²/s]
    initial_value=400.0,    # Outdoor background [ppm]
    inlet_value=400.0,      # Fresh supply air [ppm]
    source_rate=0.0         # Set per occupant later
)

SMOKE_CONFIG = SpeciesConfig(
    name="Smoke",
    diffusivity=1.0e-5,     # Particle diffusion (small)
    initial_value=0.0,      # No smoke initially
    inlet_value=0.0,        # Clean supply
    source_rate=0.0
)

AGE_OF_AIR_CONFIG = SpeciesConfig(
    name="AgeOfAir",
    diffusivity=1.0e-5,     # Same as momentum
    initial_value=0.0,      # Fresh air has age 0
    inlet_value=0.0,        # Supply air age = 0
    source_rate=1.0         # Age increases at 1 second/second everywhere
)


@dataclass
class ThermalSystemConfig:
    """Complete thermal system configuration."""
    # Physical properties
    air: AirProperties = field(default_factory=AirProperties)
    buoyancy: BuoyancyConfig = field(default_factory=BuoyancyConfig)
    
    # Temperature bounds
    T_initial: float = 293.15   # Initial room temp [K] (20°C)
    T_supply: float = 289.15    # Supply air temp [K] (16°C)
    T_min: float = 273.15       # 0°C floor
    T_max: float = 323.15       # 50°C ceiling
    
    # Species to track
    track_co2: bool = True
    track_age_of_air: bool = True
    track_smoke: bool = False
    
    # Occupant heat/CO2
    body_heat: float = 100.0       # Watts per person (seated)
    body_co2_rate: float = 0.005   # L/s CO2 per person (breathing)


# ═══════════════════════════════════════════════════════════════════════════════
# HEAT SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

class HeatSourceType(Enum):
    """Types of heat sources."""
    PERSON = "person"
    EQUIPMENT = "equipment"
    LIGHTING = "lighting"
    SOLAR = "solar"
    FIRE = "fire"


@dataclass
class HeatSource:
    """
    A volumetric heat source.
    
    Can represent:
    - Seated person (100W sensible + CO2)
    - Computer equipment (50-200W)
    - Lighting (W/m²)
    - Fire (kW + smoke)
    """
    name: str
    source_type: HeatSourceType
    
    # Position (meters)
    x: float
    y: float
    z: float
    
    # Heat output
    power: float = 100.0       # Watts
    radius: float = 0.3        # Spread [m]
    
    # CO2 emission (for people)
    co2_rate: float = 0.0      # L/s
    
    # Smoke emission (for fires)
    smoke_rate: float = 0.0    # kg/s (or arbitrary units)
    
    # Schedule (for dynamic sources)
    active: bool = True
    schedule: Optional[Dict] = None  # {hour: on/off}
    
    def is_active(self, time_seconds: float = 0) -> bool:
        """Check if source is active at given time."""
        if not self.active:
            return False
        if self.schedule is None:
            return True
        # Simple hour-based schedule
        hour = int((time_seconds / 3600) % 24)
        return self.schedule.get(hour, True)


# ═══════════════════════════════════════════════════════════════════════════════
# SCALAR TRANSPORT
# ═══════════════════════════════════════════════════════════════════════════════

class ScalarField:
    """
    A transported scalar quantity (Temperature, CO2, Smoke, Age).
    
    Solves: ∂φ/∂t + u·∇φ = D∇²φ + S
    """
    
    def __init__(
        self,
        name: str,
        shape: Tuple[int, int, int],
        device: torch.device,
        diffusivity: float,
        initial_value: float = 0.0
    ):
        self.name = name
        self.diffusivity = diffusivity
        self.device = device
        
        # The field itself
        self.phi = torch.full(shape, initial_value, device=device, dtype=torch.float32)
        
        # Source term (can be modified dynamically)
        self.source = torch.zeros(shape, device=device, dtype=torch.float32)
        
        # Inlet patches (list of dicts with region + value)
        self.inlet_bcs: List[Dict] = []
        
        # Outflow patches (zero gradient BC for scalar to exit)
        self.outflow_bcs: List[Dict] = []
        
    def add_inlet(self, ix_range: Tuple[int, int], iy_range: Tuple[int, int],
                  iz: int, value: float):
        """Add inlet boundary condition."""
        self.inlet_bcs.append({
            'ix': ix_range,
            'iy': iy_range,
            'iz': iz,
            'value': value
        })
    
    def add_outflow(self, ix_range: Tuple[int, int], iy_range: Tuple[int, int],
                    iz_range: Tuple[int, int], direction: str):
        """Add outflow boundary condition (zero gradient).
        
        Args:
            direction: 'y+', 'y-', 'z-' etc. - the direction of outflow
        """
        self.outflow_bcs.append({
            'ix': ix_range,
            'iy': iy_range,
            'iz': iz_range,
            'direction': direction
        })
    
    def apply_inlet_bcs(self):
        """Apply inlet boundary conditions."""
        for bc in self.inlet_bcs:
            ix0, ix1 = bc['ix']
            iy0, iy1 = bc['iy']
            iz = bc['iz']
            self.phi[ix0:ix1, iy0:iy1, iz] = bc['value']
    
    def apply_outflow_bcs(self):
        """Apply outflow boundary conditions (zero gradient - let scalar exit)."""
        for bc in self.outflow_bcs:
            ix0, ix1 = bc['ix']
            iy0, iy1 = bc['iy']
            iz0, iz1 = bc['iz']
            direction = bc['direction']
            
            # Zero gradient: copy interior value to boundary
            if direction == 'y+':
                # Outflow at high y: phi[boundary] = phi[boundary-1]
                self.phi[ix0:ix1, iy1-1:iy1, iz0:iz1] = self.phi[ix0:ix1, iy0:iy0+1, iz0:iz1]
            elif direction == 'y-':
                # Outflow at low y: phi[boundary] = phi[boundary+1]
                self.phi[ix0:ix1, iy0:iy0+1, iz0:iz1] = self.phi[ix0:ix1, iy1-1:iy1, iz0:iz1]
    
    def advect_diffuse(self, u: Tensor, v: Tensor, w: Tensor,
                       dx: float, dy: float, dz: float, dt: float,
                       fluid_mask: Tensor):
        """
        Advance scalar field one timestep.
        
        Uses upwind advection + central diffusion.
        """
        phi = self.phi
        D = self.diffusivity
        
        # ─────────────────────────────────────────────────────────────────
        # Advection (Upwind for stability)
        # ─────────────────────────────────────────────────────────────────
        phi_xp = torch.roll(phi, -1, 0)
        phi_xm = torch.roll(phi, 1, 0)
        dphi_dx = torch.where(u > 0, (phi - phi_xm) / dx, (phi_xp - phi) / dx)
        
        phi_yp = torch.roll(phi, -1, 1)
        phi_ym = torch.roll(phi, 1, 1)
        dphi_dy = torch.where(v > 0, (phi - phi_ym) / dy, (phi_yp - phi) / dy)
        
        phi_zp = torch.roll(phi, -1, 2)
        phi_zm = torch.roll(phi, 1, 2)
        dphi_dz = torch.where(w > 0, (phi - phi_zm) / dz, (phi_zp - phi) / dz)
        
        advection = u * dphi_dx + v * dphi_dy + w * dphi_dz
        
        # ─────────────────────────────────────────────────────────────────
        # Diffusion (Central difference)
        # ─────────────────────────────────────────────────────────────────
        lap_x = (phi_xp - 2*phi + phi_xm) / dx**2
        lap_y = (phi_yp - 2*phi + phi_ym) / dy**2
        lap_z = (phi_zp - 2*phi + phi_zm) / dz**2
        diffusion = D * (lap_x + lap_y + lap_z)
        
        # ─────────────────────────────────────────────────────────────────
        # Update
        # ─────────────────────────────────────────────────────────────────
        self.phi += dt * (-advection + diffusion + self.source) * fluid_mask
        
        # Apply BCs
        self.apply_inlet_bcs()
        self.apply_outflow_bcs()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN THERMAL SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class ThermalMultiPhysicsSolver:
    """
    Complete thermal + species transport solver.
    
    Capabilities:
    - Temperature with buoyancy coupling
    - CO2 tracking (ventilation effectiveness)
    - Age of Air (mean age for each cell)
    - Smoke/contaminant transport
    - Dynamic heat sources with schedules
    
    This is the production-ready "Phase 4" solver.
    """
    
    def __init__(
        self,
        grid: HyperGrid,
        flow_cfg: ProjectionConfig,
        thermal_cfg: ThermalSystemConfig = None
    ):
        self.grid = grid
        self.flow_cfg = flow_cfg
        self.cfg = thermal_cfg or ThermalSystemConfig()
        self.device = grid.device
        
        # ─────────────────────────────────────────────────────────────────
        # Flow solver
        # ─────────────────────────────────────────────────────────────────
        self.flow = HyperFoamSolver(grid, flow_cfg)
        
        shape = (grid.nx, grid.ny, grid.nz)
        
        # ─────────────────────────────────────────────────────────────────
        # Temperature field
        # ─────────────────────────────────────────────────────────────────
        self.temperature = ScalarField(
            name="Temperature",
            shape=shape,
            device=self.device,
            diffusivity=self.cfg.air.alpha,
            initial_value=self.cfg.T_initial
        )
        
        # ─────────────────────────────────────────────────────────────────
        # CO2 field
        # ─────────────────────────────────────────────────────────────────
        if self.cfg.track_co2:
            self.co2 = ScalarField(
                name="CO2",
                shape=shape,
                device=self.device,
                diffusivity=CO2_CONFIG.diffusivity,
                initial_value=CO2_CONFIG.initial_value
            )
        else:
            self.co2 = None
        
        # ─────────────────────────────────────────────────────────────────
        # Age of Air field
        # ─────────────────────────────────────────────────────────────────
        if self.cfg.track_age_of_air:
            self.age_of_air = ScalarField(
                name="AgeOfAir",
                shape=shape,
                device=self.device,
                diffusivity=AGE_OF_AIR_CONFIG.diffusivity,
                initial_value=0.0
            )
            # Age increases at 1 second/second everywhere
            self.age_of_air.source[:] = 1.0
        else:
            self.age_of_air = None
        
        # ─────────────────────────────────────────────────────────────────
        # Smoke field (optional)
        # ─────────────────────────────────────────────────────────────────
        if self.cfg.track_smoke:
            self.smoke = ScalarField(
                name="Smoke",
                shape=shape,
                device=self.device,
                diffusivity=SMOKE_CONFIG.diffusivity,
                initial_value=0.0
            )
        else:
            self.smoke = None
        
        # ─────────────────────────────────────────────────────────────────
        # Heat sources
        # ─────────────────────────────────────────────────────────────────
        self.heat_sources: List[HeatSource] = []
        self.heat_source_field = torch.zeros(shape, device=self.device)
        self.co2_source_field = torch.zeros(shape, device=self.device)
        self.smoke_source_field = torch.zeros(shape, device=self.device)
        
        # ─────────────────────────────────────────────────────────────────
        # Supply vents
        # ─────────────────────────────────────────────────────────────────
        self.supply_vents: List[Dict] = []
        
        # ─────────────────────────────────────────────────────────────────
        # Precompute
        # ─────────────────────────────────────────────────────────────────
        self.dx = grid.dx
        self.dy = grid.dy
        self.dz = grid.dz
        self.dt = flow_cfg.dt
        self.fluid_mask = self.flow.fluid_mask
        
        # Simulation time
        self.time = 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # HEAT SOURCE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def add_heat_source(self, source: HeatSource) -> None:
        """Add a heat source to the simulation."""
        self.heat_sources.append(source)
        self._update_source_fields()
    
    def add_person(self, x: float, y: float, z: float = 1.0,
                   name: str = "Person", power: float = None,
                   co2_rate: float = None) -> HeatSource:
        """Convenience method to add a seated person."""
        source = HeatSource(
            name=name,
            source_type=HeatSourceType.PERSON,
            x=x, y=y, z=z,
            power=power or self.cfg.body_heat,
            radius=0.3,
            co2_rate=co2_rate or self.cfg.body_co2_rate
        )
        self.add_heat_source(source)
        return source
    
    def add_fire(self, x: float, y: float, z: float = 0.5,
                 power: float = 10000.0, smoke_rate: float = 0.1,
                 name: str = "Fire") -> HeatSource:
        """Add a fire source (for smoke simulation)."""
        source = HeatSource(
            name=name,
            source_type=HeatSourceType.FIRE,
            x=x, y=y, z=z,
            power=power,
            radius=0.5,
            smoke_rate=smoke_rate
        )
        self.add_heat_source(source)
        return source
    
    def add_equipment(self, x: float, y: float, z: float,
                      power: float = 100.0, name: str = "Equipment") -> HeatSource:
        """Add equipment heat source (computer, projector, etc.)."""
        source = HeatSource(
            name=name,
            source_type=HeatSourceType.EQUIPMENT,
            x=x, y=y, z=z,
            power=power,
            radius=0.2
        )
        self.add_heat_source(source)
        return source
    
    def add_occupants_around_table(self, table_center: Tuple[float, float],
                                    table_length: float, n_per_side: int = 6) -> None:
        """Add occupants seated around a conference table."""
        cx, cy = table_center
        spacing = table_length / (n_per_side + 1)
        seated_z = 1.0
        offset = 0.5
        
        for i in range(n_per_side):
            x = cx - table_length/2 + spacing * (i + 1)
            
            # Front row
            self.add_person(x, cy - offset, seated_z, f"Person_F{i+1}")
            
            # Back row
            self.add_person(x, cy + offset, seated_z, f"Person_B{i+1}")
        
        total = n_per_side * 2
        total_heat = total * self.cfg.body_heat
        print(f"    Added {total} occupants ({self.cfg.body_heat}W each)")
        print(f"    Total heat load: {total_heat}W")
    
    def _update_source_fields(self) -> None:
        """Rebuild source fields from heat source list."""
        self.heat_source_field.zero_()
        self.co2_source_field.zero_()
        self.smoke_source_field.zero_()
        
        for src in self.heat_sources:
            if not src.is_active(self.time):
                continue
            
            # Convert position to indices
            ix = int(src.x / self.dx)
            iy = int(src.y / self.dy)
            iz = int(src.z / self.dz)
            
            # Spread radius in cells
            spread = max(1, int(src.radius / self.dx))
            
            # Bounds
            ix0 = max(0, ix - spread)
            ix1 = min(self.grid.nx, ix + spread)
            iy0 = max(0, iy - spread)
            iy1 = min(self.grid.ny, iy + spread)
            iz0 = max(0, iz - spread)
            iz1 = min(self.grid.nz, iz + spread)
            
            # Volume of source region
            vol = (ix1 - ix0) * self.dx * (iy1 - iy0) * self.dy * (iz1 - iz0) * self.dz
            
            if vol > 0:
                # Heat source [W/m³]
                q_vol = src.power / vol
                self.heat_source_field[ix0:ix1, iy0:iy1, iz0:iz1] += q_vol
                
                # CO2 source [ppm/s]
                # co2_rate is in L/s of CO2 gas
                # 1 ppm = 1e-6 = 1 mL / 1 m³ = 0.001 L / m³
                # So CO2 rate in L/s into vol m³ gives (co2_rate / vol) L/(s·m³)
                # Convert to ppm/s: × 1000 (L to mL) × 1 (mL/m³ = ppm for 1m³)
                # Actually: co2_rate [L/s] / vol [m³] = L/(s·m³) = 1000 mL/(s·m³) = 1000 ppm/s
                if src.co2_rate > 0 and self.co2 is not None:
                    # co2_rate L/s -> mL/s (×1000), divide by vol to get ppm/s per m³
                    co2_vol = src.co2_rate * 1000.0 / vol  # [ppm/s] per m³
                    self.co2_source_field[ix0:ix1, iy0:iy1, iz0:iz1] += co2_vol
                
                # Smoke source
                if src.smoke_rate > 0 and self.smoke is not None:
                    smoke_vol = src.smoke_rate / vol
                    self.smoke_source_field[ix0:ix1, iy0:iy1, iz0:iz1] += smoke_vol
        
        # Apply to scalar fields
        self.temperature.source[:] = self.heat_source_field / self.cfg.air.rho_cp
        if self.co2 is not None:
            self.co2.source[:] = self.co2_source_field
        if self.smoke is not None:
            self.smoke.source[:] = self.smoke_source_field
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUPPLY VENT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def add_supply_vent(self, ix_range: Tuple[int, int], iy_range: Tuple[int, int],
                        iz: int, velocity_w: float, velocity_u: float = 0.0,
                        temperature: float = None) -> None:
        """Add a supply vent with controlled velocity and temperature."""
        T = temperature if temperature is not None else self.cfg.T_supply
        
        vent = {
            'ix': ix_range,
            'iy': iy_range,
            'iz': iz,
            'w': velocity_w,
            'u': velocity_u,
            'T': T
        }
        self.supply_vents.append(vent)
        
        # Also add as inlet BC for scalar fields
        self.temperature.add_inlet(ix_range, iy_range, iz, T)
        
        if self.co2 is not None:
            self.co2.add_inlet(ix_range, iy_range, iz, CO2_CONFIG.inlet_value)
        
        if self.age_of_air is not None:
            self.age_of_air.add_inlet(ix_range, iy_range, iz, 0.0)  # Fresh air
        
        if self.smoke is not None:
            self.smoke.add_inlet(ix_range, iy_range, iz, 0.0)  # Clean air
    
    def _apply_supply_bcs(self) -> None:
        """Apply supply vent boundary conditions."""
        for vent in self.supply_vents:
            ix0, ix1 = vent['ix']
            iy0, iy1 = vent['iy']
            iz = vent['iz']
            
            # Velocity
            self.flow.w[ix0:ix1, iy0:iy1, iz] = vent['w']
            if 'u' in vent:
                self.flow.u[ix0:ix1, iy0:iy1, iz] = vent['u']
    
    # ═══════════════════════════════════════════════════════════════════════
    # TIME STEPPING
    # ═══════════════════════════════════════════════════════════════════════
    
    def step(self) -> None:
        """Advance simulation by one timestep."""
        
        # 1. Update dynamic sources (occupancy schedules)
        self._update_source_fields()
        
        # 2. Apply supply BCs
        self._apply_supply_bcs()
        
        # 3. Buoyancy force
        if self.cfg.buoyancy.enabled:
            T = self.temperature.phi
            T_ref = self.cfg.buoyancy.T_ref
            g = self.cfg.buoyancy.g
            beta = self.cfg.buoyancy.beta
            
            # F_buoy = g * beta * (T - T_ref)
            buoyancy = g * beta * (T - T_ref)
            self.flow.w += self.dt * buoyancy * self.fluid_mask
        
        # 4. Flow step (momentum + pressure)
        self.flow.step()
        
        # 5. Temperature transport
        u, v, w = self.flow.u, self.flow.v, self.flow.w
        self.temperature.advect_diffuse(u, v, w, self.dx, self.dy, self.dz,
                                         self.dt, self.fluid_mask)
        
        # Clamp temperature
        self.temperature.phi = torch.clamp(
            self.temperature.phi,
            self.cfg.T_min, self.cfg.T_max
        )
        
        # 6. CO2 transport
        if self.co2 is not None:
            self.co2.advect_diffuse(u, v, w, self.dx, self.dy, self.dz,
                                    self.dt, self.fluid_mask)
            # Clamp CO2 (0 to 10000 ppm)
            self.co2.phi = torch.clamp(self.co2.phi, 0, 10000)
        
        # 7. Age of Air transport
        if self.age_of_air is not None:
            self.age_of_air.advect_diffuse(u, v, w, self.dx, self.dy, self.dz,
                                           self.dt, self.fluid_mask)
            # Clamp age (0 to 1 hour)
            self.age_of_air.phi = torch.clamp(self.age_of_air.phi, 0, 3600)
        
        # 8. Smoke transport
        if self.smoke is not None:
            self.smoke.advect_diffuse(u, v, w, self.dx, self.dy, self.dz,
                                      self.dt, self.fluid_mask)
            self.smoke.phi = torch.clamp(self.smoke.phi, 0, 100)
        
        # Update time
        self.time += self.dt
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMFORT METRICS
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_comfort_metrics(self, z_range: Tuple[float, float] = (0.5, 1.5)) -> Dict:
        """
        Compute ASHRAE 55 comfort metrics in occupied zone.
        
        Returns dict with:
        - Temperature (avg, min, max) in °C
        - Velocity (avg, max) in m/s
        - CO2 concentration (avg, max) in ppm
        - Age of Air (avg, max) in seconds
        """
        nz = self.grid.nz
        lz = self.flow_cfg.Lz
        
        z_low = int(z_range[0] / (lz / nz))
        z_high = int(z_range[1] / (lz / nz))
        
        # Velocity
        u = self.flow.u[:, :, z_low:z_high]
        v = self.flow.v[:, :, z_low:z_high]
        w = self.flow.w[:, :, z_low:z_high]
        vel_mag = torch.sqrt(u**2 + v**2 + w**2 + 1e-8)
        
        # Temperature
        T = self.temperature.phi[:, :, z_low:z_high]
        
        metrics = {
            'velocity_avg': vel_mag.mean().item(),
            'velocity_max': vel_mag.max().item(),
            'temperature_avg_C': T.mean().item() - 273.15,
            'temperature_min_C': T.min().item() - 273.15,
            'temperature_max_C': T.max().item() - 273.15,
        }
        
        if self.co2 is not None:
            co2 = self.co2.phi[:, :, z_low:z_high]
            metrics['co2_avg_ppm'] = co2.mean().item()
            metrics['co2_max_ppm'] = co2.max().item()
        
        if self.age_of_air is not None:
            age = self.age_of_air.phi[:, :, z_low:z_high]
            metrics['age_of_air_avg_s'] = age.mean().item()
            metrics['age_of_air_max_s'] = age.max().item()
        
        if self.smoke is not None:
            smoke = self.smoke.phi[:, :, z_low:z_high]
            metrics['smoke_avg'] = smoke.mean().item()
            metrics['smoke_max'] = smoke.max().item()
        
        return metrics
    
    def check_ashrae_compliance(self) -> Dict:
        """Check ASHRAE 55 and 62.1 compliance."""
        m = self.get_comfort_metrics()
        
        results = {
            'thermal_comfort': 20.0 <= m['temperature_avg_C'] <= 24.0,
            'draft_risk': m['velocity_max'] < 0.25,
            'co2_acceptable': m.get('co2_max_ppm', 0) < 1000,
            'ventilation_effective': m.get('age_of_air_max_s', 0) < 600,  # < 10 min
        }
        
        results['overall_pass'] = all(results.values())
        
        return results
    
    # ═══════════════════════════════════════════════════════════════════════
    # SHORTCUTS
    # ═══════════════════════════════════════════════════════════════════════
    
    @property
    def T(self) -> Tensor:
        """Temperature field [K]"""
        return self.temperature.phi
    
    @property
    def T_celsius(self) -> Tensor:
        """Temperature field [°C]"""
        return self.temperature.phi - 273.15
    
    @property
    def u(self) -> Tensor:
        return self.flow.u
    
    @property
    def v(self) -> Tensor:
        return self.flow.v
    
    @property
    def w(self) -> Tensor:
        return self.flow.w


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_thermal_multi_physics():
    """Comprehensive test of the thermal multi-physics solver."""
    print("=" * 70)
    print("THERMAL MULTI-PHYSICS SOLVER TEST")
    print("=" * 70)
    
    # Grid
    nx, ny, nz = 64, 48, 24
    lx, ly, lz = 9.0, 6.0, 3.0
    
    grid = HyperGrid(nx, ny, nz, lx, ly, lz, device='cuda')
    grid.add_box_obstacle(2.67, 6.33, 2.39, 3.61, 0.0, 0.76)  # Table
    
    # Configs
    flow_cfg = ProjectionConfig(nx=nx, ny=ny, nz=nz, Lx=lx, Ly=ly, Lz=lz, dt=0.005)
    thermal_cfg = ThermalSystemConfig(
        track_co2=True,
        track_age_of_air=True,
        track_smoke=False
    )
    
    # Create solver
    print("\n[1] Creating solver...")
    solver = ThermalMultiPhysicsSolver(grid, flow_cfg, thermal_cfg)
    
    # Add occupants
    print("\n[2] Adding occupants...")
    solver.add_occupants_around_table((lx/2, ly/2), table_length=3.66, n_per_side=6)
    
    # Add supply vents (optimized configuration)
    print("\n[3] Adding supply vents...")
    vent_size = 4
    z_ceiling = nz - 2
    
    opt_vel = 0.93
    opt_angle = 31.5
    import math
    u_comp = opt_vel * math.sin(math.radians(opt_angle))
    w_comp = -opt_vel * math.cos(math.radians(opt_angle))
    
    solver.add_supply_vent(
        (nx//4-vent_size, nx//4+vent_size),
        (ny//4-vent_size, ny//4+vent_size),
        z_ceiling, velocity_w=w_comp, velocity_u=u_comp
    )
    solver.add_supply_vent(
        (3*nx//4-vent_size, 3*nx//4+vent_size),
        (3*ny//4-vent_size, 3*ny//4+vent_size),
        z_ceiling, velocity_w=w_comp, velocity_u=-u_comp
    )
    print(f"    Supply: {opt_vel:.2f} m/s @ {opt_angle}°")
    print(f"    Temperature: {thermal_cfg.T_supply - 273.15:.0f}°C")
    
    # Run simulation
    print("\n[4] Running simulation (60s physical time)...")
    import time
    
    n_steps = 1200  # 60s at dt=0.05
    start = time.perf_counter()
    
    for step in range(n_steps):
        solver.step()
        
        if step % 200 == 0:
            m = solver.get_comfort_metrics()
            t = solver.time
            print(f"    t={t:5.1f}s | T={m['temperature_avg_C']:.1f}°C | "
                  f"CO2={m.get('co2_avg_ppm', 0):.0f}ppm | "
                  f"Age={m.get('age_of_air_avg_s', 0):.1f}s | "
                  f"V_max={m['velocity_max']:.2f}m/s")
    
    elapsed = time.perf_counter() - start
    print(f"\n    Completed in {elapsed:.1f}s ({n_steps/elapsed:.0f} steps/s)")
    
    # Final metrics
    print("\n" + "=" * 50)
    print("FINAL COMFORT METRICS")
    print("=" * 50)
    
    m = solver.get_comfort_metrics()
    
    print(f"\nTemperature (Occupied Zone):")
    print(f"  Average: {m['temperature_avg_C']:.1f}°C")
    print(f"  Range: {m['temperature_min_C']:.1f}°C - {m['temperature_max_C']:.1f}°C")
    
    print(f"\nAir Velocity:")
    print(f"  Average: {m['velocity_avg']:.3f} m/s")
    print(f"  Maximum: {m['velocity_max']:.3f} m/s")
    
    if 'co2_avg_ppm' in m:
        print(f"\nCO2 Concentration:")
        print(f"  Average: {m['co2_avg_ppm']:.0f} ppm")
        print(f"  Maximum: {m['co2_max_ppm']:.0f} ppm")
    
    if 'age_of_air_avg_s' in m:
        print(f"\nAge of Air:")
        print(f"  Average: {m['age_of_air_avg_s']:.1f}s ({m['age_of_air_avg_s']/60:.1f} min)")
        print(f"  Maximum: {m['age_of_air_max_s']:.1f}s ({m['age_of_air_max_s']/60:.1f} min)")
    
    # ASHRAE check
    print("\n" + "-" * 50)
    compliance = solver.check_ashrae_compliance()
    print("ASHRAE Compliance:")
    print(f"  Thermal Comfort (20-24°C): {'✓' if compliance['thermal_comfort'] else '✗'}")
    print(f"  Draft Risk (<0.25 m/s):    {'✓' if compliance['draft_risk'] else '✗'}")
    print(f"  CO2 Level (<1000 ppm):     {'✓' if compliance['co2_acceptable'] else '✗'}")
    print(f"  Ventilation (<10 min age): {'✓' if compliance['ventilation_effective'] else '✗'}")
    print(f"\n  OVERALL: {'PASS ✓' if compliance['overall_pass'] else 'FAIL ✗'}")


if __name__ == "__main__":
    test_thermal_multi_physics()
