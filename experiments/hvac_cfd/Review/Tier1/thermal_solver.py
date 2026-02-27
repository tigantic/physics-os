"""
HyperFOAM Thermal Solver

Extends the pressure-velocity solver with energy transport.
This enables proper HVAC design where you must balance:
- Draft comfort (velocity < 0.25 m/s)
- Thermal comfort (temperature 20-24°C)
- Air quality (adequate ACH for CO2 removal)

Physics:
- Advection-diffusion of temperature
- Volumetric heat sources (people, equipment)
- Buoyancy coupling (Boussinesq approximation)
- Supply air at controlled temperature

The Energy Equation:
  dT/dt + u·∇T = α∇²T + Q/(ρCp)

Where:
  T = temperature [K]
  α = thermal diffusivity [m²/s]
  Q = volumetric heat source [W/m³]
  ρCp = volumetric heat capacity [J/(m³·K)]
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from hyper_grid import HyperGrid
from hyperfoam_solver import HyperFoamSolver, ProjectionConfig


@dataclass
class ThermalConfig:
    """Thermal simulation parameters"""
    # Air properties at 20°C
    rho: float = 1.2          # Density [kg/m³]
    cp: float = 1005.0        # Specific heat [J/(kg·K)]
    k: float = 0.026          # Thermal conductivity [W/(m·K)]
    
    # Derived
    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m²/s]"""
        return self.k / (self.rho * self.cp)
    
    @property
    def rho_cp(self) -> float:
        """Volumetric heat capacity [J/(m³·K)]"""
        return self.rho * self.cp
    
    # Buoyancy
    beta: float = 1.0 / 293.0  # Thermal expansion coefficient [1/K]
    g: float = 9.81            # Gravity [m/s²]
    T_ref: float = 293.15      # Reference temperature [K] (20°C)
    
    # Supply air
    T_supply: float = 289.15   # Supply temperature [K] (16°C)
    T_initial: float = 293.15  # Initial room temperature [K] (20°C)
    
    # Heat sources
    body_heat: float = 100.0   # Heat per person [W]
    n_people: int = 12


@dataclass
class HeatSource:
    """A volumetric heat source (person, equipment, etc.)"""
    x: float  # Center position [m]
    y: float
    z: float
    power: float  # Heat output [W]
    radius: float = 0.3  # Spread radius [m]


class ThermalSolver:
    """
    Coupled momentum-thermal solver for HVAC simulation.
    
    Extends HyperFoamSolver with:
    - Temperature field
    - Advection-diffusion of energy
    - Buoyancy force (hot air rises)
    - Heat sources (people)
    """
    
    def __init__(
        self,
        grid: HyperGrid,
        flow_cfg: ProjectionConfig,
        thermal_cfg: ThermalConfig = None
    ):
        self.grid = grid
        self.flow_cfg = flow_cfg
        self.thermal_cfg = thermal_cfg or ThermalConfig()
        self.device = grid.device
        
        # Initialize flow solver
        self.flow = HyperFoamSolver(grid, flow_cfg)
        
        # Temperature field
        shape = (grid.nx, grid.ny, grid.nz)
        self.T = torch.full(shape, self.thermal_cfg.T_initial, 
                           device=self.device, dtype=torch.float32)
        
        # Heat source field [W/m³]
        self.Q = torch.zeros(shape, device=self.device, dtype=torch.float32)
        
        # Supply vent locations (set by user)
        self.supply_vents = []
        
        # Precompute
        self.dx = grid.dx
        self.dy = grid.dy
        self.dz = grid.dz
        self.dt = flow_cfg.dt
        self.alpha = self.thermal_cfg.alpha
        
    def add_person(self, x: float, y: float, z: float = 1.0, power: float = 100.0):
        """Add a heat source representing a seated person."""
        # Convert to grid indices
        ix = int(x / self.dx)
        iy = int(y / self.dy)
        iz = int(z / self.dz)
        
        # Spread heat over a small region (person's thermal plume)
        spread = 2  # cells
        
        # Compute volumetric heat rate
        vol = (2*spread * self.dx) * (2*spread * self.dy) * (2*spread * self.dz)
        q_vol = power / vol  # W/m³
        
        # Add to Q field
        ix0, ix1 = max(0, ix-spread), min(self.grid.nx, ix+spread)
        iy0, iy1 = max(0, iy-spread), min(self.grid.ny, iy+spread)
        iz0, iz1 = max(0, iz-spread), min(self.grid.nz, iz+spread)
        
        self.Q[ix0:ix1, iy0:iy1, iz0:iz1] += q_vol
        
    def add_conference_table_occupants(self, table_center: Tuple[float, float], 
                                        table_length: float, n_per_side: int = 6):
        """Add heat sources for people seated around a conference table."""
        cx, cy = table_center
        spacing = table_length / (n_per_side + 1)
        
        seated_z = 1.0  # Torso height when seated [m]
        offset_from_table = 0.5  # Distance from table edge [m]
        
        for i in range(n_per_side):
            x = cx - table_length/2 + spacing * (i + 1)
            
            # Front row (y < table)
            self.add_person(x, cy - offset_from_table, seated_z)
            
            # Back row (y > table)
            self.add_person(x, cy + offset_from_table, seated_z)
            
        print(f"    Added {n_per_side * 2} occupants ({self.thermal_cfg.body_heat}W each)")
        print(f"    Total heat load: {n_per_side * 2 * self.thermal_cfg.body_heat}W")
    
    def add_supply_vent(self, ix_range: Tuple[int, int], iy_range: Tuple[int, int], 
                        iz: int, velocity_w: float, velocity_u: float = 0.0):
        """Add a supply vent with cooled air."""
        self.supply_vents.append({
            'ix': ix_range,
            'iy': iy_range,
            'iz': iz,
            'w': velocity_w,
            'u': velocity_u,
            'T': self.thermal_cfg.T_supply
        })
        
    def apply_supply_bc(self):
        """Apply supply vent boundary conditions."""
        for vent in self.supply_vents:
            ix0, ix1 = vent['ix']
            iy0, iy1 = vent['iy']
            iz = vent['iz']
            
            # Velocity
            self.flow.w[ix0:ix1, iy0:iy1, iz] = vent['w']
            if 'u' in vent:
                self.flow.u[ix0:ix1, iy0:iy1, iz] = vent['u']
            
            # Temperature (cold supply air)
            self.T[ix0:ix1, iy0:iy1, iz] = vent['T']
    
    def step(self):
        """Advance one timestep with coupled momentum-thermal solve."""
        
        # 1. Apply supply BCs
        self.apply_supply_bc()
        
        # 2. Compute buoyancy force (Boussinesq)
        # F_buoy = -rho * g * beta * (T - T_ref)
        # This adds to w-momentum (vertical)
        T_excess = self.T - self.thermal_cfg.T_ref
        buoyancy = self.thermal_cfg.g * self.thermal_cfg.beta * T_excess
        
        # Add buoyancy to w before flow step
        self.flow.w += self.dt * buoyancy * self.flow.fluid_mask
        
        # 3. Flow step (momentum + pressure)
        self.flow.step()
        
        # 4. Temperature advection-diffusion
        self._thermal_step()
        
        # 5. Enforce thermal BCs
        self.apply_supply_bc()
        
    def _thermal_step(self):
        """Solve energy equation: dT/dt + u·∇T = α∇²T + Q/(ρCp)"""
        
        u, v, w = self.flow.u, self.flow.v, self.flow.w
        T = self.T
        dt = self.dt
        dx, dy, dz = self.dx, self.dy, self.dz
        alpha = self.alpha
        
        # Advection (upwind for stability)
        # x-direction
        T_xp = torch.roll(T, -1, 0)
        T_xm = torch.roll(T, 1, 0)
        dT_dx = torch.where(u > 0, (T - T_xm) / dx, (T_xp - T) / dx)
        
        # y-direction
        T_yp = torch.roll(T, -1, 1)
        T_ym = torch.roll(T, 1, 1)
        dT_dy = torch.where(v > 0, (T - T_ym) / dy, (T_yp - T) / dy)
        
        # z-direction
        T_zp = torch.roll(T, -1, 2)
        T_zm = torch.roll(T, 1, 2)
        dT_dz = torch.where(w > 0, (T - T_zm) / dz, (T_zp - T) / dz)
        
        advection = u * dT_dx + v * dT_dy + w * dT_dz
        
        # Diffusion (central difference)
        diff_x = (T_xp - 2*T + T_xm) / dx**2
        diff_y = (T_yp - 2*T + T_ym) / dy**2
        diff_z = (T_zp - 2*T + T_zm) / dz**2
        diffusion = alpha * (diff_x + diff_y + diff_z)
        
        # Heat source term
        source = self.Q / self.thermal_cfg.rho_cp
        
        # Update temperature
        self.T += dt * (-advection + diffusion + source)
        
        # Clamp to physical range
        self.T = torch.clamp(self.T, 273.15, 323.15)  # 0°C to 50°C
        
        # Apply solid mask (adiabatic walls inside obstacles)
        self.T *= self.flow.fluid_mask
        self.T += self.thermal_cfg.T_initial * (1 - self.flow.fluid_mask)
    
    @property
    def u(self):
        return self.flow.u
    
    @property
    def v(self):
        return self.flow.v
    
    @property
    def w(self):
        return self.flow.w
    
    @property
    def p(self):
        return self.flow.p
    
    def get_comfort_metrics(self, z_range: Tuple[float, float] = (0.5, 1.5)):
        """Compute thermal comfort metrics in occupied zone."""
        
        nz = self.grid.nz
        lz = self.flow_cfg.Lz
        
        z_low = int(z_range[0] / (lz / nz))
        z_high = int(z_range[1] / (lz / nz))
        
        # Velocity magnitude
        occ_u = self.flow.u[:, :, z_low:z_high]
        occ_v = self.flow.v[:, :, z_low:z_high]
        occ_w = self.flow.w[:, :, z_low:z_high]
        vel_mag = torch.sqrt(occ_u**2 + occ_v**2 + occ_w**2 + 1e-8)
        
        # Temperature
        occ_T = self.T[:, :, z_low:z_high]
        
        return {
            'max_velocity': vel_mag.max().item(),
            'avg_velocity': vel_mag.mean().item(),
            'max_temp_C': (occ_T.max().item() - 273.15),
            'avg_temp_C': (occ_T.mean().item() - 273.15),
            'min_temp_C': (occ_T.min().item() - 273.15),
        }


def test_thermal_solver():
    """Quick test of thermal coupling."""
    print("=" * 70)
    print("THERMAL SOLVER TEST")
    print("=" * 70)
    
    # Grid
    nx, ny, nz = 64, 48, 24
    lx, ly, lz = 9.0, 6.0, 3.0
    
    grid = HyperGrid(nx, ny, nz, lx, ly, lz, device='cuda')
    
    # Add table
    grid.add_box_obstacle(2.67, 6.33, 2.39, 3.61, 0.0, 0.76)
    
    # Configs
    flow_cfg = ProjectionConfig(nx=nx, ny=ny, nz=nz, Lx=lx, Ly=ly, Lz=lz, dt=0.005)
    thermal_cfg = ThermalConfig()
    
    # Create solver
    solver = ThermalSolver(grid, flow_cfg, thermal_cfg)
    
    # Add occupants
    print("\n[1] Adding occupants...")
    solver.add_conference_table_occupants(
        table_center=(lx/2, ly/2),
        table_length=3.66,
        n_per_side=6
    )
    
    # Add supply vents
    print("\n[2] Adding supply vents...")
    vent_size = 4
    z_ceiling = nz - 2
    
    # Two ceiling vents with angled flow
    solver.add_supply_vent(
        ix_range=(nx//4-vent_size, nx//4+vent_size),
        iy_range=(ny//4-vent_size, ny//4+vent_size),
        iz=z_ceiling,
        velocity_w=-1.5,  # Down
        velocity_u=0.5    # Spread
    )
    solver.add_supply_vent(
        ix_range=(3*nx//4-vent_size, 3*nx//4+vent_size),
        iy_range=(3*ny//4-vent_size, 3*ny//4+vent_size),
        iz=z_ceiling,
        velocity_w=-1.5,
        velocity_u=-0.5
    )
    print(f"    Supply temp: {thermal_cfg.T_supply - 273.15:.1f}°C")
    
    # Run simulation
    print("\n[3] Running thermal simulation...")
    import time
    
    n_steps = 500
    start = time.perf_counter()
    
    for step in range(n_steps):
        solver.step()
        
        if step % 100 == 0:
            metrics = solver.get_comfort_metrics()
            print(f"    Step {step}: T_avg={metrics['avg_temp_C']:.1f}°C, "
                  f"T_max={metrics['max_temp_C']:.1f}°C, "
                  f"V_max={metrics['max_velocity']:.2f} m/s")
    
    elapsed = time.perf_counter() - start
    print(f"\n    Completed {n_steps} steps in {elapsed:.1f}s ({n_steps/elapsed:.0f} steps/s)")
    
    # Final metrics
    print("\n" + "=" * 50)
    print("FINAL COMFORT METRICS")
    print("=" * 50)
    metrics = solver.get_comfort_metrics()
    
    print(f"Temperature (Occupied Zone):")
    print(f"  Average: {metrics['avg_temp_C']:.1f}°C")
    print(f"  Maximum: {metrics['max_temp_C']:.1f}°C")
    print(f"  Minimum: {metrics['min_temp_C']:.1f}°C")
    
    print(f"\nAir Velocity (Occupied Zone):")
    print(f"  Average: {metrics['avg_velocity']:.3f} m/s")
    print(f"  Maximum: {metrics['max_velocity']:.3f} m/s")
    
    # Check comfort
    T_ok = 20 <= metrics['avg_temp_C'] <= 24
    V_ok = metrics['max_velocity'] < 0.25
    
    print(f"\nASHRAE 55 Compliance:")
    if T_ok:
        print(f"  ✓ Temperature in comfort range (20-24°C)")
    else:
        print(f"  ✗ Temperature outside comfort range")
    
    if V_ok:
        print(f"  ✓ Velocity below draft limit (0.25 m/s)")
    else:
        print(f"  ✗ Velocity exceeds draft limit")


if __name__ == "__main__":
    test_thermal_solver()
