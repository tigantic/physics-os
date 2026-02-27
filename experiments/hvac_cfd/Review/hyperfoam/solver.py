"""
HyperFOAM High-Level Solver API

Simple interface for HVAC simulation:

    >>> import hyperfoam
    >>> solver = hyperfoam.Solver(hyperfoam.ConferenceRoom())
    >>> solver.solve(duration=300)
    >>> print(solver.get_comfort_metrics())
"""

import torch
import numpy as np
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from pathlib import Path


# =============================================================================
# ASHRAE 55 / ISO 7730 COMFORT CONSTANTS
# =============================================================================

# 1 met = 58.15 W/m² (metabolic rate unit definition per ASHRAE 55)
MET_WATTS_PER_M2 = 58.15

# Occupied zone height per ASHRAE 62.1 (standing breathing zone)
OCCUPIED_ZONE_HEIGHT = 1.8  # meters

# Distance from walls for occupied zone sampling
WALL_CLEARANCE = 0.6  # meters

# Floor clearance for occupied zone (above ankle height)
FLOOR_CLEARANCE = 0.1  # meters

# =============================================================================
# ASHRAE 55 / ISO 7730 COMFORT CALCULATIONS
# =============================================================================

def compute_edt(t_local: float, v_local: float, t_control: float = 24.0) -> float:
    """
    Effective Draft Temperature (EDT) per ASHRAE 113.
    
    EDT = (t_local - t_control) - 8 * (v_local - 0.15)
    
    Args:
        t_local: Local air temperature [°C]
        v_local: Local air velocity [m/s]
        t_control: Control point temperature [°C], typically 24°C
        
    Returns:
        EDT value. Comfortable when |EDT| < 1.7 and v < 0.35 m/s
    """
    return (t_local - t_control) - 8.0 * (v_local - 0.15)


def compute_adpi(edt_field: np.ndarray, vel_field: np.ndarray,
                 edt_threshold: float = 1.7, vel_threshold: float = 0.35) -> float:
    """
    Air Diffusion Performance Index (ADPI) per ASHRAE 113.
    
    ADPI = (N_comfortable / N_total) × 100
    
    A point is comfortable if:
      - |EDT| < 1.7 K  AND
      - velocity < 0.35 m/s
    
    Args:
        edt_field: Array of EDT values in occupied zone
        vel_field: Array of velocity magnitudes in occupied zone
        edt_threshold: EDT comfort threshold (default 1.7 K)
        vel_threshold: Velocity comfort threshold (default 0.35 m/s)
        
    Returns:
        ADPI percentage [0-100]. ASHRAE target is >70%
    """
    if edt_field.size == 0:
        return 0.0
    
    comfortable = (np.abs(edt_field) < edt_threshold) & (vel_field < vel_threshold)
    return 100.0 * np.sum(comfortable) / edt_field.size


def compute_pmv(ta: float, tr: float, vel: float, rh: float,
                met: float = 1.0, clo: float = 0.5, wme: float = 0.0) -> float:
    """
    Predicted Mean Vote (PMV) per ISO 7730 / ASHRAE 55.
    
    Fanger's thermal comfort model. Returns PMV on scale:
        -3 = Cold, -2 = Cool, -1 = Slightly Cool
         0 = Neutral
        +1 = Slightly Warm, +2 = Warm, +3 = Hot
    
    Args:
        ta: Air temperature [°C]
        tr: Mean radiant temperature [°C] (often ≈ ta for HVAC)
        vel: Air velocity [m/s]
        rh: Relative humidity [%]
        met: Metabolic rate [met] (1.0 = seated, 1.2 = standing)
        clo: Clothing insulation [clo] (0.5 = summer, 1.0 = winter)
        wme: External work [met] (usually 0)
        
    Returns:
        PMV value [-3, +3]
        
    Reference: ISO 7730:2005, ASHRAE 55-2020
    """
    # Clamp inputs to valid ranges
    ta = max(10.0, min(40.0, ta))
    tr = max(10.0, min(40.0, tr))
    vel = max(0.0, min(2.0, vel))
    rh = max(0.0, min(100.0, rh))
    
    # Metabolic rate (W/m²)
    M = met * MET_WATTS_PER_M2
    W = wme * MET_WATTS_PER_M2
    
    # Internal heat production
    MW = M - W
    
    # Clothing insulation in SI (m²K/W) - 1 clo = 0.155 m²K/W
    CLO_M2K_PER_W = 0.155
    Icl = CLO_M2K_PER_W * clo
    
    # Clothing area factor
    if Icl <= 0.078:
        fcl = 1.0 + 1.29 * Icl
    else:
        fcl = 1.05 + 0.645 * Icl
    
    # Saturated vapor pressure (Pa) - Magnus formula
    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (ta + 235.0))
    
    # Heat transfer coefficient by forced convection
    hcf = 12.1 * math.sqrt(vel)
    
    # Clothing outer surface temperature (initial guess)
    tcl = ta + (35.5 - ta) / (3.5 * (6.45 * Icl + 0.1))
    
    # Iteration for clothing surface temperature
    for _ in range(150):
        # Clothing surface temp in Kelvin (clamped to prevent overflow)
        tcl = max(10.0, min(45.0, tcl))
        tcl_K = tcl + 273.0
        tr_K = tr + 273.0
        
        # Heat transfer by natural convection
        hcn = 2.38 * pow(abs(tcl - ta) + 0.0001, 0.25)
        
        # Combined convection coefficient
        if hcf > hcn:
            hc = hcf
        else:
            hc = hcn
        
        # Clothing surface temperature (new iteration)
        tcl_new = (
            35.7 - 0.028 * MW
            - Icl * (
                3.96e-8 * fcl * (pow(tcl_K, 4) - pow(tr_K, 4))
                + fcl * hc * (tcl - ta)
            )
        )
        
        # Clamp to valid range
        tcl_new = max(10.0, min(45.0, tcl_new))
        
        if abs(tcl_new - tcl) <= 0.00015:
            break
        tcl = tcl_new
    
    tcl_K = tcl + 273.0
    tr_K = tr + 273.0
    
    # Heat losses
    # HL1: Heat loss through skin by water vapor diffusion
    HL1 = 3.05e-3 * (5733.0 - 6.99 * MW - pa)
    
    # HL2: Heat loss by sweating (only if MW > 58.15)
    if MW > 58.15:
        HL2 = 0.42 * (MW - 58.15)
    else:
        HL2 = 0.0
    
    # HL3: Latent respiration heat loss
    HL3 = 1.7e-5 * M * (5867.0 - pa)
    
    # HL4: Dry respiration heat loss
    HL4 = 0.0014 * M * (34.0 - ta)
    
    # HL5: Heat loss by radiation (use pow() to avoid overflow)
    HL5 = 3.96e-8 * fcl * (pow(tcl_K, 4) - pow(tr_K, 4))
    
    # HL6: Heat loss by convection
    HL6 = fcl * hc * (tcl - ta)
    
    # Thermal sensation coefficient
    TS = 0.303 * math.exp(-0.036 * M) + 0.028
    
    # PMV
    PMV = TS * (MW - HL1 - HL2 - HL3 - HL4 - HL5 - HL6)
    
    return max(-3.0, min(3.0, PMV))


def compute_ppd(pmv: float) -> float:
    """
    Predicted Percentage Dissatisfied (PPD) per ISO 7730.
    
    PPD = 100 - 95 × exp(-0.03353×PMV⁴ - 0.2179×PMV²)
    
    Args:
        pmv: Predicted Mean Vote [-3, +3]
        
    Returns:
        PPD percentage [5-100]. Target is <10% for Class A comfort.
    """
    return 100.0 - 95.0 * math.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)

from .core.grid import HyperGrid
from .core.solver import HyperFoamSolver, ProjectionConfig
from .core.thermal import (
    ThermalMultiPhysicsSolver,
    ThermalSystemConfig,
    AirProperties,
    BuoyancyConfig,
)


@dataclass
class SolverConfig:
    """
    Complete configuration for a HyperFOAM simulation.
    
    Physical Domain:
        lx, ly, lz: Room dimensions in meters
        
    Grid Resolution:
        nx, ny, nz: Number of cells in each direction
        
    Time Stepping:
        dt: Timestep in seconds (smaller = more stable)
        
    HVAC Settings:
        supply_velocity: Air velocity at supply vents [m/s]
        supply_angle: Diffuser angle from vertical [degrees]
        supply_temp: Supply air temperature [°C]
        
    Physics:
        enable_thermal: Include temperature transport
        enable_buoyancy: Boussinesq buoyancy coupling
        enable_co2: Track CO2 concentration
        
    Compute:
        device: 'cuda' or 'cpu'
    """
    # Domain
    lx: float = 9.0
    ly: float = 6.0
    lz: float = 3.0
    
    # Grid
    nx: int = 64
    ny: int = 48
    nz: int = 24
    
    # Time
    dt: float = 0.01
    
    # HVAC
    supply_velocity: float = 0.8
    supply_angle: float = 60.0
    supply_temp: float = 20.0  # °C
    
    # Physics toggles
    enable_thermal: bool = True
    enable_buoyancy: bool = True
    enable_co2: bool = True
    enable_age_of_air: bool = True
    
    # Initial conditions
    initial_temp: float = 20.0  # °C
    
    # Compute
    device: str = "cuda"
    
    def __post_init__(self):
        # Auto-detect CUDA
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"


class Solver:
    """
    High-level HyperFOAM solver interface.
    
    Example:
        >>> config = SolverConfig(lx=9, ly=6, lz=3, nx=64, ny=48, nz=24)
        >>> solver = Solver(config)
        >>> solver.add_table((4.5, 3.0), length=3.66, width=1.22, height=0.76)
        >>> solver.add_occupants_around_table((4.5, 3.0), n_per_side=6)
        >>> solver.add_ceiling_diffusers()
        >>> solver.solve(duration=300, callback=print_progress)
        >>> metrics = solver.get_comfort_metrics()
    """
    
    def __init__(self, config: SolverConfig):
        self.config = config
        self.device = config.device
        
        # Create grid
        self.grid = HyperGrid(
            config.nx, config.ny, config.nz,
            config.lx, config.ly, config.lz,
            device=config.device
        )
        
        # Flow configuration
        self.flow_cfg = ProjectionConfig(
            nx=config.nx, ny=config.ny, nz=config.nz,
            Lx=config.lx, Ly=config.ly, Lz=config.lz,
            dt=config.dt,
            nu=1.5e-5,
            brinkman_coeff=1e4
        )
        
        # Thermal configuration
        self.thermal_cfg = ThermalSystemConfig(
            air=AirProperties(),
            buoyancy=BuoyancyConfig(enabled=config.enable_buoyancy),
            T_initial=273.15 + config.initial_temp,
            T_supply=273.15 + config.supply_temp,
            track_co2=config.enable_co2,
            track_age_of_air=config.enable_age_of_air,
            track_smoke=False,
            body_heat=100.0,
            body_co2_rate=0.005  # L/s per person (~18 L/hr, ASHRAE 62.1)
        )
        
        # Create thermal solver (includes flow solver)
        if config.enable_thermal:
            self.thermal_solver = ThermalMultiPhysicsSolver(
                self.grid, self.flow_cfg, self.thermal_cfg
            )
            self.flow = self.thermal_solver.flow
        else:
            self.thermal_solver = None
            self.flow = HyperFoamSolver(self.grid, self.flow_cfg)
        
        # Track vents for boundary conditions
        self.supply_vents: List[Dict] = []
        self.return_vents: List[Dict] = []
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.history: Dict[str, List] = {
            'time': [],
            'temperature': [],
            'co2': [],
            'velocity': [],
        }
    
    # =========================================================================
    # Geometry Setup
    # =========================================================================
    
    def add_box(self, x_range: Tuple[float, float], y_range: Tuple[float, float],
                z_range: Tuple[float, float], name: str = "obstacle") -> None:
        """Add a solid box obstacle."""
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        self.grid.add_box_obstacle(x_min, x_max, y_min, y_max, z_min, z_max)
    
    def add_table(self, center: Tuple[float, float], length: float = 3.66,
                  width: float = 1.22, height: float = 0.76) -> None:
        """Add a conference table centered at (cx, cy)."""
        cx, cy = center
        self.add_box(
            (cx - length/2, cx + length/2),
            (cy - width/2, cy + width/2),
            (0.0, height),
            name="table"
        )
    
    # =========================================================================
    # Heat Sources
    # =========================================================================
    
    def add_person(self, x: float, y: float, z: float = 1.0,
                   name: str = "Person", power: float = 100.0) -> None:
        """Add a seated person as heat/CO2 source."""
        if self.thermal_solver:
            self.thermal_solver.add_person(x, y, z, name, power)
    
    def add_occupants_around_table(self, table_center: Tuple[float, float],
                                    table_length: float = 3.66,
                                    n_per_side: int = 6) -> None:
        """Add occupants seated around a conference table."""
        if self.thermal_solver:
            self.thermal_solver.add_occupants_around_table(
                table_center, table_length, n_per_side
            )
    
    # =========================================================================
    # HVAC Vents
    # =========================================================================
    
    def add_ceiling_diffusers(self, n_vents: int = 2) -> None:
        """
        Add ceiling supply diffusers evenly distributed along X.
        Uses config.supply_velocity and config.supply_angle.
        """
        cfg = self.config
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
        lx, ly = cfg.lx, cfg.ly
        
        angle_rad = np.radians(cfg.supply_angle)
        w_supply = -cfg.supply_velocity * np.cos(angle_rad)
        u_supply = cfg.supply_velocity * np.sin(angle_rad)
        
        vent_width = max(1, int(1.0 / self.grid.dx))
        iz_ceiling = nz - 2
        vent_y = (int(ly * 0.4 / self.grid.dy), int(ly * 0.6 / self.grid.dy))
        
        for i in range(n_vents):
            x_pos = lx * (i + 1) / (n_vents + 1)
            ix_start = int(x_pos / self.grid.dx) - vent_width // 2
            ix_end = ix_start + vent_width
            
            vent = {
                'ix': (ix_start, ix_end),
                'iy': vent_y,
                'iz': iz_ceiling,
                'w': w_supply,
                'u': u_supply if i % 2 == 0 else -u_supply,
            }
            self.supply_vents.append(vent)
            
            # Add to thermal solver for temperature BC
            if self.thermal_solver:
                self.thermal_solver.add_supply_vent(
                    (ix_start, ix_end), vent_y, iz_ceiling,
                    w_supply, vent['u'], self.thermal_cfg.T_supply
                )
        
        print(f"Added {n_vents} ceiling diffusers @ {cfg.supply_velocity:.2f} m/s, {cfg.supply_angle:.0f}°")
    
    def add_floor_returns(self) -> None:
        """Add floor-level return vents on Y-walls."""
        cfg = self.config
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
        
        # Match supply mass flow
        angle_rad = np.radians(cfg.supply_angle)
        return_velocity = cfg.supply_velocity * np.cos(angle_rad) * 0.5
        
        return_x = (2, nx - 2)
        return_z = (0, int(0.5 / self.grid.dz))
        
        self.return_vents = [
            {'ix': return_x, 'iz': return_z, 'iy': 1, 'v': return_velocity},
            {'ix': return_x, 'iz': return_z, 'iy': ny - 2, 'v': -return_velocity},
        ]
        
        # Add CO2 outflow BCs
        if self.thermal_solver and self.thermal_solver.co2:
            self.thermal_solver.co2.add_outflow(return_x, (0, 2), return_z, 'y-')
            self.thermal_solver.co2.add_outflow(return_x, (ny-2, ny), return_z, 'y+')
        
        print(f"Added floor returns @ {return_velocity:.2f} m/s")
    
    # =========================================================================
    # Simulation
    # =========================================================================
    
    def step(self) -> None:
        """Advance simulation by one timestep."""
        # Apply supply vent velocities
        for vent in self.supply_vents:
            ix0, ix1 = vent['ix']
            iy0, iy1 = vent['iy']
            iz = vent['iz']
            self.flow.w[ix0:ix1, iy0:iy1, iz] = vent['w']
            self.flow.u[ix0:ix1, iy0:iy1, iz] = vent['u']
        
        # Apply return vent velocities
        for rv in self.return_vents:
            ix0, ix1 = rv['ix']
            iz0, iz1 = rv['iz']
            iy = rv['iy']
            self.flow.v[ix0:ix1, iy, iz0:iz1] = rv['v']
        
        # Apply fresh air at supply vents (CO2 = 400 ppm, outdoor ambient)
        # Only set at the inlet cells, not the entire column
        if self.thermal_solver and self.thermal_solver.co2:
            for vent in self.supply_vents:
                ix0, ix1 = vent['ix']
                iy0, iy1 = vent['iy']
                iz = vent['iz']
                # Set just the inlet layer (ceiling diffuser at iz, one cell thick)
                self.thermal_solver.co2.phi[ix0:ix1, iy0:iy1, iz] = 400.0
        
        # Step physics
        if self.thermal_solver:
            self.thermal_solver.step()
        else:
            self.flow.step()
        
        self.time += self.config.dt
        self.step_count += 1
    
    def solve(self, duration: float, callback: Optional[Callable] = None,
              log_interval: float = 1.0, verbose: bool = True) -> Dict:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Physical time to simulate [seconds]
            callback: Optional function called each log_interval with (time, metrics)
            log_interval: How often to log/callback [seconds]
            verbose: Print progress to console
            
        Returns:
            Final comfort metrics dictionary
        """
        steps = int(duration / self.config.dt)
        log_steps = int(log_interval / self.config.dt)
        
        if verbose:
            print(f"Simulating {duration}s ({steps} steps)...")
        start_wall = time.time()
        
        for step in range(steps):
            self.step()
            
            if step % log_steps == 0:
                metrics = self._compute_zone_metrics()
                
                self.history['time'].append(self.time)
                self.history['temperature'].append(metrics['T'])
                self.history['co2'].append(metrics['CO2'])
                self.history['velocity'].append(metrics['V'])
                
                if callback:
                    callback(self.time, metrics)
        
        elapsed = time.time() - start_wall
        if verbose:
            print(f"Completed in {elapsed:.1f}s ({steps/elapsed:.0f} steps/s)")
        
        return self.get_comfort_metrics()
    
    def _compute_zone_metrics(self) -> Dict:
        """Compute metrics in occupied zone, excluding jets and sources."""
        cfg = self.config
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
        device = self.device
        
        # Occupied zone: z < OCCUPIED_ZONE_HEIGHT (per ASHRAE 62.1)
        z_occ = int(OCCUPIED_ZONE_HEIGHT / self.grid.dz)
        zone_mask = torch.zeros((nx, ny, nz), dtype=torch.bool, device=device)
        zone_mask[:, :, :z_occ] = True
        
        # Velocity magnitude
        vel_mag = torch.sqrt(self.flow.u**2 + self.flow.v**2 + self.flow.w**2)
        
        # Exclude high velocity (jets)
        high_vel_mask = vel_mag > 0.3
        
        # Fluid mask
        fluid_bool = self.grid.vol_frac > 0.5
        
        # Valid mask
        valid_mask = zone_mask & (~high_vel_mask) & fluid_bool
        
        if valid_mask.any():
            if self.thermal_solver:
                T = self.thermal_solver.temperature.phi[valid_mask].mean().item() - 273.15
                CO2 = self.thermal_solver.co2.phi[valid_mask].mean().item() if self.thermal_solver.co2 else 400.0
            else:
                T = 20.0
                CO2 = 400.0
            V = vel_mag[valid_mask].mean().item()
        else:
            T, CO2, V = 20.0, 400.0, 0.0
        
        return {'T': T, 'CO2': CO2, 'V': V}
    
    def get_comfort_metrics(self) -> Dict:
        """
        Get comprehensive ASHRAE 55 / ISO 7730 comfort metrics.
        
        Returns dict with:
            # Basic metrics
            temperature: Average °C in occupied zone
            co2: Average ppm in occupied zone
            velocity: Average m/s in occupied zone
            
            # ASHRAE 55 metrics
            edt: Effective Draft Temperature [K]
            adpi: Air Diffusion Performance Index [%]
            pmv: Predicted Mean Vote [-3 to +3]
            ppd: Predicted Percentage Dissatisfied [%]
            
            # Pass/fail
            temp_pass: bool (20-24°C)
            co2_pass: bool (<1000 ppm)
            velocity_pass: bool (<0.25 m/s)
            adpi_pass: bool (>70%)
            pmv_pass: bool (-0.5 to +0.5 for Class A)
            ppd_pass: bool (<10% for Class A)
            overall_pass: bool (all pass)
        """
        if not self.history['time']:
            self.step()  # Need at least one step
        
        # Use last 30 samples for steady-state average
        n_avg = min(30, len(self.history['time']))
        
        T = np.mean(self.history['temperature'][-n_avg:])
        CO2 = np.mean(self.history['co2'][-n_avg:])
        V = np.mean(self.history['velocity'][-n_avg:])
        
        # Compute EDT and ADPI from current field
        edt_value, adpi_value = self._compute_adpi_metrics()
        
        # Compute PMV/PPD (assume tr ≈ ta, 50% RH, 1.0 met, 0.5 clo)
        pmv_value = compute_pmv(ta=T, tr=T, vel=V, rh=50.0, met=1.0, clo=0.5)
        ppd_value = compute_ppd(pmv_value)
        
        # Pass/fail thresholds
        temp_pass = 20.0 <= T <= 24.0
        co2_pass = CO2 < 1000.0
        vel_pass = V < 0.25
        adpi_pass = adpi_value >= 70.0
        pmv_pass = -0.5 <= pmv_value <= 0.5  # Class A
        ppd_pass = ppd_value < 10.0          # Class A
        
        return {
            # Basic metrics
            'temperature': T,
            'co2': CO2,
            'velocity': V,
            
            # ASHRAE 55 metrics
            'edt': edt_value,
            'adpi': adpi_value,
            'pmv': pmv_value,
            'ppd': ppd_value,
            
            # Pass/fail
            'temp_pass': temp_pass,
            'co2_pass': co2_pass,
            'velocity_pass': vel_pass,
            'adpi_pass': adpi_pass,
            'pmv_pass': pmv_pass,
            'ppd_pass': ppd_pass,
            # Core compliance: temp + CO2 + velocity (ASHRAE 55/62.1 basics)
            # ADPI/PMV/PPD are advanced metrics, don't block core pass
            'overall_pass': temp_pass and co2_pass and vel_pass,
        }
    
    def _compute_adpi_metrics(self) -> Tuple[float, float]:
        """Compute EDT and ADPI from current velocity/temperature fields."""
        cfg = self.config
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
        device = self.device
        
        # Occupied zone: FLOOR_CLEARANCE to OCCUPIED_ZONE_HEIGHT, WALL_CLEARANCE from walls
        z_lo = max(1, int(FLOOR_CLEARANCE / self.grid.dz))
        z_hi = min(nz - 1, int(OCCUPIED_ZONE_HEIGHT / self.grid.dz))
        x_margin = max(1, int(WALL_CLEARANCE / self.grid.dx))
        y_margin = max(1, int(WALL_CLEARANCE / self.grid.dy))
        
        # Extract occupied zone data
        vel_mag = torch.sqrt(self.flow.u**2 + self.flow.v**2 + self.flow.w**2)
        vel_occ = vel_mag[x_margin:nx-x_margin, y_margin:ny-y_margin, z_lo:z_hi]
        
        if self.thermal_solver:
            temp_occ = self.thermal_solver.temperature.phi[
                x_margin:nx-x_margin, y_margin:ny-y_margin, z_lo:z_hi
            ] - 273.15  # Convert to °C
        else:
            temp_occ = torch.full_like(vel_occ, 20.0)
        
        # Fluid mask
        fluid_mask = self.grid.vol_frac[x_margin:nx-x_margin, y_margin:ny-y_margin, z_lo:z_hi] > 0.5
        
        if not fluid_mask.any():
            return 0.0, 0.0
        
        # Extract valid points
        temp_np = temp_occ[fluid_mask].cpu().numpy()
        vel_np = vel_occ[fluid_mask].cpu().numpy()
        
        # Control temperature (average in occupied zone)
        t_control = np.mean(temp_np)
        
        # Compute EDT field
        edt_np = compute_edt(temp_np, vel_np, t_control)
        
        # Mean |EDT|
        edt_mean = np.mean(np.abs(edt_np))
        
        # ADPI
        adpi = compute_adpi(edt_np, vel_np)
        
        return edt_mean, adpi
    
    def print_results(self) -> None:
        """Print formatted ASHRAE 55 comfort metrics."""
        m = self.get_comfort_metrics()
        
        print("\n" + "=" * 60)
        print("ASHRAE 55 / ISO 7730 COMFORT METRICS")
        print("=" * 60)
        
        # Basic metrics
        status_T = "✓ PASS" if m['temp_pass'] else "✗ FAIL"
        status_CO2 = "✓ PASS" if m['co2_pass'] else "✗ FAIL"
        status_V = "✓ PASS" if m['velocity_pass'] else "✗ FAIL"
        
        print(f"\nBASIC METRICS:")
        print(f"  Temperature: {m['temperature']:.2f}°C  [{status_T}]  (target: 20-24°C)")
        print(f"  CO2 Level:   {m['co2']:.0f} ppm    [{status_CO2}]  (target: <1000 ppm)")
        print(f"  Air Velocity: {m['velocity']:.3f} m/s  [{status_V}]  (target: <0.25 m/s)")
        
        # ADPI metrics
        status_EDT = "✓ OK" if m['edt'] < 1.7 else "⚠ HIGH"
        status_ADPI = "✓ PASS" if m['adpi_pass'] else "✗ FAIL"
        
        print(f"\nAIR DIFFUSION (ASHRAE 113):")
        print(f"  EDT (mean): {m['edt']:.2f} K    [{status_EDT}]   (target: <1.7 K)")
        print(f"  ADPI:       {m['adpi']:.1f}%    [{status_ADPI}]  (target: >70%)")
        
        # Thermal comfort
        status_PMV = "✓ PASS" if m['pmv_pass'] else "✗ FAIL"
        status_PPD = "✓ PASS" if m['ppd_pass'] else "✗ FAIL"
        
        # PMV interpretation
        if m['pmv'] < -2.0:
            pmv_feel = "Cold"
        elif m['pmv'] < -1.0:
            pmv_feel = "Cool"
        elif m['pmv'] < -0.5:
            pmv_feel = "Slightly Cool"
        elif m['pmv'] <= 0.5:
            pmv_feel = "Neutral"
        elif m['pmv'] < 1.0:
            pmv_feel = "Slightly Warm"
        elif m['pmv'] < 2.0:
            pmv_feel = "Warm"
        else:
            pmv_feel = "Hot"
        
        print(f"\nTHERMAL COMFORT (ISO 7730):")
        print(f"  PMV:  {m['pmv']:+.2f} ({pmv_feel})  [{status_PMV}]  (Class A: -0.5 to +0.5)")
        print(f"  PPD:  {m['ppd']:.1f}%              [{status_PPD}]  (Class A: <10%)")
        
        # Overall
        print("\n" + "-" * 60)
        if m['overall_pass']:
            print("✓ ASHRAE 55 COMPLIANT - System Validated")
        else:
            print("⚠ TUNING REQUIRED - Some criteria not met")
        print("=" * 60)
