"""
Data Center Equipment and Transient Simulation Models

T4 Capability: Multi-Zone / Transient Data Center Thermal Analysis

Equipment Types:
- T4.01: CRAC (Computer Room Air Conditioner)
- T4.02: Server Rack (Variable Heat Load)
- T4.03: Raised Floor Plenum
- T4.04: Hot/Cold Aisle Containment
- T4.05: Transient Scenario Runner

Metrics:
- T4.06: RCI/SHI/RTI Thermal Indices
- T4.07: Critical Rack Detection
- T4.08: Time-to-Critical Prediction
- T4.09: Energy Balance Verification

Reference: ASHRAE TC 9.9 - Thermal Guidelines for Data Processing Environments
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
import math
import numpy as np

from .zone import Zone, Face


# =============================================================================
# ASHRAE THERMAL GUIDELINES
# =============================================================================

class ASHRAEClass(Enum):
    """ASHRAE TC 9.9 Equipment Environmental Classes."""
    A1 = auto()  # Mission-critical, 15-32°C
    A2 = auto()  # Less critical, 10-35°C
    A3 = auto()  # Extended envelope, 5-40°C
    A4 = auto()  # Extended envelope, 5-45°C
    
    @property
    def temp_range(self) -> Tuple[float, float]:
        """Return (min_temp_c, max_temp_c) for this class."""
        ranges = {
            ASHRAEClass.A1: (15.0, 32.0),
            ASHRAEClass.A2: (10.0, 35.0),
            ASHRAEClass.A3: (5.0, 40.0),
            ASHRAEClass.A4: (5.0, 45.0),
        }
        return ranges[self]
    
    @property
    def recommended_range(self) -> Tuple[float, float]:
        """ASHRAE recommended range (18-27°C for all classes)."""
        return (18.0, 27.0)


# =============================================================================
# T4.01: CRAC UNIT MODEL
# =============================================================================

class CRACState(Enum):
    """CRAC operating state."""
    RUNNING = auto()
    FAILED = auto()
    STANDBY = auto()
    STARTING = auto()  # Startup delay after failure


@dataclass
class CRACConfig:
    """
    Configuration for Computer Room Air Conditioner.
    
    T4.01: CRAC provides cooling to data center zones.
    
    Typical sizes:
    - Small: 20-30 kW
    - Medium: 50-80 kW
    - Large: 100-150 kW
    """
    name: str = "crac_1"
    crac_id: int = 0
    
    # Cooling capacity
    cooling_capacity_kw: float = 80.0  # Nominal cooling capacity
    
    # Supply air
    design_flow_m3s: float = 4.0       # Airflow capacity
    supply_temp_c: float = 15.0        # Supply air temperature
    
    # Control
    setpoint_temp_c: float = 24.0      # Return air setpoint
    throttling_range_c: float = 2.0    # Proportional band
    
    # Electrical
    fan_power_kw: float = 5.0          # Supply fan power
    compressor_power_kw: float = 25.0  # Compressor power at full load
    
    # Reliability
    startup_delay_s: float = 60.0      # Time to restart after failure
    
    # Position (for spatial analysis)
    position_x: float = 0.0
    position_y: float = 0.0


class CRACUnit:
    """
    Computer Room Air Conditioner.
    
    T4.01: Provides cooling airflow to data center via raised floor plenum.
    
    Operating Modes:
    - RUNNING: Normal operation, modulates capacity
    - FAILED: No cooling, no airflow (fault condition)
    - STANDBY: Ready to start (N+1 redundancy)
    - STARTING: Delayed restart after failure
    
    Physics:
    - Supply air temperature typically 15-18°C
    - Cooling capacity varies with return air temp
    - COP (Coefficient of Performance) ≈ 3-4 for typical units
    """
    
    def __init__(self, config: CRACConfig, zone: Optional[Zone] = None):
        self.config = config
        self.name = config.name
        self.crac_id = config.crac_id
        self.zone = zone
        
        # State
        self.state = CRACState.RUNNING
        self.output_fraction = 1.0  # 0-1 capacity fraction
        self.current_flow_m3s = config.design_flow_m3s
        self.supply_temp_c = config.supply_temp_c
        
        # Failure tracking
        self.failure_time: Optional[float] = None
        self.startup_remaining_s: float = 0.0
        
        # Energy tracking
        self.cooling_output_kw = 0.0
        self.total_power_kw = 0.0
        self.cooling_energy_kwh = 0.0
        self.electrical_energy_kwh = 0.0
        self.runtime_hours = 0.0
        
    def fail(self, time: float = 0.0):
        """
        Simulate CRAC failure.
        
        Args:
            time: Simulation time when failure occurs
        """
        self.state = CRACState.FAILED
        self.failure_time = time
        self.output_fraction = 0.0
        self.current_flow_m3s = 0.0
        self.cooling_output_kw = 0.0
        self.total_power_kw = 0.0
        
    def restart(self):
        """
        Begin restart sequence.
        
        CRAC enters STARTING state with delay before full operation.
        """
        if self.state == CRACState.FAILED:
            self.state = CRACState.STARTING
            self.startup_remaining_s = self.config.startup_delay_s
    
    def update(self, return_temp_c: float, dt: float = 1.0,
               current_time: float = 0.0) -> Tuple[float, float, float]:
        """
        Update CRAC state based on return air temperature.
        
        Args:
            return_temp_c: Return air temperature from zone
            dt: Timestep (seconds)
            current_time: Current simulation time
            
        Returns:
            (flow_m3s, supply_temp_c, cooling_kw)
        """
        cfg = self.config
        
        # Handle state transitions
        if self.state == CRACState.STARTING:
            self.startup_remaining_s -= dt
            if self.startup_remaining_s <= 0:
                self.state = CRACState.RUNNING
                self.startup_remaining_s = 0.0
        
        if self.state == CRACState.FAILED:
            # No output when failed
            return 0.0, 0.0, 0.0
        
        if self.state == CRACState.STANDBY:
            # No output when on standby
            return 0.0, cfg.supply_temp_c, 0.0
        
        if self.state == CRACState.STARTING:
            # Ramping up during startup
            startup_fraction = 1.0 - (self.startup_remaining_s / cfg.startup_delay_s)
            capacity_available = startup_fraction
        else:
            capacity_available = 1.0
        
        # Proportional control based on return air temp
        error = return_temp_c - cfg.setpoint_temp_c
        half_band = cfg.throttling_range_c / 2
        
        if error >= half_band:
            # Full cooling needed
            demand_fraction = 1.0
        elif error <= -half_band:
            # No cooling needed
            demand_fraction = 0.0
        else:
            # Proportional band
            demand_fraction = (error + half_band) / cfg.throttling_range_c
        
        # Actual output limited by available capacity
        self.output_fraction = min(demand_fraction, capacity_available)
        
        # Calculate cooling output
        # Simplified: cooling = capacity × load_fraction
        self.cooling_output_kw = cfg.cooling_capacity_kw * self.output_fraction
        
        # Airflow (assume constant volume for simplicity)
        self.current_flow_m3s = cfg.design_flow_m3s * capacity_available
        
        # Power consumption
        fan_power = cfg.fan_power_kw * capacity_available
        compressor_power = cfg.compressor_power_kw * self.output_fraction
        self.total_power_kw = fan_power + compressor_power
        
        # Energy tracking
        self.cooling_energy_kwh += self.cooling_output_kw * (dt / 3600)
        self.electrical_energy_kwh += self.total_power_kw * (dt / 3600)
        if self.state == CRACState.RUNNING:
            self.runtime_hours += dt / 3600
        
        return self.current_flow_m3s, cfg.supply_temp_c, self.cooling_output_kw
    
    def get_cop(self) -> float:
        """Get current Coefficient of Performance."""
        if self.total_power_kw > 0:
            return self.cooling_output_kw / self.total_power_kw
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current CRAC status."""
        return {
            'name': self.name,
            'crac_id': self.crac_id,
            'state': self.state.name,
            'output_fraction': self.output_fraction,
            'flow_m3s': self.current_flow_m3s,
            'supply_temp_c': self.config.supply_temp_c,
            'cooling_kw': self.cooling_output_kw,
            'power_kw': self.total_power_kw,
            'cop': self.get_cop(),
            'cooling_kwh': self.cooling_energy_kwh,
            'runtime_hours': self.runtime_hours
        }


# =============================================================================
# T4.02: SERVER RACK MODEL
# =============================================================================

class RackLoadProfile(Enum):
    """Typical rack load profiles."""
    CONSTANT = auto()
    SINUSOIDAL = auto()  # Day/night cycle
    STEP = auto()        # Sudden load change
    RANDOM = auto()      # Random fluctuation


@dataclass
class ServerRackConfig:
    """
    Configuration for a server rack.
    
    T4.02: Server racks are heat sources with variable load.
    
    Typical power densities:
    - Low: 3-5 kW/rack
    - Medium: 8-12 kW/rack
    - High: 15-25 kW/rack
    - Ultra-high: 30-50 kW/rack (GPU clusters)
    """
    name: str = "rack_1"
    rack_id: int = 0
    
    # Heat load
    rated_power_kw: float = 10.0       # Rated power (nameplate)
    current_load_fraction: float = 0.7  # Typical utilization 60-80%
    
    # Physical
    width_m: float = 0.6               # Standard 600mm
    depth_m: float = 1.2               # Standard 1000-1200mm
    height_m: float = 2.0              # 42U standard
    
    # Airflow (front-to-back)
    design_flow_m3s: float = 0.5       # Required airflow
    inlet_face: Face = Face.SOUTH      # Cold aisle side
    outlet_face: Face = Face.NORTH     # Hot aisle side
    
    # Porous media parameters (for CFD)
    porosity: float = 0.5              # Void fraction
    resistance_coefficient: float = 50.0  # Darcy resistance
    
    # Position
    position_x: float = 0.0            # Row position
    position_y: float = 0.0            # Aisle position
    row_id: int = 0
    
    # ASHRAE class
    ashrae_class: ASHRAEClass = ASHRAEClass.A1


class ServerRack:
    """
    Server Rack - Heat Source with Thermal Monitoring.
    
    T4.02: Models heat generation and thermal state.
    
    Key temperatures:
    - inlet_temp_c: Cold aisle air entering front of rack
    - outlet_temp_c: Hot air exhausted from rear
    - delta_T = Q / (ṁ × cp)
    
    Critical thresholds:
    - Warning: inlet > 27°C (ASHRAE recommended max)
    - Critical: inlet > 32°C (A1 allowable max)
    - Shutdown: inlet > 35°C (emergency)
    """
    
    def __init__(self, config: ServerRackConfig, zone: Optional[Zone] = None):
        self.config = config
        self.name = config.name
        self.rack_id = config.rack_id
        self.zone = zone
        
        # Current state
        self.load_fraction = config.current_load_fraction
        self.heat_output_kw = config.rated_power_kw * self.load_fraction
        
        # Temperatures
        self.inlet_temp_c = 22.0   # Cold aisle
        self.outlet_temp_c = 35.0  # Hot aisle
        
        # Airflow
        self.airflow_m3s = config.design_flow_m3s
        
        # Thermal history for time-to-critical
        self.temp_history: List[Tuple[float, float]] = []  # (time, inlet_temp)
        
        # Status flags
        self.warning_active = False
        self.critical_active = False
        self.shutdown_active = False
        
        # Energy tracking
        self.energy_kwh = 0.0
        
    def set_load(self, load_fraction: float):
        """Set current load as fraction of rated power."""
        self.load_fraction = max(0.0, min(1.0, load_fraction))
        self.heat_output_kw = self.config.rated_power_kw * self.load_fraction
        
    def update(self, inlet_temp_c: float, airflow_m3s: float,
               dt: float = 1.0, current_time: float = 0.0) -> float:
        """
        Update rack thermal state.
        
        Args:
            inlet_temp_c: Temperature of air entering rack (cold aisle)
            airflow_m3s: Airflow through rack
            dt: Timestep
            current_time: Current simulation time
            
        Returns:
            outlet_temp_c: Temperature of exhaust air
        """
        self.inlet_temp_c = inlet_temp_c
        self.airflow_m3s = airflow_m3s
        
        # Calculate temperature rise
        # Q = ṁ × cp × ΔT → ΔT = Q / (ṁ × cp)
        rho = 1.2  # kg/m³
        cp = 1005.0  # J/kg·K
        
        if airflow_m3s > 0.01:
            mass_flow = rho * airflow_m3s
            delta_t = (self.heat_output_kw * 1000) / (mass_flow * cp)
        else:
            # No airflow - thermal runaway!
            delta_t = 50.0  # Cap at 50K rise
            
        self.outlet_temp_c = inlet_temp_c + delta_t
        
        # Check thresholds
        rec_min, rec_max = self.config.ashrae_class.recommended_range
        allow_min, allow_max = self.config.ashrae_class.temp_range
        
        self.warning_active = inlet_temp_c > rec_max
        self.critical_active = inlet_temp_c > allow_max
        self.shutdown_active = inlet_temp_c > 35.0  # Emergency threshold
        
        # Record history for time-to-critical analysis
        self.temp_history.append((current_time, inlet_temp_c))
        
        # Limit history size
        if len(self.temp_history) > 1000:
            self.temp_history = self.temp_history[-500:]
        
        # Energy tracking
        self.energy_kwh += self.heat_output_kw * (dt / 3600)
        
        return self.outlet_temp_c
    
    def predict_time_to_critical(self, critical_temp_c: float = 35.0) -> Optional[float]:
        """
        T4.08: Predict time until inlet reaches critical temperature.
        
        Uses linear extrapolation from recent temperature trend.
        
        Returns:
            Estimated seconds until critical, or None if cooling or already critical
        """
        if len(self.temp_history) < 10:
            return None
            
        if self.inlet_temp_c >= critical_temp_c:
            return 0.0  # Already critical
            
        # Get last 10 samples
        recent = self.temp_history[-10:]
        times = np.array([t for t, _ in recent])
        temps = np.array([T for _, T in recent])
        
        # Linear regression
        if len(times) < 2:
            return None
            
        slope = np.polyfit(times, temps, 1)[0]
        
        if slope <= 0:
            return None  # Cooling or stable
            
        # Time to reach critical
        delta_temp = critical_temp_c - self.inlet_temp_c
        time_to_critical = delta_temp / slope
        
        return max(0.0, time_to_critical)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rack status."""
        ttc = self.predict_time_to_critical()
        return {
            'name': self.name,
            'rack_id': self.rack_id,
            'row_id': self.config.row_id,
            'heat_kw': self.heat_output_kw,
            'load_fraction': self.load_fraction,
            'inlet_temp_c': self.inlet_temp_c,
            'outlet_temp_c': self.outlet_temp_c,
            'delta_t': self.outlet_temp_c - self.inlet_temp_c,
            'airflow_m3s': self.airflow_m3s,
            'warning': self.warning_active,
            'critical': self.critical_active,
            'shutdown': self.shutdown_active,
            'time_to_critical_s': ttc,
            'energy_kwh': self.energy_kwh
        }


# =============================================================================
# T4.03: RAISED FLOOR PLENUM
# =============================================================================

@dataclass
class PlenumConfig:
    """
    Configuration for raised floor plenum.
    
    T4.03: Models underfloor air distribution.
    """
    name: str = "plenum_1"
    
    # Geometry
    length_m: float = 25.0         # Data center length
    width_m: float = 20.0          # Data center width
    height_m: float = 0.6          # Plenum depth (typical 0.4-0.9m)
    
    # Perforated tiles
    tile_size_m: float = 0.6       # Standard 600mm tile
    open_area_fraction: float = 0.25  # 25% open area typical
    
    # Number of tiles per row/column
    n_tiles_x: int = 20
    n_tiles_y: int = 15
    
    # Tile locations (list of (x, y) indices that are perforated)
    perforated_tiles: List[Tuple[int, int]] = field(default_factory=list)


class RaisedFloorPlenum:
    """
    Raised Floor Plenum - Underfloor Air Distribution.
    
    T4.03: Models pressure distribution and tile airflow.
    
    Physics:
    - CRAC units pressurize plenum
    - Air exits through perforated tiles
    - Flow ∝ √ΔP (orifice equation)
    - Uneven distribution leads to hot spots
    """
    
    def __init__(self, config: PlenumConfig):
        self.config = config
        self.name = config.name
        
        # Plenum state
        self.pressure_pa = 10.0  # Typical plenum pressure
        self.temperature_c = 15.0  # Supply air temp
        
        # Total flow from all CRACs
        self.total_supply_m3s = 0.0
        
        # Connected CRACs
        self.cracs: List[CRACUnit] = []
        
        # Tile flow rates (2D grid)
        self.tile_flows = np.zeros((config.n_tiles_x, config.n_tiles_y))
        
        # Initialize perforated tiles (default: cold aisle pattern)
        if not config.perforated_tiles:
            self._init_default_tiles()
    
    def _init_default_tiles(self):
        """Set up default perforated tile pattern (cold aisle)."""
        cfg = self.config
        # Every 3rd row is a cold aisle with perforated tiles
        for i in range(cfg.n_tiles_x):
            for j in range(cfg.n_tiles_y):
                if j % 3 == 1:  # Cold aisle rows
                    cfg.perforated_tiles.append((i, j))
    
    def add_crac(self, crac: CRACUnit):
        """Add CRAC unit feeding this plenum."""
        self.cracs.append(crac)
    
    def update(self, dt: float = 1.0) -> Dict[Tuple[int, int], float]:
        """
        Update plenum state and calculate tile flows.
        
        Returns:
            Dict mapping tile (x, y) indices to flow rates (m³/s)
        """
        cfg = self.config
        
        # Sum CRAC supplies
        self.total_supply_m3s = sum(c.current_flow_m3s for c in self.cracs)
        self.temperature_c = cfg.perforated_tiles and np.mean([
            c.config.supply_temp_c for c in self.cracs if c.state == CRACState.RUNNING
        ]) if any(c.state == CRACState.RUNNING for c in self.cracs) else 15.0
        
        # Estimate plenum pressure from total flow
        # Simplified: P ∝ Q² (assuming fixed tile resistance)
        n_tiles = len(cfg.perforated_tiles) or 1
        tile_area = cfg.tile_size_m ** 2 * cfg.open_area_fraction
        
        # Orifice equation: Q = Cd × A × √(2ΔP/ρ)
        # Rearranged: ΔP = (Q / (Cd × A))² × ρ / 2
        Cd = 0.6  # Discharge coefficient
        rho = 1.2  # kg/m³
        
        if n_tiles > 0 and tile_area > 0:
            q_per_tile = self.total_supply_m3s / n_tiles
            self.pressure_pa = (q_per_tile / (Cd * tile_area)) ** 2 * rho / 2
            self.pressure_pa = min(self.pressure_pa, 100.0)  # Cap at 100 Pa
        else:
            self.pressure_pa = 0.0
        
        # Calculate individual tile flows
        # For now, assume uniform distribution
        tile_flow_dict = {}
        for (i, j) in cfg.perforated_tiles:
            if 0 <= i < cfg.n_tiles_x and 0 <= j < cfg.n_tiles_y:
                flow = self.total_supply_m3s / max(n_tiles, 1)
                self.tile_flows[i, j] = flow
                tile_flow_dict[(i, j)] = flow
        
        return tile_flow_dict
    
    def get_status(self) -> Dict[str, Any]:
        """Get plenum status."""
        return {
            'name': self.name,
            'pressure_pa': self.pressure_pa,
            'temperature_c': self.temperature_c,
            'total_supply_m3s': self.total_supply_m3s,
            'n_cracs_running': sum(1 for c in self.cracs if c.state == CRACState.RUNNING),
            'n_perforated_tiles': len(self.config.perforated_tiles)
        }


# =============================================================================
# T4.04: CONTAINMENT MODEL
# =============================================================================

class ContainmentType(Enum):
    """Types of aisle containment."""
    HOT_AISLE = auto()   # Contain hot exhaust
    COLD_AISLE = auto()  # Contain cold supply
    CHIMNEY = auto()     # Vertical exhaust cabinet


@dataclass
class ContainmentConfig:
    """
    Configuration for aisle containment.
    
    T4.04: Containment improves cooling efficiency by separating hot/cold air.
    """
    name: str = "containment_1"
    containment_type: ContainmentType = ContainmentType.HOT_AISLE
    
    # Geometry
    length_m: float = 20.0         # Aisle length
    width_m: float = 1.2           # Aisle width
    height_m: float = 2.5          # Containment height
    
    # Leakage
    leakage_fraction: float = 0.05  # 5% air bypass
    
    # End caps
    has_end_doors: bool = True
    door_open_fraction: float = 0.0  # 0 = closed, 1 = open


class AisleContainment:
    """
    Aisle Containment - Hot or Cold.
    
    T4.04: Physical barrier separating supply and return air.
    
    Benefits:
    - Hot aisle: Prevents hot air recirculation to rack inlets
    - Cold aisle: Prevents cold air bypass to returns
    - Both improve PUE and reduce hot spots
    
    Leakage model:
    - Some air bypasses through gaps, cable cutouts, etc.
    - End doors may be propped open (operational issue)
    """
    
    def __init__(self, config: ContainmentConfig):
        self.config = config
        self.name = config.name
        
        # Associated racks (in this aisle)
        self.racks: List[ServerRack] = []
        
        # Thermal state
        self.aisle_temp_c = 22.0
        self.bypass_flow_m3s = 0.0
        
    def add_rack(self, rack: ServerRack):
        """Add a rack to this aisle."""
        self.racks.append(rack)
    
    def update(self, supply_temp_c: float, return_temp_c: float,
               total_flow_m3s: float) -> float:
        """
        Calculate aisle temperature with leakage effects.
        
        Args:
            supply_temp_c: Cold aisle / supply temperature
            return_temp_c: Hot aisle / return temperature
            total_flow_m3s: Total airflow through aisle
            
        Returns:
            Effective aisle temperature after mixing
        """
        cfg = self.config
        
        # Calculate leakage
        leakage = cfg.leakage_fraction + cfg.door_open_fraction * 0.2
        leakage = min(leakage, 0.5)  # Cap at 50%
        
        self.bypass_flow_m3s = total_flow_m3s * leakage
        
        if cfg.containment_type == ContainmentType.HOT_AISLE:
            # Hot aisle: some cold air leaks in
            self.aisle_temp_c = (
                return_temp_c * (1 - leakage) + 
                supply_temp_c * leakage
            )
        else:
            # Cold aisle: some hot air leaks in
            self.aisle_temp_c = (
                supply_temp_c * (1 - leakage) + 
                return_temp_c * leakage
            )
        
        return self.aisle_temp_c
    
    def get_status(self) -> Dict[str, Any]:
        """Get containment status."""
        return {
            'name': self.name,
            'type': self.config.containment_type.name,
            'aisle_temp_c': self.aisle_temp_c,
            'bypass_flow_m3s': self.bypass_flow_m3s,
            'n_racks': len(self.racks),
            'door_open': self.config.door_open_fraction > 0
        }


# =============================================================================
# T4.06: THERMAL METRICS (RCI, SHI, RTI)
# =============================================================================

def compute_rci(rack_inlet_temps: List[float], 
                t_rec_min: float = 18.0, t_rec_max: float = 27.0) -> Tuple[float, float]:
    """
    Compute Rack Cooling Index (RCI) per ASHRAE.
    
    T4.06: RCI measures how well cooling meets ASHRAE guidelines.
    
    RCI_Hi = (1 - Σmax(0, Ti - Tmax_rec) / (n × (Tmax_allow - Tmax_rec))) × 100%
    RCI_Lo = (1 - Σmax(0, Tmin_rec - Ti) / (n × (Tmin_rec - Tmin_allow))) × 100%
    
    Args:
        rack_inlet_temps: List of rack inlet temperatures (°C)
        t_rec_min/max: ASHRAE recommended range
        
    Returns:
        (RCI_Hi, RCI_Lo) - both as percentages (0-100)
    """
    if not rack_inlet_temps:
        return 100.0, 100.0
    
    n = len(rack_inlet_temps)
    temps = np.array(rack_inlet_temps)
    
    # Allowable range (A1 class)
    t_allow_min = 15.0
    t_allow_max = 32.0
    
    # RCI High (over-temperature)
    over_temp = np.maximum(0, temps - t_rec_max)
    rci_hi = (1 - np.sum(over_temp) / (n * (t_allow_max - t_rec_max) + 1e-6)) * 100
    rci_hi = max(0, min(100, rci_hi))
    
    # RCI Low (under-temperature)
    under_temp = np.maximum(0, t_rec_min - temps)
    rci_lo = (1 - np.sum(under_temp) / (n * (t_rec_min - t_allow_min) + 1e-6)) * 100
    rci_lo = max(0, min(100, rci_lo))
    
    return rci_hi, rci_lo


def compute_shi(supply_temp_c: float, avg_inlet_temp_c: float,
                avg_outlet_temp_c: float) -> float:
    """
    Compute Supply Heat Index (SHI).
    
    T4.06: SHI measures heat gain in supply air before reaching racks.
    
    SHI = (Tintake - Tsupply) / (Texhaust - Tsupply)
    
    Lower is better. SHI = 0 means no supply air heating (perfect separation).
    
    Returns:
        SHI as fraction (0-1)
    """
    delta_total = avg_outlet_temp_c - supply_temp_c
    if delta_total <= 0:
        return 0.0
    
    delta_supply = avg_inlet_temp_c - supply_temp_c
    shi = delta_supply / delta_total
    return max(0, min(1, shi))


def compute_rti(supply_temp_c: float, avg_inlet_temp_c: float,
                avg_outlet_temp_c: float) -> float:
    """
    Compute Return Temperature Index (RTI).
    
    T4.06: RTI measures short-circuiting of return air.
    
    RTI = (Texhaust - Tintake) / (Texhaust - Tsupply)
    
    Higher is better. RTI = 1 means no bypass (all heat removed).
    
    Returns:
        RTI as fraction (0-1)
    """
    delta_total = avg_outlet_temp_c - supply_temp_c
    if delta_total <= 0:
        return 1.0
    
    delta_rack = avg_outlet_temp_c - avg_inlet_temp_c
    rti = delta_rack / delta_total
    return max(0, min(1, rti))


# =============================================================================
# T4.05: TRANSIENT SCENARIO RUNNER
# =============================================================================

@dataclass
class TransientEvent:
    """An event that occurs during transient simulation."""
    time_s: float                  # When event occurs
    event_type: str               # Type: 'crac_fail', 'crac_restart', 'load_change'
    target_id: int                # ID of affected equipment
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransientConfig:
    """
    Configuration for transient data center simulation.
    
    T4.05: Time-stepping simulation of failure scenarios.
    """
    name: str = "crac_failure_scenario"
    
    # Timing
    duration_s: float = 300.0      # Total simulation time (5 min default)
    dt: float = 1.0                # Timestep (seconds)
    
    # Events
    events: List[TransientEvent] = field(default_factory=list)
    
    # Thresholds
    warning_temp_c: float = 27.0
    critical_temp_c: float = 32.0
    shutdown_temp_c: float = 35.0
    
    # Energy balance tolerance
    energy_balance_tolerance: float = 0.01  # 1%


class DataCenterSimulator:
    """
    Transient Data Center Thermal Simulator.
    
    T4.05: Runs time-stepping simulation with scheduled events.
    
    Usage:
        sim = DataCenterSimulator(config)
        sim.add_cracs([crac1, crac2, ...])
        sim.add_racks([rack1, rack2, ...])
        sim.add_event(TransientEvent(time_s=0, event_type='crac_fail', target_id=4))
        results = sim.run()
    """
    
    def __init__(self, config: TransientConfig):
        self.config = config
        self.name = config.name
        
        # Equipment
        self.cracs: Dict[int, CRACUnit] = {}
        self.racks: Dict[int, ServerRack] = {}
        self.plenum: Optional[RaisedFloorPlenum] = None
        self.containments: List[AisleContainment] = []
        
        # Event queue
        self.events = sorted(config.events, key=lambda e: e.time_s)
        
        # Results storage
        self.time_history: List[float] = []
        self.rack_temp_history: Dict[int, List[float]] = {}
        self.crac_status_history: Dict[int, List[str]] = {}
        self.cooling_power_history: List[float] = []
        self.heat_load_history: List[float] = []
        
        # Current time
        self.current_time = 0.0
        
    def add_crac(self, crac: CRACUnit):
        """Add a CRAC unit."""
        self.cracs[crac.crac_id] = crac
        self.crac_status_history[crac.crac_id] = []
        
    def add_cracs(self, cracs: List[CRACUnit]):
        """Add multiple CRACs."""
        for crac in cracs:
            self.add_crac(crac)
    
    def add_rack(self, rack: ServerRack):
        """Add a server rack."""
        self.racks[rack.rack_id] = rack
        self.rack_temp_history[rack.rack_id] = []
        
    def add_racks(self, racks: List[ServerRack]):
        """Add multiple racks."""
        for rack in racks:
            self.add_rack(rack)
    
    def set_plenum(self, plenum: RaisedFloorPlenum):
        """Set the raised floor plenum."""
        self.plenum = plenum
        for crac in self.cracs.values():
            plenum.add_crac(crac)
    
    def add_event(self, event: TransientEvent):
        """Add a transient event."""
        self.events.append(event)
        self.events.sort(key=lambda e: e.time_s)
    
    def _process_events(self, time_s: float):
        """Process any events scheduled for this time."""
        while self.events and self.events[0].time_s <= time_s:
            event = self.events.pop(0)
            
            if event.event_type == 'crac_fail':
                if event.target_id in self.cracs:
                    self.cracs[event.target_id].fail(time_s)
                    
            elif event.event_type == 'crac_restart':
                if event.target_id in self.cracs:
                    self.cracs[event.target_id].restart()
                    
            elif event.event_type == 'load_change':
                if event.target_id in self.racks:
                    new_load = event.parameters.get('load_fraction', 0.7)
                    self.racks[event.target_id].set_load(new_load)
    
    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """
        Advance simulation by one timestep.
        
        Returns:
            Current state metrics
        """
        if dt is None:
            dt = self.config.dt
        
        # Process scheduled events
        self._process_events(self.current_time)
        
        # Update CRACs
        total_cooling_kw = 0.0
        total_supply_m3s = 0.0
        
        for crac in self.cracs.values():
            # Get average return temp from racks
            if self.racks:
                avg_outlet = np.mean([r.outlet_temp_c for r in self.racks.values()])
            else:
                avg_outlet = 25.0
            
            flow, supply_t, cooling = crac.update(avg_outlet, dt, self.current_time)
            total_cooling_kw += cooling
            total_supply_m3s += flow
            self.crac_status_history[crac.crac_id].append(crac.state.name)
        
        # Update plenum
        if self.plenum:
            self.plenum.update(dt)
            supply_temp = self.plenum.temperature_c
        else:
            supply_temp = 15.0
        
        # Update racks
        total_heat_kw = 0.0
        n_racks = len(self.racks)
        
        for rack in self.racks.values():
            # Simplified: assume uniform cold aisle distribution
            inlet_temp = supply_temp
            
            # Add warming due to insufficient cooling
            if total_supply_m3s > 0 and n_racks > 0:
                rack_flow = total_supply_m3s / n_racks
            else:
                rack_flow = 0.01  # Minimal natural convection
            
            # Account for heat accumulation if cooling insufficient
            heat_deficit = (sum(r.heat_output_kw for r in self.racks.values()) - 
                          total_cooling_kw)
            if heat_deficit > 0 and n_racks > 0:
                # Temperature rises due to insufficient cooling
                # Simplified thermal mass model
                room_volume = 500.0 * 4.0  # Approximate m³
                rho_cp = 1.2 * 1005  # J/m³·K
                temp_rise = (heat_deficit * 1000 * dt) / (room_volume * rho_cp)
                inlet_temp += temp_rise * 0.5  # Assume 50% reaches rack inlet
            
            rack.update(inlet_temp, rack_flow, dt, self.current_time)
            total_heat_kw += rack.heat_output_kw
            self.rack_temp_history[rack.rack_id].append(rack.inlet_temp_c)
        
        # Record history
        self.time_history.append(self.current_time)
        self.cooling_power_history.append(total_cooling_kw)
        self.heat_load_history.append(total_heat_kw)
        
        # Advance time
        self.current_time += dt
        
        return self._get_current_metrics()
    
    def run(self, progress_callback: Optional[Callable[[float, Dict], None]] = None
           ) -> Dict[str, Any]:
        """
        Run complete transient simulation.
        
        Args:
            progress_callback: Optional callback(time, metrics) for progress updates
            
        Returns:
            Complete simulation results
        """
        cfg = self.config
        n_steps = int(cfg.duration_s / cfg.dt)
        
        for i in range(n_steps):
            metrics = self.step(cfg.dt)
            
            if progress_callback and i % 10 == 0:
                progress = self.current_time / cfg.duration_s
                progress_callback(progress, metrics)
        
        return self.get_results()
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current simulation metrics."""
        inlet_temps = [r.inlet_temp_c for r in self.racks.values()]
        outlet_temps = [r.outlet_temp_c for r in self.racks.values()]
        
        if inlet_temps:
            avg_inlet = np.mean(inlet_temps)
            max_inlet = np.max(inlet_temps)
            avg_outlet = np.mean(outlet_temps)
            supply_temp = self.plenum.temperature_c if self.plenum else 15.0
            
            rci_hi, rci_lo = compute_rci(inlet_temps)
            shi = compute_shi(supply_temp, avg_inlet, avg_outlet)
            rti = compute_rti(supply_temp, avg_inlet, avg_outlet)
        else:
            avg_inlet = max_inlet = avg_outlet = 22.0
            rci_hi = rci_lo = 100.0
            shi = 0.0
            rti = 1.0
        
        # Count critical racks
        n_warning = sum(1 for r in self.racks.values() if r.warning_active)
        n_critical = sum(1 for r in self.racks.values() if r.critical_active)
        
        return {
            'time_s': self.current_time,
            'avg_inlet_temp_c': avg_inlet,
            'max_inlet_temp_c': max_inlet,
            'rci_hi': rci_hi,
            'rci_lo': rci_lo,
            'shi': shi,
            'rti': rti,
            'n_warning': n_warning,
            'n_critical': n_critical,
            'total_cooling_kw': self.cooling_power_history[-1] if self.cooling_power_history else 0,
            'total_heat_kw': self.heat_load_history[-1] if self.heat_load_history else 0
        }
    
    def detect_critical_racks(self) -> List[Dict[str, Any]]:
        """
        T4.07: Identify all racks exceeding critical temperature.
        
        Returns:
            List of critical rack status dicts
        """
        critical = []
        for rack in self.racks.values():
            if rack.critical_active or rack.shutdown_active:
                status = rack.get_status()
                status['severity'] = 'SHUTDOWN' if rack.shutdown_active else 'CRITICAL'
                critical.append(status)
        
        return sorted(critical, key=lambda x: x['inlet_temp_c'], reverse=True)
    
    def get_time_to_critical_all(self) -> Dict[int, Optional[float]]:
        """
        T4.08: Get time-to-critical for all racks.
        
        Returns:
            Dict mapping rack_id to seconds until critical (None if stable)
        """
        return {
            rack_id: rack.predict_time_to_critical()
            for rack_id, rack in self.racks.items()
        }
    
    def verify_energy_balance(self) -> Tuple[bool, float]:
        """
        T4.09: Verify energy balance across simulation.
        
        Energy in (IT load) should equal energy out (cooling) ± storage.
        
        Returns:
            (passed, imbalance_fraction)
        """
        if not self.heat_load_history or not self.cooling_power_history:
            return True, 0.0
        
        # Integrate over time
        dt = self.config.dt
        total_heat = sum(self.heat_load_history) * dt / 3600  # kWh
        total_cooling = sum(self.cooling_power_history) * dt / 3600  # kWh
        
        if total_heat > 0:
            imbalance = abs(total_heat - total_cooling) / total_heat
        else:
            imbalance = 0.0
        
        passed = imbalance <= self.config.energy_balance_tolerance
        return passed, imbalance
    
    def get_results(self) -> Dict[str, Any]:
        """Get complete simulation results."""
        inlet_temps = [r.inlet_temp_c for r in self.racks.values()]
        
        # Compute final metrics
        if inlet_temps:
            rci_hi, rci_lo = compute_rci(inlet_temps)
            supply_t = self.plenum.temperature_c if self.plenum else 15.0
            avg_inlet = np.mean(inlet_temps)
            avg_outlet = np.mean([r.outlet_temp_c for r in self.racks.values()])
            shi = compute_shi(supply_t, avg_inlet, avg_outlet)
            rti = compute_rti(supply_t, avg_inlet, avg_outlet)
        else:
            rci_hi = rci_lo = 100.0
            shi = 0.0
            rti = 1.0
        
        energy_ok, imbalance = self.verify_energy_balance()
        
        return {
            'scenario_name': self.name,
            'duration_s': self.current_time,
            'n_cracs': len(self.cracs),
            'n_racks': len(self.racks),
            
            # Thermal state
            'final_metrics': self._get_current_metrics(),
            'max_inlet_temp_c': max(inlet_temps) if inlet_temps else 22.0,
            'min_inlet_temp_c': min(inlet_temps) if inlet_temps else 22.0,
            
            # ASHRAE metrics
            'rci_hi': rci_hi,
            'rci_lo': rci_lo,
            'shi': shi,
            'rti': rti,
            
            # Critical analysis
            'critical_racks': self.detect_critical_racks(),
            'n_critical': sum(1 for r in self.racks.values() if r.critical_active),
            'time_to_critical': self.get_time_to_critical_all(),
            
            # Energy balance
            'energy_balance_passed': energy_ok,
            'energy_imbalance_fraction': imbalance,
            
            # Time series (sampled)
            'time_history': self.time_history[::10],  # Every 10th point
            'rack_temp_history': {
                k: v[::10] for k, v in self.rack_temp_history.items()
            },
            
            # Energy totals
            'total_cooling_kwh': sum(c.cooling_energy_kwh for c in self.cracs.values()),
            'total_rack_energy_kwh': sum(r.energy_kwh for r in self.racks.values())
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_data_center(n_cracs: int = 12, n_racks: int = 200,
                       rack_power_kw: float = 10.0,
                       crac_capacity_kw: float = 80.0) -> DataCenterSimulator:
    """
    Create a typical data center configuration.
    
    Args:
        n_cracs: Number of CRAC units
        n_racks: Number of server racks
        rack_power_kw: Power per rack
        crac_capacity_kw: Cooling per CRAC
        
    Returns:
        Configured DataCenterSimulator
    """
    config = TransientConfig(
        name=f"datacenter_{n_racks}racks",
        duration_s=300.0,
        dt=1.0
    )
    sim = DataCenterSimulator(config)
    
    # Create CRACs
    for i in range(n_cracs):
        crac = CRACUnit(CRACConfig(
            name=f"CRAC_{i+1}",
            crac_id=i,
            cooling_capacity_kw=crac_capacity_kw
        ))
        sim.add_crac(crac)
    
    # Create racks (varied load)
    for i in range(n_racks):
        load_variation = 0.6 + 0.3 * np.random.random()  # 60-90% load
        rack = ServerRack(ServerRackConfig(
            name=f"Rack_{i+1}",
            rack_id=i,
            rated_power_kw=rack_power_kw,
            current_load_fraction=load_variation,
            row_id=i // 10
        ))
        sim.add_rack(rack)
    
    # Create plenum
    plenum = RaisedFloorPlenum(PlenumConfig(name="main_plenum"))
    sim.set_plenum(plenum)
    
    return sim


def run_crac_failure_scenario(sim: DataCenterSimulator, 
                              failed_crac_id: int = 4,
                              failure_time_s: float = 0.0,
                              restart_time_s: Optional[float] = None
                              ) -> Dict[str, Any]:
    """
    Run a CRAC failure scenario.
    
    T4.05: Standard test case for data center resilience.
    
    Args:
        sim: Configured DataCenterSimulator
        failed_crac_id: ID of CRAC to fail
        failure_time_s: When failure occurs
        restart_time_s: When to restart (None = no restart)
        
    Returns:
        Simulation results
    """
    # Schedule failure
    sim.add_event(TransientEvent(
        time_s=failure_time_s,
        event_type='crac_fail',
        target_id=failed_crac_id
    ))
    
    # Schedule restart if specified
    if restart_time_s is not None:
        sim.add_event(TransientEvent(
            time_s=restart_time_s,
            event_type='crac_restart',
            target_id=failed_crac_id
        ))
    
    # Run simulation
    return sim.run()
