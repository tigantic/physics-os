"""
HVAC Equipment Models

T3 Capability: Model realistic HVAC terminal units and air handling equipment.

Equipment Types:
- T3.06: VAV (Variable Air Volume) Terminal Box
- T3.07: Fan-Powered Box (Series/Parallel)
- T3.08: Ductwork with Pressure Drop
- T3.09: AHU (Air Handling Unit) - Simplified

All equipment connects to Zones via their inlet/outlet faces.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import math

from .zone import Zone, Face


class EquipmentType(Enum):
    """Types of HVAC terminal equipment."""
    VAV_COOLING_ONLY = auto()
    VAV_WITH_REHEAT = auto()
    FAN_POWERED_SERIES = auto()
    FAN_POWERED_PARALLEL = auto()
    CAV = auto()  # Constant Air Volume


@dataclass
class VAVConfig:
    """
    Configuration for a Variable Air Volume terminal box.
    
    T3.06: VAV boxes modulate airflow to maintain zone temperature.
    
    Physics:
        - Damper position controls flow: Q = Q_design * damper_position
        - Reheat coil adds heat when needed (VAV with reheat)
        - Minimum flow setting prevents stagnation
    """
    name: str = "vav_1"
    
    # Design conditions
    design_flow_m3s: float = 0.5      # Design max flow (m³/s)
    min_flow_fraction: float = 0.3    # Minimum flow as fraction of design
    
    # Control
    setpoint_temp_c: float = 22.0     # Zone temperature setpoint
    throttling_range_c: float = 2.0   # ±°C band for proportional control
    
    # Equipment
    has_reheat: bool = True
    reheat_capacity_w: float = 2000.0  # Reheat coil capacity (Watts)
    supply_temp_c: float = 13.0       # Supply air temperature from AHU
    
    # Loss coefficients (for pressure drop)
    pressure_drop_pa: float = 50.0    # Pressure drop at design flow
    
    def __post_init__(self):
        if self.min_flow_fraction < 0 or self.min_flow_fraction > 1:
            raise ValueError(f"min_flow_fraction must be in [0, 1], got {self.min_flow_fraction}")
        if self.design_flow_m3s <= 0:
            raise ValueError(f"design_flow_m3s must be positive")


class VAVTerminal:
    """
    Variable Air Volume terminal box.
    
    T3.06: Controls zone airflow based on temperature demand.
    
    Operation Modes:
        1. Cooling: Damper modulates 100% → min_flow as zone cools
        2. Heating (if reheat): Damper at min_flow, reheat coil energizes
        3. Deadband: Damper at position to maintain setpoint
    
    Usage:
        vav = VAVTerminal(config, zone)
        vav.update(zone_temp_c=23.5)  # Returns (flow_m3s, supply_temp_c, reheat_watts)
    """
    
    def __init__(self, config: VAVConfig, zone: Zone, face: Face = Face.TOP):
        self.config = config
        self.name = config.name
        self.zone = zone
        self.face = face
        
        # State
        self.damper_position = 1.0  # 0 = min, 1 = max
        self.current_flow_m3s = config.design_flow_m3s
        self.reheat_output_w = 0.0
        self.mode = "cooling"
        
        # Supply temperature (from AHU)
        self.supply_temp_c = config.supply_temp_c
        
        # Statistics
        self.runtime_hours = 0.0
        self.energy_kwh = 0.0
        
    def update(self, zone_temp_c: float, dt: float = 0.01) -> Tuple[float, float, float]:
        """
        Update VAV state based on zone temperature.
        
        Args:
            zone_temp_c: Current zone air temperature
            dt: Timestep (seconds)
            
        Returns:
            (flow_m3s, discharge_temp_c, reheat_watts)
        """
        cfg = self.config
        
        # Temperature error (positive = too warm, needs cooling)
        error = zone_temp_c - cfg.setpoint_temp_c
        
        # Proportional control
        half_band = cfg.throttling_range_c / 2
        
        if error > half_band:
            # Full cooling mode
            self.damper_position = 1.0
            self.reheat_output_w = 0.0
            self.mode = "cooling_max"
            
        elif error > -half_band:
            # Proportional band - modulate damper
            # Map error from [-half_band, +half_band] to [min_flow, 1.0]
            normalized_error = (error + half_band) / cfg.throttling_range_c
            self.damper_position = cfg.min_flow_fraction + (1 - cfg.min_flow_fraction) * normalized_error
            self.reheat_output_w = 0.0
            self.mode = "modulating"
            
        else:
            # Heating mode (too cold)
            self.damper_position = cfg.min_flow_fraction
            
            if cfg.has_reheat:
                # Proportional reheat
                heating_demand = (-error - half_band) / half_band  # 0 to 1+
                self.reheat_output_w = min(cfg.reheat_capacity_w, 
                                           heating_demand * cfg.reheat_capacity_w)
                self.mode = "heating"
            else:
                self.reheat_output_w = 0.0
                self.mode = "min_flow"
        
        # Calculate flow
        self.current_flow_m3s = cfg.design_flow_m3s * self.damper_position
        
        # Calculate discharge temperature
        if self.reheat_output_w > 0 and self.current_flow_m3s > 0:
            # Q = m_dot * cp * ΔT
            # ΔT = Q / (m_dot * cp) = Q / (ρ * V_dot * cp)
            rho = 1.2  # kg/m³
            cp = 1005.0  # J/kg·K
            delta_t = self.reheat_output_w / (rho * self.current_flow_m3s * cp)
            discharge_temp_c = cfg.supply_temp_c + delta_t
        else:
            discharge_temp_c = cfg.supply_temp_c
        
        # Energy tracking
        self.energy_kwh += (self.reheat_output_w / 1000) * (dt / 3600)
        
        return self.current_flow_m3s, discharge_temp_c, self.reheat_output_w
    
    def apply_to_zone(self, dt: float = 0.01):
        """
        Apply VAV output to the connected zone.
        
        Updates zone inlet velocity and temperature based on VAV state.
        """
        zone_temp_c = self.zone.get_metrics()['temperature_c']
        flow_m3s, discharge_temp_c, reheat_w = self.update(zone_temp_c, dt)
        
        # Calculate inlet velocity from flow rate
        # Assume supply area is 10% of face area
        face_area = self._get_face_area()
        supply_area = face_area * 0.1  # Typical diffuser is ~10% of ceiling
        velocity = flow_m3s / max(supply_area, 0.01)
        
        # Update inlet (already configured)
        if self.face in self.zone.inlet_faces:
            self.zone.inlet_faces[self.face]['velocity'] = velocity
            self.zone.inlet_faces[self.face]['temperature'] = discharge_temp_c + 273.15
    
    def _get_face_area(self) -> float:
        """Get the area of the connection face."""
        if self.face in (Face.WEST, Face.EAST):
            return self.zone.ly * self.zone.lz
        elif self.face in (Face.SOUTH, Face.NORTH):
            return self.zone.lx * self.zone.lz
        else:  # TOP, BOTTOM
            return self.zone.lx * self.zone.ly
    
    def get_status(self) -> Dict[str, Any]:
        """Get current VAV status."""
        return {
            'name': self.name,
            'mode': self.mode,
            'damper_position': self.damper_position,
            'flow_m3s': self.current_flow_m3s,
            'flow_cfm': self.current_flow_m3s * 2118.88,  # Convert to CFM
            'reheat_w': self.reheat_output_w,
            'energy_kwh': self.energy_kwh
        }


@dataclass
class FanPoweredBoxConfig:
    """
    Configuration for a Fan-Powered VAV box.
    
    T3.07: Fan-powered boxes recirculate plenum air for heating mode.
    
    Types:
        - Series: Fan always running, mixes primary and plenum air
        - Parallel: Fan runs only during heating, supplements primary air
    """
    name: str = "fpb_1"
    box_type: str = "parallel"  # "series" or "parallel"
    
    # Primary air from AHU
    primary_max_flow_m3s: float = 0.3
    primary_min_flow_m3s: float = 0.1
    primary_supply_temp_c: float = 13.0
    
    # Secondary (plenum) air
    fan_flow_m3s: float = 0.2  # Constant fan flow when running
    fan_power_w: float = 150.0  # Fan motor power
    
    # Control
    setpoint_temp_c: float = 22.0
    throttling_range_c: float = 2.0
    
    # Heating
    has_reheat: bool = True
    reheat_capacity_w: float = 3000.0


class FanPoweredBox:
    """
    Fan-Powered VAV Terminal Box.
    
    T3.07: Provides superior heating performance via plenum air recirculation.
    
    Series Box:
        - Fan always runs, draws primary + plenum air
        - Good for spaces needing constant air motion
        - Higher energy use
        
    Parallel Box:
        - Fan only runs during heating
        - Primary air flows normally during cooling
        - More energy efficient
    """
    
    def __init__(self, config: FanPoweredBoxConfig, zone: Zone, face: Face = Face.TOP):
        self.config = config
        self.name = config.name
        self.zone = zone
        self.face = face
        
        # State
        self.primary_damper_position = 1.0
        self.fan_running = False
        self.reheat_output_w = 0.0
        self.mode = "cooling"
        
        # Statistics
        self.fan_runtime_hours = 0.0
        self.energy_kwh = 0.0
        
    def update(self, zone_temp_c: float, plenum_temp_c: float = 25.0, 
               dt: float = 0.01) -> Tuple[float, float, float, float]:
        """
        Update fan-powered box state.
        
        Args:
            zone_temp_c: Current zone temperature
            plenum_temp_c: Return plenum temperature (warmer than supply)
            dt: Timestep
            
        Returns:
            (total_flow_m3s, discharge_temp_c, reheat_w, fan_power_w)
        """
        cfg = self.config
        
        error = zone_temp_c - cfg.setpoint_temp_c
        half_band = cfg.throttling_range_c / 2
        
        # Determine mode
        if error > half_band:
            # Full cooling
            self.primary_damper_position = 1.0
            self.fan_running = (cfg.box_type == "series")
            self.reheat_output_w = 0.0
            self.mode = "cooling_max"
            
        elif error > -half_band:
            # Proportional band
            normalized = (error + half_band) / cfg.throttling_range_c
            min_frac = cfg.primary_min_flow_m3s / cfg.primary_max_flow_m3s
            self.primary_damper_position = min_frac + (1 - min_frac) * normalized
            self.fan_running = (cfg.box_type == "series")
            self.reheat_output_w = 0.0
            self.mode = "modulating"
            
        else:
            # Heating mode
            self.primary_damper_position = cfg.primary_min_flow_m3s / cfg.primary_max_flow_m3s
            self.fan_running = True  # Both types run fan in heating
            
            if cfg.has_reheat:
                heating_demand = (-error - half_band) / half_band
                self.reheat_output_w = min(cfg.reheat_capacity_w,
                                           heating_demand * cfg.reheat_capacity_w)
            self.mode = "heating"
        
        # Calculate flows
        primary_flow = cfg.primary_max_flow_m3s * self.primary_damper_position
        secondary_flow = cfg.fan_flow_m3s if self.fan_running else 0.0
        
        if cfg.box_type == "series":
            # Series: fan draws both primary and secondary through it
            total_flow = primary_flow + secondary_flow
        else:
            # Parallel: fan only adds when running
            total_flow = primary_flow + secondary_flow
        
        # Calculate mixed temperature
        if total_flow > 0:
            primary_temp = cfg.primary_supply_temp_c
            
            if self.fan_running and secondary_flow > 0:
                # Mix primary and plenum air
                mixed_temp = (primary_flow * primary_temp + secondary_flow * plenum_temp_c) / total_flow
            else:
                mixed_temp = primary_temp
            
            # Add reheat
            if self.reheat_output_w > 0:
                rho, cp = 1.2, 1005.0
                delta_t = self.reheat_output_w / (rho * total_flow * cp)
                discharge_temp = mixed_temp + delta_t
            else:
                discharge_temp = mixed_temp
        else:
            discharge_temp = plenum_temp_c
        
        # Fan power
        fan_power = cfg.fan_power_w if self.fan_running else 0.0
        
        # Energy tracking
        self.energy_kwh += ((self.reheat_output_w + fan_power) / 1000) * (dt / 3600)
        if self.fan_running:
            self.fan_runtime_hours += dt / 3600
        
        return total_flow, discharge_temp, self.reheat_output_w, fan_power
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            'name': self.name,
            'type': self.config.box_type,
            'mode': self.mode,
            'primary_damper': self.primary_damper_position,
            'fan_running': self.fan_running,
            'reheat_w': self.reheat_output_w,
            'fan_runtime_hours': self.fan_runtime_hours,
            'energy_kwh': self.energy_kwh
        }


@dataclass
class DuctConfig:
    """
    Configuration for ductwork between zones.
    
    T3.08: Models pressure drop in duct connections.
    """
    name: str = "duct_1"
    
    # Geometry
    length_m: float = 10.0
    diameter_m: float = 0.3  # For round duct
    width_m: float = 0.4     # For rectangular
    height_m: float = 0.3    # For rectangular
    is_round: bool = True
    
    # Material
    roughness_mm: float = 0.15  # Galvanized steel ≈ 0.15mm
    
    # Fittings (simplified as equivalent length)
    n_elbows: int = 2
    n_tees: int = 0
    n_reducers: int = 0


class Duct:
    """
    Duct segment with pressure drop calculation.
    
    T3.08: Calculates pressure loss using Darcy-Weisbach equation.
    
    ΔP = f × (L/D) × (ρV²/2)
    
    Where f is the Darcy friction factor from Colebrook equation.
    """
    
    def __init__(self, config: DuctConfig):
        self.config = config
        self.name = config.name
        
        # Calculate hydraulic diameter
        if config.is_round:
            self.D_h = config.diameter_m
            self.area = math.pi * (config.diameter_m / 2) ** 2
        else:
            # Rectangular: D_h = 4A/P
            self.area = config.width_m * config.height_m
            perimeter = 2 * (config.width_m + config.height_m)
            self.D_h = 4 * self.area / perimeter
        
        # Equivalent length for fittings
        self.L_eq = (config.length_m + 
                     config.n_elbows * 10 * self.D_h +  # Elbow ≈ 10D
                     config.n_tees * 30 * self.D_h +     # Tee ≈ 30D
                     config.n_reducers * 5 * self.D_h)   # Reducer ≈ 5D
    
    def compute_pressure_drop(self, flow_m3s: float, 
                              temp_c: float = 20.0) -> float:
        """
        Calculate pressure drop for given flow rate.
        
        Args:
            flow_m3s: Volume flow rate (m³/s)
            temp_c: Air temperature (affects density/viscosity)
            
        Returns:
            Pressure drop in Pascals
        """
        if flow_m3s <= 0:
            return 0.0
        
        # Air properties at temperature
        rho = 1.2 * (293.15 / (temp_c + 273.15))  # Ideal gas approx
        mu = 1.81e-5 * ((temp_c + 273.15) / 293.15) ** 0.7  # Sutherland
        nu = mu / rho
        
        # Velocity
        V = flow_m3s / self.area
        
        # Reynolds number
        Re = V * self.D_h / nu
        
        # Friction factor (Colebrook-White, explicit approximation)
        e_D = (self.config.roughness_mm / 1000) / self.D_h
        
        if Re < 2300:
            # Laminar
            f = 64 / Re
        else:
            # Turbulent - Swamee-Jain approximation
            f = 0.25 / (math.log10(e_D / 3.7 + 5.74 / (Re ** 0.9))) ** 2
        
        # Darcy-Weisbach
        delta_p = f * (self.L_eq / self.D_h) * (rho * V ** 2 / 2)
        
        return delta_p
    
    def compute_flow_from_pressure(self, delta_p: float, 
                                   temp_c: float = 20.0) -> float:
        """
        Inverse calculation: flow from pressure drop.
        
        Uses iterative solution since f depends on Re.
        """
        if delta_p <= 0:
            return 0.0
        
        # Initial guess (assume f = 0.02)
        rho = 1.2 * (293.15 / (temp_c + 273.15))
        V_guess = math.sqrt(2 * delta_p * self.D_h / (0.02 * self.L_eq * rho))
        
        # Iterate
        for _ in range(5):
            flow = V_guess * self.area
            actual_dp = self.compute_pressure_drop(flow, temp_c)
            if abs(actual_dp - delta_p) < 1.0:  # 1 Pa tolerance
                break
            V_guess *= math.sqrt(delta_p / max(actual_dp, 1e-6))
        
        return V_guess * self.area


@dataclass
class AHUConfig:
    """
    Configuration for Air Handling Unit.
    
    T3.09: Simplified AHU model for supply air conditioning.
    """
    name: str = "ahu_1"
    
    # Capacity
    design_flow_m3s: float = 5.0      # Total supply air capacity
    supply_temp_c: float = 13.0       # Design supply temperature
    
    # Coils
    cooling_capacity_kw: float = 50.0  # Cooling coil capacity
    heating_capacity_kw: float = 30.0  # Heating coil capacity (preheat + reheat)
    
    # Fan
    fan_power_kw: float = 3.0         # Supply fan power
    fan_efficiency: float = 0.65       # Fan efficiency
    
    # Outdoor air
    min_oa_fraction: float = 0.15     # Minimum outdoor air (ASHRAE 62.1)
    economizer_enabled: bool = True   # Free cooling when OA is cool
    economizer_high_limit_c: float = 18.0  # Disable economizer above this OAT


class AHU:
    """
    Air Handling Unit - Simplified Model.
    
    T3.09: Conditions and distributes air to multiple zones.
    
    Components modeled:
    - Mixed air section (OA + return air)
    - Cooling coil
    - Heating coil (optional)
    - Supply fan
    
    Energy consumption:
    - Cooling: from chilled water or DX
    - Heating: from hot water or electric
    - Fan: electrical
    """
    
    def __init__(self, config: AHUConfig):
        self.config = config
        self.name = config.name
        
        # State
        self.supply_temp_c = config.supply_temp_c
        self.current_flow_m3s = config.design_flow_m3s
        self.oa_fraction = config.min_oa_fraction
        
        # Coil outputs
        self.cooling_output_kw = 0.0
        self.heating_output_kw = 0.0
        self.fan_power_kw = config.fan_power_kw
        
        # Energy tracking
        self.cooling_energy_kwh = 0.0
        self.heating_energy_kwh = 0.0
        self.fan_energy_kwh = 0.0
        
        # Connected VAV terminals
        self.terminals: List[VAVTerminal] = []
    
    def add_terminal(self, terminal: VAVTerminal):
        """Register a VAV terminal served by this AHU."""
        self.terminals.append(terminal)
        terminal.supply_temp_c = self.supply_temp_c
    
    def update(self, return_temp_c: float, outdoor_temp_c: float,
               outdoor_rh: float = 50.0, dt: float = 0.01):
        """
        Update AHU state based on conditions.
        
        Args:
            return_temp_c: Return air temperature from zones
            outdoor_temp_c: Outdoor air temperature
            outdoor_rh: Outdoor relative humidity
            dt: Timestep
        """
        cfg = self.config
        
        # Determine outdoor air fraction
        if cfg.economizer_enabled and outdoor_temp_c < cfg.economizer_high_limit_c:
            # Economizer mode - use up to 100% OA for free cooling
            if outdoor_temp_c < cfg.supply_temp_c:
                self.oa_fraction = 1.0  # Full economizer
            else:
                # Partial economizer
                self.oa_fraction = min(1.0, max(cfg.min_oa_fraction,
                    (return_temp_c - cfg.supply_temp_c) / 
                    max(return_temp_c - outdoor_temp_c, 1.0)))
        else:
            self.oa_fraction = cfg.min_oa_fraction
        
        # Mixed air temperature
        mixed_temp_c = (self.oa_fraction * outdoor_temp_c + 
                        (1 - self.oa_fraction) * return_temp_c)
        
        # Cooling required?
        if mixed_temp_c > cfg.supply_temp_c:
            delta_t = mixed_temp_c - cfg.supply_temp_c
            # Q = m_dot * cp * ΔT
            rho, cp = 1.2, 1005.0
            self.cooling_output_kw = (rho * self.current_flow_m3s * cp * delta_t) / 1000
            self.cooling_output_kw = min(self.cooling_output_kw, cfg.cooling_capacity_kw)
            self.heating_output_kw = 0.0
        else:
            # Heating required (preheat if OAT is very cold)
            delta_t = cfg.supply_temp_c - mixed_temp_c
            rho, cp = 1.2, 1005.0
            self.heating_output_kw = (rho * self.current_flow_m3s * cp * delta_t) / 1000
            self.heating_output_kw = min(self.heating_output_kw, cfg.heating_capacity_kw)
            self.cooling_output_kw = 0.0
        
        # Total flow from all terminals
        total_demand = sum(t.current_flow_m3s for t in self.terminals)
        if total_demand > 0:
            # Adjust fan speed (simplified)
            flow_fraction = min(1.0, total_demand / cfg.design_flow_m3s)
            self.current_flow_m3s = total_demand
            # Fan power scales with cube of flow (affinity laws)
            self.fan_power_kw = cfg.fan_power_kw * (flow_fraction ** 3)
        
        # Energy tracking
        self.cooling_energy_kwh += self.cooling_output_kw * (dt / 3600)
        self.heating_energy_kwh += self.heating_output_kw * (dt / 3600)
        self.fan_energy_kwh += self.fan_power_kw * (dt / 3600)
    
    def get_total_energy_kwh(self) -> float:
        """Get total energy consumption."""
        return self.cooling_energy_kwh + self.heating_energy_kwh + self.fan_energy_kwh
    
    def get_status(self) -> Dict[str, Any]:
        """Get current AHU status."""
        return {
            'name': self.name,
            'supply_temp_c': self.supply_temp_c,
            'flow_m3s': self.current_flow_m3s,
            'flow_cfm': self.current_flow_m3s * 2118.88,
            'oa_fraction': self.oa_fraction,
            'cooling_kw': self.cooling_output_kw,
            'heating_kw': self.heating_output_kw,
            'fan_kw': self.fan_power_kw,
            'cooling_kwh': self.cooling_energy_kwh,
            'heating_kwh': self.heating_energy_kwh,
            'fan_kwh': self.fan_energy_kwh,
            'total_kwh': self.get_total_energy_kwh()
        }


# =============================================================================
# BUILDING-LEVEL METRICS (T3.05)
# =============================================================================

def compute_building_comfort_metrics(building: 'Building') -> Dict[str, Any]:
    """
    Compute building-wide comfort metrics.
    
    T3.05: Aggregate zone metrics for whole-building assessment.
    
    Returns:
        Dict with:
        - avg_pmv, avg_ppd: Volume-weighted comfort indices
        - adpi: Building-wide ADPI (% of points comfortable)
        - temp_uniformity: Max temperature difference between zones
        - energy_consumption: Total HVAC energy if equipment present
    """
    import numpy as np
    from hyperfoam.solver import compute_pmv, compute_ppd, compute_edt, compute_adpi
    
    total_volume = 0.0
    weighted_temp = 0.0
    weighted_co2 = 0.0
    all_temps = []
    all_pmv = []
    
    for zone in building.zones.values():
        metrics = zone.get_metrics()
        volume = zone.lx * zone.ly * zone.lz
        
        total_volume += volume
        weighted_temp += metrics['temperature_c'] * volume
        weighted_co2 += metrics['co2_ppm'] * volume
        all_temps.append(metrics['temperature_c'])
        
        # Compute zone PMV (assume typical conditions)
        pmv = compute_pmv(
            ta=metrics['temperature_c'],
            tr=metrics['temperature_c'],  # Assume MRT ≈ air temp
            vel=metrics['velocity_avg'],
            rh=50.0,  # Assumed
            met=1.0,
            clo=0.5
        )
        all_pmv.append(pmv)
    
    avg_temp = weighted_temp / total_volume if total_volume > 0 else 22.0
    avg_co2 = weighted_co2 / total_volume if total_volume > 0 else 400.0
    avg_pmv = np.mean(all_pmv) if all_pmv else 0.0
    avg_ppd = compute_ppd(avg_pmv)
    
    return {
        'avg_temperature_c': avg_temp,
        'avg_co2_ppm': avg_co2,
        'avg_pmv': avg_pmv,
        'avg_ppd': avg_ppd,
        'temp_uniformity_c': max(all_temps) - min(all_temps) if all_temps else 0.0,
        'n_zones': len(building.zones),
        'total_volume_m3': total_volume
    }


# For imports
__all__ = [
    'EquipmentType',
    'VAVConfig', 'VAVTerminal',
    'FanPoweredBoxConfig', 'FanPoweredBox',
    'DuctConfig', 'Duct',
    'AHUConfig', 'AHU',
    'compute_building_comfort_metrics'
]
