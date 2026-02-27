"""
T5: Fire/Smoke/Atrium Simulation Models

Capabilities:
- T5.01: Fire Source (HRR curves)
- T5.02: Smoke Species Transport
- T5.03: Visibility Calculation
- T5.04: Tenability Metrics (NFPA 502/BS 7974)
- T5.05: Jet Fan Model
- T5.06: Smoke Extraction
- T5.07: Buoyant Plume Physics
- T5.08: ASET/RSET Analysis
- T5.09: Egress Zone Monitoring
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import math


# =============================================================================
# T5.01: FIRE SOURCE MODEL
# =============================================================================

class FireGrowthType(Enum):
    """Fire growth rate classifications (NFPA)."""
    SLOW = auto()      # α = 0.00293 kW/s²
    MEDIUM = auto()    # α = 0.01172 kW/s²
    FAST = auto()      # α = 0.04689 kW/s²
    ULTRAFAST = auto() # α = 0.1876 kW/s²


@dataclass
class FireConfig:
    """Configuration for fire source."""
    name: str = "fire_1"
    fire_id: int = 0
    
    # Position (m)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    diameter_m: float = 2.0
    
    # Heat Release Rate
    peak_hrr_kw: float = 5000.0  # 5 MW typical kiosk fire
    growth_type: FireGrowthType = FireGrowthType.FAST
    
    # Timeline
    growth_time_s: float = 300.0   # Time to peak
    steady_time_s: float = 600.0   # Duration at peak
    decay_time_s: float = 300.0    # Decay duration
    
    # Combustion products
    soot_yield: float = 0.07       # kg soot / kg fuel
    co_yield: float = 0.04         # kg CO / kg fuel
    co2_yield: float = 2.5         # kg CO2 / kg fuel
    heat_of_combustion_kj_kg: float = 20000.0


class FireSource:
    """
    T5.01: Fire source with time-varying HRR.
    
    HRR follows t² growth, steady, and linear decay.
    """
    
    def __init__(self, config: FireConfig):
        self.config = config
        self.name = config.name
        
        # Growth coefficient
        alpha_map = {
            FireGrowthType.SLOW: 0.00293,
            FireGrowthType.MEDIUM: 0.01172,
            FireGrowthType.FAST: 0.04689,
            FireGrowthType.ULTRAFAST: 0.1876,
        }
        self.alpha = alpha_map[config.growth_type]
        
        # State
        self.current_hrr_kw = 0.0
        self.total_heat_mj = 0.0
        self.smoke_production_kg = 0.0
        self.co_production_kg = 0.0
        
    def get_hrr(self, time_s: float) -> float:
        """Get HRR at given time (kW)."""
        cfg = self.config
        t_growth_end = cfg.growth_time_s
        t_steady_end = t_growth_end + cfg.steady_time_s
        t_decay_end = t_steady_end + cfg.decay_time_s
        
        if time_s < 0:
            return 0.0
        elif time_s < t_growth_end:
            # t² growth: Q = α * t²
            hrr = self.alpha * time_s ** 2
            return min(hrr, cfg.peak_hrr_kw)
        elif time_s < t_steady_end:
            return cfg.peak_hrr_kw
        elif time_s < t_decay_end:
            # Linear decay
            decay_frac = (time_s - t_steady_end) / cfg.decay_time_s
            return cfg.peak_hrr_kw * (1 - decay_frac)
        else:
            return 0.0
    
    def update(self, time_s: float, dt: float) -> Dict[str, float]:
        """Update fire state and return production rates."""
        cfg = self.config
        self.current_hrr_kw = self.get_hrr(time_s)
        
        # Mass burning rate: m_dot = Q / ΔH_c
        m_dot_kg_s = self.current_hrr_kw / cfg.heat_of_combustion_kj_kg
        
        # Production rates
        soot_rate = m_dot_kg_s * cfg.soot_yield
        co_rate = m_dot_kg_s * cfg.co_yield
        co2_rate = m_dot_kg_s * cfg.co2_yield
        
        # Accumulate
        self.total_heat_mj += self.current_hrr_kw * dt / 1000
        self.smoke_production_kg += soot_rate * dt
        self.co_production_kg += co_rate * dt
        
        return {
            'hrr_kw': self.current_hrr_kw,
            'soot_kg_s': soot_rate,
            'co_kg_s': co_rate,
            'co2_kg_s': co2_rate,
            'convective_kw': self.current_hrr_kw * 0.7,  # 70% convective
            'radiative_kw': self.current_hrr_kw * 0.3,   # 30% radiative
        }


# =============================================================================
# T5.03: VISIBILITY CALCULATION
# =============================================================================

def compute_visibility(optical_density_per_m: float, 
                       visibility_constant: float = 8.0) -> float:
    """
    T5.03: Compute visibility from smoke optical density.
    
    S = K / OD
    
    Where:
        K = 8 for light-reflecting signs
        K = 3 for light-emitting signs
        OD = optical density per meter (1/m)
    
    Returns visibility in meters.
    """
    if optical_density_per_m <= 0:
        return 100.0  # Clear air, 100m visibility
    return visibility_constant / optical_density_per_m


def smoke_to_optical_density(soot_concentration_kg_m3: float,
                              mass_specific_extinction: float = 8700.0) -> float:
    """
    Convert soot concentration to optical density.
    
    OD = σ_m × Y_s
    
    Where σ_m ≈ 8700 m²/kg for typical fire smoke.
    """
    return mass_specific_extinction * soot_concentration_kg_m3


# =============================================================================
# T5.04: TENABILITY METRICS (NFPA 502 / BS 7974)
# =============================================================================

@dataclass
class TenabilityThresholds:
    """NFPA 502 / BS 7974 tenability limits."""
    visibility_tenable_m: float = 10.0
    visibility_untenable_m: float = 3.0
    
    temperature_tenable_c: float = 60.0
    temperature_untenable_c: float = 80.0
    
    co_tenable_ppm: float = 1400.0
    co_untenable_ppm: float = 2500.0
    
    radiant_heat_tenable_kw_m2: float = 2.5


class TenabilityStatus(Enum):
    """Tenability classification."""
    TENABLE = auto()
    MARGINAL = auto()
    UNTENABLE = auto()


def assess_tenability(temperature_c: float, visibility_m: float,
                      co_ppm: float, thresholds: TenabilityThresholds = None
                      ) -> Tuple[TenabilityStatus, Dict[str, str]]:
    """
    T5.04: Assess tenability at a point.
    
    Returns overall status and per-parameter status.
    """
    if thresholds is None:
        thresholds = TenabilityThresholds()
    
    results = {}
    worst = TenabilityStatus.TENABLE
    
    # Temperature
    if temperature_c >= thresholds.temperature_untenable_c:
        results['temperature'] = 'UNTENABLE'
        worst = TenabilityStatus.UNTENABLE
    elif temperature_c >= thresholds.temperature_tenable_c:
        results['temperature'] = 'MARGINAL'
        if worst != TenabilityStatus.UNTENABLE:
            worst = TenabilityStatus.MARGINAL
    else:
        results['temperature'] = 'TENABLE'
    
    # Visibility
    if visibility_m <= thresholds.visibility_untenable_m:
        results['visibility'] = 'UNTENABLE'
        worst = TenabilityStatus.UNTENABLE
    elif visibility_m <= thresholds.visibility_tenable_m:
        results['visibility'] = 'MARGINAL'
        if worst != TenabilityStatus.UNTENABLE:
            worst = TenabilityStatus.MARGINAL
    else:
        results['visibility'] = 'TENABLE'
    
    # CO
    if co_ppm >= thresholds.co_untenable_ppm:
        results['co'] = 'UNTENABLE'
        worst = TenabilityStatus.UNTENABLE
    elif co_ppm >= thresholds.co_tenable_ppm:
        results['co'] = 'MARGINAL'
        if worst != TenabilityStatus.UNTENABLE:
            worst = TenabilityStatus.MARGINAL
    else:
        results['co'] = 'TENABLE'
    
    return worst, results


# =============================================================================
# T5.05: JET FAN MODEL
# =============================================================================

@dataclass
class JetFanConfig:
    """Configuration for tunnel/atrium jet fan."""
    name: str = "jetfan_1"
    fan_id: int = 0
    
    # Position
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # Unit vector
    
    # Performance
    thrust_n: float = 1000.0       # Thrust force
    flow_m3s: float = 30.0         # Volume flow
    jet_velocity_ms: float = 25.0  # Discharge velocity
    power_kw: float = 15.0


class JetFan:
    """
    T5.05: Jet fan for smoke control.
    
    Adds momentum source to flow field.
    """
    
    def __init__(self, config: JetFanConfig):
        self.config = config
        self.name = config.name
        self.running = True
        self.runtime_hours = 0.0
        
    def get_momentum_source(self) -> Tuple[float, float, float]:
        """Get momentum source vector (N)."""
        if not self.running:
            return (0.0, 0.0, 0.0)
        
        dx, dy, dz = self.config.direction
        mag = math.sqrt(dx**2 + dy**2 + dz**2)
        if mag < 1e-6:
            return (0.0, 0.0, 0.0)
        
        thrust = self.config.thrust_n
        return (thrust * dx / mag, thrust * dy / mag, thrust * dz / mag)
    
    def update(self, dt: float) -> Dict[str, float]:
        """Update fan state."""
        if self.running:
            self.runtime_hours += dt / 3600
        
        mx, my, mz = self.get_momentum_source()
        return {
            'running': self.running,
            'thrust_n': self.config.thrust_n if self.running else 0.0,
            'momentum_x': mx,
            'momentum_y': my,
            'momentum_z': mz,
            'power_kw': self.config.power_kw if self.running else 0.0,
        }


# =============================================================================
# T5.06: SMOKE EXTRACTION
# =============================================================================

@dataclass
class ExtractionPointConfig:
    """Configuration for smoke extraction point."""
    name: str = "extract_1"
    extract_id: int = 0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    area_m2: float = 1.0
    design_flow_m3s: float = 10.0


class SmokeExtraction:
    """T5.06: Smoke extraction system."""
    
    def __init__(self):
        self.points: List[ExtractionPointConfig] = []
        self.active = True
        self.total_extraction_m3s = 0.0
        
    def add_point(self, config: ExtractionPointConfig):
        """Add extraction point."""
        self.points.append(config)
        
    def get_total_flow(self) -> float:
        """Get total extraction flow rate."""
        if not self.active:
            return 0.0
        return sum(p.design_flow_m3s for p in self.points)
    
    def update(self, dt: float) -> Dict[str, float]:
        """Update extraction state."""
        self.total_extraction_m3s = self.get_total_flow()
        return {
            'active': self.active,
            'n_points': len(self.points),
            'total_flow_m3s': self.total_extraction_m3s,
        }


# =============================================================================
# T5.07: BUOYANT PLUME PHYSICS
# =============================================================================

def heskestad_plume_radius(z: float, hrr_kw: float, z0: float = 0.0) -> float:
    """
    Heskestad plume radius at height z.
    
    b = 0.12 * (z - z0)
    
    Where z0 is virtual origin (typically small for pool fires).
    """
    if z <= z0:
        return 0.0
    return 0.12 * (z - z0)


def heskestad_centerline_velocity(z: float, hrr_kw: float, z0: float = 0.0) -> float:
    """
    Heskestad plume centerline velocity (m/s).
    
    u_c = 0.87 * (Q_c)^(1/3) * (z - z0)^(-1/3)
    
    Q_c = convective HRR (typically 0.7 * HRR)
    """
    if z <= z0:
        return 0.0
    
    q_c = 0.7 * hrr_kw  # Convective fraction
    return 0.87 * (q_c ** (1/3)) * ((z - z0) ** (-1/3))


def heskestad_centerline_temp_rise(z: float, hrr_kw: float, 
                                    z0: float = 0.0, T_amb_c: float = 20.0) -> float:
    """
    Heskestad plume centerline temperature rise (K).
    
    ΔT = 9.1 * (T_amb / (g * c_p² * ρ_∞²))^(1/3) * Q_c^(2/3) * (z - z0)^(-5/3)
    """
    if z <= z0:
        return 0.0
    
    q_c = 0.7 * hrr_kw * 1000  # Convert to W
    T_amb_k = T_amb_c + 273.15
    g = 9.81
    cp = 1005.0
    rho = 1.2
    
    coeff = 9.1 * (T_amb_k / (g * cp**2 * rho**2)) ** (1/3)
    delta_t = coeff * (q_c ** (2/3)) * ((z - z0) ** (-5/3))
    
    return min(delta_t, 800.0)  # Cap at 800K rise


def plume_mass_flow_rate(z: float, hrr_kw: float, z0: float = 0.0) -> float:
    """
    Plume mass entrainment rate at height z (kg/s).
    
    m_dot = 0.071 * Q_c^(1/3) * (z - z0)^(5/3) + 0.0018 * Q_c
    """
    if z <= z0:
        return 0.0
    
    q_c = 0.7 * hrr_kw  # kW
    term1 = 0.071 * (q_c ** (1/3)) * ((z - z0) ** (5/3))
    term2 = 0.0018 * q_c
    return term1 + term2


# =============================================================================
# T5.08: ASET/RSET ANALYSIS
# =============================================================================

@dataclass
class EgressConfig:
    """Configuration for egress analysis."""
    n_occupants: int = 2000
    n_exits: int = 3
    exit_width_m: float = 2.0
    walking_speed_ms: float = 1.2
    max_travel_distance_m: float = 100.0
    flow_rate_persons_per_m_s: float = 1.3  # SFPE typical


def compute_rset(config: EgressConfig) -> float:
    """
    T5.08: Compute Required Safe Egress Time (RSET).
    
    RSET = t_detection + t_notification + t_pre_movement + t_travel
    """
    # Detection time (smoke detectors)
    t_detection = 30.0  # seconds
    
    # Notification/alarm time
    t_notification = 30.0
    
    # Pre-movement (decision + reaction)
    t_pre_movement = 60.0
    
    # Travel time
    t_travel = config.max_travel_distance_m / config.walking_speed_ms
    
    # Queuing time at exits
    total_exit_width = config.n_exits * config.exit_width_m
    flow_capacity = total_exit_width * config.flow_rate_persons_per_m_s
    t_queue = config.n_occupants / flow_capacity
    
    rset = t_detection + t_notification + t_pre_movement + t_travel + t_queue
    return rset


class ASETTracker:
    """
    T5.08: Track Available Safe Egress Time.
    
    ASET = time until conditions become untenable at egress routes.
    """
    
    def __init__(self, exit_positions: List[Tuple[float, float, float]]):
        self.exit_positions = exit_positions
        self.aset_per_exit: Dict[int, Optional[float]] = {
            i: None for i in range(len(exit_positions))
        }
        self.tenability_history: List[Dict] = []
        
    def update(self, time_s: float, exit_conditions: List[Dict[str, float]]):
        """
        Check tenability at each exit.
        
        exit_conditions: List of dicts with 'temperature_c', 'visibility_m', 'co_ppm'
        """
        for i, cond in enumerate(exit_conditions):
            if self.aset_per_exit[i] is not None:
                continue  # Already recorded untenable time
                
            status, _ = assess_tenability(
                cond.get('temperature_c', 20.0),
                cond.get('visibility_m', 100.0),
                cond.get('co_ppm', 0.0)
            )
            
            if status == TenabilityStatus.UNTENABLE:
                self.aset_per_exit[i] = time_s
        
        self.tenability_history.append({
            'time_s': time_s,
            'conditions': exit_conditions
        })
    
    def get_min_aset(self) -> Optional[float]:
        """Get minimum ASET across all exits."""
        valid = [t for t in self.aset_per_exit.values() if t is not None]
        return min(valid) if valid else None
    
    def check_safety(self, rset: float) -> Tuple[bool, float]:
        """
        Check if ASET > RSET (safe egress possible).
        
        Returns (is_safe, margin_seconds)
        """
        min_aset = self.get_min_aset()
        if min_aset is None:
            return True, float('inf')  # All exits still tenable
        
        margin = min_aset - rset
        return margin > 0, margin


# =============================================================================
# T5.09: EGRESS ZONE MONITORING
# =============================================================================

@dataclass
class EgressZone:
    """Monitoring zone for egress path."""
    name: str
    zone_id: int
    position: Tuple[float, float, float]
    height_m: float = 2.0  # Breathing height
    
    # Current conditions
    temperature_c: float = 20.0
    visibility_m: float = 100.0
    co_ppm: float = 0.0
    smoke_density_kg_m3: float = 0.0


class EgressMonitor:
    """T5.09: Monitor tenability along egress routes."""
    
    def __init__(self):
        self.zones: Dict[int, EgressZone] = {}
        self.history: List[Dict] = []
        
    def add_zone(self, zone: EgressZone):
        """Add monitoring zone."""
        self.zones[zone.zone_id] = zone
        
    def update_zone(self, zone_id: int, temperature_c: float,
                    visibility_m: float, co_ppm: float):
        """Update zone conditions."""
        if zone_id in self.zones:
            z = self.zones[zone_id]
            z.temperature_c = temperature_c
            z.visibility_m = visibility_m
            z.co_ppm = co_ppm
    
    def get_all_status(self) -> Dict[int, Tuple[TenabilityStatus, Dict]]:
        """Get tenability status for all zones."""
        results = {}
        for zid, zone in self.zones.items():
            status, details = assess_tenability(
                zone.temperature_c, zone.visibility_m, zone.co_ppm
            )
            results[zid] = (status, details)
        return results
    
    def get_untenable_zones(self) -> List[EgressZone]:
        """Get list of zones that are untenable."""
        untenable = []
        for zone in self.zones.values():
            status, _ = assess_tenability(
                zone.temperature_c, zone.visibility_m, zone.co_ppm
            )
            if status == TenabilityStatus.UNTENABLE:
                untenable.append(zone)
        return untenable


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # T5.01
    'FireGrowthType', 'FireConfig', 'FireSource',
    # T5.03
    'compute_visibility', 'smoke_to_optical_density',
    # T5.04
    'TenabilityThresholds', 'TenabilityStatus', 'assess_tenability',
    # T5.05
    'JetFanConfig', 'JetFan',
    # T5.06
    'ExtractionPointConfig', 'SmokeExtraction',
    # T5.07
    'heskestad_plume_radius', 'heskestad_centerline_velocity',
    'heskestad_centerline_temp_rise', 'plume_mass_flow_rate',
    # T5.08
    'EgressConfig', 'compute_rset', 'ASETTracker',
    # T5.09
    'EgressZone', 'EgressMonitor',
]
