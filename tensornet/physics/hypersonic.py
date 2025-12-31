"""
Phase 3A-1: Hypersonic Flight Hazard Field
==========================================

Converts NOAA weather data into a "Cost of Flight" tensor for trajectory
optimization. This is the foundation of the Kill Web guidance system.

Physics:
    - Dynamic Pressure: Q = 0.5 * ρ * V²  
    - Wind Shear: ∇(wind_vector) → turbulence loading
    - Thermal: Stagnation temperature ∝ Mach²
    
Cost Function:
    - Soft exponential wall when Q approaches structural limit
    - Turbulence penalty from wind shear gradients
    - Temperature penalty approaching TPS limits

Reference:
    - Anderson, Hypersonic and High Temperature Gas Dynamics (2006)
    - Tauber & Sutton, Stagnation-Point Heat Transfer (1991)
"""

import torch
from torch import Tensor
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math


# ═══════════════════════════════════════════════════════════════════════════
# Physical Constants
# ═══════════════════════════════════════════════════════════════════════════

SPEED_OF_SOUND_SEA_LEVEL = 343.0  # m/s at 15°C
GAMMA_AIR = 1.4
GAS_CONSTANT_AIR = 287.058  # J/(kg·K)
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)

# Vehicle limits (typical hypersonic cruiser)
Q_LIMIT_PA = 50000.0        # Dynamic pressure limit (Pa)
TEMP_LIMIT_K = 2000.0       # TPS temperature limit (K)
G_LIMIT = 9.0               # Structural g-limit


@dataclass
class VehicleConfig:
    """Hypersonic vehicle configuration for hazard assessment."""
    
    # Aerodynamic
    nose_radius_m: float = 0.15      # Nose radius for heating calc
    reference_area_m2: float = 25.0  # Wing area
    mass_kg: float = 15000.0         # Vehicle mass
    
    # Thermal Protection
    tps_limit_K: float = 2000.0      # Max TPS temperature
    emissivity: float = 0.85         # Surface emissivity
    
    # Structural
    q_limit_Pa: float = 50000.0      # Max dynamic pressure
    g_limit: float = 9.0             # Max structural load factor
    
    # Performance
    mach_cruise: float = 10.0        # Design cruise Mach


@dataclass  
class HazardField:
    """Container for computed hazard tensors."""
    
    # Primary cost field (sum of all hazards)
    total_cost: Tensor
    
    # Component hazards
    q_cost: Tensor           # Dynamic pressure hazard
    thermal_cost: Tensor     # Temperature hazard
    shear_cost: Tensor       # Wind shear/turbulence hazard
    
    # Physical fields
    dynamic_pressure: Tensor  # Q field (Pa)
    stagnation_temp: Tensor   # Nose temperature (K)
    wind_shear: Tensor        # Shear magnitude (1/s)
    
    # Metadata
    grid_shape: Tuple[int, ...]
    device: torch.device


# ═══════════════════════════════════════════════════════════════════════════
# Core Hazard Calculations
# ═══════════════════════════════════════════════════════════════════════════

def calculate_dynamic_pressure(
    density: Tensor,
    velocity_m_s: float,
) -> Tensor:
    """
    Calculate dynamic pressure field.
    
    Q = 0.5 * ρ * V²
    
    Args:
        density: Atmospheric density field (kg/m³) [D, H, W] or [H, W]
        velocity_m_s: Flight velocity (m/s)
        
    Returns:
        Dynamic pressure field (Pa)
    """
    return 0.5 * density * (velocity_m_s ** 2)


def calculate_stagnation_temperature(
    ambient_temp_K: Tensor,
    mach: float,
    gamma: float = GAMMA_AIR,
) -> Tensor:
    """
    Calculate stagnation (total) temperature at the nose.
    
    T₀ = T∞ * (1 + (γ-1)/2 * M²)
    
    For Mach 10: T₀/T∞ ≈ 21 → 220K ambient → 4620K stagnation
    Real vehicles use thermal protection, so we compute the
    radiation equilibrium temperature instead.
    
    Args:
        ambient_temp_K: Ambient temperature field (K)
        mach: Flight Mach number
        gamma: Ratio of specific heats
        
    Returns:
        Stagnation temperature field (K)
    """
    recovery_factor = 1.0 + ((gamma - 1) / 2) * (mach ** 2)
    return ambient_temp_K * recovery_factor


def calculate_equilibrium_wall_temperature(
    density: Tensor,
    velocity_m_s: float,
    nose_radius_m: float,
    emissivity: float = 0.85,
) -> Tensor:
    """
    Calculate radiation equilibrium wall temperature.
    
    Uses Sutton-Graves correlation for stagnation point heating:
    q̇ = k * (ρ/R_n)^0.5 * V^3
    
    At equilibrium: q̇ = ε * σ * T_w^4
    → T_w = (q̇ / (ε * σ))^0.25
    
    Args:
        density: Atmospheric density (kg/m³)
        velocity_m_s: Flight velocity (m/s)
        nose_radius_m: Nose radius (m)
        emissivity: Surface emissivity
        
    Returns:
        Wall temperature field (K)
    """
    # Sutton-Graves constant for Earth atmosphere
    k_sg = 1.7415e-4  # W/(m² * (kg/m³)^0.5 * (m/s)^3)
    
    # Stagnation point heat flux
    heat_flux = k_sg * torch.sqrt(density / nose_radius_m) * (velocity_m_s ** 3)
    
    # Radiation equilibrium temperature
    # T_w = (q / (ε * σ))^0.25
    temp_wall = torch.pow(
        heat_flux / (emissivity * STEFAN_BOLTZMANN),
        0.25
    )
    
    return temp_wall


def calculate_wind_shear(
    wind_u: Tensor,
    wind_v: Tensor,
    wind_w: Optional[Tensor] = None,
    dx: float = 1.0,
) -> Tensor:
    """
    Calculate wind shear magnitude (velocity gradient).
    
    Shear = |∇V| = sqrt((∂u/∂x)² + (∂v/∂y)² + ...)
    
    High shear → turbulence → structural loading
    
    Args:
        wind_u: U-component of wind (m/s)
        wind_v: V-component of wind (m/s)
        wind_w: W-component of wind (m/s), optional
        dx: Grid spacing (m)
        
    Returns:
        Shear magnitude field (1/s)
    """
    # Compute gradients
    if wind_u.dim() == 2:
        # 2D case
        du_dy, du_dx = torch.gradient(wind_u, spacing=dx)
        dv_dy, dv_dx = torch.gradient(wind_v, spacing=dx)
        
        shear = torch.sqrt(du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2)
        
    else:
        # 3D case
        du = torch.gradient(wind_u, spacing=dx)
        dv = torch.gradient(wind_v, spacing=dx)
        
        shear_sq = sum(g**2 for g in du) + sum(g**2 for g in dv)
        
        if wind_w is not None:
            dw = torch.gradient(wind_w, spacing=dx)
            shear_sq += sum(g**2 for g in dw)
        
        shear = torch.sqrt(shear_sq)
    
    return shear


# ═══════════════════════════════════════════════════════════════════════════
# Cost Function
# ═══════════════════════════════════════════════════════════════════════════

def calculate_hazard_field(
    density: Tensor,
    wind_u: Tensor,
    wind_v: Tensor,
    wind_w: Optional[Tensor] = None,
    temperature: Optional[Tensor] = None,
    mach: float = 10.0,
    vehicle: Optional[VehicleConfig] = None,
    weights: Optional[Dict[str, float]] = None,
) -> HazardField:
    """
    Convert weather data into a unified 'Cost of Flight' field.
    
    This is the core function for trajectory optimization - it transforms
    atmospheric conditions into a scalar field where:
        - Low values → Safe to fly
        - High values → Dangerous (structural, thermal, turbulence)
        - Infinity → No-fly zone (vehicle destruction)
    
    Args:
        density: Atmospheric density field (kg/m³)
        wind_u: U-component of wind (m/s)
        wind_v: V-component of wind (m/s)  
        wind_w: W-component of wind (m/s), optional for 3D
        temperature: Ambient temperature field (K), optional
        mach: Flight Mach number
        vehicle: Vehicle configuration
        weights: Cost component weights {'q': 1.0, 'thermal': 1.0, 'shear': 10.0}
        
    Returns:
        HazardField containing all cost components
    """
    if vehicle is None:
        vehicle = VehicleConfig(mach_cruise=mach)
    
    if weights is None:
        weights = {'q': 1.0, 'thermal': 1.0, 'shear': 10.0}
    
    device = density.device
    velocity_m_s = mach * SPEED_OF_SOUND_SEA_LEVEL
    
    # ─────────────────────────────────────────────────────────────────────
    # 1. Dynamic Pressure Hazard
    # ─────────────────────────────────────────────────────────────────────
    q_field = calculate_dynamic_pressure(density, velocity_m_s)
    
    # Soft exponential wall: cost explodes as Q → Q_limit
    # C_q = (Q / Q_limit)^4
    q_normalized = q_field / vehicle.q_limit_Pa
    q_cost = torch.pow(q_normalized, 4)
    
    # Hard cutoff: infinite cost above limit
    q_cost = torch.where(
        q_field > vehicle.q_limit_Pa * 1.2,
        torch.tensor(float('inf'), device=device),
        q_cost
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # 2. Thermal Hazard
    # ─────────────────────────────────────────────────────────────────────
    temp_wall = calculate_equilibrium_wall_temperature(
        density,
        velocity_m_s,
        vehicle.nose_radius_m,
        vehicle.emissivity,
    )
    
    # Soft exponential wall: cost explodes as T → T_limit
    temp_normalized = temp_wall / vehicle.tps_limit_K
    thermal_cost = torch.pow(temp_normalized, 4)
    
    # Hard cutoff above TPS limit
    thermal_cost = torch.where(
        temp_wall > vehicle.tps_limit_K * 1.1,
        torch.tensor(float('inf'), device=device),
        thermal_cost
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # 3. Wind Shear / Turbulence Hazard
    # ─────────────────────────────────────────────────────────────────────
    shear_field = calculate_wind_shear(wind_u, wind_v, wind_w)
    
    # Normalize by typical acceptable shear (0.01 1/s)
    shear_normalized = shear_field / 0.01
    shear_cost = torch.pow(shear_normalized, 2)
    
    # ─────────────────────────────────────────────────────────────────────
    # 4. Total Cost
    # ─────────────────────────────────────────────────────────────────────
    total_cost = (
        weights['q'] * q_cost +
        weights['thermal'] * thermal_cost +
        weights['shear'] * shear_cost
    )
    
    # Provide stagnation temperature if ambient temp given
    if temperature is not None:
        stag_temp = calculate_stagnation_temperature(temperature, mach)
    else:
        stag_temp = temp_wall  # Use wall temp as proxy
    
    return HazardField(
        total_cost=total_cost,
        q_cost=q_cost,
        thermal_cost=thermal_cost,
        shear_cost=shear_cost,
        dynamic_pressure=q_field,
        stagnation_temp=stag_temp,
        wind_shear=shear_field,
        grid_shape=tuple(density.shape),
        device=device,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def hazard_to_traversability(hazard: HazardField, threshold: float = 10.0) -> Tensor:
    """
    Convert hazard field to binary traversability mask.
    
    Args:
        hazard: HazardField from calculate_hazard_field
        threshold: Maximum acceptable cost
        
    Returns:
        Binary mask: 1 = safe, 0 = dangerous
    """
    safe_mask = (hazard.total_cost < threshold) & torch.isfinite(hazard.total_cost)
    return safe_mask.float()


def find_safe_corridors(
    hazard: HazardField,
    threshold: float = 5.0,
    min_corridor_width: int = 3,
) -> Tensor:
    """
    Find continuous safe corridors through the hazard field.
    
    Uses morphological operations to find connected safe regions
    with minimum width for vehicle passage.
    
    Args:
        hazard: HazardField
        threshold: Maximum acceptable cost
        min_corridor_width: Minimum corridor width in grid cells
        
    Returns:
        Corridor mask: 1 = safe corridor, 0 = no-go
    """
    safe = hazard_to_traversability(hazard, threshold)
    
    # Erosion then dilation to find corridors with minimum width
    # This ensures the corridor is wide enough for the vehicle
    if min_corridor_width > 1:
        import torch.nn.functional as F
        
        # Create erosion kernel
        k = min_corridor_width
        kernel = torch.ones(1, 1, k, k, device=safe.device) / (k * k)
        
        # Pad and apply
        if safe.dim() == 2:
            safe_4d = safe.unsqueeze(0).unsqueeze(0)
            eroded = F.conv2d(safe_4d, kernel, padding=k//2)
            corridors = (eroded > 0.99).squeeze()
        else:
            corridors = safe  # 3D case - skip for now
    else:
        corridors = safe
    
    return corridors


# ═══════════════════════════════════════════════════════════════════════════
# Test / Demo
# ═══════════════════════════════════════════════════════════════════════════

def demo_hazard_field():
    """Demonstrate hazard field calculation with synthetic weather."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create synthetic atmospheric data
    H, W = 256, 256
    
    # Density: exponential atmosphere with perturbations
    alt_normalized = torch.linspace(0, 1, H, device=device).unsqueeze(1).expand(H, W)
    base_density = 1.225 * torch.exp(-alt_normalized * 8)  # Scale height ~12km
    
    # Add weather perturbations (storms = higher density)
    x = torch.linspace(-3, 3, W, device=device)
    y = torch.linspace(-3, 3, H, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    storm_1 = 0.1 * torch.exp(-((X - 1)**2 + (Y - 0.5)**2) / 0.5)
    storm_2 = 0.15 * torch.exp(-((X + 1.5)**2 + (Y + 1)**2) / 0.8)
    
    density = base_density + storm_1 + storm_2
    
    # Wind field: jet stream with shear
    wind_u = 50.0 * torch.sin(Y * 2) + 20.0 * torch.randn(H, W, device=device) * 0.1
    wind_v = 30.0 * torch.cos(X * 1.5) + 20.0 * torch.randn(H, W, device=device) * 0.1
    
    # Calculate hazard
    print("\nCalculating hazard field for Mach 10 flight...")
    hazard = calculate_hazard_field(
        density=density,
        wind_u=wind_u,
        wind_v=wind_v,
        mach=10.0,
    )
    
    print(f"\n  Grid Shape: {hazard.grid_shape}")
    print(f"  Dynamic Pressure Range: {hazard.dynamic_pressure.min():.0f} - {hazard.dynamic_pressure.max():.0f} Pa")
    print(f"  Wall Temperature Range: {hazard.stagnation_temp.min():.0f} - {hazard.stagnation_temp.max():.0f} K")
    print(f"  Wind Shear Range: {hazard.wind_shear.min():.4f} - {hazard.wind_shear.max():.4f} 1/s")
    
    # Find safe regions
    safe_fraction = (hazard.total_cost < 5.0).float().mean()
    print(f"\n  Safe Flight Fraction: {safe_fraction*100:.1f}%")
    
    return hazard


# ═══════════════════════════════════════════════════════════════════════════
# Atmospheric Model (for RL Environment)
# ═══════════════════════════════════════════════════════════════════════════

class AtmosphericModel:
    """
    Standard atmosphere model for flight simulation.
    
    Provides density and temperature as a function of altitude.
    Based on US Standard Atmosphere 1976.
    """
    
    # Layer boundaries (meters)
    TROPOSPHERE = 11000
    STRATOSPHERE1 = 20000
    STRATOSPHERE2 = 32000
    STRATOPAUSE = 47000
    MESOSPHERE1 = 51000
    MESOSPHERE2 = 71000
    
    # Sea level conditions
    RHO_0 = 1.225       # kg/m³
    T_0 = 288.15        # K
    P_0 = 101325.0      # Pa
    
    # Temperature lapse rates (K/m)
    LAPSE_TROPO = -0.0065
    LAPSE_STRAT2 = 0.001
    LAPSE_STRAT3 = 0.0028
    LAPSE_MESO1 = -0.0028
    LAPSE_MESO2 = -0.002
    
    def temperature(self, altitude_m: float) -> float:
        """
        Get temperature at altitude.
        
        Args:
            altitude_m: Altitude in meters
            
        Returns:
            Temperature in Kelvin
        """
        if altitude_m < self.TROPOSPHERE:
            return self.T_0 + self.LAPSE_TROPO * altitude_m
        elif altitude_m < self.STRATOSPHERE1:
            return 216.65  # Isothermal
        elif altitude_m < self.STRATOSPHERE2:
            return 216.65 + self.LAPSE_STRAT2 * (altitude_m - self.STRATOSPHERE1)
        elif altitude_m < self.STRATOPAUSE:
            return 228.65 + self.LAPSE_STRAT3 * (altitude_m - self.STRATOSPHERE2)
        elif altitude_m < self.MESOSPHERE1:
            return 270.65  # Isothermal
        elif altitude_m < self.MESOSPHERE2:
            return 270.65 + self.LAPSE_MESO1 * (altitude_m - self.MESOSPHERE1)
        else:
            return 214.65 + self.LAPSE_MESO2 * (altitude_m - self.MESOSPHERE2)
    
    def pressure(self, altitude_m: float) -> float:
        """
        Get pressure at altitude using barometric formula.
        
        Args:
            altitude_m: Altitude in meters
            
        Returns:
            Pressure in Pascals
        """
        # Simplified exponential model
        scale_height = 8500.0  # meters
        return self.P_0 * math.exp(-altitude_m / scale_height)
    
    def density(self, altitude_m: float) -> float:
        """
        Get density at altitude.
        
        Uses ideal gas law: ρ = P / (R * T)
        
        Args:
            altitude_m: Altitude in meters
            
        Returns:
            Density in kg/m³
        """
        T = self.temperature(altitude_m)
        P = self.pressure(altitude_m)
        return P / (GAS_CONSTANT_AIR * T)
    
    def speed_of_sound(self, altitude_m: float) -> float:
        """
        Get speed of sound at altitude.
        
        a = sqrt(γ * R * T)
        
        Args:
            altitude_m: Altitude in meters
            
        Returns:
            Speed of sound in m/s
        """
        T = self.temperature(altitude_m)
        return math.sqrt(GAMMA_AIR * GAS_CONSTANT_AIR * T)


if __name__ == "__main__":
    demo_hazard_field()
