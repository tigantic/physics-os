"""
Thermal Comfort Metrics: PMV/PPD and Stratification Analysis
=============================================================

Implements ASHRAE Standard 55 / ISO 7730 thermal comfort calculations:
- PMV (Predicted Mean Vote): -3 (cold) to +3 (hot)
- PPD (Percent Persons Dissatisfied): 5-100%
- Temperature stratification analysis
- Dead zone identification

Reference: Fanger, P.O. (1970) Thermal Comfort

Author: TiganticLabz
Date: January 2026
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ontic.cfd.ns2d_qtt_native import QTT2DNativeState, qtt_2d_native_to_dense


@dataclass
class ComfortInputs:
    """Inputs for PMV/PPD calculation."""
    
    T_air: float          # Air temperature [°C]
    T_radiant: float      # Mean radiant temperature [°C]
    velocity: float       # Air velocity [m/s]
    humidity_RH: float    # Relative humidity [%]
    met: float = 1.2      # Metabolic rate [met] (1 met = 58.2 W/m²)
    clo: float = 0.5      # Clothing insulation [clo] (1 clo = 0.155 m²K/W)
    
    @property
    def metabolic_W_m2(self) -> float:
        return self.met * 58.2
    
    @property
    def clothing_m2K_W(self) -> float:
        return self.clo * 0.155


def saturated_vapor_pressure(T_C: float) -> float:
    """
    Saturated water vapor pressure [Pa] at temperature T [°C].
    Antoine equation approximation.
    """
    return 610.78 * math.exp(17.269 * T_C / (237.3 + T_C))


def calculate_pmv(inputs: ComfortInputs) -> float:
    """
    Calculate Predicted Mean Vote (PMV) using Fanger's model.
    
    PMV Scale:
        -3: Cold
        -2: Cool
        -1: Slightly cool
         0: Neutral
        +1: Slightly warm
        +2: Warm
        +3: Hot
    
    For comfort: PMV should be in [-0.5, +0.5] (ASHRAE Class B)
    """
    # Unpack inputs
    ta = inputs.T_air
    tr = inputs.T_radiant
    vel = max(inputs.velocity, 0.05)  # Minimum for natural convection
    rh = inputs.humidity_RH
    met = inputs.met
    clo = inputs.clo
    
    # Metabolic rate [W/m²]
    M = met * 58.2
    
    # Clothing insulation [m²·K/W]
    Icl = clo * 0.155
    
    # Clothing surface area factor
    if Icl <= 0.078:
        fcl = 1.0 + 1.290 * Icl
    else:
        fcl = 1.05 + 0.645 * Icl
    
    # Vapor pressure of water in air [Pa]
    pa = rh * saturated_vapor_pressure(ta) / 100.0
    
    # Iteratively solve for clothing surface temperature
    tcl = ta + (35.5 - ta) / (3.5 * (6.45 * Icl + 0.1))
    
    for _ in range(10):
        # Convective heat transfer coefficient
        hc_natural = 2.38 * abs(tcl - ta) ** 0.25
        hc_forced = 12.1 * math.sqrt(vel)
        hc = max(hc_natural, hc_forced)
        
        # Clothing surface temperature (iterative)
        tcl_new = 35.7 - 0.028 * M - Icl * (
            3.96e-8 * fcl * ((tcl + 273.15)**4 - (tr + 273.15)**4) +
            fcl * hc * (tcl - ta)
        )
        
        if abs(tcl_new - tcl) < 0.001:
            break
        tcl = tcl_new
    
    # Final convective coefficient
    hc_natural = 2.38 * abs(tcl - ta) ** 0.25
    hc_forced = 12.1 * math.sqrt(vel)
    hc = max(hc_natural, hc_forced)
    
    # PMV equation
    pmv = (0.303 * math.exp(-0.036 * M) + 0.028) * (
        M - 
        3.05e-3 * (5733 - 6.99 * M - pa) -      # Skin diffusion
        0.42 * (M - 58.15) -                      # Sweating
        1.7e-5 * M * (5867 - pa) -               # Respiration latent
        0.0014 * M * (34 - ta) -                  # Respiration sensible
        3.96e-8 * fcl * ((tcl + 273.15)**4 - (tr + 273.15)**4) -  # Radiation
        fcl * hc * (tcl - ta)                     # Convection
    )
    
    return max(-3.0, min(3.0, pmv))


def calculate_ppd(pmv: float) -> float:
    """
    Calculate Percent Persons Dissatisfied (PPD) from PMV.
    
    PPD = 100 - 95 * exp(-0.03353*PMV^4 - 0.2179*PMV^2)
    
    Note: Minimum PPD is 5% even at PMV=0 (some people are never satisfied).
    """
    ppd = 100 - 95 * math.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
    return max(5.0, min(100.0, ppd))


def pmv_ppd_vectorized(
    T_air: torch.Tensor,
    velocity: torch.Tensor,
    T_radiant: Optional[torch.Tensor] = None,
    humidity_RH: float = 50.0,
    met: float = 1.2,
    clo: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized PMV/PPD calculation for entire field.
    
    Args:
        T_air: Air temperature field [°C]
        velocity: Air velocity magnitude field [m/s]
        T_radiant: Radiant temperature field [°C] (defaults to T_air)
        humidity_RH: Relative humidity [%]
        met: Metabolic rate [met]
        clo: Clothing insulation [clo]
    
    Returns:
        pmv: PMV field
        ppd: PPD field [%]
    """
    if T_radiant is None:
        T_radiant = T_air
    
    # Metabolic rate [W/m²]
    M = met * 58.2
    
    # Clothing insulation [m²·K/W]
    Icl = clo * 0.155
    
    # Clothing surface area factor
    fcl = torch.where(
        Icl <= 0.078,
        1.0 + 1.290 * Icl,
        1.05 + 0.645 * Icl
    ) if isinstance(Icl, torch.Tensor) else (
        1.0 + 1.290 * Icl if Icl <= 0.078 else 1.05 + 0.645 * Icl
    )
    
    # Vapor pressure
    pa = humidity_RH * 610.78 * torch.exp(17.269 * T_air / (237.3 + T_air)) / 100.0
    
    # Approximate clothing surface temperature
    tcl = T_air + (35.5 - T_air) / (3.5 * (6.45 * Icl + 0.1))
    
    # Iterate a few times
    vel = torch.clamp(velocity, min=0.05)
    for _ in range(5):
        hc_natural = 2.38 * torch.abs(tcl - T_air) ** 0.25
        hc_forced = 12.1 * torch.sqrt(vel)
        hc = torch.maximum(hc_natural, hc_forced)
        
        tcl = 35.7 - 0.028 * M - Icl * (
            3.96e-8 * fcl * ((tcl + 273.15)**4 - (T_radiant + 273.15)**4) +
            fcl * hc * (tcl - T_air)
        )
    
    # Final hc
    hc_natural = 2.38 * torch.abs(tcl - T_air) ** 0.25
    hc_forced = 12.1 * torch.sqrt(vel)
    hc = torch.maximum(hc_natural, hc_forced)
    
    # PMV
    pmv = (0.303 * math.exp(-0.036 * M) + 0.028) * (
        M - 
        3.05e-3 * (5733 - 6.99 * M - pa) -
        0.42 * (M - 58.15) -
        1.7e-5 * M * (5867 - pa) -
        0.0014 * M * (34 - T_air) -
        3.96e-8 * fcl * ((tcl + 273.15)**4 - (T_radiant + 273.15)**4) -
        fcl * hc * (tcl - T_air)
    )
    
    pmv = torch.clamp(pmv, -3.0, 3.0)
    
    # PPD
    ppd = 100 - 95 * torch.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
    ppd = torch.clamp(ppd, 5.0, 100.0)
    
    return pmv, ppd


@dataclass
class StratificationResult:
    """Results from vertical temperature stratification analysis."""
    
    heights: torch.Tensor        # Sample heights [m]
    temperatures: torch.Tensor   # Mean temperature at each height [°C]
    T_floor: float               # Temperature at floor level
    T_ceiling: float             # Temperature at ceiling level
    delta_T: float               # Total stratification [K]
    gradient_K_per_m: float      # Average gradient [K/m]
    
    def __repr__(self) -> str:
        return (
            f"Stratification: ΔT = {self.delta_T:.2f}K over {self.heights[-1]:.1f}m\n"
            f"  Floor: {self.T_floor:.1f}°C, Ceiling: {self.T_ceiling:.1f}°C\n"
            f"  Gradient: {self.gradient_K_per_m:.2f} K/m"
        )


def analyze_stratification(
    T: torch.Tensor,
    Ly: float,
    n_samples: int = 10
) -> StratificationResult:
    """
    Analyze vertical temperature stratification.
    
    Args:
        T: 2D temperature field [Nx, Ny] in °C
        Ly: Physical height [m]
        n_samples: Number of height samples
    
    Returns:
        StratificationResult with vertical profile
    """
    Nx, Ny = T.shape
    dy = Ly / Ny
    
    # Sample heights
    heights = torch.linspace(0, Ly, n_samples, device=T.device)
    temperatures = torch.zeros(n_samples, device=T.device)
    
    for i, h in enumerate(heights):
        j = int(h / dy)
        j = min(j, Ny - 1)
        temperatures[i] = T[:, j].mean()
    
    T_floor = temperatures[0].item()
    T_ceiling = temperatures[-1].item()
    delta_T = T_ceiling - T_floor
    gradient = delta_T / Ly
    
    return StratificationResult(
        heights=heights,
        temperatures=temperatures,
        T_floor=T_floor,
        T_ceiling=T_ceiling,
        delta_T=delta_T,
        gradient_K_per_m=gradient
    )


@dataclass  
class DeadZoneResult:
    """Results from dead zone analysis."""
    
    dead_zone_mask: torch.Tensor   # Boolean mask of dead zones
    dead_zone_fraction: float      # Fraction of domain that is dead zone
    dead_zone_area_m2: float       # Physical area of dead zones
    velocity_threshold: float      # Threshold used [m/s]
    
    def __repr__(self) -> str:
        return (
            f"Dead Zones: {self.dead_zone_fraction*100:.1f}% of domain\n"
            f"  Area: {self.dead_zone_area_m2:.1f} m²\n"
            f"  Threshold: v < {self.velocity_threshold} m/s"
        )


def identify_dead_zones(
    u: torch.Tensor,
    v: torch.Tensor,
    Lx: float,
    Ly: float,
    velocity_threshold: float = 0.1,
    occupied_zone: Optional[tuple[float, float, float, float]] = None
) -> DeadZoneResult:
    """
    Identify stagnant regions with poor air circulation.
    
    Args:
        u, v: Velocity components [Nx, Ny]
        Lx, Ly: Physical dimensions [m]
        velocity_threshold: Below this is "dead" [m/s]
        occupied_zone: Optional (x_min, x_max, y_min, y_max) to restrict analysis
    
    Returns:
        DeadZoneResult with mask and statistics
    """
    Nx, Ny = u.shape
    dx, dy = Lx / Nx, Ly / Ny
    cell_area = dx * dy
    
    # Velocity magnitude
    vmag = torch.sqrt(u**2 + v**2)
    
    # Dead zone mask
    dead = vmag < velocity_threshold
    
    # Optionally restrict to occupied zone
    if occupied_zone is not None:
        x_min, x_max, y_min, y_max = occupied_zone
        x = torch.linspace(0, Lx, Nx, device=u.device)
        y = torch.linspace(0, Ly, Ny, device=u.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        occupied = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
        dead = dead & occupied
        total_cells = occupied.sum().item()
    else:
        total_cells = Nx * Ny
    
    dead_cells = dead.sum().item()
    fraction = dead_cells / total_cells if total_cells > 0 else 0.0
    area = dead_cells * cell_area
    
    return DeadZoneResult(
        dead_zone_mask=dead,
        dead_zone_fraction=fraction,
        dead_zone_area_m2=area,
        velocity_threshold=velocity_threshold
    )


@dataclass
class ComfortMapResult:
    """Results from comfort analysis."""
    
    pmv: torch.Tensor              # PMV field
    ppd: torch.Tensor              # PPD field [%]
    T_air: torch.Tensor            # Air temperature [°C]
    velocity: torch.Tensor         # Velocity magnitude [m/s]
    
    # Statistics
    pmv_mean: float
    pmv_std: float
    ppd_mean: float
    comfort_fraction: float        # Fraction with |PMV| < 0.5
    
    def __repr__(self) -> str:
        return (
            f"Comfort Analysis:\n"
            f"  PMV: {self.pmv_mean:.2f} ± {self.pmv_std:.2f}\n"
            f"  PPD: {self.ppd_mean:.1f}%\n"
            f"  Comfortable (|PMV|<0.5): {self.comfort_fraction*100:.1f}%"
        )


def analyze_comfort(
    T_qtt: QTT2DNativeState,
    psi_qtt: QTT2DNativeState,
    config,  # NS2DQTTConfig
    occupied_height_range: tuple[float, float] = (0.1, 1.8),
    humidity_RH: float = 50.0,
    met: float = 1.2,
    clo: float = 0.5
) -> ComfortMapResult:
    """
    Analyze thermal comfort from QTT fields.
    
    Args:
        T_qtt: Temperature field in QTT format
        psi_qtt: Streamfunction in QTT format
        config: NS2DQTTConfig with grid info
        occupied_height_range: (y_min, y_max) for occupied zone [m]
        humidity_RH, met, clo: Comfort parameters
    
    Returns:
        ComfortMapResult with PMV/PPD maps
    """
    from ontic.cfd.ns2d_qtt_native import qtt_2d_native_to_dense
    
    # Decompress to dense (for analysis only)
    T_K = qtt_2d_native_to_dense(T_qtt)
    T_C = T_K - 273.15  # Convert to Celsius
    
    psi = qtt_2d_native_to_dense(psi_qtt)
    
    # Compute velocities via finite difference
    dy = config.Ly / config.Ny
    dx = config.Lx / config.Nx
    
    # u = ∂ψ/∂y, v = -∂ψ/∂x
    u = torch.zeros_like(psi)
    v = torch.zeros_like(psi)
    
    u[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dy)
    v[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2 * dx)
    
    vmag = torch.sqrt(u**2 + v**2)
    
    # PMV/PPD
    pmv, ppd = pmv_ppd_vectorized(T_C, vmag, humidity_RH=humidity_RH, met=met, clo=clo)
    
    # Restrict to occupied zone
    y = torch.linspace(0, config.Ly, config.Ny, device=T_C.device)
    y_mask = (y >= occupied_height_range[0]) & (y <= occupied_height_range[1])
    occupied_cols = torch.where(y_mask)[0]
    
    if len(occupied_cols) > 0:
        pmv_occupied = pmv[:, occupied_cols]
        ppd_occupied = ppd[:, occupied_cols]
        
        pmv_mean = pmv_occupied.mean().item()
        pmv_std = pmv_occupied.std().item()
        ppd_mean = ppd_occupied.mean().item()
        comfort_fraction = (torch.abs(pmv_occupied) < 0.5).float().mean().item()
    else:
        pmv_mean = pmv.mean().item()
        pmv_std = pmv.std().item()
        ppd_mean = ppd.mean().item()
        comfort_fraction = (torch.abs(pmv) < 0.5).float().mean().item()
    
    return ComfortMapResult(
        pmv=pmv,
        ppd=ppd,
        T_air=T_C,
        velocity=vmag,
        pmv_mean=pmv_mean,
        pmv_std=pmv_std,
        ppd_mean=ppd_mean,
        comfort_fraction=comfort_fraction
    )
