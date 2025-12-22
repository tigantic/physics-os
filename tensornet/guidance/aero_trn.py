"""
Aero-TRN (Aerodynamic Terrain-Referenced Navigation)
=====================================================

Phase 22: Terrain navigation using aerodynamic signatures.

Uses aerodynamic measurements (pressure, drag) to correlate with
terrain features for navigation in GPS-denied environments.

Key Concept:
- Surface pressure patterns depend on vehicle position/altitude
- Terrain features create unique aerodynamic signatures
- Cross-correlation with pre-computed maps enables positioning

References:
    - Van Graas & Braasch, "GPS Alternatives" Navigation (2001)
    - Metzger et al., "Terrain Contour Matching" AIAA J Guidance (1983)
    - Lu & Strganac, "Aerodynamic Terrain Correlation" AIAA (2005)

Constitution Compliance: Article II.1 (Guidance Systems)
"""

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple, Callable
from dataclasses import dataclass, field
import math


# =============================================================================
# Constants
# =============================================================================

# Earth parameters
EARTH_RADIUS = 6371000.0  # m
GRAVITY = 9.81  # m/s²

# Atmosphere
SEA_LEVEL_PRESSURE = 101325.0  # Pa
SEA_LEVEL_TEMPERATURE = 288.15  # K
LAPSE_RATE = 0.0065  # K/m


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TerrainMap:
    """
    Pre-computed terrain elevation map.
    
    Attributes:
        elevation: 2D elevation grid (m)
        lat_min, lat_max: Latitude bounds (degrees)
        lon_min, lon_max: Longitude bounds (degrees)
        resolution: Grid resolution (m)
    """
    elevation: Tensor
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution: float
    
    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.elevation.shape)
    
    def get_elevation(self, lat: float, lon: float) -> float:
        """Interpolate elevation at given coordinates."""
        # Normalize to grid indices
        ny, nx = self.shape
        i = (lat - self.lat_min) / (self.lat_max - self.lat_min) * (ny - 1)
        j = (lon - self.lon_min) / (self.lon_max - self.lon_min) * (nx - 1)
        
        # Bilinear interpolation
        i0 = int(i)
        j0 = int(j)
        i1 = min(i0 + 1, ny - 1)
        j1 = min(j0 + 1, nx - 1)
        
        di = i - i0
        dj = j - j0
        
        z = ((1 - di) * (1 - dj) * self.elevation[i0, j0] +
             di * (1 - dj) * self.elevation[i1, j0] +
             (1 - di) * dj * self.elevation[i0, j1] +
             di * dj * self.elevation[i1, j1])
        
        return z.item()


@dataclass
class AeroSignature:
    """
    Aerodynamic signature at a location.
    
    Attributes:
        pressure_pattern: Surface pressure distribution
        drag_coefficient: Estimated drag coefficient
        gradient: Local gradient direction [dx, dy]
        roughness: Terrain roughness estimate
    """
    pressure_pattern: Tensor
    drag_coefficient: float
    gradient: Tuple[float, float]
    roughness: float


@dataclass
class NavigationState:
    """
    Current navigation estimate.
    
    Attributes:
        lat, lon: Estimated position (degrees)
        altitude: Altitude above sea level (m)
        velocity: [vx, vy, vz] velocity (m/s)
        covariance: Position uncertainty (3x3 matrix)
    """
    lat: float
    lon: float
    altitude: float
    velocity: Tuple[float, float, float]
    covariance: Optional[Tensor] = None


@dataclass
class AeroTRNConfig:
    """
    Configuration for Aero-TRN system.
    
    Attributes:
        correlation_window: Size of correlation window (m)
        search_radius: Maximum search radius (m)
        min_correlation: Minimum correlation for valid match
        pressure_weight: Weight for pressure correlation
        gradient_weight: Weight for gradient correlation
        update_rate: Navigation update rate (Hz)
    """
    correlation_window: float = 5000.0
    search_radius: float = 10000.0
    min_correlation: float = 0.6
    pressure_weight: float = 0.7
    gradient_weight: float = 0.3
    update_rate: float = 1.0


# =============================================================================
# Atmospheric Model
# =============================================================================

def standard_atmosphere(altitude: float) -> Tuple[float, float, float]:
    """
    Standard atmosphere model (troposphere only).
    
    Args:
        altitude: Altitude above sea level (m)
        
    Returns:
        (temperature, pressure, density) in (K, Pa, kg/m³)
    """
    if altitude < 0:
        altitude = 0
    if altitude > 11000:
        altitude = 11000  # Troposphere limit
    
    T = SEA_LEVEL_TEMPERATURE - LAPSE_RATE * altitude
    p = SEA_LEVEL_PRESSURE * (T / SEA_LEVEL_TEMPERATURE) ** 5.256
    rho = p / (287.05 * T)
    
    return T, p, rho


def pressure_altitude(pressure: float) -> float:
    """
    Compute pressure altitude from measured pressure.
    
    Inverts the barometric formula.
    
    Args:
        pressure: Measured pressure (Pa)
        
    Returns:
        Altitude (m)
    """
    exponent = 1 / 5.256
    T_ratio = (pressure / SEA_LEVEL_PRESSURE) ** exponent
    altitude = (SEA_LEVEL_TEMPERATURE * (1 - T_ratio)) / LAPSE_RATE
    return altitude


# =============================================================================
# Signature Generation
# =============================================================================

def compute_terrain_gradient(
    terrain: TerrainMap,
    lat: float,
    lon: float,
    radius: float = 1000.0
) -> Tuple[float, float]:
    """
    Compute local terrain gradient.
    
    Args:
        terrain: Terrain elevation map
        lat, lon: Center position
        radius: Sampling radius (m)
        
    Returns:
        (dz/dx, dz/dy) gradient in m/m
    """
    # Convert radius to degrees (approximate)
    d_lat = radius / 111000  # degrees per meter
    d_lon = radius / (111000 * math.cos(math.radians(lat)))
    
    z_center = terrain.get_elevation(lat, lon)
    z_north = terrain.get_elevation(lat + d_lat, lon)
    z_south = terrain.get_elevation(lat - d_lat, lon)
    z_east = terrain.get_elevation(lat, lon + d_lon)
    z_west = terrain.get_elevation(lat, lon - d_lon)
    
    dz_dx = (z_east - z_west) / (2 * radius)
    dz_dy = (z_north - z_south) / (2 * radius)
    
    return dz_dx, dz_dy


def compute_aero_signature(
    terrain: TerrainMap,
    lat: float,
    lon: float,
    altitude: float,
    velocity: float,
    window_size: float = 5000.0,
    n_samples: int = 21
) -> AeroSignature:
    """
    Compute aerodynamic signature for a location.
    
    Models how terrain affects surface pressure distribution
    on a vehicle flying at given altitude/velocity.
    
    Args:
        terrain: Terrain elevation map
        lat, lon: Center position
        altitude: Flight altitude (m)
        velocity: Flight velocity (m/s)
        window_size: Correlation window size (m)
        n_samples: Number of samples in window
        
    Returns:
        AeroSignature for the location
    """
    # Sample terrain in window
    d_lat = window_size / 111000
    d_lon = window_size / (111000 * math.cos(math.radians(lat)))
    
    lats = torch.linspace(lat - d_lat/2, lat + d_lat/2, n_samples)
    lons = torch.linspace(lon - d_lon/2, lon + d_lon/2, n_samples)
    
    elevations = torch.zeros(n_samples, n_samples, dtype=torch.float64)
    for i, lt in enumerate(lats):
        for j, ln in enumerate(lons):
            elevations[i, j] = terrain.get_elevation(lt.item(), ln.item())
    
    # Height above terrain
    height_above_terrain = altitude - elevations
    height_above_terrain = torch.clamp(height_above_terrain, min=100.0)
    
    # Pressure perturbation model
    # ΔP/P ~ (terrain_slope * velocity²) / (height * gravity)
    _, p_inf, rho_inf = standard_atmosphere(altitude)
    q_inf = 0.5 * rho_inf * velocity**2
    
    # Compute terrain slopes
    dz_dx = torch.zeros_like(elevations)
    dz_dy = torch.zeros_like(elevations)
    
    dz_dx[1:-1, :] = (elevations[2:, :] - elevations[:-2, :]) / (2 * window_size / n_samples)
    dz_dy[:, 1:-1] = (elevations[:, 2:] - elevations[:, :-2]) / (2 * window_size / n_samples)
    
    # Pressure pattern (simplified model)
    slope_magnitude = torch.sqrt(dz_dx**2 + dz_dy**2)
    pressure_perturbation = q_inf * slope_magnitude / height_above_terrain
    pressure_pattern = p_inf + pressure_perturbation
    
    # Drag coefficient estimate (increases with terrain roughness)
    base_cd = 0.3
    roughness = torch.std(elevations).item()
    cd_terrain_effect = 0.01 * roughness / 1000  # Normalized by 1km
    cd = base_cd + cd_terrain_effect
    
    # Gradient at center
    gradient = compute_terrain_gradient(terrain, lat, lon)
    
    return AeroSignature(
        pressure_pattern=pressure_pattern,
        drag_coefficient=cd,
        gradient=gradient,
        roughness=roughness
    )


# =============================================================================
# Correlation Engine
# =============================================================================

def cross_correlate_signatures(
    measured: AeroSignature,
    candidate: AeroSignature
) -> float:
    """
    Cross-correlate two aerodynamic signatures.
    
    Args:
        measured: Measured signature from sensors
        candidate: Candidate signature from map
        
    Returns:
        Correlation coefficient [0, 1]
    """
    # Normalize pressure patterns
    p1 = measured.pressure_pattern
    p2 = candidate.pressure_pattern
    
    p1_norm = (p1 - p1.mean()) / (p1.std() + 1e-10)
    p2_norm = (p2 - p2.mean()) / (p2.std() + 1e-10)
    
    # Correlation
    correlation = (p1_norm * p2_norm).mean()
    
    # Gradient similarity
    g1 = torch.tensor(measured.gradient)
    g2 = torch.tensor(candidate.gradient)
    grad_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
    
    # Weighted combination
    total = 0.7 * correlation.item() + 0.3 * (grad_sim + 1) / 2
    
    return max(0.0, min(1.0, total))


class AeroTRN:
    """
    Aerodynamic Terrain-Referenced Navigation System.
    
    Uses aerodynamic signatures to correlate with terrain
    maps for navigation in GPS-denied environments.
    
    Attributes:
        terrain_map: Pre-computed terrain elevation data
        config: System configuration
        current_state: Current navigation estimate
    """
    
    def __init__(
        self,
        terrain_map: TerrainMap,
        config: Optional[AeroTRNConfig] = None
    ):
        """
        Initialize Aero-TRN system.
        
        Args:
            terrain_map: Terrain elevation map
            config: Configuration parameters
        """
        self.terrain_map = terrain_map
        self.config = config or AeroTRNConfig()
        self.current_state: Optional[NavigationState] = None
        
        # Pre-computed signature cache
        self._signature_cache: Dict[Tuple[float, float], AeroSignature] = {}
    
    def initialize(
        self,
        lat: float,
        lon: float,
        altitude: float,
        velocity: Tuple[float, float, float]
    ) -> NavigationState:
        """
        Initialize navigation state.
        
        Args:
            lat, lon: Initial position estimate (degrees)
            altitude: Initial altitude (m)
            velocity: Initial velocity (m/s)
            
        Returns:
            Initial navigation state
        """
        self.current_state = NavigationState(
            lat=lat,
            lon=lon,
            altitude=altitude,
            velocity=velocity,
            covariance=torch.eye(3, dtype=torch.float64) * 1000**2  # 1km uncertainty
        )
        return self.current_state
    
    def update(
        self,
        pressure_measurement: Tensor,
        drag_measurement: float,
        velocity: float,
        dt: float
    ) -> NavigationState:
        """
        Update navigation with new measurements.
        
        Args:
            pressure_measurement: Measured pressure pattern
            drag_measurement: Measured drag coefficient
            velocity: Current velocity magnitude (m/s)
            dt: Time since last update (s)
            
        Returns:
            Updated navigation state
        """
        if self.current_state is None:
            raise ValueError("Navigation not initialized. Call initialize() first.")
        
        # Propagate state
        vx, vy, vz = self.current_state.velocity
        
        # Simple dead reckoning
        d_lat = vy * dt / 111000
        d_lon = vx * dt / (111000 * math.cos(math.radians(self.current_state.lat)))
        d_alt = vz * dt
        
        predicted_lat = self.current_state.lat + d_lat
        predicted_lon = self.current_state.lon + d_lon
        predicted_alt = self.current_state.altitude + d_alt
        
        # Build measured signature
        measured_signature = AeroSignature(
            pressure_pattern=pressure_measurement,
            drag_coefficient=drag_measurement,
            gradient=(0.0, 0.0),
            roughness=0.0
        )
        
        # Search for best match
        best_match = self._search_correlation(
            measured_signature,
            predicted_lat,
            predicted_lon,
            predicted_alt,
            velocity
        )
        
        # Update state with matched position
        self.current_state = NavigationState(
            lat=best_match[0],
            lon=best_match[1],
            altitude=predicted_alt,
            velocity=self.current_state.velocity,
            covariance=self._update_covariance(best_match[2])
        )
        
        return self.current_state
    
    def _search_correlation(
        self,
        measured: AeroSignature,
        lat_center: float,
        lon_center: float,
        altitude: float,
        velocity: float
    ) -> Tuple[float, float, float]:
        """
        Search for best correlation match.
        
        Returns:
            (lat, lon, correlation) of best match
        """
        # Search grid
        search_km = self.config.search_radius / 1000
        n_search = 11
        
        d_lat = search_km / 111
        d_lon = search_km / (111 * math.cos(math.radians(lat_center)))
        
        best_lat = lat_center
        best_lon = lon_center
        best_corr = 0.0
        
        for i in range(n_search):
            for j in range(n_search):
                lat = lat_center + (i - n_search//2) * d_lat / n_search
                lon = lon_center + (j - n_search//2) * d_lon / n_search
                
                # Compute candidate signature
                candidate = compute_aero_signature(
                    self.terrain_map,
                    lat, lon,
                    altitude,
                    velocity,
                    self.config.correlation_window
                )
                
                # Correlate
                corr = cross_correlate_signatures(measured, candidate)
                
                if corr > best_corr:
                    best_corr = corr
                    best_lat = lat
                    best_lon = lon
        
        return best_lat, best_lon, best_corr
    
    def _update_covariance(self, correlation: float) -> Tensor:
        """Update covariance based on correlation quality."""
        if self.current_state is None:
            return torch.eye(3, dtype=torch.float64) * 1000**2
        
        # Higher correlation = lower uncertainty
        uncertainty = 100.0 / (correlation + 0.1)  # meters
        
        # Kalman-like update (simplified)
        P_prior = self.current_state.covariance
        R = torch.eye(3, dtype=torch.float64) * uncertainty**2
        
        P_post = P_prior * (1 - correlation) + R * correlation
        
        return P_post
    
    def get_position_uncertainty(self) -> float:
        """Get current position uncertainty (1-sigma, meters)."""
        if self.current_state is None or self.current_state.covariance is None:
            return float('inf')
        
        return torch.sqrt(torch.trace(self.current_state.covariance[:2, :2])).item()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data structures
    'TerrainMap',
    'AeroSignature',
    'NavigationState',
    'AeroTRNConfig',
    # Functions
    'standard_atmosphere',
    'pressure_altitude',
    'compute_terrain_gradient',
    'compute_aero_signature',
    'cross_correlate_signatures',
    # Classes
    'AeroTRN',
]
