"""
Wind Farm Wake Physics Engine

Phase 5A: The Wake Physics
========================
Jensen Park Wake Model - Standard industry model for wind turbine wake effects.

The "Energy Shadow": When wind passes through a turbine rotor, it loses momentum.
Downstream turbines receive slower air and generate less power. This module
calculates the velocity deficit and enables layout optimization.

Key Physics:
- Wake expands linearly downstream: r_wake = r_rotor + k * x
- Velocity deficit follows (r_rotor / r_wake)^2 decay
- Wake decay constant k: 0.075 (land), 0.04 (offshore)
- Optimal axial induction factor a ≈ 0.33 (Betz limit)

Commercial Value:
- 10-15% power loss from poor layouts
- $100k-$500k annual revenue recovery per farm
- Target: Orsted, Shell Energy, Equinor

References:
    Jensen, N.O. (1983). "A Note on Wind Generator Interaction."
    Risø National Laboratory Report M-2411. Roskilde, Denmark.
    
    Betz, A. (1919). "Das Maximum der theoretisch möglichen Ausnützung
    des Windes durch Windmotoren." Zeitschrift für das gesamte Turbinenwesen.
    
    Katic, I., Højstrup, J., & Jensen, N.O. (1986). "A Simple Model for
    Cluster Efficiency." Proceedings of EWEC '86, Rome, Italy.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TurbineSpec:
    """Wind turbine specification."""
    x: float          # East-West position (m)
    y: float          # Height above sea level (m)
    z: float          # North-South / streamwise position (m)
    radius: float     # Rotor radius (m)
    yaw: float        # Yaw angle (degrees) - 0 = facing wind
    rated_power: float = 5.0  # MW
    hub_height: float = 100.0  # m
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'radius': self.radius, 'yaw': self.yaw,
            'rated_power': self.rated_power,
            'hub_height': self.hub_height
        }


class WindFarm:
    """
    Wind Farm Wake Model using Jensen Park formulation.
    
    Calculates velocity deficits and power output for multi-turbine arrays.
    Supports GPU acceleration via PyTorch tensors.
    
    Example:
        >>> turbines = [
        ...     {'x': 0, 'y': 100, 'z': 0, 'radius': 40, 'yaw': 0},
        ...     {'x': 0, 'y': 100, 'z': 400, 'radius': 40, 'yaw': 0},
        ... ]
        >>> farm = WindFarm(turbines, environment='offshore')
        >>> field = torch.ones((3, 100, 50, 50)) * 12.0
        >>> farm.apply_wakes(field, grid_resolution=10.0)
        >>> power_mw = farm.calculate_power_output(field, grid_resolution=10.0)
    """
    
    # Wake decay constants (empirical)
    WAKE_DECAY_LAND = 0.075
    WAKE_DECAY_OFFSHORE = 0.04
    
    # Air properties
    AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m³
    
    # Turbine efficiency
    BETZ_LIMIT = 0.593  # Maximum theoretical Cp
    TYPICAL_CP = 0.45   # Realistic power coefficient
    AXIAL_INDUCTION = 0.33  # Optimal a for max power
    
    def __init__(
        self,
        turbines: List[Dict],
        environment: str = 'offshore',
        air_density: Optional[float] = None
    ):
        """
        Initialize wind farm.
        
        Args:
            turbines: List of turbine dicts with keys:
                      x, y, z (position), radius, yaw
            environment: 'offshore' or 'land' (affects wake decay)
            air_density: Override air density (kg/m³)
        """
        self.turbines = turbines
        self.environment = environment
        
        # Select wake decay constant
        if environment == 'offshore':
            self.k = self.WAKE_DECAY_OFFSHORE
        else:
            self.k = self.WAKE_DECAY_LAND
            
        self.air_density = air_density or self.AIR_DENSITY_SEA_LEVEL
        
        # Validate turbines
        for i, t in enumerate(turbines):
            required = ['x', 'y', 'z', 'radius', 'yaw']
            for key in required:
                if key not in t:
                    raise ValueError(f"Turbine {i} missing required key: {key}")

    def apply_wakes(
        self,
        wind_field: torch.Tensor,
        grid_resolution: float,
        superposition: str = 'linear'
    ) -> torch.Tensor:
        """
        Apply wake velocity deficits to wind field.
        
        Uses Jensen Park Wake Model:
        - Wake radius expands: r_wake = r_rotor + k * x_downstream
        - Velocity deficit: u_wake = u_free * (1 - 2a * (r_rotor/r_wake)²)
        
        Args:
            wind_field: Tensor of shape (3, D, H, W) - velocity components
                       Index 0 = U (streamwise), 1 = V, 2 = W
            grid_resolution: Meters per grid cell
            superposition: Wake combination method ('linear' or 'rss')
            
        Returns:
            Modified wind_field with wake deficits applied
        """
        u_field = wind_field[0]  # Streamwise velocity
        depth, height, width = u_field.shape
        device = wind_field.device
        
        # Clone to avoid in-place mutation issues
        wake_field = u_field.clone()
        
        # Pre-compute grid coordinates
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        for t in self.turbines:
            # Convert world position to grid indices
            idx_x = int(t['x'] / grid_resolution)
            idx_y = int(t['y'] / grid_resolution)
            idx_z = int(t['z'] / grid_resolution)
            
            r_rotor = t['radius']
            
            # Apply wake to all downstream slices
            for z in range(idx_z + 1, depth):
                distance_downstream = (z - idx_z) * grid_resolution
                
                # Wake expansion (linear with distance)
                r_wake = r_rotor + (self.k * distance_downstream)
                
                # Jensen velocity deficit factor
                # (r_rotor / r_wake)² term
                loss_factor = (r_rotor / r_wake) ** 2
                
                # Calculate deficit magnitude
                # u_deficit = u_free * 2 * a * (r/R)²
                deficit = 2.0 * self.AXIAL_INDUCTION * loss_factor
                
                # Create circular wake mask
                dist_sq = (x_grid - idx_x).float()**2 + (y_grid - idx_y).float()**2
                r_wake_grid = r_wake / grid_resolution
                mask = dist_sq <= r_wake_grid**2
                
                # Apply deficit (multiplicative reduction)
                # u_new = u_old * (1 - deficit) within wake zone
                if mask.any():
                    wake_field[z][mask] *= (1.0 - deficit)
        
        # Clamp to prevent negative velocities
        wake_field = torch.clamp(wake_field, min=0.0)
        
        # Update field
        wind_field[0] = wake_field
        return wind_field

    def calculate_power_output(
        self,
        wind_field: torch.Tensor,
        grid_resolution: float,
        power_coefficient: Optional[float] = None
    ) -> float:
        """
        Calculate total power output of the farm.
        
        Power equation: P = 0.5 * ρ * A * v³ * Cp
        
        Args:
            wind_field: Tensor (3, D, H, W) with wake deficits applied
            grid_resolution: Meters per grid cell
            power_coefficient: Turbine Cp (default 0.45)
            
        Returns:
            Total power output in MW
        """
        cp = power_coefficient or self.TYPICAL_CP
        total_power_watts = 0.0
        
        for t in self.turbines:
            # Grid indices for hub location
            ix = int(t['x'] / grid_resolution)
            iy = int(t['y'] / grid_resolution)
            iz = int(t['z'] / grid_resolution)
            
            # Bounds check
            d, h, w = wind_field.shape[1], wind_field.shape[2], wind_field.shape[3]
            if not (0 <= iz < d and 0 <= iy < h and 0 <= ix < w):
                continue
                
            # Sample velocity at hub
            v_hub = float(wind_field[0, iz, iy, ix])
            
            # Rotor swept area
            area = np.pi * (t['radius'] ** 2)
            
            # Power: P = 0.5 * ρ * A * v³ * Cp
            power_watts = 0.5 * self.air_density * area * (v_hub ** 3) * cp
            total_power_watts += power_watts
        
        # Convert to MW
        return total_power_watts / 1_000_000.0

    def calculate_capacity_factor(
        self,
        actual_power: float,
        wind_speed: float = 12.0
    ) -> float:
        """
        Calculate capacity factor (actual / theoretical max).
        
        Args:
            actual_power: Measured power output (MW)
            wind_speed: Free-stream wind speed (m/s)
            
        Returns:
            Capacity factor (0.0 - 1.0)
        """
        # Theoretical max power at this wind speed
        theoretical = 0.0
        for t in self.turbines:
            area = np.pi * (t['radius'] ** 2)
            power = 0.5 * self.air_density * area * (wind_speed ** 3) * self.TYPICAL_CP
            theoretical += power
            
        theoretical_mw = theoretical / 1_000_000.0
        
        if theoretical_mw > 0:
            return actual_power / theoretical_mw
        return 0.0

    def get_wake_centerline(
        self,
        turbine_idx: int,
        max_distance: float = 1000.0,
        resolution: float = 10.0
    ) -> List[Dict]:
        """
        Get wake centerline trajectory for visualization.
        
        Args:
            turbine_idx: Which turbine to trace
            max_distance: How far downstream (m)
            resolution: Point spacing (m)
            
        Returns:
            List of {x, y, z, radius} points along wake
        """
        t = self.turbines[turbine_idx]
        points = []
        
        distance = 0.0
        while distance <= max_distance:
            r_wake = t['radius'] + (self.k * distance)
            points.append({
                'x': t['x'],
                'y': t['y'],
                'z': t['z'] + distance,
                'radius': r_wake
            })
            distance += resolution
            
        return points

    def annual_energy_production(
        self,
        power_mw: float,
        capacity_factor: float = 0.45
    ) -> float:
        """
        Estimate Annual Energy Production (AEP) in MWh.
        
        Args:
            power_mw: Rated/calculated power (MW)
            capacity_factor: Typical offshore = 0.45
            
        Returns:
            AEP in MWh
        """
        hours_per_year = 24 * 365  # 8760
        return power_mw * hours_per_year * capacity_factor

    def annual_revenue(
        self,
        power_mw: float,
        price_per_mwh: float = 50.0,
        capacity_factor: float = 1.0  # Assume already at operating point
    ) -> float:
        """
        Calculate annual revenue from power output.
        
        Args:
            power_mw: Power at current wind speed
            price_per_mwh: Wholesale electricity price ($/MWh)
            capacity_factor: Operating capacity factor
            
        Returns:
            Annual revenue in USD
        """
        hours_per_year = 24 * 365
        return power_mw * hours_per_year * price_per_mwh * capacity_factor

    def __repr__(self) -> str:
        return (
            f"WindFarm(turbines={len(self.turbines)}, "
            f"environment='{self.environment}', k={self.k})"
        )


class WakeOptimizer:
    """
    Gradient-based wind farm layout optimization.
    
    Uses numerical gradients to adjust turbine positions
    for maximum power output.
    """
    
    def __init__(
        self,
        farm: WindFarm,
        bounds: Dict[str, tuple],
        grid_resolution: float = 10.0
    ):
        """
        Args:
            farm: WindFarm instance to optimize
            bounds: {'x': (min, max), 'z': (min, max)} position limits
            grid_resolution: Grid cell size (m)
        """
        self.farm = farm
        self.bounds = bounds
        self.grid_resolution = grid_resolution
        
    def optimize(
        self,
        wind_field: torch.Tensor,
        iterations: int = 100,
        step_size: float = 10.0,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Run layout optimization.
        
        Args:
            wind_field: Base wind field (3, D, H, W)
            iterations: Optimization steps
            step_size: Position adjustment per step (m)
            verbose: Print progress
            
        Returns:
            Optimized turbine positions
        """
        best_power = 0.0
        best_layout = [t.copy() for t in self.farm.turbines]
        
        for i in range(iterations):
            # Evaluate current layout
            field = wind_field.clone()
            self.farm.apply_wakes(field, self.grid_resolution)
            power = self.farm.calculate_power_output(field, self.grid_resolution)
            
            if power > best_power:
                best_power = power
                best_layout = [t.copy() for t in self.farm.turbines]
            
            # Perturb positions (simple random search)
            for t in self.farm.turbines[1:]:  # Keep first turbine fixed
                # Random step
                dx = np.random.uniform(-step_size, step_size)
                
                # Apply with bounds
                new_x = t['x'] + dx
                if self.bounds['x'][0] <= new_x <= self.bounds['x'][1]:
                    t['x'] = new_x
                    
            if verbose and i % 20 == 0:
                print(f"  Iteration {i}: Power = {power:.3f} MW (best: {best_power:.3f})")
        
        # Restore best
        self.farm.turbines = best_layout
        return best_layout
