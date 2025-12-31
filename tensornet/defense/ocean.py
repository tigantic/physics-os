"""
Ocean Domain for Hydroacoustic Simulation

The ocean is not uniform - sound speed varies with depth due to:
- Temperature (warm = faster)
- Pressure (deeper = faster)  
- Salinity (saltier = faster)

This creates the famous SOFAR (Sound Fixing and Ranging) channel
where sound can travel thousands of kilometers without attenuation.

Physics:
- Munk Profile: c(z) = 1500 * (1 + ε(η + e^(-η) - 1))
- Thermocline: Temperature gradient layer (sound reflects)
- SOFAR Axis: Minimum sound speed at ~1300m depth

Reference: Munk, W. H. (1974). "Sound channel in an exponentially 
stratified ocean, with application to SOFAR."
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import numpy as np


@dataclass
class SoundSpeedProfile:
    """Parameters for sound speed profile generation."""
    
    # Munk profile parameters
    c0: float = 1500.0          # Reference sound speed (m/s)
    z_channel: float = 1300.0   # SOFAR channel axis depth (m)
    scale_depth: float = 1300.0 # B parameter (m)
    epsilon: float = 0.0074     # Perturbation parameter
    
    # Optional thermocline
    thermocline_depth: Optional[float] = None  # Depth of sharp gradient (m)
    thermocline_strength: float = 30.0         # Speed jump (m/s)
    
    def describe(self) -> str:
        """Human-readable description."""
        return (
            f"Munk Profile: c₀={self.c0} m/s, "
            f"SOFAR axis={self.z_channel}m, "
            f"ε={self.epsilon}"
        )


class OceanDomain:
    """
    2D Ocean slice for acoustic simulation (Depth × Range).
    
    Coordinate System:
    - z: Depth (0 = surface, positive downward)
    - x: Range (horizontal distance from source)
    
    The domain contains:
    - Sound speed field c(z, x) from Munk profile
    - Optional terrain (seamounts, ridges)
    - Boundary conditions (surface, bottom)
    """
    
    def __init__(
        self,
        depth: float = 4000.0,
        range_km: float = 50.0,
        grid_res: float = 10.0,
        ssp: Optional[SoundSpeedProfile] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Create an ocean domain.
        
        Args:
            depth: Maximum depth in meters
            range_km: Horizontal range in kilometers
            grid_res: Grid resolution in meters per cell
            ssp: Sound speed profile parameters
            device: Torch device (auto-selects CUDA if available)
        """
        self.depth = depth
        self.range_m = range_km * 1000.0
        self.range_km = range_km
        self.res = grid_res
        
        # Grid dimensions
        self.nz = int(depth / grid_res)
        self.nx = int(self.range_m / grid_res)
        
        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Sound speed profile
        self.ssp = ssp or SoundSpeedProfile()
        
        # Generate sound speed field
        self.c = self._generate_sound_speed()
        
        # Terrain mask (1.0 = solid, 0.0 = water)
        self.terrain_mask = torch.zeros((self.nz, self.nx), device=self.device)
        
        # Bottom boundary (default: flat at max depth)
        self.bottom_depth = torch.full((self.nx,), float(self.nz - 1), device=self.device)
        
        c_stats = f"c ∈ [{self.c.min():.1f}, {self.c.max():.1f}] m/s"
        print(f"[OCEAN] Domain: {self.nz}×{self.nx} ({depth}m × {range_km}km)")
        print(f"[OCEAN] {self.ssp.describe()}")
        print(f"[OCEAN] {c_stats}, SOFAR axis at {self.ssp.z_channel}m")
        
    def _generate_sound_speed(self) -> torch.Tensor:
        """
        Generate sound speed field using Munk profile.
        
        The Munk profile models the SOFAR channel:
        - Sound speed decreases with depth (temperature drop)
        - Then increases again (pressure increase)
        - Minimum at ~1300m creates a waveguide
        
        Returns:
            Sound speed tensor (nz, nx) in m/s
        """
        # Depth vector
        z = torch.linspace(0, self.depth, self.nz, device=self.device)
        
        # Munk formula
        # η = 2(z - z_axis) / B
        eta = 2.0 * (z - self.ssp.z_channel) / self.ssp.scale_depth
        
        # c(z) = c₀ * (1 + ε * (η + e^(-η) - 1))
        c_profile = self.ssp.c0 * (1.0 + self.ssp.epsilon * (eta + torch.exp(-eta) - 1.0))
        
        # Add thermocline if specified
        if self.ssp.thermocline_depth is not None:
            # Sharp gradient at thermocline
            thermo_idx = int(self.ssp.thermocline_depth / self.res)
            # Smooth transition over ~50m
            transition_width = 50.0 / self.res
            
            z_idx = torch.arange(self.nz, device=self.device, dtype=torch.float32)
            transition = torch.sigmoid((z_idx - thermo_idx) / transition_width)
            
            # Add speed jump below thermocline
            c_profile = c_profile + self.ssp.thermocline_strength * transition
        
        # Expand to 2D (nz, nx) - uniform in range for now
        c = c_profile.view(-1, 1).expand(self.nz, self.nx).clone()
        
        return c
    
    def add_seamount(
        self,
        x_pos_km: float,
        height_m: float,
        width_km: float = 2.0,
    ) -> None:
        """
        Add a seamount (underwater mountain) to the domain.
        
        Seamounts create acoustic shadows - submarines can hide behind them.
        
        Args:
            x_pos_km: Horizontal position in km
            height_m: Height from seafloor in meters
            width_km: Width of the mountain base in km
        """
        x_idx = int((x_pos_km * 1000) / self.res)
        h_idx = int(height_m / self.res)
        width_idx = int((width_km * 1000) / self.res)
        
        # Gaussian hill shape
        x_grid = torch.arange(self.nx, device=self.device, dtype=torch.float32)
        sigma = width_idx / 3.0  # 3-sigma = base width
        shape = torch.exp(-((x_grid - x_idx) ** 2) / (2 * sigma ** 2))
        
        # Height profile at each x
        terrain_height = (shape * h_idx).long()
        
        # Update terrain mask
        for x in range(self.nx):
            h = terrain_height[x].item()
            if h > 0:
                z_start = max(0, self.nz - h)
                self.terrain_mask[z_start:, x] = 1.0
                self.bottom_depth[x] = float(z_start)
        
        peak_depth = self.depth - height_m
        print(f"[OCEAN] Seamount: x={x_pos_km}km, height={height_m}m, peak at {peak_depth}m depth")
        
    def add_ridge(
        self,
        x_start_km: float,
        x_end_km: float,
        height_m: float,
    ) -> None:
        """
        Add an underwater ridge (wall) spanning a range.
        
        Args:
            x_start_km: Start position in km
            x_end_km: End position in km
            height_m: Height from seafloor
        """
        x_start_idx = int((x_start_km * 1000) / self.res)
        x_end_idx = int((x_end_km * 1000) / self.res)
        h_idx = int(height_m / self.res)
        
        z_start = max(0, self.nz - h_idx)
        
        for x in range(max(0, x_start_idx), min(self.nx, x_end_idx)):
            self.terrain_mask[z_start:, x] = 1.0
            self.bottom_depth[x] = float(z_start)
            
        print(f"[OCEAN] Ridge: x=[{x_start_km}, {x_end_km}]km, height={height_m}m")
    
    def get_depth_at_range(self, range_km: float) -> float:
        """Get water depth at a given range (accounting for terrain)."""
        x_idx = int((range_km * 1000) / self.res)
        x_idx = max(0, min(x_idx, self.nx - 1))
        return self.bottom_depth[x_idx].item() * self.res
    
    def get_sound_speed_at(self, depth_m: float, range_km: float) -> float:
        """Get sound speed at a specific location."""
        z_idx = int(depth_m / self.res)
        x_idx = int((range_km * 1000) / self.res)
        
        z_idx = max(0, min(z_idx, self.nz - 1))
        x_idx = max(0, min(x_idx, self.nx - 1))
        
        return self.c[z_idx, x_idx].item()
    
    def is_solid(self, depth_m: float, range_km: float) -> bool:
        """Check if a location is inside terrain."""
        z_idx = int(depth_m / self.res)
        x_idx = int((range_km * 1000) / self.res)
        
        z_idx = max(0, min(z_idx, self.nz - 1))
        x_idx = max(0, min(x_idx, self.nx - 1))
        
        return self.terrain_mask[z_idx, x_idx].item() > 0.5
    
    def get_ssp_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get sound speed profile for plotting.
        
        Returns:
            (depths, speeds) as numpy arrays
        """
        depths = np.linspace(0, self.depth, self.nz)
        speeds = self.c[:, 0].cpu().numpy()  # Take first column
        return depths, speeds
    
    def summary(self) -> str:
        """Generate summary string."""
        water_cells = (self.terrain_mask < 0.5).sum().item()
        solid_cells = (self.terrain_mask >= 0.5).sum().item()
        
        return (
            f"Ocean Domain Summary:\n"
            f"  Dimensions: {self.nz}×{self.nx} cells ({self.depth}m × {self.range_km}km)\n"
            f"  Resolution: {self.res}m/cell\n"
            f"  Water cells: {water_cells:,} ({100*water_cells/(self.nz*self.nx):.1f}%)\n"
            f"  Solid cells: {solid_cells:,} ({100*solid_cells/(self.nz*self.nx):.1f}%)\n"
            f"  Sound speed: [{self.c.min():.1f}, {self.c.max():.1f}] m/s\n"
            f"  SOFAR axis: {self.ssp.z_channel}m depth\n"
            f"  Device: {self.device}"
        )


def create_deep_ocean(range_km: float = 50.0, with_seamount: bool = True) -> OceanDomain:
    """
    Create a standard deep ocean domain for testing.
    
    This represents a typical open ocean scenario with:
    - 4km depth (average ocean depth)
    - Munk sound speed profile
    - Optional seamount for shadow testing
    """
    ocean = OceanDomain(depth=4000.0, range_km=range_km, grid_res=10.0)
    
    if with_seamount:
        # Add seamount at 20km, 2km high
        ocean.add_seamount(x_pos_km=20.0, height_m=2000.0, width_km=3.0)
        
    return ocean


def create_continental_shelf(range_km: float = 30.0) -> OceanDomain:
    """
    Create a shallow continental shelf scenario.
    
    Shallow water propagation is very different from deep ocean:
    - Sound bounces between surface and bottom
    - Higher attenuation
    - More complex multipath
    """
    ocean = OceanDomain(depth=200.0, range_km=range_km, grid_res=2.0)
    
    # Add some bottom roughness
    for x in range(0, int(range_km), 5):
        height = np.random.uniform(10, 30)
        ocean.add_seamount(x_pos_km=float(x), height_m=height, width_km=1.0)
        
    return ocean
