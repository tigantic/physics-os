"""
Blue Marble Topographic Substrate
=================================

Procedural generation of a NASA Blue Marble style base map.
This provides the geographic "Clear Earth" context without clouds.

Color scheme:
- Deep ocean: Dark navy (#0A1628)
- Shallow ocean: Medium blue (#1E4D6B)  
- Coastal: Light cyan (#4A8BA8)
- Lowlands: Green (#4A8A4A)
- Plains: Tan/yellow (#C4A86A)
- Highlands: Brown (#8A6A4A)
- Mountains: Light gray (#B0A090)
- Snow caps: White (#F0F0F0)
- Deserts: Sandy (#D4C4A0)
"""

import numpy as np
from typing import Tuple


def generate_blue_marble(width: int = 720, height: int = 360) -> np.ndarray:
    """
    Generate a procedural Blue Marble style topographic map.
    
    This creates a realistic-looking Earth base map with:
    - Ocean bathymetry (deep → shallow gradients)
    - Continental shelves
    - Land elevation (lowlands → mountains)
    - Desert regions
    - Polar ice
    
    Args:
        width: Output width in pixels
        height: Output height in pixels
        
    Returns:
        RGBA numpy array (height, width, 4) with values 0-1
    """
    # Create coordinate grids
    lon = np.linspace(-180, 180, width)
    lat = np.linspace(-90, 90, height)
    LON, LAT = np.meshgrid(lon, lat)
    
    # Initialize output
    output = np.zeros((height, width, 4), dtype=np.float32)
    output[:, :, 3] = 1.0  # Full opacity
    
    # ==========================================================================
    # OCEAN BATHYMETRY (base layer)
    # ==========================================================================
    # Deep ocean gradient based on latitude
    depth_factor = 1.0 - np.abs(LAT) / 90.0 * 0.3
    
    # Deep ocean color
    output[:, :, 0] = 0.04 * depth_factor  # R
    output[:, :, 1] = 0.09 * depth_factor  # G
    output[:, :, 2] = 0.16 * depth_factor  # B
    
    # ==========================================================================
    # CONTINENTAL MASKS (simplified geometry)
    # ==========================================================================
    
    def smooth_mask(condition, blur_size=3):
        """Create a smooth transition mask."""
        from scipy.ndimage import gaussian_filter
        mask = condition.astype(np.float32)
        return gaussian_filter(mask, sigma=blur_size)
    
    # North America
    na_mask = (
        (LON >= -170) & (LON <= -50) & 
        (LAT >= 15) & (LAT <= 75) &
        ~((LON >= -100) & (LON <= -80) & (LAT >= 20) & (LAT <= 32))  # Gulf
    )
    
    # South America
    sa_mask = (
        (LON >= -85) & (LON <= -32) & 
        (LAT >= -58) & (LAT <= 15)
    )
    
    # Europe
    eu_mask = (
        (LON >= -12) & (LON <= 40) & 
        (LAT >= 35) & (LAT <= 72)
    )
    
    # Africa  
    af_mask = (
        (LON >= -20) & (LON <= 55) & 
        (LAT >= -36) & (LAT <= 38)
    )
    
    # Asia
    as_mask = (
        (LON >= 25) & (LON <= 180) & 
        (LAT >= 5) & (LAT <= 78)
    ) | (
        (LON >= -180) & (LON <= -168) & 
        (LAT >= 50) & (LAT <= 72)  # Eastern Russia
    )
    
    # Australia
    au_mask = (
        (LON >= 110) & (LON <= 155) & 
        (LAT >= -45) & (LAT <= -10)
    )
    
    # Antarctica
    an_mask = LAT <= -60
    
    # Greenland
    gr_mask = (
        (LON >= -75) & (LON <= -10) & 
        (LAT >= 58) & (LAT <= 84)
    )
    
    # Combine land masks
    land_mask = na_mask | sa_mask | eu_mask | af_mask | as_mask | au_mask | an_mask | gr_mask
    
    # ==========================================================================
    # ELEVATION MODEL (procedural)
    # ==========================================================================
    
    # Base elevation noise (multi-octave)
    np.random.seed(42)  # Reproducible
    
    def perlin_like(shape, scale=50):
        """Simple noise approximation."""
        from scipy.ndimage import gaussian_filter
        noise = np.random.rand(*shape)
        return gaussian_filter(noise, sigma=scale)
    
    # Large-scale terrain
    terrain_large = perlin_like((height, width), scale=30)
    terrain_medium = perlin_like((height, width), scale=15)
    terrain_small = perlin_like((height, width), scale=5)
    
    elevation = (
        0.5 * terrain_large + 
        0.3 * terrain_medium + 
        0.2 * terrain_small
    )
    
    # Normalize to 0-1
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    
    # ==========================================================================
    # APPLY LAND COLORS
    # ==========================================================================
    
    # Smooth the land mask
    land_smooth = smooth_mask(land_mask, blur_size=2)
    
    # Land base colors based on elevation
    # Lowlands: green
    lowland_r = 0.29
    lowland_g = 0.54
    lowland_b = 0.29
    
    # Mid elevation: tan/brown
    mid_r = 0.77
    mid_g = 0.66
    mid_b = 0.42
    
    # High elevation: gray/white
    high_r = 0.69
    high_g = 0.63
    high_b = 0.56
    
    # Interpolate based on elevation
    land_r = lowland_r + (mid_r - lowland_r) * elevation + (high_r - mid_r) * np.maximum(0, elevation - 0.5) * 2
    land_g = lowland_g + (mid_g - lowland_g) * elevation + (high_g - mid_g) * np.maximum(0, elevation - 0.5) * 2
    land_b = lowland_b + (mid_b - lowland_b) * elevation + (high_b - mid_b) * np.maximum(0, elevation - 0.5) * 2
    
    # ==========================================================================
    # SPECIAL REGIONS
    # ==========================================================================
    
    # Sahara Desert
    sahara_mask = (LON >= -15) & (LON <= 35) & (LAT >= 15) & (LAT <= 35)
    sahara_smooth = smooth_mask(sahara_mask & land_mask, blur_size=5)
    
    # Arabian Desert
    arabian_mask = (LON >= 35) & (LON <= 60) & (LAT >= 15) & (LAT <= 35)
    arabian_smooth = smooth_mask(arabian_mask & land_mask, blur_size=5)
    
    # Gobi Desert
    gobi_mask = (LON >= 90) & (LON <= 120) & (LAT >= 35) & (LAT <= 50)
    gobi_smooth = smooth_mask(gobi_mask & land_mask, blur_size=4)
    
    # Australian Outback
    outback_mask = (LON >= 120) & (LON <= 145) & (LAT >= -30) & (LAT <= -18)
    outback_smooth = smooth_mask(outback_mask & land_mask, blur_size=4)
    
    desert_smooth = np.maximum.reduce([sahara_smooth, arabian_smooth, gobi_smooth, outback_smooth])
    
    # Desert colors (sandy tan)
    desert_r = 0.83
    desert_g = 0.77
    desert_b = 0.63
    
    # Blend desert
    land_r = land_r * (1 - desert_smooth * 0.7) + desert_r * desert_smooth * 0.7
    land_g = land_g * (1 - desert_smooth * 0.7) + desert_g * desert_smooth * 0.7
    land_b = land_b * (1 - desert_smooth * 0.7) + desert_b * desert_smooth * 0.7
    
    # ==========================================================================
    # MOUNTAIN RANGES (simplified)
    # ==========================================================================
    
    # Rockies
    rockies = (LON >= -125) & (LON <= -100) & (LAT >= 30) & (LAT <= 60)
    
    # Andes
    andes = (LON >= -80) & (LON <= -65) & (LAT >= -55) & (LAT <= 10)
    
    # Alps
    alps = (LON >= 5) & (LON <= 18) & (LAT >= 43) & (LAT <= 48)
    
    # Himalayas
    himalayas = (LON >= 70) & (LON <= 100) & (LAT >= 25) & (LAT <= 40)
    
    mountains = smooth_mask(rockies | andes | alps | himalayas, blur_size=3)
    
    # Add mountain boost to elevation
    mountain_boost = mountains * 0.4
    
    # Mountain colors (gray-brown)
    mountain_r = 0.60
    mountain_g = 0.55
    mountain_b = 0.50
    
    land_r = land_r * (1 - mountain_boost) + mountain_r * mountain_boost
    land_g = land_g * (1 - mountain_boost) + mountain_g * mountain_boost
    land_b = land_b * (1 - mountain_boost) + mountain_b * mountain_boost
    
    # ==========================================================================
    # POLAR ICE
    # ==========================================================================
    
    # Arctic
    arctic_mask = LAT >= 75
    arctic_smooth = smooth_mask(arctic_mask, blur_size=5)
    
    # Antarctic (already in land mask)
    antarctic_smooth = smooth_mask(an_mask, blur_size=5)
    
    ice_smooth = np.maximum(arctic_smooth, antarctic_smooth)
    
    # Ice colors
    ice_r = 0.94
    ice_g = 0.96
    ice_b = 0.98
    
    # ==========================================================================
    # OCEAN DEPTH VARIATION
    # ==========================================================================
    
    # Shallow coastal water
    ocean_mask = ~land_mask
    
    # Distance from land (simplified as inverse of land_smooth)
    coastal_factor = 1.0 - np.clip(land_smooth * 3, 0, 1)
    
    # Ocean depth gradient
    ocean_depth = perlin_like((height, width), scale=40)
    
    # Deep ocean: dark blue
    deep_r = 0.04
    deep_g = 0.08
    deep_b = 0.18
    
    # Shallow: lighter blue
    shallow_r = 0.12
    shallow_g = 0.25
    shallow_b = 0.42
    
    # Blend ocean
    ocean_blend = coastal_factor * 0.5 + ocean_depth * 0.3
    ocean_r = deep_r + (shallow_r - deep_r) * ocean_blend
    ocean_g = deep_g + (shallow_g - deep_g) * ocean_blend
    ocean_b = deep_b + (shallow_b - deep_b) * ocean_blend
    
    # ==========================================================================
    # COMPOSITE FINAL IMAGE
    # ==========================================================================
    
    # Start with ocean
    output[:, :, 0] = ocean_r
    output[:, :, 1] = ocean_g
    output[:, :, 2] = ocean_b
    
    # Blend in land
    output[:, :, 0] = output[:, :, 0] * (1 - land_smooth) + land_r * land_smooth
    output[:, :, 1] = output[:, :, 1] * (1 - land_smooth) + land_g * land_smooth
    output[:, :, 2] = output[:, :, 2] * (1 - land_smooth) + land_b * land_smooth
    
    # Add ice caps
    output[:, :, 0] = output[:, :, 0] * (1 - ice_smooth) + ice_r * ice_smooth
    output[:, :, 1] = output[:, :, 1] * (1 - ice_smooth) + ice_g * ice_smooth
    output[:, :, 2] = output[:, :, 2] * (1 - ice_smooth) + ice_b * ice_smooth
    
    # Clamp values
    output = np.clip(output, 0, 1)
    
    return output


def generate_ocean_only(width: int = 720, height: int = 360) -> np.ndarray:
    """
    Generate just ocean bathymetry (for transparent overlay mode).
    """
    lon = np.linspace(-180, 180, width)
    lat = np.linspace(-90, 90, height)
    LON, LAT = np.meshgrid(lon, lat)
    
    output = np.zeros((height, width, 4), dtype=np.float32)
    
    # Ocean depth gradient
    depth = 0.5 + 0.3 * np.sin(LON * np.pi / 60) * np.cos(LAT * np.pi / 45)
    depth = np.clip(depth, 0, 1)
    
    # Deep blue palette
    output[:, :, 0] = 0.02 + 0.08 * depth
    output[:, :, 1] = 0.06 + 0.15 * depth
    output[:, :, 2] = 0.12 + 0.25 * depth
    output[:, :, 3] = 1.0
    
    return output


if __name__ == "__main__":
    # Test generation
    print("Generating Blue Marble texture...")
    texture = generate_blue_marble(720, 360)
    print(f"Generated: {texture.shape}, range [{texture.min():.2f}, {texture.max():.2f}]")
    
    # Save as PNG for inspection
    try:
        from PIL import Image
        img = Image.fromarray((texture * 255).astype(np.uint8))
        img.save("blue_marble_test.png")
        print("Saved: blue_marble_test.png")
    except ImportError:
        print("PIL not available for PNG export")
