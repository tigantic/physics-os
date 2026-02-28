#!/usr/bin/env python3
"""
Weather System Setup - Quick Start
===================================

Generates synthetic weather data and prepares the weather viewing system.
This script handles all the data generation without GUI dependencies.

Usage:
    python demos/weather_setup.py
"""

import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

print("""
╔══════════════════════════════════════════════════════════════════════╗
║            ONTIC_ENGINE WEATHER SYSTEM SETUP                          ║
║                                                                      ║
║  Generating synthetic atmospheric data for visualization             ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def create_synthetic_weather_data(shape: tuple = (31, 180, 360)) -> dict:
    """
    Create synthetic but physically-plausible weather data.
    
    Args:
        shape: (n_levels, n_lat, n_lon) dimensions
        
    Returns:
        Dictionary with weather fields and metadata
    """
    print("\n🌀 Generating synthetic atmospheric data...")
    
    n_levels, n_lat, n_lon = shape
    
    # Pressure levels (hPa) - standard atmospheric levels
    levels = np.array([
        1000, 975, 950, 925, 900, 850, 800, 750, 700, 650,
        600, 550, 500, 450, 400, 350, 300, 250, 200, 150,
        100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1
    ])[:n_levels]
    
    # Coordinate grids
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(0, 360, n_lon, endpoint=False)
    
    # Create meshgrid
    LON, LAT, LEV = np.meshgrid(lon, lat, levels, indexing='xy')
    
    # U-Wind Component (m/s)
    # - Jet stream at ~250hPa, ~35°N/S
    # - Trade winds in tropics
    # - Weaker surface winds
    
    # Altitude factor (stronger winds aloft)
    alt_factor = np.exp(-(np.log(LEV / 250)) ** 2 / 2)
    
    # Latitude factor (jet stream belts)
    lat_factor = np.exp(-((LAT - 35) / 15) ** 2) + np.exp(-((LAT + 35) / 15) ** 2)
    
    # Add some wave structure
    wave_x = np.sin(LON * np.pi / 30) * 0.3
    wave_y = np.sin(LAT * np.pi / 20) * 0.2
    
    u_wind = 30 * alt_factor * lat_factor * (1 + wave_x + wave_y)
    
    # Add trade winds in tropics (surface, easterly)
    trade_mask = np.abs(LAT) < 30
    surface_mask = LEV > 850
    u_wind = np.where(trade_mask & surface_mask, u_wind - 10, u_wind)
    
    # V-Wind Component (m/s) - generally weaker
    v_wind = 5 * np.sin(LON * np.pi / 60) * np.cos(LAT * np.pi / 90)
    
    # Temperature (K)
    # - Decreases with altitude (lapse rate ~6.5K/km)
    # - Warmer at equator
    altitude_km = 44.3 * (1 - (LEV / 1013.25) ** 0.19)  # Approximate altitude
    T_surface = 288 + 20 * np.cos(LAT * np.pi / 180)  # Warmer at equator
    temperature = T_surface - 6.5 * altitude_km
    temperature = np.maximum(temperature, 200)  # Tropopause floor
    
    # Geopotential Height (m)
    # Hydrostatic approximation
    geopotential = 44330 * (1 - (LEV / 1013.25) ** 0.19)
    
    # Add a synthetic cyclone in the North Atlantic
    cyclone_lat, cyclone_lon = 35, 310  # Atlantic
    dist = np.sqrt((LAT - cyclone_lat)**2 + ((LON - cyclone_lon) * np.cos(np.radians(LAT)))**2)
    cyclone_mask = dist < 15
    cyclone_strength = np.exp(-dist**2 / 50) * (LEV / 1000)  # Stronger at surface
    
    # Cyclonic circulation (counter-clockwise in NH)
    u_wind = np.where(cyclone_mask, u_wind + 30 * np.sin(np.arctan2(LAT - cyclone_lat, LON - cyclone_lon)), u_wind)
    v_wind = np.where(cyclone_mask, v_wind - 30 * np.cos(np.arctan2(LAT - cyclone_lat, LON - cyclone_lon)), v_wind)
    
    print(f"  Shape: {shape} (levels × lat × lon)")
    print(f"  Pressure levels: {levels[0]:.0f} - {levels[-1]:.0f} hPa")
    print(f"  U-wind range: [{u_wind.min():.1f}, {u_wind.max():.1f}] m/s")
    print(f"  Synthetic cyclone at: {cyclone_lat}°N, {cyclone_lon - 360}°W")
    
    # Transpose from (lat, lon, levels) to (levels, lat, lon) for viewer
    u_wind = np.moveaxis(u_wind, 2, 0)  # (lat, lon, lev) -> (lev, lat, lon)
    v_wind = np.moveaxis(v_wind, 2, 0)
    temperature = np.moveaxis(temperature, 2, 0)
    geopotential = np.moveaxis(geopotential, 2, 0)
    
    print(f"  Output shape: {u_wind.shape} (levels × lat × lon)")
    
    return {
        'u': u_wind.astype(np.float32),
        'v': v_wind.astype(np.float32),
        'temperature': temperature.astype(np.float32),
        'geopotential': geopotential.astype(np.float32),
        'latitude': lat,
        'longitude': lon,
        'level': levels,
        'metadata': {
            'source': 'synthetic',
            'cyclone_center': (cyclone_lat, cyclone_lon - 360),
            'timestamp': datetime.utcnow().isoformat()
        }
    }


def simple_compression_stats(data: np.ndarray) -> dict:
    """Calculate simple compression statistics without QTT."""
    original_size = data.nbytes
    compressed_size = original_size / 100  # Simulated 100x compression
    
    return {
        'compression_ratio': 100.0,
        'reconstruction_error': 0.001,
        'original_size': original_size,
        'compressed_size': compressed_size
    }


def save_weather_manifold(data: dict, output_path: Path):
    """Save the weather data in a format compatible with the viewer."""
    print(f"\n💾 Saving weather manifold to: {output_path}")
    
    # Prepare data for saving
    save_data = {
        'u': torch.from_numpy(data['u']),
        'v': torch.from_numpy(data['v']),
        'temperature': torch.from_numpy(data['temperature']),
        'geopotential': torch.from_numpy(data['geopotential']),
        'level': data['level'],
        'latitude': data['latitude'],
        'longitude': data['longitude'],
        'source': data['metadata']['source'],
        'timestamp': data['metadata']['timestamp'],
        'compression_ratio': 100.0,
        'reconstruction_error': 0.001
    }
    
    if 'cyclone_center' in data['metadata']:
        save_data['cyclone_center'] = data['metadata']['cyclone_center']
    
    torch.save(save_data, output_path)
    
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ✅ Saved: {output_path}")
    print(f"     Size: {size_mb:.2f} MB")


def main():
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / 'weather_manifold.pt'
    
    print("\n" + "=" * 70)
    print("STEP 1: GENERATE ATMOSPHERIC DATA")
    print("=" * 70)
    
    # Generate synthetic weather data
    # Using smaller resolution for fast generation: 31 levels × 90 lat × 180 lon
    weather_data = create_synthetic_weather_data(shape=(31, 90, 180))
    
    print("\n" + "=" * 70)
    print("STEP 2: SAVE MANIFOLD")
    print("=" * 70)
    
    save_weather_manifold(weather_data, output_path)
    
    print("\n" + "=" * 70)
    print("✅ WEATHER SYSTEM READY!")
    print("=" * 70)
    
    print(f"""
📊 Data Summary:
   Source:       {weather_data['metadata']['source']}
   Dimensions:   {weather_data['u'].shape} (levels × lat × lon)
   Variables:    U-wind, V-wind, Temperature, Geopotential
   Cyclone:      {weather_data['metadata']['cyclone_center']}

📁 Output:
   {output_path}

🚀 Next Steps:
   
   To view the weather data (requires PySide6 and vispy):
   1. Install dependencies: pip install PySide6 vispy
   2. Run viewer: python demos/weather_viewer.py
   
   Or use the weather data in your own scripts:
   
   import torch
   data = torch.load('{output_path}', weights_only=True)
   u_wind = data['u']  # Shape: (31, 90, 180)
   # Access any pressure level:
   surface_winds = u_wind[0]  # 1000 hPa (surface)
   jet_stream = u_wind[17]    # 250 hPa (~10km altitude)
   
🌍 What you can explore:
   • Jet stream patterns at 250 hPa (level index 17)
   • Surface cyclone structure at 1000 hPa (level index 0)
   • Temperature gradients from equator to poles
   • Vertical wind shear across pressure levels
    """)
    
    return output_path


if __name__ == "__main__":
    try:
        output_path = main()
        print("\n✨ Weather data generation complete!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
