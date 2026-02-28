#!/usr/bin/env python3
"""
Weather System Setup - Minimal Version
=======================================

Generates synthetic weather data using only Python standard library.
Creates JSON format data for compatibility.

Usage:
    python3 demos/weather_setup_minimal.py
"""

import json
import math
from pathlib import Path
from datetime import datetime

print("""
╔══════════════════════════════════════════════════════════════════════╗
║            ONTIC_ENGINE WEATHER SYSTEM SETUP (MINIMAL)                 ║
║                                                                      ║
║  Generating synthetic atmospheric data - no dependencies required     ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def create_synthetic_weather_data_minimal(n_levels=15, n_lat=45, n_lon=90):
    """
    Create synthetic weather data using only Python standard library.
    
    Args:
        n_levels: Number of pressure levels (default: 15)
        n_lat: Number of latitude points (default: 45, 4° resolution)
        n_lon: Number of longitude points (default: 90, 4° resolution)
        
    Returns:
        Dictionary with weather fields
    """
    print(f"\n🌀 Generating synthetic atmospheric data...")
    print(f"   Resolution: {n_levels} levels × {n_lat} lat × {n_lon} lon")
    
    # Pressure levels (hPa)
    levels = [1000, 950, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50][:n_levels]
    
    # Coordinates
    lats = [-90 + i * 180 / (n_lat - 1) for i in range(n_lat)]
    lons = [i * 360 / n_lon for i in range(n_lon)]
    
    # Initialize 3D arrays as nested lists
    u_wind = [[[0.0 for _ in range(n_lon)] for _ in range(n_lat)] for _ in range(n_levels)]
    v_wind = [[[0.0 for _ in range(n_lon)] for _ in range(n_lat)] for _ in range(n_levels)]
    temperature = [[[0.0 for _ in range(n_lon)] for _ in range(n_lat)] for _ in range(n_levels)]
    
    print(f"   Calculating wind fields...")
    
    # Synthetic cyclone location
    cyclone_lat, cyclone_lon = 35, 310  # North Atlantic
    
    # Generate data
    u_min, u_max = float('inf'), float('-inf')
    
    for k in range(n_levels):
        level = levels[k]
        # Altitude factor (stronger winds aloft, peak at 250 hPa)
        alt_factor = math.exp(-((math.log(level / 250)) ** 2) / 2)
        
        for j in range(n_lat):
            lat = lats[j]
            # Jet stream factor (peaks at ±35° latitude)
            lat_factor = (math.exp(-((lat - 35) / 15) ** 2) + 
                         math.exp(-((lat + 35) / 15) ** 2))
            
            for i in range(n_lon):
                lon = lons[i]
                
                # U-Wind (East-West)
                wave = math.sin(lon * math.pi / 30) * 0.3
                u = 30 * alt_factor * lat_factor * (1 + wave)
                
                # Trade winds (surface, easterly)
                if abs(lat) < 30 and level > 850:
                    u -= 10
                
                # Add cyclone circulation
                dist = math.sqrt((lat - cyclone_lat)**2 + 
                               ((lon - cyclone_lon) * math.cos(math.radians(lat)))**2)
                if dist < 15:
                    angle = math.atan2(lat - cyclone_lat, lon - cyclone_lon)
                    cyclone_strength = math.exp(-dist**2 / 50) * (level / 1000)
                    u += 30 * math.sin(angle) * cyclone_strength
                
                u_wind[k][j][i] = u
                u_min = min(u_min, u)
                u_max = max(u_max, u)
                
                # V-Wind (North-South, generally weaker)
                v = 5 * math.sin(lon * math.pi / 60) * math.cos(lat * math.pi / 90)
                if dist < 15:
                    v -= 30 * math.cos(angle) * cyclone_strength
                v_wind[k][j][i] = v
                
                # Temperature (K)
                altitude_km = 44.3 * (1 - (level / 1013.25) ** 0.19)
                T_surface = 288 + 20 * math.cos(lat * math.pi / 180)
                T = max(T_surface - 6.5 * altitude_km, 200)
                temperature[k][j][i] = T
    
    print(f"   ✓ U-wind range: [{u_min:.1f}, {u_max:.1f}] m/s")
    print(f"   ✓ Synthetic cyclone at: {cyclone_lat}°N, {cyclone_lon - 360}°W")
    
    return {
        'u': u_wind,
        'v': v_wind,
        'temperature': temperature,
        'latitude': lats,
        'longitude': lons,
        'level': levels,
        'metadata': {
            'source': 'synthetic',
            'cyclone_center': [cyclone_lat, cyclone_lon - 360],
            'timestamp': datetime.utcnow().isoformat(),
            'resolution': f'{n_levels}x{n_lat}x{n_lon}',
            'description': 'Synthetic atmospheric data for The Ontic Engine visualization'
        }
    }


def save_weather_json(data, output_path):
    """Save weather data as JSON."""
    print(f"\n💾 Saving weather data to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"   ✅ Saved: {output_path}")
    print(f"      Size: {size_mb:.2f} MB")


def create_readme(results_dir):
    """Create a README explaining the weather data."""
    readme_path = results_dir / 'WEATHER_DATA_README.md'
    
    content = """# Ontic Weather Data

This directory contains synthetic atmospheric data generated for the Ontic Engine weather visualization system.

## Files

- `weather_data.json` - Atmospheric data in JSON format
  - 15 pressure levels (1000 hPa surface to 50 hPa stratosphere)
  - 45 latitude points (4° resolution)
  - 90 longitude points (4° resolution)
  - Variables: U-wind, V-wind, Temperature

## Data Structure

```json
{
  "u": [[[...]]], // U-wind (m/s) - shape: [levels, lat, lon]
  "v": [[[...]]], // V-wind (m/s) - shape: [levels, lat, lon]
  "temperature": [[[...]]], // Temperature (K) - shape: [levels, lat, lon]
  "latitude": [...], // Latitude values (-90 to 90)
  "longitude": [...], // Longitude values (0 to 360)
  "level": [...], // Pressure levels (hPa)
  "metadata": {...} // Source info, timestamp, etc.
}
```

## Features

- **Jet Streams**: Strong westerly winds at ~35°N/S around 250 hPa (10km altitude)
- **Trade Winds**: Easterly winds in tropics near surface
- **Synthetic Cyclone**: Low pressure system at 35°N, 50°W (North Atlantic)
- **Realistic Temperature**: Lapse rate of 6.5K/km, warmer at equator

## Usage

### Python
```python
import json

with open('results/weather_data.json') as f:
    data = json.load(f)

u_wind = data['u']  # [level][lat][lon]
surface_winds = u_wind[0]  # 1000 hPa (surface)
jet_stream = u_wind[10]    # 250 hPa (~10km altitude)
```

### Visualization

Run the weather viewer (requires PySide6 and vispy):
```bash
pip install PySide6 vispy numpy torch
python demos/weather_viewer.py
```

## About

Generated by The Ontic Engine's atmospheric data synthesis pipeline.
Part of Project The Ontic Engine - Quantum-Inspired Tensor Networks for CFD.
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"   📄 Created: {readme_path}")


def main():
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / 'weather_data.json'
    
    print("\n" + "=" * 70)
    print("STEP 1: GENERATE ATMOSPHERIC DATA")
    print("=" * 70)
    
    # Generate synthetic weather data (smaller for speed and compatibility)
    weather_data = create_synthetic_weather_data_minimal(n_levels=15, n_lat=45, n_lon=90)
    
    print("\n" + "=" * 70)
    print("STEP 2: SAVE DATA")
    print("=" * 70)
    
    save_weather_json(weather_data, output_path)
    create_readme(results_dir)
    
    print("\n" + "=" * 70)
    print("✅ WEATHER SYSTEM READY!")
    print("=" * 70)
    
    cyclone = weather_data['metadata']['cyclone_center']
    print(f"""
📊 Data Summary:
   Source:       {weather_data['metadata']['source']}
   Resolution:   {weather_data['metadata']['resolution']}
   Variables:    U-wind, V-wind, Temperature
   Cyclone:      {cyclone[0]}°N, {cyclone[1]}°W
   Total Points: {len(weather_data['level']) * len(weather_data['latitude']) * len(weather_data['longitude']):,}

📁 Output:
   {output_path}
   {results_dir / 'WEATHER_DATA_README.md'}

🌍 What you can explore:
   • Jet stream patterns at 250 hPa (level index 10)
   • Surface cyclone structure at 1000 hPa (level index 0)
   • Temperature gradients from equator to poles
   • Wind patterns across pressure levels

🚀 Next Steps:

   To convert to torch format (for full weather_viewer):
   1. Install: python3 -m pip install torch numpy --user
   2. Run: python3 demos/weather_setup.py
   
   Or integrate the JSON data into your own analysis:
   
   import json
   with open('{output_path}') as f:
       data = json.load(f)
   
   # Access wind at any level/location
   surface_u_wind = data['u'][0]  # Surface level
   jet_stream = data['u'][10]     # 250 hPa
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
        exit(1)
