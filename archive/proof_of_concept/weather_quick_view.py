#!/usr/bin/env python3
"""
Weather Data Quick Viewer
==========================

Visualize the generated weather data using matplotlib.
Works with the JSON format weather data.

Usage:
    python3 demos/weather_quick_view.py
"""

import json
import sys
from pathlib import Path

print("Loading weather data...")

# Load the JSON data
data_path = Path(__file__).parent.parent / 'results' / 'weather_data.json'

if not data_path.exists():
    print(f"❌ Weather data not found: {data_path}")
    print("   Run: python3 demos/weather_setup_minimal.py")
    sys.exit(1)

with open(data_path) as f:
    data = json.load(f)

print(f"✅ Loaded weather data")
print(f"   Source: {data['metadata']['source']}")
print(f"   Resolution: {data['metadata']['resolution']}")
print(f"   Cyclone: {data['metadata']['cyclone_center']}")

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("\n⚠️  matplotlib not available")
    print("   Install with: python3 -m pip install matplotlib --user")

if HAS_MATPLOTLIB:
    # Convert to numpy arrays for easier plotting
    u_wind = np.array(data['u'])
    v_wind = np.array(data['v'])
    temperature = np.array(data['temperature'])
    
    lats = np.array(data['latitude'])
    lons = np.array(data['longitude'])
    levels = data['level']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('HyperTensor Weather Data Visualization', fontsize=16, fontweight='bold')
    
    # 1. Surface U-Wind (1000 hPa - level 0)
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.contourf(lons, lats, u_wind[0], levels=20, cmap='RdBu_r')
    ax1.set_title(f'Surface U-Wind (1000 hPa)', fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(im1, ax=ax1, label='m/s')
    
    # Mark cyclone
    cyclone = data['metadata']['cyclone_center']
    ax1.plot(cyclone[1] + 360, cyclone[0], 'k*', markersize=15, 
             label=f'Cyclone ({cyclone[0]}°N, {cyclone[1]}°W)')
    ax1.legend()
    
    # 2. Jet Stream Level (250 hPa - level 10)
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.contourf(lons, lats, u_wind[10], levels=20, cmap='RdBu_r')
    ax2.set_title(f'Jet Stream (250 hPa, ~10km)', fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.colorbar(im2, ax=ax2, label='m/s')
    
    # 3. Upper Atmosphere (50 hPa - level 14)
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.contourf(lons, lats, u_wind[-1], levels=20, cmap='RdBu_r')
    ax3.set_title(f'Upper Atmosphere ({levels[-1]} hPa)', fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    plt.colorbar(im3, ax=ax3, label='m/s')
    
    # 4. Surface Temperature
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.contourf(lons, lats, temperature[0], levels=20, cmap='hot')
    ax4.set_title('Surface Temperature (1000 hPa)', fontweight='bold')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    plt.colorbar(im4, ax=ax4, label='K')
    
    # 5. Vertical Wind Profile at cyclone location
    # Find nearest grid point to cyclone
    cyclone_lat_idx = np.argmin(np.abs(lats - cyclone[0]))
    cyclone_lon_idx = np.argmin(np.abs(lons - (cyclone[1] + 360)))
    
    u_profile = [u_wind[k, cyclone_lat_idx, cyclone_lon_idx] for k in range(len(levels))]
    
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(u_profile, levels, 'b-o', linewidth=2, markersize=8)
    ax5.invert_yaxis()  # Higher altitude at top
    ax5.set_yscale('log')
    ax5.set_title(f'Wind Profile at Cyclone\n({cyclone[0]}°N, {cyclone[1]}°W)', 
                  fontweight='bold')
    ax5.set_xlabel('U-Wind (m/s)')
    ax5.set_ylabel('Pressure (hPa)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Wind Speed at surface
    wind_speed = np.sqrt(u_wind[0]**2 + v_wind[0]**2)
    
    ax6 = fig.add_subplot(2, 3, 6)
    im6 = ax6.contourf(lons, lats, wind_speed, levels=20, cmap='YlOrRd')
    ax6.set_title('Surface Wind Speed (1000 hPa)', fontweight='bold')
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')
    plt.colorbar(im6, ax=ax6, label='m/s')
    
    # Quiver plot of wind vectors (downsampled for clarity)
    skip = 3
    ax6.quiver(lons[::skip], lats[::skip], 
               u_wind[0, ::skip, ::skip], v_wind[0, ::skip, ::skip],
               alpha=0.6, scale=400)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent.parent / 'results' / 'weather_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {output_path}")
    
    # Show interactive plot
    print("\n🖼️  Displaying visualization...")
    print("   Close the window to exit.")
    plt.show()

else:
    # Print text-based summary without matplotlib
    print("\n" + "=" * 70)
    print("WEATHER DATA SUMMARY (Text Mode)")
    print("=" * 70)
    
    u_wind = data['u']
    
    # Surface level statistics
    surface_u = u_wind[0]
    flat_surface = [val for row in surface_u for val in row]
    
    u_min = min(flat_surface)
    u_max = max(flat_surface)
    u_mean = sum(flat_surface) / len(flat_surface)
    
    print(f"\nSurface U-Wind (1000 hPa):")
    print(f"  Min:  {u_min:6.1f} m/s")
    print(f"  Max:  {u_max:6.1f} m/s")
    print(f"  Mean: {u_mean:6.1f} m/s")
    
    # Jet stream level
    jet_u = u_wind[10]
    flat_jet = [val for row in jet_u for val in row]
    
    jet_min = min(flat_jet)
    jet_max = max(flat_jet)
    jet_mean = sum(flat_jet) / len(flat_jet)
    
    print(f"\nJet Stream (250 hPa, ~10km altitude):")
    print(f"  Min:  {jet_min:6.1f} m/s")
    print(f"  Max:  {jet_max:6.1f} m/s")
    print(f"  Mean: {jet_mean:6.1f} m/s")
    
    print(f"\n📍 Cyclone Location: {data['metadata']['cyclone_center']}")
    
    print("\n" + "=" * 70)
    print("To see graphical visualization:")
    print("  python3 -m pip install matplotlib numpy --user")
    print("  python3 demos/weather_quick_view.py")
    print("=" * 70)

print("\n✨ Done!\n")
