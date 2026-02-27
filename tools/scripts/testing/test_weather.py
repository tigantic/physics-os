#!/usr/bin/env python3
"""Quick test that weather data is valid"""
import json

data = json.load(open("results/weather_data.json"))
print(f"✅ Weather data loaded successfully!")
print(
    f"   Dimensions: {len(data['u'])} levels × {len(data['u'][0])} lat × {len(data['u'][0][0])} lon"
)
print(f"   Cyclone at: {data['metadata']['cyclone_center']}")
print(
    f"   U-wind range: [{min(min(row) for row in data['u'][0]):.1f}, {max(max(row) for row in data['u'][0]):.1f}] m/s"
)
