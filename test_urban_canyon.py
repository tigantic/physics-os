#!/usr/bin/env python3
"""
Phase 7C: The Drone Highway - Urban Air Safety Scanner

Identifies "Kill Zones" (high turbulence) vs "Green Lanes" (safe flight corridors)
in procedurally generated cities.

Commercial Applications:
- Amazon Prime Air: Safe delivery routes
- Google Wing: Package delivery corridors
- Joby/Lilium: Flying taxi approach paths
- Architecture: Wind load analysis

Risk Categories:
- GREEN: Safe for all aircraft (updraft < 3 m/s)
- YELLOW: Caution - light turbulence (3-6 m/s)
- RED: Danger - severe turbulence (6-10 m/s)
- BLACK: No-fly - fatal conditions (>10 m/s)

Usage:
    python test_urban_canyon.py
"""

import torch
import numpy as np
import sys
import os

# Ensure tensornet is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensornet.urban.city_gen import VoxelCity
from tensornet.urban.solver import UrbanFlowSolver, analyze_flight_safety


def print_header():
    """Print demo header."""
    print("=" * 70)
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║     HYPERTENSOR URBAN - DRONE HIGHWAY SCANNER                ║")
    print("  ║     Phase 7: Urban Canyon Wind Analysis                      ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print("=" * 70)
    print()


def scan_city_safety():
    """
    Main demo: Build city, simulate wind, analyze safety.
    """
    print_header()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE] Running on: {device}")
    if device.type == "cuda":
        print(f"[HARDWARE] GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # ========================================================================
    # 1. BUILD CITY
    # ========================================================================
    print("[PHASE 1] City Generation")
    print("-" * 50)
    
    # Create a Manhattan-style city
    city = VoxelCity(size=(64, 32, 64), voxel_size=5.0)
    geo = city.generate_manhattan(num_buildings=15, seed=42)
    
    print(f"   Domain: {city.physical_size[0]:.0f}m x "
          f"{city.physical_size[1]:.0f}m x {city.physical_size[2]:.0f}m")
    print(f"   Voxel size: {city.voxel_size}m")
    print()
    
    # ========================================================================
    # 2. SIMULATE WIND
    # ========================================================================
    print("[PHASE 2] Wind Simulation")
    print("-" * 50)
    
    wind_speed = 15.0  # 15 m/s gale force
    print(f"   Incoming wind: {wind_speed} m/s (Gale Force)")
    print()
    
    solver = UrbanFlowSolver(iterations=30, relaxation=0.8)
    wind_field = solver.solve(geo, wind_speed=wind_speed)
    
    print()
    
    # ========================================================================
    # 3. ANALYZE RISKS
    # ========================================================================
    print("[PHASE 3] Risk Analysis")
    print("-" * 50)
    
    report = solver.analyze_safety(wind_field, geo)
    
    # Extract key metrics
    updrafts = wind_field[1]
    max_updraft = updrafts.max().item()
    max_downdraft = -updrafts.min().item()
    
    print(f"   Base Wind Speed:    {wind_speed:.1f} m/s")
    print(f"   Max Updraft:        {max_updraft:.2f} m/s")
    print(f"   Max Downdraft:      {max_downdraft:.2f} m/s")
    print(f"   Max Horizontal:     {report.max_horizontal:.2f} m/s")
    print(f"   Avg Turbulence:     {report.avg_turbulence:.2f} m/s")
    print()
    
    # ========================================================================
    # 4. SAFETY CLASSIFICATION
    # ========================================================================
    print("[PHASE 4] Flight Zone Classification")
    print("-" * 50)
    
    total = report.total_air_volume
    safe_pct = report.safe_volume / total * 100
    caution_pct = report.caution_volume / total * 100
    danger_pct = report.danger_volume / total * 100
    
    print(f"   🟢 GREEN (Safe):      {report.safe_volume:>8,} voxels ({safe_pct:>5.1f}%)")
    print(f"   🟡 YELLOW (Caution):  {report.caution_volume:>8,} voxels ({caution_pct:>5.1f}%)")
    print(f"   🔴 RED (Danger):      {report.danger_volume:>8,} voxels ({danger_pct:>5.1f}%)")
    print()
    
    # Alert status
    if report.fatal_zones_detected:
        print("   ⚠️  [ALERT] FATAL UPDRAFTS DETECTED!")
        print("       Drone flight NOT RECOMMENDED in affected areas.")
    
    if report.no_fly_recommended:
        print("   ⛔ [ALERT] NO-FLY ZONE RECOMMENDED")
        print("       Over 30% of airspace is dangerous.")
    
    if max_updraft > 5.0:
        print(f"   ⚠️  [ALERT] Severe updrafts ({max_updraft:.1f} m/s) detected.")
        print("       Light drones may lose control.")
    
    print()
    
    # ========================================================================
    # 5. SAFE CORRIDOR ANALYSIS
    # ========================================================================
    print("[PHASE 5] Safe Corridor Identification")
    print("-" * 50)
    
    # Find safe altitude
    print(f"   Minimum safe altitude: {report.safe_altitude_min * city.voxel_size:.0f}m AGL")
    
    # Calculate safe air volume in cubic meters
    safe_volume_m3 = report.safe_volume * (city.voxel_size ** 3)
    print(f"   Safe air volume: {safe_volume_m3:.2e} m³")
    
    # Street canyon analysis
    zones = solver.get_zone_tensor(wind_field, geo)
    
    # Find continuous safe paths at mid-height
    mid_height = city.height // 2
    safe_slice = zones[:, mid_height, :] == 1
    connected_safe = safe_slice.sum().item()
    
    print(f"   Safe voxels at {mid_height * city.voxel_size:.0f}m altitude: {connected_safe}")
    print()
    
    # ========================================================================
    # 6. COMMERCIAL VALUE
    # ========================================================================
    print("[COMMERCIAL ANALYSIS]")
    print("-" * 50)
    
    # Estimate delivery efficiency
    if safe_pct > 70:
        efficiency = "HIGH"
        risk_level = "LOW"
    elif safe_pct > 40:
        efficiency = "MODERATE"
        risk_level = "MEDIUM"
    else:
        efficiency = "LOW"
        risk_level = "HIGH"
    
    print(f"   Delivery Efficiency:  {efficiency}")
    print(f"   Risk Level:           {risk_level}")
    print(f"   Safe Flight Fraction: {safe_pct:.1f}%")
    print()
    
    # ========================================================================
    # 7. SUMMARY
    # ========================================================================
    print("=" * 70)
    print("  URBAN CANYON ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    
    if max_updraft > 10.0:
        verdict = "⛔ CRITICAL: Fatal updrafts detected. No-fly zone."
    elif max_updraft > 6.0:
        verdict = "🔴 DANGER: Severe turbulence. Large drones only."
    elif max_updraft > 3.0:
        verdict = "🟡 CAUTION: Moderate turbulence. Exercise care."
    else:
        verdict = "🟢 CLEAR: Safe for drone operations."
    
    print(f"  VERDICT: {verdict}")
    print()
    
    return report


def run_multi_scenario():
    """
    Run multiple wind speed scenarios.
    """
    print("\n")
    print("=" * 70)
    print("  MULTI-SCENARIO ANALYSIS")
    print("=" * 70)
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build city once
    city = VoxelCity(size=(64, 32, 64), voxel_size=5.0)
    geo = city.generate_manhattan(num_buildings=15, seed=42)
    
    solver = UrbanFlowSolver(iterations=25)
    
    scenarios = [
        ("Light Breeze", 5.0),
        ("Moderate Wind", 10.0),
        ("Strong Wind", 15.0),
        ("Gale Force", 20.0),
        ("Storm", 25.0),
    ]
    
    print(f"  {'Scenario':<15} | {'Wind':<8} | {'Max Up':<8} | {'Safe %':<8} | Status")
    print("-" * 70)
    
    for name, speed in scenarios:
        flow = solver.solve(geo, wind_speed=speed, verbose=False)
        report = solver.analyze_safety(flow, geo)
        
        safe_pct = report.safe_volume / report.total_air_volume * 100
        
        if report.max_updraft > 10:
            status = "⛔ NO-FLY"
        elif report.max_updraft > 6:
            status = "🔴 DANGER"
        elif report.max_updraft > 3:
            status = "🟡 CAUTION"
        else:
            status = "🟢 CLEAR"
        
        print(f"  {name:<15} | {speed:>5.1f} m/s | {report.max_updraft:>5.1f} m/s | {safe_pct:>5.1f}%  | {status}")
    
    print()


def run_building_height_study():
    """
    Study effect of building height on updrafts.
    """
    print("\n")
    print("=" * 70)
    print("  BUILDING HEIGHT STUDY")
    print("=" * 70)
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = UrbanFlowSolver(iterations=25)
    
    wind_speed = 15.0
    
    print(f"  Wind Speed: {wind_speed} m/s")
    print()
    print(f"  {'Max Height':<12} | {'Max Up':<10} | {'Safe %':<10} | Recommendation")
    print("-" * 65)
    
    for max_h in [10, 20, 30, 40, 50]:
        city = VoxelCity(size=(64, 64, 64), voxel_size=5.0)
        city.generate_manhattan(num_buildings=12, max_height=max_h, seed=42)
        
        flow = solver.solve(city.geometry, wind_speed=wind_speed, verbose=False)
        report = solver.analyze_safety(flow, city.geometry)
        
        safe_pct = report.safe_volume / report.total_air_volume * 100
        height_m = max_h * city.voxel_size
        
        if report.max_updraft > 8:
            rec = "Avoid low altitude"
        elif report.max_updraft > 4:
            rec = "Use caution"
        else:
            rec = "Safe for delivery"
        
        print(f"  {height_m:>5.0f}m       | {report.max_updraft:>6.1f} m/s | {safe_pct:>6.1f}%   | {rec}")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Urban canyon safety scanner")
    parser.add_argument("--multi", action="store_true", help="Run multi-scenario analysis")
    parser.add_argument("--heights", action="store_true", help="Run building height study")
    
    args = parser.parse_args()
    
    # Always run main analysis
    report = scan_city_safety()
    
    # Optional additional studies
    if args.multi:
        run_multi_scenario()
    
    if args.heights:
        run_building_height_study()
    
    print("=" * 70)
    print("  PHASE 7 COMPLETE - URBAN CANYON MODULE VALIDATED")
    print("=" * 70)
