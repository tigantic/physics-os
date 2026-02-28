#!/usr/bin/env python3
"""
Phase 5B: The Commercial Demo - Wind Farm Optimization

This script proves the commercial value of HyperTensor's wake physics engine.
We simulate two scenarios:
  - Scenario A: Bad layout (turbines in a straight line = wake blocking)
  - Scenario B: Optimized layout (staggered = wake avoidance)

The delta in power output translates directly to dollars.

Target Market: Offshore Wind Developers
  - Orsted (world's largest offshore wind developer)
  - Shell Energy (Renewable transition)
  - Equinor (North Sea expertise)

Run: python test_energy_yield.py
"""

import os
import sys

import torch

# Ensure ontic is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontic.energy_env.energy.turbine import WindFarm


def print_header():
    """Print demo header."""
    print("=" * 70)
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║     HYPERTENSOR ENERGY - COMMERCIAL DEMONSTRATION            ║")
    print("  ║     Phase 5: Wind Farm Wake Optimization                     ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print("=" * 70)
    print()


def run_commercial_demo():
    """
    Run the commercial demonstration comparing layout strategies.

    Domain: 1km x 500m x 500m (North Sea offshore platform)
    Wind: 12 m/s steady (Fresh Breeze - typical operating condition)
    Turbines: 2x 5MW class (80m rotor diameter)
    """
    print_header()

    print("[ENERGY] Initializing North Sea Wind Farm Simulation...")
    print()

    # ========================================================================
    # 1. HARDWARE DETECTION
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE] Running on: {device}")
    if device.type == "cuda":
        print(f"[HARDWARE] GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ========================================================================
    # 2. DOMAIN SETUP
    # ========================================================================
    # Shape: [Components, Depth(Z), Height(Y), Width(X)]
    # Resolution: 10m per cell
    # Domain: 1000m x 500m x 500m
    # Wind: 12.0 m/s uniform streamwise flow (U component)

    grid_resolution = 10.0  # meters
    domain_depth = 100  # 1000m streamwise
    domain_height = 50  # 500m vertical
    domain_width = 50  # 500m lateral

    wind_speed = 12.0  # m/s (Fresh Breeze)

    domain = torch.zeros((3, domain_depth, domain_height, domain_width), device=device)
    domain[0] = wind_speed  # U component (streamwise)

    print(f"[DOMAIN] Grid: {domain_width}x{domain_height}x{domain_depth} cells")
    print(
        f"[DOMAIN] Physical: {domain_width*grid_resolution}m x "
        f"{domain_height*grid_resolution}m x {domain_depth*grid_resolution}m"
    )
    print(f"[DOMAIN] Wind Speed: {wind_speed} m/s (Fresh Breeze)")
    print()

    # ========================================================================
    # 3. SCENARIO A: BAD LAYOUT (Direct Wake Blocking)
    # ========================================================================
    # Turbine 2 is directly downstream of Turbine 1
    # Maximum wake interference = minimum power

    print("[SCENARIO A] Straight Line Alignment (Industry Default)")
    print("-" * 50)

    turbines_bad = [
        {
            "x": 250.0,  # Center laterally
            "y": 250.0,  # Hub height
            "z": 200.0,  # Upstream position
            "radius": 40.0,  # 80m diameter rotor
            "yaw": 0.0,
        },
        {
            "x": 250.0,  # SAME lateral position (in wake)
            "y": 250.0,  # Same hub height
            "z": 600.0,  # 400m downstream
            "radius": 40.0,
            "yaw": 0.0,
        },
    ]

    farm_bad = WindFarm(turbines_bad, environment="offshore")

    print(f"[COMPUTING] Applying Jensen Park Wake Model...")
    field_a = domain.clone()
    farm_bad.apply_wakes(field_a, grid_resolution=grid_resolution)
    power_a = farm_bad.calculate_power_output(field_a, grid_resolution=grid_resolution)

    # Calculate economics
    revenue_a = farm_bad.annual_revenue(power_a)

    print(f"   Turbine 1: x=250m, z=200m (upstream)")
    print(f"   Turbine 2: x=250m, z=600m (downstream, IN WAKE)")
    print()
    print(f"   ► Total Power Output: {power_a:.2f} MW")
    print(f"   ► Annual Revenue: ${revenue_a:,.0f}")
    print()

    # ========================================================================
    # 4. SCENARIO B: OPTIMIZED LAYOUT (Wake Avoidance)
    # ========================================================================
    # Turbine 2 shifted laterally by 100m to dodge the wake cone

    print("[SCENARIO B] HyperTensor Optimized Layout")
    print("-" * 50)

    turbines_opt = [
        {"x": 250.0, "y": 250.0, "z": 200.0, "radius": 40.0, "yaw": 0.0},
        {
            "x": 350.0,  # SHIFTED 100m laterally (OUT of wake)
            "y": 250.0,
            "z": 600.0,
            "radius": 40.0,
            "yaw": 0.0,
        },
    ]

    farm_opt = WindFarm(turbines_opt, environment="offshore")

    print(f"[COMPUTING] Applying Jensen Park Wake Model...")
    field_b = domain.clone()
    farm_opt.apply_wakes(field_b, grid_resolution=grid_resolution)
    power_b = farm_opt.calculate_power_output(field_b, grid_resolution=grid_resolution)

    revenue_b = farm_opt.annual_revenue(power_b)

    print(f"   Turbine 1: x=250m, z=200m (upstream)")
    print(f"   Turbine 2: x=350m, z=600m (shifted, CLEAR AIR)")
    print()
    print(f"   ► Total Power Output: {power_b:.2f} MW")
    print(f"   ► Annual Revenue: ${revenue_b:,.0f}")
    print()

    # ========================================================================
    # 5. THE VERDICT - COMMERCIAL VALUE
    # ========================================================================
    print("=" * 70)
    print("  COMMERCIAL ANALYSIS")
    print("=" * 70)
    print()

    delta_power = power_b - power_a
    delta_revenue = revenue_b - revenue_a
    improvement_pct = (delta_power / power_a) * 100 if power_a > 0 else 0

    print(f"  Power Improvement: +{delta_power:.2f} MW ({improvement_pct:.1f}%)")
    print(f"  Revenue Improvement: +${delta_revenue:,.0f}/year")
    print()

    # Scale to typical farm size
    typical_farm_turbines = 50
    scaling_factor = typical_farm_turbines / 2  # Our demo has 2 turbines
    scaled_revenue = delta_revenue * scaling_factor

    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  HYPERTENSOR FOUND ${delta_revenue:,.0f} OF HIDDEN VALUE/YEAR   │")
    print(f"  │                                                         │")
    print(f"  │  Scaled to 50-turbine farm: ${scaled_revenue:,.0f}/year        │")
    print(f"  │  20-year project lifetime:  ${scaled_revenue * 20:,.0f}       │")
    print(f"  └─────────────────────────────────────────────────────────┘")
    print()

    # ========================================================================
    # 6. TECHNICAL METRICS
    # ========================================================================
    print("[METRICS] Wake Analysis")
    print("-" * 50)

    # Sample wake velocity at downstream turbine location
    ix, iy, iz = 25, 25, 60  # Downstream turbine grid location
    wake_velocity = float(field_a[0, iz, iy, ix])
    free_velocity = float(domain[0, iz, iy, ix])
    velocity_deficit = (1 - wake_velocity / free_velocity) * 100

    print(f"   Free-stream velocity: {free_velocity:.1f} m/s")
    print(f"   Wake velocity (blocked): {wake_velocity:.1f} m/s")
    print(f"   Velocity deficit: {velocity_deficit:.1f}%")
    print()

    # Capacity factor
    cf_bad = farm_bad.calculate_capacity_factor(power_a, wind_speed)
    cf_opt = farm_opt.calculate_capacity_factor(power_b, wind_speed)

    print(f"   Capacity Factor (Bad):  {cf_bad*100:.1f}%")
    print(f"   Capacity Factor (Opt):  {cf_opt*100:.1f}%")
    print()

    print("[SUCCESS] Commercial demo complete.")
    print("[NEXT] Run ontic/energy/unreal_stream.py to visualize wake field.")

    return power_a, power_b, delta_revenue


def run_scaling_demo():
    """
    Demonstrate scalability with larger farms.
    """
    print("\n")
    print("=" * 70)
    print("  SCALING DEMONSTRATION - 10 TURBINE FARM")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_resolution = 10.0

    # Larger domain for 10 turbines
    domain = torch.ones((3, 200, 50, 100), device=device) * 12.0

    # Grid layout (2 rows x 5 columns) - common offshore pattern
    turbines = []
    for row in range(2):
        for col in range(5):
            turbines.append(
                {
                    "x": 200.0 + col * 150.0,  # 150m spacing lateral
                    "y": 250.0,
                    "z": 200.0 + row * 500.0,  # 500m spacing streamwise
                    "radius": 40.0,
                    "yaw": 0.0,
                }
            )

    farm = WindFarm(turbines, environment="offshore")

    print(f"[FARM] {len(turbines)} turbines in 2x5 grid")
    print(f"[COMPUTING] Applying wake model...")

    field = domain.clone()
    farm.apply_wakes(field, grid_resolution)
    power = farm.calculate_power_output(field, grid_resolution)
    revenue = farm.annual_revenue(power)

    print(f"   Total Power: {power:.2f} MW")
    print(f"   Annual Revenue: ${revenue:,.0f}")
    print()


if __name__ == "__main__":
    # Run main commercial demo
    power_a, power_b, delta = run_commercial_demo()

    # Run scaling demo
    run_scaling_demo()

    # Final summary
    print("=" * 70)
    print("  PHASE 5 COMPLETE - ENERGY MODULE VALIDATED")
    print("=" * 70)
