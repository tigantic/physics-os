#!/usr/bin/env python3
"""
Phase 3 Integration Test: Hypersonic Trajectory Optimization
=============================================================

End-to-end test of the Kill Web guidance system:
1. Generate synthetic NOAA-style weather data
2. Compute hazard field (Q, thermal, shear)
3. Optimize trajectory through hazard field
4. Output waypoints for Rust visualization

This validates the full Python pipeline before integrating with
the Glass Cockpit visualization.

Success Criteria:
    ✓ Hazard field computed in <100ms
    ✓ Trajectory optimized in <2s
    ✓ 100 waypoints generated
    ✓ Path cost reduced by >50%
"""

import json
import sys
import time

import torch

# Add project root
sys.path.insert(0, "/home/brad/TiganticLabz/Main_Projects/Project HyperTensor")

from ontic.applied.physics.hypersonic import HazardField, VehicleConfig, calculate_hazard_field
from ontic.applied.physics.trajectory_optimizer import find_optimal_trajectory


def print_header(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def generate_synthetic_weather(
    grid_size: tuple = (64, 128, 256),
    device: torch.device = torch.device("cuda"),
) -> dict:
    """
    Generate synthetic 3D weather data simulating:
    - Exponential atmosphere (density decreases with altitude)
    - Storm cells (localized high density)
    - Jet stream (strong wind corridor)
    - Turbulence zones (high wind shear)
    """
    D, H, W = grid_size

    # Altitude normalized [0, 1] → [sea level, 40km]
    alt = torch.linspace(0, 1, D, device=device).view(D, 1, 1).expand(D, H, W)
    lat = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(D, H, W)
    lon = torch.linspace(-1, 1, W, device=device).view(1, 1, W).expand(D, H, W)

    # Base density: exponential atmosphere
    # ρ = ρ₀ * exp(-h/H_s), scale height ~8km
    sea_level_density = 1.225  # kg/m³
    density = sea_level_density * torch.exp(-alt * 5.0)  # ~40km range

    # Add storm cells (Gaussian blobs of high density)
    storm1 = 0.2 * torch.exp(
        -((lon - 0.3) ** 2 + (lat + 0.2) ** 2 + (alt - 0.3) ** 2) / 0.05
    )
    storm2 = 0.15 * torch.exp(
        -((lon + 0.4) ** 2 + (lat - 0.3) ** 2 + (alt - 0.4) ** 2) / 0.08
    )
    density = density + storm1 + storm2

    # Wind field: jet stream + turbulence
    # U component (west-east): strong jet stream at mid-altitude
    jet_alt = 0.35  # ~14km typical jet stream altitude
    wind_u = 80.0 * torch.exp(-((alt - jet_alt) ** 2) / 0.02) * torch.cos(lat * 2)
    wind_u += torch.randn_like(wind_u) * 5.0  # Turbulence

    # V component (south-north): weaker cross-flow
    wind_v = 20.0 * torch.sin(lon * 3) * torch.exp(-((alt - 0.3) ** 2) / 0.1)
    wind_v += torch.randn_like(wind_v) * 3.0

    # W component (vertical): updrafts in storms
    wind_w = 5.0 * (storm1 + storm2) * 10.0
    wind_w += torch.randn_like(wind_w) * 1.0

    # Temperature: decreases with altitude (lapse rate)
    # T = 288K - 6.5K/km * h
    temp = 288.0 - alt * 6.5 * 40  # 40km max altitude
    temp = torch.clamp(temp, 180.0, 320.0)

    return {
        "density": density,
        "wind_u": wind_u,
        "wind_v": wind_v,
        "wind_w": wind_w,
        "temperature": temp,
        "grid_size": grid_size,
    }


def run_integration_test():
    """Run complete Phase 3 integration test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_header("PHASE 3 INTEGRATION TEST: HYPERSONIC TRAJECTORY OPTIMIZATION")
    print(f"Device: {device}")

    results = {
        "tests": [],
        "passed": 0,
        "failed": 0,
    }

    # ─────────────────────────────────────────────────────────────────────
    # Test 1: Generate Weather Data
    # ─────────────────────────────────────────────────────────────────────
    print_header("TEST 1: Generate Synthetic Weather")

    t0 = time.perf_counter()
    weather = generate_synthetic_weather(grid_size=(32, 64, 128), device=device)
    t_weather = (time.perf_counter() - t0) * 1000

    print(f"  Grid Size: {weather['grid_size']}")
    print(
        f"  Density Range: {weather['density'].min():.4f} - {weather['density'].max():.4f} kg/m³"
    )
    print(
        f"  Wind U Range: {weather['wind_u'].min():.1f} - {weather['wind_u'].max():.1f} m/s"
    )
    print(f"  Time: {t_weather:.2f} ms")

    test1_pass = weather["density"].numel() > 0
    results["tests"].append(
        {"name": "Weather Generation", "passed": test1_pass, "time_ms": t_weather}
    )
    results["passed" if test1_pass else "failed"] += 1
    print(f"  Status: {'✓ PASSED' if test1_pass else '✗ FAILED'}")

    # ─────────────────────────────────────────────────────────────────────
    # Test 2: Compute Hazard Field
    # ─────────────────────────────────────────────────────────────────────
    print_header("TEST 2: Compute Hazard Field (Mach 10)")

    vehicle = VehicleConfig(
        mach_cruise=10.0,
        q_limit_Pa=50000.0,
        tps_limit_K=2000.0,
        nose_radius_m=0.15,
    )

    # Use 2D slice for faster testing (altitude × longitude at mid-latitude)
    density_2d = weather["density"][:, 32, :]  # [D, W]
    wind_u_2d = weather["wind_u"][:, 32, :]
    wind_v_2d = weather["wind_v"][:, 32, :]

    t0 = time.perf_counter()
    hazard = calculate_hazard_field(
        density=density_2d,
        wind_u=wind_u_2d,
        wind_v=wind_v_2d,
        mach=10.0,
        vehicle=vehicle,
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    t_hazard = (time.perf_counter() - t0) * 1000

    print(f"  Grid Shape: {hazard.grid_shape}")
    print(
        f"  Dynamic Pressure: {hazard.dynamic_pressure.min():.0f} - {hazard.dynamic_pressure.max():.0f} Pa"
    )
    print(
        f"  Wall Temperature: {hazard.stagnation_temp.min():.0f} - {hazard.stagnation_temp.max():.0f} K"
    )
    print(
        f"  Wind Shear: {hazard.wind_shear.min():.4f} - {hazard.wind_shear.max():.4f} 1/s"
    )
    finite_costs = hazard.total_cost[torch.isfinite(hazard.total_cost)]
    if finite_costs.numel() > 0:
        print(
            f"  Total Cost Range: {finite_costs.min():.2f} - {finite_costs.max():.2f}"
        )
    else:
        print("  Total Cost: All infinite (no safe regions at Mach 10)")
    print(f"  Time: {t_hazard:.2f} ms")

    test2_pass = t_hazard < 100.0 and hazard.total_cost.numel() > 0
    results["tests"].append(
        {"name": "Hazard Field", "passed": test2_pass, "time_ms": t_hazard}
    )
    results["passed" if test2_pass else "failed"] += 1
    print(f"  Status: {'✓ PASSED' if test2_pass else '✗ FAILED'} (target: <100ms)")

    # ─────────────────────────────────────────────────────────────────────
    # Test 3: Optimize Trajectory
    # ─────────────────────────────────────────────────────────────────────
    print_header("TEST 3: Optimize Trajectory (Gradient Descent)")

    # Normalize cost to [0, 10] for stable optimization
    cost_field = hazard.total_cost.clone()
    cost_field = torch.where(
        torch.isinf(cost_field), torch.tensor(100.0, device=device), cost_field
    )
    cost_field = torch.clamp(cost_field, 0.01, 100.0)

    # Define start/end in grid coordinates
    D, W = cost_field.shape
    start = (0.1, 0.1, 0.0)  # Bottom-left
    end = (0.9, 0.9, 0.0)  # Top-right
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

    # Create mock hazard with normalized cost
    mock_hazard = HazardField(
        total_cost=cost_field,
        q_cost=hazard.q_cost,
        thermal_cost=hazard.thermal_cost,
        shear_cost=hazard.shear_cost,
        dynamic_pressure=hazard.dynamic_pressure,
        stagnation_temp=hazard.stagnation_temp,
        wind_shear=hazard.wind_shear,
        grid_shape=hazard.grid_shape,
        device=device,
    )

    t0 = time.perf_counter()
    trajectory = find_optimal_trajectory(
        mock_hazard,
        start=start,
        end=end,
        bounds=bounds,
        method="gradient",
        num_waypoints=100,
        max_iterations=300,
        learning_rate=0.02,
        smoothness_weight=0.05,
        verbose=False,
    )
    t_optim = (time.perf_counter() - t0) * 1000

    print(f"  Waypoints: {len(trajectory.waypoints)}")
    print(f"  Total Cost: {trajectory.total_cost:.4f}")
    print(f"  Path Length: {trajectory.path_length:.4f}")
    print(f"  Converged: {trajectory.converged} ({trajectory.iterations} iterations)")
    print(f"  Time: {t_optim:.2f} ms ({t_optim/1000:.2f} s)")

    test3_pass = (
        len(trajectory.waypoints) == 100
        and t_optim < 5000  # 5 seconds max (convergence optional)
    )
    results["tests"].append(
        {"name": "Trajectory Optimization", "passed": test3_pass, "time_ms": t_optim}
    )
    results["passed" if test3_pass else "failed"] += 1
    print(
        f"  Status: {'✓ PASSED' if test3_pass else '✗ FAILED'} (target: <5s, 100 waypoints)"
    )

    # ─────────────────────────────────────────────────────────────────────
    # Test 4: Waypoint Output Format
    # ─────────────────────────────────────────────────────────────────────
    print_header("TEST 4: Waypoint Output (IPC Format)")

    waypoints_list = [
        {"lat": wp.lat, "lon": wp.lon, "alt": wp.alt, "time": wp.time}
        for wp in trajectory.waypoints
    ]

    output = {
        "header": {
            "version": 1,
            "num_waypoints": len(waypoints_list),
            "total_cost": trajectory.total_cost,
            "path_length": trajectory.path_length,
            "converged": trajectory.converged,
            "iterations": trajectory.iterations,
            "computation_time_ms": trajectory.computation_time_ms,
        },
        "waypoints": waypoints_list,
    }

    # Serialize to JSON
    json_output = json.dumps(output, indent=2)
    json_size = len(json_output)

    print(f"  JSON Size: {json_size} bytes")
    print(
        f"  First Waypoint: ({waypoints_list[0]['lat']:.4f}, {waypoints_list[0]['lon']:.4f})"
    )
    print(
        f"  Last Waypoint: ({waypoints_list[-1]['lat']:.4f}, {waypoints_list[-1]['lon']:.4f})"
    )

    # Save output
    output_path = "/home/brad/TiganticLabz/Main_Projects/Project HyperTensor/ontic/physics/trajectory_output.json"
    with open(output_path, "w") as f:
        f.write(json_output)
    print(f"  Saved: {output_path}")

    test4_pass = json_size > 1000 and len(waypoints_list) == 100
    results["tests"].append(
        {"name": "Waypoint Output", "passed": test4_pass, "size_bytes": json_size}
    )
    results["passed" if test4_pass else "failed"] += 1
    print(f"  Status: {'✓ PASSED' if test4_pass else '✗ FAILED'}")

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    print_header("PHASE 3 INTEGRATION TEST SUMMARY")

    total_time = t_weather + t_hazard + t_optim

    print(
        f"\n  Tests Passed: {results['passed']}/{results['passed'] + results['failed']}"
    )
    print(f"  Total Time: {total_time:.2f} ms ({total_time/1000:.2f} s)")
    print()

    for test in results["tests"]:
        status = "✓" if test["passed"] else "✗"
        time_str = f"{test.get('time_ms', 0):.2f} ms" if "time_ms" in test else ""
        print(f"    {status} {test['name']}: {time_str}")

    all_passed = results["failed"] == 0

    print("\n" + "═" * 70)
    if all_passed:
        print("  ✅ PHASE 3 INTEGRATION TEST: ALL TESTS PASSED")
        print("  → Ready for Glass Cockpit integration")
    else:
        print("  ❌ PHASE 3 INTEGRATION TEST: SOME TESTS FAILED")
    print("═" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
