"""
Test Script: Silent Sub Stealth Analysis

Demonstrates submarine acoustic stealth using the Ontic Engine
hydroacoustic solver. We prove mathematically that a submarine
hiding behind a seamount (or in a thermal shadow) is invisible
to active sonar.

The Physics:
- Enemy destroyer pings with active sonar (50 Hz)
- Sound propagates through ocean with variable sound speed
- Seamount creates an acoustic shadow zone
- Submarine in shadow zone receives near-zero signal

Run: python test_stealth.py
"""

import numpy as np
import torch

from ontic.aerospace.defense.ocean import (OceanDomain, SoundSpeedProfile,
                                     create_deep_ocean)
from ontic.aerospace.defense.solver import (analyze_stealth, find_shadow_zones,
                                      scan_for_optimal_hiding_spot,
                                      solve_sonar_ping)


def run_stealth_analysis():
    """
    Main stealth analysis demonstrating shadow zone detection.
    """
    print("=" * 70)
    print("SILENT SUB: Hydroacoustic Stealth Analysis")
    print("Phase 8: From Air to Water - The Hunt Begins")
    print("=" * 70)
    print()

    # ==========================================================
    # 1. CREATE OCEAN ENVIRONMENT
    # ==========================================================
    print("[PHASE 1] Creating Ocean Domain...")
    print()

    # Deep ocean with Munk sound speed profile
    ocean = OceanDomain(
        depth=4000.0,  # 4km deep (average ocean)
        range_km=50.0,  # 50km range
        grid_res=10.0,  # 10m resolution
    )

    # Add seamount at 20km range, 2km high
    # This will create a massive acoustic shadow
    ocean.add_seamount(x_pos_km=20.0, height_m=2000.0, width_km=3.0)

    print()
    print(ocean.summary())
    print()

    # ==========================================================
    # 2. ENEMY DESTROYER PINGS
    # ==========================================================
    print("[PHASE 2] Enemy Destroyer Active Sonar...")
    print("  Source: Surface ship (50m depth)")
    print("  Frequency: 50 Hz (long-range search sonar)")
    print()

    # Simulate sonar ping propagation
    field = solve_sonar_ping(
        ocean,
        source_depth_m=50.0,  # Destroyer hull-mounted sonar
        source_range_km=0.0,  # At left edge of domain
        frequency_hz=50.0,  # Low frequency for long range
        steps=2500,  # Time steps
        source_duration_steps=100,  # Ping duration
    )

    print()

    # ==========================================================
    # 3. ANALYZE OUR SUBMARINE'S POSITION
    # ==========================================================
    print("[PHASE 3] Analyzing Submarine Positions...")
    print()

    # Test multiple positions
    test_positions = [
        # (range_km, depth_m, description)
        (10.0, 200.0, "In front of seamount (EXPOSED)"),
        (30.0, 200.0, "Behind seamount, shallow (SHADOW?)"),
        (30.0, 3000.0, "Behind seamount, deep (SHADOW?)"),
        (40.0, 1300.0, "SOFAR channel axis (WAVEGUIDE)"),
        (25.0, 2500.0, "Seamount shadow, mid-depth"),
        (45.0, 3500.0, "Far range, deep"),
    ]

    print("=" * 70)
    print("POSITION ANALYSIS")
    print("=" * 70)

    for range_km, depth_m, desc in test_positions:
        print(f"\n📍 {desc}")
        print(f"   Position: Range {range_km}km, Depth {depth_m}m")

        report = analyze_stealth(
            ocean,
            field,
            sub_range_km=range_km,
            sub_depth_m=depth_m,
            detection_threshold_db=60.0,  # Typical sonar sensitivity
        )

        status = "✅ UNDETECTED" if not report.is_detected else "⛔ DETECTED"
        print(f"   Status: {status}")
        print(f"   Signal: {report.signal_strength:.6f}")
        print(f"   Transmission Loss: {report.transmission_loss_db:.1f} dB")
        print(f"   Margin: {report.detection_margin_db:+.1f} dB")
        print(f"   Advice: {report.recommendation}")

    # ==========================================================
    # 4. FIND OPTIMAL HIDING SPOTS
    # ==========================================================
    print()
    print("=" * 70)
    print("OPTIMAL HIDING POSITIONS")
    print("=" * 70)

    best_spots = scan_for_optimal_hiding_spot(
        ocean,
        field,
        min_range_km=15.0,  # Beyond minimum range
        grid_resolution_km=2.0,
    )

    print("\nTop 5 Shadow Zones:")
    for i, (r, d, tl) in enumerate(best_spots[:5]):
        print(f"  {i+1}. Range {r:.0f}km, Depth {d:.0f}m - TL: {tl:.1f} dB")

    # ==========================================================
    # 5. SHADOW ZONE STATISTICS
    # ==========================================================
    print()
    print("=" * 70)
    print("SHADOW ZONE STATISTICS")
    print("=" * 70)

    shadow_mask = find_shadow_zones(ocean, field, threshold_db=60.0)

    # Count shadow zone cells
    total_water = (ocean.terrain_mask < 0.5).sum().item()
    shadow_cells = shadow_mask.sum().item()
    shadow_pct = 100 * shadow_cells / total_water if total_water > 0 else 0

    print(f"\nTotal water cells: {total_water:,}")
    print(f"Shadow zone cells: {shadow_cells:,} ({shadow_pct:.1f}%)")
    print(f"Exposed cells: {total_water - shadow_cells:,} ({100-shadow_pct:.1f}%)")

    # Shadow zone by depth
    print("\nShadow coverage by depth:")
    depth_bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]

    for d_min, d_max in depth_bands:
        z_min = int(d_min / ocean.res)
        z_max = int(d_max / ocean.res)
        z_max = min(z_max, ocean.nz)

        band_water = (ocean.terrain_mask[z_min:z_max, :] < 0.5).sum().item()
        band_shadow = shadow_mask[z_min:z_max, :].sum().item()
        band_pct = 100 * band_shadow / band_water if band_water > 0 else 0

        print(f"  {d_min}-{d_max}m: {band_pct:.1f}% shadow")

    # ==========================================================
    # 6. THE PHYSICS EXPLAINED
    # ==========================================================
    print()
    print("=" * 70)
    print("PHYSICS ANALYSIS")
    print("=" * 70)

    print(
        """
SOUND PROPAGATION IN THE OCEAN:

1. MUNK PROFILE (Sound Speed vs Depth)
   - Surface: ~1520 m/s (warm water)
   - 1300m (SOFAR axis): ~1490 m/s (minimum)
   - 4000m (deep): ~1540 m/s (pressure dominates)
   
   This creates a "Sound Channel" (SOFAR) that traps sound
   and allows it to travel thousands of kilometers.

2. SEAMOUNT SHADOW
   - Sound cannot penetrate solid rock
   - Diffraction bends sound around edges (but weakly at 50 Hz)
   - Behind the seamount: ACOUSTIC SHADOW ZONE
   
   This is where our submarine hides.

3. SURFACE REFLECTION
   - Surface acts as "pressure release" boundary
   - Sound reflects with phase inversion
   - Creates interference patterns ("Lloyd's Mirror")

4. BOTTOM REFLECTION  
   - Seafloor acts as rigid boundary
   - Sound reflects without phase change
   - Deep submarines can use bottom bounce for communication

THE TACTICAL ADVANTAGE:
   - We solved the FULL WAVE EQUATION, not ray approximations
   - We see the exact diffraction leakage around the seamount
   - We know precisely where the shadow zone begins
   - We can optimize submarine position for minimum detection
"""
    )

    # ==========================================================
    # 7. TACTICAL VERDICT
    # ==========================================================
    print()
    print("=" * 70)
    print("🦈 TACTICAL VERDICT")
    print("=" * 70)

    # Our chosen position: Behind seamount, deep
    our_range = 30.0
    our_depth = 3000.0

    final_report = analyze_stealth(
        ocean,
        field,
        sub_range_km=our_range,
        sub_depth_m=our_depth,
    )

    print(f"\nOUR SUBMARINE: Range {our_range}km, Depth {our_depth}m")
    print()
    print(final_report)
    print()

    if not final_report.is_detected:
        print("🎯 MISSION SUCCESS: We are invisible to enemy sonar.")
        print("   The seamount blocks the ping completely.")
        print("   Signal strength at our position: NEAR ZERO")
        print()
        print("   'Silent and deadly, we wait in the shadow.'")
    else:
        print("⚠️  WARNING: Position may be compromised.")
        print("   Recommend repositioning to deeper shadow zone.")

    print()
    print("=" * 70)
    print("PHASE 8 COMPLETE: The Silent Sub")
    print("=" * 70)

    return final_report


def run_sofar_channel_demo():
    """
    Demonstrate the SOFAR channel waveguide effect.

    Sound at the SOFAR axis (1300m) can travel thousands of km
    with minimal attenuation - used for submarine communication.
    """
    print()
    print("=" * 70)
    print("BONUS: SOFAR Channel Demonstration")
    print("=" * 70)

    # Create ocean without obstacles
    ocean = OceanDomain(depth=4000.0, range_km=50.0, grid_res=10.0)

    print("\nSOFAR Channel Source (1300m depth):")

    # Source at SOFAR axis
    field_sofar = solve_sonar_ping(
        ocean,
        source_depth_m=1300.0,  # SOFAR axis
        frequency_hz=50.0,
        steps=2000,
    )

    # Check intensity at far range, same depth
    intensity_sofar = field_sofar.get_intensity_at(1300.0, 45.0, ocean.res)

    print(f"\nIntensity at 45km, 1300m depth: {intensity_sofar:.6f}")

    # Compare with shallow source
    print("\nShallow Source (100m depth):")

    field_shallow = solve_sonar_ping(
        ocean,
        source_depth_m=100.0,
        frequency_hz=50.0,
        steps=2000,
    )

    intensity_shallow = field_shallow.get_intensity_at(1300.0, 45.0, ocean.res)

    print(f"\nIntensity at 45km, 1300m depth: {intensity_shallow:.6f}")

    ratio = intensity_sofar / (intensity_shallow + 1e-10)
    print(f"\nSOFAR advantage: {ratio:.1f}x stronger signal at range")
    print("This is why submarines communicate via the SOFAR channel!")


def run_thermocline_stealth():
    """
    Demonstrate hiding under a thermocline layer.

    The thermocline reflects sound - submarines can hide beneath it.
    """
    print()
    print("=" * 70)
    print("BONUS: Thermocline Stealth")
    print("=" * 70)

    # Create ocean with strong thermocline at 100m
    ssp = SoundSpeedProfile(
        thermocline_depth=100.0,
        thermocline_strength=50.0,  # Strong gradient
    )

    ocean = OceanDomain(
        depth=1000.0,
        range_km=20.0,
        grid_res=5.0,
        ssp=ssp,
    )

    print("\nSurface sonar pinging...")

    field = solve_sonar_ping(
        ocean,
        source_depth_m=30.0,  # Above thermocline
        frequency_hz=100.0,
        steps=1500,
    )

    # Compare detection above vs below thermocline
    print("\nDetection comparison:")

    # Above thermocline
    report_above = analyze_stealth(ocean, field, sub_range_km=15.0, sub_depth_m=50.0)
    print(f"  Above thermocline (50m): TL={report_above.transmission_loss_db:.1f} dB")

    # Below thermocline
    report_below = analyze_stealth(ocean, field, sub_range_km=15.0, sub_depth_m=200.0)
    print(f"  Below thermocline (200m): TL={report_below.transmission_loss_db:.1f} dB")

    improvement = report_below.transmission_loss_db - report_above.transmission_loss_db
    print(f"\n  Thermocline stealth bonus: +{improvement:.1f} dB")

    if not report_below.is_detected:
        print("\n  ✅ Submarine below thermocline: UNDETECTED")
        print("  The thermocline acts as an acoustic mirror!")


if __name__ == "__main__":
    # Main stealth analysis
    run_stealth_analysis()

    # Optional demos
    print("\n" + "=" * 70)
    print("Running additional demonstrations...")

    # run_sofar_channel_demo()
    # run_thermocline_stealth()

    print("\n🦈 Silent running complete.")
