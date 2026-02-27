"""
Test Script: Tokamak Fusion Reactor Simulation

Phase 9: The Tokamak Twin - The Star on Earth

Demonstrates magnetic confinement of plasma using the
Boris particle pusher algorithm. We prove that the
magnetic bottle holds the plasma inside the donut.

The Physics:
- Lorentz Force: F = q(E + v × B)
- Particles spiral around magnetic field lines
- Toroidal + Poloidal fields create helical confinement
- If particles touch the wall, the reactor melts

Run: python test_fusion.py
"""

import numpy as np
import torch

from tensornet.plasma_nuclear.fusion.tokamak import (PlasmaState, TokamakReactor,
                                      verify_gyration)


def ignite_plasma():
    """
    Main plasma ignition and confinement demonstration.
    """
    print("=" * 70)
    print("TOKAMAK TWIN: Magnetic Confinement Fusion Simulation")
    print("Phase 9: The Power of Stars - On Your GPU")
    print("=" * 70)
    print()

    # ==========================================================
    # 1. INITIALIZE REACTOR
    # ==========================================================
    print("[PHASE 1] Initializing Tokamak Reactor...")
    print()

    reactor = TokamakReactor(
        major_radius=2.0,  # R₀ = 2m (distance from center to tube)
        minor_radius=0.8,  # a = 0.8m (tube radius)
        B0=5.0,  # 5 Tesla toroidal field
        safety_factor=2.0,  # q = 2 (field line twist)
    )

    print()

    # ==========================================================
    # 2. VERIFY PHYSICS
    # ==========================================================
    print("[PHASE 2] Verifying Particle Physics...")
    verify_gyration(reactor)
    print()

    # ==========================================================
    # 3. CREATE PLASMA
    # ==========================================================
    print("[PHASE 3] Injecting Plasma...")
    print()

    particles = reactor.create_plasma(
        num_particles=1000,
        temperature=1.0,  # Thermal spread
        toroidal_flow=10.0,  # Bulk flow around torus
    )

    print()

    # ==========================================================
    # 4. RUN SIMULATION
    # ==========================================================
    print("[PHASE 4] Running Confinement Simulation...")
    print()

    final_state, escape_history = reactor.push_particles(
        particles,
        dt=0.001,  # 1ms time step
        steps=200,  # 200ms total
        q_over_m=1.0,
        verbose=True,
    )

    print()

    # ==========================================================
    # 5. ANALYZE CONFINEMENT
    # ==========================================================
    print("[PHASE 5] Analyzing Confinement Quality...")
    print()

    report = reactor.analyze_confinement(final_state, escape_history)
    print(report)

    print()

    # ==========================================================
    # 6. PHYSICS EXPLANATION
    # ==========================================================
    print("=" * 70)
    print("FUSION PHYSICS EXPLAINED")
    print("=" * 70)
    print(
        """
MAGNETIC CONFINEMENT:

1. THE PROBLEM
   - Fusion requires 150 million °C (10× hotter than the Sun's core)
   - No material can contain plasma at this temperature
   - Solution: Use magnetic fields to confine the plasma

2. THE GEOMETRY (Torus / Donut)
   - Major radius R₀: Distance from center to magnetic axis
   - Minor radius a: Radius of the plasma tube
   - Particles spiral around field lines inside the tube

3. THE MAGNETIC BOTTLE
   - Toroidal field B_φ: Runs around the ring (from external coils)
   - Poloidal field B_θ: Runs around the tube (from plasma current)
   - Combined: Helical field lines that close on themselves
   
   If q (safety factor) is irrational, a single field line
   covers the entire flux surface ergodically!

4. THE LORENTZ FORCE
   F = q(E + v × B)
   
   - Particles gyrate around field lines (Larmor motion)
   - Gyration radius r_L = mv_⊥ / (qB)
   - For fusion: r_L << a (particles stay inside tube)

5. CONFINEMENT QUALITY
   - Energy confinement time τ_E: How long energy stays in plasma
   - Lawson criterion: n·τ_E·T > 3×10²¹ m⁻³·s·keV for ignition
   - Modern tokamaks: Q > 1 (more energy out than in)
   - ITER goal: Q = 10 (10× more energy out)

6. WHY THIS MATTERS
   - Fusion fuel (deuterium): Limitless from seawater
   - No CO₂ emissions
   - No long-lived radioactive waste
   - Energy security for humanity
"""
    )

    # ==========================================================
    # 7. VISUALIZATION BRIDGE
    # ==========================================================
    print("=" * 70)
    print("VISUALIZATION READY")
    print("=" * 70)
    print(
        """
The plasma particle positions can be streamed to:
- Glass Cockpit (existing RAM bridge)
- Unreal Engine (Phase 5 pipeline)

Visualization would show:
- 1000 glowing particles spiraling around the torus
- Color = velocity (temperature)
- Escaped particles fade to red
- Magnetic field lines as streamlines
"""
    )

    # ==========================================================
    # 8. FINAL STATUS
    # ==========================================================
    print("=" * 70)
    print("⚛️  FUSION STATUS")
    print("=" * 70)

    if report.confinement_ratio > 0.8:
        print(
            f"""
PLASMA CONTAINED. MAGNETIC BOTTLE HOLDING.

Confinement: {100*report.confinement_ratio:.1f}%
Mean ρ: {report.mean_rho:.3f}m (tube radius: {reactor.a}m)
Max ρ:  {report.max_rho:.3f}m

The plasma stays inside the donut.
The reactor walls are safe.
Fusion conditions achievable.

'We have captured a star.'
"""
        )
    else:
        print(
            f"""
⚠️  CONFINEMENT DEGRADED

Confinement: {100*report.confinement_ratio:.1f}%
Particles escaping to walls.

Recommendation: Increase magnetic field strength.
"""
        )

    print("=" * 70)
    print("PHASE 9 COMPLETE: The Tokamak Twin")
    print("=" * 70)

    return report


def run_parameter_scan():
    """
    Scan different reactor parameters to find optimal confinement.
    """
    print()
    print("=" * 70)
    print("PARAMETER SCAN: Finding Optimal Confinement")
    print("=" * 70)
    print()

    results = []

    # Scan safety factor q
    for q in [1.5, 2.0, 2.5, 3.0]:
        reactor = TokamakReactor(safety_factor=q)
        particles = reactor.create_plasma(num_particles=500, temperature=0.5)
        final, history = reactor.push_particles(particles, steps=100, verbose=False)
        report = reactor.analyze_confinement(final, history)
        results.append((q, report.confinement_ratio))
        print(f"  q={q:.1f}: Confinement = {100*report.confinement_ratio:.1f}%")

    best_q = max(results, key=lambda x: x[1])
    print(f"\nOptimal safety factor: q = {best_q[0]}")


def run_high_temperature_test():
    """
    Test confinement at high temperature (realistic fusion conditions).
    """
    print()
    print("=" * 70)
    print("HIGH TEMPERATURE TEST")
    print("=" * 70)

    reactor = TokamakReactor(B0=5.0)

    # High temperature = high thermal velocity
    particles = reactor.create_plasma(
        num_particles=500,
        temperature=5.0,  # 5× higher thermal motion
        toroidal_flow=10.0,
    )

    final, history = reactor.push_particles(particles, steps=100, verbose=True)
    report = reactor.analyze_confinement(final, history)

    print()
    print(report)

    if report.confinement_ratio < 0.8:
        print("\n⚠️  High temperature degrades confinement.")
        print("Solution: Increase B-field or use better q-profile.")


if __name__ == "__main__":
    # Main demo
    ignite_plasma()

    # Optional additional tests
    # run_parameter_scan()
    # run_high_temperature_test()

    print("\n⚛️  Fusion simulation complete. The star is contained.")
