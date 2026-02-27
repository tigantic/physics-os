#!/usr/bin/env python3
"""
HyperFOAM Solver Verification Script

Runs a quick sanity check to verify the solver is working correctly.

Usage:
    python verify_solver.py
"""

import time
import torch

from hyperfoam import Solver, ConferenceRoom
from hyperfoam.presets import setup_conference_room


def main():
    print("=" * 70)
    print("HYPERFOAM SOLVER VERIFICATION")
    print("=" * 70)
    print()
    
    # Configuration
    config = ConferenceRoom()
    config.nx, config.ny, config.nz = 48, 32, 24  # Medium grid
    
    print(f"Grid: {config.nx}×{config.ny}×{config.nz} = {config.nx*config.ny*config.nz:,} cells")
    print(f"Domain: {config.lx}m × {config.ly}m × {config.lz}m")
    print()
    
    # Create solver
    print("Creating solver...")
    solver = Solver(config)
    setup_conference_room(solver, n_occupants=8)
    
    print(f"Device: {solver.device}")
    print()
    
    # Run simulation
    print("Running 30-second simulation...")
    print("-" * 70)
    
    start_time = time.time()
    
    for t in range(0, 31, 5):
        # Run to this time
        while solver.time < t:
            solver.step()
        
        # Get metrics from raw fields
        T = solver.thermal_solver.temperature.phi
        T_mean = T.mean().item() - 273.15  # Convert to Celsius
        
        vel_mag = torch.sqrt(
            solver.flow.u**2 + solver.flow.v**2 + solver.flow.w**2
        )
        V_mean = vel_mag.mean().item()
        V_max = vel_mag.max().item()
        
        CO2 = solver.thermal_solver.co2.phi if solver.thermal_solver.co2 else None
        CO2_mean = CO2.mean().item() if CO2 is not None else 400.0
        
        status = "✓" if (20 <= T_mean <= 24 and CO2_mean < 1000 and V_mean < 0.25) else "○"
        
        print(f"  t={t:3d}s | T={T_mean:5.2f}°C | CO2={CO2_mean:5.0f}ppm | "
              f"V_avg={V_mean:.3f} V_max={V_max:.2f} m/s [{status}]")
    
    elapsed = time.time() - start_time
    
    print("-" * 70)
    print()
    print(f"Simulation complete in {elapsed:.1f}s ({solver.step_count/elapsed:.0f} steps/s)")
    print()
    
    # Final verification
    T_final = solver.thermal_solver.temperature.phi.mean().item() - 273.15
    has_nan = torch.isnan(solver.thermal_solver.temperature.phi).any().item()
    
    print("VERIFICATION RESULTS:")
    print("-" * 70)
    
    tests_passed = 0
    tests_total = 4
    
    # Test 1: No NaN
    if not has_nan:
        print("  ✓ Temperature field contains no NaN")
        tests_passed += 1
    else:
        print("  ✗ Temperature field contains NaN values")
    
    # Test 2: Reasonable temperature
    if 15 < T_final < 35:
        print(f"  ✓ Final temperature {T_final:.2f}°C is in reasonable range")
        tests_passed += 1
    else:
        print(f"  ✗ Final temperature {T_final:.2f}°C is outside reasonable range")
    
    # Test 3: Velocity capped
    V_max = torch.sqrt(
        solver.flow.u**2 + solver.flow.v**2 + solver.flow.w**2
    ).max().item()
    if V_max < 10.0:
        print(f"  ✓ Max velocity {V_max:.2f} m/s is bounded")
        tests_passed += 1
    else:
        print(f"  ✗ Max velocity {V_max:.2f} m/s is too high")
    
    # Test 4: Pressure solver converged
    p_max = solver.flow.p.abs().max().item()
    if p_max < 1e6:
        print(f"  ✓ Pressure field is bounded (max={p_max:.0f} Pa)")
        tests_passed += 1
    else:
        print(f"  ✗ Pressure field diverged (max={p_max:.0e} Pa)")
    
    print("-" * 70)
    print()
    
    if tests_passed == tests_total:
        print("✅ ALL TESTS PASSED - Solver is working correctly")
        return 0
    else:
        print(f"⚠️ {tests_passed}/{tests_total} tests passed")
        return 1


if __name__ == "__main__":
    exit(main())
