"""
HyperFOAM Quick Start Example

Demonstrates the high-level API for HVAC simulation.

Run: python examples/quickstart.py
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hyperfoam
from hyperfoam.presets import setup_conference_room

def main():
    print("=" * 60)
    print("HyperFOAM Quick Start")
    print("=" * 60)
    
    # 1. Create solver with preset configuration
    print("\n[1] Creating Conference Room simulation...")
    config = hyperfoam.ConferenceRoom()
    solver = hyperfoam.Solver(config)
    
    # 2. Setup room (table, occupants, HVAC)
    print("\n[2] Setting up room...")
    setup_conference_room(solver, n_occupants=12)
    
    # 3. Run simulation
    print("\n[3] Running simulation (5 minutes)...")
    
    def progress(t, metrics):
        if t % 30 < 0.02:  # Print every 30 seconds
            print(f"  t={t:5.0f}s | T={metrics['T']:.2f}°C | "
                  f"CO2={metrics['CO2']:.0f}ppm | V={metrics['V']:.3f}m/s")
    
    solver.solve(duration=300, callback=progress)
    
    # 4. Check results
    print("\n[4] Results:")
    solver.print_results()
    
    # 5. Access raw data
    metrics = solver.get_comfort_metrics()
    if metrics['overall_pass']:
        print("\n✓ Ready for deployment!")
        return 0
    else:
        print("\n⚠ Additional tuning needed")
        return 1


if __name__ == "__main__":
    exit(main())
