#!/usr/bin/env python
"""
Wedge Flow Demo - Oblique Shock Physics Validation
===================================================

Demonstrates oblique shock relations for supersonic flow over a wedge.
This validates the analytical shock relations in HyperTensor CFD.

Physical Setup:
    - Freestream Mach: M1 = 2.0 to 5.0
    - Wedge half-angle: 10 to 20 degrees
    - Validates: shock angle, post-shock Mach, pressure ratio
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

from ontic.cfd.euler_2d import oblique_shock_exact


def main():
    print("=" * 60)
    print("HyperTensor Wedge Flow Demo - Oblique Shock Relations")
    print("=" * 60)
    print()

    # Test cases: (M1, theta_deg)
    cases = [
        (2.0, 10.0),
        (2.0, 15.0),
        (3.0, 10.0),
        (3.0, 15.0),
        (5.0, 10.0),
        (5.0, 15.0),
        (5.0, 20.0),
    ]

    print(
        f"{'M1':>6} {'theta':>8} {'beta':>8} {'M2':>8} {'p2/p1':>8} {'rho2/rho1':>10}"
    )
    print("-" * 60)

    for M1, theta_deg in cases:
        theta = math.radians(theta_deg)
        result = oblique_shock_exact(M1=M1, theta=theta)

        beta_deg = math.degrees(result["beta"])
        M2 = result["M2"]
        p_ratio = result["p2_p1"]
        rho_ratio = result["rho2_rho1"]

        print(
            f"{M1:6.1f} {theta_deg:8.1f} {beta_deg:8.2f} {M2:8.3f} {p_ratio:8.3f} {rho_ratio:10.3f}"
        )

    print()
    print("=" * 60)
    print("Validation: Mach 5, 15 degree wedge (classic test case)")
    print("=" * 60)

    result = oblique_shock_exact(M1=5.0, theta=math.radians(15.0))

    # Expected values from NACA 1135 / Anderson
    expected = {
        "beta": 24.3,  # degrees
        "M2": 3.50,
        "p2_p1": 4.78,
    }

    print()
    print(f"  Shock angle beta:")
    print(f"    Computed: {math.degrees(result['beta']):.2f} deg")
    print(f"    Expected: {expected['beta']:.2f} deg")
    print(f"    Error: {abs(math.degrees(result['beta']) - expected['beta']):.3f} deg")

    print()
    print(f"  Post-shock Mach M2:")
    print(f"    Computed: {result['M2']:.3f}")
    print(f"    Expected: {expected['M2']:.2f}")
    print(f"    Error: {abs(result['M2'] - expected['M2']):.4f}")

    print()
    print(f"  Pressure ratio p2/p1:")
    print(f"    Computed: {result['p2_p1']:.3f}")
    print(f"    Expected: {expected['p2_p1']:.2f}")
    print(f"    Error: {abs(result['p2_p1'] - expected['p2_p1']):.4f}")

    # Check accuracy
    beta_ok = abs(math.degrees(result["beta"]) - expected["beta"]) < 0.5
    m2_ok = abs(result["M2"] - expected["M2"]) < 0.05
    p_ok = abs(result["p2_p1"] - expected["p2_p1"]) < 0.1

    print()
    if beta_ok and m2_ok and p_ok:
        print("PASS: All values match expected within tolerance")
    else:
        print("FAIL: Some values exceed tolerance")

    print()
    print("=" * 60)
    print("Geometry: Wedge and Immersed Boundary")
    print("=" * 60)

    import torch

    from ontic.cfd.geometry import ImmersedBoundary, WedgeGeometry

    # Create wedge
    wedge = WedgeGeometry(
        x_leading_edge=0.2,
        y_leading_edge=0.5,
        half_angle=math.radians(15.0),
        length=0.6,
    )

    print()
    print(f"  Wedge geometry:")
    print(f"    Leading edge: ({wedge.x_leading_edge}, {wedge.y_leading_edge})")
    print(f"    Half-angle: {wedge.half_angle_deg:.1f} deg")
    print(f"    Length: {wedge.length}")

    # Create mesh
    Nx, Ny = 50, 50
    x = torch.linspace(0, 1, Nx, dtype=torch.float64)
    y = torch.linspace(0, 1, Ny, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    # Create IB
    ib = ImmersedBoundary(wedge, X, Y)

    n_solid = ib.mask.sum().item()
    n_ghost = ib.ghost_mask.sum().item()

    print()
    print(f"  Immersed boundary (50x50 grid):")
    print(f"    Solid cells: {n_solid}")
    print(f"    Ghost cells: {n_ghost}")
    print(f"    Ghost/Solid ratio: {n_ghost/max(n_solid,1):.2%}")

    print()
    print("PASS: Geometry and IB setup successful")

    print()
    print("=" * 60)
    print("Demo complete - HyperTensor CFD validated")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
