"""
Mach 5 Supersonic Wedge Flow Simulation
========================================

Simulates hypersonic flow (M=5) over a sharp wedge, demonstrating
the core capability of Project HyperTensor for aerospace CFD.

Physical Setup:
    - Freestream: M∞ = 5.0, p∞ = 1 atm, T∞ = 300 K (normalized)
    - Wedge half-angle: θ = 15°
    - Domain: 2L × 1L with wedge at x = 0.3L

Expected Physics:
    - Attached oblique shock at β ≈ 24.3°
    - Post-shock Mach M₂ ≈ 3.50
    - Pressure ratio p₂/p₁ ≈ 4.78

References:
    [1] Anderson, "Modern Compressible Flow", 3rd ed., Ch. 4
    [2] NACA Report 1135, "Equations, Tables, and Charts for
        Compressible Flow", 1953
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensornet.cfd.euler_2d import (BCType, Euler2D, Euler2DState,
                                    oblique_shock_exact, supersonic_wedge_ic)
from tensornet.cfd.geometry import ImmersedBoundary, WedgeGeometry


def run_mach5_wedge(
    Nx: int = 200,
    Ny: int = 100,
    t_final: float = 0.5,
    cfl: float = 0.4,
    wedge_angle_deg: float = 15.0,
    M_inf: float = 5.0,
    save_output: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run Mach 5 wedge flow simulation.

    Args:
        Nx, Ny: Grid resolution
        t_final: Simulation end time
        cfl: CFL number for stability
        wedge_angle_deg: Wedge half-angle in degrees
        M_inf: Freestream Mach number
        save_output: Whether to save results to file
        verbose: Print progress information

    Returns:
        Dictionary with simulation results and validation data
    """
    gamma = 1.4
    wedge_angle = math.radians(wedge_angle_deg)

    # Domain setup
    Lx = 2.0
    Ly = 1.0

    if verbose:
        print("=" * 70)
        print("PROJECT HYPERTENSOR: MACH 5 WEDGE FLOW SIMULATION")
        print("=" * 70)
        print(f"Date: {datetime.now().isoformat()}")
        print()
        print("CONFIGURATION:")
        print(f"  Freestream Mach number: M_inf = {M_inf}")
        print(f"  Wedge half-angle: theta = {wedge_angle_deg} deg")
        print(f"  Grid resolution: {Nx} x {Ny}")
        print(f"  Domain: {Lx} x {Ly}")
        print(f"  CFL number: {cfl}")
        print(f"  Final time: {t_final}")
        print()

    # Compute exact oblique shock solution
    exact = oblique_shock_exact(M_inf, wedge_angle, gamma)

    if verbose:
        print("EXACT OBLIQUE SHOCK RELATIONS (theta-beta-M):")
        print(f"  Shock angle: beta = {math.degrees(exact['beta']):.4f} deg")
        print(f"  Downstream Mach: M2 = {exact['M2']:.4f}")
        print(f"  Pressure ratio: p2/p1 = {exact['p2_p1']:.4f}")
        print(f"  Density ratio: rho2/rho1 = {exact['rho2_rho1']:.4f}")
        print(f"  Temperature ratio: T2/T1 = {exact['T2_T1']:.4f}")
        print()

    # Initialize solver
    solver = Euler2D(Nx, Ny, Lx, Ly, gamma=gamma)

    # Initial condition: uniform supersonic freestream
    ic = supersonic_wedge_ic(Nx, Ny, M_inf=M_inf, gamma=gamma)
    solver.set_initial_condition(ic)

    # Freestream conditions
    p_inf = 1.0
    rho_inf = 1.0
    a_inf = math.sqrt(gamma * p_inf / rho_inf)
    u_inf = M_inf * a_inf

    # Boundary conditions
    solver.bc_left = BCType.SUPERSONIC_INFLOW
    solver.bc_right = BCType.OUTFLOW
    solver.bc_bottom = BCType.REFLECTIVE  # Symmetry / wedge surface
    solver.bc_top = BCType.OUTFLOW

    # Set inflow state for supersonic inflow BC
    solver.inflow_state = ic

    # Create wedge geometry
    wedge = WedgeGeometry(
        x_leading_edge=0.3,
        y_leading_edge=0.0,  # Bottom of domain
        half_angle=wedge_angle,
        length=1.5,
    )

    # Create grid for immersed boundary
    dx = Lx / Nx
    dy = Ly / Ny
    x = torch.linspace(dx / 2, Lx - dx / 2, Nx, dtype=torch.float64)
    y = torch.linspace(dy / 2, Ly - dy / 2, Ny, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    # Set up immersed boundary for actual wedge
    ib = ImmersedBoundary(wedge, X, Y)

    if verbose:
        n_solid = ib.mask.sum().item()
        n_ghost = ib.ghost_mask.sum().item()
        print(f"  Wedge cells: {n_solid} solid, {n_ghost} ghost")
        print()

    if verbose:
        print("SIMULATION PROGRESS:")
        print("-" * 70)

    # Run simulation
    steps = 0
    max_steps = 10000
    output_interval = max(1, max_steps // 10)

    while solver.time < t_final and steps < max_steps:
        dt = solver.compute_dt(cfl)
        dt = min(dt, t_final - solver.time)

        solver.step(dt)

        # Apply immersed boundary condition after each step
        U = solver.state.to_conservative()
        U = ib.apply(U)
        solver.state = Euler2DState.from_conservative(U, gamma)

        steps += 1

        if verbose and steps % output_interval == 0:
            max_M = solver.state.M.max().item()
            min_p = solver.state.p.min().item()
            print(
                f"  Step {steps:5d}: t = {solver.time:.6f}, "
                f"dt = {dt:.2e}, M_max = {max_M:.3f}, p_min = {min_p:.4f}"
            )

    if verbose:
        print("-" * 70)
        print(f"Simulation completed: {steps} steps, t = {solver.time:.6f}")
        print()

    # Extract results at measurement location
    # Sample behind the expected shock location
    # For wedge at x=0.3 with shock angle beta, shock height at x is:
    # y_shock = (x - x_le) * tan(beta)
    # We sample just below the shock to get post-shock values
    beta = exact["beta"]
    x_sample = 1.2  # Further downstream where shock is developed
    y_shock = (x_sample - 0.3) * math.tan(beta)
    y_wedge = (x_sample - 0.3) * math.tan(wedge_angle)
    y_sample = (y_shock + y_wedge) / 2  # Midpoint between shock and wedge

    # Find maximum pressure location in post-shock region as alternative sample point
    state = solver.state
    # Mask out wedge region (inside solid)
    p_masked = state.p.clone()
    p_masked[ib.mask] = 0.0

    # Find location with significant pressure increase (in post-shock region)
    p_freestream = 1.0
    post_shock_mask = p_masked > 1.5 * p_freestream
    if post_shock_mask.any():
        # Use average in post-shock region
        p_ratio_field = p_masked / p_freestream
        rho_ratio_field = state.rho / 1.0
        M_field = state.M

        # Get indices where post-shock conditions exist
        post_shock_indices = torch.nonzero(post_shock_mask)
        if len(post_shock_indices) > 0:
            # Sample from center of post-shock region
            mid_idx = len(post_shock_indices) // 2
            j_sample = post_shock_indices[mid_idx, 0].item()
            i_sample = post_shock_indices[mid_idx, 1].item()
            x_sample = (i_sample + 0.5) * (Lx / Nx)
            y_sample = (j_sample + 0.5) * (Ly / Ny)
    else:
        # Fallback to geometric calculation
        pass

    i_sample = int(x_sample / (Lx / Nx))
    j_sample = int(y_sample / (Ly / Ny))

    state = solver.state

    # Measured values
    rho_measured = state.rho[j_sample, i_sample].item()
    u_measured = state.u[j_sample, i_sample].item()
    v_measured = state.v[j_sample, i_sample].item()
    p_measured = state.p[j_sample, i_sample].item()
    M_measured = state.M[j_sample, i_sample].item()

    # Compute ratios relative to freestream
    p_ratio = p_measured / p_inf
    rho_ratio = rho_measured / rho_inf

    # Compute errors
    p_error = abs(p_ratio - exact["p2_p1"]) / exact["p2_p1"] * 100
    rho_error = abs(rho_ratio - exact["rho2_rho1"]) / exact["rho2_rho1"] * 100
    M_error = abs(M_measured - exact["M2"]) / exact["M2"] * 100

    if verbose:
        print("VALIDATION RESULTS:")
        print(f"  Sample location: ({x_sample}, {y_sample})")
        print()
        print(f"  {'Quantity':<20} {'Simulated':>12} {'Exact':>12} {'Error':>10}")
        print(f"  {'-'*54}")
        print(
            f"  {'Pressure ratio':<20} {p_ratio:>12.4f} {exact['p2_p1']:>12.4f} {p_error:>9.2f}%"
        )
        print(
            f"  {'Density ratio':<20} {rho_ratio:>12.4f} {exact['rho2_rho1']:>12.4f} {rho_error:>9.2f}%"
        )
        print(
            f"  {'Downstream Mach':<20} {M_measured:>12.4f} {exact['M2']:>12.4f} {M_error:>9.2f}%"
        )
        print()

    # Flow field statistics
    M_max = state.M.max().item()
    M_min = state.M.min().item()
    p_max = state.p.max().item()
    p_min = state.p.min().item()

    if verbose:
        print("FLOW FIELD STATISTICS:")
        print(f"  Mach number range: [{M_min:.4f}, {M_max:.4f}]")
        print(f"  Pressure range: [{p_min:.4f}, {p_max:.4f}]")
        print()

    # Validation status
    tolerance = 15.0  # percent
    passed = p_error < tolerance and rho_error < tolerance and M_error < tolerance

    if verbose:
        if passed:
            print(
                "[PASS] SIMULATION VALIDATED: Results within 15% of exact oblique shock theory"
            )
        else:
            print("[WARN] SIMULATION WARNING: Some results exceed 15% tolerance")
            print(
                "  (Expected for coarse grid - increase resolution for better accuracy)"
            )
        print("=" * 70)

    # Prepare results
    results = {
        "config": {
            "M_inf": M_inf,
            "wedge_angle_deg": wedge_angle_deg,
            "Nx": Nx,
            "Ny": Ny,
            "t_final": t_final,
            "cfl": cfl,
        },
        "exact": {
            "beta_deg": math.degrees(exact["beta"]),
            "M2": exact["M2"],
            "p_ratio": exact["p2_p1"],
            "rho_ratio": exact["rho2_rho1"],
            "T_ratio": exact["T2_T1"],
        },
        "simulated": {
            "p_ratio": p_ratio,
            "rho_ratio": rho_ratio,
            "M2": M_measured,
        },
        "errors_percent": {
            "p": p_error,
            "rho": rho_error,
            "M": M_error,
        },
        "statistics": {
            "M_range": [M_min, M_max],
            "p_range": [p_min, p_max],
            "steps": steps,
            "final_time": solver.time,
        },
        "passed": passed,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_output:
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "mach5_wedge_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        if verbose:
            print(f"\nResults saved to: {output_file}")

        # Save flow field for visualization
        field_file = output_dir / "mach5_wedge_field.pt"
        torch.save(
            {
                "rho": state.rho,
                "u": state.u,
                "v": state.v,
                "p": state.p,
                "M": state.M,
                "x": solver.x,
                "y": solver.y,
            },
            field_file,
        )

        if verbose:
            print(f"Flow field saved to: {field_file}")

    return results


def generate_visualization_script():
    """Generate matplotlib visualization script for results."""

    script = '''"""
Visualization script for Mach 5 wedge simulation results.
Run after executing mach5_wedge.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_dir = Path(__file__).parent.parent / 'results'
field = torch.load(results_dir / 'mach5_wedge_field.pt', weights_only=True)

# Extract data
rho = field['rho'].numpy()
u = field['u'].numpy()
v = field['v'].numpy()
p = field['p'].numpy()
M = field['M'].numpy()
x = field['x'].numpy()
y = field['y'].numpy()

# Create meshgrid for plotting
X, Y = np.meshgrid(x, y)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Mach 5 Wedge Flow Simulation', fontsize=14, fontweight='bold')

# Mach number contour
ax = axes[0, 0]
cf = ax.contourf(X, Y, M, levels=50, cmap='jet')
ax.set_title('Mach Number')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(cf, ax=ax, label='M')

# Pressure contour
ax = axes[0, 1]
cf = ax.contourf(X, Y, p, levels=50, cmap='hot')
ax.set_title('Pressure')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(cf, ax=ax, label='p')

# Density contour
ax = axes[1, 0]
cf = ax.contourf(X, Y, rho, levels=50, cmap='viridis')
ax.set_title('Density')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(cf, ax=ax, label='ρ')

# Velocity magnitude
velocity_mag = np.sqrt(u**2 + v**2)
ax = axes[1, 1]
cf = ax.contourf(X, Y, velocity_mag, levels=50, cmap='plasma')
ax.set_title('Velocity Magnitude')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(cf, ax=ax, label='|V|')

plt.tight_layout()
plt.savefig(results_dir / 'mach5_wedge_contours.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Figure saved to {results_dir / 'mach5_wedge_contours.png'}")
'''

    return script


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mach 5 Supersonic Wedge Flow Simulation"
    )
    parser.add_argument("--nx", type=int, default=200, help="Grid cells in x")
    parser.add_argument("--ny", type=int, default=100, help="Grid cells in y")
    parser.add_argument("--mach", type=float, default=5.0, help="Freestream Mach")
    parser.add_argument("--angle", type=float, default=15.0, help="Wedge angle (deg)")
    parser.add_argument("--time", type=float, default=0.5, help="Final time")
    parser.add_argument("--cfl", type=float, default=0.4, help="CFL number")
    parser.add_argument("--no-save", action="store_true", help="Don't save output")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    results = run_mach5_wedge(
        Nx=args.nx,
        Ny=args.ny,
        M_inf=args.mach,
        wedge_angle_deg=args.angle,
        t_final=args.time,
        cfl=args.cfl,
        save_output=not args.no_save,
        verbose=not args.quiet,
    )
