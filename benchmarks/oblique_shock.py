"""
Oblique Shock Validation Benchmark

Validates the 2D Euler solver against exact oblique shock relations
for supersonic flow over a wedge.

The θ-β-M relation connects:
    θ: Flow deflection angle (wedge half-angle)
    β: Shock wave angle
    M1: Upstream Mach number

    tan(θ) = 2 cot(β) * (M1² sin²(β) - 1) / (M1² (γ + cos(2β)) + 2)

References:
    [1] Anderson, "Modern Compressible Flow", 3rd ed., Ch. 4
    [2] NACA Report 1135, "Equations, Tables, and Charts for
        Compressible Flow", 1953
"""

import math
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensornet.cfd.boundaries import BCType, FlowState
from tensornet.cfd.euler_2d import (Euler2D, Euler2DState, oblique_shock_exact,
                                    supersonic_wedge_ic)
from tensornet.cfd.geometry import ImmersedBoundary, WedgeGeometry


def run_oblique_shock_benchmark(
    M_inf: float = 2.0,
    wedge_angle_deg: float = 10.0,
    Nx: int = 200,
    Ny: int = 100,
    t_final: float = 0.5,
    cfl: float = 0.4,
    verbose: bool = True,
) -> dict:
    """
    Run oblique shock simulation and compare to exact solution.

    Args:
        M_inf: Freestream Mach number
        wedge_angle_deg: Wedge half-angle in degrees
        Nx, Ny: Grid resolution
        t_final: Final simulation time
        cfl: CFL number
        verbose: Print progress

    Returns:
        Dictionary with simulation results and errors
    """
    gamma = 1.4
    wedge_angle = math.radians(wedge_angle_deg)

    # Domain setup
    Lx = 2.0
    Ly = 1.0

    if verbose:
        print("=" * 60)
        print("OBLIQUE SHOCK VALIDATION BENCHMARK")
        print("=" * 60)
        print(f"Freestream Mach: M∞ = {M_inf}")
        print(f"Wedge half-angle: θ = {wedge_angle_deg}°")
        print(f"Grid: {Nx} × {Ny}")
        print()

    # Compute exact solution
    exact = oblique_shock_exact(M_inf, wedge_angle, gamma)

    if verbose:
        print("EXACT OBLIQUE SHOCK RELATIONS:")
        print(f"  Shock angle β = {math.degrees(exact['beta']):.4f}°")
        print(f"  Downstream Mach M2 = {exact['M2']:.4f}")
        print(f"  Pressure ratio p2/p1 = {exact['p2_p1']:.4f}")
        print(f"  Density ratio ρ2/ρ1 = {exact['rho2_rho1']:.4f}")
        print(f"  Temperature ratio T2/T1 = {exact['T2_T1']:.4f}")
        print()

    # Initialize solver
    solver = Euler2D(Nx, Ny, Lx, Ly, gamma=gamma)

    # Initial condition: uniform supersonic flow
    ic = supersonic_wedge_ic(Nx, Ny, M_inf=M_inf, gamma=gamma)
    solver.set_initial_condition(ic)

    # Freestream state
    p_inf = 1.0
    rho_inf = 1.0
    a_inf = math.sqrt(gamma * p_inf / rho_inf)
    u_inf = M_inf * a_inf

    inflow = FlowState(rho=rho_inf, u=u_inf, v=0.0, p=p_inf, gamma=gamma)

    # Boundary conditions
    solver.bc_left = BCType.INFLOW
    solver.bc_right = BCType.OUTFLOW
    solver.bc_bottom = BCType.REFLECTIVE  # Wedge surface
    solver.bc_top = BCType.OUTFLOW
    solver.inflow_state = Euler2DState(
        rho=torch.full((Ny, 1), rho_inf),
        u=torch.full((Ny, 1), u_inf),
        v=torch.zeros((Ny, 1)),
        p=torch.full((Ny, 1), p_inf),
        gamma=gamma,
    )

    # Create wedge geometry
    wedge = WedgeGeometry(
        x_leading_edge=0.2,
        y_leading_edge=0.0,  # Bottom boundary
        half_angle=wedge_angle,
        length=1.5,
    )

    # Run simulation
    if verbose:
        print("Running simulation...")

    result = solver.solve(t_final, cfl=cfl, verbose=verbose)

    if verbose:
        print(f"Completed in {result['steps']} steps")
        print()

    # Extract solution downstream of shock
    # Sample at a location behind the shock
    x_sample = 1.0
    j_sample = Ny // 4  # Above bottom boundary

    i_sample = int(x_sample / (Lx / Nx))

    # Get local flow properties
    state = solver.state
    rho_2 = state.rho[j_sample, i_sample].item()
    u_2 = state.u[j_sample, i_sample].item()
    v_2 = state.v[j_sample, i_sample].item()
    p_2 = state.p[j_sample, i_sample].item()
    M_2 = state.M[j_sample, i_sample].item()

    # Compute errors
    p_ratio_sim = p_2 / p_inf
    rho_ratio_sim = rho_2 / rho_inf

    p_error = abs(p_ratio_sim - exact["p2_p1"]) / exact["p2_p1"] * 100
    rho_error = abs(rho_ratio_sim - exact["rho2_rho1"]) / exact["rho2_rho1"] * 100
    M_error = abs(M_2 - exact["M2"]) / exact["M2"] * 100

    if verbose:
        print("SIMULATION RESULTS vs EXACT:")
        print(
            f"  Pressure ratio: {p_ratio_sim:.4f} (exact: {exact['p2_p1']:.4f}, error: {p_error:.2f}%)"
        )
        print(
            f"  Density ratio:  {rho_ratio_sim:.4f} (exact: {exact['rho2_rho1']:.4f}, error: {rho_error:.2f}%)"
        )
        print(
            f"  Downstream Mach: {M_2:.4f} (exact: {exact['M2']:.4f}, error: {M_error:.2f}%)"
        )
        print()

    # Validation
    passed = p_error < 10 and rho_error < 10 and M_error < 10

    if verbose:
        if passed:
            print("✓ BENCHMARK PASSED: Oblique shock captured within 10% tolerance")
        else:
            print("✗ BENCHMARK FAILED: Errors exceed 10% tolerance")
        print("=" * 60)

    return {
        "exact": exact,
        "simulated": {
            "p_ratio": p_ratio_sim,
            "rho_ratio": rho_ratio_sim,
            "M2": M_2,
        },
        "errors": {
            "p_percent": p_error,
            "rho_percent": rho_error,
            "M_percent": M_error,
        },
        "passed": passed,
        "steps": result["steps"],
        "time": result["time"],
    }


def convergence_study(
    M_inf: float = 2.0,
    wedge_angle_deg: float = 10.0,
    resolutions: list[int] = [50, 100, 200, 400],
    verbose: bool = True,
) -> dict:
    """
    Grid convergence study for oblique shock.

    Args:
        M_inf: Freestream Mach number
        wedge_angle_deg: Wedge half-angle
        resolutions: List of Nx values to test
        verbose: Print progress

    Returns:
        Dictionary with convergence data
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GRID CONVERGENCE STUDY")
        print("=" * 60)

    results = []

    for Nx in resolutions:
        Ny = Nx // 2
        if verbose:
            print(f"\nResolution: {Nx} × {Ny}")

        result = run_oblique_shock_benchmark(
            M_inf=M_inf,
            wedge_angle_deg=wedge_angle_deg,
            Nx=Nx,
            Ny=Ny,
            t_final=0.3,
            verbose=False,
        )

        results.append(
            {
                "Nx": Nx,
                "Ny": Ny,
                "dx": 2.0 / Nx,
                "p_error": result["errors"]["p_percent"],
                "rho_error": result["errors"]["rho_percent"],
                "M_error": result["errors"]["M_percent"],
            }
        )

        if verbose:
            print(f"  Pressure error: {result['errors']['p_percent']:.2f}%")
            print(f"  Density error:  {result['errors']['rho_percent']:.2f}%")

    # Compute convergence rate
    if len(results) >= 2:
        import math

        p_errors = [r["p_error"] for r in results]
        dx_vals = [r["dx"] for r in results]

        # Fit log(error) = p * log(dx) + c
        # Convergence rate p ≈ (log(e1) - log(e2)) / (log(dx1) - log(dx2))
        if p_errors[-1] > 0 and p_errors[-2] > 0:
            rate = (math.log(p_errors[-2]) - math.log(p_errors[-1])) / (
                math.log(dx_vals[-2]) - math.log(dx_vals[-1])
            )
        else:
            rate = float("nan")

        if verbose:
            print(f"\nEstimated convergence rate: O(Δx^{rate:.2f})")
    else:
        rate = None

    return {"results": results, "convergence_rate": rate}


def parameter_study(
    Mach_numbers: list[float] = [1.5, 2.0, 3.0, 5.0],
    wedge_angle_deg: float = 10.0,
    Nx: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Study oblique shock properties across Mach numbers.

    Args:
        Mach_numbers: List of freestream Mach numbers
        wedge_angle_deg: Wedge half-angle
        Nx: Grid resolution
        verbose: Print results

    Returns:
        Dictionary with parameter study data
    """
    if verbose:
        print("\n" + "=" * 60)
        print("MACH NUMBER PARAMETER STUDY")
        print(f"Wedge angle: θ = {wedge_angle_deg}°")
        print("=" * 60)
        print(f"{'M∞':>6} {'β (°)':>8} {'M2':>8} {'p2/p1':>8} {'ρ2/ρ1':>8}")
        print("-" * 60)

    results = []
    wedge_angle = math.radians(wedge_angle_deg)

    for M in Mach_numbers:
        exact = oblique_shock_exact(M, wedge_angle)

        results.append(
            {
                "M_inf": M,
                "beta_deg": math.degrees(exact["beta"]),
                "M2": exact["M2"],
                "p_ratio": exact["p2_p1"],
                "rho_ratio": exact["rho2_rho1"],
            }
        )

        if verbose:
            print(
                f"{M:6.1f} {math.degrees(exact['beta']):8.2f} "
                f"{exact['M2']:8.4f} {exact['p2_p1']:8.4f} "
                f"{exact['rho2_rho1']:8.4f}"
            )

    return {"results": results}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Oblique Shock Validation Benchmark")
    parser.add_argument(
        "--mach", type=float, default=2.0, help="Freestream Mach number"
    )
    parser.add_argument(
        "--angle", type=float, default=10.0, help="Wedge half-angle (degrees)"
    )
    parser.add_argument("--nx", type=int, default=100, help="Grid cells in x")
    parser.add_argument("--ny", type=int, default=50, help="Grid cells in y")
    parser.add_argument(
        "--convergence", action="store_true", help="Run convergence study"
    )
    parser.add_argument(
        "--parameter-study", action="store_true", help="Run Mach number parameter study"
    )

    args = parser.parse_args()

    if args.convergence:
        convergence_study(M_inf=args.mach, wedge_angle_deg=args.angle)
    elif args.parameter_study:
        parameter_study(wedge_angle_deg=args.angle)
    else:
        run_oblique_shock_benchmark(
            M_inf=args.mach, wedge_angle_deg=args.angle, Nx=args.nx, Ny=args.ny
        )
