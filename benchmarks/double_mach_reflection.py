"""
Double Mach Reflection Benchmark
================================

The Double Mach Reflection (DMR) problem is a stringent test for
shock-capturing schemes. A Mach 10 shock at 60° angle impacts a
reflecting wall, creating a complex pattern of shocks and slip lines.

This is the Woodward & Colella (1984) test case, widely used to
validate high-resolution schemes.

Initial Condition:
    - Domain: [0, 4] × [0, 1]
    - Mach 10 shock at 60° from vertical
    - Shock initially at x = 1/6 on bottom wall
    - Reflecting wall at y = 0 for x > 1/6

Expected Features:
    - Primary Mach stem
    - Reflected shock
    - Slip line (contact discontinuity)
    - Triple point structure

References:
    [1] Woodward & Colella, "The Numerical Simulation of Two-Dimensional
        Fluid Flow with Strong Shocks", JCP 54:115-173, 1984
    [2] Quirk, "A Contribution to the Great Riemann Solver Debate",
        IJFD 18:555-574, 1994
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tensornet.cfd.euler_2d import BCType, Euler2D, Euler2DState


def double_mach_reflection_ic(
    Nx: int,
    Ny: int,
    Lx: float = 4.0,
    Ly: float = 1.0,
    x_shock: float = 1.0 / 6.0,
    gamma: float = 1.4,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> Euler2DState:
    """
    Create initial condition for Double Mach Reflection.

    A Mach 10 shock at 60° angle from the vertical axis.

    Args:
        Nx, Ny: Grid dimensions
        Lx, Ly: Domain size
        x_shock: Initial shock foot position
        gamma: Ratio of specific heats

    Returns:
        Euler2DState with DMR initial condition
    """
    dx = Lx / Nx
    dy = Ly / Ny

    x = torch.linspace(dx / 2, Lx - dx / 2, Nx, dtype=dtype, device=device)
    y = torch.linspace(dy / 2, Ly - dy / 2, Ny, dtype=dtype, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    # Shock angle: 60° from x-axis (30° from vertical)
    # The shock moves in direction (cos(60°), sin(60°)) = (0.5, √3/2)
    angle = math.pi / 3  # 60 degrees

    # Pre-shock state (ahead of shock, right side)
    rho_R = 1.4
    p_R = 1.0
    u_R = 0.0
    v_R = 0.0

    # Post-shock state from Rankine-Hugoniot for M=10
    M_shock = 10.0

    # Normal shock relations
    p_ratio = (2 * gamma * M_shock**2 - (gamma - 1)) / (gamma + 1)
    rho_ratio = (gamma + 1) * M_shock**2 / ((gamma - 1) * M_shock**2 + 2)

    p_L = p_R * p_ratio
    rho_L = rho_R * rho_ratio

    # Shock velocity
    a_R = math.sqrt(gamma * p_R / rho_R)
    W = M_shock * a_R  # Shock speed

    # Post-shock velocity in shock-fixed frame
    u_post = W * (1 - 1 / rho_ratio)

    # Rotate to lab frame (shock at 60°)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    u_L = u_post * cos_a
    v_L = u_post * sin_a

    # Shock position: line from (x_shock, 0) at angle
    # Points left of line are post-shock
    # Line equation: x - x_shock = (y - 0) / tan(angle)
    # Or: x = x_shock + y / tan(60°) = x_shock + y / √3
    shock_x = x_shock + Y / math.tan(angle)

    # Initialize arrays
    rho = torch.where(X < shock_x, rho_L, rho_R)
    u = torch.where(X < shock_x, u_L, u_R)
    v = torch.where(X < shock_x, v_L, v_R)
    p = torch.where(X < shock_x, p_L, p_R)

    return Euler2DState(rho=rho, u=u, v=v, p=p, gamma=gamma)


def run_double_mach_reflection(
    Nx: int = 480,
    Ny: int = 120,
    t_final: float = 0.2,
    cfl: float = 0.3,
    save_output: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run Double Mach Reflection simulation.

    Args:
        Nx, Ny: Grid resolution (recommended 4:1 aspect ratio)
        t_final: Simulation end time
        cfl: CFL number
        save_output: Save results to file
        verbose: Print progress

    Returns:
        Dictionary with simulation results
    """
    gamma = 1.4
    Lx = 4.0
    Ly = 1.0
    x_shock_init = 1.0 / 6.0

    if verbose:
        print("=" * 70)
        print("DOUBLE MACH REFLECTION BENCHMARK")
        print("Woodward & Colella (1984)")
        print("=" * 70)
        print(f"Grid: {Nx} × {Ny}")
        print(f"Domain: [{0}, {Lx}] × [{0}, {Ly}]")
        print(f"Shock: Mach 10 at 60° angle")
        print(f"Final time: t = {t_final}")
        print()

    # Initialize solver
    solver = Euler2D(Nx, Ny, Lx, Ly, gamma=gamma)

    # Initial condition
    ic = double_mach_reflection_ic(Nx, Ny, Lx, Ly, x_shock_init, gamma)
    solver.set_initial_condition(ic)

    # Boundary conditions
    # Bottom: reflecting wall for x > x_shock, inflow for x < x_shock
    # Top: post-shock inflow (moving shock)
    # Left: post-shock inflow
    # Right: outflow
    solver.bc_left = BCType.INFLOW  # Post-shock
    solver.bc_right = BCType.OUTFLOW
    solver.bc_bottom = BCType.REFLECTIVE  # Simplified: full reflective
    solver.bc_top = BCType.OUTFLOW

    if verbose:
        print("Simulation progress:")
        print("-" * 70)

    # Run simulation
    steps = 0
    max_steps = 50000
    output_interval = max(1, max_steps // 20)

    while solver.time < t_final and steps < max_steps:
        dt = solver.compute_dt(cfl)
        dt = min(dt, t_final - solver.time)

        solver.step(dt)
        steps += 1

        if verbose and steps % output_interval == 0:
            M_max = solver.state.M.max().item()
            rho_max = solver.state.rho.max().item()
            print(
                f"  Step {steps:5d}: t = {solver.time:.4f}, "
                f"M_max = {M_max:.2f}, ρ_max = {rho_max:.2f}"
            )

    if verbose:
        print("-" * 70)
        print(f"Completed: {steps} steps, t = {solver.time:.6f}")
        print()

    # Analysis
    state = solver.state

    # Find approximate Mach stem height
    # Look at x = 2.0 slice
    i_slice = int(2.0 / (Lx / Nx))
    rho_slice = state.rho[:, i_slice]

    # Find jump location (Mach stem foot)
    rho_diff = torch.abs(rho_slice[1:] - rho_slice[:-1])
    j_stem = torch.argmax(rho_diff).item()
    stem_height = j_stem * (Ly / Ny)

    if verbose:
        print("RESULTS:")
        print(f"  Maximum density: {state.rho.max().item():.4f}")
        print(f"  Maximum pressure: {state.p.max().item():.4f}")
        print(f"  Maximum Mach number: {state.M.max().item():.4f}")
        print(f"  Approximate Mach stem height at x=2: {stem_height:.4f}")
        print()

    results = {
        "config": {
            "Nx": Nx,
            "Ny": Ny,
            "Lx": Lx,
            "Ly": Ly,
            "t_final": t_final,
            "cfl": cfl,
        },
        "statistics": {
            "rho_max": state.rho.max().item(),
            "rho_min": state.rho.min().item(),
            "p_max": state.p.max().item(),
            "M_max": state.M.max().item(),
            "mach_stem_height": stem_height,
            "steps": steps,
            "final_time": solver.time,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_output:
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)

        import json

        output_file = output_dir / "double_mach_reflection_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save flow field
        field_file = output_dir / "double_mach_reflection_field.pt"
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
            print(f"Results saved to: {output_file}")
            print(f"Flow field saved to: {field_file}")

    if verbose:
        print("=" * 70)

    return results


def convergence_study(
    resolutions: list[tuple[int, int]] = [(120, 30), (240, 60), (480, 120), (960, 240)],
    t_final: float = 0.2,
    verbose: bool = True,
) -> dict:
    """
    Grid convergence study for DMR.

    Args:
        resolutions: List of (Nx, Ny) tuples
        t_final: Simulation end time
        verbose: Print progress

    Returns:
        Dictionary with convergence data
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DOUBLE MACH REFLECTION: GRID CONVERGENCE STUDY")
        print("=" * 70)

    results = []

    for Nx, Ny in resolutions:
        if verbose:
            print(f"\nResolution: {Nx} × {Ny}")

        result = run_double_mach_reflection(
            Nx=Nx, Ny=Ny, t_final=t_final, verbose=False, save_output=False
        )

        results.append(
            {
                "Nx": Nx,
                "Ny": Ny,
                "rho_max": result["statistics"]["rho_max"],
                "mach_stem_height": result["statistics"]["mach_stem_height"],
                "steps": result["statistics"]["steps"],
            }
        )

        if verbose:
            print(
                f"  ρ_max = {result['statistics']['rho_max']:.4f}, "
                f"stem height = {result['statistics']['mach_stem_height']:.4f}"
            )

    return {"results": results}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Double Mach Reflection Benchmark")
    parser.add_argument("--nx", type=int, default=240, help="Grid cells in x")
    parser.add_argument("--ny", type=int, default=60, help="Grid cells in y")
    parser.add_argument("--time", type=float, default=0.2, help="Final time")
    parser.add_argument("--cfl", type=float, default=0.3, help="CFL number")
    parser.add_argument(
        "--convergence", action="store_true", help="Run convergence study"
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save output")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if args.convergence:
        convergence_study()
    else:
        run_double_mach_reflection(
            Nx=args.nx,
            Ny=args.ny,
            t_final=args.time,
            cfl=args.cfl,
            save_output=not args.no_save,
            verbose=not args.quiet,
        )
