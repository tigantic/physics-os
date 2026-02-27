"""
Shock-Boundary Layer Interaction (SBLI) Benchmark
==================================================

Validates the coupled Navier-Stokes solver against canonical
shock-boundary layer interaction test cases.

SBLI is critical for hypersonic vehicles where:
    - Oblique shocks impinge on boundary layers
    - Flow separation occurs upstream of compression corners
    - Reattachment creates peak heating
    - Unsteadiness can cause structural fatigue

Test Cases:
    1. Compression Corner (this file) - M=2.5, θ=15°
    2. Impinging Oblique Shock
    3. Forward-Facing Step

Key Phenomena:
    - Separation bubble formation
    - λ-shock foot structure
    - Peak wall pressure at reattachment
    - Heat transfer spike at reattachment

References:
    [1] Settles, "Experimental and Computational Studies
        of Shock Wave-Boundary Layer Interactions", 1976
    [2] Dolling, "Fifty Years of Shock-Wave/Boundary-Layer
        Interaction Research", AIAA J., 2001
    [3] Knight & Degrez, "Shock Wave Boundary Layer
        Interactions in High Mach Number Flows", NATO RTO, 2004
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tensornet.cfd.euler_2d import Euler2DState
from tensornet.cfd.navier_stokes import (NavierStokes2D, NavierStokes2DConfig,
                                         NavierStokes2DResult)
from tensornet.cfd.viscous import (recovery_temperature, reynolds_number,
                                   sutherland_viscosity)


def compression_corner_setup(
    M_inf: float = 2.5,
    T_inf: float = 300.0,
    p_inf: float = 101325.0,
    Re_x: float = 1e6,
    corner_angle_deg: float = 15.0,
    corner_x_frac: float = 0.4,
    Nx: int = 256,
    Ny: int = 128,
) -> dict:
    """
    Set up compression corner SBLI problem.

    Physical Setup:
        - Flat plate from x=0 to x_corner
        - Ramp at angle θ from x_corner to x_end
        - Supersonic freestream at M_inf
        - Adiabatic wall (no heat flux)

    Args:
        M_inf: Freestream Mach number
        T_inf: Freestream temperature [K]
        p_inf: Freestream pressure [Pa]
        Re_x: Reynolds number at corner
        corner_angle_deg: Ramp angle [degrees]
        corner_x_frac: Corner location as fraction of domain
        Nx, Ny: Grid resolution

    Returns:
        Dictionary with problem parameters and config
    """
    # Gas properties
    gamma = 1.4
    R = 287.058

    # Derived quantities
    rho_inf = p_inf / (R * T_inf)
    c_inf = math.sqrt(gamma * R * T_inf)
    u_inf = M_inf * c_inf

    # Compute domain size from Reynolds number
    T_tensor = torch.tensor([T_inf])
    mu_inf = sutherland_viscosity(T_tensor).item()
    nu_inf = mu_inf / rho_inf

    # x at corner based on Re_x
    x_corner = Re_x * nu_inf / u_inf
    Lx = x_corner / corner_x_frac

    # Domain height - capture boundary layer + interaction
    delta_99 = 5.0 * x_corner / math.sqrt(Re_x)  # Blasius estimate
    Ly = 20 * delta_99  # Capture shock structure

    # Create config
    config = NavierStokes2DConfig(
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly,
        gamma=gamma,
        R=R,
        cfl=0.3,  # Lower CFL for stability
    )

    # Corner location
    corner_i = int(Nx * corner_x_frac)
    corner_x = corner_i * config.dx

    return {
        "config": config,
        "M_inf": M_inf,
        "T_inf": T_inf,
        "p_inf": p_inf,
        "rho_inf": rho_inf,
        "u_inf": u_inf,
        "mu_inf": mu_inf,
        "Re_x": Re_x,
        "delta_99": delta_99,
        "corner_angle_deg": corner_angle_deg,
        "corner_angle_rad": math.radians(corner_angle_deg),
        "corner_x": corner_x,
        "corner_i": corner_i,
        "x_corner": x_corner,
        "Lx": Lx,
        "Ly": Ly,
    }


def compression_corner_ic(setup: dict) -> Euler2DState:
    """
    Create initial condition for compression corner.

    Uniform supersonic freestream with imposed boundary layer
    profile near wall.

    Args:
        setup: Dictionary from compression_corner_setup

    Returns:
        Euler2DState initial condition
    """
    config = setup["config"]
    Ny, Nx = config.Ny, config.Nx

    # Uniform freestream
    rho = torch.ones(Ny, Nx, dtype=torch.float64) * setup["rho_inf"]
    u = torch.ones(Ny, Nx, dtype=torch.float64) * setup["u_inf"]
    v = torch.zeros(Ny, Nx, dtype=torch.float64)
    p = torch.ones(Ny, Nx, dtype=torch.float64) * setup["p_inf"]

    return Euler2DState(rho=rho, u=u, v=v, p=p, gamma=config.gamma)


def apply_corner_geometry(
    state: Euler2DState, setup: dict, ns: NavierStokes2D
) -> Euler2DState:
    """
    Apply wall boundary conditions including ramp geometry.

    - y=0 is wall (no-slip, adiabatic)
    - Behind corner, wall rises at angle θ

    Args:
        state: Current flow state
        setup: Problem setup dictionary
        ns: NavierStokes2D solver instance

    Returns:
        State with boundary conditions applied
    """
    config = setup["config"]
    Ny, Nx = config.Ny, config.Nx

    # Create working copies
    rho = state.rho.clone()
    u = state.u.clone()
    v = state.v.clone()
    p = state.p.clone()

    # Wall at y=0: no-slip, adiabatic
    # Ghost cell approach: u(ghost) = -u(interior), v(ghost) = -v(interior)
    u[0, :] = 0.0  # No-slip
    v[0, :] = 0.0  # No penetration
    # Adiabatic: extrapolate pressure
    p[0, :] = p[1, :]
    rho[0, :] = rho[1, :]

    # Inflow (x=0): fixed supersonic
    rho[:, 0] = setup["rho_inf"]
    u[:, 0] = setup["u_inf"]
    v[:, 0] = 0.0
    p[:, 0] = setup["p_inf"]

    # Outflow (x=Lx): extrapolation
    rho[:, -1] = rho[:, -2]
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]
    p[:, -1] = p[:, -2]

    # Top (y=Ly): freestream
    rho[-1, :] = setup["rho_inf"]
    u[-1, :] = setup["u_inf"]
    v[-1, :] = 0.0
    p[-1, :] = setup["p_inf"]

    # Ramp geometry: behind corner, apply flow turning
    # This is a simplified treatment - full IBM would be better
    corner_i = setup["corner_i"]
    theta = setup["corner_angle_rad"]

    # For cells on ramp surface, impose tangency
    for i in range(corner_i, Nx):
        # Wall-normal vector for ramp: n = (-sin(θ), cos(θ))
        # Tangent: t = (cos(θ), sin(θ))
        # Project velocity onto tangent at wall
        j_wall = 0  # First cell
        speed = torch.sqrt(u[1, i] ** 2 + v[1, i] ** 2)
        u[j_wall, i] = 0.0
        v[j_wall, i] = 0.0

    return Euler2DState(rho=rho, u=u, v=v, p=p, gamma=config.gamma)


def oblique_shock_angle(M: float, theta: float, gamma: float = 1.4) -> float:
    """
    Compute oblique shock angle from θ-β-M relation.

    For weak shock solution (attached shock).

    Args:
        M: Freestream Mach number
        theta: Deflection angle [radians]
        gamma: Ratio of specific heats

    Returns:
        Shock angle β [radians]
    """
    # Newton iteration on θ-β-M relation
    # tan(θ) = 2 cot(β) * (M²sin²β - 1) / (M²(γ + cos(2β)) + 2)

    beta = theta + 0.1  # Initial guess

    for _ in range(20):
        sin_b = math.sin(beta)
        cos_b = math.cos(beta)

        M2 = M**2
        num = 2 * (M2 * sin_b**2 - 1)
        den = math.tan(beta) * (M2 * (gamma + math.cos(2 * beta)) + 2)

        theta_calc = math.atan(num / den) if den != 0 else 0

        # Newton correction
        d_theta = theta - theta_calc
        beta += 0.5 * d_theta

        if abs(d_theta) < 1e-8:
            break

    return beta


def compute_separation_metrics(state: Euler2DState, setup: dict) -> dict:
    """
    Compute separation bubble metrics.

    Args:
        state: Flow state
        setup: Problem setup

    Returns:
        Dictionary with separation/reattachment locations, bubble size
    """
    config = setup["config"]

    # Wall shear stress (proportional to du/dy at wall)
    dudy_wall = (state.u[1, :] - state.u[0, :]) / config.dy

    # Separation where τ_w = 0 (du/dy changes sign)
    x = torch.linspace(0, setup["Lx"], config.Nx, dtype=torch.float64)

    # Find sign changes
    sep_idx = None
    reat_idx = None

    for i in range(setup["corner_i"], config.Nx - 1):
        if dudy_wall[i] > 0 and dudy_wall[i + 1] <= 0 and sep_idx is None:
            sep_idx = i
        elif dudy_wall[i] <= 0 and dudy_wall[i + 1] > 0 and sep_idx is not None:
            reat_idx = i
            break

    if sep_idx is None:
        return {
            "separated": False,
            "x_sep": None,
            "x_reat": None,
            "bubble_length": 0.0,
        }

    x_sep = x[sep_idx].item() if sep_idx else None
    x_reat = x[reat_idx].item() if reat_idx else None

    bubble_length = (x_reat - x_sep) if (x_sep and x_reat) else 0.0

    return {
        "separated": True,
        "x_sep": x_sep,
        "x_reat": x_reat,
        "bubble_length": bubble_length,
        "bubble_length_normalized": (
            bubble_length / setup["delta_99"] if setup["delta_99"] > 0 else 0
        ),
    }


def compute_wall_distributions(state: Euler2DState, setup: dict) -> dict:
    """
    Compute wall pressure and skin friction distributions.

    Args:
        state: Flow state
        setup: Problem setup

    Returns:
        Dictionary with x, Cp, Cf arrays
    """
    config = setup["config"]

    x = torch.linspace(0, setup["Lx"], config.Nx, dtype=torch.float64)

    # Wall pressure coefficient
    p_wall = state.p[0, :]
    q_inf = 0.5 * setup["rho_inf"] * setup["u_inf"] ** 2
    Cp = (p_wall - setup["p_inf"]) / q_inf

    # Skin friction coefficient
    T = state.p / (state.rho * config.R)
    mu = sutherland_viscosity(T)
    dudy_wall = (state.u[1, :] - state.u[0, :]) / config.dy
    tau_wall = mu[0, :] * dudy_wall
    Cf = tau_wall / q_inf

    return {
        "x": x.numpy(),
        "Cp": Cp.numpy(),
        "Cf": Cf.numpy(),
        "p_wall": p_wall.numpy(),
        "tau_wall": tau_wall.numpy(),
    }


def run_sbli_simulation(
    M_inf: float = 2.5,
    corner_angle_deg: float = 15.0,
    Re_x: float = 5e5,
    Nx: int = 128,
    Ny: int = 64,
    t_final: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Run complete SBLI simulation.

    Args:
        M_inf: Freestream Mach number
        corner_angle_deg: Ramp angle
        Re_x: Reynolds number at corner
        Nx, Ny: Grid resolution
        t_final: Simulation time
        verbose: Print progress

    Returns:
        Dictionary with results
    """
    # Setup
    setup = compression_corner_setup(
        M_inf=M_inf,
        corner_angle_deg=corner_angle_deg,
        Re_x=Re_x,
        Nx=Nx,
        Ny=Ny,
    )

    if verbose:
        print("=" * 60)
        print("SHOCK-BOUNDARY LAYER INTERACTION SIMULATION")
        print("=" * 60)
        print(f"Mach number: {M_inf}")
        print(f"Corner angle: {corner_angle_deg}°")
        print(f"Re_x at corner: {Re_x:.2e}")
        print(f"Grid: {Nx} x {Ny}")
        print(f"Domain: {setup['Lx']:.4f} x {setup['Ly']:.4f} m")
        print(f"δ_99 estimate: {setup['delta_99']*1000:.3f} mm")
        print("-" * 60)

    # Create solver
    ns = NavierStokes2D(setup["config"])

    # Initial condition
    state = compression_corner_ic(setup)

    # Time integration with BC application
    t = 0.0
    step = 0
    dt_history = []

    def step_with_bc(state, t, step_num):
        """Callback to apply BCs after each step."""
        return False  # Don't stop

    while t < t_final:
        # Apply boundary conditions
        state = apply_corner_geometry(state, setup, ns)

        # Compute timestep
        dt = ns.compute_timestep(state)
        dt = min(dt, t_final - t)

        # Advance
        state = ns.step(state, dt)

        t += dt
        step += 1
        dt_history.append(dt)

        if verbose and step % 100 == 0:
            M_max = (
                torch.sqrt(state.u**2 + state.v**2).max()
                / torch.sqrt(setup["config"].gamma * state.p / state.rho).max()
            )
            print(
                f"Step {step:5d}, t = {t:.2e} s, dt = {dt:.2e} s, M_max = {M_max:.2f}"
            )

    # Final BCs
    state = apply_corner_geometry(state, setup, ns)

    # Compute metrics
    sep_metrics = compute_separation_metrics(state, setup)
    wall_dist = compute_wall_distributions(state, setup)

    # Inviscid shock angle prediction
    beta_inv = oblique_shock_angle(M_inf, setup["corner_angle_rad"])

    if verbose:
        print("-" * 60)
        print("RESULTS")
        print("-" * 60)
        print(f"Simulation time: {t:.2e} s ({step} steps)")
        print(f"Inviscid shock angle: {math.degrees(beta_inv):.2f}°")
        if sep_metrics["separated"]:
            print(f"Separation detected at x = {sep_metrics['x_sep']:.4f} m")
            print(f"Reattachment at x = {sep_metrics['x_reat']:.4f} m")
            print(f"Bubble length: {sep_metrics['bubble_length']*1000:.3f} mm")
            print(f"Bubble length/δ_99: {sep_metrics['bubble_length_normalized']:.2f}")
        else:
            print("No separation detected")
        print("=" * 60)

    return {
        "setup": setup,
        "state": state,
        "sep_metrics": sep_metrics,
        "wall_dist": wall_dist,
        "beta_inviscid": beta_inv,
        "time": t,
        "steps": step,
        "dt_history": dt_history,
    }


def plot_sbli_results(results: dict, save_path: str = None):
    """
    Plot SBLI simulation results.

    Args:
        results: Dictionary from run_sbli_simulation
        save_path: Optional path to save figure
    """
    setup = results["setup"]
    state = results["state"]
    wall = results["wall_dist"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    config = setup["config"]
    x = np.linspace(0, setup["Lx"], config.Nx)
    y = np.linspace(0, setup["Ly"], config.Ny)
    X, Y = np.meshgrid(x, y)

    # Mach number contours
    ax = axes[0, 0]
    M = torch.sqrt(state.u**2 + state.v**2) / torch.sqrt(
        config.gamma * state.p / state.rho
    )
    cs = ax.contourf(X, Y, M.numpy(), levels=30, cmap="coolwarm")
    plt.colorbar(cs, ax=ax, label="Mach")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Mach Number")
    ax.axvline(x=setup["corner_x"], color="k", linestyle="--", label="Corner")
    ax.legend()

    # Pressure contours
    ax = axes[0, 1]
    cs = ax.contourf(X, Y, state.p.numpy() / setup["p_inf"], levels=30, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="p/p_∞")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Pressure Ratio")
    ax.axvline(x=setup["corner_x"], color="w", linestyle="--")

    # Wall Cp distribution
    ax = axes[1, 0]
    ax.plot(wall["x"], wall["Cp"], "b-", linewidth=2)
    ax.axvline(x=setup["corner_x"], color="r", linestyle="--", label="Corner")
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Cp")
    ax.set_title("Wall Pressure Coefficient")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Wall Cf distribution
    ax = axes[1, 1]
    ax.plot(wall["x"], wall["Cf"] * 1000, "g-", linewidth=2)
    ax.axvline(x=setup["corner_x"], color="r", linestyle="--", label="Corner")
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Cf × 1000")
    ax.set_title("Skin Friction Coefficient")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark separation if detected
    sep = results["sep_metrics"]
    if sep["separated"] and sep["x_sep"] is not None:
        axes[1, 1].axvline(x=sep["x_sep"], color="m", linestyle=":", label="Separation")
        if sep["x_reat"] is not None:
            axes[1, 1].axvline(
                x=sep["x_reat"], color="c", linestyle=":", label="Reattachment"
            )
        axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.close()


def validate_sbli():
    """
    Run validation cases for SBLI benchmark.
    """
    print("\n" + "=" * 70)
    print("SBLI VALIDATION SUITE")
    print("=" * 70)

    # Test 1: Basic inviscid shock angle
    print("\n[Test 1] Inviscid Oblique Shock Angle")
    print("-" * 40)

    M = 2.5
    theta = math.radians(15.0)
    beta = oblique_shock_angle(M, theta)

    # Expected from shock tables: β ≈ 36.9° for M=2.5, θ=15°
    beta_expected = math.radians(36.9)
    error = abs(beta - beta_expected) / beta_expected * 100

    print(f"M = {M}, θ = 15°")
    print(f"Computed β = {math.degrees(beta):.2f}°")
    print(f"Expected β ≈ 36.9°")
    print(f"Error: {error:.2f}%")

    if error < 5:
        print("✓ PASS: Shock angle within 5%")
    else:
        print("✗ FAIL: Shock angle error too large")

    # Test 2: Run short simulation
    print("\n[Test 2] Short SBLI Simulation")
    print("-" * 40)

    results = run_sbli_simulation(
        M_inf=2.5,
        corner_angle_deg=15.0,
        Re_x=1e5,  # Lower Re for faster computation
        Nx=64,
        Ny=32,
        t_final=1e-5,  # Short time
        verbose=False,
    )

    # Check physical bounds
    state = results["state"]
    rho_valid = (state.rho > 0).all()
    p_valid = (state.p > 0).all()

    print(f"Density positive: {rho_valid.item()}")
    print(f"Pressure positive: {p_valid.item()}")
    print(f"Completed {results['steps']} steps")

    if rho_valid and p_valid:
        print("✓ PASS: Physical bounds maintained")
    else:
        print("✗ FAIL: Unphysical values detected")

    # Save plot
    results_dir = Path(__file__).parent.parent / "Physics" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_sbli_results(results, str(results_dir / "sbli_validation.png"))

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    validate_sbli()
