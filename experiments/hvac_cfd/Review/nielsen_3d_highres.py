#!/usr/bin/env python3
"""
Nielsen Benchmark - 1M+ Grid Study
==================================

High-resolution 3D simulation to see if finer grids 
capture better jet spreading and entrainment.

Current best: 18.5% RMS with slip BC
Target: <10% RMS error against Aalborg experimental data.
"""

import json
import sys
import time
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/Project HyperTensor')

import numpy as np
import torch
from ontic.hvac.solver_3d import (
    Solver3DConfig, Solver3D, Solver3DState,
    Inlet3D, Outlet3D
)

# Aalborg experimental data (from Nielsen benchmark)
AALBORG_DATA = {
    "x_H_1.0": {
        "y_H": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "u_Uinlet": np.array([-0.05, -0.08, -0.10, -0.08, -0.02, 0.05, 0.12, 0.22, 0.38, 0.58, 0.85]),
    },
    "x_H_2.0": {
        "y_H": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "u_Uinlet": np.array([-0.03, -0.05, -0.06, -0.05, -0.03, 0.00, 0.05, 0.12, 0.22, 0.35, 0.52]),
    },
}

def compute_rms_error(y_sim, u_sim, y_exp, u_exp):
    """Compute normalized RMS error."""
    u_sim_interp = np.interp(y_exp, y_sim, u_sim)
    u_scale = max(abs(u_exp.max()), abs(u_exp.min()), 1.0)
    rms = np.sqrt(np.mean((u_sim_interp - u_exp) ** 2)) / u_scale
    return rms


def extract_profile_3d(solver, state, x_position):
    """Extract vertical velocity profile at x position (centerline y)."""
    cfg = solver.config
    
    i = int(x_position / cfg.length * (cfg.nx - 1))
    i = max(0, min(i, cfg.nx - 1))
    
    j_center = cfg.ny // 2
    
    z = np.linspace(0, cfg.height, cfg.nz)
    u_profile = state.u[i, j_center, :].cpu().numpy()
    
    return z, u_profile


def run_3d_benchmark(nx=64, ny=32, nz=32, max_iter=2000, enable_les=False):
    """Run 3D Nielsen benchmark."""
    
    # Nielsen room geometry
    L = 9.0   # Length (streamwise)
    W = 1.0   # Width (spanwise)
    H = 3.0   # Height
    
    # Inlet: ceiling slot
    inlet_height = 0.168  # m
    U_in = 0.455          # m/s
    outlet_height = 0.48  # m
    outlet_width = 0.6    # m
    
    nu = 1.5e-5  # Air at 20°C
    
    inlet = Inlet3D(
        x=0.0,
        y=W / 2,
        z=H,
        width=W,
        height=inlet_height * 2,
        velocity=U_in,
        T=16.0,
        direction='x+',
    )
    
    outlet = Outlet3D(
        x=L,
        y=W / 2,
        z=outlet_height / 2,
        width=outlet_width,
        height=outlet_height,
    )
    
    config = Solver3DConfig(
        length=L,
        width=W,
        height=H,
        nx=nx,
        ny=ny,
        nz=nz,
        nu=nu,
        inlets=[inlet],
        outlets=[outlet],
        max_iterations=max_iter,
        convergence_tol=1e-5,
        dt_safety=0.18,  # Slightly more conservative for high-res
        pressure_solver='dct',
        pressure_iterations=500,
        pressure_omega=1.6,
        pressure_tol=1e-6,
        advection_scheme='tvd',
        enable_buoyancy=False,
        enable_turbulence=enable_les,
        turbulence_model='smagorinsky' if enable_les else 'laminar',
        C_s=0.10,  # Lower Smagorinsky - less dissipative
        verbose=True,
    )
    
    solver = Solver3D(config)
    state = solver.create_initial_state()
    
    print(f"\n3D Nielsen: {nx}×{ny}×{nz} = {nx*ny*nz:,} cells, Re=5096")
    if enable_les:
        print(f"  LES: Smagorinsky C_s=0.10")
    else:
        print(f"  Laminar (Direct Numerical Simulation)")
    
    t0 = time.time()
    print("\nSolving...")
    state = solver.solve(state)
    elapsed = time.time() - t0
    
    # Get info from state
    iterations = len(state.residual_history)
    converged = state.converged
    max_velocity = state.velocity_magnitude.max().item()
    
    print(f"\nSolution time: {elapsed:.1f}s")
    print(f"Iterations: {iterations}")
    print(f"Converged: {converged}")
    print(f"Max velocity: {max_velocity:.3f} m/s")
    
    # Extract profiles
    H = config.height
    U_in = 0.455
    
    # x/H = 1.0 -> x = 3.0 m
    z1, u1 = extract_profile_3d(solver, state, x_position=H * 1.0)
    z1_H = z1 / H
    u1_norm = u1 / U_in
    
    # x/H = 2.0 -> x = 6.0 m
    z2, u2 = extract_profile_3d(solver, state, x_position=H * 2.0)
    z2_H = z2 / H
    u2_norm = u2 / U_in
    
    # Compute RMS errors
    rms1 = compute_rms_error(z1_H, u1_norm, AALBORG_DATA["x_H_1.0"]["y_H"], AALBORG_DATA["x_H_1.0"]["u_Uinlet"])
    rms2 = compute_rms_error(z2_H, u2_norm, AALBORG_DATA["x_H_2.0"]["y_H"], AALBORG_DATA["x_H_2.0"]["u_Uinlet"])
    rms_avg = (rms1 + rms2) / 2
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"RMS error @ x/H=1.0: {rms1*100:.1f}%")
    print(f"RMS error @ x/H=2.0: {rms2*100:.1f}%")
    print(f"RMS average:         {rms_avg*100:.1f}%")
    print(f"Target: <10%")
    print(f"Status: {'PASS ✓' if rms_avg < 0.10 else 'FAIL'}")
    
    print(f"\nProfile @ x/H=1.0 (z/H vs u/U_in):")
    print(f"  Exp:  {np.round(AALBORG_DATA['x_H_1.0']['u_Uinlet'], 2)}")
    sim_at_exp = np.interp(AALBORG_DATA["x_H_1.0"]["y_H"], z1_H, u1_norm)
    print(f"  Sim:  {np.round(sim_at_exp, 2)}")
    
    print(f"\nProfile @ x/H=2.0 (z/H vs u/U_in):")
    print(f"  Exp:  {np.round(AALBORG_DATA['x_H_2.0']['u_Uinlet'], 2)}")
    sim_at_exp2 = np.interp(AALBORG_DATA["x_H_2.0"]["y_H"], z2_H, u2_norm)
    print(f"  Sim:  {np.round(sim_at_exp2, 2)}")
    
    return {
        'nx': nx, 'ny': ny, 'nz': nz,
        'cells': nx * ny * nz,
        'rms_1': rms1, 'rms_2': rms2, 'rms_avg': rms_avg,
        'converged': converged,
        'les': enable_les,
        'time': elapsed,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("NIELSEN 3D BENCHMARK - HIGH RESOLUTION STUDY")
    print("=" * 70)
    print("\nTesting how physics responds to increasing grid resolution...")
    print("Using SLIP BC at ceiling for wall jet attachment.")
    
    results = []
    
    # TEST 1: ~300k cells (baseline)
    print("\n" + "=" * 70)
    print("TEST 1: Laminar 128×24×96 (~295k cells) - Baseline")
    print("=" * 70)
    r = run_3d_benchmark(nx=128, ny=24, nz=96, max_iter=2000, enable_les=False)
    results.append(r)
    
    # TEST 2: ~600k cells (2x baseline)
    print("\n" + "=" * 70)
    print("TEST 2: Laminar 180×24×128 (~550k cells)")
    print("=" * 70)
    r = run_3d_benchmark(nx=180, ny=24, nz=128, max_iter=2500, enable_les=False)
    results.append(r)
    
    # TEST 3: ~1M cells
    print("\n" + "=" * 70)
    print("TEST 3: Laminar 256×24×160 (~980k cells) - 1M Target")
    print("=" * 70)
    r = run_3d_benchmark(nx=256, ny=24, nz=160, max_iter=3000, enable_les=False)
    results.append(r)
    
    # TEST 4: ~1.5M cells with LES
    print("\n" + "=" * 70)
    print("TEST 4: LES 256×32×160 (~1.3M cells) - LES on fine grid")
    print("=" * 70)
    r = run_3d_benchmark(nx=256, ny=32, nz=160, max_iter=3000, enable_les=True)
    results.append(r)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - Grid Convergence Study")
    print("=" * 70)
    print(f"{'Grid':<25} {'Cells':>12} {'RMS %':>10} {'Time':>10}")
    print("-" * 60)
    for r in results:
        grid = f"{r['nx']}×{r['ny']}×{r['nz']}"
        mode = "LES" if r['les'] else "Lam"
        status = "PASS ✓" if r['rms_avg'] < 0.10 else ""
        print(f"{grid} ({mode}){'':<10} {r['cells']:>10,} {r['rms_avg']*100:>8.1f}% {r['time']:>8.1f}s {status}")
    
    print("\n" + "-" * 60)
    print(f"Target: <10% RMS error")
    best = min(results, key=lambda x: x['rms_avg'])
    print(f"Best result: {best['nx']}×{best['ny']}×{best['nz']} with {best['rms_avg']*100:.1f}% RMS")
    
    # Save results
    with open("HVAC_CFD/nielsen_highres_result.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to HVAC_CFD/nielsen_highres_result.json")
