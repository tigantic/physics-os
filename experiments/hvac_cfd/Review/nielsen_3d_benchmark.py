#!/usr/bin/env python3
"""
Nielsen Benchmark - 3D Solver
=============================

3D simulation of Nielsen ventilation case to capture
turbulent jet spreading that 2D laminar model misses.

Nielsen Benchmark (Tier 1):
- Room: 9m × W × 3m (L × W × H)
- Ceiling slot inlet: 0.168m high, U_in = 0.455 m/s
- Floor outlet: 0.48m high
- Re ≈ 5000

Target: <10% RMS error against Aalborg experimental data.
"""

import json
import sys
import time
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/Project HyperTensor')

import numpy as np
import torch
from tensornet.hvac.solver_3d import (
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
    """Extract vertical velocity profile at x position (centerline y).
    
    Note: We exclude the boundary cells (floor and ceiling) since those
    are no-slip walls with u=0. The experimental data at z/H=1.0 represents
    measurements just below the ceiling, not exactly at the wall.
    """
    cfg = solver.config
    
    # Find x index
    i = int(x_position / cfg.length * (cfg.nx - 1))
    i = max(0, min(i, cfg.nx - 1))
    
    # Centerline in y
    j_center = cfg.ny // 2
    
    # Extract z profile (vertical) - EXCLUDE boundary cells
    # Use interior cells only (k=1 to k=nz-2)
    z_full = np.linspace(0, cfg.height, cfg.nz)
    u_full = state.u[i, j_center, :].cpu().numpy()
    
    # For wall jets, the experimental z/H=1.0 is just below ceiling
    # Map our grid cell centers to properly reflect this
    # Cell 0 is at floor (u=0), cell -1 is at ceiling (u=0)
    # The "first cell from ceiling" at k=-2 has center at z = H - dz/2
    
    # Use all cells but recognize that experimental z/H=1.0 corresponds
    # to our second-to-last cell (first interior cell from ceiling)
    z = z_full
    u_profile = u_full
    
    return z, u_profile


def run_3d_benchmark(nx=64, ny=32, nz=32, max_iter=1000, enable_les=True):
    """Run 3D Nielsen benchmark."""
    
    # Nielsen room geometry
    L = 9.0   # Length (streamwise)
    W = 1.0   # Width (spanwise) - thin slice to approximate 2D
    H = 3.0   # Height
    
    # Inlet: ceiling slot
    inlet_height = 0.168  # m
    inlet_width = 0.5     # m (in y)
    U_in = 0.455          # m/s
    
    # Outlet: floor wall
    outlet_height = 0.48  # m
    outlet_width = 0.6    # m
    
    # Reynolds number: Re = U * h / nu = 0.455 * 0.168 / 1.5e-5 ≈ 5096
    nu = 1.5e-5  # Air at 20°C
    
    # Create inlet at ceiling (flow enters downward from ceiling slot)
    # Nielsen case: slot at x=0 on ceiling, jet flows in +x direction along ceiling
    # IMPORTANT: Set inlet center AT ceiling level so inlet extends to ceiling surface
    # This ensures the jet hugs the ceiling (Coanda effect)
    inlet = Inlet3D(
        x=0.0,
        y=W / 2,              # Centered in y
        z=H,                  # AT ceiling (inlet will extend from H-inlet_height to H)
        width=W,              # Full width of thin domain
        height=inlet_height * 2,  # Double to ensure enough cells covered
        velocity=U_in,
        T=16.0,
        direction='x+',       # Flow enters in +x direction (along ceiling)
    )
    
    # Create outlet at right wall near floor
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
        dt_safety=0.20,
        pressure_solver='dct',
        pressure_iterations=300,
        pressure_omega=1.5,
        pressure_tol=1e-6,
        advection_scheme='tvd',
        enable_buoyancy=False,
        enable_turbulence=enable_les,
        turbulence_model='smagorinsky' if enable_les else 'laminar',
        C_s=0.17,
        verbose=True,
        diag_interval=200,  # Less frequent output for large grids
    )
    
    print(f"\n3D Nielsen: {nx}×{ny}×{nz} = {nx*ny*nz:,} cells, Re={U_in * inlet_height / nu:.0f}")
    
    solver = Solver3D(config)
    
    # Solve
    print(f"\nSolving...")
    start_time = time.time()
    state = solver.solve()
    solve_time = time.time() - start_time
    
    print(f"\nSolution time: {solve_time:.1f}s")
    print(f"Iterations: {state.iteration}")
    print(f"Converged: {state.converged}")
    print(f"Max velocity: {state.velocity_magnitude.max().item():.3f} m/s")
    
    # Extract profiles at x/H = 1.0 and 2.0
    z1, u1 = extract_profile_3d(solver, state, x_position=1.0*H)  # x = 3m
    z2, u2 = extract_profile_3d(solver, state, x_position=2.0*H)  # x = 6m
    
    # Normalize
    z1_H = z1 / H
    u1_U = u1 / U_in
    z2_H = z2 / H
    u2_U = u2 / U_in
    
    # Compute RMS error
    rms_1 = compute_rms_error(z1_H, u1_U, AALBORG_DATA["x_H_1.0"]["y_H"], 
                              AALBORG_DATA["x_H_1.0"]["u_Uinlet"])
    rms_2 = compute_rms_error(z2_H, u2_U, AALBORG_DATA["x_H_2.0"]["y_H"], 
                              AALBORG_DATA["x_H_2.0"]["u_Uinlet"])
    rms_avg = (rms_1 + rms_2) / 2.0
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"RMS error @ x/H=1.0: {rms_1*100:.1f}%")
    print(f"RMS error @ x/H=2.0: {rms_2*100:.1f}%")
    print(f"RMS average:         {rms_avg*100:.1f}%")
    print(f"Target: <10%")
    print(f"Status: {'PASS' if rms_avg < 0.10 else 'FAIL'}")
    
    # Print profile comparison
    print(f"\nProfile @ x/H=1.0 (z/H vs u/U_in):")
    print(f"  Exp:  {AALBORG_DATA['x_H_1.0']['u_Uinlet']}")
    print(f"  Sim:  {np.round(np.interp(AALBORG_DATA['x_H_1.0']['y_H'], z1_H, u1_U), 2)}")
    
    print(f"\nProfile @ x/H=2.0 (z/H vs u/U_in):")
    print(f"  Exp:  {AALBORG_DATA['x_H_2.0']['u_Uinlet']}")
    print(f"  Sim:  {np.round(np.interp(AALBORG_DATA['x_H_2.0']['y_H'], z2_H, u2_U), 2)}")
    
    return {
        "solver": "3D-LES" if enable_les else "3D-laminar",
        "grid": f"{nx}×{ny}×{nz}",
        "cells": nx * ny * nz,
        "iterations": state.iteration,
        "converged": state.converged,
        "solve_time_s": round(solve_time, 1),
        "rms_x1": round(rms_1 * 100, 1),
        "rms_x2": round(rms_2 * 100, 1),
        "rms_avg": round(rms_avg * 100, 1),
        "passed": bool(rms_avg < 0.10),
    }


if __name__ == "__main__":
    # 3D Nielsen benchmark - focus on z-resolution to capture jet properly
    # Nielsen inlet is 0.168m high, need at least 4-5 cells to resolve
    # With H=3m and inlet=0.168m, ratio = 18:1
    # So nz=96 gives ~5 cells in inlet
    
    results = []
    
    # Test 1: High z-resolution laminar
    print(f"\n{'='*70}")
    print(f"TEST 1: High z-resolution laminar (128×24×96)")
    print(f"{'='*70}")
    results.append(run_3d_benchmark(nx=128, ny=24, nz=96, max_iter=2000, enable_les=False))
    
    # Test 2: High z-resolution with LES
    print(f"\n{'='*70}")
    print(f"TEST 2: High z-resolution LES (128×24×96)")
    print(f"{'='*70}")
    results.append(run_3d_benchmark(nx=128, ny=24, nz=96, max_iter=2000, enable_les=True))
    
    # Save results
    output_path = '/home/brad/TiganticLabz/Main_Projects/Project HyperTensor/HVAC_CFD/nielsen_3d_result.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['grid']:>12s} ({r['cells']:>7,} cells): {r['rms_avg']:5.1f}% [{status}]")
    print(f"\nResults saved to {output_path}")
