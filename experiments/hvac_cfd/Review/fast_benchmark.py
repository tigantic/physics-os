#!/usr/bin/env python3
"""
Nielsen Benchmark - Fast Baseline
=================================

Quick 128x64 test to establish baseline RMS error.
Output saved to benchmark_result.json
"""

import json
import sys
import time
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/Project HyperTensor')

import numpy as np
import torch
from ontic.hvac.projection_solver import ProjectionConfig, ProjectionSolver

# Aalborg experimental data
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

def run_benchmark(advection_scheme='tvd', nx=128, ny=64, max_iter=800):
    """Run a single benchmark configuration."""
    
    config = ProjectionConfig(
        nx=nx, ny=ny,
        Re=5000,
        max_iterations=max_iter,
        convergence_tol=1e-5,
        dt_safety=0.25,
        advection_scheme=advection_scheme,
        tvd_limiter='van_leer',
        alpha_u=0.8,
        alpha_p=0.4,
        pressure_iterations=100,
        verbose=False,
        diag_interval=100,
    )
    
    solver = ProjectionSolver(config)
    start_time = time.time()
    state = solver.solve()
    solve_time = time.time() - start_time
    
    # Extract profiles
    H = config.height
    U_in = config.inlet_velocity
    
    y1, u1, _ = solver.extract_profile(state, x_position=3.0)
    y2, u2, _ = solver.extract_profile(state, x_position=6.0)
    
    y1_H = y1.numpy() / H
    u1_U = u1.numpy() / U_in
    y2_H = y2.numpy() / H
    u2_U = u2.numpy() / U_in
    
    rms_1 = compute_rms_error(y1_H, u1_U, AALBORG_DATA["x_H_1.0"]["y_H"], AALBORG_DATA["x_H_1.0"]["u_Uinlet"])
    rms_2 = compute_rms_error(y2_H, u2_U, AALBORG_DATA["x_H_2.0"]["y_H"], AALBORG_DATA["x_H_2.0"]["u_Uinlet"])
    rms_avg = (rms_1 + rms_2) / 2.0
    
    return {
        "scheme": advection_scheme,
        "grid": f"{nx}x{ny}",
        "iterations": state.iteration,
        "converged": state.converged,
        "solve_time_s": round(solve_time, 1),
        "rms_x1": round(rms_1 * 100, 1),
        "rms_x2": round(rms_2 * 100, 1),
        "rms_avg": round(rms_avg * 100, 1),
        "passed": bool(rms_avg < 0.10),
    }

if __name__ == "__main__":
    results = []
    
    print("Testing TVD baseline 128x64...")
    results.append(run_benchmark('tvd', 128, 64, 800))
    print(f"  TVD 128x64: {results[-1]['rms_avg']}%")
    
    print("Testing TVD 256x128...")
    results.append(run_benchmark('tvd', 256, 128, 1500))
    print(f"  TVD 256x128: {results[-1]['rms_avg']}%")
    
    print("Testing TVD 384x192 (higher resolution)...")
    results.append(run_benchmark('tvd', 384, 192, 2000))
    print(f"  TVD 384x192: {results[-1]['rms_avg']}%")
    
    # Save results
    output_path = '/home/brad/TiganticLabz/Main_Projects/Project HyperTensor/HVAC_CFD/benchmark_result.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\nSummary:")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['scheme']:12s} @ {r['grid']:>10s}: {r['rms_avg']:5.1f}% ({status})")
