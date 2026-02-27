#!/usr/bin/env python3
"""
Tier 1 Nielsen Benchmark — Skew-Symmetric Test
===============================================

Tests the projection solver with skew-symmetric advection (energy-conserving).
This scheme has zero numerical diffusion AND is stable because it conserves
kinetic energy exactly.

Reference: Morinishi et al. (1998), JCP 143, 90-124
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensornet.hvac.projection_solver import ProjectionConfig, ProjectionSolver


# ============================================================================
# AALBORG EXPERIMENTAL DATA
# ============================================================================

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


def run_central_benchmark():
    """Run benchmark with TVD Van Leer at standard resolution."""
    
    print("=" * 70)
    print("TIER 1: Nielsen Benchmark — TVD VAN LEER (256×128)")
    print("=" * 70)
    
    # Create solver config - stable TVD
    config = ProjectionConfig(
        nx=256,
        ny=128,
        Re=5000,
        max_iterations=3000,
        convergence_tol=1e-5,
        dt_safety=0.25,  # Lower CFL for stability
        advection_scheme='tvd',
        tvd_limiter='van_leer',  # More stable than superbee
        alpha_u=0.7,  # More under-relaxation
        alpha_p=0.3,
        pressure_iterations=200,
        pressure_tol=1e-6,
        pressure_omega=1.7,
        verbose=True,
        diag_interval=200,
    )
    
    print(f"\nGrid: {config.nx}×{config.ny}")
    print(f"Re = {config.Re:.0f}")
    print(f"Advection: {config.advection_scheme}")
    print(f"dt_safety = {config.dt_safety}")
    print()
    
    # Run solver
    solver = ProjectionSolver(config)
    start_time = time.perf_counter()
    state = solver.solve()
    solve_time = time.perf_counter() - start_time
    
    print(f"\n{'='*70}")
    print(f"Solver completed in {solve_time:.1f} seconds")
    print(f"Converged: {state.converged}")
    print(f"Iterations: {state.iteration}")
    
    # Extract profiles
    H = config.height
    U_in = config.inlet_velocity
    
    y1, u1, _ = solver.extract_profile(state, x_position=3.0)  # x/H = 1.0
    y2, u2, _ = solver.extract_profile(state, x_position=6.0)  # x/H = 2.0
    
    y1_H = y1.numpy() / H
    u1_U = u1.numpy() / U_in
    y2_H = y2.numpy() / H
    u2_U = u2.numpy() / U_in
    
    # Compute RMS errors
    rms_1 = compute_rms_error(y1_H, u1_U, AALBORG_DATA["x_H_1.0"]["y_H"], AALBORG_DATA["x_H_1.0"]["u_Uinlet"])
    rms_2 = compute_rms_error(y2_H, u2_U, AALBORG_DATA["x_H_2.0"]["y_H"], AALBORG_DATA["x_H_2.0"]["u_Uinlet"])
    rms_avg = (rms_1 + rms_2) / 2.0
    
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print("-" * 40)
    print(f"RMS Error at x/H = 1.0: {rms_1*100:.1f}%")
    print(f"RMS Error at x/H = 2.0: {rms_2*100:.1f}%")
    print(f"Average RMS Error:      {rms_avg*100:.1f}%")
    print(f"Target:                 <10.0%")
    print()
    
    passed = rms_avg < 0.10
    if passed:
        print("🎉 STATUS: ✓ PASS")
    else:
        print("❌ STATUS: ✗ FAIL")
    print(f"{'='*70}")
    
    return passed, rms_avg, state, solver


if __name__ == "__main__":
    passed, rms, state, solver = run_central_benchmark()
    exit(0 if passed else 1)
