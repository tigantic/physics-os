#!/usr/bin/env python3
"""
Run Nielsen Benchmark with Time-Averaging
==========================================

Uses time-averaging on the oscillating solution to get a pseudo-steady state.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ontic.hvac.projection_solver import ProjectionConfig, ProjectionSolver, ProjectionState


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


def main():
    """Run Nielsen benchmark with time-averaging."""
    
    print("=" * 60)
    print("NIELSEN BENCHMARK — TIME-AVERAGED SOLUTION")
    print("=" * 60)
    
    # Create solver config 
    config = ProjectionConfig(
        nx=256,
        ny=128,
        Re=5000,
        max_iterations=3000,  # Enough to settle
        convergence_tol=1e-5,
        dt_safety=0.25,
        advection_scheme='deferred',  # TVD + 15% central
        tvd_limiter='van_leer',
        alpha_u=0.8,
        alpha_p=0.4,
        pressure_iterations=200,
        pressure_tol=1e-6,
        pressure_omega=1.7,
        verbose=True,
        diag_interval=500,
    )
    
    print(f"\nGrid: {config.nx}×{config.ny}")
    print(f"Re = {config.Re:.0f}")
    print(f"Phase 1: Run to quasi-steady state")
    print()
    
    solver = ProjectionSolver(config)
    start_time = time.perf_counter()
    
    # Phase 1: Run to quasi-steady
    state = solver.solve()
    
    # Phase 2: Time-average over 1000 more steps
    print("\n" + "-" * 60)
    print("Phase 2: Time-averaging over 1000 steps")
    print("-" * 60)
    
    u_avg = state.u.clone()
    v_avg = state.v.clone()
    n_samples = 1
    
    for i in range(1000):
        state = solver.step(state)
        u_avg += state.u
        v_avg += state.v
        n_samples += 1
        
        if i % 200 == 0:
            print(f"  Averaging step {i}/1000")
    
    u_avg /= n_samples
    v_avg /= n_samples
    
    # Create averaged state
    avg_state = ProjectionState(
        u=u_avg,
        v=v_avg,
        p=state.p,
        iteration=state.iteration,
        converged=True,
        residual_history=state.residual_history,
    )
    
    solve_time = time.perf_counter() - start_time
    print(f"\nTotal time: {solve_time:.1f}s")
    
    # Extract profiles
    H = config.height
    U_in = config.inlet_velocity
    
    y1, u1, _ = solver.extract_profile(avg_state, x_position=3.0)  # x/H = 1.0
    y2, u2, _ = solver.extract_profile(avg_state, x_position=6.0)  # x/H = 2.0
    
    y1_H = y1.numpy() / H
    u1_U = u1.numpy() / U_in
    y2_H = y2.numpy() / H
    u2_U = u2.numpy() / U_in
    
    # Compute RMS errors
    rms_1 = compute_rms_error(y1_H, u1_U, AALBORG_DATA["x_H_1.0"]["y_H"], AALBORG_DATA["x_H_1.0"]["u_Uinlet"])
    rms_2 = compute_rms_error(y2_H, u2_U, AALBORG_DATA["x_H_2.0"]["y_H"], AALBORG_DATA["x_H_2.0"]["u_Uinlet"])
    rms_avg = (rms_1 + rms_2) / 2.0
    
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS — TIME AVERAGED")
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
    print(f"{'='*60}")
    
    return passed


if __name__ == "__main__":
    passed = main()
    exit(0 if passed else 1)
