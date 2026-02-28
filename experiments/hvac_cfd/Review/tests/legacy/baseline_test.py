#!/usr/bin/env python3
"""
Nielsen Benchmark - Baseline Test
==================================

Tests TVD solver to get baseline RMS error.
Writes output to a log file for reliable capture.
"""

import sys
import time
from pathlib import Path

# Get project root dynamically
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from ontic.hvac.projection_solver import ProjectionConfig, ProjectionSolver

# Log file path (relative to this script)
LOG_FILE = Path(__file__).parent / 'benchmark_output.log'

# Open log file
f = open(LOG_FILE, 'w')

def log(msg):
    print(msg, flush=True)
    f.write(msg + '\n')
    f.flush()

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

log("=" * 60)
log("NIELSEN BENCHMARK - BASELINE TVD")
log("=" * 60)

# Solver config - use small grid for fast test
config = ProjectionConfig(
    nx=128, ny=64,
    Re=5000,
    max_iterations=1000,
    convergence_tol=1e-5,
    dt_safety=0.25,
    advection_scheme='tvd',
    tvd_limiter='van_leer',
    alpha_u=0.8,
    alpha_p=0.4,
    pressure_iterations=100,
    verbose=True,  # Use built-in logging
    diag_interval=100,
)

log(f"Grid: {config.nx}x{config.ny}")
log(f"Re = {config.Re}")
log(f"Advection: {config.advection_scheme}")
log("")

solver = ProjectionSolver(config)
start_time = time.time()

# Use solve() directly - it computes residuals correctly
state = solver.solve()

solve_time = time.time() - start_time
log(f"\nSolve time: {solve_time:.1f}s")
log(f"Final residual: {state.residual_history[-1]:.2e}")

# Extract profiles and compute RMS
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

log("")
log("=" * 60)
log("VALIDATION RESULTS")
log("-" * 40)
log(f"RMS Error at x/H = 1.0: {rms_1*100:.1f}%")
log(f"RMS Error at x/H = 2.0: {rms_2*100:.1f}%")
log(f"Average RMS Error:      {rms_avg*100:.1f}%")
log(f"Target:                 <10.0%")
log("")

if rms_avg < 0.10:
    log("STATUS: PASS")
else:
    log("STATUS: FAIL")
log("=" * 60)

f.close()
print(f"\nOutput written to: {LOG_FILE}")
