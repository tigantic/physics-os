#!/usr/bin/env python3
"""Quick test of deferred correction solver."""

import sys
import time
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/Project HyperTensor')

from ontic.hvac.projection_solver import ProjectionConfig, ProjectionSolver

config = ProjectionConfig(
    nx=128, ny=64, Re=5000,
    max_iterations=500, convergence_tol=1e-5,
    dt_safety=0.25, advection_scheme='deferred',
    alpha_u=0.8, alpha_p=0.4,
    verbose=True, diag_interval=100,
)

print("Starting solver...")
solver = ProjectionSolver(config)
start = time.time()
state = solver.solve()
print(f"\nTime: {time.time()-start:.1f}s")
print(f"Final residual: {state.residual_history[-1]:.2e}")
print(f"Converged: {state.converged}")
