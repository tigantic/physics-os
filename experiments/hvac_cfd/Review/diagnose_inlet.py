#!/usr/bin/env python3
"""
Diagnose inlet jet behavior in 3D Nielsen benchmark.
"""

import sys
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/Project The Ontic Engine')

import numpy as np
import torch
from ontic.hvac.solver_3d import (
    Solver3DConfig, Solver3D, Solver3DState,
    Inlet3D, Outlet3D
)


def run_diagnostic():
    """Run diagnostic to see jet behavior near inlet."""
    
    # Coarse grid for fast debugging
    nx, ny, nz = 64, 16, 32
    
    L = 9.0   # Length
    W = 1.0   # Width
    H = 3.0   # Height
    
    inlet_height = 0.168
    inlet_width = 0.5
    U_in = 0.455
    nu = 1.5e-5
    
    inlet = Inlet3D(
        x=0.0,
        y=W / 2,
        z=H,                  # AT ceiling level
        width=W,
        height=inlet_height * 2,  # Extend downward to capture enough cells
        velocity=U_in,
        T=16.0,
        direction='x+',
    )
    
    outlet = Outlet3D(
        x=L,
        y=W / 2,
        z=0.24,
        width=0.6,
        height=0.48,
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
        max_iterations=100,
        convergence_tol=1e-5,
        dt_safety=0.25,
        pressure_solver='dct',
        pressure_iterations=200,
        pressure_omega=1.5,
        enable_buoyancy=False,
        enable_turbulence=False,
        verbose=False,
    )
    
    solver = Solver3D(config)
    state = solver.create_initial_state()
    
    print("=== INITIAL STATE ===")
    print(f"Inlet mask sum: {solver.inlet_mask.sum().item()}")
    print(f"Inlet mask at z=-1: {solver.inlet_mask[:, :, -1].sum().item()}")
    print(f"Inlet mask at z=-2: {solver.inlet_mask[:, :, -2].sum().item()}")
    
    # Show inlet velocity at i=0
    j_mid = ny // 2
    print(f"\nVelocity at i=0, j={j_mid} (inlet plane):")
    print(f"  u[0,{j_mid},:] = {state.u[0, j_mid, :].cpu().numpy().round(3)}")
    print(f"  v[0,{j_mid},:] = {state.v[0, j_mid, :].cpu().numpy().round(3)}")
    print(f"  w[0,{j_mid},:] = {state.w[0, j_mid, :].cpu().numpy().round(3)}")
    
    # Run a few steps
    print("\n=== STEPPING ===")
    for i in range(20):
        state = solver.step(state)
        state = solver.apply_velocity_bc(state)
        
    print(f"\nAfter 20 steps:")
    print(f"  u[0,{j_mid},:] = {state.u[0, j_mid, :].cpu().numpy().round(3)}")
    print(f"  u_max = {state.u.max().item():.4f}")
    
    # Check inlet-adjacent cells
    print(f"\nVelocity at i=1 (first interior cell):")
    print(f"  u[1,{j_mid},:] = {state.u[1, j_mid, :].cpu().numpy().round(3)}")
    print(f"  u[2,{j_mid},:] = {state.u[2, j_mid, :].cpu().numpy().round(3)}")
    
    # Check divergence
    div = solver.compute_divergence_3d(state.u, state.v, state.w)
    print(f"\nDivergence at i=1 (first interior):")
    print(f"  div[1,{j_mid},:] = {div[1, j_mid, :].cpu().numpy().round(4)}")
    print(f"  max|div| = {torch.abs(div).max().item():.4f}")
    
    # Check pressure at inlet region
    print(f"\nPressure at inlet region:")
    print(f"  p[0,{j_mid},:] = {state.p[0, j_mid, :].cpu().numpy().round(4)}")
    print(f"  p[1,{j_mid},:] = {state.p[1, j_mid, :].cpu().numpy().round(4)}")
    print(f"  p[2,{j_mid},:] = {state.p[2, j_mid, :].cpu().numpy().round(4)}")
    
    # Profile at x/H = 0.5 (just past inlet)
    i_05 = int(0.5 * H / L * nx)  # x = 1.5m (x/H = 0.5)
    i_10 = int(1.0 * H / L * nx)  # x = 3.0m (x/H = 1.0)
    
    print(f"\n=== JET PROFILES ===")
    print(f"Profile at x/H=0.5 (i={i_05}):")
    z = np.linspace(0, 1, nz)
    u_05 = state.u[i_05, j_mid, :].cpu().numpy() / U_in
    print(f"  z/H: {z.round(2)}")
    print(f"  u/U: {u_05.round(3)}")
    
    print(f"\nProfile at x/H=1.0 (i={i_10}):")
    u_10 = state.u[i_10, j_mid, :].cpu().numpy() / U_in
    print(f"  u/U: {u_10.round(3)}")
    
    # Check where max velocity is in z
    k_max_05 = u_05.argmax()
    k_max_10 = u_10.argmax()
    print(f"\nMax velocity locations:")
    print(f"  x/H=0.5: k={k_max_05}, z/H={z[k_max_05]:.2f}, u/U={u_05[k_max_05]:.3f}")
    print(f"  x/H=1.0: k={k_max_10}, z/H={z[k_max_10]:.2f}, u/U={u_10[k_max_10]:.3f}")
    
    # The experimental data shows peak at z/H=1.0 (ceiling)
    # If our peak is lower, the jet is detaching
    
    print("\n=== INLET CELLS DETAIL ===")
    inlet_k = solver.inlet_mask[0, j_mid, :]
    print(f"Inlet mask at i=0, j={j_mid}: {inlet_k.cpu().numpy()}")
    print(f"Number of inlet cells in z: {inlet_k.sum().item()}")


if __name__ == "__main__":
    run_diagnostic()
