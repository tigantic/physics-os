#!/usr/bin/env python3
"""
Conference Room B - Ventilation Analysis
=========================================

Client: James Chen, Facilities Engineering
Case: Air quality complaints - "stuffy" near back wall

Uses NS2DSolver (spectral incompressible Navier-Stokes)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import sys
import os

sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/physics-os')
os.makedirs('/home/brad/TiganticLabz/Main_Projects/physics-os/results', exist_ok=True)

from ontic.cfd.ns_2d import NS2DSolver, NSState


def run_conference_room_analysis():
    """
    Nielsen/IEA Annex 20 benchmark case.
    
    Room: 9m × 3m (L/H = 3)
    Inlet: 168mm ceiling slot, 0.455 m/s
    Outlet: 480mm floor grille
    Re_inlet ≈ 5100
    """
    
    print("=" * 60)
    print("  CONFERENCE ROOM B - VENTILATION ANALYSIS")
    print("  Client: James Chen, Facilities Engineering")
    print("=" * 60)
    print()
    
    # Room parameters
    L, H = 9.0, 3.0          # meters
    h_inlet = 0.168          # 168mm slot
    u_inlet = 0.455          # m/s
    h_outlet = 0.480         # 480mm grille
    nu = 1.5e-5              # air kinematic viscosity
    
    Re_inlet = u_inlet * h_inlet / nu
    Re_room = u_inlet * H / nu
    
    print(f"Room: {L}m × {H}m (L/H = {L/H:.1f})")
    print(f"Inlet: {h_inlet*1000:.0f}mm slot @ {u_inlet} m/s")
    print(f"Re (inlet): {Re_inlet:.0f}")
    print(f"Re (room): {Re_room:.0f}")
    print()
    
    # Grid - 64×64 is fast, captures the physics
    Nx, Ny = 128, 64
    
    print(f"Grid: {Nx}×{Ny} = {Nx*Ny:,} cells")
    print()
    
    # Initialize solver
    # Scale viscosity to match Re at this grid resolution
    # Use effective viscosity for stable low-Re simulation
    nu_eff = u_inlet * H / 5000  # Target Re~5000
    
    solver = NS2DSolver(
        Nx=Nx, Ny=Ny,
        Lx=L, Ly=H,
        nu=nu_eff,
        dtype=torch.float64,
        device="cpu"
    )
    
    # Create initial condition with inlet jet
    # Ceiling slot at x=0, y ∈ [H-h_inlet, H]
    u = torch.zeros((Nx, Ny), dtype=torch.float64)
    v = torch.zeros((Nx, Ny), dtype=torch.float64)
    
    # Set inlet profile (Gaussian jet from ceiling slot)
    y_inlet_center = H - h_inlet/2
    sigma = h_inlet / 3
    
    for j in range(Ny):
        y = j * solver.dy
        # Gaussian profile centered on slot
        u[0, j] = u_inlet * np.exp(-((y - y_inlet_center)**2) / (2*sigma**2))
    
    state = NSState(u=u, v=v, t=0.0, step=0)
    
    print("Solving to steady state...")
    print("-" * 40)
    
    # Time stepping - use built-in RK4 step
    dt = 0.002  # Small dt for stability
    t_final = 15.0  # seconds
    n_steps = int(t_final / dt)
    
    import time
    t0 = time.perf_counter()
    
    for step in range(n_steps):
        # Use RK4 + projection (handles incompressibility correctly)
        state, proj = solver.step_rk4(state, dt)
        
        # Re-apply inlet BC
        for j in range(Ny):
            y_pos = j * solver.dy
            state.u[0, j] = u_inlet * np.exp(-((y_pos - y_inlet_center)**2) / (2*sigma**2))
            state.v[0, j] = -0.05 * u_inlet  # slight downward deflection
        
        # Outlet BC (convective outflow)
        state.u[-1, :] = state.u[-2, :]
        state.v[-1, :] = state.v[-2, :]
        
        # Wall BCs (no-slip)
        state.u[:, 0] = 0   # floor
        state.v[:, 0] = 0
        state.u[1:, -1] = 0 # ceiling (except inlet)
        state.v[:, -1] = 0
        
        if step % 1000 == 0:
            u_max = torch.max(torch.abs(state.u)).item()
            print(f"  Step {step:5d}/{n_steps}: t={state.t:.2f}s, u_max={u_max:.4f} m/s")
            if step > 0 and u_max < 1e-6:
                print("    Converged!")
                break
    
    wall_time = time.perf_counter() - t0
    print(f"\nSolved in {wall_time:.2f}s")
    
    # Extract results
    u_np = state.u.numpy()
    v_np = state.v.numpy()
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, H, Ny)
    
    print()
    print("=" * 60)
    print("  VELOCITY PROFILES AT COMPLAINT LOCATIONS")
    print("=" * 60)
    
    # Profiles at x=3m and x=6m
    results = {}
    for x_pos in [3.0, 6.0]:
        i = int(x_pos / solver.dx)
        u_profile = u_np[i, :]
        v_profile = v_np[i, :]
        
        results[f"x={x_pos}m"] = {
            "y": y.tolist(),
            "u": u_profile.tolist(),
            "v": v_profile.tolist()
        }
        
        print(f"\nAt x = {x_pos}m from inlet:")
        print(f"  {'Height (m)':<12} {'u (m/s)':<12} {'v (m/s)':<12}")
        print(f"  {'-'*36}")
        
        for height in [0.5, 1.0, 1.5, 2.0, 2.5]:
            j = int(height / solver.dy)
            if j < Ny:
                print(f"  {height:<12.1f} {u_profile[j]:<12.4f} {v_profile[j]:<12.4f}")
    
    # Breathing zone analysis
    print()
    print("=" * 60)
    print("  FINDINGS FOR JAMES")
    print("=" * 60)
    
    # x=6m breathing zone (1.0-1.8m)
    i6 = int(6.0 / solver.dx)
    j_low = int(1.0 / solver.dy)
    j_high = int(1.8 / solver.dy)
    
    u_breathing_6m = np.mean(np.abs(u_np[i6, j_low:j_high]))
    
    # x=3m for comparison
    i3 = int(3.0 / solver.dx)
    u_breathing_3m = np.mean(np.abs(u_np[i3, j_low:j_high]))
    
    print(f"\n  Breathing Zone (1.0-1.8m height):")
    print(f"    At x=3m: {u_breathing_3m:.4f} m/s")
    print(f"    At x=6m: {u_breathing_6m:.4f} m/s")
    print(f"    ASHRAE minimum: 0.10 m/s")
    
    if u_breathing_6m < 0.1:
        print(f"\n  ⚠️  PROBLEM CONFIRMED:")
        print(f"      Velocity at x=6m ({u_breathing_6m:.3f} m/s) is below ASHRAE minimum")
        print(f"      This explains the 'stuffy' complaints from occupants")
    
    # Jet trajectory
    jet_core_3m = y[np.argmax(u_np[i3, :])]
    jet_core_6m = y[np.argmax(u_np[i6, :])]
    
    print(f"\n  Jet Trajectory:")
    print(f"    Core height at x=3m: {jet_core_3m:.2f}m")
    print(f"    Core height at x=6m: {jet_core_6m:.2f}m")
    
    if jet_core_6m < 2.0:
        print(f"    → Jet drops into occupied zone before reaching back wall")
    
    print(f"\n  RECOMMENDATIONS:")
    print(f"    1. Add mid-room diffuser at x=4.5m")
    print(f"    2. Increase inlet velocity to 0.7 m/s")  
    print(f"    3. Or: replace slot with high-induction diffuser")
    
    # Visualization
    print()
    print("  Generating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    vel_mag = np.sqrt(u_np**2 + v_np**2)
    
    # Velocity magnitude
    ax1 = axes[0, 0]
    c1 = ax1.contourf(X, Y, vel_mag, levels=20, cmap='viridis')
    plt.colorbar(c1, ax=ax1, label='|V| (m/s)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Velocity Magnitude')
    ax1.axvline(x=3, color='r', linestyle='--', alpha=0.7)
    ax1.axvline(x=6, color='r', linestyle='--', alpha=0.7)
    
    # Streamlines
    skip = 2
    ax1.streamplot(X[::skip, ::skip].T, Y[::skip, ::skip].T,
                   u_np[::skip, ::skip].T, v_np[::skip, ::skip].T,
                   color='white', linewidth=0.5, density=1.2)
    
    # Horizontal velocity profiles
    ax2 = axes[0, 1]
    ax2.plot(u_np[i3, :], y, 'b-', linewidth=2, label='x=3m')
    ax2.plot(u_np[i6, :], y, 'r-', linewidth=2, label='x=6m')
    ax2.axvline(x=0.1, color='g', linestyle=':', label='ASHRAE min')
    ax2.axhspan(1.0, 1.8, alpha=0.2, color='yellow', label='Breathing zone')
    ax2.set_xlabel('u (m/s)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Horizontal Velocity Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Vertical velocity profiles
    ax3 = axes[1, 0]
    ax3.plot(v_np[i3, :], y, 'b-', linewidth=2, label='x=3m')
    ax3.plot(v_np[i6, :], y, 'r-', linewidth=2, label='x=6m')
    ax3.axhspan(1.0, 1.8, alpha=0.2, color='yellow')
    ax3.set_xlabel('v (m/s)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Vertical Velocity Profiles')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Stagnant zones
    ax4 = axes[1, 1]
    stagnant = (vel_mag < 0.05).astype(float)
    c4 = ax4.contourf(X, Y, stagnant, levels=[0, 0.5, 1], colors=['lightgreen', 'red'], alpha=0.7)
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    ax4.set_title('Stagnant Zones (<0.05 m/s) - Red = Problem')
    ax4.axhspan(1.0, 1.8, alpha=0.3, color='yellow', label='Breathing zone')
    
    plt.suptitle('Conference Room B - Ventilation CFD Analysis\nClient: James Chen', fontweight='bold')
    plt.tight_layout()
    
    output_path = '/home/brad/TiganticLabz/Main_Projects/physics-os/results/conference_room_b.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    # JSON report
    report = {
        "client": "James Chen, Facilities Engineering",
        "project": "Conference Room B Ventilation",
        "date": datetime.now().isoformat(),
        "room": {"L": L, "H": H, "h_inlet_mm": h_inlet*1000, "u_inlet": u_inlet},
        "Re_inlet": Re_inlet,
        "Re_room": Re_room,
        "grid": f"{Nx}x{Ny}",
        "findings": {
            "u_breathing_3m": float(u_breathing_3m),
            "u_breathing_6m": float(u_breathing_6m),
            "jet_core_3m": float(jet_core_3m),
            "jet_core_6m": float(jet_core_6m),
            "problem_confirmed": bool(u_breathing_6m < 0.1)
        }
    }
    
    report_path = '/home/brad/TiganticLabz/Main_Projects/physics-os/results/conference_room_b.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")
    
    print()
    print("=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    run_conference_room_analysis()
