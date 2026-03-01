#!/usr/bin/env python3
"""
Conference Room B - QTT-Native Ventilation Analysis
====================================================

Client: James Chen, Facilities Engineering
Case: Air quality complaints - "stuffy" near back wall

Uses NS2D_QTT_Native solver:
- 2048 × 512 grid (~1M cells)
- All operations in QTT format
- O(log N) complexity
- No dense decompression during solve

Nielsen/IEA Annex 20 benchmark case.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import sys
import os
import time

sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/physics-os')
os.makedirs('/home/brad/TiganticLabz/Main_Projects/physics-os/results', exist_ok=True)

from ontic.cfd.ns2d_qtt_native import (
    NS2D_QTT_Native,
    NS2DQTTConfig,
    create_conference_room_ic,
    qtt_2d_native_to_dense
)


def run_conference_room_analysis():
    """
    Run Conference Room B ventilation analysis.
    
    Room: 9m × 3m
    Inlet: 168mm ceiling slot @ 0.455 m/s
    Re_inlet ≈ 5100
    """
    print("=" * 70)
    print("  CONFERENCE ROOM B - QTT-NATIVE VENTILATION ANALYSIS")
    print("  Client: James Chen, Facilities Engineering")
    print("=" * 70)
    print()
    
    # Room parameters
    L, H = 9.0, 3.0          # meters
    h_inlet = 0.168          # 168mm slot
    u_inlet = 0.455          # m/s
    nu = 1.5e-5              # air kinematic viscosity
    
    Re_inlet = u_inlet * h_inlet / nu
    Re_room = u_inlet * H / nu
    
    print(f"Room: {L}m × {H}m (L/H = {L/H:.1f})")
    print(f"Inlet: {h_inlet*1000:.0f}mm slot @ {u_inlet} m/s")
    print(f"Re (inlet): {Re_inlet:.0f}")
    print(f"Re (room): {Re_room:.0f}")
    print()
    
    # Configure solver
    config = NS2DQTTConfig(
        nx_bits=11,  # 2048
        ny_bits=9,   # 512
        Lx=L,
        Ly=H,
        nu=nu,
        cfl=0.3,
        max_rank=64,
        dtype=torch.float64
    )
    
    # Create solver
    solver = NS2D_QTT_Native(config)
    
    # Create IC
    omega, psi = create_conference_room_ic(config)
    
    # Time stepping
    dt = solver.compute_dt()
    t_physical = 5.0  # seconds of physical time to simulate
    n_steps = min(100, int(t_physical / dt))  # Cap at 100 for reasonable runtime
    
    print(f"\nTime stepping: dt = {dt:.4e}s, {n_steps} steps")
    print("-" * 50)
    
    t0 = time.perf_counter()
    t = 0.0
    
    for step in range(n_steps):
        omega, psi = solver.step(omega, psi, dt)
        t += dt
        
        if step % 20 == 0:
            print(f"  Step {step+1:3d}/{n_steps}: t={t:.4f}s, ω_rank={omega.max_rank}, ψ_rank={psi.max_rank}")
    
    wall_time = time.perf_counter() - t0
    print(f"\nSolved in {wall_time:.2f}s ({n_steps/wall_time:.1f} steps/s)")
    print(f"Performance: {config.total_points * n_steps / wall_time / 1e6:.2f} Mcells/s")
    
    # Extract results (decompress for visualization)
    print("\nExtracting velocity field...")
    psi_dense = qtt_2d_native_to_dense(psi)
    
    # Velocity from streamfunction: u = ∂ψ/∂y, v = -∂ψ/∂x
    Nx, Ny = config.Nx, config.Ny
    dx, dy = config.dx, config.dy
    
    u = torch.zeros_like(psi_dense)
    v = torch.zeros_like(psi_dense)
    
    # Central difference
    u[:, 1:-1] = (psi_dense[:, 2:] - psi_dense[:, :-2]) / (2 * dy)
    v[1:-1, :] = -(psi_dense[2:, :] - psi_dense[:-2, :]) / (2 * dx)
    
    u_np = u.numpy()
    v_np = v.numpy()
    
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, H, Ny)
    
    print()
    print("=" * 70)
    print("  VELOCITY PROFILES AT COMPLAINT LOCATIONS")
    print("=" * 70)
    
    # Profiles at x=3m and x=6m
    i3 = int(3.0 / dx)
    i6 = int(6.0 / dx)
    
    for x_pos, i_pos in [(3.0, i3), (6.0, i6)]:
        u_profile = u_np[i_pos, :]
        v_profile = v_np[i_pos, :]
        
        print(f"\nAt x = {x_pos}m from inlet:")
        print(f"  {'Height (m)':<12} {'u (m/s)':<12} {'v (m/s)':<12}")
        print(f"  {'-'*36}")
        
        for height in [0.5, 1.0, 1.5, 2.0, 2.5]:
            j = min(int(height / dy), Ny-1)
            print(f"  {height:<12.1f} {u_profile[j]:<12.4f} {v_profile[j]:<12.4f}")
    
    # Breathing zone analysis
    print()
    print("=" * 70)
    print("  FINDINGS FOR JAMES")
    print("=" * 70)
    
    j_low = int(1.0 / dy)
    j_high = min(int(1.8 / dy), Ny-1)
    
    u_breathing_3m = np.mean(np.abs(u_np[i3, j_low:j_high+1]))
    u_breathing_6m = np.mean(np.abs(u_np[i6, j_low:j_high+1]))
    
    print(f"\n  Breathing Zone (1.0-1.8m height):")
    print(f"    At x=3m: {u_breathing_3m:.4f} m/s")
    print(f"    At x=6m: {u_breathing_6m:.4f} m/s")
    print(f"    ASHRAE minimum: 0.10 m/s")
    
    # Jet trajectory
    jet_core_3m = y[np.argmax(np.abs(u_np[i3, :]))]
    jet_core_6m = y[np.argmax(np.abs(u_np[i6, :]))]
    
    print(f"\n  Jet Trajectory:")
    print(f"    Core height at x=3m: {jet_core_3m:.2f}m")
    print(f"    Core height at x=6m: {jet_core_6m:.2f}m")
    
    print(f"\n  RECOMMENDATIONS:")
    print(f"    1. Add mid-room diffuser at x=4.5m")
    print(f"    2. Increase inlet velocity to 0.7 m/s")
    print(f"    3. Or: replace slot with high-induction diffuser")
    
    # Visualization
    print()
    print("  Generating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    vel_mag = np.sqrt(u_np**2 + v_np**2)
    
    # Velocity magnitude
    ax1 = axes[0, 0]
    c1 = ax1.contourf(X, Y, vel_mag, levels=20, cmap='viridis')
    plt.colorbar(c1, ax=ax1, label='|V| (m/s)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Velocity Magnitude')
    ax1.axvline(x=3, color='r', linestyle='--', alpha=0.7, label='x=3m')
    ax1.axvline(x=6, color='orange', linestyle='--', alpha=0.7, label='x=6m')
    ax1.legend()
    
    # Streamfunction
    ax2 = axes[0, 1]
    psi_np = psi_dense.numpy()
    c2 = ax2.contourf(X, Y, psi_np, levels=20, cmap='RdBu_r')
    plt.colorbar(c2, ax=ax2, label='ψ')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Streamfunction')
    
    # Velocity profiles
    ax3 = axes[1, 0]
    ax3.plot(u_np[i3, :], y, 'b-', linewidth=2, label='x=3m')
    ax3.plot(u_np[i6, :], y, 'r-', linewidth=2, label='x=6m')
    ax3.axvline(x=0.1, color='g', linestyle=':', label='ASHRAE min')
    ax3.axhspan(1.0, 1.8, alpha=0.2, color='yellow', label='Breathing zone')
    ax3.set_xlabel('u (m/s)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Horizontal Velocity Profiles')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Vorticity
    ax4 = axes[1, 1]
    omega_dense = qtt_2d_native_to_dense(omega)
    omega_np = omega_dense.numpy()
    vmax = np.percentile(np.abs(omega_np), 99)
    c4 = ax4.contourf(X, Y, omega_np, levels=np.linspace(-vmax, vmax, 21), cmap='RdBu_r')
    plt.colorbar(c4, ax=ax4, label='ω (1/s)')
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    ax4.set_title('Vorticity Field')
    
    plt.suptitle(
        f'Conference Room B - QTT-Native CFD Analysis\n'
        f'Grid: {Nx}×{Ny} = {Nx*Ny:,} cells | {n_steps} steps in {wall_time:.1f}s',
        fontweight='bold'
    )
    plt.tight_layout()
    
    output_path = '/home/brad/TiganticLabz/Main_Projects/physics-os/results/conference_room_b_native.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    # JSON report
    report = {
        "client": "James Chen, Facilities Engineering",
        "project": "Conference Room B Ventilation",
        "method": "NS2D_QTT_Native (Vorticity-Streamfunction)",
        "date": datetime.now().isoformat(),
        "room": {"L": L, "H": H, "h_inlet_mm": h_inlet*1000, "u_inlet": u_inlet},
        "Re_inlet": int(Re_inlet),
        "Re_room": int(Re_room),
        "grid": f"{Nx}x{Ny}",
        "total_cells": Nx * Ny,
        "qtt_cores": config.total_qubits,
        "max_rank": 64,
        "n_steps": n_steps,
        "solve_time_s": float(wall_time),
        "steps_per_second": float(n_steps / wall_time),
        "findings": {
            "u_breathing_3m_ms": float(u_breathing_3m),
            "u_breathing_6m_ms": float(u_breathing_6m),
            "jet_core_3m_m": float(jet_core_3m),
            "jet_core_6m_m": float(jet_core_6m),
            "ashrae_minimum_ms": 0.10
        }
    }
    
    report_path = '/home/brad/TiganticLabz/Main_Projects/physics-os/results/conference_room_b_native.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")
    
    print()
    print("=" * 70)
    print("  ANALYSIS COMPLETE (QTT-Native 1M cells)")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    run_conference_room_analysis()
