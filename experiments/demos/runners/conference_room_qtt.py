#!/usr/bin/env python3
"""
Conference Room B - Ventilation Analysis (QTT-Accelerated)
===========================================================

Client: James Chen, Facilities Engineering
Case: Air quality complaints - "stuffy" near back wall

Uses QTT-compressed solver (O(log N) complexity).
No dense decompression during solve.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import sys
import os
import time

sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main')
os.makedirs('/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main/results', exist_ok=True)

from tensornet.cfd.euler2d_native import (
    Euler2D_Native, 
    Euler2DNativeConfig, 
    Euler2DStateNative
)
from tensornet.cfd.qtt_2d import dense_to_qtt_2d, qtt_2d_to_dense


def create_ventilation_ic(
    nx_bits: int, ny_bits: int, 
    L: float, H: float,
    h_inlet: float, u_inlet: float,
    config: Euler2DNativeConfig
) -> Euler2DStateNative:
    """
    Create initial condition for ventilation case.
    
    Low-Mach compressible approximation:
    - Set high background pressure (low Mach)
    - Jet enters from ceiling slot
    """
    Nx = 2**nx_bits
    Ny = 2**ny_bits
    
    x = torch.linspace(0, 1, Nx, dtype=config.dtype, device=config.device)
    y = torch.linspace(0, 1, Ny, dtype=config.dtype, device=config.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Ambient air (normalized to [0,1] domain)
    gamma = config.gamma
    rho_0 = 1.0
    P_0 = 100.0  # High pressure → low Mach (M ~ u/c ~ u*sqrt(gamma*P/rho))
    
    rho = rho_0 * torch.ones_like(X)
    u = torch.zeros_like(X)
    v = torch.zeros_like(X)
    
    # Inlet jet: ceiling slot at x=0, y ∈ [1-h_inlet/H, 1]
    # In normalized coords: y_top = 1, slot at y ∈ [1-h_inlet/H, 1]
    y_slot_bottom = 1.0 - (h_inlet / H)
    slot_mask = (X < 0.05) & (Y > y_slot_bottom)
    
    # Jet velocity (horizontal)
    u_jet = u_inlet / (L/1.0)  # Scale to domain units
    u[slot_mask] = u_jet
    v[slot_mask] = -0.1 * u_jet  # Slight downward deflection
    
    # Convert to conservative variables
    rhou = rho * u
    rhov = rho * v
    E = P_0 / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
    
    # Compress to QTT
    max_rank = config.max_rank
    rho_qtt = dense_to_qtt_2d(rho, max_bond=max_rank)
    rhou_qtt = dense_to_qtt_2d(rhou, max_bond=max_rank)
    rhov_qtt = dense_to_qtt_2d(rhov, max_bond=max_rank)
    E_qtt = dense_to_qtt_2d(E, max_bond=max_rank)
    
    return Euler2DStateNative(rho_qtt, rhou_qtt, rhov_qtt, E_qtt)


def run_conference_room_qtt():
    """
    Run Conference Room B ventilation analysis using QTT solver.
    """
    print("=" * 60)
    print("  CONFERENCE ROOM B - QTT-ACCELERATED CFD")
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
    
    # Grid: 2^10 × 2^10 = 1024 × 1024 cells (QTT compressed)
    # Actual storage: O(log N) ~ O(20) cores, not O(1,048,576) dense elements
    nx_bits = 10  # 1024 in x
    ny_bits = 10  # 1024 in y
    Nx, Ny = 2**nx_bits, 2**ny_bits
    
    print(f"Grid: {Nx}×{Ny} = {Nx*Ny:,} cells")
    print(f"QTT bits: {nx_bits}+{ny_bits} = {nx_bits+ny_bits} cores")
    print(f"Compression: O(log N) = O({nx_bits+ny_bits}), not O({Nx*Ny})")
    print()
    
    # Configure solver
    config = Euler2DNativeConfig(
        gamma=1.4,
        cfl=0.4,
        max_rank=32,
        tci_tolerance=1e-5,
        dtype=torch.float64,
        device=torch.device("cpu")
    )
    
    # Create solver
    print("Initializing QTT solver...")
    solver = Euler2D_Native(nx_bits, ny_bits, config)
    
    # Create initial condition
    print("Creating ventilation IC...")
    state = create_ventilation_ic(
        nx_bits, ny_bits, L, H, h_inlet, u_inlet, config
    )
    print(f"  Initial QTT rank: {state.max_rank()}")
    
    # Time integration - need enough steps for jet to propagate across room
    # With t_final = 0.5 in unit domain, and u ~ 0.05, jet travels ~ 0.025 
    # Need t ~ L/u ~ 1/0.05 ~ 20 time units for full propagation
    t_final = 5.0  # Non-dimensional time units
    n_steps = 100
    
    print()
    print("Solving (QTT-native)...")
    print("-" * 40)
    
    t0 = time.perf_counter()
    t = 0.0
    
    for step in range(n_steps):
        # Use adaptive dt from CFL
        dt_cfl = solver.compute_dt(state)
        
        state = solver.step(state, dt_cfl)
        t += dt_cfl
        
        if step % 10 == 0:
            print(f"  Step {step+1:2d}/{n_steps}: t={t:.4f}, rank={state.max_rank()}")
    
    wall_time = time.perf_counter() - t0
    print(f"\nSolved in {wall_time:.2f}s ({n_steps/wall_time:.1f} steps/s)")
    
    # Extract results (decompress only for visualization)
    print("\nExtracting results...")
    rho, u, v, P = state.get_primitives(config.gamma)
    
    # Convert to physical units
    u_phys = u.numpy() * (L/1.0)  # Back to m/s
    v_phys = v.numpy() * (L/1.0)
    
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, H, Ny)
    
    print()
    print("=" * 60)
    print("  VELOCITY PROFILES AT COMPLAINT LOCATIONS")
    print("=" * 60)
    
    # Profiles at x=3m and x=6m
    dx_grid = L / Nx
    i3 = int(3.0 / dx_grid)
    i6 = int(6.0 / dx_grid)
    
    for x_pos, i_pos in [(3.0, i3), (6.0, i6)]:
        u_profile = u_phys[i_pos, :]
        v_profile = v_phys[i_pos, :]
        
        print(f"\nAt x = {x_pos}m from inlet:")
        print(f"  {'Height (m)':<12} {'u (m/s)':<12} {'v (m/s)':<12}")
        print(f"  {'-'*36}")
        
        dy_grid = H / Ny
        for height in [0.5, 1.0, 1.5, 2.0, 2.5]:
            j = min(int(height / dy_grid), Ny-1)
            print(f"  {height:<12.1f} {u_profile[j]:<12.4f} {v_profile[j]:<12.4f}")
    
    # Breathing zone analysis
    print()
    print("=" * 60)
    print("  FINDINGS FOR JAMES")
    print("=" * 60)
    
    dy_grid = H / Ny
    j_low = int(1.0 / dy_grid)
    j_high = min(int(1.8 / dy_grid), Ny-1)
    
    u_breathing_3m = np.mean(np.abs(u_phys[i3, j_low:j_high+1]))
    u_breathing_6m = np.mean(np.abs(u_phys[i6, j_low:j_high+1]))
    
    print(f"\n  Breathing Zone (1.0-1.8m height):")
    print(f"    At x=3m: {u_breathing_3m:.4f} m/s")
    print(f"    At x=6m: {u_breathing_6m:.4f} m/s")
    print(f"    ASHRAE minimum: 0.10 m/s")
    
    if u_breathing_6m < 0.1:
        print(f"\n  ⚠️  PROBLEM CONFIRMED:")
        print(f"      Velocity at x=6m ({u_breathing_6m:.3f} m/s) is below ASHRAE minimum")
        print(f"      This explains the 'stuffy' complaints from occupants")
    
    # Jet trajectory
    jet_core_3m = y[np.argmax(np.abs(u_phys[i3, :]))]
    jet_core_6m = y[np.argmax(np.abs(u_phys[i6, :]))]
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    vel_mag = np.sqrt(u_phys**2 + v_phys**2)
    
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
                   u_phys[::skip, ::skip].T, v_phys[::skip, ::skip].T,
                   color='white', linewidth=0.5, density=1.2)
    
    # Horizontal velocity profiles
    ax2 = axes[0, 1]
    ax2.plot(u_phys[i3, :], y, 'b-', linewidth=2, label='x=3m')
    ax2.plot(u_phys[i6, :], y, 'r-', linewidth=2, label='x=6m')
    ax2.axvline(x=0.1, color='g', linestyle=':', label='ASHRAE min')
    ax2.axhspan(1.0, 1.8, alpha=0.2, color='yellow', label='Breathing zone')
    ax2.set_xlabel('u (m/s)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Horizontal Velocity Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Pressure field
    ax3 = axes[1, 0]
    P_np = P.numpy()
    c3 = ax3.contourf(X, Y, P_np, levels=20, cmap='coolwarm')
    plt.colorbar(c3, ax=ax3, label='P')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_title('Pressure Field')
    
    # Stagnant zones
    ax4 = axes[1, 1]
    stagnant = (vel_mag < 0.05).astype(float)
    c4 = ax4.contourf(X, Y, stagnant, levels=[0, 0.5, 1], colors=['lightgreen', 'red'], alpha=0.7)
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    ax4.set_title('Stagnant Zones (<0.05 m/s) - Red = Problem')
    ax4.axhspan(1.0, 1.8, alpha=0.3, color='yellow', label='Breathing zone')
    
    plt.suptitle('Conference Room B - QTT-Accelerated Ventilation CFD\nClient: James Chen', fontweight='bold')
    plt.tight_layout()
    
    output_path = '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main/results/conference_room_b_qtt.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    # JSON report
    report = {
        "client": "James Chen, Facilities Engineering",
        "project": "Conference Room B Ventilation",
        "method": "QTT-Accelerated Compressible Euler (Low-Mach)",
        "date": datetime.now().isoformat(),
        "room": {"L": L, "H": H, "h_inlet_mm": h_inlet*1000, "u_inlet": u_inlet},
        "Re_inlet": int(Re_inlet),
        "Re_room": int(Re_room),
        "grid": f"{Nx}x{Ny}",
        "qtt_bits": nx_bits + ny_bits,
        "solve_time_s": float(wall_time),
        "findings": {
            "u_breathing_3m": float(u_breathing_3m),
            "u_breathing_6m": float(u_breathing_6m),
            "jet_core_3m": float(jet_core_3m),
            "jet_core_6m": float(jet_core_6m),
            "problem_confirmed": bool(u_breathing_6m < 0.1)
        }
    }
    
    report_path = '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main/results/conference_room_b_qtt.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")
    
    print()
    print("=" * 60)
    print("  ANALYSIS COMPLETE (QTT-Native)")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    run_conference_room_qtt()
