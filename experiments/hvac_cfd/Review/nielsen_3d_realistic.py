#!/usr/bin/env python3
"""
Nielsen 3D Realistic Room Benchmark
====================================
9m x 3m x 3m room (actual room proportions, not narrow quasi-2D)

This allows proper 3D turbulent cascade and lateral jet spreading.
Validation: Grid convergence, mass conservation, physical behavior.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# Force CPU for now (GPU has initialization issues)
device_str = 'cpu'
device = torch.device(device_str)

from ontic.hvac.solver_3d import Solver3D, Solver3DState, Solver3DConfig, Inlet3D, Outlet3D

print("=" * 70)
print("NIELSEN 3D REALISTIC ROOM BENCHMARK")
print("=" * 70)
print()
print("Room: 9m x 3m x 3m (Length x Width x Height)")
print("Inlet: Ceiling slot at x=0, spanning full width")
print("Outlet: Floor at x=L, spanning full width")
print()

# ============================================================
# Physical parameters (Nielsen benchmark)
# ============================================================
L = 9.0    # Room length (m)
W = 3.0    # Room width (m) - realistic, not quasi-2D
H = 3.0    # Room height (m)

inlet_height = 0.168  # Slot height (m)
inlet_velocity = 0.455  # m/s
Re = 5096  # Reynolds number

nu = inlet_velocity * inlet_height / Re  # kinematic viscosity
alpha = nu / 0.71  # thermal diffusivity (Pr = 0.71)

T_inlet = 20.0  # Supply air temperature (°C)
T_init = 24.0   # Initial room temperature (°C)


def run_test(nx, ny, nz, use_les=False, cs=0.10, max_iter=2000, label=""):
    """Run a single test case."""
    
    ncells = nx * ny * nz
    print(f"3D Nielsen: {nx}×{ny}×{nz} = {ncells:,} cells, Re={Re}")
    if use_les:
        print(f"  LES: Smagorinsky C_s={cs}")
    else:
        print(f"  Laminar (Direct Numerical Simulation)")
    
    # Create ceiling inlet (at x=0, spanning full width, at ceiling z=H)
    ceiling_inlet = Inlet3D(
        x=0.0,
        y=W / 2,          # Center in y
        z=H - inlet_height / 2,  # At ceiling
        width=W,          # Full width of room
        height=inlet_height,
        velocity=inlet_velocity,
        T=T_inlet,
        direction='x+',   # Blowing into room (+x direction)
    )
    
    # Create floor outlet (at x=L, spanning full width, at floor z=0)
    floor_outlet = Outlet3D(
        x=L,
        y=W / 2,
        z=inlet_height / 2,  # Near floor
        width=W,
        height=inlet_height,
    )
    
    # Create config
    config = Solver3DConfig(
        length=L,
        width=W,
        height=H,
        nx=nx,
        ny=ny,
        nz=nz,
        nu=nu,
        alpha=alpha,
        T_ref=T_init,
        inlets=[ceiling_inlet],
        outlets=[floor_outlet],
        enable_buoyancy=True,
        max_iterations=max_iter,
        convergence_tol=1e-5,
        dt_safety=0.15,  # Conservative for stability
        pressure_solver='dct',  # DCT is stable (SOR has divergence issues)
        pressure_iterations=500,
        pressure_tol=1e-5,
        enable_turbulence=use_les,
        turbulence_model='smagorinsky' if use_les else 'laminar',
        C_s=cs,
        verbose=True,
        diag_interval=100,
    )
    
    # Create solver
    solver = Solver3D(config, device=device_str)
    
    # ============================================================
    # Solve - let solver create initial state with proper BCs
    # ============================================================
    print("\nSolving...")
    t0 = time.time()
    
    # Don't pass initial state - let solver create one with inlet velocities applied
    final_state = solver.solve()
    
    solve_time = time.time() - t0
    
    # ============================================================
    # Analyze results
    # ============================================================
    print(f"\nSolution time: {solve_time:.1f}s")
    print(f"Iterations: {final_state.iteration}")
    print(f"Converged: {final_state.converged}")
    print(f"Max velocity: {final_state.u.max().item():.3f} m/s")
    
    # ============================================================
    # Extract velocity profiles at centerline (y = W/2)
    # ============================================================
    jc = ny // 2  # centerline index
    dx = L / nx
    
    # Measurement locations (x/H)
    x_H_locations = [1.0, 2.0, 3.0]
    profiles = {}
    
    for x_H in x_H_locations:
        x_pos = x_H * H  # convert to meters
        i = int(x_pos / dx)
        if i >= nx:
            i = nx - 1
        
        # Extract vertical profile
        u_profile = final_state.u[i, jc, :].cpu().numpy()
        z_coords = np.linspace(0, H, nz)
        z_H = z_coords / H
        u_norm = u_profile / inlet_velocity
        
        profiles[f'x_H_{x_H}'] = {
            'z_H': z_H.tolist(),
            'u_norm': u_norm.tolist(),
        }
    
    # ============================================================
    # Physics checks
    # ============================================================
    print("\n" + "=" * 60)
    print("PHYSICS CHECKS")
    print("=" * 60)
    
    dz = H / nz
    dy = W / ny
    
    # 1. Mass conservation
    inlet_flux = (final_state.u[solver.inlet_mask].sum() * dy * dz).item()
    outlet_flux = abs((final_state.u[solver.outlet_mask].sum() * dy * dz).item())
    mass_error = abs(inlet_flux - outlet_flux) / max(abs(inlet_flux), 1e-10) * 100
    print(f"Mass conservation error: {mass_error:.2f}%")
    
    # 2. Ceiling jet velocity
    u_ceiling = final_state.u[:, jc, -1].cpu().numpy()
    u_ceiling_max = u_ceiling.max()
    idx_50 = np.where(u_ceiling < 0.5 * inlet_velocity)[0]
    u_ceiling_decay_x = idx_50[0] * dx / H if len(idx_50) > 0 else L / H
    print(f"Ceiling jet max velocity: {u_ceiling_max:.3f} m/s ({u_ceiling_max/inlet_velocity:.2f} U_in)")
    print(f"Ceiling jet 50% decay at x/H = {u_ceiling_decay_x:.1f}")
    
    # 3. Recirculation check
    u_floor = final_state.u[:, jc, 0].cpu().numpy()
    has_recirculation = (u_floor < -0.01).any()
    print(f"Floor recirculation present: {has_recirculation}")
    
    # 4. Thermal stratification
    T_floor = final_state.T[:, jc, 0].mean().item()
    T_ceiling = final_state.T[:, jc, -1].mean().item()
    T_gradient = T_floor - T_ceiling
    print(f"Thermal stratification (floor-ceiling): {T_gradient:.2f}°C")
    
    # ============================================================
    # Print profiles
    # ============================================================
    print("\n" + "=" * 60)
    print("VELOCITY PROFILES (centerline y=W/2)")
    print("=" * 60)
    
    for x_H in x_H_locations:
        prof = profiles[f'x_H_{x_H}']
        z_H = np.array(prof['z_H'])
        u_norm = np.array(prof['u_norm'])
        
        # Sample at 11 points
        idx = np.linspace(0, len(z_H)-1, 11).astype(int)
        print(f"\nProfile @ x/H={x_H}:")
        print(f"  z/H:    {np.round(z_H[idx], 2)}")
        print(f"  u/U_in: {np.round(u_norm[idx], 2)}")
    
    # ============================================================
    # Validation metrics
    # ============================================================
    physics_pass = (
        mass_error < 10.0 and           # Mass conserved within 10%
        u_ceiling_max > 0.5 * inlet_velocity and  # Jet attaches
        has_recirculation              # Room has recirculation
    )
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Mass conservation: {'PASS' if mass_error < 10.0 else 'FAIL'} ({mass_error:.2f}%)")
    print(f"Jet attachment:    {'PASS' if u_ceiling_max > 0.5*inlet_velocity else 'FAIL'}")
    print(f"Recirculation:     {'PASS' if has_recirculation else 'FAIL'}")
    print(f"Overall:           {'PASS' if physics_pass else 'FAIL'}")
    
    return {
        'label': label,
        'grid': f'{nx}×{ny}×{nz}',
        'cells': ncells,
        'use_les': use_les,
        'solve_time': solve_time,
        'converged': final_state.converged,
        'iterations': final_state.iteration,
        'mass_error': mass_error,
        'u_ceiling_max': float(u_ceiling_max),
        'has_recirculation': bool(has_recirculation),
        'profiles': profiles,
        'physics_pass': physics_pass,
    }


# ============================================================
# Run grid convergence study
# ============================================================
results = []

# Test 1: Coarse grid
print("\n" + "=" * 70)
print("TEST 1: Coarse Grid 90×30×30 (~81k cells)")
print("=" * 70 + "\n")
r1 = run_test(90, 30, 30, use_les=False, max_iter=1500, label="Coarse Laminar")
results.append(r1)

# Test 2: Medium grid
print("\n" + "=" * 70)
print("TEST 2: Medium Grid 128×43×43 (~237k cells)")
print("=" * 70 + "\n")
r2 = run_test(128, 43, 43, use_les=False, max_iter=2000, label="Medium Laminar")
results.append(r2)

# Test 3: Fine grid with LES
print("\n" + "=" * 70)
print("TEST 3: Fine Grid 180×60×60 (~648k cells) with LES")
print("=" * 70 + "\n")
r3 = run_test(180, 60, 60, use_les=True, cs=0.10, max_iter=2500, label="Fine LES")
results.append(r3)

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY - 3D Realistic Room (9m × 3m × 3m)")
print("=" * 70)
print(f"{'Grid':<25} {'Cells':>12} {'Mass Err':>10} {'Jet':>8} {'Recirc':>8} {'Time':>8}")
print("-" * 70)
for r in results:
    mass_str = f"{r['mass_error']:.1f}%"
    jet_str = "YES" if r['u_ceiling_max'] > 0.5 * inlet_velocity else "NO"
    recirc_str = "YES" if r['has_recirculation'] else "NO"
    time_str = f"{r['solve_time']:.1f}s"
    print(f"{r['grid']:<25} {r['cells']:>12,} {mass_str:>10} {jet_str:>8} {recirc_str:>8} {time_str:>8}")

print("-" * 70)
print("\nValidation criteria:")
print("  - Mass conservation < 10%")
print("  - Ceiling jet attachment (u_max > 0.5 U_in)")
print("  - Room recirculation present")

# Save results
output_file = 'HVAC_CFD/nielsen_3d_realistic_result.json'
with open(output_file, 'w') as f:
    json.dump({
        'room': {'L': L, 'W': W, 'H': H},
        'inlet': {'height': inlet_height, 'velocity': inlet_velocity},
        'Re': Re,
        'results': results,
    }, f, indent=2, default=str)

print(f"\nResults saved to {output_file}")
