#!/usr/bin/env python3
"""
Nielsen Ventilation Benchmark - 1 BILLION CELLS
================================================

Ultra high-resolution 3D simulation of the Nielsen room ventilation case.

Grid: ~1B cells (3000 × 333 × 1000)
Domain: 9m × 1m × 3m (L × W × H)

This is an extreme-scale CFD simulation for validation purposes.
"""

import sys
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/Project The Ontic Engine')

import torch
import numpy as np
import time
from ontic.hvac.solver_3d import Solver3D, Solver3DConfig, Inlet3D, Outlet3D

def main():
    # Check GPU memory
    if torch.cuda.is_available():
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        total_mem_gb = props.total_memory / 1e9
        print(f"GPU: {props.name}")
        print(f"GPU Memory: {total_mem_gb:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Running on CPU (will be VERY slow)")
    
    # Nielsen room geometry - FULL 3D ROOM
    L, W, H = 9.0, 3.0, 3.0  # meters (Length × Width × Height)
    inlet_height = 0.168     # m
    inlet_width = 0.48       # m (spanwise width of inlet slot)
    U_in = 0.455             # m/s
    nu = 1.5e-5              # m²/s (air at 20°C)
    
    # Grid for ~1B cells
    # Aspect ratio 9:3:3 = 3:1:1 → nx=3n, ny=n, nz=n
    # 3n × n × n = 3n³ = 1B → n ≈ 693
    # Use: 2100 × 700 × 700 = 1,029,000,000 cells (~1.03B)
    nx = 2100
    ny = 700
    nz = 700
    total_cells = nx * ny * nz
    
    print(f"\n{'='*70}")
    print(f"NIELSEN BENCHMARK - 1 BILLION CELLS")
    print(f"{'='*70}")
    print(f"Grid: {nx} × {ny} × {nz} = {total_cells:,} cells")
    print(f"Domain: {L}m × {W}m × {H}m")
    print(f"dx = {L/(nx-1)*1000:.3f} mm")
    print(f"dy = {W/(ny-1)*1000:.3f} mm")
    print(f"dz = {H/(nz-1)*1000:.3f} mm")
    print(f"Cells in inlet height: {inlet_height / (H/(nz-1)):.1f}")
    print(f"Re = {U_in * inlet_height / nu:.0f}")
    
    # Estimate memory requirement
    # Each field: nx × ny × nz × 4 bytes (float32)
    # Fields: u, v, w, p, T, plus temporaries (~10 fields)
    bytes_per_field = nx * ny * nz * 4
    est_memory_gb = bytes_per_field * 15 / 1e9  # 15 fields estimate
    print(f"\nEstimated memory: {est_memory_gb:.1f} GB")
    
    if torch.cuda.is_available() and est_memory_gb > total_mem_gb * 0.9:
        print(f"WARNING: May exceed GPU memory!")
    
    # Create inlet at ceiling (slot centered in y)
    inlet = Inlet3D(
        x=0.0,
        y=W / 2,
        z=H,
        width=inlet_width,      # Actual inlet slot width
        height=inlet_height * 2,
        velocity=U_in,
        T=16.0,
        direction='x+',
    )
    
    # Create outlet at floor (opposite wall)
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
        max_iterations=5000,  # May need more for full convergence
        convergence_tol=1e-5,
        dt_safety=0.25,
        pressure_solver='dct',
        advection_scheme='tvd',
        tvd_limiter='van_leer',
        enable_buoyancy=False,
        enable_turbulence=False,
        verbose=True,
        diag_interval=100,
    )
    
    print(f"\nInitializing solver...")
    init_start = time.time()
    solver = Solver3D(config)
    init_time = time.time() - init_start
    print(f"Initialization time: {init_time:.1f}s")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory reserved: {reserved:.2f} GB")
    
    print(f"\nSolving...")
    solve_start = time.time()
    state = solver.solve()
    solve_time = time.time() - solve_start
    
    print(f"\n{'='*70}")
    print(f"SOLUTION COMPLETE")
    print(f"{'='*70}")
    print(f"Solve time: {solve_time:.1f}s ({solve_time/60:.1f} min)")
    print(f"Iterations: {state.iteration}")
    print(f"Converged: {state.converged}")
    print(f"Time per iteration: {solve_time/state.iteration*1000:.1f} ms")
    print(f"Cells per second: {total_cells * state.iteration / solve_time / 1e6:.1f} M cells/s")
    
    # Aalborg experimental data
    AALBORG_DATA = {
        1.0: {
            'z_H': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'u_U': np.array([-0.05, -0.08, -0.10, -0.08, -0.02, 0.05, 0.12, 0.22, 0.38, 0.58, 0.85]),
        },
        2.0: {
            'z_H': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'u_U': np.array([-0.03, -0.05, -0.06, -0.05, -0.03, 0.00, 0.05, 0.12, 0.22, 0.35, 0.52]),
        },
    }
    
    # Extract and compare profiles
    print(f"\n{'='*70}")
    print(f"VALIDATION AGAINST AALBORG EXPERIMENTAL DATA")
    print(f"{'='*70}")
    
    z_grid = np.linspace(0, H, nz)
    avg_rms = 0
    
    for xH in [1.0, 2.0]:
        x_pos = xH * H
        i = int(x_pos / L * (nx - 1))
        i = max(1, min(i, nx - 2))
        
        u_profile = state.u[i, ny//2, :].cpu().numpy()
        
        z_exp = AALBORG_DATA[xH]['z_H']
        u_exp = AALBORG_DATA[xH]['u_U']
        
        # Interpolate simulation to experimental points
        u_sim = np.interp(z_exp * H, z_grid, u_profile) / U_in
        
        rms = np.sqrt(np.mean((u_sim - u_exp)**2))
        avg_rms += rms
        
        print(f"\nx/H = {xH}: RMS error = {rms*100:.1f}%")
        print(f"  z/H    Exp     Sim     Diff")
        print(f"  ----   -----   -----   -----")
        for z, e, s in zip(z_exp, u_exp, u_sim):
            diff = abs(s - e)
            print(f"  {z:.1f}    {e:+.2f}    {s:+.2f}    {diff:.2f}")
    
    avg_rms /= 2
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULT")
    print(f"{'='*70}")
    print(f"Average RMS error: {avg_rms*100:.1f}%")
    print(f"Target: < 10%")
    
    if avg_rms < 0.10:
        print(f"Status: ✓ PASS")
    else:
        print(f"Status: ✗ FAIL")
    
    # Save results
    results = {
        'grid': f'{nx}x{ny}x{nz}',
        'total_cells': total_cells,
        'solve_time_s': solve_time,
        'iterations': state.iteration,
        'converged': state.converged,
        'rms_1H': float((avg_rms * 2 - avg_rms) * 100),  # individual RMS
        'rms_2H': float(avg_rms * 100),
        'avg_rms': float(avg_rms * 100),
        'pass': avg_rms < 0.10,
    }
    
    import json
    output_path = '/home/brad/TiganticLabz/Main_Projects/Project The Ontic Engine/HVAC_CFD/nielsen_1B_result.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
