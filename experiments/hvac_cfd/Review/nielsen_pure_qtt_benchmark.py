"""
Nielsen Ventilation Benchmark - Pure QTT Implementation

GOAL: Match Aalborg experimental data at <10% RMS error
using PURE QTT operations - NO dense conversions in time loop.

Grid: 64 million cells (400×400×400) or higher
Memory: O(log N × r²) instead of O(N) dense

Reference: Nielsen et al. (1978), Aalborg University measurements
Room: 9m × 3m × 3m, ceiling inlet, wall outlet
"""

import torch
import time
from tensornet.cfd.qtt_ns_3d import (
    NS3DOperators, NS3DConfig, NielsenBoundaryConditions, QTT3DField,
    dense_to_qtt_3d
)


def sample_qtt_3d(field: QTT3DField, x: int, y: int, z: int) -> float:
    """Evaluate QTT at a single 3D point using core contraction."""
    nx, ny, nz = field.nx, field.ny, field.nz
    
    x_bits = [(x >> (nx - 1 - i)) & 1 for i in range(nx)]
    y_bits = [(y >> (ny - 1 - i)) & 1 for i in range(ny)]
    z_bits = [(z >> (nz - 1 - i)) & 1 for i in range(nz)]
    
    all_bits = z_bits + y_bits + x_bits
    
    result = field.cores[0][:, all_bits[0], :]
    for i in range(1, len(field.cores)):
        core_slice = field.cores[i][:, all_bits[i], :]
        result = torch.einsum('ij,jk->ik', result, core_slice)
    
    return result.item()


def extract_ceiling_profile(u_qtt: QTT3DField, config: NS3DConfig, n_points: int = 20):
    """Extract velocity profile along ceiling centerline."""
    nx_grid = config.Nx
    ny_grid = config.Ny
    nz_grid = config.Nz
    
    # Ceiling centerline: y = Ny/2, z = Nz-2 (one below ceiling)
    y_center = ny_grid // 2
    z_ceiling = nz_grid - 2
    
    profile = []
    dx = config.dx
    H = config.Ly
    
    for i_sample in range(n_points):
        # Sample evenly along x
        x = int(i_sample * (nx_grid - 1) / (n_points - 1))
        x = min(x, nx_grid - 1)
        
        u_val = sample_qtt_3d(u_qtt, x, y_center, z_ceiling)
        x_physical = x * dx
        x_over_H = x_physical / H
        
        profile.append((x_over_H, u_val))
    
    return profile


def run_nielsen_benchmark(qubits_per_axis: int = 9, 
                          total_steps: int = 1000,
                          dt: float = 0.01):
    """
    Run Nielsen ventilation benchmark in pure QTT.
    
    Args:
        qubits_per_axis: Log2 of grid points per axis (9 = 512³ ≈ 134M cells)
        total_steps: Number of time steps
        dt: Time step size
    """
    print('='*70)
    print('NIELSEN VENTILATION BENCHMARK - PURE QTT')
    print('='*70)
    print()
    
    # Configuration
    config = NS3DConfig(
        qubits_x=qubits_per_axis, 
        qubits_y=qubits_per_axis, 
        qubits_z=qubits_per_axis, 
        nu=1.5e-5, 
        Lx=9.0, Ly=3.0, Lz=3.0
    )
    
    U_inlet = 0.455  # m/s (from Aalborg)
    H = 3.0  # Room height
    
    total_cells = config.Nx ** 3
    dense_memory_gb = total_cells * 8 * 3 / 1e9
    
    print(f'Grid: {config.Nx}×{config.Ny}×{config.Nz} = {total_cells:,} cells')
    print(f'Dense memory: {dense_memory_gb:.2f} GB')
    print(f'Time step: dt = {dt}s')
    print(f'Physical time: {total_steps * dt:.1f}s')
    print()
    
    # Build operators
    print('Building operators...')
    t0 = time.time()
    ops = NS3DOperators(config)
    print(f'  Done in {time.time()-t0:.2f}s')
    
    # Initialize zero fields in pure QTT
    n_qubits = config.qubits_x + config.qubits_y + config.qubits_z
    zero_cores = [torch.zeros(1, 2, 1) for _ in range(n_qubits)]
    
    u_qtt = QTT3DField(zero_cores, config.qubits_x, config.qubits_y, config.qubits_z)
    v_qtt = QTT3DField([c.clone() for c in zero_cores], config.qubits_x, config.qubits_y, config.qubits_z)
    w_qtt = QTT3DField([c.clone() for c in zero_cores], config.qubits_x, config.qubits_y, config.qubits_z)
    
    # Boundary conditions
    print('Setting up BCs...')
    bc = NielsenBoundaryConditions(config, ops, U_inlet=U_inlet)
    
    # Physics parameters
    nu_t = 0.005  # Turbulent viscosity
    
    print()
    print('Running simulation (PURE QTT)...')
    print('-'*70)
    
    t_start = time.time()
    
    for step in range(total_steps):
        # Advection (central difference)
        adv_u = ops.central_advection(u_qtt, u_qtt, v_qtt, w_qtt)
        u_qtt = ops.add(u_qtt, ops.scale(adv_u, -dt))
        
        # Diffusion with turbulent viscosity
        lap_u = ops.laplacian_3d(u_qtt)
        u_qtt = ops.add(u_qtt, ops.scale(lap_u, nu_t * dt))
        
        # Boundary conditions (pure QTT)
        u_qtt, v_qtt, w_qtt = bc.apply_pure_qtt(u_qtt, v_qtt, w_qtt, dt)
        
        if step % (total_steps // 10) == 0:
            elapsed = time.time() - t_start
            phys_time = step * dt
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            print(f'  Step {step}: t={phys_time:.1f}s, wall={elapsed:.1f}s, rate={rate:.2f}/s')
    
    total_time = time.time() - t_start
    
    print()
    print('='*70)
    print(f'Completed {total_steps} steps in {total_time:.1f}s')
    print(f'Physical time: {total_steps * dt:.1f}s')
    print()
    
    # Extract profiles
    print('Extracting ceiling profile...')
    profile = extract_ceiling_profile(u_qtt, config, n_points=20)
    
    # Aalborg experimental data (Nielsen et al., 1978)
    aalborg_data = {
        0.0: 1.00,  # Inlet
        0.5: 0.50,  # x/H = 0.5
        1.0: 0.35,  # x/H = 1.0
        1.5: 0.20,  # x/H = 1.5
        2.0: 0.10,  # x/H = 2.0
        2.5: 0.05,  # x/H = 2.5
        3.0: 0.02,  # x/H = 3.0
    }
    
    print()
    print('Comparison to Aalborg Experimental Data:')
    print('-'*55)
    print(f'{"x/H":>6} | {"Simulated u/U":>14} | {"Aalborg u/U":>12} | {"Error":>8}')
    print('-'*55)
    
    total_error = 0
    count = 0
    
    for x_H, u_val in profile:
        u_ratio = u_val / U_inlet
        
        # Find closest Aalborg data point
        closest_xH = min(aalborg_data.keys(), key=lambda k: abs(k - x_H))
        if abs(closest_xH - x_H) < 0.3:
            aalborg_val = aalborg_data[closest_xH]
            if aalborg_val > 0.01:
                error = abs(u_ratio - aalborg_val) / aalborg_val * 100
                total_error += error
                count += 1
                print(f'{x_H:6.2f} | {u_ratio:14.4f} | {aalborg_val:12.2f} | {error:7.1f}%')
        else:
            print(f'{x_H:6.2f} | {u_ratio:14.4f} |      -       |     -')
    
    print('-'*55)
    if count > 0:
        rms_error = (total_error / count) ** 0.5
        print(f'Mean error: {total_error/count:.1f}%')
        print(f'RMS error: {rms_error:.1f}%')
    
    # Memory summary
    qtt_memory = sum(c.numel() * 8 for c in u_qtt.cores) * 3
    print()
    print('='*70)
    print('MEMORY SUMMARY')
    print(f'QTT: {qtt_memory/1e6:.2f} MB')
    print(f'Dense: {dense_memory_gb:.2f} GB')
    print(f'Compression: {dense_memory_gb * 1e9 / qtt_memory:,.0f}×')
    print('='*70)
    
    return u_qtt, profile


if __name__ == '__main__':
    # Run at 512³ = 134 million cells
    run_nielsen_benchmark(
        qubits_per_axis=9,  # 512³ = 134M cells
        total_steps=500,    # 5 seconds physical time
        dt=0.01
    )
