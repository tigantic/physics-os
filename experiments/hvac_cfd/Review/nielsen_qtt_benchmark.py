#!/usr/bin/env python3
"""
Nielsen Ventilation Benchmark - QTT Native Implementation
==========================================================

This is the CORRECT approach using QTT (Quantized Tensor Train) format
with O(log N × r²) memory instead of O(N) dense.

For 1 BILLION cells:
- Dense: 1B × 4 bytes × 5 fields = 20 GB per state
- QTT: 30 qubits × 32² × 5 fields × 4 bytes ≈ 600 KB per state

This is why HyperTensor exists.

Reference: Source_of_Truth.md §1: "Dense is Anti-QTT"
"""

import sys
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/Project HyperTensor')

import time
import math
import torch
import numpy as np

# Check available QTT infrastructure
print("Checking QTT infrastructure...")

try:
    from tensornet.cfd.pure_qtt_ops import dense_to_qtt, qtt_to_dense, QTTState
    print("  ✓ pure_qtt_ops available")
except ImportError as e:
    print(f"  ✗ pure_qtt_ops: {e}")

try:
    from tensornet.cfd.qtt_2d import QTT2DState, dense_to_qtt_2d
    print("  ✓ qtt_2d available")
except ImportError as e:
    print(f"  ✗ qtt_2d: {e}")

try:
    from tensornet.cfd.fast_euler_3d import QTT3DState, dense_to_qtt_3d, Euler3DConfig
    print("  ✓ fast_euler_3d (3D QTT) available")
except ImportError as e:
    print(f"  ✗ fast_euler_3d: {e}")

try:
    from tensornet.cfd.qtt_ns_3d import (
        NS3DSolver, NS3DConfig, NS3DState,
        create_inlet_jet_state, create_quiescent_state
    )
    print("  ✓ qtt_ns_3d (3D NS solver) available")
    HAS_NS_SOLVER = True
except ImportError as e:
    print(f"  ✗ qtt_ns_3d: {e}")
    HAS_NS_SOLVER = False

try:
    from tensornet.infra.fieldops.operators import Laplacian, Advect, PoissonSolver, Project
    print("  ✓ fieldops operators available")
except ImportError as e:
    print(f"  ✗ fieldops operators: {e}")

# Skip MPO operators - they try to compile CUDA kernels
print("  ⊘ MPO operators skipped (CUDA compilation)")

try:
    from tensornet.engine.substrate import Field
    print("  ✓ substrate Field available")
except ImportError as e:
    print(f"  ✗ substrate Field: {e}")


def memory_estimate_dense(nx: int, ny: int, nz: int, n_fields: int = 5) -> float:
    """Memory for dense storage in GB."""
    return nx * ny * nz * n_fields * 4 / 1e9


def memory_estimate_qtt(n_qubits: int, max_rank: int, n_fields: int = 5) -> float:
    """Memory for QTT storage in GB."""
    # Each field: n_qubits cores, each core is (r, 2, r) ≈ 2r² floats
    return n_qubits * 2 * max_rank**2 * n_fields * 4 / 1e9


def main():
    print("\n" + "="*70)
    print("NIELSEN BENCHMARK - QTT NATIVE ARCHITECTURE")
    print("="*70)
    
    # Nielsen room geometry
    L, W, H = 9.0, 3.0, 3.0  # meters
    inlet_height = 0.168
    U_in = 0.455
    nu = 1.5e-5
    Re = U_in * inlet_height / nu
    
    # Target: ~1 billion cells
    # Grid: 1024 × 1024 × 1024 = 1.07 billion cells
    # Requires: 10 qubits per dimension (2^10 = 1024)
    qubits_per_dim = 10
    grid_size = 2**qubits_per_dim
    total_cells = grid_size**3
    total_qubits = 3 * qubits_per_dim  # For 3D Morton interleaving
    
    # QTT parameters
    max_rank = 32  # Typical rank for smooth HVAC flows
    
    print(f"\nPhysics:")
    print(f"  Domain: {L}m × {W}m × {H}m")
    print(f"  Inlet height: {inlet_height}m")
    print(f"  Inlet velocity: {U_in} m/s")
    print(f"  Reynolds number: {Re:.0f}")
    
    print(f"\nGrid:")
    print(f"  Grid size: {grid_size}³ = {total_cells:,} cells ({total_cells/1e9:.2f} billion)")
    print(f"  Qubits per dimension: {qubits_per_dim}")
    print(f"  Total QTT qubits: {total_qubits}")
    print(f"  Resolution: dx = {L/grid_size*1000:.3f} mm")
    
    print(f"\nMemory comparison:")
    dense_mem = memory_estimate_dense(grid_size, grid_size, grid_size, n_fields=5)
    qtt_mem = memory_estimate_qtt(total_qubits, max_rank, n_fields=5)
    compression = dense_mem / qtt_mem
    print(f"  Dense: {dense_mem:.1f} GB")
    print(f"  QTT (rank {max_rank}): {qtt_mem*1000:.2f} MB")
    print(f"  Compression ratio: {compression:.0f}×")
    
    # =========================================================================
    # RUN QTT NAVIER-STOKES SOLVER
    # =========================================================================
    if HAS_NS_SOLVER:
        print("\n" + "="*70)
        print("RUNNING 3D QTT NAVIER-STOKES SOLVER")
        print("="*70)
        
        # Use a feasible grid size for demonstration
        # 64×64×64 for quick test, can scale up
        demo_qubits = 6  # 64³ = 262k cells
        
        config = NS3DConfig(
            qubits_x=demo_qubits,
            qubits_y=demo_qubits - 1,  # 32 points (room is narrower)
            qubits_z=demo_qubits - 1,  # 32 points
            Lx=L, Ly=W, Lz=H,
            nu=nu,
            max_rank=max_rank,
        )
        
        print(f"\nDemo configuration:")
        print(f"  Grid: {config.Nx} × {config.Ny} × {config.Nz} = {config.total_points:,} cells")
        print(f"  Dense memory would be: {config.dense_memory_gb*1000:.1f} MB")
        print(f"  QTT memory (estimated): {config.qtt_memory_mb:.2f} MB")
        
        # Create solver
        solver = NS3DSolver(config)
        
        # Create initial state with inlet jet
        print("\nCreating initial state with inlet jet...")
        state = create_inlet_jet_state(config, U_inlet=U_in)
        
        # Run simulation
        n_steps = 10
        print(f"\nRunning {n_steps} time steps...")
        
        total_time = 0.0
        for i in range(n_steps):
            dt = solver.compute_dt(state)
            t0 = time.time()
            state = solver.step(state, dt)
            step_time = time.time() - t0
            total_time += step_time
            
            chi = state.chi_report()
            print(f"  Step {i+1}: t={solver.time:.4f}s, χ_max={chi['chi_max']}, "
                  f"mem={chi['memory_mb']:.2f}MB, step_time={step_time*1000:.0f}ms")
        
        print(f"\nCompleted {n_steps} steps in {total_time:.1f}s")
        print(f"Average: {total_time/n_steps*1000:.0f} ms/step")
        
        # Final χ-regularity report
        print("\n" + "="*70)
        print("χ-REGULARITY REPORT (NS-MILLENNIUM)")
        print("="*70)
        final_chi = state.chi_report()
        print(f"  χ_u = {final_chi['chi_u']}")
        print(f"  χ_v = {final_chi['chi_v']}")
        print(f"  χ_w = {final_chi['chi_w']}")
        print(f"  χ_p = {final_chi['chi_p']}")
        print(f"  χ_max = {final_chi['chi_max']}")
        print(f"  Total memory: {final_chi['memory_mb']:.2f} MB")
        
        # Scale to 1B cells
        print("\n" + "="*70)
        print("SCALING TO 1 BILLION CELLS")
        print("="*70)
        
        # Memory scales with qubits (log N), not cells
        qubits_1B = 30  # 10 qubits × 3 dimensions
        qubits_demo = config.total_qubits
        memory_scale = qubits_1B / qubits_demo
        
        print(f"  Demo: {config.total_points:,} cells, {qubits_demo} qubits, {final_chi['memory_mb']:.2f} MB")
        print(f"  1B:   1,073,741,824 cells, {qubits_1B} qubits")
        print(f"  Projected 1B memory: {final_chi['memory_mb'] * memory_scale:.1f} MB")
        print(f"  Dense 1B memory: {dense_mem*1000:.0f} MB")
        print(f"  QTT compression advantage: {dense_mem*1000 / (final_chi['memory_mb'] * memory_scale):.0f}×")
        
    else:
        print("\n" + "="*70)
        print("QTT NS SOLVER NOT AVAILABLE")
        print("="*70)
        print("  qtt_ns_3d.py not found or failed to import")


if __name__ == '__main__':
    main()
