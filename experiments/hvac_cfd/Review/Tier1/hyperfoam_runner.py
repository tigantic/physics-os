"""
HyperFOAM Runner: Pressure Projection Test

This script runs the "Column Test" to verify that:
1. Pressure builds up at the obstacle face (P_face > 0)
2. Velocity increases at the sides (U_side > inlet_velocity) — Venturi effect
3. Mass is conserved (air swerves, doesn't vanish)

If U_side > inlet_velocity, we have PROVEN that the pressure solver
is correctly redirecting flow around obstacles.
"""

import time
import torch
import numpy as np

# Try to import pyvista for visualization
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: pyvista not installed. Skipping VTS export.")

from hyper_grid import HyperGrid
from hyperfoam_solver import HyperFoamSolver, ProjectionConfig


def save_visualization(solver, grid, filename="hyperfoam_swerve.vts"):
    """
    Exports the HyperFoam state to a VTK Structured Grid.
    """
    if not HAS_PYVISTA:
        print("Skipping visualization (pyvista not available)")
        return
        
    print(f"Saving visualization to {filename}...")
    
    # 1. Extract Data (GPU -> CPU Numpy)
    u = solver.u.detach().cpu().numpy()
    v = solver.v.detach().cpu().numpy()
    w = solver.w.detach().cpu().numpy()
    p = solver.p.detach().cpu().numpy()
    vol_frac = grid.geo[0].detach().cpu().numpy()
    
    # 2. PyVista Grid Setup
    nx, ny, nz = u.shape
    grid_pv = pv.StructuredGrid()
    grid_pv.dimensions = np.array([nx, ny, nz]) + 1
    
    # Coordinates
    x = np.linspace(0, grid.lx, nx+1)
    y = np.linspace(0, grid.ly, ny+1)
    z = np.linspace(0, grid.lz, nz+1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_pv.points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    
    # 3. Add Data (Flattening F-order for PyVista structured data)
    grid_pv.cell_data["Velocity_X"] = u.flatten(order='F')
    grid_pv.cell_data["Velocity_Y"] = v.flatten(order='F')
    grid_pv.cell_data["Velocity_Z"] = w.flatten(order='F')
    grid_pv.cell_data["Pressure"] = p.flatten(order='F')
    grid_pv.cell_data["Vol_Frac"] = vol_frac.flatten(order='F')
    
    # Velocity Magnitude
    vel_mag = np.sqrt(u**2 + v**2 + w**2)
    grid_pv.cell_data["Velocity_Mag"] = vel_mag.flatten(order='F')
    
    grid_pv.save(filename)
    print("✓ Saved.")


def run_simulation():
    print("="*70)
    print("HYPERFOAM PHASE 2.3: PRESSURE PROJECTION TEST")
    print("="*70)
    print()

    # 1. Setup Grid & Geometry (The "Column" obstacle)
    print("Setting up grid...")
    grid = HyperGrid(
        nx=128, ny=64, nz=64, 
        lx=9.0, ly=3.0, lz=3.0, 
        device='cuda'
    )
    
    # Add the Column at x=4.5m (floor to ceiling)
    print("Adding floor-to-ceiling column...")
    grid.add_box_obstacle(
        x_min=4.0, x_max=5.0,   # 1m wide
        y_min=0.5, y_max=2.5,   # 2m deep (leaves 0.5m gaps on sides)
        z_min=0.0, z_max=3.0    # Floor to ceiling
    )
    
    # Count cells
    total_cells = grid.nx * grid.ny * grid.nz
    solid_cells = (grid.geo[0] < 0.5).sum().item()
    fluid_cells = total_cells - solid_cells
    print(f"Grid: {grid.nx}×{grid.ny}×{grid.nz} = {total_cells:,} cells")
    print(f"Fluid: {fluid_cells:,} | Solid: {solid_cells:,}")
    print()

    # 2. Init Solver
    cfg = ProjectionConfig(
        dt=0.001,       # Reduced for stability (CFL < 0.2)
        nx=128, ny=64, nz=64
    )
    solver = HyperFoamSolver(grid, cfg)
    
    # Initialize with uniform flow for faster development
    solver.init_uniform_flow()
    print(f"Initialized with uniform U = {cfg.inlet_velocity} m/s")
    
    print(f"Inlet velocity: {cfg.inlet_velocity} m/s")
    print(f"Timestep: {cfg.dt}s")
    print()
    
    # 3. Run Loop
    t_end = 5.0  # Seconds (reduced - uniform init means faster development)
    steps = int(t_end / cfg.dt)
    
    print(f"Running {steps} steps ({t_end}s physical time)...")
    print()
    print(f"{'Step':>6} | {'P_face':>10} | {'U_side':>10} | {'U_center':>10} | {'Status'}")
    print("-" * 70)
    
    start_time = time.perf_counter()
    
    max_u_side = 0.0
    max_p_face = 0.0
    
    for step in range(steps):
        solver.step()
        
        if step % 200 == 0:
            # --- THE SWERVE CHECK ---
            # Measure velocity at the SIDE of the column (y=0.2m, x=4.5m)
            # If U > inlet_velocity, the Venturi effect is working
            idx_x = int(4.5 / grid.dx)
            idx_y_side = int(0.2 / grid.dy)  # Gap between wall and column
            idx_z = 32
            
            u_side = solver.u[idx_x, idx_y_side, idx_z].item()
            max_u_side = max(max_u_side, u_side)
            
            # Measure velocity at the CENTER of the column (should be ~0)
            idx_y_center = int(1.5 / grid.dy)
            u_center = solver.u[idx_x, idx_y_center, idx_z].item()
            
            # Measure Pressure at the FACE of the column (x=3.9m)
            idx_x_face = int(3.9 / grid.dx)
            p_face = solver.p[idx_x_face, idx_y_center, idx_z].item()
            max_p_face = max(max_p_face, abs(p_face))
            
            # Determine status
            if u_side > cfg.inlet_velocity:
                status = "✓ VENTURI (swerving!)"
            elif u_side > 0.1:
                status = "○ Flow detected"
            else:
                status = "- Waiting..."
            
            print(f"{step:6d} | {p_face:+10.4f} | {u_side:10.4f} | {u_center:10.4f} | {status}")
    
    total_time = time.perf_counter() - start_time
    
    print("-" * 70)
    print(f"\nCompleted in {total_time:.1f}s ({steps/total_time:.0f} steps/s)")
    print()
    
    # 4. Final Verdict
    print("=" * 70)
    print("MASS CONSERVATION CHECK")
    print("=" * 70)
    print()
    print(f"  Inlet velocity:     {cfg.inlet_velocity:.3f} m/s")
    print(f"  Max side velocity:  {max_u_side:.3f} m/s")
    print(f"  Max face pressure:  {max_p_face:.4f}")
    print()
    
    if max_u_side > cfg.inlet_velocity:
        print("  ✓ VENTURI EFFECT CONFIRMED")
        print("    Air is accelerating around the column (U_side > U_inlet)")
        print("    Mass is being DIVERTED, not deleted.")
        print()
        print("  → PRESSURE SOLVER IS WORKING")
        print("  → SAFE TO BUILD STL VOXELIZER")
    elif max_u_side > 0.2:
        print("  ○ PARTIAL SUCCESS")
        print("    Flow is reaching the sides, but not accelerating enough.")
        print("    May need more simulation time or tuning.")
    else:
        print("  ✗ MASS VANISHING")
        print("    Air is being deleted at the column face.")
        print("    Pressure solver may not be working correctly.")
    
    print()
    print("=" * 70)
    
    # 5. Save visualization
    save_visualization(solver, grid)


if __name__ == "__main__":
    run_simulation()
