"""
Visual Validation: Mass Conservation Check

This script verifies that the Brinkman penalization is causing flow DIVERSION,
not mass DELETION. If mass is conserved, we should see:

1. Velocity = 0 inside the column (blocking works)
2. Velocity INCREASES on the sides (Venturi acceleration)
3. Low velocity wake behind the column

If mass is being deleted, we'd see velocity just drop to zero everywhere
downstream with no acceleration on the sides.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from hyper_grid import HyperGrid
from fvm_porous import create_porous_solver

def run_column_simulation():
    """Run the column obstruction case and return solver + grid."""
    print("Setting up column simulation...")
    
    grid = HyperGrid(
        nx=128, ny=64, nz=64,
        lx=9.0, ly=3.0, lz=3.0,
        device='cuda'
    )
    
    # Floor-to-ceiling column at x=4.5m
    grid.add_box(
        x_min=4.0, x_max=5.0,  # 1m wide
        y_min=0.5, y_max=2.5,  # 2m deep, centered
        z_min=0.0, z_max=3.0   # Floor to ceiling
    )
    
    solver = create_porous_solver(grid, device='cuda', dtype=torch.float32)
    solver.step = torch.compile(solver.step)
    
    # Warmup
    dt = 0.002
    for _ in range(5):
        solver.step(dt)
    
    # Run 80s physical time
    print("Running simulation (80s physical time)...")
    t_end = 80.0
    n_steps = int(t_end / dt)
    
    for step in range(n_steps):
        solver.step(dt)
        if step % 10000 == 0:
            print(f"  Step {step}/{n_steps} ({step*dt:.0f}s)")
    
    torch.cuda.synchronize()
    print("Simulation complete.")
    
    return solver, grid


def create_visualization(solver, grid, filename_prefix="column_flow"):
    """Generate PyVista visualization."""
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed. Run: pip install pyvista")
        return
    
    print("Generating visualization...")
    
    # Convert to numpy
    u = solver.u.detach().cpu().numpy()
    v = solver.v.detach().cpu().numpy()
    w = solver.w.detach().cpu().numpy()
    vol_frac = grid.vol_frac.detach().cpu().numpy()
    
    nx, ny, nz = u.shape
    
    # Create structured grid
    x = np.linspace(0, grid.lx, nx + 1)
    y = np.linspace(0, grid.ly, ny + 1)
    z = np.linspace(0, grid.lz, nz + 1)
    
    grid_pv = pv.RectilinearGrid(x, y, z)
    
    # Add cell data (flatten in Fortran order for PyVista)
    grid_pv.cell_data["u"] = u.flatten(order='F')
    grid_pv.cell_data["v"] = v.flatten(order='F')
    grid_pv.cell_data["w"] = w.flatten(order='F')
    grid_pv.cell_data["vol_frac"] = vol_frac.flatten(order='F')
    
    # Velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2 + w**2)
    grid_pv.cell_data["velocity_mag"] = vel_mag.flatten(order='F')
    
    # Save full 3D data
    grid_pv.save(f"{filename_prefix}.vtr")
    print(f"Saved: {filename_prefix}.vtr")
    
    # Generate slices for quick viewing
    try:
        pv.start_xvfb()  # For headless rendering
    except (OSError, RuntimeError):
        pass  # X virtual framebuffer not available
    
    plotter = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1600, 1200))
    
    # === Slice 1: Horizontal slice at ceiling level (z = 2.9m) ===
    plotter.subplot(0, 0)
    slice_ceil = grid_pv.slice(normal='z', origin=(4.5, 1.5, 2.9))
    plotter.add_mesh(slice_ceil, scalars="velocity_mag", cmap="jet", 
                     clim=[0, 0.5], show_scalar_bar=True)
    plotter.add_text("Ceiling Level (z=2.9m)", font_size=10)
    plotter.view_xy()
    
    # === Slice 2: Vertical slice through column center (y = 1.5m) ===
    plotter.subplot(0, 1)
    slice_mid = grid_pv.slice(normal='y', origin=(4.5, 1.5, 1.5))
    plotter.add_mesh(slice_mid, scalars="velocity_mag", cmap="jet",
                     clim=[0, 0.5], show_scalar_bar=True)
    # Show column outline
    column = grid_pv.threshold([0.0, 0.5], scalars="vol_frac")
    if column.n_cells > 0:
        plotter.add_mesh(column.outline(), color='white', line_width=2)
    plotter.add_text("Side View (y=1.5m)", font_size=10)
    plotter.view_xz()
    
    # === Slice 3: Cross-section AT the column (x = 4.5m) ===
    plotter.subplot(1, 0)
    slice_at_col = grid_pv.slice(normal='x', origin=(4.5, 1.5, 1.5))
    plotter.add_mesh(slice_at_col, scalars="velocity_mag", cmap="jet",
                     clim=[0, 0.5], show_scalar_bar=True)
    plotter.add_text("At Column (x=4.5m)", font_size=10)
    plotter.view_yz()
    
    # === Slice 4: Cross-section AFTER column (x = 6.0m) ===
    plotter.subplot(1, 1)
    slice_after = grid_pv.slice(normal='x', origin=(6.0, 1.5, 1.5))
    plotter.add_mesh(slice_after, scalars="velocity_mag", cmap="jet",
                     clim=[0, 0.5], show_scalar_bar=True)
    plotter.add_text("After Column (x=6.0m)", font_size=10)
    plotter.view_yz()
    
    # Save screenshot
    plotter.screenshot(f"{filename_prefix}.png")
    print(f"Saved: {filename_prefix}.png")
    
    plotter.close()
    
    return grid_pv


def check_mass_conservation(solver, grid):
    """Quantitative mass conservation check."""
    print("\n" + "=" * 60)
    print("MASS CONSERVATION CHECK")
    print("=" * 60)
    
    cfg = solver.config
    u = solver.u
    
    # Sample at different x positions AROUND the column
    # Column is at x = 4.0 to 5.0
    
    positions = [2.0, 3.0, 3.5, 5.5, 6.0, 7.0]
    
    print(f"\n{'x (m)':>8} {'Mass Flux (m³/s)':>18} {'Notes':>20}")
    print("-" * 50)
    
    fluxes = []
    for x in positions:
        ix = min(int(x / cfg.dx), cfg.nx - 1)
        
        # Integrate u over the cross-section (fluid cells only)
        u_slice = u[ix, :, :]
        vol_slice = grid.vol_frac[ix, :, :]
        
        # Mass flux = integral of u * dA (only in fluid cells)
        flux = (u_slice * vol_slice * cfg.dy * cfg.dz).sum().item()
        fluxes.append(flux)
        
        note = ""
        if x < 4.0:
            note = "upstream"
        elif x >= 4.0 and x <= 5.0:
            note = "IN COLUMN"
        else:
            note = "downstream"
        
        print(f"{x:>8.1f} {flux:>18.4f} {note:>20}")
    
    # Check conservation
    upstream_flux = fluxes[0]  # x=2.0
    downstream_flux = fluxes[-1]  # x=7.0
    
    print("-" * 50)
    
    if upstream_flux > 0.01:
        conservation_error = abs(downstream_flux - upstream_flux) / upstream_flux * 100
        print(f"\nUpstream flux (x=2m):   {upstream_flux:.4f} m³/s")
        print(f"Downstream flux (x=7m): {downstream_flux:.4f} m³/s")
        print(f"Conservation error:     {conservation_error:.1f}%")
        
        if conservation_error < 20:
            print("\n✓ MASS APPROXIMATELY CONSERVED")
            print("  Flow is diverting around column, not vanishing")
        else:
            print("\n⚠ MASS CONSERVATION VIOLATION")
            print("  Flow may be getting 'deleted' at column face")
    else:
        print("\n○ Upstream flux too low to measure (jet not fully developed)")
    
    # Check for Venturi acceleration at column sides
    print("\n" + "-" * 50)
    print("VENTURI EFFECT CHECK (acceleration at column sides)")
    print("-" * 50)
    
    # At x=4.5 (middle of column), sample at y=0 and y=3 (outside column)
    ix_col = int(4.5 / cfg.dx)
    
    # Column is y=0.5 to 2.5, so check at y=0.2 and y=2.8
    iy_left = int(0.2 / cfg.dy)
    iy_right = int(2.8 / cfg.dy)
    iy_center = cfg.ny // 2
    
    # Sample at ceiling level
    iz_ceil = -1
    
    u_left = u[ix_col, iy_left, iz_ceil].item()
    u_center = u[ix_col, iy_center, iz_ceil].item()
    u_right = u[ix_col, iy_right, iz_ceil].item()
    
    # Reference: upstream velocity
    u_upstream = u[int(2.0 / cfg.dx), iy_center, iz_ceil].item()
    
    print(f"Upstream reference (x=2m, center): {u_upstream:.4f} m/s")
    print(f"At column (x=4.5m):")
    print(f"  Left side (y=0.2m):  {u_left:.4f} m/s")
    print(f"  Center (y=1.5m):     {u_center:.4f} m/s {'← BLOCKED' if u_center < 0.01 else ''}")
    print(f"  Right side (y=2.8m): {u_right:.4f} m/s")
    
    if u_left > u_upstream * 1.1 or u_right > u_upstream * 1.1:
        print("\n✓ VENTURI ACCELERATION DETECTED")
        print("  Flow is speeding up around the column sides")
    elif u_left > 0.1 or u_right > 0.1:
        print("\n○ Some side flow, but not accelerated")
    else:
        print("\n⚠ NO SIDE FLOW")
        print("  Mass may be vanishing at column face")
    
    print("=" * 60)


if __name__ == '__main__':
    solver, grid = run_column_simulation()
    check_mass_conservation(solver, grid)
    
    try:
        create_visualization(solver, grid)
    except Exception as e:
        print(f"Visualization skipped: {e}")
