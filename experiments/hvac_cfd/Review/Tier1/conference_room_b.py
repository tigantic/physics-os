"""
Conference Room B - Full HVAC CFD Simulation

This is the culmination of the HyperFOAM pipeline:
- STL/Box geometry → HyperGrid voxelization
- Porous Navier-Stokes with Brinkman penalization
- Pressure projection for incompressibility
- torch.compile for GPU kernel fusion

Room Layout (Top View):
┌────────────────────────────────────────────────────────┐
│  [SUPPLY]                                    [SUPPLY]  │  ← Ceiling vents
│                                                        │
│     ┌──────────────────────────────────────────┐       │
│     │                                          │       │
│     │   ○   ○   ○   ○   ○   ○                 │       │
│     │                                          │       │
│     │          CONFERENCE TABLE               │       │
│     │                                          │       │
│     │   ○   ○   ○   ○   ○   ○                 │       │
│     │                                          │       │
│     └──────────────────────────────────────────┘       │
│                                                        │
│  [RETURN]                                    [RETURN]  │  ← Floor returns
└────────────────────────────────────────────────────────┘

Dimensions:
- Room: 30 ft × 20 ft × 10 ft (9.14m × 6.10m × 3.05m)
- Table: 12 ft × 4 ft × 2.5 ft (centered)
- Supply vents: 2 × ceiling diffusers, 450 FPM (2.29 m/s)
- Return vents: 2 × floor grilles
"""

import torch
import time
import numpy as np
from dataclasses import dataclass

# Local imports
from hyper_grid import HyperGrid
from hyperfoam_solver import HyperFoamSolver, ProjectionConfig


@dataclass
class ConferenceRoomConfig:
    """Conference Room B specifications"""
    # Room dimensions (meters)
    room_length: float = 9.14   # 30 ft
    room_width: float = 6.10    # 20 ft
    room_height: float = 3.05   # 10 ft
    
    # Grid resolution
    nx: int = 128
    ny: int = 96
    nz: int = 48
    
    # Table dimensions (meters)
    table_length: float = 3.66  # 12 ft
    table_width: float = 1.22   # 4 ft
    table_height: float = 0.76  # 2.5 ft
    
    # HVAC specs
    supply_velocity: float = 2.29  # 450 FPM in m/s
    supply_size: float = 0.61      # 2 ft × 2 ft diffuser
    
    # Simulation
    dt: float = 0.002
    

def setup_conference_room(cfg: ConferenceRoomConfig):
    """Build the Conference Room B geometry"""
    
    print("=" * 70)
    print("CONFERENCE ROOM B - HyperFOAM Simulation")
    print("=" * 70)
    
    # 1. Create Grid
    print("\n[1] Creating HyperGrid...")
    grid = HyperGrid(
        nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
        lx=cfg.room_length, ly=cfg.room_width, lz=cfg.room_height,
        device='cuda'
    )
    print(f"    Grid: {cfg.nx}×{cfg.ny}×{cfg.nz} = {cfg.nx*cfg.ny*cfg.nz:,} cells")
    print(f"    Resolution: dx={grid.dx*100:.1f}cm, dy={grid.dy*100:.1f}cm, dz={grid.dz*100:.1f}cm")
    
    # 2. Add Conference Table (centered)
    print("\n[2] Adding Conference Table...")
    table_x_min = (cfg.room_length - cfg.table_length) / 2
    table_x_max = table_x_min + cfg.table_length
    table_y_min = (cfg.room_width - cfg.table_width) / 2
    table_y_max = table_y_min + cfg.table_width
    table_z_min = 0.0
    table_z_max = cfg.table_height
    
    grid.add_box_obstacle(
        table_x_min, table_x_max,
        table_y_min, table_y_max,
        table_z_min, table_z_max
    )
    print(f"    Table: {cfg.table_length:.2f}m × {cfg.table_width:.2f}m × {cfg.table_height:.2f}m")
    print(f"    Position: [{table_x_min:.2f}, {table_x_max:.2f}] × [{table_y_min:.2f}, {table_y_max:.2f}]")
    
    # 3. Add 12 Chairs (6 per side)
    print("\n[3] Adding 12 Chairs...")
    chair_width = 0.5   # 20 inches
    chair_depth = 0.5   # 20 inches  
    chair_height = 0.45 # seat height
    chair_spacing = cfg.table_length / 7  # Distribute 6 chairs with margins
    
    chairs_added = 0
    for i in range(6):
        chair_x = table_x_min + chair_spacing * (i + 0.5)
        
        # Front row (y < table)
        grid.add_box_obstacle(
            chair_x - chair_width/2, chair_x + chair_width/2,
            table_y_min - chair_depth - 0.1, table_y_min - 0.1,
            0.0, chair_height
        )
        chairs_added += 1
        
        # Back row (y > table)
        grid.add_box_obstacle(
            chair_x - chair_width/2, chair_x + chair_width/2,
            table_y_max + 0.1, table_y_max + chair_depth + 0.1,
            0.0, chair_height
        )
        chairs_added += 1
    
    print(f"    Added {chairs_added} chairs")
    
    return grid


def create_solver(grid, cfg: ConferenceRoomConfig):
    """Create the HyperFoam solver with HVAC boundary conditions"""
    
    print("\n[4] Initializing Solver...")
    
    # Create solver config
    solver_cfg = ProjectionConfig(
        nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
        Lx=cfg.room_length, Ly=cfg.room_width, Lz=cfg.room_height,
        dt=cfg.dt,
        inlet_velocity=cfg.supply_velocity
    )
    
    solver = HyperFoamSolver(grid, solver_cfg)
    
    # Setup supply vents (ceiling, blowing down)
    # Vent 1: Front-left ceiling
    # Vent 2: Back-right ceiling
    print("\n[5] Configuring HVAC Vents...")
    
    vent_half = int(cfg.supply_size / grid.dx / 2)
    
    # Vent positions (in grid indices)
    vents = [
        # (x_center, y_center) in meters
        (cfg.room_length * 0.25, cfg.room_width * 0.25),  # Front-left
        (cfg.room_length * 0.75, cfg.room_width * 0.75),  # Back-right
    ]
    
    solver.supply_vents = []
    for i, (vx, vy) in enumerate(vents):
        ix = int(vx / grid.dx)
        iy = int(vy / grid.dy)
        iz_top = cfg.nz - 2  # Near ceiling
        
        solver.supply_vents.append({
            'ix': (ix - vent_half, ix + vent_half),
            'iy': (iy - vent_half, iy + vent_half),
            'iz': iz_top,
            'velocity': -cfg.supply_velocity  # Negative = downward
        })
        print(f"    Supply {i+1}: ({vx:.1f}m, {vy:.1f}m) @ ceiling, {cfg.supply_velocity:.2f} m/s down")
    
    # Return vents (floor level, extracting)
    returns = [
        (cfg.room_length * 0.25, cfg.room_width * 0.75),  # Front-right
        (cfg.room_length * 0.75, cfg.room_width * 0.25),  # Back-left
    ]
    
    solver.return_vents = []
    for i, (vx, vy) in enumerate(returns):
        ix = int(vx / grid.dx)
        iy = int(vy / grid.dy)
        
        solver.return_vents.append({
            'ix': (ix - vent_half, ix + vent_half),
            'iy': (iy - vent_half, iy + vent_half),
            'iz': 1,  # Floor level
        })
        print(f"    Return {i+1}: ({vx:.1f}m, {vy:.1f}m) @ floor")
    
    return solver


def apply_hvac_bc(solver):
    """Apply HVAC boundary conditions each step"""
    # Supply vents: inject downward velocity
    for vent in solver.supply_vents:
        ix0, ix1 = vent['ix']
        iy0, iy1 = vent['iy']
        iz = vent['iz']
        # w = downward velocity
        solver.w[ix0:ix1, iy0:iy1, iz] = vent['velocity']
    
    # Return vents: zero pressure (natural outflow)
    # Handled by outlet BC in pressure solve


def run_simulation(solver, cfg: ConferenceRoomConfig, duration: float = 30.0):
    """Run the CFD simulation"""
    
    n_steps = int(duration / cfg.dt)
    
    print("\n" + "=" * 70)
    print(f"SIMULATION: {duration}s physical time, {n_steps:,} steps")
    print("=" * 70)
    
    # Warmup (JIT compilation)
    print("\n[6] JIT Compilation (one-time)...")
    t0 = time.perf_counter()
    for _ in range(5):
        apply_hvac_bc(solver)
        solver.step()
    torch.cuda.synchronize()
    print(f"    Compiled in {time.perf_counter() - t0:.1f}s")
    
    # Main simulation loop
    print("\n[7] Running simulation...")
    
    start = time.perf_counter()
    report_interval = n_steps // 10
    
    for step in range(n_steps):
        apply_hvac_bc(solver)
        solver.step()
        
        if step % report_interval == 0 and step > 0:
            t = step * cfg.dt
            
            # Sample velocities
            # Table top center
            ix_c = cfg.nx // 2
            iy_c = cfg.ny // 2
            iz_table = int(cfg.table_height / (cfg.room_height / cfg.nz)) + 2
            
            u_table = solver.u[ix_c, iy_c, iz_table].item()
            v_table = solver.v[ix_c, iy_c, iz_table].item()
            w_table = solver.w[ix_c, iy_c, iz_table].item()
            speed_table = np.sqrt(u_table**2 + v_table**2 + w_table**2)
            
            # Breathing zone (1.2m above floor, near chairs)
            iz_breath = int(1.2 / (cfg.room_height / cfg.nz))
            u_breath = solver.u[ix_c, cfg.ny//4, iz_breath].item()
            v_breath = solver.v[ix_c, cfg.ny//4, iz_breath].item()
            w_breath = solver.w[ix_c, cfg.ny//4, iz_breath].item()
            speed_breath = np.sqrt(u_breath**2 + v_breath**2 + w_breath**2)
            
            elapsed = time.perf_counter() - start
            rate = step / elapsed
            
            print(f"    t={t:5.1f}s | Table: {speed_table:.3f} m/s | Breath Zone: {speed_breath:.3f} m/s | {rate:.0f} step/s")
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    
    print("\n" + "=" * 50)
    print("SIMULATION COMPLETE")
    print("=" * 50)
    print(f"Physical time:  {duration:.1f}s")
    print(f"Wall time:      {total_time:.1f}s")
    print(f"Realtime ratio: {duration/total_time:.2f}× realtime")
    print(f"Average speed:  {n_steps/total_time:.0f} steps/sec")
    
    return solver


def analyze_comfort(solver, cfg: ConferenceRoomConfig):
    """Analyze thermal comfort metrics"""
    
    print("\n" + "=" * 70)
    print("COMFORT ANALYSIS")
    print("=" * 70)
    
    # Calculate velocity magnitude
    speed = torch.sqrt(solver.u**2 + solver.v**2 + solver.w**2)
    
    # Breathing zone: 1.0 - 1.8m above floor
    iz_low = int(1.0 / (cfg.room_height / cfg.nz))
    iz_high = int(1.8 / (cfg.room_height / cfg.nz))
    
    breath_zone = speed[:, :, iz_low:iz_high]
    
    avg_speed = breath_zone.mean().item()
    max_speed = breath_zone.max().item()
    
    print(f"\nBreathing Zone (1.0-1.8m):")
    print(f"  Average velocity: {avg_speed:.3f} m/s ({avg_speed*196.85:.0f} FPM)")
    print(f"  Maximum velocity: {max_speed:.3f} m/s ({max_speed*196.85:.0f} FPM)")
    
    # ASHRAE 55 comfort criteria
    # Draft risk: v > 0.25 m/s at occupied zone
    draft_risk = (breath_zone > 0.25).float().mean().item() * 100
    
    print(f"\nASHRAE 55 Compliance:")
    if avg_speed < 0.25:
        print(f"  ✓ Average velocity < 0.25 m/s (acceptable)")
    else:
        print(f"  ✗ Average velocity > 0.25 m/s (too drafty)")
    
    print(f"  Draft risk: {draft_risk:.1f}% of breathing zone exceeds 0.25 m/s")
    
    # Table surface velocity (papers flying?)
    iz_table = int(cfg.table_height / (cfg.room_height / cfg.nz)) + 1
    table_speed = speed[:, :, iz_table]
    max_table = table_speed.max().item()
    
    print(f"\nTable Surface:")
    print(f"  Maximum velocity: {max_table:.3f} m/s ({max_table*196.85:.0f} FPM)")
    if max_table < 0.5:
        print(f"  ✓ Papers won't blow away")
    else:
        print(f"  ⚠ Risk of papers blowing")


def export_visualization(solver, cfg: ConferenceRoomConfig, filename="conference_room_b.vts"):
    """Export to VTK for ParaView visualization"""
    try:
        import pyvista as pv
        
        print(f"\n[8] Exporting to {filename}...")
        
        # Create structured grid
        x = np.linspace(0, cfg.room_length, cfg.nx)
        y = np.linspace(0, cfg.room_width, cfg.ny)
        z = np.linspace(0, cfg.room_height, cfg.nz)
        
        grid_vtk = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing='ij'))
        
        # Add fields
        u = solver.u.cpu().numpy().flatten(order='F')
        v = solver.v.cpu().numpy().flatten(order='F')
        w = solver.w.cpu().numpy().flatten(order='F')
        
        grid_vtk.point_data['velocity'] = np.column_stack([u, v, w])
        grid_vtk.point_data['speed'] = np.sqrt(u**2 + v**2 + w**2)
        grid_vtk.point_data['pressure'] = solver.p.cpu().numpy().flatten(order='F')
        grid_vtk.point_data['vol_frac'] = solver.grid.geo[0].cpu().numpy().flatten(order='F')
        
        grid_vtk.save(filename)
        print(f"    Saved: {filename}")
        print("    Open with: paraview conference_room_b.vts")
        
    except ImportError:
        print("\n[8] PyVista not installed, skipping VTK export")
        print("    Install with: pip install pyvista")


def main():
    """Main entry point"""
    
    # Configuration
    cfg = ConferenceRoomConfig()
    
    # Build geometry
    grid = setup_conference_room(cfg)
    
    # Create solver
    solver = create_solver(grid, cfg)
    
    # Run simulation
    solver = run_simulation(solver, cfg, duration=30.0)
    
    # Analyze comfort
    analyze_comfort(solver, cfg)
    
    # Export for visualization
    export_visualization(solver, cfg)
    
    print("\n" + "=" * 70)
    print("CONFERENCE ROOM B SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
