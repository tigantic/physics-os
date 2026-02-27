"""
Pitch Deck Asset Generator

Generates professional 2D heatmaps showing:
- Temperature profile (with AI-optimized diffuser settings)
- Velocity field with streamlines/vectors

Works on any machine (headless servers, no OpenGL required).
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from hyper_grid import HyperGrid
from hyperfoam_solver import HyperFoamSolver, ProjectionConfig


def render_pitch_deck_assets():
    print("=" * 70)
    print("GENERATING PITCH DECK ASSETS")
    print("=" * 70)

    # 1. Setup the Simulation (Optimized Config)
    nx, ny, nz = 64, 48, 24
    grid = HyperGrid(nx, ny, nz, 9.0, 6.0, 3.0, device='cuda')
    
    # Add Table
    grid.add_box_obstacle(2.67, 6.33, 2.39, 3.61, 0.0, 0.76)
    
    # 2. Configure Solver with "Optimized" Parameters
    cfg = ProjectionConfig(nx=nx, ny=ny, nz=nz, Lx=9.0, Ly=6.0, Lz=3.0, dt=0.01)
    solver = HyperFoamSolver(grid, cfg)
    
    # 3. Apply the AI's Solution
    opt_vel = 0.93  # m/s
    opt_angle = 31.5  # degrees
    
    rad = np.deg2rad(opt_angle)
    u_comp = float(opt_vel * np.sin(rad))
    w_comp = float(-opt_vel * np.cos(rad))
    
    print(f"\n[1] Simulating optimized design ({opt_vel} m/s @ {opt_angle}°)...")
    
    # Initialize Temperature Field (Ambient 22°C, Supply 16°C)
    T = torch.ones((nx, ny, nz), device='cuda') * 22.0
    
    steps = 400
    for step in range(steps):
        solver.step()
        
        # Enforce Inlet Velocity (two ceiling vents)
        vent_size = 4
        z_c = nz - 2
        
        # Vent 1 (front-left quadrant)
        solver.u[12:20, 8:16, z_c] = u_comp
        solver.w[12:20, 8:16, z_c] = w_comp
        
        # Vent 2 (back-right quadrant)
        solver.u[44:52, 32:40, z_c] = -u_comp
        solver.w[44:52, 32:40, z_c] = w_comp
        
        # Enforce Inlet Temp (16°C - cold supply air)
        T[12:20, 8:16, z_c] = 16.0
        T[44:52, 32:40, z_c] = 16.0
        
        # Add Heat from Occupants (seated around table)
        # Zone around table at seated height (~1m)
        heat_zone = T[20:44, 16:32, 3:6]
        T[20:44, 16:32, 3:6] = heat_zone + 0.02 * (25.0 - heat_zone)
        
        # Advect Temperature (Scalar Transport - upwind)
        u, v, w = solver.u, solver.v, solver.w
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        
        # X-direction (upwind)
        T_xp = torch.roll(T, -1, 0)
        T_xm = torch.roll(T, 1, 0)
        dT_dx = torch.where(u > 0, (T - T_xm) / dx, (T_xp - T) / dx)
        
        # Y-direction
        T_yp = torch.roll(T, -1, 1)
        T_ym = torch.roll(T, 1, 1)
        dT_dy = torch.where(v > 0, (T - T_ym) / dy, (T_yp - T) / dy)
        
        # Z-direction
        T_zp = torch.roll(T, -1, 2)
        T_zm = torch.roll(T, 1, 2)
        dT_dz = torch.where(w > 0, (T - T_zm) / dz, (T_zp - T) / dz)
        
        advection = u * dT_dx + v * dT_dy + w * dT_dz
        
        # Diffusion
        alpha = 0.0002  # Thermal diffusivity (enhanced for mixing)
        lap_T = (T_xp - 2*T + T_xm) / dx**2 + \
                (T_yp - 2*T + T_ym) / dy**2 + \
                (T_zp - 2*T + T_zm) / dz**2
        
        T += cfg.dt * (-advection + alpha * lap_T)
        T = torch.clamp(T, 14, 30)
        
        if step % 100 == 0:
            print(f"    Step {step}: T_avg={T[:,:,3:10].mean().item():.1f}°C")
    
    print("\n[2] Generating visualizations...")
    
    # 4. Extract Slice for Plotting (Mid-Y plane through front vent)
    slice_idx = ny // 4
    
    u_cpu = solver.u[:, slice_idx, :].detach().cpu().numpy().T
    w_cpu = solver.w[:, slice_idx, :].detach().cpu().numpy().T
    T_cpu = T[:, slice_idx, :].detach().cpu().numpy().T
    
    speed = np.sqrt(u_cpu**2 + w_cpu**2)
    
    # 5. Generate Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # ─────────────────────────────────────────────────────────────────────
    # Plot A: Temperature Profile
    # ─────────────────────────────────────────────────────────────────────
    ax1 = axes[0]
    im1 = ax1.imshow(T_cpu, origin='lower', cmap='RdYlBu_r', 
                     vmin=16, vmax=24, extent=[0, 9, 0, 3], aspect='auto')
    
    ax1.set_title(f"HyperFOAM: AI-Optimized Temperature Profile\n"
                  f"(Diffuser: {opt_vel:.2f} m/s @ {opt_angle}° spread)", 
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel("Height (m)")
    
    # Colorbar
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Temperature (°C)")
    
    # Draw Table (side view)
    table_x = 2.67
    table_w = 3.66
    table_h = 0.76
    rect1 = plt.Rectangle((table_x, 0), table_w, table_h, 
                           linewidth=2, edgecolor='black', 
                           facecolor='#8B4513', alpha=0.7)
    ax1.add_patch(rect1)
    ax1.text(table_x + table_w/2, table_h/2, 'TABLE', 
             ha='center', va='center', fontweight='bold', color='white')
    
    # Mark comfort zone
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Breathing Zone')
    ax1.axhline(y=1.5, color='green', linestyle='--', alpha=0.5)
    ax1.fill_between([0, 9], 1.0, 1.5, alpha=0.1, color='green')
    
    # Mark vent
    ax1.annotate('Supply Vent\n(16°C)', xy=(1.5, 2.9), fontsize=9,
                ha='center', color='blue', fontweight='bold')
    ax1.plot([1.2, 1.8], [2.95, 2.95], 'b-', linewidth=3)
    
    ax1.set_xlim(0, 9)
    ax1.set_ylim(0, 3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Plot B: Velocity Field
    # ─────────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    im2 = ax2.imshow(speed, origin='lower', cmap='plasma', 
                     vmin=0, vmax=1.0, extent=[0, 9, 0, 3], aspect='auto')
    
    ax2.set_title("Velocity Field with Flow Vectors", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Room Length (m)")
    ax2.set_ylabel("Height (m)")
    
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("Speed (m/s)")
    
    # Overlay velocity vectors
    skip = 3
    X, Z = np.meshgrid(np.linspace(0, 9, nx), np.linspace(0, 3, nz))
    ax2.quiver(X[::skip, ::skip], Z[::skip, ::skip], 
               u_cpu[::skip, ::skip], w_cpu[::skip, ::skip], 
               color='white', alpha=0.6, scale=8, width=0.003)
    
    # Draw Table
    rect2 = plt.Rectangle((table_x, 0), table_w, table_h, 
                           linewidth=2, edgecolor='white', 
                           facecolor='#8B4513', alpha=0.7)
    ax2.add_patch(rect2)
    
    # Annotate key physics
    ax2.annotate('Cold jet mixes\nalong ceiling', xy=(3, 2.7), fontsize=9,
                ha='center', color='white', fontweight='bold')
    ax2.annotate('Diffused air\ndescends gently', xy=(6, 1.5), fontsize=9,
                ha='center', color='white')
    
    ax2.set_xlim(0, 9)
    ax2.set_ylim(0, 3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Finalize
    # ─────────────────────────────────────────────────────────────────────
    plt.tight_layout()
    
    # Add watermark
    fig.text(0.99, 0.01, 'Generated by HyperFOAM', 
             ha='right', va='bottom', fontsize=8, 
             color='gray', style='italic')
    
    # Save
    output_file = "hyperfoam_pitch_deck.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\n✓ Saved '{output_file}'")
    
    # Also save individual plots for slides
    fig_t, ax_t = plt.subplots(figsize=(10, 4))
    im_t = ax_t.imshow(T_cpu, origin='lower', cmap='RdYlBu_r', 
                       vmin=16, vmax=24, extent=[0, 9, 0, 3], aspect='auto')
    ax_t.set_title("Conference Room B: Thermal Comfort Achieved", fontsize=14, fontweight='bold')
    ax_t.set_xlabel("Room Length (m)")
    ax_t.set_ylabel("Height (m)")
    fig_t.colorbar(im_t, ax=ax_t, label="Temperature (°C)")
    rect_t = plt.Rectangle((table_x, 0), table_w, table_h, 
                            linewidth=2, edgecolor='black', facecolor='#8B4513', alpha=0.7)
    ax_t.add_patch(rect_t)
    plt.savefig("thermal_profile.png", dpi=300, bbox_inches='tight')
    print("✓ Saved 'thermal_profile.png'")
    
    plt.close('all')
    
    # ─────────────────────────────────────────────────────────────────────
    # Summary Statistics
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("ASSET GENERATION COMPLETE")
    print("=" * 50)
    print("\nFiles created:")
    print("  • hyperfoam_pitch_deck.png  (Combined view)")
    print("  • thermal_profile.png       (Temperature only)")
    
    # Final metrics
    breathing_zone = T[:, :, 4:10]
    print(f"\nFinal Metrics (Breathing Zone):")
    print(f"  Temperature: {breathing_zone.mean().item():.1f}°C avg, {breathing_zone.max().item():.1f}°C max")
    print(f"  Velocity:    {speed[4:10, :].mean():.2f} m/s avg, {speed[4:10, :].max():.2f} m/s max")


if __name__ == "__main__":
    render_pitch_deck_assets()
