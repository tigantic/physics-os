"""
Kelvin-Helmholtz Instability Animation

The "money shot" for CFD - visualizing vortex roll-up in QTT format.
Seeing a Rank-62 tensor swirl is indistinguishable from a dense simulation.

This script runs a full KH simulation and renders it to video.

Author: HyperTensor Team
Date: December 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import time
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ontic.cfd.euler2d_strang import (
    Euler2D_Strang, Euler2DConfig, Euler2DState, 
    create_kelvin_helmholtz_ic
)
from ontic.cfd.qtt_2d import qtt_2d_to_dense


def run_kh_simulation(n_bits: int = 7, t_final: float = 2.0, 
                      save_every: int = 5, max_rank: int = 64):
    """
    Run KH simulation and collect frames for animation.
    
    Args:
        n_bits: Bits per dimension (grid is 2^n x 2^n)
        t_final: Final simulation time
        save_every: Save a frame every N steps
        max_rank: Maximum QTT rank
        
    Returns:
        history: List of (t, rho_dense, u_dense, v_dense, rank)
    """
    N = 2 ** n_bits
    print(f"Running Kelvin-Helmholtz simulation on {N}x{N} grid...")
    
    config = Euler2DConfig(
        gamma=1.4,
        cfl=0.3,
        max_rank=max_rank,
        dtype=torch.float64,
        device=torch.device('cpu')
    )
    
    # Create solver and IC
    solver = Euler2D_Strang(n_bits, n_bits, config)
    state = create_kelvin_helmholtz_ic(n_bits, n_bits, config)
    
    # Get primitives from state
    def get_frame_data(state, t):
        rho, u, v, P = state.get_primitives(config.gamma)
        max_r = state.max_rank
        return (t, rho.numpy(), u.numpy(), v.numpy(), max_r)
    
    history = [get_frame_data(state, 0.0)]
    
    t = 0.0
    step = 0
    t_start = time.time()
    
    print(f"{'Step':>6}  {'Time':>8}  {'Rank':>6}  {'FPS':>8}")
    print("-" * 40)
    
    while t < t_final:
        dt = solver.compute_dt(state)
        if t + dt > t_final:
            dt = t_final - t
            
        state = solver.step(state, dt)
        t += dt
        step += 1
        
        # Save frame periodically
        if step % save_every == 0:
            history.append(get_frame_data(state, t))
            
            elapsed = time.time() - t_start
            fps = step / elapsed if elapsed > 0 else 0
            print(f"{step:6d}  {t:8.4f}  {state.max_rank:6d}  {fps:8.1f}")
    
    # Save final frame
    if step % save_every != 0:
        history.append(get_frame_data(state, t))
    
    elapsed = time.time() - t_start
    print("-" * 40)
    print(f"Simulation complete: {step} steps in {elapsed:.1f}s")
    print(f"Captured {len(history)} frames")
    
    return history


def render_kh_instability(history, filename="kelvin_helmholtz_qtt.mp4", 
                          show_vorticity=False, fps=20):
    """
    Render the KH instability animation to video.
    
    Args:
        history: List of (t, rho, u, v, rank) tuples
        filename: Output filename
        show_vorticity: If True, show vorticity instead of density
        fps: Frames per second
    """
    print(f"\nRendering animation with {len(history)} frames...")
    
    # Setup figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # Unpack first frame
    t_0, rho_0, u_0, v_0, rank_0 = history[0]
    
    # Left panel: Density
    ax1 = axes[0]
    im1 = ax1.imshow(rho_0.T, origin='lower', cmap='RdBu_r',
                     extent=[0, 1, 0, 1], vmin=0.9, vmax=2.1,
                     interpolation='bilinear')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Density Field')
    fig.colorbar(im1, ax=ax1, label='ρ')
    
    # Right panel: Velocity magnitude or vorticity
    ax2 = axes[1]
    vel_mag = np.sqrt(u_0**2 + v_0**2)
    im2 = ax2.imshow(vel_mag.T, origin='lower', cmap='viridis',
                     extent=[0, 1, 0, 1], vmin=0, vmax=1.0,
                     interpolation='bilinear')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Velocity Magnitude')
    fig.colorbar(im2, ax=ax2, label='|v|')
    
    # Text overlays
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14,
                         fontweight='bold')
    rank_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12,
                         color='blue', fontweight='bold')
    
    fig.suptitle("Kelvin-Helmholtz Instability: HyperTensor QTT Solver", 
                 fontsize=16, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    
    def update(frame_idx):
        t, rho, u, v, rank = history[frame_idx]
        
        # Update density
        im1.set_data(rho.T)
        
        # Update velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2)
        im2.set_data(vel_mag.T)
        
        # Update text
        time_text.set_text(f"Time: {t:.3f}")
        rank_text.set_text(f"QTT Max Rank: {rank}  |  O(log N) Memory")
        
        return [im1, im2, time_text, rank_text]
    
    ani = animation.FuncAnimation(fig, update, frames=len(history),
                                  interval=1000//fps, blit=True)
    
    # Save with ffmpeg
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000,
                                        extra_args=['-vcodec', 'libx264'])
        ani.save(filename, writer=writer)
        print(f"✓ Saved animation to {filename}")
    except Exception as e:
        print(f"FFmpeg error: {e}")
        print("Trying with pillow writer...")
        try:
            gif_name = filename.replace('.mp4', '.gif')
            ani.save(gif_name, writer='pillow', fps=fps)
            print(f"✓ Saved animation to {gif_name}")
        except Exception as e2:
            print(f"Pillow error: {e2}")
            print("Saving individual frames instead...")
            save_frames(history, filename.replace('.mp4', ''))
    
    plt.close(fig)


def save_frames(history, prefix="kh_frame"):
    """Save individual frames as PNG files."""
    import os
    os.makedirs(prefix, exist_ok=True)
    
    for i, (t, rho, u, v, rank) in enumerate(history):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.imshow(rho.T, origin='lower', cmap='RdBu_r',
                  extent=[0, 1, 0, 1], vmin=0.9, vmax=2.1)
        ax.set_title(f"t={t:.3f}, Rank={rank}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(f"{prefix}/frame_{i:04d}.png")
        plt.close(fig)
    
    print(f"Saved {len(history)} frames to {prefix}/")


def render_vorticity(history, filename="kh_vorticity.mp4", fps=20):
    """
    Render vorticity field (dv/dx - du/dy) animation.
    
    Vorticity shows the rolling-up of vortices more clearly.
    """
    print(f"\nRendering vorticity animation with {len(history)} frames...")
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Compute vorticity for first frame
    t_0, rho_0, u_0, v_0, rank_0 = history[0]
    
    def compute_vorticity(u, v):
        # dv/dx - du/dy using central differences
        dv_dx = np.gradient(v, axis=0)
        du_dy = np.gradient(u, axis=1)
        return dv_dx - du_dy
    
    vort_0 = compute_vorticity(u_0, v_0)
    vmax = 10.0  # Vorticity scale
    
    im = ax.imshow(vort_0.T, origin='lower', cmap='seismic',
                   extent=[0, 1, 0, 1], vmin=-vmax, vmax=vmax,
                   interpolation='bilinear')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Vorticity (ω = ∂v/∂x - ∂u/∂y)")
    fig.colorbar(im, ax=ax, label='ω')
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                        fontsize=12, color='black',
                        bbox=dict(facecolor='white', alpha=0.8))
    rank_text = ax.text(0.02, 0.88, '', transform=ax.transAxes,
                        fontsize=12, color='blue',
                        bbox=dict(facecolor='white', alpha=0.8))
    
    fig.suptitle("Kelvin-Helmholtz Vortex Roll-Up: HyperTensor QTT",
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    def update(frame_idx):
        t, rho, u, v, rank = history[frame_idx]
        vort = compute_vorticity(u, v)
        im.set_data(vort.T)
        time_text.set_text(f"Time: {t:.3f}")
        rank_text.set_text(f"Rank: {rank}")
        return [im, time_text, rank_text]
    
    ani = animation.FuncAnimation(fig, update, frames=len(history),
                                  interval=1000//fps, blit=True)
    
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        ani.save(filename, writer=writer)
        print(f"✓ Saved vorticity animation to {filename}")
    except Exception as e:
        print(f"Error: {e}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Kelvin-Helmholtz Animation")
    parser.add_argument('--n', type=int, default=7, 
                        help='Bits per dimension (grid=2^n)')
    parser.add_argument('--time', type=float, default=1.5,
                        help='Final simulation time')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save frame every N steps')
    parser.add_argument('--max-rank', type=int, default=64,
                        help='Maximum QTT rank')
    parser.add_argument('-o', '--output', type=str, 
                        default='kelvin_helmholtz_qtt.mp4',
                        help='Output filename')
    parser.add_argument('--vorticity', action='store_true',
                        help='Also render vorticity animation')
    parser.add_argument('--fps', type=int, default=20,
                        help='Video frame rate')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HyperTensor: Kelvin-Helmholtz Instability Animation")
    print("=" * 60)
    print(f"Grid: {2**args.n}×{2**args.n}")
    print(f"T_final: {args.time}")
    print(f"Max rank: {args.max_rank}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Run simulation
    history = run_kh_simulation(
        n_bits=args.n,
        t_final=args.time,
        save_every=args.save_every,
        max_rank=args.max_rank
    )
    
    # Render main animation
    render_kh_instability(history, args.output, fps=args.fps)
    
    # Optionally render vorticity
    if args.vorticity:
        vort_file = args.output.replace('.mp4', '_vorticity.mp4')
        render_vorticity(history, vort_file, fps=args.fps)
    
    print("\n" + "=" * 60)
    print("Animation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
