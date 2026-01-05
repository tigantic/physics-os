#!/usr/bin/env python3
"""
CFD Demo - Shock Wave Simulation
================================

Solves the 2D Euler equations to show real shock wave physics.
This is actual CFD - not procedural graphics.

Shows:
- Shock wave formation and propagation
- Density/pressure gradients
- Mach number field
- Real physics in real-time
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import math

# Use the actual CFD solver
from tensornet.cfd.euler_2d import Euler2D, Euler2DState, BCType


def create_explosion_ic(Nx: int, Ny: int, Lx: float = 2.0, Ly: float = 2.0) -> Euler2DState:
    """Circular explosion - high pressure bubble in ambient gas."""
    dx = Lx / Nx
    dy = Ly / Ny
    
    # Domain is [0, Lx] x [0, Ly], center at (Lx/2, Ly/2)
    x = torch.linspace(dx/2, Lx - dx/2, Nx, dtype=torch.float64)
    y = torch.linspace(dy/2, Ly - dy/2, Ny, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Distance from center
    cx, cy = Lx/2, Ly/2
    R = torch.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Ambient conditions
    rho_amb = 1.0
    p_amb = 1.0
    
    # High pressure bubble (pressure ratio 10:1)
    rho_bubble = 1.0
    p_bubble = 10.0
    
    radius = 0.2
    
    # Smooth transition
    blend = 0.5 * (1 - torch.tanh((R - radius) / 0.02))
    
    rho = rho_amb + (rho_bubble - rho_amb) * blend
    p = p_amb + (p_bubble - p_amb) * blend
    u = torch.zeros_like(X)
    v = torch.zeros_like(X)
    
    return Euler2DState(rho=rho, u=u, v=v, p=p, gamma=1.4)


def create_shock_tube_2d(Nx: int, Ny: int, Lx: float = 2.0, Ly: float = 1.0) -> Euler2DState:
    """2D Sod shock tube - classic benchmark."""
    dx = Lx / Nx
    dy = Ly / Ny
    
    x = torch.linspace(0 + dx/2, Lx - dx/2, Nx, dtype=torch.float64)
    y = torch.linspace(0 + dy/2, Ly - dy/2, Ny, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Sod shock tube conditions
    rho = torch.where(X < Lx/2, torch.tensor(1.0), torch.tensor(0.125))
    p = torch.where(X < Lx/2, torch.tensor(1.0), torch.tensor(0.1))
    u = torch.zeros_like(X)
    v = torch.zeros_like(X)
    
    return Euler2DState(rho=rho, u=u, v=v, p=p, gamma=1.4)


def create_supersonic_wedge(Nx: int, Ny: int) -> Euler2DState:
    """Supersonic flow over a wedge - creates oblique shock."""
    Lx, Ly = 3.0, 1.0
    dx = Lx / Nx
    dy = Ly / Ny
    
    x = torch.linspace(dx/2, Lx - dx/2, Nx, dtype=torch.float64)
    y = torch.linspace(dy/2, Ly - dy/2, Ny, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Mach 3 freestream
    M = 3.0
    gamma = 1.4
    rho = torch.ones_like(X)
    p = torch.ones_like(X)
    u = M * torch.sqrt(gamma * p / rho)  # Mach 3
    v = torch.zeros_like(X)
    
    return Euler2DState(rho=rho, u=u, v=v, p=p, gamma=gamma)


def run_demo():
    """Run interactive CFD demo."""
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("matplotlib required")
        return
    
    print("=" * 60)
    print("  CFD DEMO - 2D Euler Equations")
    print("=" * 60)
    print()
    
    # Grid resolution
    Nx, Ny = 200, 200
    Lx, Ly = 2.0, 2.0
    
    print(f"Grid: {Nx} × {Ny} = {Nx*Ny:,} cells")
    print(f"Domain: {Lx} × {Ly}")
    print()
    
    # Initialize solver with explosion IC
    print("Initializing circular explosion...")
    solver = Euler2D(
        Nx=Nx, Ny=Ny,
        Lx=Lx, Ly=Ly,
        gamma=1.4,
    )
    
    ic = create_explosion_ic(Nx, Ny, Lx, Ly)
    solver.set_initial_condition(ic)
    
    print("Running simulation...")
    print()
    print("Controls: SPACE=Pause, R=Reset, 1/2/3=View, Q=Quit")
    print()
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    ax.set_position([0.1, 0.08, 0.85, 0.85])
    
    # Initial plot
    state = solver.state
    data = state.rho.numpy()
    
    im = ax.imshow(data, cmap='inferno', origin='lower',
                   extent=[0, Lx, 0, Ly],
                   interpolation='bilinear')
    
    ax.set_xlabel('x', color='white', fontsize=11)
    ax.set_ylabel('y', color='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('#444')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Stats
    title = ax.set_title('', color='white', fontsize=12, pad=10)
    
    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=10, color='white', family='monospace',
                        verticalalignment='top',
                        bbox=dict(facecolor='black', alpha=0.8, edgecolor='#444'))
    
    fig.text(0.5, 0.97, '2D EULER EQUATIONS - Shock Wave', ha='center', 
            fontsize=14, color='white', fontweight='bold')
    
    sim_state = {
        'paused': False,
        't': 0.0,
        'step': 0,
        'view': 'density',  # density, pressure, mach
        'running': True,
    }
    
    def get_field():
        """Get currently selected field."""
        state = solver.state
        if sim_state['view'] == 'density':
            return state.rho.numpy(), 'Density ρ', 'inferno'
        elif sim_state['view'] == 'pressure':
            return state.p.numpy(), 'Pressure p', 'plasma'
        elif sim_state['view'] == 'mach':
            M = state.M.numpy()
            return M, 'Mach Number', 'coolwarm'
        else:
            return state.rho.numpy(), 'Density ρ', 'inferno'
    
    def update_plot():
        data, label, cmap = get_field()
        im.set_data(data)
        im.set_clim(data.min(), data.max())
        im.set_cmap(cmap)
        cbar.update_normal(im)
        
        title.set_text(f'{label} at t = {sim_state["t"]:.4f}')
        
        state = solver.state
        rho_min, rho_max = state.rho.min().item(), state.rho.max().item()
        p_min, p_max = state.p.min().item(), state.p.max().item()
        M_max = state.M.max().item()
        
        stats_text.set_text(
            f"Step: {sim_state['step']}\n"
            f"ρ: [{rho_min:.3f}, {rho_max:.3f}]\n"
            f"p: [{p_min:.3f}, {p_max:.3f}]\n"
            f"Mach max: {M_max:.2f}"
        )
    
    def step_simulation():
        if sim_state['paused']:
            return
        
        # Take one time step
        dt = solver.compute_dt()
        solver.step(dt)
        sim_state['t'] += dt
        sim_state['step'] += 1
        
        update_plot()
        fig.canvas.draw_idle()
    
    def on_key(event):
        if event.key == ' ':
            sim_state['paused'] = not sim_state['paused']
            status = "PAUSED" if sim_state['paused'] else "RUNNING"
            print(f"[{status}]")
        elif event.key == 'r':
            # Reset
            ic = create_explosion_ic(Nx, Ny, Lx, Ly)
            solver.set_initial_condition(ic)
            sim_state['t'] = 0.0
            sim_state['step'] = 0
            sim_state['paused'] = False
            print("[RESET]")
        elif event.key == '1':
            sim_state['view'] = 'density'
            print("[VIEW: Density]")
        elif event.key == '2':
            sim_state['view'] = 'pressure'
            print("[VIEW: Pressure]")
        elif event.key == '3':
            sim_state['view'] = 'mach'
            print("[VIEW: Mach]")
        elif event.key in ['q', 'escape']:
            sim_state['running'] = False
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Animation timer
    timer = fig.canvas.new_timer(interval=30)  # ~33 fps target
    timer.add_callback(step_simulation)
    timer.start()
    
    update_plot()
    plt.show()
    
    timer.stop()


if __name__ == '__main__':
    run_demo()
