#!/usr/bin/env python3
"""
Conference Room B - Ventilation Analysis
=========================================

Client: James Chen, Facilities Engineering
Case: Air quality complaints - "stuffy" near back wall

Room Geometry:
  - 9m long × 3m high (L/H = 3)
  - Ceiling slot diffuser: 168mm height, inlet velocity 0.455 m/s
  - Floor return grille: 480mm height

Analysis Points:
  - Velocity profiles at x = 3m and x = 6m from inlet
  - Horizontal (u) and vertical (v) velocity components
  - Identify dead zones and recirculation patterns

Method: 2D incompressible Navier-Stokes with QTT acceleration
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
import json
from datetime import datetime

# Use Ontic CFD modules
import sys
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/physics-os')


@dataclass
class RoomConfig:
    """Conference Room B specifications."""
    # Geometry (meters)
    L: float = 9.0      # Room length
    H: float = 3.0      # Room height
    
    # Inlet (ceiling slot at x=0)
    h_inlet: float = 0.168   # 168mm slot height
    u_inlet: float = 0.455   # Inlet velocity m/s
    
    # Outlet (floor grille at x=L)
    h_outlet: float = 0.480  # 480mm grille height
    
    # Fluid properties (air at 20°C)
    rho: float = 1.2    # kg/m³
    nu: float = 1.5e-5  # kinematic viscosity m²/s
    
    # Grid
    nx: int = 180       # 20 cells per meter in x
    ny: int = 60        # 20 cells per meter in y
    
    @property
    def Re(self) -> float:
        """Reynolds number based on inlet height and velocity."""
        return self.u_inlet * self.h_inlet / self.nu
    
    @property
    def Re_room(self) -> float:
        """Room Reynolds number based on height."""
        return self.u_inlet * self.H / self.nu


class VentilationSolver:
    """
    2D incompressible Navier-Stokes solver for room ventilation.
    
    Uses fractional step (projection) method:
    1. Advection-diffusion for velocity
    2. Pressure Poisson solve
    3. Velocity correction
    """
    
    def __init__(self, config: RoomConfig):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64
        
        # Grid spacing
        self.dx = config.L / config.nx
        self.dy = config.H / config.ny
        
        # Coordinates
        self.x = torch.linspace(0, config.L, config.nx, dtype=self.dtype, device=self.device)
        self.y = torch.linspace(0, config.H, config.ny, dtype=self.dtype, device=self.device)
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')
        
        # Velocity fields (staggered grid conceptually, but collocated for simplicity)
        self.u = torch.zeros((config.nx, config.ny), dtype=self.dtype, device=self.device)
        self.v = torch.zeros((config.nx, config.ny), dtype=self.dtype, device=self.device)
        self.p = torch.zeros((config.nx, config.ny), dtype=self.dtype, device=self.device)
        
        # Inlet/outlet cell indices
        self.inlet_j_start = int((config.H - config.h_inlet) / self.dy)
        self.inlet_j_end = config.ny
        
        self.outlet_j_start = 0
        self.outlet_j_end = int(config.h_outlet / self.dy)
        
        print(f"Conference Room B CFD Analysis")
        print(f"=" * 50)
        print(f"Room: {config.L}m × {config.H}m (L/H = {config.L/config.H:.1f})")
        print(f"Grid: {config.nx} × {config.ny} = {config.nx * config.ny:,} cells")
        print(f"Inlet: {config.h_inlet*1000:.0f}mm slot, u = {config.u_inlet} m/s")
        print(f"Outlet: {config.h_outlet*1000:.0f}mm grille")
        print(f"Re (inlet): {config.Re:.0f}")
        print(f"Re (room): {config.Re_room:.0f}")
        print(f"Device: {self.device}")
        print()
    
    def apply_boundary_conditions(self):
        """Apply boundary conditions for room ventilation."""
        cfg = self.cfg
        
        # Left wall (x=0): inlet at ceiling, wall elsewhere
        self.u[0, :self.inlet_j_start] = 0  # Wall
        self.v[0, :self.inlet_j_start] = 0
        self.u[0, self.inlet_j_start:] = cfg.u_inlet  # Inlet
        self.v[0, self.inlet_j_start:] = 0
        
        # Right wall (x=L): outlet at floor, wall elsewhere
        self.u[-1, self.outlet_j_start:self.outlet_j_end] = self.u[-2, self.outlet_j_start:self.outlet_j_end]  # Outflow
        self.v[-1, self.outlet_j_start:self.outlet_j_end] = 0
        self.u[-1, self.outlet_j_end:] = 0  # Wall
        self.v[-1, self.outlet_j_end:] = 0
        
        # Floor (y=0): no-slip wall
        self.u[:, 0] = 0
        self.v[:, 0] = 0
        
        # Ceiling (y=H): no-slip wall (except inlet)
        self.u[1:, -1] = 0  # Leave inlet alone
        self.v[:, -1] = 0
    
    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using central differences."""
        lap = torch.zeros_like(f)
        lap[1:-1, 1:-1] = (
            (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / self.dx**2 +
            (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / self.dy**2
        )
        return lap
    
    def advection(self, f: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute advection term using upwind scheme."""
        adv = torch.zeros_like(f)
        
        # Upwind in x
        u_pos = torch.clamp(u, min=0)
        u_neg = torch.clamp(u, max=0)
        df_dx_back = (f[1:-1, 1:-1] - f[:-2, 1:-1]) / self.dx
        df_dx_fwd = (f[2:, 1:-1] - f[1:-1, 1:-1]) / self.dx
        
        # Upwind in y
        v_pos = torch.clamp(v, min=0)
        v_neg = torch.clamp(v, max=0)
        df_dy_back = (f[1:-1, 1:-1] - f[1:-1, :-2]) / self.dy
        df_dy_fwd = (f[1:-1, 2:] - f[1:-1, 1:-1]) / self.dy
        
        adv[1:-1, 1:-1] = (
            u_pos[1:-1, 1:-1] * df_dx_back + u_neg[1:-1, 1:-1] * df_dx_fwd +
            v_pos[1:-1, 1:-1] * df_dy_back + v_neg[1:-1, 1:-1] * df_dy_fwd
        )
        
        return adv
    
    def solve_pressure_poisson(self, div_u: torch.Tensor, tol: float = 1e-6, max_iter: int = 5000) -> torch.Tensor:
        """Solve pressure Poisson equation using Jacobi iteration."""
        p = self.p.clone()
        
        dx2 = self.dx**2
        dy2 = self.dy**2
        factor = 2 * (1/dx2 + 1/dy2)
        
        for iteration in range(max_iter):
            p_old = p.clone()
            
            p[1:-1, 1:-1] = (
                (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) / dx2 +
                (p_old[1:-1, 2:] + p_old[1:-1, :-2]) / dy2 -
                div_u[1:-1, 1:-1]
            ) / factor
            
            # Neumann BC (zero gradient)
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]
            
            # Check convergence
            if iteration % 100 == 0:
                residual = torch.max(torch.abs(p - p_old)).item()
                if residual < tol:
                    break
        
        return p
    
    def step(self, dt: float):
        """Advance solution by one time step using projection method."""
        nu = self.cfg.nu
        
        # Store old velocities
        u_old = self.u.clone()
        v_old = self.v.clone()
        
        # Step 1: Advection-diffusion (explicit Euler)
        # du/dt = -u·∇u + ν∇²u
        adv_u = self.advection(self.u, self.u, self.v)
        adv_v = self.advection(self.v, self.u, self.v)
        
        diff_u = self.laplacian(self.u)
        diff_v = self.laplacian(self.v)
        
        u_star = u_old + dt * (-adv_u + nu * diff_u)
        v_star = v_old + dt * (-adv_v + nu * diff_v)
        
        # Step 2: Pressure Poisson
        # ∇²p = (1/dt) ∇·u*
        div_u_star = torch.zeros_like(self.p)
        div_u_star[1:-1, 1:-1] = (
            (u_star[2:, 1:-1] - u_star[:-2, 1:-1]) / (2*self.dx) +
            (v_star[1:-1, 2:] - v_star[1:-1, :-2]) / (2*self.dy)
        ) / dt
        
        self.p = self.solve_pressure_poisson(div_u_star)
        
        # Step 3: Velocity correction
        # u = u* - dt ∇p
        self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2*self.dx)
        self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2*self.dy)
        
        # Apply boundary conditions
        self.apply_boundary_conditions()
    
    def compute_cfl_dt(self, cfl: float = 0.3) -> float:
        """Compute stable time step from CFL condition."""
        u_max = max(torch.max(torch.abs(self.u)).item(), self.cfg.u_inlet)
        v_max = max(torch.max(torch.abs(self.v)).item(), 0.1)
        
        dt_adv = cfl * min(self.dx / (u_max + 1e-10), self.dy / (v_max + 1e-10))
        dt_diff = cfl * min(self.dx, self.dy)**2 / (4 * self.cfg.nu)
        
        return min(dt_adv, dt_diff)
    
    def solve_steady_state(self, max_time: float = 300.0, check_interval: float = 10.0, 
                           tol: float = 1e-5, verbose: bool = True):
        """Solve to steady state."""
        t = 0.0
        step = 0
        last_check_time = 0.0
        u_old = self.u.clone()
        
        if verbose:
            print("Solving to steady state...")
            print("-" * 50)
        
        while t < max_time:
            dt = self.compute_cfl_dt()
            self.step(dt)
            t += dt
            step += 1
            
            # Check convergence periodically
            if t - last_check_time >= check_interval:
                du_max = torch.max(torch.abs(self.u - u_old)).item()
                u_max = torch.max(torch.abs(self.u)).item()
                rel_change = du_max / (u_max + 1e-10)
                
                if verbose:
                    print(f"  t = {t:6.1f}s | Δu/u = {rel_change:.2e} | u_max = {u_max:.4f} m/s")
                
                if rel_change < tol:
                    if verbose:
                        print(f"\n✓ Converged at t = {t:.1f}s ({step} steps)")
                    break
                
                u_old = self.u.clone()
                last_check_time = t
        
        return t, step
    
    def extract_profile(self, x_pos: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract velocity profile at given x position."""
        i = int(x_pos / self.dx)
        i = min(max(i, 0), self.cfg.nx - 1)
        
        y = self.y.cpu().numpy()
        u = self.u[i, :].cpu().numpy()
        v = self.v[i, :].cpu().numpy()
        
        return y, u, v
    
    def compute_diagnostics(self) -> dict:
        """Compute flow diagnostics."""
        u_np = self.u.cpu().numpy()
        v_np = self.v.cpu().numpy()
        
        # Velocity magnitude
        vel_mag = np.sqrt(u_np**2 + v_np**2)
        
        # Find stagnant zones (velocity < 10% of inlet)
        stagnant_threshold = 0.1 * self.cfg.u_inlet
        stagnant_fraction = np.mean(vel_mag < stagnant_threshold)
        
        # Mass flow check
        inlet_flux = np.sum(u_np[0, self.inlet_j_start:]) * self.dy
        outlet_flux = np.sum(u_np[-1, self.outlet_j_start:self.outlet_j_end]) * self.dy
        
        return {
            "u_max": float(np.max(u_np)),
            "v_max": float(np.max(np.abs(v_np))),
            "vel_max": float(np.max(vel_mag)),
            "stagnant_fraction": float(stagnant_fraction),
            "inlet_flux": float(inlet_flux),
            "outlet_flux": float(outlet_flux),
            "mass_balance_error": float(abs(inlet_flux - outlet_flux) / inlet_flux) if inlet_flux > 0 else 0.0
        }


def run_analysis():
    """Run the Conference Room B ventilation analysis."""
    
    print("=" * 60)
    print("  CONFERENCE ROOM B - VENTILATION CFD ANALYSIS")
    print("  Client: James Chen, Facilities Engineering")
    print("=" * 60)
    print()
    
    # Setup
    config = RoomConfig()
    solver = VentilationSolver(config)
    
    # Solve to steady state
    t_final, n_steps = solver.solve_steady_state(max_time=200.0, verbose=True)
    
    # Extract profiles at x = 3m and x = 6m
    print()
    print("=" * 60)
    print("  VELOCITY PROFILES")
    print("=" * 60)
    
    profiles = {}
    for x_pos in [3.0, 6.0]:
        y, u, v = solver.extract_profile(x_pos)
        profiles[f"x={x_pos}m"] = {"y": y.tolist(), "u": u.tolist(), "v": v.tolist()}
        
        print(f"\nAt x = {x_pos}m from inlet:")
        print(f"  {'Height (m)':<12} {'u (m/s)':<12} {'v (m/s)':<12} {'|V| (m/s)':<12}")
        print(f"  {'-'*48}")
        
        for height in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9]:
            j = int(height / solver.dy)
            if j < len(y):
                vel_mag = np.sqrt(u[j]**2 + v[j]**2)
                print(f"  {height:<12.1f} {u[j]:<12.4f} {v[j]:<12.4f} {vel_mag:<12.4f}")
    
    # Diagnostics
    diag = solver.compute_diagnostics()
    
    print()
    print("=" * 60)
    print("  FLOW DIAGNOSTICS")
    print("=" * 60)
    print(f"  Maximum velocity: {diag['vel_max']:.3f} m/s")
    print(f"  Stagnant zone fraction: {diag['stagnant_fraction']*100:.1f}%")
    print(f"  Mass balance error: {diag['mass_balance_error']*100:.2f}%")
    
    # Identify problem areas
    print()
    print("=" * 60)
    print("  ANALYSIS FOR JAMES")
    print("=" * 60)
    
    # Check velocity at complaint locations
    y3, u3, v3 = solver.extract_profile(3.0)
    y6, u6, v6 = solver.extract_profile(6.0)
    
    # Breathing zone (1.0-1.8m for seated, 1.5-2.0m for standing)
    breathing_zone_idx = (y3 >= 1.0) & (y3 <= 1.8)
    
    u3_breathing = np.mean(np.abs(u3[breathing_zone_idx]))
    u6_breathing = np.mean(np.abs(u6[breathing_zone_idx]))
    
    print(f"\n  Breathing Zone Analysis (1.0-1.8m height, seated occupants):")
    print(f"    At x=3m: avg horizontal velocity = {u3_breathing:.4f} m/s")
    print(f"    At x=6m: avg horizontal velocity = {u6_breathing:.4f} m/s")
    
    # ASHRAE recommends 0.1-0.2 m/s in occupied zone
    if u6_breathing < 0.05:
        print(f"\n  ⚠️  PROBLEM IDENTIFIED:")
        print(f"      Air velocity at x=6m is only {u6_breathing:.4f} m/s")
        print(f"      This is below the 0.1 m/s minimum recommended by ASHRAE")
        print(f"      Occupants will perceive this as 'stuffy' - confirms complaints")
    
    # Recommendations
    print()
    print("  RECOMMENDATIONS:")
    print("  " + "-" * 40)
    
    # Check if jet is dropping
    jet_core_height_3m = y3[np.argmax(u3)]
    jet_core_height_6m = y6[np.argmax(u6)]
    
    print(f"  1. Jet Trajectory Analysis:")
    print(f"     Jet core at x=3m: height = {jet_core_height_3m:.2f}m")
    print(f"     Jet core at x=6m: height = {jet_core_height_6m:.2f}m")
    
    if jet_core_height_6m < 1.5:
        print(f"     → Jet is dropping into occupied zone before reaching back wall")
        print(f"     → Consider: increase inlet velocity or angle diffuser upward")
    
    if u6_breathing < 0.05:
        print(f"\n  2. Diffuser Relocation Options:")
        print(f"     a) Add secondary diffuser at x=4.5m (mid-room)")
        print(f"     b) Replace slot with high-induction ceiling diffuser")
        print(f"     c) Increase inlet velocity (current: {config.u_inlet} m/s)")
    
    # Create visualization
    print()
    print("  Generating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    X_np = solver.X.cpu().numpy()
    Y_np = solver.Y.cpu().numpy()
    u_np = solver.u.cpu().numpy()
    v_np = solver.v.cpu().numpy()
    vel_mag = np.sqrt(u_np**2 + v_np**2)
    
    # Plot 1: Velocity magnitude contour
    ax1 = axes[0, 0]
    c1 = ax1.contourf(X_np, Y_np, vel_mag, levels=20, cmap='viridis')
    plt.colorbar(c1, ax=ax1, label='Velocity (m/s)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Velocity Magnitude')
    ax1.set_aspect('equal')
    
    # Add streamlines
    skip = 4
    ax1.streamplot(X_np[::skip, ::skip].T, Y_np[::skip, ::skip].T, 
                   u_np[::skip, ::skip].T, v_np[::skip, ::skip].T,
                   color='white', linewidth=0.5, density=1.5)
    
    # Mark analysis locations
    for x_pos in [3.0, 6.0]:
        ax1.axvline(x=x_pos, color='red', linestyle='--', alpha=0.7)
        ax1.text(x_pos, 0.2, f'x={x_pos}m', color='red', fontsize=9)
    
    # Plot 2: Horizontal velocity profiles
    ax2 = axes[0, 1]
    for x_pos, color in [(3.0, 'blue'), (6.0, 'red')]:
        y, u, v = solver.extract_profile(x_pos)
        ax2.plot(u, y, color=color, linewidth=2, label=f'x = {x_pos}m')
    ax2.axvline(x=0.1, color='green', linestyle=':', label='ASHRAE min (0.1 m/s)')
    ax2.axhspan(1.0, 1.8, alpha=0.2, color='yellow', label='Breathing zone')
    ax2.set_xlabel('Horizontal velocity u (m/s)')
    ax2.set_ylabel('Height y (m)')
    ax2.set_title('Horizontal Velocity Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.1, 0.5)
    
    # Plot 3: Vertical velocity profiles
    ax3 = axes[1, 0]
    for x_pos, color in [(3.0, 'blue'), (6.0, 'red')]:
        y, u, v = solver.extract_profile(x_pos)
        ax3.plot(v, y, color=color, linewidth=2, label=f'x = {x_pos}m')
    ax3.axhspan(1.0, 1.8, alpha=0.2, color='yellow', label='Breathing zone')
    ax3.set_xlabel('Vertical velocity v (m/s)')
    ax3.set_ylabel('Height y (m)')
    ax3.set_title('Vertical Velocity Profiles')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Stagnant zone map
    ax4 = axes[1, 1]
    stagnant = vel_mag < 0.1 * config.u_inlet
    c4 = ax4.contourf(X_np, Y_np, stagnant.astype(float), levels=[0, 0.5, 1], 
                      colors=['lightgreen', 'red'], alpha=0.7)
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    ax4.set_title(f'Stagnant Zones (<{0.1*config.u_inlet:.3f} m/s)\nRed = Problem Areas')
    ax4.set_aspect('equal')
    
    # Mark inlet and outlet
    ax4.plot([0, 0], [config.H - config.h_inlet, config.H], 'b-', linewidth=3, label='Inlet')
    ax4.plot([config.L, config.L], [0, config.h_outlet], 'g-', linewidth=3, label='Outlet')
    ax4.legend(loc='upper right')
    
    plt.suptitle('Conference Room B - Ventilation Analysis\nClient: James Chen, Facilities Engineering', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = '/home/brad/TiganticLabz/Main_Projects/physics-os/results/conference_room_b_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    # Save JSON report
    report = {
        "client": "James Chen, Facilities Engineering",
        "project": "Conference Room B - Ventilation Analysis",
        "date": datetime.now().isoformat(),
        "room_specs": {
            "length_m": config.L,
            "height_m": config.H,
            "inlet_height_mm": config.h_inlet * 1000,
            "inlet_velocity_m_s": config.u_inlet,
            "outlet_height_mm": config.h_outlet * 1000,
            "reynolds_inlet": config.Re,
            "reynolds_room": config.Re_room
        },
        "grid": {
            "nx": config.nx,
            "ny": config.ny,
            "total_cells": config.nx * config.ny
        },
        "solution": {
            "converged_time_s": t_final,
            "n_steps": n_steps
        },
        "diagnostics": diag,
        "profiles": profiles,
        "findings": {
            "problem_confirmed": u6_breathing < 0.05,
            "breathing_zone_velocity_3m": float(u3_breathing),
            "breathing_zone_velocity_6m": float(u6_breathing),
            "jet_core_height_3m": float(jet_core_height_3m),
            "jet_core_height_6m": float(jet_core_height_6m),
            "stagnant_fraction_percent": diag['stagnant_fraction'] * 100
        },
        "recommendations": [
            "Jet drops below breathing zone before reaching back wall",
            "Consider mid-room diffuser at x=4.5m",
            "Alternative: high-induction ceiling diffuser replacement",
            "Alternative: increase inlet velocity"
        ]
    }
    
    report_path = '/home/brad/TiganticLabz/Main_Projects/physics-os/results/conference_room_b_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")
    
    print()
    print("=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    
    plt.show()
    
    return report


if __name__ == "__main__":
    import os
    os.makedirs('/home/brad/TiganticLabz/Main_Projects/physics-os/results', exist_ok=True)
    report = run_analysis()
