#!/usr/bin/env python3
"""
TigantiCFD - Tier 1 Conference Room HVAC Analysis (FIXED v2)
=============================================================

FIXES FROM v1:
1. Residual checks ALL velocity components (u, v, w), not just u
2. Uses upwind advection for propagation (skew-symmetric fails with zero fields)
3. Minimum iteration count before convergence check
4. Proper inlet depth for gradient-based advection

Client: James Morrison, Morrison & Associates Law Firm
Project: Conference Room B - Thermal Comfort Assessment
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ClientInfo:
    """Client information."""
    client_name: str = "James Morrison"
    company: str = "Morrison & Associates Law Firm"
    project_name: str = "Conference Room B - Thermal Comfort Assessment"
    project_id: str = "TGC-2026-001"


@dataclass  
class RoomGeometry:
    """Room dimensions."""
    length_ft: float = 26.0
    width_ft: float = 25.0  
    height_ft: float = 10.0
    
    supply_length_ft: float = 4.0
    supply_width_ft: float = 2.0
    supply_velocity_fpm: float = 450
    
    return_length_ft: float = 4.0
    return_width_ft: float = 2.0
    
    num_occupants: int = 12
    
    def __post_init__(self):
        self.FT_TO_M = 0.3048
        self.FPM_TO_MS = 0.00508
        
        self.length = self.length_ft * self.FT_TO_M
        self.width = self.width_ft * self.FT_TO_M
        self.height = self.height_ft * self.FT_TO_M
        
        self.supply_velocity = self.supply_velocity_fpm * self.FPM_TO_MS
        
        self.floor_area_sqft = self.length_ft * self.width_ft
        self.volume_cuft = self.floor_area_sqft * self.height_ft
        
        supply_area = self.supply_length_ft * self.supply_width_ft
        supply_cfm = supply_area * self.supply_velocity_fpm
        self.ach = (supply_cfm * 60) / self.volume_cuft


@dataclass
class SolverConfig:
    """Solver settings."""
    nx: int = 48
    ny: int = 48
    nz: int = 24
    
    nu: float = 1.5e-5
    rho: float = 1.2
    nu_t: float = 0.005  # Turbulent viscosity
    
    dt: float = 0.05
    t_end: float = 60.0  # Reduced from 120s
    
    residual_tol: float = 1e-4
    min_iterations: int = 500  # NEW: Don't check convergence before this
    max_iterations: int = 5000
    
    output_interval: int = 200
    
    @property
    def nu_eff(self) -> float:
        return self.nu + self.nu_t


# =============================================================================
# SOLVER
# =============================================================================

class ConferenceRoomSolver:
    """3D CFD solver for conference room - FIXED."""
    
    def __init__(self, geometry: RoomGeometry, config: SolverConfig, device: str = 'cpu'):
        self.geometry = geometry
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float64
        
        self.nx, self.ny, self.nz = config.nx, config.ny, config.nz
        self.dx = geometry.length / (self.nx - 1)
        self.dy = geometry.width / (self.ny - 1)
        self.dz = geometry.height / (self.nz - 1)
        
        self.x = torch.linspace(0, geometry.length, self.nx, dtype=self.dtype, device=self.device)
        self.y = torch.linspace(0, geometry.width, self.ny, dtype=self.dtype, device=self.device)
        self.z = torch.linspace(0, geometry.height, self.nz, dtype=self.dtype, device=self.device)
        
        self.u = torch.zeros(self.nx, self.ny, self.nz, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(self.nx, self.ny, self.nz, dtype=self.dtype, device=self.device)
        self.w = torch.zeros(self.nx, self.ny, self.nz, dtype=self.dtype, device=self.device)
        
        self._setup_boundaries()
        self._apply_initial_conditions()
        
        self.residual_history: List[float] = []
        
    def _setup_boundaries(self):
        """Define inlet/outlet regions."""
        geo = self.geometry
        
        # Supply diffuser at ceiling, x=0 end, centered in y
        supply_length = geo.supply_length_ft * geo.FT_TO_M
        supply_width = geo.supply_width_ft * geo.FT_TO_M
        
        self.inlet_i_end = max(2, int(supply_length / self.dx) + 1)
        
        y_center = geo.width / 2
        self.inlet_j_start = int((y_center - supply_width/2) / self.dy)
        self.inlet_j_end = int((y_center + supply_width/2) / self.dy) + 1
        
        # Return grille at ceiling, x=L end
        return_length = geo.return_length_ft * geo.FT_TO_M
        return_width = geo.return_width_ft * geo.FT_TO_M
        
        self.outlet_i_start = self.nx - int(return_length / self.dx) - 1
        self.outlet_j_start = int((y_center - return_width/2) / self.dy)
        self.outlet_j_end = int((y_center + return_width/2) / self.dy) + 1
        
        self.w_inlet = -geo.supply_velocity  # Downward
        
    def _apply_initial_conditions(self):
        """Set initial flow field - helps advection get started."""
        # Set inlet region with depth (not just surface)
        for k in range(self.nz - 3, self.nz):  # Top 3 cells
            # Ramp velocity: full at ceiling, decreasing downward
            factor = (k - (self.nz - 4)) / 3.0
            self.w[0:self.inlet_i_end, self.inlet_j_start:self.inlet_j_end, k] = self.w_inlet * factor
    
    def _apply_bcs(self):
        """Apply boundary conditions."""
        # Walls (no-slip)
        self.u[0, :, :] = 0; self.v[0, :, :] = 0; self.w[0, :, :] = 0
        self.u[-1, :, :] = 0; self.v[-1, :, :] = 0; self.w[-1, :, :] = 0
        self.u[:, 0, :] = 0; self.v[:, 0, :] = 0; self.w[:, 0, :] = 0
        self.u[:, -1, :] = 0; self.v[:, -1, :] = 0; self.w[:, -1, :] = 0
        self.u[:, :, 0] = 0; self.v[:, :, 0] = 0; self.w[:, :, 0] = 0
        
        # Ceiling - no penetration only (u, v can develop)
        self.w[:, :, -1] = 0
        
        # Inlet (supply diffuser) - set downward velocity
        self.w[0:self.inlet_i_end, self.inlet_j_start:self.inlet_j_end, -1] = self.w_inlet
        self.w[0:self.inlet_i_end, self.inlet_j_start:self.inlet_j_end, -2] = self.w_inlet * 0.8
        
        # Outlet (return grille) - zero gradient
        self.u[self.outlet_i_start:, self.outlet_j_start:self.outlet_j_end, -1] = \
            self.u[self.outlet_i_start:, self.outlet_j_start:self.outlet_j_end, -2]
        self.v[self.outlet_i_start:, self.outlet_j_start:self.outlet_j_end, -1] = \
            self.v[self.outlet_i_start:, self.outlet_j_start:self.outlet_j_end, -2]
        self.w[self.outlet_i_start:, self.outlet_j_start:self.outlet_j_end, -1] = \
            self.w[self.outlet_i_start:, self.outlet_j_start:self.outlet_j_end, -2]
    
    def _laplacian(self, f: Tensor) -> Tensor:
        """7-point Laplacian."""
        dx2, dy2, dz2 = self.dx**2, self.dy**2, self.dz**2
        return (
            (torch.roll(f, -1, dims=0) - 2*f + torch.roll(f, 1, dims=0)) / dx2 +
            (torch.roll(f, -1, dims=1) - 2*f + torch.roll(f, 1, dims=1)) / dy2 +
            (torch.roll(f, -1, dims=2) - 2*f + torch.roll(f, 1, dims=2)) / dz2
        )
    
    def _advection_upwind(self, phi: Tensor, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        """Upwind advection - stable and propagates."""
        dx, dy, dz = self.dx, self.dy, self.dz
        
        dphi_dx = torch.where(u > 0,
            (phi - torch.roll(phi, 1, dims=0)) / dx,
            (torch.roll(phi, -1, dims=0) - phi) / dx)
        dphi_dy = torch.where(v > 0,
            (phi - torch.roll(phi, 1, dims=1)) / dy,
            (torch.roll(phi, -1, dims=1) - phi) / dy)
        dphi_dz = torch.where(w > 0,
            (phi - torch.roll(phi, 1, dims=2)) / dz,
            (torch.roll(phi, -1, dims=2) - phi) / dz)
        
        return u * dphi_dx + v * dphi_dy + w * dphi_dz
    
    def step(self, dt: float) -> Dict:
        """Advance one timestep."""
        cfg = self.config
        
        # Store for residual (ALL components)
        u_old = self.u.clone()
        v_old = self.v.clone()
        w_old = self.w.clone()
        
        # Advection (upwind for propagation)
        adv_u = self._advection_upwind(self.u, self.u, self.v, self.w)
        adv_v = self._advection_upwind(self.v, self.u, self.v, self.w)
        adv_w = self._advection_upwind(self.w, self.u, self.v, self.w)
        
        # Diffusion
        lap_u = self._laplacian(self.u)
        lap_v = self._laplacian(self.v)
        lap_w = self._laplacian(self.w)
        
        # Update
        nu_eff = cfg.nu_eff
        self.u = self.u + dt * (-adv_u + nu_eff * lap_u)
        self.v = self.v + dt * (-adv_v + nu_eff * lap_v)
        self.w = self.w + dt * (-adv_w + nu_eff * lap_w)
        
        # Apply BCs
        self._apply_bcs()
        
        # Residual - CHECK ALL COMPONENTS
        du = torch.abs(self.u - u_old).max().item()
        dv = torch.abs(self.v - v_old).max().item()
        dw = torch.abs(self.w - w_old).max().item()
        residual = max(du, dv, dw)
        
        vel_mag = torch.sqrt(self.u**2 + self.v**2 + self.w**2)
        
        return {
            'residual': residual,
            'max_velocity': vel_mag.max().item(),
            'mean_velocity': vel_mag.mean().item(),
        }
    
    def solve(self, verbose: bool = True) -> Dict:
        """Run to steady state."""
        cfg = self.config
        n_steps = int(cfg.t_end / cfg.dt)
        
        if verbose:
            print("\n" + "="*70)
            print("RUNNING CFD SIMULATION (v2 FIXED)")
            print("="*70)
            print(f"Grid: {self.nx}×{self.ny}×{self.nz} = {self.nx*self.ny*self.nz:,} cells")
            print(f"Domain: {self.geometry.length:.2f}×{self.geometry.width:.2f}×{self.geometry.height:.2f} m")
            print(f"Inlet velocity: {abs(self.w_inlet):.3f} m/s downward")
            print(f"Timestep: {cfg.dt:.4f}s, Total: {cfg.t_end:.1f}s")
            print(f"Min iterations before convergence check: {cfg.min_iterations}")
            print("-"*70)
        
        start = time.perf_counter()
        converged = False
        
        for step in range(n_steps):
            diag = self.step(cfg.dt)
            self.residual_history.append(diag['residual'])
            
            # Only check convergence after min_iterations
            if step >= cfg.min_iterations and diag['residual'] < cfg.residual_tol:
                converged = True
                if verbose:
                    print(f"\n✓ Converged at step {step} (t = {step*cfg.dt:.2f}s)")
                break
            
            if verbose and step % cfg.output_interval == 0:
                t = step * cfg.dt
                print(f"  t={t:6.1f}s: res={diag['residual']:.2e}, "
                      f"max_vel={diag['max_velocity']:.3f}, mean_vel={diag['mean_velocity']:.4f} m/s")
        
        elapsed = time.perf_counter() - start
        
        if verbose:
            print("-"*70)
            print(f"Completed in {elapsed:.1f}s")
            print(f"Final: max_vel={diag['max_velocity']:.3f} m/s")
        
        return {
            'converged': converged,
            'elapsed': elapsed,
            'final_residual': diag['residual'],
            'max_velocity': diag['max_velocity'],
            'iterations': step + 1,
        }
    
    def velocity_magnitude(self) -> Tensor:
        return torch.sqrt(self.u**2 + self.v**2 + self.w**2)
    
    def extract_plane(self, plane: str, index: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor]:
        vel_mag = self.velocity_magnitude()
        
        if plane == 'xy':
            k = index if index is not None else self.nz // 2
            return self.x.cpu(), self.y.cpu(), vel_mag[:, :, k].cpu()
        elif plane == 'xz':
            j = index if index is not None else self.ny // 2
            return self.x.cpu(), self.z.cpu(), vel_mag[:, j, :].cpu()
        elif plane == 'yz':
            i = index if index is not None else self.nx // 2
            return self.y.cpu(), self.z.cpu(), vel_mag[i, :, :].cpu()
        else:
            raise ValueError(f"Unknown plane: {plane}")


# =============================================================================
# COMFORT ANALYSIS
# =============================================================================

@dataclass
class ComfortMetrics:
    max_air_speed: float = 0.0
    mean_air_speed_occupied: float = 0.0
    draft_risk_zones: int = 0
    velocity_std: float = 0.0
    coefficient_of_variation: float = 0.0
    air_changes_per_hour: float = 0.0
    ventilation_effectiveness: float = 0.0
    stagnation_zones: int = 0
    stagnation_percentage: float = 0.0
    comfort_score: float = 0.0
    assessment: str = ""


def analyze_comfort(solver: ConferenceRoomSolver, geometry: RoomGeometry) -> ComfortMetrics:
    """Analyze thermal comfort per ASHRAE 55."""
    vel_mag = solver.velocity_magnitude()
    
    # Occupied zone: 0.1-1.8m height, 0.6m from walls
    z_min = max(1, int(0.1 / solver.dz))
    z_max = min(solver.nz - 1, int(1.8 / solver.dz))
    wall_buf = int(0.6 / min(solver.dx, solver.dy))
    
    occupied = vel_mag[wall_buf:-wall_buf, wall_buf:-wall_buf, z_min:z_max]
    
    metrics = ComfortMetrics()
    metrics.max_air_speed = vel_mag.max().item()
    metrics.mean_air_speed_occupied = occupied.mean().item()
    metrics.velocity_std = occupied.std().item()
    
    draft_threshold = 0.25
    stag_threshold = 0.05
    total = occupied.numel()
    
    metrics.draft_risk_zones = int((occupied > draft_threshold).sum().item())
    stag_count = int((occupied < stag_threshold).sum().item())
    metrics.stagnation_zones = stag_count
    metrics.stagnation_percentage = 100.0 * stag_count / total
    
    metrics.coefficient_of_variation = metrics.velocity_std / (metrics.mean_air_speed_occupied + 1e-10)
    metrics.air_changes_per_hour = geometry.ach
    
    inlet_vel = abs(solver.w_inlet)
    if inlet_vel > 0:
        metrics.ventilation_effectiveness = min(1.0, metrics.mean_air_speed_occupied / (0.1 * inlet_vel))
    
    # Comfort score
    score = 100.0
    draft_pct = 100.0 * metrics.draft_risk_zones / total
    score -= draft_pct * 2.0
    score -= metrics.stagnation_percentage * 1.5
    if metrics.coefficient_of_variation > 0.5:
        score -= (metrics.coefficient_of_variation - 0.5) * 20
    
    metrics.comfort_score = max(0, min(100, score))
    
    if metrics.comfort_score >= 85:
        metrics.assessment = "EXCELLENT - Optimal thermal comfort"
    elif metrics.comfort_score >= 70:
        metrics.assessment = "GOOD - Minor improvements possible"
    elif metrics.comfort_score >= 50:
        metrics.assessment = "FAIR - Some comfort issues"
    else:
        metrics.assessment = "POOR - Significant comfort issues"
    
    return metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(solver: ConferenceRoomSolver, geometry: RoomGeometry, 
                          output_dir: Path) -> List[Path]:
    if not HAS_MATPLOTLIB:
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    
    # XY plane at occupant height
    fig, ax = plt.subplots(figsize=(12, 8))
    z_height = 1.2
    k = int(z_height / solver.dz)
    x, y, vel = solver.extract_plane('xy', k)
    
    X, Y = np.meshgrid(x.numpy(), y.numpy(), indexing='ij')
    vmax = max(0.5, vel.max().item())
    c = ax.pcolormesh(X, Y, vel.numpy(), cmap='viridis', shading='auto', vmin=0, vmax=vmax)
    plt.colorbar(c, ax=ax, label='Velocity (m/s)')
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Width (m)')
    ax.set_title(f'Velocity at z={z_height}m (breathing height)')
    ax.set_aspect('equal')
    
    path1 = output_dir / 'velocity_xy_plane.png'
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures.append(path1)
    
    # XZ plane
    fig, ax = plt.subplots(figsize=(12, 6))
    j = solver.ny // 2
    x, z, vel = solver.extract_plane('xz', j)
    
    X, Z = np.meshgrid(x.numpy(), z.numpy(), indexing='ij')
    c = ax.pcolormesh(X, Z, vel.numpy(), cmap='viridis', shading='auto', vmin=0, vmax=vmax)
    plt.colorbar(c, ax=ax, label='Velocity (m/s)')
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Velocity - Side View')
    ax.set_aspect('equal')
    
    path2 = output_dir / 'velocity_xz_plane.png'
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures.append(path2)
    
    # Convergence
    if len(solver.residual_history) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(solver.residual_history)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual')
        ax.set_title('Convergence History')
        ax.grid(True, alpha=0.3)
        
        path3 = output_dir / 'convergence_history.png'
        fig.savefig(path3, dpi=150, bbox_inches='tight')
        plt.close(fig)
        figures.append(path3)
    
    return figures


# =============================================================================
# MAIN
# =============================================================================

def run_tier1_simulation(output_dir: Optional[str] = None) -> Dict:
    """Run complete T1 analysis."""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   TigantiCFD - TIER 1 CONFERENCE ROOM ANALYSIS (v2 FIXED)       ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    if output_dir is None:
        output_dir = Path("./tiganti_output/TGC-2026-001")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = ClientInfo()
    geometry = RoomGeometry()
    config = SolverConfig()
    
    print(f"Client: {client.client_name}")
    print(f"Room: {geometry.length_ft}' × {geometry.width_ft}' × {geometry.height_ft}'")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    solver = ConferenceRoomSolver(geometry, config, device=device)
    solve_result = solver.solve(verbose=True)
    
    print("\n" + "="*70)
    print("COMFORT ANALYSIS")
    print("="*70)
    
    comfort = analyze_comfort(solver, geometry)
    
    print(f"\nComfort Score: {comfort.comfort_score:.1f}/100")
    print(f"Assessment: {comfort.assessment}")
    print(f"Mean air speed (occupied): {comfort.mean_air_speed_occupied:.4f} m/s")
    print(f"Stagnation: {comfort.stagnation_percentage:.1f}%")
    print(f"Max velocity: {comfort.max_air_speed:.3f} m/s")
    
    # Visualizations
    print("\nGenerating visualizations...")
    figures = create_visualizations(solver, geometry, output_dir)
    print(f"Generated {len(figures)} figures")
    
    # Summary
    summary = {
        'project_id': client.project_id,
        'client': client.client_name,
        'timestamp': datetime.now().isoformat(),
        'simulation': {
            'grid': f"{config.nx}×{config.ny}×{config.nz}",
            'converged': solve_result['converged'],
            'iterations': solve_result['iterations'],
            'wall_time_s': solve_result['elapsed'],
        },
        'results': {
            'max_velocity_ms': comfort.max_air_speed,
            'mean_velocity_occupied_ms': comfort.mean_air_speed_occupied,
            'stagnation_pct': comfort.stagnation_percentage,
            'comfort_score': comfort.comfort_score,
            'assessment': comfort.assessment,
        }
    }
    
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Output: {output_dir}")
    print(f"Comfort Score: {comfort.comfort_score:.1f}/100 - {comfort.assessment.split(' - ')[0]}")
    
    return {'summary': summary, 'comfort': comfort, 'solver': solver}


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_tier1_simulation(output_dir)
