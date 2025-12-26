"""
2D Riemann Quadrant Time Evolution via Strang Splitting

This demonstrates the full 2D CFD pipeline:
1. Initialize 2D Riemann problem (Configuration 3)
2. Compress to QTT2D with Morton ordering
3. Time-step using Strang splitting (X-half, Y-full, X-half)
4. Track compression ratio and rank evolution

Note: Uses dense round-trip for shifts until native MPO is complete.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import Tuple, List

from tensornet.cfd.qtt_2d import (
    dense_to_qtt_2d, qtt_2d_to_dense, QTT2DState,
    riemann_quadrant_ic, primitive_to_conservative_2d
)


@dataclass
class EulerState2D:
    """2D Euler conservative state in QTT format."""
    rho: QTT2DState   # Density
    rhou: QTT2DState  # X-momentum
    rhov: QTT2DState  # Y-momentum
    E: QTT2DState     # Total energy
    gamma: float = 1.4
    
    @property
    def max_rank(self) -> int:
        return max(
            self.rho.max_rank,
            self.rhou.max_rank,
            self.rhov.max_rank,
            self.E.max_rank
        )
    
    @property
    def total_memory_bytes(self) -> int:
        return (
            self.rho.memory_bytes +
            self.rhou.memory_bytes +
            self.rhov.memory_bytes +
            self.E.memory_bytes
        )
    
    @property
    def dense_memory_bytes(self) -> int:
        return 4 * self.rho.dense_memory_bytes()
    
    def compression_ratio(self) -> float:
        return self.dense_memory_bytes / self.total_memory_bytes


def shift_qtt2d_x(state: QTT2DState, shift: int = 1, max_rank: int = 64) -> QTT2DState:
    """Shift QTT2D in X direction via dense round-trip."""
    dense = qtt_2d_to_dense(state)
    shifted = torch.roll(dense, shifts=shift, dims=0)
    return dense_to_qtt_2d(shifted, max_bond=max_rank)


def shift_qtt2d_y(state: QTT2DState, shift: int = 1, max_rank: int = 64) -> QTT2DState:
    """Shift QTT2D in Y direction via dense round-trip."""
    dense = qtt_2d_to_dense(state)
    shifted = torch.roll(dense, shifts=shift, dims=1)
    return dense_to_qtt_2d(shifted, max_bond=max_rank)


def rusanov_flux_x_2d(
    rho: torch.Tensor,
    rhou: torch.Tensor,
    rhov: torch.Tensor,
    E: torch.Tensor,
    gamma: float = 1.4
) -> Tuple[torch.Tensor, ...]:
    """
    Compute Rusanov numerical flux in X direction.
    
    Returns flux differences: F_{i+1/2} - F_{i-1/2} for each cell.
    """
    # Primitive variables
    u = rhou / (rho + 1e-10)
    v = rhov / (rho + 1e-10)
    P = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
    P = torch.clamp(P, min=1e-10)
    
    # Sound speed
    c = torch.sqrt(gamma * P / (rho + 1e-10))
    
    # Maximum wave speed
    alpha = torch.abs(u) + c
    
    # Fluxes
    F_rho = rhou
    F_rhou = rhou * u + P
    F_rhov = rhov * u
    F_E = u * (E + P)
    
    # Left and right states (shifted)
    def flux_diff(F, U, alpha):
        # F_{i+1/2} = 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)
        F_L = F
        F_R = torch.roll(F, -1, dims=0)
        U_L = U
        U_R = torch.roll(U, -1, dims=0)
        alpha_max = torch.maximum(alpha, torch.roll(alpha, -1, dims=0))
        
        F_half = 0.5 * (F_L + F_R) - 0.5 * alpha_max * (U_R - U_L)
        
        # Flux difference: F_{i+1/2} - F_{i-1/2}
        return F_half - torch.roll(F_half, 1, dims=0)
    
    dF_rho = flux_diff(F_rho, rho, alpha)
    dF_rhou = flux_diff(F_rhou, rhou, alpha)
    dF_rhov = flux_diff(F_rhov, rhov, alpha)
    dF_E = flux_diff(F_E, E, alpha)
    
    return dF_rho, dF_rhou, dF_rhov, dF_E


def rusanov_flux_y_2d(
    rho: torch.Tensor,
    rhou: torch.Tensor,
    rhov: torch.Tensor,
    E: torch.Tensor,
    gamma: float = 1.4
) -> Tuple[torch.Tensor, ...]:
    """
    Compute Rusanov numerical flux in Y direction.
    """
    # Primitive variables
    u = rhou / (rho + 1e-10)
    v = rhov / (rho + 1e-10)
    P = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
    P = torch.clamp(P, min=1e-10)
    
    # Sound speed
    c = torch.sqrt(gamma * P / (rho + 1e-10))
    
    # Maximum wave speed (Y direction)
    alpha = torch.abs(v) + c
    
    # Y-direction fluxes
    G_rho = rhov
    G_rhou = rhou * v
    G_rhov = rhov * v + P
    G_E = v * (E + P)
    
    def flux_diff(G, U, alpha):
        G_L = G
        G_R = torch.roll(G, -1, dims=1)
        U_L = U
        U_R = torch.roll(U, -1, dims=1)
        alpha_max = torch.maximum(alpha, torch.roll(alpha, -1, dims=1))
        
        G_half = 0.5 * (G_L + G_R) - 0.5 * alpha_max * (U_R - U_L)
        return G_half - torch.roll(G_half, 1, dims=1)
    
    dG_rho = flux_diff(G_rho, rho, alpha)
    dG_rhou = flux_diff(G_rhou, rhou, alpha)
    dG_rhov = flux_diff(G_rhov, rhov, alpha)
    dG_E = flux_diff(G_E, E, alpha)
    
    return dG_rho, dG_rhou, dG_rhov, dG_E


def euler_step_x(
    rho: torch.Tensor,
    rhou: torch.Tensor,
    rhov: torch.Tensor,
    E: torch.Tensor,
    dt: float,
    dx: float,
    gamma: float = 1.4
) -> Tuple[torch.Tensor, ...]:
    """One Euler step in X direction."""
    dF_rho, dF_rhou, dF_rhov, dF_E = rusanov_flux_x_2d(rho, rhou, rhov, E, gamma)
    
    rho_new = rho - dt / dx * dF_rho
    rhou_new = rhou - dt / dx * dF_rhou
    rhov_new = rhov - dt / dx * dF_rhov
    E_new = E - dt / dx * dF_E
    
    return rho_new, rhou_new, rhov_new, E_new


def euler_step_y(
    rho: torch.Tensor,
    rhou: torch.Tensor,
    rhov: torch.Tensor,
    E: torch.Tensor,
    dt: float,
    dy: float,
    gamma: float = 1.4
) -> Tuple[torch.Tensor, ...]:
    """One Euler step in Y direction."""
    dG_rho, dG_rhou, dG_rhov, dG_E = rusanov_flux_y_2d(rho, rhou, rhov, E, gamma)
    
    rho_new = rho - dt / dy * dG_rho
    rhou_new = rhou - dt / dy * dG_rhou
    rhov_new = rhov - dt / dy * dG_rhov
    E_new = E - dt / dy * dG_E
    
    return rho_new, rhou_new, rhov_new, E_new


def strang_step(
    rho: torch.Tensor,
    rhou: torch.Tensor,
    rhov: torch.Tensor,
    E: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    gamma: float = 1.4
) -> Tuple[torch.Tensor, ...]:
    """
    One Strang splitting step:
    U^{n+1} = L_x(dt/2) L_y(dt) L_x(dt/2) U^n
    """
    # Half step in X
    rho, rhou, rhov, E = euler_step_x(rho, rhou, rhov, E, dt/2, dx, gamma)
    
    # Full step in Y
    rho, rhou, rhov, E = euler_step_y(rho, rhou, rhov, E, dt, dy, gamma)
    
    # Half step in X
    rho, rhou, rhov, E = euler_step_x(rho, rhou, rhov, E, dt/2, dx, gamma)
    
    return rho, rhou, rhov, E


def run_riemann_2d(
    nx: int = 7,
    ny: int = 7,
    n_steps: int = 100,
    cfl: float = 0.3,
    max_rank: int = 64,
    visualize: bool = True
) -> List[dict]:
    """
    Run 2D Riemann quadrant problem with QTT compression tracking.
    
    Returns list of diagnostics per step.
    """
    print("=" * 60)
    print("2D RIEMANN QUADRANT EVOLUTION")
    print("=" * 60)
    
    Nx, Ny = 2**nx, 2**ny
    gamma = 1.4
    
    # Physical domain [0,1] x [0,1]
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    
    # Initialize
    rho, u, v, P = riemann_quadrant_ic(nx, ny, config=3)
    rho_d, rhou_d, rhov_d, E_d = primitive_to_conservative_2d(rho, u, v, P, gamma)
    
    print(f"Grid: {Nx}×{Ny} = {Nx*Ny:,} points")
    print(f"CFL: {cfl}")
    print(f"Max QTT rank: {max_rank}")
    
    # Compute initial CFL time step
    u = rhou_d / (rho_d + 1e-10)
    v = rhov_d / (rho_d + 1e-10)
    P = (gamma - 1) * (E_d - 0.5 * rho_d * (u**2 + v**2))
    c = torch.sqrt(gamma * P / (rho_d + 1e-10))
    max_speed = (torch.abs(u) + torch.abs(v) + c).max().item()
    dt = cfl * min(dx, dy) / max_speed
    
    print(f"Time step: dt = {dt:.6f}")
    print(f"Simulation time: T = {n_steps * dt:.4f}")
    
    # Storage for diagnostics
    diagnostics = []
    snapshots = []
    
    # Time evolution
    for step in range(n_steps):
        # Strang step
        rho_d, rhou_d, rhov_d, E_d = strang_step(
            rho_d, rhou_d, rhov_d, E_d, dt, dx, dy, gamma
        )
        
        # Ensure positivity
        rho_d = torch.clamp(rho_d, min=1e-6)
        E_d = torch.clamp(E_d, min=1e-6)
        
        # Compress to QTT for diagnostics
        if step % 10 == 0 or step == n_steps - 1:
            rho_qtt = dense_to_qtt_2d(rho_d, max_bond=max_rank)
            
            diag = {
                'step': step,
                'time': (step + 1) * dt,
                'rho_min': rho_d.min().item(),
                'rho_max': rho_d.max().item(),
                'rank': rho_qtt.max_rank,
                'compression': rho_qtt.dense_memory_bytes() / rho_qtt.memory_bytes,
                'memory_kb': rho_qtt.memory_bytes / 1024
            }
            diagnostics.append(diag)
            
            if step % 20 == 0:
                print(f"  Step {step:4d}: ρ=[{diag['rho_min']:.3f}, {diag['rho_max']:.3f}], "
                      f"rank={diag['rank']}, compression={diag['compression']:.1f}×")
            
            if visualize and step % 25 == 0:
                snapshots.append((step, rho_d.clone()))
    
    print(f"\nFinal: rank={diagnostics[-1]['rank']}, compression={diagnostics[-1]['compression']:.1f}×")
    
    # Visualize snapshots
    if visualize and len(snapshots) > 0:
        n_snaps = min(len(snapshots), 4)
        fig, axes = plt.subplots(1, n_snaps, figsize=(4*n_snaps, 4))
        if n_snaps == 1:
            axes = [axes]
        
        for i, (step, rho_snap) in enumerate(snapshots[:n_snaps]):
            im = axes[i].imshow(rho_snap.T, origin='lower', cmap='viridis',
                               vmin=0.1, vmax=1.5)
            axes[i].set_title(f't = {step * dt:.3f}')
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        plt.suptitle(f'2D Riemann Quadrant — {Nx}×{Ny} Grid, Strang Splitting', fontsize=14)
        plt.tight_layout()
        plt.savefig('riemann_2d_evolution.png', dpi=150)
        print(f"\n✅ Saved: riemann_2d_evolution.png")
        plt.close()
    
    return diagnostics


def analyze_diagonal_shock():
    """
    Analyze rank growth for diagonal vs axis-aligned features.
    
    This is THE key test for the "Diagonal Problem."
    """
    print("\n" + "=" * 60)
    print("DIAGONAL SHOCK RANK ANALYSIS")
    print("=" * 60)
    
    nx, ny = 8, 8
    Nx, Ny = 2**nx, 2**ny
    
    x = torch.linspace(0, 1, Nx, dtype=torch.float32)
    y = torch.linspace(0, 1, Ny, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    print(f"Grid: {Nx}×{Ny}")
    print(f"\n{'Feature':<30} {'Rank':<8} {'Compression':<12}")
    print("-" * 55)
    
    # Vertical shock (x = 0.5)
    vertical = torch.where(X < 0.5, torch.ones_like(X), 0.1 * torch.ones_like(X))
    qtt_v = dense_to_qtt_2d(vertical, max_bond=64)
    print(f"{'Vertical shock (x=0.5)':<30} {qtt_v.max_rank:<8} {qtt_v.dense_memory_bytes()/qtt_v.memory_bytes:.1f}×")
    
    # Horizontal shock (y = 0.5)
    horizontal = torch.where(Y < 0.5, torch.ones_like(Y), 0.1 * torch.ones_like(Y))
    qtt_h = dense_to_qtt_2d(horizontal, max_bond=64)
    print(f"{'Horizontal shock (y=0.5)':<30} {qtt_h.max_rank:<8} {qtt_h.dense_memory_bytes()/qtt_h.memory_bytes:.1f}×")
    
    # 45° diagonal shock
    diagonal_45 = torch.where(X + Y < 1.0, torch.ones_like(X), 0.1 * torch.ones_like(X))
    qtt_d45 = dense_to_qtt_2d(diagonal_45, max_bond=64)
    print(f"{'45° diagonal shock':<30} {qtt_d45.max_rank:<8} {qtt_d45.dense_memory_bytes()/qtt_d45.memory_bytes:.1f}×")
    
    # 30° diagonal shock
    diagonal_30 = torch.where(X + 0.577*Y < 0.7, torch.ones_like(X), 0.1 * torch.ones_like(X))
    qtt_d30 = dense_to_qtt_2d(diagonal_30, max_bond=64)
    print(f"{'30° diagonal shock':<30} {qtt_d30.max_rank:<8} {qtt_d30.dense_memory_bytes()/qtt_d30.memory_bytes:.1f}×")
    
    # Circular shock (worst case)
    R = torch.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    circular = torch.where(R < 0.3, torch.ones_like(X), 0.1 * torch.ones_like(X))
    qtt_c = dense_to_qtt_2d(circular, max_bond=64)
    print(f"{'Circular shock (r=0.3)':<30} {qtt_c.max_rank:<8} {qtt_c.dense_memory_bytes()/qtt_c.memory_bytes:.1f}×")
    
    print("-" * 55)
    print("\nKEY INSIGHT:")
    print("  • Axis-aligned features: Low rank (1-2)")
    print("  • Diagonal features: Higher rank (scales with grid)")
    print("  • Circular features: Highest rank")
    print("  • This is the 'Diagonal Problem' in action!")
    
    # Visualize
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    fields = [vertical, horizontal, diagonal_45, diagonal_30, circular]
    qtts = [qtt_v, qtt_h, qtt_d45, qtt_d30, qtt_c]
    names = ['Vertical', 'Horizontal', '45° Diagonal', '30° Diagonal', 'Circular']
    
    for ax, field, qtt, name in zip(axes, fields, qtts, names):
        ax.imshow(field.T, origin='lower', cmap='viridis')
        ax.set_title(f'{name}\nrank={qtt.max_rank}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.suptitle('The Diagonal Problem: Feature Orientation vs QTT Rank', fontsize=14)
    plt.tight_layout()
    plt.savefig('diagonal_problem.png', dpi=150)
    print(f"\n✅ Saved: diagonal_problem.png")
    plt.close()


if __name__ == "__main__":
    # Run 2D Riemann evolution
    diagnostics = run_riemann_2d(
        nx=7, ny=7,  # 128×128
        n_steps=100,
        cfl=0.3,
        max_rank=64,
        visualize=True
    )
    
    # Analyze diagonal shock
    analyze_diagonal_shock()
