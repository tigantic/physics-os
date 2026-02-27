#!/usr/bin/env python3
"""
QTT 3D Navier-Stokes — v4 (INLET DEPTH FIX)
===========================================

PREVIOUS FAILURES:
  v1: Periodic wrap at boundaries → 18% mass leak
  v2: Zeroed u at ceiling → killed ceiling jet  
  v3: Single-cell inlet → upwind sees no gradient, jet stuck

THIS FIX (v4):
  - Multi-cell inlet depth (3+ cells) so upwind advection has a gradient
  - Upwind advection for stable propagation
  - Ceiling BC: w=0 only, u/v free for jet

Key insight: Upwind advection at x=1 looks at (u[1] - u[0])/dx.
             If u[0] = U and u[1] = 0, gradient exists and jet propagates.
             But we need u[0] AND u[1] set initially, not just u[0].

Tag: [HVAC] [QTT] [3D] [V4-INLET-DEPTH]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List
import time
import torch
from torch import Tensor


@dataclass
class NS3DConfig:
    """Configuration for 3D Navier-Stokes solver."""
    nx: int = 32
    ny: int = 32
    nz: int = 32
    
    Lx: float = 9.0   # Room length (m)
    Ly: float = 3.0   # Room width (m)
    Lz: float = 3.0   # Room height (m)
    
    nu: float = 1.5e-5  # Kinematic viscosity (air)
    inlet_velocity: float = 0.455  # m/s
    
    # Inlet region (fractions of domain)
    inlet_y_frac: Tuple[float, float] = (0.25, 0.75)  # Middle 50% of width
    inlet_z_frac: Tuple[float, float] = (0.944, 1.0)  # Top 5.6% (Nielsen geometry)
    
    # Inlet depth in cells (NEW: multi-cell inlet for gradient)
    inlet_depth: int = 3
    
    # Turbulent viscosity for coarse grid
    nu_t: float = 0.002
    
    def __post_init__(self):
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
        self.dz = self.Lz / (self.nz - 1)
        
        # Inlet indices
        self.inlet_j_start = int(self.inlet_y_frac[0] * self.ny)
        self.inlet_j_end = int(self.inlet_y_frac[1] * self.ny)
        self.inlet_k_start = int(self.inlet_z_frac[0] * self.nz)
        self.inlet_k_end = self.nz


class NavierStokes3D:
    """
    3D Navier-Stokes with MULTI-CELL INLET for proper advection.
    """
    
    def __init__(self, config: NS3DConfig, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float64
        
        nx, ny, nz = config.nx, config.ny, config.nz
        
        # Velocity fields
        self.u = torch.zeros(nx, ny, nz, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(nx, ny, nz, dtype=self.dtype, device=self.device)
        self.w = torch.zeros(nx, ny, nz, dtype=self.dtype, device=self.device)
        
        # Apply initial conditions (multi-cell inlet)
        self._apply_initial_conditions()
    
    def _apply_initial_conditions(self):
        """Set initial flow field with multi-cell inlet depth."""
        cfg = self.config
        U = cfg.inlet_velocity
        j_s, j_e = cfg.inlet_j_start, cfg.inlet_j_end
        k_s, k_e = cfg.inlet_k_start, cfg.inlet_k_end
        
        # Set inlet velocity in MULTIPLE cells (depth) with ramp
        # This gives upwind advection a gradient to work with
        for i in range(cfg.inlet_depth):
            # Ramp factor: 1.0 at i=0, decreasing toward interior
            factor = 1.0 - (i / (cfg.inlet_depth + 1))
            self.u[i, j_s:j_e, k_s:k_e] = U * factor
    
    def _apply_boundary_conditions(self):
        """Apply boundary conditions with multi-cell inlet."""
        cfg = self.config
        U = cfg.inlet_velocity
        j_s, j_e = cfg.inlet_j_start, cfg.inlet_j_end
        k_s, k_e = cfg.inlet_k_start, cfg.inlet_k_end
        
        # === X=0 FACE (inlet wall) ===
        # Inlet region: fixed velocity (only at x=0)
        self.u[0, j_s:j_e, k_s:k_e] = U
        self.v[0, j_s:j_e, k_s:k_e] = 0.0
        self.w[0, j_s:j_e, k_s:k_e] = 0.0
        
        # Wall region (below/beside inlet): no-slip
        self.u[0, :, :k_s] = 0.0
        self.v[0, :, :k_s] = 0.0
        self.w[0, :, :k_s] = 0.0
        self.u[0, :j_s, k_s:k_e] = 0.0
        self.v[0, :j_s, k_s:k_e] = 0.0
        self.w[0, :j_s, k_s:k_e] = 0.0
        self.u[0, j_e:, k_s:k_e] = 0.0
        self.v[0, j_e:, k_s:k_e] = 0.0
        self.w[0, j_e:, k_s:k_e] = 0.0
        
        # === X=Lx FACE (outlet) ===
        # Zero gradient (convective outflow)
        self.u[-1, :, :] = self.u[-2, :, :]
        self.v[-1, :, :] = self.v[-2, :, :]
        self.w[-1, :, :] = self.w[-2, :, :]
        
        # === Y=0 FACE (side wall) ===
        self.u[:, 0, :] = 0.0
        self.v[:, 0, :] = 0.0
        self.w[:, 0, :] = 0.0
        
        # === Y=Ly FACE (side wall) ===
        self.u[:, -1, :] = 0.0
        self.v[:, -1, :] = 0.0
        self.w[:, -1, :] = 0.0
        
        # === Z=0 FACE (floor) ===
        self.u[:, :, 0] = 0.0
        self.v[:, :, 0] = 0.0
        self.w[:, :, 0] = 0.0
        
        # === Z=Lz FACE (ceiling) ===
        # NO-PENETRATION ONLY: w=0
        # u and v are FREE (jet flows along ceiling)
        self.w[:, :, -1] = 0.0
        
        # Exception: at inlet, set inlet velocity at ceiling level
        self.u[0, j_s:j_e, -1] = U
    
    def _laplacian(self, f: Tensor) -> Tensor:
        """7-point Laplacian."""
        cfg = self.config
        dx2, dy2, dz2 = cfg.dx**2, cfg.dy**2, cfg.dz**2
        
        return (
            (torch.roll(f, -1, dims=0) - 2*f + torch.roll(f, 1, dims=0)) / dx2 +
            (torch.roll(f, -1, dims=1) - 2*f + torch.roll(f, 1, dims=1)) / dy2 +
            (torch.roll(f, -1, dims=2) - 2*f + torch.roll(f, 1, dims=2)) / dz2
        )
    
    def _advection_upwind(self, phi: Tensor, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        """First-order upwind advection."""
        cfg = self.config
        dx, dy, dz = cfg.dx, cfg.dy, cfg.dz
        
        dphi_dx = torch.where(
            u > 0,
            (phi - torch.roll(phi, 1, dims=0)) / dx,
            (torch.roll(phi, -1, dims=0) - phi) / dx
        )
        dphi_dy = torch.where(
            v > 0,
            (phi - torch.roll(phi, 1, dims=1)) / dy,
            (torch.roll(phi, -1, dims=1) - phi) / dy
        )
        dphi_dz = torch.where(
            w > 0,
            (phi - torch.roll(phi, 1, dims=2)) / dz,
            (torch.roll(phi, -1, dims=2) - phi) / dz
        )
        
        return u * dphi_dx + v * dphi_dy + w * dphi_dz
    
    def step(self, dt: float) -> Dict:
        """Advance one timestep using explicit Euler."""
        cfg = self.config
        
        # Store for residual
        u_old = self.u.clone()
        v_old = self.v.clone()
        w_old = self.w.clone()
        
        # Advection (upwind)
        adv_u = self._advection_upwind(self.u, self.u, self.v, self.w)
        adv_v = self._advection_upwind(self.v, self.u, self.v, self.w)
        adv_w = self._advection_upwind(self.w, self.u, self.v, self.w)
        
        # Diffusion
        nu_eff = cfg.nu + cfg.nu_t
        lap_u = self._laplacian(self.u)
        lap_v = self._laplacian(self.v)
        lap_w = self._laplacian(self.w)
        
        # Update
        self.u = self.u + dt * (-adv_u + nu_eff * lap_u)
        self.v = self.v + dt * (-adv_v + nu_eff * lap_v)
        self.w = self.w + dt * (-adv_w + nu_eff * lap_w)
        
        # Apply BCs
        self._apply_boundary_conditions()
        
        # Diagnostics
        du = torch.abs(self.u - u_old).max().item()
        dv = torch.abs(self.v - v_old).max().item()
        dw = torch.abs(self.w - w_old).max().item()
        residual = max(du, dv, dw)
        
        # Sample points for ceiling jet
        idx_xH1 = min(int(1.0 * cfg.Lz / cfg.dx), cfg.nx - 1)
        idx_xH2 = min(int(2.0 * cfg.Lz / cfg.dx), cfg.nx - 1)
        
        return {
            'residual': residual,
            'max_u': self.u.abs().max().item(),
            'u_at_xH1': self.u[idx_xH1, cfg.ny//2, -1].item(),
            'u_at_xH2': self.u[idx_xH2, cfg.ny//2, -1].item(),
        }
    
    def run(self, t_end: float, dt: float, diag_interval: int = 100,
            verbose: bool = True) -> Dict:
        """Run simulation to t_end."""
        n_steps = int(t_end / dt)
        cfg = self.config
        
        if verbose:
            print(f"Grid: {cfg.nx}×{cfg.ny}×{cfg.nz}")
            print(f"Domain: {cfg.Lx}×{cfg.Ly}×{cfg.Lz} m")
            print(f"Inlet: u={cfg.inlet_velocity} m/s, depth={cfg.inlet_depth} cells")
            print(f"       y=[{cfg.inlet_j_start}:{cfg.inlet_j_end}], z=[{cfg.inlet_k_start}:{cfg.inlet_k_end}]")
            print(f"nu_eff = {cfg.nu + cfg.nu_t:.2e} m²/s")
            print(f"dt={dt:.4f}s, t_end={t_end:.1f}s, steps={n_steps}")
            print("-" * 70)
        
        start = time.perf_counter()
        
        for step in range(n_steps):
            diag = self.step(dt)
            
            if verbose and step % diag_interval == 0:
                t = step * dt
                # Count propagation
                ceiling_u = self.u[:, cfg.ny//2, -1]
                nonzero = (ceiling_u.abs() > 0.01).sum().item()
                print(f"t={t:6.1f}s: res={diag['residual']:.2e}, "
                      f"cells>{0.01:.2f}: {nonzero}, "
                      f"x/H=1: {diag['u_at_xH1']:.3f}, x/H=2: {diag['u_at_xH2']:.3f}")
        
        elapsed = time.perf_counter() - start
        
        if verbose:
            print("-" * 70)
            print(f"Completed in {elapsed:.1f}s")
        
        return {'elapsed': elapsed, 'final_diag': diag}
    
    def get_ceiling_profile(self) -> Tuple[Tensor, Tensor]:
        """Extract u(x) at ceiling centerline."""
        cfg = self.config
        x = torch.linspace(0, cfg.Lx, cfg.nx, dtype=self.dtype)
        u_ceiling = self.u[:, cfg.ny//2, -1].cpu()
        return x, u_ceiling


# =============================================================================
# NIELSEN BENCHMARK
# =============================================================================

NIELSEN_AALBORG = {
    1.0: 0.68,  # u/U_inlet at x/H=1.0 (ceiling level)
    2.0: 0.45,  # u/U_inlet at x/H=2.0
}


def run_mass_test():
    """Test that jet propagates without mass explosion."""
    
    print("=" * 70)
    print("MASS/PROPAGATION TEST — v4 (INLET DEPTH FIX)")
    print("=" * 70)
    print()
    print("Key fix: Multi-cell inlet depth gives upwind advection a gradient.")
    print()
    
    config = NS3DConfig(
        nx=32, ny=32, nz=32, 
        nu_t=0.002,
        inlet_depth=4,  # 4 cells deep
    )
    solver = NavierStokes3D(config, device='cpu')
    
    print("Initial ceiling profile:")
    for i in [0, 1, 2, 3, 4, 5, 10, 15]:
        if i < config.nx:
            print(f"  x[{i}]: u = {solver.u[i, config.ny//2, -1].item():.4f}")
    print()
    
    dt = 0.02
    print("Running 300 steps...")
    print()
    
    for step in range(300):
        diag = solver.step(dt)
        
        if step % 50 == 0:
            ceiling = solver.u[:, config.ny//2, -1]
            nonzero = (ceiling.abs() > 0.01).sum().item()
            print(f"Step {step:3d}: max_u={diag['max_u']:.3f}, "
                  f"cells with u>0.01: {nonzero}, "
                  f"x/H=1: {diag['u_at_xH1']:.3f}, x/H=2: {diag['u_at_xH2']:.3f}")
    
    print()
    print("Final ceiling profile:")
    for i in [0, 3, 6, 9, 12, 15, 20, 25, 30]:
        if i < config.nx:
            u_val = solver.u[i, config.ny//2, -1].item()
            xH = i * config.dx / config.Lz
            print(f"  x[{i}] (x/H={xH:.2f}): u = {u_val:.4f}")
    
    # Check if jet propagated
    u_at_xH1 = solver.u[int(config.Lz / config.dx), config.ny//2, -1].item()
    
    print()
    if u_at_xH1 > 0.1:
        print(f"✓ Jet propagated to x/H=1.0 (u = {u_at_xH1:.3f} m/s)")
    else:
        print(f"✗ Jet did NOT propagate to x/H=1.0 (u = {u_at_xH1:.3f} m/s)")
    
    print("=" * 70)
    return solver


def run_nielsen():
    """Run Nielsen benchmark with higher resolution."""
    
    print("=" * 70)
    print("NIELSEN BENCHMARK — v4 (INLET DEPTH FIX)")
    print("=" * 70)
    print()
    print("Key fix: Multi-cell inlet depth for advection gradient.")
    print()
    
    config = NS3DConfig(
        nx=96,  # Higher resolution
        ny=32,
        nz=32,
        Lx=9.0,
        Ly=3.0, 
        Lz=3.0,
        inlet_velocity=0.455,
        nu_t=0.0003,  # Lower turbulent viscosity
        inlet_depth=6,  # 6 cells deep
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    solver = NavierStokes3D(config, device=device)
    
    # Run simulation - longer to reach steady state
    solver.run(t_end=120.0, dt=0.005, diag_interval=1000, verbose=True)
    
    # Extract and display results
    x, u_ceiling = solver.get_ceiling_profile()
    H = config.Lz
    U = config.inlet_velocity
    
    print()
    print("CEILING VELOCITY PROFILE (z = Lz)")
    print("-" * 50)
    print(f"{'x/H':<10} {'u (m/s)':<12} {'u/U_inlet':<12}")
    print("-" * 50)
    
    for i in range(0, len(x), max(1, len(x)//15)):
        xH = x[i].item() / H
        u_val = u_ceiling[i].item()
        ratio = u_val / U
        print(f"{xH:<10.2f} {u_val:<12.4f} {ratio:<12.2%}")
    
    # Compare with Aalborg
    print()
    print("COMPARISON WITH AALBORG EXPERIMENTAL DATA")
    print("-" * 50)
    
    errors = []
    for xH_target, aalborg_ratio in NIELSEN_AALBORG.items():
        idx = min(int(xH_target * H / config.dx), config.nx - 1)
        computed_ratio = u_ceiling[idx].item() / U
        error = abs(computed_ratio - aalborg_ratio) * 100
        errors.append(error)
        
        status = "✓" if error < 15 else "○" if error < 30 else "✗"
        print(f"x/H={xH_target:.1f}: computed={computed_ratio:.2%}, "
              f"Aalborg={aalborg_ratio:.2%}, error={error:.1f}% {status}")
    
    avg_error = sum(errors) / len(errors)
    
    print()
    print(f"Average error: {avg_error:.1f}%")
    print()
    
    if avg_error < 10:
        print("✓ PASS: <10% average error")
    elif avg_error < 20:
        print("○ MARGINAL: 10-20% error (acceptable for coarse grid)")
    elif avg_error < 35:
        print("○ FAIR: 20-35% error (needs grid refinement)")
    else:
        print("✗ FAIL: >35% error")
    
    print("=" * 70)
    
    return solver


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "mass":
            run_mass_test()
        elif sys.argv[1] == "nielsen":
            run_nielsen()
        elif sys.argv[1] == "both":
            run_mass_test()
            print("\n\n")
            run_nielsen()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python script.py [mass|nielsen|both]")
    else:
        run_nielsen()
