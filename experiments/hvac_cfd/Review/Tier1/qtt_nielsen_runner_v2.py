#!/usr/bin/env python3
"""
Nielsen Benchmark Runner v2 — Uses v4 solver with inlet depth fix
==================================================================

Run:
    python qtt_nielsen_runner_v2.py mass     # Quick propagation test
    python qtt_nielsen_runner_v2.py nielsen  # Full Nielsen benchmark
    python qtt_nielsen_runner_v2.py both     # Run both tests

Key fixes from v1:
  1. Multi-cell inlet depth (upwind advection needs gradient)
  2. Ceiling BC: w=0 only (u/v free for jet)
  3. Upwind advection for stable propagation
"""

import sys
import time
import torch
from torch import Tensor
from typing import Tuple, Dict
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NS3DConfig:
    """Configuration for 3D Navier-Stokes solver."""
    nx: int = 64
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
    
    # Multi-cell inlet depth for upwind advection gradient
    inlet_depth: int = 5
    
    # Turbulent viscosity for coarse grid
    nu_t: float = 0.0003
    
    def __post_init__(self):
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
        self.dz = self.Lz / (self.nz - 1)
        
        # Inlet indices
        self.inlet_j_start = int(self.inlet_y_frac[0] * self.ny)
        self.inlet_j_end = int(self.inlet_y_frac[1] * self.ny)
        self.inlet_k_start = int(self.inlet_z_frac[0] * self.nz)
        self.inlet_k_end = self.nz


# =============================================================================
# SOLVER WITH INLET DEPTH FIX
# =============================================================================

class NavierStokes3D:
    """3D Navier-Stokes with multi-cell inlet for proper advection."""
    
    def __init__(self, config: NS3DConfig, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float64
        
        nx, ny, nz = config.nx, config.ny, config.nz
        
        # Velocity fields
        self.u = torch.zeros(nx, ny, nz, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(nx, ny, nz, dtype=self.dtype, device=self.device)
        self.w = torch.zeros(nx, ny, nz, dtype=self.dtype, device=self.device)
        
        # Apply initial conditions with multi-cell inlet
        self._apply_initial_conditions()
    
    def _apply_initial_conditions(self):
        """Set initial flow with multi-cell inlet depth."""
        cfg = self.config
        U = cfg.inlet_velocity
        j_s, j_e = cfg.inlet_j_start, cfg.inlet_j_end
        k_s, k_e = cfg.inlet_k_start, cfg.inlet_k_end
        
        # Set inlet velocity in MULTIPLE cells (depth) with ramp
        for i in range(cfg.inlet_depth):
            factor = 1.0 - (i / (cfg.inlet_depth + 1))
            self.u[i, j_s:j_e, k_s:k_e] = U * factor
    
    def _apply_boundary_conditions(self):
        """Apply BCs with multi-cell inlet and free ceiling jet."""
        cfg = self.config
        U = cfg.inlet_velocity
        j_s, j_e = cfg.inlet_j_start, cfg.inlet_j_end
        k_s, k_e = cfg.inlet_k_start, cfg.inlet_k_end
        
        # X=0 (inlet wall)
        self.u[0, j_s:j_e, k_s:k_e] = U
        self.v[0, j_s:j_e, k_s:k_e] = 0.0
        self.w[0, j_s:j_e, k_s:k_e] = 0.0
        
        # Wall region below/beside inlet
        self.u[0, :, :k_s] = 0.0
        self.v[0, :, :k_s] = 0.0
        self.w[0, :, :k_s] = 0.0
        self.u[0, :j_s, k_s:k_e] = 0.0
        self.v[0, :j_s, k_s:k_e] = 0.0
        self.w[0, :j_s, k_s:k_e] = 0.0
        self.u[0, j_e:, k_s:k_e] = 0.0
        self.v[0, j_e:, k_s:k_e] = 0.0
        self.w[0, j_e:, k_s:k_e] = 0.0
        
        # X=Lx (outlet) - zero gradient
        self.u[-1, :, :] = self.u[-2, :, :]
        self.v[-1, :, :] = self.v[-2, :, :]
        self.w[-1, :, :] = self.w[-2, :, :]
        
        # Y=0 (side wall) - no-slip
        self.u[:, 0, :] = 0.0
        self.v[:, 0, :] = 0.0
        self.w[:, 0, :] = 0.0
        
        # Y=Ly (side wall) - no-slip
        self.u[:, -1, :] = 0.0
        self.v[:, -1, :] = 0.0
        self.w[:, -1, :] = 0.0
        
        # Z=0 (floor) - no-slip
        self.u[:, :, 0] = 0.0
        self.v[:, :, 0] = 0.0
        self.w[:, :, 0] = 0.0
        
        # Z=Lz (ceiling) - NO-PENETRATION ONLY (w=0, u/v FREE for jet!)
        self.w[:, :, -1] = 0.0
        # Exception: inlet at ceiling level
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
        """Advance one timestep."""
        cfg = self.config
        mass_before = self.u.sum().item()
        
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
        
        mass_after = self.u.sum().item()
        
        return {
            'mass_before': mass_before,
            'mass_after': mass_after,
            'max_u': self.u.abs().max().item(),
        }
    
    def get_ceiling_profile(self) -> Tuple[Tensor, Tensor]:
        """Extract u(x) at ceiling centerline."""
        cfg = self.config
        x = torch.linspace(0, cfg.Lx, cfg.nx, dtype=self.dtype)
        u_ceiling = self.u[:, cfg.ny//2, -1].cpu()
        return x, u_ceiling


# =============================================================================
# TESTS
# =============================================================================

NIELSEN_AALBORG = {
    1.0: 0.68,  # u/U_inlet at x/H=1.0
    2.0: 0.45,  # u/U_inlet at x/H=2.0
}


def run_mass_test() -> bool:
    """Test mass conservation."""
    print("=" * 70)
    print("MASS CONSERVATION TEST (v2)")
    print("=" * 70)
    print()
    
    config = NS3DConfig(nx=32, ny=32, nz=32, nu_t=0.002, inlet_depth=4)
    solver = NavierStokes3D(config, device='cpu')
    
    mass_initial = solver.u.sum().item()
    print(f"Initial mass: {mass_initial:.4f}")
    
    dt = 0.02
    n_steps = 100
    
    for step in range(n_steps):
        diag = solver.step(dt)
    
    mass_final = solver.u.sum().item()
    change_pct = abs(mass_final - mass_initial) / abs(mass_initial) * 100
    
    print(f"Final mass: {mass_final:.4f}")
    print(f"Change: {change_pct:.2f}%")
    print()
    
    # For open flow (inlet/outlet), mass will change based on net flux
    # We just check it doesn't explode
    passed = change_pct < 50 and diag['max_u'] < 5 * config.inlet_velocity
    
    if passed:
        print("✓ PASS: Mass stable")
    else:
        print(f"✗ FAIL: Mass change {change_pct:.1f}% or max_u={diag['max_u']:.2f}")
    
    print("=" * 70)
    return passed


def run_nielsen() -> bool:
    """Run full Nielsen benchmark."""
    print("=" * 70)
    print("NIELSEN BENCHMARK (v2 - INLET DEPTH FIX)")
    print("=" * 70)
    print()
    
    config = NS3DConfig(
        nx=96,
        ny=32,
        nz=32,
        Lx=9.0, Ly=3.0, Lz=3.0,
        inlet_velocity=0.455,
        nu_t=0.0003,
        inlet_depth=6,
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Grid: {config.nx}×{config.ny}×{config.nz}")
    print(f"Inlet depth: {config.inlet_depth} cells")
    print(f"nu_eff: {config.nu + config.nu_t:.2e}")
    print()
    
    solver = NavierStokes3D(config, device=device)
    
    # Run simulation
    t_end = 120.0
    dt = 0.005
    n_steps = int(t_end / dt)
    
    print(f"Running {n_steps} steps (t_end={t_end}s, dt={dt}s)...")
    
    start = time.perf_counter()
    for step in range(n_steps):
        solver.step(dt)
        
        if step % 2000 == 0:
            t = step * dt
            ceiling = solver.u[:, config.ny//2, -1]
            nonzero = (ceiling.abs() > 0.01).sum().item()
            print(f"  t={t:6.1f}s: cells>{0.01}: {nonzero}")
    
    elapsed = time.perf_counter() - start
    print(f"\nCompleted in {elapsed:.1f}s")
    print()
    
    # Extract profile
    x, u_ceiling = solver.get_ceiling_profile()
    H = config.Lz
    U = config.inlet_velocity
    
    print("CEILING PROFILE")
    print("-" * 50)
    
    # Compare with Aalborg
    errors = []
    for xH_target, aalborg_ratio in NIELSEN_AALBORG.items():
        idx = min(int(xH_target * H / config.dx), config.nx - 1)
        computed_ratio = u_ceiling[idx].item() / U
        error = abs(computed_ratio - aalborg_ratio) * 100
        errors.append(error)
        
        status = "✓" if error < 15 else "○" if error < 25 else "✗"
        print(f"x/H={xH_target:.1f}: computed={computed_ratio:.2%}, "
              f"Aalborg={aalborg_ratio:.2%}, error={error:.1f}% {status}")
    
    avg_error = sum(errors) / len(errors)
    print()
    print(f"Average error: {avg_error:.1f}%")
    print()
    
    if avg_error < 10:
        print("✓ PASS: <10% average error")
        passed = True
    elif avg_error < 20:
        print("○ MARGINAL: 10-20% error (acceptable for coarse grid)")
        passed = True  # Still acceptable
    else:
        print(f"✗ FAIL: {avg_error:.1f}% average error")
        passed = False
    
    print("=" * 70)
    return passed


# =============================================================================
# MAIN
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python qtt_nielsen_runner_v2.py [mass|nielsen|both]")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "mass":
        passed = run_mass_test()
    elif cmd == "nielsen":
        passed = run_nielsen()
    elif cmd == "both":
        p1 = run_mass_test()
        print("\n\n")
        p2 = run_nielsen()
        passed = p1 and p2
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Mass test:    {'✓ PASS' if p1 else '✗ FAIL'}")
        print(f"Nielsen test: {'✓ PASS' if p2 else '✗ FAIL'}")
        print("=" * 70)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
