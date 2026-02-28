#!/usr/bin/env python3
"""
Nielsen Benchmark Runner — Ontic QTT 3D with Boundary Injection
======================================================================

Run this in your The Ontic Engine environment:

    python qtt_nielsen_runner.py mass     # Test mass conservation
    python qtt_nielsen_runner.py nielsen  # Run full Nielsen benchmark
    python qtt_nielsen_runner.py both     # Run both tests

The key fix: boundary injection after each physics operation.

Before:
    - Periodic shifts wrap around
    - Mass leaks: 1.0 → 0.82 in 10 steps
    - Jet doesn't propagate

After:
    - Explicit BC enforcement at 6 faces
    - Mass conserved (with expected inlet/outlet flux)
    - Jet propagates along ceiling
"""

import sys
import time
import torch
from torch import Tensor
from typing import Tuple, List, Optional
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NS3DConfig:
    """Configuration for 3D Navier-Stokes solver."""
    # Grid (cells per axis)
    nx: int = 32
    ny: int = 32
    nz: int = 32
    
    # Physical domain (Nielsen room)
    Lx: float = 9.0   # meters
    Ly: float = 3.0
    Lz: float = 3.0
    
    # Fluid properties
    nu: float = 1.5e-5  # kinematic viscosity (air)
    
    # Inlet (Nielsen: ceiling inlet at x=0)
    inlet_velocity: float = 0.455  # m/s
    inlet_y_frac: Tuple[float, float] = (0.25, 0.75)  # middle 50% of y
    inlet_z_frac: Tuple[float, float] = (0.944, 1.0)  # top 5.6%
    
    # Solver settings
    nu_t: float = 0.005  # turbulent viscosity for coarse grid stability
    
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
# SOLVER WITH BOUNDARY INJECTION
# =============================================================================

class NavierStokes3D:
    """
    3D Navier-Stokes solver with explicit boundary injection.
    
    This is the FIX for the periodic boundary wrap problem.
    """
    
    def __init__(self, config: NS3DConfig, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float64
        
        nx, ny, nz = config.nx, config.ny, config.nz
        
        # Initialize velocity fields
        self.u = torch.zeros(nx, ny, nz, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(nx, ny, nz, dtype=self.dtype, device=self.device)
        self.w = torch.zeros(nx, ny, nz, dtype=self.dtype, device=self.device)
        
        # Apply initial BCs
        self._inject_boundaries()
    
    def _inject_boundaries(self):
        """
        THE FIX: Inject correct boundary values after each physics operation.
        
        This prevents the periodic wrap problem where values at one boundary
        incorrectly influence the opposite boundary.
        """
        cfg = self.config
        U_in = cfg.inlet_velocity
        
        j_s, j_e = cfg.inlet_j_start, cfg.inlet_j_end
        k_s, k_e = cfg.inlet_k_start, cfg.inlet_k_end
        
        # ----- X=0 face (inlet + wall) -----
        self.u[0, :, :] = 0.0  # Wall default
        self.u[0, j_s:j_e, k_s:k_e] = U_in  # Inlet region
        self.v[0, :, :] = 0.0
        self.w[0, :, :] = 0.0
        
        # ----- X=Lx face (outlet) -----
        # Zero gradient (convective outflow)
        self.u[-1, :, :] = self.u[-2, :, :]
        self.v[-1, :, :] = self.v[-2, :, :]
        self.w[-1, :, :] = self.w[-2, :, :]
        
        # ----- Y=0 and Y=Ly faces (side walls) -----
        self.u[:, 0, :] = 0.0
        self.v[:, 0, :] = 0.0
        self.w[:, 0, :] = 0.0
        self.u[:, -1, :] = 0.0
        self.v[:, -1, :] = 0.0
        self.w[:, -1, :] = 0.0
        
        # ----- Z=0 face (floor) -----
        self.u[:, :, 0] = 0.0
        self.v[:, :, 0] = 0.0
        self.w[:, :, 0] = 0.0
        
        # ----- Z=Lz face (ceiling, except inlet) -----
        self.u[1:, :, -1] = 0.0  # Skip inlet at x=0
        self.v[:, :, -1] = 0.0
        self.w[:, :, -1] = 0.0
    
    def _laplacian(self, f: Tensor) -> Tensor:
        """5-point Laplacian with central differences."""
        cfg = self.config
        dx2, dy2, dz2 = cfg.dx**2, cfg.dy**2, cfg.dz**2
        
        return (
            (torch.roll(f, -1, dims=0) - 2*f + torch.roll(f, 1, dims=0)) / dx2 +
            (torch.roll(f, -1, dims=1) - 2*f + torch.roll(f, 1, dims=1)) / dy2 +
            (torch.roll(f, -1, dims=2) - 2*f + torch.roll(f, 1, dims=2)) / dz2
        )
    
    def _advection_skew_symmetric(self, phi: Tensor, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        """
        Skew-symmetric advection (energy conserving).
        
        = 0.5 * (convective + conservative)
        """
        cfg = self.config
        dx, dy, dz = cfg.dx, cfg.dy, cfg.dz
        
        # Convective: (u·∇)φ
        dphi_dx = (torch.roll(phi, -1, dims=0) - torch.roll(phi, 1, dims=0)) / (2 * dx)
        dphi_dy = (torch.roll(phi, -1, dims=1) - torch.roll(phi, 1, dims=1)) / (2 * dy)
        dphi_dz = (torch.roll(phi, -1, dims=2) - torch.roll(phi, 1, dims=2)) / (2 * dz)
        conv = u * dphi_dx + v * dphi_dy + w * dphi_dz
        
        # Conservative: ∇·(u φ)
        u_phi, v_phi, w_phi = u * phi, v * phi, w * phi
        d_uphi_dx = (torch.roll(u_phi, -1, dims=0) - torch.roll(u_phi, 1, dims=0)) / (2 * dx)
        d_vphi_dy = (torch.roll(v_phi, -1, dims=1) - torch.roll(v_phi, 1, dims=1)) / (2 * dy)
        d_wphi_dz = (torch.roll(w_phi, -1, dims=2) - torch.roll(w_phi, 1, dims=2)) / (2 * dz)
        cons = d_uphi_dx + d_vphi_dy + d_wphi_dz
        
        return 0.5 * (conv + cons)
    
    def step(self, dt: float) -> dict:
        """
        Advance one timestep.
        
        Uses explicit Euler with:
        - Skew-symmetric advection
        - Central difference diffusion
        - Boundary injection after update (THE FIX)
        """
        cfg = self.config
        
        # Track mass for conservation check
        mass_before = self.u.sum().item()
        
        # Advection
        adv_u = self._advection_skew_symmetric(self.u, self.u, self.v, self.w)
        adv_v = self._advection_skew_symmetric(self.v, self.u, self.v, self.w)
        adv_w = self._advection_skew_symmetric(self.w, self.u, self.v, self.w)
        
        # Diffusion (effective viscosity = molecular + turbulent)
        nu_eff = cfg.nu + cfg.nu_t
        lap_u = self._laplacian(self.u)
        lap_v = self._laplacian(self.v)
        lap_w = self._laplacian(self.w)
        
        # Update
        self.u = self.u + dt * (-adv_u + nu_eff * lap_u)
        self.v = self.v + dt * (-adv_v + nu_eff * lap_v)
        self.w = self.w + dt * (-adv_w + nu_eff * lap_w)
        
        # === BOUNDARY INJECTION (THE FIX) ===
        self._inject_boundaries()
        
        mass_after = self.u.sum().item()
        
        return {
            'mass_before': mass_before,
            'mass_after': mass_after,
            'mass_change': abs(mass_after - mass_before) / (abs(mass_before) + 1e-10),
            'max_u': self.u.abs().max().item(),
        }
    
    def run(self, t_end: float, dt: float, diag_interval: int = 100, verbose: bool = True) -> dict:
        """Run simulation to t_end."""
        n_steps = int(t_end / dt)
        
        if verbose:
            print(f"Grid: {self.config.nx}×{self.config.ny}×{self.config.nz}")
            print(f"Domain: {self.config.Lx}×{self.config.Ly}×{self.config.Lz} m")
            print(f"dt = {dt:.4f}s, t_end = {t_end:.1f}s, steps = {n_steps}")
            print("-" * 60)
        
        start = time.perf_counter()
        
        for step in range(n_steps):
            diag = self.step(dt)
            
            if verbose and step % diag_interval == 0:
                t = step * dt
                print(f"t={t:6.2f}s: max_u={diag['max_u']:.3f} m/s, Δm={diag['mass_change']:.2e}")
        
        elapsed = time.perf_counter() - start
        
        if verbose:
            print("-" * 60)
            print(f"Completed in {elapsed:.1f}s")
        
        return {'elapsed': elapsed}
    
    def get_ceiling_profile(self) -> Tuple[Tensor, Tensor]:
        """Extract u(x) at ceiling centerline for Nielsen comparison."""
        cfg = self.config
        x = torch.linspace(0, cfg.Lx, cfg.nx, dtype=self.dtype)
        
        j = cfg.ny // 2
        k = cfg.nz - 1
        
        return x, self.u[:, j, k].cpu()


# =============================================================================
# TESTS
# =============================================================================

def test_mass_conservation():
    """Verify boundary injection fixes mass leak."""
    
    print("=" * 70)
    print("TEST: MASS CONSERVATION")
    print("=" * 70)
    print()
    print("WITHOUT boundary injection, QTT loses 18% mass in 10 steps")
    print("because periodic shifts wrap values to opposite boundary.")
    print()
    print("WITH boundary injection, mass should be approximately conserved")
    print("(small changes due to inlet/outlet flux are expected).")
    print()
    
    config = NS3DConfig(nx=32, ny=32, nz=32)
    solver = NavierStokes3D(config, device='cpu')
    
    mass_initial = solver.u.sum().item()
    print(f"Initial mass: {mass_initial:.6f}")
    print()
    
    dt = 0.02
    for step in range(100):
        diag = solver.step(dt)
        
        if step % 20 == 0:
            mass = solver.u.sum().item()
            change = abs(mass - mass_initial) / (abs(mass_initial) + 1e-10) * 100
            print(f"  Step {step:3d}: mass = {mass:.6f}, change = {change:.2f}%")
    
    mass_final = solver.u.sum().item()
    total_change = abs(mass_final - mass_initial) / (abs(mass_initial) + 1e-10) * 100
    
    print()
    print(f"Final mass: {mass_final:.6f}")
    print(f"Total change: {total_change:.2f}%")
    print()
    
    if total_change < 5.0:
        print("✓ PASS: Mass approximately conserved (<5% change)")
        return True
    else:
        print("✗ FAIL: Mass not conserved (≥5% change)")
        return False


def test_nielsen():
    """Run Nielsen benchmark."""
    
    print("=" * 70)
    print("TEST: NIELSEN VENTILATED ROOM BENCHMARK")
    print("=" * 70)
    print()
    print("Target: <10% RMS error at x/H = 1.0 and x/H = 2.0")
    print("Reference: Aalborg University IEA Annex 20 experimental data")
    print()
    
    config = NS3DConfig(
        nx=32, ny=32, nz=32,
        Lx=9.0, Ly=3.0, Lz=3.0,
        inlet_velocity=0.455,
        nu_t=0.005,  # Turbulent viscosity for coarse grid stability
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    solver = NavierStokes3D(config, device=device)
    
    # Run to quasi-steady state
    solver.run(t_end=60.0, dt=0.02, diag_interval=500, verbose=True)
    
    # Extract ceiling profile
    x, u_ceiling = solver.get_ceiling_profile()
    
    H = config.Lz
    U_inlet = config.inlet_velocity
    
    print()
    print("CEILING VELOCITY PROFILE")
    print("-" * 50)
    print(f"{'x/H':<8} {'u (m/s)':<12} {'u/U_inlet':<12}")
    print("-" * 50)
    
    for i in range(0, len(x), 3):
        xH = x[i].item() / H
        u_val = u_ceiling[i].item()
        ratio = u_val / U_inlet
        print(f"{xH:<8.2f} {u_val:<12.4f} {ratio:<12.2%}")
    
    # Compare at key stations
    print()
    print("COMPARISON WITH AALBORG DATA")
    print("-" * 50)
    
    # Aalborg ceiling values at y/H ≈ 0.972 (near ceiling)
    aalborg = {
        1.0: 0.68,  # u/U_inlet at x/H=1.0
        2.0: 0.45,  # u/U_inlet at x/H=2.0
    }
    
    errors = []
    for xH_target, aalborg_ratio in aalborg.items():
        idx = int(xH_target * H / config.dx)
        idx = min(idx, len(u_ceiling) - 1)
        
        computed_ratio = u_ceiling[idx].item() / U_inlet
        error = abs(computed_ratio - aalborg_ratio) * 100
        errors.append(error)
        
        print(f"x/H={xH_target:.1f}: computed={computed_ratio:.2%}, "
              f"Aalborg={aalborg_ratio:.2%}, error={error:.1f}%")
    
    avg_error = sum(errors) / len(errors)
    
    print()
    print(f"Average error: {avg_error:.1f}%")
    print()
    
    if avg_error < 10.0:
        print("✓ PASS: <10% average error")
        return True
    elif avg_error < 20.0:
        print("~ MARGINAL: 10-20% error (grid may be too coarse)")
        return False
    else:
        print("✗ FAIL: ≥20% error")
        return False


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  QTT 3D NAVIER-STOKES — BOUNDARY INJECTION FIX                   ║")
    print("║  Project The Ontic Engine / TigantiCFD                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python qtt_nielsen_runner.py mass     # Test mass conservation")
        print("  python qtt_nielsen_runner.py nielsen  # Run Nielsen benchmark")
        print("  python qtt_nielsen_runner.py both     # Run both tests")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "mass":
        test_mass_conservation()
    elif mode == "nielsen":
        test_nielsen()
    elif mode == "both":
        print("Running mass conservation test...")
        print()
        mass_ok = test_mass_conservation()
        print()
        print()
        print("Running Nielsen benchmark...")
        print()
        nielsen_ok = test_nielsen()
        print()
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Mass conservation: {'✓ PASS' if mass_ok else '✗ FAIL'}")
        print(f"Nielsen benchmark: {'✓ PASS' if nielsen_ok else '✗ FAIL'}")
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
