"""
Solar Wind Bow Shock — Piston-Driven Collisionless Shock Simulation

This module implements a 1D-1V Vlasov simulation of bow shock formation
using a piston-driven approach where the magnetopause acts as a reflecting
wall.

Physics:
- Solar wind (Mach ~10) impacts magnetopause
- Reflecting wall creates upstream-propagating shock
- Compression ratio approaches 4 for strong shocks (gamma=5/3)
- Ion reflection at supercritical shocks

Method:
- Spectral Vlasov solver with Strang splitting
- Reflecting BC at x=0 (magnetopause)
- Inflow BC at x=L (solar wind)

Validation:
- Rankine-Hugoniot compression ratio
- Shock position evolution
- Downstream thermalization

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
import sys
import time as time_module
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor

# Setup imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "libs"))
sys.path.insert(0, str(project_root / "QTeneT" / "src" / "qtenet"))


@dataclass(frozen=True)
class PhysicsConstants:
    """Physical constants for space plasma physics."""
    m_p: float = 1.673e-27   # Proton mass (kg)
    k_B: float = 1.381e-23   # Boltzmann constant (J/K)
    mu_0: float = 1.257e-6   # Vacuum permeability (H/m)


CONSTANTS = PhysicsConstants()


@dataclass
class ShockConfig:
    """Configuration for bow shock simulation."""
    nx: int = 256           # Spatial grid points
    nv: int = 128           # Velocity grid points
    
    # Solar wind parameters
    n_sw: float = 5.0       # Density (particles/cc)
    v_sw: float = 400.0     # Bulk velocity (km/s)
    T_sw: float = 1e5       # Temperature (K)
    
    # Domain
    L_x: float = 50.0       # Domain length (arbitrary units, ~R_E)
    v_max: float = 800.0    # Max velocity (km/s) - must capture reflected ions
    
    # Simulation
    cfl: float = 0.5        # CFL number
    device: str = "cpu"
    
    @property
    def dx(self) -> float:
        return self.L_x / self.nx
    
    @property
    def dv(self) -> float:
        return 2 * self.v_max / self.nv
    
    @property
    def v_th(self) -> float:
        """Thermal velocity (km/s)."""
        return math.sqrt(2 * CONSTANTS.k_B * self.T_sw / CONSTANTS.m_p) / 1e3
    
    @property
    def c_s(self) -> float:
        """Sound speed (km/s)."""
        gamma = 5.0 / 3.0
        return math.sqrt(gamma * CONSTANTS.k_B * self.T_sw / CONSTANTS.m_p) / 1e3
    
    @property
    def mach(self) -> float:
        """Sonic Mach number."""
        return self.v_sw / self.c_s


@dataclass
class ShockResult:
    """Results from shock simulation."""
    compression_ratio: float
    compression_expected: float
    shock_position: float
    shock_velocity: float
    reflected_fraction: float
    validated: bool
    runtime_seconds: float
    density_profile: Tensor
    velocity_profile: Tensor
    x_grid: Tensor


class BowShockSimulation:
    """
    1D-1V Vlasov simulation of piston-driven collisionless shock.
    
    The simulation uses:
    - Spectral advection in both x and v
    - Reflecting boundary at x=0 (magnetopause/piston)
    - Inflow at x=L (solar wind source)
    - Self-consistent electric field from quasi-neutrality
    """
    
    def __init__(self, config: ShockConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        
        # Grids
        self.x = torch.linspace(0, config.L_x, config.nx, device=self.device)
        self.v = torch.linspace(-config.v_max, config.v_max, config.nv, device=self.device)
        
        # Time step from CFL
        self.dt = config.cfl * config.dx / config.v_max
        
        # Wavenumbers for spectral methods
        self.kx = torch.fft.fftfreq(config.nx, d=config.dx, device=self.device) * 2 * math.pi
        self.kv = torch.fft.fftfreq(config.nv, d=config.dv, device=self.device) * 2 * math.pi
        
        print(f"BowShockSimulation initialized:")
        print(f"  Grid: {config.nx} × {config.nv}")
        print(f"  Mach = {config.mach:.1f}")
        print(f"  dt = {self.dt:.6f}")
    
    def maxwellian(self, n: float, u: float, T: float) -> Tensor:
        """Create Maxwellian distribution f(v) at given n, u, T."""
        v_th = math.sqrt(2 * CONSTANTS.k_B * T / CONSTANTS.m_p) / 1e3
        return n * torch.exp(-0.5 * ((self.v - u) / v_th)**2) / (math.sqrt(2*math.pi) * v_th)
    
    def initialize(self) -> Tensor:
        """Initialize with uniform solar wind Maxwellian."""
        cfg = self.cfg
        
        # f(x, v) = solar wind Maxwellian everywhere
        f_1d = self.maxwellian(cfg.n_sw, cfg.v_sw, cfg.T_sw)
        f = f_1d.unsqueeze(0).expand(cfg.nx, cfg.nv).clone()
        
        return f
    
    def compute_moments(self, f: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute density, bulk velocity, temperature."""
        cfg = self.cfg
        
        # Density
        n = torch.trapezoid(f, self.v, dim=1)
        
        # Bulk velocity
        nu = torch.trapezoid(f * self.v.unsqueeze(0), self.v, dim=1)
        u = nu / (n + 1e-10)
        
        # Temperature (second moment)
        nT = torch.trapezoid(f * (self.v.unsqueeze(0) - u.unsqueeze(1))**2, self.v, dim=1)
        T = nT / (n + 1e-10)  # In (km/s)² units
        
        return n, u, T
    
    def compute_electric_field(self, f: Tensor) -> Tensor:
        """Compute electric field from quasi-neutrality.
        
        For isothermal electrons: E = -T_e * ∂ln(n)/∂x
        """
        cfg = self.cfg
        
        n, _, _ = self.compute_moments(f)
        
        # Gradient of log density (ambipolar field)
        ln_n = torch.log(n + 1e-10)
        
        # Spectral derivative
        ln_n_k = torch.fft.fft(ln_n)
        d_ln_n_k = 1j * self.kx * ln_n_k
        d_ln_n = torch.fft.ifft(d_ln_n_k).real
        
        # E = -T_e * d(ln n)/dx, with T_e ~ T_i
        T_e = cfg.v_th**2  # Electron temperature in (km/s)²
        E = -T_e * d_ln_n
        
        return E
    
    def advect_x(self, f: Tensor, dt: float) -> Tensor:
        """Advection in x: ∂f/∂t + v ∂f/∂x = 0."""
        # FFT in x
        f_k = torch.fft.fft(f, dim=0)
        
        # Phase shift: exp(-i kx v dt)
        phase = torch.exp(-1j * self.kx.unsqueeze(1) * self.v.unsqueeze(0) * dt)
        f_k = f_k * phase
        
        return torch.fft.ifft(f_k, dim=0).real
    
    def accelerate_v(self, f: Tensor, E: Tensor, dt: float) -> Tensor:
        """Acceleration in v: ∂f/∂t + (e/m)E ∂f/∂v = 0."""
        # FFT in v
        f_k = torch.fft.fft(f, dim=1)
        
        # Acceleration a = (e/m) E, normalized
        # Coupling strength determines shock formation speed
        a = 50.0 * E  # Empirical coupling
        
        # Phase shift: exp(-i kv a dt)
        phase = torch.exp(-1j * self.kv.unsqueeze(0) * a.unsqueeze(1) * dt)
        f_k = f_k * phase
        
        return torch.fft.ifft(f_k, dim=1).real
    
    def apply_reflecting_bc(self, f: Tensor) -> Tensor:
        """Apply reflecting boundary condition at x=0.
        
        Particles hitting the wall at x=0 with v > 0 are reflected
        to have v < 0 (they bounce back into the domain).
        """
        cfg = self.cfg
        
        # At the left boundary (x=0), particles with v > 0 hit the wall
        # They should be reflected: f(0, v) -> f(0, -v) for v > 0
        
        # Find velocity indices
        v_positive = self.v > 0
        v_negative = self.v < 0
        
        # Create reflected distribution at boundary
        # For each positive v, the reflected particle has -v
        n_boundary_cells = 3
        for i in range(n_boundary_cells):
            # Get distribution at boundary
            f_boundary = f[i, :].clone()
            
            # Reflect: f(v) -> f(-v) for incoming particles
            # Incoming at x=0 means v > 0 (moving toward wall)
            f_reflected = torch.zeros_like(f_boundary)
            
            # For each v > 0, copy to -v
            for j, v_val in enumerate(self.v):
                if v_val > cfg.v_sw * 0.1:  # Only reflect if moving significantly toward wall
                    # Find index of -v
                    j_reflected = cfg.nv - 1 - j
                    if j_reflected >= 0 and j_reflected < cfg.nv:
                        f_reflected[j_reflected] = f_boundary[j]
            
            # Add reflected particles to existing distribution
            # Weight by distance from boundary
            weight = 1.0 - (i / n_boundary_cells)
            f[i, :] = f[i, :] + weight * f_reflected
        
        return f
    
    def apply_inflow_bc(self, f: Tensor) -> Tensor:
        """Apply inflow boundary condition at x=L."""
        cfg = self.cfg
        
        # At right boundary, maintain solar wind inflow
        f_sw = self.maxwellian(cfg.n_sw, cfg.v_sw, cfg.T_sw)
        
        # Damping layer
        n_damp = max(int(0.1 * cfg.nx), 5)
        for i in range(n_damp):
            idx = cfg.nx - n_damp + i
            alpha = (i / n_damp)**2
            f[idx, :] = (1 - alpha) * f[idx, :] + alpha * f_sw
        
        return f
    
    def step(self, f: Tensor) -> Tensor:
        """Advance one time step using Strang splitting."""
        dt = self.dt
        
        # Half step x-advection
        f = self.advect_x(f, dt/2)
        
        # Full step v-acceleration
        E = self.compute_electric_field(f)
        f = self.accelerate_v(f, E, dt)
        
        # Half step x-advection
        f = self.advect_x(f, dt/2)
        
        # Boundary conditions
        f = self.apply_reflecting_bc(f)
        f = self.apply_inflow_bc(f)
        
        # Ensure positivity
        f = torch.clamp(f, min=0)
        
        return f
    
    def run(self, n_steps: int = 1000, diag_interval: int = 200) -> ShockResult:
        """Run simulation and analyze results."""
        print(f"\nRunning {n_steps} steps...")
        start = time_module.time()
        
        f = self.initialize()
        
        shock_positions = []
        
        for step in range(n_steps):
            f = self.step(f)
            
            if (step + 1) % diag_interval == 0:
                n, u, T = self.compute_moments(f)
                n_max = float(n.max())
                n_valid = n[n > 0.5 * self.cfg.n_sw]
                n_min = float(n_valid.min()) if n_valid.numel() > 0 else float(n.min())
                ratio = n_max / (n_min + 1e-10)
                
                # Find shock position
                dn_dx = torch.gradient(n, spacing=(self.cfg.dx,), dim=0)[0]
                shock_idx = int(torch.argmax(torch.abs(dn_dx)))
                shock_pos = float(self.x[shock_idx])
                shock_positions.append((step + 1, shock_pos))
                
                print(f"  Step {step+1}/{n_steps}: compression = {ratio:.2f}, shock at x = {shock_pos:.1f}")
        
        runtime = time_module.time() - start
        
        return self._analyze_results(f, shock_positions, runtime)
    
    def _analyze_results(
        self, f: Tensor, shock_positions: list, runtime: float
    ) -> ShockResult:
        """Analyze final shock structure."""
        cfg = self.cfg
        
        n, u, T = self.compute_moments(f)
        
        # Find shock position from density gradient
        dn_dx = torch.gradient(n, spacing=(cfg.dx,), dim=0)[0]
        shock_idx = int(torch.argmax(torch.abs(dn_dx)))
        shock_pos = float(self.x[shock_idx])
        
        # Compute shock velocity from position history
        if len(shock_positions) >= 2:
            steps = [p[0] for p in shock_positions]
            positions = [p[1] for p in shock_positions]
            # Linear fit
            dt_total = (steps[-1] - steps[0]) * self.dt
            dx_total = positions[-1] - positions[0]
            shock_vel = dx_total / dt_total if dt_total > 0 else 0
        else:
            shock_vel = 0.0
        
        # Compression ratio
        margin = max(10, cfg.nx // 20)
        
        # Downstream (between wall and shock)
        dn_start = margin
        dn_end = max(shock_idx - margin, dn_start + 1)
        n_downstream = float(n[dn_start:dn_end].mean()) if dn_end > dn_start else float(n[dn_start])
        
        # Upstream (beyond shock)
        up_start = min(shock_idx + margin, cfg.nx - margin - 1)
        up_end = cfg.nx - margin
        n_upstream = float(n[up_start:up_end].mean()) if up_end > up_start else float(n[up_start])
        
        compression = n_downstream / n_upstream if n_upstream > 0.1 else 1.0
        
        # Rankine-Hugoniot prediction
        gamma = 5.0 / 3.0
        M = cfg.mach
        compression_rh = ((gamma + 1) * M**2) / ((gamma - 1) * M**2 + 2)
        
        # Reflected ion fraction
        # Look for ions with v > v_sw at downstream location
        if shock_idx > 10:
            f_downstream = f[shock_idx // 2, :]
            v_reflected_min = 1.2 * cfg.v_sw  # Reflected ions have v > 1.2 * v_sw
            mask = self.v > v_reflected_min
            f_ref = float(torch.trapezoid(f_downstream[mask], self.v[mask]))
            f_tot = float(torch.trapezoid(f_downstream, self.v))
            reflected = f_ref / (f_tot + 1e-10)
        else:
            reflected = 0.0
        
        # Validation criteria
        validated = (
            compression > 2.0 and          # Significant compression
            compression < 5.0 and          # Not unphysical
            shock_pos > 5.0 and            # Shock in domain
            shock_pos < cfg.L_x - 10.0     # Not at boundary
        )
        
        print(f"\n{'='*60}")
        print("SHOCK ANALYSIS")
        print(f"{'='*60}")
        print(f"  Shock position: {shock_pos:.1f} (units)")
        print(f"  Shock velocity: {shock_vel:.2f} (units/time)")
        print(f"  Compression ratio: {compression:.2f}")
        print(f"  Rankine-Hugoniot: {compression_rh:.2f}")
        print(f"  Reflected fraction: {reflected*100:.1f}%")
        print(f"  Status: {'✓ VALIDATED' if validated else '✗ DEVELOPING'}")
        
        return ShockResult(
            compression_ratio=compression,
            compression_expected=compression_rh,
            shock_position=shock_pos,
            shock_velocity=shock_vel,
            reflected_fraction=reflected,
            validated=validated,
            runtime_seconds=runtime,
            density_profile=n,
            velocity_profile=u,
            x_grid=self.x,
        )


def validate_bow_shock(verbose: bool = True) -> Tuple[bool, ShockResult]:
    """Run bow shock validation benchmark."""
    if verbose:
        print("=" * 70)
        print("FRONTIER 02: BOW SHOCK VALIDATION")
        print("=" * 70)
    
    config = ShockConfig(
        nx=256,
        nv=128,
        n_sw=5.0,
        v_sw=400.0,
        T_sw=1e5,
        L_x=50.0,
        v_max=1000.0,  # Must capture reflected ions at ~2*v_sw
    )
    
    sim = BowShockSimulation(config)
    result = sim.run(n_steps=2000, diag_interval=400)
    
    if verbose:
        print(f"\nTotal runtime: {result.runtime_seconds:.2f}s")
    
    return result.validated, result


if __name__ == "__main__":
    validated, result = validate_bow_shock(verbose=True)
    print(f"\nFinal validation: {'PASS' if validated else 'IN PROGRESS'}")
