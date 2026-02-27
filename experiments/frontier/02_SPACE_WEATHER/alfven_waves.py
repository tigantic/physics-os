"""
Alfvén Wave Benchmark — MHD Wave Propagation in Space Plasmas

This module validates the fundamental wave physics essential for space
weather prediction: Alfvén waves in the solar wind.

Physics:
- Alfvén wave: incompressible transverse oscillation of magnetic field
- Dispersion relation: ω = k · v_A where v_A = B/√(μ₀ρ)
- No damping in ideal MHD limit
- Carries energy and momentum in solar wind

This is a simpler benchmark than bow shock formation - validates
that we can correctly propagate MHD waves at the Alfvén velocity.

Validation:
- Wave propagates at v_A
- No dispersion (phase velocity = group velocity)
- No damping (energy conserved)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
import time as time_module
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class AlfvenConfig:
    """Configuration for Alfvén wave simulation."""
    nx: int = 256           # Spatial grid points
    L: float = 1.0          # Domain length (normalized)
    
    # Plasma parameters (normalized)
    rho0: float = 1.0       # Background density
    B0: float = 1.0         # Background magnetic field (z-direction)
    
    # Wave parameters
    k_mode: int = 2         # Wave number (units of 2π/L)
    amplitude: float = 0.1  # Perturbation amplitude
    
    # Simulation
    n_periods: float = 2.0  # Number of wave periods to simulate
    cfl: float = 0.5
    device: str = "cpu"
    
    @property
    def dx(self) -> float:
        return self.L / self.nx
    
    @property
    def v_A(self) -> float:
        """Alfvén velocity (normalized units: B/√ρ)."""
        return self.B0 / math.sqrt(self.rho0)
    
    @property
    def k(self) -> float:
        """Physical wavenumber."""
        return 2 * math.pi * self.k_mode / self.L
    
    @property
    def omega(self) -> float:
        """Angular frequency: ω = k·v_A."""
        return self.k * self.v_A
    
    @property
    def period(self) -> float:
        """Wave period: T = 2π/ω."""
        return 2 * math.pi / self.omega
    
    @property
    def t_final(self) -> float:
        """Total simulation time."""
        return self.n_periods * self.period


@dataclass
class AlfvenResult:
    """Results from Alfvén wave simulation."""
    phase_velocity_measured: float
    phase_velocity_expected: float
    phase_error: float
    energy_initial: float
    energy_final: float
    energy_error: float
    validated: bool
    runtime_seconds: float


class AlfvenWaveSimulation:
    """
    1D Alfvén wave propagation using linearized MHD.
    
    Equations (linearized, incompressible):
        ∂B_y/∂t = B_0 ∂v_y/∂x
        ∂v_y/∂t = (B_0/ρ_0) ∂B_y/∂x
    
    Combined: ∂²B_y/∂t² = v_A² ∂²B_y/∂x²  (wave equation)
    
    Solution: B_y = A sin(kx - ωt) with ω = k·v_A
    """
    
    def __init__(self, config: AlfvenConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        
        # Grid
        self.x = torch.linspace(0, config.L, config.nx, device=self.device)
        
        # Time step
        self.dt = config.cfl * config.dx / config.v_A
        
        # Wavenumbers for spectral derivative
        self.kx = torch.fft.fftfreq(config.nx, d=config.dx, device=self.device) * 2 * math.pi
        
        print(f"AlfvenWaveSimulation initialized:")
        print(f"  Grid: {config.nx} points")
        print(f"  v_A = {config.v_A:.4f}")
        print(f"  k = {config.k:.4f}, ω = {config.omega:.4f}")
        print(f"  Period = {config.period:.4f}")
    
    def initialize(self) -> Tuple[Tensor, Tensor]:
        """Initialize Alfvén wave perturbation.
        
        Returns:
            Tuple of (B_y, v_y) at t=0
        """
        cfg = self.cfg
        
        # B_y = A sin(kx)
        B_y = cfg.amplitude * torch.sin(cfg.k * self.x)
        
        # For a right-propagating wave: v_y = -B_y/√(μ₀ρ) = -B_y/v_A * sign
        # Actually, from the dispersion relation:
        # v_y = -(ω/k) B_y / B_0 = -v_A B_y / B_0
        v_y = -cfg.v_A * B_y / cfg.B0
        
        return B_y, v_y
    
    def spectral_derivative(self, f: Tensor) -> Tensor:
        """Compute ∂f/∂x using spectral method."""
        f_k = torch.fft.fft(f)
        df_k = 1j * self.kx * f_k
        return torch.fft.ifft(df_k).real
    
    def compute_energy(self, B_y: Tensor, v_y: Tensor) -> float:
        """Compute total wave energy.
        
        E = ∫ (ρ v_y² / 2 + B_y² / (2μ₀)) dx
        
        In normalized units (μ₀ = 1):
        E = ∫ (ρ₀ v_y² / 2 + B_y² / 2) dx
        """
        cfg = self.cfg
        kinetic = 0.5 * cfg.rho0 * v_y**2
        magnetic = 0.5 * B_y**2
        return float(torch.trapezoid(kinetic + magnetic, self.x))
    
    def step(self, B_y: Tensor, v_y: Tensor) -> Tuple[Tensor, Tensor]:
        """Advance one time step using leapfrog method.
        
        ∂B_y/∂t = B_0 ∂v_y/∂x
        ∂v_y/∂t = (B_0/ρ_0) ∂B_y/∂x
        """
        cfg = self.cfg
        dt = self.dt
        
        # Half step B_y
        dv_dx = self.spectral_derivative(v_y)
        B_y_half = B_y + 0.5 * dt * cfg.B0 * dv_dx
        
        # Full step v_y using B_y_half
        dB_dx = self.spectral_derivative(B_y_half)
        v_y_new = v_y + dt * (cfg.B0 / cfg.rho0) * dB_dx
        
        # Complete B_y step
        dv_dx_new = self.spectral_derivative(v_y_new)
        B_y_new = B_y_half + 0.5 * dt * cfg.B0 * dv_dx_new
        
        return B_y_new, v_y_new
    
    def measure_phase_velocity(self, B_y: Tensor, t: float) -> float:
        """Measure phase velocity by tracking wave peak.
        
        For sin(kx - ωt), the peak at t=0 is at x = L/4k
        At time t, peak is at x = L/4k + v_A*t
        """
        cfg = self.cfg
        
        # Find peak position
        peak_idx = int(torch.argmax(B_y))
        x_peak = float(self.x[peak_idx])
        
        # Expected position at t=0 was π/(2k)
        x_initial = math.pi / (2 * cfg.k)
        
        # Phase velocity = (x_peak - x_initial) / t
        # Handle periodic domain
        dx = x_peak - x_initial
        if dx < -cfg.L/2:
            dx += cfg.L
        elif dx > cfg.L/2:
            dx -= cfg.L
        
        return dx / t if t > 0 else 0.0
    
    def run(self, diag_interval: int = 100) -> AlfvenResult:
        """Run simulation and measure wave properties."""
        cfg = self.cfg
        
        print(f"\nSimulating {cfg.n_periods} wave periods...")
        start = time_module.time()
        
        B_y, v_y = self.initialize()
        E_initial = self.compute_energy(B_y, v_y)
        
        n_steps = int(cfg.t_final / self.dt)
        t = 0.0
        
        phase_velocities = []
        
        for step in range(n_steps):
            B_y, v_y = self.step(B_y, v_y)
            t += self.dt
            
            if (step + 1) % diag_interval == 0:
                E_current = self.compute_energy(B_y, v_y)
                v_phase = self.measure_phase_velocity(B_y, t)
                phase_velocities.append(v_phase)
                
                print(f"  t = {t:.4f}: E/E₀ = {E_current/E_initial:.6f}, v_ph = {v_phase:.4f}")
        
        runtime = time_module.time() - start
        E_final = self.compute_energy(B_y, v_y)
        
        # Average phase velocity (exclude early measurements)
        if len(phase_velocities) > 2:
            v_phase_measured = sum(phase_velocities[2:]) / len(phase_velocities[2:])
        else:
            v_phase_measured = phase_velocities[-1] if phase_velocities else 0
        
        phase_error = abs(v_phase_measured - cfg.v_A) / cfg.v_A
        energy_error = abs(E_final - E_initial) / E_initial
        
        validated = phase_error < 0.05 and energy_error < 0.01
        
        print(f"\n{'='*60}")
        print("ALFVÉN WAVE ANALYSIS")
        print(f"{'='*60}")
        print(f"  Phase velocity measured: {v_phase_measured:.4f}")
        print(f"  Phase velocity expected: {cfg.v_A:.4f}")
        print(f"  Phase error: {phase_error*100:.2f}%")
        print(f"  Energy conservation: {(1-energy_error)*100:.4f}%")
        print(f"  Status: {'✓ VALIDATED' if validated else '✗ NEEDS WORK'}")
        
        return AlfvenResult(
            phase_velocity_measured=v_phase_measured,
            phase_velocity_expected=cfg.v_A,
            phase_error=phase_error,
            energy_initial=E_initial,
            energy_final=E_final,
            energy_error=energy_error,
            validated=validated,
            runtime_seconds=runtime,
        )


def validate_alfven_waves(verbose: bool = True) -> Tuple[bool, AlfvenResult]:
    """Run Alfvén wave validation benchmark."""
    if verbose:
        print("=" * 70)
        print("FRONTIER 02: ALFVÉN WAVE VALIDATION")
        print("=" * 70)
    
    config = AlfvenConfig(
        nx=256,
        L=1.0,
        rho0=1.0,
        B0=1.0,
        k_mode=4,
        amplitude=0.1,
        n_periods=3.0,
    )
    
    sim = AlfvenWaveSimulation(config)
    result = sim.run(diag_interval=200)
    
    if verbose:
        print(f"\nRuntime: {result.runtime_seconds:.2f}s")
    
    return result.validated, result


if __name__ == "__main__":
    validated, result = validate_alfven_waves(verbose=True)
    print(f"\nFinal validation: {'PASS' if validated else 'FAIL'}")
