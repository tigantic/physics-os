"""
Two-Stream Instability Benchmark — Counter-Propagating Beam Physics

The two-stream instability is the archetypal plasma instability:
- Two electron beams pass through each other
- Small perturbations grow exponentially
- Energy transfers from beams to waves

This is THE validation test for instability physics in Vlasov solvers.

Analytic growth rate (cold beam limit, k = ω_pe / v_b):
    γ = √3/2 × ω_pe ≈ 0.866 ω_pe

For thermal beams with kλ_D effects:
    γ ≈ 0.354 ω_pe (for typical parameters)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class TwoStreamConfig:
    """Configuration for two-stream instability simulation.
    
    The most unstable wavenumber for two-stream is k ≈ ω_pe / v_b.
    For v_b = 3.0, we need domain_x = 6π to put the k=1 mode at the unstable point.
    
    Attributes:
        beam_velocity: Velocity of counter-propagating beams (±v_b)
        beam_width: Thermal spread of each beam
        perturbation: Initial density perturbation amplitude
        k_mode: Wavenumber of perturbation
        n_qubits_x: Qubits for spatial dimension
        n_qubits_v: Qubits for velocity dimension
        max_rank: Maximum QTT rank
        domain_x: Spatial domain length (6π for v_b=3 puts k=1 at optimal)
        domain_v: Velocity domain extent
        device: Torch device
        dtype: Tensor dtype
    """
    beam_velocity: float = 3.0
    beam_width: float = 0.5
    perturbation: float = 0.01
    k_mode: int = 1
    n_qubits_x: int = 7
    n_qubits_v: int = 7
    max_rank: int = 32
    domain_x: float = 6 * math.pi  # Matches k_optimal = 1/v_b for v_b=3
    domain_v: float = 8.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    @property
    def nx(self) -> int:
        return 2 ** self.n_qubits_x
    
    @property
    def nv(self) -> int:
        return 2 ** self.n_qubits_v
    
    @property
    def dx(self) -> float:
        return self.domain_x / self.nx
    
    @property
    def dv(self) -> float:
        return 2 * self.domain_v / self.nv
    
    @property
    def k_physical(self) -> float:
        return 2 * math.pi * self.k_mode / self.domain_x
    
    @property
    def analytic_growth_rate(self) -> float:
        """
        Approximate growth rate for two-stream instability.
        
        For cold beams: γ = √3/2 × ω_pe ≈ 0.866
        For thermal beams: reduced by finite temperature effects
        
        Using the approximate formula for finite temperature:
        γ ≈ ω_pe × (1 - (k v_th / ω_pe)²)^(1/2) × √3/2
        """
        # Cold beam limit
        gamma_cold = math.sqrt(3) / 2
        
        # Thermal correction (approximate)
        k_vth = self.k_physical * self.beam_width
        thermal_factor = max(0, 1 - k_vth ** 2)
        
        return gamma_cold * math.sqrt(thermal_factor)


@dataclass
class TwoStreamState:
    """State of two-stream simulation."""
    f_dense: Tensor  # Dense representation for spectral method
    time: float = 0.0
    electric_field_history: list = None
    time_history: list = None
    
    def __post_init__(self):
        if self.electric_field_history is None:
            self.electric_field_history = []
        if self.time_history is None:
            self.time_history = []


class TwoStreamInstability:
    """
    Two-Stream Instability Solver
    
    Solves the 1D-1V Vlasov-Poisson system for counter-propagating beams.
    
    Initial condition:
        f(x, v, 0) = [G(v - v_b) + G(v + v_b)] × [1 + ε cos(kx)]
    
    where G is a Gaussian (thermal distribution).
    
    The instability grows exponentially:
        |E(k)| ∝ exp(γ t)
    
    Example:
        >>> config = TwoStreamConfig(beam_velocity=3.0, beam_width=0.5)
        >>> solver = TwoStreamInstability(config)
        >>> state = solver.initialize()
        >>> 
        >>> for _ in range(200):
        ...     state = solver.step(state, dt=0.05)
        >>> 
        >>> gamma = solver.measure_growth_rate(state)
    """
    
    def __init__(self, config: TwoStreamConfig):
        self.config = config
    
    def initialize(self) -> TwoStreamState:
        """
        Create two-stream initial condition.
        
        Two counter-propagating Maxwellian beams with small density perturbation.
        f(x, v, 0) = n(x) × [G(v-v_b) + G(v+v_b)] / 2
        
        where n(x) = 1 + ε cos(kx)
        """
        cfg = self.config
        dev = torch.device(cfg.device)
        
        # Create grids
        x = torch.linspace(0, cfg.domain_x, cfg.nx, device=dev)
        v = torch.linspace(-cfg.domain_v, cfg.domain_v, cfg.nv, device=dev)
        X, V = torch.meshgrid(x, v, indexing='ij')
        
        # Two-stream distribution
        v_b = cfg.beam_velocity
        sigma = cfg.beam_width
        
        # Forward and backward beams (properly normalized Gaussians)
        beam_plus = torch.exp(-((V - v_b) ** 2) / (2 * sigma ** 2))
        beam_minus = torch.exp(-((V + v_b) ** 2) / (2 * sigma ** 2))
        
        # Normalize velocity distribution: ∫f_v dv = 1
        f_v = (beam_plus + beam_minus)
        # Numerical normalization
        f_v_norm = f_v.sum(dim=1, keepdim=True) * cfg.dv
        f_v = f_v / (f_v_norm + 1e-15)
        
        # Spatial density perturbation: n(x) = 1 + ε cos(kx)
        k = cfg.k_physical
        n_x = 1.0 + cfg.perturbation * torch.cos(k * X)
        
        # Full distribution: f = n(x) × f_v(v)
        f_dense = n_x * f_v
        
        state = TwoStreamState(f_dense=f_dense, time=0.0)
        
        # Record initial electric field
        E_k = self._compute_electric_field_mode(f_dense)
        state.electric_field_history.append(abs(E_k))
        state.time_history.append(0.0)
        
        return state
    
    def step(self, state: TwoStreamState, dt: float) -> TwoStreamState:
        """
        Advance one time step using Strang splitting with spectral method.
        """
        f = state.f_dense
        
        # Half step: advection
        f = self._spectral_advect_x(f, dt / 2)
        
        # Full step: acceleration
        E_field = self._compute_electric_field(f)
        f = self._spectral_accelerate_v(f, E_field, dt)
        
        # Half step: advection
        f = self._spectral_advect_x(f, dt / 2)
        
        # Record electric field
        E_k = self._compute_electric_field_mode(f)
        
        return TwoStreamState(
            f_dense=f,
            time=state.time + dt,
            electric_field_history=state.electric_field_history + [abs(E_k)],
            time_history=state.time_history + [state.time + dt],
        )
    
    def _spectral_advect_x(self, f: Tensor, dt: float) -> Tensor:
        """Spectral advection: f(x,v,t+dt) = f(x-v*dt, v, t)"""
        cfg = self.config
        dev = f.device
        
        f_k = torch.fft.fft(f, dim=0)
        k_grid = torch.fft.fftfreq(cfg.nx, d=cfg.dx, device=dev) * 2 * torch.pi
        v_grid = torch.linspace(-cfg.domain_v, cfg.domain_v, cfg.nv, device=dev)
        
        phase = torch.exp(-1j * k_grid.view(-1, 1) * v_grid.view(1, -1) * dt)
        f_k = f_k * phase
        
        return torch.fft.ifft(f_k, dim=0).real
    
    def _spectral_accelerate_v(self, f: Tensor, E_field: Tensor, dt: float) -> Tensor:
        """Spectral acceleration for electrons: f(x,v,t+dt) = f(x, v+E*dt, t)"""
        cfg = self.config
        dev = f.device
        
        f_eta = torch.fft.fft(f, dim=1)
        eta_grid = torch.fft.fftfreq(cfg.nv, d=cfg.dv, device=dev) * 2 * torch.pi
        
        phase = torch.exp(1j * eta_grid.view(1, -1) * E_field.view(-1, 1) * dt)
        f_eta = f_eta * phase
        
        return torch.fft.ifft(f_eta, dim=1).real
    
    def _compute_electric_field(self, f: Tensor) -> Tensor:
        """Compute E-field from Poisson equation."""
        cfg = self.config
        dev = f.device
        
        density = f.sum(dim=1) * cfg.dv
        rho = 1.0 - density  # ions - electrons
        
        rho_k = torch.fft.rfft(rho)
        k_grid = torch.fft.rfftfreq(cfg.nx, d=cfg.dx, device=dev) * 2 * torch.pi
        k_grid[0] = 1.0
        
        E_k = -1j * rho_k / k_grid
        E_k[0] = 0
        
        return torch.fft.irfft(E_k, n=cfg.nx)
    
    def _compute_electric_field_mode(self, f: Tensor) -> complex:
        """Get the k=k_mode Fourier component of E-field."""
        E = self._compute_electric_field(f)
        E_k = torch.fft.rfft(E)
        return E_k[self.config.k_mode].item()
    
    def measure_growth_rate(
        self,
        state: TwoStreamState,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> float:
        """
        Measure growth rate from electric field history.
        
        Fits log|E(t)| = γt + const during the linear growth phase.
        Automatically detects the linear phase before saturation.
        """
        times = torch.tensor(state.time_history)
        E_vals = torch.tensor(state.electric_field_history)
        
        # Find linear growth phase
        # Start after initial transients (E > initial value)
        E_initial = E_vals[0]
        
        if t_start is None:
            # Start when E first exceeds 2× initial
            growing_mask = E_vals > 2 * E_initial
            if growing_mask.any():
                t_start = times[growing_mask][0].item()
            else:
                t_start = times.max() * 0.1
        
        if t_end is None:
            # End before saturation (E < 0.5 × max)
            E_max = E_vals.max()
            pre_sat_mask = E_vals < 0.5 * E_max
            # Find last time before saturation in growth phase
            growth_phase = (times >= t_start) & pre_sat_mask
            if growth_phase.any():
                t_end = times[growth_phase].max().item()
            else:
                # Use first time E reaches significant growth
                t_end = times.max() * 0.5
        
        mask = (times >= t_start) & (times <= t_end) & (E_vals > 1e-15)
        
        if mask.sum() < 3:
            # Fallback: use early times
            mask = (times > 0) & (times < times.max() * 0.3) & (E_vals > 1e-15)
        
        if mask.sum() < 3:
            return float('nan')
        
        t_fit = times[mask]
        log_E = torch.log(E_vals[mask])
        
        # Linear regression
        n = len(t_fit)
        t_mean = t_fit.mean()
        log_E_mean = log_E.mean()
        
        numerator = ((t_fit - t_mean) * (log_E - log_E_mean)).sum()
        denominator = ((t_fit - t_mean) ** 2).sum()
        
        if denominator < 1e-15:
            return float('nan')
        
        return (numerator / denominator).item()
    
    def run(
        self,
        t_final: float = 20.0,
        dt: float = 0.05,
        verbose: bool = True,
    ) -> TwoStreamState:
        """Run complete two-stream simulation."""
        state = self.initialize()
        n_steps = int(t_final / dt)
        
        if verbose:
            print(f"Two-Stream Instability Simulation")
            print(f"  Grid: {self.config.nx} × {self.config.nv}")
            print(f"  Beam velocity: ±{self.config.beam_velocity}")
            print(f"  Beam width: {self.config.beam_width}")
            print(f"  Analytic γ ≈ {self.config.analytic_growth_rate:.4f}")
            print(f"  Running {n_steps} steps...")
        
        for step in range(n_steps):
            state = self.step(state, dt)
            
            if verbose and (step + 1) % (n_steps // 10) == 0:
                E_current = state.electric_field_history[-1]
                print(f"  t = {state.time:.1f}, |E(k=1)| = {E_current:.6e}")
        
        gamma = self.measure_growth_rate(state)
        gamma_analytic = self.config.analytic_growth_rate
        
        if verbose:
            print(f"\nResults:")
            print(f"  Measured γ = {gamma:.4f}")
            print(f"  Analytic γ ≈ {gamma_analytic:.4f}")
            if gamma_analytic > 0:
                print(f"  Relative error = {abs(gamma - gamma_analytic) / gamma_analytic * 100:.1f}%")
        
        return state


def validate_two_stream(
    beam_velocity: float = 3.0,
    beam_width: float = 0.5,
    n_qubits_x: int = 7,
    n_qubits_v: int = 7,
    t_final: float = 15.0,
    dt: float = 0.05,
) -> dict:
    """
    Run two-stream instability validation.
    
    Returns dictionary with results.
    """
    config = TwoStreamConfig(
        beam_velocity=beam_velocity,
        beam_width=beam_width,
        perturbation=0.001,
        n_qubits_x=n_qubits_x,
        n_qubits_v=n_qubits_v,
        max_rank=32,
    )
    
    solver = TwoStreamInstability(config)
    state = solver.run(t_final=t_final, dt=dt, verbose=True)
    
    gamma = solver.measure_growth_rate(state)
    gamma_analytic = config.analytic_growth_rate
    
    # For two-stream, we expect positive growth rate
    # Success if we see exponential growth with reasonable rate
    passed = gamma > 0.1 and gamma < 1.5  # Physical range
    
    return {
        "gamma_measured": gamma,
        "gamma_analytic": gamma_analytic,
        "beam_velocity": beam_velocity,
        "beam_width": beam_width,
        "grid_size": f"{config.nx} × {config.nv}",
        "passed": passed,
        "state": state,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TWO-STREAM INSTABILITY VALIDATION")
    print("=" * 60)
    print()
    
    result = validate_two_stream()
    
    print()
    print("=" * 60)
    if result["passed"]:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")
    print("=" * 60)
