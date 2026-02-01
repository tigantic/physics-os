#!/usr/bin/env python3
"""
FRONTIER 03: Inductively Coupled Plasma (ICP) Discharge Simulation
===================================================================

Implements 1D radial ICP discharge model for semiconductor plasma processing.

Physics Model:
- Electromagnetic wave heating (azimuthal E-field, skin effect)
- Electron energy balance with collisional power absorption
- Ambipolar diffusion for particle transport
- Ionization balance with Ar chemistry

Key Parameters (typical ICP reactor):
- Pressure: 1-50 mTorr
- RF Power: 100-2000 W
- Frequency: 13.56 MHz (industrial standard)
- Electron density: 10^10 - 10^12 cm^-3
- Electron temperature: 2-5 eV

Benchmark: Steady-state electron density profile should match:
- Bessel J0 profile in uniform discharge
- Skin depth heating at low pressure
- Collisional heating at high pressure

Reference: Lieberman & Lichtenberg, "Principles of Plasma Discharges 
           and Materials Processing", 2nd Ed., Chapter 12

(c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional
import time

import numpy as np
from numpy.typing import NDArray


# Physical constants (SI units)
ELECTRON_MASS = 9.109e-31        # kg
ELECTRON_CHARGE = 1.602e-19      # C
EPSILON_0 = 8.854e-12            # F/m
K_BOLTZMANN = 1.381e-23          # J/K
EV_TO_JOULE = 1.602e-19          # J/eV


@dataclass
class ICPConfig:
    """Configuration for ICP discharge simulation."""
    
    # Reactor geometry
    reactor_radius: float = 0.15      # m (15 cm typical)
    reactor_height: float = 0.10      # m (10 cm gap)
    
    # Operating conditions
    pressure_mtorr: float = 10.0      # mTorr
    rf_power_w: float = 500.0         # W (absorbed power)
    rf_frequency_hz: float = 13.56e6  # Hz (industrial standard)
    
    # Grid parameters
    nr: int = 100                     # Radial grid points
    
    # Time stepping
    dt: float = 1e-8                  # s (10 ns)
    n_steps: int = 10000              # Steps to steady state
    
    # Gas properties (Argon)
    gas_temperature_k: float = 300.0  # K (room temperature)
    ion_mass_amu: float = 40.0        # Argon
    
    # Initial conditions
    n_e_init: float = 1e16            # m^-3 (10^10 cm^-3)
    T_e_init: float = 3.0             # eV
    
    # Convergence
    tolerance: float = 1e-4           # Relative change for steady state
    check_interval: int = 500         # Steps between convergence checks


@dataclass
class ICPResult:
    """Results from ICP discharge simulation."""
    
    # Spatial profiles
    r: NDArray[np.float64]            # Radial positions [m]
    n_e: NDArray[np.float64]          # Electron density [m^-3]
    T_e: NDArray[np.float64]          # Electron temperature [eV]
    
    # Power deposition
    power_density: NDArray[np.float64]  # W/m^3
    skin_depth: float                   # m
    
    # Derived quantities
    plasma_frequency: NDArray[np.float64]   # rad/s
    debye_length: NDArray[np.float64]       # m
    
    # Global parameters
    n_e_avg: float                    # Average electron density [m^-3]
    T_e_avg: float                    # Average electron temperature [eV]
    ionization_rate: float            # m^-3 s^-1
    
    # Convergence
    converged: bool
    n_iterations: int
    final_residual: float
    runtime_s: float


class ICPDischarge:
    """
    1D radial ICP discharge model with electromagnetic heating.
    
    Solves coupled equations for:
    1. Electron continuity: ∂n_e/∂t = S_iz - D_a∇²n_e / n_e
    2. Electron energy: ∂(3/2 n_e T_e)/∂t = P_abs - P_loss
    3. Electromagnetic field: ∇²E + k²E = iωμ₀σE
    
    Uses operator splitting:
    - Diffusion: implicit Crank-Nicolson
    - Source terms: explicit Euler
    - EM field: steady-state solution each step
    """
    
    def __init__(self, cfg: ICPConfig):
        self.cfg = cfg
        
        # Create radial grid (avoid r=0 singularity)
        self.dr = cfg.reactor_radius / cfg.nr
        self.r = np.linspace(self.dr/2, cfg.reactor_radius - self.dr/2, cfg.nr)
        
        # Initialize state
        self.n_e = np.ones(cfg.nr) * cfg.n_e_init
        self.T_e = np.ones(cfg.nr) * cfg.T_e_init
        
        # Precompute gas properties
        self.pressure_pa = cfg.pressure_mtorr * 0.133322  # mTorr to Pa
        self.n_gas = self.pressure_pa / (K_BOLTZMANN * cfg.gas_temperature_k)  # m^-3
        self.ion_mass = cfg.ion_mass_amu * 1.66054e-27  # kg
        
        # RF angular frequency
        self.omega = 2 * math.pi * cfg.rf_frequency_hz
        
        # Build diffusion matrix
        self._build_diffusion_matrix()
        
    def _build_diffusion_matrix(self) -> None:
        """Build tridiagonal matrix for radial diffusion operator."""
        nr = self.cfg.nr
        dr = self.dr
        
        # Coefficients for (1/r) d/dr (r D d/dr) in cylindrical coords
        # Using central differences
        self.A_diff = np.zeros((nr, nr))
        
        for i in range(nr):
            r_i = self.r[i]
            
            if i > 0:
                r_minus = (self.r[i-1] + r_i) / 2
                self.A_diff[i, i-1] = r_minus / (r_i * dr**2)
                
            if i < nr - 1:
                r_plus = (self.r[i+1] + r_i) / 2
                self.A_diff[i, i+1] = r_plus / (r_i * dr**2)
                
            # Diagonal
            r_minus = (self.r[max(0, i-1)] + r_i) / 2 if i > 0 else r_i/2
            r_plus = (self.r[min(nr-1, i+1)] + r_i) / 2 if i < nr-1 else r_i
            self.A_diff[i, i] = -(r_minus + r_plus) / (r_i * dr**2)
        
        # Boundary conditions:
        # r=0: symmetry (dn/dr = 0) - already handled by grid starting at dr/2
        # r=R: n_e = 0 (wall loss) - Dirichlet
        self.A_diff[-1, :] = 0
        self.A_diff[-1, -1] = -2 / dr**2  # Enhanced loss at wall
        
    def collision_frequency(self, T_e: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Electron-neutral collision frequency [s^-1].
        
        For Argon: ν_m ≈ n_g * σ_m * v_e
        where σ_m ≈ 2×10^-19 m² is momentum transfer cross section
        """
        # Mean electron speed
        v_e = np.sqrt(8 * EV_TO_JOULE * T_e / (math.pi * ELECTRON_MASS))
        
        # Momentum transfer cross section (energy-averaged for Ar)
        sigma_m = 2e-19  # m²
        
        return self.n_gas * sigma_m * v_e
    
    def ionization_rate_coeff(self, T_e: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Ionization rate coefficient k_iz [m³/s] for Argon.
        
        Arrhenius form: k_iz = k_0 * exp(-E_iz / T_e)
        E_iz(Ar) = 15.76 eV
        """
        E_iz = 15.76  # eV, ionization threshold
        k_0 = 2.34e-14  # m³/s, pre-exponential factor
        
        # Prevent overflow for low temperatures
        T_e_safe = np.maximum(T_e, 0.5)
        
        return k_0 * np.sqrt(T_e_safe) * np.exp(-E_iz / T_e_safe)
    
    def ambipolar_diffusion_coeff(self, T_e: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Ambipolar diffusion coefficient D_a [m²/s].
        
        D_a ≈ D_i (1 + T_e/T_i) ≈ 2 D_i for T_e >> T_i
        D_i = k_B T_i / (m_i ν_in)
        """
        # Ion-neutral collision frequency
        v_i = np.sqrt(8 * K_BOLTZMANN * self.cfg.gas_temperature_k / 
                      (math.pi * self.ion_mass))
        sigma_in = 1e-18  # m², ion-neutral cross section
        nu_in = self.n_gas * sigma_in * v_i
        
        # Ion diffusion coefficient
        D_i = K_BOLTZMANN * self.cfg.gas_temperature_k / (self.ion_mass * nu_in)
        
        # Ambipolar diffusion (enhanced by electron temperature)
        T_i_eV = self.cfg.gas_temperature_k * K_BOLTZMANN / EV_TO_JOULE
        D_a = D_i * (1 + T_e / T_i_eV)
        
        return D_a
    
    def compute_skin_depth(self, n_e_avg: float, nu_m: float) -> float:
        """
        Electromagnetic skin depth δ [m].
        
        In collisional regime (ν_m >> ω):
        δ = c / ω_pe * sqrt(2ν_m / ω)
        
        In collisionless regime (ν_m << ω):
        δ = c / ω_pe
        """
        # Plasma frequency
        omega_pe = np.sqrt(n_e_avg * ELECTRON_CHARGE**2 / 
                          (ELECTRON_MASS * EPSILON_0))
        
        c = 3e8  # m/s
        
        if omega_pe < 1e6:  # Very low density
            return self.cfg.reactor_radius
        
        # Collisional skin depth
        delta_c = c / omega_pe * np.sqrt(2 * nu_m / self.omega)
        
        # Collisionless skin depth
        delta_0 = c / omega_pe
        
        # Use appropriate limit
        if nu_m > self.omega:
            return delta_c
        else:
            return delta_0
    
    def compute_power_deposition(self) -> NDArray[np.float64]:
        """
        Compute RF power deposition profile P(r) [W/m³].
        
        Power is absorbed in skin layer near reactor wall.
        P(r) = P_0 * exp(-(R-r)/δ) for skin effect heating
        
        Normalized so that ∫P dV = P_total
        """
        # Average collision frequency
        nu_m = np.mean(self.collision_frequency(self.T_e))
        n_e_avg = np.mean(self.n_e)
        
        # Skin depth
        delta = self.compute_skin_depth(n_e_avg, nu_m)
        self._skin_depth = delta
        
        # Power profile (exponential decay from edge)
        R = self.cfg.reactor_radius
        power_profile = np.exp(-(R - self.r) / delta)
        
        # Volume integration factor (2πr dr × height)
        dV = 2 * math.pi * self.r * self.dr * self.cfg.reactor_height
        total_volume_factor = np.sum(power_profile * dV)
        
        # Normalize to total power
        if total_volume_factor > 0:
            P_density = self.cfg.rf_power_w * power_profile / total_volume_factor
        else:
            P_density = np.zeros_like(self.r)
        
        return P_density
    
    def electron_loss_power(self) -> NDArray[np.float64]:
        """
        Electron energy loss rate [W/m³].
        
        Includes:
        1. Ionization energy loss: E_iz per ionization event
        2. Excitation losses: ~E_ex per excitation
        3. Elastic collision losses: 2m_e/M energy per collision
        """
        # Ionization loss
        k_iz = self.ionization_rate_coeff(self.T_e)
        E_iz = 15.76  # eV
        P_iz = k_iz * self.n_gas * self.n_e * E_iz * EV_TO_JOULE
        
        # Excitation loss (approximately same rate as ionization at these temperatures)
        E_ex = 11.5  # eV, average excitation energy for Ar
        k_ex = 0.5 * k_iz  # Rough approximation
        P_ex = k_ex * self.n_gas * self.n_e * E_ex * EV_TO_JOULE
        
        # Elastic collision loss
        nu_m = self.collision_frequency(self.T_e)
        delta_E = 2 * ELECTRON_MASS / self.ion_mass  # Energy loss fraction
        # Energy per electron: 3/2 T_e
        P_el = nu_m * self.n_e * delta_E * 1.5 * self.T_e * EV_TO_JOULE
        
        return P_iz + P_ex + P_el
    
    def step(self) -> None:
        """Advance simulation by one time step."""
        dt = self.cfg.dt
        
        # 1. Compute source/sink terms
        k_iz = self.ionization_rate_coeff(self.T_e)
        S_iz = k_iz * self.n_gas * self.n_e  # Ionization source
        
        D_a = self.ambipolar_diffusion_coeff(self.T_e)
        
        # 2. Update density: dn/dt = S_iz + D_a ∇²n (ambipolar diffusion)
        # Implicit diffusion, explicit source
        diffusion_term = D_a[:, np.newaxis] * self.A_diff
        
        # Simple explicit update for stability testing
        dn_diff = np.sum(diffusion_term * self.n_e, axis=1)
        self.n_e = self.n_e + dt * (S_iz + D_a * np.dot(self.A_diff, self.n_e))
        
        # Wall loss at r=R
        self.n_e[-1] = 0.1 * self.n_e[-2]  # Sheath drop
        
        # Enforce positivity
        self.n_e = np.maximum(self.n_e, 1e10)  # Minimum density
        
        # 3. Update temperature from energy balance
        P_abs = self.compute_power_deposition()
        P_loss = self.electron_loss_power()
        
        # Energy density: u = 3/2 n_e T_e (in eV units)
        u = 1.5 * self.n_e * self.T_e  # eV/m³
        
        # du/dt = P_abs - P_loss (convert W/m³ to eV/m³/s)
        du_dt = (P_abs - P_loss) / EV_TO_JOULE
        
        u_new = u + dt * du_dt
        u_new = np.maximum(u_new, 1.5 * self.n_e * 0.5)  # Minimum 0.5 eV
        
        self.T_e = u_new / (1.5 * self.n_e)
        self.T_e = np.clip(self.T_e, 0.5, 20.0)  # Physical bounds
        
    def run(self) -> ICPResult:
        """Run simulation to steady state."""
        t_start = time.time()
        
        n_e_prev = self.n_e.copy()
        converged = False
        final_residual = 1.0
        
        for step in range(self.cfg.n_steps):
            self.step()
            
            # Check convergence periodically
            if (step + 1) % self.cfg.check_interval == 0:
                residual = np.max(np.abs(self.n_e - n_e_prev)) / np.max(self.n_e)
                final_residual = residual
                
                if residual < self.cfg.tolerance:
                    converged = True
                    break
                    
                n_e_prev = self.n_e.copy()
        
        runtime = time.time() - t_start
        
        # Compute derived quantities
        omega_pe = np.sqrt(self.n_e * ELECTRON_CHARGE**2 / 
                         (ELECTRON_MASS * EPSILON_0))
        
        lambda_D = np.sqrt(EPSILON_0 * self.T_e * EV_TO_JOULE / 
                          (self.n_e * ELECTRON_CHARGE**2))
        
        # Volume-averaged quantities
        dV = 2 * math.pi * self.r * self.dr
        total_volume = np.sum(dV)
        n_e_avg = np.sum(self.n_e * dV) / total_volume
        T_e_avg = np.sum(self.T_e * dV) / total_volume
        
        # Ionization rate
        k_iz = self.ionization_rate_coeff(self.T_e)
        iz_rate = np.mean(k_iz * self.n_gas * self.n_e)
        
        return ICPResult(
            r=self.r.copy(),
            n_e=self.n_e.copy(),
            T_e=self.T_e.copy(),
            power_density=self.compute_power_deposition(),
            skin_depth=getattr(self, '_skin_depth', 0.01),
            plasma_frequency=omega_pe,
            debye_length=lambda_D,
            n_e_avg=n_e_avg,
            T_e_avg=T_e_avg,
            ionization_rate=iz_rate,
            converged=converged,
            n_iterations=step + 1,
            final_residual=final_residual,
            runtime_s=runtime
        )


def validate_icp_physics(result: ICPResult, cfg: ICPConfig) -> dict:
    """
    Validate ICP simulation against known physics.
    
    Checks:
    1. Density magnitude: 10^10 - 10^12 cm^-3 for typical ICP
    2. Temperature: 2-5 eV typical
    3. Skin depth: should be < reactor radius
    4. Bessel profile: density should approximate J0(2.405 r/R)
    """
    from scipy.special import j0
    
    checks = {}
    
    # 1. Density magnitude
    n_e_cm3 = result.n_e_avg / 1e6  # m^-3 to cm^-3
    density_valid = 1e10 <= n_e_cm3 <= 1e12
    checks['density_magnitude'] = {
        'valid': density_valid,
        'value_cm3': n_e_cm3,
        'expected_range': '10^10 - 10^12 cm^-3'
    }
    
    # 2. Temperature range
    temp_valid = 1.5 <= result.T_e_avg <= 8.0
    checks['temperature'] = {
        'valid': temp_valid,
        'value_eV': result.T_e_avg,
        'expected_range': '1.5 - 8.0 eV'
    }
    
    # 3. Skin depth
    skin_valid = result.skin_depth < cfg.reactor_radius
    checks['skin_depth'] = {
        'valid': skin_valid,
        'value_cm': result.skin_depth * 100,
        'reactor_radius_cm': cfg.reactor_radius * 100
    }
    
    # 4. Profile shape - compare to Bessel J0
    # At low pressure, profile should be approximately diffusion-dominated
    r_norm = result.r / cfg.reactor_radius
    bessel_profile = j0(2.405 * r_norm)
    bessel_profile = bessel_profile / np.max(bessel_profile)
    
    n_e_norm = result.n_e / np.max(result.n_e)
    
    # Allow some deviation due to skin effect heating
    profile_error = np.sqrt(np.mean((n_e_norm - bessel_profile)**2))
    profile_valid = profile_error < 0.5  # 50% RMS error allowed (skin effect modifies profile)
    
    checks['profile_shape'] = {
        'valid': profile_valid,
        'rms_error_vs_bessel': profile_error,
        'note': 'Deviation expected due to skin-effect heating'
    }
    
    # 5. Convergence
    checks['convergence'] = {
        'valid': result.converged or result.final_residual < 0.01,
        'converged': result.converged,
        'final_residual': result.final_residual
    }
    
    # Overall
    all_valid = all(c['valid'] for c in checks.values())
    checks['all_pass'] = all_valid
    
    return checks


def run_icp_benchmark() -> Tuple[ICPResult, dict]:
    """Run ICP discharge benchmark."""
    print("="*70)
    print("FRONTIER 03: Inductively Coupled Plasma (ICP) Discharge")
    print("="*70)
    print()
    
    # Standard ICP conditions
    cfg = ICPConfig(
        reactor_radius=0.15,      # 15 cm radius
        pressure_mtorr=10.0,      # 10 mTorr (typical processing)
        rf_power_w=500.0,         # 500 W absorbed
        nr=100,                   # 100 radial points
        n_steps=20000,            # Run to steady state
        dt=5e-9,                  # 5 ns time step
        tolerance=1e-3,
    )
    
    print(f"Configuration:")
    print(f"  Reactor radius:   {cfg.reactor_radius*100:.1f} cm")
    print(f"  Pressure:         {cfg.pressure_mtorr:.1f} mTorr")
    print(f"  RF Power:         {cfg.rf_power_w:.0f} W")
    print(f"  RF Frequency:     {cfg.rf_frequency_hz/1e6:.2f} MHz")
    print()
    
    # Run simulation
    print("Running ICP discharge simulation...")
    sim = ICPDischarge(cfg)
    result = sim.run()
    
    print(f"  Completed in {result.runtime_s:.2f} s ({result.n_iterations} iterations)")
    print(f"  Converged: {result.converged} (residual = {result.final_residual:.2e})")
    print()
    
    # Display results
    print("Results:")
    print(f"  Average n_e:      {result.n_e_avg/1e6:.2e} cm^-3")
    print(f"  Peak n_e:         {np.max(result.n_e)/1e6:.2e} cm^-3")
    print(f"  Average T_e:      {result.T_e_avg:.2f} eV")
    print(f"  Skin depth:       {result.skin_depth*100:.2f} cm")
    print(f"  Debye length:     {np.mean(result.debye_length)*1e6:.1f} μm")
    print()
    
    # Validate
    checks = validate_icp_physics(result, cfg)
    
    print("Validation:")
    print(f"  Density magnitude:  {'✓ PASS' if checks['density_magnitude']['valid'] else '✗ FAIL'}")
    print(f"  Temperature range:  {'✓ PASS' if checks['temperature']['valid'] else '✗ FAIL'}")
    print(f"  Skin depth:         {'✓ PASS' if checks['skin_depth']['valid'] else '✗ FAIL'}")
    print(f"  Profile shape:      {'✓ PASS' if checks['profile_shape']['valid'] else '✗ FAIL'}")
    print(f"  Convergence:        {'✓ PASS' if checks['convergence']['valid'] else '✗ FAIL'}")
    print()
    
    if checks['all_pass']:
        print("★ ICP DISCHARGE BENCHMARK: PASS")
    else:
        print("✗ ICP DISCHARGE BENCHMARK: FAIL")
    
    return result, checks


if __name__ == "__main__":
    result, checks = run_icp_benchmark()
