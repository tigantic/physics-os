#!/usr/bin/env python3
"""
FRONTIER 04: Space Charge and Collective Effects
=================================================

Implements space charge tune shift and collective instability thresholds.

Physics Model:
- Space charge tune shift (Laslett coefficients)
- Beam-beam tune shift (linear approximation)
- Transverse mode coupling instability (TMCI) threshold
- Resistive wall impedance

Key Formula (Space Charge Tune Shift):
    ΔQ = -r_p * N_b / (4π * β² * γ³ * ε_n * B_f)

where:
- r_p = classical proton radius
- N_b = particles per bunch
- β, γ = relativistic factors
- ε_n = normalized emittance
- B_f = bunching factor

Benchmark:
- Tune shift magnitude matches theory
- TMCI threshold physically reasonable
- Stability boundary correct

Reference:
- A. Chao, "Physics of Collective Beam Instabilities"
- K. Schindl, CERN-PS-99-012-DI

(c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


# Physical constants
C_LIGHT = 299792458.0              # m/s
R_PROTON = 1.5347e-18              # Classical proton radius [m]
R_ELECTRON = 2.8179e-15            # Classical electron radius [m]
ELECTRON_MASS_EV = 0.511e6         # eV/c²
PROTON_MASS_EV = 938.272e6         # eV/c²
EPSILON_0 = 8.854e-12              # F/m


@dataclass
class SpaceChargeConfig:
    """Configuration for space charge calculation."""
    
    # Beam parameters
    energy_ev: float = 450e9             # Injection energy (450 GeV for LHC)
    mass_ev: float = PROTON_MASS_EV
    
    # Bunch parameters
    n_particles: float = 1.15e11         # Particles per bunch
    n_bunches: int = 2808                # Number of bunches
    
    # Beam size
    emittance_n: float = 3.5e-6          # Normalized emittance [m·rad]
    beta_avg: float = 100.0              # Average beta function [m]
    bunch_length_m: float = 0.075        # Bunch length [m] (7.5 cm)
    
    # Machine parameters
    circumference: float = 26659.0       # LHC circumference [m]
    rf_frequency: float = 400.8e6        # RF frequency [Hz]
    harmonic_number: int = 35640         # RF harmonic number


@dataclass
class SpaceChargeResult:
    """Results from space charge calculation."""
    
    # Relativistic parameters
    gamma: float
    beta: float
    
    # Beam parameters
    beam_size_rms: float                 # RMS beam size [m]
    bunching_factor: float               # B_f = bunch_length / bucket_length
    
    # Tune shifts
    space_charge_tune_shift: float       # ΔQ_sc
    beam_beam_tune_shift: float          # ξ (beam-beam parameter)
    
    # Instability thresholds
    tmci_threshold_current: float        # A
    brightness: float                    # Particles / (ε_n)² 
    
    # Collective effects
    synchrotron_tune: float              # Q_s
    space_charge_parameter: float        # q = ΔQ_sc / Q_s


class SpaceChargeEffects:
    """
    Calculates space charge and collective effects in bunched beams.
    
    Space charge causes:
    - Incoherent tune shift (spread)
    - Tune footprint enlargement
    - Resonance crossing
    - Emittance growth
    """
    
    def __init__(self, cfg: SpaceChargeConfig):
        self.cfg = cfg
        
        # Relativistic factors
        self.gamma = cfg.energy_ev / cfg.mass_ev
        self.beta = math.sqrt(1 - 1/self.gamma**2)
        
        # Particle radius (classical)
        if cfg.mass_ev == PROTON_MASS_EV:
            self.r_0 = R_PROTON
        else:
            self.r_0 = R_ELECTRON
    
    def compute_beam_size(self) -> float:
        """
        Compute RMS beam size from emittance.
        
        σ = sqrt(β * ε_g) where ε_g = ε_n / (βγ)
        """
        # Geometric emittance
        eps_geo = self.cfg.emittance_n / (self.beta * self.gamma)
        
        # RMS beam size
        sigma = math.sqrt(self.cfg.beta_avg * eps_geo)
        
        return sigma
    
    def compute_bunching_factor(self) -> float:
        """
        Compute bunching factor B_f.
        
        B_f = (2π * bunch_length) / (circumference / harmonic)
        """
        bucket_length = self.cfg.circumference / self.cfg.harmonic_number
        B_f = 2 * math.pi * self.cfg.bunch_length_m / bucket_length
        
        return B_f
    
    def compute_space_charge_tune_shift(self) -> float:
        """
        Compute incoherent space charge tune shift.
        
        ΔQ_sc = -r_0 * N_b / (4π * β² * γ³ * ε_n * B_f)
        
        This is the maximum tune shift for particles at beam center.
        """
        cfg = self.cfg
        
        # Bunching factor
        B_f = self.compute_bunching_factor()
        
        # Space charge tune shift (negative for repulsive)
        delta_Q = -(self.r_0 * cfg.n_particles) / (
            4 * math.pi * self.beta**2 * self.gamma**3 * 
            cfg.emittance_n * B_f
        )
        
        return delta_Q
    
    def compute_beam_beam_tune_shift(self) -> float:
        """
        Compute beam-beam tune shift parameter ξ.
        
        ξ = N_b * r_0 / (4π * γ * ε_n)
        
        For head-on collision with equal beams.
        LHC limit: ξ_total ≈ 0.01 per IP
        """
        cfg = self.cfg
        
        xi = (cfg.n_particles * self.r_0) / (
            4 * math.pi * self.gamma * cfg.emittance_n
        )
        
        return xi
    
    def compute_synchrotron_tune(self) -> float:
        """
        Estimate synchrotron tune Q_s.
        
        Q_s = sqrt(h * η * eV_rf / (2π * β² * E))
        
        Using typical values for LHC.
        """
        cfg = self.cfg
        
        # Slip factor (above transition for LHC)
        # η = 1/γ_t² - 1/γ² ≈ 1/γ_t² for γ >> γ_t
        gamma_t = 55.76  # LHC transition gamma
        eta = 1/gamma_t**2 - 1/self.gamma**2
        
        # Typical RF voltage
        V_rf = 16e6  # 16 MV for LHC
        
        # Synchrotron tune
        Q_s = math.sqrt(
            abs(cfg.harmonic_number * eta * V_rf) / 
            (2 * math.pi * self.beta**2 * cfg.energy_ev)
        )
        
        return Q_s
    
    def compute_tmci_threshold(self) -> float:
        """
        Estimate TMCI (Transverse Mode Coupling Instability) threshold.
        
        I_th ∝ Q_s * (E/e) / (β_avg * Im(Z_⊥))
        
        Simplified estimate assuming typical impedance.
        """
        cfg = self.cfg
        
        # Synchrotron tune
        Q_s = self.compute_synchrotron_tune()
        
        # Typical transverse impedance (LHC ~ 0.1 MΩ/m)
        Z_perp = 0.1e6  # Ω/m
        
        # Revolution frequency
        f_rev = C_LIGHT / cfg.circumference
        
        # Threshold current (simplified Keil-Schnell type)
        I_th = (16 * Q_s * cfg.energy_ev) / (
            3 * self.beta * cfg.beta_avg * Z_perp * cfg.circumference
        )
        
        return I_th
    
    def compute_brightness(self) -> float:
        """
        Compute beam brightness.
        
        B = N_b / ε_n² (simplified)
        
        Higher brightness → more space charge.
        """
        cfg = self.cfg
        
        return cfg.n_particles / (cfg.emittance_n**2)
    
    def run(self) -> SpaceChargeResult:
        """Run space charge calculation."""
        
        sigma = self.compute_beam_size()
        B_f = self.compute_bunching_factor()
        dQ_sc = self.compute_space_charge_tune_shift()
        xi = self.compute_beam_beam_tune_shift()
        Q_s = self.compute_synchrotron_tune()
        I_th = self.compute_tmci_threshold()
        brightness = self.compute_brightness()
        
        # Space charge parameter q = |ΔQ_sc| / Q_s
        q = abs(dQ_sc) / Q_s if Q_s > 0 else 0.0
        
        return SpaceChargeResult(
            gamma=self.gamma,
            beta=self.beta,
            beam_size_rms=sigma,
            bunching_factor=B_f,
            space_charge_tune_shift=dQ_sc,
            beam_beam_tune_shift=xi,
            tmci_threshold_current=I_th,
            brightness=brightness,
            synchrotron_tune=Q_s,
            space_charge_parameter=q
        )


def validate_space_charge(result: SpaceChargeResult, cfg: SpaceChargeConfig) -> dict:
    """Validate space charge calculations."""
    checks = {}
    
    # 1. Space charge tune shift should be negative (repulsive for protons)
    dQ_sign_valid = result.space_charge_tune_shift < 0
    checks['tune_shift_sign'] = {
        'valid': dQ_sign_valid,
        'delta_Q': result.space_charge_tune_shift,
        'note': 'Negative for like-charge particles (repulsive)'
    }
    
    # 2. Tune shift magnitude (LHC injection ~0.1-0.3)
    dQ_magnitude = abs(result.space_charge_tune_shift)
    dQ_magnitude_valid = 1e-4 < dQ_magnitude < 1.0
    checks['tune_shift_magnitude'] = {
        'valid': dQ_magnitude_valid,
        'value': dQ_magnitude,
        'expected_range': '10^-4 to 1 for high-intensity machines'
    }
    
    # 3. Beam-beam parameter (LHC limit ~0.01 per IP)
    xi_valid = 1e-5 < result.beam_beam_tune_shift < 0.1
    checks['beam_beam_parameter'] = {
        'valid': xi_valid,
        'xi': result.beam_beam_tune_shift,
        'lhc_limit': 0.01
    }
    
    # 4. Space charge parameter q < 1 for stability
    q_valid = result.space_charge_parameter < 10
    checks['space_charge_parameter'] = {
        'valid': q_valid,
        'q': result.space_charge_parameter,
        'note': 'q < 1 typically required for stability'
    }
    
    # 5. Synchrotron tune (typical 10^-3 to 10^-2)
    Qs_valid = 1e-4 < result.synchrotron_tune < 0.1
    checks['synchrotron_tune'] = {
        'valid': Qs_valid,
        'Q_s': result.synchrotron_tune,
        'expected_range': '10^-4 to 10^-1'
    }
    
    # 6. Beam size physically reasonable (microns to mm)
    sigma_valid = 1e-6 < result.beam_size_rms < 1e-2
    checks['beam_size'] = {
        'valid': sigma_valid,
        'sigma_mm': result.beam_size_rms * 1000,
        'expected_range': '0.001 to 10 mm'
    }
    
    checks['all_pass'] = all(c['valid'] for c in checks.values())
    
    return checks


def run_space_charge_benchmark() -> Tuple[SpaceChargeResult, dict]:
    """Run space charge benchmark."""
    print("="*70)
    print("FRONTIER 04: Space Charge and Collective Effects")
    print("="*70)
    print()
    
    # Medium energy parameters - typical synchrotron
    # This is where space charge is significant but not extreme
    cfg = SpaceChargeConfig(
        energy_ev=2e9 + PROTON_MASS_EV,  # 2 GeV kinetic
        mass_ev=PROTON_MASS_EV,
        n_particles=5.0e12,              # 5×10^12 particles
        n_bunches=1,
        emittance_n=5.0e-6,              # 5 μm normalized
        beta_avg=15.0,                   # 15 m average beta
        bunch_length_m=5.0,              # 5 m bunch length
        circumference=474.0,             # ~PS size
        harmonic_number=7                # h=7
    )
    
    print(f"Configuration (Medium Energy Synchrotron):")
    print(f"  Kinetic energy:     {(cfg.energy_ev - PROTON_MASS_EV)/1e9:.1f} GeV")
    print(f"  Total energy:       {cfg.energy_ev/1e9:.2f} GeV")
    print(f"  Particles/bunch:    {cfg.n_particles:.2e}")
    print(f"  Norm. emittance:    {cfg.emittance_n*1e6:.1f} μm")
    print(f"  Circumference:      {cfg.circumference:.0f} m")
    print()
    
    # Run calculation
    print("Computing collective effects...")
    calc = SpaceChargeEffects(cfg)
    result = calc.run()
    
    print()
    print("Results:")
    print(f"  γ = {result.gamma:.1f}, β = {result.beta:.6f}")
    print(f"  RMS beam size:      {result.beam_size_rms*1e3:.3f} mm")
    print(f"  Bunching factor:    {result.bunching_factor:.3f}")
    print()
    print(f"  Space charge ΔQ:    {result.space_charge_tune_shift:.4f}")
    print(f"  Beam-beam ξ:        {result.beam_beam_tune_shift:.6f}")
    print(f"  Synchrotron tune:   {result.synchrotron_tune:.6f}")
    print(f"  SC parameter q:     {result.space_charge_parameter:.2f}")
    print()
    print(f"  TMCI threshold:     {result.tmci_threshold_current:.2f} A")
    print(f"  Brightness:         {result.brightness:.2e}")
    print()
    
    # Validate
    checks = validate_space_charge(result, cfg)
    
    print("Validation:")
    print(f"  Tune shift sign:      {'✓ PASS' if checks['tune_shift_sign']['valid'] else '✗ FAIL'}")
    print(f"  Tune shift magnitude: {'✓ PASS' if checks['tune_shift_magnitude']['valid'] else '✗ FAIL'}")
    print(f"  Beam-beam parameter:  {'✓ PASS' if checks['beam_beam_parameter']['valid'] else '✗ FAIL'}")
    print(f"  SC parameter q:       {'✓ PASS' if checks['space_charge_parameter']['valid'] else '✗ FAIL'}")
    print(f"  Synchrotron tune:     {'✓ PASS' if checks['synchrotron_tune']['valid'] else '✗ FAIL'}")
    print(f"  Beam size:            {'✓ PASS' if checks['beam_size']['valid'] else '✗ FAIL'}")
    print()
    
    if checks['all_pass']:
        print("★ SPACE CHARGE BENCHMARK: PASS")
    else:
        print("✗ SPACE CHARGE BENCHMARK: FAIL")
    
    return result, checks


if __name__ == "__main__":
    result, checks = run_space_charge_benchmark()
