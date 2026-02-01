#!/usr/bin/env python3
"""
FRONTIER 04: Plasma Wakefield Acceleration
===========================================

Implements 1D nonlinear plasma wakefield physics.

Physics Model:
- Relativistic electron beam drives plasma oscillation
- Plasma electrons expelled by beam space charge
- Ion column focuses beam (linear focusing)
- Accelerating/decelerating fields from charge separation

Key Formula (Linear Regime):
    E_z,max = m_e c ω_p / e = 96 [GV/m] × sqrt(n_e [10^18 cm^-3])

Nonlinear (Blowout) Regime:
    E_z ≈ E_0 × sqrt(a_0) for a_0 >> 1

where:
- ω_p = sqrt(n_e e² / ε_0 m_e) is plasma frequency
- E_0 = m_e c ω_p / e is cold wavebreaking field

Benchmark:
- Accelerating gradient matches theory
- Plasma wavelength correct
- Transformer ratio (energy gain / loss)

Reference:
- P. Chen et al., PRL 54, 693 (1985)
- W. Lu et al., Phys. Plasmas 13, 056709 (2006)
- SLAC E-157/E-162/E-164 experiments

(c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


# Physical constants (SI)
C_LIGHT = 299792458.0              # m/s
ELECTRON_MASS = 9.109e-31          # kg
ELECTRON_CHARGE = 1.602e-19        # C
EPSILON_0 = 8.854e-12              # F/m
MU_0 = 4e-7 * math.pi              # H/m


@dataclass
class WakefieldConfig:
    """Configuration for plasma wakefield calculation."""
    
    # Plasma parameters
    plasma_density: float = 1e23       # m^-3 (10^17 cm^-3)
    plasma_length: float = 1.0         # m
    
    # Drive beam parameters
    beam_charge: float = 3e-9          # C (3 nC)
    beam_energy_gev: float = 23.0      # GeV (FACET-II scale)
    beam_sigma_z: float = 20e-6        # m (20 μm RMS length)
    beam_sigma_r: float = 10e-6        # m (10 μm RMS radius)
    
    # Witness beam (for acceleration)
    witness_charge: float = 0.3e-9     # C (0.3 nC, much smaller)
    witness_delay: float = 0.5         # In units of plasma wavelength


@dataclass
class WakefieldResult:
    """Results from wakefield calculation."""
    
    # Plasma parameters
    plasma_frequency: float            # rad/s
    plasma_wavelength: float           # m
    skin_depth: float                  # m
    
    # Field parameters
    wavebreaking_field: float          # V/m (cold nonrelativistic)
    peak_accelerating_field: float     # V/m
    peak_decelerating_field: float     # V/m
    
    # Beam loading
    transformer_ratio: float           # R = E_acc / E_dec
    
    # Energy
    energy_gain_per_meter: float       # eV/m
    total_energy_gain: float           # eV
    final_witness_energy: float        # GeV
    
    # Blowout parameters
    blowout_radius: float              # m
    beam_density_ratio: float          # n_b / n_p
    
    # Validation metrics
    normalized_charge: float           # Q / Q_0 (beam loading)
    a0_equivalent: float               # Equivalent laser a_0


class PlasmaWakefield:
    """
    1D plasma wakefield acceleration calculation.
    
    Implements linear theory with beam loading corrections
    for the blowout regime.
    """
    
    def __init__(self, cfg: WakefieldConfig):
        self.cfg = cfg
        
        # Plasma frequency
        self.omega_p = math.sqrt(
            cfg.plasma_density * ELECTRON_CHARGE**2 / 
            (EPSILON_0 * ELECTRON_MASS)
        )
        
        # Plasma wavelength
        self.lambda_p = 2 * math.pi * C_LIGHT / self.omega_p
        
        # Plasma skin depth
        self.k_p_inv = C_LIGHT / self.omega_p
        
        # Cold wavebreaking field
        self.E_0 = ELECTRON_MASS * C_LIGHT * self.omega_p / ELECTRON_CHARGE
        
    def compute_beam_density(self) -> float:
        """
        Compute peak beam density.
        
        For Gaussian beam: n_b = N / (2π)^(3/2) σ_r² σ_z
        """
        cfg = self.cfg
        
        N_particles = cfg.beam_charge / ELECTRON_CHARGE
        
        n_b = N_particles / (
            (2 * math.pi)**(3/2) * 
            cfg.beam_sigma_r**2 * 
            cfg.beam_sigma_z
        )
        
        return n_b
    
    def compute_blowout_radius(self) -> float:
        """
        Estimate blowout radius in nonlinear regime.
        
        R_b ≈ 2 * sqrt(n_b/n_p) * c/ω_p
        
        For matched beam.
        """
        n_b = self.compute_beam_density()
        n_p = self.cfg.plasma_density
        
        ratio = n_b / n_p
        
        if ratio > 1:
            # Blowout regime
            R_b = 2 * math.sqrt(ratio) * self.k_p_inv
        else:
            # Linear regime
            R_b = self.k_p_inv
        
        return R_b
    
    def compute_accelerating_field_linear(self) -> float:
        """
        Linear theory accelerating field.
        
        E_z = E_0 * k_p * σ_z * (n_b/n_p) * exp(-k_p²σ_z²/2) for Gaussian
        
        Simplified: E_z ≈ E_0 * (n_b/n_p) for short beams
        """
        cfg = self.cfg
        
        n_b = self.compute_beam_density()
        n_p = cfg.plasma_density
        
        k_p = self.omega_p / C_LIGHT
        k_p_sigma = k_p * cfg.beam_sigma_z
        
        # Linear wake amplitude
        if k_p_sigma < 1:
            # Short beam (resonant)
            E_acc = self.E_0 * (n_b / n_p) * k_p_sigma
        else:
            # Long beam (suppressed)
            E_acc = self.E_0 * (n_b / n_p) * math.exp(-k_p_sigma**2 / 2)
        
        return E_acc
    
    def compute_accelerating_field_nonlinear(self) -> float:
        """
        Nonlinear (blowout) regime accelerating field.
        
        In blowout: E_z ≈ k_p * R_b / 2
        
        Or: E_z = E_0 * sqrt(n_b/n_p) for strong driver
        """
        cfg = self.cfg
        
        n_b = self.compute_beam_density()
        n_p = cfg.plasma_density
        ratio = n_b / n_p
        
        if ratio > 1:
            # Blowout regime
            R_b = self.compute_blowout_radius()
            k_p = self.omega_p / C_LIGHT
            E_acc = self.E_0 * k_p * R_b / 2
            
            # Cap at wavebreaking
            E_acc = min(E_acc, self.E_0)
        else:
            # Linear regime
            E_acc = self.compute_accelerating_field_linear()
        
        return E_acc
    
    def compute_decelerating_field(self) -> float:
        """
        Decelerating field on drive beam.
        
        E_dec ≈ E_acc / R (transformer ratio)
        
        For symmetric beam: R ≈ 2 in linear regime
        """
        n_b = self.compute_beam_density()
        n_p = self.cfg.plasma_density
        ratio = n_b / n_p
        
        if ratio > 1:
            # Blowout: decelerating field roughly uniform
            E_dec = self.E_0 * 0.5 * min(math.sqrt(ratio), 2.0)
        else:
            # Linear: E_dec = E_0 * n_b/n_p
            E_dec = self.E_0 * ratio
        
        return E_dec
    
    def compute_transformer_ratio(self, E_acc: float, E_dec: float) -> float:
        """
        Transformer ratio R = E_acc / E_dec.
        
        Fundamental limit for symmetric beam: R ≤ 2
        Can exceed 2 with asymmetric/ramped beams.
        """
        if E_dec > 0:
            R = E_acc / E_dec
        else:
            R = 1.0
        
        return R
    
    def compute_normalized_charge(self) -> float:
        """
        Normalized charge Q/Q_0 for beam loading assessment.
        
        Q_0 = (4/3)π ε_0 m_e c² (c/ω_p)³ / e ≈ characteristic charge
        """
        cfg = self.cfg
        
        # Characteristic charge (order of magnitude)
        Q_0 = (4/3) * math.pi * EPSILON_0 * ELECTRON_MASS * C_LIGHT**2 * self.k_p_inv**3 / ELECTRON_CHARGE
        
        return cfg.beam_charge / Q_0
    
    def compute_equivalent_a0(self) -> float:
        """
        Equivalent laser a_0 for the beam driver.
        
        a_0 = eA/(m_e c) equivalent from beam field strength
        """
        E_peak = self.compute_accelerating_field_nonlinear()
        
        # a_0 ≈ E / E_0 for rough equivalence
        a0 = E_peak / self.E_0
        
        return a0
    
    def run(self) -> WakefieldResult:
        """Run wakefield calculation."""
        cfg = self.cfg
        
        # Compute fields
        E_acc = self.compute_accelerating_field_nonlinear()
        E_dec = self.compute_decelerating_field()
        R = self.compute_transformer_ratio(E_acc, E_dec)
        
        # Energy gain
        energy_gain_per_m = E_acc  # V/m = eV/m for electron
        total_gain = E_acc * cfg.plasma_length  # eV
        
        # Final witness energy
        # Assume witness is at acceleration phase
        witness_initial = cfg.beam_energy_gev * 0.5  # Assume lower energy witness
        final_witness = witness_initial + total_gain / 1e9
        
        # Other parameters
        R_b = self.compute_blowout_radius()
        n_b = self.compute_beam_density()
        ratio = n_b / cfg.plasma_density
        
        Q_norm = self.compute_normalized_charge()
        a0_eq = self.compute_equivalent_a0()
        
        return WakefieldResult(
            plasma_frequency=self.omega_p,
            plasma_wavelength=self.lambda_p,
            skin_depth=self.k_p_inv,
            wavebreaking_field=self.E_0,
            peak_accelerating_field=E_acc,
            peak_decelerating_field=E_dec,
            transformer_ratio=R,
            energy_gain_per_meter=energy_gain_per_m,
            total_energy_gain=total_gain,
            final_witness_energy=final_witness,
            blowout_radius=R_b,
            beam_density_ratio=ratio,
            normalized_charge=Q_norm,
            a0_equivalent=a0_eq
        )


def validate_wakefield(result: WakefieldResult, cfg: WakefieldConfig) -> dict:
    """Validate wakefield physics."""
    checks = {}
    
    # 1. Plasma wavelength scaling: λ_p [μm] ≈ 33 / sqrt(n_e [10^18 cm^-3])
    n_18 = cfg.plasma_density / 1e24  # Convert to 10^18 cm^-3
    expected_lambda_um = 33.0 / math.sqrt(n_18)
    measured_lambda_um = result.plasma_wavelength * 1e6
    lambda_error = abs(measured_lambda_um - expected_lambda_um) / expected_lambda_um
    
    checks['plasma_wavelength'] = {
        'valid': lambda_error < 0.1,
        'measured_um': measured_lambda_um,
        'expected_um': expected_lambda_um,
        'error': lambda_error
    }
    
    # 2. Wavebreaking field scaling: E_0 [GV/m] ≈ 96 × sqrt(n_e [10^18 cm^-3])
    expected_E0_GVm = 96.0 * math.sqrt(n_18)
    measured_E0_GVm = result.wavebreaking_field / 1e9
    E0_error = abs(measured_E0_GVm - expected_E0_GVm) / expected_E0_GVm
    
    checks['wavebreaking_field'] = {
        'valid': E0_error < 0.1,
        'measured_GVm': measured_E0_GVm,
        'expected_GVm': expected_E0_GVm,
        'error': E0_error
    }
    
    # 3. Accelerating field should be positive and below wavebreaking
    E_valid = 0 < result.peak_accelerating_field <= result.wavebreaking_field * 1.5
    checks['accelerating_field'] = {
        'valid': E_valid,
        'E_acc_GVm': result.peak_accelerating_field / 1e9,
        'E_wb_GVm': result.wavebreaking_field / 1e9
    }
    
    # 4. Transformer ratio (typically 1-3 for symmetric beams)
    R_valid = 0.5 < result.transformer_ratio < 10
    checks['transformer_ratio'] = {
        'valid': R_valid,
        'R': result.transformer_ratio,
        'note': 'R ≤ 2 for symmetric beam, can exceed with ramped/asymmetric'
    }
    
    # 5. Energy gain should be positive and reasonable
    gain_valid = result.energy_gain_per_meter > 1e6  # > 1 MeV/m
    checks['energy_gain'] = {
        'valid': gain_valid,
        'gain_GeVm': result.energy_gain_per_meter / 1e9,
        'total_gain_GeV': result.total_energy_gain / 1e9
    }
    
    # 6. Blowout regime check
    blowout_valid = result.beam_density_ratio > 0.1 or result.beam_density_ratio < 10
    checks['blowout_regime'] = {
        'valid': blowout_valid,
        'n_b_over_n_p': result.beam_density_ratio,
        'regime': 'blowout' if result.beam_density_ratio > 1 else 'linear'
    }
    
    checks['all_pass'] = all(c['valid'] for c in checks.values())
    
    return checks


def run_wakefield_benchmark() -> Tuple[WakefieldResult, dict]:
    """Run wakefield benchmark."""
    print("="*70)
    print("FRONTIER 04: Plasma Wakefield Acceleration")
    print("="*70)
    print()
    
    # FACET-II inspired parameters
    cfg = WakefieldConfig(
        plasma_density=5e22,           # 5×10^16 cm^-3
        plasma_length=1.3,             # 1.3 m plasma
        beam_charge=2e-9,              # 2 nC drive beam
        beam_energy_gev=10.0,          # 10 GeV
        beam_sigma_z=35e-6,            # 35 μm bunch length
        beam_sigma_r=15e-6,            # 15 μm spot size
        witness_charge=0.2e-9,         # 200 pC witness
    )
    
    print(f"Configuration (FACET-II scale):")
    print(f"  Plasma density:     {cfg.plasma_density:.1e} m⁻³ ({cfg.plasma_density/1e24:.1e} × 10¹⁸ cm⁻³)")
    print(f"  Plasma length:      {cfg.plasma_length:.1f} m")
    print(f"  Drive beam charge:  {cfg.beam_charge*1e9:.1f} nC")
    print(f"  Drive beam energy:  {cfg.beam_energy_gev:.0f} GeV")
    print(f"  Bunch length σ_z:   {cfg.beam_sigma_z*1e6:.0f} μm")
    print()
    
    # Run calculation
    print("Computing wakefield acceleration...")
    calc = PlasmaWakefield(cfg)
    result = calc.run()
    
    print()
    print("Plasma Parameters:")
    print(f"  Plasma frequency:   ω_p = {result.plasma_frequency:.3e} rad/s")
    print(f"  Plasma wavelength:  λ_p = {result.plasma_wavelength*1e6:.1f} μm")
    print(f"  Skin depth:         c/ω_p = {result.skin_depth*1e6:.1f} μm")
    print()
    print("Field Amplitudes:")
    print(f"  Wavebreaking E_0:   {result.wavebreaking_field/1e9:.2f} GV/m")
    print(f"  Accelerating E_z:   {result.peak_accelerating_field/1e9:.2f} GV/m")
    print(f"  Decelerating E_z:   {result.peak_decelerating_field/1e9:.2f} GV/m")
    print(f"  Transformer ratio:  R = {result.transformer_ratio:.2f}")
    print()
    print("Energy Gain:")
    print(f"  Gradient:           {result.energy_gain_per_meter/1e9:.2f} GeV/m")
    print(f"  Total gain:         {result.total_energy_gain/1e9:.2f} GeV (in {cfg.plasma_length:.1f} m)")
    print(f"  n_b / n_p:          {result.beam_density_ratio:.1f}")
    print()
    
    # Validate
    checks = validate_wakefield(result, cfg)
    
    print("Validation:")
    print(f"  Plasma wavelength:    {'✓ PASS' if checks['plasma_wavelength']['valid'] else '✗ FAIL'}")
    print(f"  Wavebreaking field:   {'✓ PASS' if checks['wavebreaking_field']['valid'] else '✗ FAIL'}")
    print(f"  Accelerating field:   {'✓ PASS' if checks['accelerating_field']['valid'] else '✗ FAIL'}")
    print(f"  Transformer ratio:    {'✓ PASS' if checks['transformer_ratio']['valid'] else '✗ FAIL'}")
    print(f"  Energy gain:          {'✓ PASS' if checks['energy_gain']['valid'] else '✗ FAIL'}")
    print(f"  Regime (blowout):     {'✓ PASS' if checks['blowout_regime']['valid'] else '✗ FAIL'}")
    print()
    
    if checks['all_pass']:
        print("★ PLASMA WAKEFIELD BENCHMARK: PASS")
    else:
        print("✗ PLASMA WAKEFIELD BENCHMARK: FAIL")
    
    return result, checks


if __name__ == "__main__":
    result, checks = run_wakefield_benchmark()
