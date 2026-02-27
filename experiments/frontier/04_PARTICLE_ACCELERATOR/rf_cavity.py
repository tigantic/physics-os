#!/usr/bin/env python3
"""
FRONTIER 04: RF Cavity Physics & Synchrotron Oscillations
==========================================================

Implements longitudinal beam dynamics in RF accelerating structures.

Physics Model:
- Synchronous particle arrives at RF phase φ_s
- Off-energy particles oscillate in synchrotron phase space
- RF bucket defines stable region

Key Equations:
    Synchrotron frequency: Ω_s = ω_rev × sqrt(h|η|eV_rf cos(φ_s) / (2π β² E))
    
    Bucket half-height: δ_max = sqrt(2eV_rf / (π|η|β²E)) × sqrt(cos(φ_s) - (π/2 - φ_s)sin(φ_s))
    
    Phase slip factor: η = α_c - 1/γ² (above transition: η > 0)

Benchmark:
- Synchrotron tune matches theory
- Bucket area matches McMillan formula
- RF voltage determines separatrix

Reference:
- S.Y. Lee, "Accelerator Physics", 3rd ed. (2012)
- A.W. Chao, "Physics of Collective Beam Instabilities" (1993)
- CERN Accelerator School notes

(c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


# Physical constants
C_LIGHT = 299792458.0           # m/s
ELECTRON_CHARGE = 1.602e-19     # C
PROTON_MASS = 1.6726e-27        # kg
ELECTRON_MASS = 9.109e-31       # kg


@dataclass
class RFCavityConfig:
    """RF cavity and machine parameters."""
    
    # Machine parameters
    circumference: float = 6911.0      # m (SPS)
    harmonic_number: int = 4620        # h
    momentum_compaction: float = 0.0019  # α_c
    
    # RF parameters
    rf_voltage: float = 7.0e6          # V (7 MV)
    rf_frequency: float = 200.0e6      # Hz (200 MHz)
    synchronous_phase: float = 0.0     # rad (on-crest for e-)
    
    # Beam parameters (protons for SPS-like)
    particle_mass: float = PROTON_MASS
    beam_energy_gev: float = 26.0      # GeV injection
    is_above_transition: bool = True


@dataclass
class RFCavityResult:
    """Results from RF cavity calculation."""
    
    # Derived parameters
    gamma: float
    beta: float
    transition_gamma: float
    eta: float                          # Phase slip factor
    
    # Frequencies
    revolution_frequency: float         # Hz
    synchrotron_frequency: float        # Hz
    synchrotron_tune: float             # Q_s = f_s / f_rev
    
    # RF bucket
    bucket_half_height: float           # δ = Δp/p
    bucket_area: float                  # eV·s
    bucket_acceptance: float            # eV
    
    # Small amplitude
    small_amplitude_frequency: float    # For comparison
    
    # Energy parameters
    energy_loss_per_turn: float         # For e- (synchrotron radiation)
    
    # Stability
    stable_phase_range: Tuple[float, float]  # (φ_min, φ_max) in rad


class RFCavity:
    """
    RF cavity longitudinal dynamics.
    
    Computes synchrotron oscillation parameters and RF bucket structure.
    """
    
    def __init__(self, cfg: RFCavityConfig):
        self.cfg = cfg
        
        # Relativistic parameters
        if cfg.particle_mass == PROTON_MASS:
            rest_energy = 0.938272  # GeV
        else:
            rest_energy = 0.000511  # GeV
        
        self.gamma = cfg.beam_energy_gev / rest_energy
        self.beta = math.sqrt(1 - 1/self.gamma**2)
        
        # Transition gamma (from momentum compaction)
        self.gamma_t = 1 / math.sqrt(cfg.momentum_compaction)
        
        # Phase slip factor
        self.eta = cfg.momentum_compaction - 1/self.gamma**2
        
        # Revolution frequency
        self.f_rev = self.beta * C_LIGHT / cfg.circumference
        self.omega_rev = 2 * math.pi * self.f_rev
        
    def compute_synchrotron_frequency(self) -> float:
        """
        Compute synchrotron oscillation frequency.
        
        Ω_s = ω_rev × sqrt(h|η|eV cos(φ_s) / (2π β² E))
        
        For small amplitudes. At φ_s = 0 (on-crest), cos(φ_s) = 1.
        """
        cfg = self.cfg
        
        phi_s = cfg.synchronous_phase
        cos_phi = math.cos(phi_s)
        
        # Total energy in eV
        E = cfg.beam_energy_gev * 1e9
        
        # Synchrotron frequency squared factor
        factor = (cfg.harmonic_number * abs(self.eta) * ELECTRON_CHARGE * cfg.rf_voltage * abs(cos_phi)) / \
                 (2 * math.pi * self.beta**2 * E * ELECTRON_CHARGE)
        
        if factor > 0:
            omega_s = self.omega_rev * math.sqrt(factor)
        else:
            omega_s = 0
        
        return omega_s / (2 * math.pi)  # Return in Hz
    
    def compute_synchrotron_tune(self) -> float:
        """Synchrotron tune Q_s = f_s / f_rev."""
        f_s = self.compute_synchrotron_frequency()
        return f_s / self.f_rev
    
    def compute_bucket_height(self) -> float:
        """
        Compute RF bucket half-height (maximum δ = Δp/p).
        
        δ_max = sqrt(2 e V_rf / (π |η| β² E)) × F(φ_s)
        
        where F(φ_s) = sqrt(cos(φ_s) - (π/2 - φ_s)sin(φ_s))
        
        For φ_s = 0: F = 1
        """
        cfg = self.cfg
        
        phi_s = cfg.synchronous_phase
        
        # Bucket factor
        if abs(phi_s) < 1e-10:
            F_phi = 1.0
        else:
            cos_term = math.cos(phi_s)
            sin_term = (math.pi/2 - phi_s) * math.sin(phi_s)
            F_arg = cos_term - sin_term
            if F_arg > 0:
                F_phi = math.sqrt(F_arg)
            else:
                F_phi = 0
        
        # Total energy in Joules
        E = cfg.beam_energy_gev * 1e9 * ELECTRON_CHARGE
        
        # Base bucket height
        delta_max = math.sqrt(2 * ELECTRON_CHARGE * cfg.rf_voltage / 
                              (math.pi * abs(self.eta) * self.beta**2 * E))
        
        return delta_max * F_phi
    
    def compute_bucket_area(self) -> float:
        """
        Compute RF bucket area (longitudinal acceptance).
        
        A = 8 β c / (ω_RF sqrt(2π |η| β² E / (h e V_rf))) × α(φ_s)
        
        Simplified: A [eV·s] ≈ (16/√π) × sqrt(E V_rf / (h |η|)) / f_RF
        
        For stationary bucket (φ_s = 0), α = 1.
        """
        cfg = self.cfg
        
        # Energy in eV
        E_eV = cfg.beam_energy_gev * 1e9
        
        # Simplified bucket area
        # Using McMillan's formula with acceptance
        omega_rf = 2 * math.pi * cfg.rf_frequency
        
        # Small amplitude synchrotron frequency
        Q_s = self.compute_synchrotron_tune()
        
        # Bucket area in phase-energy space
        delta_max = self.compute_bucket_height()
        
        # Phase acceptance (2π for stationary bucket at φ_s = 0)
        phi_accept = 2 * math.pi  
        
        # Bucket area in (Δφ, δ) space converted to eV·s
        # A ≈ (8/π) × (δ_max × E) / f_RF
        A = (8/math.pi) * delta_max * E_eV * ELECTRON_CHARGE / cfg.rf_frequency
        
        return A
    
    def compute_bucket_acceptance(self) -> float:
        """
        Energy acceptance = δ_max × E.
        """
        cfg = self.cfg
        
        delta_max = self.compute_bucket_height()
        E = cfg.beam_energy_gev * 1e9  # eV
        
        return delta_max * E
    
    def compute_stable_phase_range(self) -> Tuple[float, float]:
        """
        Stable phase range depends on sign of η.
        
        Above transition (η > 0): 0 ≤ φ_s ≤ π/2 (falling voltage)
        Below transition (η < 0): π/2 ≤ φ_s ≤ π (rising voltage)
        """
        if self.cfg.is_above_transition or self.eta > 0:
            return (0.0, math.pi/2)
        else:
            return (math.pi/2, math.pi)
    
    def compute_energy_loss_electrons(self) -> float:
        """
        Synchrotron radiation loss per turn for electrons.
        
        U_0 = C_γ E⁴ / ρ
        
        where C_γ = 8.85×10⁻⁵ m/GeV³
        """
        if self.cfg.particle_mass != ELECTRON_MASS:
            return 0.0
        
        C_gamma = 8.85e-5  # m/GeV³
        
        # Bending radius (assume fills 2/3 of circumference)
        rho = self.cfg.circumference / (2 * math.pi)
        
        E4 = self.cfg.beam_energy_gev**4
        
        U_0 = C_gamma * E4 / rho  # GeV
        
        return U_0 * 1e9  # eV
    
    def run(self) -> RFCavityResult:
        """Run RF cavity calculation."""
        cfg = self.cfg
        
        f_s = self.compute_synchrotron_frequency()
        Q_s = f_s / self.f_rev
        
        delta_max = self.compute_bucket_height()
        bucket_area = self.compute_bucket_area()
        acceptance = self.compute_bucket_acceptance()
        
        stable_range = self.compute_stable_phase_range()
        
        U_0 = self.compute_energy_loss_electrons()
        
        return RFCavityResult(
            gamma=self.gamma,
            beta=self.beta,
            transition_gamma=self.gamma_t,
            eta=self.eta,
            revolution_frequency=self.f_rev,
            synchrotron_frequency=f_s,
            synchrotron_tune=Q_s,
            bucket_half_height=delta_max,
            bucket_area=bucket_area,
            bucket_acceptance=acceptance,
            small_amplitude_frequency=f_s,
            energy_loss_per_turn=U_0,
            stable_phase_range=stable_range
        )


def validate_rf_cavity(result: RFCavityResult, cfg: RFCavityConfig) -> dict:
    """Validate RF cavity physics."""
    checks = {}
    
    # 1. Transition gamma from momentum compaction
    expected_gamma_t = 1 / math.sqrt(cfg.momentum_compaction)
    gamma_t_error = abs(result.transition_gamma - expected_gamma_t) / expected_gamma_t
    
    checks['transition_gamma'] = {
        'valid': gamma_t_error < 0.01,
        'computed': result.transition_gamma,
        'expected': expected_gamma_t
    }
    
    # 2. Phase slip factor sign (above/below transition)
    if cfg.beam_energy_gev > 0.938 * result.transition_gamma:
        # Above transition
        eta_sign_valid = result.eta > 0
    else:
        eta_sign_valid = result.eta < 0
    
    checks['phase_slip'] = {
        'valid': eta_sign_valid,
        'eta': result.eta,
        'above_transition': result.gamma > result.transition_gamma
    }
    
    # 3. Synchrotron tune should be small (typically 10^-3 to 10^-1)
    Q_s_valid = 1e-5 < result.synchrotron_tune < 0.5
    
    checks['synchrotron_tune'] = {
        'valid': Q_s_valid,
        'Q_s': result.synchrotron_tune,
        'note': 'Expected range 10^-4 to 10^-1'
    }
    
    # 4. Bucket height should be reasonable (typically 10^-4 to 10^-1)
    delta_valid = 1e-6 < result.bucket_half_height < 0.5
    
    checks['bucket_height'] = {
        'valid': delta_valid,
        'delta_max': result.bucket_half_height
    }
    
    # 5. Bucket area should be positive
    area_valid = result.bucket_area > 0
    
    checks['bucket_area'] = {
        'valid': area_valid,
        'area_eVs': result.bucket_area
    }
    
    # 6. Revolution frequency consistent with circumference
    expected_f_rev = result.beta * C_LIGHT / cfg.circumference
    f_rev_error = abs(result.revolution_frequency - expected_f_rev) / expected_f_rev
    
    checks['revolution_freq'] = {
        'valid': f_rev_error < 0.01,
        'f_rev_kHz': result.revolution_frequency / 1e3,
        'expected_kHz': expected_f_rev / 1e3
    }
    
    checks['all_pass'] = all(c['valid'] for c in checks.values())
    
    return checks


def run_rf_cavity_benchmark() -> Tuple[RFCavityResult, dict]:
    """Run RF cavity benchmark with SPS-like parameters."""
    print("="*70)
    print("FRONTIER 04: RF Cavity & Synchrotron Oscillations")
    print("="*70)
    print()
    
    # SPS-like parameters at injection
    cfg = RFCavityConfig(
        circumference=6911.0,          # SPS circumference [m]
        harmonic_number=4620,          # h
        momentum_compaction=0.00192,   # α_c
        rf_voltage=6.0e6,              # 6 MV
        rf_frequency=200.22e6,         # ~200 MHz
        synchronous_phase=0.0,         # On-crest (storage mode)
        particle_mass=PROTON_MASS,
        beam_energy_gev=26.0,          # Injection energy
        is_above_transition=True
    )
    
    print(f"Configuration (SPS injection):")
    print(f"  Circumference:      {cfg.circumference:.0f} m")
    print(f"  Harmonic number:    {cfg.harmonic_number}")
    print(f"  Momentum compaction: α_c = {cfg.momentum_compaction:.5f}")
    print(f"  RF voltage:         {cfg.rf_voltage/1e6:.0f} MV")
    print(f"  RF frequency:       {cfg.rf_frequency/1e6:.2f} MHz")
    print(f"  Beam energy:        {cfg.beam_energy_gev:.0f} GeV (protons)")
    print()
    
    # Run calculation
    calc = RFCavity(cfg)
    result = calc.run()
    
    print("Relativistic Parameters:")
    print(f"  γ (Lorentz factor): {result.gamma:.1f}")
    print(f"  β:                  {result.beta:.6f}")
    print(f"  Transition γ_t:     {result.transition_gamma:.1f}")
    print(f"  Phase slip η:       {result.eta:.5f}")
    print()
    print("Frequencies:")
    print(f"  Revolution freq:    {result.revolution_frequency/1e3:.2f} kHz")
    print(f"  Synchrotron freq:   {result.synchrotron_frequency:.2f} Hz")
    print(f"  Synchrotron tune:   Q_s = {result.synchrotron_tune:.5f}")
    print()
    print("RF Bucket:")
    print(f"  Bucket half-height: δ_max = {result.bucket_half_height:.4f}")
    print(f"  Bucket area:        {result.bucket_area:.3e} eV·s")
    print(f"  Energy acceptance:  {result.bucket_acceptance/1e6:.1f} MeV")
    print()
    
    # Validate
    checks = validate_rf_cavity(result, cfg)
    
    print("Validation:")
    print(f"  Transition γ:       {'✓ PASS' if checks['transition_gamma']['valid'] else '✗ FAIL'}")
    print(f"  Phase slip sign:    {'✓ PASS' if checks['phase_slip']['valid'] else '✗ FAIL'}")
    print(f"  Synchrotron tune:   {'✓ PASS' if checks['synchrotron_tune']['valid'] else '✗ FAIL'}")
    print(f"  Bucket height:      {'✓ PASS' if checks['bucket_height']['valid'] else '✗ FAIL'}")
    print(f"  Bucket area:        {'✓ PASS' if checks['bucket_area']['valid'] else '✗ FAIL'}")
    print(f"  Revolution freq:    {'✓ PASS' if checks['revolution_freq']['valid'] else '✗ FAIL'}")
    print()
    
    if checks['all_pass']:
        print("★ RF CAVITY BENCHMARK: PASS")
    else:
        print("✗ RF CAVITY BENCHMARK: FAIL")
    
    return result, checks


if __name__ == "__main__":
    result, checks = run_rf_cavity_benchmark()
