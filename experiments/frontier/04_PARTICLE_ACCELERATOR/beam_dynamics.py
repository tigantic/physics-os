#!/usr/bin/env python3
"""
FRONTIER 04: Beam Dynamics in Particle Accelerators
=====================================================

Implements single-particle and envelope dynamics for charged particle beams.

Physics Model:
- Linear transport matrices (FODO cells, drift, quadrupoles)
- Betatron oscillations and Twiss parameters
- Phase space evolution and emittance
- Chromaticity and momentum compaction

Key Parameters (LHC-scale):
- Beam energy: 7 TeV
- Circumference: 26.7 km
- Betatron tune: Q_x ≈ 64.31, Q_y ≈ 59.32
- Beta function: β* = 0.55 m at IP

Benchmark:
- Symplectic transport (det(M) = 1)
- Stable betatron motion
- Liouville theorem (emittance conservation)

Reference: 
- S.Y. Lee, "Accelerator Physics", 3rd Ed.
- H. Wiedemann, "Particle Accelerator Physics"

(c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray


# Physical constants
C_LIGHT = 299792458.0              # m/s
ELECTRON_MASS_EV = 0.511e6         # eV/c²
PROTON_MASS_EV = 938.272e6         # eV/c²


@dataclass
class BeamConfig:
    """Configuration for beam dynamics simulation."""
    
    # Beam parameters
    energy_ev: float = 7e12              # Beam energy (7 TeV)
    mass_ev: float = PROTON_MASS_EV      # Particle mass
    charge: int = 1                       # Particle charge (units of e)
    
    # Lattice parameters (FODO cell)
    cell_length: float = 100.0           # FODO cell length [m]
    quad_length: float = 2.0             # Quadrupole length [m]
    quad_gradient: float = 100.0         # Quadrupole gradient [T/m]
    n_cells: int = 100                   # Number of FODO cells
    
    # Initial conditions
    x0: float = 1e-3                     # Initial x offset [m]
    xp0: float = 0.0                     # Initial x' angle [rad]
    y0: float = 0.5e-3                   # Initial y offset [m]
    yp0: float = 0.0                     # Initial y' angle [rad]
    
    # Tracking
    n_turns: int = 1000                  # Turns to track
    n_particles: int = 1000              # Particles in beam


@dataclass
class BeamResult:
    """Results from beam dynamics simulation."""
    
    # Twiss parameters
    beta_x: float                        # Horizontal beta function [m]
    beta_y: float                        # Vertical beta function [m]
    alpha_x: float                       # Horizontal alpha
    alpha_y: float                       # Vertical alpha
    gamma_x: float                       # Horizontal gamma [1/m]
    gamma_y: float                       # Vertical gamma [1/m]
    
    # Tune
    tune_x: float                        # Horizontal betatron tune
    tune_y: float                        # Vertical betatron tune
    
    # Phase space
    x_trajectory: NDArray[np.float64]    # x vs turn
    xp_trajectory: NDArray[np.float64]   # x' vs turn
    
    # Validation
    symplectic_error: float              # |det(M) - 1|
    emittance_initial: float             # Initial emittance [m·rad]
    emittance_final: float               # Final emittance [m·rad]
    emittance_conservation: float        # Relative change
    
    # Stability
    is_stable: bool


class TransferMatrix:
    """
    Linear transfer matrices for accelerator elements.
    
    Each matrix M transforms phase space coordinates:
    [x, x'] → M @ [x, x']
    
    Symplectic condition: det(M) = 1
    """
    
    @staticmethod
    def drift(L: float) -> NDArray[np.float64]:
        """Drift space of length L."""
        return np.array([
            [1.0, L],
            [0.0, 1.0]
        ])
    
    @staticmethod
    def thin_quad(f: float) -> NDArray[np.float64]:
        """
        Thin lens quadrupole with focal length f.
        f > 0: focusing, f < 0: defocusing
        """
        return np.array([
            [1.0, 0.0],
            [-1.0/f, 1.0]
        ])
    
    @staticmethod
    def thick_quad_focusing(L: float, k: float) -> NDArray[np.float64]:
        """
        Thick focusing quadrupole.
        k = |K1| where K1 = (1/Bρ) * dB_y/dx
        """
        if k <= 0:
            return TransferMatrix.drift(L)
        
        sqrt_k = math.sqrt(k)
        phi = sqrt_k * L
        
        return np.array([
            [math.cos(phi), math.sin(phi) / sqrt_k],
            [-sqrt_k * math.sin(phi), math.cos(phi)]
        ])
    
    @staticmethod
    def thick_quad_defocusing(L: float, k: float) -> NDArray[np.float64]:
        """
        Thick defocusing quadrupole.
        """
        if k <= 0:
            return TransferMatrix.drift(L)
        
        sqrt_k = math.sqrt(k)
        phi = sqrt_k * L
        
        return np.array([
            [math.cosh(phi), math.sinh(phi) / sqrt_k],
            [sqrt_k * math.sinh(phi), math.cosh(phi)]
        ])
    
    @staticmethod
    def fodo_cell(L_cell: float, L_quad: float, f: float) -> Tuple[NDArray, NDArray]:
        """
        FODO cell: F(ocus) - O(drift) - D(efocus) - O(drift)
        
        Returns (M_x, M_y) for horizontal and vertical planes.
        """
        L_drift = (L_cell - 2 * L_quad) / 2
        
        # Horizontal: QF - drift - QD - drift
        # Use thin lens approximation for simplicity
        QF_x = TransferMatrix.thin_quad(f)      # Focusing in x
        QD_x = TransferMatrix.thin_quad(-f)     # Defocusing in x
        D = TransferMatrix.drift(L_drift)
        
        # Full cell: QF/2 - D - QD - D - QF/2
        # Simplified FODO: QF - D - QD - D
        M_x = D @ QD_x @ D @ QF_x
        
        # Vertical: opposite focusing
        QF_y = TransferMatrix.thin_quad(-f)     # Defocusing in y
        QD_y = TransferMatrix.thin_quad(f)      # Focusing in y
        M_y = D @ QD_y @ D @ QF_y
        
        return M_x, M_y


class BeamDynamics:
    """
    Beam dynamics simulation using transfer matrix formalism.
    
    Tracks particles through lattice and computes Twiss parameters.
    """
    
    def __init__(self, cfg: BeamConfig):
        self.cfg = cfg
        
        # Compute beam rigidity: Bρ = p/q
        gamma = cfg.energy_ev / cfg.mass_ev
        beta = math.sqrt(1 - 1/gamma**2)
        momentum_ev = cfg.energy_ev * beta  # p = γmv ≈ E for relativistic
        
        # Bρ in T·m: (p in eV/c) / (c in m/s) / (q in C)
        # Bρ = p / (q * c) where p in eV, q in units of e
        self.B_rho = momentum_ev / (C_LIGHT * cfg.charge)  # T·m
        
        # Quadrupole focal length: f = Bρ / (G * L_q)
        self.focal_length = self.B_rho / (cfg.quad_gradient * cfg.quad_length)
        
        # Build one-turn map
        self._build_lattice()
        
    def _build_lattice(self) -> None:
        """Build the one-cell transfer matrix."""
        cfg = self.cfg
        
        # FODO cell matrices
        self.M_x, self.M_y = TransferMatrix.fodo_cell(
            cfg.cell_length, 
            cfg.quad_length,
            self.focal_length
        )
        
        # One-turn map (n_cells FODO cells)
        self.M_turn_x = np.linalg.matrix_power(self.M_x, cfg.n_cells)
        self.M_turn_y = np.linalg.matrix_power(self.M_y, cfg.n_cells)
        
    def compute_twiss(self, M: NDArray[np.float64]) -> dict:
        """
        Compute Twiss parameters from transfer matrix.
        
        For stable motion: |Tr(M)| < 2
        
        Phase advance: cos(μ) = Tr(M)/2
        
        Beta function: β = |M12| / sin(μ)
        Alpha: α = (M11 - M22) / (2 sin(μ))
        Gamma: γ = (1 + α²) / β
        """
        trace = M[0, 0] + M[1, 1]
        
        # Check stability
        is_stable = abs(trace) < 2.0
        
        if not is_stable:
            return {
                'beta': float('inf'),
                'alpha': 0.0,
                'gamma': 0.0,
                'tune': 0.0,
                'is_stable': False
            }
        
        # Phase advance
        cos_mu = trace / 2.0
        cos_mu = max(-1.0, min(1.0, cos_mu))  # Clamp for numerical safety
        mu = math.acos(cos_mu)
        
        # Handle sign of sin(μ)
        sin_mu = math.sqrt(1 - cos_mu**2)
        if M[0, 1] < 0:
            sin_mu = -sin_mu
            mu = 2 * math.pi - mu
        
        # Twiss parameters
        if abs(sin_mu) > 1e-10:
            beta = abs(M[0, 1]) / sin_mu
            alpha = (M[0, 0] - M[1, 1]) / (2 * sin_mu)
        else:
            beta = abs(M[0, 1]) if M[0, 1] != 0 else 1.0
            alpha = 0.0
        
        gamma = (1 + alpha**2) / beta if beta > 0 else 0.0
        
        # Tune (phase advance per turn / 2π)
        tune = mu / (2 * math.pi)
        
        return {
            'beta': beta,
            'alpha': alpha,
            'gamma': gamma,
            'tune': tune,
            'mu': mu,
            'is_stable': is_stable
        }
    
    def track_particle(self, x0: float, xp0: float, M: NDArray[np.float64], 
                       n_turns: int) -> Tuple[NDArray, NDArray]:
        """Track a single particle for n_turns."""
        x = np.zeros(n_turns + 1)
        xp = np.zeros(n_turns + 1)
        
        x[0] = x0
        xp[0] = xp0
        
        state = np.array([x0, xp0])
        
        for i in range(n_turns):
            state = M @ state
            x[i + 1] = state[0]
            xp[i + 1] = state[1]
        
        return x, xp
    
    def compute_emittance(self, x: NDArray, xp: NDArray, 
                          twiss: dict) -> float:
        """
        Compute RMS emittance from phase space distribution.
        
        Normalized emittance uses Twiss parameters:
        ε = √(<x²><x'²> - <xx'>²)
        
        Or with Twiss: ε = (γ<x²> + 2α<xx'> + β<x'²>)
        """
        # Geometric emittance
        x_mean = np.mean(x)
        xp_mean = np.mean(xp)
        
        x_centered = x - x_mean
        xp_centered = xp - xp_mean
        
        sigma_xx = np.mean(x_centered**2)
        sigma_xpxp = np.mean(xp_centered**2)
        sigma_xxp = np.mean(x_centered * xp_centered)
        
        # RMS emittance
        det = sigma_xx * sigma_xpxp - sigma_xxp**2
        if det > 0:
            emittance = math.sqrt(det)
        else:
            emittance = 0.0
        
        return emittance
    
    def run(self) -> BeamResult:
        """Run beam dynamics simulation."""
        cfg = self.cfg
        
        # Compute Twiss for one cell
        twiss_x_cell = self.compute_twiss(self.M_x)
        twiss_y_cell = self.compute_twiss(self.M_y)
        
        # Compute Twiss for one turn
        twiss_x = self.compute_twiss(self.M_turn_x)
        twiss_y = self.compute_twiss(self.M_turn_y)
        
        # Track particle
        x_traj, xp_traj = self.track_particle(
            cfg.x0, cfg.xp0, self.M_turn_x, cfg.n_turns
        )
        y_traj, yp_traj = self.track_particle(
            cfg.y0, cfg.yp0, self.M_turn_y, cfg.n_turns
        )
        
        # Symplectic check
        det_x = np.linalg.det(self.M_turn_x)
        det_y = np.linalg.det(self.M_turn_y)
        symplectic_error = max(abs(det_x - 1), abs(det_y - 1))
        
        # Emittance (using single particle as proxy)
        eps_initial = self.compute_emittance(
            x_traj[:100], xp_traj[:100], twiss_x
        )
        eps_final = self.compute_emittance(
            x_traj[-100:], xp_traj[-100:], twiss_x
        )
        
        if eps_initial > 0:
            eps_conservation = abs(eps_final - eps_initial) / eps_initial
        else:
            eps_conservation = 0.0
        
        # Stability check
        is_stable = twiss_x['is_stable'] and twiss_y['is_stable']
        
        # Tune per cell → tune per turn
        tune_x_turn = twiss_x['tune'] * cfg.n_cells
        tune_y_turn = twiss_y['tune'] * cfg.n_cells
        
        # Get fractional tune
        tune_x_frac = tune_x_turn % 1.0
        tune_y_frac = tune_y_turn % 1.0
        
        return BeamResult(
            beta_x=twiss_x_cell['beta'] if twiss_x_cell['is_stable'] else float('inf'),
            beta_y=twiss_y_cell['beta'] if twiss_y_cell['is_stable'] else float('inf'),
            alpha_x=twiss_x_cell['alpha'],
            alpha_y=twiss_y_cell['alpha'],
            gamma_x=twiss_x_cell['gamma'],
            gamma_y=twiss_y_cell['gamma'],
            tune_x=tune_x_turn,
            tune_y=tune_y_turn,
            x_trajectory=x_traj,
            xp_trajectory=xp_traj,
            symplectic_error=symplectic_error,
            emittance_initial=eps_initial,
            emittance_final=eps_final,
            emittance_conservation=eps_conservation,
            is_stable=is_stable
        )


def validate_beam_dynamics(result: BeamResult, cfg: BeamConfig) -> dict:
    """Validate beam dynamics against accelerator physics."""
    checks = {}
    
    # 1. Symplectic condition (det(M) = 1)
    symplectic_valid = result.symplectic_error < 1e-10
    checks['symplectic'] = {
        'valid': symplectic_valid,
        'error': result.symplectic_error,
        'requirement': 'det(M) = 1 for Hamiltonian system'
    }
    
    # 2. Stability (|Tr(M)| < 2)
    checks['stability'] = {
        'valid': result.is_stable,
        'note': 'Betatron oscillations bounded'
    }
    
    # 3. Beta function (should be positive and finite)
    beta_valid = (0 < result.beta_x < 1e6) and (0 < result.beta_y < 1e6)
    checks['beta_function'] = {
        'valid': beta_valid,
        'beta_x_m': result.beta_x,
        'beta_y_m': result.beta_y
    }
    
    # 4. Tune (should be non-integer for stability)
    tune_x_frac = result.tune_x % 1.0
    tune_y_frac = result.tune_y % 1.0
    # Allow tunes close to 0 or 1 as long as motion is stable
    tune_valid = result.is_stable and (tune_x_frac > 0.01) and (tune_y_frac > 0.01)
    checks['tune'] = {
        'valid': tune_valid,
        'tune_x': result.tune_x,
        'tune_y': result.tune_y,
        'tune_x_frac': tune_x_frac,
        'tune_y_frac': tune_y_frac,
        'note': 'Non-integer tune avoids resonances'
    }
    
    # 5. Emittance conservation (Liouville theorem)
    emittance_valid = result.emittance_conservation < 0.1 or result.emittance_initial == 0
    checks['emittance_conservation'] = {
        'valid': emittance_valid,
        'initial_m_rad': result.emittance_initial,
        'final_m_rad': result.emittance_final,
        'change': result.emittance_conservation,
        'note': "Liouville's theorem: phase space volume conserved"
    }
    
    # 6. Bounded motion
    x_bounded = np.max(np.abs(result.x_trajectory)) < 1.0  # Less than 1 m
    checks['bounded_motion'] = {
        'valid': x_bounded,
        'max_amplitude_mm': np.max(np.abs(result.x_trajectory)) * 1000
    }
    
    checks['all_pass'] = all(c['valid'] for c in checks.values())
    
    return checks


def run_beam_dynamics_benchmark() -> Tuple[BeamResult, dict]:
    """Run beam dynamics benchmark."""
    print("="*70)
    print("FRONTIER 04: Beam Dynamics (Transfer Matrix Formalism)")
    print("="*70)
    print()
    
    # LHC-inspired parameters
    cfg = BeamConfig(
        energy_ev=7e12,              # 7 TeV
        mass_ev=PROTON_MASS_EV,
        cell_length=100.0,           # 100 m FODO cell
        quad_length=2.0,             # 2 m quads
        quad_gradient=100.0,         # 100 T/m
        n_cells=100,                 # 100 cells ≈ 10 km
        x0=1e-3,                     # 1 mm initial offset
        xp0=0.0,
        y0=0.5e-3,
        yp0=0.0,
        n_turns=1000
    )
    
    print(f"Configuration (LHC-scale):")
    print(f"  Beam energy:      {cfg.energy_ev/1e12:.0f} TeV")
    print(f"  FODO cell length: {cfg.cell_length:.0f} m")
    print(f"  Number of cells:  {cfg.n_cells}")
    print(f"  Quad gradient:    {cfg.quad_gradient:.0f} T/m")
    print(f"  Tracking turns:   {cfg.n_turns}")
    print()
    
    # Run simulation
    print("Running beam dynamics simulation...")
    sim = BeamDynamics(cfg)
    
    print(f"  Focal length:     {sim.focal_length:.2f} m")
    print(f"  Beam rigidity:    {sim.B_rho:.2f} T·m")
    print()
    
    result = sim.run()
    
    # Display results
    print("Results:")
    print(f"  Beta functions:   β_x = {result.beta_x:.2f} m, β_y = {result.beta_y:.2f} m")
    print(f"  Alpha functions:  α_x = {result.alpha_x:.3f}, α_y = {result.alpha_y:.3f}")
    print(f"  Betatron tunes:   Q_x = {result.tune_x:.4f}, Q_y = {result.tune_y:.4f}")
    print(f"  Symplectic error: {result.symplectic_error:.2e}")
    print(f"  Stable motion:    {result.is_stable}")
    print(f"  Max amplitude:    {np.max(np.abs(result.x_trajectory))*1000:.3f} mm")
    print()
    
    # Validate
    checks = validate_beam_dynamics(result, cfg)
    
    print("Validation:")
    print(f"  Symplectic:            {'✓ PASS' if checks['symplectic']['valid'] else '✗ FAIL'}")
    print(f"  Stability:             {'✓ PASS' if checks['stability']['valid'] else '✗ FAIL'}")
    print(f"  Beta function:         {'✓ PASS' if checks['beta_function']['valid'] else '✗ FAIL'}")
    print(f"  Tune (non-integer):    {'✓ PASS' if checks['tune']['valid'] else '✗ FAIL'}")
    print(f"  Emittance conserved:   {'✓ PASS' if checks['emittance_conservation']['valid'] else '✗ FAIL'}")
    print(f"  Bounded motion:        {'✓ PASS' if checks['bounded_motion']['valid'] else '✗ FAIL'}")
    print()
    
    if checks['all_pass']:
        print("★ BEAM DYNAMICS BENCHMARK: PASS")
    else:
        print("✗ BEAM DYNAMICS BENCHMARK: FAIL")
    
    return result, checks


if __name__ == "__main__":
    result, checks = run_beam_dynamics_benchmark()
