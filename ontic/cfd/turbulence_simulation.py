#!/usr/bin/env python3
"""
Turbulence Simulation with Kolmogorov Spectrum Verification
============================================================

Runs the QTT Navier-Stokes solver at high Reynolds number with
large-scale forcing to achieve stationary homogeneous isotropic
turbulence (HIT), then verifies the k^(-5/3) Kolmogorov energy
spectrum in the inertial range.

Physics:
    ∂ω/∂t = (ω·∇)u - (u·∇)ω + ν∇²ω + f

where f is the large-scale forcing that injects energy at low
wavenumbers, maintaining the turbulent cascade.

Author: TiganticLabz
Date: 2026
"""

import os
import sys
import time
import math
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ontic.cfd.ns3d_turbo import TurboNS3DConfig, TurboNS3DSolver
from ontic.cfd.qtt_turbo import turbo_inner
from ontic.cfd.turbulence_forcing import (
    TurbulenceStats,
    compute_turbulence_stats,
    estimate_dissipation_rate,
    compute_taylor_reynolds,
    compute_kolmogorov_scales,
)
from ontic.cfd.kolmogorov_spectrum import fit_power_law


def compute_spectrum_qtt_native(
    omega: List[List[torch.Tensor]],
    n_bits: int,
    nu: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute energy spectrum estimate directly from QTT cores.
    
    Uses the QTT structure to estimate spectrum without dense conversion.
    The singular values of QTT cores encode the energy at different scales
    due to the hierarchical (Morton) structure.
    
    For a Morton-ordered QTT:
    - Cores 0,1,2 encode k ~ N/2 (finest scale)
    - Cores 3,4,5 encode k ~ N/4
    - etc.
    
    We estimate E(k) from the singular value distribution at each scale.
    
    Returns (k_bins, E_k_estimate, total_enstrophy)
    """
    from ontic.cfd.qtt_turbo import turbo_inner
    
    # Total enstrophy for normalization
    enstrophy = sum(turbo_inner(w, w).item() for w in omega)
    
    N = 2 ** n_bits
    k_max = N // 2
    k_bins = np.arange(k_max + 1, dtype=np.float64)
    E_k = np.zeros(k_max + 1)
    
    # Extract energy at each scale from core singular values
    # Morton interleaving: cores [0,1,2] = finest xyz bits, [3,4,5] = next, etc.
    for component in omega:
        cores = component
        n_cores = len(cores)
        
        for level in range(n_bits):
            # Each level corresponds to wavenumber k ~ 2^(n_bits - level - 1)
            k_level = 2 ** (n_bits - level - 1)
            k_idx = min(k_level, k_max)
            
            # Sum squared singular values from cores at this level (x,y,z)
            for d in range(3):
                core_idx = 3 * level + d
                if core_idx < n_cores:
                    core = cores[core_idx]
                    # Singular values of unfolded core estimate energy at this scale
                    r_l, _, r_r = core.shape
                    unfolded = core.reshape(r_l * 2, r_r)
                    try:
                        S = torch.linalg.svdvals(unfolded)
                        energy_at_scale = (S ** 2).sum().item()
                        E_k[k_idx] += energy_at_scale / 3.0  # Divide by 3 components
                    except:
                        pass
    
    # Normalize so integral matches enstrophy
    if E_k.sum() > 0:
        E_k *= enstrophy / E_k.sum()
    
    # Convert enstrophy spectrum to energy spectrum: E(k) ~ Ω(k) / k²
    k_safe = np.maximum(k_bins, 1e-10)
    E_k_energy = E_k / (k_safe ** 2)
    E_k_energy[0] = 0  # Zero mode
    
    return k_bins, E_k_energy, enstrophy


@dataclass
class TurbulenceRunConfig:
    """Configuration for turbulence simulation."""
    # Grid and physics
    n_bits: int = 6              # Grid: N = 2^n_bits (6 = 64³)
    Re: float = 1000.0           # Reynolds number
    dt: float = 0.001            # Timestep
    
    # Simulation duration
    n_spinup_steps: int = 500    # Spinup to reach stationary state
    n_analysis_steps: int = 100  # Steps for time-averaged statistics
    
    # Forcing
    forcing_epsilon: float = 0.1  # Energy injection rate
    forcing_k: int = 2            # Forcing wavenumber
    
    # Adaptive rank
    target_error: float = 1e-5   # Error budget for rank adaptation
    rank_cap: int = 128          # Maximum rank
    
    # Output
    output_dir: str = 'turbulence_results'
    save_spectra: bool = True
    verbose: bool = True


def run_turbulence_simulation(config: TurbulenceRunConfig) -> Dict:
    """
    Run turbulence simulation and analyze results.
    
    Returns comprehensive results dict.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    N = 2 ** config.n_bits
    nu = 1.0 / config.Re
    
    if config.verbose:
        print("═" * 70)
        print("   QTT TURBULENCE SIMULATION WITH KOLMOGOROV SPECTRUM VERIFICATION")
        print("═" * 70)
        print()
        print(f"Grid:          {N}³ ({N**3:,} cells)")
        print(f"Reynolds:      Re = {config.Re:.0f}")
        print(f"Viscosity:     ν = {nu:.6f}")
        print(f"Timestep:      dt = {config.dt}")
        print(f"Forcing k:     k_f = {config.forcing_k}")
        print(f"Forcing ε:     ε_target = {config.forcing_epsilon}")
        print(f"Target error:  {config.target_error:.0e}")
        print(f"Rank cap:      {config.rank_cap}")
        print()
    
    # Create solver with forcing
    solver_config = TurboNS3DConfig(
        n_bits=config.n_bits,
        nu=nu,
        dt=config.dt,
        adaptive_rank=True,
        target_error=config.target_error,
        min_rank=4,
        rank_cap=config.rank_cap,
        device='cuda',
        diffusion_only=False,
        enable_forcing=True,
        forcing_epsilon=config.forcing_epsilon,
        forcing_k=config.forcing_k,
    )
    
    solver = TurboNS3DSolver(solver_config)
    solver.initialize_taylor_green()
    
    # ════════════════════════════════════════════════════════════════════
    # PHASE 1: SPINUP
    # ════════════════════════════════════════════════════════════════════
    if config.verbose:
        print("PHASE 1: SPINUP TO STATIONARY TURBULENCE")
        print("-" * 50)
    
    spinup_stats = []
    t_start = time.perf_counter()
    
    for step in range(config.n_spinup_steps):
        diag = solver.step()
        
        if step % 50 == 0 or step == config.n_spinup_steps - 1:
            stats = compute_turbulence_stats(
                solver.omega, solver.u, nu,
                solver.t, diag['step_ms'],
            )
            spinup_stats.append(asdict(stats))
            
            if config.verbose:
                print(f"  Step {step:4d}: t={solver.t:.3f}, "
                      f"Re_λ={stats.taylor_reynolds:.1f}, "
                      f"ε={stats.dissipation_rate:.4f}, "
                      f"rank={stats.max_rank}")
    
    spinup_time = time.perf_counter() - t_start
    
    if config.verbose:
        print(f"\n  Spinup complete in {spinup_time:.1f}s")
        print()
    
    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: STATIONARY STATE ANALYSIS
    # ════════════════════════════════════════════════════════════════════
    if config.verbose:
        print("PHASE 2: STATIONARY STATE STATISTICS")
        print("-" * 50)
    
    analysis_stats = []
    spectra = []
    
    for step in range(config.n_analysis_steps):
        diag = solver.step()
        
        stats = compute_turbulence_stats(
            solver.omega, solver.u, nu,
            solver.t, diag['step_ms'],
        )
        analysis_stats.append(asdict(stats))
        
        # Compute spectrum every 10 steps
        if step % 10 == 0 and config.n_bits <= 7:  # Only for ≤128³
            try:
                k, E_k = compute_spectrum_from_qtt(solver.omega, config.n_bits)
                spectra.append({'step': step, 'k': k.tolist(), 'E_k': E_k.tolist()})
            except Exception as e:
                if config.verbose:
                    print(f"  Warning: Spectrum computation failed at step {step}: {e}")
        
        if step % 20 == 0:
            if config.verbose:
                print(f"  Step {step:4d}: Re_λ={stats.taylor_reynolds:.1f}, "
                      f"ε={stats.dissipation_rate:.4f}")
    
    # ════════════════════════════════════════════════════════════════════
    # PHASE 3: FINAL SPECTRUM ANALYSIS
    # ════════════════════════════════════════════════════════════════════
    if config.verbose:
        print()
        print("PHASE 3: KOLMOGOROV SPECTRUM ANALYSIS")
        print("-" * 50)
    
    # Compute final spectrum
    if config.n_bits <= 7:
        k, E_k = compute_spectrum_from_qtt(solver.omega, config.n_bits)
        spectrum_result = analyze_spectrum(k, E_k, nu)
        
        if config.verbose:
            print(f"  Inertial range: k = [{k[spectrum_result.inertial_range[0]]:.2f}, "
                  f"{k[spectrum_result.inertial_range[1]]:.2f}]")
            print(f"  Fitted exponent: α = {spectrum_result.fitted_exponent:.3f} "
                  f"(theory: -5/3 = -1.667)")
            print(f"  R² of fit: {spectrum_result.r_squared:.4f}")
            print(f"  Kolmogorov length: η = {spectrum_result.kolmogorov_length:.4f}")
            print(f"  Integral length: L = {spectrum_result.integral_length:.4f}")
            
            # Check if we achieved Kolmogorov scaling
            exponent_error = abs(spectrum_result.fitted_exponent - (-5/3))
            if exponent_error < 0.2 and spectrum_result.r_squared > 0.9:
                print()
                print("  ✓ KOLMOGOROV -5/3 LAW VERIFIED!")
            elif exponent_error < 0.4:
                print()
                print("  ≈ Approximate Kolmogorov scaling (limited inertial range)")
            else:
                print()
                print("  ✗ Kolmogorov scaling not achieved (Re or resolution too low)")
    else:
        spectrum_result = None
        if config.verbose:
            print("  (Spectrum computation skipped for large grids)")
    
    # ════════════════════════════════════════════════════════════════════
    # COMPILE RESULTS
    # ════════════════════════════════════════════════════════════════════
    
    # Time-averaged statistics
    avg_stats = {}
    stat_keys = ['kinetic_energy', 'enstrophy', 'dissipation_rate', 
                 'taylor_reynolds', 'max_rank', 'step_time_ms']
    for key in stat_keys:
        values = [s[key] for s in analysis_stats]
        avg_stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    results = {
        'config': asdict(config) if hasattr(config, '__dict__') else str(config),
        'grid_size': N,
        'reynolds_number': config.Re,
        'viscosity': nu,
        'spinup_time_s': spinup_time,
        'spinup_stats': spinup_stats,
        'analysis_stats': analysis_stats,
        'time_averaged': avg_stats,
        'spectra': spectra if config.save_spectra else [],
    }
    
    if spectrum_result is not None:
        results['final_spectrum'] = {
            'fitted_exponent': spectrum_result.fitted_exponent,
            'r_squared': spectrum_result.r_squared,
            'kolmogorov_length': spectrum_result.kolmogorov_length,
            'integral_length': spectrum_result.integral_length,
            'inertial_range_k': [
                float(k[spectrum_result.inertial_range[0]]),
                float(k[spectrum_result.inertial_range[1]]),
            ],
            'kolmogorov_verified': (
                abs(spectrum_result.fitted_exponent - (-5/3)) < 0.2 
                and spectrum_result.r_squared > 0.9
            ),
        }
    
    if config.verbose:
        print()
        print("═" * 70)
        print("SUMMARY")
        print("═" * 70)
        print(f"  Time-averaged Re_λ: {avg_stats['taylor_reynolds']['mean']:.1f} "
              f"± {avg_stats['taylor_reynolds']['std']:.1f}")
        print(f"  Time-averaged ε:    {avg_stats['dissipation_rate']['mean']:.4f}")
        print(f"  Mean step time:     {avg_stats['step_time_ms']['mean']:.0f}ms")
        print(f"  Mean max rank:      {avg_stats['max_rank']['mean']:.1f}")
        if spectrum_result:
            print(f"  Spectrum exponent:  {spectrum_result.fitted_exponent:.3f}")
        print("═" * 70)
    
    return results


def run_reynolds_sweep(Re_values: List[float], n_bits: int = 6) -> List[Dict]:
    """
    Run turbulence simulation at multiple Reynolds numbers.
    """
    results = []
    
    for Re in Re_values:
        print(f"\n{'='*70}")
        print(f"REYNOLDS NUMBER: Re = {Re:.0f}")
        print('='*70)
        
        config = TurbulenceRunConfig(
            n_bits=n_bits,
            Re=Re,
            dt=0.001,
            n_spinup_steps=300,
            n_analysis_steps=50,
            forcing_epsilon=0.1,
            forcing_k=2,
            target_error=1e-5,
            rank_cap=128,
            verbose=True,
        )
        
        result = run_turbulence_simulation(config)
        results.append(result)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='QTT Turbulence Simulation')
    parser.add_argument('--n_bits', type=int, default=6, help='Grid size (N=2^n_bits)')
    parser.add_argument('--Re', type=float, default=1000, help='Reynolds number')
    parser.add_argument('--steps', type=int, default=200, help='Total steps')
    parser.add_argument('--output', type=str, default='turbulence_results.json')
    args = parser.parse_args()
    
    config = TurbulenceRunConfig(
        n_bits=args.n_bits,
        Re=args.Re,
        n_spinup_steps=args.steps // 2,
        n_analysis_steps=args.steps // 2,
        verbose=True,
    )
    
    results = run_turbulence_simulation(config)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {args.output}")
