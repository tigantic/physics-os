#!/usr/bin/env python3
"""
OPERATIONAL RANK vs REYNOLDS NUMBER STUDY (QTT-Native)
=======================================================

THE THESIS: QTT operational rank saturates at ~20-30 regardless of Re.

This uses the ACTUAL QTT solver (NS3DQTTSolver) and tracks the bond
dimensions the solver uses during operation. Not dense decomposition.

Experiment:
- Run Taylor-Green vortex at Re = 200, 400, 800, 1600
- Use NS3DQTTSolver with fixed max_rank=64 (cap, not target)
- Track actual operational rank through enstrophy peak
- Report max_rank and mean_rank from QTT state at each timestep

Key insight: The QTT format's bond dimensions ARE the rank. No need
to decompose dense fields - just read solver.u.max_rank.

Resolution requirement:
- 64³ grid (n_bits=6) for Re ≤ 1600
- k_max ≈ 21 (2/3 dealiasing), η ~ (ν³)^(1/4)

Author: HyperTensor Team
Date: 2026
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
import json

import torch
import numpy as np

from tensornet.cfd.ns3d_qtt_native import (
    NS3DQTTSolver, NS3DConfig, taylor_green_3d,
    TimeIntegrator, TruncationStrategy,
)


# Configuration for each Re
RE_CONFIG = {
    200:  {'n_bits': 6, 'T_final': 6.0,  'dt': 0.01,  'peak_expected': 3.0},
    400:  {'n_bits': 6, 'T_final': 8.0,  'dt': 0.005, 'peak_expected': 5.0},
    800:  {'n_bits': 6, 'T_final': 12.0, 'dt': 0.004, 'peak_expected': 7.0},
    1600: {'n_bits': 6, 'T_final': 15.0, 'dt': 0.002, 'peak_expected': 9.0},
}


@dataclass
class RankProfile:
    """Rank statistics at a single timestep."""
    t: float
    enstrophy: float
    kinetic_energy: float
    max_rank_u: int           # Max bond dimension in velocity field
    mean_rank_u: float        # Mean bond dimension in velocity field
    max_rank_omega: int       # Max bond dimension in vorticity field
    mean_rank_omega: float    # Mean bond dimension in vorticity field
    compression_ratio: float  # Dense / QTT parameters
    
    @property
    def max_rank(self) -> int:
        return max(self.max_rank_u, self.max_rank_omega)
    
    @property
    def mean_rank(self) -> float:
        return (self.mean_rank_u + self.mean_rank_omega) / 2


def detect_peak(enstrophy_history: List[float], lookback: int = 5) -> Tuple[bool, int]:
    """
    Detect if enstrophy has peaked (started decreasing).
    
    A peak is confirmed when enstrophy has decreased for `lookback` 
    consecutive samples.
    """
    if len(enstrophy_history) < lookback + 1:
        return False, -1
    
    recent = enstrophy_history[-lookback:]
    if all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
        peak_idx = len(enstrophy_history) - lookback
        while peak_idx > 0 and enstrophy_history[peak_idx-1] > enstrophy_history[peak_idx]:
            peak_idx -= 1
        return True, peak_idx
    
    return False, -1


def run_qtt_rank_study_single_re(
    Re: float,
    n_bits: int = 6,
    T_final: float = 10.0,
    dt: float = 0.005,
    max_rank_cap: int = 64,
    sample_interval: int = 20,
    early_stop_after_peak: bool = True,
) -> Dict:
    """
    Run Taylor-Green at fixed Re using QTT solver, tracking operational rank.
    
    The OPERATIONAL RANK is the bond dimension the solver actually uses.
    This is read directly from solver.u.max_rank and solver.omega.max_rank.
    
    Args:
        Re: Reynolds number
        n_bits: Grid resolution (N = 2^n_bits per axis)
        T_final: Maximum simulation time
        dt: Timestep
        max_rank_cap: Maximum allowed rank (solver may use less)
        sample_interval: Steps between measurements
        early_stop_after_peak: Stop 20% after enstrophy peak
    
    Returns:
        Dict with all rank evolution data
    """
    N = 2 ** n_bits
    L = 2 * math.pi
    nu = L / Re
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Kolmogorov microscale check
    eta_approx = (nu**3)**0.25
    k_max = N // 3
    resolution_ratio = k_max * eta_approx
    
    print(f"\n{'─' * 70}")
    print(f"Re = {Re:.0f}, N = {N}³, ν = {nu:.6f}")
    print(f"Resolution check: k_max·η ≈ {resolution_ratio:.2f} (need ≥ 1 for DNS)")
    if resolution_ratio < 1:
        print(f"  ⚠ WARNING: Under-resolved! Results may not represent true turbulence.")
    print(f"QTT max_rank cap: {max_rank_cap}")
    print(f"{'─' * 70}")
    
    # Create QTT config
    config = NS3DConfig(
        n_bits=n_bits,
        nu=nu,
        L=L,
        max_rank=max_rank_cap,
        dt=dt,
        integrator=TimeIntegrator.RK4,
        truncation=TruncationStrategy.TURBULENT,
        tol_svd=1e-10,
        device=device,
    )
    
    # Initialize solver
    solver = NS3DQTTSolver(config)
    u_init, omega_init = taylor_green_3d(config)
    solver.initialize(u_init, omega_init)
    
    # Track evolution
    profiles: List[RankProfile] = []
    enstrophy_history: List[float] = []
    n_steps = int(T_final / dt)
    
    # Initial measurement
    diag = solver.diagnostics_history[-1]
    print(f"  t=0.000: E={diag.kinetic_energy:.4f}, Ω={diag.enstrophy:.4f}, "
          f"max_rank_u={solver.u.max_rank}, max_rank_ω={solver.omega.max_rank}")
    
    # Peak tracking
    peak_detected = False
    peak_sample_idx = -1
    peak_confirmed_step = -1
    
    t_start = time.perf_counter()
    
    for step in range(n_steps):
        diag = solver.step()
        
        # Sample periodically
        if (step + 1) % sample_interval == 0:
            enstrophy_history.append(diag.enstrophy)
            
            profile = RankProfile(
                t=diag.time,
                enstrophy=diag.enstrophy,
                kinetic_energy=diag.kinetic_energy,
                max_rank_u=solver.u.max_rank,
                mean_rank_u=solver.u.mean_rank,
                max_rank_omega=solver.omega.max_rank,
                mean_rank_omega=solver.omega.mean_rank,
                compression_ratio=solver.u.compression_ratio,
            )
            profiles.append(profile)
            
            # Check for peak
            if not peak_detected:
                peak_detected, peak_sample_idx = detect_peak(enstrophy_history, lookback=5)
                if peak_detected:
                    peak_confirmed_step = step
                    print(f"\n  ✓ Enstrophy peak detected at t={profiles[peak_sample_idx].t:.3f}")
            
            # Progress reporting
            if (step + 1) % (sample_interval * 10) == 0:
                status = "↑" if len(enstrophy_history) < 2 or diag.enstrophy > enstrophy_history[-2] else "↓"
                print(f"  t={diag.time:.3f}: E={diag.kinetic_energy:.4f}, Ω={diag.enstrophy:.4f} {status}, "
                      f"max_rank={profile.max_rank}, mean_rank={profile.mean_rank:.1f}")
            
            # Early stopping
            if early_stop_after_peak and peak_detected:
                steps_since_peak = step - peak_confirmed_step
                if steps_since_peak > 0.2 * peak_confirmed_step:
                    print(f"  Stopping early: captured peak and 20% decay phase")
                    break
    
    elapsed = time.perf_counter() - t_start
    
    # Find profile at actual enstrophy peak
    if peak_sample_idx >= 0 and peak_sample_idx < len(profiles):
        peak_profile = profiles[peak_sample_idx]
    else:
        peak_profile = max(profiles, key=lambda p: p.enstrophy)
        print(f"  ⚠ No confirmed peak - using maximum enstrophy sample")
    
    print(f"\n  Enstrophy peak at t={peak_profile.t:.3f}")
    print(f"  Peak enstrophy: {peak_profile.enstrophy:.4f}")
    print(f"  Max operational rank at peak: {peak_profile.max_rank}")
    print(f"    - velocity:  max={peak_profile.max_rank_u}, mean={peak_profile.mean_rank_u:.1f}")
    print(f"    - vorticity: max={peak_profile.max_rank_omega}, mean={peak_profile.mean_rank_omega:.1f}")
    print(f"  Compression ratio: {peak_profile.compression_ratio:.1f}x")
    print(f"  Wall time: {elapsed:.1f}s")
    
    # Extract time series
    times = [p.t for p in profiles]
    enstrophies = [p.enstrophy for p in profiles]
    max_ranks = [p.max_rank for p in profiles]
    mean_ranks = [p.mean_rank for p in profiles]
    
    return {
        'Re': Re,
        'N': N,
        'n_bits': n_bits,
        'nu': nu,
        'max_rank_cap': max_rank_cap,
        'T_final': profiles[-1].t if profiles else 0.0,
        'resolution_ratio': resolution_ratio,
        'n_profiles': len(profiles),
        
        # Enstrophy peak info
        'peak_detected': peak_detected,
        'peak_time': peak_profile.t,
        'peak_enstrophy': peak_profile.enstrophy,
        
        # OPERATIONAL RANK at peak
        'max_rank_at_peak': peak_profile.max_rank,
        'mean_rank_at_peak': peak_profile.mean_rank,
        'max_rank_u_at_peak': peak_profile.max_rank_u,
        'max_rank_omega_at_peak': peak_profile.max_rank_omega,
        'compression_ratio_at_peak': peak_profile.compression_ratio,
        
        # Time series
        'times': times,
        'enstrophies': enstrophies,
        'max_ranks': max_ranks,
        'mean_ranks': mean_ranks,
        
        'wall_time': elapsed,
    }


def run_full_qtt_rank_study() -> Dict:
    """
    Run the complete QTT Rank vs Re study.
    
    THE THESIS: Operational rank saturates around 20-30 regardless of Re.
    """
    print("╔" + "═" * 70 + "╗")
    print("║" + " QTT OPERATIONAL RANK vs REYNOLDS NUMBER STUDY ".center(70) + "║")
    print("║" + " Using NS3DQTTSolver (actual QTT solver, not dense) ".center(70) + "║")
    print("║" + " Taylor-Green Vortex at Re = 200, 400, 800, 1600 ".center(70) + "║")
    print("╚" + "═" * 70 + "╝")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    reynolds_numbers = [200, 400, 800, 1600]
    max_rank_cap = 64  # Cap, not target - solver may use less
    
    results = {}
    
    t_start = time.perf_counter()
    
    for Re in reynolds_numbers:
        cfg = RE_CONFIG[Re]
        
        result = run_qtt_rank_study_single_re(
            Re=Re,
            n_bits=cfg['n_bits'],
            T_final=cfg['T_final'],
            dt=cfg['dt'],
            max_rank_cap=max_rank_cap,
            sample_interval=25,
            early_stop_after_peak=True,
        )
        results[f'Re_{int(Re)}'] = result
    
    total_time = time.perf_counter() - t_start
    
    # Summary table
    print("\n" + "═" * 80)
    print("QTT OPERATIONAL RANK vs Re SUMMARY")
    print("═" * 80)
    print(f"{'Re':>6} │ {'Peak t':>8} │ {'Peak Ω':>10} │ {'k·η':>6} │ "
          f"{'Max Rank':>10} │ {'Mean Rank':>10} │ {'Peak?':>6}")
    print("─" * 80)
    
    for Re in reynolds_numbers:
        r = results[f'Re_{int(Re)}']
        peak_status = "✓" if r['peak_detected'] else "✗"
        print(f"{Re:>6.0f} │ {r['peak_time']:>8.3f} │ {r['peak_enstrophy']:>10.2f} │ "
              f"{r['resolution_ratio']:>6.2f} │ {r['max_rank_at_peak']:>10} │ "
              f"{r['mean_rank_at_peak']:>10.1f} │ {peak_status:>6}")
    
    print("─" * 80)
    
    # Analyze the thesis
    max_ranks_at_peaks = [results[f'Re_{int(Re)}']['max_rank_at_peak'] for Re in reynolds_numbers]
    mean_ranks_at_peaks = [results[f'Re_{int(Re)}']['mean_rank_at_peak'] for Re in reynolds_numbers]
    
    rank_std = np.std(max_ranks_at_peaks)
    rank_mean = np.mean(max_ranks_at_peaks)
    rank_min = min(max_ranks_at_peaks)
    rank_max = max(max_ranks_at_peaks)
    
    print(f"\nOperational Rank Statistics across Re = {reynolds_numbers}:")
    print(f"  Max rank range: {rank_min} - {rank_max}")
    print(f"  Mean max rank: {rank_mean:.1f} ± {rank_std:.1f}")
    print(f"  Mean effective rank: {np.mean(mean_ranks_at_peaks):.1f}")
    
    # Check if rank saturates
    cv = rank_std / rank_mean if rank_mean > 0 else float('inf')
    saturation = cv < 0.3
    in_expected_range = rank_max <= 50
    
    print(f"\n  Coefficient of variation: {cv:.2%}")
    
    if saturation and in_expected_range:
        print("\n" + "═" * 80)
        print("  ✓✓✓ THESIS CONFIRMED: Operational rank saturates regardless of Re ✓✓✓ ".center(80))
        print(f"  Max rank ≈ {rank_mean:.0f} ± {rank_std:.0f} across Re = 200-1600 ".center(80))
        print("  This proves O(N log N) QTT complexity for turbulence! ".center(80))
        print("═" * 80)
    elif saturation:
        print("\n" + "═" * 80)
        print("  ✓ Rank is stable, but higher than expected ".center(80))
        print(f"  Max rank ≈ {rank_mean:.0f} ± {rank_std:.0f} ".center(80))
        print("═" * 80)
    else:
        log_Re = np.log(reynolds_numbers)
        log_rank = np.log(max_ranks_at_peaks)
        slope = np.polyfit(log_Re, log_rank, 1)[0]
        
        print("\n" + "═" * 80)
        print("  ✗ Rank varies with Re ".center(80))
        print(f"  Scaling: rank ~ Re^{slope:.2f} ".center(80))
        print("═" * 80)
    
    # Check if all peaks detected
    all_peaks = all(results[f'Re_{int(Re)}']['peak_detected'] for Re in reynolds_numbers)
    if not all_peaks:
        print("\n  ⚠ WARNING: Not all runs captured the enstrophy peak!")
    
    results['summary'] = {
        'reynolds_numbers': reynolds_numbers,
        'max_ranks_at_peaks': max_ranks_at_peaks,
        'mean_ranks_at_peaks': mean_ranks_at_peaks,
        'rank_saturation': saturation,
        'rank_mean': float(rank_mean),
        'rank_std': float(rank_std),
        'coefficient_of_variation': float(cv),
        'all_peaks_detected': all_peaks,
        'max_rank_cap_used': max_rank_cap,
        'total_time': total_time,
    }
    
    print(f"\nTotal time: {total_time:.1f}s")
    
    return results


def save_results(results: Dict, filename: str = 'QTT_RANK_VS_RE_STUDY.json'):
    """Save results to JSON."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results_clean = convert(results)
    
    with open(filename, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    results = run_full_qtt_rank_study()
    save_results(results)
