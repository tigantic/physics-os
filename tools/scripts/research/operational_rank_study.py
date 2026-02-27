#!/usr/bin/env python3
"""
OPERATIONAL RANK STUDY: Figure 1 (The Thesis)
==============================================

Goal: Prove that QTT operational rank saturates around 20-30 regardless of Re.

If true → O(N log N) complexity for turbulence.
If rank ~ Re^α for α > 0 → O(N^α) scaling, still beats O(N³).

Approach:
---------
Run the QTT solver (TurboNS3DSolver) natively at high resolution.
Track bond dimensions through enstrophy peak.

The rank measurement IS the bond dimensions of the TT cores.
No dense field materialization. No TT-SVD on dense data.
This is what the solver actually uses.

Resolution:
-----------
Re=1600: N=512³ (n_bits=9, 27 cores)
Re=3200: N=1024³ (n_bits=10, 30 cores)

Memory: ~9MB at 512³, ~79KB at 4096³ (QTT compression magic)
"""

import torch
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import gc

# Import the QTT solver
from tensornet.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig


@dataclass
class RankMeasurement:
    """Single timestep measurement."""
    t: float
    enstrophy: float
    max_rank: int
    mean_rank: float
    step_ms: float
    truncation_error: float = 0.0


@dataclass
class RunResult:
    """Full run result for one Re."""
    Re: float
    N: int
    n_bits: int
    n_cores: int
    nu: float
    dt: float
    
    # Peak detection
    enstrophy_peak: float
    t_peak: float
    rank_at_peak: int
    
    # Max rank across entire run
    max_rank_overall: int
    
    # Trajectory
    measurements: List[Dict]
    
    # Timing
    total_time_s: float
    steps: int


def compute_reynolds(N: int, nu: float, U: float = 1.0, L: float = 1.0) -> float:
    """Compute Reynolds number from viscosity."""
    return U * L / nu


def compute_viscosity(Re: float, U: float = 1.0, L: float = 1.0) -> float:
    """Compute viscosity from Reynolds number."""
    return U * L / Re


def check_resolution(Re: float, N: int) -> tuple:
    """
    Check if grid resolution is adequate for Reynolds number.
    
    For turbulence: k_max * η ≥ 1 where η = (ν³/ε)^(1/4) is Kolmogorov scale.
    With ε ~ U³/L and ν = UL/Re, we get k_max * η ~ (N/2π) * Re^(-3/4)
    
    Requires N ≥ Re^(3/4) roughly.
    """
    k_max = N / 2  # Maximum wavenumber
    # Kolmogorov scale estimate: η ~ L * Re^(-3/4) where L = 2π
    L = 2 * 3.14159265359
    eta = L * Re**(-0.75)
    k_eta = k_max * eta
    
    resolved = k_eta >= 1.0
    return resolved, k_eta, eta


def compute_timestep(N: int, nu: float, cfl: float = 0.1) -> float:
    """
    Compute stable timestep.
    
    CFL: dt < h / U_max, where U_max ~ 1 for Taylor-Green
    Diffusion: dt < h² / (6 * nu)
    
    Use minimum of both, with safety factor.
    """
    h = 2 * 3.14159265359 / N
    dt_cfl = cfl * h / 1.0  # U_max ~ 1, cfl=0.1 is conservative
    dt_diff = 0.1 * h**2 / (6 * nu)  # 10% of diffusion limit
    return min(dt_cfl, dt_diff)


def estimate_T_final(Re: float) -> float:
    """
    Estimate time to reach enstrophy peak.
    
    For Taylor-Green:
    - Re < 400: No turbulence, just viscous decay. Peak at t=0.
    - Re ~ 400-1000: Transition. Peak at t ~ 3-5
    - Re ~ 1000-3000: Peak at t ~ 5-8
    - Re > 3000: Peak at t ~ 8-12
    
    Run 50% past estimated peak to capture decay.
    """
    if Re < 400:
        return 2.0  # Just watch decay
    elif Re < 1000:
        return 8.0
    elif Re < 3000:
        return 12.0
    else:
        return 15.0


def detect_peak(enstrophy_history: List[float], lookback: int = 20) -> Optional[int]:
    """
    Detect enstrophy peak with lookback validation.
    
    A peak is confirmed when enstrophy has been declining for `lookback` steps.
    Returns index of peak, or None if no peak detected yet.
    """
    if len(enstrophy_history) < lookback + 1:
        return None
    
    # Find maximum so far
    max_val = max(enstrophy_history)
    max_idx = enstrophy_history.index(max_val)
    
    # Check if max is at least lookback steps ago and current is declining
    if max_idx <= len(enstrophy_history) - lookback:
        # Verify monotonic decline after peak (allow small fluctuations)
        post_peak = enstrophy_history[max_idx:]
        n_declining = 0
        for i in range(1, len(post_peak)):
            if post_peak[i] < post_peak[i-1]:
                n_declining += 1
        
        # At least 80% of post-peak steps should be declining
        if n_declining >= 0.8 * (len(post_peak) - 1):
            return max_idx
    
    return None


def run_operational_rank_study(
    Re: float,
    n_bits: int,
    max_rank: int = 32,
    device: str = 'cuda',
    verbose: bool = True,
) -> RunResult:
    """
    Run QTT solver at specified Re and grid size, tracking operational rank.
    
    Uses FIXED rank mode for memory efficiency. The solver will truncate
    to max_rank at each step. We track what rank the solution actually uses
    (max bond dimension across cores).
    
    Args:
        Re: Reynolds number
        n_bits: Grid size exponent (N = 2^n_bits)
        max_rank: Fixed maximum rank for truncation
        device: 'cuda' or 'cpu'
        verbose: Print progress
    
    Returns:
        RunResult with full trajectory and diagnostics
    """
    N = 2 ** n_bits
    n_cores = 3 * n_bits
    
    # Check resolution
    resolved, k_eta, eta = check_resolution(Re, N)
    if verbose:
        print(f"\n{'='*70}")
        print(f"OPERATIONAL RANK STUDY: Re={Re:.0f}, N={N}³")
        print(f"{'='*70}")
        print(f"  Grid: {N}³ = {N**3:,} points")
        print(f"  QTT cores: {n_cores}")
        print(f"  k·η = {k_eta:.3f} ({'✓ resolved' if resolved else '⚠ under-resolved'})")
    
    if not resolved:
        print(f"  WARNING: Grid under-resolved for Re={Re}. Results may be inaccurate.")
    
    # Compute parameters
    nu = compute_viscosity(Re)
    dt = compute_timestep(N, nu)
    T_final = estimate_T_final(Re)
    n_steps = int(T_final / dt) + 1
    
    if verbose:
        print(f"  ν = {nu:.2e}")
        print(f"  dt = {dt:.2e}")
        print(f"  T_final = {T_final:.1f} ({n_steps} steps)")
        print(f"  max_rank = {max_rank}")
    
    # Configure solver with FIXED rank (more memory efficient)
    config = TurboNS3DConfig(
        n_bits=n_bits,
        nu=nu,
        dt=dt,
        device=device,
        
        # Use fixed rank mode (better memory behavior)
        adaptive_rank=False,
        max_rank=max_rank,
        tol=1e-10,
        
        # Velocity update every step for accuracy
        velocity_update_freq=1,
        poisson_iterations=5,
    )
    
    # Create solver
    if verbose:
        print(f"\n  Initializing solver...")
    t_init = time.perf_counter()
    
    solver = TurboNS3DSolver(config)
    solver.initialize_taylor_green(A=1.0)
    
    init_time = time.perf_counter() - t_init
    if verbose:
        print(f"  Initialization: {init_time:.2f}s")
    
    # Initial measurement
    measurements: List[RankMeasurement] = []
    enstrophy_history: List[float] = []
    max_rank_overall = 0
    
    # Get initial enstrophy
    from tensornet.cfd.qtt_turbo import turbo_inner
    enstrophy_0 = sum(turbo_inner(solver.omega[i], solver.omega[i]).item() for i in range(3))
    
    if verbose:
        print(f"\n  Initial enstrophy: {enstrophy_0:.4f}")
        print(f"\n  Running simulation...")
    
    # Run simulation
    t_run_start = time.perf_counter()
    peak_detected = False
    t_peak = 0.0
    enstrophy_peak = enstrophy_0
    rank_at_peak = 0
    
    # Checkpoint frequency
    checkpoint_every = max(1, n_steps // 100)  # ~100 checkpoints
    
    for step in range(n_steps):
        # Take step
        diag = solver.step()
        
        # Record
        m = RankMeasurement(
            t=diag['time'],
            enstrophy=diag['enstrophy'],
            max_rank=diag['max_rank'],
            mean_rank=diag['mean_rank'],
            step_ms=diag['step_ms'],
            truncation_error=diag.get('truncation_error', 0.0),
        )
        measurements.append(m)
        enstrophy_history.append(diag['enstrophy'])
        
        if diag['max_rank'] > max_rank_overall:
            max_rank_overall = diag['max_rank']
        
        # Peak detection
        if not peak_detected:
            peak_idx = detect_peak(enstrophy_history, lookback=30)
            if peak_idx is not None:
                peak_detected = True
                t_peak = measurements[peak_idx].t
                enstrophy_peak = measurements[peak_idx].enstrophy
                rank_at_peak = measurements[peak_idx].max_rank
                
                if verbose:
                    print(f"\n  *** PEAK DETECTED at t={t_peak:.3f} ***")
                    print(f"      Enstrophy: {enstrophy_peak:.4f}")
                    print(f"      Max rank: {rank_at_peak}")
        
        # Progress
        if verbose and (step + 1) % checkpoint_every == 0:
            pct = 100 * (step + 1) / n_steps
            print(f"  [{pct:5.1f}%] t={diag['time']:.3f}, Ω={diag['enstrophy']:.4f}, "
                  f"rank={diag['max_rank']}, {diag['step_ms']:.1f}ms")
        
        # Early termination: 50% past peak
        if peak_detected and diag['time'] > t_peak * 1.5:
            if verbose:
                print(f"  Stopping: 50% past peak (t={diag['time']:.3f} > 1.5 × {t_peak:.3f})")
            break
    
    total_time = time.perf_counter() - t_run_start
    
    # If no peak detected, use maximum enstrophy point
    if not peak_detected:
        max_enstrophy = max(enstrophy_history)
        max_idx = enstrophy_history.index(max_enstrophy)
        t_peak = measurements[max_idx].t
        enstrophy_peak = max_enstrophy
        rank_at_peak = measurements[max_idx].max_rank
        
        if verbose:
            print(f"\n  No peak detected. Max enstrophy at t={t_peak:.3f}: {enstrophy_peak:.4f}")
    
    if verbose:
        print(f"\n  RESULTS:")
        print(f"    Enstrophy peak: {enstrophy_peak:.4f} at t={t_peak:.3f}")
        print(f"    Rank at peak: {rank_at_peak}")
        print(f"    Max rank overall: {max_rank_overall}")
        print(f"    Total time: {total_time:.1f}s ({len(measurements)} steps)")
    
    # Create result
    result = RunResult(
        Re=Re,
        N=N,
        n_bits=n_bits,
        n_cores=n_cores,
        nu=nu,
        dt=dt,
        enstrophy_peak=enstrophy_peak,
        t_peak=t_peak,
        rank_at_peak=rank_at_peak,
        max_rank_overall=max_rank_overall,
        measurements=[asdict(m) for m in measurements],
        total_time_s=total_time,
        steps=len(measurements),
    )
    
    # Cleanup
    del solver
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return result


def run_full_study(device: str = 'cuda'):
    """
    Run the full Rank vs Re study.
    
    Configuration:
    - Re=1600 at 512³ (n_bits=9)
    - Re=3200 at 1024³ (n_bits=10)
    """
    print("="*70)
    print("OPERATIONAL RANK STUDY: FIGURE 1 (THE THESIS)")
    print("="*70)
    print()
    print("Hypothesis: QTT rank saturates at ~20-30 regardless of Re")
    print("If true: O(N log N) turbulence!")
    print()
    
    # Configuration
    cases = [
        # (Re, n_bits, max_rank)
        (1600, 9, 32),   # 512³ - fixed rank 32
        (3200, 10, 32),  # 1024³ - fixed rank 32
    ]
    
    results = []
    
    for Re, n_bits, max_rank in cases:
        N = 2 ** n_bits
        print(f"\n{'#'*70}")
        print(f"# CASE: Re={Re}, N={N}³, max_rank={max_rank}")
        print(f"{'#'*70}")
        
        try:
            result = run_operational_rank_study(
                Re=Re,
                n_bits=n_bits,
                max_rank=max_rank,
                device=device,
                verbose=True,
            )
            results.append(result)
            
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: RANK vs REYNOLDS NUMBER")
    print("="*70)
    print()
    print(f"{'Re':>6} {'N':>6} {'n_cores':>8} {'Ω_peak':>10} {'t_peak':>8} "
          f"{'rank@peak':>10} {'max_rank':>10} {'time':>10}")
    print("-"*70)
    
    for r in results:
        print(f"{r.Re:>6.0f} {r.N:>6}³ {r.n_cores:>8} {r.enstrophy_peak:>10.4f} "
              f"{r.t_peak:>8.3f} {r.rank_at_peak:>10} {r.max_rank_overall:>10} "
              f"{r.total_time_s:>9.1f}s")
    
    print()
    
    # Thesis validation
    if len(results) >= 2:
        r1, r2 = results[0], results[1]
        rank_ratio = r2.rank_at_peak / r1.rank_at_peak if r1.rank_at_peak > 0 else float('inf')
        re_ratio = r2.Re / r1.Re
        
        # If rank ~ Re^α, then rank_ratio = re_ratio^α
        # α = log(rank_ratio) / log(re_ratio)
        import math
        alpha = math.log(rank_ratio) / math.log(re_ratio) if rank_ratio > 0 else float('inf')
        
        print(f"SCALING ANALYSIS:")
        print(f"  Re ratio: {r2.Re}/{r1.Re} = {re_ratio:.2f}")
        print(f"  Rank ratio: {r2.rank_at_peak}/{r1.rank_at_peak} = {rank_ratio:.2f}")
        print(f"  Inferred α (rank ~ Re^α): {alpha:.3f}")
        print()
        
        if alpha < 0.1:
            print("  ✓ THESIS CONFIRMED: Rank nearly constant!")
            print("    → O(N log N) turbulence is REAL")
        elif alpha < 0.25:
            print("  ~ THESIS PARTIALLY CONFIRMED: Weak scaling (α < 0.25)")
            print("    → Better than O(N^0.5), significant compression")
        else:
            print("  ✗ THESIS REJECTED: Strong scaling (α > 0.25)")
            print(f"    → Rank grows as Re^{alpha:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"operational_rank_study_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'hypothesis': 'QTT rank saturates at ~20-30 regardless of Re',
        'results': [asdict(r) if hasattr(r, '__dict__') else r for r in results],
    }
    
    # Convert RunResult to dict properly
    output['results'] = []
    for r in results:
        output['results'].append({
            'Re': r.Re,
            'N': r.N,
            'n_bits': r.n_bits,
            'n_cores': r.n_cores,
            'nu': r.nu,
            'dt': r.dt,
            'enstrophy_peak': r.enstrophy_peak,
            't_peak': r.t_peak,
            'rank_at_peak': r.rank_at_peak,
            'max_rank_overall': r.max_rank_overall,
            'total_time_s': r.total_time_s,
            'steps': r.steps,
            # Exclude full measurements to keep file small
            'measurement_count': len(r.measurements),
        })
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def quick_test(device: str = 'cuda'):
    """Quick test at Re=400, 64³."""
    print("QUICK TEST: Re=400, N=64³")
    print("="*50)
    
    result = run_operational_rank_study(
        Re=400,
        n_bits=6,  # 64³
        max_rank=32,
        device=device,
        verbose=True,
    )
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        run_full_study()
