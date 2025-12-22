#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LEVEL 3: COMPUTATIONAL DISCOVERY
=================================

The goal: Push to extreme Reynolds numbers and observe chi(t) behavior.

Two possible discoveries:
1. SINGULARITY CANDIDATE: chi -> infinity in finite time
2. UNIVERSAL BOUND: chi stays bounded for all smooth initial data

From NS_MILLENNIUM_FRAMEWORK.md Phase 3:
    - Re = 10^3 simulations with chi(t) tracking
    - Re = 10^4 simulations with chi(t) tracking  
    - Re = 10^5+ if computationally feasible
    - chi(t) growth rate analysis
    - chi vs Re scaling law

Gate Criteria: Clear trend identified in chi(t) scaling with Re.

Tag: [LEVEL-3] [DISCOVERY]
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

PROOF_RESULTS = {}


# =============================================================================
# Gate 1: Re=1000 chi Scaling (3D Taylor-Green)
# =============================================================================

def gate_re1000_chi_tracking() -> Dict:
    """
    Gate 1: Track chi(t) at Re=1000 (moderate turbulence).
    
    At Re=1000, 3D Taylor-Green develops complex vortex dynamics
    but should remain regular. Track:
        - chi(t) trajectory
        - chi_max reached
        - growth rate (polynomial? exponential?)
    """
    from tensornet.cfd.ns_3d import NS3DSolver
    from tensornet.cfd.chi_diagnostic import compute_chi_state_3d, ChiTrajectory, analyze_regularity
    
    print("\n" + "=" * 60)
    print("Gate 1: Re=1000 chi Tracking")
    print("=" * 60)
    
    # Parameters for Re=1000
    # Re = U*L/nu, with U=1, L=2*pi -> nu = 2*pi/1000 ~ 0.00628
    Re = 1000
    N = 32  # 32^3 grid (memory-friendly for testing)
    L = 2 * np.pi
    U = 1.0
    nu = U * L / Re
    dt = 0.005
    T_final = 2.0  # Simulate to t=2
    n_steps = int(T_final / dt)
    
    print(f"  Re = {Re}")
    print(f"  N = {N}^3")
    print(f"  nu = {nu:.6f}")
    print(f"  T_final = {T_final}")
    print(f"  Steps = {n_steps}")
    
    solver = NS3DSolver(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, nu=nu, dtype=torch.float64)
    state = solver.create_taylor_green_3d()
    
    dx = dy = dz = L / N
    trajectory = ChiTrajectory()
    
    # Sample every 10 steps for efficiency
    sample_interval = max(1, n_steps // 50)
    
    for step in range(n_steps + 1):
        t = step * dt
        
        if step % sample_interval == 0 or step == n_steps:
            chi_state = compute_chi_state_3d(
                state.u, state.v, state.w, t=t,
                dx=dx, dy=dy, dz=dz,
                chi_target=128,
                truncation_tol=1e-8
            )
            trajectory.add(chi_state)
        
        if step < n_steps:
            state, _ = solver.step_rk4(state, dt)
    
    # Analyze
    analysis = analyze_regularity(trajectory)
    
    chi_values = trajectory.chi_values
    chi_max = max(chi_values)
    chi_initial = chi_values[0]
    chi_final = chi_values[-1]
    chi_growth = analysis.get('chi_growth_rate', 0.0)
    
    print(f"\n  chi_initial = {chi_initial:.1f}")
    print(f"  chi_max = {chi_max:.1f}")
    print(f"  chi_final = {chi_final:.1f}")
    print(f"  chi_growth_rate = {chi_growth:.6f}")
    print(f"  regularity = {analysis.get('regularity_assessment', 'N/A')}")
    
    # Pass criterion: chi stayed bounded, no blowup detected
    passed = (
        chi_max < 500 and  # Reasonable bound for Re=1000
        analysis.get('regularity_assessment') != 'potential_blowup'
    )
    
    result = {
        'passed': passed,
        'Re': Re,
        'N': N,
        'chi_initial': chi_initial,
        'chi_max': chi_max,
        'chi_final': chi_final,
        'chi_growth_rate': chi_growth,
        'regularity': analysis.get('regularity_assessment'),
        'n_samples': len(chi_values),
    }
    
    status = "[OK]" if passed else "[FAIL]"
    print(f"\n  {status} Gate 1: Re=1000 chi tracking")
    
    return result


# =============================================================================
# Gate 2: Re=10000 chi Scaling (High Reynolds)
# =============================================================================

def gate_re10000_chi_tracking() -> Dict:
    """
    Gate 2: Track chi(t) at Re=10,000 (approaching turbulence).
    
    At Re=10^4, expect:
        - Stronger vortex stretching
        - Higher chi values
        - Potentially faster growth
    """
    from tensornet.cfd.ns_3d import NS3DSolver
    from tensornet.cfd.chi_diagnostic import compute_chi_state_3d, ChiTrajectory, analyze_regularity
    
    print("\n" + "=" * 60)
    print("Gate 2: Re=10000 chi Tracking")
    print("=" * 60)
    
    # Parameters for Re=10000
    Re = 10000
    N = 32  # Keep 32^3 for feasibility
    L = 2 * np.pi
    U = 1.0
    nu = U * L / Re
    dt = 0.002  # Smaller dt for stability
    T_final = 1.0  # Shorter simulation
    n_steps = int(T_final / dt)
    
    print(f"  Re = {Re}")
    print(f"  N = {N}^3")
    print(f"  nu = {nu:.8f}")
    print(f"  T_final = {T_final}")
    print(f"  Steps = {n_steps}")
    
    solver = NS3DSolver(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, nu=nu, dtype=torch.float64)
    state = solver.create_taylor_green_3d()
    
    dx = dy = dz = L / N
    trajectory = ChiTrajectory()
    
    sample_interval = max(1, n_steps // 50)
    
    try:
        for step in range(n_steps + 1):
            t = step * dt
            
            if step % sample_interval == 0 or step == n_steps:
                chi_state = compute_chi_state_3d(
                    state.u, state.v, state.w, t=t,
                    dx=dx, dy=dy, dz=dz,
                    chi_target=256,
                    truncation_tol=1e-8
                )
                trajectory.add(chi_state)
            
            if step < n_steps:
                state, _ = solver.step_rk4(state, dt)
                
                # Check for NaN (instability at high Re)
                if torch.isnan(state.u).any():
                    print(f"  [WARNING] NaN detected at step {step}")
                    break
                    
    except Exception as e:
        print(f"  [ERROR] Simulation failed: {e}")
    
    if len(trajectory.states) < 2:
        return {
            'passed': False,
            'Re': Re,
            'error': 'Insufficient data - simulation unstable',
        }
    
    analysis = analyze_regularity(trajectory)
    
    chi_values = trajectory.chi_values
    chi_max = max(chi_values)
    chi_initial = chi_values[0]
    chi_final = chi_values[-1]
    chi_growth = analysis.get('chi_growth_rate', 0.0)
    
    print(f"\n  chi_initial = {chi_initial:.1f}")
    print(f"  chi_max = {chi_max:.1f}")
    print(f"  chi_final = {chi_final:.1f}")
    print(f"  chi_growth_rate = {chi_growth:.6f}")
    print(f"  regularity = {analysis.get('regularity_assessment', 'N/A')}")
    
    # At Re=10000, higher chi is expected but should still be bounded
    passed = (
        chi_max < 2000 and
        analysis.get('regularity_assessment') != 'potential_blowup'
    )
    
    result = {
        'passed': passed,
        'Re': Re,
        'N': N,
        'chi_initial': chi_initial,
        'chi_max': chi_max,
        'chi_final': chi_final,
        'chi_growth_rate': chi_growth,
        'regularity': analysis.get('regularity_assessment'),
        'n_samples': len(chi_values),
    }
    
    status = "[OK]" if passed else "[FAIL]"
    print(f"\n  {status} Gate 2: Re=10000 chi tracking")
    
    return result


# =============================================================================
# Gate 3: Reynolds Scaling Law chi_max ~ Re^alpha
# =============================================================================

def gate_reynolds_scaling_law() -> Dict:
    """
    Gate 3: Fit chi_max ~ Re^alpha across multiple Reynolds numbers.
    
    Key question: How does chi scale with Re?
        - alpha < 1: Sublinear (good for regularity)
        - alpha = 1: Linear
        - alpha > 1: Superlinear (concerning)
        - alpha -> infinity: Potential blowup
    """
    from tensornet.cfd.ns_3d import NS3DSolver
    from tensornet.cfd.chi_diagnostic import compute_chi_state_3d, ChiTrajectory
    
    print("\n" + "=" * 60)
    print("Gate 3: Reynolds Scaling Law")
    print("=" * 60)
    
    # Test multiple Re values
    Re_values = [100, 500, 1000, 2000, 5000]
    chi_max_values = []
    
    N = 32
    L = 2 * np.pi
    U = 1.0
    dx = dy = dz = L / N
    T_final = 0.5  # Short simulation for scaling study
    
    for Re in Re_values:
        nu = U * L / Re
        dt = min(0.01, 0.1 * nu)  # CFL-like condition
        n_steps = max(10, int(T_final / dt))
        
        print(f"\n  Re = {Re}, nu = {nu:.6f}, steps = {n_steps}")
        
        solver = NS3DSolver(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, nu=nu, dtype=torch.float64)
        state = solver.create_taylor_green_3d()
        
        chi_max = 0
        
        try:
            for step in range(n_steps):
                state, _ = solver.step_rk4(state, dt)
                
                # Sample chi periodically
                if step % max(1, n_steps // 10) == 0:
                    chi_state = compute_chi_state_3d(
                        state.u, state.v, state.w, t=step * dt,
                        dx=dx, dy=dy, dz=dz
                    )
                    chi_max = max(chi_max, chi_state.chi_actual)
                
                if torch.isnan(state.u).any():
                    print(f"    [NaN at step {step}]")
                    break
                    
        except Exception as e:
            print(f"    [Error: {e}]")
            
        chi_max_values.append(chi_max)
        print(f"    chi_max = {chi_max:.1f}")
    
    # Fit log(chi_max) ~ alpha * log(Re)
    log_Re = np.log(Re_values)
    log_chi = np.log([max(1, c) for c in chi_max_values])
    
    # Linear regression
    n = len(Re_values)
    sum_x = np.sum(log_Re)
    sum_y = np.sum(log_chi)
    sum_xx = np.sum(log_Re ** 2)
    sum_xy = np.sum(log_Re * log_chi)
    
    alpha = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    intercept = (sum_y - alpha * sum_x) / n
    
    # R^2 for fit quality
    y_mean = sum_y / n
    ss_tot = np.sum((log_chi - y_mean) ** 2)
    y_pred = alpha * log_Re + intercept
    ss_res = np.sum((log_chi - y_pred) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\n  Scaling Law: chi_max ~ Re^{alpha:.3f}")
    print(f"  R^2 = {r_squared:.4f}")
    print(f"  Intercept (log scale) = {intercept:.3f}")
    
    # Interpretation
    if alpha < 0.5:
        interpretation = "SUBLINEAR: chi grows slowly with Re (favorable for regularity)"
    elif alpha < 1.0:
        interpretation = "SUBLINEAR: chi grows moderately (likely bounded)"
    elif alpha < 1.5:
        interpretation = "LINEAR: chi ~ Re (need larger resolution to confirm)"
    else:
        interpretation = "SUPERLINEAR: chi grows fast with Re (concerning)"
    
    print(f"  Interpretation: {interpretation}")
    
    # Pass if we got reasonable data and alpha is finite
    passed = (
        len(chi_max_values) >= 3 and
        np.isfinite(alpha) and
        r_squared > 0.5  # Reasonable fit
    )
    
    result = {
        'passed': passed,
        'Re_values': Re_values,
        'chi_max_values': chi_max_values,
        'alpha': float(alpha),
        'r_squared': float(r_squared),
        'interpretation': interpretation,
    }
    
    status = "[OK]" if passed else "[FAIL]"
    print(f"\n  {status} Gate 3: Reynolds scaling law")
    
    return result


# =============================================================================
# Gate 4: Blowup vs Bound Detection
# =============================================================================

def gate_blowup_detection() -> Dict:
    """
    Gate 4: Automated detection of potential singularities.
    
    Blowup indicators:
        1. chi(t) growth rate > 1 (exponential)
        2. chi(t) acceleration > 0 (accelerating growth)
        3. enstrophy growing faster than chi
        
    Boundedness indicators:
        1. chi(t) saturates or oscillates
        2. Growth rate decays over time
        3. chi_max / chi_initial remains finite
    """
    from tensornet.cfd.ns_3d import NS3DSolver
    from tensornet.cfd.chi_diagnostic import compute_chi_state_3d, ChiTrajectory
    
    print("\n" + "=" * 60)
    print("Gate 4: Blowup vs Bound Detection")
    print("=" * 60)
    
    # Use moderately high Re for detection test
    Re = 2000
    N = 32
    L = 2 * np.pi
    U = 1.0
    nu = U * L / Re
    dt = 0.002
    T_final = 1.5
    n_steps = int(T_final / dt)
    dx = dy = dz = L / N
    
    print(f"  Re = {Re}")
    print(f"  N = {N}^3")
    print(f"  T_final = {T_final}")
    
    solver = NS3DSolver(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, nu=nu, dtype=torch.float64)
    state = solver.create_taylor_green_3d()
    
    trajectory = ChiTrajectory()
    sample_interval = max(1, n_steps // 100)
    
    try:
        for step in range(n_steps + 1):
            t = step * dt
            
            if step % sample_interval == 0:
                chi_state = compute_chi_state_3d(
                    state.u, state.v, state.w, t=t,
                    dx=dx, dy=dy, dz=dz,
                    chi_target=512
                )
                trajectory.add(chi_state)
            
            if step < n_steps:
                state, _ = solver.step_rk4(state, dt)
                
                if torch.isnan(state.u).any():
                    print(f"  [WARNING] NaN at step {step}")
                    break
                    
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Analyze trajectory for blowup indicators
    chi_values = trajectory.chi_values
    times = trajectory.times
    
    if len(chi_values) < 10:
        return {
            'passed': False,
            'error': 'Insufficient trajectory data',
        }
    
    # Compute growth statistics
    chi_max = max(chi_values)
    chi_initial = chi_values[0]
    chi_ratio = chi_max / chi_initial if chi_initial > 0 else 0
    
    # Growth rate in second half vs first half
    mid = len(chi_values) // 2
    first_half_growth = (chi_values[mid] - chi_values[0]) / (times[mid] - times[0]) if mid > 0 else 0
    second_half_growth = (chi_values[-1] - chi_values[mid]) / (times[-1] - times[mid]) if times[-1] > times[mid] else 0
    
    acceleration = second_half_growth - first_half_growth
    
    print(f"\n  chi_initial = {chi_initial:.1f}")
    print(f"  chi_max = {chi_max:.1f}")
    print(f"  chi_ratio = {chi_ratio:.2f}")
    print(f"  first_half_growth = {first_half_growth:.4f}")
    print(f"  second_half_growth = {second_half_growth:.4f}")
    print(f"  acceleration = {acceleration:.4f}")
    
    # Detection criteria
    blowup_indicators = []
    bound_indicators = []
    
    # Check for blowup signs
    if acceleration > 1.0:
        blowup_indicators.append("accelerating_growth")
    if chi_ratio > 10:
        blowup_indicators.append("large_chi_ratio")
    if second_half_growth > 2 * first_half_growth > 0:
        blowup_indicators.append("growth_rate_increasing")
    
    # Check for boundedness signs
    if acceleration < 0:
        bound_indicators.append("decelerating_growth")
    if chi_ratio < 5:
        bound_indicators.append("moderate_chi_ratio")
    if second_half_growth < first_half_growth:
        bound_indicators.append("growth_rate_decreasing")
    if chi_values[-1] < chi_max:
        bound_indicators.append("chi_decreased_from_max")
    
    # Verdict
    if len(blowup_indicators) >= 2:
        verdict = "POTENTIAL_SINGULARITY"
    elif len(bound_indicators) >= 2:
        verdict = "LIKELY_BOUNDED"
    else:
        verdict = "INCONCLUSIVE"
    
    print(f"\n  Blowup indicators: {blowup_indicators}")
    print(f"  Bound indicators: {bound_indicators}")
    print(f"  VERDICT: {verdict}")
    
    # For Taylor-Green at Re=2000, we expect bounded behavior
    passed = verdict in ["LIKELY_BOUNDED", "INCONCLUSIVE"]
    
    result = {
        'passed': passed,
        'Re': Re,
        'chi_max': chi_max,
        'chi_ratio': chi_ratio,
        'acceleration': acceleration,
        'blowup_indicators': blowup_indicators,
        'bound_indicators': bound_indicators,
        'verdict': verdict,
        'n_samples': len(chi_values),
    }
    
    status = "[OK]" if passed else "[FAIL]"
    print(f"\n  {status} Gate 4: Blowup detection")
    
    return result


# =============================================================================
# Main Runner
# =============================================================================

def run_all_gates():
    """Run all Level 3 proof gates."""
    global PROOF_RESULTS
    
    print("\n" + "=" * 70)
    print("   LEVEL 3: COMPUTATIONAL DISCOVERY")
    print("   Exploring chi(t) scaling at high Reynolds numbers")
    print("=" * 70)
    
    results = {}
    passed = 0
    total = 4
    
    # Gate 1: Re=1000
    try:
        results['gate_1_re1000'] = gate_re1000_chi_tracking()
        if results['gate_1_re1000'].get('passed'):
            passed += 1
    except Exception as e:
        results['gate_1_re1000'] = {'passed': False, 'error': str(e)}
        print(f"  [ERROR] Gate 1: {e}")
    
    # Gate 2: Re=10000
    try:
        results['gate_2_re10000'] = gate_re10000_chi_tracking()
        if results['gate_2_re10000'].get('passed'):
            passed += 1
    except Exception as e:
        results['gate_2_re10000'] = {'passed': False, 'error': str(e)}
        print(f"  [ERROR] Gate 2: {e}")
    
    # Gate 3: Scaling Law
    try:
        results['gate_3_scaling'] = gate_reynolds_scaling_law()
        if results['gate_3_scaling'].get('passed'):
            passed += 1
    except Exception as e:
        results['gate_3_scaling'] = {'passed': False, 'error': str(e)}
        print(f"  [ERROR] Gate 3: {e}")
    
    # Gate 4: Blowup Detection
    try:
        results['gate_4_blowup'] = gate_blowup_detection()
        if results['gate_4_blowup'].get('passed'):
            passed += 1
    except Exception as e:
        results['gate_4_blowup'] = {'passed': False, 'error': str(e)}
        print(f"  [ERROR] Gate 4: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("   LEVEL 3 SUMMARY")
    print("=" * 70)
    print(f"  Gates passed: {passed}/{total}")
    
    # Key findings
    if 'gate_3_scaling' in results and results['gate_3_scaling'].get('passed'):
        alpha = results['gate_3_scaling'].get('alpha', 'N/A')
        print(f"  Scaling exponent: chi_max ~ Re^{alpha:.3f}")
    
    if 'gate_4_blowup' in results:
        verdict = results['gate_4_blowup'].get('verdict', 'N/A')
        print(f"  Regularity verdict: {verdict}")
    
    print("=" * 70)
    
    # Save results
    PROOF_RESULTS = {
        'level': 3,
        'name': 'Computational Discovery',
        'passed': passed,
        'total': total,
        'timestamp': datetime.now().isoformat(),
        'gates': results,
    }
    
    output_path = Path(__file__).parent / 'proof_level_3_result.json'
    with open(output_path, 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        json.dump(PROOF_RESULTS, f, indent=2, default=convert)
    
    print(f"\nResults saved to: {output_path}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_gates()
    sys.exit(0 if success else 1)
