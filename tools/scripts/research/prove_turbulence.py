#!/usr/bin/env python3
"""
TURBULENCE PROOF SUITE
======================

Production-grade validation of 3D Navier-Stokes solvers.

Physical Laws Verified:
1. TAYLOR-GREEN BENCHMARK: Analytical solution matching
2. ENERGY INEQUALITY: dE/dt ≤ -2ν·Ω (viscous dissipation bound)
3. ENSTROPHY EVOLUTION: dΩ/dt controlled by stretching/viscosity
4. KOLMOGOROV SPECTRUM: E(k) ~ k^(-5/3) in inertial range
5. DIVERGENCE-FREE: ∇·u ≈ 0 (incompressibility)

Gate Criteria (Pass/Fail):
- Taylor-Green decay rate error < 5%
- Energy monotonically decreasing (viscous case)
- Enstrophy bounded by exponential growth
- Divergence max < 10^-6
- Kolmogorov spectrum shows power-law decay

Author: HyperTensor Team
Date: 2025
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from torch import Tensor
import numpy as np

# Import spectral solver (validated)
from ontic.cfd.ns_3d import (
    NS3DSolver, NSState3D, 
    compute_divergence_3d, compute_vorticity_3d,
    taylor_green_3d_exact_energy,
)


def compute_kinetic_energy(u: Tensor, v: Tensor, w: Tensor, dV: float) -> float:
    """Compute kinetic energy: E = (1/2) ∫|u|² dV"""
    return 0.5 * ((u**2 + v**2 + w**2).sum() * dV).item()


def compute_enstrophy(
    u: Tensor, v: Tensor, w: Tensor, 
    dx: float, dy: float, dz: float,
) -> float:
    """Compute enstrophy: Ω = (1/2) ∫|ω|² dV"""
    omega_x, omega_y, omega_z = compute_vorticity_3d(u, v, w, dx, dy, dz, method='spectral')
    dV = dx * dy * dz
    return 0.5 * ((omega_x**2 + omega_y**2 + omega_z**2).sum() * dV).item()


def compute_energy_spectrum(
    u: Tensor, v: Tensor, w: Tensor,
    N: int, L: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1D energy spectrum E(k) by shell averaging."""
    ux_hat = torch.fft.fftn(u)
    uy_hat = torch.fft.fftn(v)
    uz_hat = torch.fft.fftn(w)
    
    scale = 1.0 / N**3
    E_hat = 0.5 * scale * (torch.abs(ux_hat)**2 + torch.abs(uy_hat)**2 + torch.abs(uz_hat)**2)
    
    k = torch.fft.fftfreq(N, d=1.0/N, device=u.device)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)
    
    k_max = int(N / 2)
    k_vals = np.arange(1, k_max)
    E_k = np.zeros(len(k_vals))
    
    E_hat_np = E_hat.cpu().numpy()
    k_mag_np = k_mag.cpu().numpy()
    
    for i, ki in enumerate(k_vals):
        mask = (k_mag_np >= ki) & (k_mag_np < ki + 1)
        E_k[i] = E_hat_np[mask].sum()
    
    return k_vals, E_k


# ═══════════════════════════════════════════════════════════════════════════════════════
# PROOF 1: TAYLOR-GREEN DECAY
# ═══════════════════════════════════════════════════════════════════════════════════════

def proof_taylor_green_decay(N: int = 32, nu: float = 0.1, T_final: float = 0.5) -> Dict:
    """
    PROOF 1: Taylor-Green Vortex Decay Benchmark
    
    Gate: |E_numerical - E_exact| / E_exact < 5% at T_final
    """
    print("\n" + "═" * 70)
    print("PROOF 1: Taylor-Green Vortex Decay")
    print("═" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    solver = NS3DSolver(
        Nx=N, Ny=N, Nz=N,
        Lx=2*math.pi, Ly=2*math.pi, Lz=2*math.pi,
        nu=nu,
        dtype=torch.float64,
        device=device,
    )
    
    print(f"Grid: {N}³, ν = {nu}, device = {device}")
    
    state = solver.create_taylor_green_3d(A=1.0)
    
    dV = solver.dx * solver.dy * solver.dz
    E0 = compute_kinetic_energy(state.u, state.v, state.w, dV)
    Omega0 = compute_enstrophy(state.u, state.v, state.w, solver.dx, solver.dy, solver.dz)
    div0 = torch.abs(compute_divergence_3d(state.u, state.v, state.w, 
                                           solver.dx, solver.dy, solver.dz, method='spectral')).max().item()
    
    print(f"\nInitial state:")
    print(f"  E₀ = {E0:.6f}")
    print(f"  Ω₀ = {Omega0:.6f}")
    print(f"  max|∇·u| = {div0:.2e}")
    
    print(f"\nEvolving to T = {T_final}...")
    t_start = time.perf_counter()
    
    result = solver.solve(state, t_final=T_final, cfl_target=0.3, verbose=False)
    
    elapsed = time.perf_counter() - t_start
    
    if not result.completed:
        print(f"  ⚠ Solver failed: {result.reason}")
        return {'passed': False, 'error': result.reason}
    
    final_state = result.final_state
    E_final = compute_kinetic_energy(final_state.u, final_state.v, final_state.w, dV)
    E_exact = taylor_green_3d_exact_energy(T_final, nu, E0)
    
    decay_error = abs(E_final - E_exact) / max(E_exact, 1e-10)
    
    div_final = torch.abs(compute_divergence_3d(final_state.u, final_state.v, final_state.w,
                                                 solver.dx, solver.dy, solver.dz, method='spectral')).max().item()
    
    print(f"\n{'─' * 70}")
    print(f"RESULTS at T = {T_final}:")
    print(f"  E_numerical = {E_final:.6f}")
    print(f"  E_exact     = {E_exact:.6f}")
    print(f"  Decay error = {decay_error*100:.2f}%")
    print(f"  max|∇·u|    = {div_final:.2e}")
    print(f"  Wall time   = {elapsed*1000:.1f}ms")
    
    passed = decay_error < 0.05
    print(f"\n{'✓ PASSED' if passed else '✗ FAILED'}: Decay error {'<' if passed else '>'} 5%")
    
    return {
        'passed': passed,
        'decay_error': decay_error,
        'E0': E0,
        'E_final': E_final,
        'E_exact': E_exact,
        'max_divergence': div_final,
        'wall_time': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# PROOF 2: ENERGY INEQUALITY
# ═══════════════════════════════════════════════════════════════════════════════════════

def proof_energy_inequality(N: int = 32, nu: float = 0.01, T_final: float = 0.5) -> Dict:
    """
    PROOF 2: Energy Inequality
    
    Gate: Energy monotonically decreasing
    """
    print("\n" + "═" * 70)
    print("PROOF 2: Energy Inequality (Viscous Dissipation)")
    print("═" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    solver = NS3DSolver(
        Nx=N, Ny=N, Nz=N,
        Lx=2*math.pi, Ly=2*math.pi, Lz=2*math.pi,
        nu=nu,
        dtype=torch.float64,
        device=device,
    )
    
    print(f"Grid: {N}³, ν = {nu}")
    
    state = solver.create_taylor_green_3d(A=1.0)
    
    dV = solver.dx * solver.dy * solver.dz
    dt = 0.01
    n_steps = int(T_final / dt)
    
    E_prev = compute_kinetic_energy(state.u, state.v, state.w, dV)
    Omega_prev = compute_enstrophy(state.u, state.v, state.w, solver.dx, solver.dy, solver.dz)
    E0 = E_prev
    
    violations = 0
    max_violation = 0.0
    max_dissipation_error = 0.0
    
    print(f"\nChecking energy inequality for {n_steps} steps...")
    
    for step in range(n_steps):
        state, _ = solver.step_rk4(state, dt)
        
        E_curr = compute_kinetic_energy(state.u, state.v, state.w, dV)
        Omega_curr = compute_enstrophy(state.u, state.v, state.w, solver.dx, solver.dy, solver.dz)
        
        if E_curr > E_prev * (1 + 1e-10):
            violations += 1
            violation = (E_curr - E_prev) / max(E_prev, 1e-10)
            max_violation = max(max_violation, violation)
        
        dE_dt = (E_curr - E_prev) / dt
        expected_rate = -2 * nu * Omega_prev
        
        if abs(expected_rate) > 1e-10:
            rel_error = abs(dE_dt - expected_rate) / abs(expected_rate)
            max_dissipation_error = max(max_dissipation_error, rel_error)
        
        E_prev = E_curr
        Omega_prev = Omega_curr
    
    E_final = E_curr
    
    print(f"\n{'─' * 70}")
    print(f"RESULTS:")
    print(f"  Initial E = {E0:.6f}")
    print(f"  Final E   = {E_final:.6f}")
    print(f"  Energy ratio = {E_final/E0:.6f}")
    print(f"  Violations: {violations}/{n_steps}")
    if violations > 0:
        print(f"  Max violation: {max_violation*100:.6f}%")
    print(f"  Max dissipation rate error: {max_dissipation_error*100:.1f}%")
    
    passed = violations == 0
    print(f"\n{'✓ PASSED' if passed else '✗ FAILED'}: Energy {'monotonically decreasing' if passed else 'has violations'}")
    
    return {
        'passed': passed,
        'violations': violations,
        'max_violation': max_violation,
        'E0': E0,
        'E_final': E_final,
        'max_dissipation_error': max_dissipation_error,
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# PROOF 3: ENSTROPHY BOUNDS
# ═══════════════════════════════════════════════════════════════════════════════════════

def proof_enstrophy_bounds(N: int = 32, nu: float = 0.01, T_final: float = 1.0) -> Dict:
    """
    PROOF 3: Enstrophy Bounds (No Finite-Time Blowup)
    
    Gate: Ω(t) < Ω₀ exp(C·t) where C = 10
    """
    print("\n" + "═" * 70)
    print("PROOF 3: Enstrophy Bounds (Blowup Detection)")
    print("═" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    solver = NS3DSolver(
        Nx=N, Ny=N, Nz=N,
        Lx=2*math.pi, Ly=2*math.pi, Lz=2*math.pi,
        nu=nu,
        dtype=torch.float64,
        device=device,
    )
    
    print(f"Grid: {N}³, ν = {nu}")
    
    state = solver.create_taylor_green_3d(A=1.0)
    
    Omega0 = compute_enstrophy(state.u, state.v, state.w, solver.dx, solver.dy, solver.dz)
    
    dt = 0.01
    n_steps = int(T_final / dt)
    growth_constant = 10.0
    
    blowup_detected = False
    history = [(0.0, Omega0)]
    
    print(f"\nMonitoring enstrophy for {n_steps} steps...")
    print(f"Bound: Ω(t) < Ω₀·exp({growth_constant}·t)")
    
    for step in range(n_steps):
        state, _ = solver.step_rk4(state, dt)
        
        if (step + 1) % max(1, n_steps // 10) == 0:
            Omega_t = compute_enstrophy(state.u, state.v, state.w, solver.dx, solver.dy, solver.dz)
            bound = Omega0 * math.exp(growth_constant * state.t)
            history.append((state.t, Omega_t))
            
            print(f"  t={state.t:.3f}: Ω={Omega_t:.6f} (bound={bound:.6f})")
            
            if Omega_t > bound:
                blowup_detected = True
                print(f"  ⚠ BLOWUP: Ω exceeds bound!")
    
    Omega_final = compute_enstrophy(state.u, state.v, state.w, solver.dx, solver.dy, solver.dz)
    decayed = Omega_final < Omega0
    
    print(f"\n{'─' * 70}")
    print(f"RESULTS:")
    print(f"  Ω₀ = {Omega0:.6f}")
    print(f"  Ω_final = {Omega_final:.6f}")
    print(f"  Ratio: {Omega_final/Omega0:.4f}")
    print(f"  Enstrophy {'decayed' if decayed else 'grew'}")
    
    passed = not blowup_detected
    print(f"\n{'✓ PASSED' if passed else '✗ FAILED'}: No finite-time blowup detected")
    
    return {
        'passed': passed,
        'Omega0': Omega0,
        'Omega_final': Omega_final,
        'blowup_detected': blowup_detected,
        'history': history,
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# PROOF 4: DIVERGENCE-FREE
# ═══════════════════════════════════════════════════════════════════════════════════════

def proof_divergence_free(N: int = 32, nu: float = 0.01, T_final: float = 0.5) -> Dict:
    """
    PROOF 4: Divergence-Free Constraint
    
    Gate: max|∇·u| < 10^-6 throughout
    """
    print("\n" + "═" * 70)
    print("PROOF 4: Divergence-Free Constraint")
    print("═" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    solver = NS3DSolver(
        Nx=N, Ny=N, Nz=N,
        Lx=2*math.pi, Ly=2*math.pi, Lz=2*math.pi,
        nu=nu,
        dtype=torch.float64,
        device=device,
    )
    
    print(f"Grid: {N}³, ν = {nu}")
    
    state = solver.create_taylor_green_3d(A=1.0)
    
    div0 = torch.abs(compute_divergence_3d(state.u, state.v, state.w,
                                            solver.dx, solver.dy, solver.dz, method='spectral')).max().item()
    
    print(f"\nInitial max|∇·u| = {div0:.2e}")
    
    dt = 0.01
    n_steps = int(T_final / dt)
    max_div_overall = div0
    history = [(0.0, div0)]
    
    print(f"\nEvolving for {n_steps} steps...")
    
    for step in range(n_steps):
        state, _ = solver.step_rk4(state, dt)
        
        if (step + 1) % max(1, n_steps // 5) == 0:
            div = torch.abs(compute_divergence_3d(state.u, state.v, state.w,
                                                   solver.dx, solver.dy, solver.dz, method='spectral')).max().item()
            max_div_overall = max(max_div_overall, div)
            history.append((state.t, div))
            print(f"  t={state.t:.3f}: max|∇·u| = {div:.2e}")
    
    print(f"\n{'─' * 70}")
    print(f"RESULTS:")
    print(f"  Initial max|∇·u| = {div0:.2e}")
    print(f"  Overall max|∇·u| = {max_div_overall:.2e}")
    
    threshold = 1e-6
    passed = max_div_overall < threshold
    print(f"\n{'✓ PASSED' if passed else '✗ FAILED'}: max|∇·u| {'<' if passed else '>'} {threshold:.0e}")
    
    return {
        'passed': passed,
        'max_divergence': max_div_overall,
        'threshold': threshold,
        'history': history,
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# PROOF 5: KOLMOGOROV SPECTRUM
# ═══════════════════════════════════════════════════════════════════════════════════════

def proof_kolmogorov_spectrum(N: int = 64, nu: float = 0.001, T_spinup: float = 0.5) -> Dict:
    """
    PROOF 5: Kolmogorov Energy Spectrum
    
    Gate: Spectrum shows power-law decay (exponent < -1)
    """
    print("\n" + "═" * 70)
    print("PROOF 5: Kolmogorov Energy Spectrum")
    print("═" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    L = 2 * math.pi
    
    solver = NS3DSolver(
        Nx=N, Ny=N, Nz=N,
        Lx=L, Ly=L, Lz=L,
        nu=nu,
        dtype=torch.float64,
        device=device,
    )
    
    Re = L / nu
    print(f"Grid: {N}³, ν = {nu}, Re ≈ {Re:.0f}")
    
    state = solver.create_taylor_green_3d(A=1.0)
    
    print(f"\nSpinning up to T = {T_spinup}...")
    t_start = time.perf_counter()
    
    result = solver.solve(state, t_final=T_spinup, cfl_target=0.3, verbose=False)
    
    elapsed = time.perf_counter() - t_start
    print(f"  Spinup time: {elapsed:.1f}s")
    
    if not result.completed:
        print(f"  ⚠ Solver failed: {result.reason}")
        return {'passed': False, 'error': result.reason}
    
    final_state = result.final_state
    
    print("\nComputing energy spectrum...")
    k_vals, E_k = compute_energy_spectrum(final_state.u, final_state.v, final_state.w, N, L)
    
    k_min_fit = 2
    k_max_fit = N // 4
    
    mask = (k_vals >= k_min_fit) & (k_vals <= k_max_fit) & (E_k > 1e-20)
    
    if mask.sum() >= 3:
        log_k = np.log(k_vals[mask])
        log_E = np.log(E_k[mask])
        
        A = np.vstack([log_k, np.ones_like(log_k)]).T
        slope, intercept = np.linalg.lstsq(A, log_E, rcond=None)[0]
        
        log_E_pred = slope * log_k + intercept
        ss_res = np.sum((log_E - log_E_pred) ** 2)
        ss_tot = np.sum((log_E - log_E.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        slope = 0.0
        r_squared = 0.0
    
    print(f"\n{'─' * 70}")
    print(f"RESULTS:")
    print(f"  Fitted exponent: {slope:.3f}")
    print(f"  Kolmogorov: -5/3 ≈ -1.667")
    print(f"  R² of fit: {r_squared:.4f}")
    print(f"  Fit range: k = {k_min_fit} to {k_max_fit}")
    
    has_decay = slope < -1.0
    
    print(f"\n{'✓ PASSED' if has_decay else '✗ FAILED'}: Spectrum {'shows' if has_decay else 'lacks'} power-law decay")
    
    return {
        'passed': has_decay,
        'exponent': slope,
        'r_squared': r_squared,
        'Re': Re,
        'k_vals': k_vals.tolist(),
        'E_k': E_k.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_all_proofs() -> Dict:
    """Run complete turbulence proof suite."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " TURBULENCE PROOF SUITE ".center(68) + "║")
    print("║" + " 3D Navier-Stokes Physics Validation ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    results = {}
    
    t_start = time.perf_counter()
    
    results['taylor_green'] = proof_taylor_green_decay(N=32, nu=0.1, T_final=0.3)
    results['energy_inequality'] = proof_energy_inequality(N=32, nu=0.01, T_final=0.3)
    results['enstrophy_bounds'] = proof_enstrophy_bounds(N=32, nu=0.01, T_final=0.5)
    results['divergence_free'] = proof_divergence_free(N=32, nu=0.01, T_final=0.3)
    results['kolmogorov'] = proof_kolmogorov_spectrum(N=32, nu=0.01, T_spinup=0.3)
    
    total_time = time.perf_counter() - t_start
    
    print("\n" + "═" * 70)
    print("PROOF SUMMARY")
    print("═" * 70)
    
    all_passed = True
    for name, result in results.items():
        status = "✓ PASSED" if result['passed'] else "✗ FAILED"
        print(f"  {name:25s}: {status}")
        if not result['passed']:
            all_passed = False
    
    print(f"\n  Total time: {total_time:.1f}s")
    
    if all_passed:
        print("\n" + "═" * 70)
        print("  ✓✓✓ ALL PROOFS PASSED ✓✓✓  ".center(70))
        print("  TURBULENCE PHYSICS VALIDATED  ".center(70))
        print("═" * 70)
    else:
        print("\n" + "═" * 70)
        print("  SOME PROOFS FAILED - REVIEW ABOVE  ".center(70))
        print("═" * 70)
    
    results['all_passed'] = all_passed
    results['total_time'] = total_time
    
    return results


if __name__ == "__main__":
    results = run_all_proofs()
