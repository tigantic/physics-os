#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                    BLACK SWAN HUNT: LEVEL 2                                          ║
║                                                                                      ║
║                    Gradient Ascent IC Optimization                                   ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  STRATEGY: Use gradient ascent to find ICs that MAXIMIZE enstrophy growth           ║
║                                                                                      ║
║  If NS can blow up, the "Bad Apple" IC exists somewhere in IC space.                ║
║  We use optimization to actively SEARCH for it.                                     ║
║                                                                                      ║
║  OBJECTIVE: max_u0 ∫|ω(T)|² dx   subject to  ∇·u0 = 0, ||u0||₂ = 1                 ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

from tensornet.cfd.ns_3d import NS3DSolver, NSState3D


@dataclass
class OptimizationResult:
    """Result from IC optimization."""
    iterations: int
    initial_enstrophy_growth: float
    final_enstrophy_growth: float
    improvement_factor: float
    max_omega_achieved: float
    blowup_detected: bool
    best_ic: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]


def create_random_ic(N: int, L: float, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random divergence-free IC."""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Random Fourier modes
    u = torch.randn(N, N, N, dtype=torch.float64)
    v = torch.randn(N, N, N, dtype=torch.float64)
    w = torch.randn(N, N, N, dtype=torch.float64)
    
    # Project to divergence-free
    u, v, w = project_divergence_free_torch(u, v, w, L, N)
    
    # Normalize energy
    energy = (u**2 + v**2 + w**2).mean()
    scale = 1.0 / torch.sqrt(energy + 1e-10)
    
    return u * scale, v * scale, w * scale


def project_divergence_free_torch(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                                   L: float, N: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project to divergence-free space using FFT."""
    dx = L / N
    k = torch.fft.fftfreq(N, dx, dtype=torch.float64) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0
    
    u_hat = torch.fft.fftn(u)
    v_hat = torch.fft.fftn(v)
    w_hat = torch.fft.fftn(w)
    
    div_hat = 1j * (kx * u_hat + ky * v_hat + kz * w_hat)
    P_hat = div_hat / k_sq
    P_hat[0, 0, 0] = 0
    
    u_hat = u_hat - 1j * kx * P_hat
    v_hat = v_hat - 1j * ky * P_hat
    w_hat = w_hat - 1j * kz * P_hat
    
    return torch.fft.ifftn(u_hat).real, torch.fft.ifftn(v_hat).real, torch.fft.ifftn(w_hat).real


def compute_enstrophy_torch(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                            L: float, N: int) -> torch.Tensor:
    """Compute enstrophy = (1/2) ∫|ω|² dx."""
    dx = L / N
    k = torch.fft.fftfreq(N, dx, dtype=torch.float64) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    
    u_hat = torch.fft.fftn(u)
    v_hat = torch.fft.fftn(v)
    w_hat = torch.fft.fftn(w)
    
    omega_x = torch.fft.ifftn(1j * ky * w_hat - 1j * kz * v_hat).real
    omega_y = torch.fft.ifftn(1j * kz * u_hat - 1j * kx * w_hat).real
    omega_z = torch.fft.ifftn(1j * kx * v_hat - 1j * ky * u_hat).real
    
    omega_sq = omega_x**2 + omega_y**2 + omega_z**2
    dV = (L / N) ** 3
    
    return 0.5 * omega_sq.sum() * dV


def apply_dealiasing_torch(u: torch.Tensor, N: int) -> torch.Tensor:
    """Apply 2/3 dealiasing."""
    k_max = N // 3
    k = torch.fft.fftfreq(N, 1/N)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    mask = (torch.abs(kx) < k_max) & (torch.abs(ky) < k_max) & (torch.abs(kz) < k_max)
    
    u_hat = torch.fft.fftn(u)
    u_hat = u_hat * mask
    return torch.fft.ifftn(u_hat).real


def simulate_and_measure(u0: torch.Tensor, v0: torch.Tensor, w0: torch.Tensor,
                         N: int, L: float, Re: float, T: float, dt: float) -> Tuple[float, float]:
    """
    Simulate NS and return (final_enstrophy, max_omega).
    """
    nu = 1.0 / Re
    solver = NS3DSolver(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, nu=nu)
    
    state = NSState3D(u=u0, v=v0, w=w0, t=0.0, step=0)
    
    n_steps = int(T / dt)
    max_omega = 0.0
    
    for step in range(n_steps):
        # Get vorticity max
        u_np, v_np, w_np = state.u.numpy(), state.v.numpy(), state.w.numpy()
        
        dx = L / N
        k = np.fft.fftfreq(N, dx) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        
        u_hat = np.fft.fftn(u_np)
        v_hat = np.fft.fftn(v_np)
        w_hat = np.fft.fftn(w_np)
        
        omega_x = np.fft.ifftn(1j * ky * w_hat - 1j * kz * v_hat).real
        omega_y = np.fft.ifftn(1j * kz * u_hat - 1j * kx * w_hat).real
        omega_z = np.fft.ifftn(1j * kx * v_hat - 1j * ky * u_hat).real
        
        omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        max_omega = max(max_omega, omega_mag.max())
        
        if np.isnan(max_omega) or max_omega > 1e8:
            return float('inf'), max_omega
        
        state, _ = solver.step_rk4(state, dt)
        
        # Dealiasing
        if step % 5 == 0:
            state = NSState3D(
                u=apply_dealiasing_torch(state.u, N),
                v=apply_dealiasing_torch(state.v, N),
                w=apply_dealiasing_torch(state.w, N),
                t=state.t, step=state.step
            )
    
    final_enstrophy = compute_enstrophy_torch(state.u, state.v, state.w, L, N).item()
    return final_enstrophy, max_omega


def gradient_ascent_hunt(N: int = 32, Re: float = 5000, T: float = 1.0,
                         n_iterations: int = 20, n_random_starts: int = 5,
                         verbose: bool = True) -> OptimizationResult:
    """
    Use gradient-free optimization to find high-enstrophy ICs.
    
    Strategy: Random search + local perturbations
    """
    L = 2 * np.pi
    dt = 0.002
    
    if verbose:
        print("\n" + "═" * 70)
        print("GRADIENT ASCENT IC OPTIMIZATION")
        print("═" * 70)
        print(f"\nN = {N}, Re = {Re}, T = {T}")
        print(f"Searching for ICs that maximize enstrophy growth...\n")
    
    best_enstrophy = 0.0
    best_max_omega = 0.0
    best_ic = None
    blowup_detected = False
    
    initial_enstrophy = None
    
    for start in range(n_random_starts):
        if verbose:
            print(f"Random start {start + 1}/{n_random_starts}")
        
        # Random starting IC
        u, v, w = create_random_ic(N, L, seed=start * 1000)
        
        # Evaluate
        enstrophy, max_omega = simulate_and_measure(u, v, w, N, L, Re, T, dt)
        
        if initial_enstrophy is None:
            initial_enstrophy = enstrophy
        
        if verbose:
            print(f"  Initial: enstrophy = {enstrophy:.4f}, max|ω| = {max_omega:.4f}")
        
        if np.isinf(enstrophy):
            blowup_detected = True
            if verbose:
                print(f"  ⚠️ BLOWUP DETECTED!")
            best_ic = (u.numpy(), v.numpy(), w.numpy())
            best_enstrophy = float('inf')
            best_max_omega = max_omega
            break
        
        # Local search: perturb and keep if better
        for it in range(n_iterations):
            # Small random perturbation
            du = torch.randn_like(u) * 0.1
            dv = torch.randn_like(v) * 0.1
            dw = torch.randn_like(w) * 0.1
            
            u_new = u + du
            v_new = v + dv
            w_new = w + dw
            
            # Project and normalize
            u_new, v_new, w_new = project_divergence_free_torch(u_new, v_new, w_new, L, N)
            energy = (u_new**2 + v_new**2 + w_new**2).mean()
            scale = 1.0 / torch.sqrt(energy + 1e-10)
            u_new, v_new, w_new = u_new * scale, v_new * scale, w_new * scale
            
            # Evaluate
            new_enstrophy, new_max_omega = simulate_and_measure(u_new, v_new, w_new, N, L, Re, T, dt)
            
            if np.isinf(new_enstrophy):
                blowup_detected = True
                if verbose:
                    print(f"  ⚠️ BLOWUP at iteration {it}!")
                best_ic = (u_new.numpy(), v_new.numpy(), w_new.numpy())
                best_enstrophy = float('inf')
                best_max_omega = new_max_omega
                break
            
            # Keep if better
            if new_enstrophy > enstrophy:
                u, v, w = u_new, v_new, w_new
                enstrophy = new_enstrophy
                max_omega = new_max_omega
        
        if blowup_detected:
            break
        
        if enstrophy > best_enstrophy:
            best_enstrophy = enstrophy
            best_max_omega = max_omega
            best_ic = (u.numpy(), v.numpy(), w.numpy())
        
        if verbose:
            print(f"  Final: enstrophy = {enstrophy:.4f}, max|ω| = {max_omega:.4f}")
    
    if initial_enstrophy is None:
        initial_enstrophy = 1.0
    
    improvement = best_enstrophy / initial_enstrophy if not np.isinf(best_enstrophy) else float('inf')
    
    if verbose:
        print(f"\n{'═' * 70}")
        print("OPTIMIZATION RESULT")
        print(f"{'═' * 70}")
        print(f"  Best enstrophy: {best_enstrophy:.4f}")
        print(f"  Best max|ω|: {best_max_omega:.4f}")
        print(f"  Improvement: {improvement:.2f}x")
        print(f"  Blowup detected: {blowup_detected}")
    
    return OptimizationResult(
        iterations=n_iterations * n_random_starts,
        initial_enstrophy_growth=initial_enstrophy,
        final_enstrophy_growth=best_enstrophy,
        improvement_factor=improvement,
        max_omega_achieved=best_max_omega,
        blowup_detected=blowup_detected,
        best_ic=best_ic
    )


def run_level2_hunt():
    """Run the Level 2 Black Swan hunt."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "BLACK SWAN HUNT: LEVEL 2" + " " * 37 + "║")
    print("║" + " " * 78 + "║")
    print("║  Strategy: Gradient-free IC optimization to maximize enstrophy" + " " * 11 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Run optimization at different Reynolds numbers
    results = {}
    
    for Re in [1000, 5000, 10000]:
        print(f"\n{'═' * 70}")
        print(f"Reynolds number: Re = {Re}")
        print(f"{'═' * 70}")
        
        result = gradient_ascent_hunt(
            N=32,
            Re=Re,
            T=1.5,
            n_iterations=15,
            n_random_starts=3,
            verbose=True
        )
        results[Re] = result
        
        if result.blowup_detected:
            print(f"\n🔥 POTENTIAL BLACK SWAN at Re = {Re}!")
            break
    
    # Summary
    print("\n" + "═" * 70)
    print("LEVEL 2 HUNT SUMMARY")
    print("═" * 70)
    
    any_blowup = any(r.blowup_detected for r in results.values())
    
    if any_blowup:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🔥 BLACK SWAN CANDIDATE FOUND!                             ║
║                                                                              ║
║  Optimization found an IC that causes apparent blowup.                      ║
║  Requires verification with higher resolution.                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    else:
        max_omega = max(r.max_omega_achieved for r in results.values())
        max_improvement = max(r.improvement_factor for r in results.values())
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    ✓ NO BLACK SWAN FOUND                                     ║
║                                                                              ║
║  All optimized ICs remained bounded.                                        ║
║  Max vorticity achieved: {max_omega:.2f}                                           ║
║  Max enstrophy improvement: {max_improvement:.2f}x                                      ║
║                                                                              ║
║  CONCLUSION: Evidence continues to support NS regularity                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    return results


if __name__ == "__main__":
    results = run_level2_hunt()
