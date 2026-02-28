#!/usr/bin/env python3
"""
KIDA VORTEX CONVERGENCE STUDY

The Kida vortex showed explosive growth - is it:
1. A REAL singularity (physics)
2. Numerical instability (artifact)

Test: Run at multiple resolutions. If it's real physics:
- Blowup time should CONVERGE as N increases
- Growth rate should stabilize

If it's numerical instability:
- Blowup time depends on N
- Higher resolution = later/no blowup
"""

import numpy as np
import torch
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from ontic.cfd.ns_3d import NS3DSolver, NSState3D


def create_kida_vortex_ic(N: int, L: float = 2*np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kida vortex - high symmetry, proposed blowup candidate."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    u = np.sin(X) * (np.cos(3*Y) * np.cos(Z) - np.cos(Y) * np.cos(3*Z))
    v = np.sin(Y) * (np.cos(3*Z) * np.cos(X) - np.cos(Z) * np.cos(3*X))
    w = np.sin(Z) * (np.cos(3*X) * np.cos(Y) - np.cos(X) * np.cos(3*Y))
    
    energy = (u**2 + v**2 + w**2).mean()
    scale = 1.0 / np.sqrt(energy + 1e-10)
    
    return u * scale, v * scale, w * scale


def compute_vorticity(u, v, w, L, N):
    """Compute vorticity using spectral derivatives."""
    dx = L / N
    k = np.fft.fftfreq(N, dx) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)
    
    omega_x = np.fft.ifftn(1j * ky * w_hat - 1j * kz * v_hat).real
    omega_y = np.fft.ifftn(1j * kz * u_hat - 1j * kx * w_hat).real
    omega_z = np.fft.ifftn(1j * kx * v_hat - 1j * ky * u_hat).real
    
    return np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)


def run_convergence_study():
    """Run convergence study at multiple resolutions."""
    print("╔" + "═" * 70 + "╗")
    print("║" + " " * 15 + "KIDA VORTEX CONVERGENCE STUDY" + " " * 24 + "║")
    print("╚" + "═" * 70 + "╝")
    
    # Test parameters
    resolutions = [32, 48, 64]
    Re = 10000
    L = 2 * np.pi
    T_final = 2.0
    
    results = {}
    
    for N in resolutions:
        print(f"\n{'═' * 60}")
        print(f"Resolution: N = {N}")
        print(f"{'═' * 60}")
        
        nu = 1.0 / Re
        
        # CFL-adaptive time step
        dx = L / N
        u_max_est = 10.0  # Estimate
        dt = 0.1 * dx / u_max_est  # CFL ~ 0.1
        dt = min(dt, 0.001)  # Cap
        
        print(f"  dt = {dt:.6f} (CFL-adaptive)")
        
        # Initialize
        solver = NS3DSolver(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, nu=nu)
        u0, v0, w0 = create_kida_vortex_ic(N, L)
        
        state = NSState3D(
            u=torch.from_numpy(u0).double(),
            v=torch.from_numpy(v0).double(),
            w=torch.from_numpy(w0).double(),
            t=0.0, step=0
        )
        
        # Track evolution
        times = []
        omega_max_vals = []
        enstrophy_vals = []
        blowup_time = None
        
        n_steps = int(T_final / dt)
        print_interval = max(1, n_steps // 10)
        
        for step in range(n_steps):
            t = state.t
            
            u_np = state.u.numpy()
            v_np = state.v.numpy()
            w_np = state.w.numpy()
            
            omega_mag = compute_vorticity(u_np, v_np, w_np, L, N)
            omega_max = omega_mag.max()
            enstrophy = 0.5 * (omega_mag**2).mean() * L**3
            
            times.append(t)
            omega_max_vals.append(omega_max)
            enstrophy_vals.append(enstrophy)
            
            # Check for NaN or blowup
            if np.isnan(omega_max) or omega_max > 1e8:
                blowup_time = t
                print(f"  ⚠️  BLOWUP at t = {t:.4f}: max|ω| = {omega_max:.2e}")
                break
            
            # Progress
            if step % print_interval == 0:
                print(f"  t = {t:.3f}: max|ω| = {omega_max:.4f}, enstrophy = {enstrophy:.4f}")
            
            # Step
            try:
                state, _ = solver.step_rk4(state, dt)
            except Exception as e:
                print(f"  Solver failed: {e}")
                blowup_time = t
                break
        
        results[N] = {
            "times": times,
            "omega_max": omega_max_vals,
            "enstrophy": enstrophy_vals,
            "blowup_time": blowup_time,
            "final_omega_max": omega_max_vals[-1] if omega_max_vals else 0
        }
        
        if blowup_time is None:
            print(f"\n  ✓ BOUNDED: max|ω| = {max(omega_max_vals):.4f}")
        else:
            print(f"\n  ⚠️  BLOWUP at t = {blowup_time:.4f}")
    
    # Analysis
    print("\n" + "═" * 70)
    print("CONVERGENCE ANALYSIS")
    print("═" * 70)
    
    print("\n  N     | Blowup Time | Final max|ω|")
    print("  ------+--------------+-------------")
    
    for N in resolutions:
        r = results[N]
        bt = f"{r['blowup_time']:.4f}" if r['blowup_time'] else "N/A (bounded)"
        fo = f"{r['final_omega_max']:.4f}" if not np.isnan(r['final_omega_max']) else "NaN"
        print(f"  {N:4d}  | {bt:12s} | {fo:11s}")
    
    # Verdict
    blowup_times = [results[N]['blowup_time'] for N in resolutions if results[N]['blowup_time'] is not None]
    
    if len(blowup_times) == len(resolutions):
        # All resolutions blew up
        if len(set([round(t, 2) for t in blowup_times])) == 1:
            print("\n  → CONVERGENT BLOWUP: Same time at all N")
            print("  → This suggests REAL SINGULARITY FORMATION!")
        else:
            print("\n  → DIVERGENT BLOWUP: Time varies with N")
            print("  → This suggests NUMERICAL INSTABILITY")
    elif len(blowup_times) == 0:
        print("\n  → ALL BOUNDED: No blowup at any resolution")
        print("  → Evidence for REGULARITY")
    else:
        print(f"\n  → MIXED: {len(blowup_times)}/{len(resolutions)} resolutions blew up")
        print("  → Likely NUMERICAL INSTABILITY (resolution-dependent)")


if __name__ == "__main__":
    run_convergence_study()
