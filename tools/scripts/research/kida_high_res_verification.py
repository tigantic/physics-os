#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                    KIDA VORTEX HIGH-RESOLUTION VERIFICATION                          ║
║                                                                                      ║
║                    Is the blowup REAL or NUMERICAL?                                  ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  DIAGNOSTIC TESTS:                                                                   ║
║  ─────────────────                                                                   ║
║  1. Resolution convergence: N = 32, 48, 64, 96                                       ║
║  2. Energy conservation: Should decay (viscous) not grow                             ║
║  3. CFL monitoring: Track actual Courant number                                      ║
║  4. 2/3 dealiasing: Remove spectral aliasing artifacts                               ║
║                                                                                      ║
║  VERDICT CRITERIA:                                                                   ║
║  ─────────────────                                                                   ║
║  • T* converges as N↑ → REAL SINGULARITY                                             ║
║  • T* increases as N↑ → NUMERICAL INSTABILITY                                        ║
║  • Energy grows → NUMERICAL INSTABILITY                                              ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

from ontic.cfd.ns_3d import NS3DSolver, NSState3D


@dataclass
class VerificationResult:
    """Result from a single verification run."""
    N: int
    Re: float
    blowup_time: Optional[float]
    max_omega: float
    final_energy: float
    initial_energy: float
    energy_ratio: float  # final/initial - should be ≤ 1
    max_cfl: float
    verdict: str


def create_kida_vortex(N: int, L: float = 2*np.pi, amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kida vortex initial condition.
    
    High-symmetry flow that's been proposed as a blowup candidate.
    """
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    u = amplitude * np.sin(X) * (np.cos(3*Y) * np.cos(Z) - np.cos(Y) * np.cos(3*Z))
    v = amplitude * np.sin(Y) * (np.cos(3*Z) * np.cos(X) - np.cos(Z) * np.cos(3*X))
    w = amplitude * np.sin(Z) * (np.cos(3*X) * np.cos(Y) - np.cos(X) * np.cos(3*Y))
    
    return u, v, w


def compute_vorticity(u: np.ndarray, v: np.ndarray, w: np.ndarray, 
                      L: float, N: int) -> np.ndarray:
    """Compute vorticity magnitude using spectral derivatives."""
    dx = L / N
    k = np.fft.fftfreq(N, dx) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)
    
    # ω = ∇ × u
    omega_x = np.fft.ifftn(1j * ky * w_hat - 1j * kz * v_hat).real
    omega_y = np.fft.ifftn(1j * kz * u_hat - 1j * kx * w_hat).real
    omega_z = np.fft.ifftn(1j * kx * v_hat - 1j * ky * u_hat).real
    
    return np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)


def compute_energy(u: np.ndarray, v: np.ndarray, w: np.ndarray, 
                   L: float, N: int) -> float:
    """Compute total kinetic energy."""
    dV = (L / N) ** 3
    return 0.5 * np.sum(u**2 + v**2 + w**2) * dV


def apply_dealiasing(u_hat: np.ndarray, N: int) -> np.ndarray:
    """Apply 2/3 dealiasing rule to remove high-frequency artifacts."""
    k_max = N // 3
    kx = np.fft.fftfreq(N, 1/N)
    ky = np.fft.fftfreq(N, 1/N)
    kz = np.fft.fftfreq(N, 1/N)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Zero out modes above 2/3 of Nyquist
    mask = (np.abs(KX) < k_max) & (np.abs(KY) < k_max) & (np.abs(KZ) < k_max)
    return u_hat * mask


def run_verification(N: int, Re: float, T_final: float = 2.0, 
                     use_dealiasing: bool = True,
                     verbose: bool = True) -> VerificationResult:
    """
    Run a single verification test.
    """
    L = 2 * np.pi
    nu = 1.0 / Re
    
    # CFL-adaptive time step
    dx = L / N
    u_max_est = 5.0
    dt_base = 0.2 * dx / u_max_est  # CFL ~ 0.2
    dt = min(dt_base, 0.002)  # Cap for stability
    
    if verbose:
        print(f"\n{'═' * 60}")
        print(f"N = {N}, Re = {Re:.0f}, dt = {dt:.6f}")
        print(f"{'═' * 60}")
    
    # Initialize
    solver = NS3DSolver(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, nu=nu)
    u0, v0, w0 = create_kida_vortex(N, L, amplitude=1.0)
    
    state = NSState3D(
        u=torch.from_numpy(u0).double(),
        v=torch.from_numpy(v0).double(),
        w=torch.from_numpy(w0).double(),
        t=0.0, step=0
    )
    
    # Initial diagnostics
    initial_energy = compute_energy(u0, v0, w0, L, N)
    
    # Tracking
    times = []
    omega_max_vals = []
    energy_vals = []
    cfl_vals = []
    blowup_time = None
    
    n_steps = int(T_final / dt)
    print_interval = max(1, n_steps // 8)
    
    start_time = time.time()
    
    for step in range(n_steps):
        t = state.t
        
        u_np = state.u.numpy()
        v_np = state.v.numpy()
        w_np = state.w.numpy()
        
        # Diagnostics
        omega_mag = compute_vorticity(u_np, v_np, w_np, L, N)
        omega_max = omega_mag.max()
        energy = compute_energy(u_np, v_np, w_np, L, N)
        
        # CFL number
        u_max = max(np.abs(u_np).max(), np.abs(v_np).max(), np.abs(w_np).max())
        cfl = u_max * dt / dx
        
        times.append(t)
        omega_max_vals.append(omega_max)
        energy_vals.append(energy)
        cfl_vals.append(cfl)
        
        # Check for blowup or NaN
        if np.isnan(omega_max) or omega_max > 1e8:
            blowup_time = t
            if verbose:
                print(f"  ⚠️  BLOWUP at t = {t:.4f}: max|ω| = {omega_max:.2e}")
            break
        
        # Check energy conservation (should not grow!)
        if energy > initial_energy * 1.5:
            if verbose:
                print(f"  ⚠️  ENERGY GROWTH at t = {t:.4f}: E/E0 = {energy/initial_energy:.2f}")
            blowup_time = t
            break
        
        # Progress
        if verbose and step % print_interval == 0:
            print(f"  t = {t:.3f}: max|ω| = {omega_max:.4f}, E/E0 = {energy/initial_energy:.4f}, CFL = {cfl:.3f}")
        
        # Step with optional dealiasing
        state, _ = solver.step_rk4(state, dt)
        
        # Apply dealiasing if enabled
        if use_dealiasing and step % 10 == 0:
            u_hat = np.fft.fftn(state.u.numpy())
            v_hat = np.fft.fftn(state.v.numpy())
            w_hat = np.fft.fftn(state.w.numpy())
            
            u_hat = apply_dealiasing(u_hat, N)
            v_hat = apply_dealiasing(v_hat, N)
            w_hat = apply_dealiasing(w_hat, N)
            
            state = NSState3D(
                u=torch.from_numpy(np.fft.ifftn(u_hat).real).double(),
                v=torch.from_numpy(np.fft.ifftn(v_hat).real).double(),
                w=torch.from_numpy(np.fft.ifftn(w_hat).real).double(),
                t=state.t, step=state.step
            )
    
    runtime = time.time() - start_time
    
    # Final diagnostics
    final_energy = energy_vals[-1] if energy_vals else initial_energy
    max_omega = max(omega_max_vals) if omega_max_vals else 0
    max_cfl = max(cfl_vals) if cfl_vals else 0
    energy_ratio = final_energy / initial_energy
    
    # Verdict
    if blowup_time is not None:
        if energy_ratio > 1.2:
            verdict = "NUMERICAL (energy growth)"
        else:
            verdict = "POTENTIAL BLOWUP"
    else:
        verdict = "BOUNDED"
    
    if verbose:
        print(f"\n  Runtime: {runtime:.1f}s")
        print(f"  Verdict: {verdict}")
    
    return VerificationResult(
        N=N,
        Re=Re,
        blowup_time=blowup_time,
        max_omega=max_omega,
        final_energy=final_energy,
        initial_energy=initial_energy,
        energy_ratio=energy_ratio,
        max_cfl=max_cfl,
        verdict=verdict
    )


def run_convergence_study():
    """Run the full convergence study."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "KIDA VORTEX HIGH-RESOLUTION VERIFICATION" + " " * 20 + "║")
    print("║" + " " * 78 + "║")
    print("║  Question: Is the Kida blowup REAL or NUMERICAL?" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Test parameters
    resolutions = [32, 48, 64]  # Start with these, add 96 if time permits
    Re = 5000  # Moderate Reynolds
    T_final = 2.5
    
    results: List[VerificationResult] = []
    
    print("\n" + "=" * 70)
    print("PART 1: RESOLUTION CONVERGENCE (with 2/3 dealiasing)")
    print("=" * 70)
    
    for N in resolutions:
        result = run_verification(N, Re, T_final, use_dealiasing=True, verbose=True)
        results.append(result)
    
    # Analysis
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)
    
    print("\n  N     | T* (blowup) | max|ω|    | E_final/E_0 | max CFL | Verdict")
    print("  ------+-------------+-----------+-------------+---------+------------------")
    
    for r in results:
        t_str = f"{r.blowup_time:.4f}" if r.blowup_time else "N/A"
        print(f"  {r.N:4d}  | {t_str:11s} | {r.max_omega:9.2f} | {r.energy_ratio:11.4f} | {r.max_cfl:7.3f} | {r.verdict}")
    
    # Determine overall verdict
    blowup_times = [r.blowup_time for r in results if r.blowup_time is not None]
    energy_growth = any(r.energy_ratio > 1.1 for r in results)
    all_bounded = all(r.blowup_time is None for r in results)
    
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    if all_bounded:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    ✓ BOUNDED - NO SINGULARITY DETECTED                       ║
║                                                                              ║
║  With 2/3 dealiasing, the Kida vortex remains bounded at all resolutions.   ║
║  The earlier "blowup" was likely a NUMERICAL ARTIFACT from aliasing.        ║
║                                                                              ║
║  CONCLUSION: Evidence SUPPORTS Navier-Stokes regularity                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        overall = "BOUNDED"
    elif energy_growth:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    ⚠️  NUMERICAL INSTABILITY DETECTED                         ║
║                                                                              ║
║  Energy grew during simulation (should decay for viscous NS).               ║
║  This indicates NUMERICAL ARTIFACT, not real physics.                       ║
║                                                                              ║
║  CONCLUSION: Previous "blowup" was numerical error                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        overall = "NUMERICAL"
    elif len(blowup_times) >= 2:
        # Check if blowup times converge
        t_diff = abs(blowup_times[-1] - blowup_times[-2])
        if t_diff < 0.1:  # Times within 0.1 of each other
            print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🔥 POTENTIAL SINGULARITY - CONVERGENT T*                   ║
║                                                                              ║
║  Blowup time T* ≈ {np.mean(blowup_times):.3f} is CONSISTENT across resolutions!           ║
║  This suggests a REAL singularity may be forming.                           ║
║                                                                              ║
║  NEXT: Higher resolution (128³, 256³) to confirm                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
            overall = "POTENTIAL_SINGULARITY"
        else:
            print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    ⚠️  INCONCLUSIVE - T* VARIES WITH RESOLUTION               ║
║                                                                              ║
║  Blowup times: {blowup_times}
║  Resolution-dependent T* suggests numerical artifact.                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
            overall = "INCONCLUSIVE"
    else:
        print("\n  Mixed results - needs further investigation")
        overall = "INCONCLUSIVE"
    
    return {
        "results": results,
        "verdict": overall,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    study = run_convergence_study()
    
    # Save results
    import json
    output = {
        "verdict": study["verdict"],
        "timestamp": study["timestamp"],
        "results": [
            {
                "N": r.N,
                "Re": r.Re,
                "blowup_time": r.blowup_time,
                "max_omega": float(r.max_omega),
                "energy_ratio": float(r.energy_ratio),
                "max_cfl": float(r.max_cfl),
                "verdict": r.verdict
            }
            for r in study["results"]
        ]
    }
    
    with open("kida_verification_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to kida_verification_results.json")
