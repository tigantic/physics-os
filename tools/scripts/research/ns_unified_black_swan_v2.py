#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                    UNIFIED 3D BLACK SWAN SINGULARITY HUNTER v2                       ║
║                                                                                      ║
║                    Using Validated NS3DSolver + QTT χ Tracking                       ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  DUAL METRICS:                                                                       ║
║  ─────────────                                                                       ║
║  1. BKM CRITERION: ∫₀^T ‖ω‖_∞ dt → ∞  implies blowup                                ║
║  2. QTT COMPRESSION: χ(t) → ∞  implies loss of regularity                           ║
║                                                                                      ║
║  SOLVER: Validated NS3DSolver with RK4 + spectral projection                        ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import json
import time
import math

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ontic.cfd.ns_3d import NS3DSolver, NSState3D
from ontic.cfd.pure_qtt_ops import dense_to_qtt


# ═══════════════════════════════════════════════════════════════════════════════════════
# DUAL METRIC TRACKER
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class DualMetrics:
    """Combined BKM + QTT metrics."""
    time: float
    omega_max: float
    bkm_integral: float
    enstrophy: float
    chi_max: int
    chi_mean: float
    energy: float


class DualTracker:
    """Track both BKM criterion and QTT bond dimension."""
    
    def __init__(self, bkm_threshold: float = 1e6, chi_threshold: int = 128):
        self.bkm_threshold = bkm_threshold
        self.chi_threshold = chi_threshold
        self.history: List[DualMetrics] = []
        self.bkm_integral = 0.0
    
    def update(self, t: float, dt: float, omega_max: float, enstrophy: float,
               chi_max: int, chi_mean: float, energy: float):
        self.bkm_integral += omega_max * dt
        metrics = DualMetrics(
            time=t, omega_max=omega_max, bkm_integral=self.bkm_integral,
            enstrophy=enstrophy, chi_max=chi_max, chi_mean=chi_mean, energy=energy
        )
        self.history.append(metrics)
        return metrics
    
    def check_bkm_blowup(self) -> Tuple[bool, Optional[float]]:
        if self.bkm_integral > self.bkm_threshold:
            return True, self.history[-1].time if self.history else None
        if len(self.history) > 20:
            early = np.mean([m.omega_max for m in self.history[:10]])
            late = np.mean([m.omega_max for m in self.history[-10:]])
            if late > 50 * early and late > 1000:
                return True, self.history[-1].time
        return False, None
    
    def check_chi_blowup(self) -> Tuple[bool, Optional[float]]:
        if not self.history:
            return False, None
        if self.history[-1].chi_max > self.chi_threshold:
            return True, self.history[-1].time
        if len(self.history) > 10:
            early = np.mean([m.chi_max for m in self.history[:5]])
            late = np.mean([m.chi_max for m in self.history[-5:]])
            if late > 3 * early and late > 50:
                return True, self.history[-1].time
        return False, None
    
    def get_verdict(self) -> Tuple[str, bool]:
        bkm_blow, _ = self.check_bkm_blowup()
        chi_blow, _ = self.check_chi_blowup()
        if bkm_blow and chi_blow:
            return "🔥 DUAL BLOWUP", True
        elif bkm_blow:
            return "⚠️  BKM BLOWUP", True
        elif chi_blow:
            return "⚠️  χ BLOWUP", True
        return "✓ BOUNDED", False


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT COMPRESSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════════

def compute_qtt_metrics(field: torch.Tensor, max_rank: int = 64) -> Tuple[int, float]:
    """Compute QTT compression metrics for a 3D field."""
    if not torch.isfinite(field).all():
        return max_rank, float(max_rank)
    
    flat = field.flatten()
    n = len(flat)
    n_qubits = int(np.ceil(np.log2(n)))
    n_padded = 2 ** n_qubits
    
    if n_padded > n:
        flat = torch.cat([flat, torch.zeros(n_padded - n, dtype=flat.dtype)])
    
    try:
        qtt = dense_to_qtt(flat, max_bond=max_rank)
        chi_max = max(c.shape[0] for c in qtt.cores)
        chi_mean = np.mean([c.shape[0] for c in qtt.cores])
    except Exception:
        chi_max, chi_mean = max_rank, float(max_rank)
    
    return chi_max, chi_mean


def compute_vorticity_torch(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                            dx: float, dy: float, dz: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute vorticity ω = ∇ × u using spectral method."""
    Nx, Ny, Nz = u.shape
    
    kx = torch.fft.fftfreq(Nx, d=dx, device=u.device, dtype=u.dtype) * 2 * math.pi
    ky = torch.fft.fftfreq(Ny, d=dy, device=u.device, dtype=u.dtype) * 2 * math.pi
    kz = torch.fft.fftfreq(Nz, d=dz, device=u.device, dtype=u.dtype) * 2 * math.pi
    
    KX = kx.reshape(Nx, 1, 1)
    KY = ky.reshape(1, Ny, 1)
    KZ = kz.reshape(1, 1, Nz)
    
    u_hat = torch.fft.fftn(u)
    v_hat = torch.fft.fftn(v)
    w_hat = torch.fft.fftn(w)
    
    # ω_x = ∂w/∂y - ∂v/∂z
    omega_x = torch.fft.ifftn(1j * KY * w_hat - 1j * KZ * v_hat).real
    # ω_y = ∂u/∂z - ∂w/∂x
    omega_y = torch.fft.ifftn(1j * KZ * u_hat - 1j * KX * w_hat).real
    # ω_z = ∂v/∂x - ∂u/∂y
    omega_z = torch.fft.ifftn(1j * KX * v_hat - 1j * KY * u_hat).real
    
    return omega_x, omega_y, omega_z


# ═══════════════════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_simulation(solver: NS3DSolver, state: NSState3D, 
                   T_final: float, dt: float, ic_name: str,
                   max_rank: int = 64, track_interval: int = 20) -> Dict:
    """Run simulation with dual BKM + QTT tracking."""
    
    tracker = DualTracker(bkm_threshold=1e6, chi_threshold=max_rank * 2)
    
    n_steps = int(T_final / dt)
    
    print(f"\n  {ic_name}: N={solver.Nx}, Re={1/solver.nu:.0f}, T={T_final}")
    
    start_time = time.time()
    
    for step in range(n_steps + 1):
        t = state.t
        
        if step % track_interval == 0:
            # Compute vorticity
            ox, oy, oz = compute_vorticity_torch(
                state.u, state.v, state.w, solver.dx, solver.dy, solver.dz
            )
            omega_mag = torch.sqrt(ox**2 + oy**2 + oz**2)
            omega_max = omega_mag.max().item()
            enstrophy = 0.5 * (omega_mag**2).mean().item() * solver.Lx**3
            
            # Energy
            energy = 0.5 * (state.u**2 + state.v**2 + state.w**2).mean().item() * solver.Lx**3
            
            # QTT compression of velocity magnitude
            vel_mag = torch.sqrt(state.u**2 + state.v**2 + state.w**2)
            chi_max, chi_mean = compute_qtt_metrics(vel_mag, max_rank)
            
            # Update tracker
            metrics = tracker.update(t, dt * track_interval, omega_max, enstrophy,
                                     chi_max, chi_mean, energy)
            
            # Progress
            if step % (n_steps // 5) == 0 or step == 0:
                print(f"    t={t:.3f}: |ω|_max={omega_max:.2f}, BKM={metrics.bkm_integral:.1f}, "
                      f"χ={chi_max}, E={energy:.4f}")
            
            # Check for physics blowup
            verdict, blowup = tracker.get_verdict()
            if blowup:
                print(f"\n    {verdict}")
                break
        
        if step < n_steps:
            # RK4 step with the validated solver
            state = solver.step_rk4(state, dt)
            
            # Check for numerical instability
            if not torch.isfinite(state.u).all():
                print(f"\n    ⚠️  NUMERICAL INSTABILITY at t={state.t:.4f}")
                break
    
    runtime = time.time() - start_time
    verdict, blowup = tracker.get_verdict()
    
    final = tracker.history[-1] if tracker.history else None
    
    return {
        "ic_name": ic_name,
        "verdict": verdict,
        "blowup": blowup,
        "bkm_integral": final.bkm_integral if final else 0,
        "chi_max": final.chi_max if final else 0,
        "omega_max": final.omega_max if final else 0,
        "energy_final": final.energy if final else 0,
        "runtime": runtime,
        "t_final": final.time if final else 0
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_kida_ic(solver: NS3DSolver) -> NSState3D:
    """Kida vortex - symmetric blowup candidate."""
    X, Y, Z = solver.X, solver.Y, solver.Z
    
    u = torch.sin(X) * (torch.cos(3*Y) * torch.cos(Z) - torch.cos(Y) * torch.cos(3*Z))
    v = torch.sin(Y) * (torch.cos(3*Z) * torch.cos(X) - torch.cos(Z) * torch.cos(3*X))
    w = torch.sin(Z) * (torch.cos(3*X) * torch.cos(Y) - torch.cos(X) * torch.cos(3*Y))
    
    # Normalize to unit energy
    energy = (u**2 + v**2 + w**2).mean()
    scale = 1.0 / torch.sqrt(energy + 1e-10)
    
    return NSState3D(u=u*scale, v=v*scale, w=w*scale, t=0.0, step=0)


def create_abc_ic(solver: NS3DSolver, A: float = 1.0, B: float = 1.0, C: float = 1.0) -> NSState3D:
    """Arnold-Beltrami-Childress flow - exact Euler solution."""
    X, Y, Z = solver.X, solver.Y, solver.Z
    
    u = A * torch.sin(Z) + C * torch.cos(Y)
    v = B * torch.sin(X) + A * torch.cos(Z)
    w = C * torch.sin(Y) + B * torch.cos(X)
    
    return NSState3D(u=u, v=v, w=w, t=0.0, step=0)


def create_shear_layer_ic(solver: NS3DSolver, delta: float = 0.1) -> NSState3D:
    """Perturbed shear layer - rollup dynamics."""
    X, Y, Z = solver.X, solver.Y, solver.Z
    L = solver.Lx
    
    # Shear in y-direction
    y_centered = Y - L/2
    u = torch.tanh(y_centered / (delta * L))
    
    # Perturbation
    u = u + 0.05 * torch.sin(2 * math.pi * X / L) * torch.exp(-y_centered**2 / (2*(delta*L)**2))
    
    v = torch.zeros_like(u)
    w = torch.zeros_like(u)
    
    return NSState3D(u=u, v=v, w=w, t=0.0, step=0)


def create_vortex_collision_ic(solver: NS3DSolver) -> NSState3D:
    """Two colliding vortex rings."""
    X, Y, Z = solver.X, solver.Y, solver.Z
    L = solver.Lx
    
    # Center coordinates
    Xc, Yc, Zc = X - L/2, Y - L/2, Z - L/2
    
    # Ring 1 at z = -L/6, Ring 2 at z = +L/6
    z1, z2 = -L/6, L/6
    R = L/4  # Ring radius
    a = L/15  # Core size
    
    r = torch.sqrt(Xc**2 + Yc**2)
    theta = torch.atan2(Yc, Xc)
    
    # Distance from ring cores
    r1 = torch.sqrt((r - R)**2 + (Zc - z1)**2)
    r2 = torch.sqrt((r - R)**2 + (Zc - z2)**2)
    
    # Azimuthal vorticity (opposite signs)
    omega1 = torch.exp(-r1**2 / a**2)
    omega2 = -torch.exp(-r2**2 / a**2)
    
    # Simple velocity approximation
    u = -torch.sin(theta) * (omega1 + omega2) * 0.2
    v = torch.cos(theta) * (omega1 + omega2) * 0.2
    w = 0.3 * (torch.exp(-r1**2 / a**2) - torch.exp(-r2**2 / a**2))
    
    return NSState3D(u=u, v=v, w=w, t=0.0, step=0)


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN HUNT
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_unified_hunt_v2():
    """Run unified Black Swan hunt using validated NS3DSolver."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 10 + "UNIFIED 3D BLACK SWAN HUNTER v2" + " " * 35 + "║")
    print("║" + " " * 78 + "║")
    print("║  Using validated NS3DSolver with RK4 + spectral projection" + " " * 17 + "║")
    print("║  Tracking: BKM integral + QTT bond dimension χ" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Configuration
    N = 32
    L = 2 * math.pi
    Re = 1000
    nu = 1.0 / Re
    T_final = 5.0
    dt = 0.01
    max_rank = 64
    
    print(f"\n  Grid: N={N}³, Re={Re}, T={T_final}, dt={dt}")
    
    # Create solver
    solver = NS3DSolver(
        Nx=N, Ny=N, Nz=N,
        Lx=L, Ly=L, Lz=L,
        nu=nu,
        dtype=torch.float64
    )
    
    # ICs to test
    ics = [
        ("Taylor-Green (baseline)", lambda s: s.create_taylor_green_3d(A=1.0)),
        ("Kida Vortex", create_kida_ic),
        ("ABC Flow", lambda s: create_abc_ic(s, A=1.0, B=1.0, C=1.0)),
        ("Shear Layer", lambda s: create_shear_layer_ic(s, delta=0.1)),
        ("Vortex Collision", create_vortex_collision_ic),
    ]
    
    results = {}
    
    for ic_name, ic_func in ics:
        print("\n" + "═" * 70)
        
        # Create IC
        state = ic_func(solver)
        
        # Run simulation
        result = run_simulation(solver, state, T_final, dt, ic_name, 
                               max_rank=max_rank, track_interval=25)
        
        results[ic_name] = result
        print(f"\n  → {result['verdict']} (T={result['t_final']:.2f})")
    
    # Summary
    print("\n" + "═" * 80)
    print("UNIFIED BLACK SWAN HUNT v2: SUMMARY")
    print("═" * 80)
    print(f"\n  {'IC':<25} {'BKM':>10} {'χ_max':>8} {'|ω|_max':>10} {'E_final':>10} {'Verdict':>12}")
    print("  " + "-" * 78)
    
    for ic_name, r in results.items():
        short_name = ic_name[:25]
        verdict_short = "BLOWUP" if r["blowup"] else "BOUNDED"
        print(f"  {short_name:<25} {r['bkm_integral']:>10.1f} {r['chi_max']:>8} "
              f"{r['omega_max']:>10.2f} {r['energy_final']:>10.4f} {verdict_short:>12}")
    
    n_blowup = sum(1 for r in results.values() if r["blowup"])
    
    print("\n" + "═" * 80)
    if n_blowup > 0:
        print("  🔥 SINGULARITY CANDIDATE(S) FOUND!")
    else:
        print("  ✓ ALL ICs BOUNDED")
        print("    → BKM integral finite, QTT rank stable")
        print("    → Evidence supports NS regularity")
    print("═" * 80)
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"N": N, "Re": Re, "T_final": T_final, "dt": dt, "max_rank": max_rank},
        "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        "summary": {"total_ics": len(results), "blowups": n_blowup}
    }
    
    with open("unified_black_swan_v2_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to unified_black_swan_v2_results.json")
    
    return results


if __name__ == "__main__":
    run_unified_hunt_v2()
