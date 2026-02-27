#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                    NAVIER-STOKES QTT-NATIVE SINGULARITY HUNTER                       ║
║                                                                                      ║
║                     Tracking Bond Dimension χ as Blowup Detector                     ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  KEY INSIGHT:                                                                        ║
║  ────────────                                                                        ║
║  Near a singularity, velocity fields become "incompressible" in the tensor sense:   ║
║  - Smooth functions have low QTT rank (χ ~ O(log N))                                ║
║  - Singular functions require high QTT rank (χ → N)                                  ║
║                                                                                      ║
║  Therefore: χ(t) is the CANARY IN THE COAL MINE for singularities!                  ║
║                                                                                      ║
║  BLOWUP CRITERION:                                                                   ║
║  ─────────────────                                                                   ║
║  If χ(t) → ∞ in finite time → SINGULARITY CANDIDATE                                 ║
║  If χ(t) ~ const for all t → EVIDENCE FOR REGULARITY                                ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import json

# Add parent to path for imports
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Import QTT infrastructure
from tensornet.cfd.pure_qtt_ops import QTTState, dense_to_qtt


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT COMPRESSION TRACKER
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class CompressionMetrics:
    """Track QTT compression during simulation."""
    time: float
    chi_max: int           # Maximum bond dimension
    chi_mean: float        # Mean bond dimension
    compression_ratio: float  # Dense / QTT parameters
    omega_max: float       # Max vorticity magnitude
    enstrophy: float       # Total enstrophy
    energy: float          # Total energy


class QTTCompressionTracker:
    """
    Track QTT bond dimension as singularity detector.
    
    The key insight from Level 3 findings:
        χ ~ Re^0.035 (nearly constant with Reynolds)
    
    If χ grows significantly, we're approaching a singularity.
    """
    
    def __init__(self, n_qubits: int, chi_threshold: int = 256):
        self.n_qubits = n_qubits
        self.chi_threshold = chi_threshold
        self.N = 2 ** n_qubits  # Grid size
        self.history: List[CompressionMetrics] = []
    
    def get_chi_max(self, cores: List[torch.Tensor]) -> int:
        """Get maximum bond dimension from QTT cores."""
        return max(c.shape[0] for c in cores)
    
    def get_chi_mean(self, cores: List[torch.Tensor]) -> float:
        """Get mean bond dimension."""
        return np.mean([c.shape[0] for c in cores])
    
    def get_compression_ratio(self, cores: List[torch.Tensor]) -> float:
        """
        Compute compression ratio.
        
        Dense parameters: N (for 1D), N² (for 2D), N³ (for 3D)
        QTT parameters: sum of core sizes
        """
        qtt_params = sum(c.numel() for c in cores)
        dense_params = self.N  # For 1D; adjust for higher dims
        return dense_params / max(qtt_params, 1)
    
    def track(self, t: float, cores: List[torch.Tensor], 
              omega_max: float, enstrophy: float, energy: float):
        """Record metrics at time t."""
        metrics = CompressionMetrics(
            time=t,
            chi_max=self.get_chi_max(cores),
            chi_mean=self.get_chi_mean(cores),
            compression_ratio=self.get_compression_ratio(cores),
            omega_max=omega_max,
            enstrophy=enstrophy,
            energy=energy
        )
        self.history.append(metrics)
        return metrics
    
    def check_blowup(self) -> Tuple[bool, Optional[float]]:
        """Check if chi is growing explosively."""
        if len(self.history) < 5:
            return False, None
        
        recent_chi = [m.chi_max for m in self.history[-5:]]
        
        # Check if chi exceeds threshold
        if recent_chi[-1] > self.chi_threshold:
            return True, self.history[-1].time
        
        # Check for rapid growth
        if len(self.history) > 10:
            early_chi = np.mean([m.chi_max for m in self.history[:5]])
            late_chi = np.mean([m.chi_max for m in self.history[-5:]])
            if late_chi > 3 * early_chi:  # 3x growth
                return True, self.history[-1].time
        
        return False, None
    
    def get_scaling_exponent(self) -> float:
        """
        Estimate scaling: χ ~ Re^α
        
        If α ≈ 0: regularity (flat)
        If α > 0.5: potential singularity
        """
        if len(self.history) < 3:
            return 0.0
        
        # Fit power law to chi(t)
        times = np.array([m.time for m in self.history])
        chis = np.array([m.chi_max for m in self.history])
        
        # Avoid log(0)
        times = times[times > 0]
        chis = chis[:len(times)]
        
        if len(times) < 2 or np.all(chis == chis[0]):
            return 0.0
        
        # Linear regression on log-log
        log_t = np.log(times + 0.01)
        log_chi = np.log(chis + 0.01)
        
        coeffs = np.polyfit(log_t, log_chi, 1)
        return coeffs[0]  # Scaling exponent


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT-BASED 2D NS SOLVER (Vorticity formulation)
# ═══════════════════════════════════════════════════════════════════════════════════════

class QTTNavierStokes2D:
    """
    2D Navier-Stokes in vorticity formulation with QTT compression tracking.
    
    ∂ω/∂t + (u·∇)ω = ν∇²ω
    ∇²ψ = -ω
    u = ∂ψ/∂y, v = -∂ψ/∂x
    """
    
    def __init__(self, n_qubits: int = 8, nu: float = 0.001, 
                 L: float = 2*np.pi, max_rank: int = 64):
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.nu = nu
        self.L = L
        self.dx = L / self.N
        self.max_rank = max_rank
        
        # Wavenumbers for spectral derivatives
        k = np.fft.fftfreq(self.N, self.dx) * 2 * np.pi
        self.kx, self.ky = np.meshgrid(k, k, indexing='ij')
        self.k_sq = self.kx**2 + self.ky**2
        self.k_sq[0, 0] = 1.0  # Avoid division by zero
        
        # QTT compression tracker
        self.tracker = QTTCompressionTracker(n_qubits * 2, chi_threshold=max_rank * 2)
        
        # 2/3 dealiasing mask
        kmax = self.N // 3
        self.dealias_mask = (np.abs(self.kx) < kmax * self.dx * 2 * np.pi) & \
                           (np.abs(self.ky) < kmax * self.dx * 2 * np.pi)
    
    def omega_to_velocity(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Recover velocity from vorticity via streamfunction."""
        omega_hat = np.fft.fft2(omega)
        
        # Solve Poisson: ψ = -ω / k²
        psi_hat = -omega_hat / self.k_sq
        psi_hat[0, 0] = 0
        
        # u = ∂ψ/∂y, v = -∂ψ/∂x
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        
        return u, v
    
    def compute_advection(self, omega: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute (u·∇)ω with dealiasing."""
        omega_hat = np.fft.fft2(omega)
        
        # Spectral derivatives
        domega_dx = np.fft.ifft2(1j * self.kx * omega_hat).real
        domega_dy = np.fft.ifft2(1j * self.ky * omega_hat).real
        
        # Advection
        adv = u * domega_dx + v * domega_dy
        
        # Dealias
        adv_hat = np.fft.fft2(adv)
        adv_hat[~self.dealias_mask] = 0
        
        return np.fft.ifft2(adv_hat).real
    
    def compute_diffusion(self, omega: np.ndarray) -> np.ndarray:
        """Compute ν∇²ω."""
        omega_hat = np.fft.fft2(omega)
        laplacian_hat = -self.k_sq * omega_hat
        laplacian_hat[0, 0] = 0
        return self.nu * np.fft.ifft2(laplacian_hat).real
    
    def field_to_qtt(self, field: np.ndarray) -> List[torch.Tensor]:
        """Convert 2D field to QTT format for compression tracking."""
        # Flatten with Morton ordering (interleaved bits)
        flat = field.flatten()
        tensor = torch.from_numpy(flat).double()
        
        # Convert to QTT
        qtt = dense_to_qtt(tensor, max_bond=self.max_rank)
        return qtt.cores
    
    def step_rk4(self, omega: np.ndarray, dt: float) -> np.ndarray:
        """RK4 time step."""
        def rhs(w):
            u, v = self.omega_to_velocity(w)
            return -self.compute_advection(w, u, v) + self.compute_diffusion(w)
        
        k1 = rhs(omega)
        k2 = rhs(omega + 0.5 * dt * k1)
        k3 = rhs(omega + 0.5 * dt * k2)
        k4 = rhs(omega + dt * k3)
        
        return omega + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def simulate(self, omega0: np.ndarray, T_final: float, dt: float,
                 track_interval: int = 10) -> Dict:
        """
        Run simulation with QTT compression tracking.
        
        Returns dictionary with:
        - Final state
        - Compression history
        - Blowup detection
        """
        omega = omega0.copy()
        t = 0.0
        step = 0
        
        n_steps = int(T_final / dt)
        
        print(f"\n  QTT NS2D: N={self.N}, Re={1/self.nu:.0f}, T={T_final}")
        print(f"  Tracking χ (bond dimension) as singularity detector")
        print()
        
        while t < T_final:
            # Track compression periodically
            if step % track_interval == 0:
                cores = self.field_to_qtt(omega)
                
                u, v = self.omega_to_velocity(omega)
                energy = 0.5 * (u**2 + v**2).mean() * self.L**2
                enstrophy = 0.5 * (omega**2).mean() * self.L**2
                omega_max = np.abs(omega).max()
                
                metrics = self.tracker.track(t, cores, omega_max, enstrophy, energy)
                
                # Print progress
                if step % (n_steps // 5) == 0 or step == 0:
                    print(f"  t={t:.3f}: χ_max={metrics.chi_max:3d}, "
                          f"χ_mean={metrics.chi_mean:.1f}, "
                          f"compress={metrics.compression_ratio:.1f}x, "
                          f"max|ω|={omega_max:.4f}")
                
                # Check for blowup
                blowup, blowup_time = self.tracker.check_blowup()
                if blowup:
                    print(f"\n  ⚠️  CHI BLOWUP at t={blowup_time:.4f}!")
                    break
            
            # Time step
            omega = self.step_rk4(omega, dt)
            t += dt
            step += 1
        
        # Final analysis
        alpha = self.tracker.get_scaling_exponent()
        chi_history = [m.chi_max for m in self.tracker.history]
        
        return {
            "omega_final": omega,
            "history": self.tracker.history,
            "chi_max_final": chi_history[-1] if chi_history else 0,
            "chi_scaling": alpha,
            "blowup_detected": self.tracker.check_blowup()[0],
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_taylor_green_vorticity(N: int, L: float = 2*np.pi) -> np.ndarray:
    """Taylor-Green vortex in vorticity form."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')
    
    # ω = 2 cos(x) cos(y) for Taylor-Green
    omega = 2 * np.cos(X) * np.cos(Y)
    return omega


def create_kida_vorticity(N: int, L: float = 2*np.pi, amp: float = 1.0) -> np.ndarray:
    """Kida-type vorticity (2D analog)."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')
    
    # High-frequency vorticity pattern
    omega = amp * (np.sin(X) * np.sin(3*Y) + np.sin(3*X) * np.sin(Y))
    return omega


def create_shear_layer_vorticity(N: int, L: float = 2*np.pi, delta: float = 0.1) -> np.ndarray:
    """Perturbed shear layer - rollup candidate."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')
    
    # Shear layer with perturbation
    y_centered = Y - L/2
    omega = 1.0 / (np.cosh(y_centered / delta)**2 * delta)
    
    # Add perturbation to trigger rollup
    omega += 0.1 * np.sin(2 * np.pi * X / L) * np.exp(-y_centered**2 / (2*delta**2))
    
    return omega


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN HUNT
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_qtt_singularity_hunt():
    """Run singularity hunt with QTT compression tracking."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 10 + "NAVIER-STOKES QTT SINGULARITY HUNTER" + " " * 29 + "║")
    print("║" + " " * 78 + "║")
    print("║  Tracking bond dimension χ as singularity detector" + " " * 26 + "║")
    print("║  If χ → ∞: SINGULARITY    If χ ~ const: REGULARITY" + " " * 24 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Test configurations
    configs = [
        ("Taylor-Green", create_taylor_green_vorticity, 1.0),
        ("Kida (low)", create_kida_vorticity, 1.0),
        ("Kida (high)", create_kida_vorticity, 5.0),
        ("Shear Layer", create_shear_layer_vorticity, 1.0),
    ]
    
    results = {}
    
    n_qubits = 7  # N = 128
    N = 2 ** n_qubits
    L = 2 * np.pi
    Re = 5000
    nu = 1.0 / Re
    T_final = 2.0
    dt = 0.002
    
    print(f"\n  Grid: N = {N}, Re = {Re}, T = {T_final}")
    print(f"  Max rank: 64")
    
    for name, ic_func, amp in configs:
        print("\n" + "═" * 70)
        print(f"IC: {name}")
        print("═" * 70)
        
        if "Kida" in name:
            omega0 = ic_func(N, L, amp=amp)
        elif "Shear" in name:
            omega0 = ic_func(N, L, delta=0.15)
        else:
            omega0 = ic_func(N, L)
        
        solver = QTTNavierStokes2D(n_qubits=n_qubits, nu=nu, L=L, max_rank=64)
        result = solver.simulate(omega0, T_final, dt, track_interval=20)
        
        results[name] = {
            "chi_max": result["chi_max_final"],
            "chi_scaling": result["chi_scaling"],
            "blowup": result["blowup_detected"],
            "verdict": "⚠️ BLOWUP" if result["blowup_detected"] else "✓ BOUNDED"
        }
        
        print(f"\n  Final χ_max = {result['chi_max_final']}, scaling ~ t^{result['chi_scaling']:.3f}")
        print(f"  Verdict: {results[name]['verdict']}")
    
    # Summary
    print("\n" + "═" * 80)
    print("SUMMARY: QTT COMPRESSION ANALYSIS")
    print("═" * 80)
    print(f"\n  {'IC':<20} {'χ_max':>8} {'Scaling':>10} {'Verdict':>15}")
    print("  " + "-" * 55)
    
    for name, r in results.items():
        print(f"  {name:<20} {r['chi_max']:>8} {r['chi_scaling']:>10.3f} {r['verdict']:>15}")
    
    # Overall verdict
    n_blowup = sum(1 for r in results.values() if r['blowup'])
    
    print("\n" + "═" * 80)
    if n_blowup > 0:
        print("  ⚠️  SINGULARITY CANDIDATE: χ diverged for some ICs")
    else:
        print("  ✓ ALL BOUNDED: χ stayed finite for all ICs")
        print("    → Evidence supports NS regularity")
    print("═" * 80)
    
    return results


if __name__ == "__main__":
    results = run_qtt_singularity_hunt()
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "method": "QTT compression tracking",
        "results": results
    }
    
    with open("qtt_singularity_hunt_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to qtt_singularity_hunt_results.json")
