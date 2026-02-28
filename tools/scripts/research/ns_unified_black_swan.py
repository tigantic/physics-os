#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                    UNIFIED 3D BLACK SWAN SINGULARITY HUNTER                          ║
║                                                                                      ║
║                    Combining BKM Criterion + QTT χ Tracking                          ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  TWO INDEPENDENT BLOWUP DETECTORS:                                                   ║
║  ─────────────────────────────────                                                   ║
║  1. BKM CRITERION: ∫₀^T ‖ω‖_∞ dt → ∞  implies blowup (Beale-Kato-Majda 1984)        ║
║  2. QTT COMPRESSION: χ(t) → ∞  implies loss of regularity (tensor rank explosion)   ║
║                                                                                      ║
║  INITIAL CONDITIONS:                                                                 ║
║  ───────────────────                                                                 ║
║  • Hou-Luo: Counter-rotating vortex rings (best theoretical candidate)              ║
║  • Kida: High symmetry vortex (classical blowup candidate)                          ║
║  • Anti-parallel tubes: Vortex reconnection geometry                                ║
║  • Taylor-Green: Baseline (known regular)                                           ║
║                                                                                      ║
║  SUCCESS = Find IC where BOTH metrics explode simultaneously                        ║
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
import time

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ontic.cfd.pure_qtt_ops import QTTState, dense_to_qtt


# ═══════════════════════════════════════════════════════════════════════════════════════
# DUAL METRIC TRACKER
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class DualMetrics:
    """Combined BKM + QTT metrics at a time point."""
    time: float
    # BKM metrics
    omega_max: float          # max|ω|
    bkm_integral: float       # ∫₀^t max|ω| ds
    enstrophy: float          # (1/2)∫|ω|² dx
    # QTT metrics  
    chi_max: int              # Maximum bond dimension
    chi_mean: float           # Mean bond dimension
    compression_ratio: float  # Dense params / QTT params
    # Flow metrics
    energy: float
    velocity_max: float


class DualTracker:
    """
    Track BOTH BKM criterion AND QTT bond dimension.
    
    Either metric exploding → potential singularity
    Both exploding → STRONG evidence
    """
    
    def __init__(self, bkm_threshold: float = 1e6, chi_threshold: int = 256):
        self.bkm_threshold = bkm_threshold
        self.chi_threshold = chi_threshold
        self.history: List[DualMetrics] = []
        self.bkm_integral = 0.0
    
    def update(self, t: float, dt: float,
               omega_max: float, enstrophy: float,
               chi_max: int, chi_mean: float, compression_ratio: float,
               energy: float, velocity_max: float):
        """Record dual metrics."""
        self.bkm_integral += omega_max * dt
        
        metrics = DualMetrics(
            time=t,
            omega_max=omega_max,
            bkm_integral=self.bkm_integral,
            enstrophy=enstrophy,
            chi_max=chi_max,
            chi_mean=chi_mean,
            compression_ratio=compression_ratio,
            energy=energy,
            velocity_max=velocity_max
        )
        self.history.append(metrics)
        return metrics
    
    def check_bkm_blowup(self) -> Tuple[bool, Optional[float]]:
        """Check BKM criterion."""
        if self.bkm_integral > self.bkm_threshold:
            return True, self.history[-1].time if self.history else None
        
        # Check for rapid ω growth (need at least 10 points)
        if len(self.history) > 10:
            early_omega = np.mean([m.omega_max for m in self.history[:5]])
            recent_omega = np.mean([m.omega_max for m in self.history[-5:]])
            if recent_omega > 100 * early_omega and recent_omega > 1000:
                return True, self.history[-1].time
        
        return False, None
    
    def check_chi_blowup(self) -> Tuple[bool, Optional[float]]:
        """Check QTT rank explosion."""
        if not self.history:
            return False, None
        
        if self.history[-1].chi_max > self.chi_threshold:
            return True, self.history[-1].time
        
        # Check for rapid χ growth
        if len(self.history) > 5:
            early_chi = np.mean([m.chi_max for m in self.history[:5]])
            late_chi = np.mean([m.chi_max for m in self.history[-5:]])
            if late_chi > 3 * early_chi and late_chi > 32:
                return True, self.history[-1].time
        
        return False, None
    
    def get_verdict(self) -> Tuple[str, bool]:
        """Get overall verdict."""
        bkm_blow, bkm_time = self.check_bkm_blowup()
        chi_blow, chi_time = self.check_chi_blowup()
        
        if bkm_blow and chi_blow:
            return "🔥 DUAL BLOWUP - STRONG SINGULARITY CANDIDATE", True
        elif bkm_blow:
            return "⚠️  BKM BLOWUP - ω diverging", True
        elif chi_blow:
            return "⚠️  CHI BLOWUP - rank explosion", True
        else:
            return "✓ BOUNDED - evidence for regularity", False


# ═══════════════════════════════════════════════════════════════════════════════════════
# 3D SPECTRAL NS SOLVER WITH QTT COMPRESSION TRACKING
# ═══════════════════════════════════════════════════════════════════════════════════════

class NS3DWithQTT:
    """
    3D Navier-Stokes with dual BKM + QTT tracking.
    
    Uses spectral method for accuracy, but tracks QTT compressibility
    of the solution as a regularity indicator.
    """
    
    def __init__(self, N: int = 64, L: float = 2*np.pi, nu: float = 0.001,
                 max_rank: int = 64):
        self.N = N
        self.L = L
        self.nu = nu
        self.dx = L / N
        self.max_rank = max_rank
        
        # Wavenumbers
        k = np.fft.fftfreq(N, self.dx) * 2 * np.pi
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k_sq = self.kx**2 + self.ky**2 + self.kz**2
        self.k_sq[0, 0, 0] = 1.0  # Avoid div by zero
        
        # 2/3 dealiasing mask
        kmax = N // 3
        self.dealias = (np.abs(self.kx) < kmax * self.dx * 2 * np.pi) & \
                       (np.abs(self.ky) < kmax * self.dx * 2 * np.pi) & \
                       (np.abs(self.kz) < kmax * self.dx * 2 * np.pi)
        
        # Tracker
        self.tracker = DualTracker(bkm_threshold=1e6, chi_threshold=max_rank * 2)
    
    def compute_vorticity(self, u: np.ndarray, v: np.ndarray, w: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ω = ∇ × u."""
        u_hat = np.fft.fftn(u)
        v_hat = np.fft.fftn(v)
        w_hat = np.fft.fftn(w)
        
        # ω_x = ∂w/∂y - ∂v/∂z
        omega_x = np.fft.ifftn(1j * self.ky * w_hat - 1j * self.kz * v_hat).real
        # ω_y = ∂u/∂z - ∂w/∂x
        omega_y = np.fft.ifftn(1j * self.kz * u_hat - 1j * self.kx * w_hat).real
        # ω_z = ∂v/∂x - ∂u/∂y
        omega_z = np.fft.ifftn(1j * self.kx * v_hat - 1j * self.ky * u_hat).real
        
        return omega_x, omega_y, omega_z
    
    def project_divergence_free(self, u: np.ndarray, v: np.ndarray, w: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project velocity to divergence-free space."""
        u_hat = np.fft.fftn(u)
        v_hat = np.fft.fftn(v)
        w_hat = np.fft.fftn(w)
        
        # div(u) in Fourier space
        div_hat = 1j * (self.kx * u_hat + self.ky * v_hat + self.kz * w_hat)
        
        # Pressure solve: ∇²p = div(u)
        p_hat = div_hat / self.k_sq
        p_hat[0, 0, 0] = 0
        
        # Project: u = u - ∇p
        u_hat = u_hat - 1j * self.kx * p_hat
        v_hat = v_hat - 1j * self.ky * p_hat
        w_hat = w_hat - 1j * self.kz * p_hat
        
        return (np.fft.ifftn(u_hat).real,
                np.fft.ifftn(v_hat).real,
                np.fft.ifftn(w_hat).real)
    
    def rhs(self, u: np.ndarray, v: np.ndarray, w: np.ndarray
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute RHS: -u·∇u + ν∇²u."""
        u_hat = np.fft.fftn(u)
        v_hat = np.fft.fftn(v)
        w_hat = np.fft.fftn(w)
        
        # Gradients
        du_dx = np.fft.ifftn(1j * self.kx * u_hat).real
        du_dy = np.fft.ifftn(1j * self.ky * u_hat).real
        du_dz = np.fft.ifftn(1j * self.kz * u_hat).real
        
        dv_dx = np.fft.ifftn(1j * self.kx * v_hat).real
        dv_dy = np.fft.ifftn(1j * self.ky * v_hat).real
        dv_dz = np.fft.ifftn(1j * self.kz * v_hat).real
        
        dw_dx = np.fft.ifftn(1j * self.kx * w_hat).real
        dw_dy = np.fft.ifftn(1j * self.ky * w_hat).real
        dw_dz = np.fft.ifftn(1j * self.kz * w_hat).real
        
        # Advection (dealiased)
        adv_u = u * du_dx + v * du_dy + w * du_dz
        adv_v = u * dv_dx + v * dv_dy + w * dv_dz
        adv_w = u * dw_dx + v * dw_dy + w * dw_dz
        
        # Dealias
        adv_u = np.fft.ifftn(np.fft.fftn(adv_u) * self.dealias).real
        adv_v = np.fft.ifftn(np.fft.fftn(adv_v) * self.dealias).real
        adv_w = np.fft.ifftn(np.fft.fftn(adv_w) * self.dealias).real
        
        # Diffusion
        diff_u = self.nu * np.fft.ifftn(-self.k_sq * u_hat).real
        diff_v = self.nu * np.fft.ifftn(-self.k_sq * v_hat).real
        diff_w = self.nu * np.fft.ifftn(-self.k_sq * w_hat).real
        
        return -adv_u + diff_u, -adv_v + diff_v, -adv_w + diff_w
    
    def step_rk4(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, dt: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RK4 time step."""
        def rhs_proj(u, v, w):
            du, dv, dw = self.rhs(u, v, w)
            return self.project_divergence_free(du, dv, dw)
        
        k1u, k1v, k1w = rhs_proj(u, v, w)
        k2u, k2v, k2w = rhs_proj(u + 0.5*dt*k1u, v + 0.5*dt*k1v, w + 0.5*dt*k1w)
        k3u, k3v, k3w = rhs_proj(u + 0.5*dt*k2u, v + 0.5*dt*k2v, w + 0.5*dt*k2w)
        k4u, k4v, k4w = rhs_proj(u + dt*k3u, v + dt*k3v, w + dt*k3w)
        
        u_new = u + (dt/6) * (k1u + 2*k2u + 2*k3u + k4u)
        v_new = v + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
        w_new = w + (dt/6) * (k1w + 2*k2w + 2*k3w + k4w)
        
        return self.project_divergence_free(u_new, v_new, w_new)
    
    def field_to_qtt_metrics(self, field: np.ndarray) -> Tuple[int, float, float]:
        """Compute QTT compression metrics for a 3D field."""
        # Check for NaN/Inf
        if not np.isfinite(field).all():
            return self.max_rank, float(self.max_rank), 1.0
        
        # Flatten and convert to QTT
        flat = field.flatten()
        n = len(flat)
        
        # Pad to power of 2
        n_qubits = int(np.ceil(np.log2(n)))
        n_padded = 2 ** n_qubits
        if n_padded > n:
            flat = np.concatenate([flat, np.zeros(n_padded - n)])
        
        tensor = torch.from_numpy(flat).double()
        
        try:
            qtt = dense_to_qtt(tensor, max_bond=self.max_rank)
            chi_max = max(c.shape[0] for c in qtt.cores)
            chi_mean = np.mean([c.shape[0] for c in qtt.cores])
            qtt_params = sum(c.numel() for c in qtt.cores)
            compression = n_padded / max(qtt_params, 1)
        except Exception:
            # SVD failed - field is too irregular
            chi_max = self.max_rank
            chi_mean = float(self.max_rank)
            compression = 1.0
        
        return chi_max, chi_mean, compression
    
    def simulate(self, u0: np.ndarray, v0: np.ndarray, w0: np.ndarray,
                 T_final: float, dt: float, ic_name: str = "IC",
                 track_interval: int = 10) -> Dict:
        """
        Run simulation with dual tracking.
        """
        u, v, w = u0.copy(), v0.copy(), w0.copy()
        u, v, w = self.project_divergence_free(u, v, w)
        
        t = 0.0
        step = 0
        n_steps = int(T_final / dt)
        
        print(f"\n  {ic_name}: N={self.N}, Re={1/self.nu:.0f}, T={T_final}")
        print(f"  Tracking BKM integral + QTT χ")
        
        start_time = time.time()
        
        while t < T_final:
            if step % track_interval == 0:
                # Compute vorticity
                ox, oy, oz = self.compute_vorticity(u, v, w)
                omega_mag = np.sqrt(ox**2 + oy**2 + oz**2)
                omega_max = omega_mag.max()
                enstrophy = 0.5 * (omega_mag**2).mean() * self.L**3
                
                # Energy and velocity
                energy = 0.5 * (u**2 + v**2 + w**2).mean() * self.L**3
                vel_max = np.sqrt(u**2 + v**2 + w**2).max()
                
                # QTT metrics (use velocity magnitude)
                vel_mag = np.sqrt(u**2 + v**2 + w**2)
                chi_max, chi_mean, compress = self.field_to_qtt_metrics(vel_mag)
                
                # Update tracker
                metrics = self.tracker.update(
                    t, dt * track_interval,
                    omega_max, enstrophy,
                    chi_max, chi_mean, compress,
                    energy, vel_max
                )
                
                # Progress
                if step % (n_steps // 5) == 0 or step == 0:
                    print(f"    t={t:.3f}: |ω|_max={omega_max:.2f}, BKM={metrics.bkm_integral:.2f}, "
                          f"χ={chi_max}, compress={compress:.1f}x")
                
                # Check for blowup
                verdict, blowup = self.tracker.get_verdict()
                if blowup:
                    print(f"\n    {verdict}")
                    break
            
            # Time step
            u, v, w = self.step_rk4(u, v, w, dt)
            t += dt
            step += 1
            
            # Check for NaN (numerical blowup)
            if not np.isfinite(u).all() or not np.isfinite(v).all() or not np.isfinite(w).all():
                print(f"\n    ⚠️  NUMERICAL INSTABILITY at t={t:.4f}")
                break
        
        runtime = time.time() - start_time
        verdict, blowup = self.tracker.get_verdict()
        
        # Final metrics
        final = self.tracker.history[-1] if self.tracker.history else None
        
        return {
            "ic_name": ic_name,
            "verdict": verdict,
            "blowup": blowup,
            "bkm_integral": final.bkm_integral if final else 0,
            "chi_max": final.chi_max if final else 0,
            "omega_max": final.omega_max if final else 0,
            "runtime": runtime,
            "history": self.tracker.history
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def taylor_green_3d(N: int, L: float = 2*np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Taylor-Green vortex - baseline (known regular)."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    
    return u, v, w


def kida_vortex_3d(N: int, L: float = 2*np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kida vortex - high symmetry blowup candidate."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    u = np.sin(X) * (np.cos(3*Y) * np.cos(Z) - np.cos(Y) * np.cos(3*Z))
    v = np.sin(Y) * (np.cos(3*Z) * np.cos(X) - np.cos(Z) * np.cos(3*X))
    w = np.sin(Z) * (np.cos(3*X) * np.cos(Y) - np.cos(X) * np.cos(3*Y))
    
    # Normalize
    energy = (u**2 + v**2 + w**2).mean()
    scale = 1.0 / np.sqrt(energy + 1e-10)
    
    return u * scale, v * scale, w * scale


def hou_luo_rings(N: int, L: float = 2*np.pi, 
                  R: float = 0.3, sep: float = 0.2, Gamma: float = 1.0
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hou-Luo counter-rotating vortex rings.
    
    THE best theoretical candidate for blowup (Hou & Luo 2014).
    Two vortex rings approaching each other with opposite circulation.
    """
    x = np.linspace(0, L, N, endpoint=False) - L/2
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Ring 1 at z = -sep/2, circulation +Gamma
    # Ring 2 at z = +sep/2, circulation -Gamma
    z1, z2 = -sep * L / 2, +sep * L / 2
    
    # Cylindrical radius from z-axis
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    # Core thickness
    a = L * 0.05
    R_ring = R * L
    
    # Distance from ring cores
    r1 = np.sqrt((r - R_ring)**2 + (Z - z1)**2)
    r2 = np.sqrt((r - R_ring)**2 + (Z - z2)**2)
    
    # Vorticity magnitude (Gaussian cores)
    omega1 = Gamma / (np.pi * a**2) * np.exp(-r1**2 / a**2)
    omega2 = -Gamma / (np.pi * a**2) * np.exp(-r2**2 / a**2)  # Opposite sign
    
    # Vorticity is in theta direction (azimuthal)
    # ω = ω_θ ê_θ, where ê_θ = (-sin(θ), cos(θ), 0)
    omega_total = omega1 + omega2
    omega_x = -np.sin(theta) * omega_total
    omega_y = np.cos(theta) * omega_total
    omega_z = np.zeros_like(omega_x)
    
    # Recover velocity from vorticity via Biot-Savart (simplified: use streamfunction)
    # For vortex rings, the induced velocity is primarily in the z-direction near axis
    # and swirling in the r-theta plane
    
    # Simplified: create velocity from vorticity cross product approximation
    # u = -∫ (r' × ω) / |r'|³ dr' ≈ local approximation
    
    # Use Lamb-Oseen type profile
    u = -Y / (r + 0.1) * omega_total * 0.1
    v = X / (r + 0.1) * omega_total * 0.1
    w = np.zeros_like(u)
    
    # Add axial velocity component (rings moving toward each other)
    w += 0.5 * np.exp(-r1**2 / a**2) - 0.5 * np.exp(-r2**2 / a**2)
    
    # Normalize
    energy = (u**2 + v**2 + w**2).mean()
    if energy > 1e-10:
        scale = 1.0 / np.sqrt(energy)
        u, v, w = u * scale, v * scale, w * scale
    
    return u, v, w


def anti_parallel_tubes(N: int, L: float = 2*np.pi
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anti-parallel vortex tubes - reconnection geometry."""
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Two tubes along z, at different y positions
    y1, y2 = L/3, 2*L/3
    a = L * 0.08  # Core radius
    
    r1 = np.sqrt((Y - y1)**2 + (X - L/2)**2)
    r2 = np.sqrt((Y - y2)**2 + (X - L/2)**2)
    
    # Vorticity in z-direction (opposite signs)
    omega_z1 = np.exp(-r1**2 / a**2)
    omega_z2 = -np.exp(-r2**2 / a**2)
    
    # Induced velocity (azimuthal around each tube)
    # Tube 1: u = -∂ψ/∂y, v = ∂ψ/∂x where ∇²ψ = -ω_z
    u = -(Y - y1) / (r1 + 0.1) * omega_z1 * 0.3 - (Y - y2) / (r2 + 0.1) * omega_z2 * 0.3
    v = (X - L/2) / (r1 + 0.1) * omega_z1 * 0.3 + (X - L/2) / (r2 + 0.1) * omega_z2 * 0.3
    w = np.zeros_like(u)
    
    # Add perturbation to trigger reconnection
    w += 0.1 * np.sin(2 * np.pi * Z / L) * (omega_z1 - omega_z2)
    
    # Normalize
    energy = (u**2 + v**2 + w**2).mean()
    scale = 1.0 / np.sqrt(energy + 1e-10)
    
    return u * scale, v * scale, w * scale


def trefoil_knot(N: int, L: float = 2*np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trefoil vortex knot - topologically nontrivial."""
    x = np.linspace(0, L, N, endpoint=False) - L/2
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Trefoil parametrization: (sin(t) + 2sin(2t), cos(t) - 2cos(2t), -sin(3t))
    # We create a tubular neighborhood around this curve
    
    a = L * 0.08  # Core thickness
    n_pts = 200
    t_param = np.linspace(0, 2*np.pi, n_pts)
    
    # Trefoil centerline
    cx = (np.sin(t_param) + 2*np.sin(2*t_param)) * L * 0.15
    cy = (np.cos(t_param) - 2*np.cos(2*t_param)) * L * 0.15
    cz = -np.sin(3*t_param) * L * 0.15
    
    # Tangent vectors
    tx = np.cos(t_param) + 4*np.cos(2*t_param)
    ty = -np.sin(t_param) + 4*np.sin(2*t_param)
    tz = -3*np.cos(3*t_param)
    t_mag = np.sqrt(tx**2 + ty**2 + tz**2)
    tx, ty, tz = tx/t_mag, ty/t_mag, tz/t_mag
    
    # Build vorticity field
    omega_x = np.zeros_like(X)
    omega_y = np.zeros_like(Y)
    omega_z = np.zeros_like(Z)
    
    for i in range(n_pts):
        dist = np.sqrt((X - cx[i])**2 + (Y - cy[i])**2 + (Z - cz[i])**2)
        weight = np.exp(-dist**2 / a**2)
        omega_x += tx[i] * weight
        omega_y += ty[i] * weight
        omega_z += tz[i] * weight
    
    # Velocity from vorticity (simplified Biot-Savart)
    # u ≈ ω × r / |r|² locally
    r_mag = np.sqrt(X**2 + Y**2 + Z**2) + 0.1
    u = (omega_y * Z - omega_z * Y) / r_mag * 0.1
    v = (omega_z * X - omega_x * Z) / r_mag * 0.1
    w = (omega_x * Y - omega_y * X) / r_mag * 0.1
    
    # Normalize
    energy = (u**2 + v**2 + w**2).mean()
    scale = 1.0 / np.sqrt(energy + 1e-10)
    
    return u * scale, v * scale, w * scale


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN HUNT
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_unified_hunt():
    """Run the unified Black Swan hunt with dual tracking."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 12 + "UNIFIED 3D BLACK SWAN SINGULARITY HUNTER" + " " * 24 + "║")
    print("║" + " " * 78 + "║")
    print("║  DUAL METRICS: BKM integral ∫|ω|_∞ dt  +  QTT bond dimension χ" + " " * 12 + "║")
    print("║  Both exploding = STRONG singularity evidence" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Configuration
    N = 32          # Smaller grid for stability
    L = 2 * np.pi
    Re = 500        # Lower Re for stability
    nu = 1.0 / Re
    T_final = 3.0
    dt = 0.002      # Conservative dt
    max_rank = 64
    
    print(f"\n  Grid: N={N}³, Re={Re}, T={T_final}, max_rank={max_rank}")
    
    # Initial conditions to test
    ics = [
        ("Taylor-Green (baseline)", taylor_green_3d),
        ("Kida Vortex", kida_vortex_3d),
        ("Hou-Luo Rings", hou_luo_rings),
        ("Anti-parallel Tubes", anti_parallel_tubes),
        ("Trefoil Knot", trefoil_knot),
    ]
    
    results = {}
    
    for ic_name, ic_func in ics:
        print("\n" + "═" * 70)
        
        # Create IC
        if ic_func == hou_luo_rings:
            u0, v0, w0 = ic_func(N, L, R=0.3, sep=0.2, Gamma=1.0)
        else:
            u0, v0, w0 = ic_func(N, L)
        
        # Create fresh solver for each IC
        solver = NS3DWithQTT(N=N, L=L, nu=nu, max_rank=max_rank)
        
        # Run
        result = solver.simulate(u0, v0, w0, T_final, dt, ic_name, track_interval=20)
        
        results[ic_name] = {
            "verdict": result["verdict"],
            "blowup": result["blowup"],
            "bkm": result["bkm_integral"],
            "chi_max": result["chi_max"],
            "omega_max": result["omega_max"],
            "runtime": result["runtime"]
        }
        
        print(f"\n  → {result['verdict']}")
    
    # Summary
    print("\n" + "═" * 80)
    print("UNIFIED BLACK SWAN HUNT: SUMMARY")
    print("═" * 80)
    print(f"\n  {'IC':<25} {'BKM':>10} {'χ_max':>8} {'|ω|_max':>10} {'Verdict':>20}")
    print("  " + "-" * 75)
    
    for ic_name, r in results.items():
        short_name = ic_name[:25]
        verdict_short = "BLOWUP" if r["blowup"] else "BOUNDED"
        print(f"  {short_name:<25} {r['bkm']:>10.2f} {r['chi_max']:>8} {r['omega_max']:>10.2f} {verdict_short:>20}")
    
    # Final verdict
    n_blowup = sum(1 for r in results.values() if r["blowup"])
    
    print("\n" + "═" * 80)
    if n_blowup > 0:
        print("  🔥 SINGULARITY CANDIDATE(S) FOUND!")
        print("     Requires higher resolution verification")
    else:
        print("  ✓ ALL ICs BOUNDED")
        print("    → BKM integral finite")
        print("    → QTT rank stable")
        print("    → Evidence supports NS regularity")
    print("═" * 80)
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "N": N,
            "Re": Re,
            "T_final": T_final,
            "max_rank": max_rank
        },
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "history"} 
                   for k, v in results.items()},
        "summary": {
            "total_ics": len(results),
            "blowups": n_blowup,
            "verdict": "SINGULARITY_CANDIDATE" if n_blowup > 0 else "ALL_BOUNDED"
        }
    }
    
    with open("unified_black_swan_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to unified_black_swan_results.json")
    
    return results


if __name__ == "__main__":
    run_unified_hunt()
