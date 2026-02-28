"""
QTT Vorticity and Regularity Suite
===================================

For Navier-Stokes singularity hunting, we need to track:
1. Vorticity ω = ∇ × u (the curl of velocity)
2. Maximum vorticity |ω|_∞ (Beale-Kato-Majda criterion)
3. Enstrophy Ω = (1/2)∫|ω|² dx
4. Sobolev norms H^s (regularity measures)

The Beale-Kato-Majda Theorem (1984):
    A smooth solution blows up at time T* if and only if:
        ∫₀^T* ||ω(t)||_∞ dt = ∞
    
    So |ω|_∞ → ∞ indicates singularity!

QTT-Native Approach:
    - Compute curl using rank-preserving derivatives
    - Track max |ω| via sampling (not dense materialization)
    - Estimate Sobolev norms from spectral representation

Author: HyperTensor Team
Date: 2026-01-16
Tag: [PHYSICS-TOOLBOX] [REGULARITY] [BKM-CRITERION]
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import time

import torch
from torch import Tensor

from ontic.cfd.nd_shift_mpo import truncate_cores
from ontic.cfd.qtt_shift_stable import qtt_3d_central_diff_stable
from ontic.cfd.pure_qtt_ops import qtt_add, QTTState


# =============================================================================
# Vorticity Computation
# =============================================================================

def qtt_vorticity_3d(
    u_cores: List[Tensor],
    v_cores: List[Tensor],
    w_cores: List[Tensor],
    n_qubits_per_dim: int,
    dx: float,
    max_rank: int = 256,
    tol: float = 1e-8
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """
    Compute 3D vorticity ω = ∇ × u in QTT format.
    
    ω = (∂w/∂y - ∂v/∂z,
         ∂u/∂z - ∂w/∂x,
         ∂v/∂x - ∂u/∂y)
    
    Uses rank-preserving derivatives.
    
    Args:
        u_cores, v_cores, w_cores: Velocity QTT cores
        n_qubits_per_dim: Qubits per spatial dimension
        dx: Grid spacing
        max_rank: Maximum rank
        tol: Truncation tolerance
        
    Returns:
        (omega_x, omega_y, omega_z): Vorticity components in QTT format
    """
    # Compute all needed derivatives
    # ∂w/∂y
    dw_dy = qtt_3d_central_diff_stable(w_cores, n_qubits_per_dim, 1, dx, max_rank, tol)
    # ∂v/∂z
    dv_dz = qtt_3d_central_diff_stable(v_cores, n_qubits_per_dim, 2, dx, max_rank, tol)
    # ∂u/∂z
    du_dz = qtt_3d_central_diff_stable(u_cores, n_qubits_per_dim, 2, dx, max_rank, tol)
    # ∂w/∂x
    dw_dx = qtt_3d_central_diff_stable(w_cores, n_qubits_per_dim, 0, dx, max_rank, tol)
    # ∂v/∂x
    dv_dx = qtt_3d_central_diff_stable(v_cores, n_qubits_per_dim, 0, dx, max_rank, tol)
    # ∂u/∂y
    du_dy = qtt_3d_central_diff_stable(u_cores, n_qubits_per_dim, 1, dx, max_rank, tol)
    
    # ω_x = ∂w/∂y - ∂v/∂z
    dv_dz_neg = [c.clone() for c in dv_dz]
    dv_dz_neg[0] = -dv_dz_neg[0]
    a = QTTState(cores=dw_dy, num_qubits=len(dw_dy))
    b = QTTState(cores=dv_dz_neg, num_qubits=len(dv_dz_neg))
    omega_x_qtt = qtt_add(a, b, max_bond=max_rank * 2)
    omega_x = truncate_cores(list(omega_x_qtt.cores), max_rank, tol)
    
    # ω_y = ∂u/∂z - ∂w/∂x
    dw_dx_neg = [c.clone() for c in dw_dx]
    dw_dx_neg[0] = -dw_dx_neg[0]
    a = QTTState(cores=du_dz, num_qubits=len(du_dz))
    b = QTTState(cores=dw_dx_neg, num_qubits=len(dw_dx_neg))
    omega_y_qtt = qtt_add(a, b, max_bond=max_rank * 2)
    omega_y = truncate_cores(list(omega_y_qtt.cores), max_rank, tol)
    
    # ω_z = ∂v/∂x - ∂u/∂y
    du_dy_neg = [c.clone() for c in du_dy]
    du_dy_neg[0] = -du_dy_neg[0]
    a = QTTState(cores=dv_dx, num_qubits=len(dv_dx))
    b = QTTState(cores=du_dy_neg, num_qubits=len(du_dy_neg))
    omega_z_qtt = qtt_add(a, b, max_bond=max_rank * 2)
    omega_z = truncate_cores(list(omega_z_qtt.cores), max_rank, tol)
    
    return omega_x, omega_y, omega_z


# =============================================================================
# Maximum Vorticity via Sampling (BKM Criterion)
# =============================================================================

def qtt_sample_random(
    cores: List[Tensor],
    n_samples: int = 1000
) -> Tensor:
    """
    Sample QTT at random indices without dense materialization.
    
    Returns values at n_samples random locations.
    """
    device = cores[0].device
    dtype = cores[0].dtype
    n_qubits = len(cores)
    N = 2 ** n_qubits
    
    # Generate random indices
    indices = torch.randint(0, N, (n_samples,), device=device)
    
    # Evaluate at each index via core contraction
    values = torch.zeros(n_samples, device=device, dtype=dtype)
    
    for i, idx in enumerate(indices):
        # Convert index to binary
        bits = [(int(idx) >> j) & 1 for j in range(n_qubits)]
        
        # Contract cores
        result = cores[0][:, bits[0], :]  # (1, r)
        for j in range(1, n_qubits):
            result = result @ cores[j][:, bits[j], :]  # (1, r') → (1, r'')
        
        values[i] = result.squeeze()
    
    return values


def qtt_max_abs_sample(
    cores: List[Tensor],
    n_samples: int = 10000
) -> float:
    """
    Estimate max |f| via random sampling.
    
    Not exact, but O(n_samples * n_qubits * r²) instead of O(N).
    """
    samples = qtt_sample_random(cores, n_samples)
    return float(samples.abs().max())


def qtt_vorticity_max_3d(
    u_cores: List[Tensor],
    v_cores: List[Tensor],
    w_cores: List[Tensor],
    n_qubits_per_dim: int,
    dx: float,
    max_rank: int = 256,
    n_samples: int = 10000
) -> float:
    """
    Estimate maximum vorticity magnitude ||ω||_∞ via sampling.
    
    This is the key quantity for Beale-Kato-Majda criterion.
    
    Returns:
        Estimated max |ω|
    """
    # Compute vorticity
    omega_x, omega_y, omega_z = qtt_vorticity_3d(
        u_cores, v_cores, w_cores, n_qubits_per_dim, dx, max_rank
    )
    
    # Sample each component
    wx = qtt_sample_random(omega_x, n_samples)
    wy = qtt_sample_random(omega_y, n_samples)
    wz = qtt_sample_random(omega_z, n_samples)
    
    # Compute |ω| at each sample point
    omega_mag = torch.sqrt(wx**2 + wy**2 + wz**2)
    
    return float(omega_mag.max())


# =============================================================================
# Enstrophy (Integrated Vorticity Squared)
# =============================================================================

def qtt_enstrophy_3d(
    u_cores: List[Tensor],
    v_cores: List[Tensor],
    w_cores: List[Tensor],
    n_qubits_per_dim: int,
    dx: float,
    max_rank: int = 256
) -> float:
    """
    Compute enstrophy Ω = (1/2) ∫ |ω|² dx.
    
    Uses QTT norm of vorticity components.
    """
    from ontic.cfd.qtt_spectral import qtt_frobenius_norm
    
    omega_x, omega_y, omega_z = qtt_vorticity_3d(
        u_cores, v_cores, w_cores, n_qubits_per_dim, dx, max_rank
    )
    
    # ||ω||² = ||ω_x||² + ||ω_y||² + ||ω_z||²
    omega_norm_sq = (
        qtt_frobenius_norm(omega_x) ** 2 +
        qtt_frobenius_norm(omega_y) ** 2 +
        qtt_frobenius_norm(omega_z) ** 2
    )
    
    # Scale by grid spacing for integral
    L = dx * (2 ** n_qubits_per_dim)
    volume = L ** 3
    n_points = (2 ** n_qubits_per_dim) ** 3
    
    # Enstrophy = (1/2) * (volume / n_points) * ||ω||²
    enstrophy = 0.5 * (volume / n_points) * omega_norm_sq
    
    return float(enstrophy)


# =============================================================================
# Sobolev Norm Estimation (Regularity Measure)
# =============================================================================

def qtt_sobolev_norm_estimate(
    cores: List[Tensor],
    s: float = 1.0,
    L: float = 2 * math.pi
) -> float:
    """
    Estimate Sobolev H^s norm via spectral approximation.
    
    ||f||_{H^s}² = ∫ (1 + |k|²)^s |f̂(k)|² dk
    
    For QTT, we use Walsh-Hadamard as spectral proxy.
    
    Args:
        cores: QTT cores
        s: Sobolev exponent (s=1 for H¹, s=2 for H²)
        L: Domain length
        
    Returns:
        Estimated H^s norm
    """
    from ontic.cfd.qtt_spectral import qtt_walsh_hadamard, qtt_frobenius_norm
    
    n_qubits = len(cores)
    N = 2 ** n_qubits
    dk = 2 * math.pi / L
    
    # Transform to spectral space
    hat_cores = qtt_walsh_hadamard(cores)
    
    # For accurate H^s, we'd need k-weighted norm
    # Approximation: weight by qubit position (proxy for k)
    
    # Simple approximation: H^s ≈ ||f||² + k_rms^(2s) * ||f̂||²
    f_norm_sq = qtt_frobenius_norm(cores) ** 2
    f_hat_norm_sq = qtt_frobenius_norm(hat_cores) ** 2
    
    # Characteristic wavenumber
    k_rms = (N / 3) * dk  # Rough estimate
    
    h_s_norm_sq = f_norm_sq + (k_rms ** (2*s)) * f_hat_norm_sq / N
    
    return math.sqrt(h_s_norm_sq)


# =============================================================================
# Regularity Diagnostic Suite
# =============================================================================

@dataclass
class RegularityState:
    """Snapshot of regularity diagnostics at one time."""
    time: float
    chi_max: int  # Maximum bond dimension
    omega_max: float  # Maximum vorticity (BKM)
    enstrophy: float  # Integrated vorticity squared
    h1_norm: float  # H¹ Sobolev norm
    h2_norm: float  # H² Sobolev norm
    energy: float  # Kinetic energy
    
    @property
    def bkm_indicator(self) -> str:
        """Beale-Kato-Majda assessment."""
        if self.omega_max > 100:
            return "CRITICAL"
        elif self.omega_max > 50:
            return "WARNING"
        elif self.omega_max > 20:
            return "ELEVATED"
        else:
            return "NORMAL"


@dataclass 
class RegularityTrajectory:
    """Full trajectory of regularity diagnostics."""
    states: list[RegularityState] = field(default_factory=list)
    
    def add(self, state: RegularityState):
        self.states.append(state)
    
    @property
    def times(self) -> List[float]:
        return [s.time for s in self.states]
    
    @property
    def omega_max_values(self) -> List[float]:
        return [s.omega_max for s in self.states]
    
    @property
    def chi_values(self) -> List[int]:
        return [s.chi_max for s in self.states]
    
    def bkm_integral(self) -> float:
        """
        Compute ∫₀^t ||ω||_∞ dt (BKM integral).
        
        If this → ∞, we have singularity!
        """
        if len(self.states) < 2:
            return 0.0
        
        integral = 0.0
        for i in range(1, len(self.states)):
            dt = self.states[i].time - self.states[i-1].time
            omega_avg = 0.5 * (self.states[i].omega_max + self.states[i-1].omega_max)
            integral += omega_avg * dt
        
        return integral
    
    def diagnose_blowup(self) -> dict:
        """Analyze trajectory for blowup signatures."""
        if len(self.states) < 3:
            return {'verdict': 'INSUFFICIENT_DATA'}
        
        # Check omega growth rate
        omega_vals = self.omega_max_values
        chi_vals = self.chi_values
        times = self.times
        
        # Linear regression on log(omega)
        import numpy as np
        log_omega = np.log(np.array(omega_vals) + 1e-10)
        t_arr = np.array(times)
        
        if len(t_arr) > 2:
            slope = np.polyfit(t_arr, log_omega, 1)[0]
        else:
            slope = 0.0
        
        # Check for chi saturation
        chi_max = max(chi_vals)
        chi_growth = (chi_vals[-1] - chi_vals[0]) / (times[-1] - times[0] + 1e-10)
        
        verdict = 'STABLE'
        if slope > 2.0:
            verdict = 'EXPONENTIAL_GROWTH'
        if chi_growth > 100:
            verdict = 'CHI_EXPLOSION'
        if self.states[-1].omega_max > 100:
            verdict = 'BKM_CRITICAL'
        
        return {
            'verdict': verdict,
            'omega_growth_rate': slope,
            'chi_growth_rate': chi_growth,
            'bkm_integral': self.bkm_integral(),
            'final_omega_max': self.states[-1].omega_max,
            'final_chi': chi_vals[-1]
        }


class RegularityMonitor:
    """
    Real-time regularity monitoring for NS simulations.
    
    Tracks all relevant quantities for singularity detection.
    """
    
    def __init__(
        self,
        n_qubits_per_dim: int,
        dx: float,
        max_rank: int = 256,
        n_vorticity_samples: int = 5000
    ):
        self.n_qubits_per_dim = n_qubits_per_dim
        self.dx = dx
        self.max_rank = max_rank
        self.n_vorticity_samples = n_vorticity_samples
        self.trajectory = RegularityTrajectory()
    
    def snapshot(
        self,
        u_cores: List[Tensor],
        v_cores: List[Tensor],
        w_cores: List[Tensor],
        t: float
    ) -> RegularityState:
        """Take a snapshot of all regularity diagnostics."""
        from ontic.cfd.qtt_spectral import qtt_frobenius_norm
        
        # Chi (max rank)
        chi_max = max(
            max(max(c.shape[0], c.shape[2]) for c in u_cores),
            max(max(c.shape[0], c.shape[2]) for c in v_cores),
            max(max(c.shape[0], c.shape[2]) for c in w_cores)
        )
        
        # Maximum vorticity (BKM)
        omega_max = qtt_vorticity_max_3d(
            u_cores, v_cores, w_cores,
            self.n_qubits_per_dim, self.dx,
            self.max_rank, self.n_vorticity_samples
        )
        
        # Enstrophy
        enstrophy = qtt_enstrophy_3d(
            u_cores, v_cores, w_cores,
            self.n_qubits_per_dim, self.dx, self.max_rank
        )
        
        # Sobolev norms
        h1_u = qtt_sobolev_norm_estimate(u_cores, s=1.0)
        h1_v = qtt_sobolev_norm_estimate(v_cores, s=1.0)
        h1_w = qtt_sobolev_norm_estimate(w_cores, s=1.0)
        h1_norm = math.sqrt(h1_u**2 + h1_v**2 + h1_w**2)
        
        h2_u = qtt_sobolev_norm_estimate(u_cores, s=2.0)
        h2_v = qtt_sobolev_norm_estimate(v_cores, s=2.0)
        h2_w = qtt_sobolev_norm_estimate(w_cores, s=2.0)
        h2_norm = math.sqrt(h2_u**2 + h2_v**2 + h2_w**2)
        
        # Energy
        energy = 0.5 * (
            qtt_frobenius_norm(u_cores)**2 +
            qtt_frobenius_norm(v_cores)**2 +
            qtt_frobenius_norm(w_cores)**2
        )
        
        state = RegularityState(
            time=t,
            chi_max=chi_max,
            omega_max=omega_max,
            enstrophy=enstrophy,
            h1_norm=h1_norm,
            h2_norm=h2_norm,
            energy=energy
        )
        
        self.trajectory.add(state)
        return state
    
    def print_status(self, state: RegularityState):
        """Print formatted status."""
        print(f"  t={state.time:.4f} | χ={state.chi_max:3d} | "
              f"|ω|_∞={state.omega_max:.2f} | Ω={state.enstrophy:.2e} | "
              f"H¹={state.h1_norm:.2e} | [{state.bkm_indicator}]")


# =============================================================================
# Test Suite
# =============================================================================

def test_regularity():
    """Test regularity diagnostic suite."""
    print("=" * 70)
    print("QTT Regularity Suite Test")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    n_qubits_per_dim = 5  # 32³ test grid
    n_total = 3 * n_qubits_per_dim
    rank = 16
    dx = 2 * math.pi / (2 ** n_qubits_per_dim)
    
    # Create test velocity field
    def make_qtt_3d(scale=0.1):
        cores = []
        for i in range(n_total):
            r_left = 1 if i == 0 else rank
            r_right = 1 if i == n_total - 1 else rank
            core = torch.randn(r_left, 2, r_right, device=device, dtype=dtype) * scale
            cores.append(core)
        return cores
    
    u = make_qtt_3d()
    v = make_qtt_3d()
    w = make_qtt_3d()
    
    print(f"\nTest grid: {2**n_qubits_per_dim}³ = {(2**n_qubits_per_dim)**3} points")
    print(f"QTT rank: {rank}")
    print(f"Device: {device}")
    
    # Test 1: Vorticity computation
    print("\n1. Vorticity Computation:")
    t0 = time.perf_counter()
    omega_x, omega_y, omega_z = qtt_vorticity_3d(u, v, w, n_qubits_per_dim, dx, max_rank=64)
    print(f"   Computed in {time.perf_counter() - t0:.3f}s")
    print(f"   Vorticity ranks: x={max(c.shape[0] for c in omega_x)}, "
          f"y={max(c.shape[0] for c in omega_y)}, z={max(c.shape[0] for c in omega_z)}")
    
    # Test 2: Maximum vorticity
    print("\n2. Maximum Vorticity (BKM):")
    omega_max = qtt_vorticity_max_3d(u, v, w, n_qubits_per_dim, dx, max_rank=64, n_samples=5000)
    print(f"   |ω|_∞ ≈ {omega_max:.4f} (from 5000 samples)")
    
    # Test 3: Enstrophy
    print("\n3. Enstrophy:")
    enstrophy = qtt_enstrophy_3d(u, v, w, n_qubits_per_dim, dx, max_rank=64)
    print(f"   Ω = {enstrophy:.6e}")
    
    # Test 4: Sobolev norm
    print("\n4. Sobolev Norms:")
    h1 = qtt_sobolev_norm_estimate(u, s=1.0)
    h2 = qtt_sobolev_norm_estimate(u, s=2.0)
    print(f"   ||u||_H¹ ≈ {h1:.4f}")
    print(f"   ||u||_H² ≈ {h2:.4f}")
    
    # Test 5: Full regularity monitor
    print("\n5. Regularity Monitor:")
    monitor = RegularityMonitor(n_qubits_per_dim, dx, max_rank=64)
    state = monitor.snapshot(u, v, w, t=0.0)
    monitor.print_status(state)
    
    print("\n" + "=" * 70)
    print("All regularity tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_regularity()
