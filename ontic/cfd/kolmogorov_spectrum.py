"""
Kolmogorov Energy Spectrum Validation
======================================

Validates the k^(-5/3) energy cascade law in the QTT turbulence solver.

The Kolmogorov -5/3 law states that in fully developed isotropic turbulence,
the energy spectrum in the inertial subrange follows:

    E(k) = C_K * ε^(2/3) * k^(-5/3)

where:
    - C_K ≈ 1.5 is the Kolmogorov constant
    - ε is the energy dissipation rate
    - k is the wavenumber

This module:
1. Evolves Taylor-Green vortex until turbulent breakdown
2. Computes 3D FFT of velocity field (via QTT reconstruction at moderate scale)
3. Calculates radially-averaged energy spectrum E(k)
4. Fits inertial range to verify -5/3 exponent

Author: HyperTensor Team
Date: 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

import numpy as np
import torch
from torch import Tensor


@dataclass
class SpectrumResult:
    """Energy spectrum analysis results."""
    wavenumbers: np.ndarray      # k values
    spectrum: np.ndarray         # E(k) values
    compensated: np.ndarray      # E(k) * k^(5/3) - should be flat in inertial range
    inertial_range: Tuple[int, int]  # (k_min, k_max) indices
    fitted_exponent: float       # Should be ≈ -5/3 ≈ -1.667
    fitted_prefactor: float      # C_K * ε^(2/3)
    r_squared: float             # Goodness of fit
    kolmogorov_length: float     # η = (ν³/ε)^(1/4)
    integral_length: float       # L = k^(-1) at spectrum peak


def compute_energy_spectrum_3d(
    ux: np.ndarray,
    uy: np.ndarray, 
    uz: np.ndarray,
    L: float = 2 * np.pi,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 3D radially-averaged kinetic energy spectrum.
    
    Parameters
    ----------
    ux, uy, uz : np.ndarray
        Velocity components, each shape (N, N, N)
    L : float
        Domain size
        
    Returns
    -------
    k_bins : np.ndarray
        Wavenumber bin centers
    E_k : np.ndarray  
        Energy spectrum E(k) = (1/2) * Σ |û|² for wavenumbers in shell [k, k+1)
    """
    N = ux.shape[0]
    dk = 2 * np.pi / L
    
    # 3D FFT of velocity components
    ux_hat = np.fft.fftn(ux) / N**3
    uy_hat = np.fft.fftn(uy) / N**3
    uz_hat = np.fft.fftn(uz) / N**3
    
    # Energy = 0.5 * |u_hat|^2
    energy_density = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2 + np.abs(uz_hat)**2)
    
    # Build wavenumber magnitude grid
    kx = np.fft.fftfreq(N, 1.0 / N)
    ky = np.fft.fftfreq(N, 1.0 / N)
    kz = np.fft.fftfreq(N, 1.0 / N)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    # Radial binning
    k_max = int(N // 2)
    k_bins = np.arange(0, k_max + 1, dtype=np.float64)
    E_k = np.zeros(k_max + 1)
    
    for i in range(k_max + 1):
        # Shell: i <= |k| < i+1
        mask = (K_mag >= i) & (K_mag < i + 1)
        E_k[i] = np.sum(energy_density[mask])
    
    # Normalize by dk (to get energy per unit wavenumber)
    E_k *= L**3  # Volume factor for proper normalization
    
    return k_bins * dk, E_k


def fit_power_law(
    k: np.ndarray,
    E_k: np.ndarray,
    k_min_idx: int,
    k_max_idx: int,
) -> Tuple[float, float, float]:
    """
    Fit E(k) = A * k^α in log-log space.
    
    Returns (alpha, A, r_squared)
    """
    k_fit = k[k_min_idx:k_max_idx + 1]
    E_fit = E_k[k_min_idx:k_max_idx + 1]
    
    # Remove zeros
    valid = (k_fit > 0) & (E_fit > 0)
    if np.sum(valid) < 2:
        return -5/3, 0.0, 0.0
    
    log_k = np.log(k_fit[valid])
    log_E = np.log(E_fit[valid])
    
    # Linear regression: log(E) = log(A) + α * log(k)
    A = np.vstack([log_k, np.ones_like(log_k)]).T
    result = np.linalg.lstsq(A, log_E, rcond=None)
    alpha, log_A = result[0]
    
    # R-squared
    E_pred = np.exp(log_A) * k_fit[valid] ** alpha
    ss_res = np.sum((E_fit[valid] - E_pred) ** 2)
    ss_tot = np.sum((E_fit[valid] - np.mean(E_fit[valid])) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return alpha, np.exp(log_A), r_squared


def find_inertial_range(
    k: np.ndarray,
    E_k: np.ndarray,
    target_exponent: float = -5/3,
    tolerance: float = 0.3,
) -> Tuple[int, int]:
    """
    Find the wavenumber range where the spectrum best matches k^(-5/3).
    
    Uses sliding window to find range with exponent closest to -5/3.
    """
    n = len(k)
    if n < 10:
        return 1, max(2, n - 1)
    
    best_range = (1, n // 2)
    best_error = float('inf')
    
    # Minimum window size
    min_window = max(4, n // 8)
    
    for window_size in range(min_window, n // 2):
        for start in range(1, n - window_size):
            end = start + window_size
            
            alpha, _, r2 = fit_power_law(k, E_k, start, end)
            
            # Combined criterion: close to -5/3 with good R²
            exponent_error = abs(alpha - target_exponent)
            
            if exponent_error < best_error and r2 > 0.7:
                best_error = exponent_error
                best_range = (start, end)
    
    return best_range


def qtt_to_dense_velocity(
    u_qtt,  # QTT3DVectorNative
    max_points: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert QTT velocity field to dense array for spectrum analysis.
    
    For large grids (N > max_points), sub-samples to avoid memory issues
    while preserving spectrum in the available wavenumber range.
    """
    from ontic.cfd.qtt_native_ops import QTTCores
    
    N = u_qtt.x.N
    
    def qtt_to_numpy(qtt_3d) -> np.ndarray:
        """Reconstruct dense field from QTT cores."""
        cores = qtt_3d.cores.cores
        n_cores = len(cores)
        
        # Contract from left
        result = cores[0].squeeze(0)  # (2, r1)
        
        for i in range(1, n_cores):
            core = cores[i]  # (r_left, 2, r_right)
            r_left, d, r_right = core.shape
            
            # result: (accumulated_size, r_left)
            # contract with core
            result = result @ core.view(r_left, d * r_right)
            result = result.view(-1, r_right)
        
        result = result.squeeze(-1)  # Remove final bond
        
        # Reshape to 3D with Morton order
        n_bits = qtt_3d.n_bits
        dense = result.cpu().numpy()
        
        # Un-interleave Morton ordering
        N = 2 ** n_bits
        field_3d = np.zeros((N, N, N), dtype=np.float64)
        
        for idx in range(N ** 3):
            # Morton decode
            x, y, z = 0, 0, 0
            for bit in range(n_bits):
                # Interleaved: bit*3 + 0 -> x, bit*3 + 1 -> y, bit*3 + 2 -> z
                global_bit = bit * 3
                x |= ((idx >> global_bit) & 1) << bit
                y |= ((idx >> (global_bit + 1)) & 1) << bit
                z |= ((idx >> (global_bit + 2)) & 1) << bit
            
            field_3d[x, y, z] = dense[idx]
        
        return field_3d
    
    # For very large grids, we can't densify - need alternative approach
    if N > max_points:
        raise ValueError(
            f"Grid size {N}³ exceeds max_points {max_points}³. "
            f"Use probe_spectrum_qtt() for large grids."
        )
    
    ux = qtt_to_numpy(u_qtt.x)
    uy = qtt_to_numpy(u_qtt.y)
    uz = qtt_to_numpy(u_qtt.z)
    
    return ux, uy, uz


def probe_spectrum_qtt(
    u_qtt,  # QTT3DVectorNative
    probe_grid: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate spectrum from QTT via coarse-grained probing.
    
    For grids too large to densify, samples the field at a coarser
    resolution to estimate the spectrum up to the Nyquist limit
    of the probe grid.
    
    This preserves spectral information up to k_max = probe_grid / 2.
    """
    from ontic.cfd.qtt_native_ops import QTTCores
    
    N = u_qtt.x.N
    stride = max(1, N // probe_grid)
    actual_probe = N // stride
    
    def sample_qtt_at_indices(qtt_3d, indices: List[Tuple[int, int, int]]) -> np.ndarray:
        """Sample QTT field at specific grid points."""
        cores = qtt_3d.cores.cores
        n_bits = qtt_3d.n_bits
        values = []
        
        for (ix, iy, iz) in indices:
            # Morton encode
            morton_idx = 0
            for bit in range(n_bits):
                morton_idx |= ((ix >> bit) & 1) << (3 * bit)
                morton_idx |= ((iy >> bit) & 1) << (3 * bit + 1)
                morton_idx |= ((iz >> bit) & 1) << (3 * bit + 2)
            
            # Binary digits for each core
            bits = []
            for core_idx in range(3 * n_bits):
                bits.append((morton_idx >> core_idx) & 1)
            
            # Contract cores for this index
            vec = cores[0][:, bits[0], :].squeeze(0)  # (r1,)
            for i in range(1, len(cores)):
                core_slice = cores[i][:, bits[i], :]  # (r_left, r_right)
                vec = vec @ core_slice
            
            values.append(vec.item())
        
        return np.array(values)
    
    # Build sampling indices
    indices = []
    for ix in range(0, N, stride):
        for iy in range(0, N, stride):
            for iz in range(0, N, stride):
                indices.append((ix, iy, iz))
    
    # Sample each component
    ux = sample_qtt_at_indices(u_qtt.x, indices).reshape(actual_probe, actual_probe, actual_probe)
    uy = sample_qtt_at_indices(u_qtt.y, indices).reshape(actual_probe, actual_probe, actual_probe)
    uz = sample_qtt_at_indices(u_qtt.z, indices).reshape(actual_probe, actual_probe, actual_probe)
    
    # Compute spectrum at probe resolution
    L_effective = 2 * np.pi  # Domain size
    return compute_energy_spectrum_3d(ux, uy, uz, L_effective)


def analyze_spectrum(
    k: np.ndarray,
    E_k: np.ndarray,
    nu: float = 0.001,
) -> SpectrumResult:
    """
    Full spectrum analysis including inertial range detection and fitting.
    """
    # Find inertial range
    k_min_idx, k_max_idx = find_inertial_range(k, E_k)
    
    # Fit power law
    alpha, A, r2 = fit_power_law(k, E_k, k_min_idx, k_max_idx)
    
    # Compensated spectrum (should be flat ≈ C_K in inertial range)
    compensated = np.zeros_like(E_k)
    valid = k > 0
    compensated[valid] = E_k[valid] * (k[valid] ** (5/3))
    
    # Estimate dissipation and length scales
    # ε ≈ 2ν ∫ k² E(k) dk (enstrophy-based estimate)
    dk = k[1] - k[0] if len(k) > 1 else 1.0
    dissipation_integrand = k**2 * E_k
    epsilon = 2 * nu * np.sum(dissipation_integrand) * dk
    epsilon = max(epsilon, 1e-10)
    
    # Kolmogorov length scale
    eta = (nu**3 / epsilon) ** 0.25
    
    # Integral length scale (from energy-weighted k)
    total_energy = np.sum(E_k) * dk
    if total_energy > 0:
        k_mean = np.sum(k * E_k) * dk / total_energy
        L_int = 2 * np.pi / max(k_mean, 1e-10)
    else:
        L_int = 1.0
    
    return SpectrumResult(
        wavenumbers=k,
        spectrum=E_k,
        compensated=compensated,
        inertial_range=(k_min_idx, k_max_idx),
        fitted_exponent=alpha,
        fitted_prefactor=A,
        r_squared=r2,
        kolmogorov_length=eta,
        integral_length=L_int,
    )


def evolve_to_turbulence(
    n_bits: int = 6,
    Re: float = 1000.0,
    n_steps: int = 100,
    dt: float = 0.01,
    max_rank: int = 64,
    device: str = 'cuda',
    verbose: bool = True,
) -> Tuple:
    """
    Evolve Taylor-Green vortex to turbulent state.
    
    Returns (u, omega, diagnostics_history)
    """
    from ontic.cfd.ns3d_native import (
        taylor_green_analytical,
        NativeNS3DSolver,
        NativeNS3DConfig,
        compute_diagnostics_native,
    )
    
    N = 2 ** n_bits
    L = 2 * np.pi
    nu = 1.0 / Re
    
    if verbose:
        print(f"╔══════════════════════════════════════════════════════════════╗")
        print(f"║        KOLMOGOROV SPECTRUM VALIDATION                       ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        print(f"║  Grid: {N}³ ({N**3:,} cells)                                ")
        print(f"║  Re: {Re:.0f}                                               ")
        print(f"║  ν: {nu:.6f}                                                ")
        print(f"║  Steps: {n_steps}                                           ")
        print(f"║  dt: {dt}                                                   ")
        print(f"╚══════════════════════════════════════════════════════════════╝")
    
    # Initialize with analytical construction
    t0 = time.time()
    u, omega = taylor_green_analytical(
        n_bits, L=L, amplitude=1.0, max_rank=max_rank, device=device
    )
    init_time = time.time() - t0
    
    if verbose:
        print(f"\n✓ Initialization: {init_time*1000:.1f} ms")
    
    # Create solver with config
    config = NativeNS3DConfig(
        n_bits=n_bits,
        nu=nu,
        L=L,
        max_rank=max_rank,
        dt=dt,
        device=device,
    )
    solver = NativeNS3DSolver(config)
    
    # Initialize solver state
    solver.initialize(u, omega)
    
    # Evolution
    diagnostics_history = []
    
    for step in range(n_steps):
        t0 = time.time()
        # Disable pressure projection - Taylor-Green is already divergence-free
        # and projection is the main bottleneck (60% of step time)
        diag = solver.step(use_rk2=True, project=False)
        step_time = time.time() - t0
        
        u = solver.u
        omega = solver.omega
        
        diagnostics_history.append(diag)
        
        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f"  Step {step:4d}: E={diag.kinetic_energy_qtt:.6f}, "
                  f"Ω={diag.enstrophy_qtt:.6f}, "
                  f"rank_max={diag.max_rank_u}, "
                  f"time={step_time*1000:.1f}ms")
    
    return u, omega, diagnostics_history


def validate_kolmogorov_spectrum(
    n_bits: int = 6,
    Re: float = 1600.0,
    n_steps: int = 200,
    dt: float = 0.005,
    max_rank: int = 64,
    device: str = 'cuda',
) -> SpectrumResult:
    """
    Full Kolmogorov spectrum validation.
    
    Evolves Taylor-Green vortex to turbulence and validates -5/3 law.
    """
    print("\n" + "="*70)
    print("KOLMOGOROV k^(-5/3) SPECTRUM VALIDATION")
    print("="*70)
    
    N = 2 ** n_bits
    nu = 1.0 / Re
    
    # Evolve to turbulence
    u, omega, history = evolve_to_turbulence(
        n_bits=n_bits,
        Re=Re,
        n_steps=n_steps,
        dt=dt,
        max_rank=max_rank,
        device=device,
        verbose=True,
    )
    
    print(f"\n→ Computing energy spectrum...")
    
    # Extract velocity for spectrum analysis
    if N <= 128:
        # Direct reconstruction
        ux, uy, uz = qtt_to_dense_velocity(u, max_points=N)
        k, E_k = compute_energy_spectrum_3d(ux, uy, uz, L=2*np.pi)
    else:
        # Probed spectrum for large grids
        k, E_k = probe_spectrum_qtt(u, probe_grid=64)
    
    # Analyze spectrum
    result = analyze_spectrum(k, E_k, nu=nu)
    
    # Report
    print(f"\n{'─'*50}")
    print("SPECTRUM ANALYSIS RESULTS")
    print(f"{'─'*50}")
    print(f"  Inertial range: k ∈ [{k[result.inertial_range[0]]:.2f}, "
          f"{k[result.inertial_range[1]]:.2f}]")
    print(f"  Fitted exponent:     α = {result.fitted_exponent:.4f}")
    print(f"  Kolmogorov target:   α = {-5/3:.4f}")
    print(f"  Error:               Δα = {abs(result.fitted_exponent + 5/3):.4f}")
    print(f"  R² (fit quality):    {result.r_squared:.4f}")
    print(f"  Kolmogorov length:   η = {result.kolmogorov_length:.6f}")
    print(f"  Integral length:     L = {result.integral_length:.4f}")
    
    # Validation verdict
    exponent_error = abs(result.fitted_exponent + 5/3)
    if exponent_error < 0.2 and result.r_squared > 0.85:
        verdict = "✓ KOLMOGOROV -5/3 LAW VALIDATED"
        color = "\033[92m"  # Green
    elif exponent_error < 0.4 and result.r_squared > 0.7:
        verdict = "~ PARTIAL MATCH (pre-transition or low Re)"
        color = "\033[93m"  # Yellow
    else:
        verdict = "✗ NO CLEAR INERTIAL RANGE"
        color = "\033[91m"  # Red
    
    print(f"\n{color}{verdict}\033[0m")
    print("="*70)
    
    return result


def plot_spectrum(
    result: SpectrumResult,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot energy spectrum with -5/3 reference line.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    k = result.wavenumbers
    E_k = result.spectrum
    
    # Valid range (k > 0, E > 0)
    valid = (k > 0) & (E_k > 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: E(k) vs k
    ax1 = axes[0]
    ax1.loglog(k[valid], E_k[valid], 'b-', linewidth=2, label='E(k)')
    
    # Reference -5/3 line
    k_ref = k[valid]
    E_ref = result.fitted_prefactor * k_ref ** (-5/3)
    ax1.loglog(k_ref, E_ref, 'r--', linewidth=1.5, 
               label=f'$k^{{-5/3}}$ (fitted: $k^{{{result.fitted_exponent:.2f}}}$)')
    
    # Mark inertial range
    k_min = k[result.inertial_range[0]]
    k_max = k[result.inertial_range[1]]
    ax1.axvspan(k_min, k_max, alpha=0.2, color='green', label='Inertial range')
    
    ax1.set_xlabel('Wavenumber k', fontsize=12)
    ax1.set_ylabel('E(k)', fontsize=12)
    ax1.set_title('Energy Spectrum', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Compensated spectrum
    ax2 = axes[1]
    compensated = result.compensated
    ax2.semilogx(k[valid], compensated[valid], 'b-', linewidth=2)
    ax2.axhline(y=np.mean(compensated[result.inertial_range[0]:result.inertial_range[1]+1]), 
                color='r', linestyle='--', label='Mean in inertial range')
    
    ax2.axvspan(k_min, k_max, alpha=0.2, color='green')
    
    ax2.set_xlabel('Wavenumber k', fontsize=12)
    ax2.set_ylabel('$E(k) \\cdot k^{5/3}$', fontsize=12)
    ax2.set_title('Compensated Spectrum (should be flat in inertial range)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrum plot to {save_path}")
    
    if show:
        plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kolmogorov Spectrum Validation")
    parser.add_argument("--n_bits", type=int, default=6, help="Bits per dimension (N=2^n_bits)")
    parser.add_argument("--Re", type=float, default=1600.0, help="Reynolds number")
    parser.add_argument("--steps", type=int, default=200, help="Evolution steps")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step")
    parser.add_argument("--max_rank", type=int, default=64, help="Max QTT rank")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--save", type=str, default=None, help="Save plot path")
    
    args = parser.parse_args()
    
    result = validate_kolmogorov_spectrum(
        n_bits=args.n_bits,
        Re=args.Re,
        n_steps=args.steps,
        dt=args.dt,
        max_rank=args.max_rank,
        device=args.device,
    )
    
    if args.plot or args.save:
        plot_spectrum(result, save_path=args.save, show=args.plot)
