"""
Turbulence Physics Validation Suite
====================================

Validates that the QTT solver captures real turbulence physics:
1. Kolmogorov energy spectrum: E(k) ~ k^(-5/3) in inertial range
2. Structure functions and Kolmogorov's 4/5 law
3. Energy conservation and dissipation rates
4. Long-time statistical stationarity

The key question: Does O(log N) QTT compression preserve the essential
physics of turbulent flows?

Author: HyperTensor Team
Date: 2025
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math

import numpy as np
import torch
from torch import Tensor

from tensornet.cfd.ns3d_native import (
    NativeNS3DConfig,
    NativeNS3DSolver,
    taylor_green_native,
    QTT3DNative,
    QTT3DVectorNative,
)
from tensornet.cfd.qtt_native_ops import (
    QTTCores,
    qtt_inner_native,
)


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT SPECTRAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════════

def qtt_to_dense_small(qtt: QTT3DNative) -> Tensor:
    """
    Convert QTT to dense for small grids (validation only).
    
    WARNING: Only use for grids up to ~64³ for memory reasons.
    
    The QTT stores data in C-order (row-major) flattening of a 3D array.
    The n_sites = 3 * n_bits corresponds to the total bits for N³ = 2^(3*n_bits).
    """
    n_sites = qtt.cores.num_sites  # 3 * n_bits
    N = qtt.N
    n_bits = qtt.n_bits
    
    if N > 64:
        raise ValueError(f"Grid {N}³ too large for dense conversion. Use sampling instead.")
    
    # Contract QTT cores to get 1D vector in the same order it was stored
    # First core shape: (1, 2, r1)
    # Middle cores shape: (r_{k-1}, 2, r_k) 
    # Last core shape: (r_{n-1}, 2, 1)
    
    cores = qtt.cores.cores
    
    # Contract all cores left-to-right
    # Core 0: (1, 2, r1) -> (2, r1)
    result = cores[0].squeeze(0)  # (2, r1)
    
    for i, core in enumerate(cores[1:], 1):
        # result: (2^i, r_i)
        # core: (r_i, 2, r_{i+1})
        # Output: (2^i, 2, r_{i+1}) -> reshape to (2^{i+1}, r_{i+1})
        r_prev = result.shape[-1]
        r_curr = core.shape[0]
        
        if r_prev != r_curr:
            raise RuntimeError(f"Rank mismatch at core {i}: result has {r_prev}, core expects {r_curr}")
        
        # einsum: contract last dim of result with first dim of core
        contracted = torch.einsum('...r,rjk->...jk', result, core)
        # Flatten: (2^i, 2, r_{i+1}) -> (2^{i+1}, r_{i+1})
        result = contracted.reshape(-1, contracted.shape[-1])
    
    # Final result: (2^n_sites, 1) -> squeeze to (2^n_sites,)
    if result.shape[-1] == 1:
        result = result.squeeze(-1)
    else:
        raise RuntimeError(f"Final rank should be 1, got {result.shape[-1]}")
    
    # Result is in C-order (row-major): reshape directly to (N, N, N)
    expected_len = N ** 3
    if result.shape[0] != expected_len:
        raise RuntimeError(f"Expected length {expected_len}, got {result.shape[0]}")
    
    # Simple reshape - data is already in correct order
    dense = result.reshape(N, N, N)
    
    return dense


def compute_energy_spectrum_dense(u: QTT3DVectorNative) -> Tuple[Tensor, Tensor]:
    """
    Compute energy spectrum E(k) from velocity field.
    
    Uses FFT on dense representation (small grids only).
    
    Returns:
        k_bins: Wavenumber bins
        E_k: Energy per wavenumber bin
    """
    N = u.n_bits
    if N > 6:  # 64³ max
        raise ValueError("Use compute_energy_spectrum_sampled for large grids")
    
    grid_size = u.x.N
    device = u.x.device
    
    # Convert to dense
    ux = qtt_to_dense_small(u.x)
    uy = qtt_to_dense_small(u.y)
    uz = qtt_to_dense_small(u.z)
    
    # FFT
    ux_hat = torch.fft.fftn(ux)
    uy_hat = torch.fft.fftn(uy)
    uz_hat = torch.fft.fftn(uz)
    
    # Energy density in Fourier space
    E_k_3d = 0.5 * (torch.abs(ux_hat)**2 + torch.abs(uy_hat)**2 + torch.abs(uz_hat)**2)
    
    # Wavenumber magnitudes
    kx = torch.fft.fftfreq(grid_size, d=1.0/grid_size, device=device)
    ky = torch.fft.fftfreq(grid_size, d=1.0/grid_size, device=device)
    kz = torch.fft.fftfreq(grid_size, d=1.0/grid_size, device=device)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    K = torch.sqrt(KX**2 + KY**2 + KZ**2)
    
    # Bin by wavenumber magnitude
    k_max = int(grid_size // 2)
    k_bins = torch.arange(0, k_max, device=device, dtype=torch.float32)
    E_k = torch.zeros(k_max, device=device)
    
    for i in range(k_max):
        mask = (K >= i) & (K < i + 1)
        E_k[i] = E_k_3d[mask].sum()
    
    # Normalize
    E_k = E_k / (grid_size ** 3)
    
    return k_bins, E_k


def compute_energy_spectrum_sampled(
    u: QTT3DVectorNative,
    n_samples: int = 10000,
) -> Tuple[Tensor, Tensor]:
    """
    Estimate energy spectrum via Monte Carlo sampling in QTT format.
    
    For large grids where dense conversion is impossible.
    Uses random Fourier sampling to estimate E(k).
    """
    N = u.x.N
    n_bits = u.n_bits
    device = u.x.device
    dtype = u.x.dtype
    
    # Sample random spatial points
    indices = torch.randint(0, N, (n_samples, 3), device=device)
    
    # We need to evaluate QTT at these points
    # For now, use a simplified approach: sample correlations
    
    # TODO: Implement proper QTT point evaluation
    # For now, fall back to dense for small grids
    if N <= 64:
        return compute_energy_spectrum_dense(u)
    
    # For larger grids, return placeholder
    # This needs proper implementation of QTT point sampling
    k_max = N // 2
    k_bins = torch.arange(0, k_max, device=device, dtype=torch.float32)
    E_k = torch.zeros(k_max, device=device)
    
    print(f"WARNING: Spectral analysis for {N}³ requires QTT point sampling (not yet implemented)")
    
    return k_bins, E_k


def fit_kolmogorov_exponent(
    k_bins: Tensor,
    E_k: Tensor,
    k_min: int = 4,
    k_max: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Fit power law E(k) ~ k^α in the inertial range.
    
    Kolmogorov theory predicts α = -5/3 ≈ -1.667
    
    Returns:
        alpha: Fitted exponent
        r_squared: Goodness of fit
    """
    if k_max is None:
        k_max = len(k_bins) // 2
    
    # Select inertial range
    mask = (k_bins >= k_min) & (k_bins <= k_max) & (E_k > 0)
    k_inertial = k_bins[mask].cpu().numpy()
    E_inertial = E_k[mask].cpu().numpy()
    
    if len(k_inertial) < 3:
        return 0.0, 0.0
    
    # Log-log linear regression
    log_k = np.log(k_inertial)
    log_E = np.log(E_inertial)
    
    # Fit: log(E) = α * log(k) + c
    A = np.vstack([log_k, np.ones(len(log_k))]).T
    result = np.linalg.lstsq(A, log_E, rcond=None)
    alpha, c = result[0]
    
    # R² goodness of fit
    E_pred = alpha * log_k + c
    ss_res = np.sum((log_E - E_pred) ** 2)
    ss_tot = np.sum((log_E - np.mean(log_E)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return float(alpha), float(r_squared)


# ═══════════════════════════════════════════════════════════════════════════════════════
# VALIDATION SUITE
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class TurbulenceValidationResult:
    """Results from turbulence validation."""
    grid_size: int
    n_steps: int
    total_time: float
    ms_per_step: float
    
    # Energy
    initial_energy: float
    final_energy: float
    energy_ratio: float
    analytic_ratio: float
    energy_error_pct: float
    
    # Spectral
    kolmogorov_exponent: float
    kolmogorov_r_squared: float
    has_inertial_range: bool
    
    # Compression
    compression_ratio: float
    max_rank: int
    
    # Stability
    is_stable: bool
    energy_monotonic: bool


def validate_turbulence_physics(
    n_bits: int = 5,
    n_steps: int = 100,
    nu: float = 1e-3,
    max_rank: int = 16,
    dt: float = 0.001,
    device: str = 'cuda',
    verbose: bool = True,
) -> TurbulenceValidationResult:
    """
    Comprehensive turbulence physics validation.
    
    Tests:
    1. Energy conservation (vs analytic decay)
    2. Kolmogorov spectrum (E(k) ~ k^(-5/3))
    3. Long-time stability
    4. Compression efficiency
    """
    N = 1 << n_bits
    
    if verbose:
        print("=" * 70)
        print(f"TURBULENCE VALIDATION: {N}³ grid, {n_steps} steps")
        print("=" * 70)
    
    # Configure solver
    config = NativeNS3DConfig(
        n_bits=n_bits,
        nu=nu,
        max_rank=max_rank,
        dt=dt,
        device=device,
    )
    
    # Initialize
    u, omega = taylor_green_native(
        n_bits=n_bits,
        max_rank=max_rank,
        device=device,
    )
    
    solver = NativeNS3DSolver(config)
    solver.initialize(u, omega)
    
    E0 = solver.diagnostics.kinetic_energy_qtt
    energies = [E0]
    
    if verbose:
        print(f"Initial energy: {E0:.6f}")
        print(f"Initial compression: {solver.diagnostics.compression_ratio:.1f}x")
    
    # Time evolution
    start_time = time.time()
    
    for i in range(n_steps):
        try:
            diag = solver.step(use_rk2=False, project=False)
            energies.append(diag.kinetic_energy_qtt)
            
            if verbose and (i + 1) % (n_steps // 5) == 0:
                ratio = diag.kinetic_energy_qtt / E0
                print(f"  Step {i+1}/{n_steps}: E/E0={ratio:.4f}, "
                      f"ranks u={diag.max_rank_u}, ω={diag.max_rank_omega}")
        except Exception as e:
            if verbose:
                print(f"  Step {i+1} failed: {e}")
            break
    
    elapsed = time.time() - start_time
    ms_per_step = elapsed * 1000 / n_steps
    
    # Final diagnostics
    final_diag = solver.diagnostics
    E_final = final_diag.kinetic_energy_qtt
    t_final = solver.t
    
    # Analytic prediction
    analytic_ratio = np.exp(-2 * nu * t_final)
    actual_ratio = E_final / E0
    energy_error = abs(actual_ratio - analytic_ratio) * 100
    
    # Check stability
    is_stable = not (np.isnan(E_final) or np.isinf(E_final))
    energy_monotonic = all(e1 >= e2 * 0.999 for e1, e2 in zip(energies[:-1], energies[1:]))
    
    # Spectral analysis (small grids only)
    kolmogorov_exp = 0.0
    kolmogorov_r2 = 0.0
    has_inertial = False
    
    if n_bits <= 6 and is_stable:
        try:
            k_bins, E_k = compute_energy_spectrum_dense(solver.u)
            kolmogorov_exp, kolmogorov_r2 = fit_kolmogorov_exponent(k_bins, E_k)
            # Check if close to -5/3
            has_inertial = abs(kolmogorov_exp - (-5/3)) < 0.5 and kolmogorov_r2 > 0.8
            
            if verbose:
                print(f"\nSpectral analysis:")
                print(f"  Fitted exponent: {kolmogorov_exp:.3f} (Kolmogorov: -1.667)")
                print(f"  R² fit quality: {kolmogorov_r2:.3f}")
                print(f"  Inertial range detected: {has_inertial}")
        except Exception as e:
            if verbose:
                print(f"Spectral analysis failed: {e}")
    
    result = TurbulenceValidationResult(
        grid_size=N,
        n_steps=n_steps,
        total_time=elapsed,
        ms_per_step=ms_per_step,
        initial_energy=E0,
        final_energy=E_final,
        energy_ratio=actual_ratio,
        analytic_ratio=analytic_ratio,
        energy_error_pct=energy_error,
        kolmogorov_exponent=kolmogorov_exp,
        kolmogorov_r_squared=kolmogorov_r2,
        has_inertial_range=has_inertial,
        compression_ratio=final_diag.compression_ratio,
        max_rank=final_diag.max_rank_u,
        is_stable=is_stable,
        energy_monotonic=energy_monotonic,
    )
    
    if verbose:
        print(f"\n--- Results ---")
        print(f"Time: {elapsed:.1f}s ({ms_per_step:.1f} ms/step)")
        print(f"Energy: {actual_ratio:.4f} (analytic: {analytic_ratio:.4f})")
        print(f"Error: {energy_error:.2f}%")
        print(f"Compression: {final_diag.compression_ratio:.1f}x")
        print(f"Stable: {is_stable}, Monotonic: {energy_monotonic}")
    
    return result


def scaling_benchmark(
    n_bits_list: List[int] = [5, 6, 7],
    n_steps: int = 20,
    max_rank: int = 16,
    verbose: bool = True,
) -> List[TurbulenceValidationResult]:
    """
    Benchmark scaling across grid sizes.
    
    Tests O(log N) complexity claim.
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    results = []
    
    if verbose:
        print("=" * 70)
        print("SCALING BENCHMARK")
        print("=" * 70)
    
    for n_bits in n_bits_list:
        N = 1 << n_bits
        if verbose:
            print(f"\n--- {N}³ ({N**3:,} cells) ---")
        
        result = validate_turbulence_physics(
            n_bits=n_bits,
            n_steps=n_steps,
            max_rank=max_rank,
            verbose=False,
        )
        results.append(result)
        
        if verbose:
            print(f"  Time: {result.ms_per_step:.0f} ms/step")
            print(f"  Compression: {result.compression_ratio:.0f}x")
            print(f"  Energy error: {result.energy_error_pct:.2f}%")
            print(f"  Stable: {result.is_stable}")
    
    if verbose and len(results) >= 2:
        print(f"\n--- Scaling Summary ---")
        for i in range(1, len(results)):
            cells_ratio = (results[i].grid_size / results[i-1].grid_size) ** 3
            time_ratio = results[i].ms_per_step / results[i-1].ms_per_step
            print(f"{results[i-1].grid_size}³ → {results[i].grid_size}³ "
                  f"({cells_ratio:.0f}x cells): {time_ratio:.2f}x time")
        
        total_cells = (results[-1].grid_size / results[0].grid_size) ** 3
        total_time = results[-1].ms_per_step / results[0].ms_per_step
        print(f"\nTotal: {total_cells:.0f}x cells → {total_time:.2f}x time")
        print(f"Ideal O(N): {total_cells:.0f}x, Actual: {total_time:.2f}x")
        print(f"Complexity factor: O(N^{np.log(total_time)/np.log(total_cells):.3f})")
    
    return results


def long_time_stability_test(
    n_bits: int = 5,
    n_steps: int = 500,
    max_rank: int = 16,
    verbose: bool = True,
) -> Dict:
    """
    Long-time stability test.
    
    Checks if solution remains stable and energy decays properly
    over many timesteps.
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    N = 1 << n_bits
    
    if verbose:
        print("=" * 70)
        print(f"LONG-TIME STABILITY: {N}³, {n_steps} steps")
        print("=" * 70)
    
    config = NativeNS3DConfig(
        n_bits=n_bits,
        nu=1e-3,
        max_rank=max_rank,
        dt=0.002,
        device='cuda',
    )
    
    u, omega = taylor_green_native(
        n_bits=n_bits,
        max_rank=max_rank,
        device='cuda',
    )
    
    solver = NativeNS3DSolver(config)
    solver.initialize(u, omega)
    
    E0 = solver.diagnostics.kinetic_energy_qtt
    
    energies = []
    times = []
    ranks = []
    
    start = time.time()
    
    for i in range(n_steps):
        try:
            diag = solver.step(use_rk2=False, project=False)
            energies.append(diag.kinetic_energy_qtt)
            times.append(solver.t)
            ranks.append(diag.max_rank_u)
            
            if verbose and (i + 1) % 100 == 0:
                ratio = diag.kinetic_energy_qtt / E0
                elapsed = time.time() - start
                print(f"  Step {i+1}: E/E0={ratio:.4f}, rank={diag.max_rank_u}, "
                      f"elapsed={elapsed:.1f}s")
        except Exception as e:
            if verbose:
                print(f"  FAILED at step {i+1}: {e}")
            break
    
    elapsed = time.time() - start
    
    # Analysis
    energies = np.array(energies)
    times = np.array(times)
    ranks = np.array(ranks)
    
    is_stable = len(energies) == n_steps and not np.any(np.isnan(energies))
    
    if is_stable:
        # Energy should decay as exp(-2*nu*t)
        analytic = E0 * np.exp(-2 * config.nu * times)
        rel_error = np.abs(energies - analytic) / analytic
        max_error = np.max(rel_error) * 100
        mean_error = np.mean(rel_error) * 100
        
        # Check monotonicity (with small tolerance for numerical noise)
        monotonic = np.all(np.diff(energies) <= 0.01 * energies[:-1])
    else:
        max_error = float('inf')
        mean_error = float('inf')
        monotonic = False
    
    result = {
        'grid_size': N,
        'n_steps': len(energies),
        'target_steps': n_steps,
        'elapsed_s': elapsed,
        'ms_per_step': elapsed * 1000 / max(len(energies), 1),
        'is_stable': is_stable,
        'is_monotonic': monotonic,
        'max_error_pct': max_error if is_stable else float('inf'),
        'mean_error_pct': mean_error if is_stable else float('inf'),
        'final_energy_ratio': energies[-1] / E0 if len(energies) > 0 else 0,
        'max_rank': int(np.max(ranks)) if len(ranks) > 0 else 0,
        'mean_rank': float(np.mean(ranks)) if len(ranks) > 0 else 0,
    }
    
    if verbose:
        print(f"\n--- Results ---")
        print(f"Completed: {result['n_steps']}/{n_steps} steps")
        print(f"Stable: {is_stable}, Monotonic: {monotonic}")
        print(f"Time: {elapsed:.1f}s ({result['ms_per_step']:.1f} ms/step)")
        if is_stable:
            print(f"Max error: {max_error:.2f}%, Mean error: {mean_error:.2f}%")
            print(f"Final E/E0: {result['final_energy_ratio']:.4f}")
            print(f"Ranks: max={result['max_rank']}, mean={result['mean_rank']:.1f}")
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_full_validation():
    """Run comprehensive validation suite."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " QTT TURBULENCE SOLVER - PHYSICS VALIDATION ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # 1. Scaling test
    print("=" * 70)
    print("TEST 1: SCALING (O(log N) verification)")
    print("=" * 70)
    scaling_results = scaling_benchmark(
        n_bits_list=[5, 6, 7],
        n_steps=20,
        verbose=True,
    )
    
    # 2. Long-time stability
    print("\n")
    long_time = long_time_stability_test(
        n_bits=5,
        n_steps=200,
        verbose=True,
    )
    
    # 3. Spectral analysis at 32³ and 64³
    print("\n")
    print("=" * 70)
    print("TEST 3: SPECTRAL ANALYSIS (Kolmogorov k^(-5/3))")
    print("=" * 70)
    
    for n_bits in [5, 6]:
        N = 1 << n_bits
        print(f"\n--- {N}³ grid ---")
        result = validate_turbulence_physics(
            n_bits=n_bits,
            n_steps=50,
            verbose=True,
        )
    
    # Summary
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " VALIDATION SUMMARY ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("\nScaling:")
    for r in scaling_results:
        print(f"  {r.grid_size}³: {r.ms_per_step:.0f} ms/step, "
              f"{r.compression_ratio:.0f}x compression, "
              f"{r.energy_error_pct:.2f}% error")
    
    print(f"\nLong-time ({long_time['n_steps']} steps):")
    print(f"  Stable: {long_time['is_stable']}")
    print(f"  Mean error: {long_time['mean_error_pct']:.2f}%")
    
    print("\nConclusions:")
    
    # Check O(log N)
    if len(scaling_results) >= 2:
        total_cells = (scaling_results[-1].grid_size / scaling_results[0].grid_size) ** 3
        total_time = scaling_results[-1].ms_per_step / scaling_results[0].ms_per_step
        complexity = np.log(total_time) / np.log(total_cells)
        
        if complexity < 0.2:
            print("  ✓ O(log N) complexity CONFIRMED")
        else:
            print(f"  ✗ Scaling is O(N^{complexity:.2f}), not O(log N)")
    
    if long_time['is_stable']:
        print("  ✓ Long-time stability CONFIRMED")
    else:
        print("  ✗ Long-time stability FAILED")
    
    if long_time['mean_error_pct'] < 5:
        print("  ✓ Energy conservation <5% error")
    else:
        print(f"  ✗ Energy error {long_time['mean_error_pct']:.1f}% > 5%")


if __name__ == "__main__":
    run_full_validation()
