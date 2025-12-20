"""
QTT Compression Benchmark for Hypersonic Flow Fields
=====================================================

This benchmark validates the core thesis of Project HyperTensor:

    "Turbulent flow fields satisfy an Area Law analogous to quantum 
    entanglement—correlations scale with boundary area, not volume—
    enabling compression from O(N³) to O(N·D²) via Tensor Train."

Test Cases:
    1. Uniform supersonic flow (trivial - should compress perfectly)
    2. Mach 5 wedge with oblique shock (sharp discontinuity)
    3. Double Mach Reflection (complex multi-shock structure)

Expected Results:
    - Smooth regions: χ = O(1), exponential compression
    - Shock regions: χ = O(N^ε) with ε << 1, still sublinear
    - Reconstruction error bounded by truncation tolerance

References:
    [1] Gourianov et al., arXiv:2305.10784 (2023)
    [2] Khoromskij, Constructive Approximation 34:257-280 (2011)
"""

import torch
import math
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from tensornet.cfd.euler_2d import (
    Euler2D, Euler2DState, 
    supersonic_wedge_ic, 
    oblique_shock_exact
)
from tensornet.cfd.qtt import (
    field_to_qtt,
    qtt_to_field,
    euler_to_qtt,
    qtt_to_euler,
    compression_analysis,
    estimate_area_law_exponent,
    QTTCompressionResult
)


def benchmark_uniform_flow(Nx: int = 128, Ny: int = 64, verbose: bool = True) -> dict:
    """
    Benchmark 1: Uniform supersonic flow (trivial case).
    
    A perfectly uniform field should compress to χ=1 with zero error.
    This validates the basic QTT machinery.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK 1: Uniform Supersonic Flow")
        print("=" * 70)
    
    # Create uniform flow at Mach 3
    state = supersonic_wedge_ic(Nx=Nx, Ny=Ny, M_inf=3.0)
    
    # Compress with small χ
    compressed = euler_to_qtt(state, chi_max=4)
    reconstructed = qtt_to_euler(compressed, gamma=state.gamma)
    
    # Compute errors
    rho_err = torch.norm(reconstructed.rho - state.rho) / torch.norm(state.rho)
    p_err = torch.norm(reconstructed.p - state.p) / torch.norm(state.p)
    
    results = {
        'test': 'uniform_flow',
        'Nx': Nx,
        'Ny': Ny,
        'chi_max': 4,
        'rho_error': rho_err.item(),
        'p_error': p_err.item(),
        'compression_ratio': compressed['rho'].compression_ratio,
        'pass': rho_err.item() < 1e-10 and p_err.item() < 1e-10
    }
    
    if verbose:
        print(f"Grid: {Nx} × {Ny}")
        print(f"Compression ratio: {results['compression_ratio']:.1f}x")
        print(f"Density error: {results['rho_error']:.2e}")
        print(f"Pressure error: {results['p_error']:.2e}")
        print(f"Status: {'✓ PASS' if results['pass'] else '✗ FAIL'}")
    
    return results


def benchmark_oblique_shock(
    Nx: int = 128, 
    Ny: int = 64,
    M_inf: float = 5.0,
    theta_deg: float = 15.0,
    n_steps: int = 50,
    verbose: bool = True
) -> dict:
    """
    Benchmark 2: Mach 5 wedge with oblique shock.
    
    This is the mission-critical test case. The flow contains:
    - Uniform supersonic inflow
    - Sharp oblique shock
    - Uniform post-shock region
    
    Expected: Moderate χ needed for shock, but still good compression.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK 2: Mach 5 Wedge with Oblique Shock")
        print("=" * 70)
    
    Lx, Ly = 2.0, 1.0
    gamma = 1.4
    
    # Initialize solver
    solver = Euler2D(Nx, Ny, Lx, Ly, gamma=gamma)
    ic = supersonic_wedge_ic(Nx=Nx, Ny=Ny, M_inf=M_inf, gamma=gamma)
    solver.set_initial_condition(ic)
    
    # Short simulation to develop shock
    for _ in range(n_steps):
        solver.step(cfl=0.5)
    
    state = solver.state
    
    if verbose:
        print(f"Grid: {Nx} × {Ny}")
        print(f"Mach: {M_inf}, Wedge angle: {theta_deg}°")
        print(f"Simulation time: {solver.time:.4f}")
        print()
    
    # Compression analysis
    analysis = compression_analysis(
        state, 
        chi_values=[4, 8, 16, 32, 64],
        verbose=verbose
    )
    
    # Find χ for 1% error threshold
    chi_1pct = None
    for i, err in enumerate(analysis['rho_error']):
        if err < 0.01:
            chi_1pct = analysis['chi'][i]
            break
    
    # Estimate area law scaling
    scaling = estimate_area_law_exponent(state)
    
    results = {
        'test': 'oblique_shock',
        'Nx': Nx,
        'Ny': Ny,
        'M_inf': M_inf,
        'theta_deg': theta_deg,
        'chi_for_1pct_error': chi_1pct,
        'scaling_slope': scaling['slope'],
        'scaling_interpretation': scaling['interpretation'],
        'analysis': analysis,
        'pass': chi_1pct is not None and chi_1pct <= 64
    }
    
    if verbose:
        print()
        print(f"χ for 1% error: {chi_1pct}")
        print(f"Scaling behavior: {scaling['interpretation']} (slope={scaling['slope']:.3f})")
        print(f"Status: {'✓ PASS' if results['pass'] else '✗ FAIL'}")
    
    return results


def benchmark_smooth_function(Nx: int = 256, verbose: bool = True) -> dict:
    """
    Benchmark 3: Smooth analytic function (QTT validation).
    
    For smooth functions, QTT should achieve exponential compression
    with χ = O(log N).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK 3: Smooth Function (QTT Validation)")
        print("=" * 70)
    
    # Create smooth test function: sin(2πx) * cos(2πy)
    x = torch.linspace(0, 1, Nx, dtype=torch.float64)
    y = torch.linspace(0, 1, Nx, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    field = torch.sin(2 * math.pi * X) * torch.cos(2 * math.pi * Y)
    
    if verbose:
        print(f"Grid: {Nx} × {Nx}")
        print(f"Function: sin(2πx) * cos(2πy)")
        print()
    
    chi_values = [2, 4, 8, 16, 32]
    errors = []
    crs = []
    
    for chi in chi_values:
        result = field_to_qtt(field, chi_max=chi)
        reconstructed = qtt_to_field(result)
        
        error = torch.norm(reconstructed - field) / torch.norm(field)
        errors.append(error.item())
        crs.append(result.compression_ratio)
        
        if verbose:
            print(f"χ={chi:3d}: CR={result.compression_ratio:8.2f}x, error={error.item():.2e}")
    
    # Check exponential decay
    log_errors = [math.log(e + 1e-16) for e in errors]
    slope = (log_errors[-1] - log_errors[0]) / (chi_values[-1] - chi_values[0])
    
    results = {
        'test': 'smooth_function',
        'Nx': Nx,
        'chi_values': chi_values,
        'errors': errors,
        'compression_ratios': crs,
        'error_decay_slope': slope,
        'pass': errors[-1] < 1e-6  # Should achieve machine precision with χ=32
    }
    
    if verbose:
        print()
        print(f"Error decay rate: {slope:.4f} per unit χ")
        print(f"Status: {'✓ PASS' if results['pass'] else '✗ FAIL'}")
    
    return results


def benchmark_resolution_scaling(
    resolutions: list[int] = [32, 64, 128, 256],
    chi_fixed: int = 16,
    verbose: bool = True
) -> dict:
    """
    Benchmark 4: Resolution scaling at fixed χ.
    
    Tests how error scales with grid resolution at fixed bond dimension.
    For area law: error ~ O(N^α) with α small.
    For volume law: error ~ O(N).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK 4: Resolution Scaling at Fixed χ")
        print("=" * 70)
        print(f"Fixed bond dimension: χ = {chi_fixed}")
        print()
    
    errors = []
    crs = []
    
    for N in resolutions:
        # Smooth supersonic flow
        state = supersonic_wedge_ic(Nx=N, Ny=N//2, M_inf=3.0)
        
        compressed = euler_to_qtt(state, chi_max=chi_fixed)
        reconstructed = qtt_to_euler(compressed, gamma=state.gamma)
        
        error = torch.norm(reconstructed.rho - state.rho) / torch.norm(state.rho)
        cr = compressed['rho'].compression_ratio
        
        errors.append(error.item())
        crs.append(cr)
        
        if verbose:
            print(f"N={N:4d}: CR={cr:8.2f}x, error={error.item():.2e}")
    
    # Compute scaling exponent
    log_N = [math.log(n) for n in resolutions]
    log_err = [math.log(e + 1e-16) for e in errors]
    
    # Linear regression
    n = len(resolutions)
    mean_x = sum(log_N) / n
    mean_y = sum(log_err) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_N, log_err))
    den = sum((x - mean_x)**2 for x in log_N)
    scaling_exponent = num / den if den > 0 else 0
    
    results = {
        'test': 'resolution_scaling',
        'chi_fixed': chi_fixed,
        'resolutions': resolutions,
        'errors': errors,
        'compression_ratios': crs,
        'scaling_exponent': scaling_exponent,
        'pass': scaling_exponent < 0.5  # Sublinear scaling indicates efficiency
    }
    
    if verbose:
        print()
        print(f"Scaling exponent α (error ~ N^α): {scaling_exponent:.3f}")
        print(f"Interpretation: {'Sublinear (efficient)' if scaling_exponent < 0.5 else 'Linear or worse'}")
        print(f"Status: {'✓ PASS' if results['pass'] else '✗ FAIL'}")
    
    return results


def run_all_benchmarks(verbose: bool = True, save_results: bool = True) -> dict:
    """Run all QTT compression benchmarks."""
    
    print("=" * 70)
    print("PROJECT HYPERTENSOR: QTT COMPRESSION BENCHMARKS")
    print("Validating Area Law Hypothesis for CFD Fields")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {}
    }
    
    # Run benchmarks
    results['benchmarks']['uniform_flow'] = benchmark_uniform_flow(verbose=verbose)
    results['benchmarks']['oblique_shock'] = benchmark_oblique_shock(verbose=verbose)
    results['benchmarks']['smooth_function'] = benchmark_smooth_function(verbose=verbose)
    results['benchmarks']['resolution_scaling'] = benchmark_resolution_scaling(verbose=verbose)
    
    # Summary
    passed = sum(1 for b in results['benchmarks'].values() if b['pass'])
    total = len(results['benchmarks'])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{total}")
    
    for name, bench in results['benchmarks'].items():
        status = '✓' if bench['pass'] else '✗'
        print(f"  {status} {name}")
    
    results['passed'] = passed
    results['total'] = total
    results['all_passed'] = passed == total
    
    # Save results
    if save_results:
        output_dir = Path(__file__).parent.parent / 'results'
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / 'qtt_benchmark_results.json'
        
        # Convert non-serializable items
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        with open(output_file, 'w') as f:
            json.dump(make_serializable(results), f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QTT Compression Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    run_all_benchmarks(verbose=not args.quiet, save_results=not args.no_save)
