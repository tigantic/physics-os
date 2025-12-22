"""
Proof 21.1: WENO 5th-Order Convergence Verification.

This proof verifies that the WENO5-JS and WENO5-Z schemes achieve
5th-order convergence on smooth test functions.

Constitution Compliance: Article I.1 (Proof Requirements)
"""

import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensornet.cfd.weno import (
    weno5_js, weno5_z, teno5, 
    ReconstructionSide, WENOConfig,
    smoothness_indicators, convergence_order
)


def smooth_test_function(x: torch.Tensor) -> torch.Tensor:
    """Smooth test function: sin(2πx)."""
    return torch.sin(2 * np.pi * x)


def smooth_test_derivative(x: torch.Tensor) -> torch.Tensor:
    """Derivative of test function for error analysis."""
    return 2 * np.pi * torch.cos(2 * np.pi * x)


def polynomial_test_function(x: torch.Tensor) -> torch.Tensor:
    """Polynomial test: x^5 - x^3 + x (exact for 5th order)."""
    return x**5 - x**3 + x


def test_weno_convergence(
    weno_fn,
    test_fn,
    N_values: list,
    domain: tuple = (0.0, 1.0),
    name: str = "WENO"
) -> dict:
    """
    Test convergence of a WENO scheme on smooth cell averages.
    
    WENO5 finite-volume formulas reconstruct interface values from cell averages.
    For 5th-order accuracy, we must provide true cell averages, not point values.
    
    For f(x) = sin(2πx), cell average over [x-dx/2, x+dx/2] is:
    ū = (1/dx) ∫ sin(2πx) dx = (cos(2π(x-dx/2)) - cos(2π(x+dx/2))) / (2πdx)
    
    Returns:
        Dictionary with errors and estimated order
    """
    errors_L2 = []
    errors_Linf = []
    dx_values = []
    
    for N in N_values:
        # Cell-centered grid with N cells in [0, 1]
        dx = (domain[1] - domain[0]) / N
        x_left = torch.linspace(domain[0], domain[1] - dx, N, dtype=torch.float64)
        x_right = x_left + dx
        x_center = x_left + dx/2
        dx_values.append(dx)
        
        # True cell averages via exact integration of sin(2πx)
        # ∫sin(2πx)dx = -cos(2πx)/(2π)
        # Average = [-cos(2π x_right) + cos(2π x_left)] / (2π dx)
        u_avg = (torch.cos(2*np.pi*x_left) - torch.cos(2*np.pi*x_right)) / (2*np.pi*dx)
        
        # WENO reconstruction gives u_{i+1/2}^- for i in [2, N-3]
        u_recon = weno_fn(u_avg, ReconstructionSide.LEFT)
        
        # Interface positions: output[j] corresponds to interface at x_right[j+2]
        n_out = len(u_recon)
        x_interface = x_right[2:2+n_out]
        u_exact = test_fn(x_interface)
        
        # Compute errors
        err_L2 = torch.sqrt(torch.mean((u_recon - u_exact)**2)).item()
        err_Linf = torch.max(torch.abs(u_recon - u_exact)).item()
        
        errors_L2.append(err_L2)
        errors_Linf.append(err_Linf)
    
    # Compute convergence order
    order_L2 = convergence_order(errors_L2, dx_values)
    order_Linf = convergence_order(errors_Linf, dx_values)
    
    return {
        'name': name,
        'N_values': N_values,
        'dx_values': dx_values,
        'errors_L2': errors_L2,
        'errors_Linf': errors_Linf,
        'order_L2': order_L2,
        'order_Linf': order_Linf,
    }


def test_smoothness_indicators():
    """Verify smoothness indicators behave correctly."""
    # Smooth function: all beta should be similar
    x = torch.linspace(0, 1, 100, dtype=torch.float64)
    u_smooth = torch.sin(2 * np.pi * x)
    
    beta0, beta1, beta2 = smoothness_indicators(u_smooth)
    
    # In smooth regions, beta values should be comparable
    beta_ratio = beta0.max() / (beta0.min() + 1e-20)
    
    smooth_test = beta_ratio < 1e6  # All similar magnitude
    
    # Function with jump: beta should vary
    u_jump = torch.where(x < 0.5, torch.ones_like(x), 2.0 * torch.ones_like(x))
    
    beta0_j, beta1_j, beta2_j = smoothness_indicators(u_jump)
    
    # Near jump, beta should spike
    jump_detected = beta1_j.max() > 1e-6
    
    return {
        'smooth_beta_ratio': beta_ratio.item(),
        'smooth_test_passed': bool(smooth_test),
        'jump_detected': bool(jump_detected),
        'jump_beta_max': beta1_j.max().item(),
    }


def main():
    """Run all WENO order proofs."""
    print("=" * 60)
    print("PROOF 21.1: WENO 5th-Order Convergence")
    print("=" * 60)
    
    results = {
        'proof_name': 'proof_21_weno_order',
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    N_values = [32, 64, 128, 256, 512]
    
    # Test 1: WENO5-JS convergence on sin(2πx)
    print("\n[Test 1] WENO5-JS on sin(2πx)...")
    result_js = test_weno_convergence(
        weno5_js, smooth_test_function, N_values, name="WENO5-JS"
    )
    print(f"  L2 Order: {result_js['order_L2']:.3f}")
    print(f"  L∞ Order: {result_js['order_Linf']:.3f}")
    
    passed_js = result_js['order_L2'] > 4.5  # Should be ~5
    result_js['passed'] = passed_js
    results['tests'].append(result_js)
    print(f"  Status: {'✓ PASSED' if passed_js else '✗ FAILED'}")
    
    # Test 2: WENO5-Z convergence on sin(2πx)
    print("\n[Test 2] WENO5-Z on sin(2πx)...")
    result_z = test_weno_convergence(
        weno5_z, smooth_test_function, N_values, name="WENO5-Z"
    )
    print(f"  L2 Order: {result_z['order_L2']:.3f}")
    print(f"  L∞ Order: {result_z['order_Linf']:.3f}")
    
    passed_z = result_z['order_L2'] > 4.5
    result_z['passed'] = passed_z
    results['tests'].append(result_z)
    print(f"  Status: {'✓ PASSED' if passed_z else '✗ FAILED'}")
    
    # Test 3: TENO5 convergence
    print("\n[Test 3] TENO5 on sin(2πx)...")
    result_teno = test_weno_convergence(
        teno5, smooth_test_function, N_values, name="TENO5"
    )
    print(f"  L2 Order: {result_teno['order_L2']:.3f}")
    print(f"  L∞ Order: {result_teno['order_Linf']:.3f}")
    
    passed_teno = result_teno['order_L2'] > 4.0  # Slightly more relaxed for TENO
    result_teno['passed'] = passed_teno
    results['tests'].append(result_teno)
    print(f"  Status: {'✓ PASSED' if passed_teno else '✗ FAILED'}")
    
    # Test 4: Smoothness indicators
    print("\n[Test 4] Smoothness indicator behavior...")
    beta_results = test_smoothness_indicators()
    print(f"  Smooth β ratio: {beta_results['smooth_beta_ratio']:.2e}")
    print(f"  Jump detected: {beta_results['jump_detected']}")
    
    passed_beta = beta_results['smooth_test_passed'] and beta_results['jump_detected']
    beta_results['passed'] = passed_beta
    results['tests'].append({
        'name': 'smoothness_indicators',
        **beta_results
    })
    print(f"  Status: {'✓ PASSED' if passed_beta else '✗ FAILED'}")
    
    # Test 5: WENO-Z vs WENO-JS accuracy comparison
    print("\n[Test 5] WENO-Z improvement over WENO-JS...")
    # WENO-Z should be more accurate at same resolution
    N = 128
    x = torch.linspace(0, 1, N, dtype=torch.float64)
    u = smooth_test_function(x)
    x_int = 0.5 * (x[2:-2] + x[3:-1])
    u_exact = smooth_test_function(x_int)
    
    u_js = weno5_js(u, ReconstructionSide.LEFT)
    u_z = weno5_z(u, ReconstructionSide.LEFT)
    
    n = min(len(u_js), len(u_z), len(u_exact))
    err_js = torch.sqrt(torch.mean((u_js[:n] - u_exact[:n])**2)).item()
    err_z = torch.sqrt(torch.mean((u_z[:n] - u_exact[:n])**2)).item()
    
    improvement = err_js / (err_z + 1e-20)
    passed_improvement = err_z <= err_js * 1.1  # Z should be at least as good
    
    results['tests'].append({
        'name': 'weno_z_improvement',
        'error_js': err_js,
        'error_z': err_z,
        'improvement_factor': improvement,
        'passed': passed_improvement
    })
    print(f"  WENO-JS error: {err_js:.2e}")
    print(f"  WENO-Z error: {err_z:.2e}")
    print(f"  Improvement: {improvement:.2f}x")
    print(f"  Status: {'✓ PASSED' if passed_improvement else '✗ FAILED'}")
    
    # Summary
    print("\n" + "=" * 60)
    all_passed = all(t.get('passed', False) for t in results['tests'])
    results['all_passed'] = all_passed
    results['summary'] = {
        'total_tests': len(results['tests']),
        'passed': sum(1 for t in results['tests'] if t.get('passed', False)),
        'failed': sum(1 for t in results['tests'] if not t.get('passed', False)),
    }
    
    print(f"PROOF RESULT: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    print(f"Tests: {results['summary']['passed']}/{results['summary']['total_tests']} passed")
    print("=" * 60)
    
    # Save results
    output_path = Path(__file__).parent / 'proof_21_weno_order_result.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    main()
