"""
Formal Proof: NS Projection Method [DECISION-005]
==================================================

Proves correctness of the Chorin-Temam projection step for 
incompressible Navier-Stokes simulation.

Key Properties Verified:
------------------------
1. POISSON CONVERGENCE: Discrete Laplacian has O(dx^2) truncation error
2. SELF-CONSISTENCY: solve(A, A@x) = x to machine precision
3. PROJECTION GATE: Divergence after projection < 10⁻⁶
4. SPECTRAL CONSISTENCY: nabla·nablaφ = nabla^2φ exactly in Fourier space

Constitution Compliance: Article IV.1 (Verification)
Tag: [PHASE-1A] [DECISION-005] [RISK-R8]
"""

import torch
import math
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensornet.cfd.tt_poisson import (
    test_poisson_solver,
    test_projection,
    project_velocity_2d,
    compute_divergence_2d,
    compute_gradient_2d,
    laplacian_spectral_2d,
    poisson_solve_fft_2d,
)


def prove_spectral_consistency():
    """
    Prove that spectral gradient and Laplacian are consistent:
        nabla·(nablaφ) = nabla^2φ exactly in Fourier space.
    
    This is the foundation of machine-precision incompressibility.
    """
    print("\n" + "=" * 60)
    print("PROOF: Spectral Operator Consistency")
    print("=" * 60)
    
    N = 64
    dx = dy = 1.0 / N
    
    x = torch.linspace(0, 1 - dx, N, dtype=torch.float64)
    y = torch.linspace(0, 1 - dy, N, dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Test on multiple wavenumbers
    max_errors = []
    
    for k in [1, 2, 4, 8]:
        # Test function: φ = sin(2πkx)sin(2πky)
        phi = torch.sin(2 * math.pi * k * X) * torch.sin(2 * math.pi * k * Y)
        
        # Compute nablaφ (spectral)
        dphi_dx, dphi_dy = compute_gradient_2d(phi, dx, dy, method='spectral')
        
        # Compute nabla·(nablaφ) (spectral)
        div_grad = compute_divergence_2d(dphi_dx, dphi_dy, dx, dy, method='spectral')
        
        # Compute nabla^2φ directly (spectral)
        lap = laplacian_spectral_2d(phi, dx, dy)
        
        # They should be identical
        error = torch.abs(div_grad - lap).max().item()
        max_errors.append(error)
        
        print(f"  k={k}: max|nabla·nablaφ - nabla^2φ| = {error:.2e}")
    
    passed = all(e < 1e-10 for e in max_errors)
    print(f"\nSpectral consistency: {'PASS' if passed else 'FAIL'}")
    
    return {
        'proof': 'spectral_consistency',
        'max_errors': max_errors,
        'passed': passed,
        'explanation': 'Spectral nabla·nablaφ = nabla^2φ to machine precision',
    }


def prove_poisson_inversion():
    """
    Prove that FFT Poisson solve inverts spectral Laplacian exactly:
        nabla^2(solve(f)) = f to machine precision.
    """
    print("\n" + "=" * 60)
    print("PROOF: Poisson Inversion")
    print("=" * 60)
    
    N = 64
    dx = dy = 1.0 / N
    
    x = torch.linspace(0, 1 - dx, N, dtype=torch.float64)
    y = torch.linspace(0, 1 - dy, N, dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Test RHS (must have zero mean for solvability)
    f = torch.sin(4 * math.pi * X) * torch.cos(2 * math.pi * Y)
    f = f - f.mean()  # Ensure zero mean
    
    # Solve Poisson
    phi = poisson_solve_fft_2d(f, dx, dy)
    
    # Apply Laplacian to solution
    lap_phi = laplacian_spectral_2d(phi, dx, dy)
    
    # Should recover f
    error = torch.abs(lap_phi - f).max().item()
    
    print(f"  max|nabla^2φ - f| = {error:.2e}")
    
    passed = error < 1e-10
    print(f"\nPoisson inversion: {'PASS' if passed else 'FAIL'}")
    
    return {
        'proof': 'poisson_inversion',
        'max_error': error,
        'passed': passed,
        'explanation': 'FFT Poisson inverts spectral Laplacian exactly',
    }


def prove_helmholtz_decomposition():
    """
    Prove the Helmholtz decomposition:
        Any vector field u = u_div + u_curl where:
        - u_div is irrotational (curl-free): u_div = nablaφ
        - u_curl is solenoidal (div-free): nabla·u_curl = 0
    
    The projection step computes: u_curl = u - nablaφ where φ solves nabla^2φ = nabla·u
    
    Note: For spectral methods, test data MUST be periodic.
    """
    print("\n" + "=" * 60)
    print("PROOF: Helmholtz Decomposition (Projection)")
    print("=" * 60)
    
    N = 64
    dx = dy = 1.0 / N
    
    x = torch.linspace(0, 1 - dx, N, dtype=torch.float64)
    y = torch.linspace(0, 1 - dy, N, dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create arbitrary non-div-free velocity field
    # CRITICAL: Both u and v must be periodic in x and y
    # Use multiples of 2π to ensure periodicity on [0,1)
    u = torch.sin(2 * math.pi * X) * torch.cos(4 * math.pi * Y)
    v = torch.cos(4 * math.pi * X) * torch.sin(2 * math.pi * Y)
    
    # Use the validated projection function
    result = project_velocity_2d(u, v, dx, dy, dt=1.0, bc='periodic', method='spectral')
    
    print(f"  Divergence before projection: {result.divergence_before:.4e}")
    print(f"  Divergence after projection:  {result.divergence_after:.4e}")
    print(f"  Reduction factor: {result.divergence_before / max(result.divergence_after, 1e-16):.2e}")
    
    # Gate criterion: div_after < 1e-10 (machine precision for spectral)
    passed = result.divergence_after < 1e-10
    print(f"\nHelmholtz decomposition: {'PASS' if passed else 'FAIL'}")
    
    return {
        'proof': 'helmholtz_decomposition',
        'divergence_before': result.divergence_before,
        'divergence_after': result.divergence_after,
        'passed': passed,
        'explanation': 'Projection achieves machine-precision div-free field',
    }


def run_all_proofs():
    """Run all proofs and save results."""
    print("=" * 60)
    print("NS PROJECTION PROOFS [PHASE-1A] [DECISION-005]")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': '1a',
        'decision': 'DECISION-005',
        'proofs': [],
    }
    
    # Run Poisson solver test
    print("\nRunning Poisson solver tests...")
    poisson_result = test_poisson_solver()
    results['proofs'].append({
        'name': 'poisson_convergence',
        'passed': poisson_result['passed'],
        'details': poisson_result,
    })
    
    # Run projection test
    print("\nRunning projection tests...")
    proj_result = test_projection()
    results['proofs'].append({
        'name': 'projection_gate',
        'passed': proj_result['gate_passed'],
        'details': proj_result,
    })
    
    # Run additional proofs
    results['proofs'].append(prove_spectral_consistency())
    results['proofs'].append(prove_poisson_inversion())
    results['proofs'].append(prove_helmholtz_decomposition())
    
    # Summary
    all_passed = all(p['passed'] for p in results['proofs'])
    results['all_passed'] = all_passed
    
    print("\n" + "=" * 60)
    print("PROOF SUMMARY")
    print("=" * 60)
    for p in results['proofs']:
        name = p.get('name', p.get('proof', 'unknown'))
        status = 'PASS' if p['passed'] else 'FAIL'
        print(f"  {name}: {status}")
    
    print("-" * 60)
    print(f"ALL PROOFS: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)
    
    # Save results
    output_path = Path(__file__).parent / 'proof_ns_projection_result.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    results = run_all_proofs()
    sys.exit(0 if results['all_passed'] else 1)
