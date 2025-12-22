"""
Phase 1b Proof: 3D Incompressible Navier-Stokes
================================================

Comprehensive validation of 3D NS solver implementation.

Gate Criteria:
    1. 3D spectral operators: gradient, divergence, Laplacian < 10⁻¹⁰
    2. 3D Poisson self-consistency: |φ - solve(∇²φ)| < 10⁻¹⁰
    3. 3D projection: max|∇·u| < 10⁻¹⁰ after projection
    4. 3D Taylor-Green: decay error < 5%, divergence < 10⁻⁶

Constitution Compliance: Article IV.1 (Verification), Phase 1b
Tag: [PHASE-1B] [PROOF]
"""

import torch
import math
import json
from datetime import datetime
from typing import Dict, Any


def proof_3d_poisson_solver() -> Dict[str, Any]:
    """
    Proof 1b.1: 3D FFT Poisson solver correctness.
    
    Verifies:
        - Laplacian is exact for trigonometric functions
        - Poisson self-consistency: solve(∇²φ) ≈ φ
    """
    from tensornet.cfd.ns_3d import (
        laplacian_spectral_3d,
        poisson_solve_fft_3d,
    )
    
    N = 32
    dx = dy = dz = 2 * math.pi / N
    
    x = torch.linspace(0, 2*math.pi - dx, N, dtype=torch.float64)
    y = torch.linspace(0, 2*math.pi - dy, N, dtype=torch.float64)
    z = torch.linspace(0, 2*math.pi - dz, N, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Test function with k² = 3
    phi = torch.sin(X) * torch.sin(Y) * torch.sin(Z)
    
    # Exact Laplacian: ∇²φ = -3φ
    lap = laplacian_spectral_3d(phi, dx, dy, dz)
    exact_lap = -3 * phi
    laplacian_error = (lap - exact_lap).abs().max().item()
    
    # Poisson self-consistency: solve(∇²φ) ≈ φ
    rhs = -3 * phi  # ∇²φ
    phi_solved = poisson_solve_fft_3d(rhs, dx, dy, dz)
    
    # Remove mean for comparison (Poisson has arbitrary constant)
    phi_zm = phi - phi.mean()
    phi_solved_zm = phi_solved - phi_solved.mean()
    poisson_error = (phi_zm - phi_solved_zm).abs().max().item()
    
    passed = laplacian_error < 1e-10 and poisson_error < 1e-10
    
    return {
        'name': 'proof_3d_poisson_solver',
        'description': '3D FFT Poisson solver exactness',
        'laplacian_error': laplacian_error,
        'poisson_self_consistency_error': poisson_error,
        'threshold': 1e-10,
        'passed': passed,
    }


def proof_3d_spectral_operators() -> Dict[str, Any]:
    """
    Proof 1b.2: 3D spectral gradient and divergence.
    
    Verifies:
        - Gradient is exact for sin/cos
        - Divergence of div-free field is zero
    """
    from tensornet.cfd.ns_3d import (
        compute_gradient_3d,
        compute_divergence_3d,
    )
    
    N = 32
    dx = dy = dz = 2 * math.pi / N
    
    x = torch.linspace(0, 2*math.pi - dx, N, dtype=torch.float64)
    y = torch.linspace(0, 2*math.pi - dy, N, dtype=torch.float64)
    z = torch.linspace(0, 2*math.pi - dz, N, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Test gradient
    phi = torch.sin(X) * torch.sin(Y) * torch.sin(Z)
    dphi_dx, dphi_dy, dphi_dz = compute_gradient_3d(phi, dx, dy, dz, method='spectral')
    
    exact_dx = torch.cos(X) * torch.sin(Y) * torch.sin(Z)
    exact_dy = torch.sin(X) * torch.cos(Y) * torch.sin(Z)
    exact_dz = torch.sin(X) * torch.sin(Y) * torch.cos(Z)
    
    grad_x_error = (dphi_dx - exact_dx).abs().max().item()
    grad_y_error = (dphi_dy - exact_dy).abs().max().item()
    grad_z_error = (dphi_dz - exact_dz).abs().max().item()
    grad_error = max(grad_x_error, grad_y_error, grad_z_error)
    
    # Test divergence of div-free field (Taylor-Green IC is div-free)
    u = torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    v = -torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    w = torch.zeros_like(u)
    
    div = compute_divergence_3d(u, v, w, dx, dy, dz, method='spectral')
    div_error = div.abs().max().item()
    
    passed = grad_error < 1e-10 and div_error < 1e-10
    
    return {
        'name': 'proof_3d_spectral_operators',
        'description': '3D spectral gradient and divergence exactness',
        'gradient_error': grad_error,
        'divergence_of_divfree_error': div_error,
        'threshold': 1e-10,
        'passed': passed,
    }


def proof_3d_projection() -> Dict[str, Any]:
    """
    Proof 1b.3: 3D velocity projection to divergence-free.
    
    Verifies:
        - Projection enforces ∇·u = 0 to machine precision
    """
    from tensornet.cfd.ns_3d import (
        compute_divergence_3d,
        project_velocity_3d,
    )
    
    N = 32
    dx = dy = dz = 2 * math.pi / N
    
    x = torch.linspace(0, 2*math.pi - dx, N, dtype=torch.float64)
    y = torch.linspace(0, 2*math.pi - dy, N, dtype=torch.float64)
    z = torch.linspace(0, 2*math.pi - dz, N, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Create non-divergence-free field
    u_star = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v_star = torch.zeros_like(u_star)
    w_star = torch.zeros_like(u_star)
    
    div_before = compute_divergence_3d(u_star, v_star, w_star, dx, dy, dz, method='spectral')
    div_before_max = div_before.abs().max().item()
    
    # Project
    result = project_velocity_3d(u_star, v_star, w_star, dx, dy, dz, dt=1.0)
    
    passed = result.divergence_after < 1e-10
    
    return {
        'name': 'proof_3d_projection',
        'description': '3D velocity projection enforces incompressibility',
        'divergence_before': div_before_max,
        'divergence_after': result.divergence_after,
        'threshold': 1e-10,
        'passed': passed,
    }


def proof_3d_taylor_green() -> Dict[str, Any]:
    """
    Proof 1b.4: 3D Taylor-Green vortex benchmark.
    
    Gate criteria:
        - Energy decay rate error < 5%
        - max|∇·u| < 10⁻⁶ throughout simulation
    
    The 3D Taylor-Green with modes (1,1,1) has |k|² = 3,
    so kinetic energy decays as exp(-6νt).
    """
    from tensornet.cfd.ns_3d import (
        NS3DSolver,
        compute_divergence_3d,
        taylor_green_3d_exact_energy,
    )
    
    # Parameters: high viscosity for Stokes regime
    N = 32
    nu = 1.0
    A = 1.0
    t_final = 0.1
    
    solver = NS3DSolver(
        Nx=N, Ny=N, Nz=N,
        Lx=2*math.pi, Ly=2*math.pi, Lz=2*math.pi,
        nu=nu,
        dtype=torch.float64,
    )
    
    state = solver.create_taylor_green_3d(A)
    
    # Initial KE
    dV = solver.dx * solver.dy * solver.dz
    ke_init = 0.5 * (state.u**2 + state.v**2 + state.w**2).sum().item() * dV
    
    # Solve
    result = solver.solve(state, t_final, cfl_target=0.2, verbose=False)
    
    # Final diagnostics
    final_diag = result.diagnostics_history[-1]
    ke_final = final_diag.kinetic_energy
    
    # Decay rate error
    decay_rate_numerical = -math.log(ke_final / ke_init) / t_final
    decay_rate_exact = 6 * nu  # Correct for 3D TG with |k|²=3
    decay_error_pct = abs(decay_rate_numerical - decay_rate_exact) / decay_rate_exact * 100
    
    # Max divergence throughout
    max_div = max(d.max_divergence for d in result.diagnostics_history)
    
    # Expected KE
    ke_exact = taylor_green_3d_exact_energy(t_final, nu, ke_init)
    ke_error_pct = abs(ke_final - ke_exact) / ke_exact * 100
    
    decay_gate = decay_error_pct < 5.0
    div_gate = max_div < 1e-6
    passed = decay_gate and div_gate
    
    return {
        'name': 'proof_3d_taylor_green',
        'description': '3D Taylor-Green vortex energy decay and incompressibility',
        'ke_initial': ke_init,
        'ke_final': ke_final,
        'ke_exact': ke_exact,
        'ke_error_pct': ke_error_pct,
        'decay_rate_numerical': decay_rate_numerical,
        'decay_rate_exact': decay_rate_exact,
        'decay_error_pct': decay_error_pct,
        'max_divergence': max_div,
        'decay_gate_threshold': 5.0,
        'div_gate_threshold': 1e-6,
        'decay_gate': decay_gate,
        'div_gate': div_gate,
        'passed': passed,
    }


def run_all_proofs() -> Dict[str, Any]:
    """Run all Phase 1b proofs."""
    print("\n" + "=" * 70)
    print("PHASE 1B PROOFS: 3D INCOMPRESSIBLE NAVIER-STOKES")
    print("=" * 70)
    
    proofs = [
        proof_3d_poisson_solver,
        proof_3d_spectral_operators,
        proof_3d_projection,
        proof_3d_taylor_green,
    ]
    
    results = []
    all_passed = True
    
    for proof_fn in proofs:
        result = proof_fn()
        results.append(result)
        status = "PASS" if result['passed'] else "FAIL"
        all_passed = all_passed and result['passed']
        
        print(f"\n[{status}] {result['name']}")
        print(f"  Description: {result['description']}")
        
        # Print key metrics
        for k, v in result.items():
            if k not in ['name', 'description', 'passed']:
                if isinstance(v, float):
                    print(f"  {k}: {v:.2e}" if v < 0.01 else f"  {k}: {v:.4f}")
                elif isinstance(v, bool):
                    print(f"  {k}: {'✓' if v else '✗'}")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {sum(1 for r in results if r['passed'])}/{len(results)} proofs passed")
    print("=" * 70)
    
    if all_passed:
        print("[SUCCESS] ALL PHASE 1B PROOFS PASSED")
    else:
        print("[FAILURE] Some proofs failed")
    
    # Save results
    output = {
        'phase': '1b',
        'timestamp': datetime.now().isoformat(),
        'proofs': results,
        'summary': {
            'total': len(results),
            'passed': sum(1 for r in results if r['passed']),
            'failed': sum(1 for r in results if not r['passed']),
            'all_passed': all_passed,
        }
    }
    
    return output


if __name__ == '__main__':
    output = run_all_proofs()
    
    # Save to JSON
    with open('proofs/proof_phase_1b_result.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to proofs/proof_phase_1b_result.json")
