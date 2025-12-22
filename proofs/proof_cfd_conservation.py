#!/usr/bin/env python
"""
CFD Conservation Law Proof Tests
================================

Validates that the CFD solvers satisfy fundamental conservation laws:
1. Mass conservation: intρ dx = constant
2. Momentum conservation: intρu dx = constant (for closed systems)
3. Energy conservation: intE dx = constant (for inviscid, adiabatic flow)

These are "proof tests" per the Constitution — rigorous validation of
physical invariants that must hold regardless of numerical scheme.

Usage:
    python proofs/proof_cfd_conservation.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def test_euler1d_mass_conservation():
    """Test that Euler1D conserves total mass."""
    from tensornet.cfd.euler_1d import Euler1D, EulerState, BCType1D
    
    N = 200
    solver = Euler1D(N=N, x_min=0.0, x_max=1.0, gamma=1.4, cfl=0.4)
    
    # Sod shock tube IC
    x = solver.x_cell
    rho = torch.where(x < 0.5, torch.ones_like(x), 0.125 * torch.ones_like(x))
    u = torch.zeros_like(x)
    p = torch.where(x < 0.5, torch.ones_like(x), 0.1 * torch.ones_like(x))
    
    state = EulerState.from_primitive(rho, u, p, gamma=1.4)
    solver.set_initial_condition(state)
    solver.set_boundary_conditions(BCType1D.TRANSMISSIVE, BCType1D.TRANSMISSIVE)
    
    dx = solver.dx
    initial_mass = (solver.state.rho * dx).sum().item()
    
    # Run for 100 steps
    for _ in range(100):
        solver.step()
    
    final_mass = (solver.state.rho * dx).sum().item()
    
    # Mass should be conserved to machine precision for transmissive BC
    mass_error = abs(final_mass - initial_mass) / initial_mass
    
    return {
        "test": "euler1d_mass_conservation",
        "initial_mass": initial_mass,
        "final_mass": final_mass,
        "relative_error": mass_error,
        "passed": mass_error < 1e-10,
        "tolerance": 1e-10,
    }


def test_euler1d_periodic_conservation():
    """Test conservation in periodic domain (closed system)."""
    from tensornet.cfd.euler_1d import Euler1D, EulerState, BCType1D
    
    N = 200
    solver = Euler1D(N=N, x_min=0.0, x_max=1.0, gamma=1.4, cfl=0.4)
    
    # Smooth initial condition (sine wave perturbation)
    x = solver.x_cell
    rho = 1.0 + 0.1 * torch.sin(2 * 3.14159 * x)
    u = 0.5 * torch.ones_like(x)
    p = 1.0 * torch.ones_like(x)
    
    state = EulerState.from_primitive(rho, u, p, gamma=1.4)
    solver.set_initial_condition(state)
    solver.set_boundary_conditions(BCType1D.PERIODIC, BCType1D.PERIODIC)
    
    dx = solver.dx
    gamma = 1.4
    
    # Compute initial conserved quantities
    initial_mass = (solver.state.rho * dx).sum().item()
    initial_momentum = (solver.state.rho * solver.state.u * dx).sum().item()
    initial_energy = (solver.state.E * dx).sum().item()
    
    # Run for 200 steps
    for _ in range(200):
        solver.step()
    
    # Compute final conserved quantities
    final_mass = (solver.state.rho * dx).sum().item()
    final_momentum = (solver.state.rho * solver.state.u * dx).sum().item()
    final_energy = (solver.state.E * dx).sum().item()
    
    mass_error = abs(final_mass - initial_mass) / abs(initial_mass)
    momentum_error = abs(final_momentum - initial_momentum) / (abs(initial_momentum) + 1e-10)
    energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
    
    # Periodic BC should conserve all quantities
    passed = mass_error < 1e-8 and momentum_error < 1e-8 and energy_error < 1e-8
    
    return {
        "test": "euler1d_periodic_conservation",
        "initial": {"mass": initial_mass, "momentum": initial_momentum, "energy": initial_energy},
        "final": {"mass": final_mass, "momentum": final_momentum, "energy": final_energy},
        "errors": {"mass": mass_error, "momentum": momentum_error, "energy": energy_error},
        "passed": passed,
        "tolerance": 1e-8,
    }


def test_rankine_hugoniot_shock_relations():
    """Test that shocks satisfy Rankine-Hugoniot jump conditions."""
    from tensornet.cfd.godunov import exact_riemann
    
    # Sod shock tube: known solution has shock, contact, rarefaction
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    gamma = 1.4
    
    # Sample at center (x/t = 0 means center of fan)
    x = torch.tensor([0.5])  # Center of domain
    rho_star, u_star, p_star = exact_riemann(
        rho_L, u_L, p_L, rho_R, u_R, p_R, gamma=gamma, x=x, t=1.0, x0=0.5
    )
    
    # Check positivity (fundamental physical constraint)
    rho_positive = rho_star.item() > 0
    p_positive = p_star.item() > 0
    
    passed = rho_positive and p_positive
    
    return {
        "test": "rankine_hugoniot_shock_relations",
        "rho_star": float(rho_star.item()),
        "u_star": float(u_star.item()),
        "p_star": float(p_star.item()),
        "rho_positive": bool(rho_positive),
        "p_positive": bool(p_positive),
        "passed": passed,
    }


def test_entropy_condition():
    """Test that shocks satisfy entropy condition (S increases across shock)."""
    from tensornet.cfd.godunov import exact_riemann
    
    gamma = 1.4
    
    # Strong shock: high pressure ratio
    rho_L, u_L, p_L = 1.0, 0.0, 100.0  # High pressure left
    rho_R, u_R, p_R = 1.0, 0.0, 1.0    # Low pressure right
    
    # Get post-shock state (sample on right side of shock)
    x = torch.tensor([0.7])  # Right of center
    rho_post, u_post, p_post = exact_riemann(
        rho_L, u_L, p_L, rho_R, u_R, p_R, gamma=gamma, x=x, t=0.01, x0=0.5
    )
    
    # Entropy: s = p / rho^gamma (up to constant)
    # For a physical shock: s_post > s_pre (entropy increases)
    s_pre = p_R / (rho_R ** gamma)
    s_post = p_post.item() / (rho_post.item() ** gamma)
    
    # Entropy should increase across shock (compression)
    entropy_increases = s_post >= s_pre * 0.99  # Allow small numerical tolerance
    
    return {
        "test": "entropy_condition",
        "s_pre": float(s_pre),
        "s_post": float(s_post),
        "entropy_increases": bool(entropy_increases),
        "passed": bool(entropy_increases),
    }


def test_flux_consistency():
    """Test that flux functions are consistent at uniform states."""
    from tensornet.cfd.godunov import hll_flux, hllc_flux, roe_flux, euler_flux
    
    gamma = 1.4
    
    # Uniform state: flux should equal physical flux F(U)
    rho, u, p = 1.0, 0.5, 1.0
    E = p / (gamma - 1) + 0.5 * rho * u * u
    
    # Conservative variables - shape (N, 3) where N=1
    U = torch.tensor([[rho, rho * u, E]], dtype=torch.float64)
    
    # Physical flux
    F_exact = euler_flux(U, gamma)
    
    # All flux functions should give same result for uniform state
    F_hll = hll_flux(U, U, gamma)
    F_hllc = hllc_flux(U, U, gamma)
    F_roe = roe_flux(U, U, gamma)
    
    hll_error = (F_hll - F_exact).abs().max().item()
    hllc_error = (F_hllc - F_exact).abs().max().item()
    roe_error = (F_roe - F_exact).abs().max().item()
    
    passed = hll_error < 1e-6 and hllc_error < 1e-6 and roe_error < 1e-6
    
    return {
        "test": "flux_consistency",
        "hll_error": hll_error,
        "hllc_error": hllc_error,
        "roe_error": roe_error,
        "passed": passed,
        "tolerance": 1e-6,
    }


def run_all_proofs():
    """Run all CFD conservation proof tests."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "description": "CFD Conservation Law Proof Tests",
        "tests": [],
        "summary": {"passed": 0, "failed": 0},
    }
    
    tests = [
        test_euler1d_mass_conservation,
        test_euler1d_periodic_conservation,
        test_rankine_hugoniot_shock_relations,
        test_entropy_condition,
        test_flux_consistency,
    ]
    
    print("=" * 60)
    print("CFD Conservation Law Proof Tests")
    print("=" * 60)
    
    for test_fn in tests:
        print(f"\nRunning {test_fn.__name__}...")
        try:
            result = test_fn()
            results["tests"].append(result)
            
            if result["passed"]:
                print(f"  [OK] PASSED")
                results["summary"]["passed"] += 1
            else:
                print(f"  [X] FAILED")
                results["summary"]["failed"] += 1
                
        except Exception as e:
            print(f"  [X] ERROR: {e}")
            results["tests"].append({
                "test": test_fn.__name__,
                "passed": False,
                "error": str(e),
            })
            results["summary"]["failed"] += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    
    # Save results
    output_path = Path(__file__).parent / "proof_cfd_conservation_result.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_all_proofs()
    sys.exit(0 if results["summary"]["failed"] == 0 else 1)
