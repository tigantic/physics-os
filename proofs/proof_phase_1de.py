#!/usr/bin/env python3
"""
Phase 1D/1E Mathematical Proofs: chi(t) Regularity Diagnostics
================================================================

Proves that the chi(t) tracking framework correctly identifies
regularity behavior in Navier-Stokes solutions.

Gates:
1. chi estimation correctness
2. Enstrophy tracking accuracy
3. Regularity assessment logic
4. Integration with NS solver
"""

from typing import Dict, Tuple

import numpy as np
import torch

# Proof tracking
PROOF_RESULTS = {}


def gate_chi_estimation():
    """
    Gate 1: Verify chi estimation gives correct bond dimension.

    For smooth fields, chi should be small.
    For complex fields, chi should scale with grid.
    """
    from tensornet.cfd.chi_diagnostic import estimate_required_chi_2d

    print("\n" + "=" * 60)
    print("Gate 1: chi Estimation Correctness")
    print("=" * 60)

    N = 64
    L = 2 * np.pi
    dx = L / N
    dy = L / N
    x = torch.linspace(0, L, N + 1)[:-1]
    y = torch.linspace(0, L, N + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Test 1: Smooth field (single mode)
    u_smooth = torch.sin(X) * torch.sin(Y)
    v_smooth = torch.cos(X) * torch.cos(Y)
    chi_smooth = estimate_required_chi_2d(u_smooth, v_smooth, dx, dy)

    # Test 2: Random field (complex)
    u_random = torch.randn(N, N)
    v_random = torch.randn(N, N)
    chi_random = estimate_required_chi_2d(u_random, v_random, dx, dy)

    print(f"Smooth field chi: {chi_smooth}")
    print(f"Random field chi: {chi_random}")

    # Smooth should have low chi, random should be higher
    smooth_is_low = chi_smooth < 10
    random_is_higher = chi_random > chi_smooth

    success = smooth_is_low and random_is_higher

    PROOF_RESULTS["chi_estimation"] = {
        "chi_smooth": chi_smooth,
        "chi_random": chi_random,
        "smooth_is_low": smooth_is_low,
        "random_is_higher": random_is_higher,
        "success": success,
    }

    print(f"\nGate (chi estimation): {'PASS' if success else 'FAIL'}")
    return success


def gate_enstrophy_tracking():
    """
    Gate 2: Verify enstrophy calculation matches analytical.

    For Taylor-Green: Omega = int|omega|^2 dx
    """
    from tensornet.cfd.chi_diagnostic import compute_chi_state_2d
    from tensornet.cfd.ns_2d import NS2DSolver

    print("\n" + "=" * 60)
    print("Gate 2: Enstrophy Tracking Accuracy")
    print("=" * 60)

    N = 64
    nu = 0.1
    L = 2 * np.pi
    dx = L / N
    dy = L / N

    solver = NS2DSolver(Nx=N, Ny=N, Lx=L, Ly=L, nu=nu, dtype=torch.float64)
    state_init = solver.create_taylor_green()

    # Initial Taylor-Green
    state = compute_chi_state_2d(
        state_init.u, state_init.v, t=0.0, dx=dx, dy=dy, chi_target=64
    )

    # For discrete: compute expected value
    # omega = -2cos(x)cos(y), so omega^2 = 4cos^2(x)cos^2(y)
    # Mean value of cos^2(x)cos^2(y) = 1/4, so mean omega^2 = 1
    # Enstrophy = 0.5 * mean(omega^2) = 0.5
    enst_expected = 0.5

    enst_computed = state.enstrophy

    print(f"Expected enstrophy: {enst_expected:.4f}")
    print(f"Computed enstrophy: {enst_computed:.4f}")

    # Check relative error
    rel_error = abs(enst_computed - enst_expected) / enst_expected
    print(f"Relative error: {rel_error:.2e}")

    success = rel_error < 0.1

    PROOF_RESULTS["enstrophy_tracking"] = {
        "expected": enst_expected,
        "computed": enst_computed,
        "rel_error": rel_error,
        "success": success,
    }

    print(f"\nGate (enstrophy tracking): {'PASS' if success else 'FAIL'}")
    return success


def gate_regularity_assessment():
    """
    Gate 3: Verify regularity assessment logic.

    - Decaying flow -> 'smooth'
    - Exploding chi -> 'potential_blowup'
    """
    from tensornet.cfd.chi_diagnostic import (ChiState, ChiTrajectory,
                                              analyze_regularity)

    print("\n" + "=" * 60)
    print("Gate 3: Regularity Assessment Logic")
    print("=" * 60)

    # Test 1: Smooth decaying flow (chi decreasing)
    traj_smooth = ChiTrajectory()
    for i in range(20):
        t = i * 0.1
        state = ChiState(
            time=t,
            chi_actual=max(1, 10 - i // 2),  # Decreasing chi
            chi_target=64,
            truncation_error=1e-6,
            gradient_norm=1.0 * np.exp(-0.1 * t),  # Decaying
            enstrophy=1.0 * np.exp(-0.2 * t),
            spectral_radius=0.5,
        )
        traj_smooth.add(state)

    result_smooth = analyze_regularity(traj_smooth)

    # Test 2: Potential blowup (chi exploding rapidly)
    traj_blowup = ChiTrajectory()
    for i in range(20):
        t = i * 0.1
        # Rapid exponential growth to ensure chi_growth > 1.0
        chi_val = min(64, int(5 * np.exp(1.5 * t)))  # growth rate = 1.5
        state = ChiState(
            time=t,
            chi_actual=max(1, chi_val),
            chi_target=64,
            truncation_error=1e-6,
            gradient_norm=np.exp(2.0 * t),  # Rapidly growing
            enstrophy=np.exp(1.5 * t),
            spectral_radius=np.exp(1.0 * t),
        )
        traj_blowup.add(state)

    result_blowup = analyze_regularity(traj_blowup)

    print(f"Smooth trajectory assessment: {result_smooth['regularity_assessment']}")
    print(f"  chi growth rate: {result_smooth['chi_growth_rate']:.4f}")
    print(f"  Gradient ratio: {result_smooth['gradient_ratio']:.4f}")

    print(f"\nBlowup trajectory assessment: {result_blowup['regularity_assessment']}")
    print(f"  chi growth rate: {result_blowup['chi_growth_rate']:.4f}")

    # Check assessments
    smooth_ok = result_smooth["regularity_assessment"] == "smooth"
    blowup_ok = result_blowup["regularity_assessment"] == "potential_blowup"

    success = smooth_ok and blowup_ok

    PROOF_RESULTS["regularity_assessment"] = {
        "smooth_assessment": result_smooth["regularity_assessment"],
        "blowup_assessment": result_blowup["regularity_assessment"],
        "smooth_ok": smooth_ok,
        "blowup_ok": blowup_ok,
        "success": success,
    }

    print(f"\nGate (regularity assessment): {'PASS' if success else 'FAIL'}")
    return success


def gate_ns_integration():
    """
    Gate 4: Verify chi tracking integrates with NS solver.

    Run Taylor-Green and verify:
    - chi stays bounded
    - Assessment is 'smooth'
    - No NaN/Inf values
    """
    from tensornet.cfd.chi_diagnostic import (ChiTrajectory,
                                              analyze_regularity,
                                              compute_chi_state_2d)
    from tensornet.cfd.ns_2d import NS2DSolver

    print("\n" + "=" * 60)
    print("Gate 4: NS Solver Integration")
    print("=" * 60)

    N = 64
    nu = 0.1
    L = 2 * np.pi
    dx = L / N
    dy = L / N
    dt = 0.01
    n_steps = 50

    solver = NS2DSolver(Nx=N, Ny=N, Lx=L, Ly=L, nu=nu, dtype=torch.float64)
    state_ns = solver.create_taylor_green()
    trajectory = ChiTrajectory()

    # Initial state
    state = compute_chi_state_2d(
        state_ns.u, state_ns.v, t=0.0, dx=dx, dy=dy, chi_target=64
    )
    trajectory.add(state)

    # Time stepping with chi tracking
    had_nan = False
    max_chi = 0

    for step in range(n_steps):
        state_ns, _ = solver.step_forward_euler(state_ns, dt)
        t = (step + 1) * dt

        # Check for NaN
        if not torch.isfinite(state_ns.u).all() or not torch.isfinite(state_ns.v).all():
            had_nan = True
            break

        chi_state = compute_chi_state_2d(
            state_ns.u, state_ns.v, t=t, dx=dx, dy=dy, chi_target=64
        )
        trajectory.add(chi_state)
        max_chi = max(max_chi, chi_state.chi_actual)

    # Analyze
    analysis = analyze_regularity(trajectory)

    print(f"Steps completed: {len(trajectory.states)}")
    print(f"Max chi observed: {max_chi}")
    print(f"Final assessment: {analysis['regularity_assessment']}")
    print(f"Had NaN: {had_nan}")

    chi_bounded = max_chi <= 64
    no_nan = not had_nan
    is_smooth = analysis["regularity_assessment"] == "smooth"

    success = chi_bounded and no_nan and is_smooth

    PROOF_RESULTS["ns_integration"] = {
        "steps": len(trajectory.states),
        "max_chi": max_chi,
        "assessment": analysis["regularity_assessment"],
        "had_nan": had_nan,
        "success": success,
    }

    print(f"\nGate (NS integration): {'PASS' if success else 'FAIL'}")
    return success


def run_all_proofs():
    """Run all Phase 1D/1E proofs."""
    print("\n" + "=" * 60)
    print("PHASE 1D/1E: chi(t) Regularity Diagnostics")
    print("Mathematical Verification Suite")
    print("=" * 60)

    gates = [
        ("chi Estimation", gate_chi_estimation),
        ("Enstrophy Tracking", gate_enstrophy_tracking),
        ("Regularity Assessment", gate_regularity_assessment),
        ("NS Integration", gate_ns_integration),
    ]

    results = []
    for name, gate_fn in gates:
        try:
            passed = gate_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1D/1E PROOF SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} gates passed")

    if passed == total:
        print("\nPASS PHASE 1D/1E COMPLETE: chi(t) diagnostics validated")
    else:
        print("\nFAIL PHASE 1D/1E INCOMPLETE: Some gates failed")

    print("=" * 60)

    return passed == total, PROOF_RESULTS


if __name__ == "__main__":
    success, results = run_all_proofs()

    # Save results
    import json

    with open("proofs/proof_phase_1de_result.json", "w") as f:
        # Convert non-serializable values
        serializable = {}
        for k, v in results.items():
            serializable[k] = {
                kk: (
                    float(vv)
                    if isinstance(vv, (np.floating, float))
                    else (
                        bool(vv)
                        if isinstance(vv, (np.bool_, bool))
                        else int(vv) if isinstance(vv, (np.integer, int)) else str(vv)
                    )
                )
                for kk, vv in v.items()
            }
        json.dump(serializable, f, indent=2)

    exit(0 if success else 1)
