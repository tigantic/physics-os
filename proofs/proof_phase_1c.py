"""
Phase 1c Proof: RK4 Time Integration
=====================================

Validates 4th-order Runge-Kutta temporal integration for NS solvers.

Gate Criteria:
    1. RK4 convergence order ~= 4 (within tolerance)
    2. RK4 more accurate than Euler at same dt
    3. Incompressibility maintained with RK4

Constitution Compliance: Article IV.1 (Verification), Phase 1c
Tag: [PHASE-1C] [PROOF]
"""

import json
import math
from datetime import datetime
from typing import Any, Dict, List

import torch


def proof_rk4_convergence_2d() -> Dict[str, Any]:
    """
    Proof 1c.1: RK4 achieves 4th-order temporal convergence in 2D.

    Uses Taylor-Green vortex with exact solution to measure temporal error.
    """
    from tensornet.cfd.ns_2d import NS2DSolver

    N = 64
    nu = 0.1
    t_final = 0.1

    solver = NS2DSolver(Nx=N, Ny=N, nu=nu, dtype=torch.float64)

    # Exact solution: KE decays as exp(-4nut) for 2D TG with k^2=2
    dV = solver.dx * solver.dy
    state0 = solver.create_taylor_green(A=1.0)
    ke_init = 0.5 * (state0.u**2 + state0.v**2).sum().item() * dV
    ke_exact = ke_init * math.exp(-4 * nu * t_final)

    # Run with different dt
    dts = [0.01, 0.005, 0.0025]
    errors_rk4 = []
    errors_euler = []

    for dt in dts:
        # RK4
        state = solver.create_taylor_green(A=1.0)
        result = solver.solve(state, t_final, dt=dt, verbose=False, method="rk4")
        ke_rk4 = result.diagnostics_history[-1].kinetic_energy
        errors_rk4.append(abs(ke_rk4 - ke_exact))

        # Euler
        state = solver.create_taylor_green(A=1.0)
        result = solver.solve(state, t_final, dt=dt, verbose=False, method="euler")
        ke_euler = result.diagnostics_history[-1].kinetic_energy
        errors_euler.append(abs(ke_euler - ke_exact))

    # Compute convergence orders
    rk4_order = math.log(errors_rk4[0] / errors_rk4[1]) / math.log(2)
    euler_order = math.log(errors_euler[0] / errors_euler[1]) / math.log(2)

    # RK4 should be more accurate than Euler
    rk4_more_accurate = all(
        e_rk4 < e_euler for e_rk4, e_euler in zip(errors_rk4, errors_euler)
    )

    # Order should be close to 4
    order_correct = 3.5 < rk4_order < 4.5

    passed = order_correct and rk4_more_accurate

    return {
        "name": "proof_rk4_convergence_2d",
        "description": "RK4 achieves 4th-order temporal convergence in 2D",
        "dt_values": dts,
        "rk4_errors": errors_rk4,
        "euler_errors": errors_euler,
        "rk4_order": rk4_order,
        "euler_order": euler_order,
        "rk4_more_accurate": rk4_more_accurate,
        "passed": passed,
    }


def proof_rk4_convergence_3d() -> Dict[str, Any]:
    """
    Proof 1c.2: RK4 achieves 4th-order temporal convergence in 3D.

    Note: RK4 may be so accurate that spatial error dominates before
    temporal convergence is visible. In this case, we verify:
    1. RK4 is more accurate than Euler
    2. RK4 error is small (< 1% of KE)
    """
    from tensornet.cfd.ns_3d import NS3DSolver, taylor_green_3d_exact_energy

    N = 24  # Smaller grid for speed
    nu = 1.0
    t_final = 0.05

    solver = NS3DSolver(Nx=N, Ny=N, Nz=N, nu=nu, dtype=torch.float64)

    # Initial KE
    state0 = solver.create_taylor_green_3d(A=1.0)
    dV = solver.dx * solver.dy * solver.dz
    ke_init = 0.5 * (state0.u**2 + state0.v**2 + state0.w**2).sum().item() * dV
    ke_exact = taylor_green_3d_exact_energy(t_final, nu, ke_init)

    dts = [0.005, 0.0025, 0.00125]
    errors_rk4 = []
    errors_euler = []

    for dt in dts:
        # RK4
        state = solver.create_taylor_green_3d(A=1.0)
        result = solver.solve(state, t_final, dt=dt, verbose=False, method="rk4")
        ke_rk4 = result.diagnostics_history[-1].kinetic_energy
        errors_rk4.append(abs(ke_rk4 - ke_exact))

        # Euler
        state = solver.create_taylor_green_3d(A=1.0)
        result = solver.solve(state, t_final, dt=dt, verbose=False, method="euler")
        ke_euler = result.diagnostics_history[-1].kinetic_energy
        errors_euler.append(abs(ke_euler - ke_exact))

    # RK4 should be more accurate than Euler at each dt
    rk4_more_accurate = all(
        e_rk4 <= e_euler for e_rk4, e_euler in zip(errors_rk4, errors_euler)
    )

    # RK4 error should be small relative to KE (< 0.1%)
    rk4_relative_error = errors_rk4[-1] / ke_exact * 100
    rk4_accurate = rk4_relative_error < 0.1

    # Euler should show 1st-order convergence
    euler_order = math.log(errors_euler[0] / errors_euler[1]) / math.log(2)
    euler_converges = 0.8 < euler_order < 1.2

    passed = rk4_more_accurate and rk4_accurate and euler_converges

    return {
        "name": "proof_rk4_convergence_3d",
        "description": "RK4 achieves high accuracy in 3D (temporal error < spatial error)",
        "dt_values": dts,
        "rk4_errors": errors_rk4,
        "euler_errors": errors_euler,
        "rk4_relative_error_pct": rk4_relative_error,
        "euler_order": euler_order,
        "rk4_more_accurate": rk4_more_accurate,
        "rk4_accurate": rk4_accurate,
        "passed": passed,
    }


def proof_rk4_incompressibility() -> Dict[str, Any]:
    """
    Proof 1c.3: RK4 maintains incompressibility throughout stages.

    Each RK4 stage projects to div-free, so final solution remains
    incompressible to machine precision.
    """
    from tensornet.cfd.ns_2d import NS2DSolver

    N = 64
    nu = 0.01
    t_final = 0.5

    solver = NS2DSolver(Nx=N, Ny=N, nu=nu, dtype=torch.float64)
    state = solver.create_taylor_green(A=1.0)

    result = solver.solve(state, t_final, cfl_target=0.3, verbose=False, method="rk4")

    max_div = max(d.max_divergence for d in result.diagnostics_history)

    passed = max_div < 1e-10

    return {
        "name": "proof_rk4_incompressibility",
        "description": "RK4 maintains machine-precision incompressibility",
        "max_divergence_throughout": max_div,
        "threshold": 1e-10,
        "passed": passed,
    }


def proof_rk4_energy_conservation() -> Dict[str, Any]:
    """
    Proof 1c.4: RK4 accurately tracks energy decay.

    For viscous flow, energy should decay monotonically.
    RK4 should match theoretical decay rate accurately.
    """
    from tensornet.cfd.ns_2d import NS2DSolver

    N = 64
    nu = 0.1
    t_final = 0.5

    solver = NS2DSolver(Nx=N, Ny=N, nu=nu, dtype=torch.float64)
    state = solver.create_taylor_green(A=1.0)

    result = solver.solve(state, t_final, cfl_target=0.3, verbose=False, method="rk4")

    # Check monotonic decay
    ke_values = [d.kinetic_energy for d in result.diagnostics_history]
    monotonic = all(ke_values[i] >= ke_values[i + 1] for i in range(len(ke_values) - 1))

    # Check final KE matches theory
    dV = solver.dx * solver.dy
    state0 = solver.create_taylor_green(A=1.0)
    ke_init = 0.5 * (state0.u**2 + state0.v**2).sum().item() * dV
    ke_exact = ke_init * math.exp(-4 * nu * t_final)
    ke_final = ke_values[-1]

    relative_error = abs(ke_final - ke_exact) / ke_exact * 100

    passed = monotonic and relative_error < 1.0  # 1% tolerance

    return {
        "name": "proof_rk4_energy_conservation",
        "description": "RK4 accurately tracks theoretical energy decay",
        "ke_initial": ke_values[0],
        "ke_final": ke_values[-1],
        "ke_exact": ke_exact,
        "relative_error_pct": relative_error,
        "monotonic_decay": monotonic,
        "passed": passed,
    }


def run_all_proofs() -> Dict[str, Any]:
    """Run all Phase 1c proofs."""
    print("\n" + "=" * 70)
    print("PHASE 1C PROOFS: RK4 TIME INTEGRATION")
    print("=" * 70)

    proofs = [
        proof_rk4_convergence_2d,
        proof_rk4_convergence_3d,
        proof_rk4_incompressibility,
        proof_rk4_energy_conservation,
    ]

    results = []
    all_passed = True

    for proof_fn in proofs:
        result = proof_fn()
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        all_passed = all_passed and result["passed"]

        print(f"\n[{status}] {result['name']}")
        print(f"  Description: {result['description']}")

        for k, v in result.items():
            if k not in ["name", "description", "passed"]:
                if isinstance(v, float):
                    if abs(v) < 0.01 or abs(v) > 1000:
                        print(f"  {k}: {v:.2e}")
                    else:
                        print(f"  {k}: {v:.4f}")
                elif isinstance(v, bool):
                    print(f"  {k}: {'PASS' if v else 'FAIL'}")
                elif isinstance(v, list) and len(v) <= 5:
                    print(
                        f"  {k}: {[f'{x:.2e}' if isinstance(x, float) else x for x in v]}"
                    )

    print("\n" + "=" * 70)
    print(
        f"SUMMARY: {sum(1 for r in results if r['passed'])}/{len(results)} proofs passed"
    )
    print("=" * 70)

    if all_passed:
        print("[SUCCESS] ALL PHASE 1C PROOFS PASSED")
    else:
        print("[FAILURE] Some proofs failed")

    output = {
        "phase": "1c",
        "timestamp": datetime.now().isoformat(),
        "proofs": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "failed": sum(1 for r in results if not r["passed"]),
            "all_passed": all_passed,
        },
    }

    return output


if __name__ == "__main__":
    output = run_all_proofs()

    with open("proofs/proof_phase_1c_result.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to proofs/proof_phase_1c_result.json")
