"""
Phase 1a Comprehensive Proof: 2D Incompressible Navier-Stokes
==============================================================

This proof file validates the complete Phase 1a implementation:
1. Poisson solver convergence and self-consistency
2. Projection method for incompressibility
3. 2D NS solver with Taylor-Green benchmark

Phase 1a Gate Criteria:
- Poisson solver: O(dx^2) convergence, self-consistency < 1e-10
- Projection: divergence after < 10⁻⁶
- Taylor-Green: decay rate error < 5%

Constitution Compliance: Article IV.1 (Verification)
Tag: [PHASE-1A] [DECISION-005]
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_phase_1a_proofs():
    """Run all Phase 1a proofs."""

    print("=" * 70)
    print("PHASE 1A COMPREHENSIVE PROOF")
    print("2D Incompressible Navier-Stokes Implementation")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    results = {
        "timestamp": datetime.now().isoformat(),
        "phase": "1a",
        "tests": [],
        "all_passed": False,
    }

    # ===== Test 1: Poisson Solver =====
    print("TEST 1: Poisson Solver")
    print("-" * 70)
    from ontic.cfd.tt_poisson import test_poisson_solver

    poisson_result = test_poisson_solver()
    results["tests"].append(
        {
            "name": "poisson_solver",
            "passed": poisson_result["passed"],
            "details": {
                "self_consistency_passed": poisson_result["self_consistency_passed"],
                "order_passed": poisson_result["order_passed"],
            },
        }
    )

    # ===== Test 2: Projection =====
    print("\nTEST 2: Velocity Projection")
    print("-" * 70)
    from ontic.cfd.tt_poisson import test_projection

    proj_result = test_projection()
    results["tests"].append(
        {
            "name": "projection",
            "passed": proj_result["gate_passed"],
            "details": {
                "divergence_before": proj_result["divergence_before"],
                "divergence_after": proj_result["divergence_after"],
            },
        }
    )

    # ===== Test 3: Advection Operator =====
    print("\nTEST 3: Advection Operator")
    print("-" * 70)
    from ontic.cfd.tt_poisson import test_advection

    adv_result = test_advection()
    results["tests"].append(
        {
            "name": "advection",
            "passed": adv_result["passed"],
        }
    )

    # ===== Test 4: Taylor-Green Benchmark =====
    print("\nTEST 4: Taylor-Green Vortex Benchmark")
    print("-" * 70)
    from ontic.cfd.ns_2d import test_taylor_green

    tg_result = test_taylor_green()
    results["tests"].append(
        {
            "name": "taylor_green",
            "passed": tg_result["passed"],
            "details": {
                "decay_error": tg_result["decay_error"],
                "max_velocity_error": tg_result["max_velocity_error"],
                "decay_passed": tg_result["decay_passed"],
                "divergence_passed": tg_result["divergence_passed"],
            },
        }
    )

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("PHASE 1A PROOF SUMMARY")
    print("=" * 70)

    all_passed = all(t["passed"] for t in results["tests"])
    results["all_passed"] = all_passed

    for test in results["tests"]:
        status = "PASS PASS" if test["passed"] else "FAIL FAIL"
        print(f"  {test['name']:.<40} {status}")

    print("-" * 70)
    final_status = "PASS" if all_passed else "FAIL"
    print(f"  {'PHASE 1A GATE':.<40} {final_status}")
    print("=" * 70)

    # Save results
    output_path = Path(__file__).parent / "proof_phase_1a_result.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = run_phase_1a_proofs()
    sys.exit(0 if results["all_passed"] else 1)
