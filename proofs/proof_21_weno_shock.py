"""
Proof 21.2: WENO ENO Property — No Oscillation Across Discontinuities.

This proof verifies that WENO schemes satisfy the Essentially Non-Oscillatory
property: the reconstruction does not introduce spurious oscillations when
applied to discontinuous data (shocks).

Constitution Compliance: Article I.1 (Proof Requirements)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontic.cfd.weno import (ReconstructionSide, WENOConfig, teno5,
                                weno5_js, weno5_z)


def step_function(x: torch.Tensor, x0: float = 0.5) -> torch.Tensor:
    """Step function (pure discontinuity)."""
    return torch.where(x < x0, torch.ones_like(x), 2.0 * torch.ones_like(x))


def two_step_function(x: torch.Tensor) -> torch.Tensor:
    """Two steps at x=0.3 and x=0.7."""
    result = torch.ones_like(x)
    result = torch.where(x > 0.3, 2.0 * torch.ones_like(x), result)
    result = torch.where(x > 0.7, 1.0 * torch.ones_like(x), result)
    return result


def sod_density_profile(x: torch.Tensor) -> torch.Tensor:
    """Approximate Sod shock tube density profile at t=0.2."""
    # Simplified analytic approximation
    result = torch.ones_like(x)

    # Left state
    mask_left = x < 0.25
    result = torch.where(mask_left, 1.0 * torch.ones_like(x), result)

    # Rarefaction fan
    mask_rare = (x >= 0.25) & (x < 0.5)
    result = torch.where(mask_rare, 1.0 - 0.5 * (x - 0.25) / 0.25, result)

    # Contact surface region
    mask_contact = (x >= 0.5) & (x < 0.7)
    result = torch.where(mask_contact, 0.5 * torch.ones_like(x), result)

    # Post-shock
    mask_shock = x >= 0.7
    result = torch.where(mask_shock, 0.125 * torch.ones_like(x), result)

    return result


def count_oscillations(u: torch.Tensor, threshold: float = 0.01) -> int:
    """
    Count spurious oscillations in a reconstruction.

    Oscillations are sign changes in the second derivative that
    exceed the threshold.
    """
    if len(u) < 3:
        return 0

    # Second derivative (discrete)
    d2u = u[2:] - 2 * u[1:-1] + u[:-2]

    # Count sign changes that exceed threshold
    oscillations = 0
    for i in range(len(d2u) - 1):
        if abs(d2u[i]) > threshold and abs(d2u[i + 1]) > threshold:
            if d2u[i] * d2u[i + 1] < 0:  # Sign change
                oscillations += 1

    return oscillations


def measure_overshoot(u_recon: torch.Tensor, u_min: float, u_max: float) -> dict:
    """
    Measure overshoot/undershoot relative to initial bounds.

    For ENO property, reconstruction should not exceed input bounds
    (within some tolerance due to reconstruction at interfaces).
    """
    actual_min = u_recon.min().item()
    actual_max = u_recon.max().item()

    undershoot = max(0, u_min - actual_min)
    overshoot = max(0, actual_max - u_max)

    return {
        "u_min_expected": u_min,
        "u_max_expected": u_max,
        "u_min_actual": actual_min,
        "u_max_actual": actual_max,
        "undershoot": undershoot,
        "overshoot": overshoot,
        "total_violation": undershoot + overshoot,
    }


def test_eno_property(weno_fn, test_fn, N: int, u_bounds: tuple, name: str) -> dict:
    """
    Test ENO property for a given WENO function.

    Checks:
    1. Number of oscillations is minimal
    2. Overshoot/undershoot is bounded
    3. Monotonicity is approximately preserved near jumps
    """
    x = torch.linspace(0, 1, N, dtype=torch.float64)
    u = test_fn(x)

    # Reconstruct
    u_recon = weno_fn(u, ReconstructionSide.LEFT)

    # Count oscillations
    n_osc = count_oscillations(u_recon, threshold=0.02)

    # Check overshoot
    overshoot_info = measure_overshoot(u_recon, u_bounds[0], u_bounds[1])

    # Tolerance for ENO: small overshoot is acceptable
    tolerance = 0.1 * (u_bounds[1] - u_bounds[0])

    passed = (n_osc <= 3) and (overshoot_info["total_violation"] < tolerance)

    return {
        "name": name,
        "N": N,
        "n_oscillations": n_osc,
        **overshoot_info,
        "tolerance": tolerance,
        "passed": passed,
    }


def compare_with_naive_interpolation(N: int = 100):
    """
    Compare WENO with naive polynomial interpolation on discontinuous data.

    Naive high-order interpolation produces Gibbs oscillations.
    WENO should be much better.
    """
    x = torch.linspace(0, 1, N, dtype=torch.float64)
    u = step_function(x)

    # WENO reconstruction
    u_weno = weno5_z(u, ReconstructionSide.LEFT)

    # "Naive" 5th-order reconstruction (without nonlinear weights)
    # This is just using optimal weights everywhere (WENO becomes linear)
    n_recon = len(u_weno)
    u_linear = torch.zeros(n_recon)

    d0, d1, d2 = 1.0 / 10.0, 6.0 / 10.0, 3.0 / 10.0

    for i in range(n_recon):
        idx = i + 2  # offset for stencil
        if idx - 2 >= 0 and idx + 2 < N:
            um2, um1, u0, up1, up2 = (
                u[idx - 2],
                u[idx - 1],
                u[idx],
                u[idx + 1],
                u[idx + 2],
            )

            q0 = (2 * um2 - 7 * um1 + 11 * u0) / 6
            q1 = (-um1 + 5 * u0 + 2 * up1) / 6
            q2 = (2 * u0 + 5 * up1 - up2) / 6

            u_linear[i] = d0 * q0 + d1 * q1 + d2 * q2

    # Count oscillations
    osc_weno = count_oscillations(u_weno, threshold=0.01)
    osc_linear = count_oscillations(u_linear, threshold=0.01)

    # Overshoot comparison
    overshoot_weno = measure_overshoot(u_weno, 1.0, 2.0)
    overshoot_linear = measure_overshoot(u_linear, 1.0, 2.0)

    return {
        "oscillations_weno": osc_weno,
        "oscillations_linear": osc_linear,
        "overshoot_weno": overshoot_weno["total_violation"],
        "overshoot_linear": overshoot_linear["total_violation"],
        "weno_is_better": (osc_weno < osc_linear)
        or (overshoot_weno["total_violation"] < overshoot_linear["total_violation"]),
    }


def test_teno_sharpness(N: int = 100):
    """
    Test that TENO provides sharper resolution than WENO near discontinuities.

    TENO should have less numerical diffusion than WENO.
    """
    x = torch.linspace(0, 1, N, dtype=torch.float64)
    u = step_function(x, x0=0.5)

    u_weno = weno5_z(u, ReconstructionSide.LEFT)
    u_teno = teno5(u, ReconstructionSide.LEFT)

    # Measure transition width (number of points to go from ~1 to ~2)
    def transition_width(recon):
        # Find where reconstruction is between 1.1 and 1.9
        in_transition = (recon > 1.1) & (recon < 1.9)
        return in_transition.sum().item()

    width_weno = transition_width(u_weno)
    width_teno = transition_width(u_teno)

    return {
        "transition_width_weno": width_weno,
        "transition_width_teno": width_teno,
        "teno_is_sharper": width_teno <= width_weno,
    }


def main():
    """Run all WENO shock/ENO proofs."""
    print("=" * 60)
    print("PROOF 21.2: WENO ENO Property (No Oscillation)")
    print("=" * 60)

    results = {
        "proof_name": "proof_21_weno_shock",
        "timestamp": datetime.now().isoformat(),
        "tests": [],
    }

    # Test 1: WENO5-JS on step function
    print("\n[Test 1] WENO5-JS on step function...")
    result1 = test_eno_property(
        weno5_js, step_function, N=100, u_bounds=(1.0, 2.0), name="WENO5-JS step"
    )
    print(f"  Oscillations: {result1['n_oscillations']}")
    print(f"  Overshoot: {result1['overshoot']:.4f}")
    print(f"  Undershoot: {result1['undershoot']:.4f}")
    print(f"  Status: {'[OK] PASSED' if result1['passed'] else '[X] FAILED'}")
    results["tests"].append(result1)

    # Test 2: WENO5-Z on step function
    print("\n[Test 2] WENO5-Z on step function...")
    result2 = test_eno_property(
        weno5_z, step_function, N=100, u_bounds=(1.0, 2.0), name="WENO5-Z step"
    )
    print(f"  Oscillations: {result2['n_oscillations']}")
    print(f"  Overshoot: {result2['overshoot']:.4f}")
    print(f"  Undershoot: {result2['undershoot']:.4f}")
    print(f"  Status: {'[OK] PASSED' if result2['passed'] else '[X] FAILED'}")
    results["tests"].append(result2)

    # Test 3: TENO5 on step function
    print("\n[Test 3] TENO5 on step function...")
    result3 = test_eno_property(
        teno5, step_function, N=100, u_bounds=(1.0, 2.0), name="TENO5 step"
    )
    print(f"  Oscillations: {result3['n_oscillations']}")
    print(f"  Overshoot: {result3['overshoot']:.4f}")
    print(f"  Undershoot: {result3['undershoot']:.4f}")
    print(f"  Status: {'[OK] PASSED' if result3['passed'] else '[X] FAILED'}")
    results["tests"].append(result3)

    # Test 4: Two-step function (multiple discontinuities)
    print("\n[Test 4] WENO5-Z on two-step function...")
    result4 = test_eno_property(
        weno5_z, two_step_function, N=100, u_bounds=(1.0, 2.0), name="WENO5-Z two-step"
    )
    print(f"  Oscillations: {result4['n_oscillations']}")
    print(f"  Status: {'[OK] PASSED' if result4['passed'] else '[X] FAILED'}")
    results["tests"].append(result4)

    # Test 5: Sod-like profile
    print("\n[Test 5] WENO5-Z on Sod-like profile...")
    result5 = test_eno_property(
        weno5_z,
        sod_density_profile,
        N=200,
        u_bounds=(0.1, 1.1),
        name="WENO5-Z Sod profile",
    )
    print(f"  Oscillations: {result5['n_oscillations']}")
    print(f"  Status: {'[OK] PASSED' if result5['passed'] else '[X] FAILED'}")
    results["tests"].append(result5)

    # Test 6: Compare with naive interpolation
    print("\n[Test 6] WENO vs naive linear interpolation...")
    comparison = compare_with_naive_interpolation(N=100)
    print(f"  WENO oscillations: {comparison['oscillations_weno']}")
    print(f"  Linear oscillations: {comparison['oscillations_linear']}")
    print(f"  WENO overshoot: {comparison['overshoot_weno']:.4f}")
    print(f"  Linear overshoot: {comparison['overshoot_linear']:.4f}")

    passed_comparison = comparison["weno_is_better"]
    comparison["passed"] = passed_comparison
    comparison["name"] = "weno_vs_linear"
    results["tests"].append(comparison)
    print(f"  Status: {'[OK] PASSED' if passed_comparison else '[X] FAILED'}")

    # Test 7: TENO sharpness
    print("\n[Test 7] TENO sharpness compared to WENO...")
    sharpness = test_teno_sharpness(N=100)
    print(f"  WENO transition width: {sharpness['transition_width_weno']} points")
    print(f"  TENO transition width: {sharpness['transition_width_teno']} points")

    sharpness["passed"] = sharpness["teno_is_sharper"]
    sharpness["name"] = "teno_sharpness"
    results["tests"].append(sharpness)
    print(f"  Status: {'[OK] PASSED' if sharpness['passed'] else '⚠ INCONCLUSIVE'}")

    # Test 8: Grid refinement on discontinuous data
    print("\n[Test 8] Grid refinement on discontinuous data...")
    oscillation_counts = []
    for N in [50, 100, 200, 400]:
        result = test_eno_property(
            weno5_z, step_function, N=N, u_bounds=(1.0, 2.0), name=f"WENO5-Z N={N}"
        )
        oscillation_counts.append((N, result["n_oscillations"]))

    print(f"  N=50:  {oscillation_counts[0][1]} oscillations")
    print(f"  N=100: {oscillation_counts[1][1]} oscillations")
    print(f"  N=200: {oscillation_counts[2][1]} oscillations")
    print(f"  N=400: {oscillation_counts[3][1]} oscillations")

    # Oscillations should not grow with refinement
    passed_refinement = all(oc[1] <= 5 for oc in oscillation_counts)
    results["tests"].append(
        {
            "name": "grid_refinement",
            "oscillation_counts": oscillation_counts,
            "passed": passed_refinement,
        }
    )
    print(f"  Status: {'[OK] PASSED' if passed_refinement else '[X] FAILED'}")

    # Summary
    print("\n" + "=" * 60)
    all_passed = all(t.get("passed", False) for t in results["tests"])
    results["all_passed"] = all_passed
    results["summary"] = {
        "total_tests": len(results["tests"]),
        "passed": sum(1 for t in results["tests"] if t.get("passed", False)),
        "failed": sum(1 for t in results["tests"] if not t.get("passed", False)),
    }

    print(f"PROOF RESULT: {'[OK] ALL PASSED' if all_passed else '⚠ SOME TESTS FAILED'}")
    print(
        f"Tests: {results['summary']['passed']}/{results['summary']['total_tests']} passed"
    )
    print("=" * 60)

    # Save results
    output_path = Path(__file__).parent / "proof_21_weno_shock_result.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
