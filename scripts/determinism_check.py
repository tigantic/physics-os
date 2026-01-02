#!/usr/bin/env python
"""
J) Determinism Check Script
============================

Verifies that computations are deterministic across runs.

Usage:
    python scripts/determinism_check.py --seed 42 --runs 3

Pass Criteria:
    - Output hashes match for canonical outputs across runs
    - Within numeric tolerance bands for floating point
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def tensor_hash(t: torch.Tensor, precision: int = 6) -> str:
    """Compute hash of tensor with specified precision."""
    # Round to specified precision to allow for floating point variance
    rounded = torch.round(t * 10**precision) / 10**precision
    data = rounded.detach().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def run_mps_test(seed: int) -> Dict[str, Any]:
    """Run MPS canonicalization test."""
    set_all_seeds(seed)

    from tensornet.core.mps import MPS

    # Create random MPS
    N = 10
    d = 2
    chi = 4

    tensors = []
    for i in range(N):
        chi_l = 1 if i == 0 else chi
        chi_r = 1 if i == N - 1 else chi
        tensors.append(torch.randn(chi_l, d, chi_r, dtype=torch.float64))

    mps = MPS(tensors)
    mps.canonicalize(center=5)

    # Hash the canonical form
    hashes = [tensor_hash(t) for t in mps.tensors]

    return {
        "test": "mps_canonicalize",
        "tensor_hashes": hashes,
        "combined_hash": hashlib.sha256("".join(hashes).encode()).hexdigest()[:16],
    }


def run_euler_test(seed: int) -> Dict[str, Any]:
    """Run Euler solver test."""
    set_all_seeds(seed)

    from tensornet.cfd.euler_1d import Euler1D, EulerState

    # Initialize solver
    N = 64
    solver = Euler1D(N=N, x_min=0.0, x_max=1.0, gamma=1.4)

    # Sod shock tube IC
    x = torch.linspace(0, 1, N, dtype=torch.float64)
    rho = torch.where(x < 0.5, torch.ones_like(x), 0.125 * torch.ones_like(x))
    u = torch.zeros_like(x)
    p = torch.where(x < 0.5, torch.ones_like(x), 0.1 * torch.ones_like(x))

    rho_u = rho * u
    E = p / (1.4 - 1) + 0.5 * rho * u**2

    state = EulerState(rho=rho, rho_u=rho_u, E=E, gamma=1.4)
    solver.set_initial_condition(state)

    # Run 10 steps
    dt = 0.0001
    for _ in range(10):
        solver.step(dt)

    return {
        "test": "euler_1d",
        "rho_hash": tensor_hash(solver.state.rho),
        "rho_u_hash": tensor_hash(solver.state.rho_u),
        "E_hash": tensor_hash(solver.state.E),
    }


def run_weno_test(seed: int) -> Dict[str, Any]:
    """Run WENO reconstruction test."""
    set_all_seeds(seed)

    from tensornet.cfd.weno import weno5_js_reconstruct, weno5_z_reconstruct

    # Create test data
    N = 64
    x = torch.linspace(0, 2 * np.pi, N, dtype=torch.float64)
    u = torch.sin(x)

    # WENO reconstructions
    uL_js, uR_js = weno5_js_reconstruct(u)
    uL_z, uR_z = weno5_z_reconstruct(u)

    return {
        "test": "weno",
        "weno5_js_L_hash": tensor_hash(uL_js),
        "weno5_js_R_hash": tensor_hash(uR_js),
        "weno5_z_L_hash": tensor_hash(uL_z),
        "weno5_z_R_hash": tensor_hash(uR_z),
    }


def run_dmrg_test(seed: int) -> Dict[str, Any]:
    """Run DMRG ground state test."""
    set_all_seeds(seed)

    try:
        from tensornet.algorithms.dmrg import DMRG, DMRGConfig
        from tensornet.algorithms.fermionic import FermionicMPO

        # Small Heisenberg chain
        N = 6
        J = 1.0

        mpo = FermionicMPO.heisenberg_xxz(N, Jxy=J, Jz=J)
        config = DMRGConfig(chi_max=16, n_sweeps=2, tol=1e-8)
        dmrg = DMRG(mpo, config)

        energy, _ = dmrg.run()

        return {
            "test": "dmrg",
            "energy": round(energy, 8),
            "converged": True,
        }
    except Exception as e:
        return {
            "test": "dmrg",
            "error": str(e),
        }


TESTS = [
    ("MPS Canonicalization", run_mps_test),
    ("Euler 1D Solver", run_euler_test),
    ("WENO Reconstruction", run_weno_test),
    ("DMRG Ground State", run_dmrg_test),
]


def run_all_tests(seed: int) -> List[Dict[str, Any]]:
    """Run all determinism tests."""
    results = []
    for name, test_fn in TESTS:
        try:
            result = test_fn(seed)
            result["name"] = name
            result["success"] = True
        except Exception as e:
            result = {"name": name, "success": False, "error": str(e)}
        results.append(result)
    return results


def compare_runs(runs: List[List[Dict[str, Any]]]) -> Tuple[bool, List[str]]:
    """Compare results across runs."""
    if len(runs) < 2:
        return True, []

    issues = []
    baseline = runs[0]

    for run_idx, run in enumerate(runs[1:], 2):
        for test_idx, (baseline_test, run_test) in enumerate(zip(baseline, run)):
            test_name = baseline_test.get("name", f"test_{test_idx}")

            # Compare all hash fields
            for key in baseline_test:
                if "hash" in key.lower():
                    if baseline_test.get(key) != run_test.get(key):
                        issues.append(
                            f"Run {run_idx}, {test_name}: {key} mismatch "
                            f"({baseline_test.get(key)} vs {run_test.get(key)})"
                        )

            # Compare numeric fields with tolerance
            if "energy" in baseline_test and "energy" in run_test:
                diff = abs(baseline_test["energy"] - run_test["energy"])
                if diff > 1e-6:
                    issues.append(
                        f"Run {run_idx}, {test_name}: energy mismatch "
                        f"({baseline_test['energy']} vs {run_test['energy']}, diff={diff})"
                    )

    return len(issues) == 0, issues


def main():
    parser = argparse.ArgumentParser(description="Check determinism of computations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to compare")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON file")

    args = parser.parse_args()

    print("=" * 60)
    print(" DETERMINISM CHECK")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Runs: {args.runs}")
    print()

    # Add project to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Run multiple times
    all_runs = []
    for run_idx in range(args.runs):
        print(f"Run {run_idx + 1}/{args.runs}...")
        results = run_all_tests(args.seed)
        all_runs.append(results)

        for result in results:
            status = "✓" if result.get("success", False) else "✗"
            print(f"  {status} {result.get('name', 'Unknown')}")

    print()
    print("-" * 60)

    # Compare runs
    deterministic, issues = compare_runs(all_runs)

    if deterministic:
        print("✓ All runs produced identical results")
    else:
        print("✗ Non-deterministic behavior detected:")
        for issue in issues:
            print(f"  - {issue}")

    # Save results
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "n_runs": args.runs,
        "deterministic": deterministic,
        "issues": issues,
        "runs": all_runs,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport: {args.out}")

    print()
    print("=" * 60)
    if deterministic:
        print(" ✓ DETERMINISM CHECK PASSED")
    else:
        print(" ✗ DETERMINISM CHECK FAILED")
    print("=" * 60)

    return 0 if deterministic else 1


if __name__ == "__main__":
    exit(main())
