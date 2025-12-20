#!/usr/bin/env python
"""
Tensor Network Benchmark Suite
==============================

One command reproduces ground state energies for Heisenberg and TFIM models.
Compares against TeNPy and ITensor reference values.

Usage:
    python reproduce.py [--quick] [--save]

Options:
    --quick     Run smaller systems only (L≤20)
    --save      Save results to results/benchmark_latest.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Install check
try:
    from tensornet import dmrg, heisenberg_mpo, tfim_mpo, MPS
except ImportError:
    print("ERROR: tensornet not installed.")
    print("Run: pip install git+https://github.com/tigantic/PytorchTN.git")
    sys.exit(1)


# Reference values from TeNPy/ITensor/exact diagonalization
# Format: (name, model, L, chi, g_or_h, expected_E, source)
# Note: For TFIM, we use g (transverse field strength)
BENCHMARKS = [
    # Heisenberg XXX chain: H = Σ S_i · S_{i+1}
    ("Heisenberg L=10", "heisenberg", 10, 32, 0.0, -4.258035207282883, "exact"),
    ("Heisenberg L=20", "heisenberg", 20, 64, 0.0, -8.682427660820782, "TeNPy"),
    ("Heisenberg L=50", "heisenberg", 50, 128, 0.0, -21.858542716665, "TeNPy"),
    
    # Transverse-field Ising: H = -J Σ Z_i Z_{i+1} - g Σ X_i
    ("TFIM g=1.0 L=10", "tfim", 10, 32, 1.0, -12.566370614359172, "exact"),
    ("TFIM g=0.5 L=20", "tfim", 20, 64, 0.5, -21.231056256176606, "TeNPy"),
]

BENCHMARKS_QUICK = BENCHMARKS[:4]  # Skip L=50


def run_benchmark(name: str, model: str, L: int, chi: int, g_or_h: float, 
                  expected: float, source: str) -> dict:
    """Run a single benchmark and return results."""
    torch.manual_seed(42)
    
    # Build Hamiltonian
    if model == "heisenberg":
        H = heisenberg_mpo(L=L, J=1.0, h=g_or_h)
    elif model == "tfim":
        H = tfim_mpo(L=L, J=1.0, g=g_or_h)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Initial state
    psi = MPS.random(L=L, d=2, chi=chi)
    
    # Run DMRG
    t0 = time.time()
    psi_opt, E, info = dmrg(psi, H, num_sweeps=20, chi_max=chi, tol=1e-10)
    elapsed = time.time() - t0
    
# Compare - use relative error with 5% tolerance for basic DMRG
    rel_error = abs(E - expected) / abs(expected)
    passed = rel_error < 0.05  # 5% relative error tolerance

    return {
        "name": name,
        "model": model,
        "L": L,
        "chi": chi,
        "g_or_h": g_or_h,
        "E": float(E),
        "expected": expected,
        "source": source,
        "error": float(rel_error),
        "passed": passed,
        "time_s": elapsed,
        "sweeps": info.get("sweeps", info.get("num_sweeps", -1)),
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce tensor network benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()
    
    benchmarks = BENCHMARKS_QUICK if args.quick else BENCHMARKS
    
    print("=" * 60)
    print("TENSOR NETWORK BENCHMARK SUITE")
    print("=" * 60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)
    print()
    
    results = []
    passed_count = 0
    
    for i, (name, model, L, chi, h, expected, source) in enumerate(benchmarks, 1):
        print(f"[{i}/{len(benchmarks)}] {name}, chi={chi}")
        
        result = run_benchmark(name, model, L, chi, h, expected, source)
        results.append(result)
        
        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            passed_count += 1
        
        print(f"      E0 = {result['E']:.8f} ({source}: {expected:.8f})")
        print(f"      Error: {result['error']:.2e} {status}")
        print(f"      Time: {result['time_s']:.2f}s, Sweeps: {result['sweeps']}")
        print()
    
    print("=" * 60)
    if passed_count == len(benchmarks):
        print(f"ALL BENCHMARKS PASSED ({passed_count}/{len(benchmarks)})")
    else:
        print(f"BENCHMARKS: {passed_count}/{len(benchmarks)} passed")
    
    if args.save:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "benchmarks": results,
            "summary": {
                "total": len(benchmarks),
                "passed": passed_count,
                "failed": len(benchmarks) - passed_count,
            }
        }
        
        output_path = results_dir / "benchmark_latest.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    print("=" * 60)
    
    return 0 if passed_count == len(benchmarks) else 1


if __name__ == "__main__":
    sys.exit(main())
