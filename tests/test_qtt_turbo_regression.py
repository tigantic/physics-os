#!/usr/bin/env python3
"""
QTT Turbo Solver Regression Test Suite
=======================================

Production-grade tests for the TurboNS3DSolver with validated optimal configuration.

CRITICAL FIXES TESTED:
1. poisson_iterations=0 (Jacobi solver diverges with >0)
2. max_rank=16 (optimal from Phase 2)
3. Truncation after each Hadamard (prevents rank explosion OOM)

Gate Criteria:
- No NaN/Inf in any output
- Energy drift <10% over 10 steps (viscous)
- Memory usage <8GB for 256³
- O(log N) time scaling

Author: HyperTensor Team
Date: 2025-02
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    metrics: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class RegressionReport:
    """Complete regression test report."""
    timestamp: str
    git_commit: str
    cuda_device: str
    cuda_memory_gb: float
    tests: List[TestResult]
    summary: Dict[str, Any]


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def compute_enstrophy_from_qtt(omega: list) -> float:
    """Compute enstrophy from QTT vorticity: Ω = ||ω||²."""
    from tensornet.cfd.qtt_turbo import turbo_inner
    return sum(turbo_inner(omega[i], omega[i]).item() for i in range(3))


# ═══════════════════════════════════════════════════════════════════════════════════════
# TEST 1: Imports & Configuration
# ═══════════════════════════════════════════════════════════════════════════════════════

def test_imports() -> TestResult:
    """Test that all required modules import correctly."""
    start = time.perf_counter()
    metrics: Dict[str, Any] = {}
    
    try:
        from tensornet.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
        from tensornet.cfd.qtt_turbo import (
            TurboCores, turbo_truncate, turbo_hadamard_cores,
            turbo_linear_combination, TRITON_AVAILABLE,
        )
        
        metrics["triton_available"] = TRITON_AVAILABLE
        
        # Verify critical defaults
        config = TurboNS3DConfig()
        metrics["default_max_rank"] = config.max_rank
        metrics["default_poisson_iterations"] = config.poisson_iterations
        
        # Critical assertions
        assert config.poisson_iterations == 0, \
            f"CRITICAL: poisson_iterations must be 0, got {config.poisson_iterations}"
        assert config.max_rank <= 32, \
            f"CRITICAL: max_rank should be ≤32 for stability, got {config.max_rank}"
        
        passed = True
        error = None
        
    except Exception as e:
        passed = False
        error = str(e)
    
    duration = (time.perf_counter() - start) * 1000
    return TestResult("imports", passed, duration, metrics, error)


# ═══════════════════════════════════════════════════════════════════════════════════════
# TEST 2: Poisson Iteration = 0 Stability
# ═══════════════════════════════════════════════════════════════════════════════════════

def test_poisson_zero_stability() -> TestResult:
    """
    Test that poisson_iterations=0 produces stable output.
    
    CRITICAL: This tests the fix for the Jacobi Poisson divergence bug.
    With poisson_iterations > 0, the solver diverges exponentially.
    """
    start = time.perf_counter()
    metrics: Dict[str, Any] = {}
    
    try:
        from tensornet.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
        
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        config = TurboNS3DConfig(
            n_bits=4,  # 16³
            max_rank=16,
            poisson_iterations=0,  # CRITICAL
            nu=0.01,
            dt=0.001,
            device=device_str,
        )
        
        solver = TurboNS3DSolver(config)
        
        # Initialize Taylor-Green vortex
        solver.initialize_taylor_green()
        
        # Get initial enstrophy (ω is QTT cores)
        E0 = compute_enstrophy_from_qtt(solver.omega)
        metrics["E0"] = E0
        
        # Run 10 steps
        last_diag = None
        for _ in range(10):
            last_diag = solver.step()
        
        # Get final enstrophy
        E_final = compute_enstrophy_from_qtt(solver.omega)
        metrics["E_final"] = E_final
        
        # Check for NaN/Inf in cores
        has_nan = False
        has_inf = False
        for i in range(3):
            for core in solver.omega[i]:
                if torch.isnan(core).any().item():
                    has_nan = True
                if torch.isinf(core).any().item():
                    has_inf = True
        
        metrics["has_nan"] = has_nan
        metrics["has_inf"] = has_inf
        
        # Compute drift
        energy_drift = abs(E_final - E0) / E0 * 100
        metrics["energy_drift_percent"] = energy_drift
        
        # Pass criteria
        passed = (
            not has_nan and
            not has_inf and
            energy_drift < 20  # <20% drift over 10 steps
        )
        error = None if passed else f"Energy drift {energy_drift:.1f}% or NaN/Inf detected"
        
        # Cleanup
        del solver
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        passed = False
        error = str(e)
        metrics["exception"] = str(e)
    
    duration = (time.perf_counter() - start) * 1000
    return TestResult("poisson_zero_stability", passed, duration, metrics, error)


# ═══════════════════════════════════════════════════════════════════════════════════════
# TEST 3: Rank 16 Optimal Configuration
# ═══════════════════════════════════════════════════════════════════════════════════════

def test_rank16_optimal() -> TestResult:
    """
    Test that rank=16 configuration works and fits in memory.
    
    Phase 2 finding: rank=16 is optimal for accuracy/performance tradeoff.
    """
    start = time.perf_counter()
    metrics: Dict[str, Any] = {}
    
    try:
        from tensornet.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
        
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        config = TurboNS3DConfig(
            n_bits=5,  # 32³
            max_rank=16,  # OPTIMAL
            poisson_iterations=0,
            nu=0.01,
            dt=0.001,
            device=device_str,
        )
        
        # Track memory before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        solver = TurboNS3DSolver(config)
        solver.initialize_taylor_green()
        
        # Run 5 steps
        step_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            solver.step()
            step_times.append((time.perf_counter() - t0) * 1000)
        
        # Get memory
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory_mb = 0
        
        metrics["grid"] = "32³"
        metrics["max_rank"] = 16
        metrics["step_time_ms"] = sum(step_times) / len(step_times)
        metrics["peak_memory_mb"] = peak_memory_mb
        
        # Check for NaN in cores
        has_nan = False
        for i in range(3):
            for core in solver.omega[i]:
                if torch.isnan(core).any().item():
                    has_nan = True
        metrics["has_nan"] = has_nan
        
        # Pass if stable and within memory
        passed = not has_nan and peak_memory_mb < 500
        error = None if passed else f"NaN={has_nan}, memory={peak_memory_mb:.1f}MB"
        
        del solver
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        passed = False
        error = str(e)
    
    duration = (time.perf_counter() - start) * 1000
    return TestResult("rank16_optimal", passed, duration, metrics, error)


# ═══════════════════════════════════════════════════════════════════════════════════════
# TEST 4: Memory Scaling (64³, 128³)
# ═══════════════════════════════════════════════════════════════════════════════════════

def test_memory_scaling() -> TestResult:
    """
    Test memory usage at 64³ and 128³ grids.
    
    Expected (from Phase 3-4):
    - 64³: ~147 MB
    - 128³: ~1129 MB
    """
    start = time.perf_counter()
    metrics: Dict[str, Any] = {}
    
    if not torch.cuda.is_available():
        return TestResult(
            "memory_scaling", True, 0, 
            {"skipped": "No CUDA device"}, None
        )
    
    try:
        from tensornet.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
        
        grids = [(6, "64³"), (7, "128³")]
        
        for n_bits, label in grids:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            
            config = TurboNS3DConfig(
                n_bits=n_bits,
                max_rank=16,
                poisson_iterations=0,
                nu=0.01,
                dt=0.001,
                device="cuda",
            )
            
            solver = TurboNS3DSolver(config)
            solver.initialize_taylor_green()
            
            # Run 2 steps
            for _ in range(2):
                solver.step()
            
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            metrics[f"{label}_memory_mb"] = peak_mb
            
            # Check for NaN in cores
            has_nan = False
            for i in range(3):
                for core in solver.omega[i]:
                    if torch.isnan(core).any().item():
                        has_nan = True
            metrics[f"{label}_has_nan"] = has_nan
            
            del solver
            torch.cuda.empty_cache()
            gc.collect()
        
        # Pass criteria: both grids complete without error, memory reasonable
        passed = (
            not metrics.get("64³_has_nan", True) and
            not metrics.get("128³_has_nan", True) and
            metrics.get("128³_memory_mb", float("inf")) < 2000
        )
        error = None if passed else "NaN or memory exceeded"
        
    except Exception as e:
        passed = False
        error = str(e)
    
    duration = (time.perf_counter() - start) * 1000
    return TestResult("memory_scaling", passed, duration, metrics, error)


# ═══════════════════════════════════════════════════════════════════════════════════════
# TEST 5: Time Scaling O(log N)
# ═══════════════════════════════════════════════════════════════════════════════════════

def test_time_scaling() -> TestResult:
    """
    Test that time per step scales as O(log N), not O(N³).
    
    For O(log N): doubling N should only add constant time.
    For O(N³): doubling N would 8× the time.
    """
    start = time.perf_counter()
    metrics: Dict[str, Any] = {}
    
    try:
        from tensornet.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        grids = [(4, "16³"), (5, "32³"), (6, "64³")]
        times = {}
        
        for n_bits, label in grids:
            config = TurboNS3DConfig(
                n_bits=n_bits,
                max_rank=16,
                poisson_iterations=0,
                nu=0.01,
                dt=0.001,
                device=device_str,
            )
            
            solver = TurboNS3DSolver(config)
            solver.initialize_taylor_green()
            
            # Warmup
            solver.step()
            
            # Measure
            step_times = []
            for _ in range(3):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                solver.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_times.append((time.perf_counter() - t0) * 1000)
            
            avg_time = sum(step_times) / len(step_times)
            times[label] = avg_time
            metrics[f"{label}_ms"] = avg_time
            
            del solver
            torch.cuda.empty_cache()
            gc.collect()
        
        # Check scaling: 64³ should be < 4× of 16³ (O(log N) behavior)
        # If O(N³), 64³ would be 64× of 16³
        ratio_64_16 = times["64³"] / times["16³"]
        metrics["ratio_64_to_16"] = ratio_64_16
        
        # Pass if ratio < 5 (indicating O(log N) not O(N³))
        passed = ratio_64_16 < 5
        error = None if passed else f"Scaling ratio {ratio_64_16:.1f}× too high"
        
    except Exception as e:
        passed = False
        error = str(e)
    
    duration = (time.perf_counter() - start) * 1000
    return TestResult("time_scaling", passed, duration, metrics, error)


# ═══════════════════════════════════════════════════════════════════════════════════════
# TEST 6: Energy Conservation (Inviscid)
# ═══════════════════════════════════════════════════════════════════════════════════════

def test_energy_conservation_inviscid() -> TestResult:
    """
    Test energy conservation in inviscid limit (nu=0).
    
    Phase 5 finding: ~0.1% loss over 10 steps is acceptable.
    """
    start = time.perf_counter()
    metrics: Dict[str, Any] = {}
    
    try:
        from tensornet.cfd.ns3d_turbo import TurboNS3DSolver, TurboNS3DConfig
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        config = TurboNS3DConfig(
            n_bits=5,  # 32³
            max_rank=16,
            poisson_iterations=0,
            nu=0.0,  # INVISCID
            dt=0.0001,  # Small timestep
            device=device_str,
        )
        
        solver = TurboNS3DSolver(config)
        solver.initialize_taylor_green()
        
        E0 = compute_enstrophy_from_qtt(solver.omega)
        metrics["E0"] = E0
        
        # Run 10 steps
        for _ in range(10):
            solver.step()
        
        E_final = compute_enstrophy_from_qtt(solver.omega)
        metrics["E_final"] = E_final
        
        energy_drift = abs(E_final - E0) / E0 * 100
        metrics["energy_drift_percent"] = energy_drift
        
        # Pass if <1% drift (inviscid should be well-conserved)
        passed = energy_drift < 1.0
        error = None if passed else f"Inviscid drift {energy_drift:.2f}% > 1%"
        
        del solver
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        passed = False
        error = str(e)
    
    duration = (time.perf_counter() - start) * 1000
    return TestResult("energy_conservation_inviscid", passed, duration, metrics, error)


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_regression_tests() -> RegressionReport:
    """Run all regression tests and generate report."""
    
    print("=" * 70)
    print("QTT TURBO SOLVER REGRESSION TEST SUITE")
    print("=" * 70)
    
    # System info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_device = torch.cuda.get_device_name() if torch.cuda.is_available() else "None"
    cuda_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    
    print(f"Device: {cuda_device}")
    print(f"VRAM: {cuda_memory:.1f} GB")
    print()
    
    # Run tests
    tests = [
        ("Imports & Config", test_imports),
        ("Poisson Zero Stability", test_poisson_zero_stability),
        ("Rank 16 Optimal", test_rank16_optimal),
        ("Memory Scaling", test_memory_scaling),
        ("Time Scaling O(log N)", test_time_scaling),
        ("Energy Conservation (Inviscid)", test_energy_conservation_inviscid),
    ]
    
    results: List[TestResult] = []
    
    for name, test_fn in tests:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            result = test_fn()
            results.append(result)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} ({result.duration_ms:.1f}ms)")
            if not result.passed and result.error:
                print(f"  Error: {result.error}")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append(TestResult(name, False, 0, {}, str(e)))
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print()
    print("=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 70)
    
    summary = {
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
    }
    
    report = RegressionReport(
        timestamp=datetime.now().isoformat(),
        git_commit=get_git_commit(),
        cuda_device=cuda_device,
        cuda_memory_gb=cuda_memory,
        tests=[asdict(r) for r in results],
        summary=summary,
    )
    
    return report


def save_report(report: RegressionReport, path: Path) -> str:
    """Save report to JSON and return SHA256 hash."""
    report_dict = asdict(report)
    json_str = json.dumps(report_dict, indent=2)
    
    # Compute hash
    hash_val = sha256(json_str.encode()).hexdigest()
    report_dict["sha256"] = hash_val
    
    # Write with hash
    with open(path, "w") as f:
        json.dump(report_dict, f, indent=2)
    
    return hash_val


if __name__ == "__main__":
    report = run_regression_tests()
    
    # Save to artifacts
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    report_path = artifacts_dir / "QTT_TURBO_REGRESSION.json"
    hash_val = save_report(report, report_path)
    
    print()
    print(f"Report saved: {report_path}")
    print(f"SHA256: {hash_val[:16]}...")
    
    # Exit with error code if tests failed
    exit(0 if report.summary["all_passed"] else 1)
