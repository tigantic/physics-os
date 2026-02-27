#!/usr/bin/env python
"""
Performance Profiler for Flagship Pipeline
===========================================

Profiles the flagship pipeline to identify bottlenecks and generate
optimization targets for GPU acceleration.

Usage:
    python tools/scripts/profile_flagship.py

Output:
    - Console summary of hotspots
    - profile_results/flagship_profile.prof (cProfile data)
    - profile_results/flagship_report.txt (human-readable report)
"""

import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


def profile_mps_operations():
    """Profile MPS creation and manipulation."""
    from tensornet.cfd.tt_cfd import MPSState
    from tensornet.core.mps import MPS

    results = {}

    # Test various sizes
    sizes = [64, 128, 256, 512]
    chi_values = [8, 16, 32, 64]

    for N in sizes:
        for chi in chi_values:
            if chi > N:
                continue

            # Profile MPS creation
            start = time.perf_counter()
            for _ in range(10):
                rho = torch.ones(N, dtype=torch.float64)
                u = torch.zeros(N, dtype=torch.float64)
                p = torch.ones(N, dtype=torch.float64)
                state = MPSState.from_primitive(rho, u, p, chi_max=chi)
            creation_time = (time.perf_counter() - start) / 10

            # Profile round-trip
            start = time.perf_counter()
            for _ in range(10):
                rho2, u2, p2 = state.to_primitive()
            roundtrip_time = (time.perf_counter() - start) / 10

            results[f"N={N},chi={chi}"] = {
                "creation_ms": creation_time * 1000,
                "roundtrip_ms": roundtrip_time * 1000,
            }

    return results


def profile_weno_operations():
    """Profile WENO reconstruction."""
    from tensornet.cfd.weno import weno5_z

    results = {}
    sizes = [64, 128, 256, 512, 1024]

    for N in sizes:
        # Create test data as torch tensor
        x = torch.linspace(0, 1, N, dtype=torch.float64)
        f = torch.sin(2 * torch.pi * x)

        # Warmup
        for _ in range(3):
            _ = weno5_z(f, 0.01)

        # Profile
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = weno5_z(f, 0.01)
        elapsed = time.perf_counter() - start

        results[f"N={N}"] = {
            "time_ms": (elapsed / iterations) * 1000,
            "points_per_sec": N * iterations / elapsed,
        }

    return results


def profile_euler_solver():
    """Profile Euler 1D solver."""
    from tensornet.cfd.euler_1d import Euler1D, EulerState

    results = {}
    sizes = [64, 128, 256, 512]
    gamma = 1.4

    for N in sizes:
        solver = Euler1D(N=N, x_min=0.0, x_max=1.0, gamma=gamma, cfl=0.5)

        # Create Sod initial condition
        def make_sod_state():
            x = torch.linspace(0, 1, N, dtype=torch.float64)
            rho = torch.where(
                x < 0.5,
                torch.ones(N, dtype=torch.float64),
                torch.full((N,), 0.125, dtype=torch.float64),
            )
            u = torch.zeros(N, dtype=torch.float64)
            p = torch.where(
                x < 0.5,
                torch.ones(N, dtype=torch.float64),
                torch.full((N,), 0.1, dtype=torch.float64),
            )
            rho_u = rho * u
            E = p / (gamma - 1) + 0.5 * rho * u**2
            return EulerState(rho=rho, rho_u=rho_u, E=E, gamma=gamma)

        # Warmup
        for _ in range(3):
            solver.set_initial_condition(make_sod_state())
            solver.step()

        # Profile stepping
        solver.set_initial_condition(make_sod_state())
        start = time.perf_counter()
        steps = 100
        for _ in range(steps):
            solver.step()
        step_time = (time.perf_counter() - start) / steps

        results[f"N={N}"] = {
            "step_ms": step_time * 1000,
            "cells_per_sec": N / step_time,
        }

    return results


def profile_svd_operations():
    """Profile SVD (core TT operation)."""
    results = {}

    shapes = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]

    for m, n in shapes:
        A = torch.randn(m, n, dtype=torch.float64)

        # Warmup
        for _ in range(3):
            _ = torch.linalg.svd(A, full_matrices=False)

        # Profile
        start = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            _ = torch.linalg.svd(A, full_matrices=False)
        elapsed = time.perf_counter() - start

        results[f"{m}x{n}"] = {
            "time_ms": (elapsed / iterations) * 1000,
            "gflops": (4 * m * n * min(m, n)) / (elapsed / iterations) / 1e9,
        }

    return results


def profile_full_pipeline():
    """Profile the complete flagship pipeline by running it."""
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # Time the full pipeline run
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "demos" / "flagship_pipeline.py")],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    total_time = time.perf_counter() - start

    # Parse timing from output if available
    results = {
        "total_pipeline_ms": total_time * 1000,
    }

    # Extract step timings from output (rough parsing)
    if result.returncode == 0:
        output = result.stdout
        results["success"] = True
    else:
        results["success"] = False
        results["error"] = result.stderr[:500]

    return results


def generate_report(all_results: Dict) -> str:
    """Generate human-readable performance report."""
    lines = []
    lines.append("=" * 70)
    lines.append(" HYPERTENSOR PERFORMANCE PROFILE")
    lines.append("=" * 70)
    lines.append("")

    # Pipeline summary
    if "pipeline" in all_results:
        lines.append("FLAGSHIP PIPELINE")
        lines.append("-" * 40)
        pipeline = all_results["pipeline"]
        lines.append(f"  Total time: {pipeline['total_pipeline_ms']:.2f} ms")
        lines.append(f"  Success: {pipeline.get('success', False)}")
        lines.append("")

    # SVD performance
    if "svd" in all_results:
        lines.append("SVD PERFORMANCE (core TT operation)")
        lines.append("-" * 40)
        for shape, metrics in all_results["svd"].items():
            lines.append(
                f"  {shape:12s}: {metrics['time_ms']:8.3f} ms, {metrics['gflops']:.2f} GFLOPS"
            )
        lines.append("")

    # WENO performance
    if "weno" in all_results:
        lines.append("WENO-5 PERFORMANCE")
        lines.append("-" * 40)
        for size, metrics in all_results["weno"].items():
            lines.append(
                f"  {size:12s}: {metrics['time_ms']:8.3f} ms, {metrics['points_per_sec']/1e6:.2f}M pts/s"
            )
        lines.append("")

    # Euler solver
    if "euler" in all_results:
        lines.append("EULER 1D SOLVER PERFORMANCE")
        lines.append("-" * 40)
        for size, metrics in all_results["euler"].items():
            lines.append(
                f"  {size:12s}: {metrics['step_ms']:8.3f} ms/step, {metrics['cells_per_sec']/1e6:.2f}M cells/s"
            )
        lines.append("")

    # MPS operations
    if "mps" in all_results:
        lines.append("MPS STATE OPERATIONS")
        lines.append("-" * 40)
        for config, metrics in sorted(all_results["mps"].items()):
            lines.append(
                f"  {config:20s}: create={metrics['creation_ms']:.2f}ms, roundtrip={metrics['roundtrip_ms']:.2f}ms"
            )
        lines.append("")

    # Optimization recommendations
    lines.append("GPU ACCELERATION CANDIDATES")
    lines.append("-" * 40)
    lines.append("  1. SVD operations (torch.linalg.svd -> cuSOLVER)")
    lines.append("  2. WENO reconstruction (vectorized -> CUDA kernel)")
    lines.append("  3. Flux computation (batched tensor ops)")
    lines.append("  4. MPS contractions (cuTensor library)")
    lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Run all profiling benchmarks."""
    print("=" * 70)
    print(" HYPERTENSOR PERFORMANCE PROFILER")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = PROJECT_ROOT / "profile_results"
    output_dir.mkdir(exist_ok=True)

    all_results = {}

    # Profile individual components
    print("Profiling SVD operations...")
    all_results["svd"] = profile_svd_operations()

    print("Profiling WENO operations...")
    all_results["weno"] = profile_weno_operations()

    print("Profiling Euler solver...")
    all_results["euler"] = profile_euler_solver()

    print("Profiling MPS operations...")
    all_results["mps"] = profile_mps_operations()

    print("Profiling full pipeline...")
    all_results["pipeline"] = profile_full_pipeline()

    # Generate report
    report = generate_report(all_results)
    print()
    print(report)

    # Save report
    report_path = output_dir / "flagship_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Full cProfile of pipeline
    print("\nGenerating detailed cProfile...")
    profiler = cProfile.Profile()
    profiler.enable()
    _ = profile_full_pipeline()
    profiler.disable()

    # Save cProfile data
    profile_path = output_dir / "flagship_profile.prof"
    profiler.dump_stats(str(profile_path))
    print(f"cProfile data saved to: {profile_path}")

    # Print top functions
    print("\nTop 20 functions by cumulative time:")
    print("-" * 70)
    stats = pstats.Stats(profiler, stream=sys.stdout)
    stats.strip_dirs().sort_stats("cumulative").print_stats(20)

    return 0


if __name__ == "__main__":
    sys.exit(main())
