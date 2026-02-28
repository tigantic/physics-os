#!/usr/bin/env python
"""
Scaling tests for The Ontic Engine.

Measures performance scaling with problem size for:
1. DMRG: scaling with system size L and bond dimension chi
2. TEBD: scaling with L and chi
3. Euler1D: scaling with grid resolution N
4. Euler2D: scaling with grid size Nx × Ny

Usage:
    python tools/scripts/scaling_tests.py [--all|--dmrg|--cfd] [--output scaling_results.json]
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

# Add THIS project root to path (before any other ontic)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


@dataclass
class ScalingResult:
    """Result from a scaling measurement."""

    algorithm: str
    parameter: str
    values: List[int]
    times: List[float]
    memory_mb: List[float]
    metric: str  # e.g., "energy", "L2_error"
    metric_values: List[float]


def measure_dmrg_L_scaling(chi: int = 32, L_values: List[int] = None) -> ScalingResult:
    """Measure DMRG scaling with system size L."""
    from ontic import dmrg, heisenberg_mpo

    if L_values is None:
        L_values = [10, 20, 30, 40, 50]

    times = []
    memory = []
    energies = []

    print(f"DMRG L-scaling (chi={chi}):")
    for L in L_values:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        H = heisenberg_mpo(L, J=1.0, Jz=1.0)

        start = time.perf_counter()
        result = dmrg(H, chi_max=chi, num_sweeps=5)
        elapsed = time.perf_counter() - start

        # Memory estimate: 2 * L * chi^2 * d * 8 bytes (float64)
        mem_mb = 2 * L * chi * chi * 2 * 8 / 1e6

        times.append(elapsed)
        memory.append(mem_mb)
        energies.append(result.energy)

        print(f"  L={L:3d}: {elapsed:6.2f}s, E={result.energy:.8f}")

    return ScalingResult(
        algorithm="DMRG",
        parameter="L",
        values=L_values,
        times=times,
        memory_mb=memory,
        metric="energy",
        metric_values=energies,
    )


def measure_dmrg_chi_scaling(
    L: int = 20, chi_values: List[int] = None
) -> ScalingResult:
    """Measure DMRG scaling with bond dimension chi."""
    from ontic import dmrg, heisenberg_mpo

    if chi_values is None:
        chi_values = [8, 16, 32, 64, 128]

    times = []
    memory = []
    energies = []

    H = heisenberg_mpo(L, J=1.0, Jz=1.0)

    print(f"DMRG chi-scaling (L={L}):")
    for chi in chi_values:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start = time.perf_counter()
        result = dmrg(H, chi_max=chi, num_sweeps=5)
        elapsed = time.perf_counter() - start

        mem_mb = 2 * L * chi * chi * 2 * 8 / 1e6

        times.append(elapsed)
        memory.append(mem_mb)
        energies.append(result.energy)

        print(f"  chi={chi:3d}: {elapsed:6.2f}s, E={result.energy:.8f}")

    return ScalingResult(
        algorithm="DMRG",
        parameter="chi",
        values=chi_values,
        times=times,
        memory_mb=memory,
        metric="energy",
        metric_values=energies,
    )


def measure_euler1d_scaling(N_values: List[int] = None) -> ScalingResult:
    """Measure Euler1D scaling with grid resolution."""
    from ontic.cfd.euler_1d import Euler1D, EulerState

    if N_values is None:
        N_values = [100, 200, 400, 800, 1600]

    times = []
    memory = []
    final_rho_max = []

    print("Euler1D N-scaling:")
    for N in N_values:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        solver = Euler1D(N=N, x_min=0.0, x_max=1.0, gamma=1.4, cfl=0.4)

        # Sod shock tube IC
        x = solver.x_cell
        rho = torch.where(x < 0.5, torch.ones_like(x), 0.125 * torch.ones_like(x))
        u = torch.zeros_like(x)
        p = torch.where(x < 0.5, torch.ones_like(x), 0.1 * torch.ones_like(x))
        state = EulerState.from_primitive(rho, u, p, gamma=1.4)
        solver.set_initial_condition(state)

        # Run to t=0.2
        start = time.perf_counter()
        while solver.t < 0.2:
            solver.step()
        elapsed = time.perf_counter() - start

        mem_mb = 3 * N * 8 / 1e6  # 3 fields, float64

        times.append(elapsed)
        memory.append(mem_mb)
        final_rho_max.append(float(solver.state.rho.max()))

        print(f"  N={N:4d}: {elapsed:6.3f}s, rho_max={solver.state.rho.max():.4f}")

    return ScalingResult(
        algorithm="Euler1D",
        parameter="N",
        values=N_values,
        times=times,
        memory_mb=memory,
        metric="rho_max",
        metric_values=final_rho_max,
    )


def measure_euler2d_scaling(grid_sizes: List[tuple] = None) -> ScalingResult:
    """Measure Euler2D scaling with grid size."""
    from ontic.cfd.euler_2d import BCType, Euler2D, Euler2DState

    if grid_sizes is None:
        grid_sizes = [(50, 50), (100, 100), (150, 150), (200, 200)]

    times = []
    memory = []
    final_rho_max = []
    N_values = [Nx * Ny for Nx, Ny in grid_sizes]

    gamma = 1.4
    M_inf = 2.0
    Lx, Ly = 1.0, 1.0

    print("Euler2D grid-scaling:")
    for Nx, Ny in grid_sizes:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Create solver with correct API
        solver = Euler2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, gamma=gamma)

        # Set boundary conditions
        solver.bc_left = BCType.SUPERSONIC_INFLOW
        solver.bc_right = BCType.OUTFLOW
        solver.bc_bottom = BCType.REFLECTIVE
        solver.bc_top = BCType.OUTFLOW

        # Initialize uniform supersonic flow
        rho = torch.ones(Ny, Nx, dtype=torch.float64)
        u = M_inf * torch.ones(Ny, Nx, dtype=torch.float64)
        v = torch.zeros(Ny, Nx, dtype=torch.float64)
        p = (1.0 / gamma) * torch.ones(Ny, Nx, dtype=torch.float64)

        state = Euler2DState(rho, u, v, p)
        solver.state = state
        solver.inflow_state = state

        # Run 50 steps
        start = time.perf_counter()
        for _ in range(50):
            solver.step(cfl=0.4)
        elapsed = time.perf_counter() - start

        mem_mb = 4 * Nx * Ny * 8 / 1e6  # 4 fields, float64

        times.append(elapsed)
        memory.append(mem_mb)
        final_rho_max.append(float(solver.state.rho.max()))

        print(f"  {Nx}x{Ny}: {elapsed:6.3f}s, rho_max={solver.state.rho.max():.4f}")

    return ScalingResult(
        algorithm="Euler2D",
        parameter="grid_size",
        values=N_values,
        times=times,
        memory_mb=memory,
        metric="rho_max",
        metric_values=final_rho_max,
    )


def print_complexity_analysis(results: List[ScalingResult]):
    """Analyze and print complexity from scaling results."""
    import math

    print("\n" + "=" * 60)
    print("COMPLEXITY ANALYSIS")
    print("=" * 60)

    for r in results:
        if len(r.values) < 2:
            continue

        # Estimate exponent: t ~ N^alpha => log(t) = alpha * log(N) + c
        log_N = [math.log(v) for v in r.values]
        log_t = [math.log(t) for t in r.times if t > 0]

        if len(log_t) < 2:
            continue

        # Simple linear regression
        n = len(log_t)
        sum_x = sum(log_N[:n])
        sum_y = sum(log_t)
        sum_xy = sum(log_N[i] * log_t[i] for i in range(n))
        sum_x2 = sum(x * x for x in log_N[:n])

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) > 1e-10:
            alpha = (n * sum_xy - sum_x * sum_y) / denom
            print(f"{r.algorithm} ({r.parameter}): t ~ {r.parameter}^{alpha:.2f}")


def main():
    parser = argparse.ArgumentParser(description="The Ontic Engine scaling tests")
    parser.add_argument("--dmrg", action="store_true", help="Run DMRG scaling tests")
    parser.add_argument("--cfd", action="store_true", help="Run CFD scaling tests")
    parser.add_argument("--all", action="store_true", help="Run all scaling tests")
    parser.add_argument(
        "--output", type=str, default="scaling_results.json", help="Output JSON file"
    )
    args = parser.parse_args()

    if not any([args.dmrg, args.cfd, args.all]):
        args.all = True

    results = []

    if args.dmrg or args.all:
        print("\n" + "=" * 40)
        print("DMRG SCALING TESTS")
        print("=" * 40)
        results.append(measure_dmrg_L_scaling())
        results.append(measure_dmrg_chi_scaling())

    if args.cfd or args.all:
        print("\n" + "=" * 40)
        print("CFD SCALING TESTS")
        print("=" * 40)
        results.append(measure_euler1d_scaling())
        results.append(measure_euler2d_scaling())

    print_complexity_analysis(results)

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
