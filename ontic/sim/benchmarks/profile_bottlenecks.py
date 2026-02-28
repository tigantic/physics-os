#!/usr/bin/env python3
"""
Phase 2A-1: Baseline Profiling for CUDA Acceleration
=====================================================

Scientifically identifies the slowest functions in the ontic
simulation loop to guide CUDA kernel development.

Measures:
- Advection computation
- Pressure solve (Poisson)
- Tensor contraction operations
- Gradient computations

Exit Gate: Log file proving which function is the bottleneck.

Usage:
    python -m ontic.benchmarks.profile_bottlenecks
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

# Check for CUDA availability
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    import torch.cuda as cuda


@dataclass
class TimingResult:
    """Result from timing a function."""

    operation: str
    avg_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    iterations: int


def time_function(
    fn, iterations: int = 100, warmup: int = 10, name: str = "Operation"
) -> TimingResult:
    """
    Time a function with warmup and statistics.

    Args:
        fn: Callable to time (returns None)
        iterations: Number of timed iterations
        warmup: Number of warmup iterations
        name: Name for reporting

    Returns:
        TimingResult with statistics
    """
    # Warmup
    for _ in range(warmup):
        fn()
        if HAS_CUDA:
            cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iterations):
        if HAS_CUDA:
            cuda.synchronize()
        start = time.perf_counter()

        fn()

        if HAS_CUDA:
            cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # ms

    import statistics

    return TimingResult(
        operation=name,
        avg_ms=statistics.mean(times),
        min_ms=min(times),
        max_ms=max(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        iterations=iterations,
    )


def profile_advection_cpu(grid_size: int = 512) -> TimingResult:
    """Profile CPU advection (Semi-Lagrangian style)."""
    # Setup
    density = torch.rand((grid_size, grid_size), dtype=torch.float32)
    u_vel = torch.rand((grid_size, grid_size), dtype=torch.float32) * 2.0 - 1.0
    v_vel = torch.rand((grid_size, grid_size), dtype=torch.float32) * 2.0 - 1.0
    dt = 0.01

    def advect_step():
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing="ij",
        )

        # Backtrace positions
        src_x = x_coords - u_vel * dt
        src_y = y_coords - v_vel * dt

        # Clamp to boundaries
        src_x = torch.clamp(src_x, 0, grid_size - 1.001)
        src_y = torch.clamp(src_y, 0, grid_size - 1.001)

        # Integer and fractional parts for bilinear interpolation
        x0 = src_x.long()
        y0 = src_y.long()
        x1 = torch.clamp(x0 + 1, 0, grid_size - 1)
        y1 = torch.clamp(y0 + 1, 0, grid_size - 1)

        dx = src_x - x0.float()
        dy = src_y - y0.float()

        # Bilinear interpolation
        v00 = density[y0, x0]
        v10 = density[y0, x1]
        v01 = density[y1, x0]
        v11 = density[y1, x1]

        top = v00 * (1 - dx) + v10 * dx
        bot = v01 * (1 - dx) + v11 * dx

        return top * (1 - dy) + bot * dy

    return time_function(advect_step, name="Advection (CPU)", iterations=100)


def profile_advection_gpu(grid_size: int = 512) -> TimingResult:
    """Profile GPU advection (PyTorch CUDA, not custom kernel)."""
    if not HAS_CUDA:
        return TimingResult("Advection (GPU)", -1, -1, -1, 0, 0)

    device = torch.device("cuda")
    density = torch.rand((grid_size, grid_size), dtype=torch.float32, device=device)
    u_vel = (
        torch.rand((grid_size, grid_size), dtype=torch.float32, device=device) * 2.0
        - 1.0
    )
    v_vel = (
        torch.rand((grid_size, grid_size), dtype=torch.float32, device=device) * 2.0
        - 1.0
    )
    dt = 0.01

    # Pre-create coordinate grids on GPU
    y_coords, x_coords = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32, device=device),
        torch.arange(grid_size, dtype=torch.float32, device=device),
        indexing="ij",
    )

    def advect_step():
        # Backtrace positions
        src_x = x_coords - u_vel * dt
        src_y = y_coords - v_vel * dt

        # Clamp to boundaries
        src_x = torch.clamp(src_x, 0, grid_size - 1.001)
        src_y = torch.clamp(src_y, 0, grid_size - 1.001)

        # Integer and fractional parts
        x0 = src_x.long()
        y0 = src_y.long()
        x1 = torch.clamp(x0 + 1, 0, grid_size - 1)
        y1 = torch.clamp(y0 + 1, 0, grid_size - 1)

        dx = src_x - x0.float()
        dy = src_y - y0.float()

        # Bilinear interpolation
        v00 = density[y0, x0]
        v10 = density[y0, x1]
        v01 = density[y1, x0]
        v11 = density[y1, x1]

        top = v00 * (1 - dx) + v10 * dx
        bot = v01 * (1 - dx) + v11 * dx

        return top * (1 - dy) + bot * dy

    return time_function(advect_step, name="Advection (GPU PyTorch)", iterations=100)


def profile_gradient_cpu(grid_size: int = 512) -> TimingResult:
    """Profile CPU gradient computation."""
    field = torch.rand((grid_size, grid_size), dtype=torch.float32)
    dx = dy = 1.0 / grid_size

    def gradient_step():
        # Central differences
        du_dx = (torch.roll(field, -1, dims=1) - torch.roll(field, 1, dims=1)) / (
            2 * dx
        )
        du_dy = (torch.roll(field, -1, dims=0) - torch.roll(field, 1, dims=0)) / (
            2 * dy
        )
        return du_dx, du_dy

    return time_function(gradient_step, name="Gradient (CPU)", iterations=100)


def profile_laplacian_cpu(grid_size: int = 512) -> TimingResult:
    """Profile CPU Laplacian (diffusion) computation."""
    field = torch.rand((grid_size, grid_size), dtype=torch.float32)
    dx = dy = 1.0 / grid_size

    def laplacian_step():
        # 5-point stencil
        lap = (
            torch.roll(field, -1, dims=1) - 2 * field + torch.roll(field, 1, dims=1)
        ) / (dx**2) + (
            torch.roll(field, -1, dims=0) - 2 * field + torch.roll(field, 1, dims=0)
        ) / (
            dy**2
        )
        return lap

    return time_function(laplacian_step, name="Laplacian (CPU)", iterations=100)


def profile_fft_poisson_cpu(grid_size: int = 512) -> TimingResult:
    """Profile CPU FFT-based Poisson solve."""
    rhs = torch.rand((grid_size, grid_size), dtype=torch.float32)

    # Precompute wavenumbers
    kx = torch.fft.fftfreq(grid_size, d=1.0 / grid_size) * 2 * torch.pi
    ky = torch.fft.fftfreq(grid_size, d=1.0 / grid_size) * 2 * torch.pi
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # Avoid division by zero

    def poisson_step():
        rhs_hat = torch.fft.fft2(rhs)
        phi_hat = rhs_hat / (-K2)
        phi_hat[0, 0] = 0  # Zero mean
        phi = torch.fft.ifft2(phi_hat).real
        return phi

    return time_function(poisson_step, name="Poisson FFT (CPU)", iterations=100)


def profile_tensor_contraction_cpu(
    bond_dim: int = 32, grid_size: int = 512
) -> TimingResult:
    """Profile CPU tensor contraction (simulating TT/QTT operations)."""
    # Simulate contracting two tensor train cores
    # Core shape: (rank_left, phys_dim, rank_right)
    core_a = torch.rand((1, grid_size, bond_dim), dtype=torch.float32)
    core_b = torch.rand((bond_dim, grid_size, bond_dim), dtype=torch.float32)
    core_c = torch.rand((bond_dim, grid_size, 1), dtype=torch.float32)

    def contract_step():
        # Contract A with B: sum over middle index
        # (1, n, r) @ (r, n, r) -> (1, n, n, r)
        # Then contract with C
        # This is similar to MPS evaluation
        result = torch.einsum("lir,rjs->lijs", core_a, core_b)
        result = torch.einsum("lijs,sjk->lik", result, core_c)
        return result

    return time_function(contract_step, name="Tensor Contraction (CPU)", iterations=100)


def profile_full_ns_step(grid_size: int = 256) -> TimingResult:
    """Profile a full Navier-Stokes step with advection, diffusion, and projection."""
    # Setup velocity field
    u = torch.rand((grid_size, grid_size), dtype=torch.float32) * 0.1
    v = torch.rand((grid_size, grid_size), dtype=torch.float32) * 0.1
    dx = dy = 1.0 / grid_size
    dt = 0.001
    nu = 0.01  # Viscosity

    # Precompute FFT wavenumbers for Poisson
    kx = torch.fft.fftfreq(grid_size, d=1.0 / grid_size) * 2 * torch.pi
    ky = torch.fft.fftfreq(grid_size, d=1.0 / grid_size) * 2 * torch.pi
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0

    def ns_step():
        nonlocal u, v

        # 1. Compute advection: (u·∇)u
        du_dx = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dx)
        du_dy = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2 * dy)
        dv_dx = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * dx)
        dv_dy = (torch.roll(v, -1, dims=0) - torch.roll(v, 1, dims=0)) / (2 * dy)

        adv_u = u * du_dx + v * du_dy
        adv_v = u * dv_dx + v * dv_dy

        # 2. Compute diffusion: ν∇²u
        lap_u = (torch.roll(u, -1, dims=1) - 2 * u + torch.roll(u, 1, dims=1)) / (
            dx**2
        ) + (torch.roll(u, -1, dims=0) - 2 * u + torch.roll(u, 1, dims=0)) / (dy**2)
        lap_v = (torch.roll(v, -1, dims=1) - 2 * v + torch.roll(v, 1, dims=1)) / (
            dx**2
        ) + (torch.roll(v, -1, dims=0) - 2 * v + torch.roll(v, 1, dims=0)) / (dy**2)

        # 3. Predictor step
        u_star = u - dt * adv_u + dt * nu * lap_u
        v_star = v - dt * adv_v + dt * nu * lap_v

        # 4. Compute divergence
        div = (torch.roll(u_star, -1, dims=1) - torch.roll(u_star, 1, dims=1)) / (
            2 * dx
        ) + (torch.roll(v_star, -1, dims=0) - torch.roll(v_star, 1, dims=0)) / (2 * dy)

        # 5. Solve Poisson for pressure correction
        div_hat = torch.fft.fft2(div)
        phi_hat = div_hat / (-K2) * dt
        phi_hat[0, 0] = 0
        phi = torch.fft.ifft2(phi_hat).real

        # 6. Project to divergence-free
        dphi_dx = (torch.roll(phi, -1, dims=1) - torch.roll(phi, 1, dims=1)) / (2 * dx)
        dphi_dy = (torch.roll(phi, -1, dims=0) - torch.roll(phi, 1, dims=0)) / (2 * dy)

        u = u_star - dphi_dx
        v = v_star - dphi_dy

        return u, v

    return time_function(ns_step, name="Full NS Step (CPU)", iterations=50)


def run_profiling(output_file: str = "profile_results.json"):
    """Run all profiling benchmarks and save results."""
    print("=" * 70)
    print("  PHASE 2A-1: BOTTLENECK PROFILING")
    print("  Target: Identify slowest operations for CUDA acceleration")
    print("=" * 70)
    print()

    # System info
    print("System Configuration:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {HAS_CUDA}")
    if HAS_CUDA:
        print(f"  CUDA device:     {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version:    {torch.version.cuda}")
    print()

    results: list[TimingResult] = []

    # Run benchmarks
    print("[1/7] Profiling Advection (CPU)...")
    results.append(profile_advection_cpu(512))
    print(f"      → {results[-1].avg_ms:.2f} ms avg")

    print("[2/7] Profiling Advection (GPU PyTorch)...")
    results.append(profile_advection_gpu(512))
    if results[-1].avg_ms > 0:
        print(f"      → {results[-1].avg_ms:.2f} ms avg")
    else:
        print("      → CUDA not available")

    print("[3/7] Profiling Gradient (CPU)...")
    results.append(profile_gradient_cpu(512))
    print(f"      → {results[-1].avg_ms:.2f} ms avg")

    print("[4/7] Profiling Laplacian (CPU)...")
    results.append(profile_laplacian_cpu(512))
    print(f"      → {results[-1].avg_ms:.2f} ms avg")

    print("[5/7] Profiling Poisson FFT (CPU)...")
    results.append(profile_fft_poisson_cpu(512))
    print(f"      → {results[-1].avg_ms:.2f} ms avg")

    print("[6/7] Profiling Tensor Contraction (CPU)...")
    results.append(profile_tensor_contraction_cpu(32, 512))
    print(f"      → {results[-1].avg_ms:.2f} ms avg")

    print("[7/7] Profiling Full NS Step (CPU)...")
    results.append(profile_full_ns_step(256))
    print(f"      → {results[-1].avg_ms:.2f} ms avg")

    # Analyze results
    print()
    print("=" * 70)
    print("  PROFILING RESULTS")
    print("=" * 70)
    print()
    print(f"{'Operation':<30} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("-" * 60)

    for r in results:
        if r.avg_ms > 0:
            print(
                f"{r.operation:<30} {r.avg_ms:>10.2f} {r.min_ms:>10.2f} {r.max_ms:>10.2f}"
            )

    # Find bottleneck
    valid_results = [r for r in results if r.avg_ms > 0]
    bottleneck = max(valid_results, key=lambda r: r.avg_ms)

    print()
    print("=" * 70)
    print("  BOTTLENECK ANALYSIS")
    print("=" * 70)
    print()
    print(f"  🔥 BOTTLENECK: {bottleneck.operation}")
    print(f"     Average time: {bottleneck.avg_ms:.2f} ms")
    print()

    # Recommendation
    if "Advection" in bottleneck.operation:
        print("  → RECOMMENDATION: Implement CUDA advection kernel (Phase 2B)")
    elif "Contraction" in bottleneck.operation:
        print("  → RECOMMENDATION: Implement CUDA tensor contraction kernel")
    elif "Poisson" in bottleneck.operation:
        print("  → RECOMMENDATION: GPU FFT already fast, focus on advection")
    elif "NS Step" in bottleneck.operation:
        print("  → RECOMMENDATION: Full step is slowest - advection is component")

    print()

    # Save results
    output_path = Path(__file__).parent / output_file
    with open(output_path, "w") as f:
        json.dump(
            {
                "system": {
                    "pytorch_version": torch.__version__,
                    "cuda_available": HAS_CUDA,
                    "cuda_device": torch.cuda.get_device_name(0) if HAS_CUDA else None,
                },
                "results": [asdict(r) for r in results],
                "bottleneck": bottleneck.operation,
            },
            f,
            indent=2,
        )

    print(f"  ✓ Results saved to: {output_path}")
    print()

    return results


if __name__ == "__main__":
    run_profiling()
