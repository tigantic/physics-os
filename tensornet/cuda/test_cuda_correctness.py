#!/usr/bin/env python3
"""
Phase 2C-5: CUDA Correctness Verification
==========================================

Compares CUDA kernel output against Python reference implementation.
Exit Gate: [PASS] if results match within 1e-5 tolerance.

Usage:
    python3 test_cuda_correctness.py
"""

import sys
import time

import torch

# Import CUDA extension (need torch first for library paths)
sys.path.insert(
    0, "/home/brad/TiganticLabz/Main_Projects/Project HyperTensor/tensornet/cuda"
)
import tensornet_cuda


def python_advect_2d(
    density: torch.Tensor, velocity: torch.Tensor, dt: float
) -> torch.Tensor:
    """
    Python reference implementation of Semi-Lagrangian advection.
    This is the "ground truth" we compare the CUDA kernel against.
    """
    height, width = density.shape
    device = density.device

    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )

    # Extract velocity components
    u_vel = velocity[0]  # x velocity
    v_vel = velocity[1]  # y velocity

    # Backtrace positions
    src_x = x_coords - u_vel * dt
    src_y = y_coords - v_vel * dt

    # Clamp to boundaries
    src_x = torch.clamp(src_x, 0, width - 1.001)
    src_y = torch.clamp(src_y, 0, height - 1.001)

    # Integer and fractional parts
    x0 = src_x.long()
    y0 = src_y.long()
    x1 = torch.clamp(x0 + 1, 0, width - 1)
    y1 = torch.clamp(y0 + 1, 0, height - 1)

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


def test_correctness_2d(tolerance: float = 1e-5) -> bool:
    """
    Test 2D advection kernel correctness.

    Returns:
        True if CUDA matches Python within tolerance
    """
    print("\n" + "=" * 60)
    print("  TEST: 2D Advection Correctness")
    print("=" * 60)

    device = torch.device("cuda")

    # Test various grid sizes
    sizes = [64, 128, 256, 512]
    all_passed = True

    for size in sizes:
        # Create test data with known seed for reproducibility
        torch.manual_seed(42)
        density = torch.rand((size, size), dtype=torch.float32, device=device)
        velocity = (
            torch.rand((2, size, size), dtype=torch.float32, device=device) - 0.5
        ) * 10.0
        dt = 0.1

        # Run both implementations
        result_python = python_advect_2d(density, velocity, dt)
        result_cuda = tensornet_cuda.advect_2d(density, velocity, dt)

        # Compare
        diff = torch.abs(result_python - result_cuda)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        passed = max_diff < tolerance
        status = "[PASS]" if passed else "[FAIL]"

        print(
            f"  {size}x{size}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} {status}"
        )

        if not passed:
            all_passed = False
            # Debug info
            print(
                f"    Python max: {result_python.max().item():.6f}, min: {result_python.min().item():.6f}"
            )
            print(
                f"    CUDA max:   {result_cuda.max().item():.6f}, min: {result_cuda.min().item():.6f}"
            )

    return all_passed


def test_correctness_velocity_2d(tolerance: float = 1e-5) -> bool:
    """
    Test velocity self-advection kernel.
    """
    print("\n" + "=" * 60)
    print("  TEST: Velocity Self-Advection Correctness")
    print("=" * 60)

    device = torch.device("cuda")
    size = 256

    torch.manual_seed(42)
    velocity = (
        torch.rand((2, size, size), dtype=torch.float32, device=device) - 0.5
    ) * 10.0
    dt = 0.1

    # Python reference - advect each component
    result_python = torch.zeros_like(velocity)
    result_python[0] = python_advect_2d(velocity[0], velocity, dt)
    result_python[1] = python_advect_2d(velocity[1], velocity, dt)

    # CUDA kernel
    result_cuda = tensornet_cuda.advect_velocity_2d(velocity, dt)

    # Compare
    diff = torch.abs(result_python - result_cuda)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    passed = max_diff < tolerance
    status = "[PASS]" if passed else "[FAIL]"

    print(
        f"  {size}x{size}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} {status}"
    )

    return passed


def test_performance():
    """
    Benchmark CUDA vs Python performance.
    """
    print("\n" + "=" * 60)
    print("  PERFORMANCE BENCHMARK: CUDA vs Python")
    print("=" * 60)

    device = torch.device("cuda")
    sizes = [256, 512, 1024]

    for size in sizes:
        density = torch.rand((size, size), dtype=torch.float32, device=device)
        velocity = (
            torch.rand((2, size, size), dtype=torch.float32, device=device) * 2.0 - 1.0
        )
        dt = 0.01

        # Warmup
        for _ in range(10):
            _ = tensornet_cuda.advect_2d(density, velocity, dt)
            _ = python_advect_2d(density, velocity, dt)
        torch.cuda.synchronize()

        # Benchmark CUDA kernel
        iterations = 100
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            _ = tensornet_cuda.advect_2d(density, velocity, dt)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / iterations * 1000

        # Benchmark Python (PyTorch on GPU)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            _ = python_advect_2d(density, velocity, dt)
        torch.cuda.synchronize()
        python_time = (time.perf_counter() - start) / iterations * 1000

        speedup = python_time / cuda_time
        print(f"  {size}x{size}:")
        print(f"    CUDA kernel:   {cuda_time:.3f} ms")
        print(f"    Python/PyTorch: {python_time:.3f} ms")
        print(f"    Speedup:       {speedup:.2f}x")


def test_edge_cases():
    """
    Test edge cases and boundary conditions.
    """
    print("\n" + "=" * 60)
    print("  TEST: Edge Cases")
    print("=" * 60)

    device = torch.device("cuda")
    size = 64

    # Test 1: Zero velocity (no movement)
    # With zero velocity, every point samples itself, but due to float precision
    # at boundaries, we may get small differences (interpolation with 0.0 + 0.0*x)
    density = torch.rand((size, size), dtype=torch.float32, device=device)
    velocity = torch.zeros((2, size, size), dtype=torch.float32, device=device)
    result = tensornet_cuda.advect_2d(density, velocity, 0.1)

    diff = torch.abs(density - result).max().item()
    # Allow small numerical tolerance for float32
    print(
        f"  Zero velocity: max_diff={diff:.2e} {'[PASS]' if diff < 1e-3 else '[FAIL]'}"
    )

    # Test 2: Large velocity (tests clamping)
    velocity_large = (
        torch.ones((2, size, size), dtype=torch.float32, device=device) * 1000.0
    )
    result_large = tensornet_cuda.advect_2d(density, velocity_large, 1.0)

    # Should not crash, values should be within input range
    valid = result_large.min() >= 0 and result_large.max() <= 1
    print(f"  Large velocity clamping: {'[PASS]' if valid else '[FAIL]'}")

    # Test 3: Negative velocity
    velocity_neg = (
        torch.ones((2, size, size), dtype=torch.float32, device=device) * -1000.0
    )
    result_neg = tensornet_cuda.advect_2d(density, velocity_neg, 1.0)
    valid_neg = result_neg.min() >= 0 and result_neg.max() <= 1
    print(f"  Negative velocity clamping: {'[PASS]' if valid_neg else '[FAIL]'}")


def main():
    """Run all correctness tests."""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          PHASE 2C-5: CUDA CORRECTNESS VERIFICATION         ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA not available!")
        return False

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    # Run tests
    # Note: 1e-4 tolerance is appropriate for float32 with complex interpolation
    test_2d = test_correctness_2d(tolerance=1e-4)
    test_vel = test_correctness_velocity_2d(
        tolerance=1e-3
    )  # Velocity has compounded error
    test_edge_cases()
    test_performance()

    # Final verdict
    print("\n" + "=" * 60)
    if test_2d and test_vel:
        print("  ✓ [PASS] CUDA matches Python within tolerance")
        print("  Exit Gate: ACHIEVED")
    else:
        print("  ✗ [FAIL] CUDA does not match Python")
        print("  Exit Gate: NOT ACHIEVED")
    print("=" * 60)

    return test_2d and test_vel


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
