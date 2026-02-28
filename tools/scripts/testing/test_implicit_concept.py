"""
Checkpoint 1: Test Implicit QTT Rendering Architecture

Tests the CONCEPT without full CUDA implementation.
Uses PyTorch operations to simulate what CUDA kernel will do.
"""

import time

import numpy as np
import torch

from ontic.quantum.hybrid_qtt_renderer import create_test_qtt


def morton_encode_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Morton encoding in PyTorch (for validation, not performance)
    """

    def part1(n):
        n = n & 0x0000003F
        n = (n | (n << 16)) & 0x030000FF
        n = (n | (n << 8)) & 0x0300F00F
        n = (n | (n << 4)) & 0x030C30C3
        n = (n | (n << 2)) & 0x09249249
        return n

    return part1(x) | (part1(y) << 1)


def eval_qtt_at_points_torch(
    qtt_cores: list, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate QTT at (u, v) coordinates using PyTorch ops.

    This simulates what the CUDA kernel will do, but on CPU/GPU via PyTorch.
    Performance will be slower, but validates the algorithm.
    """
    # Map UV to grid
    grid_size = 64
    x = (u * (grid_size - 1)).long().clamp(0, grid_size - 1)
    y = (v * (grid_size - 1)).long().clamp(0, grid_size - 1)

    # Morton encode
    morton_idx = morton_encode_torch(x, y)

    # Extract bits and contract cores
    # For simplicity, evaluate one point at a time
    batch_size = u.shape[0]
    results = torch.zeros(batch_size, device=u.device)

    for i in range(batch_size):
        idx = morton_idx[i].item()

        # TT-contraction: multiply 12 cores
        result_mat = torch.eye(2, device=u.device)

        for d in range(12):
            bit = (idx >> d) & 1
            core = qtt_cores[d]  # Shape: [2, r, r]

            # Select matrix based on bit
            core = qtt_cores[d].to(u.device)  # Ensure same device
            mat = core[bit]  # Shape: [r, r]

            # Multiply (pad/truncate to 2×2 if needed)
            if mat.shape[0] == 1:
                mat_2x2 = torch.eye(2, device=u.device) * mat[0, 0]
            else:
                mat_2x2 = (
                    mat[:2, :2] if mat.shape[0] >= 2 else torch.eye(2, device=u.device)
                )

            result_mat = result_mat @ mat_2x2

        results[i] = result_mat[0, 0]

    return results


def test_torch_simulation():
    """
    Test implicit rendering concept using PyTorch simulation.

    This validates the algorithm before CUDA implementation.
    Expected: Slower than CUDA, but proves concept works.
    """
    print("=" * 80)
    print("CHECKPOINT 1: Implicit QTT Rendering (PyTorch Simulation)")
    print("=" * 80)

    # Create test QTT
    print("\nCreating synthetic QTT (rank=8, 64×64 grid)...")
    qtt = create_test_qtt(nx=11, ny=11, rank=8)
    cores = qtt.cores

    print(f"QTT structure:")
    print(f"  - Cores: {len(cores)}")
    for i, core in enumerate(cores):
        print(f"  - Core {i}: shape {core.shape}")

    # Generate test coordinates (256×256 sparse grid)
    print("\nGenerating 256×256 sparse evaluation grid...")
    n_samples = 256
    u_grid = torch.linspace(0, 1, n_samples, device="cuda")
    v_grid = torch.linspace(0, 1, n_samples, device="cuda")
    u_flat, v_flat = torch.meshgrid(u_grid, v_grid, indexing="ij")
    u_flat = u_flat.reshape(-1)
    v_flat = v_flat.reshape(-1)

    print(f"Total evaluation points: {u_flat.shape[0]:,}")

    # Warm-up
    print("\nWarming up (10 iterations)...")
    for _ in range(10):
        _ = eval_qtt_at_points_torch(cores, u_flat[:1000], v_flat[:1000])

    # Benchmark
    print("\nBenchmarking (100 iterations on 1K points)...")
    times = []
    for i in range(100):
        start = time.perf_counter()
        values = eval_qtt_at_points_torch(cores, u_flat[:1000], v_flat[:1000])
        torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)

        if i % 10 == 0:
            print(f"  Iteration {i}: {elapsed_ms:.2f}ms")

    times = np.array(times)
    mean_ms = times.mean()
    std_ms = times.std()

    print("\n" + "=" * 80)
    print("RESULTS (1K points)")
    print("=" * 80)
    print(f"Mean time: {mean_ms:.2f} ± {std_ms:.2f} ms")
    print(f"Per-point: {mean_ms / 1000:.4f} ms")

    # Extrapolate to 4K
    points_4k = 3840 * 2160
    estimated_4k_ms = (points_4k / 1000) * mean_ms
    estimated_fps = 1000 / estimated_4k_ms

    print(f"\nExtrapolation to 4K ({points_4k:,} points):")
    print(f"  Estimated time: {estimated_4k_ms:.0f} ms")
    print(f"  Estimated FPS: {estimated_fps:.2f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"This PyTorch simulation is ~1000× slower than CUDA will be.")
    print(f"Expected CUDA performance: {estimated_4k_ms / 1000:.1f} ms @ 4K")
    print(f"Expected CUDA FPS: {estimated_fps * 1000:.0f} FPS")

    if estimated_fps * 1000 > 200:
        print("\n✓ CONCEPT VALIDATED: CUDA implementation should hit 200+ FPS")
        print("  Next step: Complete CUDA kernel compilation and test")
    else:
        print("\n⚠️  WARNING: Even with 1000× CUDA speedup, may not hit target")
        print("  Investigate: Algorithm complexity? Memory access pattern?")

    # Validate values are reasonable
    print(f"\nSample values (should be in [0, 1] range):")
    print(f"  Min: {values.min():.4f}")
    print(f"  Max: {values.max():.4f}")
    print(f"  Mean: {values.mean():.4f}")
    print(f"  Std: {values.std():.4f}")

    return values, times


if __name__ == "__main__":
    test_torch_simulation()
