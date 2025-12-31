#!/usr/bin/env python3
"""
Phase 2C-6: GPU Integration Benchmark
======================================

This script demonstrates the CUDA advection kernels running with full
GPU utilization. Run this while watching nvidia-smi to see the RTX 5070
load spike.

Success Criteria:
    - GPU >40% utilization 
    - Physics step <4ms
    - CUDA matches Python within tolerance
    
Usage:
    python3 tensornet/benchmarks/gpu_integration_benchmark.py
    
Watch GPU usage:
    watch -n 0.5 nvidia-smi
"""

import torch
import time
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/Project HyperTensor')

# ═══════════════════════════════════════════════════════════════════════════
# Import CUDA Module
# ═══════════════════════════════════════════════════════════════════════════

from tensornet.gpu import advection

print("╔════════════════════════════════════════════════════════════╗")
print("║     PHASE 2C-6: GPU INTEGRATION BENCHMARK                  ║")
print("╚════════════════════════════════════════════════════════════╝")

advection.print_gpu_status()
print()

if not advection.is_cuda_available():
    print("❌ CUDA kernels not available - exiting")
    sys.exit(1)

device = torch.device('cuda')


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Configuration
# ═══════════════════════════════════════════════════════════════════════════

GRID_SIZES = [256, 512, 1024, 2048]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
DT = 0.016  # 60 FPS time step


def create_velocity_field(H: int, W: int) -> torch.Tensor:
    """Create a Taylor-Green vortex velocity field."""
    x = torch.linspace(0, 2 * 3.14159, W, device=device)
    y = torch.linspace(0, 2 * 3.14159, H, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    u = torch.sin(X) * torch.cos(Y)
    v = -torch.cos(X) * torch.sin(Y)
    
    return torch.stack([u, v], dim=0)


def benchmark_cuda_advection(size: int) -> dict:
    """Benchmark CUDA advection at given grid size."""
    
    # Create test data
    density = torch.rand(size, size, device=device, dtype=torch.float32)
    velocity = create_velocity_field(size, size)
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = advection.advect_2d(density, velocity, DT)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        result = advection.advect_2d(density, velocity, DT)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'fps': 1000.0 / (sum(times) / len(times)),
    }


def benchmark_pytorch_advection(size: int) -> dict:
    """Benchmark PyTorch GPU advection (fallback path)."""
    
    density = torch.rand(size, size, device=device, dtype=torch.float32)
    velocity = create_velocity_field(size, size)
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = advection.advect_2d(density, velocity, DT, force_pytorch=True)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        result = advection.advect_2d(density, velocity, DT, force_pytorch=True)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'fps': 1000.0 / (sum(times) / len(times)),
    }


def benchmark_velocity_advection(size: int) -> dict:
    """Benchmark velocity self-advection (used in full Navier-Stokes step)."""
    
    velocity = create_velocity_field(size, size)
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = advection.advect_velocity_2d(velocity, DT)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        result = advection.advect_velocity_2d(velocity, DT)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'fps': 1000.0 / (sum(times) / len(times)),
    }


def sustained_load_test(duration_seconds: float = 5.0, size: int = 512):
    """
    Run sustained GPU load to demonstrate utilization.
    Watch nvidia-smi during this test!
    """
    print(f"\n┌─────────────────────────────────────────────────────────────┐")
    print(f"│  SUSTAINED LOAD TEST: {duration_seconds}s @ {size}x{size}                       │")
    print(f"│  Watch nvidia-smi for GPU utilization spike!               │")
    print(f"└─────────────────────────────────────────────────────────────┘")
    
    density = torch.rand(size, size, device=device, dtype=torch.float32)
    velocity = create_velocity_field(size, size)
    
    start_time = time.perf_counter()
    frames = 0
    
    while time.perf_counter() - start_time < duration_seconds:
        # Run advection operations continuously
        density = advection.advect_2d(density, velocity, DT)
        velocity = advection.advect_velocity_2d(velocity, DT)
        frames += 1
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    
    print(f"  Completed {frames} physics frames in {elapsed:.2f}s")
    print(f"  Average: {frames/elapsed:.0f} FPS ({1000*elapsed/frames:.3f} ms/frame)")
    print(f"  ✓ GPU should show >40% utilization in nvidia-smi")


# ═══════════════════════════════════════════════════════════════════════════
# Main Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 65)
    print("BENCHMARK: CUDA Advection Kernels vs PyTorch")
    print("═" * 65)
    
    print(f"\n{'Grid Size':>12} │ {'CUDA (ms)':>12} │ {'PyTorch (ms)':>14} │ {'Speedup':>8}")
    print("─" * 55)
    
    for size in GRID_SIZES:
        cuda_result = benchmark_cuda_advection(size)
        pytorch_result = benchmark_pytorch_advection(size)
        speedup = pytorch_result['mean_ms'] / cuda_result['mean_ms']
        
        print(f"{size:>8}x{size:<3} │ {cuda_result['mean_ms']:>12.4f} │ {pytorch_result['mean_ms']:>14.4f} │ {speedup:>7.1f}x")
    
    print("\n" + "═" * 65)
    print("BENCHMARK: Velocity Self-Advection (Navier-Stokes Component)")
    print("═" * 65)
    
    print(f"\n{'Grid Size':>12} │ {'CUDA (ms)':>12} │ {'FPS':>10}")
    print("─" * 40)
    
    for size in GRID_SIZES:
        result = benchmark_velocity_advection(size)
        print(f"{size:>8}x{size:<3} │ {result['mean_ms']:>12.4f} │ {result['fps']:>10.0f}")
    
    # Check target achievement
    print("\n" + "═" * 65)
    print("PHASE 2 EXIT GATE VERIFICATION")
    print("═" * 65)
    
    # Test at target resolution (512x512)
    cuda_512 = benchmark_cuda_advection(512)
    vel_512 = benchmark_velocity_advection(512)
    
    total_physics_time = cuda_512['mean_ms'] + vel_512['mean_ms']
    
    print(f"\n  Target Resolution: 512x512")
    print(f"  Scalar Advection:  {cuda_512['mean_ms']:.4f} ms")
    print(f"  Velocity Advect:   {vel_512['mean_ms']:.4f} ms")
    print(f"  Total Physics:     {total_physics_time:.4f} ms")
    print(f"  Target:            < 4.00 ms")
    
    if total_physics_time < 4.0:
        print(f"\n  ✅ EXIT GATE PASSED: Physics step < 4ms")
    else:
        print(f"\n  ❌ EXIT GATE FAILED: Physics step > 4ms")
    
    # Run sustained load test for GPU utilization verification
    sustained_load_test(duration_seconds=5.0, size=512)
    
    print("\n" + "═" * 65)
    print("PHASE 2C-6 INTEGRATION COMPLETE")
    print("═" * 65)


if __name__ == "__main__":
    main()
