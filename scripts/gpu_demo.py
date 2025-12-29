#!/usr/bin/env python
"""
GPU Acceleration Demo for HyperTensor CFD

Demonstrates GPU-accelerated flux computation and compares
CPU vs GPU performance for the 2D Euler solver.

Usage:
    python scripts/gpu_demo.py [--grid-size 200]
    
Requirements:
    - CUDA-capable GPU with PyTorch CUDA support
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def check_gpu_availability():
    """Check if GPU is available and print info."""
    print("=" * 50)
    print("GPU AVAILABILITY CHECK")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        print(f"✓ Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("✗ CUDA not available - running CPU comparison only")
        return False


def benchmark_flux_cpu(N: int, iterations: int = 100):
    """Benchmark flux computation on CPU."""
    from tensornet.cfd.godunov import roe_flux
    
    # Create random conservative variables
    rho = torch.rand(N, dtype=torch.float64) + 0.1
    rhou = torch.rand(N, dtype=torch.float64)
    E = torch.rand(N, dtype=torch.float64) + 1.0
    
    U = torch.stack([rho, rhou, E], dim=-1)
    gamma = 1.4
    
    # Warmup
    for _ in range(10):
        F = roe_flux(U[:-1], U[1:], gamma)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        F = roe_flux(U[:-1], U[1:], gamma)
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations


def benchmark_flux_gpu(N: int, iterations: int = 100):
    """Benchmark flux computation on GPU."""
    from tensornet.core.gpu import roe_flux_gpu, get_device, GPUConfig, DeviceType
    
    config = GPUConfig(device=DeviceType.CUDA)
    device = get_device(config)
    
    # Create random 2D conservative variables on GPU
    rho = torch.rand(N, N, dtype=torch.float64, device=device) + 0.1
    u = torch.rand(N, N, dtype=torch.float64, device=device)
    v = torch.rand(N, N, dtype=torch.float64, device=device)
    E = torch.rand(N, N, dtype=torch.float64, device=device) + 1.0
    gamma = 1.4
    
    # Warmup
    torch.cuda.synchronize()
    for _ in range(10):
        F_rho, F_rhou, F_rhov, F_E = roe_flux_gpu(rho, rho * u, rho * v, E, gamma)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        F_rho, F_rhou, F_rhov, F_E = roe_flux_gpu(rho, rho * u, rho * v, E, gamma)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations


def benchmark_einsum_cpu_vs_gpu(N: int = 256, chi: int = 64, d: int = 2):
    """Benchmark tensor contraction (einsum) CPU vs GPU."""
    print("\n" + "=" * 50)
    print("EINSUM BENCHMARK (Tensor Contraction)")
    print("=" * 50)
    print(f"Shape: ({chi}, {d}, {chi}) x ({chi}, {d}, {chi})")
    
    # CPU
    A_cpu = torch.rand(chi, d, chi, dtype=torch.float64)
    B_cpu = torch.rand(chi, d, chi, dtype=torch.float64)
    
    # Warmup
    for _ in range(10):
        C = torch.einsum('abc,cde->abde', A_cpu, B_cpu)
    
    start = time.perf_counter()
    for _ in range(100):
        C = torch.einsum('abc,cde->abde', A_cpu, B_cpu)
    cpu_time = (time.perf_counter() - start) / 100
    print(f"CPU: {cpu_time * 1000:.3f} ms")
    
    if torch.cuda.is_available():
        A_gpu = A_cpu.cuda()
        B_gpu = B_cpu.cuda()
        
        # Warmup
        torch.cuda.synchronize()
        for _ in range(10):
            C = torch.einsum('abc,cde->abde', A_gpu, B_gpu)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            C = torch.einsum('abc,cde->abde', A_gpu, B_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) / 100
        print(f"GPU: {gpu_time * 1000:.3f} ms")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")
    else:
        print("GPU: N/A (CUDA not available)")


def benchmark_svd_cpu_vs_gpu(N: int = 512):
    """Benchmark SVD decomposition CPU vs GPU."""
    print("\n" + "=" * 50)
    print("SVD BENCHMARK")
    print("=" * 50)
    print(f"Matrix size: {N} x {N}")
    
    # CPU
    A_cpu = torch.rand(N, N, dtype=torch.float64)
    
    # Warmup
    for _ in range(3):
        U, S, Vh = torch.linalg.svd(A_cpu, full_matrices=False)
    
    start = time.perf_counter()
    for _ in range(10):
        U, S, Vh = torch.linalg.svd(A_cpu, full_matrices=False)
    cpu_time = (time.perf_counter() - start) / 10
    print(f"CPU: {cpu_time * 1000:.1f} ms")
    
    if torch.cuda.is_available():
        A_gpu = A_cpu.cuda()
        
        # Warmup
        torch.cuda.synchronize()
        for _ in range(3):
            U, S, Vh = torch.linalg.svd(A_gpu, full_matrices=False)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(10):
            U, S, Vh = torch.linalg.svd(A_gpu, full_matrices=False)
        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) / 10
        print(f"GPU: {gpu_time * 1000:.1f} ms")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")
    else:
        print("GPU: N/A (CUDA not available)")


def main():
    parser = argparse.ArgumentParser(description="GPU acceleration demo")
    parser.add_argument("--grid-size", type=int, default=200, help="Grid size for benchmarks")
    args = parser.parse_args()
    
    print("=" * 50)
    print("HyperTensor GPU Acceleration Demo")
    print("=" * 50)
    
    has_gpu = check_gpu_availability()
    
    # Run benchmarks
    benchmark_einsum_cpu_vs_gpu(chi=64)
    benchmark_svd_cpu_vs_gpu(N=512)
    
    if has_gpu:
        print("\n" + "=" * 50)
        print("CFD FLUX BENCHMARK")
        print("=" * 50)
        
        N = args.grid_size
        print(f"Grid size: {N}x{N}")
        
        cpu_time = benchmark_flux_cpu(N * N, iterations=50)
        gpu_time = benchmark_flux_gpu(N, iterations=50)
        
        print(f"CPU Roe flux: {cpu_time * 1000:.3f} ms")
        print(f"GPU Roe flux: {gpu_time * 1000:.3f} ms")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("GPU acceleration is available for:")
    print("  - Tensor contractions (einsum)")
    print("  - SVD decomposition")
    print("  - CFD flux computation (roe_flux_gpu)")
    print("  - Strain rate tensor (compute_strain_rate_gpu)")
    print("\nTo use GPU in your code:")
    print("  from tensornet.core.gpu import get_device, GPUConfig, DeviceType")
    print("  config = GPUConfig(device=DeviceType.CUDA)")
    print("  device = get_device(config)")
    print("  tensor = tensor.to(device)")


if __name__ == "__main__":
    main()
