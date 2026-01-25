#!/usr/bin/env python3
"""
TENSOR GENESIS — Comprehensive Benchmark Suite

Real performance measurements across all Genesis layers.
No shortcuts. No approximations. Actual wall-clock times and memory usage.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import gc
import hashlib
import json
import resource
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import warnings
warnings.filterwarnings('ignore')

# GPU setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64

def setup_gpu():
    """Configure GPU for benchmarking."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        # Warm up GPU
        _ = torch.randn(1000, 1000, device=DEVICE) @ torch.randn(1000, 1000, device=DEVICE)
        torch.cuda.synchronize()
        return True
    return False

USE_GPU = setup_gpu()


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    scale: int
    time_seconds: float
    memory_bytes: int
    ops_per_second: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingAnalysis:
    """Analysis of how performance scales with problem size."""
    name: str
    scales: List[int]
    times: List[float]
    memories: List[int]
    time_complexity: str
    time_coefficient: float
    memory_complexity: str
    memory_coefficient: float
    speedup_vs_dense: Optional[float] = None


def get_memory_mb() -> float:
    """Get current memory usage in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def measure_time(func: Callable, *args, n_runs: int = 3, warmup: int = 1, **kwargs) -> Tuple[float, float, Any]:
    """Measure execution time with warmup and multiple runs."""
    for _ in range(warmup):
        result = func(*args, **kwargs)
        if USE_GPU:
            torch.cuda.synchronize()
    
    times = []
    for _ in range(n_runs):
        gc.collect()
        if USE_GPU:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if USE_GPU:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times), np.std(times), result


def fit_complexity(scales: List[int], times: List[float]) -> Tuple[str, float]:
    """Fit complexity class to timing data."""
    scales = np.array(scales, dtype=float)
    times = np.array(times, dtype=float)
    
    models = [
        ("O(1)", lambda n: np.ones_like(n)),
        ("O(log N)", lambda n: np.log2(n + 1)),
        ("O(N)", lambda n: n),
        ("O(N log N)", lambda n: n * np.log2(n + 1)),
        ("O(N²)", lambda n: n ** 2),
    ]
    
    best_r2 = -np.inf
    best_model = "O(?)"
    best_coef = 1.0
    
    for name, transform in models:
        x = transform(scales)
        if np.std(x) < 1e-10:
            continue
        
        coef = np.sum(x * times) / np.sum(x * x)
        predicted = coef * x
        ss_res = np.sum((times - predicted) ** 2)
        ss_tot = np.sum((times - np.mean(times)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = name
            best_coef = coef
    
    return best_model, best_coef


def print_header():
    """Print benchmark suite header."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          T E N S O R   G E N E S I S   B E N C H M A R K   S U I T E        ║
║                                                                              ║
║                    Real Performance • No Shortcuts                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {np.__version__}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Platform: {sys.platform}")
    if USE_GPU:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        print(f"  CUDA: {torch.version.cuda}")
    else:
        print(f"  GPU: Not available (running on CPU)")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


# =============================================================================
# BENCHMARK 1: QTT-OT (Layer 20) - Optimal Transport
# =============================================================================

def benchmark_qtt_ot() -> ScalingAnalysis:
    """Benchmark Optimal Transport via Sinkhorn algorithm."""
    print("━" * 78)
    print("  BENCHMARK 1: QTT-OT — Optimal Transport (Layer 20)")
    print("━" * 78)
    
    from tensornet.genesis.ot import QTTSinkhorn, QTTDistribution
    
    scales = [2**8, 2**10, 2**12, 2**14]
    times = []
    memories = []
    dense_times = []
    
    # Fixed grid bounds for both distributions
    grid_bounds = (-10.0, 10.0)
    
    for n in scales:
        print(f"\n  N = {n:,} points")
        
        # Create distributions - CPU for QTT ops (internal impl is CPU)
        mu = QTTDistribution.gaussian(mean=-2.0, std=1.0, grid_size=n, 
                                       grid_bounds=grid_bounds, device=torch.device('cpu'))
        nu = QTTDistribution.gaussian(mean=+2.0, std=1.0, grid_size=n, 
                                       grid_bounds=grid_bounds, device=torch.device('cpu'))
        
        sinkhorn = QTTSinkhorn(epsilon=0.1, max_iter=50, tol=1e-6, verbose=False)
        
        gc.collect()
        mem_before = get_memory_mb()
        
        mean_time, std_time, result = measure_time(
            sinkhorn.solve, mu, nu,
            n_runs=3, warmup=1
        )
        
        mem_after = get_memory_mb()
        mem_used = int((mem_after - mem_before) * 1024 * 1024)
        
        times.append(mean_time)
        memories.append(max(mem_used, 1024))
        
        print(f"    QTT-Sinkhorn: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"    Wasserstein: {result.wasserstein_distance:.6f}")
        print(f"    Iterations: {result.iterations}, Converged: {result.converged}")
        
        # Dense baseline ON GPU (only for small N)
        if n <= 2**12:
            x = torch.linspace(-10, 10, n, device=DEVICE, dtype=DTYPE)
            C = (x[:, None] - x[None, :]) ** 2
            mu_dense = torch.exp(-0.5 * ((x + 2) / 1.0) ** 2)
            mu_dense = mu_dense / mu_dense.sum()
            nu_dense = torch.exp(-0.5 * ((x - 2) / 1.0) ** 2)
            nu_dense = nu_dense / nu_dense.sum()
            
            def dense_sinkhorn_gpu(C, mu, nu, eps=0.1, max_iter=50):
                K = torch.exp(-C / eps)
                u = torch.ones(len(mu), device=DEVICE, dtype=DTYPE)
                for _ in range(max_iter):
                    v = nu / (K.T @ u + 1e-10)
                    u = mu / (K @ v + 1e-10)
                return torch.sum(C * (u[:, None] * K * v[None, :])).item()
            
            dense_time, _, _ = measure_time(
                dense_sinkhorn_gpu, C, mu_dense, nu_dense,
                n_runs=3, warmup=1
            )
            dense_times.append(dense_time)
            ratio = mean_time / dense_time
            print(f"    Dense GPU: {dense_time*1000:.2f} ms (QTT/GPU ratio: {ratio:.2f}×)")
        else:
            dense_times.append(None)
            print(f"    Dense: SKIPPED (N² = {n**2:,} too large)")
    
    time_complexity, time_coef = fit_complexity(scales, times)
    mem_complexity, mem_coef = fit_complexity(scales, memories)
    
    print(f"\n  Scaling Analysis:")
    print(f"    Time complexity: {time_complexity}")
    print(f"    Memory complexity: {mem_complexity}")
    
    return ScalingAnalysis(
        name="QTT-OT",
        scales=scales,
        times=times,
        memories=memories,
        time_complexity=time_complexity,
        time_coefficient=time_coef,
        memory_complexity=mem_complexity,
        memory_coefficient=mem_coef,
        speedup_vs_dense=None,
    )


# =============================================================================
# BENCHMARK 2: QTT-SGW (Layer 21) - Spectral Graph Wavelets
# =============================================================================

def benchmark_qtt_sgw() -> ScalingAnalysis:
    """Benchmark Spectral Graph Wavelets."""
    print("\n" + "━" * 78)
    print("  BENCHMARK 2: QTT-SGW — Spectral Graph Wavelets (Layer 21)")
    print("━" * 78)
    
    from tensornet.genesis.sgw import QTTLaplacian, QTTSignal, ChebyshevApproximator
    
    scales = [2**6, 2**8, 2**10, 2**12, 2**14]
    times = []
    memories = []
    
    for n in scales:
        print(f"\n  N = {n:,} nodes (1D chain graph)")
        
        # Create 1D chain Laplacian using correct API
        laplacian = QTTLaplacian.grid_1d(n)
        
        # Create signal using QTT - CPU for internal ops
        signal_data = torch.sin(2 * torch.pi * torch.arange(n, dtype=DTYPE) / n)
        signal = QTTSignal.from_dense(signal_data)
        
        # Create Chebyshev filter approximator (n_scales wavelets)
        n_scales = 4
        
        gc.collect()
        mem_before = get_memory_mb()
        
        # Benchmark filter application at multiple scales
        def wavelet_transform():
            results = []
            for s in range(n_scales):
                scale = 2.0 ** s
                # Apply Mexican hat: g(λ) = sλ * exp(-sλ)
                def kernel(lam, sc=scale):
                    x = sc * lam
                    return x * np.exp(-x) if x >= 0 else 0.0
                
                # Use the classmethod factory
                chebyshev = ChebyshevApproximator.from_function(kernel, laplacian, order=30)
                filtered = chebyshev.apply(signal)
                results.append(filtered)
            return results
        
        mean_time, std_time, coeffs = measure_time(wavelet_transform, n_runs=3, warmup=1)
        
        mem_after = get_memory_mb()
        mem_used = int((mem_after - mem_before) * 1024 * 1024)
        
        times.append(mean_time)
        memories.append(max(mem_used, 1024))
        
        print(f"    Wavelet transform ({n_scales} scales): {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"    Laplacian rank: {laplacian.max_rank}")
        
        # Dense baseline on GPU
        if n <= 2**10:
            L = torch.zeros((n, n), device=DEVICE, dtype=DTYPE)
            for i in range(n):
                L[i, i] = 2
                if i > 0: L[i, i-1] = -1
                if i < n - 1: L[i, i+1] = -1
            L[0, 0] = 1
            L[n-1, n-1] = 1
            
            def dense_wavelet():
                eigvals, eigvecs = torch.linalg.eigh(L)
                sig = torch.sin(2 * torch.pi * torch.arange(n, device=DEVICE, dtype=DTYPE) / n)
                results = []
                for s in range(n_scales):
                    scale = 2.0 ** s
                    g = scale * eigvals * torch.exp(-scale * eigvals)
                    filtered = eigvecs @ (g * (eigvecs.T @ sig))
                    results.append(filtered)
                return results
            
            dense_time, _, _ = measure_time(dense_wavelet, n_runs=3, warmup=1)
            ratio = mean_time / dense_time
            print(f"    Dense GPU: {dense_time*1000:.2f} ms (QTT/GPU ratio: {ratio:.2f}×)")
    
    time_complexity, time_coef = fit_complexity(scales, times)
    mem_complexity, mem_coef = fit_complexity(scales, memories)
    
    print(f"\n  Scaling Analysis:")
    print(f"    Time complexity: {time_complexity}")
    print(f"    Memory complexity: {mem_complexity}")
    
    return ScalingAnalysis(
        name="QTT-SGW",
        scales=scales,
        times=times,
        memories=memories,
        time_complexity=time_complexity,
        time_coefficient=time_coef,
        memory_complexity=mem_complexity,
        memory_coefficient=mem_coef,
    )


# =============================================================================
# BENCHMARK 3: QTT-Tropical (Layer 23) - Tropical Geometry
# =============================================================================

def benchmark_qtt_tropical() -> ScalingAnalysis:
    """Benchmark Tropical Geometry operations."""
    print("\n" + "━" * 78)
    print("  BENCHMARK 3: QTT-Tropical — Tropical Geometry (Layer 23)")
    print("━" * 78)
    
    from tensornet.genesis.tropical import (
        TropicalMatrix, tropical_matmul, floyd_warshall_tropical
    )
    from tensornet.genesis.tropical.semiring import MinPlusSemiring
    
    scales = [2**5, 2**6, 2**7, 2**8, 2**9, 2**10]  # Larger scales with GPU
    times = []
    memories = []
    dense_times = []
    
    for n in scales:
        print(f"\n  N = {n:,} × {n:,} tropical matrix")
        
        # Create chain distance matrix ON GPU
        D = torch.zeros(n, n, dtype=DTYPE, device=DEVICE)
        idx = torch.arange(n, device=DEVICE)
        D = torch.abs(idx[:, None] - idx[None, :]).to(DTYPE)
        
        # Use proper semiring object
        A = TropicalMatrix(data=D, semiring=MinPlusSemiring, size=n)
        
        gc.collect()
        if USE_GPU:
            torch.cuda.empty_cache()
        mem_before = get_memory_mb()
        
        # Benchmark tropical matrix squaring
        mean_time, std_time, result = measure_time(
            tropical_matmul, A, A,
            n_runs=3, warmup=1
        )
        
        mem_after = get_memory_mb()
        mem_used = int((mem_after - mem_before) * 1024 * 1024)
        
        times.append(mean_time)
        memories.append(max(mem_used, 1024))
        
        print(f"    Tropical matmul: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
        
        # Dense min-plus baseline ON GPU
        def dense_tropical_matmul_gpu(A, B):
            return torch.min(A[:, :, None] + B[None, :, :], dim=1)[0]
        
        dense_time, _, _ = measure_time(
            dense_tropical_matmul_gpu, D, D,
            n_runs=3, warmup=1
        )
        dense_times.append(dense_time)
        ratio = mean_time / dense_time  # How our impl compares
        print(f"    Dense baseline: {dense_time*1000:.3f} ms (our/dense: {ratio:.2f}×)")
    
    time_complexity, time_coef = fit_complexity(scales, times)
    mem_complexity, mem_coef = fit_complexity(scales, memories)
    
    print(f"\n  Scaling Analysis:")
    print(f"    Time complexity: {time_complexity}")
    print(f"    Memory complexity: {mem_complexity}")
    
    return ScalingAnalysis(
        name="QTT-Tropical",
        scales=scales,
        times=times,
        memories=memories,
        time_complexity=time_complexity,
        time_coefficient=time_coef,
        memory_complexity=mem_complexity,
        memory_coefficient=mem_coef,
    )


# =============================================================================
# BENCHMARK 4: QTT-RKHS (Layer 24) - Kernel Methods
# =============================================================================

def benchmark_qtt_rkhs() -> ScalingAnalysis:
    """Benchmark RKHS kernel methods."""
    print("\n" + "━" * 78)
    print("  BENCHMARK 4: QTT-RKHS — Kernel Methods (Layer 24)")
    print("━" * 78)
    
    from tensornet.genesis.rkhs import RBFKernel, maximum_mean_discrepancy
    
    scales = [128, 256, 512, 1024, 2048]
    times = []
    memories = []
    
    kernel = RBFKernel(length_scale=1.0)
    
    for n in scales:
        print(f"\n  N = {n:,} points")
        
        # Create random point clouds - CPU for internal ops
        torch.manual_seed(42)
        X = torch.randn(n, 2, dtype=DTYPE)  # CPU
        Y = torch.randn(n, 2, dtype=DTYPE) + 0.5  # CPU
        
        gc.collect()
        mem_before = get_memory_mb()
        
        mean_time, std_time, mmd_value = measure_time(
            maximum_mean_discrepancy, X, Y, kernel,
            n_runs=3, warmup=1
        )
        
        mem_after = get_memory_mb()
        mem_used = int((mem_after - mem_before) * 1024 * 1024)
        
        times.append(mean_time)
        memories.append(max(mem_used, 1024))
        
        print(f"    MMD computation: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"    MMD value: {mmd_value:.6f}")
        
        # Dense GPU baseline
        X_gpu = X.to(DEVICE)
        Y_gpu = Y.to(DEVICE)
        
        def dense_mmd_gpu(X, Y, gamma=1.0):
            m, n_pts = X.shape[0], Y.shape[0]
            XX = (X ** 2).sum(dim=1)
            YY = (Y ** 2).sum(dim=1)
            K_XX = torch.exp(-gamma * (XX[:, None] + XX[None, :] - 2 * X @ X.T))
            K_YY = torch.exp(-gamma * (YY[:, None] + YY[None, :] - 2 * Y @ Y.T))
            K_XY = torch.exp(-gamma * (XX[:, None] + YY[None, :] - 2 * X @ Y.T))
            return (K_XX.sum() / (m*m) + K_YY.sum() / (n_pts*n_pts) - 2 * K_XY.sum() / (m*n_pts)).item()
        
        dense_time, _, _ = measure_time(dense_mmd_gpu, X_gpu, Y_gpu, n_runs=3, warmup=1)
        ratio = mean_time / dense_time
        print(f"    Dense GPU: {dense_time*1000:.2f} ms (QTT/GPU ratio: {ratio:.2f}×)")
    
    time_complexity, time_coef = fit_complexity(scales, times)
    mem_complexity, mem_coef = fit_complexity(scales, memories)
    
    print(f"\n  Scaling Analysis:")
    print(f"    Time complexity: {time_complexity}")
    print(f"    Memory complexity: {mem_complexity}")
    
    return ScalingAnalysis(
        name="QTT-RKHS",
        scales=scales,
        times=times,
        memories=memories,
        time_complexity=time_complexity,
        time_coefficient=time_coef,
        memory_complexity=mem_complexity,
        memory_coefficient=mem_coef,
    )


# =============================================================================
# BENCHMARK 5: QTT-PH (Layer 25) - Persistent Homology (QTT-Native)
# =============================================================================

def benchmark_qtt_ph() -> ScalingAnalysis:
    """Benchmark Persistent Homology using QTT-native implementation."""
    print("\n" + "━" * 78)
    print("  BENCHMARK 5: QTT-PH — Persistent Homology (Layer 25)")
    print("━" * 78)
    
    from tensornet.genesis.topology.qtt_native import (
        QTTBoundaryMatrix, qtt_persistence_grid_1d
    )
    import math
    
    # Use n_bits (log2 of N), not N directly
    bit_counts = [6, 8, 10, 12, 14]
    scales = [2**b for b in bit_counts]
    times = []
    memories = []
    
    for n_bits in bit_counts:
        n = 2 ** n_bits
        print(f"\n  N = {n:,} simplices (1D chain, {n_bits} bits)")
        
        gc.collect()
        mem_before = get_memory_mb()
        
        # Benchmark boundary operator construction (use n_bits as param)
        mean_time, std_time, boundary = measure_time(
            QTTBoundaryMatrix.for_grid_1d, n_bits,
            n_runs=3, warmup=1
        )
        
        mem_after = get_memory_mb()
        
        times.append(mean_time)
        memories.append(boundary.memory_bytes)
        
        print(f"    Boundary construction: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
        print(f"    Max rank: {boundary.max_rank}")
        print(f"    Memory: {boundary.memory_bytes:,} bytes")
        
        # Compute Betti numbers using n_bits
        result = qtt_persistence_grid_1d(n_bits)
        betti = result.betti_numbers
        print(f"    Betti numbers: β = {betti}")
        
        # Dense baseline (memory comparison)
        dense_mem = (n - 1) * n * 8
        compression = dense_mem / max(boundary.memory_bytes, 1)
        print(f"    Compression vs dense: {compression:.0f}×")
    
    time_complexity, time_coef = fit_complexity(scales, times)
    mem_complexity, mem_coef = fit_complexity(scales, memories)
    
    print(f"\n  Scaling Analysis:")
    print(f"    Time complexity: {time_complexity}")
    print(f"    Memory complexity: {mem_complexity}")
    
    # Memory growth analysis
    if len(memories) >= 2 and memories[0] > 0:
        mem_ratio = memories[-1] / memories[0]
        scale_ratio = scales[-1] / scales[0]
        log_ratio = np.log2(scale_ratio)
        print(f"    Memory ratio for {scale_ratio:.0f}× scale: {mem_ratio:.1f}×")
        print(f"    (O(log N) target: ~{log_ratio:.1f}× growth)")
    
    return ScalingAnalysis(
        name="QTT-PH",
        scales=scales,
        times=times,
        memories=memories,
        time_complexity=time_complexity,
        time_coefficient=time_coef,
        memory_complexity=mem_complexity,
        memory_coefficient=mem_coef,
    )


# =============================================================================
# BENCHMARK 6: QTT-GA (Layer 26) - Geometric Algebra
# =============================================================================

def benchmark_qtt_ga() -> ScalingAnalysis:
    """Benchmark Geometric Algebra operations."""
    print("\n" + "━" * 78)
    print("  BENCHMARK 6: QTT-GA — Geometric Algebra (Layer 26)")
    print("━" * 78)
    
    from tensornet.genesis.ga import CliffordAlgebra, Multivector, geometric_product
    
    dimensions = [3, 4, 5, 6, 7, 8]
    scales = [2**d for d in dimensions]
    times = []
    memories = []
    
    for dim in dimensions:
        n_components = 2 ** dim
        print(f"\n  Dimension = {dim} ({n_components} components)")
        
        algebra = CliffordAlgebra(dim)
        
        np.random.seed(42)
        coeffs_a = np.random.randn(n_components)
        coeffs_b = np.random.randn(n_components)
        
        a = Multivector(algebra, coeffs_a)
        b = Multivector(algebra, coeffs_b)
        
        gc.collect()
        mem_before = get_memory_mb()
        
        mean_time, std_time, result = measure_time(
            geometric_product, a, b,
            n_runs=5, warmup=2
        )
        
        mem_after = get_memory_mb()
        mem_used = int((mem_after - mem_before) * 1024 * 1024)
        
        times.append(mean_time)
        memories.append(max(mem_used, n_components * 8))
        
        print(f"    Geometric product: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
        print(f"    Operations: ~{n_components**2:,} multiplications")
        print(f"    Throughput: {n_components**2 / mean_time / 1e6:.1f} M ops/sec")
    
    time_complexity, time_coef = fit_complexity(scales, times)
    mem_complexity, mem_coef = fit_complexity(scales, memories)
    
    print(f"\n  Scaling Analysis:")
    print(f"    Time complexity: {time_complexity}")
    print(f"    Memory complexity: {mem_complexity}")
    
    return ScalingAnalysis(
        name="QTT-GA",
        scales=scales,
        times=times,
        memories=memories,
        time_complexity=time_complexity,
        time_coefficient=time_coef,
        memory_complexity=mem_complexity,
        memory_coefficient=mem_coef,
    )


# =============================================================================
# SUMMARY AND ATTESTATION
# =============================================================================

def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks and generate attestation."""
    print_header()
    
    all_results = {
        "meta": {
            "suite": "TENSOR GENESIS Benchmark Suite",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "platform": sys.platform,
        },
        "benchmarks": {},
        "scaling_analyses": [],
    }
    
    total_start = time.perf_counter()
    
    benchmarks = [
        ("QTT-OT", benchmark_qtt_ot),
        ("QTT-SGW", benchmark_qtt_sgw),
        ("QTT-Tropical", benchmark_qtt_tropical),
        ("QTT-RKHS", benchmark_qtt_rkhs),
        ("QTT-PH", benchmark_qtt_ph),
        ("QTT-GA", benchmark_qtt_ga),
    ]
    
    for name, bench_fn in benchmarks:
        try:
            result = bench_fn()
            all_results["benchmarks"][name] = {
                "scales": result.scales,
                "times_seconds": result.times,
                "times_ms": [t * 1000 for t in result.times],
                "memories_bytes": result.memories,
                "time_complexity": result.time_complexity,
                "memory_complexity": result.memory_complexity,
                "speedup_vs_dense": result.speedup_vs_dense,
            }
            all_results["scaling_analyses"].append({
                "name": result.name,
                "time_complexity": result.time_complexity,
                "memory_complexity": result.memory_complexity,
            })
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results["benchmarks"][name] = {"error": str(e)}
    
    total_time = time.perf_counter() - total_start
    all_results["meta"]["total_time_seconds"] = total_time
    
    # Print summary
    print("\n" + "═" * 78)
    print("  B E N C H M A R K   S U M M A R Y")
    print("═" * 78)
    
    print(f"\n  {'Primitive':<20} {'Time Complexity':<18} {'Memory Complexity':<18}")
    print(f"  {'-'*20} {'-'*18} {'-'*18}")
    for analysis in all_results["scaling_analyses"]:
        print(f"  {analysis['name']:<20} {analysis['time_complexity']:<18} {analysis['memory_complexity']:<18}")
    
    print(f"\n  Total benchmark time: {total_time:.1f} seconds")
    
    # Count successes
    n_success = len([1 for b in all_results["benchmarks"].values() if "error" not in b])
    n_total = len(benchmarks)
    
    # Generate attestation
    attestation = {
        "attestation": "TENSOR GENESIS BENCHMARK ATTESTATION",
        "project": "TENSOR GENESIS",
        "timestamp": all_results["meta"]["timestamp"],
        "environment": {
            "python": all_results["meta"]["python_version"],
            "numpy": all_results["meta"]["numpy_version"],
            "torch": all_results["meta"]["torch_version"],
            "platform": all_results["meta"]["platform"],
        },
        "benchmarks_completed": f"{n_success}/{n_total}",
        "results": all_results["benchmarks"],
        "scaling_summary": all_results["scaling_analyses"],
        "total_time_seconds": total_time,
    }
    
    content = json.dumps(attestation, sort_keys=True, default=str)
    attestation["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    
    with open("GENESIS_BENCHMARK_ATTESTATION.json", "w") as f:
        json.dump(attestation, f, indent=2, default=str)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  {'★★★ BENCHMARK SUITE COMPLETE ★★★':<68} ║
║                                                                              ║
║  Benchmarks: {n_success}/{n_total} passed{' ' * 56}║
║  Attestation: GENESIS_BENCHMARK_ATTESTATION.json                             ║
║  SHA256: {attestation['sha256'][:40]}...             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
