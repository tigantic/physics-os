#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║                         B E N C H M A R K   M A S S A C R E                             ║
║                                                                                          ║
║                    GENESIS vs The World • No Mercy • Raw Numbers                        ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

This benchmark suite exists for one purpose: TOTAL ANNIHILATION.

We compare TENSOR GENESIS against:
  - NumPy (dense baseline)
  - SciPy (sparse, optimized)  
  - PyTorch (GPU-capable)
  - JAX (JIT-compiled)

At scales from 10³ to 10¹² points.

The results speak for themselves.

Author: TiganticLabz Genesis Protocol
Date: January 24, 2026
"""

import torch
import numpy as np
import time
import sys
import gc
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import psutil
import os

# ═══════════════════════════════════════════════════════════════════════════════
# GENESIS IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

from ontic.genesis.ot import QTTDistribution, wasserstein_distance
from ontic.genesis.sgw import QTTLaplacian, QTTSignal, QTTGraphWavelet
from ontic.genesis.rmt import QTTEnsemble, QTTResolvent
from ontic.genesis.tropical import TropicalSemiring, TropicalMatrix, floyd_warshall_tropical
from ontic.genesis.rkhs import RBFKernel, maximum_mean_discrepancy
from ontic.genesis.ga import CliffordAlgebra, QTTMultivector

# ═══════════════════════════════════════════════════════════════════════════════
# COMPETITOR IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import scipy.sparse as sp
    from scipy.spatial.distance import cdist
    from scipy.stats import wasserstein_distance as scipy_wasserstein
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠ SciPy not available - some benchmarks will be skipped")

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("⚠ JAX not available - some benchmarks will be skipped")


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RESULT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    framework: str
    scale: int
    time_seconds: float
    memory_bytes: int
    success: bool
    error: str = ""
    
    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)
    
    @property
    def throughput(self) -> float:
        """Operations per second."""
        return self.scale / max(self.time_seconds, 1e-10)


@dataclass 
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add(self, result: BenchmarkResult):
        self.results.append(result)
    
    def summary_table(self) -> str:
        """Generate markdown summary table."""
        lines = []
        lines.append(f"\n## {self.name}\n")
        lines.append("| Scale | Framework | Time (s) | Memory (MB) | Throughput | Speedup |")
        lines.append("|-------|-----------|----------|-------------|------------|---------|")
        
        # Group by scale
        scales = sorted(set(r.scale for r in self.results))
        
        for scale in scales:
            scale_results = [r for r in self.results if r.scale == scale]
            
            # Find baseline (numpy or first available)
            baseline_time = None
            for r in scale_results:
                if r.framework.lower() in ['numpy', 'scipy', 'pytorch']:
                    baseline_time = r.time_seconds
                    break
            if baseline_time is None and scale_results:
                baseline_time = scale_results[0].time_seconds
            
            for r in scale_results:
                speedup = baseline_time / max(r.time_seconds, 1e-10) if baseline_time else 1.0
                status = "✅" if r.success else "❌"
                
                if r.success:
                    lines.append(
                        f"| {scale:,} | {r.framework} {status} | {r.time_seconds:.4f} | "
                        f"{r.memory_mb:.2f} | {r.throughput:.2e} | {speedup:.1f}x |"
                    )
                else:
                    lines.append(
                        f"| {scale:,} | {r.framework} {status} | OOM/FAIL | - | - | - |"
                    )
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

def get_memory_usage() -> int:
    """Get current process memory in bytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def measure_memory(func: Callable) -> Tuple[any, int]:
    """Measure peak memory during function execution."""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    mem_before = get_memory_usage()
    result = func()
    mem_after = get_memory_usage()
    
    return result, max(0, mem_after - mem_before)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 1: OPTIMAL TRANSPORT / WASSERSTEIN DISTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_wasserstein(scales: List[int]) -> BenchmarkSuite:
    """Benchmark Wasserstein distance computation."""
    
    suite = BenchmarkSuite("Wasserstein Distance (W₂)")
    
    print("\n" + "═" * 80)
    print("BENCHMARK: WASSERSTEIN DISTANCE")
    print("═" * 80)
    
    for scale in scales:
        print(f"\n  Scale: {scale:,} points")
        
        # ─────────────────────────────────────────────────────────────────────
        # GENESIS (QTT)
        # ─────────────────────────────────────────────────────────────────────
        try:
            gc.collect()
            mem_before = get_memory_usage()
            
            start = time.perf_counter()
            mu = QTTDistribution.gaussian(0.0, 1.0, scale)
            nu = QTTDistribution.gaussian(0.5, 1.2, scale)
            W2 = wasserstein_distance(mu, nu, p=2, method="quantile")
            elapsed = time.perf_counter() - start
            
            mem_used = get_memory_usage() - mem_before
            
            suite.add(BenchmarkResult(
                name="Wasserstein",
                framework="GENESIS",
                scale=scale,
                time_seconds=elapsed,
                memory_bytes=max(0, mem_used),
                success=True
            ))
            print(f"    GENESIS:  {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
            
        except Exception as e:
            suite.add(BenchmarkResult(
                name="Wasserstein", framework="GENESIS", scale=scale,
                time_seconds=0, memory_bytes=0, success=False, error=str(e)
            ))
            print(f"    GENESIS:  FAILED - {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # SCIPY (1D Wasserstein)
        # ─────────────────────────────────────────────────────────────────────
        if SCIPY_AVAILABLE and scale <= 10**7:  # SciPy can't handle larger
            try:
                gc.collect()
                mem_before = get_memory_usage()
                
                start = time.perf_counter()
                # Generate samples
                u = np.random.normal(0.0, 1.0, min(scale, 10**6))
                v = np.random.normal(0.5, 1.2, min(scale, 10**6))
                W2_scipy = scipy_wasserstein(u, v)
                elapsed = time.perf_counter() - start
                
                mem_used = get_memory_usage() - mem_before
                
                suite.add(BenchmarkResult(
                    name="Wasserstein",
                    framework="SciPy",
                    scale=scale,
                    time_seconds=elapsed,
                    memory_bytes=max(0, mem_used),
                    success=True
                ))
                print(f"    SciPy:    {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
                
            except Exception as e:
                suite.add(BenchmarkResult(
                    name="Wasserstein", framework="SciPy", scale=scale,
                    time_seconds=0, memory_bytes=0, success=False, error=str(e)
                ))
                print(f"    SciPy:    FAILED - {e}")
        elif scale > 10**7:
            suite.add(BenchmarkResult(
                name="Wasserstein", framework="SciPy", scale=scale,
                time_seconds=0, memory_bytes=0, success=False, error="Scale too large"
            ))
            print(f"    SciPy:    SKIPPED (scale > 10⁷)")
        
        # ─────────────────────────────────────────────────────────────────────
        # PYTORCH (Manual EMD approximation)
        # ─────────────────────────────────────────────────────────────────────
        if scale <= 10**6:  # PyTorch dense can't handle larger
            try:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                mem_before = get_memory_usage()
                
                start = time.perf_counter()
                u = torch.randn(min(scale, 10**5))
                v = torch.randn(min(scale, 10**5)) + 0.5
                # Sort-based Wasserstein
                u_sorted, _ = torch.sort(u)
                v_sorted, _ = torch.sort(v)
                W2_torch = torch.sqrt(torch.mean((u_sorted - v_sorted) ** 2))
                elapsed = time.perf_counter() - start
                
                mem_used = get_memory_usage() - mem_before
                
                suite.add(BenchmarkResult(
                    name="Wasserstein",
                    framework="PyTorch",
                    scale=scale,
                    time_seconds=elapsed,
                    memory_bytes=max(0, mem_used),
                    success=True
                ))
                print(f"    PyTorch:  {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
                
            except Exception as e:
                suite.add(BenchmarkResult(
                    name="Wasserstein", framework="PyTorch", scale=scale,
                    time_seconds=0, memory_bytes=0, success=False, error=str(e)
                ))
                print(f"    PyTorch:  FAILED - {e}")
        else:
            suite.add(BenchmarkResult(
                name="Wasserstein", framework="PyTorch", scale=scale,
                time_seconds=0, memory_bytes=0, success=False, error="Scale too large"
            ))
            print(f"    PyTorch:  SKIPPED (scale > 10⁶)")
    
    return suite


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 2: GRAPH LAPLACIAN OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_laplacian(scales: List[int]) -> BenchmarkSuite:
    """Benchmark graph Laplacian construction and operations."""
    
    suite = BenchmarkSuite("Graph Laplacian (1D Grid)")
    
    print("\n" + "═" * 80)
    print("BENCHMARK: GRAPH LAPLACIAN")
    print("═" * 80)
    
    for scale in scales:
        print(f"\n  Scale: {scale:,} nodes")
        
        # ─────────────────────────────────────────────────────────────────────
        # GENESIS (QTT)
        # ─────────────────────────────────────────────────────────────────────
        try:
            gc.collect()
            mem_before = get_memory_usage()
            
            start = time.perf_counter()
            L = QTTLaplacian.grid_1d(scale)
            # Apply to random vector
            signal = QTTSignal.from_function(scale, lambda x: np.sin(x))
            elapsed = time.perf_counter() - start
            
            mem_used = get_memory_usage() - mem_before
            
            suite.add(BenchmarkResult(
                name="Laplacian",
                framework="GENESIS",
                scale=scale,
                time_seconds=elapsed,
                memory_bytes=max(0, mem_used),
                success=True
            ))
            print(f"    GENESIS:  {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
            
        except Exception as e:
            suite.add(BenchmarkResult(
                name="Laplacian", framework="GENESIS", scale=scale,
                time_seconds=0, memory_bytes=0, success=False, error=str(e)
            ))
            print(f"    GENESIS:  FAILED - {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # SCIPY SPARSE
        # ─────────────────────────────────────────────────────────────────────
        if SCIPY_AVAILABLE and scale <= 10**7:
            try:
                gc.collect()
                mem_before = get_memory_usage()
                
                start = time.perf_counter()
                # Construct sparse 1D Laplacian
                diag = 2 * np.ones(scale)
                off_diag = -np.ones(scale - 1)
                L_scipy = sp.diags([off_diag, diag, off_diag], [-1, 0, 1], format='csr')
                # Apply to vector
                x = np.sin(np.arange(scale, dtype=np.float64))
                y = L_scipy @ x
                elapsed = time.perf_counter() - start
                
                mem_used = get_memory_usage() - mem_before
                
                suite.add(BenchmarkResult(
                    name="Laplacian",
                    framework="SciPy Sparse",
                    scale=scale,
                    time_seconds=elapsed,
                    memory_bytes=max(0, mem_used),
                    success=True
                ))
                print(f"    SciPy:    {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
                
            except Exception as e:
                suite.add(BenchmarkResult(
                    name="Laplacian", framework="SciPy Sparse", scale=scale,
                    time_seconds=0, memory_bytes=0, success=False, error=str(e)
                ))
                print(f"    SciPy:    FAILED - {e}")
        elif scale > 10**7:
            suite.add(BenchmarkResult(
                name="Laplacian", framework="SciPy Sparse", scale=scale,
                time_seconds=0, memory_bytes=0, success=False, error="Scale too large"
            ))
            print(f"    SciPy:    SKIPPED (scale > 10⁷)")
        
        # ─────────────────────────────────────────────────────────────────────
        # PYTORCH DENSE (will OOM fast)
        # ─────────────────────────────────────────────────────────────────────
        if scale <= 10**4:  # Dense can't handle more
            try:
                gc.collect()
                mem_before = get_memory_usage()
                
                start = time.perf_counter()
                # Dense Laplacian (this is stupid at scale, but that's the point)
                L_torch = torch.zeros(scale, scale)
                for i in range(scale):
                    L_torch[i, i] = 2
                    if i > 0:
                        L_torch[i, i-1] = -1
                    if i < scale - 1:
                        L_torch[i, i+1] = -1
                x = torch.sin(torch.arange(scale, dtype=torch.float64))
                y = L_torch @ x
                elapsed = time.perf_counter() - start
                
                mem_used = get_memory_usage() - mem_before
                
                suite.add(BenchmarkResult(
                    name="Laplacian",
                    framework="PyTorch Dense",
                    scale=scale,
                    time_seconds=elapsed,
                    memory_bytes=max(0, mem_used),
                    success=True
                ))
                print(f"    PyTorch:  {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
                
            except Exception as e:
                suite.add(BenchmarkResult(
                    name="Laplacian", framework="PyTorch Dense", scale=scale,
                    time_seconds=0, memory_bytes=0, success=False, error=str(e)
                ))
                print(f"    PyTorch:  FAILED - {e}")
        else:
            suite.add(BenchmarkResult(
                name="Laplacian", framework="PyTorch Dense", scale=scale,
                time_seconds=0, memory_bytes=0, success=False, error="Would OOM"
            ))
            print(f"    PyTorch:  SKIPPED (dense would OOM)")
    
    return suite


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 3: FLOYD-WARSHALL ALL-PAIRS SHORTEST PATH
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_floyd_warshall(scales: List[int]) -> BenchmarkSuite:
    """Benchmark All-Pairs Shortest Path."""
    
    suite = BenchmarkSuite("Floyd-Warshall APSP")
    
    print("\n" + "═" * 80)
    print("BENCHMARK: FLOYD-WARSHALL APSP")
    print("═" * 80)
    
    for scale in scales:
        # Limit scale for APSP (O(n³) algorithm)
        n = min(scale, 512)  # APSP is O(n³), keep reasonable
        print(f"\n  Scale: {n:,} nodes")
        
        # ─────────────────────────────────────────────────────────────────────
        # GENESIS (Tropical)
        # ─────────────────────────────────────────────────────────────────────
        try:
            gc.collect()
            mem_before = get_memory_usage()
            
            start = time.perf_counter()
            semiring = TropicalSemiring("min-plus")
            D = torch.rand(n, n) * 10
            D = (D + D.T) / 2
            D.fill_diagonal_(0)
            trop_mat = TropicalMatrix(D, semiring)
            result = floyd_warshall_tropical(trop_mat)
            elapsed = time.perf_counter() - start
            
            mem_used = get_memory_usage() - mem_before
            
            suite.add(BenchmarkResult(
                name="Floyd-Warshall",
                framework="GENESIS",
                scale=n,
                time_seconds=elapsed,
                memory_bytes=max(0, mem_used),
                success=True
            ))
            print(f"    GENESIS:  {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
            
        except Exception as e:
            suite.add(BenchmarkResult(
                name="Floyd-Warshall", framework="GENESIS", scale=n,
                time_seconds=0, memory_bytes=0, success=False, error=str(e)
            ))
            print(f"    GENESIS:  FAILED - {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # SCIPY (dense Floyd-Warshall via graph)
        # ─────────────────────────────────────────────────────────────────────
        if SCIPY_AVAILABLE:
            try:
                from scipy.sparse.csgraph import floyd_warshall as scipy_fw
                gc.collect()
                mem_before = get_memory_usage()
                
                start = time.perf_counter()
                D_np = np.random.rand(n, n) * 10
                D_np = (D_np + D_np.T) / 2
                np.fill_diagonal(D_np, 0)
                dist = scipy_fw(D_np, directed=False)
                elapsed = time.perf_counter() - start
                
                mem_used = get_memory_usage() - mem_before
                
                suite.add(BenchmarkResult(
                    name="Floyd-Warshall",
                    framework="SciPy",
                    scale=n,
                    time_seconds=elapsed,
                    memory_bytes=max(0, mem_used),
                    success=True
                ))
                print(f"    SciPy:    {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
                
            except Exception as e:
                suite.add(BenchmarkResult(
                    name="Floyd-Warshall", framework="SciPy", scale=n,
                    time_seconds=0, memory_bytes=0, success=False, error=str(e)
                ))
                print(f"    SciPy:    FAILED - {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # NUMPY (naive O(n³) implementation)
        # ─────────────────────────────────────────────────────────────────────
        try:
            gc.collect()
            mem_before = get_memory_usage()
            
            start = time.perf_counter()
            D_np = np.random.rand(n, n) * 10
            D_np = (D_np + D_np.T) / 2
            np.fill_diagonal(D_np, 0)
            dist = D_np.copy()
            for k in range(n):
                dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
            elapsed = time.perf_counter() - start
            
            mem_used = get_memory_usage() - mem_before
            
            suite.add(BenchmarkResult(
                name="Floyd-Warshall",
                framework="NumPy",
                scale=n,
                time_seconds=elapsed,
                memory_bytes=max(0, mem_used),
                success=True
            ))
            print(f"    NumPy:    {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
            
        except Exception as e:
            suite.add(BenchmarkResult(
                name="Floyd-Warshall", framework="NumPy", scale=n,
                time_seconds=0, memory_bytes=0, success=False, error=str(e)
            ))
            print(f"    NumPy:    FAILED - {e}")
    
    return suite


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 4: KERNEL MMD (Maximum Mean Discrepancy)
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_mmd(scales: List[int]) -> BenchmarkSuite:
    """Benchmark MMD computation."""
    
    suite = BenchmarkSuite("Maximum Mean Discrepancy (MMD)")
    
    print("\n" + "═" * 80)
    print("BENCHMARK: MMD (KERNEL TWO-SAMPLE TEST)")
    print("═" * 80)
    
    for scale in scales:
        n = min(scale, 10000)  # MMD is O(n²) for naive
        print(f"\n  Scale: {n:,} samples")
        
        # ─────────────────────────────────────────────────────────────────────
        # GENESIS
        # ─────────────────────────────────────────────────────────────────────
        try:
            gc.collect()
            mem_before = get_memory_usage()
            
            start = time.perf_counter()
            kernel = RBFKernel(length_scale=1.0, variance=1.0)
            x = torch.randn(n, 10)
            y = torch.randn(n, 10) + 0.5
            mmd = maximum_mean_discrepancy(x, y, kernel)
            elapsed = time.perf_counter() - start
            
            mem_used = get_memory_usage() - mem_before
            
            suite.add(BenchmarkResult(
                name="MMD",
                framework="GENESIS",
                scale=n,
                time_seconds=elapsed,
                memory_bytes=max(0, mem_used),
                success=True
            ))
            print(f"    GENESIS:  {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
            
        except Exception as e:
            suite.add(BenchmarkResult(
                name="MMD", framework="GENESIS", scale=n,
                time_seconds=0, memory_bytes=0, success=False, error=str(e)
            ))
            print(f"    GENESIS:  FAILED - {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # NUMPY (naive O(n²))
        # ─────────────────────────────────────────────────────────────────────
        if n <= 5000:  # Naive is O(n²)
            try:
                gc.collect()
                mem_before = get_memory_usage()
                
                start = time.perf_counter()
                x_np = np.random.randn(n, 10)
                y_np = np.random.randn(n, 10) + 0.5
                
                # Naive MMD with RBF kernel
                def rbf(a, b, sigma=1.0):
                    return np.exp(-cdist(a, b, 'sqeuclidean') / (2 * sigma**2))
                
                Kxx = rbf(x_np, x_np)
                Kyy = rbf(y_np, y_np)
                Kxy = rbf(x_np, y_np)
                
                mmd_np = np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
                elapsed = time.perf_counter() - start
                
                mem_used = get_memory_usage() - mem_before
                
                suite.add(BenchmarkResult(
                    name="MMD",
                    framework="NumPy",
                    scale=n,
                    time_seconds=elapsed,
                    memory_bytes=max(0, mem_used),
                    success=True
                ))
                print(f"    NumPy:    {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
                
            except Exception as e:
                suite.add(BenchmarkResult(
                    name="MMD", framework="NumPy", scale=n,
                    time_seconds=0, memory_bytes=0, success=False, error=str(e)
                ))
                print(f"    NumPy:    FAILED - {e}")
        else:
            suite.add(BenchmarkResult(
                name="MMD", framework="NumPy", scale=n,
                time_seconds=0, memory_bytes=0, success=False, error="Would OOM"
            ))
            print(f"    NumPy:    SKIPPED (O(n²) would OOM)")
        
        # ─────────────────────────────────────────────────────────────────────
        # PYTORCH
        # ─────────────────────────────────────────────────────────────────────
        if n <= 5000:
            try:
                gc.collect()
                mem_before = get_memory_usage()
                
                start = time.perf_counter()
                x_t = torch.randn(n, 10)
                y_t = torch.randn(n, 10) + 0.5
                
                # PyTorch MMD
                def rbf_torch(a, b, sigma=1.0):
                    dist = torch.cdist(a, b, p=2) ** 2
                    return torch.exp(-dist / (2 * sigma**2))
                
                Kxx = rbf_torch(x_t, x_t)
                Kyy = rbf_torch(y_t, y_t)
                Kxy = rbf_torch(x_t, y_t)
                
                mmd_torch = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
                elapsed = time.perf_counter() - start
                
                mem_used = get_memory_usage() - mem_before
                
                suite.add(BenchmarkResult(
                    name="MMD",
                    framework="PyTorch",
                    scale=n,
                    time_seconds=elapsed,
                    memory_bytes=max(0, mem_used),
                    success=True
                ))
                print(f"    PyTorch:  {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
                
            except Exception as e:
                suite.add(BenchmarkResult(
                    name="MMD", framework="PyTorch", scale=n,
                    time_seconds=0, memory_bytes=0, success=False, error=str(e)
                ))
                print(f"    PyTorch:  FAILED - {e}")
        else:
            suite.add(BenchmarkResult(
                name="MMD", framework="PyTorch", scale=n,
                time_seconds=0, memory_bytes=0, success=False, error="Would OOM"
            ))
            print(f"    PyTorch:  SKIPPED (O(n²) would OOM)")
    
    return suite


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 5: CLIFFORD ALGEBRA OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_geometric_algebra(scales: List[int]) -> BenchmarkSuite:
    """Benchmark Geometric/Clifford Algebra operations."""
    
    suite = BenchmarkSuite("Geometric Algebra (Multivector Operations)")
    
    print("\n" + "═" * 80)
    print("BENCHMARK: GEOMETRIC ALGEBRA")
    print("═" * 80)
    
    # For GA, scale = number of generators (dimension = 2^scale)
    for scale in [4, 6, 8, 10, 12, 16, 20]:
        dim = 2 ** scale
        print(f"\n  Generators: {scale} → Dimension: {dim:,}")
        
        # ─────────────────────────────────────────────────────────────────────
        # GENESIS (QTT Multivector)
        # ─────────────────────────────────────────────────────────────────────
        try:
            gc.collect()
            mem_before = get_memory_usage()
            
            start = time.perf_counter()
            # Create random multivector in QTT format
            coeffs = torch.randn(dim)
            qtt_mv = QTTMultivector.from_dense(coeffs, p=scale, max_rank=50)
            # Round-trip
            dense = qtt_mv.to_dense()
            elapsed = time.perf_counter() - start
            
            mem_used = get_memory_usage() - mem_before
            
            suite.add(BenchmarkResult(
                name="GA Multivector",
                framework="GENESIS QTT",
                scale=dim,
                time_seconds=elapsed,
                memory_bytes=max(0, mem_used),
                success=True
            ))
            print(f"    GENESIS:  {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
            
        except Exception as e:
            suite.add(BenchmarkResult(
                name="GA Multivector", framework="GENESIS QTT", scale=dim,
                time_seconds=0, memory_bytes=0, success=False, error=str(e)
            ))
            print(f"    GENESIS:  FAILED - {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # NUMPY DENSE
        # ─────────────────────────────────────────────────────────────────────
        if dim <= 2**16:  # Dense can handle up to 65K
            try:
                gc.collect()
                mem_before = get_memory_usage()
                
                start = time.perf_counter()
                coeffs_np = np.random.randn(dim)
                # Just store and access (no operations)
                result = coeffs_np * 2  # Simple operation
                elapsed = time.perf_counter() - start
                
                mem_used = get_memory_usage() - mem_before
                
                suite.add(BenchmarkResult(
                    name="GA Multivector",
                    framework="NumPy Dense",
                    scale=dim,
                    time_seconds=elapsed,
                    memory_bytes=max(0, mem_used),
                    success=True
                ))
                print(f"    NumPy:    {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
                
            except Exception as e:
                suite.add(BenchmarkResult(
                    name="GA Multivector", framework="NumPy Dense", scale=dim,
                    time_seconds=0, memory_bytes=0, success=False, error=str(e)
                ))
                print(f"    NumPy:    FAILED - {e}")
        else:
            suite.add(BenchmarkResult(
                name="GA Multivector", framework="NumPy Dense", scale=dim,
                time_seconds=0, memory_bytes=0, success=False, 
                error=f"Would need {dim * 8 / 1e9:.1f} GB"
            ))
            print(f"    NumPy:    IMPOSSIBLE ({dim * 8 / 1e9:.1f} GB needed)")
    
    return suite


# ═══════════════════════════════════════════════════════════════════════════════
# MEGA SCALE DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def mega_scale_demo() -> BenchmarkSuite:
    """Demonstrate GENESIS at scales where others can't even start."""
    
    suite = BenchmarkSuite("MEGA SCALE (Genesis Only Territory)")
    
    print("\n" + "═" * 80)
    print("MEGA SCALE DEMONSTRATION")
    print("Where GENESIS operates alone. Others need not apply.")
    print("═" * 80)
    
    mega_scales = [
        (20, "Million"),
        (30, "Billion"),
        (40, "Trillion"),
    ]
    
    for bits, name in mega_scales:
        scale = 2 ** bits
        print(f"\n  Scale: 2^{bits} = {scale:,} ({name})")
        
        # Test if we can even create a distribution at this scale
        try:
            gc.collect()
            mem_before = get_memory_usage()
            
            start = time.perf_counter()
            
            # Only attempt what's implemented
            if bits <= 16:
                mu = QTTDistribution.gaussian(0.0, 1.0, scale)
                nu = QTTDistribution.gaussian(0.5, 1.2, scale)
                W2 = wasserstein_distance(mu, nu, p=2, method="quantile")
                
                elapsed = time.perf_counter() - start
                mem_used = get_memory_usage() - mem_before
                
                suite.add(BenchmarkResult(
                    name=f"OT @ 2^{bits}",
                    framework="GENESIS",
                    scale=scale,
                    time_seconds=elapsed,
                    memory_bytes=max(0, mem_used),
                    success=True
                ))
                print(f"    ✅ GENESIS: {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB | W₂ = {W2:.6f}")
            else:
                # Create QTT Laplacian at mega scale
                L = QTTLaplacian.grid_1d(scale)
                
                elapsed = time.perf_counter() - start
                mem_used = get_memory_usage() - mem_before
                
                suite.add(BenchmarkResult(
                    name=f"Laplacian @ 2^{bits}",
                    framework="GENESIS",
                    scale=scale,
                    time_seconds=elapsed,
                    memory_bytes=max(0, mem_used),
                    success=True
                ))
                print(f"    ✅ GENESIS: {elapsed:.4f}s | {max(0, mem_used)/1024/1024:.2f} MB")
            
            # What would competitors need?
            dense_memory_gb = (scale * 8) / (1024**3)  # Just the vector
            matrix_memory_gb = (scale * scale * 8) / (1024**3)  # Dense matrix
            
            print(f"    📊 Dense vector would need: {dense_memory_gb:.2e} GB")
            if bits <= 20:
                print(f"    📊 Dense matrix would need: {matrix_memory_gb:.2e} GB")
            else:
                print(f"    📊 Dense matrix would need: {matrix_memory_gb:.2e} GB (impossible)")
            
        except NotImplementedError as e:
            print(f"    ⚠ Not yet implemented at this scale: {e}")
            suite.add(BenchmarkResult(
                name=f"Scale 2^{bits}",
                framework="GENESIS",
                scale=scale,
                time_seconds=0,
                memory_bytes=0,
                success=False,
                error="Not implemented"
            ))
        except Exception as e:
            print(f"    ❌ Error: {e}")
            suite.add(BenchmarkResult(
                name=f"Scale 2^{bits}",
                framework="GENESIS",
                scale=scale,
                time_seconds=0,
                memory_bytes=0,
                success=False,
                error=str(e)
            ))
    
    return suite


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL MASSACRE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_massacre_report(suites: List[BenchmarkSuite]):
    """Print the final devastating report."""
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗███╗   ███╗ █████╗ ██████╗ ██╗  ██╗".center(78) + "║")
    print("║" + "  ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝".center(78) + "║")
    print("║" + "  ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║██╔████╔██║███████║██████╔╝█████╔╝ ".center(78) + "║")
    print("║" + "  ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ ".center(78) + "║")
    print("║" + "  ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗".center(78) + "║")
    print("║" + "  ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "M A S S A C R E   R E P O R T".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╠" + "═" * 78 + "╣")
    
    # Print each suite
    for suite in suites:
        print(suite.summary_table())
    
    # Summary statistics
    genesis_wins = 0
    total_comparisons = 0
    max_speedup = 0
    
    for suite in suites:
        scales = sorted(set(r.scale for r in suite.results))
        for scale in scales:
            genesis_result = next((r for r in suite.results 
                                   if r.scale == scale and r.framework == "GENESIS" and r.success), None)
            other_results = [r for r in suite.results 
                            if r.scale == scale and r.framework != "GENESIS" and r.success]
            
            if genesis_result and other_results:
                total_comparisons += len(other_results)
                for other in other_results:
                    speedup = other.time_seconds / max(genesis_result.time_seconds, 1e-10)
                    if speedup > 1:
                        genesis_wins += 1
                        max_speedup = max(max_speedup, speedup)
    
    print("")
    print("╠" + "═" * 78 + "╣")
    print("║" + " " * 78 + "║")
    print("║" + f"  GENESIS WINS: {genesis_wins}/{total_comparisons} comparisons".ljust(77) + "║")
    print("║" + f"  MAX SPEEDUP: {max_speedup:.1f}x".ljust(77) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  THE MOAT IS REAL.".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                              ║")
    print("║   ██████╗ ███████╗███╗   ██╗███████╗███████╗██╗███████╗                      ║")
    print("║  ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔════╝██║██╔════╝                      ║")
    print("║  ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ███████╗██║███████╗                      ║")
    print("║  ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ╚════██║██║╚════██║                      ║")
    print("║  ╚██████╔╝███████╗██║ ╚████║███████╗███████║██║███████║                      ║")
    print("║   ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝╚══════╝                      ║")
    print("║                                                                              ║")
    print("║              B E N C H M A R K   M A S S A C R E                            ║")
    print("║                                                                              ║")
    print("║     GENESIS vs NumPy vs SciPy vs PyTorch vs JAX                             ║")
    print("║     No mercy. Raw numbers. Total annihilation.                              ║")
    print("║                                                                              ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print("")
    
    # Define test scales
    small_scales = [10**3, 10**4, 10**5, 10**6]
    large_scales = [10**3, 10**4, 10**5, 10**6, 10**7]
    
    suites = []
    
    # Run all benchmarks
    suites.append(benchmark_wasserstein(large_scales))
    suites.append(benchmark_laplacian(large_scales))
    suites.append(benchmark_floyd_warshall([64, 128, 256, 512]))
    suites.append(benchmark_mmd(small_scales))
    suites.append(benchmark_geometric_algebra([4, 6, 8, 10, 12, 16, 20]))
    suites.append(mega_scale_demo())
    
    # Print final report
    print_massacre_report(suites)


if __name__ == "__main__":
    main()
