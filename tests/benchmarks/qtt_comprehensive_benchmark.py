#!/usr/bin/env python3
"""
Comprehensive QTT Operations Benchmark Suite
=============================================

Benchmarks ALL QTT operations across the stack:
1. Core QTT Operations (compression, decompression, arithmetic)
2. MPO Operations (construction, application, composition)
3. 2D Operations (Morton encoding, shift, derivatives)
4. Rendering (separable, non-separable, various resolutions)
5. NS2D Solver Components (advection, diffusion, Poisson)
6. Memory & Compression Analysis

Run: python tests/benchmarks/qtt_comprehensive_benchmark.py
"""

import torch
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import Callable
from datetime import datetime

# Enable tensor core acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class BenchmarkResult:
    name: str
    time_ms: float
    throughput: float
    throughput_unit: str
    extra: dict = None


def benchmark_fn(fn: Callable, warmup: int = 5, iterations: int = 20, sync: bool = True) -> float:
    """Benchmark a function, return median time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    
    times = []
    for _ in range(iterations):
        if sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    return np.median(times)


def benchmark_cuda_events(fn: Callable, warmup: int = 5, iterations: int = 50) -> float:
    """Benchmark using CUDA events for GPU-only operations."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iterations):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    return np.median(times)


# =============================================================================
# SECTION 1: Core QTT Operations
# =============================================================================

def benchmark_core_qtt_ops():
    """Benchmark fundamental QTT operations."""
    print("\n" + "=" * 70)
    print("SECTION 1: CORE QTT OPERATIONS")
    print("=" * 70)
    
    from ontic.cfd.pure_qtt_ops import (
        dense_to_qtt, qtt_to_dense, truncate_qtt, 
        qtt_add, qtt_scale, qtt_hadamard, qtt_inner_product, qtt_norm,
        identity_mpo, shift_mpo, apply_mpo, QTTState
    )
    
    results = []
    device = torch.device("cuda")
    
    # Test various sizes
    sizes = [2**10, 2**14, 2**18, 2**20, 2**22]
    ranks = [8, 16, 32, 64]
    
    print("\n1.1 Dense → QTT Compression (TT-SVD)")
    print("-" * 50)
    for size in sizes:
        n_bits = int(np.log2(size))
        
        # Smooth function (compresses well)
        x = torch.linspace(0, 4*np.pi, size, device=device, dtype=torch.float32)
        data_smooth = torch.sin(x) * torch.cos(x/2)
        
        for max_rank in [16, 32]:
            if size > 2**20 and max_rank > 16:
                continue  # Skip expensive combos
            
            time_ms = benchmark_fn(
                lambda d=data_smooth, r=max_rank: dense_to_qtt(d, max_bond=r),
                warmup=2, iterations=5
            )
            
            qtt = dense_to_qtt(data_smooth, max_bond=max_rank)
            qtt_bytes = sum(c.numel() * 4 for c in qtt.cores)
            compression = (size * 4) / qtt_bytes
            
            throughput = size / (time_ms / 1000) / 1e6
            print(f"  2^{n_bits} (rank {max_rank}): {time_ms:.1f}ms, {compression:.1f}x compression, {throughput:.1f} Melem/s")
            
            results.append(BenchmarkResult(
                f"dense_to_qtt_2^{n_bits}_r{max_rank}",
                time_ms, throughput, "Melem/s",
                {"compression": compression, "rank": max_rank}
            ))
    
    print("\n1.2 QTT → Dense Decompression")
    print("-" * 50)
    for size in [2**10, 2**14, 2**18, 2**20]:
        n_bits = int(np.log2(size))
        x = torch.linspace(0, 4*np.pi, size, device=device, dtype=torch.float32)
        data = torch.sin(x)
        qtt = dense_to_qtt(data, max_bond=32)
        
        time_ms = benchmark_fn(lambda q=qtt: qtt_to_dense(q), warmup=3, iterations=10)
        throughput = size / (time_ms / 1000) / 1e6
        print(f"  2^{n_bits}: {time_ms:.2f}ms, {throughput:.1f} Melem/s")
        
        results.append(BenchmarkResult(f"qtt_to_dense_2^{n_bits}", time_ms, throughput, "Melem/s"))
    
    print("\n1.3 QTT Truncation (Rank Reduction)")
    print("-" * 50)
    for size in [2**16, 2**20]:
        n_bits = int(np.log2(size))
        x = torch.linspace(0, 4*np.pi, size, device=device, dtype=torch.float32)
        data = torch.sin(x) + 0.5*torch.sin(3*x) + 0.25*torch.sin(7*x)
        qtt = dense_to_qtt(data, max_bond=64)
        
        for target_rank in [32, 16, 8]:
            time_ms = benchmark_fn(
                lambda q=qtt, r=target_rank: truncate_qtt(q, max_bond=r),
                warmup=3, iterations=10
            )
            print(f"  2^{n_bits} (64→{target_rank}): {time_ms:.2f}ms")
            results.append(BenchmarkResult(f"truncate_2^{n_bits}_to_r{target_rank}", time_ms, 0, ""))
    
    print("\n1.4 QTT Arithmetic Operations")
    print("-" * 50)
    size = 2**18
    n_bits = 18
    x = torch.linspace(0, 4*np.pi, size, device=device, dtype=torch.float32)
    qtt1 = dense_to_qtt(torch.sin(x), max_bond=32)
    qtt2 = dense_to_qtt(torch.cos(x), max_bond=32)
    
    # Addition
    time_ms = benchmark_fn(lambda: qtt_add(qtt1, qtt2, max_bond=32), warmup=3, iterations=10)
    print(f"  qtt_add (2^{n_bits}, r32): {time_ms:.2f}ms")
    results.append(BenchmarkResult("qtt_add_2^18_r32", time_ms, 0, ""))
    
    # Scale
    time_ms = benchmark_fn(lambda: qtt_scale(qtt1, 2.5), warmup=5, iterations=20)
    print(f"  qtt_scale: {time_ms:.3f}ms")
    results.append(BenchmarkResult("qtt_scale_2^18", time_ms, 0, ""))
    
    # Hadamard product
    time_ms = benchmark_fn(lambda: qtt_hadamard(qtt1, qtt2, max_bond=32), warmup=3, iterations=10)
    print(f"  qtt_hadamard: {time_ms:.2f}ms")
    results.append(BenchmarkResult("qtt_hadamard_2^18_r32", time_ms, 0, ""))
    
    # Inner product
    time_ms = benchmark_fn(lambda: qtt_inner_product(qtt1, qtt2), warmup=5, iterations=20)
    print(f"  qtt_inner_product: {time_ms:.3f}ms")
    results.append(BenchmarkResult("qtt_inner_product_2^18", time_ms, 0, ""))
    
    # Norm
    time_ms = benchmark_fn(lambda: qtt_norm(qtt1), warmup=5, iterations=20)
    print(f"  qtt_norm: {time_ms:.3f}ms")
    results.append(BenchmarkResult("qtt_norm_2^18", time_ms, 0, ""))
    
    return results


# =============================================================================
# SECTION 2: MPO Operations
# =============================================================================

def benchmark_mpo_ops():
    """Benchmark MPO construction and application."""
    print("\n" + "=" * 70)
    print("SECTION 2: MPO OPERATIONS")
    print("=" * 70)
    
    from ontic.cfd.pure_qtt_ops import (
        identity_mpo, shift_mpo, derivative_mpo, laplacian_mpo,
        apply_mpo, dense_to_qtt, QTTState
    )
    
    results = []
    device = torch.device("cuda")
    
    print("\n2.1 MPO Construction")
    print("-" * 50)
    for n_bits in [8, 10, 12, 14]:
        # Identity
        time_ms = benchmark_fn(lambda n=n_bits: identity_mpo(n), warmup=3, iterations=10, sync=False)
        print(f"  identity_mpo({n_bits}): {time_ms:.3f}ms")
        results.append(BenchmarkResult(f"identity_mpo_{n_bits}", time_ms, 0, ""))
        
        # Shift
        time_ms = benchmark_fn(lambda n=n_bits: shift_mpo(n, direction=1), warmup=3, iterations=10, sync=False)
        print(f"  shift_mpo({n_bits}): {time_ms:.3f}ms")
        results.append(BenchmarkResult(f"shift_mpo_{n_bits}", time_ms, 0, ""))
    
    print("\n2.2 MPO-QTT Contraction")
    print("-" * 50)
    
    # MPO ops are primarily CPU-based in pure_qtt_ops
    for n_bits in [10, 12, 14]:  # Smaller sizes for CPU ops
        size = 2**n_bits
        x = torch.linspace(0, 4*np.pi, size, dtype=torch.float32)  # CPU
        qtt = dense_to_qtt(x.to(device), max_bond=32)
        
        # Move QTT cores to CPU for MPO application
        qtt_cpu = QTTState(cores=[c.cpu() for c in qtt.cores], num_qubits=n_bits)
        shift = shift_mpo(n_bits, direction=1)
        
        for max_bond in [16, 32]:
            time_ms = benchmark_fn(
                lambda m=shift, q=qtt_cpu, r=max_bond: apply_mpo(m, q, max_bond=r),
                warmup=2, iterations=5, sync=False
            )
            throughput = size / (time_ms / 1000) / 1e6
            print(f"  apply_shift_2^{n_bits}_r{max_bond}: {time_ms:.1f}ms ({throughput:.1f} Melem/s)")
            results.append(BenchmarkResult(
                f"apply_shift_2^{n_bits}_r{max_bond}", time_ms, throughput, "Melem/s"
            ))
    
    return results


# =============================================================================
# SECTION 3: 2D Operations (Morton Encoding, Shifts)
# =============================================================================

def benchmark_2d_ops():
    """Benchmark 2D QTT operations."""
    print("\n" + "=" * 70)
    print("SECTION 3: 2D OPERATIONS")
    print("=" * 70)
    
    from ontic.cfd.qtt_2d import (
        morton_encode_batch, dense_to_qtt_2d, qtt_2d_to_dense,
        shift_mpo_x_2d, shift_mpo_y_2d, apply_mpo_2d, truncate_qtt_2d
    )
    
    results = []
    device = torch.device("cuda")
    
    print("\n3.1 Morton Encoding (Batch)")
    print("-" * 50)
    for size in [256, 512, 1024, 2048, 4096]:
        n_bits = int(np.log2(size))
        x = torch.arange(size, device=device)
        y = torch.arange(size, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        time_ms = benchmark_cuda_events(
            lambda xf=X.flatten(), yf=Y.flatten(), nb=n_bits: morton_encode_batch(xf, yf, nb),
            warmup=5, iterations=20
        )
        throughput = (size * size) / (time_ms / 1000) / 1e6
        print(f"  {size}×{size}: {time_ms:.3f}ms ({throughput:.1f} Mpix/s)")
        results.append(BenchmarkResult(f"morton_encode_{size}x{size}", time_ms, throughput, "Mpix/s"))
    
    print("\n3.2 Dense → QTT 2D")
    print("-" * 50)
    for size in [128, 256, 512, 1024, 2048]:
        n_bits = int(np.log2(size))
        x = torch.linspace(0, 2*np.pi, size, device=device, dtype=torch.float32)
        y = torch.linspace(0, 2*np.pi, size, device=device, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        field = torch.sin(X) * torch.cos(Y)
        
        for max_bond in [16, 32]:
            if size >= 2048 and max_bond > 16:
                continue
            
            time_ms = benchmark_fn(
                lambda f=field, r=max_bond: dense_to_qtt_2d(f, max_bond=r),
                warmup=2, iterations=5
            )
            
            qtt = dense_to_qtt_2d(field, max_bond=max_bond)
            qtt_bytes = sum(c.numel() * 4 for c in qtt.cores)
            compression = (size * size * 4) / qtt_bytes
            
            throughput = (size * size) / (time_ms / 1000) / 1e6
            print(f"  {size}×{size} r{max_bond}: {time_ms:.1f}ms, {compression:.1f}x, {throughput:.1f} Mpix/s")
            results.append(BenchmarkResult(
                f"dense_to_qtt_2d_{size}x{size}_r{max_bond}", time_ms, throughput, "Mpix/s",
                {"compression": compression}
            ))
    
    print("\n3.3 2D Shift MPO Application")
    print("-" * 50)
    print("  (Skipped - apply_mpo_2d has einsum bug to fix)")
    # TODO: Fix einsum subscript in apply_mpo_2d
    
    return results


# =============================================================================
# SECTION 4: Rendering (Separable Contraction)
# =============================================================================

def benchmark_rendering():
    """Benchmark QTT 2D rendering."""
    print("\n" + "=" * 70)
    print("SECTION 4: RENDERING")
    print("=" * 70)
    
    from ontic.sim.visualization.tensor_slicer import TensorSlicer
    from ontic.cfd.qtt_2d import dense_to_qtt_2d
    
    results = []
    device = torch.device("cuda")
    
    resolutions = [
        (640, 480, "480p"),
        (1280, 720, "720p"),
        (1920, 1080, "1080p"),
        (2560, 1440, "1440p"),
        (3840, 2160, "4K"),
        (7680, 4320, "8K"),
    ]
    
    core_configs = [
        (16, 8, "256×256 grid"),
        (20, 16, "1K×1K grid"),
        (24, 32, "4K×4K grid"),
    ]
    
    print("\n4.1 Separable Rendering (Disjoint x/y cores)")
    print("-" * 50)
    
    for n_cores, rank, desc in core_configs:
        # Create test QTT
        cores = []
        for i in range(n_cores):
            r_left = 1 if i == 0 else rank
            r_right = 1 if i == n_cores - 1 else rank
            cores.append(np.random.randn(r_left, 2, r_right).astype(np.float32) * 0.1)
        
        slicer = TensorSlicer(cores)
        n_half = n_cores // 2
        x_cores = list(range(n_half))
        y_cores = list(range(n_half, n_cores))
        
        print(f"\n  {desc} ({n_cores} cores, rank {rank}):")
        
        for width, height, label in resolutions:
            # Warmup
            for _ in range(5):
                _ = slicer.render_slice_2d_gpu_tensor(x_cores, y_cores, resolution=(width, height), validate=False)
            torch.cuda.synchronize()
            
            # Cold
            slicer.invalidate_cache()
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = slicer.render_slice_2d_gpu_tensor(x_cores, y_cores, resolution=(width, height), validate=False)
            torch.cuda.synchronize()
            cold_ms = (time.perf_counter() - start) * 1000
            
            # Warm (cached)
            cached_ms = benchmark_cuda_events(
                lambda: slicer.render_slice_2d_gpu_tensor(x_cores, y_cores, resolution=(width, height), validate=False),
                warmup=0, iterations=50
            )
            
            fps = 1000 / cached_ms
            mpix_s = (width * height) / (cached_ms / 1000) / 1e6
            
            print(f"    {label:>6}: {cached_ms:.3f}ms ({fps:,.0f} FPS, {mpix_s:.0f} Mpix/s), cold: {cold_ms:.2f}ms")
            
            results.append(BenchmarkResult(
                f"render_{desc.replace(' ', '_')}_{label}",
                cached_ms, fps, "FPS",
                {"cold_ms": cold_ms, "mpix_s": mpix_s, "resolution": f"{width}x{height}"}
            ))
    
    return results


# =============================================================================
# SECTION 5: NS2D Solver Components
# =============================================================================

def benchmark_ns2d_components():
    """Benchmark Navier-Stokes solver components."""
    print("\n" + "=" * 70)
    print("SECTION 5: NS2D SOLVER COMPONENTS")
    print("=" * 70)
    
    from ontic.cfd.ns2d_qtt_native import (
        NS2DQTTConfig, NS2D_QTT_Native, dense_to_qtt_2d_native
    )
    
    results = []
    
    grid_configs = [
        (10, 10, "1K×1K"),   # 1024×1024
        (11, 11, "2K×2K"),   # 2048×2048
        (12, 12, "4K×4K"),   # 4096×4096
    ]
    
    for nx_bits, ny_bits, desc in grid_configs:
        print(f"\n{desc} grid ({2**nx_bits}×{2**ny_bits} = {2**(nx_bits+ny_bits):,} cells)")
        print("-" * 50)
        
        config = NS2DQTTConfig(
            nx_bits=nx_bits, ny_bits=ny_bits,
            Lx=1.0, Ly=1.0, nu=1e-4, cfl=0.3,
            max_rank=32, dtype=torch.float32
        )
        
        # Solver construction
        start = time.perf_counter()
        solver = NS2D_QTT_Native(config)
        build_ms = (time.perf_counter() - start) * 1000
        print(f"  Solver build: {build_ms:.1f}ms")
        results.append(BenchmarkResult(f"ns2d_build_{desc}", build_ms, 0, ""))
        
        # IC generation
        device = config.device
        nx, ny = 2**nx_bits, 2**ny_bits
        x = torch.linspace(0, config.Lx, nx, device=device, dtype=config.dtype)
        y = torch.linspace(0, config.Ly, ny, device=device, dtype=config.dtype)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        omega_dense = 2.0 * torch.sin(2*np.pi*X) * torch.sin(2*np.pi*Y)
        psi_dense = torch.sin(2*np.pi*X) * torch.sin(2*np.pi*Y) / (8 * np.pi**2)
        
        time_ms = benchmark_fn(
            lambda: dense_to_qtt_2d_native(omega_dense, nx_bits, ny_bits, config.max_rank),
            warmup=1, iterations=3
        )
        throughput = (nx * ny) / (time_ms / 1000) / 1e6
        print(f"  IC compression: {time_ms:.1f}ms ({throughput:.1f} Mpix/s)")
        results.append(BenchmarkResult(f"ns2d_compress_{desc}", time_ms, throughput, "Mpix/s"))
        
        omega = dense_to_qtt_2d_native(omega_dense, nx_bits, ny_bits, config.max_rank)
        psi = dense_to_qtt_2d_native(psi_dense, nx_bits, ny_bits, config.max_rank)
        
        del omega_dense, psi_dense, X, Y
        torch.cuda.empty_cache()
        
        dt = config.cfl * min(config.dx, config.dy) / 1.0
        
        # Full time step (only for smaller grids - expensive)
        if nx_bits <= 11:
            times = []
            for i in range(3):
                torch.cuda.synchronize()
                start = time.perf_counter()
                omega, psi = solver.step(omega, psi, dt=dt)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            step_ms = np.median(times)
            print(f"  Full time step: {step_ms:.1f}ms")
            results.append(BenchmarkResult(f"ns2d_step_{desc}", step_ms, 0, ""))
        else:
            print(f"  Full time step: (skipped - too expensive for benchmark)")
    
    return results


# =============================================================================
# SECTION 6: Memory & Compression Analysis
# =============================================================================

def benchmark_memory_compression():
    """Analyze memory usage and compression ratios."""
    print("\n" + "=" * 70)
    print("SECTION 6: MEMORY & COMPRESSION ANALYSIS")
    print("=" * 70)
    
    from ontic.cfd.pure_qtt_ops import dense_to_qtt
    from ontic.cfd.qtt_2d import dense_to_qtt_2d
    
    results = []
    device = torch.device("cuda")
    
    print("\n6.1 1D Compression by Data Type")
    print("-" * 50)
    
    size = 2**20
    n_bits = 20
    x = torch.linspace(0, 4*np.pi, size, device=device, dtype=torch.float32)
    
    data_types = [
        ("Constant", torch.ones_like(x)),
        ("Linear", x),
        ("Smooth sin", torch.sin(x)),
        ("Multi-freq", torch.sin(x) + 0.5*torch.sin(3*x) + 0.25*torch.sin(7*x)),
        ("Gaussian", torch.exp(-((x - 2*np.pi)**2) / 0.5)),
        ("Step function", (x > 2*np.pi).float()),
        ("Random", torch.randn_like(x)),
    ]
    
    for name, data in data_types:
        for max_rank in [8, 16, 32, 64]:
            qtt = dense_to_qtt(data, max_bond=max_rank)
            qtt_bytes = sum(c.numel() * 4 for c in qtt.cores)
            actual_rank = max(max(c.shape[0], c.shape[2]) for c in qtt.cores)
            compression = (size * 4) / qtt_bytes
            
            # Reconstruction error
            recon = torch.zeros(size, device=device, dtype=torch.float32)
            # Quick sampling
            for idx in range(0, size, size//100):
                result = None
                binary = format(idx, f'0{n_bits}b')
                for i, bit in enumerate(binary):
                    mat = qtt.cores[i][:, int(bit), :]
                    if result is None:
                        result = mat
                    else:
                        result = result @ mat
                recon[idx] = result.squeeze()
            
            samples = torch.tensor([data[i] for i in range(0, size, size//100)], device=device)
            recon_samples = recon[::size//100]
            rel_error = torch.norm(recon_samples - samples) / (torch.norm(samples) + 1e-10)
            
            if max_rank == 32:  # Print only one rank per data type
                print(f"  {name:15s} r{max_rank}: {compression:6.1f}x, actual_r={actual_rank:2d}, err={rel_error:.1e}")
        
        results.append(BenchmarkResult(
            f"compression_{name.replace(' ', '_')}_r32",
            0, compression, "x",
            {"actual_rank": actual_rank}
        ))
    
    print("\n6.2 2D Compression")
    print("-" * 50)
    
    for size in [256, 512, 1024, 2048]:
        n_bits = int(np.log2(size))
        x = torch.linspace(0, 2*np.pi, size, device=device, dtype=torch.float32)
        y = torch.linspace(0, 2*np.pi, size, device=device, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        data_2d = [
            ("Smooth", torch.sin(X) * torch.cos(Y)),
            ("Multi-scale", torch.sin(X)*torch.cos(Y) + 0.3*torch.sin(3*X)*torch.cos(3*Y)),
            ("Gaussian", torch.exp(-((X-np.pi)**2 + (Y-np.pi)**2))),
        ]
        
        for name, field in data_2d:
            qtt = dense_to_qtt_2d(field, max_bond=32)
            qtt_bytes = sum(c.numel() * 4 for c in qtt.cores)
            dense_bytes = size * size * 4
            compression = dense_bytes / qtt_bytes
            
            print(f"  {size}×{size} {name:12s}: {qtt_bytes/1024:.1f} KB, {compression:.1f}x")
            
            results.append(BenchmarkResult(
                f"compress_2d_{size}_{name.replace(' ', '_')}",
                0, compression, "x",
                {"qtt_kb": qtt_bytes/1024, "dense_mb": dense_bytes/1024/1024}
            ))
    
    return results


# =============================================================================
# SECTION 7: Point Evaluation
# =============================================================================

def benchmark_point_evaluation():
    """Benchmark single and batch point evaluation."""
    print("\n" + "=" * 70)
    print("SECTION 7: POINT EVALUATION")
    print("=" * 70)
    
    from ontic.sim.visualization.tensor_slicer import TensorSlicer
    from ontic.cfd.pure_qtt_ops import dense_to_qtt
    
    results = []
    device = torch.device("cuda")
    
    print("\n7.1 CPU Single Point Evaluation")
    print("-" * 50)
    
    for n_bits in [12, 16, 20, 24]:
        size = 2**n_bits
        # Create QTT
        cores = []
        rank = 16
        for i in range(n_bits):
            r_left = 1 if i == 0 else rank
            r_right = 1 if i == n_bits - 1 else rank
            cores.append(np.random.randn(r_left, 2, r_right).astype(np.float32) * 0.1)
        
        slicer = TensorSlicer(cores)
        
        # Single point
        times = []
        for _ in range(100):
            idx = np.random.randint(0, size)
            start = time.perf_counter()
            _ = slicer.get_element(idx)
            times.append((time.perf_counter() - start) * 1000)
        
        time_us = np.median(times) * 1000
        print(f"  2^{n_bits} (r{rank}): {time_us:.1f}μs per point")
        results.append(BenchmarkResult(f"eval_single_2^{n_bits}", time_us/1000, 0, ""))
    
    print("\n7.2 GPU Batch Point Evaluation")
    print("-" * 50)
    
    for n_bits in [16, 20, 24]:
        size = 2**n_bits
        cores = []
        rank = 16
        for i in range(n_bits):
            r_left = 1 if i == 0 else rank
            r_right = 1 if i == n_bits - 1 else rank
            cores.append(np.random.randn(r_left, 2, r_right).astype(np.float32) * 0.1)
        
        slicer = TensorSlicer(cores)
        
        batch_sizes = [100, 1000, 10000, 100000]
        for batch in batch_sizes:
            if batch > size:
                continue
            indices = np.random.randint(0, size, batch)
            
            time_ms = benchmark_fn(
                lambda idx=indices: slicer.get_elements_batch_gpu(idx),
                warmup=3, iterations=10
            )
            
            throughput = batch / (time_ms / 1000) / 1e6
            print(f"  2^{n_bits} batch={batch:>6}: {time_ms:.2f}ms ({throughput:.1f} Mpts/s)")
            
            results.append(BenchmarkResult(
                f"eval_batch_2^{n_bits}_b{batch}",
                time_ms, throughput, "Mpts/s"
            ))
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def run_all_benchmarks():
    """Run complete benchmark suite."""
    print("=" * 70)
    print("QTT COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    all_results = []
    
    # Run all sections
    all_results.extend(benchmark_core_qtt_ops())
    all_results.extend(benchmark_mpo_ops())
    all_results.extend(benchmark_2d_ops())
    all_results.extend(benchmark_rendering())
    all_results.extend(benchmark_ns2d_components())
    all_results.extend(benchmark_memory_compression())
    all_results.extend(benchmark_point_evaluation())
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total benchmarks run: {len(all_results)}")
    
    # Categorize
    categories = {}
    for r in all_results:
        cat = r.name.split("_")[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    print(f"Categories: {list(categories.keys())}")
    
    # Save results
    output_path = "/home/brad/TiganticLabz/Main_Projects/physics-os/tests/benchmarks/qtt_comprehensive_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "results": [asdict(r) for r in all_results]
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
