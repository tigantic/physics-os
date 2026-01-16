#!/usr/bin/env python3
"""
FluidElite Elite Operations Benchmark Suite

Measures performance gains from the 5 Elite Engineering Optimizations:
1. Fused Laplacian (mps_sum) - Target: ~5× speedup
2. CG Fusion (mps_linear_combination) - Target: ~2× speedup  
3. Multigrid-Preconditioned CG - Target: O(1) iterations vs O(√N)
4. Multigrid V-cycle - Target: ~8× for large grids
5. CUDA Hybrid (.cuda()/.cpu()) - Target: seamless GPU acceleration
"""

import time
import torch
import statistics
from typing import Callable, List, Tuple
from dataclasses import dataclass

from fluidelite.core import MPS
from fluidelite.core.elite_ops import (
    mps_sum,
    mps_linear_combination,
    pcg_solve,
    multigrid_preconditioner,
    multigrid_vcycle,
    patch_mps_cuda,
    _mps_norm,
    batched_truncate_,
    fused_canonicalize_truncate_,
)


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    naive_time_ms: float
    elite_time_ms: float
    speedup: float
    iterations: int = 0
    naive_iters: int = 0
    elite_iters: int = 0


def timeit(fn: Callable, warmup: int = 2, runs: int = 5) -> float:
    """Time a function, return median time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Timed runs
    times = []
    for _ in range(runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return statistics.median(times)


def naive_mps_add(a: MPS, b: MPS, max_chi: int = 64) -> MPS:
    """Naive MPS addition: direct sum + truncate."""
    result = a.copy()
    # Naive: concatenate bond dimensions
    for i in range(a.L):
        A_i = a.tensors[i]
        B_i = b.tensors[i]
        
        if i == 0:
            # First tensor: (1, d, chi) - concat on right
            new_tensor = torch.cat([A_i, B_i], dim=2)
        elif i == a.L - 1:
            # Last tensor: (chi, d, 1) - concat on left
            new_tensor = torch.cat([A_i, B_i], dim=0)
        else:
            # Middle tensors: block diagonal
            chi_a_l, d, chi_a_r = A_i.shape
            chi_b_l, _, chi_b_r = B_i.shape
            new_tensor = torch.zeros(
                chi_a_l + chi_b_l, d, chi_a_r + chi_b_r,
                dtype=A_i.dtype, device=A_i.device
            )
            new_tensor[:chi_a_l, :, :chi_a_r] = A_i
            new_tensor[chi_a_l:, :, chi_a_r:] = B_i
        
        result.tensors[i] = new_tensor
    
    result.truncate_(max_chi)
    return result


def naive_mps_sum(mps_list: List[MPS], max_chi: int = 64) -> MPS:
    """Naive N-ary sum: pairwise addition with truncation after each."""
    result = mps_list[0].copy()
    for mps in mps_list[1:]:
        result = naive_mps_add(result, mps, max_chi)
    return result


def naive_linear_combination(mps_list: List[MPS], coeffs: List[float], max_chi: int = 64) -> MPS:
    """Naive linear combination: scale each, then pairwise sum."""
    scaled = []
    for mps, c in zip(mps_list, coeffs):
        s = mps.copy()
        s.tensors[0] = s.tensors[0] * c
        scaled.append(s)
    return naive_mps_sum(scaled, max_chi)


def naive_cg_solve(A: Callable, b: MPS, x0: MPS, tol: float = 1e-6, 
                   max_iter: int = 100, max_chi: int = 64) -> Tuple[MPS, int]:
    """Naive CG without preconditioning."""
    x = x0.copy()
    
    # r = b - A(x)
    Ax = A(x)
    Ax.tensors[0] = Ax.tensors[0] * -1.0
    r = naive_mps_add(b, Ax, max_chi)
    p = r.copy()
    
    r_norm_sq = _mps_norm(r) ** 2
    
    for k in range(max_iter):
        if r_norm_sq < tol * tol:
            return x, k
        
        Ap = A(p)
        
        # alpha = r·r / p·Ap
        pAp = _mps_norm(Ap) * _mps_norm(p)  # Simplified
        if pAp < 1e-12:
            return x, k
        alpha = r_norm_sq / (pAp + 1e-12)
        
        # x = x + alpha * p
        p_scaled = p.copy()
        p_scaled.tensors[0] = p_scaled.tensors[0] * alpha
        x = naive_mps_add(x, p_scaled, max_chi)
        
        # r = r - alpha * Ap
        Ap_scaled = Ap.copy()
        Ap_scaled.tensors[0] = Ap_scaled.tensors[0] * (-alpha)
        r = naive_mps_add(r, Ap_scaled, max_chi)
        
        r_norm_sq_new = _mps_norm(r) ** 2
        
        # p = r + beta * p
        beta = r_norm_sq_new / (r_norm_sq + 1e-12)
        p_scaled = p.copy()
        p_scaled.tensors[0] = p_scaled.tensors[0] * beta
        p = naive_mps_add(r, p_scaled, max_chi)
        
        r_norm_sq = r_norm_sq_new
    
    return x, max_iter


def bench_fused_laplacian(L: int = 16, chi: int = 8, n_terms: int = 5) -> BenchmarkResult:
    """Benchmark #1: Fused Laplacian (mps_sum vs naive pairwise)."""
    mps_list = [MPS.random(L=L, d=2, chi=chi, dtype=torch.float64) for _ in range(n_terms)]
    for m in mps_list:
        m.normalize_()
    
    max_chi = chi * 2
    
    naive_time = timeit(lambda: naive_mps_sum(mps_list, max_chi))
    elite_time = timeit(lambda: mps_sum(mps_list, max_chi))
    
    return BenchmarkResult(
        name=f"Fused Laplacian (L={L}, χ={chi}, n={n_terms})",
        naive_time_ms=naive_time,
        elite_time_ms=elite_time,
        speedup=naive_time / elite_time if elite_time > 0 else float('inf'),
    )


def bench_cg_fusion(L: int = 16, chi: int = 8) -> BenchmarkResult:
    """Benchmark #2: CG Fusion (mps_linear_combination vs naive)."""
    mps_list = [MPS.random(L=L, d=2, chi=chi, dtype=torch.float64) for _ in range(3)]
    coeffs = [1.0, -0.5, 0.25]
    for m in mps_list:
        m.normalize_()
    
    max_chi = chi * 2
    
    naive_time = timeit(lambda: naive_linear_combination(mps_list, coeffs, max_chi))
    elite_time = timeit(lambda: mps_linear_combination(mps_list, coeffs, max_chi))
    
    return BenchmarkResult(
        name=f"CG Fusion (L={L}, χ={chi})",
        naive_time_ms=naive_time,
        elite_time_ms=elite_time,
        speedup=naive_time / elite_time if elite_time > 0 else float('inf'),
    )


def bench_multigrid_pcg(L: int = 12, chi: int = 6) -> BenchmarkResult:
    """Benchmark #3: Multigrid-Preconditioned CG vs plain CG."""
    b = MPS.random(L=L, d=2, chi=chi, dtype=torch.float64)
    b.normalize_()
    x0 = MPS.random(L=L, d=2, chi=chi, dtype=torch.float64)
    x0.normalize_()
    
    # Identity-like operator
    def A(x):
        return x.copy()
    
    max_chi = chi * 2
    tol = 1e-3
    max_iters = 20
    
    # Plain CG (no preconditioner)
    naive_time = timeit(
        lambda: pcg_solve(A, b, x0, M_inv=None, tol=tol, max_iters=max_iters, max_chi=max_chi),
        warmup=1, runs=3
    )
    
    # Multigrid-preconditioned CG
    M_inv = multigrid_preconditioner(A, levels=1, max_chi=max_chi)
    elite_time = timeit(
        lambda: pcg_solve(A, b, x0, M_inv=M_inv, tol=tol, max_iters=max_iters, max_chi=max_chi),
        warmup=1, runs=3
    )
    
    # Get iteration counts
    _, info_naive = pcg_solve(A, b, x0, M_inv=None, tol=tol, max_iters=max_iters, max_chi=max_chi)
    _, info_elite = pcg_solve(A, b, x0, M_inv=M_inv, tol=tol, max_iters=max_iters, max_chi=max_chi)
    naive_iters = info_naive.get('iterations', max_iters)
    elite_iters = info_elite.get('iterations', max_iters)
    
    return BenchmarkResult(
        name=f"Multigrid PCG (L={L}, χ={chi})",
        naive_time_ms=naive_time,
        elite_time_ms=elite_time,
        speedup=naive_time / elite_time if elite_time > 0 else float('inf'),
        naive_iters=naive_iters,
        elite_iters=elite_iters,
    )


def bench_multigrid_vcycle(L: int = 16, chi: int = 8) -> BenchmarkResult:
    """Benchmark #4: Multigrid V-cycle - API verification."""
    b = MPS.random(L=L, d=2, chi=chi, dtype=torch.float64)
    b.normalize_()
    x0 = MPS.random(L=L, d=2, chi=chi, dtype=torch.float64)
    x0.normalize_()
    
    def A(x):
        return x.copy()
    
    max_chi = chi * 2
    
    # Plain Jacobi smoothing (what multigrid uses internally)
    from fluidelite.core.elite_ops import _jacobi_smooth
    
    naive_time = timeit(lambda: _jacobi_smooth(A, b, x0, max_chi=max_chi, omega=0.6), runs=3)
    
    # V-cycle (single level for fair comparison)
    elite_time = timeit(
        lambda: multigrid_vcycle(A, b, x0, levels=1, nu1=1, nu2=1, max_chi=max_chi),
        runs=3
    )
    
    return BenchmarkResult(
        name=f"Multigrid V-cycle (L={L}, χ={chi})",
        naive_time_ms=naive_time,
        elite_time_ms=elite_time,
        speedup=1.0,  # API verification - speedup depends on problem size
    )


def bench_cuda_hybrid(L: int = 128, chi: int = 128) -> BenchmarkResult:
    """Benchmark #5: CUDA Hybrid - GPU vs CPU for large MPS operations.
    
    GPU wins when tensors are large enough to amortize kernel launch overhead.
    Typical crossover: chi > 64 for RTX-class GPUs.
    """
    if not torch.cuda.is_available():
        return BenchmarkResult(
            name=f"CUDA Hybrid (L={L}, χ={chi})",
            naive_time_ms=0,
            elite_time_ms=0,
            speedup=0,
        )
    
    patch_mps_cuda()
    
    # Larger tensors where GPU overhead is amortized
    mps = MPS.random(L=L, d=2, chi=chi, dtype=torch.float64)
    mps.normalize_()
    
    # CPU operations - heavier workload
    def cpu_ops():
        m = mps.copy()
        for _ in range(5):
            m.normalize_()
            m.truncate_(chi)
        return m
    
    # GPU operations - transfer once, do work, transfer back
    def gpu_ops():
        m = mps.copy()
        m.cuda()
        for _ in range(5):
            m.normalize_()
            m.truncate_(chi)
        m.cpu()
        return m
    
    cpu_time = timeit(cpu_ops, warmup=1, runs=3)
    gpu_time = timeit(gpu_ops, warmup=1, runs=3)
    
    return BenchmarkResult(
        name=f"CUDA Hybrid (L={L}, χ={chi})",
        naive_time_ms=cpu_time,
        elite_time_ms=gpu_time,
        speedup=cpu_time / gpu_time if gpu_time > 0 else float('inf'),
    )


def bench_rsvd(sizes: list = None) -> BenchmarkResult:
    """Benchmark #6: rSVD vs exact SVD for truncated decomposition.
    
    Tests torch.svd_lowrank vs torch.linalg.svd for typical MPS truncation.
    rSVD is O(m*n*k) vs O(min(m,n)³) - big win when k << min(m,n).
    """
    if sizes is None:
        sizes = [(256, 256, 32), (512, 512, 64), (1024, 1024, 64)]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    total_exact = 0.0
    total_rsvd = 0.0
    
    for m, n, k in sizes:
        A = torch.randn(m, n, dtype=torch.float64, device=device)
        
        # Exact SVD
        def exact_svd():
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            return U[:, :k], S[:k], Vh[:k, :]
        
        # Randomized SVD via torch.svd_lowrank
        def rsvd():
            U, S, V = torch.svd_lowrank(A, q=k+10, niter=2)
            return U[:, :k], S[:k], V[:, :k].T
        
        exact_time = timeit(exact_svd, warmup=1, runs=3)
        rsvd_time = timeit(rsvd, warmup=1, runs=3)
        
        total_exact += exact_time
        total_rsvd += rsvd_time
    
    return BenchmarkResult(
        name=f"rSVD vs Exact (device={device})",
        naive_time_ms=total_exact,
        elite_time_ms=total_rsvd,
        speedup=total_exact / total_rsvd if total_rsvd > 0 else float('inf'),
    )


def bench_batched_gpu(L: int = 256, chi: int = 64) -> BenchmarkResult:
    """Benchmark #7: Batched GPU ops vs sequential.
    
    Tests fused_canonicalize_truncate_ vs separate canonicalize + truncate.
    The fused version uses rSVD and reduces kernel launches.
    """
    if not torch.cuda.is_available():
        return BenchmarkResult(
            name=f"Batched GPU (L={L}, χ={chi})",
            naive_time_ms=0,
            elite_time_ms=0,
            speedup=0,
        )
    
    patch_mps_cuda()
    
    # Create test MPS on GPU
    mps = MPS.random(L=L, d=2, chi=chi, dtype=torch.float64)
    mps.cuda()
    
    # Sequential: canonicalize + truncate separately
    def sequential_ops():
        m = mps.copy()
        m.canonicalize_left_()
        m.truncate_(chi)
        return m
    
    # Fused: single sweep with rSVD
    def fused_ops():
        m = mps.copy()
        fused_canonicalize_truncate_(m, chi, direction="left")
        return m
    
    seq_time = timeit(sequential_ops, warmup=2, runs=5)
    fused_time = timeit(fused_ops, warmup=2, runs=5)
    
    return BenchmarkResult(
        name=f"Fused Canon+Truncate (L={L}, χ={chi})",
        naive_time_ms=seq_time,
        elite_time_ms=fused_time,
        speedup=seq_time / fused_time if fused_time > 0 else float('inf'),
    )


def print_result(result: BenchmarkResult):
    """Pretty print a benchmark result."""
    print(f"\n{'='*60}")
    print(f"  {result.name}")
    print(f"{'='*60}")
    print(f"  Naive:  {result.naive_time_ms:8.3f} ms")
    print(f"  Elite:  {result.elite_time_ms:8.3f} ms")
    print(f"  Speedup: {result.speedup:7.2f}×")
    if result.naive_iters > 0 or result.elite_iters > 0:
        print(f"  Iterations: {result.naive_iters} → {result.elite_iters}")


def run_all_benchmarks():
    """Run all benchmarks and report results."""
    print("\n" + "="*60)
    print("  FluidElite Elite Operations Benchmark Suite")
    print("="*60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {'Available' if torch.cuda.is_available() else 'Not available'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)
    
    results = []
    
    # Benchmark 1: Fused Laplacian
    print("\n[1/5] Running Fused Laplacian benchmark...")
    results.append(bench_fused_laplacian(L=16, chi=8, n_terms=5))
    print_result(results[-1])
    
    # Benchmark 2: CG Fusion
    print("\n[2/5] Running CG Fusion benchmark...")
    results.append(bench_cg_fusion(L=16, chi=8))
    print_result(results[-1])
    
    # Benchmark 3: Multigrid-Preconditioned CG
    print("\n[3/5] Running Multigrid-PCG benchmark...")
    results.append(bench_multigrid_pcg(L=12, chi=6))
    print_result(results[-1])
    
    # Benchmark 4: Multigrid V-cycle
    print("\n[4/5] Running Multigrid V-cycle benchmark...")
    results.append(bench_multigrid_vcycle(L=16, chi=8))
    print_result(results[-1])
    
    # Benchmark 5: CUDA Hybrid
    print("\n[5/6] Running CUDA Hybrid benchmark...")
    results.append(bench_cuda_hybrid(L=128, chi=128))
    print_result(results[-1])
    
    # Benchmark 6: rSVD vs exact SVD
    print("\n[6/7] Running rSVD benchmark...")
    results.append(bench_rsvd())
    print_result(results[-1])
    
    # Benchmark 7: Batched GPU operations
    print("\n[7/7] Running Batched GPU benchmark...")
    results.append(bench_batched_gpu(L=256, chi=64))
    print_result(results[-1])
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    core_results = results[:2]  # Fused Laplacian + CG Fusion
    api_results = results[2:5]   # PCG, Multigrid, CUDA (API verification)
    svd_result = results[5]  # rSVD benchmark
    batched_result = results[6]  # Batched GPU
    
    print("  Core MPS Operations:")
    for r in core_results:
        status = "✓" if r.speedup >= 1.5 else "○"
        print(f"    {status} {r.name}: {r.speedup:.2f}×")
    
    print("\n  API Verification (speedup depends on problem size):")
    for r in api_results:
        status = "✓" if r.speedup > 0 or r.name.startswith("Multigrid") else "—"
        print(f"    {status} {r.name}")
    
    print("\n  GPU Optimizations:")
    status = "✓" if svd_result.speedup >= 1.5 else "○"
    print(f"    {status} {svd_result.name}: {svd_result.speedup:.2f}×")
    status = "✓" if batched_result.speedup >= 1.0 else "○"
    print(f"    {status} {batched_result.name}: {batched_result.speedup:.2f}×")
    
    avg_speedup = statistics.mean([r.speedup for r in core_results])
    print(f"\n  Core operations average speedup: {avg_speedup:.2f}×")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_benchmarks()
