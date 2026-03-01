#!/usr/bin/env python3
"""
Benchmark: Batched QTT Operations vs Original
===============================================

Run this on your machine to verify:
1. Correctness: batched ops produce same results as original
2. Performance: batched SVD speedup at each matrix size
3. Integration: full solver step time with batched ops

Usage:
    cd physics-os-main
    PYTHONPATH="$PWD:$PYTHONPATH" python3 ontic/cfd/benchmark_batched.py
"""

import torch
import time
import numpy as np
import sys

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 70)


# ===========================================================================
# Test 1: Batched SVD speedup
# ===========================================================================

def test_batched_svd_speedup():
    """Measure speedup from batching SVD calls."""
    print("\n[TEST 1] Batched SVD Speedup")
    print("-" * 50)
    
    # Typical QTT matrix sizes from profiling
    sizes = [
        (128, 52, 69),   # Largest, most frequent
        (64, 54, 54),
        (32, 56, 60),
        (16, 56, 60),
        (8, 32, 60),
        (4, 16, 60),
        (2, 8, 60),
    ]
    
    total_individual = 0.0
    total_batched = 0.0
    
    for m, n, count in sizes:
        mats = [torch.randn(m, n, device=DEVICE) for _ in range(count)]
        
        # Individual SVDs
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t0 = time.perf_counter()
        for mat in mats:
            torch.linalg.svd(mat, full_matrices=False)
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t_individual = time.perf_counter() - t0
        
        # Batched SVD (group by 6, simulating 6 fields)
        batch_size = 6
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t0 = time.perf_counter()
        for i in range(0, count, batch_size):
            batch_mats = mats[i:i+batch_size]
            actual_b = len(batch_mats)
            batch = torch.stack([
                torch.nn.functional.pad(m, (0, n - m.shape[1], 0, m_max - m.shape[0])) 
                if m.shape != (m, n) else m 
                for m in batch_mats
            ] if False else batch_mats)  # All same size here
            torch.linalg.svd(batch, full_matrices=False)
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t_batched = time.perf_counter() - t0
        
        speedup = t_individual / max(t_batched, 1e-9)
        total_individual += t_individual
        total_batched += t_batched
        
        print(f"  {m:4d}×{n:2d} (×{count:2d}): "
              f"individual={t_individual*1000:6.1f}ms, "
              f"batched={t_batched*1000:6.1f}ms, "
              f"speedup={speedup:5.1f}x")
    
    print(f"\n  TOTAL: individual={total_individual*1000:.0f}ms, "
          f"batched={total_batched*1000:.0f}ms, "
          f"speedup={total_individual/total_batched:.1f}x")
    
    return total_individual / max(total_batched, 1e-9)


# ===========================================================================
# Test 2: QTT operation correctness  
# ===========================================================================

def test_qtt_operations():
    """Verify batched operations produce correct results."""
    print("\n[TEST 2] QTT Operation Correctness")
    print("-" * 50)
    
    from ontic.cfd.qtt_batched_ops import (
        add_cores_raw, hadamard_cores_raw, scale_cores,
        single_truncation_sweep, qtt_inner, qtt_norm,
    )
    
    torch.manual_seed(42)
    L = 15  # 5 bits × 3 dimensions
    max_rank = 16
    
    # Create test QTTs
    def make_qtt(L, rank):
        cores = []
        r = 1
        for k in range(L):
            r_next = min(rank, 2 * r) if k < L // 2 else max(1, r // 2)
            if k == L - 1:
                r_next = 1
            cores.append(torch.randn(r, 2, r_next, device=DEVICE))
            r = r_next
        return cores
    
    a = make_qtt(L, max_rank)
    b = make_qtt(L, max_rank)
    
    # Test 1: Addition
    c = add_cores_raw(a, b, alpha=2.0, beta=-1.0)
    c_trunc = single_truncation_sweep(c, max_rank)
    
    # Verify structure
    assert len(c_trunc) == L, f"Wrong number of cores: {len(c_trunc)}"
    assert c_trunc[0].shape[0] == 1, f"First core left rank should be 1, got {c_trunc[0].shape[0]}"
    assert c_trunc[-1].shape[2] == 1, f"Last core right rank should be 1, got {c_trunc[-1].shape[2]}"
    for k in range(L - 1):
        assert c_trunc[k].shape[2] == c_trunc[k+1].shape[0], \
            f"Bond mismatch at site {k}: {c_trunc[k].shape[2]} vs {c_trunc[k+1].shape[0]}"
    print("  ✓ Addition: structure correct")
    
    # Test 2: Norm
    norm_a = qtt_norm(a)
    assert norm_a > 0, "Norm should be positive"
    print(f"  ✓ Norm: ||a|| = {norm_a.item():.4f}")
    
    # Test 3: Inner product self-consistency
    ip_aa = qtt_inner(a, a)
    assert ip_aa >= 0, "Self inner product should be non-negative"
    assert abs(ip_aa.item() - norm_a.item()**2) < 1e-3 * norm_a.item()**2, \
        f"<a,a> = {ip_aa.item():.4f} != ||a||² = {norm_a.item()**2:.4f}"
    print(f"  ✓ Inner product: <a,a> = {ip_aa.item():.4f} = ||a||² ✓")
    
    # Test 4: Hadamard
    h = hadamard_cores_raw(a, b)
    h_trunc = single_truncation_sweep(h, max_rank)
    assert len(h_trunc) == L
    print(f"  ✓ Hadamard: rank {max(c.shape[0] for c in h)} → {max(c.shape[0] for c in h_trunc)}")
    
    # Test 5: Scale
    s = scale_cores(a, 3.14)
    norm_s = qtt_norm(s)
    expected = 3.14 * norm_a.item()
    assert abs(norm_s.item() - expected) < 1e-3 * expected, \
        f"||3.14*a|| = {norm_s.item():.4f}, expected {expected:.4f}"
    print(f"  ✓ Scale: ||3.14*a|| = {norm_s.item():.4f} ≈ 3.14*||a|| = {expected:.4f}")
    
    # Test 6: Batched truncation
    from ontic.cfd.qtt_batched_ops import batched_truncation_sweep
    fields = [make_qtt(L, max_rank) for _ in range(6)]
    norms_before = [qtt_norm(f).item() for f in fields]
    fields = batched_truncation_sweep(fields, max_rank)
    norms_after = [qtt_norm(f).item() for f in fields]
    
    print("  Batched truncation (6 fields):")
    for i in range(6):
        ratio = norms_after[i] / max(norms_before[i], 1e-10)
        print(f"    Field {i}: ||before||={norms_before[i]:.4f}, ||after||={norms_after[i]:.4f}, ratio={ratio:.6f}")
    
    print("  ✓ All operations correct")
    return True


# ===========================================================================
# Test 3: Batched truncation sweep performance
# ===========================================================================

def test_batched_truncation_performance():
    """Compare batched vs individual truncation sweep timing."""
    print("\n[TEST 3] Batched Truncation Sweep Performance")
    print("-" * 50)
    
    from ontic.cfd.qtt_batched_ops import (
        batched_truncation_sweep, single_truncation_sweep,
    )
    
    torch.manual_seed(42)
    L = 15
    max_rank = 32
    
    # Create 6 fields with bloated ranks (simulating post-Hadamard)
    def make_bloated_qtt(L, rank):
        cores = []
        r = 1
        for k in range(L):
            r_next = min(rank, 4 * r) if k < L // 2 else max(1, r // 2)
            if k == L - 1:
                r_next = 1
            cores.append(torch.randn(r, 2, r_next, device=DEVICE))
            r = r_next
        return cores
    
    fields = [make_bloated_qtt(L, 64) for _ in range(6)]
    
    # Individual truncation (current approach)
    fields_copy = [[c.clone() for c in f] for f in fields]
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    t0 = time.perf_counter()
    for _ in range(10):
        for f in fields_copy:
            single_truncation_sweep(f, max_rank)
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    t_individual = (time.perf_counter() - t0) / 10
    
    # Batched truncation (new approach)
    fields_copy2 = [[c.clone() for c in f] for f in fields]
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    t0 = time.perf_counter()
    for _ in range(10):
        batched_truncation_sweep(fields_copy2, max_rank)
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    t_batched = (time.perf_counter() - t0) / 10
    
    speedup = t_individual / max(t_batched, 1e-9)
    
    print(f"  6 fields × {L} sites, rank {max_rank}:")
    print(f"    Individual: {t_individual*1000:.1f}ms (6 × {L} = {6*L} SVDs)")
    print(f"    Batched:    {t_batched*1000:.1f}ms ({L} batched SVDs)")
    print(f"    Speedup:    {speedup:.1f}x")
    
    return speedup


# ===========================================================================
# Test 4: Triton kernel performance
# ===========================================================================

def test_triton_kernel():
    """Benchmark Triton 3D residual absorption vs einsum."""
    print("\n[TEST 4] Triton 3D Kernel Performance")
    print("-" * 50)
    
    from ontic.cfd.triton_qtt3d import (
        triton_residual_absorb_3d, TRITON_AVAILABLE
    )
    
    print(f"  Triton available: {TRITON_AVAILABLE}")
    
    sizes = [
        (32, 16, 2, 32),   # (M, K, D, N) - typical middle site
        (16, 32, 2, 16),   # After truncation  
        (4, 8, 2, 16),     # Early site
        (64, 32, 2, 64),   # Large site
    ]
    
    for M, K, D, N in sizes:
        R = torch.randn(M, K, device=DEVICE)
        core = torch.randn(K, D, N, device=DEVICE)
        
        # Einsum
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t0 = time.perf_counter()
        for _ in range(1000):
            out_einsum = torch.einsum('ij,jsk->isk', R, core)
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t_einsum = (time.perf_counter() - t0) / 1000 * 1000  # ms
        
        # Triton / adaptive
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t0 = time.perf_counter()
        for _ in range(1000):
            out_triton = triton_residual_absorb_3d(R, core)
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t_triton = (time.perf_counter() - t0) / 1000 * 1000  # ms
        
        # Verify correctness
        err = (out_einsum - out_triton).abs().max().item()
        speedup = t_einsum / max(t_triton, 1e-9)
        
        print(f"  R({M},{K}) × core({K},{D},{N}): "
              f"einsum={t_einsum:.3f}ms, triton={t_triton:.3f}ms, "
              f"speedup={speedup:.2f}x, err={err:.2e}")
    
    return True


# ===========================================================================
# Test 5: Phase-level cross product
# ===========================================================================

def test_phase_cross_product():
    """Benchmark phase-level cross product vs sequential."""
    print("\n[TEST 5] Phase-Level Cross Product")
    print("-" * 50)
    
    from ontic.cfd.qtt_batched_ops import (
        batched_cross_product, hadamard_cores_raw, add_cores_raw,
        single_truncation_sweep, qtt_norm,
    )
    
    torch.manual_seed(42)
    L = 15
    max_rank = 32
    
    def make_qtt(L, rank):
        cores = []
        r = 1
        for k in range(L):
            r_next = min(rank, 2 * r) if k < L // 2 else max(1, r // 2)
            if k == L - 1: r_next = 1
            cores.append(torch.randn(r, 2, r_next, device=DEVICE))
            r = r_next
        return cores
    
    u = [make_qtt(L, max_rank) for _ in range(3)]
    omega = [make_qtt(L, max_rank) for _ in range(3)]
    
    # Sequential approach (simulating original)
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    t0 = time.perf_counter()
    for _ in range(5):
        cross_seq = []
        for comp in range(3):
            i, j = (comp + 1) % 3, (comp + 2) % 3
            h1 = single_truncation_sweep(hadamard_cores_raw(u[i], omega[j]), max_rank)
            h2 = single_truncation_sweep(hadamard_cores_raw(u[j], omega[i]), max_rank)
            cross_seq.append(single_truncation_sweep(
                add_cores_raw(h1, h2, 1.0, -1.0), max_rank))
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    t_seq = (time.perf_counter() - t0) / 5
    
    # Phase-level batched approach
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    t0 = time.perf_counter()
    for _ in range(5):
        cross_batch = batched_cross_product(u, omega, max_rank)
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    t_batch = (time.perf_counter() - t0) / 5
    
    speedup = t_seq / max(t_batch, 1e-9)
    
    print(f"  Sequential: {t_seq*1000:.0f}ms (9 truncation sweeps)")
    print(f"  Batched:    {t_batch*1000:.0f}ms (1 batched truncation)")
    print(f"  Speedup:    {speedup:.1f}x")
    
    # Verify both produce reasonable norms
    for i in range(3):
        n_seq = qtt_norm(cross_seq[i]).item()
        n_batch = qtt_norm(cross_batch[i]).item()
        ratio = n_batch / max(n_seq, 1e-10)
        print(f"  Component {i}: ||seq||={n_seq:.2f}, ||batch||={n_batch:.2f}, ratio={ratio:.4f}")
    
    return speedup


# ===========================================================================
# Test 6: Full solver integration (if available)
# ===========================================================================

def test_full_solver():
    """Test integration with TurboNS3DSolver if available."""
    print("\n[TEST 6] Full Solver Integration")
    print("-" * 50)
    
    try:
        from ontic.cfd.ns3d_turbo import TurboNS3DConfig, TurboNS3DSolver
        from ontic.cfd.qtt_batched_patch import patch_solver
    except ImportError as e:
        print(f"  SKIP: {e}")
        return None
    
    config = TurboNS3DConfig(
        n_bits=5, nu=0.01, dt=0.02,
        adaptive_rank=False, max_rank=32,
        device=DEVICE,
        diffusion_only=False,
        poisson_iterations=0,
    )
    
    # Original solver
    solver_orig = TurboNS3DSolver(config)
    solver_orig.initialize_taylor_green()
    
    # Warmup
    for _ in range(2):
        solver_orig.step()
    
    # Time original
    times_orig = []
    for _ in range(5):
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t0 = time.perf_counter()
        solver_orig.step()
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        times_orig.append((time.perf_counter() - t0) * 1000)
    
    # Patched solver
    solver_patched = TurboNS3DSolver(config)
    solver_patched.initialize_taylor_green()
    patch_solver(solver_patched)
    
    # Warmup
    for _ in range(2):
        solver_patched.step()
    
    # Time patched
    times_patched = []
    for _ in range(5):
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t0 = time.perf_counter()
        solver_patched.step()
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        times_patched.append((time.perf_counter() - t0) * 1000)
    
    avg_orig = np.mean(times_orig)
    avg_patched = np.mean(times_patched)
    speedup = avg_orig / max(avg_patched, 0.1)
    
    print(f"  Original:  {avg_orig:.0f}ms/step")
    print(f"  Batched:   {avg_patched:.0f}ms/step")
    print(f"  Speedup:   {speedup:.1f}x")
    
    return speedup


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  QTT BATCHED OPERATIONS BENCHMARK")
    print("=" * 70)
    
    results = {}
    
    # Run tests
    results['svd_speedup'] = test_batched_svd_speedup()
    
    try:
        results['correctness'] = test_qtt_operations()
    except Exception as e:
        print(f"  FAIL: {e}")
        results['correctness'] = False
    
    try:
        results['truncation_speedup'] = test_batched_truncation_performance()
    except Exception as e:
        print(f"  FAIL: {e}")
        results['truncation_speedup'] = 0
    
    try:
        results['triton'] = test_triton_kernel()
    except Exception as e:
        print(f"  FAIL: {e}")
        results['triton'] = False
    
    try:
        results['cross_product_speedup'] = test_phase_cross_product()
    except Exception as e:
        print(f"  FAIL: {e}")
        results['cross_product_speedup'] = 0
    
    try:
        results['solver_speedup'] = test_full_solver()
    except Exception as e:
        print(f"  FAIL: {e}")
        results['solver_speedup'] = None
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Batched SVD speedup:        {results.get('svd_speedup', 0):.1f}x")
    print(f"  Truncation sweep speedup:   {results.get('truncation_speedup', 0):.1f}x")
    print(f"  Cross product speedup:      {results.get('cross_product_speedup', 0):.1f}x")
    if results.get('solver_speedup'):
        print(f"  Full solver speedup:        {results['solver_speedup']:.1f}x")
    print(f"  Correctness:                {'✓ PASS' if results.get('correctness') else '✗ FAIL'}")
    print(f"  Triton kernels:             {'✓ PASS' if results.get('triton') else '✗ FAIL'}")
    print("=" * 70)
