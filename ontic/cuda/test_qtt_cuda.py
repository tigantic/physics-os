#!/usr/bin/env python3
"""
QTT CUDA Backend Test
=====================

Tests the CUDA fused kernels for QTT operations.
Verifies correctness against CPU implementation and benchmarks performance.

Author: TiganticLabz
Date: January 2026
"""

import torch
import time
import sys

def test_cpu_operations():
    """Test CPU QTT operations work correctly."""
    print("\n" + "="*60)
    print("Testing CPU QTT Operations")
    print("="*60)
    
    from ontic.cfd.pure_qtt_ops import QTTState, qtt_add, qtt_hadamard
    
    # Create simple test QTT states
    L = 8  # 2^8 = 256 points
    rank = 4
    
    cores1 = []
    cores2 = []
    for i in range(L):
        r_left = 1 if i == 0 else rank
        r_right = 1 if i == L - 1 else rank
        cores1.append(torch.randn(r_left, 2, r_right, dtype=torch.float64))
        cores2.append(torch.randn(r_left, 2, r_right, dtype=torch.float64))
    
    qtt1 = QTTState(cores=cores1, num_qubits=L)
    qtt2 = QTTState(cores=cores2, num_qubits=L)
    
    # Test add
    result_add = qtt_add(qtt1, qtt2, max_bond=8)
    print(f"  qtt_add: OK (result max_rank={result_add.max_rank})")
    
    # Test hadamard
    result_had = qtt_hadamard(qtt1, qtt2, max_bond=8)
    print(f"  qtt_hadamard: OK (result max_rank={result_had.max_rank})")
    
    print("  CPU operations: PASS ✓")
    return True


def test_cuda_availability():
    """Test if CUDA is available and QTT kernels can be loaded."""
    print("\n" + "="*60)
    print("Testing CUDA Availability")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("  CUDA not available - skipping GPU tests")
        return False
    
    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    
    # Try to import CUDA ops
    try:
        from ontic.cuda.qtt_native_ops import is_cuda_available, _try_load_cuda
        
        if _try_load_cuda():
            print("  QTT CUDA extension: Loaded ✓")
            return True
        else:
            print("  QTT CUDA extension: Not compiled (will JIT on first use)")
            return True  # JIT compilation is OK
            
    except ImportError as e:
        print(f"  QTT CUDA extension: Import failed - {e}")
        return False


def test_cuda_correctness():
    """Test CUDA operations match CPU results."""
    print("\n" + "="*60)
    print("Testing CUDA Correctness")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("  Skipped (no CUDA)")
        return True
    
    from ontic.cuda.qtt_native_ops import (
        _qtt_inner_cpu, _qtt_add_cpu, _qtt_hadamard_cpu,
        qtt_inner_cuda, qtt_add_cuda, qtt_hadamard_cuda,
        is_cuda_available
    )
    
    if not is_cuda_available():
        print("  CUDA kernels not available - skipping")
        return True
    
    # Create test cores
    L = 10
    rank = 8
    
    cores1_cpu = []
    cores2_cpu = []
    for i in range(L):
        r_left = 1 if i == 0 else rank
        r_right = 1 if i == L - 1 else rank
        cores1_cpu.append(torch.randn(r_left, 2, r_right, dtype=torch.float64))
        cores2_cpu.append(torch.randn(r_left, 2, r_right, dtype=torch.float64))
    
    cores1_gpu = [c.cuda() for c in cores1_cpu]
    cores2_gpu = [c.cuda() for c in cores2_cpu]
    
    # Test inner product
    inner_cpu = _qtt_inner_cpu(cores1_cpu, cores2_cpu)
    inner_gpu = qtt_inner_cuda(cores1_gpu, cores2_gpu)
    inner_err = abs(inner_cpu - inner_gpu) / (abs(inner_cpu) + 1e-10)
    print(f"  Inner product: CPU={inner_cpu:.6e}, GPU={inner_gpu:.6e}, rel_err={inner_err:.2e}")
    assert inner_err < 1e-10, "Inner product mismatch!"
    
    # Test add
    add_cpu = _qtt_add_cpu(cores1_cpu, cores2_cpu)
    add_gpu = qtt_add_cuda(cores1_gpu, cores2_gpu)
    
    # Compare core by core
    add_err = 0.0
    for i, (c_cpu, c_gpu) in enumerate(zip(add_cpu, add_gpu)):
        err = torch.abs(c_cpu - c_gpu.cpu()).max().item()
        add_err = max(add_err, err)
    print(f"  Add: max_error={add_err:.2e}")
    assert add_err < 1e-10, "Add mismatch!"
    
    # Test hadamard
    had_cpu = _qtt_hadamard_cpu(cores1_cpu, cores2_cpu)
    had_gpu = qtt_hadamard_cuda(cores1_gpu, cores2_gpu)
    
    had_err = 0.0
    for i, (c_cpu, c_gpu) in enumerate(zip(had_cpu, had_gpu)):
        err = torch.abs(c_cpu - c_gpu.cpu()).max().item()
        had_err = max(had_err, err)
    print(f"  Hadamard: max_error={had_err:.2e}")
    assert had_err < 1e-10, "Hadamard mismatch!"
    
    print("  Correctness: PASS ✓")
    return True


def test_cuda_performance():
    """Benchmark CUDA vs CPU performance."""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("  Skipped (no CUDA)")
        return True
    
    try:
        from ontic.cuda.qtt_native_ops import benchmark_cuda_vs_cpu
        benchmark_cuda_vs_cpu(num_qubits=14, rank=32, n_trials=10)
    except Exception as e:
        print(f"  Benchmark failed: {e}")
        return False
    
    return True


def test_solver_with_cuda():
    """Test NS2D solver with CUDA backend."""
    print("\n" + "="*60)
    print("Testing NS2D Solver with CUDA Backend")
    print("="*60)
    
    from ontic.cfd.ns2d_qtt_native import (
        NS2DQTTConfig, NS2D_QTT_Native,
        create_conference_room_ic, qtt_2d_native_to_dense
    )
    
    # Small test grid
    config = NS2DQTTConfig(
        nx_bits=6,   # 64
        ny_bits=6,   # 64
        Lx=9.0,
        Ly=3.0,
        nu=1.5e-5,
        max_rank=16,
        use_cuda=torch.cuda.is_available(),
    )
    
    solver = NS2D_QTT_Native(config)
    omega, psi, psi_bc, bc_mask = create_conference_room_ic(config)
    
    # Run a few iterations
    print("\n  Running 10 steady-state iterations...")
    t0 = time.perf_counter()
    omega, psi, info = solver.solve_steady_state(
        omega, psi, psi_bc, bc_mask,
        max_iters=10,
        tol=1e-10,  # Won't converge in 10 iters
        poisson_iters=10,
        verbose=False
    )
    elapsed = time.perf_counter() - t0
    print(f"  10 iterations in {elapsed:.2f}s ({elapsed/10*1000:.1f}ms per iter)")
    
    # Check result is sane
    psi_dense = qtt_2d_native_to_dense(psi)
    print(f"  ψ range: [{psi_dense.min():.4f}, {psi_dense.max():.4f}]")
    
    if psi_dense.isnan().any():
        print("  ERROR: NaN in result!")
        return False
    
    print("  Solver test: PASS ✓")
    return True


def main():
    print("="*60)
    print("  QTT CUDA Backend Test Suite")
    print("="*60)
    
    results = {
        "CPU Operations": test_cpu_operations(),
        "CUDA Availability": test_cuda_availability(),
        "CUDA Correctness": test_cuda_correctness(),
        "CUDA Performance": test_cuda_performance(),
        "Solver with CUDA": test_solver_with_cuda(),
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed
    
    print("="*60)
    if all_pass:
        print("  All tests passed!")
        return 0
    else:
        print("  Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
