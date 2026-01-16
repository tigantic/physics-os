"""
QTT Native CUDA Operations
==========================

Python bindings for CUDA-accelerated QTT operations.
NO DENSE - all operations stay in O(L × r³) complexity.

Key functions:
- qtt_inner_cuda: Inner product via core contraction
- qtt_add_cuda: Addition via block-diagonal concatenation
- qtt_hadamard_cuda: Element-wise product via Kronecker cores
- apply_mpo_cuda: MPO × QTT contraction

These replace the sequential Python loops with fused CUDA kernels,
eliminating the kernel launch overhead that killed GPU performance.

Author: HyperTensor Team
Date: January 2026
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensornet.cfd.pure_qtt_ops import QTTState

# Try to import CUDA extension - fall back to CPU if not available
_cuda_available = False
_qtt_cuda = None

def _try_load_cuda():
    """Attempt to load the CUDA extension."""
    global _cuda_available, _qtt_cuda
    
    if _cuda_available:
        return True
    
    try:
        # Try loading pre-compiled extension
        from tensornet.cuda import qtt_native_cuda as _qtt_cuda
        _cuda_available = True
        return True
    except ImportError:
        pass
    
    try:
        # Try JIT compilation
        from torch.utils.cpp_extension import load
        import os
        
        cuda_src = os.path.join(os.path.dirname(__file__), "qtt_native_kernels.cu")
        if os.path.exists(cuda_src):
            _qtt_cuda = load(
                name="qtt_native_cuda",
                sources=[cuda_src],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False
            )
            _cuda_available = True
            return True
    except Exception as e:
        print(f"[QTT CUDA] JIT compile failed: {e}")
    
    return False


def is_cuda_available() -> bool:
    """Check if CUDA QTT kernels are available."""
    return _try_load_cuda()


# ============================================================================
# Core Flattening Utilities
# ============================================================================

def _flatten_cores(cores: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten QTT cores into contiguous buffer with offset/shape metadata.
    
    Returns:
        flat: Contiguous tensor with all core data
        offsets: [L+1] tensor with start offset of each core
        shapes: [L, 3] tensor with (r_left, d, r_right) for each core
    """
    L = len(cores)
    device = cores[0].device
    dtype = cores[0].dtype
    
    # Compute offsets and shapes
    offsets = [0]
    shapes = []
    total_size = 0
    
    for c in cores:
        r_left, d, r_right = c.shape
        shapes.append([r_left, d, r_right])
        total_size += r_left * d * r_right
        offsets.append(total_size)
    
    # Create flat buffer
    flat = torch.zeros(total_size, dtype=dtype, device=device)
    for i, c in enumerate(cores):
        flat[offsets[i]:offsets[i+1]] = c.flatten()
    
    offsets_t = torch.tensor(offsets, dtype=torch.int32, device=device)
    shapes_t = torch.tensor(shapes, dtype=torch.int32, device=device)
    
    return flat, offsets_t, shapes_t


def _unflatten_cores(flat: torch.Tensor, offsets: torch.Tensor, shapes: torch.Tensor) -> list[torch.Tensor]:
    """
    Unflatten core buffer back to list of tensors.
    """
    L = shapes.shape[0]
    cores = []
    
    offsets_cpu = offsets.cpu().tolist()
    shapes_cpu = shapes.cpu().tolist()
    
    for i in range(L):
        r_left, d, r_right = shapes_cpu[i]
        start, end = offsets_cpu[i], offsets_cpu[i+1]
        core = flat[start:end].reshape(r_left, d, r_right)
        cores.append(core)
    
    return cores


def _flatten_mpo_cores(cores: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten MPO cores (shape: r_left, d_out, d_in, r_right).
    """
    L = len(cores)
    device = cores[0].device
    dtype = cores[0].dtype
    
    offsets = [0]
    shapes = []
    total_size = 0
    
    for c in cores:
        r_left, d_out, d_in, r_right = c.shape
        shapes.append([r_left, d_out, d_in, r_right])
        total_size += r_left * d_out * d_in * r_right
        offsets.append(total_size)
    
    flat = torch.zeros(total_size, dtype=dtype, device=device)
    for i, c in enumerate(cores):
        flat[offsets[i]:offsets[i+1]] = c.flatten()
    
    offsets_t = torch.tensor(offsets, dtype=torch.int32, device=device)
    shapes_t = torch.tensor(shapes, dtype=torch.int32, device=device)
    
    return flat, offsets_t, shapes_t


# ============================================================================
# CUDA QTT Operations
# ============================================================================

def qtt_inner_cuda(a_cores: list[torch.Tensor], b_cores: list[torch.Tensor]) -> float:
    """
    Compute QTT inner product <a|b> using CUDA.
    
    O(L × r² × d) complexity - no dense operations.
    
    Uses PyTorch einsum on GPU (cuBLAS backend) for correctness.
    The custom kernel is available for future optimization.
    """
    # Use CPU implementation on GPU tensors - PyTorch handles this efficiently
    # The custom CUDA kernel has indexing issues to be fixed later
    return _qtt_inner_cpu(a_cores, b_cores)


def _qtt_inner_cpu(a_cores: list[torch.Tensor], b_cores: list[torch.Tensor]) -> float:
    """CPU fallback for inner product."""
    L = len(a_cores)
    env = torch.ones(1, 1, dtype=a_cores[0].dtype, device=a_cores[0].device)
    
    for i in range(L):
        ca, cb = a_cores[i], b_cores[i]
        tmp = torch.einsum('ij,idk->jdk', env, ca)
        env = torch.einsum('jdk,jdl->kl', tmp, cb)
    
    return env.item()


def qtt_add_cuda(
    cores1: list[torch.Tensor], 
    cores2: list[torch.Tensor]
) -> list[torch.Tensor]:
    """
    Add two QTT states using CUDA block-diagonal concatenation.
    
    Bond dimension doubles, requires truncation afterward.
    """
    if not _try_load_cuda():
        return _qtt_add_cpu(cores1, cores2)
    
    # Flatten inputs
    flat1, offsets1, shapes1 = _flatten_cores(cores1)
    flat2, offsets2, shapes2 = _flatten_cores(cores2)
    
    # Compute output shapes and offsets
    L = len(cores1)
    result_shapes = []
    result_offsets = [0]
    
    for i in range(L):
        r1L, d, r1R = shapes1[i].tolist()
        r2L, _, r2R = shapes2[i].tolist()
        
        if i == 0:
            # First core: cat along right
            result_shapes.append([r1L, d, r1R + r2R])
        elif i == L - 1:
            # Last core: cat along left
            result_shapes.append([r1L + r2L, d, r1R])
        else:
            # Middle: block diagonal
            result_shapes.append([r1L + r2L, d, r1R + r2R])
        
        rL, d, rR = result_shapes[-1]
        result_offsets.append(result_offsets[-1] + rL * d * rR)
    
    total_size = result_offsets[-1]
    device = flat1.device
    result_offsets_t = torch.tensor(result_offsets, dtype=torch.int32, device=device)
    result_shapes_t = torch.tensor(result_shapes, dtype=torch.int32, device=device)
    
    # Move to CUDA
    if not flat1.is_cuda:
        flat1 = flat1.cuda()
        offsets1 = offsets1.cuda()
        shapes1 = shapes1.cuda()
        flat2 = flat2.cuda()
        offsets2 = offsets2.cuda()
        shapes2 = shapes2.cuda()
        result_offsets_t = result_offsets_t.cuda()
        result_shapes_t = result_shapes_t.cuda()
    
    result_flat = _qtt_cuda.qtt_add(
        flat1, flat2, offsets1, offsets2, result_offsets_t,
        shapes1, shapes2, total_size
    )
    
    # Unflatten result
    return _unflatten_cores(result_flat, result_offsets_t, result_shapes_t)


def _qtt_add_cpu(cores1: list[torch.Tensor], cores2: list[torch.Tensor]) -> list[torch.Tensor]:
    """CPU fallback for QTT addition."""
    n = len(cores1)
    dtype = cores1[0].dtype
    device = cores1[0].device
    cores = []
    
    for i in range(n):
        c1, c2 = cores1[i], cores2[i]
        r1L, d, r1R = c1.shape
        r2L, _, r2R = c2.shape
        
        if i == 0:
            new_core = torch.cat([c1, c2], dim=2)
        elif i == n - 1:
            new_core = torch.cat([c1, c2], dim=0)
        else:
            new_core = torch.zeros(r1L + r2L, d, r1R + r2R, dtype=dtype, device=device)
            new_core[:r1L, :, :r1R] = c1
            new_core[r1L:, :, r1R:] = c2
        
        cores.append(new_core)
    
    return cores


def qtt_hadamard_cuda(
    cores1: list[torch.Tensor], 
    cores2: list[torch.Tensor]
) -> list[torch.Tensor]:
    """
    Element-wise (Hadamard) product using CUDA Kronecker core product.
    
    Bond dimension squares (r1 × r2), requires truncation afterward.
    """
    if not _try_load_cuda():
        return _qtt_hadamard_cpu(cores1, cores2)
    
    # Flatten inputs
    flat1, offsets1, shapes1 = _flatten_cores(cores1)
    flat2, offsets2, shapes2 = _flatten_cores(cores2)
    
    # Compute output shapes
    L = len(cores1)
    result_shapes = []
    result_offsets = [0]
    
    for i in range(L):
        r1L, d, r1R = shapes1[i].tolist()
        r2L, _, r2R = shapes2[i].tolist()
        result_shapes.append([r1L * r2L, d, r1R * r2R])
        rL, d, rR = result_shapes[-1]
        result_offsets.append(result_offsets[-1] + rL * d * rR)
    
    total_size = result_offsets[-1]
    device = flat1.device
    result_offsets_t = torch.tensor(result_offsets, dtype=torch.int32, device=device)
    result_shapes_t = torch.tensor(result_shapes, dtype=torch.int32, device=device)
    
    # Move to CUDA
    if not flat1.is_cuda:
        flat1 = flat1.cuda()
        offsets1 = offsets1.cuda()
        shapes1 = shapes1.cuda()
        flat2 = flat2.cuda()
        offsets2 = offsets2.cuda()
        shapes2 = shapes2.cuda()
        result_offsets_t = result_offsets_t.cuda()
        result_shapes_t = result_shapes_t.cuda()
    
    result_flat = _qtt_cuda.qtt_hadamard(
        flat1, flat2, offsets1, offsets2, result_offsets_t,
        shapes1, shapes2, total_size
    )
    
    return _unflatten_cores(result_flat, result_offsets_t, result_shapes_t)


def _qtt_hadamard_cpu(cores1: list[torch.Tensor], cores2: list[torch.Tensor]) -> list[torch.Tensor]:
    """CPU fallback for Hadamard product."""
    cores = []
    for c1, c2 in zip(cores1, cores2):
        r1L, d, r1R = c1.shape
        r2L, _, r2R = c2.shape
        # Kronecker product via einsum
        kron = torch.einsum("adb,cde->acdbe", c1, c2)
        new_core = kron.reshape(r1L * r2L, d, r1R * r2R)
        cores.append(new_core)
    return cores


def apply_mpo_cuda(
    mpo_cores: list[torch.Tensor],
    qtt_cores: list[torch.Tensor]
) -> list[torch.Tensor]:
    """
    Apply MPO to QTT using CUDA batched contraction.
    
    Each core: O[rLo, d_out, d_in, rRo] × P[rLp, d_in, rRp] → R[rLo*rLp, d_out, rRo*rRp]
    """
    if not _try_load_cuda():
        return _apply_mpo_cpu(mpo_cores, qtt_cores)
    
    # Flatten inputs
    mpo_flat, mpo_offsets, mpo_shapes = _flatten_mpo_cores(mpo_cores)
    qtt_flat, qtt_offsets, qtt_shapes = _flatten_cores(qtt_cores)
    
    # Compute output shapes
    L = len(mpo_cores)
    result_shapes = []
    result_offsets = [0]
    
    for i in range(L):
        rLo, d_out, d_in, rRo = mpo_shapes[i].tolist()
        rLp, _, rRp = qtt_shapes[i].tolist()
        result_shapes.append([rLo * rLp, d_out, rRo * rRp])
        rL, d, rR = result_shapes[-1]
        result_offsets.append(result_offsets[-1] + rL * d * rR)
    
    total_size = result_offsets[-1]
    device = qtt_flat.device
    result_offsets_t = torch.tensor(result_offsets, dtype=torch.int32, device=device)
    result_shapes_t = torch.tensor(result_shapes, dtype=torch.int32, device=device)
    
    # Move to CUDA
    if not mpo_flat.is_cuda:
        mpo_flat = mpo_flat.cuda()
        mpo_offsets = mpo_offsets.cuda()
        mpo_shapes = mpo_shapes.cuda()
        qtt_flat = qtt_flat.cuda()
        qtt_offsets = qtt_offsets.cuda()
        qtt_shapes = qtt_shapes.cuda()
        result_offsets_t = result_offsets_t.cuda()
        result_shapes_t = result_shapes_t.cuda()
    
    result_flat = _qtt_cuda.apply_mpo(
        mpo_flat, qtt_flat, mpo_offsets, qtt_offsets, result_offsets_t,
        mpo_shapes, qtt_shapes, total_size
    )
    
    return _unflatten_cores(result_flat, result_offsets_t, result_shapes_t)


def _apply_mpo_cpu(mpo_cores: list[torch.Tensor], qtt_cores: list[torch.Tensor]) -> list[torch.Tensor]:
    """CPU fallback for MPO application."""
    new_cores = []
    for O, P in zip(mpo_cores, qtt_cores):
        rLo, d_out, d_in, rRo = O.shape
        rLp, d_in_p, rRp = P.shape
        result = torch.einsum("oabr,pbq->oparq", O, P)
        result = result.reshape(rLo * rLp, d_out, rRo * rRp)
        new_cores.append(result)
    return new_cores


# ============================================================================
# High-Level QTTState Integration
# ============================================================================

def enable_cuda_backend():
    """
    Enable CUDA backend for QTT operations in the solver.
    
    Monkey-patches the pure_qtt_ops functions to use CUDA.
    """
    if not _try_load_cuda():
        print("[QTT CUDA] CUDA not available, using CPU backend")
        return False
    
    try:
        from tensornet.cfd import pure_qtt_ops
        
        # Store original functions
        pure_qtt_ops._qtt_add_original = pure_qtt_ops.qtt_add
        pure_qtt_ops._qtt_hadamard_original = pure_qtt_ops.qtt_hadamard
        pure_qtt_ops._qtt_inner_product_original = pure_qtt_ops.qtt_inner_product
        
        # Define CUDA-accelerated wrappers
        def qtt_add_cuda_wrapper(qtt1, qtt2, max_bond=64):
            from tensornet.cfd.pure_qtt_ops import QTTState, truncate_qtt
            new_cores = qtt_add_cuda(qtt1.cores, qtt2.cores)
            result = QTTState(cores=new_cores, num_qubits=qtt1.num_qubits)
            return truncate_qtt(result, max_bond=max_bond)
        
        def qtt_hadamard_cuda_wrapper(qtt1, qtt2, max_bond=64):
            from tensornet.cfd.pure_qtt_ops import QTTState, truncate_qtt
            new_cores = qtt_hadamard_cuda(qtt1.cores, qtt2.cores)
            result = QTTState(cores=new_cores, num_qubits=qtt1.num_qubits)
            max_current = max(c.shape[0] for c in new_cores)
            if max_current > max_bond:
                result = truncate_qtt(result, max_bond=max_bond)
            return result
        
        def qtt_inner_product_cuda_wrapper(qtt1, qtt2):
            return qtt_inner_cuda(qtt1.cores, qtt2.cores)
        
        # Replace functions
        pure_qtt_ops.qtt_add = qtt_add_cuda_wrapper
        pure_qtt_ops.qtt_hadamard = qtt_hadamard_cuda_wrapper
        pure_qtt_ops.qtt_inner_product = qtt_inner_product_cuda_wrapper
        
        print("[QTT CUDA] CUDA backend enabled for QTT operations")
        return True
        
    except Exception as e:
        print(f"[QTT CUDA] Failed to enable CUDA backend: {e}")
        return False


def disable_cuda_backend():
    """Restore original CPU implementations."""
    try:
        from tensornet.cfd import pure_qtt_ops
        
        if hasattr(pure_qtt_ops, '_qtt_add_original'):
            pure_qtt_ops.qtt_add = pure_qtt_ops._qtt_add_original
            pure_qtt_ops.qtt_hadamard = pure_qtt_ops._qtt_hadamard_original
            pure_qtt_ops.qtt_inner_product = pure_qtt_ops._qtt_inner_product_original
            print("[QTT CUDA] CPU backend restored")
            return True
    except:
        pass
    return False


# ============================================================================
# Benchmarking Utilities
# ============================================================================

def benchmark_cuda_vs_cpu(num_qubits: int = 14, rank: int = 32, n_trials: int = 10):
    """
    Benchmark CUDA vs CPU for QTT operations.
    """
    import time
    
    print(f"\n{'='*60}")
    print(f"QTT CUDA Benchmark: L={num_qubits}, r={rank}")
    print(f"{'='*60}")
    
    # Create random QTT cores
    cores1 = []
    cores2 = []
    for i in range(num_qubits):
        r_left = 1 if i == 0 else rank
        r_right = 1 if i == num_qubits - 1 else rank
        cores1.append(torch.randn(r_left, 2, r_right, dtype=torch.float64))
        cores2.append(torch.randn(r_left, 2, r_right, dtype=torch.float64))
    
    # Benchmark inner product
    print(f"\n--- Inner Product ---")
    
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = _qtt_inner_cpu(cores1, cores2)
    cpu_time = (time.perf_counter() - t0) / n_trials * 1000
    print(f"  CPU: {cpu_time:.3f} ms")
    
    if _try_load_cuda():
        # Move to GPU
        cores1_gpu = [c.cuda() for c in cores1]
        cores2_gpu = [c.cuda() for c in cores2]
        
        # Warmup
        _ = qtt_inner_cuda(cores1_gpu, cores2_gpu)
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(n_trials):
            _ = qtt_inner_cuda(cores1_gpu, cores2_gpu)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - t0) / n_trials * 1000
        print(f"  CUDA: {cuda_time:.3f} ms")
        print(f"  Speedup: {cpu_time/cuda_time:.1f}×")
    
    # Benchmark add
    print(f"\n--- QTT Add ---")
    
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = _qtt_add_cpu(cores1, cores2)
    cpu_time = (time.perf_counter() - t0) / n_trials * 1000
    print(f"  CPU: {cpu_time:.3f} ms")
    
    if _try_load_cuda():
        t0 = time.perf_counter()
        for _ in range(n_trials):
            _ = qtt_add_cuda(cores1_gpu, cores2_gpu)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - t0) / n_trials * 1000
        print(f"  CUDA: {cuda_time:.3f} ms")
        print(f"  Speedup: {cpu_time/cuda_time:.1f}×")
    
    # Benchmark Hadamard
    print(f"\n--- QTT Hadamard ---")
    
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = _qtt_hadamard_cpu(cores1, cores2)
    cpu_time = (time.perf_counter() - t0) / n_trials * 1000
    print(f"  CPU: {cpu_time:.3f} ms")
    
    if _try_load_cuda():
        t0 = time.perf_counter()
        for _ in range(n_trials):
            _ = qtt_hadamard_cuda(cores1_gpu, cores2_gpu)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - t0) / n_trials * 1000
        print(f"  CUDA: {cuda_time:.3f} ms")
        print(f"  Speedup: {cpu_time/cuda_time:.1f}×")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    # Run benchmark
    benchmark_cuda_vs_cpu(num_qubits=14, rank=32)
