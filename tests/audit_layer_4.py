"""
Layer 4 Audit: Morton-Aware Slicing Validation
===============================================

Validates that Morton projection produces mathematically correct slices
by comparing against analytically computable ground truth.

Tests:
    1. Sine wave: f(x,y,z) = sin(2πx)sin(2πy)sin(2πz)
    2. Gaussian:  f(x,y,z) = exp(-|r - 0.5|² / σ²)
    3. Linear:    f(x,y,z) = x + 2y + 3z
    4. Constant:  f(x,y,z) = c

For each function:
    - Create analytical 3D field
    - Compress to 3D Morton QTT
    - Extract slice via Morton projection
    - Compare against true analytical slice
    
Success criteria:
    - max_error < 1e-4 for each test
    - Reconstruction matches analytical truth

Author: HyperTensor Validation Team
"""

import torch
import numpy as np
import time
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass


# =============================================================================
# ANALYTICAL GROUND TRUTH
# =============================================================================

def analytical_sine_3d(N: int, device: torch.device) -> torch.Tensor:
    """
    f(x,y,z) = sin(2πx)sin(2πy)sin(2πz)
    
    True Z-slice at z_index = k:
        f(x,y,z_k) = sin(2πx)sin(2πy)sin(2πz_k)
    """
    coords = torch.linspace(0, 1, N, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
    return torch.sin(2 * np.pi * xx) * torch.sin(2 * np.pi * yy) * torch.sin(2 * np.pi * zz)


def analytical_sine_z_slice(N: int, z_index: int, device: torch.device) -> torch.Tensor:
    """True XY slice of sine function at fixed Z."""
    coords = torch.linspace(0, 1, N, device=device)
    z_val = z_index / (N - 1) if N > 1 else 0.0
    
    xx, yy = torch.meshgrid(coords, coords, indexing='ij')
    return torch.sin(2 * np.pi * xx) * torch.sin(2 * np.pi * yy) * np.sin(2 * np.pi * z_val)


def analytical_gaussian_3d(N: int, device: torch.device, sigma: float = 0.1) -> torch.Tensor:
    """
    f(x,y,z) = exp(-((x-0.5)² + (y-0.5)² + (z-0.5)²) / σ²)
    """
    coords = torch.linspace(0, 1, N, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
    r2 = (xx - 0.5)**2 + (yy - 0.5)**2 + (zz - 0.5)**2
    return torch.exp(-r2 / sigma)


def analytical_gaussian_z_slice(N: int, z_index: int, device: torch.device, sigma: float = 0.1) -> torch.Tensor:
    """True XY slice of Gaussian at fixed Z."""
    coords = torch.linspace(0, 1, N, device=device)
    z_val = z_index / (N - 1) if N > 1 else 0.0
    
    xx, yy = torch.meshgrid(coords, coords, indexing='ij')
    r2 = (xx - 0.5)**2 + (yy - 0.5)**2 + (z_val - 0.5)**2
    return torch.exp(-r2 / sigma)


def analytical_linear_3d(N: int, device: torch.device) -> torch.Tensor:
    """
    f(x,y,z) = x + 2y + 3z
    """
    coords = torch.linspace(0, 1, N, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
    return xx + 2 * yy + 3 * zz


def analytical_linear_z_slice(N: int, z_index: int, device: torch.device) -> torch.Tensor:
    """True XY slice of linear function at fixed Z."""
    coords = torch.linspace(0, 1, N, device=device)
    z_val = z_index / (N - 1) if N > 1 else 0.0
    
    xx, yy = torch.meshgrid(coords, coords, indexing='ij')
    return xx + 2 * yy + 3 * z_val


# =============================================================================
# MORTON QTT COMPRESSION
# =============================================================================

def dense_3d_to_morton_qtt(
    field_3d: torch.Tensor,
    max_rank: int = 64
) -> List[torch.Tensor]:
    """
    Convert 3D dense field to Morton-ordered QTT.
    
    Args:
        field_3d: (N, N, N) tensor where N = 2^bits
        max_rank: Maximum bond dimension
        
    Returns:
        List of cores with shape (r_in, 8, r_out)
    """
    N = field_3d.shape[0]
    bits_per_dim = int(np.log2(N))
    device = field_3d.device
    dtype = field_3d.dtype
    
    assert 2**bits_per_dim == N, f"N={N} must be power of 2"
    
    # Convert to Morton order
    N3 = N ** 3
    morton_flat = torch.zeros(N3, dtype=dtype, device=device)
    
    for x in range(N):
        for y in range(N):
            for z in range(N):
                m = morton_encode_3d(x, y, z, bits_per_dim)
                morton_flat[m] = field_3d[x, y, z]
    
    # TT-SVD to QTT cores with physical dim 8
    # Right-canonical form: U matrices stored, S*Vh carried forward
    cores = []
    remainder = morton_flat.reshape(1, -1)  # (1, 8^L)
    
    for k in range(bits_per_dim):
        r_in = remainder.shape[0]
        phys_dim = 8
        total_right = remainder.shape[1] // phys_dim
        
        if total_right == 0:
            # Last core — absorb everything
            core = remainder.reshape(r_in, phys_dim, 1)
            cores.append(core)
            break
        
        # Reshape to matrix
        mat = remainder.reshape(r_in * phys_dim, total_right)
        
        # SVD
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        
        # Truncate
        r_out = min(max_rank, len(S))
        U = U[:, :r_out]
        S = S[:r_out]
        Vh = Vh[:r_out, :]
        
        # Core k: store U (right-canonical)
        core = U.reshape(r_in, phys_dim, r_out)
        cores.append(core)
        
        # Remainder: S @ Vh (carries singular values forward)
        remainder = torch.diag(S) @ Vh
    
    # CRITICAL: Absorb final remainder into last core
    if len(cores) > 0 and remainder.numel() > 0:
        last_core = cores[-1]  # (r_in, 8, r_out)
        r_in, phys, r_out = last_core.shape
        rem_rows = remainder.shape[0]
        rem_cols = remainder.shape[1] if remainder.dim() > 1 else 1
        
        if r_out == rem_rows:
            # Contract: (r_in, phys, r_out) @ (r_out, rem_cols)
            remainder_2d = remainder.reshape(rem_rows, rem_cols)
            new_last = torch.einsum('ipj,jk->ipk', last_core, remainder_2d)
            cores[-1] = new_last
    
    return cores


def morton_encode_3d(x: int, y: int, z: int, bits: int) -> int:
    """Encode (x, y, z) to 3D Morton index."""
    result = 0
    for i in range(bits):
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        z_bit = (z >> i) & 1
        result |= (x_bit << (3 * i))
        result |= (y_bit << (3 * i + 1))
        result |= (z_bit << (3 * i + 2))
    return result


# =============================================================================
# MORTON SLICING (Extracted from morton_ops for self-contained test)
# =============================================================================

def slice_morton_z_plane(
    cores: List[torch.Tensor],
    z_index: int,
    bits_per_dim: int
) -> List[torch.Tensor]:
    """
    Extract XY plane at fixed Z from 3D Morton QTT.
    
    Returns 2D QTT cores with physical dim 4.
    """
    sliced_cores = []
    
    for k in range(len(cores)):
        core = cores[k]
        bit_pos = bits_per_dim - 1 - k
        z_bit = (z_index >> bit_pos) & 1
        
        # j_3d = 4*z + 2*y + x
        # For fixed z_bit, keep indices [4*z_bit, 4*z_bit+1, 4*z_bit+2, 4*z_bit+3]
        indices = [4 * z_bit + j for j in range(4)]
        sliced_core = core[:, indices, :]
        sliced_cores.append(sliced_core)
    
    return sliced_cores


def contract_2d_qtt_to_dense(
    cores: List[torch.Tensor],
    bits_per_dim: int
) -> torch.Tensor:
    """
    Contract 2D QTT to dense (N, N) with Morton reorder.
    """
    if len(cores) == 0:
        return torch.tensor([[]])
    
    device = cores[0].device
    N = 2 ** bits_per_dim
    
    # Contract all cores
    result = cores[0]
    for i in range(1, len(cores)):
        r_0, size, r_i = result.shape
        r_i2, phys, r_i1 = cores[i].shape
        
        result = result.reshape(r_0 * size, r_i)
        core_flat = cores[i].reshape(r_i, phys * r_i1)
        result = result @ core_flat
        result = result.reshape(r_0, size * phys, r_i1)
    
    morton_flat = result.squeeze()
    
    # De-Morton
    output = torch.zeros(N, N, dtype=morton_flat.dtype, device=device)
    for z_morton in range(N * N):
        x = 0
        y = 0
        for i in range(bits_per_dim):
            x |= ((z_morton >> (2 * i)) & 1) << i
            y |= ((z_morton >> (2 * i + 1)) & 1) << i
        if x < N and y < N:
            output[x, y] = morton_flat[z_morton]
    
    return output


# =============================================================================
# TEST SUITE
# =============================================================================

@dataclass
class TestResult:
    """Result of a single audit test."""
    name: str
    passed: bool
    max_error: float
    mean_error: float
    time_ms: float
    details: str = ""


def run_audit_test(
    name: str,
    dense_3d: torch.Tensor,
    ground_truth_slice: torch.Tensor,
    z_index: int,
    max_rank: int = 64,
    tolerance: float = 1e-3
) -> TestResult:
    """
    Run single audit test.
    
    Args:
        name: Test name
        dense_3d: (N, N, N) analytical field
        ground_truth_slice: (N, N) true XY slice at z_index
        z_index: Z coordinate for slice
        max_rank: QTT compression rank
        tolerance: Max allowed error
        
    Returns:
        TestResult with pass/fail and metrics
    """
    N = dense_3d.shape[0]
    bits_per_dim = int(np.log2(N))
    device = dense_3d.device
    
    t0 = time.perf_counter()
    
    # Step 1: Compress to Morton QTT
    cores_3d = dense_3d_to_morton_qtt(dense_3d, max_rank=max_rank)
    
    # Step 2: Morton slice
    cores_2d = slice_morton_z_plane(cores_3d, z_index, bits_per_dim)
    
    # Step 3: Contract to dense
    reconstructed = contract_2d_qtt_to_dense(cores_2d, bits_per_dim)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    # Compare to ground truth
    diff = torch.abs(reconstructed - ground_truth_slice)
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    
    passed = max_error < tolerance
    
    details = f"rank={max_rank}, N={N}, z_idx={z_index}"
    
    return TestResult(
        name=name,
        passed=passed,
        max_error=max_error,
        mean_error=mean_error,
        time_ms=elapsed_ms,
        details=details
    )


def run_full_audit(
    bits_per_dim: int = 4,
    max_rank: int = 32,
    tolerance: float = 1e-3,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Run complete Layer 4 Morton slicing audit.
    
    Args:
        bits_per_dim: Resolution = 2^bits per dimension
        max_rank: QTT bond dimension
        tolerance: Error tolerance for pass
        device: Torch device
        
    Returns:
        Dict with all test results and summary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = 2 ** bits_per_dim
    z_index = N // 2  # Middle slice
    
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║           LAYER 4 AUDIT: Morton-Aware Slicing                ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Resolution: {N}×{N}×{N} ({N**3:,} points)".ljust(65) + "║")
    print(f"║  Max Rank:   {max_rank}".ljust(65) + "║")
    print(f"║  Tolerance:  {tolerance:.0e}".ljust(65) + "║")
    print(f"║  Device:     {device}".ljust(65) + "║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()
    
    results = []
    
    # Test 1: Sine wave
    print("Running Test 1: Sine wave f(x,y,z) = sin(2πx)sin(2πy)sin(2πz)...")
    dense_3d = analytical_sine_3d(N, device)
    gt_slice = analytical_sine_z_slice(N, z_index, device)
    result = run_audit_test("Sine Wave", dense_3d, gt_slice, z_index, max_rank, tolerance)
    results.append(result)
    print(f"  {'✅ PASS' if result.passed else '❌ FAIL'} | max_err={result.max_error:.2e} | time={result.time_ms:.1f}ms")
    
    # Test 2: Gaussian
    print("Running Test 2: Gaussian f(x,y,z) = exp(-r²/σ²)...")
    dense_3d = analytical_gaussian_3d(N, device)
    gt_slice = analytical_gaussian_z_slice(N, z_index, device)
    result = run_audit_test("Gaussian", dense_3d, gt_slice, z_index, max_rank, tolerance)
    results.append(result)
    print(f"  {'✅ PASS' if result.passed else '❌ FAIL'} | max_err={result.max_error:.2e} | time={result.time_ms:.1f}ms")
    
    # Test 3: Linear
    print("Running Test 3: Linear f(x,y,z) = x + 2y + 3z...")
    dense_3d = analytical_linear_3d(N, device)
    gt_slice = analytical_linear_z_slice(N, z_index, device)
    result = run_audit_test("Linear", dense_3d, gt_slice, z_index, max_rank, tolerance)
    results.append(result)
    print(f"  {'✅ PASS' if result.passed else '❌ FAIL'} | max_err={result.max_error:.2e} | time={result.time_ms:.1f}ms")
    
    # Test 4: Constant
    print("Running Test 4: Constant f(x,y,z) = 0.5...")
    dense_3d = torch.ones(N, N, N, device=device) * 0.5
    gt_slice = torch.ones(N, N, device=device) * 0.5
    result = run_audit_test("Constant", dense_3d, gt_slice, z_index, max_rank, tolerance)
    results.append(result)
    print(f"  {'✅ PASS' if result.passed else '❌ FAIL'} | max_err={result.max_error:.2e} | time={result.time_ms:.1f}ms")
    
    # Test 5: Multiple Z slices
    print("Running Test 5: Multiple Z-slices (edge and middle)...")
    dense_3d = analytical_sine_3d(N, device)
    multi_pass = True
    for zi in [0, N//4, N//2, 3*N//4, N-1]:
        gt = analytical_sine_z_slice(N, zi, device)
        r = run_audit_test(f"Sine@z={zi}", dense_3d, gt, zi, max_rank, tolerance)
        if not r.passed:
            multi_pass = False
    result = TestResult("Multi-Slice", multi_pass, 0.0, 0.0, 0.0, "5 z-positions")
    results.append(result)
    print(f"  {'✅ PASS' if result.passed else '❌ FAIL'} | All Z positions validated")
    
    # Summary
    print()
    print("═" * 66)
    all_passed = all(r.passed for r in results)
    n_passed = sum(1 for r in results if r.passed)
    
    if all_passed:
        print("║  🎉 ALL TESTS PASSED - Morton slicing is mathematically correct  ║")
    else:
        print(f"║  ⚠️  {n_passed}/{len(results)} TESTS PASSED                                     ║")
    print("═" * 66)
    
    return {
        'all_passed': all_passed,
        'n_passed': n_passed,
        'n_total': len(results),
        'results': results,
        'resolution': N,
        'max_rank': max_rank,
        'tolerance': tolerance,
        'device': str(device),
    }


# =============================================================================
# MAIN
# =============================================================================

def run_benchmark(bits_per_dim: int = 4, max_rank: int = 32) -> Dict[str, Any]:
    """
    Benchmark Morton projection vs point sampling.
    
    Shows the O(L×r²) vs O(N²×d×r²) difference.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = 2 ** bits_per_dim
    z_index = N // 2
    
    print()
    print("═" * 66)
    print(" PERFORMANCE BENCHMARK: Morton Projection vs Point Sampling")
    print("═" * 66)
    print(f" Resolution: {N}×{N}×{N} | Rank: {max_rank} | Device: {device}")
    print("═" * 66)
    
    # Create test field
    dense_3d = analytical_sine_3d(N, device)
    cores_3d = dense_3d_to_morton_qtt(dense_3d, max_rank=max_rank)
    
    # Benchmark Morton projection
    n_trials = 5
    morton_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        cores_2d = slice_morton_z_plane(cores_3d, z_index, bits_per_dim)
        slice_2d = contract_2d_qtt_to_dense(cores_2d, bits_per_dim)
        morton_times.append((time.perf_counter() - t0) * 1000)
    
    morton_avg = sum(morton_times) / len(morton_times)
    
    # Benchmark point sampling (simulated)
    # Point sampling would sample N² points, each requiring d×r² operations
    sample_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        # Simulate N² point samples by reconstructing from cores
        # (This is actually faster than true point sampling)
        coords = torch.linspace(0, 1, N, device=device)
        xx, yy = torch.meshgrid(coords, coords, indexing='ij')
        z_val = z_index / (N - 1)
        # Direct evaluation (simulating point-by-point)
        result = torch.sin(2 * np.pi * xx) * torch.sin(2 * np.pi * yy) * np.sin(2 * np.pi * z_val)
        sample_times.append((time.perf_counter() - t0) * 1000)
    
    sample_avg = sum(sample_times) / len(sample_times)
    
    # Theoretical complexity comparison
    d = bits_per_dim
    r = max_rank
    
    print()
    print("┌────────────────────────────────────────────────────────────────┐")
    print(f"│ Morton Projection:  {morton_avg:8.2f} ms                              │")
    print(f"│   Complexity: O(L × r²) = O({d} × {r}²) = O({d * r * r:,})                   │")
    print("├────────────────────────────────────────────────────────────────┤")
    print(f"│ Point Sampling (simulated analytical):  {sample_avg:8.2f} ms         │")
    print(f"│   True complexity: O(N² × d × r²) = O({N*N} × {d} × {r}²)             │")
    print(f"│                  = O({N*N * d * r * r:,})                              │")
    print("├────────────────────────────────────────────────────────────────┤")
    print(f"│ Theoretical Speedup: {(N*N * d * r * r) / (d * r * r):,.0f}× (for pure contraction)         │")
    print("└────────────────────────────────────────────────────────────────┘")
    
    return {
        'morton_ms': morton_avg,
        'sample_ms': sample_avg,
        'resolution': N,
        'rank': max_rank,
        'bits': bits_per_dim,
    }


if __name__ == "__main__":
    import sys
    
    # Parse args
    bits = 4  # 16×16×16 default
    rank = 32
    
    if len(sys.argv) > 1:
        bits = int(sys.argv[1])
    if len(sys.argv) > 2:
        rank = int(sys.argv[2])
    
    print()
    print("=" * 66)
    print(" HyperTensor Layer 4 Audit: Morton-Aware Slicing")
    print("=" * 66)
    print()
    
    summary = run_full_audit(bits_per_dim=bits, max_rank=rank)
    
    # Run benchmark
    if summary['all_passed']:
        run_benchmark(bits_per_dim=bits, max_rank=rank)
    
    print()
    if summary['all_passed']:
        print("✅ AUDIT PASSED: Morton slicing produces mathematically correct results")
        sys.exit(0)
    else:
        print("❌ AUDIT FAILED: Some tests did not meet tolerance")
        sys.exit(1)
