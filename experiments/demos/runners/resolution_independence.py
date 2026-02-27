#!/usr/bin/env python3
"""
Resolution Independence Demo
=============================

Demonstrates the core insight:

    A field represented in QTT form can be queried at arbitrary points
    WITHOUT ever materializing the dense grid, while memory stays bounded
    *for bounded rank r*.

The FIELD is the physical reality (density, velocity, vorticity).
QTT is how we REPRESENT that field compactly.

Key insight:
    - Dense storage: O(N^d) memory, resolution-dependent
    - QTT storage:   O(d × r²) memory, resolution-INDEPENDENT for bounded rank r

This demo has TWO modes:

    MODE 1: RECONSTRUCT (for comparison)
        - Fully contracts QTT to dense 256×256, then resamples
        - Proves: storage is small, reconstruction works
        - But: still materializes dense grid

    MODE 2: ORACLE (the real "WHOA")
        - Evaluates QTT at arbitrary (x,y) points via partial contraction
        - NEVER materializes the dense source grid during sampling
        - Proves: true resolution-independent field oracle

Usage:
    python demos/resolution_independence.py              # Both modes
    python demos/resolution_independence.py --interactive  # With visualization
    python demos/resolution_independence.py --oracle-only  # Only oracle mode
"""

import sys
import os
import time
import argparse
import math

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# Import tt_svd from the library (canonical implementation)
from tensornet.cfd.qtt import tt_svd as _lib_tt_svd


def format_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def dense_memory_for_resolution(resolution: int, dims: int = 2, dtype_bytes: int = 4) -> int:
    """Calculate memory for dense storage at given resolution."""
    return (resolution ** dims) * dtype_bytes


def print_header():
    """Print demo header."""
    print("\n" + "=" * 74)
    print("  RESOLUTION INDEPENDENCE DEMO")
    print("  A field represented in QTT form (for bounded rank r)")
    print("=" * 74)
    print("""
  The FIELD is the physical quantity: Taylor-Green vorticity w(x,y).
  QTT is how we REPRESENT that field in compressed tensor-train format.

  Key property (for bounded rank r):
    - Dense storage scales as O(N^d) -- doubles resolution = 4x memory
    - QTT storage scales as O(d*r^2) -- resolution-INDEPENDENT

  TWO MODES:

    MODE 1: RECONSTRUCT
      - Full contraction to dense grid, then resample
      - Proves storage is small, reconstruction works
      - Still materializes dense 256x256 internally

    MODE 2: ORACLE (the real demo)
      - Pointwise evaluation via partial contraction
      - NEVER materializes the dense source grid during sampling
      - True field oracle behavior

  Dense memory shown for comparison; this run uses a 256x256 source field.
""")


# =============================================================================
# Global counter: tracks SOURCE grid materializations (O(N²) at source res)
# Output allocations O(resolution²) are fine and not counted.
# =============================================================================
DENSE_SOURCE_MATERIALIZATION_COUNT = 0


def create_taylor_green_vorticity(resolution: int, time: float = 0.0, nu: float = 0.01) -> np.ndarray:
    """
    Create actual Taylor-Green vorticity field.
    
    ω(x,y,t) = 2k cos(kx) cos(ky) exp(-2νk²t)
    
    where k = 2π (single wavelength in [0,1]² domain).
    
    Args:
        resolution: Grid resolution
        time: Physical time (for decay)
        nu: Kinematic viscosity
        
    Returns:
        Vorticity field as 2D numpy array
    """
    k = 2 * np.pi  # Wavenumber for [0,1] domain
    decay = np.exp(-2 * nu * k**2 * time)
    
    x = np.linspace(0, 1, resolution, endpoint=False)
    y = np.linspace(0, 1, resolution, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Vorticity: ω = ∂v/∂x - ∂u/∂y = 2k cos(kx) cos(ky) exp(-2νk²t)
    omega = 2 * k * np.cos(k * X) * np.cos(k * Y) * decay
    
    return omega.astype(np.float32)


# =============================================================================
# QTT Implementation - Uses library implementation from tensornet.cfd.qtt
# =============================================================================

def tt_svd(
    tensor: torch.Tensor,
    shape: tuple,
    chi_max: int = 32,
    tol: float = 1e-10
) -> tuple:
    """
    Tensor Train SVD decomposition.
    
    Decomposes a tensor T[i₁,i₂,...,i_L] into a train of tensors:
    T ≈ A₁[i₁] · A₂[i₂] · ... · A_L[i_L]
    
    where each A_k has shape (χ_{k-1}, d_k, χ_k).
    
    This is a wrapper around tensornet.cfd.qtt.tt_svd for API compatibility.
    See the library module for the canonical implementation.
    """
    # Use library implementation (normalize=False to match original behavior)
    return _lib_tt_svd(tensor, shape, chi_max=chi_max, tol=tol, normalize=False)


def field_to_qtt(field: np.ndarray, chi_max: int = 32, tol: float = 1e-10) -> dict:
    """
    Compress a 2D field to QTT format using real TT-SVD.
    
    Args:
        field: 2D numpy array
        chi_max: Maximum bond dimension
        tol: Truncation tolerance
        
    Returns:
        Dictionary with QTT data
    """
    tensor = torch.from_numpy(field.flatten()).float()
    original_shape = field.shape
    N = tensor.numel()
    
    # Pad to power of 2
    N_padded = 1 << (N - 1).bit_length()
    if N_padded > N:
        padded = torch.zeros(N_padded)
        padded[:N] = tensor
        tensor = padded
    
    num_qubits = int(math.log2(N_padded))
    qtt_shape = tuple([2] * num_qubits)
    
    # Real TT-SVD (returns cores, truncation_error, norm)
    cores, truncation_error, _norm = tt_svd(tensor, qtt_shape, chi_max=chi_max, tol=tol)
    
    # Memory calculation
    memory = sum(c.numel() * c.element_size() for c in cores)
    bond_dims = [c.shape[2] for c in cores[:-1]]
    
    return {
        'cores': cores,
        'original_shape': original_shape,
        'num_qubits': num_qubits,
        'truncation_error': truncation_error,
        'memory_bytes': memory,
        'bond_dims': bond_dims,
        'max_rank': max(bond_dims) if bond_dims else 1,
    }


# =============================================================================
# TRUE ORACLE MODE: Pointwise evaluation WITHOUT densification
# =============================================================================

def eval_qtt_at_point(qtt: dict, x: float, y: float) -> float:
    """
    Evaluate QTT at a single point (x, y) in [0, 1]² via partial contraction.
    
    This is the TRUE field oracle: O(L × r²) per point, NO dense grid.
    
    The QTT stores a flattened 2D field. To evaluate at (x, y):
    1. Convert (x, y) to grid indices (ix, iy)
    2. Compute linear index: idx = iy * Nx + ix (row-major)
    3. Convert to binary: [b0, b1, ..., b_{L-1}]
    4. Contract: result = Π_k core[k][:, b_k, :]
    
    Args:
        qtt: QTT dictionary from field_to_qtt
        x, y: Coordinates in [0, 1]
        
    Returns:
        Field value at (x, y)
    """
    cores = qtt['cores']
    original_shape = qtt['original_shape']
    num_qubits = qtt['num_qubits']
    
    Ny, Nx = original_shape
    
    # Convert normalized coords to grid indices (periodic BC)
    ix = int(np.floor(x * Nx)) % Nx
    iy = int(np.floor(y * Ny)) % Ny
    
    # Linear index (row-major: y * Nx + x)
    linear_idx = iy * Nx + ix
    
    # Get binary representation (MSB first: dimension 0 = most significant bit)
    # TT-SVD sweeps dim 0 -> dim L-1, so dim 0 is the MSB
    bits = [(linear_idx >> (num_qubits - 1 - k)) & 1 for k in range(num_qubits)]
    
    # Contract tensor train at these indices
    # Start with first core slice: shape (1, r1)
    result = cores[0][0, bits[0], :]  # shape: (r1,)
    
    # Contract through remaining cores
    for k in range(1, num_qubits):
        # core[k] has shape (r_{k-1}, 2, r_k)
        # Select the bit-th slice: shape (r_{k-1}, r_k)
        core_slice = cores[k][:, bits[k], :]
        # Contract: (r_{k-1},) @ (r_{k-1}, r_k) -> (r_k,)
        result = result @ core_slice
    
    # result is now shape (1,) - squeeze to scalar
    return result.squeeze().item()


def eval_qtt_at_grid_oracle(qtt: dict, resolution: int) -> np.ndarray:
    """
    Evaluate QTT at a grid of points via TRUE oracle mode.
    
    This NEVER materializes the dense source grid. Each point is
    evaluated independently via partial contraction.
    
    Complexity: O(resolution² × L × r²) but NO O(N²) dense allocation.
    
    For resolution << source_resolution, this is the "WHOA" moment:
    we query fewer points than the source grid has, and we never
    build that source grid.
    
    Args:
        qtt: QTT dictionary
        resolution: Output grid resolution
        
    Returns:
        2D numpy array at target resolution
    """
    output = np.zeros((resolution, resolution), dtype=np.float32)
    
    for iy in range(resolution):
        y = (iy + 0.5) / resolution  # Cell-centered
        for ix in range(resolution):
            x = (ix + 0.5) / resolution
            output[iy, ix] = eval_qtt_at_point(qtt, x, y)
    
    return output


def eval_qtt_at_grid_oracle_loop(qtt: dict, resolution: int) -> np.ndarray:
    """
    Oracle grid evaluation via nested loops (no dense source materialization).
    
    This evaluates each point independently via partial TT contraction.
    Complexity: O(resolution² × L × r²) with NO O(N²) source allocation.
    
    Note: This is a simple loop implementation. For production use,
    caching partial contractions or batched torch ops would be faster.
    
    Args:
        qtt: QTT dictionary
        resolution: Output grid resolution
        
    Returns:
        2D numpy array at target resolution (O(resolution²) allocation, which is fine)
    """
    cores = qtt['cores']
    original_shape = qtt['original_shape']
    num_qubits = qtt['num_qubits']
    
    Ny, Nx = original_shape
    # This O(resolution²) allocation is fine - it's the OUTPUT, not the source
    output = np.zeros((resolution, resolution), dtype=np.float32)
    
    # For each output pixel
    for out_iy in range(resolution):
        y = (out_iy + 0.5) / resolution
        iy = int(np.floor(y * Ny)) % Ny
        
        for out_ix in range(resolution):
            x = (out_ix + 0.5) / resolution
            ix = int(np.floor(x * Nx)) % Nx
            
            # Linear index (row-major)
            linear_idx = iy * Nx + ix
            
            # MSB-first bit extraction (matches TT-SVD core ordering)
            bits = [(linear_idx >> (num_qubits - 1 - k)) & 1 for k in range(num_qubits)]
            
            # Contract TT at these indices
            result = cores[0][0, bits[0], :]
            for k in range(1, num_qubits):
                result = result @ cores[k][:, bits[k], :]
            
            output[out_iy, out_ix] = result.squeeze().item()
    
    return output


def qtt_to_field(qtt: dict, target_resolution: int = None) -> np.ndarray:
    """
    Reconstruct field from QTT via full contraction.
    
    WARNING: This MATERIALIZES the dense grid. Use eval_qtt_at_grid_oracle()
    for true resolution-independent sampling.
    
    Args:
        qtt: QTT dictionary from field_to_qtt
        target_resolution: If specified, resample to this resolution
        
    Returns:
        2D numpy array
    """
    global DENSE_SOURCE_MATERIALIZATION_COUNT
    DENSE_SOURCE_MATERIALIZATION_COUNT += 1
    
    cores = qtt['cores']
    original_shape = qtt['original_shape']
    
    # Full contraction
    vec = cores[0]  # (1, d, chi)
    for core in cores[1:]:
        vec = torch.tensordot(vec, core, dims=([-1], [0]))
    
    # Squeeze boundaries and flatten
    vec = vec.squeeze(0).squeeze(-1).flatten()
    
    # Unpad
    original_size = original_shape[0] * original_shape[1]
    vec = vec[:original_size]
    
    # Reshape
    field = vec.reshape(original_shape).numpy()
    
    # Resample if requested
    if target_resolution is not None and target_resolution != original_shape[0]:
        # Simple resample via interpolation
        try:
            from scipy.ndimage import zoom
            factor = target_resolution / original_shape[0]
            field = zoom(field, factor, order=1)
        except ImportError:
            # Fallback: nearest-neighbor via numpy
            old_res = original_shape[0]
            new_res = target_resolution
            indices = np.linspace(0, old_res - 1, new_res).astype(int)
            field = field[np.ix_(indices, indices)]
    
    return field


def sample_qtt_at_resolution(qtt: dict, resolution: int) -> np.ndarray:
    """
    Sample QTT at specified resolution.
    
    Method: Contract to full field at original resolution,
    then resample to target. This is the honest approach.
    
    For truly resolution-independent sampling, we would need
    to implement QTT evaluation at arbitrary points via
    partial contractions with interpolation weights.
    
    Args:
        qtt: QTT dictionary
        resolution: Target resolution
        
    Returns:
        2D numpy array at target resolution
    """
    return qtt_to_field(qtt, target_resolution=resolution)


class QTTField:
    """
    A continuous field stored in Quantized Tensor Train format.
    
    This wraps real TT-SVD compression of a dense analytic field.
    
    TWO SAMPLING MODES:
        sample_reconstruct(): Full contraction then resample (materializes dense)
        sample_oracle():      Pointwise evaluation (NEVER materializes dense)
    """
    
    def __init__(self, source_resolution: int = 256, chi_max: int = 32, 
                 tol: float = 1e-10, nu: float = 0.01):
        """
        Create a QTT-compressed Taylor-Green vorticity field.
        
        Args:
            source_resolution: Resolution for initial dense field (only for compression)
            chi_max: Maximum bond dimension
            tol: Truncation tolerance for TT-SVD
            nu: Kinematic viscosity for Taylor-Green decay
        """
        self.source_resolution = source_resolution
        self.chi_max = chi_max
        self.tol = tol
        self.nu = nu
        self.time = 0.0
        
        # Create and compress the initial field
        self._compress_field()
    
    def _compress_field(self):
        """Create dense Taylor-Green and compress to QTT."""
        # Create actual Taylor-Green vorticity (dense, for compression only)
        dense = create_taylor_green_vorticity(
            self.source_resolution, 
            time=self.time, 
            nu=self.nu
        )
        
        # Store field range for consistent visualization
        self._vmin = dense.min()
        self._vmax = dense.max()
        
        # Compress via real TT-SVD
        self.qtt = field_to_qtt(dense, chi_max=self.chi_max, tol=self.tol)
    
    @property
    def memory_bytes(self) -> int:
        """Total memory of QTT cores."""
        return self.qtt['memory_bytes']
    
    @property
    def max_rank(self) -> int:
        """Maximum bond dimension achieved."""
        return self.qtt['max_rank']
    
    @property
    def truncation_error(self) -> float:
        """Truncation error from TT-SVD."""
        return self.qtt['truncation_error']
    
    @property
    def num_qubits(self) -> int:
        """Number of QTT cores (= log2 of padded size)."""
        return self.qtt['num_qubits']
    
    @property
    def field_range(self) -> tuple:
        """Fixed (vmin, vmax) for consistent visualization."""
        return (self._vmin, self._vmax)
    
    def sample_reconstruct(self, resolution: int) -> np.ndarray:
        """
        Sample via RECONSTRUCT mode: full contraction then resample.
        
        WARNING: This materializes the dense 256×256 grid internally.
        """
        return qtt_to_field(self.qtt, target_resolution=resolution)
    
    def sample_oracle(self, resolution: int) -> np.ndarray:
        """
        Sample via ORACLE mode: pointwise evaluation.
        
        This NEVER materializes the dense SOURCE grid (O(N²)).
        The output grid (O(resolution²)) is allocated, which is expected.
        """
        return eval_qtt_at_grid_oracle_loop(self.qtt, resolution)
    
    def sample(self, resolution: int, mode: str = 'oracle') -> np.ndarray:
        """
        Sample the field at specified resolution.
        
        Args:
            resolution: Output grid resolution
            mode: 'oracle' (no dense) or 'reconstruct' (dense intermediate)
        """
        if mode == 'oracle':
            return self.sample_oracle(resolution)
        else:
            return self.sample_reconstruct(resolution)
    
    def eval_point(self, x: float, y: float) -> float:
        """Evaluate field at a single point (oracle mode)."""
        return eval_qtt_at_point(self.qtt, x, y)
    
    def evolve(self, dt: float = 0.1):
        """
        Evolve the field forward in time.
        
        Re-creates the analytic Taylor-Green at the new time
        and re-compresses to QTT. Memory stays bounded.
        """
        self.time += dt
        self._compress_field()


def run_demo(interactive: bool = False, oracle_only: bool = False):
    """Run the resolution independence demo."""
    global DENSE_SOURCE_MATERIALIZATION_COUNT
    
    print_header()
    
    # Create the field
    print("Creating Taylor-Green vorticity field...")
    print("  Source: 256×256 dense grid, compressed via TT-SVD")
    print("  χ_max: 32 (maximum bond dimension)")
    print()
    
    # Reset counter before demo
    DENSE_SOURCE_MATERIALIZATION_COUNT = 0
    
    field = QTTField(source_resolution=256, chi_max=32, tol=1e-10)
    
    # Note: compression required one dense creation, but that's setup
    setup_dense_count = DENSE_SOURCE_MATERIALIZATION_COUNT
    DENSE_SOURCE_MATERIALIZATION_COUNT = 0  # Reset for demo
    
    qtt_memory = field.memory_bytes
    print(f"  QTT Memory:        {format_bytes(qtt_memory)}")
    print(f"  Achieved rank:     {field.max_rank}")
    print(f"  Number of cores:   {field.num_qubits}")
    print(f"  Truncation error:  {field.truncation_error:.2e}")
    print()
    
    # ========================================================================
    # VERIFICATION STEP: Prove oracle matches dense reconstruction
    # ========================================================================
    print("-" * 74)
    print("  VERIFICATION: Oracle vs Dense at Random Points")
    print("-" * 74)
    print()
    
    # Reconstruct dense once (this is just for verification)
    verify_dense = field.sample_reconstruct(256)  # Full 256×256 dense
    
    # Pick 50 random points and compare
    np.random.seed(42)  # Reproducible
    n_verify = 50
    max_abs_err = 0.0
    sum_abs_err = 0.0
    
    for _ in range(n_verify):
        # Random physical coordinates in [0, 1)
        x = np.random.random()
        y = np.random.random()
        
        # Oracle value
        oracle_val = field.eval_point(x, y)
        
        # Dense value via nearest-neighbor lookup
        # Note: meshgrid(indexing='ij') means omega[ix, iy]
        ix = int(x * 256) % 256
        iy = int(y * 256) % 256
        dense_val = verify_dense[ix, iy]
        
        err = abs(oracle_val - dense_val)
        max_abs_err = max(max_abs_err, err)
        sum_abs_err += err
    
    mean_abs_err = sum_abs_err / n_verify
    
    print(f"  Compared oracle vs dense at {n_verify} random points:")
    print(f"    Max absolute error:  {max_abs_err:.2e}")
    print(f"    Mean absolute error: {mean_abs_err:.2e}")
    
    # Threshold: max of 1e-5 or 10x truncation error (self-consistent)
    error_threshold = max(1e-5, 10 * field.truncation_error)
    
    if max_abs_err < error_threshold:
        print(f"    ✅ Oracle matches dense (within expected truncation/float error)")
    else:
        print(f"    ❌ Mismatch detected: check bit order / indexing")
    
    print()
    
    # Reset counter after verification
    DENSE_SOURCE_MATERIALIZATION_COUNT = 0
    
    # Memory comparison table
    print("-" * 74)
    print("  MEMORY COMPARISON TABLE")
    print("  (QTT memory is FIXED; dense memory grows as O(N²))")
    print("-" * 74)
    print()
    print("  Note: Dense memory shown for comparison.")
    print("        This run uses a 256×256 source field for compression.")
    print()
    print("  Resolution      │ Points            │ Dense Would Be │ QTT Memory │ Ratio")
    print("  ────────────────┼───────────────────┼────────────────┼────────────┼────────────")
    
    resolutions = [64, 256, 1024, 4096, 16384, 65536, 1048576]
    
    for res in resolutions:
        n_points = res * res
        dense_mem = dense_memory_for_resolution(res)
        memory_ratio = dense_mem / qtt_memory
        
        res_str = f"{format_number(res)}²"
        points_str = format_number(n_points)
        
        print(f"  {res_str:>15} │ {points_str:>17} │ {format_bytes(dense_mem):>14} │ {format_bytes(qtt_memory):>10} │ {memory_ratio:>10,.0f}×")
    
    print()
    print("  ─────────────────────────────────────────────────────────────────────────")
    print(f"  QTT storage: {format_bytes(qtt_memory)} — constant for bounded rank r={field.max_rank}")
    print()
    
    if not oracle_only:
        # MODE 1: RECONSTRUCT (for comparison)
        print("=" * 74)
        print("  MODE 1: RECONSTRUCT (full contraction → resample)")
        print("  This MATERIALIZES the dense 256×256 grid internally.")
        print("=" * 74)
        print()
        
        DENSE_SOURCE_MATERIALIZATION_COUNT = 0
        sample_resolutions = [32, 64, 128, 256]
        
        for res in sample_resolutions:
            start = time.perf_counter()
            samples = field.sample_reconstruct(res)
            elapsed = time.perf_counter() - start
            
            n_points = res * res
            
            print(f"  {res:>4}×{res:<4}: {format_number(n_points):>8} points in {elapsed*1000:>6.1f}ms "
                  f"| range: [{samples.min():>7.2f}, {samples.max():>7.2f}]")
        
        print()
        print(f"  ⚠️  Dense materializations: {DENSE_SOURCE_MATERIALIZATION_COUNT}")
        print("      (This mode builds the full 256×256 grid each time)")
        print()
    
    # MODE 2: ORACLE (the real "WHOA")
    print("=" * 74)
    print("  MODE 2: ORACLE (pointwise evaluation, NO dense grid)")
    print("  This NEVER materializes the dense source grid.")
    print("=" * 74)
    print()
    
    DENSE_SOURCE_MATERIALIZATION_COUNT = 0
    oracle_resolutions = [16, 32, 64, 128]
    
    print("  Evaluating at arbitrary points via partial TT contraction:")
    print("  (Each point: O(L × r²) ops, no O(N²) dense allocation)")
    print()
    
    for res in oracle_resolutions:
        start = time.perf_counter()
        samples = field.sample_oracle(res)
        elapsed = time.perf_counter() - start
        
        n_points = res * res
        points_per_sec = n_points / elapsed if elapsed > 0 else 0
        
        print(f"  {res:>4}×{res:<4}: {format_number(n_points):>8} points in {elapsed*1000:>6.1f}ms "
              f"({points_per_sec/1000:>5.1f}K pts/sec) "
              f"| range: [{samples.min():>7.2f}, {samples.max():>7.2f}]")
    
    print()
    print(f"  ✅ Dense materializations: {DENSE_SOURCE_MATERIALIZATION_COUNT}")
    print("     (Zero! We queried the field without ever building the dense grid)")
    print()
    
    # Single-point oracle demo
    print("-" * 74)
    print("  SINGLE-POINT ORACLE QUERIES")
    print("  Demonstrating true field oracle behavior")
    print("-" * 74)
    print()
    
    test_points = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (0.1, 0.9)]
    
    for x, y in test_points:
        start = time.perf_counter()
        value = field.eval_point(x, y)
        elapsed = time.perf_counter() - start
        print(f"  ω({x:.2f}, {y:.2f}) = {value:>8.4f}  ({elapsed*1e6:.1f} μs)")
    
    print()
    print(f"  Dense materializations after point queries: {DENSE_SOURCE_MATERIALIZATION_COUNT}")
    print()
    
    # Evolution demonstration
    print("-" * 74)
    print("  EVOLVING THE FIELD OVER TIME (Oracle mode)")
    print("  (Taylor-Green decays as exp(-2νk²t))")
    print("-" * 74)
    print()
    print("  Time    │ Field Statistics (64×64 oracle) │ Memory     │ Dense Count")
    print("  ────────┼─────────────────────────────────┼────────────┼────────────")
    
    DENSE_SOURCE_MATERIALIZATION_COUNT = 0
    
    for step in range(5):
        samples = field.sample_oracle(64)  # Oracle mode!
        mem = field.memory_bytes
        
        print(f"  t={field.time:>5.2f} │ min={samples.min():>7.2f} max={samples.max():>7.2f} σ={samples.std():>6.2f} │ {format_bytes(mem):>10} │ {DENSE_SOURCE_MATERIALIZATION_COUNT:>10}")
        
        # Evolve (this does require creating dense for recompression)
        old_count = DENSE_SOURCE_MATERIALIZATION_COUNT
        field.evolve(dt=0.2)
        # Reset because evolve needs dense for compression, that's expected
        DENSE_SOURCE_MATERIALIZATION_COUNT = old_count
    
    print()
    print("-" * 74)
    print("  CONCLUSION")
    print("-" * 74)
    print(f"""
  The FIELD is physical reality: Taylor-Green vorticity ω(x,y,t).
  
  We stored it in QTT format via real TT-SVD compression:
    • Source: 256×256 dense field (for compression)
    • Compressed: {format_bytes(qtt_memory)} with rank r={field.max_rank}
    • Truncation error: {field.truncation_error:.2e}
    
  MODE 1 (Reconstruct): Full contraction to dense, then resample.
    → Proves storage is small, reconstruction works.
    → But: materializes dense 256×256 each time.
    
  MODE 2 (Oracle): Pointwise evaluation via partial contraction.
    → NEVER materializes the dense source grid during sampling.
    → True resolution-independent field oracle.
    → This is the "WHOA" moment for hyperscalers.

  Key insight:
    Changing observation resolution does NOT require changing the field,
    and in Oracle mode, we NEVER build the O(N²) dense grid.

  Qualifier: "resolution-independent" holds *for bounded rank r*.
""")
    
    if interactive:
        run_interactive_visualization(field)


def run_interactive_visualization(field: QTTField):
    """Run interactive matplotlib visualization."""
    global DENSE_SOURCE_MATERIALIZATION_COUNT
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button, RadioButtons
    except ImportError:
        print("  [matplotlib not installed — skipping interactive visualization]")
        return
    
    print()
    print("-" * 74)
    print("  INTERACTIVE VISUALIZATION")
    print("-" * 74)
    print("  Use the slider to change sample resolution.")
    print("  Toggle between ORACLE and RECONSTRUCT modes.")
    print("  Watch the dense materialization counter!")
    print()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(bottom=0.3, wspace=0.3)
    
    # Get fixed color range from initial field
    vmin, vmax = field.field_range
    
    # Initial plot
    resolution = 64
    DENSE_SOURCE_MATERIALIZATION_COUNT = 0
    samples = field.sample_oracle(resolution)
    
    # Use viridis (scientific colormap)
    im = axes[0].imshow(samples, cmap='viridis', origin='lower',
                        extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Taylor-Green Vorticity {resolution}×{resolution}\n(Oracle mode)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im, ax=axes[0], label='ω(x,y)')
    
    # Memory comparison bar chart
    qtt_mem = field.memory_bytes
    resolutions_for_bar = [16, 32, 64, 128]
    dense_mems = [dense_memory_for_resolution(r) / 1024 for r in resolutions_for_bar]  # KB
    qtt_mems = [qtt_mem / 1024] * len(resolutions_for_bar)  # KB
    
    x_pos = np.arange(len(resolutions_for_bar))
    width = 0.35
    
    axes[1].bar(x_pos - width/2, dense_mems, width, 
                label='Dense (would need)', color='#d62728', alpha=0.7)
    axes[1].bar(x_pos + width/2, qtt_mems, width, 
                label=f'QTT (actual, r={field.max_rank})', color='#2ca02c', alpha=0.7)
    
    axes[1].set_ylabel('Memory (KB)')
    axes[1].set_xlabel('Sample Resolution')
    axes[1].set_title('Memory: Dense vs QTT')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'{r}²' for r in resolutions_for_bar])
    axes[1].legend(loc='upper left')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Dense materialization counter display
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].axis('off')
    
    counter_text = axes[2].text(0.5, 0.7, f'Dense Materializations:\n{DENSE_SOURCE_MATERIALIZATION_COUNT}',
                                 fontsize=24, ha='center', va='center',
                                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    mode_text = axes[2].text(0.5, 0.3, 'Mode: ORACLE\n(No dense grid)',
                              fontsize=14, ha='center', va='center',
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    axes[2].set_title('Dense Grid Counter\n(Should stay 0 in Oracle mode!)')
    
    # State
    state = {'mode': 'oracle'}
    
    # Resolution slider
    ax_slider = plt.axes([0.15, 0.15, 0.5, 0.03])
    slider = Slider(ax_slider, 'Resolution', 16, 128, valinit=64, valstep=16)
    
    # Mode toggle
    ax_radio = plt.axes([0.75, 0.05, 0.15, 0.15])
    radio = RadioButtons(ax_radio, ('Oracle', 'Reconstruct'))
    
    # Evolve button
    ax_button = plt.axes([0.15, 0.05, 0.1, 0.05])
    button = Button(ax_button, 'Evolve')
    
    def update(val=None):
        res = int(slider.val)
        start = time.perf_counter()
        
        if state['mode'] == 'oracle':
            samples = field.sample_oracle(res)
        else:
            samples = field.sample_reconstruct(res)
        
        elapsed = time.perf_counter() - start
        
        im.set_data(samples)
        mode_str = 'Oracle' if state['mode'] == 'oracle' else 'Reconstruct'
        axes[0].set_title(f'Taylor-Green {res}×{res} ({elapsed*1000:.1f}ms)\n({mode_str} mode)')
        
        # Update counter
        counter_text.set_text(f'Dense Materializations:\n{DENSE_SOURCE_MATERIALIZATION_COUNT}')
        if DENSE_SOURCE_MATERIALIZATION_COUNT == 0:
            counter_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            counter_text.set_bbox(dict(boxstyle='round', facecolor='salmon', alpha=0.8))
        
        fig.canvas.draw_idle()
    
    def mode_changed(label):
        state['mode'] = 'oracle' if label == 'Oracle' else 'reconstruct'
        mode_str = 'ORACLE\n(No dense grid)' if state['mode'] == 'oracle' else 'RECONSTRUCT\n(Dense intermediate)'
        mode_text.set_text(f'Mode: {mode_str}')
        if state['mode'] == 'oracle':
            mode_text.set_bbox(dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            mode_text.set_bbox(dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        update()
    
    def evolve_step(event):
        field.evolve(dt=0.2)
        vmin, vmax = field.field_range
        im.set_clim(vmin, vmax)
        update()
    
    slider.on_changed(update)
    radio.on_clicked(mode_changed)
    button.on_clicked(evolve_step)
    
    plt.suptitle('Resolution Independence: ORACLE mode queries field WITHOUT dense grid\n'
                 'Toggle to RECONSTRUCT to see dense materializations increase',
                 fontsize=11, fontweight='bold')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Demonstrate resolution-independent field oracle via QTT'
    )
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Launch interactive matplotlib visualization')
    parser.add_argument('--oracle-only', action='store_true',
                        help='Only show oracle mode (skip reconstruct comparison)')
    args = parser.parse_args()
    
    run_demo(interactive=args.interactive, oracle_only=args.oracle_only)


if __name__ == '__main__':
    main()
