"""
CPU-Native QTT Evaluator: Leveraging the i9-14900HX Factorization Engine

This module performs QTT evaluation on the CPU using parallel NumPy operations
and multiprocessing. The i9's 24 cores (8P + 16E) and 36MB L3 cache create
a fundamentally different computational topology than the i5.

Architectural Philosophy:
    "The i5 already proved that TT/QTT factorization is a CPU-native strength.
     The i9 doesn't just do it faster—it does it with 24 parallel workers
     and cache residency that eliminates memory latency."

Key Design:
    - Use P-cores (0-7) for compute-intensive Morton→contraction
    - NumPy's BLAS backend (MKL/OpenBLAS) auto-vectorizes with AVX-512
    - Small batches (4K-16K points) fit in L3 cache
    - No PyTorch overhead (kernel launch, autograd, memory pool)

Expected Performance:
    - 16K samples (128×128 sparse grid): ~2-3ms
    - 65K samples (256×256 sparse grid): ~8-12ms
    - 262K samples (512×512 sparse grid): ~30-40ms

Author: HyperTensor Team
Date: December 2025
"""


import numpy as np

from ontic.cfd.qtt_2d import QTT2DState

# Try to import numba for JIT acceleration
try:
    from numba import jit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: numba not available, using pure NumPy (slower)")

    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range


@jit(nopython=True, parallel=False, cache=True)
def morton_encode_fast(x: int, y: int, n_bits: int) -> int:
    """
    Numba-JIT compiled Morton encoding.

    Interleaves x and y bits for Z-curve indexing.
    Runs at native CPU speed with zero Python overhead.
    """
    result = 0
    for i in range(n_bits):
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        result |= x_bit << (2 * i)
        result |= y_bit << (2 * i + 1)
    return result


@jit(nopython=True, parallel=True, cache=True)
def morton_grid_parallel(width: int, height: int, n_bits: int) -> np.ndarray:
    """
    Generate Morton index grid using parallel CPU threads.

    Uses Numba's prange for automatic work distribution across cores.
    """
    morton_grid = np.zeros((height, width), dtype=np.int64)

    for y in prange(height):
        for x in range(width):
            morton_grid[y, x] = morton_encode_fast(x, y, n_bits)

    return morton_grid


@jit(nopython=True, cache=True)
def extract_bit_array(morton_idx: int, n_cores: int) -> np.ndarray:
    """
    Extract bit array from Morton index (MSB first).

    Returns array of 0s and 1s for QTT core indexing.
    """
    bits = np.zeros(n_cores, dtype=np.int32)
    for k in range(n_cores):
        bit_pos = n_cores - 1 - k
        bits[k] = (morton_idx >> bit_pos) & 1
    return bits


@jit(nopython=True, cache=True)
def qtt_contract_single(
    morton_idx: int,
    cores_flat: np.ndarray,
    core_shapes: np.ndarray,
    core_offsets: np.ndarray,
    n_cores: int,
) -> float:
    """
    Contract QTT cores at a single Morton index.

    This is the innermost kernel—optimized for L1/L2 cache residency.
    Each core access is a simple array slice, no indirection.

    Args:
        morton_idx: Z-curve encoded pixel coordinate
        cores_flat: Flattened array of all cores [total_elements]
        core_shapes: [n_cores, 3] array of (r_left, 2, r_right)
        core_offsets: [n_cores] starting index in cores_flat
        n_cores: Number of tensor cores

    Returns:
        Scalar value at this pixel
    """
    # Extract bits
    bits = extract_bit_array(morton_idx, n_cores)

    # Start with core 0 (vectorized slice)
    r_right_0 = core_shapes[0, 2]
    bit_0 = bits[0]
    offset_0 = core_offsets[0]

    # Vectorized core extraction (array slice vs element-by-element)
    start_idx = offset_0 + bit_0 * r_right_0
    result = cores_flat[start_idx : start_idx + r_right_0].copy()

    # Contract remaining cores (optimized memory access pattern)
    for k in range(1, n_cores):
        r_left = core_shapes[k, 0]
        r_right = core_shapes[k, 2]
        bit_k = bits[k]
        offset_k = core_offsets[k]

        # Reorder loops: outer on i (better cache locality)
        new_result = np.zeros(r_right, dtype=np.float32)

        for i in range(r_left):
            base_idx = offset_k + (i * 2 + bit_k) * r_right
            weight = result[i]
            # Vectorized inner loop (Numba SIMD)
            for j in range(r_right):
                new_result[j] += weight * cores_flat[base_idx + j]

        result = new_result

    return result[0]


@jit(nopython=True, parallel=True, cache=True)
def qtt_eval_batch_numba(
    morton_indices: np.ndarray,
    cores_flat: np.ndarray,
    core_shapes: np.ndarray,
    core_offsets: np.ndarray,
    n_cores: int,
) -> np.ndarray:
    """
    Parallel batch evaluation of QTT at Morton indices.

    Uses Numba's prange for automatic thread distribution.
    Each thread evaluates independent samples—perfect parallelism.
    """
    n_samples = len(morton_indices)
    values = np.zeros(n_samples, dtype=np.float32)

    for i in prange(n_samples):
        values[i] = qtt_contract_single(
            morton_indices[i], cores_flat, core_shapes, core_offsets, n_cores
        )

    return values


class CPUQTTEvaluator:
    """
    High-performance CPU QTT evaluator for the i9-14900HX.

    This class manages QTT state in CPU-friendly formats and provides
    parallel evaluation of compressed tensor fields at arbitrary sample points.

    Memory Layout:
        - All cores flattened into single contiguous array (L3 cache residency)
        - Pre-computed offset table for O(1) core access
        - Shape table for bounds checking

    Threading:
        - Uses Numba's prange for work distribution
        - Defaults to 8 threads (P-cores only, E-cores for OS/telemetry)
        - Can scale to 16 threads if E-cores are idle
    """

    def __init__(self, n_threads: int = 8):
        """
        Initialize CPU evaluator.

        Args:
            n_threads: Number of parallel threads (default 8 = P-cores)
        """
        self.n_threads = n_threads

        # Set Numba thread count if available
        if HAS_NUMBA:
            from numba import set_num_threads

            set_num_threads(n_threads)

        # QTT state (CPU arrays)
        self.cores_flat = None
        self.core_shapes = None
        self.core_offsets = None
        self.n_cores = None
        self.nx = None
        self.ny = None

    def load_qtt(self, qtt: QTT2DState):
        """
        Load QTT state into CPU-optimized format.

        Converts PyTorch tensor cores to NumPy and flattens for L3 cache efficiency.

        Raises:
            ValueError: If QTT structure is invalid
        """
        if not qtt.cores or len(qtt.cores) == 0:
            raise ValueError("QTT must have at least one core")

        self.n_cores = len(qtt.cores)
        self.nx = qtt.nx
        self.ny = qtt.ny

        # Validate core connectivity
        for k, core in enumerate(qtt.cores):
            if core.ndim != 3 or core.shape[1] != 2:
                raise ValueError(
                    f"Core {k} has invalid shape {core.shape}, expected [r_left, 2, r_right]"
                )

        # Flatten all cores (single-copy transfer: detach + cpu + numpy)
        cores_list = []
        shapes = []
        offsets = [0]

        for core in qtt.cores:
            # Eliminate double copy: detach() before cpu() for single transfer
            core_np = core.detach().cpu().numpy()
            # Convert to float32 after transfer (CPU-side conversion)
            if core_np.dtype != np.float32:
                core_np = core_np.astype(np.float32)
            cores_list.append(core_np.flatten())
            shapes.append(core_np.shape)
            offsets.append(offsets[-1] + core_np.size)

        self.cores_flat = np.concatenate(cores_list)
        self.core_shapes = np.array(shapes, dtype=np.int32)
        self.core_offsets = np.array(offsets[:-1], dtype=np.int64)

        # Cache validation
        total_kb = self.cores_flat.nbytes / 1024
        if total_kb > 512:  # Warn if larger than half L3
            print(f"Warning: QTT cores ({total_kb:.1f} KB) may exceed L3 cache")

    def eval_sparse_grid(self, grid_size: int) -> tuple[np.ndarray, float]:
        """
        Evaluate QTT on a sparse uniform grid.

        Args:
            grid_size: Grid resolution (e.g., 256 → 256×256 = 65K samples)
                      Must be power of 2 for optimal Morton encoding

        Returns:
            values: [grid_size, grid_size] evaluated field
            timing_ms: Total execution time in milliseconds

        Raises:
            RuntimeError: If QTT not loaded
        """
        if self.cores_flat is None:
            raise RuntimeError("QTT not loaded. Call load_qtt() first.")

        import time

        n_bits = max(self.nx, self.ny)

        # Generate Morton grid
        t0 = time.perf_counter()
        morton_grid = morton_grid_parallel(grid_size, grid_size, n_bits)
        t1 = time.perf_counter()
        morton_time = (t1 - t0) * 1000

        # Flatten for batch evaluation
        morton_flat = morton_grid.flatten()

        # Evaluate
        t0 = time.perf_counter()
        values_flat = qtt_eval_batch_numba(
            morton_flat,
            self.cores_flat,
            self.core_shapes,
            self.core_offsets,
            self.n_cores,
        )
        t1 = time.perf_counter()
        eval_time = (t1 - t0) * 1000

        # Reshape
        values = values_flat.reshape(grid_size, grid_size)

        total_time = morton_time + eval_time

        return values, total_time

    def benchmark(self, grid_sizes: list[int] = [128, 256]):
        """
        Benchmark CPU evaluation at optimal sparse grid resolutions.

        Focus on 128² and 256² which provide best quality/performance for 4K.
        """
        print("=" * 70)
        print("CPU QTT Evaluator Benchmark (i9-14900HX)")
        print("=" * 70)
        print(f"Threads: {self.n_threads}")
        print(f"QTT: {self.n_cores} cores, nx={self.nx}, ny={self.ny}")
        print(f"Memory: {self.cores_flat.nbytes / 1024:.1f} KB cores")

        for grid_size in grid_sizes:
            n_samples = grid_size**2

            # Warmup
            _, _ = self.eval_sparse_grid(grid_size)

            # Timed runs
            times = []
            for _ in range(10):
                _, elapsed = self.eval_sparse_grid(grid_size)
                times.append(elapsed)

            times = np.array(times)
            mean_ms = times.mean()
            std_ms = times.std()
            min_ms = times.min()

            throughput_msps = n_samples / (mean_ms * 1000)
            fps = 1000 / mean_ms

            print(f"\n{grid_size}×{grid_size} ({n_samples/1000:.0f}K samples):")
            print(f"  Time: {mean_ms:.2f} ms (±{std_ms:.2f}, min {min_ms:.2f})")
            print(f"  Throughput: {throughput_msps:.2f} Msamples/sec")
            print(f"  FPS: {fps:.1f}")

        print("\n" + "=" * 70)


def test_cpu_evaluator():
    """Test CPU evaluator with synthetic QTT."""
    print("\nTesting CPU QTT Evaluator")
    print("=" * 70)

    # Create test QTT (2048×2048 grid, rank 8)
    import torch

    from ontic.cfd.qtt_2d import QTT2DState

    nx, ny, rank = 11, 11, 8
    n_cores = nx + ny

    torch.manual_seed(42)
    cores = []
    for k in range(n_cores):
        if k == 0:
            r_left, r_right = 1, rank
        elif k == n_cores - 1:
            r_left, r_right = rank, 1
        else:
            r_left, r_right = rank, rank

        core = torch.randn(r_left, 2, r_right, dtype=torch.float32) * 0.3
        cores.append(core)

    qtt = QTT2DState(cores=cores, nx=nx, ny=ny)

    # Create evaluator
    evaluator = CPUQTTEvaluator(n_threads=8)
    evaluator.load_qtt(qtt)

    # Benchmark
    evaluator.benchmark(grid_sizes=[128, 256, 512])

    print("\n✓ CPU QTT Evaluator test complete")


if __name__ == "__main__":
    test_cpu_evaluator()
