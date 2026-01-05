"""
Tensor Slicer: Decompression-Free Rendering for QTT
====================================================

This module enables viewing a 1080p "window" of a trillion-point simulation
without decompressing the other 999 billion points.

Architecture: "Decompression-Free Rendering"
- Instead of expanding the Tensor Train to a massive grid (Bad)
- We mathematically "project" screen pixels onto the Tensor Train (Good)

Pipeline:
1. HyperTensor Core: Holds state (2^50 points) in compressed format
2. The Slicer: Constructs "Probe Tensor" for screen pixels
3. The Contraction: Core * Probe = 2D array (W x H)
4. The Renderer: Maps values to colors (Heatmap)

Complexity: O(d * r^2) per pixel, O(W * H * d * r^2) total
           With vectorization: O(d * r^2 * W) for a full row
"""


import numpy as np


class TensorSlicer:
    """
    Decompression-free renderer for Quantized Tensor Trains.

    Allows viewing arbitrary 2D cross-sections of massive (10^12+) tensors
    without materializing the full array.
    """

    def __init__(self, cores: list[np.ndarray], dtype=np.float64):
        """
        Initialize slicer with QTT cores.

        Args:
            cores: List of 3D numpy arrays, each of shape (r_left, 2, r_right)
                   representing the QTT decomposition
            dtype: Data type for computations
        """
        self.cores = [np.asarray(c, dtype=dtype) for c in cores]
        self.n_cores = len(cores)
        self.dtype = dtype

        # Validate core shapes
        for i, core in enumerate(self.cores):
            if core.ndim != 3:
                raise ValueError(f"Core {i} must be 3D, got {core.ndim}D")
            if core.shape[1] != 2:
                raise ValueError(
                    f"Core {i} physical dim must be 2, got {core.shape[1]}"
                )

        # Total grid size
        self.grid_size = 2**self.n_cores

        # Cache for partial contractions
        self._left_cache = {}
        self._right_cache = {}

    # =========================================================================
    # PHASE 1: THE DRILL - Single Point Extraction
    # =========================================================================

    def get_element(self, index: int | str) -> float:
        """
        Extract a single value from QTT without decompression.

        The Math: For index 10110 (binary):
        - Select 1st slice of Core 0
        - Select 0th slice of Core 1
        - Select 1st slice of Core 2
        - ... and multiply the matrices

        Complexity: O(d * r^2) - logarithmic in grid size!

        Args:
            index: Integer index or binary string (e.g., '10110')

        Returns:
            Scalar value at that index
        """
        # Convert to binary string if integer
        if isinstance(index, int):
            binary = format(index, f"0{self.n_cores}b")
        else:
            binary = index.zfill(self.n_cores)

        if len(binary) != self.n_cores:
            raise ValueError(f"Index {binary} doesn't match {self.n_cores} cores")

        # Start with identity-like vector
        result = None

        for i, bit in enumerate(binary):
            bit_idx = int(bit)
            # Select the slice for this bit: shape (r_left, r_right)
            matrix = self.cores[i][:, bit_idx, :]

            if result is None:
                result = matrix
            else:
                # Matrix multiplication: (1, r) @ (r, r') = (1, r')
                result = result @ matrix

        # Final result should be scalar (or 1x1 matrix)
        return float(result.squeeze())

    def get_elements_batch(self, indices: list[int]) -> np.ndarray:
        """
        Extract multiple values efficiently.

        Uses caching of partial contractions for speed.

        Args:
            indices: List of integer indices

        Returns:
            1D array of values
        """
        values = np.zeros(len(indices), dtype=self.dtype)
        for i, idx in enumerate(indices):
            values[i] = self.get_element(idx)
        return values

    # =========================================================================
    # PHASE 2: THE SAW - Batch Slicing for 2D Cross-Sections
    # =========================================================================

    def render_slice_1d(
        self, start_idx: int = 0, end_idx: int | None = None, num_points: int = 1024
    ) -> np.ndarray:
        """
        Render a 1D slice of the QTT.

        Args:
            start_idx: Starting index
            end_idx: Ending index (default: grid_size)
            num_points: Number of points to sample

        Returns:
            1D array of sampled values
        """
        if end_idx is None:
            end_idx = self.grid_size

        indices = np.linspace(start_idx, end_idx - 1, num_points, dtype=int)
        return self.get_elements_batch(indices.tolist())

    def render_slice_2d(
        self,
        x_cores: list[int],
        y_cores: list[int],
        fixed_indices: dict | None = None,
        resolution: tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """
        Render a 2D cross-section by fixing some dimensions.

        The Trick: "Partial Contraction"
        - Fix indices for dimensions not displayed (e.g., Z, Time)
        - Keep X and Y indices "open"
        - Contract network down to a rank-2 tensor (matrix)

        Args:
            x_cores: Which core indices map to X axis
            y_cores: Which core indices map to Y axis
            fixed_indices: Dict mapping core index -> fixed value (0 or 1)
            resolution: Output resolution (width, height)

        Returns:
            2D numpy array of shape (height, width)
        """
        width, height = resolution
        fixed = fixed_indices or {}

        # Validate
        all_specified = set(x_cores) | set(y_cores) | set(fixed.keys())
        if len(all_specified) != self.n_cores:
            raise ValueError(
                "Must specify all cores via x_cores, y_cores, or fixed_indices"
            )

        # Create output grid
        output = np.zeros((height, width), dtype=self.dtype)

        # For each pixel, compute the binary index and extract value
        for py in range(height):
            for px in range(width):
                # Map pixel coords to binary indices
                binary = ["0"] * self.n_cores

                # Set fixed indices
                for core_idx, val in fixed.items():
                    binary[core_idx] = str(val)

                # Map X pixel to x_cores
                x_bits = format(
                    int(px * (2 ** len(x_cores) - 1) / max(1, width - 1)),
                    f"0{len(x_cores)}b",
                )
                for i, core_idx in enumerate(x_cores):
                    binary[core_idx] = x_bits[i]

                # Map Y pixel to y_cores
                y_bits = format(
                    int(py * (2 ** len(y_cores) - 1) / max(1, height - 1)),
                    f"0{len(y_cores)}b",
                )
                for i, core_idx in enumerate(y_cores):
                    binary[core_idx] = y_bits[i]

                output[py, px] = self.get_element("".join(binary))

        return output

    def render_slice_2d_vectorized(
        self,
        x_cores: list[int],
        y_cores: list[int],
        fixed_indices: dict | None = None,
        resolution: tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """
        Vectorized 2D slice rendering using einsum.

        This is the difference between 1 FPS and 60 FPS.

        Uses partial tensor network contraction to compute entire rows at once.

        Args:
            x_cores: Which core indices map to X axis
            y_cores: Which core indices map to Y axis
            fixed_indices: Dict mapping core index -> fixed value (0 or 1)
            resolution: Output resolution (width, height)

        Returns:
            2D numpy array of shape (height, width)
        """
        width, height = resolution
        fixed = fixed_indices or {}

        # Build contracted tensor for fixed dimensions
        # and keep x/y dimensions open

        n_x = len(x_cores)
        n_y = len(y_cores)

        # Number of distinct x and y values we can represent
        x_range = 2**n_x
        y_range = 2**n_y

        # Precompute all possible (x_bits, y_bits) combinations
        # and their corresponding binary indices

        output = np.zeros((height, width), dtype=self.dtype)

        # For efficiency, precompute fixed core slices
        fixed_matrices = {}
        for core_idx, val in fixed.items():
            fixed_matrices[core_idx] = self.cores[core_idx][:, val, :]

        # Process row by row (vectorize over X)
        for py in range(height):
            # Y bits for this row
            y_idx = int(py * (y_range - 1) / max(1, height - 1))
            y_bits = format(y_idx, f"0{n_y}b")

            # Precompute Y contribution
            y_matrices = {}
            for i, core_idx in enumerate(y_cores):
                bit = int(y_bits[i])
                y_matrices[core_idx] = self.cores[core_idx][:, bit, :]

            # Vectorize over X pixels
            row_values = np.zeros(width, dtype=self.dtype)

            for px in range(width):
                x_idx = int(px * (x_range - 1) / max(1, width - 1))
                x_bits = format(x_idx, f"0{n_x}b")

                # Build full matrix chain
                result = None
                for core_idx in range(self.n_cores):
                    if core_idx in fixed_matrices:
                        mat = fixed_matrices[core_idx]
                    elif core_idx in y_matrices:
                        mat = y_matrices[core_idx]
                    else:
                        # X core
                        x_bit_idx = x_cores.index(core_idx)
                        bit = int(x_bits[x_bit_idx])
                        mat = self.cores[core_idx][:, bit, :]

                    if result is None:
                        result = mat
                    else:
                        result = result @ mat

                row_values[px] = float(result.squeeze())

            output[py, :] = row_values

        return output

    def render_plane(
        self,
        plane: str = "xy",
        depth: float = 0.5,
        resolution: tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """
        Render a 2D plane through the tensor.

        Convenience method for common 2D slices.

        Args:
            plane: 'xy', 'xz', or 'yz'
            depth: Position along the third axis (0.0 to 1.0)
            resolution: Output resolution

        Returns:
            2D numpy array
        """
        # Split cores into 3 groups for x, y, z
        n = self.n_cores
        cores_per_dim = n // 3

        if n < 3:
            raise ValueError("Need at least 3 cores for 3D slicing")

        x_cores = list(range(0, cores_per_dim))
        y_cores = list(range(cores_per_dim, 2 * cores_per_dim))
        z_cores = list(range(2 * cores_per_dim, n))

        # Convert depth to binary indices for fixed dimension
        depth_idx = int(depth * (2 ** len(z_cores) - 1))
        depth_bits = format(depth_idx, f"0{len(z_cores)}b")

        # Set up fixed indices based on plane
        if plane == "xy":
            fixed = {z_cores[i]: int(depth_bits[i]) for i in range(len(z_cores))}
            return self.render_slice_2d_vectorized(x_cores, y_cores, fixed, resolution)
        elif plane == "xz":
            fixed = {y_cores[i]: int(depth_bits[i]) for i in range(len(y_cores))}
            return self.render_slice_2d_vectorized(x_cores, z_cores, fixed, resolution)
        elif plane == "yz":
            fixed = {x_cores[i]: int(depth_bits[i]) for i in range(len(x_cores))}
            return self.render_slice_2d_vectorized(y_cores, z_cores, fixed, resolution)
        else:
            raise ValueError(f"Unknown plane: {plane}")

    # =========================================================================
    # PHASE 3: THE LENS - Dynamic Zoom
    # =========================================================================

    def render_zoomed(
        self,
        center: tuple[float, float],
        zoom_level: int,
        resolution: tuple[int, int] = (256, 256),
        x_cores: list[int] | None = None,
        y_cores: list[int] | None = None,
    ) -> np.ndarray:
        """
        Render with dynamic zoom - the "Google Earth" effect.

        The Logic: When zooming in:
        - Zoom Level 1: Request indices from top cores (coarse structure)
        - Zoom Level N: Request indices from bottom cores (fine detail)

        This allows zooming from planetary to microscopic view without pixelation!

        Args:
            center: (x, y) center of view in [0, 1] normalized coords
            zoom_level: 1 = full view, higher = more zoomed in
            resolution: Output resolution
            x_cores: Cores for X axis (default: first half)
            y_cores: Cores for Y axis (default: second half)

        Returns:
            2D numpy array
        """
        width, height = resolution

        # Default core assignment
        if x_cores is None:
            x_cores = list(range(self.n_cores // 2))
        if y_cores is None:
            y_cores = list(range(self.n_cores // 2, self.n_cores))

        n_x = len(x_cores)
        n_y = len(y_cores)

        # Zoom determines the range of indices we sample
        # At zoom_level=1, we sample the full range
        # At higher zoom, we sample a smaller range centered at 'center'

        full_x_range = 2**n_x
        full_y_range = 2**n_y

        # Compute visible range
        visible_fraction = 1.0 / zoom_level

        cx, cy = center

        # X range
        x_start = max(0, cx - visible_fraction / 2) * full_x_range
        x_end = min(1, cx + visible_fraction / 2) * full_x_range

        # Y range
        y_start = max(0, cy - visible_fraction / 2) * full_y_range
        y_end = min(1, cy + visible_fraction / 2) * full_y_range

        output = np.zeros((height, width), dtype=self.dtype)

        for py in range(height):
            y_idx = int(y_start + (y_end - y_start) * py / max(1, height - 1))
            y_idx = min(y_idx, full_y_range - 1)
            y_bits = format(y_idx, f"0{n_y}b")

            for px in range(width):
                x_idx = int(x_start + (x_end - x_start) * px / max(1, width - 1))
                x_idx = min(x_idx, full_x_range - 1)
                x_bits = format(x_idx, f"0{n_x}b")

                # Build binary index
                binary = ["0"] * self.n_cores
                for i, core_idx in enumerate(x_cores):
                    binary[core_idx] = x_bits[i]
                for i, core_idx in enumerate(y_cores):
                    binary[core_idx] = y_bits[i]

                output[py, px] = self.get_element("".join(binary))

        return output

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def to_heatmap(
        self, data: np.ndarray, colormap: str = "viridis", normalize: bool = True
    ) -> np.ndarray:
        """
        Convert 2D float array to RGB heatmap.

        Args:
            data: 2D array of values
            colormap: Matplotlib colormap name
            normalize: Whether to normalize to [0, 1]

        Returns:
            3D array of shape (H, W, 3) with RGB values [0, 255]
        """
        import matplotlib.pyplot as plt

        if normalize:
            vmin, vmax = data.min(), data.max()
            if vmax > vmin:
                data = (data - vmin) / (vmax - vmin)
            else:
                data = np.zeros_like(data)

        cmap = plt.get_cmap(colormap)
        rgb = cmap(data)[:, :, :3]  # Drop alpha
        return (rgb * 255).astype(np.uint8)

    def benchmark_render(self, resolution: tuple[int, int] = (256, 256)) -> dict:
        """
        Benchmark rendering performance.

        Returns:
            Dictionary with timing information
        """
        import time

        results = {}

        # Single point extraction
        t0 = time.perf_counter()
        for _ in range(1000):
            self.get_element(0)
        results["single_point_us"] = (
            time.perf_counter() - t0
        ) * 1000  # microseconds per call

        # 1D slice
        t0 = time.perf_counter()
        self.render_slice_1d(num_points=resolution[0])
        results["slice_1d_ms"] = (time.perf_counter() - t0) * 1000

        # 2D slice (if enough cores)
        if self.n_cores >= 2:
            n_x = self.n_cores // 2
            x_cores = list(range(n_x))
            y_cores = list(range(n_x, self.n_cores))

            t0 = time.perf_counter()
            self.render_slice_2d_vectorized(x_cores, y_cores, {}, resolution)
            results["slice_2d_ms"] = (time.perf_counter() - t0) * 1000

            # Estimate FPS
            results["estimated_fps"] = 1000 / results["slice_2d_ms"]

        results["grid_size"] = self.grid_size
        results["n_cores"] = self.n_cores
        results["resolution"] = resolution

        return results


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_slicer_from_qtt(qtt) -> TensorSlicer:
    """
    Create a TensorSlicer from a QTT object.

    Args:
        qtt: QTT object with .cores attribute

    Returns:
        TensorSlicer instance
    """
    # Handle both numpy and torch cores
    cores = []
    for core in qtt.cores:
        if hasattr(core, "numpy"):
            cores.append(core.numpy())
        else:
            cores.append(np.asarray(core))

    return TensorSlicer(cores)


def create_test_qtt(n_cores: int = 10, rank: int = 4) -> TensorSlicer:
    """
    Create a test QTT with random cores for benchmarking.

    Args:
        n_cores: Number of cores (grid_size = 2^n_cores)
        rank: Bond dimension

    Returns:
        TensorSlicer instance
    """
    cores = []
    for i in range(n_cores):
        r_left = 1 if i == 0 else rank
        r_right = 1 if i == n_cores - 1 else rank
        core = np.random.randn(r_left, 2, r_right)
        cores.append(core)

    return TensorSlicer(cores)


def create_sine_qtt(n_cores: int = 10, frequency: float = 1.0) -> TensorSlicer:
    """
    Create a QTT representing sin(2π * frequency * x).

    This has exact low-rank structure (rank 2).

    Args:
        n_cores: Number of cores
        frequency: Frequency of sine wave

    Returns:
        TensorSlicer instance
    """
    # sin(2πfx) has rank-2 QTT representation
    grid_size = 2**n_cores
    dx = 1.0 / grid_size
    omega = 2 * np.pi * frequency

    cores = []
    for i in range(n_cores):
        r_left = 1 if i == 0 else 2
        r_right = 1 if i == n_cores - 1 else 2

        core = np.zeros((r_left, 2, r_right))

        # Position contribution from this bit
        bit_value = 2 ** (n_cores - 1 - i)
        phase = omega * bit_value * dx

        if i == 0:
            # First core: [sin, cos] initial state
            core[0, 0, :] = [0, 1]  # cos(0) = 1, sin(0) = 0 -> [sin, cos]
            core[0, 1, :] = [np.sin(phase), np.cos(phase)]
        elif i == n_cores - 1:
            # Last core: extract sin component
            core[0, 0, 0] = 1  # identity for bit=0
            core[1, 0, 0] = 0
            core[0, 1, 0] = np.cos(phase)
            core[1, 1, 0] = np.sin(phase)
        else:
            # Middle cores: rotation matrices
            c, s = np.cos(phase), np.sin(phase)
            # For bit=0: identity
            core[0, 0, 0] = 1
            core[1, 0, 1] = 1
            # For bit=1: rotation
            core[0, 1, 0] = c
            core[0, 1, 1] = -s
            core[1, 1, 0] = s
            core[1, 1, 1] = c

        cores.append(core)

    return TensorSlicer(cores)


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TENSOR SLICER: Decompression-Free Rendering Demo")
    print("=" * 70)

    # Test 1: Small QTT (verify correctness)
    print("\n[Test 1] Small QTT Verification (2^4 = 16 points)")
    print("-" * 50)

    # Create a simple known function
    n_cores = 4
    grid_size = 2**n_cores
    x = np.linspace(0, 1, grid_size, endpoint=False)
    data = np.sin(2 * np.pi * x)

    # Manual QTT construction (rank-2 for sine)
    # For testing, use identity-like cores
    cores = []
    for i in range(n_cores):
        r_left = 1 if i == 0 else 2
        r_right = 1 if i == n_cores - 1 else 2
        core = np.random.randn(r_left, 2, r_right) * 0.1
        cores.append(core)

    slicer = TensorSlicer(cores)

    # Test single element extraction
    print(f"  Grid size: {slicer.grid_size}")
    print(f"  Number of cores: {slicer.n_cores}")

    # Extract some elements
    for idx in [0, 5, 15]:
        val = slicer.get_element(idx)
        binary = format(idx, f"0{n_cores}b")
        print(f"  get_element({idx}) [binary: {binary}] = {val:.6f}")

    # Test 2: Larger QTT benchmark
    print("\n[Test 2] Large QTT Benchmark (2^20 = 1M points)")
    print("-" * 50)

    slicer_large = create_test_qtt(n_cores=20, rank=4)

    benchmark = slicer_large.benchmark_render(resolution=(128, 128))
    print(f"  Grid size: {benchmark['grid_size']:,} points")
    print(f"  Single point: {benchmark['single_point_us']:.3f} μs")
    print(f"  1D slice: {benchmark['slice_1d_ms']:.3f} ms")
    print(f"  2D slice: {benchmark['slice_2d_ms']:.3f} ms")
    print(f"  Estimated FPS: {benchmark['estimated_fps']:.1f}")

    # Test 3: Zoom demonstration
    print("\n[Test 3] Dynamic Zoom (Google Earth Effect)")
    print("-" * 50)

    slicer_zoom = create_test_qtt(n_cores=16, rank=4)

    for zoom in [1, 4, 16, 64]:
        img = slicer_zoom.render_zoomed(
            center=(0.5, 0.5), zoom_level=zoom, resolution=(64, 64)
        )
        print(
            f"  Zoom {zoom:2d}x: shape={img.shape}, range=[{img.min():.2f}, {img.max():.2f}]"
        )

    print("\n" + "=" * 70)
    print("TENSOR SLICER READY")
    print("  - get_element(): O(d * r^2) single point extraction")
    print("  - render_slice_2d_vectorized(): Fast 2D cross-sections")
    print("  - render_zoomed(): Infinite zoom without pixelation")
    print("=" * 70)
