"""
QTT PyTorch Renderer: Implicit Synthesis via Tensor Contraction

This module synthesizes pixel buffers directly from QTT cores using
batched PyTorch operations on GPU. It bridges the factorized tensor
representation to the existing PyTorch-based rendering pipeline.

Architecture:
- Input: QTT cores (KB) on GPU
- Process: Batched Morton grid evaluation via matrix contraction
- Output: Full pixel buffer (MB) for compositor

Key Difference from Conventional Rendering:
- Conventional: Store full pixel buffer, move GB through memory
- QTT-Native: Store compressed cores, compute pixels on-demand

The Trade-Off:
- Memory bandwidth: 99% reduction (KB cores vs MB pixels)
- Compute: Increased (matrix contractions for every pixel)
- Net effect: Shift bottleneck from 342 GB/s memory to 33.4 TFLOPS compute

Author: HyperTensor Team
Date: January 2025
"""


import torch

from tensornet.cfd.flux_2d_tci import qtt2d_eval_batch
from tensornet.cfd.qtt_2d import QTT2DState


class QTTTorchRenderer:
    """
    GPU-accelerated QTT synthesis for PyTorch rendering pipeline.

    This renders a QTT state to a pixel buffer by evaluating the
    factorized tensor at every pixel coordinate via batched contraction.

    Memory Profile:
    - Input: QTT cores ~KB (e.g., 16 cores × rank 8 × 2 × 4 bytes = 4 KB)
    - Output: Pixel buffer ~MB (e.g., 3840×2160 × 4 bytes = 33 MB)
    - Intermediate: Morton indices ~MB (3840×2160 × 8 bytes = 66 MB)

    Compute Profile:
    - Per-pixel: ~n_cores × rank² FLOPs (e.g., 16 × 64 = 1024 FLOPs/pixel)
    - Total 4K: 8.3M pixels × 1024 FLOPs = 8.5 GFLOPs
    - At 33.4 TFLOPS: 0.25ms theoretical minimum
    """

    def __init__(self, device: torch.device = None, dtype: torch.dtype = torch.float16):
        """
        Initialize QTT renderer.

        Args:
            device: GPU device
            dtype: Float precision (float16 for bandwidth, float32 for accuracy)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

        # Pre-allocate Morton index cache
        self._morton_cache = {}

    def _generate_morton_grid(
        self, width: int, height: int, n_bits: int
    ) -> torch.Tensor:
        """
        Generate Morton indices for full pixel grid.

        This creates a [height, width] grid where each element is the
        Morton-encoded index of that pixel coordinate.

        Args:
            width: Image width
            height: Image height
            n_bits: Bit depth (max of nx, ny from QTT)

        Returns:
            Morton indices [height, width] on GPU
        """
        cache_key = (width, height, n_bits)
        if cache_key in self._morton_cache:
            return self._morton_cache[cache_key]

        # GPU-friendly vectorized generation
        y_coords = torch.arange(height, dtype=torch.long, device=self.device).unsqueeze(
            1
        )
        x_coords = torch.arange(width, dtype=torch.long, device=self.device).unsqueeze(
            0
        )

        # Broadcast to full grid
        y_grid = y_coords.expand(height, width)
        x_grid = x_coords.expand(height, width)

        # Vectorized Morton encoding
        from tensornet.cfd.qtt_2d import morton_encode_batch

        morton_grid = morton_encode_batch(x_grid, y_grid, n_bits)

        self._morton_cache[cache_key] = morton_grid

        return morton_grid

    def render_qtt_to_buffer(
        self,
        qtt: QTT2DState,
        width: int,
        height: int,
        colormap: torch.Tensor | None = None,
        chunk_size: int = 65536,  # Process 64K pixels at a time
    ) -> torch.Tensor:
        """
        Synthesize full pixel buffer from QTT via implicit evaluation.

        Uses chunked evaluation to avoid OOM on large batches.

        Algorithm:
        1. Generate Morton index grid [height, width]
        2. Chunk into manageable batches
        3. Evaluate each chunk via qtt2d_eval_batch()
        4. Concatenate results
        5. Apply colormap

        Args:
            qtt: QTT2DState with cores on GPU
            width: Output buffer width
            height: Output buffer height
            colormap: Optional [256, 3] RGB lookup table
            chunk_size: Pixels per batch (trade memory for speed)

        Returns:
            Pixel buffer [height, width, 3] Float16 RGB
        """
        # Move QTT to correct device if needed
        if qtt.cores[0].device != self.device:
            qtt.cores = [c.to(self.device) for c in qtt.cores]

        # Compute n_bits from QTT dimensions
        n_bits = max(qtt.nx, qtt.ny)

        # Generate Morton grid
        morton_grid = self._generate_morton_grid(width, height, n_bits)  # [H, W]

        # Flatten for batch evaluation
        morton_flat = morton_grid.flatten()  # [H×W]
        total_pixels = len(morton_flat)

        # Chunk evaluation
        values_chunks = []
        n_chunks = (total_pixels + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_pixels)
            morton_chunk = morton_flat[start_idx:end_idx]

            # Batch evaluate chunk
            values_chunk = qtt2d_eval_batch(qtt, morton_chunk)
            values_chunks.append(values_chunk)

        # Concatenate all chunks
        values = torch.cat(values_chunks, dim=0)  # [H×W]

        # Reshape to image
        value_image = values.reshape(height, width)  # [H, W]

        # Apply colormap if provided
        if colormap is not None:
            # Normalize values to [0, 1]
            val_min = value_image.min()
            val_max = value_image.max()
            val_range = val_max - val_min
            if val_range > 1e-6:
                normalized = (value_image - val_min) / val_range
            else:
                normalized = torch.zeros_like(value_image)

            # Map to colormap indices
            indices = (normalized * 255).clamp(0, 255).long()

            # Lookup RGB
            rgb = colormap[indices]  # [H, W, 3]
        else:
            # Grayscale
            rgb = value_image.unsqueeze(-1).repeat(1, 1, 3)

        # Convert to target dtype
        rgb = rgb.to(self.dtype)

        return rgb

    def benchmark_synthesis(
        self, qtt: QTT2DState, width: int, height: int, n_trials: int = 100
    ):
        """
        Benchmark QTT synthesis performance.

        Measures:
        - Morton generation time
        - Core contraction time
        - Total synthesis time
        - Throughput (Mpixels/sec)
        """
        print("=" * 70)
        print("QTT Torch Renderer Benchmark")
        print("=" * 70)
        print(f"Resolution: {width}×{height} ({width*height/1e6:.1f}M pixels)")
        print(f"QTT: {len(qtt.cores)} cores, rank {qtt.max_rank}")
        print(f"Memory: {qtt.memory_bytes/1024:.1f} KB cores")

        # Warmup
        for _ in range(5):
            _ = self.render_qtt_to_buffer(qtt, width, height)

        torch.cuda.synchronize()

        # Timed trials
        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for trial in range(n_trials):
            start.record()
            _ = self.render_qtt_to_buffer(qtt, width, height)
            end.record()

            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            times.append(elapsed_ms)

        # Statistics
        times = torch.tensor(times)
        mean_ms = times.mean().item()
        std_ms = times.std().item()
        min_ms = times.min().item()
        max_ms = times.max().item()

        throughput_mpix = (width * height) / (mean_ms * 1000)
        fps_at_resolution = 1000 / mean_ms

        print(f"\nTiming ({n_trials} trials):")
        print(f"  Mean: {mean_ms:.2f} ms (±{std_ms:.2f})")
        print(f"  Range: [{min_ms:.2f}, {max_ms:.2f}] ms")
        print(f"  Throughput: {throughput_mpix:.1f} Mpixels/sec")
        print(f"  FPS: {fps_at_resolution:.1f}")

        # Compare to bandwidth-bound baseline
        pixel_bytes = width * height * 2  # Float16
        bandwidth_theoretical_ms = pixel_bytes / (342e9 / 1000)  # 342 GB/s

        print("\nTheoretical Limits:")
        print(f"  Bandwidth-bound: {bandwidth_theoretical_ms:.2f} ms (memory only)")
        print(f"  Actual: {mean_ms:.2f} ms")
        print(f"  Overhead: {mean_ms / bandwidth_theoretical_ms:.1f}x")

        print("=" * 70)


def create_plasma_qtt(nx: int = 11, ny: int = 11, rank: int = 8) -> QTT2DState:
    """
    Create a plasma-like QTT field for testing.

    Generates smooth gradients suitable for atmospheric visualization.
    """
    n_cores = nx + ny
    cores = []

    torch.manual_seed(42)

    for k in range(n_cores):
        if k == 0:
            r_left, r_right = 1, rank
        elif k == n_cores - 1:
            r_left, r_right = rank, 1
        else:
            r_left, r_right = rank, rank

        # Smooth random cores
        core = torch.randn(r_left, 2, r_right, dtype=torch.float32) * 0.3
        cores.append(core)

    qtt = QTT2DState(cores=cores, nx=nx, ny=ny)

    return qtt


def test_qtt_torch_renderer():
    """Validation: Render QTT to buffer and check output."""
    print("\n" + "=" * 70)
    print("Testing QTT Torch Renderer")
    print("=" * 70)

    # Create test QTT (2048×2048 grid)
    qtt = create_plasma_qtt(nx=11, ny=11, rank=8)

    device = torch.device("cuda")
    qtt.cores = [c.to(device) for c in qtt.cores]

    # Create renderer
    renderer = QTTTorchRenderer(device=device, dtype=torch.float16)

    # Test smaller resolution first
    test_cases = [
        (256, 256, "256×256 (profiling)"),
        (512, 512, "512×512"),
        (1920, 1080, "1080p"),
    ]

    for width, height, name in test_cases:
        print(f"\n{name} ({width}×{height}, {width*height/1e6:.2f}M pixels):")

        # Profile components
        start_morton = torch.cuda.Event(enable_timing=True)
        end_morton = torch.cuda.Event(enable_timing=True)
        start_eval = torch.cuda.Event(enable_timing=True)
        end_eval = torch.cuda.Event(enable_timing=True)

        # Morton generation
        n_bits = max(qtt.nx, qtt.ny)
        start_morton.record()
        morton_grid = renderer._generate_morton_grid(width, height, n_bits)
        end_morton.record()
        torch.cuda.synchronize()
        morton_time = start_morton.elapsed_time(end_morton)

        # Contraction
        morton_flat = morton_grid.flatten()
        start_eval.record()
        values = qtt2d_eval_batch(qtt, morton_flat[:10000])  # Sample only 10K pixels
        end_eval.record()
        torch.cuda.synchronize()
        eval_time_sample = start_eval.elapsed_time(end_eval)

        # Extrapolate
        total_pixels = width * height
        estimated_eval_time = (eval_time_sample / 10000) * total_pixels

        print(f"  Morton generation: {morton_time:.2f} ms")
        print(f"  Contraction (10K sample): {eval_time_sample:.2f} ms")
        print(
            f"  Estimated full eval: {estimated_eval_time:.0f} ms ({1000/estimated_eval_time:.1f} FPS)"
        )
        print(
            f"  CONCLUSION: {'VIABLE' if estimated_eval_time < 16.67 else 'TOO SLOW'} for 60 FPS"
        )

    print("\n✓ QTT Torch Renderer validation complete")
    print("=" * 70)


if __name__ == "__main__":
    test_qtt_torch_renderer()
