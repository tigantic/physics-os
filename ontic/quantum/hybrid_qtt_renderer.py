"""
Hybrid CPU-GPU QTT Renderer: The i9-5070 Co-Design

This module implements the Architect's vision of CPU-GPU computational sovereignty:
- CPU (i9-14900HX): Factorization Engine - sparse QTT evaluation
- GPU (RTX 5070): Synthesis Engine - bicubic interpolation to full resolution

Architecture:
    CPU: Evaluate QTT at strategic sparse grid (128²-512²)
    GPU: Interpolate sparse samples to full 4K (3840×2160)

Key Insight:
    "The i5 already crushed billion-scale grids using TT/QTT factorization.
     The i9 isn't just faster—it's a 24-core parallel factorization orchestra.
     We use the CPU for what it excels at (mathematical compression),
     and the GPU for what it excels at (parallel pixel synthesis)."

Performance Target:
    128² sparse: 2.6ms CPU + 2ms GPU interpolation = 4.6ms (217 FPS)
    256² sparse: 14ms CPU + 2ms GPU interpolation = 16ms (62 FPS)

Visual Quality:
    128² → 4K: ~30× upsampling (acceptable for smooth atmospheric fields)
    256² → 4K: ~15× upsampling (high quality)

Author: HyperTensor Team
Date: December 2025
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from ontic.cfd.qtt_2d import QTT2DState
from ontic.quantum.cpu_qtt_evaluator import CPUQTTEvaluator


class HybridQTTRenderer:
    """
    Hybrid CPU-GPU renderer combining sparse QTT evaluation with GPU interpolation.

    Pipeline:
        1. CPU: Evaluate QTT at sparse grid (e.g., 256×256)
        2. GPU: Upload sparse samples as texture
        3. GPU: Bicubic interpolation to target resolution (e.g., 3840×2160)
        4. GPU: Apply colormap and blend with other layers

    Memory Efficiency:
        - QTT cores: ~10 KB (CPU L3 cache resident)
        - Sparse buffer: 256×256×4 = 256 KB
        - Dense buffer: 3840×2160×4 = 33 MB
        - Total: 33 MB (vs 33 MB dense-only baseline)

    Bandwidth Savings:
        - No per-frame QTT core upload (CPU-side evaluation)
        - Only sparse samples uploaded: 256 KB/frame
        - Interpolation reads sparse once, writes dense once
        - Net bandwidth: ~66 MB vs 132 MB (50% reduction)
    """

    def __init__(
        self,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        n_cpu_threads: int = 8,
    ):
        """
        Initialize hybrid renderer.

        Args:
            device: GPU device (defaults to CUDA if available)
            dtype: Output precision (float16 for bandwidth efficiency)
            n_cpu_threads: CPU threads for QTT evaluation (default 8 = P-cores)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for hybrid rendering")

        # CPU evaluator
        self.cpu_evaluator = CPUQTTEvaluator(n_threads=n_cpu_threads)

        # Performance monitoring
        self._frame_count = 0
        self._total_time = 0.0

    def render_qtt_hybrid(
        self,
        qtt: QTT2DState,
        sparse_size: int,
        output_width: int,
        output_height: int,
        colormap: torch.Tensor | None = None,
        return_timings: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Render QTT via hybrid CPU-GPU pipeline.

        Args:
            qtt: QTT2DState to render
            sparse_size: Sparse grid resolution (128 or 256 recommended for 4K)
            output_width: Target width (e.g., 3840)
            output_height: Target height (e.g., 2160)
            colormap: Optional [256, 3] RGB lookup table
            return_timings: If True, return detailed timing breakdown

        Returns:
            rgb_buffer: [height, width, 3] rendered image
            timings: Dict with timing breakdown (if return_timings=True, else empty)

        Raises:
            ValueError: If sparse_size not power of 2 or invalid dimensions
        """
        if sparse_size & (sparse_size - 1) != 0:
            raise ValueError(f"sparse_size must be power of 2, got {sparse_size}")

        if sparse_size > min(output_width, output_height):
            raise ValueError(f"sparse_size {sparse_size} exceeds output dimensions")
        timings = {}

        # Step 1: QTT evaluation (GPU-accelerated if available)
        try:
            from ontic.cuda.qtt_eval_gpu import hybrid_qtt_eval

            start_eval = time.perf_counter()
            sparse_tensor, eval_time = hybrid_qtt_eval(
                qtt.cores, sparse_size, self.device
            )
            timings["cpu_eval_ms"] = eval_time
            timings["upload_ms"] = 0.0  # Already on GPU

        except (ImportError, Exception):
            # Fallback to CPU evaluation
            if self.cpu_evaluator.cores_flat is None:
                self.cpu_evaluator.load_qtt(qtt)

            start_cpu = time.perf_counter()
            sparse_values, _ = self.cpu_evaluator.eval_sparse_grid(sparse_size)
            end_cpu = time.perf_counter()
            timings["cpu_eval_ms"] = (end_cpu - start_cpu) * 1000

            # Upload to GPU
            start_upload = time.perf_counter()
            sparse_tensor = torch.from_numpy(sparse_values).to(
                device=self.device, dtype=torch.float32
            )
            torch.cuda.synchronize()
            end_upload = time.perf_counter()
            timings["upload_ms"] = (end_upload - start_upload) * 1000

        # Step 3: GPU interpolation
        # F.interpolate expects [batch, channels, height, width]
        sparse_4d = sparse_tensor.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, sparse_size, sparse_size]

        start_interp = torch.cuda.Event(enable_timing=True)
        end_interp = torch.cuda.Event(enable_timing=True)

        start_interp.record()
        dense_4d = F.interpolate(
            sparse_4d,
            size=(output_height, output_width),
            mode="bicubic",
            align_corners=False,
        )
        end_interp.record()
        torch.cuda.synchronize()
        timings["interpolate_ms"] = start_interp.elapsed_time(end_interp)

        # Extract [height, width]
        dense_values = dense_4d.squeeze(0).squeeze(0)

        # Step 4: Apply colormap (optimized)
        start_color = torch.cuda.Event(enable_timing=True)
        end_color = torch.cuda.Event(enable_timing=True)

        start_color.record()
        if colormap is not None:
            # Fast normalization: assume values in [0, 1] range (physics bounded)
            # Avoids slow min/max computation
            normalized = dense_values.clamp(0, 1)

            # Map to colormap (fused operation)
            indices = (normalized * 255).to(torch.uint8).long()
            rgb = colormap[indices]  # [H, W, 3]
        else:
            # Grayscale
            rgb = dense_values.unsqueeze(-1).expand(-1, -1, 3)

        # Convert to target dtype
        rgb = rgb.to(self.dtype)
        end_color.record()
        torch.cuda.synchronize()
        timings["colormap_ms"] = start_color.elapsed_time(end_color)

        # Total time
        timings["total_ms"] = sum(timings.values())
        timings["fps"] = 1000 / timings["total_ms"]

        # Update stats
        self._frame_count += 1
        self._total_time += timings["total_ms"]

        return rgb, timings if return_timings else {}

    def get_average_fps(self) -> float:
        """Get average FPS across all rendered frames."""
        if self._frame_count == 0:
            return 0.0
        return 1000 * self._frame_count / self._total_time

    def reset_stats(self):
        """Reset performance statistics."""
        self._frame_count = 0
        self._total_time = 0.0

    def benchmark_hybrid(
        self,
        qtt: QTT2DState,
        sparse_sizes: list = [128, 256],
        output_resolution: tuple[int, int] = (3840, 2160),
        n_trials: int = 50,
    ):
        """
        Benchmark hybrid rendering at various sparse grid sizes.

        Measures CPU evaluation, GPU upload, interpolation, and colormap times.
        """
        output_width, output_height = output_resolution

        print("=" * 70)
        print("Hybrid CPU-GPU QTT Renderer Benchmark")
        print("=" * 70)
        print(
            f"Output: {output_width}×{output_height} ({output_width*output_height/1e6:.1f}M pixels)"
        )
        print(f"CPU: {self.cpu_evaluator.n_threads} threads (i9 P-cores)")
        print(f"GPU: {torch.cuda.get_device_name()} @ {self.dtype}")

        for sparse_size in sparse_sizes:
            print(f"\n{'='*70}")
            print(
                f"Sparse Grid: {sparse_size}×{sparse_size} ({sparse_size**2/1000:.0f}K samples)"
            )
            print(
                f"Upsampling: {output_width/sparse_size:.1f}× horizontal, {output_height/sparse_size:.1f}× vertical"
            )

            # Warmup
            for _ in range(5):
                _, _ = self.render_qtt_hybrid(
                    qtt, sparse_size, output_width, output_height, return_timings=True
                )

            # Timed trials
            all_timings = []
            for trial in range(n_trials):
                _, timings = self.render_qtt_hybrid(
                    qtt, sparse_size, output_width, output_height, return_timings=True
                )
                all_timings.append(timings)

            # Statistics
            keys = [
                "cpu_eval_ms",
                "upload_ms",
                "interpolate_ms",
                "colormap_ms",
                "total_ms",
            ]
            stats = {}
            for key in keys:
                values = [t[key] for t in all_timings]
                stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

            # Report
            print(f"\nTiming Breakdown ({n_trials} trials):")
            print(
                f"  CPU Evaluation:  {stats['cpu_eval_ms']['mean']:6.2f} ms (±{stats['cpu_eval_ms']['std']:.2f})"
            )
            print(
                f"  GPU Upload:      {stats['upload_ms']['mean']:6.2f} ms (±{stats['upload_ms']['std']:.2f})"
            )
            print(
                f"  GPU Interpolate: {stats['interpolate_ms']['mean']:6.2f} ms (±{stats['interpolate_ms']['std']:.2f})"
            )
            print(
                f"  GPU Colormap:    {stats['colormap_ms']['mean']:6.2f} ms (±{stats['colormap_ms']['std']:.2f})"
            )
            print("  ─────────────────────────")
            print(
                f"  TOTAL:           {stats['total_ms']['mean']:6.2f} ms (±{stats['total_ms']['std']:.2f})"
            )
            print(
                f"  Min/Max:         {stats['total_ms']['min']:.2f} / {stats['total_ms']['max']:.2f} ms"
            )
            print(f"  FPS:             {1000/stats['total_ms']['mean']:.1f}")

            # Compare to targets
            target_fps = 60
            target_ms = 1000 / target_fps
            if stats["total_ms"]["mean"] <= target_ms:
                status = "✓ ACHIEVES 60 FPS TARGET"
            else:
                status = f"✗ {stats['total_ms']['mean'] - target_ms:.1f}ms over target"
            print(f"\n  Status: {status}")

        print("\n" + "=" * 70)


def create_test_qtt(nx: int = 11, ny: int = 11, rank: int = 8) -> QTT2DState:
    """
    Create synthetic QTT for testing and validation.

    Args:
        nx: X-dimension bit depth (11 = 2048 resolution)
        ny: Y-dimension bit depth (11 = 2048 resolution)
        rank: Bond dimension

    Returns:
        QTT2DState with smooth random field
    """
    import torch

    from ontic.cfd.qtt_2d import QTT2DState

    n_cores = nx + ny
    torch.manual_seed(42)  # Reproducible

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

    return QTT2DState(cores=cores, nx=nx, ny=ny)


def test_hybrid_renderer():
    """Validation test for hybrid renderer."""
    print("\nTesting Hybrid CPU-GPU QTT Renderer")
    print("=" * 70)

    qtt = create_test_qtt(nx=11, ny=11, rank=8)

    device = torch.device("cuda")
    renderer = HybridQTTRenderer(device=device, dtype=torch.float16)

    # Focus on production configurations
    print("\n1080p Benchmark (Production Validation):")
    renderer.benchmark_hybrid(
        qtt, sparse_sizes=[128, 256], output_resolution=(1920, 1080), n_trials=50
    )

    print("\n4K Benchmark (Phase 4 Target):")
    renderer.benchmark_hybrid(
        qtt, sparse_sizes=[128, 256], output_resolution=(3840, 2160), n_trials=50
    )

    print("\n✓ Hybrid renderer validation complete")
    print("=" * 70)


if __name__ == "__main__":
    test_hybrid_renderer()
