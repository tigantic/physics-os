"""
Real-Time Tensor Streaming for Glass Cockpit Integration
Milestone 3.3: Real-Time Data Pipeline

This module continuously generates or decompresses tensor fields and streams them
to the Glass Cockpit via RAM Bridge Protocol v2. Supports both synthetic test
patterns and live Navier-Stokes simulation data.

Constitutional compliance:
- Article II: Type-safe with full type hints
- Article V: GPU-accelerated with PyTorch CUDA
- Article VIII: <5% CPU (GPU does heavy lifting)
"""

import time

import numpy as np
import torch

from .bridge_writer import TensorBridgeWriter


class RealtimeTensorStream:
    """
    Real-time tensor field streaming to Glass Cockpit.

    Supports both synthetic patterns (for testing) and live simulation data.
    Automatically applies colormap transformation and writes to RAM bridge.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        bridge_path: str = "/dev/shm/hypertensor_bridge",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize real-time tensor streamer.

        Args:
            width: Output width in pixels (default 1920)
            height: Output height in pixels (default 1080)
            bridge_path: Path to RAM bridge shared memory file
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.width = width
        self.height = height
        self.device = device
        self.bridge_writer = TensorBridgeWriter(bridge_path)

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_report = time.time()

        print("RealtimeTensorStream initialized:")
        print(f"  Resolution: {width}x{height}")
        print(f"  Device: {device}")
        print(f"  Bridge path: {bridge_path}")

    def __enter__(self):
        """Context manager entry."""
        self.bridge_writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.bridge_writer.__exit__(exc_type, exc_val, exc_tb)

    def generate_synthetic_pattern(
        self, frame_number: int, pattern: str = "waves"
    ) -> torch.Tensor:
        """
        Generate synthetic tensor pattern for testing.

        Args:
            frame_number: Current frame number for animation
            pattern: Pattern type ('waves', 'vortex', 'turbulence')

        Returns:
            Float32 tensor of shape (height, width) on configured device
        """
        # Create coordinate grids
        y = torch.linspace(-1, 1, self.height, device=self.device)
        x = torch.linspace(-1, 1, self.width, device=self.device)
        Y, X = torch.meshgrid(y, x, indexing="ij")

        # Time variable for animation
        t = frame_number * 0.05

        if pattern == "waves":
            # Interfering wave pattern
            wave1 = torch.sin(10 * X + t)
            wave2 = torch.sin(10 * Y + t * 1.2)
            field = wave1 + wave2

        elif pattern == "vortex":
            # Rotating vortex (simulates vorticity field)
            r = torch.sqrt(X**2 + Y**2)
            theta = torch.atan2(Y, X)
            field = torch.sin(5 * r - t) * torch.cos(3 * theta + t)

        elif pattern == "turbulence":
            # Multi-scale turbulence (Perlin-like noise approximation)
            field = torch.zeros_like(X)
            for scale in [1, 2, 4, 8]:
                freq = scale * 5
                amplitude = 1.0 / scale
                field += amplitude * (
                    torch.sin(freq * X + t * scale)
                    * torch.cos(freq * Y + t * scale * 1.3)
                )
        else:
            # Fallback: simple gradient
            field = X + Y + torch.sin(t)

        return field

    def tensor_to_rgba8(
        self, tensor: torch.Tensor, colormap: str = "viridis"
    ) -> tuple[np.ndarray, dict]:
        """
        Convert tensor field to RGBA8 format with colormap.

        NOTE: In Phase 3, colormap is applied GPU-side in Glass Cockpit.
        This method creates a grayscale representation for the bridge,
        and Glass Cockpit applies the colormap using the WGSL shader.

        Args:
            tensor: Float32 tensor of shape (height, width)
            colormap: Colormap name (for future use, currently unused)

        Returns:
            Tuple of (rgba8_array, statistics_dict)
        """
        # Compute statistics on GPU
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        tensor_mean = tensor.mean().item()
        tensor_std = tensor.std().item()

        stats = {
            "min": tensor_min,
            "max": tensor_max,
            "mean": tensor_mean,
            "std": tensor_std,
        }

        # Normalize to [0, 1] for transmission
        if tensor_max > tensor_min:
            normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
        else:
            normalized = torch.ones_like(tensor) * 0.5

        # Convert to RGBA8 (grayscale in RGB, alpha=255)
        # Glass Cockpit will apply colormap GPU-side
        rgb_u8 = (normalized * 255).clamp(0, 255).byte()

        # Expand to RGBA (R=G=B=grayscale, A=255)
        rgba8 = torch.zeros(
            (self.height, self.width, 4), dtype=torch.uint8, device=self.device
        )
        rgba8[:, :, 0] = rgb_u8  # Red
        rgba8[:, :, 1] = rgb_u8  # Green
        rgba8[:, :, 2] = rgb_u8  # Blue
        rgba8[:, :, 3] = 255  # Alpha (opaque)

        # D-004 FIX: Use contiguous storage instead of .numpy()
        rgba8_cpu = rgba8.cpu()
        if not rgba8_cpu.is_contiguous():
            rgba8_cpu = rgba8_cpu.contiguous()

        # Return tensor directly; caller uses .untyped_storage() for bytes
        return rgba8_cpu, stats

    def stream_synthetic(
        self,
        duration_seconds: float = 60.0,
        target_fps: float = 60.0,
        pattern: str = "waves",
        verbose: bool = True,
    ):
        """
        Stream synthetic tensor patterns for specified duration.

        Args:
            duration_seconds: How long to stream (seconds)
            target_fps: Target frame rate (FPS)
            pattern: Pattern type ('waves', 'vortex', 'turbulence')
            verbose: Print performance metrics every second
        """
        frame_time = 1.0 / target_fps
        end_time = time.time() + duration_seconds

        print("\n=== Starting synthetic stream ===")
        print(f"Pattern: {pattern}")
        print(f"Duration: {duration_seconds}s @ {target_fps} FPS")
        print(f"Target frame time: {frame_time*1000:.2f}ms")

        with self:
            while time.time() < end_time:
                frame_start = time.time()

                # Generate synthetic pattern
                tensor_field = self.generate_synthetic_pattern(
                    self.frame_count, pattern=pattern
                )

                # Convert to RGBA8
                rgba8_data, stats = self.tensor_to_rgba8(tensor_field)

                # Write to RAM bridge
                self.bridge_writer.write_frame(
                    rgba8_data,
                    tensor_min=stats["min"],
                    tensor_max=stats["max"],
                    tensor_mean=stats["mean"],
                    tensor_std=stats["std"],
                )

                self.frame_count += 1

                # Performance reporting
                if verbose and time.time() - self.last_fps_report >= 1.0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    print(
                        f"Frame {self.frame_count:6d} | "
                        f"FPS: {fps:6.2f} | "
                        f"Range: [{stats['min']:+.3f}, {stats['max']:+.3f}] | "
                        f"Mean: {stats['mean']:+.3f}"
                    )
                    self.last_fps_report = time.time()

                # Sleep to maintain target FPS
                frame_elapsed = time.time() - frame_start
                sleep_time = max(0, frame_time - frame_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # Final report
        total_elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / total_elapsed
        print("\n=== Stream complete ===")
        print(f"Total frames: {self.frame_count}")
        print(f"Total time: {total_elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")

    def stream_from_qtt(
        self,
        qtt_tensor,  # QTT-compressed tensor (future integration)
        duration_seconds: float = 60.0,
        target_fps: float = 60.0,
    ):
        """
        Stream live QTT-decompressed tensor fields.

        NOTE: This is a placeholder for future QTT integration.
        Current implementation uses synthetic patterns.

        Args:
            qtt_tensor: QTT-compressed tensor object
            duration_seconds: How long to stream (seconds)
            target_fps: Target frame rate (FPS)
        """
        raise NotImplementedError(
            "QTT streaming not yet implemented. " "Use stream_synthetic() for testing."
        )


def test_realtime_stream(
    duration: float = 10.0,
    pattern: str = "turbulence",
    fps: float = 60.0,
):
    """
    Test real-time tensor streaming with synthetic patterns.

    Args:
        duration: Test duration in seconds
        pattern: Pattern type ('waves', 'vortex', 'turbulence')
        fps: Target frames per second
    """
    print("=== Real-Time Tensor Stream Test ===")
    print(f"Testing {pattern} pattern @ {fps} FPS for {duration}s")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, using CPU (will be slow)")

    # Create streamer and run
    streamer = RealtimeTensorStream(
        width=1920,
        height=1080,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    streamer.stream_synthetic(
        duration_seconds=duration,
        target_fps=fps,
        pattern=pattern,
        verbose=True,
    )


if __name__ == "__main__":
    # Run 10-second test with turbulence pattern
    test_realtime_stream(
        duration=10.0,
        pattern="turbulence",
        fps=60.0,
    )
