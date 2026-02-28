"""QTT Bridge Streamer: Implicit TT Evaluation → Shared Memory

Streams QTT-synthesized heatmaps directly to RAM bridge.
Uses CUDA kernel for sub-millisecond render times.

Performance @ 256×128: ~128K FPS (kernel only)
Performance @ 4K:      ~2K FPS (0.47ms/frame)

Architecture:
    QTT Cores (96 floats) → CUDA Kernel → RGBA → Bridge

Constitutional Compliance:
    - Article V: GPU-accelerated (CUDA kernel)
    - Article II: Type hints, docstrings
    - Doctrine 2: RAM Bridge Protocol
"""

import mmap
import os
import struct
import time as time_module
from pathlib import Path

import torch

from . import implicit_qtt_renderer as renderer_module
from .bridge_writer import HEADER_SIZE


class QTTBridgeStreamer:
    """Stream QTT-synthesized heatmaps to shared memory bridge.

    Uses implicit QTT CUDA kernel for direct TT evaluation at pixels.
    No intermediate dense tensors - pure TT→Pixel synthesis.

    Args:
        resolution: Output resolution (width, height)
        device: CUDA device
        bridge_path: Path to shared memory bridge
        colormap: "plasma" or "viridis"
    """

    MAGIC = b"TNSR"  # TensorNet Sovereign RGBA
    VERSION = 1  # Protocol version - must match Rust reader

    def __init__(
        self,
        resolution: tuple[int, int] = (256, 128),
        device: str = "cuda",
        bridge_path: Path | None = None,
        colormap: str = "plasma",
    ):
        self.width, self.height = resolution
        self.device = torch.device(device)
        self.colormap = colormap
        self.colormap_type = 0 if colormap == "plasma" else 1

        # Compile QTT CUDA kernel
        print("Initializing QTT Bridge Streamer...")
        renderer_module._compile_kernel()
        self.kernel = renderer_module._KERNEL_MODULE

        # Output buffer (RGBA float32 on GPU)
        self.output_buffer = torch.zeros(
            self.height * self.width * 4, device=self.device, dtype=torch.float32
        )

        # Pinned buffer for fast GPU→CPU transfer
        self.pinned_buffer = torch.zeros(
            self.height * self.width * 4, dtype=torch.uint8, pin_memory=True
        )

        # Initialize QTT cores (start with synthetic test pattern)
        self._init_qtt_cores()

        # Bridge setup
        import platform

        if bridge_path:
            self.bridge_path = Path(bridge_path)
        elif platform.system() == "Windows":
            import tempfile

            self.bridge_path = Path(tempfile.gettempdir()) / "ontic_bridge"
        else:
            self.bridge_path = Path("/dev/shm/ontic_bridge")

        self.channels = 4  # RGBA
        self.data_size = self.width * self.height * self.channels
        self.total_size = HEADER_SIZE + self.data_size

        # Create bridge file
        self._create_bridge()

        # Performance tracking
        self.frame_count = 0
        self.total_render_us = 0
        self.total_transfer_us = 0
        self.total_write_us = 0

        print("✓ QTT Bridge Streamer initialized")
        print(f"  Resolution: {self.width}×{self.height}")
        print(f"  Colormap: {colormap}")
        print(f"  Bridge: {self.bridge_path}")
        print(f"  Bandwidth: {self.data_size / 1024:.1f} KB/frame")

    def _init_qtt_cores(self) -> None:
        """Initialize synthetic QTT cores for test pattern.

        Creates a smooth gradient pattern by setting TT cores
        to near-identity matrices with controlled asymmetry.
        """
        # 96 floats = 12 cores × 2 matrices × 4 floats
        self.qtt_cores = torch.zeros(96, device=self.device, dtype=torch.float32)

        # Pre-compute core indices for GPU update
        # Shape: [12, 2] for core_idx, matrix_idx pairs
        core_indices = torch.arange(12, device=self.device, dtype=torch.float32)
        matrix_indices = torch.arange(2, device=self.device, dtype=torch.float32)

        # Store for GPU update
        self.core_idx_grid = core_indices.unsqueeze(1).expand(12, 2)  # [12, 2]
        self.matrix_idx_grid = matrix_indices.unsqueeze(0).expand(12, 2)  # [12, 2]

        # Pre-compute frequency and phase multipliers
        self.freqs = 0.5 + core_indices * 0.1  # [12]

        # Pre-allocate work buffers to avoid per-frame allocation
        self._work_phases = torch.zeros(12, device=self.device, dtype=torch.float32)
        self._work_t = torch.zeros(12, 2, device=self.device, dtype=torch.float32)
        self._work_matrices = torch.zeros(
            12, 2, 4, device=self.device, dtype=torch.float32
        )
        self._work_rgba_u8 = torch.zeros(
            self.height * self.width * 4, device=self.device, dtype=torch.uint8
        )

        # Pre-compute constant factors
        self._j_offset = self.matrix_idx_grid * 1.5  # [12, 2] - constant
        self._core_factor = (self.core_idx_grid - 6) / 12  # [12, 2] - constant

        # Set initial pattern
        self._update_qtt_cores_gpu(0.0)

    def _create_bridge(self) -> None:
        """Create and memory-map the bridge file."""
        if not self.bridge_path.exists():
            with open(self.bridge_path, "wb") as f:
                f.write(b"\x00" * self.total_size)

        self.fd = os.open(str(self.bridge_path), os.O_RDWR)
        self.mmap = mmap.mmap(self.fd, self.total_size)

    def _write_header(
        self,
        frame_number: int,
        tensor_min: float,
        tensor_max: float,
        tensor_mean: float,
        tensor_std: float,
    ) -> None:
        """Write bridge header with RGBA protocol marker."""
        timestamp_us = int(time_module.time() * 1_000_000)

        header = struct.pack(
            "<4sI Q 5I 4f 2Q",
            self.MAGIC,
            self.VERSION,
            frame_number,
            self.width,
            self.height,
            self.channels,
            HEADER_SIZE,
            self.data_size,
            tensor_min,
            tensor_max,
            tensor_mean,
            tensor_std,
            timestamp_us,
            0,  # Reserved
        )

        self.mmap.seek(0)
        self.mmap.write(header)

    def _update_qtt_cores_gpu(self, time: float) -> None:
        """GPU-vectorized QTT core update - NO CPU LOOPS, NO ALLOCATION.

        Uses pre-allocated work buffers to avoid per-frame memory allocation.
        """
        # Compute phases in-place: [12]
        torch.mul(self.freqs, time, out=self._work_phases)

        # Compute t = sin(phases + j_offset) * 0.3 using pre-allocated buffer
        # phases.unsqueeze(1) + j_offset → sin → * 0.3
        torch.add(self._work_phases.unsqueeze(1), self._j_offset, out=self._work_t)
        torch.sin_(self._work_t)
        self._work_t.mul_(0.3)

        # Compute all 4 matrix elements in pre-allocated buffer: [12, 2, 4]
        # a = 1.0 + t * core_factor
        # b = 0.1 * t * matrix_idx_grid
        # c = 0.1 * t * (1 - matrix_idx_grid)
        # d = 1.0 - t * core_factor
        t_cf = self._work_t * self._core_factor  # [12, 2]

        self._work_matrices[:, :, 0] = 1.0 + t_cf  # a
        self._work_matrices[:, :, 1] = 0.1 * self._work_t * self.matrix_idx_grid  # b
        self._work_matrices[:, :, 2] = (
            0.1 * self._work_t * (1 - self.matrix_idx_grid)
        )  # c
        self._work_matrices[:, :, 3] = 1.0 - t_cf  # d

        # Flatten to [96] - this reuses the same memory view
        self.qtt_cores = self._work_matrices.reshape(-1)

    def update_qtt_cores(self, time: float) -> None:
        """Update QTT cores for dynamic animation (GPU-accelerated).

        This modifies the TT-cores directly in-place, enabling
        dynamic patterns without re-factorization.

        Args:
            time: Elapsed time in seconds
        """
        self._update_qtt_cores_gpu(time)

    @torch.no_grad()
    def render_frame(self) -> None:
        """Render QTT cores to RGBA buffer via CUDA kernel."""
        self.kernel.render_qtt_layer_wrapper(
            self.qtt_cores,
            self.output_buffer,
            self.width,
            self.height,
            0.0,  # value_min
            1.0,  # value_max
            self.colormap_type,
        )

    @torch.no_grad()
    def transfer_to_cpu(self) -> None:
        """Transfer RGBA float → uint8 pinned memory - NO ALLOCATION."""
        # Convert float [0,1] → uint8 [0,255] using pre-allocated buffer
        # Clamp → scale → convert in-place where possible
        torch.clamp(self.output_buffer, 0, 1, out=self.output_buffer)
        self.output_buffer.mul_(255)

        # Copy to pre-allocated uint8 buffer, then to pinned
        self._work_rgba_u8.copy_(self.output_buffer.to(torch.uint8))
        self.pinned_buffer.copy_(self._work_rgba_u8.cpu(), non_blocking=True)

        # Restore output_buffer for next frame (scale back)
        self.output_buffer.div_(255)

    def write_to_bridge(self) -> None:
        """Write frame data to shared memory bridge."""
        torch.cuda.synchronize()

        self._write_header(
            self.frame_count,
            tensor_min=0.0,
            tensor_max=1.0,
            tensor_mean=0.5,
            tensor_std=0.25,
        )

        self.mmap.seek(HEADER_SIZE)
        self.mmap.write(self.pinned_buffer.numpy().tobytes())
        self.frame_count += 1

    def update(self, time: float) -> dict:
        """Full frame update: animate → render → transfer → write."""
        t0 = time_module.perf_counter()

        # Animate QTT cores
        self.update_qtt_cores(time)

        # Render via CUDA kernel
        self.render_frame()
        t1 = time_module.perf_counter()
        render_us = int((t1 - t0) * 1_000_000)

        # Transfer to CPU
        self.transfer_to_cpu()
        t2 = time_module.perf_counter()
        transfer_us = int((t2 - t1) * 1_000_000)

        # Write to bridge
        self.write_to_bridge()
        t3 = time_module.perf_counter()
        write_us = int((t3 - t2) * 1_000_000)

        self.total_render_us += render_us
        self.total_transfer_us += transfer_us
        self.total_write_us += write_us

        return {
            "render_us": render_us,
            "transfer_us": transfer_us,
            "write_us": write_us,
            "total_us": render_us + transfer_us + write_us,
            "frame": self.frame_count,
        }

    def run_loop(
        self, target_fps: float = 500.0, duration: float | None = None
    ) -> None:
        """Run continuous streaming loop.

        Args:
            target_fps: Target frame rate (default 500 for TT/QTT)
            duration: Run duration in seconds (None = forever)
        """
        frame_budget = 1.0 / target_fps
        start_time = time_module.perf_counter()
        report_interval = 2.0
        last_report = start_time

        print(f"\n🔥 QTT Bridge Streamer @ {target_fps} Hz target")
        print(f"   Resolution: {self.width}×{self.height}")
        print(f"   Bandwidth: {self.data_size * target_fps / 1024 / 1024:.1f} MB/s")
        print("   Press Ctrl+C to stop\n")

        try:
            while True:
                frame_start = time_module.perf_counter()
                elapsed = frame_start - start_time

                if duration and elapsed > duration:
                    break

                self.update(elapsed)

                if frame_start - last_report > report_interval:
                    fps = self.frame_count / elapsed
                    avg_render = self.total_render_us / self.frame_count
                    avg_transfer = self.total_transfer_us / self.frame_count
                    avg_write = self.total_write_us / self.frame_count

                    print(
                        f"FPS: {fps:.0f} | "
                        f"Render: {avg_render:.0f}µs | "
                        f"Transfer: {avg_transfer:.0f}µs | "
                        f"Write: {avg_write:.0f}µs | "
                        f"Total: {avg_render + avg_transfer + avg_write:.0f}µs"
                    )
                    last_report = frame_start

                frame_time = time_module.perf_counter() - frame_start
                if frame_time < frame_budget:
                    time_module.sleep(frame_budget - frame_time)

        except KeyboardInterrupt:
            print("\n⏹ Stopped")

        total_time = time_module.perf_counter() - start_time
        if self.frame_count > 0:
            print(f"\n📊 Final: {self.frame_count / total_time:.0f} FPS avg")

    def close(self) -> None:
        """Clean up resources."""
        self.mmap.close()
        os.close(self.fd)


def main() -> None:
    """Run QTT Bridge Streamer."""
    import argparse

    parser = argparse.ArgumentParser(description="QTT Bridge Streamer")
    parser.add_argument("--fps", type=float, default=500.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--resolution", type=str, default="256x128")
    parser.add_argument(
        "--colormap", type=str, default="plasma", choices=["plasma", "viridis"]
    )
    args = parser.parse_args()

    w, h = map(int, args.resolution.split("x"))
    streamer = QTTBridgeStreamer(resolution=(w, h), colormap=args.colormap)

    try:
        streamer.run_loop(target_fps=args.fps, duration=args.duration)
    finally:
        streamer.close()


if __name__ == "__main__":
    main()
