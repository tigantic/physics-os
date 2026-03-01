"""CUDA Heatmap Generator v2: Grayscale Intensity Streaming

Sends raw uint8 intensity values to RAM bridge (1 byte/pixel).
Rust applies colormap via 1D texture lookup in fragment shader.

Benefits:
- 4x bandwidth reduction (32KB vs 128KB for 256×128)
- Dynamic colormap switching in UI
- Simpler Python code (no colormap logic)

Constitutional Compliance:
- Article V: GPU-accelerated (CUDA)
- Article II: Type hints, docstrings
- Doctrine 2: RAM Bridge Protocol v2
"""

import math
import mmap
import os
import struct
import time as time_module
from pathlib import Path

import torch

from .bridge_writer import DEFAULT_BRIDGE_PATH, HEADER_SIZE


class GrayscaleBridgeWriter:
    """Writes grayscale uint8 intensity to shared memory.

    Protocol v2: Single-channel intensity instead of RGBA8.
    Rust applies colormap via 1D texture lookup.

    Memory Layout:
        - Header (4096 bytes): Metadata
        - Data: width × height bytes (grayscale intensity)
    """

    MAGIC = b"TNSG"  # Ontic Engine Sovereign Grayscale
    VERSION = 2

    def __init__(
        self,
        path: Path = DEFAULT_BRIDGE_PATH,
        width: int = 256,
        height: int = 128,
        create: bool = True,
    ):
        self.path = path
        self.width = width
        self.height = height
        self.channels = 1  # Grayscale

        self.frame_number = 0
        self.data_size = width * height  # 1 byte per pixel
        self.total_size = HEADER_SIZE + self.data_size

        # Create or open shared memory file
        if create and not path.exists():
            self._create_bridge_file()

        # Memory-map the file
        self.fd = os.open(str(path), os.O_RDWR)
        self.mmap = mmap.mmap(self.fd, self.total_size)

        # Initialize header
        self._write_header(0, 0.0, 1.0, 0.5, 0.25)

        print(f"✓ Grayscale bridge: {self.path} ({self.data_size / 1024:.1f} KB)")

    def _create_bridge_file(self) -> None:
        """Create shared memory file."""
        with open(self.path, "wb") as f:
            f.write(b"\x00" * self.total_size)

    def _write_header(
        self,
        frame_number: int,
        tensor_min: float,
        tensor_max: float,
        tensor_mean: float,
        tensor_std: float,
    ) -> None:
        """Write header with grayscale protocol marker."""
        timestamp_us = int(time_module.time() * 1_000_000)

        header = struct.pack(
            "<4sI Q 5I 4f 2Q",
            self.MAGIC,  # "TNSG" for grayscale
            self.VERSION,
            frame_number,
            self.width,
            self.height,
            self.channels,  # 1 for grayscale
            HEADER_SIZE,
            self.data_size,
            tensor_min,
            tensor_max,
            tensor_mean,
            tensor_std,
            timestamp_us,
            0,
        )

        header += b"\x00" * (HEADER_SIZE - len(header))

        self.mmap.seek(0)
        self.mmap.write(header)

    def write_frame(
        self,
        intensity: torch.Tensor,
        tensor_min: float | None = None,
        tensor_max: float | None = None,
        tensor_mean: float | None = None,
        tensor_std: float | None = None,
    ) -> None:
        """Write grayscale intensity frame.

        Args:
            intensity: Tensor of shape (H, W) with dtype uint8 [0-255]
        """
        if intensity.shape != (self.height, self.width):
            raise ValueError(
                f"Expected {(self.height, self.width)}, got {intensity.shape}"
            )

        if intensity.dtype != torch.uint8:
            raise ValueError(f"Expected uint8, got {intensity.dtype}")

        # Transfer to CPU
        if intensity.is_cuda:
            intensity_cpu = intensity.cpu()
        else:
            intensity_cpu = intensity

        if not intensity_cpu.is_contiguous():
            intensity_cpu = intensity_cpu.contiguous()

        raw_bytes = intensity_cpu.numpy().tobytes()

        # Write header
        self.frame_number += 1
        self._write_header(
            self.frame_number,
            tensor_min or 0.0,
            tensor_max or 1.0,
            tensor_mean or 0.5,
            tensor_std or 0.25,
        )

        # Write intensity data
        self.mmap.seek(HEADER_SIZE)
        self.mmap.write(raw_bytes)

    def close(self) -> None:
        """Clean up resources."""
        self.mmap.close()
        os.close(self.fd)


class CUDAHeatmapGeneratorV2:
    """GPU-accelerated grayscale intensity generator.

    Streams raw intensity values to RAM bridge.
    Rust applies colormap via 1D texture lookup.

    Args:
        resolution: Grid resolution (width, height)
        device: CUDA device
        bridge_path: Path to RAM bridge
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (256, 128),
        device: str = "cuda",
        bridge_path: str | None = None,
    ):
        self.width, self.height = resolution
        self.device = torch.device(device)

        # Coordinate grids (cached on GPU)
        lon = torch.linspace(-math.pi, math.pi, self.width, device=self.device)
        lat = torch.linspace(-math.pi / 2, math.pi / 2, self.height, device=self.device)
        self.lon_grid, self.lat_grid = torch.meshgrid(lon, lat, indexing="xy")

        # Output tensor (pre-allocated)
        self.intensity = torch.zeros(self.height, self.width, device=self.device)

        # Pre-allocated pinned memory for fast transfer
        self.pinned_buffer = torch.zeros(
            (self.height, self.width),
            dtype=torch.uint8,
            pin_memory=True,
        )

        # Cross-platform bridge path
        import platform

        if bridge_path:
            bridge = Path(bridge_path)
        elif platform.system() == "Windows":
            import tempfile

            bridge = Path(tempfile.gettempdir()) / "ontic_bridge"
        else:
            bridge = Path("/dev/shm/ontic_bridge")

        # Grayscale bridge writer
        self.writer = GrayscaleBridgeWriter(
            path=bridge,
            width=self.width,
            height=self.height,
            create=True,
        )

        # Performance tracking
        self.frame_count = 0
        self.total_compute_us = 0
        self.total_write_us = 0

        print("✓ CUDA Heatmap Generator v2 (grayscale)")
        print(f"  Resolution: {self.width}×{self.height}")
        print(f"  Bandwidth: {self.width * self.height / 1024:.1f} KB/frame")
        print(f"  Device: {self.device}")

    @torch.no_grad()
    def generate_convergence(self, time: float) -> None:
        """Generate convergence field on GPU."""
        self.intensity.zero_()

        # Storm 1: North Atlantic hurricane
        storm1_lon = -0.8 + time * 0.05
        storm1_lat = 0.4
        dist1 = torch.sqrt(
            (self.lon_grid - storm1_lon) ** 2 + (self.lat_grid - storm1_lat) ** 2
        )
        self.intensity += torch.exp(-dist1 * 3.0) * 0.9

        # Storm 2: Pacific typhoon
        storm2_lon = 2.5 + time * 0.03
        storm2_lat = 0.3
        dist2 = torch.sqrt(
            (self.lon_grid - storm2_lon) ** 2 + (self.lat_grid - storm2_lat) ** 2
        )
        self.intensity += torch.exp(-dist2 * 4.0) * 0.85

        # Storm 3: Southern hemisphere low
        storm3_lon = 0.5
        storm3_lat = -0.5 + time * 0.02
        dist3 = torch.sqrt(
            (self.lon_grid - storm3_lon) ** 2 + (self.lat_grid - storm3_lat) ** 2
        )
        self.intensity += torch.exp(-dist3 * 3.5) * 0.7

        # Storm 4: Jet stream
        jet_lat = 0.7 + 0.2 * torch.sin(self.lon_grid * 2.0 + time * 0.1)
        jet_dist = torch.abs(self.lat_grid - jet_lat)
        self.intensity += (
            torch.exp(-jet_dist * 5.0)
            * 0.4
            * (0.5 + 0.5 * torch.sin(self.lon_grid * 3.0))
        )

        # Clamp and convert to uint8
        self.intensity.clamp_(0, 1)

    @torch.no_grad()
    def quantize_intensity(self) -> torch.Tensor:
        """Convert float intensity to uint8."""
        intensity_u8 = (self.intensity * 255).to(torch.uint8)
        self.pinned_buffer.copy_(intensity_u8, non_blocking=True)
        return self.pinned_buffer

    def update(self, time: float) -> dict:
        """Generate intensity and write to bridge."""
        t0 = time_module.perf_counter()

        # Generate on GPU
        self.generate_convergence(time)

        # Quantize to uint8
        intensity_u8 = self.quantize_intensity()

        # Sync before CPU access
        torch.cuda.synchronize()
        t1 = time_module.perf_counter()
        compute_us = int((t1 - t0) * 1_000_000)

        # Write to bridge
        self.writer.write_frame(
            intensity_u8,
            tensor_min=float(self.intensity.min()),
            tensor_max=float(self.intensity.max()),
            tensor_mean=float(self.intensity.mean()),
            tensor_std=float(self.intensity.std()),
        )

        t2 = time_module.perf_counter()
        write_us = int((t2 - t1) * 1_000_000)

        self.frame_count += 1
        self.total_compute_us += compute_us
        self.total_write_us += write_us

        return {
            "compute_us": compute_us,
            "write_us": write_us,
            "total_us": compute_us + write_us,
            "frame": self.frame_count,
        }

    def run_loop(
        self, target_fps: float = 165.0, duration: float | None = None
    ) -> None:
        """Run continuous generation loop."""
        frame_budget = 1.0 / target_fps
        start_time = time_module.perf_counter()
        report_interval = 2.0
        last_report = start_time

        print(f"\n🔥 Grayscale heatmap generator @ {target_fps} Hz")
        print(
            f"   Bandwidth: {self.width * self.height * target_fps / 1024 / 1024:.1f} MB/s"
        )
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
                    avg_compute = self.total_compute_us / self.frame_count
                    avg_write = self.total_write_us / self.frame_count

                    print(
                        f"FPS: {fps:.0f} | "
                        f"Compute: {avg_compute:.0f}µs | "
                        f"Write: {avg_write:.0f}µs | "
                        f"Total: {avg_compute + avg_write:.0f}µs"
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


def main() -> None:
    """Run standalone grayscale generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CUDA Heatmap Generator v2 (Grayscale)"
    )
    parser.add_argument("--fps", type=float, default=165.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--resolution", type=str, default="256x128")
    args = parser.parse_args()

    w, h = map(int, args.resolution.split("x"))
    gen = CUDAHeatmapGeneratorV2(resolution=(w, h))
    gen.run_loop(target_fps=args.fps, duration=args.duration)


if __name__ == "__main__":
    main()
