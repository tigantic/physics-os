#!/usr/bin/env python3
"""
Weather Stream Writer - Global Eye Phase 1B-6
==============================================

Writes weather data to shared memory for Rust consumption.

Exit Gate: Python writes data and sets is_ready=1

Usage:
    from tensornet.sovereign.weather_stream import WeatherStreamWriter
    
    writer = WeatherStreamWriter()
    writer.write_frame(u_tensor, v_tensor, timestamp)
"""

import mmap
import os
import ctypes
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .protocol import (
    WeatherHeader,
    SHM_PATH,
    SHM_SIZE,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
)


class WeatherStreamWriter:
    """
    Writes weather tensor data to shared memory for Rust visualization.
    
    Memory Layout:
        [0:72]     WeatherHeader
        [72:N]     U-wind tensor (float32)
        [N:M]      V-wind tensor (float32)
    """
    
    def __init__(self, path: str = SHM_PATH, size: int = SHM_SIZE):
        """
        Initialize the weather stream writer.
        
        Args:
            path: Path to shared memory file
            size: Maximum size of shared memory region
        """
        self.path = path
        self.size = size
        self.frame_number = 0
        self._mmap: Optional[mmap.mmap] = None
        self._file = None
        
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
        
    def open(self):
        """Open/create the shared memory file."""
        # Ensure parent directory exists (for non-/dev/shm paths)
        parent = Path(self.path).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        
        # Create/open file
        self._file = open(self.path, "w+b")
        self._file.truncate(self.size)
        self._file.flush()
        
        # Memory map
        self._mmap = mmap.mmap(
            self._file.fileno(),
            self.size,
            access=mmap.ACCESS_WRITE
        )
        
        print(f"[WEATHER_STREAM] Opened {self.path} ({self.size // (1024*1024)} MB)")
        
    def close(self):
        """Close the shared memory file."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None
            
    def is_open(self) -> bool:
        """Check if writer is open."""
        return self._mmap is not None
    
    def write_frame(
        self,
        u_tensor: np.ndarray,
        v_tensor: np.ndarray,
        timestamp: Optional[int] = None,
        valid_time: Optional[int] = None,
        lat_bounds: tuple[float, float] = (21.138, 47.843),
        lon_bounds: tuple[float, float] = (-122.720, -60.918),
    ) -> bool:
        """
        Write a weather frame to shared memory.
        
        Args:
            u_tensor: East/West wind component [H, W] float32
            v_tensor: North/South wind component [H, W] float32
            timestamp: Unix timestamp (default: now)
            valid_time: Forecast valid time (default: timestamp)
            lat_bounds: (lat_min, lat_max)
            lon_bounds: (lon_min, lon_max)
            
        Returns:
            True if write succeeded
        """
        if not self.is_open():
            self.open()
            
        # Ensure tensors are correct type
        u_tensor = np.ascontiguousarray(u_tensor, dtype=np.float32)
        v_tensor = np.ascontiguousarray(v_tensor, dtype=np.float32)
        
        if u_tensor.shape != v_tensor.shape:
            raise ValueError(f"Tensor shape mismatch: {u_tensor.shape} vs {v_tensor.shape}")
        
        if u_tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {u_tensor.shape}")
        
        h, w = u_tensor.shape
        
        # Check size
        header_size = ctypes.sizeof(WeatherHeader)
        tensor_bytes = u_tensor.nbytes + v_tensor.nbytes
        total_size = header_size + tensor_bytes
        
        if total_size > self.size:
            raise ValueError(
                f"Data too large: {total_size} bytes > {self.size} bytes available"
            )
        
        # Compute statistics
        magnitude = np.sqrt(u_tensor**2 + v_tensor**2)
        max_speed = float(np.max(magnitude))
        mean_speed = float(np.mean(magnitude))
        
        # Build header
        now = int(time.time()) if timestamp is None else timestamp
        
        header = WeatherHeader()
        header.magic = PROTOCOL_MAGIC
        header.version = PROTOCOL_VERSION
        header.timestamp = now
        header.valid_time = valid_time or now
        header.grid_w = w
        header.grid_h = h
        header.lat_min = lat_bounds[0]
        header.lat_max = lat_bounds[1]
        header.lon_min = lon_bounds[0]
        header.lon_max = lon_bounds[1]
        header.max_wind_speed = max_speed
        header.mean_wind_speed = mean_speed
        header.frame_number = self.frame_number
        header.is_ready = 0  # Will set to 1 after data write
        
        # Write header (not ready yet)
        self._mmap.seek(0)
        self._mmap.write(bytes(header))
        
        # Write U tensor
        offset = header_size
        self._mmap.seek(offset)
        self._mmap.write(u_tensor.tobytes())
        
        # Write V tensor
        offset += u_tensor.nbytes
        self._mmap.seek(offset)
        self._mmap.write(v_tensor.tobytes())
        
        # Flush to ensure data is written
        self._mmap.flush()
        
        # Now mark as ready (atomic flag)
        header.is_ready = 1
        self._mmap.seek(0)
        self._mmap.write(bytes(header))
        self._mmap.flush()
        
        self.frame_number += 1
        
        print(f"[BRIDGE] Frame {self.frame_number - 1}: {w}x{h}, max_speed={max_speed:.1f} m/s")
        
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_global_writer: Optional[WeatherStreamWriter] = None


def get_writer() -> WeatherStreamWriter:
    """Get the global weather stream writer (creates if needed)."""
    global _global_writer
    if _global_writer is None:
        _global_writer = WeatherStreamWriter()
        _global_writer.open()
    return _global_writer


def write_to_bridge(
    u_tensor: np.ndarray,
    v_tensor: np.ndarray,
    timestamp: int
):
    """
    Write weather data to the shared memory bridge.
    
    This is the simplified API matching the reference implementation.
    """
    writer = get_writer()
    writer.write_frame(u_tensor, v_tensor, timestamp)


def close_bridge():
    """Close the global writer."""
    global _global_writer
    if _global_writer is not None:
        _global_writer.close()
        _global_writer = None


# ═══════════════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_write():
    """Test writing synthetic data."""
    print("=" * 60)
    print("  Weather Stream Writer Test")
    print("=" * 60)
    
    # Create synthetic wind data
    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w]
    
    # Circular wind pattern
    cx, cy = w // 2, h // 2
    dx = (x - cx).astype(np.float32)
    dy = (y - cy).astype(np.float32)
    
    u_tensor = -dy * 0.1  # East/West component
    v_tensor = dx * 0.1   # North/South component
    
    print(f"[TEST] Created synthetic wind field: {w}x{h}")
    
    with WeatherStreamWriter() as writer:
        writer.write_frame(u_tensor, v_tensor)
        print(f"[SUCCESS] Frame written to {writer.path}")
        
    print("=" * 60)
    print("  ✓ EXIT GATE 1B-6: Python wrote data with is_ready=1")
    print("=" * 60)


if __name__ == "__main__":
    _test_write()
