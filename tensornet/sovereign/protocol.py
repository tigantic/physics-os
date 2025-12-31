"""
Weather Bridge Protocol - Global Eye Phase 1B-5
================================================

Defines the shared memory protocol between Python (producer) and Rust (consumer).
This must match EXACTLY with crates/hyper_bridge/src/weather.rs

Protocol Layout:
    Bytes 0-47:    WeatherHeader (48 bytes, padded for alignment)
    Bytes 48-N:    U-wind tensor (grid_h * grid_w * 4 bytes)
    Bytes N-M:     V-wind tensor (grid_h * grid_w * 4 bytes)

Total size depends on grid dimensions (HRRR CONUS is typically 1799x1059).
"""

import ctypes
from dataclasses import dataclass
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol Constants
# ═══════════════════════════════════════════════════════════════════════════════

PROTOCOL_MAGIC = 0x47_4C_4F_42  # "GLOB" in little-endian
PROTOCOL_VERSION = 1

SHM_PATH = "/dev/shm/hyper_weather_v1"
SHM_SIZE = 64 * 1024 * 1024  # 64MB (fits HRRR grid with margin)


# ═══════════════════════════════════════════════════════════════════════════════
# Weather Header Structure
# ═══════════════════════════════════════════════════════════════════════════════

class WeatherHeader(ctypes.Structure):
    """
    Binary header for weather data shared memory.
    
    Must match Rust definition exactly:
        #[repr(C)]
        pub struct WeatherHeader { ... }
    
    Total size: 48 bytes (with alignment padding)
    """
    _pack_ = 1  # Disable automatic padding
    _fields_ = [
        # ─── Identification ───────────────────────────────────────────────────
        ("magic", ctypes.c_uint32),       # 4 bytes: 0x474C4F42 = "GLOB"
        ("version", ctypes.c_uint32),     # 4 bytes: Protocol version
        
        # ─── Temporal ─────────────────────────────────────────────────────────
        ("timestamp", ctypes.c_uint64),   # 8 bytes: Unix timestamp (seconds)
        ("valid_time", ctypes.c_uint64),  # 8 bytes: Forecast valid time
        
        # ─── Grid Dimensions ──────────────────────────────────────────────────
        ("grid_w", ctypes.c_uint32),      # 4 bytes: Width of tensor
        ("grid_h", ctypes.c_uint32),      # 4 bytes: Height of tensor
        
        # ─── Geographic Bounds ────────────────────────────────────────────────
        ("lat_min", ctypes.c_float),      # 4 bytes: Southern boundary
        ("lat_max", ctypes.c_float),      # 4 bytes: Northern boundary
        ("lon_min", ctypes.c_float),      # 4 bytes: Western boundary
        ("lon_max", ctypes.c_float),      # 4 bytes: Eastern boundary
        
        # ─── Statistics (for shader normalization) ────────────────────────────
        ("max_wind_speed", ctypes.c_float),  # 4 bytes: Max magnitude
        ("mean_wind_speed", ctypes.c_float), # 4 bytes: Mean magnitude
        
        # ─── Synchronization ──────────────────────────────────────────────────
        ("frame_number", ctypes.c_uint64),   # 8 bytes: Monotonic counter
        ("is_ready", ctypes.c_uint32),       # 4 bytes: 1 = data ready
        
        # ─── Padding to 64-byte alignment ─────────────────────────────────────
        ("_padding", ctypes.c_uint32),       # 4 bytes: Reserved
    ]
    
    def __init__(self):
        super().__init__()
        self.magic = PROTOCOL_MAGIC
        self.version = PROTOCOL_VERSION
        self.is_ready = 0
    
    @classmethod
    def size(cls) -> int:
        """Return the size in bytes."""
        return ctypes.sizeof(cls)
    
    def validate(self) -> bool:
        """Check if header is valid."""
        if self.magic != PROTOCOL_MAGIC:
            return False
        if self.version != PROTOCOL_VERSION:
            return False
        return True
    
    def set_bounds_conus(self):
        """Set geographic bounds for CONUS (Continental US)."""
        self.lat_min = 21.138  # HRRR domain
        self.lat_max = 47.843
        self.lon_min = -122.720
        self.lon_max = -60.918


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WindFrame:
    """Processed wind frame ready for bridge."""
    header: WeatherHeader
    u_tensor: "np.ndarray"  # [H, W] float32
    v_tensor: "np.ndarray"  # [H, W] float32
    
    def total_bytes(self) -> int:
        """Total size of frame in shared memory."""
        tensor_bytes = self.u_tensor.nbytes + self.v_tensor.nbytes
        return WeatherHeader.size() + tensor_bytes


# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_protocol():
    """Print protocol sizes for verification against Rust."""
    header = WeatherHeader()
    print("=" * 50)
    print("  Weather Bridge Protocol v1")
    print("=" * 50)
    print(f"  Header size: {WeatherHeader.size()} bytes")
    print(f"  Magic:       0x{PROTOCOL_MAGIC:08X}")
    print(f"  SHM Path:    {SHM_PATH}")
    print(f"  SHM Size:    {SHM_SIZE // (1024*1024)} MB")
    print()
    print("  Field offsets:")
    for name, ctype in WeatherHeader._fields_:
        offset = getattr(WeatherHeader, name).offset
        size = ctypes.sizeof(ctype)
        print(f"    {name:20} @ {offset:3} ({size} bytes)")
    print("=" * 50)


if __name__ == "__main__":
    verify_protocol()
