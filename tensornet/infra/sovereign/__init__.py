"""
Sovereign Architecture: End-to-End Tensor Sparsity

Modules:
    - implicit_qtt_renderer: Implicit QTT evaluation without materialization
    - protocol: Wire format definitions for Python ↔ Rust IPC
    - weather_stream: Weather data writer for Global Eye
"""

from .implicit_qtt_renderer import ImplicitQTTRenderer, test_implicit_renderer

# Weather Bridge (Global Eye Phase 1)
from .protocol import (
                       PROTOCOL_MAGIC,
                       PROTOCOL_VERSION,
                       SHM_PATH,
                       SHM_SIZE,
                       WeatherHeader,
                       verify_protocol,
)
from .weather_stream import WeatherStreamWriter, close_bridge, get_writer, write_to_bridge

__all__ = [
    # QTT
    "ImplicitQTTRenderer",
    "test_implicit_renderer",
    # Protocol
    "WeatherHeader",
    "PROTOCOL_MAGIC",
    "PROTOCOL_VERSION",
    "SHM_PATH",
    "SHM_SIZE",
    "verify_protocol",
    # Weather Stream
    "WeatherStreamWriter",
    "write_to_bridge",
    "close_bridge",
    "get_writer",
]
