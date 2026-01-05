"""
HyperVisual - Resolution-Independent Visualization
===================================================

Layer 2 of HyperTensor architecture.
Tile-based rendering that never decompresses the full field.

Core Components:
    TileRenderer: LOD-aware tile generation from QTT fields
    SliceEngine: Arbitrary plane slicing, streaming extraction
    StreamProtocol: WebSocket/async streaming for real-time viz
    ColorMaps: Scientific colormaps, opacity transfer functions

Usage:
    from tensornet.hypervisual import TileRenderer, SliceEngine

    # Render at any resolution
    renderer = TileRenderer(field, tile_size=256)
    tile = renderer.get_tile(x=0, y=0, lod=3)

    # Slice any plane
    slicer = SliceEngine(field)
    image = slicer.slice(plane='xy', depth=0.5, resolution=1024)
"""

from .colormaps import (
                        COOLWARM,
                        GRAYSCALE,
                        INFERNO,
                        JET,
                        MAGMA,
                        PLASMA,
                        TURBO,
                        VIRIDIS,
                        ColorMap,
                        TransferFunction,
                        apply_colormap,
)
from .renderer import LODPyramid, RenderConfig, RenderStats, Tile, TileCoord, TileRenderer
from .slicer import SliceEngine, SlicePlane, SliceResult, VolumeRenderer
from .slicing_core import MortonSlicer, MortonSliceResult, compare_slicing_methods
from .stream import FrameBuffer, StreamConfig, StreamProtocol, StreamStats

__all__ = [
    # Renderer
    "TileRenderer",
    "Tile",
    "TileCoord",
    "LODPyramid",
    "RenderConfig",
    "RenderStats",
    # Slicer
    "SliceEngine",
    "SlicePlane",
    "SliceResult",
    "VolumeRenderer",
    # Morton-Aware Slicer (true resolution-independent)
    "MortonSlicer",
    "MortonSliceResult",
    "compare_slicing_methods",
    # Stream
    "StreamProtocol",
    "StreamConfig",
    "FrameBuffer",
    "StreamStats",
    # Colormaps
    "ColorMap",
    "TransferFunction",
    "VIRIDIS",
    "PLASMA",
    "INFERNO",
    "MAGMA",
    "TURBO",
    "COOLWARM",
    "JET",
    "GRAYSCALE",
    "apply_colormap",
]

__version__ = "0.1.0"
