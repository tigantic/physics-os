"""
Ontic Engine Fuel Module
=====================

OPERATION VALHALLA - Phase 3: THE FUEL

Orbital data integration: S3→VRAM pipeline for satellite imagery.
Direct streaming from NOAA/NASA to GPU memory.

Modules:
    - s3_fetcher: Async S3 client with HTTP/2 (requires aiohttp)
    - tile_compositor: GPU tile assembly
    - cache: LRU cache for tile management

Author: OPERATION VALHALLA
Date: 2025-12-28
"""

from .tile_compositor import TileCompositor

try:
    from .s3_fetcher import S3Fetcher

    __all__ = ["S3Fetcher", "TileCompositor"]
except ImportError:
    # aiohttp not installed - compositor still works
    __all__ = ["TileCompositor"]
