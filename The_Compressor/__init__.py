"""
The_Compressor: 63,321x QTT Compression Engine
==============================================
Compresses multi-GB satellite/volumetric data to L2-cache-resident sizes.

Usage:
    from The_Compressor import compress, decompress
    
    # Compress 17GB NOAA data to 258KB
    python -m The_Compressor.compress --input noaa_24h_raw --output data.npz
    
    # Query single point (microseconds)
    python -m The_Compressor.decompress --input data.npz --point 16,1024,1024
    
    # Reconstruct frame
    python -m The_Compressor.decompress --input data.npz --frame 16 --output frame.npy

Performance:
    - 16.95 GB → 258 KB (63,321x ratio)
    - VRAM: <100 MB on 8GB card
    - RAM: Zero (mmap streaming)
    - Point query: ~100 µs
"""

__version__ = "1.0.0"
__author__ = "Tigantic Labs"
