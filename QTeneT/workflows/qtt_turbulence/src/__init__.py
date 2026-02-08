"""
QTT Turbulence Workflow - Self-Contained Package

Production solver for 3D turbulence simulation using QTT format.
"""

from .qtt_core import QTTCores, QTT3DNative, QTT3DVectorNative, qtt_truncate_sweep
from .spectral_ns3d import SpectralNS3D, SpectralNS3DConfig

__all__ = [
    'QTTCores',
    'QTT3DNative', 
    'QTT3DVectorNative',
    'qtt_truncate_sweep',
    'SpectralNS3D',
    'SpectralNS3DConfig',
]
__version__ = '1.0.0'
