"""
TensorNet Defense Module - Hydroacoustic Warfare

Phase 8: The Silent Sub
- Ocean domain with Munk Sound Speed Profile
- FDTD Wave Equation solver for sonar propagation  
- Shadow zone detection for submarine stealth

The physics that hides submarines, now on your GPU.
"""

from .ocean import OceanDomain, SoundSpeedProfile
from .solver import solve_sonar_ping, AcousticField, StealthReport

__all__ = [
    "OceanDomain",
    "SoundSpeedProfile", 
    "solve_sonar_ping",
    "AcousticField",
    "StealthReport",
]
