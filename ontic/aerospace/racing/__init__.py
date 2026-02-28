"""
TensorNet Racing Module - Aerodynamic Wake Analysis

Phase 12: The Invisible Wall
- Dirty air wake tracking behind race cars
- Turbulent vortex visualization
- Real-time overtake window detection

The physics that wins races.
"""

from .wake import DirtyAirReport, WakeTracker, track_dirty_air

__all__ = [
    "WakeTracker",
    "DirtyAirReport",
    "track_dirty_air",
]
