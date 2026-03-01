"""
Ontic Engine Emergency Module - Wildfire Prediction

Phase 14: The Wildfire Prophet
- Fire-atmosphere coupling
- Ember transport (spotting)
- Fire front propagation prediction

The physics that saves communities.
"""

from .fire import FireReport, FireSim

__all__ = [
    "FireSim",
    "FireReport",
]
