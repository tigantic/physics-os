"""
TensorNet Agriculture Module

Phase 15: Vertical Farm Microclimate Optimization

The final frontier of the Planetary Operating System:
Precision agriculture in controlled environments.

Classes:
    VerticalFarm: Microclimate simulation for indoor agriculture
    HarvestReport: Crop yield and quality predictions
"""

from .microclimate import VerticalFarm, HarvestReport, optimize_climate

__all__ = [
    "VerticalFarm",
    "HarvestReport",
    "optimize_climate",
]
