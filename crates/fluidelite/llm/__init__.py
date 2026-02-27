"""
FluidElite LLM Module
=====================

Contains the main FluidElite model and data utilities.
"""

try:
    from fluidelite.llm.fluid_elite import FluidElite, EliteLinear
    from fluidelite.llm.data import TextStreamDataset, create_loader
except ImportError:
    from .fluid_elite import FluidElite, EliteLinear
    from .data import TextStreamDataset, create_loader

__all__ = [
    "FluidElite",
    "EliteLinear",
    "TextStreamDataset",
    "create_loader",
]
