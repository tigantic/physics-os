"""
Ontic Engine Medical Module - Hemodynamic Simulation

Phase 11: The Surgical Pre-Flight
- Non-Newtonian blood flow (shear thinning)
- Stenosis (plaque blockage) modeling
- Wall shear stress → Rupture risk prediction

The physics that saves lives before the scalpel.
"""

from .hemo import ArterySimulation, StenosisReport

__all__ = [
    "ArterySimulation",
    "StenosisReport",
]
