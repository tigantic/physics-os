"""
Matrix Product Operator (MPO) framework for direct TT-core updates.

Eliminates dense-to-QTT factorization tax (6.05ms) by updating TT-cores directly.
Academic validation: Oseledets (2011), Dolgov & Savostyanov (2014).

Target performance: 0.65ms physics update (5× speedup vs 3.33ms dense solver).
"""

from .atmospheric_solver import MPOAtmosphericSolver
from .operators import LaplacianMPO, AdvectionMPO, ProjectionMPO

__all__ = [
    "MPOAtmosphericSolver",
    "LaplacianMPO",
    "AdvectionMPO",
    "ProjectionMPO",
]
