"""
PDE Solvers - Partial differential equation solvers.
"""

from hypertensor.pde.mhd import ResistiveMHD, IdealMHD
from hypertensor.pde.fokker_planck import FokkerPlanck
from hypertensor.pde.diffusion import HeatEquation1D, CompositeWall

__all__ = [
    "ResistiveMHD",
    "IdealMHD",
    "FokkerPlanck",
    "HeatEquation1D",
    "CompositeWall",
]
