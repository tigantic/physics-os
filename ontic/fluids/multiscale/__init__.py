"""
Multiscale methods package: FE², homogenisation, quasi-continuum, bridging.

Domains: XVIII.7.
"""

from .multiscale import (
    RVEHomogenisation,
    FE2Solver,
    MicroState,
    QuasiContinuum,
    HierarchicalBridge,
)

__all__ = [
    "RVEHomogenisation",
    "FE2Solver",
    "MicroState",
    "QuasiContinuum",
    "HierarchicalBridge",
]
