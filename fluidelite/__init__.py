"""
FluidElite: Infinite-Context QTT-Native LLM
===========================================

A self-contained tensor network language model that treats
language modeling as fluid dynamics using MPS/MPO representations.

This module is ISOLATED and can be removed in one piece by
deleting the fluidelite/ directory.

Constitutional Compliance:
    - Article IV.4.4: User data separate from program files
    - Article V.5.1: All public APIs documented
    - Article VII.7.2: Definition of Done = USER-OBSERVABLE BEHAVIOR works
"""

from fluidelite.core.mps import MPS
from fluidelite.core.mpo import MPO
from fluidelite.core.decompositions import svd_truncated, qr_positive, rsvd_truncated, SafeSVD
from fluidelite.core.fast_ops import vectorized_mpo_apply, vectorized_mps_add
from fluidelite.core.cross import ProjectedActivation, gelu_mps
from fluidelite.optim.riemannian import RiemannianAdam
from fluidelite.llm.fluid_elite import FluidElite, EliteLinear
from fluidelite.llm.data import TextStreamDataset, create_loader

__version__ = "0.1.0"
__all__ = [
    # Core
    "MPS",
    "MPO",
    "svd_truncated",
    "qr_positive",
    "rsvd_truncated",
    "SafeSVD",
    "vectorized_mpo_apply",
    "vectorized_mps_add",
    "ProjectedActivation",
    "gelu_mps",
    # Optimization
    "RiemannianAdam",
    # LLM
    "FluidElite",
    "EliteLinear",
    "TextStreamDataset",
    "create_loader",
]
