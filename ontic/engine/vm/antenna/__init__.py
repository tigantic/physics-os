"""QTT Physics VM — Antenna design automation.

Parametric antenna geometry engine, material library, sweep orchestrator,
and Pareto multi-objective optimizer for automated antenna IP discovery.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from .materials import Material, MaterialLibrary
from .geometry import (
    DesignVariable,
    DesignSpace,
    PatchAntennaDesign,
    DipoleAntennaDesign,
    EShapedPatchDesign,
    USlotsDesign,
)
from .sweep import SweepOrchestrator, SweepResult, DesignPoint
from .pareto import ParetoOptimizer, ParetoResult, ScoredCandidate

__all__ = [
    "Material",
    "MaterialLibrary",
    "DesignVariable",
    "DesignSpace",
    "PatchAntennaDesign",
    "DipoleAntennaDesign",
    "EShapedPatchDesign",
    "USlotsDesign",
    "SweepOrchestrator",
    "SweepResult",
    "DesignPoint",
    "ParetoOptimizer",
    "ParetoResult",
    "ScoredCandidate",
]
