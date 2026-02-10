"""Metrics, uncertainty quantification, and optimization.

Submodules:
  aesthetic   – Proportion, symmetry, and angular metrics
  functional  – Airway resistance, valve area, flow-rate metrics
  safety      – Stress/strain limits, vascularity preservation
  uncertainty – Monte Carlo UQ and Sobol sensitivity analysis
  optimizer   – Multi-objective plan optimization (NSGA-II)
"""

from .aesthetic import AestheticMetrics, AestheticReport
from .functional import FunctionalMetrics, FunctionalReport
from .safety import SafetyMetrics, SafetyReport
from .uncertainty import UncertaintyQuantifier, UQResult, SobolIndices
from .optimizer import PlanOptimizer, OptimizationResult, ParetoFront

__all__ = [
    "AestheticMetrics",
    "AestheticReport",
    "FunctionalMetrics",
    "FunctionalReport",
    "SafetyMetrics",
    "SafetyReport",
    "UncertaintyQuantifier",
    "UQResult",
    "SobolIndices",
    "PlanOptimizer",
    "OptimizationResult",
    "ParetoFront",
]
