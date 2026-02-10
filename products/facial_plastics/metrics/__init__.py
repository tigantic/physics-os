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
from .cohort_analytics import (
    CohortAnalytics,
    CohortReport,
    DistributionStats,
    EffectSize,
    RiskFactor,
    SubgroupAnalysis,
    SurgeonProfile,
    TrendPoint,
)
from .optimizer import PlanOptimizer, OptimizationResult, ParetoFront

__all__ = [
    "AestheticMetrics",
    "AestheticReport",
    "CohortAnalytics",
    "CohortReport",
    "DistributionStats",
    "EffectSize",
    "FunctionalMetrics",
    "FunctionalReport",
    "OptimizationResult",
    "ParetoFront",
    "PlanOptimizer",
    "RiskFactor",
    "SafetyMetrics",
    "SafetyReport",
    "SobolIndices",
    "SubgroupAnalysis",
    "SurgeonProfile",
    "TrendPoint",
    "UncertaintyQuantifier",
    "UQResult",
]
