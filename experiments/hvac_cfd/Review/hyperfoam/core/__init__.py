"""
HyperFOAM Core Module

Low-level components for advanced users.
"""

from .grid import HyperGrid
from .solver import HyperFoamSolver, ProjectionConfig
from .thermal import (
    ThermalMultiPhysicsSolver,
    ThermalSystemConfig,
    AirProperties,
    BuoyancyConfig,
    HeatSource,
    HeatSourceType,
    ScalarField,
)

# New T2+ capabilities
from .turbulence import (
    TurbulenceModel,
    KEpsilonCoefficients,
    KEpsilonSolver,
    TurbulenceState,
    TurbulenceMetrics,
    analyze_turbulence,
    create_k_epsilon_solver,
)

from .grid_convergence import (
    GridLevel,
    GCIResult,
    RichardsonExtrapolation,
    run_grid_study,
    print_gci_report,
)

__all__ = [
    "HyperGrid",
    "HyperFoamSolver",
    "ProjectionConfig",
    "ThermalMultiPhysicsSolver",
    "ThermalSystemConfig",
    "AirProperties",
    "BuoyancyConfig",
    "HeatSource",
    "HeatSourceType",
    "ScalarField",
    # Turbulence
    "TurbulenceModel",
    "KEpsilonCoefficients",
    "KEpsilonSolver",
    "TurbulenceState",
    "TurbulenceMetrics",
    "analyze_turbulence",
    "create_k_epsilon_solver",
    # Grid Convergence
    "GridLevel",
    "GCIResult",
    "RichardsonExtrapolation",
    "run_grid_study",
    "print_gci_report",
]
