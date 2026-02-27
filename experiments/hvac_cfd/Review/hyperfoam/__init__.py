"""
HyperFOAM: GPU-Native CFD for HVAC Digital Twins

A PyTorch-based computational fluid dynamics solver optimized for
real-time HVAC simulation and AI-driven inverse design.

Features:
- GPU-accelerated Navier-Stokes solver (torch.compile optimized)
- Immersed boundary method for complex geometry
- Thermal transport with buoyancy coupling
- CO2 and contaminant tracking
- ASHRAE 55 compliance checking

Example:
    >>> import hyperfoam
    >>> solver = hyperfoam.Solver(hyperfoam.ConferenceRoom())
    >>> solver.solve(duration=300)
    >>> metrics = solver.get_comfort_metrics()
    >>> print(metrics)

CLI:
    python -m hyperfoam              # Launch dashboard
    python -m hyperfoam demo         # Run CLI demo
    python -m hyperfoam benchmark    # Performance test
    python -m hyperfoam optimize     # Auto-find optimal settings
    python -m hyperfoam report       # Generate PDF report

Author: TiganticLabz
License: MIT
"""

__version__ = "0.1.0"
__author__ = "TiganticLabz"

# Core classes
from .core.grid import HyperGrid
from .core.solver import HyperFoamSolver, ProjectionConfig
from .core.thermal import (
    ThermalMultiPhysicsSolver,
    ThermalSystemConfig,
    AirProperties,
    BuoyancyConfig,
    HeatSource,
    HeatSourceType,
    ScalarField,
)

# High-level API
from .solver import Solver, SolverConfig
from .presets import ConferenceRoom, OpenOffice, ServerRoom
from .optimizer import optimize_hvac, quick_optimize, HVACOptimizer, OptimizationResult

# Convenience
__all__ = [
    # Version
    "__version__",
    
    # High-level API
    "Solver",
    "SolverConfig",
    
    # Presets
    "ConferenceRoom",
    "OpenOffice",
    "ServerRoom",
    
    # Optimizer
    "optimize_hvac",
    "quick_optimize",
    "HVACOptimizer",
    "OptimizationResult",
    
    # Core (for advanced users)
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
]

# New capability modules (lazy imports for performance)
def __getattr__(name):
    """Lazy import of extended modules."""
    if name == "turbulence":
        from .core import turbulence
        return turbulence
    elif name == "grid_convergence":
        from .core import grid_convergence
        return grid_convergence
    elif name == "cleanroom":
        from . import cleanroom
        return cleanroom
    elif name == "predictive_alerts":
        from . import predictive_alerts
        return predictive_alerts
    elif name == "cad_import":
        from . import cad_import
        return cad_import
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
