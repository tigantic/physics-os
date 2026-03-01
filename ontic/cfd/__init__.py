"""Ontic CFD — public re-exports.

Every symbol listed in ``__all__`` is importable directly from
``ontic.cfd`` so that downstream code (tests, experiments, docs,
CUDA bridge) can use short-form imports like::

    from ontic.cfd import Euler1D, exact_riemann, sod_shock_tube_ic
"""

from __future__ import annotations

# --- euler_1d -----------------------------------------------------------
from ontic.cfd.euler_1d import (
    BCType1D,
    Euler1D,
    EulerState,
    sod_shock_tube_ic,
)

# --- godunov (Riemann solvers) ------------------------------------------
from ontic.cfd.godunov import exact_riemann

# --- euler_2d -----------------------------------------------------------
from ontic.cfd.euler_2d import Euler2D, supersonic_wedge_ic

# --- les (large-eddy simulation) ----------------------------------------
from ontic.cfd.les import (
    LESModel,
    LESState,
    compute_sgs_viscosity,
    filter_width,
    smagorinsky_viscosity,
    strain_rate_magnitude,
    vreman_viscosity,
    wale_viscosity,
)

# --- hybrid_les (DES / DDES) --------------------------------------------
from ontic.cfd.hybrid_les import (
    HybridLESState,
    HybridModel,
    ddes_delay_function,
    des_length_scale,
    estimate_rans_les_ratio,
    run_hybrid_les,
)

# --- multi_objective (Pareto front / MOO) --------------------------------
from ontic.cfd.multi_objective import (
    MOOAlgorithm,
    MOOConfig,
    MOOResult,
    MultiObjectiveOptimizer,
    ObjectiveSpec,
    ParetoSolution,
    dominates,
    fast_non_dominated_sort,
)

# --- navier_stokes -------------------------------------------------------
from ontic.cfd.navier_stokes import NavierStokes2D, NavierStokes2DConfig

# --- geometry / SDF library ----------------------------------------------
from ontic.cfd.geometry import ImmersedBoundary, WedgeGeometry
from ontic.cfd.sdf import (
    CircleSDF,
    ConcentricAnnulusSDF,
    EllipseSDF,
    FinArraySDF,
    FlatPlateSDF,
    MultiBodySDF,
    NACA4DigitSDF,
    PipeBendSDF,
    RectangleSDF,
    RoundedRectSDF,
    SDFGeometry,
    StepSDF,
    WedgeSDF,
)

# --- pure_qtt_ops (imported as submodule by ontic.cuda) ---------------
from ontic.cfd import pure_qtt_ops  # noqa: F401  — submodule re-export

__all__ = [
    # euler_1d
    "BCType1D",
    "Euler1D",
    "EulerState",
    "sod_shock_tube_ic",
    # godunov
    "exact_riemann",
    # euler_2d
    "Euler2D",
    "supersonic_wedge_ic",
    # les
    "LESModel",
    "LESState",
    "compute_sgs_viscosity",
    "filter_width",
    "smagorinsky_viscosity",
    "strain_rate_magnitude",
    "vreman_viscosity",
    "wale_viscosity",
    # hybrid_les
    "HybridLESState",
    "HybridModel",
    "ddes_delay_function",
    "des_length_scale",
    "estimate_rans_les_ratio",
    "run_hybrid_les",
    # multi_objective
    "MOOAlgorithm",
    "MOOConfig",
    "MOOResult",
    "MultiObjectiveOptimizer",
    "ObjectiveSpec",
    "ParetoSolution",
    "dominates",
    "fast_non_dominated_sort",
    # navier_stokes
    "NavierStokes2D",
    "NavierStokes2DConfig",
    # geometry / SDF
    "SDFGeometry",
    "CircleSDF",
    "EllipseSDF",
    "RectangleSDF",
    "RoundedRectSDF",
    "WedgeSDF",
    "NACA4DigitSDF",
    "FlatPlateSDF",
    "FinArraySDF",
    "PipeBendSDF",
    "ConcentricAnnulusSDF",
    "MultiBodySDF",
    "StepSDF",
    "ImmersedBoundary",
    "WedgeGeometry",
    # submodules
    "pure_qtt_ops",
]
