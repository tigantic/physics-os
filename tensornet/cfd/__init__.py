"""HyperTensor CFD — public re-exports.

Every symbol listed in ``__all__`` is importable directly from
``tensornet.cfd`` so that downstream code (tests, experiments, docs,
CUDA bridge) can use short-form imports like::

    from tensornet.cfd import Euler1D, exact_riemann, sod_shock_tube_ic
"""

from __future__ import annotations

# --- euler_1d -----------------------------------------------------------
from tensornet.cfd.euler_1d import (
    BCType1D,
    Euler1D,
    EulerState,
    sod_shock_tube_ic,
)

# --- godunov (Riemann solvers) ------------------------------------------
from tensornet.cfd.godunov import exact_riemann

# --- euler_2d -----------------------------------------------------------
from tensornet.cfd.euler_2d import Euler2D, supersonic_wedge_ic

# --- les (large-eddy simulation) ----------------------------------------
from tensornet.cfd.les import (
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
from tensornet.cfd.hybrid_les import (
    HybridLESState,
    HybridModel,
    ddes_delay_function,
    des_length_scale,
    estimate_rans_les_ratio,
    run_hybrid_les,
)

# --- multi_objective (Pareto front / MOO) --------------------------------
from tensornet.cfd.multi_objective import (
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
from tensornet.cfd.navier_stokes import NavierStokes2D, NavierStokes2DConfig

# --- pure_qtt_ops (imported as submodule by tensornet.cuda) ---------------
from tensornet.cfd import pure_qtt_ops  # noqa: F401  — submodule re-export

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
    # submodules
    "pure_qtt_ops",
]
