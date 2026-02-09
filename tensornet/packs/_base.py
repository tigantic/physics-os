"""
Shared utilities for domain pack implementations.

Provides helper classes so that every pack can define a ProblemSpec-conformant
class and wire up toy or full solvers with minimal boilerplate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from tensornet.platform.data_model import (
    FieldData,
    SimulationState,
    StructuredMesh,
)
from tensornet.platform.protocols import SolveResult
from tensornet.platform.solvers import RK4, RHSCallable, TimeIntegrator


# ═══════════════════════════════════════════════════════════════════════════════
# Generic ProblemSpec base
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BaseProblemSpec:
    """
    Frozen dataclass satisfying the ProblemSpec protocol.

    Subclass or instantiate directly:

        spec = BaseProblemSpec(
            _name="MyProblem",
            _ndim=1,
            _parameters={"nu": 0.01},
            _governing_equations=r"\\partial_t u = ...",
            _field_names=("u",),
            _observable_names=("energy",),
        )
    """

    _name: str
    _ndim: int
    _parameters: Dict[str, Any]
    _governing_equations: str
    _field_names: Tuple[str, ...]
    _observable_names: Tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self._name

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def parameters(self) -> Dict[str, Any]:
        return dict(self._parameters)

    @property
    def governing_equations(self) -> str:
        return self._governing_equations

    @property
    def field_names(self) -> Sequence[str]:
        return self._field_names

    @property
    def observable_names(self) -> Sequence[str]:
        return self._observable_names


# ═══════════════════════════════════════════════════════════════════════════════
# Simple ODE/PDE runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_ode_problem(
    rhs: RHSCallable,
    state0: SimulationState,
    t_span: Tuple[float, float],
    dt: float,
    integrator: Optional[TimeIntegrator] = None,
) -> SolveResult:
    """
    Run a generic ODE/PDE problem through the platform integrator stack.

    Parameters
    ----------
    rhs : RHSCallable
        Right-hand-side yielding field derivatives.
    state0 : SimulationState
        Initial state.
    t_span : (t0, tf)
        Integration window.
    dt : float
        Time step.
    integrator : TimeIntegrator, optional
        Defaults to RK4.
    """
    if integrator is None:
        integrator = RK4()
    return integrator.solve(state0, rhs, t_span, dt)


# ═══════════════════════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════════════════════


def compute_linf_error(numerical: Tensor, exact: Tensor) -> float:
    """L-infinity error between numerical and exact solutions."""
    return (numerical - exact).abs().max().item()


def compute_l2_error(numerical: Tensor, exact: Tensor, dx: float = 1.0) -> float:
    """Discrete L2 error: sqrt(sum((num - exact)^2) * dx)."""
    return math.sqrt(((numerical - exact) ** 2).sum().item() * dx)


def convergence_order(
    errors: Sequence[float],
    resolutions: Sequence[int],
) -> List[float]:
    """
    Compute observed convergence orders from a sequence of errors and
    corresponding resolutions (number of cells).

    Returns a list of length ``len(errors) - 1``.
    """
    orders: List[float] = []
    for i in range(len(errors) - 1):
        if errors[i + 1] <= 0.0 or errors[i] <= 0.0:
            orders.append(float("nan"))
            continue
        ratio = math.log(errors[i] / errors[i + 1])
        h_ratio = math.log(resolutions[i + 1] / resolutions[i])
        orders.append(ratio / h_ratio if h_ratio != 0.0 else float("nan"))
    return orders


def make_1d_state(
    field_name: str,
    ic_fn: Callable[[Tensor], Tensor],
    N: int,
    domain: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[SimulationState, StructuredMesh]:
    """
    Create a 1-D initial state on a StructuredMesh.

    Returns (state, mesh).
    """
    mesh = StructuredMesh(shape=(N,), domain=(domain,))
    x = mesh.cell_centers().squeeze(-1)
    data = ic_fn(x)
    field = FieldData(name=field_name, data=data, mesh=mesh)
    state = SimulationState(t=0.0, fields={field_name: field}, mesh=mesh)
    return state, mesh
