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

from ontic.platform.data_model import (
    FieldData,
    SimulationState,
    StructuredMesh,
)
from ontic.platform.protocols import SolveResult
from ontic.platform.solvers import RK4, RHSCallable, TimeIntegrator


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


# ═══════════════════════════════════════════════════════════════════════════════
# V0.2 Reference Solver Patterns
# ═══════════════════════════════════════════════════════════════════════════════


class ODEReferenceSolver:
    """
    Generic ODE-system solver for V0.2 reference implementations.

    Given dy/dt = f(y, t), integrates with RK4 and validates against
    an exact/reference solution.

    Parameters
    ----------
    name : str
        Solver name.
    rhs_fn : Callable[[Tensor, float], Tensor]
        Right-hand-side f(y, t) returning dy/dt.
    y0 : Tensor
        Initial state vector.
    t_span : (t0, tf)
        Integration window.
    dt : float
        Time step.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def solve_ode(
        self,
        rhs_fn: Callable[[Tensor, float], Tensor],
        y0: Tensor,
        t_span: Tuple[float, float],
        dt: float,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Fourth-order Runge-Kutta integration.

        Returns (y_final, trajectory) where trajectory is a list of
        state snapshots at each time step.
        """
        t = t_span[0]
        y = y0.clone().to(torch.float64)
        trajectory: List[Tensor] = [y.clone()]
        while t < t_span[1] - 1e-14:
            h = min(dt, t_span[1] - t)
            k1 = rhs_fn(y, t)
            k2 = rhs_fn(y + 0.5 * h * k1, t + 0.5 * h)
            k3 = rhs_fn(y + 0.5 * h * k2, t + 0.5 * h)
            k4 = rhs_fn(y + h * k3, t + h)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            trajectory.append(y.clone())
        return y, trajectory


class PDE1DReferenceSolver:
    """
    Generic 1-D PDE solver for V0.2 reference implementations.

    Uses method-of-lines: spatial discretization → ODE system → RK4.
    Supports periodic and Dirichlet BCs.

    Parameters
    ----------
    name : str
        Solver name.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def solve_pde(
        self,
        rhs_fn: Callable[[Tensor, float, float], Tensor],
        u0: Tensor,
        dx: float,
        t_span: Tuple[float, float],
        dt: float,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        RK4 integration of a semi-discrete PDE: du/dt = rhs(u, t, dx).

        Parameters
        ----------
        rhs_fn : (u, t, dx) -> du/dt
        u0 : Tensor — initial field values, shape (N,)
        dx : float — grid spacing
        t_span : (t0, tf)
        dt : float

        Returns (u_final, trajectory).
        """
        t = t_span[0]
        u = u0.clone().to(torch.float64)
        trajectory: List[Tensor] = [u.clone()]
        while t < t_span[1] - 1e-14:
            h = min(dt, t_span[1] - t)
            k1 = rhs_fn(u, t, dx)
            k2 = rhs_fn(u + 0.5 * h * k1, t + 0.5 * h, dx)
            k3 = rhs_fn(u + 0.5 * h * k2, t + 0.5 * h, dx)
            k4 = rhs_fn(u + h * k3, t + h, dx)
            u = u + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            trajectory.append(u.clone())
        return u, trajectory


class EigenReferenceSolver:
    """
    Generic eigenvalue solver for V0.2 reference implementations.

    Solves Hu = Eu (or generalized Ax = λBx).
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def solve_eigenproblem(
        H: Tensor, n_states: int = 5
    ) -> Tuple[Tensor, Tensor]:
        """
        Solve a Hermitian eigenproblem.

        Returns (eigenvalues[:n_states], eigenvectors[:, :n_states]).
        """
        H = H.to(torch.float64)
        vals, vecs = torch.linalg.eigh(H)
        return vals[:n_states], vecs[:, :n_states]


class MonteCarloReferenceSolver:
    """
    Generic Monte Carlo solver for V0.2 reference implementations.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name


def validate_v02(
    *,
    error: float,
    tolerance: float,
    label: str = "",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Standard V0.2 validation gate: error < tolerance.

    Returns a dict with 'passed', 'error', 'tolerance', 'label'.
    """
    passed = error < tolerance
    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}: error={error:.6e}, tol={tolerance:.6e}")
    return {"passed": passed, "error": error, "tolerance": tolerance, "label": label}
