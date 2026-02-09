"""
Grid & Timestep Refinement Studies
====================================

Automates the refinement-study pattern used in vertical slices:

1. **Grid refinement** — hold dt/dx ratio constant (or use sufficiently small dt),
   vary N = 32, 64, 128, 256, … and measure error vs exact solution.
2. **Timestep refinement** — hold N fixed, vary dt and measure temporal error.

Both studies compute the observed convergence order via least-squares log-log fit
and compare against the formal order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from torch import Tensor

from tensornet.platform.data_model import (
    SimulationState,
    StructuredMesh,
)
from tensornet.platform.solvers import RHSCallable, TimeIntegrator


# ═══════════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RefinementPoint:
    """Error metrics at a single refinement level."""

    level: int
    parameter: float  # dx or dt depending on study type
    linf_error: float
    l1_error: float
    l2_error: float
    steps: int
    wall_time_s: float = 0.0


@dataclass(frozen=True)
class RefinementResult:
    """Full refinement study result."""

    study_type: str  # 'grid' or 'timestep'
    points: List[RefinementPoint]
    observed_order_linf: float
    observed_order_l2: float
    formal_order: int
    passed: bool
    tolerance: float

    def summary(self) -> str:
        kind = "dx" if self.study_type == "grid" else "dt"
        lines = [
            f"{'Grid' if self.study_type == 'grid' else 'Timestep'} Refinement Study",
            "=" * 68,
            f"  Formal order:         {self.formal_order}",
            f"  Observed order (L∞):  {self.observed_order_linf:.3f}",
            f"  Observed order (L2):  {self.observed_order_l2:.3f}",
            f"  Tolerance:            ±{self.tolerance}",
            f"  Verdict:              {'PASS' if self.passed else 'FAIL'}",
            "",
            f"  {'level':>6}  {kind:>12}  {'L∞ err':>12}  {'L2 err':>12}  "
            f"{'steps':>8}  {'wall_s':>8}",
        ]
        for p in self.points:
            lines.append(
                f"  {p.level:>6}  {p.parameter:>12.4e}  {p.linf_error:>12.4e}  "
                f"{p.l2_error:>12.4e}  {p.steps:>8}  {p.wall_time_s:>8.3f}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Grid Refinement Study
# ═══════════════════════════════════════════════════════════════════════════════


ExactSolution = Callable[[Tensor, float], Tensor]
"""exact_solution(x, t) → Tensor of shape (N,)"""

SolverSetupFactory = Callable[
    [StructuredMesh],
    Tuple[TimeIntegrator, RHSCallable, float],
]
"""
Factory: (mesh,) → (integrator, rhs, dt)
"""

ICFactory = Callable[[StructuredMesh], SimulationState]
"""
Factory: (mesh,) → initial SimulationState
"""


class RefinementStudy:
    """
    Configurable refinement study runner.

    This class stores the study configuration and can run grid or timestep
    refinement studies repeatedly with the same setup.
    """

    def __init__(
        self,
        exact_solution: ExactSolution,
        ic_factory: ICFactory,
        solver_factory: SolverSetupFactory,
        domain: Tuple[Tuple[float, float], ...] = ((0.0, 1.0),),
        field_name: str = "u",
    ) -> None:
        self._exact = exact_solution
        self._ic_factory = ic_factory
        self._solver_factory = solver_factory
        self._domain = domain
        self._field_name = field_name

    def run_grid_study(
        self,
        resolutions: Sequence[int],
        t_final: float,
        formal_order: int = 2,
        tolerance: float = 0.3,
    ) -> RefinementResult:
        """Run grid refinement (vary N, adapt dt to stay stable)."""
        return grid_refinement_study(
            exact_solution=self._exact,
            ic_factory=self._ic_factory,
            solver_factory=self._solver_factory,
            resolutions=resolutions,
            t_final=t_final,
            domain=self._domain,
            field_name=self._field_name,
            formal_order=formal_order,
            tolerance=tolerance,
        )

    def run_timestep_study(
        self,
        N: int,
        dt_values: Sequence[float],
        t_final: float,
        formal_order: int = 1,
        tolerance: float = 0.3,
    ) -> RefinementResult:
        """Run timestep refinement (vary dt, hold N fixed)."""
        return timestep_refinement_study(
            exact_solution=self._exact,
            ic_factory=self._ic_factory,
            solver_factory=self._solver_factory,
            N=N,
            dt_values=dt_values,
            t_final=t_final,
            domain=self._domain,
            field_name=self._field_name,
            formal_order=formal_order,
            tolerance=tolerance,
        )


def grid_refinement_study(
    exact_solution: ExactSolution,
    ic_factory: ICFactory,
    solver_factory: SolverSetupFactory,
    resolutions: Sequence[int],
    t_final: float,
    domain: Tuple[Tuple[float, float], ...] = ((0.0, 1.0),),
    field_name: str = "u",
    formal_order: int = 2,
    tolerance: float = 0.3,
) -> RefinementResult:
    """
    Run the solver at multiple grid resolutions and measure spatial error convergence.

    Parameters
    ----------
    exact_solution : callable(x, t) → Tensor
    ic_factory : callable(mesh) → SimulationState
    solver_factory : callable(mesh) → (integrator, rhs, dt)
    resolutions : sequence of int
        Grid sizes, monotonically increasing.
    t_final : float
    domain : tuple of (lo, hi) per dimension
    field_name : str
    formal_order : int
    tolerance : float
        Maximum deviation below *formal_order* that still counts as a pass.

    Returns
    -------
    RefinementResult
    """
    import time

    points: List[RefinementPoint] = []

    for idx, N in enumerate(resolutions):
        shape = (N,) if len(domain) == 1 else tuple([N] * len(domain))
        mesh = StructuredMesh(shape=shape, domain=domain)
        dx = mesh.dx[0]

        state0 = ic_factory(mesh)
        integrator, rhs_fn, dt = solver_factory(mesh)

        t0 = time.perf_counter()
        result = integrator.solve(state0, rhs_fn, t_span=(0.0, t_final), dt=dt)
        wall = time.perf_counter() - t0

        x = mesh.cell_centers()
        if mesh.ndim == 1:
            x = x.squeeze(-1)

        u_num = result.final_state.get_field(field_name).data
        u_ex = exact_solution(x, t_final)

        err = u_num - u_ex
        linf = err.abs().max().item()
        l1 = (err.abs() * dx).sum().item()
        l2 = math.sqrt((err ** 2 * dx).sum().item())

        points.append(RefinementPoint(
            level=idx, parameter=dx,
            linf_error=linf, l1_error=l1, l2_error=l2,
            steps=result.steps_taken, wall_time_s=wall,
        ))

    observed_linf, observed_l2 = _fit_orders(points)
    passed = (
        observed_linf >= formal_order - tolerance
        and observed_l2 >= formal_order - tolerance
    )

    return RefinementResult(
        study_type="grid",
        points=points,
        observed_order_linf=observed_linf,
        observed_order_l2=observed_l2,
        formal_order=formal_order,
        passed=passed,
        tolerance=tolerance,
    )


def timestep_refinement_study(
    exact_solution: ExactSolution,
    ic_factory: ICFactory,
    solver_factory: SolverSetupFactory,
    N: int,
    dt_values: Sequence[float],
    t_final: float,
    domain: Tuple[Tuple[float, float], ...] = ((0.0, 1.0),),
    field_name: str = "u",
    formal_order: int = 1,
    tolerance: float = 0.3,
) -> RefinementResult:
    """
    Run the solver at multiple timestep sizes on a fixed grid.

    The *solver_factory* is called once for the fixed mesh; the returned dt is
    **overridden** by each value in *dt_values*.
    """
    import time

    shape = (N,) if len(domain) == 1 else tuple([N] * len(domain))
    mesh = StructuredMesh(shape=shape, domain=domain)

    points: List[RefinementPoint] = []
    integrator, rhs_fn, _base_dt = solver_factory(mesh)

    for idx, dt in enumerate(sorted(dt_values, reverse=True)):
        state0 = ic_factory(mesh)

        t0 = time.perf_counter()
        result = integrator.solve(state0, rhs_fn, t_span=(0.0, t_final), dt=dt)
        wall = time.perf_counter() - t0

        x = mesh.cell_centers()
        if mesh.ndim == 1:
            x = x.squeeze(-1)

        u_num = result.final_state.get_field(field_name).data
        u_ex = exact_solution(x, t_final)

        err = u_num - u_ex
        linf = err.abs().max().item()
        l1 = (err.abs() * mesh.dx[0]).sum().item()
        l2 = math.sqrt((err ** 2 * mesh.dx[0]).sum().item())

        points.append(RefinementPoint(
            level=idx, parameter=dt,
            linf_error=linf, l1_error=l1, l2_error=l2,
            steps=result.steps_taken, wall_time_s=wall,
        ))

    observed_linf, observed_l2 = _fit_orders(points)
    passed = (
        observed_linf >= formal_order - tolerance
        and observed_l2 >= formal_order - tolerance
    )

    return RefinementResult(
        study_type="timestep",
        points=points,
        observed_order_linf=observed_linf,
        observed_order_l2=observed_l2,
        formal_order=formal_order,
        passed=passed,
        tolerance=tolerance,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Order computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_order(params: Sequence[float], errors: Sequence[float]) -> float:
    """
    Compute convergence order via least-squares log-log fit.

    Public API wrapper around the internal helper.
    """
    return _compute_order_impl(list(params), list(errors))


def _fit_orders(
    points: List[RefinementPoint],
) -> Tuple[float, float]:
    """Extract observed orders from a list of RefinementPoints."""
    if len(points) < 2:
        return 0.0, 0.0
    params = [p.parameter for p in points]
    linf = _compute_order_impl(params, [p.linf_error for p in points])
    l2 = _compute_order_impl(params, [p.l2_error for p in points])
    return linf, l2


def _compute_order_impl(params: List[float], errors: List[float]) -> float:
    """Least-squares slope of log(error) vs log(param)."""
    valid = [
        (math.log(p), math.log(e))
        for p, e in zip(params, errors)
        if p > 0 and e > 0
    ]
    if len(valid) < 2:
        return 0.0
    n = len(valid)
    sx = sum(v[0] for v in valid)
    sy = sum(v[1] for v in valid)
    sxx = sum(v[0] ** 2 for v in valid)
    sxy = sum(v[0] * v[1] for v in valid)
    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-30:
        return 0.0
    return (n * sxy - sx * sy) / denom
