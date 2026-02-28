"""
Method of Manufactured Solutions (MMS) Framework
=================================================

MMS is the gold-standard technique for verifying that a PDE solver correctly
implements the intended equations.  The procedure:

1. **Choose** an arbitrary smooth analytical solution  u_exact(x, t).
2. **Substitute** into the governing PDE to compute the required source term  S(x, t).
3. **Solve** the PDE with the source term using the numerical method.
4. **Measure** the error  ||u_numerical − u_exact||  under grid/timestep refinement.
5. **Verify** that the observed convergence order matches the formal order.

This module provides:

- ``ManufacturedSolution``: container for exact solution + source term + BCs.
- ``MMSProblem``: wraps a manufactured solution into a ``ProblemSpec``-compatible
  form with the source term injected into the RHS.
- ``mms_convergence_study``: runs the solver at multiple resolutions and returns
  observed convergence rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from torch import Tensor

from ontic.platform.data_model import FieldData, SimulationState, StructuredMesh


# ═══════════════════════════════════════════════════════════════════════════════
# ManufacturedSolution
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ManufacturedSolution:
    """
    A manufactured analytical solution and its associated source term.

    Parameters
    ----------
    exact : callable(x: Tensor, t: float, **params) → Tensor
        The manufactured exact solution at spatial coordinates *x* and time *t*.
    source : callable(x: Tensor, t: float, **params) → Tensor
        The source term obtained by substituting *exact* into the PDE.
    bc_left : callable(t: float) → float or Tensor, optional
        Left boundary value (for 1-D structured grids).
    bc_right : callable(t: float) → float or Tensor, optional
        Right boundary value (for 1-D structured grids).
    name : str
        Human-readable label.
    params : dict
        Physical parameters passed to *exact* and *source*.
    """

    exact: Callable[..., Tensor]
    source: Callable[..., Tensor]
    bc_left: Optional[Callable[[float], Any]] = None
    bc_right: Optional[Callable[[float], Any]] = None
    name: str = "MMS"
    params: Dict[str, Any] = dc_field(default_factory=dict)

    def exact_field(
        self, mesh: StructuredMesh, t: float, field_name: str = "u"
    ) -> FieldData:
        """Evaluate exact solution on *mesh* at time *t*."""
        x = mesh.cell_centers()
        if mesh.ndim == 1:
            x = x.squeeze(-1)
        data = self.exact(x, t, **self.params)
        return FieldData(name=field_name, data=data, mesh=mesh)

    def source_field(
        self, mesh: StructuredMesh, t: float, field_name: str = "source"
    ) -> FieldData:
        """Evaluate source term on *mesh* at time *t*."""
        x = mesh.cell_centers()
        if mesh.ndim == 1:
            x = x.squeeze(-1)
        data = self.source(x, t, **self.params)
        return FieldData(name=field_name, data=data, mesh=mesh)


# ═══════════════════════════════════════════════════════════════════════════════
# MMSProblem — adaptor that injects source term into solver RHS
# ═══════════════════════════════════════════════════════════════════════════════


class MMSProblem:
    """
    Wraps a base RHS function to add the MMS source term.

    The source term from the manufactured solution is evaluated at the current
    time and added to every RHS evaluation, making it transparent to the solver.

    Usage::

        base_rhs = ops.rhs  # from your discretization
        mms_rhs = MMSProblem(ms, mesh, base_rhs).rhs
        result = integrator.solve(state, mms_rhs, ...)
    """

    def __init__(
        self,
        ms: ManufacturedSolution,
        mesh: StructuredMesh,
        base_rhs: Callable[[SimulationState, float], Dict[str, Tensor]],
        field_name: str = "u",
    ) -> None:
        self._ms = ms
        self._mesh = mesh
        self._base_rhs = base_rhs
        self._field_name = field_name

    def rhs(
        self, state: SimulationState, t: float
    ) -> Dict[str, Tensor]:
        """Base RHS + MMS source term."""
        derivs = self._base_rhs(state, t)
        src = self._ms.source_field(self._mesh, t, self._field_name)
        if self._field_name in derivs:
            derivs[self._field_name] = derivs[self._field_name] + src.data
        else:
            derivs[self._field_name] = src.data
        return derivs


# ═══════════════════════════════════════════════════════════════════════════════
# MMS Convergence Study
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConvergencePoint:
    """Error metrics at a single resolution."""

    N: int
    dx: float
    dt: float
    linf_error: float
    l1_error: float
    l2_error: float
    steps: int


@dataclass(frozen=True)
class MMSConvergenceResult:
    """Full MMS convergence study result."""

    points: List[ConvergencePoint]
    observed_order_linf: float
    observed_order_l2: float
    formal_order: int
    passed: bool
    tolerance: float

    def summary(self) -> str:
        lines = [
            "MMS Convergence Study",
            "=" * 60,
            f"  Formal order:         {self.formal_order}",
            f"  Observed order (L∞):  {self.observed_order_linf:.2f}",
            f"  Observed order (L2):  {self.observed_order_l2:.2f}",
            f"  Tolerance:            {self.tolerance}",
            f"  Verdict:              {'PASS' if self.passed else 'FAIL'}",
            "",
            f"  {'N':>6}  {'dx':>12}  {'L∞ err':>12}  {'L2 err':>12}  {'steps':>8}",
        ]
        for p in self.points:
            lines.append(
                f"  {p.N:>6}  {p.dx:>12.4e}  {p.linf_error:>12.4e}  "
                f"{p.l2_error:>12.4e}  {p.steps:>8}"
            )
        return "\n".join(lines)


SolverFactory = Callable[
    [StructuredMesh, ManufacturedSolution],
    Tuple[
        Any,  # integrator (TimeIntegrator)
        Callable[[SimulationState, float], Dict[str, Tensor]],  # rhs
        float,  # dt
    ],
]
"""
Factory:  (mesh, ms) → (integrator, rhs_with_mms_source, dt)

The factory is responsible for:
- Discretizing on the given mesh
- Wrapping via MMSProblem to inject the source term
- Choosing a stable dt (e.g. CFL-based)
"""


def mms_convergence_study(
    ms: ManufacturedSolution,
    solver_factory: SolverFactory,
    resolutions: Sequence[int],
    t_final: float,
    domain: Tuple[float, float] = (0.0, 1.0),
    formal_order: int = 2,
    tolerance: float = 0.3,
    field_name: str = "u",
) -> MMSConvergenceResult:
    """
    Run the solver at multiple resolutions and measure convergence.

    Parameters
    ----------
    ms : ManufacturedSolution
        Manufactured solution with exact + source.
    solver_factory : callable
        ``(mesh, ms) → (integrator, rhs, dt)``
    resolutions : sequence of int
        Grid sizes (e.g. ``[32, 64, 128, 256]``).
    t_final : float
        End time.
    domain : tuple
        1-D spatial domain.
    formal_order : int
        Expected spatial convergence order.
    tolerance : float
        How much the observed order can deviate from *formal_order* and still
        pass.  E.g. ``0.3`` means order ≥ formal − 0.3.
    field_name : str
        Which field to measure error on.

    Returns
    -------
    MMSConvergenceResult
    """
    points: List[ConvergencePoint] = []

    for N in resolutions:
        mesh = StructuredMesh(shape=(N,), domain=(domain,))
        dx = mesh.dx[0]
        x = mesh.cell_centers().squeeze(-1)

        # Initial condition from exact solution
        u0_data = ms.exact(x, 0.0, **ms.params)
        u0 = FieldData(name=field_name, data=u0_data, mesh=mesh)
        state0 = SimulationState(t=0.0, fields={field_name: u0}, mesh=mesh)

        integrator, rhs_fn, dt = solver_factory(mesh, ms)
        result = integrator.solve(state0, rhs_fn, t_span=(0.0, t_final), dt=dt)

        u_num = result.final_state.get_field(field_name).data
        u_exact = ms.exact(x, t_final, **ms.params)

        err = u_num - u_exact
        linf = err.abs().max().item()
        l1 = (err.abs() * dx).sum().item()
        l2 = math.sqrt((err ** 2 * dx).sum().item())

        points.append(ConvergencePoint(
            N=N, dx=dx, dt=dt,
            linf_error=linf, l1_error=l1, l2_error=l2,
            steps=result.steps_taken,
        ))

    # Compute observed orders via least-squares fit of log(error) vs log(dx)
    if len(points) >= 2:
        observed_linf = _compute_order(
            [p.dx for p in points], [p.linf_error for p in points]
        )
        observed_l2 = _compute_order(
            [p.dx for p in points], [p.l2_error for p in points]
        )
    else:
        observed_linf = 0.0
        observed_l2 = 0.0

    passed = (
        observed_linf >= formal_order - tolerance
        and observed_l2 >= formal_order - tolerance
    )

    return MMSConvergenceResult(
        points=points,
        observed_order_linf=observed_linf,
        observed_order_l2=observed_l2,
        formal_order=formal_order,
        passed=passed,
        tolerance=tolerance,
    )


def _compute_order(dxs: List[float], errors: List[float]) -> float:
    """Least-squares slope of log(error) vs log(dx)."""
    valid = [(math.log(dx), math.log(e)) for dx, e in zip(dxs, errors) if e > 0]
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
