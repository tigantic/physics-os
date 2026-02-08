"""
Solver Orchestration Layer
==========================

Time integrators, linear solvers, and nonlinear solvers that compose with the
canonical ``Solver`` protocol.

Time Integrators
----------------
All integrators accept a right-hand-side callable ``rhs(state, t) → dstate``
and produce the next state.

Linear Solvers
--------------
Krylov solvers (CG, GMRES) with a uniform ``solve(A, b, x0) → x`` signature.

Nonlinear Solvers
-----------------
Newton and Picard iterations wrapping an inner linear solver.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import torch
from torch import Tensor

from tensornet.platform.data_model import FieldData, Mesh, SimulationState
from tensornet.platform.protocols import Observable, SolveResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# RHS callable type
# ═══════════════════════════════════════════════════════════════════════════════

RHSCallable = Callable[[SimulationState, float], Dict[str, Tensor]]
"""
Right-hand-side function:  ``rhs(state, t) → {field_name: dU/dt}``
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TimeIntegrator ABC
# ═══════════════════════════════════════════════════════════════════════════════


class TimeIntegrator(ABC):
    """
    Advance a SimulationState by one time step using an explicit, implicit,
    or IMEX method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def order(self) -> int:
        """Formal order of accuracy in time."""
        ...

    @abstractmethod
    def step(
        self,
        state: SimulationState,
        rhs: RHSCallable,
        dt: float,
    ) -> SimulationState:
        """Advance *state* by one step of size *dt*."""
        ...

    def solve(
        self,
        state: SimulationState,
        rhs: RHSCallable,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Observable]] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """
        Integrate from ``t_span[0]`` to ``t_span[1]``.

        Collects observable history if *observables* is provided.
        """
        t0, tf = t_span
        if state.t != t0:
            state = SimulationState(
                t=t0,
                fields=state.fields,
                mesh=state.mesh,
                metadata=state.metadata,
                step_index=state.step_index,
            )

        obs_history: Dict[str, List[Tensor]] = {}
        if observables:
            for obs in observables:
                obs_history[obs.name] = []

        steps = 0
        limit = max_steps or int(1e9)

        while state.t < tf - 1e-14 * abs(dt) and steps < limit:
            actual_dt = min(dt, tf - state.t)
            state = self.step(state, rhs, actual_dt)
            steps += 1

            if observables:
                for obs in observables:
                    obs_history[obs.name].append(obs.compute(state))

        converged = state.t >= tf - 1e-14 * abs(dt)
        return SolveResult(
            final_state=state,
            t_final=state.t,
            steps_taken=steps,
            observable_history=obs_history,
            converged=converged,
            metadata={"integrator": self.name, "dt": dt},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Concrete time integrators
# ═══════════════════════════════════════════════════════════════════════════════


class ForwardEuler(TimeIntegrator):
    """First-order explicit Euler."""

    @property
    def name(self) -> str:
        return "ForwardEuler"

    @property
    def order(self) -> int:
        return 1

    def step(
        self,
        state: SimulationState,
        rhs: RHSCallable,
        dt: float,
    ) -> SimulationState:
        derivatives = rhs(state, state.t)
        new_fields: Dict[str, FieldData] = {}
        for fname, fdata in state.fields.items():
            if fname in derivatives:
                new_data = fdata.data + dt * derivatives[fname]
                new_fields[fname] = FieldData(
                    name=fname,
                    data=new_data,
                    mesh=state.mesh,
                    components=fdata.components,
                    units=fdata.units,
                )
            else:
                new_fields[fname] = fdata
        return state.advance(dt, new_fields)


class RK4(TimeIntegrator):
    """Classical fourth-order Runge-Kutta."""

    @property
    def name(self) -> str:
        return "RK4"

    @property
    def order(self) -> int:
        return 4

    def step(
        self,
        state: SimulationState,
        rhs: RHSCallable,
        dt: float,
    ) -> SimulationState:
        t = state.t

        # k1
        k1 = rhs(state, t)

        # k2 = rhs(state + 0.5*dt*k1, t + 0.5*dt)
        mid1_fields = _add_scaled(state, k1, 0.5 * dt)
        k2 = rhs(
            SimulationState(
                t=t + 0.5 * dt,
                fields=mid1_fields,
                mesh=state.mesh,
                step_index=state.step_index,
            ),
            t + 0.5 * dt,
        )

        # k3 = rhs(state + 0.5*dt*k2, t + 0.5*dt)
        mid2_fields = _add_scaled(state, k2, 0.5 * dt)
        k3 = rhs(
            SimulationState(
                t=t + 0.5 * dt,
                fields=mid2_fields,
                mesh=state.mesh,
                step_index=state.step_index,
            ),
            t + 0.5 * dt,
        )

        # k4 = rhs(state + dt*k3, t + dt)
        end_fields = _add_scaled(state, k3, dt)
        k4 = rhs(
            SimulationState(
                t=t + dt,
                fields=end_fields,
                mesh=state.mesh,
                step_index=state.step_index,
            ),
            t + dt,
        )

        # Combine: state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        new_fields: Dict[str, FieldData] = {}
        for fname, fdata in state.fields.items():
            if fname in k1:
                combo = (
                    k1[fname] + 2.0 * k2[fname] + 2.0 * k3[fname] + k4[fname]
                ) / 6.0
                new_fields[fname] = FieldData(
                    name=fname,
                    data=fdata.data + dt * combo,
                    mesh=state.mesh,
                    components=fdata.components,
                    units=fdata.units,
                )
            else:
                new_fields[fname] = fdata
        return state.advance(dt, new_fields)


class IMEX_Euler(TimeIntegrator):
    """
    First-order IMEX (Implicit-Explicit) Euler.

    Splits the RHS into explicit and implicit parts::

        U^{n+1} = U^n + dt * f_explicit(U^n) + dt * f_implicit(U^{n+1})

    The implicit part is resolved via a user-supplied implicit-solve callable.
    """

    def __init__(
        self,
        implicit_solve: Callable[
            [SimulationState, float, Dict[str, Tensor]], Dict[str, Tensor]
        ],
    ) -> None:
        """
        Parameters
        ----------
        implicit_solve
            ``(state_star, dt, explicit_rhs) → {field: data}`` that produces
            the fully-implicit update.
        """
        self._implicit_solve = implicit_solve

    @property
    def name(self) -> str:
        return "IMEX_Euler"

    @property
    def order(self) -> int:
        return 1

    def step(
        self,
        state: SimulationState,
        rhs: RHSCallable,
        dt: float,
    ) -> SimulationState:
        explicit_deriv = rhs(state, state.t)
        star_fields = _add_scaled(state, explicit_deriv, dt)
        star_state = SimulationState(
            t=state.t + dt,
            fields=star_fields,
            mesh=state.mesh,
            step_index=state.step_index,
        )
        implicit_fields = self._implicit_solve(star_state, dt, explicit_deriv)
        new_fields: Dict[str, FieldData] = {}
        for fname, fdata in state.fields.items():
            if fname in implicit_fields:
                new_fields[fname] = FieldData(
                    name=fname,
                    data=implicit_fields[fname],
                    mesh=state.mesh,
                    components=fdata.components,
                    units=fdata.units,
                )
            else:
                new_fields[fname] = fdata.clone() if fname not in star_fields else star_fields[fname]
        return state.advance(dt, new_fields)


class SymplecticEuler(TimeIntegrator):
    """
    First-order symplectic Euler for Hamiltonian systems.

    Expects fields ``'q'`` and ``'p'`` in the state.
    RHS must return ``{'q': dq/dt, 'p': dp/dt}``.
    """

    @property
    def name(self) -> str:
        return "SymplecticEuler"

    @property
    def order(self) -> int:
        return 1

    def step(
        self,
        state: SimulationState,
        rhs: RHSCallable,
        dt: float,
    ) -> SimulationState:
        q = state.get_field("q")
        p = state.get_field("p")

        # p half-step: use current q
        derivs = rhs(state, state.t)
        p_new_data = p.data + dt * derivs["p"]
        p_new = FieldData(name="p", data=p_new_data, mesh=state.mesh,
                          components=p.components, units=p.units)

        # q step: use new p
        temp_state = state.with_fields(p=p_new)
        derivs2 = rhs(temp_state, state.t + dt)
        q_new_data = q.data + dt * derivs2["q"]
        q_new = FieldData(name="q", data=q_new_data, mesh=state.mesh,
                          components=q.components, units=q.units)

        return state.advance(dt, {"q": q_new, "p": p_new})


class StormerVerlet(TimeIntegrator):
    """
    Second-order symplectic Störmer-Verlet (leapfrog).

    Expects fields ``'q'`` and ``'p'``.
    """

    @property
    def name(self) -> str:
        return "StormerVerlet"

    @property
    def order(self) -> int:
        return 2

    def step(
        self,
        state: SimulationState,
        rhs: RHSCallable,
        dt: float,
    ) -> SimulationState:
        q = state.get_field("q")
        p = state.get_field("p")

        # p half-step
        derivs0 = rhs(state, state.t)
        p_half_data = p.data + 0.5 * dt * derivs0["p"]
        p_half = FieldData(name="p", data=p_half_data, mesh=state.mesh,
                           components=p.components, units=p.units)

        # q full step using p_half
        temp1 = state.with_fields(p=p_half)
        derivs1 = rhs(temp1, state.t + 0.5 * dt)
        q_new_data = q.data + dt * derivs1["q"]
        q_new = FieldData(name="q", data=q_new_data, mesh=state.mesh,
                          components=q.components, units=q.units)

        # p full step using q_new
        temp2 = SimulationState(
            t=state.t + dt,
            fields={"q": q_new, "p": p_half},
            mesh=state.mesh,
            step_index=state.step_index,
        )
        derivs2 = rhs(temp2, state.t + dt)
        p_new_data = p_half_data + 0.5 * dt * derivs2["p"]
        p_new = FieldData(name="p", data=p_new_data, mesh=state.mesh,
                          components=p.components, units=p.units)

        return state.advance(dt, {"q": q_new, "p": p_new})


# ═══════════════════════════════════════════════════════════════════════════════
# Linear Solver Protocol + Implementations
# ═══════════════════════════════════════════════════════════════════════════════


class LinearSolverProto(Protocol):
    """
    Solve ``A x = b`` for x.
    """

    @property
    def name(self) -> str:
        ...

    def solve(
        self,
        matvec: Callable[[Tensor], Tensor],
        b: Tensor,
        x0: Optional[Tensor] = None,
        *,
        tol: float = 1e-8,
        max_iter: int = 1000,
    ) -> "LinearSolveResult":
        ...


@dataclass(frozen=True)
class LinearSolveResult:
    x: Tensor
    converged: bool
    iterations: int
    residual_norm: float


class ConjugateGradient:
    """CG for symmetric positive-definite systems."""

    @property
    def name(self) -> str:
        return "CG"

    def solve(
        self,
        matvec: Callable[[Tensor], Tensor],
        b: Tensor,
        x0: Optional[Tensor] = None,
        *,
        tol: float = 1e-8,
        max_iter: int = 1000,
        preconditioner: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> LinearSolveResult:
        x = x0 if x0 is not None else torch.zeros_like(b)
        r = b - matvec(x)
        if preconditioner:
            z = preconditioner(r)
        else:
            z = r.clone()
        p = z.clone()
        rz = torch.dot(r.flatten(), z.flatten())
        b_norm = torch.norm(b).item()
        if b_norm < 1e-30:
            b_norm = 1.0

        for k in range(max_iter):
            Ap = matvec(p)
            pAp = torch.dot(p.flatten(), Ap.flatten())
            if abs(pAp.item()) < 1e-30:
                break
            alpha = rz / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            res_norm = torch.norm(r).item()
            if res_norm / b_norm < tol:
                return LinearSolveResult(x=x, converged=True,
                                         iterations=k + 1,
                                         residual_norm=res_norm)
            if preconditioner:
                z = preconditioner(r)
            else:
                z = r.clone()
            rz_new = torch.dot(r.flatten(), z.flatten())
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new

        return LinearSolveResult(
            x=x, converged=False, iterations=max_iter,
            residual_norm=torch.norm(r).item(),
        )


class GMRES:
    """Restarted GMRES for general (non-symmetric) systems."""

    def __init__(self, restart: int = 30) -> None:
        self._restart = restart

    @property
    def name(self) -> str:
        return f"GMRES({self._restart})"

    def solve(
        self,
        matvec: Callable[[Tensor], Tensor],
        b: Tensor,
        x0: Optional[Tensor] = None,
        *,
        tol: float = 1e-8,
        max_iter: int = 1000,
        preconditioner: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> LinearSolveResult:
        n = b.numel()
        x = x0.flatten() if x0 is not None else torch.zeros(n, dtype=b.dtype, device=b.device)
        b_flat = b.flatten()
        b_norm = torch.norm(b_flat).item()
        if b_norm < 1e-30:
            b_norm = 1.0

        total_iters = 0
        m = self._restart

        for _outer in range(max_iter // max(m, 1) + 1):
            r = b_flat - matvec(x.view_as(b)).flatten()
            if preconditioner:
                r = preconditioner(r.view_as(b)).flatten()
            beta = torch.norm(r).item()
            if beta / b_norm < tol:
                return LinearSolveResult(
                    x=x.view_as(b), converged=True,
                    iterations=total_iters, residual_norm=beta,
                )

            V = torch.zeros(n, m + 1, dtype=b.dtype, device=b.device)
            H = torch.zeros(m + 1, m, dtype=b.dtype, device=b.device)
            V[:, 0] = r / beta

            g = torch.zeros(m + 1, dtype=b.dtype, device=b.device)
            g[0] = beta

            cs = torch.zeros(m, dtype=b.dtype, device=b.device)
            sn = torch.zeros(m, dtype=b.dtype, device=b.device)

            j = 0
            for j in range(min(m, max_iter - total_iters)):
                w = matvec(V[:, j].view_as(b)).flatten()
                if preconditioner:
                    w = preconditioner(w.view_as(b)).flatten()

                for i in range(j + 1):
                    H[i, j] = torch.dot(V[:, i], w)
                    w = w - H[i, j] * V[:, i]

                H[j + 1, j] = torch.norm(w)
                if H[j + 1, j].item() > 1e-30:
                    V[:, j + 1] = w / H[j + 1, j]

                # Apply previous Givens rotations
                for i in range(j):
                    tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                    H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                    H[i, j] = tmp

                # Compute new Givens rotation
                denom = math.sqrt(H[j, j].item() ** 2 + H[j + 1, j].item() ** 2)
                if denom < 1e-30:
                    cs[j] = 1.0
                    sn[j] = 0.0
                else:
                    cs[j] = H[j, j] / denom
                    sn[j] = H[j + 1, j] / denom

                g[j + 1] = -sn[j] * g[j]
                g[j] = cs[j] * g[j]
                H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
                H[j + 1, j] = 0.0

                total_iters += 1
                res = abs(g[j + 1].item())
                if res / b_norm < tol:
                    # Back-solve
                    y = torch.linalg.solve_triangular(
                        H[: j + 1, : j + 1], g[: j + 1].unsqueeze(1), upper=True
                    ).squeeze(1)
                    x = x + V[:, : j + 1] @ y
                    return LinearSolveResult(
                        x=x.view_as(b), converged=True,
                        iterations=total_iters, residual_norm=res,
                    )

            # End of restart cycle — back-solve and update x
            k = j + 1
            y = torch.linalg.solve_triangular(
                H[:k, :k], g[:k].unsqueeze(1), upper=True
            ).squeeze(1)
            x = x + V[:, :k] @ y

        res_final = torch.norm(b_flat - matvec(x.view_as(b)).flatten()).item()
        return LinearSolveResult(
            x=x.view_as(b), converged=False,
            iterations=total_iters, residual_norm=res_final,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Nonlinear Solver Protocol + Implementations
# ═══════════════════════════════════════════════════════════════════════════════


class NonlinearSolverProto(Protocol):
    """
    Solve ``F(x) = 0`` given an initial guess.
    """

    @property
    def name(self) -> str:
        ...

    def solve(
        self,
        residual: Callable[[Tensor], Tensor],
        x0: Tensor,
        *,
        tol: float = 1e-8,
        max_iter: int = 50,
    ) -> "NonlinearSolveResult":
        ...


@dataclass(frozen=True)
class NonlinearSolveResult:
    x: Tensor
    converged: bool
    iterations: int
    residual_norm: float


class NewtonSolver:
    """
    Newton-Raphson with finite-difference Jacobian and inner linear solver.
    """

    def __init__(
        self,
        linear_solver: Optional[Any] = None,
        fd_eps: float = 1e-7,
    ) -> None:
        self._linsolver = linear_solver or ConjugateGradient()
        self._fd_eps = fd_eps

    @property
    def name(self) -> str:
        return "Newton"

    def solve(
        self,
        residual: Callable[[Tensor], Tensor],
        x0: Tensor,
        *,
        tol: float = 1e-8,
        max_iter: int = 50,
    ) -> NonlinearSolveResult:
        x = x0.clone()
        for k in range(max_iter):
            F = residual(x)
            fn = torch.norm(F).item()
            if fn < tol:
                return NonlinearSolveResult(
                    x=x, converged=True, iterations=k, residual_norm=fn
                )

            # Jacobian-vector product via finite differences
            def jvp(v: Tensor) -> Tensor:
                return (residual(x + self._fd_eps * v) - F) / self._fd_eps

            result = self._linsolver.solve(jvp, -F, tol=max(tol * 0.1, 1e-12))
            x = x + result.x

        fn_final = torch.norm(residual(x)).item()
        return NonlinearSolveResult(
            x=x, converged=False, iterations=max_iter,
            residual_norm=fn_final,
        )


class PicardSolver:
    """Fixed-point (Picard) iteration: ``x_{k+1} = G(x_k)``."""

    @property
    def name(self) -> str:
        return "Picard"

    def solve(
        self,
        fixed_point_map: Callable[[Tensor], Tensor],
        x0: Tensor,
        *,
        tol: float = 1e-8,
        max_iter: int = 200,
        relaxation: float = 1.0,
    ) -> NonlinearSolveResult:
        x = x0.clone()
        for k in range(max_iter):
            x_new = fixed_point_map(x)
            if relaxation != 1.0:
                x_new = relaxation * x_new + (1.0 - relaxation) * x
            diff = torch.norm(x_new - x).item()
            x = x_new
            if diff < tol:
                return NonlinearSolveResult(
                    x=x, converged=True, iterations=k + 1,
                    residual_norm=diff,
                )
        return NonlinearSolveResult(
            x=x, converged=False, iterations=max_iter,
            residual_norm=torch.norm(fixed_point_map(x) - x).item(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _add_scaled(
    state: SimulationState,
    derivatives: Dict[str, Tensor],
    scale: float,
) -> Dict[str, FieldData]:
    """Return ``{fname: FieldData(data + scale * deriv)}``."""
    result: Dict[str, FieldData] = {}
    for fname, fdata in state.fields.items():
        if fname in derivatives:
            result[fname] = FieldData(
                name=fname,
                data=fdata.data + scale * derivatives[fname],
                mesh=state.mesh,
                components=fdata.components,
                units=fdata.units,
            )
        else:
            result[fname] = fdata
    return result
