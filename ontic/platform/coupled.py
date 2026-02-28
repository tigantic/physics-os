"""
Coupling Orchestrator — Multi-physics solver coupling.

Implements both monolithic and partitioned coupling strategies for
combining domain pack solvers into multi-physics workflows.

Classes:
    CoupledField        — A field shared between multiple solvers.
    CouplingInterface   — Defines how two solvers exchange data.
    MonolithicCoupler   — Advances all solvers simultaneously.
    PartitionedCoupler  — Gauss-Seidel or Jacobi sub-iteration coupling.
    CoupledWorkflow     — End-to-end coupled simulation pipeline.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ontic.platform.data_model import FieldData, Mesh, SimulationState
from ontic.platform.protocols import Observable, SolveResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Coupling Types
# ═══════════════════════════════════════════════════════════════════════════════


class CouplingStrategy(Enum):
    """Coupling iteration strategy."""
    MONOLITHIC = "monolithic"
    GAUSS_SEIDEL = "gauss_seidel"
    JACOBI = "jacobi"
    STRANG_SPLIT = "strang_split"


@dataclass(frozen=True)
class CoupledField:
    """
    A field that is shared or transferred between coupled solvers.

    Attributes
    ----------
    name : str
        Field name.
    source_solver : str
        Name of the solver that produces this field.
    target_solver : str
        Name of the solver that consumes this field.
    transfer_fn : Callable, optional
        Custom transfer/interpolation function.  Defaults to identity.
    relaxation : float
        Under-relaxation factor (0, 1] for iterative coupling.
    """

    name: str
    source_solver: str
    target_solver: str
    transfer_fn: Optional[Callable[[FieldData, Mesh], FieldData]] = None
    relaxation: float = 1.0


@dataclass(frozen=True)
class CouplingInterface:
    """
    Defines the data exchange contract between two solvers.

    Attributes
    ----------
    name : str
        Interface name.
    coupled_fields : tuple of CoupledField
        Fields exchanged through this interface.
    convergence_fields : tuple of str
        Field names used to check coupling convergence.
    tolerance : float
        Convergence tolerance for sub-iteration coupling.
    max_iterations : int
        Maximum sub-iterations per time step.
    """

    name: str
    coupled_fields: Tuple[CoupledField, ...]
    convergence_fields: Tuple[str, ...] = ()
    tolerance: float = 1e-6
    max_iterations: int = 20


# ═══════════════════════════════════════════════════════════════════════════════
# Coupling Result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CouplingStepResult:
    """Result from a single coupled time step."""

    solver_states: Dict[str, SimulationState]
    coupling_residuals: Dict[str, float]
    sub_iterations: int
    converged: bool
    elapsed_seconds: float


@dataclass
class CoupledSolveResult:
    """Result from a full coupled simulation."""

    final_states: Dict[str, SimulationState]
    t_final: float
    steps_taken: int
    converged: bool
    coupling_history: List[CouplingStepResult] = dc_field(default_factory=list)
    observable_history: Dict[str, List[Tensor]] = dc_field(default_factory=dict)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract Coupler
# ═══════════════════════════════════════════════════════════════════════════════


class CouplerBase(ABC):
    """
    Abstract base for multi-physics coupling orchestrators.
    """

    def __init__(
        self,
        solvers: Dict[str, Any],
        interface: CouplingInterface,
    ) -> None:
        self._solvers = solvers
        self._interface = interface

    @property
    def solver_names(self) -> List[str]:
        return list(self._solvers.keys())

    @abstractmethod
    def coupled_step(
        self,
        states: Dict[str, SimulationState],
        dt: float,
    ) -> CouplingStepResult:
        """Advance all coupled solvers by one time step."""
        ...

    def solve(
        self,
        initial_states: Dict[str, SimulationState],
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Dict[str, Sequence[Observable]]] = None,
        max_steps: Optional[int] = None,
    ) -> CoupledSolveResult:
        """
        Run the full coupled simulation.
        """
        current_states = {k: v.clone() for k, v in initial_states.items()}
        steps = 0
        coupling_history: List[CouplingStepResult] = []
        obs_history: Dict[str, List[Tensor]] = {}
        limit = max_steps or int(1e9)
        t_current = t_span[0]

        while t_current < t_span[1] - 1e-14 and steps < limit:
            h = min(dt, t_span[1] - t_current)
            result = self.coupled_step(current_states, h)
            current_states = result.solver_states
            coupling_history.append(result)
            t_current += h
            steps += 1

            if observables:
                for solver_name, obs_list in observables.items():
                    for obs in obs_list:
                        key = f"{solver_name}/{obs.name}"
                        if key not in obs_history:
                            obs_history[key] = []
                        obs_history[key].append(
                            obs.compute(current_states[solver_name])
                        )

        all_converged = all(r.converged for r in coupling_history)
        return CoupledSolveResult(
            final_states=current_states,
            t_final=t_current,
            steps_taken=steps,
            converged=all_converged,
            coupling_history=coupling_history,
            observable_history=obs_history,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Monolithic Coupler
# ═══════════════════════════════════════════════════════════════════════════════


class MonolithicCoupler(CouplerBase):
    """
    Monolithic coupling: advance all solvers in a single combined step.

    Each solver sees the latest fields from all other solvers.
    Fields are transferred through the coupling interface after each
    solver step within a single time step.
    """

    def coupled_step(
        self,
        states: Dict[str, SimulationState],
        dt: float,
    ) -> CouplingStepResult:
        t0 = time.perf_counter()
        current = {k: v.clone() for k, v in states.items()}

        # Advance each solver and transfer coupled fields
        for solver_name, solver in self._solvers.items():
            current[solver_name] = solver.step(current[solver_name], dt)

        # Transfer coupled fields
        residuals = self._transfer_and_measure(current)

        elapsed = time.perf_counter() - t0
        return CouplingStepResult(
            solver_states=current,
            coupling_residuals=residuals,
            sub_iterations=1,
            converged=True,
            elapsed_seconds=elapsed,
        )

    def _transfer_and_measure(
        self, states: Dict[str, SimulationState]
    ) -> Dict[str, float]:
        """Transfer coupled fields and measure residuals."""
        residuals: Dict[str, float] = {}
        for cf in self._interface.coupled_fields:
            source_state = states[cf.source_solver]
            target_state = states[cf.target_solver]

            source_field = source_state.get_field(cf.name)

            # Apply transfer function if any
            if cf.transfer_fn is not None:
                transferred = cf.transfer_fn(source_field, target_state.mesh)
            else:
                transferred = source_field

            # Under-relaxation
            if cf.relaxation < 1.0 and cf.name in target_state.fields:
                old_data = target_state.get_field(cf.name).data
                new_data = cf.relaxation * transferred.data + (1.0 - cf.relaxation) * old_data
                transferred = FieldData(
                    name=cf.name, data=new_data, mesh=transferred.mesh,
                    components=transferred.components, units=transferred.units,
                )

            # Measure residual
            if cf.name in target_state.fields:
                old = target_state.get_field(cf.name).data
                diff = (transferred.data - old).abs().max().item()
                residuals[cf.name] = diff

            # Inject into target state
            states[cf.target_solver] = target_state.with_fields(
                **{cf.name: transferred}
            )

        return residuals


# ═══════════════════════════════════════════════════════════════════════════════
# Partitioned (Gauss-Seidel) Coupler
# ═══════════════════════════════════════════════════════════════════════════════


class PartitionedCoupler(CouplerBase):
    """
    Partitioned coupling with sub-iteration convergence.

    Within each time step, solvers are advanced repeatedly in sequence
    (Gauss-Seidel) or in parallel (Jacobi) until coupling residuals
    converge below the interface tolerance.
    """

    def __init__(
        self,
        solvers: Dict[str, Any],
        interface: CouplingInterface,
        strategy: CouplingStrategy = CouplingStrategy.GAUSS_SEIDEL,
    ) -> None:
        super().__init__(solvers, interface)
        self._strategy = strategy

    def coupled_step(
        self,
        states: Dict[str, SimulationState],
        dt: float,
    ) -> CouplingStepResult:
        t0 = time.perf_counter()
        current = {k: v.clone() for k, v in states.items()}
        residuals: Dict[str, float] = {}
        converged = False

        for iteration in range(self._interface.max_iterations):
            previous = {k: v.clone() for k, v in current.items()}

            # Advance each solver
            for solver_name, solver in self._solvers.items():
                if self._strategy == CouplingStrategy.GAUSS_SEIDEL:
                    # Use latest available state
                    current[solver_name] = solver.step(current[solver_name], dt)
                else:
                    # Jacobi: use previous iteration states
                    current[solver_name] = solver.step(previous[solver_name], dt)

            # Transfer and measure
            residuals = self._transfer_and_measure(current, previous)

            # Check convergence
            if self._interface.convergence_fields:
                max_residual = max(
                    residuals.get(f, 0.0)
                    for f in self._interface.convergence_fields
                )
            else:
                max_residual = max(residuals.values()) if residuals else 0.0

            if max_residual < self._interface.tolerance:
                converged = True
                break

        elapsed = time.perf_counter() - t0
        return CouplingStepResult(
            solver_states=current,
            coupling_residuals=residuals,
            sub_iterations=iteration + 1,
            converged=converged,
            elapsed_seconds=elapsed,
        )

    def _transfer_and_measure(
        self,
        current: Dict[str, SimulationState],
        previous: Dict[str, SimulationState],
    ) -> Dict[str, float]:
        """Transfer fields and measure sub-iteration residuals."""
        residuals: Dict[str, float] = {}
        for cf in self._interface.coupled_fields:
            source_field = current[cf.source_solver].get_field(cf.name)

            if cf.transfer_fn is not None:
                transferred = cf.transfer_fn(source_field, current[cf.target_solver].mesh)
            else:
                transferred = source_field

            # Under-relaxation against previous iteration
            if cf.relaxation < 1.0 and cf.name in previous[cf.target_solver].fields:
                old_data = previous[cf.target_solver].get_field(cf.name).data
                new_data = cf.relaxation * transferred.data + (1.0 - cf.relaxation) * old_data
                transferred = FieldData(
                    name=cf.name, data=new_data, mesh=transferred.mesh,
                    components=transferred.components, units=transferred.units,
                )

            # Residual against previous iteration
            if cf.name in previous[cf.target_solver].fields:
                prev_data = previous[cf.target_solver].get_field(cf.name).data
                residuals[cf.name] = (transferred.data - prev_data).abs().max().item()

            current[cf.target_solver] = current[cf.target_solver].with_fields(
                **{cf.name: transferred}
            )

        return residuals


# ═══════════════════════════════════════════════════════════════════════════════
# N-Way Coupling Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CouplingEdge:
    """
    A directed coupling edge between two solvers.

    Attributes:
        source: Source solver name.
        target: Target solver name.
        fields: List of field names transferred source → target.
        transfer_fn: Optional interpolation/mapping callable.
        relaxation: Under-relaxation factor (0, 1].
        lag: Number of steps the coupling is lagged (0 = synchronous).
    """
    source: str
    target: str
    fields: List[str]
    transfer_fn: Optional[Callable] = None
    relaxation: float = 1.0
    lag: int = 0


class NWayCoupler:
    """
    N-way multi-physics coupling orchestrator.

    Extends the 2-way partitioned coupler to an arbitrary directed graph
    of solver couplings.  The coupling graph is defined as a set of
    :class:`CouplingEdge` objects.  At each coupled step:

    1. Solvers are topologically sorted (or iterated in registration order
       if cycles exist).
    2. Each solver is advanced one step.
    3. After each solver completes, its outgoing edges transfer data
       to the target solver states.
    4. Sub-iteration is applied until all edges converge below the
       tolerance.

    Supports:
        * **Synchronous coupling** (lag=0): transfers happen within the
          same time step.
        * **Lagged coupling** (lag≥1): the target receives data from
          *lag* steps ago (for explicit / loosely-coupled physics).
        * **Under-relaxation** per edge for stability.

    Example::

        coupler = NWayCoupler(
            solvers={"cfd": cfd_solver, "struct": fem_solver, "thermal": heat_solver},
            edges=[
                CouplingEdge(source="cfd", target="struct", fields=["pressure"]),
                CouplingEdge(source="struct", target="cfd", fields=["displacement"]),
                CouplingEdge(source="cfd", target="thermal", fields=["temperature"]),
            ],
            tolerance=1e-6,
            max_iterations=20,
        )
        result = coupler.solve(states, dt=0.01, n_steps=100)
    """

    def __init__(
        self,
        solvers: Dict[str, Any],
        edges: List[CouplingEdge],
        tolerance: float = 1e-6,
        max_iterations: int = 20,
    ) -> None:
        self._solvers = solvers
        self._edges = edges
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._history: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in solvers
        }

    @property
    def solver_names(self) -> List[str]:
        return list(self._solvers.keys())

    def _topological_order(self) -> List[str]:
        """Attempt topological sort of the coupling graph; fall back to insertion order."""
        from collections import deque

        in_degree: Dict[str, int] = {name: 0 for name in self._solvers}
        adjacency: Dict[str, List[str]] = {name: [] for name in self._solvers}

        for edge in self._edges:
            if edge.source in adjacency and edge.target in in_degree:
                adjacency[edge.source].append(edge.target)
                in_degree[edge.target] += 1

        queue: deque = deque(n for n, d in in_degree.items() if d == 0)
        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nbr in adjacency[node]:
                in_degree[nbr] -= 1
                if in_degree[nbr] == 0:
                    queue.append(nbr)

        if len(order) < len(self._solvers):
            # Cycle detected — use registration order
            logger.warning("Cycle detected in coupling graph; using registration order")
            return list(self._solvers.keys())
        return order

    def _transfer(
        self,
        edge: CouplingEdge,
        states: Dict[str, Any],
        previous: Dict[str, Any],
    ) -> float:
        """
        Transfer fields along one edge.  Returns max residual.
        """
        max_residual = 0.0
        source_state = states[edge.source]
        target_state = states[edge.target]

        for field_name in edge.fields:
            src_data = source_state.get_field(field_name) if hasattr(source_state, 'get_field') else getattr(source_state, field_name, None)
            if src_data is None:
                continue

            if edge.transfer_fn is not None:
                mesh = getattr(target_state, 'mesh', None)
                transferred = edge.transfer_fn(src_data, mesh)
            else:
                transferred = src_data

            # Under-relaxation
            if edge.relaxation < 1.0:
                prev_target = previous.get(edge.target)
                if prev_target is not None:
                    old = prev_target.get_field(field_name) if hasattr(prev_target, 'get_field') else getattr(prev_target, field_name, None)
                    if old is not None:
                        old_data = old.data if hasattr(old, 'data') else old
                        new_data = transferred.data if hasattr(transferred, 'data') else transferred
                        blended = edge.relaxation * new_data + (1.0 - edge.relaxation) * old_data
                        if hasattr(transferred, 'data'):
                            transferred = FieldData(
                                name=field_name, data=blended,
                                mesh=transferred.mesh,
                                components=transferred.components,
                                units=transferred.units,
                            )
                        else:
                            transferred = blended

            # Measure residual
            prev_target = previous.get(edge.target)
            if prev_target is not None:
                old = prev_target.get_field(field_name) if hasattr(prev_target, 'get_field') else getattr(prev_target, field_name, None)
                if old is not None:
                    old_d = old.data if hasattr(old, 'data') else old
                    new_d = transferred.data if hasattr(transferred, 'data') else transferred
                    if isinstance(old_d, Tensor):
                        res = float((new_d - old_d).abs().max().item())
                    else:
                        res = float(max(abs(new_d - old_d))) if hasattr(new_d, '__len__') else float(abs(new_d - old_d))
                    max_residual = max(max_residual, res)

            # Apply to target state
            if hasattr(target_state, 'with_fields'):
                states[edge.target] = target_state.with_fields(**{field_name: transferred})
            elif hasattr(target_state, field_name):
                setattr(states[edge.target], field_name, transferred)

        return max_residual

    def coupled_step(
        self,
        states: Dict[str, Any],
        dt: float,
    ) -> CouplingStepResult:
        """
        One N-way coupled time step with sub-iteration.

        Parameters:
            states: Dict of solver name → state.
            dt: Time step size.

        Returns:
            CouplingStepResult with updated states and diagnostics.
        """
        t0 = time.perf_counter()
        current = {k: v.clone() if hasattr(v, 'clone') else v for k, v in states.items()}
        order = self._topological_order()
        converged = False
        residuals: Dict[str, float] = {}

        for iteration in range(self._max_iterations):
            previous = {k: v.clone() if hasattr(v, 'clone') else v for k, v in current.items()}

            # Advance each solver in topological order
            for name in order:
                solver = self._solvers[name]
                current[name] = solver.step(current[name], dt)

            # Transfer along all edges
            max_res = 0.0
            for edge in self._edges:
                if edge.lag > 0:
                    # Lagged coupling: use history
                    hist = self._history.get(edge.source, [])
                    if len(hist) >= edge.lag:
                        lagged_state = hist[-edge.lag]
                        lagged_states = dict(current)
                        lagged_states[edge.source] = lagged_state
                        res = self._transfer(edge, lagged_states, previous)
                        current[edge.target] = lagged_states[edge.target]
                    else:
                        res = self._transfer(edge, current, previous)
                else:
                    res = self._transfer(edge, current, previous)
                residuals[f"{edge.source}->{edge.target}"] = res
                max_res = max(max_res, res)

            if max_res < self._tolerance:
                converged = True
                break

        # Store history
        for name in self._solvers:
            self._history[name].append(current[name])
            if len(self._history[name]) > 10:
                self._history[name].pop(0)

        elapsed = time.perf_counter() - t0
        return CouplingStepResult(
            solver_states=current,
            coupling_residuals=residuals,
            sub_iterations=iteration + 1,
            converged=converged,
            elapsed_seconds=elapsed,
        )

    def solve(
        self,
        initial_states: Dict[str, Any],
        dt: float,
        n_steps: int,
        callback: Optional[Callable] = None,
    ) -> CoupledSolveResult:
        """
        Run a full N-way coupled simulation.

        Parameters:
            initial_states: Initial state for each solver.
            dt: Time step.
            n_steps: Number of coupled steps.
            callback: Optional per-step callback(step, result).

        Returns:
            CoupledSolveResult with final states and history.
        """
        states = {k: v.clone() if hasattr(v, 'clone') else v for k, v in initial_states.items()}
        history: List[CouplingStepResult] = []

        for step in range(n_steps):
            result = self.coupled_step(states, dt)
            states = result.solver_states
            history.append(result)

            if callback is not None:
                callback(step, result)

            if not result.converged:
                logger.warning(
                    "N-way coupling did not converge at step %d "
                    "(max residual: %.2e)",
                    step,
                    max(result.coupling_residuals.values()) if result.coupling_residuals else float('nan'),
                )

        total_time = sum(h.elapsed_seconds for h in history)
        return CoupledSolveResult(
            final_states=states,
            step_history=history,
            total_steps=n_steps,
            total_elapsed_seconds=total_time,
        )
