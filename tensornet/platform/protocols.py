"""
Canonical Protocols — the six interfaces every domain pack must know about.

These are :pep:`544` ``Protocol`` classes, **not** ABC subclasses.  Domain code
that structurally conforms is accepted without explicit subclassing.

Protocols
---------
ProblemSpec     Describes *what* to solve (PDE, IC, BC, parameters, observables).
Discretization  Describes *how* a continuous problem becomes a discrete system.
OperatorProto   A callable that maps one FieldData → another.
Solver          Advances a SimulationState by one (or N) time-step(s).
Observable      Extracts a scalar / vector diagnostic from a SimulationState.
Workflow        End-to-end pipeline: spec → discretize → solve → observe → store.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from torch import Tensor


# ---------------------------------------------------------------------------
# ProblemSpec
# ---------------------------------------------------------------------------

@runtime_checkable
class ProblemSpec(Protocol):
    """
    Declares the continuous problem: governing equations, domain geometry,
    boundary conditions, initial conditions, physical parameters, and the
    observables to compute.
    """

    @property
    def name(self) -> str:
        """Human-readable problem name."""
        ...

    @property
    def ndim(self) -> int:
        """Spatial dimension (1, 2, or 3)."""
        ...

    @property
    def parameters(self) -> Dict[str, Any]:
        """Physical parameters (Reynolds number, viscosity, …)."""
        ...

    @property
    def governing_equations(self) -> str:
        """
        Symbolic or LaTeX string describing the PDE/ODE system.
        Used for documentation and registry display.
        """
        ...

    @property
    def field_names(self) -> Sequence[str]:
        """
        Ordered names of the unknown fields
        (e.g. ``['velocity', 'pressure']``).
        """
        ...

    @property
    def observable_names(self) -> Sequence[str]:
        """Names of observables that should be computed each step."""
        ...


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------

@runtime_checkable
class Discretization(Protocol):
    """
    Maps a continuous ProblemSpec onto a discrete algebraic system.

    Implementations include finite-volume, finite-element, spectral,
    finite-difference, DG, particle, QTT, …
    """

    @property
    def method(self) -> str:
        """Short tag — ``'FVM'``, ``'FEM'``, ``'spectral'``, etc."""
        ...

    @property
    def order(self) -> int:
        """Formal order of accuracy."""
        ...

    def discretize(
        self,
        spec: ProblemSpec,
        mesh: Any,
    ) -> Any:
        """
        Return the discrete system (operators, RHS assembler, mass matrix, …).
        The concrete return type is method-dependent; the Solver must know how
        to consume it.
        """
        ...


# ---------------------------------------------------------------------------
# OperatorProto
# ---------------------------------------------------------------------------

@runtime_checkable
class OperatorProto(Protocol):
    """
    A discrete operator that maps field data → field data.

    This is the *platform-level* Operator protocol.  It is intentionally
    thin so that both ``tensornet.types.Operator`` and
    ``tensornet.fieldops.Operator`` can satisfy it.
    """

    @property
    def name(self) -> str:
        ...

    def apply(self, field: Tensor, **kwargs: Any) -> Tensor:
        """Apply the operator to a dense or QTT tensor."""
        ...


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@runtime_checkable
class Solver(Protocol):
    """
    Advances a simulation state forward in time (or iterates to steady state).

    Every concrete solver in every domain pack must satisfy this protocol.
    """

    @property
    def name(self) -> str:
        ...

    def step(
        self,
        state: Any,
        dt: float,
        **kwargs: Any,
    ) -> Any:
        """
        Advance *state* by one time increment *dt* and return new state.

        *state* is a ``SimulationState`` (or duck-typed equivalent).
        """
        ...

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence["Observable"]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> "SolveResult":
        """
        Integrate from ``t_span[0]`` to ``t_span[1]``.

        Returns a ``SolveResult`` containing the final state, trajectory of
        observables, and run metadata.
        """
        ...


# ---------------------------------------------------------------------------
# Observable
# ---------------------------------------------------------------------------

@runtime_checkable
class Observable(Protocol):
    """
    Computes a scalar, vector, or small tensor diagnostic from the current
    simulation state.
    """

    @property
    def name(self) -> str:
        ...

    @property
    def units(self) -> str:
        """SI-string, e.g. ``'m/s'``, ``'J'``, ``'1'``."""
        ...

    def compute(self, state: Any) -> Tensor:
        """Return the observable value(s) for the given state."""
        ...


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@runtime_checkable
class Workflow(Protocol):
    """
    End-to-end pipeline: define → discretize → solve → observe → store.
    """

    @property
    def name(self) -> str:
        ...

    def run(self, **overrides: Any) -> "WorkflowResult":
        """
        Execute the full pipeline, returning a ``WorkflowResult`` that
        bundles the final state, observable history, and provenance metadata.
        """
        ...


# ---------------------------------------------------------------------------
# Result containers (concrete, not protocols)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as dc_field


@dataclass(frozen=True)
class SolveResult:
    """Returned by ``Solver.solve``."""

    final_state: Any
    t_final: float
    steps_taken: int
    observable_history: Dict[str, List[Tensor]] = dc_field(default_factory=dict)
    converged: bool = True
    metadata: Dict[str, Any] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowResult:
    """Returned by ``Workflow.run``."""

    solve_result: SolveResult
    provenance: Dict[str, Any] = dc_field(default_factory=dict)
    artifacts: Dict[str, Any] = dc_field(default_factory=dict)
