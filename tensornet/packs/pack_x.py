"""
Domain Pack X — Particle / High-Energy Physics
================================================

Scaffold nodes (V0.1):
  PHY-X.1  QCD
  PHY-X.2  Electroweak theory
  PHY-X.3  Beyond Standard Model (BSM)
  PHY-X.4  Lattice QCD
  PHY-X.5  Parton distributions
  PHY-X.6  Collider simulation
  PHY-X.7  Dark matter
  PHY-X.8  Neutrino physics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Type

from tensornet.platform.domain_pack import DomainPack, get_registry
from tensornet.platform.protocols import (
    Discretization,
    Observable,
    ProblemSpec,
    Solver,
    SolveResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Scaffold factories
# ═══════════════════════════════════════════════════════════════════════════════

_SCAFFOLD_NODES: Dict[str, str] = {
    "PHY-X.1": "QCD",
    "PHY-X.2": "Electroweak theory",
    "PHY-X.3": "Beyond Standard Model",
    "PHY-X.4": "Lattice QCD",
    "PHY-X.5": "Parton distributions",
    "PHY-X.6": "Collider simulation",
    "PHY-X.7": "Dark matter",
    "PHY-X.8": "Neutrino physics",
}


def _make_scaffold_spec(node_id: str, label: str) -> type:
    """Dynamically create a frozen-dataclass ProblemSpec for a scaffold node."""

    @dataclass(frozen=True)
    class _Spec:
        __doc__ = f"{node_id}: {label} (scaffold V0.1)"

        @property
        def name(self) -> str:
            return f"{node_id}_{label.replace(' ', '_')}"

        @property
        def ndim(self) -> int:
            return 1

        @property
        def parameters(self) -> Dict[str, Any]:
            return {"node": node_id}

        @property
        def governing_equations(self) -> str:
            return f"{label} — V0.1 scaffold, pending implementation"

        @property
        def field_names(self) -> Sequence[str]:
            return ("state",)

        @property
        def observable_names(self) -> Sequence[str]:
            return ("energy",)

    _Spec.__name__ = _Spec.__qualname__ = f"Spec_{node_id.replace('-', '_').replace('.', '_')}"
    return _Spec


class _ScaffoldSolver:
    """Minimal no-op solver satisfying the Solver protocol (V0.1 scaffold)."""

    def __init__(self, solver_name: str) -> None:
        self._name = solver_name

    @property
    def name(self) -> str:
        return self._name

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        return state

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        return SolveResult(
            final_state=state,
            t_final=t_span[1],
            steps_taken=0,
            metadata={"scaffold": True},
        )


# Build spec and solver tables
_SPECS: Dict[str, type] = {nid: _make_scaffold_spec(nid, lbl) for nid, lbl in _SCAFFOLD_NODES.items()}
_SOLVERS: Dict[str, type] = {}
for _nid, _lbl in _SCAFFOLD_NODES.items():
    _solver_cls = type(
        f"Solver_{_nid.replace('-', '_').replace('.', '_')}",
        (_ScaffoldSolver,),
        {},
    )
    _solver_cls.__init__ = lambda self, _n=f"{_lbl}_scaffold": _ScaffoldSolver.__init__(self, _n)  # type: ignore[assignment]
    _SOLVERS[_nid] = _solver_cls


# ═══════════════════════════════════════════════════════════════════════════════
# Pack registration
# ═══════════════════════════════════════════════════════════════════════════════


class ParticleHighEnergyPack(DomainPack):
    """Pack X: Particle / High-Energy Physics."""

    @property
    def pack_id(self) -> str:
        return "X"

    @property
    def pack_name(self) -> str:
        return "Particle / High-Energy Physics"

    @property
    def taxonomy_ids(self) -> Sequence[str]:
        return tuple(_SCAFFOLD_NODES.keys())

    def problem_specs(self) -> Dict[str, Type[ProblemSpec]]:
        return dict(_SPECS)  # type: ignore[arg-type]

    def solvers(self) -> Dict[str, Type[Solver]]:
        return dict(_SOLVERS)  # type: ignore[arg-type]

    def discretizations(self) -> Dict[str, Sequence[Type[Discretization]]]:
        return {}

    def observables(self) -> Dict[str, Sequence[Type[Observable]]]:
        return {}

    @property
    def version(self) -> str:
        return "0.1.0"


get_registry().register_pack(ParticleHighEnergyPack())
