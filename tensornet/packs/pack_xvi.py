"""
Domain Pack XVI — Materials Science
=====================================

Scaffold nodes (V0.1):
  PHY-XVI.1  Crystal growth
  PHY-XVI.2  Fracture mechanics
  PHY-XVI.3  Corrosion
  PHY-XVI.4  Thin films
  PHY-XVI.5  Nanostructures
  PHY-XVI.6  Composites
  PHY-XVI.7  Metamaterials
  PHY-XVI.8  Phase-field modeling
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
    "PHY-XVI.1": "Crystal growth",
    "PHY-XVI.2": "Fracture mechanics",
    "PHY-XVI.3": "Corrosion",
    "PHY-XVI.4": "Thin films",
    "PHY-XVI.5": "Nanostructures",
    "PHY-XVI.6": "Composites",
    "PHY-XVI.7": "Metamaterials",
    "PHY-XVI.8": "Phase-field modeling",
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
            return f"{label} — scaffold placeholder"

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


class MaterialsSciencePack(DomainPack):
    """Pack XVI: Materials Science."""

    @property
    def pack_id(self) -> str:
        return "XVI"

    @property
    def pack_name(self) -> str:
        return "Materials Science"

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


get_registry().register_pack(MaterialsSciencePack())
