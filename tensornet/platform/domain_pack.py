"""
DomainPack — the plugin unit for physics domains.

A DomainPack bundles equations + discretizations + solvers + benchmarks + docs
for one taxonomy category (e.g. Pack II: Fluid Dynamics).

Registry / discovery
--------------------
``DomainRegistry`` is a singleton in-process registry.  Domain packs register
themselves at import time via ``@DomainRegistry.register`` or by calling
``get_registry().register_pack(pack)``.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
)

from tensornet.platform.protocols import (
    Discretization,
    Observable,
    ProblemSpec,
    Solver,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DomainPack ABC
# ═══════════════════════════════════════════════════════════════════════════════


class DomainPack(ABC):
    """
    A cohesive bundle: equations + discretizations + solvers + benchmarks + docs.

    One DomainPack per taxonomy category (I–XX), though a pack may contain
    multiple sub-nodes (e.g. PHY-II.1 through PHY-II.10).
    """

    # ──────── identity ────────

    @property
    @abstractmethod
    def pack_id(self) -> str:
        """Short slug, e.g. ``'II'``, ``'VII'``."""
        ...

    @property
    @abstractmethod
    def pack_name(self) -> str:
        """Human-readable name, e.g. ``'Fluid Dynamics'``."""
        ...

    @property
    @abstractmethod
    def taxonomy_ids(self) -> Sequence[str]:
        """
        Ledger node IDs covered by this pack,
        e.g. ``['PHY-II.1', 'PHY-II.2', …, 'PHY-II.10']``.
        """
        ...

    # ──────── catalogue ────────

    @abstractmethod
    def problem_specs(self) -> Dict[str, Type[ProblemSpec]]:
        """
        Map of taxonomy-id → ProblemSpec class.

        The class is instantiated with problem-specific parameters where needed.
        """
        ...

    @abstractmethod
    def solvers(self) -> Dict[str, Type[Solver]]:
        """Map of taxonomy-id → default Solver class."""
        ...

    @abstractmethod
    def discretizations(self) -> Dict[str, Sequence[Type[Discretization]]]:
        """Map of taxonomy-id → available discretizations (ordered by preference)."""
        ...

    def observables(self) -> Dict[str, Sequence[Type[Observable]]]:
        """Map of taxonomy-id → observables.  Override to fill."""
        return {}

    # ──────── lifecycle ────────

    def on_register(self) -> None:
        """Hook called after the pack is added to the registry."""

    def on_unregister(self) -> None:
        """Hook called before the pack is removed."""

    # ──────── metadata ────────

    def benchmarks(self) -> Dict[str, Sequence[str]]:
        """Map of taxonomy-id → benchmark names."""
        return {}

    def version(self) -> str:
        """SemVer string for this pack."""
        return "0.1.0"

    # ──────── repr ────────

    def __repr__(self) -> str:
        return (
            f"<DomainPack {self.pack_id}: {self.pack_name} "
            f"({len(self.taxonomy_ids)} nodes, v{self.version()})>"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DomainRegistry (singleton)
# ═══════════════════════════════════════════════════════════════════════════════


class DomainRegistry:
    """
    In-process registry of DomainPack instances.

    Usage::

        reg = get_registry()
        reg.register_pack(MyFluidsPack())
        pack = reg.get_pack("II")
        solver_cls = reg.get_solver("PHY-II.1")
        all_ids = reg.list_nodes()
    """

    _instance: Optional["DomainRegistry"] = None

    def __init__(self) -> None:
        self._packs: Dict[str, DomainPack] = {}
        self._node_to_pack: Dict[str, str] = {}

    # ── singleton ──

    @classmethod
    def _get_instance(cls) -> "DomainRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the global singleton (test utility)."""
        cls._instance = None

    # ── registration ──

    def register_pack(self, pack: DomainPack) -> None:
        pid = pack.pack_id
        if pid in self._packs:
            raise ValueError(
                f"Pack '{pid}' ({self._packs[pid].pack_name}) already "
                f"registered.  Unregister first."
            )
        self._packs[pid] = pack
        for nid in pack.taxonomy_ids:
            if nid in self._node_to_pack:
                raise ValueError(
                    f"Taxonomy node '{nid}' already claimed by pack "
                    f"'{self._node_to_pack[nid]}'"
                )
            self._node_to_pack[nid] = pid
        pack.on_register()
        logger.info("Registered DomainPack %s (%s)", pid, pack.pack_name)

    def unregister_pack(self, pack_id: str) -> None:
        if pack_id not in self._packs:
            raise KeyError(f"Pack '{pack_id}' not registered")
        pack = self._packs.pop(pack_id)
        for nid in pack.taxonomy_ids:
            self._node_to_pack.pop(nid, None)
        pack.on_unregister()
        logger.info("Unregistered DomainPack %s", pack_id)

    # ── class-decorator shortcut ──

    @staticmethod
    def register(cls: type) -> type:
        """
        Class decorator::

            @DomainRegistry.register
            class FluidsPack(DomainPack):
                ...

        Instantiates and registers the pack in the global registry.
        """
        instance = cls()
        get_registry().register_pack(instance)
        return cls

    # ── queries ──

    def get_pack(self, pack_id: str) -> DomainPack:
        if pack_id not in self._packs:
            raise KeyError(
                f"Pack '{pack_id}' not registered.  "
                f"Available: {list(self._packs)}"
            )
        return self._packs[pack_id]

    def get_pack_for_node(self, taxonomy_id: str) -> DomainPack:
        if taxonomy_id not in self._node_to_pack:
            raise KeyError(
                f"Node '{taxonomy_id}' not claimed by any pack"
            )
        return self._packs[self._node_to_pack[taxonomy_id]]

    def get_solver(self, taxonomy_id: str) -> Type[Solver]:
        pack = self.get_pack_for_node(taxonomy_id)
        solvers = pack.solvers()
        if taxonomy_id not in solvers:
            raise KeyError(
                f"Pack '{pack.pack_id}' has no solver for '{taxonomy_id}'"
            )
        return solvers[taxonomy_id]

    def get_problem_spec(self, taxonomy_id: str) -> Type[ProblemSpec]:
        pack = self.get_pack_for_node(taxonomy_id)
        specs = pack.problem_specs()
        if taxonomy_id not in specs:
            raise KeyError(
                f"Pack '{pack.pack_id}' has no ProblemSpec for '{taxonomy_id}'"
            )
        return specs[taxonomy_id]

    def list_packs(self) -> List[str]:
        return sorted(self._packs.keys())

    def list_nodes(self) -> List[str]:
        return sorted(self._node_to_pack.keys())

    def summary(self) -> Dict[str, Any]:
        return {
            "packs": len(self._packs),
            "nodes": len(self._node_to_pack),
            "detail": {
                pid: {
                    "name": p.pack_name,
                    "nodes": len(p.taxonomy_ids),
                    "version": p.version(),
                }
                for pid, p in sorted(self._packs.items())
            },
        }

    # ── auto-discovery ──

    def discover(self, module_path: str) -> int:
        """
        Import *module_path* (e.g. ``'tensornet.packs'``) and let any
        ``@DomainRegistry.register`` decorators fire.  Returns the number
        of newly registered packs.
        """
        before = len(self._packs)
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            logger.warning("Could not import %s: %s", module_path, exc)
            return 0
        return len(self._packs) - before


def get_registry() -> DomainRegistry:
    """Return the global DomainRegistry singleton."""
    return DomainRegistry._get_instance()
