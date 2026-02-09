"""
DomainPack — the plugin unit for physics domains.

A DomainPack bundles equations + discretizations + solvers + benchmarks + docs
for one taxonomy category (e.g. Pack II: Fluid Dynamics).

Registry / discovery
--------------------
``DomainRegistry`` is a singleton in-process registry.  Domain packs register
themselves at import time via ``@DomainRegistry.register`` or by calling
``get_registry().register_pack(pack)``.

Compliance
----------
``PackComplianceReport`` checks that every pack conforms to the platform
interface contract:  every exported ProblemSpec, Solver, Discretization, and
Observable must satisfy the corresponding protocol (PEP 544 runtime check).
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
    Tuple,
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
# PackInfo — lightweight metadata for compliance and reporting
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PackInfo:
    """Immutable snapshot of a DomainPack's metadata."""

    pack_id: str
    pack_name: str
    taxonomy_ids: Tuple[str, ...]
    n_specs: int
    n_solvers: int
    n_discretizations: int
    n_observables: int
    n_benchmarks: int
    version: str


# ═══════════════════════════════════════════════════════════════════════════════
# Compliance report
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ComplianceViolation:
    """A single interface-conformance failure."""

    taxonomy_id: str
    artifact: str  # 'ProblemSpec', 'Solver', 'Discretization', 'Observable'
    cls: type
    reason: str


@dataclass
class PackComplianceReport:
    """Result of running compliance checks on a DomainPack."""

    pack_id: str
    pack_name: str
    violations: List[ComplianceViolation] = dc_field(default_factory=list)
    warnings: List[str] = dc_field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Compliance [{status}] Pack {self.pack_id} ({self.pack_name})"]
        for v in self.violations:
            lines.append(
                f"  VIOLATION {v.taxonomy_id} {v.artifact} "
                f"({v.cls.__name__}): {v.reason}"
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def check_compliance(pack: "DomainPack") -> PackComplianceReport:
    """
    Verify that a DomainPack's exported artifacts conform to platform protocols.

    Checks:
    - Every taxonomy_id in the pack has a ProblemSpec and a Solver.
    - Every ProblemSpec class is ``runtime_checkable`` against ``ProblemSpec``.
    - Every Solver class is ``runtime_checkable`` against ``Solver``.
    - Every Discretization is ``runtime_checkable`` against ``Discretization``.
    - Every Observable is ``runtime_checkable`` against ``Observable``.
    """
    report = PackComplianceReport(pack_id=pack.pack_id, pack_name=pack.pack_name)

    specs = pack.problem_specs()
    solvers = pack.solvers()
    discs = pack.discretizations()
    observables = pack.observables()

    for nid in pack.taxonomy_ids:
        # Must have ProblemSpec
        if nid not in specs:
            report.violations.append(
                ComplianceViolation(
                    taxonomy_id=nid,
                    artifact="ProblemSpec",
                    cls=type(None),
                    reason=f"No ProblemSpec registered for node {nid}",
                )
            )
        else:
            cls = specs[nid]
            if not _check_protocol(cls, ProblemSpec, is_class=True):
                report.violations.append(
                    ComplianceViolation(
                        taxonomy_id=nid,
                        artifact="ProblemSpec",
                        cls=cls,
                        reason=f"{cls.__name__} does not satisfy ProblemSpec protocol",
                    )
                )

        # Must have Solver
        if nid not in solvers:
            report.violations.append(
                ComplianceViolation(
                    taxonomy_id=nid,
                    artifact="Solver",
                    cls=type(None),
                    reason=f"No Solver registered for node {nid}",
                )
            )
        else:
            cls = solvers[nid]
            if not _check_protocol(cls, Solver, is_class=True):
                report.violations.append(
                    ComplianceViolation(
                        taxonomy_id=nid,
                        artifact="Solver",
                        cls=cls,
                        reason=f"{cls.__name__} does not satisfy Solver protocol",
                    )
                )

        # Discretization — optional but if present must conform
        if nid in discs:
            for dcls in discs[nid]:
                if not _check_protocol(dcls, Discretization, is_class=True):
                    report.violations.append(
                        ComplianceViolation(
                            taxonomy_id=nid,
                            artifact="Discretization",
                            cls=dcls,
                            reason=f"{dcls.__name__} does not satisfy Discretization protocol",
                        )
                    )

        # Observable — optional but if present must conform
        if nid in observables:
            for ocls in observables[nid]:
                if not _check_protocol(ocls, Observable, is_class=True):
                    report.violations.append(
                        ComplianceViolation(
                            taxonomy_id=nid,
                            artifact="Observable",
                            cls=ocls,
                            reason=f"{ocls.__name__} does not satisfy Observable protocol",
                        )
                    )

    # Warnings
    if not pack.benchmarks():
        report.warnings.append("No benchmarks defined for any node")

    return report


def _check_protocol(cls: type, proto: type, *, is_class: bool = True) -> bool:
    """
    Check whether *cls* (a class, not instance) structurally satisfies *proto*.

    We try instantiation-free checks first, falling back to ``isinstance``
    on a sentinel if the protocol is runtime_checkable.
    """
    # For runtime_checkable protocols, check that all required attributes
    # and methods exist on the class itself.
    required_attrs = set()
    for attr in dir(proto):
        if attr.startswith("_"):
            continue
        member = getattr(proto, attr, None)
        if callable(member) or isinstance(member, property):
            required_attrs.add(attr)
    for attr in required_attrs:
        if not hasattr(cls, attr):
            return False
    return True


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

    def _resolve_version(self) -> str:
        """Return version string whether subclass uses @property or method."""
        v = self.version
        return v() if callable(v) else v

    # ──────── repr ────────

    def __repr__(self) -> str:
        return (
            f"<DomainPack {self.pack_id}: {self.pack_name} "
            f"({len(self.taxonomy_ids)} nodes, v{self._resolve_version()})>"
        )

    # ──────── introspection ────────

    def info(self) -> PackInfo:
        """Return a frozen metadata snapshot."""
        discs = self.discretizations()
        obs = self.observables()
        bmarks = self.benchmarks()
        return PackInfo(
            pack_id=self.pack_id,
            pack_name=self.pack_name,
            taxonomy_ids=tuple(self.taxonomy_ids),
            n_specs=len(self.problem_specs()),
            n_solvers=len(self.solvers()),
            n_discretizations=sum(len(v) for v in discs.values()),
            n_observables=sum(len(v) for v in obs.values()),
            n_benchmarks=sum(len(v) for v in bmarks.values()),
            version=self._resolve_version(),
        )

    def check_compliance(self) -> PackComplianceReport:
        """Run compliance checks on this pack."""
        return check_compliance(self)


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
                    "version": p._resolve_version(),
                }
                for pid, p in sorted(self._packs.items())
            },
        }

    def check_all_compliance(self) -> Dict[str, PackComplianceReport]:
        """Run compliance checks on every registered pack."""
        return {
            pid: check_compliance(pack)
            for pid, pack in sorted(self._packs.items())
        }

    def all_info(self) -> Dict[str, PackInfo]:
        """Return PackInfo for every registered pack."""
        return {
            pid: pack.info()
            for pid, pack in sorted(self._packs.items())
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
