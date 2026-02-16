#!/usr/bin/env python3
"""
TPC Registry — 140-Domain Certificate Index
=============================================

Machine-readable registry linking all 140 computational physics domains
to their STARK trace adapters, Lean conservation proofs, and TPC certificates.

Usage:
    from tpc.registry import REGISTRY, get_domain, list_domains

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_INDEX_PATH = Path(__file__).resolve().parent.parent / "certificates" / "index.json"


def _load_index() -> Dict[str, Any]:
    """Load the certificate index from disk."""
    with open(_INDEX_PATH) as f:
        return json.load(f)


class DomainEntry:
    """A single domain registry entry."""

    __slots__ = (
        "index", "domain_id", "name", "adapter_module", "adapter_class",
        "lean_proof", "conservation_laws", "proof_system", "phase",
        "category", "certified",
    )

    def __init__(self, data: Dict[str, Any]) -> None:
        for k in self.__slots__:
            setattr(self, k, data.get(k))

    def import_adapter(self) -> type:
        """Dynamically import and return the adapter class."""
        import importlib
        mod = importlib.import_module(self.adapter_module)
        return getattr(mod, self.adapter_class)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self) -> str:
        return f"DomainEntry({self.domain_id!r}, {self.name!r}, phase={self.phase})"


class Registry:
    """The 140-domain TPC certificate registry."""

    def __init__(self) -> None:
        data = _load_index()
        self.version: str = data["version"]
        self.total_domains: int = data["total_domains"]
        self.certified: int = data["certified"]
        self.phases: Dict[str, int] = data["phases"]
        self.categories: Dict[str, int] = data["categories"]
        self.lean_proofs: List[str] = data["lean_proofs"]
        self._entries: List[DomainEntry] = [
            DomainEntry(c) for c in data["certificates"]
        ]
        self._by_id: Dict[str, DomainEntry] = {
            e.domain_id: e for e in self._entries
        }
        self._by_name: Dict[str, DomainEntry] = {
            e.name: e for e in self._entries
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def get_domain(self, domain_id: str) -> Optional[DomainEntry]:
        """Look up by domain ID (e.g., 'II.2')."""
        return self._by_id.get(domain_id)

    def get_by_name(self, name: str) -> Optional[DomainEntry]:
        """Look up by domain name."""
        return self._by_name.get(name)

    def filter_phase(self, phase: int) -> List[DomainEntry]:
        """Return all domains in a given phase."""
        return [e for e in self._entries if e.phase == phase]

    def filter_category(self, category: str) -> List[DomainEntry]:
        """Return all domains in a given category."""
        return [e for e in self._entries if e.category == category]

    def list_domains(self) -> List[str]:
        """Return all domain IDs."""
        return [e.domain_id for e in self._entries]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"TPC Registry v{self.version}: {self.certified}/{self.total_domains} certified",
            "",
            "Phases:",
        ]
        for p, n in sorted(self.phases.items(), key=lambda x: int(x[0])):
            lines.append(f"  Phase {p}: {n} domains")
        lines.append("")
        lines.append("Categories:")
        for cat, n in sorted(self.categories.items()):
            lines.append(f"  {cat}: {n}")
        return "\n".join(lines)


# Singleton instance
REGISTRY = Registry()


def get_domain(domain_id: str) -> Optional[DomainEntry]:
    """Convenience function."""
    return REGISTRY.get_domain(domain_id)


def list_domains() -> List[str]:
    """Convenience function."""
    return REGISTRY.list_domains()


if __name__ == "__main__":
    print(REGISTRY.summary())
    print(f"\nTotal: {len(REGISTRY)} domains")
