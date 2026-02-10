"""
6.3 — Data Lakehouse for Simulation Artifacts
===============================================

Unified storage layer that combines data-lake flexibility (raw files,
any format) with data-warehouse structure (schema, partitioning,
metadata catalog).

Components:
    * LakehouseCatalog — metadata registry for all artifacts
    * Partition — partitioning scheme (by domain, timestamp, run_id)
    * LakehouseStore — read/write with automatic cataloguing
    * LakehouseQuery — SQL-like filter/project/aggregate interface
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Artifact types ────────────────────────────────────────────────

class ArtifactType(Enum):
    """Types of simulation artifacts stored in the lakehouse."""
    FIELD_SNAPSHOT = auto()
    MESH = auto()
    CHECKPOINT = auto()
    METRICS = auto()
    CONFIG = auto()
    LOG = auto()
    VISUALIZATION = auto()
    DERIVED = auto()


@dataclass
class ArtifactMetadata:
    """Metadata for a single lakehouse artifact."""
    artifact_id: str
    artifact_type: ArtifactType
    domain: str = ""            # physics domain
    run_id: str = ""
    timestamp: float = 0.0
    size_bytes: int = 0
    checksum: str = ""          # SHA-256
    tags: Dict[str, str] = field(default_factory=dict)
    schema: Dict[str, str] = field(default_factory=dict)  # column → dtype
    partition_key: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.name,
            "domain": self.domain,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "tags": self.tags,
            "schema": self.schema,
            "partition_key": self.partition_key,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactMetadata":
        d = dict(d)
        d["artifact_type"] = ArtifactType[d["artifact_type"]]
        return cls(**d)


# ── Partition scheme ──────────────────────────────────────────────

class PartitionScheme(Enum):
    """How artifacts are partitioned on disk."""
    BY_DOMAIN = auto()           # domain=cfd/run_id=abc/...
    BY_TIMESTAMP = auto()        # year=2024/month=06/day=15/...
    BY_RUN = auto()              # run_id=abc/...
    FLAT = auto()                # no partitioning


def _partition_path(meta: ArtifactMetadata, scheme: PartitionScheme) -> str:
    if scheme == PartitionScheme.BY_DOMAIN:
        return f"domain={meta.domain}/run={meta.run_id}"
    elif scheme == PartitionScheme.BY_TIMESTAMP:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(meta.timestamp, tz=timezone.utc)
        return f"year={dt.year}/month={dt.month:02d}/day={dt.day:02d}"
    elif scheme == PartitionScheme.BY_RUN:
        return f"run={meta.run_id}"
    return ""


# ── Lakehouse catalog ─────────────────────────────────────────────

class LakehouseCatalog:
    """In-memory metadata catalog with persistence to JSON."""

    def __init__(self, catalog_path: Optional[Path] = None) -> None:
        self._entries: Dict[str, ArtifactMetadata] = {}
        self._catalog_path = catalog_path

    def register(self, meta: ArtifactMetadata) -> None:
        self._entries[meta.artifact_id] = meta

    def get(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        return self._entries.get(artifact_id)

    def list_all(self) -> List[ArtifactMetadata]:
        return list(self._entries.values())

    def query(
        self,
        domain: Optional[str] = None,
        run_id: Optional[str] = None,
        artifact_type: Optional[ArtifactType] = None,
        tags: Optional[Dict[str, str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[ArtifactMetadata]:
        """Filter catalog entries."""
        results: List[ArtifactMetadata] = []
        for m in self._entries.values():
            if domain and m.domain != domain:
                continue
            if run_id and m.run_id != run_id:
                continue
            if artifact_type and m.artifact_type != artifact_type:
                continue
            if tags and not all(m.tags.get(k) == v for k, v in tags.items()):
                continue
            if start_time and m.timestamp < start_time:
                continue
            if end_time and m.timestamp > end_time:
                continue
            results.append(m)
        return results

    def remove(self, artifact_id: str) -> bool:
        if artifact_id in self._entries:
            del self._entries[artifact_id]
            return True
        return False

    def save(self, path: Optional[Path] = None) -> None:
        p = path or self._catalog_path
        if p is None:
            return
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in self._entries.values()]
        p.write_text(json.dumps(data, indent=2))

    def load(self, path: Optional[Path] = None) -> int:
        p = path or self._catalog_path
        if p is None or not Path(p).exists():
            return 0
        data = json.loads(Path(p).read_text())
        for d in data:
            meta = ArtifactMetadata.from_dict(d)
            self._entries[meta.artifact_id] = meta
        return len(data)

    @property
    def count(self) -> int:
        return len(self._entries)


# ── Lakehouse store ───────────────────────────────────────────────

class LakehouseStore:
    """Read/write store with automatic cataloguing and partitioning."""

    def __init__(
        self,
        root: Path,
        scheme: PartitionScheme = PartitionScheme.BY_DOMAIN,
        catalog: Optional[LakehouseCatalog] = None,
    ) -> None:
        self.root = Path(root)
        self.scheme = scheme
        self.catalog = catalog or LakehouseCatalog(self.root / "_catalog.json")

    def put(
        self,
        data: np.ndarray,
        artifact_type: ArtifactType,
        domain: str = "",
        run_id: str = "",
        tags: Optional[Dict[str, str]] = None,
        artifact_id: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Store a NumPy array artifact."""
        raw = data.tobytes()
        checksum = hashlib.sha256(raw).hexdigest()
        aid = artifact_id or checksum[:16]

        meta = ArtifactMetadata(
            artifact_id=aid,
            artifact_type=artifact_type,
            domain=domain,
            run_id=run_id,
            timestamp=time.time(),
            size_bytes=len(raw),
            checksum=checksum,
            tags=tags or {},
            schema={"dtype": str(data.dtype), "shape": str(data.shape)},
        )
        meta.partition_key = _partition_path(meta, self.scheme)

        # Write to disk
        part_dir = self.root / meta.partition_key if meta.partition_key else self.root
        part_dir.mkdir(parents=True, exist_ok=True)
        np.save(part_dir / f"{aid}.npy", data)

        # Register
        self.catalog.register(meta)
        self.catalog.save()
        return meta

    def get(self, artifact_id: str) -> Optional[np.ndarray]:
        """Retrieve an artifact by ID."""
        meta = self.catalog.get(artifact_id)
        if meta is None:
            return None
        part_dir = self.root / meta.partition_key if meta.partition_key else self.root
        path = part_dir / f"{artifact_id}.npy"
        if not path.exists():
            return None
        return np.load(path)

    def put_json(
        self,
        data: Dict[str, Any],
        artifact_type: ArtifactType,
        domain: str = "",
        run_id: str = "",
        tags: Optional[Dict[str, str]] = None,
        artifact_id: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Store a JSON artifact."""
        raw = json.dumps(data, indent=2).encode()
        checksum = hashlib.sha256(raw).hexdigest()
        aid = artifact_id or checksum[:16]

        meta = ArtifactMetadata(
            artifact_id=aid,
            artifact_type=artifact_type,
            domain=domain,
            run_id=run_id,
            timestamp=time.time(),
            size_bytes=len(raw),
            checksum=checksum,
            tags=tags or {},
        )
        meta.partition_key = _partition_path(meta, self.scheme)

        part_dir = self.root / meta.partition_key if meta.partition_key else self.root
        part_dir.mkdir(parents=True, exist_ok=True)
        (part_dir / f"{aid}.json").write_bytes(raw)

        self.catalog.register(meta)
        self.catalog.save()
        return meta

    def query(self, **kwargs: Any) -> List[ArtifactMetadata]:
        return self.catalog.query(**kwargs)


# ── Lakehouse query builder ──────────────────────────────────────

class LakehouseQuery:
    """SQL-like query builder for lakehouse artifacts."""

    def __init__(self, catalog: LakehouseCatalog) -> None:
        self._catalog = catalog
        self._filters: List[Callable[[ArtifactMetadata], bool]] = []
        self._sort_key: Optional[str] = None
        self._limit: Optional[int] = None

    def where_domain(self, domain: str) -> "LakehouseQuery":
        self._filters.append(lambda m: m.domain == domain)
        return self

    def where_type(self, t: ArtifactType) -> "LakehouseQuery":
        self._filters.append(lambda m: m.artifact_type == t)
        return self

    def where_tag(self, key: str, value: str) -> "LakehouseQuery":
        self._filters.append(lambda m: m.tags.get(key) == value)
        return self

    def where_time_range(self, start: float, end: float) -> "LakehouseQuery":
        self._filters.append(lambda m: start <= m.timestamp <= end)
        return self

    def order_by(self, key: str = "timestamp") -> "LakehouseQuery":
        self._sort_key = key
        return self

    def limit(self, n: int) -> "LakehouseQuery":
        self._limit = n
        return self

    def execute(self) -> List[ArtifactMetadata]:
        results = self._catalog.list_all()
        for f in self._filters:
            results = [m for m in results if f(m)]
        if self._sort_key:
            results.sort(key=lambda m: getattr(m, self._sort_key, 0))
        if self._limit:
            results = results[:self._limit]
        return results


__all__ = [
    "ArtifactType",
    "ArtifactMetadata",
    "PartitionScheme",
    "LakehouseCatalog",
    "LakehouseStore",
    "LakehouseQuery",
]
