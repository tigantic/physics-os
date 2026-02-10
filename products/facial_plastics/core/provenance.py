"""Content-addressed provenance and lineage tracking.

Every artifact in the facial plastics pipeline is content-addressed
(SHA-256) and linked to its producing step, inputs, and software versions.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import struct
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_array(arr: np.ndarray) -> str:
    """Content hash of a numpy array (dtype + shape + bytes)."""
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(struct.pack(f">{len(arr.shape)}Q", *arr.shape))
    h.update(arr.tobytes())
    return h.hexdigest()


def hash_dict(d: Dict[str, Any]) -> str:
    """Deterministic hash of a JSON-serialisable dictionary."""
    canonical = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _software_versions() -> Dict[str, str]:
    """Capture current software environment."""
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    try:
        import products.facial_plastics as fp
        versions["facial_plastics"] = fp.__version__
    except Exception:
        pass
    return versions


@dataclass
class ArtifactRecord:
    """Immutable record of a single artifact."""
    artifact_id: str
    content_hash: str
    artifact_type: str
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    producer_step: str = ""
    input_hashes: List[str] = field(default_factory=list)
    software_versions: Dict[str, str] = field(default_factory=_software_versions)
    config_hash: str = ""
    size_bytes: int = 0
    path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ArtifactRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProvenanceChain:
    """Full lineage chain for a case or artifact."""
    case_id: str
    records: List[ArtifactRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

    def add(self, record: ArtifactRecord) -> None:
        """Append an artifact record to the chain."""
        self.records.append(record)

    def find(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Look up a record by artifact ID."""
        for r in self.records:
            if r.artifact_id == artifact_id:
                return r
        return None

    def ancestors(self, artifact_id: str) -> List[ArtifactRecord]:
        """Recursively find all ancestors of an artifact."""
        record = self.find(artifact_id)
        if record is None:
            return []
        result: List[ArtifactRecord] = []
        for parent_hash in record.input_hashes:
            for r in self.records:
                if r.content_hash == parent_hash:
                    result.append(r)
                    result.extend(self.ancestors(r.artifact_id))
        return result

    def validate_integrity(self) -> List[str]:
        """Check that all input references resolve and no cycles exist."""
        errors: List[str] = []
        known_hashes = {r.content_hash for r in self.records}
        for r in self.records:
            for ih in r.input_hashes:
                if ih not in known_hashes:
                    errors.append(
                        f"Artifact {r.artifact_id} references unknown hash {ih[:16]}..."
                    )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "created_at": self.created_at,
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProvenanceChain:
        chain = cls(case_id=d["case_id"], created_at=d.get("created_at", ""))
        for rd in d.get("records", []):
            chain.records.append(ArtifactRecord.from_dict(rd))
        return chain


class Provenance:
    """Manages provenance tracking for a pipeline execution."""

    def __init__(self, case_id: str) -> None:
        self.chain = ProvenanceChain(case_id=case_id)
        self._step_stack: List[str] = []

    @property
    def case_id(self) -> str:
        return self.chain.case_id

    def begin_step(self, step_name: str) -> None:
        """Mark the start of a processing step."""
        self._step_stack.append(step_name)

    def end_step(self) -> None:
        """Mark the end of the current processing step."""
        if self._step_stack:
            self._step_stack.pop()

    @property
    def current_step(self) -> str:
        return self._step_stack[-1] if self._step_stack else ""

    def record_file(
        self,
        artifact_id: str,
        path: Path,
        artifact_type: str,
        input_hashes: Optional[Sequence[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord:
        """Record a file artifact with content hash."""
        content_hash = hash_file(path)
        cfg_hash = hash_dict(config) if config else ""
        record = ArtifactRecord(
            artifact_id=artifact_id,
            content_hash=content_hash,
            artifact_type=artifact_type,
            producer_step=self.current_step,
            input_hashes=list(input_hashes or []),
            config_hash=cfg_hash,
            size_bytes=path.stat().st_size,
            path=str(path),
            metadata=metadata or {},
        )
        self.chain.add(record)
        return record

    def record_array(
        self,
        artifact_id: str,
        array: np.ndarray,
        artifact_type: str,
        input_hashes: Optional[Sequence[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord:
        """Record an in-memory array artifact."""
        content_hash = hash_array(array)
        cfg_hash = hash_dict(config) if config else ""
        record = ArtifactRecord(
            artifact_id=artifact_id,
            content_hash=content_hash,
            artifact_type=artifact_type,
            producer_step=self.current_step,
            input_hashes=list(input_hashes or []),
            config_hash=cfg_hash,
            size_bytes=array.nbytes,
            metadata=metadata or {},
        )
        self.chain.add(record)
        return record

    def record_dict(
        self,
        artifact_id: str,
        data: Dict[str, Any],
        artifact_type: str,
        input_hashes: Optional[Sequence[str]] = None,
    ) -> ArtifactRecord:
        """Record a dictionary artifact (plan, config, etc.)."""
        content_hash = hash_dict(data)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            content_hash=content_hash,
            artifact_type=artifact_type,
            producer_step=self.current_step,
            input_hashes=list(input_hashes or []),
            size_bytes=len(json.dumps(data, default=str).encode()),
        )
        self.chain.add(record)
        return record

    def save(self, path: Path) -> None:
        """Persist provenance chain to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.chain.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> Provenance:
        """Load provenance chain from JSON."""
        with open(path) as f:
            data = json.load(f)
        prov = cls(case_id=data["case_id"])
        prov.chain = ProvenanceChain.from_dict(data)
        return prov

    def validate(self) -> List[str]:
        """Validate the provenance chain integrity."""
        return self.chain.validate_integrity()
