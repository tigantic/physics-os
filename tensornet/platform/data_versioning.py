"""
6.12 — Data Versioning
=======================

Content-addressable storage with dataset snapshots, diff, merge, and
lineage tracking for simulation datasets.  Inspired by DVC / LakeFS
concepts, implemented purely in-process with JSON + NumPy persistence.

Components:
    * ContentStore     — content-addressable blob store (SHA-256 keyed)
    * DatasetSnapshot  — immutable point-in-time snapshot of a dataset
    * VersionGraph     — DAG of snapshots with parent links
    * DataVersioning   — top-level API (snapshot / diff / merge / checkout)
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ── Content-addressable store ────────────────────────────────────

class ContentStore:
    """SHA-256 keyed blob store (in-memory + optional disk)."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = Path(root) if root else None
        self._blobs: Dict[str, bytes] = {}

    def put(self, data: bytes) -> str:
        """Store blob, return its SHA-256 hash."""
        key = hashlib.sha256(data).hexdigest()
        self._blobs[key] = data
        if self._root:
            path = self._root / key[:2] / key[2:4] / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return key

    def get(self, key: str) -> Optional[bytes]:
        if key in self._blobs:
            return self._blobs[key]
        if self._root:
            path = self._root / key[:2] / key[2:4] / key
            if path.exists():
                data = path.read_bytes()
                self._blobs[key] = data
                return data
        return None

    def exists(self, key: str) -> bool:
        if key in self._blobs:
            return True
        if self._root:
            path = self._root / key[:2] / key[2:4] / key
            return path.exists()
        return False

    def put_array(self, arr: np.ndarray) -> str:
        """Store a NumPy array, return hash."""
        buf = arr.tobytes()
        meta = json.dumps({
            "dtype": str(arr.dtype), "shape": list(arr.shape),
        }).encode()
        # Prefix with metadata length + metadata
        payload = len(meta).to_bytes(4, "little") + meta + buf
        return self.put(payload)

    def get_array(self, key: str) -> Optional[np.ndarray]:
        raw = self.get(key)
        if raw is None:
            return None
        meta_len = int.from_bytes(raw[:4], "little")
        meta = json.loads(raw[4:4 + meta_len].decode())
        buf = raw[4 + meta_len:]
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(meta["shape"])

    @property
    def count(self) -> int:
        return len(self._blobs)


# ── Dataset snapshot ─────────────────────────────────────────────

@dataclass
class FileEntry:
    """A single file / array in the dataset."""
    path: str            # logical path inside the dataset
    blob_key: str        # SHA-256 content-address
    size_bytes: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetSnapshot:
    """Immutable point-in-time snapshot of a dataset."""
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    message: str = ""
    author: str = ""
    timestamp: float = field(default_factory=time.time)
    files: Dict[str, FileEntry] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    def add_file(self, path: str, blob_key: str, size: int = 0,
                 metadata: Optional[Dict[str, str]] = None) -> None:
        self.files[path] = FileEntry(
            path=path, blob_key=blob_key,
            size_bytes=size, metadata=metadata or {},
        )

    def file_paths(self) -> List[str]:
        return sorted(self.files.keys())

    def total_size(self) -> int:
        return sum(f.size_bytes for f in self.files.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "parent_id": self.parent_id,
            "message": self.message,
            "author": self.author,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "files": {
                k: {
                    "path": v.path, "blob_key": v.blob_key,
                    "size_bytes": v.size_bytes, "metadata": v.metadata,
                }
                for k, v in self.files.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetSnapshot":
        snap = cls(
            snapshot_id=d["snapshot_id"],
            parent_id=d.get("parent_id"),
            message=d.get("message", ""),
            author=d.get("author", ""),
            timestamp=d.get("timestamp", 0.0),
            tags=d.get("tags", {}),
        )
        for k, v in d.get("files", {}).items():
            snap.files[k] = FileEntry(
                path=v["path"], blob_key=v["blob_key"],
                size_bytes=v.get("size_bytes", 0),
                metadata=v.get("metadata", {}),
            )
        return snap


# ── Diff ─────────────────────────────────────────────────────────

@dataclass
class DiffEntry:
    path: str
    change: str          # "added", "removed", "modified"
    old_key: str = ""
    new_key: str = ""


def diff_snapshots(
    old: DatasetSnapshot, new: DatasetSnapshot,
) -> List[DiffEntry]:
    """Compute diff between two snapshots."""
    diffs: List[DiffEntry] = []
    all_paths = set(old.files.keys()) | set(new.files.keys())
    for p in sorted(all_paths):
        in_old = p in old.files
        in_new = p in new.files
        if in_old and not in_new:
            diffs.append(DiffEntry(p, "removed", old_key=old.files[p].blob_key))
        elif not in_old and in_new:
            diffs.append(DiffEntry(p, "added", new_key=new.files[p].blob_key))
        elif in_old and in_new:
            if old.files[p].blob_key != new.files[p].blob_key:
                diffs.append(DiffEntry(
                    p, "modified",
                    old_key=old.files[p].blob_key,
                    new_key=new.files[p].blob_key,
                ))
    return diffs


# ── Version graph ────────────────────────────────────────────────

class VersionGraph:
    """DAG of snapshots with parent links."""

    def __init__(self) -> None:
        self._snapshots: Dict[str, DatasetSnapshot] = {}
        self._head: Optional[str] = None
        self._branches: Dict[str, str] = {}  # branch_name -> snapshot_id

    def add_snapshot(self, snap: DatasetSnapshot, branch: str = "main") -> None:
        self._snapshots[snap.snapshot_id] = snap
        self._branches[branch] = snap.snapshot_id
        self._head = snap.snapshot_id

    def get_snapshot(self, sid: str) -> Optional[DatasetSnapshot]:
        return self._snapshots.get(sid)

    @property
    def head(self) -> Optional[DatasetSnapshot]:
        if self._head:
            return self._snapshots.get(self._head)
        return None

    def branch_head(self, branch: str) -> Optional[DatasetSnapshot]:
        sid = self._branches.get(branch)
        return self._snapshots.get(sid) if sid else None

    def history(self, start_id: Optional[str] = None) -> List[DatasetSnapshot]:
        """Walk parent chain from start_id (or head) back to root."""
        sid = start_id or self._head
        chain: List[DatasetSnapshot] = []
        while sid:
            snap = self._snapshots.get(sid)
            if snap is None:
                break
            chain.append(snap)
            sid = snap.parent_id
        return chain

    def branches(self) -> Dict[str, str]:
        return dict(self._branches)

    @property
    def count(self) -> int:
        return len(self._snapshots)


# ── Merge ────────────────────────────────────────────────────────

def merge_snapshots(
    base: DatasetSnapshot,
    theirs: DatasetSnapshot,
    ours: DatasetSnapshot,
    prefer: str = "ours",
) -> DatasetSnapshot:
    """Three-way merge of dataset snapshots.

    *prefer* controls conflict resolution: 'ours' or 'theirs'.
    """
    merged = DatasetSnapshot(
        parent_id=ours.snapshot_id,
        message=f"Merge {theirs.snapshot_id[:8]} into {ours.snapshot_id[:8]}",
        author=ours.author,
    )

    all_paths = set(base.files.keys()) | set(theirs.files.keys()) | set(ours.files.keys())

    for path in sorted(all_paths):
        base_key = base.files[path].blob_key if path in base.files else None
        their_key = theirs.files[path].blob_key if path in theirs.files else None
        our_key = ours.files[path].blob_key if path in ours.files else None

        # No conflict — same change or only one side changed
        if their_key == our_key:
            if our_key is not None:
                merged.files[path] = (ours if path in ours.files else theirs).files[path]
            continue

        # Only one side modified from base
        if our_key == base_key and their_key is not None:
            merged.files[path] = theirs.files[path]
        elif their_key == base_key and our_key is not None:
            merged.files[path] = ours.files[path]
        elif our_key is None and their_key is not None:
            merged.files[path] = theirs.files[path]
        elif their_key is None and our_key is not None:
            merged.files[path] = ours.files[path]
        else:
            # True conflict — use preference
            if prefer == "ours" and our_key is not None:
                merged.files[path] = ours.files[path]
            elif prefer == "theirs" and their_key is not None:
                merged.files[path] = theirs.files[path]

    return merged


# ── Top-level API ────────────────────────────────────────────────

class DataVersioning:
    """Content-addressable dataset versioning system."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.store = ContentStore(root / "blobs" if root else None)
        self.graph = VersionGraph()
        self._root = Path(root) if root else None

    def snapshot(
        self,
        dataset: Dict[str, np.ndarray],
        message: str = "",
        author: str = "",
        branch: str = "main",
        tags: Optional[Dict[str, str]] = None,
    ) -> DatasetSnapshot:
        """Create a new immutable snapshot from a dict of arrays."""
        parent = self.graph.branch_head(branch)
        snap = DatasetSnapshot(
            parent_id=parent.snapshot_id if parent else None,
            message=message,
            author=author,
            tags=tags or {},
        )
        for name, arr in dataset.items():
            key = self.store.put_array(arr)
            snap.add_file(name, key, size=arr.nbytes)

        self.graph.add_snapshot(snap, branch)
        return snap

    def checkout(self, snapshot_id: str) -> Dict[str, np.ndarray]:
        """Materialise a snapshot into a dict of arrays."""
        snap = self.graph.get_snapshot(snapshot_id)
        if snap is None:
            raise KeyError(f"Snapshot {snapshot_id!r} not found")
        result: Dict[str, np.ndarray] = {}
        for path, entry in snap.files.items():
            arr = self.store.get_array(entry.blob_key)
            if arr is not None:
                result[path] = arr
        return result

    def diff(
        self,
        old_id: str,
        new_id: str,
    ) -> List[DiffEntry]:
        old = self.graph.get_snapshot(old_id)
        new = self.graph.get_snapshot(new_id)
        if old is None or new is None:
            raise KeyError("Snapshot not found")
        return diff_snapshots(old, new)

    def merge(
        self,
        base_id: str,
        theirs_id: str,
        ours_id: str,
        prefer: str = "ours",
        branch: str = "main",
    ) -> DatasetSnapshot:
        base = self.graph.get_snapshot(base_id)
        theirs = self.graph.get_snapshot(theirs_id)
        ours = self.graph.get_snapshot(ours_id)
        if base is None or theirs is None or ours is None:
            raise KeyError("Snapshot not found")
        merged = merge_snapshots(base, theirs, ours, prefer)
        self.graph.add_snapshot(merged, branch)
        return merged

    def history(self, branch: str = "main") -> List[DatasetSnapshot]:
        head = self.graph.branch_head(branch)
        if head is None:
            return []
        return self.graph.history(head.snapshot_id)

    def save_graph(self, path: Optional[Path] = None) -> None:
        p = path or (self._root / "graph.json" if self._root else None)
        if p is None:
            return
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "snapshots": {
                k: v.to_dict() for k, v in self.graph._snapshots.items()
            },
            "branches": self.graph._branches,
            "head": self.graph._head,
        }
        Path(p).write_text(json.dumps(data, indent=2))

    def load_graph(self, path: Optional[Path] = None) -> int:
        p = path or (self._root / "graph.json" if self._root else None)
        if p is None or not Path(p).exists():
            return 0
        data = json.loads(Path(p).read_text())
        for k, v in data.get("snapshots", {}).items():
            self.graph._snapshots[k] = DatasetSnapshot.from_dict(v)
        self.graph._branches = data.get("branches", {})
        self.graph._head = data.get("head")
        return self.graph.count


__all__ = [
    "ContentStore",
    "FileEntry",
    "DatasetSnapshot",
    "DiffEntry",
    "diff_snapshots",
    "VersionGraph",
    "merge_snapshots",
    "DataVersioning",
]
