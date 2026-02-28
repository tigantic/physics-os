"""
Provenance Store
=================

Content-addressed storage for field data and commits.

Features:
- Deduplication via content hashing
- Lazy loading of field data
- Configurable backends (filesystem, memory)
- Garbage collection
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .commit import FieldCommit
from .history import Branch, HistoryGraph
from .merkle import compute_hash

# =============================================================================
# CONTENT ADDRESS
# =============================================================================


@dataclass
class ContentAddress:
    """
    Content-addressed reference to data.

    Contains hash and metadata about the stored object.
    """

    hash: str
    size_bytes: int = 0
    object_type: str = "blob"  # "blob", "commit", "tree"

    # Storage location
    path: str | None = None

    def __hash__(self):
        return hash(self.hash)

    def __eq__(self, other):
        if isinstance(other, ContentAddress):
            return self.hash == other.hash
        return False

    @property
    def short_hash(self) -> str:
        return self.hash[:8]


# =============================================================================
# STORE CONFIG
# =============================================================================


@dataclass
class StoreConfig:
    """Configuration for provenance store."""

    # Storage path
    path: str = "./.physics_os"

    # Compression
    compress: bool = True
    compression_level: int = 6

    # Garbage collection
    gc_enabled: bool = True
    gc_threshold_mb: float = 1000.0  # Run GC when store exceeds this

    # Caching
    cache_enabled: bool = True
    cache_size_mb: float = 100.0

    # Data format
    data_format: str = "numpy"  # "numpy", "torch", "safetensors"


# =============================================================================
# STORAGE BACKEND
# =============================================================================


class StorageBackend(ABC):
    """Abstract storage backend."""

    @abstractmethod
    def put(self, hash: str, data: bytes) -> ContentAddress:
        """Store data and return address."""
        pass

    @abstractmethod
    def get(self, hash: str) -> bytes | None:
        """Retrieve data by hash."""
        pass

    @abstractmethod
    def exists(self, hash: str) -> bool:
        """Check if hash exists."""
        pass

    @abstractmethod
    def delete(self, hash: str) -> bool:
        """Delete data by hash."""
        pass

    @abstractmethod
    def list_hashes(self) -> Iterator[str]:
        """List all stored hashes."""
        pass


class MemoryBackend(StorageBackend):
    """In-memory storage backend."""

    def __init__(self):
        self._data: dict[str, bytes] = {}

    def put(self, hash: str, data: bytes) -> ContentAddress:
        self._data[hash] = data
        return ContentAddress(hash=hash, size_bytes=len(data))

    def get(self, hash: str) -> bytes | None:
        return self._data.get(hash)

    def exists(self, hash: str) -> bool:
        return hash in self._data

    def delete(self, hash: str) -> bool:
        if hash in self._data:
            del self._data[hash]
            return True
        return False

    def list_hashes(self) -> Iterator[str]:
        return iter(self._data.keys())


class FileSystemBackend(StorageBackend):
    """File system storage backend."""

    def __init__(self, base_path: str, compress: bool = True):
        self.base_path = Path(base_path)
        self.compress = compress

        # Create directories
        (self.base_path / "objects").mkdir(parents=True, exist_ok=True)

    def _object_path(self, hash: str) -> Path:
        """Get path for object."""
        # Use first 2 chars as directory (like git)
        return self.base_path / "objects" / hash[:2] / hash[2:]

    def put(self, hash: str, data: bytes) -> ContentAddress:
        path = self._object_path(hash)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.compress:
            import zlib

            data = zlib.compress(data)

        with open(path, "wb") as f:
            f.write(data)

        return ContentAddress(
            hash=hash,
            size_bytes=len(data),
            path=str(path),
        )

    def get(self, hash: str) -> bytes | None:
        path = self._object_path(hash)

        if not path.exists():
            return None

        with open(path, "rb") as f:
            data = f.read()

        if self.compress:
            import zlib

            data = zlib.decompress(data)

        return data

    def exists(self, hash: str) -> bool:
        return self._object_path(hash).exists()

    def delete(self, hash: str) -> bool:
        path = self._object_path(hash)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_hashes(self) -> Iterator[str]:
        objects_dir = self.base_path / "objects"
        if not objects_dir.exists():
            return

        for prefix_dir in objects_dir.iterdir():
            if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
                for obj_file in prefix_dir.iterdir():
                    yield prefix_dir.name + obj_file.name


# =============================================================================
# PROVENANCE STORE
# =============================================================================


class ProvenanceStore:
    """
    Content-addressed store for field provenance.

    Provides git-like interface for tracking field history.

    Example:
        store = ProvenanceStore("./history")

        # Commit field state
        commit = store.commit(field_data, message="Initial state")

        # Make changes and commit again
        commit2 = store.commit(new_field_data, message="After step")

        # Browse history
        for c in store.log():
            print(f"{c.short_hash}: {c.metadata.message}")

        # Checkout old state
        old_data = store.checkout(commit.hash)

        # Create branch
        store.branch("experiment")
        store.checkout("experiment")
    """

    def __init__(
        self,
        path: str | None = None,
        config: StoreConfig | None = None,
        backend: StorageBackend | None = None,
    ):
        self.config = config or StoreConfig(path=path or "./.physics_os")

        # Initialize backend
        if backend is not None:
            self._backend = backend
        elif path is None and config is None:
            self._backend = MemoryBackend()
        else:
            self._backend = FileSystemBackend(
                self.config.path,
                compress=self.config.compress,
            )

        # Initialize history graph
        self._history = HistoryGraph()

        # Load existing history if present
        self._load_history()

        # Cache
        self._cache: dict[str, np.ndarray] = {}

    @property
    def head(self) -> str | None:
        """Current HEAD commit hash."""
        return self._history.head

    @property
    def current_branch(self) -> str | None:
        """Current branch name."""
        return self._history.current_branch

    @property
    def branches(self) -> list[str]:
        """List of branch names."""
        return self._history.branches

    def commit(
        self,
        field_data: np.ndarray | torch.Tensor,
        message: str = "",
        author: str = "system",
        operation: str | None = None,
        parameters: dict | None = None,
    ) -> FieldCommit:
        """
        Commit field state.

        Args:
            field_data: Field data to commit
            message: Commit message
            author: Author name
            operation: Operation that produced this state
            parameters: Operation parameters

        Returns:
            New FieldCommit

        Note:
            D-011: Storage serialization is background operation.
        """
        # Convert to numpy for storage (background operation)
        if isinstance(field_data, torch.Tensor):
            np_data = field_data.detach().cpu().numpy()
        else:
            np_data = np.asarray(field_data)

        # Store data
        data_bytes = np_data.tobytes()
        data_hash = compute_hash(data_bytes)

        if not self._backend.exists(data_hash):
            # Store with metadata
            metadata = {
                "shape": list(np_data.shape),
                "dtype": str(np_data.dtype),
            }

            # Store data
            self._backend.put(data_hash, data_bytes)

            # Store metadata separately
            meta_hash = data_hash + "_meta"
            self._backend.put(meta_hash, json.dumps(metadata).encode())

        # Create commit
        parents = [self.head] if self.head else None

        fc = FieldCommit.create(
            field_data=np_data,
            parents=parents,
            message=message,
            author=author,
            operation=operation,
            parameters=parameters,
        )

        # Add to history
        self._history.commit(fc)

        # Ensure branch exists
        if not self._history.branches:
            self._history.create_branch("main", fc.hash)
            self._history.checkout("main")

        # Save history
        self._save_history()

        return fc

    def get_data(self, commit_hash: str) -> np.ndarray | None:
        """
        Get field data for a commit.

        Args:
            commit_hash: Commit hash

        Returns:
            numpy array or None
        """
        # Check cache
        if commit_hash in self._cache:
            return self._cache[commit_hash]

        # Get commit
        commit = self._history.get_commit(commit_hash)
        if not commit:
            return None

        # Get data
        data_bytes = self._backend.get(commit.data_hash)
        if data_bytes is None:
            return None

        # Get metadata
        meta_bytes = self._backend.get(commit.data_hash + "_meta")
        if meta_bytes:
            metadata = json.loads(meta_bytes.decode())
            shape = tuple(metadata["shape"])
            dtype = np.dtype(metadata["dtype"])
        else:
            # Fallback to commit info
            shape = commit.field_shape
            dtype = np.dtype(commit.field_dtype)

        # Reconstruct array
        data = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

        # Cache
        if self.config.cache_enabled:
            self._cache[commit_hash] = data
            self._trim_cache()

        return data

    def checkout(self, ref: str) -> np.ndarray | None:
        """
        Checkout a ref (branch, tag, or commit) and return data.

        Args:
            ref: Branch name, tag name, or commit hash

        Returns:
            Field data at that ref
        """
        self._history.checkout(ref)

        if self.head:
            return self.get_data(self.head)
        return None

    def branch(
        self,
        name: str,
        start_point: str | None = None,
    ) -> Branch:
        """
        Create a new branch.

        Args:
            name: Branch name
            start_point: Starting commit (default: HEAD)

        Returns:
            New Branch
        """
        branch = self._history.create_branch(name, start_point)
        self._save_history()
        return branch

    def delete_branch(self, name: str):
        """Delete a branch."""
        self._history.delete_branch(name)
        self._save_history()

    def tag(
        self,
        name: str,
        target: str | None = None,
        message: str = "",
    ):
        """Create a tag."""
        self._history.tag(name, target, message)
        self._save_history()

    def log(
        self,
        ref: str | None = None,
        n: int = 10,
    ) -> Iterator[FieldCommit]:
        """
        Get commit history.

        Args:
            ref: Starting point (default: HEAD)
            n: Maximum commits

        Yields:
            FieldCommit objects
        """
        return self._history.log(ref, n)

    def diff(
        self,
        commit1: str,
        commit2: str,
    ) -> dict[str, Any] | None:
        """
        Compute difference between two commits.

        Returns:
            Dictionary with diff statistics
        """
        data1 = self.get_data(commit1)
        data2 = self.get_data(commit2)

        if data1 is None or data2 is None:
            return None

        # Compute difference
        if data1.shape != data2.shape:
            return {
                "compatible": False,
                "shape_change": (data1.shape, data2.shape),
            }

        diff = data2 - data1

        return {
            "compatible": True,
            "max_diff": float(np.max(np.abs(diff))),
            "mean_diff": float(np.mean(np.abs(diff))),
            "l2_diff": float(np.linalg.norm(diff)),
            "changed_fraction": float(np.mean(diff != 0)),
        }

    def gc(self) -> int:
        """
        Garbage collect unreachable objects.

        Returns:
            Number of objects collected
        """
        # Find all reachable data hashes
        reachable = set()
        for commit in self._history._commits.values():
            reachable.add(commit.data_hash)
            reachable.add(commit.data_hash + "_meta")

        # Delete unreachable
        collected = 0
        for hash in list(self._backend.list_hashes()):
            if hash not in reachable:
                self._backend.delete(hash)
                collected += 1

        return collected

    def _load_history(self):
        """Load history from storage."""
        history_bytes = self._backend.get("__history__")
        if history_bytes:
            data = json.loads(history_bytes.decode())
            self._history = HistoryGraph.from_dict(data)

    def _save_history(self):
        """Save history to storage."""
        data = self._history.to_dict()
        self._backend.put("__history__", json.dumps(data).encode())

    def _trim_cache(self):
        """Trim cache to configured size."""
        if not self.config.cache_enabled:
            return

        # Calculate current size
        total_bytes = sum(arr.nbytes for arr in self._cache.values())
        max_bytes = self.config.cache_size_mb * 1024 * 1024

        # Remove oldest entries if over limit
        while total_bytes > max_bytes and self._cache:
            oldest_key = next(iter(self._cache))
            total_bytes -= self._cache[oldest_key].nbytes
            del self._cache[oldest_key]

    def get_statistics(self) -> dict[str, Any]:
        """Get store statistics."""
        object_count = sum(1 for _ in self._backend.list_hashes())

        return {
            "path": self.config.path,
            "object_count": object_count,
            "commit_count": len(self._history._commits),
            "branch_count": len(self._history.branches),
            "head": self.head,
            "current_branch": self.current_branch,
        }
