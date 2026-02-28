"""
Session Management
===================

Persistent sessions with checkpointing.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .field import Field

logger = logging.getLogger(__name__)


# =============================================================================
# SESSION STATE
# =============================================================================


class SessionState(Enum):
    """Session lifecycle state."""

    NEW = "new"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SessionMetadata:
    """Session metadata."""

    id: str
    name: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    state: SessionState = SessionState.NEW
    description: str = ""
    tags: list[str] = field(default_factory=list)
    user: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state.value,
            "description": self.description,
            "tags": self.tags,
            "user": self.user,
        }


# =============================================================================
# CHECKPOINT
# =============================================================================


@dataclass
class Checkpoint:
    """
    A snapshot of session state at a point in time.
    """

    id: str
    session_id: str
    timestamp: float = field(default_factory=time.time)
    name: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "name": self.name,
            "context": self.context,
            "metadata": self.metadata,
            # Fields stored separately
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            timestamp=data.get("timestamp", time.time()),
            name=data.get("name", ""),
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path):
        """Save checkpoint to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(path / "checkpoint.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save fields
        for name, field_data in self.fields.items():
            if isinstance(field_data, np.ndarray):
                np.save(path / f"{name}.npy", field_data)
            elif isinstance(field_data, Field):
                field_data.save(str(path / f"{name}.field"))

    @classmethod
    def load(cls, path: str | Path) -> Checkpoint:
        """Load checkpoint from disk."""
        path = Path(path)

        # Load metadata
        with open(path / "checkpoint.json") as f:
            data = json.load(f)

        checkpoint = cls.from_dict(data)

        # Load fields
        for npy_file in path.glob("*.npy"):
            name = npy_file.stem
            checkpoint.fields[name] = np.load(npy_file)

        return checkpoint


# =============================================================================
# SESSION
# =============================================================================


class Session:
    """
    Persistent session with state management.

    Example:
        session = Session("experiment-1")
        session.start()

        session.set("temperature", field)
        session.checkpoint("after_init")

        # ... work ...

        session.restore("after_init")
        session.complete()
    """

    _id_counter = 0  # Class-level counter for unique IDs

    def __init__(
        self,
        name: str,
        id: str | None = None,
        storage_path: str | None = None,
    ):
        self._id = id or self._generate_id()
        self._metadata = SessionMetadata(
            id=self._id,
            name=name,
        )
        self._storage_path = Path(storage_path) if storage_path else None

        # Session data
        self._fields: dict[str, Field] = {}
        self._context: dict[str, Any] = {}
        self._checkpoints: dict[str, Checkpoint] = {}

        # History
        self._history: list[dict[str, Any]] = []

    def _generate_id(self) -> str:
        """Generate unique session ID."""
        Session._id_counter += 1
        timestamp = str(time.time()).encode() + str(Session._id_counter).encode()
        return hashlib.sha256(timestamp).hexdigest()[:12]

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._metadata.name

    @property
    def state(self) -> SessionState:
        return self._metadata.state

    @property
    def metadata(self) -> SessionMetadata:
        return self._metadata

    @property
    def fields(self) -> dict[str, Field]:
        return self._fields

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self) -> Session:
        """Start the session."""
        self._metadata.state = SessionState.ACTIVE
        self._metadata.updated_at = time.time()
        self._log("started")
        return self

    def pause(self):
        """Pause the session."""
        self._metadata.state = SessionState.PAUSED
        self._metadata.updated_at = time.time()
        self._log("paused")

    def resume(self):
        """Resume the session."""
        self._metadata.state = SessionState.ACTIVE
        self._metadata.updated_at = time.time()
        self._log("resumed")

    def complete(self):
        """Mark session as completed."""
        self._metadata.state = SessionState.COMPLETED
        self._metadata.updated_at = time.time()
        self._log("completed")

        # Auto-save if storage configured
        if self._storage_path:
            self.save()

    def fail(self, error: str = ""):
        """Mark session as failed."""
        self._metadata.state = SessionState.FAILED
        self._metadata.updated_at = time.time()
        self._log("failed", {"error": error})

    # -------------------------------------------------------------------------
    # Data Management
    # -------------------------------------------------------------------------

    def set(self, name: str, field: Field):
        """Set a field in session."""
        self._fields[name] = field
        self._metadata.updated_at = time.time()
        self._log("field_set", {"name": name})

    def get(self, name: str) -> Field | None:
        """Get a field from session."""
        return self._fields.get(name)

    def delete(self, name: str):
        """Delete a field from session."""
        if name in self._fields:
            del self._fields[name]
            self._log("field_deleted", {"name": name})

    def set_context(self, key: str, value: Any):
        """Set context value."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value."""
        return self._context.get(key, default)

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def checkpoint(self, name: str = "") -> Checkpoint:
        """
        Create a checkpoint of current state.

        Args:
            name: Optional checkpoint name

        Returns:
            Created checkpoint
        """
        checkpoint_id = self._generate_id()

        # Copy field data
        fields_copy = {}
        for fname, field in self._fields.items():
            fields_copy[fname] = field.data.copy()

        checkpoint = Checkpoint(
            id=checkpoint_id,
            session_id=self._id,
            name=name or f"checkpoint-{len(self._checkpoints)}",
            fields=fields_copy,
            context=self._context.copy(),
            metadata={
                "session_name": self._metadata.name,
                "session_state": self._metadata.state.value,
            },
        )

        self._checkpoints[checkpoint_id] = checkpoint
        self._log("checkpoint_created", {"id": checkpoint_id, "name": name})

        return checkpoint

    def restore(self, checkpoint_id: str) -> bool:
        """
        Restore state from checkpoint.

        Args:
            checkpoint_id: Checkpoint ID or name

        Returns:
            True if restored successfully
        """
        # Find checkpoint by ID or name
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            for cp in self._checkpoints.values():
                if cp.name == checkpoint_id:
                    checkpoint = cp
                    break

        if checkpoint is None:
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return False

        # Restore fields
        for fname, data in checkpoint.fields.items():
            if fname in self._fields:
                self._fields[fname].update(data, source="restore")
            else:
                # Create new field from data
                field = Field.from_array(fname, data)
                self._fields[fname] = field

        # Restore context
        self._context = checkpoint.context.copy()

        self._log("checkpoint_restored", {"id": checkpoint.id})
        return True

    def list_checkpoints(self) -> list[Checkpoint]:
        """List all checkpoints."""
        return list(self._checkpoints.values())

    def delete_checkpoint(self, checkpoint_id: str):
        """Delete a checkpoint."""
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path: str | None = None):
        """
        Save session to disk.

        Args:
            path: Optional override path
        """
        save_path = Path(path) if path else self._storage_path
        if save_path is None:
            raise ValueError("No storage path configured")

        save_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(save_path / "session.json", "w") as f:
            json.dump(
                {
                    "metadata": self._metadata.to_dict(),
                    "context": self._context,
                    "history": self._history,
                },
                f,
                indent=2,
            )

        # Save fields
        fields_dir = save_path / "fields"
        fields_dir.mkdir(exist_ok=True)
        for name, field in self._fields.items():
            # np.savez adds .npz extension, so we just use name
            field.save(str(fields_dir / name))

        # Save checkpoints
        checkpoints_dir = save_path / "checkpoints"
        for cp_id, checkpoint in self._checkpoints.items():
            checkpoint.save(checkpoints_dir / cp_id)

        logger.info(f"Session saved: {save_path}")

    @classmethod
    def load(cls, path: str | Path) -> Session:
        """
        Load session from disk.

        Args:
            path: Path to session directory

        Returns:
            Loaded Session
        """
        path = Path(path)

        # Load metadata
        with open(path / "session.json") as f:
            data = json.load(f)

        metadata = data["metadata"]
        session = cls(
            name=metadata["name"],
            id=metadata["id"],
            storage_path=str(path),
        )
        session._metadata = SessionMetadata(
            id=metadata["id"],
            name=metadata["name"],
            created_at=metadata.get("created_at", time.time()),
            updated_at=metadata.get("updated_at", time.time()),
            state=SessionState(metadata.get("state", "new")),
            description=metadata.get("description", ""),
            tags=metadata.get("tags", []),
            user=metadata.get("user"),
        )
        session._context = data.get("context", {})
        session._history = data.get("history", [])

        # Load fields
        fields_dir = path / "fields"
        if fields_dir.exists():
            for field_file in fields_dir.glob("*.npz"):
                name = field_file.stem
                session._fields[name] = Field.load(str(field_file))

        # Load checkpoints
        checkpoints_dir = path / "checkpoints"
        if checkpoints_dir.exists():
            for cp_dir in checkpoints_dir.iterdir():
                if cp_dir.is_dir():
                    checkpoint = Checkpoint.load(cp_dir)
                    session._checkpoints[checkpoint.id] = checkpoint

        logger.info(f"Session loaded: {path}")
        return session

    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------

    def _log(self, event: str, data: dict[str, Any] | None = None):
        """Log event to history."""
        self._history.append(
            {
                "event": event,
                "timestamp": time.time(),
                "data": data or {},
            }
        )

    def get_history(self) -> list[dict[str, Any]]:
        """Get session history."""
        return self._history.copy()

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> Session:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.fail(str(exc_val))
        else:
            self.complete()


# =============================================================================
# SESSION MANAGER
# =============================================================================


class SessionManager:
    """
    Manages multiple sessions.
    """

    def __init__(self, storage_root: str | None = None):
        self._storage_root = Path(storage_root) if storage_root else None
        self._sessions: dict[str, Session] = {}
        self._active_session: str | None = None

    def create(self, name: str) -> Session:
        """Create a new session."""
        storage_path = None
        if self._storage_root:
            session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
            storage_path = str(self._storage_root / session_id)

        session = Session(name, storage_path=storage_path)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def get_active(self) -> Session | None:
        """Get active session."""
        if self._active_session:
            return self._sessions.get(self._active_session)
        return None

    def set_active(self, session_id: str):
        """Set active session."""
        if session_id in self._sessions:
            self._active_session = session_id

    def list(self) -> builtins.list[Session]:
        """List all sessions."""
        return list(self._sessions.values())

    def delete(self, session_id: str):
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            if self._active_session == session_id:
                self._active_session = None

    def load_all(self) -> int:
        """Load all sessions from storage."""
        if self._storage_root is None or not self._storage_root.exists():
            return 0

        count = 0
        for session_dir in self._storage_root.iterdir():
            if session_dir.is_dir() and (session_dir / "session.json").exists():
                try:
                    session = Session.load(session_dir)
                    self._sessions[session.id] = session
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to load session {session_dir}: {e}")

        return count
