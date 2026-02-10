"""Immutable audit trail for all platform operations.

Records timestamped, content-addressed events for:
  - Case creation, modification, deletion
  - Plan creation, simulation runs
  - Report generation and export
  - Consent workflow changes
  - Access control events
  - Data import and export

Each event is hashed and chained to the previous event,
creating a tamper-evident log (similar to a blockchain).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Event types ──────────────────────────────────────────────────

class EventType:
    """Standard audit event categories."""
    CASE_CREATE = "case.create"
    CASE_UPDATE = "case.update"
    CASE_DELETE = "case.delete"
    CASE_EXPORT = "case.export"
    PLAN_CREATE = "plan.create"
    PLAN_UPDATE = "plan.update"
    SIM_START = "simulation.start"
    SIM_COMPLETE = "simulation.complete"
    SIM_FAIL = "simulation.fail"
    REPORT_GENERATE = "report.generate"
    REPORT_EXPORT = "report.export"
    CONSENT_RECORD = "consent.record"
    CONSENT_REVOKE = "consent.revoke"
    ACCESS_LOGIN = "access.login"
    ACCESS_LOGOUT = "access.logout"
    ACCESS_DENIED = "access.denied"
    ACCESS_GRANT = "access.grant"
    ACCESS_REVOKE = "access.revoke"
    DATA_IMPORT = "data.import"
    DATA_EXPORT = "data.export"
    CONFIG_CHANGE = "config.change"


# ── Audit event ──────────────────────────────────────────────────

@dataclass
class AuditEvent:
    """A single immutable audit event."""
    event_type: str
    timestamp_utc: float         # Unix timestamp
    user_id: str = ""
    case_id: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""
    event_hash: str = ""

    def compute_hash(self) -> str:
        """Compute content-addressed hash of this event."""
        payload = json.dumps({
            "event_type": self.event_type,
            "timestamp_utc": self.timestamp_utc,
            "user_id": self.user_id,
            "case_id": self.case_id,
            "description": self.description,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
        }, sort_keys=True, default=str)
        self.event_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self.event_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp_utc": self.timestamp_utc,
            "user_id": self.user_id,
            "case_id": self.case_id,
            "description": self.description,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AuditEvent:
        return cls(
            event_type=d["event_type"],
            timestamp_utc=d["timestamp_utc"],
            user_id=d.get("user_id", ""),
            case_id=d.get("case_id", ""),
            description=d.get("description", ""),
            metadata=d.get("metadata", {}),
            previous_hash=d.get("previous_hash", ""),
            event_hash=d.get("event_hash", ""),
        )

    def verify(self) -> bool:
        """Verify the event hash is consistent."""
        expected = hashlib.sha256(json.dumps({
            "event_type": self.event_type,
            "timestamp_utc": self.timestamp_utc,
            "user_id": self.user_id,
            "case_id": self.case_id,
            "description": self.description,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
        }, sort_keys=True, default=str).encode()).hexdigest()
        return expected == self.event_hash


# ── Audit log ────────────────────────────────────────────────────

class AuditLog:
    """Append-only, tamper-evident audit trail.

    Events are chained by hash: each event includes the hash of
    the previous event, creating an immutable sequence.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._events: List[AuditEvent] = []
        self._storage_path = storage_path
        self._last_hash = "genesis"

        if storage_path and storage_path.exists():
            self._load()

    def record(
        self,
        event_type: str,
        *,
        user_id: str = "",
        case_id: str = "",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Record a new audit event."""
        event = AuditEvent(
            event_type=event_type,
            timestamp_utc=time.time(),
            user_id=user_id,
            case_id=case_id,
            description=description,
            metadata=metadata or {},
            previous_hash=self._last_hash,
        )
        event.compute_hash()

        self._events.append(event)
        self._last_hash = event.event_hash

        if self._storage_path:
            self._persist(event)

        logger.debug(
            "Audit: %s by %s on %s: %s",
            event_type, user_id, case_id, description,
        )
        return event

    def verify_chain(self) -> bool:
        """Verify the entire audit chain integrity."""
        prev_hash = "genesis"
        for event in self._events:
            if event.previous_hash != prev_hash:
                logger.error(
                    "Chain break at event %s: expected prev=%s, got %s",
                    event.event_hash[:12], prev_hash[:12],
                    event.previous_hash[:12],
                )
                return False
            if not event.verify():
                logger.error(
                    "Hash mismatch at event %s",
                    event.event_hash[:12],
                )
                return False
            prev_hash = event.event_hash
        return True

    def query(
        self,
        *,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        results: List[AuditEvent] = []
        for event in reversed(self._events):
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if case_id and event.case_id != case_id:
                continue
            if since and event.timestamp_utc < since:
                continue
            if until and event.timestamp_utc > until:
                continue
            results.append(event)
            if len(results) >= limit:
                break
        return results

    @property
    def length(self) -> int:
        return len(self._events)

    def _persist(self, event: AuditEvent) -> None:
        """Append event to storage file."""
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), default=str) + "\n")

    def _load(self) -> None:
        """Load events from storage file."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        with open(self._storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    event = AuditEvent.from_dict(d)
                    self._events.append(event)
                    self._last_hash = event.event_hash
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Skipping malformed audit entry: %s", exc)

    def export(self) -> List[Dict[str, Any]]:
        """Export the full audit log."""
        return [e.to_dict() for e in self._events]
