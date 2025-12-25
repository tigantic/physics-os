"""
Audit Trail
============

Immutable audit logs for compliance and debugging.

Features:
- Append-only event log
- Structured queries
- Tamper detection
- Export to various formats
"""

from __future__ import annotations

import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Iterator, Callable
from enum import Enum


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventType(Enum):
    """Type of audit event."""
    # Field operations
    FIELD_CREATE = "field.create"
    FIELD_UPDATE = "field.update"
    FIELD_DELETE = "field.delete"
    
    # Commits
    COMMIT_CREATE = "commit.create"
    COMMIT_CHECKOUT = "commit.checkout"
    
    # Branches
    BRANCH_CREATE = "branch.create"
    BRANCH_DELETE = "branch.delete"
    BRANCH_MERGE = "branch.merge"
    
    # Storage
    STORE_GC = "store.gc"
    STORE_EXPORT = "store.export"
    STORE_IMPORT = "store.import"
    
    # Access
    ACCESS_READ = "access.read"
    ACCESS_WRITE = "access.write"
    
    # System
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    
    # Custom
    CUSTOM = "custom"


class EventSeverity(Enum):
    """Severity level of event."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# AUDIT EVENT
# =============================================================================

@dataclass
class AuditEvent:
    """
    Single audit event.
    
    Immutable record of something that happened.
    """
    id: str  # Unique event ID
    timestamp: float
    event_type: EventType
    severity: EventSeverity = EventSeverity.INFO
    
    # Context
    actor: str = "system"  # Who/what caused this
    target: Optional[str] = None  # What was affected
    
    # Details
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Integrity
    previous_hash: Optional[str] = None  # Hash of previous event
    hash: str = ""  # Hash of this event
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
        if not self.hash:
            self.hash = self._compute_hash()
    
    def _generate_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _compute_hash(self) -> str:
        """Compute hash of event content."""
        content = {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "actor": self.actor,
            "target": self.target,
            "message": self.message,
            "data": self.data,
            "previous_hash": self.previous_hash,
        }
        json_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "actor": self.actor,
            "target": self.target,
            "message": self.message,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AuditEvent':
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            event_type=EventType(data["event_type"]),
            severity=EventSeverity(data.get("severity", "info")),
            actor=data.get("actor", "system"),
            target=data.get("target"),
            message=data.get("message", ""),
            data=data.get("data", {}),
            previous_hash=data.get("previous_hash"),
            hash=data.get("hash", ""),
        )
    
    def verify(self) -> bool:
        """Verify event hash is correct."""
        expected = self._compute_hash()
        return self.hash == expected


# =============================================================================
# AUDIT QUERY
# =============================================================================

@dataclass
class AuditQuery:
    """
    Query for filtering audit events.
    """
    # Time range
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Type filters
    event_types: Optional[List[EventType]] = None
    severities: Optional[List[EventSeverity]] = None
    
    # Actor/target filters
    actor: Optional[str] = None
    target: Optional[str] = None
    
    # Text search
    message_contains: Optional[str] = None
    
    # Limits
    limit: int = 100
    offset: int = 0
    
    def matches(self, event: AuditEvent) -> bool:
        """Check if event matches query."""
        # Time range
        if self.start_time and event.timestamp < self.start_time:
            return False
        if self.end_time and event.timestamp > self.end_time:
            return False
        
        # Type filters
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.severities and event.severity not in self.severities:
            return False
        
        # Actor/target
        if self.actor and event.actor != self.actor:
            return False
        if self.target and event.target != self.target:
            return False
        
        # Text search
        if self.message_contains:
            if self.message_contains.lower() not in event.message.lower():
                return False
        
        return True


# =============================================================================
# AUDIT TRAIL
# =============================================================================

class AuditTrail:
    """
    Append-only audit trail with tamper detection.
    
    Events are chained by hash, making tampering detectable.
    
    Example:
        audit = AuditTrail()
        
        # Log events
        audit.log(
            event_type=EventType.FIELD_CREATE,
            message="Created velocity field",
            target="velocity",
            data={"shape": (256, 256)}
        )
        
        # Query events
        for event in audit.query(AuditQuery(
            event_types=[EventType.FIELD_CREATE],
            limit=10
        )):
            print(f"{event.timestamp}: {event.message}")
        
        # Verify integrity
        valid, issues = audit.verify()
        if not valid:
            print(f"Tampering detected: {issues}")
    """
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self._events: List[AuditEvent] = []
        self._last_hash: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self._events)
    
    def log(
        self,
        event_type: EventType,
        message: str = "",
        severity: EventSeverity = EventSeverity.INFO,
        actor: str = "system",
        target: Optional[str] = None,
        data: Optional[Dict] = None,
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            message: Human-readable message
            severity: Event severity
            actor: Who/what caused this
            target: What was affected
            data: Additional structured data
            
        Returns:
            Created AuditEvent
        """
        event = AuditEvent(
            id="",  # Will be generated
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            actor=actor,
            target=target,
            message=message,
            data=data or {},
            previous_hash=self._last_hash,
        )
        
        self._events.append(event)
        self._last_hash = event.hash
        
        # Trim if over limit
        while len(self._events) > self.max_events:
            self._events.pop(0)
        
        return event
    
    def log_error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs,
    ) -> AuditEvent:
        """Convenience method for logging errors."""
        data = kwargs.pop("data", {})
        if exception:
            data["exception_type"] = type(exception).__name__
            data["exception_message"] = str(exception)
        
        return self.log(
            event_type=EventType.SYSTEM_ERROR,
            message=message,
            severity=EventSeverity.ERROR,
            data=data,
            **kwargs,
        )
    
    def query(self, query: AuditQuery) -> Iterator[AuditEvent]:
        """
        Query events.
        
        Args:
            query: AuditQuery specifying filters
            
        Yields:
            Matching AuditEvent objects
        """
        count = 0
        skipped = 0
        
        # Iterate in reverse (newest first)
        for event in reversed(self._events):
            if not query.matches(event):
                continue
            
            # Handle offset
            if skipped < query.offset:
                skipped += 1
                continue
            
            yield event
            count += 1
            
            if count >= query.limit:
                break
    
    def get_recent(self, n: int = 10) -> List[AuditEvent]:
        """Get n most recent events."""
        return list(self._events[-n:])[::-1]
    
    def get_by_target(self, target: str) -> List[AuditEvent]:
        """Get all events for a target."""
        return [e for e in self._events if e.target == target]
    
    def get_by_actor(self, actor: str) -> List[AuditEvent]:
        """Get all events by an actor."""
        return [e for e in self._events if e.actor == actor]
    
    def verify(self) -> Tuple[bool, List[str]]:
        """
        Verify audit trail integrity.
        
        Checks:
        - Each event's hash is correct
        - Hash chain is unbroken
        
        Returns:
            (is_valid, list of issues)
        """
        issues = []
        
        expected_prev = None
        for i, event in enumerate(self._events):
            # Verify event hash
            if not event.verify():
                issues.append(f"Event {i} ({event.id}): hash mismatch")
            
            # Verify chain
            if i > 0 and event.previous_hash != expected_prev:
                issues.append(f"Event {i} ({event.id}): chain broken")
            
            expected_prev = event.hash
        
        return len(issues) == 0, issues
    
    def export_json(self, path: str):
        """Export audit trail to JSON file."""
        data = {
            "version": "1.0",
            "exported_at": time.time(),
            "event_count": len(self._events),
            "events": [e.to_dict() for e in self._events],
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def import_json(cls, path: str) -> 'AuditTrail':
        """Import audit trail from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        trail = cls()
        for event_data in data.get("events", []):
            event = AuditEvent.from_dict(event_data)
            trail._events.append(event)
        
        if trail._events:
            trail._last_hash = trail._events[-1].hash
        
        return trail
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_events": self.max_events,
            "events": [e.to_dict() for e in self._events],
            "last_hash": self._last_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AuditTrail':
        """Deserialize from dictionary."""
        trail = cls(max_events=data.get("max_events", 10000))
        
        for event_data in data.get("events", []):
            event = AuditEvent.from_dict(event_data)
            trail._events.append(event)
        
        trail._last_hash = data.get("last_hash")
        return trail
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics."""
        if not self._events:
            return {
                "event_count": 0,
                "time_span": 0,
            }
        
        # Count by type
        type_counts = {}
        for event in self._events:
            type_name = event.event_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for event in self._events:
            sev = event.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        return {
            "event_count": len(self._events),
            "time_span": self._events[-1].timestamp - self._events[0].timestamp,
            "oldest": self._events[0].timestamp,
            "newest": self._events[-1].timestamp,
            "by_type": type_counts,
            "by_severity": severity_counts,
        }


# Type alias for compatibility
Tuple = tuple
