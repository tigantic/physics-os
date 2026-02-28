"""
6.11 — Simulation Replay Engine
================================

Deterministic replay of simulation runs from checkpoint/seed state.
Supports step-by-step re-execution, comparison with original, and
branch-off for what-if scenarios.

Components:
    * ReplayEvent   — recorded simulation event
    * ReplayLog     — ordered event log with persistence
    * ReplayEngine  — deterministic re-execution and comparison
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Event model ──────────────────────────────────────────────────

class EventType(Enum):
    INIT = auto()
    STEP = auto()
    CHECKPOINT = auto()
    PARAM_CHANGE = auto()
    BRANCH = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class ReplayEvent:
    """Single recorded simulation event."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: EventType = EventType.STEP
    step: int = 0
    timestamp: float = field(default_factory=time.time)
    state_hash: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    # Lightweight snapshot (for small states) or reference to checkpoint
    state_snapshot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "step": self.step,
            "timestamp": self.timestamp,
            "state_hash": self.state_hash,
            "params": self.params,
            "metadata": self.metadata,
        }
        if self.state_snapshot is not None:
            serialisable: Dict[str, Any] = {}
            for k, v in self.state_snapshot.items():
                if isinstance(v, np.ndarray):
                    serialisable[k] = {
                        "__ndarray__": True,
                        "data": v.tolist(),
                        "dtype": str(v.dtype),
                        "shape": list(v.shape),
                    }
                else:
                    serialisable[k] = v
            d["state_snapshot"] = serialisable
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReplayEvent":
        snap = d.get("state_snapshot")
        if snap is not None:
            restored: Dict[str, Any] = {}
            for k, v in snap.items():
                if isinstance(v, dict) and v.get("__ndarray__"):
                    restored[k] = np.array(v["data"], dtype=v["dtype"]).reshape(v["shape"])
                else:
                    restored[k] = v
            snap = restored
        return cls(
            event_id=d["event_id"],
            event_type=EventType[d["event_type"]],
            step=d["step"],
            timestamp=d.get("timestamp", 0.0),
            state_hash=d.get("state_hash", ""),
            params=d.get("params", {}),
            metadata=d.get("metadata", {}),
            state_snapshot=snap,
        )


# ── Replay log ───────────────────────────────────────────────────

class ReplayLog:
    """Ordered, persistent event log for one simulation run."""

    def __init__(self, run_id: Optional[str] = None) -> None:
        self.run_id: str = run_id or uuid.uuid4().hex[:12]
        self._events: List[ReplayEvent] = []

    def append(self, event: ReplayEvent) -> None:
        self._events.append(event)

    @property
    def length(self) -> int:
        return len(self._events)

    def event_at(self, index: int) -> ReplayEvent:
        return self._events[index]

    def events_of_type(self, t: EventType) -> List[ReplayEvent]:
        return [e for e in self._events if e.event_type == t]

    def last_checkpoint(self) -> Optional[ReplayEvent]:
        for e in reversed(self._events):
            if e.event_type == EventType.CHECKPOINT:
                return e
        return None

    def slice(self, start_step: int, end_step: int) -> List[ReplayEvent]:
        return [e for e in self._events if start_step <= e.step <= end_step]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "events": [e.to_dict() for e in self._events],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ReplayLog":
        data = json.loads(Path(path).read_text())
        log = cls(run_id=data["run_id"])
        for ed in data["events"]:
            log.append(ReplayEvent.from_dict(ed))
        return log

    def clear(self) -> None:
        self._events.clear()


# ── State hasher ─────────────────────────────────────────────────

def _hash_state(state: Dict[str, Any]) -> str:
    """Deterministic hash of a simulation state dict."""
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        h.update(key.encode())
        val = state[key]
        if isinstance(val, np.ndarray):
            h.update(val.tobytes())
        elif isinstance(val, (int, float)):
            h.update(str(val).encode())
        else:
            h.update(json.dumps(val, sort_keys=True, default=str).encode())
    return h.hexdigest()


# ── Replay engine ────────────────────────────────────────────────

StepFunction = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


class ReplayEngine:
    """Deterministic re-execution and comparison engine.

    Usage::

        engine = ReplayEngine(step_fn=my_solver_step)
        engine.record_init(initial_state, params)
        for i in range(100):
            state = my_solver_step(state, params)
            engine.record_step(i, state, params)

        # Later: replay and compare
        replayed = engine.replay(log)
        diffs = engine.compare(original_log, replayed_log)
    """

    def __init__(self, step_fn: Optional[StepFunction] = None) -> None:
        self._step_fn = step_fn
        self._recording_log: Optional[ReplayLog] = None

    # ── Recording ─────────────────────────────────────────────

    def start_recording(self, run_id: Optional[str] = None) -> ReplayLog:
        self._recording_log = ReplayLog(run_id)
        return self._recording_log

    def record_init(
        self,
        state: Dict[str, Any],
        params: Dict[str, Any],
    ) -> ReplayEvent:
        if self._recording_log is None:
            raise RuntimeError("Recording not started")
        snapshot = self._make_snapshot(state)
        event = ReplayEvent(
            event_type=EventType.INIT,
            step=0,
            state_hash=_hash_state(state),
            params=copy.deepcopy(params),
            state_snapshot=snapshot,
        )
        self._recording_log.append(event)
        return event

    def record_step(
        self,
        step: int,
        state: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        snapshot: bool = False,
    ) -> ReplayEvent:
        if self._recording_log is None:
            raise RuntimeError("Recording not started")
        event = ReplayEvent(
            event_type=EventType.STEP,
            step=step,
            state_hash=_hash_state(state),
            params=copy.deepcopy(params) if params else {},
            state_snapshot=self._make_snapshot(state) if snapshot else None,
        )
        self._recording_log.append(event)
        return event

    def record_checkpoint(
        self, step: int, state: Dict[str, Any],
    ) -> ReplayEvent:
        if self._recording_log is None:
            raise RuntimeError("Recording not started")
        event = ReplayEvent(
            event_type=EventType.CHECKPOINT,
            step=step,
            state_hash=_hash_state(state),
            state_snapshot=self._make_snapshot(state),
        )
        self._recording_log.append(event)
        return event

    def stop_recording(self) -> Optional[ReplayLog]:
        log = self._recording_log
        if log is not None:
            log.append(ReplayEvent(event_type=EventType.COMPLETE, step=-1))
        self._recording_log = None
        return log

    # ── Replay ────────────────────────────────────────────────

    def replay(
        self,
        log: ReplayLog,
        max_steps: Optional[int] = None,
    ) -> ReplayLog:
        """Re-execute a recorded simulation from a replay log.

        Requires a step_fn and that INIT events carry a state_snapshot.
        """
        if self._step_fn is None:
            raise RuntimeError("No step function provided for replay")

        inits = log.events_of_type(EventType.INIT)
        if not inits:
            raise ValueError("Log contains no INIT event")
        init_event = inits[0]
        if init_event.state_snapshot is None:
            raise ValueError("INIT event has no state snapshot")

        state = copy.deepcopy(init_event.state_snapshot)
        params = copy.deepcopy(init_event.params)

        replayed = ReplayLog(run_id=f"{log.run_id}_replay")
        replayed.append(ReplayEvent(
            event_type=EventType.INIT,
            step=0,
            state_hash=_hash_state(state),
            params=params,
            state_snapshot=self._make_snapshot(state),
        ))

        steps = log.events_of_type(EventType.STEP)
        if max_steps is not None:
            steps = steps[:max_steps]

        for step_event in steps:
            # Apply parameter changes if recorded
            if step_event.params:
                params.update(step_event.params)

            state = self._step_fn(state, params)
            replayed.append(ReplayEvent(
                event_type=EventType.STEP,
                step=step_event.step,
                state_hash=_hash_state(state),
                params=copy.deepcopy(params),
            ))

        replayed.append(ReplayEvent(event_type=EventType.COMPLETE, step=-1))
        return replayed

    # ── Comparison ────────────────────────────────────────────

    def compare(
        self, original: ReplayLog, replayed: ReplayLog,
    ) -> Dict[str, Any]:
        """Compare two logs for deterministic equivalence."""
        orig_steps = original.events_of_type(EventType.STEP)
        repl_steps = replayed.events_of_type(EventType.STEP)

        min_len = min(len(orig_steps), len(repl_steps))
        mismatches: List[int] = []
        for i in range(min_len):
            if orig_steps[i].state_hash != repl_steps[i].state_hash:
                mismatches.append(orig_steps[i].step)

        return {
            "original_steps": len(orig_steps),
            "replayed_steps": len(repl_steps),
            "matched": min_len - len(mismatches),
            "mismatches": mismatches,
            "deterministic": len(mismatches) == 0 and len(orig_steps) == len(repl_steps),
        }

    # ── Branch-off (what-if scenario) ─────────────────────────

    def branch_from(
        self,
        log: ReplayLog,
        branch_step: int,
        new_params: Dict[str, Any],
        additional_steps: int = 10,
    ) -> ReplayLog:
        """Branch off from a recorded step with modified parameters."""
        if self._step_fn is None:
            raise RuntimeError("No step function provided for branching")

        # Find the closest checkpoint or snapshot at/before branch_step
        snapshot_event: Optional[ReplayEvent] = None
        for event in log.events_of_type(EventType.CHECKPOINT):
            if event.step <= branch_step:
                snapshot_event = event
        if snapshot_event is None:
            inits = log.events_of_type(EventType.INIT)
            if inits and inits[0].state_snapshot is not None:
                snapshot_event = inits[0]

        if snapshot_event is None or snapshot_event.state_snapshot is None:
            raise ValueError("No usable snapshot found at or before branch step")

        state = copy.deepcopy(snapshot_event.state_snapshot)
        params = copy.deepcopy(snapshot_event.params)

        # Advance to branch_step
        steps_to_run = branch_step - snapshot_event.step
        for _ in range(steps_to_run):
            state = self._step_fn(state, params)

        # Apply new params and continue
        params.update(new_params)
        branch_log = ReplayLog(run_id=f"{log.run_id}_branch_{branch_step}")
        branch_log.append(ReplayEvent(
            event_type=EventType.BRANCH,
            step=branch_step,
            state_hash=_hash_state(state),
            params=copy.deepcopy(params),
            state_snapshot=self._make_snapshot(state),
        ))

        for i in range(additional_steps):
            state = self._step_fn(state, params)
            branch_log.append(ReplayEvent(
                event_type=EventType.STEP,
                step=branch_step + i + 1,
                state_hash=_hash_state(state),
            ))

        branch_log.append(ReplayEvent(event_type=EventType.COMPLETE, step=-1))
        return branch_log

    # ── Internal ──────────────────────────────────────────────

    @staticmethod
    def _make_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-copy state, converting numpy arrays."""
        snap: Dict[str, Any] = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                snap[k] = v.copy()
            else:
                snap[k] = copy.deepcopy(v)
        return snap


__all__ = [
    "EventType",
    "ReplayEvent",
    "ReplayLog",
    "ReplayEngine",
]
