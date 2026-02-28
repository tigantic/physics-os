"""
Data Lineage DAG — Provenance tracking for multi-stage simulations.

Provides a directed acyclic graph for recording solver executions,
field transformations, coupling steps, inverse iterations, and QTT
compression events.  Every record carries wall-clock time, input/output
hashes, and parent references, enabling full reproducibility audit.

Classes:
    LineageEvent        — Typed enumeration of lineage event kinds.
    LineageNode         — Single node in the provenance DAG.
    LineageDAG          — The full provenance graph.
    LineageTracker      — Context-manager / API for recording lineage.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field as dc_field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════════
# Event Taxonomy
# ═══════════════════════════════════════════════════════════════════════════════


class LineageEvent(Enum):
    """Kind of event tracked in the lineage DAG."""

    FORWARD_SOLVE = auto()
    ADJOINT_SOLVE = auto()
    COUPLING_STEP = auto()
    QTT_COMPRESS = auto()
    QTT_DECOMPRESS = auto()
    TCI_DECOMPOSE = auto()
    INVERSE_ITERATION = auto()
    UQ_SAMPLE = auto()
    OPTIMIZATION_STEP = auto()
    FIELD_TRANSFORM = auto()
    CHECKPOINT = auto()
    CUSTOM = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# Lineage Node
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LineageNode:
    """
    Single provenance record in the lineage DAG.

    Attributes
    ----------
    node_id : str
        Unique identifier (UUID-4).
    event : LineageEvent
        What kind of operation this records.
    label : str
        Human-readable label.
    parent_ids : list[str]
        IDs of nodes that produced inputs for this operation.
    timestamp : float
        UNIX epoch time when the event started.
    elapsed_seconds : float
        Wall-clock duration of the operation.
    inputs_hash : str
        SHA-256 digest of the concatenated input tensor bytes.
    outputs_hash : str
        SHA-256 digest of the concatenated output tensor bytes.
    metadata : dict
        Arbitrary key-value pairs (solver name, dt, rank, etc.).
    """

    node_id: str
    event: LineageEvent
    label: str
    parent_ids: List[str]
    timestamp: float
    elapsed_seconds: float
    inputs_hash: str
    outputs_hash: str
    metadata: Dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "node_id": self.node_id,
            "event": self.event.name,
            "label": self.label,
            "parent_ids": self.parent_ids,
            "timestamp": self.timestamp,
            "elapsed_seconds": self.elapsed_seconds,
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LineageNode:
        """Deserialize from dict."""
        return cls(
            node_id=d["node_id"],
            event=LineageEvent[d["event"]],
            label=d["label"],
            parent_ids=d["parent_ids"],
            timestamp=d["timestamp"],
            elapsed_seconds=d["elapsed_seconds"],
            inputs_hash=d["inputs_hash"],
            outputs_hash=d["outputs_hash"],
            metadata=d.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Hash Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _tensor_hash(*tensors: Tensor) -> str:
    """SHA-256 hash of one or more tensors' byte representations."""
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _dict_hash(d: Dict[str, Tensor]) -> str:
    """Hash a dict of tensors in sorted-key order."""
    if not d:
        return hashlib.sha256(b"empty").hexdigest()
    tensors = [d[k] for k in sorted(d.keys())]
    return _tensor_hash(*tensors)


# ═══════════════════════════════════════════════════════════════════════════════
# Lineage DAG
# ═══════════════════════════════════════════════════════════════════════════════


class LineageDAG:
    """
    A directed acyclic graph of provenance nodes.

    Provides O(1) lookup by node ID, topological ordering, and
    serialization to/from JSON.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, LineageNode] = {}

    # ── Mutation ──────────────────────────────────────────────────────────

    def add(self, node: LineageNode) -> None:
        """Add a node to the DAG (validates parent existence)."""
        for pid in node.parent_ids:
            if pid not in self._nodes:
                raise KeyError(
                    f"Parent {pid} not in DAG; cannot add {node.node_id}."
                )
        self._nodes[node.node_id] = node

    # ── Queries ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get(self, node_id: str) -> LineageNode:
        return self._nodes[node_id]

    def roots(self) -> List[LineageNode]:
        """Return nodes with no parents."""
        return [n for n in self._nodes.values() if not n.parent_ids]

    def leaves(self) -> List[LineageNode]:
        """Return nodes with no children."""
        children = {pid for n in self._nodes.values() for pid in n.parent_ids}
        return [n for n in self._nodes.values() if n.node_id not in children]

    def ancestors(self, node_id: str) -> List[LineageNode]:
        """All transitive ancestors (topologically ordered, oldest first)."""
        visited: Dict[str, LineageNode] = {}
        stack = [node_id]
        while stack:
            nid = stack.pop()
            if nid not in self._nodes or nid in visited:
                continue
            node = self._nodes[nid]
            visited[nid] = node
            stack.extend(node.parent_ids)
        # Topological order
        result = list(visited.values())
        result.sort(key=lambda n: n.timestamp)
        # Exclude self
        return [n for n in result if n.node_id != node_id]

    def descendants(self, node_id: str) -> List[LineageNode]:
        """All transitive descendants (topologically ordered, oldest first)."""
        child_map: Dict[str, List[str]] = {nid: [] for nid in self._nodes}
        for n in self._nodes.values():
            for pid in n.parent_ids:
                child_map[pid].append(n.node_id)

        visited: Dict[str, LineageNode] = {}
        stack = [node_id]
        while stack:
            nid = stack.pop()
            if nid not in self._nodes or nid in visited:
                continue
            visited[nid] = self._nodes[nid]
            stack.extend(child_map.get(nid, []))

        result = list(visited.values())
        result.sort(key=lambda n: n.timestamp)
        return [n for n in result if n.node_id != node_id]

    def topological_order(self) -> List[LineageNode]:
        """All nodes in topological order (by timestamp)."""
        result = list(self._nodes.values())
        result.sort(key=lambda n: n.timestamp)
        return result

    def filter_by_event(self, event: LineageEvent) -> List[LineageNode]:
        """All nodes with a given event type."""
        return [n for n in self._nodes.values() if n.event == event]

    # ── Serialization ─────────────────────────────────────────────────────

    def to_json(self, indent: int = 2) -> str:
        """Serialize the DAG to JSON."""
        nodes = [n.to_dict() for n in self.topological_order()]
        return json.dumps({"lineage_dag": nodes}, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> LineageDAG:
        """Deserialize from JSON."""
        data = json.loads(json_str)
        dag = cls()
        for d in data["lineage_dag"]:
            dag.add(LineageNode.from_dict(d))
        return dag

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"LineageDAG: {len(self._nodes)} nodes"]
        event_counts: Dict[str, int] = {}
        total_time = 0.0
        for n in self._nodes.values():
            event_counts[n.event.name] = event_counts.get(n.event.name, 0) + 1
            total_time += n.elapsed_seconds
        for ev, ct in sorted(event_counts.items()):
            lines.append(f"  {ev}: {ct}")
        lines.append(f"  total_wall_time: {total_time:.3f}s")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Lineage Tracker (context-manager API)
# ═══════════════════════════════════════════════════════════════════════════════


class LineageTracker:
    """
    High-level API for recording lineage events into a DAG.

    Usage::

        dag = LineageDAG()
        tracker = LineageTracker(dag)

        # Record a forward solve
        with tracker.record(
            LineageEvent.FORWARD_SOLVE,
            label="Burgers RK4",
            inputs={"u": u0},
            parent_ids=[],
            metadata={"dt": 0.01, "steps": 100},
        ) as ctx:
            result = solver.solve(state, t_span, dt)
            ctx.set_outputs({"u": result.final_state.fields["u"].data})

        # ctx.node_id is available for downstream linkage
    """

    def __init__(self, dag: LineageDAG) -> None:
        self._dag = dag

    @property
    def dag(self) -> LineageDAG:
        return self._dag

    def record(
        self,
        event: LineageEvent,
        label: str,
        inputs: Dict[str, Tensor],
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _RecordContext:
        """Open a recording context."""
        return _RecordContext(
            dag=self._dag,
            event=event,
            label=label,
            inputs=inputs,
            parent_ids=parent_ids or [],
            metadata=metadata or {},
        )

    def record_instant(
        self,
        event: LineageEvent,
        label: str,
        inputs: Dict[str, Tensor],
        outputs: Dict[str, Tensor],
        elapsed_seconds: float = 0.0,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a lineage event that has already completed (non-ctx)."""
        node_id = uuid.uuid4().hex[:16]
        node = LineageNode(
            node_id=node_id,
            event=event,
            label=label,
            parent_ids=parent_ids or [],
            timestamp=time.time(),
            elapsed_seconds=elapsed_seconds,
            inputs_hash=_dict_hash(inputs),
            outputs_hash=_dict_hash(outputs),
            metadata=metadata or {},
        )
        self._dag.add(node)
        return node_id


class _RecordContext:
    """Context manager returned by ``LineageTracker.record``."""

    def __init__(
        self,
        dag: LineageDAG,
        event: LineageEvent,
        label: str,
        inputs: Dict[str, Tensor],
        parent_ids: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        self._dag = dag
        self._event = event
        self._label = label
        self._inputs_hash = _dict_hash(inputs)
        self._parent_ids = parent_ids
        self._metadata = metadata
        self._outputs_hash = hashlib.sha256(b"unset").hexdigest()
        self._node_id = uuid.uuid4().hex[:16]
        self._t0 = 0.0

    @property
    def node_id(self) -> str:
        """Access the node ID during the context."""
        return self._node_id

    def set_outputs(self, outputs: Dict[str, Tensor]) -> None:
        """Record output hash (call before exiting the context)."""
        self._outputs_hash = _dict_hash(outputs)

    def __enter__(self) -> _RecordContext:
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = time.time() - self._t0
        if exc_type is not None:
            self._metadata["error"] = str(exc_val)
        node = LineageNode(
            node_id=self._node_id,
            event=self._event,
            label=self._label,
            parent_ids=self._parent_ids,
            timestamp=self._t0,
            elapsed_seconds=elapsed,
            inputs_hash=self._inputs_hash,
            outputs_hash=self._outputs_hash,
            metadata=self._metadata,
        )
        self._dag.add(node)
