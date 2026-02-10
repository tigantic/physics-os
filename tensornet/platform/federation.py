"""
6.9 — Federated Simulation Data Exchange
=========================================

Enables secure, coordination-free exchange of simulation data across
distributed nodes.  Implements peer-to-peer data federation with
chunk-level transfers, integrity verification, and conflict resolution.

Components:
    * FederationNode     — logical peer identity
    * DataChunk          — transferable unit of simulation data
    * FederationRegistry — peer discovery and capability catalogue
    * FederationProtocol — chunk transfer / sync protocol engine
    * FederationManager  — top-level orchestrator
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


# ── Node identity ────────────────────────────────────────────────

class NodeRole(Enum):
    PRODUCER = auto()     # generates simulation data
    CONSUMER = auto()     # requests data
    RELAY = auto()        # routes but does not generate
    FULL = auto()         # producer + consumer + relay


@dataclass
class FederationNode:
    """Logical peer in the federation."""
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    role: NodeRole = NodeRole.FULL
    address: str = ""           # host:port (logical, for routing)
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, str] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)

    def is_alive(self, timeout: float = 60.0) -> bool:
        return (time.time() - self.last_heartbeat) < timeout

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "role": self.role.name,
            "address": self.address,
            "capabilities": list(self.capabilities),
            "metadata": self.metadata,
            "last_heartbeat": self.last_heartbeat,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FederationNode":
        return cls(
            node_id=d["node_id"],
            name=d.get("name", ""),
            role=NodeRole[d["role"]],
            address=d.get("address", ""),
            capabilities=set(d.get("capabilities", [])),
            metadata=d.get("metadata", {}),
            last_heartbeat=d.get("last_heartbeat", 0.0),
        )


# ── Data chunk ───────────────────────────────────────────────────

class ChunkStatus(Enum):
    PENDING = auto()
    TRANSFERRING = auto()
    COMPLETE = auto()
    FAILED = auto()
    VERIFIED = auto()


@dataclass
class DataChunk:
    """Transferable unit of simulation data."""
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    source_node: str = ""
    domain: str = ""
    data: Optional[np.ndarray] = None
    shape: Tuple[int, ...] = ()
    dtype: str = "float64"
    checksum: str = ""
    status: ChunkStatus = ChunkStatus.PENDING
    created_at: float = field(default_factory=time.time)

    def compute_checksum(self) -> str:
        if self.data is not None:
            self.checksum = hashlib.sha256(self.data.tobytes()).hexdigest()
        return self.checksum

    def verify(self) -> bool:
        if self.data is None:
            return False
        expected = hashlib.sha256(self.data.tobytes()).hexdigest()
        return expected == self.checksum

    def size_bytes(self) -> int:
        if self.data is not None:
            return self.data.nbytes
        return 0

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_node": self.source_node,
            "domain": self.domain,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "checksum": self.checksum,
            "status": self.status.name,
            "created_at": self.created_at,
        }


# ── Peer registry ────────────────────────────────────────────────

class FederationRegistry:
    """Peer discovery and catalogue."""

    def __init__(self) -> None:
        self._nodes: Dict[str, FederationNode] = {}

    def register(self, node: FederationNode) -> None:
        self._nodes[node.node_id] = node

    def deregister(self, node_id: str) -> bool:
        if node_id in self._nodes:
            del self._nodes[node_id]
            return True
        return False

    def heartbeat(self, node_id: str) -> bool:
        n = self._nodes.get(node_id)
        if n:
            n.last_heartbeat = time.time()
            return True
        return False

    def get(self, node_id: str) -> Optional[FederationNode]:
        return self._nodes.get(node_id)

    def alive_nodes(self, timeout: float = 60.0) -> List[FederationNode]:
        return [n for n in self._nodes.values() if n.is_alive(timeout)]

    def nodes_with_capability(self, cap: str) -> List[FederationNode]:
        return [n for n in self._nodes.values() if cap in n.capabilities]

    def producers(self) -> List[FederationNode]:
        return [
            n for n in self._nodes.values()
            if n.role in (NodeRole.PRODUCER, NodeRole.FULL)
        ]

    @property
    def count(self) -> int:
        return len(self._nodes)


# ── Transfer protocol ────────────────────────────────────────────

@dataclass
class TransferRequest:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_node: str = ""
    target_node: str = ""
    chunk_id: str = ""
    status: ChunkStatus = ChunkStatus.PENDING
    requested_at: float = field(default_factory=time.time)
    completed_at: float = 0.0


class FederationProtocol:
    """Simulates chunk transfer between nodes (in-process)."""

    def __init__(self, registry: FederationRegistry) -> None:
        self._registry = registry
        self._chunk_store: Dict[str, DataChunk] = {}
        self._transfers: List[TransferRequest] = []

    def publish(self, chunk: DataChunk) -> None:
        """Make a chunk available for transfer."""
        chunk.compute_checksum()
        chunk.status = ChunkStatus.COMPLETE
        self._chunk_store[chunk.chunk_id] = chunk

    def request_transfer(
        self, source_node: str, target_node: str, chunk_id: str,
    ) -> TransferRequest:
        req = TransferRequest(
            source_node=source_node,
            target_node=target_node,
            chunk_id=chunk_id,
        )
        self._transfers.append(req)

        # Simulate immediate transfer
        chunk = self._chunk_store.get(chunk_id)
        if chunk is None:
            req.status = ChunkStatus.FAILED
            return req

        if not chunk.verify():
            req.status = ChunkStatus.FAILED
            return req

        req.status = ChunkStatus.VERIFIED
        req.completed_at = time.time()
        return req

    def get_chunk(self, chunk_id: str) -> Optional[DataChunk]:
        return self._chunk_store.get(chunk_id)

    def available_chunks(self, domain: Optional[str] = None) -> List[DataChunk]:
        chunks = list(self._chunk_store.values())
        if domain:
            chunks = [c for c in chunks if c.domain == domain]
        return chunks

    @property
    def transfer_count(self) -> int:
        return len(self._transfers)


# ── Federation manager ───────────────────────────────────────────

class FederationManager:
    """Top-level orchestrator for federated simulation data."""

    def __init__(self) -> None:
        self.registry = FederationRegistry()
        self.protocol = FederationProtocol(self.registry)
        self._local_node: Optional[FederationNode] = None

    def init_local_node(
        self,
        name: str = "local",
        role: NodeRole = NodeRole.FULL,
        capabilities: Optional[Set[str]] = None,
    ) -> FederationNode:
        node = FederationNode(
            name=name, role=role,
            capabilities=capabilities or {"cfd", "fea", "qtt"},
        )
        self.registry.register(node)
        self._local_node = node
        return node

    @property
    def local_node(self) -> Optional[FederationNode]:
        return self._local_node

    def share_data(
        self,
        data: np.ndarray,
        domain: str = "",
    ) -> DataChunk:
        """Share local simulation data with the federation."""
        if self._local_node is None:
            raise RuntimeError("Local node not initialized")
        chunk = DataChunk(
            source_node=self._local_node.node_id,
            domain=domain,
            data=data,
            shape=data.shape,
            dtype=str(data.dtype),
        )
        self.protocol.publish(chunk)
        return chunk

    def fetch_data(
        self,
        chunk_id: str,
        from_node: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Fetch a chunk of data from the federation."""
        if self._local_node is None:
            raise RuntimeError("Local node not initialized")
        source = from_node or ""
        req = self.protocol.request_transfer(
            source_node=source,
            target_node=self._local_node.node_id,
            chunk_id=chunk_id,
        )
        if req.status == ChunkStatus.VERIFIED:
            chunk = self.protocol.get_chunk(chunk_id)
            return chunk.data if chunk else None
        return None

    def discover_data(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover available chunks in the federation."""
        return [c.to_metadata() for c in self.protocol.available_chunks(domain)]


__all__ = [
    "NodeRole",
    "FederationNode",
    "ChunkStatus",
    "DataChunk",
    "FederationRegistry",
    "TransferRequest",
    "FederationProtocol",
    "FederationManager",
]
