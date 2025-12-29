"""
Provenance - Merkle DAG for Field Lineage
==========================================

Layer 5 of the HyperTensor platform.

Provides immutable, verifiable history of field operations:
- Content-addressed storage
- Merkle DAG lineage tracking
- Git-like branching and commits
- Audit trails and reproducibility

Components:
    FieldCommit     - Immutable snapshot of a field state
    HistoryGraph    - DAG of field evolution
    ProvenanceStore - Content-addressed storage
    DiffEngine      - Field difference computation
    MerkleProof     - Verification proofs

Example:
    from tensornet.provenance import ProvenanceStore, FieldCommit
    
    store = ProvenanceStore("./history")
    
    # Commit field state
    commit = store.commit(field, message="Initial simulation")
    
    # Branch and modify
    branch = store.branch("experiment-1")
    field.step()
    commit2 = store.commit(field, message="After 100 steps")
    
    # View history
    for c in store.log():
        print(f"{c.hash[:8]}: {c.message}")
"""

from __future__ import annotations

# Core provenance
from .merkle import (
    MerkleNode,
    MerkleDAG,
    MerkleProof,
    compute_hash,
    verify_proof,
)

from .commit import (
    FieldCommit,
    CommitMetadata,
    make_commit,
)

from .history import (
    HistoryGraph,
    Branch,
    Tag,
    RefLog,
)

from .store import (
    ProvenanceStore,
    StoreConfig,
    ContentAddress,
    StorageBackend,
    MemoryBackend,
    FileSystemBackend,
)

from .diff import (
    FieldDiff,
    DiffEngine,
    DiffSummary,
    DiffType,
    compute_diff,
)

from .audit import (
    AuditTrail,
    AuditEvent,
    AuditQuery,
    EventType,
    EventSeverity,
)

__all__ = [
    # Merkle
    'MerkleNode',
    'MerkleDAG',
    'MerkleProof',
    'compute_hash',
    'verify_proof',
    
    # Commits
    'FieldCommit',
    'CommitMetadata',
    'make_commit',
    
    # History
    'HistoryGraph',
    'Branch',
    'Tag',
    'RefLog',
    
    # Store
    'ProvenanceStore',
    'StoreConfig',
    'ContentAddress',
    'StorageBackend',
    'MemoryBackend',
    'FileSystemBackend',
    
    # Diff
    'FieldDiff',
    'DiffEngine',
    'DiffSummary',
    'DiffType',
    'compute_diff',
    
    # Audit
    'AuditTrail',
    'AuditEvent',
    'AuditQuery',
    'EventType',
    'EventSeverity',
]

__version__ = '0.1.0'
