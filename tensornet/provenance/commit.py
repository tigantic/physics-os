"""
Field Commits
=============

Git-like commits for field states.

A commit captures:
- Field data hash
- Parent commit(s)
- Metadata (message, author, timestamp)
- Operation history
"""

from __future__ import annotations

import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union

from .merkle import compute_hash, MerkleNode, NodeType


# =============================================================================
# COMMIT METADATA
# =============================================================================

@dataclass
class CommitMetadata:
    """
    Metadata associated with a commit.
    """
    message: str = ""
    author: str = "system"
    timestamp: float = field(default_factory=time.time)
    
    # Operation tracking
    operation: Optional[str] = None  # e.g., "advect", "step", "project"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    compute_time_ms: float = 0.0
    memory_bytes: int = 0
    
    # Custom tags
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "author": self.author,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "parameters": self.parameters,
            "compute_time_ms": self.compute_time_ms,
            "memory_bytes": self.memory_bytes,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CommitMetadata':
        return cls(
            message=data.get("message", ""),
            author=data.get("author", "system"),
            timestamp=data.get("timestamp", 0.0),
            operation=data.get("operation"),
            parameters=data.get("parameters", {}),
            compute_time_ms=data.get("compute_time_ms", 0.0),
            memory_bytes=data.get("memory_bytes", 0),
            tags=data.get("tags", []),
        )


# =============================================================================
# FIELD COMMIT
# =============================================================================

@dataclass
class FieldCommit:
    """
    Immutable snapshot of a field state.
    
    Like a git commit, but for simulation fields.
    
    Contains:
    - Hash of the field data
    - Reference to parent commit(s)
    - Metadata (message, author, operation)
    - Field statistics at commit time
    
    Example:
        # Create initial commit
        commit0 = FieldCommit.create(field, message="Initial state")
        
        # Simulate
        field.step(dt=0.01)
        
        # Commit change
        commit1 = FieldCommit.create(
            field, 
            parents=[commit0.hash],
            message="After 1 step",
            operation="step",
            parameters={"dt": 0.01}
        )
    """
    hash: str
    data_hash: str  # Hash of the actual field data
    parent_hashes: Tuple[str, ...] = ()
    
    metadata: CommitMetadata = field(default_factory=CommitMetadata)
    
    # Field info at commit time
    field_shape: Tuple[int, ...] = ()
    field_dtype: str = "float32"
    field_stats: Dict[str, float] = field(default_factory=dict)
    
    # QTT compression info
    qtt_rank: Optional[int] = None
    qtt_compression_ratio: Optional[float] = None
    
    def __hash__(self):
        return hash(self.hash)
    
    def __eq__(self, other):
        if isinstance(other, FieldCommit):
            return self.hash == other.hash
        return False
    
    @property
    def short_hash(self) -> str:
        """First 8 characters of hash."""
        return self.hash[:8]
    
    @property
    def is_merge(self) -> bool:
        """Whether this commit has multiple parents."""
        return len(self.parent_hashes) > 1
    
    @property
    def is_root(self) -> bool:
        """Whether this is a root commit (no parents)."""
        return len(self.parent_hashes) == 0
    
    @classmethod
    def create(
        cls,
        field_data: Union[np.ndarray, torch.Tensor],
        parents: Optional[List[str]] = None,
        message: str = "",
        author: str = "system",
        operation: Optional[str] = None,
        parameters: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> 'FieldCommit':
        """
        Create a new commit from field data.
        
        Args:
            field_data: Field data to commit
            parents: Parent commit hashes
            message: Commit message
            author: Author name
            operation: Operation that produced this state
            parameters: Operation parameters
            tags: Custom tags
            
        Returns:
            New FieldCommit
        """
        # Convert to numpy if needed
        if isinstance(field_data, torch.Tensor):
            np_data = field_data.detach().cpu().numpy()
        else:
            np_data = field_data
        
        # Compute data hash
        data_hash = compute_hash(np_data)
        
        # Create metadata
        metadata = CommitMetadata(
            message=message,
            author=author,
            timestamp=time.time(),
            operation=operation,
            parameters=parameters or {},
            memory_bytes=np_data.nbytes,
            tags=tags or [],
        )
        
        # Compute field statistics
        field_stats = {
            "min": float(np.min(np_data)),
            "max": float(np.max(np_data)),
            "mean": float(np.mean(np_data)),
            "std": float(np.std(np_data)),
            "l2_norm": float(np.linalg.norm(np_data.flatten())),
        }
        
        # Parent hashes
        parent_hashes = tuple(parents) if parents else ()
        
        # Compute commit hash
        commit_content = {
            "data_hash": data_hash,
            "parent_hashes": parent_hashes,
            "metadata": metadata.to_dict(),
            "field_stats": field_stats,
        }
        commit_hash = compute_hash(commit_content)
        
        return cls(
            hash=commit_hash,
            data_hash=data_hash,
            parent_hashes=parent_hashes,
            metadata=metadata,
            field_shape=np_data.shape,
            field_dtype=str(np_data.dtype),
            field_stats=field_stats,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hash": self.hash,
            "data_hash": self.data_hash,
            "parent_hashes": list(self.parent_hashes),
            "metadata": self.metadata.to_dict(),
            "field_shape": list(self.field_shape),
            "field_dtype": self.field_dtype,
            "field_stats": self.field_stats,
            "qtt_rank": self.qtt_rank,
            "qtt_compression_ratio": self.qtt_compression_ratio,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FieldCommit':
        """Deserialize from dictionary."""
        return cls(
            hash=data["hash"],
            data_hash=data["data_hash"],
            parent_hashes=tuple(data.get("parent_hashes", [])),
            metadata=CommitMetadata.from_dict(data.get("metadata", {})),
            field_shape=tuple(data.get("field_shape", [])),
            field_dtype=data.get("field_dtype", "float32"),
            field_stats=data.get("field_stats", {}),
            qtt_rank=data.get("qtt_rank"),
            qtt_compression_ratio=data.get("qtt_compression_ratio"),
        )
    
    def to_merkle_node(self) -> MerkleNode:
        """Convert to MerkleNode."""
        return MerkleNode(
            hash=self.hash,
            node_type=NodeType.LEAF,
            parent_hashes=self.parent_hashes,
            data_hash=self.data_hash,
            timestamp=self.metadata.timestamp,
            metadata={
                "message": self.metadata.message,
                "author": self.metadata.author,
                "operation": self.metadata.operation,
            },
        )


# =============================================================================
# FACTORY
# =============================================================================

def make_commit(
    field_data: Union[np.ndarray, torch.Tensor],
    message: str = "",
    parent: Optional[Union[str, FieldCommit]] = None,
    **kwargs,
) -> FieldCommit:
    """
    Convenience function to create a commit.
    
    Args:
        field_data: Field to commit
        message: Commit message
        parent: Parent commit (hash or FieldCommit)
        **kwargs: Additional CommitMetadata fields
        
    Returns:
        New FieldCommit
    """
    parents = None
    if parent is not None:
        if isinstance(parent, FieldCommit):
            parents = [parent.hash]
        else:
            parents = [parent]
    
    return FieldCommit.create(
        field_data=field_data,
        parents=parents,
        message=message,
        **kwargs,
    )
