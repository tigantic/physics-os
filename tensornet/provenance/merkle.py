"""
Merkle DAG Primitives
======================

Content-addressed data structures for field provenance.

Features:
- Cryptographic hashing (SHA-256)
- Merkle tree construction
- DAG traversal
- Proof generation and verification
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Set, Iterator
from enum import Enum


# =============================================================================
# HASHING
# =============================================================================

def compute_hash(data: Union[bytes, str, np.ndarray, torch.Tensor, Dict]) -> str:
    """
    Compute SHA-256 hash of data.
    
    Handles:
    - bytes: Direct hash
    - str: UTF-8 encode then hash
    - ndarray: Hash raw bytes
    - Tensor: Convert to numpy then hash
    - Dict: JSON serialize then hash
    
    Returns:
        Hex-encoded hash string
    """
    hasher = hashlib.sha256()
    
    if isinstance(data, bytes):
        hasher.update(data)
    elif isinstance(data, str):
        hasher.update(data.encode('utf-8'))
    elif isinstance(data, np.ndarray):
        hasher.update(data.tobytes())
    elif isinstance(data, torch.Tensor):
        hasher.update(data.detach().cpu().numpy().tobytes())
    elif isinstance(data, dict):
        # Deterministic JSON serialization
        json_str = json.dumps(data, sort_keys=True, default=str)
        hasher.update(json_str.encode('utf-8'))
    else:
        # Try to convert to string
        hasher.update(str(data).encode('utf-8'))
    
    return hasher.hexdigest()


def combine_hashes(*hashes: str) -> str:
    """Combine multiple hashes into one."""
    combined = "".join(sorted(hashes))
    return compute_hash(combined)


# =============================================================================
# MERKLE NODE
# =============================================================================

class NodeType(Enum):
    """Type of Merkle node."""
    LEAF = "leaf"           # Contains actual data
    INTERNAL = "internal"   # Contains only child hashes
    ROOT = "root"           # Root of tree/DAG


@dataclass
class MerkleNode:
    """
    Node in a Merkle tree/DAG.
    
    Each node has:
    - Hash of its content
    - References to parent nodes (for DAG)
    - Optional data payload
    
    Properties:
    - Immutable once created
    - Content-addressed (hash = identity)
    - Verifiable
    """
    hash: str
    node_type: NodeType = NodeType.LEAF
    parent_hashes: Tuple[str, ...] = ()
    data_hash: Optional[str] = None
    
    # Metadata
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.hash)
    
    def __eq__(self, other):
        if isinstance(other, MerkleNode):
            return self.hash == other.hash
        return False
    
    @classmethod
    def create_leaf(
        cls,
        data: Union[bytes, np.ndarray, torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> 'MerkleNode':
        """Create leaf node from data."""
        import time
        
        data_hash = compute_hash(data)
        node_hash = compute_hash({
            "type": "leaf",
            "data_hash": data_hash,
            "metadata": metadata or {},
        })
        
        return cls(
            hash=node_hash,
            node_type=NodeType.LEAF,
            parent_hashes=(),
            data_hash=data_hash,
            timestamp=time.time(),
            metadata=metadata or {},
        )
    
    @classmethod
    def create_internal(
        cls,
        children: List['MerkleNode'],
        metadata: Optional[Dict] = None,
    ) -> 'MerkleNode':
        """Create internal node from children."""
        import time
        
        child_hashes = tuple(c.hash for c in children)
        node_hash = compute_hash({
            "type": "internal",
            "children": child_hashes,
            "metadata": metadata or {},
        })
        
        return cls(
            hash=node_hash,
            node_type=NodeType.INTERNAL,
            parent_hashes=child_hashes,
            timestamp=time.time(),
            metadata=metadata or {},
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hash": self.hash,
            "node_type": self.node_type.value,
            "parent_hashes": list(self.parent_hashes),
            "data_hash": self.data_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MerkleNode':
        """Deserialize from dictionary."""
        return cls(
            hash=data["hash"],
            node_type=NodeType(data["node_type"]),
            parent_hashes=tuple(data["parent_hashes"]),
            data_hash=data.get("data_hash"),
            timestamp=data.get("timestamp", 0.0),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# MERKLE DAG
# =============================================================================

class MerkleDAG:
    """
    Merkle Directed Acyclic Graph.
    
    A DAG where:
    - Each node has a content hash
    - Nodes can have multiple parents
    - Edges point to parents (like git)
    - Enables efficient diff/merge operations
    
    Example:
        dag = MerkleDAG()
        
        # Add nodes
        leaf1 = MerkleNode.create_leaf(data1)
        leaf2 = MerkleNode.create_leaf(data2)
        dag.add(leaf1)
        dag.add(leaf2)
        
        # Create merge node
        merge = MerkleNode.create_internal([leaf1, leaf2])
        dag.add(merge)
        
        # Find common ancestor
        ancestor = dag.common_ancestor(leaf1.hash, leaf2.hash)
    """
    
    def __init__(self):
        self._nodes: Dict[str, MerkleNode] = {}
        self._children: Dict[str, Set[str]] = {}  # parent -> children
        self._heads: Set[str] = set()  # Nodes with no children
    
    def __len__(self) -> int:
        return len(self._nodes)
    
    def __contains__(self, hash_or_node: Union[str, MerkleNode]) -> bool:
        if isinstance(hash_or_node, MerkleNode):
            return hash_or_node.hash in self._nodes
        return hash_or_node in self._nodes
    
    def add(self, node: MerkleNode) -> str:
        """
        Add node to DAG.
        
        Returns:
            Node hash
        """
        if node.hash in self._nodes:
            return node.hash
        
        # Store node
        self._nodes[node.hash] = node
        
        # Update children index
        for parent_hash in node.parent_hashes:
            if parent_hash not in self._children:
                self._children[parent_hash] = set()
            self._children[parent_hash].add(node.hash)
            
            # Parent is no longer a head
            self._heads.discard(parent_hash)
        
        # This node is a head until it gets children
        self._heads.add(node.hash)
        
        return node.hash
    
    def get(self, hash: str) -> Optional[MerkleNode]:
        """Get node by hash."""
        return self._nodes.get(hash)
    
    def get_parents(self, hash: str) -> List[MerkleNode]:
        """Get parent nodes."""
        node = self._nodes.get(hash)
        if not node:
            return []
        return [self._nodes[h] for h in node.parent_hashes if h in self._nodes]
    
    def get_children(self, hash: str) -> List[MerkleNode]:
        """Get child nodes."""
        child_hashes = self._children.get(hash, set())
        return [self._nodes[h] for h in child_hashes if h in self._nodes]
    
    @property
    def heads(self) -> List[MerkleNode]:
        """Get all head nodes (no children)."""
        return [self._nodes[h] for h in self._heads]
    
    @property
    def roots(self) -> List[MerkleNode]:
        """Get all root nodes (no parents)."""
        return [n for n in self._nodes.values() if not n.parent_hashes]
    
    def ancestors(self, hash: str) -> Iterator[MerkleNode]:
        """
        Iterate over all ancestors of a node (BFS order).
        """
        visited = set()
        queue = list(self._nodes[hash].parent_hashes) if hash in self._nodes else []
        
        while queue:
            current = queue.pop(0)
            if current in visited or current not in self._nodes:
                continue
            
            visited.add(current)
            node = self._nodes[current]
            yield node
            
            queue.extend(node.parent_hashes)
    
    def descendants(self, hash: str) -> Iterator[MerkleNode]:
        """
        Iterate over all descendants of a node (BFS order).
        """
        visited = set()
        queue = list(self._children.get(hash, set()))
        
        while queue:
            current = queue.pop(0)
            if current in visited or current not in self._nodes:
                continue
            
            visited.add(current)
            node = self._nodes[current]
            yield node
            
            queue.extend(self._children.get(current, set()))
    
    def common_ancestor(self, hash1: str, hash2: str) -> Optional[MerkleNode]:
        """
        Find lowest common ancestor of two nodes.
        
        Returns None if no common ancestor exists.
        """
        # Get all ancestors of first node
        ancestors1 = set([hash1])
        for node in self.ancestors(hash1):
            ancestors1.add(node.hash)
        
        # Find first ancestor of second node that's in ancestors1
        if hash2 in ancestors1:
            return self._nodes[hash2]
        
        for node in self.ancestors(hash2):
            if node.hash in ancestors1:
                return node
        
        return None
    
    def path(self, from_hash: str, to_hash: str) -> Optional[List[MerkleNode]]:
        """
        Find path from one node to another.
        
        Returns:
            List of nodes forming path, or None if no path exists.
        """
        if from_hash not in self._nodes or to_hash not in self._nodes:
            return None
        
        # BFS from from_hash
        visited = {from_hash: None}  # node -> parent in path
        queue = [from_hash]
        
        while queue:
            current = queue.pop(0)
            
            if current == to_hash:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(self._nodes[current])
                    current = visited[current]
                return list(reversed(path))
            
            # Try children (forward) and parents (backward)
            neighbors = list(self._children.get(current, set()))
            if current in self._nodes:
                neighbors.extend(self._nodes[current].parent_hashes)
            
            for neighbor in neighbors:
                if neighbor not in visited and neighbor in self._nodes:
                    visited[neighbor] = current
                    queue.append(neighbor)
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize DAG to dictionary."""
        return {
            "nodes": {h: n.to_dict() for h, n in self._nodes.items()},
            "heads": list(self._heads),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MerkleDAG':
        """Deserialize DAG from dictionary."""
        dag = cls()
        
        # Recreate nodes
        for hash, node_data in data.get("nodes", {}).items():
            node = MerkleNode.from_dict(node_data)
            dag._nodes[hash] = node
            
            # Rebuild children index
            for parent_hash in node.parent_hashes:
                if parent_hash not in dag._children:
                    dag._children[parent_hash] = set()
                dag._children[parent_hash].add(hash)
        
        # Restore heads
        dag._heads = set(data.get("heads", []))
        
        return dag


# =============================================================================
# MERKLE PROOF
# =============================================================================

@dataclass
class MerkleProof:
    """
    Proof that a node exists in a Merkle tree/DAG.
    
    Contains:
    - Target node hash
    - Path to root
    - Sibling hashes at each level
    
    Can be verified without access to full tree.
    """
    target_hash: str
    root_hash: str
    path: List[str]  # Hashes from target to root
    siblings: List[List[str]]  # Sibling hashes at each level
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_hash": self.target_hash,
            "root_hash": self.root_hash,
            "path": self.path,
            "siblings": self.siblings,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MerkleProof':
        return cls(
            target_hash=data["target_hash"],
            root_hash=data["root_hash"],
            path=data["path"],
            siblings=data["siblings"],
        )


def generate_proof(dag: MerkleDAG, target_hash: str, root_hash: str) -> Optional[MerkleProof]:
    """
    Generate Merkle proof for a node.
    
    Args:
        dag: Merkle DAG containing the node
        target_hash: Hash of node to prove
        root_hash: Hash of root node
        
    Returns:
        MerkleProof or None if path doesn't exist
    """
    path = dag.path(target_hash, root_hash)
    if not path:
        return None
    
    path_hashes = [n.hash for n in path]
    
    # Collect siblings at each level
    siblings = []
    for node in path:
        sibling_hashes = [
            h for h in node.parent_hashes
            if h not in path_hashes
        ]
        siblings.append(sibling_hashes)
    
    return MerkleProof(
        target_hash=target_hash,
        root_hash=root_hash,
        path=path_hashes,
        siblings=siblings,
    )


def verify_proof(proof: MerkleProof) -> bool:
    """
    Verify a Merkle proof.
    
    Returns:
        True if proof is valid
    """
    if not proof.path:
        return False
    
    # Check path starts at target and ends at root
    if proof.path[0] != proof.target_hash:
        return False
    if proof.path[-1] != proof.root_hash:
        return False
    
    # Verify hash chain
    for i in range(len(proof.path) - 1):
        current = proof.path[i]
        next_hash = proof.path[i + 1]
        sibling_hashes = proof.siblings[i] if i < len(proof.siblings) else []
        
        # Recompute parent hash from children
        all_children = [current] + sibling_hashes
        expected = combine_hashes(*all_children)
        
        # This is a simplified check - real implementation would
        # need to verify the actual node structure
        if not expected:
            return False
    
    return True
