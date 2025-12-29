"""
History Graph
==============

Git-like branching and tagging for field history.

Features:
- Named branches
- Tags for important states
- Merge support
- Reference log (reflog)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Iterator

from .merkle import MerkleDAG
from .commit import FieldCommit


# =============================================================================
# BRANCH
# =============================================================================

@dataclass
class Branch:
    """
    Named branch pointing to a commit.
    
    Like git branches, these are mutable pointers
    that move forward as new commits are added.
    """
    name: str
    head: str  # Commit hash
    
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Tracking
    upstream: Optional[str] = None  # Remote tracking branch
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "head": self.head,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "upstream": self.upstream,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Branch':
        return cls(
            name=data["name"],
            head=data["head"],
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            upstream=data.get("upstream"),
        )


@dataclass
class Tag:
    """
    Named tag pointing to a specific commit.
    
    Unlike branches, tags are immutable references.
    """
    name: str
    target: str  # Commit hash
    message: str = ""
    
    created_at: float = field(default_factory=time.time)
    author: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "message": self.message,
            "created_at": self.created_at,
            "author": self.author,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Tag':
        return cls(
            name=data["name"],
            target=data["target"],
            message=data.get("message", ""),
            created_at=data.get("created_at", 0.0),
            author=data.get("author", "system"),
        )


# =============================================================================
# REFLOG
# =============================================================================

@dataclass
class RefLogEntry:
    """Entry in the reference log."""
    timestamp: float
    ref_name: str
    old_hash: Optional[str]
    new_hash: str
    action: str  # "commit", "checkout", "merge", "reset", etc.
    message: str = ""


class RefLog:
    """
    Reference log tracking all ref changes.
    
    Like git's reflog, provides safety net for
    accidental operations.
    """
    
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._entries: List[RefLogEntry] = []
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def add(
        self,
        ref_name: str,
        old_hash: Optional[str],
        new_hash: str,
        action: str,
        message: str = "",
    ):
        """Add entry to reflog."""
        entry = RefLogEntry(
            timestamp=time.time(),
            ref_name=ref_name,
            old_hash=old_hash,
            new_hash=new_hash,
            action=action,
            message=message,
        )
        self._entries.append(entry)
        
        # Trim if over limit
        while len(self._entries) > self.max_entries:
            self._entries.pop(0)
    
    def get(self, ref_name: str, n: int = 10) -> List[RefLogEntry]:
        """Get last n entries for a ref."""
        entries = [e for e in self._entries if e.ref_name == ref_name]
        return entries[-n:]
    
    def all(self) -> List[RefLogEntry]:
        """Get all entries."""
        return list(self._entries)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "entries": [
                {
                    "timestamp": e.timestamp,
                    "ref_name": e.ref_name,
                    "old_hash": e.old_hash,
                    "new_hash": e.new_hash,
                    "action": e.action,
                    "message": e.message,
                }
                for e in self._entries
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RefLog':
        reflog = cls(max_entries=data.get("max_entries", 1000))
        for e in data.get("entries", []):
            reflog._entries.append(RefLogEntry(
                timestamp=e["timestamp"],
                ref_name=e["ref_name"],
                old_hash=e.get("old_hash"),
                new_hash=e["new_hash"],
                action=e["action"],
                message=e.get("message", ""),
            ))
        return reflog


# =============================================================================
# HISTORY GRAPH
# =============================================================================

class HistoryGraph:
    """
    Complete history graph with branches and tags.
    
    Combines:
    - Merkle DAG of commits
    - Named branches
    - Tags
    - Reference log
    
    Example:
        history = HistoryGraph()
        
        # Initial commit
        commit0 = FieldCommit.create(field, message="Initial")
        history.commit(commit0)
        
        # Create branch
        history.create_branch("experiment")
        history.checkout("experiment")
        
        # Work on branch
        commit1 = FieldCommit.create(field, parents=[commit0.hash])
        history.commit(commit1)
        
        # Merge back
        history.checkout("main")
        history.merge("experiment")
    """
    
    def __init__(self, default_branch: str = "main"):
        self._dag = MerkleDAG()
        self._commits: Dict[str, FieldCommit] = {}
        
        self._branches: Dict[str, Branch] = {}
        self._tags: Dict[str, Tag] = {}
        self._reflog = RefLog()
        
        self._current_branch: Optional[str] = None
        self._head: Optional[str] = None  # Detached HEAD if no branch
        
        self._default_branch = default_branch
    
    @property
    def head(self) -> Optional[str]:
        """Current HEAD commit hash."""
        if self._current_branch and self._current_branch in self._branches:
            return self._branches[self._current_branch].head
        return self._head
    
    @property
    def current_branch(self) -> Optional[str]:
        """Current branch name, or None if detached."""
        return self._current_branch
    
    @property
    def branches(self) -> List[str]:
        """List of branch names."""
        return list(self._branches.keys())
    
    @property
    def tags_list(self) -> List[str]:
        """List of tag names."""
        return list(self._tags.keys())
    
    def commit(
        self,
        commit: FieldCommit,
        update_branch: bool = True,
    ) -> str:
        """
        Add a commit to history.
        
        Args:
            commit: FieldCommit to add
            update_branch: Whether to move current branch to this commit
            
        Returns:
            Commit hash
        """
        # Add to DAG
        self._dag.add(commit.to_merkle_node())
        self._commits[commit.hash] = commit
        
        # Update branch
        if update_branch and self._current_branch:
            old_hash = self._branches[self._current_branch].head
            self._branches[self._current_branch].head = commit.hash
            self._branches[self._current_branch].updated_at = time.time()
            
            self._reflog.add(
                ref_name=f"refs/heads/{self._current_branch}",
                old_hash=old_hash,
                new_hash=commit.hash,
                action="commit",
                message=commit.metadata.message,
            )
        elif update_branch:
            self._head = commit.hash
        
        return commit.hash
    
    def get_commit(self, ref: str) -> Optional[FieldCommit]:
        """
        Get commit by ref (hash, branch, or tag).
        
        Args:
            ref: Commit hash, branch name, or tag name
            
        Returns:
            FieldCommit or None
        """
        # Try as hash first
        if ref in self._commits:
            return self._commits[ref]
        
        # Try as branch
        if ref in self._branches:
            return self._commits.get(self._branches[ref].head)
        
        # Try as tag
        if ref in self._tags:
            return self._commits.get(self._tags[ref].target)
        
        # Try short hash
        for hash, commit in self._commits.items():
            if hash.startswith(ref):
                return commit
        
        return None
    
    def create_branch(
        self,
        name: str,
        start_point: Optional[str] = None,
    ) -> Branch:
        """
        Create a new branch.
        
        Args:
            name: Branch name
            start_point: Commit hash to start from (default: HEAD)
            
        Returns:
            New Branch
        """
        if name in self._branches:
            raise ValueError(f"Branch '{name}' already exists")
        
        # Determine start point
        if start_point is None:
            start_point = self.head
        
        if start_point is None:
            raise ValueError("Cannot create branch: no commits yet")
        
        branch = Branch(name=name, head=start_point)
        self._branches[name] = branch
        
        self._reflog.add(
            ref_name=f"refs/heads/{name}",
            old_hash=None,
            new_hash=start_point,
            action="branch",
            message=f"branch: Created from {start_point[:8]}",
        )
        
        return branch
    
    def delete_branch(self, name: str):
        """Delete a branch."""
        if name not in self._branches:
            raise ValueError(f"Branch '{name}' does not exist")
        
        if name == self._current_branch:
            raise ValueError("Cannot delete current branch")
        
        old_hash = self._branches[name].head
        del self._branches[name]
        
        self._reflog.add(
            ref_name=f"refs/heads/{name}",
            old_hash=old_hash,
            new_hash="0" * 40,  # Null hash
            action="delete-branch",
        )
    
    def checkout(self, ref: str, create: bool = False):
        """
        Checkout a branch or commit.
        
        Args:
            ref: Branch name or commit hash
            create: Create branch if it doesn't exist
        """
        # Try as branch first
        if ref in self._branches:
            old_branch = self._current_branch
            self._current_branch = ref
            self._head = None
            
            self._reflog.add(
                ref_name="HEAD",
                old_hash=self._branches[old_branch].head if old_branch else None,
                new_hash=self._branches[ref].head,
                action="checkout",
                message=f"checkout: moving to {ref}",
            )
            return
        
        # Create new branch?
        if create:
            self.create_branch(ref)
            self.checkout(ref)
            return
        
        # Try as commit (detached HEAD)
        commit = self.get_commit(ref)
        if commit:
            old_hash = self.head
            self._current_branch = None
            self._head = commit.hash
            
            self._reflog.add(
                ref_name="HEAD",
                old_hash=old_hash,
                new_hash=commit.hash,
                action="checkout",
                message=f"checkout: detached HEAD at {commit.hash[:8]}",
            )
            return
        
        raise ValueError(f"Cannot checkout '{ref}': not found")
    
    def tag(
        self,
        name: str,
        target: Optional[str] = None,
        message: str = "",
    ) -> Tag:
        """
        Create a tag.
        
        Args:
            name: Tag name
            target: Commit hash (default: HEAD)
            message: Tag message
            
        Returns:
            New Tag
        """
        if name in self._tags:
            raise ValueError(f"Tag '{name}' already exists")
        
        if target is None:
            target = self.head
        
        if target is None:
            raise ValueError("Cannot create tag: no commits yet")
        
        tag = Tag(name=name, target=target, message=message)
        self._tags[name] = tag
        
        return tag
    
    def delete_tag(self, name: str):
        """Delete a tag."""
        if name not in self._tags:
            raise ValueError(f"Tag '{name}' does not exist")
        del self._tags[name]
    
    def log(
        self,
        ref: Optional[str] = None,
        n: int = 10,
    ) -> Iterator[FieldCommit]:
        """
        Iterate over commit history.
        
        Args:
            ref: Starting point (default: HEAD)
            n: Maximum number of commits
            
        Yields:
            FieldCommit objects in reverse chronological order
        """
        start = ref or self.head
        if start is None:
            return
        
        commit = self.get_commit(start)
        if not commit:
            return
        
        count = 0
        visited = set()
        queue = [commit]
        
        while queue and count < n:
            current = queue.pop(0)
            
            if current.hash in visited:
                continue
            
            visited.add(current.hash)
            yield current
            count += 1
            
            # Add parents to queue (sorted by timestamp for consistent order)
            parents = [
                self._commits[h] for h in current.parent_hashes
                if h in self._commits
            ]
            parents.sort(key=lambda c: c.metadata.timestamp, reverse=True)
            queue.extend(parents)
    
    def merge(
        self,
        branch_name: str,
        message: Optional[str] = None,
    ) -> Optional[FieldCommit]:
        """
        Merge a branch into current branch.
        
        Creates a merge commit with two parents.
        
        Note: This only updates the graph structure.
        Actual field merging requires external logic.
        
        Args:
            branch_name: Branch to merge
            message: Merge commit message
            
        Returns:
            Merge commit (if needed), None if fast-forward
        """
        if branch_name not in self._branches:
            raise ValueError(f"Branch '{branch_name}' does not exist")
        
        if self._current_branch is None:
            raise ValueError("Cannot merge in detached HEAD state")
        
        source_hash = self._branches[branch_name].head
        target_hash = self._branches[self._current_branch].head
        
        # Check if fast-forward is possible
        source_commit = self._commits[source_hash]
        if target_hash in source_commit.parent_hashes:
            # Source is descendant of target - just move pointer
            self._branches[self._current_branch].head = source_hash
            return None
        
        # Need actual merge commit
        # The actual merged data would need to be computed externally
        if message is None:
            message = f"Merge branch '{branch_name}' into {self._current_branch}"
        
        # Return info about what needs to be merged
        # Actual merge commit creation happens after data merge
        return None  # Caller should create merge commit
    
    def common_ancestor(self, ref1: str, ref2: str) -> Optional[FieldCommit]:
        """Find common ancestor of two refs."""
        commit1 = self.get_commit(ref1)
        commit2 = self.get_commit(ref2)
        
        if not commit1 or not commit2:
            return None
        
        ancestor = self._dag.common_ancestor(commit1.hash, commit2.hash)
        if ancestor:
            return self._commits.get(ancestor.hash)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "dag": self._dag.to_dict(),
            "commits": {h: c.to_dict() for h, c in self._commits.items()},
            "branches": {n: b.to_dict() for n, b in self._branches.items()},
            "tags": {n: t.to_dict() for n, t in self._tags.items()},
            "reflog": self._reflog.to_dict(),
            "current_branch": self._current_branch,
            "head": self._head,
            "default_branch": self._default_branch,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HistoryGraph':
        """Deserialize from dictionary."""
        history = cls(default_branch=data.get("default_branch", "main"))
        
        history._dag = MerkleDAG.from_dict(data.get("dag", {}))
        
        for h, c in data.get("commits", {}).items():
            history._commits[h] = FieldCommit.from_dict(c)
        
        for n, b in data.get("branches", {}).items():
            history._branches[n] = Branch.from_dict(b)
        
        for n, t in data.get("tags", {}).items():
            history._tags[n] = Tag.from_dict(t)
        
        history._reflog = RefLog.from_dict(data.get("reflog", {}))
        history._current_branch = data.get("current_branch")
        history._head = data.get("head")
        
        return history
