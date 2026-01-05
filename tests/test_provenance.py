"""
Tests for Layer 5: Provenance
==============================

Merkle DAG for field lineage tracking.
"""

import os
import tempfile
import time

import numpy as np
import pytest

from tensornet.provenance import (  # Merkle; Commit; History; Store; Diff; Audit
    AuditEvent, AuditQuery, AuditTrail, Branch, CommitMetadata, DiffEngine,
    DiffSummary, DiffType, EventSeverity, EventType, FieldCommit, FieldDiff,
    FileSystemBackend, HistoryGraph, MemoryBackend, MerkleDAG, MerkleNode,
    MerkleProof, ProvenanceStore, RefLog, StoreConfig, Tag, compute_diff,
    compute_hash, make_commit, verify_proof)

# =============================================================================
# MERKLE DAG TESTS
# =============================================================================


class TestComputeHash:
    """Tests for hash computation."""

    def test_bytes_hash(self):
        h = compute_hash(b"hello world")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_string_hash(self):
        h = compute_hash("hello world")
        assert isinstance(h, str)

    def test_numpy_hash(self):
        arr = np.array([1.0, 2.0, 3.0])
        h = compute_hash(arr)
        assert isinstance(h, str)

    def test_deterministic(self):
        data = b"test data"
        h1 = compute_hash(data)
        h2 = compute_hash(data)
        assert h1 == h2

    def test_different_data_different_hash(self):
        h1 = compute_hash(b"data1")
        h2 = compute_hash(b"data2")
        assert h1 != h2


class TestMerkleNode:
    """Tests for MerkleNode."""

    def test_create_leaf(self):
        node = MerkleNode.create_leaf(b"content")
        assert node.data_hash is not None
        assert node.hash

    def test_create_leaf_with_metadata(self):
        node = MerkleNode.create_leaf(b"content", metadata={"name": "test"})
        assert node.metadata.get("name") == "test"

    def test_create_internal(self):
        left = MerkleNode.create_leaf(b"left")
        right = MerkleNode.create_leaf(b"right")
        parent = MerkleNode.create_internal([left, right])

        assert len(parent.parent_hashes) == 2

    def test_hash_changes_with_content(self):
        n1 = MerkleNode.create_leaf(b"content1")
        n2 = MerkleNode.create_leaf(b"content2")
        assert n1.hash != n2.hash


class TestMerkleDAG:
    """Tests for MerkleDAG."""

    def test_empty_dag(self):
        dag = MerkleDAG()
        assert len(dag) == 0

    def test_add_nodes(self):
        dag = MerkleDAG()

        leaf1 = MerkleNode.create_leaf(b"data1")
        leaf2 = MerkleNode.create_leaf(b"data2")

        dag.add(leaf1)
        dag.add(leaf2)

        assert len(dag) == 2

    def test_get_node_by_hash(self):
        dag = MerkleDAG()
        leaf = MerkleNode.create_leaf(b"data")
        dag.add(leaf)

        retrieved = dag.get(leaf.hash)
        assert retrieved == leaf

    def test_contains(self):
        dag = MerkleDAG()
        leaf = MerkleNode.create_leaf(b"data")

        assert leaf not in dag
        dag.add(leaf)
        assert leaf in dag
        assert leaf.hash in dag

    def test_heads_and_roots(self):
        dag = MerkleDAG()

        leaf1 = MerkleNode.create_leaf(b"data1")
        leaf2 = MerkleNode.create_leaf(b"data2")
        root = MerkleNode.create_internal([leaf1, leaf2])

        dag.add(leaf1)
        dag.add(leaf2)
        dag.add(root)

        assert len(dag.heads) == 1  # Only root has no children
        assert len(dag.roots) == 2  # leaf1, leaf2 have no parents


class TestMerkleProof:
    """Tests for Merkle proofs."""

    def test_create_proof(self):
        proof = MerkleProof(
            target_hash="abc123",
            root_hash="def456",
            path=["abc123", "def456"],
            siblings=[[], []],
        )
        assert proof.target_hash == "abc123"

    def test_verify_proof_basic(self):
        # Basic test that verify_proof function exists
        proof = MerkleProof(
            target_hash="abc123",
            root_hash="abc123",
            path=["abc123"],
            siblings=[[]],
        )
        # Self-proof should work
        result = verify_proof(proof)
        assert isinstance(result, bool)


# =============================================================================
# COMMIT TESTS
# =============================================================================


class TestFieldCommit:
    """Tests for FieldCommit."""

    def test_create_commit(self):
        field_data = np.array([1.0, 2.0, 3.0])
        commit = FieldCommit.create(
            field_data=field_data,
            message="Initial commit",
        )

        assert commit.hash is not None
        assert commit.data_hash is not None
        assert commit.metadata.message == "Initial commit"

    def test_commit_with_metadata(self):
        field_data = np.array([1.0, 2.0])
        commit = FieldCommit.create(
            field_data=field_data,
            message="Test",
            author="test_user",
            tags=["baseline", "validated"],
        )

        assert commit.metadata.author == "test_user"
        assert "validated" in commit.metadata.tags

    def test_commit_with_parents(self):
        data1 = np.array([1.0, 2.0])
        commit1 = FieldCommit.create(data1, message="First")

        data2 = np.array([2.0, 3.0])
        commit2 = FieldCommit.create(data2, message="Second", parents=[commit1.hash])

        assert commit2.parent_hashes[0] == commit1.hash

    def test_commit_properties(self):
        data = np.array([1.0, 2.0])
        commit = FieldCommit.create(data, message="Test")

        assert len(commit.short_hash) == 8
        assert commit.is_root  # No parents
        assert not commit.is_merge


class TestMakeCommit:
    """Tests for make_commit function."""

    def test_make_commit_basic(self):
        field = np.array([1.0, 2.0, 3.0])

        commit = make_commit(
            field_data=field,
            message="Added field",
        )

        assert commit.metadata.message == "Added field"
        assert commit.data_hash is not None

    def test_make_commit_with_parent(self):
        field1 = np.array([1.0])
        c1 = make_commit(field1, "First")

        field2 = np.array([2.0])
        c2 = make_commit(field2, "Second", parent=c1.hash)

        assert c2.parent_hashes[0] == c1.hash


# =============================================================================
# HISTORY GRAPH TESTS
# =============================================================================


class TestBranch:
    """Tests for Branch."""

    def test_create_branch(self):
        branch = Branch(name="main", head="abc123")
        assert branch.name == "main"
        assert branch.head == "abc123"

    def test_branch_update(self):
        branch = Branch(name="feature", head="old")
        branch.head = "new"
        assert branch.head == "new"

    def test_branch_serialization(self):
        branch = Branch(name="main", head="abc123")
        d = branch.to_dict()
        restored = Branch.from_dict(d)
        assert restored.name == branch.name


class TestTag:
    """Tests for Tag."""

    def test_create_tag(self):
        tag = Tag(name="v1.0", target="abc123")
        assert tag.name == "v1.0"
        assert tag.target == "abc123"

    def test_annotated_tag(self):
        tag = Tag(
            name="v1.0",
            target="abc123",
            message="Release 1.0",
            author="user@example.com",
        )
        assert tag.message == "Release 1.0"

    def test_tag_serialization(self):
        tag = Tag(name="v1.0", target="abc123", message="Test")
        d = tag.to_dict()
        restored = Tag.from_dict(d)
        assert restored.name == tag.name


class TestHistoryGraph:
    """Tests for HistoryGraph."""

    def test_empty_history(self):
        history = HistoryGraph()
        assert history.head is None

    def test_commit_to_history(self):
        history = HistoryGraph()

        data = np.array([1.0, 2.0])
        commit = FieldCommit.create(data, message="Initial")

        history.commit(commit)
        assert history.head == commit.hash

    def test_branch_operations(self):
        history = HistoryGraph()

        # Create initial commit
        data = np.array([1.0])
        c1 = FieldCommit.create(data, message="Initial")
        history.commit(c1)

        # Create feature branch
        history.create_branch("feature")

        # Check branches
        assert "main" in history.branches or "feature" in history.branches

    def test_tag_operations(self):
        history = HistoryGraph()

        data = np.array([1.0])
        c1 = FieldCommit.create(data, message="Initial")
        history.commit(c1)

        history.tag("v1.0", message="First release")

        tags = history.tags_list
        assert "v1.0" in tags


class TestRefLog:
    """Tests for RefLog."""

    def test_reflog_tracking(self):
        reflog = RefLog()

        reflog.add("main", None, "abc123", "commit", "initial")
        reflog.add("main", "abc123", "def456", "commit", "update")

        entries = reflog.get("main")
        assert len(entries) == 2

    def test_reflog_all(self):
        reflog = RefLog()

        reflog.add("main", None, "c1", "commit")
        reflog.add("main", "c1", "c2", "commit")
        reflog.add("feature", None, "c3", "branch")

        all_entries = reflog.all()
        assert len(all_entries) == 3


# =============================================================================
# STORE TESTS
# =============================================================================


class TestMemoryBackend:
    """Tests for MemoryBackend."""

    def test_store_and_retrieve(self):
        backend = MemoryBackend()

        backend.put("key1", b"value1")
        assert backend.get("key1") == b"value1"

    def test_exists(self):
        backend = MemoryBackend()

        assert not backend.exists("key1")
        backend.put("key1", b"value1")
        assert backend.exists("key1")

    def test_delete(self):
        backend = MemoryBackend()

        backend.put("key1", b"value1")
        backend.delete("key1")
        assert not backend.exists("key1")

    def test_list_hashes(self):
        backend = MemoryBackend()

        backend.put("a", b"1")
        backend.put("b", b"2")
        backend.put("c", b"3")

        keys = list(backend.list_hashes())
        assert len(keys) == 3


class TestFileSystemBackend:
    """Tests for FileSystemBackend."""

    def test_store_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileSystemBackend(tmpdir, compress=False)

            backend.put("key1", b"value1")
            assert backend.get("key1") == b"value1"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Store with one backend
            backend1 = FileSystemBackend(tmpdir, compress=False)
            backend1.put("key1", b"value1")

            # Retrieve with new backend
            backend2 = FileSystemBackend(tmpdir, compress=False)
            assert backend2.get("key1") == b"value1"

    def test_compression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileSystemBackend(tmpdir, compress=True)

            data = b"test data " * 100
            backend.put("compressed", data)
            assert backend.get("compressed") == data


class TestProvenanceStore:
    """Tests for ProvenanceStore."""

    def test_create_memory_store(self):
        store = ProvenanceStore(backend=MemoryBackend())
        assert store.head is None

    def test_commit_field(self):
        store = ProvenanceStore(backend=MemoryBackend())

        field = np.array([1.0, 2.0, 3.0])
        commit = store.commit(field, message="Initial")

        assert commit.hash is not None
        assert store.head == commit.hash

    def test_multiple_commits(self):
        store = ProvenanceStore(backend=MemoryBackend())

        field1 = np.array([1.0, 2.0])
        c1 = store.commit(field1, message="First")

        field2 = np.array([2.0, 3.0])
        c2 = store.commit(field2, message="Second")

        assert c2.parent_hashes[0] == c1.hash
        assert store.head == c2.hash


# =============================================================================
# DIFF TESTS
# =============================================================================


class TestDiffEngine:
    """Tests for DiffEngine."""

    def test_identical_fields(self):
        engine = DiffEngine()

        field = np.array([1.0, 2.0, 3.0])
        diff = engine.compute(field, field)

        assert diff.summary.diff_type == DiffType.IDENTICAL

    def test_modified_field(self):
        engine = DiffEngine()

        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([1.0, 2.5, 3.0])

        diff = engine.compute(f1, f2)

        assert diff.summary.diff_type == DiffType.VALUE_CHANGE
        assert diff.summary.changed_elements > 0

    def test_shape_mismatch(self):
        engine = DiffEngine()

        f1 = np.array([1.0, 2.0])
        f2 = np.array([1.0, 2.0, 3.0])

        diff = engine.compute(f1, f2)

        assert diff.summary.diff_type == DiffType.SHAPE_CHANGE

    def test_hotspots(self):
        engine = DiffEngine(n_hotspots=5)

        f1 = np.zeros((10, 10))
        f2 = np.zeros((10, 10))
        f2[5, 5] = 10.0  # Big change at one location

        diff = engine.compute(f1, f2)

        assert len(diff.hotspots) > 0
        assert diff.hotspots[0][0] == (5, 5)


class TestComputeDiff:
    """Tests for compute_diff convenience function."""

    def test_compute_diff(self):
        f1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        f2 = np.array([[1.0, 2.0], [3.0, 5.0]])

        diff = compute_diff(f1, f2)

        assert diff.summary.changed_elements == 1

    def test_diff_summary_stats(self):
        f1 = np.zeros((10, 10))
        f2 = np.ones((10, 10))

        diff = compute_diff(f1, f2)

        assert diff.summary.changed_elements == 100
        assert diff.summary.max_abs_diff == 1.0


class TestFieldDiff:
    """Tests for FieldDiff dataclass."""

    def test_diff_creation(self):
        summary = DiffSummary(
            diff_type=DiffType.VALUE_CHANGE,
            changed_elements=10,
            total_elements=100,
        )

        diff = FieldDiff(summary=summary)

        assert diff.summary.changed_elements == 10
        assert diff.summary.changed_fraction == 0.1

    def test_diff_to_dict(self):
        summary = DiffSummary(diff_type=DiffType.IDENTICAL)
        diff = FieldDiff(summary=summary)

        d = diff.to_dict()
        assert "summary" in d


# =============================================================================
# AUDIT TESTS
# =============================================================================


class TestAuditEvent:
    """Tests for AuditEvent."""

    def test_create_event(self):
        event = AuditEvent(
            id="e1",
            timestamp=time.time(),
            event_type=EventType.FIELD_CREATE,
            message="Created field",
        )

        assert event.id == "e1"
        assert event.hash  # Auto-generated

    def test_event_hash_changes(self):
        t = time.time()

        e1 = AuditEvent("e1", t, EventType.FIELD_CREATE, message="msg1")
        e2 = AuditEvent("e2", t, EventType.FIELD_CREATE, message="msg2")

        assert e1.hash != e2.hash

    def test_event_verification(self):
        event = AuditEvent(
            id="e1",
            timestamp=time.time(),
            event_type=EventType.COMMIT_CREATE,
            message="Test",
        )

        assert event.verify()

    def test_event_serialization(self):
        event = AuditEvent(
            id="e1",
            timestamp=1000.0,
            event_type=EventType.BRANCH_CREATE,
            severity=EventSeverity.INFO,
            actor="user1",
            target="main",
            message="Created branch",
            data={"source": "HEAD"},
        )

        d = event.to_dict()
        restored = AuditEvent.from_dict(d)

        assert restored.id == event.id
        assert restored.event_type == event.event_type


class TestAuditQuery:
    """Tests for AuditQuery."""

    def test_empty_query_matches_all(self):
        query = AuditQuery()

        event = AuditEvent("e1", time.time(), EventType.FIELD_CREATE)

        assert query.matches(event)

    def test_type_filter(self):
        query = AuditQuery(event_types=[EventType.FIELD_CREATE])

        e1 = AuditEvent("e1", time.time(), EventType.FIELD_CREATE)
        e2 = AuditEvent("e2", time.time(), EventType.FIELD_DELETE)

        assert query.matches(e1)
        assert not query.matches(e2)

    def test_time_filter(self):
        now = time.time()
        query = AuditQuery(start_time=now - 10, end_time=now + 10)

        e_in = AuditEvent("e1", now, EventType.FIELD_CREATE)
        e_before = AuditEvent("e2", now - 20, EventType.FIELD_CREATE)
        e_after = AuditEvent("e3", now + 20, EventType.FIELD_CREATE)

        assert query.matches(e_in)
        assert not query.matches(e_before)
        assert not query.matches(e_after)

    def test_actor_filter(self):
        query = AuditQuery(actor="user1")

        e1 = AuditEvent("e1", time.time(), EventType.FIELD_CREATE, actor="user1")
        e2 = AuditEvent("e2", time.time(), EventType.FIELD_CREATE, actor="user2")

        assert query.matches(e1)
        assert not query.matches(e2)


class TestAuditTrail:
    """Tests for AuditTrail."""

    def test_empty_trail(self):
        trail = AuditTrail()
        assert len(trail) == 0

    def test_log_event(self):
        trail = AuditTrail()

        event = trail.log(
            event_type=EventType.FIELD_CREATE,
            message="Created velocity field",
            target="velocity",
        )

        assert len(trail) == 1
        assert event.message == "Created velocity field"

    def test_chain_integrity(self):
        trail = AuditTrail()

        e1 = trail.log(EventType.FIELD_CREATE, "First")
        e2 = trail.log(EventType.FIELD_UPDATE, "Second")
        e3 = trail.log(EventType.FIELD_DELETE, "Third")

        # Chain should be intact
        assert e2.previous_hash == e1.hash
        assert e3.previous_hash == e2.hash

        # Verify integrity
        valid, issues = trail.verify()
        assert valid
        assert len(issues) == 0

    def test_query_events(self):
        trail = AuditTrail()

        trail.log(EventType.FIELD_CREATE, "Create 1", target="field1")
        trail.log(EventType.FIELD_UPDATE, "Update 1", target="field1")
        trail.log(EventType.FIELD_CREATE, "Create 2", target="field2")

        # Query creates only
        query = AuditQuery(event_types=[EventType.FIELD_CREATE])
        results = list(trail.query(query))
        assert len(results) == 2

    def test_get_recent(self):
        trail = AuditTrail()

        for i in range(10):
            trail.log(EventType.CUSTOM, f"Event {i}")

        recent = trail.get_recent(5)
        assert len(recent) == 5
        assert "Event 9" in recent[0].message

    def test_log_error(self):
        trail = AuditTrail()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            event = trail.log_error("Something went wrong", exception=e)

        assert event.severity == EventSeverity.ERROR
        assert "ValueError" in event.data.get("exception_type", "")

    def test_export_import_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "audit.json")

            # Create and populate
            trail1 = AuditTrail()
            trail1.log(EventType.SYSTEM_START, "Started")
            trail1.log(EventType.FIELD_CREATE, "Created field")

            # Export
            trail1.export_json(path)

            # Import
            trail2 = AuditTrail.import_json(path)

            assert len(trail2) == len(trail1)

    def test_statistics(self):
        trail = AuditTrail()

        trail.log(EventType.FIELD_CREATE, "Create", severity=EventSeverity.INFO)
        trail.log(EventType.FIELD_CREATE, "Create", severity=EventSeverity.INFO)
        trail.log(EventType.SYSTEM_ERROR, "Error", severity=EventSeverity.ERROR)

        stats = trail.get_statistics()

        assert stats["event_count"] == 3
        assert stats["by_type"]["field.create"] == 2
        assert stats["by_severity"]["error"] == 1

    def test_max_events_limit(self):
        trail = AuditTrail(max_events=5)

        for i in range(10):
            trail.log(EventType.CUSTOM, f"Event {i}")

        # Should only keep 5
        assert len(trail) == 5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestProvenanceIntegration:
    """Integration tests for provenance system."""

    def test_full_workflow(self):
        """Test complete provenance workflow."""
        # Create store and audit trail
        store = ProvenanceStore(backend=MemoryBackend())
        audit = AuditTrail()

        # Start session
        audit.log(EventType.SYSTEM_START, "Session started")

        # Create initial fields
        velocity = np.random.randn(64, 64)

        # Commit
        commit1 = store.commit(velocity, message="Initial state")

        audit.log(
            EventType.COMMIT_CREATE,
            f"Created commit: {commit1.metadata.message}",
            target=commit1.hash,
        )

        # Modify and commit again
        velocity2 = velocity + 0.1
        commit2 = store.commit(velocity2, message="Updated velocity")

        audit.log(
            EventType.COMMIT_CREATE,
            f"Created commit: {commit2.metadata.message}",
            target=commit2.hash,
        )

        # Compare
        diff = compute_diff(velocity, velocity2)
        assert diff.summary.diff_type == DiffType.VALUE_CHANGE

        # Verify audit
        valid, issues = audit.verify()
        assert valid

    def test_merkle_tree_structure(self):
        """Test that Merkle DAG works correctly."""
        dag = MerkleDAG()

        # Create field nodes
        n1 = MerkleNode.create_leaf(np.array([1.0, 2.0, 3.0]).tobytes())
        n2 = MerkleNode.create_leaf(np.array([4.0, 5.0, 6.0]).tobytes())
        n3 = MerkleNode.create_internal([n1, n2])

        dag.add(n1)
        dag.add(n2)
        dag.add(n3)

        assert len(dag) == 3
        assert n3 in dag.heads

    def test_commit_chain(self):
        """Test commit chain with parent tracking."""
        store = ProvenanceStore(backend=MemoryBackend())

        # Create chain of commits
        data1 = np.array([1.0])
        c1 = store.commit(data1, message="First")

        data2 = np.array([2.0])
        c2 = store.commit(data2, message="Second")

        data3 = np.array([3.0])
        c3 = store.commit(data3, message="Third")

        # Verify chain
        assert c2.parent_hashes[0] == c1.hash
        assert c3.parent_hashes[0] == c2.hash
        assert store.head == c3.hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
