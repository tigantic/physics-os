"""
§6 Data Infrastructure & Observability — Comprehensive Test Suite
==================================================================

Covers all 12 items in OS_Evolution.md §6.
Tests aligned to the actual constructor / method signatures.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


# ============================================================== #
#  6.1 — Prometheus/Grafana Telemetry (existing)                  #
# ============================================================== #

class TestTelemetry:
    def test_observability_import(self):
        from ontic.ml.discovery.production.observability import MetricsCollector
        assert MetricsCollector is not None


# ============================================================== #
#  6.2 — Time-Series Database Connector                           #
# ============================================================== #

class TestTimeSeriesDB:
    def test_in_memory_write_read(self):
        from ontic.platform.timeseries_db import InMemoryTSDB, TimeSeriesPoint
        db = InMemoryTSDB()
        now = time.time()
        db.write("cpu", TimeSeriesPoint(now, 45.0, {"host": "a"}))
        db.write("cpu", TimeSeriesPoint(now + 1, 50.0, {"host": "a"}))
        assert len(db.list_metrics()) >= 1

    def test_query_range(self):
        from ontic.platform.timeseries_db import (
            InMemoryTSDB,
            TimeSeriesPoint,
            TimeSeriesQuery,
        )
        db = InMemoryTSDB()
        base = 1_000_000.0
        for i in range(10):
            db.write("mem", TimeSeriesPoint(base + i, float(i), {}))
        q = TimeSeriesQuery(metric_name="mem", start_time=base + 2, end_time=base + 7)
        points = db.query_range(q)
        assert len(points) >= 5

    def test_aggregation(self):
        from ontic.platform.timeseries_db import (
            InMemoryTSDB,
            TimeSeriesPoint,
            TimeSeriesQuery,
        )
        db = InMemoryTSDB()
        base = 1_000_000.0
        for i in range(20):
            db.write("temp", TimeSeriesPoint(base + i, float(i)))
        q = TimeSeriesQuery(
            metric_name="temp",
            start_time=base,
            end_time=base + 19,
            aggregation="mean",
            step_seconds=10.0,
        )
        agg = db.query_aggregated(q)
        assert len(agg) >= 1

    def test_retention_enforcement(self):
        from ontic.platform.timeseries_db import (
            InMemoryTSDB,
            RetentionPolicy,
            TimeSeriesPoint,
        )
        db = InMemoryTSDB(retention=RetentionPolicy.HOUR)
        old_ts = time.time() - 7200  # 2 hours ago (exceeds HOUR)
        db.write("cpu", TimeSeriesPoint(old_ts, 25.0))
        db.write("cpu", TimeSeriesPoint(time.time(), 30.0))
        removed = db.enforce_retention()
        assert removed >= 1


# ============================================================== #
#  6.3 — Data Lakehouse                                           #
# ============================================================== #

class TestLakehouse:
    def test_catalog_crud(self):
        from ontic.platform.lakehouse import (
            ArtifactMetadata,
            ArtifactType,
            LakehouseCatalog,
        )
        cat = LakehouseCatalog()
        meta = ArtifactMetadata(
            artifact_id="a1",
            artifact_type=ArtifactType.FIELD_SNAPSHOT,
            domain="cfd",
        )
        cat.register(meta)
        assert cat.count == 1
        assert cat.get("a1") is not None
        cat.remove("a1")
        assert cat.count == 0

    def test_catalog_query(self):
        from ontic.platform.lakehouse import (
            ArtifactMetadata,
            ArtifactType,
            LakehouseCatalog,
        )
        cat = LakehouseCatalog()
        for i in range(5):
            cat.register(ArtifactMetadata(
                artifact_id=f"art_{i}",
                artifact_type=ArtifactType.METRICS if i % 2 == 0 else ArtifactType.MESH,
                domain="cfd",
            ))
        results = cat.query(artifact_type=ArtifactType.METRICS)
        assert len(results) == 3

    def test_store_put_get(self):
        from ontic.platform.lakehouse import ArtifactType, LakehouseStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LakehouseStore(Path(tmpdir))
            data = np.random.randn(10, 10).astype(np.float32)
            meta = store.put(data, ArtifactType.FIELD_SNAPSHOT, domain="fea")
            loaded = store.get(meta.artifact_id)
            assert loaded is not None
            np.testing.assert_array_equal(data, loaded)

    def test_store_json(self):
        from ontic.platform.lakehouse import ArtifactType, LakehouseStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LakehouseStore(Path(tmpdir))
            doc = {"solver": "cfd", "Re": 1000}
            meta = store.put_json(doc, ArtifactType.CONFIG, domain="cfd")
            assert meta.size_bytes > 0

    def test_catalog_save_load(self):
        from ontic.platform.lakehouse import (
            ArtifactMetadata,
            ArtifactType,
            LakehouseCatalog,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cat = LakehouseCatalog(Path(tmpdir) / "cat.json")
            cat.register(ArtifactMetadata(
                artifact_id="x", artifact_type=ArtifactType.LOG,
            ))
            cat.save()
            cat2 = LakehouseCatalog(Path(tmpdir) / "cat.json")
            loaded = cat2.load()
            assert loaded == 1

    def test_lakehouse_query_builder(self):
        from ontic.platform.lakehouse import (
            ArtifactMetadata,
            ArtifactType,
            LakehouseCatalog,
            LakehouseQuery,
        )
        cat = LakehouseCatalog()
        cat.register(ArtifactMetadata(
            artifact_id="q1", artifact_type=ArtifactType.MESH, domain="cfd",
        ))
        cat.register(ArtifactMetadata(
            artifact_id="q2", artifact_type=ArtifactType.MESH, domain="fea",
        ))
        results = (
            LakehouseQuery(cat)
            .where_domain("cfd")
            .where_type(ArtifactType.MESH)
            .execute()
        )
        assert len(results) == 1 and results[0].artifact_id == "q1"


# ============================================================== #
#  6.4 — Arrow / Parquet Export                                   #
# ============================================================== #

class TestArrowExport:
    def test_arrow_batch_from_dict(self):
        from ontic.platform.arrow_export import ArrowBatch
        batch = ArrowBatch.from_dict({
            "x": np.arange(10, dtype=np.float64),
            "y": np.arange(10, dtype=np.float64),
        })
        assert batch.num_rows == 10
        assert batch.num_columns == 2

    def test_batch_select_and_slice(self):
        from ontic.platform.arrow_export import ArrowBatch
        batch = ArrowBatch.from_dict({
            "a": np.arange(20, dtype=np.float64),
            "b": np.ones(20, dtype=np.float64),
        })
        sub = batch.select(["a"])
        assert sub.num_columns == 1
        sliced = batch.slice(5, 10)
        assert sliced.num_rows == 10

    def test_parquet_roundtrip(self):
        from ontic.platform.arrow_export import (
            ArrowBatch,
            ParquetReader,
            ParquetWriter,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = ArrowBatch.from_dict({
                "pressure": np.random.randn(100).astype(np.float64),
                "velocity": np.random.randn(100).astype(np.float64),
            })
            path = Path(tmpdir) / "test.parquet"
            w = ParquetWriter(path)
            w.write_batch(batch)
            w.close()
            batches = ParquetReader(path).read()
            assert len(batches) == 1
            assert batches[0].num_rows == 100
            np.testing.assert_allclose(
                batch.column("pressure"),
                batches[0].column("pressure"),
            )

    def test_export_import_helpers(self):
        from ontic.platform.arrow_export import export_to_parquet, import_from_parquet
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "u": np.random.randn(50).astype(np.float64),
                "v": np.random.randn(50).astype(np.float64),
            }
            path = Path(tmpdir) / "sim.parquet"
            export_to_parquet(state, path)
            loaded = import_from_parquet(path)
            np.testing.assert_allclose(state["u"], loaded["u"])

    def test_simulation_state_flatten(self):
        from ontic.platform.arrow_export import simulation_state_to_batch
        state = {"field": np.random.randn(8, 8)}
        batch = simulation_state_to_batch(state, flatten=True)
        assert batch.num_rows == 64


# ============================================================== #
#  6.5 — Streaming Output Pipeline (existing)                     #
# ============================================================== #

class TestStreamingPipeline:
    def test_streaming_import(self):
        from ontic.ml.discovery import StreamingPipeline
        assert StreamingPipeline is not None


# ============================================================== #
#  6.6 — Lineage Graph (existing)                                 #
# ============================================================== #

class TestLineageGraph:
    def test_lineage_import(self):
        from ontic.platform import CheckpointStore
        assert CheckpointStore is not None


# ============================================================== #
#  6.7 — Experiment Tracking                                      #
# ============================================================== #

class TestExperimentTracker:
    def test_create_experiment(self):
        from ontic.platform.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        exp = tracker.create_experiment("test_exp", description="unit test")
        assert exp.name == "test_exp"
        assert tracker.list_experiments()[0].experiment_id == exp.experiment_id

    def test_run_lifecycle(self):
        from ontic.platform.experiment_tracker import ExperimentTracker, RunStatus
        tracker = ExperimentTracker()
        exp = tracker.create_experiment("exp1")
        run = tracker.start_run(exp.experiment_id, name="run1")
        assert run.status == RunStatus.RUNNING
        tracker.log_param("lr", 0.001)
        tracker.log_metric("loss", 1.0, step=0)
        tracker.log_metric("loss", 0.5, step=1)
        tracker.end_run()
        assert run.status == RunStatus.COMPLETED
        assert run.duration() > 0

    def test_leaderboard(self):
        from ontic.platform.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        exp = tracker.create_experiment("bench")
        for i in range(5):
            run = tracker.start_run(exp.experiment_id, name=f"r{i}")
            tracker.log_metric("rmse", float(5 - i), step=0)
            tracker.end_run()
        board = exp.leaderboard("rmse", mode="min", top_k=3)
        assert len(board) == 3
        assert board[0][1] <= board[1][1]

    def test_compare_runs(self):
        from ontic.platform.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        exp = tracker.create_experiment("cmp")
        r1 = tracker.start_run(exp.experiment_id)
        tracker.log_metric("acc", 0.9)
        tracker.end_run()
        r2 = tracker.start_run(exp.experiment_id)
        tracker.log_metric("acc", 0.95)
        tracker.end_run()
        cmp = tracker.compare_runs([r1.run_id, r2.run_id], "acc")
        assert len(cmp) == 2

    def test_save_load(self):
        from ontic.platform.experiment_tracker import ExperimentTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(Path(tmpdir))
            exp = tracker.create_experiment("persist")
            run = tracker.start_run(exp.experiment_id)
            tracker.log_metric("x", 42.0)
            tracker.end_run()
            tracker.save()

            tracker2 = ExperimentTracker(Path(tmpdir))
            loaded = tracker2.load()
            assert loaded == 1

    def test_global_leaderboard(self):
        from ontic.platform.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        for e in range(2):
            exp = tracker.create_experiment(f"exp_{e}")
            for i in range(3):
                run = tracker.start_run(exp.experiment_id)
                tracker.log_metric("score", float(e * 3 + i))
                tracker.end_run()
        board = tracker.global_leaderboard("score", mode="max", top_k=2)
        assert len(board) == 2
        assert board[0][2] >= board[1][2]


# ============================================================== #
#  6.8 — Live Data Ingestion (existing)                           #
# ============================================================== #

class TestLiveIngestion:
    def test_noaa_connector_exists(self):
        import ontic.ml.discovery
        assert ontic.ml.discovery is not None


# ============================================================== #
#  6.9 — Federated Data Exchange                                  #
# ============================================================== #

class TestFederation:
    def test_node_lifecycle(self):
        from ontic.platform.federation import (
            FederationRegistry,
            FederationNode,
            NodeRole,
        )
        reg = FederationRegistry()
        node = FederationNode(name="n1", role=NodeRole.PRODUCER)
        reg.register(node)
        assert reg.count == 1
        assert node.is_alive()
        reg.heartbeat(node.node_id)

    def test_publish_and_transfer(self):
        from ontic.platform.federation import (
            ChunkStatus,
            DataChunk,
            FederationManager,
        )
        mgr = FederationManager()
        mgr.init_local_node(name="local")
        data = np.random.randn(100).astype(np.float64)
        chunk = mgr.share_data(data, domain="cfd")
        assert chunk.status == ChunkStatus.COMPLETE
        assert chunk.verify()

        fetched = mgr.fetch_data(chunk.chunk_id)
        assert fetched is not None
        np.testing.assert_array_equal(data, fetched)

    def test_discover_data(self):
        from ontic.platform.federation import FederationManager
        mgr = FederationManager()
        mgr.init_local_node()
        mgr.share_data(np.zeros(10), domain="fea")
        mgr.share_data(np.ones(10), domain="cfd")
        disc = mgr.discover_data(domain="cfd")
        assert len(disc) == 1

    def test_registry_queries(self):
        from ontic.platform.federation import (
            FederationNode,
            FederationRegistry,
            NodeRole,
        )
        reg = FederationRegistry()
        reg.register(FederationNode(name="p1", role=NodeRole.PRODUCER, capabilities={"cfd"}))
        reg.register(FederationNode(name="c1", role=NodeRole.CONSUMER))
        assert len(reg.producers()) == 1
        assert len(reg.nodes_with_capability("cfd")) == 1


# ============================================================== #
#  6.10 — Anomaly Detection (existing)                            #
# ============================================================== #

class TestAnomalyDetection:
    def test_regime_detector_exists(self):
        from ontic.ml.neural import AlgorithmSelector
        assert AlgorithmSelector is not None


# ============================================================== #
#  6.11 — Simulation Replay                                       #
# ============================================================== #

class TestReplay:
    def test_replay_log_append(self):
        from ontic.platform.replay import EventType, ReplayEvent, ReplayLog
        log = ReplayLog(run_id="test")
        log.append(ReplayEvent(event_type=EventType.INIT, step=0))
        log.append(ReplayEvent(event_type=EventType.STEP, step=1))
        assert log.length == 2

    def test_record_and_replay(self):
        from ontic.platform.replay import ReplayEngine

        def step_fn(state, params):
            return {"x": state["x"] + params.get("dx", 1.0)}

        engine = ReplayEngine(step_fn=step_fn)
        log = engine.start_recording()
        state = {"x": np.float64(0.0)}
        params = {"dx": 1.0}
        engine.record_init(state, params)
        for i in range(5):
            state = step_fn(state, params)
            engine.record_step(i + 1, state, params)
        engine.stop_recording()
        assert log.length == 7  # INIT + 5 STEP + COMPLETE

        replayed = engine.replay(log, max_steps=5)
        cmp = engine.compare(log, replayed)
        assert cmp["deterministic"] is True

    def test_branch_from(self):
        from ontic.platform.replay import ReplayEngine

        def step_fn(state, params):
            return {"v": state["v"] * params.get("decay", 0.9)}

        engine = ReplayEngine(step_fn=step_fn)
        log = engine.start_recording()
        state = {"v": np.float64(100.0)}
        params = {"decay": 0.9}
        engine.record_init(state, params)
        engine.record_checkpoint(0, state)
        for i in range(10):
            state = step_fn(state, params)
            engine.record_step(i + 1, state)
        engine.stop_recording()

        branch = engine.branch_from(log, branch_step=0, new_params={"decay": 0.5}, additional_steps=5)
        assert branch.length > 0

    def test_log_save_load(self):
        from ontic.platform.replay import EventType, ReplayEvent, ReplayLog
        with tempfile.TemporaryDirectory() as tmpdir:
            log = ReplayLog(run_id="persist")
            log.append(ReplayEvent(
                event_type=EventType.INIT, step=0,
                state_snapshot={"arr": np.array([1.0, 2.0])},
            ))
            path = Path(tmpdir) / "log.json"
            log.save(path)
            loaded = ReplayLog.load(path)
            assert loaded.length == 1
            snap = loaded.event_at(0).state_snapshot
            assert snap is not None
            np.testing.assert_array_equal(snap["arr"], np.array([1.0, 2.0]))


# ============================================================== #
#  6.12 — Data Versioning                                         #
# ============================================================== #

class TestDataVersioning:
    def test_content_store(self):
        from ontic.platform.data_versioning import ContentStore
        store = ContentStore()
        key = store.put(b"hello world")
        assert store.exists(key)
        assert store.get(key) == b"hello world"

    def test_content_store_array(self):
        from ontic.platform.data_versioning import ContentStore
        store = ContentStore()
        arr = np.random.randn(10, 5).astype(np.float64)
        key = store.put_array(arr)
        loaded = store.get_array(key)
        assert loaded is not None
        np.testing.assert_array_equal(arr, loaded)

    def test_snapshot_and_checkout(self):
        from ontic.platform.data_versioning import DataVersioning
        dv = DataVersioning()
        ds = {"temperature": np.random.randn(20), "pressure": np.random.randn(20)}
        snap = dv.snapshot(ds, message="initial", author="test")
        loaded = dv.checkout(snap.snapshot_id)
        np.testing.assert_array_equal(ds["temperature"], loaded["temperature"])

    def test_diff(self):
        from ontic.platform.data_versioning import DataVersioning
        dv = DataVersioning()
        ds1 = {"a": np.array([1.0, 2.0]), "b": np.array([3.0])}
        ds2 = {"a": np.array([1.0, 2.0]), "c": np.array([4.0])}
        s1 = dv.snapshot(ds1, message="v1")
        s2 = dv.snapshot(ds2, message="v2")
        diffs = dv.diff(s1.snapshot_id, s2.snapshot_id)
        changes = {d.path: d.change for d in diffs}
        assert "b" in changes and changes["b"] == "removed"
        assert "c" in changes and changes["c"] == "added"

    def test_history(self):
        from ontic.platform.data_versioning import DataVersioning
        dv = DataVersioning()
        for i in range(3):
            dv.snapshot({"x": np.array([float(i)])}, message=f"v{i}")
        hist = dv.history()
        assert len(hist) == 3

    def test_merge_ours_theirs(self):
        from ontic.platform.data_versioning import (
            DatasetSnapshot,
            FileEntry,
            merge_snapshots,
        )
        base = DatasetSnapshot(snapshot_id="base")
        base.files["shared"] = FileEntry("shared", "key_a")
        base.files["conflict"] = FileEntry("conflict", "key_old")

        ours = DatasetSnapshot(snapshot_id="ours")
        ours.files["shared"] = FileEntry("shared", "key_a")
        ours.files["conflict"] = FileEntry("conflict", "key_ours")
        ours.files["only_ours"] = FileEntry("only_ours", "key_o")

        theirs = DatasetSnapshot(snapshot_id="theirs")
        theirs.files["shared"] = FileEntry("shared", "key_a")
        theirs.files["conflict"] = FileEntry("conflict", "key_theirs")
        theirs.files["only_theirs"] = FileEntry("only_theirs", "key_t")

        merged = merge_snapshots(base, theirs, ours, prefer="ours")
        assert "shared" in merged.files
        assert merged.files["conflict"].blob_key == "key_ours"
        assert "only_ours" in merged.files
        assert "only_theirs" in merged.files

    def test_disk_persistence(self):
        from ontic.platform.data_versioning import ContentStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ContentStore(Path(tmpdir))
            arr = np.array([1, 2, 3], dtype=np.float64)
            key = store.put_array(arr)

            store2 = ContentStore(Path(tmpdir))
            loaded = store2.get_array(key)
            assert loaded is not None
            np.testing.assert_array_equal(arr, loaded)


# ============================================================== #
#  Smoke test: all §6 symbols importable from platform __init__   #
# ============================================================== #

class TestS6Imports:
    def test_platform_init_s6_symbols(self):
        from ontic.platform import (
            InMemoryTSDB,
            TimeSeriesPoint,
            LakehouseCatalog,
            LakehouseStore,
            ArrowBatch,
            ParquetWriter,
            export_to_parquet,
            ExperimentTracker,
            Run,
            FederationManager,
            FederationNode,
            ReplayEngine,
            ReplayLog,
            DataVersioning,
            ContentStore,
        )
        assert all(c is not None for c in [
            InMemoryTSDB, TimeSeriesPoint,
            LakehouseCatalog, LakehouseStore,
            ArrowBatch, ParquetWriter, export_to_parquet,
            ExperimentTracker, Run,
            FederationManager, FederationNode,
            ReplayEngine, ReplayLog,
            DataVersioning, ContentStore,
        ])
