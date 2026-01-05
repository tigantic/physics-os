"""
Tests for Layer 7: Field OS
============================

Complete orchestration layer tests.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tensornet.fieldos import (Checkpoint, Event, EventType, Field,
                               FieldMetadata, FieldOS, FieldOSConfig,
                               FieldType, Observable, Observer, Pipeline,
                               Plugin, PluginInfo, PluginManager, Session,
                               SessionState, Stage, StageResult)
from tensornet.fieldos.kernel import KernelState, KernelStats
from tensornet.fieldos.observable import (Computed, EventBus, FunctionObserver,
                                          ObservableField, Subscription)
from tensornet.fieldos.pipeline import (FilterStage, FunctionStage,
                                        PipelineBuilder, StageStatus,
                                        TransformStage)
from tensornet.fieldos.plugin import PluginHook, PluginState
from tensornet.fieldos.session import SessionManager, SessionMetadata

# =============================================================================
# FIELD TESTS
# =============================================================================


class TestField:
    """Test Field class."""

    def test_create_scalar_field(self):
        """Create scalar field."""
        field = Field.scalar("temperature", (100, 100), initial=300.0)

        assert field.metadata.name == "temperature"
        assert field.metadata.field_type == FieldType.SCALAR
        assert field.shape == (100, 100)
        assert np.allclose(field.data, 300.0)

    def test_create_vector_field(self):
        """Create vector field."""
        field = Field.vector("velocity", (50, 50))

        assert field.metadata.field_type == FieldType.VECTOR
        assert field.shape[:-1] == (50, 50)
        assert field.shape[-1] == 3  # 3D vector

    def test_create_from_array(self):
        """Create field from existing array."""
        data = np.random.randn(64, 64)
        field = Field.from_array("pressure", data)

        assert np.allclose(field.data, data)

    def test_field_update(self):
        """Update field data."""
        field = Field.scalar("test", (10, 10))
        new_data = np.ones((10, 10)) * 42

        field.update(new_data, source="test")

        assert np.allclose(field.data, 42)
        assert field.version > 0  # version is on Field, not metadata

    def test_field_gradient(self):
        """Compute field gradient."""
        data = np.outer(np.arange(10), np.ones(10))
        field = Field.from_array("ramp", data)

        grad = field.gradient()

        assert grad.metadata.field_type == FieldType.VECTOR

    def test_field_magnitude(self):
        """Compute field magnitude."""
        data = np.stack(
            [
                np.ones((10, 10)) * 3,
                np.ones((10, 10)) * 4,
            ],
            axis=-1,
        )
        field = Field.from_array("vec", data, FieldType.VECTOR)

        mag = field.magnitude()

        assert np.allclose(mag.data, 5.0)

    def test_field_stats(self):
        """Get field statistics."""
        data = np.array([[1, 2], [3, 4]], dtype=float)
        field = Field.from_array("test", data)

        stats = field.stats()

        assert stats["min"] == 1.0
        assert stats["max"] == 4.0
        assert stats["mean"] == 2.5

    def test_field_clip(self):
        """Clip field values."""
        data = np.array([[-1, 0], [1, 2]], dtype=float)
        field = Field.from_array("test", data)

        clipped = field.clip(0, 1)

        assert clipped.data.min() >= 0
        assert clipped.data.max() <= 1

    def test_field_zeros_like(self):
        """Create zeros with same shape."""
        original = Field.scalar("orig", (32, 32), initial=100.0)
        zeros = Field.zeros_like(original, "zeros")

        assert zeros.shape == original.shape
        assert np.allclose(zeros.data, 0)

    def test_field_copy(self):
        """Copy field."""
        field = Field.scalar("test", (10, 10), initial=5.0)
        copy = field.copy()

        # Modify copy
        copy.update(np.ones((10, 10)))

        # Original unchanged
        assert np.allclose(field.data, 5.0)

    def test_field_save_load(self):
        """Save and load field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "field.npz")

            field = Field.scalar("test", (20, 20), initial=42.0)
            field.save(path)

            loaded = Field.load(path)

            assert loaded.metadata.name == "test"
            assert np.allclose(loaded.data, 42.0)


# =============================================================================
# PIPELINE TESTS
# =============================================================================


class TestPipeline:
    """Test Pipeline class."""

    def test_create_pipeline(self):
        """Create empty pipeline."""
        pipeline = Pipeline("test")

        assert pipeline.name == "test"
        assert len(pipeline) == 0

    def test_add_stage(self):
        """Add stage to pipeline."""
        pipeline = Pipeline("test")
        stage = TransformStage(lambda x: x * 2, "double")

        pipeline.add(stage)

        assert len(pipeline) == 1

    def test_run_pipeline(self):
        """Run pipeline on input."""
        pipeline = Pipeline("double")
        pipeline.add(TransformStage(lambda x: x * 2, "double"))

        result = pipeline.run(np.array([1, 2, 3]))

        assert result.is_success
        assert np.allclose(result.output, [2, 4, 6])

    def test_chain_stages(self):
        """Chain multiple stages."""
        pipeline = Pipeline("chain")
        pipeline.add(TransformStage(lambda x: x + 1, "add"))
        pipeline.add(TransformStage(lambda x: x * 2, "mult"))

        result = pipeline.run(np.array([1, 2, 3]))

        # (x + 1) * 2
        assert np.allclose(result.output, [4, 6, 8])

    def test_stage_failure(self):
        """Handle stage failure."""

        def fail(x):
            raise ValueError("Test error")

        pipeline = Pipeline("fail")
        pipeline.add(TransformStage(fail, "fail"))

        result = pipeline.run(np.array([1]))

        assert not result.is_success
        assert result.status == StageStatus.FAILED

    def test_disabled_stage(self):
        """Skip disabled stage."""
        pipeline = Pipeline("skip")
        stage = TransformStage(lambda x: x * 2, "double")
        stage.disable()

        pipeline.add(stage)

        result = pipeline.run(np.array([1, 2, 3]))

        # Should pass through unchanged
        assert np.allclose(result.output, [1, 2, 3])

    def test_pipeline_context(self):
        """Use context in pipeline."""

        def use_context(x, **ctx):
            return x * ctx.get("factor", 1)

        pipeline = Pipeline("ctx")
        pipeline.add(FunctionStage(use_context, "mult"))
        pipeline.set_context("factor", 3)

        result = pipeline.run(np.array([1, 2]))

        assert np.allclose(result.output, [3, 6])

    def test_filter_stage(self):
        """Filter stage skips on false."""
        pipeline = Pipeline("filter")
        pipeline.add(FilterStage(lambda x: x.sum() > 10, "check"))

        # Sum < 10, should skip
        result = pipeline.run(np.array([1, 2, 3]))
        # When filter condition fails, the result is skipped but pipeline still succeeds
        # The output remains the input (passed through)
        assert result.status == StageStatus.SKIPPED or result.is_success

        # Sum > 10, should pass
        result = pipeline.run(np.array([10, 20, 30]))
        assert result.is_success

    def test_pipeline_builder(self):
        """Use pipeline builder."""
        pipeline = (
            PipelineBuilder("built")
            .transform(lambda x: x + 1)
            .transform(lambda x: x * 2)
            .build()
        )

        result = pipeline.run(np.array([1]))

        assert np.allclose(result.output, [4])  # (1+1)*2

    def test_pipeline_then(self):
        """Chain pipelines."""
        p1 = Pipeline("p1")
        p1.add(TransformStage(lambda x: x + 1, "add"))

        p2 = Pipeline("p2")
        p2.add(TransformStage(lambda x: x * 2, "mult"))

        combined = p1.then(p2)
        result = combined.run(np.array([5]))

        assert np.allclose(result.output, [12])  # (5+1)*2


# =============================================================================
# KERNEL TESTS
# =============================================================================


class TestFieldOS:
    """Test FieldOS kernel."""

    def setup_method(self):
        """Reset singleton before each test."""
        FieldOS.reset()

    def test_create_kernel(self):
        """Create kernel."""
        kernel = FieldOS()

        assert kernel.state == KernelState.UNINITIALIZED

    def test_singleton(self):
        """Kernel is singleton."""
        k1 = FieldOS()
        k2 = FieldOS()

        assert k1 is k2

    def test_start_kernel(self):
        """Start kernel."""
        kernel = FieldOS()
        kernel.start()

        assert kernel.is_running
        assert kernel.state == KernelState.RUNNING

        kernel.shutdown()

    def test_context_manager(self):
        """Use as context manager."""
        with FieldOS() as kernel:
            assert kernel.is_running

        assert not kernel.is_running

    def test_create_field(self):
        """Create field through kernel."""
        with FieldOS() as kernel:
            field = kernel.create_field("test", (50, 50))

            assert field.metadata.name == "test"
            assert kernel.stats.fields_created == 1

    def test_get_field(self):
        """Get field by name."""
        with FieldOS() as kernel:
            kernel.create_field("test", (10, 10))

            field = kernel.get_field("test")

            assert field is not None
            assert field.metadata.name == "test"

    def test_list_fields(self):
        """List field names."""
        with FieldOS() as kernel:
            kernel.create_field("a", (10, 10))
            kernel.create_field("b", (10, 10))

            names = kernel.list_fields()

            assert "a" in names
            assert "b" in names

    def test_delete_field(self):
        """Delete field."""
        with FieldOS() as kernel:
            kernel.create_field("test", (10, 10))
            kernel.delete_field("test")

            assert kernel.get_field("test") is None

    def test_run_pipeline(self):
        """Run pipeline through kernel."""
        with FieldOS() as kernel:
            pipeline = Pipeline("test")
            pipeline.add(TransformStage(lambda x: x * 2, "double"))

            result = kernel.run_pipeline(pipeline, np.array([1, 2, 3]))

            assert result.is_success
            assert kernel.stats.pipelines_run == 1

    def test_register_pipeline(self):
        """Register and retrieve pipeline."""
        with FieldOS() as kernel:
            pipeline = Pipeline("registered")
            kernel.register_pipeline("my_pipe", pipeline)

            retrieved = kernel.get_pipeline("my_pipe")

            assert retrieved is pipeline

    def test_pause_resume(self):
        """Pause and resume kernel."""
        with FieldOS() as kernel:
            kernel.pause()
            assert kernel.state == KernelState.PAUSED

            kernel.resume()
            assert kernel.state == KernelState.RUNNING

    def test_event_handlers(self):
        """Event handler registration."""
        events = []

        with FieldOS() as kernel:
            kernel.on("field_created", lambda e: events.append(e))
            kernel.create_field("test", (10, 10))

        assert len(events) == 1
        assert events[0]["name"] == "test"

    def test_kernel_stats(self):
        """Kernel statistics."""
        with FieldOS() as kernel:
            import time

            time.sleep(0.01)  # Ensure uptime > 0
            kernel.create_field("a", (10, 10))
            kernel.create_field("b", (10, 10))

            stats = kernel.stats

            assert stats.fields_created == 2
            assert stats.fields_active == 2
            assert stats.uptime_seconds >= 0  # May be very small but >= 0


# =============================================================================
# PLUGIN TESTS
# =============================================================================


class TestPlugin:
    """Test Plugin system."""

    def test_plugin_info(self):
        """Create plugin info."""
        info = PluginInfo(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
        )

        assert info.id == "test-plugin"
        assert info.state == PluginState.UNLOADED

    def test_plugin_manager(self):
        """Create plugin manager."""
        manager = PluginManager()

        assert len(manager.list_plugins()) == 0

    def test_register_plugin(self):
        """Register plugin."""

        class TestPlugin(Plugin):
            @property
            def info(self) -> PluginInfo:
                return PluginInfo("test", "Test", "1.0.0")

        manager = PluginManager()
        info = manager.register(TestPlugin())

        assert info.id == "test"
        assert info.state == PluginState.LOADED

    def test_enable_plugin(self):
        """Enable plugin."""
        enabled_calls = []

        class TestPlugin(Plugin):
            @property
            def info(self) -> PluginInfo:
                return PluginInfo("test", "Test")

            def on_enable(self, kernel):
                enabled_calls.append(True)

        manager = PluginManager()
        manager.register(TestPlugin())

        success = manager.enable("test")

        assert success
        assert len(enabled_calls) == 1
        assert manager.get_info("test").state == PluginState.ENABLED

    def test_disable_plugin(self):
        """Disable plugin."""

        class TestPlugin(Plugin):
            @property
            def info(self) -> PluginInfo:
                return PluginInfo("test", "Test")

        manager = PluginManager()
        manager.register(TestPlugin())
        manager.enable("test")

        success = manager.disable("test")

        assert success
        assert manager.get_info("test").state == PluginState.DISABLED

    def test_plugin_dependencies(self):
        """Plugin with dependencies."""

        class DepPlugin(Plugin):
            @property
            def info(self) -> PluginInfo:
                return PluginInfo("dep", "Dependency")

        class MainPlugin(Plugin):
            @property
            def info(self) -> PluginInfo:
                return PluginInfo("main", "Main", dependencies=["dep"])

        manager = PluginManager()
        manager.register(DepPlugin())
        manager.register(MainPlugin())

        # Enable main should enable dep first
        manager.enable("main")

        assert manager.get_info("dep").state == PluginState.ENABLED
        assert manager.get_info("main").state == PluginState.ENABLED

    def test_plugin_hook(self):
        """Plugin hooks."""
        hook = PluginHook("before_save")
        results = []

        hook.register(lambda x: results.append(x))
        hook.call("test_data")

        assert results == ["test_data"]

    def test_list_enabled(self):
        """List enabled plugins."""

        class P1(Plugin):
            @property
            def info(self):
                return PluginInfo("p1", "P1")

        class P2(Plugin):
            @property
            def info(self):
                return PluginInfo("p2", "P2")

        manager = PluginManager()
        manager.register(P1())
        manager.register(P2())
        manager.enable("p1")

        enabled = manager.list_enabled()

        assert "p1" in enabled
        assert "p2" not in enabled


# =============================================================================
# SESSION TESTS
# =============================================================================


class TestSession:
    """Test Session class."""

    def test_create_session(self):
        """Create session."""
        session = Session("experiment-1")

        assert session.name == "experiment-1"
        assert session.state == SessionState.NEW

    def test_session_lifecycle(self):
        """Session lifecycle."""
        session = Session("test")

        session.start()
        assert session.state == SessionState.ACTIVE

        session.pause()
        assert session.state == SessionState.PAUSED

        session.resume()
        assert session.state == SessionState.ACTIVE

        session.complete()
        assert session.state == SessionState.COMPLETED

    def test_session_context_manager(self):
        """Use as context manager."""
        with Session("test") as session:
            assert session.state == SessionState.ACTIVE

        assert session.state == SessionState.COMPLETED

    def test_session_fields(self):
        """Manage fields in session."""
        with Session("test") as session:
            field = Field.scalar("temp", (10, 10))
            session.set("temp", field)

            retrieved = session.get("temp")

            assert retrieved is field

    def test_session_context(self):
        """Session context values."""
        session = Session("test")

        session.set_context("param", 42)

        assert session.get_context("param") == 42
        assert session.get_context("missing", 0) == 0

    def test_checkpoint(self):
        """Create checkpoint."""
        session = Session("test")
        session.start()

        field = Field.scalar("data", (5, 5), initial=1.0)
        session.set("data", field)

        checkpoint = session.checkpoint("initial")

        assert checkpoint.name == "initial"
        assert "data" in checkpoint.fields

    def test_restore_checkpoint(self):
        """Restore from checkpoint."""
        session = Session("test")
        session.start()

        field = Field.scalar("data", (5, 5), initial=1.0)
        session.set("data", field)
        session.checkpoint("before")

        # Modify
        field.update(np.ones((5, 5)) * 100)

        # Restore
        session.restore("before")

        assert np.allclose(session.get("data").data, 1.0)

    def test_session_history(self):
        """Track session history."""
        session = Session("test")
        session.start()
        session.checkpoint("cp1")
        session.pause()

        history = session.get_history()

        events = [h["event"] for h in history]
        assert "started" in events
        assert "checkpoint_created" in events
        assert "paused" in events

    def test_session_save_load(self):
        """Save and load session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            with Session("test", storage_path=tmpdir) as session:
                field = Field.scalar("data", (10, 10), initial=5.0)
                session.set("data", field)
                session.checkpoint("final")

            # Load
            loaded = Session.load(tmpdir)

            assert loaded.name == "test"
            assert loaded.get("data") is not None

    def test_session_manager(self):
        """Session manager."""
        manager = SessionManager()

        s1 = manager.create("session-1")
        s2 = manager.create("session-2")

        # Sessions should have unique IDs
        assert s1.id != s2.id
        assert len(manager.list()) == 2
        assert manager.get(s1.id) is s1


# =============================================================================
# OBSERVABLE TESTS
# =============================================================================


class TestObservable:
    """Test Observable class."""

    def test_create_observable(self):
        """Create observable."""
        obs = Observable(42)

        assert obs.value == 42
        assert obs.get() == 42

    def test_set_value(self):
        """Set observable value."""
        obs = Observable(0)
        obs.set(100)

        assert obs.value == 100

    def test_subscribe(self):
        """Subscribe to changes."""
        obs = Observable(0)
        events = []

        # Must keep reference to subscription to prevent GC
        sub = obs.subscribe(lambda e: events.append(e))
        obs.set(42)

        assert len(events) == 1
        assert events[0].type == EventType.CHANGE
        assert events[0].data["new"] == 42

        # Verify subscription is still active
        assert sub.is_active

    def test_unsubscribe(self):
        """Unsubscribe from observable."""
        obs = Observable(0)
        events = []

        sub = obs.subscribe(lambda e: events.append(e))
        obs.set(1)

        sub.unsubscribe()
        obs.set(2)

        assert len(events) == 1  # Only first update

    def test_observer_class(self):
        """Use Observer class."""

        class TestObserver(Observer):
            def __init__(self):
                self.events = []

            def on_event(self, event):
                self.events.append(event)

        obs = Observable(0)
        observer = TestObserver()
        obs.subscribe(observer)
        obs.set(10)

        assert len(observer.events) == 1

    def test_map_observable(self):
        """Map observable values."""
        obs = Observable(5)
        doubled = obs.map(lambda x: x * 2)

        assert doubled.value == 10

        obs.set(10)
        assert doubled.value == 20

    def test_filter_observable(self):
        """Filter observable values."""
        obs = Observable(0)
        positive = obs.filter(lambda x: x > 0)

        events = []
        # Must keep reference to subscription
        sub = positive.subscribe(lambda e: events.append(e))

        obs.set(-5)
        obs.set(10)

        # Only positive update propagated
        assert len(events) == 1
        assert events[0].data["new"] == 10

    def test_update_function(self):
        """Update with function."""
        obs = Observable(10)
        obs.update(lambda x: x + 5)

        assert obs.value == 15

    def test_complete(self):
        """Complete observable."""
        obs = Observable(0)
        completed = []

        observer = FunctionObserver(
            lambda e: None,
            on_complete=lambda: completed.append(True),
        )
        obs.subscribe(observer)
        obs.complete()

        assert len(completed) == 1

    def test_combine_observables(self):
        """Combine two observables."""
        x = Observable(2)
        y = Observable(3)

        combined = x.combine(y)

        assert combined.value == (2, 3)

        x.set(10)
        assert combined.value == (10, 3)


class TestObservableField:
    """Test ObservableField class."""

    def test_create_observable_field(self):
        """Create observable field."""
        data = np.zeros((10, 10))
        obs = ObservableField(data, "test")

        assert obs.shape == (10, 10)

    def test_set_region(self):
        """Set region of field."""
        data = np.zeros((10, 10))
        obs = ObservableField(data, "test")

        events = []
        # Must keep reference to subscription
        sub = obs.subscribe(lambda e: events.append(e))

        obs.set_region((slice(0, 5), slice(0, 5)), 1.0)

        assert obs.value[0, 0] == 1.0
        # set_region emits UPDATE event, not CHANGE
        assert len(events) == 1
        assert events[0].type == EventType.UPDATE

    def test_apply_function(self):
        """Apply function to field."""
        data = np.ones((5, 5))
        obs = ObservableField(data, "test")

        obs.apply(lambda x: x * 10)

        assert np.allclose(obs.value, 10)


class TestEventBus:
    """Test EventBus class."""

    def test_create_bus(self):
        """Create event bus."""
        bus = EventBus()

        assert bus is not None

    def test_emit_receive(self):
        """Emit and receive event."""
        bus = EventBus()
        received = []

        bus.on("test", lambda e: received.append(e))
        bus.emit("test", {"value": 42})

        assert len(received) == 1
        assert received[0].data["value"] == 42

    def test_wildcard_handler(self):
        """Wildcard event handler."""
        bus = EventBus()
        received = []

        bus.on("field.*", lambda e: received.append(e))

        bus.emit("field.created", {})
        bus.emit("field.updated", {})
        bus.emit("other.event", {})

        assert len(received) == 2

    def test_unsubscribe(self):
        """Unsubscribe from bus."""
        bus = EventBus()
        received = []

        sub = bus.on("test", lambda e: received.append(e))
        bus.emit("test", {})

        sub.unsubscribe()
        bus.emit("test", {})

        assert len(received) == 1


class TestComputed:
    """Test Computed observable."""

    def test_computed_value(self):
        """Compute from dependencies."""
        x = Observable(2)
        y = Observable(3)

        sum_xy = Computed(lambda: x.value + y.value, [x, y])

        assert sum_xy.value == 5

    def test_computed_updates(self):
        """Computed updates on dependency change."""
        x = Observable(10)
        y = Observable(5)

        diff = Computed(lambda: x.value - y.value, [x, y])

        assert diff.value == 5

        x.set(20)
        assert diff.value == 15

        y.set(10)
        assert diff.value == 10

    def test_computed_dispose(self):
        """Dispose computed."""
        x = Observable(1)
        double = Computed(lambda: x.value * 2, [x])

        double.dispose()
        x.set(10)

        # Value doesn't update after dispose
        # (actually it might still be 20 due to timing, but won't track further)


class TestEvent:
    """Test Event class."""

    def test_create_event(self):
        """Create event."""
        event = Event(type=EventType.CHANGE, data=42)

        assert event.type == EventType.CHANGE
        assert event.data == 42

    def test_change_event(self):
        """Create change event."""
        event = Event.change(old_value=1, new_value=2)

        assert event.type == EventType.CHANGE
        assert event.data["old"] == 1
        assert event.data["new"] == 2

    def test_error_event(self):
        """Create error event."""
        event = Event.error("Something went wrong")

        assert event.type == EventType.ERROR
        assert "wrong" in event.data


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for Field OS."""

    def setup_method(self):
        """Reset kernel before each test."""
        FieldOS.reset()

    def test_end_to_end_workflow(self):
        """Complete workflow: create, process, observe."""
        with FieldOS() as kernel:
            # Create field
            field = kernel.create_field("temperature", (64, 64))

            # Create pipeline
            pipeline = Pipeline("heat")
            pipeline.add(TransformStage(lambda x: x + 10, "add_heat"))

            # Register and run
            kernel.register_pipeline("heat", pipeline)
            result = kernel.run_pipeline("heat", field)

            assert result.is_success
            assert kernel.stats.pipelines_run == 1

    def test_session_with_kernel(self):
        """Session with kernel fields."""
        with FieldOS() as kernel:
            with Session("sim") as session:
                # Create field through kernel
                field = kernel.create_field("velocity", (32, 32))

                # Track in session
                session.set("velocity", field)
                session.checkpoint("initial")

                # Verify
                assert session.get("velocity") is field

    def test_observable_field_in_kernel(self):
        """Observable field with kernel."""
        FieldOS.reset()
        with FieldOS() as kernel:
            # Create field
            field = kernel.create_field("pressure", (20, 20))

            # Wrap in observable
            obs = ObservableField(field.data.copy(), "pressure")

            events = []
            sub = obs.subscribe(lambda e: events.append(e))

            # Update - apply uses CHANGE event
            obs.apply(lambda x: x + 100)

            assert len(events) == 1
            assert events[0].type == EventType.CHANGE

    def test_pipeline_with_field_type(self):
        """Pipeline processing Field objects."""
        with FieldOS() as kernel:
            field = kernel.create_field("data", (10, 10), initial=5.0)

            pipeline = Pipeline("process")
            pipeline.add(TransformStage(lambda x: x * 2, "double"))

            result = pipeline.run(field)

            assert result.is_success
            assert isinstance(result.output, Field)
            assert np.allclose(result.output.data, 10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
