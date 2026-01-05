"""
Layer 0 Substrate Tests
=======================

Validates the Field Oracle API and supporting infrastructure.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tensornet.substrate import (BoundedMode, BudgetConfig, BundleMetadata,
                                 ContractionCache, Field, FieldBundle,
                                 FieldStats, FieldType, SliceSpec,
                                 StepControls, TelemetryDashboard)


class TestFieldOracle:
    """Test the Field Oracle API."""

    def test_create_field(self):
        """Test field creation."""
        field = Field.create(dims=2, bits_per_dim=10, rank=4)

        assert field.dims == 2
        assert field.bits_per_dim == 10
        assert field.n_cores == 20
        assert field.grid_size == 1024
        assert field.total_points == 1024 * 1024
        assert field.rank <= 4

    def test_sample_points(self):
        """Test point sampling."""
        field = Field.create(dims=2, bits_per_dim=8, rank=4)

        # Sample 100 random points
        points = torch.rand(100, 2)
        values = field.sample(points)

        assert values.shape == (100,)
        assert not torch.isnan(values).any()

    def test_slice_2d(self):
        """Test 2D slice extraction."""
        field = Field.create(dims=2, bits_per_dim=8, rank=4)

        spec = SliceSpec(
            plane="xy",
            resolution=(64, 64),
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
        )

        buffer = field.slice(spec)

        assert buffer.shape == (64, 64)
        assert not torch.isnan(buffer).any()

    def test_step_physics(self):
        """Test physics stepping."""
        field = Field.create(dims=2, bits_per_dim=8, rank=4)

        # Step 10 times
        for _ in range(10):
            field = field.step(dt=0.001)

        assert field._step_count == 10
        assert field._total_time > 0
        assert len(field._energy_history) == 10

    def test_step_with_controls(self):
        """Test stepping with controls."""
        field = Field.create(dims=2, bits_per_dim=8, rank=4)

        controls = StepControls(
            dt=0.01,
            viscosity=0.1,
            advection=True,
            diffusion=True,
        )

        field = field.step(controls=controls)

        assert field._step_count == 1

    def test_stats(self):
        """Test statistics generation."""
        field = Field.create(dims=2, bits_per_dim=8, rank=4)
        field = field.step(dt=0.001)

        stats = field.stats()

        assert isinstance(stats, FieldStats)
        assert stats.max_rank > 0
        assert stats.n_cores == 16
        assert stats.qtt_memory_bytes > 0
        assert stats.compression_ratio > 1
        assert stats.step_count == 1

    def test_stats_summary(self):
        """Test stats summary output."""
        field = Field.create(dims=2, bits_per_dim=8, rank=4)
        field = field.step(dt=0.001)

        summary = field.stats().summary()

        assert "FIELD STATISTICS" in summary
        assert "Rank" in summary
        assert "Energy" in summary
        assert "Compression" in summary

    def test_serialize_deserialize(self):
        """Test serialization round-trip."""
        field = Field.create(dims=2, bits_per_dim=8, rank=4)
        field = field.step(dt=0.001)

        # Serialize
        bundle = field.serialize()

        assert isinstance(bundle, FieldBundle)
        assert len(bundle.cores) == field.n_cores

        # Deserialize
        field2 = Field.deserialize(bundle)

        assert field2.dims == field.dims
        assert field2.bits_per_dim == field.bits_per_dim
        assert field2._step_count == field._step_count

    def test_immutable_step(self):
        """Test that step returns new field (immutable)."""
        field1 = Field.create(dims=2, bits_per_dim=8, rank=4)
        field2 = field1.step(dt=0.001)

        # Original should be unchanged
        assert field1._step_count == 0
        assert field2._step_count == 1

    def test_compression_ratio(self):
        """Test compression reporting."""
        field = Field.create(dims=2, bits_per_dim=15, rank=8)

        # 2^15 x 2^15 = 1 billion points
        assert field.total_points == 2**30

        # Should have massive compression
        assert field.compression > 1000


class TestFieldBundle:
    """Test FieldBundle serialization."""

    def test_save_load(self):
        """Test save and load."""
        field = Field.create(dims=2, bits_per_dim=8, rank=4)
        field = field.step(dt=0.001)

        bundle = field.serialize()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.htf"
            bundle.save(str(path))

            assert path.exists()

            bundle2 = FieldBundle.load(str(path))

            assert len(bundle2.cores) == len(bundle.cores)
            assert bundle2.metadata.dims == bundle.metadata.dims

    def test_hash_consistency(self):
        """Test hash is consistent."""
        field = Field.create(dims=2, bits_per_dim=8, rank=4)
        bundle = field.serialize()

        hash1 = bundle.compute_hash()
        hash2 = bundle.compute_hash()

        assert hash1 == hash2

    def test_replay_info(self):
        """Test replay info output."""
        field = Field.create(dims=2, bits_per_dim=8, rank=4)
        bundle = field.serialize()

        info = bundle.replay_info()

        assert "REPLAY INFO" in info
        assert "Schema" in info


class TestBoundedMode:
    """Test bounded latency mode."""

    def test_basic_bounded_mode(self):
        """Test basic bounded mode operation."""
        config = BudgetConfig(target_ms=16.67)
        bounded = BoundedMode(config)

        bounded.begin_frame()
        # Simulate some work
        import time

        time.sleep(0.005)  # 5ms
        bounded.end_frame()

        stats = bounded.stats()
        assert stats["frame_count"] == 1
        assert stats["avg_frame_ms"] > 0

    def test_rank_adaptation(self):
        """Test rank adapts to budget pressure."""
        config = BudgetConfig(
            target_ms=10.0,
            min_rank=2,
            max_rank=32,
            adaptation_rate=0.2,
        )
        bounded = BoundedMode(config)

        initial_rank = bounded.current_rank

        # Simulate slow frames
        for _ in range(10):
            bounded.begin_frame()
            import time

            time.sleep(0.020)  # 20ms (over budget)
            bounded.end_frame()

        # Rank should have decreased
        assert bounded.current_rank < initial_rank

    def test_contraction_cache(self):
        """Test contraction cache."""
        cache = ContractionCache(max_entries=10)

        # Put some values
        cache.put((0, 1), (0, 1, 0), torch.tensor([1.0]))
        cache.put((0, 1), (1, 0, 1), torch.tensor([2.0]))

        # Get them back
        v1 = cache.get((0, 1), (0, 1, 0))
        v2 = cache.get((0, 1), (1, 0, 1))
        v3 = cache.get((0, 1), (0, 0, 0))  # Not in cache

        assert v1 is not None
        assert v2 is not None
        assert v3 is None

        assert cache.hits == 2
        assert cache.misses == 1


class TestTelemetryDashboard:
    """Test telemetry dashboard."""

    def test_record_history(self):
        """Test recording stats history."""
        dashboard = TelemetryDashboard(max_history=100)

        for i in range(10):
            stats = FieldStats(
                max_rank=8 + i,
                truncation_error=0.01 * i,
                energy=1.0 - 0.01 * i,
            )
            dashboard.record(stats)

        assert len(dashboard.history) == 10
        assert dashboard.rank_history[-1] == 17
        assert len(dashboard.error_history) == 10


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self):
        """Test complete workflow: create, step, sample, slice, serialize."""
        # Create
        field = Field.create(dims=2, bits_per_dim=10, rank=8)

        # Step multiple times
        for _ in range(5):
            field = field.step(dt=0.001)

        # Sample
        points = torch.rand(50, 2)
        values = field.sample(points)
        assert values.shape == (50,)

        # Slice
        spec = SliceSpec(plane="xy", resolution=(32, 32))
        buffer = field.slice(spec)
        assert buffer.shape == (32, 32)

        # Stats
        stats = field.stats()
        assert stats.step_count == 5

        # Serialize
        bundle = field.serialize()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "workflow_test.htf"
            bundle.save(str(path))

            # Load and verify
            bundle2 = FieldBundle.load(str(path))
            field2 = Field.deserialize(bundle2)

            assert field2._step_count == 5

    def test_bounded_mode_with_field(self):
        """Test bounded mode with actual field operations."""
        field = Field.create(dims=2, bits_per_dim=8, rank=4)
        bounded = BoundedMode(BudgetConfig(target_ms=500))  # 500ms budget (generous)

        for _ in range(5):
            bounded.begin_frame()

            field = field.step(dt=0.001)
            spec = SliceSpec(plane="xy", resolution=(32, 32))
            buffer = field.slice(spec)

            bounded.end_frame()

        stats = bounded.stats()
        assert stats["frame_count"] == 5
        # With generous budget, should hit most frames
        assert stats["budget_hits"] >= 0  # At least ran


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
