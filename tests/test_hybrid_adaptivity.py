"""Tests for Phase G: Hybrid Local Corrections + QoI-Driven Adaptivity.

Validates:
1. LocalTile — construction, validation, blending masks, indices
2. HybridField — backbone + tile composition, evaluation, diagnostics
3. FeatureSensor — 1D/2D gradient, jump, curvature detection
4. TileActivationPolicy — budget enforcement, deactivation, pruning
5. HybridRoundPolicy — rank reduction with tiles
6. tile_from_mask utility
7. QoITarget — specification
8. ConvergenceTrend — state detection (converging, converged, etc.)
9. QoIHistory — multi-QoI tracking, worst state
10. AdaptiveRankPolicy — rank increase/decrease/hold decisions
11. Sanitizers — whitelist-only outputs
12. IP boundary — no forbidden field leakage

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
import pytest
import numpy as np
from numpy.typing import NDArray


# ══════════════════════════════════════════════════════════════════════
# §1  LocalTile — construction and validation
# ══════════════════════════════════════════════════════════════════════

class TestLocalTile:
    """Test LocalTile construction and properties."""

    def test_basic_construction(self) -> None:
        from ontic.engine.vm.hybrid_field import LocalTile, TileKind
        data = np.zeros((10,), dtype=np.float64)
        tile = LocalTile(origin=(5,), extent=(10,), data=data)
        assert tile.n_points == 10
        assert tile.n_dims == 1
        assert tile.kind == TileKind.CUSTOM

    def test_2d_construction(self) -> None:
        from ontic.engine.vm.hybrid_field import LocalTile, TileKind
        data = np.ones((8, 8), dtype=np.float64)
        tile = LocalTile(
            origin=(10, 20), extent=(8, 8), data=data,
            kind=TileKind.SHOCK_BAND,
        )
        assert tile.n_points == 64
        assert tile.n_dims == 2

    def test_shape_mismatch_raises(self) -> None:
        from ontic.engine.vm.hybrid_field import LocalTile
        data = np.zeros((10,), dtype=np.float64)
        with pytest.raises(ValueError, match="does not match"):
            LocalTile(origin=(5,), extent=(20,), data=data)

    def test_invalid_weight_raises(self) -> None:
        from ontic.engine.vm.hybrid_field import LocalTile
        data = np.zeros((10,), dtype=np.float64)
        with pytest.raises(ValueError, match="Weight"):
            LocalTile(origin=(0,), extent=(10,), data=data, weight=1.5)

    def test_global_indices_1d(self) -> None:
        from ontic.engine.vm.hybrid_field import LocalTile
        data = np.zeros((5,), dtype=np.float64)
        tile = LocalTile(origin=(10,), extent=(5,), data=data)
        (indices,) = tile.global_indices()
        np.testing.assert_array_equal(indices, [10, 11, 12, 13, 14])

    def test_blending_mask_1d(self) -> None:
        from ontic.engine.vm.hybrid_field import LocalTile
        data = np.zeros((20,), dtype=np.float64)
        tile = LocalTile(origin=(0,), extent=(20,), data=data, weight=1.0)
        mask = tile.blending_mask(taper_width=4)
        assert mask.shape == (20,)
        # Edges should be tapered
        assert mask[0] < 0.5
        # Center should be full
        assert mask[10] == pytest.approx(1.0)

    def test_blending_mask_2d(self) -> None:
        from ontic.engine.vm.hybrid_field import LocalTile
        data = np.zeros((16, 16), dtype=np.float64)
        tile = LocalTile(origin=(0, 0), extent=(16, 16), data=data)
        mask = tile.blending_mask(taper_width=4)
        assert mask.shape == (16, 16)
        # Center should be 1.0
        assert mask[8, 8] == pytest.approx(1.0)
        # Corner should be tapered
        assert mask[0, 0] < 0.5

    def test_weight_scales_mask(self) -> None:
        from ontic.engine.vm.hybrid_field import LocalTile
        data = np.zeros((10,), dtype=np.float64)
        tile = LocalTile(origin=(0,), extent=(10,), data=data, weight=0.5)
        mask = tile.blending_mask(taper_width=2)
        assert np.max(mask) <= 0.5 + 1e-15


# ══════════════════════════════════════════════════════════════════════
# §2  HybridField — backbone + tiles
# ══════════════════════════════════════════════════════════════════════

class TestHybridField:
    """Test HybridField construction and evaluation."""

    def test_empty_construction(self) -> None:
        from ontic.engine.vm.hybrid_field import HybridField
        hf = HybridField(backbone=None, name="test")
        assert hf.n_tiles == 0
        assert hf.local_point_count == 0

    def test_add_tile(self) -> None:
        from ontic.engine.vm.hybrid_field import HybridField, LocalTile
        hf = HybridField(backbone=None)
        data = np.zeros((10,), dtype=np.float64)
        tile = LocalTile(origin=(0,), extent=(10,), data=data)
        hf.add_tile(tile)
        assert hf.n_tiles == 1
        assert hf.local_point_count == 10

    def test_remove_tile(self) -> None:
        from ontic.engine.vm.hybrid_field import HybridField, LocalTile
        hf = HybridField(backbone=None)
        data = np.zeros((10,), dtype=np.float64)
        tile = LocalTile(origin=(0,), extent=(10,), data=data)
        hf.add_tile(tile)
        removed = hf.remove_tile(0)
        assert hf.n_tiles == 0
        assert removed is tile

    def test_clear_tiles(self) -> None:
        from ontic.engine.vm.hybrid_field import HybridField, LocalTile
        hf = HybridField(backbone=None)
        for i in range(5):
            data = np.zeros((10,), dtype=np.float64)
            hf.add_tile(LocalTile(origin=(i * 10,), extent=(10,), data=data))
        hf.clear_tiles()
        assert hf.n_tiles == 0

    def test_evaluate_pure_backbone(self) -> None:
        from ontic.engine.vm.hybrid_field import HybridField
        from ontic.engine.vm.qtt_tensor import QTTTensor
        # Create a simple QTT tensor (constant = 1)
        backbone = QTTTensor.constant(1.0, bits_per_dim=(6,))
        hf = HybridField(backbone=backbone)
        result = hf.evaluate_on_grid((64,))
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_evaluate_with_correction(self) -> None:
        from ontic.engine.vm.hybrid_field import HybridField, LocalTile
        from ontic.engine.vm.qtt_tensor import QTTTensor
        backbone = QTTTensor.constant(0.0, bits_per_dim=(6,))
        hf = HybridField(backbone=backbone)
        # Add a correction of 5.0 in the middle (no taper for simplicity)
        correction = np.full((4,), 5.0, dtype=np.float64)
        tile = LocalTile(origin=(30,), extent=(4,), data=correction)
        hf.add_tile(tile)
        result = hf.evaluate_on_grid((64,))
        # Non-tile region should be ~0
        assert abs(result[0]) < 1e-10
        # Tile center: should have correction * blending ≈ 5.0 (center)
        # With taper_width=4 and extent=4, the mask is complex, but
        # the center values should be > 0
        assert result[31] > 0

    def test_coverage_fraction(self) -> None:
        from ontic.engine.vm.hybrid_field import HybridField, LocalTile
        from ontic.engine.vm.qtt_tensor import QTTTensor
        backbone = QTTTensor.constant(0.0, bits_per_dim=(8,))  # 256 pts
        hf = HybridField(backbone=backbone)
        # Add tile covering 32 points = 12.5% of 256
        data = np.zeros((32,), dtype=np.float64)
        hf.add_tile(LocalTile(origin=(0,), extent=(32,), data=data))
        assert hf.local_coverage_fraction == pytest.approx(32 / 256)
        assert hf.is_coverage_healthy  # 12.5% < 25%

    def test_coverage_unhealthy(self) -> None:
        from ontic.engine.vm.hybrid_field import (
            HybridField, LocalTile, MAX_LOCAL_COVERAGE,
        )
        from ontic.engine.vm.qtt_tensor import QTTTensor
        backbone = QTTTensor.constant(0.0, bits_per_dim=(6,))  # 64 pts
        hf = HybridField(backbone=backbone)
        # Coverage > 25%: 32 of 64 = 50%
        data = np.zeros((32,), dtype=np.float64)
        hf.add_tile(LocalTile(origin=(0,), extent=(32,), data=data))
        assert not hf.is_coverage_healthy

    def test_diagnostics(self) -> None:
        from ontic.engine.vm.hybrid_field import (
            HybridField, LocalTile, TileKind,
        )
        hf = HybridField(backbone=None)
        data = np.zeros((10,), dtype=np.float64)
        hf.add_tile(LocalTile(
            origin=(0,), extent=(10,), data=data, kind=TileKind.SHOCK_BAND,
        ))
        diag = hf.diagnostics()
        assert diag["n_tiles"] == 1
        assert diag["tile_kinds"] == ["SHOCK_BAND"]


# ══════════════════════════════════════════════════════════════════════
# §3  FeatureSensor — detection
# ══════════════════════════════════════════════════════════════════════

class TestFeatureSensor:
    """Test feature detection sensors."""

    def test_gradient_sensor_detects_shock(self) -> None:
        from ontic.engine.vm.hybrid_field import (
            detect_features_1d, FeatureSensorConfig, SensorKind,
        )
        # Sharp step function → large gradient at discontinuity
        N = 128
        field = np.zeros(N, dtype=np.float64)
        field[N // 2:] = 1.0
        h = 1.0 / N

        mask = detect_features_1d(
            field, h,
            config=FeatureSensorConfig(
                kind=SensorKind.GRADIENT_MAGNITUDE,
                threshold=0.1 / h,  # scaled to grid
                min_band_width=4,
            ),
        )
        # Should detect near the discontinuity
        assert mask[N // 2]
        # Should be clear far from discontinuity
        assert not mask[0]
        assert not mask[-1]

    def test_smooth_field_no_detection(self) -> None:
        from ontic.engine.vm.hybrid_field import (
            detect_features_1d, FeatureSensorConfig, SensorKind,
        )
        N = 128
        x = np.linspace(0, 1, N)
        field = np.sin(2 * np.pi * x)  # smooth
        h = 1.0 / N

        mask = detect_features_1d(
            field, h,
            config=FeatureSensorConfig(
                kind=SensorKind.GRADIENT_MAGNITUDE,
                threshold=1e6,  # very high threshold
            ),
        )
        assert not np.any(mask)

    def test_jump_indicator(self) -> None:
        from ontic.engine.vm.hybrid_field import (
            detect_features_1d, FeatureSensorConfig, SensorKind,
        )
        N = 128
        field = np.zeros(N, dtype=np.float64)
        field[60:68] = 10.0  # localized bump
        h = 1.0 / N

        mask = detect_features_1d(
            field, h,
            config=FeatureSensorConfig(
                kind=SensorKind.JUMP_INDICATOR,
                threshold=1.0,
                min_band_width=2,
            ),
        )
        # Should detect around the bump edges
        assert np.any(mask[55:75])

    def test_curvature_sensor(self) -> None:
        from ontic.engine.vm.hybrid_field import (
            detect_features_1d, FeatureSensorConfig, SensorKind,
        )
        N = 256
        x = np.linspace(0, 1, N)
        h = 1.0 / N
        # Smooth but high-curvature bump: narrow Gaussian
        field = np.exp(-((x - 0.5) / 0.01) ** 2)

        # Compute expected Laplacian scale to set reasonable threshold
        # |∇²f| at center ≈ 2/σ² * f ≈ 2/(0.01²) * 1 = 20000
        mask = detect_features_1d(
            field, h,
            config=FeatureSensorConfig(
                kind=SensorKind.CURVATURE,
                threshold=100.0,  # well below peak curvature
                min_band_width=4,
            ),
        )
        # Should detect near the Gaussian peak
        assert mask[N // 2]

    def test_tiles_from_mask(self) -> None:
        from ontic.engine.vm.hybrid_field import (
            tiles_from_mask_1d, TileKind,
        )
        N = 100
        mask = np.zeros(N, dtype=bool)
        mask[20:30] = True
        mask[70:80] = True
        correction = np.ones(N, dtype=np.float64) * 0.5

        tiles = tiles_from_mask_1d(mask, correction, kind=TileKind.SHOCK_BAND)
        assert len(tiles) == 2
        assert tiles[0].origin == (20,)
        assert tiles[0].extent == (10,)
        assert tiles[1].origin == (70,)


# ══════════════════════════════════════════════════════════════════════
# §4  TileActivationPolicy
# ══════════════════════════════════════════════════════════════════════

class TestTileActivationPolicy:
    """Test tile activation/deactivation policy."""

    def test_budget_enforcement(self) -> None:
        from ontic.engine.vm.hybrid_field import (
            TileActivationPolicy, LocalTile,
        )
        policy = TileActivationPolicy(max_tiles=2, max_total_points=1000)
        existing: list[LocalTile] = []
        candidates = [
            LocalTile(origin=(i * 10,), extent=(10,),
                      data=np.zeros((10,), dtype=np.float64))
            for i in range(5)
        ]
        accepted = policy.should_activate(existing, candidates)
        assert len(accepted) == 2

    def test_point_budget(self) -> None:
        from ontic.engine.vm.hybrid_field import (
            TileActivationPolicy, LocalTile,
        )
        policy = TileActivationPolicy(max_tiles=100, max_total_points=25)
        existing: list[LocalTile] = []
        candidates = [
            LocalTile(origin=(i * 10,), extent=(10,),
                      data=np.zeros((10,), dtype=np.float64))
            for i in range(5)
        ]
        accepted = policy.should_activate(existing, candidates)
        # Only 2 tiles of 10 pts each fit in 25 budget
        assert len(accepted) == 2

    def test_deactivation(self) -> None:
        from ontic.engine.vm.hybrid_field import (
            TileActivationPolicy, LocalTile,
        )
        policy = TileActivationPolicy(deactivation_threshold=1e-6)
        tiny = LocalTile(
            origin=(0,), extent=(10,),
            data=np.full((10,), 1e-10, dtype=np.float64),
        )
        big = LocalTile(
            origin=(0,), extent=(10,),
            data=np.full((10,), 1.0, dtype=np.float64),
        )
        assert policy.should_deactivate(tiny)
        assert not policy.should_deactivate(big)

    def test_prune(self) -> None:
        from ontic.engine.vm.hybrid_field import (
            TileActivationPolicy, LocalTile, HybridField,
        )
        policy = TileActivationPolicy(deactivation_threshold=0.1)
        hf = HybridField(backbone=None)
        hf.add_tile(LocalTile(
            origin=(0,), extent=(5,),
            data=np.full((5,), 1e-5, dtype=np.float64),
        ))
        hf.add_tile(LocalTile(
            origin=(10,), extent=(5,),
            data=np.full((5,), 1.0, dtype=np.float64),
        ))
        removed = policy.prune(hf)
        assert len(removed) == 1
        assert hf.n_tiles == 1


# ══════════════════════════════════════════════════════════════════════
# §5  HybridRoundPolicy
# ══════════════════════════════════════════════════════════════════════

class TestHybridRoundPolicy:
    """Test hybrid truncation policy."""

    def test_no_tiles_full_rank(self) -> None:
        from ontic.engine.vm.hybrid_field import HybridRoundPolicy
        policy = HybridRoundPolicy(backbone_max_rank=64)
        assert policy.effective_backbone_rank(0) == 64

    def test_with_tiles_reduced_rank(self) -> None:
        from ontic.engine.vm.hybrid_field import HybridRoundPolicy
        policy = HybridRoundPolicy(
            backbone_max_rank=64, aggressive_factor=0.5,
        )
        rank = policy.effective_backbone_rank(5)
        assert rank == 32  # 64 * 0.5

    def test_never_below_four(self) -> None:
        from ontic.engine.vm.hybrid_field import HybridRoundPolicy
        policy = HybridRoundPolicy(
            backbone_max_rank=8, aggressive_factor=0.1,
        )
        rank = policy.effective_backbone_rank(10)
        assert rank >= 4


# ══════════════════════════════════════════════════════════════════════
# §6  QoITarget
# ══════════════════════════════════════════════════════════════════════

class TestQoITarget:
    """Test QoI target specification."""

    def test_defaults(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import QoITarget
        t = QoITarget(name="drag")
        assert t.abs_tolerance == 1e-6
        assert t.rel_tolerance == 1e-4
        assert t.priority == 1
        assert t.field_name == ""

    def test_custom(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import QoITarget
        t = QoITarget(
            name="L2_error_u",
            abs_tolerance=1e-8,
            rel_tolerance=1e-6,
            priority=2,
            field_name="u",
        )
        assert t.name == "L2_error_u"
        assert t.field_name == "u"

    def test_frozen(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import QoITarget
        t = QoITarget(name="test")
        with pytest.raises(AttributeError):
            t.name = "other"  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════
# §7  ConvergenceTrend
# ══════════════════════════════════════════════════════════════════════

class TestConvergenceTrend:
    """Test convergence trend detection."""

    def test_insufficient_data(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            ConvergenceTrend, QoITarget, ConvergenceState,
        )
        trend = ConvergenceTrend(target=QoITarget(name="test"))
        assert trend.state == ConvergenceState.INSUFFICIENT_DATA
        trend.record(1.0, 0)
        assert trend.state == ConvergenceState.INSUFFICIENT_DATA

    def test_converged_absolute(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            ConvergenceTrend, QoITarget, ConvergenceState,
        )
        trend = ConvergenceTrend(
            target=QoITarget(name="test", abs_tolerance=0.01),
        )
        # Values within abs_tolerance of each other
        trend.record(1.000, 0)
        trend.record(1.001, 1)
        trend.record(1.002, 2)
        assert trend.state == ConvergenceState.CONVERGED

    def test_converged_relative(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            ConvergenceTrend, QoITarget, ConvergenceState,
        )
        trend = ConvergenceTrend(
            target=QoITarget(name="test", abs_tolerance=1e-15, rel_tolerance=0.01),
        )
        trend.record(100.0, 0)
        trend.record(100.1, 1)
        trend.record(100.05, 2)
        # Spread = 0.1, relative = 0.1/100.05 ≈ 0.001 < 0.01
        assert trend.state == ConvergenceState.CONVERGED

    def test_diverging(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            ConvergenceTrend, QoITarget, ConvergenceState,
        )
        trend = ConvergenceTrend(
            target=QoITarget(name="test", abs_tolerance=1e-15),
        )
        # Differences increasing: 1, 10, 100
        trend.record(0.0, 0)
        trend.record(1.0, 1)
        trend.record(11.0, 2)
        trend.record(111.0, 3)
        assert trend.state == ConvergenceState.DIVERGING

    def test_converging(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            ConvergenceTrend, QoITarget, ConvergenceState,
        )
        trend = ConvergenceTrend(
            target=QoITarget(name="test", abs_tolerance=1e-15),
        )
        # Differences decreasing but not yet converged
        trend.record(100.0, 0)
        trend.record(10.0, 1)
        trend.record(1.0, 2)
        # Diffs: 90, 9 — ratio < 0.9, so converging
        assert trend.state == ConvergenceState.CONVERGING

    def test_latest(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            ConvergenceTrend, QoITarget,
        )
        trend = ConvergenceTrend(target=QoITarget(name="test"))
        assert math.isnan(trend.latest)
        trend.record(42.0, 0)
        assert trend.latest == 42.0

    def test_estimated_rate(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            ConvergenceTrend, QoITarget,
        )
        trend = ConvergenceTrend(target=QoITarget(name="test"))
        assert math.isnan(trend.estimated_rate)
        trend.record(100.0, 0)
        trend.record(10.0, 1)
        trend.record(1.0, 2)
        rate = trend.estimated_rate
        # d1 = 90, d2 = 9 → rate = ln(9/90) ≈ -2.3
        assert rate < 0  # converging: negative rate

    def test_clear(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            ConvergenceTrend, QoITarget,
        )
        trend = ConvergenceTrend(target=QoITarget(name="test"))
        trend.record(1.0, 0)
        trend.record(2.0, 1)
        trend.clear()
        assert trend.n_samples == 0


# ══════════════════════════════════════════════════════════════════════
# §8  QoIHistory
# ══════════════════════════════════════════════════════════════════════

class TestQoIHistory:
    """Test multi-QoI tracking."""

    def test_construction(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import QoIHistory, QoITarget
        history = QoIHistory(targets=[
            QoITarget(name="drag"),
            QoITarget(name="lift"),
        ])
        assert len(history.targets) == 2

    def test_record_and_get(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import QoIHistory, QoITarget
        history = QoIHistory(targets=[QoITarget(name="drag")])
        history.record("drag", 0.5, 0)
        trend = history.get_trend("drag")
        assert trend.latest == 0.5

    def test_record_unknown_raises(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import QoIHistory, QoITarget
        history = QoIHistory(targets=[QoITarget(name="drag")])
        with pytest.raises(KeyError, match="lift"):
            history.record("lift", 0.5, 0)

    def test_add_target(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import QoIHistory, QoITarget
        history = QoIHistory()
        history.add_target(QoITarget(name="drag"))
        history.record("drag", 1.0, 0)
        assert history.get_trend("drag").latest == 1.0

    def test_all_converged(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            QoIHistory, QoITarget, ConvergenceState,
        )
        history = QoIHistory(targets=[
            QoITarget(name="a", abs_tolerance=0.1),
            QoITarget(name="b", abs_tolerance=0.1),
        ])
        for i in range(5):
            history.record("a", 1.0 + 0.001 * i, i)
            history.record("b", 2.0 + 0.001 * i, i)
        assert history.all_converged

    def test_any_diverging(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import QoIHistory, QoITarget
        history = QoIHistory(targets=[
            QoITarget(name="ok", abs_tolerance=0.1),
            QoITarget(name="bad", abs_tolerance=1e-15),
        ])
        # "ok" converges
        for i in range(5):
            history.record("ok", 1.0, i)
        # "bad" diverges
        history.record("bad", 0.0, 0)
        history.record("bad", 1.0, 1)
        history.record("bad", 11.0, 2)
        history.record("bad", 111.0, 3)
        assert history.any_diverging

    def test_worst_state(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            QoIHistory, QoITarget, ConvergenceState,
        )
        history = QoIHistory(targets=[
            QoITarget(name="a", abs_tolerance=0.1),
        ])
        # Insufficient data
        assert history.worst_state == ConvergenceState.INSUFFICIENT_DATA

    def test_summary(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import QoIHistory, QoITarget
        history = QoIHistory(targets=[QoITarget(name="drag")])
        history.record("drag", 0.5, 0)
        s = history.summary()
        assert "drag" in s
        assert "all_converged" in s
        assert "worst_state" in s


# ══════════════════════════════════════════════════════════════════════
# §9  AdaptiveRankPolicy
# ══════════════════════════════════════════════════════════════════════

class TestAdaptiveRankPolicy:
    """Test QoI-driven rank adaptation."""

    def test_default_construction(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import AdaptiveRankPolicy
        policy = AdaptiveRankPolicy()
        assert policy.base_max_rank == 64
        assert policy.rank_increment == 4
        assert policy.min_rank == 4

    def test_register_field(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import AdaptiveRankPolicy
        policy = AdaptiveRankPolicy()
        policy.register_field("u", initial_rank=32)
        spec = policy.get_field_spec("u")
        assert spec.max_rank == 32

    def test_auto_register(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import AdaptiveRankPolicy
        policy = AdaptiveRankPolicy(base_max_rank=48)
        spec = policy.get_field_spec("v")  # not pre-registered
        assert spec.max_rank == 48

    def test_rank_increase_on_stagnation(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            AdaptiveRankPolicy, QoIHistory, QoITarget, RankAction,
        )
        policy = AdaptiveRankPolicy(
            base_max_rank=32,
            rank_increment=8,
            evaluation_interval=1,
        )
        policy.register_field("u", initial_rank=32)
        history = QoIHistory(targets=[
            QoITarget(name="drag", field_name="u", abs_tolerance=1e-15),
        ])
        # Record stagnating values: diffs are ~ constant
        history.record("drag", 10.0, 0)
        history.record("drag", 9.0, 1)
        history.record("drag", 8.0, 2)
        history.record("drag", 7.0, 3)

        specs = policy.evaluate(history, timestep=1)
        spec = specs["u"]
        # Stagnating → increase
        assert spec.max_rank > 32

    def test_rank_decrease_on_convergence(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            AdaptiveRankPolicy, QoIHistory, QoITarget, RankAction,
        )
        policy = AdaptiveRankPolicy(
            base_max_rank=64,
            rank_decrement=4,
            evaluation_interval=1,
        )
        policy.register_field("u", initial_rank=64)
        history = QoIHistory(targets=[
            QoITarget(name="drag", field_name="u", abs_tolerance=0.1),
        ])
        # Converged: values within tolerance
        history.record("drag", 1.000, 0)
        history.record("drag", 1.001, 1)
        history.record("drag", 1.002, 2)

        specs = policy.evaluate(history, timestep=1)
        assert specs["u"].max_rank < 64

    def test_hold_when_converging(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            AdaptiveRankPolicy, QoIHistory, QoITarget, RankAction,
        )
        policy = AdaptiveRankPolicy(
            base_max_rank=64,
            evaluation_interval=1,
        )
        policy.register_field("u", initial_rank=64)
        history = QoIHistory(targets=[
            QoITarget(name="drag", field_name="u", abs_tolerance=1e-15),
        ])
        # Converging: diffs decreasing
        history.record("drag", 100.0, 0)
        history.record("drag", 10.0, 1)
        history.record("drag", 1.0, 2)

        specs = policy.evaluate(history, timestep=1)
        assert specs["u"].max_rank == 64  # no change

    def test_respects_ceiling(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            AdaptiveRankPolicy, QoIHistory, QoITarget,
        )
        policy = AdaptiveRankPolicy(
            base_max_rank=120,
            max_rank_ceiling=128,
            rank_increment=16,
            evaluation_interval=1,
        )
        policy.register_field("u", initial_rank=120)
        history = QoIHistory(targets=[
            QoITarget(name="drag", field_name="u", abs_tolerance=1e-15),
        ])
        # Diverging
        history.record("drag", 0.0, 0)
        history.record("drag", 1.0, 1)
        history.record("drag", 11.0, 2)
        history.record("drag", 111.0, 3)

        specs = policy.evaluate(history, timestep=1)
        assert specs["u"].max_rank <= 128

    def test_respects_floor(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            AdaptiveRankPolicy, QoIHistory, QoITarget,
        )
        policy = AdaptiveRankPolicy(
            base_max_rank=8, min_rank=4,
            rank_decrement=8,
            evaluation_interval=1,
        )
        policy.register_field("u", initial_rank=8)
        history = QoIHistory(targets=[
            QoITarget(name="drag", field_name="u", abs_tolerance=1.0),
        ])
        history.record("drag", 1.0, 0)
        history.record("drag", 1.0, 1)
        history.record("drag", 1.0, 2)

        specs = policy.evaluate(history, timestep=1)
        assert specs["u"].max_rank >= 4

    def test_should_stop(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            AdaptiveRankPolicy, QoIHistory, QoITarget,
        )
        policy = AdaptiveRankPolicy()
        history = QoIHistory(targets=[
            QoITarget(name="drag", abs_tolerance=0.1),
        ])
        history.record("drag", 1.0, 0)
        history.record("drag", 1.0, 1)
        history.record("drag", 1.0, 2)
        assert policy.should_stop(history)

    def test_diagnostics(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import AdaptiveRankPolicy
        policy = AdaptiveRankPolicy()
        policy.register_field("u", initial_rank=32)
        diag = policy.diagnostics()
        assert "u" in diag
        assert diag["u"]["max_rank"] == 32  # type: ignore[index]

    def test_skip_non_evaluation_timestep(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            AdaptiveRankPolicy, QoIHistory, QoITarget,
        )
        policy = AdaptiveRankPolicy(evaluation_interval=10)
        policy.register_field("u", initial_rank=64)
        history = QoIHistory(targets=[
            QoITarget(name="drag", field_name="u", abs_tolerance=0.1),
        ])
        history.record("drag", 1.0, 0)
        history.record("drag", 1.0, 1)
        history.record("drag", 1.0, 2)

        # timestep 5 is not a multiple of 10
        specs = policy.evaluate(history, timestep=5)
        assert specs["u"].max_rank == 64  # unchanged


# ══════════════════════════════════════════════════════════════════════
# §10  Sanitizers
# ══════════════════════════════════════════════════════════════════════

class TestHybridSanitizer:
    """Test hybrid field diagnostic sanitizer."""

    def test_allows_whitelisted(self) -> None:
        from ontic.engine.vm.hybrid_field import sanitize_hybrid_diagnostics
        raw = {
            "n_tiles": 3,
            "local_point_count": 100,
            "coverage_healthy": True,
        }
        safe = sanitize_hybrid_diagnostics(raw)
        assert len(safe) == 3

    def test_strips_forbidden(self) -> None:
        from ontic.engine.vm.hybrid_field import sanitize_hybrid_diagnostics
        raw = {
            "n_tiles": 3,
            "tile_data_raw": "FORBIDDEN",
            "backbone_cores": "FORBIDDEN",
        }
        safe = sanitize_hybrid_diagnostics(raw)
        assert len(safe) == 1


class TestAdaptivitySanitizer:
    """Test adaptivity diagnostic sanitizer."""

    def test_allows_whitelisted(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            sanitize_adaptivity_diagnostics,
        )
        raw = {
            "all_converged": True,
            "worst_state": "CONVERGED",
            "total_evaluations": 100,
        }
        safe = sanitize_adaptivity_diagnostics(raw)
        assert len(safe) == 3

    def test_strips_forbidden(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import (
            sanitize_adaptivity_diagnostics,
        )
        raw = {
            "all_converged": True,
            "internal_rank_trajectory": [1, 2, 3],
            "field_core_shapes": "FORBIDDEN",
        }
        safe = sanitize_adaptivity_diagnostics(raw)
        assert len(safe) == 1


# ══════════════════════════════════════════════════════════════════════
# §11  IP Boundary
# ══════════════════════════════════════════════════════════════════════

class TestIPBoundaryPhaseG:
    """Verify Phase G modules do not leak forbidden fields."""

    def test_hybrid_diagnostics_clean(self) -> None:
        from ontic.engine.vm.hybrid_field import HybridField
        from physics_os.core.sanitizer import FORBIDDEN_FIELDS
        hf = HybridField(backbone=None)
        diag = hf.diagnostics()
        for key in diag:
            assert key not in FORBIDDEN_FIELDS

    def test_adaptivity_diagnostics_clean(self) -> None:
        from ontic.engine.vm.qoi_adaptivity import AdaptiveRankPolicy
        from physics_os.core.sanitizer import FORBIDDEN_FIELDS
        policy = AdaptiveRankPolicy()
        policy.register_field("u")
        diag = policy.diagnostics()
        for key in diag:
            assert key not in FORBIDDEN_FIELDS
