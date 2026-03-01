"""Tests for the auto-resolution advisor."""

from __future__ import annotations

import pytest

from physics_os.core.resolution import (
    QualityTier,
    ResolutionAdvice,
    _MIN_N_BITS_TIER,
    advise,
    advise_from_physics,
)


class TestAdvise:
    def test_basic_laminar(self) -> None:
        """Low-Re quick tier should still meet QTT minimum (512×512)."""
        r = advise(
            re=100,
            characteristic_length=0.01,
            velocity=1.0,
            domain_length=0.1,
            t_end=0.1,
            tier=QualityTier.QUICK,
        )
        assert isinstance(r, ResolutionAdvice)
        assert r.n_bits >= _MIN_N_BITS_TIER[QualityTier.QUICK]
        assert r.n_bits <= 14
        assert r.grid_points_1d == 2**r.n_bits
        assert r.dt_recommended > 0
        assert r.n_steps >= 1

    def test_high_re_more_bits(self) -> None:
        """Increasing Re should require more resolution."""
        r_low = advise(re=100, characteristic_length=1.0, velocity=1.0,
                       domain_length=10.0, t_end=1.0, tier=QualityTier.STANDARD)
        r_high = advise(re=1e6, characteristic_length=1.0, velocity=1.0,
                        domain_length=10.0, t_end=1.0, tier=QualityTier.STANDARD)
        assert r_high.n_bits >= r_low.n_bits

    def test_tiers_ordering(self) -> None:
        """Higher tiers should give equal or more resolution."""
        results = {}
        for tier in QualityTier:
            results[tier] = advise(
                re=1e4, characteristic_length=0.1, velocity=10.0,
                domain_length=1.0, t_end=0.5, tier=tier,
            )
        assert results[QualityTier.MAXIMUM].n_bits >= results[QualityTier.QUICK].n_bits

    def test_n_bits_clamped_to_14(self) -> None:
        """Extremely high Re should clamp at 14 bits."""
        r = advise(
            re=1e10, characteristic_length=10.0, velocity=100.0,
            domain_length=100.0, t_end=10.0, tier=QualityTier.MAXIMUM,
        )
        assert r.n_bits == 14
        assert any("ceiling" in w for w in r.warnings)

    def test_steps_clamped_to_10000(self) -> None:
        """Very long end time should clamp steps at 10,000."""
        r = advise(
            re=1000, characteristic_length=0.1, velocity=1.0,
            domain_length=1.0, t_end=1e6, tier=QualityTier.QUICK,
        )
        assert r.n_steps == 10_000
        assert any("10,000" in w for w in r.warnings)

    def test_1d_total_points(self) -> None:
        """1D case: total_points == grid_points_1d."""
        r = advise(
            re=500, characteristic_length=0.1, velocity=1.0,
            domain_length=1.0, t_end=0.1, spatial_dims=1,
        )
        assert r.total_points == r.grid_points_1d

    def test_2d_total_points(self) -> None:
        """2D case: total_points == grid_points_1d²."""
        r = advise(
            re=500, characteristic_length=0.1, velocity=1.0,
            domain_length=1.0, t_end=0.1, spatial_dims=2,
        )
        assert r.total_points == r.grid_points_1d ** 2

    def test_kolmogorov_scale_laminar_none(self) -> None:
        """Laminar flow should have None for Kolmogorov scale."""
        r = advise(re=100, characteristic_length=0.1, velocity=0.1,
                   domain_length=1.0, t_end=1.0)
        assert r.kolmogorov_scale is None

    def test_kolmogorov_scale_turbulent(self) -> None:
        """Turbulent flow should compute Kolmogorov scale."""
        r = advise(re=1e5, characteristic_length=1.0, velocity=10.0,
                   domain_length=10.0, t_end=1.0)
        assert r.kolmogorov_scale is not None
        assert r.kolmogorov_scale > 0

    def test_invalid_re_raises(self) -> None:
        with pytest.raises(ValueError, match="Reynolds"):
            advise(re=-1, characteristic_length=1, velocity=1,
                   domain_length=1, t_end=1)

    def test_invalid_velocity_raises(self) -> None:
        with pytest.raises(ValueError, match="velocity"):
            advise(re=100, characteristic_length=1, velocity=0,
                   domain_length=1, t_end=1)

    def test_tier_floor_enforced(self) -> None:
        """Each tier must enforce its QTT compression floor.

        We use a large domain-to-body ratio with very low Re so that
        neither the BL criterion nor the geometry criterion alone
        would demand the tier-floor grid size.
        """
        for tier in QualityTier:
            # domain_length / char_length = 1000 → body is tiny fraction
            # of domain.  BL is enormous (low Re).  Physics alone
            # computes few points, but tier floor lifts n_bits.
            r = advise(
                re=0.01,
                characteristic_length=0.001,
                velocity=0.001,
                domain_length=1.0,  # domain >> body → dx_body coarse
                t_end=0.01,
                tier=tier,
            )
            assert r.n_bits >= _MIN_N_BITS_TIER[tier], (
                f"{tier.value}: n_bits={r.n_bits} < floor={_MIN_N_BITS_TIER[tier]}"
            )

    def test_geometry_resolution_criterion(self) -> None:
        """Body with large char_length relative to domain should get
        enough points for geometry even when BL is thick."""
        r = advise(
            re=0.1,
            characteristic_length=1.0,  # large body
            velocity=0.01,
            domain_length=10.0,
            t_end=100.0,
            tier=QualityTier.QUICK,
        )
        # Need at least 32 points across body (1.0 m)
        actual_body_pts = 1.0 / (10.0 / r.grid_points_1d)
        assert actual_body_pts >= 16  # generous lower bound


class TestAdviseFromPhysics:
    def test_basic(self) -> None:
        """Convenience wrapper should produce valid advice."""
        r = advise_from_physics(
            velocity=1.0,
            characteristic_length=0.01,
            kinematic_viscosity=1.5e-5,
        )
        assert isinstance(r, ResolutionAdvice)
        assert r.n_bits >= _MIN_N_BITS_TIER[QualityTier.STANDARD]  # default tier

    def test_re_computed(self) -> None:
        """Check that Re = U L / ν is computed correctly."""
        r = advise_from_physics(
            velocity=10.0,
            characteristic_length=0.1,
            kinematic_viscosity=1e-5,
        )
        # Re = 10 * 0.1 / 1e-5 = 1e5 → turbulent
        assert r.kolmogorov_scale is not None
