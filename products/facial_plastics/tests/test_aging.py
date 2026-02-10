"""Tests for sim/aging.py — long-horizon aging trajectory predictor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from products.facial_plastics.sim.aging import (
    AgingFactor,
    AgingRiskProfile,
    AgingTrajectory,
    AgingTrajectoryResult,
    GraftType,
    GRAFT_HALF_LIVES,
    TissueSnapshot,
    _bone_resorption,
    _collagen_decay,
    _elastin_decay,
    _fat_volume_evolution,
    _graft_resorption,
    _gravity_descent,
    _muscle_atrophy,
    _skin_stiffness_evolution,
    _skin_thickness_evolution,
)
from products.facial_plastics.core.types import StructureType


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def default_profile() -> AgingRiskProfile:
    return AgingRiskProfile()


@pytest.fixture
def high_risk_profile() -> AgingRiskProfile:
    return AgingRiskProfile(
        baseline_age_years=60,
        uv_multiplier=2.0,
        smoking_multiplier=2.0,
        genetic_multiplier=1.5,
        bmi_change_per_year=-0.5,
        skin_type_factor=1.3,
    )


@pytest.fixture
def predictor() -> AgingTrajectory:
    return AgingTrajectory()


@pytest.fixture
def high_risk_predictor(high_risk_profile: AgingRiskProfile) -> AgingTrajectory:
    return AgingTrajectory(risk_profile=high_risk_profile)


# ── AgingRiskProfile ─────────────────────────────────────────────

class TestAgingRiskProfile:
    def test_defaults(self, default_profile: AgingRiskProfile) -> None:
        assert default_profile.baseline_age_years == 45.0
        assert default_profile.uv_multiplier == 1.0
        assert default_profile.smoking_multiplier == 1.0

    def test_composite_multiplier_default(self, default_profile: AgingRiskProfile) -> None:
        # All factors at 1.0 → composite = 1.0
        assert default_profile.composite_multiplier == pytest.approx(1.0, rel=0.01)

    def test_composite_multiplier_elevated(self, high_risk_profile: AgingRiskProfile) -> None:
        assert high_risk_profile.composite_multiplier > 1.0

    def test_to_dict(self, default_profile: AgingRiskProfile) -> None:
        d = default_profile.to_dict()
        assert "baseline_age" in d
        assert "composite" in d
        assert d["composite"] == pytest.approx(1.0, rel=0.01)


# ── Tissue decay functions ────────────────────────────────────────

class TestCollagenDecay:
    def test_zero_years(self) -> None:
        frac = _collagen_decay(0.0, 45.0)
        assert frac == pytest.approx(1.0, abs=1e-10)

    def test_decreases_over_time(self) -> None:
        f5 = _collagen_decay(5.0, 45.0)
        f10 = _collagen_decay(10.0, 45.0)
        f20 = _collagen_decay(20.0, 45.0)
        assert 1.0 > f5 > f10 > f20 > 0.0

    def test_rate_multiplier(self) -> None:
        normal = _collagen_decay(10.0, 45.0, 1.0)
        fast = _collagen_decay(10.0, 45.0, 2.0)
        assert fast < normal

    def test_floor(self) -> None:
        # Even after many years, should not go to zero
        frac = _collagen_decay(200.0, 45.0, 5.0)
        assert frac >= 0.05


class TestElastinDecay:
    def test_zero_years(self) -> None:
        frac = _elastin_decay(0.0, 45.0)
        assert frac == pytest.approx(1.0, abs=1e-10)

    def test_decreases(self) -> None:
        assert _elastin_decay(10.0, 45.0) < 1.0

    def test_uv_accelerates(self) -> None:
        normal = _elastin_decay(10.0, 45.0, 1.0)
        uv_heavy = _elastin_decay(10.0, 45.0, 3.0)
        assert uv_heavy < normal


class TestSkinStiffness:
    def test_full_collagen_elastin(self) -> None:
        s = _skin_stiffness_evolution(1.0, 1.0)
        assert s == pytest.approx(1.0, rel=0.01)

    def test_reduced_collagen(self) -> None:
        s = _skin_stiffness_evolution(0.5, 1.0)
        assert s < 1.0

    def test_monotonic_with_collagen(self) -> None:
        s_low = _skin_stiffness_evolution(0.3, 1.0)
        s_mid = _skin_stiffness_evolution(0.6, 1.0)
        s_high = _skin_stiffness_evolution(0.9, 1.0)
        assert s_low < s_mid < s_high


class TestSkinThickness:
    def test_zero_years_young(self) -> None:
        frac = _skin_thickness_evolution(0.0, 30.0)
        assert frac == pytest.approx(1.0, abs=1e-6)

    def test_thins_over_time(self) -> None:
        f10 = _skin_thickness_evolution(10.0, 45.0)
        f20 = _skin_thickness_evolution(20.0, 45.0)
        assert f20 < f10 < 1.0


class TestFatVolume:
    def test_zero_years(self) -> None:
        frac = _fat_volume_evolution(0.0, 45.0)
        assert frac == pytest.approx(1.0, rel=0.05)

    def test_decreases_after_35(self) -> None:
        assert _fat_volume_evolution(20.0, 40.0) < 1.0

    def test_bmi_gain_increases(self) -> None:
        gaining = _fat_volume_evolution(10.0, 45.0, bmi_change_per_year=0.5)
        losing = _fat_volume_evolution(10.0, 45.0, bmi_change_per_year=-0.5)
        assert gaining > losing


class TestBoneResorption:
    def test_zero_years_young(self) -> None:
        # Before age 40, no resorption
        frac = _bone_resorption(0.0, 30.0)
        assert frac == pytest.approx(1.0, abs=1e-6)

    def test_zero_years_already_over_40(self) -> None:
        # At age 45, already 5 active years of resorption
        frac = _bone_resorption(0.0, 45.0)
        assert frac < 1.0
        assert frac > 0.95

    def test_slow_decline(self) -> None:
        f10 = _bone_resorption(10.0, 50.0)
        assert 0.9 < f10 < 1.0  # slow rate

    def test_floor(self) -> None:
        frac = _bone_resorption(200.0, 45.0)
        assert frac >= 0.6


class TestMuscleAtrophy:
    def test_zero_years_young(self) -> None:
        # Before age 50, no atrophy
        frac = _muscle_atrophy(0.0, 40.0)
        assert frac == pytest.approx(1.0, abs=1e-6)

    def test_slow_for_young(self) -> None:
        frac = _muscle_atrophy(10.0, 35.0)
        # Should be minimal before age 50
        assert frac > 0.99


class TestGravityDescent:
    def test_zero_years(self) -> None:
        d = _gravity_descent(0.0, 1.0, 1.0)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_increases_over_time(self) -> None:
        d5 = _gravity_descent(5.0, 1.0, 1.0)
        d10 = _gravity_descent(10.0, 1.0, 1.0)
        assert d10 > d5 > 0

    def test_more_descent_with_weaker_tissue(self) -> None:
        strong = _gravity_descent(10.0, 1.0, 1.0)
        weak = _gravity_descent(10.0, 0.5, 0.5)
        assert weak > strong


# ── Graft resorption ─────────────────────────────────────────────

class TestGraftResorption:
    def test_septal_stable(self) -> None:
        frac = _graft_resorption(20.0, GraftType.AUTOLOGOUS_SEPTAL)
        assert frac == pytest.approx(1.0)

    def test_permanent_implant(self) -> None:
        frac = _graft_resorption(50.0, GraftType.POROUS_POLYETHYLENE)
        assert frac == pytest.approx(1.0)

    def test_fascia_resorbs(self) -> None:
        frac = _graft_resorption(10.0, GraftType.FASCIA_LATA)
        # Half-life 5 years → after 10 years, ~25% remaining
        assert frac < 0.5

    def test_zero_years(self) -> None:
        for gt in GraftType:
            frac = _graft_resorption(0.0, gt)
            assert frac == pytest.approx(1.0, abs=1e-6)


# ── TissueSnapshot ───────────────────────────────────────────────

class TestTissueSnapshot:
    def test_to_dict(self) -> None:
        snap = TissueSnapshot(
            time_years=5.0, patient_age=50.0,
            collagen_fraction=0.95, elastin_fraction=0.97,
            skin_stiffness_multiplier=0.93, skin_thickness_fraction=0.98,
            fat_volume_fraction=0.96, bone_volume_fraction=0.99,
            muscle_mass_fraction=1.0, gravity_descent_mm=1.5,
            graft_volume_fraction=1.0,
        )
        d = snap.to_dict()
        assert d["time_years"] == 5.0
        assert d["patient_age"] == 50.0
        assert "collagen" in d


# ── AgingTrajectoryResult ─────────────────────────────────────────

class TestAgingTrajectoryResult:
    def test_at_year_zero(self, predictor: AgingTrajectory) -> None:
        result = predictor.predict(horizon_years=10, n_points=20)
        snap = result.at_year(0.0)
        assert snap is not None
        assert snap.collagen_fraction == pytest.approx(1.0, abs=0.01)

    def test_at_year_interpolation(self, predictor: AgingTrajectory) -> None:
        result = predictor.predict(horizon_years=20, n_points=5)
        snap = result.at_year(7.5)
        assert snap is not None
        # Should be between year 5 and year 10 snapshots
        s5 = result.at_year(5.0)
        s10 = result.at_year(10.0)
        assert s5 is not None and s10 is not None
        assert s5.collagen_fraction >= snap.collagen_fraction >= s10.collagen_fraction

    def test_at_year_beyond_horizon(self, predictor: AgingTrajectory) -> None:
        result = predictor.predict(horizon_years=10, n_points=5)
        snap = result.at_year(50.0)
        assert snap is not None  # clamps to last

    def test_summary(self, predictor: AgingTrajectory) -> None:
        result = predictor.predict(horizon_years=10, n_points=5)
        s = result.summary()
        assert "10" in s

    def test_to_dict(self, predictor: AgingTrajectory) -> None:
        result = predictor.predict(horizon_years=10, n_points=5)
        d = result.to_dict()
        assert "snapshots" in d
        assert len(d["snapshots"]) == 5


# ── AgingTrajectory predictor ─────────────────────────────────────

class TestAgingTrajectory:
    def test_predict_default(self, predictor: AgingTrajectory) -> None:
        result = predictor.predict()
        assert isinstance(result, AgingTrajectoryResult)
        assert result.n_points == 50
        assert result.horizon_years == 20.0
        assert len(result.snapshots) == 50

    def test_predict_custom_horizon(self, predictor: AgingTrajectory) -> None:
        result = predictor.predict(horizon_years=30, n_points=10)
        assert result.horizon_years == 30.0
        assert result.n_points == 10

    def test_predict_monotonic_collagen(self, predictor: AgingTrajectory) -> None:
        result = predictor.predict(horizon_years=20, n_points=20)
        collagens = [s.collagen_fraction for s in result.snapshots]
        for i in range(1, len(collagens)):
            assert collagens[i] <= collagens[i - 1]

    def test_predict_monotonic_descent(self, predictor: AgingTrajectory) -> None:
        result = predictor.predict(horizon_years=20, n_points=20)
        descents = [s.gravity_descent_mm for s in result.snapshots]
        for i in range(1, len(descents)):
            assert descents[i] >= descents[i - 1]

    def test_high_risk_worse(
        self,
        predictor: AgingTrajectory,
        high_risk_predictor: AgingTrajectory,
    ) -> None:
        normal = predictor.predict(horizon_years=20, n_points=5)
        risky = high_risk_predictor.predict(horizon_years=20, n_points=5)

        last_normal = normal.snapshots[-1]
        last_risky = risky.snapshots[-1]

        assert last_risky.collagen_fraction < last_normal.collagen_fraction
        assert last_risky.gravity_descent_mm > last_normal.gravity_descent_mm

    def test_invalid_horizon(self, predictor: AgingTrajectory) -> None:
        with pytest.raises(ValueError, match="positive"):
            predictor.predict(horizon_years=0)

    def test_invalid_n_points(self, predictor: AgingTrajectory) -> None:
        with pytest.raises(ValueError, match="n_points"):
            predictor.predict(n_points=1)

    def test_predict_tissue_params(self, predictor: AgingTrajectory) -> None:
        base = {"mu": 30.0e3, "kappa": 300.0e3}
        modified = predictor.predict_tissue_params(
            StructureType.SKIN_ENVELOPE, base, years_post_op=10.0,
        )
        assert modified["mu"] < base["mu"]
        assert modified["kappa"] <= base["kappa"]

    def test_predict_tissue_params_fat(self, predictor: AgingTrajectory) -> None:
        base = {"mu": 0.5e3, "kappa": 50.0e3}
        modified = predictor.predict_tissue_params(
            StructureType.FAT_SUBCUTANEOUS, base, years_post_op=10.0,
        )
        assert "mu" in modified

    def test_compare_scenarios(self, predictor: AgingTrajectory) -> None:
        base = AgingRiskProfile(smoking_multiplier=1.0)
        smoker = AgingRiskProfile(smoking_multiplier=2.0)
        comparison = predictor.compare_scenarios(base, smoker, horizon_years=10, n_points=5)
        assert "base" in comparison
        assert "modified" in comparison
        assert "differentials" in comparison
        assert len(comparison["differentials"]) == 5

    def test_graft_type_property(self) -> None:
        t = AgingTrajectory(graft_type=GraftType.AUTOLOGOUS_RIB)
        assert t.graft_type == GraftType.AUTOLOGOUS_RIB

    def test_risk_profile_property(self, high_risk_profile: AgingRiskProfile) -> None:
        t = AgingTrajectory(risk_profile=high_risk_profile)
        assert t.risk_profile.uv_multiplier == 2.0
