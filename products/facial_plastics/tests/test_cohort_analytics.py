"""Tests for metrics/cohort_analytics.py — cohort statistical analysis engine."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pytest

from products.facial_plastics.metrics.cohort_analytics import (
    CohortAnalytics,
    CohortReport,
    DistributionStats,
    EffectSize,
    RiskFactor,
    SubgroupAnalysis,
    SurgeonProfile,
    TrendPoint,
    _cohens_d,
    _distribution_stats,
    _logistic_regression,
    _sigmoid,
)


# ── Fixtures ──────────────────────────────────────────────────────

def _make_case(
    i: int,
    rng: np.random.RandomState,
    *,
    surgeon_pool: List[str] = ["Dr. A", "Dr. B", "Dr. C"],
    procedure_pool: List[str] = ["primary_rhinoplasty", "revision_rhinoplasty"],
) -> Dict[str, Any]:
    """Build a single case dict matching the CohortAnalytics schema."""
    return {
        "case_id": f"case_{i:03d}",
        "procedure": procedure_pool[i % len(procedure_pool)],
        "surgeon_id": surgeon_pool[i % len(surgeon_pool)],
        "date": f"2024-{(i % 12) + 1:02d}-15",
        "age": float(rng.normal(45, 10)),
        "bmi": float(rng.normal(25, 4)),
        "aesthetic_score": float(rng.normal(0.85, 0.05)),
        "safety_score": float(rng.normal(0.9, 0.04)),
        "functional_score": float(rng.normal(0.88, 0.06)),
        "complication": bool(rng.random() < 0.2),
        "revision": bool(rng.random() < 0.1),
    }


@pytest.fixture
def analytics() -> CohortAnalytics:
    return CohortAnalytics()


@pytest.fixture
def populated_analytics() -> CohortAnalytics:
    """Analytics engine with 30 cases added."""
    ca = CohortAnalytics()
    rng = np.random.RandomState(42)
    for i in range(30):
        ca.add_case(_make_case(i, rng))
    return ca


# ── Statistical helpers ──────────────────────────────────────────

class TestStatHelpers:
    def test_sigmoid_zero(self) -> None:
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self) -> None:
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_sigmoid_large_negative(self) -> None:
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_distribution_stats_constant(self) -> None:
        vals = np.array([5.0, 5.0, 5.0, 5.0])
        stats = _distribution_stats("const", vals)
        assert isinstance(stats, DistributionStats)
        assert stats.variable == "const"
        assert stats.mean == pytest.approx(5.0)
        assert stats.std == pytest.approx(0.0, abs=1e-10)
        assert stats.median == pytest.approx(5.0)
        assert stats.n == 4

    def test_distribution_stats_known(self) -> None:
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = _distribution_stats("seq", vals)
        assert stats.mean == pytest.approx(3.0)
        assert stats.median == pytest.approx(3.0)
        assert stats.min_val == pytest.approx(1.0)
        assert stats.max_val == pytest.approx(5.0)
        assert stats.n == 5
        assert stats.q25 <= stats.median <= stats.q75
        assert stats.iqr == pytest.approx(stats.q75 - stats.q25)

    def test_distribution_stats_empty(self) -> None:
        stats = _distribution_stats("empty", np.array([], dtype=np.float64))
        assert stats.n == 0
        assert stats.mean == 0.0

    def test_distribution_stats_to_dict(self) -> None:
        vals = np.array([1.0, 2.0, 3.0])
        stats = _distribution_stats("test", vals)
        d = stats.to_dict()
        assert d["variable"] == "test"
        assert "mean" in d
        assert "std" in d
        assert "skewness" in d

    def test_cohens_d_equal_groups(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d, interp = _cohens_d(a, b)
        assert d == pytest.approx(0.0, abs=0.01)
        assert interp == "negligible"

    def test_cohens_d_large_effect(self) -> None:
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        d, interp = _cohens_d(a, b)
        assert abs(d) > 3.0
        assert interp == "large"

    def test_cohens_d_small_groups(self) -> None:
        a = np.array([1.0])
        b = np.array([2.0])
        d, interp = _cohens_d(a, b)
        assert d == 0.0
        assert interp == "negligible"

    def test_logistic_regression_separable(self) -> None:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([0, 0, 1, 1], dtype=np.float64)
        weights = _logistic_regression(X, y, lr=0.1, n_iter=200, l2_reg=0.01)
        assert weights.shape == (2,)
        # First feature should have positive weight (it determines y)
        assert weights[0] > 0

    def test_logistic_regression_constant(self) -> None:
        X = np.ones((10, 2), dtype=np.float64)
        y = np.zeros(10, dtype=np.float64)
        weights = _logistic_regression(X, y)
        # With all-zero labels, weights should be negative (pushing towards 0)
        assert (weights <= 0).all()
        # Weights should stay moderate with L2 regularisation
        assert np.abs(weights).max() < 5.0


# ── CohortAnalytics ──────────────────────────────────────────────

class TestCohortAnalytics:
    def test_empty_instantiation(self, analytics: CohortAnalytics) -> None:
        assert analytics.n_cases == 0

    def test_add_case_requires_case_id(self, analytics: CohortAnalytics) -> None:
        with pytest.raises(ValueError, match="case_id"):
            analytics.add_case({"procedure": "rhinoplasty"})

    def test_add_case(self, analytics: CohortAnalytics) -> None:
        analytics.add_case({
            "case_id": "test-001",
            "procedure": "primary_rhinoplasty",
            "surgeon_id": "Dr. X",
            "aesthetic_score": 0.9,
            "complication": False,
        })
        assert analytics.n_cases == 1

    def test_add_cases_batch(self, analytics: CohortAnalytics) -> None:
        cases = [
            {"case_id": f"c_{i}", "procedure": "primary_rhinoplasty"}
            for i in range(5)
        ]
        analytics.add_cases(cases)
        assert analytics.n_cases == 5

    def test_clear(self, populated_analytics: CohortAnalytics) -> None:
        assert populated_analytics.n_cases == 30
        populated_analytics.clear()
        assert populated_analytics.n_cases == 0

    def test_add_multiple_cases(self, populated_analytics: CohortAnalytics) -> None:
        assert populated_analytics.n_cases == 30

    def test_compute_distributions(self, populated_analytics: CohortAnalytics) -> None:
        dists = populated_analytics.compute_distributions(["aesthetic_score"])
        assert "aesthetic_score" in dists
        assert isinstance(dists["aesthetic_score"], DistributionStats)
        assert dists["aesthetic_score"].n == 30
        assert 0.5 < dists["aesthetic_score"].mean < 1.0

    def test_compute_distributions_auto_detect(
        self, populated_analytics: CohortAnalytics,
    ) -> None:
        dists = populated_analytics.compute_distributions()
        # Should auto-detect numeric fields
        assert len(dists) > 0
        assert "age" in dists

    def test_procedure_counts(self, populated_analytics: CohortAnalytics) -> None:
        counts = populated_analytics.procedure_counts()
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == 30

    def test_surgeon_profiles(self, populated_analytics: CohortAnalytics) -> None:
        profiles = populated_analytics.surgeon_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) == 3
        for profile in profiles:
            assert isinstance(profile, SurgeonProfile)
            assert profile.n_cases > 0
            assert profile.surgeon_id in ("Dr. A", "Dr. B", "Dr. C")
            assert 0.0 <= profile.complication_rate <= 1.0
            assert 0.0 <= profile.revision_rate <= 1.0

    def test_surgeon_profile_to_dict(
        self, populated_analytics: CohortAnalytics,
    ) -> None:
        profiles = populated_analytics.surgeon_profiles()
        d = profiles[0].to_dict()
        assert "surgeon_id" in d
        assert "n_cases" in d
        assert "mean_aesthetic_score" in d

    def test_identify_risk_factors(
        self, populated_analytics: CohortAnalytics,
    ) -> None:
        risks = populated_analytics.identify_risk_factors()
        assert isinstance(risks, list)
        for rf in risks:
            assert isinstance(rf, RiskFactor)
            assert rf.factor_name
            assert isinstance(rf.coefficient, float)
            assert rf.odds_ratio > 0
            assert 0.0 <= rf.relative_importance <= 1.0

    def test_identify_risk_factors_explicit_predictors(
        self, populated_analytics: CohortAnalytics,
    ) -> None:
        risks = populated_analytics.identify_risk_factors(
            outcome_field="complication",
            predictor_fields=["age", "bmi"],
        )
        assert isinstance(risks, list)
        factor_names = {r.factor_name for r in risks}
        assert factor_names <= {"age", "bmi"}

    def test_subgroup_analysis(self, populated_analytics: CohortAnalytics) -> None:
        sub = populated_analytics.subgroup_analysis(
            "aesthetic_score", "procedure",
        )
        assert isinstance(sub, SubgroupAnalysis)
        assert sub.metric == "aesthetic_score"
        assert isinstance(sub.subgroups, dict)
        assert len(sub.subgroups) > 0
        for name, stats in sub.subgroups.items():
            assert isinstance(stats, DistributionStats)
            assert stats.n > 0

    def test_subgroup_analysis_effect_sizes(
        self, populated_analytics: CohortAnalytics,
    ) -> None:
        sub = populated_analytics.subgroup_analysis(
            "aesthetic_score", "surgeon_id",
        )
        # With 3 surgeons we get C(3,2) = 3 pairwise comparisons
        assert len(sub.effect_sizes) == 3
        for es in sub.effect_sizes:
            assert isinstance(es, EffectSize)
            assert es.metric == "aesthetic_score"
            assert es.interpretation in (
                "negligible", "small", "medium", "large",
            )

    def test_temporal_trends(self, populated_analytics: CohortAnalytics) -> None:
        trends = populated_analytics.temporal_trends("aesthetic_score")
        assert isinstance(trends, list)
        for tp in trends:
            assert isinstance(tp, TrendPoint)
            assert tp.n_cases > 0
            assert tp.period  # non-empty string

    def test_generate_report(self, populated_analytics: CohortAnalytics) -> None:
        report = populated_analytics.generate_report()
        assert isinstance(report, CohortReport)
        assert report.n_cases == 30
        assert isinstance(report.demographics, dict)
        assert isinstance(report.outcome_distributions, dict)
        assert isinstance(report.procedure_counts, dict)
        assert isinstance(report.risk_factors, list)
        assert isinstance(report.surgeon_profiles, list)

    def test_generate_report_with_subgroups_and_trends(
        self, populated_analytics: CohortAnalytics,
    ) -> None:
        report = populated_analytics.generate_report(
            subgroup_metrics=[("aesthetic_score", "procedure")],
            trend_metrics=["aesthetic_score"],
        )
        assert len(report.subgroup_analyses) == 1
        assert isinstance(report.subgroup_analyses[0], SubgroupAnalysis)
        assert "aesthetic_score" in report.trends

    def test_generate_report_to_dict(
        self, populated_analytics: CohortAnalytics,
    ) -> None:
        report = populated_analytics.generate_report()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["n_cases"] == 30
        assert "demographics" in d
        assert "procedure_counts" in d
        assert "risk_factors" in d
        assert "surgeon_profiles" in d


# ── DistributionStats ─────────────────────────────────────────────

class TestDistributionStats:
    def test_to_dict(self) -> None:
        stats = DistributionStats(
            variable="test",
            n=100,
            mean=5.0,
            std=1.0,
            median=5.0,
            q25=4.0,
            q75=6.0,
            iqr=2.0,
            min_val=2.0,
            max_val=8.0,
            skewness=0.1,
            kurtosis=-0.2,
        )
        d = stats.to_dict()
        assert d["mean"] == 5.0
        assert d["n"] == 100
        assert d["variable"] == "test"
        assert "skewness" in d
        assert "kurtosis" in d


# ── EffectSize ────────────────────────────────────────────────────

class TestEffectSize:
    def test_creation(self) -> None:
        es = EffectSize(
            group_a="male",
            group_b="female",
            metric="aesthetic_score",
            cohens_d=0.15,
            mean_a=0.85,
            mean_b=0.87,
            n_a=20,
            n_b=25,
            interpretation="negligible",
        )
        assert es.cohens_d == 0.15
        assert es.interpretation == "negligible"

    def test_to_dict(self) -> None:
        es = EffectSize(
            group_a="A",
            group_b="B",
            metric="m",
            cohens_d=1.5,
            mean_a=1.0,
            mean_b=3.0,
            n_a=10,
            n_b=10,
            interpretation="large",
        )
        d = es.to_dict()
        assert d["cohens_d"] == 1.5
        assert d["interpretation"] == "large"
        assert d["group_a"] == "A"
