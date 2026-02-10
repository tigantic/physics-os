"""Tests for postop/dashboard.py — validation dashboard builder."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from products.facial_plastics.postop.dashboard import (
    AccuracyPanel,
    CalibrationPanel,
    CohortPanel,
    DashboardPayload,
    OutlierCase,
    OutlierPanel,
    RiskPanel,
    SurgeonPanel,
    TrendPanel,
    ValidationDashboard,
)


# ── Helpers ───────────────────────────────────────────────────────

def _make_cohort_cases(n: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    """Create a batch of case dicts matching CohortAnalytics schema."""
    rng = np.random.RandomState(seed)
    surgeons = ["Dr. A", "Dr. B", "Dr. C"]
    cases: List[Dict[str, Any]] = []
    for i in range(n):
        cases.append({
            "case_id": f"case_{i:03d}",
            "procedure": "primary_rhinoplasty",
            "surgeon_id": surgeons[i % 3],
            "date": f"2024-{(i % 12) + 1:02d}-15",
            "age": float(rng.normal(45, 10)),
            "bmi": float(rng.normal(25, 4)),
            "aesthetic_score": float(rng.normal(0.85, 0.05)),
            "safety_score": float(rng.normal(0.90, 0.04)),
            "functional_score": float(rng.normal(0.88, 0.06)),
            "complication": bool(rng.random() < 0.2),
            "revision": bool(rng.random() < 0.1),
        })
    return cases


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def dashboard() -> ValidationDashboard:
    return ValidationDashboard()


@pytest.fixture
def cohort_dashboard() -> ValidationDashboard:
    """Dashboard with 20 cohort cases added."""
    d = ValidationDashboard()
    d.add_cohort_cases(_make_cohort_cases(20))
    return d


# ── Dashboard instantiation ──────────────────────────────────────

class TestValidationDashboard:
    def test_empty_dashboard(self, dashboard: ValidationDashboard) -> None:
        assert dashboard is not None

    def test_add_cohort_cases(self, dashboard: ValidationDashboard) -> None:
        cases = _make_cohort_cases(5)
        dashboard.add_cohort_cases(cases)
        # build shouldn't crash
        payload = dashboard.build()
        assert isinstance(payload, DashboardPayload)

    def test_build_empty_dashboard(self, dashboard: ValidationDashboard) -> None:
        """Building with no data should still return a valid payload."""
        payload = dashboard.build()
        assert isinstance(payload, DashboardPayload)
        assert payload.cohort.n_cases == 0
        assert payload.accuracy.n_cases == 0
        assert payload.calibration.is_calibrated is False

    def test_build_with_cohort_data(
        self, cohort_dashboard: ValidationDashboard,
    ) -> None:
        payload = cohort_dashboard.build()
        assert isinstance(payload, DashboardPayload)

    def test_build_has_cohort_panel(
        self, cohort_dashboard: ValidationDashboard,
    ) -> None:
        payload = cohort_dashboard.build()
        assert isinstance(payload.cohort, CohortPanel)
        assert payload.cohort.n_cases == 20
        assert isinstance(payload.cohort.procedure_counts, dict)

    def test_build_has_surgeon_panel(
        self, cohort_dashboard: ValidationDashboard,
    ) -> None:
        payload = cohort_dashboard.build()
        assert isinstance(payload.surgeons, SurgeonPanel)
        assert payload.surgeons.n_surgeons == 3
        assert len(payload.surgeons.surgeons) == 3
        assert payload.surgeons.top_performer is not None

    def test_build_has_risk_panel(
        self, cohort_dashboard: ValidationDashboard,
    ) -> None:
        payload = cohort_dashboard.build()
        assert isinstance(payload.risks, RiskPanel)
        assert payload.risks.n_factors >= 0

    def test_build_has_trend_panel(
        self, cohort_dashboard: ValidationDashboard,
    ) -> None:
        payload = cohort_dashboard.build()
        assert isinstance(payload.trends, TrendPanel)
        assert isinstance(payload.trends.metrics, dict)

    def test_set_outlier_threshold(self, dashboard: ValidationDashboard) -> None:
        dashboard.set_outlier_threshold(3.0)
        payload = dashboard.build()
        assert payload.outliers.z_threshold == 3.0


# ── Panel dataclasses ─────────────────────────────────────────────

class TestPanels:
    def test_accuracy_panel(self) -> None:
        panel = AccuracyPanel(
            n_cases=50,
            metrics=["tip_projection", "dorsal_height"],
            mae={"tip_projection": 0.8, "dorsal_height": 0.5},
            rmse={"tip_projection": 1.0, "dorsal_height": 0.7},
            max_error={"tip_projection": 3.0, "dorsal_height": 2.0},
            overall_grade="B",
            accuracy_profiles=[],
            bland_altman=[],
        )
        assert panel.n_cases == 50
        assert panel.overall_grade == "B"
        d = panel.to_dict()
        assert d["panel"] == "accuracy_overview"
        assert d["n_cases"] == 50

    def test_calibration_panel(self) -> None:
        panel = CalibrationPanel(
            is_calibrated=True,
            last_calibration_residual=0.002,
            n_calibration_cases=100,
            parameter_count=5,
            converged=True,
            parameter_summary={"param_0": 1.5, "param_1": 0.3},
        )
        assert panel.is_calibrated is True
        assert panel.converged is True
        d = panel.to_dict()
        assert d["panel"] == "calibration_status"
        assert d["converged"] is True

    def test_cohort_panel(self) -> None:
        panel = CohortPanel(
            n_cases=100,
            procedure_counts={"primary_rhinoplasty": 60, "septoplasty": 40},
            demographics={"age": {"mean": 45.0, "std": 10.0}},
            outcome_distributions={"aesthetic_score": {"mean": 0.85}},
        )
        assert panel.n_cases == 100
        d = panel.to_dict()
        assert d["panel"] == "cohort_summary"

    def test_surgeon_panel(self) -> None:
        panel = SurgeonPanel(
            surgeons=[{"surgeon_id": "Dr. A", "n_cases": 30}],
            n_surgeons=1,
            top_performer="Dr. A",
        )
        assert panel.n_surgeons == 1
        assert panel.top_performer == "Dr. A"
        d = panel.to_dict()
        assert d["panel"] == "surgeon_performance"

    def test_risk_panel(self) -> None:
        panel = RiskPanel(
            risk_factors=[{"factor": "age", "odds_ratio": 1.5}],
            n_factors=1,
            top_risk="age",
        )
        assert panel.n_factors == 1
        d = panel.to_dict()
        assert d["panel"] == "risk_map"

    def test_trend_panel(self) -> None:
        panel = TrendPanel(
            metrics={
                "aesthetic_score": [
                    {"period": "2024-Q1", "n_cases": 5, "mean": 0.85},
                ],
            },
        )
        assert "aesthetic_score" in panel.metrics
        d = panel.to_dict()
        assert d["panel"] == "trends"

    def test_outlier_panel(self) -> None:
        oc = OutlierCase(
            case_id="c1",
            metric="tip_projection",
            predicted=5.0,
            actual=8.5,
            error=3.5,
            z_score=3.5,
        )
        panel = OutlierPanel(outliers=[oc], n_outliers=1, z_threshold=2.0)
        assert panel.n_outliers == 1
        assert panel.outliers[0].z_score == 3.5
        d = panel.to_dict()
        assert d["panel"] == "outliers"
        assert len(d["cases"]) == 1


# ── DashboardPayload ─────────────────────────────────────────────

class TestDashboardPayload:
    def test_to_dict(self, cohort_dashboard: ValidationDashboard) -> None:
        payload = cohort_dashboard.build()
        d = payload.to_dict()
        assert isinstance(d, dict)
        assert "cohort" in d
        assert "accuracy" in d
        assert "calibration" in d
        assert "surgeons" in d
        assert "risks" in d
        assert "trends" in d
        assert "outliers" in d

    def test_payload_cohort_not_empty(
        self, cohort_dashboard: ValidationDashboard,
    ) -> None:
        payload = cohort_dashboard.build()
        assert payload.cohort.n_cases == 20

    def test_payload_calibration_defaults(
        self, cohort_dashboard: ValidationDashboard,
    ) -> None:
        """Without calibration data, defaults should be sensible."""
        payload = cohort_dashboard.build()
        assert payload.calibration.is_calibrated is False
        assert payload.calibration.converged is False
        assert payload.calibration.parameter_count == 0


# ── OutlierCase ───────────────────────────────────────────────────

class TestOutlierCase:
    def test_creation(self) -> None:
        oc = OutlierCase(
            case_id="outlier-001",
            metric="healing_weeks",
            predicted=12.0,
            actual=28.0,
            error=16.0,
            z_score=4.2,
        )
        assert oc.case_id == "outlier-001"
        assert oc.z_score > 3.0
        assert oc.error == 16.0

    def test_to_dict(self) -> None:
        oc = OutlierCase(
            case_id="oc1",
            metric="m",
            predicted=1.0,
            actual=5.0,
            error=4.0,
            z_score=2.5,
        )
        d = oc.to_dict()
        assert d["case_id"] == "oc1"
        assert d["z_score"] == 2.5
        assert d["predicted"] == 1.0
        assert d["actual"] == 5.0
