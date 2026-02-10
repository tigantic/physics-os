"""Validation dashboard — data provider for surgeon/admin dashboards.

Aggregates prediction validation data, cohort analytics, and
calibration status into structured dashboard payloads that feed
the UI's analytics views.

Dashboard panels:
  1. Accuracy overview — MAE, RMSE, correlation by metric
  2. Calibration status — parameter drift, convergence health
  3. Cohort summary — demographics, procedure mix, outcomes
  4. Surgeon performance — per-surgeon scorecards
  5. Risk map — highlighted risk factors from cohort analysis
  6. Trend charts — temporal metric evolution
  7. Outlier detection — flagged cases beyond LOA
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import ProcedureType
from ..metrics.cohort_analytics import (
    CohortAnalytics,
    CohortReport,
    DistributionStats,
    EffectSize,
    RiskFactor,
    SurgeonProfile,
)
from .calibration import CalibrationResult, ModelCalibrator
from .validation import (
    AccuracyProfile,
    BlandAltmanResult,
    MetricComparison,
    PredictionValidator,
    ValidationReport,
)

logger = logging.getLogger(__name__)


# ── Dashboard panel data structures ───────────────────────────────

@dataclass
class AccuracyPanel:
    """Accuracy overview panel data."""
    n_cases: int
    metrics: List[str]
    mae: Dict[str, float]
    rmse: Dict[str, float]
    max_error: Dict[str, float]
    overall_grade: str
    accuracy_profiles: List[Dict[str, Any]]
    bland_altman: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel": "accuracy_overview",
            "n_cases": self.n_cases,
            "metrics": self.metrics,
            "mae": self.mae,
            "rmse": self.rmse,
            "max_error": self.max_error,
            "overall_grade": self.overall_grade,
            "accuracy_profiles": self.accuracy_profiles,
            "bland_altman": self.bland_altman,
        }


@dataclass
class CalibrationPanel:
    """Calibration health panel data."""
    is_calibrated: bool
    last_calibration_residual: float
    n_calibration_cases: int
    parameter_count: int
    converged: bool
    parameter_summary: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel": "calibration_status",
            "is_calibrated": self.is_calibrated,
            "last_residual": round(self.last_calibration_residual, 6),
            "n_cases": self.n_calibration_cases,
            "n_parameters": self.parameter_count,
            "converged": self.converged,
            "parameters": self.parameter_summary,
        }


@dataclass
class CohortPanel:
    """Cohort summary panel data."""
    n_cases: int
    procedure_counts: Dict[str, int]
    demographics: Dict[str, Dict[str, Any]]
    outcome_distributions: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel": "cohort_summary",
            "n_cases": self.n_cases,
            "procedure_counts": self.procedure_counts,
            "demographics": self.demographics,
            "outcomes": self.outcome_distributions,
        }


@dataclass
class SurgeonPanel:
    """Surgeon performance panel data."""
    surgeons: List[Dict[str, Any]]
    n_surgeons: int
    top_performer: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel": "surgeon_performance",
            "n_surgeons": self.n_surgeons,
            "top_performer": self.top_performer,
            "surgeons": self.surgeons,
        }


@dataclass
class RiskPanel:
    """Risk factor panel data."""
    risk_factors: List[Dict[str, Any]]
    n_factors: int
    top_risk: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel": "risk_map",
            "n_factors": self.n_factors,
            "top_risk": self.top_risk,
            "factors": self.risk_factors,
        }


@dataclass
class TrendPanel:
    """Temporal trend panel data."""
    metrics: Dict[str, List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel": "trends",
            "metrics": self.metrics,
        }


@dataclass
class OutlierCase:
    """A case flagged as an outlier."""
    case_id: str
    metric: str
    predicted: float
    actual: float
    error: float
    z_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "metric": self.metric,
            "predicted": round(self.predicted, 4),
            "actual": round(self.actual, 4),
            "error": round(self.error, 4),
            "z_score": round(self.z_score, 2),
        }


@dataclass
class OutlierPanel:
    """Outlier detection panel data."""
    outliers: List[OutlierCase]
    n_outliers: int
    z_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel": "outliers",
            "n_outliers": self.n_outliers,
            "z_threshold": self.z_threshold,
            "cases": [o.to_dict() for o in self.outliers],
        }


@dataclass
class DashboardPayload:
    """Complete dashboard data payload."""
    accuracy: AccuracyPanel
    calibration: CalibrationPanel
    cohort: CohortPanel
    surgeons: SurgeonPanel
    risks: RiskPanel
    trends: TrendPanel
    outliers: OutlierPanel

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy.to_dict(),
            "calibration": self.calibration.to_dict(),
            "cohort": self.cohort.to_dict(),
            "surgeons": self.surgeons.to_dict(),
            "risks": self.risks.to_dict(),
            "trends": self.trends.to_dict(),
            "outliers": self.outliers.to_dict(),
        }


# ── ValidationDashboard ──────────────────────────────────────────

class ValidationDashboard:
    """Assembles dashboard data from validation and analytics sources.

    Combines:
      - PredictionValidator → accuracy panels
      - ModelCalibrator → calibration health
      - CohortAnalytics → demographics, outcomes, risk, trends
      - Outlier detection → flagged cases outside limits of agreement

    Usage::

        dashboard = ValidationDashboard()
        dashboard.set_validation_report(report)
        dashboard.set_calibration_result(cal_result)
        dashboard.add_cohort_cases(case_records)
        payload = dashboard.build()
        json_data = payload.to_dict()
    """

    def __init__(self) -> None:
        self._validation_report: Optional[ValidationReport] = None
        self._calibration_result: Optional[CalibrationResult] = None
        self._cohort = CohortAnalytics()
        self._z_threshold: float = 2.0

    def set_validation_report(self, report: ValidationReport) -> None:
        """Set the prediction validation report."""
        self._validation_report = report

    def set_calibration_result(self, result: CalibrationResult) -> None:
        """Set the latest calibration result."""
        self._calibration_result = result

    def add_cohort_cases(self, cases: List[Dict[str, Any]]) -> None:
        """Add case records for cohort analytics."""
        self._cohort.add_cases(cases)

    def set_outlier_threshold(self, z: float) -> None:
        """Set Z-score threshold for outlier detection."""
        self._z_threshold = z

    # ── Panel builders ────────────────────────────────────────

    def _build_accuracy_panel(self) -> AccuracyPanel:
        """Build the accuracy overview panel."""
        if self._validation_report is None:
            return AccuracyPanel(
                n_cases=0, metrics=[], mae={}, rmse={}, max_error={},
                overall_grade="N/A", accuracy_profiles=[], bland_altman=[],
            )

        vr = self._validation_report
        acc_profiles = []
        for ap in vr.accuracy_profiles:
            acc_profiles.append({
                "metric": ap.metric_name,
                "thresholds_mm": ap.thresholds_mm,
                "cumulative_pct": ap.cumulative_pct,
                "auc": round(ap.auc, 4),
                "pct_within_1mm": round(ap.pct_within_1mm, 1),
                "pct_within_2mm": round(ap.pct_within_2mm, 1),
            })

        ba_results = []
        for ba in vr.bland_altman:
            ba_results.append({
                "metric": ba.metric_name,
                "bias": round(ba.bias, 4),
                "sd_diff": round(ba.sd_diff, 4),
                "upper_loa": round(ba.upper_loa, 4),
                "lower_loa": round(ba.lower_loa, 4),
                "within_loa_pct": round(ba.within_loa_pct, 1),
                "proportional_bias": ba.proportional_bias,
            })

        return AccuracyPanel(
            n_cases=vr.n_cases,
            metrics=vr.metrics_evaluated,
            mae=vr.mae,
            rmse=vr.rmse,
            max_error=vr.max_error,
            overall_grade=vr.overall_grade,
            accuracy_profiles=acc_profiles,
            bland_altman=ba_results,
        )

    def _build_calibration_panel(self) -> CalibrationPanel:
        """Build the calibration status panel."""
        if self._calibration_result is None:
            return CalibrationPanel(
                is_calibrated=False,
                last_calibration_residual=0.0,
                n_calibration_cases=0,
                parameter_count=0,
                converged=False,
                parameter_summary={},
            )

        cr = self._calibration_result
        params = {}
        if cr.parameters_after is not None:
            for name, val in cr.parameters_after.items():
                params[name] = round(float(val), 6)

        return CalibrationPanel(
            is_calibrated=True,
            last_calibration_residual=float(cr.residual_after),
            n_calibration_cases=len(cr.case_ids),
            parameter_count=len(params),
            converged=cr.converged,
            parameter_summary=params,
        )

    def _build_cohort_panel(self) -> CohortPanel:
        """Build the cohort summary panel."""
        if self._cohort.n_cases == 0:
            return CohortPanel(
                n_cases=0, procedure_counts={},
                demographics={}, outcome_distributions={},
            )

        demo = self._cohort.compute_distributions(["age", "bmi"])
        outcomes = self._cohort.compute_distributions([
            "aesthetic_score", "safety_score", "functional_score",
        ])
        procs = self._cohort.procedure_counts()

        return CohortPanel(
            n_cases=self._cohort.n_cases,
            procedure_counts=procs,
            demographics={k: v.to_dict() for k, v in demo.items()},
            outcome_distributions={k: v.to_dict() for k, v in outcomes.items()},
        )

    def _build_surgeon_panel(self) -> SurgeonPanel:
        """Build the surgeon performance panel."""
        if self._cohort.n_cases == 0:
            return SurgeonPanel(surgeons=[], n_surgeons=0, top_performer=None)

        profiles = self._cohort.surgeon_profiles()
        surgeon_dicts = [p.to_dict() for p in profiles]

        top = None
        if profiles:
            best = max(profiles, key=lambda p: p.mean_aesthetic_score)
            top = best.surgeon_id

        return SurgeonPanel(
            surgeons=surgeon_dicts,
            n_surgeons=len(profiles),
            top_performer=top,
        )

    def _build_risk_panel(self) -> RiskPanel:
        """Build the risk factor panel."""
        if self._cohort.n_cases < 10:
            return RiskPanel(risk_factors=[], n_factors=0, top_risk=None)

        factors = self._cohort.identify_risk_factors()
        factor_dicts = [f.to_dict() for f in factors]

        top = factors[0].factor_name if factors else None

        return RiskPanel(
            risk_factors=factor_dicts,
            n_factors=len(factors),
            top_risk=top,
        )

    def _build_trend_panel(self) -> TrendPanel:
        """Build the temporal trend panel."""
        metrics_to_trend = ["aesthetic_score", "safety_score", "functional_score"]
        trends: Dict[str, List[Dict[str, Any]]] = {}

        for metric in metrics_to_trend:
            trend_data = self._cohort.temporal_trends(metric)
            if trend_data:
                trends[metric] = [t.to_dict() for t in trend_data]

        return TrendPanel(metrics=trends)

    def _build_outlier_panel(self) -> OutlierPanel:
        """Build the outlier detection panel."""
        if self._validation_report is None:
            return OutlierPanel(outliers=[], n_outliers=0, z_threshold=self._z_threshold)

        vr = self._validation_report
        outliers: List[OutlierCase] = []

        for case_id, comparisons in vr.case_comparisons.items():
            for mc in comparisons:
                # Compute z-score using the metric's RMSE as spread
                metric_rmse = vr.rmse.get(mc.metric_name, 1.0)
                if metric_rmse < 1e-12:
                    continue
                z = abs(mc.error) / metric_rmse
                if z >= self._z_threshold:
                    outliers.append(OutlierCase(
                        case_id=case_id,
                        metric=mc.metric_name,
                        predicted=mc.predicted,
                        actual=mc.actual,
                        error=mc.error,
                        z_score=z,
                    ))

        outliers.sort(key=lambda o: abs(o.z_score), reverse=True)

        return OutlierPanel(
            outliers=outliers,
            n_outliers=len(outliers),
            z_threshold=self._z_threshold,
        )

    # ── Full dashboard ────────────────────────────────────────

    def build(self) -> DashboardPayload:
        """Build the complete dashboard payload."""
        return DashboardPayload(
            accuracy=self._build_accuracy_panel(),
            calibration=self._build_calibration_panel(),
            cohort=self._build_cohort_panel(),
            surgeons=self._build_surgeon_panel(),
            risks=self._build_risk_panel(),
            trends=self._build_trend_panel(),
            outliers=self._build_outlier_panel(),
        )
