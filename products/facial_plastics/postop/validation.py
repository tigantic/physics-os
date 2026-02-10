"""Prediction validation — track prediction accuracy across cases.

Provides:
  - Per-metric prediction vs actual comparison
  - Bland–Altman analysis (bias + limits of agreement)
  - Pearson / Spearman correlation
  - Mean absolute error, RMSE per metric
  - Cumulative accuracy profiles
  - Multi-case aggregate reports
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricComparison:
    """Comparison of one metric between predicted and actual values."""
    metric_name: str
    unit: str
    predicted: float
    actual: float

    @property
    def error(self) -> float:
        return self.predicted - self.actual

    @property
    def abs_error(self) -> float:
        return abs(self.error)

    @property
    def pct_error(self) -> float:
        if abs(self.actual) < 1e-12:
            return 0.0 if abs(self.predicted) < 1e-12 else float("inf")
        return abs(self.error / self.actual) * 100.0


@dataclass
class BlandAltmanResult:
    """Bland–Altman analysis result for one metric across cases."""
    metric_name: str
    bias: float                  # Mean difference (predicted - actual)
    sd_diff: float               # SD of differences
    upper_loa: float             # Upper limit of agreement (bias + 1.96*SD)
    lower_loa: float             # Lower limit of agreement (bias - 1.96*SD)
    n_cases: int
    within_loa_pct: float        # % of cases within limits of agreement
    proportional_bias: bool      # True if bias correlates with mean


@dataclass
class CorrelationResult:
    """Correlation analysis for one metric."""
    metric_name: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    n_cases: int


@dataclass
class AccuracyProfile:
    """Cumulative accuracy profile for one metric."""
    metric_name: str
    thresholds_mm: List[float]      # Error thresholds
    cumulative_pct: List[float]     # % of cases within each threshold
    auc: float                      # Area under cumulative accuracy curve

    @property
    def pct_within_1mm(self) -> float:
        for t, p in zip(self.thresholds_mm, self.cumulative_pct):
            if t >= 1.0:
                return p
        return 0.0

    @property
    def pct_within_2mm(self) -> float:
        for t, p in zip(self.thresholds_mm, self.cumulative_pct):
            if t >= 2.0:
                return p
        return 0.0


@dataclass
class ValidationReport:
    """Aggregate validation report across cases."""
    n_cases: int
    metrics_evaluated: List[str]

    # Per-metric statistics
    mae: Dict[str, float] = field(default_factory=dict)    # Mean absolute error
    rmse: Dict[str, float] = field(default_factory=dict)   # Root mean squared error
    max_error: Dict[str, float] = field(default_factory=dict)

    # Detailed analyses
    bland_altman: List[BlandAltmanResult] = field(default_factory=list)
    correlations: List[CorrelationResult] = field(default_factory=list)
    accuracy_profiles: List[AccuracyProfile] = field(default_factory=list)

    # Comparisons per case
    case_comparisons: Dict[str, List[MetricComparison]] = field(default_factory=dict)

    # Overall grade
    overall_grade: str = "N/A"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "metrics_evaluated": self.metrics_evaluated,
            "mae": self.mae,
            "rmse": self.rmse,
            "max_error": self.max_error,
            "overall_grade": self.overall_grade,
            "bland_altman": [
                {
                    "metric": ba.metric_name,
                    "bias": ba.bias,
                    "sd_diff": ba.sd_diff,
                    "upper_loa": ba.upper_loa,
                    "lower_loa": ba.lower_loa,
                    "n_cases": ba.n_cases,
                    "within_loa_pct": ba.within_loa_pct,
                    "proportional_bias": ba.proportional_bias,
                }
                for ba in self.bland_altman
            ],
            "correlations": [
                {
                    "metric": c.metric_name,
                    "pearson_r": c.pearson_r,
                    "pearson_p": c.pearson_p,
                    "spearman_rho": c.spearman_rho,
                    "spearman_p": c.spearman_p,
                    "n_cases": c.n_cases,
                }
                for c in self.correlations
            ],
            "accuracy_profiles": [
                {
                    "metric": ap.metric_name,
                    "pct_within_1mm": ap.pct_within_1mm,
                    "pct_within_2mm": ap.pct_within_2mm,
                    "auc": ap.auc,
                }
                for ap in self.accuracy_profiles
            ],
        }


class PredictionValidator:
    """Validate prediction accuracy across cases.

    Usage:
        validator = PredictionValidator()
        validator.add_case("case_001", predicted_metrics, actual_metrics)
        validator.add_case("case_002", predicted_metrics, actual_metrics)
        report = validator.generate_report()
    """

    def __init__(self) -> None:
        self._cases: Dict[str, List[MetricComparison]] = {}

    def add_case(
        self,
        case_id: str,
        predicted: Dict[str, Tuple[float, str]],
        actual: Dict[str, Tuple[float, str]],
    ) -> None:
        """Add predicted vs actual metrics for a case.

        Args:
            case_id: Unique case identifier.
            predicted: {metric_name: (value, unit)} predicted values.
            actual: {metric_name: (value, unit)} actual measurements.
        """
        comparisons: List[MetricComparison] = []
        common_metrics = set(predicted.keys()) & set(actual.keys())

        for metric in sorted(common_metrics):
            pred_val, pred_unit = predicted[metric]
            act_val, act_unit = actual[metric]

            if pred_unit != act_unit:
                logger.warning(
                    "Unit mismatch for %s in case %s: %s vs %s",
                    metric, case_id, pred_unit, act_unit,
                )

            comparisons.append(MetricComparison(
                metric_name=metric,
                unit=pred_unit,
                predicted=pred_val,
                actual=act_val,
            ))

        self._cases[case_id] = comparisons
        logger.debug(
            "Added case %s with %d metric comparisons",
            case_id, len(comparisons),
        )

    @property
    def n_cases(self) -> int:
        return len(self._cases)

    def generate_report(self) -> ValidationReport:
        """Generate comprehensive validation report."""
        if not self._cases:
            return ValidationReport(n_cases=0, metrics_evaluated=[])

        # Collect all metric names
        all_metrics: set = set()
        for comparisons in self._cases.values():
            for c in comparisons:
                all_metrics.add(c.metric_name)

        metric_names = sorted(all_metrics)
        report = ValidationReport(
            n_cases=len(self._cases),
            metrics_evaluated=metric_names,
            case_comparisons=dict(self._cases),
        )

        # Per-metric aggregation
        for metric in metric_names:
            errors = self._collect_errors(metric)
            if len(errors) == 0:
                continue

            abs_errors = np.abs(errors)
            report.mae[metric] = float(np.mean(abs_errors))
            report.rmse[metric] = float(np.sqrt(np.mean(errors ** 2)))
            report.max_error[metric] = float(np.max(abs_errors))

        # Bland-Altman
        for metric in metric_names:
            ba = self._bland_altman(metric)
            if ba is not None:
                report.bland_altman.append(ba)

        # Correlations
        for metric in metric_names:
            corr = self._correlation(metric)
            if corr is not None:
                report.correlations.append(corr)

        # Accuracy profiles
        for metric in metric_names:
            profile = self._accuracy_profile(metric)
            if profile is not None:
                report.accuracy_profiles.append(profile)

        # Overall grade
        report.overall_grade = self._compute_grade(report)

        return report

    def _collect_errors(self, metric: str) -> np.ndarray:
        """Collect signed error for a specific metric across all cases."""
        errors: List[float] = []
        for comparisons in self._cases.values():
            for c in comparisons:
                if c.metric_name == metric:
                    errors.append(c.error)
        return np.array(errors, dtype=np.float64)

    def _collect_predicted_actual(
        self, metric: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect paired predicted & actual values for a metric."""
        pred: List[float] = []
        actual: List[float] = []
        for comparisons in self._cases.values():
            for c in comparisons:
                if c.metric_name == metric:
                    pred.append(c.predicted)
                    actual.append(c.actual)
        return np.array(pred), np.array(actual)

    def _bland_altman(self, metric: str) -> Optional[BlandAltmanResult]:
        """Bland–Altman analysis for one metric."""
        errors = self._collect_errors(metric)
        pred, actual = self._collect_predicted_actual(metric)

        if len(errors) < 3:
            return None

        bias = float(np.mean(errors))
        sd_diff = float(np.std(errors, ddof=1))
        upper_loa = bias + 1.96 * sd_diff
        lower_loa = bias - 1.96 * sd_diff

        within_loa = np.sum((errors >= lower_loa) & (errors <= upper_loa))
        within_loa_pct = float(within_loa / len(errors) * 100.0)

        # Check proportional bias: correlation of diff vs mean
        means = (pred + actual) / 2.0
        proportional_bias = False
        if np.std(means) > 1e-12 and np.std(errors) > 1e-12:
            r = np.corrcoef(means, errors)[0, 1]
            n = len(errors)
            if n > 3:
                t_stat = r * math.sqrt((n - 2) / max(1 - r ** 2, 1e-30))
                # Approximate two-tailed p-value via t-distribution
                proportional_bias = abs(t_stat) > 2.0  # ~p < 0.05

        return BlandAltmanResult(
            metric_name=metric,
            bias=bias,
            sd_diff=sd_diff,
            upper_loa=upper_loa,
            lower_loa=lower_loa,
            n_cases=len(errors),
            within_loa_pct=within_loa_pct,
            proportional_bias=proportional_bias,
        )

    def _correlation(self, metric: str) -> Optional[CorrelationResult]:
        """Pearson and Spearman correlation for one metric."""
        pred, actual = self._collect_predicted_actual(metric)

        if len(pred) < 3:
            return None

        # Pearson
        pearson_r, pearson_p = self._pearson(pred, actual)

        # Spearman (rank-based)
        spearman_rho, spearman_p = self._spearman(pred, actual)

        return CorrelationResult(
            metric_name=metric,
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            spearman_rho=spearman_rho,
            spearman_p=spearman_p,
            n_cases=len(pred),
        )

    @staticmethod
    def _pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Pearson correlation coefficient with approximate p-value."""
        n = len(x)
        if n < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0, 1.0

        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) >= 1.0:
            return r, 0.0

        t_stat = r * math.sqrt((n - 2) / max(1 - r ** 2, 1e-30))
        # Approximate p-value (two-tailed) from t-distribution using
        # incomplete beta function relationship; for simplicity use
        # the normal approximation valid for n > 30
        if n > 30:
            p = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
        else:
            # Use betainc approximation for small n
            p = _t_distribution_p(t_stat, n - 2)

        return r, p

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Spearman rank correlation coefficient with approximate p-value."""
        n = len(x)
        if n < 3:
            return 0.0, 1.0

        rank_x = _rankdata(x)
        rank_y = _rankdata(y)

        d = rank_x - rank_y
        d2_sum = float(np.sum(d ** 2))

        rho = 1.0 - 6.0 * d2_sum / (n * (n ** 2 - 1))
        rho = np.clip(rho, -1.0, 1.0)

        # Approximate p using t-distribution
        if abs(rho) >= 1.0:
            return float(rho), 0.0

        t_stat = rho * math.sqrt((n - 2) / max(1 - rho ** 2, 1e-30))
        p = _t_distribution_p(t_stat, n - 2)

        return float(rho), p

    def _accuracy_profile(self, metric: str) -> Optional[AccuracyProfile]:
        """Cumulative accuracy profile for one metric."""
        errors = self._collect_errors(metric)
        if len(errors) < 2:
            return None

        abs_errors = np.abs(errors)
        max_err = float(np.max(abs_errors))

        # Create threshold grid extending to standard clinical thresholds
        clinical_max = max(max_err * 1.1, 5.0)  # Always include up to 5 mm
        n_thresholds = 100
        thresholds = np.linspace(0, clinical_max, n_thresholds).tolist()
        cumulative = []

        for t in thresholds:
            pct = float(np.sum(abs_errors <= t) / len(abs_errors) * 100.0)
            cumulative.append(pct)

        # AUC (trapezoidal) normalized by max threshold
        if len(thresholds) > 1:
            auc = float(np.trapz(cumulative, thresholds)) / max(
                thresholds[-1], 1e-12
            )
        else:
            auc = 0.0

        return AccuracyProfile(
            metric_name=metric,
            thresholds_mm=thresholds,
            cumulative_pct=cumulative,
            auc=auc,
        )

    @staticmethod
    def _compute_grade(report: ValidationReport) -> str:
        """Assign an overall validation grade.

        Grading criteria (for distance-based metrics):
          A : Mean RMSE < 1.0 mm
          B : Mean RMSE < 2.0 mm
          C : Mean RMSE < 3.0 mm
          D : Mean RMSE < 5.0 mm
          F : Mean RMSE >= 5.0 mm
        """
        if not report.rmse:
            return "N/A"

        mean_rmse = float(np.mean(list(report.rmse.values())))

        if mean_rmse < 1.0:
            return "A"
        elif mean_rmse < 2.0:
            return "B"
        elif mean_rmse < 3.0:
            return "C"
        elif mean_rmse < 5.0:
            return "D"
        else:
            return "F"

    def save_report(self, report: ValidationReport, path: Path) -> None:
        """Persist validation report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    @staticmethod
    def load_report(path: Path) -> Dict[str, Any]:
        """Load a previously saved validation report."""
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
            return data


# ── Statistical helper functions ────────────────────────────────────────────


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun 26.2.17."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = (
        1.330274429
        - 1.821255978 * t
        + 1.781477937 * t ** 2
        - 0.356563782 * t ** 3
        + 0.319381530 * t ** 4
    )
    # Note: coefficients are applied in reverse order for Horner form
    val = d * math.exp(-x * x / 2.0) * t * poly
    if x >= 0:
        return 1.0 - val
    else:
        return val


def _t_distribution_p(t_stat: float, df: int) -> float:
    """Approximate two-tailed p-value from t-distribution.

    Uses the relationship between t and normal distributions for
    df > 4, and a cruder fallback for small df.
    """
    if df <= 0:
        return 1.0

    # For large df, t ≈ normal
    if df > 30:
        return 2.0 * (1.0 - _normal_cdf(abs(t_stat)))

    # For moderate df, use Abramowitz & Stegun 26.7.5 approximation
    x = abs(t_stat)
    g = math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0)
    a = math.exp(g) / math.sqrt(df * math.pi)
    val = a * (1.0 + x ** 2 / df) ** (-(df + 1) / 2.0)

    # Integrate tail using Simpson's rule on a finite grid
    n_steps = 200
    upper = max(x, 10.0)
    h = (upper - x) / n_steps
    integral = val

    for i in range(1, n_steps):
        ti = x + i * h
        yi = a * (1.0 + ti ** 2 / df) ** (-(df + 1) / 2.0)
        if i % 2 == 0:
            integral += 2.0 * yi
        else:
            integral += 4.0 * yi

    t_end = x + n_steps * h
    y_end = a * (1.0 + t_end ** 2 / df) ** (-(df + 1) / 2.0)
    integral += y_end
    integral *= h / 3.0

    # Two-tailed
    p = 2.0 * integral
    return float(min(max(p, 0.0), 1.0))

def _rankdata(x: np.ndarray) -> np.ndarray:
    """Assign ranks to data, handling ties with average ranks."""
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j < n - 1 and x[order[j + 1]] == x[order[j]]:
            j += 1
        # Average rank for tied values
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    return ranks
