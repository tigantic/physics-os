"""Cohort analytics — cross-case statistical analysis and insights.

Provides multi-case aggregation capabilities:
  - Distribution analysis (demographics, outcomes, procedures)
  - Cross-case metric aggregation (safety, aesthetic, functional)
  - Surgeon performance profiling
  - Risk factor identification via logistic regression
  - Procedure comparison (effect sizes between technique variants)
  - Temporal trend analysis (improvement over time)
  - Population subgroup analysis (age, ethnicity, sex)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import ProcedureType

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────

@dataclass
class DistributionStats:
    """Descriptive statistics for a single variable across cases."""
    variable: str
    n: int
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    iqr: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "n": self.n,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "median": round(self.median, 4),
            "q25": round(self.q25, 4),
            "q75": round(self.q75, 4),
            "iqr": round(self.iqr, 4),
            "min": round(self.min_val, 4),
            "max": round(self.max_val, 4),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
        }


@dataclass
class EffectSize:
    """Cohen's d effect size between two groups."""
    group_a: str
    group_b: str
    metric: str
    cohens_d: float
    mean_a: float
    mean_b: float
    n_a: int
    n_b: int
    interpretation: str  # "negligible", "small", "medium", "large"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_a": self.group_a,
            "group_b": self.group_b,
            "metric": self.metric,
            "cohens_d": round(self.cohens_d, 4),
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "n_a": self.n_a,
            "n_b": self.n_b,
            "interpretation": self.interpretation,
        }


@dataclass
class RiskFactor:
    """Identified risk factor from logistic regression."""
    factor_name: str
    coefficient: float
    odds_ratio: float
    relative_importance: float  # normalised 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor_name,
            "coefficient": round(self.coefficient, 4),
            "odds_ratio": round(self.odds_ratio, 4),
            "relative_importance": round(self.relative_importance, 4),
        }


@dataclass
class SurgeonProfile:
    """Performance profile for a surgeon across cases."""
    surgeon_id: str
    n_cases: int
    procedures: Dict[str, int]  # procedure → count
    mean_aesthetic_score: float
    mean_safety_score: float
    mean_functional_score: float
    complication_rate: float
    revision_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surgeon_id": self.surgeon_id,
            "n_cases": self.n_cases,
            "procedures": self.procedures,
            "mean_aesthetic_score": round(self.mean_aesthetic_score, 4),
            "mean_safety_score": round(self.mean_safety_score, 4),
            "mean_functional_score": round(self.mean_functional_score, 4),
            "complication_rate": round(self.complication_rate, 4),
            "revision_rate": round(self.revision_rate, 4),
        }


@dataclass
class TrendPoint:
    """Single point in a temporal trend."""
    period: str  # e.g. "2025-Q1"
    n_cases: int
    mean_value: float
    std_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "n_cases": self.n_cases,
            "mean": round(self.mean_value, 4),
            "std": round(self.std_value, 4),
        }


@dataclass
class SubgroupAnalysis:
    """Analysis of a metric across population subgroups."""
    metric: str
    subgroups: Dict[str, DistributionStats]  # group_name → stats
    effect_sizes: List[EffectSize]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "subgroups": {k: v.to_dict() for k, v in self.subgroups.items()},
            "effect_sizes": [e.to_dict() for e in self.effect_sizes],
        }


@dataclass
class CohortReport:
    """Complete cohort analytics report."""
    n_cases: int
    demographics: Dict[str, DistributionStats]
    outcome_distributions: Dict[str, DistributionStats]
    procedure_counts: Dict[str, int]
    risk_factors: List[RiskFactor]
    surgeon_profiles: List[SurgeonProfile]
    subgroup_analyses: List[SubgroupAnalysis]
    trends: Dict[str, List[TrendPoint]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "demographics": {k: v.to_dict() for k, v in self.demographics.items()},
            "outcome_distributions": {k: v.to_dict() for k, v in self.outcome_distributions.items()},
            "procedure_counts": self.procedure_counts,
            "risk_factors": [r.to_dict() for r in self.risk_factors],
            "surgeon_profiles": [s.to_dict() for s in self.surgeon_profiles],
            "subgroup_analyses": [s.to_dict() for s in self.subgroup_analyses],
            "trends": {k: [t.to_dict() for t in v] for k, v in self.trends.items()},
        }


# ── Statistical helpers ───────────────────────────────────────────

def _distribution_stats(name: str, values: np.ndarray) -> DistributionStats:
    """Compute descriptive statistics for an array of values."""
    if len(values) == 0:
        return DistributionStats(
            variable=name, n=0, mean=0.0, std=0.0, median=0.0,
            q25=0.0, q75=0.0, iqr=0.0, min_val=0.0, max_val=0.0,
            skewness=0.0, kurtosis=0.0,
        )

    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sorted_v = np.sort(values)
    median = float(np.median(values))
    q25 = float(np.percentile(values, 25))
    q75 = float(np.percentile(values, 75))
    iqr = q75 - q25

    # Skewness (Fisher)
    if std > 1e-12 and n > 2:
        skew = float(np.mean(((values - mean) / std) ** 3))
        skew *= n / ((n - 1) * (n - 2)) * n  # bias correction approximation
    else:
        skew = 0.0

    # Excess kurtosis
    if std > 1e-12 and n > 3:
        kurt = float(np.mean(((values - mean) / std) ** 4)) - 3.0
    else:
        kurt = 0.0

    return DistributionStats(
        variable=name, n=n, mean=mean, std=std, median=median,
        q25=q25, q75=q75, iqr=iqr,
        min_val=float(sorted_v[0]), max_val=float(sorted_v[-1]),
        skewness=skew, kurtosis=kurt,
    )


def _cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> Tuple[float, str]:
    """Compute Cohen's d effect size and interpretation."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0, "negligible"

    mean_a = float(np.mean(group_a))
    mean_b = float(np.mean(group_b))
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))

    # Pooled standard deviation
    pooled_std = math.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )

    if pooled_std < 1e-12:
        return 0.0, "negligible"

    d = (mean_a - mean_b) / pooled_std
    abs_d = abs(d)

    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return d, interp


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.01,
    n_iter: int = 500,
    l2_reg: float = 0.01,
) -> np.ndarray:
    """Simple L2-regularised logistic regression via gradient descent.

    Parameters
    ----------
    X : (N, D) feature matrix (should be standardised)
    y : (N,) binary labels (0 or 1)
    lr : learning rate
    n_iter : number of iterations
    l2_reg : L2 regularisation strength

    Returns
    -------
    weights : (D,) coefficient vector
    """
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)

    for _ in range(n_iter):
        logits = X @ w
        probs = np.array([_sigmoid(float(z)) for z in logits])
        grad = X.T @ (probs - y) / n + l2_reg * w
        w -= lr * grad

    return w


# ── CohortAnalytics engine ───────────────────────────────────────

class CohortAnalytics:
    """Cross-case cohort analytics engine.

    Ingests per-case data records and produces cohort-level insights
    including distributions, risk factors, subgroup comparisons, and
    temporal trends.

    Each case is represented as a flat dict with fields like:
      - case_id, procedure, surgeon_id, date
      - age, sex, ethnicity, bmi
      - aesthetic_score, safety_score, functional_score
      - complication (bool), revision (bool)
      - any numeric metric values

    Usage::

        engine = CohortAnalytics()
        for case_data in load_all_cases():
            engine.add_case(case_data)
        report = engine.generate_report()
    """

    def __init__(self) -> None:
        self._cases: List[Dict[str, Any]] = []

    @property
    def n_cases(self) -> int:
        return len(self._cases)

    def add_case(self, case_data: Dict[str, Any]) -> None:
        """Add a single case record to the cohort."""
        if "case_id" not in case_data:
            raise ValueError("case_data must contain 'case_id'")
        self._cases.append(dict(case_data))

    def add_cases(self, cases: Sequence[Dict[str, Any]]) -> None:
        """Add multiple case records."""
        for c in cases:
            self.add_case(c)

    def clear(self) -> None:
        """Remove all case data."""
        self._cases.clear()

    # ── Distribution analysis ─────────────────────────────────

    def compute_distributions(
        self,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, DistributionStats]:
        """Compute descriptive statistics for numeric variables."""
        if variables is None:
            variables = self._detect_numeric_fields()

        results: Dict[str, DistributionStats] = {}
        for var in variables:
            values = self._extract_numeric(var)
            if len(values) > 0:
                results[var] = _distribution_stats(var, values)

        return results

    # ── Procedure counts ──────────────────────────────────────

    def procedure_counts(self) -> Dict[str, int]:
        """Count cases per procedure type."""
        counts: Dict[str, int] = {}
        for c in self._cases:
            proc = c.get("procedure", "unknown")
            if isinstance(proc, ProcedureType):
                proc = proc.value
            counts[proc] = counts.get(proc, 0) + 1
        return counts

    # ── Risk factor analysis ──────────────────────────────────

    def identify_risk_factors(
        self,
        outcome_field: str = "complication",
        predictor_fields: Optional[List[str]] = None,
    ) -> List[RiskFactor]:
        """Identify risk factors for a binary outcome using logistic
        regression.

        Predictors are standardised before fitting. Returns factors
        sorted by absolute coefficient magnitude.
        """
        if predictor_fields is None:
            predictor_fields = self._detect_numeric_fields()
            predictor_fields = [f for f in predictor_fields if f != outcome_field]

        # Build feature matrix
        y_list = []
        X_rows = []
        for c in self._cases:
            outcome = c.get(outcome_field)
            if outcome is None:
                continue
            row = []
            valid = True
            for pf in predictor_fields:
                val = c.get(pf)
                if val is None or not isinstance(val, (int, float)):
                    valid = False
                    break
                row.append(float(val))
            if valid:
                y_list.append(1.0 if outcome else 0.0)
                X_rows.append(row)

        if len(X_rows) < 10 or not predictor_fields:
            return []

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)

        # Standardise
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds[stds < 1e-12] = 1.0
        X_std = (X - means) / stds

        # Fit
        weights = _logistic_regression(X_std, y)

        # Build risk factors
        abs_w = np.abs(weights)
        max_w = abs_w.max() if abs_w.max() > 0 else 1.0

        factors = []
        for i, pf in enumerate(predictor_fields):
            factors.append(RiskFactor(
                factor_name=pf,
                coefficient=float(weights[i]),
                odds_ratio=math.exp(float(weights[i])),
                relative_importance=float(abs_w[i] / max_w),
            ))

        factors.sort(key=lambda r: abs(r.coefficient), reverse=True)
        return factors

    # ── Surgeon profiling ─────────────────────────────────────

    def surgeon_profiles(self) -> List[SurgeonProfile]:
        """Build performance profiles for each surgeon."""
        surgeon_cases: Dict[str, List[Dict[str, Any]]] = {}
        for c in self._cases:
            sid = c.get("surgeon_id", "unknown")
            surgeon_cases.setdefault(sid, []).append(c)

        profiles = []
        for sid, cases in surgeon_cases.items():
            n = len(cases)
            procs: Dict[str, int] = {}
            aes_scores = []
            saf_scores = []
            func_scores = []
            complications = 0
            revisions = 0

            for c in cases:
                proc = c.get("procedure", "unknown")
                if isinstance(proc, ProcedureType):
                    proc = proc.value
                procs[proc] = procs.get(proc, 0) + 1

                if "aesthetic_score" in c:
                    aes_scores.append(float(c["aesthetic_score"]))
                if "safety_score" in c:
                    saf_scores.append(float(c["safety_score"]))
                if "functional_score" in c:
                    func_scores.append(float(c["functional_score"]))
                if c.get("complication"):
                    complications += 1
                if c.get("revision"):
                    revisions += 1

            profiles.append(SurgeonProfile(
                surgeon_id=sid,
                n_cases=n,
                procedures=procs,
                mean_aesthetic_score=float(np.mean(aes_scores)) if aes_scores else 0.0,
                mean_safety_score=float(np.mean(saf_scores)) if saf_scores else 0.0,
                mean_functional_score=float(np.mean(func_scores)) if func_scores else 0.0,
                complication_rate=complications / max(n, 1),
                revision_rate=revisions / max(n, 1),
            ))

        profiles.sort(key=lambda p: p.n_cases, reverse=True)
        return profiles

    # ── Subgroup analysis ─────────────────────────────────────

    def subgroup_analysis(
        self,
        metric: str,
        group_by: str,
    ) -> SubgroupAnalysis:
        """Compare a metric across subgroups defined by a categorical
        variable (e.g. sex, ethnicity, procedure).
        """
        groups: Dict[str, List[float]] = {}
        for c in self._cases:
            group = c.get(group_by)
            if group is None:
                continue
            val = c.get(metric)
            if val is None or not isinstance(val, (int, float)):
                continue
            group_key = str(group)
            groups.setdefault(group_key, []).append(float(val))

        subgroup_stats: Dict[str, DistributionStats] = {}
        for gname, vals in groups.items():
            subgroup_stats[gname] = _distribution_stats(
                f"{metric}_{gname}", np.array(vals, dtype=np.float64),
            )

        # Pairwise effect sizes
        effect_sizes: List[EffectSize] = []
        group_names = sorted(groups.keys())
        for i, ga in enumerate(group_names):
            for gb in group_names[i + 1:]:
                arr_a = np.array(groups[ga], dtype=np.float64)
                arr_b = np.array(groups[gb], dtype=np.float64)
                d, interp = _cohens_d(arr_a, arr_b)
                effect_sizes.append(EffectSize(
                    group_a=ga,
                    group_b=gb,
                    metric=metric,
                    cohens_d=d,
                    mean_a=float(np.mean(arr_a)),
                    mean_b=float(np.mean(arr_b)),
                    n_a=len(arr_a),
                    n_b=len(arr_b),
                    interpretation=interp,
                ))

        return SubgroupAnalysis(
            metric=metric,
            subgroups=subgroup_stats,
            effect_sizes=effect_sizes,
        )

    # ── Temporal trends ───────────────────────────────────────

    def temporal_trends(
        self,
        metric: str,
        period_field: str = "date",
        resolution: str = "quarter",
    ) -> List[TrendPoint]:
        """Compute temporal trend of a metric over time.

        Groups cases by period (quarter, month, year) and computes
        mean ± std of the metric within each period.
        """
        period_data: Dict[str, List[float]] = {}
        for c in self._cases:
            date_val = c.get(period_field)
            val = c.get(metric)
            if date_val is None or val is None:
                continue
            if not isinstance(val, (int, float)):
                continue

            period_key = self._date_to_period(str(date_val), resolution)
            period_data.setdefault(period_key, []).append(float(val))

        trend: List[TrendPoint] = []
        for period in sorted(period_data.keys()):
            vals = np.array(period_data[period], dtype=np.float64)
            trend.append(TrendPoint(
                period=period,
                n_cases=len(vals),
                mean_value=float(np.mean(vals)),
                std_value=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            ))

        return trend

    # ── Full report ───────────────────────────────────────────

    def generate_report(
        self,
        *,
        outcome_field: str = "complication",
        subgroup_metrics: Optional[List[Tuple[str, str]]] = None,
        trend_metrics: Optional[List[str]] = None,
    ) -> CohortReport:
        """Generate a complete cohort analytics report.

        Parameters
        ----------
        outcome_field : field name for binary outcome in risk analysis
        subgroup_metrics : list of (metric, group_by) pairs for subgroup analysis
        trend_metrics : list of metric names for temporal trend analysis
        """
        demographics_vars = ["age", "bmi"]
        outcome_vars = ["aesthetic_score", "safety_score", "functional_score"]

        demographics = self.compute_distributions(demographics_vars)
        outcomes = self.compute_distributions(outcome_vars)
        procs = self.procedure_counts()
        risk_factors = self.identify_risk_factors(outcome_field=outcome_field)
        profiles = self.surgeon_profiles()

        subgroups: List[SubgroupAnalysis] = []
        if subgroup_metrics:
            for metric, group_by in subgroup_metrics:
                subgroups.append(self.subgroup_analysis(metric, group_by))

        trends: Dict[str, List[TrendPoint]] = {}
        if trend_metrics:
            for metric in trend_metrics:
                trends[metric] = self.temporal_trends(metric)

        return CohortReport(
            n_cases=self.n_cases,
            demographics=demographics,
            outcome_distributions=outcomes,
            procedure_counts=procs,
            risk_factors=risk_factors,
            surgeon_profiles=profiles,
            subgroup_analyses=subgroups,
            trends=trends,
        )

    # ── Private helpers ───────────────────────────────────────

    def _detect_numeric_fields(self) -> List[str]:
        """Auto-detect numeric fields from case data."""
        if not self._cases:
            return []
        field_counts: Dict[str, int] = {}
        for c in self._cases:
            for k, v in c.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    field_counts[k] = field_counts.get(k, 0) + 1

        # Keep fields present in > 50% of cases
        threshold = len(self._cases) * 0.5
        return sorted(k for k, cnt in field_counts.items() if cnt >= threshold)

    def _extract_numeric(self, field_name: str) -> np.ndarray:
        """Extract a numeric field across all cases."""
        values = []
        for c in self._cases:
            val = c.get(field_name)
            if val is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
                values.append(float(val))
        return np.array(values, dtype=np.float64)

    @staticmethod
    def _date_to_period(date_str: str, resolution: str) -> str:
        """Convert a date string to a period key."""
        # Accept ISO format: YYYY-MM-DD or similar
        parts = date_str.replace("/", "-").split("-")
        if len(parts) < 2:
            return date_str

        year = parts[0]
        month = int(parts[1]) if len(parts) >= 2 else 1

        if resolution == "year":
            return year
        elif resolution == "month":
            return f"{year}-{month:02d}"
        else:  # quarter
            q = (month - 1) // 3 + 1
            return f"{year}-Q{q}"
