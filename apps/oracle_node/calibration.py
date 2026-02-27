#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║              C A L I B R A T I O N   F R A M E W O R K   —   O R A C L E   N O D E      ║
║                                                                                          ║
║                    Statistical Threshold Derivation & Validation                         ║
║                                                                                          ║
║     Principle: Thresholds are DERIVED from data, not assumed.                           ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

This module provides:
1. Ground truth ingestion (labeled normal vs. anomaly samples)
2. Statistical threshold derivation per stage
3. ROC curve analysis for optimal cutoffs
4. K-fold cross-validation
5. Calibration profile storage and versioning

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import warnings

import numpy as np
from scipy import stats
from scipy import ndimage
import torch

# Type aliases
ArrayLike = np.ndarray


class Label(Enum):
    """Ground truth labels for calibration samples."""
    NORMAL = "normal"
    ANOMALY_MILD = "anomaly_mild"
    ANOMALY_MODERATE = "anomaly_moderate"
    ANOMALY_SEVERE = "anomaly_severe"
    
    @property
    def is_anomaly(self) -> bool:
        return self != Label.NORMAL
    
    @property
    def severity(self) -> int:
        """0 = normal, 1 = mild, 2 = moderate, 3 = severe"""
        mapping = {
            Label.NORMAL: 0,
            Label.ANOMALY_MILD: 1,
            Label.ANOMALY_MODERATE: 2,
            Label.ANOMALY_SEVERE: 3,
        }
        return mapping[self]


@dataclass
class CalibrationSample:
    """A single labeled sample for calibration."""
    distribution_a: np.ndarray
    distribution_b: np.ndarray
    label: Label
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.distribution_a = np.asarray(self.distribution_a, dtype=np.float64)
        self.distribution_b = np.asarray(self.distribution_b, dtype=np.float64)


@dataclass
class StageMetrics:
    """Computed metrics for a single stage on a single sample."""
    stage_name: str
    primary_metric: float
    all_metrics: Dict[str, float]
    computation_time_ms: float


@dataclass 
class ThresholdConfig:
    """Threshold configuration for a single stage."""
    stage_name: str
    thresholds: Dict[str, float]  # e.g., {"mild": 0.05, "moderate": 0.2, "severe": 0.5}
    derivation_method: str  # e.g., "roc_optimal", "percentile_95", "bayesian"
    confidence_interval: Tuple[float, float]
    sample_size: int
    validation_auc: float
    validation_f1: float


@dataclass
class CalibrationProfile:
    """Complete calibration profile for a domain."""
    domain: str
    version: str
    created_at: str
    stage_thresholds: Dict[str, ThresholdConfig]
    validation_metrics: Dict[str, float]
    sample_count: int
    cross_validation_folds: int
    sha256: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "domain": self.domain,
            "version": self.version,
            "created_at": self.created_at,
            "stage_thresholds": {},
            "validation_metrics": self.validation_metrics,
            "sample_count": self.sample_count,
            "cross_validation_folds": self.cross_validation_folds,
        }
        for stage_name, config in self.stage_thresholds.items():
            result["stage_thresholds"][stage_name] = {
                "thresholds": config.thresholds,
                "derivation_method": config.derivation_method,
                "confidence_interval": list(config.confidence_interval),
                "sample_size": config.sample_size,
                "validation_auc": config.validation_auc,
                "validation_f1": config.validation_f1,
            }
        content = json.dumps(result, sort_keys=True)
        result["sha256"] = hashlib.sha256(content.encode()).hexdigest()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationProfile":
        stage_thresholds = {}
        for stage_name, config in data.get("stage_thresholds", {}).items():
            stage_thresholds[stage_name] = ThresholdConfig(
                stage_name=stage_name,
                thresholds=config["thresholds"],
                derivation_method=config["derivation_method"],
                confidence_interval=tuple(config["confidence_interval"]),
                sample_size=config["sample_size"],
                validation_auc=config["validation_auc"],
                validation_f1=config["validation_f1"],
            )
        return cls(
            domain=data["domain"],
            version=data["version"],
            created_at=data["created_at"],
            stage_thresholds=stage_thresholds,
            validation_metrics=data["validation_metrics"],
            sample_count=data["sample_count"],
            cross_validation_folds=data["cross_validation_folds"],
            sha256=data.get("sha256", ""),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC EXTRACTORS — Stage-specific metric computation
# ═══════════════════════════════════════════════════════════════════════════════

class MetricExtractor:
    """
    Extract metrics from distribution pairs for each pipeline stage.
    
    This mirrors the server.py stages but returns raw metrics only.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._imports_done = False
        
    def _ensure_imports(self):
        """Lazy import to avoid circular dependencies."""
        if self._imports_done:
            return
        
        # Import all required modules
        from tensornet.genesis.ot import QTTDistribution, wasserstein_distance
        from tensornet.genesis.sgw import QTTLaplacian, QTTSignal, QTTGraphWavelet
        from tensornet.genesis.rkhs import RBFKernel, maximum_mean_discrepancy
        from tensornet.genesis.topology.qtt_native import qtt_persistence_grid_1d
        from tensornet.genesis.ga import CliffordAlgebra, Multivector
        
        self._QTTDistribution = QTTDistribution
        self._wasserstein_distance = wasserstein_distance
        self._QTTLaplacian = QTTLaplacian
        self._QTTSignal = QTTSignal
        self._QTTGraphWavelet = QTTGraphWavelet
        self._RBFKernel = RBFKernel
        self._maximum_mean_discrepancy = maximum_mean_discrepancy
        self._qtt_persistence_grid_1d = qtt_persistence_grid_1d
        self._CliffordAlgebra = CliffordAlgebra
        self._Multivector = Multivector
        
        self._imports_done = True
    
    def _to_qtt_distribution(self, arr: np.ndarray) -> Any:
        """Convert numpy array to QTTDistribution."""
        self._ensure_imports()
        
        # Normalize to probability distribution
        arr = arr - arr.min()
        if arr.sum() > 0:
            arr = arr / arr.sum()
        else:
            arr = np.ones_like(arr) / len(arr)
        
        # Pad to power of 2
        n = len(arr)
        n_bits = max(4, int(np.ceil(np.log2(n))))
        target_size = 2 ** n_bits
        
        if n < target_size:
            padded = np.zeros(target_size)
            padded[:n] = arr
            arr = padded / padded.sum() if padded.sum() > 0 else padded
        elif n > target_size:
            arr = np.interp(np.linspace(0, n-1, target_size), np.arange(n), arr)
            arr = arr / arr.sum() if arr.sum() > 0 else arr
        
        tensor = torch.tensor(arr, dtype=torch.float64, device=torch.device('cpu'))
        return self._QTTDistribution.from_dense(tensor, grid_bounds=(0.0, 1.0))
    
    def extract_stage_1_ot(self, dist_a: np.ndarray, dist_b: np.ndarray) -> StageMetrics:
        """Extract optimal transport metrics."""
        self._ensure_imports()
        start = time.perf_counter()
        
        qtt_a = self._to_qtt_distribution(dist_a)
        qtt_b = self._to_qtt_distribution(dist_b)
        
        W1 = self._wasserstein_distance(qtt_a, qtt_b, p=1, method='quantile')
        W2 = self._wasserstein_distance(qtt_a, qtt_b, p=2, method='quantile')
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return StageMetrics(
            stage_name="ot",
            primary_metric=float(W2),
            all_metrics={"wasserstein_1": float(W1), "wasserstein_2": float(W2)},
            computation_time_ms=elapsed,
        )
    
    def extract_stage_2_sgw(self, dist_a: np.ndarray, dist_b: np.ndarray) -> StageMetrics:
        """Extract spectral graph wavelet metrics."""
        self._ensure_imports()
        start = time.perf_counter()
        
        qtt_a = self._to_qtt_distribution(dist_a)
        qtt_b = self._to_qtt_distribution(dist_b)
        
        n = qtt_a.grid_size
        L = self._QTTLaplacian.grid_1d(n)
        
        # Create QTTSignals directly from cores — NO to_dense()
        signal_a = self._QTTSignal.from_cores(qtt_a.cores)
        signal_b = self._QTTSignal.from_cores(qtt_b.cores)
        # Native subtraction — NO to_dense()
        diff_signal = signal_b.sub(signal_a)
        
        scales = [0.5, 1.0, 2.0, 4.0, 8.0]
        wavelet = self._QTTGraphWavelet.create(L, scales=scales, kernel='mexican_hat')
        coeffs = wavelet.transform(diff_signal)
        energies = coeffs.energy_per_scale()
        
        max_energy_idx = int(np.argmax(energies))
        dominant_scale = scales[max_energy_idx]
        total_energy = float(np.sum(energies))
        max_energy = float(np.max(energies))
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return StageMetrics(
            stage_name="sgw",
            primary_metric=dominant_scale,
            all_metrics={
                "dominant_scale": dominant_scale,
                "total_energy": total_energy,
                "max_energy": max_energy,
                "energy_ratio": max_energy / (total_energy + 1e-10),
                **{f"scale_{s}": float(e) for s, e in zip(scales, energies)},
            },
            computation_time_ms=elapsed,
        )
    
    def extract_stage_3_rkhs(self, dist_a: np.ndarray, dist_b: np.ndarray) -> StageMetrics:
        """
        Extract RKHS anomaly detection metrics using QTT sampling.
        
        Instead of materializing the full dense vector (O(N) memory),
        we sample a fixed number of points via qtt_evaluate_at_indices.
        This gives O(k * d * r^2) complexity where k = sample count.
        """
        self._ensure_imports()
        start = time.perf_counter()
        
        qtt_a = self._to_qtt_distribution(dist_a)
        qtt_b = self._to_qtt_distribution(dist_b)
        
        n = qtt_a.grid_size
        
        # Sample a fixed number of points (not full dense)
        # This is O(k * d * r^2) instead of O(N)
        n_samples = min(1024, n)  # At most 1024 points
        sample_indices = torch.linspace(0, n - 1, n_samples, dtype=torch.long)
        
        # Create signals from cores for native evaluation
        signal_a = self._QTTSignal.from_cores(qtt_a.cores)
        signal_b = self._QTTSignal.from_cores(qtt_b.cores)
        
        # Sample values at specific indices
        from tensornet.genesis.core import qtt_evaluate_at_indices
        values_a = qtt_evaluate_at_indices(signal_a.cores, sample_indices)
        values_b = qtt_evaluate_at_indices(signal_b.cores, sample_indices)
        
        # Create 2D point clouds: (x_position, pdf_value)
        x = torch.linspace(0, 1, n_samples).unsqueeze(1)
        points_a = torch.cat([x, values_a.unsqueeze(1)], dim=1)
        points_b = torch.cat([x, values_b.unsqueeze(1)], dim=1)
        
        length_scales = [0.1, 0.5, 1.0, 2.0]
        mmd_scores = {}
        
        for ls in length_scales:
            kernel = self._RBFKernel(length_scale=ls)
            mmd = self._maximum_mean_discrepancy(points_a, points_b, kernel)
            mmd_scores[f"mmd_ls_{ls}"] = float(mmd)
        
        avg_mmd = float(np.mean(list(mmd_scores.values())))
        max_mmd = float(np.max(list(mmd_scores.values())))
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return StageMetrics(
            stage_name="rkhs",
            primary_metric=max_mmd,
            all_metrics={"average_mmd": avg_mmd, "max_mmd": max_mmd, **mmd_scores},
            computation_time_ms=elapsed,
        )
    
    def extract_stage_4_ph(self, dist_a: np.ndarray, dist_b: np.ndarray) -> StageMetrics:
        """
        Extract persistent homology metrics using QTT-native operations.
        
        Instead of materializing full dense difference, we:
        1. Use QTT subtraction to get diff cores
        2. Sample the difference signal for sign analysis
        3. Use native PH on the QTT structure
        """
        self._ensure_imports()
        start = time.perf_counter()
        
        qtt_a = self._to_qtt_distribution(dist_a)
        qtt_b = self._to_qtt_distribution(dist_b)
        
        # Native QTT subtraction — NO to_dense()
        signal_a = self._QTTSignal.from_cores(qtt_a.cores)
        signal_b = self._QTTSignal.from_cores(qtt_b.cores)
        diff_signal = signal_b.sub(signal_a)
        
        # Sample difference at sparse points for component/hole analysis
        from tensornet.genesis.core import qtt_evaluate_at_indices
        n = qtt_a.grid_size
        n_samples = min(1024, n)
        sample_indices = torch.linspace(0, n - 1, n_samples, dtype=torch.long)
        diff_samples = qtt_evaluate_at_indices(diff_signal.cores, sample_indices).cpu().numpy()
        
        # Connected components analysis on sampled difference
        threshold = np.std(diff_samples) * 2
        significant_regions = np.abs(diff_samples) > threshold
        labeled, n_components = ndimage.label(significant_regions.astype(int))
        
        # Hole detection (sign changes) on samples
        sign_changes = np.diff(np.sign(diff_samples))
        n_holes = int(np.sum(np.abs(sign_changes) == 2))
        
        # Persistence-based metrics from native QTT PH
        n_bits = int(np.log2(qtt_a.grid_size))
        ph_result = self._qtt_persistence_grid_1d(n_bits)
        betti = ph_result.betti_numbers
        betti_0 = betti[0] if len(betti) > 0 else 0
        betti_1 = betti[1] if len(betti) > 1 else 0
        
        # Topological complexity from sampled analysis
        complexity = float(n_components + n_holes)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return StageMetrics(
            stage_name="ph",
            primary_metric=complexity,
            all_metrics={
                "connected_components": int(n_components),
                "topological_holes": n_holes,
                "complexity": complexity,
                "betti_0": betti_0,
                "betti_1": betti_1,
            },
            computation_time_ms=elapsed,
        )
    
    def extract_stage_5_ga(self, dist_a: np.ndarray, dist_b: np.ndarray) -> StageMetrics:
        """
        Extract geometric algebra metrics using QTT-native operations.
        
        Instead of dense materialization, we:
        1. Use QTT subtraction for diff
        2. Sample at strategic points for gradient/center analysis
        3. Use native QTT norm for magnitude
        """
        self._ensure_imports()
        start = time.perf_counter()
        
        qtt_a = self._to_qtt_distribution(dist_a)
        qtt_b = self._to_qtt_distribution(dist_b)
        
        # Native QTT subtraction — NO to_dense()
        signal_a = self._QTTSignal.from_cores(qtt_a.cores)
        signal_b = self._QTTSignal.from_cores(qtt_b.cores)
        diff_signal = signal_b.sub(signal_a)
        
        # Sample difference at sparse points
        from tensornet.genesis.core import qtt_evaluate_at_indices
        n = qtt_a.grid_size
        n_samples = min(1024, n)
        sample_indices = torch.linspace(0, n - 1, n_samples, dtype=torch.long)
        diff_samples = qtt_evaluate_at_indices(diff_signal.cores, sample_indices).cpu().numpy()
        
        x = np.linspace(0, 1, n_samples)
        
        # Center of mass (on samples)
        if np.sum(np.abs(diff_samples)) > 1e-10:
            center = float(np.sum(x * np.abs(diff_samples)) / np.sum(np.abs(diff_samples)))
        else:
            center = 0.5
        
        # Gradient analysis (on samples)
        gradient = np.gradient(diff_samples)
        avg_gradient = float(np.mean(gradient))
        max_gradient = float(np.max(np.abs(gradient)))
        
        # Magnitude via native QTT norm — NO to_dense()
        magnitude = diff_signal.norm()
        
        # Spread (on samples)
        significant_idx = np.where(np.abs(diff_samples) > np.std(diff_samples))[0]
        spread = float(np.std(significant_idx) / n_samples) if len(significant_idx) > 0 else 0.0
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return StageMetrics(
            stage_name="ga",
            primary_metric=abs(avg_gradient),
            all_metrics={
                "center_of_change": center,
                "average_gradient": avg_gradient,
                "max_gradient": max_gradient,
                "magnitude": magnitude,
                "spread": spread,
            },
            computation_time_ms=elapsed,
        )
    
    def extract_all(self, dist_a: np.ndarray, dist_b: np.ndarray) -> Dict[str, StageMetrics]:
        """Extract metrics from all stages."""
        return {
            "ot": self.extract_stage_1_ot(dist_a, dist_b),
            "sgw": self.extract_stage_2_sgw(dist_a, dist_b),
            "rkhs": self.extract_stage_3_rkhs(dist_a, dist_b),
            "ph": self.extract_stage_4_ph(dist_a, dist_b),
            "ga": self.extract_stage_5_ga(dist_a, dist_b),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD DERIVATION — Statistical methods for cutoff selection
# ═══════════════════════════════════════════════════════════════════════════════

class ThresholdDerivation:
    """
    Statistical methods for deriving optimal classification thresholds.
    """
    
    @staticmethod
    def roc_optimal(
        normal_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        weights: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Find optimal threshold using ROC curve analysis.
        
        Uses Youden's J statistic: J = sensitivity + specificity - 1
        
        Args:
            normal_scores: Metric values for normal samples
            anomaly_scores: Metric values for anomaly samples
            weights: (false_positive_weight, false_negative_weight)
        
        Returns:
            (optimal_threshold, auc, best_j_statistic)
        """
        if weights is None:
            weights = (1.0, 1.0)
        
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.concatenate([
            np.zeros(len(normal_scores)),
            np.ones(len(anomaly_scores)),
        ])
        
        # Sort by score
        sorted_idx = np.argsort(all_scores)
        sorted_scores = all_scores[sorted_idx]
        sorted_labels = all_labels[sorted_idx]
        
        # Compute TPR and FPR at each threshold
        n_pos = len(anomaly_scores)
        n_neg = len(normal_scores)
        
        if n_pos == 0 or n_neg == 0:
            return float(np.median(all_scores)), 0.5, 0.0
        
        thresholds = []
        tprs = []
        fprs = []
        
        for i, threshold in enumerate(sorted_scores):
            # Predictions at this threshold
            predictions = all_scores >= threshold
            
            tp = np.sum((predictions == 1) & (all_labels == 1))
            fp = np.sum((predictions == 1) & (all_labels == 0))
            fn = np.sum((predictions == 0) & (all_labels == 1))
            tn = np.sum((predictions == 0) & (all_labels == 0))
            
            tpr = tp / n_pos if n_pos > 0 else 0
            fpr = fp / n_neg if n_neg > 0 else 0
            
            thresholds.append(threshold)
            tprs.append(tpr)
            fprs.append(fpr)
        
        tprs = np.array(tprs)
        fprs = np.array(fprs)
        thresholds = np.array(thresholds)
        
        # AUC via trapezoidal rule
        sorted_fpr_idx = np.argsort(fprs)
        auc = float(np.trapz(tprs[sorted_fpr_idx], fprs[sorted_fpr_idx]))
        
        # Youden's J statistic with weights
        sensitivities = tprs
        specificities = 1 - fprs
        j_stats = (weights[1] * sensitivities + weights[0] * specificities) / sum(weights) - 0.5
        
        best_idx = np.argmax(j_stats)
        optimal_threshold = float(thresholds[best_idx])
        best_j = float(j_stats[best_idx])
        
        return optimal_threshold, auc, best_j
    
    @staticmethod
    def percentile_based(
        normal_scores: np.ndarray,
        percentile: float = 95.0,
    ) -> Tuple[float, float, float]:
        """
        Threshold at percentile of normal distribution.
        
        Args:
            normal_scores: Metric values for normal samples
            percentile: Percentile to use (e.g., 95 = 5% false positive rate)
        
        Returns:
            (threshold, lower_ci, upper_ci)
        """
        threshold = float(np.percentile(normal_scores, percentile))
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_thresholds = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(normal_scores, size=len(normal_scores), replace=True)
            bootstrap_thresholds.append(np.percentile(sample, percentile))
        
        lower_ci = float(np.percentile(bootstrap_thresholds, 2.5))
        upper_ci = float(np.percentile(bootstrap_thresholds, 97.5))
        
        return threshold, lower_ci, upper_ci
    
    @staticmethod
    def bayesian_decision(
        normal_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        prior_anomaly: float = 0.1,
        cost_fp: float = 1.0,
        cost_fn: float = 10.0,
    ) -> Tuple[float, float]:
        """
        Bayesian decision theory threshold.
        
        Minimizes expected cost: E[cost] = P(anomaly|normal) * cost_fp + P(normal|anomaly) * cost_fn
        
        Args:
            normal_scores: Metric values for normal samples
            anomaly_scores: Metric values for anomaly samples
            prior_anomaly: Prior probability of anomaly
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative
        
        Returns:
            (optimal_threshold, expected_cost)
        """
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        
        # Fit kernel density estimates
        if len(normal_scores) < 2 or len(anomaly_scores) < 2:
            return float(np.median(all_scores)), float('inf')
        
        try:
            kde_normal = stats.gaussian_kde(normal_scores)
            kde_anomaly = stats.gaussian_kde(anomaly_scores)
        except Exception:
            return float(np.median(all_scores)), float('inf')
        
        # Search for optimal threshold
        test_thresholds = np.linspace(min_score, max_score, 1000)
        
        best_threshold = test_thresholds[0]
        best_cost = float('inf')
        
        for t in test_thresholds:
            # P(score > t | normal) = false positive rate
            fp_rate = 1 - stats.norm.cdf(t, loc=np.mean(normal_scores), scale=np.std(normal_scores) + 1e-10)
            
            # P(score < t | anomaly) = false negative rate
            fn_rate = stats.norm.cdf(t, loc=np.mean(anomaly_scores), scale=np.std(anomaly_scores) + 1e-10)
            
            # Expected cost
            expected_cost = (
                (1 - prior_anomaly) * fp_rate * cost_fp +
                prior_anomaly * fn_rate * cost_fn
            )
            
            if expected_cost < best_cost:
                best_cost = expected_cost
                best_threshold = t
        
        return float(best_threshold), float(best_cost)
    
    @staticmethod
    def multi_class_thresholds(
        scores_by_severity: Dict[int, np.ndarray],
    ) -> Dict[str, float]:
        """
        Derive thresholds for multi-class severity levels.
        
        Args:
            scores_by_severity: {severity_level: scores} where 0=normal, 1=mild, 2=moderate, 3=severe
        
        Returns:
            {"mild": t1, "moderate": t2, "severe": t3}
        """
        if 0 not in scores_by_severity or len(scores_by_severity[0]) == 0:
            raise ValueError("Must have normal (severity=0) samples for calibration")
        
        normal_scores = scores_by_severity[0]
        thresholds = {}
        
        # Mild threshold: separates normal from mild
        if 1 in scores_by_severity and len(scores_by_severity[1]) > 0:
            t_mild, _, _ = ThresholdDerivation.roc_optimal(normal_scores, scores_by_severity[1])
            thresholds["mild"] = t_mild
        else:
            thresholds["mild"] = float(np.percentile(normal_scores, 90))
        
        # Moderate threshold: separates mild from moderate
        if 2 in scores_by_severity and len(scores_by_severity[2]) > 0:
            mild_scores = scores_by_severity.get(1, normal_scores)
            t_mod, _, _ = ThresholdDerivation.roc_optimal(mild_scores, scores_by_severity[2])
            thresholds["moderate"] = t_mod
        else:
            thresholds["moderate"] = thresholds["mild"] * 2
        
        # Severe threshold: separates moderate from severe
        if 3 in scores_by_severity and len(scores_by_severity[3]) > 0:
            mod_scores = scores_by_severity.get(2, normal_scores)
            t_sev, _, _ = ThresholdDerivation.roc_optimal(mod_scores, scores_by_severity[3])
            thresholds["severe"] = t_sev
        else:
            thresholds["severe"] = thresholds["moderate"] * 2
        
        return thresholds


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION — Cross-validation and performance metrics
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationMetrics:
    """Compute validation metrics for calibration."""
    
    @staticmethod
    def confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, int]:
        """Compute confusion matrix."""
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    
    @staticmethod
    def precision_recall_f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute precision, recall, and F1 score."""
        cm = ValidationMetrics.confusion_matrix(y_true, y_pred)
        
        precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) > 0 else 0.0
        recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (cm["tp"] + cm["tn"]) / (cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"])
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            **cm,
        }
    
    @staticmethod
    def cross_validate(
        samples: List[CalibrationSample],
        extractor: MetricExtractor,
        stage_name: str,
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        K-fold cross-validation for a single stage.
        
        Returns average metrics and per-fold breakdown.
        """
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        fold_size = len(samples) // n_folds
        
        all_metrics = []
        all_thresholds = []
        
        for fold in range(n_folds):
            # Split
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else len(samples)
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
            
            train_samples = [samples[i] for i in train_idx]
            test_samples = [samples[i] for i in test_idx]
            
            # Extract metrics for training
            train_normal_scores = []
            train_anomaly_scores = []
            
            for s in train_samples:
                stage_method = getattr(extractor, f"extract_stage_{stage_name.replace('ot', '1_ot').replace('sgw', '2_sgw').replace('rkhs', '3_rkhs').replace('ph', '4_ph').replace('ga', '5_ga')}")
                
                # Simpler mapping
                method_map = {
                    "ot": extractor.extract_stage_1_ot,
                    "sgw": extractor.extract_stage_2_sgw,
                    "rkhs": extractor.extract_stage_3_rkhs,
                    "ph": extractor.extract_stage_4_ph,
                    "ga": extractor.extract_stage_5_ga,
                }
                metrics = method_map[stage_name](s.distribution_a, s.distribution_b)
                
                if s.label == Label.NORMAL:
                    train_normal_scores.append(metrics.primary_metric)
                else:
                    train_anomaly_scores.append(metrics.primary_metric)
            
            train_normal_scores = np.array(train_normal_scores)
            train_anomaly_scores = np.array(train_anomaly_scores)
            
            if len(train_normal_scores) == 0 or len(train_anomaly_scores) == 0:
                continue
            
            # Derive threshold on training data
            threshold, auc, _ = ThresholdDerivation.roc_optimal(train_normal_scores, train_anomaly_scores)
            all_thresholds.append(threshold)
            
            # Evaluate on test data
            y_true = []
            y_pred = []
            
            method_map = {
                "ot": extractor.extract_stage_1_ot,
                "sgw": extractor.extract_stage_2_sgw,
                "rkhs": extractor.extract_stage_3_rkhs,
                "ph": extractor.extract_stage_4_ph,
                "ga": extractor.extract_stage_5_ga,
            }
            
            for s in test_samples:
                metrics = method_map[stage_name](s.distribution_a, s.distribution_b)
                y_true.append(1 if s.label.is_anomaly else 0)
                y_pred.append(1 if metrics.primary_metric >= threshold else 0)
            
            fold_metrics = ValidationMetrics.precision_recall_f1(np.array(y_true), np.array(y_pred))
            fold_metrics["auc"] = auc
            all_metrics.append(fold_metrics)
        
        if len(all_metrics) == 0:
            return {
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mean_f1": 0.0,
                "mean_auc": 0.0,
                "std_f1": 0.0,
                "mean_threshold": 0.0,
                "std_threshold": 0.0,
                "per_fold": [],
            }
        
        return {
            "mean_precision": float(np.mean([m["precision"] for m in all_metrics])),
            "mean_recall": float(np.mean([m["recall"] for m in all_metrics])),
            "mean_f1": float(np.mean([m["f1"] for m in all_metrics])),
            "mean_auc": float(np.mean([m.get("auc", 0.5) for m in all_metrics])),
            "std_f1": float(np.std([m["f1"] for m in all_metrics])),
            "mean_threshold": float(np.mean(all_thresholds)),
            "std_threshold": float(np.std(all_thresholds)),
            "per_fold": all_metrics,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATOR — Main calibration engine
# ═══════════════════════════════════════════════════════════════════════════════

class OracleCalibrator:
    """
    Main calibration engine for Oracle Node.
    
    Usage:
        calibrator = OracleCalibrator(domain="climate")
        calibrator.add_samples([...])
        profile = calibrator.calibrate(n_folds=5)
        profile.save("calibration_climate_v1.json")
    """
    
    def __init__(self, domain: str):
        self.domain = domain
        self.samples: List[CalibrationSample] = []
        self.extractor = MetricExtractor()
        self.stage_names = ["ot", "sgw", "rkhs", "ph", "ga"]
    
    def add_sample(
        self,
        distribution_a: ArrayLike,
        distribution_b: ArrayLike,
        label: Label,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single calibration sample."""
        self.samples.append(CalibrationSample(
            distribution_a=np.asarray(distribution_a),
            distribution_b=np.asarray(distribution_b),
            label=label,
            domain=self.domain,
            metadata=metadata or {},
        ))
    
    def add_samples(self, samples: List[CalibrationSample]) -> None:
        """Add multiple calibration samples."""
        self.samples.extend(samples)
    
    def generate_synthetic_samples(
        self,
        n_normal: int = 100,
        n_mild: int = 50,
        n_moderate: int = 50,
        n_severe: int = 50,
        size: int = 1000,
    ) -> None:
        """
        Generate synthetic calibration samples.
        
        For real calibration, use actual domain data. This is for testing
        the calibration framework itself.
        """
        np.random.seed(42)
        
        # Normal: small natural variation
        for _ in range(n_normal):
            base = np.random.randn(size)
            shift = np.random.uniform(-0.1, 0.1)
            noise = np.random.randn(size) * 0.05
            self.add_sample(base, base + shift + noise, Label.NORMAL)
        
        # Mild anomaly: small but detectable shift
        for _ in range(n_mild):
            base = np.random.randn(size)
            shift = np.random.uniform(0.3, 0.6) * np.random.choice([-1, 1])
            self.add_sample(base, base + shift, Label.ANOMALY_MILD)
        
        # Moderate anomaly: clear shift + shape change
        for _ in range(n_moderate):
            base = np.random.randn(size)
            shift = np.random.uniform(0.6, 1.0) * np.random.choice([-1, 1])
            scale_change = np.random.uniform(1.2, 1.5)
            self.add_sample(base, base * scale_change + shift, Label.ANOMALY_MODERATE)
        
        # Severe anomaly: major distribution change
        for _ in range(n_severe):
            base = np.random.randn(size)
            # Bimodal or completely different
            if np.random.rand() > 0.5:
                altered = np.concatenate([
                    np.random.randn(size // 2) - 2,
                    np.random.randn(size // 2) + 2,
                ])
                np.random.shuffle(altered)
            else:
                altered = np.random.exponential(2, size) - 2
            self.add_sample(base, altered, Label.ANOMALY_SEVERE)
    
    def _extract_all_metrics(self) -> Dict[str, Dict[int, List[float]]]:
        """
        Extract metrics for all samples.
        
        Returns: {stage_name: {severity_level: [scores]}}
        """
        results: Dict[str, Dict[int, List[float]]] = {
            stage: {0: [], 1: [], 2: [], 3: []} for stage in self.stage_names
        }
        
        print(f"Extracting metrics for {len(self.samples)} samples...")
        
        for i, sample in enumerate(self.samples):
            if (i + 1) % 10 == 0:
                print(f"  Processing sample {i + 1}/{len(self.samples)}")
            
            all_metrics = self.extractor.extract_all(sample.distribution_a, sample.distribution_b)
            severity = sample.label.severity
            
            for stage_name, metrics in all_metrics.items():
                results[stage_name][severity].append(metrics.primary_metric)
        
        return results
    
    def calibrate(self, n_folds: int = 5) -> CalibrationProfile:
        """
        Run full calibration pipeline.
        
        1. Extract metrics for all samples
        2. Derive thresholds for each stage using ROC analysis
        3. Cross-validate thresholds
        4. Generate calibration profile
        """
        if len(self.samples) < 10:
            raise ValueError(f"Need at least 10 samples for calibration, got {len(self.samples)}")
        
        print(f"\n{'='*70}")
        print(f"CALIBRATING ORACLE NODE — Domain: {self.domain}")
        print(f"{'='*70}")
        print(f"Total samples: {len(self.samples)}")
        
        # Count by label
        label_counts = {}
        for s in self.samples:
            label_counts[s.label.name] = label_counts.get(s.label.name, 0) + 1
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")
        
        # Extract all metrics
        all_metrics = self._extract_all_metrics()
        
        # Derive thresholds and validate for each stage
        stage_thresholds: Dict[str, ThresholdConfig] = {}
        overall_validation: Dict[str, float] = {}
        
        for stage_name in self.stage_names:
            print(f"\n--- Stage: {stage_name.upper()} ---")
            
            scores_by_severity = all_metrics[stage_name]
            normal_scores = np.array(scores_by_severity[0])
            anomaly_scores = np.concatenate([
                np.array(scores_by_severity[1]),
                np.array(scores_by_severity[2]),
                np.array(scores_by_severity[3]),
            ])
            
            if len(normal_scores) == 0 or len(anomaly_scores) == 0:
                print(f"  SKIPPED: Insufficient samples")
                continue
            
            # Derive binary threshold (normal vs any anomaly)
            binary_threshold, auc, j_stat = ThresholdDerivation.roc_optimal(normal_scores, anomaly_scores)
            print(f"  Binary threshold: {binary_threshold:.4f} (AUC={auc:.3f}, J={j_stat:.3f})")
            
            # Derive multi-class thresholds
            multi_thresholds = ThresholdDerivation.multi_class_thresholds(scores_by_severity)
            print(f"  Multi-class thresholds: mild={multi_thresholds['mild']:.4f}, moderate={multi_thresholds['moderate']:.4f}, severe={multi_thresholds['severe']:.4f}")
            
            # Cross-validation
            cv_results = ValidationMetrics.cross_validate(
                self.samples, self.extractor, stage_name, n_folds
            )
            print(f"  CV Results: F1={cv_results['mean_f1']:.3f}±{cv_results['std_f1']:.3f}, AUC={cv_results['mean_auc']:.3f}")
            
            # Confidence interval via bootstrap
            _, ci_low, ci_high = ThresholdDerivation.percentile_based(normal_scores, 95)
            
            stage_thresholds[stage_name] = ThresholdConfig(
                stage_name=stage_name,
                thresholds=multi_thresholds,
                derivation_method="roc_optimal_with_multiclass",
                confidence_interval=(ci_low, ci_high),
                sample_size=len(self.samples),
                validation_auc=cv_results["mean_auc"],
                validation_f1=cv_results["mean_f1"],
            )
            
            overall_validation[f"{stage_name}_auc"] = cv_results["mean_auc"]
            overall_validation[f"{stage_name}_f1"] = cv_results["mean_f1"]
        
        # Create profile
        profile = CalibrationProfile(
            domain=self.domain,
            version="1.0.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            stage_thresholds=stage_thresholds,
            validation_metrics=overall_validation,
            sample_count=len(self.samples),
            cross_validation_folds=n_folds,
        )
        
        print(f"\n{'='*70}")
        print(f"CALIBRATION COMPLETE")
        print(f"{'='*70}")
        
        return profile
    
    def save_profile(self, profile: CalibrationProfile, path: str) -> None:
        """Save calibration profile to JSON file."""
        data = profile.to_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved calibration profile to: {path}")
    
    @staticmethod
    def load_profile(path: str) -> CalibrationProfile:
        """Load calibration profile from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return CalibrationProfile.from_dict(data)


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATED CLASSIFIER — Use calibrated thresholds for classification
# ═══════════════════════════════════════════════════════════════════════════════

class CalibratedClassifier:
    """
    Classifier using calibrated thresholds.
    
    This replaces the hardcoded thresholds in server.py.
    """
    
    def __init__(self, profile: CalibrationProfile):
        self.profile = profile
        self.extractor = MetricExtractor()
    
    def classify_stage(
        self,
        stage_name: str,
        metric_value: float,
    ) -> Tuple[str, str]:
        """
        Classify a single stage metric using calibrated thresholds.
        
        Returns: (severity_level, description)
        """
        if stage_name not in self.profile.stage_thresholds:
            return "UNKNOWN", "Stage not calibrated"
        
        thresholds = self.profile.stage_thresholds[stage_name].thresholds
        
        if metric_value >= thresholds.get("severe", float('inf')):
            return "SEVERE", f"Metric {metric_value:.4f} >= severe threshold {thresholds['severe']:.4f}"
        elif metric_value >= thresholds.get("moderate", float('inf')):
            return "MODERATE", f"Metric {metric_value:.4f} >= moderate threshold {thresholds['moderate']:.4f}"
        elif metric_value >= thresholds.get("mild", float('inf')):
            return "MILD", f"Metric {metric_value:.4f} >= mild threshold {thresholds['mild']:.4f}"
        else:
            return "NORMAL", f"Metric {metric_value:.4f} below all thresholds"
    
    def classify_full(
        self,
        distribution_a: np.ndarray,
        distribution_b: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run full classification with calibrated thresholds.
        """
        all_metrics = self.extractor.extract_all(distribution_a, distribution_b)
        
        results = {}
        for stage_name, metrics in all_metrics.items():
            severity, description = self.classify_stage(stage_name, metrics.primary_metric)
            results[stage_name] = {
                "primary_metric": metrics.primary_metric,
                "all_metrics": metrics.all_metrics,
                "severity": severity,
                "description": description,
                "computation_time_ms": metrics.computation_time_ms,
            }
        
        # Overall severity (max across stages)
        severities = [r["severity"] for r in results.values()]
        severity_order = {"NORMAL": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3, "UNKNOWN": 0}
        max_severity = max(severities, key=lambda s: severity_order.get(s, 0))
        
        results["overall"] = {
            "severity": max_severity,
            "calibration_domain": self.profile.domain,
            "calibration_version": self.profile.version,
        }
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_calibration_test():
    """Run a full calibration test with synthetic data."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║              C A L I B R A T I O N   F R A M E W O R K   T E S T                        ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create calibrator
    calibrator = OracleCalibrator(domain="synthetic_test")
    
    # Generate synthetic samples
    calibrator.generate_synthetic_samples(
        n_normal=100,
        n_mild=50,
        n_moderate=50,
        n_severe=50,
        size=256,  # Smaller for faster testing
    )
    
    # Run calibration
    profile = calibrator.calibrate(n_folds=5)
    
    # Save profile
    output_path = Path(__file__).parent / "calibration_synthetic_test.json"
    calibrator.save_profile(profile, str(output_path))
    
    # Test classifier
    print("\n--- Testing Calibrated Classifier ---")
    classifier = CalibratedClassifier(profile)
    
    # Test with new samples
    np.random.seed(999)  # Different seed
    
    test_cases = [
        ("Normal", np.random.randn(256), np.random.randn(256) + 0.05),
        ("Mild", np.random.randn(256), np.random.randn(256) + 0.5),
        ("Severe", np.random.randn(256), np.random.exponential(2, 256)),
    ]
    
    for name, dist_a, dist_b in test_cases:
        result = classifier.classify_full(dist_a, dist_b)
        print(f"\n{name} case: Overall severity = {result['overall']['severity']}")
        for stage in ["ot", "rkhs"]:
            if stage in result:
                print(f"  {stage}: {result[stage]['severity']} (metric={result[stage]['primary_metric']:.4f})")
    
    print("\n✓ Calibration framework test complete!")
    return profile


if __name__ == "__main__":
    run_calibration_test()
