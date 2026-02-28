"""QTT Physics VM — Multi-objective Pareto optimizer and claim scorer.

Implements the claim scoring engine from the EXASCALE_IP_EXECUTION_PLAN:
non-dominated sorting, composite scoring across RF performance,
manufacturability, novelty, claim breadth, market relevance, and
verification quality.

Architecture:
    ParetoOptimizer
      ├── Non-dominated sorting (NSGA-II style fast sort)
      ├── Crowding distance for diversity preservation
      ├── Composite claim scoring per the execution plan weights
      └── Decision thresholds: FILE / HOLD / DISCARD

    ScoredCandidate
      ├── DesignPoint (raw simulation metrics)
      ├── Pareto front index
      ├── Dimensional scores (RF, mfg, novelty, breadth, market, verif)
      └── Composite score + triage decision

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .sweep import DesignPoint


# ─────────────────────────────────────────────────────────────────────
# Scoring weights from EXASCALE_IP_EXECUTION_PLAN.md
# ─────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "rf_performance": 0.25,
    "manufacturability": 0.15,
    "novelty_proxy": 0.20,
    "claim_breadth": 0.15,
    "market_relevance": 0.15,
    "verification_quality": 0.10,
}

# Decision thresholds
THRESHOLD_FILE_NOW = 0.75
THRESHOLD_NOVELTY_FOR_FILE = 0.6
THRESHOLD_HOLD_SCORE = 0.50
THRESHOLD_HOLD_BREADTH = 0.6
THRESHOLD_DISCARD_BREADTH = 0.3
THRESHOLD_DISCARD_MARKET = 0.3


@dataclass
class ScoredCandidate:
    """A design point scored by the claim scoring engine.

    Extends ``DesignPoint`` with Pareto analysis, dimensional scores,
    composite score, and a triage decision.
    """

    point: DesignPoint
    """Original simulation metrics."""

    pareto_front: int = -1
    """Pareto front index (0 = first/optimal front)."""

    crowding_distance: float = 0.0
    """Crowding distance within the Pareto front."""

    # ── Dimensional scores (0–1, higher is better) ──────────────────
    rf_performance: float = 0.0
    """RF performance score: weighted Pareto rank across gain,
    efficiency, bandwidth, pattern quality, impedance match."""

    manufacturability: float = 0.0
    """Manufacturability score: based on minimum feature size,
    layer count, substrate feasibility."""

    novelty_proxy: float = 0.0
    """Novelty score: embedding distance from known topology families."""

    claim_breadth: float = 0.0
    """Claim breadth: number of variants coverable by one claim family."""

    market_relevance: float = 0.0
    """Market relevance: band support, OEM constraints, integration fit."""

    verification_quality: float = 0.0
    """Verification quality: attested reproducibility, metric variance."""

    composite_score: float = 0.0
    """Weighted sum of dimensional scores."""

    triage: str = "pending"
    """Decision: ``"FILE_NOW"``, ``"HOLD"``, ``"DISCARD"``, or ``"pending"``."""

    triage_reason: str = ""
    """Human-readable reason for the triage decision."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON output."""
        return {
            "candidate_id": self.point.candidate_id,
            "params": self.point.params,
            "param_hash": self.point.param_hash,
            "pareto_front": self.pareto_front,
            "crowding_distance": self.crowding_distance,
            "scores": {
                "rf_performance": self.rf_performance,
                "manufacturability": self.manufacturability,
                "novelty_proxy": self.novelty_proxy,
                "claim_breadth": self.claim_breadth,
                "market_relevance": self.market_relevance,
                "verification_quality": self.verification_quality,
            },
            "composite_score": self.composite_score,
            "triage": self.triage,
            "triage_reason": self.triage_reason,
            "metrics": {
                "s11_min_db": self.point.s11_min_db,
                "peak_gain_dbi": self.point.peak_gain_dbi,
                "fractional_bandwidth": self.point.fractional_bandwidth,
                "radiation_efficiency": self.point.radiation_efficiency,
                "vswr_min": self.point.vswr_min,
                "wall_time_s": self.point.wall_time_s,
            },
        }


@dataclass
class ParetoResult:
    """Result of Pareto analysis on a sweep result."""

    candidates: list[ScoredCandidate] = field(default_factory=list)
    """All scored candidates, sorted by composite score (descending)."""

    n_fronts: int = 0
    """Number of Pareto fronts identified."""

    objectives_used: list[str] = field(default_factory=list)
    """Names of objectives used in Pareto sorting."""

    weights_used: dict[str, float] = field(default_factory=dict)
    """Weights used for composite scoring."""

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    def front(self, idx: int = 0) -> list[ScoredCandidate]:
        """Return candidates on a specific Pareto front."""
        return [c for c in self.candidates if c.pareto_front == idx]

    def file_now(self) -> list[ScoredCandidate]:
        """Return candidates triaged as FILE_NOW."""
        return [c for c in self.candidates if c.triage == "FILE_NOW"]

    def hold(self) -> list[ScoredCandidate]:
        """Return candidates triaged as HOLD."""
        return [c for c in self.candidates if c.triage == "HOLD"]

    def top_n(self, n: int = 10) -> list[ScoredCandidate]:
        """Return the top N candidates by composite score."""
        return self.candidates[:n]

    def summary(self) -> dict[str, Any]:
        """Summary statistics."""
        return {
            "n_candidates": self.n_candidates,
            "n_fronts": self.n_fronts,
            "n_file_now": len(self.file_now()),
            "n_hold": len(self.hold()),
            "n_discard": len(
                [c for c in self.candidates if c.triage == "DISCARD"]
            ),
            "objectives": self.objectives_used,
            "top_score": (
                self.candidates[0].composite_score
                if self.candidates
                else 0.0
            ),
            "top_candidate": (
                self.candidates[0].point.candidate_id
                if self.candidates
                else ""
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise full Pareto result to JSON."""
        data = {
            "summary": self.summary(),
            "candidates": [c.to_dict() for c in self.candidates],
        }
        return json.dumps(data, indent=indent, default=str)


class ParetoOptimizer:
    """Multi-objective Pareto optimizer and claim scoring engine.

    Implements the scoring framework from the EXASCALE_IP_EXECUTION_PLAN:

    1. Non-dominated sorting across RF objectives
    2. Crowding distance for diversity
    3. Dimensional scoring (RF, manufacturability, novelty, etc.)
    4. Composite weighted score
    5. Triage decisions (FILE_NOW, HOLD, DISCARD)

    Parameters
    ----------
    objectives : list[str]
        Objective names for Pareto sorting.  Each must be a numeric
        attribute of ``DesignPoint``.  Maximised by default.
    maximize : list[bool] | None
        Whether each objective is maximised (True) or minimised (False).
        If None, all are maximised.
    weights : dict[str, float] | None
        Override default claim scoring weights.
    min_feature_um : float
        Minimum etchable feature size for manufacturability scoring.
    target_bands : list[tuple[float, float]]
        Target frequency bands [(f_low, f_high), ...] for market
        relevance scoring.
    """

    def __init__(
        self,
        objectives: list[str] | None = None,
        maximize: list[bool] | None = None,
        weights: dict[str, float] | None = None,
        min_feature_um: float = 75.0,
        target_bands: list[tuple[float, float]] | None = None,
    ) -> None:
        self._objectives = objectives or [
            "peak_gain_dbi",
            "fractional_bandwidth",
            "radiation_efficiency",
        ]
        self._maximize = maximize or [True] * len(self._objectives)
        self._weights = weights or dict(DEFAULT_WEIGHTS)
        self._min_feature_um = min_feature_um
        self._target_bands = target_bands or []

    def optimize(
        self,
        points: list[DesignPoint],
    ) -> ParetoResult:
        """Run Pareto analysis and scoring on a collection of design points.

        Parameters
        ----------
        points : list[DesignPoint]
            Design points from a sweep (only successful ones are scored).

        Returns
        -------
        ParetoResult
        """
        # Filter to successful points only
        valid = [p for p in points if p.success]
        if not valid:
            return ParetoResult(
                objectives_used=self._objectives,
                weights_used=self._weights,
            )

        # ── 1. Extract objective matrix ──────────────────────────────
        n = len(valid)
        m = len(self._objectives)
        obj_matrix = np.zeros((n, m))
        for i, point in enumerate(valid):
            for j, obj_name in enumerate(self._objectives):
                val = getattr(point, obj_name, 0.0)
                if isinstance(val, complex):
                    val = abs(val)
                obj_matrix[i, j] = float(val)

        # ── 2. Non-dominated sorting ────────────────────────────────
        fronts = self._fast_non_dominated_sort(obj_matrix)

        # ── 3. Crowding distance ────────────────────────────────────
        crowding = np.zeros(n)
        for front_indices in fronts:
            if len(front_indices) < 3:
                for idx in front_indices:
                    crowding[idx] = float("inf")
            else:
                front_crowding = self._crowding_distance(
                    obj_matrix[front_indices]
                )
                for k, idx in enumerate(front_indices):
                    crowding[idx] = front_crowding[k]

        # ── 4. Build ScoredCandidates ────────────────────────────────
        candidates: list[ScoredCandidate] = []
        for i, point in enumerate(valid):
            front_idx = -1
            for fi, front_indices in enumerate(fronts):
                if i in front_indices:
                    front_idx = fi
                    break

            sc = ScoredCandidate(
                point=point,
                pareto_front=front_idx,
                crowding_distance=crowding[i],
            )

            # ── 5. Dimensional scoring ────────────────────────────
            sc.rf_performance = self._score_rf(point, valid)
            sc.manufacturability = self._score_manufacturability(point)
            sc.novelty_proxy = self._score_novelty(point, valid)
            sc.claim_breadth = self._score_claim_breadth(point, valid)
            sc.market_relevance = self._score_market_relevance(point)
            sc.verification_quality = self._score_verification(point)

            # ── 6. Composite score ────────────────────────────────
            sc.composite_score = (
                self._weights["rf_performance"] * sc.rf_performance
                + self._weights["manufacturability"] * sc.manufacturability
                + self._weights["novelty_proxy"] * sc.novelty_proxy
                + self._weights["claim_breadth"] * sc.claim_breadth
                + self._weights["market_relevance"] * sc.market_relevance
                + self._weights["verification_quality"]
                * sc.verification_quality
            )

            # ── 7. Triage decision ────────────────────────────────
            sc.triage, sc.triage_reason = self._triage(sc)

            candidates.append(sc)

        # Sort by composite score descending
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

        return ParetoResult(
            candidates=candidates,
            n_fronts=len(fronts),
            objectives_used=self._objectives,
            weights_used=self._weights,
        )

    # ── Non-dominated sorting (NSGA-II) ──────────────────────────────

    def _fast_non_dominated_sort(
        self, obj_matrix: NDArray
    ) -> list[list[int]]:
        """Fast non-dominated sorting (NSGA-II algorithm).

        Returns a list of fronts, where each front is a list of indices.
        Front 0 is the Pareto-optimal set.
        """
        n = obj_matrix.shape[0]
        domination_count = np.zeros(n, dtype=int)
        dominated_by: list[list[int]] = [[] for _ in range(n)]
        fronts: list[list[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(obj_matrix[i], obj_matrix[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(obj_matrix[j], obj_matrix[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front: list[int] = []
            for i in fronts[k]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _dominates(self, a: NDArray, b: NDArray) -> bool:
        """Check if solution a dominates solution b.

        a dominates b if a is at least as good in all objectives
        and strictly better in at least one.
        """
        at_least_as_good = True
        strictly_better = False

        for k in range(len(a)):
            if self._maximize[k]:
                if a[k] < b[k]:
                    at_least_as_good = False
                    break
                if a[k] > b[k]:
                    strictly_better = True
            else:
                if a[k] > b[k]:
                    at_least_as_good = False
                    break
                if a[k] < b[k]:
                    strictly_better = True

        return at_least_as_good and strictly_better

    def _crowding_distance(self, front_obj: NDArray) -> NDArray:
        """Compute crowding distance for a single Pareto front."""
        n = front_obj.shape[0]
        m = front_obj.shape[1]
        distances = np.zeros(n)

        for j in range(m):
            sorted_idx = np.argsort(front_obj[:, j])
            distances[sorted_idx[0]] = float("inf")
            distances[sorted_idx[-1]] = float("inf")

            obj_range = (
                front_obj[sorted_idx[-1], j] - front_obj[sorted_idx[0], j]
            )
            if obj_range < 1e-30:
                continue

            for k in range(1, n - 1):
                distances[sorted_idx[k]] += (
                    front_obj[sorted_idx[k + 1], j]
                    - front_obj[sorted_idx[k - 1], j]
                ) / obj_range

        return distances

    # ── Dimensional scoring functions ────────────────────────────────

    def _score_rf(
        self, point: DesignPoint, all_points: list[DesignPoint]
    ) -> float:
        """RF performance score (0–1).

        Weighted combination of:
        - Gain rank among all candidates (0.30)
        - Bandwidth rank (0.25)
        - Return loss depth (S₁₁) rank (0.25)
        - Efficiency rank (0.20)
        """
        if not all_points:
            return 0.0

        gains = [p.peak_gain_dbi for p in all_points]
        bws = [p.fractional_bandwidth for p in all_points]
        s11s = [p.s11_min_db for p in all_points]
        effs = [p.radiation_efficiency for p in all_points]

        def _rank_score(val: float, population: list[float], higher_better: bool = True) -> float:
            if len(population) <= 1:
                return 0.5
            sorted_pop = sorted(population, reverse=higher_better)
            rank = sorted_pop.index(val)
            return 1.0 - rank / (len(population) - 1)

        gain_s = _rank_score(point.peak_gain_dbi, gains, True)
        bw_s = _rank_score(point.fractional_bandwidth, bws, True)
        s11_s = _rank_score(point.s11_min_db, s11s, False)  # Lower is better
        eff_s = _rank_score(point.radiation_efficiency, effs, True)

        return 0.30 * gain_s + 0.25 * bw_s + 0.25 * s11_s + 0.20 * eff_s

    def _score_manufacturability(self, point: DesignPoint) -> float:
        """Manufacturability score (0–1).

        Based on parameter values and feature sizes.  Penalises designs
        with very thin dimensions or extreme aspect ratios.
        """
        score = 1.0

        # Penalty for very small features (< min_feature requirement)
        params = point.params
        for key in ("patch_width", "patch_length", "slot_width",
                     "slot_arm_width", "wire_radius", "gap_half"):
            if key in params:
                # Normalised dimensions → assume 1 mm/unit for scoring
                dim_um = params[key] * 1000.0  # rough conversion
                if dim_um < self._min_feature_um:
                    score *= dim_um / self._min_feature_um

        # Bonus for standard substrate heights
        if "substrate_height" in params:
            h = params["substrate_height"]
            # Standard thicknesses in normalised units
            standard = [0.02, 0.04, 0.06, 0.08, 0.10]
            min_dist = min(abs(h - s) for s in standard)
            if min_dist < 0.005:
                score *= 1.05  # Small bonus for standard thickness

        return min(1.0, max(0.0, score))

    def _score_novelty(
        self, point: DesignPoint, all_points: list[DesignPoint]
    ) -> float:
        """Novelty proxy score (0–1).

        Measures geometric distance from all other candidates in
        normalised parameter space.  Outliers score higher.
        """
        if len(all_points) <= 1:
            return 0.5

        # Build parameter vectors for all points
        all_keys = sorted(point.params.keys())
        if not all_keys:
            return 0.5

        # Normalise each parameter to [0, 1] across the population
        param_matrix = np.zeros((len(all_points), len(all_keys)))
        for i, p in enumerate(all_points):
            for j, key in enumerate(all_keys):
                param_matrix[i, j] = p.params.get(key, 0.0)

        # Min-max normalisation
        mins = param_matrix.min(axis=0)
        maxs = param_matrix.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-30] = 1.0  # Avoid division by zero
        normalised = (param_matrix - mins) / ranges

        # Find this point's index
        target_idx = -1
        for i, p in enumerate(all_points):
            if p.param_hash == point.param_hash:
                target_idx = i
                break
        if target_idx < 0:
            return 0.5

        # Average Euclidean distance to all other points
        target_vec = normalised[target_idx]
        distances = np.sqrt(
            np.sum((normalised - target_vec) ** 2, axis=1)
        )
        distances[target_idx] = 0.0  # Exclude self
        avg_dist = np.mean(distances[distances > 0])

        # Max possible distance in unit hypercube
        max_dist = np.sqrt(len(all_keys))
        if max_dist < 1e-30:
            return 0.5

        return min(1.0, avg_dist / (0.5 * max_dist))

    def _score_claim_breadth(
        self, point: DesignPoint, all_points: list[DesignPoint]
    ) -> float:
        """Claim breadth score (0–1).

        Estimates how many nearby variants could be covered by one
        claim family.  Points with many similar-performing neighbours
        score higher (broader claim coverage).
        """
        if len(all_points) <= 1:
            return 0.5

        # Count neighbours within a performance similarity threshold
        threshold_gain = 1.0  # dBi
        threshold_bw = 0.05   # 5% fractional BW
        threshold_s11 = 3.0   # dB

        n_similar = 0
        for p in all_points:
            if p.param_hash == point.param_hash:
                continue
            gain_close = abs(p.peak_gain_dbi - point.peak_gain_dbi) < threshold_gain
            bw_close = abs(p.fractional_bandwidth - point.fractional_bandwidth) < threshold_bw
            s11_close = abs(p.s11_min_db - point.s11_min_db) < threshold_s11
            if gain_close and bw_close and s11_close:
                n_similar += 1

        # Normalise: 0 similar → 0.2, many → 1.0
        max_expected = max(1, len(all_points) // 3)
        return min(1.0, 0.2 + 0.8 * n_similar / max_expected)

    def _score_market_relevance(self, point: DesignPoint) -> float:
        """Market relevance score (0–1).

        Based on resonant frequency matching target bands and
        whether the design achieves usable S₁₁ (< -10 dB).
        """
        score = 0.5  # Base score

        # Check if resonant frequency falls in a target band
        if self._target_bands:
            for f_lo, f_hi in self._target_bands:
                if f_lo <= point.f_resonance <= f_hi:
                    score += 0.3
                    break
                # Check if bandwidth overlaps the band
                if (
                    point.bandwidth_f_low <= f_hi
                    and point.bandwidth_f_high >= f_lo
                ):
                    score += 0.15
                    break

        # Bonus for achievable S₁₁
        if point.s11_min_db < -10.0:
            score += 0.2
        elif point.s11_min_db < -6.0:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _score_verification(self, point: DesignPoint) -> float:
        """Verification quality score (0–1).

        Based on simulation health metrics:
        - Non-zero frequency bins
        - Finite DFT norms
        - Reasonable gain values
        - Bounded rank (QTT health)
        """
        score = 0.0

        # Frequency bins extracted
        if point.n_freq_bins > 10:
            score += 0.3
        elif point.n_freq_bins > 0:
            score += 0.15

        # DFT norms non-zero
        n_nonzero = sum(
            1 for v in point.dft_norms.values() if abs(v) > 1e-30
        )
        if n_nonzero >= 8:
            score += 0.25
        elif n_nonzero >= 4:
            score += 0.15

        # Gain is physically reasonable (-30 to +40 dBi)
        if -30.0 < point.peak_gain_dbi < 40.0:
            score += 0.25

        # Rank is bounded and reasonable
        if 0 < point.chi_max <= 64:
            score += 0.2

        return min(1.0, score)

    def _triage(
        self, sc: ScoredCandidate
    ) -> tuple[str, str]:
        """Apply decision thresholds from EXASCALE_IP_EXECUTION_PLAN.

        Returns (decision, reason).
        """
        total = sc.composite_score
        novelty = sc.novelty_proxy
        breadth = sc.claim_breadth
        market = sc.market_relevance

        if total >= THRESHOLD_FILE_NOW and novelty >= THRESHOLD_NOVELTY_FOR_FILE:
            return "FILE_NOW", (
                f"Score {total:.2f} ≥ {THRESHOLD_FILE_NOW} "
                f"and novelty {novelty:.2f} ≥ {THRESHOLD_NOVELTY_FOR_FILE}"
            )

        if total >= THRESHOLD_FILE_NOW and novelty < THRESHOLD_NOVELTY_FOR_FILE:
            return "HOLD", (
                f"Score {total:.2f} ≥ {THRESHOLD_FILE_NOW} "
                f"but novelty {novelty:.2f} < {THRESHOLD_NOVELTY_FOR_FILE} "
                "— search for broadening variants"
            )

        if total >= THRESHOLD_HOLD_SCORE and breadth >= THRESHOLD_HOLD_BREADTH:
            return "HOLD", (
                f"Score {total:.2f} ≥ {THRESHOLD_HOLD_SCORE} "
                f"with breadth {breadth:.2f} ≥ {THRESHOLD_HOLD_BREADTH} "
                "— portfolio candidate for family bundling"
            )

        if (
            total < THRESHOLD_HOLD_SCORE
            or (breadth < THRESHOLD_DISCARD_BREADTH
                and market < THRESHOLD_DISCARD_MARKET)
        ):
            return "DISCARD", (
                f"Score {total:.2f} < {THRESHOLD_HOLD_SCORE} "
                f"or breadth {breadth:.2f} and market {market:.2f} "
                "both below thresholds"
            )

        return "HOLD", f"Score {total:.2f} — general hold for review"
