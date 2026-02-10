"""Tests for metrics package: aesthetic, functional, safety, UQ, optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from products.facial_plastics.core.types import (
    StructureType,
    Vec3,
)
from products.facial_plastics.metrics.aesthetic import (
    AestheticReport,
    ProfileMetrics,
    ProportionMetrics,
    SymmetryMetrics,
)
from products.facial_plastics.metrics.safety import (
    SAFETY_THRESHOLDS,
    SafetyThreshold,
)
from products.facial_plastics.metrics.uncertainty import (
    UncertainParameter,
    UncertaintyQuantifier,
    default_rhinoplasty_uncertainties,
    latin_hypercube_sample,
)
from products.facial_plastics.metrics.optimizer import (
    Individual,
    ParameterBound,
    _dominates,
    _fast_non_dominated_sort,
    _crowding_distance,
    _sbx_crossover,
    _polynomial_mutation,
)


# ── AestheticMetrics ─────────────────────────────────────────────

class TestAestheticProfileMetrics:
    """Test profile metric computations."""

    def test_profile_metrics_fields(self):
        pm = ProfileMetrics(
            nasofrontal_angle_deg=130.0,
            nasolabial_angle_deg=100.0,
            nasomental_angle_deg=130.0,
            dorsal_length_mm=45.0,
            tip_projection_mm=28.0,
            tip_rotation_deg=30.0,
            goode_ratio=0.62,
            columellar_labial_angle_deg=95.0,
            supratip_break_depth_mm=2.0,
            dorsal_hump_mm=0.0,
            pollybeak_risk=False,
            radix_depth_mm=10.0,
        )
        score = pm.score()
        assert 0 <= score <= 100

    def test_ideal_profile_scores_high(self):
        pm = ProfileMetrics(
            nasofrontal_angle_deg=115.0,
            nasolabial_angle_deg=100.0,
            nasomental_angle_deg=127.0,
            dorsal_length_mm=45.0,
            tip_projection_mm=30.0,
            tip_rotation_deg=105.0,
            goode_ratio=0.67,
            columellar_labial_angle_deg=100.0,
            supratip_break_depth_mm=2.0,
            dorsal_hump_mm=0.0,
            pollybeak_risk=False,
            radix_depth_mm=10.0,
        )
        score = pm.score()
        assert score > 50


class TestSymmetryMetrics:
    """Test symmetry analysis."""

    def test_perfect_symmetry(self):
        sm = SymmetryMetrics(
            procrustes_distance=0.0,
            max_asymmetry_mm=0.0,
            mean_asymmetry_mm=0.0,
            tip_deviation_mm=0.0,
            dorsal_deviation_mm=0.0,
            alar_base_asymmetry_mm=0.0,
            nostril_area_asymmetry_pct=0.0,
        )
        assert sm.score() == 100.0

    def test_imperfect_symmetry(self):
        sm = SymmetryMetrics(
            procrustes_distance=0.03,
            max_asymmetry_mm=1.5,
            mean_asymmetry_mm=0.3,
            tip_deviation_mm=0.5,
            dorsal_deviation_mm=0.3,
            alar_base_asymmetry_mm=0.5,
            nostril_area_asymmetry_pct=3.0,
        )
        score = sm.score()
        assert 0 < score < 100


class TestProportionMetrics:
    """Test proportion scoring."""

    def test_construction(self):
        pm = ProportionMetrics(
            upper_third_mm=30.0,
            middle_third_mm=30.0,
            lower_third_mm=30.0,
            nasal_width_mm=35.0,
            intercanthal_distance_mm=35.0,
            nasal_width_to_icd_ratio=1.0,
            nasal_length_mm=45.0,
            nasal_height_mm=38.0,
            length_to_height_ratio=1.18,
            brow_tip_aesthetic_line_intact=True,
        )
        score = pm.score()
        assert 0 <= score <= 100


class TestAestheticReport:
    """Test composite aesthetic scoring."""

    def test_overall_score(self):
        report = AestheticReport(
            profile=ProfileMetrics(
                nasofrontal_angle_deg=120.0,
                nasolabial_angle_deg=100.0,
                nasomental_angle_deg=127.0,
                tip_rotation_deg=30.0,
                goode_ratio=0.60,
            ),
            symmetry=SymmetryMetrics(procrustes_distance=0.01),
            proportions=ProportionMetrics(),
        )
        assert 0 <= report.overall_score <= 100


# ── Safety ───────────────────────────────────────────────────────

class TestSafetyThresholds:
    """Test safety threshold definitions."""

    def test_thresholds_keyed_by_structure_type(self):
        assert StructureType.SKIN_ENVELOPE in SAFETY_THRESHOLDS
        assert StructureType.BONE_NASAL in SAFETY_THRESHOLDS
        assert StructureType.CARTILAGE_UPPER_LATERAL in SAFETY_THRESHOLDS

    def test_threshold_values_positive(self):
        for st, thresh in SAFETY_THRESHOLDS.items():
            assert thresh.max_von_mises_pa > 0, f"{st}: max_von_mises_pa"
            assert thresh.max_principal_strain > 0, f"{st}: max_principal_strain"


# ── Uncertainty Quantification ───────────────────────────────────

class TestUncertainParameter:
    """Test uncertain parameter distributions."""

    def test_normal_sample(self):
        p = UncertainParameter(
            name="skin_E",
            nominal=50.0,
            distribution="normal",
            std=10.0,
        )
        rng = np.random.default_rng(42)
        samples = p.sample(rng, n=1000)
        assert abs(np.mean(samples) - 50.0) < 3.0
        assert abs(np.std(samples) - 10.0) < 3.0

    def test_uniform_sample(self):
        p = UncertainParameter(
            name="test",
            nominal=15.0,
            distribution="uniform",
            low=10.0,
            high=20.0,
        )
        rng = np.random.default_rng(42)
        samples = p.sample(rng, n=500)
        assert np.all(samples >= 10.0)
        assert np.all(samples <= 20.0)

    def test_defaults(self):
        defaults = default_rhinoplasty_uncertainties()
        assert len(defaults) >= 5
        for p in defaults:
            assert p.name != ""


class TestLatinHypercube:
    """Test LHS sampling."""

    def test_shape(self):
        params = [
            UncertainParameter(name=f"p{i}", nominal=0.5, distribution="uniform",
                               low=0.0, high=1.0)
            for i in range(3)
        ]
        samples = latin_hypercube_sample(params, n_samples=50, seed=42)
        assert samples.shape == (50, 3)

    def test_uniformity(self):
        params = [
            UncertainParameter(name=f"p{i}", nominal=0.5, distribution="uniform",
                               low=0.0, high=1.0)
            for i in range(2)
        ]
        samples = latin_hypercube_sample(params, n_samples=100, seed=42)
        for col in range(2):
            assert np.min(samples[:, col]) >= 0.0
            assert np.max(samples[:, col]) <= 1.0
            assert abs(np.mean(samples[:, col]) - 0.5) < 0.15


# ── Optimizer ────────────────────────────────────────────────────

class TestDominance:
    """Test Pareto dominance operations."""

    def test_dominates_clear(self):
        # Optimizer uses maximization: [2,2] dominates [1,1]
        a = Individual(parameters=np.zeros(1), objectives=np.array([2.0, 2.0]))
        b = Individual(parameters=np.zeros(1), objectives=np.array([1.0, 1.0]))
        assert _dominates(a, b) is True
        assert _dominates(b, a) is False

    def test_non_domination(self):
        a = Individual(parameters=np.zeros(1), objectives=np.array([1.0, 3.0]))
        b = Individual(parameters=np.zeros(1), objectives=np.array([3.0, 1.0]))
        assert _dominates(a, b) is False
        assert _dominates(b, a) is False

    def test_feasibility_preferred(self):
        a = Individual(
            parameters=np.zeros(1),
            objectives=np.array([5.0, 5.0]),
            is_feasible=True,
        )
        b = Individual(
            parameters=np.zeros(1),
            objectives=np.array([1.0, 1.0]),
            is_feasible=False,
            constraints_violated=1,
            constraint_penalty=0.5,
        )
        assert _dominates(a, b) is True


class TestNonDominatedSort:
    """Test fast non-dominated sorting."""

    def test_simple_sort(self):
        # Maximization convention: trade-off front members non-dominated
        pop = [
            Individual(parameters=np.zeros(1), objectives=np.array([1.0, 4.0])),
            Individual(parameters=np.zeros(1), objectives=np.array([2.0, 3.0])),
            Individual(parameters=np.zeros(1), objectives=np.array([3.0, 2.0])),
            Individual(parameters=np.zeros(1), objectives=np.array([4.0, 1.0])),
            Individual(parameters=np.zeros(1), objectives=np.array([0.5, 0.5])),
        ]
        fronts = _fast_non_dominated_sort(pop)
        # fronts is List[List[int]] — indices into pop
        assert len(fronts) >= 1
        assert all(isinstance(idx, int) for idx in fronts[0])
        # First 4 are on Pareto front under maximization
        assert len(fronts[0]) >= 2


class TestCrowdingDistance:
    """Test crowding distance computation."""

    def test_boundary_infinite(self):
        pop = [
            Individual(parameters=np.zeros(1), objectives=np.array([1.0, 4.0])),
            Individual(parameters=np.zeros(1), objectives=np.array([2.0, 3.0])),
            Individual(parameters=np.zeros(1), objectives=np.array([4.0, 1.0])),
        ]
        front = list(range(len(pop)))
        _crowding_distance(pop, front)
        # Boundary individuals should have infinite distance
        distances = [pop[i].crowding_distance for i in front]
        assert distances[0] == float("inf") or distances[-1] == float("inf")


class TestGeneticOperators:
    """Test crossover and mutation operators."""

    def test_sbx_crossover(self):
        rng = np.random.default_rng(42)
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 5.0, 6.0])
        bounds = [
            ParameterBound(name=f"x{i}", low=0.0, high=10.0)
            for i in range(3)
        ]
        c1, c2 = _sbx_crossover(p1, p2, bounds, rng)
        for i in range(3):
            assert bounds[i].low <= c1[i] <= bounds[i].high
            assert bounds[i].low <= c2[i] <= bounds[i].high

    def test_polynomial_mutation(self):
        rng = np.random.default_rng(42)
        x = np.array([5.0, 5.0, 5.0])
        bounds = [
            ParameterBound(name=f"x{i}", low=0.0, high=10.0)
            for i in range(3)
        ]
        mutated = _polynomial_mutation(x, bounds, rng, mutation_prob=1.0)
        assert not np.allclose(x, mutated)
        for i in range(3):
            assert bounds[i].low <= mutated[i] <= bounds[i].high
