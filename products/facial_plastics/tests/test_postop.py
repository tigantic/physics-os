"""Tests for post-operative outcome loop: ingest, alignment, calibration, validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from products.facial_plastics.core.types import (
    ClinicalMeasurement,
    LandmarkType,
    SurfaceMesh,
    Vec3,
)
from products.facial_plastics.postop.outcome_ingest import (
    OutcomeIngester,
    OutcomeRecord,
    OutcomeTimepoint,
    PatientReportedOutcome,
)
from products.facial_plastics.postop.alignment import (
    AlignmentResult,
    OutcomeAligner,
    RegionalDeviation,
    NASAL_REGIONS,
)
from products.facial_plastics.postop.calibration import (
    CalibrationResult,
    ModelCalibrator,
    ParameterPrior,
)
from products.facial_plastics.postop.validation import (
    AccuracyProfile,
    BlandAltmanResult,
    MetricComparison,
    PredictionValidator,
    ValidationReport,
    _normal_cdf,
    _rankdata,
)
from products.facial_plastics.tests.conftest import (
    make_nose_surface_mesh,
    make_rhinoplasty_landmarks,
)


# ── Outcome Ingest ───────────────────────────────────────────────

class TestOutcomeTimepoint:
    """Test timepoint constants."""

    def test_timepoints(self) -> None:
        assert OutcomeTimepoint.WEEK_1 == "1_week"
        assert len(OutcomeTimepoint.ALL) >= 5


class TestPatientReportedOutcome:
    """Test PRO scoring."""

    def test_normalized_score(self) -> None:
        pro = PatientReportedOutcome(
            instrument="NOSE",
            score=40.0,
            max_score=100.0,
            timepoint=OutcomeTimepoint.MONTH_3,
        )
        assert abs(pro.normalized_score - 40.0) < 1e-10

    def test_zero_max_score(self) -> None:
        pro = PatientReportedOutcome(
            instrument="NOSE",
            score=0.0,
            max_score=0.0,
            timepoint=OutcomeTimepoint.MONTH_1,
        )
        assert pro.normalized_score == 0.0


class TestOutcomeRecord:
    """Test outcome record serialization."""

    def test_to_dict(self) -> None:
        record = OutcomeRecord(
            case_id="case001",
            timepoint=OutcomeTimepoint.MONTH_3,
            complications=["minor_ecchymosis"],
        )
        d = record.to_dict()
        assert d["case_id"] == "case001"
        assert d["timepoint"] == "3_months"
        assert "minor_ecchymosis" in d["complications"]


class TestOutcomeIngester:
    """Test outcome data ingestion."""

    def test_record_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            case_dir = Path(td) / "case001"
            case_dir.mkdir()
            ingester = OutcomeIngester(case_dir)

            measurements = [
                ClinicalMeasurement(name="nasolabial_angle", value=98.0, unit="degrees"),
                ClinicalMeasurement(name="tip_projection", value=28.5, unit="mm"),
            ]
            record = ingester.record_measurements(measurements, OutcomeTimepoint.MONTH_3)
            assert len(record.measurements) == 2

    def test_record_landmarks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            case_dir = Path(td) / "case002"
            case_dir.mkdir()
            ingester = OutcomeIngester(case_dir)

            landmarks = {
                LandmarkType.PRONASALE: Vec3(0.0, 5.0, 35.0),
                LandmarkType.SUBNASALE: Vec3(0.0, 0.0, 25.0),
            }
            record = ingester.record_landmarks(landmarks, OutcomeTimepoint.MONTH_6)
            assert len(record.landmarks) == 2

    def test_record_pro(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            case_dir = Path(td) / "case003"
            case_dir.mkdir()
            ingester = OutcomeIngester(case_dir)

            record = ingester.record_pro(
                instrument="NOSE",
                score=35.0,
                max_score=100.0,
                timepoint=OutcomeTimepoint.MONTH_12,
            )
            assert len(record.pro_scores) == 1
            assert record.pro_scores[0].instrument == "NOSE"

    def test_list_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            case_dir = Path(td) / "case004"
            case_dir.mkdir()
            ingester = OutcomeIngester(case_dir)

            ingester.record_pro("NOSE", 40.0, 100.0, OutcomeTimepoint.MONTH_1)
            ingester.record_pro("NOSE", 25.0, 100.0, OutcomeTimepoint.MONTH_6)

            outcomes = ingester.list_outcomes()
            assert len(outcomes) == 2


# ── Alignment ────────────────────────────────────────────────────

class TestAlignmentResult:
    """Test alignment result properties."""

    def test_clinical_acceptable(self) -> None:
        result = AlignmentResult(case_id="c1", timepoint="3_months")
        result.rms_distance_mm = 1.5
        assert result.clinical_acceptable is True

        result.rms_distance_mm = 2.5
        assert result.clinical_acceptable is False

    def test_to_dict(self) -> None:
        result = AlignmentResult(
            case_id="c1",
            timepoint="3_months",
            rms_distance_mm=1.2,
            hausdorff_distance_mm=3.5,
        )
        d = result.to_dict()
        assert d["rms_distance_mm"] == 1.2


class TestOutcomeAligner:
    """Test mesh alignment pipeline."""

    def test_icp_identity(self) -> None:
        """Aligning a mesh to itself should give near-zero error."""
        mesh = make_nose_surface_mesh(n_verts=100)
        aligner = OutcomeAligner(max_icp_iterations=50)

        result = aligner.align(
            predicted_mesh=mesh,
            actual_mesh=mesh,
            case_id="test",
            timepoint="3_months",
        )
        assert result.rms_distance_mm < 0.1
        assert result.icp_converged is True

    def test_icp_with_translation(self) -> None:
        """Aligning a translated mesh should recover the translation."""
        mesh_a = make_nose_surface_mesh(n_verts=100)
        translated_verts = mesh_a.vertices + np.array([2.0, 1.0, 0.5])
        mesh_b = SurfaceMesh(
            vertices=translated_verts,
            triangles=mesh_a.triangles.copy(),
        )

        aligner = OutcomeAligner(max_icp_iterations=100)
        result = aligner.align(
            predicted_mesh=mesh_a,
            actual_mesh=mesh_b,
            case_id="test",
            timepoint="3_months",
        )
        assert result.rms_distance_mm < 2.0

    def test_landmark_alignment(self) -> None:
        """Test landmark-based initial alignment."""
        mesh = make_nose_surface_mesh(n_verts=100)

        landmarks_pred = {
            LandmarkType.PRONASALE: Vec3(0.0, 5.0, 35.0),
            LandmarkType.NASION: Vec3(0.0, 35.0, 15.0),
            LandmarkType.SUBNASALE: Vec3(0.0, 0.0, 25.0),
        }
        landmarks_actual = {
            LandmarkType.PRONASALE: Vec3(1.0, 6.0, 35.0),
            LandmarkType.NASION: Vec3(1.0, 36.0, 15.0),
            LandmarkType.SUBNASALE: Vec3(1.0, 1.0, 25.0),
        }

        aligner = OutcomeAligner()
        result = aligner.align(
            predicted_mesh=mesh,
            actual_mesh=mesh,
            case_id="test",
            timepoint="3_months",
            predicted_landmarks=landmarks_pred,
            actual_landmarks=landmarks_actual,
        )
        assert isinstance(result, AlignmentResult)

    def test_regional_deviation(self) -> None:
        """Test that regional analysis produces results."""
        mesh = make_nose_surface_mesh(n_verts=100)
        landmarks = make_rhinoplasty_landmarks()

        aligner = OutcomeAligner()
        result = aligner.align(
            predicted_mesh=mesh,
            actual_mesh=mesh,
            case_id="test",
            timepoint="3_months",
            predicted_landmarks=landmarks,
            actual_landmarks=landmarks,
        )
        assert isinstance(result.regional_deviations, list)


# ── Calibration ──────────────────────────────────────────────────

class TestParameterPrior:
    """Test parameter prior."""

    def test_log_prior_at_mean(self) -> None:
        p = ParameterPrior(
            name="test", structure=None,  # type: ignore[arg-type]
            mean=50.0, std=10.0,
            lower_bound=0.0, upper_bound=100.0,
        )
        assert abs(p.log_prior(50.0)) < 1e-10

    def test_log_prior_out_of_bounds(self) -> None:
        p = ParameterPrior(
            name="test", structure=None,  # type: ignore[arg-type]
            mean=50.0, std=10.0,
            lower_bound=20.0, upper_bound=80.0,
        )
        assert p.log_prior(10.0) == float("-inf")
        assert p.log_prior(90.0) == float("-inf")


class TestModelCalibrator:
    """Test model calibration machinery."""

    def test_calibration_with_simple_model(self) -> None:
        """Test calibration with a simple linear simulator."""

        def simple_simulator(**params: float) -> dict[str, np.ndarray]:
            E = params.get("E", 50.0)
            nu = params.get("nu", 0.45)
            return {
                "pt_A": np.array([0.1 * E, 0.05 * E, -0.02 * nu]),
                "pt_B": np.array([-0.03 * E, 0.08 * E, 0.01 * nu]),
            }

        actual = simple_simulator(E=55.0, nu=0.42)

        calibrator = ModelCalibrator(
            simulator=simple_simulator,
            priors={
                "E": ParameterPrior("E", None, 50.0, 15.0, 10.0, 150.0),  # type: ignore[arg-type]
                "nu": ParameterPrior("nu", None, 0.45, 0.03, 0.3, 0.499),  # type: ignore[arg-type]
            },
            max_iterations=20,
            convergence_tol=1e-6,
        )

        result = calibrator.calibrate(
            actual_displacements=actual,
            initial_parameters={"E": 50.0, "nu": 0.45},
            case_ids=["test_case"],
        )

        assert isinstance(result, CalibrationResult)
        assert result.residual_after <= result.residual_before
        assert result.n_iterations > 0
        assert abs(result.parameters_after["E"] - 55.0) < 5.0

    def test_no_simulator_raises(self) -> None:
        calibrator = ModelCalibrator(simulator=None)
        with pytest.raises(ValueError, match="simulator"):
            calibrator.calibrate({}, {})

    def test_save_load_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = CalibrationResult(
                case_ids=["c1"],
                parameters_before={"E": 50.0},
                parameters_after={"E": 55.0},
                residual_before=10.0,
                residual_after=1.0,
                n_iterations=5,
                converged=True,
                improvement_pct=90.0,
            )
            path = Path(td) / "cal.json"
            calibrator = ModelCalibrator()
            calibrator.save_calibration(result, path)
            loaded = ModelCalibrator.load_calibration(path)
            assert loaded["parameters_after"]["E"] == 55.0


# ── Validation ───────────────────────────────────────────────────

class TestMetricComparison:
    """Test per-metric comparison."""

    def test_error_computation(self) -> None:
        c = MetricComparison(
            metric_name="tip_projection",
            unit="mm",
            predicted=29.0,
            actual=28.0,
        )
        assert abs(c.error - 1.0) < 1e-10
        assert abs(c.abs_error - 1.0) < 1e-10
        assert abs(c.pct_error - (1.0 / 28.0 * 100)) < 0.1


class TestStatisticalHelpers:
    """Test statistical utility functions."""

    def test_normal_cdf_symmetry(self) -> None:
        assert abs(_normal_cdf(0.0) - 0.5) < 0.01
        assert _normal_cdf(3.0) > 0.99
        assert _normal_cdf(-3.0) < 0.01

    def test_rankdata(self) -> None:
        x = np.array([3.0, 1.0, 2.0])
        ranks = _rankdata(x)
        assert ranks[0] == 3.0
        assert ranks[1] == 1.0
        assert ranks[2] == 2.0

    def test_rankdata_ties(self) -> None:
        x = np.array([1.0, 1.0, 3.0])
        ranks = _rankdata(x)
        assert ranks[0] == 1.5
        assert ranks[1] == 1.5
        assert ranks[2] == 3.0


class TestPredictionValidator:
    """Test prediction validation pipeline."""

    def test_add_cases(self) -> None:
        validator = PredictionValidator()
        validator.add_case(
            "case001",
            {"tip_proj": (29.0, "mm"), "angle": (100.0, "deg")},
            {"tip_proj": (28.0, "mm"), "angle": (98.0, "deg")},
        )
        assert validator.n_cases == 1

    def test_generate_report_basic(self) -> None:
        validator = PredictionValidator()
        rng = np.random.default_rng(42)
        for i in range(10):
            actual_tip = 28.0 + rng.normal(0, 1)
            pred_tip = actual_tip + rng.normal(0, 0.5)
            validator.add_case(
                f"case_{i:03d}",
                {"tip_projection": (pred_tip, "mm")},
                {"tip_projection": (actual_tip, "mm")},
            )

        report = validator.generate_report()
        assert report.n_cases == 10
        assert "tip_projection" in report.mae
        assert report.mae["tip_projection"] > 0

    def test_bland_altman(self) -> None:
        validator = PredictionValidator()
        rng = np.random.default_rng(42)
        for i in range(20):
            a = 100 + rng.normal(0, 5)
            p = a + rng.normal(0.5, 1.0)
            validator.add_case(f"c{i}", {"angle": (p, "deg")}, {"angle": (a, "deg")})

        report = validator.generate_report()
        assert len(report.bland_altman) == 1
        ba = report.bland_altman[0]
        assert ba.bias > 0
        assert ba.upper_loa > ba.lower_loa

    def test_correlation(self) -> None:
        validator = PredictionValidator()
        rng = np.random.default_rng(42)
        for i in range(30):
            a = 25 + rng.normal(0, 3)
            p = a + rng.normal(0, 0.3)
            validator.add_case(f"c{i}", {"dist": (p, "mm")}, {"dist": (a, "mm")})

        report = validator.generate_report()
        assert len(report.correlations) == 1
        corr = report.correlations[0]
        assert corr.pearson_r > 0.9

    def test_accuracy_profile(self) -> None:
        validator = PredictionValidator()
        rng = np.random.default_rng(42)
        for i in range(50):
            a = 30 + rng.normal(0, 2)
            p = a + rng.normal(0, 0.5)
            validator.add_case(f"c{i}", {"m": (p, "mm")}, {"m": (a, "mm")})

        report = validator.generate_report()
        assert len(report.accuracy_profiles) == 1
        profile = report.accuracy_profiles[0]
        assert profile.pct_within_2mm > 50

    def test_grade_assignment(self) -> None:
        validator = PredictionValidator()
        rng = np.random.default_rng(42)
        for i in range(20):
            a = 10.0
            p = a + rng.normal(0, 0.3)
            validator.add_case(f"c{i}", {"x": (p, "mm")}, {"x": (a, "mm")})

        report = validator.generate_report()
        assert report.overall_grade == "A"

    def test_save_load_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            validator = PredictionValidator()
            validator.add_case("c1", {"x": (10.0, "mm")}, {"x": (10.5, "mm")})
            validator.add_case("c2", {"x": (11.0, "mm")}, {"x": (10.8, "mm")})
            validator.add_case("c3", {"x": (9.0, "mm")}, {"x": (9.2, "mm")})
            report = validator.generate_report()

            path = Path(td) / "report.json"
            validator.save_report(report, path)
            loaded = PredictionValidator.load_report(path)
            assert loaded["n_cases"] == 3
