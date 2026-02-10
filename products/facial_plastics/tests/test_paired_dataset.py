"""Tests for data/paired_dataset.py — paired CT+surface dataset builder."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from products.facial_plastics.data.paired_dataset import (
    PairedDatasetBuilder,
    PairedQCThresholds,
    PairedSample,
    PairedDatasetReport,
    _add_scan_noise,
    _simulate_partial_coverage,
    _add_scan_holes,
    _extract_gt_surface,
    _compute_landmark_rms,
    _compute_alignment_error,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def default_thresholds() -> PairedQCThresholds:
    return PairedQCThresholds()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def sample_vertices() -> np.ndarray:
    """A grid of 25 vertices in a 5×5 plane (no mesh faces yet)."""
    coords = []
    for i in range(5):
        for j in range(5):
            coords.append([float(i) * 5.0, float(j) * 5.0, 0.0])
    return np.array(coords, dtype=np.float64)


@pytest.fixture
def sample_faces(sample_vertices: np.ndarray) -> np.ndarray:
    """Triangulate the 5×5 grid into faces."""
    faces = []
    for i in range(4):
        for j in range(4):
            v0 = i * 5 + j
            v1 = v0 + 1
            v2 = v0 + 5
            v3 = v0 + 6
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])
    return np.array(faces, dtype=np.int64)


@pytest.fixture
def sample_volume() -> np.ndarray:
    """3D binary volume with a filled sphere in the centre."""
    vol = np.zeros((20, 20, 20), dtype=np.float64)
    centre = np.array([10.0, 10.0, 10.0])
    for z in range(20):
        for y in range(20):
            for x in range(20):
                if np.linalg.norm(np.array([z, y, x], dtype=np.float64) - centre) < 6.0:
                    vol[z, y, x] = 1.0
    return vol


# ── PairedQCThresholds ───────────────────────────────────────────

class TestPairedQCThresholds:
    def test_defaults_positive(self, default_thresholds: PairedQCThresholds) -> None:
        assert default_thresholds.max_landmark_rms_mm > 0
        assert default_thresholds.max_alignment_error_mm > 0
        assert default_thresholds.min_surface_coverage_pct > 0
        assert default_thresholds.min_surface_coverage_pct <= 100.0
        assert default_thresholds.min_surface_vertices > 0
        assert default_thresholds.min_ct_voxels > 0

    def test_custom_thresholds(self) -> None:
        t = PairedQCThresholds(
            max_landmark_rms_mm=0.5,
            max_alignment_error_mm=0.3,
            min_surface_coverage_pct=95.0,
            min_surface_vertices=200,
            min_ct_voxels=500,
        )
        assert t.max_landmark_rms_mm == 0.5
        assert t.max_alignment_error_mm == 0.3
        assert t.min_surface_coverage_pct == 95.0

    def test_frozen(self, default_thresholds: PairedQCThresholds) -> None:
        with pytest.raises(AttributeError):
            default_thresholds.max_landmark_rms_mm = 999.0  # type: ignore[misc]


# ── Noise / artifact functions ────────────────────────────────────

class TestScanArtifacts:
    def test_add_scan_noise_preserves_shape(
        self, sample_vertices: np.ndarray, rng: np.random.Generator,
    ) -> None:
        noisy = _add_scan_noise(sample_vertices.copy(), rng, sigma_mm=0.5)
        assert noisy.shape == sample_vertices.shape

    def test_add_scan_noise_perturbs(
        self, sample_vertices: np.ndarray, rng: np.random.Generator,
    ) -> None:
        noisy = _add_scan_noise(sample_vertices.copy(), rng, sigma_mm=1.0)
        assert not np.allclose(noisy, sample_vertices)

    def test_add_scan_noise_zero_sigma(
        self, sample_vertices: np.ndarray, rng: np.random.Generator,
    ) -> None:
        noisy = _add_scan_noise(sample_vertices.copy(), rng, sigma_mm=0.0)
        np.testing.assert_allclose(noisy, sample_vertices)

    def test_simulate_partial_coverage_reduces_vertices(
        self,
        sample_vertices: np.ndarray,
        sample_faces: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        kept_v, kept_f, coverage = _simulate_partial_coverage(
            sample_vertices, sample_faces, rng, coverage_fraction=0.5,
        )
        assert kept_v.shape[1] == 3
        assert len(kept_v) <= len(sample_vertices)
        assert coverage >= 0.0

    def test_simulate_full_coverage(
        self,
        sample_vertices: np.ndarray,
        sample_faces: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        kept_v, kept_f, coverage = _simulate_partial_coverage(
            sample_vertices, sample_faces, rng, coverage_fraction=1.0,
        )
        assert len(kept_v) == len(sample_vertices)
        assert coverage == pytest.approx(100.0)

    def test_add_scan_holes_returns_tuple(
        self,
        sample_vertices: np.ndarray,
        sample_faces: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        holed_v, holed_f = _add_scan_holes(
            sample_vertices, sample_faces, rng, n_holes=2, hole_radius_mm=3.0,
        )
        assert holed_v.shape[1] == 3
        assert holed_f.ndim == 2

    def test_add_scan_holes_removes_vertices(
        self,
        sample_vertices: np.ndarray,
        sample_faces: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        holed_v, holed_f = _add_scan_holes(
            sample_vertices, sample_faces, rng, n_holes=5, hole_radius_mm=5.0,
        )
        assert len(holed_v) <= len(sample_vertices)

    def test_add_scan_holes_empty_mesh(self, rng: np.random.Generator) -> None:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        empty_f = np.zeros((0, 3), dtype=np.int64)
        result_v, result_f = _add_scan_holes(empty_v, empty_f, rng)
        assert len(result_v) == 0


# ── GT surface extraction ─────────────────────────────────────────

class TestGTSurface:
    def test_extract_gt_surface_from_volume(self, sample_volume: np.ndarray) -> None:
        verts, faces = _extract_gt_surface(sample_volume, spacing_mm=1.0)
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert len(verts) > 0  # sphere boundary should produce vertices

    def test_extract_gt_surface_faces_valid_indices(
        self, sample_volume: np.ndarray,
    ) -> None:
        verts, faces = _extract_gt_surface(sample_volume)
        if len(faces) > 0:
            assert faces.max() < len(verts)
            assert faces.min() >= 0

    def test_extract_gt_surface_empty_volume(self) -> None:
        empty = np.zeros((5, 5, 5), dtype=np.float64)
        verts, faces = _extract_gt_surface(empty)
        assert len(verts) == 0

    def test_extract_gt_surface_solid_volume(self) -> None:
        """A fully solid cube should produce boundary vertices on its surface."""
        solid = np.ones((8, 8, 8), dtype=np.float64)
        verts, faces = _extract_gt_surface(solid)
        # Only boundary voxels should appear
        assert len(verts) > 0


# ── Metric computations ──────────────────────────────────────────

class TestMetrics:
    def test_landmark_rms_identical(self) -> None:
        pts: List[np.ndarray] = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
        ]
        rms = _compute_landmark_rms(pts, [p.copy() for p in pts])
        assert rms == pytest.approx(0.0, abs=1e-10)

    def test_landmark_rms_known_offset(self) -> None:
        gt: List[np.ndarray] = [np.zeros(3, dtype=np.float64) for _ in range(5)]
        reg: List[np.ndarray] = [np.ones(3, dtype=np.float64) for _ in range(5)]
        rms = _compute_landmark_rms(gt, reg)
        # Each point offset by sqrt(3) ≈ 1.732
        expected = np.sqrt(3.0)
        assert rms == pytest.approx(expected, rel=0.01)

    def test_landmark_rms_empty(self) -> None:
        rms = _compute_landmark_rms([], [])
        assert rms == pytest.approx(0.0)

    def test_alignment_error_identical(self, sample_vertices: np.ndarray) -> None:
        err = _compute_alignment_error(sample_vertices, sample_vertices.copy())
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_alignment_error_nonzero(self, sample_vertices: np.ndarray) -> None:
        shifted = sample_vertices + 1.0
        err = _compute_alignment_error(sample_vertices, shifted)
        assert err > 0.0

    def test_alignment_error_empty(self) -> None:
        empty = np.zeros((0, 3), dtype=np.float64)
        err = _compute_alignment_error(empty, empty)
        assert err == float("inf")


# ── PairedSample dataclass ────────────────────────────────────────

class TestPairedSample:
    def test_instantiation(self) -> None:
        s = PairedSample(
            case_id="test_001",
            ct_shape=(128, 128, 128),
            surface_n_vertices=1000,
            surface_n_faces=2000,
            gt_alignment_error_mm=0.3,
            landmark_rms_error_mm=0.5,
            surface_coverage_pct=92.5,
            qc_passed=True,
            generation_time_s=1.2,
        )
        assert s.case_id == "test_001"
        assert s.qc_passed is True
        assert s.landmark_rms_error_mm == 0.5
        assert s.ct_shape == (128, 128, 128)

    def test_to_dict(self) -> None:
        s = PairedSample(
            case_id="d_001",
            ct_shape=(64, 64, 64),
            surface_n_vertices=500,
            surface_n_faces=900,
            gt_alignment_error_mm=1.0,
            landmark_rms_error_mm=1.5,
            surface_coverage_pct=85.0,
            qc_passed=False,
            generation_time_s=0.8,
        )
        d = s.to_dict()
        assert d["case_id"] == "d_001"
        assert d["qc_passed"] is False
        assert d["ct_shape"] == [64, 64, 64]


# ── PairedDatasetReport ──────────────────────────────────────────

class TestPairedDatasetReport:
    def test_report_fields(self) -> None:
        report = PairedDatasetReport(
            n_requested=10,
            n_generated=10,
            n_passed_qc=8,
            mean_alignment_error_mm=0.5,
            mean_landmark_error_mm=0.7,
            mean_coverage_pct=90.0,
            total_time_s=5.0,
        )
        assert report.n_requested == 10
        assert report.n_generated == 10
        assert report.n_passed_qc == 8

    def test_report_pass_rate(self) -> None:
        report = PairedDatasetReport(
            n_requested=10,
            n_generated=10,
            n_passed_qc=8,
            mean_alignment_error_mm=0.5,
            mean_landmark_error_mm=0.7,
            mean_coverage_pct=90.0,
            total_time_s=5.0,
        )
        # pass_rate returns percentage
        assert report.pass_rate == pytest.approx(80.0)

    def test_report_pass_rate_zero_generated(self) -> None:
        report = PairedDatasetReport(
            n_requested=5,
            n_generated=0,
            n_passed_qc=0,
            mean_alignment_error_mm=0.0,
            mean_landmark_error_mm=0.0,
            mean_coverage_pct=0.0,
            total_time_s=0.0,
        )
        # Should not divide by zero
        assert report.pass_rate == pytest.approx(0.0)

    def test_report_to_dict(self) -> None:
        report = PairedDatasetReport(
            n_requested=3,
            n_generated=3,
            n_passed_qc=2,
            mean_alignment_error_mm=0.5,
            mean_landmark_error_mm=0.7,
            mean_coverage_pct=90.0,
            total_time_s=2.0,
        )
        d = report.to_dict()
        assert d["n_requested"] == 3
        assert "pass_rate_pct" in d
        assert "samples" in d

    def test_report_with_samples(self) -> None:
        samples = [
            PairedSample(
                case_id=f"c_{i}",
                ct_shape=(64, 64, 64),
                surface_n_vertices=500,
                surface_n_faces=900,
                gt_alignment_error_mm=0.5,
                landmark_rms_error_mm=0.7,
                surface_coverage_pct=90.0,
                qc_passed=True,
                generation_time_s=1.0,
            )
            for i in range(3)
        ]
        report = PairedDatasetReport(
            n_requested=3,
            n_generated=3,
            n_passed_qc=3,
            mean_alignment_error_mm=0.5,
            mean_landmark_error_mm=0.7,
            mean_coverage_pct=90.0,
            total_time_s=3.0,
            samples=samples,
        )
        assert len(report.samples) == 3
        d = report.to_dict()
        assert len(d["samples"]) == 3
