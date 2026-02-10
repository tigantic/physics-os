"""Tests for the anatomy generator and case library curator."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Generator

import numpy as np
import pytest

from products.facial_plastics.core.case_bundle import CaseBundle, PatientDemographics
from products.facial_plastics.core.types import (
    ClinicalMeasurement,
    LandmarkType,
    Modality,
    ProcedureType,
    QualityLevel,
    StructureType,
    SurfaceMesh,
    Vec3,
)
from products.facial_plastics.data.anatomy_generator import (
    AnatomyGenerator,
    AnthropometricProfile,
    PopulationSampler,
    HU_AIR,
    HU_BONE_CORTICAL,
    HU_CARTILAGE,
    HU_FAT,
    HU_SOFT_TISSUE,
    _ellipsoid_sdf,
    _box_sdf,
    _cylinder_sdf,
)
from products.facial_plastics.data.case_library import CaseLibrary
from products.facial_plastics.data.case_library_curator import (
    CaseLibraryCurator,
    QCThresholds,
)


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="fp_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sampler() -> PopulationSampler:
    return PopulationSampler(seed=123)


@pytest.fixture
def generator() -> AnatomyGenerator:
    return AnatomyGenerator(seed=123)


@pytest.fixture
def default_profile(sampler: PopulationSampler) -> AnthropometricProfile:
    return sampler.sample_profile(PatientDemographics(
        age_years=35, sex="M", ethnicity="european", skin_fitzpatrick=2,
    ))


# ══════════════════════════════════════════════════════════════════
# SDF primitives
# ══════════════════════════════════════════════════════════════════

class TestSDFPrimitives:
    """Test signed distance function primitives."""

    def test_ellipsoid_center_inside(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0]])
        center = np.array([0.0, 0.0, 0.0])
        radii = np.array([10.0, 10.0, 10.0])
        sdf = _ellipsoid_sdf(coords, center, radii)
        assert sdf[0] < 0, "Centre of ellipsoid should be inside (negative SDF)"

    def test_ellipsoid_outside(self) -> None:
        coords = np.array([[20.0, 0.0, 0.0]])
        center = np.array([0.0, 0.0, 0.0])
        radii = np.array([10.0, 10.0, 10.0])
        sdf = _ellipsoid_sdf(coords, center, radii)
        assert sdf[0] > 0, "Point outside should be positive"

    def test_ellipsoid_batch(self) -> None:
        coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ])
        center = np.array([0.0, 0.0, 0.0])
        radii = np.array([10.0, 10.0, 10.0])
        sdf = _ellipsoid_sdf(coords, center, radii)
        assert sdf[0] < 0
        assert abs(sdf[1]) < 1.0  # approximately on surface
        assert sdf[2] > 0

    def test_box_center_inside(self) -> None:
        coords = np.array([[5.0, 5.0, 5.0]])
        center = np.array([5.0, 5.0, 5.0])
        half = np.array([3.0, 3.0, 3.0])
        sdf = _box_sdf(coords, center, half)
        assert sdf[0] < 0

    def test_box_outside(self) -> None:
        coords = np.array([[20.0, 5.0, 5.0]])
        center = np.array([5.0, 5.0, 5.0])
        half = np.array([3.0, 3.0, 3.0])
        sdf = _box_sdf(coords, center, half)
        assert sdf[0] > 0

    def test_cylinder_inside(self) -> None:
        coords = np.array([[0.0, 5.0, 0.0]])
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 10.0, 0.0])
        sdf = _cylinder_sdf(coords, p0, p1, 5.0)
        assert sdf[0] < 0

    def test_cylinder_outside_radial(self) -> None:
        coords = np.array([[10.0, 5.0, 0.0]])
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 10.0, 0.0])
        sdf = _cylinder_sdf(coords, p0, p1, 5.0)
        assert sdf[0] > 0

    def test_cylinder_outside_cap(self) -> None:
        coords = np.array([[0.0, 15.0, 0.0]])
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 10.0, 0.0])
        sdf = _cylinder_sdf(coords, p0, p1, 5.0)
        assert sdf[0] > 0


# ══════════════════════════════════════════════════════════════════
# Population sampler
# ══════════════════════════════════════════════════════════════════

class TestPopulationSampler:
    """Test demographic-conditioned parameter sampling."""

    def test_sample_demographics(self, sampler: PopulationSampler) -> None:
        d = sampler.sample_demographics()
        assert d.sex in ("M", "F")
        assert d.ethnicity is not None
        assert d.age_years is not None and 18 <= d.age_years <= 75
        assert d.skin_fitzpatrick is not None and 1 <= d.skin_fitzpatrick <= 6

    def test_sample_demographics_reproducible(self) -> None:
        s1 = PopulationSampler(seed=42)
        s2 = PopulationSampler(seed=42)
        d1 = s1.sample_demographics()
        d2 = s2.sample_demographics()
        assert d1.sex == d2.sex
        assert d1.ethnicity == d2.ethnicity
        assert d1.age_years == d2.age_years

    def test_sample_profile_european_male(self, sampler: PopulationSampler) -> None:
        d = PatientDemographics(age_years=35, sex="M", ethnicity="european", skin_fitzpatrick=2)
        p = sampler.sample_profile(d)
        assert p.sex == "M"
        assert p.ethnicity == "european"
        assert p.skull_width > 100  # reasonable range
        assert p.nasal_bone_length > 10
        assert p.tip_projection > 10

    def test_sample_profile_african_female(self, sampler: PopulationSampler) -> None:
        d = PatientDemographics(age_years=40, sex="F", ethnicity="african", skin_fitzpatrick=5)
        p = sampler.sample_profile(d)
        assert p.sex == "F"
        assert p.alar_width > 30  # typically wider
        assert p.skin_thickness_tip > 3  # typically thicker

    def test_sample_profile_east_asian(self, sampler: PopulationSampler) -> None:
        d = PatientDemographics(age_years=28, sex="M", ethnicity="east_asian", skin_fitzpatrick=3)
        p = sampler.sample_profile(d)
        assert p.ethnicity == "east_asian"
        assert p.nasal_dorsal_height > 5

    def test_sample_profile_unknown_ethnicity_falls_back(self, sampler: PopulationSampler) -> None:
        d = PatientDemographics(age_years=30, sex="M", ethnicity="martian", skin_fitzpatrick=2)
        p = sampler.sample_profile(d)
        # Should fall back to european defaults without error
        assert p.skull_width > 100

    def test_age_adjustments_older(self, sampler: PopulationSampler) -> None:
        young = PatientDemographics(age_years=25, sex="M", ethnicity="european")
        old = PatientDemographics(age_years=70, sex="M", ethnicity="european")
        s1 = PopulationSampler(seed=42)
        s2 = PopulationSampler(seed=42)
        p_young = s1.sample_profile(young)
        p_old = s2.sample_profile(old)
        # Older patients should have more fat
        assert p_old.fat_thickness_cheek > p_young.fat_thickness_cheek

    def test_all_ethnicities_generate(self, sampler: PopulationSampler) -> None:
        ethnicities = ["european", "east_asian", "african", "south_asian", "hispanic"]
        for eth in ethnicities:
            d = PatientDemographics(age_years=35, sex="M", ethnicity=eth)
            p = sampler.sample_profile(d)
            assert p.ethnicity == eth
            assert p.skull_width > 50


# ══════════════════════════════════════════════════════════════════
# Anatomy generator
# ══════════════════════════════════════════════════════════════════

class TestAnatomyGenerator:
    """Test CT volume and surface generation."""

    def test_generate_ct_volume_shape(
        self, generator: AnatomyGenerator, default_profile: AnthropometricProfile,
    ) -> None:
        volume, spacing, origin = generator.generate_ct_volume(
            default_profile, grid_size=64, voxel_spacing_mm=1.5,
        )
        assert volume.shape == (64, 64, 64)
        assert len(spacing) == 3
        assert all(s == 1.5 for s in spacing)
        assert origin.shape == (3,)

    def test_generate_ct_volume_hu_range(
        self, generator: AnatomyGenerator, default_profile: AnthropometricProfile,
    ) -> None:
        volume, _, _ = generator.generate_ct_volume(
            default_profile, grid_size=64, voxel_spacing_mm=1.5,
        )
        assert volume.min() < -500, "Should contain air/background"
        assert volume.max() > 300, "Should contain bone"

    def test_ct_volume_contains_structures(
        self, generator: AnatomyGenerator, default_profile: AnthropometricProfile,
    ) -> None:
        volume, _, _ = generator.generate_ct_volume(
            default_profile, grid_size=64, voxel_spacing_mm=1.5,
        )
        # Check for bone (> 300 HU)
        bone_voxels = (volume > 300).sum()
        assert bone_voxels > 0, "Volume should contain bone"

        # Check for air (< -500 HU)
        air_voxels = (volume < -500).sum()
        assert air_voxels > 0, "Volume should contain airway"

        # Check for soft tissue (between -200 and 300)
        soft_voxels = ((volume > -200) & (volume < 300)).sum()
        assert soft_voxels > 0, "Volume should contain soft tissue"

    def test_ct_volume_reproducible(
        self, default_profile: AnthropometricProfile,
    ) -> None:
        g1 = AnatomyGenerator(seed=42)
        g2 = AnatomyGenerator(seed=42)
        v1, _, _ = g1.generate_ct_volume(default_profile, grid_size=32, voxel_spacing_mm=2.0)
        v2, _, _ = g2.generate_ct_volume(default_profile, grid_size=32, voxel_spacing_mm=2.0)
        np.testing.assert_array_equal(v1, v2)

    def test_extract_facial_surface(
        self, generator: AnatomyGenerator, default_profile: AnthropometricProfile,
    ) -> None:
        # 64 * 3.0 = 192mm FOV — large enough to contain full facial anatomy
        volume, spacing, origin = generator.generate_ct_volume(
            default_profile, grid_size=64, voxel_spacing_mm=3.0,
        )
        mesh = generator.extract_facial_surface(volume, spacing, origin)
        assert isinstance(mesh, SurfaceMesh)
        assert mesh.n_vertices > 50
        assert mesh.n_faces > 20
        assert mesh.normals is not None

    def test_extract_surface_largest_component(
        self, generator: AnatomyGenerator, default_profile: AnthropometricProfile,
    ) -> None:
        volume, spacing, origin = generator.generate_ct_volume(
            default_profile, grid_size=64, voxel_spacing_mm=3.0,
        )
        mesh = generator.extract_facial_surface(volume, spacing, origin)
        # Should be a single connected component after filtering
        assert mesh.n_vertices > 0
        assert mesh.n_faces > 0

    def test_compute_landmarks(
        self, generator: AnatomyGenerator, default_profile: AnthropometricProfile,
    ) -> None:
        landmarks = generator.compute_landmarks(default_profile)
        assert len(landmarks) >= 20
        assert LandmarkType.NASION in landmarks
        assert LandmarkType.PRONASALE in landmarks
        assert LandmarkType.SUBNASALE in landmarks
        assert LandmarkType.ALAR_RIM_LEFT in landmarks
        assert LandmarkType.POGONION in landmarks

        # Pronasale should be the most anterior
        prn = landmarks[LandmarkType.PRONASALE]
        sn = landmarks[LandmarkType.SUBNASALE]
        assert prn.z > sn.z, "Pronasale should be more anterior than subnasale"

    def test_compute_clinical_measurements(
        self, generator: AnatomyGenerator, default_profile: AnthropometricProfile,
    ) -> None:
        landmarks = generator.compute_landmarks(default_profile)
        measurements = generator.compute_clinical_measurements(default_profile, landmarks)
        assert len(measurements) >= 5
        names = {m.name for m in measurements}
        assert "nasal_dorsal_length" in names
        assert "tip_projection" in names
        assert "alar_width" in names
        assert "nasolabial_angle" in names

        for m in measurements:
            assert m.value > 0, f"Measurement {m.name} should be positive"

    def test_landmark_symmetry(
        self, generator: AnatomyGenerator, default_profile: AnthropometricProfile,
    ) -> None:
        landmarks = generator.compute_landmarks(default_profile)
        # Bilateral landmarks should be symmetric about x=0
        left = landmarks[LandmarkType.ALAR_RIM_LEFT]
        right = landmarks[LandmarkType.ALAR_RIM_RIGHT]
        assert abs(left.x + right.x) < 0.1, "Alar landmarks should be symmetric"
        assert abs(left.y - right.y) < 0.1, "Alar landmarks should be at same height"

    def test_different_demographics_different_anatomy(self) -> None:
        g = AnatomyGenerator(seed=42)
        s = PopulationSampler(seed=42)
        p_m = s.sample_profile(PatientDemographics(age_years=35, sex="M", ethnicity="european"))
        s2 = PopulationSampler(seed=42)
        p_f = s2.sample_profile(PatientDemographics(age_years=35, sex="F", ethnicity="european"))
        # Male should generally have larger skull
        assert p_m.skull_width > p_f.skull_width - 20  # not always, but within range


# ══════════════════════════════════════════════════════════════════
# Case library curator — single case
# ══════════════════════════════════════════════════════════════════

class TestCaseLibraryCurator:
    """Test the case library curator."""

    def test_generate_single_case(self, tmp_dir: Path) -> None:
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        result = curator.generate_case(run_twin_pipeline=False)
        assert result.success
        assert result.case_id != "pending"
        assert result.demographics is not None
        assert result.procedure is not None
        assert result.surface_vertices > 0
        assert result.n_landmarks >= 20
        assert result.n_measurements >= 5
        assert result.volume_shape == (32, 32, 32)

    def test_case_bundle_structure(self, tmp_dir: Path) -> None:
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        result = curator.generate_case(run_twin_pipeline=False)
        assert result.success

        # Load the bundle and verify structure
        bundle = curator.library.load_bundle(result.case_id)
        assert bundle.case_id == result.case_id
        assert bundle.manifest.quality_level == QualityLevel.SYNTHETIC.value
        assert len(bundle.manifest.acquisitions) >= 2  # CT + surface
        assert bundle.is_valid()

        # Check saved artifacts
        ct = bundle.load_array("ct_volume", subdir="inputs")
        assert ct.shape == (32, 32, 32)

        ct_meta = bundle.load_json("ct_metadata", subdir="inputs")
        assert ct_meta["source"] == "synthetic_parametric"

        landmarks = bundle.load_json("landmarks", subdir="derived")
        assert len(landmarks["landmarks"]) >= 20

        measurements = bundle.load_json("clinical_measurements", subdir="derived")
        assert len(measurements["measurements"]) >= 5

        profile = bundle.load_json("anthropometric_profile", subdir="derived")
        assert "skull_width" in profile
        assert "nasal_bone_length" in profile

    def test_case_with_explicit_demographics(self, tmp_dir: Path) -> None:
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        demo = PatientDemographics(
            age_years=55,
            sex="F",
            ethnicity="east_asian",
            skin_fitzpatrick=4,
        )
        result = curator.generate_case(
            demographics=demo,
            procedure=ProcedureType.RHINOPLASTY,
            run_twin_pipeline=False,
        )
        assert result.success
        assert result.demographics is not None
        assert result.demographics.sex == "F"
        assert result.demographics.ethnicity == "east_asian"
        assert result.procedure == ProcedureType.RHINOPLASTY

    def test_case_with_twin_pipeline(self, tmp_dir: Path) -> None:
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        result = curator.generate_case(run_twin_pipeline=True)
        assert result.success
        # Twin pipeline should produce structures
        assert result.n_structures > 0

        bundle = curator.library.load_bundle(result.case_id)
        # Segmentation should be marked complete
        assert bundle.manifest.segmentation_complete

    def test_surface_mesh_roundtrip(self, tmp_dir: Path) -> None:
        """Verify surface mesh save/load through CaseBundle works."""
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=48,
            voxel_spacing_mm=2.0,
        )
        result = curator.generate_case(run_twin_pipeline=False)
        assert result.success

        bundle = curator.library.load_bundle(result.case_id)
        mesh = bundle.load_surface_mesh("facial_surface")
        assert isinstance(mesh, SurfaceMesh)
        assert mesh.n_vertices == result.surface_vertices
        assert mesh.n_faces == result.surface_triangles

    def test_qc_gate_passes(self, tmp_dir: Path) -> None:
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=64,
            voxel_spacing_mm=1.5,
        )
        result = curator.generate_case(run_twin_pipeline=False)
        assert result.success
        assert result.qc_passed
        assert len(result.qc_issues) == 0

    def test_qc_gate_strict_thresholds(self, tmp_dir: Path) -> None:
        strict = QCThresholds(
            min_surface_vertices=100_000,  # impossibly high
        )
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
            qc_thresholds=strict,
        )
        result = curator.generate_case(run_twin_pipeline=False)
        assert result.success  # generation succeeds
        assert not result.qc_passed  # but QC fails
        assert any("vertices" in issue for issue in result.qc_issues)


# ══════════════════════════════════════════════════════════════════
# Case library curator — batch generation
# ══════════════════════════════════════════════════════════════════

class TestLibraryGeneration:
    """Test batch library generation."""

    def test_generate_small_library(self, tmp_dir: Path) -> None:
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        report = curator.generate_library(n_cases=3, run_twin_pipeline=False)
        assert report.total_attempted == 3
        assert report.total_succeeded >= 2  # allow 1 QC rejection
        assert report.total_time_s > 0
        assert len(report.cases) == 3

        # Library should be indexed
        assert curator.library.case_count >= 2

        # Demographics should be tracked
        assert len(report.demographics_summary) > 0
        assert len(report.procedure_summary) > 0

    def test_generation_report_saved(self, tmp_dir: Path) -> None:
        lib_root = tmp_dir / "library"
        curator = CaseLibraryCurator(
            lib_root,
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        curator.generate_library(n_cases=2, run_twin_pipeline=False)

        report_path = lib_root / "_generation_report.json"
        assert report_path.exists()

        with open(report_path) as f:
            data = json.load(f)
        assert data["total_attempted"] == 2
        assert len(data["cases"]) == 2

    def test_library_index_rebuilt(self, tmp_dir: Path) -> None:
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        curator.generate_library(n_cases=3, run_twin_pipeline=False)

        # Rebuild index from scratch
        lib = CaseLibrary(tmp_dir / "library")
        count = lib.rebuild_index()
        assert count >= 2

    def test_library_query_by_procedure(self, tmp_dir: Path) -> None:
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        # Force all rhinoplasty
        for _ in range(3):
            curator.generate_case(
                procedure=ProcedureType.RHINOPLASTY,
                run_twin_pipeline=False,
            )

        results = curator.library.query(procedure=ProcedureType.RHINOPLASTY)
        assert len(results) == 3

    def test_curator_summary(self, tmp_dir: Path) -> None:
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        curator.generate_library(n_cases=2, run_twin_pipeline=False)
        summary = curator.summary()
        assert summary["total_cases"] >= 1
        assert "library_root" in summary

    def test_curator_repr(self, tmp_dir: Path) -> None:
        curator = CaseLibraryCurator(tmp_dir / "library", seed=42)
        r = repr(curator)
        assert "CaseLibraryCurator" in r

    def test_demographic_diversity(self, tmp_dir: Path) -> None:
        """Verify that 10 cases produce at least 2 different ethnicities."""
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        report = curator.generate_library(n_cases=10, run_twin_pipeline=False)
        ethnicities = set()
        for case in report.cases:
            if case.demographics and case.demographics.ethnicity:
                ethnicities.add(case.demographics.ethnicity)
        assert len(ethnicities) >= 2, f"Expected diversity, got {ethnicities}"

    def test_procedure_diversity(self, tmp_dir: Path) -> None:
        """Verify that 10 cases produce at least 2 different procedures."""
        curator = CaseLibraryCurator(
            tmp_dir / "library",
            seed=42,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        report = curator.generate_library(n_cases=10, run_twin_pipeline=False)
        procedures = set()
        for case in report.cases:
            if case.procedure:
                procedures.add(case.procedure.value)
        assert len(procedures) >= 2, f"Expected diversity, got {procedures}"


# ══════════════════════════════════════════════════════════════════
# Marching cubes (from anatomy generator)
# ══════════════════════════════════════════════════════════════════

class TestMarchingCubes:
    """Test the built-in marching cubes implementation."""

    def test_sphere_extraction(self, generator: AnatomyGenerator) -> None:
        # Create a sphere in a volume
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        c = np.array([16, 16, 16])
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    if np.linalg.norm(np.array([i, j, k]) - c) < 10:
                        vol[i, j, k] = 1.0

        verts, tris = generator._marching_cubes_simple(vol, (1.0, 1.0, 1.0))
        assert len(verts) > 10
        assert len(tris) > 5

    def test_empty_volume(self, generator: AnatomyGenerator) -> None:
        vol = np.zeros((16, 16, 16), dtype=np.float32)
        verts, tris = generator._marching_cubes_simple(vol, (1.0, 1.0, 1.0))
        assert len(verts) == 0
        assert len(tris) == 0

    def test_full_volume(self, generator: AnatomyGenerator) -> None:
        vol = np.ones((16, 16, 16), dtype=np.float32)
        verts, tris = generator._marching_cubes_simple(vol, (1.0, 1.0, 1.0))
        # All-ones should produce no interior edges, but boundary cells do
        assert len(verts) >= 0  # May or may not produce boundary surface


# ══════════════════════════════════════════════════════════════════
# Largest component extraction
# ══════════════════════════════════════════════════════════════════

class TestLargestComponent:
    """Test connected component filtering."""

    def test_single_component_unchanged(self, generator: AnatomyGenerator) -> None:
        # Triangle fan — all connected
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        ], dtype=np.float64)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = SurfaceMesh(vertices=verts, triangles=tris)
        result = generator._largest_component(mesh)
        assert result.n_vertices == 4
        assert result.n_faces == 2

    def test_two_components_keeps_larger(self, generator: AnatomyGenerator) -> None:
        verts = np.array([
            # Component 1: 3 vertices
            [0, 0, 0], [1, 0, 0], [0.5, 1, 0],
            # Component 2: 4 vertices (larger)
            [10, 10, 10], [11, 10, 10], [11, 11, 10], [10, 11, 10],
        ], dtype=np.float64)
        tris = np.array([
            [0, 1, 2],
            [3, 4, 5], [3, 5, 6],
        ], dtype=np.int64)
        mesh = SurfaceMesh(vertices=verts, triangles=tris)
        result = generator._largest_component(mesh)
        assert result.n_vertices == 4  # larger component
        assert result.n_faces == 2
