"""Tests for core types, provenance, config, and CaseBundle."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from products.facial_plastics.core.types import (
    BoundingBox,
    ClinicalMeasurement,
    Landmark,
    LandmarkType,
    MaterialModel,
    MeshElementType,
    Modality,
    ProcedureType,
    QualityLevel,
    StructureType,
    SurfaceMesh,
    TissueProperties,
    Vec3,
    VolumeMesh,
    generate_case_id,
)
from products.facial_plastics.core.provenance import Provenance, hash_bytes, hash_file
from products.facial_plastics.core.config import PlatformConfig
from products.facial_plastics.core.case_bundle import CaseBundle


# ── Vec3 ─────────────────────────────────────────────────────────

class TestVec3:
    def test_construction(self) -> None:
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0

    def test_frozen(self) -> None:
        v = Vec3(1.0, 2.0, 3.0)
        with pytest.raises(AttributeError):
            v.x = 5.0  # type: ignore[misc]

    def test_add(self) -> None:
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        c = a + b
        assert c.x == 5 and c.y == 7 and c.z == 9

    def test_sub(self) -> None:
        a = Vec3(4, 5, 6)
        b = Vec3(1, 2, 3)
        c = a - b
        assert c.x == 3 and c.y == 3 and c.z == 3

    def test_mul(self) -> None:
        v = Vec3(1, 2, 3)
        r = v * 2.0
        assert r.x == 2 and r.y == 4 and r.z == 6

    def test_norm(self) -> None:
        v = Vec3(3, 4, 0)
        assert abs(v.norm() - 5.0) < 1e-10

    def test_dot(self) -> None:
        a = Vec3(1, 0, 0)
        b = Vec3(0, 1, 0)
        assert abs(a.dot(b)) < 1e-10
        assert abs(a.dot(a) - 1.0) < 1e-10

    def test_cross(self) -> None:
        x = Vec3(1, 0, 0)
        y = Vec3(0, 1, 0)
        z = x.cross(y)
        assert abs(z.x) < 1e-10
        assert abs(z.y) < 1e-10
        assert abs(z.z - 1.0) < 1e-10

    def test_to_from_array(self) -> None:
        v = Vec3(1.5, 2.5, 3.5)
        arr = v.to_array()
        assert arr.shape == (3,)
        v2 = Vec3.from_array(arr)
        assert abs(v2.x - v.x) < 1e-10


# ── Enums ────────────────────────────────────────────────────────

class TestEnums:
    def test_modality_values(self) -> None:
        assert Modality.CT.value == "ct"
        assert Modality.SURFACE_SCAN.value == "surface_scan"

    def test_structure_type_count(self) -> None:
        assert len(StructureType) >= 20

    def test_procedure_type_rhinoplasty(self) -> None:
        assert ProcedureType.RHINOPLASTY.value == "rhinoplasty"

    def test_material_model_neohookean(self) -> None:
        assert MaterialModel.NEO_HOOKEAN.value == "neo_hookean"

    def test_landmark_type_bilateral(self) -> None:
        assert hasattr(LandmarkType, "ALAR_RIM_LEFT")
        assert hasattr(LandmarkType, "ALAR_RIM_RIGHT")
        assert hasattr(LandmarkType, "TIP_DEFINING_POINT_LEFT")
        assert hasattr(LandmarkType, "TIP_DEFINING_POINT_RIGHT")


# ── SurfaceMesh ──────────────────────────────────────────────────

class TestSurfaceMesh:
    def test_basic_properties(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = SurfaceMesh(vertices=verts, triangles=tris)
        assert mesh.n_vertices == 3
        assert mesh.n_faces == 1

    def test_compute_normals(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = SurfaceMesh(vertices=verts, triangles=tris)
        mesh.compute_normals()
        assert mesh.normals is not None
        assert mesh.normals.shape == (3, 3)
        for i in range(3):
            assert abs(mesh.normals[i, 2]) > 0.5

    def test_surface_area(self) -> None:
        verts = np.array([
            [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
        ], dtype=np.float64)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = SurfaceMesh(vertices=verts, triangles=tris)
        area = mesh.surface_area_mm2()
        assert abs(area - 100.0) < 1e-6

    def test_bounding_box(self) -> None:
        verts = np.array([
            [-5, -5, -5], [5, 5, 5], [0, 0, 0],
        ], dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = SurfaceMesh(vertices=verts, triangles=tris)
        bb = mesh.bounding_box()
        assert abs(bb.extent.x - 10.0) < 1e-6
        # volume_mm3 is a property, not a method
        assert abs(bb.volume_mm3 - 1000.0) < 1e-6


# ── VolumeMesh ───────────────────────────────────────────────────

class TestVolumeMesh:
    def test_basic(self) -> None:
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
        mesh = VolumeMesh(
            nodes=nodes,
            elements=elems,
            element_type=MeshElementType.TET4,
            region_ids=np.array([0], dtype=np.int32),
        )
        assert mesh.n_nodes == 4
        assert mesh.n_elements == 1


# ── TissueProperties ────────────────────────────────────────────

class TestTissueProperties:
    def test_validate_valid(self) -> None:
        tp = TissueProperties(
            structure_type=StructureType.SKIN_ENVELOPE,
            material_model=MaterialModel.NEO_HOOKEAN,
            parameters={"mu": 10.0, "kappa": 100.0},
            density_kg_m3=1050.0,
        )
        errors = tp.validate()
        assert isinstance(errors, list)


# ── Provenance ───────────────────────────────────────────────────

class TestProvenance:
    def test_hash_bytes(self) -> None:
        h1 = hash_bytes(b"hello world")
        h2 = hash_bytes(b"hello world")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_bytes_different(self) -> None:
        h1 = hash_bytes(b"hello")
        h2 = hash_bytes(b"world")
        assert h1 != h2

    def test_hash_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"test content for hashing")
            path = Path(f.name)
        try:
            h = hash_file(path)
            assert len(h) == 64
        finally:
            path.unlink()

    def test_provenance_tracking(self) -> None:
        p = Provenance(case_id="test_case_001")
        assert p.case_id == "test_case_001"
        p.begin_step("test_op")
        p.record_dict("input_data", {"input": "data"}, "test_artifact")
        p.end_step()
        # Chain should have at least one record
        assert len(p.chain.records) >= 1


# ── CaseBundle ───────────────────────────────────────────────────

class TestCaseBundle:
    def test_create(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bundle = CaseBundle.create(
                library_root=td,
                procedure=ProcedureType.RHINOPLASTY,
            )
            assert bundle.case_id is not None
            assert bundle.procedure == ProcedureType.RHINOPLASTY

    def test_metadata_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bundle = CaseBundle.create(
                library_root=td,
                procedure=ProcedureType.RHINOPLASTY,
                case_id="test_round_trip_01",
            )
            bundle.save()
            loaded = CaseBundle.load(Path(td) / "test_round_trip_01")
            assert loaded.case_id == bundle.case_id
            assert loaded.procedure == bundle.procedure


# ── PlatformConfig ───────────────────────────────────────────────

class TestPlatformConfig:
    def test_defaults(self) -> None:
        cfg = PlatformConfig()
        # Convergence tol lives in solver sub-config
        assert cfg.solver.convergence_tol > 0
        assert cfg.solver.max_iterations > 0

    def test_sub_configs(self) -> None:
        cfg = PlatformConfig()
        assert cfg.mesh.target_edge_length_mm > 0
        assert cfg.cfd.air_density_kg_m3 > 0
        assert cfg.uq.n_samples > 0

    def test_ensure_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = PlatformConfig(
                data_root=Path(td) / "data",
                case_library_root=Path(td) / "library",
                output_root=Path(td) / "output",
                model_root=Path(td) / "models",
                report_root=Path(td) / "reports",
            )
            cfg.ensure_dirs()
            assert cfg.data_root.exists()
            assert cfg.case_library_root.exists()


# ── generate_case_id ─────────────────────────────────────────────

class TestGenerateCaseId:
    def test_format(self) -> None:
        cid = generate_case_id()
        assert cid.startswith("FP-")
        assert len(cid) == 15  # "FP-" + 12 hex chars

    def test_unique(self) -> None:
        ids = {generate_case_id() for _ in range(100)}
        assert len(ids) == 100
