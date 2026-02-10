"""Tests for the twin/ sub-package — segmentation, meshing, registration,
landmarks, materials, and twin_builder."""

from __future__ import annotations

import numpy as np
import pytest

from products.facial_plastics.core.config import (
    MeshConfig,
    PlatformConfig,
    SegmentationConfig,
)
from products.facial_plastics.core.types import (
    Landmark,
    LandmarkType,
    MaterialModel,
    MeshElementType,
    MeshQualityReport,
    Modality,
    RegistrationResult,
    StructureType,
    SurfaceMesh,
    TissueProperties,
    Vec3,
    VolumeMesh,
)
from products.facial_plastics.tests.conftest import (
    make_box_surface_mesh,
    make_nose_surface_mesh,
    make_rhinoplasty_landmarks,
    make_volume_mesh,
)
from products.facial_plastics.twin.landmarks import (
    CANONICAL_LANDMARKS,
    LandmarkDetector,
)
from products.facial_plastics.twin.materials import (
    MaterialAssigner,
    MaterialAssignment,
    STRUCTURE_MODEL_MAP,
)
from products.facial_plastics.twin.meshing import (
    MeshRegion,
    VolumetricMesher,
)
from products.facial_plastics.twin.registration import (
    ICPConfig,
    MultiModalRegistrar,
)
from products.facial_plastics.twin.segmentation import (
    INV_LABEL_MAP,
    LABEL_MAP,
    MultiStructureSegmenter,
    SegmentationResult,
)
from products.facial_plastics.twin.twin_builder import (
    TwinBuilder,
    TwinBuildReport,
)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _make_synthetic_ct(
    shape: tuple[int, int, int] = (30, 40, 40),
    seed: int = 42,
) -> np.ndarray:
    """Build a synthetic CT volume with known HU regions.

    Layout (D=30 slices along z-axis):
      - Bone (HU ≈ 500) : central block near top, midline
      - Cartilage (HU ≈ 200) : below bone, midline
      - Airway (HU ≈ −700) : central column, interior (not touching boundary)
      - Soft tissue (HU ≈ 40) : fill around structures
      - Background (HU = −1000) : outermost shell

    Spacing: (1.0, 1.0, 1.0) mm.
    """
    rng = np.random.default_rng(seed)
    dz, dy, dx = shape

    # Start with background air (−1000)
    vol = np.full(shape, -1000.0, dtype=np.float32)

    # Body ellipsoid (soft tissue ≈ 40 HU) filling centre
    zz, yy, xx = np.mgrid[0:dz, 0:dy, 0:dx]
    cz, cy, cx = dz / 2, dy / 2, dx / 2
    rz, ry, rx = dz / 2 - 2, dy / 2 - 2, dx / 2 - 2
    inside_body = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1.0
    vol[inside_body] = 40.0 + rng.normal(0, 5, size=int(inside_body.sum())).astype(np.float32)

    # Bone block (HU ≈ 500): slices 4–9, rows 10–20, cols 15–25
    bone_slc = (slice(4, 10), slice(10, 21), slice(15, 26))
    vol[bone_slc] = 500.0 + rng.normal(0, 20, size=vol[bone_slc].shape).astype(np.float32)

    # Cartilage block (HU ≈ 200): slices 10–16, rows 8–18, cols 16–24
    cart_slc = (slice(10, 17), slice(8, 19), slice(16, 25))
    vol[cart_slc] = 200.0 + rng.normal(0, 10, size=vol[cart_slc].shape).astype(np.float32)

    # Airway pocket (HU ≈ −700): small interior cavity NOT touching boundary
    air_slc = (slice(12, 18), slice(14, 20), slice(18, 24))
    vol[air_slc] = -700.0 + rng.normal(0, 10, size=vol[air_slc].shape).astype(np.float32)

    return vol


def _make_sphere_surface(radius: float = 10.0, n_lat: int = 12, n_lon: int = 24) -> SurfaceMesh:
    """Triangulated UV-sphere."""
    verts: list[list[float]] = []
    tris: list[list[int]] = []

    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            verts.append([x, y, z])

    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            tris.append([a, b, c])
            tris.append([b, d, c])

    mesh = SurfaceMesh(
        vertices=np.array(verts, dtype=np.float64),
        triangles=np.array(tris, dtype=np.int64),
    )
    mesh.compute_normals()
    return mesh


# ══════════════════════════════════════════════════════════════════
#  1. SEGMENTATION TESTS
# ══════════════════════════════════════════════════════════════════

class TestLabelMap:
    """Sanity checks on the segmentation label map."""

    def test_label_map_not_empty(self) -> None:
        assert len(LABEL_MAP) > 0

    def test_label_map_values_unique(self) -> None:
        vals = list(LABEL_MAP.values())
        assert len(vals) == len(set(vals))

    def test_inv_label_map_consistent(self) -> None:
        for st, lbl in LABEL_MAP.items():
            assert INV_LABEL_MAP[lbl] is st


class TestSegmentation:
    """Tests for MultiStructureSegmenter."""

    @pytest.fixture()
    def segmenter(self) -> MultiStructureSegmenter:
        cfg = SegmentationConfig(min_structure_volume_mm3=1.0)
        return MultiStructureSegmenter(config=cfg)

    @pytest.fixture()
    def ct_volume(self) -> np.ndarray:
        return _make_synthetic_ct()

    @pytest.fixture()
    def spacing(self) -> tuple[float, float, float]:
        return (1.0, 1.0, 1.0)

    # ── Happy-path tests ──────────────────────────────────────

    def test_segment_returns_result(
        self, segmenter: MultiStructureSegmenter, ct_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        result = segmenter.segment(ct_volume, spacing)
        assert isinstance(result, SegmentationResult)

    def test_label_shape_matches_volume(
        self, segmenter: MultiStructureSegmenter, ct_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        result = segmenter.segment(ct_volume, spacing)
        assert result.labels.shape == ct_volume.shape

    def test_label_dtype(
        self, segmenter: MultiStructureSegmenter, ct_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        result = segmenter.segment(ct_volume, spacing)
        assert result.labels.dtype == np.int16

    def test_structures_found_is_list(
        self, segmenter: MultiStructureSegmenter, ct_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        result = segmenter.segment(ct_volume, spacing)
        assert isinstance(result.structures_found, list)
        for s in result.structures_found:
            assert isinstance(s, StructureType)

    def test_bone_detected(
        self, segmenter: MultiStructureSegmenter, ct_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        result = segmenter.segment(ct_volume, spacing)
        bone_types = {
            StructureType.BONE_NASAL,
            StructureType.BONE_MAXILLA,
            StructureType.BONE_MANDIBLE,
            StructureType.BONE_FRONTAL,
        }
        found_bones = bone_types & set(result.structures_found)
        assert len(found_bones) >= 1, (
            f"Expected at least one bone structure; found: {result.structures_found}"
        )

    def test_cartilage_detected(
        self, segmenter: MultiStructureSegmenter, ct_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        result = segmenter.segment(ct_volume, spacing)
        cart_types = {
            StructureType.CARTILAGE_SEPTUM,
            StructureType.CARTILAGE_UPPER_LATERAL,
            StructureType.CARTILAGE_LOWER_LATERAL,
            StructureType.CARTILAGE_EAR,
        }
        found_cart = cart_types & set(result.structures_found)
        assert len(found_cart) >= 1, (
            f"Expected at least one cartilage structure; found: {result.structures_found}"
        )

    def test_soft_tissue_detected(
        self, segmenter: MultiStructureSegmenter, ct_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        result = segmenter.segment(ct_volume, spacing)
        soft_types = {
            StructureType.FAT_SUBCUTANEOUS,
            StructureType.MUSCLE_MIMETIC,
            StructureType.SMAS,
            StructureType.SKIN_THICK,
        }
        found_soft = soft_types & set(result.structures_found)
        assert len(found_soft) >= 1, (
            f"Expected at least one soft tissue type; found: {result.structures_found}"
        )

    def test_volumes_positive(
        self, segmenter: MultiStructureSegmenter, ct_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        result = segmenter.segment(ct_volume, spacing)
        for st, vol in result.volumes_mm3.items():
            assert vol > 0, f"Volume for {st} must be positive"

    def test_quality_scores_bounded(
        self, segmenter: MultiStructureSegmenter, ct_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        result = segmenter.segment(ct_volume, spacing)
        for st, q in result.quality_scores.items():
            assert 0.0 <= q <= 1.0, f"Quality score for {st} out of [0,1]: {q}"

    def test_bounding_boxes_present(
        self, segmenter: MultiStructureSegmenter, ct_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        result = segmenter.segment(ct_volume, spacing)
        for st in result.structures_found:
            assert st in result.bounding_boxes

    def test_voxel_spacing_round_trips(
        self, segmenter: MultiStructureSegmenter, ct_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        result = segmenter.segment(ct_volume, spacing)
        np.testing.assert_allclose(result.voxel_spacing_mm, spacing)

    # ── Edge cases ────────────────────────────────────────────

    def test_empty_volume_no_structures(
        self, segmenter: MultiStructureSegmenter,
        spacing: tuple[float, float, float],
    ) -> None:
        # Use −1024 HU (pure air) so nothing crosses any threshold
        vol = np.full((10, 10, 10), -1024.0, dtype=np.float32)
        result = segmenter.segment(vol, spacing)
        assert result.structures_found == []
        assert result.volumes_mm3 == {}

    def test_uniform_bone_volume(
        self, segmenter: MultiStructureSegmenter,
        spacing: tuple[float, float, float],
    ) -> None:
        # Entire volume is bone
        vol = np.full((15, 15, 15), 600.0, dtype=np.float32)
        result = segmenter.segment(vol, spacing)
        bone_types = {
            StructureType.BONE_NASAL,
            StructureType.BONE_MAXILLA,
            StructureType.BONE_MANDIBLE,
            StructureType.BONE_FRONTAL,
        }
        assert len(bone_types & set(result.structures_found)) >= 1

    # ── Morphological operations ──────────────────────────────

    def test_dilate_3d_expands_mask(self) -> None:
        mask = np.zeros((9, 9, 9), dtype=bool)
        mask[4, 4, 4] = True
        dilated = MultiStructureSegmenter._dilate_3d(mask, radius=1)
        # 6-connected → 1+6 = 7 voxels
        assert dilated.sum() >= 7
        assert dilated[4, 4, 4]
        assert dilated[4, 4, 5]  # one neighbour set

    def test_erode_3d_shrinks_mask(self) -> None:
        mask = np.ones((9, 9, 9), dtype=bool)
        eroded = MultiStructureSegmenter._erode_3d(mask, radius=1)
        # Boundary ring removed
        assert eroded.sum() < mask.sum()
        assert eroded[4, 4, 4]  # interior still set
        assert not eroded[0, 0, 0]  # corner gone

    def test_fill_holes_3d(self) -> None:
        mask = np.ones((9, 9, 9), dtype=bool)
        mask[4, 4, 4] = False  # interior hole
        filled = MultiStructureSegmenter._fill_holes_3d(mask)
        assert filled[4, 4, 4]

    def test_connected_components_separate_blobs(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[1, 1, 1] = True
        mask[8, 8, 8] = True
        labels = MultiStructureSegmenter._connected_components_3d(mask)
        assert labels.max() == 2
        assert labels[1, 1, 1] != labels[8, 8, 8]


# ══════════════════════════════════════════════════════════════════
#  2. MESHING TESTS
# ══════════════════════════════════════════════════════════════════

class TestVolumetricMesher:
    """Tests for VolumetricMesher."""

    @pytest.fixture()
    def mesher(self) -> VolumetricMesher:
        return VolumetricMesher(config=MeshConfig(target_edge_length_mm=3.0))

    @pytest.fixture()
    def box_surface(self) -> SurfaceMesh:
        return make_box_surface_mesh(size=20.0, n_per_edge=5)

    @pytest.fixture()
    def segmentation_labels(self) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Small label volume with one bone block."""
        labels = np.zeros((12, 12, 12), dtype=np.int16)
        labels[2:10, 2:10, 2:10] = LABEL_MAP[StructureType.BONE_NASAL]
        return labels, (1.0, 1.0, 1.0)

    # ── mesh_from_labels ──────────────────────────────────────

    def test_mesh_from_labels_returns_volume_mesh(
        self, mesher: VolumetricMesher,
        segmentation_labels: tuple[np.ndarray, tuple[float, float, float]],
    ) -> None:
        labels, spacing = segmentation_labels
        mesh = mesher.mesh_from_labels(labels, spacing)
        assert isinstance(mesh, VolumeMesh)

    def test_mesh_from_labels_has_elements(
        self, mesher: VolumetricMesher,
        segmentation_labels: tuple[np.ndarray, tuple[float, float, float]],
    ) -> None:
        labels, spacing = segmentation_labels
        mesh = mesher.mesh_from_labels(labels, spacing)
        assert mesh.n_elements > 0
        assert mesh.n_nodes > 0

    def test_mesh_from_labels_element_type(
        self, mesher: VolumetricMesher,
        segmentation_labels: tuple[np.ndarray, tuple[float, float, float]],
    ) -> None:
        labels, spacing = segmentation_labels
        mesh = mesher.mesh_from_labels(labels, spacing)
        assert mesh.element_type == MeshElementType.TET4

    def test_mesh_from_labels_positive_volumes(
        self, mesher: VolumetricMesher,
        segmentation_labels: tuple[np.ndarray, tuple[float, float, float]],
    ) -> None:
        labels, spacing = segmentation_labels
        mesh = mesher.mesh_from_labels(labels, spacing)
        report = mesher.compute_quality(mesh)
        assert report.n_inverted == 0, f"{report.n_inverted} inverted elements"

    def test_mesh_from_labels_empty_raises(
        self, mesher: VolumetricMesher,
    ) -> None:
        labels = np.zeros((8, 8, 8), dtype=np.int16)
        with pytest.raises(ValueError, match="No non-zero labels"):
            mesher.mesh_from_labels(labels, (1.0, 1.0, 1.0))

    # ── mesh_from_surface ─────────────────────────────────────

    def test_mesh_from_surface_box(
        self, mesher: VolumetricMesher, box_surface: SurfaceMesh,
    ) -> None:
        mesh = mesher.mesh_from_surface(box_surface, target_edge_length_mm=5.0)
        assert isinstance(mesh, VolumeMesh)
        assert mesh.n_elements > 0

    def test_mesh_from_surface_sphere(self, mesher: VolumetricMesher) -> None:
        sphere = _make_sphere_surface(radius=8.0, n_lat=8, n_lon=16)
        mesh = mesher.mesh_from_surface(sphere, target_edge_length_mm=4.0)
        assert mesh.n_elements > 0

    # ── compute_quality ───────────────────────────────────────

    def test_compute_quality_on_conftest_mesh(
        self, mesher: VolumetricMesher,
    ) -> None:
        vol_mesh = make_volume_mesh()
        report = mesher.compute_quality(vol_mesh)
        assert isinstance(report, MeshQualityReport)
        assert report.n_nodes == vol_mesh.n_nodes
        assert report.n_elements == vol_mesh.n_elements
        assert report.volume_mm3 > 0
        assert report.surface_area_mm2 > 0
        assert report.min_edge_length_mm > 0
        assert report.max_edge_length_mm >= report.min_edge_length_mm
        assert report.max_aspect_ratio >= 1.0

    def test_compute_quality_no_inverted(
        self, mesher: VolumetricMesher,
    ) -> None:
        vol_mesh = make_volume_mesh()
        report = mesher.compute_quality(vol_mesh)
        assert report.n_inverted == 0

    # ── Delaunay wrapper ──────────────────────────────────────

    def test_delaunay_3d_tetrahedra(self, mesher: VolumetricMesher) -> None:
        rng = np.random.default_rng(42)
        pts = rng.uniform(-5, 5, size=(30, 3))
        tets = mesher._delaunay_3d(pts)
        assert tets.shape[1] == 4
        assert tets.shape[0] > 0
        # All indices valid
        assert tets.min() >= 0
        assert tets.max() < len(pts)

    def test_delaunay_3d_too_few_points(self, mesher: VolumetricMesher) -> None:
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        tets = mesher._delaunay_3d(pts)
        assert tets.shape[0] == 0  # need ≥4 for 3-D

    # ── Tiny volume edge case ─────────────────────────────────

    def test_mesh_from_labels_tiny_volume(
        self, mesher: VolumetricMesher,
    ) -> None:
        labels = np.zeros((5, 5, 5), dtype=np.int16)
        labels[1:4, 1:4, 1:4] = LABEL_MAP[StructureType.BONE_NASAL]
        mesh = mesher.mesh_from_labels(labels, (1.0, 1.0, 1.0))
        assert mesh.n_elements > 0


# ══════════════════════════════════════════════════════════════════
#  3. REGISTRATION TESTS
# ══════════════════════════════════════════════════════════════════

class TestRegistration:
    """Tests for MultiModalRegistrar."""

    @pytest.fixture()
    def registrar(self) -> MultiModalRegistrar:
        return MultiModalRegistrar(icp_config=ICPConfig(max_iterations=50))

    @pytest.fixture()
    def box_mesh(self) -> SurfaceMesh:
        return make_box_surface_mesh(size=20.0, n_per_edge=5)

    # ── SVD rigid registration ────────────────────────────────

    def test_svd_rigid_identity(self, registrar: MultiModalRegistrar) -> None:
        rng = np.random.default_rng(7)
        P = rng.uniform(-10, 10, size=(20, 3))
        R, t = registrar._svd_rigid(P, P)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-10)

    def test_svd_rigid_known_translation(self, registrar: MultiModalRegistrar) -> None:
        rng = np.random.default_rng(7)
        P = rng.uniform(-10, 10, size=(20, 3))
        shift = np.array([5.0, -3.0, 7.0])
        Q = P + shift
        R, t = registrar._svd_rigid(P, Q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(t, shift, atol=1e-8)

    def test_svd_rigid_known_rotation_90z(self, registrar: MultiModalRegistrar) -> None:
        rng = np.random.default_rng(7)
        P = rng.uniform(-10, 10, size=(30, 3))
        Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        Q = (Rz @ P.T).T
        R_est, t_est = registrar._svd_rigid(P, Q)
        np.testing.assert_allclose(R_est, Rz, atol=1e-8)
        np.testing.assert_allclose(t_est, np.zeros(3), atol=1e-8)

    # ── Landmark registration ─────────────────────────────────

    def _make_landmark_set(
        self, prefix: str, positions: np.ndarray,
    ) -> list[Landmark]:
        return [
            Landmark(
                landmark_type=LandmarkType.NASION,
                name=f"pt_{i}",
                position=Vec3(float(p[0]), float(p[1]), float(p[2])),
                confidence=1.0,
                source=prefix,
            )
            for i, p in enumerate(positions)
        ]

    def test_register_landmarks_known_translation(
        self, registrar: MultiModalRegistrar,
    ) -> None:
        rng = np.random.default_rng(42)
        pts = rng.uniform(-20, 20, size=(6, 3))
        shift = np.array([4.0, -2.0, 8.0])
        src_lm = self._make_landmark_set("src", pts)
        tgt_lm = self._make_landmark_set("tgt", pts + shift)
        result = registrar.register_landmarks(src_lm, tgt_lm)
        assert isinstance(result, RegistrationResult)
        assert result.residual_mm < 1e-6
        np.testing.assert_allclose(result.translation, shift, atol=1e-6)

    def test_register_landmarks_too_few_raises(
        self, registrar: MultiModalRegistrar,
    ) -> None:
        lm = [
            Landmark(LandmarkType.NASION, Vec3(0, 0, 0), name="a"),
            Landmark(LandmarkType.RHINION, Vec3(1, 0, 0), name="b"),
        ]
        with pytest.raises(ValueError, match="≥ 3"):
            registrar.register_landmarks(lm, lm)

    # ── ICP (surface) registration ────────────────────────────

    def test_icp_identity_zero_displacement(
        self, registrar: MultiModalRegistrar, box_mesh: SurfaceMesh,
    ) -> None:
        result = registrar.register_surfaces(box_mesh, box_mesh)
        assert isinstance(result, RegistrationResult)
        assert result.residual_mm < 1.0

    def test_icp_known_translation(
        self, registrar: MultiModalRegistrar,
    ) -> None:
        src = make_box_surface_mesh(size=20.0, n_per_edge=5)
        shift = np.array([2.0, 0.0, 0.0])
        tgt_verts = src.vertices + shift
        tgt = SurfaceMesh(vertices=tgt_verts, triangles=src.triangles.copy())
        tgt.compute_normals()
        result = registrar.register_surfaces(src, tgt)
        # ICP should recover the translation within ~2 mm tolerance
        np.testing.assert_allclose(
            result.translation, shift, atol=3.0,
        )

    def test_icp_with_initial_transform(
        self, registrar: MultiModalRegistrar,
    ) -> None:
        src = make_box_surface_mesh(size=20.0, n_per_edge=5)
        shift = np.array([3.0, 0.0, 0.0])
        tgt_verts = src.vertices + shift
        tgt = SurfaceMesh(vertices=tgt_verts, triangles=src.triangles.copy())
        tgt.compute_normals()

        init = np.eye(4, dtype=np.float64)
        init[:3, 3] = shift * 0.8  # close initial guess
        result = registrar.register_surfaces(src, tgt, initial_transform=init)
        assert result.residual_mm < 5.0

    def test_registration_result_fields(
        self, registrar: MultiModalRegistrar, box_mesh: SurfaceMesh,
    ) -> None:
        result = registrar.register_surfaces(box_mesh, box_mesh)
        assert result.rotation.shape == (3, 3)
        assert result.translation.shape == (3,)
        assert result.n_correspondences > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.method == "icp"
        T = result.rigid_transform
        assert T.shape == (4, 4)

    # ── TPS deformable registration ───────────────────────────

    def test_tps_identity(self, registrar: MultiModalRegistrar) -> None:
        sphere = _make_sphere_surface(radius=5.0, n_lat=6, n_lon=12)
        rng = np.random.default_rng(42)
        lm = rng.uniform(-4, 4, size=(8, 3))
        warped, disp = registrar.register_deformable(lm, lm, sphere)
        np.testing.assert_allclose(disp, 0.0, atol=1e-4)
        np.testing.assert_allclose(
            warped.vertices, sphere.vertices, atol=1e-3,
        )

    def test_tps_known_shift(self, registrar: MultiModalRegistrar) -> None:
        sphere = _make_sphere_surface(radius=5.0, n_lat=6, n_lon=12)
        rng = np.random.default_rng(42)
        src_lm = rng.uniform(-4, 4, size=(8, 3))
        shift = np.array([2.0, -1.0, 0.5])
        tgt_lm = src_lm + shift
        warped, disp = registrar.register_deformable(src_lm, tgt_lm, sphere)
        mean_disp = disp.mean(axis=0)
        np.testing.assert_allclose(mean_disp, shift, atol=1.0)


# ══════════════════════════════════════════════════════════════════
#  4. LANDMARK TESTS
# ══════════════════════════════════════════════════════════════════

class TestLandmarkDetector:
    """Tests for LandmarkDetector."""

    @pytest.fixture()
    def detector(self) -> LandmarkDetector:
        return LandmarkDetector()

    @pytest.fixture()
    def nose_surface(self) -> SurfaceMesh:
        mesh = make_nose_surface_mesh(n_verts=200)
        mesh.compute_normals()
        return mesh

    # ── Surface-based landmark detection ──────────────────────

    def test_detect_from_surface_returns_landmarks(
        self, detector: LandmarkDetector, nose_surface: SurfaceMesh,
    ) -> None:
        landmarks = detector.detect_from_surface(nose_surface)
        assert isinstance(landmarks, list)
        assert len(landmarks) > 0
        for lm in landmarks:
            assert isinstance(lm, Landmark)

    def test_detected_landmarks_have_valid_positions(
        self, detector: LandmarkDetector, nose_surface: SurfaceMesh,
    ) -> None:
        landmarks = detector.detect_from_surface(nose_surface)
        bb = nose_surface.bounding_box()
        for lm in landmarks:
            # Position should be within the mesh bounding box (with margin)
            margin = 5.0
            assert lm.position.x >= bb.origin.x - margin
            assert lm.position.y >= bb.origin.y - margin
            assert lm.position.z >= bb.origin.z - margin

    def test_detected_landmarks_have_confidence(
        self, detector: LandmarkDetector, nose_surface: SurfaceMesh,
    ) -> None:
        landmarks = detector.detect_from_surface(nose_surface)
        for lm in landmarks:
            assert 0.0 <= lm.confidence <= 1.0

    def test_detect_subset(
        self, detector: LandmarkDetector, nose_surface: SurfaceMesh,
    ) -> None:
        subset = [LandmarkType.PRONASALE, LandmarkType.MENTON]
        landmarks = detector.detect_from_surface(nose_surface, subset=subset)
        for lm in landmarks:
            assert lm.landmark_type in subset

    def test_detect_from_surface_sphere(
        self, detector: LandmarkDetector,
    ) -> None:
        sphere = _make_sphere_surface(radius=10.0, n_lat=10, n_lon=20)
        landmarks = detector.detect_from_surface(sphere)
        assert isinstance(landmarks, list)
        # On a sphere the heuristics may not match well, but should not crash
        for lm in landmarks:
            dist = lm.position.norm()
            # Should be on or near the sphere surface
            assert dist <= 12.0

    # ── Volume-based landmark detection ───────────────────────

    def test_detect_from_volume(
        self, detector: LandmarkDetector,
    ) -> None:
        vol = _make_synthetic_ct(shape=(30, 40, 40))
        spacing = (1.0, 1.0, 1.0)
        landmarks = detector.detect_from_volume(vol, spacing)
        assert isinstance(landmarks, list)
        for lm in landmarks:
            assert isinstance(lm, Landmark)
            assert lm.source == "volume_detection"

    def test_detect_from_volume_with_labels(
        self, detector: LandmarkDetector,
    ) -> None:
        vol = _make_synthetic_ct(shape=(30, 40, 40))
        spacing = (1.0, 1.0, 1.0)
        seg = MultiStructureSegmenter(SegmentationConfig(min_structure_volume_mm3=1.0))
        seg_result = seg.segment(vol, spacing)
        landmarks = detector.detect_from_volume(vol, spacing, labels=seg_result.labels)
        assert isinstance(landmarks, list)

    # ── Curvature computation ─────────────────────────────────

    def test_curvature_computation_plausible(
        self, detector: LandmarkDetector,
    ) -> None:
        sphere = _make_sphere_surface(radius=10.0, n_lat=10, n_lon=20)
        curvs = detector._compute_vertex_curvatures(sphere)
        assert curvs.shape[0] == sphere.n_vertices
        assert np.isfinite(curvs).all()
        # On a sphere of radius R, mean curvature ≈ 1/R = 0.1
        median_abs = float(np.median(np.abs(curvs)))
        assert 0.001 < median_abs < 10.0, f"Median curvature {median_abs} implausible"

    def test_curvature_on_nose_surface(
        self, detector: LandmarkDetector, nose_surface: SurfaceMesh,
    ) -> None:
        curvs = detector._compute_vertex_curvatures(nose_surface)
        assert curvs.shape[0] == nose_surface.n_vertices
        assert np.isfinite(curvs).all()


# ══════════════════════════════════════════════════════════════════
#  5. MATERIALS TESTS
# ══════════════════════════════════════════════════════════════════

class TestStructureModelMap:
    """Sanity checks on STRUCTURE_MODEL_MAP."""

    def test_bone_entries_exist(self) -> None:
        for st in (StructureType.BONE_NASAL, StructureType.BONE_MAXILLA):
            assert st in STRUCTURE_MODEL_MAP

    def test_cartilage_entries_exist(self) -> None:
        for st in (
            StructureType.CARTILAGE_SEPTUM,
            StructureType.CARTILAGE_UPPER_LATERAL,
            StructureType.CARTILAGE_LOWER_LATERAL,
        ):
            assert st in STRUCTURE_MODEL_MAP

    def test_values_are_valid_tuples(self) -> None:
        for st, (model, key) in STRUCTURE_MODEL_MAP.items():
            assert isinstance(model, MaterialModel)
            assert isinstance(key, str)
            assert len(key) > 0


class TestMaterialAssigner:
    """Tests for MaterialAssigner."""

    @pytest.fixture()
    def assigner(self) -> MaterialAssigner:
        return MaterialAssigner()

    @pytest.fixture()
    def mesh_with_labels(self) -> tuple[VolumeMesh, np.ndarray]:
        """Volume mesh + element labels spanning bone and cartilage."""
        vol_mesh = make_volume_mesh()
        n_elem = vol_mesh.n_elements
        labels = np.zeros(n_elem, dtype=np.int32)
        # First third → bone_nasal, second third → cartilage_septum,
        # rest → fat_subcutaneous
        t1 = n_elem // 3
        t2 = 2 * n_elem // 3
        labels[:t1] = LABEL_MAP[StructureType.BONE_NASAL]
        labels[t1:t2] = LABEL_MAP[StructureType.CARTILAGE_SEPTUM]
        labels[t2:] = LABEL_MAP[StructureType.FAT_SUBCUTANEOUS]
        return vol_mesh, labels

    # ── Assign ────────────────────────────────────────────────

    def test_assign_returns_assignments(
        self, assigner: MaterialAssigner,
        mesh_with_labels: tuple[VolumeMesh, np.ndarray],
    ) -> None:
        mesh, labels = mesh_with_labels
        results = assigner.assign(mesh, labels)
        assert isinstance(results, list)
        assert len(results) == 3  # bone, cartilage, fat

    def test_all_elements_covered(
        self, assigner: MaterialAssigner,
        mesh_with_labels: tuple[VolumeMesh, np.ndarray],
    ) -> None:
        mesh, labels = mesh_with_labels
        results = assigner.assign(mesh, labels)
        all_indices: set[int] = set()
        for a in results:
            all_indices.update(a.element_indices.tolist())
        # Every element with a known label should be assigned
        assigned_label_mask = labels > 0
        expected = set(int(i) for i in np.where(assigned_label_mask)[0])
        assert expected == all_indices

    def test_tissue_properties_valid(
        self, assigner: MaterialAssigner,
        mesh_with_labels: tuple[VolumeMesh, np.ndarray],
    ) -> None:
        mesh, labels = mesh_with_labels
        for a in assigner.assign(mesh, labels):
            tp = a.properties
            assert isinstance(tp, TissueProperties)
            assert tp.material_model in MaterialModel
            params = tp.parameters
            assert len(params) > 0
            # All parameter values should be finite
            for k, v in params.items():
                assert np.isfinite(v), f"param {k}={v} not finite"

    def test_assignment_param_dict(
        self, assigner: MaterialAssigner,
        mesh_with_labels: tuple[VolumeMesh, np.ndarray],
    ) -> None:
        mesh, labels = mesh_with_labels
        for a in assigner.assign(mesh, labels):
            d = a.param_dict()
            assert isinstance(d, dict)
            assert all(isinstance(v, float) for v in d.values())

    # ── Age adjustment ────────────────────────────────────────

    def test_age_adjustment_changes_cartilage(
        self, assigner: MaterialAssigner,
        mesh_with_labels: tuple[VolumeMesh, np.ndarray],
    ) -> None:
        mesh, labels = mesh_with_labels
        young = assigner.assign(mesh, labels, age_years=25)
        old = assigner.assign(mesh, labels, age_years=65)
        # Find cartilage assignment in each
        young_cart = [a for a in young if "cartilage" in a.structure.value][0]
        old_cart = [a for a in old if "cartilage" in a.structure.value][0]
        # Old cartilage should be stiffer (higher C1)
        if "C1" in young_cart.param_dict() and "C1" in old_cart.param_dict():
            assert old_cart.param_dict()["C1"] > young_cart.param_dict()["C1"]

    def test_age_adjustment_changes_skin(self) -> None:
        params_young = MaterialAssigner._adjust_for_age(
            {"mu": 10.0e3, "kappa": 100.0e3}, MaterialModel.NEO_HOOKEAN,
            StructureType.SKIN_THICK, 25,
        )
        params_old = MaterialAssigner._adjust_for_age(
            {"mu": 10.0e3, "kappa": 100.0e3}, MaterialModel.NEO_HOOKEAN,
            StructureType.SKIN_THICK, 65,
        )
        # Older skin should be less stiff (mu decreases)
        assert params_old["mu"] < params_young["mu"]

    def test_age_adjustment_bone_after_40(self) -> None:
        params_40 = MaterialAssigner._adjust_for_age(
            {"E": 2.0e9, "density": 1800.0}, MaterialModel.LINEAR_ELASTIC,
            StructureType.BONE_NASAL, 40,
        )
        params_70 = MaterialAssigner._adjust_for_age(
            {"E": 2.0e9, "density": 1800.0}, MaterialModel.LINEAR_ELASTIC,
            StructureType.BONE_NASAL, 70,
        )
        assert params_70["E"] < params_40["E"]
        assert params_70["density"] < params_40["density"]

    # ── Skin type adjustment ──────────────────────────────────

    def test_skin_type_adjustment(self) -> None:
        base = {"mu": 10.0e3, "kappa": 100.0e3, "thickness": 3.0}
        adj3 = MaterialAssigner._adjust_for_skin_type(dict(base), 3)
        adj5 = MaterialAssigner._adjust_for_skin_type(dict(base), 5)
        assert adj3["mu"] > base["mu"]
        assert adj5["mu"] > adj3["mu"]
        assert adj5["thickness"] > adj3["thickness"]

    # ── Export for solver ─────────────────────────────────────

    def test_export_for_solver(
        self, assigner: MaterialAssigner,
        mesh_with_labels: tuple[VolumeMesh, np.ndarray],
    ) -> None:
        mesh, labels = mesh_with_labels
        assignments = assigner.assign(mesh, labels)
        exported = assigner.export_for_solver(assignments)
        assert isinstance(exported, dict)
        for key, val in exported.items():
            assert "model" in val
            assert "parameters" in val
            assert "n_elements" in val
            assert val["n_elements"] > 0


# ══════════════════════════════════════════════════════════════════
#  6. TWIN BUILDER TESTS
# ══════════════════════════════════════════════════════════════════

class TestTwinBuildReport:
    """TwinBuildReport dataclass."""

    def test_default_report(self) -> None:
        r = TwinBuildReport(case_id="TEST-001")
        assert r.case_id == "TEST-001"
        assert r.total_time_s == 0.0
        assert r.stages_completed == []
        assert r.stages_failed == []

    def test_report_mutable(self) -> None:
        r = TwinBuildReport(case_id="TEST-002")
        r.stages_completed.append("segmentation")
        r.n_structures_segmented = 5
        assert "segmentation" in r.stages_completed
        assert r.n_structures_segmented == 5


class TestTwinBuilder:
    """Tests for TwinBuilder initialization and sub-component access."""

    def test_init_default_config(self) -> None:
        builder = TwinBuilder()
        assert builder._segmenter is not None
        assert builder._registrar is not None
        assert builder._mesher is not None
        assert builder._landmark_detector is not None
        assert builder._material_assigner is not None

    def test_init_custom_config(self) -> None:
        cfg = PlatformConfig(
            segmentation=SegmentationConfig(bone_hu_threshold=400.0),
            mesh=MeshConfig(target_edge_length_mm=2.0),
        )
        builder = TwinBuilder(config=cfg)
        assert builder._config.segmentation.bone_hu_threshold == 400.0
        assert builder._config.mesh.target_edge_length_mm == 2.0

    def test_segmenter_can_segment(self) -> None:
        """Verify the builder's segmenter works on synthetic data."""
        builder = TwinBuilder(config=PlatformConfig(
            segmentation=SegmentationConfig(min_structure_volume_mm3=1.0),
        ))
        vol = _make_synthetic_ct()
        result = builder._segmenter.segment(vol, (1.0, 1.0, 1.0))
        assert len(result.structures_found) > 0

    def test_registrar_can_do_svd(self) -> None:
        builder = TwinBuilder()
        rng = np.random.default_rng(99)
        P = rng.uniform(-10, 10, size=(10, 3))
        shift = np.array([1.0, 2.0, 3.0])
        R, t = builder._registrar._svd_rigid(P, P + shift)
        np.testing.assert_allclose(t, shift, atol=1e-8)

    def test_mesher_can_mesh_box(self) -> None:
        builder = TwinBuilder(config=PlatformConfig(
            mesh=MeshConfig(target_edge_length_mm=5.0),
        ))
        box = make_box_surface_mesh(size=20.0, n_per_edge=5)
        mesh = builder._mesher.mesh_from_surface(box, target_edge_length_mm=5.0)
        assert mesh.n_elements > 0

    def test_material_assigner_can_assign(self) -> None:
        builder = TwinBuilder()
        vmesh = make_volume_mesh()
        labels = np.full(vmesh.n_elements, LABEL_MAP[StructureType.FAT_SUBCUTANEOUS], dtype=np.int32)
        assignments = builder._material_assigner.assign(vmesh, labels)
        assert len(assignments) >= 1

    def test_landmark_detector_can_detect(self) -> None:
        builder = TwinBuilder()
        nose = make_nose_surface_mesh(n_verts=200)
        nose.compute_normals()
        landmarks = builder._landmark_detector.detect_from_surface(nose)
        assert isinstance(landmarks, list)

    def test_deduplicate_landmarks(self) -> None:
        dup = [
            Landmark(LandmarkType.NASION, Vec3(0, 0, 0), confidence=0.5, name="nasion"),
            Landmark(LandmarkType.NASION, Vec3(1, 1, 1), confidence=0.9, name="nasion"),
        ]
        result = TwinBuilder._deduplicate_landmarks(dup)
        assert len(result) == 1
        assert result[0].confidence == 0.9
        np.testing.assert_allclose(
            result[0].position.to_array(), [1.0, 1.0, 1.0],
        )


# ══════════════════════════════════════════════════════════════════
#  Integration-style cross-module tests
# ══════════════════════════════════════════════════════════════════

class TestSegmentThenMesh:
    """End-to-end: segment a synthetic CT, then mesh the result."""

    def test_segment_and_mesh(self) -> None:
        vol = _make_synthetic_ct(shape=(20, 24, 24))
        spacing = (1.0, 1.0, 1.0)

        seg = MultiStructureSegmenter(SegmentationConfig(min_structure_volume_mm3=1.0))
        seg_result = seg.segment(vol, spacing)
        assert len(seg_result.structures_found) > 0

        mesher = VolumetricMesher(MeshConfig(target_edge_length_mm=4.0))
        mesh = mesher.mesh_from_labels(seg_result.labels, spacing)
        assert mesh.n_elements > 0

        report = mesher.compute_quality(mesh)
        assert report.volume_mm3 > 0
        assert report.n_inverted == 0


class TestSegmentThenMaterials:
    """Segment → assign materials to elements."""

    def test_assign_materials_from_segmentation(self) -> None:
        vol = _make_synthetic_ct(shape=(20, 24, 24))
        spacing = (1.0, 1.0, 1.0)

        seg = MultiStructureSegmenter(SegmentationConfig(min_structure_volume_mm3=1.0))
        seg_result = seg.segment(vol, spacing)

        mesher = VolumetricMesher(MeshConfig(target_edge_length_mm=4.0))
        mesh = mesher.mesh_from_labels(seg_result.labels, spacing)

        regions = VolumetricMesher._assign_regions(
            mesh.nodes, mesh.elements, seg_result.labels, spacing,
        )

        assigner = MaterialAssigner()
        assignments = assigner.assign(mesh, regions)
        assert len(assignments) >= 1
        for a in assignments:
            assert a.element_indices.shape[0] > 0
