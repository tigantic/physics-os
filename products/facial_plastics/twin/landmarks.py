"""Anatomical landmark detection from CT volumes and surface meshes.

Detects standardized craniofacial landmarks used for:
  - Registration alignment
  - Surgical plan parameterization
  - Metric computation (dorsal length, tip projection, etc.)
  - Quality control
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import (
    Landmark,
    LandmarkType,
    SegmentationMask,
    SurfaceMesh,
    Vec3,
)

logger = logging.getLogger(__name__)


# ── Canonical landmark definitions ────────────────────────────────

@dataclass
class LandmarkDefinition:
    """Specification for how to detect a landmark."""
    landmark_type: LandmarkType
    name: str
    description: str
    detection_method: str  # "curvature", "template", "extremal", "midpoint"
    structure_context: str  # which structure to search near


CANONICAL_LANDMARKS: List[LandmarkDefinition] = [
    # Nasal landmarks
    LandmarkDefinition(LandmarkType.NASION, "nasion", "Deepest point of nasal bridge at frontonasal suture", "curvature", "nasal_bone"),
    LandmarkDefinition(LandmarkType.RHINION, "rhinion", "Distal end of nasal bones", "extremal", "nasal_bone"),
    LandmarkDefinition(LandmarkType.PRONASALE, "pronasale", "Most anterior point of nasal tip", "extremal", "skin"),
    LandmarkDefinition(LandmarkType.SUBNASALE, "subnasale", "Junction of columella and upper lip", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.COLUMELLA_BREAKPOINT, "columella_break", "Inflection point of columella", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.SUPRATIP_BREAKPOINT, "supratip_break", "Inflection above tip defining point", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.ALAR_RIM_LEFT, "alar_rim_left", "Left alar rim lowest point", "extremal", "skin"),
    LandmarkDefinition(LandmarkType.ALAR_RIM_RIGHT, "alar_rim_right", "Right alar rim lowest point", "extremal", "skin"),
    LandmarkDefinition(LandmarkType.ALAR_CREASE_LEFT, "alar_base_left", "Left alar base insertion", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.ALAR_CREASE_RIGHT, "alar_base_right", "Right alar base insertion", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.INTERNAL_VALVE_LEFT, "nasal_valve_left", "Narrowest point of left internal valve", "extremal", "airway"),
    LandmarkDefinition(LandmarkType.INTERNAL_VALVE_RIGHT, "nasal_valve_right", "Narrowest point of right internal valve", "extremal", "airway"),
    # Orbital landmarks
    LandmarkDefinition(LandmarkType.ENDOCANTHION_LEFT, "endocanthion_left", "Left medial canthus", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.ENDOCANTHION_RIGHT, "endocanthion_right", "Right medial canthus", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.EXOCANTHION_LEFT, "exocanthion_left", "Left lateral canthus", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.EXOCANTHION_RIGHT, "exocanthion_right", "Right lateral canthus", "curvature", "skin"),
    # Midface
    LandmarkDefinition(LandmarkType.GLABELLA, "glabella", "Most prominent point between brows", "extremal", "skin"),
    LandmarkDefinition(LandmarkType.POGONION, "pogonion", "Most anterior chin point", "extremal", "skin"),
    LandmarkDefinition(LandmarkType.MENTON, "menton", "Lowest chin point", "extremal", "skin"),
    LandmarkDefinition(LandmarkType.TRAGION_LEFT, "tragion_left", "Left tragus notch", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.TRAGION_RIGHT, "tragion_right", "Right tragus notch", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.GONION_LEFT, "gonion_left", "Left mandibular angle", "curvature", "bone"),
    LandmarkDefinition(LandmarkType.GONION_RIGHT, "gonion_right", "Right mandibular angle", "curvature", "bone"),
    # Lip
    LandmarkDefinition(LandmarkType.LABRALE_SUPERIUS, "labiale_superius", "Upper lip vermilion midpoint", "extremal", "skin"),
    LandmarkDefinition(LandmarkType.LABRALE_INFERIUS, "labiale_inferius", "Lower lip vermilion midpoint", "extremal", "skin"),
    LandmarkDefinition(LandmarkType.CHEILION_LEFT, "cheilion_left", "Left oral commissure", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.CHEILION_RIGHT, "cheilion_right", "Right oral commissure", "curvature", "skin"),
    # Dorsal line
    LandmarkDefinition(LandmarkType.SELLION, "sellion", "Deepest point of nasofrontal angle (soft tissue)", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.TIP_DEFINING_POINT_LEFT, "tip_defining_left", "Left tip defining point (light reflex)", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.TIP_DEFINING_POINT_RIGHT, "tip_defining_right", "Right tip defining point (light reflex)", "curvature", "skin"),
    LandmarkDefinition(LandmarkType.ANS, "anterior_nasal_spine", "Anterior nasal spine on bone", "extremal", "bone"),
    LandmarkDefinition(LandmarkType.PNS, "posterior_nasal_spine", "Posterior nasal spine on bone", "extremal", "bone"),
    LandmarkDefinition(LandmarkType.A_POINT, "a_point", "Subspinale (deepest point on premaxilla)", "curvature", "bone"),
]


class LandmarkDetector:
    """Detect anatomical landmarks from CT volumes and surface meshes.

    Detection strategies:
      1. Curvature-based: Find local extremals of mean/Gaussian curvature
      2. Extremal: Find min/max of coordinate along specific axis
      3. Template matching: Match against known landmark configurations
      4. Midpoint: Computed as midpoint between two other landmarks
    """

    def detect_from_surface(
        self,
        mesh: SurfaceMesh,
        *,
        subset: Optional[List[LandmarkType]] = None,
    ) -> List[Landmark]:
        """Detect landmarks from a surface mesh.

        Parameters
        ----------
        mesh : SurfaceMesh
            Facial surface mesh.
        subset : list of LandmarkType, optional
            If provided, only detect these landmarks.

        Returns
        -------
        List of detected Landmark objects with confidence scores.
        """
        verts = mesh.vertices
        normals = mesh.normals if mesh.normals is not None else np.zeros_like(verts)

        # Precompute geometry
        curvatures = self._compute_vertex_curvatures(mesh)

        landmarks = []
        definitions = CANONICAL_LANDMARKS
        if subset:
            definitions = [d for d in definitions if d.landmark_type in subset]

        for defn in definitions:
            try:
                pos, confidence = self._detect_single(
                    defn, verts, normals, curvatures, mesh
                )
                if pos is not None:
                    landmarks.append(Landmark(
                        name=defn.name,
                        position=pos,
                        confidence=confidence,
                        source="surface_detection",
                    ))
            except Exception as e:
                logger.debug("Failed to detect %s: %s", defn.name, e)

        logger.info("Detected %d / %d landmarks from surface", len(landmarks), len(definitions))
        return landmarks

    def detect_from_volume(
        self,
        volume_hu: np.ndarray,
        voxel_spacing_mm: Tuple[float, float, float],
        labels: Optional[np.ndarray] = None,
        *,
        subset: Optional[List[LandmarkType]] = None,
    ) -> List[Landmark]:
        """Detect landmarks from CT volume (optionally with segmentation labels).

        Parameters
        ----------
        volume_hu : ndarray (D, H, W)
            CT volume in Hounsfield Units.
        voxel_spacing_mm : tuple of 3 floats
            Voxel spacing.
        labels : ndarray (D, H, W), optional
            Segmentation label volume.
        """
        landmarks = []
        definitions = [d for d in CANONICAL_LANDMARKS if d.structure_context in ("bone", "airway")]
        if subset:
            definitions = [d for d in definitions if d.landmark_type in subset]

        sz, sy, sx = voxel_spacing_mm

        for defn in definitions:
            try:
                pos, confidence = self._detect_from_volume_single(
                    defn, volume_hu, voxel_spacing_mm, labels
                )
                if pos is not None:
                    landmarks.append(Landmark(
                        name=defn.name,
                        position=pos,
                        confidence=confidence,
                        source="volume_detection",
                    ))
            except Exception as e:
                logger.debug("Failed to detect %s from volume: %s", defn.name, e)

        logger.info("Detected %d landmarks from volume", len(landmarks))
        return landmarks

    # ── Single landmark detection from surface ────────────────

    def _detect_single(
        self,
        defn: LandmarkDefinition,
        verts: np.ndarray,
        normals: np.ndarray,
        curvatures: np.ndarray,
        mesh: SurfaceMesh,
    ) -> Tuple[Optional[Vec3], float]:
        """Detect a single landmark from surface mesh."""
        if defn.detection_method == "extremal":
            return self._detect_extremal(defn, verts)
        elif defn.detection_method == "curvature":
            return self._detect_curvature(defn, verts, curvatures)
        else:
            return None, 0.0

    def _detect_extremal(
        self,
        defn: LandmarkDefinition,
        verts: np.ndarray,
    ) -> Tuple[Optional[Vec3], float]:
        """Find extremal point along a specific axis."""
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min

        lt = defn.landmark_type

        # Pronasale: most anterior point in the nasal region
        if lt == LandmarkType.PRONASALE:
            # Filter to nasal region (center ± 15% of width, upper 40-60% of height)
            mask = (
                (np.abs(verts[:, 0] - bbox_center[0]) < bbox_size[0] * 0.15)
                & (verts[:, 1] > bbox_center[1] - bbox_size[1] * 0.1)
                & (verts[:, 1] < bbox_center[1] + bbox_size[1] * 0.1)
            )
            if not mask.any():
                return None, 0.0
            # Most anterior = max Y (assuming Y is anterior-posterior)
            candidates = verts[mask]
            idx = candidates[:, 1].argmin()  # most anterior may be min Y depending on orientation
            # Use the point most protruding
            idx = np.argmax(candidates[:, 2]) if candidates[:, 2].std() > candidates[:, 1].std() else np.argmin(candidates[:, 1])
            pos = candidates[idx]
            return Vec3(float(pos[0]), float(pos[1]), float(pos[2])), 0.7

        # Glabella: most prominent midline point in superior region
        elif lt == LandmarkType.GLABELLA:
            mask = (
                (np.abs(verts[:, 0] - bbox_center[0]) < bbox_size[0] * 0.1)
                & (verts[:, 2] > bbox_center[2] + bbox_size[2] * 0.2)
            )
            if not mask.any():
                return None, 0.0
            candidates = verts[mask]
            idx = np.argmin(candidates[:, 1])
            pos = candidates[idx]
            return Vec3(float(pos[0]), float(pos[1]), float(pos[2])), 0.6

        # Menton: lowest chin point
        elif lt == LandmarkType.MENTON:
            mask = np.abs(verts[:, 0] - bbox_center[0]) < bbox_size[0] * 0.1
            if not mask.any():
                return None, 0.0
            candidates = verts[mask]
            idx = np.argmin(candidates[:, 2])
            pos = candidates[idx]
            return Vec3(float(pos[0]), float(pos[1]), float(pos[2])), 0.7

        # Pogonion: most anterior chin point
        elif lt == LandmarkType.POGONION:
            mask = (
                (np.abs(verts[:, 0] - bbox_center[0]) < bbox_size[0] * 0.1)
                & (verts[:, 2] < bbox_center[2] - bbox_size[2] * 0.15)
            )
            if not mask.any():
                return None, 0.0
            candidates = verts[mask]
            idx = np.argmin(candidates[:, 1])
            pos = candidates[idx]
            return Vec3(float(pos[0]), float(pos[1]), float(pos[2])), 0.6

        # Generic extremal: default to most prominent point
        else:
            idx = np.argmin(verts[:, 1])
            pos = verts[idx]
            return Vec3(float(pos[0]), float(pos[1]), float(pos[2])), 0.3

    def _detect_curvature(
        self,
        defn: LandmarkDefinition,
        verts: np.ndarray,
        curvatures: np.ndarray,
    ) -> Tuple[Optional[Vec3], float]:
        """Find landmark at curvature extremal in a region."""
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min

        lt = defn.landmark_type

        # Nasion: deepest concavity on midline at bridge level
        if lt == LandmarkType.NASION:
            mask = (
                (np.abs(verts[:, 0] - bbox_center[0]) < bbox_size[0] * 0.08)
                & (verts[:, 2] > bbox_center[2] + bbox_size[2] * 0.1)
                & (verts[:, 2] < bbox_center[2] + bbox_size[2] * 0.3)
            )
            if not mask.any():
                return None, 0.0
            candidates = verts[mask]
            curvs = curvatures[mask]
            # Deepest concavity = most negative curvature
            if len(curvs[curvs < 0]) > 0:
                neg_mask = curvs < 0
                idx = np.argmin(curvs[neg_mask])
                pos = candidates[neg_mask][idx]
            else:
                idx = np.argmin(curvs)
                pos = candidates[idx]
            return Vec3(float(pos[0]), float(pos[1]), float(pos[2])), 0.65

        # Subnasale: concavity at columella-lip junction
        elif lt == LandmarkType.SUBNASALE:
            mask = (
                (np.abs(verts[:, 0] - bbox_center[0]) < bbox_size[0] * 0.08)
                & (verts[:, 2] > bbox_center[2] - bbox_size[2] * 0.05)
                & (verts[:, 2] < bbox_center[2] + bbox_size[2] * 0.05)
            )
            if not mask.any():
                return None, 0.0
            candidates = verts[mask]
            curvs = curvatures[mask]
            if len(curvs[curvs < 0]) > 0:
                neg_mask = curvs < 0
                idx = np.argmin(curvs[neg_mask])
                pos = candidates[neg_mask][idx]
            else:
                idx = np.argmin(curvs)
                pos = candidates[idx]
            return Vec3(float(pos[0]), float(pos[1]), float(pos[2])), 0.6

        # Sellion
        elif lt == LandmarkType.SELLION:
            return self._detect_curvature(
                LandmarkDefinition(LandmarkType.NASION, defn.name, defn.description,
                                   "curvature", defn.structure_context),
                verts, curvatures,
            )

        # Default curvature detection
        else:
            # Find highest absolute curvature
            idx = np.argmax(np.abs(curvatures))
            pos = verts[idx]
            return Vec3(float(pos[0]), float(pos[1]), float(pos[2])), 0.3

    # ── Volume-based detection ────────────────────────────────

    def _detect_from_volume_single(
        self,
        defn: LandmarkDefinition,
        volume_hu: np.ndarray,
        spacing: Tuple[float, float, float],
        labels: Optional[np.ndarray],
    ) -> Tuple[Optional[Vec3], float]:
        """Detect a single landmark from volumetric data."""
        sz, sy, sx = spacing
        lt = defn.landmark_type

        # ANS: most anterior bone point on midline at nasal floor level
        if lt == LandmarkType.ANS:
            bone_mask = volume_hu > 300
            dz, dy, dx = bone_mask.shape
            midline = dx // 2
            search_mask = bone_mask.copy()
            search_mask[:, :, :max(0, midline - 5)] = False
            search_mask[:, :, min(dx, midline + 5):] = False

            if not search_mask.any():
                return None, 0.0

            coords = np.argwhere(search_mask)
            # Most anterior = minimum row index (depends on orientation)
            idx = np.argmin(coords[:, 1])
            voxel = coords[idx]
            pos = Vec3(
                float(voxel[2] * sx),
                float(voxel[1] * sy),
                float(voxel[0] * sz),
            )
            return pos, 0.6

        # PNS: most posterior bone point on palate
        elif lt == LandmarkType.PNS:
            bone_mask = volume_hu > 300
            dz, dy, dx = bone_mask.shape
            midline = dx // 2
            search_mask = bone_mask.copy()
            search_mask[:, :, :max(0, midline - 5)] = False
            search_mask[:, :, min(dx, midline + 5):] = False
            # Lower half of volume
            search_mask[:dz // 2, :, :] = False

            if not search_mask.any():
                return None, 0.0

            coords = np.argwhere(search_mask)
            idx = np.argmax(coords[:, 1])
            voxel = coords[idx]
            pos = Vec3(
                float(voxel[2] * sx),
                float(voxel[1] * sy),
                float(voxel[0] * sz),
            )
            return pos, 0.5

        return None, 0.0

    # ── Curvature computation ─────────────────────────────────

    @staticmethod
    def _compute_vertex_curvatures(mesh: SurfaceMesh) -> np.ndarray:
        """Compute mean curvature at each vertex using discrete Laplacian.

        Uses cotangent weights for more accurate curvature estimation.
        """
        verts = mesh.vertices.astype(np.float64)
        faces = mesh.faces
        n_verts = len(verts)

        # Accumulate cotangent Laplacian
        laplacian = np.zeros((n_verts, 3), dtype=np.float64)
        area = np.zeros(n_verts, dtype=np.float64)

        for face in faces:
            i, j, k = face
            vi, vj, vk = verts[i], verts[j], verts[k]

            # Edge vectors
            eij = vj - vi
            eik = vk - vi
            ejk = vk - vj

            # Face area
            face_area = 0.5 * np.linalg.norm(np.cross(eij, eik))
            if face_area < 1e-12:
                continue

            # Cotangent weights
            def _cot(a: np.ndarray, b: np.ndarray) -> float:
                cross = np.linalg.norm(np.cross(a, b))
                if cross < 1e-12:
                    return 0.0
                return float(np.dot(a, b) / cross)

            cot_i = _cot(eij, eik)
            cot_j = _cot(-eij, ejk)
            cot_k = _cot(-eik, -ejk)

            # Accumulate
            laplacian[i] += cot_j * (vk - vi) + cot_k * (vj - vi)
            laplacian[j] += cot_i * (vk - vj) + cot_k * (vi - vj)
            laplacian[k] += cot_i * (vj - vk) + cot_j * (vi - vk)

            area[i] += face_area / 3
            area[j] += face_area / 3
            area[k] += face_area / 3

        # Normalize by area
        area = np.maximum(area, 1e-12)
        laplacian /= (2 * area[:, None])

        # Mean curvature = 0.5 * |Laplacian|
        # Sign from dot product with normal
        curvature_magnitude = 0.5 * np.linalg.norm(laplacian, axis=1)

        if mesh.normals is not None:
            normals = mesh.normals.astype(np.float64)
            signs = np.sign(np.sum(laplacian * normals, axis=1))
            curvatures = curvature_magnitude * signs
        else:
            curvatures = curvature_magnitude

        return curvatures.astype(np.float32)
