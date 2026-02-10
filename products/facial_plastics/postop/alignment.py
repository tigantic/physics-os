"""Outcome alignment — register predicted outcomes to actual results.

Core operations:
  - Rigid alignment (ICP) of prediction mesh to post-op scan
  - Landmark-based registration as initial guess
  - Surface-to-surface distance computation (Hausdorff, RMS, percentile)
  - Per-vertex deviation map (signed + unsigned)
  - Regional deviation analysis (dorsum, tip, alar, columella, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree as _KDTree

from ..core.types import (
    LandmarkType,
    SurfaceMesh,
    Vec3,
)

logger = logging.getLogger(__name__)


# ── Anatomical region definitions ───────────────────────────────────────────

# Mapping from region name to the landmark cluster that defines it.
# Region membership for vertices is determined by proximity to these landmarks.
NASAL_REGIONS: Dict[str, List[LandmarkType]] = {
    "dorsum": [
        LandmarkType.NASION,
        LandmarkType.RHINION,
    ],
    "tip": [
        LandmarkType.PRONASALE,
        LandmarkType.TIP_DEFINING_POINT_LEFT,
        LandmarkType.TIP_DEFINING_POINT_RIGHT,
    ],
    "alar_left": [
        LandmarkType.ALAR_RIM_LEFT,
        LandmarkType.ALAR_CREASE_LEFT,
    ],
    "alar_right": [
        LandmarkType.ALAR_RIM_RIGHT,
        LandmarkType.ALAR_CREASE_RIGHT,
    ],
    "columella": [
        LandmarkType.SUBNASALE,
        LandmarkType.COLUMELLA_BREAKPOINT,
    ],
    "radix": [
        LandmarkType.NASION,
        LandmarkType.SELLION,
    ],
}


@dataclass
class RegionalDeviation:
    """Deviation statistics for an anatomical region."""
    region: str
    mean_unsigned_mm: float
    rms_mm: float
    max_unsigned_mm: float
    mean_signed_mm: float
    percentile_95_mm: float
    n_vertices: int


@dataclass
class AlignmentResult:
    """Complete result of aligning prediction to actual outcome."""
    case_id: str
    timepoint: str

    # Transformation
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0

    # Global metrics
    rms_distance_mm: float = 0.0
    hausdorff_distance_mm: float = 0.0
    mean_distance_mm: float = 0.0
    percentile_95_mm: float = 0.0

    # Per-vertex deviation
    signed_distances: Optional[np.ndarray] = None
    unsigned_distances: Optional[np.ndarray] = None

    # Regional analysis
    regional_deviations: List[RegionalDeviation] = field(default_factory=list)

    # ICP convergence
    icp_iterations: int = 0
    icp_converged: bool = False
    icp_residual: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "timepoint": self.timepoint,
            "rms_distance_mm": self.rms_distance_mm,
            "hausdorff_distance_mm": self.hausdorff_distance_mm,
            "mean_distance_mm": self.mean_distance_mm,
            "percentile_95_mm": self.percentile_95_mm,
            "icp_iterations": self.icp_iterations,
            "icp_converged": self.icp_converged,
            "icp_residual": self.icp_residual,
            "regional_deviations": [
                {
                    "region": rd.region,
                    "mean_unsigned_mm": rd.mean_unsigned_mm,
                    "rms_mm": rd.rms_mm,
                    "max_unsigned_mm": rd.max_unsigned_mm,
                    "mean_signed_mm": rd.mean_signed_mm,
                    "percentile_95_mm": rd.percentile_95_mm,
                    "n_vertices": rd.n_vertices,
                }
                for rd in self.regional_deviations
            ],
        }

    @property
    def clinical_acceptable(self) -> bool:
        """Check if overall deviation is within clinical acceptability (<2mm RMS)."""
        return self.rms_distance_mm < 2.0


class OutcomeAligner:
    """Align predicted post-op mesh to actual outcome scan.

    Pipeline:
      1. Optional landmark-based initial alignment
      2. ICP rigid registration
      3. Signed distance computation
      4. Global + regional deviation analysis
    """

    def __init__(
        self,
        max_icp_iterations: int = 100,
        icp_tolerance: float = 1e-6,
        allow_scaling: bool = False,
    ) -> None:
        self._max_icp = max_icp_iterations
        self._icp_tol = icp_tolerance
        self._allow_scaling = allow_scaling

    def align(
        self,
        predicted_mesh: SurfaceMesh,
        actual_mesh: SurfaceMesh,
        case_id: str,
        timepoint: str,
        *,
        predicted_landmarks: Optional[Dict[LandmarkType, Vec3]] = None,
        actual_landmarks: Optional[Dict[LandmarkType, Vec3]] = None,
    ) -> AlignmentResult:
        """Run alignment pipeline."""
        pred_verts = np.array(predicted_mesh.vertices, dtype=np.float64)
        actual_verts = np.array(actual_mesh.vertices, dtype=np.float64)

        result = AlignmentResult(case_id=case_id, timepoint=timepoint)

        # Step 1: Landmark-based initial alignment
        if predicted_landmarks and actual_landmarks:
            R_init, t_init, s_init = self._landmark_alignment(
                predicted_landmarks, actual_landmarks,
            )
            pred_verts = (s_init * (R_init @ pred_verts.T)).T + t_init
            result.rotation = R_init
            result.translation = t_init
            result.scale = s_init

        # Step 2: ICP refinement
        R_icp, t_icp, icp_iters, converged, residual = self._icp(
            pred_verts, actual_verts,
        )
        pred_verts = (R_icp @ pred_verts.T).T + t_icp

        result.rotation = R_icp @ result.rotation
        result.translation = R_icp @ result.translation + t_icp
        result.icp_iterations = icp_iters
        result.icp_converged = converged
        result.icp_residual = residual

        # Step 3: Compute distances
        signed, unsigned = self._compute_distances(
            pred_verts, actual_verts, actual_mesh,
        )
        result.signed_distances = signed
        result.unsigned_distances = unsigned

        # Step 4: Global statistics
        result.rms_distance_mm = float(np.sqrt(np.mean(unsigned ** 2)))
        result.hausdorff_distance_mm = float(np.max(unsigned))
        result.mean_distance_mm = float(np.mean(unsigned))
        result.percentile_95_mm = float(np.percentile(unsigned, 95))

        # Step 5: Regional analysis
        if predicted_landmarks:
            aligned_landmarks = self._transform_landmarks(
                predicted_landmarks, result.rotation,
                result.translation, result.scale,
            )
            result.regional_deviations = self._regional_analysis(
                pred_verts, signed, unsigned, aligned_landmarks,
            )

        logger.info(
            "Alignment complete: RMS=%.3f mm, Hausdorff=%.3f mm, ICP iters=%d",
            result.rms_distance_mm, result.hausdorff_distance_mm,
            result.icp_iterations,
        )

        return result

    def _landmark_alignment(
        self,
        pred_lm: Dict[LandmarkType, Vec3],
        actual_lm: Dict[LandmarkType, Vec3],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute rigid (+ optional scale) alignment from paired landmarks."""
        common = set(pred_lm.keys()) & set(actual_lm.keys())
        if len(common) < 3:
            logger.warning(
                "Only %d common landmarks; need at least 3 for alignment",
                len(common),
            )
            return np.eye(3), np.zeros(3), 1.0

        common_list = sorted(common, key=lambda x: x.value)
        P = np.array([pred_lm[k].to_array() for k in common_list], dtype=np.float64)
        Q = np.array([actual_lm[k].to_array() for k in common_list], dtype=np.float64)

        p_centroid = P.mean(axis=0)
        q_centroid = Q.mean(axis=0)
        P_c = P - p_centroid
        Q_c = Q - q_centroid

        # Scale factor
        if self._allow_scaling:
            scale = float(np.sqrt(np.sum(Q_c ** 2) / max(np.sum(P_c ** 2), 1e-30)))
        else:
            scale = 1.0

        # SVD for rotation
        H = (scale * P_c).T @ Q_c
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        S = np.eye(3)
        if d < 0:
            S[2, 2] = -1
        R = Vt.T @ S @ U.T

        t = q_centroid - R @ (scale * p_centroid)

        return R, t, scale

    def _icp(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int, bool, float]:
        """Iterative Closest Point registration.

        Uses point-to-point ICP with SVD-based rotation estimation.
        """
        src = source.copy()
        R_total = np.eye(3)
        t_total = np.zeros(3)
        prev_error = float("inf")

        for iteration in range(self._max_icp):
            # Find nearest neighbors (brute force — fine up to ~50k vertices)
            correspondences = self._find_nearest(src, target)
            matched_target = target[correspondences]

            # Compute centroids
            src_centroid = src.mean(axis=0)
            tgt_centroid = matched_target.mean(axis=0)

            src_c = src - src_centroid
            tgt_c = matched_target - tgt_centroid

            # SVD
            H = src_c.T @ tgt_c
            U, _, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            S = np.eye(3)
            if d < 0:
                S[2, 2] = -1
            R = Vt.T @ S @ U.T
            t = tgt_centroid - R @ src_centroid

            # Apply
            src = (R @ src.T).T + t
            R_total = R @ R_total
            t_total = R @ t_total + t

            # Check convergence
            residuals = np.linalg.norm(src - matched_target, axis=1)
            error = float(np.mean(residuals))

            if abs(prev_error - error) < self._icp_tol:
                return R_total, t_total, iteration + 1, True, error

            prev_error = error

        return R_total, t_total, self._max_icp, False, prev_error

    @staticmethod
    def _find_nearest(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Find index of nearest target point for each source point.

        Uses scipy KDTree for O(n log m) performance.
        """
        tree = _KDTree(target)
        _, indices = tree.query(source)
        return np.asarray(indices, dtype=np.int64)

    def _compute_distances(
        self,
        aligned_pred: np.ndarray,
        actual_verts: np.ndarray,
        actual_mesh: SurfaceMesh,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute signed and unsigned point-to-surface distances.

        For signed distances, uses vertex normals on the actual mesh to
        determine inside/outside.
        """
        # Point-to-point distances as baseline
        correspondences = self._find_nearest(aligned_pred, actual_verts)
        closest_pts = actual_verts[correspondences]

        displacement = aligned_pred - closest_pts
        unsigned = np.linalg.norm(displacement, axis=1)

        # Estimate normals on actual mesh for sign determination
        normals = self._estimate_vertex_normals(actual_verts, actual_mesh)

        if normals is not None:
            matched_normals = normals[correspondences]
            signs = np.sign(np.sum(displacement * matched_normals, axis=1))
            signs[signs == 0] = 1.0
            signed = unsigned * signs
        else:
            signed = unsigned.copy()

        return signed, unsigned

    @staticmethod
    def _estimate_vertex_normals(
        vertices: np.ndarray,
        mesh: SurfaceMesh,
    ) -> Optional[np.ndarray]:
        """Estimate per-vertex normals from triangle faces."""
        if mesh.triangles is None or len(mesh.triangles) == 0:
            return None

        normals = np.zeros_like(vertices)
        triangles = np.array(mesh.triangles, dtype=np.int64)

        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            face_normal = np.cross(e1, e2)
            normals[tri[0]] += face_normal
            normals[tri[1]] += face_normal
            normals[tri[2]] += face_normal

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normals /= norms

        return normals

    def _transform_landmarks(
        self,
        landmarks: Dict[LandmarkType, Vec3],
        R: np.ndarray,
        t: np.ndarray,
        s: float,
    ) -> Dict[LandmarkType, Vec3]:
        """Apply rigid transformation to landmarks."""
        result: Dict[LandmarkType, Vec3] = {}
        for lm_type, pos in landmarks.items():
            p = pos.to_array() if hasattr(pos, 'to_array') else np.array(pos, dtype=np.float64)
            transformed = R @ (s * p) + t
            result[lm_type] = Vec3.from_array(transformed)
        return result

    def _regional_analysis(
        self,
        vertices: np.ndarray,
        signed: np.ndarray,
        unsigned: np.ndarray,
        landmarks: Dict[LandmarkType, Vec3],
    ) -> List[RegionalDeviation]:
        """Compute per-region deviation statistics."""
        # Assign each vertex to nearest region
        region_assignments = self._assign_regions(vertices, landmarks)
        deviations: List[RegionalDeviation] = []

        for region_name in NASAL_REGIONS:
            mask = region_assignments == region_name
            if not np.any(mask):
                continue

            region_unsigned = unsigned[mask]
            region_signed = signed[mask]

            deviations.append(RegionalDeviation(
                region=region_name,
                mean_unsigned_mm=float(np.mean(region_unsigned)),
                rms_mm=float(np.sqrt(np.mean(region_unsigned ** 2))),
                max_unsigned_mm=float(np.max(region_unsigned)),
                mean_signed_mm=float(np.mean(region_signed)),
                percentile_95_mm=float(np.percentile(region_unsigned, 95)),
                n_vertices=int(np.sum(mask)),
            ))

        return deviations

    def _assign_regions(
        self,
        vertices: np.ndarray,
        landmarks: Dict[LandmarkType, Vec3],
    ) -> np.ndarray:
        """Assign each vertex to its nearest anatomical region."""
        n_verts = vertices.shape[0]
        assignments = np.full(n_verts, "unassigned", dtype=object)

        # Build region centroids from landmarks
        region_centroids: Dict[str, np.ndarray] = {}
        for region_name, lm_types in NASAL_REGIONS.items():
            pts = []
            for lt in lm_types:
                if lt in landmarks:
                    lm = landmarks[lt]
                    pts.append(lm.to_array() if hasattr(lm, 'to_array') else np.array(lm, dtype=np.float64))
            if pts:
                region_centroids[region_name] = np.mean(pts, axis=0)

        if not region_centroids:
            return assignments

        # Assign each vertex to nearest region centroid
        names = list(region_centroids.keys())
        centroids = np.array([region_centroids[n] for n in names])

        for i in range(n_verts):
            dists = np.linalg.norm(centroids - vertices[i], axis=1)
            assignments[i] = names[int(np.argmin(dists))]

        return assignments
