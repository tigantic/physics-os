"""Multi-modal registration — CT ↔ surface scan ↔ photography.

Provides rigid (6-DOF) and deformable registration for aligning
multiple data sources into a common coordinate frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import (
    Landmark,
    RegistrationResult,
    SurfaceMesh,
    Vec3,
)

logger = logging.getLogger(__name__)


@dataclass
class ICPConfig:
    """Configuration for Iterative Closest Point registration."""
    max_iterations: int = 200
    convergence_tol: float = 1e-6
    correspondence_threshold_mm: float = 10.0
    use_trimmed: bool = True
    trim_fraction: float = 0.9  # keep 90% closest pairs
    point_to_plane: bool = True


class MultiModalRegistrar:
    """Register multi-modal data into a common coordinate frame.

    Methods:
      - Landmark-based rigid registration (SVD)
      - Surface-based ICP (point-to-point and point-to-plane)
      - Thin-plate spline deformable registration
      - Combined landmark + ICP pipeline
    """

    def __init__(self, icp_config: Optional[ICPConfig] = None) -> None:
        self._icp_cfg = icp_config or ICPConfig()

    # ── Landmark-based rigid registration ─────────────────────

    def register_landmarks(
        self,
        source_landmarks: List[Landmark],
        target_landmarks: List[Landmark],
    ) -> RegistrationResult:
        """Compute rigid transform aligning source landmarks to target.

        Uses SVD-based optimal rigid registration (Arun et al. 1987).
        Requires ≥ 3 corresponding landmark pairs.
        """
        if len(source_landmarks) < 3 or len(target_landmarks) < 3:
            raise ValueError("Need ≥ 3 landmark pairs for rigid registration")

        # Match landmarks by name
        src_pts = []
        tgt_pts = []
        for sl in source_landmarks:
            for tl in target_landmarks:
                if sl.name == tl.name:
                    src_pts.append([sl.position.x, sl.position.y, sl.position.z])
                    tgt_pts.append([tl.position.x, tl.position.y, tl.position.z])
                    break

        if len(src_pts) < 3:
            raise ValueError(
                f"Only {len(src_pts)} matching landmarks found, need ≥ 3"
            )

        P = np.array(src_pts, dtype=np.float64)
        Q = np.array(tgt_pts, dtype=np.float64)

        rotation, translation = self._svd_rigid(P, Q)

        # Compute residuals
        P_transformed = (rotation @ P.T).T + translation
        residuals = np.linalg.norm(P_transformed - Q, axis=1)
        rms_error = float(np.sqrt(np.mean(residuals ** 2)))

        # Build 4×4 transform
        rigid_transform = np.eye(4, dtype=np.float64)
        rigid_transform[:3, :3] = rotation
        rigid_transform[:3, 3] = translation

        logger.info(
            "Landmark registration: %d pairs, RMS=%.3f mm",
            len(src_pts), rms_error,
        )

        return RegistrationResult(
            rigid_transform=rigid_transform,
            nonrigid_field=None,
            rms_error_mm=rms_error,
            n_correspondences=len(src_pts),
        )

    # ── Surface ICP registration ──────────────────────────────

    def register_surfaces(
        self,
        source: SurfaceMesh,
        target: SurfaceMesh,
        *,
        initial_transform: Optional[np.ndarray] = None,
    ) -> RegistrationResult:
        """Register source surface to target using ICP.

        Parameters
        ----------
        source : SurfaceMesh
            Source mesh to transform.
        target : SurfaceMesh
            Fixed target mesh.
        initial_transform : ndarray (4,4), optional
            Initial rigid transform (e.g. from landmarks).
        """
        cfg = self._icp_cfg
        src_pts = source.vertices.astype(np.float64).copy()

        # Apply initial transform if provided
        if initial_transform is not None:
            R0 = initial_transform[:3, :3]
            t0 = initial_transform[:3, 3]
            src_pts = (R0 @ src_pts.T).T + t0

        tgt_pts = target.vertices.astype(np.float64)

        # Precompute target normals for point-to-plane
        tgt_normals = None
        if cfg.point_to_plane and target.normals is not None:
            tgt_normals = target.normals.astype(np.float64)

        # Build KD-tree equivalent (brute force with chunking for pure numpy)
        cumulative_R = np.eye(3, dtype=np.float64)
        cumulative_t = np.zeros(3, dtype=np.float64)

        prev_error = float("inf")

        for iteration in range(cfg.max_iterations):
            # Find correspondences
            correspondences, distances = self._find_correspondences(
                src_pts, tgt_pts, cfg.correspondence_threshold_mm
            )

            if len(correspondences) < 3:
                logger.warning("ICP: too few correspondences (%d)", len(correspondences))
                break

            # Trim outliers
            if cfg.use_trimmed:
                n_keep = max(3, int(len(correspondences) * cfg.trim_fraction))
                sorted_idx = np.argsort(distances)[:n_keep]
                correspondences = [correspondences[i] for i in sorted_idx]
                distances = distances[sorted_idx]

            src_corr = np.array([src_pts[i] for i, _ in correspondences])
            tgt_corr = np.array([tgt_pts[j] for _, j in correspondences])

            if cfg.point_to_plane and tgt_normals is not None:
                # Point-to-plane ICP step
                tgt_norm_corr = np.array([tgt_normals[j] for _, j in correspondences])
                R, t = self._point_to_plane_step(src_corr, tgt_corr, tgt_norm_corr)
            else:
                # Point-to-point ICP step
                R, t = self._svd_rigid(src_corr, tgt_corr)

            # Apply transform
            src_pts = (R @ src_pts.T).T + t
            cumulative_R = R @ cumulative_R
            cumulative_t = R @ cumulative_t + t

            # Check convergence
            rms = float(np.sqrt(np.mean(distances ** 2)))
            delta = abs(prev_error - rms)
            if delta < cfg.convergence_tol:
                logger.info("ICP converged at iteration %d, RMS=%.4f mm", iteration, rms)
                break
            prev_error = rms

        # Build result
        rigid_transform = np.eye(4, dtype=np.float64)
        rigid_transform[:3, :3] = cumulative_R
        rigid_transform[:3, 3] = cumulative_t

        if initial_transform is not None:
            rigid_transform = rigid_transform @ initial_transform

        final_correspondences, final_distances = self._find_correspondences(
            src_pts, tgt_pts, cfg.correspondence_threshold_mm * 2
        )
        rms_error = float(np.sqrt(np.mean(final_distances ** 2))) if len(final_distances) > 0 else float("inf")

        return RegistrationResult(
            rigid_transform=rigid_transform,
            nonrigid_field=None,
            rms_error_mm=rms_error,
            n_correspondences=len(final_correspondences),
        )

    # ── Combined pipeline ─────────────────────────────────────

    def register_ct_to_surface(
        self,
        ct_surface: SurfaceMesh,
        scan_surface: SurfaceMesh,
        ct_landmarks: Optional[List[Landmark]] = None,
        scan_landmarks: Optional[List[Landmark]] = None,
    ) -> RegistrationResult:
        """Full registration pipeline: landmarks → ICP refinement."""
        initial_transform = None

        if ct_landmarks and scan_landmarks:
            lm_result = self.register_landmarks(ct_landmarks, scan_landmarks)
            initial_transform = lm_result.rigid_transform
            logger.info("Landmark pre-alignment: RMS=%.3f mm", lm_result.rms_error_mm)

        result = self.register_surfaces(
            ct_surface, scan_surface, initial_transform=initial_transform
        )
        logger.info("Final registration: RMS=%.3f mm, %d correspondences",
                     result.rms_error_mm, result.n_correspondences)
        return result

    # ── Thin-plate spline deformable registration ─────────────

    def register_deformable(
        self,
        source_landmarks: np.ndarray,
        target_landmarks: np.ndarray,
        source_mesh: SurfaceMesh,
        *,
        lambda_smooth: float = 1e-3,
    ) -> Tuple[SurfaceMesh, np.ndarray]:
        """Apply thin-plate spline warp from source to target landmarks.

        Parameters
        ----------
        source_landmarks : ndarray (K, 3)
            Control points in source space.
        target_landmarks : ndarray (K, 3)
            Corresponding points in target space.
        source_mesh : SurfaceMesh
            Mesh to warp.
        lambda_smooth : float
            Regularization parameter.

        Returns
        -------
        (warped_mesh, displacement_field) where displacement_field is (N, 3).
        """
        K = len(source_landmarks)
        assert len(target_landmarks) == K

        # Build TPS kernel matrix
        # U(r) = r^2 * log(r) for 3D TPS
        def tps_kernel(r: np.ndarray) -> np.ndarray:
            r = np.maximum(r, 1e-10)
            return r ** 2 * np.log(r)

        # Distance matrix between control points
        diff = source_landmarks[:, None, :] - source_landmarks[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=2))
        K_mat = tps_kernel(D)

        # Regularize
        K_mat += lambda_smooth * np.eye(K)

        # Build system: [K P; P^T 0] [w; a] = [target; 0]
        P = np.hstack([np.ones((K, 1)), source_landmarks])  # (K, 4)
        n_sys = K + 4
        A = np.zeros((n_sys, n_sys))
        A[:K, :K] = K_mat
        A[:K, K:] = P
        A[K:, :K] = P.T

        rhs = np.zeros((n_sys, 3))
        rhs[:K] = target_landmarks

        # Solve for each coordinate
        coeffs = np.linalg.solve(A, rhs)  # (K+4, 3)
        weights = coeffs[:K]    # (K, 3)
        affine = coeffs[K:]     # (4, 3)

        # Apply TPS to all mesh vertices
        verts = source_mesh.vertices.astype(np.float64)
        N = len(verts)

        # Distances from each vertex to each control point
        diff = verts[:, None, :] - source_landmarks[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))  # (N, K)
        K_eval = tps_kernel(dists)

        P_eval = np.hstack([np.ones((N, 1)), verts])  # (N, 4)
        warped = K_eval @ weights + P_eval @ affine   # (N, 3)
        displacement = warped - verts

        warped_mesh = SurfaceMesh(
            vertices=warped.astype(np.float32),
            faces=source_mesh.faces.copy(),
        )
        warped_mesh.compute_normals()

        logger.info(
            "TPS warp: %d control points, mean displacement=%.2f mm",
            K, float(np.linalg.norm(displacement, axis=1).mean()),
        )

        return warped_mesh, displacement.astype(np.float32)

    # ── Internal methods ──────────────────────────────────────

    @staticmethod
    def _svd_rigid(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SVD-based optimal rigid registration.

        P (source) → R @ P + t ≈ Q (target)
        """
        centroid_p = P.mean(axis=0)
        centroid_q = Q.mean(axis=0)

        P_centered = P - centroid_p
        Q_centered = Q - centroid_q

        H = P_centered.T @ Q_centered
        U, S, Vt = np.linalg.svd(H)

        # Handle reflection
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1, 1, np.sign(d)])

        R = Vt.T @ sign_matrix @ U.T
        t = centroid_q - R @ centroid_p

        return R, t

    @staticmethod
    def _find_correspondences(
        src: np.ndarray,
        tgt: np.ndarray,
        threshold: float,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Find nearest-neighbor correspondences (chunked for memory)."""
        n_src = len(src)
        n_tgt = len(tgt)
        chunk_size = 1000

        correspondences = []
        distances_list = []

        for start in range(0, n_src, chunk_size):
            end = min(start + chunk_size, n_src)
            chunk = src[start:end]

            # Compute pairwise distances
            diff = chunk[:, None, :] - tgt[None, :, :]
            dist_sq = (diff ** 2).sum(axis=2)
            min_idx = dist_sq.argmin(axis=1)
            min_dist = np.sqrt(dist_sq[np.arange(end - start), min_idx])

            for i, (j, d) in enumerate(zip(min_idx, min_dist)):
                if d < threshold:
                    correspondences.append((start + i, int(j)))
                    distances_list.append(d)

        return correspondences, np.array(distances_list) if distances_list else np.array([])

    @staticmethod
    def _point_to_plane_step(
        src: np.ndarray,
        tgt: np.ndarray,
        normals: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linearized point-to-plane ICP step.

        Minimizes sum_i ( (R @ src_i + t - tgt_i) · n_i )^2
        using the small-angle approximation.
        """
        N = len(src)
        assert len(tgt) == N and len(normals) == N

        # Build linear system: A @ x = b
        # x = [alpha, beta, gamma, tx, ty, tz]
        A = np.zeros((N, 6))
        b = np.zeros(N)

        for i in range(N):
            s = src[i]
            n = normals[i]
            # Cross product s × n
            cn = np.cross(s, n)
            A[i, :3] = cn
            A[i, 3:] = n
            b[i] = np.dot(tgt[i] - s, n)

        # Solve least squares
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        alpha, beta, gamma = x[0], x[1], x[2]
        tx, ty, tz = x[3], x[4], x[5]

        # Small-angle rotation matrix
        R = np.eye(3)
        R[0, 1] = -gamma
        R[0, 2] = beta
        R[1, 0] = gamma
        R[1, 2] = -alpha
        R[2, 0] = -beta
        R[2, 1] = alpha

        t = np.array([tx, ty, tz])
        return R, t
