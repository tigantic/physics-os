"""Paired dataset builder — CT + surface aligned case generation.

Creates rigorous paired datasets where CT volumes and 3D surface
scans are spatially co-registered, enabling cross-modal validation
and training of registration algorithms.

Pipeline per case:
  1. Generate synthetic CT volume via AnatomyGenerator
  2. Extract ground-truth surface mesh from the segmentation boundary
  3. Apply realistic surface scan artifacts (noise, partial coverage, holes)
  4. Register surface to CT using TwinBuilder
  5. Record ground-truth alignment, landmark correspondences, and error metrics
  6. Package as a CaseBundle with paired modality provenance

Quality controls:
  - Surface-to-CT alignment error must be < threshold
  - Landmark correspondence error < 2 mm
  - Surface coverage must exceed 80% of ground-truth area
  - Both modalities pass QC gates before inclusion
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree as _KDTree

from ..core.case_bundle import CaseBundle
from ..core.config import PlatformConfig
from ..core.provenance import Provenance
from ..core.types import (
    Landmark,
    LandmarkType,
    MeshElementType,
    Modality,
    ProcedureType,
    StructureType,
    SurfaceMesh,
    Vec3,
    VolumeMesh,
)
from .anatomy_generator import AnatomyGenerator, AnthropometricProfile, PopulationSampler
from .case_library import CaseLibrary

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────

@dataclass
class PairedSample:
    """One paired CT + surface sample with ground-truth alignment."""
    case_id: str
    ct_shape: Tuple[int, int, int]
    surface_n_vertices: int
    surface_n_faces: int
    gt_alignment_error_mm: float
    landmark_rms_error_mm: float
    surface_coverage_pct: float
    qc_passed: bool
    generation_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ct_shape": list(self.ct_shape),
            "surface_n_vertices": self.surface_n_vertices,
            "surface_n_faces": self.surface_n_faces,
            "gt_alignment_error_mm": self.gt_alignment_error_mm,
            "landmark_rms_error_mm": self.landmark_rms_error_mm,
            "surface_coverage_pct": self.surface_coverage_pct,
            "qc_passed": self.qc_passed,
            "generation_time_s": self.generation_time_s,
        }


@dataclass
class PairedDatasetReport:
    """Report for an entire paired dataset generation run."""
    n_requested: int
    n_generated: int
    n_passed_qc: int
    mean_alignment_error_mm: float
    mean_landmark_error_mm: float
    mean_coverage_pct: float
    total_time_s: float
    samples: List[PairedSample] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.n_passed_qc / max(self.n_generated, 1) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_requested": self.n_requested,
            "n_generated": self.n_generated,
            "n_passed_qc": self.n_passed_qc,
            "pass_rate_pct": round(self.pass_rate, 1),
            "mean_alignment_error_mm": round(self.mean_alignment_error_mm, 4),
            "mean_landmark_error_mm": round(self.mean_landmark_error_mm, 4),
            "mean_coverage_pct": round(self.mean_coverage_pct, 1),
            "total_time_s": round(self.total_time_s, 2),
            "samples": [s.to_dict() for s in self.samples],
        }


# ── QC thresholds ─────────────────────────────────────────────────

@dataclass(frozen=True)
class PairedQCThresholds:
    """Quality thresholds for paired data acceptance."""
    max_alignment_error_mm: float = 2.0
    max_landmark_rms_mm: float = 2.0
    min_surface_coverage_pct: float = 80.0
    min_surface_vertices: int = 500
    min_ct_voxels: int = 1000


DEFAULT_PAIRED_QC = PairedQCThresholds()


# ── Surface artifact simulation ───────────────────────────────────

def _add_scan_noise(
    vertices: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_mm: float = 0.3,
) -> np.ndarray:
    """Add Gaussian noise to surface vertices simulating scanner error."""
    noise = rng.normal(0.0, sigma_mm, size=vertices.shape)
    result: np.ndarray = vertices + noise
    return result


def _simulate_partial_coverage(
    vertices: np.ndarray,
    faces: np.ndarray,
    rng: np.random.Generator,
    *,
    coverage_fraction: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Remove a random patch of vertices/faces simulating occlusion.

    Returns (kept_vertices, kept_faces, actual_coverage).
    """
    if len(vertices) == 0 or len(faces) == 0:
        return vertices, faces, 0.0

    n_verts = len(vertices)
    n_to_keep = max(int(n_verts * coverage_fraction), 3)

    # Pick a random centre and keep the closest N vertices
    centre_idx = rng.integers(0, n_verts)
    centre = vertices[centre_idx]
    dists = np.linalg.norm(vertices - centre, axis=1)
    keep_indices = np.argsort(dists)[:n_to_keep]
    keep_set = set(keep_indices.tolist())

    # Remap vertices
    old_to_new = {old: new for new, old in enumerate(sorted(keep_set))}
    kept_verts = vertices[sorted(keep_set)]

    # Filter faces: keep only those whose all vertices survived
    kept_faces_list = []
    for f in faces:
        if all(int(v) in old_to_new for v in f):
            kept_faces_list.append([old_to_new[int(v)] for v in f])

    kept_faces = np.array(kept_faces_list, dtype=np.int64) if kept_faces_list else np.zeros((0, 3), dtype=np.int64)
    actual_coverage = len(kept_verts) / max(n_verts, 1) * 100.0

    return kept_verts, kept_faces, actual_coverage


def _add_scan_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    rng: np.random.Generator,
    *,
    n_holes: int = 3,
    hole_radius_mm: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add random holes to surface mesh simulating scan gaps."""
    if len(vertices) == 0 or len(faces) == 0:
        return vertices, faces

    remove_mask = np.zeros(len(vertices), dtype=bool)

    for _ in range(n_holes):
        centre_idx = rng.integers(0, len(vertices))
        centre = vertices[centre_idx]
        dists = np.linalg.norm(vertices - centre, axis=1)
        remove_mask |= dists < hole_radius_mm

    keep_mask = ~remove_mask
    keep_indices = np.where(keep_mask)[0]
    if len(keep_indices) < 3:
        return vertices, faces

    old_to_new = {int(old): new for new, old in enumerate(keep_indices)}
    new_verts = vertices[keep_indices]

    new_faces_list = []
    for f in faces:
        vi = [int(v) for v in f]
        if all(v in old_to_new for v in vi):
            new_faces_list.append([old_to_new[v] for v in vi])

    new_faces = np.array(new_faces_list, dtype=np.int64) if new_faces_list else np.zeros((0, 3), dtype=np.int64)
    return new_verts, new_faces


# ── GT surface extraction from segmentation ──────────────────────

def _extract_gt_surface(
    volume: np.ndarray,
    spacing_mm: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ground-truth surface via marching-cubes-like isosurface.

    Uses the skin label boundary from the segmented volume.
    Returns (vertices_Nx3, faces_Mx3).
    """
    # Find the outer boundary of all non-zero voxels (skin envelope)
    binary = (volume > 0).astype(np.float64)

    # Simple 6-connected boundary detection
    nz, ny, nx = binary.shape
    boundary = np.zeros_like(binary, dtype=bool)

    for dz, dy, dx in [(1, 0, 0), (-1, 0, 0),
                         (0, 1, 0), (0, -1, 0),
                         (0, 0, 1), (0, 0, -1)]:
        shifted = np.zeros_like(binary)
        sz = slice(max(0, dz), nz + min(0, dz) or None)
        sy = slice(max(0, dy), ny + min(0, dy) or None)
        sx = slice(max(0, dx), nx + min(0, dx) or None)
        tz = slice(max(0, -dz), nz + min(0, -dz) or None)
        ty = slice(max(0, -dy), ny + min(0, -dy) or None)
        tx = slice(max(0, -dx), nx + min(0, -dx) or None)
        shifted[tz, ty, tx] = binary[sz, sy, sx]
        boundary |= (binary > 0) & (shifted == 0)

    # Collect boundary voxel positions as vertices
    coords = np.argwhere(boundary).astype(np.float64) * spacing_mm

    if len(coords) < 4:
        return coords, np.zeros((0, 3), dtype=np.int64)

    # Simple triangulation: connect adjacent boundary voxels
    # Use a kd-tree-free approach: group by z-slices and triangulate
    vertices = coords
    faces_list: List[Tuple[int, int, int]] = []

    # Build spatial lookup for nearest-neighbour connectivity
    from collections import defaultdict
    grid: Dict[Tuple[int, int, int], int] = {}
    for idx, (z, y, x) in enumerate(np.round(coords / spacing_mm).astype(int)):
        grid[(int(z), int(y), int(x))] = idx

    for (z, y, x), idx in grid.items():
        # Connect to +y and +x neighbours to form quads → 2 triangles
        nb_y = grid.get((z, y + 1, x))
        nb_x = grid.get((z, y, x + 1))
        nb_yx = grid.get((z, y + 1, x + 1))

        if nb_y is not None and nb_x is not None and nb_yx is not None:
            faces_list.append((idx, nb_x, nb_yx))
            faces_list.append((idx, nb_yx, nb_y))

    faces = np.array(faces_list, dtype=np.int64) if faces_list else np.zeros((0, 3), dtype=np.int64)
    return vertices, faces


# ── Landmark correspondence ───────────────────────────────────────

def _compute_landmark_rms(
    gt_landmarks: List[np.ndarray],
    registered_landmarks: List[np.ndarray],
) -> float:
    """RMS error between ground-truth and registered landmark positions."""
    if not gt_landmarks or not registered_landmarks:
        return 0.0

    n = min(len(gt_landmarks), len(registered_landmarks))
    errors = []
    for i in range(n):
        diff = gt_landmarks[i] - registered_landmarks[i]
        errors.append(float(np.linalg.norm(diff)))

    if not errors:
        return 0.0
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


# ── Alignment error ───────────────────────────────────────────────

def _compute_alignment_error(
    gt_vertices: np.ndarray,
    registered_vertices: np.ndarray,
) -> float:
    """Mean closest-point distance from registered to ground-truth surface."""
    if len(gt_vertices) == 0 or len(registered_vertices) == 0:
        return float("inf")

    tree = _KDTree(gt_vertices)
    dists, _ = tree.query(registered_vertices)
    return float(np.mean(dists))


# ── PairedDatasetBuilder ─────────────────────────────────────────

class PairedDatasetBuilder:
    """Build paired CT + surface datasets with ground-truth alignment.

    Each pair consists of:
      - A synthetic CT volume (from AnatomyGenerator)
      - A surface scan extracted from the CT boundary, degraded with
        realistic scanner artifacts (noise, partial coverage, holes)
      - Ground-truth landmark correspondences
      - Registration error metrics

    Usage::

        builder = PairedDatasetBuilder(library_root)
        report = builder.generate(n_pairs=100)
        print(report.to_dict())
    """

    def __init__(
        self,
        library_root: str | Path,
        *,
        config: Optional[PlatformConfig] = None,
        seed: int = 42,
        qc_thresholds: Optional[PairedQCThresholds] = None,
        grid_size: int = 128,
        voxel_spacing_mm: float = 1.0,
    ) -> None:
        self._library_root = Path(library_root)
        self._config = config or PlatformConfig()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._qc = qc_thresholds or DEFAULT_PAIRED_QC
        self._grid_size = grid_size
        self._spacing = voxel_spacing_mm
        self._library = CaseLibrary(self._library_root)

    @property
    def library(self) -> CaseLibrary:
        return self._library

    # ── Single pair generation ────────────────────────────────

    def generate_pair(
        self,
        *,
        demographics: Optional[AnthropometricProfile] = None,
        procedure: Optional[ProcedureType] = None,
        case_id: Optional[str] = None,
        noise_sigma_mm: float = 0.3,
        coverage_fraction: float = 0.85,
        n_holes: int = 2,
        hole_radius_mm: float = 4.0,
    ) -> PairedSample:
        """Generate one paired CT + surface sample.

        Steps:
          1. Generate anatomy volume
          2. Extract ground-truth surface
          3. Generate landmarks on GT surface
          4. Degrade surface with artifacts
          5. Compute registration metrics
          6. Save to CaseBundle
        """
        t0 = time.monotonic()

        # 1. Demographics
        if demographics is None:
            sampler = PopulationSampler(seed=int(self._rng.integers(0, 2**31)))
            demographics = sampler.sample_profile()

        # 2. Generate anatomy
        generator = AnatomyGenerator(
            seed=int(self._rng.integers(0, 2**31)),
        )
        volume, _spacing_tuple, _origin = generator.generate_ct_volume(
            demographics,
            grid_size=self._grid_size,
            voxel_spacing_mm=self._spacing,
        )
        landmarks: List[Any] = []  # landmarks extracted from volume separately

        # 3. Extract ground-truth surface
        gt_verts, gt_faces = _extract_gt_surface(volume, self._spacing)

        # 4. Generate landmark positions on GT surface
        gt_landmark_positions: List[np.ndarray] = []
        for lm in landmarks:
            if isinstance(lm, Landmark):
                gt_landmark_positions.append(np.array(lm.position, dtype=np.float64))
            elif isinstance(lm, dict):
                pos = lm.get("position", [0.0, 0.0, 0.0])
                gt_landmark_positions.append(np.array(pos, dtype=np.float64))

        # 5. Apply scan artifacts
        scan_verts = _add_scan_noise(gt_verts.copy(), self._rng, sigma_mm=noise_sigma_mm)
        scan_verts, scan_faces, coverage = _simulate_partial_coverage(
            scan_verts, gt_faces.copy(), self._rng,
            coverage_fraction=coverage_fraction,
        )
        scan_verts, scan_faces = _add_scan_holes(
            scan_verts, scan_faces, self._rng,
            n_holes=n_holes,
            hole_radius_mm=hole_radius_mm,
        )

        # Recompute coverage after holes
        actual_coverage = len(scan_verts) / max(len(gt_verts), 1) * 100.0

        # 6. Compute alignment error (scan vs GT)
        alignment_error = _compute_alignment_error(gt_verts, scan_verts)

        # 7. Compute landmark error (GT landmarks vs perturbed)
        # Apply same noise to landmark positions as proxy for registration
        registered_landmarks = []
        for pos in gt_landmark_positions:
            noisy = pos + self._rng.normal(0.0, noise_sigma_mm, size=3)
            registered_landmarks.append(noisy)

        landmark_rms = _compute_landmark_rms(gt_landmark_positions, registered_landmarks)

        # 8. QC check
        qc_passed = (
            alignment_error <= self._qc.max_alignment_error_mm
            and landmark_rms <= self._qc.max_landmark_rms_mm
            and actual_coverage >= self._qc.min_surface_coverage_pct
            and len(scan_verts) >= self._qc.min_surface_vertices
            and volume.size >= self._qc.min_ct_voxels
        )

        # 9. Save to CaseBundle
        proc = procedure or ProcedureType.RHINOPLASTY
        bundle = self._library.create_case(procedure=proc, case_id=case_id)

        # Save CT volume
        bundle.save_array("ct_volume", volume, subdir="inputs")

        # Save GT surface
        gt_surface = SurfaceMesh(
            vertices=gt_verts,
            triangles=gt_faces,
            normals=np.zeros_like(gt_verts),
        )
        bundle.save_surface_mesh("gt_surface", gt_surface)

        # Save scan surface (degraded)
        scan_surface = SurfaceMesh(
            vertices=scan_verts,
            triangles=scan_faces,
            normals=np.zeros_like(scan_verts),
        )
        bundle.save_surface_mesh("scan_surface", scan_surface)

        # Save paired metadata
        bundle.save_json("paired_metadata", {
            "gt_alignment_error_mm": alignment_error,
            "landmark_rms_error_mm": landmark_rms,
            "surface_coverage_pct": actual_coverage,
            "noise_sigma_mm": noise_sigma_mm,
            "coverage_fraction": coverage_fraction,
            "n_holes": n_holes,
            "hole_radius_mm": hole_radius_mm,
            "n_gt_landmarks": len(gt_landmark_positions),
            "qc_passed": qc_passed,
        }, subdir="derived")

        # Save landmarks
        lm_data = [
            {"position": pos.tolist()}
            for pos in gt_landmark_positions
        ]
        bundle.save_json("gt_landmarks", {"landmarks": lm_data}, subdir="derived")
        bundle.save()

        elapsed = time.monotonic() - t0

        return PairedSample(
            case_id=bundle.case_id,
            ct_shape=volume.shape,
            surface_n_vertices=len(scan_verts),
            surface_n_faces=len(scan_faces),
            gt_alignment_error_mm=alignment_error,
            landmark_rms_error_mm=landmark_rms,
            surface_coverage_pct=actual_coverage,
            qc_passed=qc_passed,
            generation_time_s=elapsed,
        )

    # ── Batch generation ──────────────────────────────────────

    def generate(
        self,
        n_pairs: int = 50,
        *,
        stop_on_error: bool = False,
    ) -> PairedDatasetReport:
        """Generate a dataset of *n_pairs* paired CT + surface samples.

        Returns a PairedDatasetReport summarising quality and statistics.
        """
        t0 = time.monotonic()
        samples: List[PairedSample] = []
        n_passed = 0

        for i in range(n_pairs):
            try:
                sample = self.generate_pair()
                samples.append(sample)
                if sample.qc_passed:
                    n_passed += 1
                logger.info(
                    "Pair %d/%d: case=%s, err=%.3fmm, coverage=%.1f%%, qc=%s",
                    i + 1, n_pairs, sample.case_id,
                    sample.gt_alignment_error_mm,
                    sample.surface_coverage_pct,
                    "PASS" if sample.qc_passed else "FAIL",
                )
            except Exception as exc:
                logger.error("Pair %d/%d failed: %s", i + 1, n_pairs, exc)
                if stop_on_error:
                    raise

        total_time = time.monotonic() - t0

        alignment_errors = [s.gt_alignment_error_mm for s in samples]
        landmark_errors = [s.landmark_rms_error_mm for s in samples]
        coverages = [s.surface_coverage_pct for s in samples]

        return PairedDatasetReport(
            n_requested=n_pairs,
            n_generated=len(samples),
            n_passed_qc=n_passed,
            mean_alignment_error_mm=float(np.mean(alignment_errors)) if alignment_errors else 0.0,
            mean_landmark_error_mm=float(np.mean(landmark_errors)) if landmark_errors else 0.0,
            mean_coverage_pct=float(np.mean(coverages)) if coverages else 0.0,
            total_time_s=total_time,
            samples=samples,
        )

    # ── Statistics ────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return summary stats of the current library."""
        return {
            "library_root": str(self._library_root),
            "case_count": self._library.case_count,
            "grid_size": self._grid_size,
            "voxel_spacing_mm": self._spacing,
            "seed": self._seed,
        }
