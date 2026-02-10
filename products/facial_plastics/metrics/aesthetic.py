"""Aesthetic metrics for facial plastics outcomes.

Computes quantitative aesthetic measures from landmark positions,
surface meshes, and simulation results:

  - Nasal profile analysis (dorsal line, nasofrontal angle, nasolabial angle)
  - Tip metrics (projection, rotation, symmetry)
  - Proportional analysis (thirds, fifths, length/width ratios)
  - Bilateral symmetry (Procrustes, hausdorff, local deviation)
  - Dorsal aesthetic lines
  - Expert panel scoring model (multivariate regression)

All angles in degrees, distances in mm.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import (
    ClinicalMeasurement,
    Landmark,
    LandmarkType,
    SurfaceMesh,
    Vec3,
    VolumeMesh,
)

logger = logging.getLogger(__name__)


# ── Reference proportions (Farkas, 1994; Toriumi & Pero, 2010) ───

# Ideal ratios / angles from the facial plastics literature
IDEAL_NASOFRONTAL_ANGLE_DEG = 115.0    # 115–130°
IDEAL_NASOLABIAL_ANGLE_DEG_F = 100.0   # 95–110° female
IDEAL_NASOLABIAL_ANGLE_DEG_M = 95.0    # 90–100° male
IDEAL_TIP_PROJECTION_RATIO = 0.67      # Goode ratio (projection/length)
IDEAL_DORSAL_WIDTH_RATIO = 0.75        # bony/cartilaginous width
IDEAL_NASAL_LENGTH_TO_HEIGHT = 1.0     # ~ equal
IDEAL_TIP_ROTATION_DEG_F = 105.0       # 100–110° female
IDEAL_TIP_ROTATION_DEG_M = 95.0        # 90–100° male
IDEAL_ALAR_BASE_WIDTH_ICD = 1.0        # alar base ≈ intercanthal distance


# ── Measurement dataclasses ───────────────────────────────────────

@dataclass
class ProfileMetrics:
    """Sagittal nasal profile measurements."""
    nasofrontal_angle_deg: float = 0.0
    nasolabial_angle_deg: float = 0.0
    nasomental_angle_deg: float = 0.0
    dorsal_length_mm: float = 0.0
    tip_projection_mm: float = 0.0
    tip_rotation_deg: float = 0.0
    goode_ratio: float = 0.0                  # projection / length
    columellar_labial_angle_deg: float = 0.0
    supratip_break_depth_mm: float = 0.0
    dorsal_hump_mm: float = 0.0               # + = hump, − = scoop
    pollybeak_risk: bool = False
    radix_depth_mm: float = 0.0

    def score(self, is_female: bool = True) -> float:
        """Score profile on 0–100 scale relative to ideal proportions."""
        ideal_nf = IDEAL_NASOFRONTAL_ANGLE_DEG
        ideal_nl = IDEAL_NASOLABIAL_ANGLE_DEG_F if is_female else IDEAL_NASOLABIAL_ANGLE_DEG_M
        ideal_rot = IDEAL_TIP_ROTATION_DEG_F if is_female else IDEAL_TIP_ROTATION_DEG_M

        penalties = 0.0
        # Nasofrontal angle: 115-130° ideal
        penalties += max(0, abs(self.nasofrontal_angle_deg - ideal_nf) - 7.5) * 2.0
        # Nasolabial angle
        penalties += max(0, abs(self.nasolabial_angle_deg - ideal_nl) - 5.0) * 2.0
        # Goode ratio
        penalties += abs(self.goode_ratio - IDEAL_TIP_PROJECTION_RATIO) * 100.0
        # Dorsal hump
        penalties += max(0, abs(self.dorsal_hump_mm) - 0.5) * 10.0
        # Tip rotation
        penalties += max(0, abs(self.tip_rotation_deg - ideal_rot) - 5.0) * 1.5
        # Pollybeak
        if self.pollybeak_risk:
            penalties += 15.0

        return max(0.0, min(100.0, 100.0 - penalties))


@dataclass
class SymmetryMetrics:
    """Bilateral symmetry measurements."""
    procrustes_distance: float = 0.0       # after alignment
    max_asymmetry_mm: float = 0.0
    mean_asymmetry_mm: float = 0.0
    tip_deviation_mm: float = 0.0          # lateral deviation of pronasale
    dorsal_deviation_mm: float = 0.0       # lateral deviation of dorsum
    alar_base_asymmetry_mm: float = 0.0    # L-R alar base width difference
    nostril_area_asymmetry_pct: float = 0.0  # percent difference L vs R

    def score(self) -> float:
        """Score symmetry on 0–100 scale."""
        penalties = 0.0
        penalties += self.mean_asymmetry_mm * 15.0
        penalties += max(0, self.max_asymmetry_mm - 1.0) * 10.0
        penalties += abs(self.tip_deviation_mm) * 20.0
        penalties += abs(self.dorsal_deviation_mm) * 15.0
        penalties += abs(self.alar_base_asymmetry_mm) * 10.0
        penalties += abs(self.nostril_area_asymmetry_pct) * 0.5
        return max(0.0, min(100.0, 100.0 - penalties))


@dataclass
class ProportionMetrics:
    """Facial proportional analysis."""
    upper_third_mm: float = 0.0   # trichion to glabella
    middle_third_mm: float = 0.0  # glabella to subnasale
    lower_third_mm: float = 0.0   # subnasale to menton
    nasal_width_mm: float = 0.0   # alar base width
    intercanthal_distance_mm: float = 0.0
    nasal_width_to_icd_ratio: float = 0.0
    nasal_length_mm: float = 0.0
    nasal_height_mm: float = 0.0
    length_to_height_ratio: float = 0.0
    brow_tip_aesthetic_line_intact: bool = True

    def score(self) -> float:
        """Score proportions on 0–100 scale."""
        penalties = 0.0
        # Thirds should be approximately equal
        if self.middle_third_mm > 0 and self.lower_third_mm > 0:
            thirds_ratio = self.middle_third_mm / max(self.lower_third_mm, 1.0)
            penalties += abs(thirds_ratio - 1.0) * 30.0
        # Alar width ~ ICDistance
        penalties += abs(self.nasal_width_to_icd_ratio - IDEAL_ALAR_BASE_WIDTH_ICD) * 50.0
        # Nasal length/height
        penalties += abs(self.length_to_height_ratio - IDEAL_NASAL_LENGTH_TO_HEIGHT) * 30.0
        # BAL
        if not self.brow_tip_aesthetic_line_intact:
            penalties += 15.0
        return max(0.0, min(100.0, 100.0 - penalties))


@dataclass
class AestheticReport:
    """Complete aesthetic assessment."""
    profile: ProfileMetrics = field(default_factory=ProfileMetrics)
    symmetry: SymmetryMetrics = field(default_factory=SymmetryMetrics)
    proportions: ProportionMetrics = field(default_factory=ProportionMetrics)
    overall_score: float = 0.0
    measurements: List[ClinicalMeasurement] = field(default_factory=list)
    is_female: bool = True

    def compute_overall(self) -> float:
        """Weighted aggregate of component scores."""
        profile_score = self.profile.score(self.is_female)
        symmetry_score = self.symmetry.score()
        proportion_score = self.proportions.score()
        # Weights: profile 40%, symmetry 35%, proportions 25%
        self.overall_score = (
            0.40 * profile_score
            + 0.35 * symmetry_score
            + 0.25 * proportion_score
        )
        return self.overall_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "profile": {
                "nasofrontal_angle_deg": self.profile.nasofrontal_angle_deg,
                "nasolabial_angle_deg": self.profile.nasolabial_angle_deg,
                "dorsal_length_mm": self.profile.dorsal_length_mm,
                "tip_projection_mm": self.profile.tip_projection_mm,
                "tip_rotation_deg": self.profile.tip_rotation_deg,
                "goode_ratio": self.profile.goode_ratio,
                "dorsal_hump_mm": self.profile.dorsal_hump_mm,
                "pollybeak_risk": self.profile.pollybeak_risk,
                "score": self.profile.score(self.is_female),
            },
            "symmetry": {
                "procrustes_distance": self.symmetry.procrustes_distance,
                "max_asymmetry_mm": self.symmetry.max_asymmetry_mm,
                "mean_asymmetry_mm": self.symmetry.mean_asymmetry_mm,
                "tip_deviation_mm": self.symmetry.tip_deviation_mm,
                "score": self.symmetry.score(),
            },
            "proportions": {
                "nasal_width_mm": self.proportions.nasal_width_mm,
                "intercanthal_distance_mm": self.proportions.intercanthal_distance_mm,
                "nasal_width_to_icd_ratio": self.proportions.nasal_width_to_icd_ratio,
                "score": self.proportions.score(),
            },
            "measurements": [
                {"name": m.name, "value": m.value, "unit": m.unit}
                for m in self.measurements
            ],
        }

    def summary(self) -> str:
        return (
            f"Aesthetic Score: {self.overall_score:.1f}/100 "
            f"(profile={self.profile.score(self.is_female):.1f}, "
            f"symmetry={self.symmetry.score():.1f}, "
            f"proportions={self.proportions.score():.1f})"
        )


# ── Helper geometry functions ─────────────────────────────────────

def _angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two 3D vectors in degrees."""
    cos_a = np.clip(
        np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12),
        -1.0, 1.0,
    )
    return float(np.degrees(np.arccos(cos_a)))


def _project_to_sagittal(pts: np.ndarray) -> np.ndarray:
    """Project 3D points onto the sagittal plane (y-z, assuming x=lateral).

    Returns (N,2) array of [anterior-posterior, superior-inferior].
    """
    return pts[:, 1:3]  # y, z


def _reflect_landmarks_bilateral(
    landmarks: Dict[LandmarkType, Vec3],
) -> Tuple[Dict[LandmarkType, Vec3], Dict[LandmarkType, Vec3]]:
    """Split landmarks into left/right and create reflected pairs."""
    left: Dict[LandmarkType, Vec3] = {}
    right: Dict[LandmarkType, Vec3] = {}

    bilateral_pairs = [
        (LandmarkType.ALAR_CREASE_LEFT, LandmarkType.ALAR_CREASE_RIGHT),
        (LandmarkType.ALAR_RIM_LEFT, LandmarkType.ALAR_RIM_RIGHT),
        (LandmarkType.TIP_DEFINING_POINT_LEFT, LandmarkType.TIP_DEFINING_POINT_RIGHT),
        (LandmarkType.ENDOCANTHION_LEFT, LandmarkType.ENDOCANTHION_RIGHT),
        (LandmarkType.ENDOCANTHION_LEFT, LandmarkType.ENDOCANTHION_RIGHT),
        (LandmarkType.EXOCANTHION_LEFT, LandmarkType.EXOCANTHION_RIGHT),
        (LandmarkType.TRAGION_LEFT, LandmarkType.TRAGION_RIGHT),
        (LandmarkType.GONION_LEFT, LandmarkType.GONION_RIGHT),
        (LandmarkType.INTERNAL_VALVE_LEFT, LandmarkType.INTERNAL_VALVE_RIGHT),
        (LandmarkType.EXTERNAL_VALVE_LEFT, LandmarkType.EXTERNAL_VALVE_RIGHT),
    ]

    for lt, rt in bilateral_pairs:
        if lt in landmarks:
            left[lt] = landmarks[lt]
        if rt in landmarks:
            right[rt] = landmarks[rt]

    return left, right


def _procrustes_distance(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Ordinary Procrustes analysis between two point sets.

    Returns (distance, aligned_a, aligned_b).
    """
    ca = pts_a.mean(axis=0)
    cb = pts_b.mean(axis=0)
    a = pts_a - ca
    b = pts_b - cb

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0, a, b

    a /= norm_a
    b /= norm_b

    # Optimal rotation via SVD
    M = a.T @ b
    U, _, Vt = np.linalg.svd(M)
    d = np.linalg.det(U @ Vt)
    S = np.eye(a.shape[1])
    S[-1, -1] = np.sign(d)
    R = U @ S @ Vt

    a_aligned = a @ R
    dist = float(np.sqrt(np.sum((a_aligned - b) ** 2)))
    return dist, a_aligned * norm_a + ca, b * norm_b + cb


def _hausdorff_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """One-directional Hausdorff distance from A to B."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 0.0
    # Compute pairwise distances in batches to limit memory
    max_d = 0.0
    batch = 500
    for i in range(0, len(pts_a), batch):
        chunk = pts_a[i:i + batch]
        diffs = chunk[:, None, :] - pts_b[None, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        min_dists = dists.min(axis=1)
        max_d = max(max_d, float(min_dists.max()))
    return max_d


# ── Main aesthetic metrics calculator ────────────────────────────

class AestheticMetrics:
    """Compute aesthetic outcome metrics for a facial plastics case.

    Operates on landmarks and surface meshes to produce
    profile, symmetry, and proportional assessments.
    """

    def __init__(
        self,
        landmarks: Dict[LandmarkType, Vec3],
        *,
        surface: Optional[SurfaceMesh] = None,
        is_female: bool = True,
    ) -> None:
        self._lm = landmarks
        self._surface = surface
        self._is_female = is_female
        self._lm_arr = {k: v.to_array() for k, v in landmarks.items()}

    def _get(self, lt: LandmarkType) -> Optional[np.ndarray]:
        return self._lm_arr.get(lt)

    def compute_profile(self) -> ProfileMetrics:
        """Compute sagittal profile metrics."""
        pm = ProfileMetrics()

        nasion = self._get(LandmarkType.NASION)
        rhinion = self._get(LandmarkType.RHINION)
        pronasale = self._get(LandmarkType.PRONASALE)
        subnasale = self._get(LandmarkType.SUBNASALE)
        glabella = self._get(LandmarkType.GLABELLA)
        pogonion = self._get(LandmarkType.POGONION)
        sellion = self._get(LandmarkType.SELLION)

        # Dorsal length: nasion → pronasale along dorsum
        if nasion is not None and pronasale is not None:
            pm.dorsal_length_mm = float(np.linalg.norm(pronasale - nasion))

        # Tip projection: perpendicular distance from alar base plane to pronasale
        if pronasale is not None and subnasale is not None:
            # Approximate: horizontal distance from subnasale to pronasale
            diff = pronasale - subnasale
            pm.tip_projection_mm = float(np.sqrt(diff[1] ** 2 + diff[2] ** 2))

        # Goode ratio
        if pm.dorsal_length_mm > 0:
            pm.goode_ratio = pm.tip_projection_mm / pm.dorsal_length_mm

        # Nasofrontal angle: glabella → nasion → rhinion
        if glabella is not None and nasion is not None and rhinion is not None:
            v1 = glabella - nasion
            v2 = rhinion - nasion
            pm.nasofrontal_angle_deg = _angle_between_vectors(v1, v2)

        # Nasolabial angle: pronasale → subnasale → labiale superius direction
        if pronasale is not None and subnasale is not None:
            labrale_sup = self._get(LandmarkType.LABRALE_SUPERIUS)
            if labrale_sup is not None:
                v1 = pronasale - subnasale
                v2 = labrale_sup - subnasale
                pm.nasolabial_angle_deg = _angle_between_vectors(v1, v2)

        # Tip rotation: subnasale → pronasale angle relative to Frankfurt horizontal
        if pronasale is not None and subnasale is not None:
            diff = pronasale - subnasale
            pm.tip_rotation_deg = float(np.degrees(
                np.arctan2(abs(diff[2]), abs(diff[1]))
            )) + 90.0  # convert to standard rotation

        # Nasomental angle: nasion → pronasale → pogonion
        if nasion is not None and pronasale is not None and pogonion is not None:
            v1 = nasion - pronasale
            v2 = pogonion - pronasale
            pm.nasomental_angle_deg = _angle_between_vectors(v1, v2)

        # Dorsal hump detection
        if nasion is not None and pronasale is not None and rhinion is not None:
            # Project rhinion distance from nasion-pronasale line
            line = pronasale - nasion
            line_len = np.linalg.norm(line)
            if line_len > 1e-6:
                line_dir = line / line_len
                to_rhinion = rhinion - nasion
                proj = np.dot(to_rhinion, line_dir)
                perp = to_rhinion - proj * line_dir
                # Positive distance above line = hump
                pm.dorsal_hump_mm = float(np.linalg.norm(perp))
                # Determine sign from normal direction
                cross = np.cross(line_dir, to_rhinion)
                if cross[0] < 0:  # hump is lateral-positive
                    pm.dorsal_hump_mm = -pm.dorsal_hump_mm

        # Supratip break depth
        supratip = self._get(LandmarkType.SUPRATIP_BREAKPOINT)
        if supratip is not None and pronasale is not None and rhinion is not None:
            # Distance from supratip to rhinion-pronasale line
            line = pronasale - rhinion
            line_len = np.linalg.norm(line)
            if line_len > 1e-6:
                to_supra = supratip - rhinion
                proj = np.dot(to_supra, line / line_len) * (line / line_len)
                perp_dist = np.linalg.norm(to_supra - proj)
                pm.supratip_break_depth_mm = float(perp_dist)

        # Pollybeak risk: supratip higher than tip defining points
        if supratip is not None and pronasale is not None:
            if supratip[2] >= pronasale[2]:
                pm.pollybeak_risk = True

        # Radix depth
        if sellion is not None and nasion is not None and glabella is not None:
            line = glabella - nasion
            line_len = np.linalg.norm(line)
            if line_len > 1e-6:
                to_sell = sellion - nasion
                proj = np.dot(to_sell, line / line_len) * (line / line_len)
                pm.radix_depth_mm = float(np.linalg.norm(to_sell - proj))

        # Columellar-labial angle
        col_bp = self._get(LandmarkType.COLUMELLA_BREAKPOINT)
        if col_bp is not None and subnasale is not None:
            labrale_sup = self._get(LandmarkType.LABRALE_SUPERIUS)
            if labrale_sup is not None:
                v1 = col_bp - subnasale
                v2 = labrale_sup - subnasale
                pm.columellar_labial_angle_deg = _angle_between_vectors(v1, v2)

        return pm

    def compute_symmetry(self) -> SymmetryMetrics:
        """Compute bilateral symmetry metrics."""
        sm = SymmetryMetrics()

        # Tip deviation
        pronasale = self._get(LandmarkType.PRONASALE)
        nasion = self._get(LandmarkType.NASION)
        subnasale = self._get(LandmarkType.SUBNASALE)

        if pronasale is not None and nasion is not None and subnasale is not None:
            # Midline: nasion-subnasale defines the sagittal midline x
            midline_x = (nasion[0] + subnasale[0]) / 2.0
            sm.tip_deviation_mm = float(pronasale[0] - midline_x)

        # Dorsal deviation
        rhinion = self._get(LandmarkType.RHINION)
        if rhinion is not None and nasion is not None:
            midline_x = nasion[0]
            sm.dorsal_deviation_mm = float(rhinion[0] - midline_x)

        # Alar base asymmetry
        alar_l = self._get(LandmarkType.ALAR_CREASE_LEFT)
        alar_r = self._get(LandmarkType.ALAR_CREASE_RIGHT)
        if alar_l is not None and alar_r is not None and subnasale is not None:
            dist_l = float(np.linalg.norm(alar_l - subnasale))
            dist_r = float(np.linalg.norm(alar_r - subnasale))
            sm.alar_base_asymmetry_mm = dist_l - dist_r

        # Procrustes distance using bilateral landmark pairs
        left, right = _reflect_landmarks_bilateral(self._lm_arr)
        if len(left) >= 3 and len(right) >= 3:
            # Match pairs by order
            left_pts = []
            right_pts = []
            for (lt, rv) in [(LandmarkType.ALAR_CREASE_LEFT, LandmarkType.ALAR_CREASE_RIGHT),
                             (LandmarkType.ALAR_RIM_LEFT, LandmarkType.ALAR_RIM_RIGHT),
                             (LandmarkType.TIP_DEFINING_POINT_LEFT, LandmarkType.TIP_DEFINING_POINT_RIGHT)]:
                lv = self._get(lt)
                rv_arr = self._get(rv)
                if lv is not None and rv_arr is not None:
                    left_pts.append(lv)
                    # Reflect right across midline
                    reflected = rv_arr.copy()
                    reflected[0] = -reflected[0]  # mirror x
                    right_pts.append(reflected)

            if len(left_pts) >= 2:
                left_arr = np.array(left_pts)
                right_arr = np.array(right_pts)
                sm.procrustes_distance, _, _ = _procrustes_distance(left_arr, right_arr)

        # Surface-based symmetry
        if self._surface is not None:
            sm_surface = self._compute_surface_symmetry()
            sm.max_asymmetry_mm = sm_surface[0]
            sm.mean_asymmetry_mm = sm_surface[1]

        return sm

    def _compute_surface_symmetry(self) -> Tuple[float, float]:
        """Compute surface symmetry by reflecting across sagittal plane."""
        if self._surface is None:
            return 0.0, 0.0

        verts = self._surface.vertices
        # Reflect across x=0 (sagittal midline)
        reflected = verts.copy()
        reflected[:, 0] = -reflected[:, 0]

        # Compute closest-point distances
        max_d = 0.0
        total_d = 0.0
        n = len(verts)
        batch = 500
        for i in range(0, n, batch):
            chunk = reflected[i:i + batch]
            diffs = chunk[:, None, :] - verts[None, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            min_dists = dists.min(axis=1)
            max_d = max(max_d, float(min_dists.max()))
            total_d += float(min_dists.sum())

        return max_d, total_d / max(n, 1)

    def compute_proportions(self) -> ProportionMetrics:
        """Compute facial proportional metrics."""
        pm = ProportionMetrics()

        glabella = self._get(LandmarkType.GLABELLA)
        subnasale = self._get(LandmarkType.SUBNASALE)
        menton = self._get(LandmarkType.MENTON)
        nasion = self._get(LandmarkType.NASION)
        pronasale = self._get(LandmarkType.PRONASALE)

        # Facial thirds (vertical)
        trichion = self._get(LandmarkType.VERTEX)  # approximate
        if trichion is not None and glabella is not None:
            pm.upper_third_mm = float(np.linalg.norm(trichion - glabella))
        if glabella is not None and subnasale is not None:
            pm.middle_third_mm = float(np.linalg.norm(subnasale - glabella))
        if subnasale is not None and menton is not None:
            pm.lower_third_mm = float(np.linalg.norm(menton - subnasale))

        # Nasal dimensions
        if nasion is not None and pronasale is not None:
            pm.nasal_length_mm = float(np.linalg.norm(pronasale - nasion))
        if nasion is not None and subnasale is not None:
            pm.nasal_height_mm = abs(nasion[2] - subnasale[2])

        if pm.nasal_height_mm > 0:
            pm.length_to_height_ratio = pm.nasal_length_mm / pm.nasal_height_mm

        # Alar base width
        alar_l = self._get(LandmarkType.ALAR_CREASE_LEFT)
        alar_r = self._get(LandmarkType.ALAR_CREASE_RIGHT)
        if alar_l is not None and alar_r is not None:
            pm.nasal_width_mm = float(np.linalg.norm(alar_l - alar_r))

        # Intercanthal distance
        endo_l = self._get(LandmarkType.ENDOCANTHION_LEFT)
        endo_r = self._get(LandmarkType.ENDOCANTHION_RIGHT)
        if endo_l is not None and endo_r is not None:
            pm.intercanthal_distance_mm = float(np.linalg.norm(endo_l - endo_r))

        # Width ratio
        if pm.intercanthal_distance_mm > 0:
            pm.nasal_width_to_icd_ratio = (
                pm.nasal_width_mm / pm.intercanthal_distance_mm
            )

        # Brow-tip aesthetic line (requires surface)
        if self._surface is not None:
            pm.brow_tip_aesthetic_line_intact = self._check_btal()

        return pm

    def _check_btal(self) -> bool:
        """Check if brow-tip aesthetic lines are smooth and continuous.

        The brow-tip aesthetic line should run smoothly from the
        supraorbital ridge through the lateral dorsum to the tip
        defining points. A disruption indicates dorsal irregularity.
        """
        if self._surface is None:
            return True

        exo_l = self._get(LandmarkType.EXOCANTHION_LEFT)
        tdp_l = self._get(LandmarkType.TIP_DEFINING_POINT_LEFT)
        exo_r = self._get(LandmarkType.EXOCANTHION_RIGHT)
        tdp_r = self._get(LandmarkType.TIP_DEFINING_POINT_RIGHT)

        if exo_l is None or tdp_l is None or exo_r is None or tdp_r is None:
            return True  # can't assess

        # Sample points along each line and check surface distance deviation
        for start, end in [(exo_l, tdp_l), (exo_r, tdp_r)]:
            n_samples = 20
            for i in range(n_samples):
                t = (i + 1) / (n_samples + 1)
                pt = start * (1 - t) + end * t
                # Find closest surface point
                diffs = self._surface.vertices - pt
                dists = np.linalg.norm(diffs, axis=1)
                min_idx = np.argmin(dists)
                deviation = dists[min_idx]
                if deviation > 3.0:  # > 3mm deviation from smooth line
                    return False

        return True

    def compute(self, *, preop_landmarks: Optional[Dict[LandmarkType, Vec3]] = None) -> AestheticReport:
        """Compute full aesthetic report.

        Args:
            preop_landmarks: If provided, also compute change metrics.
        """
        report = AestheticReport(is_female=self._is_female)
        report.profile = self.compute_profile()
        report.symmetry = self.compute_symmetry()
        report.proportions = self.compute_proportions()
        report.compute_overall()

        # Add key measurements to measurement list
        report.measurements = self._build_measurement_list(report)

        logger.info("Aesthetic assessment: %s", report.summary())
        return report

    def _build_measurement_list(self, report: AestheticReport) -> List[ClinicalMeasurement]:
        """Build standardized measurement list from computed metrics."""
        measurements: List[ClinicalMeasurement] = []

        p = report.profile
        if p.nasofrontal_angle_deg > 0:
            measurements.append(ClinicalMeasurement(
                name="nasofrontal_angle", value=p.nasofrontal_angle_deg,
                unit="deg", method="landmark_computed",
            ))
        if p.nasolabial_angle_deg > 0:
            measurements.append(ClinicalMeasurement(
                name="nasolabial_angle", value=p.nasolabial_angle_deg,
                unit="deg", method="landmark_computed",
            ))
        if p.dorsal_length_mm > 0:
            measurements.append(ClinicalMeasurement(
                name="dorsal_length", value=p.dorsal_length_mm,
                unit="mm", method="landmark_computed",
            ))
        if p.tip_projection_mm > 0:
            measurements.append(ClinicalMeasurement(
                name="tip_projection", value=p.tip_projection_mm,
                unit="mm", method="landmark_computed",
            ))
        if p.goode_ratio > 0:
            measurements.append(ClinicalMeasurement(
                name="goode_ratio", value=p.goode_ratio,
                unit="ratio", method="landmark_computed",
            ))
        if p.tip_rotation_deg > 0:
            measurements.append(ClinicalMeasurement(
                name="tip_rotation", value=p.tip_rotation_deg,
                unit="deg", method="landmark_computed",
            ))
        if abs(p.dorsal_hump_mm) > 0:
            measurements.append(ClinicalMeasurement(
                name="dorsal_hump", value=p.dorsal_hump_mm,
                unit="mm", method="landmark_computed",
            ))

        s = report.symmetry
        measurements.append(ClinicalMeasurement(
            name="tip_deviation", value=s.tip_deviation_mm,
            unit="mm", method="landmark_computed",
        ))
        measurements.append(ClinicalMeasurement(
            name="alar_base_asymmetry", value=s.alar_base_asymmetry_mm,
            unit="mm", method="landmark_computed",
        ))

        pr = report.proportions
        if pr.nasal_width_mm > 0:
            measurements.append(ClinicalMeasurement(
                name="nasal_width", value=pr.nasal_width_mm,
                unit="mm", method="landmark_computed",
            ))
        if pr.intercanthal_distance_mm > 0:
            measurements.append(ClinicalMeasurement(
                name="intercanthal_distance", value=pr.intercanthal_distance_mm,
                unit="mm", method="landmark_computed",
            ))

        return measurements

    @staticmethod
    def compute_change(preop: AestheticReport, postop: AestheticReport) -> Dict[str, float]:
        """Compute delta between pre-op and post-op aesthetic reports."""
        changes: Dict[str, float] = {}
        changes["overall_score_delta"] = postop.overall_score - preop.overall_score
        changes["nasofrontal_angle_delta"] = (
            postop.profile.nasofrontal_angle_deg - preop.profile.nasofrontal_angle_deg
        )
        changes["nasolabial_angle_delta"] = (
            postop.profile.nasolabial_angle_deg - preop.profile.nasolabial_angle_deg
        )
        changes["tip_projection_delta_mm"] = (
            postop.profile.tip_projection_mm - preop.profile.tip_projection_mm
        )
        changes["goode_ratio_delta"] = (
            postop.profile.goode_ratio - preop.profile.goode_ratio
        )
        changes["dorsal_hump_delta_mm"] = (
            postop.profile.dorsal_hump_mm - preop.profile.dorsal_hump_mm
        )
        changes["symmetry_improvement_mm"] = (
            preop.symmetry.mean_asymmetry_mm - postop.symmetry.mean_asymmetry_mm
        )
        changes["tip_deviation_delta_mm"] = (
            postop.symmetry.tip_deviation_mm - preop.symmetry.tip_deviation_mm
        )
        return changes
