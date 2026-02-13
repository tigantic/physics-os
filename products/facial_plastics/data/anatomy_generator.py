"""Parametric facial anatomy generator for synthetic case creation.

Generates anatomically realistic CT volumes and surface meshes from
anthropometric parameters conditioned on demographics (age, sex,
ethnicity, Fitzpatrick skin type).

Architecture
------------
1. **AnthropometricProfile** — Facial measurement distributions
   derived from published anthropometric literature (Farkas 1994,
   Leong & White 2004, Porter & Olson 2001).
2. **ParametricAnatomy** — Implicit-function definitions for bone,
   cartilage, airway, and soft tissue structures placed in a
   coordinate system aligned with Frankfurt Horizontal plane.
3. **VolumeRenderer** — Rasterises implicit anatomy into a voxel
   grid at configurable resolution, assigning correct HU values.
4. **SurfaceExtractor** — Extracts a facial skin surface mesh from
   the outer boundary of the rendered volume.

Units: all spatial quantities in millimetres unless otherwise noted.
HU values follow standard CT convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import lil_matrix

try:
    from skimage.measure import marching_cubes as _skimage_marching_cubes
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

from ..core.case_bundle import CaseBundle, PatientDemographics
from ..core.types import (
    ClinicalMeasurement,
    LandmarkType,
    Modality,
    ProcedureType,
    StructureType,
    SurfaceMesh,
    Vec3,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# HU value ranges (literature consensus)
# ══════════════════════════════════════════════════════════════════

HU_AIR: float = -1000.0
HU_FAT: float = -80.0
HU_SOFT_TISSUE: float = 40.0
HU_MUSCLE: float = 50.0
HU_MUCOSA: float = 55.0
HU_CARTILAGE: float = 200.0
HU_BONE_CANCELLOUS: float = 400.0
HU_BONE_CORTICAL: float = 1200.0
HU_BACKGROUND: float = -1024.0

# ══════════════════════════════════════════════════════════════════
# Anthropometric parameter distributions
# ══════════════════════════════════════════════════════════════════


@dataclass
class AnthropometricProfile:
    """Facial parameter set drawn from population distributions.

    All lengths in mm, all angles in degrees.
    Parameters are referenced to published anthropometric norms
    (Farkas 1994 — *Anthropometry of the Head and Face*).
    """
    # Global
    age_years: int = 35
    sex: str = "M"
    ethnicity: str = "european"
    skin_fitzpatrick: int = 2

    # Craniofacial skeleton
    skull_width: float = 145.0       # eu-eu (euryon to euryon)
    skull_height: float = 130.0      # v-gn (vertex to gnathion)
    skull_depth: float = 190.0       # g-op (glabella to opisthocranion)
    face_height_upper: float = 70.0  # n-sto
    face_height_lower: float = 65.0  # sn-gn

    # Nasal skeleton
    nasal_bone_length: float = 25.0      # n-rhi
    nasal_bone_width: float = 10.0
    nasal_bone_thickness: float = 2.5
    nasal_dorsal_height: float = 18.0    # dorsal depth above facial plane
    nasal_dorsal_length: float = 45.0    # n-prn

    # Nasal cartilage
    septum_length: float = 35.0
    septum_height: float = 25.0
    septum_thickness: float = 2.0
    ulc_length: float = 15.0
    ulc_width: float = 12.0
    ulc_thickness: float = 1.5
    llc_length: float = 12.0
    llc_width: float = 20.0
    llc_thickness: float = 1.2

    # Nasal tip and ala
    tip_projection: float = 28.0     # sn-prn (subnasale to pronasale)
    tip_rotation: float = 105.0      # nasolabial angle (deg)
    alar_width: float = 35.0         # al-al (alar to alar)
    columella_length: float = 10.0

    # Nasal airway
    airway_valve_width: float = 9.0
    airway_valve_angle: float = 15.0  # deg
    turbinate_size: float = 8.0
    nasal_cavity_height: float = 50.0
    nasal_cavity_depth: float = 60.0

    # Skin envelope
    skin_thickness_dorsum: float = 2.5
    skin_thickness_tip: float = 4.0
    skin_thickness_alar: float = 3.0
    skin_thickness_cheek: float = 2.0
    skin_thickness_forehead: float = 3.0
    fat_thickness_cheek: float = 8.0
    fat_thickness_submental: float = 5.0

    # Maxilla and mandible
    maxilla_width: float = 62.0
    maxilla_depth: float = 55.0
    mandible_width: float = 100.0
    mandible_depth: float = 80.0
    mandible_angle: float = 125.0  # deg

    # Orbit
    intercanthal_distance: float = 33.0
    interpupillary_distance: float = 63.0
    orbit_width: float = 30.0
    orbit_height: float = 35.0


# ── Population sampling ───────────────────────────────────────────

# Anthropometric parameter means and CVs conditioned on sex/ethnicity.
# Derived from: Farkas 1994, Leong & White 2004, Porter & Olson 2001,
# Ofodile & Bokhari 1995, Wang et al. 2009.

_POPULATION_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # (mean, coefficient_of_variation)
    "european_M": {
        "skull_width": (150.0, 0.04),
        "skull_height": (135.0, 0.05),
        "skull_depth": (195.0, 0.04),
        "nasal_bone_length": (25.0, 0.12),
        "nasal_dorsal_height": (20.0, 0.15),
        "nasal_dorsal_length": (48.0, 0.08),
        "tip_projection": (29.0, 0.12),
        "tip_rotation": (100.0, 0.06),
        "alar_width": (35.0, 0.08),
        "septum_length": (36.0, 0.10),
        "skin_thickness_tip": (3.5, 0.20),
        "fat_thickness_cheek": (7.0, 0.25),
        "intercanthal_distance": (33.0, 0.08),
    },
    "european_F": {
        "skull_width": (140.0, 0.04),
        "skull_height": (125.0, 0.05),
        "skull_depth": (180.0, 0.04),
        "nasal_bone_length": (22.0, 0.12),
        "nasal_dorsal_height": (17.0, 0.15),
        "nasal_dorsal_length": (43.0, 0.08),
        "tip_projection": (26.0, 0.12),
        "tip_rotation": (108.0, 0.06),
        "alar_width": (31.0, 0.08),
        "septum_length": (32.0, 0.10),
        "skin_thickness_tip": (3.8, 0.20),
        "fat_thickness_cheek": (8.0, 0.25),
        "intercanthal_distance": (31.0, 0.08),
    },
    "east_asian_M": {
        "skull_width": (155.0, 0.04),
        "skull_height": (130.0, 0.05),
        "skull_depth": (185.0, 0.04),
        "nasal_bone_length": (20.0, 0.15),
        "nasal_dorsal_height": (14.0, 0.18),
        "nasal_dorsal_length": (42.0, 0.08),
        "tip_projection": (24.0, 0.12),
        "tip_rotation": (102.0, 0.06),
        "alar_width": (39.0, 0.08),
        "septum_length": (33.0, 0.10),
        "skin_thickness_tip": (4.5, 0.20),
        "fat_thickness_cheek": (9.0, 0.25),
        "intercanthal_distance": (35.0, 0.08),
    },
    "east_asian_F": {
        "skull_width": (148.0, 0.04),
        "skull_height": (122.0, 0.05),
        "skull_depth": (175.0, 0.04),
        "nasal_bone_length": (18.0, 0.15),
        "nasal_dorsal_height": (12.0, 0.18),
        "nasal_dorsal_length": (38.0, 0.08),
        "tip_projection": (22.0, 0.12),
        "tip_rotation": (106.0, 0.06),
        "alar_width": (36.0, 0.08),
        "septum_length": (30.0, 0.10),
        "skin_thickness_tip": (4.8, 0.20),
        "fat_thickness_cheek": (9.5, 0.25),
        "intercanthal_distance": (34.0, 0.08),
    },
    "african_M": {
        "skull_width": (152.0, 0.04),
        "skull_height": (132.0, 0.05),
        "skull_depth": (190.0, 0.04),
        "nasal_bone_length": (18.0, 0.15),
        "nasal_dorsal_height": (13.0, 0.18),
        "nasal_dorsal_length": (40.0, 0.08),
        "tip_projection": (24.0, 0.12),
        "tip_rotation": (98.0, 0.06),
        "alar_width": (43.0, 0.10),
        "septum_length": (32.0, 0.10),
        "skin_thickness_tip": (5.0, 0.20),
        "fat_thickness_cheek": (8.0, 0.25),
        "intercanthal_distance": (36.0, 0.08),
    },
    "african_F": {
        "skull_width": (145.0, 0.04),
        "skull_height": (125.0, 0.05),
        "skull_depth": (180.0, 0.04),
        "nasal_bone_length": (16.0, 0.15),
        "nasal_dorsal_height": (11.0, 0.18),
        "nasal_dorsal_length": (37.0, 0.08),
        "tip_projection": (22.0, 0.12),
        "tip_rotation": (100.0, 0.06),
        "alar_width": (40.0, 0.10),
        "septum_length": (29.0, 0.10),
        "skin_thickness_tip": (5.5, 0.20),
        "fat_thickness_cheek": (8.5, 0.25),
        "intercanthal_distance": (35.0, 0.08),
    },
    "south_asian_M": {
        "skull_width": (148.0, 0.04),
        "skull_height": (128.0, 0.05),
        "skull_depth": (185.0, 0.04),
        "nasal_bone_length": (22.0, 0.12),
        "nasal_dorsal_height": (16.0, 0.15),
        "nasal_dorsal_length": (44.0, 0.08),
        "tip_projection": (26.0, 0.12),
        "tip_rotation": (100.0, 0.06),
        "alar_width": (37.0, 0.08),
        "septum_length": (34.0, 0.10),
        "skin_thickness_tip": (4.0, 0.20),
        "fat_thickness_cheek": (7.5, 0.25),
        "intercanthal_distance": (34.0, 0.08),
    },
    "south_asian_F": {
        "skull_width": (142.0, 0.04),
        "skull_height": (120.0, 0.05),
        "skull_depth": (175.0, 0.04),
        "nasal_bone_length": (20.0, 0.12),
        "nasal_dorsal_height": (14.0, 0.15),
        "nasal_dorsal_length": (40.0, 0.08),
        "tip_projection": (24.0, 0.12),
        "tip_rotation": (104.0, 0.06),
        "alar_width": (34.0, 0.08),
        "septum_length": (31.0, 0.10),
        "skin_thickness_tip": (4.2, 0.20),
        "fat_thickness_cheek": (8.0, 0.25),
        "intercanthal_distance": (32.0, 0.08),
    },
    "hispanic_M": {
        "skull_width": (149.0, 0.04),
        "skull_height": (131.0, 0.05),
        "skull_depth": (188.0, 0.04),
        "nasal_bone_length": (23.0, 0.12),
        "nasal_dorsal_height": (17.0, 0.15),
        "nasal_dorsal_length": (45.0, 0.08),
        "tip_projection": (27.0, 0.12),
        "tip_rotation": (101.0, 0.06),
        "alar_width": (37.0, 0.08),
        "septum_length": (34.0, 0.10),
        "skin_thickness_tip": (4.0, 0.20),
        "fat_thickness_cheek": (7.5, 0.25),
        "intercanthal_distance": (33.0, 0.08),
    },
    "hispanic_F": {
        "skull_width": (142.0, 0.04),
        "skull_height": (123.0, 0.05),
        "skull_depth": (178.0, 0.04),
        "nasal_bone_length": (20.0, 0.12),
        "nasal_dorsal_height": (15.0, 0.15),
        "nasal_dorsal_length": (41.0, 0.08),
        "tip_projection": (25.0, 0.12),
        "tip_rotation": (105.0, 0.06),
        "alar_width": (34.0, 0.08),
        "septum_length": (31.0, 0.10),
        "skin_thickness_tip": (4.2, 0.20),
        "fat_thickness_cheek": (8.0, 0.25),
        "intercanthal_distance": (31.0, 0.08),
    },
}

_ETHNICITY_ALIAS: Dict[str, str] = {
    "caucasian": "european",
    "white": "european",
    "asian": "east_asian",
    "chinese": "east_asian",
    "japanese": "east_asian",
    "korean": "east_asian",
    "black": "african",
    "african_american": "african",
    "indian": "south_asian",
    "pakistani": "south_asian",
    "latino": "hispanic",
    "latina": "hispanic",
    "middle_eastern": "south_asian",
}

# Default Fitzpatrick by ethnicity (mode, not exclusive)
_DEFAULT_FITZPATRICK: Dict[str, int] = {
    "european": 2,
    "east_asian": 3,
    "african": 5,
    "south_asian": 4,
    "hispanic": 3,
}


def _resolve_ethnicity(ethnicity: str) -> str:
    """Normalise ethnicity string to a canonical key."""
    key = ethnicity.lower().strip().replace(" ", "_")
    return _ETHNICITY_ALIAS.get(key, key)


class PopulationSampler:
    """Draw demographically conditioned anthropometric parameters.

    Parameters are sampled from normal distributions with
    population-specific means and coefficients of variation.
    Age adjustments are applied post-sampling (nasal elongation
    with age, skin thickness changes, etc.).
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def sample_demographics(self) -> PatientDemographics:
        """Sample a random set of patient demographics."""
        sex = self._rng.choice(["M", "F"])
        ethnicity_pool = list({
            "european", "east_asian", "african", "south_asian", "hispanic",
        })
        ethnicity = self._rng.choice(ethnicity_pool)
        age = int(self._rng.integers(18, 76))
        fitz = _DEFAULT_FITZPATRICK.get(ethnicity, 3)
        # Add +/- 1 variation
        fitz = int(np.clip(fitz + self._rng.integers(-1, 2), 1, 6))
        return PatientDemographics(
            age_years=age,
            sex=sex,
            ethnicity=ethnicity,
            skin_fitzpatrick=fitz,
        )

    def sample_profile(
        self,
        demographics: Optional[PatientDemographics] = None,
    ) -> AnthropometricProfile:
        """Sample a full anthropometric profile.

        Parameters
        ----------
        demographics : PatientDemographics, optional
            If None, demographics are sampled randomly.

        Returns
        -------
        AnthropometricProfile with all measurements filled.
        """
        if demographics is None:
            demographics = self.sample_demographics()

        sex = demographics.sex or "M"
        ethnicity = _resolve_ethnicity(demographics.ethnicity or "european")
        age = demographics.age_years or 35
        fitz = demographics.skin_fitzpatrick or 2

        pop_key = f"{ethnicity}_{sex}"
        if pop_key not in _POPULATION_PARAMS:
            pop_key = f"european_{sex}"

        pop = _POPULATION_PARAMS[pop_key]

        profile = AnthropometricProfile(
            age_years=age,
            sex=sex,
            ethnicity=ethnicity,
            skin_fitzpatrick=fitz,
        )

        # Sample each parameter from its distribution
        for param_name, (mean, cv) in pop.items():
            sigma = mean * cv
            value = float(self._rng.normal(mean, sigma))
            # Clamp to ±3σ
            value = float(np.clip(value, mean - 3 * sigma, mean + 3 * sigma))
            setattr(profile, param_name, value)

        # Age adjustments (Boehm et al. 2006, Pessa & Rohrich 2012)
        age_factor = (age - 30) / 50.0  # normalised age deviation
        profile.skin_thickness_tip *= (1.0 - 0.08 * age_factor)
        profile.skin_thickness_dorsum *= (1.0 - 0.06 * age_factor)
        profile.fat_thickness_cheek *= (1.0 + 0.15 * age_factor)
        profile.fat_thickness_submental *= (1.0 + 0.20 * age_factor)
        profile.tip_projection *= (1.0 - 0.04 * age_factor)  # tip ptosis
        profile.tip_rotation -= 3.0 * age_factor  # tip drop

        # Fill remaining parameters from defaults scaled by sex
        sex_scale = 1.0 if sex == "M" else 0.92
        if "maxilla_width" not in pop:
            profile.maxilla_width = 62.0 * sex_scale + self._rng.normal(0, 2)
        if "mandible_width" not in pop:
            profile.mandible_width = 100.0 * sex_scale + self._rng.normal(0, 3)

        return profile


# ══════════════════════════════════════════════════════════════════
# Parametric anatomy — implicit function definitions
# ══════════════════════════════════════════════════════════════════

@dataclass
class _ImplicitStructure:
    """A parametric implicit-function shape definition."""
    structure: StructureType
    hu_value: float
    label_id: int  # from LABEL_MAP


def _ellipsoid_sdf(
    coords: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Signed distance to an axis-aligned ellipsoid (negative inside).

    Uses the Inigo Quilez approximation which is exact on the surface
    and along all principal axes, and a close approximation elsewhere.
    Much more accurate than the naive normalised-sphere approach for
    non-spherical ellipsoids.

    Parameters
    ----------
    coords : (N, 3) array
        Query coordinates.
    center, radii : (3,) arrays
        Ellipsoid centre and semi-axis lengths.

    Returns
    -------
    (N,) signed distance values.
    """
    p = coords - center
    # k0 = ||p/r||, k1 = ||p/r²||
    pr = p / radii
    pr2 = p / (radii * radii)
    k0 = np.linalg.norm(pr, axis=1)
    k1 = np.linalg.norm(pr2, axis=1)
    k0 = np.maximum(k0, 1e-10)
    k1 = np.maximum(k1, 1e-10)
    result: np.ndarray = k0 * (k0 - 1.0) / k1
    return result


def _box_sdf(
    coords: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    """Signed distance to an axis-aligned box."""
    d = np.abs(coords - center) - half_extents
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    result: np.ndarray = outside + inside
    return result


def _cylinder_sdf(
    coords: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Signed distance to a capped cylinder between p0 and p1."""
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-8:
        return np.full(coords.shape[0], 1e6)
    d = axis / length
    proj = (coords - p0) @ d
    proj_clamped = np.clip(proj, 0.0, length)
    closest = p0 + proj_clamped[:, None] * d
    radial = np.linalg.norm(coords - closest, axis=1) - radius
    cap_dist_low = -proj
    cap_dist_high = proj - length
    cap_dist = np.maximum(cap_dist_low, cap_dist_high)
    result: np.ndarray = np.maximum(radial, cap_dist)
    return result


def _torus_sdf(
    coords: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    major_radius: float,
    minor_radius: float,
) -> np.ndarray:
    """Signed distance to a torus centred at ``center`` with given axis.

    Parameters
    ----------
    coords : (N, 3) array
    center : (3,) array — torus centre.
    axis : (3,) array — torus rotation axis (unit vector).
    major_radius : float — distance from centre to tube centre.
    minor_radius : float — tube radius.

    Returns
    -------
    (N,) signed distance (negative inside).
    """
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    p = coords - center
    # Project onto torus plane
    proj_along = (p @ ax)[:, None] * ax  # component along axis
    proj_plane = p - proj_along  # component in the torus plane
    dist_plane = np.linalg.norm(proj_plane, axis=1)
    dist_plane = np.maximum(dist_plane, 1e-10)
    # Distance from the tube centre ring
    ring_dist = np.sqrt(
        (dist_plane - major_radius) ** 2
        + np.sum(proj_along ** 2, axis=1)
    )
    result: np.ndarray = ring_dist - minor_radius
    return result


def _smooth_union(d1: np.ndarray, d2: np.ndarray, k: float = 4.0) -> np.ndarray:
    """Smooth (polynomial) union of two distance fields.

    Produces blended transitions between shapes instead of hard edges.
    Smaller *k* = tighter blend; larger *k* = softer blend.
    """
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    result: np.ndarray = d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)
    return result


def _smooth_subtraction(
    d_cut: np.ndarray, d_base: np.ndarray, k: float = 4.0,
) -> np.ndarray:
    """Smooth subtraction: subtract *d_cut* from *d_base*."""
    h = np.clip(0.5 - 0.5 * (d_base + d_cut) / k, 0.0, 1.0)
    result: np.ndarray = d_base * (1.0 - h) + (-d_cut) * h + k * h * (1.0 - h)
    return result


class AnatomyGenerator:
    """Generate anatomically realistic CT volumes and surface meshes.

    Coordinate system (Frankfurt Horizontal oriented):
      X: right (+) / left (-)
      Y: superior (+) / inferior (-)
      Z: anterior (+) / posterior (-)
    Origin at subnasale.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_ct_volume(
        self,
        profile: AnthropometricProfile,
        *,
        grid_size: int = 128,
        voxel_spacing_mm: float = 1.0,
    ) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray]:
        """Render a synthetic CT volume from an anthropometric profile.

        Parameters
        ----------
        profile : AnthropometricProfile
            Fully populated facial parameter set.
        grid_size : int
            Voxels per dimension (cubic grid). Default 128.
        voxel_spacing_mm : float
            Isotropic voxel size in mm. Default 1.0.

        Returns
        -------
        volume_hu : ndarray (D, H, W) float32
            Hounsfield unit volume.
        spacing : (float, float, float)
            Voxel spacing (z, y, x) in mm.
        origin_mm : ndarray (3,)
            Physical coordinate of voxel [0, 0, 0].
        """
        sp = voxel_spacing_mm
        half = grid_size * sp / 2.0
        origin = np.array([-half, -half, -half])
        spacing = (sp, sp, sp)

        # Create voxel coordinate grid
        z_idx, y_idx, x_idx = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
        coords_flat = np.column_stack([
            x_idx.ravel() * sp + origin[0],
            y_idx.ravel() * sp + origin[1],
            z_idx.ravel() * sp + origin[2],
        ]).astype(np.float32)  # (N, 3) — (x, y, z)

        n_voxels = grid_size ** 3

        # Initialise to background air
        hu = np.full(n_voxels, HU_BACKGROUND, dtype=np.float32)

        # ── Place structures in order: deep → superficial ─────
        # Coordinate origin at centre of grid (≈ subnasale)

        # 1. Skull vault — large ellipsoid of bone
        self._place_skull(hu, coords_flat, profile)

        # 2. Maxilla
        self._place_maxilla(hu, coords_flat, profile)

        # 3. Mandible
        self._place_mandible(hu, coords_flat, profile)

        # 4. Nasal bones
        self._place_nasal_bones(hu, coords_flat, profile)

        # 5. Airway passages (carve through existing structures)
        self._place_airway(hu, coords_flat, profile)

        # 6. Nasal septum (cartilage)
        self._place_septum(hu, coords_flat, profile)

        # 7. Upper lateral cartilage
        self._place_upper_lateral_cartilage(hu, coords_flat, profile)

        # 8. Lower lateral cartilage
        self._place_lower_lateral_cartilage(hu, coords_flat, profile)

        # 9. Turbinates
        self._place_turbinates(hu, coords_flat, profile)

        # 10. Muscle layer
        self._place_muscle(hu, coords_flat, profile)

        # 11. Fat layer
        self._place_fat(hu, coords_flat, profile)

        # 12. Skin envelope (outermost)
        self._place_skin(hu, coords_flat, profile)

        # 13. Mucosa lining airway
        self._place_mucosa(hu, coords_flat, profile)

        # 14. Orbital cavities (eye sockets)
        self._place_orbits(hu, coords_flat, profile)

        # 15. Zygomatic arches (cheekbones)
        self._place_zygomatic(hu, coords_flat, profile)

        # 16. Pyriform aperture (bony nasal opening)
        self._place_pyriform(hu, coords_flat, profile)

        # 17. Lip structures
        self._place_lips(hu, coords_flat, profile)

        # 18. Enhanced nasal tip and alar detail
        self._place_nasal_tip_detail(hu, coords_flat, profile)

        # 19. Forehead and brow ridge
        self._place_forehead(hu, coords_flat, profile)

        # 20. Chin projection
        self._place_chin(hu, coords_flat, profile)

        # Add realistic noise (scanner noise ~ N(0, 15) HU)
        noise_sigma = 12.0 + self._rng.uniform(0, 8)
        noise = self._rng.normal(0, noise_sigma, size=n_voxels).astype(np.float32)
        # Only add noise to non-background voxels
        non_bg = hu > HU_BACKGROUND + 100
        hu[non_bg] += noise[non_bg]

        volume = hu.reshape(grid_size, grid_size, grid_size)

        logger.info(
            "Generated CT volume: %s, spacing=%.1f mm, HU range=[%.0f, %.0f]",
            volume.shape, sp, volume.min(), volume.max(),
        )
        return volume, spacing, origin

    # ── Structure placement (each writes into the flat HU array) ──

    def _place_skull(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Cranial vault — superior-posterior ellipsoid of cortical bone."""
        center = np.array([0.0, p.skull_height * 0.3, -p.skull_depth * 0.25])
        radii = np.array([p.skull_width / 2.0, p.skull_height / 2.0, p.skull_depth / 2.0])
        shell_thickness = 6.0  # mm cortical shell

        sdf = _ellipsoid_sdf(coords, center, radii)
        # Cortical shell: between outer surface and inner margin
        outer = sdf < 0
        inner_radii = radii - shell_thickness
        sdf_inner = _ellipsoid_sdf(coords, center, inner_radii)
        shell = outer & (sdf_inner >= 0)
        hu[shell] = HU_BONE_CORTICAL + self._rng.uniform(-100, 100, int(shell.sum())).astype(np.float32)

        # Cancellous interior (partial)
        deep_inner = sdf_inner < -shell_thickness
        hu[deep_inner & (hu < HU_BACKGROUND + 200)] = HU_SOFT_TISSUE

    def _place_maxilla(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Bilateral maxilla — box-like bone below orbits."""
        for side in [-1, 1]:
            center = np.array([
                side * p.maxilla_width * 0.25,
                -p.face_height_upper * 0.2,
                -p.maxilla_depth * 0.15,
            ])
            half_ext = np.array([
                p.maxilla_width * 0.22,
                p.face_height_upper * 0.35,
                p.maxilla_depth * 0.3,
            ])
            sdf = _box_sdf(coords, center, half_ext)
            mask = sdf < 0
            hu[mask] = np.where(
                hu[mask] < HU_BONE_CANCELLOUS,
                HU_BONE_CORTICAL + self._rng.uniform(-150, 50, int(mask.sum())).astype(np.float32),
                hu[mask],
            )

    def _place_mandible(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Mandible — U-shaped bone at inferior face."""
        # Body
        center = np.array([0.0, -p.skull_height * 0.35, -5.0])
        half = np.array([p.mandible_width * 0.4, 8.0, p.mandible_depth * 0.15])
        sdf = _box_sdf(coords, center, half)
        mask = sdf < 0
        hu[mask] = HU_BONE_CORTICAL + self._rng.uniform(-100, 50, int(mask.sum())).astype(np.float32)

        # Rami (bilateral)
        for side in [-1, 1]:
            c = np.array([side * p.mandible_width * 0.35, -p.skull_height * 0.25, -p.mandible_depth * 0.15])
            h = np.array([6.0, p.skull_height * 0.15, p.mandible_depth * 0.12])
            sdf_r = _box_sdf(coords, c, h)
            mask_r = sdf_r < 0
            hu[mask_r] = HU_BONE_CORTICAL + self._rng.uniform(-100, 50, int(mask_r.sum())).astype(np.float32)

    def _place_nasal_bones(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Paired nasal bones — thin plates at the nasal bridge."""
        center = np.array([0.0, p.nasal_dorsal_length * 0.45, p.nasal_dorsal_height * 0.5])
        half = np.array([
            p.nasal_bone_width / 2.0,
            p.nasal_bone_length / 2.0,
            p.nasal_bone_thickness / 2.0,
        ])
        sdf = _box_sdf(coords, center, half)
        mask = sdf < 0
        hu[mask] = HU_BONE_CORTICAL + self._rng.uniform(-200, 0, int(mask.sum())).astype(np.float32)

    def _place_airway(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Bilateral nasal airway — two parallel passages."""
        for side in [-1, 1]:
            lateral_offset = side * p.airway_valve_width * 0.4
            p0 = np.array([lateral_offset, -5.0, 5.0])
            p1 = np.array([lateral_offset, p.nasal_cavity_height * 0.6, -p.nasal_cavity_depth * 0.5])
            sdf = _cylinder_sdf(coords, p0, p1, p.airway_valve_width * 0.35)
            mask = sdf < 0
            hu[mask] = HU_AIR

        # Nasopharynx — wider passage posterior
        p0 = np.array([0.0, -10.0, -p.nasal_cavity_depth * 0.3])
        p1 = np.array([0.0, 15.0, -p.nasal_cavity_depth * 0.5])
        sdf_np = _cylinder_sdf(coords, p0, p1, p.airway_valve_width * 0.6)
        mask_np = sdf_np < 0
        hu[mask_np] = HU_AIR

    def _place_septum(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Nasal septum — midline cartilage plate."""
        center = np.array([0.0, p.septum_height * 0.3, p.nasal_dorsal_height * 0.2])
        half = np.array([p.septum_thickness / 2.0, p.septum_height / 2.0, p.septum_length / 2.0])
        sdf = _box_sdf(coords, center, half)
        mask = sdf < 0
        # Only place cartilage where not already airway
        place = mask & (hu > HU_AIR + 100)
        hu[place] = HU_CARTILAGE + self._rng.uniform(-30, 30, int(place.sum())).astype(np.float32)

    def _place_upper_lateral_cartilage(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Bilateral ULC — connects nasal bones to septum."""
        for side in [-1, 1]:
            center = np.array([
                side * p.ulc_width * 0.3,
                p.nasal_bone_length * 0.4,
                p.nasal_dorsal_height * 0.35,
            ])
            half = np.array([p.ulc_width / 2.5, p.ulc_length / 2.0, p.ulc_thickness / 2.0])
            sdf = _box_sdf(coords, center, half)
            mask = sdf < 0
            place = mask & (hu > HU_AIR + 100)
            hu[place] = HU_CARTILAGE + self._rng.uniform(-20, 20, int(place.sum())).astype(np.float32)

    def _place_lower_lateral_cartilage(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Bilateral LLC — tip and alar cartilage."""
        for side in [-1, 1]:
            center = np.array([
                side * p.llc_width * 0.2,
                2.0,
                p.tip_projection * 0.6,
            ])
            radii = np.array([p.llc_width / 2.5, p.llc_length / 2.0, p.llc_thickness * 2.0])
            sdf = _ellipsoid_sdf(coords, center, radii)
            mask = sdf < 0
            place = mask & (hu > HU_AIR + 100)
            hu[place] = HU_CARTILAGE + self._rng.uniform(-20, 20, int(place.sum())).astype(np.float32)

    def _place_turbinates(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Inferior and middle turbinates — bony/mucosal scrolls."""
        for side in [-1, 1]:
            # Inferior turbinate
            center = np.array([
                side * p.airway_valve_width * 0.5,
                5.0,
                -p.nasal_cavity_depth * 0.15,
            ])
            radii = np.array([p.turbinate_size * 0.3, p.turbinate_size * 0.8, p.turbinate_size * 0.4])
            sdf = _ellipsoid_sdf(coords, center, radii)
            mask = sdf < 0
            hu[mask] = HU_MUCOSA + self._rng.uniform(-10, 30, int(mask.sum())).astype(np.float32)

            # Middle turbinate (smaller, more superior)
            center_mid = center + np.array([0, 10.0, 0])
            radii_mid = radii * 0.7
            sdf_mid = _ellipsoid_sdf(coords, center_mid, radii_mid)
            mask_mid = sdf_mid < 0
            hu[mask_mid] = HU_MUCOSA + self._rng.uniform(-10, 20, int(mask_mid.sum())).astype(np.float32)

    def _place_muscle(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Mimetic muscles — thin layer deep to subcutaneous fat."""
        # Facial muscle sheet — ellipsoidal shell
        center = np.array([0.0, p.skull_height * 0.05, 5.0])
        outer_radii = np.array([p.skull_width * 0.38, p.skull_height * 0.4, p.skull_depth * 0.35])
        inner_radii = outer_radii - 3.0  # 3mm thick muscle layer
        sdf_outer = _ellipsoid_sdf(coords, center, outer_radii)
        sdf_inner = _ellipsoid_sdf(coords, center, inner_radii)
        mask = (sdf_outer < 0) & (sdf_inner >= 0)
        # Only where not already a labelled structure (bone or cartilage)
        place = mask & (hu < HU_CARTILAGE - 50) & (hu > HU_AIR + 100)
        hu[place] = HU_MUSCLE + self._rng.uniform(-10, 10, int(place.sum())).astype(np.float32)

    def _place_fat(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Subcutaneous fat layer."""
        center = np.array([0.0, p.skull_height * 0.05, 8.0])
        outer_radii = np.array([p.skull_width * 0.40, p.skull_height * 0.42, p.skull_depth * 0.37])
        inner_radii = outer_radii - np.array([
            p.fat_thickness_cheek,
            p.fat_thickness_cheek,
            p.fat_thickness_submental,
        ])
        sdf_outer = _ellipsoid_sdf(coords, center, outer_radii)
        sdf_inner = _ellipsoid_sdf(coords, center, inner_radii)
        mask = (sdf_outer < 0) & (sdf_inner >= 0)
        place = mask & (hu < HU_MUSCLE - 20) & (hu > HU_AIR + 100)
        hu[place] = HU_FAT + self._rng.uniform(-15, 15, int(place.sum())).astype(np.float32)

    def _place_skin(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Skin envelope — outermost soft tissue shell."""
        center = np.array([0.0, p.skull_height * 0.05, 8.0])
        outer_radii = np.array([p.skull_width * 0.42, p.skull_height * 0.44, p.skull_depth * 0.39])
        avg_thickness = (p.skin_thickness_dorsum + p.skin_thickness_tip + p.skin_thickness_cheek) / 3.0
        inner_radii = outer_radii - avg_thickness
        sdf_outer = _ellipsoid_sdf(coords, center, outer_radii)
        sdf_inner = _ellipsoid_sdf(coords, center, inner_radii)
        mask = (sdf_outer < 0) & (sdf_inner >= 0)
        place = mask & (hu < HU_FAT - 20)
        hu[place] = HU_SOFT_TISSUE + self._rng.uniform(-5, 15, int(place.sum())).astype(np.float32)

    def _place_mucosa(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Nasal mucosa — lining the airway surfaces."""
        thickness = 2.0  # mm
        for side in [-1, 1]:
            lateral_offset = side * p.airway_valve_width * 0.4
            p0 = np.array([lateral_offset, -5.0, 5.0])
            p1 = np.array([lateral_offset, p.nasal_cavity_height * 0.6, -p.nasal_cavity_depth * 0.5])
            inner_r = p.airway_valve_width * 0.35
            outer_r = inner_r + thickness
            sdf_inner = _cylinder_sdf(coords, p0, p1, inner_r)
            sdf_outer = _cylinder_sdf(coords, p0, p1, outer_r)
            mask = (sdf_outer < 0) & (sdf_inner >= 0)
            place = mask & (hu > HU_AIR + 100)
            hu[place] = HU_MUCOSA + self._rng.uniform(-5, 10, int(place.sum())).astype(np.float32)

    # ── Additional anatomical structures ──────────────────────

    def _place_orbits(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Bilateral orbital cavities — air-filled eye sockets carved into skull."""
        for side in [-1, 1]:
            center = np.array([
                side * p.interpupillary_distance * 0.5,
                p.face_height_upper * 0.5,
                p.skull_depth * 0.05,
            ])
            radii = np.array([
                p.orbit_width / 2.0,
                p.orbit_height / 2.0,
                p.orbit_width / 2.5,
            ])
            sdf = _ellipsoid_sdf(coords, center, radii)
            # Carve orbital cavity (fill with soft tissue, not air — globe)
            mask = sdf < 0
            # Inner core = globe (vitreous humor ≈ water ≈ 0 HU)
            inner_radii = radii * 0.6
            sdf_inner = _ellipsoid_sdf(coords, center, inner_radii)
            globe = sdf_inner < 0
            hu[globe] = HU_SOFT_TISSUE + self._rng.uniform(-5, 5, int(globe.sum())).astype(np.float32)
            # Periorbital fat fills the rest of the orbit
            orbital_fat = mask & (~globe)
            hu[orbital_fat] = HU_FAT + self._rng.uniform(-20, 20, int(orbital_fat.sum())).astype(np.float32)

    def _place_zygomatic(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Bilateral zygomatic arches — bony cheekbone projections."""
        for side in [-1, 1]:
            # Main body of zygoma
            center = np.array([
                side * p.skull_width * 0.35,
                p.face_height_upper * 0.25,
                -p.skull_depth * 0.02,
            ])
            half = np.array([8.0, 7.0, 10.0])
            sdf = _box_sdf(coords, center, half)
            mask = sdf < 0
            hu[mask] = HU_BONE_CORTICAL + self._rng.uniform(-150, 50, int(mask.sum())).astype(np.float32)

            # Zygomatic arch (thin bar connecting to temporal bone)
            arch_p0 = center + np.array([side * 5.0, 3.0, -5.0])
            arch_p1 = np.array([
                side * p.skull_width * 0.42,
                p.face_height_upper * 0.35,
                -p.skull_depth * 0.15,
            ])
            sdf_arch = _cylinder_sdf(coords, arch_p0, arch_p1, 3.5)
            mask_arch = sdf_arch < 0
            hu[mask_arch] = HU_BONE_CORTICAL + self._rng.uniform(-100, 50, int(mask_arch.sum())).astype(np.float32)

    def _place_pyriform(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Pyriform aperture — the bony nasal opening carved into the maxilla."""
        center = np.array([0.0, p.nasal_bone_length * 0.1, p.nasal_dorsal_height * 0.3])
        # Pear-shaped opening: wider at bottom
        radii_top = np.array([p.alar_width * 0.22, p.nasal_bone_length * 0.35, 3.0])
        radii_bot = np.array([p.alar_width * 0.28, p.nasal_bone_length * 0.3, 3.5])

        center_top = center + np.array([0.0, p.nasal_bone_length * 0.2, 0.0])
        center_bot = center - np.array([0.0, p.nasal_bone_length * 0.15, 0.0])

        sdf_top = _ellipsoid_sdf(coords, center_top, radii_top)
        sdf_bot = _ellipsoid_sdf(coords, center_bot, radii_bot)
        # Smooth union of the two to get the pear shape
        sdf_pyriform = _smooth_union(sdf_top, sdf_bot, k=5.0)
        mask = sdf_pyriform < 0
        # Carve air through bone at the aperture
        place = mask & (hu > HU_CARTILAGE)
        hu[place] = HU_AIR + self._rng.uniform(0, 50, int(place.sum())).astype(np.float32)

    def _place_lips(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Lips — soft tissue projection at the lower face.

        Creates upper and lower lip as ellipsoidal soft tissue masses
        with a smooth blend to the surrounding skin.
        """
        # Upper lip (labrale superius area)
        upper_center = np.array([0.0, -p.face_height_lower * 0.08, 14.0])
        upper_radii = np.array([13.0, 4.5, 5.0])
        sdf_upper = _ellipsoid_sdf(coords, upper_center, upper_radii)
        mask_upper = sdf_upper < 0
        place = mask_upper & (hu < HU_SOFT_TISSUE - 20)
        hu[place] = HU_SOFT_TISSUE + self._rng.uniform(5, 20, int(place.sum())).astype(np.float32)

        # Lower lip
        lower_center = np.array([0.0, -p.face_height_lower * 0.22, 12.0])
        lower_radii = np.array([14.0, 5.0, 5.5])
        sdf_lower = _ellipsoid_sdf(coords, lower_center, lower_radii)
        mask_lower = sdf_lower < 0
        place = mask_lower & (hu < HU_SOFT_TISSUE - 20)
        hu[place] = HU_SOFT_TISSUE + self._rng.uniform(5, 20, int(place.sum())).astype(np.float32)

        # Oral fissure (thin air gap between lips)
        oral_center = np.array([0.0, -p.face_height_lower * 0.15, 13.0])
        oral_half = np.array([12.0, 1.0, 3.0])
        sdf_oral = _box_sdf(coords, oral_center, oral_half)
        mask_oral = sdf_oral < 0
        hu[mask_oral] = HU_AIR

    def _place_nasal_tip_detail(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Enhanced nasal tip and alar detail using smooth blending.

        Produces more anatomically correct tip with:
        - Dome highlight
        - Alar lobule thickening
        - Columella definition
        """
        tip_z = p.tip_projection * 0.8

        # Tip dome — small prominent sphere at nasal tip
        dome_center = np.array([0.0, 4.0, tip_z])
        dome_radius = np.array([4.0, 3.0, 4.0])
        sdf_dome = _ellipsoid_sdf(coords, dome_center, dome_radius)
        mask = sdf_dome < 0
        place = mask & (hu > HU_AIR + 100) & (hu < HU_BONE_CANCELLOUS)
        hu[place] = HU_CARTILAGE + self._rng.uniform(-10, 10, int(place.sum())).astype(np.float32)

        # Alar lobules — thickened soft tissue alongside the nostrils
        for side in [-1, 1]:
            alar_center = np.array([
                side * p.alar_width * 0.38,
                0.0,
                p.tip_projection * 0.35,
            ])
            alar_radii = np.array([5.0, 5.0, 6.0])
            sdf_alar = _ellipsoid_sdf(coords, alar_center, alar_radii)
            mask_alar = sdf_alar < 0
            place = mask_alar & (hu < HU_SOFT_TISSUE - 10)
            hu[place] = HU_SOFT_TISSUE + self._rng.uniform(5, 25, int(place.sum())).astype(np.float32)

        # Columella — midline strut between nostrils
        col_p0 = np.array([0.0, -2.0, p.tip_projection * 0.15])
        col_p1 = np.array([0.0, 3.0, tip_z * 0.7])
        sdf_col = _cylinder_sdf(coords, col_p0, col_p1, 2.5)
        mask_col = sdf_col < 0
        place = mask_col & (hu > HU_AIR + 100)
        hu[place] = np.where(
            hu[place] < HU_CARTILAGE - 50,
            HU_CARTILAGE + self._rng.uniform(-20, 10, int(place.sum())).astype(np.float32),
            hu[place],
        )

    def _place_forehead(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Frontal bone contour — superiorly convex forehead.

        Adds prominence to the frontal bone region of the skull
        for more realistic head silhouette.
        """
        center = np.array([0.0, p.skull_height * 0.38, p.skull_depth * 0.12])
        radii = np.array([
            p.skull_width * 0.32,
            p.skull_height * 0.12,
            p.skull_depth * 0.15,
        ])
        sdf = _ellipsoid_sdf(coords, center, radii)
        mask = sdf < 0
        # Only place where not already dense bone
        place = mask & (hu < HU_BONE_CANCELLOUS)
        hu[place] = HU_BONE_CORTICAL + self._rng.uniform(-100, 50, int(place.sum())).astype(np.float32)

        # Supraorbital ridge — brow prominence
        for side in [-1, 1]:
            ridge_center = np.array([
                side * p.intercanthal_distance * 0.55,
                p.face_height_upper * 0.55,
                p.skull_depth * 0.08,
            ])
            ridge_radii = np.array([8.0, 4.0, 6.0])
            sdf_ridge = _ellipsoid_sdf(coords, ridge_center, ridge_radii)
            mask_ridge = sdf_ridge < 0
            hu[mask_ridge] = HU_BONE_CORTICAL + self._rng.uniform(-80, 30, int(mask_ridge.sum())).astype(np.float32)

    def _place_chin(
        self, hu: np.ndarray, coords: np.ndarray, p: AnthropometricProfile,
    ) -> None:
        """Mentum (chin point) — anterior bony prominence at the mandible.

        Adds a pogonion projection for realistic profile contour.
        """
        sex_scale = 1.1 if p.sex == "M" else 0.9
        chin_center = np.array([
            0.0,
            -p.face_height_lower * 0.55,
            8.0 * sex_scale,
        ])
        chin_radii = np.array([10.0 * sex_scale, 8.0, 7.0 * sex_scale])
        sdf = _ellipsoid_sdf(coords, chin_center, chin_radii)
        mask = sdf < 0
        # Bone core
        inner_radii = chin_radii - 4.0
        sdf_inner = _ellipsoid_sdf(coords, chin_center, inner_radii)
        bone_shell = mask & (sdf_inner >= 0)
        hu[bone_shell] = HU_BONE_CORTICAL + self._rng.uniform(-100, 50, int(bone_shell.sum())).astype(np.float32)
        # Soft tissue covering
        soft = mask & (sdf_inner < 0) & (hu < HU_SOFT_TISSUE - 20)
        hu[soft] = HU_SOFT_TISSUE + self._rng.uniform(-5, 15, int(soft.sum())).astype(np.float32)

    # ── Parametric face mesh ──────────────────────────────────

    def generate_parametric_face_mesh(
        self,
        profile: AnthropometricProfile,
        *,
        n_lat: int = 200,
        n_lon: int = 300,
    ) -> SurfaceMesh:
        """Generate a high-fidelity parametric face surface mesh.

        Builds a recognisable human face from an asymmetric ellipsoidal
        base (shallow anterior face, deep posterior cranium) with
        height-varying half-widths for jaw narrowing and ~35 Gaussian
        displacement bumps positioned per craniofacial anthropometry
        (Farkas 1994).

        Displacement is applied along the **ellipsoid surface normal**
        (gradient of the implicit function) rather than along the
        radial direction, ensuring that anterior features (nose, lips,
        chin) protrude in +Z and lateral features (cheekbones)
        protrude in +/-X regardless of the oblate base geometry.

        Coordinate system (matching CT volume):
            X -- right (+), left (-)
            Y -- superior (+), inferior (-)
            Z -- anterior (+), posterior (-)
            Origin at subnasale.

        Parameters
        ----------
        profile : AnthropometricProfile
            Facial measurement set (used for proportional scaling).
        n_lat : int
            Polar resolution (top to bottom).
        n_lon : int
            Azimuthal resolution.

        Returns
        -------
        SurfaceMesh
        """
        # ── 0.  Anthropometric scale factors ─────────────────
        nose_length: float = getattr(profile, "nose_length", 48.0)
        nose_protrusion: float = getattr(profile, "nose_protrusion", 21.0)
        nose_width: float = getattr(profile, "nasal_width", 35.0)
        face_height: float = getattr(profile, "face_height", 120.0)
        bizyg_width: float = getattr(profile, "bizygomatic_breadth", 140.0)

        s_vert = face_height / 120.0
        s_lat = bizyg_width / 140.0
        s_nose_p = nose_protrusion / 21.0
        s_nose_w = nose_width / 35.0
        s_nose_l = nose_length / 48.0

        # ── 1.  Parameter grid ───────────────────────────────
        theta = np.linspace(0.0, np.pi, n_lat)
        phi = np.linspace(-np.pi, np.pi, n_lon, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing="ij")

        st, ct = np.sin(TH), np.cos(TH)
        sp, cp = np.sin(PH), np.cos(PH)
        t = ct  # +1 top, -1 bottom

        # ── 2.  Asymmetric base ellipsoid ────────────────────
        # Half-width (X) -- with height-dependent jaw narrowing.
        RX = np.full_like(t, 62.0 * s_lat)
        RX = np.where(
            t > 0.4,
            62.0 * s_lat - 18.0 * np.abs((t - 0.4) / 0.6) ** 1.6,
            RX,
        )
        RX = np.where(
            (t <= -0.05) & (t > -0.5),
            62.0 * s_lat - 28.0 * s_lat * np.abs((-0.05 - t) / 0.45) ** 1.2,
            RX,
        )
        RX = np.where(
            t <= -0.5,
            np.maximum(6.0, 34.0 * s_lat + 50.0 * s_lat * (t + 1.0) / 0.5),
            RX,
        )

        # Half-height (Y).
        RY = 55.0 * s_vert

        # Half-depth (Z) -- ASYMMETRIC: shallow face, deep cranium.
        # Smooth blend: cp=+1 (face front) -> RZ_front,
        #               cp=-1 (back)       -> RZ_back.
        blend_front = np.clip(0.5 * (1.0 + cp), 0.0, 1.0)
        RZ_FRONT = 8.0
        RZ_BACK = 45.0
        RZ = RZ_FRONT * blend_front + RZ_BACK * (1.0 - blend_front)
        # Slight jaw-depth reduction
        RZ = np.where(t < -0.3, RZ - 6.0 * np.abs((-0.3 - t) / 0.7) ** 2, RZ)
        RZ = np.maximum(RZ, 4.0)

        # ── 3.  Base vertex positions ────────────────────────
        # phi=0 -> face front (+Z),  phi=pi/2 -> right (+X)
        X = RX * st * sp
        Y = RY * ct
        Z = RZ * st * cp

        # ── 4.  Face weight ──────────────────────────────────
        face_w = np.clip(cp, 0.0, 1.0) ** 1.4

        # ── 5.  Displacement field  D(X, Y)  ─────────────────
        eps = 1e-8

        def G(u0: float, v0: float, su: float, sv: float) -> np.ndarray:
            return np.exp(-0.5 * ((X - u0) / su) ** 2
                         - 0.5 * ((Y - v0) / sv) ** 2)

        def G_rot(
            u0: float, v0: float,
            s_major: float, s_minor: float,
            angle_deg: float,
        ) -> np.ndarray:
            a = np.radians(angle_deg)
            ca, sa = np.cos(a), np.sin(a)
            du, dv = X - u0, Y - v0
            d1 = du * ca + dv * sa
            d2 = -du * sa + dv * ca
            return np.exp(-0.5 * (d1 / s_major) ** 2
                         - 0.5 * (d2 / s_minor) ** 2)

        D = np.zeros_like(TH)

        # ── 5a. Face flattening ──────────────────────────────
        D -= face_w * 5.0

        # ── 5b. Occipital prominence ─────────────────────────
        back_w = np.clip(-cp, 0.0, 1.0) ** 2
        D += back_w * G(0, 15, 40, 30) * 6.0

        # ── 5c. Forehead ─────────────────────────────────────
        D += G(0, 48, 36, 12) * face_w * 4.0
        D += G(18, 48, 14, 10) * face_w * 1.5
        D += G(-18, 48, 14, 10) * face_w * 1.5

        # ── 5d. Brow ridge ───────────────────────────────────
        D += G(0, 35, 30, 4.5) * face_w * 3.5
        D += G(0, 38, 8, 5) * face_w * 2.0

        # ── 5e. Supraorbital ridge ───────────────────────────
        D += G(24, 37, 12, 3.0) * face_w * 2.5
        D += G(-24, 37, 12, 3.0) * face_w * 2.5

        # ── 5f. Eye sockets ──────────────────────────────────
        D -= G(30, 32, 11, 8) * face_w * 12.0
        D -= G(-30, 32, 11, 8) * face_w * 12.0
        D -= G(17, 33, 6, 7) * face_w * 5.0
        D -= G(-17, 33, 6, 7) * face_w * 5.0

        # ── 5g. Infraorbital rim ─────────────────────────────
        D += G(30, 25, 13, 3) * face_w * 3.0
        D += G(-30, 25, 13, 3) * face_w * 3.0

        # ── 5h. Nose root / nasion ───────────────────────────
        D += G(0, 33, 5, 6) * face_w * 5.0 * s_nose_p

        # ── 5i. Nose bridge (dorsum) ─────────────────────────
        nose_len = 28.0 * s_nose_l
        D += G(0, 15, 5.0, nose_len * 0.45) * face_w * 12.0 * s_nose_p
        D += G(0, 12, 4.0, 5) * face_w * 2.0 * s_nose_p

        # ── 5j. Nose tip (pronasale) ─────────────────────────
        D += G(0, 3, 7.0, 5.0) * face_w * 18.0 * s_nose_p
        D += G(3.5, 3, 2.8, 2.8) * face_w * 3.0 * s_nose_p
        D += G(-3.5, 3, 2.8, 2.8) * face_w * 3.0 * s_nose_p

        # ── 5k. Columella ────────────────────────────────────
        D += G(0, -1, 2.5, 3) * face_w * 5.0 * s_nose_p

        # ── 5l. Nasal alae ───────────────────────────────────
        ala_x = 14.0 * s_nose_w
        D += G(ala_x, 1, 5.5, 4.0) * face_w * 5.5
        D += G(-ala_x, 1, 5.5, 4.0) * face_w * 5.5
        D -= G(ala_x + 3, -1, 2.5, 2.5) * face_w * 2.0
        D -= G(-ala_x - 3, -1, 2.5, 2.5) * face_w * 2.0

        # ── 5m. Nostrils (pyriform concavity) ────────────────
        D -= G(7, -3, 3.5, 3.0) * face_w * 4.0
        D -= G(-7, -3, 3.5, 3.0) * face_w * 4.0

        # ── 5n. Nasal sidewalls ──────────────────────────────
        D += G(10, 12, 5.0, 10.0) * face_w * 3.5
        D += G(-10, 12, 5.0, 10.0) * face_w * 3.5

        # ── 5o. Cheekbones (zygomatic) ───────────────────────
        D += G(48 * s_lat, 10, 13, 9) * face_w * 7.0
        D += G(-48 * s_lat, 10, 13, 9) * face_w * 7.0
        D += G(55 * s_lat, 15, 10, 6) * 3.5
        D += G(-55 * s_lat, 15, 10, 6) * 3.5

        # ── 5p. Malar hollow ─────────────────────────────────
        D -= G(38 * s_lat, -2, 10, 8) * face_w * 3.0
        D -= G(-38 * s_lat, -2, 10, 8) * face_w * 3.0

        # ── 5q. Nasolabial folds ─────────────────────────────
        for sign in (1.0, -1.0):
            for k in range(6):
                frac = k / 5.0
                fu = sign * (ala_x - 2 + frac * 14.0)
                fv = -2.0 - frac * 16.0
                D -= G_rot(fu, fv, 4.0, 2.0,
                           sign * (10 + frac * 25)) * face_w * 1.8

        # ── 5r. Philtrum ─────────────────────────────────────
        D -= G(0, -5, 2.8, 4.5) * face_w * 2.5
        D += G(3.5, -5, 1.5, 4.5) * face_w * 1.2
        D += G(-3.5, -5, 1.5, 4.5) * face_w * 1.2

        # ── 5s. Upper lip ────────────────────────────────────
        D += G(0, -10, 12, 3.5) * face_w * 5.0
        D += G(4, -9, 3.0, 2.0) * face_w * 1.8
        D += G(-4, -9, 3.0, 2.0) * face_w * 1.8
        D += G(0, -11, 5, 2.5) * face_w * 2.0

        # ── 5t. Lower lip ────────────────────────────────────
        D += G(0, -17, 12, 3.5) * face_w * 4.5
        D += G(0, -16, 8, 2.5) * face_w * 1.8

        # ── 5u. Oral fissure ─────────────────────────────────
        D -= G(0, -13, 13, 1.2) * face_w * 2.0
        D -= G(15, -13, 3, 2) * face_w * 1.2
        D -= G(-15, -13, 3, 2) * face_w * 1.2

        # ── 5v. Labiomental groove ───────────────────────────
        D -= G(0, -24, 12, 3.5) * face_w * 3.0

        # ── 5w. Chin (mentum) ────────────────────────────────
        D += G(0, -38, 14, 8) * face_w * 8.0
        D += G(0, -39, 5, 3) * face_w * 3.5
        D += G(0, -40, 7, 4) * face_w * 2.5

        # ── 5x. Temple hollowing ─────────────────────────────
        D -= G(52 * s_lat, 35, 10, 12) * 4.0
        D -= G(-52 * s_lat, 35, 10, 12) * 4.0

        # ── 5y. Jaw angle / masseteric ───────────────────────
        D += G(48 * s_lat, -18, 10, 8) * 4.5
        D += G(-48 * s_lat, -18, 10, 8) * 4.5
        D += G(35 * s_lat, -32, 12, 4) * face_w * 2.5
        D += G(-35 * s_lat, -32, 12, 4) * face_w * 2.5

        # ── 5z. Neck transition ──────────────────────────────
        neck_drop = np.clip(-ct - 0.55, 0.0, 0.45) / 0.45
        neck_back = np.clip(-cp, 0.0, 1.0)
        D -= neck_drop * neck_back * 6.0

        # ── 6.  Displace along ellipsoid surface normal ──────
        # Gradient of implicit x^2/RX^2 + y^2/RY^2 + z^2/RZ^2 = 1
        # gives the outward normal direction.  This ensures that
        # anterior features push in +Z and lateral features push
        # in +/-X, regardless of the oblate front geometry.
        RX_sq = RX ** 2 + eps
        RY_sq = RY ** 2 + eps
        RZ_sq = RZ ** 2 + eps

        nx = X / RX_sq
        ny = Y / RY_sq
        nz = Z / RZ_sq
        n_mag = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        n_mag = np.maximum(n_mag, eps)
        nx /= n_mag
        ny /= n_mag
        nz /= n_mag

        X = X + D * nx
        Y = Y + D * ny
        Z = Z + D * nz

        # ── 7.  Origin alignment ─────────────────────────────
        # The CT coordinate system origin is at subnasale (base
        # of nose at Y = 0).  The parametric sphere equator
        # (Y = 0) aligns with the subnasale region already.  A
        # small vertical tweak centres the landmarks correctly.
        Y -= 1.0

        # ── 8.  Build triangle mesh ──────────────────────────
        verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float64)

        faces_list: list[np.ndarray] = []
        for i in range(n_lat - 1):
            for j in range(n_lon):
                jn = (j + 1) % n_lon
                v0 = i * n_lon + j
                v1 = i * n_lon + jn
                v2 = (i + 1) * n_lon + j
                v3 = (i + 1) * n_lon + jn
                faces_list.append([v0, v2, v1])
                faces_list.append([v1, v2, v3])
        triangles = np.array(faces_list, dtype=np.int64)

        # ── 9.  Light Laplacian smoothing (3 passes) ─────────
        verts = self._laplacian_smooth(
            verts, triangles, iterations=3, lamb=0.4,
        )

        # ── 10. Region labelling ─────────────────────────────
        labels = np.zeros(len(verts), dtype=np.int8)
        vx, vy, vz = verts[:, 0], verts[:, 1], verts[:, 2]
        front = vz > 0
        labels[:] = 0                                                # cranium
        labels[front & (vy > 38)] = 1                                # forehead
        labels[front & (np.abs(vx) > 15) & (np.abs(vx) < 42)
               & (vy > 24) & (vy < 42)] = 2                         # orbit
        labels[front & (np.abs(vx) < 15) & (vy > -8)
               & (vy < 35)] = 3                                      # nose
        labels[front & (np.abs(vx) > 25)
               & (vy > -15) & (vy < 25)] = 4                        # cheek
        labels[front & (np.abs(vx) < 18)
               & (vy >= -15) & (vy < -8)] = 5                       # upper lip
        labels[front & (np.abs(vx) < 18)
               & (vy >= -22) & (vy < -15)] = 6                      # lower lip
        labels[front & (vy < -30) & (vy > -50)] = 7                 # chin
        labels[front & (np.abs(vx) > 30)
               & (vy < -15) & (vy > -40)] = 8                       # jaw

        # ── 11. Assemble SurfaceMesh ─────────────────────────
        mesh = SurfaceMesh(
            vertices=verts,
            triangles=triangles,
            vertex_labels=labels,
            metadata={
                "generator": "parametric_face_v2",
                "n_lat": n_lat,
                "n_lon": n_lon,
            },
        )
        mesh.compute_normals()

        logger.info(
            "Generated parametric face mesh: %d verts, %d tris, "
            "X=[%.1f,%.1f] Y=[%.1f,%.1f] Z=[%.1f,%.1f]",
            mesh.n_vertices, mesh.n_faces,
            verts[:, 0].min(), verts[:, 0].max(),
            verts[:, 1].min(), verts[:, 1].max(),
            verts[:, 2].min(), verts[:, 2].max(),
        )
        return mesh

    # ── Surface extraction ────────────────────────────────────

    def extract_facial_surface(
        self,
        volume_hu: np.ndarray,
        voxel_spacing_mm: Tuple[float, float, float],
        origin_mm: np.ndarray,
        *,
        threshold_hu: float = -200.0,
    ) -> SurfaceMesh:
        """Extract the outer facial skin surface from a CT volume.

        Uses scikit-image ``marching_cubes`` for smooth, interpolated
        isosurface extraction when available.  Falls back to binary
        face-adjacency extraction otherwise.

        Parameters
        ----------
        volume_hu : (D, H, W) array
            CT volume.
        voxel_spacing_mm : tuple
            Voxel spacing (z, y, x).
        origin_mm : (3,) array
            Physical origin of voxel [0, 0, 0].
        threshold_hu : float
            HU threshold for surface extraction.

        Returns
        -------
        SurfaceMesh of the outer skin surface.
        """
        sz, sy, sx = voxel_spacing_mm

        if _HAS_SKIMAGE:
            # Real marching cubes with proper edge interpolation.
            # Volume is indexed [dim0=z, dim1=y, dim2=x]; spacing
            # must match that axis order.
            verts, faces, normals_mc, _ = _skimage_marching_cubes(
                volume_hu,
                level=threshold_hu,
                spacing=(sz, sy, sx),
                step_size=1,
                allow_degenerate=False,
            )

            # Convert from array-index order (z, y, x) to world order
            # (x, y, z).  The column swap is a single transposition
            # (changes handedness), so flip face winding to compensate.
            vertices = verts[:, [2, 1, 0]].copy().astype(np.float64)
            triangles = faces[:, [0, 2, 1]].copy().astype(np.int64)
        else:
            # Fallback: binary face-adjacency extraction (staircase mesh)
            binary = (volume_hu > threshold_hu).astype(np.float32)
            vertices, triangles = self._marching_cubes_simple(binary, (sx, sy, sz))
            vertices = vertices.astype(np.float64)
            triangles = triangles.astype(np.int64)

        if len(vertices) == 0:
            raise ValueError("No surface extracted — check volume and threshold")

        # Offset to physical coordinates
        vertices[:, 0] += origin_mm[0]
        vertices[:, 1] += origin_mm[1]
        vertices[:, 2] += origin_mm[2]

        # Taubin λ|μ Laplacian smoothing for clean surface
        vertices = self._laplacian_smooth(
            vertices, triangles, iterations=8, lamb=0.5,
        )

        mesh = SurfaceMesh(
            vertices=vertices,
            triangles=triangles,
        )
        mesh.compute_normals()

        # Keep only the largest connected component (the face)
        mesh = self._largest_component(mesh)

        logger.info(
            "Extracted facial surface: %d vertices, %d triangles (method=%s)",
            mesh.n_vertices, mesh.n_faces,
            "skimage_mc" if _HAS_SKIMAGE else "binary_face_adjacency",
        )
        return mesh

    @staticmethod
    def _laplacian_smooth(
        vertices: np.ndarray,
        triangles: np.ndarray,
        iterations: int = 8,
        lamb: float = 0.5,
    ) -> np.ndarray:
        """Taubin λ|μ Laplacian smoothing (vectorised via sparse matrix).

        Alternates positive (λ) and negative (μ) steps to smooth the
        mesh without the uniform shrinkage that plain Laplacian
        smoothing produces.

        Parameters
        ----------
        vertices : (V, 3)
        triangles : (F, 3)
        iterations : int
            Number of smoothing passes (even number recommended).
        lamb : float
            Positive smoothing weight (0 < λ < 1).

        Returns
        -------
        (V, 3) smoothed vertex positions.
        """
        n = len(vertices)
        if n == 0 or len(triangles) == 0:
            return vertices

        # Build sparse adjacency matrix
        rows: List[int] = []
        cols: List[int] = []
        for tri in triangles:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            rows.extend([i, j, i, k, j, k])
            cols.extend([j, i, k, i, k, j])

        data = np.ones(len(rows), dtype=np.float64)
        adj = lil_matrix((n, n), dtype=np.float64)
        for r, c in zip(rows, cols):
            adj[r, c] = 1.0
        adj = adj.tocsr()

        degree = np.array(adj.sum(axis=1)).flatten()
        degree[degree == 0] = 1.0  # prevent division by zero

        mu = -lamb - 0.02  # Taubin's μ — slightly stronger than -λ

        verts = vertices.astype(np.float64).copy()
        for it in range(iterations):
            weight = lamb if it % 2 == 0 else mu
            neighbour_mean = (adj @ verts) / degree[:, None]
            verts += weight * (neighbour_mean - verts)

        return verts

    def _marching_cubes_simple(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised binary surface extraction via face adjacency.

        For every pair of adjacent voxels that differ (one inside,
        one outside), emits a quad (two triangles) on the shared
        face.  Produces a watertight, consistently-oriented mesh for
        any binary volume.  No lookup-table required.

        Parameters
        ----------
        volume : (D, H, W) float32 array
            Binary (0/1) volume.
        spacing : (sx, sy, sz) tuple
            Voxel size in mm per axis.

        Returns
        -------
        vertices : (V, 3) float64 — physical coordinates.
        triangles : (F, 3) int64  — face indices.
        """
        sx, sy, sz = spacing
        level = 0.5

        all_verts: List[np.ndarray] = []
        all_tris: List[np.ndarray] = []
        vert_offset = 0

        def _emit_axis(axis: int) -> None:
            nonlocal vert_offset
            a = np.take(volume, range(volume.shape[axis] - 1), axis=axis)
            b = np.take(volume, range(1, volume.shape[axis]), axis=axis)
            straddle = (a >= level) != (b >= level)
            inside_a = a >= level

            zs, ys, xs = np.where(straddle)
            if len(zs) == 0:
                return

            n_faces = len(zs)

            if axis == 0:
                z_pos = (zs + 1).astype(np.float64) * sz
                x0 = xs.astype(np.float64) * sx
                x1 = (xs + 1).astype(np.float64) * sx
                y0 = ys.astype(np.float64) * sy
                y1 = (ys + 1).astype(np.float64) * sy
                p0 = np.column_stack([x0, y0, z_pos])
                p1 = np.column_stack([x1, y0, z_pos])
                p2 = np.column_stack([x1, y1, z_pos])
                p3 = np.column_stack([x0, y1, z_pos])
            elif axis == 1:
                y_pos = (ys + 1).astype(np.float64) * sy
                x0 = xs.astype(np.float64) * sx
                x1 = (xs + 1).astype(np.float64) * sx
                z0 = zs.astype(np.float64) * sz
                z1 = (zs + 1).astype(np.float64) * sz
                p0 = np.column_stack([x0, y_pos, z0])
                p1 = np.column_stack([x0, y_pos, z1])
                p2 = np.column_stack([x1, y_pos, z1])
                p3 = np.column_stack([x1, y_pos, z0])
            else:
                x_pos = (xs + 1).astype(np.float64) * sx
                y0 = ys.astype(np.float64) * sy
                y1 = (ys + 1).astype(np.float64) * sy
                z0 = zs.astype(np.float64) * sz
                z1 = (zs + 1).astype(np.float64) * sz
                p0 = np.column_stack([x_pos, y0, z0])
                p1 = np.column_stack([x_pos, y1, z0])
                p2 = np.column_stack([x_pos, y1, z1])
                p3 = np.column_stack([x_pos, y0, z1])

            flip = ~inside_a[zs, ys, xs]

            verts = np.empty((n_faces * 4, 3), dtype=np.float64)
            verts[0::4] = p0
            verts[1::4] = np.where(flip[:, None], p3, p1)
            verts[2::4] = p2
            verts[3::4] = np.where(flip[:, None], p1, p3)

            base = np.arange(n_faces, dtype=np.int64) * 4 + vert_offset
            tris = np.empty((n_faces * 2, 3), dtype=np.int64)
            tris[0::2, 0] = base
            tris[0::2, 1] = base + 1
            tris[0::2, 2] = base + 2
            tris[1::2, 0] = base
            tris[1::2, 1] = base + 2
            tris[1::2, 2] = base + 3

            all_verts.append(verts)
            all_tris.append(tris)
            vert_offset += len(verts)

        _emit_axis(0)
        _emit_axis(1)
        _emit_axis(2)

        if not all_verts:
            return np.empty((0, 3), np.float64), np.empty((0, 3), np.int64)

        verts = np.concatenate(all_verts)
        tris = np.concatenate(all_tris)

        # Merge duplicate vertices that share the same coordinate
        from ..data.surface_ingest import _merge_vertices
        verts_f32 = verts.astype(np.float32)
        tris_i32 = tris.astype(np.int32)
        verts_f32, tris_i32 = _merge_vertices(verts_f32, tris_i32, tol=min(spacing) * 0.01)

        return verts_f32.astype(np.float64), tris_i32.astype(np.int64)

    @staticmethod
    def _largest_component(mesh: SurfaceMesh) -> SurfaceMesh:
        """Keep only the largest connected component of a triangle mesh."""
        n = mesh.n_vertices
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for tri in mesh.triangles:
            union(int(tri[0]), int(tri[1]))
            union(int(tri[1]), int(tri[2]))

        # Count component sizes
        comp_size: Dict[int, int] = {}
        for i in range(n):
            r = find(i)
            comp_size[r] = comp_size.get(r, 0) + 1

        if not comp_size:
            return mesh

        largest_root = max(comp_size, key=comp_size.get)  # type: ignore[arg-type]
        keep_verts = np.array([find(i) == largest_root for i in range(n)])

        if keep_verts.all():
            return mesh

        # Re-index vertices
        new_idx = np.full(n, -1, dtype=np.int64)
        count = 0
        for i in range(n):
            if keep_verts[i]:
                new_idx[i] = count
                count += 1

        new_verts = mesh.vertices[keep_verts]
        # Filter triangles where all vertices are kept
        tri_mask = keep_verts[mesh.triangles[:, 0]] & keep_verts[mesh.triangles[:, 1]] & keep_verts[mesh.triangles[:, 2]]
        new_tris = new_idx[mesh.triangles[tri_mask]]

        result = SurfaceMesh(
            vertices=new_verts,
            triangles=new_tris,
        )
        result.compute_normals()
        return result

    # ── Landmark computation ──────────────────────────────────

    def compute_landmarks(
        self,
        profile: AnthropometricProfile,
    ) -> Dict[LandmarkType, Vec3]:
        """Compute ground-truth landmark positions from the profile.

        Returns landmarks in the same coordinate system as the
        generated CT volume (origin at grid centre ≈ subnasale).
        """
        p = profile
        # All positions in (x, y, z) = (lateral, superior, anterior)
        tip_z = p.tip_projection * 0.8
        sn_y = 0.0  # subnasale at origin-y

        landmarks: Dict[LandmarkType, Vec3] = {
            LandmarkType.SUBNASALE: Vec3(0.0, sn_y, p.tip_projection * 0.1),
            LandmarkType.PRONASALE: Vec3(0.0, sn_y + 3.0, tip_z),
            LandmarkType.RHINION: Vec3(0.0, p.nasal_dorsal_length * 0.5, p.nasal_dorsal_height * 0.5),
            LandmarkType.NASION: Vec3(0.0, p.nasal_dorsal_length * 0.8, p.nasal_dorsal_height * 0.3),
            LandmarkType.SELLION: Vec3(0.0, p.nasal_dorsal_length * 0.75, p.nasal_dorsal_height * 0.28),
            LandmarkType.GLABELLA: Vec3(0.0, p.nasal_dorsal_length * 0.9, p.nasal_dorsal_height * 0.2),
            LandmarkType.COLUMELLA_BREAKPOINT: Vec3(0.0, sn_y + 1.5, p.tip_projection * 0.3),
            LandmarkType.SUPRATIP_BREAKPOINT: Vec3(0.0, sn_y + 5.0, tip_z * 0.95),
            LandmarkType.TIP_DEFINING_POINT_LEFT: Vec3(-3.0, sn_y + 3.0, tip_z * 0.95),
            LandmarkType.TIP_DEFINING_POINT_RIGHT: Vec3(3.0, sn_y + 3.0, tip_z * 0.95),
            LandmarkType.ALAR_RIM_LEFT: Vec3(-p.alar_width / 2.0, sn_y + 1.0, p.tip_projection * 0.2),
            LandmarkType.ALAR_RIM_RIGHT: Vec3(p.alar_width / 2.0, sn_y + 1.0, p.tip_projection * 0.2),
            LandmarkType.ALAR_CREASE_LEFT: Vec3(-p.alar_width / 2.0 + 1.0, sn_y, p.tip_projection * 0.1),
            LandmarkType.ALAR_CREASE_RIGHT: Vec3(p.alar_width / 2.0 - 1.0, sn_y, p.tip_projection * 0.1),
            LandmarkType.POGONION: Vec3(0.0, -p.face_height_lower * 0.6, 5.0),
            LandmarkType.MENTON: Vec3(0.0, -p.face_height_lower * 0.85, 0.0),
            LandmarkType.STOMION: Vec3(0.0, -p.face_height_lower * 0.2, 10.0),
            LandmarkType.LABRALE_SUPERIUS: Vec3(0.0, -p.face_height_lower * 0.1, 13.0),
            LandmarkType.LABRALE_INFERIUS: Vec3(0.0, -p.face_height_lower * 0.25, 11.0),
            LandmarkType.ENDOCANTHION_LEFT: Vec3(-p.intercanthal_distance / 2.0, p.face_height_upper * 0.5, 8.0),
            LandmarkType.ENDOCANTHION_RIGHT: Vec3(p.intercanthal_distance / 2.0, p.face_height_upper * 0.5, 8.0),
            LandmarkType.EXOCANTHION_LEFT: Vec3(-p.interpupillary_distance / 2.0 - 5, p.face_height_upper * 0.5, 4.0),
            LandmarkType.EXOCANTHION_RIGHT: Vec3(p.interpupillary_distance / 2.0 + 5, p.face_height_upper * 0.5, 4.0),
            LandmarkType.TRICHION: Vec3(0.0, p.skull_height * 0.4, 0.0),
            LandmarkType.CHEILION_LEFT: Vec3(-15.0, -p.face_height_lower * 0.2, 8.0),
            LandmarkType.CHEILION_RIGHT: Vec3(15.0, -p.face_height_lower * 0.2, 8.0),
            LandmarkType.INTERNAL_VALVE_LEFT: Vec3(-p.airway_valve_width * 0.3, 8.0, p.nasal_dorsal_height * 0.3),
            LandmarkType.INTERNAL_VALVE_RIGHT: Vec3(p.airway_valve_width * 0.3, 8.0, p.nasal_dorsal_height * 0.3),
            LandmarkType.ANS: Vec3(0.0, -2.0, 5.0),
        }
        return landmarks

    def compute_clinical_measurements(
        self,
        profile: AnthropometricProfile,
        landmarks: Dict[LandmarkType, Vec3],
    ) -> List[ClinicalMeasurement]:
        """Derive clinical measurements from landmarks."""
        p = profile

        def dist(a: Vec3, b: Vec3) -> float:
            return (a - b).norm()

        measurements = [
            ClinicalMeasurement(
                name="nasal_dorsal_length",
                value=dist(landmarks[LandmarkType.NASION], landmarks[LandmarkType.PRONASALE]),
                unit="mm",
                landmark_pair=(LandmarkType.NASION, LandmarkType.PRONASALE),
                method="synthetic_parametric",
            ),
            ClinicalMeasurement(
                name="tip_projection",
                value=dist(landmarks[LandmarkType.SUBNASALE], landmarks[LandmarkType.PRONASALE]),
                unit="mm",
                landmark_pair=(LandmarkType.SUBNASALE, LandmarkType.PRONASALE),
                method="synthetic_parametric",
            ),
            ClinicalMeasurement(
                name="alar_width",
                value=dist(landmarks[LandmarkType.ALAR_RIM_LEFT], landmarks[LandmarkType.ALAR_RIM_RIGHT]),
                unit="mm",
                landmark_pair=(LandmarkType.ALAR_RIM_LEFT, LandmarkType.ALAR_RIM_RIGHT),
                method="synthetic_parametric",
            ),
            ClinicalMeasurement(
                name="intercanthal_distance",
                value=dist(landmarks[LandmarkType.ENDOCANTHION_LEFT], landmarks[LandmarkType.ENDOCANTHION_RIGHT]),
                unit="mm",
                landmark_pair=(LandmarkType.ENDOCANTHION_LEFT, LandmarkType.ENDOCANTHION_RIGHT),
                method="synthetic_parametric",
            ),
            ClinicalMeasurement(
                name="nasolabial_angle",
                value=p.tip_rotation,
                unit="deg",
                method="synthetic_parametric",
            ),
            ClinicalMeasurement(
                name="face_height_upper",
                value=p.face_height_upper,
                unit="mm",
                method="synthetic_parametric",
            ),
            ClinicalMeasurement(
                name="face_height_lower",
                value=p.face_height_lower,
                unit="mm",
                method="synthetic_parametric",
            ),
            ClinicalMeasurement(
                name="nasal_bone_length",
                value=p.nasal_bone_length,
                unit="mm",
                method="synthetic_parametric",
            ),
        ]
        return measurements

# No additional module-level constants needed — surface extraction
# uses face-adjacency scanning (no marching-cubes lookup tables).
