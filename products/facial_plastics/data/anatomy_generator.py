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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

    Parameters
    ----------
    coords : (N, 3) array
        Query coordinates.
    center, radii : (3,) arrays
        Ellipsoid centre and semi-axis lengths.

    Returns
    -------
    (N,) signed distance values (approximate; exact for spheres).
    """
    normalised = (coords - center) / radii
    dist_unit = np.linalg.norm(normalised, axis=1) - 1.0
    # Scale back to physical space (approximate)
    return dist_unit * np.min(radii)


def _box_sdf(
    coords: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    """Signed distance to an axis-aligned box."""
    d = np.abs(coords - center) - half_extents
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return outside + inside


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
    return np.maximum(radial, cap_dist)


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

        Uses a marching-cubes-style isosurface extraction at the
        air/tissue boundary.

        Parameters
        ----------
        volume_hu : (D, H, W) array
            CT volume.
        voxel_spacing_mm : tuple
            Voxel spacing.
        origin_mm : (3,) array
            Physical origin of [0, 0, 0].
        threshold_hu : float
            HU threshold for surface extraction.

        Returns
        -------
        SurfaceMesh of the outer skin surface.
        """
        sz, sy, sx = voxel_spacing_mm
        binary = (volume_hu > threshold_hu).astype(np.float32)

        # Simple marching-cubes extraction (pure numpy)
        vertices, triangles = self._marching_cubes_simple(binary, (sx, sy, sz))

        if len(vertices) == 0:
            raise ValueError("No surface extracted — check volume and threshold")

        # Offset to physical coordinates
        vertices[:, 0] = vertices[:, 0] + origin_mm[0]
        vertices[:, 1] = vertices[:, 1] + origin_mm[1]
        vertices[:, 2] = vertices[:, 2] + origin_mm[2]

        mesh = SurfaceMesh(
            vertices=vertices.astype(np.float64),
            triangles=triangles.astype(np.int64),
        )
        mesh.compute_normals()

        # Keep only the largest connected component (the face)
        mesh = self._largest_component(mesh)

        logger.info(
            "Extracted facial surface: %d vertices, %d triangles",
            mesh.n_vertices, mesh.n_faces,
        )
        return mesh

    def _marching_cubes_simple(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Binary surface extraction via face adjacency.

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
        dz, dy, dx = volume.shape

        vert_map: Dict[Tuple[float, float, float], int] = {}
        vertices_list: List[List[float]] = []
        triangles_list: List[List[int]] = []

        def _vert(x: float, y: float, z: float) -> int:
            key = (x, y, z)
            idx = vert_map.get(key)
            if idx is not None:
                return idx
            idx = len(vertices_list)
            vertices_list.append([x, y, z])
            vert_map[key] = idx
            return idx

        def _add_quad(
            p0: Tuple[float, float, float],
            p1: Tuple[float, float, float],
            p2: Tuple[float, float, float],
            p3: Tuple[float, float, float],
        ) -> None:
            v0, v1, v2, v3 = _vert(*p0), _vert(*p1), _vert(*p2), _vert(*p3)
            triangles_list.append([v0, v1, v2])
            triangles_list.append([v0, v2, v3])

        v = volume
        # Scan X-axis faces  (between voxels [i,j,k] and [i,j,k+1])
        for i in range(dz):
            for j in range(dy):
                for k in range(dx - 1):
                    a, b = v[i, j, k], v[i, j, k + 1]
                    if (a > 0.5) != (b > 0.5):
                        fx = (k + 1) * sx
                        y0, y1 = j * sy, (j + 1) * sy
                        z0, z1 = i * sz, (i + 1) * sz
                        if a > 0.5:
                            _add_quad((fx, y0, z0), (fx, y1, z0),
                                      (fx, y1, z1), (fx, y0, z1))
                        else:
                            _add_quad((fx, y0, z0), (fx, y0, z1),
                                      (fx, y1, z1), (fx, y1, z0))

        # Scan Y-axis faces  (between voxels [i,j,k] and [i,j+1,k])
        for i in range(dz):
            for j in range(dy - 1):
                for k in range(dx):
                    a, b = v[i, j, k], v[i, j + 1, k]
                    if (a > 0.5) != (b > 0.5):
                        fy = (j + 1) * sy
                        x0, x1 = k * sx, (k + 1) * sx
                        z0, z1 = i * sz, (i + 1) * sz
                        if a > 0.5:
                            _add_quad((x0, fy, z0), (x0, fy, z1),
                                      (x1, fy, z1), (x1, fy, z0))
                        else:
                            _add_quad((x0, fy, z0), (x1, fy, z0),
                                      (x1, fy, z1), (x0, fy, z1))

        # Scan Z-axis faces  (between voxels [i,j,k] and [i+1,j,k])
        for i in range(dz - 1):
            for j in range(dy):
                for k in range(dx):
                    a, b = v[i, j, k], v[i + 1, j, k]
                    if (a > 0.5) != (b > 0.5):
                        fz = (i + 1) * sz
                        x0, x1 = k * sx, (k + 1) * sx
                        y0, y1 = j * sy, (j + 1) * sy
                        if a > 0.5:
                            _add_quad((x0, y0, fz), (x1, y0, fz),
                                      (x1, y1, fz), (x0, y1, fz))
                        else:
                            _add_quad((x0, y0, fz), (x0, y1, fz),
                                      (x1, y1, fz), (x1, y0, fz))

        if not vertices_list:
            return np.empty((0, 3), np.float64), np.empty((0, 3), np.int64)

        return np.array(vertices_list, np.float64), np.array(triangles_list, np.int64)

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
