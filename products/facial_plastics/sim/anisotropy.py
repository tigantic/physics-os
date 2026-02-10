"""Anisotropic tissue constitutive models for facial biomechanics.

Extends the isotropic models in ``fem_soft_tissue.py`` with fiber-
direction-dependent constitutive laws.  Facial soft tissues exhibit
significant anisotropy due to:

  - Collagen fiber families in skin (Langer's lines)
  - Oriented collagen bundles in SMAS and periosteum
  - Muscle fiber directions in mimetic muscles
  - Cartilage collagen orientation (surface-tangential zones)

Material models implemented:

  1. **Holzapfel–Gasser–Ogden (HGO)** — two collagen fiber families
     with isotropic ground matrix.  Used for skin, SMAS, periosteum.
  2. **Transversely isotropic NeoHookean** — single preferred direction
     with additional fiber stretch stiffness.  For muscle, cartilage.
  3. **Fiber-reinforced Mooney–Rivlin** — Mooney–Rivlin ground matrix
     with embedded fiber stiffness.  For composite graft tissues.

Each model returns 2nd Piola–Kirchhoff stress and tangent modulus
in 6-component Voigt form, compatible with the FEM solver.

References:
  - Holzapfel, Gasser & Ogden (2000). J Elast 61:1–48
  - Gasser, Ogden & Holzapfel (2006). J R Soc Interface 3:15–35
  - Flynn, Taberner & Nielsen (2011). Biomech Model Mechanobiol 10:339
  - Annaidh et al (2012). J Mech Behav Biomed Mater 5:139–148
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import StructureType, Vec3

logger = logging.getLogger(__name__)


# ── Enumerations ──────────────────────────────────────────────────

class FiberArchitecture(Enum):
    """Fiber architecture classification for facial tissues."""
    ISOTROPIC = auto()              # no preferred direction
    UNIAXIAL = auto()               # single fiber family
    BIAXIAL_SYMMETRIC = auto()      # two symmetric fiber families
    BIAXIAL_ASYMMETRIC = auto()     # two fiber families, different angles
    PLANAR_RANDOM = auto()          # random in-plane fibers (SMAS)
    ORTHOTROPIC = auto()            # three orthogonal families


class AnisotropicModel(Enum):
    """Available anisotropic constitutive model types."""
    HGO = auto()
    TRANSVERSE_ISO_NEOHOOKEAN = auto()
    FIBER_MOONEY_RIVLIN = auto()


# ── Fiber direction field ─────────────────────────────────────────

@dataclass
class FiberFamily:
    """A single collagen fiber family.

    Attributes
    ----------
    direction : unit direction vector in reference configuration
    k1 : fiber stiffness parameter (Pa)
    k2 : fiber nonlinearity parameter (dimensionless)
    kappa_dispersion : dispersion parameter κ ∈ [0, 1/3]
        0 = perfectly aligned, 1/3 = isotropic dispersion
    """
    direction: np.ndarray
    k1: float = 50.0e3  # 50 kPa — skin collagen
    k2: float = 10.0    # moderate strain-stiffening
    kappa_dispersion: float = 0.1

    def __post_init__(self) -> None:
        d = np.asarray(self.direction, dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            raise ValueError("Fiber direction must be non-zero")
        self.direction = d / norm


@dataclass
class FiberField:
    """Spatially-varying fiber orientation field over a mesh.

    Stores per-element fiber families:
      element_families[elem_id] -> list of FiberFamily
    """
    element_families: Dict[int, List[FiberFamily]] = field(default_factory=dict)
    default_families: List[FiberFamily] = field(default_factory=list)
    architecture: FiberArchitecture = FiberArchitecture.ISOTROPIC

    def families_for_element(self, elem_id: int) -> List[FiberFamily]:
        return self.element_families.get(elem_id, self.default_families)


# ── Pre-built fiber fields for facial regions ─────────────────────

# Langer's line orientations (approximate) in local anatomical frame
# x = lateral, y = superior, z = anterior

LANGERS_LINE_DIRECTIONS: Dict[str, np.ndarray] = {
    "forehead": np.array([1.0, 0.0, 0.0]),        # transverse
    "periorbital": np.array([0.7, 0.7, 0.0]),      # oblique
    "cheek": np.array([0.5, 0.866, 0.0]),           # ~60° from horizontal
    "nasolabial": np.array([0.0, 1.0, 0.0]),        # vertical
    "chin": np.array([1.0, 0.0, 0.0]),              # transverse
    "upper_lip": np.array([1.0, 0.0, 0.0]),         # transverse
    "lower_lid": np.array([1.0, 0.0, 0.0]),         # transverse
}


def build_skin_fiber_field(
    region: str = "cheek",
    k1: float = 50.0e3,
    k2: float = 10.0,
    dispersion: float = 0.2,
    angle_offset_deg: float = 30.0,
) -> FiberField:
    """Build a biaxial-symmetric fiber field for skin.

    Two collagen fiber families are arranged symmetrically about
    the mean Langer's line direction.
    """
    base_dir = LANGERS_LINE_DIRECTIONS.get(
        region, np.array([1.0, 0.0, 0.0]),
    )
    base_dir = base_dir / np.linalg.norm(base_dir)

    # Build rotation for the offset angle
    theta = math.radians(angle_offset_deg)

    # Find a perpendicular vector in the plane
    if abs(base_dir[2]) < 0.9:
        perp = np.cross(base_dir, np.array([0.0, 0.0, 1.0]))
    else:
        perp = np.cross(base_dir, np.array([1.0, 0.0, 0.0]))
    perp = perp / np.linalg.norm(perp)

    # Two fiber directions at ±angle_offset from base
    d1 = math.cos(theta) * base_dir + math.sin(theta) * perp
    d2 = math.cos(theta) * base_dir - math.sin(theta) * perp

    families = [
        FiberFamily(direction=d1, k1=k1, k2=k2, kappa_dispersion=dispersion),
        FiberFamily(direction=d2, k1=k1, k2=k2, kappa_dispersion=dispersion),
    ]

    return FiberField(
        default_families=families,
        architecture=FiberArchitecture.BIAXIAL_SYMMETRIC,
    )


def build_muscle_fiber_field(
    muscle_direction: np.ndarray,
    k1: float = 20.0e3,
    k2: float = 5.0,
) -> FiberField:
    """Build a single-family fiber field for mimetic muscle."""
    d = np.asarray(muscle_direction, dtype=np.float64)
    d = d / max(np.linalg.norm(d), 1e-12)

    family = FiberFamily(direction=d, k1=k1, k2=k2, kappa_dispersion=0.05)

    return FiberField(
        default_families=[family],
        architecture=FiberArchitecture.UNIAXIAL,
    )


def build_smas_fiber_field(
    planar_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
    k1: float = 30.0e3,
    k2: float = 8.0,
    n_directions: int = 6,
) -> FiberField:
    """Build a planar-random fiber field for SMAS.

    SMAS (superficial musculoaponeurotic system) has collagen fibers
    randomly oriented within the fascial plane.

    Modeled as multiple equally-spaced families in the plane,
    each with high dispersion.
    """
    n = np.asarray(planar_normal, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)

    # Find two orthogonal vectors in the plane
    if abs(n[0]) < 0.9:
        v1 = np.cross(n, np.array([1.0, 0.0, 0.0]))
    else:
        v1 = np.cross(n, np.array([0.0, 1.0, 0.0]))
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(n, v1)

    families = []
    for i in range(n_directions):
        angle = math.pi * i / n_directions
        d = math.cos(angle) * v1 + math.sin(angle) * v2
        families.append(FiberFamily(
            direction=d,
            k1=k1 / n_directions,  # distribute stiffness equally
            k2=k2,
            kappa_dispersion=0.25,  # high dispersion
        ))

    return FiberField(
        default_families=families,
        architecture=FiberArchitecture.PLANAR_RANDOM,
    )


# ── Tissue-type to fiber field mapping ────────────────────────────

TISSUE_ANISOTROPY: Dict[StructureType, Dict[str, Any]] = {
    StructureType.SKIN_ENVELOPE: {
        "model": AnisotropicModel.HGO,
        "architecture": FiberArchitecture.BIAXIAL_SYMMETRIC,
        "k1": 50.0e3,
        "k2": 10.0,
        "kappa_dispersion": 0.2,
    },
    StructureType.SKIN_THICK: {
        "model": AnisotropicModel.HGO,
        "architecture": FiberArchitecture.BIAXIAL_SYMMETRIC,
        "k1": 70.0e3,
        "k2": 12.0,
        "kappa_dispersion": 0.15,
    },
    StructureType.SKIN_THIN: {
        "model": AnisotropicModel.HGO,
        "architecture": FiberArchitecture.BIAXIAL_SYMMETRIC,
        "k1": 30.0e3,
        "k2": 8.0,
        "kappa_dispersion": 0.25,
    },
    StructureType.SMAS: {
        "model": AnisotropicModel.HGO,
        "architecture": FiberArchitecture.PLANAR_RANDOM,
        "k1": 30.0e3,
        "k2": 8.0,
        "kappa_dispersion": 0.28,
    },
    StructureType.MUSCLE_MIMETIC: {
        "model": AnisotropicModel.TRANSVERSE_ISO_NEOHOOKEAN,
        "architecture": FiberArchitecture.UNIAXIAL,
        "k1": 20.0e3,
        "k2": 5.0,
        "kappa_dispersion": 0.05,
    },
    StructureType.PERIOSTEUM: {
        "model": AnisotropicModel.FIBER_MOONEY_RIVLIN,
        "architecture": FiberArchitecture.BIAXIAL_ASYMMETRIC,
        "k1": 100.0e3,
        "k2": 15.0,
        "kappa_dispersion": 0.1,
    },
}


def get_anisotropy_params(
    structure: StructureType,
) -> Optional[Dict[str, Any]]:
    """Get anisotropy parameters for a tissue type, or None if isotropic."""
    return TISSUE_ANISOTROPY.get(structure)


# ── HGO constitutive model ───────────────────────────────────────

def compute_hgo_stress(
    F: np.ndarray,
    mu: float,
    kappa_bulk: float,
    fiber_families: List[FiberFamily],
) -> Tuple[np.ndarray, np.ndarray]:
    """Holzapfel–Gasser–Ogden stress and tangent.

    Ground matrix: NeoHookean with parameters (mu, kappa_bulk).
    Fiber contribution: exponential strain-energy per family:

        Ψ_f = (k1 / 2k2) * [exp(k2 * E_bar²) - 1]

    where E_bar = κ*I1_bar + (1-3κ)*I4_bar - 1
          κ is the dispersion parameter
          I4_bar = fiber_stretch² (deviatoric)

    Parameters
    ----------
    F : (3,3) deformation gradient
    mu : ground matrix shear modulus (Pa)
    kappa_bulk : bulk modulus (Pa)
    fiber_families : list of FiberFamily

    Returns
    -------
    S : (6,) 2nd Piola-Kirchhoff stress in Voigt form
    C_tangent : (6,6) material tangent in Voigt form
    """
    C_tensor = F.T @ F
    J = np.linalg.det(F)
    if J < 1e-12:
        return np.zeros(6, dtype=np.float64), np.eye(6, dtype=np.float64) * 2.0 * mu

    C_inv = np.linalg.inv(C_tensor)
    J_23 = J ** (-2.0 / 3.0)
    I1 = np.trace(C_tensor)
    I1_bar = J_23 * I1

    I = np.eye(3, dtype=np.float64)

    # ── Ground matrix (NeoHookean) ──
    S_iso = mu * I
    S_dev = J_23 * (S_iso - (np.trace(S_iso @ C_tensor) / 3.0) * C_inv)
    S_vol = kappa_bulk * (J - 1.0) * J * C_inv
    S_ground = S_dev + S_vol

    # ── Fiber contributions ──
    S_fiber = np.zeros((3, 3), dtype=np.float64)
    C_fiber = np.zeros((6, 6), dtype=np.float64)

    for fam in fiber_families:
        a0 = fam.direction
        A0 = np.outer(a0, a0)  # structural tensor

        # Deviatoric fiber pseudo-invariant I4_bar
        I4 = float(a0 @ C_tensor @ a0)
        I4_bar = J_23 * I4

        # Dispersion-weighted invariant
        kd = fam.kappa_dispersion
        E_bar = kd * I1_bar + (1.0 - 3.0 * kd) * I4_bar - 1.0

        # Only contribute when fibers are in tension
        if E_bar <= 0.0:
            continue

        # Exponential stiffening
        exp_term = math.exp(fam.k2 * E_bar ** 2)
        dPsi_dE = fam.k1 * E_bar * exp_term

        # Generalized structural tensor H = κ*I + (1-3κ)*A0
        H = kd * I + (1.0 - 3.0 * kd) * A0

        # Deviatoric projection of H
        H_dev = J_23 * (H - (np.trace(H @ C_tensor) / 3.0) * C_inv)

        S_fiber += 2.0 * dPsi_dE * H_dev

        # Tangent contribution from fibers
        d2Psi_dE2 = fam.k1 * (1.0 + 2.0 * fam.k2 * E_bar ** 2) * exp_term
        H_voigt = _tensor_to_voigt_4th(np.einsum("ij,kl->ijkl", H_dev, H_dev))
        C_fiber += 4.0 * d2Psi_dE2 * H_voigt

    S_total = S_ground + S_fiber

    # Convert to Voigt
    S = np.array([
        S_total[0, 0], S_total[1, 1], S_total[2, 2],
        S_total[0, 1], S_total[1, 2], S_total[0, 2],
    ], dtype=np.float64)

    # Ground matrix tangent
    lam_eff = kappa_bulk * J * (2.0 * J - 1.0)
    mu_eff = mu * J_23 - kappa_bulk * (J - 1.0) * J

    inv_v = np.array([
        C_inv[0, 0], C_inv[1, 1], C_inv[2, 2],
        C_inv[0, 1], C_inv[1, 2], C_inv[0, 2],
    ], dtype=np.float64)

    C_tangent = lam_eff * np.outer(inv_v, inv_v) + 2.0 * mu_eff * np.eye(6)
    C_tangent += C_fiber

    return S, C_tangent


# ── Transversely isotropic NeoHookean ─────────────────────────────

def compute_transverse_iso_stress(
    F: np.ndarray,
    mu: float,
    kappa_bulk: float,
    fiber_family: FiberFamily,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transversely isotropic NeoHookean with one fiber family.

    Ground matrix: NeoHookean (mu, kappa_bulk)
    Fiber: exponential reinforcement along a single direction.

    Used for mimetic muscles with a well-defined fiber direction.
    """
    return compute_hgo_stress(F, mu, kappa_bulk, [fiber_family])


# ── Fiber-reinforced Mooney–Rivlin ────────────────────────────────

def compute_fiber_mooney_rivlin_stress(
    F: np.ndarray,
    C1: float,
    C2: float,
    kappa_bulk: float,
    fiber_families: List[FiberFamily],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mooney–Rivlin ground matrix with fiber reinforcement.

    For periosteum and other dense connective tissues.

    Ψ = C1*(I1_bar - 3) + C2*(I2_bar - 3) + U(J) + Ψ_fibers

    Parameters
    ----------
    F : (3,3) deformation gradient
    C1, C2 : Mooney–Rivlin parameters (Pa)
    kappa_bulk : bulk modulus (Pa)
    fiber_families : list of FiberFamily
    """
    C_tensor = F.T @ F
    J = np.linalg.det(F)
    if J < 1e-12:
        return np.zeros(6, dtype=np.float64), np.eye(6, dtype=np.float64) * 2.0 * (C1 + C2)

    C_inv = np.linalg.inv(C_tensor)
    J_23 = J ** (-2.0 / 3.0)
    I = np.eye(3, dtype=np.float64)

    I1 = np.trace(C_tensor)
    I1_bar = J_23 * I1

    # Mooney–Rivlin ground matrix
    S_iso = 2.0 * (C1 + C2 * I1) * I - 2.0 * C2 * C_tensor
    S_dev = J_23 * (S_iso - (np.trace(S_iso @ C_tensor) / 3.0) * C_inv)
    S_vol = kappa_bulk * (J - 1.0) * J * C_inv
    S_ground = S_dev + S_vol

    # Fiber contributions (same as HGO)
    S_fiber = np.zeros((3, 3), dtype=np.float64)
    C_fiber = np.zeros((6, 6), dtype=np.float64)

    for fam in fiber_families:
        a0 = fam.direction
        A0 = np.outer(a0, a0)
        I4 = float(a0 @ C_tensor @ a0)
        I4_bar = J_23 * I4
        kd = fam.kappa_dispersion
        E_bar = kd * I1_bar + (1.0 - 3.0 * kd) * I4_bar - 1.0

        if E_bar <= 0.0:
            continue

        exp_term = math.exp(fam.k2 * E_bar ** 2)
        dPsi_dE = fam.k1 * E_bar * exp_term
        H = kd * I + (1.0 - 3.0 * kd) * A0
        H_dev = J_23 * (H - (np.trace(H @ C_tensor) / 3.0) * C_inv)
        S_fiber += 2.0 * dPsi_dE * H_dev

        d2Psi_dE2 = fam.k1 * (1.0 + 2.0 * fam.k2 * E_bar ** 2) * exp_term
        H_voigt = _tensor_to_voigt_4th(np.einsum("ij,kl->ijkl", H_dev, H_dev))
        C_fiber += 4.0 * d2Psi_dE2 * H_voigt

    S_total = S_ground + S_fiber

    S = np.array([
        S_total[0, 0], S_total[1, 1], S_total[2, 2],
        S_total[0, 1], S_total[1, 2], S_total[0, 2],
    ], dtype=np.float64)

    mu_eff = 2.0 * (C1 + C2) * J_23
    lam_eff = kappa_bulk * J * (2.0 * J - 1.0)
    inv_v = np.array([
        C_inv[0, 0], C_inv[1, 1], C_inv[2, 2],
        C_inv[0, 1], C_inv[1, 2], C_inv[0, 2],
    ], dtype=np.float64)
    C_tangent = lam_eff * np.outer(inv_v, inv_v) + 2.0 * mu_eff * np.eye(6)
    C_tangent += C_fiber

    return S, C_tangent


# ── Voigt utilities ───────────────────────────────────────────────

def _tensor_to_voigt_4th(T: np.ndarray) -> np.ndarray:
    """Convert 4th-order tensor (3,3,3,3) to Voigt matrix (6,6).

    Voigt index map: 0↔00, 1↔11, 2↔22, 3↔01, 4↔12, 5↔02
    """
    voigt_map = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    C = np.zeros((6, 6), dtype=np.float64)
    for I_v, (i, j) in enumerate(voigt_map):
        for J_v, (k, l) in enumerate(voigt_map):
            C[I_v, J_v] = T[i, j, k, l]
    return C


def _voigt_to_tensor(v: np.ndarray) -> np.ndarray:
    """Convert Voigt stress vector (6,) to symmetric tensor (3,3)."""
    return np.array([
        [v[0], v[3], v[5]],
        [v[3], v[1], v[4]],
        [v[5], v[4], v[2]],
    ], dtype=np.float64)


# ── Anisotropic stress dispatcher ─────────────────────────────────

def evaluate_anisotropic_stress(
    F: np.ndarray,
    model: AnisotropicModel,
    isotropic_params: Dict[str, float],
    fiber_families: List[FiberFamily],
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-level dispatcher for anisotropic constitutive evaluation.

    Parameters
    ----------
    F : (3,3) deformation gradient
    model : which anisotropic model to use
    isotropic_params : dict with keys mu, kappa (and optionally C1, C2)
    fiber_families : oriented fiber families

    Returns
    -------
    S : (6,) 2nd Piola–Kirchhoff stress (Voigt)
    C_tangent : (6,6) material tangent (Voigt)
    """
    mu = isotropic_params.get("mu", 1e4)
    kappa = isotropic_params.get("kappa", 1e5)

    if model == AnisotropicModel.HGO:
        return compute_hgo_stress(F, mu, kappa, fiber_families)

    elif model == AnisotropicModel.TRANSVERSE_ISO_NEOHOOKEAN:
        if not fiber_families:
            raise ValueError("TRANSVERSE_ISO requires at least one fiber family")
        return compute_transverse_iso_stress(F, mu, kappa, fiber_families[0])

    elif model == AnisotropicModel.FIBER_MOONEY_RIVLIN:
        C1 = isotropic_params.get("C1", 0.5e3)
        C2 = isotropic_params.get("C2", 0.05e3)
        return compute_fiber_mooney_rivlin_stress(F, C1, C2, kappa, fiber_families)

    else:
        raise ValueError(f"Unknown anisotropic model: {model}")


# ── Effective stiffness tensor ────────────────────────────────────

def compute_effective_stiffness(
    tangent_voigt: np.ndarray,
) -> Dict[str, float]:
    """Extract effective engineering constants from a 6×6 tangent.

    Returns approximate Young's modulus, shear modulus, Poisson's ratio
    along each axis.  Useful for comparing isotropic vs. anisotropic
    models and verifying anisotropy ratios.
    """
    # Compliance = inverse of tangent
    try:
        S = np.linalg.inv(tangent_voigt)
    except np.linalg.LinAlgError:
        return {"E_1": 0.0, "E_2": 0.0, "E_3": 0.0}

    E1 = 1.0 / S[0, 0] if abs(S[0, 0]) > 1e-30 else 0.0
    E2 = 1.0 / S[1, 1] if abs(S[1, 1]) > 1e-30 else 0.0
    E3 = 1.0 / S[2, 2] if abs(S[2, 2]) > 1e-30 else 0.0

    G12 = 1.0 / S[3, 3] if abs(S[3, 3]) > 1e-30 else 0.0
    G23 = 1.0 / S[4, 4] if abs(S[4, 4]) > 1e-30 else 0.0
    G13 = 1.0 / S[5, 5] if abs(S[5, 5]) > 1e-30 else 0.0

    nu12 = -S[0, 1] * E1 if abs(E1) > 1e-30 else 0.0
    nu13 = -S[0, 2] * E1 if abs(E1) > 1e-30 else 0.0
    nu23 = -S[1, 2] * E2 if abs(E2) > 1e-30 else 0.0

    # Anisotropy ratio: max(E)/min(E)
    E_vals = [abs(E1), abs(E2), abs(E3)]
    E_nonzero = [e for e in E_vals if e > 1e-12]
    anisotropy_ratio = max(E_nonzero) / min(E_nonzero) if len(E_nonzero) >= 2 else 1.0

    return {
        "E_1": E1,
        "E_2": E2,
        "E_3": E3,
        "G_12": G12,
        "G_23": G23,
        "G_13": G13,
        "nu_12": nu12,
        "nu_13": nu13,
        "nu_23": nu23,
        "anisotropy_ratio": anisotropy_ratio,
    }


# ── Fiber field from StructureType ────────────────────────────────

def build_fiber_field_for_tissue(
    structure: StructureType,
    *,
    element_count: int = 0,
    region_name: str = "cheek",
    muscle_direction: Optional[np.ndarray] = None,
    planar_normal: Optional[np.ndarray] = None,
) -> Optional[FiberField]:
    """Build an appropriate FiberField for a given tissue type.

    Returns None if the tissue is isotropic.
    """
    params = TISSUE_ANISOTROPY.get(structure)
    if params is None:
        return None

    arch = params["architecture"]
    k1 = params["k1"]
    k2 = params["k2"]
    disp = params["kappa_dispersion"]

    if arch == FiberArchitecture.UNIAXIAL:
        direction = muscle_direction if muscle_direction is not None else np.array([0.0, 1.0, 0.0])
        return build_muscle_fiber_field(direction, k1=k1, k2=k2)

    elif arch in (FiberArchitecture.BIAXIAL_SYMMETRIC, FiberArchitecture.BIAXIAL_ASYMMETRIC):
        return build_skin_fiber_field(
            region=region_name,
            k1=k1,
            k2=k2,
            dispersion=disp,
        )

    elif arch == FiberArchitecture.PLANAR_RANDOM:
        normal = planar_normal if planar_normal is not None else np.array([0.0, 0.0, 1.0])
        return build_smas_fiber_field(normal, k1=k1, k2=k2)

    return None
