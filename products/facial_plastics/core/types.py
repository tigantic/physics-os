"""Core type definitions for the facial plastics platform."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np


# ── Enumerations ──────────────────────────────────────────────────

class Modality(enum.Enum):
    """Imaging modality for case data."""
    CT = "ct"
    CBCT = "cbct"
    MRI = "mri"
    SURFACE_SCAN = "surface_scan"
    PHOTO_2D = "photo_2d"
    ULTRASOUND = "ultrasound"
    LASER_SCAN = "laser_scan"


class StructureType(enum.Enum):
    """Anatomical structure labels for segmentation."""
    BONE_MAXILLA = "bone_maxilla"
    BONE_NASAL = "bone_nasal"
    BONE_FRONTAL = "bone_frontal"
    BONE_ZYGOMATIC = "bone_zygomatic"
    BONE_MANDIBLE = "bone_mandible"
    BONE_ORBIT = "bone_orbit"
    CARTILAGE_SEPTUM = "cartilage_septum"
    CARTILAGE_UPPER_LATERAL = "cartilage_upper_lateral"
    CARTILAGE_LOWER_LATERAL = "cartilage_lower_lateral"
    CARTILAGE_ALAR = "cartilage_alar"
    CARTILAGE_EAR = "cartilage_ear"
    AIRWAY_NASAL = "airway_nasal"
    AIRWAY_NASOPHARYNX = "airway_nasopharynx"
    AIRWAY_SINUS_MAXILLARY = "airway_sinus_maxillary"
    AIRWAY_SINUS_FRONTAL = "airway_sinus_frontal"
    SKIN_ENVELOPE = "skin_envelope"
    SKIN_THICK = "skin_thick"
    SKIN_THIN = "skin_thin"
    FAT_SUBCUTANEOUS = "fat_subcutaneous"
    FAT_BUCCAL = "fat_buccal"
    FAT_MALAR = "fat_malar"
    FAT_NASOLABIAL = "fat_nasolabial"
    MUSCLE_MIMETIC = "muscle_mimetic"
    MUSCLE_PLATYSMA = "muscle_platysma"
    SMAS = "smas"
    PERIOSTEUM = "periosteum"
    PERICHONDRIUM = "perichondrium"
    VESSEL_ARTERY = "vessel_artery"
    VESSEL_VEIN = "vessel_vein"
    NERVE = "nerve"
    MUCOSA_NASAL = "mucosa_nasal"
    TURBINATE_INFERIOR = "turbinate_inferior"
    TURBINATE_MIDDLE = "turbinate_middle"
    AIRWAY_VALVE_INTERNAL = "airway_valve_internal"
    AIRWAY_VALVE_EXTERNAL = "airway_valve_external"
    SKIN_SEBACEOUS = "skin_sebaceous"
    FAT_DEEP = "fat_deep"
    FAT_ORBITAL = "fat_orbital"
    FAT_PREAPONEUROTIC = "fat_preaponeurotic"
    MUSCLE_ORBICULARIS = "muscle_orbicularis"
    MUSCLE_FRONTALIS = "muscle_frontalis"
    MUSCLE_CORRUGATOR = "muscle_corrugator"
    MUSCLE_PROCERUS = "muscle_procerus"
    VESSEL_NASAL = "vessel_nasal"
    NERVE_NASAL = "nerve_nasal"


class ProcedureType(enum.Enum):
    """Surgical procedure categories."""
    RHINOPLASTY = "rhinoplasty"
    SEPTOPLASTY = "septoplasty"
    SEPTORHINOPLASTY = "septorhinoplasty"
    FACELIFT = "facelift"
    NECKLIFT = "necklift"
    BLEPHAROPLASTY_UPPER = "blepharoplasty_upper"
    BLEPHAROPLASTY_LOWER = "blepharoplasty_lower"
    FILLER_INJECTION = "filler_injection"
    FAT_GRAFTING = "fat_grafting"
    OTOPLASTY = "otoplasty"
    BROW_LIFT = "brow_lift"
    CHIN_AUGMENTATION = "chin_augmentation"


class MaterialModel(enum.Enum):
    """Constitutive models available for tissue simulation."""
    LINEAR_ELASTIC = "linear_elastic"
    NEO_HOOKEAN = "neo_hookean"
    MOONEY_RIVLIN = "mooney_rivlin"
    OGDEN = "ogden"
    FUNG = "fung"
    VISCOELASTIC_QLV = "viscoelastic_qlv"
    VISCOELASTIC_PRONY = "viscoelastic_prony"
    DRUCKER_PRAGER = "drucker_prager"
    J2_PLASTICITY = "j2_plasticity"
    COHESIVE_ZONE = "cohesive_zone"
    RIGID = "rigid"


class MeshElementType(enum.Enum):
    """Volumetric mesh element types."""
    TET4 = "tet4"
    TET10 = "tet10"
    HEX8 = "hex8"
    HEX20 = "hex20"
    WEDGE6 = "wedge6"
    PYRAMID5 = "pyramid5"
    TRI3 = "tri3"
    TRI6 = "tri6"
    QUAD4 = "quad4"
    QUAD8 = "quad8"
    BAR2 = "bar2"


class SolverType(enum.Enum):
    """Physics solver types available."""
    FEM_STATIC = "fem_static"
    FEM_QUASISTATIC = "fem_quasistatic"
    FEM_DYNAMIC = "fem_dynamic"
    CFD_STEADY = "cfd_steady"
    CFD_TRANSIENT = "cfd_transient"
    FSI = "fsi"
    THERMAL = "thermal"
    THERMO_MECHANICAL = "thermo_mechanical"


class QualityLevel(enum.Enum):
    """Data quality classification."""
    CLINICAL = "clinical"
    RESEARCH = "research"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class LandmarkType(enum.Enum):
    """Facial anthropometric landmark types."""
    NASION = "nasion"
    GLABELLA = "glabella"
    RHINION = "rhinion"
    PRONASALE = "pronasale"
    SUBNASALE = "subnasale"
    COLUMELLA_BREAKPOINT = "columella_breakpoint"
    ALAR_CREASE_LEFT = "alar_crease_left"
    ALAR_CREASE_RIGHT = "alar_crease_right"
    ALAR_RIM_LEFT = "alar_rim_left"
    ALAR_RIM_RIGHT = "alar_rim_right"
    TIP_DEFINING_POINT_LEFT = "tip_defining_point_left"
    TIP_DEFINING_POINT_RIGHT = "tip_defining_point_right"
    SELLION = "sellion"
    POGONION = "pogonion"
    MENTON = "menton"
    STOMION = "stomion"
    LABRALE_SUPERIUS = "labrale_superius"
    LABRALE_INFERIUS = "labrale_inferius"
    EXOCANTHION_LEFT = "exocanthion_left"
    EXOCANTHION_RIGHT = "exocanthion_right"
    ENDOCANTHION_LEFT = "endocanthion_left"
    ENDOCANTHION_RIGHT = "endocanthion_right"
    TRAGION_LEFT = "tragion_left"
    TRAGION_RIGHT = "tragion_right"
    GONION_LEFT = "gonion_left"
    GONION_RIGHT = "gonion_right"
    TRICHION = "trichion"
    VERTEX = "vertex"
    SOFT_TISSUE_B_POINT = "soft_tissue_b_point"
    SUPRATIP_BREAKPOINT = "supratip_breakpoint"
    INFRATIP_LOBULE = "infratip_lobule"
    # Nasal valve landmarks
    INTERNAL_VALVE_LEFT = "internal_valve_left"
    INTERNAL_VALVE_RIGHT = "internal_valve_right"
    EXTERNAL_VALVE_LEFT = "external_valve_left"
    EXTERNAL_VALVE_RIGHT = "external_valve_right"
    CHEILION_LEFT = "cheilion_left"
    CHEILION_RIGHT = "cheilion_right"
    ANS = "anterior_nasal_spine"
    PNS = "posterior_nasal_spine"
    A_POINT = "a_point"


# ── Data structures ───────────────────────────────────────────────

@dataclass(frozen=True)
class Vec3:
    """Immutable 3D vector."""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Vec3:
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s: float) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    def norm(self) -> float:
        return float(np.sqrt(self.x**2 + self.y**2 + self.z**2))

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


@dataclass
class Landmark:
    """A named anatomical landmark in 3D space."""
    landmark_type: LandmarkType
    position: Vec3
    confidence: float = 1.0
    source: str = ""
    is_synthetic: bool = False
    name: str = ""


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in physical (mm) coordinates."""
    origin: Vec3
    extent: Vec3

    @property
    def center(self) -> Vec3:
        return Vec3(
            self.origin.x + self.extent.x / 2,
            self.origin.y + self.extent.y / 2,
            self.origin.z + self.extent.z / 2,
        )

    @property
    def volume_mm3(self) -> float:
        return abs(self.extent.x * self.extent.y * self.extent.z)


@dataclass
class TissueProperties:
    """Material property assignment for a tissue region."""
    structure_type: StructureType
    material_model: MaterialModel
    parameters: Dict[str, float] = field(default_factory=dict)
    density_kg_m3: float = 1000.0
    is_anisotropic: bool = False
    fiber_direction: Optional[Vec3] = None
    source: str = "literature"
    confidence: float = 0.8

    def validate(self) -> List[str]:
        """Return list of validation errors, empty if valid."""
        errors: List[str] = []
        if self.density_kg_m3 <= 0:
            errors.append(f"density must be positive, got {self.density_kg_m3}")
        if self.is_anisotropic and self.fiber_direction is None:
            errors.append("anisotropic material requires fiber_direction")
        required: FrozenSet[str] = _REQUIRED_PARAMS.get(self.material_model, frozenset())
        missing = required - set(self.parameters.keys())
        if missing:
            errors.append(f"missing parameters for {self.material_model.value}: {missing}")
        return errors


# Required parameter names per material model
_REQUIRED_PARAMS: Dict[MaterialModel, FrozenSet[str]] = {
    MaterialModel.LINEAR_ELASTIC: frozenset({"E", "nu"}),
    MaterialModel.NEO_HOOKEAN: frozenset({"mu", "kappa"}),
    MaterialModel.MOONEY_RIVLIN: frozenset({"C1", "C2", "kappa"}),
    MaterialModel.OGDEN: frozenset({"mu_1", "alpha_1", "kappa"}),
    MaterialModel.FUNG: frozenset({"c", "A1", "A2", "kappa"}),
    MaterialModel.VISCOELASTIC_QLV: frozenset({"mu", "kappa", "tau_1", "g_1"}),
    MaterialModel.VISCOELASTIC_PRONY: frozenset({"E_inf", "E_1", "tau_1"}),
    MaterialModel.DRUCKER_PRAGER: frozenset({"E", "nu", "cohesion", "friction_angle"}),
    MaterialModel.J2_PLASTICITY: frozenset({"E", "nu", "sigma_y"}),
    MaterialModel.COHESIVE_ZONE: frozenset({"sigma_max", "delta_c", "G_c"}),
    MaterialModel.RIGID: frozenset(),
}


@dataclass
class MeshQualityReport:
    """Quality metrics for a volumetric mesh."""
    n_nodes: int
    n_elements: int
    element_type: MeshElementType
    min_jacobian: float
    max_aspect_ratio: float
    min_edge_length_mm: float
    max_edge_length_mm: float
    mean_quality: float
    n_inverted: int
    volume_mm3: float
    surface_area_mm2: float
    n_regions: int
    min_quality: float = 0.0
    max_quality: float = 1.0
    min_aspect_ratio: float = 1.0
    is_valid: bool = True

    def summary(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"Mesh QC [{status}]: {self.n_nodes:,} nodes, {self.n_elements:,} elements "
            f"({self.element_type.value}), "
            f"J_min={self.min_jacobian:.4f}, AR_max={self.max_aspect_ratio:.2f}, "
            f"inverted={self.n_inverted}, vol={self.volume_mm3:.1f} mm³"
        )


@dataclass
class RegistrationResult:
    """Result of aligning two coordinate systems."""
    source_modality: Modality
    target_modality: Modality
    rotation: np.ndarray     # (3,3) rotation matrix
    translation: np.ndarray  # (3,) translation vector
    scale: float = 1.0
    residual_mm: float = 0.0
    n_correspondences: int = 0
    confidence: float = 0.0
    method: str = ""
    nonrigid_field: Optional[np.ndarray] = None  # (N,3) displacement field

    @property
    def transform_4x4(self) -> np.ndarray:
        """Homogeneous 4x4 transformation matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation * self.scale
        T[:3, 3] = self.translation
        return T

    @property
    def rigid_transform(self) -> np.ndarray:
        """Alias for transform_4x4 — 4×4 homogeneous rigid transform."""
        return self.transform_4x4

    @property
    def rms_error_mm(self) -> float:
        """Alias for residual_mm — RMS registration error."""
        return self.residual_mm

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply rigid transform to (N,3) point cloud."""
        result: np.ndarray = (self.rotation @ (self.scale * points.T)).T + self.translation
        return result


@dataclass
class SegmentationMask:
    """Binary or label segmentation volume."""
    data: np.ndarray            # (D,H,W) uint8 label array
    voxel_spacing: Vec3         # mm per voxel
    origin: Vec3                # physical origin
    direction_cosines: np.ndarray  # (3,3) orientation
    structure_labels: Dict[int, StructureType] = field(default_factory=dict)
    confidence_map: Optional[np.ndarray] = None  # (D,H,W) float32 0-1

    @property
    def shape(self) -> Tuple[int, int, int]:
        s = self.data.shape
        return (int(s[0]), int(s[1]), int(s[2]))

    def extract_structure(self, structure: StructureType) -> np.ndarray:
        """Return binary mask for a single structure."""
        for label_id, st in self.structure_labels.items():
            if st == structure:
                result: np.ndarray = (self.data == label_id).astype(np.uint8)
                return result
        raise KeyError(f"Structure {structure.value} not found in segmentation")

    def volume_mm3(self, structure: StructureType) -> float:
        """Compute volume of a structure in mm³."""
        mask = self.extract_structure(structure)
        voxel_vol = abs(self.voxel_spacing.x * self.voxel_spacing.y * self.voxel_spacing.z)
        return float(np.sum(mask)) * voxel_vol


@dataclass
class SurfaceMesh:
    """Triangle surface mesh with optional per-vertex data."""
    vertices: np.ndarray        # (V,3) float64 positions
    triangles: np.ndarray       # (F,3) int64 face indices
    normals: Optional[np.ndarray] = None  # (V,3) or (F,3)
    vertex_colors: Optional[np.ndarray] = None  # (V,3) or (V,4) uint8
    texture_coords: Optional[np.ndarray] = None  # (V,2)
    vertex_labels: Optional[np.ndarray] = None   # (V,) int8 region labels
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def faces(self) -> np.ndarray:
        """Alias for triangles (common mesh convention)."""
        return self.triangles

    def compute_normals(self) -> None:
        """Compute per-vertex normals via area-weighted face normals."""
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        vertex_normals = np.zeros_like(self.vertices)
        for i in range(3):
            np.add.at(vertex_normals, self.triangles[:, i], face_normals)
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self.normals = vertex_normals / norms

    def surface_area_mm2(self) -> float:
        """Total surface area in mm²."""
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        return float(0.5 * np.sum(np.linalg.norm(cross, axis=1)))

    def centroid(self) -> Vec3:
        """Mesh centroid."""
        c = self.vertices.mean(axis=0)
        return Vec3(float(c[0]), float(c[1]), float(c[2]))

    def bounding_box(self) -> BoundingBox:
        """Axis-aligned bounding box."""
        mn = self.vertices.min(axis=0)
        mx = self.vertices.max(axis=0)
        return BoundingBox(
            origin=Vec3(float(mn[0]), float(mn[1]), float(mn[2])),
            extent=Vec3(float(mx[0] - mn[0]), float(mx[1] - mn[1]), float(mx[2] - mn[2])),
        )


@dataclass
class VolumeMesh:
    """Volumetric (tetrahedral/hexahedral) mesh for FEM."""
    nodes: np.ndarray           # (N,3) float64
    elements: np.ndarray        # (E,K) int64 connectivity
    element_type: MeshElementType
    region_ids: np.ndarray      # (E,) int32 region assignment
    region_materials: Dict[int, TissueProperties] = field(default_factory=dict)
    surface_tags: Dict[str, np.ndarray] = field(default_factory=dict)  # name → face indices
    quality_report: Optional[MeshQualityReport] = None

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])


@dataclass
class DicomMetadata:
    """Extracted DICOM metadata for a volume."""
    patient_id: str = ""
    study_date: str = ""
    modality: str = ""
    manufacturer: str = ""
    institution: str = ""
    slice_thickness_mm: float = 0.0
    pixel_spacing_mm: Tuple[float, float] = (0.0, 0.0)
    kvp: float = 0.0
    exposure_mas: float = 0.0
    rows: int = 0
    columns: int = 0
    n_slices: int = 0
    series_description: str = ""
    study_uid: str = ""
    series_uid: str = ""
    voxel_spacing_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume_shape: Tuple[int, int, int] = (0, 0, 0)
    origin_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    hu_range: Tuple[float, float] = (-1024.0, 3071.0)
    window_center: float = 40.0
    window_width: float = 400.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClinicalMeasurement:
    """A named clinical measurement (distance, angle, or ratio)."""
    name: str
    value: float
    unit: str = "mm"
    landmark_pair: Optional[Tuple[LandmarkType, LandmarkType]] = None
    method: str = "computed"
    confidence: float = 1.0


def generate_case_id() -> str:
    """Generate a unique case identifier."""
    return f"FP-{uuid.uuid4().hex[:12].upper()}"
