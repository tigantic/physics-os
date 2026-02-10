"""Multi-structure segmentation from volumetric imaging.

Segments CT/CBCT volumes into anatomical structures:
  - Bone (nasal, maxilla, mandible, frontal)
  - Cartilage (septal, upper lateral, lower lateral, ear graft donor)
  - Airway (nasal cavity, nasopharynx)
  - Soft tissue layers (skin, subcutaneous fat, SMAS, muscle)
  - Mucosa, periosteum
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.config import SegmentationConfig
from ..core.types import (
    BoundingBox,
    SegmentationMask,
    StructureType,
    Vec3,
)

logger = logging.getLogger(__name__)

# ── Structure label mapping ───────────────────────────────────────

LABEL_MAP: Dict[StructureType, int] = {
    StructureType.BONE_NASAL: 1,
    StructureType.BONE_MAXILLA: 2,
    StructureType.BONE_MANDIBLE: 3,
    StructureType.BONE_FRONTAL: 4,
    StructureType.CARTILAGE_SEPTUM: 10,
    StructureType.CARTILAGE_UPPER_LATERAL: 11,
    StructureType.CARTILAGE_LOWER_LATERAL: 12,
    StructureType.CARTILAGE_EAR: 13,
    StructureType.AIRWAY_NASAL: 20,
    StructureType.AIRWAY_NASOPHARYNX: 21,
    StructureType.AIRWAY_VALVE_INTERNAL: 22,
    StructureType.AIRWAY_VALVE_EXTERNAL: 23,
    StructureType.TURBINATE_INFERIOR: 24,
    StructureType.TURBINATE_MIDDLE: 25,
    StructureType.SKIN_THICK: 30,
    StructureType.SKIN_THIN: 31,
    StructureType.SKIN_SEBACEOUS: 32,
    StructureType.FAT_SUBCUTANEOUS: 33,
    StructureType.FAT_DEEP: 34,
    StructureType.FAT_BUCCAL: 35,
    StructureType.SMAS: 36,
    StructureType.MUSCLE_MIMETIC: 37,
    StructureType.PERIOSTEUM: 38,
    StructureType.MUCOSA_NASAL: 39,
    StructureType.VESSEL_NASAL: 40,
    StructureType.NERVE_NASAL: 41,
}

INV_LABEL_MAP: Dict[int, StructureType] = {v: k for k, v in LABEL_MAP.items()}


@dataclass
class SegmentationResult:
    """Complete multi-structure segmentation output."""
    labels: np.ndarray  # (D, H, W) int16
    voxel_spacing_mm: Tuple[float, float, float]
    structures_found: List[StructureType]
    volumes_mm3: Dict[StructureType, float]
    quality_scores: Dict[StructureType, float]  # per-structure confidence
    bounding_boxes: Dict[StructureType, BoundingBox]


class MultiStructureSegmenter:
    """Multi-structure segmentation pipeline.

    Method hierarchy:
      1. ML-based segmentation (when model available)
      2. Threshold + morphological operations (default)
      3. Atlas-based segmentation (when atlas available)

    The threshold method works well for bone and airway on CT.
    Cartilage and soft tissue layers require ML or atlas methods
    for production accuracy.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None) -> None:
        self._config = config or SegmentationConfig()

    def segment(
        self,
        volume_hu: np.ndarray,
        voxel_spacing_mm: Tuple[float, float, float],
    ) -> SegmentationResult:
        """Segment a CT volume into anatomical structures.

        Parameters
        ----------
        volume_hu : ndarray, shape (D, H, W), float32
            3D volume in Hounsfield Units.
        voxel_spacing_mm : tuple of 3 floats
            Voxel spacing (slice, row, col) in mm.

        Returns
        -------
        SegmentationResult with label volume and per-structure metrics.
        """
        dz, dy, dx = volume_hu.shape
        labels = np.zeros((dz, dy, dx), dtype=np.int16)
        cfg = self._config

        logger.info("Segmenting volume %s at spacing %.2fx%.2fx%.2f mm",
                     volume_hu.shape, *voxel_spacing_mm)

        # Phase 1: Bone segmentation (highest HU)
        bone_mask = volume_hu >= cfg.bone_hu_threshold
        bone_mask = self._morphological_clean(bone_mask, voxel_spacing_mm)
        labels = self._assign_bone_regions(labels, bone_mask, volume_hu)

        # Phase 2: Cartilage segmentation (intermediate HU)
        cart_low, cart_high = cfg.cartilage_hu_range
        cart_mask = (volume_hu >= cart_low) & (volume_hu < cart_high)
        cart_mask = cart_mask & ~bone_mask  # exclude bone
        cart_mask = self._morphological_clean(cart_mask, voxel_spacing_mm)
        labels = self._assign_cartilage_regions(labels, cart_mask, volume_hu)

        # Phase 3: Airway segmentation (low HU)
        airway_mask = volume_hu <= cfg.airway_hu_threshold
        airway_mask = self._morphological_clean(airway_mask, voxel_spacing_mm, min_size_factor=0.5)
        labels = self._assign_airway_regions(labels, airway_mask)

        # Phase 4: Soft tissue (everything between airway and bone)
        soft_mask = (
            (labels == 0)
            & (volume_hu > cfg.airway_hu_threshold)
            & (volume_hu < cfg.bone_hu_threshold)
        )
        labels = self._assign_soft_tissue_regions(labels, soft_mask, volume_hu)

        # Phase 5: Skin surface detection
        labels = self._detect_skin_surface(labels, volume_hu)

        # Compute per-structure metrics
        structures_found = []
        volumes_mm3 = {}
        quality_scores = {}
        bounding_boxes = {}
        voxel_vol = float(np.prod(voxel_spacing_mm))

        for struct_type, label_val in LABEL_MAP.items():
            mask = labels == label_val
            n_voxels = int(mask.sum())
            if n_voxels > 0:
                vol = n_voxels * voxel_vol
                if vol >= cfg.min_structure_volume_mm3:
                    structures_found.append(struct_type)
                    volumes_mm3[struct_type] = vol
                    quality_scores[struct_type] = self._compute_quality_score(
                        mask, struct_type
                    )
                    bounding_boxes[struct_type] = self._compute_bbox(
                        mask, voxel_spacing_mm
                    )

        logger.info("Found %d structures", len(structures_found))

        return SegmentationResult(
            labels=labels,
            voxel_spacing_mm=voxel_spacing_mm,
            structures_found=structures_found,
            volumes_mm3=volumes_mm3,
            quality_scores=quality_scores,
            bounding_boxes=bounding_boxes,
        )

    # ── Bone region assignment ────────────────────────────────

    def _assign_bone_regions(
        self,
        labels: np.ndarray,
        bone_mask: np.ndarray,
        volume_hu: np.ndarray,
    ) -> np.ndarray:
        """Sub-classify bone regions by spatial location."""
        if not bone_mask.any():
            return labels

        dz, dy, dx = labels.shape
        # Use connected components to separate bone structures
        components = self._connected_components_3d(bone_mask)
        n_comp = int(components.max())

        for comp_id in range(1, n_comp + 1):
            comp_mask = components == comp_id
            n_voxels = int(comp_mask.sum())

            if n_voxels < 10:
                continue

            # Compute centroid
            coords = np.argwhere(comp_mask)
            centroid = coords.mean(axis=0)
            cz, cy, cx = centroid / np.array([dz, dy, dx])

            # Classify by position
            if cz < 0.3:  # superior → frontal bone
                label_val = LABEL_MAP[StructureType.BONE_FRONTAL]
            elif cx < 0.55 and cx > 0.45 and cy < 0.5:  # midline anterior
                label_val = LABEL_MAP[StructureType.BONE_NASAL]
            elif cy > 0.4:  # inferior
                label_val = LABEL_MAP[StructureType.BONE_MANDIBLE]
            else:
                label_val = LABEL_MAP[StructureType.BONE_MAXILLA]

            labels[comp_mask] = label_val

        return labels

    # ── Cartilage region assignment ───────────────────────────

    def _assign_cartilage_regions(
        self,
        labels: np.ndarray,
        cart_mask: np.ndarray,
        volume_hu: np.ndarray,
    ) -> np.ndarray:
        """Sub-classify cartilage regions."""
        if not cart_mask.any():
            return labels

        dz, dy, dx = labels.shape
        components = self._connected_components_3d(cart_mask)
        n_comp = int(components.max())

        for comp_id in range(1, n_comp + 1):
            comp_mask = components == comp_id
            if int(comp_mask.sum()) < 5:
                continue

            coords = np.argwhere(comp_mask)
            centroid = coords.mean(axis=0)
            cz, cy, cx = centroid / np.array([dz, dy, dx])

            # Nasal cartilage classification by position relative to nasal bone
            if cy < 0.5 and 0.4 < cx < 0.6:
                if cz < 0.5:
                    label_val = LABEL_MAP[StructureType.CARTILAGE_UPPER_LATERAL]
                else:
                    label_val = LABEL_MAP[StructureType.CARTILAGE_LOWER_LATERAL]
            elif cy < 0.4 and 0.4 < cx < 0.6:
                label_val = LABEL_MAP[StructureType.CARTILAGE_SEPTUM]
            else:
                label_val = LABEL_MAP[StructureType.CARTILAGE_EAR]

            labels[comp_mask] = label_val

        return labels

    # ── Airway region assignment ──────────────────────────────

    def _assign_airway_regions(
        self,
        labels: np.ndarray,
        airway_mask: np.ndarray,
    ) -> np.ndarray:
        """Classify airway into nasal cavity, nasopharynx, valves."""
        if not airway_mask.any():
            return labels

        dz, dy, dx = labels.shape

        # Find the internal airway (not external air)
        # Use largest non-background connected component
        components = self._connected_components_3d(airway_mask)
        n_comp = int(components.max())

        if n_comp == 0:
            return labels

        # Score components: internal airway = not touching volume boundary
        for comp_id in range(1, n_comp + 1):
            comp_mask = components == comp_id
            coords = np.argwhere(comp_mask)

            # Check if touching boundaries (likely external air)
            on_boundary = (
                (coords[:, 0] == 0).any()
                or (coords[:, 0] == dz - 1).any()
                or (coords[:, 1] == 0).any()
                or (coords[:, 1] == dy - 1).any()
                or (coords[:, 2] == 0).any()
                or (coords[:, 2] == dx - 1).any()
            )

            if on_boundary:
                # This is external air — skip
                continue

            centroid = coords.mean(axis=0)
            cz, cy, cx = centroid / np.array([dz, dy, dx])

            if cz > 0.6:  # posterior/inferior → nasopharynx
                label_val = LABEL_MAP[StructureType.AIRWAY_NASOPHARYNX]
            else:
                label_val = LABEL_MAP[StructureType.AIRWAY_NASAL]

            labels[comp_mask] = label_val

        return labels

    # ── Soft tissue region assignment ─────────────────────────

    def _assign_soft_tissue_regions(
        self,
        labels: np.ndarray,
        soft_mask: np.ndarray,
        volume_hu: np.ndarray,
    ) -> np.ndarray:
        """Classify soft tissue by HU ranges and position."""
        if not soft_mask.any():
            return labels

        # Sub-classify by HU value
        # Fat: -100 to -50 HU
        fat_mask = soft_mask & (volume_hu >= -100) & (volume_hu <= -50)
        labels[fat_mask] = LABEL_MAP[StructureType.FAT_SUBCUTANEOUS]

        # Muscle: 10 to 80 HU
        muscle_mask = soft_mask & (volume_hu >= 10) & (volume_hu <= 80)
        labels[muscle_mask] = LABEL_MAP[StructureType.MUSCLE_MIMETIC]

        # Dense connective (SMAS-like): 80 to 150 HU
        dense_mask = soft_mask & (volume_hu > 80) & (volume_hu < 150)
        labels[dense_mask] = LABEL_MAP[StructureType.SMAS]

        # Mucosa: near airway, low-moderate HU
        mucosa_mask = soft_mask & (volume_hu >= -50) & (volume_hu < 80)
        # Check proximity to airway
        airway_labels = set()
        for st in (StructureType.AIRWAY_NASAL, StructureType.AIRWAY_NASOPHARYNX):
            if st in LABEL_MAP:
                airway_labels.add(LABEL_MAP[st])

        if airway_labels:
            airway_present = np.isin(labels, list(airway_labels))
            if airway_present.any():
                dilated = self._dilate_3d(airway_present, radius=3)
                near_airway = mucosa_mask & dilated
                labels[near_airway] = LABEL_MAP[StructureType.MUCOSA_NASAL]

        return labels

    # ── Skin surface detection ────────────────────────────────

    def _detect_skin_surface(
        self,
        labels: np.ndarray,
        volume_hu: np.ndarray,
    ) -> np.ndarray:
        """Detect the outer skin surface of the head."""
        # Skin is the outermost soft tissue layer
        # Threshold at ~ -200 HU to find the body boundary
        body_mask = volume_hu > -200

        if not body_mask.any():
            return labels

        # Find outer surface by erosion difference
        eroded = self._erode_3d(body_mask, radius=2)
        skin_shell = body_mask & ~eroded

        # Only label voxels that aren't already assigned
        unassigned_skin = skin_shell & (labels == 0)
        labels[unassigned_skin] = LABEL_MAP[StructureType.SKIN_THICK]

        return labels

    # ── Morphological operations (pure numpy) ─────────────────

    def _morphological_clean(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        min_size_factor: float = 1.0,
    ) -> np.ndarray:
        """Clean a binary mask with opening then closing."""
        cfg = self._config
        min_vol = cfg.min_structure_volume_mm3 * min_size_factor
        voxel_vol = float(np.prod(spacing))
        min_voxels = max(1, int(min_vol / voxel_vol))

        # Remove small components
        components = self._connected_components_3d(mask)
        n_comp = int(components.max())
        cleaned = np.zeros_like(mask)
        for cid in range(1, n_comp + 1):
            comp = components == cid
            if int(comp.sum()) >= min_voxels:
                cleaned |= comp

        # Fill holes if requested
        if cfg.fill_holes:
            cleaned = self._fill_holes_3d(cleaned)

        return cleaned

    @staticmethod
    def _connected_components_3d(mask: np.ndarray) -> np.ndarray:
        """6-connected component labeling in 3D (pure numpy BFS)."""
        labels = np.zeros_like(mask, dtype=np.int32)
        dz, dy, dx = mask.shape
        current_label = 0

        visited = np.zeros_like(mask, dtype=bool)
        # 6-connected neighbors
        neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        for z in range(dz):
            for y in range(dy):
                for x in range(dx):
                    if mask[z, y, x] and not visited[z, y, x]:
                        current_label += 1
                        # BFS
                        queue = [(z, y, x)]
                        visited[z, y, x] = True
                        labels[z, y, x] = current_label
                        head = 0
                        while head < len(queue):
                            cz, cy, cx = queue[head]
                            head += 1
                            for dz_, dy_, dx_ in neighbors:
                                nz, ny, nx = cz + dz_, cy + dy_, cx + dx_
                                if (
                                    0 <= nz < dz
                                    and 0 <= ny < dy
                                    and 0 <= nx < dx
                                    and mask[nz, ny, nx]
                                    and not visited[nz, ny, nx]
                                ):
                                    visited[nz, ny, nx] = True
                                    labels[nz, ny, nx] = current_label
                                    queue.append((nz, ny, nx))

        return labels

    @staticmethod
    def _dilate_3d(mask: np.ndarray, radius: int = 1) -> np.ndarray:
        """Binary dilation using box structuring element."""
        result = mask.copy()
        for _ in range(radius):
            dilated = result.copy()
            dilated[1:] |= result[:-1]
            dilated[:-1] |= result[1:]
            dilated[:, 1:] |= result[:, :-1]
            dilated[:, :-1] |= result[:, 1:]
            dilated[:, :, 1:] |= result[:, :, :-1]
            dilated[:, :, :-1] |= result[:, :, 1:]
            result = dilated
        return result

    @staticmethod
    def _erode_3d(mask: np.ndarray, radius: int = 1) -> np.ndarray:
        """Binary erosion using box structuring element."""
        result = mask.copy()
        for _ in range(radius):
            eroded = result.copy()
            eroded[1:] &= result[:-1]
            eroded[:-1] &= result[1:]
            eroded[:, 1:] &= result[:, :-1]
            eroded[:, :-1] &= result[:, 1:]
            eroded[:, :, 1:] &= result[:, :, :-1]
            eroded[:, :, :-1] &= result[:, :, 1:]
            result = eroded
        return result

    @staticmethod
    def _fill_holes_3d(mask: np.ndarray) -> np.ndarray:
        """Fill internal holes using flood-fill from boundary."""
        dz, dy, dx = mask.shape
        # Create inverted mask
        inv = ~mask
        # Flood-fill from all boundary voxels
        visited = np.zeros_like(inv, dtype=bool)
        queue = []

        # Seed from volume boundaries
        for z in range(dz):
            for y in range(dy):
                for x in [0, dx - 1]:
                    if inv[z, y, x] and not visited[z, y, x]:
                        visited[z, y, x] = True
                        queue.append((z, y, x))
            for x in range(dx):
                for y in [0, dy - 1]:
                    if inv[z, y, x] and not visited[z, y, x]:
                        visited[z, y, x] = True
                        queue.append((z, y, x))
        for y in range(dy):
            for x in range(dx):
                for z in [0, dz - 1]:
                    if inv[z, y, x] and not visited[z, y, x]:
                        visited[z, y, x] = True
                        queue.append((z, y, x))

        neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        head = 0
        while head < len(queue):
            cz, cy, cx = queue[head]
            head += 1
            for dz_, dy_, dx_ in neighbors:
                nz, ny, nx = cz + dz_, cy + dy_, cx + dx_
                if (
                    0 <= nz < dz and 0 <= ny < dy and 0 <= nx < dx
                    and inv[nz, ny, nx] and not visited[nz, ny, nx]
                ):
                    visited[nz, ny, nx] = True
                    queue.append((nz, ny, nx))

        # Holes = inverted voxels not reachable from boundary
        filled = mask.copy()
        filled[~visited & inv] = True
        return filled

    def _compute_quality_score(
        self, mask: np.ndarray, struct_type: StructureType,
    ) -> float:
        """Compute a 0–1 quality/confidence score for a segmented structure."""
        n_voxels = int(mask.sum())
        if n_voxels == 0:
            return 0.0

        # Compactness: ratio of volume to bounding box volume
        coords = np.argwhere(mask)
        bb_shape = coords.max(axis=0) - coords.min(axis=0) + 1
        bb_vol = float(np.prod(bb_shape))
        compactness = n_voxels / max(bb_vol, 1)

        # Surface smoothness: ratio of surface voxels to total
        eroded = self._erode_3d(mask, radius=1)
        surface_voxels = int((mask & ~eroded).sum())
        surface_ratio = surface_voxels / max(n_voxels, 1)

        # Combined score (higher compactness and moderate surface ratio = better)
        score = min(1.0, compactness * 1.5) * (1.0 - abs(surface_ratio - 0.3))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _compute_bbox(
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> BoundingBox:
        """Compute bounding box in physical coordinates."""
        coords = np.argwhere(mask)
        min_vox = coords.min(axis=0)
        max_vox = coords.max(axis=0)
        sz, sy, sx = spacing

        return BoundingBox(
            min_corner=Vec3(
                float(min_vox[2] * sx),
                float(min_vox[1] * sy),
                float(min_vox[0] * sz),
            ),
            max_corner=Vec3(
                float(max_vox[2] * sx),
                float(max_vox[1] * sy),
                float(max_vox[0] * sz),
            ),
        )
