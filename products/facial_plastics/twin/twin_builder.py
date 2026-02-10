"""Twin Builder — orchestrates the full digital twin construction pipeline.

Pipeline stages:
  1. DICOM ingestion → normalized CT volume
  2. Surface scan ingestion → facial surface mesh
  3. Multi-structure segmentation
  4. Landmark detection (volume + surface)
  5. CT ↔ surface registration
  6. Volumetric meshing
  7. Material assignment
  8. Quality checks at every stage
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.case_bundle import CaseBundle
from ..core.config import PlatformConfig
from ..core.types import (
    Landmark,
    Modality,
    ProcedureType,
    SurfaceMesh,
    VolumeMesh,
)
from .landmarks import LandmarkDetector
from .materials import MaterialAssigner, MaterialAssignment
from .meshing import VolumetricMesher
from .registration import MultiModalRegistrar
from .segmentation import MultiStructureSegmenter, SegmentationResult

logger = logging.getLogger(__name__)


@dataclass
class TwinBuildReport:
    """Report from the twin construction pipeline."""
    case_id: str
    total_time_s: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    n_structures_segmented: int = 0
    n_landmarks_detected: int = 0
    registration_rms_mm: float = 0.0
    n_mesh_elements: int = 0
    n_mesh_nodes: int = 0
    n_materials_assigned: int = 0
    mesh_quality_min: float = 0.0
    mesh_quality_mean: float = 0.0
    warnings: List[str] = field(default_factory=list)


class TwinBuilder:
    """Orchestrate the complete Digital Twin construction.

    Usage
    -----
    builder = TwinBuilder(config)
    report = builder.build(bundle)

    The builder operates on a CaseBundle, reading inputs and writing
    derived artifacts, meshes, and material assignments.
    """

    def __init__(self, config: Optional[PlatformConfig] = None) -> None:
        self._config = config or PlatformConfig()
        self._segmenter = MultiStructureSegmenter(self._config.segmentation)
        self._registrar = MultiModalRegistrar()
        self._landmark_detector = LandmarkDetector()
        self._mesher = VolumetricMesher(self._config.mesh)
        self._material_assigner = MaterialAssigner()

    def build(
        self,
        bundle: CaseBundle,
        *,
        skip_registration: bool = False,
        skip_cfd_mesh: bool = False,
    ) -> TwinBuildReport:
        """Execute the full twin construction pipeline.

        Parameters
        ----------
        bundle : CaseBundle
            Case bundle containing inputs (DICOM, surface scan).
        skip_registration : bool
            If True, skip the CT↔surface registration step.
        skip_cfd_mesh : bool
            If True, don't generate a separate CFD airway mesh.

        Returns
        -------
        TwinBuildReport summarizing what was built and any issues.
        """
        t0 = time.monotonic()
        report = TwinBuildReport(case_id=bundle.case_id)

        # Stage 1: Load CT volume
        volume_hu, voxel_spacing = self._load_ct_volume(bundle, report)
        if volume_hu is None or voxel_spacing is None:
            return report

        # Stage 2: Segmentation
        seg_result = self._run_segmentation(bundle, volume_hu, voxel_spacing, report)
        if seg_result is None:
            return report

        # Stage 3: Load surface scan
        facial_surface = self._load_surface_scan(bundle, report)

        # Stage 4: Landmark detection
        landmarks = self._detect_landmarks(
            bundle, volume_hu, voxel_spacing, seg_result, facial_surface, report
        )

        # Stage 5: Registration
        if not skip_registration and facial_surface is not None:
            self._run_registration(bundle, seg_result, facial_surface, landmarks, report)

        # Stage 6: Volumetric meshing
        volume_mesh, element_regions = self._generate_mesh(
            bundle, seg_result, report
        )
        if volume_mesh is None or element_regions is None:
            return report

        # Stage 7: Material assignment
        self._assign_materials(bundle, volume_mesh, element_regions, report)

        # Stage 8: Mark twin complete
        bundle.mark_twin_complete()
        report.stages_completed.append("twin_finalized")

        report.total_time_s = time.monotonic() - t0
        logger.info(
            "Twin built for case %s in %.1f s: %d structures, %d landmarks, "
            "%d elements, %d materials",
            bundle.case_id, report.total_time_s,
            report.n_structures_segmented, report.n_landmarks_detected,
            report.n_mesh_elements, report.n_materials_assigned,
        )

        # Save report
        bundle.save_json("twin_build_report", {
            "case_id": report.case_id,
            "total_time_s": report.total_time_s,
            "stages_completed": report.stages_completed,
            "stages_failed": report.stages_failed,
            "n_structures_segmented": report.n_structures_segmented,
            "n_landmarks_detected": report.n_landmarks_detected,
            "registration_rms_mm": report.registration_rms_mm,
            "n_mesh_elements": report.n_mesh_elements,
            "n_mesh_nodes": report.n_mesh_nodes,
            "n_materials_assigned": report.n_materials_assigned,
            "mesh_quality_min": report.mesh_quality_min,
            "mesh_quality_mean": report.mesh_quality_mean,
            "warnings": report.warnings,
        }, subdir="derived", step_name="save_twin_report")

        return report

    # ── Stage implementations ─────────────────────────────────

    def _load_ct_volume(
        self,
        bundle: CaseBundle,
        report: TwinBuildReport,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
        """Load CT volume from bundle inputs."""
        try:
            volume_hu = bundle.load_array("ct_volume", subdir="inputs")
            cached_meta = bundle.load_json("ct_metadata", subdir="inputs")
            spacing = tuple(cached_meta["voxel_spacing_mm"])
            report.stages_completed.append("ct_load")
            return volume_hu, spacing
        except FileNotFoundError:
            pass

        # Try to find raw DICOM in inputs/
        inputs_dir = bundle.subdir("inputs")
        dicom_dirs = [
            d for d in inputs_dir.iterdir()
            if d.is_dir() and (d.name.startswith("CT") or d.name.startswith("ct"))
        ]

        if not dicom_dirs:
            report.stages_failed.append("ct_load")
            report.warnings.append("No CT data found in inputs/")
            return None, None

        from ..data.dicom_ingest import DicomIngester
        ingester = DicomIngester()
        try:
            volume_hu, metadata, _hashes = ingester.ingest(
                dicom_dirs[0],
                target_spacing_mm=self._config.mesh.surface_fidelity_mm,
            )
            spacing = metadata.voxel_spacing_mm

            # Save for future loads
            bundle.save_array("ct_volume", volume_hu, subdir="inputs", step_name="ct_ingest")
            bundle.save_json("ct_metadata", {
                "voxel_spacing_mm": list(spacing),
                "volume_shape": list(volume_hu.shape),
                "hu_range": [float(volume_hu.min()), float(volume_hu.max())],
            }, subdir="inputs", step_name="ct_metadata_save")

            report.stages_completed.append("ct_load")
            return volume_hu, spacing
        except Exception as e:
            report.stages_failed.append("ct_load")
            report.warnings.append(f"CT ingestion failed: {e}")
            return None, None

    def _run_segmentation(
        self,
        bundle: CaseBundle,
        volume_hu: np.ndarray,
        spacing: Tuple[float, float, float],
        report: TwinBuildReport,
    ) -> Optional[SegmentationResult]:
        """Run multi-structure segmentation."""
        try:
            result = self._segmenter.segment(volume_hu, spacing)

            # Save results
            bundle.save_array(
                "segmentation_labels", result.labels,
                subdir="derived", step_name="segmentation",
            )
            bundle.save_json("segmentation_summary", {
                "structures_found": [s.value for s in result.structures_found],
                "volumes_mm3": {s.value: v for s, v in result.volumes_mm3.items()},
                "quality_scores": {s.value: q for s, q in result.quality_scores.items()},
            }, subdir="derived", step_name="segmentation_summary")

            bundle.mark_segmentation_complete()
            report.stages_completed.append("segmentation")
            report.n_structures_segmented = len(result.structures_found)

            return result

        except Exception as e:
            report.stages_failed.append("segmentation")
            report.warnings.append(f"Segmentation failed: {e}")
            return None

    def _load_surface_scan(
        self,
        bundle: CaseBundle,
        report: TwinBuildReport,
    ) -> Optional[SurfaceMesh]:
        """Load facial surface scan from bundle."""
        try:
            return bundle.load_surface_mesh("facial_surface")
        except FileNotFoundError:
            pass

        # Try to find mesh files in inputs/
        inputs_dir = bundle.subdir("inputs")
        mesh_files = list(inputs_dir.glob("*.obj")) + list(inputs_dir.glob("*.stl")) + list(inputs_dir.glob("*.ply"))

        if not mesh_files:
            report.warnings.append("No surface scan found in inputs/")
            return None

        from ..data.surface_ingest import SurfaceIngester
        ingester = SurfaceIngester()
        try:
            mesh = ingester.ingest(mesh_files[0])
            bundle.save_surface_mesh("facial_surface", mesh)
            report.stages_completed.append("surface_load")
            return mesh
        except Exception as e:
            report.warnings.append(f"Surface ingestion failed: {e}")
            return None

    def _detect_landmarks(
        self,
        bundle: CaseBundle,
        volume_hu: np.ndarray,
        spacing: Tuple[float, float, float],
        seg_result: SegmentationResult,
        surface: Optional[SurfaceMesh],
        report: TwinBuildReport,
    ) -> List[Landmark]:
        """Detect anatomical landmarks from all available sources."""
        all_landmarks: List[Landmark] = []

        # Volume-based landmarks (bone, airway)
        try:
            vol_landmarks = self._landmark_detector.detect_from_volume(
                volume_hu, spacing, seg_result.labels
            )
            all_landmarks.extend(vol_landmarks)
        except Exception as e:
            report.warnings.append(f"Volume landmark detection failed: {e}")

        # Surface-based landmarks
        if surface is not None:
            try:
                surf_landmarks = self._landmark_detector.detect_from_surface(surface)
                all_landmarks.extend(surf_landmarks)
            except Exception as e:
                report.warnings.append(f"Surface landmark detection failed: {e}")

        # De-duplicate (prefer higher confidence)
        landmarks = self._deduplicate_landmarks(all_landmarks)

        # Save
        if landmarks:
            bundle.save_json("landmarks", {
                "landmarks": [
                    {
                        "name": lm.name,
                        "position": [lm.position.x, lm.position.y, lm.position.z],
                        "confidence": lm.confidence,
                        "source": lm.source,
                    }
                    for lm in landmarks
                ],
            }, subdir="derived", step_name="landmarks")
            bundle.mark_landmarks_complete()

        report.stages_completed.append("landmarks")
        report.n_landmarks_detected = len(landmarks)
        return landmarks

    def _run_registration(
        self,
        bundle: CaseBundle,
        seg_result: SegmentationResult,
        surface: SurfaceMesh,
        landmarks: List[Landmark],
        report: TwinBuildReport,
    ) -> None:
        """Register CT-derived surface to scan surface."""
        try:
            # Extract CT skin surface from segmentation
            from .segmentation import LABEL_MAP
            from ..core.types import StructureType

            skin_label = LABEL_MAP.get(StructureType.SKIN_THICK, 30)
            skin_mask = seg_result.labels == skin_label

            if not skin_mask.any():
                report.warnings.append("No skin surface in segmentation for registration")
                return

            # Generate isosurface from skin
            ct_surface = self._mesher._extract_isosurface(
                skin_mask.astype(np.float32),
                seg_result.voxel_spacing_mm,
                level=0.5,
            )
            if ct_surface is None:
                report.warnings.append("Failed to extract CT skin surface")
                return

            # Split landmarks by source
            ct_landmarks = [lm for lm in landmarks if lm.source == "volume_detection"]
            scan_landmarks = [lm for lm in landmarks if lm.source == "surface_detection"]

            result = self._registrar.register_ct_to_surface(
                ct_surface, surface, ct_landmarks, scan_landmarks
            )

            # Save registration
            bundle.save_array(
                "registration_transform", result.rigid_transform,
                subdir="derived", step_name="registration",
            )
            bundle.save_json("registration_summary", {
                "rms_error_mm": result.rms_error_mm,
                "n_correspondences": result.n_correspondences,
            }, subdir="derived", step_name="registration_summary")

            bundle.mark_registration_complete()
            report.stages_completed.append("registration")
            report.registration_rms_mm = result.rms_error_mm

        except Exception as e:
            report.stages_failed.append("registration")
            report.warnings.append(f"Registration failed: {e}")

    def _generate_mesh(
        self,
        bundle: CaseBundle,
        seg_result: SegmentationResult,
        report: TwinBuildReport,
    ) -> Tuple[Optional[VolumeMesh], Optional[np.ndarray]]:
        """Generate FEM volume mesh."""
        try:
            mesh = self._mesher.mesh_from_labels(
                seg_result.labels,
                seg_result.voxel_spacing_mm,
            )

            # Compute element regions
            element_regions = self._mesher._assign_regions(
                mesh.nodes, mesh.elements,
                seg_result.labels, seg_result.voxel_spacing_mm,
            )

            # Quality check
            quality = self._mesher.compute_quality(mesh)

            # Save
            bundle.save_volume_mesh("fem_mesh", mesh)
            bundle.save_array(
                "element_regions", element_regions,
                subdir="mesh", step_name="element_regions",
            )
            bundle.save_json("mesh_quality", {
                "n_elements": quality.n_elements,
                "n_nodes": quality.n_nodes,
                "min_quality": quality.min_quality,
                "max_quality": quality.max_quality,
                "mean_quality": quality.mean_quality,
                "min_aspect_ratio": quality.min_aspect_ratio,
                "max_aspect_ratio": quality.max_aspect_ratio,
                "n_inverted": quality.n_inverted,
            }, subdir="mesh", step_name="mesh_quality")

            bundle.mark_mesh_complete()
            report.stages_completed.append("meshing")
            report.n_mesh_elements = quality.n_elements
            report.n_mesh_nodes = quality.n_nodes
            report.mesh_quality_min = quality.min_quality
            report.mesh_quality_mean = quality.mean_quality

            if quality.n_inverted > 0:
                report.warnings.append(f"{quality.n_inverted} inverted elements")
            if quality.max_aspect_ratio > self._config.mesh.max_aspect_ratio:
                report.warnings.append(
                    f"Max aspect ratio {quality.max_aspect_ratio:.1f} "
                    f"exceeds limit {self._config.mesh.max_aspect_ratio}"
                )

            return mesh, element_regions

        except Exception as e:
            report.stages_failed.append("meshing")
            report.warnings.append(f"Meshing failed: {e}")
            return None, None

    def _assign_materials(
        self,
        bundle: CaseBundle,
        mesh: VolumeMesh,
        element_regions: np.ndarray,
        report: TwinBuildReport,
    ) -> Optional[List[MaterialAssignment]]:
        """Assign material properties to mesh elements."""
        try:
            # Get demographics for age/skin adjustments
            age = bundle.manifest.demographics.age_years
            fitz = bundle.manifest.demographics.skin_fitzpatrick

            assignments = self._material_assigner.assign(
                mesh, element_regions,
                age_years=age,
                skin_fitzpatrick=fitz,
            )

            # Export and save
            materials_dict = self._material_assigner.export_for_solver(assignments)
            bundle.save_json(
                "materials", materials_dict,
                subdir="models", step_name="material_assignment",
            )

            bundle.mark_materials_complete()
            report.stages_completed.append("materials")
            report.n_materials_assigned = len(assignments)

            return assignments

        except Exception as e:
            report.stages_failed.append("materials")
            report.warnings.append(f"Material assignment failed: {e}")
            return None

    # ── Utilities ─────────────────────────────────────────────

    @staticmethod
    def _deduplicate_landmarks(landmarks: List[Landmark]) -> List[Landmark]:
        """De-duplicate landmarks by name, keeping highest confidence."""
        by_name: Dict[str, Landmark] = {}
        for lm in landmarks:
            if lm.name not in by_name or lm.confidence > by_name[lm.name].confidence:
                by_name[lm.name] = lm
        return list(by_name.values())
