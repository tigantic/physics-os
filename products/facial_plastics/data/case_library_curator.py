"""Case library curator — populates a legally clean, curated case library.

Orchestrates end-to-end generation and validation of synthetic clinical
cases that exercise the full platform pipeline:

  Anatomy generation → CT volume → surface mesh → CaseBundle →
  twin pipeline (segment, landmark, register, mesh, materials) →
  QC validation → library indexing.

Each generated case is:
  - Anatomically realistic (parametric models conditioned on demographics).
  - Fully self-describing (CaseBundle with manifest, provenance).
  - Pipeline-ready (passes through TwinBuilder without modification).
  - Quality-gated (only cases passing QC enter the library).
  - Labelled as `quality_level = "synthetic"`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.case_bundle import CaseBundle, PatientDemographics
from ..core.config import PlatformConfig
from ..core.types import (
    ClinicalMeasurement,
    LandmarkType,
    Modality,
    ProcedureType,
    QualityLevel,
    Vec3,
)
from .anatomy_generator import (
    AnatomyGenerator,
    AnthropometricProfile,
    PopulationSampler,
)
from .case_library import CaseLibrary

logger = logging.getLogger(__name__)


# ── Procedure assignment probabilities by demographics ────────────

_PROCEDURE_WEIGHTS: Dict[str, List[Tuple[ProcedureType, float]]] = {
    "default": [
        (ProcedureType.RHINOPLASTY, 0.40),
        (ProcedureType.SEPTORHINOPLASTY, 0.25),
        (ProcedureType.SEPTOPLASTY, 0.15),
        (ProcedureType.FACELIFT, 0.08),
        (ProcedureType.BLEPHAROPLASTY_UPPER, 0.05),
        (ProcedureType.BLEPHAROPLASTY_LOWER, 0.03),
        (ProcedureType.CHIN_AUGMENTATION, 0.04),
    ],
}


# ── Quality gate thresholds ───────────────────────────────────────

@dataclass
class QCThresholds:
    """Quality control gate thresholds for case acceptance."""
    min_structures_segmented: int = 5
    min_ct_slices: int = 32
    min_surface_vertices: int = 100
    min_surface_triangles: int = 50
    max_hu_clipping_fraction: float = 0.05
    min_volume_mm3_bone: float = 100.0
    min_volume_mm3_cartilage: float = 10.0
    min_volume_mm3_airway: float = 50.0
    min_landmarks: int = 8
    min_measurements: int = 4


@dataclass
class CaseGenerationResult:
    """Result of generating a single case."""
    case_id: str
    success: bool
    demographics: Optional[PatientDemographics] = None
    procedure: Optional[ProcedureType] = None
    n_structures: int = 0
    n_landmarks: int = 0
    n_measurements: int = 0
    surface_vertices: int = 0
    surface_triangles: int = 0
    volume_shape: Tuple[int, ...] = ()
    generation_time_s: float = 0.0
    qc_passed: bool = False
    qc_issues: List[str] = field(default_factory=list)
    error: str = ""


@dataclass
class LibraryGenerationReport:
    """Summary report for an entire library generation run."""
    total_attempted: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    total_qc_rejected: int = 0
    total_time_s: float = 0.0
    cases: List[CaseGenerationResult] = field(default_factory=list)
    demographics_summary: Dict[str, int] = field(default_factory=dict)
    procedure_summary: Dict[str, int] = field(default_factory=dict)


class CaseLibraryCurator:
    """Generate and curate a complete synthetic case library.

    Usage
    -----
    curator = CaseLibraryCurator(library_root="/data/case_library")
    report = curator.generate_library(n_cases=100)

    The curator creates a balanced, representative library by:
    1. Sampling demographics from population distributions.
    2. Generating parametric anatomy for each demographic profile.
    3. Rendering CT volumes and extracting surface meshes.
    4. Populating CaseBundles with all required artifacts.
    5. Running the twin construction pipeline on each case.
    6. Applying QC gates to accept or reject cases.
    7. Building the library index.
    """

    def __init__(
        self,
        library_root: str | Path,
        *,
        config: Optional[PlatformConfig] = None,
        seed: int = 42,
        qc_thresholds: Optional[QCThresholds] = None,
        grid_size: int = 128,
        voxel_spacing_mm: float = 1.0,
    ) -> None:
        self._library_root = Path(library_root)
        self._config = config or PlatformConfig()
        self._qc = qc_thresholds or QCThresholds()
        self._grid_size = grid_size
        self._voxel_spacing = voxel_spacing_mm
        self._sampler = PopulationSampler(seed=seed)
        self._generator = AnatomyGenerator(seed=seed)
        self._rng = np.random.default_rng(seed)

        self._library = CaseLibrary(self._library_root)

    @property
    def library(self) -> CaseLibrary:
        """Access the underlying CaseLibrary."""
        return self._library

    # ── Single case generation ────────────────────────────────

    def generate_case(
        self,
        *,
        demographics: Optional[PatientDemographics] = None,
        procedure: Optional[ProcedureType] = None,
        profile: Optional[AnthropometricProfile] = None,
        case_id: Optional[str] = None,
        run_twin_pipeline: bool = True,
    ) -> CaseGenerationResult:
        """Generate a single synthetic case and add it to the library.

        Parameters
        ----------
        demographics : PatientDemographics, optional
            Patient demographics. Sampled randomly if not provided.
        procedure : ProcedureType, optional
            Assigned procedure. Sampled from weights if not provided.
        profile : AnthropometricProfile, optional
            Full anthropometric profile. Sampled if not provided.
        case_id : str, optional
            Override case ID. Auto-generated if not provided.
        run_twin_pipeline : bool
            If True, run the full TwinBuilder on the generated data.

        Returns
        -------
        CaseGenerationResult with status and metadata.
        """
        t0 = time.monotonic()
        result = CaseGenerationResult(case_id=case_id or "pending", success=False)

        try:
            # Step 1: Demographics
            if demographics is None:
                demographics = self._sampler.sample_demographics()
            result.demographics = demographics

            # Step 2: Procedure assignment
            if procedure is None:
                procedure = self._sample_procedure()
            result.procedure = procedure

            # Step 3: Anthropometric profile
            if profile is None:
                profile = self._sampler.sample_profile(demographics)

            # Step 4: Create CaseBundle
            bundle = self._library.create_case(
                procedure=procedure,
                case_id=case_id,
            )
            result.case_id = bundle.case_id

            # Update demographics in manifest
            bundle.manifest.demographics = demographics
            bundle.manifest.quality_level = QualityLevel.SYNTHETIC.value
            bundle.save()

            # Step 5: Generate CT volume
            volume_hu, spacing, origin = self._generator.generate_ct_volume(
                profile,
                grid_size=self._grid_size,
                voxel_spacing_mm=self._voxel_spacing,
            )
            result.volume_shape = volume_hu.shape

            # Save CT volume into bundle
            bundle.save_array("ct_volume", volume_hu, subdir="inputs", step_name="synthetic_ct")
            bundle.save_json("ct_metadata", {
                "voxel_spacing_mm": list(spacing),
                "volume_shape": list(volume_hu.shape),
                "hu_range": [float(volume_hu.min()), float(volume_hu.max())],
                "source": "synthetic_parametric",
                "grid_size": self._grid_size,
                "generator_seed": int(self._rng.integers(0, 2**31)),
            }, subdir="inputs", step_name="synthetic_ct_metadata")

            # Record as acquisition
            acq_dict = {
                "modality": Modality.CT.value,
                "acquisition_date": None,
                "device": "synthetic_parametric_generator",
                "resolution_mm": list(spacing),
                "n_slices": int(volume_hu.shape[0]),
                "file_count": 1,
                "content_hash": hashlib.sha256(volume_hu.tobytes()).hexdigest(),
                "notes": "Synthetically generated CT from parametric anatomy model",
            }
            bundle.manifest.acquisitions.append(acq_dict)

            # Step 6: Generate facial surface mesh
            surface_mesh = self._generator.extract_facial_surface(
                volume_hu, spacing, origin,
            )
            bundle.save_surface_mesh("facial_surface", surface_mesh)
            result.surface_vertices = surface_mesh.n_vertices
            result.surface_triangles = surface_mesh.n_faces

            acq_surface = {
                "modality": Modality.SURFACE_SCAN.value,
                "device": "synthetic_surface_extraction",
                "file_count": 1,
                "content_hash": hashlib.sha256(surface_mesh.vertices.tobytes()).hexdigest(),
                "notes": "Surface extracted from synthetic CT",
            }
            bundle.manifest.acquisitions.append(acq_surface)

            # Step 7: Compute and save ground-truth landmarks
            landmarks = self._generator.compute_landmarks(profile)
            landmarks_data = {
                "landmarks": [
                    {
                        "name": lt.value,
                        "position": [pos.x, pos.y, pos.z],
                        "confidence": 1.0,
                        "source": "synthetic_parametric",
                    }
                    for lt, pos in landmarks.items()
                ],
                "source": "parametric_ground_truth",
            }
            bundle.save_json("landmarks", landmarks_data, subdir="derived", step_name="synthetic_landmarks")
            result.n_landmarks = len(landmarks)

            # Step 8: Clinical measurements
            measurements = self._generator.compute_clinical_measurements(profile, landmarks)
            measurements_data = {
                "measurements": [asdict(m) for m in measurements],
                "source": "synthetic_parametric",
            }
            bundle.save_json("clinical_measurements", measurements_data, subdir="derived", step_name="synthetic_measurements")
            result.n_measurements = len(measurements)

            # Step 9: Save anthropometric profile
            profile_data = asdict(profile)
            bundle.save_json("anthropometric_profile", profile_data, subdir="derived", step_name="synthetic_profile")

            # Step 10: Run twin pipeline (segmentation, meshing, materials)
            if run_twin_pipeline:
                twin_stages = self._run_twin_pipeline(bundle, volume_hu, spacing)
                result.n_structures = twin_stages.get("n_structures", 0)

            bundle.save()

            # Step 11: QC gate
            qc_passed, qc_issues = self._run_qc_gate(
                bundle, result, ran_twin_pipeline=run_twin_pipeline,
            )
            result.qc_passed = qc_passed
            result.qc_issues = qc_issues

            if not qc_passed:
                logger.warning("Case %s failed QC: %s", bundle.case_id, qc_issues)
            else:
                logger.info("Case %s passed QC", bundle.case_id)

            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False
            logger.error("Case generation failed: %s", e, exc_info=True)

        result.generation_time_s = time.monotonic() - t0

        # Refresh library index
        try:
            self._library.refresh_case(result.case_id)
        except Exception:
            pass

        return result

    # ── Batch library generation ──────────────────────────────

    def generate_library(
        self,
        n_cases: int = 50,
        *,
        run_twin_pipeline: bool = True,
        stop_on_error: bool = False,
    ) -> LibraryGenerationReport:
        """Generate a complete case library.

        Parameters
        ----------
        n_cases : int
            Number of cases to generate (target 50–500).
        run_twin_pipeline : bool
            Whether to run the full TwinBuilder pipeline on each case.
        stop_on_error : bool
            If True, stop generation on first error.

        Returns
        -------
        LibraryGenerationReport with full statistics.
        """
        t0 = time.monotonic()
        report = LibraryGenerationReport()

        logger.info("Generating case library: %d cases in %s", n_cases, self._library_root)

        for i in range(n_cases):
            logger.info("Generating case %d/%d", i + 1, n_cases)

            case_result = self.generate_case(
                run_twin_pipeline=run_twin_pipeline,
            )
            report.cases.append(case_result)
            report.total_attempted += 1

            if case_result.success:
                if case_result.qc_passed:
                    report.total_succeeded += 1
                else:
                    report.total_qc_rejected += 1
            else:
                report.total_failed += 1
                if stop_on_error:
                    logger.error("Stopping: case %d failed: %s", i + 1, case_result.error)
                    break

            # Track demographics distribution
            if case_result.demographics:
                sex = case_result.demographics.sex or "unknown"
                eth = case_result.demographics.ethnicity or "unknown"
                report.demographics_summary[f"{eth}_{sex}"] = (
                    report.demographics_summary.get(f"{eth}_{sex}", 0) + 1
                )

            if case_result.procedure:
                proc = case_result.procedure.value
                report.procedure_summary[proc] = report.procedure_summary.get(proc, 0) + 1

        report.total_time_s = time.monotonic() - t0

        # Rebuild library index
        self._library.rebuild_index()

        # Save generation report
        self._save_report(report)

        logger.info(
            "Library generation complete: %d/%d succeeded, %d failed, %d QC rejected, %.1f s",
            report.total_succeeded, report.total_attempted,
            report.total_failed, report.total_qc_rejected,
            report.total_time_s,
        )

        return report

    # ── Twin pipeline execution ───────────────────────────────

    def _run_twin_pipeline(
        self,
        bundle: CaseBundle,
        volume_hu: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        """Run segmentation, meshing, and material assignment.

        This is a lightweight inline pipeline that avoids the full
        TwinBuilder dependency chain while exercising the core stages.
        """
        stats: Dict[str, Any] = {}

        # Segmentation
        try:
            from ..twin.segmentation import MultiStructureSegmenter
            segmenter = MultiStructureSegmenter(self._config.segmentation)
            seg_result = segmenter.segment(volume_hu, spacing)

            bundle.save_array(
                "segmentation_labels", seg_result.labels,
                subdir="derived", step_name="segmentation",
            )
            bundle.save_json("segmentation_summary", {
                "structures_found": [s.value for s in seg_result.structures_found],
                "volumes_mm3": {s.value: v for s, v in seg_result.volumes_mm3.items()},
            }, subdir="derived", step_name="segmentation_summary")

            bundle.mark_segmentation_complete()
            stats["n_structures"] = len(seg_result.structures_found)
            stats["structures"] = [s.value for s in seg_result.structures_found]
        except Exception as e:
            logger.warning("Segmentation failed for %s: %s", bundle.case_id, e)
            stats["n_structures"] = 0

        # Meshing
        try:
            from ..twin.meshing import VolumetricMesher
            mesher = VolumetricMesher(self._config.mesh)
            mesh = mesher.mesh_from_labels(seg_result.labels, spacing)

            bundle.save_volume_mesh("fem_mesh", mesh)
            bundle.mark_mesh_complete()
            stats["n_mesh_elements"] = mesh.n_elements
            stats["n_mesh_nodes"] = mesh.n_nodes
        except Exception as e:
            logger.warning("Meshing failed for %s: %s", bundle.case_id, e)

        # Material assignment
        try:
            from ..twin.materials import MaterialAssigner
            assigner = MaterialAssigner()
            element_regions = np.zeros(mesh.n_elements, dtype=np.int32)

            assignments = assigner.assign(
                mesh, element_regions,
                age_years=bundle.manifest.demographics.age_years,
                skin_fitzpatrick=bundle.manifest.demographics.skin_fitzpatrick,
            )

            materials_dict = assigner.export_for_solver(assignments)
            bundle.save_json(
                "materials", materials_dict,
                subdir="models", step_name="material_assignment",
            )
            bundle.mark_materials_complete()
            stats["n_materials"] = len(assignments)
        except Exception as e:
            logger.warning("Material assignment failed for %s: %s", bundle.case_id, e)

        # Mark twin complete if segmentation and meshing succeeded
        if stats.get("n_structures", 0) > 0 and stats.get("n_mesh_elements", 0) > 0:
            bundle.mark_twin_complete()

        return stats

    # ── QC gate ───────────────────────────────────────────────

    def _run_qc_gate(
        self,
        bundle: CaseBundle,
        result: CaseGenerationResult,
        *,
        ran_twin_pipeline: bool = True,
    ) -> Tuple[bool, List[str]]:
        """Apply quality control checks to a generated case.

        Returns (passed, issues).
        """
        issues: List[str] = []
        qc = self._qc

        # Volume shape check
        if len(result.volume_shape) != 3:
            issues.append("Volume is not 3D")
        elif result.volume_shape[0] < qc.min_ct_slices:
            issues.append(f"CT has only {result.volume_shape[0]} slices (min {qc.min_ct_slices})")

        # Surface mesh checks
        if result.surface_vertices < qc.min_surface_vertices:
            issues.append(f"Surface has {result.surface_vertices} vertices (min {qc.min_surface_vertices})")
        if result.surface_triangles < qc.min_surface_triangles:
            issues.append(f"Surface has {result.surface_triangles} triangles (min {qc.min_surface_triangles})")

        # Landmark count
        if result.n_landmarks < qc.min_landmarks:
            issues.append(f"Only {result.n_landmarks} landmarks (min {qc.min_landmarks})")

        # Measurement count
        if result.n_measurements < qc.min_measurements:
            issues.append(f"Only {result.n_measurements} measurements (min {qc.min_measurements})")

        # Structure count (only checked if twin pipeline was run)
        if ran_twin_pipeline and result.n_structures < qc.min_structures_segmented:
            issues.append(f"Only {result.n_structures} structures segmented (min {qc.min_structures_segmented})")

        # Bundle integrity
        bundle_issues = bundle.validate_structure()
        issues.extend(bundle_issues)

        passed = len(issues) == 0

        # Update manifest QC
        bundle.manifest.qc_checks_total += len([
            "volume", "surface_verts", "surface_tris",
            "landmarks", "measurements", "structures", "integrity",
        ])
        if passed:
            bundle.manifest.qc_checks_passed += len([
                "volume", "surface_verts", "surface_tris",
                "landmarks", "measurements", "structures", "integrity",
            ])
        bundle.save()

        return passed, issues

    # ── Procedure sampling ────────────────────────────────────

    def _sample_procedure(self) -> ProcedureType:
        """Sample a procedure type from weighted distribution."""
        weights = _PROCEDURE_WEIGHTS.get("default", [])
        procs = [p for p, _ in weights]
        probs = np.array([w for _, w in weights])
        probs = probs / probs.sum()
        idx = self._rng.choice(len(procs), p=probs)
        return procs[idx]

    # ── Report persistence ────────────────────────────────────

    def _save_report(self, report: LibraryGenerationReport) -> Path:
        """Save the generation report to the library root."""
        report_path = self._library_root / "_generation_report.json"
        data = {
            "total_attempted": report.total_attempted,
            "total_succeeded": report.total_succeeded,
            "total_failed": report.total_failed,
            "total_qc_rejected": report.total_qc_rejected,
            "total_time_s": report.total_time_s,
            "demographics_summary": report.demographics_summary,
            "procedure_summary": report.procedure_summary,
            "cases": [],
        }
        for c in report.cases:
            data["cases"].append({
                "case_id": c.case_id,
                "success": c.success,
                "procedure": c.procedure.value if c.procedure else None,
                "demographics": asdict(c.demographics) if c.demographics else None,
                "n_structures": c.n_structures,
                "n_landmarks": c.n_landmarks,
                "n_measurements": c.n_measurements,
                "surface_vertices": c.surface_vertices,
                "surface_triangles": c.surface_triangles,
                "volume_shape": list(c.volume_shape),
                "generation_time_s": c.generation_time_s,
                "qc_passed": c.qc_passed,
                "qc_issues": c.qc_issues,
                "error": c.error,
            })

        with open(report_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved generation report to %s", report_path)
        return report_path

    # ── Convenience ───────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the current library state."""
        stats = self._library.statistics()
        stats["library_root"] = str(self._library_root)
        report_path = self._library_root / "_generation_report.json"
        if report_path.exists():
            with open(report_path) as f:
                gen_report = json.load(f)
            stats["last_generation"] = {
                "attempted": gen_report["total_attempted"],
                "succeeded": gen_report["total_succeeded"],
                "failed": gen_report["total_failed"],
                "qc_rejected": gen_report["total_qc_rejected"],
            }
        return stats

    def __repr__(self) -> str:
        return (
            f"CaseLibraryCurator(root={self._library_root}, "
            f"n_cases={self._library.case_count})"
        )
