"""CaseBundle — the canonical data container for every clinical case.

A CaseBundle is a filesystem-backed, content-addressed, self-describing
container that holds *all* artifacts for a single surgical planning case.

Directory layout under ``<bundle_root>/<case_id>/``:

    inputs/          Raw imports (DICOM, scans, photos, clinical notes)
    derived/         Segmentations, registrations, landmarks
    models/          Material assignments, tissue property maps
    mesh/            Volume and surface meshes
    plan/            Surgical plan DSL files
    runs/            Simulation run directories (one per plan × parameter set)
    results/         Post-processed simulation outputs
    metrics/         Quantitative metric reports
    reports/         Clinical reports (PDF, HTML)
    validation/      Post-op comparison data
    manifest.json    Machine-readable bundle manifest
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .provenance import Provenance, hash_bytes, hash_dict, hash_file
from .types import (
    ClinicalMeasurement,
    DicomMetadata,
    MaterialModel,
    MeshElementType,
    Modality,
    ProcedureType,
    QualityLevel,
    StructureType,
    SurfaceMesh,
    TissueProperties,
    VolumeMesh,
)

# ── Subdirectory names (canonical, immutable) ────────────────────

SUBDIRS: tuple[str, ...] = (
    "inputs",
    "derived",
    "models",
    "mesh",
    "plan",
    "runs",
    "results",
    "metrics",
    "reports",
    "validation",
)


# ── Manifest schema ──────────────────────────────────────────────

@dataclass
class PatientDemographics:
    """De-identified patient demographics."""
    age_years: Optional[int] = None
    sex: Optional[str] = None  # M / F / X
    ethnicity: Optional[str] = None
    skin_fitzpatrick: Optional[int] = None  # I–VI


@dataclass
class AcquisitionRecord:
    """Metadata for a single data acquisition."""
    modality: Modality
    acquisition_date: Optional[str] = None  # ISO-8601
    device: Optional[str] = None
    resolution_mm: Optional[tuple[float, ...]] = None
    n_slices: Optional[int] = None
    file_count: int = 0
    content_hash: str = ""
    notes: str = ""


@dataclass
class RunRecord:
    """Metadata for a single simulation run."""
    run_id: str = ""
    plan_hash: str = ""
    parameter_hash: str = ""
    solver: str = ""
    wall_time_s: float = 0.0
    converged: bool = False
    n_iterations: int = 0
    result_hash: str = ""
    created_utc: str = ""


@dataclass
class BundleManifest:
    """Machine-readable manifest for a CaseBundle."""
    case_id: str = ""
    bundle_version: str = "1.0.0"
    created_utc: str = ""
    updated_utc: str = ""
    procedure_type: Optional[str] = None

    # Patient
    demographics: PatientDemographics = field(default_factory=PatientDemographics)

    # Acquisitions
    acquisitions: List[Dict[str, Any]] = field(default_factory=list)

    # Pipeline progress
    segmentation_complete: bool = False
    registration_complete: bool = False
    landmarks_complete: bool = False
    mesh_complete: bool = False
    materials_complete: bool = False
    twin_complete: bool = False

    # Plans & runs
    active_plan_hash: str = ""
    runs: List[Dict[str, Any]] = field(default_factory=list)

    # Quality
    quality_level: str = QualityLevel.UNKNOWN.value
    qc_checks_passed: int = 0
    qc_checks_total: int = 0

    # Governance
    consent_hash: str = ""
    access_log_hash: str = ""
    provenance_hash: str = ""

    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── CaseBundle class ─────────────────────────────────────────────

class CaseBundle:
    """Content-addressed, self-describing container for a clinical case.

    Usage
    -----
    Create new:
        bundle = CaseBundle.create("/data/cases", procedure=ProcedureType.RHINOPLASTY)

    Load existing:
        bundle = CaseBundle.load("/data/cases/abc123")

    Add input:
        bundle.add_input(path_to_dicom_dir, modality=Modality.CT)

    Record pipeline stage:
        bundle.mark_segmentation_complete(seg_mask)
    """

    def __init__(self, root: Path, manifest: BundleManifest, provenance: Provenance) -> None:
        self._root = root
        self._manifest = manifest
        self._provenance = provenance

    # ── Properties ────────────────────────────────────────────

    @property
    def case_id(self) -> str:
        return self._manifest.case_id

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> BundleManifest:
        return self._manifest

    @property
    def procedure(self) -> Optional[ProcedureType]:
        pt = self._manifest.procedure_type
        if pt is None:
            return None
        return ProcedureType(pt)

    @property
    def provenance(self) -> Provenance:
        return self._provenance

    def subdir(self, name: str) -> Path:
        """Return the path to a canonical subdirectory."""
        if name not in SUBDIRS:
            raise ValueError(f"Unknown subdirectory: {name!r}. Must be one of {SUBDIRS}")
        return self._root / name

    # ── Factory methods ───────────────────────────────────────

    @classmethod
    def create(
        cls,
        library_root: str | Path,
        procedure: Optional[ProcedureType] = None,
        case_id: Optional[str] = None,
        demographics: Optional[PatientDemographics] = None,
    ) -> CaseBundle:
        """Create a new CaseBundle on disk."""
        library_root = Path(library_root)
        cid = case_id or uuid.uuid4().hex[:12]
        bundle_root = library_root / cid

        if bundle_root.exists():
            raise FileExistsError(f"Case bundle already exists: {bundle_root}")

        # Create directory tree
        bundle_root.mkdir(parents=True, exist_ok=False)
        for sd in SUBDIRS:
            (bundle_root / sd).mkdir()

        now_utc = datetime.now(timezone.utc).isoformat()
        manifest = BundleManifest(
            case_id=cid,
            created_utc=now_utc,
            updated_utc=now_utc,
            procedure_type=procedure.value if procedure else None,
            demographics=demographics or PatientDemographics(),
        )

        provenance = Provenance(case_id=cid)
        provenance.begin_step("bundle_create")
        provenance.record_dict("manifest", asdict(manifest), "bundle_manifest")
        provenance.end_step()

        bundle = cls(root=bundle_root, manifest=manifest, provenance=provenance)
        bundle.save()
        return bundle

    @classmethod
    def load(cls, bundle_root: str | Path) -> CaseBundle:
        """Load an existing CaseBundle from disk."""
        bundle_root = Path(bundle_root)
        manifest_path = bundle_root / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json in {bundle_root}")

        with open(manifest_path, "r") as f:
            raw = json.load(f)

        # Reconstruct manifest
        demos = PatientDemographics(**raw.pop("demographics", {}))
        manifest = BundleManifest(**{**raw, "demographics": demos})

        # Load provenance
        prov_path = bundle_root / "provenance.json"
        if prov_path.exists():
            provenance = Provenance.load(prov_path)
        else:
            provenance = Provenance(case_id=manifest.case_id)

        return cls(root=bundle_root, manifest=manifest, provenance=provenance)

    # ── Save ──────────────────────────────────────────────────

    def save(self) -> None:
        """Persist manifest and provenance to disk."""
        self._manifest.updated_utc = datetime.now(timezone.utc).isoformat()

        manifest_path = self._root / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(asdict(self._manifest), f, indent=2, default=str)

        prov_path = self._root / "provenance.json"
        self._provenance.save(prov_path)

        # Update provenance hash in manifest
        self._manifest.provenance_hash = hash_file(prov_path)

    # ── Input management ──────────────────────────────────────

    def add_input(
        self,
        source: str | Path,
        modality: Modality,
        *,
        device: Optional[str] = None,
        acquisition_date: Optional[str] = None,
        notes: str = "",
        copy: bool = True,
    ) -> Path:
        """Import a file or directory into the inputs/ subdirectory.

        Parameters
        ----------
        source : path
            File or directory to import.
        modality : Modality
            Data modality.
        copy : bool
            If True, copies data. If False, moves it.

        Returns
        -------
        Path to the imported data inside the bundle.
        """
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        self._provenance.begin_step("add_input")

        dest_name = f"{modality.value}_{int(time.time())}_{source.name}"
        dest = self.subdir("inputs") / dest_name

        if source.is_dir():
            if copy:
                shutil.copytree(source, dest)
            else:
                shutil.move(str(source), str(dest))
            file_count = sum(1 for _ in dest.rglob("*") if _.is_file())
            content_hash = hash_bytes(
                b"".join(
                    open(p, "rb").read()
                    for p in sorted(dest.rglob("*"))
                    if p.is_file()
                )
            )
        else:
            if copy:
                shutil.copy2(source, dest)
            else:
                shutil.move(str(source), str(dest))
            file_count = 1
            content_hash = hash_file(dest)

        acq = AcquisitionRecord(
            modality=modality,
            acquisition_date=acquisition_date,
            device=device,
            file_count=file_count,
            content_hash=content_hash,
            notes=notes,
        )
        self._manifest.acquisitions.append(asdict(acq))
        self._provenance.record_file(f"acquisition_{dest.stem}", dest, "acquisition")
        self._provenance.end_step()
        self.save()
        return dest

    # ── Derived artifacts ─────────────────────────────────────

    def save_array(
        self,
        name: str,
        array: np.ndarray,
        subdir: str = "derived",
        *,
        step_name: Optional[str] = None,
    ) -> Path:
        """Save a numpy array into the bundle."""
        dest = self.subdir(subdir) / f"{name}.npy"
        if step_name:
            self._provenance.begin_step(step_name)
        np.save(dest, array)
        self._provenance.record_file(name, dest, "array")
        if step_name:
            self._provenance.end_step()
        return dest

    def load_array(self, name: str, subdir: str = "derived") -> np.ndarray:
        """Load a numpy array from the bundle."""
        path = self.subdir(subdir) / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Array not found: {path}")
        arr: np.ndarray = np.load(path, allow_pickle=False)
        return arr

    def save_json(
        self,
        name: str,
        data: Dict[str, Any],
        subdir: str = "derived",
        *,
        step_name: Optional[str] = None,
    ) -> Path:
        """Save JSON data into the bundle."""
        dest = self.subdir(subdir) / f"{name}.json"
        if step_name:
            self._provenance.begin_step(step_name)
        with open(dest, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self._provenance.record_file(name, dest, "json")
        if step_name:
            self._provenance.end_step()
        return dest

    def load_json(self, name: str, subdir: str = "derived") -> Dict[str, Any]:
        """Load JSON data from the bundle."""
        path = self.subdir(subdir) / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {path}")
        with open(path) as f:
            data: Dict[str, Any] = json.load(f)
            return data

    # ── Mesh I/O ──────────────────────────────────────────────

    def save_surface_mesh(self, name: str, mesh: SurfaceMesh) -> Path:
        """Save a SurfaceMesh to the mesh/ subdirectory in NPZ format.

        Preserves all optional fields (vertex_colors, texture_coords,
        vertex_labels) alongside the required geometry data.
        """
        dest = self.subdir("mesh") / f"{name}_surface.npz"
        self._provenance.begin_step(f"save_surface_mesh_{name}")
        save_dict: Dict[str, Any] = {
            "vertices": mesh.vertices,
            "faces": mesh.triangles,
            "normals": mesh.normals if mesh.normals is not None else np.array([]),
        }
        if mesh.vertex_colors is not None:
            save_dict["vertex_colors"] = mesh.vertex_colors
        if mesh.texture_coords is not None:
            save_dict["texture_coords"] = mesh.texture_coords
        if mesh.vertex_labels is not None:
            save_dict["vertex_labels"] = mesh.vertex_labels
        np.savez_compressed(dest, **save_dict)
        # Save metadata as sidecar JSON if present
        if mesh.metadata:
            import json as _json
            meta_path = dest.with_suffix(".meta.json")
            with open(meta_path, "w") as _mf:
                _json.dump(mesh.metadata, _mf, indent=2, default=str)
        self._provenance.record_file(f"surface_mesh_{name}", dest, "surface_mesh")
        self._provenance.end_step()
        return dest

    def load_surface_mesh(self, name: str) -> SurfaceMesh:
        """Load a SurfaceMesh from the mesh/ subdirectory.

        Restores all optional fields that were saved alongside geometry.
        """
        path = self.subdir("mesh") / f"{name}_surface.npz"
        if not path.exists():
            raise FileNotFoundError(f"Surface mesh not found: {path}")
        data = np.load(path, allow_pickle=False)
        normals = data["normals"] if data["normals"].size > 0 else None
        vertex_colors = data["vertex_colors"] if "vertex_colors" in data else None
        texture_coords = data["texture_coords"] if "texture_coords" in data else None
        vertex_labels = data["vertex_labels"] if "vertex_labels" in data else None
        metadata: Dict[str, Any] = {}
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            import json as _json
            with open(meta_path) as _mf:
                metadata = _json.load(_mf)
        return SurfaceMesh(
            vertices=data["vertices"],
            triangles=data["faces"],
            normals=normals,
            vertex_colors=vertex_colors,
            texture_coords=texture_coords,
            vertex_labels=vertex_labels,
            metadata=metadata,
        )

    def save_volume_mesh(self, name: str, mesh: VolumeMesh) -> Path:
        """Save a VolumeMesh to the mesh/ subdirectory."""
        dest = self.subdir("mesh") / f"{name}_volume.npz"
        self._provenance.begin_step(f"save_volume_mesh_{name}")
        save_dict: Dict[str, Any] = {
            "nodes": mesh.nodes,
            "elements": mesh.elements,
            "region_ids": mesh.region_ids,
        }
        # Store each surface tag as a separate named array to avoid
        # object-array pickle issues with np.savez_compressed.
        if mesh.surface_tags:
            for tag_name, tag_arr in mesh.surface_tags.items():
                save_dict[f"st_{tag_name}"] = tag_arr
        np.savez_compressed(dest, **save_dict)

        # Serialize region_materials as proper JSON-safe dicts preserving
        # TissueProperties fields so they can be reconstructed on load.
        def _serialize_tissue_props(v: Any) -> Any:
            if isinstance(v, TissueProperties):
                return {
                    "structure_type": v.structure_type.value,
                    "material_model": v.material_model.value,
                    "parameters": dict(v.parameters),
                    "density_kg_m3": v.density_kg_m3,
                    "is_anisotropic": v.is_anisotropic,
                    "fiber_direction": (
                        list(v.fiber_direction) if v.fiber_direction is not None else None
                    ),
                    "source": v.source,
                    "confidence": v.confidence,
                }
            if isinstance(v, (str, int, float, bool, dict, list, type(None))):
                return v
            return str(v)

        # Save region materials and surface tags as JSON sidecar
        sidecar = dest.with_suffix(".json")
        with open(sidecar, "w") as f:
            json.dump(
                {
                    "element_type": mesh.element_type.value,
                    "region_materials": {
                        str(k): _serialize_tissue_props(v)
                        for k, v in (mesh.region_materials or {}).items()
                    },
                    "surface_tags": {
                        str(k): (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in (mesh.surface_tags or {}).items()
                    },
                },
                f,
                indent=2,
            )
        self._provenance.record_file(f"volume_mesh_{name}", dest, "volume_mesh")
        self._provenance.record_file(f"volume_mesh_{name}_sidecar", sidecar, "volume_mesh_sidecar")
        self._provenance.end_step()
        return dest

    def load_volume_mesh(self, name: str) -> VolumeMesh:
        """Load a VolumeMesh previously saved to the mesh/ subdirectory.

        Reads ``mesh/{name}_volume.npz`` (nodes, elements, region_ids) and
        the companion ``.json`` sidecar (element_type, region_materials,
        surface_tags).  Raises ``FileNotFoundError`` when the archive
        does not exist.
        """
        npz_path = self.subdir("mesh") / f"{name}_volume.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Volume mesh '{name}' not found at {npz_path}")

        data = np.load(npz_path, allow_pickle=False)
        nodes: np.ndarray = data["nodes"]
        elements: np.ndarray = data["elements"]

        # Load region_ids from npz if present; fall back to all-zeros.
        if "region_ids" in data:
            region_ids: np.ndarray = data["region_ids"]
        else:
            region_ids = np.zeros(elements.shape[0], dtype=np.int32)

        # Sidecar carries element_type, region_materials, surface_tags
        sidecar_path = npz_path.with_suffix(".json")
        element_type = MeshElementType.TET4
        region_materials: Dict[int, Any] = {}
        surface_tags_meta: Dict[str, Any] = {}

        if sidecar_path.exists():
            try:
                with open(sidecar_path, "r") as f:
                    sc = json.load(f)
                element_type = MeshElementType(sc.get("element_type", "tet4"))
                raw_rm = sc.get("region_materials", {})
                for k, v in raw_rm.items():
                    rid = int(k)
                    if isinstance(v, dict) and "structure_type" in v:
                        # Reconstruct TissueProperties from serialized dict
                        fiber_dir = v.get("fiber_direction")
                        region_materials[rid] = TissueProperties(
                            structure_type=StructureType(v["structure_type"]),
                            material_model=MaterialModel(v["material_model"]),
                            parameters=dict(v.get("parameters", {})),
                            density_kg_m3=float(v.get("density_kg_m3", 1000.0)),
                            is_anisotropic=bool(v.get("is_anisotropic", False)),
                            fiber_direction=(
                                tuple(fiber_dir) if fiber_dir is not None else None
                            ),
                            source=str(v.get("source", "literature")),
                            confidence=float(v.get("confidence", 0.8)),
                        )
                    else:
                        # Legacy: keep raw value
                        region_materials[rid] = v
                surface_tags_meta = sc.get("surface_tags", {})
            except (json.JSONDecodeError, ValueError):
                pass  # Corrupt sidecar — use defaults

        # Reconstruct surface_tags from npz (st_{name} keys) and sidecar
        surface_tags: Dict[str, np.ndarray] = {}
        for key in data.files:
            if key.startswith("st_"):
                tag_name = key[3:]
                surface_tags[tag_name] = data[key]

        # Legacy: check sidecar-listed names in npz
        if not surface_tags:
            for tag_name in surface_tags_meta:
                candidate_key = f"surface_tags_{tag_name}"
                if candidate_key in data:
                    surface_tags[tag_name] = data[candidate_key]
                elif tag_name in data and tag_name not in ("nodes", "elements"):
                    surface_tags[tag_name] = data[tag_name]

        return VolumeMesh(
            nodes=nodes,
            elements=elements,
            element_type=element_type,
            region_ids=region_ids,
            region_materials=region_materials,
            surface_tags=surface_tags,
        )

    # ── Pipeline stage markers ────────────────────────────────

    def mark_segmentation_complete(self) -> None:
        self._manifest.segmentation_complete = True
        self._manifest.qc_checks_passed += 1
        self._manifest.qc_checks_total += 1
        self.save()

    def mark_registration_complete(self) -> None:
        self._manifest.registration_complete = True
        self._manifest.qc_checks_passed += 1
        self._manifest.qc_checks_total += 1
        self.save()

    def mark_landmarks_complete(self) -> None:
        self._manifest.landmarks_complete = True
        self._manifest.qc_checks_passed += 1
        self._manifest.qc_checks_total += 1
        self.save()

    def mark_mesh_complete(self) -> None:
        self._manifest.mesh_complete = True
        self._manifest.qc_checks_passed += 1
        self._manifest.qc_checks_total += 1
        self.save()

    def mark_materials_complete(self) -> None:
        self._manifest.materials_complete = True
        self._manifest.qc_checks_passed += 1
        self._manifest.qc_checks_total += 1
        self.save()

    def mark_twin_complete(self) -> None:
        self._manifest.twin_complete = True
        self._manifest.qc_checks_passed += 1
        self._manifest.qc_checks_total += 1
        self.save()

    # ── Run management ────────────────────────────────────────

    def create_run(self, plan_hash: str, parameter_hash: str, solver: str) -> tuple[str, Path]:
        """Create a new simulation run directory.

        Returns (run_id, run_dir).
        """
        run_id = uuid.uuid4().hex[:8]
        run_dir = self.subdir("runs") / run_id
        run_dir.mkdir()
        (run_dir / "logs").mkdir()

        record = RunRecord(
            run_id=run_id,
            plan_hash=plan_hash,
            parameter_hash=parameter_hash,
            solver=solver,
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._manifest.runs.append(asdict(record))
        self.save()
        return run_id, run_dir

    def finalize_run(
        self,
        run_id: str,
        converged: bool,
        n_iterations: int,
        wall_time_s: float,
        result_hash: str,
    ) -> None:
        """Mark a simulation run as complete."""
        for run in self._manifest.runs:
            if run["run_id"] == run_id:
                run["converged"] = converged
                run["n_iterations"] = n_iterations
                run["wall_time_s"] = wall_time_s
                run["result_hash"] = result_hash
                break
        else:
            raise ValueError(f"Run not found: {run_id}")
        self.save()

    # ── Validation ────────────────────────────────────────────

    def validate_structure(self) -> List[str]:
        """Check bundle integrity. Returns list of issues."""
        issues: List[str] = []

        # Check subdirectories exist
        for sd in SUBDIRS:
            if not (self._root / sd).is_dir():
                issues.append(f"Missing subdirectory: {sd}/")

        # Check manifest exists
        if not (self._root / "manifest.json").exists():
            issues.append("Missing manifest.json")

        # Check case_id consistency
        if not self._manifest.case_id:
            issues.append("Empty case_id in manifest")

        # Check for orphaned runs
        run_ids = {r["run_id"] for r in self._manifest.runs}
        actual_runs = set()
        runs_dir = self.subdir("runs")
        if runs_dir.exists():
            actual_runs = {d.name for d in runs_dir.iterdir() if d.is_dir()}
        orphaned = actual_runs - run_ids
        if orphaned:
            issues.append(f"Orphaned run directories: {orphaned}")

        missing_runs = run_ids - actual_runs
        if missing_runs:
            issues.append(f"Missing run directories: {missing_runs}")

        return issues

    def is_valid(self) -> bool:
        """Return True if bundle passes all integrity checks."""
        return len(self.validate_structure()) == 0

    # ── Convenience ───────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of the bundle."""
        return {
            "case_id": self.case_id,
            "procedure": self._manifest.procedure_type,
            "created": self._manifest.created_utc,
            "updated": self._manifest.updated_utc,
            "n_acquisitions": len(self._manifest.acquisitions),
            "twin_complete": self._manifest.twin_complete,
            "n_runs": len(self._manifest.runs),
            "quality": self._manifest.quality_level,
            "qc": f"{self._manifest.qc_checks_passed}/{self._manifest.qc_checks_total}",
            "issues": self.validate_structure(),
        }

    def __repr__(self) -> str:
        proc = self._manifest.procedure_type or "unspecified"
        return f"CaseBundle({self.case_id!r}, procedure={proc}, root={self._root})"
