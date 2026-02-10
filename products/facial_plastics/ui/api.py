"""UI interaction contract and backend API — G9 interaction layer.

Maps every UI action to a well-typed backend call.  The API is consumed
by the built-in HTTP server (``server.py``) and can also be imported
directly for programmatic / notebook use.

Design principles
-----------------
- Every public method returns a plain ``dict`` (JSON-serialisable).
- No UI framework dependency — pure Python + the facial-plastics backend.
- Deterministic: same inputs → same outputs.
- Thread-safe: all state lives in the CaseBundle / library; the API is stateless.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.case_bundle import CaseBundle
from ..core.config import PlatformConfig
from ..core.provenance import Provenance
from ..core.types import (
    Landmark,
    MaterialModel,
    MeshElementType,
    Modality,
    ProcedureType,
    QualityLevel,
    SolverType,
    StructureType,
    SurfaceMesh,
    TissueProperties,
    Vec3,
    VolumeMesh,
    generate_case_id,
)
from ..data import CaseLibrary, CaseLibraryCurator, DicomIngester
from ..governance import AccessControl, AuditLog, ConsentManager, Permission, Role
from ..metrics import (
    AestheticMetrics,
    FunctionalMetrics,
    PlanOptimizer,
    SafetyMetrics,
    UncertaintyQuantifier,
)
from ..plan import PlanCompiler, SurgicalOp, SurgicalPlan
from ..plan.dsl import OpCategory, OperatorParam, ParamType
from ..plan.operators import (
    BLEPHAROPLASTY_OPERATORS,
    FACELIFT_OPERATORS,
    FILLER_OPERATORS,
    RHINOPLASTY_OPERATORS,
    BlepharoplastyPlanBuilder,
    FaceliftPlanBuilder,
    FillerPlanBuilder,
    RhinoplastyPlanBuilder,
)
from ..reports import ReportBuilder
from ..sim import SimOrchestrator
from ..twin import TwinBuilder

logger = logging.getLogger(__name__)


# ── Aggregate operator registry ───────────────────────────────────

ALL_OPERATORS: Dict[str, Any] = {
    **RHINOPLASTY_OPERATORS,
    **FACELIFT_OPERATORS,
    **BLEPHAROPLASTY_OPERATORS,
    **FILLER_OPERATORS,
}

ALL_PLAN_BUILDERS: Dict[str, Any] = {
    "rhinoplasty": RhinoplastyPlanBuilder,
    "facelift": FaceliftPlanBuilder,
    "blepharoplasty": BlepharoplastyPlanBuilder,
    "fillers": FillerPlanBuilder,
}


# ── JSON helpers ──────────────────────────────────────────────────

def _ndarray_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ndarray_to_list(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _op_to_dict(op: SurgicalOp) -> Dict[str, Any]:
    """Serialise a SurgicalOp to a plain dict."""
    return {
        "name": op.name,
        "category": op.category.value,
        "procedure": op.procedure.value,
        "params": _ndarray_to_list(op.params),
        "affected_structures": [s.value for s in op.affected_structures],
        "description": op.description,
        "param_defs": {
            k: {
                "name": v.name,
                "param_type": v.param_type.value,
                "unit": v.unit,
                "description": v.description,
                "default": v.default,
                "min_value": v.min_value,
                "max_value": v.max_value,
                "enum_values": list(v.enum_values) if v.enum_values else None,
            }
            for k, v in op.param_defs.items()
        },
    }


def _plan_to_dict(plan: SurgicalPlan) -> Dict[str, Any]:
    """Serialise a SurgicalPlan to a plain dict."""
    return {
        "name": plan.name,
        "procedure": plan.procedure.value,
        "description": plan.description,
        "n_steps": len(plan.steps),
        "steps": [_op_to_dict(op) for op in plan.all_ops()],
        "content_hash": plan.content_hash(),
    }


# ── UIApplication ─────────────────────────────────────────────────

class UIApplication:
    """Stateless backend API that the HTTP server and SPA talk to.

    Every method corresponds to a UI action; every return value is a
    JSON-serialisable dict that the frontend renders.

    The caller must supply paths:
    - ``library_root``: directory for the CaseLibrary
    - ``config``: PlatformConfig instance (or None for defaults)
    """

    def __init__(
        self,
        library_root: Path,
        *,
        config: Optional[PlatformConfig] = None,
    ) -> None:
        self._library_root = Path(library_root)
        self._config = config or PlatformConfig()
        self._library = CaseLibrary(self._library_root)
        self._audit = AuditLog(self._library_root / "audit.jsonl")
        self._access = AccessControl()

    # ── Internal helpers ──────────────────────────────────────────

    def _load_bundle(self, case_id: str) -> Optional[CaseBundle]:
        """Safely load a case bundle, returning *None* on failure."""
        try:
            return self._library.load_bundle(case_id)
        except Exception:
            return None

    @staticmethod
    def _bundle_mesh(bundle: CaseBundle) -> Optional[VolumeMesh]:
        """Attempt to retrieve the volume mesh from a bundle.

        The twin pipeline saves the mesh via
        ``bundle.save_volume_mesh("volume_mesh", mesh)``.  This helper
        tries to load it back; returns *None* if unavailable.
        """
        # Check the instance attribute (set by TwinBuilder in-memory)
        mesh = getattr(bundle, "volume_mesh", None)
        if mesh is not None:
            return mesh
        # Try persisted twin data via conventional names
        for loader_name in ("load_volume_mesh",):
            loader = getattr(bundle, loader_name, None)
            if loader is not None:
                for name in ("volume_mesh", "twin_mesh", "mesh"):
                    try:
                        return loader(name)
                    except Exception:
                        continue
        return None

    @staticmethod
    def _bundle_landmarks(bundle: CaseBundle) -> List[Landmark]:
        """Load landmark list from a bundle."""
        lm = getattr(bundle, "landmarks", None)
        if lm:
            return list(lm)
        try:
            raw = bundle.load_json("landmarks", subdir="twin")
            if isinstance(raw, list):
                return raw  # type: ignore[return-value]
        except Exception:
            pass
        return []

    @staticmethod
    def _bundle_segmentation(bundle: CaseBundle) -> Optional[Any]:
        """Load segmentation from a bundle."""
        seg = getattr(bundle, "segmentation", None)
        if seg is not None:
            return seg
        return None

    # ── G1: Case Library ──────────────────────────────────────────

    def list_cases(
        self,
        *,
        procedure: Optional[str] = None,
        quality: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List cases in the library with optional filters."""
        cases = self._library.list_cases()

        if procedure:
            proc = ProcedureType(procedure)
            cases = [c for c in cases if c.procedure_type == proc.value]

        if quality:
            ql = QualityLevel(quality)
            cases = [c for c in cases if c.quality_level == ql.value]

        total = len(cases)
        page = cases[offset: offset + limit]

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "cases": [_ndarray_to_list(asdict(c)) for c in page],
        }

    def get_case(self, case_id: str) -> Dict[str, Any]:
        """Get full case metadata."""
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}
        return {
            "case_id": bundle.case_id,
            "metadata": _ndarray_to_list(bundle.summary()),
        }

    def create_case(
        self,
        *,
        patient_age: int = 0,
        patient_sex: str = "unknown",
        procedure: str = "rhinoplasty",
        notes: str = "",
    ) -> Dict[str, Any]:
        """Create a new empty case in the library."""
        proc_type = ProcedureType(procedure)
        bundle = self._library.create_case(procedure=proc_type)
        self._audit.record(
            "case_created",
            case_id=bundle.case_id,
            metadata={
                "patient_age": patient_age,
                "patient_sex": patient_sex,
                "notes": notes,
            },
        )
        return {"case_id": bundle.case_id, "status": "created"}

    def delete_case(self, case_id: str) -> Dict[str, Any]:
        """Delete a case from the library."""
        try:
            self._library.delete_case(case_id, confirm=True)
            self._audit.record("case_deleted", case_id=case_id)
            return {"case_id": case_id, "status": "deleted"}
        except Exception:
            return {"error": f"Case '{case_id}' not found"}

    def curate_library(self) -> Dict[str, Any]:
        """Run the case library curator (QC, stats, dedup)."""
        curator = CaseLibraryCurator(self._library_root)
        report = curator.summary()
        return _ndarray_to_list(report)

    # ── G2: Twin Inspect ──────────────────────────────────────────

    def get_twin_summary(self, case_id: str) -> Dict[str, Any]:
        """Get summary of the digital twin for a case."""
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        summary: Dict[str, Any] = {"case_id": case_id}

        mesh = self._bundle_mesh(bundle)
        if mesh is not None:
            summary["mesh"] = {
                "n_nodes": mesh.n_nodes,
                "n_elements": mesh.n_elements,
                "element_type": mesh.element_type.value,
                "n_regions": len(mesh.region_materials),
                "regions": {
                    str(rid): {
                        "structure": props.structure_type.value,
                        "material": props.material_model.value,
                    }
                    for rid, props in mesh.region_materials.items()
                },
            }
        else:
            summary["mesh"] = None

        landmarks = self._bundle_landmarks(bundle)
        if landmarks:
            summary["landmarks"] = {
                lm.landmark_type.value: _ndarray_to_list(lm.position)
                for lm in landmarks
                if isinstance(lm, Landmark)
            }
        else:
            summary["landmarks"] = {}

        seg = self._bundle_segmentation(bundle)
        if seg is not None:
            summary["segmentation"] = {
                "shape": list(seg.mask.shape),
                "n_labels": int(seg.mask.max()) + 1,
            }
        else:
            summary["segmentation"] = None

        return summary

    def get_mesh_data(self, case_id: str) -> Dict[str, Any]:
        """Get mesh geometry data for 3D rendering.

        Returns node positions and element connectivity in a
        format suitable for Three.js BufferGeometry.
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        mesh = self._bundle_mesh(bundle)
        if mesh is None:
            return {"error": "No volume mesh available"}

        # Extract surface triangles for rendering
        surface_nodes, surface_tris = _extract_surface(mesh)

        return {
            "case_id": case_id,
            "positions": surface_nodes.tolist(),
            "indices": surface_tris.tolist(),
            "n_vertices": len(surface_nodes),
            "n_triangles": len(surface_tris),
            "region_ids": mesh.region_ids.tolist() if mesh.region_ids is not None else [],
        }

    def get_landmarks(self, case_id: str) -> Dict[str, Any]:
        """Get landmark positions for overlay in 3D view."""
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        landmarks = self._bundle_landmarks(bundle)
        return {
            "case_id": case_id,
            "landmarks": [
                {
                    "type": lm.landmark_type.value,
                    "position": _ndarray_to_list(lm.position),
                    "confidence": getattr(lm, "confidence", 1.0),
                }
                for lm in landmarks
                if isinstance(lm, Landmark)
            ],
        }

    # ── G3: Plan Author ──────────────────────────────────────────

    def list_operators(
        self,
        *,
        procedure: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List all available surgical operators with their parameter schemas."""
        ops = {}
        for name, factory in ALL_OPERATORS.items():
            # Create with defaults to introspect
            try:
                op = factory()
                if procedure and op.procedure.value != procedure:
                    continue
                ops[name] = _op_to_dict(op)
            except Exception as exc:
                logger.warning("Failed to introspect operator '%s': %s", name, exc)
                continue

        return {"operators": ops, "count": len(ops)}

    def list_templates(self) -> Dict[str, Any]:
        """List available plan builder templates."""
        templates: Dict[str, List[str]] = {}
        for category, builder_cls in ALL_PLAN_BUILDERS.items():
            methods = [
                m for m in dir(builder_cls)
                if not m.startswith("_") and callable(getattr(builder_cls, m))
            ]
            templates[category] = methods
        return {"templates": templates}

    def create_plan_from_template(
        self,
        *,
        category: str,
        template: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a surgical plan from a named template."""
        builder_cls = ALL_PLAN_BUILDERS.get(category)
        if builder_cls is None:
            return {"error": f"Unknown category '{category}'"}

        method = getattr(builder_cls, template, None)
        if method is None:
            return {"error": f"Unknown template '{template}' in '{category}'"}

        try:
            plan = method(**(params or {}))
            return {"plan": _plan_to_dict(plan)}
        except Exception as exc:
            return {"error": f"Template error: {exc}"}

    def create_custom_plan(
        self,
        *,
        name: str,
        procedure: str,
        steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create a custom surgical plan from a list of operator specs.

        Each step is ``{"operator": "op_name", "params": {...}}``.
        """
        proc = ProcedureType(procedure)
        plan = SurgicalPlan(name=name, procedure=proc, description=f"Custom {name}")

        for i, step in enumerate(steps):
            op_name = step.get("operator", "")
            factory = ALL_OPERATORS.get(op_name)
            if factory is None:
                return {"error": f"Step {i}: unknown operator '{op_name}'"}
            try:
                op = factory(**(step.get("params") or {}))
                plan.add_step(op)
            except Exception as exc:
                return {"error": f"Step {i} ({op_name}): {exc}"}

        return {"plan": _plan_to_dict(plan)}

    def compile_plan(
        self,
        case_id: str,
        plan_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compile a surgical plan against a case's mesh.

        ``plan_dict`` is the output of create_plan_from_template or
        create_custom_plan (the ``plan`` key).
        """
        if not plan_dict or "procedure" not in plan_dict:
            return {"error": "Invalid plan: missing 'procedure' key"}

        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        mesh = self._bundle_mesh(bundle)
        if mesh is None:
            return {"error": "No volume mesh available for compilation"}

        # Reconstruct plan from dict
        proc = ProcedureType(plan_dict["procedure"])
        plan = SurgicalPlan(
            name=plan_dict.get("name", "unnamed"),
            procedure=proc,
            description=plan_dict.get("description", ""),
        )
        for step_dict in plan_dict.get("steps", []):
            op_name = step_dict["name"]
            factory = ALL_OPERATORS.get(op_name)
            if factory is None:
                return {"error": f"Unknown operator '{op_name}'"}
            op = factory(**step_dict.get("params", {}))
            plan.add_step(op)

        compiler = PlanCompiler(mesh)
        result = compiler.compile(plan)

        self._audit.record(
            "plan_compiled",
            case_id=case_id,
            metadata={
                "plan": plan_dict.get("name", "unnamed"),
                "result_hash": result.content_hash(),
            },
        )

        return _ndarray_to_list(result.to_dict())

    # ── G4: Consult (What-if exploration) ─────────────────────────

    def run_whatif(
        self,
        case_id: str,
        plan_dict: Dict[str, Any],
        modified_params: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run a what-if scenario: modify plan params and compare.

        ``modified_params`` maps step name → param overrides.
        Returns compilation result for the modified plan.
        """
        if not plan_dict or "procedure" not in plan_dict:
            return {"error": "Invalid plan: missing 'procedure' key"}

        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": "Case not available"}

        mesh = self._bundle_mesh(bundle)
        if mesh is None:
            return {"error": "Case or mesh not available"}

        # Build modified plan
        proc = ProcedureType(plan_dict["procedure"])
        plan = SurgicalPlan(
            name=plan_dict.get("name", "unnamed") + "_whatif",
            procedure=proc,
            description="What-if scenario",
        )
        for step_dict in plan_dict.get("steps", []):
            op_name = step_dict["name"]
            factory = ALL_OPERATORS.get(op_name)
            if factory is None:
                return {"error": f"Unknown operator '{op_name}'"}

            params = dict(step_dict.get("params", {}))
            if op_name in modified_params:
                params.update(modified_params[op_name])

            op = factory(**params)
            plan.add_step(op)

        compiler = PlanCompiler(mesh)
        result = compiler.compile(plan)

        return {
            "scenario": "whatif",
            "modified_operators": list(modified_params.keys()),
            "result": _ndarray_to_list(result.to_dict()),
        }

    def parameter_sweep(
        self,
        case_id: str,
        plan_dict: Dict[str, Any],
        sweep_op: str,
        sweep_param: str,
        values: List[Any],
    ) -> Dict[str, Any]:
        """Sweep a single parameter across multiple values, returning
        compilation results for each.
        """
        if not plan_dict or "procedure" not in plan_dict:
            return {"error": "Invalid plan: missing 'procedure' key"}

        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": "Case not available"}

        mesh = self._bundle_mesh(bundle)
        if mesh is None:
            return {"error": "Case or mesh not available"}

        results = []
        for val in values:
            proc = ProcedureType(plan_dict["procedure"])
            plan = SurgicalPlan(
                name=f"sweep_{sweep_param}_{val}",
                procedure=proc,
            )
            for step_dict in plan_dict.get("steps", []):
                op_name = step_dict["name"]
                factory = ALL_OPERATORS.get(op_name)
                if factory is None:
                    continue
                params = dict(step_dict.get("params", {}))
                if op_name == sweep_op:
                    params[sweep_param] = val
                plan.add_step(factory(**params))

            compiler = PlanCompiler(mesh)
            res = compiler.compile(plan)
            results.append({
                "value": val,
                "result": _ndarray_to_list(res.to_dict()),
            })

        return {
            "sweep_op": sweep_op,
            "sweep_param": sweep_param,
            "n_points": len(results),
            "results": results,
        }

    # ── G5: Report ────────────────────────────────────────────────

    def generate_report(
        self,
        case_id: str,
        plan_dict: Dict[str, Any],
        *,
        format: str = "html",
    ) -> Dict[str, Any]:
        """Generate a surgical report for a case + plan.

        Returns the report content as a string (HTML, Markdown, or JSON).
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        builder = ReportBuilder(case_id=case_id)

        builder.add_surgical_plan(_ndarray_to_list(plan_dict))

        if format == "html":
            content = builder.build_html()
        elif format == "markdown":
            content = builder.build_markdown()
        else:
            content = json.dumps(builder.build_json(), indent=2, default=str)

        self._audit.record(
            "report_generated",
            case_id=case_id,
            metadata={"format": format},
        )

        return {"format": format, "content": content}

    # ── G6: 3D Visualization data ─────────────────────────────────

    def get_visualization_data(
        self,
        case_id: str,
        *,
        include_landmarks: bool = True,
        include_regions: bool = True,
    ) -> Dict[str, Any]:
        """Get all data needed for 3D visualization.

        Returns positions, indices, region colours, landmark markers.
        """
        mesh_data = self.get_mesh_data(case_id)
        if "error" in mesh_data:
            return mesh_data

        result: Dict[str, Any] = {
            "mesh": mesh_data,
        }

        if include_landmarks:
            result["landmarks"] = self.get_landmarks(case_id)

        if include_regions:
            bundle = self._load_bundle(case_id)
            if bundle is not None:
                mesh = self._bundle_mesh(bundle)
                if mesh is not None:
                    region_colors = _generate_region_colors(mesh)
                    result["region_colors"] = region_colors

        return result

    # ── G7: Timeline ──────────────────────────────────────────────

    def get_timeline(self, case_id: str) -> Dict[str, Any]:
        """Get the event timeline for a case (audit trail)."""
        events = self._audit.query(case_id=case_id)
        return {
            "case_id": case_id,
            "events": [_ndarray_to_list(e.to_dict()) for e in events],
            "n_events": len(events),
        }

    def get_simulation_timeline(
        self,
        case_id: str,
        n_frames: int = 20,
    ) -> Dict[str, Any]:
        """Get interpolated simulation timeline frames.

        If simulation results are stored on the case, returns
        displacement fields at evenly-spaced load steps.
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        # Try to load simulation results from bundle storage
        try:
            sim_results = bundle.load_json("simulation_results", subdir="runs")
        except Exception:
            sim_results = None

        if sim_results is None:
            return {
                "case_id": case_id,
                "frames": [],
                "message": "No simulation results stored on case",
            }

        return {
            "case_id": case_id,
            "n_frames": n_frames,
            "frames": _ndarray_to_list(sim_results),
        }

    # ── G8: Compare ───────────────────────────────────────────────

    def compare_plans(
        self,
        case_id: str,
        plan_a: Dict[str, Any],
        plan_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two plans by compiling both against the same mesh."""
        result_a = self.compile_plan(case_id, plan_a)
        result_b = self.compile_plan(case_id, plan_b)

        diff: Dict[str, Any] = {
            "case_id": case_id,
            "plan_a": {"name": plan_a.get("name"), "result": result_a},
            "plan_b": {"name": plan_b.get("name"), "result": result_b},
        }

        # Compute diff summary
        if "error" not in result_a and "error" not in result_b:
            diff["delta"] = {
                "n_bcs_diff": result_a.get("n_bcs", 0) - result_b.get("n_bcs", 0),
                "n_material_mods_diff": (
                    result_a.get("n_material_mods", 0) -
                    result_b.get("n_material_mods", 0)
                ),
                "n_mesh_mods_diff": (
                    result_a.get("n_mesh_mods", 0) -
                    result_b.get("n_mesh_mods", 0)
                ),
            }

        return diff

    def compare_cases(
        self,
        case_id_a: str,
        case_id_b: str,
    ) -> Dict[str, Any]:
        """Compare two cases (twin geometry, landmark differences)."""
        twin_a = self.get_twin_summary(case_id_a)
        twin_b = self.get_twin_summary(case_id_b)

        return {
            "case_a": twin_a,
            "case_b": twin_b,
            "mesh_diff": _compare_mesh_summaries(
                twin_a.get("mesh"), twin_b.get("mesh")
            ),
        }

    # ── G9: Interaction contract metadata ─────────────────────────

    def get_contract(self) -> Dict[str, Any]:
        """Return the complete interaction contract — all available
        API endpoints, operator schemas, and template listings.

        Used by the SPA to dynamically build the UI.
        """
        return {
            "version": "0.1.0",
            "modes": {
                "case_library": {
                    "actions": [
                        "list_cases", "get_case", "create_case",
                        "delete_case", "curate_library",
                    ],
                },
                "twin_inspect": {
                    "actions": [
                        "get_twin_summary", "get_mesh_data", "get_landmarks",
                    ],
                },
                "plan_author": {
                    "actions": [
                        "list_operators", "list_templates",
                        "create_plan_from_template", "create_custom_plan",
                        "compile_plan",
                    ],
                },
                "consult": {
                    "actions": [
                        "run_whatif", "parameter_sweep",
                    ],
                },
                "report": {
                    "actions": ["generate_report"],
                },
                "visualization": {
                    "actions": ["get_visualization_data"],
                },
                "timeline": {
                    "actions": ["get_timeline", "get_simulation_timeline"],
                },
                "compare": {
                    "actions": ["compare_plans", "compare_cases"],
                },
            },
            "operators": self.list_operators(),
            "templates": self.list_templates(),
            "procedures": [p.value for p in ProcedureType],
            "structures": [s.value for s in StructureType],
            "material_models": [m.value for m in MaterialModel],
        }


# ── Internal helpers ──────────────────────────────────────────────

def _extract_surface(mesh: VolumeMesh) -> Tuple[np.ndarray, np.ndarray]:
    """Extract surface triangles from a volume mesh for rendering.

    Returns (nodes_Nx3, triangles_Mx3).
    """
    if mesh.n_elements == 0 or mesh.n_nodes == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    # Face extraction: collect faces and find boundary faces (appear once)
    face_count: Dict[Tuple[int, ...], int] = {}
    face_list: List[Tuple[int, ...]] = []

    etype = mesh.element_type
    for eid in range(mesh.n_elements):
        elem = mesh.elements[eid]
        faces = _element_faces(elem, etype)
        for f in faces:
            key = tuple(sorted(f))
            face_count[key] = face_count.get(key, 0) + 1
            if face_count[key] == 1:
                face_list.append(f)

    # Keep only boundary faces (count == 1)
    boundary_faces = []
    for f in face_list:
        key = tuple(sorted(f))
        if face_count.get(key, 0) == 1:
            boundary_faces.append(f)

    if not boundary_faces:
        return mesh.nodes.copy(), np.zeros((0, 3), dtype=np.int64)

    triangles = np.array(boundary_faces, dtype=np.int64)
    return mesh.nodes.copy(), triangles


def _element_faces(
    elem: np.ndarray,
    etype: MeshElementType,
) -> List[Tuple[int, ...]]:
    """Return the triangular faces of an element."""
    n = len(elem)
    if etype in (MeshElementType.TET4, MeshElementType.TET10):
        if n < 4:
            return []
        a, b, c, d = int(elem[0]), int(elem[1]), int(elem[2]), int(elem[3])
        return [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]
    elif etype in (MeshElementType.HEX8, MeshElementType.HEX20):
        if n < 8:
            return []
        e = [int(elem[i]) for i in range(8)]
        return [
            (e[0], e[1], e[2]), (e[0], e[2], e[3]),  # bottom
            (e[4], e[5], e[6]), (e[4], e[6], e[7]),  # top
            (e[0], e[1], e[5]), (e[0], e[5], e[4]),  # front
            (e[2], e[3], e[7]), (e[2], e[7], e[6]),  # back
            (e[0], e[3], e[7]), (e[0], e[7], e[4]),  # left
            (e[1], e[2], e[6]), (e[1], e[6], e[5]),  # right
        ]
    elif etype in (MeshElementType.TRI3, MeshElementType.TRI6):
        if n < 3:
            return []
        return [(int(elem[0]), int(elem[1]), int(elem[2]))]
    else:
        # Generic: triangulate first face
        if n >= 3:
            return [(int(elem[0]), int(elem[1]), int(elem[2]))]
        return []


def _generate_region_colors(mesh: VolumeMesh) -> Dict[str, str]:
    """Generate distinct hex colours for each mesh region."""
    # Predefined palette for common structures
    palette = {
        "bone": "#f5f5dc",
        "cartilage": "#87ceeb",
        "skin": "#ffdbac",
        "fat": "#fff44f",
        "muscle": "#cd5c5c",
        "smas": "#dda0dd",
        "airway": "#add8e6",
        "periosteum": "#d2b48c",
        "vessel": "#ff6347",
        "nerve": "#ffd700",
        "mucosa": "#ffb6c1",
    }

    colors: Dict[str, str] = {}
    for rid, props in mesh.region_materials.items():
        struct_name = props.structure_type.value
        for prefix, color in palette.items():
            if struct_name.startswith(prefix):
                colors[str(rid)] = color
                break
        else:
            # Hash-based fallback colour
            h = hash(struct_name) % 360
            colors[str(rid)] = f"hsl({h}, 60%, 70%)"

    return colors


def _compare_mesh_summaries(
    mesh_a: Optional[Dict[str, Any]],
    mesh_b: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare two mesh summaries."""
    if mesh_a is None or mesh_b is None:
        return {"comparable": False, "reason": "One or both meshes missing"}

    return {
        "comparable": True,
        "node_diff": mesh_a.get("n_nodes", 0) - mesh_b.get("n_nodes", 0),
        "element_diff": mesh_a.get("n_elements", 0) - mesh_b.get("n_elements", 0),
        "region_diff": mesh_a.get("n_regions", 0) - mesh_b.get("n_regions", 0),
    }
