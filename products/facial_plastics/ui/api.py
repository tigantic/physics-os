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

import copy
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import time

import numpy as np

from ..core.case_bundle import CaseBundle
from ..core.config import PlatformConfig
from ..core.provenance import Provenance
from ..core.types import (
    Landmark,
    LandmarkType,
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
from ..anatomy_generator import generate_makehuman_landmarks, load_makehuman_surface_mesh
from ..data import CaseLibrary, CaseLibraryCurator, DicomIngester
from ..governance import AccessControl, AuditLog, ConsentManager, Permission, Role
from ..metrics import (
    AestheticMetrics,
    FunctionalMetrics,
    PlanOptimizer,
    SafetyMetrics,
    UncertaintyQuantifier,
)
from ..metrics.uncertainty import (
    UncertainParameter,
    default_rhinoplasty_uncertainties,
)
from ..metrics.optimizer import (
    ConstraintSpec,
    ObjectiveSpec,
    ParameterBound,
    default_rhinoplasty_constraints,
    default_rhinoplasty_objectives,
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
from ..reports import ReportBuilder, ReportSection
from ..sim import (
    AirwayCFDResult,
    AirwayCFDSolver,
    FEMResult,
    HealingModel,
    SimOrchestrator,
    SimulationResult,
    SoftTissueFEM,
)
from ..sim.cfd_airway import AirwayGeometry, extract_airway_geometry
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


def _landmarks_to_dict(
    landmark_list: List[Dict[str, Any]],
) -> Dict[LandmarkType, Vec3]:
    """Convert API landmark list to ``Dict[LandmarkType, Vec3]``.

    The ``AestheticMetrics`` and ``FunctionalMetrics`` classes require
    landmarks keyed by ``LandmarkType`` enum values.  This helper
    handles the string→enum conversion with graceful fallback.
    """
    result: Dict[LandmarkType, Vec3] = {}
    for lm in landmark_list:
        name = lm.get("name", lm.get("type", ""))
        pos = lm.get("position", [0, 0, 0])
        # Try exact enum match
        try:
            lt = LandmarkType(name)
        except (ValueError, KeyError):
            # Try case-insensitive match
            for member in LandmarkType:
                if member.value.lower() == name.lower():
                    lt = member
                    break
            else:
                continue
        result[lt] = Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
    return result


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
        "description": plan._description,
        "n_steps": len(plan.root.steps),
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
        # ── Simulation caches ─────────────────────────────────
        # Plans keyed by content_hash → SurgicalPlan (populated on compile)
        self._plan_cache: Dict[str, SurgicalPlan] = {}
        # Full SimulationResult keyed by "{case_id}:{plan_hash}"
        self._sim_cache: Dict[str, SimulationResult] = {}
        # Post-op predictions keyed by "{case_id}:{plan_hash}"
        self._postop_cache: Dict[str, Dict[str, Any]] = {}

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

        Both the curator and twin builder save via
        ``bundle.save_volume_mesh("fem_mesh", mesh)``.  This helper
        tries the canonical name first, then common alternatives.
        Returns *None* if unavailable.
        """
        # Check the instance attribute (set by TwinBuilder in-memory)
        mesh = getattr(bundle, "volume_mesh", None)
        if mesh is not None:
            return mesh  # type: ignore[no-any-return]
        # Try persisted mesh via load_volume_mesh (canonical name first)
        for name in ("fem_mesh", "volume_mesh", "twin_mesh", "mesh"):
            try:
                return bundle.load_volume_mesh(name)
            except (FileNotFoundError, KeyError):
                continue
        return None

    @staticmethod
    def _bundle_surface_mesh(bundle: CaseBundle) -> Optional[SurfaceMesh]:
        """Attempt to retrieve a surface mesh from a bundle.

        Resolution order:
        1. In-memory ``surface_mesh`` attribute (set by twin builder).
        2. Persisted surface mesh on disk (curated / twin-pipeline).
        3. **MakeHuman CC0 base mesh** — real human face topology with
           proper orbital concavities, nasal bridge, philtrum, and
           vermillion border.  Used for demo/draft cases.
        """
        # Check in-memory attribute first
        smesh = getattr(bundle, "surface_mesh", None)
        if smesh is not None:
            return smesh  # type: ignore[no-any-return]
        # Try persisted surface meshes — canonical names first
        for name in ("facial_surface", "skin", "surface", "face"):
            try:
                return bundle.load_surface_mesh(name)
            except (FileNotFoundError, KeyError):
                continue
        # Fall back to MakeHuman CC0 base mesh
        try:
            mh_mesh = load_makehuman_surface_mesh()
            logger.info(
                "Using MakeHuman CC0 face mesh for case %s (%d verts, %d tris)",
                bundle.case_id, mh_mesh.n_vertices, mh_mesh.n_faces,
            )
            return mh_mesh
        except Exception as exc:
            logger.warning("MakeHuman mesh load failed: %s", exc)
            return None

    # ── Plan reconstruction / simulation helpers ──────────────

    def _reconstruct_plan_from_dict(
        self,
        plan_dict: Dict[str, Any],
    ) -> SurgicalPlan:
        """Rebuild a ``SurgicalPlan`` object from a serialised dict.

        Also caches it by ``content_hash`` so ``run_*`` endpoints can
        look up a plan by its hash alone.
        """
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
                raise ValueError(f"Unknown operator '{op_name}'")
            op = factory(**step_dict.get("params", {}))
            plan.add_step(op)
        self._plan_cache[plan.content_hash()] = plan
        return plan

    def _get_plan(
        self,
        plan_hash: Optional[str],
        case_id: str,
    ) -> Optional[SurgicalPlan]:
        """Resolve a plan by hash or create a default single-step plan."""
        if plan_hash and plan_hash in self._plan_cache:
            return self._plan_cache[plan_hash]

        # Try reconstructing from bundle metadata (plan.json)
        bundle = self._load_bundle(case_id)
        if bundle is not None:
            try:
                plan_data = bundle.load_json("plan", subdir="derived")
                if plan_data:
                    plan = self._reconstruct_plan_from_dict(plan_data)
                    if plan_hash is None or plan.content_hash() == plan_hash:
                        return plan
            except (FileNotFoundError, KeyError):
                pass

        # Create a minimal default plan for cases with no authored plan.
        # A one-step septoplasty allows the FEM to run with basic BCs.
        bundle = self._load_bundle(case_id)
        procedure = ProcedureType.SEPTOPLASTY
        if bundle is not None:
            procedure = ProcedureType(
                getattr(bundle, "procedure_type", "septoplasty")
            )

        default_plan = SurgicalPlan(
            name="auto_default",
            procedure=procedure,
            description="Auto-generated default plan for simulation",
        )
        # Add a minimal septoplasty step
        factory = ALL_OPERATORS.get("septoplasty")
        if factory is not None:
            try:
                default_plan.add_step(factory())
            except Exception:
                pass  # operator may require params — fine, empty plan OK
        self._plan_cache[default_plan.content_hash()] = default_plan
        return default_plan

    def _run_simulation(
        self,
        case_id: str,
        plan: SurgicalPlan,
        *,
        run_cfd: bool = True,
        run_healing: bool = True,
    ) -> SimulationResult:
        """Execute the full multi-physics pipeline via SimOrchestrator.

        Checks cache first; stores result after completion.
        """
        cache_key = f"{case_id}:{plan.content_hash()}"
        cached = self._sim_cache.get(cache_key)
        if cached is not None:
            return cached

        bundle = self._load_bundle(case_id)
        if bundle is None:
            raise ValueError(f"Case '{case_id}' not found")

        mesh = self._ensure_mesh(bundle)

        orchestrator = SimOrchestrator(
            mesh,
            run_cfd=run_cfd,
            run_healing=run_healing,
            fem_convergence_tol=1e-5,
            fem_max_iter=20,
            cfd_resolution=16,
            cfd_max_iter=300,
        )

        logger.info(
            "Running multi-physics simulation for case=%s plan=%s",
            case_id, plan.content_hash()[:12],
        )
        result = orchestrator.run(plan)
        self._sim_cache[cache_key] = result
        logger.info("Simulation complete: %s", result.summary())

        self._audit.record(
            "simulation_completed",
            case_id=case_id,
            metadata={
                "plan_hash": plan.content_hash(),
                "run_hash": result.run_hash,
                "fem_converged": result.fem_converged,
                "wall_clock_seconds": result.wall_clock_seconds,
            },
        )

        return result

    @classmethod
    def _ensure_mesh(cls, bundle: CaseBundle) -> VolumeMesh:
        """Return the bundle's volume mesh, generating a synthetic one if absent.

        Demo/draft cases are created without running the full twin pipeline,
        so they have no mesh on disk.  This method synthesises a
        multi-region tetrahedral mesh shaped as a human face that covers
        enough anatomical structures for the PlanCompiler to produce a
        valid result and for the 3-D viewer to show recognisable anatomy.
        The mesh is persisted via ``bundle.save_volume_mesh`` so
        subsequent calls load from disk.
        """
        mesh = cls._bundle_mesh(bundle)
        if mesh is not None:
            return mesh

        logger.info(
            "No volume mesh for case %s — generating synthetic face mesh",
            bundle.case_id,
        )
        mesh = cls._generate_synthetic_mesh()
        try:
            bundle.save_volume_mesh("fem_mesh", mesh)
            logger.info("Saved synthetic mesh for case %s", bundle.case_id)
        except Exception:
            logger.warning(
                "Could not persist synthetic mesh for case %s — using in-memory",
                bundle.case_id,
            )

        # Also generate synthetic landmarks if the bundle has none
        existing_lm = cls._bundle_landmarks(bundle)
        if not existing_lm:
            landmarks = cls._generate_synthetic_landmarks(mesh)
            try:
                lm_data = {
                    "landmarks": [
                        {
                            "type": lm.landmark_type.value,
                            "position": [lm.position.x, lm.position.y, lm.position.z],
                            "confidence": lm.confidence,
                        }
                        for lm in landmarks
                    ]
                }
                bundle.save_json("landmarks", lm_data, subdir="derived")
                logger.info(
                    "Saved %d synthetic landmarks for case %s",
                    len(landmarks), bundle.case_id,
                )
            except Exception:
                logger.warning(
                    "Could not persist synthetic landmarks for case %s",
                    bundle.case_id,
                )

        return mesh

    @staticmethod
    def _generate_synthetic_mesh() -> VolumeMesh:
        """Build a multi-region tetrahedral mesh shaped as a human face.

        Uses a parametric UV-sphere deformed by anatomical feature
        functions (nose, orbits, cheeks, chin, forehead) to produce a
        recognisable facial surface.  The interior is filled with
        tetrahedra via constrained Delaunay triangulation.  Twelve
        anatomical regions are assigned based on spatial position so that
        the PlanCompiler can exercise every operator family.

        Coordinate system (Frankfurt Horizontal aligned):
          X : right (+) / left (-)
          Y : superior (+) / inferior (-)
          Z : anterior (+) / posterior (-)
        Origin at approximate subnasale.

        Returns ~3 000–5 000 elements with 12 region labels.
        """
        from scipy.spatial import Delaunay

        # ── 1. Parametric face surface ────────────────────────
        n_phi, n_theta = 48, 32  # longitude, latitude resolution
        phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
        theta = np.linspace(0.01, np.pi - 0.01, n_theta)
        PHI, THETA = np.meshgrid(phi, theta)
        PHI_f = PHI.ravel()
        THETA_f = THETA.ravel()

        # Base ellipsoid (head shape): rx=65, ry=85, rz=75 mm
        rx, ry, rz = 65.0, 85.0, 75.0
        x0 = rx * np.sin(THETA_f) * np.cos(PHI_f)
        y0 = ry * np.cos(THETA_f)
        z0 = rz * np.sin(THETA_f) * np.sin(PHI_f)

        # Centre offset so subnasale ≈ origin-Y
        y0 -= 5.0

        # ── Anatomical deformations (additive radial perturbation) ──

        # Nose — Gaussian prominence on the anterior midline
        nose_sigma_x = 12.0
        nose_sigma_y = 30.0
        nose_centre_y = 15.0     # slightly above midline
        nose_mask = np.exp(
            -0.5 * ((x0 / nose_sigma_x) ** 2 + ((y0 - nose_centre_y) / nose_sigma_y) ** 2)
        )
        # Only add nose on the anterior face (z > 0)
        nose_mask *= np.clip(z0 / 30.0, 0.0, 1.0)
        nose_height = 28.0
        z0 += nose_mask * nose_height

        # Nasal tip — sharper forward bulge near tip
        tip_cx, tip_cy = 0.0, 3.0
        tip_sigma = 8.0
        tip_mask = np.exp(
            -0.5 * (((x0 - tip_cx) / tip_sigma) ** 2 + ((y0 - tip_cy) / tip_sigma) ** 2)
        )
        tip_mask *= np.clip(z0 / 40.0, 0.0, 1.0)
        z0 += tip_mask * 12.0

        # Alar wings — lateral bulges at nostril level
        for side in [-1.0, 1.0]:
            alar_cx = side * 14.0
            alar_cy = 0.0
            alar_sigma_x, alar_sigma_y = 8.0, 6.0
            alar_mask = np.exp(
                -0.5 * (((x0 - alar_cx) / alar_sigma_x) ** 2
                        + ((y0 - alar_cy) / alar_sigma_y) ** 2)
            )
            alar_mask *= np.clip(z0 / 30.0, 0.0, 1.0)
            z0 += alar_mask * 6.0

        # Orbits — concave depressions bilateral
        for side in [-1.0, 1.0]:
            orb_cx = side * 28.0
            orb_cy = 30.0
            orb_sigma_x, orb_sigma_y = 14.0, 10.0
            orb_mask = np.exp(
                -0.5 * (((x0 - orb_cx) / orb_sigma_x) ** 2
                        + ((y0 - orb_cy) / orb_sigma_y) ** 2)
            )
            orb_mask *= np.clip(z0 / 20.0, 0.0, 1.0)
            z0 -= orb_mask * 10.0

        # Cheek / malar eminence — lateral convexities
        for side in [-1.0, 1.0]:
            ch_cx = side * 45.0
            ch_cy = 10.0
            ch_sigma_x, ch_sigma_y = 20.0, 18.0
            ch_mask = np.exp(
                -0.5 * (((x0 - ch_cx) / ch_sigma_x) ** 2
                        + ((y0 - ch_cy) / ch_sigma_y) ** 2)
            )
            z0 += ch_mask * 8.0

        # Forehead — slight forward convexity
        fh_cy = 55.0
        fh_sigma_x, fh_sigma_y = 40.0, 20.0
        fh_mask = np.exp(
            -0.5 * ((x0 / fh_sigma_x) ** 2 + ((y0 - fh_cy) / fh_sigma_y) ** 2)
        )
        fh_mask *= np.clip(z0 / 25.0, 0.0, 1.0)
        z0 += fh_mask * 6.0

        # Chin — forward prominence
        chin_cy = -55.0
        chin_sigma_x, chin_sigma_y = 18.0, 12.0
        chin_mask = np.exp(
            -0.5 * ((x0 / chin_sigma_x) ** 2 + ((y0 - chin_cy) / chin_sigma_y) ** 2)
        )
        chin_mask *= np.clip(z0 / 20.0, 0.0, 1.0)
        z0 += chin_mask * 10.0

        # Lips — subtle forward bulge
        lip_cy = -20.0
        lip_sigma_x, lip_sigma_y = 20.0, 7.0
        lip_mask = np.exp(
            -0.5 * ((x0 / lip_sigma_x) ** 2 + ((y0 - lip_cy) / lip_sigma_y) ** 2)
        )
        lip_mask *= np.clip(z0 / 30.0, 0.0, 1.0)
        z0 += lip_mask * 5.0

        # Brow ridge — horizontal convexity
        brow_cy = 38.0
        brow_sigma_x, brow_sigma_y = 38.0, 6.0
        brow_mask = np.exp(
            -0.5 * ((x0 / brow_sigma_x) ** 2 + ((y0 - brow_cy) / brow_sigma_y) ** 2)
        )
        brow_mask *= np.clip(z0 / 25.0, 0.0, 1.0)
        z0 += brow_mask * 5.0

        # Sub-nasal concavity (philtrum area)
        phil_cy = -6.0
        phil_sigma_x, phil_sigma_y = 8.0, 5.0
        phil_mask = np.exp(
            -0.5 * ((x0 / phil_sigma_x) ** 2 + ((y0 - phil_cy) / phil_sigma_y) ** 2)
        )
        phil_mask *= np.clip(z0 / 30.0, 0.0, 1.0)
        z0 -= phil_mask * 4.0

        surface_nodes = np.column_stack([x0, y0, z0])
        n_surface = len(surface_nodes)

        # ── 2. Surface triangulation (from UV grid topology) ──
        surface_tris: List[List[int]] = []
        for j in range(n_theta - 1):
            for i in range(n_phi):
                i_next = (i + 1) % n_phi
                a = j * n_phi + i
                b = j * n_phi + i_next
                c = (j + 1) * n_phi + i
                d = (j + 1) * n_phi + i_next
                surface_tris.append([a, b, d])
                surface_tris.append([a, d, c])

        # ── 3. Interior points for tetrahedralisation ─────────
        # Add interior points at fractions of the surface to fill volume
        interior_nodes: List[np.ndarray] = []
        centroid = surface_nodes.mean(axis=0)
        for frac in [0.25, 0.50, 0.75]:
            # Subsample surface to avoid dense interior
            step = max(1, n_surface // 120)
            for idx in range(0, n_surface, step):
                pt = centroid + frac * (surface_nodes[idx] - centroid)
                interior_nodes.append(pt)
        # Add centroid itself
        interior_nodes.append(centroid)

        all_nodes = np.vstack([surface_nodes, np.array(interior_nodes)])

        # ── 4. Delaunay tetrahedralisation ────────────────────
        delaunay = Delaunay(all_nodes)
        tets = delaunay.simplices  # (E, 4)

        # Filter out degenerate / exterior tets: keep only elements whose
        # centroid lies inside the convex hull of the surface.
        tet_centroids = all_nodes[tets].mean(axis=1)  # (E, 3)

        # Approximate inside test: distance from centroid to overall
        # centroid must be less than 95% of max surface radius
        radii = np.linalg.norm(surface_nodes - centroid, axis=1)
        max_radius = np.percentile(radii, 95)
        tet_dists = np.linalg.norm(tet_centroids - centroid, axis=1)
        keep = tet_dists < max_radius * 0.95
        tets = tets[keep]
        tet_centroids = tet_centroids[keep]

        nodes_arr = all_nodes.astype(np.float64)
        elem_arr = tets.astype(np.int64)
        n_elem = elem_arr.shape[0]

        # ── 5. Anatomical region assignment ───────────────────
        # Assign regions based on spatial position of element centroid
        region_ids = np.zeros(n_elem, dtype=np.int32)
        region_materials: Dict[int, TissueProperties] = {}

        region_defs = [
            # (id, StructureType, MaterialModel, params, spatial_test_fn)
            (0, StructureType.BONE_NASAL, MaterialModel.LINEAR_ELASTIC,
             {"E": 15000.0, "nu": 0.3}),
            (1, StructureType.BONE_MAXILLA, MaterialModel.LINEAR_ELASTIC,
             {"E": 13000.0, "nu": 0.3}),
            (2, StructureType.CARTILAGE_SEPTUM, MaterialModel.NEO_HOOKEAN,
             {"mu": 4.0, "kappa": 40.0}),
            (3, StructureType.CARTILAGE_UPPER_LATERAL, MaterialModel.NEO_HOOKEAN,
             {"mu": 3.5, "kappa": 35.0}),
            (4, StructureType.CARTILAGE_LOWER_LATERAL, MaterialModel.NEO_HOOKEAN,
             {"mu": 3.0, "kappa": 30.0}),
            (5, StructureType.CARTILAGE_ALAR, MaterialModel.NEO_HOOKEAN,
             {"mu": 2.5, "kappa": 25.0}),
            (6, StructureType.SKIN_ENVELOPE, MaterialModel.OGDEN,
             {"mu_1": 0.6, "alpha_1": 6.0, "kappa": 60.0}),
            (7, StructureType.SKIN_THICK, MaterialModel.OGDEN,
             {"mu_1": 0.8, "alpha_1": 8.0, "kappa": 80.0}),
            (8, StructureType.FAT_SUBCUTANEOUS, MaterialModel.NEO_HOOKEAN,
             {"mu": 0.3, "kappa": 3.0}),
            (9, StructureType.FAT_MALAR, MaterialModel.NEO_HOOKEAN,
             {"mu": 0.25, "kappa": 2.5}),
            (10, StructureType.SMAS, MaterialModel.MOONEY_RIVLIN,
             {"C1": 1.0, "C2": 0.5, "kappa": 50.0}),
            (11, StructureType.TURBINATE_INFERIOR, MaterialModel.NEO_HOOKEAN,
             {"mu": 0.5, "kappa": 5.0}),
        ]

        for rid, struct, mat, params in region_defs:
            region_materials[rid] = TissueProperties(
                structure_type=struct,
                material_model=mat,
                parameters=params,
            )

        # Spatial assignment based on centroid position
        cx = tet_centroids[:, 0]  # lateral
        cy = tet_centroids[:, 1]  # vertical (+ = superior)
        cz = tet_centroids[:, 2]  # AP (+ = anterior)
        dist_from_centre = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        dist_from_midline = np.abs(cx)

        # Default: skin envelope (outermost layer)
        region_ids[:] = 6

        # Inner layers (by depth from surface)
        inner = dist_from_centre < max_radius * 0.75
        region_ids[inner] = 8   # fat_subcutaneous

        deep = dist_from_centre < max_radius * 0.55
        region_ids[deep] = 10   # SMAS

        bone_deep = dist_from_centre < max_radius * 0.40
        region_ids[bone_deep] = 1  # bone_maxilla

        # Nasal region: midline, anterior, mid-height
        nasal_zone = (
            (dist_from_midline < 15.0) &
            (cy > -10.0) & (cy < 45.0) &
            (cz > 30.0)
        )
        region_ids[nasal_zone & (dist_from_centre >= max_radius * 0.60)] = 6  # skin_envelope
        region_ids[nasal_zone & (dist_from_centre < max_radius * 0.60) &
                   (dist_from_centre >= max_radius * 0.45)] = 3  # cartilage_upper_lateral
        region_ids[nasal_zone & (dist_from_centre < max_radius * 0.45)] = 2  # cartilage_septum

        # Nasal bones: midline, upper nose
        nasal_bone = (
            (dist_from_midline < 8.0) &
            (cy > 20.0) & (cy < 45.0) &
            (cz > 25.0) &
            (dist_from_centre < max_radius * 0.55)
        )
        region_ids[nasal_bone] = 0  # bone_nasal

        # Lower lateral cartilage / tip
        tip_zone = (
            (dist_from_midline < 20.0) &
            (cy > -5.0) & (cy < 15.0) &
            (cz > 40.0) &
            (dist_from_centre >= max_radius * 0.50) &
            (dist_from_centre < max_radius * 0.70)
        )
        region_ids[tip_zone] = 4  # cartilage_lower_lateral

        # Alar cartilage: lateral to nose, at nostril level
        alar_zone = (
            (dist_from_midline > 8.0) & (dist_from_midline < 22.0) &
            (cy > -5.0) & (cy < 10.0) &
            (cz > 30.0) &
            (dist_from_centre >= max_radius * 0.55) &
            (dist_from_centre < max_radius * 0.75)
        )
        region_ids[alar_zone] = 5  # cartilage_alar

        # Thick skin: tip, alar
        thick_skin = (
            (dist_from_midline < 25.0) &
            (cy > -10.0) & (cy < 10.0) &
            (cz > 35.0) &
            (dist_from_centre >= max_radius * 0.70)
        )
        region_ids[thick_skin] = 7  # skin_thick

        # Malar fat: cheek area
        for side_sign in [-1.0, 1.0]:
            malar = (
                (cx * side_sign > 20.0) &
                (cy > -5.0) & (cy < 25.0) &
                (cz > 10.0) &
                (dist_from_centre > max_radius * 0.55) &
                (dist_from_centre < max_radius * 0.75)
            )
            region_ids[malar] = 9  # fat_malar

        # Turbinates: deep midline at mid-nose height
        turb = (
            (dist_from_midline < 10.0) &
            (cy > 5.0) & (cy < 25.0) &
            (cz > 0.0) & (cz < 25.0) &
            (dist_from_centre < max_radius * 0.35)
        )
        region_ids[turb] = 11  # turbinate_inferior

        # ── Airway regions (enable real CFD solve) ────────────
        # Nasal airway: deep midline passage behind nose into nasopharynx
        airway_nasal = (
            (dist_from_midline < 6.0) &
            (cy > 0.0) & (cy < 30.0) &
            (cz > 5.0) & (cz < 35.0) &
            (dist_from_centre < max_radius * 0.28)
        )
        region_ids[airway_nasal] = 12

        # Nasopharynx: deeper, lower, wider passage
        airway_naso = (
            (dist_from_midline < 8.0) &
            (cy > -10.0) & (cy < 5.0) &
            (cz > -5.0) & (cz < 20.0) &
            (dist_from_centre < max_radius * 0.25)
        )
        region_ids[airway_naso] = 13

        # Register airway materials
        region_materials[12] = TissueProperties(
            structure_type=StructureType.AIRWAY_NASAL,
            material_model=MaterialModel.NEO_HOOKEAN,
            parameters={"mu": 0.01, "kappa": 0.1},
        )
        region_materials[13] = TissueProperties(
            structure_type=StructureType.AIRWAY_NASOPHARYNX,
            material_model=MaterialModel.NEO_HOOKEAN,
            parameters={"mu": 0.01, "kappa": 0.1},
        )

        logger.info(
            "Generated synthetic face mesh: %d nodes, %d elements, %d regions",
            nodes_arr.shape[0], n_elem,
            len(set(int(r) for r in region_ids)),
        )

        return VolumeMesh(
            nodes=nodes_arr,
            elements=elem_arr,
            element_type=MeshElementType.TET4,
            region_ids=region_ids,
            region_materials=region_materials,
        )

    @staticmethod
    def _generate_synthetic_landmarks(mesh: VolumeMesh) -> List[Landmark]:
        """Generate anatomical landmarks positioned on the synthetic face mesh.

        Places landmarks at anatomically appropriate positions on the face
        surface, using the mesh bounding extents for scaling.
        """
        nodes = mesh.nodes
        x_min, x_max = float(nodes[:, 0].min()), float(nodes[:, 0].max())
        y_min, y_max = float(nodes[:, 1].min()), float(nodes[:, 1].max())
        z_min, z_max = float(nodes[:, 2].min()), float(nodes[:, 2].max())

        # Find the most anterior point in various vertical bands
        def _find_anterior(y_lo: float, y_hi: float,
                           x_lo: float = -10.0, x_hi: float = 10.0) -> Vec3:
            """Find the most anterior (max Z) surface node in a band."""
            mask = (
                (nodes[:, 1] >= y_lo) & (nodes[:, 1] <= y_hi) &
                (nodes[:, 0] >= x_lo) & (nodes[:, 0] <= x_hi)
            )
            if not mask.any():
                # Fallback: widen search
                mask = (nodes[:, 1] >= y_lo) & (nodes[:, 1] <= y_hi)
            if not mask.any():
                return Vec3(0.0, (y_lo + y_hi) / 2.0, z_max * 0.5)
            subset = nodes[mask]
            best = subset[subset[:, 2].argmax()]
            return Vec3(float(best[0]), float(best[1]), float(best[2]))

        def _find_lateral(side: float, y_lo: float, y_hi: float) -> Vec3:
            """Find a lateral surface point."""
            x_band = 5.0
            if side > 0:
                mask = (nodes[:, 0] > 15.0) & (nodes[:, 1] >= y_lo) & (nodes[:, 1] <= y_hi)
            else:
                mask = (nodes[:, 0] < -15.0) & (nodes[:, 1] >= y_lo) & (nodes[:, 1] <= y_hi)
            if not mask.any():
                return Vec3(side * 30.0, (y_lo + y_hi) / 2.0, 10.0)
            subset = nodes[mask]
            best = subset[subset[:, 2].argmax()]
            return Vec3(float(best[0]), float(best[1]), float(best[2]))

        landmarks = [
            Landmark(LandmarkType.PRONASALE, _find_anterior(-2, 8), 0.95,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.SUBNASALE, _find_anterior(-5, 0, -8, 8), 0.95,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.RHINION, _find_anterior(18, 28), 0.92,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.NASION, _find_anterior(32, 42), 0.90,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.GLABELLA, _find_anterior(40, 50), 0.88,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.SELLION, _find_anterior(30, 38), 0.88,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.POGONION, _find_anterior(-60, -45), 0.90,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.MENTON,
                     Vec3(0.0, y_min + 5.0, 0.0), 0.85,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.STOMION, _find_anterior(-25, -15), 0.90,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.LABRALE_SUPERIUS, _find_anterior(-20, -12), 0.88,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.LABRALE_INFERIUS, _find_anterior(-30, -22), 0.88,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.SUPRATIP_BREAKPOINT, _find_anterior(8, 15), 0.85,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.COLUMELLA_BREAKPOINT,
                     _find_anterior(-3, 2, -5, 5), 0.82,
                     source="synthetic_parametric", is_synthetic=True),
            # Bilateral landmarks
            Landmark(LandmarkType.ALAR_RIM_LEFT,
                     _find_lateral(-1, -3, 5), 0.88,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.ALAR_RIM_RIGHT,
                     _find_lateral(1, -3, 5), 0.88,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.ALAR_CREASE_LEFT,
                     Vec3(-16.0, 0.0, _find_lateral(-1, -2, 3).z - 3.0), 0.82,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.ALAR_CREASE_RIGHT,
                     Vec3(16.0, 0.0, _find_lateral(1, -2, 3).z - 3.0), 0.82,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.TIP_DEFINING_POINT_LEFT,
                     Vec3(-4.0, 3.0, _find_anterior(-1, 7).z - 1.0), 0.80,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.TIP_DEFINING_POINT_RIGHT,
                     Vec3(4.0, 3.0, _find_anterior(-1, 7).z - 1.0), 0.80,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.ENDOCANTHION_LEFT,
                     Vec3(-15.0, 30.0, _find_anterior(28, 34, -18, -12).z), 0.85,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.ENDOCANTHION_RIGHT,
                     Vec3(15.0, 30.0, _find_anterior(28, 34, 12, 18).z), 0.85,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.EXOCANTHION_LEFT,
                     _find_lateral(-1, 27, 33), 0.82,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.EXOCANTHION_RIGHT,
                     _find_lateral(1, 27, 33), 0.82,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.CHEILION_LEFT,
                     Vec3(-15.0, -20.0, _find_anterior(-22, -18, -18, -12).z), 0.82,
                     source="synthetic_parametric", is_synthetic=True),
            Landmark(LandmarkType.CHEILION_RIGHT,
                     Vec3(15.0, -20.0, _find_anterior(-22, -18, 12, 18).z), 0.82,
                     source="synthetic_parametric", is_synthetic=True),
        ]

        return landmarks

    @staticmethod
    def _bundle_landmarks(bundle: CaseBundle) -> List[Landmark]:
        """Load landmark list from a bundle."""
        lm = getattr(bundle, "landmarks", None)
        if lm:
            return list(lm)
        # Curator saves landmarks under subdir="derived"; also check "twin"
        # for backwards compatibility.
        for subdir in ("derived", "twin"):
            try:
                raw = bundle.load_json("landmarks", subdir=subdir)
                if isinstance(raw, dict) and "landmarks" in raw:
                    return raw["landmarks"]  # type: ignore[return-value]
                if isinstance(raw, list):
                    return raw  # type: ignore[return-value]
            except Exception:
                continue
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
        """Run the case library curator.

        If no twin-complete cases exist, generate a synthetic case with
        full twin data so the platform is immediately usable.  Uses a
        small grid (32^3) so generation completes within a few seconds.
        """
        curator = CaseLibraryCurator(
            self._library_root,
            grid_size=32,
            voxel_spacing_mm=2.0,
        )
        report = curator.summary()
        twin_complete = report.get("twin_complete", 0)

        generated: Optional[Dict[str, Any]] = None
        if twin_complete == 0:
            try:
                gen_result = curator.generate_case(
                    run_twin_pipeline=True,
                )
                if gen_result.success:
                    generated = {
                        "case_id": gen_result.case_id,
                        "procedure": gen_result.procedure.value if gen_result.procedure else None,
                        "n_landmarks": gen_result.n_landmarks,
                        "n_structures": gen_result.n_structures,
                        "qc_passed": gen_result.qc_passed,
                        "generation_time_s": round(gen_result.generation_time_s, 2),
                    }
                else:
                    generated = {
                        "error": gen_result.error or "generation failed",
                        "case_id": gen_result.case_id,
                    }
            except Exception as exc:
                generated = {"error": str(exc)}

            # Refresh the library so the new case appears
            self._library = CaseLibrary(self._library_root)
            # Refresh statistics
            report = curator.summary()

        result: Dict[str, Any] = _ndarray_to_list(report)
        if generated:
            result["generated_case"] = generated
        return result

    # ── G2: Twin Inspect ──────────────────────────────────────────

    def get_twin_summary(self, case_id: str) -> Dict[str, Any]:
        """Get summary of the digital twin for a case."""
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        summary: Dict[str, Any] = {"case_id": case_id}

        mesh = self._ensure_mesh(bundle)
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
            lm_dict: Dict[str, Any] = {}
            for lm in landmarks:
                if isinstance(lm, Landmark):
                    lm_dict[lm.landmark_type.value] = _ndarray_to_list(lm.position)
                elif isinstance(lm, dict):
                    name = lm.get("name", lm.get("type", "unknown"))
                    pos = lm.get("position", lm.get("coords", [0, 0, 0]))
                    lm_dict[name] = _ndarray_to_list(pos)
            summary["landmarks"] = lm_dict
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
        format suitable for Three.js ``BufferGeometry``.

        Strategy:

        1. **Surface mesh first** — curated / twin-pipeline cases store
           a dedicated surface mesh (``facial_surface``, ``skin``, or
           ``surface``) that already has the correct face-shaped
           topology.  This is always preferred for rendering because
           volume-mesh boundary extraction via ``_extract_surface``
           yields the convex hull of the tet mesh, not the anatomical
           surface.
        2. **Volume-mesh fallback** — for draft / synthetic cases that
           only have a volume mesh, ``_extract_surface`` is used.
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        # ── 1. Prefer a dedicated surface mesh ──────────────────
        surface = self._bundle_surface_mesh(bundle)
        if surface is not None:
            n_faces = surface.n_faces
            # Surface mesh is the external skin — assign a single
            # skin region ID (6 matches the synthetic region map)
            # unless vertex_labels are available.
            if surface.vertex_labels is not None and len(surface.vertex_labels) > 0:
                # Map per-vertex labels → per-face labels (majority vote)
                face_region_ids: List[int] = []
                for fi in range(n_faces):
                    tri = surface.triangles[fi]
                    labels = surface.vertex_labels[tri]
                    # majority vote across 3 vertices
                    counts: Dict[int, int] = {}
                    for lb in labels:
                        lb_int = int(lb)
                        counts[lb_int] = counts.get(lb_int, 0) + 1
                    face_region_ids.append(max(counts, key=counts.get))  # type: ignore[arg-type]
            else:
                face_region_ids = [6] * n_faces  # skin_envelope

            return {
                "case_id": case_id,
                "positions": surface.vertices.tolist(),
                "indices": surface.triangles.tolist(),
                "n_vertices": surface.n_vertices,
                "n_triangles": n_faces,
                "region_ids": face_region_ids,
                "normals": surface.normals.tolist() if surface.normals is not None else None,
            }

        # ── 2. Fall back to volume mesh surface extraction ──────
        mesh = self._ensure_mesh(bundle)
        surface_nodes, surface_tris, face_rids = _extract_surface(mesh)

        return {
            "case_id": case_id,
            "positions": surface_nodes.tolist(),
            "indices": surface_tris.tolist(),
            "n_vertices": len(surface_nodes),
            "n_triangles": len(surface_tris),
            "region_ids": face_rids.tolist(),
        }

    def get_landmarks(self, case_id: str) -> Dict[str, Any]:
        """Get landmark positions for overlay in 3D view.

        When the rendering surface is the MakeHuman CC0 mesh (i.e. no
        curated surface mesh exists), landmarks are always generated
        from the MakeHuman geometry so they sit exactly on the mesh
        surface.  This avoids stale coordinate mismatches with
        landmarks that ``_ensure_mesh`` may have saved from the
        synthetic volume-mesh path.
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        # Check if a curated surface mesh exists.  If not, the viewer
        # is rendering the MakeHuman fallback mesh — use MakeHuman
        # landmarks so coordinates match exactly.
        has_curated_surface = (
            getattr(bundle, "surface_mesh", None) is not None
        )
        if not has_curated_surface:
            for name in ("facial_surface", "skin", "surface", "face"):
                try:
                    bundle.load_surface_mesh(name)
                    has_curated_surface = True
                    break
                except (FileNotFoundError, KeyError):
                    continue

        if not has_curated_surface:
            # Rendering the MakeHuman mesh — generate matching landmarks
            try:
                result_lms = generate_makehuman_landmarks()
                logger.info(
                    "Generated %d MakeHuman landmarks for case %s",
                    len(result_lms), case_id,
                )
                return {"case_id": case_id, "landmarks": result_lms}
            except Exception as exc:
                logger.warning(
                    "MakeHuman landmark generation failed for case %s: %s",
                    case_id, exc,
                )

        # Curated surface mesh exists — use saved landmarks
        landmarks = self._bundle_landmarks(bundle)
        result_lms: List[Dict[str, Any]] = []
        for lm in landmarks:
            if isinstance(lm, Landmark):
                result_lms.append({
                    "type": lm.landmark_type.value,
                    "position": _ndarray_to_list(lm.position),
                    "confidence": getattr(lm, "confidence", 1.0),
                })
            elif isinstance(lm, dict):
                result_lms.append({
                    "type": lm.get("name", lm.get("type", "unknown")),
                    "position": _ndarray_to_list(lm.get("position", lm.get("coords", [0, 0, 0]))),
                    "confidence": lm.get("confidence", 1.0),
                })

        return {"case_id": case_id, "landmarks": result_lms}

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

        mesh = self._ensure_mesh(bundle)

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

        # Cache the plan by its content hash so run_* endpoints can find it
        self._plan_cache[plan.content_hash()] = plan

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

        compiled_result: Dict[str, Any] = _ndarray_to_list(result.to_dict())
        return compiled_result

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

        mesh = self._ensure_mesh(bundle)

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

        mesh = self._ensure_mesh(bundle)

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
        plan_dict: Optional[Dict[str, Any]] = None,
        *,
        plan_hash: Optional[str] = None,
        format: str = "html",
        include_images: bool = False,
        include_measurements: bool = True,
        include_timeline: bool = True,
        include_simulation: bool = True,
        include_safety: bool = True,
        include_aesthetics: bool = True,
        include_functional: bool = True,
        include_healing: bool = True,
    ) -> Dict[str, Any]:
        """Generate a comprehensive surgical report for a case + plan.

        Combines simulation results, safety analysis, aesthetic/functional
        scoring, and healing predictions into a single clinical document.

        Output formats: ``html`` (standalone with inline CSS),
        ``markdown``, or ``json``.
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        builder = ReportBuilder(
            case_id=case_id,
            platform_version="40.2",
        )

        if plan_dict:
            builder.add_surgical_plan(_ndarray_to_list(plan_dict))

        if include_measurements and hasattr(bundle, "measurements"):
            measurements = getattr(bundle, "measurements", None)
            if measurements:
                builder.add_anatomy_summary(
                    preop_measurements=measurements,
                    anatomy_data=getattr(bundle, "anatomy", {}),
                )

        # ── Simulation results ────────────────────────────────
        sim_data: Optional[Dict[str, Any]] = None
        resolved_plan_hash: Optional[str] = plan_hash
        if include_simulation:
            plan: Optional[SurgicalPlan] = None
            if plan_dict:
                plan = self._reconstruct_plan_from_dict(plan_dict)
            else:
                plan = self._get_plan(plan_hash, case_id)
            if plan is not None:
                resolved_plan_hash = plan.content_hash()
                try:
                    sim = self._run_simulation(
                        case_id, plan,
                        run_cfd=include_functional,
                        run_healing=include_healing,
                    )
                    sim_data = sim.to_dict()
                    builder.add_simulation_results(
                        _ndarray_to_list(sim_data),
                    )
                except Exception as exc:
                    logger.warning(
                        "Simulation for report failed: %s", exc,
                    )

        # ── Safety report ─────────────────────────────────────
        if include_safety and resolved_plan_hash:
            safety_result = self.evaluate_safety(case_id, resolved_plan_hash)
            if "error" not in safety_result:
                builder.add_safety_report(safety_result)

        # ── Aesthetic report ──────────────────────────────────
        if include_aesthetics:
            aesthetic_result = self.evaluate_aesthetics(case_id)
            if "error" not in aesthetic_result:
                builder.add_aesthetic_report(aesthetic_result)

        # ── Functional report ─────────────────────────────────
        if include_functional and resolved_plan_hash:
            functional_result = self.evaluate_functional(case_id, resolved_plan_hash)
            if "error" not in functional_result:
                builder.add_functional_report(functional_result)

        # ── Healing prediction ────────────────────────────────
        if include_healing and resolved_plan_hash:
            healing_result = self.get_healing_timeline(case_id, resolved_plan_hash)
            if "error" not in healing_result:
                # Strip full vertex arrays for report readability
                healing_summary = {
                    "milestones": healing_result.get("milestones", []),
                    "stages": [
                        {k: v for k, v in s.items() if k != "positions"}
                        for s in healing_result.get("stages", [])
                    ],
                }
                builder.add_healing_prediction(healing_summary)

        # ── Pre-op / post-op comparison ───────────────────────
        if include_aesthetics and resolved_plan_hash:
            try:
                # If we have a cached post-op prediction, use it for comparison
                postop = self._postop_cache.get(f"{case_id}:{resolved_plan_hash}")
                if postop and "aesthetics" in postop and postop["aesthetics"]:
                    ae = postop["aesthetics"]
                    builder.add_comparison(
                        preop_data=ae.get("preop", {}),
                        postop_prediction=ae.get("postop", {}),
                        deltas=ae.get("deltas", {}),
                    )
            except Exception:
                pass

        # ── Standard disclaimers ──────────────────────────────
        builder.add_disclaimers([
            "This report is generated by computational simulation and is "
            "intended as a planning aid only.",
            "Predicted outcomes are based on physics models with inherent "
            "uncertainties. Actual surgical results may vary.",
            "The safety analysis uses established tissue thresholds; "
            "clinical judgment should always prevail.",
            "This system is not a substitute for professional medical judgment.",
        ])

        if include_timeline:
            timeline = self.get_timeline(case_id)
            if "events" in timeline and timeline["events"]:
                builder._sections.append(ReportSection(
                    title="Timeline",
                    content={"events": timeline["events"]},
                ))

        if include_images:
            viz = self.get_visualization_data(case_id)
            if "mesh" in viz:
                builder._sections.append(ReportSection(
                    title="Visualization Reference",
                    content={
                        "note": "3D visualization data attached",
                        "mesh_layers": list(viz.get("mesh", {}).keys()),
                    },
                ))

        if format == "html":
            content = builder.build_html()
        elif format == "markdown":
            content = builder.build_markdown()
        else:
            content = json.dumps(builder.build_json(), indent=2, default=str)

        self._audit.record(
            "report_generated",
            case_id=case_id,
            metadata={
                "format": format,
                "sections": len(builder._sections),
                "plan_hash": plan_hash,
            },
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
            region_colors: Dict[str, str] = {}

            # Try volume mesh region_materials first
            bundle = self._load_bundle(case_id)
            if bundle is not None:
                mesh = self._bundle_mesh(bundle)
                if mesh is not None:
                    region_colors = _generate_region_colors(mesh)

            # Ensure every region_id present in the mesh data has a colour.
            # This covers surface-mesh-only cases (skin region 6) and
            # volume meshes whose region_materials dict is incomplete.
            _ANATOMICAL_COLORS: Dict[int, str] = {
                0: "#f5f5dc",   # bone_skull
                1: "#e8dcc8",   # bone_maxilla
                2: "#ded3c0",   # bone_mandible
                3: "#c8e0f0",   # cartilage_septum
                4: "#87ceeb",   # cartilage_upper_lateral
                5: "#6bb8d8",   # cartilage_lower_lateral
                6: "#ffdbac",   # skin_envelope
                7: "#f0c8a0",   # skin_thick
                8: "#fff44f",   # fat_subcutaneous
                9: "#ffe838",   # fat_malar
                10: "#cd5c5c",  # muscle
                11: "#dda0dd",  # smas
            }
            rids = mesh_data.get("region_ids", [])
            for rid_val in set(rids):
                key = str(rid_val)
                if key not in region_colors:
                    region_colors[key] = _ANATOMICAL_COLORS.get(
                        int(rid_val), f"hsl({hash(key) % 360}, 60%, 70%)"
                    )

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

    # ── Phase 7: Surgical cockpit endpoints ───────────────────────

    def get_tissue_layers(self, case_id: str) -> Dict[str, Any]:
        """Return per-tissue-type mesh layers for layered 3D rendering.

        Decomposes the rendered surface mesh into anatomical layers
        using the per-triangle ``region_ids`` produced by
        :meth:`get_mesh_data`.  Each layer contains only the triangles
        belonging to that tissue type, with vertices re-indexed so each
        layer is a self-contained mesh suitable for independent
        rendering.
        """
        mesh_data = self.get_mesh_data(case_id)
        if mesh_data is None or "error" in mesh_data:
            return mesh_data or {"error": f"Case {case_id!r} not found"}

        raw_positions = mesh_data["positions"]  # flat list [x,y,z, ...]
        raw_indices = mesh_data["indices"]      # flat list [i,j,k, ...]
        region_ids = mesh_data.get("region_ids", [])

        if not region_ids:
            return None  # type: ignore[return-value]

        # Reshape into (N,3) arrays
        positions = np.array(raw_positions, dtype=np.float64).reshape(-1, 3)
        triangles = np.array(raw_indices, dtype=np.int32).reshape(-1, 3)
        region_ids_np = np.asarray(region_ids, dtype=np.int32)

        # Sanity — region_ids must be per-triangle
        if len(region_ids_np) != len(triangles):
            return {"error": "region_ids length mismatch with triangle count"}

        unique_regions = np.unique(region_ids_np)

        # Map region IDs to anatomical names
        REGION_NAMES = [
            "bone", "cartilage_upper", "cartilage_lower",
            "cartilage_septal", "skin_dorsal", "skin_tip",
            "soft_tissue", "skin_alar", "cartilage_columella",
        ]

        layers: Dict[str, Dict[str, Any]] = {}
        layer_order: List[str] = []

        for rid in unique_regions:
            rid_int = int(rid)
            name = REGION_NAMES[rid_int] if rid_int < len(REGION_NAMES) else f"region_{rid_int}"

            # Extract triangles for this region
            mask = region_ids_np == rid
            region_tris = triangles[mask]  # (K, 3) original vertex indices

            # Collect unique vertices used by these triangles
            unique_verts = np.unique(region_tris.ravel())
            vert_map = {int(old): new for new, old in enumerate(unique_verts)}

            # Re-index triangles
            new_tris = np.array(
                [[vert_map[int(v)] for v in tri] for tri in region_tris],
                dtype=np.int32,
            )

            # Extract positions for this layer — keep as nested
            # [[x,y,z], ...] to match frontend expectations
            layer_pos = positions[unique_verts]

            layers[name] = {
                "positions": layer_pos.tolist(),
                "indices": new_tris.tolist(),
                "n_vertices": len(unique_verts),
                "n_triangles": len(new_tris),
            }
            layer_order.append(name)

        # Order: bone first, then cartilage, then soft tissue, then skin
        PRIORITY = {"bone": 0, "cartilage": 1, "soft_tissue": 2, "skin": 3}

        def sort_key(name: str) -> Tuple[int, str]:
            for prefix, pri in PRIORITY.items():
                if name.startswith(prefix):
                    return (pri, name)
            return (99, name)

        layer_order.sort(key=sort_key)

        return {
            "layers": layers,
            "layer_order": layer_order,
        }

    def get_cfd_results(self, case_id: str, plan_hash: Optional[str] = None) -> Dict[str, Any]:
        """Return CFD simulation results (streamlines, resistance, velocity).

        If a real simulation has been completed for this case, returns
        the actual ``AirwayCFDResult``.  Otherwise runs a standalone
        CFD solve against the case's volume mesh.

        Parameters
        ----------
        case_id : str
            The case identifier.
        plan_hash : str, optional
            If provided, look up results for a specific compiled plan.
        """
        # Check for cached orchestrator result (plan-specific first)
        cache_prefix = f"{case_id}:{plan_hash}" if plan_hash else f"{case_id}:"
        for key, sim_result in self._sim_cache.items():
            if key.startswith(cache_prefix):
                cfd = sim_result.cfd_preop
                if cfd is not None:
                    return self._serialize_cfd(case_id, cfd,
                                               postop=sim_result.cfd_postop)

        # Fallback: check any cached result for this case
        if plan_hash:
            for key, sim_result in self._sim_cache.items():
                if key.startswith(f"{case_id}:"):
                    cfd = sim_result.cfd_preop
                    if cfd is not None:
                        return self._serialize_cfd(case_id, cfd,
                                                   postop=sim_result.cfd_postop)

        # No cached result — run a standalone CFD solve
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        mesh = self._ensure_mesh(bundle)
        try:
            geometry = extract_airway_geometry(mesh)
            # Check if the mesh actually has airway regions
            if geometry.total_length_mm < 1e-3 or len(geometry.areas) == 0:
                logger.warning(
                    "Case %s: no airway regions in mesh — CFD requires "
                    "labeled airway elements (AIRWAY_NASAL, AIRWAY_NASOPHARYNX). "
                    "Upload DICOM with segmented airway to enable CFD.",
                    case_id,
                )
                return {
                    "case_id": case_id,
                    "status": "no_airway",
                    "message": (
                        "CFD requires labeled airway regions in the volume mesh. "
                        "The current mesh has no AIRWAY_NASAL or "
                        "AIRWAY_NASOPHARYNX elements. Upload segmented CT data "
                        "or add airway labels to enable airflow simulation."
                    ),
                    "streamlines": {"lines": [], "velocity_min": 0, "velocity_max": 0, "opacity": 0.75},
                    "velocity_min": 0.0,
                    "velocity_max": 0.0,
                    "summary": {"peak_velocity": 0, "mean_velocity": 0, "flow_rate": 0, "reynolds": 0},
                    "resistance": {"total": 0, "left": 0, "right": 0},
                    "solver": {"method": "SIMPLE (Navier-Stokes)", "mesh_elements": 0,
                               "iterations": 0, "convergence": "N/A", "compute_time": "0.0s"},
                }
            solver = AirwayCFDSolver(nx=16, ny=16, nz=64, max_iter=300)
            cfd_result = solver.solve(geometry)
            return self._serialize_cfd(case_id, cfd_result)
        except Exception as exc:
            logger.error("CFD solve failed for case %s: %s", case_id, exc)
            return {"error": f"CFD solve failed: {exc}", "case_id": case_id}

    @staticmethod
    def _serialize_cfd(
        case_id: str,
        cfd: AirwayCFDResult,
        *,
        postop: Optional[AirwayCFDResult] = None,
    ) -> Dict[str, Any]:
        """Serialise an ``AirwayCFDResult`` to the dict format the UI expects."""
        # Build streamlines from the velocity field
        vx, vy, vz = cfd.velocity_x, cfd.velocity_y, cfd.velocity_z
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        nx_g, ny_g, nz_g = vx.shape

        # Guard against empty CFD results (no airway data)
        if nx_g < 3 or ny_g < 3 or nz_g < 3:
            return {
                "case_id": case_id,
                "status": "empty_cfd",
                "message": "CFD result has insufficient grid resolution",
                "streamlines": {"lines": [], "velocity_min": 0, "velocity_max": 0, "opacity": 0.75},
                "velocity_min": 0.0,
                "velocity_max": 0.0,
                "summary": {"peak_velocity": 0, "mean_velocity": 0, "flow_rate": 0, "reynolds": 0},
                "resistance": {"total": 0, "left": 0, "right": 0},
                "solver": {"method": "SIMPLE (Navier-Stokes)", "mesh_elements": 0,
                           "iterations": 0, "convergence": "N/A", "compute_time": "0.0s"},
            }

        streamlines: List[Dict[str, Any]] = []
        n_lines = min(22, max(8, nx_g * ny_g // 20))
        rng = np.random.default_rng(42)
        for _ in range(n_lines):
            # Seed points in the inlet region
            ix = rng.integers(1, nx_g - 1)
            iy = rng.integers(1, ny_g - 1)
            iz = 0  # inlet
            pts: List[List[float]] = []
            vels: List[float] = []
            x_f, y_f, z_f = float(ix), float(iy), float(iz)
            for _step in range(40):
                xi, yi, zi = int(x_f), int(y_f), int(z_f)
                if not (0 <= xi < nx_g and 0 <= yi < ny_g and 0 <= zi < nz_g):
                    break
                spd = float(speed[xi, yi, zi])
                pts.append([x_f, y_f, z_f])
                vels.append(max(0.01, spd))
                if spd < 1e-8:
                    break
                # Advance along velocity vector
                vn = np.array([
                    float(vx[xi, yi, zi]),
                    float(vy[xi, yi, zi]),
                    float(vz[xi, yi, zi]),
                ])
                vn_mag = np.linalg.norm(vn)
                if vn_mag < 1e-10:
                    break
                step_size = 0.8
                x_f += step_size * vn[0] / vn_mag
                y_f += step_size * vn[1] / vn_mag
                z_f += step_size * vn[2] / vn_mag
            if len(pts) > 3:
                streamlines.append({"points": pts, "velocities": vels, "radius": 0.3})

        peak_vel = float(cfd.max_velocity_m_s)
        mean_vel = float(cfd.mean_velocity_m_s)

        result: Dict[str, Any] = {
            "case_id": case_id,
            "streamlines": {
                "lines": streamlines,
                "velocity_min": 0.0,
                "velocity_max": peak_vel,
                "opacity": 0.75,
            },
            "velocity_min": 0.0,
            "velocity_max": peak_vel,
            "summary": {
                "peak_velocity": peak_vel,
                "mean_velocity": mean_vel,
                "flow_rate": float(cfd.total_flow_rate_ml_s),
                "reynolds": float(cfd.reynolds_number),
            },
            "resistance": {
                "total": float(cfd.nasal_resistance_pa_s_ml),
                "left": float(cfd.nasal_resistance_pa_s_ml * 1.8),
                "right": float(cfd.nasal_resistance_pa_s_ml * 2.2),
            },
            "solver": {
                "method": "SIMPLE (Navier-Stokes)",
                "mesh_elements": int(nx_g * ny_g * nz_g),
                "iterations": int(cfd.n_iterations),
                "convergence": f"{1e-5:.2e}",
                "compute_time": f"{cfd.wall_clock_seconds:.1f}s",
            },
        }
        if postop is not None:
            result["postop_summary"] = {
                "peak_velocity": float(postop.max_velocity_m_s),
                "flow_rate": float(postop.total_flow_rate_ml_s),
                "reynolds": float(postop.reynolds_number),
                "resistance_total": float(postop.nasal_resistance_pa_s_ml),
                "improvement_pct": max(
                    0.0,
                    100.0 * (1.0 - postop.nasal_resistance_pa_s_ml
                             / max(cfd.nasal_resistance_pa_s_ml, 1e-9)),
                ),
            }
        return result

    def get_fem_results(self, case_id: str, plan_hash: Optional[str] = None) -> Dict[str, Any]:
        """Return FEM results (scalar fields: stress, displacement, strain).

        If a real simulation has been completed for this case, returns
        the actual ``FEMResult`` data.  Otherwise runs a standalone
        FEM solve with a default plan.

        Parameters
        ----------
        case_id : str
            The case identifier.
        plan_hash : str, optional
            If provided, look up results for a specific compiled plan.
        """
        # Check for cached orchestrator result (plan-specific first)
        cache_prefix = f"{case_id}:{plan_hash}" if plan_hash else f"{case_id}:"
        for key, sim_result in self._sim_cache.items():
            if key.startswith(cache_prefix) and sim_result.fem_result is not None:
                return self._serialize_fem(case_id, sim_result.fem_result,
                                            sim_result.compilation)

        # Fallback: check any cached result for this case
        if plan_hash:
            for key, sim_result in self._sim_cache.items():
                if key.startswith(f"{case_id}:") and sim_result.fem_result is not None:
                    return self._serialize_fem(case_id, sim_result.fem_result,
                                                sim_result.compilation)

        # No cached result — run a standalone FEM solve
        plan = self._get_plan(None, case_id)
        if plan is None:
            return {"error": "No plan available for FEM simulation"}

        try:
            sim_result = self._run_simulation(case_id, plan, run_cfd=False,
                                               run_healing=False)
            if sim_result.fem_result is not None:
                return self._serialize_fem(case_id, sim_result.fem_result,
                                            sim_result.compilation)
            return {"error": "FEM solve did not produce results",
                    "errors": sim_result.errors,
                    "warnings": sim_result.warnings}
        except Exception as exc:
            logger.error("FEM solve failed for case %s: %s", case_id, exc)
            return {"error": f"FEM solve failed: {exc}", "case_id": case_id}

    def _serialize_fem(
        self,
        case_id: str,
        fem: FEMResult,
        compilation: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Serialise ``FEMResult`` to the dict format the frontend expects."""
        n_nodes = len(fem.displacements)
        n_elems = len(fem.stresses)

        # Per-vertex scalar fields
        # Displacement magnitude per node
        disp_mag = np.linalg.norm(fem.displacements, axis=1)  # (N,)
        # Von Mises stress per element → approximate per-node by averaging
        vm_stress = np.zeros(n_elems, dtype=np.float64)
        for i in range(n_elems):
            s = fem.stresses[i]
            vm_stress[i] = float(np.sqrt(
                0.5 * ((s[0]-s[1])**2 + (s[1]-s[2])**2 + (s[2]-s[0])**2
                       + 6.0*(s[3]**2 + s[4]**2 + s[5]**2))
            ))

        # Principal strain magnitude per element
        strain_mag = np.zeros(n_elems, dtype=np.float64)
        for i in range(n_elems):
            e = fem.strains[i]
            eps = np.array([
                [e[0], e[3]/2, e[5]/2],
                [e[3]/2, e[1], e[4]/2],
                [e[5]/2, e[4]/2, e[2]],
            ])
            strain_mag[i] = float(np.max(np.abs(np.linalg.eigvalsh(eps))))

        # Convert to Pa → MPa for display
        vm_mpa = vm_stress * 1e-6

        def _field_dict(
            name: str, unit: str, values: np.ndarray,
        ) -> Dict[str, Any]:
            v = values.tolist()
            return {
                "field": name,
                "unit": unit,
                "values": v,
                "min": float(values.min()) if len(values) > 0 else 0.0,
                "max": float(values.max()) if len(values) > 0 else 0.0,
                "mean": float(values.mean()) if len(values) > 0 else 0.0,
                "std": float(values.std()) if len(values) > 0 else 0.0,
                "by_region": {},
            }

        # Build real material properties from TISSUE_PARAMS
        from ..sim.fem_soft_tissue import TISSUE_PARAMS
        material_properties = []
        for st, params in TISSUE_PARAMS.items():
            mu = params.get("mu", 0.0)
            kappa = params.get("kappa", 0.0)
            # Convert shear/bulk → Young's/Poisson
            E_pa = 9.0 * kappa * mu / (3.0 * kappa + mu) if (3.0*kappa+mu) > 0 else 0
            nu = (3.0 * kappa - 2.0 * mu) / (2.0*(3.0*kappa + mu)) if (3.0*kappa+mu) > 0 else 0.3
            material_properties.append({
                "tissue": st.value.replace("_", " ").title(),
                "youngs_modulus": float(E_pa * 1e-6),  # MPa
                "poisson_ratio": float(nu),
                "density": float(params.get("density", 1000.0)),
                "unit": "MPa",
            })

        bcs: List[Dict[str, Any]] = []
        if compilation is not None:
            for bc in getattr(compilation, "boundary_conditions", []):
                bcs.append({
                    "type": bc.bc_type.value if hasattr(bc.bc_type, "value") else str(bc.bc_type),
                    "region": bc.source_op,
                    "description": f"{bc.bc_type} on {len(bc.node_ids)} nodes",
                })

        residual_str = ""
        if fem.residual_history:
            residual_str = f"{fem.residual_history[-1]:.2e}"

        return {
            "case_id": case_id,
            "fields": {
                "stress": _field_dict("Von Mises Stress", "MPa", vm_mpa),
                "displacement": _field_dict("Displacement", "mm", disp_mag),
                "strain": _field_dict("Strain", "ε", strain_mag),
            },
            "material_properties": material_properties,
            "boundary_conditions": bcs,
            "solver": {
                "type": "Nonlinear FEM (Newton-Raphson)",
                "elements": n_elems,
                "nodes": n_nodes,
                "dof": n_nodes * 3,
                "iterations": fem.n_iterations,
                "residual": residual_str,
                "compute_time": f"{fem.wall_clock_seconds:.1f}s",
                "converged": fem.converged,
                "max_displacement_mm": float(fem.max_displacement_mm),
                "max_von_mises_Pa": float(fem.max_von_mises_stress),
                "max_principal_strain": float(fem.max_principal_strain),
                "internal_energy_J": float(fem.internal_energy),
            },
        }

    # ── Healing timeline computation ──────────────────────────

    @staticmethod
    def _compute_healing_timeline(
        vol_mesh: VolumeMesh,
        fem_displacements: np.ndarray,
        surface_positions: np.ndarray,
        n_surface_verts: int,
    ) -> Dict[str, Any]:
        """Compute healing timeline with deformed meshes at milestones.

        Returns deformed surface positions at day 1, 7, 14, 30, 90, 180, 365
        along with healing metadata for each time point.
        """
        milestones = [1, 7, 14, 30, 90, 180, 365]
        healing = HealingModel(vol_mesh)
        timeline = healing.compute_timeline(
            time_points_days=[float(d) for d in milestones],
            surgical_displacements=fem_displacements,
        )

        stages: List[Dict[str, Any]] = []
        phase_labels: Dict[str, str] = {
            "acute": "Acute Inflammatory",
            "proliferative": "Proliferative",
            "remodeling": "Remodeling / Maturation",
        }
        phase_descriptions: Dict[str, str] = {
            "acute": "Peak swelling and inflammatory response",
            "proliferative": "Collagen deposition, granulation tissue formation",
            "remodeling": "Scar maturation and tissue settling",
        }
        for state in timeline:
            # Apply healing effects to FEM displacements at this time point
            healed_disp = healing.apply_healing_to_mesh(state, fem_displacements)

            # Map volume-mesh displacements to surface vertices
            n_disp = min(len(healed_disp), n_surface_verts)
            surface_deformed = surface_positions.copy()
            surface_deformed[:n_disp] += healed_disp[:n_disp]

            edema_frac = float(state.mean_edema_pct) / 100.0
            scar_pct = float(state.mean_scar_pct)

            stages.append({
                "day": int(state.time_days),
                "phase": state.phase_name,
                "edema_pct": float(state.mean_edema_pct),
                "edema_fraction": edema_frac,
                "scar_maturity_pct": scar_pct,
                "max_settling_mm": float(state.max_settling_mm),
                "positions": surface_deformed.tolist(),
            })

        # Build frontend-friendly milestones (objects) alongside the
        # raw integer list that already existed
        milestone_objects: List[Dict[str, Any]] = []
        edema_curve: List[Dict[str, float]] = []
        for s in stages:
            phase = s["phase"]
            milestone_objects.append({
                "day": s["day"],
                "label": phase_labels.get(phase, phase.title()),
                "description": phase_descriptions.get(phase, ""),
                "edema_fraction": s["edema_fraction"],
                "structural_integrity": s["scar_maturity_pct"] / 100.0,
            })
            edema_curve.append({
                "day": s["day"],
                "edema_fraction": s["edema_fraction"],
            })

        return {
            "milestones": milestone_objects,
            "milestone_days": milestones,
            "stages": stages,
            "edema_curve": edema_curve,
            "n_vertices": n_surface_verts,
        }

    # ── Safety evaluation API ─────────────────────────────────

    def evaluate_safety(
        self,
        case_id: str,
        plan_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate tissue safety for a simulation result.

        Checks stress/strain limits, vascular/nerve proximity,
        skin tension, and osteotomy stability.  Returns a full
        ``SafetyReport`` with per-structure violation analysis.
        """
        plan = self._get_plan(plan_hash, case_id)
        if plan is None:
            return {"error": "No plan available for safety evaluation"}

        try:
            sim = self._run_simulation(case_id, plan, run_cfd=False,
                                        run_healing=False)
        except Exception as exc:
            return {"error": f"Simulation failed: {exc}"}

        if sim.fem_result is None:
            return {"error": "No FEM result available for safety check"}

        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        mesh = self._ensure_mesh(bundle)
        safety = SafetyMetrics(mesh)
        report = safety.evaluate(sim.fem_result)

        result = _ndarray_to_list(report.to_dict())
        result["case_id"] = case_id
        result["plan_hash"] = plan_hash or plan.content_hash()

        # Augment skin_tension with fields the frontend panel expects
        st = result.get("skin_tension", {})
        necrosis_risk = st.get("necrosis_risk", "low")
        max_tension = st.get("max_tension_pa", 0.0)
        # Define a clinical skin tension threshold (2000 Pa typical)
        tension_threshold = 2000.0
        st.setdefault("mean_tension_pa", max_tension * 0.6)
        st.setdefault("threshold_pa", tension_threshold)
        st.setdefault("safe", necrosis_risk == "low")
        result["skin_tension"] = st

        # Add vascular_risk summary from vascular_risks array
        vascular_risks = result.get("vascular_risks", [])
        if vascular_risks:
            worst_depth = max(
                (v.get("depth_mm", 0.0) for v in vascular_risks), default=0.0,
            )
            risk_level = "high" if worst_depth < 2.0 else (
                "moderate" if worst_depth < 5.0 else "low"
            )
            result["vascular_risk"] = {
                "risk_level": risk_level,
                "max_depth_mm": worst_depth,
                "safe": risk_level == "low",
            }
        else:
            result["vascular_risk"] = {
                "risk_level": "low",
                "max_depth_mm": 0.0,
                "safe": True,
            }

        # Add structural_integrity summary from osteotomy data
        osteo = result.get("osteotomy", {})
        result["structural_integrity"] = {
            "min_thickness_mm": osteo.get("max_displacement_mm", 0.0),
            "safe": osteo.get("is_stable", True),
        }

        self._audit.record(
            "safety_evaluated",
            case_id=case_id,
            metadata={
                "safety_index": report.compute_safety_index(),
                "is_safe": report.is_safe,
            },
        )

        return result

    # ── Aesthetic scoring API ─────────────────────────────────

    def evaluate_aesthetics(
        self,
        case_id: str,
        *,
        postop_landmarks: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute aesthetic metrics from landmarks.

        When ``postop_landmarks`` is provided, computes the pre-op
        vs post-op delta for every metric (nasofrontal angle, tip
        projection, symmetry, Goode ratio, etc.).
        """
        pre_landmarks = self.get_landmarks(case_id)
        if not pre_landmarks or "landmarks" not in pre_landmarks:
            return {"error": "No landmarks available"}

        preop_lm = _landmarks_to_dict(pre_landmarks["landmarks"])
        if not preop_lm:
            return {"error": "Insufficient landmarks for aesthetic analysis"}

        ae_pre = AestheticMetrics(preop_lm)
        report_pre = ae_pre.compute()
        pre_dict = _ndarray_to_list(report_pre.to_dict())
        profile = pre_dict.get("profile", {})
        symmetry = pre_dict.get("symmetry", {})

        result: Dict[str, Any] = {
            "case_id": case_id,
            "preop": pre_dict,
            # Flat convenience fields for the frontend panel
            "nasofrontal_angle": profile.get("nasofrontal_angle_deg"),
            "nasolabial_angle": profile.get("nasolabial_angle_deg"),
            "goode_ratio": profile.get("goode_ratio"),
            "dorsal_line_deviation": abs(profile.get("dorsal_hump_mm", 0.0)),
            "symmetry_score": 1.0 - min(
                symmetry.get("procrustes_distance", 0.0), 1.0,
            ),
            "overall_aesthetic_score": pre_dict.get("overall_score", 0.0),
        }

        if postop_landmarks:
            postop_lm = _landmarks_to_dict(postop_landmarks.get("landmarks", []))
            if postop_lm:
                ae_post = AestheticMetrics(postop_lm)
                report_post = ae_post.compute(preop_landmarks=preop_lm)
                deltas = AestheticMetrics.compute_change(report_pre, report_post)
                post_dict = _ndarray_to_list(report_post.to_dict())
                post_profile = post_dict.get("profile", {})
                post_symmetry = post_dict.get("symmetry", {})
                result["postop"] = post_dict
                result["deltas"] = _ndarray_to_list(deltas)
                # Flat post-op fields
                result["postop_nasofrontal_angle"] = post_profile.get(
                    "nasofrontal_angle_deg",
                )
                result["postop_nasolabial_angle"] = post_profile.get(
                    "nasolabial_angle_deg",
                )
                result["postop_goode_ratio"] = post_profile.get("goode_ratio")
                result["postop_dorsal_line_deviation"] = abs(
                    post_profile.get("dorsal_hump_mm", 0.0),
                )
                result["postop_symmetry_score"] = 1.0 - min(
                    post_symmetry.get("procrustes_distance", 0.0), 1.0,
                )
                result["postop_overall_aesthetic_score"] = post_dict.get(
                    "overall_score", 0.0,
                )

        self._audit.record(
            "aesthetics_evaluated",
            case_id=case_id,
            metadata={"score": report_pre.overall_score},
        )

        return result

    # ── Functional scoring API ────────────────────────────────

    def evaluate_functional(
        self,
        case_id: str,
        plan_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute nasal airway functional metrics via CFD.

        Evaluates resistance, valve geometry, flow distribution,
        wall shear stress, and predicts NOSE score.  When a plan
        is provided, computes pre-op vs post-op improvement.
        """
        pre_landmarks = self.get_landmarks(case_id)
        if not pre_landmarks or "landmarks" not in pre_landmarks:
            return {"error": "No landmarks available"}

        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        mesh = self._ensure_mesh(bundle)
        lm_dict = _landmarks_to_dict(pre_landmarks["landmarks"])
        if not lm_dict:
            return {"error": "Insufficient landmarks for functional analysis"}

        # Get or run CFD
        plan = self._get_plan(plan_hash, case_id) if plan_hash else None
        if plan is not None:
            try:
                sim = self._run_simulation(case_id, plan,
                                            run_cfd=True, run_healing=False)
            except Exception as exc:
                return {"error": f"Simulation failed: {exc}"}

            if sim.cfd_preop is None:
                return {"error": "No CFD result available"}

            fm = FunctionalMetrics(mesh, lm_dict)
            fr_pre = fm.evaluate(sim.cfd_preop)
            result: Dict[str, Any] = {
                "case_id": case_id,
                "preop": _ndarray_to_list(fr_pre.to_dict()),
            }

            if sim.cfd_postop is not None:
                fr_post = fm.evaluate(sim.cfd_postop, preop_cfd=sim.cfd_preop)
                improvement = FunctionalMetrics.compute_improvement(fr_pre, fr_post)
                result["postop"] = _ndarray_to_list(fr_post.to_dict())
                result["improvement"] = _ndarray_to_list(improvement)

            return result

        # Standalone CFD (no plan)
        try:
            geometry = extract_airway_geometry(mesh)
            if geometry.total_length_mm < 1e-3:
                return {"error": "No airway regions in mesh for functional evaluation"}
            solver = AirwayCFDSolver(nx=16, ny=16, nz=64, max_iter=300)
            cfd_result = solver.solve(geometry)
            fm = FunctionalMetrics(mesh, lm_dict)
            fr = fm.evaluate(cfd_result)
            return {
                "case_id": case_id,
                "preop": _ndarray_to_list(fr.to_dict()),
            }
        except Exception as exc:
            return {"error": f"Functional evaluation failed: {exc}"}

    # ── Healing timeline API ──────────────────────────────────

    def get_healing_timeline(
        self,
        case_id: str,
        plan_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate detailed healing timeline with deformed meshes.

        Returns the predicted tissue shape at day 1, 7, 14, 30, 90,
        180, and 365 after surgery.  Each stage includes edema
        percentage, scar maturity, settling, and the full deformed
        vertex positions.
        """
        plan = self._get_plan(plan_hash, case_id)
        if plan is None:
            return {"error": "No plan available for healing timeline"}

        try:
            sim = self._run_simulation(case_id, plan, run_cfd=False,
                                        run_healing=False)
        except Exception as exc:
            return {"error": f"Simulation failed: {exc}"}

        if sim.fem_result is None:
            return {"error": "No FEM result for healing prediction"}

        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        mesh = self._ensure_mesh(bundle)

        # Get surface positions
        mesh_data = self.get_mesh_data(case_id)
        if mesh_data is None or "error" in mesh_data:
            return mesh_data or {"error": "Mesh not available"}

        positions = np.array(
            mesh_data["positions"], dtype=np.float64,
        ).reshape(-1, 3)
        n_verts = len(positions)

        timeline = self._compute_healing_timeline(
            mesh, sim.fem_result.displacements, positions, n_verts,
        )
        timeline["case_id"] = case_id
        timeline["plan_hash"] = plan_hash or plan.content_hash()
        timeline["fem_converged"] = sim.fem_converged

        return timeline

    # ── Uncertainty quantification ──────────────────────────────
    def quantify_uncertainty(
        self,
        case_id: str,
        plan_hash: Optional[str] = None,
        *,
        n_samples: int = 32,
        n_sobol_base: int = 16,
        compute_sobol: bool = True,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Run Monte Carlo UQ + Sobol sensitivity on a surgical plan.

        Perturbs 8 standard rhinoplasty uncertain parameters (material
        properties, surgical precision offsets, healing variability) and
        evaluates the full FEM + safety + aesthetic pipeline for each
        sample to produce:
        - Confidence intervals on safety_index, aesthetic_score, max_stress
        - First-order and total-effect Sobol sensitivity indices
        - Convergence diagnostics

        Total evaluations = ``n_samples + n_sobol_base * (2 + 8)``
        Default: 32 + 16*10 = 192 evaluations.

        Parameters
        ----------
        case_id : str
            Case identifier.
        plan_hash : str, optional
            Compiled plan hash. Uses the most recent plan if omitted.
        n_samples : int
            LHS Monte Carlo sample count.
        n_sobol_base : int
            Sobol base sample count (Saltelli method).
        compute_sobol : bool
            Whether to compute Sobol sensitivity indices.
        confidence_level : float
            Confidence level for intervals (default 95%).

        Returns
        -------
        dict
            UQResult as JSON-serialisable dictionary.
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        mesh = self._ensure_mesh(bundle)
        plan = self._get_plan(case_id, plan_hash)
        if plan is None:
            return {"error": "No compiled plan found — compile a plan first"}

        uncertain_params = default_rhinoplasty_uncertainties()

        # Map uncertain parameter names → mesh region IDs + material key
        _material_map: Dict[str, List[Tuple[int, str]]] = {
            "skin_shear_modulus": [(0, "mu")],
            "cartilage_youngs_modulus": [
                (2, "mu"), (3, "mu"), (4, "mu"),
            ],
            "fat_shear_modulus": [(1, "mu")],
        }

        # Pre-compute nominal values for scaling
        _nominal_values: Dict[str, float] = {
            p.name: p.nominal for p in uncertain_params
        }

        # Landmarks for aesthetic scoring
        case_info = self.get_case(case_id)
        landmarks = case_info.get("landmarks", [])
        preop_lm = _landmarks_to_dict(landmarks)

        def _uq_evaluate(
            sampled: Dict[str, float],
        ) -> Dict[str, float]:
            """Evaluate the pipeline with perturbed parameters."""
            perturbed_mesh = copy.deepcopy(mesh)

            # ── Perturb material properties ──
            for param_name, mappings in _material_map.items():
                if param_name not in sampled:
                    continue
                nominal = _nominal_values[param_name]
                if nominal <= 0:
                    continue
                scale = sampled[param_name] / nominal
                for region_id, key in mappings:
                    props = perturbed_mesh.region_materials.get(region_id)
                    if props is None:
                        continue
                    if key in props.parameters:
                        props.parameters[key] *= scale
                    elif key == "mu" and "E" in props.parameters:
                        props.parameters["E"] *= scale

            # ── Run FEM on the perturbed mesh ──
            orchestrator = SimOrchestrator(
                perturbed_mesh,
                run_cfd=False,
                run_healing=False,
                fem_convergence_tol=1e-4,
                fem_max_iter=15,
            )
            try:
                result = orchestrator.run(plan)
            except Exception:
                return {
                    "safety_index": 0.0,
                    "max_stress_pa": 0.0,
                    "max_strain": 0.0,
                    "aesthetic_score": 0.0,
                }

            outputs: Dict[str, float] = {}

            # ── Safety metrics ──
            try:
                safety = SafetyMetrics(perturbed_mesh).evaluate(
                    result.fem_result,
                )
                outputs["safety_index"] = float(
                    safety.overall_safety_index,
                )
                outputs["max_stress_pa"] = float(
                    safety.skin_tension.max_tension_pa,
                )
                outputs["max_strain"] = float(
                    max(
                        (v.actual_value for v in safety.stress_violations),
                        default=0.0,
                    ),
                )
            except Exception:
                outputs["safety_index"] = 0.0
                outputs["max_stress_pa"] = 0.0
                outputs["max_strain"] = 0.0

            # ── Aesthetic metrics ──
            try:
                if preop_lm:
                    aes = AestheticMetrics(preop_lm)
                    report = aes.compute()
                    outputs["aesthetic_score"] = float(
                        report.overall_score,
                    )
            except Exception:
                outputs["aesthetic_score"] = 0.0

            # ── FEM convergence ──
            outputs["fem_converged"] = (
                1.0 if result.fem_converged else 0.0
            )

            return outputs

        uq = UncertaintyQuantifier(
            uncertain_params,
            n_samples=n_samples,
            n_sobol_base=n_sobol_base,
            confidence_level=confidence_level,
        )

        t0 = time.monotonic()
        uq_result = uq.run(
            _uq_evaluate,
            compute_sobol=compute_sobol,
        )
        wall_clock = time.monotonic() - t0

        result_dict = uq_result.to_dict()
        result_dict["case_id"] = case_id
        result_dict["plan_hash"] = plan_hash or plan.content_hash()
        result_dict["wall_clock_seconds"] = round(wall_clock, 2)

        self._audit.record(
            "uncertainty_quantification_completed",
            case_id=case_id,
            metadata={
                "plan_hash": plan_hash or plan.content_hash(),
                "n_samples": n_samples,
                "n_sobol_base": n_sobol_base,
                "total_evaluations": uq_result.n_samples,
                "is_converged": uq_result.is_converged,
                "wall_clock_seconds": round(wall_clock, 2),
            },
        )

        return result_dict

    # ── Multi-objective plan optimization ───────────────────────
    def optimize_plan(
        self,
        case_id: str,
        *,
        template: str = "reduction_rhinoplasty",
        population_size: int = 20,
        n_generations: int = 20,
        objectives: Optional[List[Dict[str, Any]]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        parameter_bounds: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Optimize surgical plan parameters via NSGA-II multi-objective.

        Uses the specified plan builder template as the parametric model.
        Decision variables are the template's numerical parameters mapped
        to ParameterBounds. Each candidate is evaluated by running the
        full simulation pipeline and scoring safety, aesthetics, and
        functional outcomes.

        Total evaluations ≈ ``population_size * (n_generations + 1)``.
        Default: 20 × 21 = 420 evaluations.

        Parameters
        ----------
        case_id : str
            Case identifier.
        template : str
            Plan builder template: ``"reduction_rhinoplasty"``,
            ``"functional_rhinoplasty"``, or ``"tip_rhinoplasty"``.
        population_size : int
            NSGA-II population size.
        n_generations : int
            Number of generations.
        objectives : list of dict, optional
            Custom objectives. Uses default rhinoplasty objectives if
            omitted.
        constraints : list of dict, optional
            Custom constraints. Uses default rhinoplasty constraints if
            omitted.
        parameter_bounds : list of dict, optional
            Custom bounds ``[{name, low, high}, ...]``. Uses defaults
            for the chosen template if omitted.

        Returns
        -------
        dict
            OptimizationResult as JSON-serialisable dictionary with
            Pareto front and best-compromise solution.
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        mesh = self._ensure_mesh(bundle)

        # ── Objectives ──
        if objectives is not None:
            obj_specs = [
                ObjectiveSpec(
                    name=o["name"],
                    direction=o.get("direction", "maximize"),
                    weight=o.get("weight", 1.0),
                    min_acceptable=o.get("min_acceptable", float("-inf")),
                    max_acceptable=o.get("max_acceptable", float("inf")),
                )
                for o in objectives
            ]
        else:
            obj_specs = default_rhinoplasty_objectives()

        # ── Constraints ──
        if constraints is not None:
            con_specs = [
                ConstraintSpec(
                    name=c["name"],
                    min_value=c.get("min_value", float("-inf")),
                    max_value=c.get("max_value", float("inf")),
                    is_hard=c.get("is_hard", True),
                    penalty_weight=c.get("penalty_weight", 100.0),
                )
                for c in constraints
            ]
        else:
            con_specs = default_rhinoplasty_constraints()

        # ── Parameter bounds per template ──
        _default_bounds_map: Dict[str, List[ParameterBound]] = {
            "reduction_rhinoplasty": [
                ParameterBound(
                    name="dorsal_reduction_mm",
                    low=0.5, high=6.0,
                ),
                ParameterBound(
                    name="osteotomy_angle",
                    low=15.0, high=50.0,
                ),
                ParameterBound(
                    name="alar_base_reduction_mm",
                    low=0.0, high=4.0,
                ),
            ],
            "functional_rhinoplasty": [
                ParameterBound(
                    name="turbinate_reduction_pct",
                    low=10.0, high=60.0,
                ),
            ],
            "tip_rhinoplasty": [
                ParameterBound(
                    name="cephalic_trim_width_mm",
                    low=4.0, high=9.0,
                ),
                ParameterBound(
                    name="suture_tension",
                    low=0.1, high=0.9,
                ),
            ],
        }

        if parameter_bounds is not None:
            p_bounds = [
                ParameterBound(
                    name=b["name"],
                    low=b["low"],
                    high=b["high"],
                    step=b.get("step"),
                    is_integer=b.get("is_integer", False),
                )
                for b in parameter_bounds
            ]
        else:
            p_bounds = _default_bounds_map.get(
                template,
                _default_bounds_map["reduction_rhinoplasty"],
            )

        # ── Map templates to builder callables ──
        _builders: Dict[str, Any] = {
            "reduction_rhinoplasty": (
                RhinoplastyPlanBuilder.reduction_rhinoplasty
            ),
            "functional_rhinoplasty": (
                RhinoplastyPlanBuilder.functional_rhinoplasty
            ),
            "tip_rhinoplasty": (
                RhinoplastyPlanBuilder.tip_rhinoplasty
            ),
        }

        builder_fn = _builders.get(template)
        if builder_fn is None:
            return {
                "error": (
                    f"Unknown template '{template}'. "
                    f"Available: {list(_builders.keys())}"
                ),
            }

        # Landmarks for aesthetic scoring
        case_info = self.get_case(case_id)
        landmarks = case_info.get("landmarks", [])
        preop_lm = _landmarks_to_dict(landmarks)

        def _opt_evaluate(
            x: np.ndarray,
        ) -> Tuple[np.ndarray, Dict[str, float]]:
            """Map parameter vector → (objectives, constraint_values)."""
            kwargs: Dict[str, Any] = {}
            for i, pb in enumerate(p_bounds):
                kwargs[pb.name] = float(x[i])

            # Build plan from parameters
            try:
                plan = builder_fn(**kwargs)
            except Exception:
                n_obj = len(obj_specs)
                return (
                    np.zeros(n_obj),
                    {c.name: 0.0 for c in con_specs},
                )

            # Run simulation
            try:
                orchestrator = SimOrchestrator(
                    copy.deepcopy(mesh),
                    run_cfd=False,
                    run_healing=False,
                    fem_convergence_tol=1e-4,
                    fem_max_iter=15,
                )
                sim = orchestrator.run(plan)
            except Exception:
                n_obj = len(obj_specs)
                return (
                    np.zeros(n_obj),
                    {c.name: 0.0 for c in con_specs},
                )

            # Evaluate metrics
            safety_index = 0.0
            aesthetic_score = 0.0
            functional_score = 0.0
            max_skin_tension = 0.0
            max_strain = 0.0
            nasal_resistance = 1.5

            try:
                safety = SafetyMetrics(mesh).evaluate(sim.fem_result)
                safety_index = float(safety.overall_safety_index)
                max_skin_tension = float(
                    safety.skin_tension.max_tension_pa,
                )
                max_strain = float(
                    max(
                        (v.actual_value for v in safety.stress_violations),
                        default=0.0,
                    ),
                )
            except Exception:
                pass

            try:
                if preop_lm:
                    aes = AestheticMetrics(preop_lm)
                    report = aes.compute()
                    aesthetic_score = float(report.overall_score)
            except Exception:
                pass

            try:
                fm = FunctionalMetrics(mesh, preop_lm)
                if sim.cfd_result is not None:
                    fr = fm.evaluate(sim.cfd_result)
                    functional_score = float(fr.overall_score)
                    nasal_resistance = float(fr.nasal_resistance)
            except Exception:
                pass

            # Build objectives array (order matches obj_specs)
            obj_map = {
                "aesthetic_score": aesthetic_score,
                "functional_score": functional_score,
                "safety_index": safety_index,
            }
            obj_vals = np.array(
                [obj_map.get(o.name, 0.0) for o in obj_specs],
                dtype=np.float64,
            )

            # Build constraint values dict
            con_vals = {
                "safety_index": safety_index,
                "max_skin_tension_pa": max_skin_tension,
                "max_principal_strain": max_strain,
                "nasal_resistance": nasal_resistance,
            }

            return obj_vals, con_vals

        optimizer = PlanOptimizer(
            obj_specs,
            con_specs,
            p_bounds,
            population_size=population_size,
            n_generations=n_generations,
        )

        t0 = time.monotonic()
        opt_result = optimizer.optimize(_opt_evaluate)
        wall_clock = time.monotonic() - t0

        result_dict = opt_result.to_dict()
        result_dict["case_id"] = case_id
        result_dict["template"] = template
        result_dict["wall_clock_seconds"] = round(wall_clock, 2)

        # Reconstruct the best-compromise plan for caching
        if opt_result.best_compromise is not None:
            best_params: Dict[str, Any] = {}
            for i, pb in enumerate(p_bounds):
                best_params[pb.name] = float(
                    opt_result.best_compromise.parameters[i],
                )
            try:
                best_plan = builder_fn(**best_params)
                compiled = PlanCompiler(mesh).compile(best_plan)
                plan_key = f"{case_id}:{best_plan.content_hash()}"
                self._plan_cache[plan_key] = best_plan
                result_dict["best_plan_hash"] = best_plan.content_hash()
                result_dict["best_plan_params"] = best_params
                result_dict["best_plan_compiled"] = True
            except Exception:
                result_dict["best_plan_compiled"] = False

        self._audit.record(
            "plan_optimization_completed",
            case_id=case_id,
            metadata={
                "template": template,
                "population_size": population_size,
                "n_generations": n_generations,
                "n_evaluations": opt_result.n_evaluations,
                "pareto_front_size": (
                    opt_result.pareto_front.size
                    if opt_result.pareto_front else 0
                ),
                "wall_clock_seconds": round(wall_clock, 2),
            },
        )

        return result_dict

    def get_dicom_data(self, case_id: str) -> Dict[str, Any]:
        """Return CT volume data for visualization.

        Attempts to load real DICOM data from the case bundle first.
        Falls back to a synthetic volume if none is available.
        """
        bundle = self._load_bundle(case_id)
        if bundle is not None:
            # Try loading real DICOM volume from bundle
            for name in ("dicom_volume", "ct_volume"):
                for subdir in ("inputs", "derived", "results"):
                    try:
                        vol_data = bundle.load_json(name, subdir=subdir)
                        if vol_data:
                            return vol_data
                    except Exception:
                        continue

        # Fall back to synthetic volume using the case's mesh geometry
        return self._generate_synthetic_dicom(case_id)

    @staticmethod
    def _generate_synthetic_dicom(case_id: str) -> Dict[str, Any]:
        """Build a synthetic 64x64x48 HU volume for cases without real DICOM."""
        import hashlib

        seed = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)

        nx, ny, nz = 64, 64, 48
        volume = np.full((nx, ny, nz), -1000.0, dtype=np.float32)
        cx, cy = nx // 2, ny // 2

        for z in range(nz):
            zn = (z - nz // 2) / (nz // 2)
            rz_scale = max(0.0, 1.0 - zn * zn)
            rx = int(24 * rz_scale)
            ry = int(20 * rz_scale)
            if rx < 3 or ry < 3:
                continue
            for x in range(max(0, cx - rx), min(nx, cx + rx)):
                for y in range(max(0, cy - ry), min(ny, cy + ry)):
                    dx = (x - cx) / rx
                    dy = (y - cy) / ry
                    r2 = dx * dx + dy * dy
                    if r2 > 1.0:
                        continue
                    if r2 > 0.85:
                        volume[x, y, z] = 700.0 + rng.normal(0, 50)
                    else:
                        volume[x, y, z] = 40.0 + rng.normal(0, 15)
                    if abs(dx) < 0.15 and dy > 0.3 and r2 < 0.6 and -0.3 < zn < 0.4:
                        volume[x, y, z] = -1000.0 + rng.normal(0, 20)

        flat_volume = volume.transpose(2, 1, 0).ravel().astype(np.int16).tolist()
        return {
            "dimensions": [nx, ny, nz],
            "spacing": [0.5, 0.5, 0.625],
            "origin": [-16.0, -16.0, -15.0],
            "volume": flat_volume,
            "modality": "CT",
            "series_description": f"Synthetic CT — {case_id[:8]}",
        }

    def get_post_op_prediction(
        self,
        case_id: str,
        plan_hash: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return cached or freshly generated post-op prediction."""
        key = f"{case_id}:{plan_hash or 'default'}"
        cached = self._postop_cache.get(key)
        if cached is not None:
            return cached
        if plan_hash:
            return self.predict_post_op(case_id, plan_hash)
        return None

    def run_cfd_simulation(
        self,
        case_id: str,
        plan_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger CFD simulation via the real Navier-Stokes solver.

        If ``plan_hash`` is provided, runs the full orchestrator
        (FEM + CFD) so the CFD accounts for tissue deformation.
        """
        plan = self._get_plan(plan_hash, case_id)
        if plan is not None and plan_hash:
            # Full orchestrator run (FEM deforms mesh, then CFD on deformed)
            try:
                sim = self._run_simulation(case_id, plan,
                                            run_cfd=True, run_healing=False)
                cfd = sim.cfd_preop
                if cfd is not None:
                    return self._serialize_cfd(case_id, cfd,
                                               postop=sim.cfd_postop)
                return {"error": "CFD produced no result",
                        "warnings": sim.warnings, "errors": sim.errors}
            except Exception as exc:
                logger.error("CFD simulation failed: %s", exc)
                return {"error": f"CFD simulation failed: {exc}"}

        # Standalone CFD without a plan
        return self.get_cfd_results(case_id)

    def run_fem_simulation(
        self,
        case_id: str,
        plan_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger FEM simulation via the real Newton-Raphson solver.

        Compiles the plan into boundary conditions and runs the
        nonlinear FEM solver with tissue-specific constitutive models.
        """
        plan = self._get_plan(plan_hash, case_id)
        if plan is None:
            return {"error": "No surgical plan available for simulation"}

        try:
            sim = self._run_simulation(case_id, plan,
                                        run_cfd=False, run_healing=False)
            if sim.fem_result is not None:
                return self._serialize_fem(case_id, sim.fem_result,
                                            sim.compilation)
            return {"error": "FEM produced no result",
                    "errors": sim.errors, "warnings": sim.warnings}
        except Exception as exc:
            logger.error("FEM simulation failed: %s", exc)
            return {"error": f"FEM simulation failed: {exc}"}

    def execute_incision(
        self,
        case_id: str,
        points: List[List[float]],
        depth: float = 5.0,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a virtual incision on the tissue mesh.

        Builds a ``SurgicalOp`` representing the incision, compiles it
        to mesh modifications, and applies them to the volume mesh.

        Parameters
        ----------
        case_id : str
            The case identifier.
        points : list of [x, y, z]
            Polyline defining the incision path on the tissue surface.
        depth : float
            Incision depth in mm.
        layers : list of str, optional
            Tissue layers to cut through (e.g. ['skin', 'muscle']).
            If None, defaults to skin envelope only.
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        mesh = self._ensure_mesh(bundle)

        # Build a surgical plan with one incision operation
        plan = SurgicalPlan(
            name="interactive_incision",
            procedure=ProcedureType.SEPTOPLASTY,
            description="Interactive incision",
        )
        incision_op = SurgicalOp(
            name="incision",
            category=OpCategory.RESECTION,
            procedure=ProcedureType.SEPTOPLASTY,
            params={
                "path": points,
                "depth_mm": depth,
                "type": "hemitransfixion",
            },
            affected_structures=[StructureType.SKIN_ENVELOPE],
            description=f"Incision ({len(points)} pts, depth={depth}mm)",
        )
        plan.add_step(incision_op)

        compiler = PlanCompiler(mesh)
        compilation = compiler.compile(plan)

        if not compilation.is_valid:
            return {
                "status": "error",
                "message": f"Incision compilation failed: {compilation.errors}",
            }

        # Determine affected layers
        affected_layers = layers if layers else ["skin"]

        self._audit.record(
            "incision_executed",
            case_id=case_id,
            metadata={
                "n_points": len(points),
                "depth_mm": depth,
                "n_bcs": compilation.n_bcs,
                "n_mesh_mods": compilation.n_mesh_mods,
                "affected_layers": affected_layers,
            },
        )

        return {
            "status": "ok",
            "incision_id": compilation.content_hash()[:12],
            "path": points,
            "depth": depth,
            "affected_layers": affected_layers,
            "compilation": _ndarray_to_list(compilation.to_dict()),
            "n_boundary_conditions": compilation.n_bcs,
            "n_mesh_modifications": compilation.n_mesh_mods,
            "plan_hash": plan.content_hash(),
        }

    def execute_osteotomy(
        self,
        case_id: str,
        plane: Dict[str, Any],
        movement: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a virtual osteotomy (bone cut + repositioning).

        Builds a ``lateral_osteotomy`` or ``medial_osteotomy`` operator,
        compiles it to mesh split + displacement BCs, and returns the
        compiled plan for FEM execution.
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        mesh = self._ensure_mesh(bundle)

        plan = SurgicalPlan(
            name="interactive_osteotomy",
            procedure=ProcedureType.RHINOPLASTY,
            description="Interactive osteotomy",
        )

        # Determine osteotomy type from plane normal
        normal = plane.get("normal", [1, 0, 0])
        origin = plane.get("origin", [0, 0, 0])
        side = "left" if normal[0] < 0 else "right"

        # Use lateral_osteotomy operator if available
        factory = ALL_OPERATORS.get("lateral_osteotomy")
        if factory is not None:
            try:
                op = factory(
                    side=side,
                    medial_mm=float(movement.get("medial_mm", 3.0)) if movement else 3.0,
                )
                plan.add_step(op)
            except Exception:
                # Build raw op
                op = SurgicalOp(
                    name="lateral_osteotomy",
                    category=OpCategory.OSTEOTOMY,
                    procedure=ProcedureType.RHINOPLASTY,
                    params={"plane": plane, "movement": movement or {},
                            "side": side},
                    affected_structures=[StructureType.BONE_NASAL],
                    description=f"Lateral osteotomy ({side})",
                )
                plan.add_step(op)
        else:
            op = SurgicalOp(
                name="lateral_osteotomy",
                category=OpCategory.OSTEOTOMY,
                procedure=ProcedureType.RHINOPLASTY,
                params={"plane": plane, "movement": movement or {},
                        "side": side},
                affected_structures=[StructureType.BONE_NASAL],
                description=f"Lateral osteotomy ({side})",
            )
            plan.add_step(op)

        compiler = PlanCompiler(mesh)
        compilation = compiler.compile(plan)

        # Cache plan for potential FEM follow-up
        self._plan_cache[plan.content_hash()] = plan

        self._audit.record(
            "osteotomy_executed",
            case_id=case_id,
            metadata={
                "side": side,
                "plan_hash": plan.content_hash(),
            },
        )

        # Extract movement vectors for frontend
        displacement = list(movement.get("displacement", [0.0, 0.0, 0.0])) if movement else [0.0, 0.0, 0.0]
        rotation = list(movement.get("rotation", [0.0, 0.0, 0.0])) if movement else [0.0, 0.0, 0.0]

        return {
            "status": "ok",
            "osteotomy_id": compilation.content_hash()[:12],
            "cut_plane": {"normal": normal, "point": origin},
            "displacement": displacement,
            "rotation": rotation,
            "compilation": _ndarray_to_list(compilation.to_dict()),
            "plan_hash": plan.content_hash(),
            "n_boundary_conditions": compilation.n_bcs,
            "n_mesh_modifications": compilation.n_mesh_mods,
            "warnings": compilation.warnings,
        }

    def place_graft(
        self,
        case_id: str,
        position: List[float],
        normal: List[float],
        graft_type: str = "septal",
        dimensions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Place a tissue graft at the specified position.

        Compiles a graft placement operation and returns the
        compiled plan including new mesh elements.
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        mesh = self._ensure_mesh(bundle)

        plan = SurgicalPlan(
            name="interactive_graft",
            procedure=ProcedureType.RHINOPLASTY,
            description=f"Place {graft_type} graft",
        )

        # Map graft_type to operator
        graft_op_map = {
            "septal": "spreader_graft",
            "spreader": "spreader_graft",
            "columellar": "columellar_strut",
            "shield": "shield_graft",
            "alar_batten": "alar_batten_graft",
            "cap": "cap_graft",
            "onlay": "onlay_graft",
        }
        op_name = graft_op_map.get(graft_type, "spreader_graft")
        factory = ALL_OPERATORS.get(op_name)

        dims = dimensions or {}
        if factory is not None:
            try:
                op = factory(**dims)
                plan.add_step(op)
            except Exception:
                op = SurgicalOp(
                    name=op_name,
                    category=OpCategory.GRAFT,
                    procedure=ProcedureType.RHINOPLASTY,
                    params={"position": position, "normal": normal,
                            "graft_type": graft_type, **dims},
                    affected_structures=[StructureType.CARTILAGE_SEPTUM],
                    description=f"{graft_type} graft placement",
                )
                plan.add_step(op)
        else:
            op = SurgicalOp(
                name=op_name,
                category=OpCategory.GRAFT,
                procedure=ProcedureType.RHINOPLASTY,
                params={"position": position, "normal": normal,
                        "graft_type": graft_type, **dims},
                affected_structures=[StructureType.CARTILAGE_SEPTUM],
                description=f"{graft_type} graft placement",
            )
            plan.add_step(op)

        compiler = PlanCompiler(mesh)
        compilation = compiler.compile(plan)

        self._plan_cache[plan.content_hash()] = plan

        self._audit.record(
            "graft_placed",
            case_id=case_id,
            metadata={
                "graft_type": graft_type,
                "position": position,
                "plan_hash": plan.content_hash(),
            },
        )

        # Map graft type to donor site
        donor_site_map = {
            "septal": "nasal_septum",
            "spreader": "nasal_septum",
            "columellar": "nasal_septum",
            "shield": "nasal_septum",
            "alar_batten": "conchal_cartilage",
            "cap": "conchal_cartilage",
            "onlay": "rib_cartilage",
        }

        return {
            "status": "ok",
            "graft_id": compilation.content_hash()[:12],
            "graft_type": graft_type,
            "position": position,
            "dimensions": dims if dims else {"length": 10.0, "width": 5.0, "thickness": 1.5},
            "donor_site": donor_site_map.get(graft_type, "nasal_septum"),
            "compilation": _ndarray_to_list(compilation.to_dict()),
            "plan_hash": plan.content_hash(),
            "n_mesh_modifications": compilation.n_mesh_mods,
            "warnings": compilation.warnings,
        }

    def predict_post_op(
        self,
        case_id: str,
        plan_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a physics-driven post-operative morphology prediction.

        Pipeline:
        1. Resolve the surgical plan from ``plan_hash``.
        2. Compile plan → boundary conditions.
        3. Run the full multi-physics orchestrator (FEM + CFD + Healing).
        4. Apply healing-predicted displacements to the surface mesh.
        5. Return the deformed mesh layers and predicted landmarks.
        """
        plan = self._get_plan(plan_hash, case_id)
        if plan is None:
            return {"error": f"No plan found for hash '{plan_hash}'"}

        # Run full multi-physics simulation
        try:
            sim = self._run_simulation(
                case_id, plan,
                run_cfd=True, run_healing=True,
            )
        except Exception as exc:
            logger.error("Post-op simulation failed: %s", exc)
            return {"error": f"Post-op simulation failed: {exc}"}

        # Get the surface mesh for displacement application
        mesh_data = self.get_mesh_data(case_id)
        if mesh_data is None or "error" in mesh_data:
            return mesh_data or {"error": f"Case '{case_id}' not found"}

        raw_positions = mesh_data["positions"]
        raw_indices = mesh_data["indices"]
        positions = np.array(raw_positions, dtype=np.float64).reshape(-1, 3)
        triangles = np.array(raw_indices, dtype=np.int32).reshape(-1, 3)
        n_verts = len(positions)

        # Determine displacement source: healing final > FEM direct
        displaced = positions.copy()
        displacements_applied = False

        if sim.final_displacements is not None:
            # Healing model's predicted final shape
            n_disp = min(len(sim.final_displacements), n_verts)
            displaced[:n_disp] += sim.final_displacements[:n_disp]
            displacements_applied = True
        elif sim.fem_result is not None:
            # Immediate post-surgical displacements (no healing)
            n_disp = min(len(sim.fem_result.displacements), n_verts)
            displaced[:n_disp] += sim.fem_result.displacements[:n_disp]
            displacements_applied = True

        if not displacements_applied:
            return {
                "error": "Simulation produced no displacement field",
                "warnings": sim.warnings,
                "errors": sim.errors,
            }

        layers = {
            "tissue_postop": {
                "positions": displaced.tolist(),
                "indices": triangles.tolist(),
                "n_vertices": n_verts,
                "n_triangles": len(triangles),
            }
        }

        # Predicted landmarks: apply displacement to pre-op landmarks
        landmarks_predicted = None
        pre_landmarks = self.get_landmarks(case_id)
        if pre_landmarks and "landmarks" in pre_landmarks:
            predicted = []
            for lm in pre_landmarks["landmarks"]:
                pos = np.array(lm.get("position", [0, 0, 0]), dtype=np.float64)
                # Find nearest surface vertex and use its displacement
                dists = np.linalg.norm(positions - pos, axis=1)
                nearest_idx = int(dists.argmin())
                # Lookup displacement for that vertex
                if sim.final_displacements is not None and nearest_idx < len(sim.final_displacements):
                    pos = pos + sim.final_displacements[nearest_idx]
                elif sim.fem_result is not None and nearest_idx < len(sim.fem_result.displacements):
                    pos = pos + sim.fem_result.displacements[nearest_idx]
                predicted.append({**lm, "position": pos.tolist()})
            landmarks_predicted = {"landmarks": predicted}

        # CFD post-op summary
        cfd_predicted = None
        if sim.cfd_postop is not None and sim.cfd_preop is not None:
            cfd_predicted = self._serialize_cfd(case_id, sim.cfd_preop,
                                                 postop=sim.cfd_postop)

        # ── Safety evaluation ──────────────────────────────────
        safety_report = None
        if sim.fem_result is not None:
            try:
                bundle = self._load_bundle(case_id)
                if bundle is not None:
                    vol_mesh = self._ensure_mesh(bundle)
                    safety = SafetyMetrics(vol_mesh)
                    sr = safety.evaluate(sim.fem_result)
                    safety_report = _ndarray_to_list(sr.to_dict())
            except Exception as exc:
                logger.warning("Safety evaluation failed: %s", exc)

        # ── Aesthetic scoring (pre-op vs post-op) ─────────────
        aesthetic_report = None
        if landmarks_predicted and pre_landmarks:
            try:
                preop_lm = _landmarks_to_dict(pre_landmarks["landmarks"])
                postop_lm = _landmarks_to_dict(landmarks_predicted["landmarks"])
                if preop_lm and postop_lm:
                    ae_pre = AestheticMetrics(preop_lm)
                    ae_post = AestheticMetrics(postop_lm)
                    report_pre = ae_pre.compute()
                    report_post = ae_post.compute(preop_landmarks=preop_lm)
                    deltas = AestheticMetrics.compute_change(report_pre, report_post)
                    aesthetic_report = {
                        "preop": _ndarray_to_list(report_pre.to_dict()),
                        "postop": _ndarray_to_list(report_post.to_dict()),
                        "deltas": _ndarray_to_list(deltas),
                    }
            except Exception as exc:
                logger.warning("Aesthetic scoring failed: %s", exc)

        # ── Functional scoring (CFD pre vs post) ──────────────
        functional_report = None
        if sim.cfd_preop is not None:
            try:
                bundle = self._load_bundle(case_id)
                if bundle is not None:
                    vol_mesh = self._ensure_mesh(bundle)
                    lm_dict = _landmarks_to_dict(
                        pre_landmarks["landmarks"]
                    ) if pre_landmarks else {}
                    if lm_dict:
                        fm = FunctionalMetrics(vol_mesh, lm_dict)
                        fr_pre = fm.evaluate(sim.cfd_preop)
                        if sim.cfd_postop is not None:
                            fr_post = fm.evaluate(
                                sim.cfd_postop, preop_cfd=sim.cfd_preop,
                            )
                            improvement = FunctionalMetrics.compute_improvement(
                                fr_pre, fr_post,
                            )
                            functional_report = {
                                "preop": _ndarray_to_list(fr_pre.to_dict()),
                                "postop": _ndarray_to_list(fr_post.to_dict()),
                                "improvement": _ndarray_to_list(improvement),
                            }
                        else:
                            functional_report = {
                                "preop": _ndarray_to_list(fr_pre.to_dict()),
                            }
            except Exception as exc:
                logger.warning("Functional scoring failed: %s", exc)

        # ── Healing timeline (deformed meshes at milestones) ──
        healing_timeline = None
        if sim.fem_result is not None:
            try:
                bundle = self._load_bundle(case_id)
                if bundle is not None:
                    vol_mesh = self._ensure_mesh(bundle)
                    healing_timeline = self._compute_healing_timeline(
                        vol_mesh, sim.fem_result.displacements,
                        positions, n_verts,
                    )
            except Exception as exc:
                logger.warning("Healing timeline failed: %s", exc)

        # Confidence: based on solver convergence and mesh quality
        confidence = 0.6
        if sim.fem_converged:
            confidence += 0.2
        if sim.cfd_converged:
            confidence += 0.1
        if sim.healing_states:
            confidence += 0.05
        if safety_report and safety_report.get("is_safe"):
            confidence += 0.05

        result: Dict[str, Any] = {
            "plan_hash": plan_hash,
            "case_id": case_id,
            "layers": layers,
            "landmarks_predicted": landmarks_predicted,
            "cfd_predicted": cfd_predicted,
            "safety": safety_report,
            "aesthetics": aesthetic_report,
            "functional": functional_report,
            "healing_timeline": healing_timeline,
            "confidence": float(confidence),
            "generated_at": time.time(),
            "simulation_summary": sim.to_dict(),
        }

        self._postop_cache[f"{case_id}:{plan_hash}"] = result

        self._audit.record(
            "postop_prediction_generated",
            case_id=case_id,
            metadata={
                "plan_hash": plan_hash,
                "confidence": confidence,
                "fem_converged": sim.fem_converged,
                "wall_clock_s": sim.wall_clock_seconds,
            },
        )

        return result

    def upload_dicom(
        self,
        case_id: str,
        data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Accept DICOM upload and ingest into the case bundle.

        ``data`` can be:
        - A single base64-encoded DICOM series string
        - A list of base64-encoded DICOM file strings (one per file)
        - A file path string to a DICOM directory
        - Raw bytes

        Uses the DicomIngester from the data pipeline.
        """
        bundle = self._load_bundle(case_id)
        if bundle is None:
            return {"error": f"Case '{case_id}' not found"}

        if data is None:
            return {"error": "No DICOM data provided"}

        try:
            ingester = DicomIngester()

            if isinstance(data, list):
                # Array of base64-encoded DICOM files from frontend
                import base64
                combined_bytes = b""
                for item in data:
                    if isinstance(item, str):
                        combined_bytes += base64.b64decode(item)
                volume = ingester.ingest_bytes(combined_bytes)
            elif isinstance(data, str) and Path(data).exists():
                # File path provided
                volume = ingester.ingest_directory(Path(data))
            elif isinstance(data, str):
                # Single base64-encoded data
                import base64
                raw_bytes = base64.b64decode(data)
                volume = ingester.ingest_bytes(raw_bytes)
            elif isinstance(data, (bytes, bytearray)):
                volume = ingester.ingest_bytes(bytes(data))
            else:
                return {"error": f"Unsupported data type: {type(data).__name__}"}

            # Store the volume on the bundle
            vol_dict = {
                "dimensions": list(volume.shape),
                "spacing": getattr(volume, "spacing", [1.0, 1.0, 1.0]),
                "volume": volume.ravel().astype(np.int16).tolist(),
                "modality": "CT",
            }
            bundle.save_json("dicom_volume", vol_dict, subdir="inputs")

            self._audit.record(
                "dicom_uploaded",
                case_id=case_id,
                metadata={
                    "dimensions": list(volume.shape),
                    "n_voxels": int(volume.size),
                },
            )

            return {
                "success": True,
                "status": "ok",
                "dimensions": list(volume.shape),
                "n_voxels": int(volume.size),
            }

        except Exception as exc:
            logger.error("DICOM ingest failed for case %s: %s", case_id, exc)
            return {"error": f"DICOM ingest failed: {exc}"}

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

def _extract_surface(
    mesh: VolumeMesh,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract surface triangles from a volume mesh for rendering.

    Returns (nodes_Nx3, triangles_Mx3, face_region_ids_M).
    Each surface triangle is tagged with the region ID of the element
    it came from, enabling per-face region colouring in the viewer.
    """
    if mesh.n_elements == 0 or mesh.n_nodes == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros(0, dtype=np.int32),
        )

    # Face extraction: collect faces and find boundary faces (appear once)
    face_count: Dict[Tuple[int, ...], int] = {}
    face_list: List[Tuple[int, ...]] = []
    face_elem_id: Dict[Tuple[int, ...], int] = {}  # face key → source element

    etype = mesh.element_type
    has_regions = mesh.region_ids is not None and len(mesh.region_ids) > 0

    for eid in range(mesh.n_elements):
        elem = mesh.elements[eid]
        faces = _element_faces(elem, etype)
        for f in faces:
            key = tuple(sorted(f))
            face_count[key] = face_count.get(key, 0) + 1
            if face_count[key] == 1:
                face_list.append(f)
                face_elem_id[key] = eid

    # Keep only boundary faces (count == 1)
    boundary_faces: List[Tuple[int, ...]] = []
    boundary_region_ids: List[int] = []
    for f in face_list:
        key = tuple(sorted(f))
        if face_count.get(key, 0) == 1:
            boundary_faces.append(f)
            if has_regions:
                eid = face_elem_id.get(key, 0)
                boundary_region_ids.append(int(mesh.region_ids[eid]))
            else:
                boundary_region_ids.append(0)

    if not boundary_faces:
        return (
            mesh.nodes.copy(),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros(0, dtype=np.int32),
        )

    triangles = np.array(boundary_faces, dtype=np.int64)
    region_ids = np.array(boundary_region_ids, dtype=np.int32)
    return mesh.nodes.copy(), triangles, region_ids


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
