"""Multi-physics simulation orchestrator.

Coordinates the execution of all simulation components:
  1. Plan compilation → boundary conditions
  2. Mesh modification (element removal, graft insertion)
  3. Cartilage scoring / bending pre-processing
  4. Suture element setup
  5. FEM soft tissue analysis
  6. CFD nasal airway analysis
  7. Healing timeline prediction
  8. Result aggregation and provenance

The orchestrator enforces execution order and manages data flow
between the physics solvers while maintaining determinism.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.provenance import Provenance, hash_dict
from ..core.types import (
    MeshElementType,
    SolverType,
    StructureType,
    Vec3,
    VolumeMesh,
)
from ..plan.compiler import (
    BCType,
    CompilationResult,
    MaterialModification,
    MeshModification,
    PlanCompiler,
)
from ..plan.dsl import SurgicalPlan
from .cartilage import CartilageSolver, ScoreLine, CARTILAGE_LIBRARY
from .cfd_airway import AirwayCFDResult, AirwayCFDSolver, extract_airway_geometry
from .fem_soft_tissue import FEMResult, SoftTissueFEM
from .healing import HealingModel, HealingState
from .sutures import SutureSystem

logger = logging.getLogger(__name__)


# ── Simulation result ─────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Complete result of a multi-physics simulation run."""
    # FEM results
    fem_result: Optional[FEMResult] = None

    # CFD results
    cfd_preop: Optional[AirwayCFDResult] = None
    cfd_postop: Optional[AirwayCFDResult] = None

    # Healing
    healing_states: List[HealingState] = field(default_factory=list)
    final_displacements: Optional[np.ndarray] = None

    # Compilation
    compilation: Optional[CompilationResult] = None

    # Provenance
    run_hash: str = ""
    plan_hash: str = ""
    mesh_hash: str = ""
    wall_clock_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def fem_converged(self) -> bool:
        return self.fem_result is not None and self.fem_result.converged

    @property
    def cfd_converged(self) -> bool:
        if self.cfd_postop is None:
            return True  # no CFD requested
        return self.cfd_postop.converged

    def summary(self) -> str:
        lines = [
            f"SimulationResult (hash={self.run_hash[:12]})",
            f"  Plan: {self.plan_hash[:12]}",
            f"  Mesh: {self.mesh_hash[:12]}",
        ]
        if self.fem_result:
            lines.append(f"  FEM: {self.fem_result.summary()}")
        if self.cfd_preop:
            lines.append(f"  CFD pre-op: R={self.cfd_preop.nasal_resistance_pa_s_ml:.3f}")
        if self.cfd_postop:
            lines.append(f"  CFD post-op: R={self.cfd_postop.nasal_resistance_pa_s_ml:.3f}")
        if self.healing_states:
            final = self.healing_states[-1]
            lines.append(f"  Healing: {final.summary()}")
        lines.append(f"  Time: {self.wall_clock_seconds:.1f}s")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
        if self.errors:
            lines.append(f"  ERRORS: {len(self.errors)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "run_hash": self.run_hash,
            "plan_hash": self.plan_hash,
            "mesh_hash": self.mesh_hash,
            "wall_clock_seconds": self.wall_clock_seconds,
            "warnings": self.warnings,
            "errors": self.errors,
        }
        if self.fem_result is not None:
            result["fem"] = {
                "converged": self.fem_result.converged,
                "max_displacement_mm": self.fem_result.max_displacement_mm,
                "max_von_mises_stress": self.fem_result.max_von_mises_stress,
                "max_principal_strain": self.fem_result.max_principal_strain,
                "n_iterations": self.fem_result.n_iterations,
                "internal_energy": self.fem_result.internal_energy,
            }
        if self.cfd_preop is not None:
            result["cfd_preop"] = {
                "resistance": self.cfd_preop.nasal_resistance_pa_s_ml,
                "flow_rate": self.cfd_preop.total_flow_rate_ml_s,
                "pressure_drop": self.cfd_preop.pressure_drop_pa,
                "reynolds": self.cfd_preop.reynolds_number,
            }
        if self.cfd_postop is not None:
            result["cfd_postop"] = {
                "resistance": self.cfd_postop.nasal_resistance_pa_s_ml,
                "flow_rate": self.cfd_postop.total_flow_rate_ml_s,
                "pressure_drop": self.cfd_postop.pressure_drop_pa,
                "reynolds": self.cfd_postop.reynolds_number,
            }
        if self.healing_states:
            result["healing"] = [
                {
                    "time_days": s.time_days,
                    "phase": s.phase_name,
                    "mean_edema_pct": s.mean_edema_pct,
                    "mean_scar_pct": s.mean_scar_pct,
                    "max_settling_mm": s.max_settling_mm,
                }
                for s in self.healing_states
            ]
        return result


# ── Orchestrator ──────────────────────────────────────────────────

class SimOrchestrator:
    """Orchestrate multi-physics simulation for a surgical plan.

    Execution pipeline:
        1. Compile plan → BCs
        2. Apply mesh modifications (graft insertion, element removal)
        3. Configure cartilage mechanics
        4. Set up suture system
        5. Run FEM (structural deformation)
        6. Extract post-op airway geometry
        7. Run CFD (pre-op and post-op)
        8. Predict healing timeline
        9. Aggregate and hash results
    """

    def __init__(
        self,
        mesh: VolumeMesh,
        *,
        run_cfd: bool = True,
        run_healing: bool = True,
        healing_time_points: Optional[List[float]] = None,
        fem_convergence_tol: float = 1e-6,
        fem_max_iter: int = 25,
        cfd_resolution: int = 20,
        cfd_max_iter: int = 500,
        provenance: Optional[Provenance] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self._mesh = mesh
        self._run_cfd = run_cfd
        self._run_healing = run_healing
        self._healing_times = healing_time_points or [1, 7, 14, 30, 90, 180, 365]
        self._fem_tol = fem_convergence_tol
        self._fem_max_iter = fem_max_iter
        self._cfd_res = cfd_resolution
        self._cfd_max_iter = cfd_max_iter
        self._provenance = provenance
        self._progress = progress_callback

    def _report_progress(self, stage: str, fraction: float) -> None:
        if self._progress is not None:
            self._progress(stage, fraction)

    def run(self, plan: SurgicalPlan) -> SimulationResult:
        """Execute the full simulation pipeline."""
        t0 = time.monotonic()
        result = SimulationResult()

        self._report_progress("compiling", 0.0)

        # ── Stage 1: Compile plan ─────────────────────────────
        logger.info("Stage 1/8: Compiling plan '%s'", plan.name)
        compiler = PlanCompiler(self._mesh)
        compilation = compiler.compile(plan)
        result.compilation = compilation
        result.plan_hash = compilation.plan_hash
        result.mesh_hash = compilation.mesh_hash

        if not compilation.is_valid:
            result.errors.extend(compilation.errors)
            result.wall_clock_seconds = time.monotonic() - t0
            return result

        result.warnings.extend(compilation.warnings)
        self._report_progress("compiling", 1.0)

        # ── Stage 2: Apply mesh modifications ─────────────────
        logger.info("Stage 2/8: Applying mesh modifications")
        self._report_progress("mesh_mods", 0.0)
        working_mesh = self._apply_mesh_modifications(
            self._mesh, compilation.mesh_modifications, result,
        )
        self._report_progress("mesh_mods", 1.0)

        # ── Stage 3: Configure cartilage mechanics ────────────
        logger.info("Stage 3/8: Configuring cartilage mechanics")
        self._report_progress("cartilage", 0.0)
        cartilage_solver = CartilageSolver(working_mesh)
        self._configure_cartilage(cartilage_solver, compilation, result)
        self._report_progress("cartilage", 1.0)

        # ── Stage 4: Set up suture system ─────────────────────
        logger.info("Stage 4/8: Setting up sutures")
        self._report_progress("sutures", 0.0)
        suture_system = SutureSystem()
        self._configure_sutures(suture_system, compilation, working_mesh, result)
        self._report_progress("sutures", 1.0)

        # ── Stage 5: Run FEM ──────────────────────────────────
        logger.info("Stage 5/8: Running FEM analysis")
        self._report_progress("fem", 0.0)
        try:
            fem = SoftTissueFEM(
                working_mesh,
                convergence_tol=self._fem_tol,
                max_newton_iter=self._fem_max_iter,
            )

            # Apply cartilage material map
            cart_map = cartilage_solver.compute_material_map()
            for eid, params in cart_map.items():
                if 0 <= eid < working_mesh.n_elements:
                    fem._elem_materials[eid] = (
                        fem._elem_materials[eid][0],
                        params,
                    )

            fem_result = fem.solve(
                compilation,
                progress_callback=lambda step, total, res: self._report_progress(
                    "fem", float(step) / max(total, 1)
                ),
            )
            result.fem_result = fem_result

            if not fem_result.converged:
                result.warnings.append("FEM did not fully converge")

        except Exception as exc:
            result.errors.append(f"FEM failed: {exc}")
            logger.error("FEM failed: %s", exc, exc_info=True)

        self._report_progress("fem", 1.0)

        # ── Stage 6: CFD pre-op (baseline) ────────────────────
        if self._run_cfd:
            logger.info("Stage 6/8: Running pre-op CFD")
            self._report_progress("cfd_preop", 0.0)
            try:
                preop_geometry = extract_airway_geometry(self._mesh)
                cfd_solver = AirwayCFDSolver(
                    nx=self._cfd_res,
                    ny=self._cfd_res,
                    nz=self._cfd_res * 4,
                    max_iter=self._cfd_max_iter,
                )
                result.cfd_preop = cfd_solver.solve(preop_geometry)
            except Exception as exc:
                result.warnings.append(f"Pre-op CFD failed: {exc}")
                logger.warning("Pre-op CFD failed: %s", exc)
            self._report_progress("cfd_preop", 1.0)

        # ── Stage 7: CFD post-op ──────────────────────────────
        if self._run_cfd and result.fem_result is not None:
            logger.info("Stage 7/8: Running post-op CFD")
            self._report_progress("cfd_postop", 0.0)
            try:
                # Create deformed mesh for post-op airway
                deformed_mesh = self._create_deformed_mesh(
                    working_mesh, result.fem_result.displacements,
                )
                postop_geometry = extract_airway_geometry(deformed_mesh)
                cfd_solver = AirwayCFDSolver(
                    nx=self._cfd_res,
                    ny=self._cfd_res,
                    nz=self._cfd_res * 4,
                    max_iter=self._cfd_max_iter,
                )
                result.cfd_postop = cfd_solver.solve(postop_geometry)
            except Exception as exc:
                result.warnings.append(f"Post-op CFD failed: {exc}")
                logger.warning("Post-op CFD failed: %s", exc)
            self._report_progress("cfd_postop", 1.0)

        # ── Stage 8: Healing prediction ───────────────────────
        if self._run_healing and result.fem_result is not None:
            logger.info("Stage 8/8: Running healing prediction")
            self._report_progress("healing", 0.0)
            try:
                healing = HealingModel(working_mesh)
                result.healing_states = healing.compute_timeline(
                    self._healing_times,
                    surgical_displacements=result.fem_result.displacements,
                )

                # Predict final shape
                final_disps, final_state = healing.predict_final_shape(
                    result.fem_result.displacements,
                )
                result.final_displacements = final_disps

            except Exception as exc:
                result.warnings.append(f"Healing prediction failed: {exc}")
                logger.warning("Healing prediction failed: %s", exc)
            self._report_progress("healing", 1.0)

        # ── Finalize ──────────────────────────────────────────
        result.wall_clock_seconds = time.monotonic() - t0
        result.run_hash = hash_dict(result.to_dict())

        # Record provenance
        if self._provenance is not None:
            self._provenance.begin_step("simulation")
            self._provenance.record_dict(
                "simulation_result",
                {
                    "inputs": {"plan_hash": result.plan_hash, "mesh_hash": result.mesh_hash},
                    "outputs": {"run_hash": result.run_hash},
                    "metadata": result.to_dict(),
                },
                "simulation",
            )
            self._provenance.end_step()

        logger.info("Simulation complete:\n%s", result.summary())
        return result

    def _apply_mesh_modifications(
        self,
        mesh: VolumeMesh,
        modifications: List[MeshModification],
        result: SimulationResult,
    ) -> VolumeMesh:
        """Apply mesh modifications (element removal, graft insertion).

        Creates a new VolumeMesh with modifications applied.
        Element removal sets region_ids to -1 (inactive).
        Element insertion appends new elements.
        """
        if not modifications:
            return mesh

        # Copy arrays for modification
        nodes = mesh.nodes.copy()
        elements = mesh.elements.copy()
        region_ids = mesh.region_ids.copy()
        region_materials = dict(mesh.region_materials)
        surface_tags = dict(mesh.surface_tags)

        new_nodes_list: List[np.ndarray] = [nodes]
        new_elems_list: List[np.ndarray] = [elements]
        new_region_list: List[np.ndarray] = [region_ids]

        node_offset = nodes.shape[0]

        for mod in modifications:
            if mod.mod_type == "remove":
                # Mark elements as inactive (region = -1)
                for eid in mod.element_ids:
                    if 0 <= eid < len(region_ids):
                        region_ids[eid] = -1
                logger.info("Removed %d elements: %s", len(mod.element_ids), mod.description)

            elif mod.mod_type == "insert":
                if mod.new_nodes is not None and mod.new_elements is not None:
                    # Offset element connectivity
                    offset_elems = mod.new_elements + node_offset
                    new_nodes_list.append(mod.new_nodes)
                    new_elems_list.append(offset_elems)
                    new_regions = np.full(len(offset_elems), mod.new_region_id, dtype=np.int32)
                    new_region_list.append(new_regions)
                    node_offset += mod.new_nodes.shape[0]
                    logger.info(
                        "Inserted %d nodes, %d elements (region %d): %s",
                        mod.new_nodes.shape[0], len(offset_elems),
                        mod.new_region_id, mod.description,
                    )

            elif mod.mod_type == "split":
                # Element splitting (osteotomy): duplicate nodes at cut
                # For now, mark cut-adjacent elements for stiffness reduction
                for eid in mod.element_ids:
                    if 0 <= eid < len(region_ids):
                        pass  # Splitting is handled by the FEM solver via contact
                logger.info("Split %d elements: %s", len(mod.element_ids), mod.description)

            elif mod.mod_type == "refine":
                logger.info("Refinement (mesh adapt) not yet applied: %s", mod.description)
                result.warnings.append(f"Mesh refinement deferred: {mod.description}")

        # Rebuild mesh
        if len(new_nodes_list) > 1 or np.any(region_ids == -1):
            all_nodes = np.vstack(new_nodes_list)
            all_elems = np.vstack(new_elems_list) if len(new_elems_list) > 1 else elements
            all_regions = np.concatenate(new_region_list) if len(new_region_list) > 1 else region_ids

            # Filter out removed elements (region == -1)
            active = all_regions != -1
            all_elems = all_elems[active]
            all_regions = all_regions[active]

            return VolumeMesh(
                nodes=all_nodes,
                elements=all_elems,
                element_type=mesh.element_type,
                region_ids=all_regions,
                region_materials=region_materials,
                surface_tags=surface_tags,
            )

        return mesh

    def _configure_cartilage(
        self,
        solver: CartilageSolver,
        compilation: CompilationResult,
        result: SimulationResult,
    ) -> None:
        """Configure cartilage mechanics from compilation."""
        # Extract scoring operations from material modifications
        for mod in compilation.material_modifications:
            if "Score line" in mod.description:
                # Parse score line info from description
                # Format: "Score line N/M at y=XXmm"
                try:
                    parts = mod.description.split("at y=")
                    if len(parts) == 2:
                        y_pos = float(parts[1].replace("mm", ""))
                        for rid, props in self._mesh.region_materials.items():
                            if props.structure_type in CARTILAGE_LIBRARY:
                                score = ScoreLine(
                                    position_y=y_pos,
                                    depth_fraction=0.5,
                                )
                                solver.apply_scoring(props.structure_type, [score])
                                break
                except (ValueError, IndexError):
                    pass

    def _configure_sutures(
        self,
        suture_system: SutureSystem,
        compilation: CompilationResult,
        mesh: VolumeMesh,
        result: SimulationResult,
    ) -> None:
        """Configure suture system from compilation BCs."""
        for bc in compilation.boundary_conditions:
            if bc.bc_type == BCType.NODAL_FORCE and "suture" in bc.metadata.get("suture", ""):
                # Create suture from force BCs
                technique = bc.metadata.get("suture", "generic")
                logger.info("Configured suture: %s", technique)

    def _create_deformed_mesh(
        self,
        mesh: VolumeMesh,
        displacements: np.ndarray,
    ) -> VolumeMesh:
        """Create a deformed copy of the mesh for post-op CFD."""
        deformed_nodes = mesh.nodes.copy()
        # Displacements may have more/fewer nodes after mesh mods
        n = min(len(displacements), len(deformed_nodes))
        deformed_nodes[:n] += displacements[:n]

        return VolumeMesh(
            nodes=deformed_nodes,
            elements=mesh.elements.copy(),
            element_type=mesh.element_type,
            region_ids=mesh.region_ids.copy(),
            region_materials=mesh.region_materials,
            surface_tags=mesh.surface_tags,
        )
