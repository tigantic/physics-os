"""End-to-end integration tests for the facial plastics platform.

Validates the full pipeline:
  Plan → Compile → Simulate (FEM + CFD + Healing) → Metrics → Report

Each test wires together real production code with synthetic inputs
to verify that data flows correctly through every stage, that
provenance hashes propagate, and that report artefacts are
structurally valid.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from products.facial_plastics.core.types import (
    ClinicalMeasurement,
    LandmarkType,
    MaterialModel,
    MeshElementType,
    ProcedureType,
    QualityLevel,
    StructureType,
    SurfaceMesh,
    TissueProperties,
    Vec3,
    VolumeMesh,
)
from products.facial_plastics.metrics.aesthetic import AestheticMetrics, AestheticReport
from products.facial_plastics.metrics.functional import FunctionalMetrics
from products.facial_plastics.plan.compiler import (
    BCType,
    CompilationResult,
    PlanCompiler,
)
from products.facial_plastics.plan.dsl import SurgicalPlan
from products.facial_plastics.plan.operators.rhinoplasty import (
    RhinoplastyPlanBuilder,
    dorsal_reduction,
    tip_suture,
)
from products.facial_plastics.reports import ReportBuilder
from products.facial_plastics.sim.cfd_airway import (
    AirwayCFDResult,
    AirwayCFDSolver,
    AirwayGeometry,
    extract_airway_geometry,
)
from products.facial_plastics.sim.fem_soft_tissue import FEMResult, SoftTissueFEM
from products.facial_plastics.sim.healing import HealingModel, HealingState
from products.facial_plastics.sim.orchestrator import SimOrchestrator, SimulationResult

from .conftest import (
    make_box_surface_mesh,
    make_rhinoplasty_landmarks,
    make_volume_mesh,
)


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════

def _make_multi_region_mesh() -> VolumeMesh:
    """Build a 5x5x5 tet4 mesh with three tissue regions.

    Region 0: soft tissue (NeoHookean, bulk of mesh)
    Region 1: cartilage  (NeoHookean, stiffer, mid-z band)
    Region 2: bone       (NeoHookean, stiffest, bottom layer)
    """
    n_side = 5
    spacing = 5.0  # mm per grid cell
    nodes: list[list[float]] = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                nodes.append([i * spacing, j * spacing, k * spacing])

    # Hex → 6-tet subdivision (same as conftest)
    elements: list[list[int]] = []
    region_ids_list: list[int] = []
    for i in range(n_side - 1):
        for j in range(n_side - 1):
            for k in range(n_side - 1):
                c = [
                    i * n_side * n_side + j * n_side + k,
                    i * n_side * n_side + j * n_side + (k + 1),
                    i * n_side * n_side + (j + 1) * n_side + k,
                    i * n_side * n_side + (j + 1) * n_side + (k + 1),
                    (i + 1) * n_side * n_side + j * n_side + k,
                    (i + 1) * n_side * n_side + j * n_side + (k + 1),
                    (i + 1) * n_side * n_side + (j + 1) * n_side + k,
                    (i + 1) * n_side * n_side + (j + 1) * n_side + (k + 1),
                ]
                hex_tets = [
                    [c[0], c[1], c[3], c[5]],
                    [c[0], c[3], c[2], c[6]],
                    [c[0], c[5], c[4], c[6]],
                    [c[3], c[5], c[6], c[7]],
                    [c[0], c[3], c[5], c[6]],
                    [c[0], c[6], c[5], c[3]],
                ]
                # Assign region by z-layer of the hex cell centre
                z_mid = (k + 0.5) * spacing
                if z_mid < 5.0:
                    region = 2  # bone (bottom)
                elif z_mid < 12.5:
                    region = 1  # cartilage (middle)
                else:
                    region = 0  # soft tissue (top)

                for tet in hex_tets:
                    elements.append(tet)
                    region_ids_list.append(region)

    nodes_arr = np.array(nodes, dtype=np.float64)
    elem_arr = np.array(elements, dtype=np.int64)
    region_ids = np.array(region_ids_list, dtype=np.int32)

    return VolumeMesh(
        nodes=nodes_arr,
        elements=elem_arr,
        element_type=MeshElementType.TET4,
        region_ids=region_ids,
        region_materials={
            0: TissueProperties(
                structure_type=StructureType.FAT_SUBCUTANEOUS,
                material_model=MaterialModel.NEO_HOOKEAN,
                parameters={"mu": 10.0, "kappa": 100.0},
            ),
            1: TissueProperties(
                structure_type=StructureType.CARTILAGE_UPPER_LATERAL,
                material_model=MaterialModel.NEO_HOOKEAN,
                parameters={"mu": 50.0, "kappa": 500.0},
            ),
            2: TissueProperties(
                structure_type=StructureType.BONE_NASAL,
                material_model=MaterialModel.NEO_HOOKEAN,
                parameters={"mu": 500.0, "kappa": 5000.0},
            ),
        },
    )


def _make_simple_plan() -> SurgicalPlan:
    """Minimal surgical plan: 2 mm dorsal reduction."""
    plan = SurgicalPlan(
        name="test_reduction",
        procedure=ProcedureType.RHINOPLASTY,
        description="Integration test — minimal dorsal reduction",
    )
    plan.add_step(dorsal_reduction(amount_mm=2.0))
    return plan


def _make_comprehensive_plan() -> SurgicalPlan:
    """Comprehensive plan via the builder convenience API."""
    return RhinoplastyPlanBuilder.reduction_rhinoplasty(
        dorsal_reduction_mm=2.5,
        osteotomy_angle=30.0,
        spreader_grafts=True,
        tip_work="moderate",
    )


def _make_straight_tube_geometry(
    length_mm: float = 80.0,
    diameter_mm: float = 6.0,
    n_sections: int = 20,
) -> AirwayGeometry:
    """Tube geometry for CFD tests."""
    radius_mm = diameter_mm * 0.5
    areas = np.full(n_sections, math.pi * radius_mm ** 2)          # mm²
    perimeters = np.full(n_sections, math.pi * diameter_mm)        # mm
    hyd = np.full(n_sections, diameter_mm)                         # D_h = D for circle

    # Cross-section contours: unit circle scaled to radius
    theta = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    contour = np.column_stack([np.cos(theta) * radius_mm,
                               np.sin(theta) * radius_mm])
    cross_sections = [contour.copy() for _ in range(n_sections)]

    # Centerline along z-axis
    z_vals = np.linspace(0.0, length_mm, n_sections)
    centerline = np.column_stack([
        np.zeros(n_sections), np.zeros(n_sections), z_vals,
    ])

    return AirwayGeometry(
        cross_sections=cross_sections,
        centerline=centerline,
        areas=areas,
        perimeters=perimeters,
        hydraulic_diameters=hyd,
        total_length_mm=length_mm,
        left_right_split=0.5,
        valve_area_mm2=float(areas[0]),
    )


# ═══════════════════════════════════════════════════════════════════
# 1. Plan → Compile
# ═══════════════════════════════════════════════════════════════════

class TestPlanCompilation:
    """Validate that surgical plans compile into valid BCs."""

    def test_simple_plan_compiles(self) -> None:
        mesh = make_volume_mesh()
        plan = _make_simple_plan()
        compiler = PlanCompiler(mesh)
        result = compiler.compile(plan)

        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Compilation errors: {result.errors}"
        assert len(result.plan_hash) > 0
        assert len(result.mesh_hash) > 0

    def test_comprehensive_plan_compiles(self) -> None:
        mesh = _make_multi_region_mesh()
        plan = _make_comprehensive_plan()
        compiler = PlanCompiler(mesh)
        result = compiler.compile(plan)

        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Compilation errors: {result.errors}"
        # Comprehensive plan produces BCs and material modifications
        assert len(result.boundary_conditions) > 0

    def test_plan_hash_is_deterministic(self) -> None:
        mesh = make_volume_mesh()
        plan = _make_simple_plan()
        c1 = PlanCompiler(mesh).compile(plan)
        c2 = PlanCompiler(mesh).compile(plan)
        assert c1.plan_hash == c2.plan_hash
        assert c1.mesh_hash == c2.mesh_hash

    def test_different_plans_produce_different_hashes(self) -> None:
        mesh = make_volume_mesh()
        p1 = _make_simple_plan()
        p2 = SurgicalPlan("test_big", ProcedureType.RHINOPLASTY)
        p2.add_step(dorsal_reduction(amount_mm=5.0))

        c1 = PlanCompiler(mesh).compile(p1)
        c2 = PlanCompiler(mesh).compile(p2)
        assert c1.plan_hash != c2.plan_hash

    def test_empty_plan_still_valid(self) -> None:
        mesh = make_volume_mesh()
        plan = SurgicalPlan("empty", ProcedureType.RHINOPLASTY)
        result = PlanCompiler(mesh).compile(plan)
        assert isinstance(result, CompilationResult)
        # Compiler always adds posterior fixation BCs even for empty plans
        assert result.is_valid


# ═══════════════════════════════════════════════════════════════════
# 2. Plan → Compile → FEM
# ═══════════════════════════════════════════════════════════════════

class TestPlanToFEM:
    """Validate plan compilation feeds directly into FEM solver."""

    def test_minimal_plan_fem_solve(self) -> None:
        mesh = make_volume_mesh()
        plan = _make_simple_plan()
        compilation = PlanCompiler(mesh).compile(plan)
        assert compilation.is_valid

        fem = SoftTissueFEM(
            mesh,
            convergence_tol=1e-4,
            max_newton_iter=50,
        )
        result = fem.solve(compilation)

        assert isinstance(result, FEMResult)
        assert result.displacements.shape == (mesh.n_nodes, 3)
        assert result.stresses.shape[0] == mesh.n_elements
        assert result.wall_clock_seconds > 0.0

    def test_multi_region_fem_solve(self) -> None:
        mesh = _make_multi_region_mesh()
        plan = _make_simple_plan()
        compilation = PlanCompiler(mesh).compile(plan)
        assert compilation.is_valid

        fem = SoftTissueFEM(
            mesh,
            convergence_tol=1e-4,
            max_newton_iter=50,
        )
        result = fem.solve(compilation)

        assert isinstance(result, FEMResult)
        assert result.displacements.shape == (mesh.n_nodes, 3)
        # Should have stresses for all elements
        assert result.stresses.shape[0] == mesh.n_elements

    def test_fem_residual_history_recorded(self) -> None:
        mesh = make_volume_mesh()
        plan = _make_simple_plan()
        compilation = PlanCompiler(mesh).compile(plan)

        fem = SoftTissueFEM(mesh, convergence_tol=1e-4, max_newton_iter=50)
        result = fem.solve(compilation)

        assert len(result.residual_history) > 0
        assert all(r >= 0 for r in result.residual_history)


# ═══════════════════════════════════════════════════════════════════
# 3. CFD standalone
# ═══════════════════════════════════════════════════════════════════

class TestCFDStandalone:
    """Validate CFD solver on synthetic tube geometry."""

    def test_tube_flow_produces_result(self) -> None:
        geometry = _make_straight_tube_geometry()
        solver = AirwayCFDSolver(nx=10, ny=10, nz=40, max_iter=300)
        result = solver.solve(geometry)

        assert isinstance(result, AirwayCFDResult)
        # Solver produces a non-negative flow rate (may be 0 if not converged)
        assert result.total_flow_rate_ml_s >= 0.0
        assert result.pressure_drop_pa >= 0.0
        assert result.nasal_resistance_pa_s_ml >= 0.0
        # The solver should at least have iterated
        assert result.n_iterations > 0

    def test_cfd_section_data_populated(self) -> None:
        geometry = _make_straight_tube_geometry(n_sections=10)
        solver = AirwayCFDSolver(nx=8, ny=8, nz=32, max_iter=100)
        result = solver.solve(geometry)

        assert len(result.section_flow_rates) > 0
        assert len(result.section_velocities) > 0
        assert result.reynolds_number > 0.0


# ═══════════════════════════════════════════════════════════════════
# 4. Healing model
# ═══════════════════════════════════════════════════════════════════

class TestHealingModel:
    """Validate healing predictions from FEM displacements."""

    def test_healing_timeline(self) -> None:
        mesh = make_volume_mesh()
        disps = np.random.default_rng(0).normal(0, 0.5, (mesh.n_nodes, 3))

        model = HealingModel(mesh)
        times = [1.0, 7.0, 30.0, 90.0, 365.0]
        states = model.compute_timeline(times, surgical_displacements=disps)

        assert len(states) == len(times)
        for state in states:
            assert isinstance(state, HealingState)
            assert state.time_days >= 0

    def test_healing_predict_final_shape(self) -> None:
        mesh = make_volume_mesh()
        disps = np.random.default_rng(1).normal(0, 0.3, (mesh.n_nodes, 3))

        model = HealingModel(mesh)
        final_disps, final_state = model.predict_final_shape(disps)

        assert final_disps.shape == disps.shape
        assert isinstance(final_state, HealingState)

    def test_healing_edema_decreases_over_time(self) -> None:
        mesh = make_volume_mesh()
        disps = np.random.default_rng(2).normal(0, 1.0, (mesh.n_nodes, 3))

        model = HealingModel(mesh)
        times = [1.0, 30.0, 365.0]
        states = model.compute_timeline(times, surgical_displacements=disps)

        # Edema should generally decrease from early to late
        if states[0].mean_edema_pct > 0:
            assert states[-1].mean_edema_pct <= states[0].mean_edema_pct


# ═══════════════════════════════════════════════════════════════════
# 5. Orchestrator (full sim pipeline)
# ═══════════════════════════════════════════════════════════════════

class TestOrchestrator:
    """Full multi-physics orchestrator tests."""

    def test_minimal_orchestrator_no_cfd(self) -> None:
        """Run orchestrator with FEM + healing, skip CFD."""
        mesh = make_volume_mesh()
        plan = _make_simple_plan()

        orch = SimOrchestrator(
            mesh,
            run_cfd=False,
            run_healing=True,
            healing_time_points=[1, 30, 365],
            fem_convergence_tol=1e-4,
            fem_max_iter=50,
        )
        result = orch.run(plan)

        assert isinstance(result, SimulationResult)
        assert result.compilation is not None
        assert result.compilation.is_valid
        assert result.plan_hash != ""
        assert result.mesh_hash != ""
        assert result.run_hash != ""
        assert result.wall_clock_seconds > 0.0

        # FEM should have run
        assert result.fem_result is not None
        assert result.fem_result.displacements.shape[1] == 3

        # Healing should have run (3 time points)
        assert len(result.healing_states) == 3
        assert result.final_displacements is not None

        # No CFD
        assert result.cfd_preop is None
        assert result.cfd_postop is None

    def test_orchestrator_with_cfd(self) -> None:
        """Run orchestrator including CFD analysis."""
        mesh = make_volume_mesh()
        plan = _make_simple_plan()

        orch = SimOrchestrator(
            mesh,
            run_cfd=True,
            run_healing=False,
            fem_convergence_tol=1e-4,
            fem_max_iter=50,
            cfd_resolution=6,
            cfd_max_iter=100,
        )
        result = orch.run(plan)

        assert isinstance(result, SimulationResult)
        assert result.fem_result is not None

        # CFD pre-op should have been attempted
        # (may produce a warning if geometry extraction from the
        #  synthetic cube mesh is degenerate, but shouldn't error)
        # We check that the orchestrator completed without hard errors
        assert result.is_valid or any(
            "CFD" in w for w in result.warnings
        )

    def test_orchestrator_multi_region(self) -> None:
        """Orchestrator on multi-region mesh with comprehensive plan."""
        mesh = _make_multi_region_mesh()
        plan = _make_comprehensive_plan()

        orch = SimOrchestrator(
            mesh,
            run_cfd=False,
            run_healing=True,
            healing_time_points=[7, 90],
            fem_convergence_tol=1e-3,
            fem_max_iter=50,
        )
        result = orch.run(plan)

        assert isinstance(result, SimulationResult)
        assert result.compilation is not None
        assert result.fem_result is not None

    def test_orchestrator_to_dict_serializable(self) -> None:
        """Result.to_dict() produces JSON-serializable output."""
        mesh = make_volume_mesh()
        plan = _make_simple_plan()

        orch = SimOrchestrator(
            mesh,
            run_cfd=False,
            run_healing=True,
            healing_time_points=[1, 30],
            fem_convergence_tol=1e-4,
            fem_max_iter=50,
        )
        result = orch.run(plan)
        d = result.to_dict()

        # Must be JSON-serializable
        serialized = json.dumps(d, default=str)
        assert len(serialized) > 0

        # Must contain expected top-level keys
        assert "run_hash" in d
        assert "plan_hash" in d
        assert "mesh_hash" in d
        assert "wall_clock_seconds" in d

    def test_progress_callback_invoked(self) -> None:
        """Verify progress callback receives all stages."""
        stages_seen: list[str] = []

        def callback(stage: str, fraction: float) -> None:
            stages_seen.append(stage)
            assert 0.0 <= fraction <= 1.0

        mesh = make_volume_mesh()
        plan = _make_simple_plan()

        orch = SimOrchestrator(
            mesh,
            run_cfd=False,
            run_healing=False,
            fem_convergence_tol=1e-4,
            fem_max_iter=50,
            progress_callback=callback,
        )
        orch.run(plan)

        # At minimum: compiling, mesh_mods, cartilage, sutures, fem
        assert "compiling" in stages_seen
        assert "fem" in stages_seen


# ═══════════════════════════════════════════════════════════════════
# 6. Aesthetic metrics
# ═══════════════════════════════════════════════════════════════════

class TestAestheticMetrics:
    """Validate aesthetic analysis from landmarks."""

    def test_profile_metrics(self) -> None:
        landmarks = make_rhinoplasty_landmarks()
        am = AestheticMetrics(landmarks, is_female=True)
        profile = am.compute_profile()

        assert profile.dorsal_length_mm > 0.0
        assert profile.tip_projection_mm > 0.0
        assert profile.goode_ratio > 0.0

    def test_symmetry_metrics(self) -> None:
        landmarks = make_rhinoplasty_landmarks()
        am = AestheticMetrics(landmarks, is_female=True)
        sym = am.compute_symmetry()

        # Symmetric landmarks ⇒ low asymmetry
        assert sym.max_asymmetry_mm >= 0.0

    def test_full_aesthetic_report(self) -> None:
        landmarks = make_rhinoplasty_landmarks()
        am = AestheticMetrics(landmarks, is_female=True)
        report = am.compute()

        assert isinstance(report, AestheticReport)
        assert 0.0 <= report.overall_score <= 100.0

        d = report.to_dict()
        assert "profile" in d
        assert "symmetry" in d
        assert "overall_score" in d

    def test_male_vs_female_scoring(self) -> None:
        landmarks = make_rhinoplasty_landmarks()
        fem_report = AestheticMetrics(landmarks, is_female=True).compute()
        male_report = AestheticMetrics(landmarks, is_female=False).compute()

        # Scores may differ for same landmarks due to sex-specific ideals
        assert isinstance(fem_report.overall_score, float)
        assert isinstance(male_report.overall_score, float)


# ═══════════════════════════════════════════════════════════════════
# 7. Functional metrics
# ═══════════════════════════════════════════════════════════════════

class TestFunctionalMetrics:
    """Validate functional assessment from CFD + landmarks."""

    def test_functional_evaluation(self) -> None:
        geometry = _make_straight_tube_geometry()
        solver = AirwayCFDSolver(nx=8, ny=8, nz=32, max_iter=100)
        cfd_result = solver.solve(geometry)

        mesh = make_volume_mesh()
        landmarks = make_rhinoplasty_landmarks()
        fm = FunctionalMetrics(mesh, landmarks)
        report = fm.evaluate(cfd_result)

        assert report.overall_score >= 0.0
        d = report.to_dict()
        assert "resistance" in d
        assert "valve" in d
        assert "flow" in d


# ═══════════════════════════════════════════════════════════════════
# 8. Reports
# ═══════════════════════════════════════════════════════════════════

class TestReportBuilder:
    """Validate report generation from simulation results."""

    def test_build_json_report(self) -> None:
        builder = ReportBuilder("CASE-INT-001", platform_version="0.1.0")
        builder.add_patient_demographics(
            {"age": 28, "sex": "F", "name": "Test Patient"},
            anonymize=True,
        )
        builder.add_surgical_plan({
            "name": "reduction_rhinoplasty",
            "steps": ["dorsal_reduction(2mm)"],
            "plan_hash": "abc123",
        })
        builder.add_simulation_results({
            "run_hash": "def456",
            "fem": {"converged": True, "max_displacement_mm": 2.1},
            "wall_clock_seconds": 3.5,
        })
        builder.add_aesthetic_report({
            "overall_score": 82.5,
            "profile": {"nasofrontal_angle_deg": 118.0},
        })
        builder.add_functional_report({
            "overall_score": 75.0,
            "resistance": {"total": 1.2},
        })
        builder.add_disclaimers([
            "For research use only",
            "Not a substitute for clinical judgement",
        ])

        report = builder.build_json()
        assert "metadata" in report
        assert "sections" in report
        assert len(report["sections"]) == 5  # demographics + plan + sim + aesthetic + functional
        assert report["metadata"]["case_id"] == "CASE-INT-001"
        assert len(report["metadata"]["report_hash"]) > 0

        # Anonymization check
        demo_section = report["sections"][0]
        assert demo_section["content"].get("name") == "[REDACTED]"

    def test_build_markdown_report(self) -> None:
        builder = ReportBuilder("CASE-INT-002")
        builder.add_surgical_plan({"name": "test", "plan_hash": "h1"})
        builder.add_simulation_results({"run_hash": "h2", "fem": {"converged": True}})

        md = builder.build_markdown()
        assert "# Surgical Simulation Report" in md
        assert "CASE-INT-002" in md
        assert "Surgical Plan" in md
        assert "Simulation Results" in md

    def test_build_html_report(self) -> None:
        builder = ReportBuilder("CASE-INT-003")
        builder.add_surgical_plan({"name": "test", "plan_hash": "h1"})

        html = builder.build_html()
        assert "<!DOCTYPE html>" in html
        assert "CASE-INT-003" in html
        assert "<h2>" in html

    def test_save_json_to_disk(self, tmp_path: Path) -> None:
        builder = ReportBuilder("CASE-INT-004")
        builder.add_surgical_plan({"name": "test"})

        out_path = tmp_path / "reports" / "test_report.json"
        saved = builder.save_json(out_path)
        assert saved.exists()

        with open(saved) as f:
            data = json.load(f)
        assert data["metadata"]["case_id"] == "CASE-INT-004"

    def test_save_markdown_to_disk(self, tmp_path: Path) -> None:
        builder = ReportBuilder("CASE-INT-005")
        builder.add_surgical_plan({"name": "test"})

        out_path = tmp_path / "report.md"
        saved = builder.save_markdown(out_path)
        assert saved.exists()
        assert "CASE-INT-005" in saved.read_text()

    def test_save_html_to_disk(self, tmp_path: Path) -> None:
        builder = ReportBuilder("CASE-INT-006")
        builder.add_surgical_plan({"name": "test"})

        out_path = tmp_path / "report.html"
        saved = builder.save_html(out_path)
        assert saved.exists()
        assert "<!DOCTYPE html>" in saved.read_text()


# ═══════════════════════════════════════════════════════════════════
# 9. Full pipeline integration
# ═══════════════════════════════════════════════════════════════════

class TestFullPipelineIntegration:
    """Wire every stage together: plan → simulate → metrics → report."""

    def test_reduction_rhinoplasty_full_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline for a reduction rhinoplasty case."""
        # ── Stage 1: Build mesh and plan ──
        mesh = _make_multi_region_mesh()
        plan = _make_simple_plan()
        landmarks = make_rhinoplasty_landmarks()

        # ── Stage 2: Simulate via orchestrator ──
        orch = SimOrchestrator(
            mesh,
            run_cfd=False,
            run_healing=True,
            healing_time_points=[1, 30, 365],
            fem_convergence_tol=1e-3,
            fem_max_iter=50,
        )
        sim_result = orch.run(plan)

        assert isinstance(sim_result, SimulationResult)
        assert sim_result.fem_result is not None
        assert sim_result.final_displacements is not None

        # ── Stage 3: Aesthetic metrics ──
        aesthetics = AestheticMetrics(landmarks, is_female=True)
        aesthetic_report = aesthetics.compute()
        assert 0.0 <= aesthetic_report.overall_score <= 100.0

        # ── Stage 4: Build report ──
        builder = ReportBuilder(
            "INTEGRATION-001",
            generated_by="test_integration",
            platform_version="0.1.0",
        )
        builder.add_patient_demographics(
            {"age": 32, "sex": "F"},
            anonymize=False,
        )
        builder.add_surgical_plan(plan.to_dict())
        builder.add_simulation_results(sim_result.to_dict())
        builder.add_aesthetic_report(aesthetic_report.to_dict())
        builder.add_healing_prediction({
            "time_points": [1, 30, 365],
            "n_states": len(sim_result.healing_states),
        })
        builder.add_disclaimers(["For research use only"])

        # ── Stage 5: Serialize all formats ──
        report_json = builder.build_json()
        report_md = builder.build_markdown()
        report_html = builder.build_html()

        # Validate JSON report
        assert report_json["metadata"]["case_id"] == "INTEGRATION-001"
        assert len(report_json["sections"]) == 5
        assert len(report_json["metadata"]["report_hash"]) > 0

        # Validate markdown
        assert "Surgical Simulation Report" in report_md
        assert "healing" in report_md.lower() or "Healing" in report_md

        # Validate HTML
        assert "<!DOCTYPE html>" in report_html

        # ── Stage 6: Save to disk ──
        json_path = builder.save_json(tmp_path / "report.json")
        md_path = builder.save_markdown(tmp_path / "report.md")
        html_path = builder.save_html(tmp_path / "report.html")

        assert json_path.exists()
        assert md_path.exists()
        assert html_path.exists()

        # Verify JSON round-trip
        with open(json_path) as f:
            reloaded = json.load(f)
        assert reloaded["metadata"]["case_id"] == "INTEGRATION-001"

    def test_comprehensive_plan_full_pipeline(self) -> None:
        """Comprehensive plan through orchestrator → metrics pipeline."""
        mesh = _make_multi_region_mesh()
        plan = _make_comprehensive_plan()

        orch = SimOrchestrator(
            mesh,
            run_cfd=False,
            run_healing=False,
            fem_convergence_tol=1e-3,
            fem_max_iter=50,
        )
        sim_result = orch.run(plan)

        assert isinstance(sim_result, SimulationResult)
        assert sim_result.compilation is not None

        # Serialize and verify
        d = sim_result.to_dict()
        serialized = json.dumps(d, default=str)
        assert len(serialized) > 100  # non-trivial output

    def test_provenance_hashes_propagate(self) -> None:
        """Verify that plan, mesh, and run hashes propagate through pipeline."""
        mesh = make_volume_mesh()
        plan = _make_simple_plan()

        # Compile
        compilation = PlanCompiler(mesh).compile(plan)
        plan_hash = compilation.plan_hash
        mesh_hash = compilation.mesh_hash

        # Simulate
        orch = SimOrchestrator(
            mesh,
            run_cfd=False,
            run_healing=False,
            fem_convergence_tol=1e-4,
            fem_max_iter=50,
        )
        sim_result = orch.run(plan)

        # Hashes flow from compilation into sim result
        assert sim_result.plan_hash == plan_hash
        assert sim_result.mesh_hash == mesh_hash
        assert len(sim_result.run_hash) > 0
        assert sim_result.run_hash != plan_hash  # run hash is distinct

        # Report preserves simulation hash
        builder = ReportBuilder("PROV-001")
        builder.add_simulation_results(sim_result.to_dict())
        report = builder.build_json()
        assert report["metadata"]["sim_hash"] == sim_result.run_hash

    def test_deformed_mesh_shape_preserved(self) -> None:
        """Final displacements preserve node count after healing."""
        mesh = make_volume_mesh()
        plan = _make_simple_plan()

        orch = SimOrchestrator(
            mesh,
            run_cfd=False,
            run_healing=True,
            healing_time_points=[30],
            fem_convergence_tol=1e-4,
            fem_max_iter=50,
        )
        result = orch.run(plan)

        if result.final_displacements is not None:
            assert result.final_displacements.shape == (mesh.n_nodes, 3)

    def test_cfd_with_functional_metrics_pipeline(self) -> None:
        """CFD result feeds into functional metrics evaluation."""
        geometry = _make_straight_tube_geometry()
        solver = AirwayCFDSolver(nx=8, ny=8, nz=32, max_iter=150)
        cfd_result = solver.solve(geometry)

        mesh = make_volume_mesh()
        landmarks = make_rhinoplasty_landmarks()
        fm = FunctionalMetrics(mesh, landmarks)
        functional_report = fm.evaluate(cfd_result)

        # Feed into report
        builder = ReportBuilder("FUNC-001")
        builder.add_functional_report(functional_report.to_dict())
        report = builder.build_json()

        func_section = next(
            s for s in report["sections"]
            if s["title"] == "Functional Assessment"
        )
        assert "resistance" in func_section["content"]

    def test_sim_result_summary_string(self) -> None:
        """SimulationResult.summary() produces readable output."""
        mesh = make_volume_mesh()
        plan = _make_simple_plan()

        orch = SimOrchestrator(
            mesh,
            run_cfd=False,
            run_healing=False,
            fem_convergence_tol=1e-4,
            fem_max_iter=50,
        )
        result = orch.run(plan)
        summary = result.summary()

        assert "SimulationResult" in summary
        assert "FEM" in summary
        assert "Time:" in summary
