"""Tests for expanded plan compiler — facelift, blepharoplasty, and filler compilation.

Verifies that all 21 new operators can compile against a mock volume mesh
without errors, and that the resulting CompilationResult contains appropriate
boundary conditions and/or modifications.
"""

from __future__ import annotations

import pytest

from products.facial_plastics.core.types import ProcedureType
from products.facial_plastics.plan.compiler import (
    CompilationResult,
    PlanCompiler,
    _OP_COMPILERS,
)
from products.facial_plastics.plan.dsl import SurgicalPlan
from products.facial_plastics.tests.conftest import make_volume_mesh

# ── Operator factories ────────────────────────────────────────────

from products.facial_plastics.plan.operators.facelift import (
    FACELIFT_OPERATORS,
    FaceliftPlanBuilder,
    smas_plication,
    smas_flap,
    deep_plane_dissection,
    skin_excision,
    fat_repositioning,
    platysma_plication,
    submentoplasty,
    malar_fat_suspension,
)
from products.facial_plastics.plan.operators.blepharoplasty import (
    BLEPHAROPLASTY_OPERATORS,
    BlepharoplastyPlanBuilder,
    upper_lid_skin_excision,
    upper_lid_fat_removal,
    lower_lid_skin_excision,
    lower_lid_fat_transposition,
    canthopexy,
    orbicularis_tightening,
    skin_pinch,
)
from products.facial_plastics.plan.operators.fillers import (
    FILLER_OPERATORS,
    FillerPlanBuilder,
    ha_filler_injection,
    fat_harvest,
    fat_graft_injection,
    biostimulatory_filler,
    thread_lift,
    implant_placement,
)
from products.facial_plastics.plan.operators.rhinoplasty import (
    RHINOPLASTY_OPERATORS,
)


# =====================================================================
# Compiler registry tests
# =====================================================================


class TestOpCompilersRegistry:
    """Verify _OP_COMPILERS maps all registered operators."""

    def test_total_entries(self):
        assert len(_OP_COMPILERS) == 34  # 13 rhino + 8 facelift + 7 bleph + 6 filler

    def test_rhinoplasty_entries(self):
        for name in RHINOPLASTY_OPERATORS:
            assert name in _OP_COMPILERS, f"Missing compiler for rhinoplasty op: {name}"

    def test_facelift_entries(self):
        for name in FACELIFT_OPERATORS:
            assert name in _OP_COMPILERS, f"Missing compiler for facelift op: {name}"

    def test_blepharoplasty_entries(self):
        for name in BLEPHAROPLASTY_OPERATORS:
            assert name in _OP_COMPILERS, f"Missing compiler for bleph op: {name}"

    def test_filler_entries(self):
        for name in FILLER_OPERATORS:
            assert name in _OP_COMPILERS, f"Missing compiler for filler op: {name}"


# =====================================================================
# Facelift compilation
# =====================================================================


class TestFaceliftCompilation:
    """Test compiler output for facelift operators."""

    @pytest.fixture
    def compiler(self):
        mesh = make_volume_mesh()
        return PlanCompiler(mesh)

    def test_compile_smas_plication(self, compiler):
        plan = SurgicalPlan("test_smas", ProcedureType.FACELIFT)
        plan.add_step(smas_plication())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_smas_flap(self, compiler):
        plan = SurgicalPlan("test_smas_flap", ProcedureType.FACELIFT)
        plan.add_step(smas_flap())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_deep_plane(self, compiler):
        plan = SurgicalPlan("test_deep_plane", ProcedureType.FACELIFT)
        plan.add_step(deep_plane_dissection())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_skin_excision(self, compiler):
        plan = SurgicalPlan("test_skin_excision", ProcedureType.FACELIFT)
        plan.add_step(skin_excision())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_fat_repositioning(self, compiler):
        plan = SurgicalPlan("test_fat_repo", ProcedureType.FACELIFT)
        plan.add_step(fat_repositioning())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_platysma_plication(self, compiler):
        plan = SurgicalPlan("test_platysma", ProcedureType.FACELIFT)
        plan.add_step(platysma_plication())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_submentoplasty(self, compiler):
        plan = SurgicalPlan("test_subment", ProcedureType.FACELIFT)
        plan.add_step(submentoplasty())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_malar_fat_suspension(self, compiler):
        plan = SurgicalPlan("test_malar", ProcedureType.FACELIFT)
        plan.add_step(malar_fat_suspension())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_full_smas_plication_template(self, compiler):
        plan = FaceliftPlanBuilder.smas_plication_facelift()
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"
        # SMAS plication plan should have multiple BCs
        assert result.n_bcs > 0

    def test_compile_full_deep_plane_template(self, compiler):
        plan = FaceliftPlanBuilder.deep_plane_facelift()
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_full_necklift_template(self, compiler):
        plan = FaceliftPlanBuilder.necklift()
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"


# =====================================================================
# Blepharoplasty compilation
# =====================================================================


class TestBlepharoplastyCompilation:
    """Test compiler output for blepharoplasty operators."""

    @pytest.fixture
    def compiler(self):
        mesh = make_volume_mesh()
        return PlanCompiler(mesh)

    def test_compile_upper_lid_skin_excision(self, compiler):
        plan = SurgicalPlan("test_upper_skin", ProcedureType.BLEPHAROPLASTY_UPPER)
        plan.add_step(upper_lid_skin_excision())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_upper_lid_fat_removal(self, compiler):
        plan = SurgicalPlan("test_upper_fat", ProcedureType.BLEPHAROPLASTY_UPPER)
        plan.add_step(upper_lid_fat_removal())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_lower_lid_skin_excision(self, compiler):
        plan = SurgicalPlan("test_lower_skin", ProcedureType.BLEPHAROPLASTY_LOWER)
        plan.add_step(lower_lid_skin_excision())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_lower_lid_fat_transposition(self, compiler):
        plan = SurgicalPlan("test_lower_fat", ProcedureType.BLEPHAROPLASTY_LOWER)
        plan.add_step(lower_lid_fat_transposition())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_canthopexy(self, compiler):
        plan = SurgicalPlan("test_cantho", ProcedureType.BLEPHAROPLASTY_LOWER)
        plan.add_step(canthopexy())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_orbicularis_tightening(self, compiler):
        plan = SurgicalPlan("test_orbic", ProcedureType.BLEPHAROPLASTY_LOWER)
        plan.add_step(orbicularis_tightening())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_skin_pinch(self, compiler):
        plan = SurgicalPlan("test_pinch", ProcedureType.BLEPHAROPLASTY_LOWER)
        plan.add_step(skin_pinch())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_upper_blepharoplasty_template(self, compiler):
        plan = BlepharoplastyPlanBuilder.upper_blepharoplasty()
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_four_lid_template(self, compiler):
        plan = BlepharoplastyPlanBuilder.four_lid_blepharoplasty()
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"


# =====================================================================
# Filler compilation
# =====================================================================


class TestFillerCompilation:
    """Test compiler output for filler / fat graft / implant operators."""

    @pytest.fixture
    def compiler(self):
        mesh = make_volume_mesh()
        return PlanCompiler(mesh)

    def test_compile_ha_filler(self, compiler):
        plan = SurgicalPlan("test_ha", ProcedureType.FILLER_INJECTION)
        plan.add_step(ha_filler_injection())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_fat_harvest(self, compiler):
        plan = SurgicalPlan("test_harvest", ProcedureType.FAT_GRAFTING)
        plan.add_step(fat_harvest())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_fat_graft_injection(self, compiler):
        plan = SurgicalPlan("test_fat_graft", ProcedureType.FAT_GRAFTING)
        plan.add_step(fat_graft_injection())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_biostimulatory_filler(self, compiler):
        plan = SurgicalPlan("test_biostim", ProcedureType.FILLER_INJECTION)
        plan.add_step(biostimulatory_filler())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_thread_lift(self, compiler):
        plan = SurgicalPlan("test_thread", ProcedureType.FILLER_INJECTION)
        plan.add_step(thread_lift())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_implant_placement(self, compiler):
        plan = SurgicalPlan("test_implant", ProcedureType.CHIN_AUGMENTATION)
        plan.add_step(implant_placement())
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_liquid_facelift_template(self, compiler):
        plan = FillerPlanBuilder.liquid_facelift()
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_structural_fat_grafting_template(self, compiler):
        plan = FillerPlanBuilder.structural_fat_grafting()
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_compile_chin_augmentation_template(self, compiler):
        plan = FillerPlanBuilder.chin_augmentation()
        result = compiler.compile(plan)
        assert isinstance(result, CompilationResult)
        assert result.is_valid, f"Errors: {result.errors}"


# =====================================================================
# Cross-procedure compilation tests
# =====================================================================


class TestCrossProcedureCompilation:
    """Test compilation result consistency across all new procedures."""

    @pytest.fixture
    def compiler(self):
        mesh = make_volume_mesh()
        return PlanCompiler(mesh)

    def test_compilation_result_has_plan_hash(self, compiler):
        plan = FaceliftPlanBuilder.smas_plication_facelift()
        result = compiler.compile(plan)
        assert result.plan_hash != ""

    def test_compilation_result_has_mesh_hash(self, compiler):
        plan = FaceliftPlanBuilder.smas_plication_facelift()
        result = compiler.compile(plan)
        assert result.mesh_hash != ""

    def test_compilation_result_deterministic(self, compiler):
        plan1 = FaceliftPlanBuilder.necklift()
        plan2 = FaceliftPlanBuilder.necklift()
        result1 = compiler.compile(plan1)
        result2 = compiler.compile(plan2)
        assert result1.content_hash() == result2.content_hash()

    def test_different_plans_different_hashes(self, compiler):
        plan1 = FaceliftPlanBuilder.necklift()
        plan2 = FaceliftPlanBuilder.deep_plane_facelift()
        result1 = compiler.compile(plan1)
        result2 = compiler.compile(plan2)
        assert result1.plan_hash != result2.plan_hash

    def test_summary_does_not_crash(self, compiler):
        """Verify .summary() returns a non-empty string for all templates."""
        templates = [
            FaceliftPlanBuilder.smas_plication_facelift(),
            FaceliftPlanBuilder.deep_plane_facelift(),
            FaceliftPlanBuilder.necklift(),
            BlepharoplastyPlanBuilder.upper_blepharoplasty(),
            BlepharoplastyPlanBuilder.four_lid_blepharoplasty(),
            FillerPlanBuilder.liquid_facelift(),
            FillerPlanBuilder.chin_augmentation(),
        ]
        for plan in templates:
            result = compiler.compile(plan)
            summary = result.summary()
            assert isinstance(summary, str)
            assert len(summary) > 10

    def test_to_dict_does_not_crash(self, compiler):
        """Verify .to_dict() returns a dict for all templates."""
        templates = [
            FaceliftPlanBuilder.smas_plication_facelift(),
            BlepharoplastyPlanBuilder.upper_blepharoplasty(),
            FillerPlanBuilder.liquid_facelift(),
        ]
        for plan in templates:
            result = compiler.compile(plan)
            d = result.to_dict()
            assert isinstance(d, dict)
            assert "plan_hash" in d
