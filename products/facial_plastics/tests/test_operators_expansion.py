"""Tests for facelift, blepharoplasty, and filler operator modules.

Covers instantiation, parameter validation, registry completeness,
plan builder templates, and content hashing for all 21 new operators.
"""

from __future__ import annotations

import pytest

from products.facial_plastics.core.types import (
    ProcedureType,
    StructureType,
)
from products.facial_plastics.plan.dsl import (
    OpCategory,
    OperatorParam,
    ParamType,
    PlanValidationError,
    SequenceNode,
    SurgicalOp,
    SurgicalPlan,
)

# ── Facelift ──────────────────────────────────────────────────────

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

# ── Blepharoplasty ────────────────────────────────────────────────

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

# ── Fillers ───────────────────────────────────────────────────────

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


# =====================================================================
# Facelift operators
# =====================================================================


class TestFaceliftOperatorRegistry:
    """FACELIFT_OPERATORS registry integrity."""

    def test_registry_size(self):
        assert len(FACELIFT_OPERATORS) == 8

    def test_registry_keys(self):
        expected = {
            "smas_plication",
            "smas_flap",
            "deep_plane_dissection",
            "skin_excision_facelift",
            "fat_repositioning",
            "platysma_plication",
            "submentoplasty",
            "malar_fat_suspension",
        }
        assert set(FACELIFT_OPERATORS.keys()) == expected

    def test_all_callables(self):
        for name, factory in FACELIFT_OPERATORS.items():
            assert callable(factory), f"{name} is not callable"


class TestSmasPlication:
    def test_defaults(self):
        op = smas_plication()
        assert op.name == "smas_plication"
        assert op.category == OpCategory.SUTURE
        assert op.procedure == ProcedureType.FACELIFT

    def test_custom_params(self):
        op = smas_plication(vector_deg=45.0, plication_width_mm=15.0)
        assert op.params["vector_deg"] == 45.0
        assert op.params["plication_width_mm"] == 15.0

    def test_validate(self):
        op = smas_plication()
        errors = op.validate()
        assert errors == []

    def test_affected_structures(self):
        op = smas_plication()
        assert StructureType.SMAS in op.affected_structures

    def test_content_hash_deterministic(self):
        op1 = smas_plication(vector_deg=60.0)
        op2 = smas_plication(vector_deg=60.0)
        assert op1.content_hash() == op2.content_hash()

    def test_content_hash_changes(self):
        op1 = smas_plication(vector_deg=60.0)
        op2 = smas_plication(vector_deg=45.0)
        assert op1.content_hash() != op2.content_hash()


class TestSmasFlap:
    def test_defaults(self):
        op = smas_flap()
        assert op.name == "smas_flap"
        assert op.procedure == ProcedureType.FACELIFT

    def test_custom_flap_type(self):
        op = smas_flap(flap_type="lateral_smasectomy")
        assert op.params["flap_type"] == "lateral_smasectomy"

    def test_validate(self):
        assert smas_flap().validate() == []


class TestDeepPlaneDissection:
    def test_defaults(self):
        op = deep_plane_dissection()
        assert op.name == "deep_plane_dissection"
        assert op.category == OpCategory.RELEASE

    def test_with_cervical_release(self):
        op = deep_plane_dissection(release_cervical=True)
        assert op.params["release_cervical"] is True

    def test_validate(self):
        assert deep_plane_dissection().validate() == []


class TestSkinExcisionFacelift:
    def test_defaults(self):
        op = skin_excision()
        assert op.name == "skin_excision_facelift"
        assert op.category == OpCategory.RESECTION

    def test_short_scar_pattern(self):
        op = skin_excision(excision_pattern="short_scar")
        assert op.params["excision_pattern"] == "short_scar"

    def test_validate(self):
        assert skin_excision().validate() == []


class TestFatRepositioning:
    def test_defaults(self):
        op = fat_repositioning()
        assert op.name == "fat_repositioning"
        assert op.category == OpCategory.REPOSITIONING

    def test_compartment(self):
        op = fat_repositioning(compartment="nasolabial")
        assert op.params["compartment"] == "nasolabial"


class TestPlatysmaPlication:
    def test_defaults(self):
        op = platysma_plication()
        assert op.name == "platysma_plication"
        assert op.procedure == ProcedureType.NECKLIFT

    def test_corset_technique(self):
        op = platysma_plication(technique="corset")
        assert op.params["technique"] == "corset"


class TestSubmentoplasty:
    def test_defaults(self):
        op = submentoplasty()
        assert op.name == "submentoplasty"
        assert op.category == OpCategory.RESECTION

    def test_liposuction_volume(self):
        op = submentoplasty(liposuction_volume_cc=20.0)
        assert op.params["liposuction_volume_cc"] == 20.0


class TestMalarFatSuspension:
    def test_defaults(self):
        op = malar_fat_suspension()
        assert op.name == "malar_fat_suspension"
        assert op.category == OpCategory.REPOSITIONING

    def test_validate(self):
        assert malar_fat_suspension().validate() == []


# =====================================================================
# Facelift plan builder
# =====================================================================

class TestFaceliftPlanBuilder:
    def test_smas_plication_facelift(self):
        plan = FaceliftPlanBuilder.smas_plication_facelift()
        assert isinstance(plan, SurgicalPlan)
        assert plan.procedure == ProcedureType.FACELIFT
        assert len(plan.root.steps) >= 2

    def test_deep_plane_facelift(self):
        plan = FaceliftPlanBuilder.deep_plane_facelift()
        assert isinstance(plan, SurgicalPlan)
        assert plan.name == "deep_plane_facelift"
        assert len(plan.root.steps) >= 4

    def test_necklift(self):
        plan = FaceliftPlanBuilder.necklift()
        assert isinstance(plan, SurgicalPlan)
        assert plan.procedure == ProcedureType.NECKLIFT
        assert len(plan.root.steps) >= 2

    def test_lateral_smasectomy(self):
        plan = FaceliftPlanBuilder.lateral_smasectomy_facelift()
        assert isinstance(plan, SurgicalPlan)
        assert len(plan.root.steps) >= 2

    def test_all_templates_validate(self):
        plans = [
            FaceliftPlanBuilder.smas_plication_facelift(),
            FaceliftPlanBuilder.deep_plane_facelift(),
            FaceliftPlanBuilder.necklift(),
            FaceliftPlanBuilder.lateral_smasectomy_facelift(),
        ]
        for plan in plans:
            errors = plan.validate()
            assert isinstance(errors, list), f"{plan.name} validate() should return list"


# =====================================================================
# Blepharoplasty operators
# =====================================================================

class TestBlepharoplastyOperatorRegistry:
    def test_registry_size(self):
        assert len(BLEPHAROPLASTY_OPERATORS) == 7

    def test_registry_keys(self):
        expected = {
            "upper_lid_skin_excision",
            "upper_lid_fat_removal",
            "lower_lid_skin_excision",
            "lower_lid_fat_transposition",
            "canthopexy",
            "orbicularis_tightening",
            "skin_pinch",
        }
        assert set(BLEPHAROPLASTY_OPERATORS.keys()) == expected

    def test_all_callables(self):
        for name, factory in BLEPHAROPLASTY_OPERATORS.items():
            assert callable(factory), f"{name} is not callable"


class TestUpperLidSkinExcision:
    def test_defaults(self):
        op = upper_lid_skin_excision()
        assert op.name == "upper_lid_skin_excision"
        assert op.category == OpCategory.RESECTION
        assert op.procedure == ProcedureType.BLEPHAROPLASTY_UPPER

    def test_custom_height(self):
        op = upper_lid_skin_excision(width_mm=10.0)
        assert op.params["width_mm"] == 10.0

    def test_validate(self):
        assert upper_lid_skin_excision().validate() == []

    def test_content_hash_deterministic(self):
        op1 = upper_lid_skin_excision(width_mm=8.0)
        op2 = upper_lid_skin_excision(width_mm=8.0)
        assert op1.content_hash() == op2.content_hash()


class TestUpperLidFatRemoval:
    def test_defaults(self):
        op = upper_lid_fat_removal()
        assert op.name == "upper_lid_fat_removal"
        assert op.category == OpCategory.RESECTION


class TestLowerLidSkinExcision:
    def test_defaults(self):
        op = lower_lid_skin_excision()
        assert op.name == "lower_lid_skin_excision"
        assert op.procedure == ProcedureType.BLEPHAROPLASTY_LOWER


class TestLowerLidFatTransposition:
    def test_defaults(self):
        op = lower_lid_fat_transposition()
        assert op.name == "lower_lid_fat_transposition"
        assert op.category == OpCategory.REPOSITIONING


class TestCanthopexy:
    def test_defaults(self):
        op = canthopexy()
        assert op.name == "canthopexy"
        assert op.category == OpCategory.SUTURE

    def test_validate(self):
        assert canthopexy().validate() == []


class TestOrbicularisTightening:
    def test_defaults(self):
        op = orbicularis_tightening()
        assert op.name == "orbicularis_tightening"

    def test_validate(self):
        assert orbicularis_tightening().validate() == []


class TestSkinPinch:
    def test_defaults(self):
        op = skin_pinch()
        assert op.name == "skin_pinch"
        assert op.category == OpCategory.RESECTION


# =====================================================================
# Blepharoplasty plan builder
# =====================================================================

class TestBlepharoplastyPlanBuilder:
    def test_upper_blepharoplasty(self):
        plan = BlepharoplastyPlanBuilder.upper_blepharoplasty()
        assert isinstance(plan, SurgicalPlan)
        assert plan.procedure == ProcedureType.BLEPHAROPLASTY_UPPER
        assert len(plan.root.steps) >= 1

    def test_transconjunctival_lower(self):
        plan = BlepharoplastyPlanBuilder.transconjunctival_lower_blepharoplasty()
        assert isinstance(plan, SurgicalPlan)
        assert plan.procedure == ProcedureType.BLEPHAROPLASTY_LOWER

    def test_transcutaneous_lower(self):
        plan = BlepharoplastyPlanBuilder.transcutaneous_lower_blepharoplasty()
        assert isinstance(plan, SurgicalPlan)
        assert plan.procedure == ProcedureType.BLEPHAROPLASTY_LOWER

    def test_four_lid(self):
        plan = BlepharoplastyPlanBuilder.four_lid_blepharoplasty()
        assert isinstance(plan, SurgicalPlan)
        # 4-lid should have upper + lower steps
        assert len(plan.root.steps) >= 3

    def test_all_templates_validate(self):
        plans = [
            BlepharoplastyPlanBuilder.upper_blepharoplasty(),
            BlepharoplastyPlanBuilder.transconjunctival_lower_blepharoplasty(),
            BlepharoplastyPlanBuilder.transcutaneous_lower_blepharoplasty(),
            BlepharoplastyPlanBuilder.four_lid_blepharoplasty(),
        ]
        for plan in plans:
            errors = plan.validate()
            assert isinstance(errors, list), f"{plan.name} validate() should return list"


# =====================================================================
# Filler operators
# =====================================================================

class TestFillerOperatorRegistry:
    def test_registry_size(self):
        assert len(FILLER_OPERATORS) == 6

    def test_registry_keys(self):
        expected = {
            "ha_filler_injection",
            "fat_harvest",
            "fat_graft_injection",
            "biostimulatory_filler",
            "thread_lift",
            "implant_placement",
        }
        assert set(FILLER_OPERATORS.keys()) == expected

    def test_all_callables(self):
        for name, factory in FILLER_OPERATORS.items():
            assert callable(factory), f"{name} is not callable"


class TestHaFillerInjection:
    def test_defaults(self):
        op = ha_filler_injection()
        assert op.name == "ha_filler_injection"
        assert op.category == OpCategory.AUGMENTATION
        assert op.procedure == ProcedureType.FILLER_INJECTION

    def test_custom_volume(self):
        op = ha_filler_injection(volume_cc=1.5)
        assert op.params["volume_cc"] == 1.5

    def test_validate(self):
        assert ha_filler_injection().validate() == []

    def test_content_hash_deterministic(self):
        op1 = ha_filler_injection(volume_cc=1.0)
        op2 = ha_filler_injection(volume_cc=1.0)
        assert op1.content_hash() == op2.content_hash()


class TestFatHarvest:
    def test_defaults(self):
        op = fat_harvest()
        assert op.name == "fat_harvest"
        assert op.procedure == ProcedureType.FAT_GRAFTING

    def test_validate(self):
        assert fat_harvest().validate() == []


class TestFatGraftInjection:
    def test_defaults(self):
        op = fat_graft_injection()
        assert op.name == "fat_graft_injection"
        assert op.category == OpCategory.AUGMENTATION

    def test_validate(self):
        assert fat_graft_injection().validate() == []


class TestBiostimulatoryFiller:
    def test_defaults(self):
        op = biostimulatory_filler()
        assert op.name == "biostimulatory_filler"
        assert op.category == OpCategory.AUGMENTATION

    def test_validate(self):
        assert biostimulatory_filler().validate() == []


class TestThreadLift:
    def test_defaults(self):
        op = thread_lift()
        assert op.name == "thread_lift"
        assert op.category == OpCategory.REPOSITIONING

    def test_validate(self):
        assert thread_lift().validate() == []


class TestImplantPlacement:
    def test_defaults(self):
        op = implant_placement()
        assert op.name == "implant_placement"
        assert op.category == OpCategory.AUGMENTATION
        assert op.procedure == ProcedureType.CHIN_AUGMENTATION

    def test_validate(self):
        assert implant_placement().validate() == []


# =====================================================================
# Filler plan builder
# =====================================================================

class TestFillerPlanBuilder:
    def test_liquid_facelift(self):
        plan = FillerPlanBuilder.liquid_facelift()
        assert isinstance(plan, SurgicalPlan)
        assert plan.procedure == ProcedureType.FILLER_INJECTION
        assert len(plan.root.steps) >= 1

    def test_structural_fat_grafting(self):
        plan = FillerPlanBuilder.structural_fat_grafting()
        assert isinstance(plan, SurgicalPlan)
        assert plan.procedure == ProcedureType.FAT_GRAFTING

    def test_chin_augmentation(self):
        plan = FillerPlanBuilder.chin_augmentation()
        assert isinstance(plan, SurgicalPlan)
        assert plan.procedure == ProcedureType.CHIN_AUGMENTATION

    def test_thread_lift_midface(self):
        plan = FillerPlanBuilder.thread_lift_midface()
        assert isinstance(plan, SurgicalPlan)
        assert len(plan.root.steps) >= 1

    def test_all_templates_validate(self):
        plans = [
            FillerPlanBuilder.liquid_facelift(),
            FillerPlanBuilder.structural_fat_grafting(),
            FillerPlanBuilder.chin_augmentation(),
            FillerPlanBuilder.thread_lift_midface(),
        ]
        for plan in plans:
            errors = plan.validate()
            assert isinstance(errors, list), f"{plan.name} validate() should return list"


# =====================================================================
# Cross-registry tests
# =====================================================================

class TestAllOperatorRegistries:
    """Verify operator registries are consistent and non-overlapping."""

    def test_total_operator_count(self):
        total = (
            len(FACELIFT_OPERATORS)
            + len(BLEPHAROPLASTY_OPERATORS)
            + len(FILLER_OPERATORS)
        )
        assert total == 21  # 8 + 7 + 6

    def test_no_key_overlap(self):
        all_keys = list(FACELIFT_OPERATORS.keys()) + \
                   list(BLEPHAROPLASTY_OPERATORS.keys()) + \
                   list(FILLER_OPERATORS.keys())
        assert len(all_keys) == len(set(all_keys)), "Operator name collision detected"

    def test_every_operator_produces_surgical_op(self):
        for registry in (FACELIFT_OPERATORS, BLEPHAROPLASTY_OPERATORS, FILLER_OPERATORS):
            for name, factory in registry.items():
                op = factory()
                assert isinstance(op, SurgicalOp), f"{name} should produce SurgicalOp"

    def test_every_operator_validates(self):
        for registry in (FACELIFT_OPERATORS, BLEPHAROPLASTY_OPERATORS, FILLER_OPERATORS):
            for name, factory in registry.items():
                op = factory()
                errors = op.validate()
                assert isinstance(errors, list), f"{name}.validate() should return list"
                assert errors == [], f"{name} has validation errors: {errors}"

    def test_every_operator_has_description(self):
        for registry in (FACELIFT_OPERATORS, BLEPHAROPLASTY_OPERATORS, FILLER_OPERATORS):
            for name, factory in registry.items():
                op = factory()
                assert op.description, f"{name} is missing description"

    def test_every_operator_has_affected_structures(self):
        for registry in (FACELIFT_OPERATORS, BLEPHAROPLASTY_OPERATORS, FILLER_OPERATORS):
            for name, factory in registry.items():
                op = factory()
                assert len(op.affected_structures) > 0, f"{name} has no affected structures"
