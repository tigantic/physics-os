"""Tests for plan DSL, operators, and compiler."""

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
from products.facial_plastics.plan.operators.rhinoplasty import (
    RHINOPLASTY_OPERATORS,
    RhinoplastyPlanBuilder,
    dorsal_reduction,
    lateral_osteotomy,
    spreader_graft,
    tip_suture,
    cephalic_trim,
)
from products.facial_plastics.plan.compiler import PlanCompiler
from products.facial_plastics.tests.conftest import make_volume_mesh


# ── OperatorParam ────────────────────────────────────────────────

class TestOperatorParam:
    """Test parameter definitions and validation."""

    def test_float_param_in_range(self):
        p = OperatorParam(
            name="amount_mm",
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=10.0,
        )
        val = p.validate(5.0)
        assert val == 5.0

    def test_float_param_out_of_range(self):
        p = OperatorParam(
            name="amount_mm",
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=10.0,
        )
        with pytest.raises(PlanValidationError):
            p.validate(15.0)

    def test_bool_param(self):
        p = OperatorParam(name="preserve_dorsum", param_type=ParamType.BOOL)
        assert p.validate(True) is True
        assert p.validate(False) is False


# ── SurgicalOp ──────────────────────────────────────────────────

class TestSurgicalOp:
    """Test surgical operation node."""

    def test_construction_from_factory(self):
        op = dorsal_reduction(amount_mm=2.5)
        assert op.name == "dorsal_reduction"
        assert op.category == OpCategory.REDUCTION

    def test_validate(self):
        op = dorsal_reduction(amount_mm=2.0)
        errors = op.validate()
        assert errors == []

    def test_content_hash_deterministic(self):
        op1 = dorsal_reduction(amount_mm=2.0)
        op2 = dorsal_reduction(amount_mm=2.0)
        assert op1.content_hash() == op2.content_hash()

    def test_content_hash_changes_with_params(self):
        op1 = dorsal_reduction(amount_mm=2.0)
        op2 = dorsal_reduction(amount_mm=3.0)
        assert op1.content_hash() != op2.content_hash()

    def test_to_dict(self):
        op = dorsal_reduction(amount_mm=2.0)
        d = op.to_dict()
        assert d["name"] == "dorsal_reduction"
        assert d["type"] == "op"


# ── Rhinoplasty Operators ────────────────────────────────────────

class TestRhinoplastyOperators:
    """Test rhinoplasty operator functions."""

    def test_operator_registry(self):
        assert "dorsal_reduction" in RHINOPLASTY_OPERATORS
        assert "lateral_osteotomy" in RHINOPLASTY_OPERATORS
        assert "spreader_graft" in RHINOPLASTY_OPERATORS
        assert len(RHINOPLASTY_OPERATORS) >= 10

    def test_dorsal_reduction(self):
        op = dorsal_reduction(amount_mm=2.5)
        assert op.params["amount_mm"] == 2.5

    def test_lateral_osteotomy(self):
        op = lateral_osteotomy()
        assert op.category == OpCategory.OSTEOTOMY

    def test_spreader_graft(self):
        op = spreader_graft()
        assert op.category == OpCategory.GRAFT

    def test_tip_suture(self):
        op = tip_suture()
        assert op.category == OpCategory.SUTURE

    def test_cephalic_trim(self):
        op = cephalic_trim()
        assert op.category == OpCategory.RESECTION


# ── SequenceNode ─────────────────────────────────────────────────

class TestSequenceNode:
    """Test sequence node composition."""

    def test_add_steps(self):
        seq = SequenceNode(name="primary_rhinoplasty")
        seq.add(dorsal_reduction(amount_mm=2.0))
        seq.add(spreader_graft())
        assert len(seq.steps) == 2

    def test_step_ordering(self):
        seq = SequenceNode(name="test")
        seq.add(dorsal_reduction(amount_mm=2.0))
        seq.add(lateral_osteotomy())
        seq.add(spreader_graft())
        # Steps should have sequential order set by add()
        for i, step in enumerate(seq.steps):
            if isinstance(step, SurgicalOp):
                assert step.order == i

    def test_validate_sequence(self):
        seq = SequenceNode(name="test")
        seq.add(dorsal_reduction(amount_mm=2.0))
        seq.add(spreader_graft())
        errors = seq.validate()
        assert errors == []


# ── SurgicalPlan ────────────────────────────────────────────────

class TestSurgicalPlan:
    """Test surgical plan construction."""

    def test_create_plan(self):
        plan = SurgicalPlan(
            name="Primary rhinoplasty",
            procedure=ProcedureType.RHINOPLASTY,
        )
        plan.add_step(dorsal_reduction(amount_mm=2.0))
        plan.add_step(tip_suture())
        assert plan.name == "Primary rhinoplasty"
        assert plan.procedure == ProcedureType.RHINOPLASTY
        assert len(plan.root.steps) == 2

    def test_plan_validate(self):
        plan = SurgicalPlan(
            name="test",
            procedure=ProcedureType.RHINOPLASTY,
        )
        plan.add_step(dorsal_reduction(amount_mm=2.0))
        errors = plan.validate()
        assert isinstance(errors, list)


# ── RhinoplastyPlanBuilder ───────────────────────────────────────

class TestRhinoplastyPlanBuilder:
    """Test convenience plan builder."""

    def test_builder_creates_plan(self):
        plan = RhinoplastyPlanBuilder.reduction_rhinoplasty(
            dorsal_reduction_mm=2.5,
        )
        assert isinstance(plan, SurgicalPlan)
        assert plan.procedure == ProcedureType.RHINOPLASTY
        # Should have at least dorsal reduction + osteotomies
        assert len(plan.root.steps) >= 3


# ── Plan Compiler ────────────────────────────────────────────────

class TestPlanCompiler:
    """Test plan compilation."""

    def test_compiler_creation(self):
        mesh = make_volume_mesh()
        compiler = PlanCompiler(mesh)
        assert compiler is not None
