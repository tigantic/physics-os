"""Tests for the problem compiler (Phase 3)."""

from __future__ import annotations

import pytest

from physics_os.templates.models import (
    BoundarySpec,
    FlowConditions,
    GeometrySpec,
    GeometryType,
    ProblemClass,
    ProblemResult,
    ProblemSpec,
)
from physics_os.templates.registry import TemplateInfo, TemplateRegistry
from physics_os.templates.compiler import compile_problem


# ──────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────

class TestTemplateRegistry:
    def test_all_8_classes_registered(self) -> None:
        reg = TemplateRegistry()
        assert len(reg.list_all()) == 8

    def test_get_external_flow(self) -> None:
        reg = TemplateRegistry()
        info = reg.get(ProblemClass.EXTERNAL_FLOW)
        assert isinstance(info, TemplateInfo)
        assert GeometryType.CIRCLE in info.supported_geometries

    def test_list_keys(self) -> None:
        reg = TemplateRegistry()
        keys = reg.list_keys()
        assert "external_flow" in keys
        assert "channel_flow" in keys


# ──────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────

class TestProblemSpec:
    def test_minimal_valid(self) -> None:
        spec = ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            flow=FlowConditions(velocity=1.0),
        )
        assert spec.quality == "standard"
        assert spec.geometry.shape == GeometryType.NONE

    def test_with_geometry(self) -> None:
        spec = ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.CIRCLE,
                params={"radius": 0.01},
            ),
            flow=FlowConditions(velocity=10.0, fluid="water"),
        )
        assert spec.geometry.params["radius"] == 0.01

    def test_invalid_quality_raises(self) -> None:
        with pytest.raises(ValueError, match="quality"):
            ProblemSpec(
                problem_class=ProblemClass.EXTERNAL_FLOW,
                flow=FlowConditions(velocity=1.0),
                quality="ultra",
            )


# ──────────────────────────────────────────────────────────────────
# Compiler
# ──────────────────────────────────────────────────────────────────

class TestCompileProblem:
    def test_external_flow_circle(self) -> None:
        """Cylinder in air crossflow → navier_stokes_2d."""
        spec = ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.CIRCLE,
                params={"radius": 0.01},
            ),
            flow=FlowConditions(velocity=1.0, fluid="air"),
        )
        result = compile_problem(spec)
        assert isinstance(result, ProblemResult)
        assert result.domain == "navier_stokes_2d"
        assert result.reynolds_number > 0
        assert result.n_bits >= 4
        assert result.n_steps >= 1
        assert result.dt > 0

    def test_external_flow_naca(self) -> None:
        """NACA airfoil in air."""
        spec = ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.NACA_AIRFOIL,
                params={"chord": 0.5},
            ),
            flow=FlowConditions(velocity=30.0, fluid="air"),
        )
        result = compile_problem(spec)
        assert result.domain == "navier_stokes_2d"
        assert result.characteristic_length == 0.5
        # Re = 30 * 0.5 / 1.516e-5 ≈ 9.9e5
        assert result.reynolds_number > 9e5

    def test_internal_flow_step(self) -> None:
        """Backward-facing step in water."""
        spec = ProblemSpec(
            problem_class=ProblemClass.INTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.BACKWARD_STEP,
                params={"step_height": 0.01, "channel_height": 0.02},
            ),
            flow=FlowConditions(velocity=0.5, fluid="water"),
        )
        result = compile_problem(spec)
        assert result.domain == "navier_stokes_2d"
        assert result.fluid_name == "Water"

    def test_heat_transfer_plate(self) -> None:
        """Flat plate heat transfer → advection_diffusion."""
        spec = ProblemSpec(
            problem_class=ProblemClass.HEAT_TRANSFER,
            geometry=GeometrySpec(
                shape=GeometryType.FLAT_PLATE,
                params={"length": 0.1},
            ),
            flow=FlowConditions(velocity=2.0, fluid="air"),
        )
        result = compile_problem(spec)
        assert result.domain == "advection_diffusion"

    def test_wave_propagation(self) -> None:
        """Acoustic wave → maxwell domain."""
        spec = ProblemSpec(
            problem_class=ProblemClass.WAVE_PROPAGATION,
            flow=FlowConditions(velocity=343.0, fluid="air"),
        )
        result = compile_problem(spec)
        assert result.domain == "maxwell"

    def test_vortex_dynamics(self) -> None:
        """Vortex shedding behind cylinder."""
        spec = ProblemSpec(
            problem_class=ProblemClass.VORTEX_DYNAMICS,
            geometry=GeometrySpec(
                shape=GeometryType.CIRCLE,
                params={"radius": 0.005},
            ),
            flow=FlowConditions(velocity=0.3, fluid="water"),
        )
        result = compile_problem(spec)
        assert result.domain == "navier_stokes_2d"
        assert result.geometry_type == "circle"

    def test_unsupported_geometry_raises(self) -> None:
        """FIN_ARRAY is not supported for EXTERNAL_FLOW."""
        spec = ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.FIN_ARRAY,
                params={"fin_height": 0.01},
            ),
            flow=FlowConditions(velocity=1.0),
        )
        with pytest.raises(ValueError, match="not supported"):
            compile_problem(spec)

    def test_unknown_fluid_raises(self) -> None:
        """Unknown fluid key raises KeyError."""
        spec = ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            flow=FlowConditions(velocity=1.0, fluid="kryptonite"),
        )
        with pytest.raises(KeyError, match="Unknown fluid"):
            compile_problem(spec)

    def test_quality_tiers_affect_resolution(self) -> None:
        """Higher tier → equal or more grid points."""
        results = {}
        for tier in ["quick", "standard", "high", "maximum"]:
            spec = ProblemSpec(
                problem_class=ProblemClass.EXTERNAL_FLOW,
                geometry=GeometrySpec(
                    shape=GeometryType.CIRCLE,
                    params={"radius": 0.01},
                ),
                flow=FlowConditions(velocity=5.0),
                quality=tier,
            )
            results[tier] = compile_problem(spec)
        assert results["maximum"].n_bits >= results["quick"].n_bits

    def test_explicit_t_end(self) -> None:
        """Explicit t_end overrides auto-computation."""
        spec = ProblemSpec(
            problem_class=ProblemClass.CHANNEL_FLOW,
            flow=FlowConditions(velocity=1.0),
            t_end=0.001,
        )
        result = compile_problem(spec)
        assert result.n_steps >= 1

    def test_result_has_warnings_list(self) -> None:
        """Even a clean result has a warnings list (possibly empty)."""
        spec = ProblemSpec(
            problem_class=ProblemClass.BOUNDARY_LAYER,
            geometry=GeometrySpec(
                shape=GeometryType.FLAT_PLATE,
                params={"length": 0.1},
            ),
            flow=FlowConditions(velocity=1.0),
        )
        result = compile_problem(spec)
        assert isinstance(result.warnings, list)

    def test_result_serializable(self) -> None:
        """ProblemResult can be serialized to JSON."""
        spec = ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            geometry=GeometrySpec(shape=GeometryType.CIRCLE, params={"radius": 0.01}),
            flow=FlowConditions(velocity=1.0),
        )
        result = compile_problem(spec)
        json_str = result.model_dump_json()
        assert "navier_stokes_2d" in json_str
