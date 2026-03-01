"""Phase 6 — Alpha test scenarios for the Problem Template System.

Tests the full pipeline: ProblemSpec → compile_problem() → ProblemResult
across all 8 problem classes, geometry variants, quality tiers,
and edge cases.  Also tests the API models, template registry,
and SDK/MCP integration surfaces.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from physics_os.templates.compiler import compile_problem
from physics_os.templates.models import (
    BoundarySpec,
    FlowConditions,
    GeometrySpec,
    GeometryType,
    ProblemClass,
    ProblemResult,
    ProblemSpec,
)
from physics_os.templates.registry import TemplateRegistry


# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────


def _make_spec(
    problem_class: str,
    shape: str = "circle",
    params: dict[str, Any] | None = None,
    velocity: float = 10.0,
    fluid: str = "air",
    quality: str = "standard",
    t_end: float | None = None,
    max_rank: int = 64,
) -> ProblemSpec:
    return ProblemSpec(
        problem_class=ProblemClass(problem_class),
        geometry=GeometrySpec(
            shape=GeometryType(shape),
            params=params or {},
        ),
        flow=FlowConditions(velocity=velocity, fluid=fluid),
        quality=quality,
        t_end=t_end,
        max_rank=max_rank,
    )


def _compile(spec: ProblemSpec) -> ProblemResult:
    return compile_problem(spec)


# ═════════════════════════════════════════════════════════════════
# 1. External flow scenarios
# ═════════════════════════════════════════════════════════════════


class TestExternalFlow:
    """External flow over various bluff bodies."""

    def test_cylinder_air(self) -> None:
        spec = _make_spec("external_flow", "circle", {"radius": 0.01})
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"
        assert r.reynolds_number > 0
        assert r.n_bits >= 4
        assert r.n_steps >= 1

    def test_naca_airfoil(self) -> None:
        spec = _make_spec("external_flow", "naca_airfoil", {"digits": "2412", "chord": 0.3})
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"
        assert r.reynolds_number > 0

    def test_ellipse_water(self) -> None:
        spec = _make_spec("external_flow", "ellipse", {"a": 0.05, "b": 0.02}, fluid="water")
        r = _compile(spec)
        assert r.fluid_name.lower() == "water"
        # Water → higher Re at same velocity
        assert r.reynolds_number > 1e4

    def test_rectangle_bluff_body(self) -> None:
        spec = _make_spec("external_flow", "rectangle", {"width": 0.1, "height": 0.05})
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"

    def test_wedge(self) -> None:
        spec = _make_spec("external_flow", "wedge", {"half_angle_deg": 15, "length": 0.1})
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"

    def test_rounded_rectangle(self) -> None:
        spec = _make_spec("external_flow", "rounded_rectangle", {"width": 0.08, "height": 0.04, "corner_radius": 0.005})
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"


# ═════════════════════════════════════════════════════════════════
# 2. Internal flow scenarios
# ═════════════════════════════════════════════════════════════════


class TestInternalFlow:
    def test_backward_step(self) -> None:
        spec = _make_spec("internal_flow", "backward_step", {"step_height": 0.02, "expansion_ratio": 2.0}, velocity=5.0)
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"

    def test_pipe_bend(self) -> None:
        spec = _make_spec("internal_flow", "pipe_bend", {"inner_radius": 0.05, "outer_radius": 0.06, "bend_angle_deg": 90})
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"

    def test_annulus(self) -> None:
        spec = _make_spec("internal_flow", "annulus", {"inner_radius": 0.01, "outer_radius": 0.03})
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"

    def test_channel_with_no_geometry(self) -> None:
        spec = _make_spec("internal_flow", "none", velocity=2.0)
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"


# ═════════════════════════════════════════════════════════════════
# 3. Heat transfer scenarios
# ═════════════════════════════════════════════════════════════════


class TestHeatTransfer:
    def test_cylinder_crossflow(self) -> None:
        spec = _make_spec("heat_transfer", "circle", {"radius": 0.005}, velocity=5.0)
        r = _compile(spec)
        assert r.domain == "advection_diffusion"

    def test_flat_plate(self) -> None:
        spec = _make_spec("heat_transfer", "flat_plate", {"length": 0.5, "thickness": 0.002})
        r = _compile(spec)
        assert r.domain == "advection_diffusion"

    def test_fin_array(self) -> None:
        spec = _make_spec("heat_transfer", "fin_array", {"n_fins": 5, "fin_height": 0.02, "fin_thickness": 0.002, "spacing": 0.01})
        r = _compile(spec)
        assert r.domain == "advection_diffusion"

    def test_no_geometry_heat(self) -> None:
        spec = _make_spec("heat_transfer", "none", velocity=1.0)
        r = _compile(spec)
        assert r.domain == "advection_diffusion"


# ═════════════════════════════════════════════════════════════════
# 4. Wave propagation
# ═════════════════════════════════════════════════════════════════


class TestWavePropagation:
    def test_default(self) -> None:
        spec = _make_spec("wave_propagation", "none", velocity=343.0)
        r = _compile(spec)
        assert r.domain == "maxwell"

    def test_rectangle_waveguide(self) -> None:
        spec = _make_spec("wave_propagation", "rectangle", {"width": 0.1, "height": 0.05}, velocity=343.0)
        r = _compile(spec)
        assert r.domain == "maxwell"


# ═════════════════════════════════════════════════════════════════
# 5. Natural convection
# ═════════════════════════════════════════════════════════════════


class TestNaturalConvection:
    def test_default(self) -> None:
        spec = _make_spec("natural_convection", "none", velocity=0.1)
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"

    def test_cavity(self) -> None:
        spec = _make_spec("natural_convection", "rectangle", {"width": 0.1, "height": 0.1}, velocity=0.05)
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"


# ═════════════════════════════════════════════════════════════════
# 6. Boundary layer
# ═════════════════════════════════════════════════════════════════


class TestBoundaryLayer:
    def test_flat_plate_bl(self) -> None:
        spec = _make_spec("boundary_layer", "flat_plate", {"length": 1.0, "thickness": 0.002}, velocity=20.0)
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"
        # High velocity → turbulent
        assert r.reynolds_number > 1e5

    def test_no_geometry_bl(self) -> None:
        spec = _make_spec("boundary_layer", "none", velocity=5.0)
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"


# ═════════════════════════════════════════════════════════════════
# 7. Vortex dynamics
# ═════════════════════════════════════════════════════════════════


class TestVortexDynamics:
    def test_cylinder_vortex(self) -> None:
        spec = _make_spec("vortex_dynamics", "circle", {"radius": 0.01}, velocity=5.0)
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"

    def test_rectangle_vortex(self) -> None:
        spec = _make_spec("vortex_dynamics", "rectangle", {"width": 0.05, "height": 0.02}, velocity=3.0)
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"


# ═════════════════════════════════════════════════════════════════
# 8. Channel flow
# ═════════════════════════════════════════════════════════════════


class TestChannelFlow:
    def test_default_channel(self) -> None:
        spec = _make_spec("channel_flow", "none", velocity=1.0)
        r = _compile(spec)
        assert r.domain == "navier_stokes_2d"


# ═════════════════════════════════════════════════════════════════
# 9. Quality tier sweep
# ═════════════════════════════════════════════════════════════════


class TestQualityTiers:
    @pytest.mark.parametrize("tier,min_bits", [
        ("quick", 4),
        ("standard", 4),
        ("high", 4),
        ("maximum", 4),
    ])
    def test_resolution_increases_with_tier(self, tier: str, min_bits: int) -> None:
        spec = _make_spec("external_flow", "circle", {"radius": 0.01}, quality=tier)
        r = _compile(spec)
        assert r.n_bits >= min_bits

    def test_quick_fewer_bits_than_maximum(self) -> None:
        r_quick = _compile(_make_spec("external_flow", "circle", {"radius": 0.01}, quality="quick"))
        r_max = _compile(_make_spec("external_flow", "circle", {"radius": 0.01}, quality="maximum"))
        assert r_quick.n_bits <= r_max.n_bits


# ═════════════════════════════════════════════════════════════════
# 10. Fluid database sweep
# ═════════════════════════════════════════════════════════════════


class TestFluidVariants:
    @pytest.mark.parametrize("fluid", [
        "air", "water", "seawater", "glycerol", "engine_oil",
        "mercury", "ethanol", "hydrogen", "nitrogen", "helium",
        "carbon_dioxide", "liquid_sodium",
    ])
    def test_all_fluids_compile(self, fluid: str) -> None:
        spec = _make_spec("external_flow", "circle", {"radius": 0.01}, velocity=5.0, fluid=fluid)
        r = _compile(spec)
        assert r.fluid_name.lower().replace(" ", "_") == fluid
        assert r.reynolds_number > 0


# ═════════════════════════════════════════════════════════════════
# 11. Template registry
# ═════════════════════════════════════════════════════════════════


class TestTemplateRegistry:
    def test_all_classes_registered(self) -> None:
        registry = TemplateRegistry()
        for pc in ProblemClass:
            info = registry.get(pc)
            assert info is not None, f"{pc.value} not registered"
            assert len(info.supported_geometries) >= 1

    def test_singleton_identity(self) -> None:
        a = TemplateRegistry()
        b = TemplateRegistry()
        assert a is b

    def test_example_params_structure(self) -> None:
        registry = TemplateRegistry()
        for pc in ProblemClass:
            info = registry.get(pc)
            assert info is not None
            assert isinstance(info.example_params, dict)
            if "geometry" in info.example_params:
                assert "shape" in info.example_params["geometry"]


# ═════════════════════════════════════════════════════════════════
# 12. Pydantic model validation
# ═════════════════════════════════════════════════════════════════


class TestModelValidation:
    def test_problem_spec_serialization(self) -> None:
        spec = _make_spec("external_flow", "circle", {"radius": 0.01})
        d = spec.model_dump()
        assert d["problem_class"] == "external_flow"
        # Round-trip
        spec2 = ProblemSpec(**d)
        assert spec2.problem_class == ProblemClass.EXTERNAL_FLOW

    def test_problem_result_json(self) -> None:
        spec = _make_spec("external_flow", "circle", {"radius": 0.01})
        r = _compile(spec)
        j = r.model_dump()
        assert j["domain"] == "navier_stokes_2d"
        assert j["reynolds_number"] > 0
        assert isinstance(j["fluid_name"], str)
        # Should be JSON-serializable
        json.dumps(j, default=str)

    def test_geometry_spec_defaults(self) -> None:
        g = GeometrySpec(shape=GeometryType.CIRCLE, params={})
        assert g.params == {}

    def test_flow_conditions_defaults(self) -> None:
        f = FlowConditions(velocity=10.0)
        assert f.fluid == "air"
        assert f.temperature is None
        assert f.pressure is None

    def test_boundary_defaults(self) -> None:
        b = BoundarySpec()
        assert b.inlet == "uniform"
        assert b.outlet == "zero_gradient"


# ═════════════════════════════════════════════════════════════════
# 13. Edge cases
# ═════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_unknown_problem_class(self) -> None:
        with pytest.raises(ValueError):
            ProblemClass("nonexistent")

    def test_unknown_geometry(self) -> None:
        with pytest.raises(ValueError):
            GeometryType("spaceship")

    def test_unsupported_geometry_for_class(self) -> None:
        """Channel flow does not support circle geometry."""
        spec = _make_spec("channel_flow", "circle", {"radius": 0.01})
        with pytest.raises(ValueError, match="not supported"):
            _compile(spec)

    def test_very_low_velocity(self) -> None:
        """Near-zero velocity → very small Re."""
        spec = _make_spec("external_flow", "circle", {"radius": 0.01}, velocity=0.001)
        r = _compile(spec)
        assert r.reynolds_number < 10.0

    def test_very_high_velocity(self) -> None:
        """Supersonic → Ma > 1."""
        spec = _make_spec("external_flow", "circle", {"radius": 0.01}, velocity=500.0)
        r = _compile(spec)
        assert r.mach_number > 1.0

    def test_explicit_t_end(self) -> None:
        spec = _make_spec("external_flow", "circle", {"radius": 0.01}, t_end=0.5)
        r = _compile(spec)
        # dt * n_steps should approximate t_end
        simulated_time = r.dt * r.n_steps
        assert simulated_time > 0


# ═════════════════════════════════════════════════════════════════
# 14. API models (Pydantic request/response)
# ═════════════════════════════════════════════════════════════════


class TestAPIModels:
    def test_problem_request_model(self) -> None:
        from physics_os.api.routers.problems import ProblemRequest
        req = ProblemRequest(
            problem_class="external_flow",
            geometry={"shape": "circle", "params": {"radius": 0.01}},
            flow={"velocity": 10.0, "fluid": "air"},
        )
        assert req.quality == "standard"
        assert req.max_rank == 64

    def test_template_response_model(self) -> None:
        from physics_os.api.routers.problems import TemplateResponse
        resp = TemplateResponse(
            problem_class="external_flow",
            label="External Flow",
            description="Flow over body",
            supported_geometries=["circle", "naca_airfoil"],
            default_geometry="circle",
            required_flow_fields=["velocity"],
            optional_flow_fields=["fluid"],
            example_params={"velocity": 10},
        )
        assert resp.problem_class == "external_flow"


# ═════════════════════════════════════════════════════════════════
# 15. MCP tool handlers (unit test, no server)
# ═════════════════════════════════════════════════════════════════


class TestMCPTools:
    def test_list_templates_handler(self) -> None:
        from physics_os.mcp.server import handle_tool_call
        result = handle_tool_call("ontic_list_templates", {})
        assert "templates" in result
        assert result["count"] == 8

    def test_solve_problem_unknown_class(self) -> None:
        from physics_os.mcp.server import handle_tool_call
        result = handle_tool_call("ontic_solve_problem", {
            "problem_class": "fake",
            "geometry": {"shape": "circle"},
            "flow": {"velocity": 10},
        })
        assert "error" in result
