"""Tests for sim/fsi_valve.py — fluid-structure interaction nasal valve solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

from products.facial_plastics.sim.fsi_valve import (
    AIR_DENSITY_KG_M3,
    AIR_VISCOSITY_PA_S,
    CRITICAL_AREA_FRACTION,
    BeamProperties,
    FSIResult,
    FSIValveSolver,
    ValveGeometry,
    _beam_deflection,
    _compute_reduced_area,
    _convex_hull_area_2d,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def default_geometry() -> ValveGeometry:
    return ValveGeometry()


@pytest.fixture
def narrow_geometry() -> ValveGeometry:
    """Geometry with a narrow valve angle — collapse-prone."""
    return ValveGeometry(
        upper_lateral_length_mm=14.0,
        valve_angle_deg=8.0,
        cross_section_area_mm2=30.0,
        ulc_thickness_mm=0.5,
    )


@pytest.fixture
def wide_geometry() -> ValveGeometry:
    """Geometry with a wide valve — collapse-resistant."""
    return ValveGeometry(
        upper_lateral_length_mm=10.0,
        valve_angle_deg=18.0,
        cross_section_area_mm2=80.0,
        ulc_thickness_mm=1.2,
    )


@pytest.fixture
def beam() -> BeamProperties:
    return BeamProperties(
        length_mm=12.0,
        thickness_mm=0.8,
        width_mm=5.0,
        youngs_modulus_pa=5.0e6,
    )


@pytest.fixture
def solver() -> FSIValveSolver:
    return FSIValveSolver(
        max_fsi_iterations=30,
        n_breathing_frames=10,  # small for speed
    )


# ── ValveGeometry ─────────────────────────────────────────────────

class TestValveGeometry:
    def test_default_values(self, default_geometry: ValveGeometry) -> None:
        assert default_geometry.upper_lateral_length_mm == 12.0
        assert default_geometry.valve_angle_deg == 12.0
        assert default_geometry.cross_section_area_mm2 == 55.0

    def test_valve_angle_rad(self, default_geometry: ValveGeometry) -> None:
        expected = math.radians(12.0)
        assert default_geometry.valve_angle_rad == pytest.approx(expected, rel=1e-6)

    def test_hydraulic_diameter(self, default_geometry: ValveGeometry) -> None:
        dh = default_geometry.hydraulic_diameter_mm
        assert dh > 0
        # D_h = 4A/P should be reasonable
        assert 1.0 < dh < 20.0

    def test_to_dict(self, default_geometry: ValveGeometry) -> None:
        d = default_geometry.to_dict()
        assert "ulc_length_mm" in d
        assert "valve_angle_deg" in d
        assert "area_mm2" in d
        assert "hydraulic_diameter_mm" in d


# ── BeamProperties ────────────────────────────────────────────────

class TestBeamProperties:
    def test_second_moment(self, beam: BeamProperties) -> None:
        # I = bh³/12
        expected = 5.0 * 0.8 ** 3 / 12.0
        assert beam.second_moment_mm4 == pytest.approx(expected, rel=1e-6)

    def test_flexural_rigidity(self, beam: BeamProperties) -> None:
        EI = beam.flexural_rigidity_n_mm2
        assert EI > 0
        # E in N/mm² × I in mm⁴ = N·mm²
        E_n_mm2 = 5.0e6 * 1e-6  # = 5.0 N/mm²
        expected = E_n_mm2 * beam.second_moment_mm4
        assert EI == pytest.approx(expected, rel=1e-6)


# ── Beam deflection ───────────────────────────────────────────────

class TestBeamDeflection:
    def test_zero_pressure(self, beam: BeamProperties) -> None:
        defl = _beam_deflection(beam, 0.0)
        np.testing.assert_allclose(defl, 0.0, atol=1e-15)

    def test_positive_deflection(self, beam: BeamProperties) -> None:
        defl = _beam_deflection(beam, 100.0)
        assert len(defl) == 50
        # Max deflection at free end (last point)
        assert defl[-1] > 0

    def test_cantilever_boundary(self, beam: BeamProperties) -> None:
        defl = _beam_deflection(beam, 100.0, n_points=20)
        # Fixed end should have zero deflection
        assert defl[0] == pytest.approx(0.0, abs=1e-12)
        # Max at free end
        assert defl[-1] == max(defl)

    def test_deflection_scales_with_pressure(self, beam: BeamProperties) -> None:
        d1 = _beam_deflection(beam, 50.0)
        d2 = _beam_deflection(beam, 100.0)
        # Should scale linearly with load
        np.testing.assert_allclose(d2, 2.0 * d1, rtol=1e-10)


# ── Convex hull area ─────────────────────────────────────────────

class TestConvexHullArea:
    def test_triangle(self) -> None:
        pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        area = _convex_hull_area_2d(pts)
        assert area == pytest.approx(0.5, abs=0.01)

    def test_square(self) -> None:
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        area = _convex_hull_area_2d(pts)
        assert area == pytest.approx(1.0, abs=0.05)

    def test_too_few_points(self) -> None:
        pts = np.array([[0, 0], [1, 0]], dtype=np.float64)
        area = _convex_hull_area_2d(pts)
        assert area == 0.0


# ── Reduced area computation ─────────────────────────────────────

class TestComputeReducedArea:
    def test_no_deflection(self, default_geometry: ValveGeometry) -> None:
        area = _compute_reduced_area(default_geometry, np.zeros(10))
        assert area == pytest.approx(default_geometry.cross_section_area_mm2, rel=0.01)

    def test_deflection_reduces_area(self, default_geometry: ValveGeometry) -> None:
        defl = np.ones(10) * 1.0  # 1 mm deflection
        area = _compute_reduced_area(default_geometry, defl)
        assert area < default_geometry.cross_section_area_mm2

    def test_large_deflection_approaches_zero(self, default_geometry: ValveGeometry) -> None:
        defl = np.ones(10) * 50.0  # very large deflection
        area = _compute_reduced_area(default_geometry, defl)
        assert area == pytest.approx(0.0, abs=1e-6)


# ── FSIResult ─────────────────────────────────────────────────────

class TestFSIResult:
    def test_default_values(self) -> None:
        result = FSIResult()
        assert result.collapsed is False
        assert result.area_ratio == 1.0
        assert result.converged is False

    def test_summary(self) -> None:
        result = FSIResult(collapsed=True, area_ratio=0.2, peak_ulc_deflection_mm=2.5, safety_margin=0.3)
        s = result.summary()
        assert "COLLAPSED" in s

    def test_to_dict(self) -> None:
        result = FSIResult(resting_area_mm2=55.0)
        d = result.to_dict()
        assert d["resting_area_mm2"] == 55.0
        assert "converged" in d


# ── FSIValveSolver ────────────────────────────────────────────────

class TestFSIValveSolver:
    def test_instantiation(self, solver: FSIValveSolver) -> None:
        assert solver is not None

    def test_solve_default_geometry(
        self, solver: FSIValveSolver, default_geometry: ValveGeometry,
    ) -> None:
        result = solver.solve(default_geometry)
        assert isinstance(result, FSIResult)
        assert result.resting_area_mm2 > 0
        assert result.peak_area_mm2 >= 0  # may be zero if collapsed
        assert result.wall_clock_seconds >= 0
        assert result.n_fsi_iterations > 0

    def test_solve_convergence(
        self, solver: FSIValveSolver, wide_geometry: ValveGeometry,
    ) -> None:
        # Use resting (mild) breathing pressure for convergence test
        result = solver.solve(wide_geometry, peak_pressure_pa=-15.0)
        assert result.converged is True
        assert result.peak_area_mm2 > 0

    def test_narrow_valve_more_collapse_prone(
        self, solver: FSIValveSolver,
        narrow_geometry: ValveGeometry,
        wide_geometry: ValveGeometry,
    ) -> None:
        narrow_result = solver.solve(narrow_geometry)
        wide_result = solver.solve(wide_geometry)
        # Narrow valve should have lower area ratio (more collapse)
        assert narrow_result.area_ratio <= wide_result.area_ratio

    def test_breathing_cycle(
        self, solver: FSIValveSolver, default_geometry: ValveGeometry,
    ) -> None:
        result = solver.solve(default_geometry)
        assert result.n_breathing_frames == 10
        assert len(result.area_timeline) == 10
        assert len(result.pressure_timeline) == 10
        assert len(result.deflection_timeline) == 10

    def test_flow_metrics(
        self, solver: FSIValveSolver, default_geometry: ValveGeometry,
    ) -> None:
        result = solver.solve(default_geometry)
        assert result.mean_flow_rate_ml_s >= 0
        assert result.peak_flow_rate_ml_s >= result.mean_flow_rate_ml_s

    def test_collapse_pressure_detection(
        self, solver: FSIValveSolver, narrow_geometry: ValveGeometry,
    ) -> None:
        result = solver.solve(narrow_geometry)
        # Should either find a collapse pressure or report 0.0 (never collapses)
        assert isinstance(result.collapse_pressure_pa, float)

    def test_safety_margin(
        self, solver: FSIValveSolver, wide_geometry: ValveGeometry,
    ) -> None:
        result = solver.solve(wide_geometry)
        assert result.safety_margin >= 0

    def test_surgical_effect(
        self, solver: FSIValveSolver,
        narrow_geometry: ValveGeometry,
    ) -> None:
        wide_postop = ValveGeometry(
            upper_lateral_length_mm=12.0,
            valve_angle_deg=15.0,
            cross_section_area_mm2=60.0,
            ulc_thickness_mm=1.0,
        )
        comparison = solver.evaluate_surgical_effect(
            narrow_geometry, wide_postop,
            spreader_graft=True,
        )
        assert "preop" in comparison
        assert "postop" in comparison
        assert "improvement" in comparison
        # With spreader graft, postop should improve
        improvement = comparison["improvement"]
        assert improvement["area_ratio_change"] > 0 or not comparison["preop"]["collapsed"]

    def test_stiffness_override(
        self, solver: FSIValveSolver, default_geometry: ValveGeometry,
    ) -> None:
        # Increasing stiffness should reduce deflection
        result_default = solver.solve(default_geometry)
        result_stiff = solver.solve(
            default_geometry,
            ulc_E_pa=20.0e6,
            ulc_thickness_mm=2.0,
        )
        assert result_stiff.peak_ulc_deflection_mm <= result_default.peak_ulc_deflection_mm
