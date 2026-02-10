"""Tests for the functional outcome metrics module."""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from products.facial_plastics.core.types import (
    ClinicalMeasurement,
    LandmarkType,
    MaterialModel,
    MeshElementType,
    StructureType,
    TissueProperties,
    Vec3,
    VolumeMesh,
)
from products.facial_plastics.metrics.functional import (
    CottleArea,
    FlowDistribution,
    FunctionalMetrics,
    FunctionalReport,
    NORMAL_RESISTANCE_MAX,
    NORMAL_RESISTANCE_MIN,
    NORMAL_VALVE_AREA_MAX_MM2,
    NORMAL_VALVE_AREA_MIN_MM2,
    NORMAL_WSS_MAX_PA,
    NORMAL_WSS_MIN_PA,
    OBSTRUCTION_THRESHOLD,
    RE_LAMINAR_MAX,
    RE_TURBULENT,
    ResistanceProfile,
    ValveGeometry,
    WSSAnalysis,
)
from products.facial_plastics.sim.cfd_airway import AirwayCFDResult


# ═══════════════════════════════════════════════════════════════════
#  Helpers — synthetic CFD results
# ═══════════════════════════════════════════════════════════════════

def _make_cfd_result(
    *,
    nx: int = 10,
    ny: int = 10,
    nz: int = 20,
    n_sections: int = 20,
    resistance: float = 1.5,
    flow_rate_ml_s: float = 250.0,
    pressure_drop_pa: float = 15.0,
    max_velocity: float = 3.0,
    mean_velocity: float = 1.0,
    reynolds: float = 800.0,
    n_wss: int = 100,
    wss_mean: float = 0.5,
    wss_std: float = 0.3,
    converged: bool = True,
) -> AirwayCFDResult:
    """Create a synthetic AirwayCFDResult for testing."""
    rng = np.random.default_rng(42)

    vx = rng.standard_normal((nx, ny, nz)) * 0.1
    vy = rng.standard_normal((nx, ny, nz)) * 0.1
    vz = rng.standard_normal((nx, ny, nz)) * mean_velocity
    p = np.linspace(pressure_drop_pa, 0, nz)[np.newaxis, np.newaxis, :]
    p = np.broadcast_to(p, (nx, ny, nz)).copy()

    wss = np.abs(rng.normal(wss_mean, wss_std, size=n_wss))

    section_velocities = np.full(n_sections, mean_velocity, dtype=np.float64)
    section_flow_rates = np.full(n_sections, flow_rate_ml_s / n_sections, dtype=np.float64)

    return AirwayCFDResult(
        velocity_x=vx,
        velocity_y=vy,
        velocity_z=vz,
        pressure=p,
        wall_shear_stress=wss,
        nasal_resistance_pa_s_ml=resistance,
        total_flow_rate_ml_s=flow_rate_ml_s,
        pressure_drop_pa=pressure_drop_pa,
        max_velocity_m_s=max_velocity,
        mean_velocity_m_s=mean_velocity,
        max_wall_shear_pa=float(np.max(wss)) if len(wss) > 0 else 0.0,
        mean_wall_shear_pa=float(np.mean(wss)) if len(wss) > 0 else 0.0,
        reynolds_number=reynolds,
        valve_velocity_m_s=max_velocity * 0.8,
        section_flow_rates=section_flow_rates,
        section_velocities=section_velocities,
        converged=converged,
        n_iterations=100,
        wall_clock_seconds=1.5,
    )


def _make_simple_volume_mesh() -> VolumeMesh:
    """Small volume mesh for metric computation."""
    n = 4
    nodes: list[list[float]] = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                nodes.append([i * 5.0, j * 5.0, k * 5.0])
    elements: list[list[int]] = []
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                c = [
                    i * n * n + j * n + k,
                    i * n * n + j * n + (k + 1),
                    i * n * n + (j + 1) * n + k,
                    (i + 1) * n * n + j * n + k,
                ]
                elements.append(c)
    return VolumeMesh(
        nodes=np.array(nodes, dtype=np.float64),
        elements=np.array(elements, dtype=np.int64),
        element_type=MeshElementType.TET4,
        region_ids=np.zeros(len(elements), dtype=np.int32),
        region_materials={
            0: TissueProperties(
                structure_type=StructureType.AIRWAY_NASAL,
                material_model=MaterialModel.RIGID,
                parameters={},
            ),
        },
    )


def _make_functional_landmarks() -> dict[LandmarkType, Vec3]:
    return {
        LandmarkType.TIP_DEFINING_POINT_LEFT: Vec3(-3.0, 5.0, 34.0),
        LandmarkType.TIP_DEFINING_POINT_RIGHT: Vec3(3.0, 5.0, 34.0),
        LandmarkType.INTERNAL_VALVE_LEFT: Vec3(-8.0, 10.0, 20.0),
        LandmarkType.INTERNAL_VALVE_RIGHT: Vec3(8.0, 10.0, 20.0),
        LandmarkType.EXTERNAL_VALVE_LEFT: Vec3(-12.0, 5.0, 18.0),
        LandmarkType.EXTERNAL_VALVE_RIGHT: Vec3(12.0, 5.0, 18.0),
        LandmarkType.NASION: Vec3(0.0, 35.0, 15.0),
        LandmarkType.PRONASALE: Vec3(0.0, 5.0, 35.0),
        LandmarkType.SUBNASALE: Vec3(0.0, 0.0, 25.0),
    }


# ═══════════════════════════════════════════════════════════════════
#  ResistanceProfile tests
# ═══════════════════════════════════════════════════════════════════

class TestResistanceProfile:
    """Test nasal resistance computation and classification."""

    def test_normal_resistance_classified(self) -> None:
        rp = ResistanceProfile(total_resistance=1.5)
        level = rp.classify()
        assert level == "normal"

    def test_mild_obstruction(self) -> None:
        rp = ResistanceProfile(total_resistance=2.8)
        assert rp.classify() == "mild"

    def test_moderate_obstruction(self) -> None:
        rp = ResistanceProfile(total_resistance=3.2)
        assert rp.classify() == "moderate"

    def test_severe_obstruction(self) -> None:
        rp = ResistanceProfile(total_resistance=4.0)
        assert rp.classify() == "severe"

    def test_bottleneck_area_identified(self) -> None:
        rp = ResistanceProfile(
            total_resistance=3.0,
            per_area={
                CottleArea.AREA_1: 0.5,
                CottleArea.AREA_2: 1.5,
                CottleArea.AREA_3: 0.3,
                CottleArea.AREA_4: 0.4,
                CottleArea.AREA_5: 0.3,
            },
        )
        rp.classify()
        assert rp.bottleneck_area == CottleArea.AREA_2


# ═══════════════════════════════════════════════════════════════════
#  ValveGeometry tests
# ═══════════════════════════════════════════════════════════════════

class TestValveGeometry:
    """Test valve geometry metrics."""

    def test_stenotic_detection(self) -> None:
        vg = ValveGeometry(internal_valve_area_mm2=30.0)
        assert vg.is_stenotic() is True

    def test_normal_valve_not_stenotic(self) -> None:
        vg = ValveGeometry(internal_valve_area_mm2=50.0)
        assert vg.is_stenotic() is False

    def test_borderline_valve(self) -> None:
        vg = ValveGeometry(internal_valve_area_mm2=NORMAL_VALVE_AREA_MIN_MM2)
        assert vg.is_stenotic() is False  # >= threshold


# ═══════════════════════════════════════════════════════════════════
#  FlowDistribution tests
# ═══════════════════════════════════════════════════════════════════

class TestFlowDistribution:
    """Test flow distribution analysis."""

    def test_balanced_flow(self) -> None:
        fd = FlowDistribution(
            left_flow_ml_s=130.0, right_flow_ml_s=120.0,
            total_flow_ml_s=250.0,
            left_fraction=0.52, right_fraction=0.48,
            flow_asymmetry_pct=4.0,
        )
        assert fd.is_balanced() is True

    def test_unbalanced_flow(self) -> None:
        fd = FlowDistribution(
            left_flow_ml_s=200.0, right_flow_ml_s=50.0,
            total_flow_ml_s=250.0,
            left_fraction=0.8, right_fraction=0.2,
            flow_asymmetry_pct=60.0,
        )
        assert fd.is_balanced() is False

    def test_custom_threshold(self) -> None:
        fd = FlowDistribution(flow_asymmetry_pct=25.0)
        assert fd.is_balanced(threshold_pct=30.0) is True
        assert fd.is_balanced(threshold_pct=20.0) is False


# ═══════════════════════════════════════════════════════════════════
#  WSSAnalysis tests
# ═══════════════════════════════════════════════════════════════════

class TestWSSAnalysis:
    """Test wall shear stress analysis."""

    def test_normal_wss(self) -> None:
        wa = WSSAnalysis(
            max_wss_pa=1.5, mean_wss_pa=0.5, std_wss_pa=0.2,
            high_wss_area_fraction=0.05, low_wss_area_fraction=0.10,
        )
        assert wa.has_abnormal_wss() is False

    def test_high_wss_abnormal(self) -> None:
        wa = WSSAnalysis(high_wss_area_fraction=0.15, low_wss_area_fraction=0.05)
        assert wa.has_abnormal_wss() is True

    def test_stagnation_abnormal(self) -> None:
        wa = WSSAnalysis(high_wss_area_fraction=0.0, low_wss_area_fraction=0.35)
        assert wa.has_abnormal_wss() is True


# ═══════════════════════════════════════════════════════════════════
#  FunctionalMetrics.evaluate tests
# ═══════════════════════════════════════════════════════════════════

class TestFunctionalMetrics:
    """Test the full FunctionalMetrics.evaluate pipeline."""

    @pytest.fixture()
    def metrics(self) -> FunctionalMetrics:
        mesh = _make_simple_volume_mesh()
        landmarks = _make_functional_landmarks()
        return FunctionalMetrics(mesh, landmarks)

    @pytest.fixture()
    def normal_cfd(self) -> AirwayCFDResult:
        return _make_cfd_result(
            resistance=1.5, flow_rate_ml_s=250.0, reynolds=800.0,
        )

    @pytest.fixture()
    def obstructed_cfd(self) -> AirwayCFDResult:
        return _make_cfd_result(
            resistance=4.0, flow_rate_ml_s=80.0, reynolds=300.0,
        )

    def test_evaluate_returns_report(
        self, metrics: FunctionalMetrics, normal_cfd: AirwayCFDResult,
    ) -> None:
        report = metrics.evaluate(normal_cfd)
        assert isinstance(report, FunctionalReport)

    def test_resistance_in_report(
        self, metrics: FunctionalMetrics, normal_cfd: AirwayCFDResult,
    ) -> None:
        report = metrics.evaluate(normal_cfd)
        assert report.resistance.total_resistance == pytest.approx(1.5)
        assert report.resistance.obstruction_level == "normal"

    def test_obstructed_resistance(
        self, metrics: FunctionalMetrics, obstructed_cfd: AirwayCFDResult,
    ) -> None:
        report = metrics.evaluate(obstructed_cfd)
        assert report.resistance.total_resistance == pytest.approx(4.0)
        assert report.resistance.obstruction_level == "severe"

    def test_flow_distribution_populated(
        self, metrics: FunctionalMetrics, normal_cfd: AirwayCFDResult,
    ) -> None:
        report = metrics.evaluate(normal_cfd)
        assert report.flow.total_flow_ml_s == pytest.approx(250.0)
        assert report.flow.left_fraction + report.flow.right_fraction == pytest.approx(1.0, abs=1e-6)

    def test_wss_computed(
        self, metrics: FunctionalMetrics, normal_cfd: AirwayCFDResult,
    ) -> None:
        report = metrics.evaluate(normal_cfd)
        assert report.wss.max_wss_pa > 0
        assert report.wss.mean_wss_pa > 0

    def test_flow_regime_laminar(
        self, metrics: FunctionalMetrics, normal_cfd: AirwayCFDResult,
    ) -> None:
        report = metrics.evaluate(normal_cfd)
        assert report.flow_regime == "laminar"
        assert report.reynolds_number == pytest.approx(800.0)

    def test_flow_regime_turbulent(self, metrics: FunctionalMetrics) -> None:
        cfd = _make_cfd_result(reynolds=5000.0)
        report = metrics.evaluate(cfd)
        assert report.flow_regime == "turbulent"

    def test_flow_regime_transitional(self, metrics: FunctionalMetrics) -> None:
        cfd = _make_cfd_result(reynolds=3000.0)
        report = metrics.evaluate(cfd)
        assert report.flow_regime == "transitional"


# ═══════════════════════════════════════════════════════════════════
#  NOSE score prediction tests
# ═══════════════════════════════════════════════════════════════════

class TestNoseScorePrediction:
    """Test the NOSE score regression model."""

    @pytest.fixture()
    def metrics(self) -> FunctionalMetrics:
        mesh = _make_simple_volume_mesh()
        landmarks = _make_functional_landmarks()
        return FunctionalMetrics(mesh, landmarks)

    def test_nose_score_normal_low(self, metrics: FunctionalMetrics) -> None:
        """Normal resistance → low NOSE (less symptoms)."""
        cfd = _make_cfd_result(resistance=1.0, flow_rate_ml_s=300.0, reynolds=900.0)
        report = metrics.evaluate(cfd)
        assert 0 <= report.predicted_nose_score <= 100
        # Normal case should have low NOSE score
        assert report.predicted_nose_score < 50

    def test_nose_score_obstructed_high(self, metrics: FunctionalMetrics) -> None:
        """Severe obstruction → high NOSE score."""
        cfd = _make_cfd_result(resistance=5.0, flow_rate_ml_s=50.0, reynolds=200.0)
        report = metrics.evaluate(cfd)
        assert report.predicted_nose_score > 30  # should be elevated

    def test_nose_score_bounded(self, metrics: FunctionalMetrics) -> None:
        """NOSE score always in [0, 100]."""
        for r in [0.5, 1.0, 2.0, 5.0, 10.0]:
            cfd = _make_cfd_result(resistance=r)
            report = metrics.evaluate(cfd)
            assert 0 <= report.predicted_nose_score <= 100


# ═══════════════════════════════════════════════════════════════════
#  FunctionalReport overall score tests
# ═══════════════════════════════════════════════════════════════════

class TestFunctionalReport:
    """Test the composite FunctionalReport methods."""

    def test_overall_score_perfect(self) -> None:
        """Perfect metrics → near-100 score."""
        report = FunctionalReport(
            resistance=ResistanceProfile(total_resistance=1.5),
            valve=ValveGeometry(internal_valve_area_mm2=50.0),
            flow=FlowDistribution(flow_asymmetry_pct=0.0),
            wss=WSSAnalysis(high_wss_area_fraction=0.0, low_wss_area_fraction=0.0),
            flow_regime="laminar",
            reynolds_number=800.0,
        )
        score = report.compute_overall()
        assert score >= 90.0

    def test_overall_score_obstructed(self) -> None:
        """Severely obstructed → low score."""
        report = FunctionalReport(
            resistance=ResistanceProfile(total_resistance=6.0),
            valve=ValveGeometry(internal_valve_area_mm2=15.0),
            flow=FlowDistribution(flow_asymmetry_pct=50.0),
            wss=WSSAnalysis(high_wss_area_fraction=0.3, low_wss_area_fraction=0.4),
            flow_regime="turbulent",
            reynolds_number=5000.0,
        )
        score = report.compute_overall()
        assert score < 50.0

    def test_to_dict_keys(self) -> None:
        report = FunctionalReport()
        report.compute_overall()
        d = report.to_dict()
        assert "overall_score" in d
        assert "resistance" in d
        assert "valve" in d
        assert "flow" in d
        assert "wss" in d
        assert "flow_regime" in d

    def test_summary_string(self) -> None:
        report = FunctionalReport(
            resistance=ResistanceProfile(total_resistance=1.5, obstruction_level="normal"),
            valve=ValveGeometry(internal_valve_area_mm2=50.0),
            flow=FlowDistribution(total_flow_ml_s=250.0),
            reynolds_number=800.0,
            flow_regime="laminar",
            predicted_nose_score=20.0,
        )
        report.compute_overall()
        s = report.summary()
        assert "1.50" in s or "1.5" in s  # resistance
        assert "normal" in s
        assert "laminar" in s

    def test_measurements_built(self) -> None:
        mesh = _make_simple_volume_mesh()
        landmarks = _make_functional_landmarks()
        fm = FunctionalMetrics(mesh, landmarks)
        cfd = _make_cfd_result()
        report = fm.evaluate(cfd)
        assert len(report.measurements) > 0
        names = {m.name for m in report.measurements}
        assert "nasal_resistance" in names
        assert "total_flow_rate" in names
        assert "predicted_nose_score" in names


# ═══════════════════════════════════════════════════════════════════
#  Improvement computation tests
# ═══════════════════════════════════════════════════════════════════

class TestImprovementComputation:
    """Test compute_improvement static method."""

    def test_improvement_direction(self) -> None:
        preop = FunctionalReport(
            resistance=ResistanceProfile(total_resistance=4.0),
            valve=ValveGeometry(internal_valve_area_mm2=25.0),
            flow=FlowDistribution(total_flow_ml_s=100.0, flow_asymmetry_pct=40.0),
            predicted_nose_score=70.0,
        )
        preop.overall_score = 40.0

        postop = FunctionalReport(
            resistance=ResistanceProfile(total_resistance=1.5),
            valve=ValveGeometry(internal_valve_area_mm2=50.0),
            flow=FlowDistribution(total_flow_ml_s=250.0, flow_asymmetry_pct=10.0),
            predicted_nose_score=20.0,
        )
        postop.overall_score = 85.0

        imp = FunctionalMetrics.compute_improvement(preop, postop)

        assert imp["resistance_change"] < 0  # resistance decreased (better)
        assert imp["resistance_improvement_pct"] > 0
        assert imp["valve_area_change_mm2"] > 0  # area increased
        assert imp["flow_improvement_ml_s"] > 0  # more flow
        assert imp["symmetry_improvement_pct"] > 0  # more symmetric
        assert imp["nose_score_improvement"] > 0  # NOSE decreased (better)
        assert imp["overall_score_improvement"] > 0

    def test_no_change(self) -> None:
        report = FunctionalReport(
            resistance=ResistanceProfile(total_resistance=2.0),
            valve=ValveGeometry(internal_valve_area_mm2=45.0),
            flow=FlowDistribution(total_flow_ml_s=200.0, flow_asymmetry_pct=10.0),
            predicted_nose_score=30.0,
        )
        report.overall_score = 70.0

        imp = FunctionalMetrics.compute_improvement(report, report)
        assert imp["resistance_change"] == pytest.approx(0.0)
        assert imp["valve_area_change_mm2"] == pytest.approx(0.0)
        assert imp["overall_score_improvement"] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════
#  Per-area (Cottle) resistance tests
# ═══════════════════════════════════════════════════════════════════

class TestCottleAreaResistance:
    """Test per-area resistance computation."""

    def test_cottle_areas_populated(self) -> None:
        mesh = _make_simple_volume_mesh()
        landmarks = _make_functional_landmarks()
        fm = FunctionalMetrics(mesh, landmarks)
        # Need enough sections for 5 Cottle areas
        cfd = _make_cfd_result(n_sections=25)
        report = fm.evaluate(cfd)
        if report.resistance.per_area:
            assert len(report.resistance.per_area) == 5
            for area_name, r_val in report.resistance.per_area.items():
                assert np.isfinite(r_val)

    def test_empty_wss_handled(self) -> None:
        """CFD result with zero WSS entries doesn't crash."""
        mesh = _make_simple_volume_mesh()
        landmarks = _make_functional_landmarks()
        fm = FunctionalMetrics(mesh, landmarks)
        cfd = _make_cfd_result(n_wss=0)
        # Manually set empty WSS
        cfd_mod = AirwayCFDResult(
            velocity_x=cfd.velocity_x, velocity_y=cfd.velocity_y,
            velocity_z=cfd.velocity_z, pressure=cfd.pressure,
            wall_shear_stress=np.zeros(0),
            nasal_resistance_pa_s_ml=cfd.nasal_resistance_pa_s_ml,
            total_flow_rate_ml_s=cfd.total_flow_rate_ml_s,
            pressure_drop_pa=cfd.pressure_drop_pa,
            max_velocity_m_s=cfd.max_velocity_m_s,
            mean_velocity_m_s=cfd.mean_velocity_m_s,
            max_wall_shear_pa=0.0, mean_wall_shear_pa=0.0,
            reynolds_number=cfd.reynolds_number,
            valve_velocity_m_s=cfd.valve_velocity_m_s,
            section_flow_rates=cfd.section_flow_rates,
            section_velocities=cfd.section_velocities,
            converged=True, n_iterations=50, wall_clock_seconds=1.0,
        )
        report = fm.evaluate(cfd_mod)
        assert report.wss.max_wss_pa == 0.0
