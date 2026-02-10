"""Tests for the simulation sub-package."""

from __future__ import annotations

import numpy as np
import pytest

from products.facial_plastics.core.types import (
    MeshElementType,
    StructureType,
    VolumeMesh,
)
from products.facial_plastics.sim.fem_soft_tissue import (
    FEMResult,
    SoftTissueFEM,
    TISSUE_PARAMS,
)
from products.facial_plastics.sim.cartilage import (
    CARTILAGE_LIBRARY,
    CartilageParams,
    CartilageSolver,
)
from products.facial_plastics.sim.sutures import (
    SutureMaterial,
    SutureElement,
    SutureSystem,
)
from products.facial_plastics.sim.cfd_airway import (
    AirwayCFDResult,
    AirwayCFDSolver,
)
from products.facial_plastics.sim.healing import (
    HealingModel,
    HealingPhase,
    HealingState,
    NASAL_HEALING_PHASES,
)
from products.facial_plastics.sim.orchestrator import (
    SimOrchestrator,
    SimulationResult,
)
from products.facial_plastics.tests.conftest import make_volume_mesh


# ── Tissue Parameters ────────────────────────────────────────────

class TestTissueParams:
    """Test tissue parameter database."""

    def test_tissue_params_populated(self) -> None:
        assert len(TISSUE_PARAMS) >= 10

    def test_tissue_params_keys_are_structure_type(self) -> None:
        for key in TISSUE_PARAMS:
            assert isinstance(key, StructureType)

    def test_tissue_params_values(self) -> None:
        for st, params in TISSUE_PARAMS.items():
            assert params["mu"] > 0, f"{st}: mu must be positive"
            assert params["kappa"] > 0, f"{st}: kappa must be positive"
            assert params["density"] > 0, f"{st}: density must be positive"


# ── Cartilage ────────────────────────────────────────────────────

class TestCartilageLibrary:
    """Test cartilage material library."""

    def test_library_entries(self) -> None:
        assert StructureType.CARTILAGE_UPPER_LATERAL in CARTILAGE_LIBRARY
        assert StructureType.CARTILAGE_LOWER_LATERAL in CARTILAGE_LIBRARY
        assert StructureType.CARTILAGE_SEPTUM in CARTILAGE_LIBRARY

    def test_params_valid(self) -> None:
        for st, params in CARTILAGE_LIBRARY.items():
            assert isinstance(params, CartilageParams)
            assert params.E_fiber > 0
            assert 0 < params.nu < 0.5
            assert params.thickness_mm > 0

    def test_cartilage_params_properties(self) -> None:
        params = CARTILAGE_LIBRARY[StructureType.CARTILAGE_SEPTUM]
        assert params.E_eff > 0
        assert params.mu > 0
        assert params.kappa > 0


# ── Suture Materials ─────────────────────────────────────────────

class TestSutureElement:
    """Test suture element construction."""

    def test_construction(self) -> None:
        elem = SutureElement(
            suture_id="s001",
            node_a=0,
            node_b=1,
            material=SutureMaterial.PDS,
            gauge="5-0",
        )
        assert elem.material == SutureMaterial.PDS
        assert elem.node_a == 0

    def test_default_material(self) -> None:
        elem = SutureElement(
            suture_id="s002",
            node_a=5,
            node_b=10,
        )
        assert elem.material == SutureMaterial.PDS


class TestSutureSystem:
    """Test suture system construction."""

    def test_add_suture(self) -> None:
        system = SutureSystem()
        elem = SutureElement(
            suture_id="s001",
            node_a=0,
            node_b=10,
            material=SutureMaterial.PDS,
        )
        system.add_suture(elem)
        assert system.n_sutures == 1

    def test_multiple_sutures(self) -> None:
        system = SutureSystem()
        for i in range(5):
            elem = SutureElement(
                suture_id=f"s{i:03d}",
                node_a=i,
                node_b=i + 10,
                material=SutureMaterial.NYLON,
            )
            system.add_suture(elem)
        assert system.n_sutures == 5


# ── Healing Model ────────────────────────────────────────────────

class TestHealingPhases:
    """Test healing phase definitions."""

    def test_phases_count(self) -> None:
        phases = NASAL_HEALING_PHASES
        assert len(phases) >= 3

    def test_phases_ordered_by_onset(self) -> None:
        phases = NASAL_HEALING_PHASES
        for i in range(len(phases) - 1):
            assert phases[i].onset_days < phases[i + 1].onset_days

    def test_phase_is_frozen_dataclass(self) -> None:
        phase = NASAL_HEALING_PHASES[0]
        assert isinstance(phase, HealingPhase)
        with pytest.raises(AttributeError):
            phase.name = "tampered"  # type: ignore[misc]


class TestHealingModel:
    """Test healing model evolution."""

    def test_compute_state(self) -> None:
        mesh = make_volume_mesh()
        model = HealingModel(mesh=mesh)
        state = model.compute_state(time_days=0.0)
        assert isinstance(state, HealingState)

    def test_compute_timeline(self) -> None:
        mesh = make_volume_mesh()
        model = HealingModel(mesh=mesh)
        time_points = [0.0, 7.0, 30.0, 90.0, 365.0]
        timeline = model.compute_timeline(time_points)
        assert len(timeline) == len(time_points)


# ── FEMResult ────────────────────────────────────────────────────

class TestFEMResult:
    """Test FEM result dataclass."""

    def test_basic_result(self) -> None:
        n_nodes = 10
        result = FEMResult(
            displacements=np.zeros((n_nodes, 3)),
            stresses=np.zeros((n_nodes, 6)),
            strains=np.zeros((n_nodes, 6)),
            reaction_forces=np.zeros((n_nodes, 3)),
            internal_energy=0.0,
            n_iterations=5,
            n_load_steps=1,
            converged=True,
            max_displacement_mm=0.0,
            max_von_mises_stress=0.0,
            max_principal_strain=0.0,
            wall_clock_seconds=0.1,
            residual_history=[1e-2, 1e-4, 1e-8],
        )
        assert result.converged is True
        assert result.max_displacement_mm == 0.0


# ── SimulationResult ─────────────────────────────────────────────

class TestSimulationResult:
    """Test SimulationResult dataclass."""

    def test_default_fields(self) -> None:
        result = SimulationResult()
        summary = result.summary()
        assert isinstance(summary, str)

    def test_properties(self) -> None:
        result = SimulationResult()
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.fem_converged, bool)
        assert isinstance(result.cfd_converged, bool)
