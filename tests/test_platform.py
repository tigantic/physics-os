"""
Platform substrate integration tests.

Tests the canonical interfaces, data model, solver orchestration,
domain-pack registry, reproducibility, and checkpointing.

Runs as part of the standard test suite:  pytest tests/test_platform.py
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence, Type

import pytest
import torch
from torch import Tensor

from ontic.platform.checkpoint import load_checkpoint, save_checkpoint
from ontic.platform.data_model import (
    BCType,
    BoundaryCondition,
    FieldData,
    InitialCondition,
    Mesh,
    SimulationState,
    StructuredMesh,
    UnstructuredMesh,
)
from ontic.platform.domain_pack import DomainPack, DomainRegistry, get_registry
from ontic.platform.protocols import (
    Discretization,
    Observable,
    OperatorProto,
    ProblemSpec,
    Solver,
    SolveResult,
    Workflow,
)
from ontic.platform.reproduce import (
    ArtifactHash,
    ReproducibilityContext,
    capture_environment,
    hash_tensor,
    lock_seeds,
)
from ontic.platform.solvers import (
    ConjugateGradient,
    ForwardEuler,
    GMRES,
    LinearSolveResult,
    NewtonSolver,
    NonlinearSolveResult,
    PicardSolver,
    RK4,
    StormerVerlet,
    SymplecticEuler,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mesh_1d():
    return StructuredMesh(shape=(100,), domain=((0.0, 1.0),))


@pytest.fixture
def mesh_2d():
    return StructuredMesh(shape=(32, 32), domain=((0.0, 1.0), (0.0, 1.0)))


@pytest.fixture
def simple_state(mesh_1d):
    u = FieldData(
        name="u",
        data=torch.sin(math.pi * torch.linspace(0.005, 0.995, 100, dtype=torch.float64)),
        mesh=mesh_1d,
    )
    return SimulationState(t=0.0, fields={"u": u}, mesh=mesh_1d)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure each test gets a clean registry."""
    DomainRegistry.reset()
    yield
    DomainRegistry.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# Data Model
# ═══════════════════════════════════════════════════════════════════════════════


class TestStructuredMesh:
    def test_1d(self):
        m = StructuredMesh(shape=(10,), domain=((0.0, 1.0),))
        assert m.ndim == 1
        assert m.n_cells == 10
        assert len(m.dx) == 1
        assert abs(m.dx[0] - 0.1) < 1e-12
        vols = m.cell_volumes()
        assert vols.shape == (10,)
        assert torch.allclose(vols, torch.full((10,), 0.1, dtype=torch.float64))

    def test_2d(self):
        m = StructuredMesh(shape=(4, 8), domain=((0.0, 2.0), (0.0, 4.0)))
        assert m.ndim == 2
        assert m.n_cells == 32
        centers = m.cell_centers()
        assert centers.shape == (32, 2)

    def test_cell_centers_1d(self):
        m = StructuredMesh(shape=(4,), domain=((0.0, 1.0),))
        c = m.cell_centers()
        expected = torch.tensor([[0.125], [0.375], [0.625], [0.875]], dtype=torch.float64)
        assert torch.allclose(c, expected)


class TestUnstructuredMesh:
    def test_triangle(self):
        nodes = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float64)
        elements = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        m = UnstructuredMesh(nodes, elements)
        assert m.ndim == 2
        assert m.n_cells == 2
        vols = m.cell_volumes()
        assert vols.shape == (2,)
        assert torch.allclose(vols, torch.tensor([0.5, 0.5], dtype=torch.float64))


class TestFieldData:
    def test_basic(self, mesh_1d):
        data = torch.randn(100, dtype=torch.float64)
        f = FieldData(name="u", data=data, mesh=mesh_1d)
        assert f.name == "u"
        assert f.norm().item() > 0

    def test_clone(self, mesh_1d):
        data = torch.randn(100, dtype=torch.float64)
        f = FieldData(name="u", data=data, mesh=mesh_1d)
        c = f.clone()
        c.data[0] = 999.0
        assert f.data[0] != 999.0

    def test_shape_mismatch(self, mesh_1d):
        with pytest.raises(ValueError, match="length"):
            FieldData(name="u", data=torch.randn(50), mesh=mesh_1d)


class TestSimulationState:
    def test_advance(self, simple_state):
        new_u = simple_state.get_field("u").clone()
        s2 = simple_state.advance(0.1, {"u": new_u})
        assert abs(s2.t - 0.1) < 1e-15
        assert s2.step_index == 1

    def test_with_fields(self, simple_state, mesh_1d):
        v = FieldData(name="v", data=torch.zeros(100, dtype=torch.float64), mesh=mesh_1d)
        s2 = simple_state.with_fields(v=v)
        assert "v" in s2.fields
        assert "u" in s2.fields


class TestBoundaryCondition:
    def test_dirichlet_1d(self, mesh_1d):
        data = torch.ones(100, dtype=torch.float64)
        f = FieldData(name="u", data=data, mesh=mesh_1d)
        bc = BoundaryCondition(
            field_name="u", region="left", bc_type=BCType.DIRICHLET, value=0.0
        )
        f2 = bc.apply(f, mesh_1d)
        assert f2.data[0].item() == 0.0
        assert f2.data[50].item() == 1.0


class TestInitialCondition:
    def test_uniform(self, mesh_1d):
        ic = InitialCondition(field_name="u", ic_type="uniform", value=5.0)
        f = ic.generate(mesh_1d)
        assert f.name == "u"
        assert torch.allclose(f.data, torch.full((100,), 5.0, dtype=torch.float64))

    def test_gaussian(self, mesh_1d):
        ic = InitialCondition(
            field_name="u", ic_type="gaussian",
            metadata={"mu": 0.5, "sigma": 0.1},
        )
        f = ic.generate(mesh_1d)
        assert f.data.max().item() > 0.9  # peak near 1

    def test_function(self, mesh_1d):
        ic = InitialCondition(
            field_name="u", ic_type="function",
            function=lambda x: torch.sin(math.pi * x.squeeze()),
        )
        f = ic.generate(mesh_1d)
        assert f.data.shape == (100,)


# ═══════════════════════════════════════════════════════════════════════════════
# Protocols
# ═══════════════════════════════════════════════════════════════════════════════


class TestProtocols:
    """Ensure protocol checking works via isinstance / structural conformance."""

    def test_problem_spec(self):
        from ontic.platform.vertical_ode import HarmonicOscillatorSpec
        spec = HarmonicOscillatorSpec()
        assert isinstance(spec, ProblemSpec)

    def test_observable(self):
        from ontic.platform.vertical_ode import EnergyObservable, HarmonicOscillatorSpec
        obs = EnergyObservable(HarmonicOscillatorSpec())
        assert isinstance(obs, Observable)


# ═══════════════════════════════════════════════════════════════════════════════
# Time Integrators
# ═══════════════════════════════════════════════════════════════════════════════


class TestForwardEuler:
    def test_exponential_decay(self):
        """du/dt = -u, exact: u(t) = exp(-t)."""
        mesh = StructuredMesh(shape=(1,), domain=((0.0, 1.0),))
        u0 = FieldData(name="u", data=torch.tensor([1.0], dtype=torch.float64), mesh=mesh)
        state = SimulationState(t=0.0, fields={"u": u0}, mesh=mesh)

        def rhs(s, t):
            return {"u": -s.get_field("u").data}

        result = ForwardEuler().solve(state, rhs, t_span=(0.0, 1.0), dt=0.001)
        exact = math.exp(-1.0)
        assert abs(result.final_state.get_field("u").data.item() - exact) < 0.01


class TestRK4:
    def test_exponential_decay(self):
        mesh = StructuredMesh(shape=(1,), domain=((0.0, 1.0),))
        u0 = FieldData(name="u", data=torch.tensor([1.0], dtype=torch.float64), mesh=mesh)
        state = SimulationState(t=0.0, fields={"u": u0}, mesh=mesh)

        def rhs(s, t):
            return {"u": -s.get_field("u").data}

        result = RK4().solve(state, rhs, t_span=(0.0, 1.0), dt=0.01)
        exact = math.exp(-1.0)
        assert abs(result.final_state.get_field("u").data.item() - exact) < 1e-8


class TestSymplecticIntegrators:
    def test_verlet_energy_conservation(self):
        """Harmonic oscillator energy should be conserved to machine precision."""
        mesh = StructuredMesh(shape=(1,), domain=((0.0, 1.0),))
        q0 = FieldData(name="q", data=torch.tensor([1.0], dtype=torch.float64), mesh=mesh)
        p0 = FieldData(name="p", data=torch.tensor([0.0], dtype=torch.float64), mesh=mesh)
        state = SimulationState(t=0.0, fields={"q": q0, "p": p0}, mesh=mesh)

        omega = 1.0

        def rhs(s, t):
            q = s.get_field("q").data
            p = s.get_field("p").data
            return {"q": p, "p": -omega**2 * q}

        result = StormerVerlet().solve(state, rhs, t_span=(0.0, 100.0), dt=0.001)
        E0 = 0.5  # p²/2 + ω²q²/2 = 0 + 0.5*1 = 0.5
        q_f = result.final_state.get_field("q").data.item()
        p_f = result.final_state.get_field("p").data.item()
        E_f = 0.5 * p_f**2 + 0.5 * omega**2 * q_f**2
        assert abs(E_f - E0) / E0 < 1e-7  # 100k steps, Verlet bounded drift


# ═══════════════════════════════════════════════════════════════════════════════
# Linear Solvers
# ═══════════════════════════════════════════════════════════════════════════════


class TestConjugateGradient:
    def test_spd_system(self):
        """Solve A x = b where A = diag(1,2,3,...,10)."""
        n = 10
        diag = torch.arange(1, n + 1, dtype=torch.float64)
        b = torch.ones(n, dtype=torch.float64)

        def matvec(x):
            return diag * x

        result = ConjugateGradient().solve(matvec, b)
        assert result.converged
        expected = 1.0 / diag
        assert torch.allclose(result.x, expected, atol=1e-8)


class TestGMRES:
    def test_nonsymmetric(self):
        """Solve a lower-triangular system."""
        n = 5
        L = torch.eye(n, dtype=torch.float64) + 0.5 * torch.tril(torch.ones(n, n, dtype=torch.float64), -1)
        b = torch.ones(n, dtype=torch.float64)
        x_exact = torch.linalg.solve(L, b)

        def matvec(x):
            return L @ x

        result = GMRES().solve(matvec, b)
        assert result.converged
        assert torch.allclose(result.x, x_exact, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Nonlinear Solvers
# ═══════════════════════════════════════════════════════════════════════════════


class TestNewtonSolver:
    def test_scalar_root(self):
        """Solve x² - 2 = 0."""
        def residual(x):
            return x**2 - 2.0

        result = NewtonSolver().solve(residual, torch.tensor([1.0], dtype=torch.float64))
        assert result.converged
        assert abs(result.x.item() - math.sqrt(2)) < 1e-6


class TestPicardSolver:
    def test_fixed_point(self):
        """x = cos(x), fixed point ≈ 0.7391."""
        result = PicardSolver().solve(
            torch.cos,
            torch.tensor([0.5], dtype=torch.float64),
            tol=1e-8,
            max_iter=100,
        )
        assert result.converged
        assert abs(result.x.item() - 0.7390851332) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# Domain Pack Registry
# ═══════════════════════════════════════════════════════════════════════════════


class _DummySolver:
    @property
    def name(self):
        return "DummySolver"

    def step(self, state, dt, **kw):
        return state

    def solve(self, state, t_span, dt, **kw):
        return SolveResult(final_state=state, t_final=t_span[1],
                           steps_taken=0)


class _DummySpec:
    @property
    def name(self):
        return "Dummy"

    @property
    def ndim(self):
        return 1

    @property
    def parameters(self):
        return {}

    @property
    def governing_equations(self):
        return "du/dt = 0"

    @property
    def field_names(self):
        return ["u"]

    @property
    def observable_names(self):
        return []


class _DummyDisc:
    @property
    def method(self):
        return "FVM"

    @property
    def order(self):
        return 1

    def discretize(self, spec, mesh):
        return None


class _TestPack(DomainPack):
    @property
    def pack_id(self):
        return "TEST"

    @property
    def pack_name(self):
        return "Test Pack"

    @property
    def taxonomy_ids(self):
        return ["PHY-TEST.1", "PHY-TEST.2"]

    def problem_specs(self):
        return {"PHY-TEST.1": _DummySpec, "PHY-TEST.2": _DummySpec}

    def solvers(self):
        return {"PHY-TEST.1": _DummySolver, "PHY-TEST.2": _DummySolver}

    def discretizations(self):
        return {"PHY-TEST.1": [_DummyDisc], "PHY-TEST.2": [_DummyDisc]}


class TestDomainRegistry:
    def test_register_and_query(self):
        reg = get_registry()
        pack = _TestPack()
        reg.register_pack(pack)
        assert "TEST" in reg.list_packs()
        assert "PHY-TEST.1" in reg.list_nodes()
        assert reg.get_solver("PHY-TEST.1") is _DummySolver

    def test_duplicate_pack_raises(self):
        reg = get_registry()
        reg.register_pack(_TestPack())
        with pytest.raises(ValueError, match="already registered"):
            reg.register_pack(_TestPack())

    def test_unregister(self):
        reg = get_registry()
        reg.register_pack(_TestPack())
        reg.unregister_pack("TEST")
        assert "TEST" not in reg.list_packs()

    def test_summary(self):
        reg = get_registry()
        reg.register_pack(_TestPack())
        s = reg.summary()
        assert s["packs"] == 1
        assert s["nodes"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════


class TestReproducibility:
    def test_hash_tensor_deterministic(self):
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        h1 = hash_tensor(t)
        h2 = hash_tensor(t)
        assert h1.digest == h2.digest

    def test_hash_different_tensors(self):
        h1 = hash_tensor(torch.tensor([1.0], dtype=torch.float64))
        h2 = hash_tensor(torch.tensor([2.0], dtype=torch.float64))
        assert h1.digest != h2.digest

    def test_environment_capture(self):
        env = capture_environment()
        assert "python_version" in env
        assert "torch_version" in env
        assert "timestamp_utc" in env

    def test_context_provenance(self):
        with ReproducibilityContext(seed=123) as ctx:
            t = torch.randn(10)
            ctx.record("test", hash_tensor(t))
        prov = ctx.provenance()
        assert "seed" in prov
        assert "environment" in prov
        assert "test" in prov["artifacts"]
        assert prov["wall_time_seconds"] >= 0

    def test_seed_determinism(self):
        lock_seeds(42)
        a = torch.randn(100)
        lock_seeds(42)
        b = torch.randn(100)
        assert torch.equal(a, b)


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpointing:
    def test_roundtrip_1d(self, simple_state):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = save_checkpoint(simple_state, tmpdir, name="test")
            restored = load_checkpoint(ckpt)
            assert abs(restored.t - simple_state.t) < 1e-15
            assert restored.step_index == simple_state.step_index
            assert torch.equal(
                restored.get_field("u").data,
                simple_state.get_field("u").data,
            )

    def test_roundtrip_preserves_hash(self, simple_state):
        h_before = hash_tensor(simple_state.get_field("u").data)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = save_checkpoint(simple_state, tmpdir)
            restored = load_checkpoint(ckpt)
        h_after = hash_tensor(restored.get_field("u").data)
        assert h_before.digest == h_after.digest


# ═══════════════════════════════════════════════════════════════════════════════
# Vertical Slices (integration tests)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestVerticalSliceODE:
    def test_harmonic_oscillator_v04(self):
        from ontic.platform.vertical_ode import run_ode_vertical_slice
        m = run_ode_vertical_slice(verbose=False)
        assert m["energy_relative_error"] < 1e-8
        assert m["phase_error_deg"] < 0.5
        assert m["checkpoint_roundtrip_ok"]
        assert m["deterministic"]


@pytest.mark.integration
class TestVerticalSlicePDE:
    def test_heat_equation_v04(self):
        from ontic.platform.vertical_pde import run_pde_vertical_slice
        m = run_pde_vertical_slice(verbose=False)
        assert m["finest_linf"] < 1e-4
        assert m["refinement_ratios"]["r12"] > 3.5
        assert m["refinement_ratios"]["r23"] > 3.5
        assert m["monotone_decay"]
        assert m["checkpoint_roundtrip_ok"]
        assert m["deterministic"]
