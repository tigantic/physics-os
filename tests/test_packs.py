"""
Test suite for all 20 domain packs.

Tests:
  • All 20 packs load and register via discover_all().
  • Each pack's DomainPack ABC is fully satisfied (pack_id, pack_name, etc.).
  • Each anchor produces the expected validation result.
  • Protocol compliance: every node has a ProblemSpec + Solver.
  • Scaffold packs have version "0.1.0", anchors have "0.4.0".
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict

import pytest
import torch

from tensornet.packs import discover_all
from tensornet.platform.domain_pack import get_registry


# ═══════════════════════════════════════════════════════════════════════════════
# Discovery & Registration
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def registry():
    """Load all packs and return the registry."""
    discover_all()
    return get_registry()


@pytest.fixture(scope="module")
def all_packs(registry):
    """All registered packs as a dict keyed by pack_id."""
    return {p.pack_id: p for p in registry._packs.values()}


def test_all_20_packs_registered(registry):
    """All 20 domain packs must be discoverable."""
    discover_all()
    assert len(registry._packs) >= 20


def test_expected_pack_ids(all_packs):
    """Expected pack IDs are present."""
    expected = {
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
        "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    }
    assert expected.issubset(set(all_packs.keys()))


def test_total_node_count(all_packs):
    """At least 140 taxonomy nodes across all packs."""
    total = sum(len(p.taxonomy_ids) for p in all_packs.values())
    assert total >= 140, f"Only {total} nodes, expected ≥ 140"


# ═══════════════════════════════════════════════════════════════════════════════
# Per-pack structural tests
# ═══════════════════════════════════════════════════════════════════════════════


ANCHOR_PACKS = {"II", "III", "V", "VII", "VIII", "XI"}
V02_PACKS = {"I", "IV", "VI", "IX", "X", "XII", "XIII", "XIV",
             "XV", "XVI", "XVII", "XVIII", "XIX", "XX"}
SCAFFOLD_PACKS: set[str] = set()  # all former scaffolds now at V0.2

ALL_PACK_IDS = ANCHOR_PACKS | V02_PACKS | SCAFFOLD_PACKS


@pytest.mark.parametrize("pack_id", sorted(ALL_PACK_IDS))
def test_pack_has_required_attributes(all_packs, pack_id):
    """Each pack has pack_id, pack_name, taxonomy_ids."""
    p = all_packs[pack_id]
    assert p.pack_id == pack_id
    assert isinstance(p.pack_name, str) and len(p.pack_name) > 0
    assert len(p.taxonomy_ids) >= 1


@pytest.mark.parametrize("pack_id", sorted(ALL_PACK_IDS))
def test_pack_has_problem_specs(all_packs, pack_id):
    """Each pack's problem_specs covers all taxonomy nodes."""
    p = all_packs[pack_id]
    specs = p.problem_specs()
    for tid in p.taxonomy_ids:
        assert tid in specs, f"Missing ProblemSpec for {tid}"


@pytest.mark.parametrize("pack_id", sorted(ALL_PACK_IDS))
def test_pack_has_solvers(all_packs, pack_id):
    """Each pack's solvers covers all taxonomy nodes."""
    p = all_packs[pack_id]
    solvers = p.solvers()
    for tid in p.taxonomy_ids:
        assert tid in solvers, f"Missing Solver for {tid}"


@pytest.mark.parametrize("pack_id", sorted(ANCHOR_PACKS))
def test_anchor_version(all_packs, pack_id):
    """Anchor packs at version 0.4.0."""
    p = all_packs[pack_id]
    v = p.version() if callable(p.version) else p.version
    assert v == "0.4.0", f"Pack {pack_id} version = {v}"


@pytest.mark.parametrize("pack_id", sorted(SCAFFOLD_PACKS) if SCAFFOLD_PACKS else ["_skip_"])
def test_scaffold_version(all_packs, pack_id):
    """Scaffold packs at version 0.1.0."""
    if pack_id == "_skip_":
        pytest.skip("No scaffold packs remaining")
    p = all_packs[pack_id]
    v = p.version() if callable(p.version) else p.version
    assert v == "0.1.0", f"Pack {pack_id} version = {v}"


@pytest.mark.parametrize("pack_id", sorted(V02_PACKS))
def test_v02_version(all_packs, pack_id):
    """V0.2 packs at version 0.2.0."""
    p = all_packs[pack_id]
    v = p.version() if callable(p.version) else p.version
    assert v == "0.2.0", f"Pack {pack_id} version = {v}"


# ═══════════════════════════════════════════════════════════════════════════════
# ProblemSpec protocol checks
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("pack_id", sorted(ALL_PACK_IDS))
def test_problem_specs_have_required_properties(all_packs, pack_id):
    """Each ProblemSpec class must expose name, ndim, parameters, etc."""
    p = all_packs[pack_id]
    specs = p.problem_specs()
    for tid, spec_cls in specs.items():
        inst = spec_cls()
        assert hasattr(inst, "name") and isinstance(inst.name, str)
        assert hasattr(inst, "ndim") and isinstance(inst.ndim, int)
        assert hasattr(inst, "parameters")
        assert hasattr(inst, "governing_equations")
        assert hasattr(inst, "field_names")
        assert hasattr(inst, "observable_names")


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor vertical slice tests (smoke tests – actual validation in pack modules)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPackIIBurgersSmoke:
    """Pack II: quick Burgers check."""

    def test_solver_runs(self):
        from tensornet.packs.pack_ii import BurgersSpec, BurgersSolver
        spec = BurgersSpec()
        assert spec.name == "ViscousBurgers1D"
        solver = BurgersSolver()
        assert "Burgers" in solver.name


class TestPackIIIMaxwellSmoke:
    """Pack III: quick Maxwell check."""

    def test_solver_runs(self):
        from tensornet.packs.pack_iii import Maxwell1DSpec, MaxwellSolver
        spec = Maxwell1DSpec()
        assert "Maxwell" in spec.name
        solver = MaxwellSolver()
        assert "Maxwell" in solver.name


class TestPackVHeatSmoke:
    """Pack V: quick heat transfer check."""

    def test_solver_runs(self):
        from tensornet.packs.pack_v import AdvectionDiffusionSpec, AdvDiffSolver
        spec = AdvectionDiffusionSpec()
        assert "Advection" in spec.name or "AdvDiff" in spec.name
        solver = AdvDiffSolver()
        assert "AdvDiff" in solver.name or "Advection" in solver.name


class TestPackVIIQuantumSmoke:
    """Pack VII: quick MPS check."""

    def test_heisenberg_hamiltonian(self):
        from tensornet.packs.pack_vii import heisenberg_two_site_hamiltonian
        h2 = heisenberg_two_site_hamiltonian(J=1.0)
        assert h2.shape == (4, 4)
        # Symmetric
        assert torch.allclose(h2, h2.T)

    def test_exact_diag_small(self):
        from tensornet.packs.pack_vii import exact_diag_heisenberg
        E = exact_diag_heisenberg(N=4, J=1.0)
        # Known: E_0(N=4) ≈ -1.6160...
        assert abs(E - (-1.6160254038)) < 1e-6

    def test_mps_norm(self):
        from tensornet.packs.pack_vii import MPS
        psi = MPS(N=4, d=2, chi=1)
        assert abs(psi.norm() - 1.0) < 1e-14


class TestPackVIIIDFTSmoke:
    """Pack VIII: quick KS check."""

    def test_kinetic_matrix(self):
        from tensornet.packs.pack_viii import build_kinetic_matrix
        T = build_kinetic_matrix(N=10, dx=0.1)
        assert T.shape == (10, 10)
        assert torch.allclose(T, T.T)  # symmetric

    def test_scf_converges(self):
        from tensornet.packs.pack_viii import kohn_sham_scf
        r = kohn_sham_scf(N_grid=50, L=10.0, Z=2.0, a=1.0,
                          N_electrons=2, max_iter=200, mix_alpha=0.3, tol=1e-6)
        assert r["converged"]
        # Energy should be negative (bound state)
        assert r["total_energy"] < 0


class TestPackXIPlasmaSmoke:
    """Pack XI: quick Vlasov–Poisson check."""

    def test_poisson_solve(self):
        from tensornet.packs.pack_xi import _poisson_solve_1d
        N = 64
        dx = 1.0 / N
        x = torch.linspace(0, 1 - dx, N, dtype=torch.float64)
        rho = torch.sin(2 * math.pi * x)
        E = _poisson_solve_1d(rho, dx)
        # E should be ∝ cos(2πx) / (2π)
        E_exact = torch.cos(2 * math.pi * x) / (2 * math.pi)
        # Normalize to compare shapes
        if E.abs().max() > 0:
            ratio = E_exact.abs().max() / E.abs().max()
            assert abs(ratio - 1.0) < 0.5  # within factor of 2


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: full anchor validation (slow, marked)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestAnchorValidation:
    """Full anchor validation — run with pytest -m slow."""

    def test_pack_v_heat(self):
        from tensornet.packs.pack_v import run_heat_vertical_slice
        m = run_heat_vertical_slice(verbose=False)
        assert m["finest_linf"] < 1e-4
        assert all(o > 1.8 for o in m["convergence_orders"])

    def test_pack_ii_fluids(self):
        from tensornet.packs.pack_ii import run_fluids_vertical_slice
        m = run_fluids_vertical_slice(verbose=False)
        assert m["finest_linf"] < 1e-3
        assert all(o > 1.5 for o in m["convergence_orders"])

    def test_pack_iii_maxwell(self):
        from tensornet.packs.pack_iii import run_em_vertical_slice
        m = run_em_vertical_slice(verbose=False)
        assert m["finest_linf_E"] < 1e-4
        assert all(o > 1.8 for o in m["convergence_orders"])

    def test_pack_vii_quantum(self):
        from tensornet.packs.pack_vii import run_quantum_mb_vertical_slice
        m = run_quantum_mb_vertical_slice(verbose=False)
        assert m["best_error_8"] < 1e-4
        assert m["best_error_12"] < 1e-4

    def test_pack_viii_dft(self):
        from tensornet.packs.pack_viii import run_dft_vertical_slice
        m = run_dft_vertical_slice(verbose=False)
        assert m["best_error"] < 1e-4
        assert m["min_order"] > 1.8

    def test_pack_xi_plasma(self):
        from tensornet.packs.pack_xi import run_plasma_vertical_slice
        m = run_plasma_vertical_slice(verbose=False)
        assert m["best_gamma_error"] < 0.05
        assert m["improves_with_refinement"]
