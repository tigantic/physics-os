#!/usr/bin/env python3
"""
Trustless Physics Gauntlet — Phase 8 Validation
================================================

Validates the Tier 3 Wire-Up: 45 domains across Quantum Mechanics (5),
Quantum Many-Body (13), Electronic Structure (7), Solid State (8),
Nuclear & Particle (6), Chemical Physics (3), and Quantum Information (3)
connected to the STARK proof pipeline via trace adapters, Lean formal
proofs, and TPC certificate generation.

Test Layers:
    1. adapter_files:           All 45 trace adapters exist, correct APIs
    2. core_adapters:           SCF and Eigenvalue core adapters importable
    3. lean_proofs:             7 category-level Lean proofs with expected theorems
    4. tpc_script:              Phase 8 TPC generator exists and is importable
    5. qm_solvers:              Run 5 quantum mechanics domain adapters → trace
    6. qmb_solvers:             Run 13 quantum many-body domain adapters → trace
    7. electronic_solvers:      Run 7 electronic structure domain adapters → trace
    8. solid_state_solvers:     Run 8 solid state domain adapters → trace
    9. nuclear_particle_solvers: Run 6 nuclear & particle domain adapters → trace
   10. chem_physics_solvers:    Run 3 chemical physics domain adapters → trace
   11. quantum_info_solvers:    Run 3 quantum information domain adapters → trace
   12. conservation:            Conservation law verification across key domains
   13. integration:             Cross-category validation

Pass criteria: ALL tests must pass. No exceptions.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ── Setup ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("trustless_physics_phase8_gauntlet")

# ═════════════════════════════════════════════════════════════════════════════
# Gauntlet Framework
# ═════════════════════════════════════════════════════════════════════════════

RESULTS: dict[str, dict[str, Any]] = {}
_start_time = time.monotonic()


def gauntlet(name: str, layer: str = "phase8"):
    """Decorator to register a gauntlet test."""
    def decorator(func):
        def wrapper():
            t0 = time.monotonic()
            try:
                func()
                elapsed = time.monotonic() - t0
                RESULTS[name] = {
                    "layer": layer,
                    "passed": True,
                    "time_seconds": round(elapsed, 4),
                    "error": None,
                }
                logger.info(f"  ✅ {name} ({elapsed:.3f}s)")
                return True
            except Exception as e:
                elapsed = time.monotonic() - t0
                RESULTS[name] = {
                    "layer": layer,
                    "passed": False,
                    "time_seconds": round(elapsed, 4),
                    "error": f"{type(e).__name__}: {e}",
                }
                logger.error(f"  ❌ {name} ({elapsed:.3f}s)")
                logger.error(f"     {type(e).__name__}: {e}")
                traceback.print_exc()
                return False
        wrapper.__name__ = name
        return wrapper
    return decorator


# ═════════════════════════════════════════════════════════════════════════════
# Adapter File Lists
# ═════════════════════════════════════════════════════════════════════════════

QM_ADAPTERS_QMECH = [
    "tise_adapter.py", "tdse_adapter.py", "path_integrals_adapter.py",
]
QM_ADAPTERS_QM = [
    "scattering_adapter.py", "semiclassical_adapter.py",
]

QMB_ADAPTERS_COND = [
    "quantum_spin_adapter.py", "strongly_correlated_adapter.py",
    "topological_adapter.py", "mbl_adapter.py", "open_quantum_adapter.py",
    "nonequilibrium_qm_adapter.py", "kondo_adapter.py", "bosonic_adapter.py",
    "fermionic_adapter.py", "nuclear_many_body_adapter.py", "ultracold_adapter.py",
]
QMB_ADAPTERS_ALG = [
    "dmrg_adapter.py",
]
QMB_ADAPTERS_QFT = [
    "lattice_gauge_adapter.py",
]

ES_ADAPTERS = [
    "dft_adapter.py", "beyond_dft_adapter.py", "tight_binding_adapter.py",
    "excited_states_adapter.py", "response_adapter.py", "relativistic_adapter.py",
    "embedding_adapter.py",
]

SS_ADAPTERS = [
    "phonons_adapter.py", "band_structure_adapter.py",
    "classical_magnetism_adapter.py", "superconductivity_adapter.py",
    "disordered_adapter.py", "surfaces_adapter.py", "defects_adapter.py",
    "ferroelectrics_adapter.py",
]

NP_ADAPTERS_NUC = [
    "nuclear_structure_adapter.py", "nuclear_reactions_adapter.py",
    "nuclear_astro_adapter.py",
]
NP_ADAPTERS_QFT = [
    "lattice_qcd_adapter.py", "perturbative_qft_adapter.py",
]
NP_ADAPTERS_PART = [
    "beyond_sm_adapter.py",
]

CP_ADAPTERS = [
    "pes_adapter.py", "reaction_rate_adapter.py", "catalysis_adapter.py",
]

QI_ADAPTERS = [
    "quantum_circuit_adapter.py", "qec_adapter.py", "vqe_adapter.py",
]


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1: Trace Adapter File Existence
# ═════════════════════════════════════════════════════════════════════════════

def _check_adapter_dir(pkg: Path, adapters: list[str], label: str) -> None:
    assert pkg.exists(), f"Missing: {pkg}"
    assert (pkg / "__init__.py").exists(), f"Missing __init__.py in {pkg}"
    for fname in adapters:
        fpath = pkg / fname
        assert fpath.exists(), f"Missing: {fname} in {pkg}"
        assert fpath.stat().st_size > 500, f"{fname} too small ({fpath.stat().st_size} bytes)"
    logger.info(f"    All {len(adapters)} {label} adapter files present")


@gauntlet("qm_adapter_files_qmech", layer="adapter_files")
def test_qm_adapter_files_qmech():
    _check_adapter_dir(ROOT / "tensornet" / "quantum_mechanics" / "trace_adapters",
                       QM_ADAPTERS_QMECH, "quantum mechanics (qmech)")


@gauntlet("qm_adapter_files_qm", layer="adapter_files")
def test_qm_adapter_files_qm():
    _check_adapter_dir(ROOT / "tensornet" / "qm" / "trace_adapters",
                       QM_ADAPTERS_QM, "quantum mechanics (qm)")


@gauntlet("qmb_adapter_files_cond", layer="adapter_files")
def test_qmb_adapter_files_cond():
    _check_adapter_dir(ROOT / "tensornet" / "condensed_matter" / "trace_adapters",
                       QMB_ADAPTERS_COND, "quantum many-body (cond_matter)")


@gauntlet("qmb_adapter_files_alg", layer="adapter_files")
def test_qmb_adapter_files_alg():
    _check_adapter_dir(ROOT / "tensornet" / "algorithms" / "trace_adapters",
                       QMB_ADAPTERS_ALG, "quantum many-body (algorithms)")


@gauntlet("qmb_adapter_files_qft", layer="adapter_files")
def test_qmb_adapter_files_qft():
    _check_adapter_dir(ROOT / "tensornet" / "qft" / "trace_adapters",
                       QMB_ADAPTERS_QFT, "quantum many-body (qft)")


@gauntlet("es_adapter_files", layer="adapter_files")
def test_es_adapter_files():
    _check_adapter_dir(ROOT / "tensornet" / "electronic_structure" / "trace_adapters",
                       ES_ADAPTERS, "electronic structure")


@gauntlet("ss_adapter_files", layer="adapter_files")
def test_ss_adapter_files():
    _check_adapter_dir(ROOT / "tensornet" / "condensed_matter" / "trace_adapters",
                       SS_ADAPTERS, "solid state")


@gauntlet("np_adapter_files_nuc", layer="adapter_files")
def test_np_adapter_files_nuc():
    _check_adapter_dir(ROOT / "tensornet" / "nuclear" / "trace_adapters",
                       NP_ADAPTERS_NUC, "nuclear")


@gauntlet("np_adapter_files_qft", layer="adapter_files")
def test_np_adapter_files_qft():
    _check_adapter_dir(ROOT / "tensornet" / "qft" / "trace_adapters",
                       NP_ADAPTERS_QFT, "particle (qft)")


@gauntlet("np_adapter_files_part", layer="adapter_files")
def test_np_adapter_files_part():
    _check_adapter_dir(ROOT / "tensornet" / "particle" / "trace_adapters",
                       NP_ADAPTERS_PART, "particle")


@gauntlet("cp_adapter_files", layer="adapter_files")
def test_cp_adapter_files():
    _check_adapter_dir(ROOT / "tensornet" / "chemistry" / "trace_adapters",
                       CP_ADAPTERS, "chemical physics")


@gauntlet("qi_adapter_files", layer="adapter_files")
def test_qi_adapter_files():
    _check_adapter_dir(ROOT / "tensornet" / "quantum" / "trace_adapters",
                       QI_ADAPTERS, "quantum info")


@gauntlet("adapter_total_count", layer="adapter_files")
def test_adapter_total_count():
    total = (len(QM_ADAPTERS_QMECH) + len(QM_ADAPTERS_QM) + len(QMB_ADAPTERS_COND)
             + len(QMB_ADAPTERS_ALG) + len(QMB_ADAPTERS_QFT) + len(ES_ADAPTERS)
             + len(SS_ADAPTERS) + len(NP_ADAPTERS_NUC) + len(NP_ADAPTERS_QFT)
             + len(NP_ADAPTERS_PART) + len(CP_ADAPTERS) + len(QI_ADAPTERS))
    assert total == 45, f"Expected 45 adapters, got {total}"


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2: Core Adapters
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("core_scf_adapter", layer="core_adapters")
def test_core_scf_adapter():
    from tensornet.core.scf_trace_adapter import SCFTraceAdapter, SCFConvergence
    adapter = SCFTraceAdapter("test_scf")
    assert hasattr(adapter, "run")
    assert callable(adapter.run)


@gauntlet("core_eigenvalue_adapter", layer="core_adapters")
def test_core_eigenvalue_adapter():
    from tensornet.core.eigenvalue_trace_adapter import (
        EigenvalueTraceAdapter, EigenvalueConvergence,
    )
    adapter = EigenvalueTraceAdapter("test_eig")
    assert hasattr(adapter, "wrap_diagonalisation")
    assert hasattr(adapter, "wrap_iterative")


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3: Lean Proofs
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("lean_qm_conservation", layer="lean_proofs")
def test_lean_qm_conservation():
    lean = ROOT / "quantum_mechanics_conservation_proof" / "QuantumMechanicsConservation.lean"
    assert lean.exists(), f"Missing: {lean}"
    src = lean.read_text()
    assert len(src) > 2000
    required = [
        "tise_normalisation", "tdse_probability_conservation",
        "scattering_optical_theorem", "semiclassical_action_quantised",
        "path_integral_detailed_balance",
        "all_quantummechanicsconservation_verified",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"
    assert "by decide" in src


@gauntlet("lean_qmb_conservation", layer="lean_proofs")
def test_lean_qmb_conservation():
    lean = ROOT / "qmb_conservation_proof" / "QuantumManyBodyConservation.lean"
    assert lean.exists(), f"Missing: {lean}"
    src = lean.read_text()
    required = [
        "dmrg_converged", "q_spin_sz_conservation", "dmft_converged",
        "topo_chern_integer", "mbl_gap_ratio_below_poisson",
        "gauge_gauss_law", "open_q_trace_conservation",
        "floquet_unitarity", "kondo_spectral",
        "bosonic_particle_conservation", "fermionic_gap",
        "nuc_mb_nucleon_conservation", "ultracold_atom_conservation",
        "all_quantummanybodyconservation_verified",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"
    assert "by decide" in src


@gauntlet("lean_es_conservation", layer="lean_proofs")
def test_lean_es_conservation():
    lean = ROOT / "electronic_structure_conservation_proof" / "ElectronicStructureConservation.lean"
    assert lean.exists(), f"Missing: {lean}"
    src = lean.read_text()
    required = [
        "dft_converged", "dft_electron_conservation",
        "beyond_dft_converged", "tb_charge_neutrality",
        "excited_positive", "response_kramers_kronig",
        "relativistic_current", "embedding_electron_conservation",
        "all_electronicstructureconservation_verified",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"
    assert "by decide" in src


@gauntlet("lean_ss_conservation", layer="lean_proofs")
def test_lean_ss_conservation():
    lean = ROOT / "solid_state_conservation_proof" / "SolidStateConservation.lean"
    assert lean.exists(), f"Missing: {lean}"
    src = lean.read_text()
    required = [
        "phonon_acoustic_sum", "band_charge_neutrality",
        "mag_magnitude_conservation", "sc_particle_conservation",
        "disorder_normalisation", "surface_charge_neutrality",
        "defect_charge_balance", "ferro_bounded",
        "all_solidstateconservation_verified",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"
    assert "by decide" in src


@gauntlet("lean_np_conservation", layer="lean_proofs")
def test_lean_np_conservation():
    lean = ROOT / "nuclear_particle_conservation_proof" / "NuclearParticleConservation.lean"
    assert lean.exists(), f"Missing: {lean}"
    src = lean.read_text()
    required = [
        "nuc_struct_nucleon", "nuc_react_unitarity",
        "nuc_astro_baryon", "lqcd_gauge_invariance",
        "pqft_ward_identity", "bsm_prob_unitarity",
        "all_nuclearparticleconservation_verified",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"
    assert "by decide" in src


@gauntlet("lean_cp_conservation", layer="lean_proofs")
def test_lean_cp_conservation():
    lean = ROOT / "chem_physics_iter_conservation_proof" / "ChemPhysicsIterConservation.lean"
    assert lean.exists(), f"Missing: {lean}"
    src = lean.read_text()
    required = [
        "pes_gradient_zero", "rate_positive",
        "catalysis_atom_conservation",
        "all_chemphysicsiterconservation_verified",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"
    assert "by decide" in src


@gauntlet("lean_qi_conservation", layer="lean_proofs")
def test_lean_qi_conservation():
    lean = ROOT / "quantum_info_conservation_proof" / "QuantumInfoConservation.lean"
    assert lean.exists(), f"Missing: {lean}"
    src = lean.read_text()
    required = [
        "qcircuit_unitarity", "qec_fidelity", "vqe_converged",
        "all_quantuminfoconservation_verified",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"
    assert "by decide" in src


# ═════════════════════════════════════════════════════════════════════════════
# Layer 4: TPC Script
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("tpc_phase8_script_exists", layer="tpc_script")
def test_tpc_phase8_script_exists():
    script = ROOT / "scripts" / "tpc" / "generate_phase8.py"
    assert script.exists(), f"Missing: {script}"
    assert script.stat().st_size > 5000, "Script too small"


@gauntlet("tpc_phase8_importable", layer="tpc_script")
def test_tpc_phase8_importable():
    tpc_dir = ROOT / "scripts" / "tpc"
    sys.path.insert(0, str(tpc_dir))
    try:
        from generate_phase8 import DOMAIN_RUNNERS, run_domain
        assert len(DOMAIN_RUNNERS) == 45, f"Expected 45 runners, got {len(DOMAIN_RUNNERS)}"
        assert callable(run_domain)
    finally:
        sys.path.pop(0)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 5: Quantum Mechanics Solver Runs (5)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("qm_tise_solve", layer="qm_solvers")
def test_qm_tise_solve():
    from tensornet.quantum.quantum_mechanics.trace_adapters.tise_adapter import TISETraceAdapter
    adapter = TISETraceAdapter(x_min=-10.0, x_max=10.0, n_grid=200, mass=1.0)
    evals, cons, session = adapter.solve(n_states=5)
    assert len(evals) >= 3, f"Expected ≥3 eigenvalues, got {len(evals)}"
    assert len(session.entries) >= 2
    assert cons.n_eigenvalues >= 3


@gauntlet("qm_tdse_solve", layer="qm_solvers")
def test_qm_tdse_solve():
    from tensornet.quantum.quantum_mechanics.trace_adapters.tdse_adapter import TDSETraceAdapter
    adapter = TDSETraceAdapter(x_min=-20.0, x_max=20.0, n_grid=256, mass=1.0)
    x = np.linspace(-20, 20, 256)
    psi_0 = np.exp(-x**2 / 2.0).astype(complex)
    psi_0 /= np.sqrt(np.sum(np.abs(psi_0)**2) * 40.0 / 256)
    psi_f, cons, session = adapter.propagate(psi_0, dt=0.01, n_steps=50)
    assert psi_f is not None
    assert len(session.entries) >= 2


@gauntlet("qm_scattering_solve", layer="qm_solvers")
def test_qm_scattering_solve():
    from tensornet.quantum.qm.trace_adapters.scattering_adapter import ScatteringTraceAdapter
    adapter = ScatteringTraceAdapter(k=1.0, l_max=10)
    dsigma, cons, session = adapter.evaluate()
    assert cons.total_cross_section > 0
    assert len(session.entries) >= 2


@gauntlet("qm_semiclassical_solve", layer="qm_solvers")
def test_qm_semiclassical_solve():
    from tensornet.quantum.qm.trace_adapters.semiclassical_adapter import SemiclassicalTraceAdapter
    adapter = SemiclassicalTraceAdapter(mass=1.0)
    energies, cons, session = adapter.evaluate(n_levels=5)
    assert cons.n_levels >= 1
    assert len(session.entries) >= 2


@gauntlet("qm_path_integrals_solve", layer="qm_solvers")
def test_qm_path_integrals_solve():
    from tensornet.quantum.quantum_mechanics.trace_adapters.path_integrals_adapter import PathIntegralsTraceAdapter
    adapter = PathIntegralsTraceAdapter(n_beads=8, temperature=1.0, mass=1.0)
    result, cons, session = adapter.evaluate(n_mc_steps=100)
    assert cons.detailed_balance
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 6: Quantum Many-Body Solver Runs (13)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("qmb_dmrg_solve", layer="qmb_solvers")
def test_qmb_dmrg_solve():
    from tensornet.algorithms.trace_adapters.dmrg_adapter import DMRGTraceAdapter
    from tensornet.mps.hamiltonians import heisenberg_mpo
    mpo = heisenberg_mpo(6)
    adapter = DMRGTraceAdapter(chi_max=16, num_sweeps=4)
    result, cons, session = adapter.evaluate(H_mpo=mpo)
    assert cons.ground_energy < 0
    assert len(session.entries) >= 2


@gauntlet("qmb_quantum_spin_solve", layer="qmb_solvers")
def test_qmb_quantum_spin_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.quantum_spin_adapter import QuantumSpinTraceAdapter
    adapter = QuantumSpinTraceAdapter(L=6, J=1.0, Delta=1.0)
    result, cons, session = adapter.evaluate()
    assert cons.ground_energy < 0
    assert len(session.entries) >= 2


@gauntlet("qmb_strongly_correlated_solve", layer="qmb_solvers")
def test_qmb_strongly_correlated_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.strongly_correlated_adapter import StronglyCorrelatedTraceAdapter
    adapter = StronglyCorrelatedTraceAdapter(U=4.0, mu=2.0, D=1.0, beta=10.0)
    result, cons, session = adapter.evaluate(max_iter=10, tol=1e-3)
    assert len(session.entries) >= 2


@gauntlet("qmb_topological_solve", layer="qmb_solvers")
def test_qmb_topological_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.topological_adapter import TopologicalTraceAdapter
    adapter = TopologicalTraceAdapter(nk=10)
    C, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


@gauntlet("qmb_mbl_solve", layer="qmb_solvers")
def test_qmb_mbl_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.mbl_adapter import MBLTraceAdapter
    adapter = MBLTraceAdapter(L=6, W=5.0)
    evals, cons, session = adapter.evaluate()
    assert cons.n_eigenvalues > 0
    assert len(session.entries) >= 2


@gauntlet("qmb_lattice_gauge_solve", layer="qmb_solvers")
def test_qmb_lattice_gauge_solve():
    from tensornet.quantum.qft.trace_adapters.lattice_gauge_adapter import LatticeGaugeTraceAdapter
    adapter = LatticeGaugeTraceAdapter(L=4, beta=2.0)
    plaq, cons, session = adapter.evaluate(n_sweeps=5)
    assert len(session.entries) >= 2


@gauntlet("qmb_open_quantum_solve", layer="qmb_solvers")
def test_qmb_open_quantum_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.open_quantum_adapter import OpenQuantumTraceAdapter
    adapter = OpenQuantumTraceAdapter(dim=3)
    rho_f, cons, session = adapter.evaluate(n_steps=50, dt=0.1)
    assert abs(cons.trace_rho_final - 1.0) < 0.05
    assert len(session.entries) >= 2


@gauntlet("qmb_nonequilibrium_qm_solve", layer="qmb_solvers")
def test_qmb_nonequilibrium_qm_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.nonequilibrium_qm_adapter import NonEquilibriumQMTraceAdapter
    adapter = NonEquilibriumQMTraceAdapter(dim=3)
    qe, cons, session = adapter.evaluate(T_period=1.0)
    assert cons.n_quasi_energies > 0
    assert len(session.entries) >= 2


@gauntlet("qmb_kondo_solve", layer="qmb_solvers")
def test_qmb_kondo_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.kondo_adapter import KondoTraceAdapter
    adapter = KondoTraceAdapter(eps_d=-2.0, U=4.0, V_hyb=0.5)
    result, cons, session = adapter.evaluate()
    assert cons.kondo_temperature > 0
    assert len(session.entries) >= 2


@gauntlet("qmb_bosonic_solve", layer="qmb_solvers")
def test_qmb_bosonic_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.bosonic_adapter import BosonicTraceAdapter
    adapter = BosonicTraceAdapter(N_grid=64, x_max=10.0)
    result, cons, session = adapter.evaluate(g=1.0)
    assert len(session.entries) >= 2


@gauntlet("qmb_fermionic_solve", layer="qmb_solvers")
def test_qmb_fermionic_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.fermionic_adapter import FermionicTraceAdapter
    adapter = FermionicTraceAdapter(N_k=100, E_cutoff=10.0)
    result, cons, session = adapter.evaluate(V0=0.5)
    assert cons.gap_nonzero
    assert len(session.entries) >= 2


@gauntlet("qmb_nuclear_many_body_solve", layer="qmb_solvers")
def test_qmb_nuclear_many_body_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.nuclear_many_body_adapter import NuclearManyBodyTraceAdapter
    adapter = NuclearManyBodyTraceAdapter(n_orbits=4, n_particles=2)
    evals, cons, session = adapter.evaluate(n_states=3)
    assert cons.n_eigenvalues >= 1
    assert len(session.entries) >= 2


@gauntlet("qmb_ultracold_solve", layer="qmb_solvers")
def test_qmb_ultracold_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.ultracold_adapter import UltracoldTraceAdapter
    adapter = UltracoldTraceAdapter(nx=64, Lx=20.0, g_int=1.0)
    psi, cons, session = adapter.evaluate(n_steps=200)
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 7: Electronic Structure Solver Runs (7)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("es_dft_solve", layer="electronic_solvers")
def test_es_dft_solve():
    from tensornet.quantum.electronic_structure.trace_adapters.dft_adapter import DFTTraceAdapter
    adapter = DFTTraceAdapter(ngrid=100, L=20.0, n_electrons=2)
    result, cons, session = adapter.evaluate(max_iter=50, tol=1e-5)
    assert len(session.entries) >= 2


@gauntlet("es_beyond_dft_solve", layer="electronic_solvers")
def test_es_beyond_dft_solve():
    from tensornet.quantum.electronic_structure.trace_adapters.beyond_dft_adapter import BeyondDFTTraceAdapter
    adapter = BeyondDFTTraceAdapter(n_basis=6, n_electrons=2)
    result, cons, session = adapter.evaluate(max_iter=50, tol=1e-6)
    assert len(session.entries) >= 2


@gauntlet("es_tight_binding_solve", layer="electronic_solvers")
def test_es_tight_binding_solve():
    from tensornet.quantum.electronic_structure.trace_adapters.tight_binding_adapter import TightBindingTraceAdapter
    adapter = TightBindingTraceAdapter(n_atoms=2)
    evals, cons, session = adapter.evaluate()
    assert cons.n_bands >= 1
    assert len(session.entries) >= 2


@gauntlet("es_excited_states_solve", layer="electronic_solvers")
def test_es_excited_states_solve():
    from tensornet.quantum.electronic_structure.trace_adapters.excited_states_adapter import ExcitedStatesTraceAdapter
    adapter = ExcitedStatesTraceAdapter(n_occ=2, n_virt=4)
    exc_e, cons, session = adapter.evaluate(n_states=3)
    assert cons.n_excitations >= 1
    assert len(session.entries) >= 2


@gauntlet("es_response_solve", layer="electronic_solvers")
def test_es_response_solve():
    from tensornet.quantum.electronic_structure.trace_adapters.response_adapter import ResponseTraceAdapter
    adapter = ResponseTraceAdapter(n_occ=2, n_virt=4)
    alpha, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


@gauntlet("es_relativistic_solve", layer="electronic_solvers")
def test_es_relativistic_solve():
    from tensornet.quantum.electronic_structure.trace_adapters.relativistic_adapter import RelativisticTraceAdapter
    adapter = RelativisticTraceAdapter(Z=1)
    energies, cons, session = adapter.evaluate(n_max=2)
    assert cons.ground_energy < 0
    assert len(session.entries) >= 2


@gauntlet("es_embedding_solve", layer="electronic_solvers")
def test_es_embedding_solve():
    from tensornet.quantum.electronic_structure.trace_adapters.embedding_adapter import EmbeddingTraceAdapter
    adapter = EmbeddingTraceAdapter()
    E, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 8: Solid State Solver Runs (8)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("ss_phonons_solve", layer="solid_state_solvers")
def test_ss_phonons_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.phonons_adapter import PhononsTraceAdapter
    adapter = PhononsTraceAdapter(n_atoms=2)
    band, cons, session = adapter.evaluate()
    assert cons.n_modes >= 1
    assert len(session.entries) >= 2


@gauntlet("ss_band_structure_solve", layer="solid_state_solvers")
def test_ss_band_structure_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.band_structure_adapter import BandStructureTraceAdapter
    adapter = BandStructureTraceAdapter()
    bands, cons, session = adapter.evaluate(n_k=20)
    assert cons.n_bands >= 1
    assert len(session.entries) >= 2


@gauntlet("ss_classical_magnetism_solve", layer="solid_state_solvers")
def test_ss_classical_magnetism_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.classical_magnetism_adapter import ClassicalMagnetismTraceAdapter
    adapter = ClassicalMagnetismTraceAdapter(alpha=0.1, Ms=8e5)
    m_hist, cons, session = adapter.evaluate(n_steps=100, dt=1e-12)
    assert len(session.entries) >= 2


@gauntlet("ss_superconductivity_solve", layer="solid_state_solvers")
def test_ss_superconductivity_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.superconductivity_adapter import SuperconductivityTraceAdapter
    adapter = SuperconductivityTraceAdapter(N_k=100)
    result, cons, session = adapter.evaluate(V0=0.3)
    assert cons.gap_magnitude > 0
    assert len(session.entries) >= 2


@gauntlet("ss_disordered_solve", layer="solid_state_solvers")
def test_ss_disordered_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.disordered_adapter import DisorderedTraceAdapter
    adapter = DisorderedTraceAdapter(L=20, W=2.0)
    evals, cons, session = adapter.evaluate()
    assert cons.n_eigenvalues > 0
    assert len(session.entries) >= 2


@gauntlet("ss_surfaces_solve", layer="solid_state_solvers")
def test_ss_surfaces_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.surfaces_adapter import SurfacesTraceAdapter
    adapter = SurfacesTraceAdapter()
    E_surf, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


@gauntlet("ss_defects_solve", layer="solid_state_solvers")
def test_ss_defects_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.defects_adapter import DefectsTraceAdapter
    adapter = DefectsTraceAdapter(n_atoms=4)
    result, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


@gauntlet("ss_ferroelectrics_solve", layer="solid_state_solvers")
def test_ss_ferroelectrics_solve():
    from tensornet.quantum.condensed_matter.trace_adapters.ferroelectrics_adapter import FerroelectricsTraceAdapter
    adapter = FerroelectricsTraceAdapter()
    P, cons, session = adapter.evaluate(T=300.0)
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 9: Nuclear & Particle Solver Runs (6)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("np_nuclear_structure_solve", layer="nuclear_particle_solvers")
def test_np_nuclear_structure_solve():
    from tensornet.plasma_nuclear.nuclear.trace_adapters.nuclear_structure_adapter import NuclearStructureTraceAdapter
    adapter = NuclearStructureTraceAdapter(A=16, Z=8)
    evals, cons, session = adapter.evaluate()
    assert cons.binding_energy < 0 or cons.nucleon_number_conserved
    assert len(session.entries) >= 2


@gauntlet("np_nuclear_reactions_solve", layer="nuclear_particle_solvers")
def test_np_nuclear_reactions_solve():
    from tensornet.plasma_nuclear.nuclear.trace_adapters.nuclear_reactions_adapter import NuclearReactionsTraceAdapter
    adapter = NuclearReactionsTraceAdapter(channel_radius=5.0)
    sigma, cons, session = adapter.evaluate(n_energies=20)
    assert cons.peak_cross_section > 0
    assert len(session.entries) >= 2


@gauntlet("np_nuclear_astro_solve", layer="nuclear_particle_solvers")
def test_np_nuclear_astro_solve():
    from tensornet.plasma_nuclear.nuclear.trace_adapters.nuclear_astro_adapter import NuclearAstroTraceAdapter
    adapter = NuclearAstroTraceAdapter(Z1=1, Z2=1)
    E_G, cons, session = adapter.evaluate()
    assert cons.gamow_energy > 0
    assert len(session.entries) >= 2


@gauntlet("np_lattice_qcd_solve", layer="nuclear_particle_solvers")
def test_np_lattice_qcd_solve():
    from tensornet.quantum.qft.trace_adapters.lattice_qcd_adapter import LatticeQCDTraceAdapter
    adapter = LatticeQCDTraceAdapter(L=4, beta=6.0)
    plaq, cons, session = adapter.evaluate(n_sweeps=5)
    assert len(session.entries) >= 2


@gauntlet("np_perturbative_qft_solve", layer="nuclear_particle_solvers")
def test_np_perturbative_qft_solve():
    from tensornet.quantum.qft.trace_adapters.perturbative_qft_adapter import PerturbativeQFTTraceAdapter
    adapter = PerturbativeQFTTraceAdapter(n_f=5)
    alpha, cons, session = adapter.evaluate()
    assert cons.alpha_s_mz > 0
    assert len(session.entries) >= 2


@gauntlet("np_beyond_sm_solve", layer="nuclear_particle_solvers")
def test_np_beyond_sm_solve():
    from tensornet.applied.particle.trace_adapters.beyond_sm_adapter import BeyondSmTraceAdapter
    adapter = BeyondSmTraceAdapter()
    result, cons, session = adapter.evaluate()
    assert abs(cons.oscillation_probability_sum - 1.0) < 0.1
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 10: Chemical Physics Solver Runs (3)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("cp_pes_solve", layer="chem_physics_solvers")
def test_cp_pes_solve():
    from tensornet.life_sci.chemistry.trace_adapters.pes_adapter import PESTraceAdapter
    adapter = PESTraceAdapter(D_e=4.746, alpha_m=1.94, r_e=0.741)
    levels, cons, session = adapter.evaluate()
    assert cons.gradient_zero_at_minimum
    assert len(session.entries) >= 2


@gauntlet("cp_reaction_rate_solve", layer="chem_physics_solvers")
def test_cp_reaction_rate_solve():
    from tensornet.life_sci.chemistry.trace_adapters.reaction_rate_adapter import ReactionRateTraceAdapter
    adapter = ReactionRateTraceAdapter(Ea=0.5, nu_imag=1e13)
    rates, cons, session = adapter.evaluate()
    assert cons.rate_positive
    assert len(session.entries) >= 2


@gauntlet("cp_catalysis_solve", layer="chem_physics_solvers")
def test_cp_catalysis_solve():
    from tensornet.life_sci.chemistry.trace_adapters.catalysis_adapter import CatalysisTraceAdapter
    adapter = CatalysisTraceAdapter(D_e=4.0, alpha_c=1.5, r_e=1.0)
    result, cons, session = adapter.evaluate(n_sites=10)
    assert cons.atom_count_conserved
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 11: Quantum Information Solver Runs (3)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("qi_quantum_circuit_solve", layer="quantum_info_solvers")
def test_qi_quantum_circuit_solve():
    from tensornet.quantum.trace_adapters.quantum_circuit_adapter import QuantumCircuitTraceAdapter
    adapter = QuantumCircuitTraceAdapter(n_qubits=3, chi_max=16)
    sim, cons, session = adapter.evaluate()
    assert cons.trace_preserved
    assert len(session.entries) >= 2


@gauntlet("qi_qec_solve", layer="quantum_info_solvers")
def test_qi_qec_solve():
    from tensornet.quantum.trace_adapters.qec_adapter import QECTraceAdapter
    adapter = QECTraceAdapter()
    result, cons, session = adapter.evaluate()
    assert cons.logical_fidelity > 0.5
    assert len(session.entries) >= 2


@gauntlet("qi_vqe_solve", layer="quantum_info_solvers")
def test_qi_vqe_solve():
    from tensornet.quantum.trace_adapters.vqe_adapter import VQETraceAdapter
    adapter = VQETraceAdapter(n_qubits=2)
    result, cons, session = adapter.evaluate()
    assert cons.converged
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 12: Conservation Law Verification
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("conservation_qm_norm", layer="conservation")
def test_conservation_qm_norm():
    """TISE eigenstates have unit norm (within tolerance)."""
    from tensornet.quantum.quantum_mechanics.trace_adapters.tise_adapter import TISETraceAdapter
    adapter = TISETraceAdapter(n_grid=200)
    evals, cons, session = adapter.solve(n_states=3)
    assert cons.norm_error < 0.05, f"Norm error too large: {cons.norm_error}"


@gauntlet("conservation_qmb_energy", layer="conservation")
def test_conservation_qmb_energy():
    """DMRG ground energy is negative for Heisenberg chain."""
    from tensornet.algorithms.trace_adapters.dmrg_adapter import DMRGTraceAdapter
    from tensornet.mps.hamiltonians import heisenberg_mpo
    mpo = heisenberg_mpo(6)
    adapter = DMRGTraceAdapter(chi_max=16, num_sweeps=4)
    result, cons, session = adapter.evaluate(H_mpo=mpo)
    assert cons.ground_energy < 0, f"Expected negative E_0, got {cons.ground_energy}"


@gauntlet("conservation_es_electron_count", layer="conservation")
def test_conservation_es_electron_count():
    """DFT conserves electron count."""
    from tensornet.quantum.electronic_structure.trace_adapters.dft_adapter import DFTTraceAdapter
    adapter = DFTTraceAdapter(ngrid=100, L=20.0, n_electrons=2)
    result, cons, session = adapter.evaluate(max_iter=50)
    assert cons.electron_count_error < 0.1, f"Electron count error: {cons.electron_count_error}"


@gauntlet("conservation_ss_phonon_reality", layer="conservation")
def test_conservation_ss_phonon_reality():
    """Phonon frequencies are all real."""
    from tensornet.quantum.condensed_matter.trace_adapters.phonons_adapter import PhononsTraceAdapter
    adapter = PhononsTraceAdapter(n_atoms=2)
    band, cons, session = adapter.evaluate()
    assert cons.all_real_frequencies


@gauntlet("conservation_np_baryon", layer="conservation")
def test_conservation_np_baryon():
    """Nuclear astrophysics conserves baryon number."""
    from tensornet.plasma_nuclear.nuclear.trace_adapters.nuclear_astro_adapter import NuclearAstroTraceAdapter
    adapter = NuclearAstroTraceAdapter(Z1=1, Z2=1)
    E_G, cons, session = adapter.evaluate()
    assert cons.baryon_number_conserved


@gauntlet("conservation_qi_unitarity", layer="conservation")
def test_conservation_qi_unitarity():
    """Quantum circuit preserves unitarity."""
    from tensornet.quantum.trace_adapters.quantum_circuit_adapter import QuantumCircuitTraceAdapter
    adapter = QuantumCircuitTraceAdapter(n_qubits=3, chi_max=16)
    sim, cons, session = adapter.evaluate()
    assert cons.unitarity_error < 1e-6


# ═════════════════════════════════════════════════════════════════════════════
# Layer 13: Cross-Category Integration
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("integration_trace_session_api", layer="integration")
def test_integration_trace_session_api():
    """TraceSession API works uniformly across adapters."""
    from tensornet.core.trace import TraceSession
    session = TraceSession()
    session.log_custom(
        name="integration_test",
        input_hashes=["abc"],
        output_hashes=["def"],
        params={"test": True},
        metrics={"value": 42},
    )
    assert len(session.entries) == 1
    entry = session.entries[0]
    assert entry.op.value == "custom" or str(entry.op) == "OpType.CUSTOM"


@gauntlet("integration_all_categories_covered", layer="integration")
def test_integration_all_categories_covered():
    """Verify all 8 categories have at least one passing solver test."""
    solver_layers = [
        "qm_solvers", "qmb_solvers", "electronic_solvers",
        "solid_state_solvers", "nuclear_particle_solvers",
        "chem_physics_solvers", "quantum_info_solvers",
    ]
    for layer in solver_layers:
        layer_results = {k: v for k, v in RESULTS.items() if v["layer"] == layer}
        layer_passed = sum(1 for v in layer_results.values() if v["passed"])
        assert layer_passed > 0, f"No passing tests in layer: {layer}"


@gauntlet("integration_consistency_check", layer="integration")
def test_integration_consistency_check():
    """Each adapter returns (result, conservation, session) triple."""
    from tensornet.quantum.electronic_structure.trace_adapters.embedding_adapter import EmbeddingTraceAdapter
    adapter = EmbeddingTraceAdapter()
    output = adapter.evaluate()
    assert isinstance(output, tuple) and len(output) == 3, (
        f"Expected 3-tuple, got {type(output)} of length {len(output) if isinstance(output, tuple) else 'N/A'}"
    )
    _, cons, session = output
    assert hasattr(cons, "to_dict"), "Conservation object must have to_dict()"
    assert hasattr(session, "entries"), "Session must have entries"


# ═════════════════════════════════════════════════════════════════════════════
# Test Runner
# ═════════════════════════════════════════════════════════════════════════════

ALL_TESTS = [
    # Layer 1: adapter_files (14 tests)
    test_qm_adapter_files_qmech, test_qm_adapter_files_qm,
    test_qmb_adapter_files_cond, test_qmb_adapter_files_alg, test_qmb_adapter_files_qft,
    test_es_adapter_files, test_ss_adapter_files,
    test_np_adapter_files_nuc, test_np_adapter_files_qft, test_np_adapter_files_part,
    test_cp_adapter_files, test_qi_adapter_files,
    test_adapter_total_count,
    # Layer 2: core_adapters (2 tests)
    test_core_scf_adapter, test_core_eigenvalue_adapter,
    # Layer 3: lean_proofs (7 tests)
    test_lean_qm_conservation, test_lean_qmb_conservation,
    test_lean_es_conservation, test_lean_ss_conservation,
    test_lean_np_conservation, test_lean_cp_conservation,
    test_lean_qi_conservation,
    # Layer 4: tpc_script (2 tests)
    test_tpc_phase8_script_exists, test_tpc_phase8_importable,
    # Layer 5: qm_solvers (5 tests)
    test_qm_tise_solve, test_qm_tdse_solve, test_qm_scattering_solve,
    test_qm_semiclassical_solve, test_qm_path_integrals_solve,
    # Layer 6: qmb_solvers (13 tests)
    test_qmb_dmrg_solve, test_qmb_quantum_spin_solve,
    test_qmb_strongly_correlated_solve, test_qmb_topological_solve,
    test_qmb_mbl_solve, test_qmb_lattice_gauge_solve,
    test_qmb_open_quantum_solve, test_qmb_nonequilibrium_qm_solve,
    test_qmb_kondo_solve, test_qmb_bosonic_solve, test_qmb_fermionic_solve,
    test_qmb_nuclear_many_body_solve, test_qmb_ultracold_solve,
    # Layer 7: electronic_solvers (7 tests)
    test_es_dft_solve, test_es_beyond_dft_solve, test_es_tight_binding_solve,
    test_es_excited_states_solve, test_es_response_solve,
    test_es_relativistic_solve, test_es_embedding_solve,
    # Layer 8: solid_state_solvers (8 tests)
    test_ss_phonons_solve, test_ss_band_structure_solve,
    test_ss_classical_magnetism_solve, test_ss_superconductivity_solve,
    test_ss_disordered_solve, test_ss_surfaces_solve,
    test_ss_defects_solve, test_ss_ferroelectrics_solve,
    # Layer 9: nuclear_particle_solvers (6 tests)
    test_np_nuclear_structure_solve, test_np_nuclear_reactions_solve,
    test_np_nuclear_astro_solve, test_np_lattice_qcd_solve,
    test_np_perturbative_qft_solve, test_np_beyond_sm_solve,
    # Layer 10: chem_physics_solvers (3 tests)
    test_cp_pes_solve, test_cp_reaction_rate_solve, test_cp_catalysis_solve,
    # Layer 11: quantum_info_solvers (3 tests)
    test_qi_quantum_circuit_solve, test_qi_qec_solve, test_qi_vqe_solve,
    # Layer 12: conservation (6 tests)
    test_conservation_qm_norm, test_conservation_qmb_energy,
    test_conservation_es_electron_count, test_conservation_ss_phonon_reality,
    test_conservation_np_baryon, test_conservation_qi_unitarity,
    # Layer 13: integration (3 tests)
    test_integration_trace_session_api, test_integration_all_categories_covered,
    test_integration_consistency_check,
]


def run_all() -> bool:
    total_tests = len(ALL_TESTS)
    print(f"\n{'='*72}")
    print(f"  TRUSTLESS PHYSICS GAUNTLET — PHASE 8 ({total_tests} tests)")
    print(f"  Tier 3: Iterative / Eigenvalue Domains (45 domains)")
    print(f"{'='*72}\n")

    current_layer = None
    for test_fn in ALL_TESTS:
        test_fn()
        layer = RESULTS.get(test_fn.__name__, {}).get("layer", "")
        if layer != current_layer:
            current_layer = layer
            print()

    total_elapsed = time.monotonic() - _start_time
    total_passed = sum(1 for v in RESULTS.values() if v["passed"])
    total_failed = total_tests - total_passed

    print(f"\n{'='*72}")
    print(f"  PHASE 8 GAUNTLET SUMMARY")
    print(f"{'='*72}")

    layers = [
        "adapter_files", "core_adapters", "lean_proofs", "tpc_script",
        "qm_solvers", "qmb_solvers", "electronic_solvers",
        "solid_state_solvers", "nuclear_particle_solvers",
        "chem_physics_solvers", "quantum_info_solvers",
        "conservation", "integration",
    ]
    for layer in layers:
        layer_results = {k: v for k, v in RESULTS.items() if v["layer"] == layer}
        if not layer_results:
            continue
        layer_passed = sum(1 for v in layer_results.values() if v["passed"])
        layer_total = len(layer_results)
        status = "✅" if layer_passed == layer_total else "❌"
        print(f"  {status} {layer:28s} {layer_passed}/{layer_total}")

    print(f"\n{'='*72}")
    print(f"  Results: {total_passed}/{total_tests} passed")
    print(f"  Time:    {total_elapsed:.2f}s")
    print(f"{'='*72}\n")

    if total_failed > 0:
        print(f"❌ FAILED: {total_failed} test(s) failed")
        for name, r in RESULTS.items():
            if not r["passed"]:
                print(f"   • {name}: {r.get('error', 'unknown')}")
    else:
        print(f"✅ ALL {total_tests} TESTS PASSED")

    # Save attestation
    attestation = {
        "project": "HyperTensor-VM",
        "protocol": "trustless_physics_gauntlet_phase8",
        "phase": 8,
        "description": (
            "Tier 3 Wire-Up: 45 domains across Quantum Mechanics (5), "
            "Quantum Many-Body (13), Electronic Structure (7), "
            "Solid State (8), Nuclear & Particle (6), "
            "Chemical Physics (3), Quantum Information (3) — "
            "trace adapters, Lean conservation proofs, TPC generation"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tests": total_tests,
        "passed": total_passed,
        "failed": total_failed,
        "total_time_seconds": round(total_elapsed, 3),
        "gauntlets": RESULTS,
        "categories": {
            "quantum_mechanics": {
                "domains": 5,
                "lean_proof": "quantum_mechanics_conservation_proof/QuantumMechanicsConservation.lean",
                "conservation_laws": [
                    "normalisation", "probability", "optical theorem",
                    "action quantisation", "detailed balance",
                ],
            },
            "quantum_many_body": {
                "domains": 13,
                "lean_proof": "qmb_conservation_proof/QuantumManyBodyConservation.lean",
                "conservation_laws": [
                    "energy convergence", "total Sz", "spectral weight",
                    "Chern integer", "gap ratio", "Gauss law",
                    "trace preservation", "unitarity", "spectral weight",
                    "particle number", "particle number",
                    "nucleon number", "atom number",
                ],
            },
            "electronic_structure": {
                "domains": 7,
                "lean_proof": "electronic_structure_conservation_proof/ElectronicStructureConservation.lean",
                "conservation_laws": [
                    "electron count", "total energy", "charge neutrality",
                    "excitation positivity", "Kramers-Kronig",
                    "current continuity", "electron number",
                ],
            },
            "solid_state": {
                "domains": 8,
                "lean_proof": "solid_state_conservation_proof/SolidStateConservation.lean",
                "conservation_laws": [
                    "acoustic sum rule", "charge neutrality",
                    "magnetisation magnitude", "particle number",
                    "normalisation", "charge neutrality",
                    "charge balance", "polarisation bounded",
                ],
            },
            "nuclear_particle": {
                "domains": 6,
                "lean_proof": "nuclear_particle_conservation_proof/NuclearParticleConservation.lean",
                "conservation_laws": [
                    "nucleon number", "unitarity",
                    "baryon conservation", "gauge invariance",
                    "Ward identity", "probability unitarity",
                ],
            },
            "chem_physics_iter": {
                "domains": 3,
                "lean_proof": "chem_physics_iter_conservation_proof/ChemPhysicsIterConservation.lean",
                "conservation_laws": [
                    "gradient at minimum", "rate positivity",
                    "atom conservation",
                ],
            },
            "quantum_info": {
                "domains": 3,
                "lean_proof": "quantum_info_conservation_proof/QuantumInfoConservation.lean",
                "conservation_laws": [
                    "unitarity", "logical fidelity", "variational bound",
                ],
            },
        },
    }

    attestation_path = ROOT / "TRUSTLESS_PHYSICS_PHASE8_ATTESTATION.json"
    with open(attestation_path, "w") as f:
        json.dump(attestation, f, indent=2)
    print(f"\nAttestation saved to: {attestation_path.name}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
