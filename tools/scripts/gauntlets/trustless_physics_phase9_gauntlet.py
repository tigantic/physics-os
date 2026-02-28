#!/usr/bin/env python3
"""
Trustless Physics Gauntlet — Phase 9 Validation
================================================

Validates the Tier 4 Wire-Up: 26 domains across StatMech Stochastic (2),
Biophysics (6), Computational Methods (6), Quantum Information Extended (2),
and Special / Applied Physics (10) connected to the STARK proof pipeline via
trace adapters, Lean formal proofs, and TPC certificate generation.

Test Layers:
    1.  adapter_files:       All 26 trace adapters + 2 core adapters exist
    2.  core_adapters:       Stochastic & ML core adapters importable
    3.  lean_proofs:         6 category-level Lean proofs with expected theorems
    4.  tpc_script:          Phase 9 TPC generator exists, 26 domains registered
    5.  statmech_solvers:    Run 2 stat-mech stochastic adapters → trace
    6.  biophysics_solvers:  Run 6 biophysics domain adapters → trace
    7.  compmeth_solvers:    Run 6 computational methods adapters → trace
    8.  qinfo_ext_solvers:   Run 2 quantum information (ext) adapters → trace
    9.  special_solvers:     Run 10 special / applied physics adapters → trace
   10.  conservation:        Conservation law verification across key domains
   11.  integration:         Cross-category validation

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
logger = logging.getLogger("trustless_physics_phase9_gauntlet")

# ═════════════════════════════════════════════════════════════════════════════
# Gauntlet Framework
# ═════════════════════════════════════════════════════════════════════════════

RESULTS: dict[str, dict[str, Any]] = {}
_start_time = time.monotonic()


def gauntlet(name: str, layer: str = "phase9"):
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

STATMECH_ADAPTERS = [
    "equilibrium_mc_adapter.py",
    "monte_carlo_general_adapter.py",
]

BIOPHYSICS_ADAPTERS = [
    "protein_structure_adapter.py",
    "drug_design_adapter.py",
    "membrane_adapter.py",
    "nucleic_acids_adapter.py",
    "systems_biology_adapter.py",
    "neuroscience_adapter.py",
]

COMPMETH_ADAPTERS = [
    "optimization_adapter.py",
    "inverse_problems_adapter.py",
    "mesh_generation_adapter.py",
    "large_scale_linalg_adapter.py",
    "hpc_adapter.py",
]

MLPHYSICS_ADAPTERS = [
    "ml_physics_adapter.py",
]

QINFO_EXT_ADAPTERS = [
    "quantum_simulation_adapter.py",
    "quantum_crypto_adapter.py",
]

SPECIAL_RELATIVITY_ADAPTERS = [
    "special_relativity_adapter.py",
    "numerical_gr_adapter.py",
]

SPECIAL_APPLIED_ADAPTERS = [
    "astrodynamics_adapter.py",
]

ROBOTICS_ADAPTERS = [
    "robotics_adapter.py",
]

ACOUSTICS_ADAPTERS = [
    "acoustics_adapter.py",
]

BIOMEDICAL_ADAPTERS = [
    "biomedical_adapter.py",
]

ENVIRONMENTAL_ADAPTERS = [
    "environmental_adapter.py",
]

ENERGY_ADAPTERS = [
    "energy_systems_adapter.py",
]

MANUFACTURING_ADAPTERS = [
    "manufacturing_adapter.py",
]

SEMICONDUCTOR_ADAPTERS = [
    "semiconductor_adapter.py",
]


# ═════════════════════════════════════════════════════════════════════════════
# Helper
# ═════════════════════════════════════════════════════════════════════════════

def _check_adapter_dir(pkg: Path, adapters: list[str], label: str) -> None:
    assert pkg.exists(), f"Missing directory: {pkg}"
    assert (pkg / "__init__.py").exists(), f"Missing __init__.py in {pkg}"
    for fname in adapters:
        fpath = pkg / fname
        assert fpath.exists(), f"Missing adapter: {fpath}"
        assert fpath.stat().st_size > 500, f"Adapter too small: {fpath}"


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1: Trace Adapter File Existence
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("adapter_files_statmech", layer="adapter_files")
def test_adapter_files_statmech():
    _check_adapter_dir(ROOT / "ontic" / "statmech" / "trace_adapters",
                       STATMECH_ADAPTERS, "statmech")


@gauntlet("adapter_files_biophysics", layer="adapter_files")
def test_adapter_files_biophysics():
    _check_adapter_dir(ROOT / "ontic" / "biophysics" / "trace_adapters",
                       BIOPHYSICS_ADAPTERS, "biophysics")


@gauntlet("adapter_files_compmeth", layer="adapter_files")
def test_adapter_files_compmeth():
    _check_adapter_dir(ROOT / "ontic" / "computational_methods" / "trace_adapters",
                       COMPMETH_ADAPTERS, "computational_methods")


@gauntlet("adapter_files_mlphysics", layer="adapter_files")
def test_adapter_files_mlphysics():
    _check_adapter_dir(ROOT / "ontic" / "ml_physics" / "trace_adapters",
                       MLPHYSICS_ADAPTERS, "ml_physics")


@gauntlet("adapter_files_qinfo_ext", layer="adapter_files")
def test_adapter_files_qinfo_ext():
    _check_adapter_dir(ROOT / "ontic" / "quantum" / "trace_adapters",
                       QINFO_EXT_ADAPTERS, "quantum_info_ext")


@gauntlet("adapter_files_relativity", layer="adapter_files")
def test_adapter_files_relativity():
    _check_adapter_dir(ROOT / "ontic" / "relativity" / "trace_adapters",
                       SPECIAL_RELATIVITY_ADAPTERS, "relativity")


@gauntlet("adapter_files_astro", layer="adapter_files")
def test_adapter_files_astro():
    _check_adapter_dir(ROOT / "ontic" / "special_applied" / "trace_adapters",
                       SPECIAL_APPLIED_ADAPTERS, "astrodynamics")


@gauntlet("adapter_files_robotics", layer="adapter_files")
def test_adapter_files_robotics():
    _check_adapter_dir(ROOT / "ontic" / "robotics_physics" / "trace_adapters",
                       ROBOTICS_ADAPTERS, "robotics")


@gauntlet("adapter_files_acoustics", layer="adapter_files")
def test_adapter_files_acoustics():
    _check_adapter_dir(ROOT / "ontic" / "acoustics" / "trace_adapters",
                       ACOUSTICS_ADAPTERS, "acoustics")


@gauntlet("adapter_files_biomedical", layer="adapter_files")
def test_adapter_files_biomedical():
    _check_adapter_dir(ROOT / "ontic" / "biomedical" / "trace_adapters",
                       BIOMEDICAL_ADAPTERS, "biomedical")


@gauntlet("adapter_files_environmental", layer="adapter_files")
def test_adapter_files_environmental():
    _check_adapter_dir(ROOT / "ontic" / "environmental" / "trace_adapters",
                       ENVIRONMENTAL_ADAPTERS, "environmental")


@gauntlet("adapter_files_energy", layer="adapter_files")
def test_adapter_files_energy():
    _check_adapter_dir(ROOT / "ontic" / "energy" / "trace_adapters",
                       ENERGY_ADAPTERS, "energy")


@gauntlet("adapter_files_manufacturing", layer="adapter_files")
def test_adapter_files_manufacturing():
    _check_adapter_dir(ROOT / "ontic" / "manufacturing" / "trace_adapters",
                       MANUFACTURING_ADAPTERS, "manufacturing")


@gauntlet("adapter_files_semiconductor", layer="adapter_files")
def test_adapter_files_semiconductor():
    _check_adapter_dir(ROOT / "ontic" / "semiconductor" / "trace_adapters",
                       SEMICONDUCTOR_ADAPTERS, "semiconductor")


@gauntlet("adapter_total_count", layer="adapter_files")
def test_adapter_total_count():
    """Total Phase 9 adapters = 26."""
    total = (len(STATMECH_ADAPTERS) + len(BIOPHYSICS_ADAPTERS)
             + len(COMPMETH_ADAPTERS) + len(MLPHYSICS_ADAPTERS)
             + len(QINFO_EXT_ADAPTERS) + len(SPECIAL_RELATIVITY_ADAPTERS)
             + len(SPECIAL_APPLIED_ADAPTERS) + len(ROBOTICS_ADAPTERS)
             + len(ACOUSTICS_ADAPTERS) + len(BIOMEDICAL_ADAPTERS)
             + len(ENVIRONMENTAL_ADAPTERS) + len(ENERGY_ADAPTERS)
             + len(MANUFACTURING_ADAPTERS) + len(SEMICONDUCTOR_ADAPTERS))
    assert total == 26, f"Expected 26 adapters, got {total}"


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2: Core Adapters
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("core_stochastic_adapter", layer="core_adapters")
def test_core_stochastic_adapter():
    from ontic.core.stochastic_trace_adapter import StochasticTraceAdapter, StochasticConvergence
    assert hasattr(StochasticTraceAdapter, "run_traced")
    assert hasattr(StochasticConvergence, "to_dict") or hasattr(StochasticConvergence, "__dataclass_fields__")


@gauntlet("core_ml_adapter", layer="core_adapters")
def test_core_ml_adapter():
    from ontic.core.ml_trace_adapter import MLTraceAdapter, MLConvergence
    assert hasattr(MLTraceAdapter, "run_traced")
    assert hasattr(MLConvergence, "to_dict") or hasattr(MLConvergence, "__dataclass_fields__")


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3: Lean Proofs
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("lean_statmech_stochastic", layer="lean_proofs")
def test_lean_statmech_stochastic():
    path = ROOT / "statmech_stochastic_conservation_proof" / "StatMechStochasticConservation.lean"
    assert path.exists(), f"Missing: {path}"
    src = path.read_text()
    required = [
        "equilibrium_mc_detailed_balance",
        "mc_general_replica_exchange",
        "equilibrium_mc_energy_convergence",
        "mc_general_flat_histogram",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"


@gauntlet("lean_biophysics", layer="lean_proofs")
def test_lean_biophysics():
    path = ROOT / "biophysics_conservation_proof" / "BiophysicsConservation.lean"
    assert path.exists(), f"Missing: {path}"
    src = path.read_text()
    required = [
        "protein_angle_bounded",
        "drug_energy_components_sum",
        "membrane_bending_nonneg",
        "nucleic_mfe_nonpositive",
        "sysbio_species_nonneg",
        "neuro_voltage_bounded",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"


@gauntlet("lean_computational_methods", layer="lean_proofs")
def test_lean_computational_methods():
    path = ROOT / "computational_methods_conservation_proof" / "ComputationalMethodsConservation.lean"
    assert path.exists(), f"Missing: {path}"
    src = path.read_text()
    required = [
        "optimization_objective_decrease",
        "inverse_residual_decrease",
        "ml_loss_convergence",
        "mesh_two_to_one_balance",
        "linalg_lanczos_convergence",
        "hpc_reproducibility",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"


@gauntlet("lean_quantum_info_ext", layer="lean_proofs")
def test_lean_quantum_info_ext():
    path = ROOT / "quantum_info_ext_conservation_proof" / "QuantumInfoExtConservation.lean"
    assert path.exists(), f"Missing: {path}"
    src = path.read_text()
    required = [
        "qsim_fidelity_bound",
        "qsim_energy_conservation",
        "qcrypto_bell_violation",
        "qcrypto_key_rate_positive",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"


@gauntlet("lean_special_relativity", layer="lean_proofs")
def test_lean_special_relativity():
    path = ROOT / "special_relativity_conservation_proof" / "SpecialRelativityConservation.lean"
    assert path.exists(), f"Missing: {path}"
    src = path.read_text()
    required = [
        "sr_invariant_mass_conservation",
        "sr_lorentz_invariance",
        "gr_hamiltonian_constraint",
        "gr_adm_mass_conservation",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"


@gauntlet("lean_applied_physics", layer="lean_proofs")
def test_lean_applied_physics():
    path = ROOT / "applied_physics_conservation_proof" / "AppliedPhysicsConservation.lean"
    assert path.exists(), f"Missing: {path}"
    src = path.read_text()
    required = [
        "astro_energy_conservation",
        "robotics_ke_nonneg",
        "acoustics_energy_positive",
        "biomed_drug_mass_balance",
        "env_concentration_nonneg",
        "energy_efficiency_bounded",
        "mfg_enthalpy_balance",
        "semi_charge_neutrality",
    ]
    for thm in required:
        assert thm in src, f"Missing theorem: {thm}"


# ═════════════════════════════════════════════════════════════════════════════
# Layer 4: TPC Script
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("tpc_phase9_script_exists", layer="tpc_script")
def test_tpc_phase9_script_exists():
    script = ROOT / "scripts" / "tpc" / "generate_phase9.py"
    assert script.exists(), f"Missing: {script}"
    assert script.stat().st_size > 1000, "Script too small"


@gauntlet("tpc_phase9_importable", layer="tpc_script")
def test_tpc_phase9_importable():
    tpc_dir = ROOT / "scripts" / "tpc"
    sys.path.insert(0, str(tpc_dir))
    try:
        from generate_phase9 import DOMAIN_RUNNERS
        assert len(DOMAIN_RUNNERS) == 26, f"Expected 26 runners, got {len(DOMAIN_RUNNERS)}"
    finally:
        sys.path.pop(0)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 5: StatMech Stochastic Solver Runs (2)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("statmech_equilibrium_mc_solve", layer="statmech_solvers")
def test_statmech_equilibrium_mc_solve():
    from ontic.quantum.statmech.trace_adapters.equilibrium_mc_adapter import EquilibriumMCTraceAdapter
    adapter = EquilibriumMCTraceAdapter(L=8, temperature=2.269)
    result, cons, session = adapter.evaluate(n_sweeps=500, n_warmup=100)
    assert cons.detailed_balance
    assert cons.acceptance_rate > 0
    assert len(session.entries) >= 2


@gauntlet("statmech_monte_carlo_general_solve", layer="statmech_solvers")
def test_statmech_monte_carlo_general_solve():
    from ontic.quantum.statmech.trace_adapters.monte_carlo_general_adapter import MonteCarloGeneralTraceAdapter
    adapter = MonteCarloGeneralTraceAdapter(n_replicas=4)
    result, cons, session = adapter.evaluate(n_sweeps=200)
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 6: Biophysics Solver Runs (6)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("bio_protein_structure_solve", layer="biophysics_solvers")
def test_bio_protein_structure_solve():
    from ontic.life_sci.biophysics.trace_adapters.protein_structure_adapter import ProteinStructureTraceAdapter
    adapter = ProteinStructureTraceAdapter(sequence="AVILMFYW")
    result, cons, session = adapter.evaluate()
    assert cons.rg > 0
    assert len(session.entries) >= 2


@gauntlet("bio_drug_design_solve", layer="biophysics_solvers")
def test_bio_drug_design_solve():
    from ontic.life_sci.biophysics.trace_adapters.drug_design_adapter import DrugDesignTraceAdapter
    adapter = DrugDesignTraceAdapter()
    result, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


@gauntlet("bio_membrane_solve", layer="biophysics_solvers")
def test_bio_membrane_solve():
    from ontic.life_sci.biophysics.trace_adapters.membrane_adapter import MembraneTraceAdapter
    adapter = MembraneTraceAdapter(kappa=20.0)
    result, cons, session = adapter.evaluate()
    assert cons.bending_energy >= 0
    assert len(session.entries) >= 2


@gauntlet("bio_nucleic_acids_solve", layer="biophysics_solvers")
def test_bio_nucleic_acids_solve():
    from ontic.life_sci.biophysics.trace_adapters.nucleic_acids_adapter import NucleicAcidsTraceAdapter
    adapter = NucleicAcidsTraceAdapter(sequence="GGGAAACCC")
    result, cons, session = adapter.evaluate()
    assert cons.mfe <= 0
    assert len(session.entries) >= 2


@gauntlet("bio_systems_biology_solve", layer="biophysics_solvers")
def test_bio_systems_biology_solve():
    from ontic.life_sci.biophysics.trace_adapters.systems_biology_adapter import SystemsBiologyTraceAdapter
    adapter = SystemsBiologyTraceAdapter(n_species=2)
    result, cons, session = adapter.evaluate(t_max=10.0)
    assert cons.species_nonneg
    assert len(session.entries) >= 2


@gauntlet("bio_neuroscience_solve", layer="biophysics_solvers")
def test_bio_neuroscience_solve():
    from ontic.life_sci.biophysics.trace_adapters.neuroscience_adapter import NeuroscienceTraceAdapter
    adapter = NeuroscienceTraceAdapter(n_neurons=10, n_steps=1000)
    result, cons, session = adapter.evaluate()
    assert cons.spike_count >= 0
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 7: Computational Methods Solver Runs (6)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("comp_optimization_solve", layer="compmeth_solvers")
def test_comp_optimization_solve():
    from ontic.fluids.computational_methods.trace_adapters.optimization_adapter import OptimizationTraceAdapter
    adapter = OptimizationTraceAdapter(n_control_points=8)
    result, cons, session = adapter.evaluate(n_iterations=20)
    assert cons.objective_decreased
    assert len(session.entries) >= 2


@gauntlet("comp_inverse_problems_solve", layer="compmeth_solvers")
def test_comp_inverse_problems_solve():
    from ontic.fluids.computational_methods.trace_adapters.inverse_problems_adapter import InverseProblemsTraceAdapter
    adapter = InverseProblemsTraceAdapter(Nx=16, Ny=16)
    result, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


@gauntlet("comp_ml_physics_solve", layer="compmeth_solvers")
def test_comp_ml_physics_solve():
    from ontic.ml.ml_physics.trace_adapters.ml_physics_adapter import MLPhysicsTraceAdapter
    adapter = MLPhysicsTraceAdapter(n_colloc=50)
    result, cons, session = adapter.evaluate(n_steps=100)
    assert len(session.entries) >= 2


@gauntlet("comp_mesh_generation_solve", layer="compmeth_solvers")
def test_comp_mesh_generation_solve():
    from ontic.fluids.computational_methods.trace_adapters.mesh_generation_adapter import MeshGenerationTraceAdapter
    adapter = MeshGenerationTraceAdapter(max_level=5)
    result, cons, session = adapter.evaluate()
    assert cons.n_cells > 0
    assert len(session.entries) >= 2


@gauntlet("comp_large_scale_linalg_solve", layer="compmeth_solvers")
def test_comp_large_scale_linalg_solve():
    from ontic.fluids.computational_methods.trace_adapters.large_scale_linalg_adapter import LargeScaleLinAlgTraceAdapter
    adapter = LargeScaleLinAlgTraceAdapter(dim=100)
    result, cons, session = adapter.evaluate(num_iter=50)
    assert cons.eigenvalue_computed
    assert len(session.entries) >= 2


@gauntlet("comp_hpc_solve", layer="compmeth_solvers")
def test_comp_hpc_solve():
    from ontic.fluids.computational_methods.trace_adapters.hpc_adapter import HPCTraceAdapter
    adapter = HPCTraceAdapter()
    result, cons, session = adapter.evaluate(n_trials=5)
    assert cons.bit_exact
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 8: Quantum Information Extended Solver Runs (2)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("qinfo_quantum_simulation_solve", layer="qinfo_ext_solvers")
def test_qinfo_quantum_simulation_solve():
    from ontic.quantum.trace_adapters.quantum_simulation_adapter import QuantumSimulationTraceAdapter
    adapter = QuantumSimulationTraceAdapter(n_sites=2)
    result, cons, session = adapter.evaluate(n_steps=40)
    assert len(session.entries) >= 2


@gauntlet("qinfo_quantum_crypto_solve", layer="qinfo_ext_solvers")
def test_qinfo_quantum_crypto_solve():
    from ontic.quantum.trace_adapters.quantum_crypto_adapter import QuantumCryptoTraceAdapter
    adapter = QuantumCryptoTraceAdapter()
    result, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 9: Special / Applied Physics Solver Runs (10)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("special_relativity_solve", layer="special_solvers")
def test_special_relativity_solve():
    from ontic.astro.relativity.trace_adapters.special_relativity_adapter import SpecialRelativityTraceAdapter
    adapter = SpecialRelativityTraceAdapter()
    result, cons, session = adapter.evaluate()
    assert cons.invariant_mass_conserved
    assert len(session.entries) >= 2


@gauntlet("numerical_gr_solve", layer="special_solvers")
def test_numerical_gr_solve():
    from ontic.astro.relativity.trace_adapters.numerical_gr_adapter import NumericalGRTraceAdapter
    adapter = NumericalGRTraceAdapter(n=16, dx=0.5)
    result, cons, session = adapter.evaluate(n_steps=5)
    assert len(session.entries) >= 2


@gauntlet("astrodynamics_solve", layer="special_solvers")
def test_astrodynamics_solve():
    from ontic.applied.special_applied.trace_adapters.astrodynamics_adapter import AstrodynamicsTraceAdapter
    adapter = AstrodynamicsTraceAdapter()
    result, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


@gauntlet("robotics_solve", layer="special_solvers")
def test_robotics_solve():
    from ontic.applied.robotics_physics.trace_adapters.robotics_adapter import RoboticsTraceAdapter
    adapter = RoboticsTraceAdapter(n_links=3)
    result, cons, session = adapter.evaluate()
    assert cons.ke_nonneg
    assert len(session.entries) >= 2


@gauntlet("acoustics_solve", layer="special_solvers")
def test_acoustics_solve():
    from ontic.applied.acoustics.trace_adapters.acoustics_adapter import AcousticsTraceAdapter
    adapter = AcousticsTraceAdapter(radius=0.5)
    result, cons, session = adapter.evaluate(frequency=1000.0)
    assert cons.tl_positive
    assert len(session.entries) >= 2


@gauntlet("biomedical_solve", layer="special_solvers")
def test_biomedical_solve():
    from ontic.life_sci.biomedical.trace_adapters.biomedical_adapter import BiomedicalTraceAdapter
    adapter = BiomedicalTraceAdapter()
    result, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


@gauntlet("environmental_solve", layer="special_solvers")
def test_environmental_solve():
    from ontic.energy_env.environmental.trace_adapters.environmental_adapter import EnvironmentalTraceAdapter
    adapter = EnvironmentalTraceAdapter(Q=100.0, H=50.0, u=5.0)
    result, cons, session = adapter.evaluate()
    assert cons.concentration_nonneg
    assert len(session.entries) >= 2


@gauntlet("energy_systems_solve", layer="special_solvers")
def test_energy_systems_solve():
    from ontic.energy_env.energy.trace_adapters.energy_systems_adapter import EnergySystemsTraceAdapter
    adapter = EnergySystemsTraceAdapter(L=1e-4, nx=100)
    result, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


@gauntlet("manufacturing_solve", layer="special_solvers")
def test_manufacturing_solve():
    from ontic.materials.manufacturing.trace_adapters.manufacturing_adapter import ManufacturingTraceAdapter
    adapter = ManufacturingTraceAdapter()
    result, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


@gauntlet("semiconductor_solve", layer="special_solvers")
def test_semiconductor_solve():
    from ontic.quantum.semiconductor.trace_adapters.semiconductor_adapter import SemiconductorTraceAdapter
    adapter = SemiconductorTraceAdapter()
    result, cons, session = adapter.evaluate()
    assert len(session.entries) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# Layer 10: Conservation Law Verification
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("conservation_equilibrium_mc_detailed_balance", layer="conservation")
def test_conservation_equilibrium_mc_detailed_balance():
    from ontic.quantum.statmech.trace_adapters.equilibrium_mc_adapter import EquilibriumMCTraceAdapter
    adapter = EquilibriumMCTraceAdapter(L=8, temperature=2.269)
    _, cons, _ = adapter.evaluate(n_sweeps=500, n_warmup=100)
    assert cons.detailed_balance, "Detailed balance violated"
    assert 0.0 < cons.acceptance_rate <= 1.0, f"Bad acceptance rate: {cons.acceptance_rate}"


@gauntlet("conservation_protein_energy", layer="conservation")
def test_conservation_protein_energy():
    from ontic.life_sci.biophysics.trace_adapters.protein_structure_adapter import ProteinStructureTraceAdapter
    adapter = ProteinStructureTraceAdapter(sequence="AVILMFYW")
    _, cons, _ = adapter.evaluate()
    assert cons.rg > 0, "Radius of gyration must be positive"
    assert np.isfinite(cons.total_energy), "Energy must be finite"


@gauntlet("conservation_neuroscience_spike", layer="conservation")
def test_conservation_neuroscience_spike():
    from ontic.life_sci.biophysics.trace_adapters.neuroscience_adapter import NeuroscienceTraceAdapter
    adapter = NeuroscienceTraceAdapter(n_neurons=10, n_steps=1000)
    _, cons, _ = adapter.evaluate()
    assert cons.spike_count >= 0, "Spike count must be non-negative"


@gauntlet("conservation_mesh_balance", layer="conservation")
def test_conservation_mesh_balance():
    from ontic.fluids.computational_methods.trace_adapters.mesh_generation_adapter import MeshGenerationTraceAdapter
    adapter = MeshGenerationTraceAdapter(max_level=5)
    _, cons, _ = adapter.evaluate()
    assert cons.n_cells > 0, "Mesh must have cells"
    assert cons.two_one_balanced, "2:1 balance violated"


@gauntlet("conservation_sr_invariant_mass", layer="conservation")
def test_conservation_sr_invariant_mass():
    from ontic.astro.relativity.trace_adapters.special_relativity_adapter import SpecialRelativityTraceAdapter
    adapter = SpecialRelativityTraceAdapter()
    _, cons, _ = adapter.evaluate()
    assert cons.invariant_mass_conserved, "Invariant mass not conserved"


@gauntlet("conservation_environmental_nonneg", layer="conservation")
def test_conservation_environmental_nonneg():
    from ontic.energy_env.environmental.trace_adapters.environmental_adapter import EnvironmentalTraceAdapter
    adapter = EnvironmentalTraceAdapter(Q=100.0, H=50.0, u=5.0)
    _, cons, _ = adapter.evaluate()
    assert cons.concentration_nonneg, "Concentration must be non-negative"


# ═════════════════════════════════════════════════════════════════════════════
# Layer 11: Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("integration_trace_session_api", layer="integration")
def test_integration_trace_session_api():
    """TraceSession API works uniformly across adapters."""
    from ontic.core.trace import TraceSession
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
    """Verify all 5 main categories have at least one passing solver test."""
    solver_layers = [
        "statmech_solvers", "biophysics_solvers", "compmeth_solvers",
        "qinfo_ext_solvers", "special_solvers",
    ]
    for layer in solver_layers:
        layer_results = {k: v for k, v in RESULTS.items() if v["layer"] == layer}
        layer_passed = sum(1 for v in layer_results.values() if v["passed"])
        assert layer_passed > 0, f"No passing tests in layer: {layer}"


@gauntlet("integration_consistency_check", layer="integration")
def test_integration_consistency_check():
    """Each adapter returns (result, conservation, session) triple."""
    from ontic.life_sci.biomedical.trace_adapters.biomedical_adapter import BiomedicalTraceAdapter
    adapter = BiomedicalTraceAdapter()
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
    # Layer 1: adapter_files (15 tests)
    test_adapter_files_statmech, test_adapter_files_biophysics,
    test_adapter_files_compmeth, test_adapter_files_mlphysics,
    test_adapter_files_qinfo_ext, test_adapter_files_relativity,
    test_adapter_files_astro, test_adapter_files_robotics,
    test_adapter_files_acoustics, test_adapter_files_biomedical,
    test_adapter_files_environmental, test_adapter_files_energy,
    test_adapter_files_manufacturing, test_adapter_files_semiconductor,
    test_adapter_total_count,
    # Layer 2: core_adapters (2 tests)
    test_core_stochastic_adapter, test_core_ml_adapter,
    # Layer 3: lean_proofs (6 tests)
    test_lean_statmech_stochastic, test_lean_biophysics,
    test_lean_computational_methods, test_lean_quantum_info_ext,
    test_lean_special_relativity, test_lean_applied_physics,
    # Layer 4: tpc_script (2 tests)
    test_tpc_phase9_script_exists, test_tpc_phase9_importable,
    # Layer 5: statmech_solvers (2 tests)
    test_statmech_equilibrium_mc_solve, test_statmech_monte_carlo_general_solve,
    # Layer 6: biophysics_solvers (6 tests)
    test_bio_protein_structure_solve, test_bio_drug_design_solve,
    test_bio_membrane_solve, test_bio_nucleic_acids_solve,
    test_bio_systems_biology_solve, test_bio_neuroscience_solve,
    # Layer 7: compmeth_solvers (6 tests)
    test_comp_optimization_solve, test_comp_inverse_problems_solve,
    test_comp_ml_physics_solve, test_comp_mesh_generation_solve,
    test_comp_large_scale_linalg_solve, test_comp_hpc_solve,
    # Layer 8: qinfo_ext_solvers (2 tests)
    test_qinfo_quantum_simulation_solve, test_qinfo_quantum_crypto_solve,
    # Layer 9: special_solvers (10 tests)
    test_special_relativity_solve, test_numerical_gr_solve,
    test_astrodynamics_solve, test_robotics_solve, test_acoustics_solve,
    test_biomedical_solve, test_environmental_solve, test_energy_systems_solve,
    test_manufacturing_solve, test_semiconductor_solve,
    # Layer 10: conservation (6 tests)
    test_conservation_equilibrium_mc_detailed_balance,
    test_conservation_protein_energy,
    test_conservation_neuroscience_spike,
    test_conservation_mesh_balance,
    test_conservation_sr_invariant_mass,
    test_conservation_environmental_nonneg,
    # Layer 11: integration (3 tests)
    test_integration_trace_session_api,
    test_integration_all_categories_covered,
    test_integration_consistency_check,
]


def run_all() -> bool:
    total_tests = len(ALL_TESTS)
    print(f"\n{'='*72}")
    print(f"  TRUSTLESS PHYSICS GAUNTLET — PHASE 9 ({total_tests} tests)")
    print(f"  Tier 4: Stochastic / ML / Special Domains (26 domains)")
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
    print(f"  PHASE 9 GAUNTLET SUMMARY")
    print(f"{'='*72}")

    layers = [
        "adapter_files", "core_adapters", "lean_proofs", "tpc_script",
        "statmech_solvers", "biophysics_solvers", "compmeth_solvers",
        "qinfo_ext_solvers", "special_solvers",
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
        "protocol": "trustless_physics_gauntlet_phase9",
        "phase": 9,
        "description": (
            "Tier 4 Wire-Up: 26 domains across StatMech Stochastic (2), "
            "Biophysics (6), Computational Methods (6), "
            "Quantum Information Extended (2), "
            "Special / Applied Physics (10) — "
            "trace adapters, Lean conservation proofs, TPC generation"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tests": total_tests,
        "passed": total_passed,
        "failed": total_failed,
        "total_time_seconds": round(total_elapsed, 3),
        "gauntlets": RESULTS,
        "categories": {
            "statmech_stochastic": {
                "domains": 2,
                "lean_proof": "statmech_stochastic_conservation_proof/StatMechStochasticConservation.lean",
                "conservation_laws": [
                    "detailed balance", "energy convergence",
                    "replica exchange", "flat histogram",
                ],
            },
            "biophysics": {
                "domains": 6,
                "lean_proof": "biophysics_conservation_proof/BiophysicsConservation.lean",
                "conservation_laws": [
                    "Ramachandran angles", "binding energy",
                    "bending energy", "MFE", "stoichiometry",
                    "membrane voltage",
                ],
            },
            "computational_methods": {
                "domains": 6,
                "lean_proof": "computational_methods_conservation_proof/ComputationalMethodsConservation.lean",
                "conservation_laws": [
                    "objective decrease", "residual decrease",
                    "loss convergence", "2:1 balance",
                    "Lanczos convergence", "bit-exact reproducibility",
                ],
            },
            "quantum_info_extended": {
                "domains": 2,
                "lean_proof": "quantum_info_ext_conservation_proof/QuantumInfoExtConservation.lean",
                "conservation_laws": [
                    "Trotter fidelity", "energy conservation",
                    "Bell violation", "key rate positivity",
                ],
            },
            "special_applied": {
                "domains": 10,
                "lean_proof_relativity": "special_relativity_conservation_proof/SpecialRelativityConservation.lean",
                "lean_proof_applied": "applied_physics_conservation_proof/AppliedPhysicsConservation.lean",
                "conservation_laws": [
                    "invariant mass", "Lorentz invariance",
                    "Hamiltonian constraint", "ADM mass",
                    "orbital energy", "kinetic energy",
                    "acoustic energy", "drug mass balance",
                    "concentration non-neg", "efficiency bounds",
                    "enthalpy balance", "charge neutrality",
                ],
            },
        },
    }

    attestation_path = ROOT / "TRUSTLESS_PHYSICS_PHASE9_ATTESTATION.json"
    with open(attestation_path, "w") as f:
        json.dump(attestation, f, indent=2)
    print(f"\nAttestation saved to: {attestation_path.name}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
