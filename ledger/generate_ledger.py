#!/usr/bin/env python3
"""
Generate the full HyperTensor-VM Capability Ledger.

Produces:
  ledger/nodes/PHY-{PACK}.{NODE}.yaml   (140 files)
  ledger/index.yaml                      (aggregated index)

Based on the audit of 2026-02-08 against the Commercial Execution Plan taxonomy.
"""

from __future__ import annotations

import pathlib
import textwrap
from dataclasses import dataclass, field
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "ledger"
NODES_DIR = LEDGER / "nodes"


@dataclass
class Node:
    id: str
    name: str
    pack: str
    pack_name: str
    tier: str
    state: str
    description: str = ""
    owner: str = "unassigned"
    source_files: list[str] = field(default_factory=list)
    tests: dict[str, list[str]] = field(default_factory=dict)
    benchmarks: list[str] = field(default_factory=list)
    discretizations: list[str] = field(default_factory=list)
    solvers: list[str] = field(default_factory=list)
    qtt_hooks: list[str] = field(default_factory=list)
    notes: str = ""


def _yaml_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return "\n".join(f"  - \"{item}\"" for item in items)


def _yaml_test_dict(tests: dict[str, list[str]]) -> str:
    if not tests:
        return "{}"
    lines = []
    for category, items in tests.items():
        lines.append(f"  {category}:")
        for item in items:
            lines.append(f"    - \"{item}\"")
    return "\n".join(lines)


def node_to_yaml(n: Node) -> str:
    lines = [
        "# =============================================================================",
        f"# HyperTensor-VM Capability Ledger — {n.id}",
        "# Generated: 2026-02-08 | Schema: 1.0",
        "# =============================================================================",
        "",
        f'id: "{n.id}"',
        f'name: "{n.name}"',
        f'pack: "{n.pack}"',
        f'pack_name: "{n.pack_name}"',
        f'owner: "{n.owner}"',
        f'tier: "{n.tier}"',
        f'state: "{n.state}"',
        f'description: "{n.description}"',
        "",
        "source_files:",
        _yaml_list(n.source_files) if n.source_files else "  []",
        "",
        "tests:",
        _yaml_test_dict(n.tests) if n.tests else "  {}",
        "",
        "benchmarks:",
        _yaml_list(n.benchmarks) if n.benchmarks else "  []",
        "",
        "discretizations:",
        _yaml_list(n.discretizations) if n.discretizations else "  []",
        "",
        "solvers:",
        _yaml_list(n.solvers) if n.solvers else "  []",
        "",
        "qtt_hooks:",
        _yaml_list(n.qtt_hooks) if n.qtt_hooks else "  []",
        "",
        f'notes: "{n.notes}"',
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full 140-node taxonomy with audit data
# ---------------------------------------------------------------------------

NODES: list[Node] = []

# ===== Pack I: Classical Mechanics (6 nodes) =====
_pack, _pname = "I", "Classical Mechanics"

NODES += [
    Node("PHY-I.1", "Newtonian Particle Dynamics", _pack, _pname, "B", "V0.2",
         "N-body, rigid body, contact, integrators",
         source_files=["tensornet/guidance/trajectory.py"],
         tests={"smoke": ["test_140_domains"], "dedicated": ["test_ballistics"]},
         discretizations=["symplectic", "RK4"],
         solvers=["leapfrog", "velocity-Verlet"]),
    Node("PHY-I.2", "Lagrangian/Hamiltonian Mechanics", _pack, _pname, "B", "V0.2",
         "Symplectic and variational integrators",
         source_files=["tensornet/mechanics/symplectic.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["symplectic"],
         solvers=["Stoermer-Verlet", "implicit-midpoint"]),
    Node("PHY-I.3", "Continuum Mechanics", _pack, _pname, "B", "V0.2",
         "Elasticity, viscoelasticity, plasticity, fracture, contact, ALE",
         source_files=["tensornet/mechanics/continuum.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FEM"],
         solvers=["Newton-Raphson", "arc-length"]),
    Node("PHY-I.4", "Structural Mechanics", _pack, _pname, "B", "V0.2",
         "Beams, plates, shells, buckling, vibration, composites",
         source_files=["tensornet/mechanics/structural.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FEM"],
         solvers=["direct", "modal"]),
    Node("PHY-I.5", "Nonlinear Dynamics and Chaos", _pack, _pname, "B", "V0.2",
         "Lyapunov exponents, bifurcation, maps",
         source_files=["tensornet/cfd/hou_luo_ansatz.py"],
         tests={"smoke": ["test_140_domains"]},
         solvers=["RK45", "adaptive"]),
    Node("PHY-I.6", "Acoustics and Vibration", _pack, _pname, "B", "V0.2",
         "Wave/Helmholtz, scattering, vibroacoustics",
         source_files=["tensornet/acoustics/applied_acoustics.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FEM", "BEM"],
         solvers=["direct-frequency", "time-domain"]),
]

# ===== Pack II: Fluid Dynamics (10 nodes) =====
_pack, _pname = "II", "Fluid Dynamics"

NODES += [
    Node("PHY-II.1", "Incompressible Navier-Stokes", _pack, _pname, "A", "V0.2",
         "Projection, SIMPLE, PISO methods",
         source_files=["tensornet/cfd/ns_2d.py"],
         tests={"smoke": ["test_140_domains"], "dedicated": ["test_navier_stokes"],
                "integration": ["test_lid_driven_cavity", "test_taylor_green_benchmark"]},
         benchmarks=["lid_driven_cavity", "taylor_green_vortex", "blasius_validation"],
         discretizations=["FVM", "FEM", "spectral"],
         solvers=["projection", "SIMPLE", "PISO"],
         qtt_hooks=["velocity_field_qtt", "poisson_preconditioner_qtt"]),
    Node("PHY-II.2", "Compressible Flow", _pack, _pname, "A", "V0.2",
         "Riemann solvers, WENO, DG",
         source_files=["tensornet/cfd/euler_3d.py"],
         tests={"smoke": ["test_140_domains"],
                "integration": ["test_shu_osher_benchmark", "test_euler2d_mms", "test_euler3d_mms"]},
         benchmarks=["sod_shock_tube", "oblique_shock", "double_mach_reflection", "shu_osher"],
         discretizations=["FVM-WENO", "DG"],
         solvers=["Roe", "HLLC", "Rusanov"],
         qtt_hooks=["euler_field_qtt"]),
    Node("PHY-II.3", "Turbulence", _pack, _pname, "A", "V0.2",
         "DNS, LES, RANS, spectra",
         source_files=["tensornet/cfd/turbulence.py"],
         tests={"smoke": ["test_140_domains"],
                "integration": ["test_taylor_green_benchmark"]},
         benchmarks=["taylor_green_vortex", "sbli_benchmark"],
         discretizations=["FVM", "spectral"],
         solvers=["Smagorinsky", "k-epsilon", "k-omega-SST"]),
    Node("PHY-II.4", "Multiphase Flow", _pack, _pname, "A", "V0.2",
         "VOF, level set, phase field",
         source_files=["tensornet/multiphase/multiphase_flow.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FVM"],
         solvers=["VOF", "level-set", "phase-field"]),
    Node("PHY-II.5", "Reactive Flow / Combustion", _pack, _pname, "A", "V0.2",
         "Species transport, stiff kinetics",
         source_files=["tensornet/cfd/reactive_ns.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FVM"],
         solvers=["operator-split", "stiff-ODE"]),
    Node("PHY-II.6", "Rarefied Gas / Kinetic", _pack, _pname, "B", "V0.2",
         "Boltzmann/BGK, DSMC",
         source_files=["tensornet/cfd/dsmc.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["particle"],
         solvers=["DSMC", "BGK"]),
    Node("PHY-II.7", "Shallow Water / Geophysical Fluids", _pack, _pname, "B", "V0.2",
         "Coriolis, quasi-geostrophic",
         source_files=["tensornet/geophysics/oceanography.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FVM"],
         solvers=["Roe-shallow", "central"]),
    Node("PHY-II.8", "Non-Newtonian / Complex Fluids", _pack, _pname, "B", "V0.2",
         "Oldroyd-B, Bingham",
         source_files=["tensornet/cfd/non_newtonian.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FVM", "FEM"],
         solvers=["DEVSS", "log-conformation"]),
    Node("PHY-II.9", "Porous Media", _pack, _pname, "B", "V0.2",
         "Darcy/Richards, multiphase",
         source_files=["tensornet/porous_media/__init__.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FVM", "FEM"],
         solvers=["mixed-FEM", "TPFA"]),
    Node("PHY-II.10", "Free Surface / Interfacial", _pack, _pname, "B", "V0.2",
         "Surface tension, thin film",
         source_files=["tensornet/free_surface/__init__.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FVM"],
         solvers=["VOF", "ALE"]),
]

# ===== Pack III: Electromagnetism (7 nodes) =====
_pack, _pname = "III", "Electromagnetism"

NODES += [
    Node("PHY-III.1", "Electrostatics", _pack, _pname, "A", "V0.2",
         "Poisson, capacitance",
         source_files=["tensornet/em/electrostatics.py"],
         tests={"smoke": ["test_140_domains"], "cross": ["test_cross_domain_integration"]},
         discretizations=["FEM", "BEM"],
         solvers=["direct", "CG-AMG"]),
    Node("PHY-III.2", "Magnetostatics", _pack, _pname, "A", "V0.2",
         "Biot-Savart, inductance",
         source_files=["tensornet/em/magnetostatics.py"],
         tests={"smoke": ["test_140_domains"], "cross": ["test_cross_domain_integration"],
                "physics": ["test_physics_validation"]},
         discretizations=["FEM"],
         solvers=["A-phi", "T-Omega"]),
    Node("PHY-III.3", "Full Maxwell Time-Domain", _pack, _pname, "A", "V0.2",
         "FDTD, PML",
         source_files=["tensornet/em/wave_propagation.py"],
         tests={"smoke": ["test_140_domains"], "cross": ["test_cross_domain_integration"]},
         discretizations=["FDTD"],
         solvers=["Yee", "ADI-FDTD"],
         qtt_hooks=["maxwell_field_qtt"]),
    Node("PHY-III.4", "Frequency-Domain EM", _pack, _pname, "A", "V0.2",
         "Helmholtz, modes, scattering",
         source_files=["tensornet/em/frequency_domain.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FEM"],
         solvers=["direct", "iterative"]),
    Node("PHY-III.5", "Wave Propagation", _pack, _pname, "B", "V0.2",
         "Ray, parabolic, fibers, atmosphere",
         source_files=["tensornet/em/wave_propagation.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FDTD", "ray-tracing"],
         solvers=["split-step", "BPM"]),
    Node("PHY-III.6", "Computational Photonics", _pack, _pname, "B", "V0.2",
         "RCWA, PWE, plasmonics",
         source_files=["tensornet/em/computational_photonics.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["RCWA", "PWE"],
         solvers=["transfer-matrix", "FDFD"]),
    Node("PHY-III.7", "Antennas and Microwaves", _pack, _pname, "B", "V0.2",
         "MoM/FDTD/FEM hybrids",
         source_files=["tensornet/em/antenna_microwave.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]},
         discretizations=["MoM", "FDTD", "FEM"],
         solvers=["hybrid-MoM-FDTD"]),
]

# ===== Pack IV: Optics and Photonics (4 nodes) =====
_pack, _pname = "IV", "Optics and Photonics"

NODES += [
    Node("PHY-IV.1", "Physical Optics", _pack, _pname, "B", "V0.2",
         "Diffraction, coherence, polarization",
         source_files=["tensornet/optics/physical_optics.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-IV.2", "Quantum Optics", _pack, _pname, "B", "V0.2",
         "Master equation, trajectories",
         source_files=["tensornet/optics/quantum_optics.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]}),
    Node("PHY-IV.3", "Laser Physics", _pack, _pname, "B", "V0.2",
         "Rate equations, resonators",
         source_files=["tensornet/optics/laser_physics.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-IV.4", "Ultrafast Optics", _pack, _pname, "B", "V0.2",
         "NLSE split-step, HHG interfaces",
         source_files=["tensornet/optics/ultrafast_optics.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]}),
]

# ===== Pack V: Thermodynamics and Statistical Mechanics (6 nodes) =====
_pack, _pname = "V", "Thermodynamics and Statistical Mechanics"

NODES += [
    Node("PHY-V.1", "Equilibrium Statistical Mechanics", _pack, _pname, "B", "V0.2",
         "Ising/Potts/XY, Monte Carlo",
         source_files=["tensornet/statmech/equilibrium.py"],
         tests={"smoke": ["test_140_domains"], "physics": ["test_physics_validation"]}),
    Node("PHY-V.2", "Non-Equilibrium Statistical Mechanics", _pack, _pname, "B", "V0.2",
         "Master equation, Fokker-Planck",
         source_files=["tensornet/statmech/non_equilibrium.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-V.3", "Molecular Dynamics", _pack, _pname, "B", "V0.2",
         "Force fields, thermostats, sampling",
         source_files=["tensornet/md/engine.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]}),
    Node("PHY-V.4", "Monte Carlo Methods", _pack, _pname, "B", "V0.2",
         "MCMC, PIMC, QMC entrypoints",
         source_files=["tensornet/statmech/monte_carlo.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]}),
    Node("PHY-V.5", "Heat Transfer", _pack, _pname, "A", "V0.2",
         "Conduction, convection, radiation",
         source_files=["tensornet/heat_transfer/radiation.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FVM", "FEM"],
         solvers=["implicit-Euler", "Crank-Nicolson"],
         qtt_hooks=["thermal_field_qtt"]),
    Node("PHY-V.6", "Lattice Models and Spin Systems", _pack, _pname, "B", "V0.2",
         "TRG/TNR hooks",
         source_files=["tensornet/mps/hamiltonians.py"],
         tests={"smoke": ["test_140_domains"], "dedicated": ["test_mpo_hamiltonians"]},
         benchmarks=["heisenberg_ground_state", "tfim_ground_state"],
         qtt_hooks=["spin_chain_mpo"]),
]

# ===== Pack VI: Quantum Mechanics — Single/Few-Body (5 nodes) =====
_pack, _pname = "VI", "Quantum Mechanics (Single/Few-Body)"

NODES += [
    Node("PHY-VI.1", "Time-Independent Schrodinger", _pack, _pname, "B", "V0.2",
         "Shooting, spectral, DVR",
         source_files=["tensornet/quantum_mechanics/stationary.py"],
         tests={"smoke": ["test_140_domains"], "physics": ["test_physics_validation"]}),
    Node("PHY-VI.2", "TDSE Propagation", _pack, _pname, "B", "V0.2",
         "Split-operator, Chebyshev",
         source_files=["tensornet/quantum_mechanics/propagator.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VI.3", "Scattering Theory", _pack, _pname, "B", "V0.2",
         "Partial waves, T-matrix, R-matrix",
         source_files=["tensornet/qm/scattering.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VI.4", "Semiclassical Methods", _pack, _pname, "B", "V0.2",
         "IVR, surface hopping interfaces",
         source_files=["tensornet/qm/semiclassical_wkb.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VI.5", "Path Integrals", _pack, _pname, "B", "V0.2",
         "PIMC, ring polymer MD",
         source_files=["tensornet/quantum_mechanics/path_integrals.py"],
         tests={"smoke": ["test_140_domains"]}),
]

# ===== Pack VII: Quantum Many-Body Physics (13 nodes) =====
_pack, _pname = "VII", "Quantum Many-Body Physics"

NODES += [
    Node("PHY-VII.1", "Tensor Network Methods", _pack, _pname, "A", "V0.2",
         "MPS/MPO, PEPS, MERA, iDMRG",
         source_files=["tensornet/algorithms/dmrg.py", "tensornet/algorithms/tebd.py",
                        "tensornet/algorithms/tdvp.py", "tensornet/algorithms/lanczos.py"],
         tests={"smoke": ["test_140_domains"],
                "integration": ["test_dmrg_physics"],
                "dedicated": ["test_qtt_pipeline"]},
         benchmarks=["compare_tenpy", "heisenberg_ground_state"],
         qtt_hooks=["mps_compression", "mpo_application"]),
    Node("PHY-VII.2", "Quantum Spin Systems", _pack, _pname, "A", "V0.2",
         "DMRG, QMC, ED",
         source_files=["tensornet/mps/hamiltonians.py"],
         tests={"smoke": ["test_140_domains"], "dedicated": ["test_mpo_hamiltonians"]},
         benchmarks=["tfim_ground_state"],
         qtt_hooks=["spin_hamiltonian_mpo"]),
    Node("PHY-VII.3", "Strongly Correlated Electrons", _pack, _pname, "B", "V0.2",
         "Hubbard, DMFT hooks",
         source_files=["tensornet/condensed_matter/strongly_correlated.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VII.4", "Topological Phases", _pack, _pname, "B", "V0.2",
         "Invariants, entanglement",
         source_files=["tensornet/condensed_matter/topological_phases.py"],
         tests={"smoke": ["test_140_domains"], "cross": ["test_cross_domain_integration"]}),
    Node("PHY-VII.5", "MBL and Disorder", _pack, _pname, "B", "V0.2",
         "Shift-invert ED, statistics",
         source_files=["tensornet/condensed_matter/mbl_disorder.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VII.6", "Lattice Gauge Theory", _pack, _pname, "B", "V0.2",
         "Wilson, Kogut-Susskind",
         source_files=["yangmills/su2.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VII.7", "Open Quantum Systems", _pack, _pname, "B", "V0.2",
         "Lindblad MPO",
         source_files=["tensornet/condensed_matter/open_quantum.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VII.8", "Non-Equilibrium Quantum Dynamics", _pack, _pname, "B", "V0.2",
         "Quenches, Floquet",
         source_files=["tensornet/condensed_matter/nonequilibrium_qm.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VII.9", "Quantum Impurity", _pack, _pname, "B", "V0.2",
         "NRG, CT-QMC hooks",
         source_files=["tensornet/condensed_matter/kondo_impurity.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VII.10", "Bosonic Many-Body", _pack, _pname, "B", "V0.2",
         "Bose-Hubbard, GPE",
         source_files=["tensornet/condensed_matter/bosonic.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VII.11", "Fermionic Systems", _pack, _pname, "B", "V0.2",
         "BCS, AFQMC/DMRG",
         source_files=["tensornet/condensed_matter/fermionic.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VII.12", "Nuclear Many-Body", _pack, _pname, "B", "V0.2",
         "IMSRG, CC hooks",
         source_files=["tensornet/condensed_matter/nuclear_many_body.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VII.13", "Ultracold Atoms", _pack, _pname, "B", "V0.2",
         "Optical lattices, SOC",
         source_files=["tensornet/condensed_matter/ultracold_atoms.py"],
         tests={"smoke": ["test_140_domains"]}),
]

# ===== Pack VIII: Electronic Structure (7 nodes) =====
_pack, _pname = "VIII", "Electronic Structure and Quantum Chemistry"

NODES += [
    Node("PHY-VIII.1", "DFT", _pack, _pname, "A", "V0.2",
         "Kohn-Sham, SCF",
         source_files=["tensornet/electronic_structure/dft.py"],
         tests={"smoke": ["test_140_domains"]},
         solvers=["SCF", "DIIS"],
         qtt_hooks=["density_field_qtt"]),
    Node("PHY-VIII.2", "Beyond-DFT Correlated Methods", _pack, _pname, "B", "V0.2",
         "HF/MP2/CC/CI, DMRG-FCI",
         source_files=["tensornet/electronic_structure/beyond_dft.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VIII.3", "Semi-Empirical and Tight-Binding", _pack, _pname, "B", "V0.2",
         "O(N), Green's functions",
         source_files=["tensornet/electronic_structure/tight_binding.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VIII.4", "Excited States", _pack, _pname, "B", "V0.2",
         "TDDFT, GW, BSE",
         source_files=["tensornet/electronic_structure/excited_states.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VIII.5", "Response Properties", _pack, _pname, "B", "V0.2",
         "DFPT, spectra",
         source_files=["tensornet/electronic_structure/response.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VIII.6", "Relativistic Electronic Structure", _pack, _pname, "B", "V0.2",
         "SOC, 4c/2c",
         source_files=["tensornet/electronic_structure/relativistic.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-VIII.7", "Quantum Embedding", _pack, _pname, "B", "V0.2",
         "DFT+DMFT, QM/MM",
         source_files=["tensornet/electronic_structure/embedding.py"],
         tests={"smoke": ["test_140_domains"]}),
]

# ===== Pack IX: Solid State / Condensed Matter (8 nodes) =====
_pack, _pname = "IX", "Solid State / Condensed Matter"

NODES += [
    Node("PHY-IX.1", "Phonons and Lattice Dynamics", _pack, _pname, "B", "V0.2",
         "Phonon dispersion, thermal properties",
         source_files=["tensornet/condensed_matter/phonons.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"],
                "cross": ["test_cross_domain_integration"]}),
    Node("PHY-IX.2", "Band Structure and Transport", _pack, _pname, "B", "V0.2",
         "Wannier, BTE, NEGF hooks",
         source_files=["tensornet/condensed_matter/band_structure.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]}),
    Node("PHY-IX.3", "Magnetism", _pack, _pname, "B", "V0.2",
         "LLG, micromagnetics",
         source_files=["tensornet/condensed_matter/classical_magnetism.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]}),
    Node("PHY-IX.4", "Superconductivity", _pack, _pname, "B", "V0.2",
         "Eliashberg, BdG",
         source_files=["tensornet/condensed_matter/fermionic.py"],
         tests={"smoke": ["test_140_domains"]},
         notes="Shares source with PHY-VII.11"),
    Node("PHY-IX.5", "Disordered Systems", _pack, _pname, "B", "V0.2",
         "KPM, transfer matrix",
         source_files=["tensornet/condensed_matter/disordered.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-IX.6", "Surfaces and Interfaces", _pack, _pname, "B", "V0.2",
         "Slab methods, Green's functions",
         source_files=["tensornet/condensed_matter/surfaces_interfaces.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]}),
    Node("PHY-IX.7", "Defects in Solids", _pack, _pname, "B", "V0.2",
         "NEB, dislocations",
         source_files=["tensornet/condensed_matter/defects.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-IX.8", "Ferroelectrics and Multiferroics", _pack, _pname, "B", "V0.2",
         "Berry phase polarization",
         source_files=["tensornet/condensed_matter/ferroelectrics.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"],
                "cross": ["test_cross_domain_integration"],
                "physics": ["test_physics_validation"]}),
]

# ===== Pack X: Nuclear and Particle Physics (6 nodes) =====
_pack, _pname = "X", "Nuclear and Particle Physics"

NODES += [
    Node("PHY-X.1", "Nuclear Structure", _pack, _pname, "B", "V0.2",
         "CI, CC, nuclear DFT",
         source_files=["tensornet/nuclear/structure.py"],
         tests={"smoke": ["test_140_domains"], "cross": ["test_cross_domain_integration"]}),
    Node("PHY-X.2", "Nuclear Reactions", _pack, _pname, "B", "V0.2",
         "Coupled channels, optical model",
         source_files=["tensornet/nuclear/reactions.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-X.3", "Nuclear Astrophysics", _pack, _pname, "B", "V0.2",
         "Reaction networks + hydro coupling",
         source_files=["tensornet/nuclear/astrophysics.py"],
         tests={"smoke": ["test_140_domains"], "cross": ["test_cross_domain_integration"]}),
    Node("PHY-X.4", "Lattice QCD", _pack, _pname, "B", "V0.2",
         "HMC, multigrid interfaces",
         source_files=["tensornet/qft/lattice_qcd.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-X.5", "Perturbative QFT", _pack, _pname, "B", "V0.2",
         "Diagram eval, sector decomposition hooks",
         source_files=["tensornet/qft/perturbative.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-X.6", "Beyond Standard Model", _pack, _pname, "B", "V0.2",
         "Boltzmann scans, oscillations",
         source_files=["tensornet/particle/beyond_sm.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"],
                "cross": ["test_cross_domain_integration"],
                "physics": ["test_physics_validation"]}),
]

# ===== Pack XI: Plasma Physics (8 nodes) =====
_pack, _pname = "XI", "Plasma Physics"

NODES += [
    Node("PHY-XI.1", "Ideal MHD", _pack, _pname, "A", "V0.2",
         "Ideal magnetohydrodynamics",
         source_files=["tensornet/fusion/tokamak.py"],
         tests={"smoke": ["test_140_domains"], "dedicated": ["test_fusion"]},
         discretizations=["FVM"],
         solvers=["Roe-MHD", "HLL-MHD"],
         qtt_hooks=["mhd_field_qtt"]),
    Node("PHY-XI.2", "Resistive/Extended MHD", _pack, _pname, "A", "V0.2",
         "Resistive, Hall MHD",
         source_files=["tensornet/plasma/extended_mhd.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["FVM", "FEM"],
         solvers=["implicit-MHD"]),
    Node("PHY-XI.3", "Kinetic Theory Plasma", _pack, _pname, "A", "V0.2",
         "Vlasov, PIC, continuum",
         source_files=["tensornet/cfd/fast_vlasov_5d.py"],
         tests={"smoke": ["test_140_domains"]},
         discretizations=["PIC", "continuum-Vlasov"],
         solvers=["Boris-pusher", "semi-Lagrangian"],
         qtt_hooks=["vlasov_distribution_qtt"]),
    Node("PHY-XI.4", "Gyrokinetics", _pack, _pname, "B", "V0.2",
         "5D gyrokinetic equations",
         source_files=["tensornet/plasma/gyrokinetics.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XI.5", "Magnetic Reconnection", _pack, _pname, "B", "V0.2",
         "MHD and kinetic reconnection",
         source_files=["tensornet/plasma/magnetic_reconnection.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XI.6", "Laser-Plasma Interaction", _pack, _pname, "B", "V0.2",
         "PIC, rad-hydro hooks",
         source_files=["tensornet/plasma/laser_plasma.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XI.7", "Dusty Plasmas", _pack, _pname, "B", "V0.2",
         "Charged dust dynamics",
         source_files=["tensornet/plasma/dusty_plasmas.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]}),
    Node("PHY-XI.8", "Space and Astrophysical Plasma", _pack, _pname, "B", "V0.2",
         "Global MHD, cosmic rays",
         source_files=["tensornet/plasma/space_plasma.py"],
         tests={"smoke": ["test_140_domains"]}),
]

# ===== Pack XII: Astrophysics and Cosmology (6 nodes) =====
_pack, _pname = "XII", "Astrophysics and Cosmology"

NODES += [
    Node("PHY-XII.1", "Stellar Structure and Evolution", _pack, _pname, "B", "V0.2",
         "1D + reaction network",
         source_files=["tensornet/astro/stellar_structure.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XII.2", "Compact Objects", _pack, _pname, "B", "V0.2",
         "TOV, accretion, GRMHD hooks",
         source_files=["tensornet/astro/compact_objects.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XII.3", "Gravitational Waves", _pack, _pname, "B", "V0.2",
         "PN/EOB + NR interfaces",
         source_files=["tensornet/astro/gravitational_waves.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XII.4", "Cosmological Simulations", _pack, _pname, "B", "V0.2",
         "N-body, AMR hydro",
         source_files=["tensornet/astro/cosmological_sims.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XII.5", "CMB and Early Universe", _pack, _pname, "C", "V0.2",
         "Boltzmann codes hooks",
         source_files=["tensornet/astro/cmb_early_universe.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XII.6", "Radiative Transfer Astrophysical", _pack, _pname, "B", "V0.2",
         "MC, S_N, diffusion",
         source_files=["tensornet/astro/radiative_transfer.py"],
         tests={"smoke": ["test_140_domains"]}),
]

# ===== Pack XIII: Geophysics and Earth Science (6 nodes) =====
_pack, _pname = "XIII", "Geophysics and Earth Science"

NODES += [
    Node("PHY-XIII.1", "Seismology", _pack, _pname, "B", "V0.2",
         "Elastic waves, FWI interfaces",
         source_files=["tensornet/geophysics/seismology.py"],
         tests={"smoke": ["test_140_domains"], "cross": ["test_cross_domain_integration"]}),
    Node("PHY-XIII.2", "Mantle Convection", _pack, _pname, "B", "V0.2",
         "Stokes with variable viscosity",
         source_files=["tensornet/geophysics/mantle_convection.py"],
         tests={"smoke": ["test_140_domains"], "cross": ["test_cross_domain_integration"]}),
    Node("PHY-XIII.3", "Geomagnetism and Dynamo", _pack, _pname, "B", "V0.2",
         "Rotating MHD shells",
         source_files=["tensornet/geophysics/geodynamo.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XIII.4", "Atmospheric Physics", _pack, _pname, "B", "V0.2",
         "Radiation, chemistry hooks",
         source_files=["tensornet/geophysics/atmosphere.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XIII.5", "Oceanography", _pack, _pname, "B", "V0.2",
         "Primitive equations",
         source_files=["tensornet/geophysics/oceanography.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XIII.6", "Glaciology", _pack, _pname, "B", "V0.2",
         "SIA and full Stokes",
         source_files=["tensornet/geophysics/glaciology.py"],
         tests={"smoke": ["test_140_domains"]}),
]

# ===== Pack XIV: Materials Science (7 nodes) =====
_pack, _pname = "XIV", "Materials Science"

NODES += [
    Node("PHY-XIV.1", "First-Principles Materials Design", _pack, _pname, "C", "V0.2",
         "Workflow engine hooks",
         source_files=["tensornet/materials/first_principles_design.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XIV.2", "Mechanical Properties", _pack, _pname, "B", "V0.2",
         "Elastic constants, fracture, fatigue",
         source_files=["tensornet/materials/mechanical_properties.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"],
                "cross": ["test_cross_domain_integration"]}),
    Node("PHY-XIV.3", "Phase-Field Methods", _pack, _pname, "B", "V0.2",
         "Cahn-Hilliard, Allen-Cahn",
         source_files=["tensornet/phase_field/__init__.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XIV.4", "Microstructure Evolution", _pack, _pname, "B", "V0.2",
         "Potts, KMC",
         source_files=["tensornet/materials/microstructure.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XIV.5", "Radiation Damage", _pack, _pname, "B", "V0.2",
         "BCA/MD cascades/OKMC",
         source_files=["tensornet/materials/radiation_damage.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XIV.6", "Polymers and Soft Matter", _pack, _pname, "B", "V0.2",
         "SCFT, coarse-grained",
         source_files=["tensornet/materials/polymers_soft_matter.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XIV.7", "Ceramics and High-Temperature Materials", _pack, _pname, "B", "V0.2",
         "High-temp material properties",
         source_files=["tensornet/materials/ceramics.py"],
         tests={"smoke": ["test_140_domains"]}),
]

# ===== Pack XV: Chemical Physics (7 nodes) =====
_pack, _pname = "XV", "Chemical Physics and Reaction Dynamics"

NODES += [
    Node("PHY-XV.1", "Potential Energy Surfaces", _pack, _pname, "B", "V0.2",
         "NEB, ML PES hooks",
         source_files=["tensornet/chemistry/pes.py"],
         tests={"smoke": ["test_140_domains"],
                "cross": ["test_cross_domain_integration"],
                "physics": ["test_physics_validation"]}),
    Node("PHY-XV.2", "Reaction Rate Theory", _pack, _pname, "B", "V0.2",
         "TST, instantons",
         source_files=["tensornet/chemistry/reaction_rate.py"],
         tests={"smoke": ["test_140_domains"], "cross": ["test_cross_domain_integration"]}),
    Node("PHY-XV.3", "Quantum Reaction Dynamics", _pack, _pname, "B", "V0.2",
         "Wavepackets, MCTDH hooks",
         source_files=["tensornet/chemistry/quantum_reactive.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XV.4", "Nonadiabatic Dynamics", _pack, _pname, "B", "V0.2",
         "Surface hopping, AIMS hooks",
         source_files=["tensornet/chemistry/nonadiabatic.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XV.5", "Photochemistry", _pack, _pname, "B", "V0.2",
         "Excited state dynamics",
         source_files=["tensornet/chemistry/photochemistry.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XV.6", "Catalysis", _pack, _pname, "B", "V0.2",
         "Microkinetics, KMC",
         source_files=["tensornet/fusion/resonant_catalysis.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XV.7", "Spectroscopy", _pack, _pname, "B", "V0.2",
         "IR/Raman/NMR/XAS/XPS",
         source_files=["tensornet/chemistry/spectroscopy.py"],
         tests={"smoke": ["test_140_domains"]}),
]

# ===== Pack XVI: Biophysics (6 nodes) =====
_pack, _pname = "XVI", "Biophysics and Computational Biology"

NODES += [
    Node("PHY-XVI.1", "Protein Structure and Dynamics", _pack, _pname, "B", "V0.2",
         "MD workflows",
         source_files=["tensornet/md/engine.py"],
         tests={"smoke": ["test_140_domains"]},
         notes="Shares source with PHY-V.3"),
    Node("PHY-XVI.2", "Drug Design and Binding", _pack, _pname, "C", "V0.2",
         "Docking, FEP/TI hooks",
         source_files=["tensornet/chemistry/pes.py"],
         tests={"smoke": ["test_140_domains"]},
         notes="Shares source with PHY-XV.1"),
    Node("PHY-XVI.3", "Membrane Biophysics", _pack, _pname, "B", "V0.2",
         "CGMD/all-atom pipelines",
         source_files=["tensornet/membrane_bio/__init__.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XVI.4", "Nucleic Acids", _pack, _pname, "B", "V0.2",
         "Folding, mechanics",
         source_files=["tensornet/biology/systems_biology.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XVI.5", "Systems Biology", _pack, _pname, "B", "V0.2",
         "FBA, ODE, stochastic",
         source_files=["tensornet/biology/systems_biology.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]}),
    Node("PHY-XVI.6", "Neuroscience Computational", _pack, _pname, "B", "V0.2",
         "HH, networks",
         source_files=["tensornet/medical/hemo.py"],
         tests={"smoke": ["test_140_domains"]}),
]

# ===== Pack XVII: Cross-Cutting Methods (6 nodes) =====
_pack, _pname = "XVII", "Cross-Cutting Computational Methods"

NODES += [
    Node("PHY-XVII.1", "Optimization", _pack, _pname, "A", "V0.2",
         "Adjoint-ready, topology optimization hooks",
         source_files=["tensornet/cfd/optimization.py"],
         tests={"smoke": ["test_140_domains"]},
         solvers=["L-BFGS", "gradient-descent", "SIMP"]),
    Node("PHY-XVII.2", "Inverse Problems", _pack, _pname, "A", "V0.2",
         "Regularization, Bayesian",
         source_files=["tensornet/cfd/adjoint.py"],
         tests={"smoke": ["test_140_domains"]},
         solvers=["Tikhonov", "adjoint-gradient"]),
    Node("PHY-XVII.3", "ML for Physics", _pack, _pname, "B", "V0.2",
         "PINNs, neural operators, equivariant nets hooks",
         source_files=["tensornet/ml_physics/__init__.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XVII.4", "Mesh Generation and Adaptivity", _pack, _pname, "B", "V0.2",
         "AMR, immersed",
         source_files=["tensornet/mesh_amr/__init__.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XVII.5", "Linear Algebra Large-Scale", _pack, _pname, "B", "V0.2",
         "Krylov, AMG, eigensolvers",
         source_files=["tensornet/algorithms/lanczos.py"],
         tests={"smoke": ["test_140_domains"], "dedicated": ["test_linalg"]}),
    Node("PHY-XVII.6", "High-Performance Computing", _pack, _pname, "B", "V0.2",
         "MPI/OpenMP/GPU, I/O",
         source_files=["tensornet/distributed_tn/distributed_dmrg.py",
                        "tensornet/distributed_tn/load_balancer.py",
                        "tensornet/distributed_tn/parallel_tebd.py"],
         tests={"smoke": ["test_140_domains"]}),
]

# ===== Pack XVIII: Coupled Physics (7 nodes) =====
_pack, _pname = "XVIII", "Continuum Coupled Physics"

NODES += [
    Node("PHY-XVIII.1", "Fluid-Structure Interaction", _pack, _pname, "B", "V0.2",
         "Partitioned/monolithic",
         source_files=["tensornet/fsi/__init__.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XVIII.2", "Thermo-Mechanical Coupling", _pack, _pname, "B", "V0.2",
         "Thermal stress, thermal expansion",
         source_files=["tensornet/coupled/thermo_mechanical.py"],
         tests={"smoke": ["test_140_domains"], "cross": ["test_cross_domain_integration"]}),
    Node("PHY-XVIII.3", "Electro-Mechanical Coupling", _pack, _pname, "B", "V0.2",
         "Piezo, MEMS",
         source_files=["tensornet/coupled/electro_mechanical.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XVIII.4", "Magnetohydrodynamics Coupled", _pack, _pname, "B", "V0.2",
         "Liquid metals, braking",
         source_files=["tensornet/coupled/coupled_mhd.py"],
         tests={"smoke": ["test_140_domains"], "cross": ["test_cross_domain_integration"]}),
    Node("PHY-XVIII.5", "Chemically Reacting Flows", _pack, _pname, "B", "V0.2",
         "Turbulence-chemistry coupling",
         source_files=["tensornet/cfd/chemistry.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XVIII.6", "Radiation-Hydrodynamics", _pack, _pname, "B", "V0.2",
         "FLD, IMC",
         source_files=["tensornet/radiation/__init__.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XVIII.7", "Multiscale Methods", _pack, _pname, "B", "V0.2",
         "FE-squared, HMM, QM/MM bridges",
         source_files=["tensornet/multiscale/multiscale.py"],
         tests={"smoke": ["test_140_domains"]}),
]

# ===== Pack XIX: Quantum Information (5 nodes) =====
_pack, _pname = "XIX", "Quantum Information and Computation"

NODES += [
    Node("PHY-XIX.1", "Quantum Circuit Simulation", _pack, _pname, "B", "V0.2",
         "TN contraction, stabilizers",
         source_files=["tensornet/quantum/hybrid.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XIX.2", "Quantum Error Correction", _pack, _pname, "B", "V0.2",
         "Codes, decoding hooks",
         source_files=["tensornet/quantum/error_mitigation.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XIX.3", "Quantum Algorithms", _pack, _pname, "B", "V0.2",
         "VQE/QAOA/HHL simulators",
         source_files=["tensornet/quantum/hybrid.py"],
         tests={"smoke": ["test_140_domains"]},
         notes="Shares source with PHY-XIX.1"),
    Node("PHY-XIX.4", "Quantum Simulation", _pack, _pname, "B", "V0.2",
         "Digital/analog interfaces",
         source_files=["tensornet/algorithms/tebd.py", "tensornet/algorithms/tdvp.py"],
         tests={"smoke": ["test_140_domains"], "dedicated": ["test_qtt_pipeline"]},
         benchmarks=["compare_tenpy"]),
    Node("PHY-XIX.5", "Quantum Cryptography and Communication", _pack, _pname, "B", "V0.2",
         "QKD, repeaters",
         source_files=["tensornet/oracle/core/oracle.py"],
         tests={"smoke": ["test_140_domains"], "dedicated": ["test_oracle"]}),
]

# ===== Pack XX: Special and Applied (10 nodes) =====
_pack, _pname = "XX", "Special and Applied Domains"

NODES += [
    Node("PHY-XX.1", "Relativistic Mechanics", _pack, _pname, "B", "V0.2",
         "SR dynamics",
         source_files=["tensornet/relativity/relativistic_mechanics.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XX.2", "General Relativity Numerical", _pack, _pname, "B", "V0.2",
         "ADM/BSSN/Z4",
         source_files=["tensornet/relativity/numerical_gr.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XX.3", "Astrodynamics", _pack, _pname, "B", "V0.2",
         "Orbital mechanics, debris",
         source_files=["tensornet/guidance/trajectory.py"],
         tests={"smoke": ["test_140_domains"]},
         notes="Shares source with PHY-I.1"),
    Node("PHY-XX.4", "Robotics Physics", _pack, _pname, "B", "V0.2",
         "Rigid/soft body, contact",
         source_files=["tensornet/robotics_physics/__init__.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XX.5", "Acoustics Applied", _pack, _pname, "B", "V0.2",
         "CAA, FW-H",
         source_files=["tensornet/acoustics/applied_acoustics.py"],
         tests={"smoke": ["test_140_domains"]},
         notes="Shares source with PHY-I.6"),
    Node("PHY-XX.6", "Biomedical Engineering", _pack, _pname, "B", "V0.2",
         "Hemodynamics, cardiac EP, imaging hooks",
         source_files=["tensornet/biomedical/biomedical.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XX.7", "Environmental Physics", _pack, _pname, "B", "V0.2",
         "Climate, wildfire, hydrology",
         source_files=["tensornet/environmental/environmental.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]}),
    Node("PHY-XX.8", "Energy Systems", _pack, _pname, "B", "V0.2",
         "Battery, fuel cells, reactors",
         source_files=["tensornet/energy/energy_systems.py"],
         tests={"smoke": ["test_140_domains"]}),
    Node("PHY-XX.9", "Manufacturing Simulation", _pack, _pname, "B", "V0.2",
         "Casting, welding, additive",
         source_files=["tensornet/manufacturing/manufacturing.py"],
         tests={"smoke": ["test_140_domains", "test_smoke_tests"]}),
    Node("PHY-XX.10", "Semiconductor Device Physics", _pack, _pname, "B", "V0.2",
         "TCAD, NEGF hooks",
         source_files=["tensornet/radiation/__init__.py"],
         tests={"smoke": ["test_140_domains"]},
         notes="Shares source with PHY-XVIII.6"),
]


def generate_index(nodes: list[Node]) -> str:
    """Generate the aggregated ledger index."""

    # Compute stats
    by_state: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    by_pack: dict[str, dict[str, int]] = {}

    for n in nodes:
        by_state[n.state] = by_state.get(n.state, 0) + 1
        by_tier[n.tier] = by_tier.get(n.tier, 0) + 1

        if n.pack not in by_pack:
            by_pack[n.pack] = {"total": 0, "V0.0": 0, "V0.1": 0, "V0.2": 0,
                               "V0.3": 0, "V0.4": 0, "V0.5": 0, "V0.6": 0, "V1.0": 0}
        by_pack[n.pack]["total"] += 1
        by_pack[n.pack][n.state] += 1

    # Build YAML
    lines = [
        "# =============================================================================",
        "# HyperTensor-VM Capability Ledger — Aggregated Index",
        "# Generated: 2026-02-08 | Schema: 1.0",
        "# =============================================================================",
        "",
        f"total_nodes: {len(nodes)}",
        f"schema_version: \"1.0\"",
        f"generated: \"2026-02-08\"",
        "",
        "# --- State Distribution ---",
        "state_distribution:",
    ]
    for state in ["V0.0", "V0.1", "V0.2", "V0.3", "V0.4", "V0.5", "V0.6", "V1.0"]:
        lines.append(f"  {state}: {by_state.get(state, 0)}")

    lines += [
        "",
        "# --- Tier Distribution ---",
        "tier_distribution:",
    ]
    for tier in ["A", "B", "C"]:
        lines.append(f"  {tier}: {by_tier.get(tier, 0)}")

    lines += [
        "",
        "# --- Per-Pack Summary ---",
        "packs:",
    ]

    # Order packs by Roman numeral
    pack_order = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                  "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]

    pack_names = {}
    for n in nodes:
        pack_names[n.pack] = n.pack_name

    for pk in pack_order:
        if pk in by_pack:
            lines.append(f"  {pk}:")
            lines.append(f"    name: \"{pack_names.get(pk, pk)}\"")
            lines.append(f"    total: {by_pack[pk]['total']}")
            for state in ["V0.0", "V0.1", "V0.2", "V0.3", "V0.4", "V0.5", "V0.6", "V1.0"]:
                if by_pack[pk].get(state, 0) > 0:
                    lines.append(f"    {state}: {by_pack[pk][state]}")

    lines += [
        "",
        "# --- Node Listing ---",
        "nodes:",
    ]

    for n in nodes:
        lines.append(f"  - id: \"{n.id}\"")
        lines.append(f"    name: \"{n.name}\"")
        lines.append(f"    pack: \"{n.pack}\"")
        lines.append(f"    tier: \"{n.tier}\"")
        lines.append(f"    state: \"{n.state}\"")
        lines.append(f"    file: \"nodes/{n.id}.yaml\"")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    NODES_DIR.mkdir(parents=True, exist_ok=True)

    # Validate we have exactly 140 nodes
    assert len(NODES) == 140, f"Expected 140 nodes, got {len(NODES)}"

    # Check for duplicate IDs
    ids = [n.id for n in NODES]
    dupes = [x for x in ids if ids.count(x) > 1]
    assert not dupes, f"Duplicate IDs: {set(dupes)}"

    # Write individual node files
    for node in NODES:
        path = NODES_DIR / f"{node.id}.yaml"
        path.write_text(node_to_yaml(node))

    # Write index
    index_path = LEDGER / "index.yaml"
    index_path.write_text(generate_index(NODES))

    print(f"Generated {len(NODES)} node files in {NODES_DIR}/")
    print(f"Generated index at {index_path}")

    # Print summary
    tiers = {}
    for n in NODES:
        tiers[n.tier] = tiers.get(n.tier, 0) + 1
    print(f"Tier distribution: {tiers}")
    print(f"All nodes at state: V0.2")


if __name__ == "__main__":
    main()
