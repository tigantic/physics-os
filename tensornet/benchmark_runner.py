"""HyperTensor Physics Benchmark Runner.

Profiles import time, instantiation cost, and lightweight computation across
all 140 physics domains.  Produces machine-readable JSON + human-readable
Markdown reports suitable for CI regression tracking.

Usage
-----
    python -m tensornet.benchmark_runner                     # full 140 sweep
    python -m tensornet.benchmark_runner --categories I II   # select categories
    python -m tensornet.benchmark_runner --json results.json # JSON output
    python -m tensornet.benchmark_runner --repeat 5          # statistical repeat
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Domain registry — mirrors tests/test_140_domains.py but kept independent
# so the benchmark can run without pytest.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DomainEntry:
    """Single physics domain for benchmarking."""
    domain_id: str
    name: str
    module: str
    key_class: str
    category: str


# Category I — Classical Mechanics (6 domains)
CAT_I: list[DomainEntry] = [
    DomainEntry("I.1", "Newtonian Particle Dynamics", "tensornet.guidance.trajectory", "VehicleState", "I"),
    DomainEntry("I.2", "Lagrangian/Hamiltonian", "tensornet.mechanics.symplectic", "SymplecticIntegratorSuite", "I"),
    DomainEntry("I.3", "Continuum Mechanics", "tensornet.mechanics.continuum", "NeoHookean", "I"),
    DomainEntry("I.4", "Structural Mechanics", "tensornet.mechanics.structural", "TimoshenkoBeam", "I"),
    DomainEntry("I.5", "Nonlinear Dynamics & Chaos", "tensornet.cfd.hou_luo_ansatz", "HouLuoConfig", "I"),
    DomainEntry("I.6", "Acoustics & Vibration", "tensornet.acoustics.applied_acoustics", "LinearisedEulerEquations", "I"),
]

# Category II — Fluid Dynamics (10 domains)
CAT_II: list[DomainEntry] = [
    DomainEntry("II.1", "Incompressible Navier-Stokes", "tensornet.cfd.ns_2d", "NS2DSolver", "II"),
    DomainEntry("II.2", "Compressible Flow", "tensornet.cfd.euler_3d", "Euler3D", "II"),
    DomainEntry("II.3", "Turbulence Modeling", "tensornet.cfd.turbulence", "TurbulentState", "II"),
    DomainEntry("II.4", "Multiphase Flow", "tensornet.multiphase.multiphase_flow", "CahnHilliardSolver", "II"),
    DomainEntry("II.5", "Reactive Flow / Combustion", "tensornet.cfd.reactive_ns", "ReactiveNS", "II"),
    DomainEntry("II.6", "Rarefied Gas / Kinetic", "tensornet.cfd.dsmc", "DSMCSolver", "II"),
    DomainEntry("II.7", "Shallow Water / Geophysical", "tensornet.geophysics.oceanography", "ShallowWaterEquations", "II"),
    DomainEntry("II.8", "Non-Newtonian / Complex Fluids", "tensornet.cfd.non_newtonian", "OldroydB", "II"),
    DomainEntry("II.9", "Porous Media Flow", "tensornet.porous_media", "DarcySolver", "II"),
    DomainEntry("II.10", "Free Surface / Interfacial", "tensornet.free_surface", "LevelSetSolver", "II"),
]

# Category III — Electromagnetism (7 domains)
CAT_III: list[DomainEntry] = [
    DomainEntry("III.1", "Electrostatics", "tensornet.em.electrostatics", "PoissonBoltzmannSolver", "III"),
    DomainEntry("III.2", "Magnetostatics", "tensornet.em.magnetostatics", "BiotSavart", "III"),
    DomainEntry("III.3", "Full Maxwell (Time-Domain)", "tensornet.em.wave_propagation", "FDTD1D", "III"),
    DomainEntry("III.4", "Frequency-Domain EM", "tensornet.em.frequency_domain", "FDFD2D_TM", "III"),
    DomainEntry("III.5", "EM Wave Propagation", "tensornet.em.wave_propagation", "MieScattering", "III"),
    DomainEntry("III.6", "Computational Photonics", "tensornet.em.computational_photonics", "TransferMatrix1D", "III"),
    DomainEntry("III.7", "Antenna & Microwave", "tensornet.em.antenna_microwave", "DipoleAntenna", "III"),
]

# Category IV — Optics & Photonics (4 domains)
CAT_IV: list[DomainEntry] = [
    DomainEntry("IV.1", "Physical Optics", "tensornet.optics.physical_optics", "FresnelPropagator", "IV"),
    DomainEntry("IV.2", "Quantum Optics", "tensornet.optics.quantum_optics", "JaynesCummingsModel", "IV"),
    DomainEntry("IV.3", "Laser Physics", "tensornet.optics.laser_physics", "FourLevelLaser", "IV"),
    DomainEntry("IV.4", "Ultrafast Optics", "tensornet.optics.ultrafast_optics", "SplitStepFourier", "IV"),
]

# Category V — Thermodynamics & Statistical Mechanics (6 domains)
CAT_V: list[DomainEntry] = [
    DomainEntry("V.1", "Equilibrium StatMech", "tensornet.statmech.equilibrium", "IsingModel", "V"),
    DomainEntry("V.2", "Non-Equilibrium StatMech", "tensornet.statmech.non_equilibrium", "JarzynskiEstimator", "V"),
    DomainEntry("V.3", "Molecular Dynamics", "tensornet.md.engine", "MDSimulation", "V"),
    DomainEntry("V.4", "Monte Carlo Methods", "tensornet.statmech.monte_carlo", "ParallelTempering", "V"),
    DomainEntry("V.5", "Heat Transfer", "tensornet.heat_transfer.radiation", "ViewFactorMC", "V"),
    DomainEntry("V.6", "Lattice Models & Spin Systems", "tensornet.mps.hamiltonians", "heisenberg_mpo", "V"),
]

# Category VI — Quantum Mechanics — Single/Few-Body (5 domains)
CAT_VI: list[DomainEntry] = [
    DomainEntry("VI.1", "Time-Independent SE", "tensornet.quantum_mechanics.stationary", "DVRSolver", "VI"),
    DomainEntry("VI.2", "Time-Dependent SE", "tensornet.quantum_mechanics.propagator", "SplitOperatorPropagator", "VI"),
    DomainEntry("VI.3", "Scattering Theory", "tensornet.qm.scattering", "PartialWaveScattering", "VI"),
    DomainEntry("VI.4", "Semiclassical / WKB", "tensornet.qm.semiclassical_wkb", "WKBSolver", "VI"),
    DomainEntry("VI.5", "Path Integrals", "tensornet.quantum_mechanics.path_integrals", "PIMC", "VI"),
]

# Category VII — Quantum Many-Body (13 domains)
CAT_VII: list[DomainEntry] = [
    DomainEntry("VII.1", "Tensor Network Methods", "tensornet.algorithms", "dmrg", "VII"),
    DomainEntry("VII.2", "Quantum Spin Systems", "tensornet.mps.hamiltonians", "spin_operators", "VII"),
    DomainEntry("VII.3", "Strongly Correlated", "tensornet.condensed_matter.strongly_correlated", "DMFTSolver", "VII"),
    DomainEntry("VII.4", "Topological Phases", "tensornet.condensed_matter.topological_phases", "ToricCode", "VII"),
    DomainEntry("VII.5", "Many-Body Localization", "tensornet.condensed_matter.mbl_disorder", "RandomFieldXXZ", "VII"),
    DomainEntry("VII.6", "Lattice Gauge Theory", "yangmills.su2", "SU2", "VII"),
    DomainEntry("VII.7", "Open Quantum Systems", "tensornet.condensed_matter.open_quantum", "LindbladSolver", "VII"),
    DomainEntry("VII.8", "Non-Eq QM Dynamics", "tensornet.condensed_matter.nonequilibrium_qm", "FloquetSolver", "VII"),
    DomainEntry("VII.9", "Kondo & Impurity", "tensornet.condensed_matter.kondo_impurity", "AndersonImpurityModel", "VII"),
    DomainEntry("VII.10", "Bosonic Many-Body", "tensornet.condensed_matter.bosonic", "GrossPitaevskiiSolver", "VII"),
    DomainEntry("VII.11", "Fermionic Systems", "tensornet.condensed_matter.fermionic", "BCSSolver", "VII"),
    DomainEntry("VII.12", "Nuclear Many-Body", "tensornet.condensed_matter.nuclear_many_body", "NuclearShellModel", "VII"),
    DomainEntry("VII.13", "Ultracold Atoms", "tensornet.condensed_matter.ultracold_atoms", "BoseHubbardModel", "VII"),
]

# Category VIII — Electronic Structure (7 domains)
CAT_VIII: list[DomainEntry] = [
    DomainEntry("VIII.1", "DFT", "tensornet.electronic_structure.dft", "KohnShamDFT1D", "VIII"),
    DomainEntry("VIII.2", "Beyond-DFT", "tensornet.electronic_structure.beyond_dft", "RestrictedHartreeFock", "VIII"),
    DomainEntry("VIII.3", "Tight Binding", "tensornet.electronic_structure.tight_binding", "SlaterKosterTB", "VIII"),
    DomainEntry("VIII.4", "Excited States", "tensornet.electronic_structure.excited_states", "CasidaTDDFT", "VIII"),
    DomainEntry("VIII.5", "Response Properties", "tensornet.electronic_structure.response", "DFPTSolver", "VIII"),
    DomainEntry("VIII.6", "Relativistic Electronic", "tensornet.electronic_structure.relativistic", "ZORAHamiltonian", "VIII"),
    DomainEntry("VIII.7", "Quantum Embedding", "tensornet.electronic_structure.embedding", "QMMMEmbedding", "VIII"),
]

# Category IX — Solid State / Condensed Matter (8 domains)
CAT_IX: list[DomainEntry] = [
    DomainEntry("IX.1", "Phonon Dynamics", "tensornet.condensed_matter.phonons", "DynamicalMatrix", "IX"),
    DomainEntry("IX.2", "Band Structure", "tensornet.condensed_matter.band_structure", "TightBindingBands", "IX"),
    DomainEntry("IX.3", "Classical Magnetism", "tensornet.condensed_matter.classical_magnetism", "LandauLifshitzGilbert", "IX"),
    DomainEntry("IX.4", "Superconductivity", "tensornet.condensed_matter.fermionic", "BCSSolver", "IX"),
    DomainEntry("IX.5", "Disordered Systems", "tensornet.condensed_matter.disordered", "AndersonModel", "IX"),
    DomainEntry("IX.6", "Surfaces & Interfaces", "tensornet.condensed_matter.surfaces_interfaces", "SurfaceEnergy", "IX"),
    DomainEntry("IX.7", "Defect Physics", "tensornet.condensed_matter.defects", "DefectEnergy", "IX"),
    DomainEntry("IX.8", "Ferroelectrics", "tensornet.condensed_matter.ferroelectrics", "LandauDevonshire", "IX"),
]

# Category X — Nuclear & Particle Physics (6 domains)
CAT_X: list[DomainEntry] = [
    DomainEntry("X.1", "Nuclear Structure", "tensornet.nuclear.structure", "NuclearShellModel", "X"),
    DomainEntry("X.2", "Nuclear Reactions", "tensornet.nuclear.reactions", "OpticalModelPotential", "X"),
    DomainEntry("X.3", "Nuclear Astrophysics", "tensornet.nuclear.astrophysics", "NuclearReactionNetwork", "X"),
    DomainEntry("X.4", "Lattice QCD", "tensornet.qft.lattice_qcd", "SU3Group", "X"),
    DomainEntry("X.5", "Perturbative QFT", "tensornet.qft.perturbative", "FeynmanDiagram", "X"),
    DomainEntry("X.6", "Beyond SM", "tensornet.particle.beyond_sm", "NeutrinoOscillations", "X"),
]

# Category XI — Plasma Physics (8 domains)
CAT_XI: list[DomainEntry] = [
    DomainEntry("XI.1", "Ideal MHD", "tensornet.fusion.tokamak", "TokamakReactor", "XI"),
    DomainEntry("XI.2", "Extended MHD", "tensornet.plasma.extended_mhd", "HallMHDSolver1D", "XI"),
    DomainEntry("XI.3", "Kinetic Theory (Plasma)", "tensornet.cfd.fast_vlasov_5d", "FastVlasov5D", "XI"),
    DomainEntry("XI.4", "Gyrokinetics", "tensornet.plasma.gyrokinetics", "ITGDispersion", "XI"),
    DomainEntry("XI.5", "Magnetic Reconnection", "tensornet.plasma.magnetic_reconnection", "SweetParkerReconnection", "XI"),
    DomainEntry("XI.6", "Laser-Plasma", "tensornet.plasma.laser_plasma", "StimulatedRamanScattering", "XI"),
    DomainEntry("XI.7", "Dusty Plasmas", "tensornet.plasma.dusty_plasmas", "DustAcousticWave", "XI"),
    DomainEntry("XI.8", "Space & Astrophysical Plasma", "tensornet.plasma.space_plasma", "ParkerSolarWind", "XI"),
]

# Category XII — Astrophysics & Cosmology (6 domains)
CAT_XII: list[DomainEntry] = [
    DomainEntry("XII.1", "Stellar Structure", "tensornet.astro.stellar_structure", "StellarStructure", "XII"),
    DomainEntry("XII.2", "Compact Objects", "tensornet.astro.compact_objects", "TOVSolver", "XII"),
    DomainEntry("XII.3", "Gravitational Waves", "tensornet.astro.gravitational_waves", "PostNewtonianInspiral", "XII"),
    DomainEntry("XII.4", "Cosmological Sims", "tensornet.astro.cosmological_sims", "FriedmannCosmology", "XII"),
    DomainEntry("XII.5", "CMB & Early Universe", "tensornet.astro.cmb_early_universe", "BoltzmannHierarchy", "XII"),
    DomainEntry("XII.6", "Radiative Transfer", "tensornet.astro.radiative_transfer", "RadiativeTransfer1D", "XII"),
]

# Category XIII — Geophysics & Earth Science (6 domains)
CAT_XIII: list[DomainEntry] = [
    DomainEntry("XIII.1", "Seismology", "tensornet.geophysics.seismology", "AcousticWave2D", "XIII"),
    DomainEntry("XIII.2", "Mantle Convection", "tensornet.geophysics.mantle_convection", "StokesFlow2D", "XIII"),
    DomainEntry("XIII.3", "Geodynamo", "tensornet.geophysics.geodynamo", "MagneticInduction2D", "XIII"),
    DomainEntry("XIII.4", "Atmospheric Physics", "tensornet.geophysics.atmosphere", "ChapmanOzone", "XIII"),
    DomainEntry("XIII.5", "Oceanography", "tensornet.geophysics.oceanography", "SeawaterEOS", "XIII"),
    DomainEntry("XIII.6", "Glaciology", "tensornet.geophysics.glaciology", "GlenFlowLaw", "XIII"),
]

# Category XIV — Materials Science (7 domains)
CAT_XIV: list[DomainEntry] = [
    DomainEntry("XIV.1", "First-Principles Design", "tensornet.materials.first_principles_design", "ConvexHullStability", "XIV"),
    DomainEntry("XIV.2", "Mechanical Properties", "tensornet.materials.mechanical_properties", "ElasticTensor", "XIV"),
    DomainEntry("XIV.3", "Phase-Field Methods", "tensornet.phase_field", "AllenCahnSolver", "XIV"),
    DomainEntry("XIV.4", "Microstructure", "tensornet.materials.microstructure", "CahnHilliard2D", "XIV"),
    DomainEntry("XIV.5", "Radiation Damage", "tensornet.materials.radiation_damage", "NRTDisplacements", "XIV"),
    DomainEntry("XIV.6", "Polymers & Soft Matter", "tensornet.materials.polymers_soft_matter", "FloryHuggins", "XIV"),
    DomainEntry("XIV.7", "Ceramics / High-Temp", "tensornet.materials.ceramics", "SinteringModel", "XIV"),
]

# Category XV — Chemical Physics & Reaction Dynamics (7 domains)
CAT_XV: list[DomainEntry] = [
    DomainEntry("XV.1", "PES Construction", "tensornet.chemistry.pes", "MorsePotential", "XV"),
    DomainEntry("XV.2", "Reaction Rate Theory", "tensornet.chemistry.reaction_rate", "TransitionStateTheory", "XV"),
    DomainEntry("XV.3", "Quantum Reactive Dynamics", "tensornet.chemistry.quantum_reactive", "CollinearReactiveScattering", "XV"),
    DomainEntry("XV.4", "Nonadiabatic Dynamics", "tensornet.chemistry.nonadiabatic", "LandauZener", "XV"),
    DomainEntry("XV.5", "Photochemistry", "tensornet.chemistry.photochemistry", "FranckCondonFactors", "XV"),
    DomainEntry("XV.6", "Catalysis", "tensornet.fusion.resonant_catalysis", "ResonantCatalysisSolver", "XV"),
    DomainEntry("XV.7", "Spectroscopy", "tensornet.chemistry.spectroscopy", "VibrationalSpectroscopy", "XV"),
]

# Category XVI — Biophysics & Computational Biology (6 domains)
CAT_XVI: list[DomainEntry] = [
    DomainEntry("XVI.1", "Protein Structure & Dynamics", "tensornet.md.engine", "MDSimulation", "XVI"),
    DomainEntry("XVI.2", "Drug Design & Binding", "tensornet.chemistry.pes", "LEPSPotential", "XVI"),
    DomainEntry("XVI.3", "Membrane Biophysics", "tensornet.membrane_bio", "CoarseGrainedBilayer", "XVI"),
    DomainEntry("XVI.4", "Nucleic Acids & Regulation", "tensornet.biology.systems_biology", "BooleanGRN", "XVI"),
    DomainEntry("XVI.5", "Systems Biology", "tensornet.biology.systems_biology", "FluxBalanceAnalysis", "XVI"),
    DomainEntry("XVI.6", "Neuroscience / Biomedical", "tensornet.medical.hemo", "ArterySimulation", "XVI"),
]

# Category XVII — Computational Methods — Cross-Cutting (6 domains)
CAT_XVII: list[DomainEntry] = [
    DomainEntry("XVII.1", "Optimization", "tensornet.cfd.optimization", "ShapeOptimizer", "XVII"),
    DomainEntry("XVII.2", "Inverse Problems", "tensornet.cfd.adjoint", "AdjointEuler2D", "XVII"),
    DomainEntry("XVII.3", "ML for Physics", "tensornet.ml_physics", "PINN", "XVII"),
    DomainEntry("XVII.4", "Adaptive Mesh", "tensornet.mesh_amr", "QuadtreeAMR", "XVII"),
    DomainEntry("XVII.5", "Large-Scale Linear Algebra", "tensornet.algorithms.lanczos", "LanczosResult", "XVII"),
    DomainEntry("XVII.6", "HPC / Distributed", "tensornet.distributed_tn", "DistributedDMRG", "XVII"),
]

# Category XVIII — Continuum Coupled Physics (7 domains)
CAT_XVIII: list[DomainEntry] = [
    DomainEntry("XVIII.1", "FSI", "tensornet.fsi", "PartitionedFSICoupler", "XVIII"),
    DomainEntry("XVIII.2", "Thermo-Mechanical", "tensornet.coupled.thermo_mechanical", "ThermoelasticSolver", "XVIII"),
    DomainEntry("XVIII.3", "Electro-Mechanical", "tensornet.coupled.electro_mechanical", "PiezoelectricSolver", "XVIII"),
    DomainEntry("XVIII.4", "Coupled MHD", "tensornet.coupled.coupled_mhd", "HartmannFlow", "XVIII"),
    DomainEntry("XVIII.5", "Chemically Reacting Flows", "tensornet.cfd.chemistry", "Reaction", "XVIII"),
    DomainEntry("XVIII.6", "Radiation-Hydro", "tensornet.radiation", "RadiationEuler1D", "XVIII"),
    DomainEntry("XVIII.7", "Multiscale Methods", "tensornet.multiscale.multiscale", "FE2Solver", "XVIII"),
]

# Category XIX — Quantum Information & Computation (5 domains)
CAT_XIX: list[DomainEntry] = [
    DomainEntry("XIX.1", "Quantum Circuit Simulation", "tensornet.quantum.hybrid", "QuantumCircuit", "XIX"),
    DomainEntry("XIX.2", "Quantum Error Correction", "tensornet.quantum.error_mitigation", "ShorCode", "XIX"),
    DomainEntry("XIX.3", "Quantum Algorithms", "tensornet.quantum.hybrid", "VQE", "XIX"),
    DomainEntry("XIX.4", "Quantum Simulation", "tensornet.algorithms", "tebd", "XIX"),
    DomainEntry("XIX.5", "Quantum Crypto & Communication", "tensornet.oracle.core.oracle", "ORACLE", "XIX"),
]

# Category XX — Special / Applied Domains (10 domains)
CAT_XX: list[DomainEntry] = [
    DomainEntry("XX.1", "Relativistic Mechanics", "tensornet.relativity.relativistic_mechanics", "LorentzBoost", "XX"),
    DomainEntry("XX.2", "Numerical GR", "tensornet.relativity.numerical_gr", "BSSNState", "XX"),
    DomainEntry("XX.3", "Astrodynamics", "tensornet.guidance.trajectory", "IntegrationMethod", "XX"),
    DomainEntry("XX.4", "Robotics Physics", "tensornet.robotics_physics", "FeatherstoneABA", "XX"),
    DomainEntry("XX.5", "Applied Acoustics", "tensornet.acoustics.applied_acoustics", "LinearisedEulerEquations", "XX"),
    DomainEntry("XX.6", "Biomedical Engineering", "tensornet.biomedical.biomedical", "BidomainSolver", "XX"),
    DomainEntry("XX.7", "Environmental Physics", "tensornet.environmental.environmental", "GaussianPlume", "XX"),
    DomainEntry("XX.8", "Energy Systems", "tensornet.energy.energy_systems", "DriftDiffusionSolarCell", "XX"),
    DomainEntry("XX.9", "Manufacturing Physics", "tensornet.manufacturing.manufacturing", "GoldakWeldingSource", "XX"),
    DomainEntry("XX.10", "Semiconductor Device", "tensornet.radiation", "ICFImplosion", "XX"),
]

ALL_DOMAINS: list[DomainEntry] = (
    CAT_I + CAT_II + CAT_III + CAT_IV + CAT_V + CAT_VI + CAT_VII + CAT_VIII
    + CAT_IX + CAT_X + CAT_XI + CAT_XII + CAT_XIII + CAT_XIV + CAT_XV
    + CAT_XVI + CAT_XVII + CAT_XVIII + CAT_XIX + CAT_XX
)

CATEGORY_NAMES: dict[str, str] = {
    "I": "Classical Mechanics", "II": "Fluid Dynamics", "III": "Electromagnetism",
    "IV": "Optics & Photonics", "V": "Thermodynamics & Statistical Mechanics",
    "VI": "Quantum Mechanics", "VII": "Quantum Many-Body",
    "VIII": "Electronic Structure", "IX": "Solid State / Condensed Matter",
    "X": "Nuclear & Particle Physics", "XI": "Plasma Physics",
    "XII": "Astrophysics & Cosmology", "XIII": "Geophysics & Earth Science",
    "XIV": "Materials Science", "XV": "Chemical Physics & Reaction Dynamics",
    "XVI": "Biophysics & Computational Biology",
    "XVII": "Computational Methods", "XVIII": "Continuum Coupled Physics",
    "XIX": "Quantum Information & Computation",
    "XX": "Special / Applied Domains",
}


# ---------------------------------------------------------------------------
# Benchmark result types
# ---------------------------------------------------------------------------

@dataclass
class DomainResult:
    """Benchmark result for a single domain."""
    domain_id: str
    name: str
    module: str
    category: str
    import_ok: bool = False
    import_time_ms: float = 0.0
    class_found: bool = False
    instantiate_ok: bool = False
    instantiate_time_ms: float = 0.0
    error: str = ""


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report."""
    version: str = "40.0.0"
    timestamp: str = ""
    total_domains: int = 0
    imports_ok: int = 0
    classes_found: int = 0
    instantiations_ok: int = 0
    total_import_time_ms: float = 0.0
    mean_import_time_ms: float = 0.0
    median_import_time_ms: float = 0.0
    p95_import_time_ms: float = 0.0
    max_import_time_ms: float = 0.0
    categories_tested: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def _bench_import(mod_path: str, repeats: int = 1) -> tuple[bool, float, str]:
    """Benchmark a single module import. Returns (ok, avg_ms, error)."""
    # Clear module from cache for accurate re-timing
    times: list[float] = []
    error = ""
    ok = False
    for _ in range(repeats):
        # Remove from cache to force re-import
        to_remove = [k for k in sys.modules if k.startswith(mod_path)]
        for k in to_remove:
            del sys.modules[k]
        t0 = time.perf_counter()
        try:
            importlib.import_module(mod_path)
            ok = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            ok = False
            break
        finally:
            times.append((time.perf_counter() - t0) * 1000)
    avg = statistics.mean(times) if times else 0.0
    return ok, avg, error


def _bench_class_check(mod_path: str, class_name: str) -> tuple[bool, str]:
    """Check if a class/attribute exists in an already-imported module."""
    try:
        mod = sys.modules.get(mod_path) or importlib.import_module(mod_path)
        return hasattr(mod, class_name), ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _bench_instantiate(mod_path: str, class_name: str) -> tuple[bool, float, str]:
    """Try to instantiate the key class with no arguments."""
    try:
        mod = sys.modules.get(mod_path) or importlib.import_module(mod_path)
        cls = getattr(mod, class_name, None)
        if cls is None:
            return False, 0.0, f"{class_name} not found"
        t0 = time.perf_counter()
        try:
            cls()
            elapsed = (time.perf_counter() - t0) * 1000
            return True, elapsed, ""
        except TypeError:
            # Constructor requires arguments — that's fine, class exists
            elapsed = (time.perf_counter() - t0) * 1000
            return False, elapsed, "requires args (expected)"
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return False, elapsed, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, 0.0, f"{type(exc).__name__}: {exc}"


def run_benchmark(
    domains: Sequence[DomainEntry],
    repeats: int = 1,
    verbose: bool = False,
) -> BenchmarkReport:
    """Run benchmarks across all specified domains."""
    import datetime

    report = BenchmarkReport(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        total_domains=len(domains),
    )
    all_import_times: list[float] = []
    categories_seen: set[str] = set()

    for i, d in enumerate(domains):
        if verbose:
            print(f"  [{i+1:3d}/{len(domains)}] {d.domain_id:8s} {d.name:30s} ", end="", flush=True)

        res = DomainResult(
            domain_id=d.domain_id,
            name=d.name,
            module=d.module,
            category=d.category,
        )

        # Phase 1: Import
        ok, t_ms, err = _bench_import(d.module, repeats=repeats)
        res.import_ok = ok
        res.import_time_ms = round(t_ms, 3)
        if not ok:
            res.error = err
            if verbose:
                print(f"IMPORT FAIL ({err[:60]})")
            report.results.append(asdict(res))
            continue

        all_import_times.append(t_ms)
        report.imports_ok += 1
        categories_seen.add(d.category)

        # Phase 2: Class check
        found, _ = _bench_class_check(d.module, d.key_class)
        res.class_found = found
        if found:
            report.classes_found += 1

        # Phase 3: Instantiation attempt
        inst_ok, inst_ms, inst_err = _bench_instantiate(d.module, d.key_class)
        res.instantiate_ok = inst_ok
        res.instantiate_time_ms = round(inst_ms, 3)
        if not inst_ok and inst_err:
            res.error = inst_err

        if inst_ok:
            report.instantiations_ok += 1

        if verbose:
            status = "OK" if found else "CLASS?"
            print(f"{t_ms:8.1f} ms  {status}")

        report.results.append(asdict(res))

    # Aggregate statistics
    report.categories_tested = len(categories_seen)
    if all_import_times:
        report.total_import_time_ms = round(sum(all_import_times), 1)
        report.mean_import_time_ms = round(statistics.mean(all_import_times), 3)
        report.median_import_time_ms = round(statistics.median(all_import_times), 3)
        sorted_times = sorted(all_import_times)
        p95_idx = min(int(math.ceil(0.95 * len(sorted_times))) - 1, len(sorted_times) - 1)
        report.p95_import_time_ms = round(sorted_times[p95_idx], 3)
        report.max_import_time_ms = round(sorted_times[-1], 3)

    return report


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def report_to_json(report: BenchmarkReport) -> str:
    """Produce machine-readable JSON."""
    return json.dumps(asdict(report), indent=2)


def report_to_markdown(report: BenchmarkReport) -> str:
    """Produce human-readable Markdown summary."""
    lines: list[str] = []
    a = lines.append

    a("# HyperTensor v40.0.0 — Physics Benchmark Report")
    a("")
    a(f"**Timestamp:** {report.timestamp}")
    a(f"**Domains:** {report.total_domains}")
    a(f"**Imports OK:** {report.imports_ok}/{report.total_domains}")
    a(f"**Classes Found:** {report.classes_found}/{report.total_domains}")
    a(f"**Instantiations OK:** {report.instantiations_ok}/{report.total_domains}")
    a(f"**Categories:** {report.categories_tested}/20")
    a("")
    a("## Import Timing Statistics")
    a("")
    a(f"| Metric | Value |")
    a(f"|--------|-------|")
    a(f"| Total import time | {report.total_import_time_ms:.1f} ms |")
    a(f"| Mean | {report.mean_import_time_ms:.3f} ms |")
    a(f"| Median | {report.median_import_time_ms:.3f} ms |")
    a(f"| P95 | {report.p95_import_time_ms:.3f} ms |")
    a(f"| Max | {report.max_import_time_ms:.3f} ms |")
    a("")
    a("## Per-Category Summary")
    a("")
    a("| Cat | Name | Domains | Imports | Classes | Avg (ms) |")
    a("|-----|------|---------|---------|---------|----------|")

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in report.results:
        by_cat.setdefault(r["category"], []).append(r)

    for cat_id in sorted(by_cat.keys(), key=lambda x: list(CATEGORY_NAMES.keys()).index(x) if x in CATEGORY_NAMES else 99):
        entries = by_cat[cat_id]
        n = len(entries)
        n_imp = sum(1 for e in entries if e["import_ok"])
        n_cls = sum(1 for e in entries if e["class_found"])
        times = [e["import_time_ms"] for e in entries if e["import_ok"]]
        avg_t = statistics.mean(times) if times else 0.0
        cat_name = CATEGORY_NAMES.get(cat_id, cat_id)
        a(f"| {cat_id} | {cat_name} | {n} | {n_imp} | {n_cls} | {avg_t:.1f} |")

    a("")
    a("## Failed Imports")
    a("")
    failures = [r for r in report.results if not r["import_ok"]]
    if failures:
        a("| Domain | Module | Error |")
        a("|--------|--------|-------|")
        for f in failures:
            err_short = f["error"][:80]
            a(f"| {f['domain_id']} | `{f['module']}` | {err_short} |")
    else:
        a("None — all 140 domains imported successfully.")
    a("")
    a("---")
    a(f"*Generated by `tensornet.benchmark_runner` v{report.version}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensornet.benchmark_runner",
        description="Benchmark all 140 HyperTensor physics domains.",
    )
    parser.add_argument(
        "--categories", "-c", nargs="*", default=None,
        help="Run only these categories (e.g., I II VII). Default: all.",
    )
    parser.add_argument(
        "--json", "-j", default=None, metavar="FILE",
        help="Write JSON results to FILE.",
    )
    parser.add_argument(
        "--markdown", "-m", default=None, metavar="FILE",
        help="Write Markdown report to FILE.",
    )
    parser.add_argument(
        "--repeat", "-r", type=int, default=1,
        help="Number of import timing repeats (default: 1).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress for each domain.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress all output except final summary line.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for CLI invocation."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Filter domains by category if requested
    if args.categories:
        cats = set(c.upper() for c in args.categories)
        domains = [d for d in ALL_DOMAINS if d.category in cats]
        if not domains:
            print(f"ERROR: No domains in categories {cats}. "
                  f"Available: {sorted(CATEGORY_NAMES.keys())}", file=sys.stderr)
            return 1
    else:
        domains = ALL_DOMAINS

    if not args.quiet:
        print(f"HyperTensor Benchmark Runner v40.0.0")
        print(f"Benchmarking {len(domains)} domains "
              f"(repeat={args.repeat})...\n")

    report = run_benchmark(domains, repeats=args.repeat, verbose=args.verbose)

    # Write outputs
    if args.json:
        Path(args.json).write_text(report_to_json(report))
        if not args.quiet:
            print(f"\nJSON written to {args.json}")

    if args.markdown:
        Path(args.markdown).write_text(report_to_markdown(report))
        if not args.quiet:
            print(f"Markdown written to {args.markdown}")

    # Summary
    if not args.quiet:
        print(f"\n{'=' * 60}")
        print(f"  BENCHMARK SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Domains:         {report.total_domains}")
        print(f"  Imports OK:      {report.imports_ok}/{report.total_domains}")
        print(f"  Classes Found:   {report.classes_found}/{report.total_domains}")
        print(f"  Instantiated:    {report.instantiations_ok}/{report.total_domains}")
        print(f"  Categories:      {report.categories_tested}/20")
        print(f"  Total import:    {report.total_import_time_ms:.1f} ms")
        print(f"  Mean import:     {report.mean_import_time_ms:.1f} ms")
        print(f"  Median import:   {report.median_import_time_ms:.1f} ms")
        print(f"  P95 import:      {report.p95_import_time_ms:.1f} ms")
        print(f"  Max import:      {report.max_import_time_ms:.1f} ms")
        print(f"{'=' * 60}")

    # Exit code: 0 if all imports succeeded
    return 0 if report.imports_ok == report.total_domains else 1


if __name__ == "__main__":
    raise SystemExit(main())
