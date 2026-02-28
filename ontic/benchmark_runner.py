"""HyperTensor Physics Benchmark Runner.

Profiles import time, instantiation cost, and lightweight computation across
all 140 physics domains.  Produces machine-readable JSON + human-readable
Markdown reports suitable for CI regression tracking.

Usage
-----
    python -m ontic.benchmark_runner                     # full 140 sweep
    python -m ontic.benchmark_runner --categories I II   # select categories
    python -m ontic.benchmark_runner --json results.json # JSON output
    python -m ontic.benchmark_runner --repeat 5          # statistical repeat
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
    DomainEntry("I.1", "Newtonian Particle Dynamics", "ontic.guidance.trajectory", "VehicleState", "I"),
    DomainEntry("I.2", "Lagrangian/Hamiltonian", "ontic.mechanics.symplectic", "SymplecticIntegratorSuite", "I"),
    DomainEntry("I.3", "Continuum Mechanics", "ontic.mechanics.continuum", "NeoHookean", "I"),
    DomainEntry("I.4", "Structural Mechanics", "ontic.mechanics.structural", "TimoshenkoBeam", "I"),
    DomainEntry("I.5", "Nonlinear Dynamics & Chaos", "ontic.cfd.hou_luo_ansatz", "HouLuoConfig", "I"),
    DomainEntry("I.6", "Acoustics & Vibration", "ontic.acoustics.applied_acoustics", "LinearisedEulerEquations", "I"),
]

# Category II — Fluid Dynamics (10 domains)
CAT_II: list[DomainEntry] = [
    DomainEntry("II.1", "Incompressible Navier-Stokes", "ontic.cfd.ns_2d", "NS2DSolver", "II"),
    DomainEntry("II.2", "Compressible Flow", "ontic.cfd.euler_3d", "Euler3D", "II"),
    DomainEntry("II.3", "Turbulence Modeling", "ontic.cfd.turbulence", "TurbulentState", "II"),
    DomainEntry("II.4", "Multiphase Flow", "ontic.multiphase.multiphase_flow", "CahnHilliardSolver", "II"),
    DomainEntry("II.5", "Reactive Flow / Combustion", "ontic.cfd.reactive_ns", "ReactiveNS", "II"),
    DomainEntry("II.6", "Rarefied Gas / Kinetic", "ontic.cfd.dsmc", "DSMCSolver", "II"),
    DomainEntry("II.7", "Shallow Water / Geophysical", "ontic.geophysics.oceanography", "ShallowWaterEquations", "II"),
    DomainEntry("II.8", "Non-Newtonian / Complex Fluids", "ontic.cfd.non_newtonian", "OldroydB", "II"),
    DomainEntry("II.9", "Porous Media Flow", "ontic.porous_media", "DarcySolver", "II"),
    DomainEntry("II.10", "Free Surface / Interfacial", "ontic.free_surface", "LevelSetSolver", "II"),
]

# Category III — Electromagnetism (7 domains)
CAT_III: list[DomainEntry] = [
    DomainEntry("III.1", "Electrostatics", "ontic.em.electrostatics", "PoissonBoltzmannSolver", "III"),
    DomainEntry("III.2", "Magnetostatics", "ontic.em.magnetostatics", "BiotSavart", "III"),
    DomainEntry("III.3", "Full Maxwell (Time-Domain)", "ontic.em.wave_propagation", "FDTD1D", "III"),
    DomainEntry("III.4", "Frequency-Domain EM", "ontic.em.frequency_domain", "FDFD2D_TM", "III"),
    DomainEntry("III.5", "EM Wave Propagation", "ontic.em.wave_propagation", "MieScattering", "III"),
    DomainEntry("III.6", "Computational Photonics", "ontic.em.computational_photonics", "TransferMatrix1D", "III"),
    DomainEntry("III.7", "Antenna & Microwave", "ontic.em.antenna_microwave", "DipoleAntenna", "III"),
]

# Category IV — Optics & Photonics (4 domains)
CAT_IV: list[DomainEntry] = [
    DomainEntry("IV.1", "Physical Optics", "ontic.optics.physical_optics", "FresnelPropagator", "IV"),
    DomainEntry("IV.2", "Quantum Optics", "ontic.optics.quantum_optics", "JaynesCummingsModel", "IV"),
    DomainEntry("IV.3", "Laser Physics", "ontic.optics.laser_physics", "FourLevelLaser", "IV"),
    DomainEntry("IV.4", "Ultrafast Optics", "ontic.optics.ultrafast_optics", "SplitStepFourier", "IV"),
]

# Category V — Thermodynamics & Statistical Mechanics (6 domains)
CAT_V: list[DomainEntry] = [
    DomainEntry("V.1", "Equilibrium StatMech", "ontic.statmech.equilibrium", "IsingModel", "V"),
    DomainEntry("V.2", "Non-Equilibrium StatMech", "ontic.statmech.non_equilibrium", "JarzynskiEstimator", "V"),
    DomainEntry("V.3", "Molecular Dynamics", "ontic.md.engine", "MDSimulation", "V"),
    DomainEntry("V.4", "Monte Carlo Methods", "ontic.statmech.monte_carlo", "ParallelTempering", "V"),
    DomainEntry("V.5", "Heat Transfer", "ontic.heat_transfer.radiation", "ViewFactorMC", "V"),
    DomainEntry("V.6", "Lattice Models & Spin Systems", "ontic.mps.hamiltonians", "heisenberg_mpo", "V"),
]

# Category VI — Quantum Mechanics — Single/Few-Body (5 domains)
CAT_VI: list[DomainEntry] = [
    DomainEntry("VI.1", "Time-Independent SE", "ontic.quantum_mechanics.stationary", "DVRSolver", "VI"),
    DomainEntry("VI.2", "Time-Dependent SE", "ontic.quantum_mechanics.propagator", "SplitOperatorPropagator", "VI"),
    DomainEntry("VI.3", "Scattering Theory", "ontic.qm.scattering", "PartialWaveScattering", "VI"),
    DomainEntry("VI.4", "Semiclassical / WKB", "ontic.qm.semiclassical_wkb", "WKBSolver", "VI"),
    DomainEntry("VI.5", "Path Integrals", "ontic.quantum_mechanics.path_integrals", "PIMC", "VI"),
]

# Category VII — Quantum Many-Body (13 domains)
CAT_VII: list[DomainEntry] = [
    DomainEntry("VII.1", "Tensor Network Methods", "ontic.algorithms", "dmrg", "VII"),
    DomainEntry("VII.2", "Quantum Spin Systems", "ontic.mps.hamiltonians", "spin_operators", "VII"),
    DomainEntry("VII.3", "Strongly Correlated", "ontic.condensed_matter.strongly_correlated", "DMFTSolver", "VII"),
    DomainEntry("VII.4", "Topological Phases", "ontic.condensed_matter.topological_phases", "ToricCode", "VII"),
    DomainEntry("VII.5", "Many-Body Localization", "ontic.condensed_matter.mbl_disorder", "RandomFieldXXZ", "VII"),
    DomainEntry("VII.6", "Lattice Gauge Theory", "yangmills.su2", "SU2", "VII"),
    DomainEntry("VII.7", "Open Quantum Systems", "ontic.condensed_matter.open_quantum", "LindbladSolver", "VII"),
    DomainEntry("VII.8", "Non-Eq QM Dynamics", "ontic.condensed_matter.nonequilibrium_qm", "FloquetSolver", "VII"),
    DomainEntry("VII.9", "Kondo & Impurity", "ontic.condensed_matter.kondo_impurity", "AndersonImpurityModel", "VII"),
    DomainEntry("VII.10", "Bosonic Many-Body", "ontic.condensed_matter.bosonic", "GrossPitaevskiiSolver", "VII"),
    DomainEntry("VII.11", "Fermionic Systems", "ontic.condensed_matter.fermionic", "BCSSolver", "VII"),
    DomainEntry("VII.12", "Nuclear Many-Body", "ontic.condensed_matter.nuclear_many_body", "NuclearShellModel", "VII"),
    DomainEntry("VII.13", "Ultracold Atoms", "ontic.condensed_matter.ultracold_atoms", "BoseHubbardModel", "VII"),
]

# Category VIII — Electronic Structure (7 domains)
CAT_VIII: list[DomainEntry] = [
    DomainEntry("VIII.1", "DFT", "ontic.electronic_structure.dft", "KohnShamDFT1D", "VIII"),
    DomainEntry("VIII.2", "Beyond-DFT", "ontic.electronic_structure.beyond_dft", "RestrictedHartreeFock", "VIII"),
    DomainEntry("VIII.3", "Tight Binding", "ontic.electronic_structure.tight_binding", "SlaterKosterTB", "VIII"),
    DomainEntry("VIII.4", "Excited States", "ontic.electronic_structure.excited_states", "CasidaTDDFT", "VIII"),
    DomainEntry("VIII.5", "Response Properties", "ontic.electronic_structure.response", "DFPTSolver", "VIII"),
    DomainEntry("VIII.6", "Relativistic Electronic", "ontic.electronic_structure.relativistic", "ZORAHamiltonian", "VIII"),
    DomainEntry("VIII.7", "Quantum Embedding", "ontic.electronic_structure.embedding", "QMMMEmbedding", "VIII"),
]

# Category IX — Solid State / Condensed Matter (8 domains)
CAT_IX: list[DomainEntry] = [
    DomainEntry("IX.1", "Phonon Dynamics", "ontic.condensed_matter.phonons", "DynamicalMatrix", "IX"),
    DomainEntry("IX.2", "Band Structure", "ontic.condensed_matter.band_structure", "TightBindingBands", "IX"),
    DomainEntry("IX.3", "Classical Magnetism", "ontic.condensed_matter.classical_magnetism", "LandauLifshitzGilbert", "IX"),
    DomainEntry("IX.4", "Superconductivity", "ontic.condensed_matter.fermionic", "BCSSolver", "IX"),
    DomainEntry("IX.5", "Disordered Systems", "ontic.condensed_matter.disordered", "AndersonModel", "IX"),
    DomainEntry("IX.6", "Surfaces & Interfaces", "ontic.condensed_matter.surfaces_interfaces", "SurfaceEnergy", "IX"),
    DomainEntry("IX.7", "Defect Physics", "ontic.condensed_matter.defects", "DefectEnergy", "IX"),
    DomainEntry("IX.8", "Ferroelectrics", "ontic.condensed_matter.ferroelectrics", "LandauDevonshire", "IX"),
]

# Category X — Nuclear & Particle Physics (6 domains)
CAT_X: list[DomainEntry] = [
    DomainEntry("X.1", "Nuclear Structure", "ontic.nuclear.structure", "NuclearShellModel", "X"),
    DomainEntry("X.2", "Nuclear Reactions", "ontic.nuclear.reactions", "OpticalModelPotential", "X"),
    DomainEntry("X.3", "Nuclear Astrophysics", "ontic.nuclear.astrophysics", "NuclearReactionNetwork", "X"),
    DomainEntry("X.4", "Lattice QCD", "ontic.qft.lattice_qcd", "SU3Group", "X"),
    DomainEntry("X.5", "Perturbative QFT", "ontic.qft.perturbative", "FeynmanDiagram", "X"),
    DomainEntry("X.6", "Beyond SM", "ontic.particle.beyond_sm", "NeutrinoOscillations", "X"),
]

# Category XI — Plasma Physics (8 domains)
CAT_XI: list[DomainEntry] = [
    DomainEntry("XI.1", "Ideal MHD", "ontic.fusion.tokamak", "TokamakReactor", "XI"),
    DomainEntry("XI.2", "Extended MHD", "ontic.plasma.extended_mhd", "HallMHDSolver1D", "XI"),
    DomainEntry("XI.3", "Kinetic Theory (Plasma)", "ontic.cfd.fast_vlasov_5d", "FastVlasov5D", "XI"),
    DomainEntry("XI.4", "Gyrokinetics", "ontic.plasma.gyrokinetics", "ITGDispersion", "XI"),
    DomainEntry("XI.5", "Magnetic Reconnection", "ontic.plasma.magnetic_reconnection", "SweetParkerReconnection", "XI"),
    DomainEntry("XI.6", "Laser-Plasma", "ontic.plasma.laser_plasma", "StimulatedRamanScattering", "XI"),
    DomainEntry("XI.7", "Dusty Plasmas", "ontic.plasma.dusty_plasmas", "DustAcousticWave", "XI"),
    DomainEntry("XI.8", "Space & Astrophysical Plasma", "ontic.plasma.space_plasma", "ParkerSolarWind", "XI"),
]

# Category XII — Astrophysics & Cosmology (6 domains)
CAT_XII: list[DomainEntry] = [
    DomainEntry("XII.1", "Stellar Structure", "ontic.astro.stellar_structure", "StellarStructure", "XII"),
    DomainEntry("XII.2", "Compact Objects", "ontic.astro.compact_objects", "TOVSolver", "XII"),
    DomainEntry("XII.3", "Gravitational Waves", "ontic.astro.gravitational_waves", "PostNewtonianInspiral", "XII"),
    DomainEntry("XII.4", "Cosmological Sims", "ontic.astro.cosmological_sims", "FriedmannCosmology", "XII"),
    DomainEntry("XII.5", "CMB & Early Universe", "ontic.astro.cmb_early_universe", "BoltzmannHierarchy", "XII"),
    DomainEntry("XII.6", "Radiative Transfer", "ontic.astro.radiative_transfer", "RadiativeTransfer1D", "XII"),
]

# Category XIII — Geophysics & Earth Science (6 domains)
CAT_XIII: list[DomainEntry] = [
    DomainEntry("XIII.1", "Seismology", "ontic.geophysics.seismology", "AcousticWave2D", "XIII"),
    DomainEntry("XIII.2", "Mantle Convection", "ontic.geophysics.mantle_convection", "StokesFlow2D", "XIII"),
    DomainEntry("XIII.3", "Geodynamo", "ontic.geophysics.geodynamo", "MagneticInduction2D", "XIII"),
    DomainEntry("XIII.4", "Atmospheric Physics", "ontic.geophysics.atmosphere", "ChapmanOzone", "XIII"),
    DomainEntry("XIII.5", "Oceanography", "ontic.geophysics.oceanography", "SeawaterEOS", "XIII"),
    DomainEntry("XIII.6", "Glaciology", "ontic.geophysics.glaciology", "GlenFlowLaw", "XIII"),
]

# Category XIV — Materials Science (7 domains)
CAT_XIV: list[DomainEntry] = [
    DomainEntry("XIV.1", "First-Principles Design", "ontic.materials.first_principles_design", "ConvexHullStability", "XIV"),
    DomainEntry("XIV.2", "Mechanical Properties", "ontic.materials.mechanical_properties", "ElasticTensor", "XIV"),
    DomainEntry("XIV.3", "Phase-Field Methods", "ontic.phase_field", "AllenCahnSolver", "XIV"),
    DomainEntry("XIV.4", "Microstructure", "ontic.materials.microstructure", "CahnHilliard2D", "XIV"),
    DomainEntry("XIV.5", "Radiation Damage", "ontic.materials.radiation_damage", "NRTDisplacements", "XIV"),
    DomainEntry("XIV.6", "Polymers & Soft Matter", "ontic.materials.polymers_soft_matter", "FloryHuggins", "XIV"),
    DomainEntry("XIV.7", "Ceramics / High-Temp", "ontic.materials.ceramics", "SinteringModel", "XIV"),
]

# Category XV — Chemical Physics & Reaction Dynamics (7 domains)
CAT_XV: list[DomainEntry] = [
    DomainEntry("XV.1", "PES Construction", "ontic.chemistry.pes", "MorsePotential", "XV"),
    DomainEntry("XV.2", "Reaction Rate Theory", "ontic.chemistry.reaction_rate", "TransitionStateTheory", "XV"),
    DomainEntry("XV.3", "Quantum Reactive Dynamics", "ontic.chemistry.quantum_reactive", "CollinearReactiveScattering", "XV"),
    DomainEntry("XV.4", "Nonadiabatic Dynamics", "ontic.chemistry.nonadiabatic", "LandauZener", "XV"),
    DomainEntry("XV.5", "Photochemistry", "ontic.chemistry.photochemistry", "FranckCondonFactors", "XV"),
    DomainEntry("XV.6", "Catalysis", "ontic.fusion.resonant_catalysis", "ResonantCatalysisSolver", "XV"),
    DomainEntry("XV.7", "Spectroscopy", "ontic.chemistry.spectroscopy", "VibrationalSpectroscopy", "XV"),
]

# Category XVI — Biophysics & Computational Biology (6 domains)
CAT_XVI: list[DomainEntry] = [
    DomainEntry("XVI.1", "Protein Structure & Dynamics", "ontic.md.engine", "MDSimulation", "XVI"),
    DomainEntry("XVI.2", "Drug Design & Binding", "ontic.chemistry.pes", "LEPSPotential", "XVI"),
    DomainEntry("XVI.3", "Membrane Biophysics", "ontic.membrane_bio", "CoarseGrainedBilayer", "XVI"),
    DomainEntry("XVI.4", "Nucleic Acids & Regulation", "ontic.biology.systems_biology", "BooleanGRN", "XVI"),
    DomainEntry("XVI.5", "Systems Biology", "ontic.biology.systems_biology", "FluxBalanceAnalysis", "XVI"),
    DomainEntry("XVI.6", "Neuroscience / Biomedical", "ontic.medical.hemo", "ArterySimulation", "XVI"),
]

# Category XVII — Computational Methods — Cross-Cutting (6 domains)
CAT_XVII: list[DomainEntry] = [
    DomainEntry("XVII.1", "Optimization", "ontic.cfd.optimization", "ShapeOptimizer", "XVII"),
    DomainEntry("XVII.2", "Inverse Problems", "ontic.cfd.adjoint", "AdjointEuler2D", "XVII"),
    DomainEntry("XVII.3", "ML for Physics", "ontic.ml_physics", "PINN", "XVII"),
    DomainEntry("XVII.4", "Adaptive Mesh", "ontic.mesh_amr", "QuadtreeAMR", "XVII"),
    DomainEntry("XVII.5", "Large-Scale Linear Algebra", "ontic.algorithms.lanczos", "LanczosResult", "XVII"),
    DomainEntry("XVII.6", "HPC / Distributed", "ontic.distributed_tn", "DistributedDMRG", "XVII"),
]

# Category XVIII — Continuum Coupled Physics (7 domains)
CAT_XVIII: list[DomainEntry] = [
    DomainEntry("XVIII.1", "FSI", "ontic.fsi", "PartitionedFSICoupler", "XVIII"),
    DomainEntry("XVIII.2", "Thermo-Mechanical", "ontic.coupled.thermo_mechanical", "ThermoelasticSolver", "XVIII"),
    DomainEntry("XVIII.3", "Electro-Mechanical", "ontic.coupled.electro_mechanical", "PiezoelectricSolver", "XVIII"),
    DomainEntry("XVIII.4", "Coupled MHD", "ontic.coupled.coupled_mhd", "HartmannFlow", "XVIII"),
    DomainEntry("XVIII.5", "Chemically Reacting Flows", "ontic.cfd.chemistry", "Reaction", "XVIII"),
    DomainEntry("XVIII.6", "Radiation-Hydro", "ontic.radiation", "RadiationEuler1D", "XVIII"),
    DomainEntry("XVIII.7", "Multiscale Methods", "ontic.multiscale.multiscale", "FE2Solver", "XVIII"),
]

# Category XIX — Quantum Information & Computation (5 domains)
CAT_XIX: list[DomainEntry] = [
    DomainEntry("XIX.1", "Quantum Circuit Simulation", "ontic.quantum.hybrid", "QuantumCircuit", "XIX"),
    DomainEntry("XIX.2", "Quantum Error Correction", "ontic.quantum.error_mitigation", "ShorCode", "XIX"),
    DomainEntry("XIX.3", "Quantum Algorithms", "ontic.quantum.hybrid", "VQE", "XIX"),
    DomainEntry("XIX.4", "Quantum Simulation", "ontic.algorithms", "tebd", "XIX"),
    DomainEntry("XIX.5", "Quantum Crypto & Communication", "ontic.oracle.core.oracle", "ORACLE", "XIX"),
]

# Category XX — Special / Applied Domains (10 domains)
CAT_XX: list[DomainEntry] = [
    DomainEntry("XX.1", "Relativistic Mechanics", "ontic.relativity.relativistic_mechanics", "LorentzBoost", "XX"),
    DomainEntry("XX.2", "Numerical GR", "ontic.relativity.numerical_gr", "BSSNState", "XX"),
    DomainEntry("XX.3", "Astrodynamics", "ontic.guidance.trajectory", "IntegrationMethod", "XX"),
    DomainEntry("XX.4", "Robotics Physics", "ontic.robotics_physics", "FeatherstoneABA", "XX"),
    DomainEntry("XX.5", "Applied Acoustics", "ontic.acoustics.applied_acoustics", "LinearisedEulerEquations", "XX"),
    DomainEntry("XX.6", "Biomedical Engineering", "ontic.biomedical.biomedical", "BidomainSolver", "XX"),
    DomainEntry("XX.7", "Environmental Physics", "ontic.environmental.environmental", "GaussianPlume", "XX"),
    DomainEntry("XX.8", "Energy Systems", "ontic.energy.energy_systems", "DriftDiffusionSolarCell", "XX"),
    DomainEntry("XX.9", "Manufacturing Physics", "ontic.manufacturing.manufacturing", "GoldakWeldingSource", "XX"),
    DomainEntry("XX.10", "Semiconductor Device", "ontic.radiation", "ICFImplosion", "XX"),
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
    a(f"*Generated by `ontic.benchmark_runner` v{report.version}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ontic.benchmark_runner",
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
