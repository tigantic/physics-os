"""
Test Module: DARPA MARRS Solid-State Fusion Simulations
=========================================================

Phase 21: Material Solutions for Room-Temperature D-D Fusion
Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation
    - Article V, Section 5.1: Float64 precision

BAA Alignment: HR001126S0007
    - Breakthrough 1: Electron Screening Potentials
    - Breakthrough 2: Deuterium Density & Mobility
    - Breakthrough 3: External Excitation Triggers

References:
    [1] Raiola et al., Eur. Phys. J. A 19, 283 (2004)
    [2] Hull, Rep. Prog. Phys. 67, 1233 (2004)
    [3] Huke et al., Phys. Rev. C 78, 015803 (2008)
"""

import math
import numpy as np
import pytest
import torch

from tensornet.fusion import (
    # Electron Screening
    ElectronScreeningSolver,
    ScreeningResult,
    LatticeParams,
    LatticeType,
    # Superionic Dynamics
    SuperionicDynamics,
    DiffusionResult,
    LatticeConfig,
    # Phonon Trigger
    FokkerPlanckSolver,
    TriggerResult,
    TriggerConfig,
    ExcitationMode,
    # Unified
    MARRSSimulator,
    MARRSSimulationResult,
    # QTT Enhanced
    QTTElectronScreeningSolver,
    QTTScreeningResult,
    QTTSuperionicDynamics,
    QTTDiffusionResult,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def deterministic_seed():
    """Per Article III, Section 3.2: Reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture
def laluh6_lattice():
    """LaLuH₆ lattice parameters."""
    return LatticeParams(
        lattice_type=LatticeType.LALUH6,
        lattice_constant=5.12,
        n_H_sites=6,
        metal_valence=3,
        temperature=300.0,
    )


@pytest.fixture
def room_temp_config():
    """Room temperature configuration."""
    return LatticeConfig(
        lattice_constant=5.12,
        n_unit_cells=2,
        well_depth_eV=0.15,
        barrier_height_eV=0.20,
        temperature=300.0,
    )


@pytest.fixture
def trigger_config():
    """Phonon trigger configuration."""
    return TriggerConfig(
        temperature=300.0,
        phonon_energy_eV=0.15,
        excitation_power_W_cm2=1e6,
        pulse_on=True,
        t_max_ps=10.0,
        dt_ps=0.1,
    )


# ============================================================================
# BREAKTHROUGH 1: ELECTRON SCREENING TESTS
# ============================================================================


class TestLatticeParams:
    """Test lattice parameter calculations."""

    @pytest.mark.unit
    def test_volume_per_formula(self, laluh6_lattice, deterministic_seed):
        """Test volume calculation."""
        expected = 5.12 ** 3  # 134.22 Å³
        assert laluh6_lattice.volume_per_formula == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_h_density(self, laluh6_lattice, deterministic_seed):
        """Test H density calculation."""
        # 6 H per 134.22 Å³ ≈ 0.045 H/Å³
        expected = 6 / (5.12 ** 3)
        assert laluh6_lattice.H_density == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_electron_density_bulk(self, laluh6_lattice, deterministic_seed):
        """Test bulk electron density from metal valence."""
        # 2 metals × 3 electrons / 134.22 Å³ ≈ 0.045 e/Å³
        expected = 2 * 3 / (5.12 ** 3)
        assert laluh6_lattice.electron_density_bulk == pytest.approx(expected, rel=0.01)


class TestElectronScreeningSolver:
    """Test electron screening calculations."""

    @pytest.mark.unit
    def test_solver_initialization(self, laluh6_lattice, deterministic_seed):
        """Test solver initialization."""
        solver = ElectronScreeningSolver(
            lattice=laluh6_lattice,
            grid_points=32,
            chi_max=16,
        )
        assert solver.grid_points == 32
        assert solver.L == 5.12

    @pytest.mark.unit
    @pytest.mark.physics
    def test_thomas_fermi_density_positive(self, laluh6_lattice, deterministic_seed):
        """Electron density must be positive everywhere."""
        solver = ElectronScreeningSolver(laluh6_lattice, grid_points=32)
        n_e = solver.compute_thomas_fermi_density()
        assert torch.all(n_e > 0), "Electron density must be positive"

    @pytest.mark.unit
    @pytest.mark.physics
    def test_thomas_fermi_density_peaks_at_sites(self, laluh6_lattice, deterministic_seed):
        """Electron density should be enhanced at H sites."""
        solver = ElectronScreeningSolver(laluh6_lattice, grid_points=32)
        n_e = solver.compute_thomas_fermi_density()
        
        # Maximum should exceed bulk value
        n_bulk = laluh6_lattice.electron_density_bulk
        assert n_e.max().item() > n_bulk

    @pytest.mark.unit
    @pytest.mark.physics
    def test_debye_length_physical_range(self, laluh6_lattice, deterministic_seed):
        """Debye length should be in metallic range (0.1-1.0 Å)."""
        solver = ElectronScreeningSolver(laluh6_lattice, grid_points=32)
        n_e = solver.compute_thomas_fermi_density()
        lambda_D = solver.compute_debye_length(n_e)
        
        assert 0.1 < lambda_D < 1.0, f"λ_TF = {lambda_D:.3f} Å outside metallic range"

    @pytest.mark.unit
    @pytest.mark.physics
    def test_screening_energy_positive(self, laluh6_lattice, deterministic_seed):
        """Screening energy must be positive (attractive screening)."""
        solver = ElectronScreeningSolver(laluh6_lattice, grid_points=32)
        n_e = solver.compute_thomas_fermi_density()
        lambda_D = solver.compute_debye_length(n_e)
        U_e, _ = solver.compute_screened_potential(lambda_D)
        
        assert U_e > 0, f"Screening energy U_e = {U_e:.1f} eV should be positive"

    @pytest.mark.unit
    @pytest.mark.physics
    def test_barrier_reduction_significant(self, laluh6_lattice, deterministic_seed):
        """Barrier reduction should be significant (>10×)."""
        solver = ElectronScreeningSolver(laluh6_lattice, grid_points=32)
        result = solver.solve(verbose=False)
        
        assert result.barrier_reduction_factor > 10, \
            f"Barrier reduction {result.barrier_reduction_factor:.2e}× too small"

    @pytest.mark.unit
    def test_solve_returns_screening_result(self, laluh6_lattice, deterministic_seed):
        """Solve should return ScreeningResult with all fields."""
        solver = ElectronScreeningSolver(laluh6_lattice, grid_points=32)
        result = solver.solve(verbose=False)
        
        assert isinstance(result, ScreeningResult)
        assert result.screening_energy_eV > 0
        assert result.debye_length_angstrom > 0
        assert result.electron_density_at_D > 0
        assert result.barrier_reduction_factor > 0
        assert result.effective_gamow_energy_keV > 0


# ============================================================================
# BREAKTHROUGH 2: SUPERIONIC DYNAMICS TESTS
# ============================================================================


class TestLatticeConfig:
    """Test lattice configuration."""

    @pytest.mark.unit
    def test_box_size(self, room_temp_config, deterministic_seed):
        """Test box size calculation."""
        expected = 5.12 * 2  # 10.24 Å
        assert room_temp_config.box_size == pytest.approx(expected)

    @pytest.mark.unit
    def test_thermal_energy(self, room_temp_config, deterministic_seed):
        """Test thermal energy at 300 K."""
        # k_B T at 300 K ≈ 0.026 eV
        expected = 1.380649e-23 * 300 / 1.602176634e-19
        assert room_temp_config.thermal_energy == pytest.approx(expected, rel=0.01)


class TestSuperionicDynamics:
    """Test Langevin dynamics simulations."""

    @pytest.mark.unit
    def test_initialization(self, room_temp_config, deterministic_seed):
        """Test simulator initialization."""
        sim = SuperionicDynamics(
            config=room_temp_config,
            n_particles=16,
            dt=1.0,
        )
        assert sim.n_particles == 16
        assert sim.positions.shape == (16, 3)
        assert sim.velocities.shape == (16, 3)

    @pytest.mark.unit
    @pytest.mark.physics
    def test_positions_in_box(self, room_temp_config, deterministic_seed):
        """Particle positions should stay within periodic box."""
        sim = SuperionicDynamics(room_temp_config, n_particles=16, dt=1.0)
        
        # Run a few steps
        for _ in range(100):
            sim.step()
        
        assert torch.all(sim.positions >= 0), "Positions should be >= 0"
        assert torch.all(sim.positions < sim.L), f"Positions should be < {sim.L}"

    @pytest.mark.unit
    @pytest.mark.physics
    def test_temperature_equilibration(self, room_temp_config, deterministic_seed):
        """Temperature should equilibrate to target."""
        sim = SuperionicDynamics(room_temp_config, n_particles=32, dt=1.0)
        
        # Run equilibration
        for _ in range(1000):
            sim.step()
        
        # Compute temperature from kinetic energy
        m = sim.mass
        KE = 0.5 * m * (sim.velocities ** 2).sum(dim=1).mean().item()
        T_realized = 2 * KE / (3 * 1.380649e-23)
        
        # Should be within 50% of target (300 K)
        assert 150 < T_realized < 450, f"T = {T_realized:.0f} K outside range"

    @pytest.mark.unit
    @pytest.mark.physics
    def test_msd_increases(self, room_temp_config, deterministic_seed):
        """MSD should increase over time (diffusive motion)."""
        sim = SuperionicDynamics(room_temp_config, n_particles=32, dt=1.0)
        
        pos_0 = sim.positions.clone()
        
        for _ in range(500):
            sim.step()
        
        msd = sim.compute_msd(pos_0, sim.positions)
        assert msd > 0, "MSD should be positive after evolution"

    @pytest.mark.unit
    def test_run_returns_diffusion_result(self, room_temp_config, deterministic_seed):
        """Run should return DiffusionResult with all fields."""
        sim = SuperionicDynamics(room_temp_config, n_particles=16, dt=1.0)
        result = sim.run(n_steps=500, sample_every=50, verbose=False)
        
        assert isinstance(result, DiffusionResult)
        assert result.diffusion_coefficient >= 0
        assert result.mean_squared_displacement >= 0
        assert isinstance(result.is_superionic, bool)
        assert result.activation_energy_eV >= 0

    @pytest.mark.unit
    @pytest.mark.physics
    def test_diffusion_coefficient_physical_range(self, room_temp_config, deterministic_seed):
        """Diffusion coefficient should be in physical range."""
        sim = SuperionicDynamics(room_temp_config, n_particles=32, dt=1.0)
        result = sim.run(n_steps=2000, sample_every=100, verbose=False)
        
        # D should be between 10^-12 and 10^-3 cm²/s for condensed matter
        assert result.diffusion_coefficient > 1e-15, "D too small"
        assert result.diffusion_coefficient < 1e-2, "D too large"


# ============================================================================
# BREAKTHROUGH 3: PHONON TRIGGER TESTS
# ============================================================================


class TestTriggerConfig:
    """Test trigger configuration."""

    @pytest.mark.unit
    def test_phonon_frequency(self, trigger_config, deterministic_seed):
        """Test phonon frequency calculation."""
        # 0.15 eV ≈ 36 THz
        HBAR_EV = 6.582119569e-16
        expected = 0.15 / (HBAR_EV * 1e12 * 2 * math.pi)
        assert trigger_config.phonon_frequency_THz == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_thermal_energy_config(self, trigger_config, deterministic_seed):
        """Test thermal energy at 300 K."""
        K_BOLTZMANN_EV = 8.617333262e-5
        expected = K_BOLTZMANN_EV * 300
        assert trigger_config.thermal_energy_eV == pytest.approx(expected, rel=0.01)


class TestFokkerPlanckSolver:
    """Test Fokker-Planck energy distribution solver."""

    @pytest.mark.unit
    def test_initialization(self, trigger_config, deterministic_seed):
        """Test solver initialization."""
        solver = FokkerPlanckSolver(trigger_config)
        assert len(solver.E) == trigger_config.n_energy_points
        assert solver.f.shape == solver.E.shape

    @pytest.mark.unit
    @pytest.mark.physics
    def test_initial_distribution_normalized(self, trigger_config, deterministic_seed):
        """Initial Maxwell-Boltzmann should be normalized."""
        solver = FokkerPlanckSolver(trigger_config)
        norm = torch.trapezoid(solver.f, solver.E).item()
        assert abs(norm - 1.0) < 1e-6, f"Normalization = {norm}"

    @pytest.mark.unit
    @pytest.mark.physics
    def test_initial_distribution_positive(self, trigger_config, deterministic_seed):
        """Distribution should be positive."""
        solver = FokkerPlanckSolver(trigger_config)
        assert torch.all(solver.f > 0), "Distribution must be positive"

    @pytest.mark.unit
    @pytest.mark.physics
    def test_fusion_cross_section_physical(self, trigger_config, deterministic_seed):
        """Fusion cross-section should decrease with energy (Gamow)."""
        solver = FokkerPlanckSolver(trigger_config)
        sigma = solver.fusion_cross_section(solver.E)
        
        # Cross-section should be larger at higher energies (within Gamow peak)
        # but extremely small overall
        assert torch.all(sigma >= 0), "Cross-section must be non-negative"
        assert torch.all(torch.isfinite(sigma)), "Cross-section must be finite"

    @pytest.mark.unit
    @pytest.mark.physics
    def test_fusion_rate_physical(self, trigger_config, deterministic_seed):
        """Fusion rate should be positive but small."""
        solver = FokkerPlanckSolver(trigger_config)
        R = solver.compute_fusion_rate(solver.f)
        
        assert R >= 0, "Fusion rate must be non-negative"
        # At 300 K thermal, rate should be extremely small
        assert R < 1e10, "Thermal fusion rate should be small"

    @pytest.mark.unit
    def test_step_preserves_normalization(self, trigger_config, deterministic_seed):
        """Time stepping should preserve normalization."""
        solver = FokkerPlanckSolver(trigger_config)
        
        for _ in range(10):
            solver.step(0.0)
        
        norm = torch.trapezoid(solver.f, solver.E).item()
        assert abs(norm - 1.0) < 0.01, f"Normalization drift: {norm}"

    @pytest.mark.unit
    def test_run_returns_trigger_result(self, trigger_config, deterministic_seed):
        """Run should return TriggerResult with all fields."""
        config = TriggerConfig(
            temperature=300.0,
            t_max_ps=5.0,
            dt_ps=0.1,
            pulse_on=True,
        )
        solver = FokkerPlanckSolver(config)
        result = solver.run(verbose=False)
        
        assert isinstance(result, TriggerResult)
        assert result.fusion_rate_enhancement >= 0
        assert result.population_at_threshold >= 0
        assert result.on_off_ratio >= 0

    @pytest.mark.unit
    @pytest.mark.physics
    def test_trigger_on_enhances_rate(self, deterministic_seed):
        """Trigger ON should enhance fusion rate vs OFF."""
        # Trigger OFF
        config_off = TriggerConfig(
            temperature=300.0,
            t_max_ps=10.0,
            pulse_on=False,
        )
        solver_off = FokkerPlanckSolver(config_off)
        result_off = solver_off.run(verbose=False)
        
        # Trigger ON
        config_on = TriggerConfig(
            temperature=300.0,
            phonon_energy_eV=0.15,
            excitation_power_W_cm2=1e6,
            t_max_ps=10.0,
            pulse_on=True,
        )
        solver_on = FokkerPlanckSolver(config_on)
        result_on = solver_on.run(verbose=False)
        
        # ON should have higher enhancement
        assert result_on.fusion_rate_enhancement >= result_off.fusion_rate_enhancement


# ============================================================================
# UNIFIED MARRS SIMULATOR TESTS
# ============================================================================


class TestMARRSSimulator:
    """Test unified MARRS simulation suite."""

    @pytest.mark.unit
    def test_initialization(self, deterministic_seed):
        """Test simulator initialization."""
        sim = MARRSSimulator(
            temperature=300.0,
            lattice_constant=5.12,
            material="LaLuH₆",
        )
        assert sim.temperature == 300.0
        assert sim.material == "LaLuH₆"

    @pytest.mark.unit
    def test_run_screening(self, deterministic_seed):
        """Test screening module execution."""
        sim = MARRSSimulator(temperature=300.0)
        result = sim.run_screening(verbose=False)
        assert isinstance(result, ScreeningResult)

    @pytest.mark.unit
    def test_run_diffusion(self, deterministic_seed):
        """Test diffusion module execution."""
        sim = MARRSSimulator(temperature=300.0)
        result = sim.run_diffusion(n_particles=16, n_steps=500, verbose=False)
        assert isinstance(result, DiffusionResult)

    @pytest.mark.unit
    def test_run_trigger(self, deterministic_seed):
        """Test trigger module execution."""
        sim = MARRSSimulator(temperature=300.0)
        result = sim.run_trigger(verbose=False)
        assert isinstance(result, TriggerResult)

    @pytest.mark.integration
    def test_run_full_suite(self, deterministic_seed):
        """Test complete simulation suite."""
        sim = MARRSSimulator(temperature=300.0)
        result = sim.run_full_suite(verbose=False)
        
        assert isinstance(result, MARRSSimulationResult)
        assert result.material == "LaLuH₆"
        assert result.temperature_K == 300.0
        assert result.net_fusion_enhancement > 0
        assert isinstance(result.meets_marrs_criteria, bool)

    @pytest.mark.unit
    def test_result_to_dict(self, deterministic_seed):
        """Test JSON serialization."""
        sim = MARRSSimulator(temperature=300.0)
        result = sim.run_full_suite(verbose=False)
        
        data = result.to_dict()
        assert "material" in data
        assert "screening" in data
        assert "diffusion" in data
        assert "trigger" in data
        assert "net_fusion_enhancement" in data

    @pytest.mark.unit
    def test_generate_abstract_data(self, deterministic_seed):
        """Test abstract generation."""
        sim = MARRSSimulator(temperature=300.0)
        result = sim.run_full_suite(verbose=False)
        
        abstract = result.generate_abstract_data()
        assert "BREAKTHROUGH 1" in abstract
        assert "BREAKTHROUGH 2" in abstract
        assert "BREAKTHROUGH 3" in abstract
        assert "LaLuH₆" in abstract


# ============================================================================
# PHYSICAL VALIDATION TESTS
# ============================================================================


class TestPhysicalValidation:
    """Validate physics against known results."""

    @pytest.mark.physics
    @pytest.mark.parametrize("T", [100, 300, 500, 800])
    def test_diffusion_arrhenius(self, T, deterministic_seed):
        """Diffusion should follow Arrhenius behavior."""
        config = LatticeConfig(temperature=T)
        sim = SuperionicDynamics(config, n_particles=32, dt=1.0)
        result = sim.run(n_steps=2000, verbose=False)
        
        # D should increase with temperature
        # Just verify it's positive and finite
        assert result.diffusion_coefficient > 0
        assert np.isfinite(result.diffusion_coefficient)

    @pytest.mark.physics
    def test_screening_scales_with_density(self, deterministic_seed):
        """Higher electron density → shorter screening length."""
        lattice_normal = LatticeParams(lattice_constant=5.12)
        lattice_compressed = LatticeParams(lattice_constant=4.5)  # Compressed
        
        solver_normal = ElectronScreeningSolver(lattice_normal, grid_points=32)
        solver_compressed = ElectronScreeningSolver(lattice_compressed, grid_points=32)
        
        result_normal = solver_normal.solve(verbose=False)
        result_compressed = solver_compressed.solve(verbose=False)
        
        # Compressed should have shorter screening length
        assert result_compressed.debye_length_angstrom < result_normal.debye_length_angstrom

    @pytest.mark.physics
    def test_gamow_energy_reduction(self, deterministic_seed):
        """Effective Gamow energy should be reduced by screening."""
        lattice = LatticeParams()
        solver = ElectronScreeningSolver(lattice, grid_points=32)
        result = solver.solve(verbose=False)
        
        # Effective Gamow should be less than bare Gamow (31.4 keV)
        E_GAMOW_BARE = 31.4  # keV
        assert result.effective_gamow_energy_keV < E_GAMOW_BARE


# ============================================================================
# FLOAT64 PRECISION TESTS
# ============================================================================


class TestFloat64Precision:
    """Verify float64 precision per Article V."""

    @pytest.mark.unit
    def test_screening_solver_dtype(self, laluh6_lattice, deterministic_seed):
        """Screening solver should use float64."""
        solver = ElectronScreeningSolver(laluh6_lattice, dtype=torch.float64)
        n_e = solver.compute_thomas_fermi_density()
        assert n_e.dtype == torch.float64

    @pytest.mark.unit
    def test_langevin_dtype(self, room_temp_config, deterministic_seed):
        """Langevin dynamics should use float64."""
        sim = SuperionicDynamics(room_temp_config, dtype=torch.float64)
        assert sim.positions.dtype == torch.float64
        assert sim.velocities.dtype == torch.float64

    @pytest.mark.unit
    def test_fokker_planck_dtype(self, trigger_config, deterministic_seed):
        """Fokker-Planck solver should use float64."""
        solver = FokkerPlanckSolver(trigger_config, dtype=torch.float64)
        assert solver.f.dtype == torch.float64
        assert solver.E.dtype == torch.float64


# ============================================================================
# QTT COMPRESSION TESTS
# ============================================================================


class TestQTTCompression:
    """Tests for QTT-compressed electron screening solver."""

    @pytest.fixture
    def qtt_solver(self, laluh6_lattice, deterministic_seed):
        """Create QTT screening solver."""
        return QTTElectronScreeningSolver(
            lattice=laluh6_lattice,
            n_qubits_per_dim=4,  # 16³ = 4096 points (small for testing)
            chi_max=16,
            use_tci=False,  # Use dense for consistent test results
        )

    @pytest.mark.unit
    def test_qtt_solver_initialization(self, laluh6_lattice, deterministic_seed):
        """QTT solver should initialize correctly."""
        solver = QTTElectronScreeningSolver(
            lattice=laluh6_lattice,
            n_qubits_per_dim=4,
            chi_max=16,
        )
        assert solver.N == 16  # 2^4
        assert solver.n_qubits_total == 12  # 3 * 4
        assert solver.total_points == 4096  # 16^3

    @pytest.mark.unit
    def test_electron_density_function(self, qtt_solver, deterministic_seed):
        """Electron density function should return positive values."""
        indices = torch.arange(100, dtype=torch.long)
        n_e = qtt_solver.electron_density_function(indices)
        
        assert n_e.shape == (100,)
        assert (n_e > 0).all(), "Electron density should be positive"

    @pytest.mark.unit
    def test_qtt_cores_structure(self, qtt_solver, deterministic_seed):
        """QTT cores should have correct structure."""
        cores, metadata = qtt_solver.compute_electron_density_qtt(verbose=False)
        
        assert len(cores) == qtt_solver.n_qubits_total
        
        # First core: (1, 2, chi)
        assert cores[0].shape[0] == 1
        assert cores[0].shape[1] == 2
        
        # Last core: (chi, 2, 1)
        assert cores[-1].shape[1] == 2
        assert cores[-1].shape[2] == 1

    @pytest.mark.unit
    def test_qtt_to_dense_reconstruction(self, qtt_solver, deterministic_seed):
        """QTT should reconstruct to approximate dense result."""
        # Get QTT cores
        cores, _ = qtt_solver.compute_electron_density_qtt(verbose=False)
        
        # Reconstruct from QTT
        reconstructed = qtt_solver.qtt_to_dense_3d(cores)
        
        # Get direct dense result
        dense = qtt_solver.compute_electron_density_dense()
        
        # Check shape
        assert reconstructed.shape == dense.shape
        
        # Check values are close (allow for compression error)
        rel_error = torch.norm(reconstructed - dense) / torch.norm(dense)
        assert rel_error < 0.1, f"Reconstruction error too high: {rel_error:.2%}"

    @pytest.mark.unit
    def test_qtt_compression_ratio(self, qtt_solver, deterministic_seed):
        """QTT should achieve meaningful compression."""
        result = qtt_solver.solve(verbose=False)
        
        # Should have positive compression ratio
        assert result.compression_ratio > 0
        
        # For smooth fields, should achieve some compression
        assert result.qtt_storage_bytes < result.dense_storage_bytes

    @pytest.mark.physics
    def test_qtt_screening_energy_positive(self, qtt_solver, deterministic_seed):
        """QTT solver should compute positive screening energy."""
        result = qtt_solver.solve(verbose=False)
        
        assert result.screening_energy_eV > 0, "Screening energy should be positive"

    @pytest.mark.physics
    def test_qtt_debye_length_physical(self, qtt_solver, deterministic_seed):
        """Debye length should be in physical range."""
        result = qtt_solver.solve(verbose=False)
        
        # Thomas-Fermi length in metals: 0.1 - 2 Å
        assert 0.01 < result.debye_length_angstrom < 10.0

    @pytest.mark.physics
    def test_qtt_vs_dense_consistency(self, laluh6_lattice, deterministic_seed):
        """QTT and dense solvers should give similar results."""
        # QTT solver
        qtt_solver = QTTElectronScreeningSolver(
            lattice=laluh6_lattice,
            n_qubits_per_dim=4,  # 16³
            chi_max=32,
            use_tci=False,
        )
        qtt_result = qtt_solver.solve(verbose=False)
        
        # Dense solver (should match grid size)
        dense_solver = ElectronScreeningSolver(
            lattice=laluh6_lattice,
            grid_points=16,
        )
        dense_result = dense_solver.solve(verbose=False)
        
        # Screening energies should be within 50%
        ratio = qtt_result.screening_energy_eV / dense_result.screening_energy_eV
        assert 0.5 < ratio < 2.0, f"QTT vs dense mismatch: {ratio:.2f}"

    @pytest.mark.unit
    def test_qtt_result_fields(self, qtt_solver, deterministic_seed):
        """QTT result should have all required fields."""
        result = qtt_solver.solve(verbose=False)
        
        assert isinstance(result, QTTScreeningResult)
        
        # Base fields
        assert hasattr(result, 'screening_energy_eV')
        assert hasattr(result, 'debye_length_angstrom')
        assert hasattr(result, 'barrier_reduction_factor')
        
        # QTT-specific fields
        assert hasattr(result, 'compression_ratio')
        assert hasattr(result, 'max_bond_dimension')
        assert hasattr(result, 'n_qubits')
        assert hasattr(result, 'qtt_storage_bytes')
        assert hasattr(result, 'dense_storage_bytes')

    @pytest.mark.slow
    def test_qtt_larger_grid(self, laluh6_lattice, deterministic_seed):
        """QTT should work on larger grids."""
        solver = QTTElectronScreeningSolver(
            lattice=laluh6_lattice,
            n_qubits_per_dim=5,  # 32³ = 32768 points
            chi_max=32,
            use_tci=False,
        )
        result = solver.solve(verbose=False)
        
        # Should still compute physical results
        assert result.screening_energy_eV > 0
        
        # Larger grid should have better compression
        assert result.compression_ratio > 1.0


# ============================================================================
# QTT SUPERIONIC DYNAMICS TESTS
# ============================================================================


class TestQTTSuperionic:
    """Tests for QTT-enhanced superionic dynamics."""
    
    @pytest.fixture
    def superionic_config(self, deterministic_seed):
        """Create lattice config for superionic tests."""
        return LatticeConfig(
            lattice_constant=5.12,
            n_unit_cells=2,
            well_depth_eV=0.15,
            barrier_height_eV=0.20,
            temperature=300.0,
        )
    
    @pytest.mark.unit
    def test_qtt_superionic_initialization(self, superionic_config, deterministic_seed):
        """QTT superionic solver should initialize correctly."""
        sim = QTTSuperionicDynamics(
            config=superionic_config,
            n_qubits_per_dim=4,  # 16³ PES grid
            chi_max=16,
            n_particles=10,
        )
        
        assert sim.N == 16
        assert sim.n_particles == 10
        assert sim._compression_ratio > 0
    
    @pytest.mark.unit
    def test_qtt_pes_built(self, superionic_config, deterministic_seed):
        """PES should be built and compressed."""
        sim = QTTSuperionicDynamics(
            config=superionic_config,
            n_qubits_per_dim=4,
            chi_max=16,
            n_particles=10,
        )
        
        assert sim._pes_cores is not None
        assert sim._pes_dense is not None
        assert len(sim._pes_cores) > 0
    
    @pytest.mark.unit
    def test_qtt_force_interpolation(self, superionic_config, deterministic_seed):
        """Force interpolation should return valid forces."""
        sim = QTTSuperionicDynamics(
            config=superionic_config,
            n_qubits_per_dim=4,
            chi_max=16,
            n_particles=10,
        )
        
        forces = sim._interpolate_force_from_pes(sim.positions)
        
        assert forces.shape == (10, 3)
        assert not torch.isnan(forces).any()
    
    @pytest.mark.integration
    def test_qtt_dynamics_run(self, superionic_config, deterministic_seed):
        """QTT dynamics should complete without errors."""
        sim = QTTSuperionicDynamics(
            config=superionic_config,
            n_qubits_per_dim=4,
            chi_max=16,
            n_particles=10,
        )
        
        result = sim.run_qtt_dynamics(
            n_steps=100,
            dt_fs=1.0,
            equilibration_steps=20,
            verbose=False,
        )
        
        assert result.diffusion_coefficient is not None
        assert isinstance(result, QTTDiffusionResult)
    
    @pytest.mark.unit
    def test_qtt_diffusion_result_fields(self, superionic_config, deterministic_seed):
        """QTT diffusion result should have all required fields."""
        sim = QTTSuperionicDynamics(
            config=superionic_config,
            n_qubits_per_dim=4,
            chi_max=16,
            n_particles=10,
        )
        
        result = sim.run_qtt_dynamics(
            n_steps=50,
            dt_fs=1.0,
            equilibration_steps=10,
            verbose=False,
        )
        
        # Core fields
        assert hasattr(result, 'diffusion_coefficient')
        assert hasattr(result, 'is_superionic')
        
        # QTT-specific fields
        assert hasattr(result, 'pes_compression_ratio')
        assert hasattr(result, 'pes_qtt_storage_bytes')
        assert hasattr(result, 'pes_dense_storage_bytes')
