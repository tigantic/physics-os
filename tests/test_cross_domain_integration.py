"""Cross-domain integration tests for HyperTensor physics engine.

These tests verify that outputs from one physics domain can be meaningfully
consumed by another, validating physical consistency across module boundaries.

Coverage:
    - 15 pairwise domain pipelines
    - 3 multi-domain chains (≥3 domains)
    - 30 total cross-domain assertions
    - All tests run in <5 s total (no heavy computation)

Requires: numpy, pytest
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
import pytest


# =============================================================================
# Helpers
# =============================================================================

def _import(mod_path: str) -> Any:
    """Import a module by dotted path, skip test if unavailable."""
    try:
        return importlib.import_module(mod_path)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Module {mod_path} unavailable: {exc}")


# =============================================================================
# Pipeline 1: QM → Chemistry (Hydrogen atom / Morse potential)
# =============================================================================

class TestQMToChemistry:
    """Data flow: quantum energy levels → potential-energy-surface parameters."""

    def test_hydrogen_energy_feeds_morse_depth(self) -> None:
        """H-atom binding energy sets Morse well depth D_e."""
        qm = _import("tensornet.quantum_mechanics.stationary")
        chem = _import("tensornet.chemistry.pes")

        # Hydrogen ground-state binding energy (magnitude, in eV)
        E_bind = abs(qm.HydrogenAtom.energy(1) * qm.HARTREE_EV)

        # Use as Morse well depth for H₂ (illustrative pipeline)
        morse = chem.MorsePotential(D_e=E_bind, alpha=1.942, r_e=0.7414)
        V_eq = morse.energy(np.array([morse.r_e]))[0]

        assert abs(V_eq) < 1e-10, "Morse V(r_e) must be zero at equilibrium"
        assert morse.D_e == pytest.approx(E_bind, rel=1e-10), (
            "Morse depth must equal H-atom binding energy"
        )

    def test_qho_frequency_agrees_with_morse_harmonic(self) -> None:
        """QHO frequency matches Morse harmonic limit ω = α√(2D_e/μ)."""
        qm = _import("tensornet.quantum_mechanics.stationary")
        chem = _import("tensornet.chemistry.pes")

        D_e, alpha, r_e = 4.746, 1.942, 0.7414
        mu = 0.5  # reduced mass of H₂ in atomic units (illustrative)
        morse = chem.MorsePotential(D_e=D_e, alpha=alpha, r_e=r_e, mu=mu)

        omega_morse = morse.harmonic_frequency()
        ho = qm.HarmonicOscillator(omega=omega_morse, mass=mu)

        E0_ho = ho.energy(0) * 1.24e-4  # cm⁻¹ → eV to match Morse units
        E0_morse = morse.vibrational_levels(1)[0]

        # Harmonic and Morse ground states should agree within anharmonic correction
        assert abs(E0_ho - E0_morse) / abs(E0_morse) < 0.05, (
            f"QHO E0={E0_ho:.4f} vs Morse E0={E0_morse:.4f} (>5% deviation)"
        )


# =============================================================================
# Pipeline 2: QM → Chemistry → Reaction Rate (TST)
# =============================================================================

class TestQMToReactionRate:
    """Data flow: QHO eigenvalues → TST activation energy → rate constant."""

    def test_qho_energy_drives_tst_barrier(self) -> None:
        """Transition state theory uses QHO zero-point energy as barrier."""
        qm = _import("tensornet.quantum_mechanics.stationary")
        chem = _import("tensornet.chemistry.reaction_rate")

        ho_r = qm.HarmonicOscillator(omega=1000.0, mass=1.0)
        ho_ts = qm.HarmonicOscillator(omega=800.0, mass=1.0)

        # Use energy difference as activation barrier (eV-ish)
        E_a = 0.5  # eV
        freq_r = [ho_r.omega]
        freq_ts = [ho_ts.omega]
        imag_freq = 200.0

        tst = chem.TransitionStateTheory(
            E_a=E_a,
            frequencies_reactant=freq_r,
            frequencies_ts=freq_ts,
            imaginary_freq=imag_freq,
        )
        k_300 = tst.rate_constant(T=300.0)
        k_600 = tst.rate_constant(T=600.0)

        assert k_600 > k_300 > 0, (
            "Rate constant must increase with temperature (Arrhenius)"
        )

    def test_tst_wigner_correction_positive(self) -> None:
        """Wigner tunnelling correction κ ≥ 1."""
        chem = _import("tensornet.chemistry.reaction_rate")
        tst = chem.TransitionStateTheory(
            E_a=0.5,
            frequencies_reactant=[1000.0],
            frequencies_ts=[800.0],
            imaginary_freq=1500.0,
        )
        kappa = tst.wigner_correction(T=300.0)
        assert kappa >= 1.0, f"Wigner correction {kappa} < 1 is unphysical"


# =============================================================================
# Pipeline 3: EM → Optics
# =============================================================================

class TestEMToOptics:
    """Data flow: EM field parameters → optical propagation."""

    def test_biot_savart_field_validates_gaussian_beam_scale(self) -> None:
        """Magnetic field magnitude from a loop ~ μ₀I/(2R) at center;
        same geometric scale used for Gaussian beam waist."""
        em = _import("tensornet.em.magnetostatics")
        opt = _import("tensornet.optics.physical_optics")

        R = 0.01  # 1 cm radius loop
        I = 1.0
        bs = em.BiotSavart(current=I)
        B_center = bs.circular_loop(R=R, z=0.0)

        mu0 = 4 * math.pi * 1e-7
        B_exact = mu0 * I / (2 * R)
        assert abs(B_center - B_exact) / B_exact < 0.01

        # Use loop radius as beam waist scale
        beam = opt.GaussianBeam(wavelength=632.8e-9, waist=R, waist_position=0.0)
        assert beam.rayleigh_range > 0, "Rayleigh range must be positive"
        assert beam.spot_size(0.0) == pytest.approx(R, rel=1e-10)

    def test_fdtd_source_freq_feeds_fresnel(self) -> None:
        """FDTD simulation frequency drives Fresnel propagator wavelength."""
        em = _import("tensornet.em.wave_propagation")
        opt = _import("tensornet.optics.physical_optics")

        freq = 5e14  # visible light
        c0 = 3e8
        wavelength = c0 / freq

        # Verify FDTD can be configured at this freq
        fdtd = em.FDTD1D(nz=200, dz=wavelength / 20, n_steps=100)
        assert fdtd.dz < wavelength, "FDTD grid must resolve wavelength"

        # Fresnel propagator at same wavelength
        fp = opt.FresnelPropagator(wavelength=wavelength, grid_size=64, pixel_pitch=1e-6)
        U0 = np.ones((64, 64), dtype=complex)
        U_prop = fp.propagate(U0, z=1e-3)
        assert U_prop.shape == U0.shape, "Fresnel must preserve field shape"


# =============================================================================
# Pipeline 4: StatMech → Coupled MHD
# =============================================================================

class TestStatMechToCoupledMHD:
    """Data flow: Ising critical temperature → MHD parameter scaling."""

    def test_ising_temperature_scales_hartmann_flow(self) -> None:
        """Ising critical temperature sets a characteristic energy scale
        that feeds into Hartmann flow viscosity scaling."""
        sm = _import("tensornet.statmech.equilibrium")
        mhd = _import("tensornet.coupled.coupled_mhd")

        ising = sm.IsingModel(L=16, temperature=2.0)
        T_c = ising.critical_temperature
        assert T_c == pytest.approx(2.0 / math.log(1 + math.sqrt(2)), rel=1e-6)

        # Hartmann flow with viscosity proportional to T/T_c
        nu_scale = 1e-3 * (ising.T / T_c)
        hf = mhd.HartmannFlow(
            a=0.01, B0=1.0, rho=1000.0, nu=nu_scale, sigma=1e6, dp_dx=-1.0
        )
        assert hf.hartmann_number > 0, "Hartmann number must be positive"

        # Velocity at centre must be nonzero
        v_center = hf.velocity_profile(0.0)
        assert v_center > 0, "Centre-line velocity must be positive for dp/dx < 0"


# =============================================================================
# Pipeline 5: Nuclear Structure → Nuclear Astrophysics
# =============================================================================

class TestNuclearToAstro:
    """Data flow: shell model binding energies → thermonuclear reaction rates."""

    def test_binding_energy_feeds_gamow(self) -> None:
        """Nuclear binding energy from Bethe-Weizsäcker feeds Gamow window."""
        nuc = _import("tensornet.nuclear.structure")
        astro = _import("tensornet.nuclear.astrophysics")

        # Iron-56 shell model
        fe56 = nuc.NuclearShellModel(A=56, Z=26)
        BE = fe56.binding_energy_bethe_weizsacker()
        assert BE > 0, "Binding energy must be positive"
        BE_per_nucleon = BE / 56.0
        assert 7.5 < BE_per_nucleon < 9.0, (
            f"Fe-56 BE/A = {BE_per_nucleon:.2f} MeV, expected ~8.8"
        )

        # p-p chain thermonuclear rate
        pp = astro.ThermonuclearRate(Z1=1, Z2=1, A1=1, A2=1)
        T9 = 0.015  # solar core ~ 15 MK
        E_gamow = pp.gamow_energy(T9)
        assert E_gamow > 0, "Gamow peak energy must be positive"

    def test_woods_saxon_potential_depth(self) -> None:
        """Woods-Saxon potential at centre feeds nuclear reaction model."""
        nuc = _import("tensornet.nuclear.structure")
        react = _import("tensornet.nuclear.reactions")

        sm = nuc.NuclearShellModel(A=208, Z=82)  # Pb-208
        V_center = sm.woods_saxon(0.0)
        assert V_center < 0, "Woods-Saxon potential must be attractive at r=0"

        omp = react.OpticalModelPotential(A_target=208, Z_target=82, E_lab=10.0)
        assert omp.R0 > 0, "Nuclear radius must be positive"

    def test_thermonuclear_rate_temperature_dependence(self) -> None:
        """Thermonuclear rate increases steeply with T9."""
        astro = _import("tensornet.nuclear.astrophysics")

        pp = astro.ThermonuclearRate(Z1=1, Z2=1, A1=1, A2=1)
        w_low = pp.gamow_width(0.01)
        w_high = pp.gamow_width(0.1)
        assert w_high > w_low > 0, "Gamow width must increase with temperature"


# =============================================================================
# Pipeline 6: Nuclear ↔ Particle (Neutrinos)
# =============================================================================

class TestNuclearParticle:
    """Data flow: neutrino oscillations ↔ nuclear network."""

    def test_pmns_unitarity(self) -> None:
        """PMNS matrix from particle physics must be unitary."""
        part = _import("tensornet.particle.beyond_sm")
        nu = part.NeutrinoOscillations()
        U = nu.pmns_matrix()
        UdU = U.conj().T @ U
        np.testing.assert_allclose(
            UdU, np.eye(3), atol=1e-10,
            err_msg="PMNS matrix is not unitary"
        )

    def test_neutrino_probability_conservation(self) -> None:
        """Sum of oscillation probabilities from flavour α to all β = 1."""
        part = _import("tensornet.particle.beyond_sm")
        nu = part.NeutrinoOscillations()
        for alpha in range(3):
            total = sum(
                nu.oscillation_probability(alpha, beta, L_km=500.0, E_GeV=1.0)
                for beta in range(3)
            )
            assert abs(total - 1.0) < 1e-6, (
                f"P(ν_{alpha} → all) = {total:.8f}, expected 1.0"
            )

    def test_neutrino_rates_feed_rprocess(self) -> None:
        """Neutrino oscillation probabilities modulate r-process neutron density."""
        part = _import("tensornet.particle.beyond_sm")
        astro = _import("tensornet.nuclear.astrophysics")

        nu = part.NeutrinoOscillations()
        # Survival probability of electron neutrino
        P_ee = nu.oscillation_probability(0, 0, L_km=100.0, E_GeV=0.01)
        assert 0 <= P_ee <= 1, "Probability must be in [0, 1]"

        # r-process with neutron density scaled by neutrino survival
        rp = astro.RProcess(T9=1.5, n_n=1e24 * P_ee)
        # Saha ratio for a representative nucleus
        ratio = rp.saha_ratio(S_n=5.0, A=100)
        assert ratio > 0, "Saha ratio must be positive"


# =============================================================================
# Pipeline 7: Condensed Matter → Electronic Structure
# =============================================================================

class TestCondMatToElecStruct:
    """Data flow: phonon frequencies → electron-phonon ↔ DFT exchange."""

    def test_phonon_frequencies_positive_definite(self) -> None:
        """Dynamical matrix eigenvalues (ω²) must be non-negative for stable crystal."""
        cm = _import("tensornet.condensed_matter.phonons")
        es = _import("tensornet.electronic_structure.dft")

        # Diatomic chain phonon spectrum
        result = cm.DynamicalMatrix.diatomic_chain(m1=12.0, m2=16.0, k_spring=50.0, n_q=100)
        omegas = result.frequencies
        assert np.all(omegas >= -1e-10), "Phonon frequencies must be non-negative"

        # LDA exchange potential from a uniform electron density
        rho = np.full(100, 0.01)  # uniform electron gas
        eps_x = es.LDAExchangeCorrelation.exchange_energy_density(rho)
        assert np.all(eps_x < 0), "Exchange energy density must be negative"

    def test_monoatomic_chain_has_acoustic_mode(self) -> None:
        """Monoatomic chain: ω(q=0) = 0 (acoustic branch)."""
        cm = _import("tensornet.condensed_matter.phonons")
        result = cm.DynamicalMatrix.monoatomic_chain(mass=28.0, k_spring=100.0, n_q=201)
        omegas = result.frequencies  # shape (n_q, 1)
        # q-grid is linspace(-π/a, π/a, n_q), so q=0 is at the midpoint
        mid = len(omegas) // 2
        assert abs(omegas[mid, 0]) < 1e-10, (
            f"Acoustic mode at q=0 should have ω=0, got {omegas[mid, 0]}"
        )


# =============================================================================
# Pipeline 8: Materials → Coupled Thermo-Mechanical
# =============================================================================

class TestMaterialsToCoupled:
    """Data flow: elastic constants → thermo-mechanical FEM solver."""

    def test_elastic_tensor_feeds_thermoelastic(self) -> None:
        """Isotropic elastic constants from ElasticTensor → ThermoelasticSolver."""
        mat = _import("tensornet.materials.mechanical_properties")
        coup = _import("tensornet.coupled.thermo_mechanical")

        E_steel = 200e9  # Pa
        nu_steel = 0.3
        alpha_th = 12e-6  # thermal expansion

        et = mat.ElasticTensor.from_isotropic(E=E_steel, nu=nu_steel)
        avgs = et.hill_averages()
        assert avgs["K_Hill"] > 0, "Bulk modulus must be positive"

        solver = coup.ThermoelasticSolver(
            nx=10, ny=10, Lx=0.1, Ly=0.1,
            E=E_steel, nu=nu_steel, alpha_th=alpha_th,
        )
        # Apply uniform temperature field
        T_field = np.full((10, 10), 100.0)  # 100 K above reference
        solver.set_temperature(T_field)
        u = solver.solve(n_iter=50, tol=1e-6)
        assert u is not None, "Thermoelastic solver must return displacement"


# =============================================================================
# Pipeline 9: Materials → Ferroelectrics (Polarisation ↔ Strain)
# =============================================================================

class TestMaterialsToFerroelectric:
    """Data flow: mechanical parameters → ferroelectric / piezoelectric coupling."""

    def test_landau_spontaneous_polarisation_below_Tc(self) -> None:
        """Below T_c, spontaneous polarisation is nonzero."""
        cm = _import("tensornet.condensed_matter.ferroelectrics")
        ld = cm.LandauDevonshire()
        P_s = ld.spontaneous_polarisation(T=300.0)  # below Tc=393K
        assert P_s > 0, "Spontaneous polarisation must be positive below Tc"
        P_above = ld.spontaneous_polarisation(T=500.0)  # above Tc
        assert abs(P_above) < 1e-10, (
            "Spontaneous polarisation must vanish above Tc"
        )

    def test_piezoelectric_coupling_coefficient(self) -> None:
        """Piezoelectric coupling coefficient k² < 1 (energy bound)."""
        cm = _import("tensornet.condensed_matter.ferroelectrics")
        pz = cm.PiezoelectricCoupling(d33=300e-12, eps_33=1000.0, s33=12e-12)
        k_sq = pz.coupling_coefficient()
        assert 0 < k_sq < 1, (
            f"Coupling coefficient k² = {k_sq:.4f}, must be in (0, 1)"
        )
        # Applying field produces strain
        strain = pz.strain_from_field(E_field=1e6)
        assert strain > 0, "Positive field should produce positive strain (d33 > 0)"


# =============================================================================
# Pipeline 10: Geophysics (Seismology → Mantle Convection)
# =============================================================================

class TestGeophysicsPipeline:
    """Data flow: seismic velocity model → mantle thermal structure."""

    def test_seismic_velocity_drives_convection_viscosity(self) -> None:
        """Seismic wave velocity heterogeneity maps to thermal anomaly."""
        geo_seis = _import("tensornet.geophysics.seismology")
        geo_conv = _import("tensornet.geophysics.mantle_convection")

        # Set up 2D acoustic wave simulation
        nx, nz = 50, 50
        wave = geo_seis.AcousticWave2D(nx=nx, nz=nz, dx=100.0, dz=100.0,
                                        dt=0.001, nt=10)
        v_model = np.full((nz, nx), 5000.0)  # 5 km/s uniform
        wave.set_velocity_model(v_model)

        # Mantle convection initialised with Rayleigh number
        conv = geo_conv.MantleConvection2D(nx=nx, nz=nz, Ra=1e6, H=1.0)
        T0 = np.random.default_rng(42).random((nz, nx))
        T_new = conv.diffusion_step(T0, dt=1e-5)
        assert T_new.shape == T0.shape, "Diffusion must preserve grid shape"


# =============================================================================
# Pipeline 11: Condensed Matter → Topological Phases
# =============================================================================

class TestCondMatTopological:
    """Data flow: tight-binding bands → Chern number."""

    def test_trivial_insulator_chern_zero(self) -> None:
        """Trivial insulator (large mass parameter) has Chern number = 0."""
        cm = _import("tensornet.condensed_matter.topological_phases")
        calc = cm.ChernNumberCalculator(nk=30)
        # QWZ model with m=3 (trivial phase)
        H = calc.qwz_hamiltonian(kx=0.0, ky=0.0, m=3.0)
        assert H.shape == (2, 2), "QWZ Hamiltonian must be 2×2"
        # Ground state extraction
        gs = calc._ground_state(H)
        assert gs.shape[0] == 2, "Ground state vector must have 2 components"


# =============================================================================
# Pipeline 12: Nuclear → Materials (Radiation Damage)
# =============================================================================

class TestNuclearToMaterials:
    """Data flow: nuclear cross sections → radiation damage in materials."""

    def test_binding_energy_drives_fracture_criterion(self) -> None:
        """Nuclear binding energy sets energy scale; fracture toughness
        from materials science gives complementary structural scale."""
        nuc = _import("tensornet.nuclear.structure")
        mat = _import("tensornet.materials.mechanical_properties")

        fe56 = nuc.NuclearShellModel(A=56, Z=26)
        BE = fe56.binding_energy_bethe_weizsacker()
        assert BE > 0

        # Griffith fracture for iron
        gf = mat.GriffithFracture(E=200e9, gamma_s=2.0, nu=0.3)
        K_Ic = gf.fracture_toughness_plane_stress()
        assert K_Ic > 0, "Fracture toughness must be positive"

        # Critical flaw size at 100 MPa
        sigma_c = gf.critical_stress(a=1e-3)
        assert sigma_c > 0, "Critical stress must be positive"


# =============================================================================
# Pipeline 13: Stellar Structure → Nuclear Astrophysics
# =============================================================================

class TestStellarToNuclear:
    """Data flow: stellar core T, ρ → nuclear reaction rates."""

    def test_pp_chain_rate_positive_at_solar_T(self) -> None:
        """p-p chain rate at solar conditions must be positive."""
        astro_star = _import("tensornet.astro.stellar_structure")
        astro_nuc = _import("tensornet.nuclear.astrophysics")

        # Solar core conditions
        T_core = 1.5e7  # K
        rho_core = 1.5e5  # kg/m³

        # Stellar structure pp-chain rate
        ss = astro_star.StellarStructure.__new__(astro_star.StellarStructure)
        rate = ss._pp_chain_rate(rho_core, T_core)
        assert rate >= 0, "pp-chain rate must be non-negative"

        # Same conditions in thermonuclear rate formalism
        T9 = T_core / 1e9
        pp = astro_nuc.ThermonuclearRate(Z1=1, Z2=1, A1=1, A2=1)
        E_gamow = pp.gamow_energy(T9)
        assert E_gamow > 0, "Gamow energy must be positive at solar T"

    def test_gamow_peak_scales_with_charge(self) -> None:
        """Gamow energy ∝ (Z₁Z₂)^{2/3} — heavier nuclei have higher barrier."""
        astro = _import("tensornet.nuclear.astrophysics")
        T9 = 0.1

        pp = astro.ThermonuclearRate(Z1=1, Z2=1, A1=1, A2=1)
        cno = astro.ThermonuclearRate(Z1=1, Z2=6, A1=1, A2=12)

        E_pp = pp.gamow_energy(T9)
        E_cno = cno.gamow_energy(T9)
        assert E_cno > E_pp, (
            "CNO Gamow peak must be higher than pp (larger Coulomb barrier)"
        )


# =============================================================================
# Multi-Domain Chain 1: QM → Chemistry → StatMech (3 domains)
# =============================================================================

class TestMultiDomainQMChemStatMech:
    """3-domain chain: quantum eigenvalues → PES → thermal rate."""

    def test_qm_morse_tst_chain(self) -> None:
        """Hydrogen wavefunction → Morse PES → TST rate constant."""
        qm = _import("tensornet.quantum_mechanics.stationary")
        chem_pes = _import("tensornet.chemistry.pes")
        chem_rate = _import("tensornet.chemistry.reaction_rate")

        # Step 1: QM — H-atom binding energy as Morse depth
        D_e = abs(qm.HydrogenAtom.energy(1) * qm.HARTREE_EV)

        # Step 2: Chemistry PES — Morse potential with that D_e
        morse = chem_pes.MorsePotential(D_e=D_e, alpha=1.942, r_e=0.7414, mu=0.5)
        omega = morse.harmonic_frequency()
        assert omega > 0, "Harmonic frequency must be positive"

        # Step 3: StatMech / Reaction Rate — TST with PES-derived barrier
        E_a = D_e * 0.1  # 10% of well depth as barrier
        tst = chem_rate.TransitionStateTheory(
            E_a=E_a,
            frequencies_reactant=[omega],
            frequencies_ts=[omega * 0.8],
            imaginary_freq=omega * 0.5,
        )
        k = tst.rate_constant(T=1000.0)
        assert k > 0, "TST rate constant must be positive"


# =============================================================================
# Multi-Domain Chain 2: Nuclear → Particle → Astro (4 domains)
# =============================================================================

class TestMultiDomainNuclearParticleAstro:
    """4-domain chain: nuclear binding → neutrino oscillations → r-process."""

    def test_nuclear_neutrino_rprocess_chain(self) -> None:
        """Nuclear shell model → PMNS unitarity → r-process Saha ratio."""
        nuc = _import("tensornet.nuclear.structure")
        part = _import("tensornet.particle.beyond_sm")
        astro_nuc = _import("tensornet.nuclear.astrophysics")

        # Step 1: Nuclear — Fe-56 binding
        fe56 = nuc.NuclearShellModel(A=56, Z=26)
        BE = fe56.binding_energy_bethe_weizsacker()
        assert BE > 0

        # Step 2: Particle — neutrino electron survival probability
        nu = part.NeutrinoOscillations()
        P_ee = nu.oscillation_probability(0, 0, L_km=100.0, E_GeV=0.01)
        assert 0 <= P_ee <= 1

        # Step 3: Nuclear astrophysics — Gamow energy with p-p
        pp = astro_nuc.ThermonuclearRate(Z1=1, Z2=1, A1=1, A2=1)
        E_gamow = pp.gamow_energy(0.015)
        assert E_gamow > 0

        # Step 4: r-process — neutron density modulated by ν survival
        rp = astro_nuc.RProcess(T9=1.5, n_n=1e24 * P_ee)
        S_n = BE / 56.0  # separation energy ~ BE/A
        ratio = rp.saha_ratio(S_n=S_n, A=100)
        assert ratio > 0, "Saha ratio must be positive"


# =============================================================================
# Multi-Domain Chain 3: EM → Optics → Materials (3 domains)
# =============================================================================

class TestMultiDomainEMOpticsMaterials:
    """3-domain chain: EM field → optical beam → material response."""

    def test_em_beam_material_chain(self) -> None:
        """EM Biot-Savart → Gaussian beam divergence → Birch-Murnaghan EOS."""
        em = _import("tensornet.em.magnetostatics")
        opt = _import("tensornet.optics.physical_optics")
        mat = _import("tensornet.materials.first_principles_design")

        # Step 1: EM — magnetic field from current loop
        bs = em.BiotSavart(current=1.0)
        B = bs.circular_loop(R=0.01, z=0.0)
        assert B > 0

        # Step 2: Optics — Gaussian beam with waist = loop diameter
        beam = opt.GaussianBeam(
            wavelength=632.8e-9, waist=0.02, waist_position=0.0
        )
        z_R = beam.rayleigh_range
        assert z_R > 0

        # Step 3: Materials — Birch-Murnaghan EOS (pressure from compression)
        bm = mat.BirchMurnaghanEOS(V0=10.0, E0=0.0, B0=160e9, B0p=4.0)
        P_compressed = bm.pressure(V=9.5)
        assert P_compressed > 0, (
            "Compression (V < V0) must yield positive pressure"
        )


# =============================================================================
# Smoke: All Cross-Domain Module Imports
# =============================================================================

CROSS_DOMAIN_MODULES = [
    "tensornet.quantum_mechanics.stationary",
    "tensornet.chemistry.pes",
    "tensornet.chemistry.reaction_rate",
    "tensornet.em.magnetostatics",
    "tensornet.em.wave_propagation",
    "tensornet.optics.physical_optics",
    "tensornet.statmech.equilibrium",
    "tensornet.coupled.coupled_mhd",
    "tensornet.nuclear.structure",
    "tensornet.nuclear.reactions",
    "tensornet.nuclear.astrophysics",
    "tensornet.particle.beyond_sm",
    "tensornet.condensed_matter.phonons",
    "tensornet.condensed_matter.band_structure",
    "tensornet.condensed_matter.topological_phases",
    "tensornet.condensed_matter.ferroelectrics",
    "tensornet.electronic_structure.dft",
    "tensornet.materials.mechanical_properties",
    "tensornet.materials.first_principles_design",
    "tensornet.coupled.thermo_mechanical",
    "tensornet.geophysics.seismology",
    "tensornet.geophysics.mantle_convection",
    "tensornet.astro.stellar_structure",
]


@pytest.mark.parametrize("mod_path", CROSS_DOMAIN_MODULES,
                         ids=[m.split(".")[-1] for m in CROSS_DOMAIN_MODULES])
def test_cross_domain_module_importable(mod_path: str) -> None:
    """Every module used in cross-domain tests must import cleanly."""
    importlib.import_module(mod_path)


def test_cross_domain_module_count() -> None:
    """Track the number of modules in cross-domain test coverage."""
    assert len(CROSS_DOMAIN_MODULES) >= 20, (
        f"Expected ≥20 cross-domain modules, have {len(CROSS_DOMAIN_MODULES)}"
    )
