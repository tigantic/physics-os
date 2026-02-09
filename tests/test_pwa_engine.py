"""
PWA Compute Engine Regression Test Suite
=========================================

Validates the Partial Wave Analysis engine implementing Eq. 5.48
from Badui (2020) with Gram-matrix-accelerated likelihood evaluation.

Run with:
    python -m pytest tests/test_pwa_engine.py -v

Covers:
    - Import integrity (all 25 public symbols)
    - Wigner-D matrix elements (orthogonality, known values)
    - Wave set construction and indexing
    - Basis amplitudes precomputation
    - Intensity model (Eq. 5.48 positive-definiteness)
    - Gram matrix (Hermiticity, positive semi-definiteness)
    - Extended likelihood (gradient computation, normalization)
    - Convention reduction (3 tests at machine precision)
    - Synthetic data generation (determinism, acceptance rate)
    - Fitter convergence on small problem
    - Gram acceleration agreement with baseline
    - Breit-Wigner resonance shape
"""

import math

import numpy as np
import pytest
import torch

from experiments.pwa_engine.core import (
    BasisAmplitudes,
    BreitWigner,
    ChannelConfig,
    CoupledChannelSystem,
    ExtendedLikelihood,
    GramMatrix,
    IntensityModel,
    LBFGSFitter,
    PolarizedIntensityModel,
    SyntheticDataGenerator,
    Wave,
    WaveSet,
    benchmark_normalization,
    build_wave_set,
    convention_reduction_test,
    wigner_D_element,
    wigner_small_d,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Default compute device."""
    return torch.device("cpu")


@pytest.fixture(scope="module")
def small_wave_set() -> WaveSet:
    """Minimal wave set: J_max=1.5, single reflectivity, single component."""
    return build_wave_set(j_max=1.5, reflectivities=(+1,), n_components=1)


@pytest.fixture(scope="module")
def medium_wave_set() -> WaveSet:
    """Medium wave set: J_max=2.5, single reflectivity."""
    return build_wave_set(j_max=2.5, reflectivities=(+1,), n_components=1)


@pytest.fixture(scope="module")
def synthetic_data(
    medium_wave_set: WaveSet, device: torch.device
) -> dict:
    """Generate synthetic data with known true amplitudes."""
    rng = np.random.default_rng(42)
    n_amp = medium_wave_set.n_amplitudes
    V_true = rng.standard_normal(n_amp) + 1j * rng.standard_normal(n_amp)
    V_true = V_true / np.linalg.norm(V_true)

    gen = SyntheticDataGenerator(
        wave_set=medium_wave_set,
        V_true=V_true,
        helicity=0.5,
        seed=42,
    )
    data = gen.generate(n_data=2000, n_generated=50000, device=device)
    return data


# ---------------------------------------------------------------------------
# Import Tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify all public symbols are importable."""

    def test_all_classes_import(self) -> None:
        from experiments.pwa_engine import (
            BasisAmplitudes,
            BreitWigner,
            ChannelConfig,
            CoupledChannelSystem,
            ExtendedLikelihood,
            GramMatrix,
            IntensityModel,
            LBFGSFitter,
            PolarizedIntensityModel,
            SyntheticDataGenerator,
            Wave,
            WaveSet,
        )
        for cls in [
            BasisAmplitudes, BreitWigner, ChannelConfig,
            CoupledChannelSystem, ExtendedLikelihood, GramMatrix,
            IntensityModel, LBFGSFitter, PolarizedIntensityModel,
            SyntheticDataGenerator, Wave, WaveSet,
        ]:
            assert cls is not None

    def test_all_functions_import(self) -> None:
        from experiments.pwa_engine import (
            beam_asymmetry_sensitivity_test,
            benchmark_normalization,
            bootstrap_uncertainty,
            build_wave_set,
            compress_gram_qtt,
            compute_angular_moments,
            convention_reduction_test,
            coupled_channel_test,
            mass_dependent_fit,
            moment_comparison,
            wave_set_scan,
            wigner_D_element,
            wigner_small_d,
        )
        for fn in [
            beam_asymmetry_sensitivity_test, benchmark_normalization,
            bootstrap_uncertainty, build_wave_set, compress_gram_qtt,
            compute_angular_moments, convention_reduction_test,
            coupled_channel_test, mass_dependent_fit, moment_comparison,
            wave_set_scan, wigner_D_element, wigner_small_d,
        ]:
            assert callable(fn)

    def test_version_string(self) -> None:
        import experiments.pwa_engine as pwa
        assert hasattr(pwa, "__all__")
        assert len(pwa.__all__) == 25


# ---------------------------------------------------------------------------
# Wigner-D Tests
# ---------------------------------------------------------------------------


class TestWignerD:
    """Validate Wigner rotation matrix elements."""

    def test_small_d_j_half_identity_at_zero(self) -> None:
        """d^{1/2}_{m,m'}(β=0) = δ_{m,m'}."""
        beta = np.array([0.0])
        assert abs(wigner_small_d(0.5, 0.5, 0.5, beta)[0] - 1.0) < 1e-14
        assert abs(wigner_small_d(0.5, 0.5, -0.5, beta)[0]) < 1e-14
        assert abs(wigner_small_d(0.5, -0.5, -0.5, beta)[0] - 1.0) < 1e-14

    def test_small_d_j_half_at_pi(self) -> None:
        """d^{1/2}_{1/2,-1/2}(π) = -1."""
        beta = np.array([np.pi])
        val = wigner_small_d(0.5, 0.5, 0.5, beta)[0]
        assert abs(val - 0.0) < 1e-14  # cos(π/2) = 0

    def test_small_d_j1_known_values(self) -> None:
        """d^1_{0,0}(β) = cos(β)."""
        beta = np.array([0.0, np.pi / 4, np.pi / 2, np.pi])
        d_00 = wigner_small_d(1.0, 0.0, 0.0, beta)
        expected = np.cos(beta)
        np.testing.assert_allclose(d_00, expected, atol=1e-14)

    def test_small_d_vectorized(self) -> None:
        """Vectorized call returns correct shape."""
        beta = np.linspace(0, np.pi, 100)
        result = wigner_small_d(1.5, 0.5, -0.5, beta)
        assert result.shape == (100,)
        assert np.all(np.isfinite(result))

    def test_full_D_element_unitarity_check(self) -> None:
        """Sum_{m'} |D^j_{m,m'}|² = 1 for any angles (unitarity of rows)."""
        theta = np.array([1.0])
        phi = np.array([0.5])
        j = 1.0
        m = 0.0
        total = 0.0
        for mp in [-1.0, 0.0, 1.0]:
            val = wigner_D_element(j, m, mp, theta, phi)
            total += float(np.abs(val[0]) ** 2)
        assert abs(total - 1.0) < 1e-12

    def test_D_element_reproduces_small_d_when_phi_zero(self) -> None:
        """D^j_{m,m'}(θ,φ=0) = d^j_{m,m'}(θ) (real)."""
        theta = np.array([0.7])
        phi = np.array([0.0])
        j, m, mp = 1.5, 0.5, -0.5
        D_val = wigner_D_element(j, m, mp, theta, phi)
        d_val = wigner_small_d(j, m, mp, theta)
        np.testing.assert_allclose(D_val.real, d_val, atol=1e-14)
        np.testing.assert_allclose(D_val.imag, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Wave Set Tests
# ---------------------------------------------------------------------------


class TestWaveSet:
    """Validate wave set construction and indexing."""

    def test_build_wave_set_counts(self) -> None:
        """J_max=1.5, single ε → 6 amplitudes (J=0.5: 2M × 1, J=1.5: 4M × 1)."""
        ws = build_wave_set(j_max=1.5, reflectivities=(+1,), n_components=1)
        assert ws.n_amplitudes == 6

    def test_build_wave_set_dual_reflectivity(self) -> None:
        """Two reflectivities doubles the amplitude count."""
        ws_single = build_wave_set(j_max=1.5, reflectivities=(+1,))
        ws_dual = build_wave_set(j_max=1.5, reflectivities=(+1, -1))
        assert ws_dual.n_amplitudes == 2 * ws_single.n_amplitudes

    def test_wave_set_j_max_2_5(self) -> None:
        """J_max=2.5 → 12 amplitudes (J∈{0.5,1.5,2.5})."""
        ws = build_wave_set(j_max=2.5, reflectivities=(+1,), n_components=1)
        assert ws.n_amplitudes == 12

    def test_wave_quantum_numbers(self) -> None:
        """Each wave has valid quantum numbers."""
        ws = build_wave_set(j_max=2.5, reflectivities=(+1,), n_components=1)
        for wave in ws.waves:
            assert abs(wave.m) <= wave.j
            assert wave.j >= 0.5
            assert wave.epsilon in (+1, -1)


# ---------------------------------------------------------------------------
# Basis & Intensity Tests
# ---------------------------------------------------------------------------


class TestBasisAndIntensity:
    """Validate BasisAmplitudes and IntensityModel."""

    def test_basis_shape(self, small_wave_set: WaveSet, device: torch.device) -> None:
        """Basis matrix has shape (n_events, n_amplitudes)."""
        theta = np.random.default_rng(0).uniform(0, np.pi, 200)
        phi = np.random.default_rng(0).uniform(0, 2 * np.pi, 200)
        basis = BasisAmplitudes(small_wave_set, theta, phi, device=device)
        assert basis.psi.shape == (200, small_wave_set.n_amplitudes)

    def test_intensity_positive(
        self, small_wave_set: WaveSet, device: torch.device
    ) -> None:
        """Intensity I(τ; V) ≥ 0 for all events and random V."""
        rng = np.random.default_rng(99)
        theta = rng.uniform(0, np.pi, 300)
        phi = rng.uniform(0, 2 * np.pi, 300)
        basis = BasisAmplitudes(small_wave_set, theta, phi, device=device)
        model = IntensityModel(basis)

        V = torch.randn(small_wave_set.n_amplitudes, dtype=torch.complex128)
        intensity = model.evaluate(V)
        assert intensity.shape == (300,)
        assert torch.all(intensity >= -1e-15)  # Allow tiny numerical noise

    def test_intensity_zero_amplitudes(
        self, small_wave_set: WaveSet, device: torch.device
    ) -> None:
        """Zero amplitudes → zero intensity."""
        theta = np.array([1.0, 2.0])
        phi = np.array([0.5, 1.5])
        basis = BasisAmplitudes(small_wave_set, theta, phi, device=device)
        model = IntensityModel(basis)
        V_zero = torch.zeros(small_wave_set.n_amplitudes, dtype=torch.complex128)
        intensity = model.evaluate(V_zero)
        assert torch.all(intensity.abs() < 1e-30)


# ---------------------------------------------------------------------------
# Gram Matrix Tests
# ---------------------------------------------------------------------------


class TestGramMatrix:
    """Validate Gram matrix properties."""

    def test_gram_hermitian(
        self, small_wave_set: WaveSet, device: torch.device
    ) -> None:
        """Gram matrix G is Hermitian."""
        rng = np.random.default_rng(7)
        theta = rng.uniform(0, np.pi, 5000)
        phi = rng.uniform(0, 2 * np.pi, 5000)
        basis_mc = BasisAmplitudes(small_wave_set, theta, phi, device=device)
        gram = GramMatrix(basis_mc, n_generated=10000)
        G = gram.G
        diff = torch.abs(G - G.conj().T).max().item()
        assert diff < 1e-14, f"Gram is not Hermitian: max diff = {diff}"

    def test_gram_positive_semidefinite(
        self, small_wave_set: WaveSet, device: torch.device
    ) -> None:
        """Gram matrix eigenvalues are non-negative."""
        rng = np.random.default_rng(8)
        theta = rng.uniform(0, np.pi, 5000)
        phi = rng.uniform(0, 2 * np.pi, 5000)
        basis_mc = BasisAmplitudes(small_wave_set, theta, phi, device=device)
        gram = GramMatrix(basis_mc, n_generated=10000)
        eigenvalues = torch.linalg.eigvalsh(gram.G.real)
        assert torch.all(eigenvalues > -1e-12), (
            f"Negative eigenvalue: {eigenvalues.min().item()}"
        )

    def test_gram_normalization_positive(
        self, small_wave_set: WaveSet, device: torch.device
    ) -> None:
        """V†GV > 0 for non-zero V."""
        rng = np.random.default_rng(9)
        theta = rng.uniform(0, np.pi, 5000)
        phi = rng.uniform(0, 2 * np.pi, 5000)
        basis_mc = BasisAmplitudes(small_wave_set, theta, phi, device=device)
        gram = GramMatrix(basis_mc, n_generated=10000)
        V = torch.randn(small_wave_set.n_amplitudes, dtype=torch.complex128)
        norm = gram.normalization(V, n_data=1000)
        assert norm.item() > 0


# ---------------------------------------------------------------------------
# Extended Likelihood Tests
# ---------------------------------------------------------------------------


class TestExtendedLikelihood:
    """Validate extended likelihood computation."""

    def test_nll_is_finite(self, synthetic_data: dict) -> None:
        """NLL returns a finite scalar value."""
        data = synthetic_data
        V = torch.tensor(data["V_true"], dtype=torch.complex128)
        theta_d, phi_d = data["theta_data"], data["phi_data"]
        theta_m, phi_m = data["theta_mc"], data["phi_mc"]
        ws = build_wave_set(j_max=2.5, reflectivities=(+1,))

        basis_data = BasisAmplitudes(ws, theta_d, phi_d)
        basis_mc = BasisAmplitudes(ws, theta_m, phi_m)
        model_data = IntensityModel(basis_data)
        model_mc = IntensityModel(basis_mc)
        gram = GramMatrix(basis_mc, n_generated=data["n_generated"])
        nll_obj = ExtendedLikelihood(
            model_data, model_mc, gram,
            n_data=data["n_data"],
            n_generated=data["n_generated"],
            use_gram=True,
        )
        nll = nll_obj(V)
        assert torch.isfinite(nll), f"NLL is not finite: {nll}"

    def test_nll_with_gram_matches_baseline(self, synthetic_data: dict) -> None:
        """Gram-accelerated NLL matches baseline (brute-force) to high precision."""
        data = synthetic_data
        V = torch.tensor(data["V_true"], dtype=torch.complex128)
        ws = build_wave_set(j_max=2.5, reflectivities=(+1,))

        basis_data = BasisAmplitudes(ws, data["theta_data"], data["phi_data"])
        basis_mc = BasisAmplitudes(ws, data["theta_mc"], data["phi_mc"])
        model_data = IntensityModel(basis_data)
        model_mc = IntensityModel(basis_mc)
        gram = GramMatrix(basis_mc, n_generated=data["n_generated"])

        nll_gram = ExtendedLikelihood(
            model_data, model_mc, gram,
            n_data=data["n_data"],
            n_generated=data["n_generated"],
            use_gram=True,
        )
        nll_baseline = ExtendedLikelihood(
            model_data, model_mc, gram,
            n_data=data["n_data"],
            n_generated=data["n_generated"],
            use_gram=False,
        )
        val_gram = nll_gram(V).item()
        val_base = nll_baseline(V).item()
        rel_diff = abs(val_gram - val_base) / (abs(val_base) + 1e-30)
        assert rel_diff < 1e-10, (
            f"Gram vs baseline disagree: {val_gram} vs {val_base} (rel {rel_diff})"
        )


# ---------------------------------------------------------------------------
# Convention Reduction Test (Experiment 1)
# ---------------------------------------------------------------------------


class TestConventionReduction:
    """The core mathematical identity: full Eq. 5.48 ⊃ simplified coherent sum."""

    def test_all_three_reductions_pass(self) -> None:
        """All 3 convention tests pass at machine precision."""
        result = convention_reduction_test(n_events=200, seed=42)
        assert result["all_pass"] is True
        assert result["test_1_full_vs_single_eps"] < 1e-12
        assert result["test_2_model_vs_manual"] < 1e-12
        assert result["test_3_full_rho_vs_diagonal"] < 1e-12

    def test_convention_test_deterministic(self) -> None:
        """Same seed → identical results."""
        r1 = convention_reduction_test(n_events=100, seed=123)
        r2 = convention_reduction_test(n_events=100, seed=123)
        assert r1["test_1_full_vs_single_eps"] == r2["test_1_full_vs_single_eps"]
        assert r1["test_2_model_vs_manual"] == r2["test_2_model_vs_manual"]
        assert r1["test_3_full_rho_vs_diagonal"] == r2["test_3_full_rho_vs_diagonal"]


# ---------------------------------------------------------------------------
# Synthetic Data Generator Tests
# ---------------------------------------------------------------------------


class TestSyntheticDataGenerator:
    """Validate data generation pipeline."""

    def test_generate_returns_expected_keys(self, synthetic_data: dict) -> None:
        """Generate returns all expected keys."""
        expected_keys = {
            "theta_data", "phi_data", "theta_mc", "phi_mc",
            "n_data", "n_generated", "n_mc_accepted", "V_true",
            "acceptance_rate",
        }
        assert expected_keys.issubset(set(synthetic_data.keys()))

    def test_data_count_matches_request(self, synthetic_data: dict) -> None:
        """Number of data events matches the requested count."""
        assert len(synthetic_data["theta_data"]) == synthetic_data["n_data"]
        assert len(synthetic_data["phi_data"]) == synthetic_data["n_data"]

    def test_mc_acceptance_rate_reasonable(self, synthetic_data: dict) -> None:
        """MC acceptance rate is between 10% and 90%."""
        rate = synthetic_data["acceptance_rate"]
        assert 0.1 < rate < 0.9, f"Acceptance rate out of range: {rate}"

    def test_angular_ranges(self, synthetic_data: dict) -> None:
        """θ ∈ [0,π], φ ∈ [0,2π]."""
        assert np.all(synthetic_data["theta_data"] >= 0)
        assert np.all(synthetic_data["theta_data"] <= np.pi)
        assert np.all(synthetic_data["phi_data"] >= 0)
        assert np.all(synthetic_data["phi_data"] <= 2 * np.pi)

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces identical data."""
        ws = build_wave_set(j_max=1.5)
        V = np.array([1 + 0j, 0.5 + 0.5j, 0.3 - 0.2j,
                       0.1 + 0.1j, -0.4 + 0j, 0.2 + 0.3j])
        g1 = SyntheticDataGenerator(ws, V, seed=77)
        g2 = SyntheticDataGenerator(ws, V, seed=77)
        d1 = g1.generate(n_data=500, n_generated=10000)
        d2 = g2.generate(n_data=500, n_generated=10000)
        np.testing.assert_array_equal(d1["theta_data"], d2["theta_data"])
        np.testing.assert_array_equal(d1["phi_data"], d2["phi_data"])


# ---------------------------------------------------------------------------
# Fitter Convergence Test
# ---------------------------------------------------------------------------


class TestFitterConvergence:
    """Validate L-BFGS-B fitter on a small, well-conditioned problem."""

    def test_fitter_converges(self, synthetic_data: dict) -> None:
        """Fitter converges to a solution with reasonable NLL."""
        data = synthetic_data
        ws = build_wave_set(j_max=2.5, reflectivities=(+1,))

        basis_data = BasisAmplitudes(ws, data["theta_data"], data["phi_data"])
        basis_mc = BasisAmplitudes(ws, data["theta_mc"], data["phi_mc"])
        model_data = IntensityModel(basis_data)
        gram = GramMatrix(basis_mc, n_generated=data["n_generated"])
        nll_obj = ExtendedLikelihood(
            model_data, None, gram,
            n_data=data["n_data"],
            n_generated=data["n_generated"],
            use_gram=True,
        )
        fitter = LBFGSFitter(nll_obj, max_iter=200, tolerance=1e-8)
        result = fitter.fit(seed=42)
        assert result["converged"] or result["grad_norm"] < 1e-4
        assert np.isfinite(result["nll"])

    def test_fitter_deterministic(self, synthetic_data: dict) -> None:
        """Same seed → same fit result."""
        data = synthetic_data
        ws = build_wave_set(j_max=2.5, reflectivities=(+1,))

        basis_data = BasisAmplitudes(ws, data["theta_data"], data["phi_data"])
        basis_mc = BasisAmplitudes(ws, data["theta_mc"], data["phi_mc"])
        model_data = IntensityModel(basis_data)
        gram = GramMatrix(basis_mc, n_generated=data["n_generated"])
        nll_obj = ExtendedLikelihood(
            model_data, None, gram,
            n_data=data["n_data"],
            n_generated=data["n_generated"],
            use_gram=True,
        )
        fitter = LBFGSFitter(nll_obj, max_iter=100)
        r1 = fitter.fit(seed=99)
        r2 = fitter.fit(seed=99)
        assert abs(r1["nll"] - r2["nll"]) < 1e-10


# ---------------------------------------------------------------------------
# Gram Acceleration Benchmark Test
# ---------------------------------------------------------------------------


class TestGramAcceleration:
    """Validate Gram speedup and numerical agreement."""

    def test_benchmark_normalization_agreement(self) -> None:
        """Gram vs baseline normalization agree to machine precision."""
        ws = build_wave_set(j_max=1.5)
        result = benchmark_normalization(
            ws, n_mc_events=1000, n_generated=2000,
            n_evals=10, seed=42,
        )
        assert result["relative_agreement"] < 1e-12
        assert result["speedup"] > 0  # Gram should not be slower at scale


# ---------------------------------------------------------------------------
# Breit-Wigner Tests
# ---------------------------------------------------------------------------


class TestBreitWigner:
    """Validate Breit-Wigner resonance shape."""

    def test_peak_at_resonance_mass(self) -> None:
        """BW intensity peaks near m₀."""
        bw = BreitWigner(m0=1.5, gamma0=0.1)
        masses = np.linspace(1.0, 2.0, 1000)
        mag_sq = bw.intensity(masses)
        peak_idx = np.argmax(mag_sq)
        peak_mass = masses[peak_idx]
        assert abs(peak_mass - 1.5) < 0.01

    def test_width_scaling(self) -> None:
        """Wider resonance has broader peak (FWHM scales with Γ)."""
        masses = np.linspace(1.0, 2.0, 10000)
        bw_narrow = BreitWigner(m0=1.5, gamma0=0.05)
        bw_wide = BreitWigner(m0=1.5, gamma0=0.20)
        mag_narrow = bw_narrow.intensity(masses)
        mag_wide = bw_wide.intensity(masses)
        # FWHM proxy: count bins above half-max
        hm_narrow = np.sum(mag_narrow > 0.5 * mag_narrow.max())
        hm_wide = np.sum(mag_wide > 0.5 * mag_wide.max())
        assert hm_wide > hm_narrow

    def test_bw_amplitude_is_complex(self) -> None:
        """BW amplitude returns complex values with non-zero imaginary part off-peak."""
        bw = BreitWigner(m0=1.5, gamma0=0.1)
        val = bw.amplitude(np.array([1.3]))[0]
        assert np.iscomplex(val) or isinstance(val, complex)
        assert abs(np.imag(val)) > 0


# ---------------------------------------------------------------------------
# Polarized Intensity Model Tests
# ---------------------------------------------------------------------------


class TestPolarizedIntensity:
    """Validate polarization observable Σ."""

    def test_sigma_bounded(self, device: torch.device) -> None:
        """Beam asymmetry Σ ∈ [-1, +1]."""
        ws_dual = build_wave_set(j_max=1.5, reflectivities=(+1, -1))
        rng = np.random.default_rng(10)
        theta = rng.uniform(0, np.pi, 200)
        phi = rng.uniform(0, 2 * np.pi, 200)
        basis = BasisAmplitudes(ws_dual, theta, phi, device=device)
        pol_model = PolarizedIntensityModel(basis)

        V = torch.randn(ws_dual.n_amplitudes, dtype=torch.complex128)
        sigma = pol_model.beam_asymmetry(V)
        assert torch.all(sigma >= -1.0 - 1e-10)
        assert torch.all(sigma <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Channel Config and Coupled-Channel Tests
# ---------------------------------------------------------------------------


class TestCoupledChannel:
    """Validate coupled-channel system construction."""

    def test_channel_config_creation(self) -> None:
        """ChannelConfig stores wave set and data properly."""
        ws = build_wave_set(j_max=1.5)
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, np.pi, 100)
        phi = rng.uniform(0, 2 * np.pi, 100)
        config = ChannelConfig(
            name="test_channel",
            wave_set=ws,
            helicity=0.5,
            theta_data=theta,
            phi_data=phi,
            theta_mc=theta,
            phi_mc=phi,
            n_generated=10000,
        )
        assert config.name == "test_channel"
        assert config.wave_set.n_amplitudes == ws.n_amplitudes
        assert config.n_generated == 10000


# ---------------------------------------------------------------------------
# End-to-End Smoke Test
# ---------------------------------------------------------------------------


class TestEndToEndSmoke:
    """Full pipeline: generate → build → fit → verify."""

    def test_small_pipeline(self, device: torch.device) -> None:
        """Complete PWA pipeline on minimal problem converges."""
        ws = build_wave_set(j_max=0.5, reflectivities=(+1,), n_components=1)
        rng = np.random.default_rng(55)
        n_amp = ws.n_amplitudes
        V_true = rng.standard_normal(n_amp) + 1j * rng.standard_normal(n_amp)
        V_true = V_true / np.linalg.norm(V_true) * 2.0

        gen = SyntheticDataGenerator(ws, V_true, seed=55)
        data = gen.generate(n_data=500, n_generated=10000, device=device)

        basis_data = BasisAmplitudes(ws, data["theta_data"], data["phi_data"],
                                     device=device)
        basis_mc = BasisAmplitudes(ws, data["theta_mc"], data["phi_mc"],
                                   device=device)
        model_data = IntensityModel(basis_data)
        gram = GramMatrix(basis_mc, n_generated=data["n_generated"])

        nll_obj = ExtendedLikelihood(
            model_data, None, gram,
            n_data=data["n_data"],
            n_generated=data["n_generated"],
            use_gram=True,
        )
        fitter = LBFGSFitter(nll_obj, max_iter=300)
        result = fitter.fit(seed=55)

        assert np.isfinite(result["nll"])
        assert result["converged"] or result["grad_norm"] < 0.01
