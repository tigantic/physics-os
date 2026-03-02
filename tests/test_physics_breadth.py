"""Tests for Phase F: UGv2 Physics Breadth.

Validates:
1. Compressible Euler 1D compiler — construction, IR, initial conditions,
   exact Riemann solution, boundedness predicates, conservation balance
2. CHT coupling — coefficient compilation, 1D compiler, energy bookkeeping,
   interface flux, diagnostic sanitizer
3. Phase-field multiphase — Cahn-Hilliard compiler, initial conditions,
   free energy, phase fraction, density/viscosity mixing, sanitizer
4. Evidence — BOUNDEDNESS claim tag
5. IP boundary compliance — no forbidden field leakage

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
import pytest
import numpy as np
from numpy.typing import NDArray


# ══════════════════════════════════════════════════════════════════════
# §1  Compressible Euler 1D — configuration and construction
# ══════════════════════════════════════════════════════════════════════

class TestEulerConfig:
    """Test EulerConfig defaults and construction."""

    def test_defaults(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import EulerConfig
        cfg = EulerConfig()
        assert cfg.gamma == 1.4
        assert cfg.cfl == 0.3
        assert cfg.artificial_viscosity == 0.01

    def test_custom(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import EulerConfig
        cfg = EulerConfig(gamma=1.6667, cfl=0.1, artificial_viscosity=0.0)
        assert cfg.gamma == pytest.approx(1.6667)
        assert cfg.artificial_viscosity == 0.0


class TestCompressibleEulerCompiler:
    """Test CompressibleEuler1DCompiler construction and compilation."""

    def test_default_construction(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            CompressibleEuler1DCompiler,
        )
        c = CompressibleEuler1DCompiler()
        assert c.domain == "compressible_euler_1d"
        assert c.domain_label == "1D Compressible Euler Equations"

    def test_compile_sod(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            CompressibleEuler1DCompiler,
        )
        c = CompressibleEuler1DCompiler(n_bits=8, n_steps=50, ic_type="sod")
        prog = c.compile()
        assert prog.domain == "compressible_euler_1d"
        assert prog.n_registers == 13
        assert "rho" in prog.fields
        assert "rho_u" in prog.fields
        assert "E" in prog.fields
        assert "inv_rho" in prog.fields

    def test_compile_smooth_sine(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            CompressibleEuler1DCompiler,
        )
        from ontic.engine.vm.ir import BCKind
        c = CompressibleEuler1DCompiler(
            n_bits=8, n_steps=50,
            ic_type="smooth_sine",
            bc_kind=BCKind.PERIODIC,
        )
        prog = c.compile()
        assert prog.fields["rho"].bc == BCKind.PERIODIC

    def test_compile_shu_osher(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            CompressibleEuler1DCompiler,
        )
        c = CompressibleEuler1DCompiler(
            n_bits=8, n_steps=50,
            domain_bounds=(-5.0, 5.0),
            ic_type="shu_osher",
        )
        prog = c.compile()
        assert prog.metadata["ic_type"] == "shu_osher"
        assert prog.metadata["gamma"] == 1.4

    def test_invalid_ic_type(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            CompressibleEuler1DCompiler,
        )
        c = CompressibleEuler1DCompiler(ic_type="bogus")
        with pytest.raises(ValueError, match="Unknown IC type"):
            c.compile()

    def test_metadata_contains_boundedness(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            CompressibleEuler1DCompiler,
        )
        c = CompressibleEuler1DCompiler()
        prog = c.compile()
        preds = prog.metadata["boundedness_predicates"]
        assert "rho_positive" in preds
        assert "pressure_positive" in preds

    def test_conserved_quantities(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            CompressibleEuler1DCompiler,
        )
        c = CompressibleEuler1DCompiler()
        prog = c.compile()
        assert prog.fields["rho"].conserved_quantity == "total_mass"
        assert prog.fields["rho_u"].conserved_quantity == "total_momentum"
        assert prog.fields["E"].conserved_quantity == "total_energy"

    def test_artificial_viscosity_instructions(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            CompressibleEuler1DCompiler, EulerConfig,
        )
        from ontic.engine.vm.ir import OpCode
        # With eps > 0
        c1 = CompressibleEuler1DCompiler(n_bits=8, n_steps=10)
        prog1 = c1.compile()
        opcodes1 = [i.opcode for i in prog1.instructions]
        laplace_count = opcodes1.count(OpCode.LAPLACE)
        # Three fields × 1 Laplace each for artificial viscosity
        assert laplace_count >= 3

        # Without eps
        c2 = CompressibleEuler1DCompiler(
            n_bits=8, n_steps=10,
            config=EulerConfig(artificial_viscosity=0.0),
        )
        prog2 = c2.compile()
        opcodes2 = [i.opcode for i in prog2.instructions]
        laplace_count2 = opcodes2.count(OpCode.LAPLACE)
        assert laplace_count2 < laplace_count


# ══════════════════════════════════════════════════════════════════════
# §2  Compressible Euler — Initial conditions
# ══════════════════════════════════════════════════════════════════════

class TestEulerIC:
    """Test initial condition functions."""

    def test_sod_shock_tube(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            sod_shock_tube_ic,
        )
        x = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        rho, rho_u, E = sod_shock_tube_ic(x)
        # Left state: rho=1
        assert rho[0] == 1.0
        assert rho[1] == 1.0
        # Right state: rho=0.125
        assert rho[3] == 0.125
        assert rho[4] == 0.125
        # Zero velocity everywhere
        assert np.all(rho_u == 0.0)
        # Energy matches p / (gamma-1)
        assert E[0] == pytest.approx(1.0 / 0.4)  # p=1, gamma=1.4
        assert E[4] == pytest.approx(0.1 / 0.4)  # p=0.1

    def test_shu_osher(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            shu_osher_ic,
        )
        x = np.array([-5.0, -4.5, -3.0, 0.0, 5.0])
        rho, rho_u, E = shu_osher_ic(x)
        # Left state (x < -4)
        assert rho[0] == pytest.approx(3.857143)
        assert rho[1] == pytest.approx(3.857143)
        # Right state (x >= -4): rho = 1 + 0.2*sin(5x)
        assert rho[3] == pytest.approx(1.0 + 0.2 * np.sin(0.0))

    def test_smooth_sine(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            smooth_sine_ic,
        )
        x = np.linspace(0, 1, 64, endpoint=False)
        rho, rho_u, E = smooth_sine_ic(x)
        # Density should be 1 + 0.2*sin(2πx)
        expected_rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
        np.testing.assert_allclose(rho, expected_rho)
        # Velocity = 1 everywhere → rho_u = rho
        np.testing.assert_allclose(rho_u, rho)


# ══════════════════════════════════════════════════════════════════════
# §3  Compressible Euler — Boundedness predicates
# ══════════════════════════════════════════════════════════════════════

class TestBoundednessPredicates:
    """Test density and pressure positivity checks."""

    def test_density_positive(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            check_density_positive,
        )
        assert check_density_positive(np.array([1.0, 0.5, 0.125]))
        assert not check_density_positive(np.array([1.0, -0.01, 0.5]))

    def test_pressure_positive(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            check_pressure_positive,
        )
        # Normal state: ρ=1, u=0, E = p/(γ-1) with p=1 → E=2.5
        rho = np.array([1.0])
        rho_u = np.array([0.0])
        E = np.array([2.5])
        assert check_pressure_positive(E, rho, rho_u)

    def test_pressure_negative(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            check_pressure_positive,
        )
        # Unphysical: E too small → p < 0
        rho = np.array([1.0])
        rho_u = np.array([10.0])  # large momentum
        E = np.array([0.1])       # small energy
        assert not check_pressure_positive(E, rho, rho_u)

    def test_conservation_balance(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            compute_conservation_balance,
        )
        rho_init = np.array([1.0, 1.0, 1.0, 1.0])
        rho_final = np.array([1.0, 1.0, 1.0, 1.0])
        h = 0.25
        assert compute_conservation_balance(rho_init, rho_final, h) < 1e-14

    def test_conservation_balance_nonzero(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            compute_conservation_balance,
        )
        rho_init = np.array([1.0, 1.0, 1.0, 1.0])
        rho_final = np.array([0.9, 1.0, 1.1, 1.0])
        h = 0.25
        assert compute_conservation_balance(rho_init, rho_final, h) == 0.0


# ══════════════════════════════════════════════════════════════════════
# §4  Compressible Euler — Exact Riemann solution
# ══════════════════════════════════════════════════════════════════════

class TestSodExactSolution:
    """Test exact Riemann solution for Sod shock tube."""

    def test_sod_exact_at_t0(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            sod_exact_solution,
        )
        x = np.linspace(0, 1, 100)
        rho, u, p = sod_exact_solution(x, t=1e-15)
        # At t≈0, should match Sod IC
        np.testing.assert_allclose(
            rho[x < 0.49], 1.0, atol=1e-3,
        )
        np.testing.assert_allclose(
            rho[x > 0.51], 0.125, atol=1e-3,
        )

    def test_sod_exact_at_t02(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            sod_exact_solution,
        )
        x = np.linspace(0, 1, 500)
        rho, u, p = sod_exact_solution(x, t=0.2)
        # Physical constraints
        assert np.all(rho > 0)
        assert np.all(p > 0)
        # Velocity should be zero at far boundaries
        assert abs(u[0]) < 1e-10
        assert abs(u[-1]) < 1e-10

    def test_sod_symmetry(self) -> None:
        """Velocity should be positive in expansion region."""
        from ontic.engine.vm.compilers.compressible_euler import (
            sod_exact_solution,
        )
        x = np.linspace(0, 1, 500)
        _, u, _ = sod_exact_solution(x, t=0.2)
        assert np.max(u) > 0


# ══════════════════════════════════════════════════════════════════════
# §5  CHT Coupling — configuration
# ══════════════════════════════════════════════════════════════════════

class TestCHTConfig:
    """Test CHT material specs and configuration."""

    def test_material_spec(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import CHTMaterialSpec
        mat = CHTMaterialSpec(
            name="copper",
            thermal_conductivity=401.0,
            density=8960.0,
            specific_heat=385.0,
        )
        assert mat.rho_cp == pytest.approx(8960.0 * 385.0)
        assert mat.thermal_diffusivity == pytest.approx(
            401.0 / (8960.0 * 385.0),
        )

    def test_default_config(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import CHTConfig
        cfg = CHTConfig()
        assert cfg.fluid.name == "water"
        assert cfg.solid.name == "aluminum"
        assert cfg.t_initial == 300.0


class TestCHTCoefficients:
    """Test CHT coefficient field compilation."""

    def test_compile_coefficients(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import (
            CHTConfig, compile_cht_coefficients_1d,
        )
        cfg = CHTConfig()
        result = compile_cht_coefficients_1d(
            config=cfg,
            interface_x=0.5,
            bits_per_dim=(8,),
            max_rank=16,
        )
        assert "k_field" in result
        assert "rho_cp_field" in result
        assert "inv_rho_cp_field" in result
        assert "source_field" in result
        assert "metadata" in result

    def test_coefficient_fields_are_qtt(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import (
            CHTConfig, compile_cht_coefficients_1d,
        )
        from ontic.engine.vm.qtt_tensor import QTTTensor
        cfg = CHTConfig()
        result = compile_cht_coefficients_1d(
            config=cfg,
            interface_x=0.5,
            bits_per_dim=(8,),
            max_rank=16,
        )
        assert isinstance(result["k_field"], QTTTensor)
        assert isinstance(result["rho_cp_field"], QTTTensor)
        assert isinstance(result["inv_rho_cp_field"], QTTTensor)

    def test_conductivity_transition(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import (
            CHTConfig, compile_cht_coefficients_1d,
        )
        cfg = CHTConfig()
        result = compile_cht_coefficients_1d(
            config=cfg,
            interface_x=0.5,
            bits_per_dim=(6,),
            max_rank=16,
        )
        k_dense = result["k_field"].to_dense()
        N = 2 ** 6
        # First half (solid) should have higher k than second half (fluid)
        k_solid_mean = np.mean(k_dense[:N // 4])
        k_fluid_mean = np.mean(k_dense[3 * N // 4:])
        # Aluminum k=237 > Water k=0.6
        assert k_solid_mean > k_fluid_mean


# ══════════════════════════════════════════════════════════════════════
# §6  CHT Coupling — compiler
# ══════════════════════════════════════════════════════════════════════

class TestCHTCompiler:
    """Test CHT 1D compiler."""

    def test_compile(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import CHTCompiler1D
        c = CHTCompiler1D(n_bits=6, n_steps=20)
        prog = c.compile()
        assert prog.domain == "cht_1d"
        assert prog.n_registers == 10
        assert "T" in prog.fields
        assert "k" in prog.fields
        assert "inv_rho_cp" in prog.fields
        assert "source" in prog.fields

    def test_metadata(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import CHTCompiler1D
        c = CHTCompiler1D()
        prog = c.compile()
        assert prog.metadata["equations"] == "ρCp ∂T/∂t = ∇·(k∇T) + Q"
        assert "materials" in prog.metadata
        assert prog.metadata["materials"]["solid"]["name"] == "aluminum"
        assert prog.metadata["materials"]["fluid"]["name"] == "water"

    def test_custom_config(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import (
            CHTCompiler1D, CHTConfig, CHTMaterialSpec,
        )
        cfg = CHTConfig(
            solid=CHTMaterialSpec("steel", 50.0, 7800.0, 500.0),
            fluid=CHTMaterialSpec("air", 0.026, 1.225, 1005.0),
            source_power=1e6,
        )
        c = CHTCompiler1D(config=cfg, n_bits=6, n_steps=10)
        prog = c.compile()
        assert prog.metadata["materials"]["solid"]["name"] == "steel"
        assert prog.metadata["source_power"] == 1e6


class TestCHTBookkeeping:
    """Test CHT energy bookkeeping functions."""

    def test_thermal_energy(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import (
            compute_thermal_energy,
        )
        T = np.ones(100) * 300.0
        rho_cp = np.ones(100) * 4.186e6  # water
        h = 0.01
        E = compute_thermal_energy(T, rho_cp, h)
        expected = 0.01 * 100 * 4.186e6 * 300.0
        assert E == pytest.approx(expected)

    def test_interface_flux(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import (
            compute_interface_flux,
        )
        # Linear temperature: T = 300 + 100*x → dT/dx = 100
        T = np.linspace(300, 400, 100)
        k = np.ones(100) * 237.0  # aluminum
        h = 1.0 / 100
        flux = compute_interface_flux(T, k, 50, h)
        # -k * dT/dx ≈ -237 * 100 = -23700
        assert abs(flux + 23700.0) < 500.0  # rough tolerance

    def test_temperature_finite(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import (
            check_temperature_finite,
        )
        assert check_temperature_finite(np.array([300.0, 350.0, 400.0]))
        assert not check_temperature_finite(np.array([300.0, np.nan, 400.0]))
        assert not check_temperature_finite(np.array([300.0, np.inf, 400.0]))


class TestCHTSanitizer:
    """Test CHT diagnostic sanitizer."""

    def test_sanitize_allows_whitelisted(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import (
            sanitize_cht_diagnostics,
        )
        raw = {
            "total_thermal_energy": 1e6,
            "interface_heat_flux": -23700.0,
            "temperature_max": 400.0,
            "temperature_min": 300.0,
        }
        safe = sanitize_cht_diagnostics(raw)
        assert len(safe) == 4

    def test_sanitize_strips_forbidden(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import (
            sanitize_cht_diagnostics,
        )
        raw = {
            "total_thermal_energy": 1e6,
            "tt_cores": "FORBIDDEN",
            "k_field_dense": "FORBIDDEN",
        }
        safe = sanitize_cht_diagnostics(raw)
        assert "tt_cores" not in safe
        assert "k_field_dense" not in safe


# ══════════════════════════════════════════════════════════════════════
# §7  Phase-Field — configuration and construction
# ══════════════════════════════════════════════════════════════════════

class TestPhaseFieldConfig:
    """Test PhaseFieldConfig defaults."""

    def test_defaults(self) -> None:
        from ontic.engine.vm.compilers.phase_field import PhaseFieldConfig
        cfg = PhaseFieldConfig()
        assert cfg.epsilon == 0.02
        assert cfg.mobility == 1e-3
        assert cfg.sigma == 0.1
        assert cfg.rho_1 == 1.0
        assert cfg.rho_2 == 100.0


class TestPhaseFieldCompiler:
    """Test PhaseField2DCompiler."""

    def test_compile_droplet(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            PhaseField2DCompiler,
        )
        c = PhaseField2DCompiler(n_bits=5, n_steps=20, ic_type="droplet")
        prog = c.compile()
        assert prog.domain == "phase_field_2d"
        assert prog.n_registers == 8
        assert "phi" in prog.fields
        assert prog.fields["phi"].conserved_quantity == "total_phase"

    def test_compile_rayleigh_taylor(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            PhaseField2DCompiler,
        )
        c = PhaseField2DCompiler(
            n_bits=5, n_steps=20, ic_type="rayleigh_taylor",
        )
        prog = c.compile()
        assert prog.metadata["ic_type"] == "rayleigh_taylor"

    def test_invalid_ic_type(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            PhaseField2DCompiler,
        )
        c = PhaseField2DCompiler(ic_type="bogus")
        with pytest.raises(ValueError, match="Unknown IC type"):
            c.compile()

    def test_instruction_stream(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            PhaseField2DCompiler,
        )
        from ontic.engine.vm.ir import OpCode
        c = PhaseField2DCompiler(n_bits=5, n_steps=10)
        prog = c.compile()
        opcodes = [i.opcode for i in prog.instructions]
        # Must contain Hadamard for φ²·φ = φ³
        assert opcodes.count(OpCode.HADAMARD) >= 2
        # Must contain Laplace for ∇²φ and ∇²μ
        assert opcodes.count(OpCode.LAPLACE) >= 2
        # Must contain SUB for μ = -ε²∇²φ + φ³ - φ
        assert OpCode.SUB in opcodes

    def test_metadata_contains_physics(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            PhaseField2DCompiler, PhaseFieldConfig,
        )
        cfg = PhaseFieldConfig(sigma=0.05, epsilon=0.03)
        c = PhaseField2DCompiler(n_bits=5, n_steps=10, config=cfg)
        prog = c.compile()
        assert prog.metadata["epsilon"] == 0.03
        assert prog.metadata["sigma"] == 0.05
        assert prog.metadata["mobility"] == 1e-3


# ══════════════════════════════════════════════════════════════════════
# §8  Phase-Field — initial conditions
# ══════════════════════════════════════════════════════════════════════

class TestPhaseFieldIC:
    """Test phase-field initial conditions."""

    def test_circle_droplet_center(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            circle_droplet_ic,
        )
        # At center of droplet: φ ≈ -1
        phi = circle_droplet_ic(
            np.array([0.5]), np.array([0.5]),
            cx=0.5, cy=0.5, radius=0.2, epsilon=0.02,
        )
        assert phi[0] < -0.99

    def test_circle_droplet_outside(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            circle_droplet_ic,
        )
        # Far outside: φ ≈ +1
        phi = circle_droplet_ic(
            np.array([0.0]), np.array([0.0]),
            cx=0.5, cy=0.5, radius=0.2, epsilon=0.02,
        )
        assert phi[0] > 0.99

    def test_circle_droplet_range(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            circle_droplet_ic,
        )
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        phi = circle_droplet_ic(X, Y)
        assert np.all(phi >= -1.0)
        assert np.all(phi <= 1.0)

    def test_rayleigh_taylor_phases(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            rayleigh_taylor_ic,
        )
        # Well above interface: heavy fluid φ > 0
        phi_top = rayleigh_taylor_ic(
            np.array([0.5]), np.array([0.9]),
            y_interface=0.5, epsilon=0.02,
        )
        assert phi_top[0] > 0.9

        # Well below interface: light fluid φ < 0
        phi_bot = rayleigh_taylor_ic(
            np.array([0.5]), np.array([0.1]),
            y_interface=0.5, epsilon=0.02,
        )
        assert phi_bot[0] < -0.9


# ══════════════════════════════════════════════════════════════════════
# §9  Phase-Field — utilities
# ══════════════════════════════════════════════════════════════════════

class TestPhaseFieldUtilities:
    """Test phase-field utility functions."""

    def test_interface_energy_flat(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            compute_interface_energy,
        )
        # Uniform φ = 1 → no gradient, F(1) = 0
        phi = np.ones((32, 32))
        energy = compute_interface_energy(phi, epsilon=0.02, h=1.0 / 32)
        assert energy < 1e-10

    def test_interface_energy_positive(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            compute_interface_energy, circle_droplet_ic,
        )
        x = np.linspace(0, 1, 64)
        y = np.linspace(0, 1, 64)
        X, Y = np.meshgrid(x, y)
        phi = circle_droplet_ic(X, Y, epsilon=0.02)
        energy = compute_interface_energy(phi, epsilon=0.02, h=1.0 / 64)
        assert energy > 0

    def test_phase_fraction_uniform(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            compute_phase_fraction,
        )
        phi = np.ones((10, 10))  # all phase 2
        f1, f2 = compute_phase_fraction(phi)
        assert f1 == pytest.approx(0.0)
        assert f2 == pytest.approx(1.0)

    def test_phase_fraction_half(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            compute_phase_fraction,
        )
        phi = np.zeros((10, 10))  # at interface
        f1, f2 = compute_phase_fraction(phi)
        assert f1 == pytest.approx(0.5)
        assert f2 == pytest.approx(0.5)

    def test_density_field(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            compute_density_field,
        )
        phi = np.array([-1.0, 0.0, 1.0])
        rho = compute_density_field(phi, rho_1=1.0, rho_2=100.0)
        assert rho[0] == pytest.approx(1.0)    # φ=-1 → ρ₁
        assert rho[1] == pytest.approx(50.5)    # φ=0 → average
        assert rho[2] == pytest.approx(100.0)   # φ=+1 → ρ₂

    def test_viscosity_field(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            compute_viscosity_field,
        )
        phi = np.array([-1.0, 0.0, 1.0])
        mu = compute_viscosity_field(phi, mu_1=0.01, mu_2=0.1)
        assert mu[0] == pytest.approx(0.01)
        assert mu[1] == pytest.approx(0.055)
        assert mu[2] == pytest.approx(0.1)


class TestPhaseFieldSanitizer:
    """Test phase-field diagnostic sanitizer."""

    def test_sanitize_allows_whitelisted(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            sanitize_phase_field_diagnostics,
        )
        raw = {
            "total_phase": 0.0,
            "interface_energy": 0.5,
            "phase_1_fraction": 0.6,
            "phase_2_fraction": 0.4,
        }
        safe = sanitize_phase_field_diagnostics(raw)
        assert len(safe) == 4

    def test_sanitize_strips_forbidden(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            sanitize_phase_field_diagnostics,
        )
        raw = {
            "total_phase": 0.0,
            "phi_field_dense": "FORBIDDEN",
            "chemical_potential_field": "FORBIDDEN",
            "tt_cores": "FORBIDDEN",
        }
        safe = sanitize_phase_field_diagnostics(raw)
        assert len(safe) == 1
        assert "total_phase" in safe


# ══════════════════════════════════════════════════════════════════════
# §10  Evidence — BOUNDEDNESS claim tag
# ══════════════════════════════════════════════════════════════════════

class TestBoundednessClaim:
    """Test BOUNDEDNESS evidence claim generation."""

    def test_boundedness_claim_satisfied(self) -> None:
        from physics_os.core.evidence import generate_claims
        result = {
            "boundedness": {
                "predicates": {
                    "rho_positive": True,
                    "pressure_positive": True,
                },
            },
        }
        claims = generate_claims(result, "compressible_euler_1d")
        bound_claims = [c for c in claims if c["tag"] == "BOUNDEDNESS"]
        assert len(bound_claims) == 1
        assert bound_claims[0]["satisfied"] is True
        assert bound_claims[0]["witness"]["all_satisfied"] is True

    def test_boundedness_claim_violated(self) -> None:
        from physics_os.core.evidence import generate_claims
        result = {
            "boundedness": {
                "predicates": {
                    "rho_positive": True,
                    "pressure_positive": False,
                },
            },
        }
        claims = generate_claims(result, "compressible_euler_1d")
        bound_claims = [c for c in claims if c["tag"] == "BOUNDEDNESS"]
        assert len(bound_claims) == 1
        assert bound_claims[0]["satisfied"] is False
        assert "pressure_positive" in bound_claims[0]["witness"]["failed"]

    def test_boundedness_not_generated_without_data(self) -> None:
        from physics_os.core.evidence import generate_claims
        result = {}
        claims = generate_claims(result, "navier_stokes_2d")
        bound_claims = [c for c in claims if c["tag"] == "BOUNDEDNESS"]
        assert len(bound_claims) == 0


# ══════════════════════════════════════════════════════════════════════
# §11  IP Boundary — compiler metadata does not leak
# ══════════════════════════════════════════════════════════════════════

class TestIPBoundary:
    """Verify Phase F compilers do not leak forbidden fields."""

    def test_euler_no_forbidden_in_metadata(self) -> None:
        from ontic.engine.vm.compilers.compressible_euler import (
            CompressibleEuler1DCompiler,
        )
        from physics_os.core.sanitizer import FORBIDDEN_FIELDS
        c = CompressibleEuler1DCompiler()
        prog = c.compile()
        for key in prog.metadata:
            assert key not in FORBIDDEN_FIELDS, f"Forbidden key in metadata: {key}"

    def test_cht_no_forbidden_in_metadata(self) -> None:
        from ontic.engine.vm.compilers.cht_coupling import CHTCompiler1D
        from physics_os.core.sanitizer import FORBIDDEN_FIELDS
        c = CHTCompiler1D()
        prog = c.compile()
        for key in prog.metadata:
            assert key not in FORBIDDEN_FIELDS, f"Forbidden key in metadata: {key}"

    def test_phasefield_no_forbidden_in_metadata(self) -> None:
        from ontic.engine.vm.compilers.phase_field import (
            PhaseField2DCompiler,
        )
        from physics_os.core.sanitizer import FORBIDDEN_FIELDS
        c = PhaseField2DCompiler()
        prog = c.compile()
        for key in prog.metadata:
            assert key not in FORBIDDEN_FIELDS, f"Forbidden key in metadata: {key}"
