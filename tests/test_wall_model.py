"""Tests for Phase E: Wall Model + Wall Benchmarks.

Validates:
1. WallModel — precomputation, penalization, diagnostics, sanitization
2. NS2D compiler — wall model integration, instruction generation
3. Wall benchmarks — registry, gate evaluation, QoI utilities
4. IP boundary compliance — no wall internals leak

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
import pytest
import numpy as np
from numpy.typing import NDArray


# ══════════════════════════════════════════════════════════════════════
# §1  Wall Model — configuration and construction
# ══════════════════════════════════════════════════════════════════════

class TestWallModelConfig:
    """Test WallModelConfig defaults and construction."""

    def test_default_config(self) -> None:
        from ontic.engine.vm.models.wall_model import WallModelConfig
        cfg = WallModelConfig()
        assert cfg.eta_permeability == 1e-4
        assert cfg.d_min == 1e-3
        assert cfg.viscosity == 1e-3
        assert cfg.thermal_conductivity == 0.0
        assert cfg.t_wall == 0.0
        assert cfg.max_rank == 64
        assert cfg.cutoff == 1e-12

    def test_custom_config(self) -> None:
        from ontic.engine.vm.models.wall_model import WallModelConfig
        cfg = WallModelConfig(
            eta_permeability=1e-6,
            d_min=0.01,
            viscosity=0.1,
            thermal_conductivity=0.5,
            t_wall=300.0,
            max_rank=32,
        )
        assert cfg.eta_permeability == 1e-6
        assert cfg.viscosity == 0.1
        assert cfg.thermal_conductivity == 0.5
        assert cfg.t_wall == 300.0

    def test_config_immutable(self) -> None:
        from ontic.engine.vm.models.wall_model import WallModelConfig
        cfg = WallModelConfig()
        with pytest.raises(AttributeError):
            cfg.viscosity = 99.9  # type: ignore[misc]


class TestWallModel:
    """Test WallModel construction and API."""

    def test_construction_default(self) -> None:
        from ontic.engine.vm.models.wall_model import WallModel
        wm = WallModel()
        assert wm.config.eta_permeability == 1e-4

    def test_construction_custom(self) -> None:
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )
        cfg = WallModelConfig(viscosity=0.05)
        wm = WallModel(config=cfg)
        assert wm.config.viscosity == 0.05


# ══════════════════════════════════════════════════════════════════════
# §2  Wall Model — precomputation
# ══════════════════════════════════════════════════════════════════════

class TestWallModelPrecompute:
    """Test precomputation of wall-model QTT fields."""

    @pytest.fixture
    def channel_geometry(self):
        """Compile a simple channel geometry."""
        from ontic.engine.vm.compilers.geometry_coeffs import (
            GeometryCompiler, GeometryScene, GeometrySpec,
            GeometryPrimitive,
        )
        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.CHANNEL,
                    params={"wall_y_lo": 0.2, "wall_y_hi": 0.8},
                    is_solid=True,
                ),
            ],
            bits_per_dim=(6, 6),
            domain=((0.0, 1.0), (0.0, 1.0)),
            penalization_strength=1e6,
            wall_band_width=0.05,
        )
        compiler = GeometryCompiler(max_rank=32)
        return compiler.compile(scene)

    def test_precompute_returns_wall_fields(self, channel_geometry) -> None:
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig, WallFields,
        )
        wm = WallModel(config=WallModelConfig(viscosity=0.01))
        fields = wm.precompute(channel_geometry)
        assert isinstance(fields, WallFields)
        assert fields.near_wall_mask is not None
        assert fields.reciprocal_distance is not None
        assert fields.penalization_coeff is not None

    def test_precompute_shear_coeff_with_viscosity(
        self, channel_geometry,
    ) -> None:
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )
        wm = WallModel(config=WallModelConfig(viscosity=0.01))
        fields = wm.precompute(channel_geometry)
        assert fields.shear_coeff is not None

    def test_precompute_no_shear_without_viscosity(
        self, channel_geometry,
    ) -> None:
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )
        wm = WallModel(config=WallModelConfig(viscosity=0.0))
        fields = wm.precompute(channel_geometry)
        assert fields.shear_coeff is None

    def test_precompute_thermal_coeff(self, channel_geometry) -> None:
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )
        wm = WallModel(config=WallModelConfig(
            viscosity=0.01,
            thermal_conductivity=0.5,
            t_wall=300.0,
        ))
        fields = wm.precompute(channel_geometry)
        assert fields.thermal_coeff is not None

    def test_precompute_no_thermal_without_conductivity(
        self, channel_geometry,
    ) -> None:
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )
        wm = WallModel(config=WallModelConfig(thermal_conductivity=0.0))
        fields = wm.precompute(channel_geometry)
        assert fields.thermal_coeff is None

    def test_precompute_rank_stats_private(
        self, channel_geometry,
    ) -> None:
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )
        wm = WallModel(config=WallModelConfig(viscosity=0.01))
        fields = wm.precompute(channel_geometry)
        assert "penalization_coeff_max_rank" in fields.rank_stats
        assert "near_wall_mask_max_rank" in fields.rank_stats
        assert "reciprocal_distance_max_rank" in fields.rank_stats
        assert "shear_coeff_max_rank" in fields.rank_stats

    def test_penalization_coeff_large_in_solid(
        self, channel_geometry,
    ) -> None:
        """Penalization coefficient should be large inside solid."""
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )
        wm = WallModel(config=WallModelConfig(
            eta_permeability=1e-4,
            viscosity=0.01,
        ))
        fields = wm.precompute(channel_geometry)
        # Penalization = (1/eta) * chi_solid → large in solid
        penal_sum = abs(fields.penalization_coeff.sum())
        assert penal_sum > 0.0, "Penalization should be non-zero in solid"


# ══════════════════════════════════════════════════════════════════════
# §3  Wall Model — penalization and diagnostics
# ══════════════════════════════════════════════════════════════════════

class TestWallModelPenalization:
    """Test Brinkman penalization application."""

    @pytest.fixture
    def wall_setup(self):
        """Set up wall model with precomputed fields for testing."""
        from ontic.engine.vm.compilers.geometry_coeffs import (
            GeometryCompiler, GeometryScene, GeometrySpec,
            GeometryPrimitive,
        )
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )
        from ontic.engine.vm.qtt_tensor import QTTTensor

        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.CHANNEL,
                    params={"wall_y_lo": 0.2, "wall_y_hi": 0.8},
                    is_solid=True,
                ),
            ],
            bits_per_dim=(5, 5),
            domain=((0.0, 1.0), (0.0, 1.0)),
            penalization_strength=1e4,
            wall_band_width=0.05,
        )
        geometry = GeometryCompiler(max_rank=32).compile(scene)
        wm = WallModel(config=WallModelConfig(
            eta_permeability=1e-4,
            d_min=0.01,
            viscosity=0.01,
            max_rank=32,
        ))
        wall_fields = wm.precompute(geometry)

        # Create a test velocity field (uniform u=1)
        u_field = QTTTensor.from_function(
            lambda x, y: np.ones_like(x),
            bits_per_dim=(5, 5),
            domain=((0.0, 1.0), (0.0, 1.0)),
            max_rank=4,
        )

        return wm, wall_fields, u_field

    def test_penalization_reduces_velocity_in_solid(
        self, wall_setup,
    ) -> None:
        """Penalization should reduce field values in solid regions."""
        wm, wall_fields, u_field = wall_setup
        original_norm = u_field.norm()
        # Use small dt so dt/eta < 1 (stable regime).
        # With eta=1e-4, dt=1e-6 → dt/eta = 0.01.
        penalized = wm.apply_penalization(u_field, wall_fields, dt=1e-6)
        # After penalization, the norm should decrease (energy removed from solid)
        new_norm = penalized.norm()
        assert new_norm <= original_norm * 1.01  # Allow small numerical growth

    def test_penalization_returns_qtt(self, wall_setup) -> None:
        from ontic.engine.vm.qtt_tensor import QTTTensor
        wm, wall_fields, u_field = wall_setup
        result = wm.apply_penalization(u_field, wall_fields, dt=0.001)
        assert isinstance(result, QTTTensor)
        assert len(result.cores) == len(u_field.cores)

    def test_penalization_zero_dt_is_identity(self, wall_setup) -> None:
        """With dt=0, penalization should not change the field."""
        wm, wall_fields, u_field = wall_setup
        result = wm.apply_penalization(u_field, wall_fields, dt=0.0)
        # u - 0 * penal * u = u
        # QTT Hadamard + scale(0) + truncation introduces O(1e-7) noise
        diff = u_field.sub(result)
        assert diff.norm() < 1e-4 * u_field.norm()


class TestWallModelDiagnostics:
    """Test wall-model diagnostic computation."""

    @pytest.fixture
    def diag_setup(self):
        from ontic.engine.vm.compilers.geometry_coeffs import (
            GeometryCompiler, GeometryScene, GeometrySpec,
            GeometryPrimitive,
        )
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )
        from ontic.engine.vm.qtt_tensor import QTTTensor

        scene = GeometryScene(
            objects=[
                GeometrySpec(
                    primitive=GeometryPrimitive.CHANNEL,
                    params={"wall_y_lo": 0.2, "wall_y_hi": 0.8},
                    is_solid=True,
                ),
            ],
            bits_per_dim=(5, 5),
            domain=((0.0, 1.0), (0.0, 1.0)),
            penalization_strength=1e4,
            wall_band_width=0.05,
        )
        geometry = GeometryCompiler(max_rank=32).compile(scene)
        wm = WallModel(config=WallModelConfig(
            eta_permeability=1e-4,
            d_min=0.01,
            viscosity=0.01,
            max_rank=32,
        ))
        wall_fields = wm.precompute(geometry)

        u_field = QTTTensor.from_function(
            lambda x, y: np.sin(np.pi * y),
            bits_per_dim=(5, 5),
            domain=((0.0, 1.0), (0.0, 1.0)),
            max_rank=16,
        )

        return wm, wall_fields, u_field

    def test_diagnostics_keys(self, diag_setup) -> None:
        wm, wall_fields, u_field = diag_setup
        diags = wm.compute_diagnostics(u_field, wall_fields)
        assert "penalization_energy" in diags
        assert "integrated_wall_shear" in diags
        assert "max_wall_shear_proxy" in diags
        assert "integrated_heat_flux" in diags

    def test_diagnostics_non_negative(self, diag_setup) -> None:
        wm, wall_fields, u_field = diag_setup
        diags = wm.compute_diagnostics(u_field, wall_fields)
        assert diags["penalization_energy"] >= 0.0
        assert diags["integrated_wall_shear"] >= 0.0
        assert diags["integrated_heat_flux"] >= 0.0

    def test_diagnostics_penalization_nonzero(self, diag_setup) -> None:
        """With solid regions and nonzero field, penalization energy > 0."""
        wm, wall_fields, u_field = diag_setup
        diags = wm.compute_diagnostics(u_field, wall_fields)
        # Field has values in solid region → penalization energy > 0
        assert diags["penalization_energy"] > 0.0


# ══════════════════════════════════════════════════════════════════════
# §4  Wall Model — IP boundary compliance
# ══════════════════════════════════════════════════════════════════════

class TestWallModelIPBoundary:
    """Verify wall-model diagnostics respect §20.4 IP boundary."""

    def test_sanitize_diagnostics_whitelist(self) -> None:
        from ontic.engine.vm.models.wall_model import WallModel
        raw = {
            "integrated_wall_shear": 1.23,
            "max_wall_shear_proxy": 4.56,
            "integrated_heat_flux": 0.0,
            "penalization_energy": 7.89,
            "distance_proxy_field": "FORBIDDEN",
            "reciprocal_distance_field": "FORBIDDEN",
            "wall_stress_profile": "FORBIDDEN",
            "tt_cores": "FORBIDDEN",
        }
        safe = WallModel.sanitize_diagnostics(raw)
        assert "integrated_wall_shear" in safe
        assert "max_wall_shear_proxy" in safe
        assert "penalization_energy" in safe
        assert "integrated_heat_flux" in safe
        # Forbidden keys must be stripped
        assert "distance_proxy_field" not in safe
        assert "reciprocal_distance_field" not in safe
        assert "wall_stress_profile" not in safe
        assert "tt_cores" not in safe

    def test_sanitize_empty(self) -> None:
        from ontic.engine.vm.models.wall_model import WallModel
        safe = WallModel.sanitize_diagnostics({})
        assert safe == {}

    def test_rank_stats_not_in_sanitized(self) -> None:
        """Rank stats in WallFields must not appear in sanitized output."""
        from ontic.engine.vm.models.wall_model import WallModel
        raw = {
            "penalization_coeff_max_rank": 16,
            "near_wall_mask_max_rank": 8,
            "reciprocal_distance_max_rank": 12,
            "integrated_wall_shear": 0.5,
        }
        safe = WallModel.sanitize_diagnostics(raw)
        assert "penalization_coeff_max_rank" not in safe
        assert "near_wall_mask_max_rank" not in safe
        assert "reciprocal_distance_max_rank" not in safe
        assert "integrated_wall_shear" in safe


# ══════════════════════════════════════════════════════════════════════
# §5  Wall Model — IR generation
# ══════════════════════════════════════════════════════════════════════

class TestWallModelIR:
    """Test IR instruction generation for penalization."""

    def test_generate_ir_penalization_count(self) -> None:
        from ontic.engine.vm.models.wall_model import WallModel
        wm = WallModel()
        instructions = wm.generate_ir_penalization(
            field_reg=0, penal_reg=12, tmp_reg=13, dt=0.001,
        )
        # Should be: hadamard, scale, truncate, sub, truncate = 5
        assert len(instructions) == 5

    def test_generate_ir_penalization_opcodes(self) -> None:
        from ontic.engine.vm.models.wall_model import WallModel
        from ontic.engine.vm.ir import OpCode
        wm = WallModel()
        instructions = wm.generate_ir_penalization(
            field_reg=0, penal_reg=12, tmp_reg=13, dt=0.001,
        )
        opcodes = [i.opcode for i in instructions]
        assert opcodes[0] == OpCode.HADAMARD
        assert opcodes[1] == OpCode.SCALE
        assert opcodes[2] == OpCode.TRUNCATE
        assert opcodes[3] == OpCode.SUB
        assert opcodes[4] == OpCode.TRUNCATE

    def test_generate_ir_penalization_registers(self) -> None:
        from ontic.engine.vm.models.wall_model import WallModel
        wm = WallModel()
        instructions = wm.generate_ir_penalization(
            field_reg=0, penal_reg=12, tmp_reg=13, dt=0.001,
        )
        # First instruction: hadamard(tmp, penal, field)
        assert instructions[0].dst == 13
        assert instructions[0].src == (12, 0)
        # Last instruction: truncate(field)
        assert instructions[4].dst == 0


# ══════════════════════════════════════════════════════════════════════
# §6  NS2D Compiler — wall model integration
# ══════════════════════════════════════════════════════════════════════

class TestNS2DWallModel:
    """Test NS2D compiler with wall model enabled."""

    def test_compile_without_wall_model(self) -> None:
        from ontic.engine.vm.compilers.navier_stokes_2d import (
            NavierStokes2DCompiler,
        )
        compiler = NavierStokes2DCompiler(n_bits=5, n_steps=10)
        prog = compiler.compile()
        assert prog.n_registers == 12
        assert "penalization_coeff" not in prog.fields

    def test_compile_with_wall_model(self) -> None:
        from ontic.engine.vm.compilers.navier_stokes_2d import (
            NavierStokes2DCompiler,
        )
        compiler = NavierStokes2DCompiler(
            n_bits=5, n_steps=10, wall_model=True,
        )
        prog = compiler.compile()
        assert prog.n_registers == 14
        assert "penalization_coeff" in prog.fields

    def test_wall_model_metadata(self) -> None:
        from ontic.engine.vm.compilers.navier_stokes_2d import (
            NavierStokes2DCompiler,
        )
        compiler = NavierStokes2DCompiler(
            n_bits=5, n_steps=10, wall_model=True,
            eta_permeability=1e-6,
        )
        prog = compiler.compile()
        assert prog.metadata["wall_model"] is True
        assert prog.metadata["eta_permeability"] == 1e-6
        assert prog.metadata["penalization_beta"] == pytest.approx(1e6)

    def test_wall_model_instructions_include_penalization(self) -> None:
        from ontic.engine.vm.compilers.navier_stokes_2d import (
            NavierStokes2DCompiler,
        )
        from ontic.engine.vm.ir import OpCode
        compiler = NavierStokes2DCompiler(
            n_bits=5, n_steps=10, wall_model=True,
        )
        prog = compiler.compile()
        opcodes = [i.opcode for i in prog.instructions]
        # Should contain SUB for penalization (field -= dt * penal * field)
        assert OpCode.SUB in opcodes
        # Should load penalization_coeff
        load_names = [
            i.params.get("name")
            for i in prog.instructions
            if i.opcode == OpCode.LOAD_FIELD
        ]
        assert "penalization_coeff" in load_names

    def test_wall_model_no_metadata_without_flag(self) -> None:
        from ontic.engine.vm.compilers.navier_stokes_2d import (
            NavierStokes2DCompiler,
        )
        compiler = NavierStokes2DCompiler(n_bits=5, n_steps=10)
        prog = compiler.compile()
        assert "wall_model" not in prog.metadata

    def test_bc_kind_parameter(self) -> None:
        from ontic.engine.vm.compilers.navier_stokes_2d import (
            NavierStokes2DCompiler,
        )
        from ontic.engine.vm.ir import BCKind
        compiler = NavierStokes2DCompiler(
            n_bits=5, n_steps=10, bc_kind=BCKind.DIRICHLET,
        )
        prog = compiler.compile()
        omega_spec = prog.fields["omega"]
        assert omega_spec.bc == BCKind.DIRICHLET

    def test_backward_compatible_default_compile(self) -> None:
        """Existing code that calls NavierStokes2DCompiler() must still work."""
        from ontic.engine.vm.compilers.navier_stokes_2d import (
            NavierStokes2DCompiler,
        )
        compiler = NavierStokes2DCompiler()
        prog = compiler.compile()
        assert prog.domain == "navier_stokes_2d"
        assert prog.n_registers == 12
        assert len(prog.instructions) > 0


# ══════════════════════════════════════════════════════════════════════
# §7  Wall Benchmarks — registry and specifications
# ══════════════════════════════════════════════════════════════════════

class TestWallBenchmarkRegistry:
    """Test wall benchmark registry construction."""

    def test_build_registry(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            build_wall_benchmark_registry,
        )
        reg = build_wall_benchmark_registry()
        assert len(reg) == 5
        assert "W010_channel_poiseuille" in reg
        assert "W020_cavity_re100" in reg
        assert "W030_cavity_re400" in reg
        assert "W040_cavity_re1000" in reg
        assert "W050_cylinder_re20" in reg

    def test_benchmark_fields(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            build_wall_benchmark_registry,
        )
        reg = build_wall_benchmark_registry()
        spec = reg["W010_channel_poiseuille"]
        assert spec.category == "wall_verification"
        assert spec.domain_key == "navier_stokes_2d"
        assert spec.reynolds_number == 100.0
        assert len(spec.qoi) == 2
        assert len(spec.refinement_levels) == 3

    def test_cavity_re100_has_ghia_gate(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            build_wall_benchmark_registry,
        )
        reg = build_wall_benchmark_registry()
        spec = reg["W020_cavity_re100"]
        assert "centerline_u_l2_error" in spec.qoi
        assert spec.qoi["centerline_u_l2_error"]["gate_type"] == "absolute_max"

    def test_cylinder_has_drag_gate(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            build_wall_benchmark_registry,
        )
        reg = build_wall_benchmark_registry()
        spec = reg["W050_cylinder_re20"]
        assert "drag_coefficient_error" in spec.qoi


# ══════════════════════════════════════════════════════════════════════
# §8  Wall Benchmarks — Ghia reference data
# ══════════════════════════════════════════════════════════════════════

class TestGhiaReferenceData:
    """Test Ghia et al. (1982) reference data integrity."""

    def test_ghia_re100_exists(self) -> None:
        from ontic.platform.vv.wall_benchmarks import GHIA_TABLES
        assert 100 in GHIA_TABLES
        assert len(GHIA_TABLES[100]) == 17

    def test_ghia_re400_exists(self) -> None:
        from ontic.platform.vv.wall_benchmarks import GHIA_TABLES
        assert 400 in GHIA_TABLES
        assert len(GHIA_TABLES[400]) == 17

    def test_ghia_re1000_exists(self) -> None:
        from ontic.platform.vv.wall_benchmarks import GHIA_TABLES
        assert 1000 in GHIA_TABLES
        assert len(GHIA_TABLES[1000]) == 17

    def test_ghia_boundary_values(self) -> None:
        """Ghia data should have u=0 at y=0 and u=1 at y=1 (lid)."""
        from ontic.platform.vv.wall_benchmarks import GHIA_TABLES
        for re_num, data in GHIA_TABLES.items():
            # y=0 → u=0 (no-slip bottom wall)
            bottom = [v for y, v in data if y == 0.0]
            assert len(bottom) == 1
            assert bottom[0] == pytest.approx(0.0, abs=1e-10)
            # y=1 → u=1 (lid velocity)
            top = [v for y, v in data if y == 1.0]
            assert len(top) == 1
            assert top[0] == pytest.approx(1.0, abs=1e-10)


# ══════════════════════════════════════════════════════════════════════
# §9  Wall Benchmarks — QoI utilities
# ══════════════════════════════════════════════════════════════════════

class TestQoIUtilities:
    """Test QoI extraction and error computation."""

    def test_compute_ghia_l2_error_perfect(self) -> None:
        """Perfect match should give zero error."""
        from ontic.platform.vv.wall_benchmarks import (
            compute_ghia_l2_error, GHIA_TABLES,
        )
        # Construct a "perfect" simulation matching Ghia data exactly
        # Ghia data is in DECREASING y order — sort ascending for np.interp
        ghia_data = sorted(GHIA_TABLES[100], key=lambda t: t[0])
        y_coords = np.array([y for y, _ in ghia_data])
        u_values = np.array([u for _, u in ghia_data])
        error = compute_ghia_l2_error(y_coords, u_values, 100)
        assert error < 1e-12

    def test_compute_ghia_l2_error_bad(self) -> None:
        """Random data should give nonzero error."""
        from ontic.platform.vv.wall_benchmarks import compute_ghia_l2_error
        np.random.seed(42)
        y_coords = np.linspace(0, 1, 64)
        u_values = np.random.randn(64)
        error = compute_ghia_l2_error(y_coords, u_values, 100)
        assert error > 0.1

    def test_compute_ghia_invalid_re(self) -> None:
        from ontic.platform.vv.wall_benchmarks import compute_ghia_l2_error
        with pytest.raises(ValueError, match="No Ghia reference"):
            compute_ghia_l2_error(np.array([0.0]), np.array([0.0]), 999)

    def test_poiseuille_exact_zero_at_walls(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            compute_poiseuille_l2_error,
        )
        y = np.array([0.0, 0.5, 1.0])
        u = np.array([0.0, 1.0, 0.0])  # Exact Poiseuille (u_max=1, h=1)
        error = compute_poiseuille_l2_error(y, u, h_channel=1.0, u_max=1.0)
        assert error < 1e-10

    def test_poiseuille_error_with_noise(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            compute_poiseuille_l2_error, _poiseuille_exact,
        )
        y = np.linspace(0, 1, 64)
        u_exact = _poiseuille_exact(y, h_channel=1.0, u_max=1.0)
        u_noisy = u_exact + 0.01 * np.random.randn(64)
        error = compute_poiseuille_l2_error(y, u_noisy, 1.0, 1.0)
        assert 0 < error < 0.1

    def test_convergence_order_exact(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            compute_convergence_order,
        )
        # Perfect 2nd-order: error = C * h^2
        h_values = [0.1, 0.05, 0.025]
        errors = [0.01, 0.0025, 0.000625]
        order = compute_convergence_order(errors, h_values)
        assert order == pytest.approx(2.0, abs=0.1)

    def test_convergence_order_insufficient_data(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            compute_convergence_order,
        )
        order = compute_convergence_order([0.01], [0.1])
        assert order == 0.0

    def test_centerline_extraction_shape(self) -> None:
        from ontic.platform.vv.wall_benchmarks import extract_centerline_u
        N = 32
        omega = np.zeros((N, N))
        psi = np.zeros((N, N))
        y, u = extract_centerline_u(omega, psi, n_bits=5)
        assert y.shape == (N,)
        assert u.shape == (N,)


# ══════════════════════════════════════════════════════════════════════
# §10  Wall Benchmarks — gate evaluation
# ══════════════════════════════════════════════════════════════════════

class TestWallGateEvaluation:
    """Test wall-benchmark gate evaluation."""

    def test_gate_pass_absolute_max(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            build_wall_benchmark_registry, evaluate_wall_gates,
        )
        reg = build_wall_benchmark_registry()
        spec = reg["W010_channel_poiseuille"]
        results = evaluate_wall_gates(spec, {
            "velocity_profile_l2_error": 0.01,
            "convergence_order": 2.0,
        })
        assert all(r.passed for r in results)

    def test_gate_fail_absolute_max(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            build_wall_benchmark_registry, evaluate_wall_gates,
        )
        reg = build_wall_benchmark_registry()
        spec = reg["W010_channel_poiseuille"]
        results = evaluate_wall_gates(spec, {
            "velocity_profile_l2_error": 0.99,  # Exceeds 0.05 threshold
            "convergence_order": 2.0,
        })
        error_gate = [r for r in results if r.gate_name == "velocity_profile_l2_error"]
        assert not error_gate[0].passed

    def test_gate_order_min(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            build_wall_benchmark_registry, evaluate_wall_gates,
        )
        reg = build_wall_benchmark_registry()
        spec = reg["W010_channel_poiseuille"]
        results = evaluate_wall_gates(spec, {
            "velocity_profile_l2_error": 0.01,
            "convergence_order": 0.5,  # Below 1.5 threshold
        })
        order_gate = [r for r in results if r.gate_name == "convergence_order"]
        assert not order_gate[0].passed

    def test_gate_missing_qoi(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            build_wall_benchmark_registry, evaluate_wall_gates,
        )
        reg = build_wall_benchmark_registry()
        spec = reg["W010_channel_poiseuille"]
        # Missing QoI → defaults to inf → should fail
        results = evaluate_wall_gates(spec, {})
        assert not all(r.passed for r in results)


# ══════════════════════════════════════════════════════════════════════
# §11  Wall Benchmarks — diagnostic sanitizer
# ══════════════════════════════════════════════════════════════════════

class TestWallDiagnosticSanitizer:
    """Test wall diagnostic sanitizer."""

    def test_sanitize_allows_whitelisted(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            sanitize_wall_diagnostics,
        )
        raw = {
            "integrated_wall_shear": 1.0,
            "penalization_energy": 2.0,
            "convergence_order": 1.8,
        }
        safe = sanitize_wall_diagnostics(raw)
        assert len(safe) == 3

    def test_sanitize_strips_forbidden(self) -> None:
        from ontic.platform.vv.wall_benchmarks import (
            sanitize_wall_diagnostics,
        )
        raw = {
            "integrated_wall_shear": 1.0,
            "tt_cores": [1, 2, 3],
            "bond_dimensions": [4, 8],
            "distance_field_dense": np.zeros(100),
        }
        safe = sanitize_wall_diagnostics(raw)
        assert "tt_cores" not in safe
        assert "bond_dimensions" not in safe
        assert "distance_field_dense" not in safe
        assert "integrated_wall_shear" in safe


# ══════════════════════════════════════════════════════════════════════
# §12  Integration: wall model with geometry compiler
# ══════════════════════════════════════════════════════════════════════

class TestWallModelGeometryIntegration:
    """Test wall model integration with geometry coefficient compiler."""

    def test_cylinder_geometry_wall_model(self) -> None:
        """Wall model works with cylinder-in-channel geometry."""
        from ontic.engine.vm.compilers.geometry_coeffs import (
            compile_cylinder_in_channel,
        )
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )

        geometry = compile_cylinder_in_channel(
            bits_per_dim=(5, 5),
            max_rank=16,
        )
        wm = WallModel(config=WallModelConfig(
            eta_permeability=1e-4,
            viscosity=0.01,
            max_rank=16,
        ))
        wall_fields = wm.precompute(geometry)
        assert wall_fields.penalization_coeff is not None
        assert wall_fields.shear_coeff is not None

    def test_cavity_geometry_wall_model(self) -> None:
        """Wall model works with lid-driven cavity geometry."""
        from ontic.engine.vm.compilers.geometry_coeffs import (
            compile_lid_driven_cavity,
        )
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )

        geometry = compile_lid_driven_cavity(
            bits_per_dim=(5, 5),
            max_rank=16,
        )
        wm = WallModel(config=WallModelConfig(
            eta_permeability=1e-4,
            viscosity=0.01,
            max_rank=16,
        ))
        wall_fields = wm.precompute(geometry)
        assert wall_fields.penalization_coeff is not None

    def test_step_geometry_wall_model(self) -> None:
        """Wall model works with backward-facing step geometry."""
        from ontic.engine.vm.compilers.geometry_coeffs import (
            compile_backward_facing_step,
        )
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )

        geometry = compile_backward_facing_step(
            bits_per_dim=(5, 5),
            max_rank=16,
        )
        wm = WallModel(config=WallModelConfig(
            eta_permeability=1e-4,
            viscosity=0.01,
            max_rank=16,
        ))
        wall_fields = wm.precompute(geometry)
        assert wall_fields.penalization_coeff is not None

    def test_wall_diagnostics_with_real_field(self) -> None:
        """Full integration: compile geometry, precompute, run diagnostics."""
        from ontic.engine.vm.compilers.geometry_coeffs import (
            compile_backward_facing_step,
        )
        from ontic.engine.vm.models.wall_model import (
            WallModel, WallModelConfig,
        )
        from ontic.engine.vm.qtt_tensor import QTTTensor

        geometry = compile_backward_facing_step(
            bits_per_dim=(5, 5),
            max_rank=16,
        )
        wm = WallModel(config=WallModelConfig(
            eta_permeability=1e-4,
            viscosity=0.01,
            max_rank=16,
        ))
        wall_fields = wm.precompute(geometry)

        # Create a velocity-like field
        u = QTTTensor.from_function(
            lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y),
            bits_per_dim=(5, 5),
            domain=((0.0, 10.0), (0.0, 2.0)),
            max_rank=16,
        )

        diags = wm.compute_diagnostics(u, wall_fields)
        safe = WallModel.sanitize_diagnostics(diags)
        assert "penalization_energy" in safe
        assert safe["penalization_energy"] > 0.0
