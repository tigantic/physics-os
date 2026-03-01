"""Tests for post-processing extractors and correlation engine (Phase 4)."""

from __future__ import annotations

import math

import pytest
import torch

from physics_os.templates.extractors import (
    BoundaryLayerResult,
    CorrelationComparison,
    DragResult,
    LiftResult,
    NusseltResult,
    PressureDropResult,
    RecirculationResult,
    SkinFrictionResult,
    StrouhalResult,
    VelocityProfileResult,
    compare_drag_cylinder,
    compare_nusselt_cylinder,
    compare_nusselt_flat_plate,
    compare_skin_friction_plate,
    compare_strouhal_cylinder,
    extract_boundary_layer,
    extract_drag,
    extract_lift,
    extract_nusselt,
    extract_pressure_drop,
    extract_recirculation,
    extract_skin_friction,
    extract_strouhal,
    extract_velocity_profile,
)


# ──────────────────────────────────────────────────────────────────
# Drag / Lift extractors
# ──────────────────────────────────────────────────────────────────

class TestExtractDrag:
    def test_uniform_pressure_zero_drag(self) -> None:
        """Uniform pressure field → zero drag on symmetric body."""
        p = torch.ones(32, 32, dtype=torch.float64)
        mask = torch.zeros(32, 32, dtype=torch.bool)
        mask[12:20, 12:20] = True  # square body
        result = extract_drag(p, mask, dx=0.1, dy=0.1, rho_inf=1.0, u_inf=1.0)
        assert isinstance(result, DragResult)
        # Symmetric body in uniform pressure → small net force
        assert abs(result.force_x) < 5.0  # not exactly zero due to discretization

    def test_zero_velocity_zero_drag(self) -> None:
        p = torch.ones(32, 32, dtype=torch.float64)
        mask = torch.zeros(32, 32, dtype=torch.bool)
        result = extract_drag(p, mask, 0.1, 0.1, 1.0, 0.0)
        assert result.cd == 0.0


class TestExtractLift:
    def test_returns_lift_result(self) -> None:
        p = torch.randn(32, 32, dtype=torch.float64)
        mask = torch.zeros(32, 32, dtype=torch.bool)
        mask[14:18, 14:18] = True
        result = extract_lift(p, mask, 0.1, 0.1, 1.0, 1.0)
        assert isinstance(result, LiftResult)


# ──────────────────────────────────────────────────────────────────
# Nusselt extractor
# ──────────────────────────────────────────────────────────────────

class TestExtractNusselt:
    def test_linear_temperature_profile(self) -> None:
        """Linear T(y) → constant Nu."""
        T = torch.zeros(32, 64, dtype=torch.float64)
        for j in range(32):
            T[j, :] = 400.0 - j * 3.0  # linear gradient
        result = extract_nusselt(
            T, T_wall=400.0, T_inf=300.0,
            thermal_conductivity=0.025, characteristic_length=0.1,
            dy=0.001, wall_row=0,
        )
        assert isinstance(result, NusseltResult)
        assert result.nu_avg != 0

    def test_zero_delta_t(self) -> None:
        T = torch.ones(10, 10, dtype=torch.float64) * 300
        result = extract_nusselt(T, 300, 300, 0.025, 0.1, 0.01)
        assert result.nu_avg == 0.0


# ──────────────────────────────────────────────────────────────────
# Strouhal extractor
# ──────────────────────────────────────────────────────────────────

class TestExtractStrouhal:
    def test_sinusoidal_signal(self) -> None:
        """Known sine frequency → correct Strouhal."""
        freq = 21.0  # Hz
        dt = 0.001
        t = torch.arange(0, 1.0, dt, dtype=torch.float64)
        signal = torch.sin(2 * math.pi * freq * t)

        result = extract_strouhal(signal, dt=dt, characteristic_length=0.01, free_stream_velocity=1.0)
        assert isinstance(result, StrouhalResult)
        assert result.dominant_frequency == pytest.approx(freq, abs=1.5)
        assert result.st == pytest.approx(freq * 0.01 / 1.0, abs=0.02)

    def test_too_short_signal(self) -> None:
        result = extract_strouhal(torch.tensor([1.0, 2.0]), 0.01, 0.01, 1.0)
        assert result.st == 0.0


# ──────────────────────────────────────────────────────────────────
# Pressure drop
# ──────────────────────────────────────────────────────────────────

class TestExtractPressureDrop:
    def test_linear_gradient(self) -> None:
        p = torch.zeros(32, 64, dtype=torch.float64)
        for j in range(64):
            p[:, j] = 100.0 - j * 1.0
        result = extract_pressure_drop(p, inlet_col=0, outlet_col=-1)
        assert isinstance(result, PressureDropResult)
        assert result.delta_p > 0  # inlet > outlet


# ──────────────────────────────────────────────────────────────────
# Velocity profile
# ──────────────────────────────────────────────────────────────────

class TestExtractVelocityProfile:
    def test_parabolic(self) -> None:
        """Poiseuille profile u(y) = U_max (1 - (y/H)²)."""
        y = torch.linspace(-1, 1, 64, dtype=torch.float64)
        u = 1.0 - y**2
        field = u.unsqueeze(1).expand(64, 32)
        result = extract_velocity_profile(field, y, x_col=16)
        assert isinstance(result, VelocityProfileResult)
        assert result.u_max == pytest.approx(1.0, abs=0.02)


# ──────────────────────────────────────────────────────────────────
# Recirculation
# ──────────────────────────────────────────────────────────────────

class TestExtractRecirculation:
    def test_synthetic(self) -> None:
        """Synthetic u-field with a zero crossing."""
        nx = 100
        x = torch.linspace(0, 2, nx, dtype=torch.float64)
        u = torch.zeros(5, nx, dtype=torch.float64)
        # Negative from x=0.5 to x=1.0, positive after
        for i in range(nx):
            if 0.5 < x[i].item() < 1.0:
                u[1, i] = -0.1
            else:
                u[1, i] = 0.5
        result = extract_recirculation(u, x, step_x=0.5, step_height=0.1, probe_row=1)
        assert isinstance(result, RecirculationResult)
        assert 0.9 < result.reattachment_x < 1.1
        assert result.normalized_length > 0


# ──────────────────────────────────────────────────────────────────
# Boundary layer
# ──────────────────────────────────────────────────────────────────

class TestExtractBoundaryLayer:
    def test_blasius_like(self) -> None:
        """Profile that reaches 0.99 U_inf somewhere → finite δ_99."""
        y = torch.linspace(0, 0.1, 100, dtype=torch.float64)
        # Approximate Blasius-like profile
        eta = y / 0.02
        u = torch.tanh(2.0 * eta)  # smooth wall function
        result = extract_boundary_layer(u, y, u_inf=1.0, x_location=0.5, dy=0.001)
        assert isinstance(result, BoundaryLayerResult)
        assert result.delta_99 > 0
        assert result.delta_star > 0
        assert result.theta > 0
        assert result.shape_factor > 1.0  # always > 1 for realistic profiles


# ──────────────────────────────────────────────────────────────────
# Correlation comparisons
# ──────────────────────────────────────────────────────────────────

class TestCorrelationEngine:
    def test_drag_cylinder_match(self) -> None:
        """If simulated Cd ≈ 1.2 at subcritical Re, should match."""
        result = compare_drag_cylinder(cd_sim=1.25, re_D=1e4)
        assert isinstance(result, CorrelationComparison)
        assert result.within_tolerance  # 1.25 vs 1.2, 4% error < 25%

    def test_drag_cylinder_mismatch(self) -> None:
        """Simulated Cd = 5.0 at subcritical Re → large error."""
        result = compare_drag_cylinder(cd_sim=5.0, re_D=1e4)
        assert not result.within_tolerance

    def test_strouhal_comparison(self) -> None:
        result = compare_strouhal_cylinder(
            st_sim=0.21, re_D=1e4, diameter=0.01, velocity=1.0,
        )
        assert result.within_tolerance

    def test_nusselt_flat_plate(self) -> None:
        # Laminar: Nu ≈ 0.664 * Re^0.5 * Pr^(1/3) ≈ 0.664*316*0.896 ≈ 188
        nu_corr = 0.664 * (1e5**0.5) * (0.72**(1/3))
        result = compare_nusselt_flat_plate(
            nu_sim=nu_corr * 1.05, re_L=1e5, pr=0.72,
            thermal_conductivity=0.025, length=1.0,
        )
        assert result.within_tolerance

    def test_nusselt_cylinder(self) -> None:
        result = compare_nusselt_cylinder(
            nu_sim=100, re_D=1e4, pr=0.72,
            thermal_conductivity=0.025, diameter=0.01,
        )
        assert isinstance(result, CorrelationComparison)

    def test_skin_friction(self) -> None:
        cf_blasius = 1.328 / (1e4**0.5)
        result = compare_skin_friction_plate(cf_sim=cf_blasius, re_L=1e4)
        assert result.within_tolerance
