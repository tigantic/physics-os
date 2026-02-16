#!/usr/bin/env python3
"""Smoke-tests for the genuine Vlasov solvers.

Tests cover:
  1. 1D+1V solver construction + a few steps
  2. 6D solver construction + IC + a few steps (small grid)
  3. 6D diagnostics (norm conservation, compression, E-field energy)
  4. 3D Poisson solve: known analytic charge → E-field check
  5. Morton LUT round-trip consistency

Run:
    python -m pytest tests/test_vlasov_genuine.py -v
    # or
    make vlasov-test

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "QTeneT" / "src" / "qtenet"))

from qtenet.solvers.vlasov6d_genuine import (
    Vlasov6DGenuine,
    Vlasov6DGenuineConfig,
    Vlasov6DGenuineState,
    _build_morton_lut_3d,
    _qtt_inner,
    dense_to_qtt_3d,
    poisson_solve_3d,
    qtt_to_dense_3d,
)
from qtenet.solvers.vlasov_genuine import (
    Vlasov1D1V,
    Vlasov1D1VConfig,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

DEVICE = "cpu"
DTYPE = torch.float32


@pytest.fixture(scope="module")
def small_solver() -> Vlasov6DGenuine:
    """4^6 = 4,096-point solver for fast testing."""
    cfg = Vlasov6DGenuineConfig(
        qubits_per_dim=2,
        max_rank=16,
        svd_tol=1e-6,
        device=DEVICE,
        dtype=DTYPE,
    )
    return Vlasov6DGenuine(cfg)


@pytest.fixture(scope="module")
def small_state(small_solver: Vlasov6DGenuine) -> Vlasov6DGenuineState:
    """IC for the 4^6 solver."""
    return small_solver.two_stream_ic()


# ─────────────────────────────────────────────────────────────────────────
# 1. 1D+1V Solver
# ─────────────────────────────────────────────────────────────────────────

class Test1D1V:
    """Basic construction and stepping of the 1D+1V Vlasov-Poisson solver."""

    def test_construction(self) -> None:
        cfg = Vlasov1D1VConfig(qubits_per_dim=4, max_rank=16, device=DEVICE)
        solver = Vlasov1D1V(cfg)
        state = solver.landau_ic()
        assert len(state.cores) == 2 * 4  # 2 dims × 4 bits
        assert state.time == 0.0

    def test_step_preserves_norm(self) -> None:
        cfg = Vlasov1D1VConfig(qubits_per_dim=5, max_rank=32, device=DEVICE)
        solver = Vlasov1D1V(cfg)
        state = solver.landau_ic()
        norm_before = _qtt_inner(state.cores)
        # Use the dense-step path (validated to 0.6% Landau damping accuracy)
        state = solver.step_dense(state, dt=0.01)
        norm_after = _qtt_inner(state.cores)
        # Dense spectral step conserves norm well
        drift = abs(norm_after - norm_before) / max(abs(norm_before), 1e-30)
        assert drift < 0.05, f"1D+1V norm drift {drift:.2e} exceeds 5%"
        assert state.step_count == 1


# ─────────────────────────────────────────────────────────────────────────
# 2. 6D Solver — Construction & Stepping
# ─────────────────────────────────────────────────────────────────────────

class Test6DConstruction:
    """Build solver, create IC, take a few steps at 4^6."""

    def test_ic_shape(
        self, small_solver: Vlasov6DGenuine, small_state: Vlasov6DGenuineState
    ) -> None:
        expected_cores = 6 * 2  # 6 dims × 2 bits
        assert len(small_state.cores) == expected_cores
        assert small_state.total_qubits == expected_cores
        assert small_state.num_dims == 6
        assert small_state.qubits_per_dim == 2
        assert small_state.grid_size == 4
        assert small_state.total_points == 4 ** 6

    def test_step_returns_new_state(
        self, small_solver: Vlasov6DGenuine, small_state: Vlasov6DGenuineState
    ) -> None:
        new_state = small_solver.step(small_state, dt=0.005)
        assert new_state.step_count == 1
        assert new_state.time == pytest.approx(0.005, abs=1e-10)
        assert len(new_state.E_energy) == 1
        assert len(new_state.norm_l2_sq) == 1

    def test_three_steps_norm_stable(
        self, small_solver: Vlasov6DGenuine, small_state: Vlasov6DGenuineState
    ) -> None:
        norm_before = _qtt_inner(small_state.cores)
        state = small_state
        for _ in range(3):
            state = small_solver.step(state, dt=0.005)
        norm_after = _qtt_inner(state.cores)
        drift = abs(norm_after - norm_before) / max(abs(norm_before), 1e-30)
        assert drift < 1e-3, f"6D norm drift {drift:.2e} after 3 steps exceeds 1e-3"

    def test_diagnostics(
        self, small_solver: Vlasov6DGenuine, small_state: Vlasov6DGenuineState
    ) -> None:
        diag = small_solver.compute_diagnostics(small_state)
        assert "norm_l2_sq" in diag
        assert "E_energy" in diag
        assert "max_rank" in diag
        assert "compression_ratio" in diag
        assert diag["norm_l2_sq"] > 0.0
        assert diag["compression_ratio"] > 1.0

    def test_particle_count_positive(
        self, small_solver: Vlasov6DGenuine, small_state: Vlasov6DGenuineState
    ) -> None:
        count = small_solver.compute_particle_count(small_state)
        assert count > 0.0, "Particle count must be positive for two-stream IC"


# ─────────────────────────────────────────────────────────────────────────
# 3. Morton LUT Round-Trip
# ─────────────────────────────────────────────────────────────────────────

class TestMortonLUT:
    """Verify 3D Morton LUT consistency at multiple resolutions."""

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_round_trip(self, L: int) -> None:
        N = 1 << L
        lut, inv_lut = _build_morton_lut_3d(L)
        assert lut.shape == (N ** 3, 3)
        assert inv_lut.shape == (N, N, N)

        # Every Morton index maps back uniquely
        for m in range(N ** 3):
            x, y, z = lut[m].tolist()
            assert inv_lut[x, y, z].item() == m, (
                f"Morton round-trip failed: m={m} → ({x},{y},{z}) → "
                f"{inv_lut[x, y, z].item()}"
            )


# ─────────────────────────────────────────────────────────────────────────
# 4. 3D Dense ↔ QTT Round-Trip
# ─────────────────────────────────────────────────────────────────────────

class TestDenseQTT3D:
    """Compress a 3D tensor → QTT → decompress and verify."""

    def test_round_trip_identity(self) -> None:
        L = 3  # 8×8×8
        N = 1 << L
        arr = torch.randn(N, N, N, dtype=torch.float64)
        cores = dense_to_qtt_3d(arr, L, max_rank=64, tol=1e-12)
        recovered = qtt_to_dense_3d(cores, L)
        err = (arr - recovered).norm() / arr.norm()
        assert err < 1e-6, f"3D QTT round-trip error {err:.2e} exceeds 1e-6"


# ─────────────────────────────────────────────────────────────────────────
# 5. 3D Poisson Solve
# ─────────────────────────────────────────────────────────────────────────

class TestPoisson3D:
    """Poisson solve on a known charge distribution to verify E-field."""

    def test_zero_charge_zero_field(self) -> None:
        N = 8
        dx = 1.0
        rho = torch.zeros(N, N, N, dtype=torch.float64)
        Ex, Ey, Ez = poisson_solve_3d(rho, dx)
        assert Ex.abs().max() < 1e-12
        assert Ey.abs().max() < 1e-12
        assert Ez.abs().max() < 1e-12

    def test_sinusoidal_charge(self) -> None:
        """A single-mode charge sin(kx) should produce E_x = cos(kx)/k."""
        N = 64
        L_box = 2.0 * math.pi
        dx = L_box / N
        x = torch.linspace(0, L_box - dx, N, dtype=torch.float64)
        k = 1.0  # wavenumber

        # ρ(x,y,z) = sin(k·x) (only varies in x)
        rho = torch.sin(k * x).unsqueeze(1).unsqueeze(2).expand(N, N, N).clone()

        Ex, Ey, Ez = poisson_solve_3d(rho, dx)

        # Expected: Ex = -dφ/dx where -k²φ_hat = ρ_hat → φ = sin(kx)/k²
        # So Ex = -cos(kx) · 1/k × (-1) = cos(kx)/k ... wait let me derive:
        # -∇²φ = ρ → k²φ̂ = ρ̂ → φ(x) = sin(kx)/k²
        # Ex = -∂φ/∂x = -cos(kx)·k / k² = -cos(kx)/k
        expected_Ex = -torch.cos(k * x) / k

        # Compare along the x-axis (averaged over y,z since ρ is x-only)
        Ex_mean = Ex.mean(dim=(1, 2))
        err = (Ex_mean - expected_Ex).norm() / expected_Ex.norm()
        assert err < 0.05, f"Poisson Ex error {err:.2e} exceeds 5%"

        # Ey, Ez should be negligible
        assert Ey.abs().max() < 1e-6
        assert Ez.abs().max() < 1e-6


# ─────────────────────────────────────────────────────────────────────────
# 6. State Properties
# ─────────────────────────────────────────────────────────────────────────

class TestStateProperties:
    """Verify derived properties on Vlasov6DGenuineState."""

    def test_memory_bytes(self, small_state: Vlasov6DGenuineState) -> None:
        mem = small_state.memory_bytes
        assert mem > 0
        # Should be sum of numel × element_size for each core
        expected = sum(c.numel() * c.element_size() for c in small_state.cores)
        assert mem == expected

    def test_max_rank(self, small_state: Vlasov6DGenuineState) -> None:
        assert small_state.max_rank >= 1
        assert small_state.max_rank <= 16  # chi_max for small solver


# ─────────────────────────────────────────────────────────────────────────
# Entry Point (for running outside pytest)
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
