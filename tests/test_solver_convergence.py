#!/usr/bin/env python3
"""
Solver convergence and benchmark tests for the Ahmed Body IB solver.

Validates physics correctness, energy behavior, rank stability,
diagnostics completeness, integrator comparison, and wall-clock performance
on a real (small) QTT Navier-Stokes solve. No mocks, no stubs — every
test exercises the full solver hot path on GPU.

Author: Brad Adams / Tigantic Holdings LLC
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

# ── project root on sys.path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ahmed_body_ib_solver import (  # noqa: E402
    AhmedBodyConfig,
    AhmedBodyIBSolver,
    AhmedBodyParams,
)

# ── skip module if CUDA unavailable ───────────────────────────────
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for solver convergence tests",
)

# ── constants ─────────────────────────────────────────────────────
DEVICE = "cuda"
SMALL_N_BITS = 5        # 32³ — fast for CI
SMALL_MAX_RANK = 16     # low rank ceiling for speed
FEW_STEPS = 5           # minimal step count
MODERATE_STEPS = 20     # enough to see energy trends
CFL = 0.08              # standard CFL

# ── all diagnostic keys we expect from step() ─────────────────────
EXPECTED_DIAG_KEYS = {
    "step", "time", "energy", "max_rank_u", "mean_rank_u",
    "compression_ratio", "clamped",
    "u_max", "cfl_actual",
    "enstrophy", "divergence_max",
    "gpu_mem_mb", "gpu_peak_mb",
}


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

def _make_config(
    n_bits: int = SMALL_N_BITS,
    max_rank: int = SMALL_MAX_RANK,
    n_steps: int = FEW_STEPS,
    integrator: str = "rk2",
    use_projection: bool = False,
    cfl: float = CFL,
    diagnostics_interval: int = 1,
    smagorinsky_cs: float = 0.3,
) -> AhmedBodyConfig:
    """Create a lightweight AhmedBodyConfig for testing."""
    body = AhmedBodyParams(velocity=40.0)
    return AhmedBodyConfig(
        n_bits=n_bits,
        max_rank=max_rank,
        n_steps=n_steps,
        cfl=cfl,
        body_params=body,
        convergence_tol=1e-4,
        integrator=integrator,
        use_projection=use_projection,
        diagnostics_interval=diagnostics_interval,
        smagorinsky_cs=smagorinsky_cs,
        device=DEVICE,
    )


def _build_solver(cfg: AhmedBodyConfig) -> AhmedBodyIBSolver:
    """Instantiate solver and clear GPU caches for accurate memory tracking."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return AhmedBodyIBSolver(cfg)


def _run_n_steps(solver: AhmedBodyIBSolver, n: int) -> List[Dict[str, Any]]:
    """Execute *n* timesteps, returning all diagnostics dicts."""
    diags: List[Dict[str, Any]] = []
    for _ in range(n):
        diags.append(solver.step())
    return diags


# ═══════════════════════════════════════════════════════════════════
# 1. Instantiation
# ═══════════════════════════════════════════════════════════════════

class TestSolverInstantiation:
    """Verify the solver builds correctly from config."""

    def test_solver_creates_without_error(self) -> None:
        cfg = _make_config()
        solver = _build_solver(cfg)
        assert solver is not None

    def test_initial_energy_positive(self) -> None:
        cfg = _make_config()
        solver = _build_solver(cfg)
        e0 = solver._energy(solver.u)
        assert e0 > 0.0, f"Initial energy must be positive, got {e0}"

    def test_config_derived_fields(self) -> None:
        cfg = _make_config()
        assert cfg.N == (1 << SMALL_N_BITS)
        assert cfg.dx > 0.0
        assert cfg.dt > 0.0
        assert cfg.nu_eff > 0.0
        assert cfg.Re_eff > 0.0

    @pytest.mark.parametrize("integrator", ["euler", "rk2"])
    def test_integrator_accepted(self, integrator: str) -> None:
        cfg = _make_config(integrator=integrator)
        solver = _build_solver(cfg)
        assert solver.config.integrator == integrator


# ═══════════════════════════════════════════════════════════════════
# 2. Diagnostics Completeness
# ═══════════════════════════════════════════════════════════════════

class TestDiagnostics:
    """Ensure step() returns all expected diagnostics."""

    def test_first_step_has_all_keys(self) -> None:
        """First step should always compute extended diagnostics."""
        cfg = _make_config(n_steps=1, diagnostics_interval=1)
        solver = _build_solver(cfg)
        diag = solver.step()
        missing = EXPECTED_DIAG_KEYS - set(diag.keys())
        assert not missing, f"Missing diagnostic keys: {missing}"

    def test_diagnostics_types(self) -> None:
        cfg = _make_config(n_steps=1, diagnostics_interval=1)
        solver = _build_solver(cfg)
        diag = solver.step()

        assert isinstance(diag["step"], int)
        assert isinstance(diag["time"], float)
        assert isinstance(diag["energy"], float)
        assert isinstance(diag["max_rank_u"], int)
        assert isinstance(diag["mean_rank_u"], float)
        assert isinstance(diag["compression_ratio"], float)
        assert isinstance(diag["clamped"], bool)
        assert isinstance(diag["u_max"], float)
        assert isinstance(diag["cfl_actual"], float)
        assert isinstance(diag["enstrophy"], float)
        assert isinstance(diag["divergence_max"], float)
        assert isinstance(diag["gpu_mem_mb"], float)
        assert isinstance(diag["gpu_peak_mb"], float)


# ═══════════════════════════════════════════════════════════════════
# 3. Physics Invariants (per step)
# ═══════════════════════════════════════════════════════════════════

class TestPhysicsInvariants:
    """Verify physics invariants hold at every timestep."""

    @pytest.fixture(autouse=True)
    def _solver_and_diags(self) -> None:
        cfg = _make_config(n_steps=MODERATE_STEPS, diagnostics_interval=1)
        self.solver = _build_solver(cfg)
        self.diags = _run_n_steps(self.solver, MODERATE_STEPS)

    def test_energy_always_positive(self) -> None:
        for d in self.diags:
            assert d["energy"] > 0.0, (
                f"Step {d['step']}: energy must be positive, got {d['energy']}"
            )

    def test_energy_finite(self) -> None:
        for d in self.diags:
            assert math.isfinite(d["energy"]), (
                f"Step {d['step']}: energy is not finite ({d['energy']})"
            )

    def test_rank_bounded(self) -> None:
        for d in self.diags:
            assert d["max_rank_u"] <= SMALL_MAX_RANK, (
                f"Step {d['step']}: max_rank {d['max_rank_u']} exceeds "
                f"ceiling {SMALL_MAX_RANK}"
            )

    def test_compression_positive(self) -> None:
        for d in self.diags:
            assert d["compression_ratio"] > 1.0, (
                f"Step {d['step']}: compression_ratio must be > 1.0, "
                f"got {d['compression_ratio']}"
            )

    def test_cfl_stability(self) -> None:
        """Actual CFL should not wildly exceed the configured CFL target."""
        for d in self.diags:
            assert d["cfl_actual"] < CFL * 5.0, (
                f"Step {d['step']}: cfl_actual {d['cfl_actual']:.4f} "
                f"exceeds 5× configured CFL ({CFL})"
            )

    def test_u_max_positive(self) -> None:
        for d in self.diags:
            assert d["u_max"] > 0.0, (
                f"Step {d['step']}: u_max must be positive, got {d['u_max']}"
            )

    def test_no_nan_in_diagnostics(self) -> None:
        """No NaN or Inf in any numeric diagnostic."""
        numeric_keys = [
            "energy", "max_rank_u", "mean_rank_u", "compression_ratio",
            "u_max", "cfl_actual", "enstrophy", "divergence_max",
            "gpu_mem_mb", "gpu_peak_mb",
        ]
        for d in self.diags:
            for k in numeric_keys:
                v = d[k]
                if isinstance(v, float):
                    assert math.isfinite(v), (
                        f"Step {d['step']}: {k} = {v} is not finite"
                    )

    def test_enstrophy_non_negative(self) -> None:
        for d in self.diags:
            assert d["enstrophy"] >= 0.0, (
                f"Step {d['step']}: enstrophy must be ≥ 0, "
                f"got {d['enstrophy']}"
            )

    def test_divergence_non_negative(self) -> None:
        for d in self.diags:
            assert d["divergence_max"] >= 0.0, (
                f"Step {d['step']}: divergence_max must be ≥ 0, "
                f"got {d['divergence_max']}"
            )


# ═══════════════════════════════════════════════════════════════════
# 4. Energy Convergence
# ═══════════════════════════════════════════════════════════════════

class TestEnergyConvergence:
    """Validate energy behavior over the simulation timeline."""

    def test_energy_decreases_over_run(self) -> None:
        """Overall energy should decrease from step 1 to final step."""
        cfg = _make_config(n_steps=MODERATE_STEPS, diagnostics_interval=1)
        solver = _build_solver(cfg)
        diags = _run_n_steps(solver, MODERATE_STEPS)

        e_first = diags[0]["energy"]
        e_last = diags[-1]["energy"]
        # Allow up to 5% growth due to clamps / transient instabilities
        assert e_last <= e_first * 1.05, (
            f"Energy grew from {e_first:.4e} to {e_last:.4e} "
            f"({(e_last / e_first - 1) * 100:.1f}% increase)"
        )

    def test_energy_conservation_per_step(self) -> None:
        """Step-to-step energy ratio should stay near 1 (no blow-up)."""
        cfg = _make_config(n_steps=MODERATE_STEPS, diagnostics_interval=1)
        solver = _build_solver(cfg)
        diags = _run_n_steps(solver, MODERATE_STEPS)

        for i in range(1, len(diags)):
            e_prev = diags[i - 1]["energy"]
            e_curr = diags[i]["energy"]
            if e_prev > 0.0:
                ratio = e_curr / e_prev
                # Allow ≤ 0.5% growth per step (energy clamp at 1.005)
                assert ratio <= 1.006, (
                    f"Step {diags[i]['step']}: energy ratio "
                    f"{ratio:.6f} exceeds conservation bound"
                )

    def test_energy_monotone_with_clamp(self) -> None:
        """When clamp fires, energy should not increase."""
        cfg = _make_config(n_steps=MODERATE_STEPS, diagnostics_interval=1)
        solver = _build_solver(cfg)
        diags = _run_n_steps(solver, MODERATE_STEPS)

        for i in range(1, len(diags)):
            if diags[i]["clamped"]:
                assert diags[i]["energy"] <= diags[i - 1]["energy"] * 1.001, (
                    f"Step {diags[i]['step']}: clamped but energy grew"
                )


# ═══════════════════════════════════════════════════════════════════
# 5. Integrator Comparison
# ═══════════════════════════════════════════════════════════════════

class TestIntegratorComparison:
    """Compare Euler and RK2 on identical setups."""

    def _run_integrator(self, integrator: str, n_steps: int = 10) -> List[Dict[str, Any]]:
        cfg = _make_config(
            n_steps=n_steps,
            integrator=integrator,
            diagnostics_interval=1,
        )
        solver = _build_solver(cfg)
        return _run_n_steps(solver, n_steps)

    def test_both_integrators_produce_valid_energy(self) -> None:
        for name in ("euler", "rk2"):
            diags = self._run_integrator(name)
            for d in diags:
                assert d["energy"] > 0.0 and math.isfinite(d["energy"]), (
                    f"{name} step {d['step']}: invalid energy {d['energy']}"
                )

    def test_rk2_energy_not_worse_than_euler(self) -> None:
        """RK2 should produce comparable or lower final energy than Euler
        (second-order method should not be less stable)."""
        n = 10
        euler_diags = self._run_integrator("euler", n)
        rk2_diags = self._run_integrator("rk2", n)

        e_euler_final = euler_diags[-1]["energy"]
        e_rk2_final = rk2_diags[-1]["energy"]

        # We allow a generous 20% tolerance for stochastic rSVD variations
        assert e_rk2_final <= e_euler_final * 1.20, (
            f"RK2 final energy {e_rk2_final:.4e} significantly exceeds "
            f"Euler {e_euler_final:.4e}"
        )


# ═══════════════════════════════════════════════════════════════════
# 6. Rank Stability
# ═══════════════════════════════════════════════════════════════════

class TestRankStability:
    """Ensure TT-rank behavior is stable and bounded."""

    @pytest.fixture(autouse=True)
    def _run(self) -> None:
        cfg = _make_config(n_steps=MODERATE_STEPS, diagnostics_interval=1)
        self.solver = _build_solver(cfg)
        self.diags = _run_n_steps(self.solver, MODERATE_STEPS)

    def test_max_rank_never_exceeds_ceiling(self) -> None:
        for d in self.diags:
            assert d["max_rank_u"] <= SMALL_MAX_RANK

    def test_mean_rank_below_max(self) -> None:
        for d in self.diags:
            assert d["mean_rank_u"] <= d["max_rank_u"], (
                f"Step {d['step']}: mean_rank {d['mean_rank_u']} > "
                f"max_rank {d['max_rank_u']}"
            )

    def test_rank_variance_bounded(self) -> None:
        """Max rank should not oscillate wildly between steps."""
        ranks = [d["max_rank_u"] for d in self.diags]
        if len(ranks) > 1:
            diffs = [abs(ranks[i] - ranks[i - 1]) for i in range(1, len(ranks))]
            max_jump = max(diffs)
            # Rank should not jump more than half the ceiling in a single step
            assert max_jump <= SMALL_MAX_RANK // 2, (
                f"Rank jumped by {max_jump} in one step"
            )


# ═══════════════════════════════════════════════════════════════════
# 7. GPU Memory
# ═══════════════════════════════════════════════════════════════════

class TestGPUMemory:
    """Verify GPU memory tracking is sane."""

    def test_gpu_memory_reported(self) -> None:
        cfg = _make_config(n_steps=3, diagnostics_interval=1)
        solver = _build_solver(cfg)
        diags = _run_n_steps(solver, 3)
        for d in diags:
            assert d["gpu_mem_mb"] >= 0.0
            assert d["gpu_peak_mb"] >= d["gpu_mem_mb"]

    def test_memory_does_not_grow_unboundedly(self) -> None:
        """Memory at step N should be within 3× of step 1."""
        n = 10
        cfg = _make_config(n_steps=n, diagnostics_interval=1)
        solver = _build_solver(cfg)
        diags = _run_n_steps(solver, n)

        mem_first = diags[0]["gpu_mem_mb"]
        mem_last = diags[-1]["gpu_mem_mb"]

        if mem_first > 0.0:
            ratio = mem_last / mem_first
            assert ratio < 3.0, (
                f"GPU memory grew {ratio:.1f}× from "
                f"{mem_first:.0f} MB to {mem_last:.0f} MB"
            )


# ═══════════════════════════════════════════════════════════════════
# 8. Smagorinsky Parameter Sensitivity
# ═══════════════════════════════════════════════════════════════════

class TestSmagorinskyParameterSensitivity:
    """Verify Cs parameter actually affects the solve."""

    def test_different_cs_yields_different_energy(self) -> None:
        energies = {}
        for cs in (0.1, 0.3):
            cfg = _make_config(
                n_steps=FEW_STEPS,
                smagorinsky_cs=cs,
                diagnostics_interval=1,
            )
            solver = _build_solver(cfg)
            diags = _run_n_steps(solver, FEW_STEPS)
            energies[cs] = diags[-1]["energy"]

        # Different Cs should produce measurably different viscosity → energy
        e_low = energies[0.1]
        e_high = energies[0.3]
        assert e_low != e_high, (
            "Cs=0.1 and Cs=0.3 produced identical final energies — "
            "Smagorinsky coefficient may not be wired correctly"
        )


# ═══════════════════════════════════════════════════════════════════
# 9. Benchmark Timing
# ═══════════════════════════════════════════════════════════════════

class TestBenchmarkTiming:
    """Measure and assert wall-clock performance bounds."""

    def _timed_steps(
        self, n_bits: int, max_rank: int, n_steps: int, integrator: str = "rk2"
    ) -> float:
        """Return wall-clock seconds for *n_steps* at given resolution."""
        cfg = _make_config(
            n_bits=n_bits,
            max_rank=max_rank,
            n_steps=n_steps,
            integrator=integrator,
            diagnostics_interval=max(1, n_steps),  # minimize overhead
        )
        solver = _build_solver(cfg)
        # warm-up: 1 step (Triton compilation, cuDNN plans)
        solver.step()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            solver.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return elapsed

    def test_32_cube_step_under_30s(self) -> None:
        """5 steps at 32³ should complete in < 30 seconds (post warm-up).
        Note: Triton JIT compilation in the warm-up step adds ~10s on first
        invocation per unique kernel configuration. Steady-state per-step
        cost is ~3s at 32³/rank-16."""
        elapsed = self._timed_steps(
            n_bits=5, max_rank=16, n_steps=5, integrator="rk2"
        )
        assert elapsed < 30.0, (
            f"32³ / rank-16 / 5 steps took {elapsed:.2f}s (limit 30s)"
        )

    def test_rk2_not_more_than_3x_euler(self) -> None:
        """RK2 does 2 RHS evaluations per step, so should be ≤ 3× Euler."""
        n = 5
        t_euler = self._timed_steps(5, 16, n, "euler")
        t_rk2 = self._timed_steps(5, 16, n, "rk2")
        if t_euler > 0:
            ratio = t_rk2 / t_euler
            assert ratio < 3.0, (
                f"RK2 is {ratio:.1f}× slower than Euler (limit 3×)"
            )

    def test_step_timing_consistent(self) -> None:
        """Individual step times should not vary more than 5× from median."""
        cfg = _make_config(n_bits=5, max_rank=16, n_steps=8, diagnostics_interval=8)
        solver = _build_solver(cfg)
        # warm-up
        solver.step()
        torch.cuda.synchronize()

        times = []
        for _ in range(8):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            solver.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        times.sort()
        median = times[len(times) // 2]
        for t in times:
            assert t < median * 5.0, (
                f"Step took {t:.3f}s vs median {median:.3f}s — "
                f"exceeds 5× variance bound"
            )


# ═══════════════════════════════════════════════════════════════════
# 10. End-to-End run() API
# ═══════════════════════════════════════════════════════════════════

class TestRunAPI:
    """Validate the high-level run() API."""

    def test_run_returns_list_of_dicts(self) -> None:
        cfg = _make_config(n_steps=FEW_STEPS, diagnostics_interval=1)
        solver = _build_solver(cfg)
        history = solver.run(verbose=False)
        assert isinstance(history, list)
        assert len(history) == FEW_STEPS
        for d in history:
            assert isinstance(d, dict)

    def test_run_step_indices_sequential(self) -> None:
        cfg = _make_config(n_steps=FEW_STEPS, diagnostics_interval=1)
        solver = _build_solver(cfg)
        history = solver.run(verbose=False)
        steps = [d["step"] for d in history]
        assert steps == list(range(1, FEW_STEPS + 1))

    def test_run_time_monotonically_increases(self) -> None:
        cfg = _make_config(n_steps=FEW_STEPS, diagnostics_interval=1)
        solver = _build_solver(cfg)
        history = solver.run(verbose=False)
        times = [d["time"] for d in history]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1], (
                f"Time did not increase: step {i} t={times[i - 1]:.6e} "
                f"→ step {i + 1} t={times[i]:.6e}"
            )


# ═══════════════════════════════════════════════════════════════════
# 11. Edge Cases & Robustness
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_single_step_works(self) -> None:
        cfg = _make_config(n_steps=1, diagnostics_interval=1)
        solver = _build_solver(cfg)
        diag = solver.step()
        assert diag["step"] == 1

    def test_solver_state_persists_across_steps(self) -> None:
        """Energy at step 2 should reflect changes from step 1."""
        cfg = _make_config(n_steps=2, diagnostics_interval=1)
        solver = _build_solver(cfg)
        d1 = solver.step()
        d2 = solver.step()
        # d2 should differ from d1 (time evolution)
        assert d2["time"] > d1["time"]
        # energy may differ (unless very early transient)
        assert d2["step"] == d1["step"] + 1

    def test_very_low_cfl(self) -> None:
        """Extremely low CFL should still produce valid results."""
        cfg = _make_config(n_steps=3, cfl=0.01, diagnostics_interval=1)
        solver = _build_solver(cfg)
        diags = _run_n_steps(solver, 3)
        for d in diags:
            assert d["energy"] > 0.0
            assert math.isfinite(d["energy"])


# ═══════════════════════════════════════════════════════════════════
# 12. Multi-Resolution Consistency
# ═══════════════════════════════════════════════════════════════════

class TestMultiResolutionConsistency:
    """Verify physics plausibility across resolutions (32³ vs 64³)."""

    @pytest.mark.parametrize("n_bits", [5, 6])
    def test_energy_positive_across_resolutions(self, n_bits: int) -> None:
        cfg = _make_config(n_bits=n_bits, n_steps=3, diagnostics_interval=1)
        solver = _build_solver(cfg)
        diags = _run_n_steps(solver, 3)
        for d in diags:
            assert d["energy"] > 0.0

    def test_higher_resolution_has_higher_energy(self) -> None:
        """Energy at finer resolution (larger N³ domain) should be higher
        because kinetic energy ~ N³ at fixed velocity."""
        energies = {}
        for n_bits in (5, 6):
            cfg = _make_config(
                n_bits=n_bits,
                max_rank=SMALL_MAX_RANK,
                n_steps=1,
                diagnostics_interval=1,
            )
            solver = _build_solver(cfg)
            d = solver.step()
            energies[n_bits] = d["energy"]

        # 64³ should have ~8× more energy than 32³ (same velocity, 8× cells)
        # Allow wide tolerance — QTT rank truncation affects this
        assert energies[6] > energies[5], (
            f"64³ energy ({energies[6]:.4e}) should exceed "
            f"32³ energy ({energies[5]:.4e})"
        )
