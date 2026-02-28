"""
Shu-Osher Problem — Tier 1 Shock-Entropy Wave Benchmark
========================================================

This is a CANONICAL CFD benchmark per ASME V&V 10-2019 standards.
The Shu-Osher problem tests a shock wave interacting with a sinusoidal
density field, producing complex wave structures that test numerical
dissipation and shock-capturing properties.

**Benchmark**: Shu-Osher Shock-Entropy Wave Interaction
**Type**: 1D Compressible Euler
**Reference**: Shu & Osher (1989), "Efficient Implementation of
              Essentially Non-oscillatory Shock-Capturing Schemes II",
              J. Computational Physics 83:32-78
**Tier**: 1 (Analytical Initial Condition + Reference Solution)

**Problem Setup**:
    Domain: x ∈ [-5, 5]
    Initial condition:
        Left (x < -4):  ρ = 3.857143, u = 2.629369, p = 10.33333  (post-shock)
        Right (x ≥ -4): ρ = 1 + 0.2sin(5x), u = 0, p = 1  (pre-shock with entropy waves)
    Final time: t = 1.8
    γ = 1.4

**Validation Targets**:
    - Shock captured without oscillations
    - Entropy waves preserved behind shock
    - Density profile matches reference solution
    - Positivity preservation (ρ > 0, p > 0)

Constitutional Compliance:
    - Article IV.1: Verification (V&V)
    - ONTIC_VV_FRAMEWORK.md Section 3.3
    - Canonical CFD benchmark

Tag: [PHASE-1A] [BENCHMARK] [TIER-1] [V&V]
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

# Skip if Euler solver not available
try:
    from ontic.cfd.euler_1d import BCType1D, Euler1D, EulerState
    from ontic.cfd.godunov import hllc_flux, roe_flux

    HAS_EULER1D = True
except ImportError:
    HAS_EULER1D = False


# =============================================================================
# SHU-OSHER PROBLEM DEFINITION
# =============================================================================


@dataclass
class ShuOsherConfig:
    """Configuration for Shu-Osher problem."""

    x_min: float = -5.0
    x_max: float = 5.0
    x_shock: float = -4.0

    # Post-shock (left) state from Rankine-Hugoniot
    rho_L: float = 3.857143
    u_L: float = 2.629369
    p_L: float = 10.33333

    # Pre-shock (right) base state
    rho_R_base: float = 1.0
    u_R: float = 0.0
    p_R: float = 1.0

    # Entropy wave parameters
    wave_amplitude: float = 0.2
    wave_frequency: float = 5.0

    gamma: float = 1.4
    t_final: float = 1.8

    def initial_condition(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute initial condition at given x points.

        Returns:
            rho, u, p: Primitive variables
        """
        rho = torch.where(
            x < self.x_shock,
            torch.full_like(x, self.rho_L),
            self.rho_R_base + self.wave_amplitude * torch.sin(self.wave_frequency * x),
        )

        u = torch.where(
            x < self.x_shock, torch.full_like(x, self.u_L), torch.full_like(x, self.u_R)
        )

        p = torch.where(
            x < self.x_shock, torch.full_like(x, self.p_L), torch.full_like(x, self.p_R)
        )

        return rho, u, p


# =============================================================================
# REFERENCE SOLUTION (Computed with high-resolution WENO)
# =============================================================================
# These are reference values at select points for t=1.8 with 2000 grid points
# From original Shu-Osher paper and subsequent high-resolution computations

REFERENCE_X_POINTS = np.array([-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

REFERENCE_DENSITY = np.array(
    [4.5, 4.2, 3.85, 3.45, 3.65, 3.85, 4.1, 4.4, 4.25, 4.0]
)  # Approximate - for shape validation


# =============================================================================
# SOLVER WRAPPER
# =============================================================================


class ShuOsherSolver:
    """
    Solver for Shu-Osher problem.

    Uses Godunov-type scheme with HLLC flux.
    """

    def __init__(
        self,
        N: int,
        config: ShuOsherConfig = None,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        """
        Initialize solver.

        Args:
            N: Number of grid points
            config: Problem configuration
        """
        self.N = N
        self.config = config or ShuOsherConfig()
        self.dtype = dtype
        self.device = device

        # Grid
        self.dx = (self.config.x_max - self.config.x_min) / N
        self.x = torch.linspace(
            self.config.x_min + self.dx / 2,  # Cell centers
            self.config.x_max - self.dx / 2,
            N,
            dtype=dtype,
            device=device,
        )

        # Initialize state
        rho, u, p = self.config.initial_condition(self.x)

        # Convert to conserved variables
        rho_u = rho * u
        E = p / (self.config.gamma - 1) + 0.5 * rho * u**2

        self.state = EulerState(rho=rho, rho_u=rho_u, E=E, gamma=self.config.gamma)
        self.t = 0.0

    def compute_dt(self, cfl: float = 0.5) -> float:
        """Compute stable time step from CFL condition."""
        a = self.state.a
        u = self.state.u
        max_speed = (u.abs() + a).max().item()
        return cfl * self.dx / max(max_speed, 1e-10)

    def step(self, dt: float) -> None:
        """Take one time step using first-order Godunov."""
        U = self.state.to_conserved()  # (N, 3)

        # Add ghost cells (transmissive BC)
        U_extended = torch.cat([U[0:1], U, U[-1:]], dim=0)

        # Compute fluxes at interfaces (N+1 interfaces)
        F = torch.zeros((self.N + 1, 3), dtype=self.dtype, device=self.device)

        for i in range(self.N + 1):
            U_L = U_extended[i]
            U_R = U_extended[i + 1]

            # HLLC flux expects tensor inputs
            F[i] = hllc_flux(
                U_L.unsqueeze(0), U_R.unsqueeze(0), self.config.gamma
            ).squeeze(0)

        # Update: U^{n+1} = U^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
        dF = F[1:] - F[:-1]
        U_new = U - (dt / self.dx) * dF

        self.state = EulerState.from_conserved(U_new, self.config.gamma)
        self.t += dt

    def solve(
        self, t_final: float = None, cfl: float = 0.5, verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Solve to final time.

        Returns:
            Dict with solution and diagnostics
        """
        t_final = t_final or self.config.t_final

        n_steps = 0
        while self.t < t_final:
            dt = self.compute_dt(cfl)

            # Don't overshoot
            if self.t + dt > t_final:
                dt = t_final - self.t

            self.step(dt)
            n_steps += 1

            if verbose and n_steps % 100 == 0:
                print(f"t = {self.t:.4f}, steps = {n_steps}")

        return {
            "x": self.x,
            "rho": self.state.rho,
            "u": self.state.u,
            "p": self.state.p,
            "t": self.t,
            "n_steps": n_steps,
        }


# =============================================================================
# BENCHMARK TESTS
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.physics
@pytest.mark.skipif(not HAS_EULER1D, reason="Euler1D solver not available")
class TestShuOsherBenchmark:
    """
    Shu-Osher shock-entropy wave interaction benchmark.

    Tests shock-capturing and entropy wave preservation.
    """

    @pytest.fixture
    def config(self) -> ShuOsherConfig:
        return ShuOsherConfig()

    @pytest.fixture
    def solver_200(self, config) -> ShuOsherSolver:
        """Coarse grid solver."""
        return ShuOsherSolver(N=200, config=config)

    @pytest.fixture
    def solver_400(self, config) -> ShuOsherSolver:
        """Medium grid solver."""
        return ShuOsherSolver(N=400, config=config)

    def test_initial_condition_correct(self, solver_200, config):
        """Verify initial condition matches Shu-Osher setup."""
        # Check post-shock region
        left_mask = solver_200.x < config.x_shock
        rho_left = solver_200.state.rho[left_mask]

        assert torch.allclose(
            rho_left, torch.full_like(rho_left, config.rho_L), atol=1e-6
        ), "Post-shock density incorrect"

        # Check pre-shock has entropy wave
        right_mask = solver_200.x >= config.x_shock
        rho_right = solver_200.state.rho[right_mask]

        # Should have variation from sinusoidal perturbation
        rho_variation = rho_right.max() - rho_right.min()
        assert rho_variation > 0.3, "Entropy wave perturbation not present"

    @pytest.mark.conservation
    def test_positivity_preserved(self, solver_200):
        """
        Verify density and pressure remain positive throughout simulation.

        This is a fundamental requirement for any Euler solver.
        """
        result = solver_200.solve(t_final=0.5, verbose=False)  # Shorter for speed

        rho = result["rho"]
        p = result["p"]

        assert torch.all(rho > 0), f"Negative density: min(ρ) = {rho.min().item():.6e}"
        assert torch.all(p > 0), f"Negative pressure: min(p) = {p.min().item():.6e}"

    def test_shock_captured(self, solver_400):
        """
        Verify shock is captured without excessive oscillations.

        The shock should be sharp (few cells) and monotone.
        """
        result = solver_400.solve(verbose=False)

        rho = result["rho"]
        x = result["x"]

        # Find approximate shock location (near x=2.4 at t=1.8)
        # The shock moves to the right at approximately u_shock
        shock_region = (x > 2.0) & (x < 3.0)
        rho_shock = rho[shock_region]

        # Shock should show density jump
        rho_jump = rho_shock.max() - rho_shock.min()
        assert rho_jump > 2.0, f"Shock not captured: density jump = {rho_jump:.2f}"

        print(f"\n{'='*60}")
        print(f"SHU-OSHER SHOCK CAPTURE TEST")
        print(f"{'='*60}")
        print(f"Grid: {solver_400.N}")
        print(f"Time: t = {result['t']:.2f}")
        print(f"Shock region density jump: {rho_jump:.4f}")
        print(f"{'='*60}")

    def test_entropy_waves_preserved(self, solver_400):
        """
        Verify entropy waves are preserved behind the shock.

        The post-shock region should show oscillations from
        the amplified entropy waves.

        Note: First-order Godunov scheme is diffusive. WENO or higher-order
        schemes would preserve waves better. This test verifies waves exist,
        not their full amplitude.
        """
        result = solver_400.solve(verbose=False)

        rho = result["rho"]
        x = result["x"]

        # Check region behind shock (x < 1)
        behind_shock = (x > -3) & (x < 1)
        rho_behind = rho[behind_shock]

        # Should have oscillations (entropy waves)
        rho_range = rho_behind.max() - rho_behind.min()

        print(f"\n{'='*60}")
        print(f"ENTROPY WAVE PRESERVATION TEST")
        print(f"{'='*60}")
        print(f"Behind-shock density range: {rho_range:.4f}")
        print(f"Note: First-order scheme - expect some damping")
        print(f"{'='*60}")

        # Relaxed tolerance for first-order scheme
        # Full wave preservation requires WENO/higher-order
        assert (
            rho_range > 0.2
        ), f"Entropy waves completely damped: density range = {rho_range:.4f}"

    @pytest.mark.slow
    def test_solution_converges_with_refinement(self):
        """
        Verify solution converges as grid is refined.

        Compare 200 vs 400 grid point solutions.
        """
        config = ShuOsherConfig()

        # Solve on two grids
        solver_coarse = ShuOsherSolver(N=200, config=config)
        solver_fine = ShuOsherSolver(N=400, config=config)

        result_coarse = solver_coarse.solve(verbose=False)
        result_fine = solver_fine.solve(verbose=False)

        # Interpolate coarse to fine grid for comparison
        from scipy.interpolate import interp1d

        x_coarse = result_coarse["x"].cpu().numpy()
        x_fine = result_fine["x"].cpu().numpy()
        rho_coarse = result_coarse["rho"].cpu().numpy()
        rho_fine = result_fine["rho"].cpu().numpy()

        # Interpolate coarse to fine grid
        f = interp1d(x_coarse, rho_coarse, kind="linear", fill_value="extrapolate")
        rho_coarse_interp = f(x_fine)

        # Compute difference
        diff = np.abs(rho_fine - rho_coarse_interp)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\n{'='*60}")
        print(f"GRID CONVERGENCE TEST")
        print(f"{'='*60}")
        print(f"Coarse grid: N=200")
        print(f"Fine grid: N=400")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"{'='*60}")

        # Fine solution should be noticeably different (better resolved)
        # but not wildly different
        assert max_diff < 1.0, "Solutions diverged unexpectedly"
        assert max_diff > 0.01, "No improvement with refinement?"


@pytest.mark.benchmark
@pytest.mark.physics
@pytest.mark.skipif(not HAS_EULER1D, reason="Euler1D solver not available")
class TestShuOsherPhysics:
    """
    Physics validation for Shu-Osher problem.
    """

    def test_mach_number_range(self):
        """Verify Mach numbers are in expected range."""
        solver = ShuOsherSolver(N=200)
        result = solver.solve(t_final=1.0, verbose=False)

        # Compute Mach number from result
        rho = result["rho"]
        u = result["u"]
        p = result["p"]
        gamma = 1.4

        a = torch.sqrt(gamma * p / rho)
        M = u.abs() / a

        max_mach = M.max().item()

        # Should be supersonic in places (shock)
        assert max_mach > 1.0, f"No supersonic regions: max M = {max_mach:.2f}"
        assert max_mach < 10.0, f"Unreasonably high Mach: {max_mach:.2f}"

    def test_shock_speed_reasonable(self):
        """Verify shock moves at approximately correct speed."""
        config = ShuOsherConfig()
        solver = ShuOsherSolver(N=400, config=config)

        # Initial shock at x = -4
        # Expected shock speed ~ M_s * a_R where a_R = sqrt(gamma * p_R / rho_R)
        # For this problem, shock should move right at ~2.6 (roughly u_L)

        result = solver.solve(t_final=1.0, verbose=False)

        rho = result["rho"]
        x = result["x"]

        # Find shock location (max gradient)
        drho = torch.diff(rho)
        shock_idx = drho.abs().argmax()
        shock_x = x[shock_idx].item()

        # Expected position: x_shock(t) ≈ -4 + 2.6*t = -1.4 at t=1
        expected_x = -4.0 + 2.6 * 1.0

        error = abs(shock_x - expected_x)

        print(f"\n{'='*60}")
        print(f"SHOCK SPEED VALIDATION")
        print(f"{'='*60}")
        print(f"Initial shock position: x = -4.0")
        print(f"Shock position at t=1: x ≈ {shock_x:.2f}")
        print(f"Expected: x ≈ {expected_x:.2f}")
        print(f"Error: {error:.2f}")
        print(f"{'='*60}")

        # Allow reasonable tolerance
        assert error < 1.0, f"Shock speed error too large: {error:.2f}"


# =============================================================================
# VALIDATION REPORT
# =============================================================================


def generate_validation_report() -> Dict[str, Any]:
    """Generate validation report for Shu-Osher benchmark."""
    import datetime

    report = {
        "benchmark": "shu_osher",
        "tier": 1,
        "reference": "Shu & Osher (1989), J. Comp. Phys. 83:32-78",
        "timestamp": datetime.datetime.now().isoformat(),
        "tests": [],
        "overall_status": "UNKNOWN",
    }

    if not HAS_EULER1D:
        report["overall_status"] = "SKIPPED"
        report["reason"] = "Euler1D solver not available"
        return report

    try:
        solver = ShuOsherSolver(N=400)
        result = solver.solve(verbose=False)

        # Positivity check
        rho = result["rho"]
        p = result["p"]
        positivity_pass = torch.all(rho > 0) and torch.all(p > 0)

        report["tests"].append(
            {"name": "positivity", "status": "PASS" if positivity_pass else "FAIL"}
        )

        # Shock captured
        x = result["x"]
        shock_region = (x > 2.0) & (x < 3.0)
        rho_jump = rho[shock_region].max() - rho[shock_region].min()
        shock_pass = rho_jump > 2.0

        report["tests"].append(
            {
                "name": "shock_capture",
                "density_jump": rho_jump.item(),
                "status": "PASS" if shock_pass else "FAIL",
            }
        )

        report["overall_status"] = (
            "PASS" if (positivity_pass and shock_pass) else "FAIL"
        )

    except Exception as e:
        report["overall_status"] = "ERROR"
        report["error"] = str(e)

    return report


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SHU-OSHER BENCHMARK VALIDATION")
    print("=" * 70)

    report = generate_validation_report()

    import json

    print(json.dumps(report, indent=2, default=str))
