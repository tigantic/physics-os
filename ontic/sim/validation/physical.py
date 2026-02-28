"""
Physical Validation Module for Project The Ontic Engine.

Provides physics-based validation tests including:
- Conservation law verification (mass, momentum, energy)
- Analytical solution comparisons
- Benchmark problem validation

These tests ensure numerical methods satisfy fundamental physical principles.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ValidationSeverity(Enum):
    """Severity level for validation results."""

    PASS = auto()
    WARNING = auto()
    FAIL = auto()
    CRITICAL = auto()


@dataclass
class ValidationResult:
    """
    Result of a single validation test.

    Attributes:
        test_name: Name of the validation test
        passed: Whether the test passed
        severity: Severity level of any issues
        metric_name: Name of the metric being tested
        computed_value: Computed value from simulation
        expected_value: Expected/reference value
        tolerance: Tolerance used for comparison
        relative_error: Relative error if applicable
        absolute_error: Absolute error if applicable
        message: Human-readable result message
        details: Additional test details
    """

    test_name: str
    passed: bool
    severity: ValidationSeverity
    metric_name: str
    computed_value: float
    expected_value: float
    tolerance: float
    relative_error: float | None = None
    absolute_error: float | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "severity": self.severity.name,
            "metric_name": self.metric_name,
            "computed_value": float(self.computed_value),
            "expected_value": float(self.expected_value),
            "tolerance": float(self.tolerance),
            "relative_error": (
                float(self.relative_error) if self.relative_error else None
            ),
            "absolute_error": (
                float(self.absolute_error) if self.absolute_error else None
            ),
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """
    Comprehensive validation report containing multiple test results.

    Attributes:
        title: Report title
        timestamp: When the validation was run
        results: List of individual validation results
        summary: Summary statistics
        configuration: Test configuration used
    """

    title: str
    timestamp: float
    results: list[ValidationResult]
    summary: dict[str, int] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute summary statistics."""
        if not self.summary:
            self.summary = self._compute_summary()

    def _compute_summary(self) -> dict[str, int]:
        """Compute summary statistics from results."""
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "warnings": sum(
                1 for r in self.results if r.severity == ValidationSeverity.WARNING
            ),
            "critical": sum(
                1 for r in self.results if r.severity == ValidationSeverity.CRITICAL
            ),
        }

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return all(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        """Compute pass rate as percentage."""
        if not self.results:
            return 100.0
        return 100.0 * self.summary["passed"] / self.summary["total"]

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# {self.title}",
            "",
            f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"**Pass Rate**: {self.pass_rate:.1f}% ({self.summary['passed']}/{self.summary['total']})",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tests | {self.summary['total']} |",
            f"| Passed | {self.summary['passed']} |",
            f"| Failed | {self.summary['failed']} |",
            f"| Warnings | {self.summary['warnings']} |",
            f"| Critical | {self.summary['critical']} |",
            "",
            "## Detailed Results",
            "",
        ]

        for result in self.results:
            status = "✅" if result.passed else "❌"
            lines.append(f"### {status} {result.test_name}")
            lines.append("")
            lines.append(f"- **Metric**: {result.metric_name}")
            lines.append(f"- **Computed**: {result.computed_value:.6e}")
            lines.append(f"- **Expected**: {result.expected_value:.6e}")
            lines.append(f"- **Tolerance**: {result.tolerance:.6e}")
            if result.relative_error is not None:
                lines.append(f"- **Relative Error**: {result.relative_error:.6e}")
            if result.message:
                lines.append(f"- **Message**: {result.message}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "configuration": self.configuration,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, filepath: str | Path):
        """Save report to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ConservationValidator(ABC):
    """
    Abstract base class for conservation law validators.

    Conservation laws are fundamental physical principles that must be
    satisfied by any valid numerical simulation.
    """

    def __init__(
        self,
        tolerance: float = 1e-10,
        relative: bool = True,
    ):
        """
        Initialize conservation validator.

        Args:
            tolerance: Tolerance for conservation check
            relative: Use relative (True) or absolute (False) tolerance
        """
        self.tolerance = tolerance
        self.relative = relative

    @abstractmethod
    def compute_conserved_quantity(
        self,
        state: torch.Tensor,
        **kwargs,
    ) -> float:
        """
        Compute the conserved quantity from the state.

        Args:
            state: Current simulation state
            **kwargs: Additional parameters (geometry, etc.)

        Returns:
            Value of the conserved quantity
        """
        pass

    def validate(
        self,
        initial_state: torch.Tensor,
        final_state: torch.Tensor,
        test_name: str = "Conservation",
        **kwargs,
    ) -> ValidationResult:
        """
        Validate conservation between initial and final states.

        Args:
            initial_state: State at beginning
            final_state: State at end
            test_name: Name for this validation test
            **kwargs: Additional parameters

        Returns:
            ValidationResult with conservation check outcome
        """
        initial_value = self.compute_conserved_quantity(initial_state, **kwargs)
        final_value = self.compute_conserved_quantity(final_state, **kwargs)

        abs_error = abs(final_value - initial_value)

        if self.relative and abs(initial_value) > 1e-15:
            rel_error = abs_error / abs(initial_value)
            passed = rel_error < self.tolerance
            error_value = rel_error
        else:
            passed = abs_error < self.tolerance
            error_value = abs_error
            rel_error = None

        if passed:
            severity = ValidationSeverity.PASS
            message = "Conservation maintained within tolerance"
        elif error_value < 10 * self.tolerance:
            severity = ValidationSeverity.WARNING
            message = "Conservation marginally violated"
        else:
            severity = ValidationSeverity.FAIL
            message = "Conservation law violated"

        return ValidationResult(
            test_name=test_name,
            passed=passed,
            severity=severity,
            metric_name="Conserved Quantity",
            computed_value=final_value,
            expected_value=initial_value,
            tolerance=self.tolerance,
            relative_error=rel_error,
            absolute_error=abs_error,
            message=message,
            details={
                "initial_value": float(initial_value),
                "final_value": float(final_value),
                "relative": self.relative,
            },
        )


class MassConservationTest(ConservationValidator):
    """
    Validator for mass conservation in CFD simulations.

    For compressible flow, total mass = ∫ρ dV should be conserved
    in the absence of mass sources/sinks.
    """

    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        tolerance: float = 1e-10,
    ):
        """
        Initialize mass conservation validator.

        Args:
            dx, dy, dz: Grid spacing in each dimension
            tolerance: Conservation tolerance
        """
        super().__init__(tolerance=tolerance, relative=True)
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def compute_conserved_quantity(
        self,
        state: torch.Tensor,
        **kwargs,
    ) -> float:
        """
        Compute total mass from density field.

        Args:
            state: State tensor with density in first channel
                   Shape: (nvar, nx) for 1D, (nvar, nx, ny) for 2D, etc.

        Returns:
            Total mass in domain
        """
        # Extract density (first conserved variable)
        if state.dim() == 2:
            rho = state[0, :]  # 1D: (nvar, nx)
            dV = self.dx
        elif state.dim() == 3:
            rho = state[0, :, :]  # 2D: (nvar, nx, ny)
            dV = self.dx * self.dy
        elif state.dim() == 4:
            rho = state[0, :, :, :]  # 3D: (nvar, nx, ny, nz)
            dV = self.dx * self.dy * self.dz
        else:
            rho = state
            dV = self.dx

        return float(torch.sum(rho) * dV)


class MomentumConservationTest(ConservationValidator):
    """
    Validator for momentum conservation in CFD simulations.

    Total momentum = ∫ρu dV should be conserved in the absence
    of external forces and with appropriate boundary conditions.
    """

    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        component: int = 0,
        tolerance: float = 1e-10,
    ):
        """
        Initialize momentum conservation validator.

        Args:
            dx, dy, dz: Grid spacing
            component: Momentum component to check (0=x, 1=y, 2=z)
            tolerance: Conservation tolerance
        """
        super().__init__(tolerance=tolerance, relative=True)
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.component = component

    def compute_conserved_quantity(
        self,
        state: torch.Tensor,
        **kwargs,
    ) -> float:
        """
        Compute total momentum from state.

        Args:
            state: Conservative state tensor
                   For Euler: (rho, rho*u, rho*v, rho*w, E)

        Returns:
            Total momentum component
        """
        momentum_idx = 1 + self.component

        if state.dim() == 2:
            momentum = state[momentum_idx, :]
            dV = self.dx
        elif state.dim() == 3:
            momentum = state[momentum_idx, :, :]
            dV = self.dx * self.dy
        else:
            momentum = state[momentum_idx, :, :, :]
            dV = self.dx * self.dy * self.dz

        return float(torch.sum(momentum) * dV)


class EnergyConservationTest(ConservationValidator):
    """
    Validator for total energy conservation in CFD simulations.

    Total energy = ∫E dV should be conserved in inviscid flow
    with adiabatic boundaries.
    """

    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        energy_index: int = -1,
        tolerance: float = 1e-10,
    ):
        """
        Initialize energy conservation validator.

        Args:
            dx, dy, dz: Grid spacing
            energy_index: Index of energy in state vector (-1 for last)
            tolerance: Conservation tolerance
        """
        super().__init__(tolerance=tolerance, relative=True)
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.energy_index = energy_index

    def compute_conserved_quantity(
        self,
        state: torch.Tensor,
        **kwargs,
    ) -> float:
        """
        Compute total energy from state.

        Args:
            state: Conservative state tensor with energy as last variable

        Returns:
            Total energy in domain
        """
        if state.dim() == 2:
            energy = state[self.energy_index, :]
            dV = self.dx
        elif state.dim() == 3:
            energy = state[self.energy_index, :, :]
            dV = self.dx * self.dy
        else:
            energy = state[self.energy_index, :, :, :]
            dV = self.dx * self.dy * self.dz

        return float(torch.sum(energy) * dV)


class AnalyticalValidator(ABC):
    """
    Abstract base class for validation against analytical solutions.

    These validators compare numerical results against known exact solutions
    to verify accuracy and convergence rates.
    """

    def __init__(
        self,
        tolerance: float = 1e-3,
        error_norm: str = "L2",
    ):
        """
        Initialize analytical validator.

        Args:
            tolerance: Maximum acceptable error
            error_norm: Error norm to use (L1, L2, Linf)
        """
        self.tolerance = tolerance
        self.error_norm = error_norm

    @abstractmethod
    def compute_analytical_solution(
        self,
        x: torch.Tensor,
        t: float,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the analytical solution.

        Args:
            x: Spatial coordinates
            t: Time
            **kwargs: Problem parameters

        Returns:
            Analytical solution at given points and time
        """
        pass

    def compute_error(
        self,
        numerical: torch.Tensor,
        analytical: torch.Tensor,
    ) -> float:
        """
        Compute error between numerical and analytical solutions.

        Args:
            numerical: Numerical solution
            analytical: Analytical solution

        Returns:
            Error in specified norm
        """
        diff = numerical - analytical

        if self.error_norm == "L1":
            return float(torch.mean(torch.abs(diff)))
        elif self.error_norm == "L2":
            return float(torch.sqrt(torch.mean(diff**2)))
        elif self.error_norm == "Linf":
            return float(torch.max(torch.abs(diff)))
        else:
            raise ValueError(f"Unknown error norm: {self.error_norm}")

    def validate(
        self,
        numerical: torch.Tensor,
        x: torch.Tensor,
        t: float,
        test_name: str = "Analytical Comparison",
        **kwargs,
    ) -> ValidationResult:
        """
        Validate numerical solution against analytical.

        Args:
            numerical: Numerical solution
            x: Spatial coordinates
            t: Time
            test_name: Name for this validation
            **kwargs: Problem parameters

        Returns:
            ValidationResult with comparison outcome
        """
        analytical = self.compute_analytical_solution(x, t, **kwargs)
        error = self.compute_error(numerical, analytical)

        passed = error < self.tolerance

        if passed:
            severity = ValidationSeverity.PASS
            message = "Error within tolerance"
        elif error < 10 * self.tolerance:
            severity = ValidationSeverity.WARNING
            message = "Error marginally exceeds tolerance"
        else:
            severity = ValidationSeverity.FAIL
            message = "Error significantly exceeds tolerance"

        return ValidationResult(
            test_name=test_name,
            passed=passed,
            severity=severity,
            metric_name=f"{self.error_norm} Error",
            computed_value=error,
            expected_value=0.0,
            tolerance=self.tolerance,
            relative_error=None,
            absolute_error=error,
            message=message,
            details={
                "error_norm": self.error_norm,
            },
        )


class SodShockValidator(AnalyticalValidator):
    """
    Validator for Sod shock tube problem.

    The Sod shock tube is a 1D Riemann problem with known analytical solution,
    widely used for CFD code verification.
    """

    def __init__(
        self,
        gamma: float = 1.4,
        tolerance: float = 0.01,
        x_discontinuity: float = 0.5,
    ):
        """
        Initialize Sod shock tube validator.

        Args:
            gamma: Ratio of specific heats
            tolerance: Maximum L2 error
            x_discontinuity: Initial discontinuity location
        """
        super().__init__(tolerance=tolerance, error_norm="L2")
        self.gamma = gamma
        self.x_discontinuity = x_discontinuity

        # Initial conditions
        self.rho_L, self.u_L, self.p_L = 1.0, 0.0, 1.0
        self.rho_R, self.u_R, self.p_R = 0.125, 0.0, 0.1

    def compute_analytical_solution(
        self,
        x: torch.Tensor,
        t: float,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute Sod shock tube analytical solution.

        Uses the exact Riemann solver for the shock tube problem.

        Args:
            x: Spatial coordinates
            t: Time

        Returns:
            Primitive state (rho, u, p) at each point
        """
        gamma = self.gamma

        # Solve for post-shock state iteratively
        # Using Newton-Raphson to find p*
        p_star = self._solve_for_pstar()

        # Compute wave speeds and states
        a_L = np.sqrt(gamma * self.p_L / self.rho_L)
        a_R = np.sqrt(gamma * self.p_R / self.rho_R)

        # Star region velocity
        u_star = 0.5 * (self.u_L + self.u_R) + 0.5 * (
            self._f_pressure(p_star, self.rho_R, self.p_R, gamma)
            - self._f_pressure(p_star, self.rho_L, self.p_L, gamma)
        )

        # Densities in star region
        if p_star > self.p_L:  # Left shock
            rho_star_L = self.rho_L * (
                (p_star / self.p_L + (gamma - 1) / (gamma + 1))
                / ((gamma - 1) / (gamma + 1) * p_star / self.p_L + 1)
            )
        else:  # Left rarefaction
            rho_star_L = self.rho_L * (p_star / self.p_L) ** (1 / gamma)

        if p_star > self.p_R:  # Right shock
            rho_star_R = self.rho_R * (
                (p_star / self.p_R + (gamma - 1) / (gamma + 1))
                / ((gamma - 1) / (gamma + 1) * p_star / self.p_R + 1)
            )
        else:  # Right rarefaction
            rho_star_R = self.rho_R * (p_star / self.p_R) ** (1 / gamma)

        # Sample solution
        x_np = x.numpy() if isinstance(x, torch.Tensor) else x
        xi = (x_np - self.x_discontinuity) / t if t > 0 else np.zeros_like(x_np)

        rho = np.zeros_like(x_np)
        u = np.zeros_like(x_np)
        p = np.zeros_like(x_np)

        # Left state
        head_L = self.u_L - a_L if p_star <= self.p_L else self._shock_speed_L(p_star)

        # Right state
        head_R = self.u_R + a_R if p_star <= self.p_R else self._shock_speed_R(p_star)

        for i, xi_val in enumerate(xi):
            if xi_val < head_L:
                # Left original state
                rho[i] = self.rho_L
                u[i] = self.u_L
                p[i] = self.p_L
            elif xi_val < u_star:
                # Left of contact
                if p_star <= self.p_L:
                    # Rarefaction fan
                    a_star_L = a_L * (p_star / self.p_L) ** ((gamma - 1) / (2 * gamma))
                    tail_L = u_star - a_star_L
                    if xi_val < tail_L:
                        # Inside rarefaction
                        rho[i] = self.rho_L * (
                            (
                                2 / (gamma + 1)
                                + (gamma - 1)
                                / ((gamma + 1) * a_L)
                                * (self.u_L - xi_val)
                            )
                            ** (2 / (gamma - 1))
                        )
                        u[i] = (
                            2
                            / (gamma + 1)
                            * (a_L + (gamma - 1) / 2 * self.u_L + xi_val)
                        )
                        p[i] = self.p_L * (
                            (
                                2 / (gamma + 1)
                                + (gamma - 1)
                                / ((gamma + 1) * a_L)
                                * (self.u_L - xi_val)
                            )
                            ** (2 * gamma / (gamma - 1))
                        )
                    else:
                        rho[i] = rho_star_L
                        u[i] = u_star
                        p[i] = p_star
                else:
                    rho[i] = rho_star_L
                    u[i] = u_star
                    p[i] = p_star
            elif xi_val < head_R:
                # Right of contact, left of right wave
                rho[i] = rho_star_R
                u[i] = u_star
                p[i] = p_star
            else:
                # Right original state
                rho[i] = self.rho_R
                u[i] = self.u_R
                p[i] = self.p_R

        # Stack into state tensor
        result = torch.tensor(np.stack([rho, u, p], axis=0), dtype=torch.float64)
        return result

    def _f_pressure(self, p: float, rho: float, p_k: float, gamma: float) -> float:
        """Pressure function for Riemann solver."""
        A = 2 / ((gamma + 1) * rho)
        B = (gamma - 1) / (gamma + 1) * p_k
        a = np.sqrt(gamma * p_k / rho)

        if p > p_k:
            return (p - p_k) * np.sqrt(A / (p + B))
        else:
            return 2 * a / (gamma - 1) * ((p / p_k) ** ((gamma - 1) / (2 * gamma)) - 1)

    def _solve_for_pstar(self) -> float:
        """Solve for pressure in star region using Newton-Raphson."""
        gamma = self.gamma

        # Initial guess (linearized solution)
        a_L = np.sqrt(gamma * self.p_L / self.rho_L)
        a_R = np.sqrt(gamma * self.p_R / self.rho_R)

        p = 0.5 * (self.p_L + self.p_R)

        for _ in range(100):
            f = (
                self._f_pressure(p, self.rho_L, self.p_L, gamma)
                + self._f_pressure(p, self.rho_R, self.p_R, gamma)
                + self.u_R
                - self.u_L
            )

            df = self._df_pressure(p, self.rho_L, self.p_L, gamma) + self._df_pressure(
                p, self.rho_R, self.p_R, gamma
            )

            p_new = p - f / df

            if abs(p_new - p) < 1e-10:
                break
            p = max(p_new, 1e-10)

        return p

    def _df_pressure(self, p: float, rho: float, p_k: float, gamma: float) -> float:
        """Derivative of pressure function."""
        A = 2 / ((gamma + 1) * rho)
        B = (gamma - 1) / (gamma + 1) * p_k
        a = np.sqrt(gamma * p_k / rho)

        if p > p_k:
            return np.sqrt(A / (p + B)) * (1 - (p - p_k) / (2 * (p + B)))
        else:
            return 1 / (rho * a) * (p / p_k) ** (-(gamma + 1) / (2 * gamma))

    def _shock_speed_L(self, p_star: float) -> float:
        """Left shock speed."""
        gamma = self.gamma
        return self.u_L - np.sqrt(gamma * self.p_L / self.rho_L) * np.sqrt(
            (gamma + 1) / (2 * gamma) * p_star / self.p_L + (gamma - 1) / (2 * gamma)
        )

    def _shock_speed_R(self, p_star: float) -> float:
        """Right shock speed."""
        gamma = self.gamma
        return self.u_R + np.sqrt(gamma * self.p_R / self.rho_R) * np.sqrt(
            (gamma + 1) / (2 * gamma) * p_star / self.p_R + (gamma - 1) / (2 * gamma)
        )


class BlasiusValidator(AnalyticalValidator):
    """
    Validator for Blasius flat plate boundary layer.

    The Blasius solution is the similarity solution for laminar flow
    over a flat plate, providing exact velocity profiles.
    """

    def __init__(
        self,
        U_inf: float = 1.0,
        nu: float = 1e-4,
        tolerance: float = 0.02,
    ):
        """
        Initialize Blasius validator.

        Args:
            U_inf: Freestream velocity
            nu: Kinematic viscosity
            tolerance: Maximum acceptable error
        """
        super().__init__(tolerance=tolerance, error_norm="L2")
        self.U_inf = U_inf
        self.nu = nu

        # Pre-compute Blasius profile (using tabulated values)
        self._compute_blasius_profile()

    def _compute_blasius_profile(self):
        """Compute Blasius profile using shooting method."""
        from scipy.integrate import solve_ivp

        # Blasius ODE: f''' + 0.5 * f * f'' = 0
        # with f(0) = 0, f'(0) = 0, f'(inf) = 1

        def blasius_ode(eta, y):
            f, fp, fpp = y
            return [fp, fpp, -0.5 * f * fpp]

        # Shooting for f''(0)
        fpp0 = 0.332057336215196  # Known value

        eta_max = 10.0
        eta = np.linspace(0, eta_max, 1000)

        sol = solve_ivp(
            blasius_ode,
            [0, eta_max],
            [0, 0, fpp0],
            t_eval=eta,
            method="RK45",
        )

        self.eta_table = sol.t
        self.f_table = sol.y[0]
        self.fp_table = sol.y[1]  # u/U_inf
        self.fpp_table = sol.y[2]

    def compute_analytical_solution(
        self,
        x: torch.Tensor,
        t: float,
        y: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute Blasius velocity profile.

        Args:
            x: Streamwise coordinates
            t: Not used (steady flow)
            y: Wall-normal coordinates

        Returns:
            Velocity u/U_inf at each (x, y) point
        """
        if y is None:
            raise ValueError("y coordinates required for Blasius validation")

        x_np = x.numpy() if isinstance(x, torch.Tensor) else x
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y

        # Similarity variable: eta = y * sqrt(U_inf / (nu * x))
        eta = y_np * np.sqrt(self.U_inf / (self.nu * np.maximum(x_np, 1e-10)))

        # Interpolate from pre-computed profile
        u_over_Uinf = np.interp(eta.flatten(), self.eta_table, self.fp_table)
        u_over_Uinf = u_over_Uinf.reshape(eta.shape)

        return torch.tensor(u_over_Uinf * self.U_inf, dtype=torch.float64)


class ObliqueShockValidator(AnalyticalValidator):
    """
    Validator for oblique shock waves.

    Compares numerical shock angles and post-shock states against
    the θ-β-M analytical relations.
    """

    def __init__(
        self,
        gamma: float = 1.4,
        tolerance: float = 0.01,
    ):
        """
        Initialize oblique shock validator.

        Args:
            gamma: Ratio of specific heats
            tolerance: Maximum acceptable error
        """
        super().__init__(tolerance=tolerance, error_norm="Linf")
        self.gamma = gamma

    def compute_analytical_solution(
        self,
        x: torch.Tensor,
        t: float,
        M1: float = 2.0,
        theta: float = 10.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute post-shock state from oblique shock relations.

        Args:
            x: Not used directly
            t: Not used (steady flow)
            M1: Upstream Mach number
            theta: Wedge/deflection angle in degrees

        Returns:
            Post-shock state (rho2/rho1, p2/p1, M2)
        """
        theta_rad = np.radians(theta)
        gamma = self.gamma

        # Solve for shock angle β using Newton-Raphson
        beta = self._solve_for_beta(M1, theta_rad)

        # Normal component of Mach number
        M1n = M1 * np.sin(beta)

        # Pressure ratio
        p2_p1 = 1 + 2 * gamma / (gamma + 1) * (M1n**2 - 1)

        # Density ratio
        rho2_rho1 = ((gamma + 1) * M1n**2) / ((gamma - 1) * M1n**2 + 2)

        # Post-shock normal Mach
        M2n_sq = (M1n**2 + 2 / (gamma - 1)) / (2 * gamma / (gamma - 1) * M1n**2 - 1)
        M2n = np.sqrt(M2n_sq)

        # Post-shock Mach number
        M2 = M2n / np.sin(beta - theta_rad)

        return torch.tensor([rho2_rho1, p2_p1, M2], dtype=torch.float64)

    def _solve_for_beta(self, M1: float, theta: float) -> float:
        """Solve θ-β-M relation for shock angle."""
        gamma = self.gamma

        # Newton-Raphson with initial guess
        beta = theta + np.radians(5)  # Start slightly above deflection

        for _ in range(50):
            tan_theta = (
                2
                / np.tan(beta)
                * (
                    (M1**2 * np.sin(beta) ** 2 - 1)
                    / (M1**2 * (gamma + np.cos(2 * beta)) + 2)
                )
            )

            f = np.arctan(tan_theta) - theta

            # Numerical derivative
            h = 1e-6
            tan_theta_h = (
                2
                / np.tan(beta + h)
                * (
                    (M1**2 * np.sin(beta + h) ** 2 - 1)
                    / (M1**2 * (gamma + np.cos(2 * (beta + h))) + 2)
                )
            )
            df = (np.arctan(tan_theta_h) - np.arctan(tan_theta)) / h

            beta_new = beta - f / df

            if abs(beta_new - beta) < 1e-10:
                break
            beta = beta_new

        return beta

    def compute_shock_angle(self, M1: float, theta: float) -> float:
        """
        Compute shock angle for given Mach and deflection.

        Args:
            M1: Upstream Mach number
            theta: Deflection angle in degrees

        Returns:
            Shock angle in degrees
        """
        beta = self._solve_for_beta(M1, np.radians(theta))
        return np.degrees(beta)


class IsentropicVortexValidator(AnalyticalValidator):
    """
    Validator for isentropic vortex advection.

    The isentropic vortex is an exact solution to the Euler equations
    that advects without changing shape, making it ideal for accuracy testing.
    """

    def __init__(
        self,
        gamma: float = 1.4,
        vortex_strength: float = 5.0,
        tolerance: float = 0.01,
    ):
        """
        Initialize isentropic vortex validator.

        Args:
            gamma: Ratio of specific heats
            vortex_strength: Vortex circulation parameter
            tolerance: Maximum acceptable error
        """
        super().__init__(tolerance=tolerance, error_norm="L2")
        self.gamma = gamma
        self.beta = vortex_strength

        # Background state
        self.rho_inf = 1.0
        self.u_inf = 1.0
        self.v_inf = 0.0
        self.p_inf = 1.0

    def compute_analytical_solution(
        self,
        x: torch.Tensor,
        t: float,
        y: torch.Tensor | None = None,
        x0: float = 5.0,
        y0: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute isentropic vortex solution.

        Args:
            x: x-coordinates
            t: Time
            y: y-coordinates
            x0, y0: Initial vortex center

        Returns:
            Primitive state (rho, u, v, p) at each point
        """
        if y is None:
            raise ValueError("y coordinates required for vortex validation")

        gamma = self.gamma

        x_np = x.numpy() if isinstance(x, torch.Tensor) else x
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y

        # Vortex center at time t
        xc = x0 + self.u_inf * t
        yc = y0 + self.v_inf * t

        # Distance from vortex center
        r2 = (x_np - xc) ** 2 + (y_np - yc) ** 2

        # Temperature and velocity perturbations
        dT = -(gamma - 1) * self.beta**2 / (8 * gamma * np.pi**2) * np.exp(1 - r2)

        du = -self.beta / (2 * np.pi) * (y_np - yc) * np.exp(0.5 * (1 - r2))
        dv = self.beta / (2 * np.pi) * (x_np - xc) * np.exp(0.5 * (1 - r2))

        # Perturbed state
        T = self.p_inf / self.rho_inf + dT
        rho = self.rho_inf * T ** (1 / (gamma - 1))
        u = self.u_inf + du
        v = self.v_inf + dv
        p = rho * T

        return torch.tensor(np.stack([rho, u, v, p], axis=0), dtype=torch.float64)


def run_physical_validation(
    validation_tests: list[ConservationValidator | AnalyticalValidator],
    **kwargs,
) -> ValidationReport:
    """
    Run a suite of physical validation tests.

    Args:
        validation_tests: List of validators to run
        **kwargs: Arguments passed to each validator

    Returns:
        ValidationReport with all results
    """
    results = []

    for validator in validation_tests:
        try:
            if isinstance(validator, ConservationValidator):
                result = validator.validate(**kwargs)
            elif isinstance(validator, AnalyticalValidator):
                result = validator.validate(**kwargs)
            else:
                continue
            results.append(result)
        except Exception as e:
            results.append(
                ValidationResult(
                    test_name=type(validator).__name__,
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    metric_name="Execution",
                    computed_value=0.0,
                    expected_value=1.0,
                    tolerance=0.0,
                    message=f"Test failed with error: {str(e)}",
                )
            )

    return ValidationReport(
        title="Physical Validation Report",
        timestamp=time.time(),
        results=results,
        configuration=kwargs,
    )
