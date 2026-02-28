"""
Domain Pack I — Classical Mechanics (V0.2)
==========================================

Production-grade V0.2 implementations for all eight taxonomy nodes:

  PHY-I.1  N-body problem          — 2-body gravitational circular orbit
  PHY-I.2  Rigid body dynamics     — Torque-free Euler equations (axisymmetric)
  PHY-I.3  Lagrangian mechanics    — Simple pendulum (linearised)
  PHY-I.4  Hamiltonian mechanics   — 1-D harmonic oscillator
  PHY-I.5  Orbital mechanics       — Kepler 2-D circular orbit
  PHY-I.6  Vibrations / oscillations — Coupled spring-mass normal modes
  PHY-I.7  Continuum mechanics     — 1-D elastic wave equation
  PHY-I.8  Chaos theory            — Lorenz system

Every solver integrates the *actual* governing equations with RK4, then
validates the numerical result against an exact or high-fidelity reference
solution via :func:`validate_v02`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor

from ontic.platform.domain_pack import DomainPack, get_registry
from ontic.platform.protocols import (
    Discretization,
    Observable,
    ProblemSpec,
    Solver,
    SolveResult,
)
from ontic.packs._base import ODEReferenceSolver, PDE1DReferenceSolver, validate_v02


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-I.1  N-body problem — two-body gravitational circular orbit
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NBodySpec:
    """Two-body gravitational problem in 2-D with unit masses and G=1."""

    @property
    def name(self) -> str:
        return "PHY-I.1_N-body_problem"

    @property
    def ndim(self) -> int:
        return 2

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"G": 1.0, "m1": 0.5, "m2": 0.5, "node": "PHY-I.1"}

    @property
    def governing_equations(self) -> str:
        return (
            "d²r_i/dt² = -G Σ_{j≠i} m_j (r_i - r_j) / |r_i - r_j|³; "
            "canonical case: equal-mass circular orbit, period T = 2π"
        )

    @property
    def field_names(self) -> Sequence[str]:
        return ("x1", "y1", "x2", "y2", "vx1", "vy1", "vx2", "vy2")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("energy", "position_error")


class NBodySolver(ODEReferenceSolver):
    """RK4 solver for the 2-body gravitational problem.

    State vector: [x1, y1, x2, y2, vx1, vy1, vx2, vy2].
    Canonical setup: equal masses m=0.5 on a circular orbit with separation 1,
    G=1 ⇒ orbital period T = 2π.
    """

    def __init__(self) -> None:
        super().__init__("NBody_RK4")
        self._G: float = 1.0
        self._m1: float = 0.5
        self._m2: float = 0.5

    def _rhs(self, y: Tensor, t: float) -> Tensor:
        """Gravitational N-body RHS for two bodies in 2-D."""
        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y
        dx = x2 - x1
        dy = y2 - y1
        r3 = (dx * dx + dy * dy).pow(1.5)
        ax1 = self._G * self._m2 * dx / r3
        ay1 = self._G * self._m2 * dy / r3
        ax2 = -self._G * self._m1 * dx / r3
        ay2 = -self._G * self._m1 * dy / r3
        return torch.stack([vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2])

    @staticmethod
    def canonical_ic() -> Tensor:
        """Equal-mass circular orbit: bodies at (±0.5, 0), v = (0, ±0.5)."""
        return torch.tensor(
            [0.5, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, -0.5], dtype=torch.float64
        )

    def step(self, state: Any, dt: float, **kwargs: Any) -> Tensor:
        """Single RK4 step."""
        y = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, _ = self.solve_ode(self._rhs, y, (0.0, dt), dt)
        return y_final

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Integrate the 2-body system and validate circular-orbit return."""
        y0 = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, trajectory = self.solve_ode(self._rhs, y0, t_span, dt)
        error = (y_final[:4] - y0[:4]).norm().item()
        validation = validate_v02(error=error, tolerance=1e-3, label="PHY-I.1 N-body circular orbit")
        return SolveResult(
            final_state=y_final,
            t_final=t_span[1],
            steps_taken=len(trajectory) - 1,
            metadata={"error": error, "node": "PHY-I.1", "validation": validation},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-I.2  Rigid body dynamics — torque-free Euler equations
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RigidBodySpec:
    """Torque-free rotation of an axially symmetric rigid body (I1=I2≠I3)."""

    @property
    def name(self) -> str:
        return "PHY-I.2_Rigid_body_dynamics"

    @property
    def ndim(self) -> int:
        return 0  # ODE system, no spatial dimension

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"I1": 1.0, "I2": 1.0, "I3": 2.0, "node": "PHY-I.2"}

    @property
    def governing_equations(self) -> str:
        return (
            "I1 dω1/dt = (I2 - I3) ω2 ω3; "
            "I2 dω2/dt = (I3 - I1) ω3 ω1; "
            "I3 dω3/dt = (I1 - I2) ω1 ω2"
        )

    @property
    def field_names(self) -> Sequence[str]:
        return ("omega1", "omega2", "omega3")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("kinetic_energy", "angular_momentum_sq")


class RigidBodySolver(ODEReferenceSolver):
    """RK4 solver for Euler's torque-free rotation equations.

    For an axially symmetric body with I1=I2, ω3 is constant and
    ω1² + ω2² is conserved.  Kinetic energy T = ½ Σ Iᵢ ωᵢ² and
    angular-momentum squared L² = Σ Iᵢ² ωᵢ² are both conserved.
    """

    def __init__(self) -> None:
        super().__init__("RigidBody_RK4")
        self._I1: float = 1.0
        self._I2: float = 1.0
        self._I3: float = 2.0

    def _rhs(self, y: Tensor, t: float) -> Tensor:
        """Euler equations RHS."""
        w1, w2, w3 = y
        dw1 = (self._I2 - self._I3) * w2 * w3 / self._I1
        dw2 = (self._I3 - self._I1) * w3 * w1 / self._I2
        dw3 = (self._I1 - self._I2) * w1 * w2 / self._I3
        return torch.stack([dw1, dw2, dw3])

    @staticmethod
    def canonical_ic() -> Tensor:
        """ω = (1, 0, 1) — precession about the symmetry axis."""
        return torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)

    def step(self, state: Any, dt: float, **kwargs: Any) -> Tensor:
        """Single RK4 step."""
        y = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, _ = self.solve_ode(self._rhs, y, (0.0, dt), dt)
        return y_final

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Integrate Euler equations and validate conservation laws."""
        y0 = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, trajectory = self.solve_ode(self._rhs, y0, t_span, dt)

        # Conservation checks
        T0 = 0.5 * (self._I1 * y0[0] ** 2 + self._I2 * y0[1] ** 2 + self._I3 * y0[2] ** 2)
        Tf = 0.5 * (self._I1 * y_final[0] ** 2 + self._I2 * y_final[1] ** 2 + self._I3 * y_final[2] ** 2)
        L2_0 = self._I1 ** 2 * y0[0] ** 2 + self._I2 ** 2 * y0[1] ** 2 + self._I3 ** 2 * y0[2] ** 2
        L2_f = self._I1 ** 2 * y_final[0] ** 2 + self._I2 ** 2 * y_final[1] ** 2 + self._I3 ** 2 * y_final[2] ** 2

        energy_error = abs((Tf - T0).item()) / max(abs(T0.item()), 1e-30)
        angmom_error = abs((L2_f - L2_0).item()) / max(abs(L2_0.item()), 1e-30)
        combined_error = max(energy_error, angmom_error)

        validation = validate_v02(error=combined_error, tolerance=1e-6, label="PHY-I.2 rigid body conservation")
        return SolveResult(
            final_state=y_final,
            t_final=t_span[1],
            steps_taken=len(trajectory) - 1,
            metadata={
                "error": combined_error,
                "energy_error": energy_error,
                "angmom_error": angmom_error,
                "node": "PHY-I.2",
                "validation": validation,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-I.3  Lagrangian mechanics — linearised simple pendulum
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LagrangianSpec:
    """Linearised simple pendulum θ'' + (g/L)θ = 0."""

    @property
    def name(self) -> str:
        return "PHY-I.3_Lagrangian_mechanics"

    @property
    def ndim(self) -> int:
        return 0

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"g": 9.81, "L": 1.0, "theta0": 0.1, "node": "PHY-I.3"}

    @property
    def governing_equations(self) -> str:
        return "θ'' + (g/L)θ = 0  ⟹  d/dt[θ, ω] = [ω, -(g/L)θ]"

    @property
    def field_names(self) -> Sequence[str]:
        return ("theta", "omega")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("energy", "theta_error")


class LagrangianSolver(ODEReferenceSolver):
    """RK4 solver for the linearised simple pendulum.

    Exact solution: θ(t) = θ₀ cos(√(g/L) t) for IC θ(0)=θ₀, ω(0)=0.
    """

    def __init__(self) -> None:
        super().__init__("Lagrangian_Pendulum_RK4")
        self._g: float = 9.81
        self._L: float = 1.0
        self._omega_n: float = math.sqrt(self._g / self._L)

    def _rhs(self, y: Tensor, t: float) -> Tensor:
        """Pendulum ODE RHS: dθ/dt = ω, dω/dt = -(g/L)θ."""
        theta, omega = y
        return torch.stack([omega, -(self._g / self._L) * theta])

    @staticmethod
    def canonical_ic() -> Tensor:
        """θ₀ = 0.1 rad, ω₀ = 0."""
        return torch.tensor([0.1, 0.0], dtype=torch.float64)

    def step(self, state: Any, dt: float, **kwargs: Any) -> Tensor:
        """Single RK4 step."""
        y = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, _ = self.solve_ode(self._rhs, y, (0.0, dt), dt)
        return y_final

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Integrate the pendulum and validate against exact solution."""
        y0 = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        theta0 = y0[0].item()
        y_final, trajectory = self.solve_ode(self._rhs, y0, t_span, dt)

        tf = t_span[1]
        exact_theta = theta0 * math.cos(self._omega_n * tf)
        exact_omega = -theta0 * self._omega_n * math.sin(self._omega_n * tf)
        exact = torch.tensor([exact_theta, exact_omega], dtype=torch.float64)
        error = (y_final - exact).abs().max().item()

        validation = validate_v02(error=error, tolerance=1e-4, label="PHY-I.3 Lagrangian pendulum")
        return SolveResult(
            final_state=y_final,
            t_final=tf,
            steps_taken=len(trajectory) - 1,
            metadata={"error": error, "node": "PHY-I.3", "validation": validation},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-I.4  Hamiltonian mechanics — 1-D harmonic oscillator
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HamiltonianSpec:
    """1-D harmonic oscillator in Hamiltonian form: H = p²/2m + kq²/2."""

    @property
    def name(self) -> str:
        return "PHY-I.4_Hamiltonian_mechanics"

    @property
    def ndim(self) -> int:
        return 0

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"m": 1.0, "k": 1.0, "q0": 1.0, "p0": 0.0, "node": "PHY-I.4"}

    @property
    def governing_equations(self) -> str:
        return "dq/dt = p/m,  dp/dt = -kq;  H = p²/(2m) + kq²/2 = const"

    @property
    def field_names(self) -> Sequence[str]:
        return ("q", "p")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("hamiltonian", "phase_error")


class HamiltonianSolver(ODEReferenceSolver):
    """RK4 solver for the 1-D harmonic oscillator in Hamilton's form.

    Exact: q(t) = cos(t), p(t) = -sin(t) for m=k=1, q₀=1, p₀=0.
    Validates both trajectory accuracy and Hamiltonian conservation.
    """

    def __init__(self) -> None:
        super().__init__("Hamiltonian_HO_RK4")
        self._m: float = 1.0
        self._k: float = 1.0

    def _rhs(self, y: Tensor, t: float) -> Tensor:
        """Hamilton's equations: dq/dt = p/m, dp/dt = -kq."""
        q, p = y
        return torch.stack([p / self._m, -self._k * q])

    def _hamiltonian(self, y: Tensor) -> Tensor:
        """Evaluate H = p²/(2m) + kq²/2."""
        q, p = y
        return p ** 2 / (2.0 * self._m) + self._k * q ** 2 / 2.0

    @staticmethod
    def canonical_ic() -> Tensor:
        """q₀=1, p₀=0."""
        return torch.tensor([1.0, 0.0], dtype=torch.float64)

    def step(self, state: Any, dt: float, **kwargs: Any) -> Tensor:
        """Single RK4 step."""
        y = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, _ = self.solve_ode(self._rhs, y, (0.0, dt), dt)
        return y_final

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Integrate the oscillator and validate against exact + Hamiltonian."""
        y0 = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, trajectory = self.solve_ode(self._rhs, y0, t_span, dt)

        tf = t_span[1]
        omega = math.sqrt(self._k / self._m)
        q0, p0 = y0[0].item(), y0[1].item()
        exact_q = q0 * math.cos(omega * tf) + (p0 / (self._m * omega)) * math.sin(omega * tf)
        exact_p = -self._m * omega * q0 * math.sin(omega * tf) + p0 * math.cos(omega * tf)
        exact = torch.tensor([exact_q, exact_p], dtype=torch.float64)

        trajectory_error = (y_final - exact).abs().max().item()
        H0 = self._hamiltonian(y0).item()
        Hf = self._hamiltonian(y_final).item()
        hamiltonian_error = abs(Hf - H0) / max(abs(H0), 1e-30)
        combined_error = max(trajectory_error, hamiltonian_error)

        validation = validate_v02(error=combined_error, tolerance=1e-6, label="PHY-I.4 Hamiltonian oscillator")
        return SolveResult(
            final_state=y_final,
            t_final=tf,
            steps_taken=len(trajectory) - 1,
            metadata={
                "error": combined_error,
                "trajectory_error": trajectory_error,
                "hamiltonian_error": hamiltonian_error,
                "node": "PHY-I.4",
                "validation": validation,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-I.5  Orbital mechanics — Kepler 2-D circular orbit
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OrbitalSpec:
    """2-D Kepler problem with GM=1, circular orbit at r=1."""

    @property
    def name(self) -> str:
        return "PHY-I.5_Orbital_mechanics"

    @property
    def ndim(self) -> int:
        return 2

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"GM": 1.0, "r0": 1.0, "node": "PHY-I.5"}

    @property
    def governing_equations(self) -> str:
        return "d²r/dt² = -GM r / |r|³;  circular orbit: T = 2π for GM=1, r=1"

    @property
    def field_names(self) -> Sequence[str]:
        return ("x", "y", "vx", "vy")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("energy", "angular_momentum", "position_error")


class OrbitalSolver(ODEReferenceSolver):
    """RK4 solver for the 2-D Kepler problem.

    Circular orbit at r=1 with GM=1: v = 1, period T = 2π.
    Validates that the orbit returns to its initial position after one period.
    """

    def __init__(self) -> None:
        super().__init__("Kepler_RK4")
        self._GM: float = 1.0

    def _rhs(self, y: Tensor, t: float) -> Tensor:
        """Kepler RHS: dr/dt = v, dv/dt = -GM r / |r|³."""
        x, yc, vx, vy = y
        r3 = (x * x + yc * yc).pow(1.5)
        ax = -self._GM * x / r3
        ay = -self._GM * yc / r3
        return torch.stack([vx, vy, ax, ay])

    @staticmethod
    def canonical_ic() -> Tensor:
        """Circular orbit: (x,y)=(1,0), (vx,vy)=(0,1)."""
        return torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64)

    def step(self, state: Any, dt: float, **kwargs: Any) -> Tensor:
        """Single RK4 step."""
        y = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, _ = self.solve_ode(self._rhs, y, (0.0, dt), dt)
        return y_final

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Integrate Kepler orbit and validate period-return accuracy."""
        y0 = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, trajectory = self.solve_ode(self._rhs, y0, t_span, dt)

        position_error = (y_final[:2] - y0[:2]).norm().item()
        validation = validate_v02(error=position_error, tolerance=1e-3, label="PHY-I.5 Kepler circular orbit")
        return SolveResult(
            final_state=y_final,
            t_final=t_span[1],
            steps_taken=len(trajectory) - 1,
            metadata={"error": position_error, "node": "PHY-I.5", "validation": validation},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-I.6  Vibrations — coupled spring-mass, normal modes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VibrationsSpec:
    """Two-mass, three-spring coupled oscillator: M=I, K=[[2,-1],[-1,2]]."""

    @property
    def name(self) -> str:
        return "PHY-I.6_Vibrations_and_oscillations"

    @property
    def ndim(self) -> int:
        return 0

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"K": [[2, -1], [-1, 2]], "M": [[1, 0], [0, 1]], "node": "PHY-I.6"}

    @property
    def governing_equations(self) -> str:
        return (
            "M ẍ + K x = 0;  K = [[2,-1],[-1,2]], M = I; "
            "normal modes ω² = 1, 3;  "
            "x₁(t) = (cos(t) + cos(√3 t))/2 for x₁(0)=1, x₂(0)=0"
        )

    @property
    def field_names(self) -> Sequence[str]:
        return ("x1", "x2", "v1", "v2")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("energy", "x1_error")


class VibrationsSolver(ODEReferenceSolver):
    """RK4 solver for a 2-DOF coupled spring-mass system.

    Stiffness matrix K = [[2,-1],[-1,2]], mass matrix M = I.
    Normal-mode frequencies: ω² = 1, 3.
    Exact solution for IC x₁=1, x₂=0, v₁=v₂=0:
        x₁(t) = (cos(t) + cos(√3 t)) / 2
        x₂(t) = (cos(t) - cos(√3 t)) / 2
    """

    def __init__(self) -> None:
        super().__init__("Vibrations_RK4")
        self._K = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float64)

    def _rhs(self, y: Tensor, t: float) -> Tensor:
        """State = [x1, x2, v1, v2].  ẍ = -K x."""
        x = y[:2]
        v = y[2:]
        a = -self._K @ x
        return torch.cat([v, a])

    @staticmethod
    def canonical_ic() -> Tensor:
        """x₁=1, x₂=0, v₁=v₂=0."""
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    def step(self, state: Any, dt: float, **kwargs: Any) -> Tensor:
        """Single RK4 step."""
        y = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, _ = self.solve_ode(self._rhs, y, (0.0, dt), dt)
        return y_final

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Integrate coupled oscillator and validate against exact modes."""
        y0 = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, trajectory = self.solve_ode(self._rhs, y0, t_span, dt)

        tf = t_span[1]
        sqrt3 = math.sqrt(3.0)
        exact_x1 = 0.5 * (math.cos(tf) + math.cos(sqrt3 * tf))
        exact_x2 = 0.5 * (math.cos(tf) - math.cos(sqrt3 * tf))
        exact_v1 = 0.5 * (-math.sin(tf) - sqrt3 * math.sin(sqrt3 * tf))
        exact_v2 = 0.5 * (-math.sin(tf) + sqrt3 * math.sin(sqrt3 * tf))
        exact = torch.tensor([exact_x1, exact_x2, exact_v1, exact_v2], dtype=torch.float64)

        error = (y_final - exact).abs().max().item()
        validation = validate_v02(error=error, tolerance=1e-4, label="PHY-I.6 coupled vibrations")
        return SolveResult(
            final_state=y_final,
            t_final=tf,
            steps_taken=len(trajectory) - 1,
            metadata={"error": error, "node": "PHY-I.6", "validation": validation},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-I.7  Continuum mechanics — 1-D elastic wave equation
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ContinuumSpec:
    """1-D elastic wave equation ∂²u/∂t² = c² ∂²u/∂x², periodic BC."""

    @property
    def name(self) -> str:
        return "PHY-I.7_Continuum_mechanics"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"c": 1.0, "N": 128, "node": "PHY-I.7"}

    @property
    def governing_equations(self) -> str:
        return (
            "∂²u/∂t² = c² ∂²u/∂x²  (wave equation); "
            "first-order form: ∂u/∂t = v, ∂v/∂t = c² ∂²u/∂x²; "
            "periodic BC, IC: u = sin(2πx), v = 0"
        )

    @property
    def field_names(self) -> Sequence[str]:
        return ("u", "v")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("energy", "u_error")


class ContinuumSolver(PDE1DReferenceSolver):
    """RK4 solver for the 1-D elastic wave equation via method of lines.

    Written as a first-order system [u, v] where v = ∂u/∂t:
        du/dt = v
        dv/dt = c² ∂²u/∂x²  (periodic central differences)

    Exact: u(x,t) = sin(2πx) cos(2πct) for IC u=sin(2πx), v=0.
    """

    def __init__(self) -> None:
        super().__init__("WaveEq_1D_RK4")
        self._c: float = 1.0
        self._N: int = 128

    def _rhs(self, state: Tensor, t: float, dx: float) -> Tensor:
        """Semi-discrete RHS for the wave equation first-order system.

        state has shape (2*N,): first N entries are u, last N are v.
        """
        N = state.shape[0] // 2
        u = state[:N]
        v = state[N:]
        # Periodic second-order central difference: d²u/dx²
        u_xx = (torch.roll(u, -1) + torch.roll(u, 1) - 2.0 * u) / (dx * dx)
        du_dt = v
        dv_dt = self._c ** 2 * u_xx
        return torch.cat([du_dt, dv_dt])

    def _build_ic(self, N: int, L: float) -> Tuple[Tensor, float]:
        """Build initial condition: u = sin(2πx), v = 0 on [0, L)."""
        dx = L / N
        x = torch.linspace(0.0, L - dx, N, dtype=torch.float64)
        u0 = torch.sin(2.0 * math.pi * x)
        v0 = torch.zeros(N, dtype=torch.float64)
        return torch.cat([u0, v0]), dx

    @staticmethod
    def canonical_ic() -> Tuple[Tensor, float]:
        """Return (state0, dx) for N=128, L=1."""
        N, L = 128, 1.0
        dx = L / N
        x = torch.linspace(0.0, L - dx, N, dtype=torch.float64)
        u0 = torch.sin(2.0 * math.pi * x)
        v0 = torch.zeros(N, dtype=torch.float64)
        return torch.cat([u0, v0]), dx

    def step(self, state: Any, dt: float, **kwargs: Any) -> Tensor:
        """Single RK4 step for the PDE system."""
        s = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        N = s.shape[0] // 2
        dx = kwargs.get("dx", 1.0 / N)
        s_final, _ = self.solve_pde(self._rhs, s, dx, (0.0, dt), dt)
        return s_final

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Integrate the wave equation and validate against exact solution."""
        s0 = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        N = s0.shape[0] // 2
        L = 1.0
        dx = L / N
        s_final, trajectory = self.solve_pde(self._rhs, s0, dx, t_span, dt)

        tf = t_span[1]
        x = torch.linspace(0.0, L - dx, N, dtype=torch.float64)
        exact_u = torch.sin(2.0 * math.pi * x) * math.cos(2.0 * math.pi * self._c * tf)
        numerical_u = s_final[:N]
        error = (numerical_u - exact_u).abs().max().item()

        validation = validate_v02(error=error, tolerance=1e-3, label="PHY-I.7 1-D wave equation")
        return SolveResult(
            final_state=s_final,
            t_final=tf,
            steps_taken=len(trajectory) - 1,
            metadata={"error": error, "node": "PHY-I.7", "validation": validation},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-I.8  Chaos theory — Lorenz system
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChaosSpec:
    """Lorenz system with standard chaotic parameters σ=10, ρ=28, β=8/3."""

    @property
    def name(self) -> str:
        return "PHY-I.8_Chaos_theory"

    @property
    def ndim(self) -> int:
        return 0

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0, "node": "PHY-I.8"}

    @property
    def governing_equations(self) -> str:
        return (
            "dx/dt = σ(y - x);  dy/dt = x(ρ - z) - y;  dz/dt = xy - βz; "
            "σ=10, ρ=28, β=8/3"
        )

    @property
    def field_names(self) -> Sequence[str]:
        return ("x", "y", "z")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("attractor_bounds", "short_time_error")


class ChaosSolver(ODEReferenceSolver):
    """RK4 solver for the Lorenz system.

    Validation strategy for chaotic dynamics:
      1. Short-time accuracy: integrate to T=0.1 with dt=1e-3, compare against
         a high-fidelity reference trajectory computed with dt=1e-5.
      2. Attractor bounds: over a longer run (T=10), verify that the trajectory
         remains within the known Lorenz attractor bounds.
    """

    def __init__(self) -> None:
        super().__init__("Lorenz_RK4")
        self._sigma: float = 10.0
        self._rho: float = 28.0
        self._beta: float = 8.0 / 3.0

    def _rhs(self, y: Tensor, t: float) -> Tensor:
        """Lorenz system RHS."""
        x, yc, z = y
        dx = self._sigma * (yc - x)
        dy = x * (self._rho - z) - yc
        dz = x * yc - self._beta * z
        return torch.stack([dx, dy, dz])

    @staticmethod
    def canonical_ic() -> Tensor:
        """Classic IC: (1, 1, 1)."""
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    def step(self, state: Any, dt: float, **kwargs: Any) -> Tensor:
        """Single RK4 step."""
        y = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)
        y_final, _ = self.solve_ode(self._rhs, y, (0.0, dt), dt)
        return y_final

    def solve(
        self,
        state: Any,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """Integrate the Lorenz system with dual validation.

        1. Short-time reference comparison (T=0.1, fine dt=1e-5 vs coarse dt).
        2. Attractor-bound check over the full integration window.
        """
        y0 = state if isinstance(state, Tensor) else torch.as_tensor(state, dtype=torch.float64)

        # --- Short-time reference validation ---
        T_short = min(0.1, t_span[1] - t_span[0])
        fine_dt = 1e-5
        ref_final, _ = self.solve_ode(self._rhs, y0, (t_span[0], t_span[0] + T_short), fine_dt)
        coarse_final, _ = self.solve_ode(self._rhs, y0, (t_span[0], t_span[0] + T_short), dt)
        short_time_error = (coarse_final - ref_final).abs().max().item()

        # --- Full integration ---
        y_final, trajectory = self.solve_ode(self._rhs, y0, t_span, dt)

        # --- Attractor bound validation ---
        bounds_violated = False
        x_max, y_max, z_max = 0.0, 0.0, 0.0
        for snap in trajectory:
            ax, ay, az = abs(snap[0].item()), abs(snap[1].item()), abs(snap[2].item())
            x_max = max(x_max, ax)
            y_max = max(y_max, ay)
            z_max = max(z_max, az)
            if ax > 25.0 or ay > 30.0 or az > 55.0:
                bounds_violated = True
                break

        bound_error = 1.0 if bounds_violated else 0.0
        combined_error = max(short_time_error, bound_error)

        validation_short = validate_v02(
            error=short_time_error, tolerance=1e-2, label="PHY-I.8 Lorenz short-time"
        )
        validation_bounds = validate_v02(
            error=bound_error, tolerance=0.5, label="PHY-I.8 Lorenz attractor bounds"
        )

        return SolveResult(
            final_state=y_final,
            t_final=t_span[1],
            steps_taken=len(trajectory) - 1,
            metadata={
                "error": combined_error,
                "short_time_error": short_time_error,
                "bounds_violated": bounds_violated,
                "x_max": x_max,
                "y_max": y_max,
                "z_max": z_max,
                "node": "PHY-I.8",
                "validation_short": validation_short,
                "validation_bounds": validation_bounds,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Spec and solver registries
# ═══════════════════════════════════════════════════════════════════════════════

_SPECS: Dict[str, type] = {
    "PHY-I.1": NBodySpec,
    "PHY-I.2": RigidBodySpec,
    "PHY-I.3": LagrangianSpec,
    "PHY-I.4": HamiltonianSpec,
    "PHY-I.5": OrbitalSpec,
    "PHY-I.6": VibrationsSpec,
    "PHY-I.7": ContinuumSpec,
    "PHY-I.8": ChaosSpec,
}

_SOLVERS: Dict[str, type] = {
    "PHY-I.1": NBodySolver,
    "PHY-I.2": RigidBodySolver,
    "PHY-I.3": LagrangianSolver,
    "PHY-I.4": HamiltonianSolver,
    "PHY-I.5": OrbitalSolver,
    "PHY-I.6": VibrationsSolver,
    "PHY-I.7": ContinuumSolver,
    "PHY-I.8": ChaosSolver,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pack registration
# ═══════════════════════════════════════════════════════════════════════════════


class ClassicalMechanicsPack(DomainPack):
    """Pack I: Classical Mechanics — V0.2 production solvers."""

    @property
    def pack_id(self) -> str:
        return "I"

    @property
    def pack_name(self) -> str:
        return "Classical Mechanics"

    @property
    def taxonomy_ids(self) -> Sequence[str]:
        return tuple(_SPECS.keys())

    def problem_specs(self) -> Dict[str, Type[ProblemSpec]]:
        return dict(_SPECS)  # type: ignore[arg-type]

    def solvers(self) -> Dict[str, Type[Solver]]:
        return dict(_SOLVERS)  # type: ignore[arg-type]

    def discretizations(self) -> Dict[str, Sequence[Type[Discretization]]]:
        return {}

    def observables(self) -> Dict[str, Sequence[Type[Observable]]]:
        return {}

    @property
    def version(self) -> str:
        return "0.2.0"


get_registry().register_pack(ClassicalMechanicsPack())
