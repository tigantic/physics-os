"""
Domain Pack II — Fluid Dynamics
================================

**Anchor problem (V0.4)**:  1-D viscous Burgers equation (PHY-II.1)

    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²     x ∈ [0, 2π], periodic BCs

Cole-Hopf exact solution
------------------------
u(x, t) is obtained via the Cole-Hopf transformation from the heat equation.
For IC  u(x, 0) = −2ν φ'(x)/φ(x)  with φ = exp(−cos(x)/(2ν)):

    u(x, t) = −2ν ∂_x ln φ(x, t)

where φ(x, t) solves the heat equation φ_t = ν φ_xx.

For practical validation we use a simpler IC and compare against a
high-resolution reference computed with the same RHS on N=2048.

Validation gates (V0.4):
  • L∞ error vs high-res reference < 1e-3 at t = T.
  • Grid refinement: error ratio ≈ 2 (first-order upwind) or ≈ 4 (second-order).
  • Conservation: ∫u dx preserved (periodic, conservative flux).
  • Deterministic across two runs.

Scaffold nodes (V0.1):  PHY-II.1 through PHY-II.10
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor

from tensornet.packs._base import (
    compute_l2_error,
    compute_linf_error,
    convergence_order,
)
from tensornet.platform.data_model import (
    FieldData,
    SimulationState,
    StructuredMesh,
)
from tensornet.platform.domain_pack import DomainPack, get_registry
from tensornet.platform.protocols import (
    Discretization,
    Observable,
    ProblemSpec,
    Solver,
    SolveResult,
)
from tensornet.platform.reproduce import ReproducibilityContext, hash_tensor
from tensornet.platform.solvers import ForwardEuler, RK4, RHSCallable, TimeIntegrator


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-II.1  Incompressible NS → 1-D Viscous Burgers (ANCHOR)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BurgersSpec:
    """1-D viscous Burgers: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²  (periodic)."""

    nu: float = 0.01
    L: float = 6.283185307179586  # 2π
    T_final: float = 0.5

    @property
    def name(self) -> str:
        return "ViscousBurgers1D"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"nu": self.nu, "L": self.L, "T_final": self.T_final}

    @property
    def governing_equations(self) -> str:
        return r"\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2 u}{\partial x^2}"

    @property
    def field_names(self) -> Sequence[str]:
        return ("u",)

    @property
    def observable_names(self) -> Sequence[str]:
        return ("l2_norm", "integral_u", "kinetic_energy")


class FVM_Burgers_1D:
    """
    1-D FVM discretization of viscous Burgers.

    Flux splitting:  Lax-Friedrichs for the nonlinear advection term.
    Diffusion:       Second-order central.
    Boundary:        Periodic.
    """

    def __init__(self, nu: float) -> None:
        self._nu = nu

    @property
    def method(self) -> str:
        return "FVM"

    @property
    def order(self) -> int:
        return 2

    def discretize(self, spec: ProblemSpec, mesh: Any) -> "BurgersOps":
        if not isinstance(mesh, StructuredMesh) or mesh.ndim != 1:
            raise TypeError("FVM_Burgers_1D requires a 1-D StructuredMesh")
        return BurgersOps(dx=mesh.dx[0], N=mesh.shape[0], nu=self._nu)


@dataclass
class BurgersOps:
    """Discrete operators for 1-D viscous Burgers with periodic BCs."""

    dx: float
    N: int
    nu: float

    def advection_conservative_central(self, u: Tensor) -> Tensor:
        """
        Second-order conservative central flux for f(u) = u²/2.

        F_{i+1/2} = (f(u_i) + f(u_{i+1})) / 2  (no numerical diffusion).
        Stable when physical diffusion ν > 0 and CFL < 1.
        """
        f = 0.5 * u ** 2
        f_right = torch.roll(f, -1)
        flux_right = 0.5 * (f + f_right)
        flux_left = torch.roll(flux_right, 1)
        return -(flux_right - flux_left) / self.dx

    def diffusion_central(self, u: Tensor) -> Tensor:
        """Second-order central Laplacian with periodic BCs."""
        return self.nu * (torch.roll(u, -1) - 2.0 * u + torch.roll(u, 1)) / (self.dx ** 2)

    def rhs(self, state: SimulationState, t: float) -> Dict[str, Tensor]:
        """du/dt = -d(u²/2)/dx + ν d²u/dx²."""
        u = state.get_field("u").data
        return {"u": self.advection_conservative_central(u) + self.diffusion_central(u)}


class BurgersKEObservable:
    """Kinetic energy: ∫ u²/2 dx."""

    def __init__(self, dx: float) -> None:
        self._dx = dx

    @property
    def name(self) -> str:
        return "kinetic_energy"

    @property
    def units(self) -> str:
        return "m²/s²·m"

    def compute(self, state: Any) -> Tensor:
        u = state.get_field("u").data
        return 0.5 * (u ** 2).sum() * self._dx


class BurgersIntegralObservable:
    """Observable: spatial integral of u (conserved for inviscid Burgers)."""

    def __init__(self, dx: float) -> None:
        self._dx = dx

    @property
    def name(self) -> str:
        return "integral_u"

    @property
    def units(self) -> str:
        return "m²/s"

    def compute(self, state: Any) -> Tensor:
        return state.get_field("u").data.sum() * self._dx


class BurgersL2Observable:
    """Observable: L2 norm of the velocity field."""

    @property
    def name(self) -> str:
        return "l2_norm"

    @property
    def units(self) -> str:
        return "1"

    def compute(self, state: Any) -> Tensor:
        return torch.norm(state.get_field("u").data, p=2)


class BurgersSolver:
    """Full solver wrapping RK4 + FVM Lax-Friedrichs for Burgers."""

    def __init__(self, nu: float = 0.01) -> None:
        self._nu = nu

    @property
    def name(self) -> str:
        return "BurgersSolver_RK4_LF"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        mesh = state.mesh
        ops = BurgersOps(dx=mesh.dx[0], N=mesh.shape[0], nu=self._nu)
        integrator = RK4()
        return integrator.step(state, ops.rhs, dt)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        mesh = state.mesh
        ops = BurgersOps(dx=mesh.dx[0], N=mesh.shape[0], nu=self._nu)
        integrator = RK4()
        return integrator.solve(
            state, ops.rhs, t_span, dt,
            observables=observables, max_steps=max_steps,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-II.2 through II.10  Scaffold ProblemSpecs (V0.1)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CompressibleFlowSpec:
    """PHY-II.2: 1-D Sod shock tube (Euler equations)."""
    gamma: float = 1.4
    @property
    def name(self) -> str: return "SodShockTube1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"gamma": self.gamma}
    @property
    def governing_equations(self) -> str: return r"\partial_t \mathbf{U} + \partial_x \mathbf{F}(\mathbf{U}) = 0"
    @property
    def field_names(self) -> Sequence[str]: return ("density", "velocity", "pressure")
    @property
    def observable_names(self) -> Sequence[str]: return ("total_mass", "total_energy")


@dataclass(frozen=True)
class TurbulenceSpec:
    """PHY-II.3: 1-D Burgers turbulence (decaying)."""
    nu: float = 0.001
    @property
    def name(self) -> str: return "BurgersTurbulence1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"nu": self.nu}
    @property
    def governing_equations(self) -> str: return r"\partial_t u + u\partial_x u = \nu\partial_{xx} u"
    @property
    def field_names(self) -> Sequence[str]: return ("u",)
    @property
    def observable_names(self) -> Sequence[str]: return ("energy_spectrum",)


@dataclass(frozen=True)
class MultiphaseSpec:
    """PHY-II.4: 1-D advection of phase-field marker."""
    @property
    def name(self) -> str: return "PhaseAdvection1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {}
    @property
    def governing_equations(self) -> str: return r"\partial_t \phi + u \partial_x \phi = 0"
    @property
    def field_names(self) -> Sequence[str]: return ("phi",)
    @property
    def observable_names(self) -> Sequence[str]: return ("total_phi",)


@dataclass(frozen=True)
class ReactiveFlowSpec:
    """PHY-II.5: 1-D premixed flame (scalar transport + reaction)."""
    Da: float = 10.0
    @property
    def name(self) -> str: return "PremixedFlame1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"Da": self.Da}
    @property
    def governing_equations(self) -> str: return r"\partial_t Y + u\partial_x Y = D\partial_{xx}Y + \omega(Y)"
    @property
    def field_names(self) -> Sequence[str]: return ("species_Y",)
    @property
    def observable_names(self) -> Sequence[str]: return ("flame_speed",)


@dataclass(frozen=True)
class RarefiedGasSpec:
    """PHY-II.6: Free-streaming Boltzmann (collisionless)."""
    @property
    def name(self) -> str: return "FreeStreamBoltzmann1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {}
    @property
    def governing_equations(self) -> str: return r"\partial_t f + v\partial_x f = 0"
    @property
    def field_names(self) -> Sequence[str]: return ("f",)
    @property
    def observable_names(self) -> Sequence[str]: return ("density",)


@dataclass(frozen=True)
class ShallowWaterSpec:
    """PHY-II.7: 1-D shallow water (dam break)."""
    g: float = 9.81
    @property
    def name(self) -> str: return "ShallowWaterDamBreak1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"g": self.g}
    @property
    def governing_equations(self) -> str: return r"\partial_t h + \partial_x(hu)=0;\;\partial_t(hu)+\partial_x(hu^2+gh^2/2)=0"
    @property
    def field_names(self) -> Sequence[str]: return ("h", "hu")
    @property
    def observable_names(self) -> Sequence[str]: return ("total_mass",)


@dataclass(frozen=True)
class NonNewtonianSpec:
    """PHY-II.8: 1-D Couette flow with Bingham fluid."""
    tau_y: float = 1.0
    mu_p: float = 0.1
    @property
    def name(self) -> str: return "BinghamCouette1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"tau_y": self.tau_y, "mu_p": self.mu_p}
    @property
    def governing_equations(self) -> str: return r"\partial_t u = \partial_x[\mu_{eff}\partial_x u]"
    @property
    def field_names(self) -> Sequence[str]: return ("u",)
    @property
    def observable_names(self) -> Sequence[str]: return ("wall_shear",)


@dataclass(frozen=True)
class PorousMediaSpec:
    """PHY-II.9: 1-D Darcy flow in porous column."""
    K: float = 1e-10
    mu: float = 1e-3
    @property
    def name(self) -> str: return "DarcyColumn1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"K": self.K, "mu": self.mu}
    @property
    def governing_equations(self) -> str: return r"u = -\frac{K}{\mu}\nabla p"
    @property
    def field_names(self) -> Sequence[str]: return ("pressure",)
    @property
    def observable_names(self) -> Sequence[str]: return ("flow_rate",)


@dataclass(frozen=True)
class FreeSurfaceSpec:
    """PHY-II.10: 1-D thin film equation."""
    @property
    def name(self) -> str: return "ThinFilm1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {}
    @property
    def governing_equations(self) -> str: return r"\partial_t h + \partial_x(h^3\partial_{xxx}h) = 0"
    @property
    def field_names(self) -> Sequence[str]: return ("h",)
    @property
    def observable_names(self) -> Sequence[str]: return ("total_volume",)


# ═══════════════════════════════════════════════════════════════════════════════
# V0.2 Physics Solvers for PHY-II.2 through II.10
# ═══════════════════════════════════════════════════════════════════════════════


class CompressibleFlowSolver:
    """PHY-II.2: 1-D Euler equations — Sod shock tube via Rusanov scheme.

    Conservative variables  U = [ρ, ρu, E].
    Flux  F(U) = [ρu, ρu² + p, u(E + p)].
    EOS: p = (γ − 1)(E − ½ρu²).
    Outflow (zero-gradient) boundary conditions via ghost cells.
    """

    def __init__(self, gamma: float = 1.4) -> None:
        self._gamma = gamma

    @property
    def name(self) -> str:
        return "CompressibleFlow_Rusanov"

    # ── primitive ↔ conservative helpers ──────────────────────────────────

    def _to_conservative(
        self, rho: Tensor, u: Tensor, p: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Primitive (ρ, u, p) → conservative (ρ, ρu, E)."""
        rho_u = rho * u
        E = p / (self._gamma - 1.0) + 0.5 * rho * u ** 2
        return rho, rho_u, E

    def _to_primitive(
        self, rho: Tensor, rho_u: Tensor, E: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Conservative → primitive with positivity enforcement."""
        rho = torch.clamp(rho, min=1e-14)
        u = rho_u / rho
        p = (self._gamma - 1.0) * (E - 0.5 * rho * u ** 2)
        p = torch.clamp(p, min=1e-14)
        return rho, u, p

    def _euler_flux(
        self, rho: Tensor, rho_u: Tensor, E: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute physical flux vector F(U)."""
        rho_s = torch.clamp(rho, min=1e-14)
        u = rho_u / rho_s
        p = (self._gamma - 1.0) * (E - 0.5 * rho_s * u ** 2)
        p = torch.clamp(p, min=1e-14)
        return rho_u, rho_u * u + p, u * (E + p)

    def _sound_speed(self, rho: Tensor, p: Tensor) -> Tensor:
        """Speed of sound c = √(γ p / ρ)."""
        return torch.sqrt(
            self._gamma * torch.clamp(p, min=1e-14)
            / torch.clamp(rho, min=1e-14)
        )

    # ── time stepping ─────────────────────────────────────────────────────

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        rho = state.get_field("density").data.to(torch.float64)
        vel = state.get_field("velocity").data.to(torch.float64)
        prs = state.get_field("pressure").data.to(torch.float64)
        dx: float = state.mesh.dx[0]

        rho_c, mom, E = self._to_conservative(rho, vel, prs)
        U = torch.stack([rho_c, mom, E])  # (3, N)

        # Zero-gradient ghost cells for outflow BCs
        U_ext = torch.cat([U[:, 0:1], U, U[:, -1:]], dim=1)  # (3, N+2)

        U_L = U_ext[:, :-1]  # (3, N+1)  left state at each face
        U_R = U_ext[:, 1:]   # (3, N+1)  right state

        # Physical fluxes at left / right states
        fL0, fL1, fL2 = self._euler_flux(U_L[0], U_L[1], U_L[2])
        fR0, fR1, fR2 = self._euler_flux(U_R[0], U_R[1], U_R[2])
        FL = torch.stack([fL0, fL1, fL2])
        FR = torch.stack([fR0, fR1, fR2])

        # Maximum wave speed at each face
        _, uL_p, pL = self._to_primitive(U_L[0], U_L[1], U_L[2])
        _, uR_p, pR = self._to_primitive(U_R[0], U_R[1], U_R[2])
        alpha = torch.maximum(
            uL_p.abs() + self._sound_speed(U_L[0], pL),
            uR_p.abs() + self._sound_speed(U_R[0], pR),
        )  # (N+1,)

        # Rusanov numerical flux at each face
        F_half = 0.5 * (FL + FR) - 0.5 * alpha.unsqueeze(0) * (U_R - U_L)

        # Conservative update
        U_new = U - (dt / dx) * (F_half[:, 1:] - F_half[:, :-1])

        new_rho, new_vel, new_prs = self._to_primitive(
            U_new[0], U_new[1], U_new[2],
        )

        mesh = state.mesh
        fields = dict(state.fields)
        fields["density"] = FieldData(name="density", data=new_rho, mesh=mesh)
        fields["velocity"] = FieldData(name="velocity", data=new_vel, mesh=mesh)
        fields["pressure"] = FieldData(name="pressure", data=new_prs, mesh=mesh)
        return state.advance(dt, fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        steps = 0
        limit = max_steps or int((tf - t0) / dt) + 1

        dx: float = state.mesh.dx[0]
        rho0 = state.get_field("density").data.to(torch.float64)
        initial_mass = rho0.sum().item() * dx

        while state.t < tf - 1e-14 and steps < limit:
            state = self.step(state, min(dt, tf - state.t))
            steps += 1

        rho_f = state.get_field("density").data.to(torch.float64)
        final_mass = rho_f.sum().item() * dx

        return SolveResult(
            final_state=state,
            t_final=state.t,
            steps_taken=steps,
            metadata={
                "scheme": "Rusanov",
                "gamma": self._gamma,
                "initial_mass": initial_mass,
                "final_mass": final_mass,
                "mass_conservation_error": abs(final_mass - initial_mass)
                / max(abs(initial_mass), 1e-15),
            },
        )


class TurbulenceSolver:
    """PHY-II.3: Burgers turbulence with broadband IC; ν = 0.001.

    Uses RK4 time integration + conservative central advective flux + central
    diffusion (identical numerics to the anchor BurgersSolver, but in the
    low-viscosity regime).  Validation: kinetic energy must decay monotonically.
    """

    def __init__(self, nu: float = 0.001) -> None:
        self._nu = nu

    @property
    def name(self) -> str:
        return "BurgersTurbulence_RK4"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        mesh = state.mesh
        ops = BurgersOps(dx=mesh.dx[0], N=mesh.shape[0], nu=self._nu)
        integrator = RK4()
        return integrator.step(state, ops.rhs, dt)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        steps = 0
        limit = max_steps or int((tf - t0) / dt) + 1
        dx: float = state.mesh.dx[0]

        u0 = state.get_field("u").data.to(torch.float64)
        initial_ke = 0.5 * (u0 ** 2).sum().item() * dx
        ke_history: List[float] = [initial_ke]

        while state.t < tf - 1e-14 and steps < limit:
            state = self.step(state, min(dt, tf - state.t))
            steps += 1
            ke = 0.5 * (state.get_field("u").data ** 2).sum().item() * dx
            ke_history.append(ke)

        monotonic_decay = all(
            ke_history[i] >= ke_history[i + 1] - 1e-12
            for i in range(len(ke_history) - 1)
        )

        return SolveResult(
            final_state=state,
            t_final=state.t,
            steps_taken=steps,
            metadata={
                "nu": self._nu,
                "initial_ke": initial_ke,
                "final_ke": ke_history[-1],
                "ke_decay_ratio": ke_history[-1] / max(initial_ke, 1e-15),
                "energy_monotonic_decay": monotonic_decay,
            },
        )


class MultiphaseSolver:
    """PHY-II.4: Phase advection  ∂φ/∂t + c ∂φ/∂x = 0  (first-order upwind).

    Periodic BCs.  Validates: shape preservation and conservation of ∫φ dx.
    """

    def __init__(self, c: float = 1.0) -> None:
        self._c = c

    @property
    def name(self) -> str:
        return "PhaseAdvection_Upwind"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        phi = state.get_field("phi").data.to(torch.float64)
        dx: float = state.mesh.dx[0]
        c = self._c

        if c >= 0.0:
            # upwind: backward difference
            dphi = -c * (phi - torch.roll(phi, 1)) / dx
        else:
            # upwind: forward difference
            dphi = -c * (torch.roll(phi, -1) - phi) / dx

        new_phi = phi + dt * dphi
        fields = dict(state.fields)
        fields["phi"] = FieldData(name="phi", data=new_phi, mesh=state.mesh)
        return state.advance(dt, fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        steps = 0
        limit = max_steps or int((tf - t0) / dt) + 1
        dx: float = state.mesh.dx[0]

        phi0 = state.get_field("phi").data.to(torch.float64)
        initial_integral = phi0.sum().item() * dx

        while state.t < tf - 1e-14 and steps < limit:
            state = self.step(state, min(dt, tf - state.t))
            steps += 1

        phi_f = state.get_field("phi").data.to(torch.float64)
        final_integral = phi_f.sum().item() * dx

        return SolveResult(
            final_state=state,
            t_final=state.t,
            steps_taken=steps,
            metadata={
                "advection_speed": self._c,
                "initial_integral": initial_integral,
                "final_integral": final_integral,
                "conservation_error": abs(final_integral - initial_integral)
                / max(abs(initial_integral), 1e-15),
            },
        )


class ReactiveFlowSolver:
    """PHY-II.5: 1-D reaction-diffusion  ∂Y/∂t = D ∂²Y/∂x² − Da·Y(1−Y).

    Strang splitting: half-step diffusion (FTCS central), full-step reaction
    (analytical logistic ODE), half-step diffusion.  Periodic BCs.
    """

    def __init__(self, D: float = 0.01, Da: float = 10.0) -> None:
        self._D = D
        self._Da = Da

    @property
    def name(self) -> str:
        return "ReactiveFlow_StrangSplit"

    # ── splitting operators ───────────────────────────────────────────────

    def _diffusion_half(self, Y: Tensor, dx: float, dt_half: float) -> Tensor:
        """Forward Euler central-difference diffusion (periodic)."""
        laplacian = (
            torch.roll(Y, -1) - 2.0 * Y + torch.roll(Y, 1)
        ) / (dx ** 2)
        return Y + dt_half * self._D * laplacian

    def _reaction_full(self, Y: Tensor, dt: float) -> Tensor:
        """Exact logistic ODE:  dY/dt = −Da Y(1−Y).

        Solution: Y(t+dt) = Y exp(−Da dt) / (1 − Y + Y exp(−Da dt)).
        Stable for Y ∈ [0, 1]; clamped for safety.
        """
        exp_neg = torch.exp(torch.tensor(-self._Da * dt, dtype=Y.dtype))
        numer = Y * exp_neg
        denom = (1.0 - Y) + Y * exp_neg
        return torch.clamp(numer / torch.clamp(denom, min=1e-30), 0.0, 1.0)

    # ── time stepping ─────────────────────────────────────────────────────

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        Y = state.get_field("species_Y").data.to(torch.float64)
        dx: float = state.mesh.dx[0]

        Y = self._diffusion_half(Y, dx, 0.5 * dt)
        Y = self._reaction_full(Y, dt)
        Y = self._diffusion_half(Y, dx, 0.5 * dt)

        fields = dict(state.fields)
        fields["species_Y"] = FieldData(
            name="species_Y", data=Y, mesh=state.mesh,
        )
        return state.advance(dt, fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        steps = 0
        limit = max_steps or int((tf - t0) / dt) + 1
        dx: float = state.mesh.dx[0]

        Y0 = state.get_field("species_Y").data.to(torch.float64)
        initial_integral = Y0.sum().item() * dx

        while state.t < tf - 1e-14 and steps < limit:
            state = self.step(state, min(dt, tf - state.t))
            steps += 1

        Yf = state.get_field("species_Y").data.to(torch.float64)
        final_integral = Yf.sum().item() * dx

        return SolveResult(
            final_state=state,
            t_final=state.t,
            steps_taken=steps,
            metadata={
                "D": self._D,
                "Da": self._Da,
                "splitting": "Strang",
                "initial_Y_integral": initial_integral,
                "final_Y_integral": final_integral,
                "Y_min": Yf.min().item(),
                "Y_max": Yf.max().item(),
            },
        )


class RarefiedGasSolver:
    """PHY-II.6: Collisionless Boltzmann  ∂f/∂t + v ∂f/∂x = 0.

    Single velocity v (default 1.0).  First-order upwind with periodic BCs.
    Exact solution: f(x, t) = f₀(x − v t).
    """

    def __init__(self, v: float = 1.0) -> None:
        self._v = v

    @property
    def name(self) -> str:
        return "RarefiedGas_Upwind"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        f = state.get_field("f").data.to(torch.float64)
        dx: float = state.mesh.dx[0]
        v = self._v

        if v >= 0.0:
            df = -v * (f - torch.roll(f, 1)) / dx
        else:
            df = -v * (torch.roll(f, -1) - f) / dx

        new_f = f + dt * df
        fields = dict(state.fields)
        fields["f"] = FieldData(name="f", data=new_f, mesh=state.mesh)
        return state.advance(dt, fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        steps = 0
        limit = max_steps or int((tf - t0) / dt) + 1
        dx: float = state.mesh.dx[0]

        f0 = state.get_field("f").data.to(torch.float64)
        initial_mass = f0.sum().item() * dx

        while state.t < tf - 1e-14 and steps < limit:
            state = self.step(state, min(dt, tf - state.t))
            steps += 1

        ff = state.get_field("f").data.to(torch.float64)
        final_mass = ff.sum().item() * dx

        # Compare against exact shifted solution (periodic domain)
        elapsed = state.t - t0
        shift_cells = int(round(self._v * elapsed / dx))
        f_exact = torch.roll(f0, -shift_cells)
        l2_err = torch.sqrt(((ff - f_exact) ** 2).sum() * dx).item()

        return SolveResult(
            final_state=state,
            t_final=state.t,
            steps_taken=steps,
            metadata={
                "velocity": self._v,
                "initial_mass": initial_mass,
                "final_mass": final_mass,
                "mass_conservation_error": abs(final_mass - initial_mass)
                / max(abs(initial_mass), 1e-15),
                "l2_error_vs_exact_shift": l2_err,
            },
        )


class ShallowWaterSolver:
    """PHY-II.7: 1-D shallow water equations — dam break via Rusanov scheme.

    Conservative variables  [h, hu].
    Flux  F = [hu,  hu²/h + g h²/2].
    Wave speed: |u| + √(g h).
    Zero-gradient outflow BCs via ghost cells.
    """

    def __init__(self, g: float = 9.81) -> None:
        self._g = g

    @property
    def name(self) -> str:
        return "ShallowWater_Rusanov"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        h = state.get_field("h").data.to(torch.float64)
        hu = state.get_field("hu").data.to(torch.float64)
        dx: float = state.mesh.dx[0]
        g = self._g

        U = torch.stack([h, hu])  # (2, N)

        # Ghost cells (zero-gradient outflow)
        U_ext = torch.cat([U[:, 0:1], U, U[:, -1:]], dim=1)  # (2, N+2)

        U_L = U_ext[:, :-1]  # (2, N+1)
        U_R = U_ext[:, 1:]

        h_L = torch.clamp(U_L[0], min=1e-14)
        hu_L = U_L[1]
        u_L = hu_L / h_L
        h_R = torch.clamp(U_R[0], min=1e-14)
        hu_R = U_R[1]
        u_R = hu_R / h_R

        # Physical fluxes
        FL = torch.stack([hu_L, hu_L * u_L + 0.5 * g * h_L ** 2])
        FR = torch.stack([hu_R, hu_R * u_R + 0.5 * g * h_R ** 2])

        # Maximum wave speed at each face
        alpha = torch.maximum(
            u_L.abs() + torch.sqrt(g * h_L),
            u_R.abs() + torch.sqrt(g * h_R),
        )

        # Rusanov numerical flux
        F_half = 0.5 * (FL + FR) - 0.5 * alpha.unsqueeze(0) * (U_R - U_L)

        # Conservative update
        U_new = U - (dt / dx) * (F_half[:, 1:] - F_half[:, :-1])

        new_h = torch.clamp(U_new[0], min=1e-14)
        new_hu = U_new[1]

        mesh = state.mesh
        fields = dict(state.fields)
        fields["h"] = FieldData(name="h", data=new_h, mesh=mesh)
        fields["hu"] = FieldData(name="hu", data=new_hu, mesh=mesh)
        return state.advance(dt, fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        steps = 0
        limit = max_steps or int((tf - t0) / dt) + 1
        dx: float = state.mesh.dx[0]
        g = self._g

        h0 = state.get_field("h").data.to(torch.float64)
        initial_mass = h0.sum().item() * dx

        while state.t < tf - 1e-14 and steps < limit:
            state = self.step(state, min(dt, tf - state.t))
            steps += 1

        hf = state.get_field("h").data.to(torch.float64)
        final_mass = hf.sum().item() * dx

        # Ritter's intermediate depth for dam-break reference
        h_L_val = h0.max().item()
        h_R_val = h0.min().item()
        h_m_ritter = (1.0 / g) * (
            0.5 * (math.sqrt(g * h_L_val) + math.sqrt(g * h_R_val))
        ) ** 2

        return SolveResult(
            final_state=state,
            t_final=state.t,
            steps_taken=steps,
            metadata={
                "g": g,
                "initial_mass": initial_mass,
                "final_mass": final_mass,
                "mass_conservation_error": abs(final_mass - initial_mass)
                / max(abs(initial_mass), 1e-15),
                "ritter_h_intermediate": h_m_ritter,
            },
        )


class NonNewtonianSolver:
    """PHY-II.8: Analytical Bingham–Poiseuille channel flow.

    Pressure-driven channel [y₀, y₀ + H].  Steady-state analytical profile:
      plug region |y − center| ≤ y_c:  u = u_plug = G(H/2 − y_c)² / (2 μ_p)
      shear region:  u(d) = (G / 2μ_p)(H/2 − d)(H/2 + d − 2 y_c)
    where d = |y − center|, G = |dP/dx|, y_c = τ_y / G.
    """

    def __init__(
        self,
        tau_y: float = 1.0,
        mu_p: float = 0.1,
        dP_dx: float = -2.0,
    ) -> None:
        self._tau_y = tau_y
        self._mu_p = mu_p
        self._dP_dx = dP_dx

    @property
    def name(self) -> str:
        return "BinghamCouette_Analytical"

    def _analytical_profile(self, mesh: Any) -> Tensor:
        """Compute exact Bingham–Poiseuille velocity on *mesh*."""
        y = mesh.cell_centers().squeeze(-1).to(torch.float64)
        lo, hi = mesh.domain[0]
        H = hi - lo
        H_half = H / 2.0
        center = 0.5 * (lo + hi)
        G = abs(self._dP_dx)
        tau_y = self._tau_y
        mu_p = self._mu_p

        y_c = tau_y / G  # plug half-width

        u = torch.zeros_like(y)
        if y_c >= H_half:
            return u  # yield stress exceeds driving force — no flow

        u_plug = G * (H_half - y_c) ** 2 / (2.0 * mu_p)
        d = (y - center).abs()

        sheared = d > y_c
        u[sheared] = (
            (G / (2.0 * mu_p))
            * (H_half - d[sheared])
            * (H_half + d[sheared] - 2.0 * y_c)
        )
        u[~sheared] = u_plug

        return torch.clamp(u, min=0.0)

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        u_a = self._analytical_profile(state.mesh)
        fields = dict(state.fields)
        fields["u"] = FieldData(name="u", data=u_a, mesh=state.mesh)
        return state.advance(dt, fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        u_a = self._analytical_profile(state.mesh)
        fields = dict(state.fields)
        fields["u"] = FieldData(name="u", data=u_a, mesh=state.mesh)
        final = state.advance(t_span[1] - t_span[0], fields)

        G = abs(self._dP_dx)
        H = state.mesh.domain[0][1] - state.mesh.domain[0][0]
        H_half = H / 2.0
        y_c = self._tau_y / G
        u_plug = (
            G * (H_half - y_c) ** 2 / (2.0 * self._mu_p)
            if y_c < H_half
            else 0.0
        )

        return SolveResult(
            final_state=final,
            t_final=t_span[1],
            steps_taken=1,
            metadata={
                "solution_type": "analytical",
                "tau_y": self._tau_y,
                "mu_p": self._mu_p,
                "dP_dx": self._dP_dx,
                "plug_half_width_y_c": y_c,
                "plug_velocity": u_plug,
                "reference_u_max": u_plug,
            },
        )


class PorousMediaSolver:
    """PHY-II.9: 1-D Darcy diffusion  ∂P/∂t = (K / μ φ) ∂²P/∂x².

    Dirichlet BCs via ghost-cell mirroring.
    Steady state: P(x) = P_left + (P_right − P_left)(x − x₀) / L.
    """

    def __init__(
        self,
        K: float = 1e-10,
        mu: float = 1e-3,
        porosity: float = 0.3,
        P_left: float = 1.0,
        P_right: float = 0.0,
    ) -> None:
        self._K = K
        self._mu = mu
        self._porosity = porosity
        self._P_left = P_left
        self._P_right = P_right
        self._alpha = K / (mu * porosity)

    @property
    def name(self) -> str:
        return "PorousMedia_Darcy"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        P = state.get_field("pressure").data.to(torch.float64)
        dx: float = state.mesh.dx[0]

        # Ghost cells enforce Dirichlet at domain faces:
        #   P(x_left) = P_left → P_ghost = 2·P_left − P[0]
        P_gL = (2.0 * self._P_left - P[0]).unsqueeze(0)
        P_gR = (2.0 * self._P_right - P[-1]).unsqueeze(0)
        P_ext = torch.cat([P_gL, P, P_gR])

        laplacian = (P_ext[2:] - 2.0 * P_ext[1:-1] + P_ext[:-2]) / (dx ** 2)
        new_P = P + dt * self._alpha * laplacian

        fields = dict(state.fields)
        fields["pressure"] = FieldData(
            name="pressure", data=new_P, mesh=state.mesh,
        )
        return state.advance(dt, fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        steps = 0
        limit = max_steps or int((tf - t0) / dt) + 1

        while state.t < tf - 1e-14 and steps < limit:
            state = self.step(state, min(dt, tf - state.t))
            steps += 1

        # Analytical steady-state reference
        x = state.mesh.cell_centers().squeeze(-1).to(torch.float64)
        x0, xL = state.mesh.domain[0]
        L = xL - x0
        P_steady = self._P_left + (self._P_right - self._P_left) * (x - x0) / L
        P_final = state.get_field("pressure").data.to(torch.float64)
        error = (P_final - P_steady).abs().max().item()

        darcy_u = -self._K / self._mu * (self._P_right - self._P_left) / L

        return SolveResult(
            final_state=state,
            t_final=state.t,
            steps_taken=steps,
            metadata={
                "K": self._K,
                "mu": self._mu,
                "porosity": self._porosity,
                "alpha": self._alpha,
                "darcy_velocity": darcy_u,
                "steady_state_linf_error": error,
                "reference_solution": "linear_pressure",
            },
        )


class FreeSurfaceSolver:
    """PHY-II.10: 1-D thin film equation  ∂h/∂t = −(1/3) ∂/∂x [h³ ∂³h/∂x³].

    FTCS with conservative face fluxes.  Periodic BCs.
    Linear stability: perturbation mode k decays as exp(−k⁴ t / 3).
    """

    @property
    def name(self) -> str:
        return "FreeSurface_ThinFilm"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        h = state.get_field("h").data.to(torch.float64)
        dx: float = state.mesh.dx[0]

        # Face-centered h at i+1/2
        h_face = 0.5 * (h + torch.roll(h, -1))

        # Third derivative at face i+1/2 (second-order centered):
        #   h'''_{i+1/2} ≈ (h_{i+2} − 3 h_{i+1} + 3 h_i − h_{i−1}) / dx³
        h_xxx = (
            torch.roll(h, -2)
            - 3.0 * torch.roll(h, -1)
            + 3.0 * h
            - torch.roll(h, 1)
        ) / (dx ** 3)

        # Flux J_{i+1/2} = (h_face³ / 3) · h'''
        J = (h_face ** 3 / 3.0) * h_xxx

        # Conservative update:  dh/dt = −(J_{i+1/2} − J_{i−1/2}) / dx
        dh_dt = -(J - torch.roll(J, 1)) / dx

        new_h = h + dt * dh_dt

        fields = dict(state.fields)
        fields["h"] = FieldData(name="h", data=new_h, mesh=state.mesh)
        return state.advance(dt, fields)

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        steps = 0
        limit = max_steps or int((tf - t0) / dt) + 1
        dx: float = state.mesh.dx[0]

        h0 = state.get_field("h").data.to(torch.float64)
        initial_volume = h0.sum().item() * dx

        while state.t < tf - 1e-14 and steps < limit:
            state = self.step(state, min(dt, tf - state.t))
            steps += 1

        hf = state.get_field("h").data.to(torch.float64)
        final_volume = hf.sum().item() * dx

        # Linear stability reference: mode k = 2π/L decays as exp(−k⁴ t / 3)
        L = state.mesh.domain[0][1] - state.mesh.domain[0][0]
        k = 2.0 * math.pi / L
        elapsed = state.t - t0
        decay_rate = k ** 4 / 3.0
        amp0 = (h0 - h0.mean()).abs().max().item()
        predicted_amp = amp0 * math.exp(-decay_rate * elapsed)
        actual_amp = (hf - hf.mean()).abs().max().item()

        return SolveResult(
            final_state=state,
            t_final=state.t,
            steps_taken=steps,
            metadata={
                "initial_volume": initial_volume,
                "final_volume": final_volume,
                "volume_conservation_error": abs(final_volume - initial_volume)
                / max(abs(initial_volume), 1e-15),
                "predicted_amplitude": predicted_amp,
                "actual_amplitude": actual_amp,
                "decay_rate_theoretical": decay_rate,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pack registration
# ═══════════════════════════════════════════════════════════════════════════════


class FluidDynamicsPack(DomainPack):
    """Pack II: Fluid Dynamics."""

    @property
    def pack_id(self) -> str:
        return "II"

    @property
    def pack_name(self) -> str:
        return "Fluid Dynamics"

    @property
    def taxonomy_ids(self) -> Sequence[str]:
        return tuple(f"PHY-II.{i}" for i in range(1, 11))

    def problem_specs(self) -> Dict[str, Type[ProblemSpec]]:
        return {
            "PHY-II.1": BurgersSpec,
            "PHY-II.2": CompressibleFlowSpec,
            "PHY-II.3": TurbulenceSpec,
            "PHY-II.4": MultiphaseSpec,
            "PHY-II.5": ReactiveFlowSpec,
            "PHY-II.6": RarefiedGasSpec,
            "PHY-II.7": ShallowWaterSpec,
            "PHY-II.8": NonNewtonianSpec,
            "PHY-II.9": PorousMediaSpec,
            "PHY-II.10": FreeSurfaceSpec,
        }

    def solvers(self) -> Dict[str, Type[Solver]]:
        return {
            "PHY-II.1": BurgersSolver,
            "PHY-II.2": CompressibleFlowSolver,
            "PHY-II.3": TurbulenceSolver,
            "PHY-II.4": MultiphaseSolver,
            "PHY-II.5": ReactiveFlowSolver,
            "PHY-II.6": RarefiedGasSolver,
            "PHY-II.7": ShallowWaterSolver,
            "PHY-II.8": NonNewtonianSolver,
            "PHY-II.9": PorousMediaSolver,
            "PHY-II.10": FreeSurfaceSolver,
        }

    def discretizations(self) -> Dict[str, Sequence[Type[Discretization]]]:
        return {"PHY-II.1": [FVM_Burgers_1D]}

    def observables(self) -> Dict[str, Sequence[Type[Observable]]]:
        return {"PHY-II.1": [BurgersL2Observable, BurgersIntegralObservable, BurgersKEObservable]}

    def benchmarks(self) -> Dict[str, Sequence[str]]:
        return {
            "PHY-II.1": ["burgers_sinusoidal_convergence"],
            "PHY-II.2": ["sod_shock_tube"],
            "PHY-II.7": ["dam_break_1d"],
        }

    def version(self) -> str:
        return "0.4.0"


get_registry().register_pack(FluidDynamicsPack())


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor Vertical Slice — run_fluids_vertical_slice
# ═══════════════════════════════════════════════════════════════════════════════


def _run_burgers(
    N: int,
    spec: BurgersSpec,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Burgers at resolution N, return metrics."""
    L = spec.L
    mesh = StructuredMesh(shape=(N,), domain=((0.0, L),))
    dx = mesh.dx[0]

    # CFL: dt ≤ min(dx/u_max, dx²/(2ν))
    u_max = 1.5  # expected max |u| for sin IC
    dt_adv = dx / u_max
    dt_diff = dx ** 2 / (2.0 * spec.nu) if spec.nu > 0.0 else float("inf")
    dt = 0.3 * min(dt_adv, dt_diff)

    # IC: u(x,0) = sin(x)
    x = mesh.cell_centers().squeeze(-1)
    u0_data = torch.sin(x)
    field = FieldData(name="u", data=u0_data, mesh=mesh)
    state0 = SimulationState(t=0.0, fields={"u": field}, mesh=mesh)

    ops = BurgersOps(dx=dx, N=N, nu=spec.nu)
    ke_obs = BurgersKEObservable(dx)
    int_obs = BurgersIntegralObservable(dx)

    with ReproducibilityContext(seed=seed) as ctx:
        integrator = RK4()
        result = integrator.solve(
            state0, ops.rhs,
            t_span=(0.0, spec.T_final), dt=dt,
            observables=[ke_obs, int_obs],
        )
        ctx.record("final_u", hash_tensor(result.final_state.get_field("u").data))

    # Conservation check: ∫u dx should be 0 for sin(x) on [0, 2π] (periodic)
    int_hist = result.observable_history.get("integral_u", [])
    conservation_ok = True
    if int_hist:
        initial_int = u0_data.sum().item() * dx
        max_drift = max(abs(v.item() - initial_int) for v in int_hist)
        # Machine-precision conservation for central flux (no numerical dissipation)
        conservation_ok = max_drift < 1e-10

    return {
        "N": N, "dt": dt, "steps": result.steps_taken,
        "final_state": result.final_state,
        "conservation_ok": conservation_ok,
        "provenance": ctx.provenance(),
    }


def _compute_reference(spec: BurgersSpec, N_ref: int = 2048, seed: int = 42) -> Tensor:
    """Compute high-resolution reference solution."""
    result = _run_burgers(N_ref, spec, seed=seed)
    return result["final_state"].get_field("u").data


def run_fluids_vertical_slice(
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute Pack II anchor vertical slice (Burgers) at V0.4."""
    spec = BurgersSpec(nu=0.01, T_final=0.5)

    # Reference on very fine grid
    ref = _compute_reference(spec, N_ref=2048, seed=seed)

    resolutions = [128, 256, 512]
    runs = {N: _run_burgers(N, spec, seed=seed) for N in resolutions}

    # Errors vs reference (interpolated via downsampling)
    errors = []
    for N in resolutions:
        u_num = runs[N]["final_state"].get_field("u").data
        # Downsample reference to match resolution N
        factor = 2048 // N
        ref_down = ref.reshape(-1, factor).mean(dim=-1)
        errors.append(compute_linf_error(u_num, ref_down))

    orders = convergence_order(errors, resolutions)

    # Determinism
    run2 = _run_burgers(resolutions[-1], spec, seed=seed)
    det_diff = (
        run2["final_state"].get_field("u").data
        - runs[resolutions[-1]]["final_state"].get_field("u").data
    ).abs().max().item()
    deterministic = det_diff == 0.0

    metrics = {
        "problem": spec.name,
        "resolutions": resolutions,
        "linf_errors": errors,
        "convergence_orders": orders,
        "finest_linf": errors[-1],
        "conservation": all(runs[N]["conservation_ok"] for N in resolutions),
        "deterministic": deterministic,
    }

    if verbose:
        print("=" * 72)
        print("  ANCHOR VERTICAL SLICE — Pack II: 1-D Viscous Burgers")
        print("=" * 72)
        print(f"  ν={spec.nu}, L=2π, T={spec.T_final}")
        print()
        for i, N in enumerate(resolutions):
            print(f"  N={N:>4}  L∞={errors[i]:.4e}  steps={runs[N]['steps']}")
        print()
        for i, o in enumerate(orders):
            print(f"  Order {resolutions[i]}→{resolutions[i+1]}: {o:.2f}")
        print(f"  Conservation:    {'PASS' if metrics['conservation'] else 'FAIL'}")
        print(f"  Deterministic:   {'PASS' if deterministic else 'FAIL'}")
        print()

        gates = {
            "L∞ error < 1e-3 (finest vs ref)": errors[-1] < 1e-3,
            "Convergence order > 1.5": all(o > 1.5 for o in orders),
            "Conservation (∫u dx ≈ const)": metrics["conservation"],
            "Deterministic": deterministic,
        }
        all_pass = all(gates.values())
        for label, ok in gates.items():
            print(f"  [{'✓' if ok else '✗'}] {label}")
        print(f"\n  RESULT: {'V0.4 VALIDATED' if all_pass else 'FAILED'}")
        print("=" * 72)

    return metrics


if __name__ == "__main__":
    m = run_fluids_vertical_slice()
    ok = (
        m["finest_linf"] < 1e-3
        and all(o > 1.5 for o in m["convergence_orders"])
        and m["conservation"]
        and m["deterministic"]
    )
    sys.exit(0 if ok else 1)
