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
# Scaffold solvers for non-anchor nodes (V0.1 — toy end-to-end)
# ═══════════════════════════════════════════════════════════════════════════════


class _GenericAdvectionSolver:
    """Generic linear advection solver for scaffold nodes."""

    def __init__(self, field_name: str = "u", c: float = 1.0) -> None:
        self._field_name = field_name
        self._c = c

    @property
    def name(self) -> str:
        return f"GenericAdvection_{self._field_name}"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        u = state.get_field(self._field_name).data
        dx = state.mesh.dx[0]
        # Upwind
        du = -self._c * (u - torch.roll(u, 1)) / dx
        new_data = u + dt * du
        new_field = FieldData(name=self._field_name, data=new_data, mesh=state.mesh)
        return state.advance(dt, {self._field_name: new_field})

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        steps = 0
        limit = max_steps or int((tf - t0) / dt) + 1
        while state.t < tf - 1e-14 and steps < limit:
            state = self.step(state, min(dt, tf - state.t))
            steps += 1
        return SolveResult(final_state=state, t_final=state.t, steps_taken=steps)


class CompressibleFlowSolver(_GenericAdvectionSolver):
    def __init__(self) -> None: super().__init__("density", 1.0)
    @property
    def name(self) -> str: return "CompressibleFlow_Scaffold"

class TurbulenceSolver(BurgersSolver):
    @property
    def name(self) -> str: return "BurgersTurbulence_Scaffold"

class MultiphaseSolver(_GenericAdvectionSolver):
    def __init__(self) -> None: super().__init__("phi", 1.0)
    @property
    def name(self) -> str: return "Multiphase_Scaffold"

class ReactiveFlowSolver(_GenericAdvectionSolver):
    def __init__(self) -> None: super().__init__("species_Y", 0.5)
    @property
    def name(self) -> str: return "ReactiveFlow_Scaffold"

class RarefiedGasSolver(_GenericAdvectionSolver):
    def __init__(self) -> None: super().__init__("f", 1.0)
    @property
    def name(self) -> str: return "RarefiedGas_Scaffold"

class ShallowWaterSolver(_GenericAdvectionSolver):
    def __init__(self) -> None: super().__init__("h", 1.0)
    @property
    def name(self) -> str: return "ShallowWater_Scaffold"

class NonNewtonianSolver(_GenericAdvectionSolver):
    def __init__(self) -> None: super().__init__("u", 0.1)
    @property
    def name(self) -> str: return "NonNewtonian_Scaffold"

class PorousMediaSolver(_GenericAdvectionSolver):
    def __init__(self) -> None: super().__init__("pressure", 0.01)
    @property
    def name(self) -> str: return "PorousMedia_Scaffold"

class FreeSurfaceSolver(_GenericAdvectionSolver):
    def __init__(self) -> None: super().__init__("h", 0.5)
    @property
    def name(self) -> str: return "FreeSurface_Scaffold"


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
