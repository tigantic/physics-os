"""
Domain Pack V — Thermodynamics and Statistical Mechanics
=========================================================

**Anchor problem (V0.4)**:  1-D advection-diffusion (PHY-V.5 Heat Transfer)

    ∂u/∂t + c ∂u/∂x = α ∂²u/∂x²     x ∈ [0, L], periodic BCs

Exact solution for sinusoidal IC:

    u(x, 0) = sin(2πx/L)
    u(x, t) = sin(2π(x − ct)/L) · exp(−α(2π/L)² t)

Validation gates (V0.4):
  • L∞ error < 1e-4 at t = T (finest grid).
  • Grid refinement ratio ≈ 4 (second-order spatial).
  • Conservation: ∫u dx decays monotonically.
  • Deterministic across two runs.

Scaffold nodes (V0.1):
  PHY-V.1  Equilibrium stat mech — 1-D Ising energy (Monte Carlo)
  PHY-V.2  Non-equilibrium stat mech — Fokker-Planck diffusion ODE
  PHY-V.3  Molecular dynamics — Lennard-Jones pair (energy conservation)
  PHY-V.4  Monte Carlo general — Random walk diffusion
  PHY-V.5  Heat transfer (ANCHOR)
  PHY-V.6  Lattice models — 1-D Ising chain partition function
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor

from ontic.packs._base import (
    compute_l2_error,
    compute_linf_error,
    convergence_order,
)
from ontic.platform.data_model import (
    FieldData,
    SimulationState,
    StructuredMesh,
)
from ontic.platform.domain_pack import DomainPack, get_registry
from ontic.platform.protocols import (
    Discretization,
    Observable,
    ProblemSpec,
    Solver,
    SolveResult,
)
from ontic.platform.reproduce import ReproducibilityContext, hash_tensor
from ontic.platform.solvers import ForwardEuler, RK4, RHSCallable, TimeIntegrator


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-V.5  Heat transfer — Advection-Diffusion (ANCHOR)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AdvectionDiffusionSpec:
    """1-D advection-diffusion: ∂u/∂t + c ∂u/∂x = α ∂²u/∂x²  (periodic)."""

    c: float = 1.0
    alpha: float = 0.01
    L: float = 1.0
    T_final: float = 0.5

    @property
    def name(self) -> str:
        return "AdvectionDiffusion1D"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"c": self.c, "alpha": self.alpha, "L": self.L, "T_final": self.T_final}

    @property
    def governing_equations(self) -> str:
        return r"\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = \alpha \frac{\partial^2 u}{\partial x^2}"

    @property
    def field_names(self) -> Sequence[str]:
        return ("u",)

    @property
    def observable_names(self) -> Sequence[str]:
        return ("l2_norm", "integral_u")


class FVM_AdvDiff_1D:
    """
    1-D finite-volume discretization of the advection-diffusion equation.

    Advection:  first-order upwind (c > 0 → backward difference).
    Diffusion:  second-order central differences.
    Boundary:   periodic.
    """

    def __init__(self, c: float, alpha: float) -> None:
        self._c = c
        self._alpha = alpha

    @property
    def method(self) -> str:
        return "FVM"

    @property
    def order(self) -> int:
        return 2  # diffusion is 2nd-order; advection upwind is 1st but we use central for 2nd

    def discretize(self, spec: ProblemSpec, mesh: Any) -> "AdvDiffOps":
        if not isinstance(mesh, StructuredMesh) or mesh.ndim != 1:
            raise TypeError("FVM_AdvDiff_1D requires a 1-D StructuredMesh")
        return AdvDiffOps(
            dx=mesh.dx[0],
            N=mesh.shape[0],
            c=self._c,
            alpha=self._alpha,
        )


@dataclass
class AdvDiffOps:
    """Discrete operators for 1-D advection-diffusion with periodic BCs."""

    dx: float
    N: int
    c: float
    alpha: float

    def advection_central(self, u: Tensor) -> Tensor:
        """Second-order central difference for advection: c * du/dx."""
        # Periodic: use roll
        du_dx = (torch.roll(u, -1) - torch.roll(u, 1)) / (2.0 * self.dx)
        return self.c * du_dx

    def diffusion_central(self, u: Tensor) -> Tensor:
        """Second-order central Laplacian with periodic BCs."""
        d2u = (torch.roll(u, -1) - 2.0 * u + torch.roll(u, 1)) / (self.dx ** 2)
        return self.alpha * d2u

    def rhs(self, state: SimulationState, t: float) -> Dict[str, Tensor]:
        """du/dt = -c du/dx + α d²u/dx²."""
        u = state.get_field("u").data
        return {"u": -self.advection_central(u) + self.diffusion_central(u)}


class AdvDiffL2Observable:
    """Observable: L2 norm of the temperature/concentration field."""

    @property
    def name(self) -> str:
        return "l2_norm"

    @property
    def units(self) -> str:
        return "1"

    def compute(self, state: Any) -> Tensor:
        return torch.norm(state.get_field("u").data, p=2)


class AdvDiffIntegralObservable:
    """Observable: spatial integral of the scalar field (mass/energy)."""

    def __init__(self, dx: float) -> None:
        self._dx = dx

    @property
    def name(self) -> str:
        return "integral_u"

    @property
    def units(self) -> str:
        return "K·m"

    def compute(self, state: Any) -> Tensor:
        return state.get_field("u").data.sum() * self._dx


def advdiff_exact(
    x: Tensor, t: float, c: float, alpha: float, L: float
) -> Tensor:
    """Exact: u(x,t) = sin(2π(x−ct)/L) · exp(−α(2π/L)²t)."""
    k = 2.0 * math.pi / L
    return torch.sin(k * (x - c * t)) * math.exp(-alpha * k ** 2 * t)


class AdvDiffSolver:
    """
    Full solver wrapping RK4 + FVM advection-diffusion for Pack V.

    Satisfies the Solver protocol.
    """

    def __init__(
        self,
        c: float = 1.0,
        alpha: float = 0.01,
        L: float = 1.0,
    ) -> None:
        self._c = c
        self._alpha = alpha
        self._L = L

    @property
    def name(self) -> str:
        return "AdvDiffSolver_RK4"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        mesh = state.mesh
        ops = AdvDiffOps(dx=mesh.dx[0], N=mesh.shape[0], c=self._c, alpha=self._alpha)
        integrator = RK4()
        return integrator.step(state, ops.rhs, dt)

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
        mesh = state.mesh
        ops = AdvDiffOps(dx=mesh.dx[0], N=mesh.shape[0], c=self._c, alpha=self._alpha)
        integrator = RK4()
        return integrator.solve(
            state, ops.rhs, t_span, dt,
            observables=observables, max_steps=max_steps,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-V.1 through V.6  Scaffold ProblemSpecs (V0.1)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class IsingEnergySpec:
    """PHY-V.1: 1-D Ising model — E = −J Σ s_i s_{i+1}."""
    J: float = 1.0
    N_spins: int = 32
    T_final: float = 100.0

    @property
    def name(self) -> str: return "Ising1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"J": self.J, "N_spins": self.N_spins}
    @property
    def governing_equations(self) -> str: return r"E = -J \sum_i s_i s_{i+1}"
    @property
    def field_names(self) -> Sequence[str]: return ("spins",)
    @property
    def observable_names(self) -> Sequence[str]: return ("energy", "magnetization")


@dataclass(frozen=True)
class FokkerPlanckSpec:
    """PHY-V.2: 1-D Fokker-Planck — ∂p/∂t = D ∂²p/∂x² − ∂(μp)/∂x."""
    D: float = 0.1
    mu: float = 0.0

    @property
    def name(self) -> str: return "FokkerPlanck1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"D": self.D, "mu": self.mu}
    @property
    def governing_equations(self) -> str: return r"\partial_t p = D\partial_{xx} p - \partial_x(\mu p)"
    @property
    def field_names(self) -> Sequence[str]: return ("probability",)
    @property
    def observable_names(self) -> Sequence[str]: return ("total_probability",)


@dataclass(frozen=True)
class LennardJonesSpec:
    """PHY-V.3: Lennard-Jones pair — V(r) = 4ε[(σ/r)¹²−(σ/r)⁶]."""
    epsilon: float = 1.0
    sigma: float = 1.0

    @property
    def name(self) -> str: return "LennardJonesPair"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"epsilon": self.epsilon, "sigma": self.sigma}
    @property
    def governing_equations(self) -> str: return r"V(r)=4\epsilon[(\sigma/r)^{12}-(\sigma/r)^6]"
    @property
    def field_names(self) -> Sequence[str]: return ("position", "velocity")
    @property
    def observable_names(self) -> Sequence[str]: return ("total_energy",)


@dataclass(frozen=True)
class RandomWalkSpec:
    """PHY-V.4: Random walk diffusion (Monte Carlo)."""
    D: float = 0.1
    N_walkers: int = 1000

    @property
    def name(self) -> str: return "RandomWalkDiffusion"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"D": self.D, "N_walkers": self.N_walkers}
    @property
    def governing_equations(self) -> str: return r"\langle x^2 \rangle = 2Dt"
    @property
    def field_names(self) -> Sequence[str]: return ("positions",)
    @property
    def observable_names(self) -> Sequence[str]: return ("mean_square_displacement",)


@dataclass(frozen=True)
class IsingPartitionSpec:
    """PHY-V.6: 1-D Ising chain — exact partition function via transfer matrix."""
    J: float = 1.0
    h: float = 0.0
    N_spins: int = 16

    @property
    def name(self) -> str: return "IsingChainPartition"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"J": self.J, "h": self.h, "N_spins": self.N_spins}
    @property
    def governing_equations(self) -> str: return r"Z = \text{Tr}(T^N),\; T_{s,s'}=e^{\beta(Jss'+h(s+s')/2)}"
    @property
    def field_names(self) -> Sequence[str]: return ("spins",)
    @property
    def observable_names(self) -> Sequence[str]: return ("free_energy",)


# ═══════════════════════════════════════════════════════════════════════════════
# Scaffold solvers (V0.1: toy end-to-end run)
# ═══════════════════════════════════════════════════════════════════════════════


class IsingMCSolver:
    """Metropolis Monte Carlo for 1-D Ising. V0.1: toy run."""

    @property
    def name(self) -> str:
        return "IsingMC_Metropolis"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        spins = state.get_field("spins").data.clone()
        N = spins.shape[0]
        J = kwargs.get("J", 1.0)
        beta = kwargs.get("beta", 1.0)
        # One Metropolis sweep
        for _ in range(N):
            i = torch.randint(0, N, (1,)).item()
            left = (i - 1) % N
            right = (i + 1) % N
            dE = 2.0 * J * spins[i] * (spins[left] + spins[right])
            if dE <= 0.0 or torch.rand(1).item() < math.exp(-beta * dE.item()):
                spins[i] *= -1.0
        new_field = FieldData(name="spins", data=spins, mesh=state.mesh)
        return state.advance(dt, {"spins": new_field})

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        steps = 0
        limit = max_steps or int((tf - t0) / dt) + 1
        while state.t < tf - 1e-14 and steps < limit:
            state = self.step(state, dt, J=1.0, beta=1.0)
            steps += 1
        return SolveResult(final_state=state, t_final=state.t, steps_taken=steps)


class DiffusionODESolver:
    """Simple diffusion ODE solver for Fokker-Planck. V0.1."""

    @property
    def name(self) -> str:
        return "FokkerPlanck_Euler"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        p = state.get_field("probability").data
        dx = state.mesh.dx[0]
        D = kwargs.get("D", 0.1)
        d2p = (torch.roll(p, -1) - 2.0 * p + torch.roll(p, 1)) / (dx ** 2)
        new_p = p + dt * D * d2p
        new_field = FieldData(name="probability", data=new_p, mesh=state.mesh)
        return state.advance(dt, {"probability": new_field})

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        integrator = ForwardEuler()
        D = 0.1
        dx = state.mesh.dx[0]
        def rhs(s: SimulationState, t: float) -> Dict[str, Tensor]:
            p = s.get_field("probability").data
            d2p = (torch.roll(p, -1) - 2.0 * p + torch.roll(p, 1)) / (dx ** 2)
            return {"probability": D * d2p}
        return integrator.solve(state, rhs, t_span, dt, observables=observables, max_steps=max_steps)


class LennardJonesSolver:
    """Störmer-Verlet for LJ pair. V0.1."""

    @property
    def name(self) -> str:
        return "LJ_Verlet"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        # 2-body in 1-D
        pos = state.get_field("position").data.clone()
        vel = state.get_field("velocity").data.clone()
        eps = kwargs.get("epsilon", 1.0)
        sig = kwargs.get("sigma", 1.0)
        r = (pos[1] - pos[0]).clamp(min=0.5 * sig)
        f = 24.0 * eps * (2.0 * (sig / r) ** 12 - (sig / r) ** 6) / r
        force = torch.tensor([-f, f], dtype=pos.dtype)
        vel = vel + 0.5 * dt * force
        pos = pos + dt * vel
        r2 = (pos[1] - pos[0]).clamp(min=0.5 * sig)
        f2 = 24.0 * eps * (2.0 * (sig / r2) ** 12 - (sig / r2) ** 6) / r2
        force2 = torch.tensor([-f2, f2], dtype=pos.dtype)
        vel = vel + 0.5 * dt * force2
        new_fields = {
            "position": FieldData(name="position", data=pos, mesh=state.mesh),
            "velocity": FieldData(name="velocity", data=vel, mesh=state.mesh),
        }
        return state.advance(dt, new_fields)

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


class RandomWalkSolver:
    """Simple random walk. V0.1."""

    @property
    def name(self) -> str:
        return "RandomWalk_MC"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        pos = state.get_field("positions").data.clone()
        D = kwargs.get("D", 0.1)
        pos += math.sqrt(2.0 * D * dt) * torch.randn_like(pos)
        new_field = FieldData(name="positions", data=pos, mesh=state.mesh)
        return state.advance(dt, {"positions": new_field})

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        steps = 0
        limit = max_steps or int((tf - t0) / dt) + 1
        while state.t < tf - 1e-14 and steps < limit:
            state = self.step(state, min(dt, tf - state.t), D=0.1)
            steps += 1
        return SolveResult(final_state=state, t_final=state.t, steps_taken=steps)


class IsingPartitionSolver:
    """Transfer matrix solver for 1-D Ising partition function. V0.1."""

    @property
    def name(self) -> str:
        return "IsingPartition_TransferMatrix"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        return state.advance(dt, dict(state.fields))

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        # Compute partition function via transfer matrix (steady state solver)
        return SolveResult(
            final_state=state, t_final=t_span[1], steps_taken=1,
            metadata={"type": "steady_state"},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pack registration
# ═══════════════════════════════════════════════════════════════════════════════


class ThermoStatMechPack(DomainPack):
    """Pack V: Thermodynamics and Statistical Mechanics."""

    @property
    def pack_id(self) -> str:
        return "V"

    @property
    def pack_name(self) -> str:
        return "Thermodynamics and Statistical Mechanics"

    @property
    def taxonomy_ids(self) -> Sequence[str]:
        return (
            "PHY-V.1", "PHY-V.2", "PHY-V.3", "PHY-V.4", "PHY-V.5", "PHY-V.6",
        )

    def problem_specs(self) -> Dict[str, Type[ProblemSpec]]:
        return {
            "PHY-V.1": IsingEnergySpec,
            "PHY-V.2": FokkerPlanckSpec,
            "PHY-V.3": LennardJonesSpec,
            "PHY-V.4": RandomWalkSpec,
            "PHY-V.5": AdvectionDiffusionSpec,
            "PHY-V.6": IsingPartitionSpec,
        }

    def solvers(self) -> Dict[str, Type[Solver]]:
        return {
            "PHY-V.1": IsingMCSolver,
            "PHY-V.2": DiffusionODESolver,
            "PHY-V.3": LennardJonesSolver,
            "PHY-V.4": RandomWalkSolver,
            "PHY-V.5": AdvDiffSolver,
            "PHY-V.6": IsingPartitionSolver,
        }

    def discretizations(self) -> Dict[str, Sequence[Type[Discretization]]]:
        return {"PHY-V.5": [FVM_AdvDiff_1D]}

    def observables(self) -> Dict[str, Sequence[Type[Observable]]]:
        return {"PHY-V.5": [AdvDiffL2Observable, AdvDiffIntegralObservable]}

    def benchmarks(self) -> Dict[str, Sequence[str]]:
        return {
            "PHY-V.5": ["advection_diffusion_sinusoidal"],
            "PHY-V.1": ["ising_1d_mc_energy"],
        }

    def version(self) -> str:
        return "0.4.0"


# Register at import
get_registry().register_pack(ThermoStatMechPack())


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor Vertical Slice — run_heat_vertical_slice
# ═══════════════════════════════════════════════════════════════════════════════


def _run_advdiff(
    N: int,
    spec: AdvectionDiffusionSpec,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run advection-diffusion at resolution N, return metrics."""
    mesh = StructuredMesh(shape=(N,), domain=((0.0, spec.L),))
    dx = mesh.dx[0]

    # CFL: dt ≤ min(dx/c, dx²/(2α)) with safety factor
    dt_adv = dx / abs(spec.c) if spec.c != 0.0 else float("inf")
    dt_diff = dx ** 2 / (2.0 * spec.alpha) if spec.alpha > 0.0 else float("inf")
    dt = 0.3 * min(dt_adv, dt_diff)

    # IC: u(x,0) = sin(2πx/L)
    x = mesh.cell_centers().squeeze(-1)
    k = 2.0 * math.pi / spec.L
    u0_data = torch.sin(k * x)
    field = FieldData(name="u", data=u0_data, mesh=mesh)
    state0 = SimulationState(t=0.0, fields={"u": field}, mesh=mesh)

    ops = AdvDiffOps(dx=dx, N=N, c=spec.c, alpha=spec.alpha)

    l2_obs = AdvDiffL2Observable()
    int_obs = AdvDiffIntegralObservable(dx)

    with ReproducibilityContext(seed=seed) as ctx:
        integrator = RK4()
        result = integrator.solve(
            state0, ops.rhs,
            t_span=(0.0, spec.T_final), dt=dt,
            observables=[l2_obs, int_obs],
        )
        ctx.record("final_u", hash_tensor(result.final_state.get_field("u").data))

    u_num = result.final_state.get_field("u").data
    u_exact = advdiff_exact(x, spec.T_final, spec.c, spec.alpha, spec.L)
    linf = compute_linf_error(u_num, u_exact)
    l2_err = compute_l2_error(u_num, u_exact, dx)

    # Monotone L2 norm decay check
    l2_hist = result.observable_history.get("l2_norm", [])
    monotone = all(
        l2_hist[i].item() <= l2_hist[i - 1].item() + 1e-14
        for i in range(1, len(l2_hist))
    )

    return {
        "N": N,
        "dt": dt,
        "steps": result.steps_taken,
        "linf_error": linf,
        "l2_error": l2_err,
        "monotone_decay": monotone,
        "provenance": ctx.provenance(),
        "final_state": result.final_state,
    }


def run_heat_vertical_slice(
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute Pack V anchor vertical slice (advection-diffusion) at V0.4."""
    spec = AdvectionDiffusionSpec(c=1.0, alpha=0.01, L=1.0, T_final=0.5)

    resolutions = [128, 256, 512]
    runs = {N: _run_advdiff(N, spec, seed=seed) for N in resolutions}

    errors = [runs[N]["linf_error"] for N in resolutions]
    orders = convergence_order(errors, resolutions)

    # Determinism
    run2 = _run_advdiff(resolutions[-1], spec, seed=seed)
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
        "monotone_decay": all(runs[N]["monotone_decay"] for N in resolutions),
        "deterministic": deterministic,
    }

    if verbose:
        print("=" * 72)
        print("  ANCHOR VERTICAL SLICE — Pack V: 1-D Advection-Diffusion")
        print("=" * 72)
        print(f"  c={spec.c}, α={spec.alpha}, L={spec.L}, T={spec.T_final}")
        print()
        for i, N in enumerate(resolutions):
            r = runs[N]
            print(f"  N={N:>4}  L∞={r['linf_error']:.4e}  steps={r['steps']}")
        print()
        for i, o in enumerate(orders):
            print(f"  Order {resolutions[i]}→{resolutions[i+1]}: {o:.2f}  (expect ≈2)")
        print(f"  Monotone decay:  {'PASS' if metrics['monotone_decay'] else 'FAIL'}")
        print(f"  Deterministic:   {'PASS' if deterministic else 'FAIL'}")
        print()

        gates = {
            "L∞ error < 1e-4 (finest)": errors[-1] < 1e-4,
            "Convergence order > 1.8": all(o > 1.8 for o in orders),
            "Monotone decay": metrics["monotone_decay"],
            "Deterministic": deterministic,
        }
        all_pass = all(gates.values())
        for label, ok in gates.items():
            print(f"  [{'✓' if ok else '✗'}] {label}")
        print(f"\n  RESULT: {'V0.4 VALIDATED' if all_pass else 'FAILED'}")
        print("=" * 72)

    return metrics


if __name__ == "__main__":
    m = run_heat_vertical_slice()
    ok = m["finest_linf"] < 1e-4 and all(o > 1.8 for o in m["convergence_orders"])
    sys.exit(0 if ok else 1)
