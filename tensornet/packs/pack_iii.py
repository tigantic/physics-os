"""
Domain Pack III — Electromagnetism
====================================

**Anchor problem (V0.4)**:  1-D FDTD Maxwell equations (PHY-III.3)

    ε ∂E/∂t = ∂H/∂x
    μ ∂H/∂t = ∂E/∂x

Yee scheme: E at integer grid points, H at half-integer points.
Leapfrog in time: E at integer time steps, H at half-integer time steps.

Validation: Gaussian pulse propagation in vacuum (ε=ε₀, μ=μ₀).
At CFL = 1 (Δt = Δx/c), the Yee scheme is *exact* for 1-D propagation.
At CFL < 1, the scheme is second-order in space and time.

Validation gates (V0.4):
  • L∞ error < 1e-4 at t = T (finest grid).
  • Convergence order ≈ 2 at CFL < 1.
  • Energy conservation: ∫(εE² + μH²) dx ≈ const.
  • Deterministic across two runs.

Scaffold nodes (V0.1): PHY-III.1 through PHY-III.7
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor

from tensornet.packs._base import (
    BaseProblemSpec,
    compute_linf_error,
    convergence_order,
)
from tensornet.platform.data_model import (
    FieldData,
    Mesh,
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


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-III.3  Full Maxwell time-domain — 1-D FDTD (ANCHOR)
# ═══════════════════════════════════════════════════════════════════════════════

# Normalized units: c = 1, ε = 1, μ = 1


@dataclass(frozen=True)
class Maxwell1DSpec:
    """1-D Maxwell: ε ∂E/∂t = ∂H/∂x, μ ∂H/∂t = ∂E/∂x (normalized)."""

    epsilon: float = 1.0
    mu: float = 1.0
    L: float = 10.0
    T_final: float = 4.0
    sigma_pulse: float = 0.3
    x0_pulse: float = 5.0

    @property
    def name(self) -> str:
        return "Maxwell1D_FDTD"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon, "mu": self.mu,
            "L": self.L, "T_final": self.T_final,
            "sigma": self.sigma_pulse, "x0": self.x0_pulse,
        }

    @property
    def governing_equations(self) -> str:
        return r"\varepsilon\partial_t E = \partial_x H;\; \mu\partial_t H = \partial_x E"

    @property
    def field_names(self) -> Sequence[str]:
        return ("E", "H")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("em_energy",)


class FDTD_1D:
    """
    1-D Yee FDTD discretization.

    Grid layout:
      E[i] at x = i*dx,     i = 0, …, N-1
      H[i] at x = (i+0.5)*dx, i = 0, …, N-2

    Time layout:
      E at t^n = n*dt
      H at t^{n+1/2} = (n+0.5)*dt

    PEC boundaries: E[0] = E[N-1] = 0 (reflective).
    """

    def __init__(self, epsilon: float = 1.0, mu: float = 1.0) -> None:
        self._epsilon = epsilon
        self._mu = mu

    @property
    def method(self) -> str:
        return "FDTD"

    @property
    def order(self) -> int:
        return 2

    def discretize(self, spec: ProblemSpec, mesh: Any) -> "FDTD_1D_Ops":
        if not isinstance(mesh, StructuredMesh) or mesh.ndim != 1:
            raise TypeError("FDTD_1D requires a 1-D StructuredMesh")
        return FDTD_1D_Ops(
            dx=mesh.dx[0],
            N_E=mesh.shape[0],
            epsilon=self._epsilon,
            mu=self._mu,
        )


@dataclass
class FDTD_1D_Ops:
    """Discrete operators for the 1-D Yee scheme."""

    dx: float
    N_E: int
    epsilon: float
    mu: float

    @property
    def N_H(self) -> int:
        return self.N_E - 1

    @property
    def c(self) -> float:
        return 1.0 / math.sqrt(self.epsilon * self.mu)

    @property
    def max_dt(self) -> float:
        """CFL limit: dt ≤ dx/c."""
        return self.dx / self.c

    def update_H(self, E: Tensor, H: Tensor, dt: float) -> Tensor:
        """H^{n+1/2} = H^{n-1/2} + (dt / μ dx) * (E[i+1] - E[i])."""
        return H + (dt / (self.mu * self.dx)) * (E[1:] - E[:-1])

    def update_E(self, E: Tensor, H: Tensor, dt: float) -> Tensor:
        """E^{n+1} = E^n + (dt / ε dx) * (H[i] - H[i-1]),  PEC at boundaries."""
        new_E = E.clone()
        new_E[1:-1] = E[1:-1] + (dt / (self.epsilon * self.dx)) * (H[1:] - H[:-1])
        # PEC: new_E[0] = 0, new_E[-1] = 0 (already zero for initial Gaussian)
        new_E[0] = 0.0
        new_E[-1] = 0.0
        return new_E


class EMEnergyObservable:
    """Total EM energy: 0.5 * ∫(εE² + μH²) dx."""

    def __init__(self, dx: float, epsilon: float, mu: float) -> None:
        self._dx = dx
        self._epsilon = epsilon
        self._mu = mu

    @property
    def name(self) -> str:
        return "em_energy"

    @property
    def units(self) -> str:
        return "J/m²"

    def compute(self, state: Any) -> Tensor:
        E = state.get_field("E").data
        H = state.get_field("H").data
        energy = 0.5 * (self._epsilon * (E ** 2).sum() + self._mu * (H ** 2).sum()) * self._dx
        return energy


class MaxwellSolver:
    """
    FDTD leapfrog solver for 1-D Maxwell.

    This is a *custom* time integrator (not using the generic TimeIntegrator)
    because the Yee scheme requires staggered E/H updates, which is a
    fundamentally different integration pattern from method-of-lines RK.
    """

    def __init__(self, epsilon: float = 1.0, mu: float = 1.0) -> None:
        self._epsilon = epsilon
        self._mu = mu

    @property
    def name(self) -> str:
        return "MaxwellFDTD_Leapfrog"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        E = state.get_field("E").data
        H = state.get_field("H").data
        mesh = state.mesh
        dx = mesh.dx[0]
        ops = FDTD_1D_Ops(dx=dx, N_E=E.shape[0], epsilon=self._epsilon, mu=self._mu)
        new_H = ops.update_H(E, H, dt)
        new_E = ops.update_E(E, new_H, dt)
        # Build mesh for H (one fewer cell)
        h_mesh = StructuredMesh(shape=(new_H.shape[0],), domain=mesh.domain)
        new_fields = {
            "E": FieldData(name="E", data=new_E, mesh=mesh),
            "H": FieldData(name="H", data=new_H, mesh=h_mesh),
        }
        return SimulationState(
            t=state.t + dt,
            fields=new_fields,
            mesh=mesh,
            step_index=state.step_index + 1,
        )

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        t0, tf = t_span
        obs_history: Dict[str, List[Tensor]] = {}
        if observables:
            for obs in observables:
                obs_history[obs.name] = []

        steps = 0
        limit = max_steps or int(1e9)

        while state.t < tf - 1e-14 * abs(dt) and steps < limit:
            actual_dt = min(dt, tf - state.t)
            state = self.step(state, actual_dt)
            steps += 1
            if observables:
                for obs in observables:
                    obs_history[obs.name].append(obs.compute(state))

        return SolveResult(
            final_state=state, t_final=state.t, steps_taken=steps,
            observable_history=obs_history,
            metadata={"integrator": self.name, "dt": dt},
        )


def gaussian_pulse(x: Tensor, x0: float, sigma: float) -> Tensor:
    """Gaussian pulse: exp(-(x-x0)²/(2σ²))."""
    return torch.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))


def maxwell_exact_1d(
    x_E: Tensor, x_H: Tensor, t: float,
    x0: float, sigma: float, c: float,
) -> Tuple[Tensor, Tensor]:
    """
    Exact solution for Gaussian pulse in infinite 1-D domain.

    Initial: E(x,0) = exp(-(x-x0)²/(2σ²)), H(x,0) = 0.

    Splits into two counter-propagating pulses:
      E(x,t) = 0.5 * [g(x - ct) + g(x + ct)]
      H(x,t) = 0.5 / η * [g(x - ct) - g(x + ct)]
    where g(ξ) = exp(-(ξ-x0)²/(2σ²)), η = √(μ/ε) = 1.
    """
    g_right_E = gaussian_pulse(x_E, x0 + c * t, sigma)
    g_left_E = gaussian_pulse(x_E, x0 - c * t, sigma)
    E_exact = 0.5 * (g_right_E + g_left_E)

    g_right_H = gaussian_pulse(x_H, x0 + c * t, sigma)
    g_left_H = gaussian_pulse(x_H, x0 - c * t, sigma)
    # H = (1/(2μc)) [φ(x+ct) − φ(x−ct)] where φ(ξ)=G(ξ, x0)
    # φ(x+ct) = G(x, x0−ct) = g_left,  φ(x−ct) = G(x, x0+ct) = g_right
    H_exact = 0.5 * (g_left_H - g_right_H)

    return E_exact, H_exact


# ═══════════════════════════════════════════════════════════════════════════════
# PHY-III.1 through III.7  Scaffold ProblemSpecs (V0.1)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ElectrostaticsSpec:
    """PHY-III.1: 1-D Poisson: ∇²φ = -ρ/ε."""
    @property
    def name(self) -> str: return "Electrostatics1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"epsilon": 1.0}
    @property
    def governing_equations(self) -> str: return r"\nabla^2 \phi = -\rho/\varepsilon"
    @property
    def field_names(self) -> Sequence[str]: return ("phi",)
    @property
    def observable_names(self) -> Sequence[str]: return ("electric_field",)


@dataclass(frozen=True)
class MagnetostaticsSpec:
    """PHY-III.2: 1-D vector potential: ∇²A = -μJ."""
    @property
    def name(self) -> str: return "Magnetostatics1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"mu": 1.0}
    @property
    def governing_equations(self) -> str: return r"\nabla^2 A = -\mu J"
    @property
    def field_names(self) -> Sequence[str]: return ("A",)
    @property
    def observable_names(self) -> Sequence[str]: return ("B_field",)


@dataclass(frozen=True)
class FreqDomainEMSpec:
    """PHY-III.4: 1-D Helmholtz: ∇²E + k²E = 0."""
    k: float = 6.283
    @property
    def name(self) -> str: return "HelmholtzEM1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"k": self.k}
    @property
    def governing_equations(self) -> str: return r"\nabla^2 E + k^2 E = 0"
    @property
    def field_names(self) -> Sequence[str]: return ("E",)
    @property
    def observable_names(self) -> Sequence[str]: return ("scattering_cross_section",)


@dataclass(frozen=True)
class WavePropagationSpec:
    """PHY-III.5: 1-D wave: ∂²E/∂t² = c² ∂²E/∂x²."""
    c: float = 1.0
    @property
    def name(self) -> str: return "WavePropagation1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"c": self.c}
    @property
    def governing_equations(self) -> str: return r"\partial_{tt} E = c^2 \partial_{xx} E"
    @property
    def field_names(self) -> Sequence[str]: return ("E", "dE_dt")
    @property
    def observable_names(self) -> Sequence[str]: return ("wave_energy",)


@dataclass(frozen=True)
class PhotonicsSpec:
    """PHY-III.6: 1-D RCWA (transfer matrix method)."""
    @property
    def name(self) -> str: return "TransferMatrix1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {}
    @property
    def governing_equations(self) -> str: return r"M = \prod_i M_i(n_i, d_i, \lambda)"
    @property
    def field_names(self) -> Sequence[str]: return ("reflectance", "transmittance")
    @property
    def observable_names(self) -> Sequence[str]: return ("R", "T")


@dataclass(frozen=True)
class AntennasSpec:
    """PHY-III.7: 1-D dipole radiation (far field)."""
    @property
    def name(self) -> str: return "DipoleRadiation1D"
    @property
    def ndim(self) -> int: return 1
    @property
    def parameters(self) -> Dict[str, Any]: return {"frequency": 1.0}
    @property
    def governing_equations(self) -> str: return r"E_{far} \propto \frac{e^{-jkr}}{r}"
    @property
    def field_names(self) -> Sequence[str]: return ("E_far",)
    @property
    def observable_names(self) -> Sequence[str]: return ("directivity",)


# Scaffold solvers — all use a simple wave/diffusion equation
class _ScaffoldWaveSolver:
    """Generic 1D wave solver for scaffold nodes."""

    def __init__(self, field_name: str = "E") -> None:
        self._fn = field_name

    @property
    def name(self) -> str:
        return f"Wave_{self._fn}_Scaffold"

    def step(self, state: Any, dt: float, **kwargs: Any) -> Any:
        u = state.get_field(self._fn).data.clone()
        return state.advance(dt, {self._fn: FieldData(name=self._fn, data=u, mesh=state.mesh)})

    def solve(
        self, state: Any, t_span: Tuple[float, float], dt: float, *,
        observables: Optional[Sequence[Any]] = None, callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        return SolveResult(final_state=state, t_final=t_span[1], steps_taken=1)


class ElectrostaticsSolver(_ScaffoldWaveSolver):
    def __init__(self) -> None: super().__init__("phi")
    @property
    def name(self) -> str: return "Electrostatics_Scaffold"

class MagnetostaticsSolver(_ScaffoldWaveSolver):
    def __init__(self) -> None: super().__init__("A")
    @property
    def name(self) -> str: return "Magnetostatics_Scaffold"

class FreqDomainEMSolver(_ScaffoldWaveSolver):
    @property
    def name(self) -> str: return "FreqDomainEM_Scaffold"

class WavePropagationSolver(_ScaffoldWaveSolver):
    @property
    def name(self) -> str: return "WavePropagation_Scaffold"

class PhotonicsSolver(_ScaffoldWaveSolver):
    def __init__(self) -> None: super().__init__("reflectance")
    @property
    def name(self) -> str: return "Photonics_Scaffold"

class AntennasSolver(_ScaffoldWaveSolver):
    def __init__(self) -> None: super().__init__("E_far")
    @property
    def name(self) -> str: return "Antennas_Scaffold"


# ═══════════════════════════════════════════════════════════════════════════════
# Pack registration
# ═══════════════════════════════════════════════════════════════════════════════


class ElectromagnetismPack(DomainPack):
    """Pack III: Electromagnetism."""

    @property
    def pack_id(self) -> str:
        return "III"

    @property
    def pack_name(self) -> str:
        return "Electromagnetism"

    @property
    def taxonomy_ids(self) -> Sequence[str]:
        return tuple(f"PHY-III.{i}" for i in range(1, 8))

    def problem_specs(self) -> Dict[str, Type[ProblemSpec]]:
        return {
            "PHY-III.1": ElectrostaticsSpec,
            "PHY-III.2": MagnetostaticsSpec,
            "PHY-III.3": Maxwell1DSpec,
            "PHY-III.4": FreqDomainEMSpec,
            "PHY-III.5": WavePropagationSpec,
            "PHY-III.6": PhotonicsSpec,
            "PHY-III.7": AntennasSpec,
        }

    def solvers(self) -> Dict[str, Type[Solver]]:
        return {
            "PHY-III.1": ElectrostaticsSolver,
            "PHY-III.2": MagnetostaticsSolver,
            "PHY-III.3": MaxwellSolver,
            "PHY-III.4": FreqDomainEMSolver,
            "PHY-III.5": WavePropagationSolver,
            "PHY-III.6": PhotonicsSolver,
            "PHY-III.7": AntennasSolver,
        }

    def discretizations(self) -> Dict[str, Sequence[Type[Discretization]]]:
        return {"PHY-III.3": [FDTD_1D]}

    def observables(self) -> Dict[str, Sequence[Type[Observable]]]:
        return {"PHY-III.3": [EMEnergyObservable]}

    def benchmarks(self) -> Dict[str, Sequence[str]]:
        return {
            "PHY-III.3": ["gaussian_pulse_propagation"],
            "PHY-III.1": ["capacitor_1d"],
        }

    def version(self) -> str:
        return "0.4.0"


get_registry().register_pack(ElectromagnetismPack())


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor Vertical Slice — run_em_vertical_slice
# ═══════════════════════════════════════════════════════════════════════════════


def _run_maxwell(
    N: int,
    spec: Maxwell1DSpec,
    cfl: float = 0.9,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run 1-D FDTD at resolution N, return metrics."""
    mesh = StructuredMesh(shape=(N,), domain=((0.0, spec.L),))
    dx = mesh.dx[0]
    c = 1.0 / math.sqrt(spec.epsilon * spec.mu)
    dt = cfl * dx / c

    # E grid: x_i = i * dx,  i = 0 .. N-1
    x_E = torch.linspace(0.0, spec.L - dx, N, dtype=torch.float64)
    # H grid: x_{i+0.5} = (i + 0.5) * dx,  i = 0 .. N-2
    x_H = torch.linspace(0.5 * dx, spec.L - 1.5 * dx, N - 1, dtype=torch.float64)

    # IC: E = Gaussian pulse, H at t=-dt/2 (half-step initialization)
    E0 = gaussian_pulse(x_E, spec.x0_pulse, spec.sigma_pulse)
    E0[0] = 0.0  # PEC
    E0[-1] = 0.0

    # Half-step initialization: H(x, -dt/2) ≈ -(dt/2)/μ * ∂E/∂x
    # This is critical for 2nd-order accuracy with the Yee leapfrog scheme.
    dEdx = (E0[1:] - E0[:-1]) / dx
    H0 = -(dt / 2.0) / spec.mu * dEdx

    h_mesh = StructuredMesh(shape=(N - 1,), domain=((0.5 * dx, spec.L - 1.5 * dx),))

    state0 = SimulationState(
        t=0.0,
        fields={
            "E": FieldData(name="E", data=E0, mesh=mesh),
            "H": FieldData(name="H", data=H0, mesh=h_mesh),
        },
        mesh=mesh,
    )

    energy_obs = EMEnergyObservable(dx, spec.epsilon, spec.mu)
    solver = MaxwellSolver(epsilon=spec.epsilon, mu=spec.mu)

    with ReproducibilityContext(seed=seed) as ctx:
        result = solver.solve(
            state0, t_span=(0.0, spec.T_final), dt=dt,
            observables=[energy_obs],
        )
        ctx.record("final_E", hash_tensor(result.final_state.get_field("E").data))

    # Error vs exact (only valid before pulse reaches boundary)
    E_num = result.final_state.get_field("E").data
    H_num = result.final_state.get_field("H").data
    # The Yee H is at t+dt/2 relative to its last E update. Since the
    # solver's final state has the E at t_final and H was updated within
    # the same step to t_final+dt/2 before E was updated to t_final, the
    # H is actually at t_final−dt/2 (updated *before* E in the last step).
    # Compare E at t_final against exact.
    t_compare = result.t_final
    E_exact, _ = maxwell_exact_1d(
        x_E, x_H, t_compare,
        spec.x0_pulse, spec.sigma_pulse, c,
    )
    linf_E = compute_linf_error(E_num, E_exact)

    # Energy conservation: check drift
    e_hist = result.observable_history.get("em_energy", [])
    if e_hist:
        e_initial = energy_obs.compute(state0).item()
        max_drift = max(abs(v.item() - e_initial) / max(e_initial, 1e-30) for v in e_hist)
    else:
        max_drift = 0.0

    return {
        "N": N, "dt": dt, "steps": result.steps_taken,
        "linf_E": linf_E,
        "energy_drift": max_drift,
        "final_state": result.final_state,
        "provenance": ctx.provenance(),
    }


def run_em_vertical_slice(
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute Pack III anchor vertical slice (FDTD Maxwell) at V0.4."""
    # Short propagation so pulse stays away from boundary (PEC would reflect)
    spec = Maxwell1DSpec(
        epsilon=1.0, mu=1.0, L=20.0, T_final=2.0,
        sigma_pulse=0.5, x0_pulse=10.0,
    )
    cfl = 0.8  # sub-CFL for second-order convergence study

    resolutions = [400, 800, 1600]
    runs = {N: _run_maxwell(N, spec, cfl=cfl, seed=seed) for N in resolutions}

    errors = [runs[N]["linf_E"] for N in resolutions]
    orders = convergence_order(errors, resolutions)

    # Determinism
    run2 = _run_maxwell(resolutions[-1], spec, cfl=cfl, seed=seed)
    det_diff = (
        run2["final_state"].get_field("E").data
        - runs[resolutions[-1]]["final_state"].get_field("E").data
    ).abs().max().item()
    deterministic = det_diff == 0.0

    metrics = {
        "problem": spec.name,
        "resolutions": resolutions,
        "linf_errors": errors,
        "convergence_orders": orders,
        "finest_linf_E": errors[-1],
        "energy_drift_finest": runs[resolutions[-1]]["energy_drift"],
        "deterministic": deterministic,
    }

    if verbose:
        print("=" * 72)
        print("  ANCHOR VERTICAL SLICE — Pack III: 1-D FDTD Maxwell")
        print("=" * 72)
        print(f"  ε={spec.epsilon}, μ={spec.mu}, L={spec.L}, T={spec.T_final}, CFL={cfl}")
        print()
        for i, N in enumerate(resolutions):
            r = runs[N]
            print(
                f"  N={N:>4}  L∞(E)={r['linf_E']:.4e}  "
                f"ΔW={r['energy_drift']:.2e}  steps={r['steps']}"
            )
        print()
        for i, o in enumerate(orders):
            print(f"  Order {resolutions[i]}→{resolutions[i+1]}: {o:.2f}  (expect ≈2)")
        print(f"  Energy drift:    {metrics['energy_drift_finest']:.2e}")
        print(f"  Deterministic:   {'PASS' if deterministic else 'FAIL'}")
        print()

        gates = {
            "L∞ error < 1e-4 (E, finest)": errors[-1] < 1e-4,
            "Convergence order > 1.8": all(o > 1.8 for o in orders),
            "Energy drift < 1e-2 (finest)": metrics["energy_drift_finest"] < 1e-2,
            "Deterministic": deterministic,
        }
        all_pass = all(gates.values())
        for label, ok in gates.items():
            print(f"  [{'✓' if ok else '✗'}] {label}")
        print(f"\n  RESULT: {'V0.4 VALIDATED' if all_pass else 'FAILED'}")
        print("=" * 72)

    return metrics


if __name__ == "__main__":
    m = run_em_vertical_slice()
    ok = (
        m["finest_linf_E"] < 1e-4
        and all(o > 1.8 for o in m["convergence_orders"])
        and m["energy_drift_finest"] < 1e-2
        and m["deterministic"]
    )
    sys.exit(0 if ok else 1)
