"""
Vertical Slice #2 — PDE: 1-D Heat Equation (Full Stack @ V0.4)
================================================================

Traverses the *entire* Phase 1 platform stack for a PDE:

    ProblemSpec → StructuredMesh → Discretization (FVM) → IC/BC
    → Solver (RK4 / ForwardEuler) → Observable → Checkpoint → Reproduce

The equation:

    ∂u/∂t = α ∂²u/∂x²     x ∈ [0, L],  t ∈ [0, T]

With initial condition  u(x, 0) = sin(π x / L)  and  u(0,t) = u(L,t) = 0
(Dirichlet).

Exact solution:  u(x, t) = sin(π x / L) exp(−α (π/L)² t)

Success criteria (V0.4 Validated):
  • L∞ error vs exact < 1e-4 at t = T.
  • Conservation: ∫ u dx decays monotonically (no spurious growth).
  • Grid refinement: error ratio between N and 2N ≈ 4 (second-order).
  • Deterministic across two runs.
  • Checkpoint round-trip.

Usage:
    python3 -m ontic.platform.vertical_pde
"""

from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ontic.platform.checkpoint import load_checkpoint, save_checkpoint
from ontic.platform.data_model import (
    BCType,
    BoundaryCondition,
    FieldData,
    InitialCondition,
    Mesh,
    SimulationState,
    StructuredMesh,
)
from ontic.platform.protocols import Discretization, Observable, ProblemSpec, SolveResult
from ontic.platform.reproduce import (
    ReproducibilityContext,
    hash_tensor,
)
from ontic.platform.solvers import ForwardEuler, RK4, RHSCallable


# ═══════════════════════════════════════════════════════════════════════════════
# ProblemSpec
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HeatEquationSpec:
    """1-D heat equation: ∂u/∂t = α ∂²u/∂x²."""

    alpha: float = 0.01
    L: float = 1.0
    T_final: float = 1.0

    @property
    def name(self) -> str:
        return "HeatEquation1D"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "L": self.L, "T_final": self.T_final}

    @property
    def governing_equations(self) -> str:
        return r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}"

    @property
    def field_names(self) -> Sequence[str]:
        return ("u",)

    @property
    def observable_names(self) -> Sequence[str]:
        return ("l2_norm", "max_u", "integral_u")


# ═══════════════════════════════════════════════════════════════════════════════
# Discretization — 2nd-order FVM (central differences)
# ═══════════════════════════════════════════════════════════════════════════════


class FVM_1D_Heat:
    """
    1-D finite-volume discretization of the Laplacian for the heat equation.

    Ghost cells enforce Dirichlet BCs.
    """

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha

    @property
    def method(self) -> str:
        return "FVM"

    @property
    def order(self) -> int:
        return 2

    def discretize(
        self,
        spec: ProblemSpec,
        mesh: Any,
    ) -> "FVM_1D_HeatOps":
        if not isinstance(mesh, StructuredMesh) or mesh.ndim != 1:
            raise TypeError("FVM_1D_Heat requires a 1-D StructuredMesh")
        return FVM_1D_HeatOps(
            dx=mesh.dx[0],
            N=mesh.shape[0],
            alpha=self._alpha,
        )


@dataclass
class FVM_1D_HeatOps:
    """Discrete operators returned by FVM_1D_Heat.discretize."""

    dx: float
    N: int
    alpha: float

    def laplacian(self, u: Tensor) -> Tensor:
        """
        Second-order central Laplacian with Dirichlet BCs (u=0 at boundaries).

        u is shape (N,) on a cell-centred grid.  Cell centres sit at
        x_i = (i + 0.5) * dx.  The Dirichlet value at x=0 and x=L lies
        on the *face*, so the ghost cell value is the antisymmetric
        reflection:  u_ghost = −u_0  (ensures linear interpolation to
        face = 0, giving 2nd-order accuracy).
        """
        d2u = torch.zeros_like(u)
        dx2 = self.dx ** 2

        # Interior
        d2u[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx2

        # Boundaries — ghost = −u_boundary  (antisymmetric reflection)
        d2u[0] = (u[1] - 2.0 * u[0] + (-u[0])) / dx2    # = (u[1] - 3*u[0]) / dx²
        d2u[-1] = ((-u[-1]) - 2.0 * u[-1] + u[-2]) / dx2  # = (u[-2] - 3*u[-1]) / dx²

        return d2u

    def rhs(self, state: SimulationState, t: float) -> Dict[str, Tensor]:
        """du/dt = α ∇²u."""
        u = state.get_field("u").data
        return {"u": self.alpha * self.laplacian(u)}


# ═══════════════════════════════════════════════════════════════════════════════
# Observables
# ═══════════════════════════════════════════════════════════════════════════════


class L2NormObservable:
    @property
    def name(self) -> str:
        return "l2_norm"

    @property
    def units(self) -> str:
        return "1"

    def compute(self, state: Any) -> Tensor:
        u = state.get_field("u").data
        return torch.norm(u, p=2)


class MaxUObservable:
    @property
    def name(self) -> str:
        return "max_u"

    @property
    def units(self) -> str:
        return "K"

    def compute(self, state: Any) -> Tensor:
        return state.get_field("u").data.max()


class IntegralUObservable:
    """∫ u dx (scalar)."""

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


# ═══════════════════════════════════════════════════════════════════════════════
# Exact solution
# ═══════════════════════════════════════════════════════════════════════════════


def exact_solution(x: Tensor, t: float, alpha: float, L: float) -> Tensor:
    """Exact: u(x,t) = sin(π x / L) exp(−α (π/L)² t)."""
    return torch.sin(math.pi * x / L) * math.exp(-alpha * (math.pi / L) ** 2 * t)


# ═══════════════════════════════════════════════════════════════════════════════
# Single-resolution run
# ═══════════════════════════════════════════════════════════════════════════════


def _run_one(
    N: int,
    spec: HeatEquationSpec,
    integrator_cls: type,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run at resolution *N* and return metrics."""
    mesh = StructuredMesh(shape=(N,), domain=((0.0, spec.L),))
    dx = mesh.dx[0]

    # CFL: dt ≤ dx² / (2α)
    dt = 0.4 * dx ** 2 / spec.alpha  # safety factor 0.4

    # IC: u(x,0) = sin(πx/L)
    x = mesh.cell_centers().squeeze(-1)
    u0_data = torch.sin(math.pi * x / spec.L)
    u0 = FieldData(name="u", data=u0_data, mesh=mesh)
    state0 = SimulationState(t=0.0, fields={"u": u0}, mesh=mesh)

    # Discretize
    disc = FVM_1D_Heat(alpha=spec.alpha)
    ops = disc.discretize(spec, mesh)

    # Observables
    l2_obs = L2NormObservable()
    max_obs = MaxUObservable()
    int_obs = IntegralUObservable(dx)

    with ReproducibilityContext(seed=seed) as ctx:
        integrator = integrator_cls()
        result = integrator.solve(
            state0,
            ops.rhs,
            t_span=(0.0, spec.T_final),
            dt=dt,
            observables=[l2_obs, max_obs, int_obs],
        )
        ctx.record("final_u", hash_tensor(result.final_state.get_field("u").data))

    # Error vs exact
    u_num = result.final_state.get_field("u").data
    u_exact = exact_solution(x, spec.T_final, spec.alpha, spec.L)
    linf = (u_num - u_exact).abs().max().item()
    l2_err = torch.sqrt(((u_num - u_exact) ** 2).sum() * dx).item()

    # Monotone decay check
    int_hist = result.observable_history.get("integral_u", [])
    monotone = True
    for i in range(1, len(int_hist)):
        if int_hist[i].item() > int_hist[i - 1].item() + 1e-15:
            monotone = False
            break

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


# ═══════════════════════════════════════════════════════════════════════════════
# Full vertical slice
# ═══════════════════════════════════════════════════════════════════════════════


def run_pde_vertical_slice(
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the full PDE vertical slice and return validation metrics."""
    spec = HeatEquationSpec(alpha=0.01, L=1.0, T_final=0.5)

    # ── Grid refinement study ──
    resolutions = [64, 128, 256]
    runs = {}
    for N in resolutions:
        runs[N] = _run_one(N, spec, RK4, seed=seed)

    # Error ratio (expect ~4 for 2nd-order spatial)
    e1 = runs[resolutions[0]]["linf_error"]
    e2 = runs[resolutions[1]]["linf_error"]
    e3 = runs[resolutions[2]]["linf_error"]
    ratio_12 = e1 / e2 if e2 > 1e-30 else float("inf")
    ratio_23 = e2 / e3 if e3 > 1e-30 else float("inf")

    # ── Checkpoint round-trip (finest grid) ──
    finest = runs[resolutions[-1]]
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = save_checkpoint(finest["final_state"], tmpdir, name="final")
        restored = load_checkpoint(ckpt_path)
        u_diff = (
            restored.get_field("u").data - finest["final_state"].get_field("u").data
        ).abs().max().item()
        checkpoint_ok = u_diff < 1e-15

    # ── Determinism ──
    run2 = _run_one(resolutions[-1], spec, RK4, seed=seed)
    det_diff = (
        run2["final_state"].get_field("u").data
        - finest["final_state"].get_field("u").data
    ).abs().max().item()
    deterministic = det_diff == 0.0

    metrics = {
        "problem": spec.name,
        "integrator": "RK4",
        "resolutions": resolutions,
        "linf_errors": {N: runs[N]["linf_error"] for N in resolutions},
        "refinement_ratios": {"r12": ratio_12, "r23": ratio_23},
        "finest_linf": e3,
        "monotone_decay": all(runs[N]["monotone_decay"] for N in resolutions),
        "checkpoint_roundtrip_ok": checkpoint_ok,
        "deterministic": deterministic,
    }

    if verbose:
        print("=" * 72)
        print("  VERTICAL SLICE #2 — PDE: 1-D Heat Equation")
        print("=" * 72)
        print(f"  α = {spec.alpha},  L = {spec.L},  T = {spec.T_final}")
        print()
        print("  Grid Refinement Study (RK4 + FVM 2nd-order):")
        print(f"  {'N':>6}  {'L∞ error':>12}  {'steps':>8}  {'dt':>12}")
        for N in resolutions:
            r = runs[N]
            print(f"  {N:>6}  {r['linf_error']:>12.4e}  {r['steps']:>8}  {r['dt']:>12.4e}")
        print()
        print(f"  Error ratio N1→N2:  {ratio_12:.2f}  (expect ≈4)")
        print(f"  Error ratio N2→N3:  {ratio_23:.2f}  (expect ≈4)")
        print(f"  Monotone decay:     {'PASS' if metrics['monotone_decay'] else 'FAIL'}")
        print(f"  Checkpoint r/t:     {'PASS' if checkpoint_ok else 'FAIL'}")
        print(f"  Deterministic:      {'PASS' if deterministic else 'FAIL'}")
        print()

        gates = {
            "L∞ error < 1e-4 (finest)": e3 < 1e-4,
            "Refinement ratio > 3.5": ratio_12 > 3.5 and ratio_23 > 3.5,
            "Monotone decay": metrics["monotone_decay"],
            "Checkpoint OK": checkpoint_ok,
            "Deterministic": deterministic,
        }
        all_pass = all(gates.values())
        for label, ok in gates.items():
            mark = "✓" if ok else "✗"
            print(f"  [{mark}] {label}")
        print()
        print(f"  RESULT: {'V0.4 VALIDATED' if all_pass else 'FAILED'}")
        print("=" * 72)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    m = run_pde_vertical_slice()
    ok = (
        m["finest_linf"] < 1e-4
        and m["refinement_ratios"]["r12"] > 3.5
        and m["refinement_ratios"]["r23"] > 3.5
        and m["monotone_decay"]
        and m["checkpoint_roundtrip_ok"]
        and m["deterministic"]
    )
    sys.exit(0 if ok else 1)
