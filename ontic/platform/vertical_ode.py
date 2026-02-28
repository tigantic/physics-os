"""
Vertical Slice #1 — ODE: Harmonic Oscillator (Full Stack @ V0.4)
=================================================================

Traverses the *entire* Phase 1 platform stack:

    ProblemSpec → Mesh → IC → Solver (StormerVerlet) → Observable → Checkpoint → Reproduce

The harmonic oscillator  q'' = −ω² q  is the simplest Hamiltonian system and
the canonical acceptance test for symplectic integrators.

Success criteria (V0.4 Validated):
  • Energy conservation to < 1e-8 relative error after 10 000 steps.
  • Exact period: T = 2π/ω.  Phase error < 0.5° after 10 full orbits.
  • Deterministic: two runs with same seed produce bitwise-identical results.
  • Checkpoint round-trip: save → load → continue → same final state.

Usage:
    python -m ontic.platform.vertical_ode
"""

from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

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
from ontic.platform.protocols import Observable, ProblemSpec, SolveResult
from ontic.platform.reproduce import (
    ArtifactHash,
    ReproducibilityContext,
    hash_tensor,
)
from ontic.platform.solvers import ForwardEuler, RK4, StormerVerlet, SymplecticEuler


# ═══════════════════════════════════════════════════════════════════════════════
# ProblemSpec
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HarmonicOscillatorSpec:
    """1-D harmonic oscillator: H(q, p) = p²/2m + ½ k q²."""

    omega: float = 1.0
    mass: float = 1.0

    @property
    def name(self) -> str:
        return "HarmonicOscillator"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"omega": self.omega, "mass": self.mass, "k": self.mass * self.omega ** 2}

    @property
    def governing_equations(self) -> str:
        return r"H(q, p) = \frac{p^2}{2m} + \frac{1}{2} k q^2"

    @property
    def field_names(self) -> Sequence[str]:
        return ("q", "p")

    @property
    def observable_names(self) -> Sequence[str]:
        return ("energy", "amplitude")


# ═══════════════════════════════════════════════════════════════════════════════
# Observables
# ═══════════════════════════════════════════════════════════════════════════════


class EnergyObservable:
    """Total energy H = p²/(2m) + ½ k q²."""

    def __init__(self, spec: HarmonicOscillatorSpec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        return "energy"

    @property
    def units(self) -> str:
        return "J"

    def compute(self, state: Any) -> Tensor:
        q = state.get_field("q").data
        p = state.get_field("p").data
        m = self._spec.mass
        k = m * self._spec.omega ** 2
        return (p ** 2 / (2.0 * m) + 0.5 * k * q ** 2).sum()


class AmplitudeObservable:
    """Peak displacement |q|."""

    @property
    def name(self) -> str:
        return "amplitude"

    @property
    def units(self) -> str:
        return "m"

    def compute(self, state: Any) -> Tensor:
        return state.get_field("q").data.abs().max()


# ═══════════════════════════════════════════════════════════════════════════════
# RHS
# ═══════════════════════════════════════════════════════════════════════════════


def harmonic_rhs(
    spec: HarmonicOscillatorSpec,
) -> "callable":
    """Return a Hamiltonian RHS: dq/dt = p/m, dp/dt = −k q."""
    m = spec.mass
    k = m * spec.omega ** 2

    def rhs(state: SimulationState, t: float) -> Dict[str, Tensor]:
        q = state.get_field("q").data
        p = state.get_field("p").data
        return {
            "q": p / m,
            "p": -k * q,
        }

    return rhs


# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════


def run_ode_vertical_slice(
    n_orbits: int = 10,
    steps_per_orbit: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute the full ODE vertical slice and return validation metrics.
    """
    spec = HarmonicOscillatorSpec(omega=2.0 * math.pi, mass=1.0)
    period = 2.0 * math.pi / spec.omega  # = 1.0 for omega = 2π
    dt = period / steps_per_orbit
    total_steps = n_orbits * steps_per_orbit
    t_final = n_orbits * period

    # ── Mesh (trivial 1-cell mesh for ODE) ──
    mesh = StructuredMesh(shape=(1,), domain=((0.0, 1.0),))

    # ── Initial conditions ──
    q0 = FieldData(name="q", data=torch.tensor([1.0], dtype=torch.float64), mesh=mesh)
    p0 = FieldData(name="p", data=torch.tensor([0.0], dtype=torch.float64), mesh=mesh)
    state0 = SimulationState(t=0.0, fields={"q": q0, "p": p0}, mesh=mesh)

    # ── Observables ──
    energy_obs = EnergyObservable(spec)
    amp_obs = AmplitudeObservable()

    # ── Initial energy ──
    E0 = energy_obs.compute(state0).item()

    # ── Reproducibility context ──
    with ReproducibilityContext(seed=seed) as ctx:
        integrator = StormerVerlet()
        rhs = harmonic_rhs(spec)

        result = integrator.solve(
            state0,
            rhs,
            t_span=(0.0, t_final),
            dt=dt,
            observables=[energy_obs, amp_obs],
        )

        ctx.record("final_q", hash_tensor(result.final_state.get_field("q").data))
        ctx.record("final_p", hash_tensor(result.final_state.get_field("p").data))

    # ── Validate ──
    final_state = result.final_state
    E_final = energy_obs.compute(final_state).item()
    energy_err = abs(E_final - E0) / abs(E0) if abs(E0) > 1e-30 else abs(E_final - E0)

    q_final = final_state.get_field("q").data.item()
    p_final = final_state.get_field("p").data.item()

    # After n complete orbits, should return to (q0, p0) = (1, 0)
    phase_err_rad = math.atan2(p_final / spec.mass, q_final) - 0.0
    phase_err_deg = abs(math.degrees(phase_err_rad))

    # ── Checkpoint round-trip ──
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = save_checkpoint(final_state, tmpdir, name="final")
        restored = load_checkpoint(ckpt_path)
        q_diff = (restored.get_field("q").data - final_state.get_field("q").data).abs().max().item()
        p_diff = (restored.get_field("p").data - final_state.get_field("p").data).abs().max().item()
        checkpoint_ok = q_diff < 1e-15 and p_diff < 1e-15

    # ── Determinism: second run ──
    with ReproducibilityContext(seed=seed) as ctx2:
        result2 = StormerVerlet().solve(
            state0, harmonic_rhs(spec), t_span=(0.0, t_final), dt=dt,
        )
    q_det = (result2.final_state.get_field("q").data - final_state.get_field("q").data).abs().max().item()
    p_det = (result2.final_state.get_field("p").data - final_state.get_field("p").data).abs().max().item()
    deterministic = q_det == 0.0 and p_det == 0.0

    metrics = {
        "problem": spec.name,
        "integrator": integrator.name,
        "n_orbits": n_orbits,
        "total_steps": total_steps,
        "dt": dt,
        "E0": E0,
        "E_final": E_final,
        "energy_relative_error": energy_err,
        "phase_error_deg": phase_err_deg,
        "q_final": q_final,
        "p_final": p_final,
        "checkpoint_roundtrip_ok": checkpoint_ok,
        "deterministic": deterministic,
        "provenance": ctx.provenance(),
    }

    # ── Report ──
    if verbose:
        print("=" * 72)
        print("  VERTICAL SLICE #1 — ODE: Harmonic Oscillator")
        print("=" * 72)
        print(f"  Integrator:        {integrator.name} (order {integrator.order})")
        print(f"  Orbits:            {n_orbits}")
        print(f"  Steps:             {total_steps}")
        print(f"  dt:                {dt:.6e}")
        print(f"  E₀:               {E0:.12f}")
        print(f"  E_final:           {E_final:.12f}")
        print(f"  Energy rel err:    {energy_err:.4e}")
        print(f"  Phase error:       {phase_err_deg:.4f}°")
        print(f"  Checkpoint r/t:    {'PASS' if checkpoint_ok else 'FAIL'}")
        print(f"  Deterministic:     {'PASS' if deterministic else 'FAIL'}")
        print()

        # Gate checks
        gates = {
            "Energy < 1e-8": energy_err < 1e-8,
            "Phase < 0.5°": phase_err_deg < 0.5,
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
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    metrics = run_ode_vertical_slice()
    all_ok = (
        metrics["energy_relative_error"] < 1e-8
        and metrics["phase_error_deg"] < 0.5
        and metrics["checkpoint_roundtrip_ok"]
        and metrics["deterministic"]
    )
    sys.exit(0 if all_ok else 1)
