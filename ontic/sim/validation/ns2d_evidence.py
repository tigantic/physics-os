"""
NS2D QTT MG-DC Evidence — V&V integration.

Registers six evidence panels as formal VVTest cases in the V&V harness:

  1. Tolerance sensitivity  — proves noise-floor and production sweet spot
  2. Reproducibility        — proves seed-invariant QoIs under rSVD randomness
  3. Divergence constraint  — proves ∇·u ≈ 0 under QTT truncation
  4. Taylor-Green benchmark — proves enstrophy accuracy vs exact solution
  5. Long-horizon TG 512²  — stability + analytic accuracy over 1049 steps
  6. Dense FFT cross-check  — QTT vs exact discrete reference at 256²

Panels 1–4 are quick evidence; panels 5–6 are scheduled scenarios
(weekly / nightly).  See ``docs/reports/NS2D_SCHEDULED_VV_SCENARIOS.md``.

Each panel is a callable returning a dict of metrics with acceptance criteria.
Use ``build_ns2d_evidence_plan()`` to get a runnable ``VVPlan``.

Data is also persisted to ``scenario_output/data/evidence_package.json`` by
the standalone script ``scripts/run_evidence_package.py``.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Any

from ..validation.vv import VVCategory, VVLevel, VVPlan, VVTest

# Ensure project root is importable
_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Constants (mirror scripts/run_evidence_package.py)
# ---------------------------------------------------------------------------
N_BITS = 9
GRID = 2 ** N_BITS
VISCOSITY = 0.01
MAX_RANK = 64
TRUNC_TOL = 1e-10
PRODUCTION_TOL = 1e-3


# ---------------------------------------------------------------------------
# Thin wrappers that import the real evidence-panel functions lazily
# ---------------------------------------------------------------------------

def _lazy_evidence():
    """Import evidence-package helpers on first call (avoids import-time GPU init)."""
    # The canonical implementation lives in scripts/run_evidence_package.py.
    # We import it by path to keep the single source of truth.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_evidence_package",
        os.path.join(_ROOT, "scripts", "run_evidence_package.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Panel executors (return dict expected by VVTest.executor protocol)
# ---------------------------------------------------------------------------

def _exec_tolerance_sensitivity() -> dict[str, Any]:
    """VVTest executor for Panel 1."""
    mod = _lazy_evidence()
    data = mod.panel_tolerance_sensitivity()
    prod = data["runs"]["tol_1e-03"]
    return {
        "enstrophy_error_rel": prod["enstrophy_error_rel"],
        "poisson_max_residual": prod["poisson_max_residual"],
        "all_steps_converged": 1.0 if prod["all_steps_converged"] else 0.0,
        "wall_time_s": prod["wall_time_s"],
        "noise_floor_1e4_residual": data["runs"]["tol_1e-04"]["poisson_max_residual"],
        "_raw": data,
    }


def _exec_reproducibility() -> dict[str, Any]:
    """VVTest executor for Panel 2."""
    mod = _lazy_evidence()
    data = mod.panel_reproducibility()
    s = data["summary"]
    return {
        "enstrophy_cv_pct": s["enstrophy"]["cv_pct"],
        "poisson_residual_cv_pct": s["poisson_max_residual"]["cv_pct"],
        "wall_time_cv_pct": s["wall_time_s"]["cv_pct"],
        "n_seeds": data["n_seeds"],
        "_raw": data,
    }


def _exec_divergence() -> dict[str, Any]:
    """VVTest executor for Panel 3."""
    mod = _lazy_evidence()
    fresh = mod._run(n_steps=10, tol=PRODUCTION_TOL, max_iters=80, seed=42)
    data = mod.panel_divergence(fresh)
    div = data.get("divergence", {})
    return {
        "div_relative_to_vel": div.get("div_relative_to_vel", float("nan")),
        "div_l2_norm": div.get("div_l2_norm", float("nan")),
        "poisson_drift_ratio": data.get("poisson_drift", {}).get(
            "ratio_last_over_first", float("nan")
        ),
        "_raw": data,
    }


def _exec_taylor_green() -> dict[str, Any]:
    """VVTest executor for Panel 4."""
    mod = _lazy_evidence()
    data = mod.panel_benchmark_taylor_green()
    snap = data["snapshot_10step"]
    return {
        "enstrophy_error_rel": snap["enstrophy_error_rel"],
        "enstrophy_error_abs": snap["enstrophy_error_abs"],
        "circulation_error": snap["circulation_error"],
        "dt_convergence_order_mean": (
            sum(data["dt_convergence"]["observed_convergence_orders"])
            / max(len(data["dt_convergence"]["observed_convergence_orders"]), 1)
        ),
        "_raw": data,
    }


# ---------------------------------------------------------------------------
# Panel 5 & 6 — Scheduled scenarios (lazy import)
# ---------------------------------------------------------------------------

def _lazy_long_horizon():
    """Import run_tg_long_horizon lazily."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_tg_long_horizon",
        os.path.join(_ROOT, "scripts", "run_tg_long_horizon.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _lazy_fft_crosscheck():
    """Import run_tg_fft_crosscheck lazily."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_tg_fft_crosscheck",
        os.path.join(_ROOT, "scripts", "run_tg_fft_crosscheck.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _exec_long_horizon() -> dict[str, Any]:
    """VVTest executor for Panel 5 — long-horizon TG 512²."""
    mod = _lazy_long_horizon()
    result = mod.run_scenario()
    # Extract the seed-0 final metrics as the primary result
    s0 = result["seeds"]["0"]["final"]
    return {
        "enstrophy_error_rel": s0["enstrophy_error_rel"],
        "omega_l2_error_rel": s0["omega_l2_error_rel"],
        "div_relative_to_vel": s0["div_relative_to_vel"],
        "repro_enstrophy_delta": result["reproducibility"]["delta_enstrophy_rel"],
        "repro_omega_l2_delta": result["reproducibility"]["delta_omega_l2_rel"],
        "wall_time_s": result["total_wall_time_s"],
        "_raw": result,
    }


def _exec_fft_crosscheck() -> dict[str, Any]:
    """VVTest executor for Panel 6 — dense FFT cross-check 256²."""
    mod = _lazy_fft_crosscheck()
    result = mod.run_scenario()
    comp = result["comparison"]
    return {
        "omega_rel_l2": comp.get("omega_rel_l2", float("nan")),
        "enstrophy_rel_diff": comp.get("enstrophy_rel_diff", float("nan")),
        "omega_l2_rel_diff": comp.get("omega_l2_rel_diff", float("nan")),
        "qtt_wall_time_s": result["qtt"]["wall_time_s"],
        "dense_wall_time_s": result["dense"]["wall_time_s"],
        "_raw": result,
    }


# ---------------------------------------------------------------------------
# Public API: Build the V&V plan
# ---------------------------------------------------------------------------

def build_ns2d_evidence_plan() -> VVPlan:
    """Construct a VVPlan containing all six NS2D evidence panels.

    Returns a fully configured ``VVPlan`` ready for ``plan.run()`` or
    ``run_vv_plan(plan)``.

    Panels 1–4: Quick evidence (short-horizon, ~minutes each)
    Panel 5: Long-horizon TG 512² (weekly, ~60-90 min)
    Panel 6: Dense FFT cross-check 256² (nightly, ~5-10 min)

    Acceptance criteria encode the proven production bounds:
      - Enstrophy relative error < 1e-6  (observed ~9e-8)
      - Poisson residual < 1e-3          (production gate)
      - Divergence ||∇·u||/||u|| < 1e-3  (observed ~8e-5)
      - Reproducibility CV < 1%          (observed 0%)
      - Long-horizon enstrophy error < 1e-4 (spec threshold)
      - QTT vs dense field rel_L2 < 5e-3 (spec threshold)
    """
    plan = VVPlan(
        name="NS2D QTT MG-DC Evidence Package",
        version="2.0.0",
        description=(
            "Six-panel evidence package for the QTT Navier-Stokes 2D solver "
            "with MG-DC (defect correction + geometric multigrid V-cycle) "
            "Poisson preconditioner. "
            "Panels 1–4: tolerance sensitivity, reproducibility, divergence "
            "constraint health, and Taylor-Green benchmark accuracy (512²). "
            "Panel 5: long-horizon (1049 steps) stability + analytic accuracy (512²). "
            "Panel 6: dense FFT cross-check against exact discrete reference (256²)."
        ),
    )

    # Panel 1: Tolerance sensitivity
    plan.add_test(VVTest(
        name="ns2d_tolerance_sensitivity",
        category=VVCategory.SOLUTION_VERIFICATION,
        level=VVLevel.RIGOROUS,
        description=(
            "Poisson residual at tol=1e-3/1e-4/1e-5. Proves QTT noise floor "
            "~3e-4 and that enstrophy error is invariant across tolerances."
        ),
        executor=_exec_tolerance_sensitivity,
        acceptance_criteria={
            "enstrophy_error_rel": 1e-6,
            "poisson_max_residual": 1e-3,
        },
        priority=1,
    ))

    # Panel 2: Reproducibility
    plan.add_test(VVTest(
        name="ns2d_reproducibility",
        category=VVCategory.SOLUTION_VERIFICATION,
        level=VVLevel.RIGOROUS,
        description=(
            "QoI variance across 5 rSVD seeds. Proves physics QoIs are "
            "deterministic despite stochastic rank truncation."
        ),
        executor=_exec_reproducibility,
        acceptance_criteria={
            "enstrophy_cv_pct": 1.0,  # Must be < 1% CV
            "poisson_residual_cv_pct": 5.0,  # Residual noise accepted < 5%
        },
        priority=1,
    ))

    # Panel 3: Divergence constraint
    plan.add_test(VVTest(
        name="ns2d_divergence_constraint",
        category=VVCategory.VALIDATION,
        level=VVLevel.RIGOROUS,
        description=(
            "Velocity divergence ||∇·u||/||u|| in ω-ψ formulation. "
            "Measures QTT truncation impact on incompressibility constraint."
        ),
        executor=_exec_divergence,
        acceptance_criteria={
            "div_relative_to_vel": 1e-3,  # Must be < 0.1%
        },
        priority=1,
    ))

    # Panel 4: Taylor-Green benchmark
    plan.add_test(VVTest(
        name="ns2d_taylor_green_benchmark",
        category=VVCategory.VALIDATION,
        level=VVLevel.CERTIFICATION,
        description=(
            "Enstrophy error vs exact Taylor-Green solution. "
            "Tier 1 benchmark (Taylor & Green, 1937)."
        ),
        executor=_exec_taylor_green,
        acceptance_criteria={
            "enstrophy_error_rel": 1e-6,
            "circulation_error": 1e-10,
        },
        priority=1,
    ))

    # Panel 5: Long-horizon Taylor-Green @ 512² (weekly / stress)
    plan.add_test(VVTest(
        name="ns2d_tg_long_horizon_512",
        category=VVCategory.VALIDATION,
        level=VVLevel.CERTIFICATION,
        description=(
            "Long-horizon Taylor-Green vortex at 512² (1049 steps, t_final=0.05). "
            "Proves analytic accuracy (enstrophy error ≤ 1e-4), constraint health "
            "(div ≤ 1e-3), Poisson stability, and seed reproducibility (≤ 1e-6). "
            "Run ID: TG_LH_512_QTT_PROD. Schedule: weekly / release gate."
        ),
        executor=_exec_long_horizon,
        acceptance_criteria={
            "enstrophy_error_rel": 1e-4,
            "omega_l2_error_rel": 1e-4,
            "div_relative_to_vel": 1e-3,
            "repro_enstrophy_delta": 1e-6,
            "repro_omega_l2_delta": 1e-6,
        },
        priority=2,
    ))

    # Panel 6: Dense FFT cross-check @ 256² (nightly)
    plan.add_test(VVTest(
        name="ns2d_tg_fft_crosscheck_256",
        category=VVCategory.VALIDATION,
        level=VVLevel.CERTIFICATION,
        description=(
            "QTT vs dense FFT reference for the same discrete PDE at 256². "
            "Dense solver uses FFT diagonalization of the 5-point periodic "
            "Laplacian (exact discrete inverse). "
            "Proves field-level agreement (rel_L2 ≤ 5e-3) and QoI match "
            "(enstrophy diff ≤ 1e-4). "
            "Run ID: TG_XCHECK_256_QTT_vs_DENSEFDFFT. Schedule: nightly."
        ),
        executor=_exec_fft_crosscheck,
        acceptance_criteria={
            "omega_rel_l2": 5e-3,
            "enstrophy_rel_diff": 1e-4,
            "omega_l2_rel_diff": 1e-4,
        },
        priority=2,
    ))

    return plan
