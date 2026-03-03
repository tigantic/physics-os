"""
NS2D QTT MG-DC Evidence — V&V integration.

Registers four evidence panels as formal VVTest cases in the V&V harness:

  1. Tolerance sensitivity  — proves noise-floor and production sweet spot
  2. Reproducibility        — proves seed-invariant QoIs under rSVD randomness
  3. Divergence constraint  — proves ∇·u ≈ 0 under QTT truncation
  4. Taylor-Green benchmark — proves enstrophy accuracy vs exact solution

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
# Public API: Build the V&V plan
# ---------------------------------------------------------------------------

def build_ns2d_evidence_plan() -> VVPlan:
    """Construct a VVPlan containing all four NS2D evidence panels.

    Returns a fully configured ``VVPlan`` ready for ``plan.run()`` or
    ``run_vv_plan(plan)``.

    Acceptance criteria encode the proven production bounds:
      - Enstrophy relative error < 1e-6  (observed ~9e-8)
      - Poisson residual < 1e-3          (production gate)
      - Divergence ||∇·u||/||u|| < 1e-3  (observed ~8e-5)
      - Reproducibility CV < 1%          (observed 0%)
    """
    plan = VVPlan(
        name="NS2D QTT MG-DC Evidence Package",
        version="1.0.0",
        description=(
            "Four-panel evidence package for the QTT Navier-Stokes 2D solver "
            "with MG-DC (defect correction + geometric multigrid V-cycle) "
            "Poisson preconditioner at 512² (n_bits=9). "
            "Proves tolerance sensitivity, reproducibility, divergence "
            "constraint health, and Taylor-Green benchmark accuracy."
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

    return plan
