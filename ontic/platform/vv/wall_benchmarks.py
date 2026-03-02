"""QTT Physics VM — Wall-Model V&V Benchmarks (Phase E).

Provides channel-flow and cavity-flow validation benchmarks with
convergence trend tracking for the wall model (Lane A: penalization +
calibrated wall function).

These benchmarks run as internal V&V tooling.  Diagnostics are
sanitizer-safe: only whitelisted scalar aggregates (integrated shear,
penalization energy) are exposed.  TT cores, bond dimensions, and
field-level wall data never leak.

Benchmarks
----------
1. **Channel flow** (Poiseuille):
   Laminar channel flow at Re = 100.  Parabolic velocity profile
   validated against exact solution.  Convergence trend monitored
   over grid refinement.

2. **Lid-driven cavity** (Re = 100, 400, 1000):
   Classical benchmark for wall-bounded recirculating flows.
   Primary QoI: center-line u-velocity at mid-height (compared
   to Ghia et al. 1982 tabulated data).

3. **Cylinder in channel** (Re = 20):
   Steady flow around cylinder.  QoI: drag coefficient C_d
   compared to Schäfer & Turek benchmark.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray


# ══════════════════════════════════════════════════════════════════════
# Benchmark specifications
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WallBenchmarkSpec:
    """Specification for a wall-model V&V benchmark.

    Parameters
    ----------
    benchmark_id : str
        Unique identifier (e.g., "W010_channel_poiseuille").
    name : str
        Human-readable benchmark name.
    category : str
        Benchmark category: "wall_verification" or "wall_validation".
    domain_key : str
        Domain pack key for the VM compiler.
    reynolds_number : float
        Reynolds number.
    geometry_type : str
        Geometry type for the geometry compiler.
    qoi : dict[str, Any]
        Quantities of interest and their reference values / gates.
    refinement_levels : list[int]
        Grid resolutions (n_bits) for convergence study.
    n_steps_per_level : list[int]
        Time steps for each refinement level.
    wall_model_params : dict[str, Any]
        Wall model configuration overrides.
    """
    benchmark_id: str
    name: str
    category: str
    domain_key: str
    reynolds_number: float
    geometry_type: str
    qoi: dict[str, Any]
    refinement_levels: list[int] = field(default_factory=lambda: [5, 6, 7])
    n_steps_per_level: list[int] = field(default_factory=lambda: [100, 200, 400])
    wall_model_params: dict[str, Any] = field(default_factory=dict)


# ── Ghia et al. (1982) reference data for lid-driven cavity ─────────

# u-velocity along vertical center-line at Re=100
# (y, u) pairs from Table I, Ghia et al. JCP 48, 387-411
GHIA_RE_100_CENTERLINE_U: list[tuple[float, float]] = [
    (1.0000, 1.00000),
    (0.9766, 0.84123),
    (0.9688, 0.78871),
    (0.9609, 0.73722),
    (0.9531, 0.68717),
    (0.8516, 0.23151),
    (0.7344, 0.00332),
    (0.6172, -0.13641),
    (0.5000, -0.20581),
    (0.4531, -0.21090),
    (0.2813, -0.15662),
    (0.1719, -0.10150),
    (0.1016, -0.06434),
    (0.0703, -0.04775),
    (0.0625, -0.04192),
    (0.0547, -0.03717),
    (0.0000, 0.00000),
]

# u-velocity along vertical center-line at Re=400
GHIA_RE_400_CENTERLINE_U: list[tuple[float, float]] = [
    (1.0000, 1.00000),
    (0.9766, 0.75837),
    (0.9688, 0.68439),
    (0.9609, 0.61756),
    (0.9531, 0.55892),
    (0.8516, 0.29093),
    (0.7344, 0.16256),
    (0.6172, 0.02135),
    (0.5000, -0.11477),
    (0.4531, -0.17119),
    (0.2813, -0.32726),
    (0.1719, -0.24299),
    (0.1016, -0.14612),
    (0.0703, -0.10338),
    (0.0625, -0.09266),
    (0.0547, -0.08186),
    (0.0000, 0.00000),
]

# u-velocity along vertical center-line at Re=1000
GHIA_RE_1000_CENTERLINE_U: list[tuple[float, float]] = [
    (1.0000, 1.00000),
    (0.9766, 0.65928),
    (0.9688, 0.57492),
    (0.9609, 0.51117),
    (0.9531, 0.46604),
    (0.8516, 0.33304),
    (0.7344, 0.18719),
    (0.6172, 0.05702),
    (0.5000, -0.06080),
    (0.4531, -0.10648),
    (0.2813, -0.27805),
    (0.1719, -0.38289),
    (0.1016, -0.29730),
    (0.0703, -0.22220),
    (0.0625, -0.20196),
    (0.0547, -0.18109),
    (0.0000, 0.00000),
]

GHIA_TABLES: dict[int, list[tuple[float, float]]] = {
    100: GHIA_RE_100_CENTERLINE_U,
    400: GHIA_RE_400_CENTERLINE_U,
    1000: GHIA_RE_1000_CENTERLINE_U,
}


# ── Schäfer & Turek (1996) reference for cylinder drag ───────────────

# DFG 2D-1 steady benchmark: C_d reference at Re=20
SCHAFER_TUREK_CD_RE20: float = 5.5795
SCHAFER_TUREK_CD_RE20_TOL: float = 0.15  # 15% tolerance for immersed boundary


# ══════════════════════════════════════════════════════════════════════
# Benchmark registry
# ══════════════════════════════════════════════════════════════════════

def _poiseuille_exact(y: NDArray, h_channel: float, u_max: float) -> NDArray:
    """Exact Poiseuille profile u(y) = u_max * 4y(h-y)/h²."""
    return u_max * 4.0 * y * (h_channel - y) / (h_channel ** 2)


def build_wall_benchmark_registry() -> dict[str, WallBenchmarkSpec]:
    """Build the wall-model V&V benchmark registry.

    Returns
    -------
    dict[str, WallBenchmarkSpec]
        Benchmark ID → specification mapping.
    """
    registry: dict[str, WallBenchmarkSpec] = {}

    # ── W010: Channel Poiseuille (laminar, exact solution) ───────────
    registry["W010_channel_poiseuille"] = WallBenchmarkSpec(
        benchmark_id="W010_channel_poiseuille",
        name="Laminar Channel Flow (Poiseuille)",
        category="wall_verification",
        domain_key="navier_stokes_2d",
        reynolds_number=100.0,
        geometry_type="channel",
        qoi={
            "velocity_profile_l2_error": {
                "gate_type": "absolute_max",
                "threshold": 0.05,
                "description": "L² error of u-velocity vs. Poiseuille exact.",
            },
            "convergence_order": {
                "gate_type": "observed_order_min",
                "threshold": 1.5,
                "description": "Spatial convergence rate of profile error.",
            },
        },
        refinement_levels=[5, 6, 7],
        n_steps_per_level=[200, 400, 800],
        wall_model_params={
            "eta_permeability": 1e-4,
            "viscosity": 0.01,
        },
    )

    # ── W020: Lid-driven cavity Re=100 ───────────────────────────────
    registry["W020_cavity_re100"] = WallBenchmarkSpec(
        benchmark_id="W020_cavity_re100",
        name="Lid-Driven Cavity (Re=100)",
        category="wall_validation",
        domain_key="navier_stokes_2d",
        reynolds_number=100.0,
        geometry_type="cavity",
        qoi={
            "centerline_u_l2_error": {
                "gate_type": "absolute_max",
                "threshold": 0.10,
                "description": (
                    "L² error of center-line u vs. Ghia et al. (1982)."
                ),
            },
            "convergence_trend": {
                "gate_type": "observed_order_min",
                "threshold": 0.5,
                "description": "Error reduction with refinement.",
            },
        },
        refinement_levels=[5, 6, 7],
        n_steps_per_level=[500, 1000, 2000],
        wall_model_params={
            "eta_permeability": 1e-4,
            "viscosity": 0.01,
        },
    )

    # ── W030: Lid-driven cavity Re=400 ───────────────────────────────
    registry["W030_cavity_re400"] = WallBenchmarkSpec(
        benchmark_id="W030_cavity_re400",
        name="Lid-Driven Cavity (Re=400)",
        category="wall_validation",
        domain_key="navier_stokes_2d",
        reynolds_number=400.0,
        geometry_type="cavity",
        qoi={
            "centerline_u_l2_error": {
                "gate_type": "absolute_max",
                "threshold": 0.15,
                "description": (
                    "L² error of center-line u vs. Ghia et al. (1982)."
                ),
            },
        },
        refinement_levels=[6, 7, 8],
        n_steps_per_level=[2000, 4000, 8000],
        wall_model_params={
            "eta_permeability": 1e-5,
            "viscosity": 0.0025,
        },
    )

    # ── W040: Lid-driven cavity Re=1000 ──────────────────────────────
    registry["W040_cavity_re1000"] = WallBenchmarkSpec(
        benchmark_id="W040_cavity_re1000",
        name="Lid-Driven Cavity (Re=1000)",
        category="wall_validation",
        domain_key="navier_stokes_2d",
        reynolds_number=1000.0,
        geometry_type="cavity",
        qoi={
            "centerline_u_l2_error": {
                "gate_type": "absolute_max",
                "threshold": 0.20,
                "description": (
                    "L² error of center-line u vs. Ghia et al. (1982)."
                ),
            },
        },
        refinement_levels=[6, 7, 8],
        n_steps_per_level=[5000, 10000, 20000],
        wall_model_params={
            "eta_permeability": 1e-6,
            "viscosity": 0.001,
        },
    )

    # ── W050: Cylinder in channel (DFG 2D-1, Re=20) ─────────────────
    registry["W050_cylinder_re20"] = WallBenchmarkSpec(
        benchmark_id="W050_cylinder_re20",
        name="Cylinder in Channel (DFG 2D-1, Re=20)",
        category="wall_validation",
        domain_key="navier_stokes_2d",
        reynolds_number=20.0,
        geometry_type="cylinder_in_channel",
        qoi={
            "drag_coefficient_error": {
                "gate_type": "absolute_max",
                "threshold": 0.50,
                "description": (
                    "Relative error of C_d vs. Schäfer & Turek (1996)."
                ),
            },
        },
        refinement_levels=[6, 7, 8],
        n_steps_per_level=[1000, 2000, 4000],
        wall_model_params={
            "eta_permeability": 1e-5,
            "viscosity": 0.001,
        },
    )

    return registry


# ══════════════════════════════════════════════════════════════════════
# QoI extraction utilities
# ══════════════════════════════════════════════════════════════════════

def extract_centerline_u(
    omega_field: NDArray,
    psi_field: NDArray,
    n_bits: int,
    domain: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 1.0), (0.0, 1.0),
    ),
) -> tuple[NDArray, NDArray]:
    """Extract u-velocity along the vertical center-line (x = 0.5).

    Computes u = ∂ψ/∂y via finite differences on the psi field,
    then samples along x = N//2.

    Parameters
    ----------
    omega_field : NDArray
        Vorticity field (N×N).
    psi_field : NDArray
        Stream function field (N×N).
    n_bits : int
        Resolution bits (N = 2^n_bits).
    domain : tuple
        Physical domain bounds.

    Returns
    -------
    (y_coords, u_centerline) : tuple[NDArray, NDArray]
        y-coordinates and u-velocity along the center-line.
    """
    N = 2 ** n_bits
    hy = (domain[1][1] - domain[1][0]) / N

    # u = ∂ψ/∂y (centered differences)
    u = np.zeros_like(psi_field)
    u[:, 1:-1] = (psi_field[:, 2:] - psi_field[:, :-2]) / (2.0 * hy)
    # Boundary: forward/backward
    u[:, 0] = (psi_field[:, 1] - psi_field[:, 0]) / hy
    u[:, -1] = (psi_field[:, -1] - psi_field[:, -2]) / hy

    # Sample along vertical center-line x = 0.5
    mid_x = N // 2
    u_centerline = u[mid_x, :]
    y_coords = np.linspace(
        domain[1][0], domain[1][1], N, endpoint=False,
    ) + 0.5 * hy

    return y_coords, u_centerline


def compute_ghia_l2_error(
    y_coords: NDArray,
    u_centerline: NDArray,
    reynolds_number: int,
) -> float:
    """Compute L² error against Ghia et al. (1982) reference data.

    Interpolates the simulation data onto the Ghia y-locations and
    computes the discrete L² norm of the difference.

    Parameters
    ----------
    y_coords : NDArray
        y-coordinates from the simulation.
    u_centerline : NDArray
        u-velocity along the center-line.
    reynolds_number : int
        Reynolds number (100, 400, or 1000).

    Returns
    -------
    float
        L² error norm.
    """
    if reynolds_number not in GHIA_TABLES:
        raise ValueError(
            f"No Ghia reference data for Re={reynolds_number}. "
            f"Available: {list(GHIA_TABLES.keys())}"
        )

    ghia_data = GHIA_TABLES[reynolds_number]
    errors: list[float] = []

    for y_ref, u_ref in ghia_data:
        # Interpolate simulation u at y_ref
        u_sim = float(np.interp(y_ref, y_coords, u_centerline))
        errors.append((u_sim - u_ref) ** 2)

    return math.sqrt(sum(errors) / len(errors))


def compute_poiseuille_l2_error(
    y_coords: NDArray,
    u_centerline: NDArray,
    h_channel: float,
    u_max: float,
) -> float:
    """Compute L² error against exact Poiseuille profile.

    Parameters
    ----------
    y_coords : NDArray
        y-coordinates from the simulation.
    u_centerline : NDArray
        u-velocity along the center-line.
    h_channel : float
        Channel height.
    u_max : float
        Maximum velocity (centerline).

    Returns
    -------
    float
        Normalized L² error.
    """
    u_exact = _poiseuille_exact(y_coords, h_channel, u_max)
    err = np.linalg.norm(u_centerline - u_exact) / max(
        np.linalg.norm(u_exact), 1e-15,
    )
    return float(err)


def compute_convergence_order(
    errors: list[float],
    grid_spacings: list[float],
) -> float:
    """Estimate convergence order from a sequence of errors.

    Uses least-squares fit to log(error) = p * log(h) + const.

    Parameters
    ----------
    errors : list[float]
        Error values at each refinement level.
    grid_spacings : list[float]
        Grid spacings at each level.

    Returns
    -------
    float
        Estimated convergence order p.
    """
    if len(errors) < 2:
        return 0.0

    # Filter out zero or negative errors
    valid = [
        (h, e) for h, e in zip(grid_spacings, errors)
        if e > 1e-15 and h > 0
    ]
    if len(valid) < 2:
        return 0.0

    log_h = np.log([v[0] for v in valid])
    log_e = np.log([v[1] for v in valid])

    # Least-squares: log_e = p * log_h + b
    A = np.vstack([log_h, np.ones_like(log_h)]).T
    result = np.linalg.lstsq(A, log_e, rcond=None)
    p = result[0][0]

    return float(max(p, 0.0))


# ══════════════════════════════════════════════════════════════════════
# Gate evaluation
# ══════════════════════════════════════════════════════════════════════

@dataclass
class WallGateResult:
    """Result of evaluating a wall-benchmark gate.

    Attributes
    ----------
    gate_name : str
        Name of the QoI gate.
    passed : bool
        True if the gate is satisfied.
    value : float
        Observed value.
    threshold : float
        Gate threshold.
    description : str
        Human-readable description.
    """
    gate_name: str
    passed: bool
    value: float
    threshold: float
    description: str


def evaluate_wall_gates(
    benchmark_spec: WallBenchmarkSpec,
    qoi_values: dict[str, float],
) -> list[WallGateResult]:
    """Evaluate all gates for a wall benchmark.

    Parameters
    ----------
    benchmark_spec : WallBenchmarkSpec
        Benchmark specification with gate definitions.
    qoi_values : dict[str, float]
        Observed QoI values.

    Returns
    -------
    list[WallGateResult]
        Gate results for each QoI.
    """
    results: list[WallGateResult] = []

    for qoi_name, qoi_spec in benchmark_spec.qoi.items():
        gate_type = qoi_spec["gate_type"]
        threshold = float(qoi_spec["threshold"])
        desc = qoi_spec.get("description", "")

        observed = qoi_values.get(qoi_name, float("inf"))

        if gate_type == "absolute_max":
            passed = observed <= threshold
        elif gate_type == "observed_order_min":
            passed = observed >= threshold
        elif gate_type == "relative_max":
            passed = observed <= threshold
        else:
            passed = False

        results.append(WallGateResult(
            gate_name=qoi_name,
            passed=passed,
            value=observed,
            threshold=threshold,
            description=desc,
        ))

    return results


# ══════════════════════════════════════════════════════════════════════
# Diagnostic sanitizer
# ══════════════════════════════════════════════════════════════════════

_WALL_DIAGNOSTIC_WHITELIST = frozenset({
    # Scalar aggregates only — no field-level data
    "integrated_wall_shear",
    "max_wall_shear_proxy",
    "integrated_heat_flux",
    "penalization_energy",
    # Benchmark QoI scalars
    "velocity_profile_l2_error",
    "centerline_u_l2_error",
    "drag_coefficient_error",
    "convergence_order",
    "convergence_trend",
})


def sanitize_wall_diagnostics(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Filter wall-model diagnostics to whitelisted aggregates.

    Enforces §20.4 IP boundary: no wall-model internals (distance
    proxies, reciprocal distance fields, stress profiles, TT cores)
    leak through the sanitizer.

    Parameters
    ----------
    diagnostics : dict[str, Any]
        Raw diagnostics dictionary.

    Returns
    -------
    dict[str, Any]
        Only whitelisted scalar aggregates.
    """
    return {
        k: v for k, v in diagnostics.items()
        if k in _WALL_DIAGNOSTIC_WHITELIST
    }
