#!/usr/bin/env python3
"""Calibration ladder: sweep n_bits × n_steps vs. conservation error.

For one or two NS2D scenarios, systematically vary the grid resolution
(n_bits) and time-step count (n_steps) to find the "knee" where
conservation relative error drops below the production threshold of
1 × 10⁻⁴.

Outputs
-------
``scenario_output/calibration_ladder.json``
    Structured sweep data: each entry has (scenario, n_bits, n_steps,
    conservation_rel_err, wall_time_s, status).

``scenario_output/images/calibration_ladder.png``
    Log-scale heatmap / line plot of conservation error vs. resolution.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ── Ensure project root is on sys.path ───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from physics_os.core.executor import ExecutionConfig, execute
from physics_os.core.sanitizer import sanitize_result
from physics_os.templates.compiler import compile_problem
from physics_os.templates.models import (
    BoundarySpec,
    FlowConditions,
    GeometrySpec,
    GeometryType,
    ProblemClass,
    ProblemSpec,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("calibration_ladder")


# ═══════════════════════════════════════════════════════════════════
# Ladder scenarios — one external flow, one easy-mode
# ═══════════════════════════════════════════════════════════════════

LADDER_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "Highway Side-Mirror (Cylinder, Re ≈ 200k)",
        "spec": ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.CIRCLE, params={"radius": 0.05}
            ),
            flow=FlowConditions(velocity=30.0, fluid="air"),
            boundaries=BoundarySpec(),
            quality="quick",
            max_rank=32,
        ),
    },
    {
        "name": "Stokes Creeping Flow (Glycerol, Re ≈ 0.09)",
        "spec": ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.CIRCLE, params={"radius": 0.005}
            ),
            flow=FlowConditions(velocity=0.01, fluid="glycerol"),
            boundaries=BoundarySpec(),
            quality="quick",
            max_rank=32,
        ),
    },
]

# ── Sweep grid ────────────────────────────────────────────────────
N_BITS_RANGE: list[int] = [4, 5, 6, 7, 8]
N_STEPS_RANGE: list[int] = [50, 100, 200, 500, 1000]

# Maximum wall time per single run (seconds).  Skip if exceeded.
MAX_WALL_TIME_PER_RUN: float = 120.0

OUTPUT_DIR = PROJECT_ROOT / "scenario_output"


# ═══════════════════════════════════════════════════════════════════
# Sweep runner
# ═══════════════════════════════════════════════════════════════════


def run_single(
    spec: ProblemSpec,
    n_bits: int,
    n_steps: int,
) -> dict[str, Any]:
    """Execute one (n_bits, n_steps) point and return conservation data."""
    t0 = time.perf_counter()

    try:
        compiled = compile_problem(spec)

        # Scale dt to the new resolution
        base_n = 2 ** compiled.n_bits
        new_n = 2 ** n_bits
        if new_n < base_n:
            scale = base_n / new_n
            dt = compiled.dt * scale
        else:
            dt = compiled.dt

        config = ExecutionConfig(
            domain=compiled.domain,
            n_bits=n_bits,
            n_steps=n_steps,
            dt=dt,
            max_rank=compiled.max_rank,
            truncation_tol=1e-10,
            parameters=compiled.parameters,
        )

        result = execute(config)
        wall = time.perf_counter() - t0

        if not result.success:
            return {
                "n_bits": n_bits,
                "n_steps": n_steps,
                "status": "execution_error",
                "error": str(result.error),
                "wall_time_s": round(wall, 4),
            }

        execution_context = {"n_bits": n_bits, "n_steps": n_steps}
        sanitized = sanitize_result(
            result, compiled.domain,
            execution_context=execution_context,
        )

        conservation = sanitized.get("conservation", {})
        return {
            "n_bits": n_bits,
            "n_steps": n_steps,
            "grid_points_1d": 2 ** n_bits,
            "status": "success",
            "conservation_quantity": conservation.get("quantity", ""),
            "conservation_rel_err": conservation.get("relative_error"),
            "conservation_status": conservation.get("status", ""),
            "conservation_initial": conservation.get("initial_value"),
            "conservation_final": conservation.get("final_value"),
            "resolution_tier": conservation.get("resolution_tier", ""),
            "tier_threshold": conservation.get("tier_threshold"),
            "wall_time_s": round(wall, 4),
        }

    except Exception as exc:
        wall = time.perf_counter() - t0
        return {
            "n_bits": n_bits,
            "n_steps": n_steps,
            "status": "error",
            "error": str(exc),
            "wall_time_s": round(wall, 4),
        }


def run_ladder() -> dict[str, Any]:
    """Run the full calibration ladder sweep."""
    results: dict[str, Any] = {
        "sweep_parameters": {
            "n_bits_range": N_BITS_RANGE,
            "n_steps_range": N_STEPS_RANGE,
            "max_wall_time_per_run": MAX_WALL_TIME_PER_RUN,
        },
        "scenarios": [],
    }

    total_runs = len(LADDER_SCENARIOS) * len(N_BITS_RANGE) * len(N_STEPS_RANGE)
    run_idx = 0

    for scenario in LADDER_SCENARIOS:
        scenario_name = scenario["name"]
        spec = scenario["spec"]
        sweep_data: list[dict[str, Any]] = []

        logger.info("━━━ Scenario: %s ━━━", scenario_name)

        for n_bits in N_BITS_RANGE:
            for n_steps in N_STEPS_RANGE:
                run_idx += 1
                logger.info(
                    "[%d/%d] n_bits=%d  n_steps=%d ...",
                    run_idx, total_runs, n_bits, n_steps,
                )

                entry = run_single(spec, n_bits, n_steps)
                sweep_data.append(entry)

                rel_err = entry.get("conservation_rel_err")
                wall = entry.get("wall_time_s", 0)
                logger.info(
                    "        → rel_err=%s  wall=%.2fs  status=%s",
                    f"{rel_err:.2e}" if rel_err is not None else "N/A",
                    wall,
                    entry.get("status", "?"),
                )

        results["scenarios"].append({
            "name": scenario_name,
            "entries": sweep_data,
        })

    return results


def generate_ladder_plot(results: dict[str, Any], output_dir: Path) -> None:
    """Generate a calibration ladder visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n_scenarios = len(results["scenarios"])
    fig, axes = plt.subplots(
        1, n_scenarios,
        figsize=(7 * n_scenarios, 5),
        squeeze=False,
    )

    for idx, scenario_data in enumerate(results["scenarios"]):
        ax = axes[0, idx]
        name = scenario_data["name"]
        entries = scenario_data["entries"]

        # Group by n_bits
        by_bits: dict[int, tuple[list[int], list[float]]] = {}
        for e in entries:
            if e.get("conservation_rel_err") is None:
                continue
            nb = e["n_bits"]
            if nb not in by_bits:
                by_bits[nb] = ([], [])
            by_bits[nb][0].append(e["n_steps"])
            by_bits[nb][1].append(e["conservation_rel_err"])

        for nb in sorted(by_bits):
            steps, errs = by_bits[nb]
            order = sorted(range(len(steps)), key=lambda i: steps[i])
            steps_sorted = [steps[i] for i in order]
            errs_sorted = [errs[i] for i in order]
            label = f"n_bits={nb} ({2**nb}×{2**nb})"
            ax.semilogy(steps_sorted, errs_sorted, "o-", label=label, markersize=5)

        # Threshold lines
        ax.axhline(y=1e-4, color="green", linestyle="--", linewidth=1.5,
                    alpha=0.7, label="production (1e-4)")
        ax.axhline(y=1e-2, color="orange", linestyle="--", linewidth=1.0,
                    alpha=0.7, label="standard (1e-2)")
        ax.axhline(y=5e-1, color="red", linestyle="--", linewidth=1.0,
                    alpha=0.7, label="preview (5e-1)")

        ax.set_xlabel("Time steps")
        ax.set_ylabel("Conservation relative error")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=1e-16)

    fig.suptitle("Calibration Ladder: Conservation Error vs. Resolution", fontsize=13)
    fig.tight_layout()
    out_path = images_dir / "calibration_ladder.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved calibration plot → %s", out_path)


def main() -> None:
    """Entry point."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Calibration Ladder Sweep")
    logger.info(
        "Scenarios: %d  |  n_bits: %s  |  n_steps: %s",
        len(LADDER_SCENARIOS), N_BITS_RANGE, N_STEPS_RANGE,
    )
    logger.info("=" * 60)

    t0 = time.perf_counter()
    results = run_ladder()
    total_time = time.perf_counter() - t0
    results["total_wall_time_s"] = round(total_time, 2)

    # Write JSON
    out_path = OUTPUT_DIR / "calibration_ladder.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)

    # Generate plot
    generate_ladder_plot(results, OUTPUT_DIR)

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION LADDER SUMMARY")
    logger.info("=" * 60)
    for scenario_data in results["scenarios"]:
        logger.info("\n  %s:", scenario_data["name"])
        logger.info("  %-8s  %-8s  %-12s  %-10s  %-8s", "n_bits", "n_steps", "rel_err", "tier", "wall_s")
        logger.info("  " + "-" * 52)
        for e in scenario_data["entries"]:
            rel_err = e.get("conservation_rel_err")
            err_str = f"{rel_err:.2e}" if rel_err is not None else "N/A"
            tier = e.get("resolution_tier", "?")
            wall = e.get("wall_time_s", 0)
            logger.info(
                "  %-8d  %-8d  %-12s  %-10s  %-8.2f",
                e["n_bits"], e["n_steps"], err_str, tier, wall,
            )

    logger.info("\nTotal sweep time: %.1f s", total_time)


if __name__ == "__main__":
    main()
