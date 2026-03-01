#!/usr/bin/env python3
"""Real-world simulation scenario runner with full output packages.

Runs 10 diverse engineering scenarios through the full Problem Template
pipeline:

    ProblemSpec → compile_problem() → ExecutionConfig → execute()
    → sanitize_result() → generate_validation_report()
    → generate_claims() → issue_certificate()

Each scenario models a legitimate engineering situation with real
fluid properties, realistic geometry, and correct flow conditions.

Outputs (all written to ``scenario_output/``):
- ``scenario_results.json``  — structured data for every scenario
- ``images/``                — field contour PNGs, diagnostics, summary charts
- ``videos/``                — GIF / MP4 time-evolution animations
- ``reports/scenario_report.html`` — standalone HTML report
- ``reports/scenario_report.pdf``  — PDF report (via WeasyPrint)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Ensure project root is on sys.path ───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import physics_os
from physics_os.core.certificates import issue_certificate
from physics_os.core.evidence import generate_claims, generate_validation_report
from physics_os.core.executor import ExecutionConfig, execute
from physics_os.core.hasher import content_hash
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
from scenario_viz import generate_all_visuals
from scenario_report import (
    generate_html_report,
    generate_pdf_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("scenario_runner")


# ═══════════════════════════════════════════════════════════════════
# Scenario definitions — 10 real-world engineering cases
# ═══════════════════════════════════════════════════════════════════

SCENARIOS: list[dict[str, Any]] = [
    # ── 1. Automotive: Side-mirror simplified (cylinder) at highway speed ──
    {
        "name": "Highway Side-Mirror (Cylinder in Air)",
        "description": (
            "Simplified side-mirror: circular cylinder D=0.10 m in "
            "air at 30 m/s (108 km/h). Re ≈ 200,000. Expect vortex "
            "street, Cd ≈ 1.0–1.2."
        ),
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
    # ── 2. Aerospace: NACA 2412 airfoil at cruise ────────────────
    {
        "name": "GA Aircraft Wing (NACA 2412)",
        "description": (
            "General-aviation wing section: NACA 2412, chord 1.5 m, "
            "cruise at 70 m/s in air (Ma ≈ 0.20). Re ≈ 7 × 10⁶. "
            "Expect attached flow, non-zero lift."
        ),
        "spec": ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.NACA_AIRFOIL,
                params={"code": 2412, "chord": 1.5},
            ),
            flow=FlowConditions(velocity=70.0, fluid="air"),
            boundaries=BoundarySpec(),
            quality="quick",
            max_rank=32,
        ),
    },
    # ── 3. HVAC: Heated pipe in air crossflow ────────────────────
    {
        "name": "HVAC Pipe Crossflow (Heated Cylinder)",
        "description": (
            "HVAC duct: 25 mm diameter hot-water pipe in 2.5 m/s "
            "air crossflow.  Re ≈ 4,200.  Expect mixed convection "
            "heat transfer, Churchill-Bernstein Nu correlation test."
        ),
        "spec": ProblemSpec(
            problem_class=ProblemClass.HEAT_TRANSFER,
            geometry=GeometrySpec(
                shape=GeometryType.CIRCLE, params={"radius": 0.0125}
            ),
            flow=FlowConditions(velocity=2.5, fluid="air", temperature=350.0),
            boundaries=BoundarySpec(),
            quality="quick",
            max_rank=32,
        ),
    },
    # ── 4. Chemical Eng: 90° pipe bend in water ─────────────────
    {
        "name": "Pipe Bend (Water, 90°)",
        "description": (
            "Industrial pipe bend: 50 mm bore, 90° bend, water at "
            "1.5 m/s.  Re ≈ 75,000.  Expect secondary flow, "
            "pressure drop prediction."
        ),
        "spec": ProblemSpec(
            problem_class=ProblemClass.INTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.PIPE_BEND,
                params={"radius": 0.025, "bend_angle": 90.0},
            ),
            flow=FlowConditions(velocity=1.5, fluid="water"),
            boundaries=BoundarySpec(),
            quality="quick",
            max_rank=32,
        ),
    },
    # ── 5. Marine: Elliptical hull cross-section in seawater ─────
    {
        "name": "Hull Cross-Section (Ellipse in Seawater)",
        "description": (
            "Simplified ship hull cross-section: ellipse (a=2.0 m, "
            "b=0.6 m) in seawater at 3.0 m/s (≈ 6 knots).  "
            "Re ≈ 3.5 × 10⁶.  Expect turbulent wake."
        ),
        "spec": ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.ELLIPSE,
                params={"semi_major": 2.0, "semi_minor": 0.6},
            ),
            flow=FlowConditions(velocity=3.0, fluid="seawater"),
            boundaries=BoundarySpec(),
            quality="quick",
            max_rank=32,
        ),
    },
    # ── 6. Electronics: Fin-array heat sink ──────────────────────
    {
        "name": "Electronics Heat Sink (Fin Array)",
        "description": (
            "CPU heat sink: fin array with 8 fins, 50 mm base, "
            "20 mm fin height, 2 mm fin thickness, forced air at "
            "3 m/s.  Expect convective cooling, Nu estimation."
        ),
        "spec": ProblemSpec(
            problem_class=ProblemClass.HEAT_TRANSFER,
            geometry=GeometrySpec(
                shape=GeometryType.FIN_ARRAY,
                params={
                    "base_width": 0.050,
                    "fin_height": 0.020,
                    "fin_thickness": 0.002,
                    "n_fins": 8,
                },
            ),
            flow=FlowConditions(velocity=3.0, fluid="air", temperature=340.0),
            boundaries=BoundarySpec(),
            quality="quick",
            max_rank=32,
        ),
    },
    # ── 7. Civil: Wind load on rectangular building ──────────────
    {
        "name": "Wind Load on Building (Rectangle)",
        "description": (
            "Wind engineering: rectangular building cross-section "
            "(width=20 m, height=60 m) in 15 m/s wind.  "
            "Re ≈ 2.4 × 10⁷.  Expect massive separation, "
            "vortex shedding, high drag."
        ),
        "spec": ProblemSpec(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.RECTANGLE,
                params={"width": 20.0, "height": 60.0},
            ),
            flow=FlowConditions(velocity=15.0, fluid="air"),
            boundaries=BoundarySpec(),
            quality="quick",
            max_rank=32,
        ),
    },
    # ── 8. Boundary layer: Flat plate transition ─────────────────
    {
        "name": "Flat Plate Boundary Layer (Air)",
        "description": (
            "Canonical BL study: flat plate L=1.0 m in air at "
            "10 m/s.  Re_L ≈ 667,000 — transition regime.  "
            "Expect Blasius profile, Cf from correlation."
        ),
        "spec": ProblemSpec(
            problem_class=ProblemClass.BOUNDARY_LAYER,
            geometry=GeometrySpec(
                shape=GeometryType.FLAT_PLATE,
                params={"length": 1.0},
            ),
            flow=FlowConditions(velocity=10.0, fluid="air"),
            boundaries=BoundarySpec(),
            quality="quick",
            max_rank=32,
        ),
    },
    # ── 9. Nuclear: Liquid sodium flow over cylinder ─────────────
    {
        "name": "Reactor Coolant (Liquid Sodium over Cylinder)",
        "description": (
            "Liquid-metal heat transfer: circular cylinder D=20 mm "
            "in liquid sodium at 0.5 m/s.  Very low Prandtl number "
            "(Pr ≈ 0.005).  Re ≈ 19,000.  Unique thermal BL."
        ),
        "spec": ProblemSpec(
            problem_class=ProblemClass.HEAT_TRANSFER,
            geometry=GeometrySpec(
                shape=GeometryType.CIRCLE, params={"radius": 0.01}
            ),
            flow=FlowConditions(
                velocity=0.5, fluid="liquid_sodium", temperature=573.0,
            ),
            boundaries=BoundarySpec(),
            quality="quick",
            max_rank=32,
        ),
    },
    # ── 10. Backward-facing step: Separated flow ─────────────────
    {
        "name": "Backward-Facing Step (Water)",
        "description": (
            "Classic validation case: backward-facing step with "
            "h=0.01 m, expansion ratio 2:1, water at 0.5 m/s.  "
            "Re_h ≈ 5,000.  Expect reattachment length ≈ 7h."
        ),
        "spec": ProblemSpec(
            problem_class=ProblemClass.INTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.BACKWARD_STEP,
                params={"step_height": 0.01, "expansion_ratio": 2.0},
            ),
            flow=FlowConditions(velocity=0.5, fluid="water"),
            boundaries=BoundarySpec(
                inlet="uniform",
                outlet="zero_gradient",
                walls="no_slip",
                top="symmetry",
                bottom="no_slip",
            ),
            quality="quick",
            max_rank=32,
        ),
    },
    # ── 11. Easy-mode: Stokes creeping flow (glycerol, Re < 1) ──
    {
        "name": "Stokes Creeping Flow (Glycerol, Re ≈ 0.09)",
        "description": (
            "Validation case: very low Re creeping flow.  Glycerol "
            "(ν = 1.12 × 10⁻³ m²/s) past a 10 mm cylinder at "
            "0.01 m/s → Re ≈ 0.09.  Smooth, diffusion-dominated "
            "solution — should conserve circulation even at 64×64."
        ),
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
    # ── 12. Easy-mode: Slow laminar pipe bend (engine oil, Re ≈ 5) ──
    {
        "name": "Laminar Pipe Bend (Engine Oil, Re ≈ 5)",
        "description": (
            "Validation case: laminar internal flow.  Engine oil "
            "(ν = 5.5 × 10⁻⁴ m²/s) through a 20 mm bore pipe "
            "bend at 0.025 m/s → Re ≈ 5.  Well-resolved at "
            "64×64.  Validates NS solver+validator handshake."
        ),
        "spec": ProblemSpec(
            problem_class=ProblemClass.INTERNAL_FLOW,
            geometry=GeometrySpec(
                shape=GeometryType.PIPE_BEND,
                params={"radius": 0.01, "bend_angle": 90.0},
            ),
            flow=FlowConditions(velocity=0.025, fluid="engine_oil"),
            boundaries=BoundarySpec(),
            quality="quick",
            max_rank=32,
        ),
    },
]

# ── Snapshot schedule ──────────────────────────────────────────────
# Each snapshot re-executes from step 0 so we cap the snapshot
# step depth to keep total runtime bounded.
_SNAPSHOT_COUNT = 4
_SNAPSHOT_CAP = 500


def _compute_snapshot_steps(n_steps: int) -> list[int]:
    """Return evenly-spaced snapshot step counts, capped for cost.

    Each snapshot triggers a fresh execution from step 0, so we
    limit the deepest snapshot to min(n_steps, _SNAPSHOT_CAP) to
    keep overhead bounded even for 10 000-step production runs.
    """
    cap = min(n_steps, _SNAPSHOT_CAP)
    if cap <= _SNAPSHOT_COUNT:
        return list(range(1, cap))
    spacing = cap // (_SNAPSHOT_COUNT + 1)
    return [spacing * (i + 1) for i in range(_SNAPSHOT_COUNT)]

# Output directory for all generated artifacts
OUTPUT_DIR = PROJECT_ROOT / "scenario_output"


# ═══════════════════════════════════════════════════════════════════
# Telemetry / snapshot extraction helpers
# ═══════════════════════════════════════════════════════════════════


def _extract_telemetry_steps(result: Any) -> list[dict[str, Any]]:
    """Extract per-step telemetry from an execution result.

    Returns a list of dicts (one per step) with chi_max,
    wall_time_s, field_norms, invariant_values, etc.
    """
    try:
        if hasattr(result, "telemetry") and result.telemetry is not None:
            steps = result.telemetry.steps
            if steps:
                return [asdict(s) for s in steps]
    except Exception as exc:
        logger.debug("Could not extract telemetry steps: %s", exc)
    return []


def _capture_field_snapshots(
    config_template: ExecutionConfig,
    domain: str,
    step_counts: list[int],
) -> dict[str, list[dict[str, Any]]]:
    """Capture intermediate field states by running sub-simulations.

    For each step_count in *step_counts*, runs a fresh execution
    from step 0 and extracts the sanitized field values.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from field_name → list of {step, values} dicts.
    """
    snapshots: dict[str, list[dict[str, Any]]] = {}

    for n_steps in step_counts:
        try:
            sub_config = ExecutionConfig(
                domain=config_template.domain,
                n_bits=config_template.n_bits,
                n_steps=n_steps,
                dt=config_template.dt,
                max_rank=config_template.max_rank,
                truncation_tol=config_template.truncation_tol,
                parameters=config_template.parameters,
            )
            sub_result = execute(sub_config)
            if not sub_result.success:
                continue

            sanitized = sanitize_result(sub_result, domain)
            fields = sanitized.get("fields", {})

            for fname, fdata in fields.items():
                values = fdata.get("values", [])
                if not values:
                    continue
                if fname not in snapshots:
                    snapshots[fname] = []
                snapshots[fname].append({
                    "step": n_steps,
                    "values": values,
                })
        except Exception as exc:
            logger.debug("Snapshot at step %d failed: %s", n_steps, exc)
            continue

    return snapshots


# ═══════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════


def run_scenario(
    scenario: dict[str, Any],
    index: int,
    *,
    skip_snapshots: bool = False,
) -> dict[str, Any]:
    """Run a single scenario through the full pipeline.

    Returns a structured output package for the scenario.
    """
    spec: ProblemSpec = scenario["spec"]
    name: str = scenario["name"]
    description: str = scenario["description"]

    package: dict[str, Any] = {
        "scenario_index": index,
        "name": name,
        "description": description,
        "status": "pending",
        "input_spec": spec.model_dump(),
    }

    t0 = time.perf_counter()

    # ── Step 1: Compile ──────────────────────────────────────────
    try:
        compiled = compile_problem(spec)
        package["compilation"] = {
            "domain": compiled.domain,
            "n_bits": compiled.n_bits,
            "n_steps": compiled.n_steps,
            "dt": compiled.dt,
            "max_rank": compiled.max_rank,
            "parameters": compiled.parameters,
            "reynolds_number": compiled.reynolds_number,
            "mach_number": compiled.mach_number,
            "characteristic_length": compiled.characteristic_length,
            "fluid_name": compiled.fluid_name,
            "geometry_type": compiled.geometry_type,
            "problem_class": compiled.problem_class,
            "quality_tier": compiled.quality_tier,
            "warnings": compiled.warnings,
            "resolution_grid_1d": compiled.resolution_grid_1d,
            "boundary_layer_thickness": compiled.boundary_layer_thickness,
            "domain_extent": compiled.domain_extent,
        }
        logger.info(
            "[%d] %-45s  compiled → domain=%s  n_bits=%d  Re=%.2e",
            index, name, compiled.domain, compiled.n_bits,
            compiled.reynolds_number,
        )
    except Exception as exc:
        package["status"] = "compilation_error"
        package["error"] = str(exc)
        package["traceback"] = traceback.format_exc()
        logger.error("[%d] %-45s  COMPILE FAILED: %s", index, name, exc)
        return package

    # ── Step 2: Execute at compiler-recommended resolution ────────
    try:
        exec_n_bits = compiled.n_bits
        exec_n_steps = compiled.n_steps
        exec_dt = compiled.dt
        package["execution_params"] = {
            "n_bits": exec_n_bits,
            "n_steps": exec_n_steps,
            "dt": exec_dt,
            "grid": f"{2**exec_n_bits}×{2**exec_n_bits}",
        }
        logger.info(
            "[%d] %-45s  executing → grid=%d×%d  steps=%d  dt=%.4e",
            index, name, 2**exec_n_bits, 2**exec_n_bits,
            exec_n_steps, exec_dt,
        )
        config = ExecutionConfig(
            domain=compiled.domain,
            n_bits=exec_n_bits,
            n_steps=exec_n_steps,
            dt=exec_dt,
            max_rank=compiled.max_rank,
            truncation_tol=1e-10,
            parameters=compiled.parameters,
        )
        result = execute(config)
        if not result.success:
            package["status"] = "execution_error"
            package["error"] = str(result.error)
            logger.error("[%d] %-45s  EXECUTE FAILED: %s", index, name, result.error)
            return package
        exec_time = time.perf_counter() - t0
        logger.info(
            "[%d] %-45s  executed in %.3f s",
            index, name, exec_time,
        )

        # Extract per-step telemetry for diagnostics
        telemetry_steps = _extract_telemetry_steps(result)
        if telemetry_steps:
            package["telemetry_steps"] = telemetry_steps
            logger.info(
                "[%d] %-45s  captured %d telemetry steps",
                index, name, len(telemetry_steps),
            )

    except Exception as exc:
        package["status"] = "execution_error"
        package["error"] = str(exc)
        package["traceback"] = traceback.format_exc()
        logger.error("[%d] %-45s  EXECUTE FAILED: %s", index, name, exc)
        return package

    # ── Step 3: Sanitize ─────────────────────────────────────────
    try:
        execution_context = {
            "n_bits": exec_n_bits,
            "n_steps": exec_n_steps,
            "recommended_n_bits": compiled.n_bits,
            "recommended_n_steps": compiled.n_steps,
        }
        sanitized = sanitize_result(
            result, compiled.domain,
            execution_context=execution_context,
        )
        package["result"] = sanitized
        logger.info(
            "[%d] %-45s  sanitized → %d field(s)",
            index, name,
            len(sanitized.get("fields", {})),
        )
    except Exception as exc:
        package["status"] = "sanitize_error"
        package["error"] = str(exc)
        package["traceback"] = traceback.format_exc()
        logger.error("[%d] %-45s  SANITIZE FAILED: %s", index, name, exc)
        return package

    # ── Step 3b: Capture field snapshots for animation ───────────
    if skip_snapshots:
        package["field_snapshots"] = {}
        logger.info("[%d] %-45s  snapshots skipped (--no-snapshots)", index, name)
    else:
        try:
            snapshot_steps = _compute_snapshot_steps(exec_n_steps)
            if snapshot_steps:
                logger.info(
                    "[%d] %-45s  capturing snapshots at steps %s ...",
                    index, name, snapshot_steps,
                )
                field_snapshots = _capture_field_snapshots(
                    config, compiled.domain, snapshot_steps,
                )
                # Add the final field state as the last snapshot
                for fname, fdata in sanitized.get("fields", {}).items():
                    if fdata.get("values"):
                        if fname not in field_snapshots:
                            field_snapshots[fname] = []
                        field_snapshots[fname].append({
                            "step": exec_n_steps,
                            "values": fdata["values"],
                        })
                package["field_snapshots"] = field_snapshots
                total_snaps = sum(len(v) for v in field_snapshots.values())
                logger.info(
                    "[%d] %-45s  captured %d total snapshots across %d fields",
                    index, name, total_snaps, len(field_snapshots),
                )
        except Exception as exc:
            logger.warning(
                "[%d] %-45s  snapshot capture failed (non-fatal): %s",
                index, name, exc,
            )
            package["field_snapshots"] = {}

    # ── Step 4: Validate ─────────────────────────────────────────
    try:
        validation = generate_validation_report(sanitized, compiled.domain)
        package["validation"] = validation
        logger.info(
            "[%d] %-45s  valid=%s  checks=%d",
            index, name, validation["valid"],
            len(validation["checks"]),
        )
    except Exception as exc:
        package["status"] = "validation_error"
        package["error"] = str(exc)
        package["traceback"] = traceback.format_exc()
        logger.error("[%d] %-45s  VALIDATE FAILED: %s", index, name, exc)
        return package

    # ── Step 5: Claims + Certificate ─────────────────────────────
    try:
        claims = generate_claims(sanitized, compiled.domain)
        result_hash = content_hash(sanitized)
        config_hash = content_hash(config.merged_parameters)
        input_hash = content_hash(spec.model_dump())

        cert = issue_certificate(
            job_id=f"scenario-{index:02d}-{content_hash({'t': time.time()})[:12]}",
            claims=claims,
            input_manifest_hash=input_hash,
            result_hash=result_hash,
            config_hash=config_hash,
            runtime_version=physics_os.RUNTIME_VERSION,
        )
        package["claims"] = claims
        package["certificate"] = cert
        package["hashes"] = {
            "input_manifest": input_hash,
            "result": result_hash,
            "config": config_hash,
        }

        all_satisfied = all(c["satisfied"] for c in claims)
        logger.info(
            "[%d] %-45s  %d claim(s), all_satisfied=%s, cert issued",
            index, name, len(claims), all_satisfied,
        )
    except Exception as exc:
        package["status"] = "attestation_error"
        package["error"] = str(exc)
        package["traceback"] = traceback.format_exc()
        logger.error("[%d] %-45s  ATTEST FAILED: %s", index, name, exc)
        return package

    # ── Finalize ─────────────────────────────────────────────────
    total_time = time.perf_counter() - t0
    package["status"] = "success"
    package["total_wall_time_s"] = round(total_time, 4)

    return package


# ═══════════════════════════════════════════════════════════════════
# Summary builder
# ═══════════════════════════════════════════════════════════════════


def build_summary(packages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a high-level summary of all scenario runs."""
    succeeded = [p for p in packages if p["status"] == "success"]
    failed = [p for p in packages if p["status"] != "success"]

    # Aggregate stats
    total_time = sum(p.get("total_wall_time_s", 0) for p in succeeded)
    total_grid_pts = 0
    total_fields = 0
    all_valid = True
    all_claims_satisfied = True

    for p in succeeded:
        perf = p.get("result", {}).get("performance", {})
        total_grid_pts += perf.get("grid_points", 0)
        total_fields += len(p.get("result", {}).get("fields", {}))
        if not p.get("validation", {}).get("valid", False):
            all_valid = False
        for c in p.get("claims", []):
            if not c.get("satisfied", False):
                all_claims_satisfied = False

    return {
        "total_scenarios": len(packages),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "total_wall_time_s": round(total_time, 3),
        "total_grid_points_computed": total_grid_pts,
        "total_fields_produced": total_fields,
        "all_validations_passed": all_valid,
        "all_claims_satisfied": all_claims_satisfied,
        "runtime_version": physics_os.RUNTIME_VERSION,
        "platform_version": physics_os.__version__,
        "failed_scenarios": [
            {"index": p["scenario_index"], "name": p["name"],
             "status": p["status"], "error": p.get("error", "")}
            for p in failed
        ],
        "scenario_table": [
            {
                "index": p["scenario_index"],
                "name": p["name"],
                "status": p["status"],
                "domain": p.get("compilation", {}).get("domain", ""),
                "reynolds": p.get("compilation", {}).get("reynolds_number", 0),
                "mach": p.get("compilation", {}).get("mach_number", 0),
                "n_bits": p.get("compilation", {}).get("n_bits", 0),
                "quality": p.get("compilation", {}).get("quality_tier", ""),
                "valid": p.get("validation", {}).get("valid", None),
                "wall_time_s": p.get("total_wall_time_s", 0),
            }
            for p in packages
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """Run all scenarios and write combined output package."""
    parser = argparse.ArgumentParser(
        description="Physics OS — Real-World Scenario Runner",
    )
    parser.add_argument(
        "--scenarios", type=str, default=None,
        help=(
            "Comma-separated 1-based scenario indices to run "
            "(e.g. '6,11,12'). Default: all."
        ),
    )
    parser.add_argument(
        "--no-snapshots", action="store_true",
        help="Skip field snapshot capture (faster runs).",
    )
    args = parser.parse_args()

    # Build the scenario subset
    if args.scenarios:
        indices = [int(x.strip()) for x in args.scenarios.split(",")]
        selected = [
            (i, SCENARIOS[i - 1])
            for i in indices
            if 1 <= i <= len(SCENARIOS)
        ]
    else:
        selected = [(i, s) for i, s in enumerate(SCENARIOS, 1)]

    banner = (
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║  Physics OS — Real-World Scenario Runner                ║\n"
        "║  Full Pipeline: Compile → Execute → Sanitize            ║\n"
        "║                 → Validate → Attest → Certificate       ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )
    print(banner)
    logger.info("Running %d scenarios ...", len(selected))

    packages: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for idx, scenario in selected:
        print(f"\n{'─' * 60}")
        print(f"  Scenario {idx}/{len(SCENARIOS)}: {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"{'─' * 60}")
        pkg = run_scenario(scenario, idx, skip_snapshots=args.no_snapshots)
        packages.append(pkg)

        # Print quick status
        if pkg["status"] == "success":
            v = pkg.get("validation", {})
            c = pkg.get("claims", [])
            perf = pkg.get("result", {}).get("performance", {})
            print(
                f"  ✓ SUCCESS  valid={v.get('valid')}  "
                f"claims={sum(1 for x in c if x['satisfied'])}/{len(c)}  "
                f"wall={perf.get('wall_time_s', 0):.3f}s  "
                f"throughput={perf.get('throughput_gp_per_s', 0):.0f} gp/s"
            )
        else:
            print(f"  ✗ {pkg['status'].upper()}: {pkg.get('error', 'unknown')}")

    t_total = time.perf_counter() - t_start

    # ── Build summary ────────────────────────────────────────────
    summary = build_summary(packages)
    summary["total_runner_time_s"] = round(t_total, 3)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()

    combined = {
        "meta": {
            "title": "Physics OS — Real-World Scenario Output Package",
            "generated_at": summary["generated_at"],
            "runtime_version": physics_os.RUNTIME_VERSION,
            "platform_version": physics_os.__version__,
            "total_scenarios": len(SCENARIOS),
        },
        "summary": summary,
        "scenarios": packages,
    }

    # ── Write data output ──────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = OUTPUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_path = data_dir / "scenario_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    # Also keep the legacy location
    legacy_path = PROJECT_ROOT / "scenario_results.json"
    with open(legacy_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n{'═' * 60}")
    print(f"  SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Scenarios:        {summary['total_scenarios']}")
    print(f"  Succeeded:        {summary['succeeded']}")
    print(f"  Failed:           {summary['failed']}")
    print(f"  Total wall time:  {summary['total_wall_time_s']:.3f} s")
    print(f"  Grid points:      {summary['total_grid_points_computed']:,}")
    print(f"  Fields produced:  {summary['total_fields_produced']}")
    print(f"  All valid:        {summary['all_validations_passed']}")
    print(f"  All claims met:   {summary['all_claims_satisfied']}")
    print(f"  Data output:      {out_path}")
    print(f"{'═' * 60}\n")

    if summary["failed"]:
        print("  FAILED SCENARIOS:")
        for f_item in summary["failed_scenarios"]:
            print(f"    [{f_item['index']}] {f_item['name']}: {f_item['error']}")
        print()

    # ── Generate visualizations ──────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  GENERATING VISUALIZATIONS")
    print(f"{'═' * 60}\n")

    try:
        manifest = generate_all_visuals(packages, OUTPUT_DIR)
        n_images = sum(
            len(v) if isinstance(v, dict) else 0
            for v in manifest.get("images", {}).values()
        )
        n_videos = len(manifest.get("videos", {}))
        print(f"  Images:  {n_images}")
        print(f"  Videos:  {n_videos}")
    except Exception as exc:
        logger.error("Visualization generation failed: %s", exc)
        traceback.print_exc()
        manifest = {"images": {}, "videos": {}, "base64": {}}

    # ── Generate reports ─────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  GENERATING REPORTS")
    print(f"{'═' * 60}\n")

    reports_dir = OUTPUT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        html_path = generate_html_report(
            combined, manifest,
            reports_dir / "scenario_report.html",
        )
        print(f"  HTML:    {html_path}")

        pdf_path = generate_pdf_report(
            html_path,
            reports_dir / "scenario_report.pdf",
        )
        if pdf_path.suffix == ".pdf":
            print(f"  PDF:     {pdf_path}")
        else:
            print(f"  PDF:     skipped (WeasyPrint not available)")
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        traceback.print_exc()

    # ── Final summary ────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  OUTPUT PACKAGE COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Directory:  {OUTPUT_DIR}")
    print(f"  Data:       {out_path.relative_to(OUTPUT_DIR)}")
    print(f"  Images:     images/")
    print(f"  Videos:     videos/")
    print(f"  Reports:    reports/")
    t_grand_total = time.perf_counter() - t_start
    print(f"  Total time: {t_grand_total:.1f}s")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
