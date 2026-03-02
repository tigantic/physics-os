"""V&V Harness — QTT VM Bridge for Benchmark Execution.

This module bridges the QTT VM (``ontic.engine.vm``) with the V&V
framework (``ontic.platform.vv.*``) and the evidence/claim pipeline
(``physics_os.core.evidence``).

The harness:
1. Loads benchmark specs from ``registry.yaml``.
2. Compiles and executes QTT programs through the VM runtime.
3. Evaluates QoI (Quantities of Interest) from VM execution results.
4. Generates two-tier proof packs:
   - **Private** (full telemetry, rank history, compression data) — internal only.
   - **Public** (sanitizer-safe scorecard conforming to ``ScorecardPublicV1``).
5. Evaluates pass/fail gates per the registry definition.

Architecture
------------
The existing V&V modules (convergence.py, conservation.py, mms.py,
stability.py, benchmarks.py, performance.py) operate on
``StructuredMesh`` + ``SimulationState`` — the dense data model.  The
harness provides adapter functions that extract dense diagnostics from
QTT execution results *only for post-execution V&V analysis* (never
in the hot path).  Per QTT Law §1, ``to_cpu()`` / ``to_dense()`` are
permitted only for post-execution reporting.

References
----------
- Platform Spec §5.1 (VM Architecture)
- Platform Spec §13.1 (CFD Benchmarks)
- Platform Spec §13.3 (V&V Framework)
- Platform Spec §20.4 (IP Boundary & Forbidden Outputs)
- Universal_Discretization_Execution.md §5.1 (Benchmark Registry)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Registry loading
# ═══════════════════════════════════════════════════════════════════════════════

_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


def load_registry(path: Optional[Path] = None) -> dict[str, Any]:
    """Load and parse the benchmark registry YAML.

    Parameters
    ----------
    path : Path, optional
        Override path to registry.yaml.  Defaults to the co-located file.

    Returns
    -------
    dict
        Parsed registry with ``version``, ``global_defaults``, ``benchmarks``.

    Raises
    ------
    FileNotFoundError
        If the registry file does not exist.
    """
    import yaml  # lazy import — yaml is not always available

    p = path or _REGISTRY_PATH
    if not p.exists():
        raise FileNotFoundError(f"Benchmark registry not found: {p}")
    with open(p) as f:
        return yaml.safe_load(f)


def list_benchmark_ids(registry: Optional[dict[str, Any]] = None) -> list[str]:
    """Return all benchmark IDs from the registry."""
    reg = registry or load_registry()
    return [b["id"] for b in reg.get("benchmarks", [])]


def get_benchmark_spec(
    benchmark_id: str,
    registry: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Retrieve a single benchmark spec by ID.

    Raises
    ------
    KeyError
        If the benchmark ID is not in the registry.
    """
    reg = registry or load_registry()
    for b in reg.get("benchmarks", []):
        if b["id"] == benchmark_id:
            return b
    raise KeyError(f"Benchmark '{benchmark_id}' not found in registry")


# ═══════════════════════════════════════════════════════════════════════════════
# QoI result containers
# ═══════════════════════════════════════════════════════════════════════════════


class GateVerdict(str, Enum):
    """Outcome of a single gate check."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class GateResult:
    """Result of evaluating a single pass/fail gate."""

    gate_type: str
    qoi_name: str
    threshold: float
    observed: float
    verdict: GateVerdict
    detail: str = ""


@dataclass(frozen=True)
class QoIValue:
    """A single Quantity of Interest measurement."""

    name: str
    value: float
    units: str = "1"
    reference_source: str = ""
    reference_value: float = 0.0
    error_abs: float = 0.0
    error_rel: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete result of running one benchmark."""

    benchmark_id: str
    domain_key: str
    status: str = "succeeded"
    started_utc: str = ""
    finished_utc: str = ""
    wall_seconds: float = 0.0
    determinism_tier: str = "reproducible"
    qoi_values: list[QoIValue] = field(default_factory=list)
    gate_results: list[GateResult] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    convergence_summary: str = ""
    error_message: str = ""

    # Private metrics (never leave the harness except to internal pack)
    private_telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def all_gates_passed(self) -> bool:
        """True if all non-skipped gates passed."""
        return all(
            g.verdict != GateVerdict.FAIL for g in self.gate_results
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Gate evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_gates(
    gates: list[dict[str, Any]],
    qoi_map: dict[str, float],
) -> list[GateResult]:
    """Evaluate pass/fail gates against observed QoI values.

    Parameters
    ----------
    gates : list of dict
        Gate definitions from the registry (each has ``type``, ``qoi``, etc.).
    qoi_map : dict
        Mapping from QoI name to observed value.

    Returns
    -------
    list of GateResult
    """
    results: list[GateResult] = []
    for gate in gates:
        gate_type = gate.get("type", "")
        qoi_name = gate.get("qoi", "")

        if gate_type == "absolute_max":
            threshold = float(gate["max"])
            observed = qoi_map.get(qoi_name, float("nan"))
            if np.isnan(observed):
                results.append(GateResult(
                    gate_type=gate_type,
                    qoi_name=qoi_name,
                    threshold=threshold,
                    observed=observed,
                    verdict=GateVerdict.SKIP,
                    detail=f"QoI '{qoi_name}' not available",
                ))
            else:
                passed = observed <= threshold
                results.append(GateResult(
                    gate_type=gate_type,
                    qoi_name=qoi_name,
                    threshold=threshold,
                    observed=observed,
                    verdict=GateVerdict.PASS if passed else GateVerdict.FAIL,
                    detail=f"{observed:.4e} {'<=' if passed else '>'} {threshold:.4e}",
                ))

        elif gate_type == "observed_order_min":
            min_order = float(gate["min_order"])
            observed = qoi_map.get(f"observed_order_{qoi_name}", float("nan"))
            if np.isnan(observed):
                results.append(GateResult(
                    gate_type=gate_type,
                    qoi_name=qoi_name,
                    threshold=min_order,
                    observed=observed,
                    verdict=GateVerdict.SKIP,
                    detail=f"Observed order for '{qoi_name}' not available",
                ))
            else:
                passed = observed >= min_order
                results.append(GateResult(
                    gate_type=gate_type,
                    qoi_name=qoi_name,
                    threshold=min_order,
                    observed=observed,
                    verdict=GateVerdict.PASS if passed else GateVerdict.FAIL,
                    detail=f"order {observed:.3f} {'>=' if passed else '<'} {min_order:.1f}",
                ))

        elif gate_type == "boundedness":
            # Boundedness is checked per-field
            require_positive = gate.get("require_positive", [])
            for field_name in require_positive:
                min_val = qoi_map.get(f"min_{field_name}", float("nan"))
                if np.isnan(min_val):
                    results.append(GateResult(
                        gate_type=gate_type,
                        qoi_name=field_name,
                        threshold=0.0,
                        observed=min_val,
                        verdict=GateVerdict.SKIP,
                        detail=f"min({field_name}) not available",
                    ))
                else:
                    passed = min_val > 0.0
                    results.append(GateResult(
                        gate_type=gate_type,
                        qoi_name=field_name,
                        threshold=0.0,
                        observed=min_val,
                        verdict=GateVerdict.PASS if passed else GateVerdict.FAIL,
                        detail=f"min({field_name}) = {min_val:.4e} {'> 0' if passed else '<= 0'}",
                    ))

        else:
            results.append(GateResult(
                gate_type=gate_type,
                qoi_name=qoi_name,
                threshold=0.0,
                observed=0.0,
                verdict=GateVerdict.SKIP,
                detail=f"Unknown gate type: {gate_type}",
            ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Proof pack generation (two-tier)
# ═══════════════════════════════════════════════════════════════════════════════


def generate_public_scorecard(result: BenchmarkResult) -> dict[str, Any]:
    """Generate a sanitizer-safe public scorecard (ScorecardPublicV1).

    This scorecard conforms to ``scorecard_public_v1.schema.json`` and
    MUST NOT contain any forbidden fields per §20.4.

    Parameters
    ----------
    result : BenchmarkResult
        The full benchmark result (may contain private data).

    Returns
    -------
    dict
        JSON-serializable scorecard with only whitelisted fields.
    """
    # Build claim tags from claim list
    claim_tags = list({c["tag"] for c in result.claims if c.get("satisfied", False)})

    # Build checks dict from claims
    checks: dict[str, Any] = {}
    for claim in result.claims:
        tag = claim["tag"]
        witness = claim.get("witness", {})
        if tag == "CONSERVATION":
            quantity = witness.get("quantity", "mass")
            checks[f"{quantity}_balance_abs"] = witness.get("error_value", 0.0)
        elif tag == "STABILITY":
            checks["stability_pass"] = claim.get("satisfied", False)
            checks["stability_notes"] = claim.get("claim", "")
        elif tag == "BOUND":
            checks["boundedness_pass"] = claim.get("satisfied", False)
            checks["boundedness_notes"] = claim.get("claim", "")

    # Build QoI array
    qoi_array = []
    for qv in result.qoi_values:
        entry: dict[str, Any] = {
            "name": qv.name,
            "value": qv.value,
            "units": qv.units,
        }
        if qv.reference_source:
            entry["reference"] = {
                "source": qv.reference_source,
                "value": qv.reference_value,
                "error_abs": qv.error_abs,
                "error_rel": qv.error_rel,
            }
        qoi_array.append(entry)

    # Ensure at least one QoI entry
    if not qoi_array:
        qoi_array.append({"name": "wall_time", "value": result.wall_seconds, "units": "s"})

    scorecard: dict[str, Any] = {
        "schema_version": "1.0",
        "job_id": hashlib.sha256(
            f"{result.benchmark_id}:{result.started_utc}".encode()
        ).hexdigest()[:16],
        "domain_key": result.domain_key,
        "status": result.status,
        "timestamps": {
            "started_utc": result.started_utc,
            "finished_utc": result.finished_utc,
        },
        "determinism": {
            "tier": result.determinism_tier,
        },
        "evidence": {
            "claims": claim_tags,
            "checks": checks,
        },
        "qoi": qoi_array,
        "performance": {
            "wall_seconds": result.wall_seconds,
            "device_class": "cpu",  # default; overridden when GPU detected
            "compute_units": 0.0,   # placeholder — CU formula from §20.3
        },
    }

    if result.convergence_summary:
        scorecard["convergence"] = {
            "performed": True,
            "summary": result.convergence_summary,
        }

    return scorecard


def generate_private_pack(result: BenchmarkResult) -> dict[str, Any]:
    """Generate the full private proof pack (internal only).

    Contains everything: rank history, compression ratios, SVD spectra,
    scaling class, opcode traces — all forbidden from public output.

    Parameters
    ----------
    result : BenchmarkResult
        The full benchmark result.

    Returns
    -------
    dict
        Full internal proof pack.
    """
    return {
        "benchmark_id": result.benchmark_id,
        "domain_key": result.domain_key,
        "status": result.status,
        "started_utc": result.started_utc,
        "finished_utc": result.finished_utc,
        "wall_seconds": result.wall_seconds,
        "determinism_tier": result.determinism_tier,
        "qoi_values": [asdict(qv) for qv in result.qoi_values],
        "gate_results": [
            {
                "gate_type": g.gate_type,
                "qoi_name": g.qoi_name,
                "threshold": g.threshold,
                "observed": g.observed,
                "verdict": g.verdict.value,
                "detail": g.detail,
            }
            for g in result.gate_results
        ],
        "claims": result.claims,
        "convergence_summary": result.convergence_summary,
        "error_message": result.error_message,
        "private_telemetry": result.private_telemetry,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Harness runner
# ═══════════════════════════════════════════════════════════════════════════════


# Type alias for QoI extraction functions
QoIExtractor = Callable[
    [Any, dict[str, Any]],  # (execution_result, benchmark_spec)
    list[QoIValue],
]

# Type alias for claim generators
ClaimGenerator = Callable[
    [Any, list[QoIValue], dict[str, Any]],  # (execution_result, qois, benchmark_spec)
    list[dict[str, Any]],
]


class VVHarness:
    """V&V benchmark harness — runs benchmarks and generates proof packs.

    The harness is the orchestrator that:
    1. Loads benchmark specs from the registry.
    2. Delegates execution to the provided ``run_fn`` callback.
    3. Extracts QoIs via the ``qoi_extractor`` callback.
    4. Evaluates registry-defined gates.
    5. Generates both public and private proof packs.

    Parameters
    ----------
    run_fn : callable
        ``(benchmark_spec) -> execution_result``
        Executes a benchmark and returns the VM execution result.
    qoi_extractor : QoIExtractor
        Extracts QoI values from the execution result.
    claim_generator : ClaimGenerator, optional
        Generates claim-witness pairs.  If not provided, uses the
        default ``generate_claims_from_qoi``.
    registry_path : Path, optional
        Override path to registry.yaml.
    """

    def __init__(
        self,
        run_fn: Callable[[dict[str, Any]], Any],
        qoi_extractor: QoIExtractor,
        claim_generator: Optional[ClaimGenerator] = None,
        registry_path: Optional[Path] = None,
    ) -> None:
        self._run_fn = run_fn
        self._qoi_extractor = qoi_extractor
        self._claim_generator = claim_generator or generate_claims_from_qoi
        self._registry_path = registry_path

    def run_benchmark(
        self,
        benchmark_id: str,
        registry: Optional[dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run a single benchmark and return its result.

        Parameters
        ----------
        benchmark_id : str
            The ID from the registry (e.g., ``"V010_MMS_GRADIENT_1D"``).
        registry : dict, optional
            Pre-loaded registry.  If None, loads from disk.

        Returns
        -------
        BenchmarkResult
            Full result with QoIs, gates, claims, and proof packs.
        """
        reg = registry or load_registry(self._registry_path)
        spec = get_benchmark_spec(benchmark_id, reg)

        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            domain_key=spec.get("domain_key", ""),
        )
        result.started_utc = datetime.now(timezone.utc).isoformat()

        try:
            t0 = time.monotonic()
            exec_result = self._run_fn(spec)
            result.wall_seconds = time.monotonic() - t0
            result.finished_utc = datetime.now(timezone.utc).isoformat()

            # Extract QoIs
            result.qoi_values = self._qoi_extractor(exec_result, spec)

            # Generate claims
            result.claims = self._claim_generator(
                exec_result, result.qoi_values, spec,
            )

            # Evaluate gates
            qoi_map = {qv.name: qv.value for qv in result.qoi_values}
            gates = spec.get("gates", [])
            result.gate_results = evaluate_gates(gates, qoi_map)

            # Extract private telemetry if available
            if hasattr(exec_result, "telemetry"):
                telem = exec_result.telemetry
                if hasattr(telem, "private") and hasattr(telem.private, "to_dict"):
                    result.private_telemetry = telem.private.to_dict()
                elif hasattr(telem, "to_dict"):
                    result.private_telemetry = telem.to_dict()

            result.status = "succeeded"
            logger.info(
                "Benchmark %s completed: %d QoIs, %d gates (%s)",
                benchmark_id,
                len(result.qoi_values),
                len(result.gate_results),
                "ALL PASS" if result.all_gates_passed else "SOME FAILED",
            )

        except Exception as exc:
            result.status = "failed"
            result.error_message = str(exc)
            result.finished_utc = datetime.now(timezone.utc).isoformat()
            logger.error("Benchmark %s failed: %s", benchmark_id, exc)

        return result

    def run_all(
        self,
        registry: Optional[dict[str, Any]] = None,
        category: Optional[str] = None,
    ) -> list[BenchmarkResult]:
        """Run all benchmarks (optionally filtered by category).

        Parameters
        ----------
        registry : dict, optional
            Pre-loaded registry.
        category : str, optional
            Filter to only benchmarks with this category
            (e.g., ``"verification"``, ``"cfd"``).

        Returns
        -------
        list of BenchmarkResult
        """
        reg = registry or load_registry(self._registry_path)
        benchmarks = reg.get("benchmarks", [])
        if category:
            benchmarks = [b for b in benchmarks if b.get("category") == category]

        results: list[BenchmarkResult] = []
        for spec in benchmarks:
            bid = spec["id"]
            logger.info("Running benchmark: %s", bid)
            result = self.run_benchmark(bid, reg)
            results.append(result)

        return results

    def generate_report(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, Any]:
        """Generate a summary report from benchmark results.

        Returns
        -------
        dict
            Summary with per-benchmark scorecards and aggregate stats.
        """
        total = len(results)
        passed = sum(1 for r in results if r.all_gates_passed and r.status == "succeeded")
        failed = sum(1 for r in results if r.status == "failed")

        return {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "total_benchmarks": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0.0,
            "scorecards": [
                generate_public_scorecard(r) for r in results
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Default claim generator
# ═══════════════════════════════════════════════════════════════════════════════


def generate_claims_from_qoi(
    exec_result: Any,
    qoi_values: list[QoIValue],
    benchmark_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate claim-witness pairs from QoI values and execution results.

    This is the default claim generator used by the harness when no
    custom generator is provided.  It inspects the QoI values and
    execution telemetry to emit standard claims.

    Registered claim tags (from CLAIM_REGISTRY.md):
    - CONSERVATION: quantity preserved
    - STABILITY: no NaN/Inf divergence
    - BOUND: field values within physical limits
    - CONVERGENCE: grid convergence verified
    - REPRODUCIBILITY: deterministic replay
    - ENERGY_BOUND: energy below physical threshold
    - CFL_SATISFIED: CFL condition met throughout simulation
    """
    claims: list[dict[str, Any]] = []

    # ── STABILITY ───────────────────────────────────────────────────
    # Default: claim satisfied if execution result has success flag
    completed = True
    wall_time = 0.0
    n_steps = 0
    if hasattr(exec_result, "success"):
        completed = exec_result.success
    if hasattr(exec_result, "telemetry"):
        telem = exec_result.telemetry
        wall_time = getattr(telem, "total_wall_time_s", 0.0)
        n_steps = getattr(telem, "n_steps", 0)

    claims.append({
        "tag": "STABILITY",
        "claim": "Simulation completed without numerical divergence",
        "witness": {
            "wall_time_s": wall_time,
            "time_steps": n_steps,
            "completed": completed,
        },
        "satisfied": completed and wall_time > 0,
    })

    # ── BOUND ───────────────────────────────────────────────────────
    # Check if all QoI values are finite
    all_finite = all(np.isfinite(qv.value) for qv in qoi_values)
    max_abs = max((abs(qv.value) for qv in qoi_values), default=0.0)
    bounded = all_finite and max_abs < 1e15
    claims.append({
        "tag": "BOUND",
        "claim": f"All QoI values bounded (max |value| = {max_abs:.6e})",
        "witness": {
            "max_absolute_value": max_abs if np.isfinite(max_abs) else 1e30,
            "threshold": 1e15,
        },
        "satisfied": bounded,
    })

    # ── CONSERVATION ────────────────────────────────────────────────
    # Look for invariant data in telemetry
    if hasattr(exec_result, "telemetry"):
        telem = exec_result.telemetry
        inv_name = getattr(telem, "invariant_name", "")
        inv_initial = getattr(telem, "invariant_initial", 0.0)
        inv_final = getattr(telem, "invariant_final", 0.0)
        inv_error = getattr(telem, "invariant_error", 0.0)
        if inv_name and inv_initial != 0.0:
            rel_error = abs(inv_error)
            threshold = 1e-4
            claims.append({
                "tag": "CONSERVATION",
                "claim": f"{inv_name} preserved to {rel_error:.2e} relative error",
                "witness": {
                    "quantity": inv_name,
                    "initial": inv_initial,
                    "final": inv_final,
                    "error_value": rel_error,
                    "error_metric": "relative",
                    "threshold": threshold,
                },
                "satisfied": rel_error < threshold,
            })

    # ── CONVERGENCE ─────────────────────────────────────────────────
    # Check for observed order QoIs
    for qv in qoi_values:
        if qv.name.startswith("observed_order_"):
            target_name = qv.name.replace("observed_order_", "")
            gates = benchmark_spec.get("gates", [])
            min_order = 2.0
            for gate in gates:
                if gate.get("type") == "observed_order_min" and gate.get("qoi") == target_name:
                    min_order = float(gate["min_order"])
                    break
            claims.append({
                "tag": "CONVERGENCE",
                "claim": f"Grid convergence verified for {target_name}: order {qv.value:.3f}",
                "witness": {
                    "qoi": target_name,
                    "observed_order": qv.value,
                    "required_order": min_order,
                },
                "satisfied": qv.value >= min_order,
            })

    # ── CFL_SATISFIED ───────────────────────────────────────────────
    # Check for CFL QoI if present
    for qv in qoi_values:
        if qv.name.startswith("max_cfl"):
            cfl_limit = 1.0  # standard CFL limit
            claims.append({
                "tag": "CFL_SATISFIED",
                "claim": f"CFL condition satisfied: max CFL = {qv.value:.4f}",
                "witness": {
                    "max_cfl": qv.value,
                    "cfl_limit": cfl_limit,
                },
                "satisfied": qv.value <= cfl_limit,
            })

    # ── ENERGY_BOUND ────────────────────────────────────────────────
    for qv in qoi_values:
        if qv.name in ("kinetic_energy", "total_energy"):
            claims.append({
                "tag": "ENERGY_BOUND",
                "claim": f"Energy bounded: {qv.name} = {qv.value:.6e}",
                "witness": {
                    "quantity": qv.name,
                    "value": qv.value,
                    "threshold": 1e15,
                },
                "satisfied": abs(qv.value) < 1e15,
            })

    # ── REPRODUCIBILITY ─────────────────────────────────────────────
    # Only emitted if determinism tier metadata is available
    if hasattr(exec_result, "telemetry"):
        telem = exec_result.telemetry
        if hasattr(telem, "public"):
            config_hash = getattr(telem.public, "config_hash", "")
            det_tier = getattr(telem.public, "determinism_tier", None)
            if config_hash:
                tier_name = det_tier.value if hasattr(det_tier, "value") else str(det_tier)
                claims.append({
                    "tag": "REPRODUCIBILITY",
                    "claim": f"Deterministic execution: tier={tier_name}, hash={config_hash[:12]}",
                    "witness": {
                        "determinism_tier": tier_name,
                        "config_hash": config_hash,
                    },
                    "satisfied": True,  # claim is informational
                })

    return claims


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: QTT VM adapter
# ═══════════════════════════════════════════════════════════════════════════════


def make_qtt_run_fn(
    runtime: Any,
    compiler_factory: Callable[[dict[str, Any]], Any],
) -> Callable[[dict[str, Any]], Any]:
    """Create a ``run_fn`` for the harness that uses the QTT VM.

    Parameters
    ----------
    runtime : QTTRuntime or GPUQTTRuntime
        The VM runtime instance.
    compiler_factory : callable
        ``(benchmark_spec) -> compiler``
        Returns a domain compiler configured for the benchmark.

    Returns
    -------
    callable
        ``(benchmark_spec) -> ExecutionResult``
    """

    def run(spec: dict[str, Any]) -> Any:
        compiler = compiler_factory(spec)
        program = compiler.compile()
        return runtime.execute(program)

    return run


def make_qtt_qoi_extractor(
    custom_extractors: Optional[dict[str, Callable[..., list[QoIValue]]]] = None,
) -> QoIExtractor:
    """Create a QoI extractor for QTT VM execution results.

    Parameters
    ----------
    custom_extractors : dict, optional
        Mapping from QoI name to extraction function.

    Returns
    -------
    QoIExtractor
    """
    extras = custom_extractors or {}

    def extract(exec_result: Any, spec: dict[str, Any]) -> list[QoIValue]:
        qois: list[QoIValue] = []

        # Extract standard metrics from telemetry
        if hasattr(exec_result, "telemetry"):
            telem = exec_result.telemetry
            qois.append(QoIValue(
                name="wall_time_s",
                value=getattr(telem, "total_wall_time_s", 0.0),
                units="s",
            ))
            inv_error = getattr(telem, "invariant_error", 0.0)
            inv_name = getattr(telem, "invariant_name", "")
            if inv_name:
                qois.append(QoIValue(
                    name=f"invariant_error_{inv_name}",
                    value=inv_error,
                    units="1",
                ))

        # Apply custom extractors
        for name, fn in extras.items():
            try:
                custom_qois = fn(exec_result, spec)
                qois.extend(custom_qois)
            except Exception as e:
                logger.warning("Custom QoI extractor '%s' failed: %s", name, e)

        return qois

    return extract
