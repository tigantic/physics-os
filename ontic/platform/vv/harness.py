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


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Convergence Study Runner
# ═══════════════════════════════════════════════════════════════════════════════


def _taylor_green_analytic_omega(
    x: np.ndarray, y: np.ndarray, t: float, nu: float,
) -> np.ndarray:
    """Analytic vorticity for 2D Taylor–Green vortex on [0,1]².

    ω(x, y, t) = 2 sin(2πx) sin(2πy) exp(-8π²νt)
    """
    decay = np.exp(-8.0 * np.pi**2 * nu * t)
    return 2.0 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y) * decay


def _taylor_green_analytic_psi(
    x: np.ndarray, y: np.ndarray, t: float, nu: float,
) -> np.ndarray:
    """Analytic stream function for 2D Taylor–Green vortex on [0,1]².

    ∇²ψ = -ω  with  ω = 2 sin(2πx) sin(2πy)
    ψ(x, y, t) = [1/(4π²)] sin(2πx) sin(2πy) exp(-8π²νt)

    Eigenvalue:  ∇²[sin(kx)sin(ky)] = -2k² sin(kx)sin(ky),  k=2π
    So  ψ = ω / (2k²) = 2 sin(...)/(8π²) = sin(...)/(4π²).
    """
    decay = np.exp(-8.0 * np.pi**2 * nu * t)
    return (
        (1.0 / (4.0 * np.pi**2))
        * np.sin(2.0 * np.pi * x)
        * np.sin(2.0 * np.pi * y)
        * decay
    )


def _taylor_green_kinetic_energy(t: float, nu: float) -> float:
    """Analytic kinetic energy E(t) = 1/(16π²) exp(-16π²νt).

    Derived from E = (1/2)∫|∇ψ|² dA with the single-mode solution.
    """
    return (1.0 / (16.0 * np.pi**2)) * np.exp(-16.0 * np.pi**2 * nu * t)


def _taylor_green_enstrophy(t: float, nu: float) -> float:
    """Analytic enstrophy Z(t) = (1/2)∫ω² dA on [0,1]².

    For ω = 2 sin(2πx) sin(2πy) exp(-8π²νt):
      Z = (1/2) · 4 · (1/4) · exp(-16π²νt) = (1/2) exp(-16π²νt)
    """
    return 0.5 * np.exp(-16.0 * np.pi**2 * nu * t)


def _compute_qoi_from_result(
    result: Any,
    n_bits: int,
    nu: float,
    dt: float,
    n_steps: int,
    qoi_names: list[str],
) -> dict[str, float]:
    """Compute all requested QoIs from a GPUExecutionResult.

    This runs post-execution: ``to_cpu()`` / dense reconstruction
    are permitted per QTT Law §1 (post-execution reporting only).

    Parameters
    ----------
    result : GPUExecutionResult
        Execution result with GPU-resident fields and probes.
    n_bits : int
        Bits per spatial dimension.
    nu : float
        Kinematic viscosity.
    dt : float
        Time step.
    n_steps : int
        Number of time steps executed.
    qoi_names : list of str
        Which QoIs to compute.

    Returns
    -------
    dict mapping QoI name → value
    """
    T = dt * n_steps
    N = 2 ** n_bits
    h = 1.0 / N
    qois: dict[str, float] = {}

    # ── Dense field reconstruction (post-execution only) ────────────
    # QTT Law §5: to_dense() is ONLY for external diagnostics.
    x1d = np.linspace(0.0, 1.0, N, endpoint=False) + 0.5 * h
    xx, yy = np.meshgrid(x1d, x1d, indexing="ij")

    cpu_fields: dict[str, Any] = {}
    for name, gpu_t in result.fields.items():
        cpu_fields[name] = gpu_t.to_cpu()

    # Determine which fields need dense reconstruction
    OMEGA_QOIS = {
        "omega_error_L2", "omega_error_L2_rel",
        "enstrophy", "enstrophy_error_rel",
        "kinetic_energy_error_rel",
        "total_circulation_abs_error",
    }
    PSI_QOIS = {
        "psi_error_L2", "psi_error_L2_rel",
        "kinetic_energy_error_rel",
    }
    need_omega = bool(OMEGA_QOIS & set(qoi_names))
    need_psi = bool(PSI_QOIS & set(qoi_names))

    omega_dense: np.ndarray | None = None
    psi_dense: np.ndarray | None = None

    if need_omega and "omega" in cpu_fields:
        omega_cpu = cpu_fields["omega"]
        if hasattr(omega_cpu, "to_dense"):
            omega_dense = omega_cpu.to_dense()
            if hasattr(omega_dense, "numpy"):
                omega_dense = omega_dense.numpy()
            omega_dense = np.asarray(omega_dense, dtype=np.float64).reshape(N, N)

    if need_psi and "psi" in cpu_fields:
        psi_cpu = cpu_fields["psi"]
        if hasattr(psi_cpu, "to_dense"):
            psi_dense = psi_cpu.to_dense()
            if hasattr(psi_dense, "numpy"):
                psi_dense = psi_dense.numpy()
            psi_dense = np.asarray(psi_dense, dtype=np.float64).reshape(N, N)

    # Pre-compute analytic fields (shared across multiple QoIs)
    omega_exact = _taylor_green_analytic_omega(xx, yy, T, nu) if need_omega else None
    psi_exact = _taylor_green_analytic_psi(xx, yy, T, nu) if need_psi else None

    # ═════════════════════════════════════════════════════════════════
    # FIELD ERROR NORMS
    # Continuous L²([0,1]²) norms with h² midpoint-rule quadrature:
    #   ||f||_L² = sqrt(∫|f|² dA) ≈ sqrt(h² · Σᵢ |f(xᵢ)|²)
    # ═════════════════════════════════════════════════════════════════

    # ── omega_error_L2 / omega_error_L2_rel ────────────────────────
    if omega_dense is not None and omega_exact is not None:
        omega_diff = omega_dense - omega_exact
        omega_err_l2 = float(np.sqrt(np.sum(omega_diff**2) * h * h))
        omega_exact_l2 = float(np.sqrt(np.sum(omega_exact**2) * h * h))

        if "omega_error_L2" in qoi_names:
            qois["omega_error_L2"] = omega_err_l2
        if "omega_error_L2_rel" in qoi_names and omega_exact_l2 > 0:
            qois["omega_error_L2_rel"] = omega_err_l2 / omega_exact_l2

    # ── psi_error_L2 / psi_error_L2_rel ────────────────────────────
    if psi_dense is not None and psi_exact is not None:
        psi_diff = psi_dense - psi_exact
        psi_err_l2 = float(np.sqrt(np.sum(psi_diff**2) * h * h))
        psi_exact_l2 = float(np.sqrt(np.sum(psi_exact**2) * h * h))

        if "psi_error_L2" in qoi_names:
            qois["psi_error_L2"] = psi_err_l2
        if "psi_error_L2_rel" in qoi_names and psi_exact_l2 > 0:
            qois["psi_error_L2_rel"] = psi_err_l2 / psi_exact_l2

    # ═════════════════════════════════════════════════════════════════
    # SCALE-CONSISTENT PHYSICAL QUANTITIES
    # These converge to grid-independent constants (the analytic values).
    # ═════════════════════════════════════════════════════════════════

    # ── enstrophy / enstrophy_error_rel ─────────────────────────────
    if omega_dense is not None and any(
        q in qoi_names for q in ("enstrophy", "enstrophy_error_rel")
    ):
        enstrophy_num = float(0.5 * np.sum(omega_dense**2) * h * h)
        enstrophy_exact = _taylor_green_enstrophy(T, nu)
        if "enstrophy" in qoi_names:
            qois["enstrophy"] = enstrophy_num
        if "enstrophy_error_rel" in qoi_names and enstrophy_exact > 0:
            qois["enstrophy_error_rel"] = float(
                abs(enstrophy_num - enstrophy_exact) / enstrophy_exact
            )

    # ── kinetic_energy_error_rel ─────────────────────────────────────
    if "kinetic_energy_error_rel" in qoi_names:
        ke_computed: float | None = None
        if omega_dense is not None and psi_dense is not None:
            # KE = (1/2)∫ω·ψ dA for periodic domains (from ∇²ψ = -ω)
            ke_computed = float(
                0.5 * np.abs(np.sum(omega_dense * psi_dense) * h * h)
            )
        ke_exact = _taylor_green_kinetic_energy(T, nu)
        if ke_computed is not None and ke_exact > 0:
            qois["kinetic_energy_error_rel"] = float(
                abs(ke_computed - ke_exact) / ke_exact
            )

    # ── total_circulation_abs_error ──────────────────────────────────
    if "total_circulation_abs_error" in qoi_names and omega_dense is not None:
        # Analytic: Γ = ∫ω dA = 0 for periodic Taylor–Green on [0,1]²
        circ = float(np.sum(omega_dense) * h * h)
        qois["total_circulation_abs_error"] = abs(circ)

    # ═════════════════════════════════════════════════════════════════
    # POISSON SOLVER DIAGNOSTICS
    # CG criterion: ||r||/||b|| < tol  (RELATIVE).
    # Both absolute and relative residuals are reported so the reader
    # can verify convergence-flag consistency.
    # ═════════════════════════════════════════════════════════════════
    probes = getattr(result, "probes", {})

    # ── poisson_residual_abs_max ─────────────────────────────────────
    if "poisson_residual_abs_max" in qoi_names:
        res_sq = probes.get("poisson_residual_sq", [])
        if res_sq:
            qois["poisson_residual_abs_max"] = float(np.sqrt(max(res_sq)))
        else:
            qois["poisson_residual_abs_max"] = float("nan")

    # ── poisson_residual_rel_max ─────────────────────────────────────
    if "poisson_residual_rel_max" in qoi_names:
        rel_res = probes.get("poisson_relative_residual", [])
        if rel_res:
            qois["poisson_residual_rel_max"] = float(max(rel_res))
        else:
            qois["poisson_residual_rel_max"] = float("nan")

    # ── cg_iters ─────────────────────────────────────────────────────
    if "cg_iters" in qoi_names:
        cg = probes.get("poisson_cg_iters", [])
        if cg:
            qois["cg_iters"] = float(max(cg))
        else:
            qois["cg_iters"] = float("nan")

    # ── nan_inf_steps ────────────────────────────────────────────────
    if "nan_inf_steps" in qoi_names:
        nan_count = 0
        for name, gpu_t in result.fields.items():
            cpu_t = gpu_t.to_cpu()
            for core in cpu_t.cores:
                arr = core.numpy() if hasattr(core, "numpy") else np.asarray(core)
                if not np.all(np.isfinite(arr)):
                    nan_count += 1
                    break
        qois["nan_inf_steps"] = float(nan_count)

    return qois


def run_convergence_cli() -> None:
    """CLI entry point for ``python -m ontic.platform.vv.harness``.

    Runs a multi-resolution convergence study on the QTT GPU VM.
    Supports Taylor–Green vortex 2D with analytic QoI comparison.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="ontic.platform.vv.harness",
        description="QTT VM V&V Convergence Harness",
    )
    parser.add_argument(
        "--case", type=str, required=True,
        help="Case identifier (e.g. C210_TAYLOR_GREEN_VORTEX_2D)",
    )
    parser.add_argument(
        "--domain", type=str, required=True,
        help="Domain key: navier_stokes_2d",
    )
    parser.add_argument(
        "--n_bits", type=str, required=True,
        help="Comma-separated list of n_bits levels (e.g. 8,9,10)",
    )
    parser.add_argument(
        "--dt", type=float, required=True,
        help="Time step size",
    )
    parser.add_argument(
        "--n_steps", type=int, required=True,
        help="Number of time steps",
    )
    parser.add_argument(
        "--viscosity", type=float, default=0.01,
        help="Kinematic viscosity (default: 0.01)",
    )
    parser.add_argument(
        "--op_variant", type=str, default="ns2d_vorticity_v1",
        help="NS2D operator variant tag",
    )
    parser.add_argument(
        "--lap_variant", type=str, default="lap_v1",
        help="Laplacian MPO variant: lap_v1 (2nd) or lap_v2_high_order (4th)",
    )
    parser.add_argument(
        "--grad_variant", type=str, default="grad_v1",
        help="Gradient MPO variant: grad_v1 (2nd) or grad_v2_high_order (4th)",
    )
    parser.add_argument(
        "--poisson_solver", type=str, default="cg",
        help="Poisson solver method (default: cg)",
    )
    parser.add_argument(
        "--poisson_tol", type=float, default=1e-8,
        help="Poisson CG convergence tolerance",
    )
    parser.add_argument(
        "--poisson_max_iters", type=int, default=80,
        help="Poisson CG maximum iterations",
    )
    parser.add_argument(
        "--truncate_rel", type=float, default=1e-10,
        help="rSVD relative truncation tolerance for GPURankGovernor",
    )
    parser.add_argument(
        "--max_rank_policy", type=int, default=64,
        help="Maximum bond dimension for GPURankGovernor",
    )
    parser.add_argument(
        "--qoi", type=str, default="",
        help="Comma-separated QoI names to evaluate",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON scorecard (default: stdout summary only)",
    )

    args = parser.parse_args()

    # Parse n_bits list
    n_bits_list = [int(b.strip()) for b in args.n_bits.split(",")]
    qoi_names = [q.strip() for q in args.qoi.split(",") if q.strip()]

    if not qoi_names:
        qoi_names = [
            "omega_error_L2",
            "omega_error_L2_rel",
            "psi_error_L2",
            "psi_error_L2_rel",
            "enstrophy",
            "enstrophy_error_rel",
            "kinetic_energy_error_rel",
            "poisson_residual_abs_max",
            "poisson_residual_rel_max",
            "cg_iters",
            "total_circulation_abs_error",
            "nan_inf_steps",
        ]

    # Validate domain
    if args.domain != "navier_stokes_2d":
        print(f"ERROR: domain '{args.domain}' not yet supported by CLI harness",
              file=sys.stderr)
        sys.exit(1)

    # ── Header ───────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          QTT VM V&V Convergence Harness                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Case:           {args.case}")
    print(f"  Domain:         {args.domain}")
    print(f"  n_bits:         {n_bits_list}")
    print(f"  dt:             {args.dt:.6e}")
    print(f"  n_steps:        {args.n_steps}")
    print(f"  ν:              {args.viscosity}")
    print(f"  op_variant:     {args.op_variant}")
    print(f"  grad_variant:   {args.grad_variant}")
    print(f"  lap_variant:    {args.lap_variant}")
    print(f"  poisson_solver: {args.poisson_solver}")
    print(f"  poisson_tol:    {args.poisson_tol:.2e}")
    print(f"  poisson_iters:  {args.poisson_max_iters}")
    print(f"  truncate_rel:   {args.truncate_rel:.2e}")
    print(f"  max_rank:       {args.max_rank_policy}")
    print(f"  QoIs:           {qoi_names}")
    print()

    # ── Imports (lazy to avoid import overhead for --help) ───────────
    import torch

    from ontic.engine.vm.compilers.navier_stokes_2d import NavierStokes2DCompiler
    from ontic.engine.vm.gpu_runtime import GPURankGovernor, GPURuntime

    # ── Run across n_bits levels ─────────────────────────────────────
    all_results: list[dict[str, Any]] = []
    t_global_start = time.monotonic()

    for nb in n_bits_list:
        N = 2 ** nb
        print(f"── n_bits={nb}  (grid {N}×{N} = {N*N:,} DOF) ", "─" * 30)

        # Compile
        compiler = NavierStokes2DCompiler(
            n_bits=nb,
            n_steps=args.n_steps,
            viscosity=args.viscosity,
            dt=args.dt,
            grad_variant=args.grad_variant,
            lap_variant=args.lap_variant,
            op_variant=args.op_variant,
            poisson_tol=args.poisson_tol,
            poisson_max_iters=args.poisson_max_iters,
        )
        program = compiler.compile()
        n_ir = len(program.instructions)
        print(f"   Compiled: {n_ir} IR instructions")

        # Runtime
        governor = GPURankGovernor(
            max_rank=args.max_rank_policy,
            rel_tol=args.truncate_rel,
            adaptive=True,
            base_rank=args.max_rank_policy,
        )
        runtime = GPURuntime(governor=governor)

        # Execute
        torch.cuda.synchronize()
        t0 = time.monotonic()
        exec_result = runtime.execute(program)
        torch.cuda.synchronize()
        wall = time.monotonic() - t0

        if not exec_result.success:
            print(f"   FAILED: {exec_result.error}")
            all_results.append({
                "n_bits": nb, "N": N, "wall_s": wall,
                "status": "failed", "error": exec_result.error,
                "qois": {},
            })
            continue

        # Telemetry summary
        telem = exec_result.telemetry
        chi_max = getattr(telem, "chi_max", 0)
        if chi_max == 0:
            # Fall back to governor peak if telemetry collector hadn't
            # updated chi_max (first-timer Triton compile delay edge case)
            chi_max = governor.peak_rank or 1
        compression = (N * N) / max(chi_max, 1)
        inv_err = getattr(telem, "invariant_error", 0.0)
        print(f"   Wall:        {wall:.3f}s")
        print(f"   χ_max:       {chi_max}")
        print(f"   Compression: {compression:.1f}×")
        print(f"   |ΔΓ/Γ₀|:    {abs(inv_err):.4e}")

        # Compute QoIs
        qois = _compute_qoi_from_result(
            exec_result, nb, args.viscosity,
            args.dt, args.n_steps, qoi_names,
        )

        for name in qoi_names:
            val = qois.get(name, float("nan"))
            print(f"   {name:40s} = {val:.6e}")

        # ── Hard gate: Poisson convergence ──────────────────────────
        # Check per-step convergence flags from the runtime probe data.
        # The CG solver reports converged=True/False per solve; any
        # unconverged step contaminates the tier and we flag it.
        probes = getattr(exec_result, "probes", {})
        conv_flags = probes.get("poisson_converged", [])
        rel_resids = probes.get("poisson_relative_residual", [])

        poisson_failed = False
        poisson_fail_reason = ""

        if conv_flags:
            n_unconverged = sum(1 for f in conv_flags if f < 0.5)
            if n_unconverged > 0:
                cg_max = qois.get("cg_iters", 0.0)
                rel_max = max(rel_resids) if rel_resids else float("nan")
                poisson_failed = True
                poisson_fail_reason = (
                    f"{n_unconverged}/{len(conv_flags)} steps unconverged, "
                    f"max_cg_iters={int(cg_max)}, "
                    f"max_relative_residual={rel_max:.2e}"
                )
        else:
            # No Poisson probe data at all — no CG ran (edge case)
            pass

        tier_status = "succeeded"
        if poisson_failed:
            tier_status = "poisson_unconverged"
            print(f"   ⚠ POISSON GATE FAIL: {poisson_fail_reason}")
            print(f"   ⚠ QoIs for this tier are NOT trustworthy for convergence claims.")

        all_results.append({
            "n_bits": nb,
            "N": N,
            "wall_s": wall,
            "status": tier_status,
            "chi_max": chi_max,
            "compression": compression,
            "invariant_error": float(inv_err),
            "qois": qois,
            "poisson_converged": not poisson_failed,
            "poisson_fail_reason": poisson_fail_reason,
        })
        print()

    t_global = time.monotonic() - t_global_start

    # ── Convergence Table ────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                        CONVERGENCE SUMMARY                                 ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")

    # Header row
    hdr_qois = [q[:20] for q in qoi_names]
    hdr = f"  {'n_bits':>6}  {'N':>6}  {'wall_s':>8}"
    for q in hdr_qois:
        hdr += f"  {q:>20}"
    print(hdr)
    print("  " + "─" * (20 + 8 + 6 + 6 + len(hdr_qois) * 22))

    for r in all_results:
        status_tag = ""
        if r.get("status") == "poisson_unconverged":
            status_tag = "  ⚠POISSON"
        elif r.get("status") == "failed":
            status_tag = "  ✗FAILED"
        row = f"  {r['n_bits']:>6}  {r['N']:>6}  {r['wall_s']:>8.3f}"
        for q in qoi_names:
            val = r["qois"].get(q, float("nan"))
            row += f"  {val:>20.6e}"
        row += status_tag
        print(row)

    # Compute observed convergence orders (log-log slope between successive levels)
    # ONLY use tiers where Poisson actually converged.
    converged_results = [
        r for r in all_results if r.get("poisson_converged", True)
    ]
    unconverged_results = [
        r for r in all_results if not r.get("poisson_converged", True)
    ]
    if unconverged_results:
        print()
        print("  ⚠ EXCLUDED FROM CONVERGENCE (Poisson unconverged):")
        for r in unconverged_results:
            print(f"    n_bits={r['n_bits']}  N={r['N']}  reason: {r.get('poisson_fail_reason', '?')}")

    if len(converged_results) >= 2:
        print()
        print("  Observed convergence orders (log₂ ratio between successive levels):")
        for q in qoi_names:
            vals = [
                (r["N"], r["qois"].get(q, float("nan")))
                for r in converged_results
            ]
            orders = []
            for i in range(1, len(vals)):
                N_prev, e_prev = vals[i - 1]
                N_curr, e_curr = vals[i]
                if (
                    np.isfinite(e_prev)
                    and np.isfinite(e_curr)
                    and e_prev > 0
                    and e_curr > 0
                    and N_curr != N_prev
                ):
                    h_ratio = N_prev / N_curr  # h = 1/N, so h_prev/h_curr = N_curr/N_prev
                    order = np.log2(e_prev / e_curr) / np.log2(N_curr / N_prev)
                    orders.append(order)
            if orders:
                avg_order = np.mean(orders)
                order_str = ", ".join(f"{o:.3f}" for o in orders)
                print(f"    {q:40s}  orders=[{order_str}]  avg={avg_order:.3f}")
            else:
                print(f"    {q:40s}  (insufficient data)")

    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"  Total wall time: {t_global:.3f}s")
    print()

    # ── JSON Scorecard ───────────────────────────────────────────────
    scorecard: dict[str, Any] = {
        "schema_version": "3.0",
        "harness": "ontic.platform.vv.harness",
        "case": args.case,
        "domain": args.domain,
        "config": {
            "n_bits_levels": n_bits_list,
            "dt": args.dt,
            "n_steps": args.n_steps,
            "viscosity": args.viscosity,
            "op_variant": args.op_variant,
            "grad_variant": args.grad_variant,
            "lap_variant": args.lap_variant,
            "poisson_solver": args.poisson_solver,
            "poisson_tol": args.poisson_tol,
            "poisson_max_iters": args.poisson_max_iters,
            "poisson_convergence_criterion": "relative: ||r||/||b|| < tol",
            "truncate_rel": args.truncate_rel,
            "max_rank_policy": args.max_rank_policy,
        },
        "qoi_definitions": {
            "omega_error_L2": "||ω_h - ω_exact||_L²([0,1]²) with h² midpoint quadrature",
            "omega_error_L2_rel": "||ω_h - ω_exact||_L² / ||ω_exact||_L² (dimensionless)",
            "psi_error_L2": "||ψ_h - ψ_exact||_L²([0,1]²) with h² midpoint quadrature",
            "psi_error_L2_rel": "||ψ_h - ψ_exact||_L² / ||ψ_exact||_L² (dimensionless)",
            "enstrophy": "(1/2)∫ω² dA — grid-independent physical quantity",
            "enstrophy_error_rel": "|Z_h - Z_exact|/Z_exact",
            "kinetic_energy_error_rel": "|KE_h - KE_exact|/KE_exact, KE=(1/2)∫ω·ψ dA",
            "poisson_residual_abs_max": "max over steps of sqrt(||r||²) — absolute",
            "poisson_residual_rel_max": "max over steps of ||r||/||b|| — relative (convergence metric)",
            "cg_iters": "max CG iterations over all timesteps",
            "total_circulation_abs_error": "|∫ω dA| (analytic = 0)",
            "nan_inf_steps": "count of fields with NaN/Inf in QTT cores",
        },
        "qoi_names": qoi_names,
        "levels": all_results,
        "total_wall_s": t_global,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(scorecard, indent=2, default=str))
        print(f"  Scorecard written to {out_path}")
    else:
        # Print JSON to stdout as well
        print(json.dumps(scorecard, indent=2, default=str))


if __name__ == "__main__":
    run_convergence_cli()
