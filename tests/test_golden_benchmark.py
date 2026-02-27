"""Golden benchmark regression suite — G8.1–G8.4.

G8.1: 1 job per domain defined — 7 canonical jobs, one per physics domain.
G8.2: Expected tolerances documented — conservation error bands per domain.
G8.3: Conservation baselines recorded — ``benchmarks/golden_baselines.json``.
G8.4: Automated regression test — this file.

Runs all 7 domains through the full pipeline (compile → execute →
sanitize → validate → attest → verify) and checks:

    1. Execution succeeds (no RuntimeError, no divergence).
    2. Conservation relative error within the documented band.
    3. Certificate issues and verifies.
    4. Wall time within documented band.

Baselines are loaded from ``benchmarks/golden_baselines.json``.
"""

from __future__ import annotations

import json
import pathlib
import signal
import time
from typing import Any

import pytest

from hypertensor.core.certificates import issue_certificate, verify_certificate
from hypertensor.core.evidence import generate_claims, generate_validation_report
from hypertensor.core.executor import ExecutionConfig, execute
from hypertensor.core.hasher import content_hash
from hypertensor.core.sanitizer import sanitize_result

# ── Load baselines ──────────────────────────────────────────────────

_BASELINE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "experiments" / "benchmarks" / "benchmarks" / "golden_baselines.json"
)

with open(_BASELINE_PATH) as _f:
    _BASELINES: dict[str, Any] = json.load(_f)

_DOMAINS: dict[str, dict[str, Any]] = _BASELINES["domains"]

# Per-job hard timeout (seconds) to prevent CI hangs.
_JOB_TIMEOUT = 120


# ── Timeout helper ──────────────────────────────────────────────────


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Golden benchmark job exceeded hard timeout")


# ── Domain parametrization ──────────────────────────────────────────


def _golden_ids() -> list[str]:
    return list(_DOMAINS.keys())


def _golden_params() -> list[tuple[str, dict[str, Any]]]:
    return [(k, v) for k, v in _DOMAINS.items()]


# ════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════


class TestGoldenBenchmark:
    """Full-pipeline regression for every physics domain."""

    @pytest.fixture(params=_golden_params(), ids=_golden_ids())
    def golden(self, request: pytest.FixtureRequest) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Run the golden job and yield (domain, spec, results).

        This fixture runs once per domain.  It executes the full
        pipeline and caches the results for all assertions.
        """
        domain, spec = request.param
        config = ExecutionConfig(
            domain=domain,
            n_bits=spec["n_bits"],
            n_steps=spec["n_steps"],
            max_rank=spec["max_rank"],
        )

        # Arm SIGALRM for hard timeout
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_JOB_TIMEOUT)
        t0 = time.monotonic()
        try:
            raw_result = execute(config)
            elapsed = time.monotonic() - t0
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        assert raw_result.success, (
            f"[{domain}] Execution failed: {getattr(raw_result, 'error', 'unknown')}"
        )

        sanitized = sanitize_result(raw_result, domain, include_fields=False)
        validation = generate_validation_report(sanitized, domain)
        claims = generate_claims(sanitized, domain)
        result_hash = content_hash(sanitized)
        input_spec = {"domain": domain, **{k: v for k, v in spec.items()}}
        certificate = issue_certificate(
            job_id=f"golden-{domain}",
            claims=claims,
            input_manifest_hash=content_hash(input_spec),
            result_hash=result_hash,
            config_hash=content_hash(input_spec),
            runtime_version="3.1.0",
            device_class="cpu",
        )

        results = {
            "raw_result": raw_result,
            "sanitized": sanitized,
            "validation": validation,
            "claims": claims,
            "certificate": certificate,
            "elapsed_s": elapsed,
        }
        return domain, spec, results

    # ── G8.1 — execution succeeds ───────────────────────────────────

    def test_execution_succeeds(self, golden: tuple[str, dict, dict]) -> None:
        """Every golden job must complete without error."""
        domain, spec, results = golden
        assert results["raw_result"].success, f"[{domain}] Execution failed"

    # ── G8.2 — conservation within tolerance ────────────────────────

    def test_conservation_within_band(self, golden: tuple[str, dict, dict]) -> None:
        """Conservation relative error must stay within the documented band."""
        domain, spec, results = golden
        cons = results["sanitized"].get("conservation")
        if cons is None:
            pytest.skip(f"[{domain}] No conservation data reported")

        max_err = spec["conservation_error_max"]
        actual_err = cons["relative_error"]
        assert actual_err <= max_err, (
            f"[{domain}] Conservation error {actual_err:.2e} exceeds "
            f"maximum {max_err:.2e} for {cons['quantity']}"
        )

    # ── G8.3 — certificate verifies ────────────────────────────────

    def test_certificate_verifies(self, golden: tuple[str, dict, dict]) -> None:
        """Certificate must verify with the current signing key."""
        domain, spec, results = golden
        assert verify_certificate(results["certificate"]), (
            f"[{domain}] Certificate verification failed"
        )

    # ── G8.4 — wall time within band ───────────────────────────────

    def test_wall_time_within_band(self, golden: tuple[str, dict, dict]) -> None:
        """Wall time must be within the documented maximum."""
        domain, spec, results = golden
        perf = results["sanitized"]["performance"]
        max_wall = spec["wall_time_max_s"]
        actual_wall = perf["wall_time_s"]
        assert actual_wall <= max_wall, (
            f"[{domain}] Wall time {actual_wall:.2f}s exceeds "
            f"maximum {max_wall:.1f}s"
        )

    # ── Pipeline completeness ───────────────────────────────────────

    def test_claims_generated(self, golden: tuple[str, dict, dict]) -> None:
        """At least one claim must be generated per domain."""
        domain, spec, results = golden
        assert len(results["claims"]) >= 1, (
            f"[{domain}] Expected >= 1 claims, got {len(results['claims'])}"
        )

    def test_certificate_has_required_fields(self, golden: tuple[str, dict, dict]) -> None:
        """Certificate contains all mandatory fields."""
        domain, spec, results = golden
        cert = results["certificate"]
        required_fields = {"job_id", "claims", "result_hash", "signature", "issued_at"}
        missing = required_fields - set(cert.keys())
        assert not missing, (
            f"[{domain}] Certificate missing fields: {missing}"
        )
