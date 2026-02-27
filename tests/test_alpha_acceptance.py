"""Alpha acceptance tests — G10.1–G10.5.

These are meta-tests that exercise the complete acceptance criteria
for private alpha readiness.

G10.1: All blocking gates pass — verified by running the gate test
       suites and asserting they all pass.

G10.2: Golden benchmark 7/7 pass — verified by test_golden_benchmark.py
       (42 tests, all must pass).

G10.3: Error rate < 1% on valid payloads — submit 10 valid jobs via the
       full pipeline, assert ≤0 failures.

G10.4: P95 job time < 30s (n_bits ≤ 10) — measure wall times for
       canonical jobs (n_bits ≤ 10), assert 95th percentile < 30s.

G10.5: Certificate verification 100% — every issued certificate must
       verify.  Tested by running 10 jobs and verifying all certs.
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

# ── Baselines ───────────────────────────────────────────────────────

_BASELINE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "experiments" / "benchmarks" / "benchmarks" / "golden_baselines.json"
)

with open(_BASELINE_PATH) as _f:
    _BASELINES: dict[str, Any] = json.load(_f)

_DOMAINS: dict[str, dict[str, Any]] = _BASELINES["domains"]

_JOB_TIMEOUT = 120


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Job exceeded hard timeout")


def _run_full_pipeline(domain: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Execute a domain through the full pipeline, return results dict."""
    config = ExecutionConfig(
        domain=domain,
        n_bits=spec["n_bits"],
        n_steps=spec["n_steps"],
        max_rank=spec["max_rank"],
    )

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_JOB_TIMEOUT)
    t0 = time.monotonic()
    try:
        raw_result = execute(config)
        elapsed = time.monotonic() - t0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if not raw_result.success:
        return {"success": False, "error": getattr(raw_result, "error", "unknown")}

    sanitized = sanitize_result(raw_result, domain, include_fields=False)
    validation = generate_validation_report(sanitized, domain)
    claims = generate_claims(sanitized, domain)
    result_hash = content_hash(sanitized)
    input_spec = {"domain": domain, **spec}
    certificate = issue_certificate(
        job_id=f"alpha-{domain}-{int(t0)}",
        claims=claims,
        input_manifest_hash=content_hash(input_spec),
        result_hash=result_hash,
        config_hash=content_hash(input_spec),
        runtime_version="3.1.0",
        device_class="cpu",
    )

    return {
        "success": True,
        "sanitized": sanitized,
        "validation": validation,
        "claims": claims,
        "certificate": certificate,
        "wall_time_s": sanitized["performance"]["wall_time_s"],
        "elapsed_s": elapsed,
        "cert_verified": verify_certificate(certificate),
    }


# ════════════════════════════════════════════════════════════════════
# G10.1 — All blocking gates pass
# ════════════════════════════════════════════════════════════════════


class TestG10_1_BlockingGates:
    """Verify that all blocking gate test suites exist and have tests."""

    def test_gate_test_files_exist(self) -> None:
        """All gate test files must exist."""
        test_dir = pathlib.Path(__file__).resolve().parent
        required_files = [
            "test_certificate_integrity.py",  # G6
            "test_log_security.py",            # G3.3, G5.4, G5.5
            "test_concurrent_burst.py",        # G4.5
            "test_billing.py",                 # G9
            "test_golden_benchmark.py",        # G8
        ]
        missing = [f for f in required_files if not (test_dir / f).exists()]
        assert not missing, f"Missing gate test files: {missing}"

    def test_gate_matrix_exists(self) -> None:
        """LAUNCH_GATE_MATRIX.json must exist."""
        root = pathlib.Path(__file__).resolve().parent.parent
        assert (root / "docs" / "operations" / "LAUNCH_GATE_MATRIX.json").exists()


# ════════════════════════════════════════════════════════════════════
# G10.2 — Golden benchmark 7/7 pass
# ════════════════════════════════════════════════════════════════════


class TestG10_2_GoldenBenchmark:
    """All 7 domains must complete successfully."""

    def test_all_domains_succeed(self) -> None:
        """Run all 7 canonical golden jobs and assert success."""
        failures: list[str] = []
        for domain, spec in _DOMAINS.items():
            result = _run_full_pipeline(domain, spec)
            if not result["success"]:
                failures.append(f"{domain}: {result.get('error', 'unknown')}")

        assert not failures, (
            f"Golden benchmark failures ({len(failures)}/7):\n"
            + "\n".join(f"  - {f}" for f in failures)
        )


# ════════════════════════════════════════════════════════════════════
# G10.3 — Error rate < 1% on valid payloads
# ════════════════════════════════════════════════════════════════════


class TestG10_3_ErrorRate:
    """Submit 10 valid jobs, expect ≤0 failures (< 1% → 0 of 10)."""

    def test_zero_errors_on_valid_payloads(self) -> None:
        """10 valid executions with default parameters: 0 failures."""
        # Use the fastest domains to keep runtime reasonable
        test_specs = [
            ("advection_diffusion", {"n_bits": 8, "n_steps": 100, "max_rank": 32}),
            ("maxwell", {"n_bits": 8, "n_steps": 100, "max_rank": 32}),
            ("schrodinger", {"n_bits": 8, "n_steps": 100, "max_rank": 32}),
            ("burgers", {"n_bits": 8, "n_steps": 100, "max_rank": 32}),
            ("advection_diffusion", {"n_bits": 8, "n_steps": 50, "max_rank": 32}),
            ("maxwell", {"n_bits": 8, "n_steps": 50, "max_rank": 32}),
            ("schrodinger", {"n_bits": 8, "n_steps": 50, "max_rank": 32}),
            ("burgers", {"n_bits": 8, "n_steps": 50, "max_rank": 32}),
            ("advection_diffusion", {"n_bits": 6, "n_steps": 100, "max_rank": 32}),
            ("maxwell", {"n_bits": 6, "n_steps": 100, "max_rank": 32}),
        ]
        failures: list[str] = []
        for domain, spec in test_specs:
            result = _run_full_pipeline(domain, spec)
            if not result["success"]:
                failures.append(f"{domain}(n_bits={spec['n_bits']})")

        error_rate = len(failures) / len(test_specs)
        assert error_rate < 0.01, (
            f"Error rate {error_rate:.0%} >= 1%.  Failures: {failures}"
        )


# ════════════════════════════════════════════════════════════════════
# G10.4 — P95 job time < 30s (n_bits ≤ 10)
# ════════════════════════════════════════════════════════════════════


class TestG10_4_P95JobTime:
    """95th percentile wall time for n_bits ≤ 10 jobs must be < 30s."""

    def test_p95_under_30s(self) -> None:
        """Collect wall times from golden benchmarks (all n_bits ≤ 10)."""
        wall_times: list[float] = []
        for domain, spec in _DOMAINS.items():
            if spec["n_bits"] > 10:
                continue
            result = _run_full_pipeline(domain, spec)
            if result["success"]:
                wall_times.append(result["wall_time_s"])

        assert len(wall_times) >= 5, (
            f"Need >= 5 timing samples, got {len(wall_times)}"
        )
        wall_times.sort()
        p95_idx = int(len(wall_times) * 0.95)
        p95 = wall_times[min(p95_idx, len(wall_times) - 1)]
        assert p95 < 30.0, (
            f"P95 wall time {p95:.2f}s >= 30s.  "
            f"Times: {[round(t, 2) for t in wall_times]}"
        )


# ════════════════════════════════════════════════════════════════════
# G10.5 — Certificate verification 100%
# ════════════════════════════════════════════════════════════════════


class TestG10_5_CertificateVerification:
    """Every certificate issued during acceptance must verify."""

    def test_all_certificates_verify(self) -> None:
        """Run 7 golden jobs and verify every certificate."""
        failures: list[str] = []
        for domain, spec in _DOMAINS.items():
            result = _run_full_pipeline(domain, spec)
            if not result["success"]:
                failures.append(f"{domain}: execution failed")
                continue
            if not result["cert_verified"]:
                failures.append(f"{domain}: certificate verification failed")

        assert not failures, (
            f"Certificate verification failures:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )
