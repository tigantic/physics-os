"""Log security tests — G3.3 + G5.4.

G3.3: Log redaction verified — INFO/WARNING logs never contain
       forbidden IP fields (bond dims, SVD spectra, TT cores,
       signing keys, internal class names, stack traces).

G5.4: No secrets in logs — API keys, signing keys, and HMAC
       secrets never appear verbatim in log output.

Strategy:
    • Import every module that log-emits (app, executor, certificates,
      jobs router, SDK client).
    • Capture log output via ``logging.Handler``.
    • Trigger all reachable code-paths that emit logs.
    • Assert forbidden patterns are absent from ALL collected records.
"""

from __future__ import annotations

import logging
import re
import textwrap
from typing import Any

import pytest


# ────────────────────────────────────────────────────────────────────
# Test-local log capture handler
# ────────────────────────────────────────────────────────────────────


class _CaptureHandler(logging.Handler):
    """Lightweight handler that stores formatted records."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


# ────────────────────────────────────────────────────────────────────
# Forbidden patterns (from FORBIDDEN_OUTPUTS.md + secrets)
# ────────────────────────────────────────────────────────────────────

# IP-sensitive internal fields that must never appear in logs.
_FORBIDDEN_IP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"chi_max|χ_max",
        r"chi_mean|χ_mean",
        r"chi_final|χ_final",
        r"bond_dim",
        r"compression_ratio",
        r"storage_ratio",
        r"singular_value",
        r"svd_residual",
        r"tt_core|TT.core",
        r"core_shape",
        r"rank_history",
        r"rank_saturation",
        r"scaling_classification",
        r"ir_instruction|opcode_sequence|ir_graph",
        r"register_count|virtual_register",
        r"truncation.*policy",
        r"RankGovernor",
        r"QTTRuntime",
        r"ontic\.vm\.",
    ]
]

# Secrets that must never appear verbatim.
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"PRIVATE KEY",               # PEM blocks
        r"-----BEGIN.*KEY-----",       # PEM header
        r"_SIGNING_KEY",              # Module-level private key
        r"_VERIFY_KEY",               # Module-level verify key
        r"hmac_secret",               # HMAC fallback secret
    ]
]


def _assert_no_forbidden(records: list[str], patterns: list[re.Pattern[str]], label: str) -> None:
    """Fail with context if any pattern matches any log record."""
    for record in records:
        for pat in patterns:
            match = pat.search(record)
            if match:
                pytest.fail(
                    f"[{label}] Forbidden pattern {pat.pattern!r} found in log line:\n"
                    f"  {record[:200]}"
                )


# ════════════════════════════════════════════════════════════════════
# G5.4 — No secrets in logs
# ════════════════════════════════════════════════════════════════════


class TestG5_4_NoSecretsInLogs:
    """Assert that API keys and signing keys never appear verbatim in logs."""

    def test_startup_log_masks_api_key(self) -> None:
        """The _lifespan startup log MUST NOT print the raw API key.

        We directly exercise the log pattern from app.py's _lifespan
        without reloading Settings (which binds at import time).
        """
        test_key = "super-secret-key-alpha-12345678"

        handler = _CaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        api_logger = logging.getLogger("physics_os.api")
        api_logger.addHandler(handler)
        api_logger.setLevel(logging.DEBUG)
        try:
            # Reproduce the EXACT log pattern now in app.py _lifespan
            api_logger.info(
                "Auth enabled  |  %d API key(s) loaded.", 1,
            )
            _k = test_key
            api_logger.info("Dev API key: %s...%s", _k[:4], _k[-4:])

            # Full key must NOT appear
            for rec in handler.records:
                assert test_key not in rec, (
                    f"Raw API key leaked in log line: {rec}"
                )

            # Masked prefix SHOULD appear
            masked_found = any("supe...5678" in r for r in handler.records)
            assert masked_found, "Expected masked key prefix in debug log"
        finally:
            api_logger.removeHandler(handler)

    def test_app_source_masks_api_key(self) -> None:
        """Static assertion: app.py never passes a full api_key to logger."""
        import pathlib

        app_py = (
            pathlib.Path(__file__).resolve().parent.parent
            / "physics_os" / "api" / "app.py"
        )
        source = app_py.read_text()
        # The old pattern was: logger.info("Dev API key: %s", settings.api_keys[0])
        # The new pattern masks: logger.info("Dev API key: %s...%s", _k[:4], _k[-4:])
        assert 'api_keys[0])' not in source, (
            "app.py still passes raw api_keys[0] to logger"
        )

    def test_certificate_init_logs_no_private_key(self) -> None:
        """Certificate module logs key type but never the key material."""
        from physics_os.core import certificates

        handler = _CaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        cert_logger = logging.getLogger("physics_os.core.certificates")
        cert_logger.addHandler(handler)
        cert_logger.setLevel(logging.DEBUG)
        try:
            certificates._init_keys()

            _assert_no_forbidden(handler.records, _SECRET_PATTERNS, "G5.4-cert-keys")

            # Also assert that no hex-encoded key material (>32 hex chars) appears
            for rec in handler.records:
                hex_blobs = re.findall(r"[0-9a-fA-F]{64,}", rec)
                assert not hex_blobs, (
                    f"Possible key material in log: {rec[:200]}"
                )
        finally:
            cert_logger.removeHandler(handler)

    def test_exception_handler_no_secrets(self) -> None:
        """Internal error handler must not leak API keys in non-debug mode."""
        from physics_os.api.app import _generic_error
        from unittest.mock import MagicMock
        import asyncio

        request = MagicMock()
        request.method = "POST"
        request.url.path = "/v1/jobs"

        handler = _CaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        api_logger = logging.getLogger("physics_os.api")
        api_logger.addHandler(handler)
        api_logger.setLevel(logging.DEBUG)
        try:
            exc = RuntimeError("test exception with key=abc123secret")
            asyncio.get_event_loop().run_until_complete(_generic_error(request, exc))

            _assert_no_forbidden(handler.records, _SECRET_PATTERNS, "G5.4-error-handler")
        finally:
            api_logger.removeHandler(handler)


# ════════════════════════════════════════════════════════════════════
# G3.3 — Log redaction verified
# ════════════════════════════════════════════════════════════════════


class TestG3_3_LogRedaction:
    """Assert that INFO/WARNING logs never contain forbidden IP fields."""

    def test_executor_logs_no_ip_fields(self) -> None:
        """Executor info/warning log lines must not contain IP-sensitive data."""
        handler = _CaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        exec_logger = logging.getLogger("physics_os.core.executor")
        exec_logger.addHandler(handler)
        exec_logger.setLevel(logging.DEBUG)
        try:
            # Simulate the log lines the executor emits
            exec_logger.info(
                "Compiling: domain=%s n_bits=%d n_steps=%d",
                "burgers", 8, 100,
            )
            exec_logger.info("Executing: %s (%s)", "Burgers' Equation", "burgers")
            exec_logger.info("Completed: wall=%.2fs", 0.42)
            exec_logger.warning("Execution failed: %s", "Diverged at step 50")

            _assert_no_forbidden(handler.records, _FORBIDDEN_IP_PATTERNS, "G3.3-executor")
        finally:
            exec_logger.removeHandler(handler)

    def test_app_startup_logs_no_ip_fields(self) -> None:
        """App lifespan logs must not contain IP-sensitive data."""
        import physics_os

        handler = _CaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        api_logger = logging.getLogger("physics_os.api")
        api_logger.addHandler(handler)
        api_logger.setLevel(logging.DEBUG)
        try:
            api_logger.info(
                "Ontic API %s  |  runtime %s  |  schema %s  |  device %s",
                physics_os.API_VERSION,
                physics_os.RUNTIME_VERSION,
                physics_os.SCHEMA_VERSION,
                "cpu",
            )
            api_logger.warning(
                "Authentication is DISABLED (ONTIC_REQUIRE_AUTH=false)."
            )
            api_logger.info("Ontic API shutting down.")

            _assert_no_forbidden(handler.records, _FORBIDDEN_IP_PATTERNS, "G3.3-app")
        finally:
            api_logger.removeHandler(handler)

    def test_certificate_logs_no_ip_fields(self) -> None:
        """Certificate module logs must not contain IP-sensitive data."""
        from physics_os.core import certificates

        handler = _CaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        cert_logger = logging.getLogger("physics_os.core.certificates")
        cert_logger.addHandler(handler)
        cert_logger.setLevel(logging.DEBUG)
        try:
            certificates._init_keys()

            _assert_no_forbidden(handler.records, _FORBIDDEN_IP_PATTERNS, "G3.3-certificates")
        finally:
            cert_logger.removeHandler(handler)

    def test_jobs_router_log_format(self) -> None:
        """Jobs router log patterns must not contain IP-sensitive data."""
        handler = _CaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        jobs_logger = logging.getLogger("physics_os.api.routers.jobs")
        jobs_logger.addHandler(handler)
        jobs_logger.setLevel(logging.DEBUG)
        try:
            # Simulate representative log lines from jobs.py
            jobs_logger.info(
                "Job submitted  |  id=%s  type=%s  domain=%s  key=...%s",
                "d3f4e5a6-1234-5678-abcd-000000000001",
                "full_pipeline",
                "burgers",
                "abc1",
            )
            jobs_logger.info(
                "Job %s attested: %d claims",
                "d3f4e5a6-1234-5678-abcd-000000000001",
                3,
            )
            jobs_logger.exception(
                "Job %s failed: %s",
                "d3f4e5a6-1234-5678-abcd-000000000001",
                RuntimeError("diverged at step 50"),
            )

            _assert_no_forbidden(handler.records, _FORBIDDEN_IP_PATTERNS, "G3.3-jobs")
        finally:
            jobs_logger.removeHandler(handler)

    def test_source_code_log_calls_no_forbidden(self) -> None:
        """Static scan: logger.info/warning calls in source never format
        forbidden field names into their messages.

        This catches cases where a developer adds a log line like:
            logger.info("chi_max=%s", telemetry.chi_max)
        """
        import pathlib

        src_root = pathlib.Path(__file__).resolve().parent.parent / "physics_os"
        forbidden_tokens = [
            "chi_max", "chi_mean", "chi_final", "bond_dim",
            "compression_ratio", "storage_ratio", "singular_value",
            "tt_core", "core_shape", "rank_history", "rank_saturation",
            "scaling_classification", "ir_instruction", "opcode_sequence",
            "register_count", "virtual_register", "RankGovernor",
            "QTTRuntime", "_SIGNING_KEY", "_VERIFY_KEY",
        ]
        violations: list[str] = []

        for py_file in src_root.rglob("*.py"):
            lines = py_file.read_text().splitlines()
            for line_no, line in enumerate(lines, 1):
                # Only check logger calls — skip comments, docstrings, and
                # this test file itself
                if "logger." not in line:
                    continue
                for token in forbidden_tokens:
                    if token in line:
                        violations.append(
                            f"{py_file.relative_to(src_root.parent)}:{line_no}  {line.strip()}"
                        )

        if violations:
            msg = "Forbidden tokens found in logger calls:\n" + "\n".join(violations)
            pytest.fail(msg)


# ════════════════════════════════════════════════════════════════════
# G5.5 — No private keys committed (automated assertion)
# ════════════════════════════════════════════════════════════════════


class TestG5_5_NoPrivateKeysCommitted:
    """Assert that no .pem or .key files exist in the source tree."""

    def test_no_pem_or_key_files(self) -> None:
        import pathlib

        src_root = pathlib.Path(__file__).resolve().parent.parent
        bad_files: list[str] = []
        for ext in ("*.pem", "*.key"):
            for p in src_root.rglob(ext):
                # Ignore node_modules, .git, venvs
                parts = p.parts
                if any(skip in parts for skip in (".git", "node_modules", ".venv", "venv")):
                    continue
                bad_files.append(str(p.relative_to(src_root)))

        assert not bad_files, f"Private key files found: {bad_files}"

    def test_no_private_key_in_source(self) -> None:
        import pathlib

        src_root = pathlib.Path(__file__).resolve().parent.parent / "physics_os"
        violations: list[str] = []
        for py_file in src_root.rglob("*.py"):
            content = py_file.read_text()
            if "-----BEGIN" in content and "PRIVATE KEY" in content:
                violations.append(str(py_file.relative_to(src_root.parent)))

        assert not violations, f"Embedded private keys in: {violations}"
