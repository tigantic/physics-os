"""Tests for auth middleware — products.facial_plastics.ui.auth."""

from __future__ import annotations

import json
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pytest

from products.facial_plastics.ui.auth import (
    APIKeyRecord,
    AuthMiddleware,
    KeyStore,
    RateLimitMiddleware,
)


# ── Helpers ───────────────────────────────────────────────────────


class _CapturedResponse:
    """Capture start_response calls."""

    def __init__(self) -> None:
        self.status: str = ""
        self.headers: List[Tuple[str, str]] = []

    def __call__(
        self,
        status: str,
        headers: List[Tuple[str, str]],
        exc_info: Any = None,
    ) -> None:
        self.status = status
        self.headers = headers


def _make_environ(
    method: str = "GET",
    path: str = "/",
    api_key: str = "",
    bearer: str = "",
    remote_addr: str = "127.0.0.1",
) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": "",
        "CONTENT_LENGTH": "0",
        "wsgi.input": BytesIO(b""),
        "REMOTE_ADDR": remote_addr,
    }
    if api_key:
        env["HTTP_X_API_KEY"] = api_key
    if bearer:
        env["HTTP_AUTHORIZATION"] = f"Bearer {bearer}"
    return env


def _echo_app(
    environ: Dict[str, Any],
    start_response: Any,
) -> Iterable[bytes]:
    """Trivial WSGI app that echoes request info."""
    data = {
        "path": environ.get("PATH_INFO", "/"),
        "tenant_id": environ.get("fp.tenant_id", ""),
        "role": environ.get("fp.role", ""),
    }
    body = json.dumps(data).encode()
    start_response("200 OK", [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(body))),
    ])
    return [body]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KeyStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKeyStore:
    def test_generate_and_validate(self) -> None:
        store = KeyStore()
        plaintext, record = store.generate_key("tenant-1", "surgeon")
        assert plaintext.startswith("fp_")
        assert record.tenant_id == "tenant-1"
        assert record.role == "surgeon"
        assert record.is_active

        # Validate
        result = store.validate(plaintext)
        assert result is not None
        assert result.tenant_id == "tenant-1"

    def test_invalid_key_returns_none(self) -> None:
        store = KeyStore()
        assert store.validate("fp_bogus_key") is None

    def test_revoke(self) -> None:
        store = KeyStore()
        plaintext, record = store.generate_key("t1", "admin")
        assert store.validate(plaintext) is not None

        store.revoke(record.key_hash)
        assert store.validate(plaintext) is None

    def test_persistence(self, tmp_path: Path) -> None:
        key_file = tmp_path / "keys.json"

        # Generate
        store1 = KeyStore(key_file=key_file)
        plaintext, _ = store1.generate_key("t1", "surgeon", "test key")
        assert key_file.exists()

        # Reload
        store2 = KeyStore(key_file=key_file)
        result = store2.validate(plaintext)
        assert result is not None
        assert result.tenant_id == "t1"

    def test_list_keys(self) -> None:
        store = KeyStore()
        store.generate_key("t1", "surgeon")
        store.generate_key("t2", "researcher")
        keys = store.list_keys()
        assert len(keys) == 2

    def test_revoke_nonexistent_returns_false(self) -> None:
        store = KeyStore()
        assert store.revoke("nonexistent_hash") is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AuthMiddleware
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAuthMiddleware:
    def _make_middleware(
        self,
        key_file: Path | None = None,
    ) -> AuthMiddleware:
        return AuthMiddleware(_echo_app, key_file=key_file)

    def test_api_without_key_returns_401(self) -> None:
        mw = self._make_middleware()
        resp = _CapturedResponse()
        environ = _make_environ("GET", "/api/contract")
        body_parts = mw(environ, resp)
        assert resp.status.startswith("401")
        data = json.loads(b"".join(body_parts))
        assert "error" in data

    def test_api_with_valid_key_passes_through(self) -> None:
        mw = self._make_middleware()
        plaintext, _ = mw.key_store.generate_key("t1", "surgeon")

        resp = _CapturedResponse()
        environ = _make_environ("GET", "/api/contract", api_key=plaintext)
        body_parts = mw(environ, resp)
        assert resp.status.startswith("200")
        data = json.loads(b"".join(body_parts))
        assert data["tenant_id"] == "t1"
        assert data["role"] == "surgeon"

    def test_api_with_bearer_token(self) -> None:
        mw = self._make_middleware()
        plaintext, _ = mw.key_store.generate_key("t2", "researcher")

        resp = _CapturedResponse()
        environ = _make_environ("GET", "/api/cases", bearer=plaintext)
        body_parts = mw(environ, resp)
        assert resp.status.startswith("200")
        data = json.loads(b"".join(body_parts))
        assert data["tenant_id"] == "t2"

    def test_api_with_invalid_key_returns_401(self) -> None:
        mw = self._make_middleware()
        resp = _CapturedResponse()
        environ = _make_environ("GET", "/api/contract", api_key="fp_bad_key")
        body_parts = mw(environ, resp)
        assert resp.status.startswith("401")

    def test_options_skips_auth(self) -> None:
        mw = self._make_middleware()
        resp = _CapturedResponse()
        environ = _make_environ("OPTIONS", "/api/contract")
        body_parts = mw(environ, resp)
        assert resp.status.startswith("200")

    def test_static_skips_auth(self) -> None:
        mw = self._make_middleware()
        resp = _CapturedResponse()
        environ = _make_environ("GET", "/index.html")
        body_parts = mw(environ, resp)
        assert resp.status.startswith("200")

    def test_metrics_exempt(self) -> None:
        mw = self._make_middleware()
        resp = _CapturedResponse()
        environ = _make_environ("GET", "/metrics")
        body_parts = mw(environ, resp)
        assert resp.status.startswith("200")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RateLimitMiddleware
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRateLimitMiddleware:
    def test_under_limit_passes(self) -> None:
        mw = RateLimitMiddleware(_echo_app, rpm=10)
        for _ in range(10):
            resp = _CapturedResponse()
            environ = _make_environ("GET", "/api/contract")
            body_parts = mw(environ, resp)
            assert resp.status.startswith("200")

    def test_over_limit_returns_429(self) -> None:
        mw = RateLimitMiddleware(_echo_app, rpm=5)
        for _ in range(5):
            resp = _CapturedResponse()
            mw(_make_environ("GET", "/api/contract"), resp)

        # 6th request should be rate limited
        resp = _CapturedResponse()
        body_parts = mw(_make_environ("GET", "/api/contract"), resp)
        assert resp.status.startswith("429")
        data = json.loads(b"".join(body_parts))
        assert "retry_after_seconds" in data

    def test_different_ips_have_separate_limits(self) -> None:
        mw = RateLimitMiddleware(_echo_app, rpm=2)

        # Exhaust IP1
        for _ in range(2):
            resp = _CapturedResponse()
            mw(_make_environ("GET", "/api/x", remote_addr="1.1.1.1"), resp)

        # IP1 is limited
        resp = _CapturedResponse()
        mw(_make_environ("GET", "/api/x", remote_addr="1.1.1.1"), resp)
        assert resp.status.startswith("429")

        # IP2 is still fine
        resp = _CapturedResponse()
        mw(_make_environ("GET", "/api/x", remote_addr="2.2.2.2"), resp)
        assert resp.status.startswith("200")

    def test_x_forwarded_for_respected(self) -> None:
        mw = RateLimitMiddleware(_echo_app, rpm=2)

        for _ in range(2):
            resp = _CapturedResponse()
            env = _make_environ("GET", "/api/x", remote_addr="127.0.0.1")
            env["HTTP_X_FORWARDED_FOR"] = "10.0.0.1, 192.168.1.1"
            mw(env, resp)

        # Client 10.0.0.1 is now limited
        resp = _CapturedResponse()
        env = _make_environ("GET", "/api/x", remote_addr="127.0.0.1")
        env["HTTP_X_FORWARDED_FOR"] = "10.0.0.1"
        mw(env, resp)
        assert resp.status.startswith("429")
