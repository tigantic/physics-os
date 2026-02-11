"""Tests for WSGI adapter — products.facial_plastics.ui.wsgi."""

from __future__ import annotations

import json
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from products.facial_plastics.ui.wsgi import WSGIApplication, create_app


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

    def header(self, name: str) -> str:
        for k, v in self.headers:
            if k.lower() == name.lower():
                return v
        return ""


def _make_environ(
    method: str = "GET",
    path: str = "/",
    query: str = "",
    body: bytes = b"",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a minimal WSGI environ dict."""
    env: Dict[str, Any] = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": query,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": BytesIO(body),
        "REMOTE_ADDR": "127.0.0.1",
        "SERVER_NAME": "localhost",
        "SERVER_PORT": "8420",
    }
    if extra:
        env.update(extra)
    return env


@pytest.fixture()
def wsgi_app(tmp_path: Path) -> WSGIApplication:
    """Create a WSGIApplication with a temp library root."""
    return WSGIApplication(library_root=tmp_path)


# ── Contract route ────────────────────────────────────────────────


class TestWSGIContractRoute:
    def test_get_contract_returns_json(self, wsgi_app: WSGIApplication) -> None:
        resp = _CapturedResponse()
        environ = _make_environ("GET", "/api/contract")
        body_parts = wsgi_app(environ, resp)
        assert resp.status.startswith("200")
        assert resp.header("Content-Type").startswith("application/json")
        data = json.loads(b"".join(body_parts))
        assert "modes" in data or "contract" in data or isinstance(data, dict)


# ── CORS ──────────────────────────────────────────────────────────


class TestWSGICors:
    def test_options_returns_204(self, wsgi_app: WSGIApplication) -> None:
        resp = _CapturedResponse()
        environ = _make_environ("OPTIONS", "/api/contract")
        body_parts = wsgi_app(environ, resp)
        assert resp.status.startswith("204")
        assert resp.header("Access-Control-Allow-Origin") != ""

    def test_get_includes_cors_header(self, wsgi_app: WSGIApplication) -> None:
        resp = _CapturedResponse()
        environ = _make_environ("GET", "/api/contract")
        wsgi_app(environ, resp)
        assert resp.header("Access-Control-Allow-Origin") == "*"


# ── Static serving ────────────────────────────────────────────────


class TestWSGIStatic:
    def test_root_returns_index_or_404(self, wsgi_app: WSGIApplication) -> None:
        resp = _CapturedResponse()
        environ = _make_environ("GET", "/")
        body_parts = wsgi_app(environ, resp)
        # Either serves index.html or 404 if static dir is empty
        code = int(resp.status.split()[0])
        assert code in (200, 404)

    def test_path_traversal_blocked(self, wsgi_app: WSGIApplication) -> None:
        resp = _CapturedResponse()
        environ = _make_environ("GET", "/../../../etc/passwd")
        body_parts = wsgi_app(environ, resp)
        code = int(resp.status.split()[0])
        assert code in (403, 404)


# ── Metrics endpoint ──────────────────────────────────────────────


class TestWSGIMetrics:
    def test_metrics_returns_prometheus_format(
        self,
        wsgi_app: WSGIApplication,
    ) -> None:
        # Make a few requests first
        for _ in range(3):
            resp = _CapturedResponse()
            wsgi_app(_make_environ("GET", "/api/contract"), resp)

        resp = _CapturedResponse()
        environ = _make_environ("GET", "/metrics")
        body_parts = wsgi_app(environ, resp)
        assert resp.status.startswith("200")
        body = b"".join(body_parts).decode("utf-8")
        assert "fp_requests_total" in body
        assert "fp_errors_total" in body
        assert "fp_avg_latency_ms" in body


# ── POST routes ───────────────────────────────────────────────────


class TestWSGIPost:
    def test_unknown_post_route(self, wsgi_app: WSGIApplication) -> None:
        resp = _CapturedResponse()
        body = json.dumps({"foo": "bar"}).encode()
        environ = _make_environ("POST", "/api/nonexistent", body=body)
        body_parts = wsgi_app(environ, resp)
        data = json.loads(b"".join(body_parts))
        assert "error" in data

    def test_method_not_allowed(self, wsgi_app: WSGIApplication) -> None:
        resp = _CapturedResponse()
        environ = _make_environ("DELETE", "/api/contract")
        body_parts = wsgi_app(environ, resp)
        assert resp.status.startswith("405")


# ── Request/error counters ────────────────────────────────────────


class TestWSGICounters:
    def test_request_count_increments(
        self,
        wsgi_app: WSGIApplication,
    ) -> None:
        assert wsgi_app.request_count == 0
        resp = _CapturedResponse()
        wsgi_app(_make_environ("GET", "/api/contract"), resp)
        assert wsgi_app.request_count == 1
        wsgi_app(_make_environ("GET", "/api/contract"), resp)
        assert wsgi_app.request_count == 2


# ── Factory function ──────────────────────────────────────────────


class TestCreateApp:
    def test_create_app_default_has_auth(self, tmp_path: Path) -> None:
        os.environ["HYPERTENSOR_DATA_ROOT"] = str(tmp_path)
        try:
            app = create_app()
            # With auth enabled (default), outermost layer is RateLimitMiddleware
            from products.facial_plastics.ui.auth import RateLimitMiddleware
            assert isinstance(app, RateLimitMiddleware)
        finally:
            del os.environ["HYPERTENSOR_DATA_ROOT"]

    def test_create_app_auth_disabled(self, tmp_path: Path) -> None:
        app = create_app(library_root=tmp_path, enable_auth=False, enable_rate_limit=False)
        assert isinstance(app, WSGIApplication)

    def test_create_app_with_explicit_root(self, tmp_path: Path) -> None:
        app = create_app(library_root=tmp_path, enable_auth=False, enable_rate_limit=False)
        assert isinstance(app, WSGIApplication)

    def test_create_app_with_cors_env(self, tmp_path: Path) -> None:
        os.environ["FP_CORS_ORIGINS"] = "https://a.com, https://b.com"
        try:
            app = create_app(library_root=tmp_path, enable_auth=False, enable_rate_limit=False)
            assert isinstance(app, WSGIApplication)
        finally:
            del os.environ["FP_CORS_ORIGINS"]

    def test_create_app_auth_requires_key_for_api(self, tmp_path: Path) -> None:
        """Verify that create_app with auth enabled rejects unauthenticated API calls."""
        app = create_app(library_root=tmp_path, enable_auth=True, enable_rate_limit=False)
        resp = _CapturedResponse()
        environ = _make_environ("GET", "/api/contract")
        body_parts = app(environ, resp)
        assert resp.status.startswith("401")

    def test_create_app_env_bool_disable(self, tmp_path: Path) -> None:
        os.environ["FP_AUTH_ENABLED"] = "false"
        os.environ["FP_RATE_LIMIT_ENABLED"] = "0"
        try:
            app = create_app(library_root=tmp_path)
            assert isinstance(app, WSGIApplication)
        finally:
            del os.environ["FP_AUTH_ENABLED"]
            del os.environ["FP_RATE_LIMIT_ENABLED"]
