"""Tests for the HTTP server module (handler logic, routing, CORS, security)."""

from __future__ import annotations

import io
import json
import tempfile
from http.server import HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from products.facial_plastics.ui.server import (
    _RequestHandler,
    _STATIC_DIR,
    start_server,
)


# ═══════════════════════════════════════════════════════════════════
#  Test helpers — lightweight handler harness
# ═══════════════════════════════════════════════════════════════════

class _FakeWFile(io.BytesIO):
    """In-memory write file that captures response bytes."""
    pass


class _FakeRFile(io.BytesIO):
    """In-memory read file that supplies request body bytes."""
    pass


def _make_handler(
    method: str,
    path: str,
    body: Optional[bytes] = None,
    app: Optional[Any] = None,
) -> _RequestHandler:
    """Construct a _RequestHandler with fake I/O for testing.

    We set up just enough of the BaseHTTPRequestHandler machinery to
    call do_GET / do_POST / do_OPTIONS directly, capturing the
    output written to wfile.
    """
    if app is None:
        app = MagicMock()

    rfile = _FakeRFile(body or b"")
    wfile = _FakeWFile()

    # Build a minimal fake request line
    request_line = f"{method} {path} HTTP/1.1"

    # Create a handler class with the app bound
    handler_cls = type("_TestHandler", (_RequestHandler,), {"app": app})

    # BaseHTTPRequestHandler calls parse_request in __init__,
    # which reads from rfile. We bypass __init__ and set attributes manually.
    handler = handler_cls.__new__(handler_cls)  # type: ignore[call-overload]
    handler.rfile = rfile
    handler.wfile = wfile
    handler.path = path
    handler.command = method
    handler.request_version = "HTTP/1.1"
    handler.headers = _FakeHeaders(
        {"Content-Length": str(len(body)) if body else "0"}
    )
    handler.requestline = request_line
    handler.client_address = ("127.0.0.1", 12345)
    handler.server = MagicMock()
    handler.close_connection = True
    handler.responses = _RequestHandler.responses

    return handler  # type: ignore[no-any-return]


class _FakeHeaders:
    """Minimal headers dict-like for the handler."""

    def __init__(self, mapping: Dict[str, str]) -> None:
        self._map = {k.lower(): v for k, v in mapping.items()}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._map.get(key.lower(), default)


def _extract_response(handler: _RequestHandler) -> tuple[int, Dict[str, str], bytes]:
    """Parse the raw HTTP response from wfile into (status, headers, body)."""
    raw = handler.wfile.getvalue()  # type: ignore[attr-defined]
    # Split header and body
    parts = raw.split(b"\r\n\r\n", 1)
    header_block = parts[0].decode("utf-8", errors="replace")
    body = parts[1] if len(parts) > 1 else b""

    lines = header_block.split("\r\n")
    # Status line: "HTTP/1.1 200 OK"
    status_line = lines[0]
    status_code = int(status_line.split(" ")[1])

    headers: Dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    return status_code, headers, body


# ═══════════════════════════════════════════════════════════════════
#  CORS tests
# ═══════════════════════════════════════════════════════════════════

class TestCORS:
    """CORS headers should be present on all responses."""

    def test_options_preflight(self) -> None:
        handler = _make_handler("OPTIONS", "/api/cases")
        handler.do_OPTIONS()

        status, headers, _ = _extract_response(handler)
        assert status == 204
        assert headers.get("access-control-allow-origin") == "*"
        assert "GET" in headers.get("access-control-allow-methods", "")
        assert "POST" in headers.get("access-control-allow-methods", "")
        assert "Content-Type" in headers.get("access-control-allow-headers", "")

    def test_cors_on_json_response(self) -> None:
        app = MagicMock()
        app.get_contract.return_value = {"version": "1.0"}
        handler = _make_handler("GET", "/api/contract", app=app)
        handler.do_GET()

        _, headers, _ = _extract_response(handler)
        assert headers.get("access-control-allow-origin") == "*"


# ═══════════════════════════════════════════════════════════════════
#  Route dispatch tests
# ═══════════════════════════════════════════════════════════════════

class TestRouteDispatch:
    """Test that URL paths dispatch to the correct UIApplication methods."""

    def test_get_contract(self) -> None:
        app = MagicMock()
        app.get_contract.return_value = {"name": "test"}
        handler = _make_handler("GET", "/api/contract", app=app)
        handler.do_GET()

        status, _, body = _extract_response(handler)
        assert status == 200
        data = json.loads(body)
        assert data["name"] == "test"
        app.get_contract.assert_called_once()

    def test_get_cases_list(self) -> None:
        app = MagicMock()
        app.list_cases.return_value = {"cases": []}
        handler = _make_handler("GET", "/api/cases?limit=10&offset=0", app=app)
        handler.do_GET()

        status, _, body = _extract_response(handler)
        assert status == 200
        app.list_cases.assert_called_once()

    def test_get_case_by_id(self) -> None:
        app = MagicMock()
        app.get_case.return_value = {"case_id": "abc123"}
        handler = _make_handler("GET", "/api/cases/abc123", app=app)
        handler.do_GET()

        status, _, body = _extract_response(handler)
        assert status == 200
        app.get_case.assert_called_once_with("abc123")

    def test_get_case_twin(self) -> None:
        app = MagicMock()
        app.get_twin_summary.return_value = {"twin": True}
        handler = _make_handler("GET", "/api/cases/abc123/twin", app=app)
        handler.do_GET()

        status, _, _ = _extract_response(handler)
        assert status == 200
        app.get_twin_summary.assert_called_once_with("abc123")

    def test_get_case_mesh(self) -> None:
        app = MagicMock()
        app.get_mesh_data.return_value = {"mesh": "data"}
        handler = _make_handler("GET", "/api/cases/case1/mesh", app=app)
        handler.do_GET()
        app.get_mesh_data.assert_called_once_with("case1")

    def test_get_operators(self) -> None:
        app = MagicMock()
        app.list_operators.return_value = {"operators": []}
        handler = _make_handler("GET", "/api/operators", app=app)
        handler.do_GET()
        app.list_operators.assert_called_once()

    def test_get_templates(self) -> None:
        app = MagicMock()
        app.list_templates.return_value = {"templates": []}
        handler = _make_handler("GET", "/api/templates", app=app)
        handler.do_GET()
        app.list_templates.assert_called_once()

    def test_unknown_get_route(self) -> None:
        handler = _make_handler("GET", "/api/nonexistent")
        handler.do_GET()
        status, _, body = _extract_response(handler)
        assert status == 200
        data = json.loads(body)
        assert "error" in data

    def test_post_create_case(self) -> None:
        app = MagicMock()
        app.create_case.return_value = {"case_id": "new123"}
        payload = json.dumps({"patient_id": "P1"}).encode()
        handler = _make_handler("POST", "/api/cases", body=payload, app=app)
        handler.do_POST()

        status, _, body = _extract_response(handler)
        assert status == 200
        app.create_case.assert_called_once()

    def test_post_curate(self) -> None:
        app = MagicMock()
        app.curate_library.return_value = {"curated": True}
        handler = _make_handler("POST", "/api/curate", body=b"{}", app=app)
        handler.do_POST()
        app.curate_library.assert_called_once()

    def test_post_delete_case(self) -> None:
        app = MagicMock()
        app.delete_case.return_value = {"deleted": True}
        handler = _make_handler("POST", "/api/cases/xyz/delete", body=b"{}", app=app)
        handler.do_POST()
        app.delete_case.assert_called_once_with("xyz")

    def test_post_plan_template(self) -> None:
        app = MagicMock()
        app.create_plan_from_template.return_value = {"plan": {}}
        payload = json.dumps({"template": "standard"}).encode()
        handler = _make_handler("POST", "/api/plan/template", body=payload, app=app)
        handler.do_POST()
        app.create_plan_from_template.assert_called_once()

    def test_post_invalid_json(self) -> None:
        handler = _make_handler("POST", "/api/cases", body=b"not json{{{")
        handler.do_POST()
        status, _, body = _extract_response(handler)
        assert status == 400
        data = json.loads(body)
        assert "error" in data

    def test_post_unknown_route(self) -> None:
        handler = _make_handler("POST", "/api/unknown", body=b"{}")
        handler.do_POST()
        status, _, body = _extract_response(handler)
        assert status == 200
        data = json.loads(body)
        assert "error" in data

    def test_api_exception_returns_500(self) -> None:
        app = MagicMock()
        app.get_contract.side_effect = RuntimeError("database down")
        handler = _make_handler("GET", "/api/contract", app=app)
        handler.do_GET()

        status, _, body = _extract_response(handler)
        assert status == 500
        data = json.loads(body)
        assert "error" in data
        assert "database down" in data["error"]


# ═══════════════════════════════════════════════════════════════════
#  Path traversal protection tests
# ═══════════════════════════════════════════════════════════════════

class TestPathTraversalProtection:
    """Ensure path traversal attacks are blocked."""

    def test_dotdot_blocked(self) -> None:
        handler = _make_handler("GET", "/../../../etc/passwd")
        handler.do_GET()
        status, _, body = _extract_response(handler)
        # Should get 403 Forbidden or 404 (never the file contents)
        assert status in (403, 404)

    def test_encoded_traversal_safe(self) -> None:
        """URL-encoded traversal sequences stay as literal characters.

        Python's ``urlparse`` does not decode percent-encoding in the path
        component, so ``%2e%2e`` becomes a literal directory named ``%2e%2e``
        which resolves inside _STATIC_DIR.  The resolved path never escapes
        the static root, so the request is safe — it either returns 200 via
        SPA fallback (index.html) or 404 when index.html is absent.
        """
        handler = _make_handler("GET", "/..%2F..%2Fetc/passwd")
        handler.do_GET()
        status, _, _ = _extract_response(handler)
        # The literal path won't exist; SPA fallback returns index.html (200)
        # or 404 if index.html is missing.  Either way, no file disclosure.
        assert status in (200, 404)

    def test_null_byte_in_path(self) -> None:
        """Null bytes in path should not bypass checks."""
        handler = _make_handler("GET", "/index.html\x00.jpg")
        handler.do_GET()
        status, _, _ = _extract_response(handler)
        # 404 is acceptable (file won't exist); must not be 200 with wrong content
        assert status in (403, 404)


# ═══════════════════════════════════════════════════════════════════
#  SPA fallback tests
# ═══════════════════════════════════════════════════════════════════

class TestSPAFallback:
    """Test SPA client-side routing fallback behavior."""

    def test_root_serves_index(self, tmp_path: Path) -> None:
        """GET / serves index.html."""
        # Temporarily point _STATIC_DIR to tmp_path
        index = tmp_path / "index.html"
        index.write_text("<html>hello</html>")

        handler = _make_handler("GET", "/")
        with patch("products.facial_plastics.ui.server._STATIC_DIR", tmp_path):
            handler._serve_static("/")

        status, headers, body = _extract_response(handler)
        assert status == 200
        assert b"hello" in body

    def test_unknown_path_falls_back_to_index(self, tmp_path: Path) -> None:
        """Non-existent static paths serve index.html for SPA routing."""
        index = tmp_path / "index.html"
        index.write_text("<html>spa</html>")

        handler = _make_handler("GET", "/some/deep/route")
        with patch("products.facial_plastics.ui.server._STATIC_DIR", tmp_path):
            handler._serve_static("/some/deep/route")

        status, _, body = _extract_response(handler)
        assert status == 200
        assert b"spa" in body

    def test_no_index_returns_404(self, tmp_path: Path) -> None:
        """Missing index.html returns 404."""
        handler = _make_handler("GET", "/anywhere")
        with patch("products.facial_plastics.ui.server._STATIC_DIR", tmp_path):
            handler._serve_static("/anywhere")

        status, _, body = _extract_response(handler)
        assert status == 404

    def test_existing_static_file_served(self, tmp_path: Path) -> None:
        """An existing static file is served with correct content-type."""
        css_file = tmp_path / "style.css"
        css_file.write_text("body { color: red; }")

        handler = _make_handler("GET", "/style.css")
        with patch("products.facial_plastics.ui.server._STATIC_DIR", tmp_path):
            handler._serve_static("/style.css")

        status, headers, body = _extract_response(handler)
        assert status == 200
        assert b"color: red" in body
        assert "text/css" in headers.get("content-type", "")


# ═══════════════════════════════════════════════════════════════════
#  Handler class construction test
# ═══════════════════════════════════════════════════════════════════

class TestHandlerConstruction:
    """Test the handler class factory used by start_server."""

    def test_start_server_creates_httpserver(self, tmp_path: Path) -> None:
        """start_server returns an HTTPServer, binding the UIApplication."""
        server = start_server(port=0, library_root=tmp_path, host="127.0.0.1")
        try:
            assert isinstance(server, HTTPServer)
            # Handler class should have app attribute
            handler_cls = server.RequestHandlerClass
            assert hasattr(handler_cls, "app")
        finally:
            server.server_close()

    def test_library_root_created(self, tmp_path: Path) -> None:
        """start_server creates the library root directory if missing."""
        lib_dir = tmp_path / "new_library"
        assert not lib_dir.exists()
        server = start_server(port=0, library_root=lib_dir, host="127.0.0.1")
        try:
            assert lib_dir.exists()
        finally:
            server.server_close()


# ═══════════════════════════════════════════════════════════════════
#  JSON response helper tests
# ═══════════════════════════════════════════════════════════════════

class TestJSONResponse:
    """Test _send_json and _send_error helpers."""

    def test_send_json_content_type(self) -> None:
        handler = _make_handler("GET", "/api/contract")
        handler._send_json({"ok": True}, status=200)

        status, headers, body = _extract_response(handler)
        assert status == 200
        assert "application/json" in headers.get("content-type", "")
        data = json.loads(body)
        assert data["ok"] is True

    def test_send_error(self) -> None:
        handler = _make_handler("GET", "/api/bad")
        handler._send_error(404, "Not Found")

        status, _, body = _extract_response(handler)
        assert status == 404
        data = json.loads(body)
        assert data["error"] == "Not Found"

    def test_content_length_header(self) -> None:
        handler = _make_handler("GET", "/api/data")
        handler._send_json({"key": "value"})

        _, headers, body = _extract_response(handler)
        assert int(headers.get("content-length", 0)) == len(body)
