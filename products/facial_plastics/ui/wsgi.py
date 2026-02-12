"""WSGI application for the Facial Plastics platform.

``create_app()`` returns a fully composed middleware stack::

    RateLimitMiddleware → AuthMiddleware → WSGIApplication

Usage with gunicorn (production)::

    gunicorn "products.facial_plastics.ui.wsgi:create_app()" \\
        --config products/facial_plastics/gunicorn.conf.py

Or create the app programmatically::

    # Production (auth + rate limiting on by default)
    app = create_app(library_root=Path("/data"))

    # Development (no auth)
    app = create_app(
        library_root=Path("/data"),
        enable_auth=False,
        enable_rate_limit=False,
    )

Environment variables consumed by ``create_app()``:

    HYPERTENSOR_DATA_ROOT   Case library root (default: ./cases)
    FP_CORS_ORIGINS         Comma-separated CORS origins (default: *)
    FP_AUTH_ENABLED          true/false (default: true)
    FP_RATE_LIMIT_ENABLED    true/false (default: true)
    FP_RATE_LIMIT_RPM        Requests/min per IP (default: 120)
    FP_KEY_FILE              Path to API key store JSON
"""

from __future__ import annotations

import io
import json
import logging
import mimetypes
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .api import UIApplication

logger = logging.getLogger(__name__)

# Type aliases for WSGI
_Environ = Dict[str, Any]
_StartResponse = Callable[..., Any]
_WSGIApp = Callable[[_Environ, _StartResponse], Iterable[bytes]]

# Static asset directory
_STATIC_DIR = Path(__file__).parent / "static"


def _get_version() -> str:
    """Return the package version string."""
    try:
        from products.facial_plastics import __version__
        return str(__version__)
    except ImportError:
        return "unknown"


class WSGIApplication:
    """Production WSGI application for the Facial Plastics platform.

    Wraps :class:`UIApplication` with WSGI compliance, structured logging,
    and request metrics collection.

    Parameters
    ----------
    library_root : Path
        Root directory for the case library.
    allowed_origins : tuple[str, ...]
        Origins permitted for CORS.  ``("*",)`` for development.
    """

    def __init__(
        self,
        library_root: Path,
        *,
        allowed_origins: Tuple[str, ...] = ("*",),
    ) -> None:
        self._library_root = Path(library_root)
        self._library_root.mkdir(parents=True, exist_ok=True)
        self._app = UIApplication(self._library_root)
        self._allowed_origins = allowed_origins
        self._request_count: int = 0
        self._error_count: int = 0
        self._total_latency_ms: float = 0.0
        logger.info(
            "WSGIApplication initialised — library: %s", self._library_root,
        )

    # ── WSGI entry point ──────────────────────────────────────────

    def __call__(
        self,
        environ: _Environ,
        start_response: _StartResponse,
    ) -> Iterable[bytes]:
        """WSGI callable."""
        t0 = time.monotonic()
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path_info = environ.get("PATH_INFO", "/").rstrip("/") or "/"
        query_string = environ.get("QUERY_STRING", "")
        query = parse_qs(query_string)

        # Operational endpoints — exempt from request counters so
        # infrastructure healthchecks don't inflate business metrics.
        if path_info == "/health":
            return self._health_response(start_response)
        if path_info == "/metrics":
            return self._metrics_response(start_response)

        self._request_count += 1

        try:
            if method == "OPTIONS":
                return self._cors_preflight(start_response)

            if path_info.startswith("/api/"):
                if method == "GET":
                    return self._handle_api_get(
                        path_info, query, start_response,
                    )
                if method == "POST":
                    body = self._read_body(environ)
                    return self._handle_api_post(
                        path_info, body, start_response,
                    )
                return self._json_response(
                    {"error": f"Method {method} not allowed"},
                    start_response,
                    status=405,
                )

            # Static / SPA fallback
            return self._serve_static(path_info, start_response)

        except Exception as exc:
            self._error_count += 1
            logger.error(
                "Unhandled error: %s\n%s", exc, traceback.format_exc(),
            )
            return self._json_response(
                {"error": "Internal Server Error"},
                start_response,
                status=500,
            )
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._total_latency_ms += elapsed_ms
            logger.info(
                "%s %s %s — %.1fms",
                method,
                path_info,
                query_string[:80] if query_string else "",
                elapsed_ms,
            )

    # ── API dispatch ──────────────────────────────────────────────

    def _handle_api_get(
        self,
        path: str,
        query: Dict[str, List[str]],
        start_response: _StartResponse,
    ) -> Iterable[bytes]:
        """Route GET /api/* requests."""
        try:
            result = self._dispatch_get(path, query)
            return self._json_response(result, start_response)
        except Exception as exc:
            self._error_count += 1
            logger.error("API GET error: %s\n%s", exc, traceback.format_exc())
            return self._json_response(
                {"error": str(exc)}, start_response, status=500,
            )

    def _handle_api_post(
        self,
        path: str,
        body: Dict[str, Any],
        start_response: _StartResponse,
    ) -> Iterable[bytes]:
        """Route POST /api/* requests."""
        try:
            result = self._dispatch_post(path, body)
            return self._json_response(result, start_response)
        except Exception as exc:
            self._error_count += 1
            logger.error("API POST error: %s\n%s", exc, traceback.format_exc())
            return self._json_response(
                {"error": str(exc)}, start_response, status=500,
            )

    def _dispatch_get(
        self,
        path: str,
        query: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Route GET requests to UIApplication methods."""
        app = self._app

        def _q(key: str, default: Optional[str] = None) -> Optional[str]:
            vals = query.get(key)
            return vals[0] if vals else default

        if path == "/api/contract":
            return app.get_contract()

        if path == "/api/cases":
            return app.list_cases(
                procedure=_q("procedure"),
                quality=_q("quality"),
                limit=int(_q("limit", "100") or "100"),
                offset=int(_q("offset", "0") or "0"),
            )

        if path.startswith("/api/cases/"):
            parts = path.split("/")
            if len(parts) >= 4:
                case_id = parts[3]
                if len(parts) == 4:
                    return app.get_case(case_id)
                sub = parts[4] if len(parts) > 4 else ""
                if sub == "twin":
                    return app.get_twin_summary(case_id)
                if sub == "mesh":
                    return app.get_mesh_data(case_id)
                if sub == "landmarks":
                    return app.get_landmarks(case_id)
                if sub == "visualization":
                    return app.get_visualization_data(case_id)
                if sub == "timeline":
                    return app.get_timeline(case_id)

        if path == "/api/operators":
            return app.list_operators(procedure=_q("procedure"))

        if path == "/api/templates":
            return app.list_templates()

        return {"error": f"Unknown GET route: {path}"}

    def _dispatch_post(
        self,
        path: str,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route POST requests to UIApplication methods."""
        app = self._app

        if path == "/api/cases":
            return app.create_case(**body)

        if path.startswith("/api/cases/") and path.endswith("/delete"):
            case_id = path.split("/")[3]
            return app.delete_case(case_id)

        if path == "/api/curate":
            return app.curate_library()

        if path == "/api/plan/template":
            return app.create_plan_from_template(**body)

        if path == "/api/plan/custom":
            return app.create_custom_plan(**body)

        if path == "/api/plan/compile":
            return app.compile_plan(
                case_id=body["case_id"],
                plan_dict=body["plan"],
            )

        if path == "/api/whatif":
            return app.run_whatif(
                case_id=body["case_id"],
                plan_dict=body["plan"],
                modified_params=body.get("modified_params", {}),
            )

        if path == "/api/sweep":
            return app.parameter_sweep(
                case_id=body["case_id"],
                plan_dict=body["plan"],
                sweep_op=body["sweep_op"],
                sweep_param=body["sweep_param"],
                values=body["values"],
            )

        if path == "/api/report":
            return app.generate_report(
                case_id=body["case_id"],
                plan_dict=body["plan"],
                format=body.get("format", "html"),
            )

        if path == "/api/compare/plans":
            return app.compare_plans(
                case_id=body["case_id"],
                plan_a=body["plan_a"],
                plan_b=body["plan_b"],
            )

        if path == "/api/compare/cases":
            return app.compare_cases(
                case_id_a=body["case_id_a"],
                case_id_b=body["case_id_b"],
            )

        return {"error": f"Unknown POST route: {path}"}

    # ── Helpers ───────────────────────────────────────────────────

    def _read_body(self, environ: _Environ) -> Dict[str, Any]:
        """Read and parse JSON body from WSGI environ."""
        content_length = int(environ.get("CONTENT_LENGTH") or 0)
        if content_length == 0:
            return {}
        body_bytes = environ["wsgi.input"].read(content_length)
        try:
            return json.loads(body_bytes)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError):
            return {}

    def _json_response(
        self,
        data: Dict[str, Any],
        start_response: _StartResponse,
        *,
        status: int = 200,
    ) -> Iterable[bytes]:
        """Emit a JSON response."""
        body = json.dumps(data, indent=None, default=str).encode("utf-8")
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ]
        headers.extend(self._cors_headers())
        start_response(f"{status} {_STATUS_PHRASES.get(status, 'OK')}", headers)
        return [body]

    def _cors_preflight(
        self,
        start_response: _StartResponse,
    ) -> Iterable[bytes]:
        """Handle CORS OPTIONS preflight."""
        headers = list(self._cors_headers())
        headers.append(("Content-Length", "0"))
        start_response("204 No Content", headers)
        return [b""]

    def _security_headers(self) -> List[Tuple[str, str]]:
        """Security headers applied to every response.

        CSP is intentionally omitted here — it is enforced at the
        reverse-proxy layer (Caddyfile) for production.  Emitting a
        CSP from the application creates a second policy that the
        browser intersects with any extension-injected policy (e.g.
        MetaMask SES lockdown), producing false-positive console
        warnings about ``eval`` that cannot be silenced.
        """
        return [
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
            ("Referrer-Policy", "strict-origin-when-cross-origin"),
        ]

    def _cors_headers(self) -> List[Tuple[str, str]]:
        origin = self._allowed_origins[0] if self._allowed_origins else "*"
        headers = [
            ("Access-Control-Allow-Origin", origin),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key"),
            ("Access-Control-Max-Age", "86400"),
        ]
        headers.extend(self._security_headers())
        return headers

    def _serve_static(
        self,
        path: str,
        start_response: _StartResponse,
    ) -> Iterable[bytes]:
        """Serve static assets with SPA fallback."""
        if path in ("", "/"):
            path = "/index.html"

        file_path = _STATIC_DIR / path.lstrip("/")

        try:
            file_path = file_path.resolve()
            if not str(file_path).startswith(str(_STATIC_DIR.resolve())):
                return self._json_response(
                    {"error": "Forbidden"}, start_response, status=403,
                )
        except (ValueError, OSError):
            return self._json_response(
                {"error": "Forbidden"}, start_response, status=403,
            )

        if not file_path.is_file():
            file_path = _STATIC_DIR / "index.html"
            if not file_path.is_file():
                return self._json_response(
                    {"error": "Not Found"}, start_response, status=404,
                )

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        try:
            data = file_path.read_bytes()
            headers = [
                ("Content-Type", content_type),
                ("Content-Length", str(len(data))),
                ("Cache-Control", "public, max-age=3600"),
            ]
            headers.extend(self._cors_headers())
            start_response("200 OK", headers)
            return [data]
        except OSError:
            return self._json_response(
                {"error": "Read error"}, start_response, status=500,
            )

    def _health_response(
        self,
        start_response: _StartResponse,
    ) -> Iterable[bytes]:
        """Lightweight health check — no auth required."""
        data = {
            "status": "healthy",
            "version": _get_version(),
            "requests": self._request_count,
            "errors": self._error_count,
        }
        return self._json_response(data, start_response)

    def _metrics_response(
        self,
        start_response: _StartResponse,
    ) -> Iterable[bytes]:
        """Expose Prometheus-compatible metrics at /metrics."""
        lines = [
            "# HELP fp_requests_total Total HTTP requests handled.",
            "# TYPE fp_requests_total counter",
            f"fp_requests_total {self._request_count}",
            "",
            "# HELP fp_errors_total Total HTTP 5xx errors.",
            "# TYPE fp_errors_total counter",
            f"fp_errors_total {self._error_count}",
            "",
            "# HELP fp_avg_latency_ms Average request latency in milliseconds.",
            "# TYPE fp_avg_latency_ms gauge",
            f"fp_avg_latency_ms {self._total_latency_ms / max(self._request_count, 1):.1f}",
            "",
        ]
        body = "\n".join(lines).encode("utf-8")
        headers = [
            ("Content-Type", "text/plain; version=0.0.4; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ]
        headers.extend(self._security_headers())
        start_response("200 OK", headers)
        return [body]

    # ── Properties for monitoring ─────────────────────────────────

    @property
    def request_count(self) -> int:
        """Total requests handled since startup."""
        return self._request_count

    @property
    def error_count(self) -> int:
        """Total 5xx errors since startup."""
        return self._error_count

    @property
    def avg_latency_ms(self) -> float:
        """Average request latency in milliseconds."""
        if self._request_count == 0:
            return 0.0
        return self._total_latency_ms / self._request_count


# ── HTTP status phrases ───────────────────────────────────────────

_STATUS_PHRASES: Dict[int, str] = {
    200: "OK",
    204: "No Content",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    429: "Too Many Requests",
    500: "Internal Server Error",
}


# ── Factory ───────────────────────────────────────────────────────

def create_app(
    library_root: Optional[Path] = None,
    allowed_origins: Optional[Tuple[str, ...]] = None,
    *,
    enable_auth: Optional[bool] = None,
    enable_rate_limit: Optional[bool] = None,
    rate_limit_rpm: int = 120,
) -> _WSGIApp:
    """Create the production WSGI application.

    Composes the middleware stack::

        RateLimitMiddleware → AuthMiddleware → WSGIApplication

    Parameters
    ----------
    library_root : Path | None
        Case library directory.  Falls back to ``$HYPERTENSOR_DATA_ROOT``
        then ``./cases``.
    allowed_origins : tuple[str, ...] | None
        CORS origins.  Falls back to ``$FP_CORS_ORIGINS`` (comma-separated)
        then ``("*",)``.
    enable_auth : bool | None
        Enable API-key authentication.  Falls back to
        ``$FP_AUTH_ENABLED`` ("0" / "false" to disable), default **True**.
    enable_rate_limit : bool | None
        Enable per-IP rate limiting.  Falls back to
        ``$FP_RATE_LIMIT_ENABLED`` ("0" / "false" to disable), default **True**.
    rate_limit_rpm : int
        Requests per minute per IP.  Falls back to ``$FP_RATE_LIMIT_RPM``,
        default 120.

    Returns
    -------
    WSGI application (may be WSGIApplication or a middleware wrapper)
    """
    from .auth import AuthMiddleware, RateLimitMiddleware

    if library_root is None:
        env_root = os.environ.get("HYPERTENSOR_DATA_ROOT")
        library_root = Path(env_root) if env_root else Path("cases")

    if allowed_origins is None:
        env_origins = os.environ.get("FP_CORS_ORIGINS")
        if env_origins:
            allowed_origins = tuple(o.strip() for o in env_origins.split(","))
        else:
            allowed_origins = ("*",)

    def _env_bool(key: str, default: bool) -> bool:
        val = os.environ.get(key, "").strip().lower()
        if val in ("0", "false", "no", "off"):
            return False
        if val in ("1", "true", "yes", "on"):
            return True
        return default

    if enable_auth is None:
        enable_auth = _env_bool("FP_AUTH_ENABLED", True)
    if enable_rate_limit is None:
        enable_rate_limit = _env_bool("FP_RATE_LIMIT_ENABLED", True)

    env_rpm = os.environ.get("FP_RATE_LIMIT_RPM")
    if env_rpm:
        try:
            rate_limit_rpm = int(env_rpm)
        except ValueError:
            pass

    # Core app
    app: _WSGIApp = WSGIApplication(
        library_root=library_root,
        allowed_origins=allowed_origins,
    )

    # Auth layer
    if enable_auth:
        key_file_str = os.environ.get("FP_KEY_FILE")
        key_file = Path(key_file_str) if key_file_str else None
        auth_mw = AuthMiddleware(app, key_file=key_file)
        app = auth_mw
        logger.info(
            "Auth middleware enabled — key_file=%s",
            key_file or "(in-memory)",
        )
    else:
        logger.warning("Auth middleware DISABLED — API is unauthenticated")

    # Rate-limit layer (outermost)
    if enable_rate_limit:
        app = RateLimitMiddleware(app, rpm=rate_limit_rpm)
        logger.info("Rate limiting enabled — %d rpm/IP", rate_limit_rpm)
    else:
        logger.warning("Rate limiting DISABLED")

    return app
