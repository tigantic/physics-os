"""HTTP server — stdlib-based JSON API + static SPA serving.

No Flask, FastAPI, or any third-party dependency.  Uses only
``http.server`` and ``json`` from the standard library.

Usage::

    python -m products.facial_plastics.ui.server --port 8420 --library ./cases

Or programmatically::

    from products.facial_plastics.ui.server import start_server
    start_server(port=8420, library_root=Path("./cases"))
"""

from __future__ import annotations

import json
import logging
import mimetypes
import sys
import traceback
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from .api import UIApplication

logger = logging.getLogger(__name__)

# Static asset directory
_STATIC_DIR = Path(__file__).parent / "static"


class _RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler that dispatches to UIApplication methods."""

    app: UIApplication  # set by the factory

    # ── HTTP methods ──────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        # API routes
        if path.startswith("/api/"):
            self._handle_api_get(path, query)
        else:
            self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        self._handle_api_post(path, body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Handle CORS preflight."""
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    # ── API dispatch ──────────────────────────────────────────────

    def _handle_api_get(
        self,
        path: str,
        query: Dict[str, list],
    ) -> None:
        """Dispatch GET /api/* requests."""
        try:
            result = self._dispatch_get(path, query)
            self._send_json(result)
        except Exception as exc:
            logger.error("API error: %s\n%s", exc, traceback.format_exc())
            self._send_json({"error": str(exc)}, status=500)

    def _handle_api_post(
        self,
        path: str,
        body: Dict[str, Any],
    ) -> None:
        """Dispatch POST /api/* requests."""
        try:
            result = self._dispatch_post(path, body)
            self._send_json(result)
        except Exception as exc:
            logger.error("API error: %s\n%s", exc, traceback.format_exc())
            self._send_json({"error": str(exc)}, status=500)

    def _dispatch_get(
        self,
        path: str,
        query: Dict[str, list],
    ) -> Dict[str, Any]:
        """Route GET requests to UIApplication methods."""
        app = self.app
        _q = lambda key, default=None: query.get(key, [default])[0]  # noqa: E731

        if path == "/api/contract":
            return app.get_contract()

        if path == "/api/cases":
            return app.list_cases(
                procedure=_q("procedure"),
                quality=_q("quality"),
                limit=int(_q("limit", "100")),
                offset=int(_q("offset", "0")),
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
        app = self.app

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

    # ── Static file serving ───────────────────────────────────────

    def _serve_static(self, path: str) -> None:
        """Serve static files for the SPA."""
        if path in ("", "/"):
            path = "/index.html"

        file_path = _STATIC_DIR / path.lstrip("/")

        # Security: prevent path traversal
        try:
            file_path = file_path.resolve()
            if not str(file_path).startswith(str(_STATIC_DIR.resolve())):
                self._send_error(403, "Forbidden")
                return
        except (ValueError, OSError):
            self._send_error(403, "Forbidden")
            return

        if not file_path.is_file():
            # SPA fallback: serve index.html for client-side routing
            file_path = _STATIC_DIR / "index.html"
            if not file_path.is_file():
                self._send_error(404, "Not Found")
                return

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

        try:
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            self._send_error(500, "Read error")

    # ── Response helpers ──────────────────────────────────────────

    def _send_json(
        self,
        data: Dict[str, Any],
        *,
        status: int = 200,
    ) -> None:
        """Send a JSON response."""
        body = json.dumps(data, indent=None, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        """Override to use Python logging instead of stderr."""
        logger.debug(format, *args)


# ── Public entry point ────────────────────────────────────────────

def start_server(
    *,
    port: int = 8420,
    library_root: Path = Path("cases"),
    host: str = "127.0.0.1",
) -> HTTPServer:
    """Start the HTTP server.

    Parameters
    ----------
    port : int
        TCP port (default 8420).
    library_root : Path
        Root directory for the CaseLibrary.
    host : str
        Bind address (default localhost).

    Returns
    -------
    HTTPServer
        The running server instance. Call ``.serve_forever()`` to block,
        or ``.shutdown()`` from another thread to stop.
    """
    library_root = Path(library_root)
    library_root.mkdir(parents=True, exist_ok=True)

    app = UIApplication(library_root)

    handler_class = type(
        "_BoundHandler",
        (_RequestHandler,),
        {"app": app},
    )

    server = HTTPServer((host, port), handler_class)
    logger.info(
        "HyperTensor Facial Plastics UI — http://%s:%d  (library: %s)",
        host, port, library_root,
    )
    return server


# ── CLI entry ─────────────────────────────────────────────────────

def _main() -> None:
    """Run as ``python -m products.facial_plastics.ui.server``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="HyperTensor Facial Plastics UI server",
    )
    parser.add_argument("--port", type=int, default=8420, help="TCP port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address")
    parser.add_argument(
        "--library", type=str, default="cases",
        help="Case library root directory",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    server = start_server(
        port=args.port,
        host=args.host,
        library_root=Path(args.library),
    )

    try:
        print(f"Serving on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    _main()
