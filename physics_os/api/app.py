"""Ontic API — FastAPI application factory.

``create_app()`` wires every router, middleware, and lifecycle hook
into a single ``FastAPI`` instance.  It is the only entry-point to
the HTTP surface::

    uvicorn physics_os.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

import physics_os

from .config import settings

logger = logging.getLogger("physics_os.api")


# ── Request ID middleware ────────────────────────────────────────────


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique ``X-Request-ID`` to every request/response cycle.

    If the client sends ``X-Request-ID``, it is preserved.  Otherwise a
    UUID-4 is generated.  The ID is stored in ``request.state.request_id``
    for use in log lines and error responses, and returned in the
    ``X-Request-ID`` response header.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ── Lifecycle ───────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hooks."""
    _configure_logging()
    logger.info(
        "Ontic API %s  |  runtime %s  |  schema %s  |  device %s",
        physics_os.API_VERSION,
        physics_os.RUNTIME_VERSION,
        physics_os.SCHEMA_VERSION,
        settings.device,
    )
    if not settings.require_auth:
        logger.warning("Authentication is DISABLED (ONTIC_REQUIRE_AUTH=false).")
    else:
        logger.info("Auth enabled  |  %d API key(s) loaded.", len(settings.api_keys))
        # Print masked key prefix for local dev convenience
        if settings.debug and settings.api_keys:
            _k = settings.api_keys[0]
            logger.info("Dev API key: %s...%s", _k[:4], _k[-4:])
    yield
    logger.info("Ontic API shutting down.")


def _configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger("physics_os")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


# ── Exception handlers ──────────────────────────────────────────────


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return machine-readable error for malformed requests."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "code": "E003",
            "message": "Invalid request payload.",
            "details": exc.errors(),
            "retryable": False,
        },
    )


async def _generic_error(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler.  Never leak stack traces in production."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": "E012",
            "message": "Internal server error." if not settings.debug else str(exc),
            "retryable": True,
        },
    )


# ── Factory ─────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Build the Ontic Engine API application."""
    app = FastAPI(
        title="Ontic Runtime API",
        summary=(
            "Licensed execution access + evidence guarantees "
            "for compression-native physics compute."
        ),
        version=physics_os.API_VERSION,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=_lifespan,
    )

    # ── Middleware ───────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    # ── Exception handlers ──────────────────────────────────────────
    app.add_exception_handler(RequestValidationError, _validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _generic_error)  # type: ignore[arg-type]

    # ── Routers ─────────────────────────────────────────────────────
    from .routers.billing import router as billing_router
    from .routers.capabilities import router as capabilities_router
    from .routers.contracts import router as contracts_router
    from .routers.health import router as health_router
    from .routers.jobs import router as jobs_router
    from .routers.validate import router as validate_router

    app.include_router(health_router)
    app.include_router(capabilities_router)
    app.include_router(contracts_router)
    app.include_router(jobs_router)
    app.include_router(validate_router)
    app.include_router(billing_router)

    # ── Root (convenience) ──────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "Ontic Runtime API",
            "version": physics_os.API_VERSION,
            "docs": "/docs" if settings.debug else "disabled",
        }

    return app


app = create_app()
