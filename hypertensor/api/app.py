"""HyperTensor API — FastAPI application factory.

``create_app()`` wires every router, middleware, and lifecycle hook
into a single ``FastAPI`` instance.  It is the only entry-point to
the HTTP surface::

    uvicorn hypertensor.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import hypertensor

from .config import settings

logger = logging.getLogger("hypertensor.api")


# ── Lifecycle ───────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hooks."""
    _configure_logging()
    logger.info(
        "HyperTensor API %s  |  runtime %s  |  schema %s  |  device %s",
        hypertensor.API_VERSION,
        hypertensor.RUNTIME_VERSION,
        hypertensor.SCHEMA_VERSION,
        settings.device,
    )
    if not settings.require_auth:
        logger.warning("Authentication is DISABLED (HYPERTENSOR_REQUIRE_AUTH=false).")
    else:
        logger.info("Auth enabled  |  %d API key(s) loaded.", len(settings.api_keys))
        # Print first key for local dev convenience
        if settings.debug and settings.api_keys:
            logger.info("Dev API key: %s", settings.api_keys[0])
    yield
    logger.info("HyperTensor API shutting down.")


def _configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger("hypertensor")
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
            "code": "E001",
            "message": "Internal server error." if not settings.debug else str(exc),
            "retryable": True,
        },
    )


# ── Factory ─────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Build the HyperTensor API application."""
    app = FastAPI(
        title="HyperTensor Runtime API",
        summary=(
            "Licensed execution access + evidence guarantees "
            "for compression-native physics compute."
        ),
        version=hypertensor.API_VERSION,
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

    # ── Exception handlers ──────────────────────────────────────────
    app.add_exception_handler(RequestValidationError, _validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _generic_error)  # type: ignore[arg-type]

    # ── Routers ─────────────────────────────────────────────────────
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

    # ── Root (convenience) ──────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "HyperTensor Runtime API",
            "version": hypertensor.API_VERSION,
            "docs": "/docs" if settings.debug else "disabled",
        }

    return app


app = create_app()
