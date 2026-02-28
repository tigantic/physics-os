"""Ontic Physics API — Application factory.

Launch with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Or via the convenience script::

    python -m api.main
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import domains, health, simulate

logger = logging.getLogger("physics_os.api")


def _detect_device() -> str:
    """Auto-detect CUDA availability and update settings."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("GPU detected: %s", name)
            return "cuda"
    except ImportError:
        pass
    logger.info("No GPU detected — running on CPU")
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle."""
    # ── Startup ─────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    settings.device = _detect_device()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║        Ontic Physics API  v1.0.0          ║")
    logger.info("╠══════════════════════════════════════════════════╣")
    logger.info("║  Device    : %-35s ║", settings.device)
    logger.info("║  Auth      : %-35s ║", "enabled" if settings.require_auth else "disabled")
    logger.info("║  Max bits  : %-35d ║", settings.max_n_bits)
    logger.info("║  Max rank  : %-35d ║", settings.max_rank)
    logger.info("║  Rate limit: %-35s ║", f"{settings.rate_limit_rpm} req/min")
    logger.info("╚══════════════════════════════════════════════════╝")

    if settings.require_auth:
        for i, key in enumerate(settings.api_keys):
            logger.info("API key [%d]: %s", i, key)

    yield

    # ── Shutdown ────────────────────────────────────────────────────
    logger.info("Ontic API shutting down.")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="Ontic Physics API",
        description=(
            "Production inference API for the QTT Physics VM.  "
            "Solve physics problems across 7 domains with polylogarithmic "
            "complexity.  Returns physical observables and conservation "
            "diagnostics — no internal solver state is ever exposed."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        license_info={
            "name": "Proprietary",
            "url": "https://github.com/tigantic/physics-os",
        },
        contact={
            "name": "The Ontic Engine",
            "url": "https://github.com/tigantic/physics-os",
        },
    )

    # ── CORS ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ─────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(domains.router)
    app.include_router(simulate.router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "Ontic Physics API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/v1/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
        reload=settings.debug,
    )
