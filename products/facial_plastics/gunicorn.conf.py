"""Gunicorn configuration for Facial Plastics production deployment.

Usage::

    gunicorn "products.facial_plastics.ui.wsgi:create_app()" \\
        --config products/facial_plastics/gunicorn.conf.py

All values can be overridden via environment variables prefixed with
``GUNICORN_`` (gunicorn native) or the explicit env vars listed below.
"""

from __future__ import annotations

import multiprocessing
import os

# ── Binding ───────────────────────────────────────────────────────

bind = os.environ.get("FP_BIND", "0.0.0.0:8420")

# ── Workers ───────────────────────────────────────────────────────
# Default: min(2 * CPU + 1, 8) — capped so small boxes aren't overwhelmed.

_default_workers = min(2 * multiprocessing.cpu_count() + 1, 8)
workers = int(os.environ.get("FP_WORKERS", str(_default_workers)))
worker_class = "sync"

# ── Timeouts ──────────────────────────────────────────────────────
# Large simulation sweeps can take minutes.

timeout = int(os.environ.get("FP_TIMEOUT", "300"))
graceful_timeout = 30
keepalive = 5

# ── Logging ───────────────────────────────────────────────────────

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("FP_LOG_LEVEL", "info")
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sμs'
)

# ── Process naming ────────────────────────────────────────────────

proc_name = "facial-plastics"

# ── Security ──────────────────────────────────────────────────────

limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# ── Server hooks ──────────────────────────────────────────────────


def on_starting(server: object) -> None:
    """Log startup configuration."""
    import logging

    logger = logging.getLogger("gunicorn.error")
    logger.info(
        "Facial Plastics starting — workers=%s, bind=%s, timeout=%ss",
        workers,
        bind,
        timeout,
    )


def post_fork(server: object, worker: object) -> None:
    """Per-worker initialisation."""
    import logging

    logger = logging.getLogger("gunicorn.error")
    logger.info("Worker spawned: pid=%s", os.getpid())


def worker_exit(server: object, worker: object) -> None:
    """Cleanup on worker exit."""
    import logging

    logger = logging.getLogger("gunicorn.error")
    logger.info("Worker exiting: pid=%s", os.getpid())
