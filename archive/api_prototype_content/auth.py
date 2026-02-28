"""Ontic API — API-key authentication.

Simple bearer-token scheme.  Keys are configured via
``ONTIC_ENGINE_API_KEYS`` (comma-separated) or auto-generated
at startup.  Send as::

    Authorization: Bearer <key>

Rate limiting is per-key with a token-bucket algorithm.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

_bearer = HTTPBearer(auto_error=False)


# ── Token-bucket rate limiter ───────────────────────────────────────


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    """In-memory per-key token-bucket rate limiter."""

    def __init__(self, rpm: int, burst: int) -> None:
        self._rpm = rpm
        self._burst = burst
        self._refill_rate = rpm / 60.0  # tokens per second
        self._buckets: dict[str, _Bucket] = {}

    def _get_bucket(self, key: str) -> _Bucket:
        if key not in self._buckets:
            self._buckets[key] = _Bucket(
                tokens=float(self._burst), last_refill=time.monotonic()
            )
        return self._buckets[key]

    def allow(self, key: str) -> bool:
        bucket = self._get_bucket(key)
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(
            float(self._burst), bucket.tokens + elapsed * self._refill_rate
        )
        bucket.last_refill = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False


_limiter = RateLimiter(rpm=settings.rate_limit_rpm, burst=settings.rate_limit_burst)


# ── Dependency ──────────────────────────────────────────────────────


async def require_api_key(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(_bearer)
    ] = None,
) -> str:
    """FastAPI dependency that validates the bearer token.

    Returns the authenticated API key string on success.
    Raises 401 / 429 on failure.
    """
    if not settings.require_auth:
        return "anonymous"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.  Use: Authorization: Bearer <api_key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    key = credentials.credentials
    if key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not _limiter.allow(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({settings.rate_limit_rpm} requests/min).",
            headers={"Retry-After": "60"},
        )

    return key
