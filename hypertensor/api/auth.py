"""HyperTensor API — Authentication and rate limiting.

Bearer token auth with per-key token-bucket rate limiting.
"""

from __future__ import annotations

import time

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

_bearer = HTTPBearer(auto_error=False)


class _RateLimiter:
    """In-memory per-key token-bucket rate limiter."""

    def __init__(self, rpm: int, burst: int) -> None:
        self._refill_rate = rpm / 60.0
        self._burst = float(burst)
        self._buckets: dict[str, tuple[float, float]] = {}  # key → (tokens, last)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last = self._buckets.get(key, (self._burst, now))
        tokens = min(self._burst, tokens + (now - last) * self._refill_rate)
        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, now)
            return True
        self._buckets[key] = (tokens, now)
        return False


_limiter = _RateLimiter(rpm=settings.rate_limit_rpm, burst=settings.rate_limit_burst)


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> str:
    """Validate bearer token.  Returns the API key on success."""
    if not settings.require_auth:
        return "anonymous"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "E011",
                "message": "Missing Authorization header.  Use: Authorization: Bearer <api_key>",
                "retryable": False,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    key = credentials.credentials
    if key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "E011",
                "message": "Invalid API key.",
                "retryable": False,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not _limiter.allow(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "E010",
                "message": f"Rate limit exceeded ({settings.rate_limit_rpm} requests/min).",
                "retryable": True,
            },
            headers={"Retry-After": "60"},
        )

    return key
