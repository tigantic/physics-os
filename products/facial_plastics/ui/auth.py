"""Authentication and rate-limiting WSGI middleware.

Provides two composable middleware layers:

* **AuthMiddleware** — validates ``X-API-Key`` / ``Authorization: Bearer``
  headers against a configured key store and maps keys to tenant + role.
* **RateLimitMiddleware** — fixed-window per-IP rate limiter with
  configurable burst size.

Both are WSGI middleware (decorator pattern) that wrap the inner
``WSGIApplication``.

Usage::

    from products.facial_plastics.ui.wsgi import create_app
    from products.facial_plastics.ui.auth import (
        AuthMiddleware, RateLimitMiddleware,
    )

    app = create_app()
    app = AuthMiddleware(app, key_file=Path("/etc/fp/keys.json"))
    app = RateLimitMiddleware(app, rpm=120)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────

_Environ = Dict[str, Any]
_StartResponse = Callable[..., Any]
_WSGIApp = Callable[[_Environ, _StartResponse], Iterable[bytes]]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  API Key Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class APIKeyRecord:
    """A registered API key and its associated identity."""

    key_hash: str
    tenant_id: str
    role: str
    description: str = ""
    created_at: float = 0.0
    is_active: bool = True


class KeyStore:
    """Manages API key records.

    Keys are stored hashed (SHA-256).  The plaintext key is only shown
    once at generation time and never persisted.

    Parameters
    ----------
    key_file : Path | None
        JSON file to load/persist keys from.  If ``None``, the store
        is in-memory only.
    """

    def __init__(self, key_file: Optional[Path] = None) -> None:
        self._keys: Dict[str, APIKeyRecord] = {}  # hash → record
        self._key_file = key_file
        self._lock = threading.Lock()

        if key_file and key_file.exists():
            self._load()

    # ── Key operations ────────────────────────────────────────────

    def generate_key(
        self,
        tenant_id: str,
        role: str,
        description: str = "",
    ) -> Tuple[str, APIKeyRecord]:
        """Generate a new API key.

        Returns
        -------
        tuple[str, APIKeyRecord]
            The plaintext key (show once) and its record.
        """
        plaintext = f"fp_{secrets.token_urlsafe(32)}"
        key_hash = self._hash(plaintext)
        record = APIKeyRecord(
            key_hash=key_hash,
            tenant_id=tenant_id,
            role=role,
            description=description,
            created_at=time.time(),
            is_active=True,
        )
        with self._lock:
            self._keys[key_hash] = record
            self._persist()
        logger.info(
            "Generated API key for tenant=%s role=%s desc=%s",
            tenant_id,
            role,
            description,
        )
        return plaintext, record

    def validate(self, plaintext: str) -> Optional[APIKeyRecord]:
        """Validate a plaintext key.  Returns ``None`` if invalid."""
        key_hash = self._hash(plaintext)
        with self._lock:
            record = self._keys.get(key_hash)
        if record and record.is_active:
            return record
        return None

    def revoke(self, key_hash: str) -> bool:
        """Revoke a key by its hash."""
        with self._lock:
            record = self._keys.get(key_hash)
            if not record:
                return False
            self._keys[key_hash] = APIKeyRecord(
                key_hash=record.key_hash,
                tenant_id=record.tenant_id,
                role=record.role,
                description=record.description,
                created_at=record.created_at,
                is_active=False,
            )
            self._persist()
        logger.info("Revoked key %s…", key_hash[:12])
        return True

    def list_keys(self) -> List[APIKeyRecord]:
        """List all (active) key records — hashes only, no plaintext."""
        with self._lock:
            return [r for r in self._keys.values() if r.is_active]

    # ── Persistence ───────────────────────────────────────────────

    def _load(self) -> None:
        if not self._key_file or not self._key_file.exists():
            return
        try:
            data = json.loads(self._key_file.read_text("utf-8"))
            for entry in data.get("keys", []):
                rec = APIKeyRecord(
                    key_hash=entry["key_hash"],
                    tenant_id=entry["tenant_id"],
                    role=entry["role"],
                    description=entry.get("description", ""),
                    created_at=entry.get("created_at", 0.0),
                    is_active=entry.get("is_active", True),
                )
                self._keys[rec.key_hash] = rec
            logger.info("Loaded %d API keys from %s", len(self._keys), self._key_file)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.error("Failed to load key file: %s", exc)

    def _persist(self) -> None:
        if not self._key_file:
            return
        try:
            self._key_file.parent.mkdir(parents=True, exist_ok=True)
            entries = [
                {
                    "key_hash": r.key_hash,
                    "tenant_id": r.tenant_id,
                    "role": r.role,
                    "description": r.description,
                    "created_at": r.created_at,
                    "is_active": r.is_active,
                }
                for r in self._keys.values()
            ]
            self._key_file.write_text(
                json.dumps({"keys": entries}, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to persist key file: %s", exc)

    @staticmethod
    def _hash(plaintext: str) -> str:
        return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Auth Middleware
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AuthMiddleware:
    """WSGI middleware that validates API keys before passing to the app.

    If a valid key is found, ``environ`` is enriched with:

    * ``fp.tenant_id`` — the tenant ID associated with the key
    * ``fp.role`` — the role string
    * ``fp.key_hash`` — for audit correlation

    Unauthenticated requests to ``/api/*`` receive 401.
    Static assets, ``/metrics``, and ``OPTIONS`` are exempt.

    Parameters
    ----------
    inner_app : WSGI app
        The downstream application.
    key_file : Path | None
        Key store JSON file.
    exempt_paths : tuple[str, ...]
        Path prefixes that skip auth.
    """

    def __init__(
        self,
        inner_app: _WSGIApp,
        *,
        key_file: Optional[Path] = None,
        exempt_paths: Tuple[str, ...] = ("/metrics", "/health"),
    ) -> None:
        self._inner = inner_app
        self._exempt_paths = exempt_paths
        env_key_file = os.environ.get("FP_KEY_FILE")
        if key_file is None and env_key_file:
            key_file = Path(env_key_file)
        self._store = KeyStore(key_file=key_file)

    @property
    def key_store(self) -> KeyStore:
        """Expose the key store for admin operations."""
        return self._store

    def __call__(
        self,
        environ: _Environ,
        start_response: _StartResponse,
    ) -> Iterable[bytes]:
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")

        # Skip auth for OPTIONS, exempt paths, and static assets
        if method == "OPTIONS":
            return self._inner(environ, start_response)

        for exempt in self._exempt_paths:
            if path.startswith(exempt):
                return self._inner(environ, start_response)

        if not path.startswith("/api/"):
            return self._inner(environ, start_response)

        # Extract key from header
        key = self._extract_key(environ)
        if key is None:
            return self._unauthorized(start_response, "Missing API key")

        record = self._store.validate(key)
        if record is None:
            return self._unauthorized(start_response, "Invalid API key")

        # Enrich environ for downstream use
        environ["fp.tenant_id"] = record.tenant_id
        environ["fp.role"] = record.role
        environ["fp.key_hash"] = record.key_hash

        return self._inner(environ, start_response)

    @staticmethod
    def _extract_key(environ: _Environ) -> Optional[str]:
        """Extract API key from X-API-Key or Authorization: Bearer."""
        api_key = environ.get("HTTP_X_API_KEY")
        if api_key:
            return str(api_key)

        auth = str(environ.get("HTTP_AUTHORIZATION", ""))
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()

        return None

    @staticmethod
    def _unauthorized(
        start_response: _StartResponse,
        message: str,
    ) -> Iterable[bytes]:
        body = json.dumps({"error": message}).encode("utf-8")
        start_response("401 Unauthorized", [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("WWW-Authenticate", 'Bearer realm="facial-plastics"'),
        ])
        return [body]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Rate Limiting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class _RateBucket:
    """Fixed-window rate limit bucket."""

    window_start: float = 0.0
    count: int = 0


class RateLimitMiddleware:
    """WSGI middleware — fixed-window per-IP rate limiter.

    Parameters
    ----------
    inner_app : WSGI app
        The downstream application.
    rpm : int
        Maximum requests per minute per client IP.
    """

    def __init__(
        self,
        inner_app: _WSGIApp,
        *,
        rpm: int = 120,
    ) -> None:
        self._inner = inner_app
        self._rpm = rpm
        self._window_seconds = 60.0
        self._buckets: Dict[str, _RateBucket] = {}
        self._lock = threading.Lock()

    def __call__(
        self,
        environ: _Environ,
        start_response: _StartResponse,
    ) -> Iterable[bytes]:
        client_ip = self._get_client_ip(environ)
        now = time.monotonic()

        with self._lock:
            bucket = self._buckets.get(client_ip)
            if bucket is None:
                bucket = _RateBucket(window_start=now, count=0)
                self._buckets[client_ip] = bucket

            # Rotate window
            if now - bucket.window_start >= self._window_seconds:
                bucket.window_start = now
                bucket.count = 0

            bucket.count += 1
            current_count = bucket.count

        if current_count > self._rpm:
            remaining = max(
                0.0,
                self._window_seconds - (now - bucket.window_start),
            )
            return self._too_many_requests(start_response, remaining)

        return self._inner(environ, start_response)

    @staticmethod
    def _get_client_ip(environ: _Environ) -> str:
        """Extract client IP, respecting X-Forwarded-For behind a proxy."""
        forwarded = environ.get("HTTP_X_FORWARDED_FOR")
        if forwarded:
            return str(forwarded).split(",")[0].strip()
        return str(environ.get("REMOTE_ADDR", "127.0.0.1"))

    @staticmethod
    def _too_many_requests(
        start_response: _StartResponse,
        retry_after: float,
    ) -> Iterable[bytes]:
        body = json.dumps({
            "error": "Rate limit exceeded",
            "retry_after_seconds": round(retry_after, 1),
        }).encode("utf-8")
        start_response("429 Too Many Requests", [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Retry-After", str(int(retry_after) + 1)),
        ])
        return [body]
