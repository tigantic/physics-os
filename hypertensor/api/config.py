"""HyperTensor API — Configuration.

All settings read from ``HYPERTENSOR_*`` environment variables,
with sensible defaults for local development.
"""

from __future__ import annotations

import secrets
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Server configuration.  Env prefix: ``HYPERTENSOR_``."""

    model_config = {"env_prefix": "HYPERTENSOR_"}

    # ── Server ──────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False
    log_level: str = "info"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ── Auth ────────────────────────────────────────────────────────
    api_keys: list[str] = Field(
        default_factory=lambda: [secrets.token_urlsafe(32)],
        description="Comma-separated valid API keys.",
    )
    require_auth: bool = True

    # ── Compute ─────────────────────────────────────────────────────
    device: str = "cpu"  # auto-detected at startup
    max_n_bits: int = 14
    max_n_steps: int = 10_000
    max_rank: int = 128
    truncation_tol: float = 1e-10
    job_timeout_s: float = 300.0

    # ── Rate limiting ───────────────────────────────────────────────
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10

    # ── Result delivery ─────────────────────────────────────────────
    max_field_points: int = 500_000
    field_precision: int = 8

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v  # type: ignore[return-value]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v  # type: ignore[return-value]


settings = Settings()
