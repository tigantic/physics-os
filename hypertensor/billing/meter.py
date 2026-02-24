"""Compute-unit metering and usage ledger.

The meter captures CU consumption for every successful job.

CU formula (from METERING_POLICY.md §1.1)::

    CU = wall_time_s × device_multiplier
    where device_multiplier = 10.0 if cuda else 1.0
    minimum CU per job = 0.01

Measurement point:
    ``sanitized_result["performance"]["wall_time_s"]``
    (recorded AFTER sanitization per METERING_POLICY.md §3.1)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Device multipliers (from PRICING_MODEL.md §2.1) ────────────────

_DEVICE_MULTIPLIERS: dict[str, float] = {
    "cpu": 1.0,
    "cuda": 10.0,
}

_MINIMUM_CU = 0.01


# ── Meter record ────────────────────────────────────────────────────


@dataclass(frozen=True)
class MeterRecord:
    """Immutable record of compute consumption for a single job."""

    job_id: str
    api_key_suffix: str
    domain: str
    device_class: str
    wall_time_s: float
    compute_units: float
    timestamp: str  # ISO-8601 UTC


# ── CU calculator ──────────────────────────────────────────────────


def calculate_cu(wall_time_s: float, device_class: str) -> float:
    """Compute the CU consumed by a job.

    Parameters
    ----------
    wall_time_s : float
        Wall-clock execution time from the sanitized result.
    device_class : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    float
        Compute units, rounded to 4 decimal places,
        with a floor of ``_MINIMUM_CU``.
    """
    multiplier = _DEVICE_MULTIPLIERS.get(device_class, 1.0)
    cu = wall_time_s * multiplier
    return round(max(cu, _MINIMUM_CU), 4)


# ── Usage ledger ────────────────────────────────────────────────────


class UsageLedger:
    """Thread-safe, append-only metering ledger.

    In alpha this is in-memory.  Post-alpha: swap for a persistent
    append-only store (Postgres, SQLite, etc.).
    """

    def __init__(self) -> None:
        self._records: list[MeterRecord] = []
        self._lock = threading.Lock()

    # ── Write ───────────────────────────────────────────────────────

    def record(
        self,
        *,
        job_id: str,
        api_key_suffix: str,
        domain: str,
        device_class: str,
        wall_time_s: float,
    ) -> MeterRecord:
        """Record metered usage for a completed job.

        Returns the created ``MeterRecord``.
        """
        cu = calculate_cu(wall_time_s, device_class)
        entry = MeterRecord(
            job_id=job_id,
            api_key_suffix=api_key_suffix,
            domain=domain,
            device_class=device_class,
            wall_time_s=round(wall_time_s, 4),
            compute_units=cu,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._records.append(entry)
        logger.info(
            "Metered  |  job=%s  cu=%.4f  domain=%s  device=%s",
            job_id, cu, domain, device_class,
        )
        return entry

    # ── Read ────────────────────────────────────────────────────────

    def get_records(
        self,
        *,
        api_key_suffix: str | None = None,
        since: str | None = None,
    ) -> list[MeterRecord]:
        """Return metering records, optionally filtered by key and/or date.

        Parameters
        ----------
        api_key_suffix : str, optional
            Filter to records for this API key suffix.
        since : str, optional
            ISO-8601 date/datetime.  Return only records on or after this.
        """
        with self._lock:
            snapshot = list(self._records)

        if api_key_suffix:
            snapshot = [r for r in snapshot if r.api_key_suffix == api_key_suffix]

        if since:
            snapshot = [r for r in snapshot if r.timestamp >= since]

        return snapshot

    def total_cu(self, *, api_key_suffix: str | None = None) -> float:
        """Sum of CU consumed, optionally scoped to one API key."""
        records = self.get_records(api_key_suffix=api_key_suffix)
        return round(sum(r.compute_units for r in records), 4)

    def count(self) -> int:
        """Total number of metering records."""
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        """Clear all records.  For testing only."""
        with self._lock:
            self._records.clear()


# ── Singleton ───────────────────────────────────────────────────────

ledger = UsageLedger()
