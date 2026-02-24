"""Shadow invoice generation.

Produces shadow invoices per PRICING_MODEL.md §5.1.
During alpha, invoices are generated but NOT charged or sent.
They validate metering accuracy and generation pipeline.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .meter import MeterRecord, UsageLedger, ledger as _default_ledger

# ── Alpha package (all users are Explorer) ──────────────────────────

_ALPHA_PACKAGE = "explorer"
_ALPHA_INCLUDED_CU = 100.0
_ALPHA_BASE_PRICE = 0.0
_ALPHA_OVERAGE_RATE = 0.0   # No overage charges in alpha
_ALPHA_UNIT_PRICE = 0.05    # Shadow unit price for line items


# ── Invoice model ───────────────────────────────────────────────────


def generate_invoice(
    *,
    api_key_suffix: str,
    period: str,
    ledger: UsageLedger | None = None,
) -> dict[str, Any]:
    """Generate a shadow invoice for the given API key and period.

    Parameters
    ----------
    api_key_suffix : str
        Last 8 characters of the API key.
    period : str
        Billing period in ``YYYY-MM`` format (e.g. ``"2025-07"``).
    ledger : UsageLedger, optional
        Ledger to pull records from.  Defaults to the singleton.

    Returns
    -------
    dict
        Shadow invoice matching the format in PRICING_MODEL.md §5.1.
    """
    _ledger = ledger or _default_ledger

    records = _ledger.get_records(api_key_suffix=api_key_suffix)

    # Filter to the requested billing period
    period_records = [
        r for r in records
        if r.timestamp[:7] == period
    ]

    # Build line items
    line_items: list[dict[str, Any]] = []
    for r in period_records:
        line_items.append({
            "date": r.timestamp[:10],
            "job_id": r.job_id,
            "domain": r.domain,
            "device_class": r.device_class,
            "wall_time_s": r.wall_time_s,
            "compute_units": round(r.compute_units, 2),
            "unit_price_usd": _ALPHA_UNIT_PRICE,
        })

    total_cu = round(sum(li["compute_units"] for li in line_items), 2)
    overage_cu = round(max(0.0, total_cu - _ALPHA_INCLUDED_CU), 2)
    overage_usd = round(overage_cu * _ALPHA_OVERAGE_RATE, 2)
    total_usd = round(_ALPHA_BASE_PRICE + overage_usd, 2)

    return {
        "invoice_id": f"INV-{period}-{api_key_suffix}",
        "period": period,
        "api_key_suffix": api_key_suffix,
        "line_items": line_items,
        "total_cu": total_cu,
        "total_usd": total_usd,
        "package": _ALPHA_PACKAGE,
        "included_cu": _ALPHA_INCLUDED_CU,
        "overage_cu": overage_cu,
        "overage_usd": overage_usd,
        "shadow": True,
    }


def export_invoice_json(invoice: dict[str, Any]) -> str:
    """Serialize an invoice to deterministic JSON.

    Returns
    -------
    str
        Compact JSON string, sorted keys.
    """
    return json.dumps(invoice, sort_keys=True, indent=2, ensure_ascii=False)
