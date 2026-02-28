"""Billing shadow tests — G9.1, G9.2, G9.4.

G9.1: Billing calculator on every job — ``calculate_cu()`` returns
       correct CU for CPU + GPU, respects minimum floor.

G9.2: Usage ledger records — ``UsageLedger`` records correctly,
       filters by key/date, totals accurately, is thread-safe.

G9.4: Invoice export available — ``generate_invoice()`` produces
       valid shadow invoice JSON matching PRICING_MODEL.md §5.1.
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import pytest

from physics_os.billing.meter import (
    MeterRecord,
    UsageLedger,
    calculate_cu,
)
from physics_os.billing.invoice import (
    export_invoice_json,
    generate_invoice,
)


# ════════════════════════════════════════════════════════════════════
# G9.1 — Billing calculator on every job
# ════════════════════════════════════════════════════════════════════


class TestG9_1_BillingCalculator:
    """CU calculation per METERING_POLICY.md §1.1."""

    def test_cpu_multiplier(self) -> None:
        assert calculate_cu(1.0, "cpu") == 1.0

    def test_cuda_multiplier(self) -> None:
        assert calculate_cu(1.0, "cuda") == 10.0

    def test_fractional_cpu(self) -> None:
        assert calculate_cu(0.3456, "cpu") == 0.3456

    def test_fractional_cuda(self) -> None:
        assert calculate_cu(0.3456, "cuda") == 3.456

    def test_minimum_cu_floor(self) -> None:
        """Very short jobs (< 0.01s) are floored to 0.01 CU."""
        assert calculate_cu(0.001, "cpu") == 0.01
        assert calculate_cu(0.0, "cpu") == 0.01

    def test_zero_wall_time_cuda(self) -> None:
        assert calculate_cu(0.0, "cuda") == 0.01

    def test_unknown_device_defaults_to_cpu(self) -> None:
        """Unknown device classes use multiplier 1.0."""
        assert calculate_cu(2.0, "tpu") == 2.0

    def test_rounding_precision(self) -> None:
        """CU rounded to 4 decimal places."""
        cu = calculate_cu(1.23456789, "cpu")
        assert cu == 1.2346

    def test_large_wall_time(self) -> None:
        """300s job on GPU = 3000 CU."""
        assert calculate_cu(300.0, "cuda") == 3000.0


# ════════════════════════════════════════════════════════════════════
# G9.2 — Usage ledger records
# ════════════════════════════════════════════════════════════════════


class TestG9_2_UsageLedger:
    """In-memory ledger correctness and thread safety."""

    def setup_method(self) -> None:
        self.ledger = UsageLedger()

    def test_record_and_retrieve(self) -> None:
        rec = self.ledger.record(
            job_id="j1", api_key_suffix="suffix1",
            domain="burgers", device_class="cpu", wall_time_s=1.0,
        )
        assert isinstance(rec, MeterRecord)
        assert rec.compute_units == 1.0
        records = self.ledger.get_records()
        assert len(records) == 1
        assert records[0].job_id == "j1"

    def test_filter_by_key(self) -> None:
        self.ledger.record(
            job_id="j1", api_key_suffix="aaa",
            domain="burgers", device_class="cpu", wall_time_s=1.0,
        )
        self.ledger.record(
            job_id="j2", api_key_suffix="bbb",
            domain="maxwell", device_class="cpu", wall_time_s=2.0,
        )
        filtered = self.ledger.get_records(api_key_suffix="aaa")
        assert len(filtered) == 1
        assert filtered[0].job_id == "j1"

    def test_filter_by_since(self) -> None:
        self.ledger.record(
            job_id="j1", api_key_suffix="aaa",
            domain="burgers", device_class="cpu", wall_time_s=1.0,
        )
        # Records created now should pass a "since" filter from yesterday
        filtered = self.ledger.get_records(since="2000-01-01")
        assert len(filtered) == 1

    def test_total_cu(self) -> None:
        for i in range(5):
            self.ledger.record(
                job_id=f"j{i}", api_key_suffix="aaa",
                domain="burgers", device_class="cpu", wall_time_s=1.0,
            )
        assert self.ledger.total_cu() == 5.0
        assert self.ledger.total_cu(api_key_suffix="aaa") == 5.0
        assert self.ledger.total_cu(api_key_suffix="zzz") == 0.0

    def test_count(self) -> None:
        for i in range(3):
            self.ledger.record(
                job_id=f"j{i}", api_key_suffix="x",
                domain="burgers", device_class="cpu", wall_time_s=0.5,
            )
        assert self.ledger.count() == 3

    def test_clear(self) -> None:
        self.ledger.record(
            job_id="j1", api_key_suffix="x",
            domain="burgers", device_class="cpu", wall_time_s=1.0,
        )
        assert self.ledger.count() == 1
        self.ledger.clear()
        assert self.ledger.count() == 0

    def test_thread_safety(self) -> None:
        """Concurrent writes don't lose records."""
        errors: list[str] = []

        def _write(n: int) -> None:
            try:
                for i in range(50):
                    self.ledger.record(
                        job_id=f"thread{n}-job{i}",
                        api_key_suffix="ts",
                        domain="burgers",
                        device_class="cpu",
                        wall_time_s=0.1,
                    )
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=_write, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert self.ledger.count() == 200  # 4 threads × 50

    def test_record_immutability(self) -> None:
        """MeterRecord is frozen — cannot be modified after creation."""
        rec = self.ledger.record(
            job_id="j1", api_key_suffix="x",
            domain="burgers", device_class="cpu", wall_time_s=1.0,
        )
        with pytest.raises(AttributeError):
            rec.compute_units = 999.0  # type: ignore[misc]

    def test_timestamp_is_utc(self) -> None:
        rec = self.ledger.record(
            job_id="j1", api_key_suffix="x",
            domain="burgers", device_class="cpu", wall_time_s=1.0,
        )
        assert rec.timestamp.endswith("+00:00") or rec.timestamp.endswith("Z")


# ════════════════════════════════════════════════════════════════════
# G9.4 — Invoice export available
# ════════════════════════════════════════════════════════════════════


class TestG9_4_InvoiceExport:
    """Shadow invoice generation per PRICING_MODEL.md §5.1."""

    def setup_method(self) -> None:
        self.ledger = UsageLedger()

    def _populate(self, n: int = 5, suffix: str = "key1") -> None:
        for i in range(n):
            self.ledger.record(
                job_id=str(uuid.uuid4()),
                api_key_suffix=suffix,
                domain="burgers",
                device_class="cpu",
                wall_time_s=1.0 + i * 0.1,
            )

    def test_invoice_structure(self) -> None:
        """Invoice has all required top-level fields."""
        self._populate()
        now = datetime.now(timezone.utc)
        period = now.strftime("%Y-%m")
        inv = generate_invoice(
            api_key_suffix="key1", period=period, ledger=self.ledger,
        )
        required = {
            "invoice_id", "period", "api_key_suffix", "line_items",
            "total_cu", "total_usd", "package", "included_cu",
            "overage_cu", "overage_usd", "shadow",
        }
        assert required.issubset(inv.keys()), (
            f"Missing fields: {required - inv.keys()}"
        )

    def test_invoice_shadow_flag(self) -> None:
        """Alpha invoices always have ``shadow: true``."""
        self._populate()
        now = datetime.now(timezone.utc)
        period = now.strftime("%Y-%m")
        inv = generate_invoice(
            api_key_suffix="key1", period=period, ledger=self.ledger,
        )
        assert inv["shadow"] is True

    def test_invoice_line_items_match_records(self) -> None:
        """Each metered job appears as a line item."""
        self._populate(n=3)
        now = datetime.now(timezone.utc)
        period = now.strftime("%Y-%m")
        inv = generate_invoice(
            api_key_suffix="key1", period=period, ledger=self.ledger,
        )
        assert len(inv["line_items"]) == 3

    def test_line_item_fields(self) -> None:
        """Each line item has the required fields."""
        self._populate(n=1)
        now = datetime.now(timezone.utc)
        period = now.strftime("%Y-%m")
        inv = generate_invoice(
            api_key_suffix="key1", period=period, ledger=self.ledger,
        )
        li = inv["line_items"][0]
        required = {"date", "job_id", "domain", "device_class",
                     "wall_time_s", "compute_units", "unit_price_usd"}
        assert required.issubset(li.keys())

    def test_total_cu_matches_sum(self) -> None:
        """Total CU equals sum of line-item CU."""
        self._populate(n=5)
        now = datetime.now(timezone.utc)
        period = now.strftime("%Y-%m")
        inv = generate_invoice(
            api_key_suffix="key1", period=period, ledger=self.ledger,
        )
        line_sum = sum(li["compute_units"] for li in inv["line_items"])
        assert inv["total_cu"] == round(line_sum, 2)

    def test_no_overage_within_included(self) -> None:
        """Explorer package has 100 CU included; 5 CU => no overage."""
        self._populate(n=5)  # ~5 CU total
        now = datetime.now(timezone.utc)
        period = now.strftime("%Y-%m")
        inv = generate_invoice(
            api_key_suffix="key1", period=period, ledger=self.ledger,
        )
        assert inv["overage_cu"] == 0.0
        assert inv["overage_usd"] == 0.0

    def test_empty_invoice(self) -> None:
        """Invoice for a key with no jobs has empty line items."""
        inv = generate_invoice(
            api_key_suffix="nobody", period="2025-07", ledger=self.ledger,
        )
        assert inv["line_items"] == []
        assert inv["total_cu"] == 0.0

    def test_period_filter(self) -> None:
        """Only jobs from the requested period appear."""
        self._populate(n=3)
        inv = generate_invoice(
            api_key_suffix="key1", period="1999-01", ledger=self.ledger,
        )
        assert len(inv["line_items"]) == 0

    def test_export_json(self) -> None:
        """Invoice exports to valid, deterministic JSON."""
        self._populate(n=2)
        now = datetime.now(timezone.utc)
        period = now.strftime("%Y-%m")
        inv = generate_invoice(
            api_key_suffix="key1", period=period, ledger=self.ledger,
        )
        exported = export_invoice_json(inv)
        parsed = json.loads(exported)
        assert parsed["invoice_id"] == inv["invoice_id"]
        assert parsed["shadow"] is True

    def test_invoice_id_format(self) -> None:
        """Invoice ID follows INV-YYYY-MM-suffix format."""
        inv = generate_invoice(
            api_key_suffix="abcd1234", period="2025-07", ledger=self.ledger,
        )
        assert inv["invoice_id"] == "INV-2025-07-abcd1234"

    def test_package_is_explorer(self) -> None:
        """All alpha invoices use Explorer package."""
        inv = generate_invoice(
            api_key_suffix="x", period="2025-07", ledger=self.ledger,
        )
        assert inv["package"] == "explorer"
        assert inv["included_cu"] == 100.0
