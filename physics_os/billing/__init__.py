"""Billing and metering for the Ontic Engine.

Dual-mode billing:
  - **shadow**: CU accounting only, no charges (default for alpha)
  - **live**: Stripe metered billing with real charges

Pipeline: job execution → meter.record() → stripe_billing.report_usage()
"""

from .meter import MeterRecord, UsageLedger, calculate_cu, ledger
from .stripe_billing import (
    BillingMode,
    Customer,
    StripeBilling,
    StripeConfig,
    Tier,
    TierSpec,
    get_billing,
)

__all__ = [
    "BillingMode",
    "Customer",
    "MeterRecord",
    "StripeBilling",
    "StripeConfig",
    "Tier",
    "TierSpec",
    "UsageLedger",
    "calculate_cu",
    "get_billing",
    "ledger",
]
