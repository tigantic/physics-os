"""Stripe billing integration.

Connects the existing CU metering pipeline to Stripe for real charges.

Architecture:
    MeterRecord → stripe_billing.report_usage() → Stripe Usage Records
    Monthly    → stripe_billing.generate_invoice() → Stripe Invoice
    Webhook    → stripe_billing.handle_webhook() → Event processing

Configuration (env vars):
    ONTIC_STRIPE_SECRET_KEY        Stripe secret key (sk_live_... or sk_test_...)
    ONTIC_STRIPE_WEBHOOK_SECRET    Webhook endpoint signing secret (whsec_...)
    ONTIC_STRIPE_PRICE_BUILDER     Stripe Price ID for Builder tier metered usage
    ONTIC_STRIPE_PRICE_PRO         Stripe Price ID for Professional tier metered usage
    ONTIC_BILLING_MODE             "shadow" (default) | "live"
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────


class BillingMode(str, Enum):
    """Billing mode: shadow (meter only) or live (charge via Stripe)."""
    SHADOW = "shadow"
    LIVE = "live"


@dataclass(frozen=True)
class StripeConfig:
    """Stripe billing configuration from environment variables."""
    secret_key: str
    webhook_secret: str
    price_id_builder: str
    price_id_pro: str
    billing_mode: BillingMode

    @classmethod
    def from_env(cls) -> "StripeConfig":
        """Load configuration from ONTIC_* environment variables."""
        return cls(
            secret_key=os.environ.get("ONTIC_STRIPE_SECRET_KEY", ""),
            webhook_secret=os.environ.get("ONTIC_STRIPE_WEBHOOK_SECRET", ""),
            price_id_builder=os.environ.get(
                "ONTIC_STRIPE_PRICE_BUILDER", ""
            ),
            price_id_pro=os.environ.get("ONTIC_STRIPE_PRICE_PRO", ""),
            billing_mode=BillingMode(
                os.environ.get("ONTIC_BILLING_MODE", "shadow")
            ),
        )

    @property
    def is_live(self) -> bool:
        return self.billing_mode == BillingMode.LIVE

    @property
    def is_configured(self) -> bool:
        return bool(self.secret_key and self.webhook_secret)


# ── Tier definitions ────────────────────────────────────────────────


class Tier(str, Enum):
    """Pricing tier per PRICING_MODEL.md §2.2."""
    EXPLORER = "explorer"
    BUILDER = "builder"
    PROFESSIONAL = "professional"


@dataclass(frozen=True)
class TierSpec:
    """Tier specification."""
    name: str
    included_cu: float
    base_price_usd: float
    overage_rate_usd: float  # per CU above included
    stripe_price_id: str = ""


# ── Customer record ─────────────────────────────────────────────────


@dataclass
class Customer:
    """Maps an API key to a Stripe customer and subscription."""
    api_key_hash: str
    stripe_customer_id: str
    stripe_subscription_id: str
    stripe_subscription_item_id: str  # metered usage line item
    tier: Tier
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    active: bool = True


# ── Billing engine ──────────────────────────────────────────────────


class StripeBilling:
    """Manages Stripe subscriptions and usage-based billing.

    Designed to be used as a singleton. Degrade to shadow mode
    if Stripe is not configured.
    """

    def __init__(self, config: StripeConfig | None = None) -> None:
        self._config = config or StripeConfig.from_env()
        self._customers: dict[str, Customer] = {}  # api_key_hash → Customer
        self._stripe: Any = None

        self._tiers: dict[Tier, TierSpec] = {
            Tier.EXPLORER: TierSpec(
                name="Explorer",
                included_cu=100.0,
                base_price_usd=0.0,
                overage_rate_usd=0.0,
            ),
            Tier.BUILDER: TierSpec(
                name="Builder",
                included_cu=1_000.0,
                base_price_usd=49.0,
                overage_rate_usd=0.05,
                stripe_price_id=self._config.price_id_builder,
            ),
            Tier.PROFESSIONAL: TierSpec(
                name="Professional",
                included_cu=10_000.0,
                base_price_usd=299.0,
                overage_rate_usd=0.03,
                stripe_price_id=self._config.price_id_pro,
            ),
        }

        if self._config.is_live and self._config.is_configured:
            self._init_stripe()
        else:
            logger.info(
                "Stripe billing in %s mode%s",
                self._config.billing_mode.value,
                " (not configured)" if not self._config.is_configured else "",
            )

    def _init_stripe(self) -> None:
        """Initialize the Stripe client."""
        try:
            import stripe
            stripe.api_key = self._config.secret_key
            stripe.api_version = "2024-12-18.acacia"
            self._stripe = stripe
            logger.info("Stripe billing initialized — LIVE mode")
        except ImportError:
            logger.error(
                "stripe package not installed. "
                "Install with: pip install stripe>=10.0.0"
            )
            raise

    @property
    def is_live(self) -> bool:
        return self._config.is_live and self._stripe is not None

    # ── API Key Hashing ─────────────────────────────────────────────

    @staticmethod
    def _hash_key(api_key: str) -> str:
        """One-way hash of API key for storage. Never store raw keys."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]

    # ── Customer Management ─────────────────────────────────────────

    def create_customer(
        self,
        *,
        api_key: str,
        email: str,
        name: str = "",
        tier: Tier = Tier.EXPLORER,
    ) -> Customer:
        """Create a Stripe customer and metered subscription.

        Parameters
        ----------
        api_key : str
            The API key to associate with this customer.
        email : str
            Customer email for invoices.
        name : str, optional
            Customer name.
        tier : Tier
            Pricing tier.

        Returns
        -------
        Customer
            The created customer record.
        """
        key_hash = self._hash_key(api_key)

        if key_hash in self._customers:
            return self._customers[key_hash]

        tier_spec = self._tiers[tier]

        if not self.is_live:
            # Shadow mode: create local record only
            customer = Customer(
                api_key_hash=key_hash,
                stripe_customer_id=f"cus_shadow_{key_hash}",
                stripe_subscription_id=f"sub_shadow_{key_hash}",
                stripe_subscription_item_id=f"si_shadow_{key_hash}",
                tier=tier,
            )
            self._customers[key_hash] = customer
            logger.info(
                "Created shadow customer: hash=%s tier=%s",
                key_hash, tier.value,
            )
            return customer

        # Live mode: create in Stripe
        stripe_customer = self._stripe.Customer.create(
            email=email,
            name=name or email,
            metadata={
                "api_key_hash": key_hash,
                "tier": tier.value,
                "platform": "physics-os",
            },
        )

        # Create a metered subscription
        subscription_params: dict[str, Any] = {
            "customer": stripe_customer.id,
            "items": [],
            "metadata": {
                "api_key_hash": key_hash,
                "tier": tier.value,
            },
        }

        if tier != Tier.EXPLORER and tier_spec.stripe_price_id:
            subscription_params["items"].append(
                {"price": tier_spec.stripe_price_id}
            )
        else:
            # Explorer tier: no Stripe subscription needed
            customer = Customer(
                api_key_hash=key_hash,
                stripe_customer_id=stripe_customer.id,
                stripe_subscription_id="",
                stripe_subscription_item_id="",
                tier=tier,
            )
            self._customers[key_hash] = customer
            logger.info(
                "Created Stripe customer (Explorer): cus=%s",
                stripe_customer.id,
            )
            return customer

        subscription = self._stripe.Subscription.create(
            **subscription_params
        )

        si_id = subscription["items"]["data"][0]["id"]

        customer = Customer(
            api_key_hash=key_hash,
            stripe_customer_id=stripe_customer.id,
            stripe_subscription_id=subscription.id,
            stripe_subscription_item_id=si_id,
            tier=tier,
        )
        self._customers[key_hash] = customer
        logger.info(
            "Created Stripe customer: cus=%s sub=%s tier=%s",
            stripe_customer.id, subscription.id, tier.value,
        )
        return customer

    def get_customer(self, api_key: str) -> Customer | None:
        """Look up customer by API key."""
        return self._customers.get(self._hash_key(api_key))

    # ── Usage Reporting ─────────────────────────────────────────────

    def report_usage(
        self,
        *,
        api_key: str,
        compute_units: float,
        job_id: str,
    ) -> dict[str, Any]:
        """Report metered CU usage to Stripe.

        Called after every successful job. In shadow mode, this
        records the usage locally without hitting Stripe.

        Parameters
        ----------
        api_key : str
            The API key that ran the job.
        compute_units : float
            CU consumed by this job.
        job_id : str
            The job ID for idempotency.

        Returns
        -------
        dict
            Usage record (shadow or Stripe).
        """
        key_hash = self._hash_key(api_key)
        customer = self._customers.get(key_hash)

        # Quantize CU to integer cents for Stripe (1 unit = 0.01 CU)
        quantity = max(1, int(compute_units * 100))

        record = {
            "api_key_hash": key_hash,
            "job_id": job_id,
            "compute_units": compute_units,
            "quantity": quantity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "live": self.is_live,
        }

        if not self.is_live or customer is None:
            record["status"] = "shadow"
            logger.debug(
                "Shadow usage: hash=%s cu=%.4f job=%s",
                key_hash, compute_units, job_id,
            )
            return record

        if not customer.stripe_subscription_item_id:
            # Explorer tier: no metered billing
            record["status"] = "explorer_free"
            return record

        try:
            usage_record = self._stripe.SubscriptionItem.create_usage_record(
                customer.stripe_subscription_item_id,
                quantity=quantity,
                timestamp=int(time.time()),
                action="increment",
                idempotency_key=f"usage_{job_id}",
            )
            record["stripe_usage_record_id"] = usage_record.id
            record["status"] = "reported"
            logger.info(
                "Stripe usage reported: cus=%s cu=%.4f job=%s",
                customer.stripe_customer_id, compute_units, job_id,
            )
        except Exception:
            logger.exception(
                "Failed to report usage to Stripe: job=%s", job_id
            )
            record["status"] = "failed"

        return record

    # ── Webhook Processing ──────────────────────────────────────────

    def verify_webhook(
        self,
        payload: bytes,
        signature: str,
    ) -> dict[str, Any] | None:
        """Verify and parse a Stripe webhook event.

        Parameters
        ----------
        payload : bytes
            Raw request body.
        signature : str
            Stripe-Signature header value.

        Returns
        -------
        dict or None
            Parsed event dict if valid, None if verification fails.
        """
        if not self._config.webhook_secret:
            logger.error("Webhook secret not configured")
            return None

        if self._stripe is None:
            # In shadow mode, do basic HMAC verification
            return self._verify_webhook_manual(payload, signature)

        try:
            event = self._stripe.Webhook.construct_event(
                payload, signature, self._config.webhook_secret
            )
            return dict(event)
        except self._stripe.error.SignatureVerificationError:
            logger.warning("Webhook signature verification failed")
            return None
        except Exception:
            logger.exception("Webhook processing error")
            return None

    def _verify_webhook_manual(
        self, payload: bytes, signature: str,
    ) -> dict[str, Any] | None:
        """Manual HMAC verification for shadow mode."""
        try:
            parts = dict(
                item.split("=", 1)
                for item in signature.split(",")
                if "=" in item
            )
            timestamp = parts.get("t", "")
            sig_v1 = parts.get("v1", "")

            if not timestamp or not sig_v1:
                return None

            signed_payload = f"{timestamp}.".encode() + payload
            expected = hmac.new(
                self._config.webhook_secret.encode(),
                signed_payload,
                hashlib.sha256,
            ).hexdigest()

            if not hmac.compare_digest(expected, sig_v1):
                return None

            return json.loads(payload)
        except Exception:
            logger.exception("Manual webhook verification failed")
            return None

    def handle_webhook_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Process a verified Stripe webhook event.

        Handles:
            - invoice.payment_succeeded → update customer status
            - invoice.payment_failed → flag for follow-up
            - customer.subscription.deleted → deactivate customer

        Parameters
        ----------
        event : dict
            Verified Stripe event.

        Returns
        -------
        dict
            Processing result.
        """
        event_type = event.get("type", "")
        event_id = event.get("id", "unknown")

        logger.info("Processing webhook: type=%s id=%s", event_type, event_id)

        if event_type == "invoice.payment_succeeded":
            return self._handle_payment_succeeded(event)
        elif event_type == "invoice.payment_failed":
            return self._handle_payment_failed(event)
        elif event_type == "customer.subscription.deleted":
            return self._handle_subscription_deleted(event)
        else:
            return {"status": "ignored", "type": event_type}

    def _handle_payment_succeeded(
        self, event: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle successful payment."""
        invoice = event.get("data", {}).get("object", {})
        customer_id = invoice.get("customer", "")
        amount = invoice.get("amount_paid", 0)

        logger.info(
            "Payment succeeded: cus=%s amount=%d cents",
            customer_id, amount,
        )
        return {
            "status": "processed",
            "type": "invoice.payment_succeeded",
            "customer": customer_id,
            "amount_cents": amount,
        }

    def _handle_payment_failed(
        self, event: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle failed payment — flag for manual review."""
        invoice = event.get("data", {}).get("object", {})
        customer_id = invoice.get("customer", "")

        logger.warning("Payment failed: cus=%s", customer_id)

        # Find and flag the customer
        for customer in self._customers.values():
            if customer.stripe_customer_id == customer_id:
                # Don't deactivate immediately — Stripe retries
                logger.warning(
                    "Customer payment failed: hash=%s",
                    customer.api_key_hash,
                )
                break

        return {
            "status": "processed",
            "type": "invoice.payment_failed",
            "customer": customer_id,
        }

    def _handle_subscription_deleted(
        self, event: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle subscription cancellation — deactivate customer."""
        subscription = event.get("data", {}).get("object", {})
        customer_id = subscription.get("customer", "")

        for customer in self._customers.values():
            if customer.stripe_customer_id == customer_id:
                customer.active = False
                logger.info(
                    "Customer deactivated: hash=%s cus=%s",
                    customer.api_key_hash, customer_id,
                )
                break

        return {
            "status": "processed",
            "type": "customer.subscription.deleted",
            "customer": customer_id,
        }

    # ── Usage Summary ───────────────────────────────────────────────

    def get_usage_summary(self, api_key: str) -> dict[str, Any]:
        """Get billing summary for a customer.

        Returns
        -------
        dict
            Current tier, CU consumed, limit, overage status.
        """
        from .meter import ledger

        key_hash = self._hash_key(api_key)
        key_suffix = api_key[-6:] if len(api_key) > 6 else api_key
        customer = self._customers.get(key_hash)

        tier = customer.tier if customer else Tier.EXPLORER
        tier_spec = self._tiers[tier]

        total_cu = ledger.total_cu(api_key_suffix=key_suffix)
        overage_cu = max(0.0, total_cu - tier_spec.included_cu)
        overage_usd = round(overage_cu * tier_spec.overage_rate_usd, 2)

        return {
            "tier": tier.value,
            "tier_name": tier_spec.name,
            "included_cu": tier_spec.included_cu,
            "used_cu": total_cu,
            "remaining_cu": max(0.0, tier_spec.included_cu - total_cu),
            "overage_cu": overage_cu,
            "overage_usd": overage_usd,
            "base_price_usd": tier_spec.base_price_usd,
            "total_estimated_usd": round(
                tier_spec.base_price_usd + overage_usd, 2
            ),
            "billing_mode": self._config.billing_mode.value,
            "active": customer.active if customer else True,
        }


# ── Singleton ───────────────────────────────────────────────────────

billing: StripeBilling | None = None


def get_billing() -> StripeBilling:
    """Get or create the billing singleton."""
    global billing
    if billing is None:
        billing = StripeBilling()
    return billing
