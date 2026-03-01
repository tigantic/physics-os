"""Ontic API — Billing endpoints.

GET   /v1/billing/usage       → Current period usage summary
POST  /v1/billing/checkout    → Create Stripe checkout session
GET   /v1/billing/portal      → Redirect to Stripe customer portal
POST  /v1/billing/webhook     → Stripe webhook handler (no auth)
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from pydantic import BaseModel, Field

from ...billing.stripe_billing import (
    Tier,
    get_billing,
)
from ..auth import require_api_key
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/billing", tags=["billing"])


# ── Request / Response Models ───────────────────────────────────────


class UsageSummaryResponse(BaseModel):
    tier: str
    tier_name: str
    included_cu: float
    used_cu: float
    remaining_cu: float
    overage_cu: float
    overage_usd: float
    base_price_usd: float
    total_estimated_usd: float
    billing_mode: str
    active: bool


class CheckoutRequest(BaseModel):
    email: str = Field(
        description="Customer email for invoices",
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    )
    name: str = ""
    tier: str = Field(
        default="builder",
        description="Pricing tier: 'builder' or 'professional'",
    )
    success_url: str = Field(
        description="URL to redirect on successful checkout",
    )
    cancel_url: str = Field(
        description="URL to redirect on cancelled checkout",
    )


class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str
    tier: str


class PortalResponse(BaseModel):
    portal_url: str


class WebhookResponse(BaseModel):
    status: str
    type: str = ""


# ── GET /v1/billing/usage ──────────────────────────────────────────


@router.get(
    "/usage",
    response_model=UsageSummaryResponse,
    summary="Current billing period usage",
)
async def get_usage(
    api_key: Annotated[str, Depends(require_api_key)],
) -> UsageSummaryResponse:
    """Return compute unit usage for the current billing period.

    Includes tier info, included CU, used CU, remaining CU,
    overage charges, and estimated total.
    """
    billing = get_billing()
    summary = billing.get_usage_summary(api_key)
    return UsageSummaryResponse(**summary)


# ── POST /v1/billing/checkout ──────────────────────────────────────


@router.post(
    "/checkout",
    response_model=CheckoutResponse,
    summary="Create a Stripe checkout session",
)
async def create_checkout(
    body: CheckoutRequest,
    api_key: Annotated[str, Depends(require_api_key)],
) -> CheckoutResponse:
    """Create a Stripe Checkout session for a paid tier subscription.

    Returns a URL to redirect the user to Stripe's hosted checkout page.
    Explorer tier is free and does not require checkout.
    """
    billing = get_billing()

    # Validate tier
    try:
        tier = Tier(body.tier)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E030",
                "message": f"Invalid tier: {body.tier!r}. Use 'builder' or 'professional'.",
                "retryable": False,
            },
        )

    if tier == Tier.EXPLORER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E031",
                "message": "Explorer tier is free. No checkout required.",
                "retryable": False,
            },
        )

    if not billing.is_live:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "E032",
                "message": "Billing is in shadow mode. Stripe checkout not available.",
                "retryable": False,
            },
        )

    # Ensure customer exists
    customer = billing.get_customer(api_key)
    if customer is None:
        customer = billing.create_customer(
            api_key=api_key,
            email=body.email,
            name=body.name,
            tier=tier,
        )

    # Create checkout session
    tier_spec = billing._tiers[tier]
    if not tier_spec.stripe_price_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "E033",
                "message": "Stripe price not configured for this tier.",
                "retryable": False,
            },
        )

    try:
        import stripe
        session = stripe.checkout.Session.create(
            customer=customer.stripe_customer_id,
            mode="subscription",
            line_items=[
                {
                    "price": tier_spec.stripe_price_id,
                },
            ],
            success_url=body.success_url,
            cancel_url=body.cancel_url,
            metadata={
                "api_key_hash": customer.api_key_hash,
                "tier": tier.value,
            },
        )
        return CheckoutResponse(
            checkout_url=session.url or "",
            session_id=session.id,
            tier=tier.value,
        )
    except Exception as exc:
        logger.exception("Stripe checkout session creation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "E034",
                "message": f"Stripe API error: {exc}",
                "retryable": True,
            },
        )


# ── GET /v1/billing/portal ─────────────────────────────────────────


@router.get(
    "/portal",
    response_model=PortalResponse,
    summary="Stripe customer portal link",
)
async def get_portal(
    api_key: Annotated[str, Depends(require_api_key)],
) -> PortalResponse:
    """Generate a link to the Stripe Customer Portal.

    Lets customers manage payment methods, view invoices,
    and change/cancel subscriptions.
    """
    billing = get_billing()

    if not billing.is_live:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "E032",
                "message": "Billing is in shadow mode. Portal not available.",
                "retryable": False,
            },
        )

    customer = billing.get_customer(api_key)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "E035",
                "message": "No billing account found for this API key.",
                "retryable": False,
            },
        )

    try:
        import stripe
        session = stripe.billing_portal.Session.create(
            customer=customer.stripe_customer_id,
        )
        return PortalResponse(portal_url=session.url)
    except Exception as exc:
        logger.exception("Stripe portal session creation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "E034",
                "message": f"Stripe API error: {exc}",
                "retryable": True,
            },
        )


# ── POST /v1/billing/webhook ──────────────────────────────────────


@router.post(
    "/webhook",
    response_model=WebhookResponse,
    summary="Stripe webhook handler",
    include_in_schema=False,  # Not for public docs
)
async def stripe_webhook(
    request: Request,
    stripe_signature: Annotated[str | None, Header(alias="stripe-signature")] = None,
) -> WebhookResponse:
    """Handle Stripe webhook events.

    This endpoint does NOT require API key auth. It validates
    incoming events using the Stripe webhook signature.

    Handles:
        - invoice.payment_succeeded
        - invoice.payment_failed
        - customer.subscription.deleted
    """
    if stripe_signature is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E036",
                "message": "Missing Stripe-Signature header.",
                "retryable": False,
            },
        )

    payload = await request.body()
    billing = get_billing()

    event = billing.verify_webhook(payload, stripe_signature)
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E037",
                "message": "Webhook signature verification failed.",
                "retryable": False,
            },
        )

    result = billing.handle_webhook_event(event)
    return WebhookResponse(
        status=result.get("status", "unknown"),
        type=result.get("type", ""),
    )
