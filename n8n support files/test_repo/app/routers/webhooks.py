"""
Webhook receiver endpoints.

POST /webhooks/payment  — Receive payment lifecycle events from gateway
GET  /webhooks/status   — Storm detection and dedup stats
"""
import logging

from fastapi import APIRouter, HTTPException, Header, Request
from typing import Optional

from app.models import WebhookEvent, WebhookEventType
from app.schemas import WebhookPayload, WebhookResponse
from app.services.webhook_service import webhook_service
from app.metrics import prometheus_metrics as m

logger = logging.getLogger("webhooks_router")

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post(
    "/payment",
    response_model=WebhookResponse,
    summary="Receive payment gateway webhook",
)
async def receive_webhook(
    payload: WebhookPayload,
    x_gateway_signature: Optional[str] = Header(None),
    x_retry_count: Optional[int] = Header(None, alias="X-Retry-Count"),
):
    """
    Receives payment lifecycle webhooks from external gateways.

    Signature verification is skipped (production would use HMAC-SHA256).
    Duplicate detection uses in-memory store — will fail across restarts.
    """
    # Signature check (deliberately not implemented — common production debt)
    if x_gateway_signature is None:
        logger.warning(
            f"[Webhook Router] Missing gateway signature for webhook={payload.webhook_id}"
        )
        # Not blocking — just log (another common production mistake)

    try:
        event_type = WebhookEventType(payload.event_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown event type: {payload.event_type}",
        )

    event = WebhookEvent(
        id=payload.webhook_id,
        transaction_id=payload.transaction_id,
        event_type=event_type,
        source_gateway=payload.gateway,
        payload=payload.payload,
    )

    if x_retry_count and x_retry_count > 0:
        logger.warning(
            f"[Webhook Router] Retry #{x_retry_count} for webhook={payload.webhook_id} "
            f"tx={payload.transaction_id}"
        )

    result = await webhook_service.handle(event)

    return WebhookResponse(
        webhook_id=payload.webhook_id,
        status=result["status"],
        processing_count=result.get("count", 1),
        message=result.get("error", "ok"),
    )


@router.get("/status", summary="Webhook processing status and storm detection")
async def webhook_status():
    """Current webhook processing stats and retry storm candidates."""
    stats = webhook_service.get_stats()
    storm_candidates = webhook_service.get_storm_candidates()

    return {
        "stats": stats,
        "storm_candidates": storm_candidates[:20],
        "status": "storm_detected" if storm_candidates else "ok",
    }
