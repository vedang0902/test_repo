"""
Webhook Event Handler.

Receives and processes payment lifecycle events from external gateways
(Stripe, Razorpay, PayPal mock). Stores processed state so duplicates
can be detected.

=============================================================================
BUG: Webhook Retry Storm (In-Memory Dedup)
=============================================================================
Root cause:
  The "processed" flag for webhooks is stored only in-memory (`_processed`
  dict). When the application restarts or when a second worker process
  handles the request, the dict is empty — every webhook appears unprocessed.

  Additionally, the dedup check is not atomic:
    1. Check if webhook_id in _processed  → False (not there)
    2. [race window — another coroutine runs]
    3. Process webhook                    → double process
    4. Set _processed[webhook_id] = True  → too late

  In production this manifests as:
  - Payment confirmations processed 3-5× per transaction
  - Duplicate charges / credits
  - Exponentially growing retry loop as gateway keeps retrying 4xx-flagged
    events that are actually being processed (just taking too long)

Fix (NOT applied):
  Persist processed state to DB with a unique constraint on webhook_id.
  Use SELECT FOR UPDATE or redis SETNX for atomic check-and-set.

Symptoms in logs:
  WARNING webhook_service | RETRY STORM: webhook wh_abc processed 4 times
  WARNING webhook_service | Duplicate processing detected: webhook wh_abc (count=2)
  ERROR   webhook_service | Gateway retry loop: webhook wh_abc — 7 total deliveries

Prometheus metrics:
  webhook_duplicate_processing_total ↑
  webhook_retry_storm_total          ↑
  webhook_received_total             ↑↑ (storm inflates this)
"""
import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Dict, Optional

from app.config import settings
from app.models import WebhookEvent, WebhookEventType
from app.metrics import prometheus_metrics as m

logger = logging.getLogger("webhook_service")

cfg = settings.webhook


class WebhookService:
    """
    Handles inbound webhook events from payment gateways.

    BUG: In-memory dedup store — not persisted, not shared across workers.
    """

    def __init__(self):
        # BUG: In-memory only — cleared on restart, invisible to other workers
        self._processed: Dict[str, datetime] = {}
        # BUG: Tracks processing counts in memory — another memory leak
        self._processing_counts: Dict[str, int] = {}
        self._rate_window: list = []   # Timestamps for rate calculation

    async def handle(self, event: WebhookEvent) -> dict:
        """
        Process a webhook event.

        Returns processing result.
        Intentionally never raises — gateway must see 200 to stop retrying
        (but sometimes we return 500 which causes the retry storm).
        """
        wh_id = event.id
        now = datetime.utcnow()

        # Track processing count (in memory — bug)
        count = self._processing_counts.get(wh_id, 0) + 1
        self._processing_counts[wh_id] = count
        event.processing_count = count

        # Update rate tracking
        self._rate_window.append(time.monotonic())
        cutoff = time.monotonic() - 10.0
        self._rate_window = [t for t in self._rate_window if t > cutoff]
        rate = len(self._rate_window) / 10.0
        m.webhook_processing_rate.set(rate)

        # Emit received metric
        m.webhook_received_total.labels(
            event_type=event.event_type.value,
            gateway=event.source_gateway,
        ).inc()

        # ── Dedup check (BUG: not atomic, not persistent) ─────────────────────
        if wh_id in self._processed:
            # We "know" this was processed — but this state is lost on restart
            m.webhook_duplicate_processing_total.inc()
            logger.warning(
                f"[Webhook] Duplicate processing detected: webhook={wh_id} "
                f"count={count} first_processed={self._processed[wh_id].isoformat()}"
            )
            # Return success to prevent further retries (correct behaviour)
            # But we still log the duplicate so it shows in metrics
            return {"status": "already_processed", "webhook_id": wh_id, "count": count}

        # ── Storm detection ───────────────────────────────────────────────────
        if count > cfg.storm_threshold:
            m.webhook_retry_storm_total.inc()
            m.app_errors_total.labels(
                component="webhook_service", error_type="retry_storm"
            ).inc()
            m.app_error_rate.set(1)
            logger.error(
                f"[Webhook] RETRY STORM DETECTED: webhook={wh_id} "
                f"processed {count} times | event_type={event.event_type.value} | "
                f"tx={event.transaction_id} | rate={rate:.1f}/s"
            )

        if count > 1:
            m.webhook_duplicate_processing_total.inc()
            logger.warning(
                f"[Webhook] Processing webhook {wh_id} again (count={count}) — "
                f"possible restart or race condition"
            )

        # ── Simulate processing delay ─────────────────────────────────────────
        processing_time = random.uniform(0.05, 0.4)
        await asyncio.sleep(processing_time)

        # ── BUG: Race window — concurrent coroutine can pass dedup check here ──
        # (The check above and this mark below are not atomic)

        # Mark as processed (in memory only — will be lost on restart)
        self._processed[wh_id] = now
        event.processed = True
        event.last_processed_at = now

        m.webhook_dedup_cache_size.set(len(self._processed))

        # Simulate occasional processing failures that cause gateway retries
        if random.random() < 0.08 and count == 1:
            # Return error — gateway will retry, causing count > 1 next time
            m.app_errors_total.labels(
                component="webhook_service", error_type="processing_failure"
            ).inc()
            logger.error(
                f"[Webhook] Processing failed for webhook={wh_id} "
                f"event_type={event.event_type.value} — gateway will retry"
            )
            # BUG: We don't add to _processed here, so next retry will process
            return {
                "status": "error",
                "webhook_id": wh_id,
                "error": "processing_failure",
                "will_retry": True,
            }

        logger.info(
            f"[Webhook] Processed webhook={wh_id} event={event.event_type.value} "
            f"tx={event.transaction_id} count={count} took={processing_time:.2f}s"
        )

        return {
            "status": "processed",
            "webhook_id": wh_id,
            "event_type": event.event_type.value,
            "processing_count": count,
        }

    def get_storm_candidates(self) -> list:
        """Return webhooks with processing count above storm threshold."""
        return [
            {"webhook_id": wid, "count": cnt}
            for wid, cnt in self._processing_counts.items()
            if cnt > cfg.storm_threshold
        ]

    def get_stats(self) -> dict:
        return {
            "total_processed_in_memory": len(self._processed),
            "total_webhooks_seen": len(self._processing_counts),
            "storm_candidates": len(self.get_storm_candidates()),
            "current_rate_per_second": len(self._rate_window) / 10.0,
        }


# Singleton
webhook_service = WebhookService()
