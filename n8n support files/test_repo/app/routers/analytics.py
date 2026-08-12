"""
Analytics and reporting endpoints.
Provides aggregated stats for the ops dashboard.
"""
from datetime import datetime

from fastapi import APIRouter, Query

from app.schemas import AnalyticsSummary, ReconciliationResponse
from app.services.reconciliation import reconciliation_service
from app.services.fraud_detector import fraud_detector
from app.services.webhook_service import webhook_service
from app.services.transaction_generator import transaction_generator
from app.database import db_pool
from app.metrics import prometheus_metrics as m

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary, summary="Overall pipeline summary")
async def get_summary():
    """Real-time summary of payment pipeline health and error counts."""
    recon = reconciliation_service.get_summary()
    pool = db_pool.pool_status()

    # Pull from DB for transaction counts
    total = 0
    completed = 0
    failed = 0
    partial = 0
    fraud_blocked = 0
    total_volume = 0.0
    total_fees = 0.0

    try:
        with db_pool.connection() as conn:
            row = conn.fetchone(
                """SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status='partial_commit' THEN 1 ELSE 0 END) as partial,
                    SUM(CASE WHEN status='fraud_blocked' THEN 1 ELSE 0 END) as fraud_blocked,
                    SUM(CASE WHEN status='completed' THEN amount ELSE 0 END) as total_volume,
                    SUM(CASE WHEN status='completed' THEN fee ELSE 0 END) as total_fees
                   FROM transactions"""
            )
            if row:
                total = row["total"] or 0
                completed = row["completed"] or 0
                failed = row["failed"] or 0
                partial = row["partial"] or 0
                fraud_blocked = row["fraud_blocked"] or 0
                total_volume = row["total_volume"] or 0.0
                total_fees = row["total_fees"] or 0.0
    except Exception:
        pass

    wh_stats = webhook_service.get_stats()

    return AnalyticsSummary(
        total_transactions=total,
        successful_transactions=completed,
        failed_transactions=failed,
        partial_commits=partial,
        fraud_blocked=fraud_blocked,
        total_volume_usd=round(total_volume, 2),
        total_fees_collected=round(total_fees, 4),
        reconciliation_drift=recon["drift"],
        avg_processing_time_ms=0.0,  # Would need histogram scrape
        webhook_retry_storms=wh_stats["storm_candidates"],
        idempotency_violations=0,  # Tracked in Prometheus
        deadlock_events=0,         # Tracked in Prometheus
        db_pool_exhaustions=pool["total_exhausted"],
        fraud_cascade_events=0,    # Tracked in Prometheus
    )


@router.get("/reconciliation", response_model=ReconciliationResponse, summary="Reconciliation status")
async def get_reconciliation():
    """Current reconciliation report — shows float drift accumulation."""
    report = reconciliation_service.run_reconciliation()
    drift_pct = (report.drift / max(report.actual_balance, 0.01)) * 100

    return ReconciliationResponse(
        report_id=report.report_id,
        ledger_balance=report.ledger_balance,
        actual_balance=report.actual_balance,
        drift=report.drift,
        drift_pct=drift_pct,
        transaction_count=report.transaction_count,
        status=report.status,
        generated_at=report.generated_at,
    )


@router.get("/fraud", summary="Fraud detection stats")
async def get_fraud_stats(account_id: str = Query(None)):
    """Fraud scoring stats. Optionally filter by account."""
    if account_id:
        return fraud_detector.get_account_risk_profile(account_id)

    return {
        "score_accumulator_entries": len(fraud_detector._score_accumulator),
        "check_count_entries": len(fraud_detector._check_count),
        "flagged_account_count": len(fraud_detector._flagged_accounts),
        "high_risk_accounts": [
            {"account_id": k, "score": v}
            for k, v in sorted(
                fraud_detector._score_accumulator.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        ],
    }


@router.get("/webhooks", summary="Webhook processing stats")
async def get_webhook_stats():
    """Webhook dedup cache and storm detection stats."""
    wh = webhook_service.get_stats()
    return {
        **wh,
        "storm_candidates": webhook_service.get_storm_candidates()[:20],
    }


@router.get("/generator", summary="Transaction generator stats")
async def get_generator_stats():
    """Stats from the autonomous transaction generator."""
    return {
        "total_generated": transaction_generator._total_generated,
        "total_failed": transaction_generator._total_failed,
        "cache_size": len(transaction_generator._transaction_history_cache),
        "completed_tx_ids_tracked": len(transaction_generator._completed_tx_ids),
        "accounts_seeded": transaction_generator._accounts_seeded,
    }
