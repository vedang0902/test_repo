"""
Health check endpoints.
Used by load balancers, K8s liveness/readiness probes, and monitoring.
"""
import time
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import db_pool
from app.schemas import HealthResponse
from app.services.reconciliation import reconciliation_service
from app.services.fraud_detector import fraud_detector
from app.services.webhook_service import webhook_service

router = APIRouter(prefix="/health", tags=["health"])

_start_time = time.monotonic()


@router.get("/live", summary="Liveness probe")
async def liveness():
    """K8s liveness probe — returns 200 if process is alive."""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/ready", summary="Readiness probe")
async def readiness():
    """K8s readiness probe — checks DB pool and critical services."""
    pool = db_pool.pool_status()
    db_ok = pool["available"] > 0

    status_code = 200 if db_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if db_ok else "not_ready",
            "db_pool": pool,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@router.get("", response_model=HealthResponse, summary="Full health status")
async def health():
    """Detailed system health including all component statuses."""
    pool = db_pool.pool_status()
    recon = reconciliation_service.get_summary()
    wh_stats = webhook_service.get_stats()
    uptime = time.monotonic() - _start_time

    # Determine component statuses
    db_status = "ok" if pool["available"] > 0 else "degraded"
    recon_status = recon["status"]
    fraud_status = "ok"  # Fraud detector is always available (in-memory)
    webhook_status = "storm" if wh_stats["storm_candidates"] > 0 else "ok"

    overall = "ok"
    if db_status != "ok" or recon_status == "critical":
        overall = "critical"
    elif recon_status == "mismatch" or webhook_status == "storm":
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version=settings.version,
        environment=settings.environment,
        uptime_seconds=uptime,
        db_pool_available=pool["available"],
        db_pool_in_use=pool["in_use"],
        active_transactions=0,
        components={
            "database": db_status,
            "reconciliation": recon_status,
            "fraud_detector": fraud_status,
            "webhook_service": webhook_status,
        },
    )
