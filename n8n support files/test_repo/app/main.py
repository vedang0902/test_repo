"""
PaymentPipeline — FastAPI Application Entry Point

Wires together:
  - FastAPI app with CORS and structured logging middleware
  - All API routers (payments, webhooks, health, analytics)
  - Prometheus metrics server (port 8000)
  - Background tasks: transaction generator, reconciliation runner

Startup sequence:
  1. Start Prometheus metrics HTTP server on :8000
  2. Seed synthetic accounts into SQLite DB
  3. Start reconciliation background runner (every 30s)
  4. Start autonomous transaction generator (continuous)
  5. Accept API requests on :5000
"""
import asyncio
import logging
import logging.config
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from app.config import settings
from app.metrics.prometheus_metrics import start_metrics_server
from app.middleware.logging_middleware import RequestLoggingMiddleware
from app.routers.payments import router as payments_router, accounts_router
from app.routers.webhooks import router as webhooks_router
from app.routers.health import router as health_router
from app.routers.analytics import router as analytics_router
from app.services.transaction_generator import transaction_generator
from app.services.reconciliation import reconciliation_service

# ── Logging configuration ─────────────────────────────────────────────────────

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "structured": {
            "format": (
                "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
            ),
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "structured",
        },
    },
    "root": {
        "level": settings.log_level,
        "handlers": ["console"],
    },
    "loggers": {
        "uvicorn": {"level": "WARNING", "propagate": True},
        "uvicorn.access": {"level": "WARNING", "propagate": True},
        "payment_processor": {"level": "DEBUG", "propagate": True},
        "fraud_detector": {"level": "DEBUG", "propagate": True},
        "reconciliation": {"level": "DEBUG", "propagate": True},
        "webhook_service": {"level": "DEBUG", "propagate": True},
        "transaction_generator": {"level": "INFO", "propagate": True},
        "database": {"level": "INFO", "propagate": True},
        "access": {"level": "INFO", "propagate": True},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("main")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="PaymentPipeline API",
    description=(
        "Production-grade payment processing pipeline. "
        "Processes card, bank transfer, wallet, and crypto payments. "
        "Includes fraud detection, reconciliation, and webhook handling."
    ),
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — wide open for internal dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(payments_router)
app.include_router(accounts_router)
app.include_router(webhooks_router)
app.include_router(health_router)
app.include_router(analytics_router)

# ── Static frontend ───────────────────────────────────────────────────────────

try:
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
except Exception:
    pass  # Frontend directory optional


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    try:
        return FileResponse("frontend/index.html")
    except Exception:
        return JSONResponse({"service": "PaymentPipeline", "version": settings.version})


# ── Lifespan ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    logger.info(
        f"╔══════════════════════════════════════════════════════╗\n"
        f"  PaymentPipeline v{settings.version} starting up\n"
        f"  Environment : {settings.environment}\n"
        f"  API         : http://0.0.0.0:{settings.api_port}\n"
        f"  Metrics     : http://0.0.0.0:{settings.metrics_port}/metrics\n"
        f"  Docs        : http://0.0.0.0:{settings.api_port}/docs\n"
        f"╚══════════════════════════════════════════════════════╝"
    )

    # Start Prometheus metrics server
    start_metrics_server(settings.metrics_port)
    logger.info(f"[Startup] Prometheus metrics server started on :{settings.metrics_port}")

    # Seed accounts
    await transaction_generator.seed_accounts()

    # Start background tasks
    asyncio.create_task(reconciliation_service.run_periodic(), name="reconciliation_runner")
    asyncio.create_task(transaction_generator.run(), name="transaction_generator")

    logger.info("[Startup] All background tasks started. Pipeline is live.")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("[Shutdown] PaymentPipeline shutting down gracefully")


# ── Root info endpoint ────────────────────────────────────────────────────────

@app.get("/info", tags=["meta"])
async def info():
    return {
        "service": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "endpoints": {
            "payments": "/payments",
            "accounts": "/accounts",
            "webhooks": "/webhooks",
            "health": "/health",
            "analytics": "/analytics",
            "metrics": f"http://0.0.0.0:{settings.metrics_port}/metrics",
            "docs": "/docs",
        },
    }
