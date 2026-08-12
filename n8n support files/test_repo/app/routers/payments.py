"""
Payment API endpoints.

POST /payments          — Create a new payment
GET  /payments/{id}     — Get transaction by ID
POST /payments/{id}/refund — Initiate a refund
GET  /payments          — List recent transactions
GET  /accounts          — List synthetic accounts
"""
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Header
from fastapi.responses import JSONResponse

from app.database import db_pool, DBConnectionError, PoolExhaustedError
from app.models import Transaction, PaymentMethod, TransactionStatus
from app.schemas import (
    PaymentRequest,
    RefundRequest,
    TransactionResponse,
    AccountResponse,
    ErrorResponse,
)
from app.services.payment_processor import payment_processor
from app.services.transaction_generator import SYNTHETIC_ACCOUNTS

import logging

logger = logging.getLogger("payments_router")

router = APIRouter(prefix="/payments", tags=["payments"])
accounts_router = APIRouter(prefix="/accounts", tags=["accounts"])


def _tx_to_response(tx: Transaction) -> TransactionResponse:
    return TransactionResponse(
        id=tx.id,
        idempotency_key=tx.idempotency_key,
        from_account=tx.from_account,
        to_account=tx.to_account,
        amount=tx.amount,
        currency=tx.currency,
        method=tx.method.value,
        status=tx.status.value,
        fraud_score=tx.fraud_score,
        fee=tx.fee,
        net_amount=tx.net_amount,
        error_message=tx.error_message,
        created_at=tx.created_at,
        updated_at=tx.updated_at,
        completed_at=tx.completed_at,
    )


@router.post(
    "",
    response_model=TransactionResponse,
    status_code=202,
    summary="Create payment",
    responses={
        202: {"description": "Payment accepted and processing"},
        402: {"description": "Fraud blocked"},
        409: {"description": "Idempotency key collision"},
        503: {"description": "Database unavailable"},
    },
)
async def create_payment(
    request: PaymentRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """
    Submit a payment for processing.

    Uses Idempotency-Key header (falls back to body field) to deduplicate
    retries. Due to a race condition in the dedup check, concurrent requests
    with the same key may both be processed.
    """
    # Header overrides body field
    eff_key = idempotency_key or request.idempotency_key

    tx = Transaction(
        idempotency_key=eff_key,
        from_account=request.from_account,
        to_account=request.to_account,
        amount=request.amount,
        currency=request.currency,
        method=PaymentMethod(request.method),
        metadata=request.metadata,
    )

    try:
        result = await payment_processor.process_payment(tx)
    except (DBConnectionError, PoolExhaustedError) as e:
        logger.error(f"[Router] DB unavailable for payment: {e}")
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")
    except Exception as e:
        logger.error(f"[Router] Payment processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if result.status == TransactionStatus.FRAUD_BLOCKED:
        return JSONResponse(status_code=402, content=_tx_to_response(result).model_dump(mode="json"))

    return _tx_to_response(result)


@router.get(
    "",
    response_model=List[TransactionResponse],
    summary="List recent transactions",
)
async def list_transactions(
    limit: int = Query(default=50, ge=1, le=500),
    status: Optional[str] = Query(default=None),
    account_id: Optional[str] = Query(default=None),
):
    """List recent transactions with optional filtering."""
    try:
        with db_pool.connection() as conn:
            query = "SELECT * FROM transactions"
            conditions = []
            params = []

            if status:
                conditions.append("status = ?")
                params.append(status)
            if account_id:
                conditions.append("(from_account = ? OR to_account = ?)")
                params.extend([account_id, account_id])

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.fetchall(query, tuple(params))

            results = []
            for row in rows:
                tx = Transaction(
                    id=row["id"],
                    idempotency_key=row["idempotency_key"],
                    from_account=row["from_account"],
                    to_account=row["to_account"],
                    amount=row["amount"],
                    currency=row["currency"],
                    method=PaymentMethod(row["method"]),
                    status=TransactionStatus(row["status"]),
                    fraud_score=row["fraud_score"],
                    fee=row["fee"],
                    net_amount=row["net_amount"],
                    error_message=row["error_message"],
                )
                results.append(_tx_to_response(tx))

            return results
    except DBConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get(
    "/{transaction_id}",
    response_model=TransactionResponse,
    summary="Get transaction by ID",
)
async def get_transaction(transaction_id: str):
    """Retrieve a specific transaction by ID."""
    try:
        with db_pool.connection() as conn:
            row = conn.fetchone(
                "SELECT * FROM transactions WHERE id = ?",
                (transaction_id,),
            )
            if not row:
                raise HTTPException(status_code=404, detail="Transaction not found")

            tx = Transaction(
                id=row["id"],
                idempotency_key=row["idempotency_key"],
                from_account=row["from_account"],
                to_account=row["to_account"],
                amount=row["amount"],
                currency=row["currency"],
                method=PaymentMethod(row["method"]),
                status=TransactionStatus(row["status"]),
                fraud_score=row["fraud_score"],
                fee=row["fee"],
                net_amount=row["net_amount"],
                error_message=row["error_message"],
            )
            return _tx_to_response(tx)
    except HTTPException:
        raise
    except DBConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post(
    "/{transaction_id}/refund",
    response_model=TransactionResponse,
    summary="Initiate refund",
)
async def create_refund(transaction_id: str, request: RefundRequest):
    """
    Initiate a refund for a completed transaction.

    Warning: Refund processing acquires locks in a different order than
    payment processing. Running this concurrently with payments may trigger
    the deadlock scenario.
    """
    try:
        with db_pool.connection() as conn:
            row = conn.fetchone(
                "SELECT * FROM transactions WHERE id = ?",
                (transaction_id,),
            )
            if not row:
                raise HTTPException(status_code=404, detail="Transaction not found")
            if row["status"] != "completed":
                raise HTTPException(
                    status_code=422,
                    detail=f"Cannot refund transaction in status: {row['status']}",
                )

            original_tx = Transaction(
                id=row["id"],
                idempotency_key=row["idempotency_key"],
                from_account=row["from_account"],
                to_account=row["to_account"],
                amount=row["amount"],
                currency=row["currency"],
                method=PaymentMethod(row["method"]),
                status=TransactionStatus(row["status"]),
            )
    except HTTPException:
        raise
    except DBConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

    refund_amount = request.amount
    if refund_amount and refund_amount > original_tx.amount:
        raise HTTPException(
            status_code=422,
            detail=f"Refund amount ${refund_amount} exceeds original ${original_tx.amount}",
        )

    try:
        refund_tx = await payment_processor.process_refund(
            original_tx, request.reason, refund_amount
        )
        return _tx_to_response(refund_tx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Accounts router ──────────────────────────────────────────────────────────

@accounts_router.get(
    "",
    response_model=List[AccountResponse],
    summary="List accounts",
)
async def list_accounts():
    """List all synthetic accounts with current balances from DB."""
    try:
        with db_pool.connection() as conn:
            rows = conn.fetchall("SELECT * FROM accounts ORDER BY name")
            return [
                AccountResponse(
                    id=row["id"],
                    name=row["name"],
                    balance=row["balance"],
                    currency=row["currency"],
                    is_active=bool(row["is_active"]),
                )
                for row in rows
            ]
    except DBConnectionError as e:
        # Fall back to in-memory list if DB unavailable
        return [
            AccountResponse(
                id=a.id,
                name=a.name,
                balance=a.balance,
                currency=a.currency,
                is_active=a.is_active,
            )
            for a in SYNTHETIC_ACCOUNTS
        ]
