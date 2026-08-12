"""
Pytest fixtures for PaymentPipeline test suite.
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.config import settings


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """FastAPI test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_db_pool():
    """Mock the DB pool to avoid needing SQLite in unit tests."""
    mock_conn = MagicMock()
    mock_conn.fetchone.return_value = None
    mock_conn.fetchall.return_value = []
    mock_conn.execute.return_value = MagicMock()
    mock_conn.commit.return_value = None

    with patch("app.database.db_pool") as mock_pool:
        mock_pool.acquire.return_value = mock_conn
        mock_pool.connection.return_value.__enter__ = lambda s: mock_conn
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_pool


@pytest.fixture
def sample_payment_request():
    return {
        "idempotency_key": "test-idem-key-001",
        "from_account": "acct_001",
        "to_account": "acct_merchant_A",
        "amount": 250.00,
        "currency": "USD",
        "method": "card",
        "metadata": {"test": True},
    }


@pytest.fixture
def sample_webhook_payload():
    return {
        "webhook_id": "wh_test_001",
        "event_type": "payment.confirmed",
        "transaction_id": "tx_test_001",
        "gateway": "stripe_mock",
        "payload": {"amount": 250.00, "status": "completed"},
        "retry_count": 0,
    }
