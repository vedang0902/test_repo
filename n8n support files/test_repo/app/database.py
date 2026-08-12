"""
Mock PostgreSQL connection pool backed by SQLite.

Simulates real-world production database behaviour:
  - Limited connection pool (configurable, default 5)
  - Intermittent connection timeouts (configurable failure rate)
  - Pool exhaustion under load
  - Slow queries with random jitter
  - Connection acquisition metrics exposed to Prometheus

BUG SOURCES embedded here:
  - Pool exhaustion when too many concurrent transactions arrive
  - Intermittent OperationalError (simulates network blips / PG restarts)
  - No connection retry / exponential backoff → callers get immediate failures
"""
import sqlite3
import threading
import random
import time
import logging
import uuid
from contextlib import contextmanager
from typing import Optional
from datetime import datetime

from app.config import settings
from app.metrics import prometheus_metrics as m

logger = logging.getLogger("database")


class DBConnectionError(Exception):
    """Raised when a connection cannot be obtained from the pool."""


class PoolExhaustedError(DBConnectionError):
    """Raised when all connections in the pool are in use."""


class IntermittentConnectionError(DBConnectionError):
    """Simulates transient network/postgres error."""


class MockConnection:
    """Wraps a sqlite3 connection to mimic a psycopg2 connection interface."""

    def __init__(self, conn_id: str, db_path: str):
        self.conn_id = conn_id
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._acquired_at: Optional[float] = None
        self._query_count = 0
        logger.debug(f"[DB] Connection {conn_id} opened")

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        start = time.monotonic()
        cursor = self._conn.execute(query, params)
        elapsed_ms = (time.monotonic() - start) * 1000

        self._query_count += 1
        op = query.strip().split()[0].upper()
        m.db_query_duration_seconds.labels(operation=op).observe(elapsed_ms / 1000)

        if elapsed_ms > settings.database.slow_query_threshold_ms:
            m.db_slow_queries_total.labels(operation=op).inc()
            logger.warning(
                f"[DB] Slow query detected: {elapsed_ms:.1f}ms | op={op} | conn={self.conn_id}"
            )
        return cursor

    def executemany(self, query: str, params_seq):
        return self._conn.executemany(query, params_seq)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def fetchall(self, query: str, params: tuple = ()):
        return self.execute(query, params).fetchall()

    def fetchone(self, query: str, params: tuple = ()):
        return self.execute(query, params).fetchone()

    def close(self):
        self._conn.close()
        logger.debug(f"[DB] Connection {self.conn_id} closed")


class MockPostgresPool:
    """
    Simulates a PostgreSQL connection pool (like psycopg2's pool or asyncpg).

    Production bugs reproduced:
    1. Pool exhaustion — all connections checked out, new requests fail immediately
       (no queue/wait mechanism, mimicking a misconfigured PgBouncer)
    2. Intermittent connection failure — 12% of acquire() calls fail with a
       transient error (simulates PG checkpoint, network blip)
    3. Slow connection release — connections held for realistic durations
    """

    def __init__(self):
        cfg = settings.database
        self._max = cfg.pool_size
        self._db_path = cfg.sqlite_path
        self._failure_rate = cfg.connection_failure_rate
        self._pool: list[MockConnection] = []
        self._in_use: set[str] = set()
        self._lock = threading.Lock()
        self._total_acquired = 0
        self._total_exhausted = 0

        # Pre-warm pool
        for _ in range(self._max):
            conn = MockConnection(str(uuid.uuid4())[:8], self._db_path)
            self._pool.append(conn)

        self._init_schema()
        self._update_metrics()
        logger.info(
            f"[DB] Mock PostgreSQL pool initialized: size={self._max}, "
            f"failure_rate={self._failure_rate:.0%}, db={self._db_path}"
        )

    def _init_schema(self):
        with self._borrow_internal() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    balance REAL NOT NULL DEFAULT 0.0,
                    currency TEXT NOT NULL DEFAULT 'USD',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL,
                    from_account TEXT NOT NULL,
                    to_account TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    fraud_score REAL NOT NULL DEFAULT 0.0,
                    fee REAL NOT NULL DEFAULT 0.0,
                    net_amount REAL NOT NULL DEFAULT 0.0,
                    error_message TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    transaction_id TEXT NOT NULL,
                    merchant_id TEXT NOT NULL,
                    total_amount REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(transaction_id) REFERENCES transactions(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhook_events (
                    id TEXT PRIMARY KEY,
                    transaction_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    gateway TEXT NOT NULL,
                    processing_count INTEGER NOT NULL DEFAULT 0,
                    processed INTEGER NOT NULL DEFAULT 0,
                    received_at TEXT NOT NULL,
                    last_processed_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fraud_scores (
                    id TEXT PRIMARY KEY,
                    transaction_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    base_score REAL NOT NULL,
                    compounded_score REAL NOT NULL,
                    is_flagged INTEGER NOT NULL,
                    checked_at TEXT NOT NULL
                )
            """)
            conn.commit()
            logger.info("[DB] Schema initialised")

    @contextmanager
    def _borrow_internal(self):
        """Internal: borrow without failure injection (for schema init)."""
        with self._lock:
            conn = self._pool.pop(0)
            self._in_use.add(conn.conn_id)
        try:
            yield conn
        finally:
            with self._lock:
                self._in_use.discard(conn.conn_id)
                self._pool.append(conn)
            self._update_metrics()

    def acquire(self) -> MockConnection:
        """
        Check out a connection from the pool.

        BUG: No retry logic. If pool is exhausted or a transient error fires,
        callers get an immediate exception — just like a badly tuned PgBouncer.
        """
        # BUG injection: intermittent connection failure (simulates PG network blip)
        if random.random() < self._failure_rate:
            err_type = random.choice(["timeout", "auth_failure", "pg_restart"])
            m.db_connection_errors_total.labels(error_type=err_type).inc()
            m.app_errors_total.labels(component="database", error_type=err_type).inc()
            logger.error(
                f"[DB] Intermittent connection error: {err_type} "
                f"(pool_in_use={len(self._in_use)}/{self._max})"
            )
            raise IntermittentConnectionError(
                f"could not connect to server: {err_type}"
            )

        with self._lock:
            if not self._pool:
                # Pool exhausted
                self._total_exhausted += 1
                m.db_pool_exhausted_total.inc()
                m.app_errors_total.labels(
                    component="database", error_type="pool_exhausted"
                ).inc()
                logger.error(
                    f"[DB] Connection pool exhausted: all {self._max} connections in use "
                    f"(total exhaustions: {self._total_exhausted})"
                )
                raise PoolExhaustedError(
                    f"connection pool exhausted: {self._max}/{self._max} in use"
                )

            conn = self._pool.pop(0)
            conn._acquired_at = time.monotonic()
            self._in_use.add(conn.conn_id)
            self._total_acquired += 1

        self._update_metrics()
        logger.debug(
            f"[DB] Connection {conn.conn_id} acquired "
            f"(pool: {len(self._pool)} available, {len(self._in_use)} in use)"
        )
        return conn

    def release(self, conn: MockConnection):
        """Return a connection to the pool."""
        hold_time = (
            time.monotonic() - conn._acquired_at if conn._acquired_at else 0
        )
        with self._lock:
            self._in_use.discard(conn.conn_id)
            conn._acquired_at = None
            self._pool.append(conn)

        self._update_metrics()
        logger.debug(
            f"[DB] Connection {conn.conn_id} released after {hold_time:.3f}s "
            f"(pool: {len(self._pool)} available)"
        )

    @contextmanager
    def connection(self):
        """Context manager for safe connection borrow + release."""
        conn = self.acquire()
        try:
            yield conn
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self.release(conn)

    def _update_metrics(self):
        available = len(self._pool)
        in_use = len(self._in_use)
        m.db_pool_connections_available.set(available)
        m.db_pool_connections_in_use.set(in_use)

    def pool_status(self) -> dict:
        return {
            "available": len(self._pool),
            "in_use": len(self._in_use),
            "max": self._max,
            "total_acquired": self._total_acquired,
            "total_exhausted": self._total_exhausted,
        }


# Singleton pool — shared across the entire application
db_pool = MockPostgresPool()
