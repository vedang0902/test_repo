"""
Seed script — populates the database with initial data.
Run once before starting the app if you want a pre-populated state.

Usage: python scripts/seed.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import uuid
from datetime import datetime, timedelta
import random

DB_PATH = os.getenv("SQLITE_PATH", "./paymentpipeline.db")

ACCOUNTS = [
    ("acct_001", "Arjun Sharma",       50000.0, "USD"),
    ("acct_002", "Priya Patel",        32000.0, "USD"),
    ("acct_003", "Rohan Mehta",        18500.0, "USD"),
    ("acct_004", "Ananya Singh",       75000.0, "USD"),
    ("acct_005", "Vikram Nair",        12000.0, "USD"),
    ("acct_006", "Kavya Reddy",        95000.0, "USD"),
    ("acct_007", "Siddharth Joshi",    28000.0, "USD"),
    ("acct_008", "Neha Gupta",         61000.0, "USD"),
    ("acct_merchant_A", "Shopify Store A",       200000.0, "USD"),
    ("acct_merchant_B", "Razorpay Merchant B",   150000.0, "USD"),
]

STATUSES = ["completed", "failed", "partial_commit", "fraud_blocked"]
METHODS  = ["card", "bank_transfer", "wallet", "crypto"]
CURRENCIES = ["USD", "EUR", "GBP", "INR", "SGD"]


def init_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY, name TEXT NOT NULL,
            balance REAL NOT NULL DEFAULT 0.0, currency TEXT NOT NULL DEFAULT 'USD',
            is_active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL,
            from_account TEXT NOT NULL, to_account TEXT NOT NULL,
            amount REAL NOT NULL, currency TEXT NOT NULL, method TEXT NOT NULL,
            status TEXT NOT NULL, fraud_score REAL NOT NULL DEFAULT 0.0,
            fee REAL NOT NULL DEFAULT 0.0, net_amount REAL NOT NULL DEFAULT 0.0,
            error_message TEXT, metadata TEXT, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, completed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY, transaction_id TEXT NOT NULL,
            merchant_id TEXT NOT NULL, total_amount REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS webhook_events (
            id TEXT PRIMARY KEY, transaction_id TEXT NOT NULL, event_type TEXT NOT NULL,
            gateway TEXT NOT NULL, processing_count INTEGER NOT NULL DEFAULT 0,
            processed INTEGER NOT NULL DEFAULT 0, received_at TEXT NOT NULL,
            last_processed_at TEXT
        );
    """)
    conn.commit()
    print("Schema initialised")


def seed_accounts(conn):
    for acct_id, name, balance, currency in ACCOUNTS:
        conn.execute(
            "INSERT OR IGNORE INTO accounts (id, name, balance, currency, is_active, created_at) VALUES (?, ?, ?, ?, 1, ?)",
            (acct_id, name, balance, currency, datetime.utcnow().isoformat()),
        )
    conn.commit()
    print(f"Seeded {len(ACCOUNTS)} accounts")


def seed_transactions(conn, count=200):
    sender_ids   = [a[0] for a in ACCOUNTS[:8]]
    receiver_ids = [a[0] for a in ACCOUNTS[8:]]

    for i in range(count):
        tx_id    = str(uuid.uuid4())
        idem_key = f"idem_{uuid.uuid4().hex[:16]}"
        status   = random.choices(STATUSES, weights=[0.70, 0.12, 0.08, 0.10])[0]
        amount   = round(random.uniform(5, 15000), 2)
        fee      = amount * 0.029
        net      = amount - fee
        created  = (datetime.utcnow() - timedelta(minutes=random.randint(0, 480))).isoformat()

        conn.execute(
            """INSERT OR IGNORE INTO transactions
               (id, idempotency_key, from_account, to_account, amount, currency, method, status,
                fraud_score, fee, net_amount, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tx_id, idem_key,
                random.choice(sender_ids), random.choice(receiver_ids),
                amount, random.choice(CURRENCIES), random.choice(METHODS), status,
                round(random.uniform(0, 0.95), 3), fee, net, created, created,
            ),
        )

    conn.commit()
    print(f"Seeded {count} transactions")


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    init_schema(conn)
    seed_accounts(conn)
    seed_transactions(conn)
    conn.close()
    print(f"Done. DB: {DB_PATH}")
