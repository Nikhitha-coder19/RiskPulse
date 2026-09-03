from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "runtime" / "riskpulse_state.db"


def get_connection():
    """Return a connection to the RiskPulse SQLite database."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def fetch_one(query, params=()):
    with get_connection() as connection:
        row = connection.execute(query, params).fetchone()
        return dict(row) if row else None


def fetch_all(query, params=()):
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def insert_risk_decision(
    transaction_id,
    final_risk_score,
    action,
    traditional_probability,
    behavioral_probability,
    merchant_probability,
    created_at,
):
    """
    Persist the RiskPulse decision for a transaction.

    This records what RiskPulse decided.
    It does NOT record ground truth.
    """

    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO risk_decisions (
                transaction_id,
                final_risk_score,
                action,
                traditional_probability,
                behavioral_probability,
                merchant_probability,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                final_risk_score,
                action,
                traditional_probability,
                behavioral_probability,
                merchant_probability,
                created_at,
            ),
        )

        connection.commit()