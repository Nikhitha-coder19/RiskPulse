from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "runtime" / "riskpulse_state.db"


def initialize_operations_schema():
    connection = sqlite3.connect(DB_PATH)

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS risk_decisions (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            final_risk_score REAL NOT NULL,
            action TEXT NOT NULL,
            traditional_probability REAL NOT NULL,
            behavioral_probability REAL NOT NULL,
            merchant_probability REAL NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(transaction_id)
        );

        CREATE TABLE IF NOT EXISTS challenge_events (
            challenge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            outcome TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS analyst_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            employee_id TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            confirmed_outcome TEXT,
            comments TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            action TEXT NOT NULL,
            transaction_id TEXT,
            entity_type TEXT,
            entity_id TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_risk_decisions_transaction
            ON risk_decisions(transaction_id);

        CREATE INDEX IF NOT EXISTS idx_risk_decisions_action
            ON risk_decisions(action);

        CREATE INDEX IF NOT EXISTS idx_risk_decisions_created
            ON risk_decisions(created_at);

        CREATE INDEX IF NOT EXISTS idx_challenge_transaction
            ON challenge_events(transaction_id);

        CREATE INDEX IF NOT EXISTS idx_feedback_transaction
            ON analyst_feedback(transaction_id);

        CREATE INDEX IF NOT EXISTS idx_audit_transaction
            ON audit_log(transaction_id);

        CREATE INDEX IF NOT EXISTS idx_audit_employee
            ON audit_log(employee_id);

        CREATE INDEX IF NOT EXISTS idx_audit_created
            ON audit_log(created_at);
        """
    )

    connection.commit()
    connection.close()

    print("RiskPulse operations schema initialized.")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    initialize_operations_schema()