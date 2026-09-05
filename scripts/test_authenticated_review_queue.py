import sqlite3
import sys
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "runtime" / "riskpulse_state.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui import auth, services


TRANSACTION_ID = f"test_review_queue_{uuid.uuid4().hex[:12]}"
TEST_REASON = "Authenticated review queue test"


def cleanup(review_id):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "DELETE FROM review_queue WHERE review_id = ?",
            (review_id,),
        )
        connection.execute(
            """
            DELETE FROM audit_log
            WHERE entity_type = 'review_case' AND entity_id = ?
            """,
            (str(review_id),),
        )
        connection.execute(
            "DELETE FROM risk_decisions WHERE transaction_id = ?",
            (TRANSACTION_ID,),
        )
        connection.execute(
            "DELETE FROM transactions WHERE transaction_id = ?",
            (TRANSACTION_ID,),
        )


def ensure_transaction_and_decision():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO transactions (
                transaction_id,
                timestamp,
                customer_id,
                merchant_id,
                device_id,
                ip_id,
                shipping_address_id,
                billing_address_id,
                amount,
                is_fraud
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                TRANSACTION_ID,
                "2026-09-05T12:00:00",
                "test_review_customer",
                "test_review_merchant",
                "test_review_device",
                "test_review_ip",
                "test_review_ship",
                "test_review_bill",
                100.0,
                None,
            ),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO risk_decisions (
                transaction_id,
                final_risk_score,
                action,
                traditional_probability,
                behavioral_probability,
                merchant_probability,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                TRANSACTION_ID,
                0.72,
                "REVIEW",
                0.40,
                0.75,
                0.60,
                "2026-09-05T12:00:00",
            ),
        )


def main():
    assert auth.authenticate("analyst.alex", "riskpulse-alex-2026") == "analyst.alex"
    assert auth.authenticate("analyst.alex", "invalid") is None

    ensure_transaction_and_decision()
    case = services.create_review_case(TRANSACTION_ID, TEST_REASON)
    try:
        assigned = services.assign_review_case(case["review_id"], "analyst.alex")
        assert assigned["assigned_employee_id"] == "analyst.alex"
        assert assigned["status"] == "IN_PROGRESS"

        in_progress_investigation = services.get_transaction_investigation(
            TRANSACTION_ID
        )
        assert in_progress_investigation["decision"]["action"] == "REVIEW"
        assert (
            in_progress_investigation["review_case"]["status"]
            == "IN_PROGRESS"
        )

        try:
            services.resolve_review_case(
                case["review_id"],
                "analyst.priya",
                "ALLOW",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Unauthorized resolution was accepted")

        resolved = services.resolve_review_case(
            case["review_id"],
            "analyst.alex",
            "ESCALATE",
            "Authenticated analyst resolution",
        )
        assert resolved["status"] == "RESOLVED"
        assert resolved["analyst_decision"] == "ESCALATE"

        with sqlite3.connect(DB_PATH) as connection:
            transaction_queue_counts = connection.execute(
                """
                SELECT status, COUNT(*)
                FROM review_queue
                WHERE transaction_id = ?
                GROUP BY status
                ORDER BY status
                """,
                (TRANSACTION_ID,),
            ).fetchall()

        assert transaction_queue_counts == [("RESOLVED", 1)]

        resolved_investigation = services.get_transaction_investigation(
            TRANSACTION_ID
        )
        assert resolved_investigation["decision"]["action"] == "REVIEW"
        assert (
            resolved_investigation["review_case"]["analyst_decision"]
            == "ESCALATE"
        )

        with sqlite3.connect(DB_PATH) as connection:
            audit_rows = connection.execute(
                """
                SELECT action, employee_id
                FROM audit_log
                WHERE entity_type = 'review_case' AND entity_id = ?
                ORDER BY audit_id
                """,
                (str(case["review_id"]),),
            ).fetchall()

        assert audit_rows == [
            ("ASSIGN_REVIEW_CASE", "analyst.alex"),
            ("RESOLVE_REVIEW_CASE", "analyst.alex"),
        ]
        print("AUTHENTICATED REVIEW QUEUE TEST: PASS")
    finally:
        cleanup(case["review_id"])


if __name__ == "__main__":
    main()
