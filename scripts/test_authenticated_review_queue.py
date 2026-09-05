import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui import auth, services


TRANSACTION_ID = "demo_console_repeat_002"
TEST_REASON = "Authenticated review queue test"


def cleanup(review_id):
    with sqlite3.connect(PROJECT_ROOT / "runtime" / "riskpulse_state.db") as connection:
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


def main():
    assert auth.authenticate("analyst.alex", "riskpulse-alex-2026") == "analyst.alex"
    assert auth.authenticate("analyst.alex", "invalid") is None

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
        assert services.count_review_queue(status="ACTIVE") == 0
        assert services.count_review_queue(status="RESOLVED") == 1

        resolved_investigation = services.get_transaction_investigation(
            TRANSACTION_ID
        )
        assert resolved_investigation["decision"]["action"] == "REVIEW"
        assert (
            resolved_investigation["review_case"]["analyst_decision"]
            == "ESCALATE"
        )

        with sqlite3.connect(PROJECT_ROOT / "runtime" / "riskpulse_state.db") as connection:
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
