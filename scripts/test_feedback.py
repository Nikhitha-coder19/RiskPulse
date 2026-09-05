import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui import services
from ui import components
from ui.init_operations_db import initialize_operations_schema


DB_PATH = PROJECT_ROOT / "runtime" / "riskpulse_state.db"
TRANSACTION_ID = "demo_console_challenge_001"
EMPLOYEE_A = "analyst.alex"
EMPLOYEE_B = "analyst.priya"


def fetch_snapshot():
    with sqlite3.connect(DB_PATH) as connection:
        return {
            "transaction": connection.execute(
                "SELECT is_fraud FROM transactions WHERE transaction_id = ?",
                (TRANSACTION_ID,),
            ).fetchone(),
            "decision": connection.execute(
                """
                SELECT action, final_risk_score, created_at
                FROM risk_decisions
                WHERE transaction_id = ?
                """,
                (TRANSACTION_ID,),
            ).fetchone(),
            "review": connection.execute(
                """
                SELECT review_id, status, assigned_employee_id, analyst_decision
                FROM review_queue
                WHERE transaction_id = ?
                """,
                (TRANSACTION_ID,),
            ).fetchone(),
            "challenges": connection.execute(
                """
                SELECT challenge_id, event_type, outcome, notes, created_at
                FROM challenge_events
                WHERE transaction_id = ?
                ORDER BY challenge_id
                """,
                (TRANSACTION_ID,),
            ).fetchall(),
        }


def expect_value_error(callback, message):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError(message)


def test_compact_feedback_entrypoint():
    class Column:
        def metric(self, *args, **kwargs):
            pass

    class StreamlitStub:
        def __getattr__(self, name):
            if name == "columns":
                return lambda count: [Column() for _ in range(count)]
            if name == "button":
                return lambda label, **kwargs: "feedback_entry_" in kwargs.get("key", "")
            return lambda *args, **kwargs: None

    original_streamlit = components.st
    components.st = StreamlitStub()
    try:
        for action in ("ALLOW", "BLOCK", "REVIEW", "CHALLENGE"):
            investigation = {
                "transaction": {
                    "transaction_id": f"compact_{action.lower()}",
                    "timestamp": "2026-09-05T00:00:00",
                    "customer_id": "customer",
                    "merchant_id": "merchant",
                    "device_id": "device",
                    "ip_id": "ip",
                    "shipping_address_id": "ship",
                    "billing_address_id": "bill",
                    "amount": 10.0,
                },
                "decision": {
                    "action": action,
                    "final_risk_score": 0.5,
                    "traditional_probability": 0.2,
                    "behavioral_probability": 0.2,
                    "merchant_probability": 0.2,
                },
                "review_case": None,
                "challenge": None,
                "feedback_count": 0,
            }
            result = components.render_investigation(
                investigation,
                {"warning_present": False},
            )
            assert result == {"action": "view_feedback"}
    finally:
        components.st = original_streamlit


def main():
    initialize_operations_schema()
    test_compact_feedback_entrypoint()
    before = fetch_snapshot()
    created_feedback_ids = []

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "DELETE FROM analyst_feedback WHERE transaction_id = ?",
            (TRANSACTION_ID,),
        )
        connection.execute(
            "DELETE FROM audit_log WHERE entity_type = 'analyst_feedback' "
            "AND transaction_id = ?",
            (TRANSACTION_ID,),
        )

    try:
        actual_only = services.record_feedback(
            TRANSACTION_ID,
            EMPLOYEE_A,
            confirmed_outcome="FRAUD",
            created_at="2026-09-05T10:00:00+00:00",
        )
        created_feedback_ids.append(actual_only["feedback_id"])
        assert actual_only["confirmed_outcome"] == "FRAUD"
        assert actual_only["recommended_action"] is None

        recommendation_only = services.record_feedback(
            TRANSACTION_ID,
            EMPLOYEE_B,
            recommended_action="CHALLENGE",
            created_at="2026-09-05T11:00:00+00:00",
        )
        created_feedback_ids.append(recommendation_only["feedback_id"])
        assert recommendation_only["recommended_action"] == "CHALLENGE"
        assert recommendation_only["confirmed_outcome"] is None

        complete = services.record_feedback(
            TRANSACTION_ID,
            EMPLOYEE_A,
            confirmed_outcome="LEGITIMATE",
            recommended_action="ALLOW",
            comments="Customer completed verification successfully.",
            created_at="2026-09-05T12:00:00+00:00",
        )
        created_feedback_ids.append(complete["feedback_id"])

        legacy = services.record_feedback(
            TRANSACTION_ID,
            EMPLOYEE_B,
            feedback_type="CONFIRMED_OUTCOME",
            confirmed_outcome="FRAUD",
            comments="Legacy feedback call remains supported.",
            created_at="2026-09-05T13:00:00+00:00",
        )
        created_feedback_ids.append(legacy["feedback_id"])
        assert legacy["feedback_type"] == "CONFIRMED_OUTCOME"

        expect_value_error(
            lambda: services.record_feedback(
                TRANSACTION_ID,
                EMPLOYEE_A,
                confirmed_outcome="UNKNOWN",
            ),
            "Invalid actual outcome was accepted",
        )
        expect_value_error(
            lambda: services.record_feedback(
                TRANSACTION_ID,
                EMPLOYEE_A,
                recommended_action="UNKNOWN",
            ),
            "Invalid recommended action was accepted",
        )
        expect_value_error(
            lambda: services.record_feedback(
                TRANSACTION_ID,
                "not.an.employee",
                comments="Unknown employee",
            ),
            "Unknown employee identity was accepted",
        )
        expect_value_error(
            lambda: services.record_feedback(
                TRANSACTION_ID,
                EMPLOYEE_A,
            ),
            "Empty feedback was accepted",
        )

        history = services.get_feedback(
            transaction_id=TRANSACTION_ID,
            page=1,
            page_size=20,
        )
        assert [row["feedback_id"] for row in history] == list(
            reversed(created_feedback_ids)
        )
        assert len(history) == 4
        assert services.count_feedback(transaction_id=TRANSACTION_ID) == 4

        with sqlite3.connect(DB_PATH) as connection:
            audit_rows = connection.execute(
                """
                SELECT action, employee_id, transaction_id, entity_type, entity_id
                FROM audit_log
                WHERE entity_type = 'analyst_feedback'
                  AND transaction_id = ?
                ORDER BY audit_id
                """,
                (TRANSACTION_ID,),
            ).fetchall()
        assert len(audit_rows) == 4
        assert all(row[0] == "RECORD_FEEDBACK" for row in audit_rows)
        assert all(row[2] == TRANSACTION_ID for row in audit_rows)
        assert all(row[3] == "analyst_feedback" for row in audit_rows)

        after = fetch_snapshot()
        assert after == before, "Feedback changed protected transaction state"
        print("FEEDBACK SERVICE TEST: PASS")
    finally:
        with sqlite3.connect(DB_PATH) as connection:
            if created_feedback_ids:
                placeholders = ",".join("?" for _ in created_feedback_ids)
                connection.execute(
                    f"DELETE FROM analyst_feedback WHERE feedback_id IN ({placeholders})",
                    created_feedback_ids,
                )
                connection.execute(
                    f"DELETE FROM audit_log WHERE entity_type = 'analyst_feedback' "
                    f"AND entity_id IN ({placeholders})",
                    [str(feedback_id) for feedback_id in created_feedback_ids],
                )


if __name__ == "__main__":
    main()
