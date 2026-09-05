import sqlite3
import uuid
from pathlib import Path

from risk_engine import RiskEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "runtime" / "riskpulse_state.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

TEST_PREFIX = f"runtime_integration_{uuid.uuid4().hex[:8]}"

CUSTOMER_ID = f"{TEST_PREFIX}_customer"
MERCHANT_ID = f"{TEST_PREFIX}_merchant"
DEVICE_ID = f"{TEST_PREFIX}_device"
IP_ID = f"{TEST_PREFIX}_ip"
SHIPPING_ID = f"{TEST_PREFIX}_ship"
BILLING_ID = f"{TEST_PREFIX}_bill"

TRANSACTION_1 = f"{TEST_PREFIX}_001"
TRANSACTION_2 = f"{TEST_PREFIX}_002"


def make_transaction(transaction_id, timestamp):
    return {
        "transaction_id": transaction_id,
        "customer_id": CUSTOMER_ID,
        "merchant_id": MERCHANT_ID,
        "timestamp": timestamp,
        "transaction_amount": 2500.0,
        "quantity": 2,
        "payment_method": "credit card",
        "product_category": "electronics",
        "customer_age": 30,
        "account_age_days": 180,
        "customer_location": "New York",
        "device_id": DEVICE_ID,
        "device_type": "desktop",
        "ip_id": IP_ID,
        "shipping_address_id": SHIPPING_ID,
        "billing_address_id": BILLING_ID,
    }


def cleanup():
    conn = sqlite3.connect(DB_PATH)

    try:
        conn.execute(
            """
            DELETE FROM challenge_events
            WHERE transaction_id IN (?, ?)
            """,
            (TRANSACTION_1, TRANSACTION_2),
        )

        conn.execute(
            """
            DELETE FROM risk_decisions
            WHERE transaction_id IN (?, ?)
            """,
            (TRANSACTION_1, TRANSACTION_2),
        )

        conn.execute(
            """
            DELETE FROM transactions
            WHERE transaction_id IN (?, ?)
            """,
            (TRANSACTION_1, TRANSACTION_2),
        )

        conn.execute(
            """
            DELETE FROM entity_relationships
            WHERE entity_id LIKE ?
               OR related_id LIKE ?
            """,
            (f"{TEST_PREFIX}%", f"{TEST_PREFIX}%"),
        )

        conn.execute(
            """
            DELETE FROM customers
            WHERE customer_id = ?
            """,
            (CUSTOMER_ID,),
        )

        conn.execute(
            """
            DELETE FROM merchants
            WHERE merchant_id = ?
            """,
            (MERCHANT_ID,),
        )

        conn.commit()

    finally:
        conn.close()


def assert_cleanup_complete():
    conn = sqlite3.connect(DB_PATH)

    try:
        remaining_transactions = conn.execute(
            """
            SELECT COUNT(*)
            FROM transactions
            WHERE transaction_id LIKE ?
            """,
            (f"{TEST_PREFIX}%",),
        ).fetchone()[0]

        remaining_decisions = conn.execute(
            """
            SELECT COUNT(*)
            FROM risk_decisions
            WHERE transaction_id LIKE ?
            """,
            (f"{TEST_PREFIX}%",),
        ).fetchone()[0]

        remaining_challenges = conn.execute(
            """
            SELECT COUNT(*)
            FROM challenge_events
            WHERE transaction_id LIKE ?
            """,
            (f"{TEST_PREFIX}%",),
        ).fetchone()[0]

        remaining_customers = conn.execute(
            """
            SELECT COUNT(*)
            FROM customers
            WHERE customer_id LIKE ?
            """,
            (f"{TEST_PREFIX}%",),
        ).fetchone()[0]

        remaining_merchants = conn.execute(
            """
            SELECT COUNT(*)
            FROM merchants
            WHERE merchant_id LIKE ?
            """,
            (f"{TEST_PREFIX}%",),
        ).fetchone()[0]

        remaining_relationships = conn.execute(
            """
            SELECT COUNT(*)
            FROM entity_relationships
            WHERE entity_id LIKE ?
               OR related_id LIKE ?
            """,
            (f"{TEST_PREFIX}%", f"{TEST_PREFIX}%"),
        ).fetchone()[0]

        assert remaining_transactions == 0, (
            "Integration-test transactions were not cleaned up."
        )
        assert remaining_decisions == 0, (
            "Integration-test risk decisions were not cleaned up."
        )
        assert remaining_challenges == 0, (
            "Integration-test challenge events were not cleaned up."
        )
        assert remaining_customers == 0, (
            "Integration-test customers were not cleaned up."
        )
        assert remaining_merchants == 0, (
            "Integration-test merchants were not cleaned up."
        )
        assert remaining_relationships == 0, (
            "Integration-test relationships were not cleaned up."
        )

    finally:
        conn.close()


def assert_transaction_exists(transaction_id):
    conn = sqlite3.connect(DB_PATH)

    try:
        row = conn.execute(
            """
            SELECT transaction_id
            FROM transactions
            WHERE transaction_id = ?
            """,
            (transaction_id,),
        ).fetchone()

        assert row is not None, (
            f"Transaction {transaction_id} was not persisted."
        )

    finally:
        conn.close()


def get_customer_count():
    conn = sqlite3.connect(DB_PATH)

    try:
        row = conn.execute(
            """
            SELECT transaction_count
            FROM customers
            WHERE customer_id = ?
            """,
            (CUSTOMER_ID,),
        ).fetchone()

        return row[0] if row else 0

    finally:
        conn.close()


def main():
    print("RiskPulse runtime integration test")
    print("=" * 45)

    # Ensure this test starts from a clean state.
    cleanup()

    engine = RiskEngine()

    tx1 = make_transaction(
        TRANSACTION_1,
        "2026-09-03T16:00:00",
    )

    tx2 = make_transaction(
        TRANSACTION_2,
        "2026-09-03T16:05:00",
    )

    try:
        # ---------------------------------------------------------
        # Transaction 1
        # ---------------------------------------------------------
        print("\n[1] Processing first transaction...")

        result1 = engine.process_transaction(
            tx1,
            fraud_label=None,
        )

        assert result1["decision"] in {
            "ALLOW",
            "REVIEW",
            "CHALLENGE",
            "BLOCK",
        }

        assert result1["state_update"]["updated"] is True
        assert result1["state_update"]["fraud_label_recorded"] is False

        print(
            f"    Decision: {result1['decision']}"
        )

        print(
            f"    Behavioral probability: "
            f"{result1['model_scores']['behavioral']['fraud_probability']:.6f}"
        )

        print(
            f"    Merchant probability: "
            f"{result1['model_scores']['merchant']['fraud_probability']:.6f}"
        )

        # The first transaction must now exist in state.
        assert_transaction_exists(TRANSACTION_1)

        customer_count_after_first = get_customer_count()

        assert customer_count_after_first == 1, (
            "Customer history was not updated after first transaction."
        )

        print("    ✓ First transaction persisted")
        print("    ✓ Customer state updated")

        # ---------------------------------------------------------
        # Transaction 2
        # ---------------------------------------------------------
        print("\n[2] Processing second transaction...")
        print("    Same customer / merchant / device / IP / addresses")

        result2 = engine.process_transaction(
            tx2,
            fraud_label=None,
        )

        assert result2["decision"] in {
            "ALLOW",
            "REVIEW",
            "CHALLENGE",
            "BLOCK",
        }

        assert result2["state_update"]["updated"] is True
        assert result2["state_update"]["fraud_label_recorded"] is False

        print(
            f"    Decision: {result2['decision']}"
        )

        print(
            f"    Behavioral probability: "
            f"{result2['model_scores']['behavioral']['fraud_probability']:.6f}"
        )

        print(
            f"    Merchant probability: "
            f"{result2['model_scores']['merchant']['fraud_probability']:.6f}"
        )

        # ---------------------------------------------------------
        # Persistence checks
        # ---------------------------------------------------------
        assert_transaction_exists(TRANSACTION_2)

        customer_count_after_second = get_customer_count()

        assert customer_count_after_second == 2, (
            "Customer history did not increase to two transactions."
        )

        print("    ✓ Second transaction persisted")
        print("    ✓ Customer history increased to 2")

        # ---------------------------------------------------------
        # Statefulness check
        # ---------------------------------------------------------
        behavioral_1 = result1["model_scores"]["behavioral"][
            "fraud_probability"
        ]

        behavioral_2 = result2["model_scores"]["behavioral"][
            "fraud_probability"
        ]

        merchant_1 = result1["model_scores"]["merchant"][
            "fraud_probability"
        ]

        merchant_2 = result2["model_scores"]["merchant"][
            "fraud_probability"
        ]

        print("\n[3] Statefulness verification")

        print(
            f"    Behavioral: "
            f"{behavioral_1:.6f} → {behavioral_2:.6f}"
        )

        print(
            f"    Merchant:   "
            f"{merchant_1:.6f} → {merchant_2:.6f}"
        )

        assert behavioral_1 != behavioral_2, (
            "Behavioral probability did not change after history update."
        )

        assert merchant_1 != merchant_2, (
            "Merchant probability did not change after history update."
        )

        print("    ✓ Behavioral model responded to runtime history")
        print("    ✓ Merchant model responded to runtime history")

        # ---------------------------------------------------------
        # Feedback contamination check
        # ---------------------------------------------------------
        print("\n[4] Ground-truth safety verification")

        conn = sqlite3.connect(DB_PATH)

        try:
            fraud_rows = conn.execute(
                """
                SELECT transaction_id, is_fraud
                FROM transactions
                WHERE transaction_id IN (?, ?)
                """,
                (TRANSACTION_1, TRANSACTION_2),
            ).fetchall()

        finally:
            conn.close()

        assert len(fraud_rows) == 2

        for transaction_id, is_fraud in fraud_rows:
            assert is_fraud is None, (
                f"{transaction_id} incorrectly recorded model prediction "
                "as ground truth."
            )

        print("    ✓ Model predictions were not recorded as fraud labels")

        # ---------------------------------------------------------
        # Final integration checks
        # ---------------------------------------------------------
        print("\n[5] Integration checks")

        assert "fusion" in result1
        assert "fusion" in result2

        assert 0.0 <= result1["fusion"]["final_risk_score"] <= 1.0
        assert 0.0 <= result2["fusion"]["final_risk_score"] <= 1.0

        assert result1["decision"] == result1["fusion"]["action"]
        assert result2["decision"] == result2["fusion"]["action"]

        print("    ✓ All three models executed")
        print("    ✓ Fusion executed")
        print("    ✓ Final decision generated")
        print("    ✓ State updated after decision")

        print("\n" + "=" * 45)
        print("RUNTIME INTEGRATION TEST: PASS")
        print("=" * 45)

    finally:
        cleanup()
        assert_cleanup_complete()


if __name__ == "__main__":
    main()