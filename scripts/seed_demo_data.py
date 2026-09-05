import argparse
import json
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_engine import RiskEngine
from ui.database import DB_PATH
from ui.init_operations_db import initialize_operations_schema
from ui.services import get_transaction_explorer


DEMO_PREFIX = "demo_console_"


def _transaction(
    transaction_id,
    timestamp,
    customer_id,
    merchant_id,
    device_id,
    ip_id,
    shipping_address_id,
    billing_address_id,
    amount,
    quantity,
    payment_method,
    product_category,
    customer_age,
    account_age_days,
    customer_location,
    device_type,
):
    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "merchant_id": merchant_id,
        "timestamp": timestamp,
        "transaction_amount": amount,
        "quantity": quantity,
        "payment_method": payment_method,
        "product_category": product_category,
        "customer_age": customer_age,
        "account_age_days": account_age_days,
        "customer_location": customer_location,
        "device_id": device_id,
        "device_type": device_type,
        "ip_id": ip_id,
        "shipping_address_id": shipping_address_id,
        "billing_address_id": billing_address_id,
    }


def build_demo_transactions():
    return [
        _transaction(
            "demo_console_low_001",
            "2026-09-04T09:00:00",
            "demo_console_customer_low",
            "demo_console_merchant_standard",
            "demo_console_device_low",
            "demo_console_ip_low",
            "demo_console_ship_low",
            "demo_console_bill_low",
            42.50,
            1,
            "debit card",
            "books",
            38,
            980,
            "Pune",
            "desktop",
        ),
        _transaction(
            "demo_console_moderate_001",
            "2026-09-04T09:05:00",
            "demo_console_customer_moderate",
            "demo_console_merchant_standard",
            "demo_console_device_moderate",
            "demo_console_ip_moderate",
            "demo_console_ship_moderate",
            "demo_console_bill_moderate",
            680.00,
            2,
            "credit card",
            "home appliances",
            29,
            120,
            "Bengaluru",
            "mobile",
        ),
        _transaction(
            "demo_console_challenge_001",
            "2026-09-04T09:10:00",
            "demo_console_customer_challenge",
            "demo_console_merchant_electronics",
            "demo_console_device_challenge",
            "demo_console_ip_challenge",
            "demo_console_ship_challenge",
            "demo_console_bill_challenge",
            1850.00,
            1,
            "credit card",
            "electronics",
            23,
            45,
            "Hyderabad",
            "mobile",
        ),
        _transaction(
            "demo_console_block_candidate_001",
            "2026-09-04T09:15:00",
            "demo_console_customer_block",
            "demo_console_merchant_new",
            "demo_console_device_block",
            "demo_console_ip_block",
            "demo_console_ship_block",
            "demo_console_bill_block",
            4999.99,
            4,
            "prepaid card",
            "electronics",
            19,
            8,
            "Delhi",
            "mobile",
        ),
        _transaction(
            "demo_console_repeat_001",
            "2026-09-04T09:20:00",
            "demo_console_customer_repeat",
            "demo_console_merchant_repeat",
            "demo_console_device_repeat",
            "demo_console_ip_repeat",
            "demo_console_ship_repeat",
            "demo_console_bill_repeat",
            120.00,
            1,
            "credit card",
            "clothing",
            34,
            365,
            "Chennai",
            "desktop",
        ),
        _transaction(
            "demo_console_repeat_002",
            "2026-09-04T09:25:00",
            "demo_console_customer_repeat",
            "demo_console_merchant_repeat",
            "demo_console_device_repeat",
            "demo_console_ip_repeat",
            "demo_console_ship_repeat",
            "demo_console_bill_repeat",
            2100.00,
            3,
            "credit card",
            "electronics",
            34,
            365,
            "Chennai",
            "desktop",
        ),
    ]


def cleanup_demo_data():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "DELETE FROM challenge_events WHERE transaction_id LIKE ?",
            (f"{DEMO_PREFIX}%",),
        )
        connection.execute(
            "DELETE FROM risk_decisions WHERE transaction_id LIKE ?",
            (f"{DEMO_PREFIX}%",),
        )
        connection.execute(
            "DELETE FROM transactions WHERE transaction_id LIKE ?",
            (f"{DEMO_PREFIX}%",),
        )
        connection.execute(
            """
            DELETE FROM entity_relationships
            WHERE entity_id LIKE ? OR related_id LIKE ?
            """,
            (f"{DEMO_PREFIX}%", f"{DEMO_PREFIX}%"),
        )
        connection.execute(
            "DELETE FROM customers WHERE customer_id LIKE ?",
            (f"{DEMO_PREFIX}%",),
        )
        connection.execute(
            "DELETE FROM merchants WHERE merchant_id LIKE ?",
            (f"{DEMO_PREFIX}%",),
        )


def _read_demo_rows():
    with sqlite3.connect(DB_PATH) as connection:
        transactions = connection.execute(
            "SELECT transaction_id FROM transactions WHERE transaction_id LIKE ?",
            (f"{DEMO_PREFIX}%",),
        ).fetchall()
        decisions = connection.execute(
            """
            SELECT transaction_id, final_risk_score, action,
                   traditional_probability, behavioral_probability,
                   merchant_probability
            FROM risk_decisions
            WHERE transaction_id LIKE ?
            ORDER BY transaction_id
            """,
            (f"{DEMO_PREFIX}%",),
        ).fetchall()
        orphan_decisions = connection.execute(
            """
            SELECT COUNT(*)
            FROM risk_decisions AS rd
            LEFT JOIN transactions AS t
                ON t.transaction_id = rd.transaction_id
            WHERE rd.transaction_id LIKE ? AND t.transaction_id IS NULL
            """,
            (f"{DEMO_PREFIX}%",),
        ).fetchone()[0]
        repeat_state = connection.execute(
            """
            SELECT transaction_count
            FROM customers
            WHERE customer_id = 'demo_console_customer_repeat'
            """
        ).fetchone()

    return transactions, decisions, orphan_decisions, repeat_state


def seed_demo_data():
    initialize_operations_schema()
    cleanup_demo_data()

    engine = RiskEngine()
    results = []
    for transaction in build_demo_transactions():
        results.append(engine.process_transaction(transaction, fraud_label=None))

    transactions, decisions, orphan_decisions, repeat_state = _read_demo_rows()
    expected_ids = {
        transaction["transaction_id"]
        for transaction in build_demo_transactions()
    }
    actual_transaction_ids = {row[0] for row in transactions}
    actual_decision_ids = {row[0] for row in decisions}

    assert actual_transaction_ids == expected_ids
    assert actual_decision_ids == expected_ids
    assert orphan_decisions == 0
    assert repeat_state is not None and repeat_state[0] == 2

    repeat_results = [
        result for result in results
        if result["transaction_id"] in {
            "demo_console_repeat_001",
            "demo_console_repeat_002",
        }
    ]
    assert (
        repeat_results[0]["model_scores"]["behavioral"]["fraud_probability"]
        != repeat_results[1]["model_scores"]["behavioral"]["fraud_probability"]
    ), "Behavioral probability did not change for repeated customer."

    ui_rows = get_transaction_explorer(page=1, page_size=20)
    ui_ids = {row["transaction_id"] for row in ui_rows}
    assert expected_ids.issubset(ui_ids)

    print("RiskPulse demo data seeded through RiskEngine.process_transaction()")
    print(json.dumps([
        {
            "transaction_id": result["transaction_id"],
            "decision": result["decision"],
            "final_risk_score": result["fusion"]["final_risk_score"],
            "traditional_probability": result["fusion"]["traditional_probability"],
            "behavioral_probability": result["fusion"]["behavioral_probability"],
            "merchant_probability": result["fusion"]["merchant_probability"],
        }
        for result in results
    ], indent=2))
    print(f"Verified {len(actual_transaction_ids)} matching transactions and decisions.")
    print("Verified zero orphan demo decisions.")
    print("Verified repeated-customer behavioral state changed.")
    print("Verified UI transaction explorer visibility.")


def main():
    parser = argparse.ArgumentParser(description="Seed or remove RiskPulse demo data.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("seed", "cleanup"),
        default="seed",
    )
    args = parser.parse_args()

    initialize_operations_schema()
    if args.command == "cleanup":
        cleanup_demo_data()
        print(f"Removed data with prefix {DEMO_PREFIX}")
        return

    seed_demo_data()


if __name__ == "__main__":
    main()