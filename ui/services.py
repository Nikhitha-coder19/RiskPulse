from .database import fetch_all, fetch_one


def get_dashboard_stats():
    """Return high-level transaction statistics."""

    total = fetch_one(
        "SELECT COUNT(*) AS count FROM transactions"
    )["count"]

    blocked = fetch_one(
        "SELECT COUNT(*) AS count FROM transactions WHERE is_fraud = 1"
    )["count"]

    return {
        "total_transactions": total,
        "blocked": blocked,
    }


def get_recent_transactions(limit=50):
    """Return recent transactions for the transaction explorer."""

    return fetch_all(
        """
        SELECT
            transaction_id,
            timestamp,
            customer_id,
            merchant_id,
            device_id,
            ip_id,
            amount,
            is_fraud
        FROM transactions
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    )


def get_transaction(transaction_id):
    """Return one transaction by ID."""

    return fetch_one(
        """
        SELECT
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
        FROM transactions
        WHERE transaction_id = ?
        """,
        (transaction_id,),
    )


def get_customer(customer_id):
    """Return customer runtime state."""

    return fetch_one(
        """
        SELECT
            customer_id,
            transaction_count,
            total_amount,
            max_amount,
            previous_timestamp,
            created_at
        FROM customers
        WHERE customer_id = ?
        """,
        (customer_id,),
    )


def get_merchant(merchant_id):
    """Return merchant runtime state."""

    return fetch_one(
        """
        SELECT
            merchant_id,
            transaction_count,
            total_amount,
            previous_amount,
            previous_timestamp,
            fraud_count,
            created_at
        FROM merchants
        WHERE merchant_id = ?
        """,
        (merchant_id,),
    )