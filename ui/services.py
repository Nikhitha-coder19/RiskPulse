from .database import fetch_all, fetch_one


def get_dashboard_stats():
    """Return aggregate RiskPulse decision statistics."""

    stats = fetch_one(
        """
        SELECT
            COUNT(*) AS total_decisions,
            SUM(CASE WHEN action = 'ALLOW' THEN 1 ELSE 0 END) AS allow,
            SUM(CASE WHEN action = 'REVIEW' THEN 1 ELSE 0 END) AS review,
            SUM(CASE WHEN action = 'CHALLENGE' THEN 1 ELSE 0 END)
                AS challenge,
            SUM(CASE WHEN action = 'BLOCK' THEN 1 ELSE 0 END) AS block
        FROM risk_decisions
        """
    )

    return {
        key: int(value or 0)
        for key, value in stats.items()
    }


def _transaction_explorer_filters(
    transaction_id=None,
    customer_id=None,
    merchant_id=None,
    device_id=None,
    ip_id=None,
    action=None,
):
    """Build shared parameterized filters for explorer queries."""

    filters = []
    params = []

    search_fields = (
        ("t.transaction_id", transaction_id),
        ("t.customer_id", customer_id),
        ("t.merchant_id", merchant_id),
        ("t.device_id", device_id),
        ("t.ip_id", ip_id),
    )

    for field, value in search_fields:
        if value is not None and value != "":
            filters.append(f"{field} LIKE ?")
            params.append(f"%{value}%")

    if action is not None and action != "":
        filters.append("rd.action = ?")
        params.append(action)

    where_clause = (
        "WHERE " + " AND ".join(filters)
        if filters
        else ""
    )

    return where_clause, params


def _validate_pagination(page, page_size):
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("page must be a positive integer")

    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or page_size < 1
    ):
        raise ValueError("page_size must be a positive integer")


def get_transaction_explorer(
    page=1,
    page_size=50,
    transaction_id=None,
    customer_id=None,
    merchant_id=None,
    device_id=None,
    ip_id=None,
    action=None,
):
    """Return one page of transactions and their RiskPulse decisions."""

    _validate_pagination(page, page_size)

    where_clause, params = _transaction_explorer_filters(
        transaction_id=transaction_id,
        customer_id=customer_id,
        merchant_id=merchant_id,
        device_id=device_id,
        ip_id=ip_id,
        action=action,
    )

    params.extend([page_size, (page - 1) * page_size])

    return fetch_all(
        f"""
        SELECT
            t.transaction_id,
            t.timestamp,
            t.customer_id,
            t.merchant_id,
            t.device_id,
            t.ip_id,
            t.amount,
            rd.action,
            rd.final_risk_score
        FROM transactions AS t
        LEFT JOIN risk_decisions AS rd
            ON rd.transaction_id = t.transaction_id
        {where_clause}
        ORDER BY t.timestamp DESC, t.transaction_id DESC
        LIMIT ? OFFSET ?
        """,
        params,
    )


def count_transaction_explorer(
    transaction_id=None,
    customer_id=None,
    merchant_id=None,
    device_id=None,
    ip_id=None,
    action=None,
):
    """Return the total number of transactions matching explorer filters."""

    where_clause, params = _transaction_explorer_filters(
        transaction_id=transaction_id,
        customer_id=customer_id,
        merchant_id=merchant_id,
        device_id=device_id,
        ip_id=ip_id,
        action=action,
    )

    result = fetch_one(
        f"""
        SELECT COUNT(*) AS count
        FROM transactions AS t
        LEFT JOIN risk_decisions AS rd
            ON rd.transaction_id = t.transaction_id
        {where_clause}
        """,
        params,
    )

    return int(result["count"])


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


def _get_related_entity_history(entity_type, entity_id):
    """Return relationship history for one exact entity."""

    relationships = fetch_all(
        """
        SELECT
            related_type,
            related_id,
            first_seen,
            last_seen,
            count
        FROM entity_relationships
        WHERE entity_type = ?
          AND entity_id = ?
        ORDER BY related_type ASC, related_id ASC
        """,
        (entity_type, entity_id),
    )

    return {
        "entity_id": entity_id,
        "relationships": relationships,
    }


def get_transaction_investigation(transaction_id):
    """Return the complete read-only investigation payload for a transaction."""

    if (
        not isinstance(transaction_id, str)
        or not transaction_id.strip()
    ):
        raise ValueError("transaction_id must be a non-empty string")

    transaction_id = transaction_id.strip()

    transaction = fetch_one(
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

    if transaction is None:
        return None

    decision = fetch_one(
        """
        SELECT
            transaction_id,
            final_risk_score,
            action,
            traditional_probability,
            behavioral_probability,
            merchant_probability,
            created_at
        FROM risk_decisions
        WHERE transaction_id = ?
        """,
        (transaction_id,),
    )

    customer_history = fetch_one(
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
        (transaction["customer_id"],),
    )

    merchant_history = fetch_one(
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
        (transaction["merchant_id"],),
    )

    related_entities = {
        "device": _get_related_entity_history(
            "device",
            transaction["device_id"],
        ),
        "ip": _get_related_entity_history(
            "ip",
            transaction["ip_id"],
        ),
        "shipping_address": _get_related_entity_history(
            "shipping_address",
            transaction["shipping_address_id"],
        ),
        "billing_address": _get_related_entity_history(
            "billing_address",
            transaction["billing_address_id"],
        ),
    }

    return {
        "transaction": transaction,
        "decision": decision,
        "customer_history": customer_history,
        "merchant_history": merchant_history,
        "related_entities": related_entities,
    }


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