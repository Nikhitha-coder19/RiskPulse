import json
from datetime import datetime, timezone

from .database import fetch_all, fetch_one, get_connection


CHALLENGE_EVENT_TYPES = {
    "CHALLENGE_CREATED",
    "VERIFICATION_STARTED",
    "VERIFICATION_COMPLETED",
    "VERIFICATION_FAILED",
    "CHALLENGE_EXPIRED",
    "CHALLENGE_CANCELLED",
}


def _required_challenge_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )

    return value.strip()


def _optional_challenge_string(value, field_name):
    if value is None:
        return None

    return _required_challenge_string(value, field_name)


def _challenge_timestamp(created_at):
    if created_at is not None:
        return created_at

    return datetime.now(timezone.utc).isoformat()


def _insert_challenge_event(
    transaction_id,
    event_type,
    outcome=None,
    notes=None,
    created_at=None,
):
    transaction_id = _required_challenge_string(
        transaction_id,
        "transaction_id",
    )
    event_type = _required_challenge_string(event_type, "event_type")
    if event_type not in CHALLENGE_EVENT_TYPES:
        raise ValueError(
            f"Unsupported challenge event type: {event_type}"
        )
    outcome = _optional_challenge_string(outcome, "outcome")
    notes = _optional_challenge_string(notes, "notes")
    created_at = _challenge_timestamp(created_at)

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO challenge_events (
                transaction_id,
                event_type,
                outcome,
                notes,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                event_type,
                outcome,
                notes,
                created_at,
            ),
        )
        challenge_id = cursor.lastrowid

        row = connection.execute(
            """
            SELECT
                challenge_id,
                transaction_id,
                event_type,
                outcome,
                notes,
                created_at
            FROM challenge_events
            WHERE challenge_id = ?
            """,
            (challenge_id,),
        ).fetchone()

        return dict(row)


def create_challenge(transaction_id, created_at=None, notes=None):
    """Create a challenge lifecycle by recording its initial event."""

    transaction_id = _required_challenge_string(
        transaction_id,
        "transaction_id",
    )

    transaction = fetch_one(
        """
        SELECT transaction_id
        FROM transactions
        WHERE transaction_id = ?
        """,
        (transaction_id,),
    )

    if transaction is None:
        raise ValueError(
            f"Cannot create challenge: transaction '{transaction_id}' "
            "does not exist"
        )

    decision = fetch_one(
        """
        SELECT action
        FROM risk_decisions
        WHERE transaction_id = ?
        """,
        (transaction_id,),
    )

    if decision is None or decision["action"] != "CHALLENGE":
        raise ValueError(
            f"Cannot create challenge: transaction '{transaction_id}' "
            "does not have a CHALLENGE decision"
        )

    return _insert_challenge_event(
        transaction_id=transaction_id,
        event_type="CHALLENGE_CREATED",
        outcome="PENDING",
        notes=notes,
        created_at=created_at,
    )


def record_challenge_event(
    transaction_id,
    event_type,
    outcome=None,
    notes=None,
    created_at=None,
):
    """Record one challenge or verification lifecycle event."""

    transaction_id = _required_challenge_string(
        transaction_id,
        "transaction_id",
    )
    event_type = _required_challenge_string(event_type, "event_type")

    if event_type == "CHALLENGE_CREATED":
        raise ValueError(
            "CHALLENGE_CREATED can only be recorded by create_challenge"
        )

    if event_type not in CHALLENGE_EVENT_TYPES:
        raise ValueError(
            f"Unsupported challenge event type: {event_type}"
        )

    created_event = fetch_one(
        """
        SELECT challenge_id
        FROM challenge_events
        WHERE transaction_id = ?
          AND event_type = ?
        LIMIT 1
        """,
        (transaction_id, "CHALLENGE_CREATED"),
    )

    if created_event is None:
        raise ValueError(
            f"Cannot record challenge event: transaction '{transaction_id}' "
            "has no CHALLENGE_CREATED event"
        )

    return _insert_challenge_event(
        transaction_id=transaction_id,
        event_type=event_type,
        outcome=outcome,
        notes=notes,
        created_at=created_at,
    )


def get_challenge_events(transaction_id, limit=100):
    """Return bounded challenge events for one transaction."""

    transaction_id = _required_challenge_string(
        transaction_id,
        "transaction_id",
    )

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    return fetch_all(
        """
        SELECT
            challenge_id,
            transaction_id,
            event_type,
            outcome,
            notes,
            created_at
        FROM challenge_events
        WHERE transaction_id = ?
        ORDER BY created_at DESC, challenge_id DESC
        LIMIT ?
        """,
        (transaction_id, limit),
    )


def _required_audit_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )

    return value.strip()


def _optional_audit_string(value, field_name):
    if value is None:
        return None

    return _required_audit_string(value, field_name)


def log_audit_event(
    employee_id,
    action,
    transaction_id=None,
    entity_type=None,
    entity_id=None,
    metadata=None,
    created_at=None,
):
    """Insert and return one employee/system audit event."""

    employee_id = _required_audit_string(employee_id, "employee_id")
    action = _required_audit_string(action, "action")
    transaction_id = _optional_audit_string(
        transaction_id,
        "transaction_id",
    )
    entity_type = _optional_audit_string(entity_type, "entity_type")
    entity_id = _optional_audit_string(entity_id, "entity_id")

    metadata_json = (
        json.dumps(metadata)
        if metadata is not None
        else None
    )

    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO audit_log (
                employee_id,
                action,
                transaction_id,
                entity_type,
                entity_id,
                metadata,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                employee_id,
                action,
                transaction_id,
                entity_type,
                entity_id,
                metadata_json,
                created_at,
            ),
        )
        audit_id = cursor.lastrowid

        row = connection.execute(
            """
            SELECT
                audit_id,
                employee_id,
                action,
                transaction_id,
                entity_type,
                entity_id,
                metadata,
                created_at
            FROM audit_log
            WHERE audit_id = ?
            """,
            (audit_id,),
        ).fetchone()

        return dict(row)


def _validate_audit_limit(limit):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")


def get_audit_logs(
    transaction_id=None,
    employee_id=None,
    action=None,
    entity_type=None,
    entity_id=None,
    limit=100,
):
    """Return the newest matching audit events within a bounded limit."""

    _validate_audit_limit(limit)

    filters = []
    params = []

    for column, value in (
        ("transaction_id", transaction_id),
        ("employee_id", employee_id),
        ("action", action),
        ("entity_type", entity_type),
        ("entity_id", entity_id),
    ):
        if value is not None:
            filters.append(f"{column} = ?")
            params.append(value)

    where_clause = (
        "WHERE " + " AND ".join(filters)
        if filters
        else ""
    )

    params.append(limit)

    return fetch_all(
        f"""
        SELECT
            audit_id,
            employee_id,
            action,
            transaction_id,
            entity_type,
            entity_id,
            metadata,
            created_at
        FROM audit_log
        {where_clause}
        ORDER BY created_at DESC, audit_id DESC
        LIMIT ?
        """,
        params,
    )


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