import json
import sqlite3
from datetime import datetime, timezone

from .database import fetch_all, fetch_one, get_connection


REVIEW_STATUSES = {
    "OPEN",
    "IN_PROGRESS",
    "RESOLVED",
}

REVIEW_ANALYST_DECISIONS = {
    "ALLOW",
    "BLOCK",
    "ESCALATE",
}

FEEDBACK_TYPES = {
    "CONFIRMED_OUTCOME",
    "ANALYST_NOTE",
}

CONFIRMED_OUTCOMES = {
    "LEGITIMATE",
    "FRAUD",
}


def _review_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )

    return value.strip()


def _review_timestamp(created_at=None):
    if created_at is not None:
        return created_at

    return datetime.now(timezone.utc).isoformat()


def _feedback_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )

    return value.strip()


def _feedback_optional_string(value, field_name):
    if value is None:
        return None

    return _feedback_string(value, field_name)


def _feedback_filters(
    transaction_id=None,
    employee_id=None,
    feedback_type=None,
    confirmed_outcome=None,
):
    values = {
        "transaction_id": transaction_id,
        "employee_id": employee_id,
        "feedback_type": feedback_type,
        "confirmed_outcome": confirmed_outcome,
    }

    for field_name, value in values.items():
        if value is not None:
            values[field_name] = _feedback_string(value, field_name)

    if (
        values["feedback_type"] is not None
        and values["feedback_type"] not in FEEDBACK_TYPES
    ):
        raise ValueError(
            f"Invalid feedback type: {values['feedback_type']}"
        )

    if (
        values["confirmed_outcome"] is not None
        and values["confirmed_outcome"] not in CONFIRMED_OUTCOMES
    ):
        raise ValueError(
            "Invalid confirmed outcome: "
            f"{values['confirmed_outcome']}"
        )

    filters = []
    params = []

    for column, value in values.items():
        if value is not None:
            filters.append(f"{column} = ?")
            params.append(value)

    where_clause = (
        "WHERE " + " AND ".join(filters)
        if filters
        else ""
    )

    return where_clause, params


def record_feedback(
    transaction_id,
    employee_id,
    feedback_type,
    confirmed_outcome=None,
    comments=None,
    created_at=None,
):
    """Record validated feedback and audit its creation."""

    transaction_id = _feedback_string(transaction_id, "transaction_id")
    employee_id = _feedback_string(employee_id, "employee_id")
    feedback_type = _feedback_string(feedback_type, "feedback_type")
    comments = _feedback_optional_string(comments, "comments")

    if feedback_type not in FEEDBACK_TYPES:
        raise ValueError(
            f"Invalid feedback type: {feedback_type}"
        )

    if feedback_type == "CONFIRMED_OUTCOME":
        if confirmed_outcome is None:
            raise ValueError(
                "CONFIRMED_OUTCOME feedback requires confirmed_outcome"
            )
        confirmed_outcome = _feedback_string(
            confirmed_outcome,
            "confirmed_outcome",
        )
    elif confirmed_outcome is not None:
        raise ValueError(
            "ANALYST_NOTE feedback cannot include confirmed_outcome"
        )

    if (
        confirmed_outcome is not None
        and confirmed_outcome not in CONFIRMED_OUTCOMES
    ):
        raise ValueError(
            f"Invalid confirmed outcome: {confirmed_outcome}"
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
            f"Cannot record feedback: transaction '{transaction_id}' "
            "does not exist"
        )

    timestamp = _review_timestamp(created_at)

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyst_feedback (
                transaction_id,
                employee_id,
                feedback_type,
                confirmed_outcome,
                comments,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                employee_id,
                feedback_type,
                confirmed_outcome,
                comments,
                timestamp,
            ),
        )
        feedback_id = cursor.lastrowid

    feedback = fetch_one(
        """
        SELECT
            feedback_id,
            transaction_id,
            employee_id,
            feedback_type,
            confirmed_outcome,
            comments,
            created_at
        FROM analyst_feedback
        WHERE feedback_id = ?
        """,
        (feedback_id,),
    )

    log_audit_event(
        employee_id=employee_id,
        action="RECORD_FEEDBACK",
        transaction_id=transaction_id,
        entity_type="analyst_feedback",
        entity_id=str(feedback_id),
        metadata={
            "feedback_type": feedback_type,
            "confirmed_outcome": confirmed_outcome,
        },
    )

    return feedback


def get_feedback(
    transaction_id=None,
    employee_id=None,
    feedback_type=None,
    confirmed_outcome=None,
    page=1,
    page_size=50,
):
    """Return a bounded, newest-first page of feedback records."""

    _validate_pagination(page, page_size)
    where_clause, params = _feedback_filters(
        transaction_id=transaction_id,
        employee_id=employee_id,
        feedback_type=feedback_type,
        confirmed_outcome=confirmed_outcome,
    )
    params.extend([page_size, (page - 1) * page_size])

    return fetch_all(
        f"""
        SELECT
            feedback_id,
            transaction_id,
            employee_id,
            feedback_type,
            confirmed_outcome,
            comments,
            created_at
        FROM analyst_feedback
        {where_clause}
        ORDER BY created_at DESC, feedback_id DESC
        LIMIT ? OFFSET ?
        """,
        params,
    )


def count_feedback(
    transaction_id=None,
    employee_id=None,
    feedback_type=None,
    confirmed_outcome=None,
):
    """Return the count of feedback records matching the filters."""

    where_clause, params = _feedback_filters(
        transaction_id=transaction_id,
        employee_id=employee_id,
        feedback_type=feedback_type,
        confirmed_outcome=confirmed_outcome,
    )

    result = fetch_one(
        f"""
        SELECT COUNT(*) AS count
        FROM analyst_feedback
        {where_clause}
        """,
        params,
    )

    return int(result["count"])


def get_customer_protection(
    transaction_id,
    merchant_warning_threshold=0.75,
):
    """Return safe customer-facing protection guidance for a transaction."""

    transaction_id = _feedback_string(transaction_id, "transaction_id")

    investigation = get_transaction_investigation(transaction_id)

    if investigation is None:
        raise ValueError(
            f"Cannot get customer protection: transaction "
            f"'{transaction_id}' does not exist"
        )

    decision = investigation["decision"]

    if decision is None:
        raise ValueError(
            f"Cannot get customer protection: transaction "
            f"'{transaction_id}' has no persisted RiskPulse decision"
        )

    merchant_probability = float(decision["merchant_probability"])
    warning_present = (
        merchant_probability >= merchant_warning_threshold
    )

    warning_message = (
        "This merchant has unusual risk indicators. Proceed only if "
        "you trust this merchant."
        if warning_present
        else None
    )

    action_responses = {
        "ALLOW": (
            "PROCEED",
            "Your transaction can proceed.",
        ),
        "CHALLENGE": (
            "VERIFY",
            "Additional verification is required to continue.",
        ),
        "BLOCK": (
            "STOP",
            "This transaction could not be completed.",
        ),
        "REVIEW": (
            "REVIEW",
            "This transaction requires additional review before it can continue.",
        ),
    }

    try:
        customer_action, message = action_responses[decision["action"]]
    except KeyError as error:
        raise ValueError(
            f"Unsupported persisted RiskPulse action: {decision['action']}"
        ) from error

    response = {
        "transaction_id": transaction_id,
        "customer_action": customer_action,
        "message": message,
        "warning_present": warning_present,
        "warning_message": warning_message,
        "warning_options": (
            ["CONTINUE", "CANCEL"]
            if warning_present
            else []
        ),
    }

    if decision["action"] == "CHALLENGE":
        response["verification_required"] = True
        response["verification_completed"] = any(
            event["event_type"] == "VERIFICATION_COMPLETED"
            for event in get_challenge_events(transaction_id)
        )
    else:
        response["verification_required"] = False

    return response


def _review_id(review_id):
    if (
        isinstance(review_id, bool)
        or not isinstance(review_id, int)
        or review_id < 1
    ):
        raise ValueError("review_id must be a positive integer")

    return review_id


def _get_review_case(review_id):
    return fetch_one(
        """
        SELECT
            review_id,
            transaction_id,
            status,
            assigned_employee_id,
            review_reason,
            analyst_decision,
            analyst_comments,
            created_at,
            updated_at,
            resolved_at
        FROM review_queue
        WHERE review_id = ?
        """,
        (review_id,),
    )


def _validate_review_status(status):
    if status is not None and status not in REVIEW_STATUSES:
        raise ValueError(f"Invalid review status: {status}")


def _review_queue_filters(status=None, assigned_employee_id=None):
    _validate_review_status(status)

    if assigned_employee_id is not None:
        assigned_employee_id = _review_string(
            assigned_employee_id,
            "assigned_employee_id",
        )

    filters = []
    params = []

    if status is not None:
        filters.append("status = ?")
        params.append(status)

    if assigned_employee_id is not None:
        filters.append("assigned_employee_id = ?")
        params.append(assigned_employee_id)

    where_clause = (
        "WHERE " + " AND ".join(filters)
        if filters
        else ""
    )

    return where_clause, params


def create_review_case(transaction_id, review_reason, created_at=None):
    """Create one OPEN review case for a REVIEW risk decision."""

    transaction_id = _review_string(transaction_id, "transaction_id")
    review_reason = _review_string(review_reason, "review_reason")

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
            f"Cannot create review case: transaction '{transaction_id}' "
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

    if decision is None or decision["action"] != "REVIEW":
        raise ValueError(
            f"Cannot create review case: transaction '{transaction_id}' "
            "does not have a REVIEW decision"
        )

    timestamp = _review_timestamp(created_at)

    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO review_queue (
                    transaction_id,
                    status,
                    assigned_employee_id,
                    review_reason,
                    analyst_decision,
                    analyst_comments,
                    created_at,
                    updated_at,
                    resolved_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    "OPEN",
                    None,
                    review_reason,
                    None,
                    None,
                    timestamp,
                    timestamp,
                    None,
                ),
            )
            review_id = cursor.lastrowid
    except sqlite3.IntegrityError as error:
        raise ValueError(
            f"Review case already exists for transaction '{transaction_id}'"
        ) from error

    return _get_review_case(review_id)


def get_review_queue(
    status=None,
    assigned_employee_id=None,
    page=1,
    page_size=50,
):
    """Return a bounded, newest-first page of review cases."""

    _validate_pagination(page, page_size)
    where_clause, params = _review_queue_filters(
        status=status,
        assigned_employee_id=assigned_employee_id,
    )
    params.extend([page_size, (page - 1) * page_size])

    return fetch_all(
        f"""
        SELECT
            review_id,
            transaction_id,
            status,
            assigned_employee_id,
            review_reason,
            analyst_decision,
            analyst_comments,
            created_at,
            updated_at,
            resolved_at
        FROM review_queue
        {where_clause}
        ORDER BY created_at DESC, review_id DESC
        LIMIT ? OFFSET ?
        """,
        params,
    )


def count_review_queue(status=None, assigned_employee_id=None):
    """Return the count of review cases matching the supplied filters."""

    where_clause, params = _review_queue_filters(
        status=status,
        assigned_employee_id=assigned_employee_id,
    )

    result = fetch_one(
        f"""
        SELECT COUNT(*) AS count
        FROM review_queue
        {where_clause}
        """,
        params,
    )

    return int(result["count"])


def assign_review_case(review_id, employee_id):
    """Assign an OPEN review case and move it to IN_PROGRESS."""

    review_id = _review_id(review_id)
    employee_id = _review_string(employee_id, "employee_id")
    review_case = _get_review_case(review_id)

    if review_case is None:
        raise ValueError(f"Review case '{review_id}' does not exist")

    if review_case["status"] != "OPEN":
        raise ValueError(
            f"Review case '{review_id}' can only be assigned while OPEN"
        )

    updated_at = _review_timestamp()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE review_queue
            SET assigned_employee_id = ?,
                status = ?,
                updated_at = ?
            WHERE review_id = ?
              AND status = ?
            """,
            (
                employee_id,
                "IN_PROGRESS",
                updated_at,
                review_id,
                "OPEN",
            ),
        )

        if cursor.rowcount != 1:
            raise ValueError(
                f"Review case '{review_id}' could not be assigned because "
                "its state changed or it is no longer OPEN"
            )

    updated_case = _get_review_case(review_id)

    log_audit_event(
        employee_id=employee_id,
        action="ASSIGN_REVIEW_CASE",
        transaction_id=updated_case["transaction_id"],
        entity_type="review_case",
        entity_id=str(review_id),
        metadata={
            "review_id": review_id,
            "previous_status": "OPEN",
            "new_status": "IN_PROGRESS",
        },
    )

    return updated_case


def resolve_review_case(
    review_id,
    analyst_decision,
    analyst_comments=None,
):
    """Resolve an IN_PROGRESS review case with an analyst decision."""

    review_id = _review_id(review_id)
    analyst_decision = _review_string(
        analyst_decision,
        "analyst_decision",
    )

    if analyst_decision not in REVIEW_ANALYST_DECISIONS:
        raise ValueError(
            f"Invalid analyst decision: {analyst_decision}"
        )

    if analyst_comments is not None:
        analyst_comments = _review_string(
            analyst_comments,
            "analyst_comments",
        )

    review_case = _get_review_case(review_id)

    if review_case is None:
        raise ValueError(f"Review case '{review_id}' does not exist")

    if review_case["status"] != "IN_PROGRESS":
        raise ValueError(
            f"Review case '{review_id}' can only be resolved while "
            "IN_PROGRESS"
        )

    timestamp = _review_timestamp()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE review_queue
            SET status = ?,
                analyst_decision = ?,
                analyst_comments = ?,
                updated_at = ?,
                resolved_at = ?
            WHERE review_id = ?
              AND status = ?
            """,
            (
                "RESOLVED",
                analyst_decision,
                analyst_comments,
                timestamp,
                timestamp,
                review_id,
                "IN_PROGRESS",
            ),
        )

        if cursor.rowcount != 1:
            raise ValueError(
                f"Review case '{review_id}' could not be resolved because "
                "its state changed or it is no longer IN_PROGRESS"
            )

    resolved_case = _get_review_case(review_id)

    log_audit_event(
        employee_id=resolved_case["assigned_employee_id"],
        action="RESOLVE_REVIEW_CASE",
        transaction_id=resolved_case["transaction_id"],
        entity_type="review_case",
        entity_id=str(review_id),
        metadata={
            "review_id": review_id,
            "previous_status": "IN_PROGRESS",
            "new_status": "RESOLVED",
            "analyst_decision": analyst_decision,
        },
    )

    return resolved_case


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