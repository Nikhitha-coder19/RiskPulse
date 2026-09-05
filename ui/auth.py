import hashlib
import hmac
import base64
import json
import os
import secrets


AUTHENTICATED_EMPLOYEE_KEY = "authenticated_employee_id"
_ITERATIONS = 120000
_TOKEN_QUERY_KEY = "rp_session"
_configured_token_secret = os.environ.get(
    "RISKPULSE_PROTOTYPE_SESSION_SECRET"
)
_TOKEN_SECRET = (
    _configured_token_secret.encode("utf-8")
    if _configured_token_secret
    else secrets.token_bytes(32)
)

# Replace this directory with enterprise identity integration later.
_EMPLOYEE_DIRECTORY = {
    "analyst.alex": {
        "salt": "riskpulse-salt-alex",
        "password_hash": (
            "aa5c61ea2bc8b463fd6efc6142c67d7ae9de787bcbe40b8e1ef4136bd7293148"
        ),
    },
    "analyst.priya": {
        "salt": "riskpulse-salt-priya",
        "password_hash": (
            "3b615b72906c94e9864049a79707b79e8ede30b7f640fea524023cf541954be6"
        ),
    },
    "supervisor.morgan": {
        "salt": "riskpulse-salt-morgan",
        "password_hash": (
            "6e78cce56c333bab6eaa91260b8ce461858653e22621770ccf285578017d52d6"
        ),
    },
}


def _password_hash(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _ITERATIONS,
    ).hex()


def is_known_employee(employee_id):
    return isinstance(employee_id, str) and employee_id in _EMPLOYEE_DIRECTORY


def authenticate(employee_id, password):
    if not isinstance(employee_id, str) or not isinstance(password, str):
        return None

    employee = _EMPLOYEE_DIRECTORY.get(employee_id.strip())
    if employee is None:
        return None

    candidate_hash = _password_hash(password, employee["salt"])
    if not hmac.compare_digest(candidate_hash, employee["password_hash"]):
        return None

    return employee_id.strip()


def get_authenticated_employee(session_state):
    employee_id = session_state.get(AUTHENTICATED_EMPLOYEE_KEY)
    return employee_id if is_known_employee(employee_id) else None


def create_session_token(employee_id, navigation_state=None):
    if not is_known_employee(employee_id):
        raise ValueError("Cannot create a session for an unknown employee")

    payload_data = {"employee_id": employee_id}
    if navigation_state is not None:
        payload_data["navigation"] = navigation_state
    payload = base64.urlsafe_b64encode(
        json.dumps(payload_data, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    signature = hmac.new(
        _TOKEN_SECRET,
        payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}.{signature}"


def employee_from_session_token(token):
    payload = session_payload(token)
    return payload.get("employee_id") if payload else None


def session_payload(token):
    if not isinstance(token, str) or "." not in token:
        return None

    payload, signature = token.rsplit(".", 1)
    expected_signature = hmac.new(
        _TOKEN_SECRET,
        payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        return None

    try:
        padding = "=" * (-len(payload) % 4)
        data = json.loads(
            base64.urlsafe_b64decode(f"{payload}{padding}").decode("utf-8")
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    employee_id = data.get("employee_id")
    if not is_known_employee(employee_id):
        return None
    return data


def session_query_key():
    return _TOKEN_QUERY_KEY


def logout(session_state):
    for key in (
        AUTHENTICATED_EMPLOYEE_KEY,
        "navigation_target",
        "last_rendered_page",
        "investigation_transaction_id",
        "investigation_return_page",
        "challenge_transaction_id",
        "challenge_return_page",
        "feedback_transaction_id",
        "explorer_filter_signature",
        "explorer_page",
        "review_queue_filter_signature",
        "review_queue_page",
        "review_case_id",
        "selected_review_case",
        "audited_investigations",
        "audited_challenges",
        "audit_logs_viewed",
    ):
        session_state.pop(key, None)
