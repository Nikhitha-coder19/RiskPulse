import math
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from ui import components
from ui import auth
from ui.init_operations_db import initialize_operations_schema
from ui import services


PAGE_SIZE = 25
REVIEW_QUEUE_FILTER_PAGE_SIZE = 100
AUDIT_LOG_PAGE_SIZE = 100
NAVIGATION_PAGES = {
	"Dashboard",
	"Transaction Explorer",
	"Review Queue",
	"Challenge Monitoring",
	"Audit Logs",
}


st.set_page_config(
	page_title="RiskPulse Operations Console",
	page_icon="RP",
	layout="wide",
	initial_sidebar_state="expanded",
)

initialize_operations_schema()


def _navigation_state():
	state = {
		"page": st.session_state.get("active_page", "Dashboard"),
	}
	for key in (
		"investigation_transaction_id",
		"challenge_transaction_id",
		"feedback_transaction_id",
		"review_case_id",
	):
		value = st.session_state.get(key)
		if isinstance(value, str) and value:
			state[key] = value
	review_case_id = st.session_state.get(
		"review_case_id",
		st.session_state.get("selected_review_case"),
	)
	if isinstance(review_case_id, int) and review_case_id > 0:
		state["review_case_id"] = review_case_id
	return state


def _persist_navigation(employee_id):
	st.query_params[auth.session_query_key()] = auth.create_session_token(
		employee_id,
		navigation_state=_navigation_state(),
	)


def _restore_navigation(token):
	payload = auth.session_payload(token)
	navigation = payload.get("navigation") if payload else None
	if not isinstance(navigation, dict):
		return

	page = navigation.get("page")
	if page not in NAVIGATION_PAGES:
		page = "Dashboard"
	st.session_state["active_page"] = page
	for key in (
		"investigation_transaction_id",
		"challenge_transaction_id",
		"feedback_transaction_id",
		"review_case_id",
		"selected_review_case",
	):
		st.session_state.pop(key, None)
	for key in (
		"investigation_transaction_id",
		"challenge_transaction_id",
		"feedback_transaction_id",
	):
		value = navigation.get(key)
		if isinstance(value, str) and value:
			st.session_state[key] = value
	review_case_id = navigation.get("review_case_id")
	if isinstance(review_case_id, int) and review_case_id > 0:
		st.session_state["review_case_id"] = review_case_id
		st.session_state["selected_review_case"] = review_case_id


def _clear_inactive_context(page):
	if page != "Transaction Explorer":
		st.session_state.pop("investigation_transaction_id", None)
		st.session_state.pop("feedback_transaction_id", None)
	if page != "Challenge Monitoring":
		st.session_state.pop("challenge_transaction_id", None)
	if page != "Review Queue":
		st.session_state.pop("review_case_id", None)
		st.session_state.pop("selected_review_case", None)


def dashboard_page():
	st.title("Operations Dashboard")
	st.caption("Risk decision activity")

	try:
		stats = services.get_dashboard_stats()
	except Exception:
		st.error("Dashboard data is currently unavailable.")
		return

	components.render_dashboard_metrics(stats)


def explorer_page(authenticated_employee_id):
	if st.session_state.get("feedback_transaction_id"):
		feedback_page(
			st.session_state["feedback_transaction_id"],
			authenticated_employee_id,
		)
		return

	if st.session_state.get("investigation_transaction_id"):
		investigation_page(
			st.session_state["investigation_transaction_id"],
			authenticated_employee_id,
		)
		return

	st.title("Transaction Explorer")
	st.caption("Search and investigate transactions")

	filters = components.render_explorer_filters()
	filter_signature = tuple(sorted(filters.items()))

	if st.session_state.get("explorer_filter_signature") != filter_signature:
		st.session_state["explorer_filter_signature"] = filter_signature
		st.session_state["explorer_page"] = 1

	page = st.session_state.get("explorer_page", 1)

	try:
		total_count = services.count_transaction_explorer(**filters)
		total_pages = max(1, math.ceil(total_count / PAGE_SIZE))
		page = min(page, total_pages)
		rows = services.get_transaction_explorer(
			page=page,
			page_size=PAGE_SIZE,
			**filters,
		)
	except Exception:
		st.error("Transaction data is currently unavailable.")
		return

	open_clicked, transaction_id = components.render_explorer_table(rows)
	if open_clicked:
		open_investigation(transaction_id, authenticated_employee_id, "Transaction Explorer")

	selected_page = components.render_pagination(
		page,
		total_pages,
		total_count,
	)

	if selected_page != page:
		st.session_state["explorer_page"] = selected_page
		st.rerun()


def investigation_page(transaction_id, authenticated_employee_id):
	st.title("Transaction Investigation")
	if st.button("Back to explorer"):
		st.session_state.get("audited_investigations", set()).discard(
			transaction_id
		)
		del st.session_state["investigation_transaction_id"]
		st.session_state["navigation_target"] = st.session_state.pop(
			"investigation_return_page",
			"Transaction Explorer",
		)
		st.rerun()

	try:
		investigation = services.get_transaction_investigation(transaction_id)
		customer_protection = services.get_customer_protection(transaction_id)
	except Exception:
		st.error("Investigation data is currently unavailable.")
		return

	if investigation is None:
		st.error("Transaction not found.")
		return

	action = components.render_investigation(investigation, customer_protection)
	if action is not None and action["action"] == "view_challenge":
		open_challenge_details(
			transaction_id,
			authenticated_employee_id,
			"Transaction Explorer",
		)
	elif action is not None and action["action"] == "view_feedback":
		open_feedback_page(
			transaction_id,
			authenticated_employee_id,
		)


def feedback_page(transaction_id, authenticated_employee_id):
	try:
		investigation = services.get_transaction_investigation(transaction_id)
		feedback_rows = services.get_feedback(
			transaction_id=transaction_id,
			page=1,
			page_size=100,
		)
	except Exception:
		st.error("Feedback data is currently unavailable.")
		return

	if investigation is None:
		st.error("Transaction not found.")
		return

	if st.button("Back to transaction"):
		st.session_state.pop("feedback_transaction_id", None)
		st.session_state["investigation_transaction_id"] = transaction_id
		st.session_state["navigation_target"] = "Transaction Explorer"
		st.rerun()

	action = components.render_feedback_page(investigation, feedback_rows)
	if action is not None and action["action"] == "record_feedback":
		try:
			services.record_feedback(
				transaction_id=transaction_id,
				employee_id=authenticated_employee_id,
				confirmed_outcome=action["confirmed_outcome"],
				recommended_action=action["recommended_action"],
				comments=action["comments"],
			)
		except ValueError as error:
			st.error(str(error))
		else:
			st.rerun()


def open_feedback_page(transaction_id, authenticated_employee_id):
	st.session_state["feedback_transaction_id"] = transaction_id
	st.session_state["navigation_target"] = "Transaction Explorer"
	st.rerun()


def challenge_monitoring_page(authenticated_employee_id):
	if st.session_state.get("challenge_transaction_id"):
		challenge_details_page(
			st.session_state["challenge_transaction_id"],
			authenticated_employee_id,
		)
		return

	st.title("Challenge Monitoring")
	st.caption("Monitor customer verification challenges created by RiskPulse.")

	try:
		challenges = services.get_challenge_monitoring()
	except Exception:
		st.error("Challenge data is currently unavailable.")
		return

	st.caption(f"{len(challenges)} challenge(s)")
	action = components.render_challenge_monitoring_table(challenges)
	if action is not None and action["action"] == "view_challenge":
		open_challenge_details(
			action["transaction_id"],
			authenticated_employee_id,
			"Challenge Monitoring",
		)


def challenge_details_page(transaction_id, authenticated_employee_id):
	if st.button("Back to challenge monitoring"):
		st.session_state.pop("challenge_transaction_id", None)
		st.session_state["navigation_target"] = "Challenge Monitoring"
		st.rerun()

	try:
		challenge = services.get_challenge_details(transaction_id)
	except Exception:
		st.error("Challenge data is currently unavailable.")
		return

	if challenge is None:
		st.error("Challenge not found.")
		return

	audited_challenges = st.session_state.setdefault(
		"audited_challenges",
		set(),
	)
	if transaction_id not in audited_challenges:
		try:
			services.log_audit_event(
				employee_id=authenticated_employee_id,
				action="VIEW_CHALLENGE_DETAILS",
				transaction_id=transaction_id,
				entity_type="challenge",
				entity_id=str(challenge["challenge_id"]),
			)
			audited_challenges.add(transaction_id)
		except Exception:
			st.error("Challenge could not be opened.")
			return

	st.title("Challenge Details")
	action = components.render_challenge_details(challenge)
	if st.button("View transaction"):
		open_investigation(
			transaction_id,
			authenticated_employee_id,
			"Challenge Monitoring",
		)


def open_challenge_details(transaction_id, authenticated_employee_id, return_page):
	st.session_state["challenge_transaction_id"] = transaction_id
	st.session_state["challenge_return_page"] = return_page
	st.session_state["navigation_target"] = "Challenge Monitoring"
	st.rerun()


def review_queue_page(authenticated_employee_id):
	st.title("Review Queue")
	st.caption("Review transactions requiring analyst attention.")

	try:
		employee_source_cases = services.get_review_queue(
			page=1,
			page_size=REVIEW_QUEUE_FILTER_PAGE_SIZE,
		)
		employee_ids = sorted({
			case["assigned_employee_id"]
			for case in employee_source_cases
			if case["assigned_employee_id"]
		})
		filters = components.render_review_queue_filters(employee_ids)
	except ValueError as error:
		st.error(str(error))
		return

	filter_signature = tuple(sorted(filters.items()))
	if st.session_state.get("review_queue_filter_signature") != filter_signature:
		st.session_state["review_queue_filter_signature"] = filter_signature
		st.session_state["review_queue_page"] = 1

	page = st.session_state.get("review_queue_page", 1)

	try:
		total_count = services.count_review_queue(**filters)
		total_pages = max(1, math.ceil(total_count / PAGE_SIZE))
		page = min(page, total_pages)
		cases = services.get_review_queue(
			page=page,
			page_size=PAGE_SIZE,
			**filters,
		)
	except ValueError as error:
		st.error(str(error))
		return

	st.caption(f"{total_count} matching review case(s)")
	selected_case = components.render_review_queue_table(cases)
	if selected_case is not None and selected_case["action"] == "investigate":
		open_investigation(
			selected_case["transaction_id"],
			authenticated_employee_id,
			"Review Queue",
		)
	if selected_case is not None and selected_case["action"] == "select":
		selected_case = selected_case["case"]
		st.session_state["review_case_id"] = selected_case["review_id"]
		action = components.render_review_case(selected_case)
		if selected_case["status"] == "IN_PROGRESS":
			action = components.render_in_progress_controls(
			selected_case,
			selected_case["assigned_employee_id"]
			== authenticated_employee_id,
		)
		if action is not None:
			try:
				if action["action"] == "assign":
					services.assign_review_case(
						action["review_id"],
						authenticated_employee_id,
					)
				else:
					services.resolve_review_case(
						action["review_id"],
						authenticated_employee_id,
						action["analyst_decision"],
						action["analyst_comments"],
					)
			except ValueError as error:
				st.error(str(error))
			else:
				st.rerun()

	selected_page = components.render_pagination(
		page,
		total_pages,
		total_count,
		item_label="review cases",
	)
	if selected_page != page:
		st.session_state["review_queue_page"] = selected_page
		st.rerun()


def open_investigation(transaction_id, authenticated_employee_id, return_page):
	audited_transactions = st.session_state.setdefault(
		"audited_investigations",
		set(),
	)
	if transaction_id not in audited_transactions:
		try:
			services.log_audit_event(
				employee_id=authenticated_employee_id,
				action="VIEW_TRANSACTION_INVESTIGATION",
				transaction_id=transaction_id,
				entity_type="transaction",
				entity_id=transaction_id,
			)
			audited_transactions.add(transaction_id)
		except Exception:
			st.error("Investigation could not be opened.")
			return
	st.session_state["investigation_transaction_id"] = transaction_id
	st.session_state["investigation_return_page"] = return_page
	st.session_state["navigation_target"] = "Transaction Explorer"
	st.rerun()


def audit_logs_page():
	st.title("Audit Logs")
	st.caption("Review employee activity recorded by RiskPulse.")

	try:
		filter_source = services.get_audit_logs(limit=AUDIT_LOG_PAGE_SIZE)
		employee_ids = sorted({
			row["employee_id"] for row in filter_source
			if row["employee_id"]
		})
		actions = sorted({
			row["action"] for row in filter_source if row["action"]
		})
		entity_types = sorted({
			row["entity_type"] for row in filter_source
			if row["entity_type"]
		})
	except ValueError as error:
		st.error(str(error))
		return

	columns = st.columns(5)
	selected_employee = columns[0].selectbox(
		"Employee ID",
		["All employees", *employee_ids],
		key="audit_employee_filter",
	)
	selected_action = columns[1].selectbox(
		"Action",
		["All actions", *actions],
		key="audit_action_filter",
	)
	transaction_id = columns[2].text_input(
		"Transaction ID",
		key="audit_transaction_filter",
	)
	selected_entity_type = columns[3].selectbox(
		"Entity type",
		["All entity types", *entity_types],
		key="audit_entity_type_filter",
	)
	entity_id = columns[4].text_input(
		"Entity ID",
		key="audit_entity_filter",
	)

	try:
		rows = services.get_audit_logs(
			employee_id=(
				None if selected_employee == "All employees"
				else selected_employee
			),
			action=(
				None if selected_action == "All actions"
				else selected_action
			),
			transaction_id=transaction_id or None,
			entity_type=(
				None if selected_entity_type == "All entity types"
				else selected_entity_type
			),
			entity_id=entity_id or None,
			limit=AUDIT_LOG_PAGE_SIZE,
		)
	except ValueError as error:
		st.error(str(error))
		return

	st.caption(f"Showing up to {AUDIT_LOG_PAGE_SIZE} newest matching events.")
	components.render_audit_logs_table(rows)


if st.session_state.get("navigation_target"):
	st.session_state["active_page"] = st.session_state.pop(
		"navigation_target"
	)

authenticated_employee_id = auth.get_authenticated_employee(st.session_state)
if authenticated_employee_id is None:
	query_token = st.query_params.get(auth.session_query_key())
	authenticated_employee_id = auth.employee_from_session_token(query_token)
	if authenticated_employee_id is not None:
		st.session_state[auth.AUTHENTICATED_EMPLOYEE_KEY] = authenticated_employee_id
		_restore_navigation(query_token)

if authenticated_employee_id is None:
	st.session_state["active_page"] = "Dashboard"
	login_clicked, employee_id, password = components.render_login()
	if login_clicked:
		authenticated_employee_id = auth.authenticate(employee_id, password)
		if authenticated_employee_id is None:
			st.error("Invalid employee ID or password.")
		else:
			auth.logout(st.session_state)
			st.session_state[auth.AUTHENTICATED_EMPLOYEE_KEY] = authenticated_employee_id
			st.session_state["active_page"] = "Dashboard"
			_persist_navigation(authenticated_employee_id)
			st.rerun()
	st.stop()

selected_page, logout_clicked = components.render_sidebar(
	authenticated_employee_id,
)
if logout_clicked:
	auth.logout(st.session_state)
	st.query_params.clear()
	st.rerun()

_clear_inactive_context(selected_page)
_persist_navigation(authenticated_employee_id)

previous_page = st.session_state.get("last_rendered_page")
if selected_page == "Audit Logs" and previous_page != "Audit Logs":
	services.log_audit_event(
		employee_id=authenticated_employee_id,
		action="VIEW_AUDIT_LOGS",
		entity_type="audit_log",
	)
st.session_state["last_rendered_page"] = selected_page

if selected_page == "Dashboard":
	dashboard_page()
elif selected_page == "Review Queue":
	review_queue_page(authenticated_employee_id)
elif selected_page == "Challenge Monitoring":
	challenge_monitoring_page(authenticated_employee_id)
elif selected_page == "Audit Logs":
	audit_logs_page()
else:
	explorer_page(authenticated_employee_id)
