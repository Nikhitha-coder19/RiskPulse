import streamlit as st


ACTION_LABELS = {
	"ALLOW": "Allowed",
	"BLOCK": "Blocked",
	"CHALLENGE": "Challenged",
	"REVIEW": "Review",
}

CHALLENGE_STATUS_LABELS = {
	"CHALLENGE_CREATED": "Awaiting customer verification",
	"VERIFICATION_STARTED": "Verification in progress",
	"VERIFICATION_COMPLETED": "Verification completed",
	"VERIFICATION_FAILED": "Verification failed",
	"CHALLENGE_EXPIRED": "Expired",
	"CHALLENGE_CANCELLED": "Cancelled",
}

def render_login():
	"""Render the honest prototype authentication entry point."""

	st.title("RiskPulse Operations Console")
	st.caption("Prototype authentication for the internal employee console.")
	st.info("This buildathon authentication layer is replaceable with enterprise identity later.")
	employee_id = st.text_input("Employee ID", key="login_employee_id")
	password = st.text_input(
		"Password",
		type="password",
		key="login_password",
	)
	return st.button("Log in", key="login_submit"), employee_id, password


def render_sidebar(authenticated_employee_id):
	"""Render the prototype console sidebar and return the selected page."""

	with st.sidebar:
		st.markdown("# RiskPulse")
		st.caption("Operations Console")
		st.caption(f"Logged in: {authenticated_employee_id}")
		logout_clicked = st.button("Log out", key="logout")
		selected_page = st.radio(
			"Navigate",
			(
				"Dashboard",
				"Transaction Explorer",
				"Review Queue",
				"Challenge Monitoring",
				"Audit Logs",
			),
			label_visibility="collapsed",
			key="active_page",
		)
	return selected_page, logout_clicked


def render_dashboard_metrics(stats):
	"""Render persisted decision counts as dashboard metrics."""

	metrics = (
		("Total Decisions", stats.get("total_decisions", 0)),
		("Allowed", stats.get("allow", 0)),
		("Challenged", stats.get("challenge", 0)),
		("Blocked", stats.get("block", 0)),
		("Review", stats.get("review", 0)),
		("Open Review Cases", stats.get("open_review_cases", 0)),
		("Resolved Reviews", stats.get("resolved_review_cases", 0)),
	)

	columns = st.columns(len(metrics))
	for column, (label, value) in zip(columns, metrics):
		column.metric(label, value)


def render_explorer_filters():
	"""Render explorer inputs and return service-compatible filter values."""

	st.subheader("Find transactions")
	first_row = st.columns(3)
	second_row = st.columns(3)

	transaction_id = first_row[0].text_input("Transaction ID")
	customer_id = first_row[1].text_input("Customer ID")
	merchant_id = first_row[2].text_input("Merchant ID")
	device_id = second_row[0].text_input("Device ID")
	ip_id = second_row[1].text_input("IP ID")
	action = second_row[2].selectbox(
		"RiskPulse action",
		("All actions", "ALLOW", "REVIEW", "CHALLENGE", "BLOCK"),
		format_func=format_action,
	)

	return {
		"transaction_id": transaction_id or None,
		"customer_id": customer_id or None,
		"merchant_id": merchant_id or None,
		"device_id": device_id or None,
		"ip_id": ip_id or None,
		"action": None if action == "All actions" else action,
	}


def render_explorer_table(rows):
	"""Render transaction rows without exposing internal model details."""

	if not rows:
		st.info("No transactions match the current filters.")
		return False, None

	display_rows = []
	for row in rows:
		display_rows.append(
			{
				"Transaction ID": row["transaction_id"],
				"Timestamp": row["timestamp"],
				"Customer ID": row["customer_id"],
				"Merchant ID": row["merchant_id"],
				"Device ID": row["device_id"],
				"IP ID": row["ip_id"],
				"Amount": row["amount"],
				"Action": row["action"] or "No decision",
			}
		)

	st.dataframe(
		display_rows,
		width="stretch",
		hide_index=True,
	)

	transaction_ids = [row["transaction_id"] for row in rows]
	selected_transaction_id = st.selectbox(
		"Transaction to investigate",
		transaction_ids,
		format_func=lambda transaction_id: (
			f"{transaction_id} | "
			f"{format_action(next(
				row['action'] for row in rows
				if row['transaction_id'] == transaction_id
			))}"
		),
	)

	return st.button(
		"Open investigation",
		key="open_investigation",
	), selected_transaction_id


def render_investigation(investigation, customer_protection):
	"""Render the read-only investigation view from service payloads."""

	transaction = investigation["transaction"]
	decision = investigation["decision"]
	review_case = investigation.get("review_case")
	challenge = investigation.get("challenge")

	if decision is None:
		st.warning("This transaction has no persisted RiskPulse decision.")
	else:
		st.subheader("Risk decision")
		decision_columns = st.columns(2)
		decision_columns[0].metric(
			"RiskPulse action",
			format_action(decision["action"]),
		)
		decision_columns[1].metric(
			"Overall risk score",
			f"{float(decision['final_risk_score']):.3f}",
		)

		probabilities = {
			"Traditional": float(decision["traditional_probability"]),
			"Behavioral": float(decision["behavioral_probability"]),
			"Merchant": float(decision["merchant_probability"]),
		}
		strongest_model = max(probabilities, key=probabilities.get)
		st.subheader("Risk signals")
		st.caption("Model probabilities used by RiskPulse for this decision.")
		st.dataframe(
			[
				{"Model": model, "Fraud probability": probability}
				for model, probability in probabilities.items()
			],
			width="stretch",
			hide_index=True,
		)
		if review_case is not None:
			st.subheader("Analyst review outcome")
			st.write(f"Review status: {review_case['status']}")
			if review_case["analyst_decision"]:
				st.write(
					f"Analyst decision: {review_case['analyst_decision']}"
				)
				if review_case["analyst_comments"]:
					st.write(
						f"Analyst comments: {review_case['analyst_comments']}"
					)
			elif review_case["status"] != "RESOLVED":
				st.caption("No analyst outcome has been recorded yet.")

	st.subheader("Transaction details")
	detail_rows = [
		("Transaction ID", transaction["transaction_id"]),
		("Timestamp", transaction["timestamp"]),
		("Customer ID", transaction["customer_id"]),
		("Merchant ID", transaction["merchant_id"]),
		("Device ID", transaction["device_id"]),
		("IP ID", transaction["ip_id"]),
		("Shipping address ID", transaction["shipping_address_id"]),
		("Billing address ID", transaction["billing_address_id"]),
		("Amount", transaction["amount"]),
		("Quantity", "Not available"),
		("Payment method", "Not available"),
		("Product category", "Not available"),
	]
	st.dataframe(
		[
			{"Field": field, "Value": str(value)}
			for field, value in detail_rows
		],
		width="stretch",
		hide_index=True,
	)

	if decision is not None and decision["action"] == "CHALLENGE":
		st.subheader("Challenge")
		if challenge is None:
			st.info("Additional customer verification is pending.")
		else:
			challenge_status = challenge["status"]
			st.info(
				f"Challenge: {CHALLENGE_STATUS_LABELS.get(challenge_status, challenge_status)}"
			)
			st.caption(
				"Monitor the challenge for the verification outcome."
			)
		if st.button(
			"View challenge",
			key=f"view_challenge_{transaction['transaction_id']}",
		):
			return {"action": "view_challenge"}

	st.subheader("Protection status")

	if decision is None:
		st.info("No protection status is available without a persisted decision.")
	elif decision["action"] == "ALLOW":
		st.success("Transaction is allowed.")
	elif decision["action"] == "REVIEW":
		if (
			review_case is not None
			and review_case["status"] == "RESOLVED"
			and review_case["analyst_decision"]
		):
			st.info(
				"Analyst review is completed; analyst decision: "
				f"{review_case['analyst_decision']}."
			)
		else:
			st.info("Transaction is under review.")
	elif decision["action"] == "CHALLENGE":
		if challenge is None:
			st.warning("Additional customer verification is pending.")
		else:
			st.warning(
				"Challenge status: "
				f"{CHALLENGE_STATUS_LABELS.get(challenge['status'], challenge['status'])}."
			)
	else:
		st.error("Transaction is blocked.")

	if customer_protection.get("warning_present"):
		st.warning("Merchant risk warning signal is present.")

	st.subheader("Operational next action")
	if decision is None:
		st.info("No operational action is available without a persisted decision.")
	elif decision["action"] == "ALLOW":
		st.success("No intervention is currently required.")
	elif decision["action"] == "REVIEW":
		if (
			review_case is not None
			and review_case["status"] == "RESOLVED"
			and review_case["analyst_decision"] == "ALLOW"
		):
			st.success("No further review intervention is currently required.")
		elif (
			review_case is not None
			and review_case["status"] == "RESOLVED"
			and review_case["analyst_decision"] == "BLOCK"
		):
			st.error("Transaction should remain blocked based on analyst review.")
		elif (
			review_case is not None
			and review_case["status"] == "RESOLVED"
			and review_case["analyst_decision"] == "ESCALATE"
		):
			st.warning("Further analyst or supervisor action is required.")
		else:
			st.info("This transaction is ready to enter the existing review queue.")
	elif decision["action"] == "CHALLENGE":
		challenge_message = {
			"CHALLENGE_CREATED": "Additional customer verification is pending. Monitor the challenge for the verification outcome.",
			"VERIFICATION_STARTED": "Customer verification is in progress. Monitor the challenge for the outcome.",
			"VERIFICATION_COMPLETED": "Customer verification is complete. The RiskPulse action remains CHALLENGE.",
			"VERIFICATION_FAILED": "Customer verification failed. Review the challenge outcome and associated transaction.",
			"CHALLENGE_EXPIRED": "The customer challenge expired without completion.",
			"CHALLENGE_CANCELLED": "The customer challenge was cancelled.",
		}
		st.info(
			challenge_message.get(
				challenge["status"] if challenge is not None else None,
				"Additional customer verification is pending. Monitor the challenge for the verification outcome.",
			)
		)
	else:
		st.error("This transaction is blocked. No further transaction intervention is available.")

	st.subheader("Analyst feedback")
	feedback_count = investigation.get("feedback_count", 0)
	if feedback_count:
		st.caption(
			f"{feedback_count} analyst feedback record(s) have been recorded "
			"for this transaction."
		)
		feedback_label = "View / Give Feedback"
	else:
		st.caption("No analyst feedback has been recorded for this transaction.")
		feedback_label = "Give feedback"
	if st.button(
		feedback_label,
		key=f"feedback_entry_{transaction['transaction_id']}",
	):
		return {"action": "view_feedback"}

	return None


def render_challenge_monitoring_table(challenges):
	"""Render challenge records and return the selected challenge."""

	if not challenges:
		st.info("No challenges have been created.")
		return None

	st.dataframe(
		[
			{
				"Challenge ID": challenge["challenge_id"],
				"Transaction ID": challenge["transaction_id"],
				"Current status": CHALLENGE_STATUS_LABELS.get(
					challenge["status"], challenge["status"]
				),
				"Created at": challenge["created_at"],
				"Verification outcome": challenge["outcome"] or "Pending",
				"Last lifecycle state": challenge["last_event"],
			}
			for challenge in challenges
		],
		width="stretch",
		hide_index=True,
	)

	selected_transaction_id = st.selectbox(
		"Challenge to inspect",
		[challenge["transaction_id"] for challenge in challenges],
		key="selected_challenge_transaction_id",
	)
	if st.button("Open challenge details", key="open_challenge_details"):
		return {"action": "view_challenge", "transaction_id": selected_transaction_id}
	return None


def render_challenge_details(challenge):
	"""Render read-only challenge monitoring details."""

	st.subheader("Challenge overview")
	st.dataframe(
		[
			{"Field": "Challenge ID", "Value": challenge["challenge_id"]},
			{"Field": "Transaction ID", "Value": challenge["transaction_id"]},
			{
				"Field": "Current status",
				"Value": CHALLENGE_STATUS_LABELS.get(
					challenge["status"], challenge["status"]
				),
			},
			{"Field": "Created at", "Value": challenge["created_at"]},
			{"Field": "Outcome", "Value": challenge["outcome"] or "Pending"},
		],
		width="stretch",
		hide_index=True,
	)

	st.subheader("Why was this transaction challenged?")
	st.write(challenge["reason"])
	st.subheader("Verification requirement")
	st.info(challenge["verification_requirement"])
	st.subheader("Challenge lifecycle")
	st.dataframe(
		[
			{
				"Event": event["event_type"],
				"Status": CHALLENGE_STATUS_LABELS.get(
					event["event_type"], event["event_type"]
				),
				"Timestamp": event["created_at"],
				"Outcome": event["outcome"] or "",
				"Notes": event["notes"] or "",
			}
			for event in challenge["events"]
		],
		width="stretch",
		hide_index=True,
	)


def render_feedback_page(investigation, feedback_rows):
		"""Render the transaction-scoped retrospective feedback page."""

		transaction = investigation["transaction"]
		decision = investigation["decision"]
		st.title("Analyst Feedback")
		st.caption(
			"Feedback is retrospective information and does not edit the original "
			"RiskPulse decision or review outcome."
		)
		st.dataframe(
			[
				{"Field": "Transaction ID", "Value": transaction["transaction_id"]},
				{
					"Field": "Original RiskPulse action",
					"Value": format_action(decision["action"]) if decision else "Not recorded",
				},
				{
					"Field": "Overall risk score",
					"Value": (
						f"{float(decision['final_risk_score']):.3f}"
						if decision else "Not recorded"
					),
				},
			],
			width="stretch",
			hide_index=True,
		)

		st.subheader("Analyst feedback")
		st.write(
			"Record retrospective feedback about what actually happened and what "
			"RiskPulse should have done. Feedback does not change the original risk "
			"decision or review outcome."
		)

		actual_outcome = st.selectbox(
			"Actual real-world outcome",
			("Not recorded", "Legitimate", "Fraud"),
			key=f"feedback_outcome_{transaction['transaction_id']}",
		)
		recommended_action = st.selectbox(
			"Recommended RiskPulse action",
			("Not recorded", "Allow", "Review", "Challenge", "Block"),
			key=f"feedback_action_{transaction['transaction_id']}",
		)
		comments = st.text_area(
			"Comments",
			key=f"feedback_comments_{transaction['transaction_id']}",
		)

		if st.button(
			"Record feedback",
			key=f"record_feedback_{transaction['transaction_id']}",
		):
			return {
				"action": "record_feedback",
				"confirmed_outcome": {
					"Legitimate": "LEGITIMATE",
					"Fraud": "FRAUD",
				}.get(actual_outcome),
				"recommended_action": {
					"Allow": "ALLOW",
					"Review": "REVIEW",
					"Challenge": "CHALLENGE",
					"Block": "BLOCK",
				}.get(recommended_action),
				"comments": comments or None,
			}

		st.subheader("Feedback history")
		if not feedback_rows:
			st.info("No analyst feedback has been recorded for this transaction.")
		else:
			for feedback in feedback_rows:
				with st.container(border=True):
					st.write(f"**{feedback['employee_id']}** | {feedback['created_at']}")
					st.write(
						"Actual outcome: "
						f"{format_feedback_outcome(feedback['confirmed_outcome'])}"
					)
					st.write(
						"Recommended action: "
						f"{format_feedback_action(feedback['recommended_action'])}"
					)
					st.write(f"Comments: {feedback['comments'] or 'No comments'}")

		return None




def format_feedback_outcome(outcome):
		return {
			"LEGITIMATE": "Legitimate",
			"FRAUD": "Fraud",
		}.get(outcome, "Not recorded")




def format_feedback_action(action):
		return {
			"ALLOW": "Allow",
			"REVIEW": "Review",
			"CHALLENGE": "Challenge",
			"BLOCK": "Block",
		}.get(action, "Not recorded")


def render_review_queue_filters(employee_ids):
	"""Render review queue filters and return service-compatible values."""

	columns = st.columns(2)
	status = columns[0].selectbox(
		"Status",
		("ACTIVE", "All statuses", "OPEN", "IN_PROGRESS", "RESOLVED"),
		format_func=lambda value: (
			"Active reviews" if value == "ACTIVE"
			else value
		),
		key="review_queue_status_filter",
	)
	assigned_employee = columns[1].selectbox(
		"Assigned employee",
		["All employees", *employee_ids],
		key="review_queue_employee_filter",
	)

	return {
		"status": None if status == "All statuses" else status,
		"assigned_employee_id": (
			None if assigned_employee == "All employees"
			else assigned_employee
		),
	}


def render_review_queue_table(cases):
	"""Render review cases and return the selected case ID."""

	if not cases:
		st.info("No review cases match the current filters.")
		return None

	st.dataframe(
		[
			{
				"Review ID": case["review_id"],
				"Transaction ID": case["transaction_id"],
				"Status": case["status"],
				"Assigned employee": case["assigned_employee_id"] or "Unassigned",
				"Review reason": case["review_reason"],
				"Created at": case["created_at"],
				"Updated at": case["updated_at"],
				"Analyst decision": case["analyst_decision"] or "Pending",
			}
			for case in cases
		],
		width="stretch",
		hide_index=True,
	)

	for case in cases:
		if st.button(
			case["transaction_id"],
			key=f"investigate_review_{case['review_id']}",
		):
			return {
				"action": "investigate",
				"transaction_id": case["transaction_id"],
			}

	case_by_id = {case["review_id"]: case for case in cases}
	selected_review_id = st.selectbox(
		"Review case",
		list(case_by_id),
		format_func=lambda review_id: (
			f"#{review_id} | "
			f"{case_by_id[review_id]['transaction_id']} | "
			f"{case_by_id[review_id]['status']}"
		),
		key="selected_review_case",
	)
	return {"action": "select", "case": case_by_id[selected_review_id]}


def render_review_case(case):
	"""Render the selected review case and return an explicit UI action."""

	st.subheader(f"Review case #{case['review_id']}")
	st.write(f"Transaction ID: {case['transaction_id']}")
	st.write(f"Status: {case['status']}")
	st.write(f"Review reason: {case['review_reason']}")
	st.write(
		"Assigned employee: "
		f"{case['assigned_employee_id'] or 'Unassigned'}"
	)

	if case["status"] == "OPEN":
		if st.button(
			"Assign to me",
			key=f"assign_case_{case['review_id']}",
		):
			return {
				"action": "assign",
				"review_id": case["review_id"],
			}

	elif case["status"] == "IN_PROGRESS":
		st.caption(f"Assigned to {case['assigned_employee_id']}.")

	else:
		st.write(
			f"Analyst decision: {case['analyst_decision'] or 'Not recorded'}"
		)
		if case["analyst_comments"]:
			st.write(f"Analyst comments: {case['analyst_comments']}")
		st.write(f"Resolved at: {case['resolved_at'] or 'Not recorded'}")

	return None


def render_in_progress_controls(case, can_resolve):
	"""Render resolution controls only for the assigned employee."""

	if not can_resolve:
		st.info(
			"This case is assigned to another employee and cannot be resolved "
			"by the current employee."
		)
		return None

	analyst_decision = st.selectbox(
		"Analyst decision",
		("ALLOW", "BLOCK", "ESCALATE"),
		key=f"resolve_decision_{case['review_id']}",
	)
	analyst_comments = st.text_area(
		"Analyst comments",
		key=f"resolve_comments_{case['review_id']}",
	)
	if st.button(
		"Resolve case",
		key=f"resolve_case_{case['review_id']}",
	):
		return {
			"action": "resolve",
			"review_id": case["review_id"],
			"analyst_decision": analyst_decision,
			"analyst_comments": analyst_comments,
		}

	return None


def render_audit_logs_table(rows):
	"""Render bounded audit events from the service payload."""

	if not rows:
		st.info("No audit events match the current filters.")
		return

	st.dataframe(
		[
			{
				"Timestamp": row["created_at"],
				"Employee ID": row["employee_id"],
				"Action": row["action"],
				"Transaction ID": row["transaction_id"] or "",
				"Entity type": row["entity_type"] or "",
				"Entity ID": row["entity_id"] or "",
				"Metadata": row["metadata"] or "",
			}
			for row in rows
		],
		width="stretch",
		hide_index=True,
	)


def render_pagination(page, total_pages, total_count, item_label="transactions"):
	"""Render bounded explorer pagination controls."""

	previous, summary, next_page = st.columns([1, 2, 1])

	with previous:
		previous_clicked = st.button(
			"Previous",
			disabled=page <= 1,
			width="stretch",
		)

	with summary:
		st.caption(
			f"Page {page} of {total_pages} | "
			f"{total_count} matching {item_label}"
		)

	with next_page:
		next_clicked = st.button(
			"Next",
			disabled=page >= total_pages,
			width="stretch",
		)

	if previous_clicked:
		return page - 1
	if next_clicked:
		return page + 1
	return page


def format_action(action):
	"""Return a readable action label for persisted RiskPulse actions."""

	if action is None:
		return "No decision"
	return ACTION_LABELS.get(action, action)
