from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_PATH = MODEL_DIR / "random_forest.pkl"
LOCATION_MAP_PATH = MODEL_DIR / "customer_location_freq_map.json"
METADATA_PATH = MODEL_DIR / "preprocessing_metadata.json"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RiskPulse",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOAD MODEL + METADATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
location_map = load_json(LOCATION_MAP_PATH)
metadata = load_json(METADATA_PATH)
feature_importance = load_json(IMPORTANCE_PATH)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_NAMES = [
    "Transaction Amount",
    "Quantity",
    "Customer Age",
    "Account Age Days",
    "Transaction Hour",
    "transaction_day_of_week",
    "transaction_day_of_month",
    "transaction_month",
    "customer_location_freq",
    "Payment Method_PayPal",
    "Payment Method_bank transfer",
    "Payment Method_credit card",
    "Payment Method_debit card",
    "Product Category_clothing",
    "Product Category_electronics",
    "Product Category_health & beauty",
    "Product Category_home & garden",
    "Product Category_toys & games",
    "Device Used_desktop",
    "Device Used_mobile",
    "Device Used_tablet",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def location_frequency(location):
    """
    Convert a customer location into the same frequency feature
    used during model training.
    """

    return float(location_map.get(location, 0.0))


def build_features(
    amount,
    quantity,
    customer_age,
    account_age,
    transaction_hour,
    day_of_week,
    day_of_month,
    month,
    location,
    payment_method,
    product_category,
    device,
):
    """
    Convert user input into the exact 21-feature format
    expected by the trained Random Forest.
    """

    features = {
        "Transaction Amount": amount,
        "Quantity": quantity,
        "Customer Age": customer_age,
        "Account Age Days": account_age,
        "Transaction Hour": transaction_hour,
        "transaction_day_of_week": day_of_week,
        "transaction_day_of_month": day_of_month,
        "transaction_month": month,
        "customer_location_freq": location_frequency(location),

        "Payment Method_PayPal": 1 if payment_method == "PayPal" else 0,
        "Payment Method_bank transfer": 1 if payment_method == "bank transfer" else 0,
        "Payment Method_credit card": 1 if payment_method == "credit card" else 0,
        "Payment Method_debit card": 1 if payment_method == "debit card" else 0,

        "Product Category_clothing": 1 if product_category == "clothing" else 0,
        "Product Category_electronics": 1 if product_category == "electronics" else 0,
        "Product Category_health & beauty": 1 if product_category == "health & beauty" else 0,
        "Product Category_home & garden": 1 if product_category == "home & garden" else 0,
        "Product Category_toys & games": 1 if product_category == "toys & games" else 0,

        "Device Used_desktop": 1 if device == "desktop" else 0,
        "Device Used_mobile": 1 if device == "mobile" else 0,
        "Device Used_tablet": 1 if device == "tablet" else 0,
    }

    return pd.DataFrame(
        [[features[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )


def calculate_risk(probability):
    """
    Convert model probability into an easy-to-understand
    0-100 risk score and a risk category.
    """

    score = int(round(probability * 100))

    if score >= 70:
        risk = "HIGH"
        action = "Manual Review"
    elif score >= 33:
        risk = "MEDIUM"
        action = "Additional Verification"
    else:
        risk = "LOW"
        action = "Allow Transaction"

    return score, risk, action


def generate_reasons(
    amount,
    account_age,
    transaction_hour,
    probability,
):
    """
    Generate simple human-readable risk signals.

    These are rule-based explanations of observable
    transaction characteristics. They are NOT claims
    that a specific feature caused the model prediction.
    """

    reasons = []

    if account_age <= 30:
        reasons.append("Very young customer account")

    if amount >= 1000:
        reasons.append("High transaction amount")

    if 0 <= transaction_hour <= 5:
        reasons.append("Transaction occurred during a high-risk night period")

    if probability >= 0.70:
        reasons.append("Model assigns a high fraud probability")

    if not reasons:
        reasons.append("No major high-risk signal detected")

    return reasons


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0b1020;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 1rem 0 2rem 0;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -2px;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        opacity: 0.75;
        margin-bottom: 1rem;
    }

    .risk-card {
        padding: 2rem;
        border-radius: 18px;
        text-align: center;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        margin-top: 1rem;
    }

    .risk-score {
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
    }

    .risk-label {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .reason {
        padding: 0.7rem;
        margin: 0.4rem 0;
        border-radius: 10px;
        background: rgba(255,255,255,0.05);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛡️ RiskPulse</div>
        <div class="hero-subtitle">
            AI-powered transaction risk intelligence for fraud prevention
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Transaction Analysis</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=500.0,
        step=10.0,
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
    )

    customer_age = st.number_input(
        "Customer Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1,
    )

with col2:
    account_age = st.number_input(
        "Account Age (days)",
        min_value=1,
        max_value=3650,
        value=180,
        step=1,
    )

    transaction_hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=12,
    )

    location = st.text_input(
        "Customer Location",
        value="Mumbai",
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "credit card",
            "debit card",
            "PayPal",
            "bank transfer",
        ],
    )

    product_category = st.selectbox(
        "Product Category",
        [
            "electronics",
            "clothing",
            "health & beauty",
            "home & garden",
            "toys & games",
        ],
    )

    device = st.selectbox(
        "Device",
        [
            "mobile",
            "desktop",
            "tablet",
        ],
    )


st.markdown("### Transaction Date")

date_col1, date_col2, date_col3 = st.columns(3)

with date_col1:
    day_of_week = st.selectbox(
        "Day of Week",
        list(range(7)),
        index=3,
        help="0 = Monday, 6 = Sunday",
    )

with date_col2:
    day_of_month = st.selectbox(
        "Day of Month",
        list(range(1, 32)),
        index=14,
    )

with date_col3:
    month = st.selectbox(
        "Month",
        list(range(1, 13)),
        index=1,
    )


st.markdown("")

analyze = st.button(
    "🔍  ANALYZE TRANSACTION",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    X = build_features(
        amount=amount,
        quantity=quantity,
        customer_age=customer_age,
        account_age=account_age,
        transaction_hour=transaction_hour,
        day_of_week=day_of_week,
        day_of_month=day_of_month,
        month=month,
        location=location,
        payment_method=payment_method,
        product_category=product_category,
        device=device,
    )

    probability = float(model.predict_proba(X)[0][1])

    score, risk, action = calculate_risk(probability)

    reasons = generate_reasons(
        amount=amount,
        account_age=account_age,
        transaction_hour=transaction_hour,
        probability=probability,
    )

    st.divider()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:

        st.markdown(
            f"""
            <div class="risk-card">
                <div>RISK SCORE</div>
                <div class="risk-score">{score}</div>
                <div class="risk-label">{risk} RISK</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col2:

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.1f}%",
        )

        st.progress(
            min(probability, 1.0),
            text=f"Model confidence: {probability * 100:.1f}%",
        )

        st.info(
            f"Recommended action: **{action}**"
        )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Risk Signals</div>',
        unsafe_allow_html=True,
    )

    for reason in reasons:
        st.markdown(
            f'<div class="reason">⚠️ {reason}</div>',
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # TRANSACTION SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Transaction Summary</div>',
        unsafe_allow_html=True,
    )

    summary = pd.DataFrame(
        {
            "Field": [
                "Amount",
                "Quantity",
                "Customer Age",
                "Account Age",
                "Transaction Hour",
                "Payment Method",
                "Product Category",
                "Device",
                "Location",
            ],
            "Value": [
                f"{amount:.2f}",
                quantity,
                customer_age,
                f"{account_age} days",
                f"{transaction_hour}:00",
                payment_method,
                product_category,
                device,
                location,
            ],
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RiskPulse is a defensive fraud-risk prototype. "
    "The underlying dataset is synthetic; model performance "
    "should not be interpreted as real-world performance."
)