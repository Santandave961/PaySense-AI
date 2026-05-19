import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils.helpers import (
    load_model, apply_css, score_payment,
    risk_css, risk_color, risk_icon,
    BANKS, METHODS, NETWORKS, CATEGORIES
)

st.set_page_config(page_title="Payment Scanner | PaySense AI", page_icon=":zap:", layout="wide")
apply_css()

st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800;
           background:linear-gradient(135deg,#34d399,#10b981);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    ⚡ Payment Scanner
</h1>
<p style='color:#64748b;'>Enter payment details to predict failure risk in real time.</p>
""", unsafe_allow_html=True)

try:
    model, encoders, features = load_model()
except Exception:
    st.error("Model not found. Please restart the app.")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

with st.form("scan_form"):
    st.markdown("### 💳 Payment Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        amount           = st.number_input("Amount (₦)", 100.0, 10000000.0, 5000.0, step=500.0)
        bank             = st.selectbox("Customer Bank", BANKS)
        payment_method   = st.selectbox("Payment Method", METHODS)
        network          = st.selectbox("Mobile Network", NETWORKS)

    with col2:
        category         = st.selectbox("Transaction Category", CATEGORIES)
        hour             = st.slider("Hour of Day", 0, 23, 14)
        day_of_week      = st.slider("Day of Week (0=Mon)", 0, 6, 2)
        account_age_days = st.number_input("Account Age (days)", 1, 1825, 365)

    with col3:
        prev_failures_24h = st.slider("Failed Attempts (24h)", 0, 5, 0)
        prev_success_7d   = st.slider("Successful Txns (7 days)", 0, 30, 5)
        balance_ratio     = st.slider("Balance/Amount Ratio", 0.0, 3.0, 1.5, step=0.1)
        is_international  = st.selectbox("International Payment?", ["No", "Yes"])
        device_changed    = st.selectbox("New Device?", ["No", "Yes"])
        retry_attempt     = st.slider("Retry Attempt #", 0, 4, 0)
        daily_txn_count   = st.slider("Transactions Today", 1, 20, 3)

    submitted = st.form_submit_button("💳 Predict Payment Outcome", use_container_width=True)

if submitted:
    input_dict = {
        "amount":             amount,
        "bank":               encoders["bank"].transform([bank])[0],
        "payment_method":     encoders["payment_method"].transform([payment_method])[0],
        "network":            encoders["network"].transform([network])[0],
        "category":           encoders["category"].transform([category])[0],
        "hour":               hour,
        "day_of_week":        day_of_week,
        "account_age_days":   int(account_age_days),
        "prev_failures_24h":  prev_failures_24h,
        "prev_success_7d":    prev_success_7d,
        "balance_ratio":      balance_ratio,
        "is_international":   1 if is_international == "Yes" else 0,
        "device_changed":     1 if device_changed == "Yes" else 0,
        "retry_attempt":      retry_attempt,
        "daily_txn_count":    daily_txn_count,
    }

    result = score_payment(model, encoders, features, input_dict)
    level  = result["risk"]
    color  = risk_color(level)
    icon   = risk_icon(level)
    css    = risk_css(level)

    outcomes = {
        "HIGH":   "🔴 High failure risk — consider flagging for review or requesting re-authentication.",
        "MEDIUM": "🟡 Moderate risk — monitor closely, may proceed with additional verification.",
        "LOW":    "🟢 Low failure risk — payment likely to succeed. Proceed normally."
    }

    st.markdown(f"""
    <div style="{css} border-radius:12px; padding:2rem; margin-top:1.5rem; text-align:center;">
        <div style="font-size:2.2rem; font-weight:800; font-family:Syne,sans-serif;
                    color:{color};">{icon} {level} FAILURE RISK</div>
        <div style="font-size:3.5rem; font-weight:800; color:{color}; margin:0.5rem 0;">
            {result['fail_prob']:.1%}
        </div>
        <div style="color:#94a3b8; font-size:0.85rem;">Failure Probability</div>
        <div style="margin-top:1rem; color:{color}; font-weight:600; font-size:0.9rem;">
            {outcomes[level]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(result["fail_prob"]))

    st.markdown("### 🔍 Risk Signals")
    signals = []
    if prev_failures_24h >= 2:    signals.append(f"⚠️ {prev_failures_24h} failed attempts in last 24h")
    if balance_ratio < 0.5:       signals.append("⚠️ Low balance relative to payment amount")
    if retry_attempt >= 2:        signals.append(f"⚠️ Retry attempt #{retry_attempt}")
    if is_international == "Yes": signals.append("⚠️ International payment")
    if device_changed == "Yes":   signals.append("⚠️ New/unrecognised device")
    if hour in range(0, 6):       signals.append(f"⚠️ Unusual hour: {hour}:00")
    if int(account_age_days) < 30: signals.append(f"⚠️ New account: {int(account_age_days)} days old")

    if signals:
        for s in signals:
            st.markdown(f"- {s}")
    else:
        st.success("✅ No major risk signals detected for this payment.")