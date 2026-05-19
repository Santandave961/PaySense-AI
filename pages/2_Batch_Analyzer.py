import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from utils.helpers import load_model, apply_css, score_payment, risk_color

st.set_page_config(page_title="Batch Analyzer | PaySense AI", page_icon=":package:", layout="wide")
apply_css()

st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800;
           background:linear-gradient(135deg,#34d399,#10b981);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    📦 Batch Payment Analyzer
</h1>
<p style='color:#64748b;'>Upload a CSV of payments — get instant failure risk predictions.</p>
""", unsafe_allow_html=True)

try:
    model, encoders, features = load_model()
except Exception:
    st.error("Model not found. Please restart the app.")
    st.stop()

# ── Sample CSV ────────────────────────────────────────────────────────────────
st.markdown("### 📥 Download Sample CSV Template")
sample = pd.DataFrame([
    {"amount": 5000, "bank": "GTBank", "payment_method": "Card",
     "network": "MTN", "category": "Shopping", "hour": 14,
     "day_of_week": 2, "account_age_days": 365, "prev_failures_24h": 0,
     "prev_success_7d": 10, "balance_ratio": 2.0, "is_international": 0,
     "device_changed": 0, "retry_attempt": 0, "daily_txn_count": 3},
    {"amount": 250000, "bank": "Access Bank", "payment_method": "Transfer",
     "network": "Airtel", "category": "Transfer", "hour": 2,
     "day_of_week": 6, "account_age_days": 15, "prev_failures_24h": 3,
     "prev_success_7d": 1, "balance_ratio": 0.3, "is_international": 1,
     "device_changed": 1, "retry_attempt": 2, "daily_txn_count": 12},
])
st.download_button("⬇️ Download Template", sample.to_csv(index=False).encode(),
                   "sample_payments.csv", "text/csv")

st.markdown("---")
st.markdown("### 📤 Upload Payments CSV")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    required = ["amount", "bank", "payment_method", "network", "category",
                "hour", "day_of_week", "account_age_days", "prev_failures_24h",
                "prev_success_7d", "balance_ratio", "is_international",
                "device_changed", "retry_attempt", "daily_txn_count"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # Encode
    for col in ["bank", "payment_method", "network", "category"]:
        df[col] = encoders[col].transform(df[col])

    results = []
    for _, row in df.iterrows():
        res = score_payment(model, encoders, features, row.to_dict())
        results.append(res)

    results_df = pd.DataFrame(results)
    df_out = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    # Summary
    total   = len(df_out)
    high    = (results_df["risk"] == "HIGH").sum()
    medium  = (results_df["risk"] == "MEDIUM").sum()
    low     = (results_df["risk"] == "LOW").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Payments", total)
    c2.metric("🔴 High Risk",   int(high))
    c3.metric("🟡 Medium Risk", int(medium))
    c4.metric("🟢 Low Risk",    int(low))

    st.markdown("### 🚨 High Risk Payments")
    high_df = df_out[results_df["risk"] == "HIGH"].sort_values("fail_prob", ascending=False)

    if len(high_df) > 0:
        st.dataframe(
            high_df[["amount", "fail_prob", "risk", "is_failure"]].style.background_gradient(
                subset=["fail_prob"], cmap="Reds"
            ),
            use_container_width=True
        )
        st.download_button("⬇️ Download High Risk Payments",
                           high_df.to_csv(index=False).encode(),
                           "high_risk_payments.csv", "text/csv")
    else:
        st.success("✅ No high risk payments in this batch!")

    with st.expander("View All Results"):
        st.dataframe(df_out[["amount", "fail_prob", "risk", "is_failure"]],
                     use_container_width=True)