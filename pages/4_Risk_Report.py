import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from utils.helpers import load_test_data, apply_css

st.set_page_config(page_title="Risk Report | PaySense AI", page_icon=":chart_with_upwards_trend:", layout="wide")
apply_css()

st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800;
           background:linear-gradient(135deg,#34d399,#10b981);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    📈 Risk Report
</h1>
<p style='color:#64748b;'>Payment failure rates by bank, method, and time patterns.</p>
""", unsafe_allow_html=True)

DARK   = "#030712"
CARD   = "#0a1628"
BORDER = "#1e3a5f"
GREEN  = "#10b981"
ACCENT = "#34d399"
TEXT   = "#f1f5f9"
SUBTLE = "#475569"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": SUBTLE, "ytick.color": SUBTLE,
    "text.color": TEXT, "grid.color": BORDER,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    df = load_test_data()
    with open(os.path.join(ROOT, "model", "label_maps.json")) as f:
        label_maps = json.load(f)
except Exception:
    st.error("Data not found. Please restart the app.")
    st.stop()

# Decode labels back
from sklearn.preprocessing import LabelEncoder
import pickle
encoders = pickle.load(open(os.path.join(ROOT, "model", "encoders.pkl"), "rb"))

df["bank_name"]    = encoders["bank"].inverse_transform(df["bank"].astype(int))
df["method_name"]  = encoders["payment_method"].inverse_transform(df["payment_method"].astype(int))
df["category_name"]= encoders["category"].inverse_transform(df["category"].astype(int))

col_left, col_right = st.columns(2)

# ── Failure Rate by Bank ──────────────────────────────────────────────────────
with col_left:
    bank_fail = df.groupby("bank_name")["failed"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#ef4444" if v > 0.35 else GREEN for v in bank_fail.values]
    ax.barh(bank_fail.index, bank_fail.values * 100, color=colors, alpha=0.85)
    ax.set_xlabel("Failure Rate (%)", fontsize=9)
    ax.set_title("Failure Rate by Bank", fontsize=12,
                 fontweight="bold", color=ACCENT)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Failure Rate by Payment Method ───────────────────────────────────────────
with col_right:
    method_fail = df.groupby("method_name")["failed"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#ef4444" if v > 0.35 else GREEN for v in method_fail.values]
    ax.barh(method_fail.index, method_fail.values * 100, color=colors, alpha=0.85)
    ax.set_xlabel("Failure Rate (%)", fontsize=9)
    ax.set_title("Failure Rate by Payment Method", fontsize=12,
                 fontweight="bold", color=ACCENT)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Failure Rate by Category ──────────────────────────────────────────────────
st.markdown("### 🛒 Failure Rate by Transaction Category")
cat_fail = df.groupby("category_name")["failed"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 3.5))
colors = ["#ef4444" if v > 0.35 else GREEN for v in cat_fail.values]
ax.bar(cat_fail.index, cat_fail.values * 100, color=colors, alpha=0.85)
ax.set_ylabel("Failure Rate (%)", fontsize=9)
ax.set_title("Failure Rate by Category", fontsize=11,
             fontweight="bold", color=ACCENT)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Summary Table ─────────────────────────────────────────────────────────────
st.markdown("### 📋 Bank Risk Summary")
bank_summary = df.groupby("bank_name").agg(
    total=("failed", "count"),
    failures=("failed", "sum"),
    failure_rate=("failed", "mean"),
    avg_amount=("amount", "mean")
).reset_index()
bank_summary["failure_rate"] = (bank_summary["failure_rate"] * 100).round(1)
bank_summary["avg_amount"]   = bank_summary["avg_amount"].round(0)
bank_summary.columns = ["Bank", "Total Txns", "Failures", "Failure Rate (%)", "Avg Amount (₦)"]
bank_summary = bank_summary.sort_values("Failure Rate (%)", ascending=False)

st.dataframe(
    bank_summary.style.background_gradient(subset=["Failure Rate (%)"], cmap="Reds"),
    use_container_width=True
)