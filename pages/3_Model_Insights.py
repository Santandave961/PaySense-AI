import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import load_model, load_metrics, load_test_data, apply_css

st.set_page_config(page_title="Model Insights | PaySense AI", page_icon=":bar_chart:", layout="wide")
apply_css()

st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800;
           background:linear-gradient(135deg,#34d399,#10b981);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    📊 Model Insights
</h1>
<p style='color:#64748b;'>XGBoost performance metrics and payment failure patterns.</p>
""", unsafe_allow_html=True)

DARK   = "#030712"
CARD   = "#0a1628"
BORDER = "#1e3a5f"
GREEN  = "#10b981"
ACCENT = "#34d399"
TEXT   = "#f1f5f9"
SUBTLE = "#475569"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "xtick.color":      SUBTLE,
    "ytick.color":      SUBTLE,
    "text.color":       TEXT,
    "grid.color":       BORDER,
})

try:
    model, encoders, features = load_model()
    metrics = load_metrics()
    df      = load_test_data()
except Exception:
    st.error("Model data not found. Please restart the app.")
    st.stop()

# ── Metric Cards ──────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
for col, (name, val) in zip([c1,c2,c3,c4,c5], [
    ("Accuracy",     metrics["accuracy"]),
    ("ROC-AUC",      metrics["roc_auc"]),
    ("F1 Score",     metrics["f1_score"]),
    ("Precision",    metrics["precision"]),
    ("Recall",       metrics["recall"]),
]):
    col.markdown(f"""
    <div class="metric-card">
        <div style="font-size:1.7rem; font-weight:800; font-family:Syne,sans-serif;
                    color:{ACCENT};">{val:.3f}</div>
        <div style="color:{SUBTLE}; font-size:0.75rem; margin-top:0.3rem;">{name}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; margin-top:1rem; color:{SUBTLE}; font-size:0.8rem;">
    Overall Payment Failure Rate:
    <strong style="color:{ACCENT};">{metrics['failure_rate']:.1%}</strong> ·
    Trained on <strong style="color:{ACCENT};">{metrics['total_trained']:,}</strong> transactions
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature Importance ────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    importance = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": features, "Importance": importance})
    feat_df = feat_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color=GREEN, alpha=0.8)
    ax.set_xlabel("Feature Importance", fontsize=9)
    ax.set_title("XGBoost Feature Importance", fontsize=12,
                 fontweight="bold", color=ACCENT)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Failure Probability Distribution ─────────────────────────────────────────
with col_right:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(df[df["failed"]==0]["fail_prob"], bins=40, alpha=0.7,
            color="#22c55e", label="Successful")
    ax.hist(df[df["failed"]==1]["fail_prob"], bins=40, alpha=0.7,
            color="#ef4444", label="Failed")
    ax.axvline(0.5, color=ACCENT, lw=2, linestyle="--", label="Threshold")
    ax.set_xlabel("Failure Probability", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Predicted Probability Distribution", fontsize=12,
                 fontweight="bold", color=ACCENT)
    ax.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Failure by Hour ───────────────────────────────────────────────────────────
st.markdown("### ⏰ Payment Failures by Hour of Day")
hour_fail = df[df["failed"]==1].groupby("hour").size()
all_hours = pd.Series(0, index=range(24))
hour_fail = (all_hours + hour_fail).fillna(0)

fig, ax = plt.subplots(figsize=(12, 3.5))
bars = ax.bar(hour_fail.index, hour_fail.values, color=GREEN, alpha=0.8)
for bar, val in zip(bars, hour_fail.values):
    if val > hour_fail.mean() * 1.5:
        bar.set_color("#ef4444")
        bar.set_alpha(0.9)
ax.set_xlabel("Hour of Day", fontsize=9)
ax.set_ylabel("Failed Payments", fontsize=9)
ax.set_title("Failure Count by Hour — Red = peak failure hours",
             fontsize=11, fontweight="bold", color=ACCENT)
ax.set_xticks(range(24))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)