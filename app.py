import streamlit as st
import os
import subprocess
import sys

st.set_page_config(
    page_title="PaySense AI",
    page_icon=":credit_card:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Auto-train on first run ──────────────────────────────────────────────────
if not os.path.exists("model/paysense_model.pkl"):
    with st.spinner("Setting up PaySense AI for first time..."):
        subprocess.run([sys.executable, "model/train_model.py"], check=True)
    st.rerun()

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #030712;
    color: #f1f5f9;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050e1a 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}
.stButton > button {
    background: linear-gradient(135deg, #059669, #10b981);
    color: white; border: none; border-radius: 6px;
    font-family: 'Syne', sans-serif; font-weight: 600;
    letter-spacing: 0.05em; transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(16,185,129,0.4);
}
.metric-card {
    background: linear-gradient(135deg, #0a1628, #0f2040);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 1.4rem; text-align: center;
}
.fail-high {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444; border-radius: 12px; padding: 1.5rem;
    text-align: center;
}
.fail-medium {
    background: linear-gradient(135deg, #451a03, #78350f);
    border: 1px solid #f59e0b; border-radius: 12px; padding: 1.5rem;
    text-align: center;
}
.fail-low {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e; border-radius: 12px; padding: 1.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2.5rem 1rem 1.5rem;">
    <h1 style="font-family:'Syne',sans-serif; font-size:3rem; font-weight:800;
               background:linear-gradient(135deg,#34d399,#10b981,#059669);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        💳 PaySense AI
    </h1>
    <p style="color:#64748b; font-size:1rem; margin-top:0.3rem;">
        Smart Payment Failure Prediction · Nigerian Fintech Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
cards = [
    ("🎯", "XGBoost", "Failure Prediction"),
    ("⚡", "Real-Time", "Instant Scoring"),
    ("📦", "Batch Mode", "CSV Processing"),
    ("🇳🇬", "NGN-Native", "Nigerian Banks"),
]
for col, (icon, title, desc) in zip([c1,c2,c3,c4], cards):
    col.markdown(f"""
    <div class="metric-card">
        <div style="font-size:1.8rem;">{icon}</div>
        <div style="font-family:'Syne',sans-serif; font-weight:700;
                    color:#34d399; font-size:1rem; margin:0.3rem 0;">{title}</div>
        <div style="color:#475569; font-size:0.75rem;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── About ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0a1628,#0f2040);
            border:1px solid #1e3a5f; border-radius:12px; padding:2rem;">
    <h3 style="font-family:'Syne',sans-serif; color:#34d399;">About PaySense AI</h3>
    <p style="color:#94a3b8; line-height:1.8;">
        PaySense AI predicts payment failures <strong style="color:#f1f5f9;">before they happen</strong>,
        helping Nigerian fintech companies reduce declined transaction rates, improve customer
        experience, and protect revenue. Powered by <strong style="color:#f1f5f9;">XGBoost</strong>
        trained on Nigerian payment behavioral patterns across all major banks and payment methods.
    </p>
    <ul style="color:#94a3b8; line-height:2;">
        <li><strong style="color:#34d399;">Payment Scanner</strong> — score a single payment instantly</li>
        <li><strong style="color:#34d399;">Batch Analyzer</strong> — upload CSV, flag high-risk payments</li>
        <li><strong style="color:#34d399;">Model Insights</strong> — feature importance and failure patterns</li>
        <li><strong style="color:#34d399;">Risk Report</strong> — failure rates by bank, method, and time</li>
    </ul>
</div>
""", unsafe_allow_html=True)