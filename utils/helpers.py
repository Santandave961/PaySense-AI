import os
import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")

BANKS      = ["GTBank","Access Bank","Zenith Bank","UBA","First Bank",
              "Fidelity","Sterling","Wema","Polaris","Union Bank"]
METHODS    = ["Card","Transfer","USSD","QR Code","Bank Debit"]
NETWORKS   = ["MTN","Airtel","Glo","9mobile"]
CATEGORIES = ["Airtime","Bills","Transfer","Shopping","Food","Transport","Savings"]

@st.cache_resource
def load_model():
    model    = pickle.load(open(os.path.join(MODEL_DIR,"paysense_model.pkl"),"rb"))
    encoders = pickle.load(open(os.path.join(MODEL_DIR,"encoders.pkl"),"rb"))
    features = pickle.load(open(os.path.join(MODEL_DIR,"features.pkl"),"rb"))
    return model, encoders, features

@st.cache_data
def load_metrics():
    with open(os.path.join(MODEL_DIR,"metrics.json")) as f:
        return json.load(f)

@st.cache_data
def load_test_data():
    return pd.read_csv(os.path.join(MODEL_DIR,"test_data.csv"))

def score_payment(model, encoders, features, input_dict):
    X = pd.DataFrame([input_dict])[features]
    prob = float(model.predict_proba(X)[0][1])
    is_failure = prob >= 0.5
    if prob >= 0.7:   risk = "HIGH"
    elif prob >= 0.4: risk = "MEDIUM"
    else:             risk = "LOW"
    return {"fail_prob": round(prob,4), "is_failure": is_failure, "risk": risk}

def risk_color(level):
    return {"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#22c55e"}[level]

def risk_icon(level):
    return {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}[level]

def risk_css(level):
    return {
        "HIGH":   "background:linear-gradient(135deg,#450a0a,#7f1d1d);border:1px solid #ef4444;",
        "MEDIUM": "background:linear-gradient(135deg,#451a03,#78350f);border:1px solid #f59e0b;",
        "LOW":    "background:linear-gradient(135deg,#052e16,#14532d);border:1px solid #22c55e;",
    }[level]

def apply_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
    html,body,[class*="css"]{font-family:'DM Mono',monospace;background-color:#030712;color:#f1f5f9;}
    h1,h2,h3{font-family:'Syne',sans-serif;}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#050e1a,#0a1628);border-right:1px solid #1e3a5f;}
    .stButton>button{background:linear-gradient(135deg,#059669,#10b981);color:white;border:none;border-radius:6px;font-family:'Syne',sans-serif;font-weight:600;}
    .metric-card{background:linear-gradient(135deg,#0a1628,#0f2040);border:1px solid #1e3a5f;border-radius:12px;padding:1.4rem;text-align:center;}
    </style>
    """, unsafe_allow_html=True)