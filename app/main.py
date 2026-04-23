"""
main.py — Streamlit entry point for the Churn Decision Engine.

Launch:  streamlit run app/main.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import APP_ICON, APP_TITLE
from app.utils import load_scored

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.markdown("---")

    scored = load_scored()
    if not scored.empty:
        total     = len(scored)
        high_risk = int((scored["churn_predicted"] == 1).sum())
        rev_risk  = scored["revenue_at_risk"].sum()
        st.markdown("**Live snapshot**")
        st.metric("Customers",        f"{total:,}")
        st.metric("High-risk",        f"{high_risk:,}  ({high_risk/total:.0%})")
        st.metric("Rev at risk/mo",   f"${rev_risk:,.0f}")
        model_name = scored["scoring_model"].iloc[0].replace("_", " ").title()
        st.caption(f"Model: {model_name}")
    else:
        st.warning("Run the pipeline first:\n`python -m src.models.run_pipeline`")

    st.markdown("---")
    st.markdown("**Navigate**")
    st.caption(
        "📊 Executive Summary  \n"
        "📉 Churn Overview  \n"
        "🗺️ Segment Explorer  \n"
        "🔍 Customer Lookup  \n"
        "💰 Retention Simulator  \n"
        "🤖 Model Performance  \n"
        "📋 Recommendations"
    )
    st.markdown("---")
    st.caption("Dataset: IBM Telco Customer Churn")

# ── Landing page ───────────────────────────────────────────────────────────────
st.title(f"{APP_ICON} Churn Decision Engine")
st.markdown(
    "#### Turning churn risk scores into retention actions — with a budget constraint."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**The problem this solves**")
    st.markdown(
        "Most churn tools stop at a prediction. "
        "This engine goes further: it scores every customer, "
        "segments them by revenue value, maps each group to an action, "
        "and tells you who to call first when your budget is limited."
    )
with col2:
    st.markdown("**Six questions answered**")
    for q in [
        "Which customers are most likely to churn?",
        "Which segments drive the most churn?",
        "Which customers are most valuable to save?",
        "What action fits each customer segment?",
        "Who should we target first on a fixed budget?",
        "What's the revenue impact of acting vs. doing nothing?",
    ]:
        st.markdown(f"- {q}")
with col3:
    st.markdown("**How it works**")
    st.markdown(
        "1. **Data pipeline** — clean, engineer 36 features  \n"
        "2. **Three models** — LR, RF, XGBoost with threshold tuning  \n"
        "3. **Best model** selected by holdout AUC  \n"
        "4. **Score** all 7,043 customers with P(churn)  \n"
        "5. **Segment** into 2×2 risk-value matrix  \n"
        "6. **Optimise** spend by ROI-ranked outreach list  \n"
        "7. **Simulate** any budget scenario interactively"
    )

st.markdown("---")

# ── Page guide ────────────────────────────────────────────────────────────────
st.markdown("### Where to start")

pages = [
    ("📊", "Executive Summary", "Top-line KPIs, revenue exposure, and the top 15 customers to call now"),
    ("📉", "Churn Overview",    "Churn rates by contract, tenure, charges, services — all interactive"),
    ("🗺️", "Segment Explorer",  "The 2×2 risk-value matrix with drill-down into each customer group"),
    ("🔍", "Customer Lookup",   "Search any customer ID and see their full risk profile and action card"),
    ("💰", "Retention Simulator","Set a budget and see who to target, what you'll save, and the ROI"),
    ("🤖", "Model Performance", "ROC curves, confusion matrices, SHAP feature importance"),
    ("📋", "Recommendations",   "Full retention action playbook by segment with success KPIs"),
]

for icon, title, desc in pages:
    with st.container():
        c1, c2 = st.columns([0.08, 0.92])
        c1.markdown(f"## {icon}")
        c2.markdown(f"**{title}**  \n{desc}")
