"""
01_executive_summary.py — Top-line KPIs, revenue exposure, segment snapshot, priority list.
"""

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.utils import load_scored, load_display, guard, fmt_pct, fmt_dollar

st.set_page_config(page_title="Executive Summary", layout="wide")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 Executive Summary")
st.caption(
    "Single-screen snapshot of churn exposure, revenue at risk, "
    "and the customers who need attention right now."
)

scored  = load_scored()
display = load_display()
guard(scored, "Executive Summary")

# ── KPI strip ─────────────────────────────────────────────────────────────────
total       = len(scored)
high_risk   = int((scored["churn_predicted"] == 1).sum())
rev_at_risk = scored["revenue_at_risk"].sum()
actual_rate = scored["Churn"].mean() if "Churn" in scored.columns else None
save_imm_n  = int((scored["segment_label"] == "Save Immediately").sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Customers",         f"{total:,}")
k2.metric("Predicted High-Risk",     f"{high_risk:,}",
          f"{fmt_pct(high_risk/total)} of base")
k3.metric("Monthly Revenue at Risk", fmt_dollar(rev_at_risk))
k4.metric("'Save Immediately' List", f"{save_imm_n:,}")
if actual_rate is not None:
    k5.metric("Actual Churn Rate",   fmt_pct(actual_rate))

st.markdown("---")

# ── Segment summary + revenue bar ─────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader("Segment Breakdown")
    seg = (
        scored.groupby(["segment_label", "segment_priority"])
        .agg(
            Customers             =("customerID", "count"),
            Avg_Churn_Prob        =("churn_probability", "mean"),
            Monthly_Rev_at_Risk   =("revenue_at_risk", "sum"),
            Avg_Monthly_Charges   =("MonthlyCharges", "mean"),
        )
        .reset_index()
        .sort_values("segment_priority")
        .drop(columns="segment_priority")
        .rename(columns={"segment_label": "Segment"})
    )
    seg["Avg_Churn_Prob"]      = seg["Avg_Churn_Prob"].map(fmt_pct)
    seg["Monthly_Rev_at_Risk"] = seg["Monthly_Rev_at_Risk"].map(lambda x: fmt_dollar(x))
    seg["Avg_Monthly_Charges"] = seg["Avg_Monthly_Charges"].map(lambda x: fmt_dollar(x, 2))
    st.dataframe(seg, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Revenue Exposure by Segment")
    rev_df = (
        scored.groupby("segment_label")["revenue_at_risk"]
        .sum().reset_index()
        .sort_values("revenue_at_risk", ascending=False)
    )
    color_map = {
        "Save Immediately": "#d62728",
        "Assess & Offer":   "#ff7f0e",
        "Nurture & Upsell": "#2ca02c",
        "Monitor Only":     "#7f7f7f",
    }
    fig_rev = px.bar(
        rev_df, x="segment_label", y="revenue_at_risk",
        color="segment_label", color_discrete_map=color_map,
        text=rev_df["revenue_at_risk"].map(fmt_dollar),
        labels={"revenue_at_risk": "Monthly Rev at Risk ($)", "segment_label": ""},
    )
    fig_rev.update_traces(textposition="outside")
    fig_rev.update_layout(showlegend=False, height=320, margin=dict(t=10))
    st.plotly_chart(fig_rev, use_container_width=True)

st.markdown("---")

# ── Do-nothing vs act comparison ──────────────────────────────────────────────
st.subheader("Business Case: Act vs. Do Nothing")

from src.decision.budget_optimizer import allocate_budget
DEFAULT_BUDGET = 5_000
result = allocate_budget(scored, budget=DEFAULT_BUDGET, lifetime_months=12)

b1, b2, b3, b4 = st.columns(4)
b1.metric("Budget",                   fmt_dollar(DEFAULT_BUDGET))
b2.metric("Customers Targeted",       f"{result['n_targeted']:,}")
b3.metric("Est. Annual Revenue Saved", fmt_dollar(result["expected_revenue_saved"]))
b4.metric("ROI",                      f"{result['roi_multiple']:.1f}×")

fig_cmp = go.Figure(go.Bar(
    x=["Do Nothing (annual)", f"With {fmt_dollar(DEFAULT_BUDGET)} Budget (annual)"],
    y=[rev_at_risk * 12, rev_at_risk * 12 - result["expected_revenue_saved"]],
    marker_color=["#d62728", "#2ca02c"],
    text=[fmt_dollar(rev_at_risk * 12), fmt_dollar(rev_at_risk * 12 - result["expected_revenue_saved"])],
    textposition="outside",
))
fig_cmp.update_layout(
    yaxis_title="Projected Annual Revenue Lost ($)",
    yaxis=dict(range=[0, rev_at_risk * 12 * 1.15]),
    height=320, margin=dict(t=10),
)
st.plotly_chart(fig_cmp, use_container_width=True)

st.markdown("---")

# ── Priority contact list ──────────────────────────────────────────────────────
st.subheader("🚨 Top 15 Customers — Contact Now")
st.caption(
    "Ranked by monthly revenue at risk. "
    "All are 'Save Immediately': high churn probability + high monthly value."
)

display_cols = [
    "customerID", "churn_probability", "MonthlyCharges",
    "revenue_at_risk", "tenure", "recommended_action",
]
# Get display-friendly version (Contract etc. readable)
if not display.empty and "Contract" in display.columns:
    display_cols_ext = display_cols + ["Contract", "InternetService"]
    top15 = (
        display[display["segment_label"] == "Save Immediately"]
        .sort_values("revenue_at_risk", ascending=False)
        .head(15)[[c for c in display_cols_ext if c in display.columns]]
    )
else:
    top15 = (
        scored[scored["segment_label"] == "Save Immediately"]
        .sort_values("revenue_at_risk", ascending=False)
        .head(15)[[c for c in display_cols if c in scored.columns]]
    )

top15 = top15.copy()
top15["churn_probability"] = top15["churn_probability"].map(fmt_pct)
top15["revenue_at_risk"]   = top15["revenue_at_risk"].map(lambda x: fmt_dollar(x, 2))
top15["MonthlyCharges"]    = top15["MonthlyCharges"].map(lambda x: fmt_dollar(x, 2))
st.dataframe(top15, use_container_width=True, hide_index=True)

st.caption(
    f"Model: {scored['scoring_model'].iloc[0].replace('_',' ').title()}  |  "
    f"Dataset: IBM Telco Customer Churn  |  "
    f"Revenue at risk = MonthlyCharges × P(churn)"
)
