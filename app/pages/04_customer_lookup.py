"""
04_customer_lookup.py — Individual customer risk profile with gauge, attributes, and action card.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.utils import load_scored, load_display, guard, fmt_pct, fmt_dollar
from config.settings import SEGMENT_LABELS

st.set_page_config(page_title="Customer Lookup", layout="wide")
st.title("🔍 Customer Risk Lookup")
st.caption(
    "Search any customer by ID to see their churn probability, "
    "revenue value, segment, and recommended action."
)

scored  = load_scored()
display = load_display()
guard(scored, "Customer Lookup")

# ── Search ─────────────────────────────────────────────────────────────────────
search_col, filter_col = st.columns([2, 1])

with filter_col:
    seg_filter = st.selectbox(
        "Filter by segment (optional)",
        ["All segments"] + list(scored["segment_label"].unique()),
    )

filtered = scored if seg_filter == "All segments" else scored[scored["segment_label"] == seg_filter]

with search_col:
    customer_id = st.selectbox(
        "Select or search a Customer ID",
        options=[""] + filtered["customerID"].tolist(),
        index=0,
        format_func=lambda x: x if x else "— type or pick a customer ID —",
    )

if not customer_id:
    st.info("Select a customer above. Use the segment filter to narrow the list.")

    # Show top-risk summary when no customer selected
    st.markdown("---")
    st.markdown("#### Highest-Risk Customers (by revenue at risk)")
    preview = scored.nlargest(10, "revenue_at_risk")[
        ["customerID", "segment_label", "churn_probability",
         "MonthlyCharges", "revenue_at_risk"]
    ].copy()
    preview["churn_probability"] = preview["churn_probability"].map(fmt_pct)
    preview["revenue_at_risk"]   = preview["revenue_at_risk"].map(lambda x: fmt_dollar(x, 2))
    preview["MonthlyCharges"]    = preview["MonthlyCharges"].map(lambda x: fmt_dollar(x, 2))
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.stop()

# ── Load customer row ──────────────────────────────────────────────────────────
score_row   = scored[scored["customerID"] == customer_id].iloc[0]
display_row = (
    display[display["customerID"] == customer_id].iloc[0]
    if not display.empty and customer_id in display["customerID"].values
    else score_row
)

prob     = float(score_row["churn_probability"])
segment  = score_row.get("segment_label", "—")
seg_info = next((v for v in SEGMENT_LABELS.values() if v["label"] == segment), {})
color    = seg_info.get("color", "#888")

# ── Risk gauge ─────────────────────────────────────────────────────────────────
col_gauge, col_detail = st.columns([1, 2])

with col_gauge:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        title={"text": "Churn Risk Score", "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": "#d62728" if prob >= 0.5 else "#2ca02c", "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  40], "color": "#d4f1d4"},
                {"range": [40, 65], "color": "#fff3cd"},
                {"range": [65,100], "color": "#ffcdd2"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.75,
                "value": float(score_row.get("churn_probability", 0.5)) * 100,
            },
        },
    ))
    gauge.update_layout(height=270, margin=dict(t=50, b=0, l=10, r=10))
    st.plotly_chart(gauge, use_container_width=True)

with col_detail:
    st.markdown(f"### {customer_id}")
    st.markdown(
        f"<span style='background:{color};color:white;padding:5px 14px;"
        f"border-radius:4px;font-weight:bold;font-size:15px'>"
        f"Priority {seg_info.get('priority','—')} — {segment}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    r1, r2, r3 = st.columns(3)
    r1.metric("Monthly Charges",   fmt_dollar(score_row["MonthlyCharges"], 2))
    r2.metric("Revenue at Risk/mo",fmt_dollar(score_row["revenue_at_risk"], 2))
    r3.metric("Tenure",            f"{int(score_row['tenure'])} months")

    r4, r5, r6 = st.columns(3)
    r4.metric("Risk Rank",         f"#{int(score_row['risk_rank']):,} of {len(scored):,}")
    r5.metric("Churn Predicted",   "Yes ⚠️" if score_row["churn_predicted"] == 1 else "No ✓")
    r6.metric("High Value",        "Yes" if score_row.get("is_high_value", 0) == 1 else "No")

    st.markdown(
        f"<div style='margin-top:12px;background:#f8f8f8;"
        f"border-left:5px solid {color};padding:10px 14px;border-radius:4px'>"
        f"<b>Recommended Action</b><br>{seg_info.get('action','—')}"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Customer profile table ────────────────────────────────────────────────────
st.subheader("Customer Profile")

profile_groups = {
    "Account": ["tenure", "Contract", "PaymentMethod", "PaperlessBilling",
                "is_autopay", "is_longterm_contract"],
    "Services": ["InternetService", "PhoneService", "MultipleLines",
                 "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies", "service_count"],
    "Financials": ["MonthlyCharges", "TotalCharges", "avg_monthly_revenue",
                   "charges_delta", "clv_estimate"],
    "Risk scores": ["churn_probability", "revenue_at_risk", "risk_rank",
                    "segment_label", "risk_tier", "value_tier"],
}

tab_names = list(profile_groups.keys())
tabs = st.tabs(tab_names)

for tab, (group, fields) in zip(tabs, profile_groups.items()):
    with tab:
        rows = []
        for field in fields:
            val = display_row.get(field, score_row.get(field, "—"))
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = "—"
            elif field in ("churn_probability",):
                val = fmt_pct(float(val))
            elif field in ("MonthlyCharges", "TotalCharges", "avg_monthly_revenue",
                           "charges_delta", "clv_estimate", "revenue_at_risk"):
                val = fmt_dollar(float(val), 2)
            rows.append({"Attribute": field.replace("_", " ").title(), "Value": str(val)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Similar at-risk customers ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Similar Customers in Same Segment")
st.caption("Nearest neighbours by monthly charge range — useful for batch outreach planning.")

charge     = float(score_row["MonthlyCharges"])
similar    = scored[
    (scored["segment_label"] == segment) &
    (scored["customerID"]    != customer_id) &
    (scored["MonthlyCharges"].between(charge * 0.85, charge * 1.15))
].nlargest(8, "revenue_at_risk")[[
    "customerID", "churn_probability", "MonthlyCharges", "revenue_at_risk"
]].copy()

similar["churn_probability"] = similar["churn_probability"].map(fmt_pct)
similar["revenue_at_risk"]   = similar["revenue_at_risk"].map(lambda x: fmt_dollar(x, 2))
similar["MonthlyCharges"]    = similar["MonthlyCharges"].map(lambda x: fmt_dollar(x, 2))
st.dataframe(similar, use_container_width=True, hide_index=True)
