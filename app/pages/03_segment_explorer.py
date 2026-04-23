"""
03_segment_explorer.py — Interactive 2×2 risk-value matrix with segment drill-down.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.utils import load_scored, load_display, guard, fmt_pct, fmt_dollar
from config.settings import SEGMENT_LABELS
from src.decision.segmenter import matrix_figure, churn_rate_by_segment_figure

st.set_page_config(page_title="Segment Explorer", layout="wide")
st.title("🗺️ Segment Explorer")
st.caption(
    "The 2×2 matrix places every customer on two axes — churn risk and revenue value. "
    "Each quadrant has a distinct priority and action playbook."
)

scored  = load_scored()
display = load_display()
guard(scored, "Segment Explorer")

# ── 2×2 Matrix ────────────────────────────────────────────────────────────────
st.subheader("Risk-Value Matrix")
st.plotly_chart(matrix_figure(scored), use_container_width=True)

with st.expander("How segments are defined"):
    st.markdown("""
    | Axis | Definition |
    |------|------------|
    | **High Risk** | `P(churn) ≥ tuned threshold` from the best model |
    | **High Value** | Monthly charges ≥ median of all customers |

    Combining these two binary splits produces four mutually exclusive groups.
    Every customer belongs to exactly one segment.
    """)

st.markdown("---")

# ── Segment metrics overview ───────────────────────────────────────────────────
st.subheader("Segment Metrics at a Glance")

color_map = {v["label"]: v["color"] for v in SEGMENT_LABELS.values()}
ordered   = sorted(SEGMENT_LABELS.values(), key=lambda x: x["priority"])
cols      = st.columns(4)

for col, info in zip(cols, ordered):
    seg_df = scored[scored["segment_label"] == info["label"]]
    with col:
        st.markdown(
            f"<div style='border-top:4px solid {info['color']};padding:8px 0'>"
            f"<b style='color:{info['color']}'>{info['label']}</b><br>"
            f"Priority {info['priority']}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.metric("Customers",       f"{len(seg_df):,}")
        st.metric("Avg P(churn)",    fmt_pct(seg_df["churn_probability"].mean()))
        st.metric("Rev at Risk/mo",  fmt_dollar(seg_df["revenue_at_risk"].sum()))

st.markdown("---")

# ── Actual churn validation ────────────────────────────────────────────────────
if "Churn" in scored.columns:
    st.subheader("Model vs Reality: Actual Churn Rate per Segment")
    st.caption(
        "Validates the segmentation — 'Save Immediately' should have the highest "
        "actual churn rate in the labelled dataset."
    )
    st.plotly_chart(churn_rate_by_segment_figure(scored), use_container_width=True)

st.markdown("---")

# ── Drill-down ─────────────────────────────────────────────────────────────────
st.subheader("Drill Into a Segment")

segment_options = [
    v["label"]
    for v in sorted(SEGMENT_LABELS.values(), key=lambda x: x["priority"])
]
selected = st.selectbox("Select segment", options=segment_options)

info    = next(v for v in SEGMENT_LABELS.values() if v["label"] == selected)
seg_scored = scored[scored["segment_label"] == selected].copy()

# Use display data if available for readable columns
if not display.empty and "Contract" in display.columns:
    seg_display = display[display["segment_label"] == selected].copy()
else:
    seg_display = seg_scored

# KPIs for selected segment
m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers",           f"{len(seg_scored):,}")
m2.metric("Avg Churn Prob",      fmt_pct(seg_scored["churn_probability"].mean()))
m3.metric("Total Rev at Risk",   fmt_dollar(seg_scored["revenue_at_risk"].sum()))
m4.metric("Median Monthly Spend",fmt_dollar(seg_scored["MonthlyCharges"].median(), 2))

st.markdown(
    f"<div style='background:#f8f8f8;border-left:5px solid {info['color']};"
    f"padding:10px 16px;border-radius:4px;margin:8px 0'>"
    f"<b>Recommended action:</b> {info['action']}"
    f"</div>",
    unsafe_allow_html=True,
)

# Charge distribution within segment
fig_hist = px.histogram(
    seg_scored, x="MonthlyCharges",
    nbins=30, color_discrete_sequence=[info["color"]],
    title=f"Monthly Charges Distribution — {selected}",
    labels={"MonthlyCharges": "Monthly Charges ($)"},
)
fig_hist.update_layout(height=300)
st.plotly_chart(fig_hist, use_container_width=True)

# Customer table
st.markdown(f"#### Customers in '{selected}' (top 100 by revenue at risk)")
readable_cols = ["customerID", "churn_probability", "MonthlyCharges",
                 "revenue_at_risk", "tenure"]
extra_cols    = ["Contract", "InternetService", "PaymentMethod"]
all_cols      = readable_cols + [c for c in extra_cols if c in seg_display.columns]
available     = [c for c in all_cols if c in seg_display.columns]

show = (
    seg_display[available]
    .sort_values("revenue_at_risk", ascending=False)
    .head(100)
    .copy()
)
show["churn_probability"] = show["churn_probability"].map(fmt_pct)
show["revenue_at_risk"]   = show["revenue_at_risk"].map(lambda x: fmt_dollar(x, 2))
show["MonthlyCharges"]    = show["MonthlyCharges"].map(lambda x: fmt_dollar(x, 2))
st.dataframe(show, use_container_width=True, hide_index=True)

# Download button
csv = seg_display[available].sort_values("revenue_at_risk", ascending=False).to_csv(index=False)
st.download_button(
    label=f"⬇️ Download '{selected}' list as CSV",
    data=csv,
    file_name=f"segment_{selected.lower().replace(' ','_')}.csv",
    mime="text/csv",
)
