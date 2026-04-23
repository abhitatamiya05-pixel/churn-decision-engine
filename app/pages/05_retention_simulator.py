"""
05_retention_simulator.py — Budget allocation simulator with impact curves and ROI projections.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.utils import load_scored, load_display, guard, fmt_pct, fmt_dollar
from config.settings import (
    DEFAULT_BUDGET, DEFAULT_INTERVENTION_COST, DEFAULT_SAVE_RATE, SEGMENT_LABELS,
)
from src.decision.budget_optimizer import (
    allocate_budget, impact_curve, impact_curve_figure, comparison_bar_figure,
    segment_roi_figure,
)

st.set_page_config(page_title="Retention Simulator", layout="wide")
st.title("💰 Retention Budget Simulator")
st.caption(
    "Set your budget, intervention cost, and save-rate assumption. "
    "The engine ranks customers by expected annual revenue saved and shows projected impact."
)

scored = load_scored()
guard(scored, "Retention Simulator")

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Simulation Parameters")
    st.markdown("---")

    budget = st.slider(
        "Retention Budget ($)",
        min_value=500, max_value=50_000, value=DEFAULT_BUDGET, step=500,
        format="$%d",
        help="Total one-time budget for customer outreach.",
    )
    intervention_cost = st.slider(
        "Cost per Intervention ($)",
        min_value=10, max_value=500, value=DEFAULT_INTERVENTION_COST, step=10,
        format="$%d",
        help="Fully loaded cost to contact and offer a deal to one customer.",
    )
    save_rate = st.slider(
        "Assumed Save Rate (%)",
        min_value=5, max_value=70, value=int(DEFAULT_SAVE_RATE * 100), step=5,
        format="%d%%",
        help="Fraction of contacted customers assumed to be successfully retained.",
    ) / 100
    lifetime_months = st.slider(
        "Customer Lifetime Horizon (months)",
        min_value=1, max_value=36, value=12, step=1,
        help="How many months of future revenue we attribute to a saved customer.",
    )
    st.markdown("---")

    segment_filter = st.selectbox(
        "Restrict targeting to segment",
        ["All Segments"] + [v["label"] for v in sorted(
            SEGMENT_LABELS.values(), key=lambda x: x["priority"]
        )],
        help="Optionally focus the budget on one segment only.",
    )
    seg_arg = None if segment_filter == "All Segments" else segment_filter

    st.markdown("---")
    st.caption(
        "**ROI formula:**  \n"
        "`expected_save = MonthlyCharges`  \n"
        "`× P(churn) × save_rate`  \n"
        "`× lifetime_months`  \n"
        "`roi = expected_save / intervention_cost`"
    )

# ── Compute allocation ─────────────────────────────────────────────────────────
result = allocate_budget(
    scored,
    budget=budget,
    intervention_cost=intervention_cost,
    save_rate=save_rate,
    segment_filter=seg_arg,
    lifetime_months=lifetime_months,
)

# ── KPI strip ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Budget",                    fmt_dollar(budget))
k2.metric("Customers Targeted",        f"{result['n_targeted']:,}")
k3.metric("Total Spend",               fmt_dollar(result["total_cost"]))
k4.metric(f"Est. Revenue Saved ({lifetime_months}mo)",
                                        fmt_dollar(result["expected_revenue_saved"]))
k5.metric("ROI Multiple",              f"{result['roi_multiple']:.1f}×",
          f"{fmt_pct(result['pct_risk_covered'])} of total risk")

st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────────
col_curve, col_bar = st.columns([2, 1])

with col_curve:
    st.plotly_chart(
        impact_curve_figure(scored, budget, intervention_cost, save_rate),
        use_container_width=True,
    )

with col_bar:
    st.plotly_chart(
        comparison_bar_figure(result),
        use_container_width=True,
    )

st.markdown("---")

# ── ROI by segment ─────────────────────────────────────────────────────────────
st.subheader("Where Is Spend Most Efficient?")
st.caption("Average ROI score per segment — helps decide whether to focus or spread the budget.")
st.plotly_chart(
    segment_roi_figure(scored, intervention_cost, save_rate),
    use_container_width=True,
)

st.markdown("---")

# ── Multi-budget scenario table ────────────────────────────────────────────────
st.subheader("Budget Scenario Comparison")
scenario_rows = []
for b in [1_000, 2_500, 5_000, 10_000, 25_000, 50_000]:
    r = allocate_budget(scored, budget=b,
                        intervention_cost=intervention_cost,
                        save_rate=save_rate,
                        lifetime_months=lifetime_months)
    scenario_rows.append({
        "Budget":                  fmt_dollar(b),
        "Customers":               f"{r['n_targeted']:,}",
        "Spend":                   fmt_dollar(r["total_cost"]),
        f"Revenue Saved ({lifetime_months}mo)": fmt_dollar(r["expected_revenue_saved"]),
        "ROI":                     f"{r['roi_multiple']:.1f}×",
        "% Risk Covered":          fmt_pct(r["pct_risk_covered"]),
    })
scenario_df = pd.DataFrame(scenario_rows)
# Highlight the currently selected budget row
st.dataframe(scenario_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Targeted customer list ─────────────────────────────────────────────────────
st.subheader(f"Targeted Customer List — {result['n_targeted']} customers")
st.caption("Sorted by ROI score (expected revenue saved ÷ intervention cost). Top rows = act first.")

targeted = result["targeted"][[
    "customerID", "segment_label", "churn_probability",
    "MonthlyCharges", "revenue_at_risk",
    "expected_save_value", "roi_score", "recommended_action",
]].copy()

targeted["churn_probability"]   = targeted["churn_probability"].map(fmt_pct)
targeted["MonthlyCharges"]      = targeted["MonthlyCharges"].map(lambda x: fmt_dollar(x, 2))
targeted["revenue_at_risk"]     = targeted["revenue_at_risk"].map(lambda x: fmt_dollar(x, 2))
targeted["expected_save_value"] = targeted["expected_save_value"].map(lambda x: fmt_dollar(x, 2))
targeted["roi_score"]           = targeted["roi_score"].map("{:.1f}×".format)
targeted.columns                = [c.replace("_", " ").title() for c in targeted.columns]

st.dataframe(targeted, use_container_width=True, hide_index=True)

csv = result["targeted"].to_csv(index=False)
st.download_button(
    "⬇️ Download targeted list as CSV",
    data=csv,
    file_name=f"retention_targets_budget_{int(budget)}.csv",
    mime="text/csv",
)
