"""
02_churn_overview.py — Churn rates by segment, tenure, charges, services, and cohort matrix.
"""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.utils import load_display, guard, fmt_pct, fmt_dollar
from src.analysis.eda import (
    overall_churn_rate, churn_by_category,
    churn_by_tenure, charges_distribution,
    revenue_at_risk_by_segment, churn_heatmap,
)
from src.analysis.cohorts import (
    churn_by_tenure_contract, service_adoption_vs_churn,
    revenue_lifecycle, cohort_churn_table,
)

st.set_page_config(page_title="Churn Overview", layout="wide")
st.title("📉 Churn Overview")
st.caption("Where churn is highest, who is churning, and how revenue exposure is distributed.")

df = load_display()
guard(df, "Churn Overview")

stats = overall_churn_rate(df)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Overall Churn Rate",      fmt_pct(stats["churn_rate"]))
k2.metric("Customers Churned",       f"{stats['churned']:,}")
k3.metric("Customers Retained",      f"{stats['retained']:,}")
k4.metric("Monthly Revenue at Risk", fmt_dollar(stats["monthly_rev_at_risk"]))

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 By Segment", "📅 Tenure & Cohorts", "💰 Revenue Analysis", "🔥 Heatmaps",
])

# ── Tab 1: By Segment ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### What customer attributes predict churn?")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            churn_by_category(df, "Contract", "Churn Rate by Contract Type"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            churn_by_category(df, "InternetService", "Churn Rate by Internet Service"),
            use_container_width=True,
        )
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(
            churn_by_category(df, "PaymentMethod", "Churn Rate by Payment Method"),
            use_container_width=True,
        )
    with col_d:
        if "SeniorCitizen" in df.columns:
            df_s = df.copy()
            df_s["Senior"] = df_s["SeniorCitizen"].map({1: "Senior", 0: "Non-Senior"})
            st.plotly_chart(
                churn_by_category(df_s, "Senior", "Churn Rate: Senior vs Non-Senior"),
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("#### Key insight table")
    insight_data = []
    for col, label in [
        ("Contract", "Contract Type"),
        ("InternetService", "Internet Service"),
        ("PaymentMethod", "Payment Method"),
    ]:
        if col in df.columns:
            g = df.groupby(col)["Churn"].agg(["mean", "count"]).reset_index()
            g.columns = ["Value", "Churn Rate", "Customers"]
            g["Attribute"] = label
            g["Churn Rate"] = g["Churn Rate"].map(fmt_pct)
            insight_data.append(g[["Attribute", "Value", "Customers", "Churn Rate"]])
    if insight_data:
        import pandas as pd
        st.dataframe(
            pd.concat(insight_data, ignore_index=True),
            use_container_width=True, hide_index=True,
        )

# ── Tab 2: Tenure & Cohorts ───────────────────────────────────────────────────
with tab2:
    st.markdown("#### How does churn change as customers age?")
    st.plotly_chart(churn_by_tenure(df),            use_container_width=True)
    st.plotly_chart(churn_by_tenure_contract(df),   use_container_width=True)
    st.plotly_chart(service_adoption_vs_churn(df),  use_container_width=True)

    st.markdown("#### Cohort Churn Matrix — Contract × Tenure Band")
    st.caption(
        "Each cell shows the historical churn rate for that contract-tenure combination. "
        "Darker = higher churn. The top-left cell (new month-to-month customers) is the riskiest."
    )
    cohort = cohort_churn_table(df)
    # Format as percentages for display
    pct_cols = [c for c in cohort.columns if c != "tenure_band"]
    cohort_display = cohort.copy()
    for c in pct_cols:
        cohort_display[c] = cohort_display[c].map(
            lambda x: fmt_pct(x) if isinstance(x, float) else x
        )
    st.dataframe(cohort_display, use_container_width=True, hide_index=True)

# ── Tab 3: Revenue Analysis ───────────────────────────────────────────────────
with tab3:
    st.markdown("#### How does spend relate to churn behaviour?")
    st.plotly_chart(charges_distribution(df),                        use_container_width=True)
    st.plotly_chart(revenue_lifecycle(df),                           use_container_width=True)
    st.plotly_chart(revenue_at_risk_by_segment(df, "Contract"),      use_container_width=True)
    if "InternetService" in df.columns:
        st.plotly_chart(
            revenue_at_risk_by_segment(df, "InternetService"),
            use_container_width=True,
        )

    # Churned vs retained spend comparison
    st.markdown("#### Revenue comparison: churned vs retained")
    import pandas as pd
    compare = (
        df.groupby(df["Churn"].map({1: "Churned", 0: "Retained"}))
        .agg(
            Customers       =("customerID", "count"),
            Avg_Monthly     =("MonthlyCharges", "mean"),
            Avg_Tenure      =("tenure", "mean"),
            Avg_Total       =("TotalCharges", "mean"),
        )
        .reset_index()
        .rename(columns={"Churn": "Status"})
    )
    compare["Avg_Monthly"] = compare["Avg_Monthly"].map(lambda x: fmt_dollar(x, 2))
    compare["Avg_Total"]   = compare["Avg_Total"].map(lambda x: fmt_dollar(x, 2))
    compare["Avg_Tenure"]  = compare["Avg_Tenure"].map(lambda x: f"{x:.1f} mo")
    st.dataframe(compare, use_container_width=True, hide_index=True)

# ── Tab 4: Heatmaps ───────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Cross-dimensional churn rate heatmaps")
    col_e, col_f = st.columns(2)
    with col_e:
        st.plotly_chart(
            churn_heatmap(df, "Contract", "tenure_band"),
            use_container_width=True,
        )
    with col_f:
        st.plotly_chart(
            churn_heatmap(df, "InternetService", "Contract"),
            use_container_width=True,
        )
    if "PaymentMethod" in df.columns:
        st.plotly_chart(
            churn_heatmap(df, "PaymentMethod", "Contract"),
            use_container_width=True,
        )
