"""
eda.py — Exploratory data analysis: churn rates, distributions, segment breakdowns.

All functions return Plotly figures so they can be embedded directly in Streamlit.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def overall_churn_rate(df: pd.DataFrame) -> dict:
    """Return summary stats: overall churn rate, counts, revenue at risk."""
    total = len(df)
    churned = df["Churn"].sum()
    rate = churned / total
    rev_at_risk = df.loc[df["Churn"] == 1, "MonthlyCharges"].sum()
    return {
        "total_customers": total,
        "churned": int(churned),
        "retained": total - int(churned),
        "churn_rate": rate,
        "monthly_rev_at_risk": rev_at_risk,
    }


def churn_by_category(df: pd.DataFrame, col: str, title: str = "") -> go.Figure:
    """Bar chart of churn rate by a categorical column."""
    summary = (
        df.groupby(col)["Churn"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "churn_rate", "count": "customers"})
        .sort_values("churn_rate", ascending=False)
    )
    fig = px.bar(
        summary,
        x=col,
        y="churn_rate",
        color="churn_rate",
        color_continuous_scale="RdYlGn_r",
        text=summary["churn_rate"].map("{:.1%}".format),
        hover_data={"customers": True},
        title=title or f"Churn Rate by {col}",
        labels={"churn_rate": "Churn Rate"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
    return fig


def churn_by_tenure(df: pd.DataFrame) -> go.Figure:
    """Churn rate and volume by tenure band."""
    summary = (
        df.groupby("tenure_band")["Churn"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "churn_rate", "count": "customers"})
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=summary["tenure_band"], y=summary["customers"],
               name="Customers", marker_color="#aec7e8"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=summary["tenure_band"], y=summary["churn_rate"],
                   name="Churn Rate", mode="lines+markers",
                   marker=dict(color="#d62728", size=8), line=dict(width=2)),
        secondary_y=True,
    )
    fig.update_layout(title="Customer Volume & Churn Rate by Tenure Band")
    fig.update_yaxes(title_text="Customers", secondary_y=False)
    fig.update_yaxes(title_text="Churn Rate", tickformat=".0%", secondary_y=True)
    return fig


def charges_distribution(df: pd.DataFrame) -> go.Figure:
    """Overlapping histogram of MonthlyCharges for churned vs retained."""
    fig = px.histogram(
        df,
        x="MonthlyCharges",
        color=df["Churn"].map({1: "Churned", 0: "Retained"}),
        barmode="overlay",
        opacity=0.6,
        nbins=40,
        title="Monthly Charges Distribution: Churned vs Retained",
        color_discrete_map={"Churned": "#d62728", "Retained": "#2ca02c"},
        labels={"color": "Status"},
    )
    return fig


def revenue_at_risk_by_segment(df: pd.DataFrame, segment_col: str) -> go.Figure:
    """Stacked bar: monthly revenue split by churn status within each segment."""
    summary = (
        df.groupby([segment_col, df["Churn"].map({1: "At Risk", 0: "Retained"})])
        ["MonthlyCharges"].sum()
        .reset_index()
        .rename(columns={"MonthlyCharges": "revenue", "Churn": "status"})
    )
    fig = px.bar(
        summary,
        x=segment_col,
        y="revenue",
        color="status",
        color_discrete_map={"At Risk": "#d62728", "Retained": "#2ca02c"},
        title=f"Monthly Revenue at Risk by {segment_col}",
        labels={"revenue": "Monthly Revenue ($)"},
    )
    return fig


def churn_heatmap(df: pd.DataFrame, col_x: str, col_y: str) -> go.Figure:
    """Heatmap of churn rate across two categorical dimensions."""
    pivot = df.pivot_table(
        values="Churn", index=col_y, columns=col_x, aggfunc="mean"
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        zmin=0, zmax=1,
        text_auto=".0%",
        title=f"Churn Rate: {col_y} × {col_x}",
    )
    return fig
