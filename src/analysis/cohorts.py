"""
cohorts.py — Cohort and lifecycle analysis by tenure, contract age, and service mix.

All functions return Plotly figures.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def churn_by_tenure_contract(df: pd.DataFrame) -> go.Figure:
    """Churn rate across tenure bands, faceted by contract type."""
    summary = (
        df.groupby(["tenure_band", "Contract"])["Churn"]
        .mean()
        .reset_index()
        .rename(columns={"Churn": "churn_rate"})
    )
    fig = px.line(
        summary,
        x="tenure_band",
        y="churn_rate",
        color="Contract",
        markers=True,
        title="Churn Rate by Tenure Band and Contract Type",
        labels={"churn_rate": "Churn Rate", "tenure_band": "Tenure"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def service_adoption_vs_churn(df: pd.DataFrame) -> go.Figure:
    """Scatter: service count vs churn rate, sized by customer count."""
    summary = (
        df.groupby("service_count")
        .agg(churn_rate=("Churn", "mean"), customers=("Churn", "count"))
        .reset_index()
    )
    fig = px.scatter(
        summary,
        x="service_count",
        y="churn_rate",
        size="customers",
        color="churn_rate",
        color_continuous_scale="RdYlGn_r",
        title="Churn Rate vs. Number of Services Subscribed",
        labels={"service_count": "# Active Services", "churn_rate": "Churn Rate"},
        text="service_count",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(coloraxis_showscale=False)
    return fig


def revenue_lifecycle(df: pd.DataFrame) -> go.Figure:
    """Average monthly charges over tenure months — shows upsell and lifecycle arc."""
    summary = (
        df.groupby("tenure")["MonthlyCharges"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_charge", "median": "median_charge"})
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary["tenure"], y=summary["avg_charge"],
        mode="lines", name="Avg Monthly Charge",
        line=dict(color="#1f77b4", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=summary["tenure"], y=summary["median_charge"],
        mode="lines", name="Median Monthly Charge",
        line=dict(color="#ff7f0e", width=2, dash="dash")
    ))
    fig.update_layout(
        title="Monthly Charges Over Customer Lifetime",
        xaxis_title="Tenure (months)",
        yaxis_title="Monthly Charges ($)",
    )
    return fig


def cohort_churn_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a pivot table: churn rate by tenure_band × contract type."""
    return (
        df.pivot_table(
            values="Churn",
            index="tenure_band",
            columns="Contract",
            aggfunc="mean",
        )
        .round(3)
        .reset_index()
    )
