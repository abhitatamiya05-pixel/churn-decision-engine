"""
segmenter.py — Assign each customer to a 2×2 risk-value segment with action labels.

Segments are built from two independent axes:
  Risk tier  : high_risk  if churn_predicted == 1  else low_risk
  Value tier : high_value if MonthlyCharges >= median  else low_value

Each combination maps to a priority label and recommended retention action.

Run standalone:  python -m src.decision.segmenter
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import SCORED_FILE, SEGMENT_LABELS


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Risk tier from tuned-threshold binary prediction
    df["risk_tier"] = df["churn_predicted"].map(
        {1: "high_risk", 0: "low_risk"}
    )

    # Value tier: top 50% by monthly charges = high value
    median_charge = df["MonthlyCharges"].median()
    df["value_tier"] = (df["MonthlyCharges"] >= median_charge).map(
        {True: "high_value", False: "low_value"}
    )

    def _lookup(row):
        key = (row["risk_tier"], row["value_tier"])
        seg = SEGMENT_LABELS[key]
        return pd.Series({
            "segment_label":      seg["label"],
            "segment_priority":   seg["priority"],
            "recommended_action": seg["action"],
            "segment_color":      seg["color"],
        })

    df[["segment_label", "segment_priority",
        "recommended_action", "segment_color"]] = df.apply(_lookup, axis=1)

    return df


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per segment for the executive summary."""
    return (
        df.groupby(["segment_label", "segment_priority", "segment_color"])
        .agg(
            customers             =("customerID", "count"),
            avg_churn_prob        =("churn_probability", "mean"),
            total_revenue_at_risk =("revenue_at_risk", "sum"),
            avg_monthly_charges   =("MonthlyCharges", "mean"),
            median_tenure         =("tenure", "median"),
        )
        .reset_index()
        .sort_values("segment_priority")
        .round(2)
    )


def matrix_figure(df: pd.DataFrame) -> go.Figure:
    """Bubble chart of the 2×2 matrix, bubble size = revenue at risk."""
    from config.settings import SEGMENT_LABELS
    agg = (
        df.groupby(["segment_label", "risk_tier", "value_tier", "segment_color"])
        .agg(
            customers             =("customerID", "count"),
            avg_churn_prob        =("churn_probability", "mean"),
            total_revenue_at_risk =("revenue_at_risk", "sum"),
        )
        .reset_index()
    )
    color_map = {row["segment_label"]: row["segment_color"] for _, row in agg.iterrows()}
    fig = px.scatter(
        agg,
        x="value_tier", y="risk_tier",
        size="total_revenue_at_risk",
        color="segment_label",
        color_discrete_map=color_map,
        text="segment_label",
        hover_data={
            "customers": True,
            "avg_churn_prob": ":.1%",
            "total_revenue_at_risk": ":$,.0f",
            "risk_tier": False, "value_tier": False,
        },
        title="Customer Risk-Value Matrix  (bubble = monthly revenue at risk)",
        size_max=90,
    )
    fig.update_traces(textposition="middle center",
                      textfont=dict(size=10, color="white"))
    fig.update_layout(
        xaxis=dict(title="Customer Value",
                   categoryorder="array",
                   categoryarray=["low_value", "high_value"]),
        yaxis=dict(title="Churn Risk",
                   categoryorder="array",
                   categoryarray=["low_risk", "high_risk"]),
        height=440,
    )
    return fig


def churn_rate_by_segment_figure(df: pd.DataFrame) -> go.Figure:
    """Actual churn rate per segment (if ground-truth Churn column present)."""
    if "Churn" not in df.columns:
        return go.Figure()
    summary = (
        df.groupby("segment_label")
        .agg(
            actual_churn_rate=("Churn", "mean"),
            customers=("customerID", "count"),
        )
        .reset_index()
        .sort_values("actual_churn_rate", ascending=False)
    )
    fig = px.bar(
        summary, x="segment_label", y="actual_churn_rate",
        color="segment_label",
        color_discrete_map={
            "Save Immediately": "#d62728",
            "Assess & Offer":   "#ff7f0e",
            "Nurture & Upsell": "#2ca02c",
            "Monitor Only":     "#7f7f7f",
        },
        text=summary["actual_churn_rate"].map("{:.1%}".format),
        title="Actual Churn Rate by Segment",
        labels={"actual_churn_rate": "Churn Rate", "segment_label": "Segment"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%", height=380)
    return fig


if __name__ == "__main__":
    df = pd.read_csv(SCORED_FILE)
    df = assign_segments(df)
    df.to_csv(SCORED_FILE, index=False)

    print(f"[segmenter] Segments assigned. Distribution:")
    print(df["segment_label"].value_counts().to_string())
    print()
    print(segment_summary(df)[[
        "segment_label", "customers", "avg_churn_prob",
        "total_revenue_at_risk", "avg_monthly_charges"
    ]].to_string(index=False))
