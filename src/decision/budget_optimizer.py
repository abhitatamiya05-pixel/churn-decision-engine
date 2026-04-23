"""
budget_optimizer.py — Allocate a fixed retention budget to maximise expected revenue saved.

Core formula:
  expected_save_value = MonthlyCharges × churn_probability × save_rate
  roi_score           = expected_save_value / intervention_cost

Customers are ranked by roi_score and selected greedily until budget exhausted.

Run standalone:  python -m src.decision.budget_optimizer
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    DEFAULT_BUDGET, DEFAULT_INTERVENTION_COST,
    DEFAULT_SAVE_RATE, SCORED_FILE,
)


# ── Core computation ───────────────────────────────────────────────────────────

def compute_roi(
    df: pd.DataFrame,
    save_rate: float = DEFAULT_SAVE_RATE,
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
    lifetime_months: int = 12,
) -> pd.DataFrame:
    """
    Attach expected_save_value and roi_score columns, sorted best-first.

    expected_save_value = MonthlyCharges × churn_probability × save_rate × lifetime_months
    This represents the total revenue protected over `lifetime_months` if the customer is saved.
    Default 12 months is conservative; real CLV horizon is longer.
    """
    df = df.copy()
    df["expected_save_value"] = (
        df["MonthlyCharges"] * df["churn_probability"] * save_rate * lifetime_months
    )
    df["roi_score"] = df["expected_save_value"] / intervention_cost
    return df.sort_values("roi_score", ascending=False).reset_index(drop=True)


def allocate_budget(
    df: pd.DataFrame,
    budget: float = DEFAULT_BUDGET,
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
    save_rate: float = DEFAULT_SAVE_RATE,
    segment_filter: str | None = None,
    lifetime_months: int = 12,
) -> dict:
    """
    Select the optimal customers to contact within a fixed budget.

    Parameters
    ----------
    df                : scored_customers DataFrame
    budget            : total spend available ($)
    intervention_cost : cost per customer contacted ($)
    save_rate         : assumed fraction of contacted customers who are retained
    segment_filter    : if set, restrict targeting to this segment label

    Returns
    -------
    dict with 'targeted' DataFrame and summary scalars
    """
    working = df.copy()
    if segment_filter:
        working = working[working["segment_label"] == segment_filter]

    working = compute_roi(working, save_rate, intervention_cost, lifetime_months)
    max_n   = int(budget // intervention_cost)
    targeted = working.head(max_n).copy()

    # Cumulative cost column for plotting
    targeted["cumulative_cost"] = (
        (targeted.index + 1) * intervention_cost
    )

    total_cost     = len(targeted) * intervention_cost
    expected_saved = targeted["expected_save_value"].sum()
    # do-nothing = sum of all revenue_at_risk in the working set (not just targeted)
    do_nothing     = df["revenue_at_risk"].sum()

    return {
        "targeted":               targeted,
        "n_targeted":             len(targeted),
        "total_cost":             total_cost,
        "expected_revenue_saved": expected_saved,
        "do_nothing_revenue_loss":do_nothing,
        "roi_multiple":           expected_saved / total_cost if total_cost > 0 else 0,
        "pct_risk_covered":       expected_saved / do_nothing if do_nothing > 0 else 0,
    }


# ── Impact curve ───────────────────────────────────────────────────────────────

def impact_curve(
    df: pd.DataFrame,
    max_budget: float = 50_000,
    step: float = 500,
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
    save_rate: float = DEFAULT_SAVE_RATE,
    lifetime_months: int = 12,
) -> pd.DataFrame:
    """
    Return DataFrame of (budget → expected_revenue_saved, n_customers)
    for plotting the simulator curve.
    """
    ranked = compute_roi(df, save_rate, intervention_cost, lifetime_months)
    rows   = []
    for budget in range(0, int(max_budget) + 1, int(step)):
        n     = int(budget // intervention_cost)
        saved = ranked.head(n)["expected_save_value"].sum()
        rows.append({
            "budget":                 budget,
            "expected_revenue_saved": saved,
            "n_customers":            n,
        })
    return pd.DataFrame(rows)


# ── Figures ────────────────────────────────────────────────────────────────────

def impact_curve_figure(
    df: pd.DataFrame,
    budget: float,
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
    save_rate: float = DEFAULT_SAVE_RATE,
) -> go.Figure:
    curve = impact_curve(df, max_budget=50_000, step=500,
                         intervention_cost=intervention_cost,
                         save_rate=save_rate)
    do_nothing = df["revenue_at_risk"].sum()
    result = allocate_budget(df, budget, intervention_cost, save_rate)

    fig = px.area(
        curve, x="budget", y="expected_revenue_saved",
        title="Expected Revenue Saved vs Retention Budget",
        labels={"budget": "Budget ($)", "expected_revenue_saved": "Est. Revenue Saved ($)"},
        color_discrete_sequence=["#2ca02c"],
    )
    fig.add_vline(
        x=budget, line_dash="dash", line_color="#d62728",
        annotation_text=f"Current ${budget:,.0f} → saves ${result['expected_revenue_saved']:,.0f}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=do_nothing, line_dash="dot", line_color="grey",
        annotation_text=f"Do-nothing loss ${do_nothing:,.0f}",
        annotation_position="right",
    )
    fig.update_layout(height=400)
    return fig


def comparison_bar_figure(result: dict) -> go.Figure:
    """Revenue lost: do-nothing vs with-budget comparison."""
    do_nothing = result["do_nothing_revenue_loss"]
    with_budget = do_nothing - result["expected_revenue_saved"]
    fig = go.Figure(go.Bar(
        x=["Do Nothing", f"With ${result['total_cost']:,.0f} Budget"],
        y=[do_nothing, with_budget],
        marker_color=["#d62728", "#2ca02c"],
        text=[f"${do_nothing:,.0f}", f"${with_budget:,.0f}"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Monthly Revenue Loss: With vs Without Retention Spend",
        yaxis_title="Revenue Lost ($)",
        yaxis=dict(range=[0, do_nothing * 1.2]),
        height=380,
    )
    return fig


def segment_roi_figure(df: pd.DataFrame,
                       intervention_cost: float = DEFAULT_INTERVENTION_COST,
                       save_rate: float = DEFAULT_SAVE_RATE) -> go.Figure:
    """Average ROI score per segment — shows where spend is most efficient."""
    tmp = compute_roi(df, save_rate, intervention_cost)
    agg = (
        tmp.groupby("segment_label")["roi_score"]
        .mean()
        .reset_index()
        .sort_values("roi_score", ascending=False)
    )
    fig = px.bar(
        agg, x="segment_label", y="roi_score",
        color="segment_label",
        color_discrete_map={
            "Save Immediately": "#d62728",
            "Assess & Offer":   "#ff7f0e",
            "Nurture & Upsell": "#2ca02c",
            "Monitor Only":     "#7f7f7f",
        },
        text=agg["roi_score"].map("{:.2f}×".format),
        title="Average ROI Score by Segment",
        labels={"roi_score": "Avg ROI Score", "segment_label": "Segment"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=380)
    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(SCORED_FILE)

    print(f"\n{'='*56}")
    print(f"  BUDGET ALLOCATION — ${DEFAULT_BUDGET:,} SCENARIO")
    print(f"  Intervention cost : ${DEFAULT_INTERVENTION_COST}/customer")
    print(f"  Assumed save rate : {DEFAULT_SAVE_RATE:.0%}")
    print(f"{'='*56}")

    result = allocate_budget(df, budget=DEFAULT_BUDGET)

    print(f"  Customers targeted    : {result['n_targeted']:,}")
    print(f"  Total spend           : ${result['total_cost']:,.0f}")
    print(f"  Expected revenue saved: ${result['expected_revenue_saved']:,.2f}/mo")
    print(f"  Do-nothing loss       : ${result['do_nothing_revenue_loss']:,.2f}/mo")
    print(f"  ROI multiple          : {result['roi_multiple']:.2f}×")
    print(f"  % of risk covered     : {result['pct_risk_covered']:.1%}")

    print(f"\n{'─'*56}")
    print("  Top 10 targeted customers:")
    top = result["targeted"][[
        "customerID", "segment_label", "churn_probability",
        "MonthlyCharges", "expected_save_value", "roi_score"
    ]].head(10)
    top["churn_probability"]   = top["churn_probability"].map("{:.1%}".format)
    top["expected_save_value"] = top["expected_save_value"].map("${:.2f}".format)
    top["roi_score"]           = top["roi_score"].map("{:.2f}×".format)
    print(top.to_string(index=False))
