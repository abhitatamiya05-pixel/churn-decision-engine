"""
utils.py — Shared data loading helpers for all Streamlit pages.

Three data layers:
  scored   → scored_customers.csv  (model scores, segments, one-hot features)
  display  → cleaned.csv merged with scored columns (readable categories for EDA)
  raw      → telco_churn.csv (original, for reference)
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import DATA_PROCESSED, SCORED_FILE

_CLEANED_FILE = DATA_PROCESSED / "cleaned.csv"

# Columns to pull from scored into the display frame
_SCORE_COLS = [
    "customerID", "churn_probability", "churn_predicted",
    "revenue_at_risk", "risk_rank", "scoring_model",
    "risk_tier", "value_tier", "is_high_value",
    "segment_label", "segment_priority", "recommended_action", "segment_color",
    "service_count", "tenure_band", "is_autopay", "is_longterm_contract",
    "avg_monthly_revenue", "charges_delta", "clv_estimate",
]


@st.cache_data(show_spinner="Loading customer data…")
def load_scored() -> pd.DataFrame:
    """Full scored dataset — one-hot encoded features + all model outputs."""
    if not SCORED_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(SCORED_FILE)


@st.cache_data(show_spinner="Loading display data…")
def load_display() -> pd.DataFrame:
    """
    Cleaned dataset (readable categories) merged with scored model outputs.
    Use this for any chart that needs Contract / InternetService / PaymentMethod
    as human-readable strings.
    """
    if not _CLEANED_FILE.exists() or not SCORED_FILE.exists():
        return pd.DataFrame()

    cleaned = pd.read_csv(_CLEANED_FILE)
    scored  = pd.read_csv(SCORED_FILE)

    available = [c for c in _SCORE_COLS if c in scored.columns]
    merged = cleaned.merge(scored[available], on="customerID", how="left")

    # Derive tenure_band from cleaned if not present (shouldn't happen)
    if "tenure_band" not in merged.columns:
        merged["tenure_band"] = pd.cut(
            merged["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"],
            right=True, include_lowest=True,
        ).astype(str)

    return merged


def guard(df: pd.DataFrame, page: str = "") -> bool:
    """Return False and show an error if the dataframe is empty."""
    if df.empty:
        st.error(
            "Pipeline outputs not found. "
            "Run `bash run.sh` (or `python -m src.models.run_pipeline`) first."
        )
        if page:
            st.caption(f"Missing data needed by: {page}")
        st.stop()
        return False
    return True


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def fmt_dollar(x: float, decimals: int = 0) -> str:
    return f"${x:,.{decimals}f}"
