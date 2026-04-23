"""
features.py — Feature engineering pipeline.

Produces the final feature matrix saved to data/processed/features.csv.
Run standalone:  python -m src.data.features
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    DATA_PROCESSED, FEATURES_FILE, HIGH_VALUE_PERCENTILE,
    TENURE_BINS, TENURE_LABELS,
)


def load_cleaned() -> pd.DataFrame:
    path = DATA_PROCESSED / "cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run cleaner.py first. Expected: {path}")
    return pd.read_csv(path)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Tenure band ────────────────────────────────────────────────────────────
    df["tenure_band"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS,
        right=True, include_lowest=True   # include tenure=0
    )

    # ── Revenue features ───────────────────────────────────────────────────────
    df["avg_monthly_revenue"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )
    df["charges_delta"] = df["MonthlyCharges"] - df["avg_monthly_revenue"]

    # Rough CLV: assume average remaining tenure is 24 months if not churned
    df["clv_estimate"] = df["MonthlyCharges"] * 24

    # ── Service breadth ────────────────────────────────────────────────────────
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["service_count"] = (
        df[service_cols]
        .apply(lambda col: (col == "Yes").astype(int))
        .sum(axis=1)
    )

    # ── Convenience binary flags ───────────────────────────────────────────────
    df["is_autopay"] = df["PaymentMethod"].str.contains(
        "automatic", case=False, na=False
    ).astype(int)

    df["is_longterm_contract"] = (
        df["Contract"].isin(["One year", "Two year"])
    ).astype(int)

    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)

    df["is_senior"] = df["SeniorCitizen"].astype(int)

    # ── Value tier (used by decision engine, not model) ────────────────────────
    threshold = df["MonthlyCharges"].quantile(HIGH_VALUE_PERCENTILE)
    df["is_high_value"] = (df["MonthlyCharges"] >= threshold).astype(int)

    # ── Encode categoricals for ML ─────────────────────────────────────────────
    binary_map = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines",
    ]
    for col in binary_cols:
        df[col] = df[col].map(binary_map).fillna(0).astype(int)

    # One-hot encode remaining categoricals (cast to int for sklearn compatibility)
    df = pd.get_dummies(
        df,
        columns=["gender", "InternetService", "Contract", "PaymentMethod"],
        drop_first=False,
        dtype=int,
    )

    # Drop tenure_band (string) — already captured numerically; keep for display
    df["tenure_band"] = df["tenure_band"].astype(str)

    print(f"[features] Final shape: {df.shape}")
    return df


def get_model_columns(df: pd.DataFrame) -> list[str]:
    """Return only numeric columns suitable as ML features (exclude IDs/target/flags)."""
    exclude = {"customerID", "Churn", "tenure_band", "is_high_value", "is_charges_outlier"}
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]


def save(df: pd.DataFrame) -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_FILE, index=False)
    print(f"[features] Saved to {FEATURES_FILE}")
    return FEATURES_FILE


if __name__ == "__main__":
    cleaned = load_cleaned()
    featured = engineer(cleaned)
    save(featured)
    model_cols = get_model_columns(featured)
    print(f"Model feature columns ({len(model_cols)}): {model_cols[:10]} ...")
