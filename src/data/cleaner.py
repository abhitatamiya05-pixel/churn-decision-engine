"""
cleaner.py — Data cleaning: types, missing values, outlier flags.

Run standalone:  python -m src.data.cleaner
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import DATA_PROCESSED, RAW_FILE
from src.data.loader import load_raw


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is stored as object in the raw file due to whitespace on new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # New customers (tenure == 0) have no TotalCharges — impute with MonthlyCharges
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"]
    print(f"[cleaner] Imputed {mask.sum()} missing TotalCharges values.")

    # Binary target: Yes/No → 1/0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Normalise SeniorCitizen (already 0/1, but ensure int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # Flag extreme MonthlyCharges outliers (>3 IQR fences) — don't remove, just mark
    q1, q3 = df["MonthlyCharges"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["is_charges_outlier"] = (
        (df["MonthlyCharges"] < q1 - 3 * iqr) |
        (df["MonthlyCharges"] > q3 + 3 * iqr)
    ).astype(int)

    print(f"[cleaner] Cleaning complete. Shape: {df.shape}")
    return df


def save(df: pd.DataFrame, filename: str = "cleaned.csv") -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    df.to_csv(path, index=False)
    print(f"[cleaner] Saved to {path}")
    return path


if __name__ == "__main__":
    raw = load_raw(RAW_FILE)
    cleaned = clean(raw)
    save(cleaned)
    print(cleaned.dtypes)
    print(f"Churn rate: {cleaned['Churn'].mean():.2%}")
