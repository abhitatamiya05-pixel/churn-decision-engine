"""
scorer.py — Score all 7,043 customers using the best trained model.

Best model is determined by holdout AUC from model_results.json.
Outputs data/processed/scored_customers.csv with:
  - churn_probability    raw model probability
  - churn_predicted      binary prediction at tuned threshold
  - revenue_at_risk      MonthlyCharges × churn_probability
  - risk_rank            rank by revenue_at_risk descending

Run standalone:  python -m src.decision.scorer
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    DATA_OUTPUTS, FEATURES_FILE, MODEL_LOGISTIC, MODEL_RF, MODEL_XGB,
    SCORED_FILE, TARGET_COLUMN,
)
from src.data.features import get_model_columns

RESULTS_FILE = DATA_OUTPUTS / "model_results.json"

_MODEL_PATHS = {
    "logistic":      MODEL_LOGISTIC,
    "random_forest": MODEL_RF,
    "xgboost":       MODEL_XGB,
}


def pick_best_model() -> tuple[object, float, str]:
    """Return (fitted_model, threshold, model_name) for the model with highest test AUC."""
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    best_name = max(results, key=lambda k: results[k]["test_auc"])
    model     = joblib.load(_MODEL_PATHS[best_name])
    threshold = results[best_name]["threshold"]
    print(f"[scorer] Using best model: {best_name}  "
          f"(AUC={results[best_name]['test_auc']:.4f}, "
          f"threshold={threshold})")
    return model, threshold, best_name


def score_all() -> pd.DataFrame:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError("model_results.json missing — run trainer.py first.")

    df = pd.read_csv(FEATURES_FILE)
    feature_cols = get_model_columns(df)
    model, threshold, model_name = pick_best_model()

    X = df[feature_cols]
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["churn_predicted"]   = (df["churn_probability"] >= threshold).astype(int)

    # Revenue at risk — expected monthly revenue lost if this customer churns
    df["revenue_at_risk"] = df["churn_probability"] * df["MonthlyCharges"]

    # Rank 1 = most urgent (highest expected revenue loss)
    df["risk_rank"] = (
        df["revenue_at_risk"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    df["scoring_model"] = model_name

    scored = df.sort_values("risk_rank").reset_index(drop=True)
    SCORED_FILE.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(SCORED_FILE, index=False)
    print(f"[scorer] Scored {len(scored):,} customers → {SCORED_FILE}")

    # Quick sanity summary
    high_risk = (scored["churn_predicted"] == 1).sum()
    print(f"[scorer] High-risk customers (at threshold): {high_risk:,} "
          f"({high_risk/len(scored):.1%})")
    print(f"[scorer] Total monthly revenue at risk: "
          f"${scored['revenue_at_risk'].sum():,.0f}")
    return scored


if __name__ == "__main__":
    scored = score_all()
    print(scored[[
        "customerID", "churn_probability", "churn_predicted",
        "revenue_at_risk", "risk_rank"
    ]].head(10).to_string(index=False))
