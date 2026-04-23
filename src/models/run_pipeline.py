"""
run_pipeline.py — Full modeling + decision engine orchestrator.

Runs in sequence:
  1. Train all three models
  2. Evaluate on locked holdout test set
  3. Generate feature importance + SHAP
  4. Score all customers with best model
  5. Assign retention segments
  6. Run budget allocation
  7. Print business summary

Run:  python -m src.models.run_pipeline
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    DATA_OUTPUTS, DEFAULT_BUDGET, DEFAULT_INTERVENTION_COST,
    DEFAULT_SAVE_RATE, FEATURES_FILE, SCORED_FILE,
)
from src.data.features import get_model_columns
from src.models.trainer import (
    load_features, three_way_split, train_all,
    save_results_json, best_model_name, RESULTS_FILE, SPLITS_FILE,
)
from src.models.evaluator import (
    load_models, load_thresholds, load_test_set, metrics_table,
)
from src.models.explainer import (
    load_artifacts, feature_importance_df, shap_mean_abs, save_importance,
)
from src.decision.scorer import score_all
from src.decision.segmenter import assign_segments, segment_summary
from src.decision.budget_optimizer import allocate_budget


def _banner(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def run():
    # ── 1. Train ───────────────────────────────────────────────────────────────
    _banner("STEP 1/6 — Training Models")
    X, y, feature_cols = load_features()
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y)

    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_FILE, "w") as f:
        json.dump({"test_indices": X_test.index.tolist()}, f)

    results = train_all(X_train, y_train, X_val, y_val)
    save_results_json(results, X_test, y_test)

    # ── 2. Evaluate ────────────────────────────────────────────────────────────
    _banner("STEP 2/6 — Model Evaluation (Holdout Test Set)")
    models     = load_models()
    thresholds = load_thresholds()
    X_t, y_t  = load_test_set()
    mtable     = metrics_table(models, thresholds, X_t, y_t)

    header = f"{'Model':<22} {'Thr':>5} {'AUC':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}"
    print(header)
    print("─" * len(header))
    for _, row in mtable.iterrows():
        marker = " ← best" if row["F1 (Churn)"] == mtable["F1 (Churn)"].max() else ""
        print(
            f"  {row['Model']:<20} {row['Threshold']:>5.2f}"
            f" {row['ROC-AUC']:>7.4f} {row['Precision (Churn)']:>7.4f}"
            f" {row['Recall (Churn)']:>7.4f} {row['F1 (Churn)']:>7.4f}{marker}"
        )

    with open(RESULTS_FILE) as f:
        rjson = json.load(f)
    best = best_model_name(rjson)
    print(f"\n  ✓ Best model (AUC): {best}  "
          f"AUC={rjson[best]['test_auc']}  "
          f"F1={rjson[best]['test_f1']}")

    # ── 3. Explainability ──────────────────────────────────────────────────────
    _banner("STEP 3/6 — Feature Importance + SHAP")
    df_feat, feat_cols, rf, xgb = load_artifacts()
    sample   = df_feat[feat_cols].sample(500, random_state=42)
    rf_imp   = feature_importance_df(rf,  feat_cols, "Random Forest")
    xgb_imp  = feature_importance_df(xgb, feat_cols, "XGBoost")
    shap_imp = shap_mean_abs(xgb, sample, feat_cols)
    save_importance(rf_imp, xgb_imp, shap_imp)

    print("\n  Top 10 drivers (SHAP):")
    for _, row in shap_imp.head(10).iterrows():
        bar = "▓" * int(row["importance"] / shap_imp["importance"].max() * 20)
        print(f"    {row['feature']:<35} {row['importance']:.4f}  {bar}")

    # ── 4. Score ───────────────────────────────────────────────────────────────
    _banner("STEP 4/6 — Scoring All Customers")
    scored = score_all()

    # ── 5. Segment ─────────────────────────────────────────────────────────────
    _banner("STEP 5/6 — Retention Segmentation")
    scored = assign_segments(scored)
    scored.to_csv(SCORED_FILE, index=False)
    seg_sum = segment_summary(scored)
    print(f"\n  {'Segment':<22} {'N':>6} {'Avg P(churn)':>13} {'Rev at Risk':>13} {'Avg Charge':>11}")
    print("  " + "─" * 70)
    for _, row in seg_sum.iterrows():
        print(
            f"  {row['segment_label']:<22} {row['customers']:>6,}"
            f" {row['avg_churn_prob']:>12.1%}"
            f" ${row['total_revenue_at_risk']:>11,.0f}"
            f" ${row['avg_monthly_charges']:>9.2f}"
        )

    # ── 6. Budget allocation ───────────────────────────────────────────────────
    _banner("STEP 6/6 — Budget Allocation Scenarios")
    for budget in [2_500, 5_000, 10_000, 25_000]:
        res = allocate_budget(
            scored, budget=budget,
            intervention_cost=DEFAULT_INTERVENTION_COST,
            save_rate=DEFAULT_SAVE_RATE,
            lifetime_months=12,
        )
        print(
            f"  ${budget:>6,} budget → {res['n_targeted']:>4} customers"
            f"  |  Est. revenue saved: ${res['expected_revenue_saved']:>8,.0f}/yr"
            f"  |  ROI: {res['roi_multiple']:.1f}×"
        )

    # ── Final summary ──────────────────────────────────────────────────────────
    _banner("MODELING COMPLETE — BUSINESS SUMMARY")
    total        = len(scored)
    high_risk_n  = (scored["churn_predicted"] == 1).sum()
    save_imm     = scored[scored["segment_label"] == "Save Immediately"]
    do_nothing   = scored["revenue_at_risk"].sum()

    print(f"""
  Dataset           : {total:,} customers
  Overall churn rate: {scored['Churn'].mean():.1%}  (actual labels)
  Predicted at-risk : {high_risk_n:,} ({high_risk_n/total:.1%})

  ┌─ Revenue Exposure ──────────────────────────────────────┐
  │  Monthly revenue at risk (all)   : ${do_nothing:>10,.0f}        │
  │  From "Save Immediately" segment : ${save_imm['revenue_at_risk'].sum():>10,.0f}        │
  │  % from top segment              : {save_imm['revenue_at_risk'].sum()/do_nothing:.0%}                     │
  └─────────────────────────────────────────────────────────┘

  ┌─ Top Churn Drivers (SHAP) ──────────────────────────────┐""")
    for _, row in shap_imp.head(5).iterrows():
        print(f"  │  {row['feature']:<35} {row['importance']:.4f}          │")
    print("  └─────────────────────────────────────────────────────────┘")
    print(f"\n  ✓ All artifacts saved to data/outputs/")
    print(f"  ✓ Scored customers saved to data/processed/scored_customers.csv")
    print(f"  ✓ Launch dashboard: streamlit run app/main.py\n")


if __name__ == "__main__":
    run()
