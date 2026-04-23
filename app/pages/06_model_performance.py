"""
06_model_performance.py — ROC, PR, confusion matrices, metrics table, SHAP explainability.
Gracefully skips any model whose .pkl is not present (e.g. model_rf.pkl on Streamlit Cloud).
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.settings import DATA_OUTPUTS, FEATURES_FILE, MODEL_LOGISTIC, MODEL_RF, MODEL_XGB
from src.data.features import get_model_columns
from src.models.evaluator import (
    load_models, load_thresholds, load_test_set,
    roc_curve_figure, precision_recall_figure,
    confusion_matrix_figure, metrics_table,
)
from src.models.explainer import (
    load_artifacts, feature_importance_df,
    importance_figure, shap_summary, logistic_coef_figure,
)

RESULTS_FILE = DATA_OUTPUTS / "model_results.json"

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("🤖 Model Performance")
st.caption(
    "Compare models across ROC-AUC, precision, recall, and F1. "
    "Understand what the model learned and what drives individual predictions."
)

if not MODEL_XGB.exists() or not RESULTS_FILE.exists():
    st.error("Core model artifacts not found. Commit model_xgb.pkl and model_results.json.")
    st.stop()

if not MODEL_RF.exists():
    st.info("ℹ️ Random Forest model not available in this deployment — showing Logistic Regression and XGBoost only.")


@st.cache_resource(show_spinner="Loading models…")
def get_models():
    return load_models()   # returns only models whose pkl files exist


@st.cache_data(show_spinner="Loading test set…")
def get_test():
    return load_test_set()


@st.cache_data
def get_thresholds():
    return load_thresholds()


models     = get_models()
X_test, y_test = get_test()
thresholds = get_thresholds()

# ── Metrics table ──────────────────────────────────────────────────────────────
st.subheader("Model Comparison — Holdout Test Set")
mtable = metrics_table(models, thresholds, X_test, y_test)

with open(RESULTS_FILE) as f:
    results_json = json.load(f)

key_map = {
    "Logistic Regression": "logistic",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}
mtable["CV-AUC (mean±std)"] = mtable["Model"].map(
    lambda n: (
        f"{results_json[key_map[n]]['cv_auc_mean']:.4f}"
        f"±{results_json[key_map[n]]['cv_auc_std']:.4f}"
    ) if key_map.get(n) in results_json else "—"
)

highlight_cols = [c for c in ["ROC-AUC", "Recall (Churn)", "F1 (Churn)"] if c in mtable.columns]
st.dataframe(
    mtable.style.highlight_max(subset=highlight_cols, color="#d4f1d4")
              .highlight_max(subset=["Precision (Churn)"], color="#cce5ff"),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")
with st.expander("📖 How to read these metrics"):
    st.markdown("""
    | Metric | Business implication |
    |--------|---------------------|
    | **ROC-AUC** | Higher = better at ranking customers by churn risk |
    | **Precision (Churn)** | Low = wasted outreach budget |
    | **Recall (Churn)** | Low = churners missed, revenue lost silently |
    | **F1 (Churn)** | Primary selection metric for imbalanced data |
    | **Threshold** | Tuned per-model on val set to maximise F1(churn) — not assumed 0.5 |
    """)

st.markdown("---")

# ── ROC and PR curves ─────────────────────────────────────────────────────────
st.subheader("ROC and Precision-Recall Curves")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(roc_curve_figure(models, X_test, y_test), use_container_width=True)
with col2:
    st.plotly_chart(precision_recall_figure(models, X_test, y_test), use_container_width=True)

st.markdown("---")

# ── Confusion matrices — one per available model ──────────────────────────────
st.subheader("Confusion Matrices")
st.caption("False Negatives cost the most — churners we failed to flag.")
cm_cols = st.columns(len(models))
for col, (name, model) in zip(cm_cols, models.items()):
    with col:
        st.plotly_chart(
            confusion_matrix_figure(model, thresholds[name], X_test, y_test, name),
            use_container_width=True,
        )

st.markdown("---")

# ── Feature importance and SHAP ───────────────────────────────────────────────
st.subheader("What Drives Churn Predictions?")

df_feat, feat_cols, rf, xgb = load_artifacts()
X_sample = df_feat[feat_cols].sample(500, random_state=42)

# Build tabs dynamically based on what's available
tab_labels = ["🔮 SHAP (XGBoost)", "🌲 XGBoost Importance", "📐 Logistic Coefficients"]
if rf is not None:
    tab_labels.insert(2, "🌳 Random Forest Importance")

tabs = st.tabs(tab_labels)
tab_iter = iter(tabs)

with next(tab_iter):
    st.plotly_chart(shap_summary(xgb, X_sample, feat_cols), use_container_width=True)
    st.caption(
        "Mean |SHAP value| across 500 sampled customers. "
        "Features at top have the most impact on predictions."
    )

with next(tab_iter):
    st.plotly_chart(importance_figure(feature_importance_df(xgb, feat_cols, "XGBoost")), use_container_width=True)

if rf is not None:
    with next(tab_iter):
        st.plotly_chart(importance_figure(feature_importance_df(rf, feat_cols, "Random Forest")), use_container_width=True)

with next(tab_iter):
    if MODEL_LOGISTIC.exists():
        lr_model = joblib.load(MODEL_LOGISTIC)
        st.plotly_chart(logistic_coef_figure(lr_model, feat_cols), use_container_width=True)
        st.caption("Red = increases churn risk · Green = decreases churn risk · Coefficients are standardised.")
    else:
        st.info("Logistic Regression model not available.")

st.markdown("---")

# ── Driver table ──────────────────────────────────────────────────────────────
st.subheader("Business Interpretation of Top Drivers")
st.markdown("""
| Rank | Feature | What it means | Action lever |
|------|---------|---------------|--------------|
| 1 | `is_longterm_contract` | No long-term contract → highest churn | Offer contract upgrade incentives |
| 2 | `tenure` | Short-tenure customers churn at 5× the rate | Improve onboarding experience |
| 3 | `avg_monthly_revenue` | Higher spenders feel pressure to evaluate alternatives | Personalise value communication |
| 4 | `MonthlyCharges` | Directly correlated with revenue at risk | Tier retention spend by charge level |
| 5 | `is_fiber` | Fiber churn at 42% vs 7% no-internet | Investigate service quality or pricing |
| 6 | `Electronic check` | Manual pay method linked to 45% churn | Promote auto-pay adoption |
| 7 | `OnlineSecurity` | No security add-on → lower perceived value | Bundle security at renewal |
""")
