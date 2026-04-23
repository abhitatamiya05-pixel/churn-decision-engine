"""
06_model_performance.py — ROC, PR, confusion matrices, metrics table, and SHAP explainability.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.settings import DATA_OUTPUTS, FEATURES_FILE, MODEL_XGB
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
    "Compare all three models across ROC-AUC, precision, recall, and F1. "
    "Understand what the model learned and what drives individual predictions."
)

if not MODEL_XGB.exists() or not RESULTS_FILE.exists():
    st.error("Model artifacts not found. Run `python -m src.models.run_pipeline` first.")
    st.stop()


@st.cache_resource(show_spinner="Loading models…")
def get_models():
    return load_models()


@st.cache_data(show_spinner="Loading test set…")
def get_test():
    return load_test_set()


@st.cache_data
def get_thresholds():
    return load_thresholds()


models     = get_models()
X_test, y_test = get_test()
thresholds = get_thresholds()

# ── Metrics table ─────────────────────────────────────────────────────────────
st.subheader("Model Comparison — Holdout Test Set")
mtable = metrics_table(models, thresholds, X_test, y_test)

# Load CV metrics from JSON for extra context
with open(RESULTS_FILE) as f:
    results_json = json.load(f)

key_map = {
    "Logistic Regression": "logistic",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}
mtable["CV-AUC (mean±std)"] = mtable["Model"].map(
    lambda n: f"{results_json[key_map[n]]['cv_auc_mean']:.4f}"
              f"±{results_json[key_map[n]]['cv_auc_std']:.4f}"
)

st.dataframe(
    mtable.style.highlight_max(
        subset=["ROC-AUC", "Recall (Churn)", "F1 (Churn)"],
        color="#d4f1d4",
    ).highlight_max(
        subset=["Precision (Churn)"],
        color="#cce5ff",
    ),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")
with st.expander("📖 How to read these metrics"):
    st.markdown("""
    | Metric | What it measures | Business implication |
    |--------|-----------------|---------------------|
    | **ROC-AUC** | Overall discrimination between churners and non-churners | Higher = model is better at ranking customers by risk |
    | **Precision (Churn)** | Of customers we flag as at-risk, how many actually churn | Low precision = wasted outreach budget |
    | **Recall (Churn)** | Of all actual churners, how many did we catch | Low recall = missed revenue — silent loss |
    | **F1 (Churn)** | Harmonic mean of precision and recall | Best single metric for imbalanced classification |
    | **CV-AUC** | 5-fold cross-validated AUC on training set | ±std tells you how stable the model is |

    **Threshold** is tuned per model on the validation set to maximise F1(churn).
    The default 0.5 threshold is sub-optimal for imbalanced data.
    """)

st.markdown("---")

# ── ROC and PR curves ─────────────────────────────────────────────────────────
st.subheader("ROC and Precision-Recall Curves")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(roc_curve_figure(models, X_test, y_test), use_container_width=True)
with col2:
    st.plotly_chart(precision_recall_figure(models, X_test, y_test), use_container_width=True)

st.caption(
    "The PR curve is more informative than ROC for imbalanced datasets. "
    "A random classifier would sit at the dashed baseline (≈26% churn rate). "
    "All three models substantially outperform random."
)

st.markdown("---")

# ── Confusion matrices ────────────────────────────────────────────────────────
st.subheader("Confusion Matrices")
st.caption(
    "False Negatives (churner predicted as retained) cost the most — "
    "they represent revenue lost without any intervention attempt."
)
cm_cols = st.columns(3)
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

tab_shap, tab_xgb, tab_rf, tab_lr = st.tabs([
    "🔮 SHAP (XGBoost)", "🌲 XGBoost Importance", "🌳 Random Forest Importance", "📐 Logistic Coefficients",
])

with tab_shap:
    st.plotly_chart(shap_summary(xgb, X_sample, feat_cols), use_container_width=True)
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows the average contribution of each "
        "feature to the churn prediction across 500 sampled customers. "
        "Features at the top have the most impact — in either direction."
    )

with tab_xgb:
    xgb_imp = feature_importance_df(xgb, feat_cols, "XGBoost")
    st.plotly_chart(importance_figure(xgb_imp), use_container_width=True)

with tab_rf:
    rf_imp = feature_importance_df(rf, feat_cols, "Random Forest")
    st.plotly_chart(importance_figure(rf_imp), use_container_width=True)

with tab_lr:
    import joblib
    from config.settings import MODEL_LOGISTIC
    lr_model = joblib.load(MODEL_LOGISTIC)
    st.plotly_chart(logistic_coef_figure(lr_model, feat_cols), use_container_width=True)
    st.caption(
        "Red bars = feature pushes the model toward predicting churn.  \n"
        "Green bars = feature pushes toward retention.  \n"
        "Coefficients are standardised — comparable in magnitude across features."
    )

st.markdown("---")

# ── Top driver interpretation ─────────────────────────────────────────────────
st.subheader("Business Interpretation of Top Drivers")
st.markdown("""
| Rank | Feature | What it means | Action lever |
|------|---------|---------------|--------------|
| 1 | `is_longterm_contract` | Customers **without** a long-term contract churn most | Offer contract upgrade incentives |
| 2 | `tenure` | Short-tenure customers churn at 5× the rate of long-tenure | Improve early onboarding experience |
| 3 | `avg_monthly_revenue` | Higher spenders feel more pressure to evaluate alternatives | Personalise value communication |
| 4 | `MonthlyCharges` | Directly correlated with revenue at risk | Tier retention spend by charge level |
| 5 | `is_fiber` | Fiber optic customers churn at 42% vs 7% for no-internet | Investigate service quality or pricing |
| 6 | `charges_delta` | Customers paying more than their avg feel overcharged | Proactive rate-review outreach |
| 7 | `Electronic check` | Manual pay method linked to 45% churn | Promote auto-pay adoption |
| 8 | `PaperlessBilling` | Linked to higher digital engagement and higher churn | Add value with digital-first perks |
| 9 | `OnlineSecurity` | Customers without security add-ons churn more | Bundle security at renewal |
""")
