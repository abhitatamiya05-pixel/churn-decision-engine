"""
explainer.py — Feature importance (native + SHAP) for RF and XGBoost.

Saves feature_importance.csv. Returns Plotly figures for the dashboard.
Run standalone:  python -m src.models.explainer
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import FEAT_IMP_FILE, FEATURES_FILE, MODEL_RF, MODEL_XGB
from src.data.features import get_model_columns

_COLORS = {"Random Forest": "#ff7f0e", "XGBoost": "#2ca02c", "SHAP": "#9467bd"}


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_artifacts():
    """Load feature data and available tree models. RF is None if pkl not committed."""
    df = pd.read_csv(FEATURES_FILE)
    feature_cols = get_model_columns(df)
    rf  = joblib.load(MODEL_RF) if MODEL_RF.exists() else None
    xgb = joblib.load(MODEL_XGB)
    return df, feature_cols, rf, xgb


# ── Native importance ──────────────────────────────────────────────────────────

def feature_importance_df(
    model, feature_cols: list[str], model_name: str
) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "feature":    feature_cols,
            "importance": model.feature_importances_,
            "model":      model_name,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def importance_figure(df_imp: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = df_imp.head(top_n).sort_values("importance")
    name = top["model"].iloc[0]
    fig = px.bar(
        top, x="importance", y="feature", orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        title=f"Top {top_n} Feature Importances — {name}",
        labels={"importance": "Importance Score", "feature": ""},
    )
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(tickfont=dict(size=11)),
                      height=460)
    return fig


# ── SHAP ───────────────────────────────────────────────────────────────────────

def shap_mean_abs(
    model, X_sample: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """
    Compute mean |SHAP| per feature using TreeExplainer.
    Falls back to native feature_importances_ if SHAP fails.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):   # binary: [neg_class, pos_class]
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        return (
            pd.DataFrame({"feature": feature_cols, "importance": mean_abs, "model": "SHAP"})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        print(f"[explainer] SHAP unavailable ({e}), falling back to feature importances.")
        return feature_importance_df(model, feature_cols, "SHAP-fallback")


def shap_summary(
    model, X_sample: pd.DataFrame, feature_cols: list[str]
) -> go.Figure:
    df_shap = shap_mean_abs(model, X_sample, feature_cols)
    top = df_shap.head(15).sort_values("importance")
    fig = px.bar(
        top, x="importance", y="feature", orientation="h",
        title="SHAP Feature Impact — XGBoost (mean |SHAP value|)",
        labels={"importance": "Mean |SHAP|", "feature": ""},
        color="importance", color_continuous_scale="Purples",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(tickfont=dict(size=11)),
                      height=460)
    return fig


# ── Logistic coefficients ──────────────────────────────────────────────────────

def logistic_coef_figure(model, feature_cols: list[str], top_n: int = 15) -> go.Figure:
    """
    Signed coefficient plot for logistic regression.
    Positive = higher probability of churn; negative = lower.
    """
    # Pipeline: first step is StandardScaler, clf is second
    clf  = model.named_steps["clf"]
    coef = clf.coef_[0]
    df_c = (
        pd.DataFrame({"feature": feature_cols, "coefficient": coef})
        .reindex(pd.Series(coef).abs().sort_values(ascending=False).index)
        .head(top_n)
        .sort_values("coefficient")
    )
    colors = ["#d62728" if c > 0 else "#2ca02c" for c in df_c["coefficient"]]
    fig = go.Figure(go.Bar(
        x=df_c["coefficient"], y=df_c["feature"],
        orientation="h", marker_color=colors,
    ))
    fig.update_layout(
        title=f"Logistic Regression — Top {top_n} Feature Coefficients",
        xaxis_title="Coefficient (+ = higher churn risk)",
        yaxis=dict(tickfont=dict(size=11)),
        height=460,
    )
    fig.add_vline(x=0, line_color="black", line_width=1)
    return fig


# ── Save ───────────────────────────────────────────────────────────────────────

def save_importance(rf_imp: pd.DataFrame, xgb_imp: pd.DataFrame,
                    shap_imp: pd.DataFrame) -> None:
    combined = pd.concat([rf_imp, xgb_imp, shap_imp], ignore_index=True)
    FEAT_IMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(FEAT_IMP_FILE, index=False)
    print(f"[explainer] Feature importance saved → {FEAT_IMP_FILE}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, feature_cols, rf, xgb = load_artifacts()
    sample = df[feature_cols].sample(500, random_state=42)

    rf_imp   = feature_importance_df(rf,  feature_cols, "Random Forest")
    xgb_imp  = feature_importance_df(xgb, feature_cols, "XGBoost")
    shap_imp = shap_mean_abs(xgb, sample, feature_cols)

    save_importance(rf_imp, xgb_imp, shap_imp)

    print("\n--- Top 10 by SHAP ---")
    print(shap_imp.head(10)[["feature", "importance"]].to_string(index=False))
