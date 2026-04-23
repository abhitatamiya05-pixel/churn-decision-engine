"""
evaluator.py — Load saved models and produce all evaluation figures + metrics table.

All chart functions return Plotly figures for Streamlit embedding.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    DATA_OUTPUTS, FEATURES_FILE, MODEL_LOGISTIC, MODEL_RF, MODEL_XGB,
    RANDOM_STATE, TARGET_COLUMN,
)
from src.data.features import get_model_columns

RESULTS_FILE = DATA_OUTPUTS / "model_results.json"
SPLITS_FILE  = DATA_OUTPUTS / "test_indices.json"


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_models() -> dict:
    """Load only models whose .pkl files are present on disk."""
    candidates = {
        "Logistic Regression": MODEL_LOGISTIC,
        "Random Forest":       MODEL_RF,
        "XGBoost":             MODEL_XGB,
    }
    return {
        name: joblib.load(path)
        for name, path in candidates.items()
        if path.exists()
    }


def load_thresholds() -> dict:
    """Return tuned thresholds only for models that are actually loaded."""
    key_map = {
        "Logistic Regression": "logistic",
        "Random Forest":       "random_forest",
        "XGBoost":             "xgboost",
    }
    with open(RESULTS_FILE) as f:
        raw = json.load(f)
    return {
        display: raw[key]["threshold"]
        for display, key in key_map.items()
        if key in raw
    }


def load_test_set() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(FEATURES_FILE)
    feature_cols = get_model_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    with open(SPLITS_FILE) as f:
        idx = json.load(f)["test_indices"]
    return X.loc[idx], y.loc[idx]


# ── Figures ────────────────────────────────────────────────────────────────────

_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest":       "#ff7f0e",
    "XGBoost":             "#2ca02c",
}


def roc_curve_figure(
    models: dict, X_test: pd.DataFrame, y_test: pd.Series
) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="grey", width=1))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name}  AUC={auc:.3f}",
            line=dict(color=_COLORS[name], width=2.5)
        ))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.58, y=0.08, bgcolor="rgba(255,255,255,0.85)"),
        height=420,
    )
    return fig


def precision_recall_figure(
    models: dict, X_test: pd.DataFrame, y_test: pd.Series
) -> go.Figure:
    fig = go.Figure()
    baseline = y_test.mean()
    fig.add_hline(y=baseline, line_dash="dot", line_color="grey",
                  annotation_text=f"Random ({baseline:.1%})", annotation_position="right")
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines", name=f"{name}  AP={ap:.3f}",
            line=dict(color=_COLORS[name], width=2.5)
        ))
    fig.update_layout(
        title="Precision-Recall Curves — All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=420,
    )
    return fig


def confusion_matrix_figure(
    model, threshold: float,
    X_test: pd.DataFrame, y_test: pd.Series, name: str
) -> go.Figure:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Retained (0)", "Churned (1)"]
    fig = px.imshow(
        cm, x=labels, y=labels,
        color_continuous_scale="Blues",
        text_auto=True,
        title=f"Confusion Matrix — {name}  (threshold={threshold})",
        labels=dict(x="Predicted", y="Actual"),
        aspect="equal",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=380)
    return fig


def threshold_f1_figure(
    model, X_val: pd.DataFrame, y_val: pd.Series, name: str
) -> go.Figure:
    """F1(churn) vs threshold curve — shows where the optimal threshold is."""
    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.10, 0.71, 0.01)
    f1s = [f1_score(y_val, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=f1s, mode="lines",
        line=dict(color=_COLORS.get(name, "#333"), width=2)
    ))
    fig.add_vline(x=best_t, line_dash="dash", line_color="red",
                  annotation_text=f"Best t={best_t:.2f}", annotation_position="top right")
    fig.update_layout(
        title=f"F1(Churn) vs Classification Threshold — {name}",
        xaxis_title="Threshold", yaxis_title="F1 Score (churn class)",
        height=350,
    )
    return fig


# ── Metrics table ──────────────────────────────────────────────────────────────

def metrics_table(
    models: dict, thresholds: dict,
    X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        t      = thresholds[name]
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "Model":              name,
            "Threshold":          t,
            "ROC-AUC":            round(roc_auc_score(y_test, y_prob), 4),
            "Precision (Churn)":  round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall (Churn)":     round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 (Churn)":         round(f1_score(y_test, y_pred, zero_division=0), 4),
            "Accuracy":           round((y_pred == y_test).mean(), 4),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )
