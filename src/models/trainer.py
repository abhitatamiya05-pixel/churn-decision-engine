"""
trainer.py — Train Logistic Regression, Random Forest, and XGBoost with:
  - Stratified 3-way split (train 70 / val 15 / test 15)
  - Class-imbalance handling per model
  - Threshold tuning on the validation set (maximise F1-churn)
  - 5-fold stratified CV on train set
  - Saves all model artifacts + model_results.json

Run standalone:  python -m src.models.trainer
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    CV_FOLDS, DATA_OUTPUTS, FEATURES_FILE,
    MODEL_LOGISTIC, MODEL_RF, MODEL_XGB,
    RANDOM_STATE, TARGET_COLUMN, TEST_SIZE,
)
from src.data.features import get_model_columns

RESULTS_FILE = DATA_OUTPUTS / "model_results.json"
SPLITS_FILE  = DATA_OUTPUTS / "test_indices.json"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_features() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Run features.py first. Expected: {FEATURES_FILE}")
    df = pd.read_csv(FEATURES_FILE)
    feature_cols = get_model_columns(df)
    return df[feature_cols], df[TARGET_COLUMN], feature_cols


# ── Split ──────────────────────────────────────────────────────────────────────

def three_way_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Stratified 70 / 15 / 15 split.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Model definitions ──────────────────────────────────────────────────────────

def build_models(n_neg: int, n_pos: int) -> dict:
    scale_pos = round(n_neg / n_pos, 2)
    return {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=0.5,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


# ── Threshold tuning ───────────────────────────────────────────────────────────

def tune_threshold(
    model, X_val: pd.DataFrame, y_val: pd.Series
) -> tuple[float, float]:
    """
    Search thresholds [0.10, 0.70] and return the one that maximises
    F1 on the positive (churn) class. Also returns that best F1.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.71, 0.01):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, round(float(t), 2)
    return best_t, best_f1


# ── Training orchestrator ──────────────────────────────────────────────────────

def train_all(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
) -> dict:
    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    models = build_models(n_neg, n_pos)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print(f"\n[trainer] Train: {len(y_train):,}  Val: {len(y_val):,}")
    print(f"[trainer] Class balance — pos:{n_pos} neg:{n_neg} "
          f"ratio:{n_neg/n_pos:.2f}:1\n")

    for name, model in models.items():
        # 5-fold CV on training set
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        # Fit on full train split
        model.fit(X_train, y_train)

        # Tune threshold on val set
        threshold, val_f1 = tune_threshold(model, X_val, y_val)

        results[name] = {
            "model":     model,
            "threshold": threshold,
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std":  float(cv_scores.std()),
            "val_f1":    float(val_f1),
        }
        print(
            f"  {name:<15}  CV-AUC: {cv_scores.mean():.4f}±{cv_scores.std():.4f}"
            f"  |  Val-F1@{threshold}: {val_f1:.4f}"
        )

    # Persist models
    joblib.dump(results["logistic"]["model"],      MODEL_LOGISTIC)
    joblib.dump(results["random_forest"]["model"], MODEL_RF)
    joblib.dump(results["xgboost"]["model"],       MODEL_XGB)
    print(f"\n[trainer] Models saved to {DATA_OUTPUTS}")
    return results


# ── Save results metadata ──────────────────────────────────────────────────────

def save_results_json(results: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    output = {}
    for name, res in results.items():
        model     = res["model"]
        threshold = res["threshold"]
        y_prob    = model.predict_proba(X_test)[:, 1]
        y_pred    = (y_prob >= threshold).astype(int)
        output[name] = {
            "cv_auc_mean": res["cv_auc_mean"],
            "cv_auc_std":  res["cv_auc_std"],
            "val_f1":      res["val_f1"],
            "threshold":   threshold,
            "test_auc":    round(roc_auc_score(y_test, y_prob), 4),
            "test_precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "test_recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "test_f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[trainer] Results saved to {RESULTS_FILE}")


# ── Identify best model ────────────────────────────────────────────────────────

def best_model_name(results_json: dict) -> str:
    return max(results_json, key=lambda k: results_json[k]["test_f1"])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y, feature_cols = load_features()
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y)

    # Save test-set indices for reproducible evaluation
    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_FILE, "w") as f:
        json.dump({"test_indices": X_test.index.tolist()}, f)

    results = train_all(X_train, y_train, X_val, y_val)
    save_results_json(results, X_test, y_test)

    with open(RESULTS_FILE) as f:
        r = json.load(f)
    best = best_model_name(r)
    print(f"\n[trainer] Best model (by test F1): {best}  "
          f"AUC={r[best]['test_auc']}  F1={r[best]['test_f1']}")
