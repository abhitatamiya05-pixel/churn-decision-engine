"""
test_pipeline.py — Sanity checks for data pipeline, models, and decision engine.

Run:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (
    DATA_OUTPUTS, FEATURES_FILE, MODEL_LOGISTIC, MODEL_RF, MODEL_XGB,
    SCORED_FILE, TARGET_COLUMN,
)
from src.data.loader import load_raw
from src.data.cleaner import clean
from src.data.features import engineer, get_model_columns

RESULTS_FILE = DATA_OUTPUTS / "model_results.json"
FEAT_IMP_FILE = DATA_OUTPUTS / "feature_importance.csv"


# ── Data pipeline ─────────────────────────────────────────────────────────────

class TestCleaner:
    def test_total_charges_no_nulls(self):
        raw = load_raw()
        cleaned = clean(raw)
        assert cleaned["TotalCharges"].isna().sum() == 0

    def test_churn_is_binary(self):
        raw = load_raw()
        cleaned = clean(raw)
        assert set(cleaned[TARGET_COLUMN].unique()).issubset({0, 1})

    def test_row_count_preserved(self):
        raw = load_raw()
        assert len(clean(raw)) == len(raw)


class TestFeatures:
    @pytest.fixture(scope="class")
    def featured(self):
        return engineer(clean(load_raw()))

    def test_engineered_columns_exist(self, featured):
        for col in ["tenure_band", "service_count", "is_autopay",
                    "clv_estimate", "is_high_value"]:
            assert col in featured.columns, f"Missing: {col}"

    def test_no_negative_clv(self, featured):
        assert (featured["clv_estimate"] >= 0).all()

    def test_model_columns_are_numeric(self, featured):
        for col in get_model_columns(featured):
            assert featured[col].dtype.kind in ("f", "i", "u"), \
                f"{col} is not numeric (dtype={featured[col].dtype})"

    def test_no_missing_values(self, featured):
        nulls = featured[get_model_columns(featured)].isnull().sum()
        bad = nulls[nulls > 0]
        assert len(bad) == 0, f"Unexpected nulls: {bad.to_dict()}"

    def test_tenure_band_covers_all_rows(self, featured):
        assert featured["tenure_band"].notna().all()


# ── Models ────────────────────────────────────────────────────────────────────

def _models_available():
    return MODEL_XGB.exists() and MODEL_RF.exists() and MODEL_LOGISTIC.exists()


def _results_available():
    return RESULTS_FILE.exists()


@pytest.mark.skipif(not _models_available(), reason="model artifacts not found — run trainer.py")
class TestModels:
    @pytest.fixture(scope="class")
    def models_and_data(self):
        import joblib
        from src.models.evaluator import load_test_set
        models = {
            "logistic":      joblib.load(MODEL_LOGISTIC),
            "random_forest": joblib.load(MODEL_RF),
            "xgboost":       joblib.load(MODEL_XGB),
        }
        X_test, y_test = load_test_set()
        return models, X_test, y_test

    def test_model_artifacts_exist(self):
        for path in [MODEL_LOGISTIC, MODEL_RF, MODEL_XGB]:
            assert path.exists(), f"Missing: {path}"

    @pytest.mark.skipif(not _results_available(), reason="model_results.json not found")
    def test_results_json_structure(self):
        with open(RESULTS_FILE) as f:
            r = json.load(f)
        for model_key in ["logistic", "random_forest", "xgboost"]:
            assert model_key in r
            for field in ["test_auc", "test_f1", "test_recall", "threshold"]:
                assert field in r[model_key], f"Missing '{field}' for {model_key}"

    @pytest.mark.skipif(not _results_available(), reason="model_results.json not found")
    def test_all_models_auc_above_baseline(self):
        with open(RESULTS_FILE) as f:
            r = json.load(f)
        for name, m in r.items():
            assert m["test_auc"] > 0.70, \
                f"{name} AUC={m['test_auc']:.4f} below minimum 0.70"

    def test_predict_proba_shape_and_range(self, models_and_data):
        models, X_test, _ = models_and_data
        for name, model in models.items():
            proba = model.predict_proba(X_test)
            assert proba.shape == (len(X_test), 2), \
                f"{name}: unexpected output shape {proba.shape}"
            assert proba.min() >= 0 and proba.max() <= 1, \
                f"{name}: probabilities outside [0, 1]"

    @pytest.mark.skipif(not _results_available(), reason="model_results.json not found")
    def test_best_model_recall_above_minimum(self):
        with open(RESULTS_FILE) as f:
            r = json.load(f)
        best_recall = max(m["test_recall"] for m in r.values())
        assert best_recall >= 0.55, \
            f"Best recall {best_recall:.4f} below minimum 0.55"

    def test_logistic_uses_scaler(self, models_and_data):
        import joblib
        from sklearn.pipeline import Pipeline
        model = joblib.load(MODEL_LOGISTIC)
        assert isinstance(model, Pipeline), "Logistic model should be wrapped in a Pipeline"
        assert "scaler" in model.named_steps

    def test_feature_importance_saved(self):
        assert FEAT_IMP_FILE.exists(), "feature_importance.csv not found"
        df = pd.read_csv(FEAT_IMP_FILE)
        assert len(df) >= 10
        assert "feature" in df.columns and "importance" in df.columns


# ── Decision engine ───────────────────────────────────────────────────────────

def _scored_available():
    return SCORED_FILE.exists()


@pytest.mark.skipif(not _scored_available(), reason="scored_customers.csv not found")
class TestDecisionEngine:
    @pytest.fixture(scope="class")
    def scored(self):
        return pd.read_csv(SCORED_FILE)

    def test_probabilities_in_range(self, scored):
        assert scored["churn_probability"].between(0, 1).all()

    def test_risk_rank_is_unique(self, scored):
        assert scored["risk_rank"].nunique() == len(scored)

    def test_segment_labels_valid(self, scored):
        valid = {"Save Immediately", "Assess & Offer", "Nurture & Upsell", "Monitor Only"}
        assert set(scored["segment_label"].unique()).issubset(valid)

    def test_all_customers_have_action(self, scored):
        assert scored["recommended_action"].notna().all()

    def test_revenue_at_risk_non_negative(self, scored):
        assert (scored["revenue_at_risk"] >= 0).all()

    def test_budget_optimizer_output(self, scored):
        from src.decision.budget_optimizer import allocate_budget
        result = allocate_budget(scored, budget=5_000)
        assert result["n_targeted"] == 100   # $5000 / $50 = 100
        assert result["expected_revenue_saved"] > 0
        assert result["roi_multiple"] > 0

    def test_save_immediately_is_highest_risk(self, scored):
        si = scored[scored["segment_label"] == "Save Immediately"]
        mo = scored[scored["segment_label"] == "Monitor Only"]
        assert si["churn_probability"].mean() > mo["churn_probability"].mean()
