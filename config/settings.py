"""
Central configuration for the Churn Decision Engine.
All thresholds, paths, and constants live here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
DATA_OUTPUTS    = ROOT / "data" / "outputs"

RAW_FILE        = DATA_RAW       / "telco_churn.csv"
FEATURES_FILE   = DATA_PROCESSED / "features.csv"
SCORED_FILE     = DATA_PROCESSED / "scored_customers.csv"

MODEL_LOGISTIC  = DATA_OUTPUTS / "model_logistic.pkl"
MODEL_RF        = DATA_OUTPUTS / "model_rf.pkl"
MODEL_XGB       = DATA_OUTPUTS / "model_xgb.pkl"
FEAT_IMP_FILE   = DATA_OUTPUTS / "feature_importance.csv"

# ── Modelling ──────────────────────────────────────────────────────────────────
TARGET_COLUMN   = "Churn"
ID_COLUMN       = "customerID"
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
CV_FOLDS        = 5

# Probability threshold for classifying a customer as high-risk
CHURN_THRESHOLD = 0.50

# ── Decision engine ────────────────────────────────────────────────────────────
# Revenue percentile above which a customer is considered "high value"
HIGH_VALUE_PERCENTILE = 0.50          # top 50% by MonthlyCharges

# Assumed one-time cost to intervene on a single customer (e.g., discount + agent time)
DEFAULT_INTERVENTION_COST = 50        # USD

# Assumed probability that a targeted customer is actually retained
DEFAULT_SAVE_RATE = 0.30              # 30%

# Default budget for the retention simulator slider
DEFAULT_BUDGET = 5_000               # USD

# ── Segmentation labels ────────────────────────────────────────────────────────
SEGMENT_LABELS = {
    ("high_risk", "high_value"): {
        "label": "Save Immediately",
        "priority": 1,
        "action": "Personal outreach + tailored discount offer",
        "color": "#d62728",
    },
    ("high_risk", "low_value"): {
        "label": "Assess & Offer",
        "priority": 2,
        "action": "Self-serve retention offer or automated email",
        "color": "#ff7f0e",
    },
    ("low_risk", "high_value"): {
        "label": "Nurture & Upsell",
        "priority": 3,
        "action": "Loyalty program or upgrade promotion",
        "color": "#2ca02c",
    },
    ("low_risk", "low_value"): {
        "label": "Monitor Only",
        "priority": 4,
        "action": "No immediate action — standard lifecycle comms",
        "color": "#7f7f7f",
    },
}

# ── Tenure bands ───────────────────────────────────────────────────────────────
TENURE_BINS   = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]

# ── App display ────────────────────────────────────────────────────────────────
APP_TITLE    = "Churn Decision Engine"
APP_ICON     = "📉"
BRAND_COLOR  = "#1f77b4"
