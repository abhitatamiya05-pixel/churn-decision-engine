# 📉 Churn Decision Engine

> An end-to-end customer retention analytics platform that identifies at-risk customers, segments them by revenue impact, recommends targeted actions, and allocates a fixed budget to maximise expected revenue saved.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/dataset-IBM%20Telco%20Churn-lightgrey)](https://github.com/IBM/telco-customer-churn-on-icp4d)
[![Tests](https://img.shields.io/badge/tests-22%20passing-brightgreen)](#running-tests)

---

## The Business Problem

Customer churn is a direct, compounding revenue leak. For subscription businesses, even a 2–3% monthly churn rate compounds into significant annual losses — yet most teams know *that* customers leave without knowing **which** to save, **in what order**, or **within what budget**.

This project answers six concrete business questions:

| # | Question |
|---|----------|
| 1 | Which customers are most likely to churn? |
| 2 | Which segments are driving the highest churn? |
| 3 | Which customers are most important to save by revenue? |
| 4 | What retention action fits each customer segment? |
| 5 | On a limited budget, who should we target first? |
| 6 | What is the estimated business impact of acting vs. doing nothing? |

---

## Live Dashboard — 7 Pages

| Page | Purpose |
|------|---------|
| **📊 Executive Summary** | Top-line KPIs, segment snapshot, do-nothing vs. act comparison, top-15 priority contacts |
| **📉 Churn Overview** | Churn rates by contract, tenure, charges, services — cohort matrix and revenue heatmaps |
| **🗺️ Segment Explorer** | Interactive 2×2 risk-value matrix, per-segment metrics, drill-down, CSV export |
| **🔍 Customer Lookup** | Per-customer risk gauge, profile tabs, action card, similar-customer finder |
| **💰 Retention Simulator** | Budget/cost/save-rate sliders, impact curve, ROI by scenario, downloadable outreach list |
| **🤖 Model Performance** | ROC + PR curves, confusion matrices, SHAP + RF + LR importances, business interpretation |
| **📋 Recommendations** | Segment-specific action playbook with tactics, what-to-avoid, success KPIs, root-cause table |

---

## Dataset

**IBM Telco Customer Churn** — 7,043 customers, 21 features. Downloaded automatically on first run.

Chosen because:
- Industry gold standard for churn analytics — widely recognised in data/analytics hiring
- Contains both a binary churn label and revenue data (monthly + total charges)
- Rich mix of contract type, services, demographics, and payment method
- Clean enough to focus on business logic; realistic enough to demonstrate data prep skill

Source: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)

---

## Architecture

```
churn-decision-engine/
│
├── app/                        # Streamlit dashboard (7 pages)
│   ├── main.py                 # Entry point + sidebar with live KPIs
│   ├── utils.py                # Shared data loaders and formatters
│   └── pages/                  # One file per dashboard page
│
├── src/
│   ├── data/                   # Ingestion, cleaning, feature engineering
│   │   ├── loader.py           # Downloads dataset if not present
│   │   ├── cleaner.py          # Type fixes, imputation, outlier flags
│   │   └── features.py         # 36 engineered ML features
│   ├── analysis/               # EDA and cohort analysis (Plotly figures)
│   │   ├── eda.py              # Churn rate, revenue, segment charts
│   │   └── cohorts.py          # Lifecycle, tenure-contract cohort matrix
│   ├── models/                 # Training, evaluation, explainability
│   │   ├── trainer.py          # 3-way split, threshold tuning, 5-fold CV
│   │   ├── evaluator.py        # ROC, PR, confusion matrix, metrics table
│   │   ├── explainer.py        # SHAP + native importance + LR coefficients
│   │   └── run_pipeline.py     # Orchestrator: train → evaluate → score → segment
│   └── decision/               # The engine that makes predictions operational
│       ├── scorer.py           # Score all customers; pick best model by AUC
│       ├── segmenter.py        # 2×2 risk-value segmentation + action labels
│       └── budget_optimizer.py # ROI-ranked allocation within a fixed budget
│
├── data/
│   ├── raw/                    # telco_churn.csv (auto-downloaded)
│   ├── processed/              # features.csv, cleaned.csv, scored_customers.csv
│   └── outputs/                # model_*.pkl, model_results.json, feature_importance.csv
│
├── sql/
│   └── analytics_queries.sql   # 12 production-quality analytical queries
│
├── tests/
│   └── test_pipeline.py        # 22 Pytest checks (data → models → decision engine)
│
├── config/
│   └── settings.py             # All thresholds, paths, constants — single source of truth
│
├── requirements.txt
├── run.sh                      # One-command end-to-end launch
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-username/churn-decision-engine.git
cd churn-decision-engine

# 2. Create environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run full pipeline + launch dashboard (one command)
bash run.sh
```

**Or step by step:**

```bash
# Data pipeline
python -m src.data.loader
python -m src.data.cleaner
python -m src.data.features

# Modeling + decision engine (all-in-one)
python -m src.models.run_pipeline

# Launch dashboard
streamlit run app/main.py
```

---

## Modeling

Three models trained on a **stratified 70/15/15 split** with class-imbalance handling and **per-model threshold tuning on the validation set** (F1-maximising — the default 0.5 threshold is sub-optimal for imbalanced data).

| Model | CV-AUC | Holdout AUC | Precision | Recall | F1 | Threshold |
|-------|--------|-------------|-----------|--------|----|-----------|
| Logistic Regression | 0.844±0.012 | **0.844** | 0.539 | 0.747 | 0.626 | 0.54 |
| Random Forest | 0.846±0.010 | 0.839 | **0.577** | 0.694 | **0.630** | 0.54 |
| XGBoost | 0.839±0.015 | 0.833 | 0.526 | **0.758** | 0.621 | 0.48 |

**Best model** (selected by holdout AUC) is used for scoring. SHAP values explain individual predictions.

---

## Decision Engine

### 2×2 Segmentation Matrix

```
                    HIGH VALUE ($93 avg)    LOW VALUE ($50 avg)
                 ┌──────────────────────┬───────────────────────┐
  HIGH RISK      │   Save Immediately   │    Assess & Offer     │
  (P ≥ thresh)   │   1,898 customers    │    817 customers      │
                 │   $129K rev at risk  │   $29K rev at risk    │
                 │   Personal outreach  │   Automated offer     │
                 ├──────────────────────┼───────────────────────┤
  LOW RISK       │   Nurture & Upsell   │    Monitor Only       │
  (P < thresh)   │   1,626 customers    │    2,702 customers    │
                 │   $36K rev at risk   │   $19K rev at risk    │
                 │   Loyalty / upsell   │   No action needed    │
                 └──────────────────────┴───────────────────────┘
```

### Budget Optimizer

```
expected_save_value = MonthlyCharges × P(churn) × save_rate × lifetime_months
roi_score           = expected_save_value / intervention_cost
→ Sort by ROI score descending
→ Select customers greedily until budget exhausted
→ Report: n targeted, revenue saved, vs. do-nothing, ROI multiple
```

**Budget ROI (12-month, 30% save rate, $50/intervention):**

| Budget | Customers | Est. Revenue Saved | ROI |
|--------|-----------|--------------------|-----|
| $2,500 | 50 | $16,400/yr | 6.6× |
| $5,000 | 100 | $32,200/yr | 6.4× |
| $10,000 | 200 | $62,700/yr | 6.3× |
| $25,000 | 500 | $148,400/yr | 5.9× |

---

## Key Findings (IBM Telco)

| Finding | Churn Rate | Implication |
|---------|-----------|-------------|
| Month-to-month contracts | **42.7%** | Strongest single predictor — offer contract upgrades |
| Fiber optic internet | **41.9%** | Possible pricing or quality issue — investigate |
| Electronic check payment | **45.3%** | Manual payers are disengaged — promote auto-pay |
| Tenure 0–12 months | **47.7%** | New-customer experience is the highest-leverage fix |
| Senior citizens | **41.7%** | Distinct high-risk demographic — separate strategy |
| Two-year contracts | **2.8%** | Retention benchmark — incentivise long-term commitment |

Top SHAP drivers: `is_longterm_contract` → `tenure` → `avg_monthly_revenue` → `MonthlyCharges` → `is_fiber`

---

## Configuration

All thresholds and constants are in [`config/settings.py`](config/settings.py):

```python
CHURN_THRESHOLD          = 0.50    # overridden by per-model tuning
HIGH_VALUE_PERCENTILE    = 0.50    # top 50% by MonthlyCharges = high value
DEFAULT_INTERVENTION_COST = 50     # USD per customer contacted
DEFAULT_SAVE_RATE         = 0.30   # 30% of contacted customers retained
DEFAULT_BUDGET            = 5_000  # USD — starting value for simulator
```

---

## Running Tests

```bash
pytest tests/ -v
# 22 passed — covers data pipeline, model outputs, and decision engine logic
```

---

## Deployment

### Streamlit Community Cloud (free, recommended for portfolio)

1. Push to a **public** GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file path**: `app/main.py`
4. Before pushing, run the full pipeline locally and **commit** these files:
   - `data/processed/cleaned.csv`
   - `data/processed/features.csv`
   - `data/processed/scored_customers.csv`
   - `data/outputs/model_results.json`
   - `data/outputs/test_indices.json`
   - `data/outputs/feature_importance.csv`
   - `data/outputs/model_logistic.pkl`
   - `data/outputs/model_xgb.pkl`
5. The 11MB `model_rf.pkl` exceeds Streamlit Cloud's recommended size — exclude it and ensure `model_results.json` points to logistic or xgb as best model

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m src.data.loader && \
    python -m src.data.cleaner && \
    python -m src.data.features && \
    python -m src.models.run_pipeline
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Local with venv (recommended for interviews/demos)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run.sh
# Dashboard at http://localhost:8501
```

---

## Resume Summary

Built an end-to-end customer churn analytics platform combining predictive modelling (Logistic Regression, Random Forest, XGBoost), SHAP explainability, 2×2 customer segmentation, and a budget-constrained ROI-ranked retention optimizer. Delivered a 7-page Streamlit dashboard enabling retention teams to identify at-risk customers, understand churn drivers, and prioritise outreach by expected revenue impact — all within an interactive budget constraint.

### Resume Bullets

- **Engineered a churn decision engine** on 7,043 telecom customers using a stratified train/val/test pipeline, threshold-tuned XGBoost and Logistic Regression (AUC 0.844), and SHAP explainability — translating model output into a ranked retention list that maximises annual revenue saved within a configurable budget constraint
- **Built a 7-page interactive Streamlit dashboard** integrating predictive risk scoring, 2×2 customer segmentation (risk × revenue value), cohort analysis, and a budget simulator with ROI projections — designed for non-technical retention team use with CSV export and per-customer action cards
- **Delivered a full analytics-to-action pipeline** (data ingestion → 36-feature engineering → multi-model training → decision engine → deployment) with modular Python architecture, 12 SQL analytical queries, 22 Pytest checks, and one-command reproducibility via `bash run.sh`

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data pipeline | Python · Pandas · NumPy |
| Modelling | scikit-learn · XGBoost · SHAP |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| SQL analytics | SQLite / DuckDB / BigQuery compatible |
| Testing | Pytest |
| Deployment | Streamlit Cloud · Docker |

---

*Dataset: IBM Telco Customer Churn &nbsp;|&nbsp; Built with Python, scikit-learn, XGBoost, SHAP, Streamlit, Plotly*
