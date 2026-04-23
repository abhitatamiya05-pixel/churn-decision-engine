#!/bin/bash
set -e

echo "=== Churn Decision Engine ==="

# 1. Install dependencies
pip install -r requirements.txt --quiet

# 2. Run data pipeline
echo "[1/3] Running data pipeline..."
python -m src.data.loader
python -m src.data.cleaner
python -m src.data.features

# 3. Train models
echo "[2/3] Training models..."
python -m src.models.trainer

# 4. Score customers and build decision engine
echo "[3/3] Running decision engine..."
python -m src.decision.scorer
python -m src.decision.segmenter
python -m src.decision.budget_optimizer

# 5. Launch app
echo "Launching dashboard..."
streamlit run app/main.py
