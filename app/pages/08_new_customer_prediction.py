"""
08_new_customer_prediction.py — Live churn prediction for a new customer.

Transforms form inputs through the same feature engineering as the training pipeline,
then runs the deployed scoring model to produce a churn probability, segment, and action.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.settings import (
    DATA_OUTPUTS, FEATURES_FILE, MODEL_LOGISTIC, MODEL_XGB, SEGMENT_LABELS,
)
from src.data.features import get_model_columns

RESULTS_FILE = DATA_OUTPUTS / "model_results.json"

st.set_page_config(page_title="New Customer Prediction", layout="wide")
st.title("🆕 New Customer Prediction")
st.caption(
    "Enter a new customer's details to get an instant churn probability, "
    "segment assignment, and recommended retention action."
)

# ── Load model and metadata ────────────────────────────────────────────────────
if not RESULTS_FILE.exists():
    st.error("model_results.json not found. Ensure committed artifacts are present.")
    st.stop()

with open(RESULTS_FILE) as f:
    results_json = json.load(f)

# Pick best available model by AUC
_MODEL_PATHS = {"logistic": MODEL_LOGISTIC, "xgboost": MODEL_XGB}
best_key  = max(
    (k for k in _MODEL_PATHS if _MODEL_PATHS[k].exists()),
    key=lambda k: results_json.get(k, {}).get("test_auc", 0),
)
model      = joblib.load(_MODEL_PATHS[best_key])
threshold  = results_json[best_key]["threshold"]
model_name = best_key.replace("_", " ").title()

# Load feature column list and training-set median for value-tier split
@st.cache_data
def _load_feature_meta():
    df = pd.read_csv(FEATURES_FILE)
    cols   = get_model_columns(df)
    median_charge = float(df["MonthlyCharges"].median())
    return cols, median_charge

feat_cols, median_charge = _load_feature_meta()

# ── Feature engineering for a single input row ────────────────────────────────

def build_feature_row(inputs: dict) -> pd.DataFrame:
    """
    Reproduce the same transformations as src/data/features.py for one customer.
    Returns a single-row DataFrame aligned to feat_cols.
    """
    i = inputs
    tenure = max(i["tenure"], 0)

    avg_monthly = i["TotalCharges"] / tenure if tenure > 0 else i["MonthlyCharges"]

    row = {
        # Passthrough numerics
        "SeniorCitizen":        int(i["SeniorCitizen"]),
        "tenure":               tenure,
        "MonthlyCharges":       i["MonthlyCharges"],
        "TotalCharges":         i["TotalCharges"],
        # Binary yes/no
        "Partner":              1 if i["Partner"] == "Yes" else 0,
        "Dependents":           1 if i["Dependents"] == "Yes" else 0,
        "PhoneService":         1 if i["PhoneService"] == "Yes" else 0,
        "MultipleLines":        1 if i["MultipleLines"] == "Yes" else 0,
        "OnlineSecurity":       1 if i["OnlineSecurity"] == "Yes" else 0,
        "OnlineBackup":         1 if i["OnlineBackup"] == "Yes" else 0,
        "DeviceProtection":     1 if i["DeviceProtection"] == "Yes" else 0,
        "TechSupport":          1 if i["TechSupport"] == "Yes" else 0,
        "StreamingTV":          1 if i["StreamingTV"] == "Yes" else 0,
        "StreamingMovies":      1 if i["StreamingMovies"] == "Yes" else 0,
        "PaperlessBilling":     1 if i["PaperlessBilling"] == "Yes" else 0,
        # Engineered
        "avg_monthly_revenue":  avg_monthly,
        "charges_delta":        i["MonthlyCharges"] - avg_monthly,
        "clv_estimate":         i["MonthlyCharges"] * 24,
        "service_count":        sum([
            i["PhoneService"] == "Yes", i["MultipleLines"] == "Yes",
            i["OnlineSecurity"] == "Yes", i["OnlineBackup"] == "Yes",
            i["DeviceProtection"] == "Yes", i["TechSupport"] == "Yes",
            i["StreamingTV"] == "Yes", i["StreamingMovies"] == "Yes",
        ]),
        "is_autopay":           1 if "automatic" in i["PaymentMethod"].lower() else 0,
        "is_longterm_contract": 1 if i["Contract"] in ("One year", "Two year") else 0,
        "has_internet":         1 if i["InternetService"] != "No" else 0,
        "is_fiber":             1 if i["InternetService"] == "Fiber optic" else 0,
        "is_senior":            int(i["SeniorCitizen"]),
        "is_high_value":        1 if i["MonthlyCharges"] >= median_charge else 0,
        "is_charges_outlier":   0,
        # One-hot: gender
        "gender_Female":        1 if i["gender"] == "Female" else 0,
        "gender_Male":          1 if i["gender"] == "Male" else 0,
        # One-hot: InternetService
        "InternetService_DSL":         1 if i["InternetService"] == "DSL" else 0,
        "InternetService_Fiber optic":  1 if i["InternetService"] == "Fiber optic" else 0,
        "InternetService_No":           1 if i["InternetService"] == "No" else 0,
        # One-hot: Contract
        "Contract_Month-to-month":  1 if i["Contract"] == "Month-to-month" else 0,
        "Contract_One year":        1 if i["Contract"] == "One year" else 0,
        "Contract_Two year":        1 if i["Contract"] == "Two year" else 0,
        # One-hot: PaymentMethod
        "PaymentMethod_Bank transfer (automatic)":  1 if i["PaymentMethod"] == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)":    1 if i["PaymentMethod"] == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check":           1 if i["PaymentMethod"] == "Electronic check" else 0,
        "PaymentMethod_Mailed check":               1 if i["PaymentMethod"] == "Mailed check" else 0,
    }

    df_row = pd.DataFrame([row])
    # Align to training columns — fill any gap with 0
    for col in feat_cols:
        if col not in df_row.columns:
            df_row[col] = 0
    return df_row[feat_cols]


def get_segment(prob: float, monthly_charge: float, thresh: float) -> dict:
    risk  = "high_risk"  if prob >= thresh      else "low_risk"
    value = "high_value" if monthly_charge >= median_charge else "low_value"
    return SEGMENT_LABELS[(risk, value)]


# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("customer_form"):
    st.markdown("### Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender        = st.selectbox("Gender",           ["Male", "Female"])
        senior        = st.selectbox("Senior Citizen",   ["No", "Yes"])
        partner       = st.selectbox("Partner",          ["No", "Yes"])
        dependents    = st.selectbox("Dependents",       ["No", "Yes"])

        st.markdown("**Account**")
        tenure        = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        contract      = st.selectbox("Contract",         ["Month-to-month", "One year", "Two year"])
        paperless     = st.selectbox("Paperless Billing",["No", "Yes"])
        payment       = st.selectbox("Payment Method",   [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])

    with col2:
        st.markdown("**Services**")
        phone         = st.selectbox("Phone Service",    ["No", "Yes"])
        multiline     = st.selectbox("Multiple Lines",   ["No", "Yes"])
        internet      = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        security      = st.selectbox("Online Security",  ["No", "Yes"])
        backup        = st.selectbox("Online Backup",    ["No", "Yes"])
        device        = st.selectbox("Device Protection",["No", "Yes"])
        tech          = st.selectbox("Tech Support",     ["No", "Yes"])
        tv            = st.selectbox("Streaming TV",     ["No", "Yes"])
        movies        = st.selectbox("Streaming Movies", ["No", "Yes"])

    with col3:
        st.markdown("**Charges**")
        monthly       = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
        total         = st.number_input("Total Charges ($)",   min_value=0.0, max_value=10000.0,
                                        value=float(monthly * max(tenure, 1)), step=1.0)
        st.caption(
            f"ℹ️ Median monthly charge in training data: **${median_charge:.2f}**  \n"
            f"Customers above this are classified as high-value."
        )
        st.markdown("")
        st.markdown(f"**Model:** {model_name}  \n**Threshold:** {threshold}")

    submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    inputs = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiline,
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": device,
        "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }

    X_input = build_feature_row(inputs)
    prob    = float(model.predict_proba(X_input)[0, 1])
    seg     = get_segment(prob, monthly, threshold)
    rev_at_risk = prob * monthly

    st.markdown("---")
    st.subheader("Prediction Result")

    # Risk gauge
    gauge_col, result_col = st.columns([1, 2])
    with gauge_col:
        color = "#d62728" if prob >= threshold else "#2ca02c"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 38}},
            title={"text": "Churn Probability", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color, "thickness": 0.25},
                "steps": [
                    {"range": [0,  40], "color": "#d4f1d4"},
                    {"range": [40, 65], "color": "#fff3cd"},
                    {"range": [65,100], "color": "#ffcdd2"},
                ],
                "threshold": {"line": {"color": "#333", "width": 3},
                              "thickness": 0.75, "value": threshold * 100},
            },
        ))
        fig.update_layout(height=260, margin=dict(t=50, b=0, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with result_col:
        predicted_label = "⚠️ Likely to Churn" if prob >= threshold else "✅ Likely to Stay"
        st.markdown(f"### {predicted_label}")

        seg_color = seg["color"]
        st.markdown(
            f"<span style='background:{seg_color};color:white;padding:5px 14px;"
            f"border-radius:4px;font-weight:bold'>Priority {seg['priority']} — {seg['label']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability",     f"{prob:.1%}")
        m2.metric("Monthly Rev at Risk",   f"${rev_at_risk:.2f}")
        m3.metric("Customer Value Tier",   "High" if monthly >= median_charge else "Low")

        st.markdown(
            f"<div style='margin-top:12px;background:#f8f8f8;"
            f"border-left:5px solid {seg_color};padding:10px 14px;border-radius:4px'>"
            f"<b>Recommended Action</b><br>{seg['action']}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Key risk factors for this customer
    st.markdown("---")
    st.subheader("Risk Factor Summary")
    flags = {
        "Month-to-month contract":    contract == "Month-to-month",
        "Short tenure (< 12 months)": tenure < 12,
        "Fiber optic internet":        internet == "Fiber optic",
        "Electronic check payment":    payment == "Electronic check",
        "No online security":          security == "No" and internet != "No",
        "No tech support":             tech == "No" and internet != "No",
        "High monthly charges":        monthly >= median_charge,
        "Not on auto-pay":             "automatic" not in payment.lower(),
    }
    active   = [k for k, v in flags.items() if v]
    inactive = [k for k, v in flags.items() if not v]

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**🔴 Risk factors present**")
        if active:
            for f in active:
                st.markdown(f"- {f}")
        else:
            st.markdown("- None identified")
    with r2:
        st.markdown("**🟢 Protective factors**")
        if inactive:
            for f in inactive[:5]:
                st.markdown(f"- No {f.lower()}")
        else:
            st.markdown("- None identified")

    st.caption(
        f"Model: {model_name} · Threshold: {threshold} · "
        f"Training-set median charge: ${median_charge:.2f}"
    )
