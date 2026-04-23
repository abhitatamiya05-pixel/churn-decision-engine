"""
07_recommendations.py — Action playbook: full retention strategy by segment + KPIs.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.utils import load_scored, load_display, guard, fmt_pct, fmt_dollar
from config.settings import SEGMENT_LABELS, DATA_OUTPUTS, FEAT_IMP_FILE

st.set_page_config(page_title="Recommendations", layout="wide")
st.title("📋 Retention Recommendations")
st.caption(
    "Segment-specific action playbook grounded in churn risk, customer value, "
    "and the model's top drivers. Designed for retention team use."
)

scored  = load_scored()
display = load_display()
guard(scored, "Recommendations")

# ── Full playbook data ─────────────────────────────────────────────────────────
PLAYBOOK = {
    ("high_risk", "high_value"): {
        "why": (
            "These customers are your highest revenue contributors AND most likely to leave. "
            "A single churn event here costs ~$89/mo in perpetuity. "
            "This is where every dollar of retention spend has the highest ROI."
        ),
        "tactics": [
            "Assign a dedicated retention agent — personal phone outreach within 48 hours",
            "Offer a tailored contract upgrade (month-to-month → 1-year) with a loyalty discount",
            "Highlight unused services they're already paying adjacent products for",
            "If first contact fails, escalate to a manager with authority to offer deeper discount",
            "Log outcome in CRM and set 30-day follow-up regardless of result",
        ],
        "avoid": [
            "Don't send generic email — it signals low effort and can accelerate churn",
            "Don't offer a discount without understanding the churn reason first",
        ],
        "kpi": "Target: ≥ 30% save rate, < 48h response time, 100% contact attempt rate",
    },
    ("high_risk", "low_value"): {
        "why": (
            "High churn probability but lower monthly revenue ($50 avg). "
            "Worth saving at low intervention cost — automated offers are efficient here. "
            "Don't assign human agents unless automation fails."
        ),
        "tactics": [
            "Trigger automated SMS/email retention offer within 24h of model scoring",
            "Offer self-serve discount portal or a bundle promotion (add OnlineSecurity free for 3mo)",
            "Provide proactive tech support if on Fiber optic — Fiber has elevated churn linked to service issues",
            "A/B test offer types (price vs. service bundle) to improve conversion rate",
        ],
        "avoid": [
            "Don't assign human agents as first touch — cost exceeds expected return",
            "Don't offer discounts deeper than 15% — reduces margin without improving save rate",
        ],
        "kpi": "Target: automated offer conversion ≥ 15%, cost per save < $40",
    },
    ("low_risk", "high_value"): {
        "why": (
            "Stable, high-revenue customers ($93 avg monthly). "
            "They're not leaving now, but a contract expiry or bad experience could trigger churn. "
            "Proactive investment here protects a large, predictable revenue base."
        ),
        "tactics": [
            "Enrol in a loyalty or rewards programme (early renewal credit, usage perks)",
            "Offer proactive contract renewal 60 days before expiry at current pricing",
            "Pitch premium add-ons as an upsell — StreamingTV, DeviceProtection",
            "Quarterly satisfaction check-in call — signals value and surfaces issues early",
            "Celebrate tenure milestones (1yr, 2yr) with a personalised thank-you offer",
        ],
        "avoid": [
            "Don't ignore — a single competitor offer to this segment is expensive to counter",
            "Don't upsell aggressively — this segment values stability over features",
        ],
        "kpi": "Target: upsell conversion ≥ 10%, churn rate < 5%, NPS improvement",
    },
    ("low_risk", "low_value"): {
        "why": (
            "Low risk and low monthly revenue ($36 avg). "
            "Intervention cost likely exceeds expected save value in the short term. "
            "Monitor passively and re-score monthly."
        ),
        "tactics": [
            "No proactive outreach unless risk score rises above threshold",
            "Include in standard lifecycle email and SMS campaigns",
            "Monitor for contract-end proximity — flag for low-cost offer at renewal",
            "Use as control group for A/B testing retention campaign lift measurement",
        ],
        "avoid": [
            "Don't include in budget-constrained outreach — ROI is too low",
            "Don't ignore indefinitely — check monthly for risk tier migration",
        ],
        "kpi": "Monitor passively; re-score monthly; budget allocation priority = 4",
    },
}

# ── Render playbook cards ──────────────────────────────────────────────────────
ordered = sorted(SEGMENT_LABELS.items(), key=lambda x: x[1]["priority"])

for (key, info) in ordered:
    seg_df  = scored[scored["segment_label"] == info["label"]]
    n       = len(seg_df)
    rev     = seg_df["revenue_at_risk"].sum()
    details = PLAYBOOK.get(key, {})
    color   = info["color"]

    # Segment header card
    st.markdown(
        f"""<div style="border-left:6px solid {color};padding:12px 18px;
        border-radius:6px;background:#fafafa;margin-bottom:8px">
        <h3 style="color:{color};margin:0;font-size:20px">
            Priority {info['priority']} &nbsp;—&nbsp; {info['label']}
        </h3>
        <p style="color:#555;margin:4px 0 0;font-size:14px">
            <b>{n:,}</b> customers &nbsp;|&nbsp;
            <b>{fmt_dollar(rev)}/mo</b> revenue at risk &nbsp;|&nbsp;
            Avg P(churn): <b>{fmt_pct(seg_df['churn_probability'].mean())}</b>
        </p></div>""",
        unsafe_allow_html=True,
    )

    col_why, col_tactics = st.columns([1, 1])

    with col_why:
        st.markdown(f"**Why this segment matters**")
        st.markdown(details.get("why", ""))
        st.markdown(f"**Success KPI:**  \n{details.get('kpi', '')}")
        st.markdown("**What to avoid:**")
        for a in details.get("avoid", []):
            st.markdown(f"- ❌ {a}")

    with col_tactics:
        st.markdown("**Recommended tactics (in order):**")
        for i, tactic in enumerate(details.get("tactics", []), 1):
            st.markdown(f"{i}. {tactic}")

    # Mini metrics for segment
    with st.expander(f"View {info['label']} customers (top 20 by revenue at risk)"):
        seg_display = (
            display[display["segment_label"] == info["label"]]
            if not display.empty and "Contract" in display.columns
            else seg_df
        )
        cols_to_show = ["customerID", "churn_probability", "MonthlyCharges",
                        "revenue_at_risk", "tenure"]
        if "Contract" in seg_display.columns:
            cols_to_show += ["Contract", "InternetService"]
        avail = [c for c in cols_to_show if c in seg_display.columns]
        preview = seg_display[avail].nlargest(20, "revenue_at_risk").copy()
        preview["churn_probability"] = preview["churn_probability"].map(fmt_pct)
        preview["revenue_at_risk"]   = preview["revenue_at_risk"].map(lambda x: fmt_dollar(x, 2))
        preview["MonthlyCharges"]    = preview["MonthlyCharges"].map(lambda x: fmt_dollar(x, 2))
        st.dataframe(preview, use_container_width=True, hide_index=True)

    st.markdown("---")

# ── Key churn drivers ──────────────────────────────────────────────────────────
st.subheader("Root-Cause Playbook: Top Churn Drivers")
st.markdown(
    "These are the features with the highest SHAP impact. "
    "Each represents a lever the business can pull to reduce churn structurally."
)

driver_table = pd.DataFrame([
    {
        "Driver":         "Month-to-month contract",
        "Churn Rate":     "~43%",
        "vs Baseline":    "1.6× avg",
        "Root cause":     "No switching cost, flexible exit",
        "Structural fix": "Incentivise annual contract at onboarding (e.g., 1st month free)",
    },
    {
        "Driver":         "Short tenure (< 12 months)",
        "Churn Rate":     "~48%",
        "vs Baseline":    "1.8× avg",
        "Root cause":     "Unmet expectations post-sign-up",
        "Structural fix": "Improve onboarding: proactive check-ins at 30/60/90 days",
    },
    {
        "Driver":         "Fiber optic internet",
        "Churn Rate":     "~42%",
        "vs Baseline":    "1.6× avg",
        "Root cause":     "Service quality or price-to-value perception",
        "Structural fix": "Audit Fiber SLA; consider loyalty price lock for Fiber subscribers",
    },
    {
        "Driver":         "Electronic check payment",
        "Churn Rate":     "~45%",
        "vs Baseline":    "1.7× avg",
        "Root cause":     "Manual payers are less engaged; friction = churn signal",
        "Structural fix": "Convert to auto-pay at onboarding; offer one-time bill credit",
    },
    {
        "Driver":         "No OnlineSecurity add-on",
        "Churn Rate":     "Higher",
        "vs Baseline":    "Elevated",
        "Root cause":     "Lower perceived value of the subscription",
        "Structural fix": "Bundle OnlineSecurity free for 3 months as a retention hook",
    },
    {
        "Driver":         "High monthly charges without long contract",
        "Churn Rate":     "Higher",
        "vs Baseline":    "Elevated",
        "Root cause":     "High spend + no commitment = willingness to shop around",
        "Structural fix": "Proactive pricing review for high-charge month-to-month customers",
    },
])
st.dataframe(driver_table, use_container_width=True, hide_index=True)

# ── Feature importance summary bar ────────────────────────────────────────────
if FEAT_IMP_FILE.exists():
    st.markdown("---")
    st.subheader("Feature Importance Summary (SHAP)")
    imp_df = pd.read_csv(FEAT_IMP_FILE)
    shap_df = (
        imp_df[imp_df["model"] == "SHAP"]
        .nlargest(15, "importance")
        .sort_values("importance")
    )
    if not shap_df.empty:
        fig = px.bar(
            shap_df, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Purples",
            title="Top 15 SHAP Features — XGBoost",
            labels={"importance": "Mean |SHAP|", "feature": ""},
        )
        fig.update_layout(coloraxis_showscale=False, height=480)
        st.plotly_chart(fig, use_container_width=True)
