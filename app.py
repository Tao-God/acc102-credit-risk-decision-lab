import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Risk Decision Lab", page_icon="🏦", layout="wide")

st.markdown(
    """
<style>
.stApp {
  background: linear-gradient(140deg, #f4f8ff 0%, #edf3ff 48%, #f9fbff 100%);
}
[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.70);
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(90,120,255,0.18);
}
.block-container {
  padding-top: 2rem;
}
.glass-card {
  background: rgba(255,255,255,0.70);
  border: 1px solid rgba(114,140,255,0.22);
  box-shadow: 0 10px 30px rgba(76,96,175,0.14);
  border-radius: 16px;
  padding: 14px 16px;
  margin-bottom: 10px;
}
h1, h2, h3, h4, p, label, span {
  color: #1f2a44 !important;
}
.metric-box {
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(114,140,255,0.2);
  border-radius: 14px;
  padding: 8px 10px;
}
.stButton > button, .stDownloadButton > button {
  border-radius: 10px !important;
  border: 1px solid #4f6eff !important;
  background: linear-gradient(180deg, #6e87ff, #4f6eff) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  box-shadow: 0 8px 18px rgba(79,110,255,0.30);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  filter: brightness(1.05);
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Credit Risk Decision Lab")
st.caption("Interactive credit-risk scoring for new loan applications.")

MODELS_DIR = "models"
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
CLEAN_DATA_PATH = "data/processed/loan_clean.csv"

if not os.path.exists(CLEAN_DATA_PATH):
    st.error("Clean data not found. Please run: `python clean_data.py`")
    st.stop()

sample_df = pd.read_csv(CLEAN_DATA_PATH)
feature_cols = [c for c in sample_df.columns if c != "default_flag"]
app_feature_cols = [c for c in feature_cols if c != "loan_percent_income"]

if not os.path.exists(METRICS_PATH):
    st.warning("Model metrics not found. Please run: `python train_model.py`")
    st.stop()

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics_pack = json.load(f)

model_options = []
for name in ["logistic", "random_forest"]:
    p = os.path.join(MODELS_DIR, f"{name}_pipeline.joblib")
    if os.path.exists(p):
        model_options.append(name)

if not model_options:
    st.error("No model files found in /models. Run: `python train_model.py`")
    st.stop()

best_model_name = metrics_pack.get("best_model", model_options[0])
default_idx = model_options.index(best_model_name) if best_model_name in model_options else 0

with st.sidebar:
    st.subheader("Scoring Settings")
    selected_model_name = st.selectbox("Model", model_options, index=default_idx)
    default_threshold = float(metrics_pack.get("threshold_default", 0.35))
    threshold = st.slider("Approval threshold", min_value=0.10, max_value=0.80, value=default_threshold, step=0.01)
    st.caption("Lower threshold = stricter approval policy.")

model_path = os.path.join(MODELS_DIR, f"{selected_model_name}_pipeline.joblib")
model = joblib.load(model_path)

model_metrics = metrics_pack.get("models", {}).get(selected_model_name, {})
if model_metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-box'><b>ROC-AUC</b><br>{model_metrics.get('roc_auc', 0):.3f}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'><b>F1</b><br>{model_metrics.get('f1', 0):.3f}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'><b>Recall</b><br>{model_metrics.get('recall', 0):.3f}</div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-box'><b>Precision</b><br>{model_metrics.get('precision', 0):.3f}</div>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### New Applicant Test")
st.caption("Enter applicant information and click Predict Risk.")

preferred_fields = [
    "person_age",
    "person_income",
    "person_emp_length",
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]
use_fields = [c for c in preferred_fields if c in app_feature_cols]
if len(use_fields) < 6:
    use_fields = app_feature_cols[:11]

field_labels = {
    "person_age": "Age",
    "person_income": "Annual Income",
    "person_emp_length": "Employment Length (years)",
    "person_home_ownership": "Home Ownership",
    "loan_intent": "Loan Intent",
    "loan_grade": "Loan Grade",
    "loan_amnt": "Loan Amount",
    "loan_int_rate": "Interest Rate (%)",
    "loan_percent_income": "Loan/Income Ratio",
    "cb_person_default_on_file": "Historical Default On File",
    "cb_person_cred_hist_length": "Credit History Length (years)",
}

value_help = {
    "person_home_ownership": "OWN: own house, RENT: rent, MORTGAGE: mortgage",
    "cb_person_default_on_file": "Y: has previous default record, N: no previous default",
}

input_row = {}
for col in use_fields:
    label = field_labels.get(col, col)
    if pd.api.types.is_numeric_dtype(sample_df[col]):
        default_val = float(sample_df[col].median()) if sample_df[col].notna().any() else 0.0
        input_row[col] = st.number_input(label, value=default_val)
    else:
        options = sample_df[col].fillna("Unknown").astype(str).value_counts().head(25).index.tolist()
        input_row[col] = st.selectbox(label, options=options or ["Unknown"], index=0, help=value_help.get(col))

def build_full_row(partial_row: dict) -> pd.DataFrame:
    full_row = {}
    for c in feature_cols:
        if c in partial_row:
            full_row[c] = partial_row[c]
        else:
            if pd.api.types.is_numeric_dtype(sample_df[c]):
                full_row[c] = float(sample_df[c].median()) if sample_df[c].notna().any() else 0.0
            else:
                full_row[c] = sample_df[c].fillna("Unknown").astype(str).mode().iloc[0]
    return pd.DataFrame([full_row])


def driver_snapshot(partial_row: dict, top_n: int = 3):
    drivers = []
    for c, v in partial_row.items():
        if c in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df[c]):
            med = sample_df[c].median()
            iqr = sample_df[c].quantile(0.75) - sample_df[c].quantile(0.25)
            iqr = iqr if iqr and iqr > 0 else 1.0
            score = abs((float(v) - float(med)) / float(iqr))
            drivers.append((c, score, float(v), float(med)))
    drivers = sorted(drivers, key=lambda x: x[1], reverse=True)[:top_n]
    return drivers

def feature_compare_df(partial_row: dict, top_n: int = 5) -> pd.DataFrame:
    rows = []
    for c, v in partial_row.items():
        if c in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df[c]):
            med = float(sample_df[c].median()) if sample_df[c].notna().any() else 0.0
            rows.append(
                {
                    "feature": c,
                    "input_value": float(v),
                    "dataset_median": med,
                    "gap": float(v) - med,
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["abs_gap"] = out["gap"].abs()
    return out.sort_values("abs_gap", ascending=False).head(top_n)

if st.button("Predict Risk", type="primary"):
    x_new = build_full_row(input_row)
    pd_default = float(model.predict_proba(x_new)[:, 1][0])

    low_cut = threshold * 0.7
    if pd_default <= low_cut:
        decision = "APPROVE"
        color = "green"
    elif pd_default <= threshold:
        decision = "MANUAL REVIEW"
        color = "orange"
    else:
        decision = "REJECT / HIGH RISK"
        color = "red"

    st.markdown(f"### Predicted Default Probability: **{pd_default:.2%}**")
    st.markdown(f"### Recommendation: :{color}[**{decision}**]")
    st.markdown("#### Risk Level Gauge")
    st.progress(min(max(pd_default, 0.0), 1.0), text=f"Current default risk: {pd_default:.2%} (threshold: {threshold:.0%})")

    compare_df = feature_compare_df(input_row, top_n=5)
    if not compare_df.empty:
        st.markdown("#### Input vs Dataset Median (Top Numeric Differences)")
        st.bar_chart(compare_df.set_index("feature")[["input_value", "dataset_median"]], use_container_width=True)

    drivers = driver_snapshot(input_row, top_n=3)
    if drivers:
        st.markdown("#### Driver Snapshot (distance from dataset median)")
        for col, score, val, med in drivers:
            st.write(f"- `{col}`: input={val:.2f}, median={med:.2f}, relative_distance={score:.2f}")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### Batch Scoring (CSV)")
uploaded = st.file_uploader("Upload CSV for batch scoring", type=["csv"])
if uploaded is not None:
    batch = pd.read_csv(uploaded)
    if (
        "loan_percent_income" in feature_cols
        and "loan_percent_income" not in batch.columns
        and "loan_amnt" in batch.columns
        and "person_income" in batch.columns
    ):
        safe_income = batch["person_income"].replace(0, np.nan)
        fallback = float(sample_df["loan_percent_income"].median()) if "loan_percent_income" in sample_df.columns else 0.0
        batch["loan_percent_income"] = (batch["loan_amnt"] / safe_income).fillna(fallback)
        st.info("`loan_percent_income` was auto-generated from `loan_amnt / person_income` for batch scoring.")

    missing = [c for c in app_feature_cols if c not in batch.columns]
    if missing:
        st.error(f"Missing columns in uploaded CSV: {missing[:12]}")
    else:
        prob = model.predict_proba(batch[app_feature_cols])[:, 1]
        out = batch.copy()
        out["pred_default_prob"] = prob
        out["decision"] = np.where(
            out["pred_default_prob"] <= threshold * 0.7,
            "APPROVE",
            np.where(out["pred_default_prob"] <= threshold, "MANUAL REVIEW", "REJECT / HIGH RISK"),
        )
        st.dataframe(out.head(30), use_container_width=True)
        st.download_button(
            "Download Batch Results",
            out.to_csv(index=False).encode("utf-8"),
            file_name="batch_scoring_results.csv",
            mime="text/csv",
        )
st.markdown("</div>", unsafe_allow_html=True)
