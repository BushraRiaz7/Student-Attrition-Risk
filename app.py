# app.py
# Student Dropout Risk Assessment System (Both Models)
# - UI matches the "old" screenshots (short labels + sliders)
# - Reverse-coded items are handled so that 5 always means Strongly Agree (and higher = higher risk)
# - Trains BOTH Logistic Regression and Random Forest on your uploaded Excel dataset
# - Shows BOTH results together (no dropdown)
# - Keeps mandatory fields + shows red star (*) + blocks assessment until required fields are filled

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ==================================================
# SETTINGS
# ==================================================
DATASET_FILENAME = "student attrition data-1.xlsx"   # keep in same folder as app.py
GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbwPwkSqxdYi1lTX-2w-liZPVzqkZ8k4k1BU_skT4jmgTF6npu_XBd4XLI4BYXSFORy8jA/exec"

# If you don't want to save to sheet, keep it blank:
# GOOGLE_SCRIPT_URL = ""

st.set_page_config(page_title="Student Dropout Risk System", page_icon="🎓", layout="centered")

# ==================================================
# SMALL UI HELPERS
# ==================================================
def required_label(text: str) -> str:
    # red star like your screenshot
    return f"""{text} <span style="color:#d11a2a;font-weight:700">*</span>"""

def risk_bucket(prob_pct: float):
    # same thresholds as your app (0-33, 33-66, 66-100)
    if prob_pct < 33:
        return "Low Risk", "green"
    elif prob_pct < 66:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"

def gauge(prob_pct: float, color: str, title: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(prob_pct),
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 33], "color": "lightgreen"},
                {"range": [33, 66], "color": "gold"},
                {"range": [66, 100], "color": "salmon"},
            ],
        },
    ))
    return fig

# ==================================================
# IMPORTANT: REVERSE-CODE LIST (FROM YOUR THESIS APPENDIX)
# After reverse-coding, higher always means higher risk.
# ==================================================
REVERSE_CODED = {
    # Financial
    "F5", "F7",
    # Psychological
    "P3", "P4", "P7",
    # Social
    "S1", "S2", "S4", "S6",
    # Institutional
    "I1", "I2", "I3", "I4", "I5", "I7",
    # Retention
    "R2", "R4",
}

LIKERT_MIN, LIKERT_MAX = 1, 5

def reverse_score(x):
    # 1<->5, 2<->4, 3 stays 3
    return (LIKERT_MIN + LIKERT_MAX) - x

# ==================================================
# LOAD + PREPROCESS DATA (same idea as thesis)
# ==================================================
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.basename(path)}. "
            f"Please place '{DATASET_FILENAME}' in the SAME folder as app.py"
        )
    df = pd.read_excel(path)  # needs openpyxl installed
    df.columns = [str(c).strip() for c in df.columns]
    return df

def coerce_likert(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # clip to 1..5 (invalid -> NaN)
        df.loc[(df[c] < LIKERT_MIN) | (df[c] > LIKERT_MAX), c] = np.nan
    return df

def build_features_and_target(df: pd.DataFrame):
    # Expected questionnaire items
    F_cols = [f"F{i}" for i in range(1, 8)]
    P_cols = [f"P{i}" for i in range(1, 8)]
    S_cols = [f"S{i}" for i in range(1, 8)]
    I_cols = [f"I{i}" for i in range(1, 8)]
    R_cols = [f"R{i}" for i in range(1, 5)]

    all_items = F_cols + P_cols + S_cols + I_cols + R_cols

    # Validate columns exist
    missing = [c for c in all_items if c not in df.columns]
    if missing:
        raise ValueError(
            "Your Excel file is missing these required columns: "
            + ", ".join(missing)
            + ".\nFix: ensure your dataset contains F1..F7, P1..P7, S1..S7, I1..I7, R1..R4."
        )

    # Likert cleaning
    df = coerce_likert(df.copy(), all_items)

    # Impute missing on item level (median)
    for c in all_items:
        med = df[c].median()
        df[c] = df[c].fillna(med)

    # Reverse-code selected items so that HIGH = HIGH RISK
    for c in REVERSE_CODED:
        df[c] = reverse_score(df[c])

    # Factor scores (risk-direction)
    df["Financial"] = df[F_cols].mean(axis=1)
    df["Psychological"] = df[P_cols].mean(axis=1)
    df["Social"] = df[S_cols].mean(axis=1)
    df["Institutional"] = df[I_cols].mean(axis=1)

    # Integrated early-warning target from R1-R4 (after reverse-coding)
    df["Dropout_Score"] = df[R_cols].mean(axis=1)
    median_score = float(df["Dropout_Score"].median())
    df["Dropout_Risk"] = (df["Dropout_Score"] > median_score).astype(int)  # 1 = at-risk

    X = df[["Financial", "Psychological", "Social", "Institutional"]].values
    y = df["Dropout_Risk"].values.astype(int)

    return X, y, median_score

@st.cache_resource(show_spinner=True)
def train_models():
    dataset_path = os.path.join(os.path.dirname(__file__), DATASET_FILENAME)
    df = load_dataset(dataset_path)
    X, y, median_score = build_features_and_target(df)

    # Logistic Regression (scaled) - explainable
    lr = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    lr.fit(X, y)

    # Random Forest (best performer in your thesis table)
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X, y)

    return lr, rf, median_score

# ==================================================
# PAGE HEADER
# ==================================================
st.title("🎓 Student Dropout Risk Assessment System")
st.caption("Machine-learning based early warning system for student attrition")

# Confidentiality paragraph (exact style like your screenshot)
st.markdown(
    """
    <div style="background:#EAF2FF;border-radius:10px;padding:18px;border:1px solid #d7e6ff;">
      <div style="font-size:15px;line-height:1.6;color:#114a8b;">
        <b>All information provided in this survey will be kept strictly confidential</b> and used solely for academic
        research and early-warning risk assessment purposes. The objective of this system is to support
        universities in identifying risk patterns and implementing timely, student-focused intervention strategies.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ==================================================
# TRAIN/LOAD MODELS (once)
# ==================================================
try:
    LR_MODEL, RF_MODEL, RETENTION_MEDIAN = train_models()
except Exception as e:
    st.error("Model training/loading failed.")
    st.exception(e)
    st.stop()

# ==================================================
# BASIC INFO (MANDATORY like your old app)
# ==================================================
# --- Required fields (render star via HTML, widget label hidden to avoid showing HTML code)
st.markdown(required_label("Student ID"), unsafe_allow_html=True)
student_id = st.text_input("", key="student_id", help="Required", placeholder="", label_visibility="collapsed")

st.markdown(required_label("Program (e.g., Master of Data Science)"), unsafe_allow_html=True)
program = st.text_input("", key="program", help="Required", placeholder="", label_visibility="collapsed")

st.markdown(required_label("Gender"), unsafe_allow_html=True)
gender = st.selectbox("", ["Select", "Female", "Male"], index=0, key="gender", label_visibility="collapsed")

st.markdown(required_label("Level of Study"), unsafe_allow_html=True)
level = st.selectbox("", ["Select", "Undergraduate", "Graduate"], index=0, key="level", label_visibility="collapsed")

# Required validation message (like your screenshot)
missing_required = (
    (student_id.strip() == "") or
    (program.strip() == "") or
    (gender == "Select") or
    (level == "Select")
)
if missing_required:
    st.markdown(
        """
        <div style="background:#fff9db;border:1px solid #ffe08a;border-radius:10px;padding:14px;color:#8a5a00;">
        Please complete all required fields (<span style="color:#d11a2a;font-weight:700">*</span>) before proceeding.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ==================================================
# IMPORTANT NOTE ABOUT SLIDERS + REVERSE CODING
# We keep UI SHORT like your screenshots, but internally we interpret scores as "risk-direction".
# That means: 5 always means "Strongly Agree" AND higher = higher risk.
# For items that were reverse-coded in the questionnaire, your UI already uses the negative phrasing,
# so we treat the slider score as already reverse-coded (risk-direction).
# ==================================================

# FINANCIAL (short labels like old screenshots)
st.subheader("💰 Financial Factors")
F1 = st.slider("Difficulty paying tuition fees", 1, 5, 3)
F2 = st.slider("Financial pressure from family", 1, 5, 3)
F3 = st.slider("Need to work alongside studies", 1, 5, 3)
F4 = st.slider("Unexpected financial expenses", 1, 5, 3)
# Reverse-coded in questionnaire (F5/F7) -> UI is negative, so we keep as risk-direction
F5 = st.slider("Lack of financial aid or scholarships", 1, 5, 3)
F6 = st.slider("Accommodation or transport costs", 1, 5, 3)
F7 = st.slider("Overall financial stress", 1, 5, 3)

st.subheader("🧠 Psychological Factors")
P1 = st.slider("Academic stress", 1, 5, 3)
P2 = st.slider("Loss of motivation", 1, 5, 3)
# Reverse-coded items in questionnaire (P3/P4/P7) -> UI is negative, so keep as risk-direction
P3 = st.slider("Fear of academic failure", 1, 5, 3)
P4 = st.slider("Symptoms of anxiety or depression", 1, 5, 3)
P5 = st.slider("Difficulty concentrating on studies", 1, 5, 3)
P6 = st.slider("Low academic self-confidence", 1, 5, 3)
P7 = st.slider("Emotional exhaustion", 1, 5, 3)

st.subheader("👥 Social Factors")
# Reverse-coded in questionnaire (S1,S2,S4,S6) -> UI is negative, so keep as risk-direction
S1 = st.slider("Lack of peer support", 1, 5, 3)
S2 = st.slider("Family responsibilities", 1, 5, 3)
S3 = st.slider("Feelings of social isolation", 1, 5, 3)
S4 = st.slider("Poor social integration at university", 1, 5, 3)
S5 = st.slider("Lack of sense of belonging", 1, 5, 3)
S6 = st.slider("Limited participation in social activities", 1, 5, 3)
S7 = st.slider("Cultural or language barriers", 1, 5, 3)

st.subheader("🏫 Institutional Factors")
# Many are reverse-coded in questionnaire -> UI is negative, so keep as risk-direction
I1 = st.slider("Poor academic advising", 1, 5, 3)
I2 = st.slider("Administrative difficulties", 1, 5, 3)
I3 = st.slider("Rigid academic policies", 1, 5, 3)
I4 = st.slider("Poor teaching quality", 1, 5, 3)
I5 = st.slider("Limited learning resources", 1, 5, 3)
I6 = st.slider("Lack of institutional support services", 1, 5, 3)
I7 = st.slider("Unsupportive campus environment", 1, 5, 3)

st.markdown("---")

# ==================================================
# ASSESS
# ==================================================
assess_disabled = missing_required

if st.button("🔍 Assess Dropout Risk", disabled=assess_disabled):

    # Factor scores (risk-direction)
    financial = float(np.mean([F1, F2, F3, F4, F5, F6, F7]))
    psychological = float(np.mean([P1, P2, P3, P4, P5, P6, P7]))
    social = float(np.mean([S1, S2, S3, S4, S5, S6, S7]))
    institutional = float(np.mean([I1, I2, I3, I4, I5, I6, I7]))

    X_user = np.array([[financial, psychological, social, institutional]], dtype=float)

    # Predict BOTH models as probability of class 1 (At-Risk)
    lr_prob = float(LR_MODEL.predict_proba(X_user)[0, 1]) * 100.0
    rf_prob = float(RF_MODEL.predict_proba(X_user)[0, 1]) * 100.0

    lr_risk, lr_color = risk_bucket(lr_prob)
    rf_risk, rf_color = risk_bucket(rf_prob)

    st.subheader("📊 Risk Assessment Result (Both Models)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Random Forest (Best Model)")
        st.markdown(
            f"**Risk Level:** <span style='color:{rf_color}; font-weight:bold'>{rf_risk}</span>",
            unsafe_allow_html=True,
        )
        st.write(f"**Dropout Risk Probability:** {rf_prob:.2f}%")
        st.plotly_chart(gauge(rf_prob, rf_color, "RF Dropout Risk (%)"), use_container_width=True)

    with c2:
        st.markdown("### 🧾 Logistic Regression (Explainable)")
        st.markdown(
            f"**Risk Level:** <span style='color:{lr_color}; font-weight:bold'>{lr_risk}</span>",
            unsafe_allow_html=True,
        )
        st.write(f"**Dropout Risk Probability:** {lr_prob:.2f}%")
        st.plotly_chart(gauge(lr_prob, lr_color, "LR Dropout Risk (%)"), use_container_width=True)

    # Dominant factor
    factors = {
        "Financial": financial,
        "Psychological": psychological,
        "Social": social,
        "Institutional": institutional,
    }

    # Academic safeguards:
    # 1) Avoid over-interpreting tiny differences across domains (balanced profile)
    # 2) Handle ties (e.g., Financial and Psychological both high) so we don't pick one arbitrarily
    THRESHOLD = 0.3   # if max-min < 0.3 -> balanced profile
    TIE_EPS = 0.10    # if a domain is within 0.10 of the max -> treat as co-dominant

    max_score = float(max(factors.values()))
    min_score = float(min(factors.values()))
    balanced_profile = (max_score - min_score) < THRESHOLD

    # Find all domains that are effectively "top" (ties)
    top_factors = [k for k, v in factors.items() if (max_score - float(v)) <= TIE_EPS]

    st.subheader("🔎 Dominant Contributing Factor")
    if balanced_profile:
        dominant_text = "No single dominant factor (balanced risk profile)."
        st.write(dominant_text)
    else:
        if len(top_factors) >= 2:
            dominant_text = "Co-dominant factors: " + ", ".join([f"**{x}**" for x in top_factors]) + "."
            st.write(dominant_text)
        else:
            dominant_text = f"The **{top_factors[0]}** domain shows the strongest contribution to dropout risk."
            st.write(dominant_text)

# Recommendations (same as old)
    st.subheader("✅ Recommended Support Actions")
    if balanced_profile:
        st.write("- Student advising & mentoring")
        st.write("- Academic support / tutoring")
        st.write("- Counselling & wellbeing support")
        st.write("- Financial guidance / scholarships (if needed)")
        st.write("- Peer support & engagement activities")
    else:
        action_map = {
            "Financial": [
                "- Fee installment plans",
                "- Emergency financial aid or scholarships",
                "- On-campus part-time employment opportunities",
            ],
            "Psychological": [
                "- Academic & mental health counselling",
                "- Stress management workshops",
                "- Faculty mentoring",
            ],
            "Social": [
                "- Peer mentoring programs",
                "- Student support groups",
                "- Family engagement initiatives",
            ],
            "Institutional": [
                "- Improved academic advising",
                "- Administrative support services",
                "- Flexible institutional policies",
            ],
        }

        # If we have co-dominant factors, show combined actions (deduplicated)
        shown = set()
        for fac in top_factors:
            for line in action_map.get(fac, []):
                if line not in shown:
                    st.write(line)
                    shown.add(line)
# Save payload (optional)
    payload = {
        "student_id": student_id,
        "program": program,
        "gender": gender,
        "level": level,
        # raw items
        "F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6, "F7": F7,
        "P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5, "P6": P6, "P7": P7,
        "S1": S1, "S2": S2, "S3": S3, "S4": S4, "S5": S5, "S6": S6, "S7": S7,
        "I1": I1, "I2": I2, "I3": I3, "I4": I4, "I5": I5, "I6": I6, "I7": I7,
        # factor scores
        "financial_score": financial,
        "psychological_score": psychological,
        "social_score": social,
        "institutional_score": institutional,
        # outputs
        "rf_dropout_probability": rf_prob,
        "rf_risk_level": rf_risk,
        "lr_dropout_probability": lr_prob,
        "lr_risk_level": lr_risk,
        # audit
        "retention_median_used": RETENTION_MEDIAN,
        "note": "Scores are risk-direction (higher = higher risk). Reverse-coded items handled per thesis."
    }

    if GOOGLE_SCRIPT_URL.strip():
        try:
            r = requests.post(GOOGLE_SCRIPT_URL, json=payload, timeout=25)
            if getattr(r, "status_code", None) == 200:
                st.success("Data successfully saved to Google Sheet.")
            else:
                st.info("Save request sent, but confirmation was not received. Please check Google Sheet.")
        except requests.exceptions.Timeout:
            st.info("Save request timed out. Data may still be saved—please check Google Sheet.")
        except Exception:
            st.warning("Data could not be saved to Google Sheet (network/script issue).")
st.caption(
        "Models are trained on your dataset using the integrated early-warning target derived from R1–R4 "
        "(with reverse-coding applied as specified in the thesis)."
    )
