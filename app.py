import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise – Loan Approval Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.15) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 24px;
    padding: 40px 48px;
    text-align: center;
    margin-bottom: 36px;
    backdrop-filter: blur(12px);
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc, #fb7185);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
}
.hero p {
    color: #a5b4fc;
    font-size: 1.1rem;
    font-weight: 400;
    margin: 0;
}

/* ── Section cards ── */
.section-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    backdrop-filter: blur(8px);
}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #a5b4fc;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Streamlit widget labels ── */
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ── Inputs styling ── */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background: rgba(15,12,41,0.6) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
.stNumberInput input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129,140,248,0.25) !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 32px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.6) !important;
}

/* ── Result cards ── */
.result-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.25));
    border: 2px solid rgba(16,185,129,0.6);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    animation: fadeInUp 0.5s ease;
}
.result-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.25));
    border: 2px solid rgba(239,68,68,0.6);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    animation: fadeInUp 0.5s ease;
}
.result-icon { font-size: 4rem; margin-bottom: 12px; }
.result-label {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 8px;
}
.result-approved .result-label { color: #6ee7b7; }
.result-rejected .result-label { color: #fca5a5; }
.result-sub {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}
.result-prob {
    display: inline-block;
    margin-top: 16px;
    padding: 6px 20px;
    border-radius: 50px;
    font-size: 0.9rem;
    font-weight: 600;
}
.result-approved .result-prob {
    background: rgba(16,185,129,0.2);
    color: #6ee7b7;
}
.result-rejected .result-prob {
    background: rgba(239,68,68,0.2);
    color: #fca5a5;
}

/* ── Info metric tiles ── */
.metric-row {
    display: flex;
    gap: 16px;
    margin-top: 28px;
}
.metric-tile {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #a5b4fc;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(BASE_DIR / "ohe_encoder.pkl", "rb") as f:
        ohe = pickle.load(f)
    with open(BASE_DIR / "feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, ohe, feature_cols

model, scaler, ohe, feature_cols = load_artifacts()

# ─── OHE column options (from training data) ────────────────────────────────
OHE_COLS    = ["Marital_Status", "Employment_Status", "Loan_Purpose", "Gender", "Employer_Category", "Property_Area"]
OHE_OPTIONS = {
    "Marital_Status"    : ["Married", "Single"],
    "Employment_Status" : ["Contract", "Salaried", "Self-employed", "Unemployed"],
    "Loan_Purpose"      : ["Business", "Car", "Education", "Home", "Personal"],
    "Gender"            : ["Female", "Male"],
    "Employer_Category" : ["Business", "Government", "MNC", "Private", "Unemployed"],
    "Property_Area"     : ["Rural", "Semiurban", "Urban"],
}

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>💳 CreditWise</h1>
    <p>AI-powered loan approval prediction · Fill in the applicant details below to get an instant decision</p>
</div>
""", unsafe_allow_html=True)

# ─── Form ────────────────────────────────────────────────────────────────────
with st.form("loan_form"):

    # ── Section 1: Personal Info ──
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    age         = c1.number_input("Age",               min_value=18,   max_value=80,    value=35,     step=1)
    gender      = c2.selectbox("Gender",               OHE_OPTIONS["Gender"])
    marital     = c3.selectbox("Marital Status",       OHE_OPTIONS["Marital_Status"])
    dependents  = c4.number_input("Dependents",        min_value=0,    max_value=10,    value=1,      step=1)

    c5, c6 = st.columns(2)
    education   = c5.selectbox("Education Level",     ["Graduate", "Not Graduate"])
    property_a  = c6.selectbox("Property Area",       OHE_OPTIONS["Property_Area"])

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section 2: Employment & Income ──
    st.markdown('<div class="section-title">💼 Employment & Income</div>', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    emp_status  = c7.selectbox("Employment Status",   OHE_OPTIONS["Employment_Status"])
    employer    = c8.selectbox("Employer Category",   OHE_OPTIONS["Employer_Category"])
    app_income  = c9.number_input("Applicant Income (₹)", min_value=0, max_value=500000, value=50000, step=500)

    c10, c11 = st.columns(2)
    coapp_inc   = c10.number_input("Co-applicant Income (₹)", min_value=0, max_value=300000, value=0,     step=500)
    savings     = c11.number_input("Savings (₹)",             min_value=0, max_value=1000000, value=15000, step=500)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section 3: Loan Details ──
    st.markdown('<div class="section-title">🏦 Loan Details</div>', unsafe_allow_html=True)
    c12, c13, c14 = st.columns(3)
    loan_purpose  = c12.selectbox("Loan Purpose",     OHE_OPTIONS["Loan_Purpose"])
    loan_amount   = c13.number_input("Loan Amount (₹)",  min_value=1000,  max_value=10000000, value=200000, step=1000)
    loan_term     = c14.number_input("Loan Term (months)", min_value=6, max_value=360,  value=120,    step=6)

    c15, c16, c17 = st.columns(3)
    credit_score  = c15.number_input("Credit Score",  min_value=300,   max_value=900,   value=650,    step=1)
    dti_ratio     = c16.number_input("DTI Ratio",     min_value=0.0,   max_value=1.0,   value=0.35,   step=0.01, format="%.2f")
    collateral    = c17.number_input("Collateral Value (₹)", min_value=0, max_value=10000000, value=50000, step=1000)

    c18, c19 = st.columns(2)
    existing_loans = c18.number_input("Existing Loans", min_value=0, max_value=20, value=1, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍  Predict Loan Approval")

# ─── Prediction ──────────────────────────────────────────────────────────────
if submitted:
    # 1. Education Level: Graduate → 0, Not Graduate → 1 (sklearn alphabetical)
    edu_encoded = 0 if education == "Graduate" else 1

    # 2. Build base numeric dict
    base = {
        "Applicant_Income"   : app_income,
        "Coapplicant_Income" : coapp_inc,
        "Age"                : age,
        "Dependents"         : dependents,
        "Existing_Loans"     : existing_loans,
        "Savings"            : savings,
        "Collateral_Value"   : collateral,
        "Loan_Amount"        : loan_amount,
        "Loan_Term"          : loan_term,
        "Education_Level"    : edu_encoded,
    }

    # 3. OHE encode categorical columns
    cat_df  = pd.DataFrame([[marital, emp_status, loan_purpose, gender, employer, property_a]],
                            columns=OHE_COLS)
    ohe_arr = ohe.transform(cat_df)
    ohe_df  = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(OHE_COLS))

    # 4. Feature engineering: squared terms
    base["DTI_Ratio_sq"]    = dti_ratio    ** 2
    base["Credit_Score_sq"] = credit_score ** 2

    # 5. Combine into one row aligned to training columns
    input_df = pd.DataFrame([base])
    input_df = pd.concat([input_df, ohe_df], axis=1)
    input_df = input_df[feature_cols]          # Align column order exactly

    # 6. Scale & predict
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0]

    approved     = prediction == 1
    confidence   = proba[1] if approved else proba[0]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Result card ──
    if approved:
        st.markdown(f"""
        <div class="result-approved">
            <div class="result-icon">✅</div>
            <div class="result-label">Loan Approved!</div>
            <p class="result-sub">Based on the applicant's profile, the loan is likely to be approved.</p>
            <span class="result-prob">Confidence: {confidence*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
            <div class="result-icon">❌</div>
            <div class="result-label">Loan Rejected</div>
            <p class="result-sub">Based on the applicant's profile, the loan is unlikely to be approved.</p>
            <span class="result-prob">Confidence: {confidence*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Key metrics summary ──
    dti_pct   = f"{dti_ratio*100:.0f}%"
    income_to_loan = f"{(app_income / loan_amount * 100):.1f}%" if loan_amount > 0 else "N/A"

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="metric-value">{credit_score}</div>
            <div class="metric-label">Credit Score</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">{dti_pct}</div>
            <div class="metric-label">DTI Ratio</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">₹{loan_amount:,}</div>
            <div class="metric-label">Loan Amount</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">{income_to_loan}</div>
            <div class="metric-label">Income / Loan</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">{loan_term}m</div>
            <div class="metric-label">Loan Term</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#334155; font-size:0.8rem;'>
    CreditWise · Powered by Logistic Regression · Built with Streamlit
</p>
""", unsafe_allow_html=True)
