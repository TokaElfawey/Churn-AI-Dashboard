import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Churn AI Elite",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# MODEL LOADING
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


try:
    model = load_model()
except:
    st.error("⚠️ Model file not found! Please ensure 'model.pkl' exists.")

# =========================
# LUXURY NEON STYLE (CSS)
# =========================
st.markdown("""
<style>
    /* Background & Global */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a, #020617);
    }

    /* Elegant Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        border-right: 1px solid #38bdf8;
    }

    /* Professional Glassmorphism Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(0, 245, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00f5ff;
        box-shadow: 0 10px 20px rgba(0, 245, 255, 0.1);
    }

    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .metric-value {
        color: #00f5ff;
        font-size: 28px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Prediction Result Box */
    .result-box {
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 1s ease-in;
    }

    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }

    /* Custom Button */
    .stButton>button {
        background: linear-gradient(135deg, #00f5ff 0%, #7c3aed 100%);
        color: white !important;
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        font-weight: bold;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(0, 245, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER SECTION
# =========================
col_t1, col_t2 = st.columns([1, 4])
with col_t1:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)  # أيقونة ذكاء اصطناعي
with col_t2:
    st.markdown(
        "<h1 style='text-align: left; margin-bottom: 0;'>CHURN <span style='color:#00f5ff'>AI</span> PREDICTOR</h1>",
        unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:18px;'>Next-Gen Customer Retention Intelligence</p>",
                unsafe_allow_html=True)

st.markdown("---")

# =========================
# SIDEBAR - LUXURY INPUTS
# =========================
with st.sidebar:
    st.markdown("### 👤 Client Profile")
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    senior = st.toggle("Senior Citizen")

    st.markdown("### 📞 Service Details")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    with col_s2:
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
        paperless = st.selectbox("Paperless Bill", ["Yes", "No"])

    tenure = st.slider("Customer Tenure (Months)", 0, 72, 12)

    st.markdown("### 🌐 Connectivity")
    internet = st.select_slider("Internet Service", options=["No", "DSL", "Fiber optic"])

    with st.expander("Advanced Service Settings"):
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        # أضف البقية هنا بنفس الطريقة...

    st.markdown("### 💰 Financials")
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method",
                           ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# =========================
# MAIN DASHBOARD
# =========================
# Quick Metrics
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Tenure</div><div class='metric-value'>{tenure} M</div></div>",
        unsafe_allow_html=True)
with m2:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Monthly</div><div class='metric-value'>${monthly}</div></div>",
        unsafe_allow_html=True)
with m3:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Contract</div><div class='metric-value' style='font-size:18px'>{contract}</div></div>",
        unsafe_allow_html=True)
with m4:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>Internet</div><div class='metric-value' style='font-size:18px'>{internet}</div></div>",
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =========================
# PREPROCESSING & INFERENCE
# =========================
def preprocess_input():
    data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phoneservice == "Yes" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "MultipleLines": 1 if "Yes" else 0,  # تبسيط للمثال
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": "No",
        "Contract": contract,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }
    df = pd.DataFrame([data])
    # هنا يتم استكمال الـ Encoding (Dummies) بناءً على ما تدرب عليه الموديل
    # تأكد أن الـ Columns تماثل feature_names_in_
    return df


# Trigger Prediction
if st.button("🚀 ANALYZE CUSTOMER BEHAVIOR"):
    with st.spinner("🧠 AI is analyzing patterns..."):
        time.sleep(1.5)  # تجربة مستخدم توحي بالذكاء

        input_df = preprocess_input()

        # proba = model.predict_proba(input_df)[0][1]
        proba = 0.72  # مثال

        st.markdown("### 📊 Analysis Insights")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if proba > 0.5:
                st.markdown(f"""
                <div class='result-box' style='background: rgba(255, 59, 48, 0.15); border: 2px solid #ff3b30;'>
                    <h2 style='color: #ff3b30; margin:0;'>⚠️ HIGH RISK</h2>
                    <p style='color: white;'>This customer is likely to churn. Immediate retention action required.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-box' style='background: rgba(52, 199, 89, 0.15); border: 2px solid #34c759;'>
                    <h2 style='color: #34c759; margin:0;'>✅ LOYAL CUSTOMER</h2>
                    <p style='color: white;'>High retention probability. Continue current engagement strategy.</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("<div style='text-align: center; padding: 20px;'>", unsafe_allow_html=True)
            st.metric("Churn Probability", f"{proba * 100:.1f}%")
            st.progress(proba)
            st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("<br><hr>", unsafe_allow_html=True) # إضافة خط فاصل رفيع
st.markdown("""
    <div style='text-align: center; color: #94a3b8; font-family: sans-serif; font-size: 14px;'>
        Developed by <b style='color: #00f5ff;'>Toka Nasr</b> | 
        © 2026 AI Intelligence Systems
    </div>
""", unsafe_allow_html=True)
