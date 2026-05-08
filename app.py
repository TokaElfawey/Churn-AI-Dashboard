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
    .stApp { background: radial-gradient(circle at top right, #1e293b, #0f172a, #020617); }
    [data-testid="stSidebar"] { background-color: rgba(15, 23, 42, 0.8); border-right: 1px solid #38bdf8; }
    .metric-card {
        background: rgba(255, 255, 255, 0.03); padding: 25px; border-radius: 20px;
        border: 1px solid rgba(0, 245, 255, 0.2); backdrop-filter: blur(10px);
        transition: transform 0.3s ease; text-align: center;
    }
    .metric-card:hover { transform: translateY(-5px); border-color: #00f5ff; }
    .metric-title { color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px; }
    .metric-value { color: #00f5ff; font-size: 28px; font-weight: 800; margin-top: 5px; }
    .result-box { padding: 30px; border-radius: 25px; text-align: center; margin-top: 20px; animation: fadeIn 1s ease-in; }
    @keyframes fadeIn { 0% { opacity: 0; } 100% { opacity: 1; } }
    .stButton>button {
        background: linear-gradient(135deg, #00f5ff 0%, #7c3aed 100%);
        color: white !important; border: none; padding: 15px 30px;
        border-radius: 50px; font-weight: bold; letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER SECTION
# =========================
col_t1, col_t2 = st.columns([1, 4])
with col_t1:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
with col_t2:
    st.markdown("<h1 style='text-align: left; margin-bottom: 0;'>CHURN <span style='color:#00f5ff'>AI</span> PREDICTOR</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:18px;'>Next-Gen Customer Retention Intelligence</p>", unsafe_allow_html=True)

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

    st.markdown("### 💰 Financials")
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# =========================
# MAIN DASHBOARD METRICS
# =========================
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Tenure</div><div class='metric-value'>{tenure} M</div></div>", unsafe_allow_html=True)
with m2:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Monthly</div><div class='metric-value'>${monthly}</div></div>", unsafe_allow_html=True)
with m3:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Contract</div><div class='metric-value' style='font-size:18px'>{contract}</div></div>", unsafe_allow_html=True)
with m4:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Internet</div><div class='metric-value' style='font-size:18px'>{internet}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# PREPROCESSING FUNCTION
# =========================
def preprocess_input():
    # 1. تجهيز الداتا من الـ Widgets
    data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phoneservice == "Yes" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        
        # تحويل الاختيارات لـ أرقام (Mapping)
        "InternetService": {"No": 0, "DSL": 1, "Fiber optic": 2}[internet],
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        "PaymentMethod": {
            "Electronic check": 0, 
            "Mailed check": 1, 
            "Bank transfer (automatic)": 2, 
            "Credit card (automatic)": 3
        }[payment],
        
        # إعدادات متقدمة (Yes/No)
        "OnlineSecurity": 1 if online_security == "Yes" else 0,
        "TechSupport": 1 if tech_support == "Yes" else 0,
        "StreamingTV": 1 if streaming_tv == "Yes" else 0,
        
        # أي أعمدة تانية الموديل محتاجها ضيفيها كأصفار لو مش موجودة
        "MultipleLines": 0,
        "OnlineBackup": 0,
        "DeviceProtection": 0,
        "StreamingMovies": 0
    }
    
    df = pd.DataFrame([data])
    
    # 2. السطر السحري لترتيب الأعمدة
    if model:
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # 3. التأكد أن كل القيم أرقام (Floats)
    return df.astype(float)

# =========================
# TRIGGER PREDICTION
# =========================
if st.button("🚀 ANALYZE CUSTOMER BEHAVIOR"):
    with st.spinner("🧠 AI is analyzing patterns..."):
        time.sleep(1.2)

        input_df = preprocess_input()
        
        # التوقع الحقيقي من الموديل
        try:
            proba = model.predict_proba(input_df)[0][1]
            
            st.markdown("### 📊 Analysis Insights")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if proba > 0.10:
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
                st.progress(float(proba))
                st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# =========================
# FOOTER
# =========================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #94a3b8; font-family: sans-serif; font-size: 14px;'>
        Developed with ❤️ by <b style='color: #00f5ff;'>Section 1 Team</b><br>
        <span style='font-size: 12px;'>Toka Nasr | Aya Ahmed | Toka Alaa | Bavly Hany | Paula Moukhtar</span><br>
        © 2026 Academic Project
    </div>
""", unsafe_allow_html=True)
