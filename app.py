import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── Load Model ──────────────────────────────────────────────
model = joblib.load('decision_tree_churn_model.pkl')

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ── Header ──────────────────────────────────────────────────
st.title("📡 Prediksi Customer Churn")
st.markdown("**Telco Customer Churn — Decision Tree Classifier**")
st.markdown("Masukkan data pelanggan di bawah ini untuk memprediksi apakah pelanggan akan **churn** atau **tidak**.")
st.divider()

# ── Input Form ──────────────────────────────────────────────
st.subheader("📋 Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )
    senior = st.selectbox(
        "Senior Citizen",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )
    partner = st.selectbox(
        "Memiliki Partner",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )
    dependents = st.selectbox(
        "Memiliki Tanggungan",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )
    tenure = st.slider(
        "Lama Berlangganan (bulan)",
        min_value=0, max_value=72, value=12
    )
    phone_service = st.selectbox(
        "Phone Service",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )
    multiple_lines = st.selectbox(
        "Multiple Lines",
        options=[0, 1, 2],
        format_func=lambda x: ["Tidak ada layanan telepon", "Tidak", "Ya"][x]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        options=[0, 1, 2],
        format_func=lambda x: ["DSL", "Fiber optic", "Tidak ada"][x]
    )
    online_security = st.selectbox(
        "Online Security",
        options=[0, 1, 2],
        format_func=lambda x: ["Tidak ada layanan internet", "Tidak", "Ya"][x]
    )
    contract = st.selectbox(
        "Jenis Kontrak",
        options=[0, 1, 2],
        format_func=lambda x: ["Month-to-month", "One year", "Two year"][x]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )
    payment_method = st.selectbox(
        "Metode Pembayaran",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ][x]
    )
    monthly_charges = st.number_input(
        "Biaya Bulanan ($)",
        min_value=0.0, max_value=200.0, value=70.0, step=0.5
    )
    total_charges = st.number_input(
        "Total Biaya ($)",
        min_value=0.0, max_value=10000.0, value=850.0, step=10.0
    )

st.divider()

# ── Predict Button ───────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([{
        'gender':           gender,
        'SeniorCitizen':    senior,
        'Partner':          partner,
        'Dependents':       dependents,
        'tenure':           tenure,
        'PhoneService':     phone_service,
        'MultipleLines':    multiple_lines,
        'InternetService':  internet_service,
        'OnlineSecurity':   online_security,
        'Contract':         contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod':    payment_method,
        'MonthlyCharges':   monthly_charges,
        'TotalCharges':     total_charges,
    }])

    hasil = model.predict(input_data)
    prob  = model.predict_proba(input_data)

    st.subheader("📊 Hasil Prediksi")

    if hasil[0] == 1:
        st.error("⚠️  **Customer diprediksi akan CHURN**")
        st.markdown("Pelanggan ini berisiko tinggi meninggalkan layanan. Pertimbangkan untuk memberikan penawaran retensi.")
    else:
        st.success("✅  **Customer diprediksi TIDAK CHURN**")
        st.markdown("Pelanggan ini kemungkinan akan tetap menggunakan layanan.")

    # Probabilitas
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probabilitas Tidak Churn", f"{prob[0][0]*100:.1f}%")
    with col_b:
        st.metric("Probabilitas Churn", f"{prob[0][1]*100:.1f}%")

    # Ringkasan input
    with st.expander("📄 Lihat Ringkasan Data Input"):
        st.dataframe(input_data, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("M. Nabil Yudhistira — 23.12.2960 | Universitas AMIKOM Yogyakarta 2026")
