import streamlit as st
import pandas as pd
import joblib

# ── Load Model ───────────────────────────────────────────────
model = joblib.load('decision_tree_churn_model.pkl')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ── Header ───────────────────────────────────────────────────
st.title("📡 Prediksi Customer Churn")
st.markdown("**Telco Customer Churn — Decision Tree Classifier**")
st.markdown("Masukkan data pelanggan untuk memprediksi kemungkinan **churn**.")
st.divider()

# ── Input Form ───────────────────────────────────────────────
st.subheader("📋 Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["Tidak", "Ya"])
    partner = st.selectbox("Memiliki Partner", ["Tidak", "Ya"])
    dependents = st.selectbox("Memiliki Tanggungan", ["Tidak", "Ya"])
    tenure = st.slider("Lama Berlangganan (bulan)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Tidak", "Ya"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Tidak ada layanan telepon", "Tidak", "Ya"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "Tidak ada"]
    )
    online_security = st.selectbox(
        "Online Security",
        ["Tidak ada layanan internet", "Tidak", "Ya"]
    )
    contract = st.selectbox(
        "Jenis Kontrak",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Tidak", "Ya"])
    payment_method = st.selectbox(
        "Metode Pembayaran",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ]
    )
    monthly_charges = st.number_input(
        "Biaya Bulanan ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5
    )
    total_charges = st.number_input(
        "Total Biaya ($)", min_value=0.0, max_value=10000.0, value=850.0, step=10.0
    )

st.divider()

# ── Encode input (sama persis seperti saat training) ─────────
def encode_input():
    return pd.DataFrame([{
        'gender':           0 if gender == "Female" else 1,
        'SeniorCitizen':    0 if senior == "Tidak" else 1,
        'Partner':          0 if partner == "Tidak" else 1,
        'Dependents':       0 if dependents == "Tidak" else 1,
        'tenure':           tenure,
        'PhoneService':     0 if phone_service == "Tidak" else 1,
        'MultipleLines':    ["Tidak ada layanan telepon", "Tidak", "Ya"].index(multiple_lines),
        'InternetService':  ["DSL", "Fiber optic", "Tidak ada"].index(internet_service),
        'OnlineSecurity':   ["Tidak ada layanan internet", "Tidak", "Ya"].index(online_security),
        'Contract':         ["Month-to-month", "One year", "Two year"].index(contract),
        'PaperlessBilling': 0 if paperless_billing == "Tidak" else 1,
        'PaymentMethod':    [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ].index(payment_method),
        'MonthlyCharges':   monthly_charges,
        'TotalCharges':     total_charges,
    }])

# ── Predict ──────────────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", width="stretch", type="primary"):

    input_data = encode_input()
    hasil = model.predict(input_data)
    prob  = model.predict_proba(input_data)

    st.subheader("📊 Hasil Prediksi")

    if hasil[0] == 1:
        st.error("⚠️  **Customer diprediksi akan CHURN**")
        st.markdown("Pelanggan ini berisiko tinggi meninggalkan layanan. "
                    "Pertimbangkan untuk memberikan penawaran retensi.")
    else:
        st.success("✅  **Customer diprediksi TIDAK CHURN**")
        st.markdown("Pelanggan ini kemungkinan akan tetap menggunakan layanan.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probabilitas Tidak Churn", f"{prob[0][0]*100:.1f}%")
    with col_b:
        st.metric("Probabilitas Churn", f"{prob[0][1]*100:.1f}%")

    with st.expander("📄 Lihat Data Input (Encoded)"):
        st.dataframe(input_data, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────
