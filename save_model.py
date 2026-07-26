# ============================================================
# JALANKAN SCRIPT INI DI GOOGLE COLAB / JUPYTER NOTEBOOK
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ── 1. Load dataset ──────────────────────────────────────────
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ── 2. Pilih atribut ─────────────────────────────────────────
df = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
         'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
         'OnlineSecurity', 'Contract', 'PaperlessBilling',
         'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']]

# ── 3. FIX: Mapping manual (BUKAN LabelEncoder) ──────────────
# Ini memastikan nilai encoding SELALU SAMA antara training & prediksi

df['gender']           = df['gender'].map({'Female': 0, 'Male': 1})
df['Partner']          = df['Partner'].map({'No': 0, 'Yes': 1})
df['Dependents']       = df['Dependents'].map({'No': 0, 'Yes': 1})
df['PhoneService']     = df['PhoneService'].map({'No': 0, 'Yes': 1})
df['MultipleLines']    = df['MultipleLines'].map({'No phone service': 0, 'No': 1, 'Yes': 2})
df['InternetService']  = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
df['OnlineSecurity']   = df['OnlineSecurity'].map({'No internet service': 0, 'No': 1, 'Yes': 2})
df['Contract']         = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})
df['PaymentMethod']    = df['PaymentMethod'].map({
    'Bank transfer (automatic)': 0,
    'Credit card (automatic)': 1,
    'Electronic check': 2,
    'Mailed check': 3
})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn']        = df['Churn'].map({'No': 0, 'Yes': 1})

# ── 4. Cek distribusi kelas ───────────────────────────────────
print("Distribusi kelas Churn:")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True).round(3))

# ── 5. Split ─────────────────────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify menjaga proporsi kelas
)

# ── 6. FIX: Train dengan class_weight='balanced' ─────────────
# Ini membuat model memperhatikan kelas minoritas (churn) dengan lebih baik
model = DecisionTreeClassifier(
    class_weight='balanced',   # ← FIX UTAMA
    random_state=42,
    max_depth=10               # mencegah overfitting
)
model.fit(X_train, y_train)

# ── 7. Evaluasi ───────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 8. Simpan model ──────────────────────────────────────────
joblib.dump(model, 'decision_tree_churn_model.pkl')
print("\n✅ Model berhasil disimpan: decision_tree_churn_model.pkl")

# ── 9. (Khusus Google Colab) Download ────────────────────────
try:
    from google.colab import files
    files.download('decision_tree_churn_model.pkl')
    print("📥 File sedang didownload...")
except ImportError:
    print("📁 File tersimpan di direktori lokal.")
