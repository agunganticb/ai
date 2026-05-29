import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Diabetes", layout="wide")
st.title("🔮 Prediksi Risiko Diabetes")

# Load model dan scaler
@st.cache_resource
def load_model():
    model = joblib.load("model_knn.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.subheader("📋 Input Data Pasien")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies (Jumlah Kehamilan)", min_value=0, max_value=20, value=3)
    glucose = st.number_input("Glucose (Kadar Glukosa)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (Tekanan Darah)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=23)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=31.6, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.25, step=0.01)
    age = st.number_input("Age (Usia)", min_value=0, max_value=120, value=32)

if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
    # Buat array input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scaling menggunakan MinMaxScaler yang sudah disimpan
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    
    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")
    
    col_result, col_confidence = st.columns(2)
    
    with col_result:
        if pred == 0:
            st.success("✅ **NEGATIF DIABETES**")
            st.write("Pasien tidak berisiko diabetes")
        else:
            st.error("⚠️ **POSITIF DIABETES**")
            st.write("Pasien berisiko diabetes")
    
    with col_confidence:
        confidence = max(proba) * 100
        st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
        st.progress(confidence / 100)
    
    # Detail probabilitas
    with st.expander("📈 Detail Probabilitas"):
        st.write(f"Probabilitas Negatif: **{proba[0]*100:.2f}%**")
        st.write(f"Probabilitas Positif: **{proba[1]*100:.2f}%**")