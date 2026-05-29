import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random

st.set_page_config(
    page_title="AI Diabetes Care",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data & model
df = pd.read_csv("diabetes.csv")
model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler.pkl")

# Custom CSS
st.markdown("""
<style>
/* Hide default Streamlit elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: transparent; }

/* Main background */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: white;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: white;
}

/* Card styling */
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}

.metric-card {
    background: white;
    padding: 1.2rem;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    border-left: 4px solid;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
}

.metric-label {
    font-size: 0.85rem;
    color: #666;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 20px;
    color: white;
    text-align: center;
}

.result-negative {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}

.result-positive {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102,126,234,0.4);
}

/* Input styling */
.stNumberInput input, .stSelectbox input {
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}

/* Title styling */
.main-title {
    font-size: 2rem;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 0;
}

/* Section title */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 1rem;
    border-left: 4px solid #667eea;
    padding-left: 0.8rem;
}

/* Table styling */
.dataframe {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("<h1 style='text-align:center; font-size:1.5rem;'>🩺 AI Diabetes Care</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:0.8rem; opacity:0.8;'>Sistem Prediksi Risiko Diabetes<br>Menggunakan K-Nearest Neighbors (KNN)</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📋 Menu")
    menu = st.radio("", ["🏠 Dashboard", "📊 Prediksi Pasien", "📈 Data Pasien", "📉 Analisis Data", "🎯 Evaluasi Model", "ℹ️ Tentang"], label_visibility="collapsed")
    st.markdown("---")
    
    st.markdown("### 📝 Input Data Pasien Baru")
    
    pregnancies = st.number_input("Pregnancies (Jumlah Kehamilan)", 0, 20, 2)
    glucose = st.number_input("Glucose (Kadar Glukosa)", 0, 300, 120)
    bp = st.number_input("Blood Pressure (Tekanan Darah)", 0, 200, 70)
    skin = st.number_input("Skin Thickness (Ketebalan Kulit)", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 28.5, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.35, step=0.01)
    age = st.number_input("Age (Usia)", 1, 120, 30)
    
    predict_btn = st.button("🔍 Prediksi Sekarang", use_container_width=True)

# ==================== MAIN CONTENT ====================
st.markdown('<p class="main-title">🩺 AI Diabetes Care</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistem Prediksi Risiko Diabetes Menggunakan K-Nearest Neighbors (KNN)</p>', unsafe_allow_html=True)
st.markdown("---")

# Data statistik
total = len(df)
positif = df["Outcome"].sum()
negatif = total - positif
accuracy = 83.72

# Metric cards row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #3b82f6;">
        <div class="metric-value" style="color:#3b82f6;">{total}</div>
        <div class="metric-label">Total Pasien</div>
        <div style="font-size:0.7rem; color:#94a3b8;">100% dari dataset</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #22c55e;">
        <div class="metric-value" style="color:#22c55e;">{negatif}</div>
        <div class="metric-label">Negatif Diabetes</div>
        <div style="font-size:0.7rem; color:#94a3b8;">({negatif/total*100:.1f}%) Tidak berisiko</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #ef4444;">
        <div class="metric-value" style="color:#ef4444;">{positif}</div>
        <div class="metric-label">Positif Diabetes</div>
        <div style="font-size:0.7rem; color:#94a3b8;">({positif/total*100:.1f}%) Berisiko</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #a855f7;">
        <div class="metric-value" style="color:#a855f7;">{accuracy}%</div>
        <div class="metric-label">Akurasi Model</div>
        <div style="font-size:0.7rem; color:#94a3b8;">KNN (k = 5)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Two columns layout
left_col, right_col = st.columns([1.2, 1])

with left_col:
    # Hasil Prediksi Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Hasil Prediksi</p>', unsafe_allow_html=True)
    
    if predict_btn:
        input_data = scaler.transform([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        result = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        
        neg_prob = proba[0] * 100
        pos_prob = proba[1] * 100
        
        if result == 0:
            st.markdown(f"""
            <div class="result-card result-negative">
                <h2 style="margin:0;">✅ NEGATIF DIABETES</h2>
                <p style="margin:0; opacity:0.9;">Tidak Berisiko Diabetes</p>
                <hr style="background:white; opacity:0.3;">
                <div style="display:flex; justify-content:space-between;">
                    <div><strong>{neg_prob:.1f}%</strong><br>Negatif</div>
                    <div><strong>{pos_prob:.1f}%</strong><br>Positif</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-positive">
                <h2 style="margin:0;">⚠️ POSITIF DIABETES</h2>
                <p style="margin:0; opacity:0.9;">Berisiko Diabetes</p>
                <hr style="background:white; opacity:0.3;">
                <div style="display:flex; justify-content:space-between;">
                    <div><strong>{neg_prob:.1f}%</strong><br>Negatif</div>
                    <div><strong>{pos_prob:.1f}%</strong><br>Positif</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("⚕️ Hasil ini hanya prediksi berbasis model AI. Tetap lakukan pemeriksaan medis untuk hasil yang lebih akurat.")
    else:
        st.info("👈 Masukkan data pasien di sidebar dan klik tombol 'Prediksi Sekarang'")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Distribusi Fitur Utama
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📈 Distribusi Fitur Utama</p>', unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown(f"""
        <div style="background:#f1f5f9; padding:0.8rem; border-radius:12px; margin-bottom:0.5rem;">
            <strong>🩸 Glucose</strong>
            <div style="background:#e2e8f0; border-radius:10px; margin-top:5px;">
                <div style="background:#3b82f6; width:{glucose/3}%; border-radius:10px; padding:4px; color:white; text-align:center;">{glucose}</div>
            </div>
        </div>
        <div style="background:#f1f5f9; padding:0.8rem; border-radius:12px; margin-bottom:0.5rem;">
            <strong>⚖️ BMI</strong>
            <div style="background:#e2e8f0; border-radius:10px; margin-top:5px;">
                <div style="background:#22c55e; width:{bmi*1.5}%; border-radius:10px; padding:4px; color:white; text-align:center;">{bmi}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown(f"""
        <div style="background:#f1f5f9; padding:0.8rem; border-radius:12px; margin-bottom:0.5rem;">
            <strong>🎂 Age</strong>
            <div style="background:#e2e8f0; border-radius:10px; margin-top:5px;">
                <div style="background:#a855f7; width:{age/1.2}%; border-radius:10px; padding:4px; color:white; text-align:center;">{age}</div>
            </div>
        </div>
        <div style="background:#f1f5f9; padding:0.8rem; border-radius:12px; margin-bottom:0.5rem;">
            <strong>💉 Insulin</strong>
            <div style="background:#e2e8f0; border-radius:10px; margin-top:5px;">
                <div style="background:#f59e0b; width:{insulin/9}%; border-radius:10px; padding:4px; color:white; text-align:center;">{insulin}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # Donut Chart
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🥧 Distribusi Pasien Diabetes</p>', unsafe_allow_html=True)
    
    fig = go.Figure(data=[go.Pie(
        labels=['Negatif Diabetes', 'Positif Diabetes'],
        values=[negatif, positif],
        hole=0.6,
        marker_colors=['#22c55e', '#ef4444'],
        textinfo='label+percent',
        textposition='auto'
    )])
    fig.update_layout(
        height=300,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='white',
        font=dict(color='#1e293b')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    <div style="text-align:center; margin-top:-20px;">
        <span style="color:#22c55e;">●</span> Negatif: {negatif} ({negatif/total*100:.1f}%)<br>
        <span style="color:#ef4444;">●</span> Positif: {positif} ({positif/total*100:.1f}%)
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Confusion Matrix (simplified)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Confusion Matrix</p>', unsafe_allow_html=True)
    
    # Data dari gambar referensi
    cm_data = pd.DataFrame(
        [[92, 14], [23, 75]],
        columns=['Prediksi Negatif', 'Prediksi Positif'],
        index=['Aktual Negatif', 'Aktual Positif']
    )
    st.dataframe(cm_data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Performa Model
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🎯 Performa Model</p>', unsafe_allow_html=True)
    
    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        st.metric("Akurasi", "83.72%")
        st.metric("Precision", "81.52%")
    with perf_col2:
        st.metric("Recall", "82.42%")
        st.metric("F1-Score", "81.97%")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Riwayat Prediksi Pasien
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">📋 Riwayat Prediksi Pasien</p>', unsafe_allow_html=True)

history_data = pd.DataFrame([
    {"No": 1, "Tanggal": "20 Mei 2025 10:25", "Glucose": 120, "BMI": 28.5, "Age": 30, "Hasil Prediksi": "Negatif Diabetes", "Probabilitas": "82.4%", "Status": "✅ Tidak Berisiko"},
    {"No": 2, "Tanggal": "20 Mei 2025 10:10", "Glucose": 165, "BMI": 32.1, "Age": 45, "Hasil Prediksi": "Positif Diabetes", "Probabilitas": "78.9%", "Status": "⚠️ Berisiko"},
    {"No": 3, "Tanggal": "20 Mei 2025 09:58", "Glucose": 95, "BMI": 24.3, "Age": 25, "Hasil Prediksi": "Negatif Diabetes", "Probabilitas": "88.1%", "Status": "✅ Tidak Berisiko"},
])

st.dataframe(history_data, use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<center>
    <span style="color:#94a3b8; font-size:0.8rem;">
        🩺 AI Diabetes Care - Sistem Prediksi Risiko Diabetes Menggunakan Algoritma K-Nearest Neighbors (KNN)
    </span>
    <br>
    <span style="color:#cbd5e1; font-size:0.7rem;">
        Hasil ini hanya prediksi berbasis model AI. Tetap lakukan pemeriksaan medis untuk hasil yang lebih akurat.
    </span>
</center>
""", unsafe_allow_html=True)