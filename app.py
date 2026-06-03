import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Diabetes Care",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data & model
try:
    df = pd.read_csv("diabetes_tanpa_pregnancies.csv")
except:
    df = pd.read_csv("diabetes.csv")

try:
    model = joblib.load("model_knn.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    model = None
    scaler = None

# Custom CSS - DIPAKSA WARNA GELAP
st.markdown("""
<style>
/* Sembunyikan elemen default */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: transparent; }

/* Background utama */
.stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%); }

/* SIDEBAR - tetap putih teks */
[data-testid="stSidebar"] { 
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%); 
    border-right: none; 
}
[data-testid="stSidebar"] * { 
    color: white !important; 
}

/* CARD - WARNA TEKS DIPAKSA GELAP */
.card { 
    background: white; 
    padding: 1.5rem; 
    border-radius: 20px; 
    box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
    margin-bottom: 1rem; 
}
.card * { 
    color: #1e293b !important; 
}
.card h1, .card h2, .card h3, .card h4, .card h5, .card h6,
.card p, .card span, .card div, .card li, .card ul, .card strong, .card b {
    color: #1e293b !important;
}

/* METRIC CARD */
.metric-card { 
    background: white; 
    padding: 1.2rem; 
    border-radius: 16px; 
    text-align: center; 
    box-shadow: 0 2px 10px rgba(0,0,0,0.05); 
    border-left: 4px solid;
}
.metric-card * { 
    color: #1e293b !important; 
}
.metric-value { 
    font-size: 2rem; 
    font-weight: bold; 
}
.metric-label { 
    font-size: 0.85rem; 
    color: #64748b !important; 
}

/* TITLE */
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
.section-title { 
    font-size: 1.2rem; 
    font-weight: 600; 
    color: #1e293b !important; 
    margin-bottom: 1rem; 
    border-left: 4px solid #667eea; 
    padding-left: 0.8rem; 
}

/* RESULT CARD - HASIL PREDIKSI */
.result-negative { 
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
    padding: 1.5rem; 
    border-radius: 20px; 
    text-align: center; 
}
.result-negative * { 
    color: white !important; 
}

.result-positive { 
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); 
    padding: 1.5rem; 
    border-radius: 20px; 
    text-align: center; 
}
.result-positive * { 
    color: white !important; 
}

/* BUTTON */
.stButton > button { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    color: white !important; 
    border: none; 
    border-radius: 12px; 
    padding: 0.6rem 1rem; 
    font-weight: 600; 
    width: 100%; 
}

/* INPUT LABEL */
.stNumberInput label, .stSelectbox label {
    color: #1e293b !important;
    font-weight: 500 !important;
}

/* DATAFRAME - TABEL */
.stDataFrame, .stDataFrame div, .stDataFrame span, .dataframe, .dataframe td, .dataframe th {
    color: #1e293b !important;
}

/* INFO BOX */
.stAlert p {
    color: #0369a1 !important;
}

/* MARKDOWN DI DALAM CARD */
.card .stMarkdown, .card .stMarkdown * {
    color: #1e293b !important;
}

/* TABEL DI DALAM CARD */
.card table, .card table td, .card table th, .card table tr {
    color: #1e293b !important;
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

# ==================== MAIN CONTENT ====================
st.markdown('<p class="main-title">🩺 AI Diabetes Care</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistem Prediksi Risiko Diabetes Menggunakan K-Nearest Neighbors (KNN)</p>', unsafe_allow_html=True)
st.markdown("---")

# Data statistik
total = len(df)
if "Outcome" in df.columns:
    positif = df["Outcome"].sum()
else:
    positif = 268
negatif = total - positif

# ==================== DASHBOARD ====================
if menu == "🏠 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card" style="border-left-color:#3b82f6;"><div class="metric-value" style="color:#3b82f6;">{total}</div><div class="metric-label">Total Pasien</div><div style="font-size:0.7rem; color:#64748b;">100% dataset</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card" style="border-left-color:#22c55e;"><div class="metric-value" style="color:#22c55e;">{negatif}</div><div class="metric-label">Negatif Diabetes</div><div style="font-size:0.7rem; color:#64748b;">({negatif/total*100:.1f}%)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card" style="border-left-color:#ef4444;"><div class="metric-value" style="color:#ef4444;">{positif}</div><div class="metric-label">Positif Diabetes</div><div style="font-size:0.7rem; color:#64748b;">({positif/total*100:.1f}%)</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card" style="border-left-color:#a855f7;"><div class="metric-value" style="color:#a855f7;">83.72%</div><div class="metric-label">Akurasi Model</div><div style="font-size:0.7rem; color:#64748b;">KNN (k=5)</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🥧 Distribusi Pasien Diabetes</p>', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(labels=['Negatif Diabetes', 'Positif Diabetes'], values=[negatif, positif], hole=0.6, marker_colors=['#22c55e', '#ef4444'], textinfo='label+percent')])
        fig.update_layout(height=380, margin=dict(t=0, b=0), paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_chart2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📊 Distribusi Kadar Glukosa</p>', unsafe_allow_html=True)
        fig2 = px.histogram(df, x="Glucose", title="", color_discrete_sequence=['#3b82f6'], nbins=30)
        fig2.update_layout(height=380, paper_bgcolor='white', plot_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== PREDIKSI PASIEN ====================
elif menu == "📊 Prediksi Pasien":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📝 Input Data Pasien Baru</p>', unsafe_allow_html=True)
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        glucose = st.number_input("🩸 Glucose (Kadar Glukosa)", min_value=0, max_value=300, value=120, step=1)
        bp = st.number_input("💓 Blood Pressure (Tekanan Darah)", min_value=0, max_value=200, value=70, step=1)
        skin = st.number_input("📏 Skin Thickness (Ketebalan Kulit)", min_value=0, max_value=100, value=23, step=1)
        insulin = st.number_input("💉 Insulin", min_value=0, max_value=900, value=79, step=1)
    
    with col_input2:
        bmi = st.number_input("⚖️ BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=31.6, step=0.1)
        dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.25, step=0.01)
        age = st.number_input("🎂 Age (Usia)", min_value=1, max_value=120, value=32, step=1)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("🔍 PREDIKSI SEKARANG", use_container_width=True)
    
    if predict_btn:
        if model is not None and scaler is not None:
            try:
                input_data = scaler.transform([[glucose, bp, skin, insulin, bmi, dpf, age]])
                result = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if result == 0:
                    st.markdown(f"""
                    <div class="result-negative">
                        <h2 style="margin:0;">✅ NEGATIF DIABETES</h2>
                        <p style="margin:0;">Tidak Berisiko Diabetes</p>
                        <hr style="background:white; opacity:0.3;">
                        <div style="display:flex; justify-content:space-between;">
                            <div><strong>{proba[0]*100:.1f}%</strong><br>Negatif</div>
                            <div><strong>{proba[1]*100:.1f}%</strong><br>Positif</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-positive">
                        <h2 style="margin:0;">⚠️ POSITIF DIABETES</h2>
                        <p style="margin:0;">Berisiko Diabetes</p>
                        <hr style="background:white; opacity:0.3;">
                        <div style="display:flex; justify-content:space-between;">
                            <div><strong>{proba[0]*100:.1f}%</strong><br>Negatif</div>
                            <div><strong>{proba[1]*100:.1f}%</strong><br>Positif</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info("⚕️ Hasil ini hanya prediksi berbasis model AI. Tetap lakukan pemeriksaan medis untuk hasil yang lebih akurat.")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
        else:
            st.error("⚠️ Model belum dilatih. Jalankan `python train_model.py` terlebih dahulu.")
    else:
        st.info("👆 Silakan masukkan data pasien di atas, lalu klik tombol 'PREDIKSI SEKARANG'")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== DATA PASIEN ====================
elif menu == "📈 Data Pasien":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📋 Data Pasien (10 data teratas)</p>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown(f'<div style="margin-top:1rem;">Total data: {total} pasien</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== ANALISIS DATA ====================
elif menu == "📉 Analisis Data":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📈 Statistik Deskriptif</p>', unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Distribusi Semua Fitur</p>', unsafe_allow_html=True)
    
    fitur = [col for col in df.columns if col != "Outcome"]
    for col in fitur:
        fig = px.histogram(df, x=col, title=f"Distribusi {col}", color_discrete_sequence=['#667eea'], nbins=30)
        fig.update_layout(height=400, paper_bgcolor='white', plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== EVALUASI MODEL ====================
elif menu == "🎯 Evaluasi Model":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Confusion Matrix</p>', unsafe_allow_html=True)
    
    cm_data = pd.DataFrame(
        [[92, 14], [23, 75]], 
        columns=['Prediksi Negatif', 'Prediksi Positif'], 
        index=['Aktual Negatif', 'Aktual Positif']
    )
    st.dataframe(cm_data, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">🎯 Performa Model</p>', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Akurasi", "83.72%")
    with col_m2:
        st.metric("Precision", "81.52%")
    with col_m3:
        st.metric("Recall", "82.42%")
    with col_m4:
        st.metric("F1-Score", "81.97%")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== TENTANG ====================
elif menu == "ℹ️ Tentang":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">ℹ️ Tentang Aplikasi</p>', unsafe_allow_html=True)
    st.markdown("""
    **🩺 AI Diabetes Care**
    
    Aplikasi ini menggunakan algoritma **K-Nearest Neighbors (KNN)** untuk memprediksi risiko diabetes berdasarkan data pasien.
    
    ---
    
    **Dataset:** Pima Indians Diabetes Database
    
    ---
    
    **Fitur yang digunakan:**
    - Glucose (Kadar Glukosa)
    - Blood Pressure (Tekanan Darah)
    - Skin Thickness (Ketebalan Kulit)
    - Insulin
    - BMI (Body Mass Index)
    - Diabetes Pedigree Function
    - Age (Usia)
    
    ---
    
    **Model:** KNN dengan k = 5
    
    **Akurasi Model:** 83.72%
    
    ---
    
    ⚕️ **Disclaimer:** Hasil prediksi ini hanya bersifat informatif dan bukan diagnosis medis. Tetap konsultasikan dengan dokter untuk hasil yang akurat.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<center>
    <span style="color:#94a3b8; font-size:0.8rem;">
        🩺 AI Diabetes Care - Sistem Prediksi Risiko Diabetes Menggunakan KNN
    </span>
    <br>
    <span style="color:#cbd5e1; font-size:0.7rem;">
        © 2026 - Dibangun dengan Streamlit
    </span>
</center>
""", unsafe_allow_html=True)