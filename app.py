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

# Custom CSS - Tema Kesehatan dengan warna font hijau tua dan hitam
st.markdown("""
<style>
/* Sembunyikan elemen default */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: transparent; }

/* Background utama */
.stApp {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
}

/* SIDEBAR */
[data-testid="stSidebar"] { 
    background: linear-gradient(180deg, #064e3b 0%, #047857 100%); 
    border-right: none; 
}
[data-testid="stSidebar"] * { 
    color: white !important; 
}

/* Sidebar logo */
.sidebar-logo {
    text-align: center;
    padding: 1rem;
}

/* CARD UTAMA */
.main-card {
    background: white;
    border-radius: 32px;
    padding: 2rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

/* FORM CARD */
.form-card {
    background: white;
    border-radius: 24px;
    padding: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

/* JUDUL BESAR - HIJAU TUA */
.form-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #064e3b !important;
    margin-bottom: 1.5rem;
    border-left: 4px solid #10b981;
    padding-left: 1rem;
}

/* Judul utama halaman - HIJAU TUA */
.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: #064e3b !important;
    margin-bottom: 0;
}

/* Subtitle - HITAM KECIL */
.subtitle {
    font-size: 0.9rem;
    color: #1e293b !important;
    margin-top: 0;
}

/* Section title dalam card - HIJAU TUA */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #064e3b !important;
    margin-bottom: 1rem;
    border-left: 4px solid #10b981;
    padding-left: 0.8rem;
}

/* Label input - HITAM */
.input-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #1e293b !important;
    margin-bottom: 0.25rem;
    display: block;
}

/* Caption - HITAM KECIL */
.input-sub, .caption, st.caption {
    font-size: 0.7rem;
    color: #4a5568 !important;
}

/* METRIC CARD - VALUE HIJAU TUA, LABEL HITAM */
.metric-card { 
    background: white; 
    padding: 1rem; 
    border-radius: 20px; 
    text-align: center; 
    box-shadow: 0 2px 10px rgba(0,0,0,0.05); 
    border-top: 4px solid;
}
.metric-value { 
    font-size: 1.8rem; 
    font-weight: 700;
    color: #064e3b !important;
}
.metric-label { 
    font-size: 0.75rem; 
    color: #1e293b !important; 
    font-weight: 500;
}
.metric-sub {
    font-size: 0.65rem;
    color: #4a5568 !important;
}

/* RESULT CARD */
.result-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    border-radius: 24px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid #10b981;
}
.result-title {
    font-size: 1rem;
    color: #1e293b !important;
    margin-bottom: 0.5rem;
}
.result-probability {
    font-size: 3rem;
    font-weight: 700;
    color: #064e3b !important;
}
.result-risk-high {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border-color: #ef4444;
}
.result-risk-high .result-probability {
    color: #dc2626 !important;
}
.result-risk-low .result-probability {
    color: #064e3b !important;
}
.result-status {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
.result-suggestion {
    font-size: 0.8rem;
    color: #1e293b !important;
    margin-top: 1rem;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white !important;
    border: none;
    border-radius: 40px;
    padding: 0.7rem 1.5rem;
    font-weight: 600;
    width: auto;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(5,150,105,0.3);
}

/* RESET BUTTON */
.reset-btn {
    background: #f1f5f9 !important;
    color: #64748b !important;
}

/* TABEL - teks hitam */
.stDataFrame, .stDataFrame div, .stDataFrame span, .dataframe, .dataframe td, .dataframe th {
    color: #1e293b !important;
}

/* INFO BOX - teks hitam */
.stAlert p {
    color: #1e293b !important;
}

/* MARKDOWN umum - teks hitam untuk ukuran kecil */
p, li, span, div:not(.metric-value):not(.result-probability) {
    color: #1e293b !important;
}

/* Heading - HIJAU TUA */
h1, h2, h3, .main-title, .form-title {
    color: #064e3b !important;
}

h4, h5, h6 {
    color: #064e3b !important;
}

/* Number input label */
.stNumberInput label {
    color: #1e293b !important;
    font-weight: 500 !important;
}

/* Metric display */
[data-testid="stMetricValue"] {
    color: #064e3b !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #1e293b !important;
}

/* Divider */
hr {
    margin: 1rem 0;
    border-color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size: 2.5rem;">🩺</div>
        <h2 style="color: white !important;">AI Diabetes Care</h2>
        <p style="font-size: 0.7rem; color: white !important;">Sistem Prediksi Risiko Diabetes</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📋 Menu")
    menu = st.radio("", ["🏠 Dashboard", "📝 Prediksi Pasien", "📈 Data Pasien", "📉 Analisis Data", "🎯 Evaluasi Model", "ℹ️ Tentang"], label_visibility="collapsed")
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
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-title">🩺 AI Diabetes Care</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Sistem Prediksi Risiko Diabetes Menggunakan K-Nearest Neighbors (KNN)</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="metric-card" style="border-top-color:#3b82f6;">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total Pasien</div>
            <div class="metric-sub">100% dataset</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="metric-card" style="border-top-color:#10b981;">
            <div class="metric-value">{negatif}</div>
            <div class="metric-label">Negatif Diabetes</div>
            <div class="metric-sub">({negatif/total*100:.1f}%)</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="metric-card" style="border-top-color:#ef4444;">
            <div class="metric-value">{positif}</div>
            <div class="metric-label">Positif Diabetes</div>
            <div class="metric-sub">({positif/total*100:.1f}%)</div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown(f'''
        <div class="metric-card" style="border-top-color:#8b5cf6;">
            <div class="metric-value">83.72%</div>
            <div class="metric-label">Akurasi Model</div>
            <div class="metric-sub">KNN (k=5)</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown('<p class="section-title">🥧 Distribusi Pasien Diabetes</p>', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(labels=['Negatif', 'Positif'], values=[negatif, positif], hole=0.6, marker_colors=['#10b981', '#ef4444'], textinfo='label+percent')])
        fig.update_layout(height=350, margin=dict(t=0, b=0), paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        st.markdown('<p class="section-title">📊 Distribusi Kadar Glukosa</p>', unsafe_allow_html=True)
        fig2 = px.histogram(df, x="Glucose", title="", color_discrete_sequence=['#10b981'], nbins=30)
        fig2.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PREDIKSI PASIEN ====================
elif menu == "📝 Prediksi Pasien":
    
    # Title
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">📝 Formulir Prediksi Risiko Diabetes Pasien</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Silakan isi data pasien di bawah ini untuk mendapatkan prediksi risiko diabetes.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Dua kolom: FORM (kiri) dan HASIL (kanan)
    col_form, col_result = st.columns([1.5, 1])
    
    with col_form:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<p class="form-title">📋 Data Pasien</p>', unsafe_allow_html=True)
        
        # Row 1: Glucose & Blood Pressure
        col_r1a, col_r1b = st.columns(2)
        with col_r1a:
            glucose = st.number_input("🩸 Kadar Glukosa (mg/dL)", min_value=0, max_value=300, value=120, step=1)
            st.caption("Normal: 70-140 mg/dL")
        with col_r1b:
            bp = st.number_input("💓 Tekanan Darah (mmHg)", min_value=0, max_value=200, value=70, step=1)
            st.caption("Normal: 60-80 mmHg")
        
        # Row 2: Skin Thickness & Insulin
        col_r2a, col_r2b = st.columns(2)
        with col_r2a:
            skin = st.number_input("📏 Ketebalan Kulit (mm)", min_value=0, max_value=100, value=23, step=1)
            st.caption("Ukuran lipatan kulit trisep")
        with col_r2b:
            insulin = st.number_input("💉 Insulin (µU/mL)", min_value=0, max_value=900, value=79, step=1)
            st.caption("Normal: 16-166 µU/mL")
        
        # Row 3: BMI & Diabetes Pedigree
        col_r3a, col_r3b = st.columns(2)
        with col_r3a:
            bmi = st.number_input("⚖️ Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=70.0, value=31.6, step=0.1)
            st.caption("Normal: 18.5 - 24.9")
        with col_r3b:
            dpf = st.number_input("🧬 Silsilah Diabetes (0.0-2.5)", min_value=0.0, max_value=3.0, value=0.25, step=0.01)
            st.caption("Faktor riwayat genetik")
        
        # Row 4: Age
        col_r4a, col_r4b = st.columns(2)
        with col_r4a:
            age = st.number_input("🎂 Usia (Tahun)", min_value=1, max_value=120, value=32, step=1)
            st.caption("Usia dalam tahun")
        with col_r4b:
            st.markdown('<div style="height: 65px;"></div>', unsafe_allow_html=True)
        
        # Tombol
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_btn = st.button("🔍 PREDIKSI SEKARANG", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Panduan Pengisian
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<p class="form-title">📌 Keterangan Data & Panduan Pengisian</p>', unsafe_allow_html=True)
        with st.expander("ℹ️ Keterangan Setiap Fitur"):
            st.markdown("""
            - **Kadar Glukosa**: Kadar gula dalam darah, semakin tinggi semakin berisiko
            - **Tekanan Darah**: Tekanan darah sistolik, hipertensi meningkatkan risiko
            - **Ketebalan Kulit**: Ukuran lipatan kulit trisep, indikator lemak tubuh
            - **Insulin**: Kadar insulin dalam darah, ketidaknormalan bisa jadi indikasi
            - **BMI**: Body Mass Index, berat badan berlebih meningkatkan risiko
            - **Silsilah Diabetes**: Faktor genetik dari keluarga
            - **Usia**: Semakin tua umumnya risiko meningkat
            """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_result:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<p class="form-title">📊 Hasil Prediksi</p>', unsafe_allow_html=True)
        
        if predict_btn:
            if model is not None and scaler is not None:
                try:
                    input_data = scaler.transform([[glucose, bp, skin, insulin, bmi, dpf, age]])
                    result = model.predict(input_data)[0]
                    proba = model.predict_proba(input_data)[0]
                    
                    risk_percentage = proba[1] * 100
                    
                    if result == 1:
                        st.markdown(f"""
                        <div class="result-card result-risk-high">
                            <div class="result-title">Probabilitas Risiko Diabetes</div>
                            <div class="result-probability">{risk_percentage:.1f}%</div>
                            <div class="result-status" style="color:#dc2626;">⚠️ RISIKO TINGGI</div>
                            <div class="result-suggestion">Disarankan tindak lanjut medis segera.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card result-risk-low">
                            <div class="result-title">Probabilitas Risiko Diabetes</div>
                            <div class="result-probability">{risk_percentage:.1f}%</div>
                            <div class="result-status" style="color:#064e3b;">✅ RISIKO RENDAH</div>
                            <div class="result-suggestion">Tetap jaga pola hidup sehat.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown('<p style="font-weight:600; color:#064e3b;">📈 Detail Probabilitas</p>', unsafe_allow_html=True)
                    st.progress(risk_percentage/100)
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.metric("Negatif Diabetes", f"{proba[0]*100:.1f}%")
                    with col_p2:
                        st.metric("Positif Diabetes", f"{proba[1]*100:.1f}%")
                    
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
            else:
                st.error("⚠️ Model belum dilatih. Jalankan `python train_model.py` terlebih dahulu.")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem;">🔮</div>
                <p style="color: #4a5568;">Silakan isi data pasien di samping<br>lalu klik tombol "PREDIKSI SEKARANG"</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== DATA PASIEN ====================
elif menu == "📈 Data Pasien":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">📋 Data Pasien</h1>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown(f'<p class="subtitle">📊 Total data: {total} pasien</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== ANALISIS DATA ====================
elif menu == "📉 Analisis Data":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">📈 Analisis Data</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">Statistik Deskriptif</p>', unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown('<p class="section-title">Distribusi Semua Fitur</p>', unsafe_allow_html=True)
    fitur = [col for col in df.columns if col != "Outcome"]
    for col in fitur:
        fig = px.histogram(df, x=col, title=f"Distribusi {col}", color_discrete_sequence=['#10b981'], nbins=30)
        fig.update_layout(height=400, paper_bgcolor='white', plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== EVALUASI MODEL ====================
elif menu == "🎯 Evaluasi Model":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">🎯 Evaluasi Model KNN</h1>', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Akurasi", "83.72%")
    with col_m2:
        st.metric("Precision", "81.52%")
    with col_m3:
        st.metric("Recall", "82.42%")
    with col_m4:
        st.metric("F1-Score", "81.97%")
    
    st.markdown("---")
    st.markdown('<p class="section-title">📊 Confusion Matrix</p>', unsafe_allow_html=True)
    cm_data = pd.DataFrame(
        [[92, 14], [23, 75]], 
        columns=['Prediksi Negatif', 'Prediksi Positif'], 
        index=['Aktual Negatif', 'Aktual Positif']
    )
    st.dataframe(cm_data, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== TENTANG ====================
elif menu == "ℹ️ Tentang":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">ℹ️ Tentang Aplikasi</h1>', unsafe_allow_html=True)
    st.markdown("""
    ### 🩺 AI Diabetes Care
    
    Aplikasi ini menggunakan algoritma **K-Nearest Neighbors (KNN)** untuk memprediksi risiko diabetes.
    
    ---
    
    ### 📊 Dataset
    **Pima Indians Diabetes Database** - 768 pasien
    
    ---
    
    ### 🧠 Model AI
    - **Algoritma:** KNN
    - **k = 5** (5 tetangga terdekat)
    - **Akurasi:** 83.72%
    
    ---
    
    ### ⚕️ Disclaimer
    > Hasil prediksi ini **bukan diagnosis medis**. Konsultasikan dengan dokter untuk hasil akurat.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<center style="margin-top: 2rem;">
    <span style="color:#4a5568; font-size:0.7rem;">
        🩺 AI Diabetes Care - Deteksi Dini, Hidup Lebih Sehat
    </span>
</center>
""", unsafe_allow_html=True)