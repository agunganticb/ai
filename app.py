import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman
st.set_page_config(
    page_title="Pendeteksi Risiko Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #e0f2e0 0%, #c8e6c9 100%);
    }
    
    /* SEMBUNYIKAN SEMUA ALERT, INFO, WARNING, ERROR */
    .stAlert, 
    .stInformation, 
    .stInfo, 
    .stWarning, 
    .stError,
    .stSuccess,
    .stToast,
    div[data-testid="stAlert"],
    div[data-testid="stInformation"],
    div[data-testid="stInfo"],
    .element-container:has(.stAlert),
    .element-container:has(.stInfo) {
        display: none !important;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #2e7d32 !important;
    }
    
    .main-subtitle {
        font-size: 1rem;
        text-align: center;
        color: #555555 !important;
        margin-bottom: 1.5rem;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .hero-section .hero-title,
    .hero-section .hero-subtitle {
        color: white !important;
    }
    
    .hero-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .hero-subtitle {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        text-align: center;
        margin: 1.5rem 0 1rem 0;
        color: #2e7d32 !important;
    }
    
    .step-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(102,187,106,0.15);
        border: 1px solid #a5d6a7;
        height: 100%;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .step-number {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.8rem auto;
        font-size: 1rem;
        font-weight: bold;
        color: white !important;
    }
    
    .step-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #2e7d32 !important;
    }
    
    .step-description {
        font-size: 0.8rem;
        color: #555555 !important;
        line-height: 1.4;
    }
    
    .security-banner {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .security-banner .security-title,
    .security-banner .security-text {
        color: white !important;
    }
    
    .security-title {
        font-size: 0.95rem;
        font-weight: 700;
    }
    
    .security-text {
        font-size: 0.75rem;
        opacity: 0.9;
    }
    
    .form-container {
        background-color: white;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(102,187,106,0.1);
        border: 1px solid #c8e6c9;
    }
    
    .form-section-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #66bb6a;
        display: inline-block;
        color: #2e7d32 !important;
    }
    
    .input-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #333333 !important;
        margin-bottom: 0.2rem;
        margin-top: 0.5rem;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #c8e6c9, transparent);
        margin: 1.5rem 0;
    }
    
    .result-low {
        background-color: #e8f5e9;
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        border: 2px solid #66bb6a;
    }
    
    .result-low .result-title {
        color: #2e7d32 !important;
    }
    
    .result-low .result-text {
        color: #555555 !important;
    }
    
    .result-high {
        background-color: #ffebee;
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        border: 2px solid #ef9a9a;
    }
    
    .result-high .result-title {
        color: #c62828 !important;
    }
    
    .result-high .result-text {
        color: #555555 !important;
    }
    
    .result-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .result-text {
        font-size: 0.85rem;
    }
    
    .footer {
        text-align: center;
        padding: 1.2rem;
        color: #666666 !important;
        font-size: 0.7rem;
        border-top: 1px solid #c8e6c9;
        margin-top: 2rem;
    }
    
    .stNumberInput input {
        border-radius: 10px;
        border: 1px solid #a5d6a7;
        padding: 0.6rem 0.8rem;
        font-size: 0.9rem;
        background-color: white;
        color: #333333 !important;
    }
    
    .stNumberInput input:focus {
        border-color: #66bb6a;
        box-shadow: 0 0 0 2px rgba(102,187,106,0.2);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: 30px;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,187,106,0.4);
    }
    
    .stProgress > div > div {
        background-color: #66bb6a !important;
    }
    
    .streamlit-expanderHeader {
        font-weight: 500;
        background-color: #e8f5e9;
        border-radius: 8px;
        color: #2e7d32 !important;
    }
    
    .stMetric label {
        color: #333333 !important;
    }
    
    .stMetric .metric-value {
        color: #2e7d32 !important;
    }
    
    .stCaption {
        color: #666666 !important;
        font-size: 0.7rem !important;
    }
    
    h3, h4, h5 {
        color: #2e7d32 !important;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_knn.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except:
        return None, None

model, scaler = load_model()


if st.session_state.page == 'home':
    st.markdown('<h1 class="main-title">Pendeteksi Risiko Diabetes</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Masukkan data kesehatan Anda dan dapatkan prediksi risiko diabetes secara cepat dan mudah.</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Deteksi Dini, Hidup Lebih Sehat</div>
        <div class="hero-subtitle">Gunakan kecerdasan buatan untuk mengetahui risiko diabetes Anda</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Mulai Prediksi", use_container_width=True):
            st.session_state.page = 'prediksi'
            st.rerun()
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Bagaimana Cara Kerjanya?</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">1</div>
            <div class="step-title">Isi Data Kesehatan</div>
            <div class="step-description">Masukkan data seperti usia, berat badan, gula darah, tekanan darah, dan lainnya.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">2</div>
            <div class="step-title">Proses Analisis</div>
            <div class="step-description">Sistem akan menganalisis data yang Anda masukkan menggunakan teknologi AI.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">3</div>
            <div class="step-title">Dapatkan Hasil</div>
            <div class="step-description">Lihat hasil prediksi risiko diabetes beserta saran kesehatan untuk Anda.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="security-banner">
        <div class="security-title">Data Anda Aman</div>
        <div class="security-text">Kami menjaga kerahasiaan data pribadi Anda dan tidak akan membagikannya kepada pihak manapun.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="footer">2026 Pendeteksi Risiko Diabetes | Didukung oleh Teknologi AI</div>', unsafe_allow_html=True)

# ==================== HALAMAN PREDIKSI ====================
elif st.session_state.page == 'prediksi':
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Kembali"):
            st.session_state.page = 'home'
            st.rerun()
    
    st.markdown('<h1 class="main-title">Formulir Data Kesehatan</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Silakan isi data kesehatan Anda di bawah ini</p>', unsafe_allow_html=True)
    
    # Form Input
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title">Data Dasar</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-label">Usia (Tahun)</div>', unsafe_allow_html=True)
        usia = st.number_input("usia", min_value=1, max_value=120, value=45, label_visibility="collapsed")
        
        st.markdown('<div class="input-label">BMI (Body Mass Index)</div>', unsafe_allow_html=True)
        bmi = st.number_input("bmi", min_value=10.0, max_value=60.0, value=30.0, step=0.1, label_visibility="collapsed")
        
        st.markdown('<div class="input-label">Jumlah Kehamilan</div>', unsafe_allow_html=True)
        kehamilan = st.number_input("kehamilan", min_value=0, max_value=20, value=2, label_visibility="collapsed")
        
        st.markdown('<div class="input-label">Ketebalan Kulit (mm)</div>', unsafe_allow_html=True)
        kulit = st.number_input("kulit", min_value=0, max_value=100, value=25, label_visibility="collapsed")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title">Data Medis</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-label">Kadar Glukosa (mg/dL)</div>', unsafe_allow_html=True)
        glukosa = st.number_input("glukosa", min_value=50, max_value=300, value=150, label_visibility="collapsed")
        
        st.markdown('<div class="input-label">Tekanan Darah (mm Hg)</div>', unsafe_allow_html=True)
        tekanan_darah = st.number_input("tekanan", min_value=50, max_value=200, value=85, label_visibility="collapsed")
        
        st.markdown('<div class="input-label">Kadar Insulin (µU/ml)</div>', unsafe_allow_html=True)
        insulin = st.number_input("insulin", min_value=0, max_value=900, value=100, label_visibility="collapsed")
        
        st.markdown('<div class="input-label">Diabetes Pedigree Function</div>', unsafe_allow_html=True)
        dpf = st.number_input("dpf", min_value=0.0, max_value=2.5, value=0.50, step=0.01, label_visibility="collapsed")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict = st.button("Analisis Risiko Sekarang", use_container_width=True)
    
    if predict:
        if model is not None and scaler is not None:
            input_data = np.array([[kehamilan, glukosa, tekanan_darah, kulit, insulin, bmi, dpf, usia]])
            
            try:
                input_scaled = scaler.transform(input_data)
                pred = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]
                
                st.markdown("---")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if pred == 0:
                        st.markdown("""
                        <div class="result-low">
                            <div class="result-title">Risiko Rendah</div>
                            <div class="result-text">Anda memiliki risiko diabetes yang rendah</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("Saran untuk Anda", expanded=True):
                            st.write("- Pertahankan pola hidup sehat")
                            st.write("- Rutin berolahraga minimal 30 menit per hari")
                            st.write("- Konsumsi makanan bergizi seimbang")
                            st.write("- Perbanyak konsumsi sayur dan buah")
                            st.write("- Cek gula darah secara rutin")
                    else:
                        st.markdown("""
                        <div class="result-high">
                            <div class="result-title">Risiko Tinggi</div>
                            <div class="result-text">Anda memiliki risiko diabetes yang tinggi</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("Saran untuk Anda", expanded=True):
                            st.write("- Segera konsultasikan dengan dokter")
                            st.write("- Lakukan pemeriksaan gula darah lebih lanjut")
                            st.write("- Mulai diet rendah gula dan karbohidrat")
                            st.write("- Perbanyak aktivitas fisik minimal 30 menit per hari")
                            st.write("- Pantau tekanan darah secara teratur")
                            st.write("- Ikuti saran dan pengobatan dari dokter")
                
                with col2:
                    st.markdown("### Tingkat Keyakinan")
                    
                    if pred == 0:
                        st.write(f"Probabilitas Risiko Rendah: {proba[0]*100:.1f}%")
                        st.progress(int(proba[0]*100))
                        st.write(f"Probabilitas Risiko Tinggi: {proba[1]*100:.1f}%")
                    else:
                        st.write(f"Probabilitas Risiko Tinggi: {proba[1]*100:.1f}%")
                        st.progress(int(proba[1]*100))
                        st.write(f"Probabilitas Risiko Rendah: {proba[0]*100:.1f}%")
                    
                    st.markdown("---")
                    st.markdown("### Skor Risiko")
                    risk_score = proba[1] * 100
                    st.progress(int(risk_score))
                    st.caption(f"Skor risiko Anda: {risk_score:.1f} dari 100")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
        else:
            st.error("Model belum tersedia.")
    
    st.markdown('<div class="footer">2026 Pendeteksi Risiko Diabetes</div>', unsafe_allow_html=True)