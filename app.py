import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from database import *

st.set_page_config(
    page_title="Pendeteksi Risiko Diabetes",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

create_database()

st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    }
    
    /* Sidebar styling - selaras dengan tema hijau */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2e7d32 0%, #1b5e20 100%);
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stButton button {
        background-color: rgba(255,255,255,0.2);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3);
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(255,255,255,0.3);
        border: 1px solid white;
    }
    
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.3);
    }
    
    /* Styling untuk selectbox (dropdown) di halaman kelola user */
    .stSelectbox label {
        color: #1b5e20 !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #a5d6a7;
    }
    
    .stSelectbox div[data-baseweb="select"] div {
        color: #000000 !important;
    }
    
    /* Styling untuk option di dropdown */
    div[role="listbox"] div {
        color: #000000 !important;
        background-color: white !important;
    }
    
    div[role="listbox"] div:hover {
        background-color: #e8f5e9 !important;
    }
    
    /* Styling untuk input text di kelola user */
    .stTextInput input {
        border-radius: 10px;
        border: 1px solid #a5d6a7;
        padding: 0.8rem;
        font-size: 0.9rem;
        background-color: white !important;
        color: #000000 !important;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #1b5e20;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    
    h1, h2, h3 {
        color: #1b5e20;
        font-weight: 700;
    }
    
    p, li, span, .stMarkdown, .stCaption, .stAlert, label {
        color: #000000 !important;
        font-size: 0.85rem;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 30px;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46,125,50,0.4);
    }
    
    .login-card {
        background-color: transparent;
        padding: 1rem;
        margin-top: 0rem;
    }
    
    .stNumberInput input {
        border-radius: 10px;
        border: 1px solid #a5d6a7;
        padding: 0.6rem 0.8rem;
        font-size: 0.9rem;
        background-color: white;
        color: #000000;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .hero-section .hero-title,
    .hero-section .hero-subtitle {
        color: white !important;
    }
    
    .hero-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .hero-subtitle {
        font-size: 0.9rem;
        opacity: 0.95;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        margin: 1.5rem 0 1rem 0;
        color: #1b5e20;
    }
    
    .step-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(46,125,50,0.15);
        border: 1px solid #a5d6a7;
        height: 100%;
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .step-number {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.8rem auto;
        font-size: 1.1rem;
        font-weight: bold;
        color: white;
    }
    
    .step-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #1b5e20;
    }
    
    .step-description {
        font-size: 0.85rem;
        color: #000000;
        line-height: 1.4;
    }
    
    .security-banner {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .security-banner .security-title,
    .security-banner .security-text {
        color: white !important;
    }
    
    .security-title {
        font-size: 1rem;
        font-weight: 700;
    }
    
    .security-text {
        font-size: 0.85rem;
        opacity: 0.95;
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #a5d6a7, transparent);
        margin: 1.5rem 0;
    }
    
    .result-low {
        background-color: #e8f5e9;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid #2e7d32;
    }
    
    .result-low .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1b5e20;
    }
    
    .result-low .result-text {
        font-size: 0.9rem;
        color: #000000;
    }
    
    .result-high {
        background-color: #ffebee;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid #c62828;
    }
    
    .result-high .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #c62828;
    }
    
    .result-high .result-text {
        font-size: 0.9rem;
        color: #000000;
    }
    
    .footer {
        text-align: center;
        padding: 1.2rem;
        color: #666666;
        font-size: 0.75rem;
        border-top: 1px solid #c8e6c9;
        margin-top: 2rem;
    }
    
    .stProgress > div > div {
        background-color: #2e7d32;
    }
    
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #e8f5e9;
        border-radius: 8px;
        color: #1b5e20;
        font-size: 0.9rem;
    }
    
    .stDataFrame {
        background-color: white;
        border-radius: 16px;
        padding: 0.5rem;
        border: 1px solid #a5d6a7;
        box-shadow: 0 2px 10px rgba(46,125,50,0.1);
    }
    
    .stMetric label {
        color: #000000 !important;
        font-size: 0.85rem;
    }
    
    .stMetric .metric-value {
        color: #1b5e20 !important;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .stCaption {
        color: #666666 !important;
        font-size: 0.7rem;
    }
    
    input {
        color: #000000 !important;
    }
    
    .stAlert {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        border-radius: 10px;
    }
    
    .stAlert p {
        color: #000000 !important;
    }
    
    /* Styling untuk tombol di kelola user */
    .stButton button {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_nama" not in st.session_state:
    st.session_state.user_nama = ""
if "user_role" not in st.session_state:
    st.session_state.user_role = ""
if "page" not in st.session_state:
    st.session_state.page = "login"

@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_knn.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

def do_login(username, password):
    user = check_login(username, password)
    if user:
        st.session_state.logged_in = True
        st.session_state.user_id = user[0]
        st.session_state.username = user[1]
        st.session_state.user_nama = user[2] if user[2] else user[1]
        st.session_state.user_role = user[4]
        st.session_state.page = "home"
        return True
    return False

def do_logout():
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = ""
    st.session_state.user_nama = ""
    st.session_state.user_role = ""
    st.session_state.page = "login"
    st.rerun()

def login_page():
    st.markdown('<h1 class="main-title">Pendeteksi Risiko Diabetes</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Sistem Prediksi Risiko Diabetes Berbasis AI</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Deteksi Dini, Hidup Lebih Sehat</div>
        <div class="hero-subtitle">Gunakan kecerdasan buatan untuk mengetahui risiko diabetes Anda</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align:center; color:#1b5e20; margin-bottom:1.5rem;">Login ke Akun Anda</h3>', unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Masukkan username", key="login_username")
        password = st.text_input("Password", type="password", placeholder="Masukkan password", key="login_password")
        
        if st.button("Login", use_container_width=True, key="login_btn"):
            if do_login(username, password):
                st.rerun()
            else:
                st.error("Username atau password salah")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
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

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h3 style="color: white; margin-bottom: 0.3rem;">{st.session_state.user_nama}</h3>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">@{st.session_state.username}</p>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.75rem;">Role: {st.session_state.user_role}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        if st.button("Beranda", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        
        if st.button("Prediksi", use_container_width=True):
            st.session_state.page = "prediksi"
            st.rerun()
        
        if st.button("Riwayat", use_container_width=True):
            st.session_state.page = "riwayat"
            st.rerun()
        
        if st.session_state.user_role == "admin":
            if st.button("Kelola User", use_container_width=True):
                st.session_state.page = "kelola_user"
                st.rerun()
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        if st.button("Logout", use_container_width=True):
            do_logout()

def home_page():
    render_sidebar()
    
    st.markdown(f'<h1 class="main-title">Halo, {st.session_state.user_nama}</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Deteksi Dini, Hidup Lebih Sehat</div>
        <div class="hero-subtitle">Gunakan kecerdasan buatan untuk mengetahui risiko diabetes Anda</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Mulai Prediksi", use_container_width=True):
            st.session_state.page = "prediksi"
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

def prediksi_page():
    render_sidebar()
    
    st.markdown('<h1 class="main-title">Formulir Data Kesehatan</h1>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<h3 style="color: #1b5e20; margin-bottom: 1rem;">Data Dasar</h3>', unsafe_allow_html=True)
        
        st.markdown('<label style="font-size:0.85rem; font-weight:600; color:#000000;">Usia (Tahun)</label>', unsafe_allow_html=True)
        usia = st.number_input("usia", min_value=0, max_value=120, value=45, step=1, format="%d", key="usia_input")
        
        st.markdown('<label style="font-size:0.85rem; font-weight:600; color:#000000; margin-top:0.8rem; display:block;">BMI (Body Mass Index)</label>', unsafe_allow_html=True)
        bmi = st.number_input("bmi", min_value=0.0, max_value=60.0, value=30.0, step=0.1, format="%.1f", key="bmi_input")
        
        st.markdown('<label style="font-size:0.85rem; font-weight:600; color:#000000; margin-top:0.8rem; display:block;">Jumlah Kehamilan</label>', unsafe_allow_html=True)
        kehamilan = st.number_input("kehamilan", min_value=0, max_value=20, value=2, step=1, format="%d", key="kehamilan_input")
        
        st.markdown('<label style="font-size:0.85rem; font-weight:600; color:#000000; margin-top:0.8rem; display:block;">Ketebalan Kulit (mm)</label>', unsafe_allow_html=True)
        kulit = st.number_input("kulit", min_value=0, max_value=100, value=25, step=1, format="%d", key="kulit_input")
    
    with col_right:
        st.markdown('<h3 style="color: #1b5e20; margin-bottom: 1rem;">Data Medis</h3>', unsafe_allow_html=True)
        
        st.markdown('<label style="font-size:0.85rem; font-weight:600; color:#000000;">Kadar Glukosa (mg/dL)</label>', unsafe_allow_html=True)
        glukosa = st.number_input("glukosa", min_value=0, max_value=300, value=150, step=1, format="%d", key="glukosa_input")
        
        st.markdown('<label style="font-size:0.85rem; font-weight:600; color:#000000; margin-top:0.8rem; display:block;">Tekanan Darah (mm Hg)</label>', unsafe_allow_html=True)
        tekanan_darah = st.number_input("tekanan", min_value=0, max_value=200, value=85, step=1, format="%d", key="tekanan_input")
        
        st.markdown('<label style="font-size:0.85rem; font-weight:600; color:#000000; margin-top:0.8rem; display:block;">Kadar Insulin (µU/ml)</label>', unsafe_allow_html=True)
        insulin = st.number_input("insulin", min_value=0, max_value=900, value=100, step=1, format="%d", key="insulin_input")
        
        st.markdown('<label style="font-size:0.85rem; font-weight:600; color:#000000; margin-top:0.8rem; display:block;">Diabetes Pedigree Function</label>', unsafe_allow_html=True)
        dpf = st.number_input("dpf", min_value=0.0, max_value=2.5, value=0.50, step=0.01, format="%.2f", key="dpf_input")
    
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
                
                data_prediksi = {
                    'usia': usia, 'bmi': bmi, 'kehamilan': kehamilan, 'kulit': kulit,
                    'glukosa': glukosa, 'tekanan_darah': tekanan_darah, 'insulin': insulin, 'dpf': dpf
                }
                hasil = "Positif" if pred == 1 else "Negatif"
                save_prediction(st.session_state.user_id, st.session_state.username, data_prediksi, hasil, proba)
                
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
                    st.markdown("<h3 style='color: #1b5e20;'>Tingkat Keyakinan</h3>", unsafe_allow_html=True)
                    if pred == 0:
                        st.write(f"Probabilitas Risiko Rendah: {proba[0]*100:.1f}%")
                        st.progress(int(proba[0]*100))
                        st.write(f"Probabilitas Risiko Tinggi: {proba[1]*100:.1f}%")
                    else:
                        st.write(f"Probabilitas Risiko Tinggi: {proba[1]*100:.1f}%")
                        st.progress(int(proba[1]*100))
                        st.write(f"Probabilitas Risiko Rendah: {proba[0]*100:.1f}%")
                    
                    st.markdown("---")
                    st.markdown("<h3 style='color: #1b5e20;'>Skor Risiko</h3>", unsafe_allow_html=True)
                    risk_score = proba[1] * 100
                    st.progress(int(risk_score))
                    st.caption(f"Skor risiko Anda: {risk_score:.1f} dari 100")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
        else:
            st.error("Model belum tersedia. Silakan jalankan python train_model.py")
    
    st.markdown('<div class="footer">2026 Pendeteksi Risiko Diabetes</div>', unsafe_allow_html=True)

def riwayat_page():
    render_sidebar()
    
    st.markdown('<h1 class="main-title">Riwayat Prediksi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Daftar riwayat prediksi risiko diabetes Anda</p>', unsafe_allow_html=True)
    
    history = get_user_history(st.session_state.user_id, limit=20)
    
    if history:
        data = []
        for h in history:
            tanggal = h[1]
            if isinstance(tanggal, str):
                tanggal = tanggal[:16] if len(tanggal) > 16 else tanggal
            
            data.append({
                "Tanggal": tanggal,
                "Hasil": h[2],
                "Probabilitas": f"{h[3]*100:.1f}%",
                "Glukosa": h[4],
                "BMI": h[5],
                "Usia": h[6]
            })
        
        df_history = pd.DataFrame(data)
        
        st.dataframe(
            df_history,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Tanggal": st.column_config.TextColumn("Tanggal", width="medium"),
                "Hasil": st.column_config.TextColumn("Hasil", width="small"),
                "Probabilitas": st.column_config.TextColumn("Probabilitas", width="small"),
                "Glukosa": st.column_config.NumberColumn("Glukosa", width="small"),
                "BMI": st.column_config.NumberColumn("BMI", width="small"),
                "Usia": st.column_config.NumberColumn("Usia", width="small")
            }
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Ringkasan Statistik")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_prediksi = len(history)
            st.metric("Total Prediksi", total_prediksi)
        with col2:
            positif_count = len([h for h in history if h[2] == "Positif"])
            st.metric("Hasil Positif", positif_count)
        with col3:
            negatif_count = len([h for h in history if h[2] == "Negatif"])
            st.metric("Hasil Negatif", negatif_count)
    else:
        st.info("Belum ada riwayat prediksi. Silakan lakukan prediksi terlebih dahulu dengan mengklik menu Prediksi.")
    
    st.markdown('<div class="footer">2026 Pendeteksi Risiko Diabetes</div>', unsafe_allow_html=True)

def kelola_user_page():
    render_sidebar()
    
    st.markdown('<h1 class="main-title">Kelola Pengguna</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Kelola data pengguna aplikasi</p>', unsafe_allow_html=True)
    
    with st.expander("Tambah Pengguna Baru", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            new_username = st.text_input("Username", key="new_username")
            new_password = st.text_input("Password", type="password", key="new_password")
        with col2:
            new_nama = st.text_input("Nama Lengkap", key="new_nama")
            new_email = st.text_input("Email", key="new_email")
            new_role = st.selectbox("Role", ["user", "dokter", "admin"], key="new_role")
        
        if st.button("Tambah Pengguna", key="add_user_btn"):
            if new_username and new_password:
                if add_user(new_username, new_password, new_nama, new_email, new_role):
                    st.success(f"User {new_username} berhasil ditambahkan")
                    st.rerun()
                else:
                    st.error("Username sudah ada")
            else:
                st.warning("Username dan password harus diisi")
    
    st.markdown("### Daftar Pengguna")
    users = get_all_users()
    
    if users:
        data = []
        for u in users:
            data.append({
                "ID": u[0],
                "Username": u[1],
                "Nama": u[2] if u[2] else "-",
                "Email": u[3] if u[3] else "-",
                "Role": u[4],
                "Bergabung": u[5][:16] if u[5] else "-",
                "Terakhir Login": u[6][:16] if u[6] else "-"
            })
        df_users = pd.DataFrame(data)
        st.dataframe(df_users, use_container_width=True, hide_index=True)
        
        st.markdown("### Hapus Pengguna")
        user_options = [f"{u[1]} (ID: {u[0]})" for u in users if u[1] != st.session_state.username]
        if user_options:
            selected_user = st.selectbox(
                "Pilih pengguna yang akan dihapus", 
                user_options,
                key="delete_user_select"
            )
            user_id = int(selected_user.split("ID: ")[1].replace(")", ""))
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("Hapus Pengguna", key="delete_user_btn"):
                    delete_user(user_id)
                    st.success("Pengguna berhasil dihapus")
                    st.rerun()
        else:
            st.info("Tidak ada pengguna lain yang dapat dihapus")
    
    st.markdown('<div class="footer">2026 Pendeteksi Risiko Diabetes</div>', unsafe_allow_html=True)

def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.page == "home":
            home_page()
        elif st.session_state.page == "prediksi":
            prediksi_page()
        elif st.session_state.page == "kelola_user":
            kelola_user_page()
        elif st.session_state.page == "riwayat":
            riwayat_page()
        else:
            home_page()

if __name__ == "__main__":
    main()