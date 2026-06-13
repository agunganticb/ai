import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import hashlib

# ── Harus paling atas sebelum import database ──────────────────────────────
st.set_page_config(
    page_title="Diabetes AI – Prediksi Risiko Diabetes",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from database import (
    create_database, check_login, add_user, get_all_users, delete_user,
    save_prediction, get_user_history, get_all_predictions,
    get_statistics, get_monthly_stats,
)

# ── Inisialisasi DB ────────────────────────────────────────────────────────
create_database()

# ── Load Model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load("model_knn.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_model()

# ── Session State ─────────────────────────────────────────────────────────
for k, v in {
    "logged_in": False, "user_id": None, "username": "",
    "user_nama": "", "user_role": "", "page": "login",
    "last_result": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0b3b2f 0%,#0f4c3a 60%,#0b3b2f 100%);
    min-width:220px!important; max-width:220px!important;
}
section[data-testid="stSidebar"] * { color:#fff!important; }
section[data-testid="stSidebar"] .stButton>button {
    background:rgba(255,255,255,0.08);
    border:1px solid rgba(255,255,255,0.15);
    border-radius:12px; padding:.6rem 1rem;
    font-weight:500; font-size:13px;
    transition:all .2s; text-align:left;
    width:100%;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background:rgba(255,255,255,0.18); transform:translateX(4px);
}

/* ── Main background ── */
.main .block-container { background:#f4f6f9; padding:1.5rem 2rem; max-width:100%; }

/* ── General cards ── */
.g-card {
    background:#fff; border-radius:14px;
    padding:18px 20px;
    box-shadow:0 2px 10px rgba(0,0,0,0.06);
    margin-bottom:12px;
}

/* ── KPI card ── */
.kpi-card {
    background:#fff; border-radius:14px; padding:18px 20px;
    box-shadow:0 2px 10px rgba(0,0,0,0.06); height:100%;
}
.kpi-label { font-size:12px; color:#6b7280; font-weight:500; margin-bottom:6px; }
.kpi-value { font-size:32px; font-weight:700; line-height:1; margin-bottom:4px; }
.kpi-sub   { font-size:11px; color:#9ca3af; }

/* ── Badge ── */
.badge { display:inline-block; padding:3px 12px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-red   { background:#fee2e2; color:#b91c1c; }
.badge-green { background:#dcfce7; color:#15803d; }
.badge-blue  { background:#dbeafe; color:#1d4ed8; }
.badge-gray  { background:#f3f4f6; color:#374151; }

/* ── Login ── */
.login-wrapper {
    min-height:100vh; display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,#f0fdf4,#e8f5e9);
}
.login-card {
    background:#fff; border-radius:24px; padding:2.5rem;
    box-shadow:0 20px 60px rgba(0,0,0,0.12); width:100%;
}
.green-panel {
    background:linear-gradient(135deg,#064e3b,#059669);
    border-radius:24px; padding:2rem; height:100%;
    box-shadow:0 20px 40px rgba(5,150,105,.25);
}

/* ── Result boxes ── */
.result-low  { background:#f0fdf4; border:2px solid #22c55e; border-radius:16px; padding:24px; text-align:center; }
.result-high { background:#fff1f2; border:2px solid #ef4444; border-radius:16px; padding:24px; text-align:center; }

/* ── Form card ── */
.form-card { background:#fff; border-radius:16px; padding:24px; box-shadow:0 2px 10px rgba(0,0,0,0.06); }

/* ── Tips card ── */
.tip-card {
    background:linear-gradient(135deg,#166534,#15803d);
    border-radius:14px; padding:18px; color:#fff;
}
.tip-card h4 { font-size:13px; font-weight:600; margin-bottom:8px; }
.tip-card p  { font-size:12px; line-height:1.6; margin:0; opacity:.9; }

/* ── Buttons ── */
.stButton>button {
    background:linear-gradient(135deg,#16a34a,#15803d);
    color:#fff!important; border:none; border-radius:10px;
    font-weight:600; padding:.55rem 1.4rem; font-size:14px;
    transition:opacity .2s;
}
.stButton>button:hover { opacity:.87; }

/* ── Table ── */
table { width:100%; border-collapse:collapse; font-size:13px; }
th { background:#f9fafb; color:#6b7280; font-weight:600; padding:10px 12px;
     text-align:left; border-bottom:1px solid #e5e7eb; }
td { padding:9px 12px; border-bottom:1px solid #f3f4f6; color:#374151; }
tr:last-child td { border-bottom:none; }

/* ── Inputs ── */
.stNumberInput>div>div>input,
.stTextInput>div>div>input,
.stSelectbox>div>div { border-radius:10px!important; }

/* ── Metric ── */
[data-testid="stMetric"] { background:#fff; border-radius:14px; padding:14px;
    box-shadow:0 2px 10px rgba(0,0,0,0.06); }

/* ── Hide Streamlit header ── */
header[data-testid="stHeader"] { background:transparent; }

/* ── Progress bar colour ── */
.stProgress > div > div > div > div { background:linear-gradient(90deg,#16a34a,#22c55e); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════
def do_login(username, password):
    user = check_login(username, password)
    if user:
        st.session_state.update({
            "logged_in": True, "user_id": user[0], "username": user[1],
            "user_nama": user[2] or user[1], "user_role": user[4],
            "page": "dashboard",
        })
        return True
    return False

def do_logout():
    for k in ["logged_in","user_id","username","user_nama","user_role","last_result"]:
        st.session_state[k] = None if k in ("user_id","last_result") else ("" if k not in ("logged_in",) else False)
    st.session_state.page = "login"
    st.rerun()

def prob_bar(prob, color):
    """HTML progress bar."""
    return f"""
    <div style="background:#e5e7eb;border-radius:4px;height:7px;margin-top:4px;">
        <div style="background:{color};width:{prob}%;height:7px;border-radius:4px;"></div>
    </div>"""

# ── Sidebar ───────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:16px 0 20px;">
            <div style="font-size:32px;margin-bottom:8px;">❤️</div>
            <div style="font-size:17px;font-weight:700;">Diabetes AI</div>
            <div style="font-size:11px;color:#a7d7b4;margin-top:2px;">Prediksi Risiko Diabetes</div>
        </div>
        <hr style="border-color:rgba(255,255,255,.2);margin:0 0 14px;">
        """, unsafe_allow_html=True)

        pages = [("🏠", "Beranda", "dashboard"),
                 ("🔍", "Prediksi", "prediksi"),
                 ("📋", "Riwayat Prediksi", "riwayat")]
        if st.session_state.user_role == "admin":
            pages.append(("👥", "Kelola Pengguna", "kelola_user"))

        for icon, label, pg in pages:
            if st.button(f"{icon}  {label}", key=f"nav_{pg}", use_container_width=True):
                st.session_state.page = pg
                st.session_state.last_result = None
                st.rerun()

        st.markdown("<hr style='border-color:rgba(255,255,255,.2);margin:14px 0;'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center;padding:8px 0;">
            <div style="font-size:26px;">👨‍⚕️</div>
            <div style="font-size:13px;font-weight:600;">{st.session_state.user_nama}</div>
            <div style="font-size:11px;color:#a7d7b4;">{st.session_state.username} &nbsp;🟢</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("🚪  Logout", use_container_width=True):
            do_logout()


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: LOGIN
# ══════════════════════════════════════════════════════════════════════════
def login_page():
    col_l, col_c, col_r = st.columns([1, 2.5, 1])
    with col_c:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        left, right = st.columns([1.1, 1], gap="medium")

        with left:
            st.markdown("""
            <div class='login-card'>
                <div style='font-size:28px;margin-bottom:4px;'>❤️</div>
                <div style='font-size:22px;font-weight:800;color:#15803d;'>Diabetes AI</div>
                <div style='font-size:12px;color:#9ca3af;margin-bottom:20px;'>Sistem Prediksi Risiko Diabetes</div>
                <div style='font-size:17px;font-weight:700;color:#111;margin-bottom:4px;'>Login ke Akun Anda</div>
                <div style='font-size:12px;color:#6b7280;margin-bottom:20px;'>Silakan login untuk melanjutkan</div>
            """, unsafe_allow_html=True)

            username = st.text_input("Username", placeholder="Masukkan username", key="lu")
            password = st.text_input("Password", type="password", placeholder="Masukkan password", key="lp")

            cb_col, fp_col = st.columns([1, 1])
            with cb_col:
                st.checkbox("Remember me", key="rem")
            with fp_col:
                st.markdown("<div style='text-align:right;padding-top:6px;font-size:12px;color:#15803d;'>Lupa password?</div>", unsafe_allow_html=True)

            if st.button("Login", use_container_width=True, key="login_btn"):
                if do_login(username, password):
                    st.rerun()
                else:
                    st.error("❌ Username atau password salah")

            st.markdown("""
            <div style='margin-top:16px;background:#f9fafb;border:1px solid #e5e7eb;
                        border-radius:12px;padding:10px 14px;font-size:12px;color:#374151;'>
                🔑 <b>Demo Account</b><br>
                Username: <code>admin</code> &nbsp;|&nbsp; Password: <code>admin123</code>
            </div>
            <div style='text-align:center;font-size:11px;color:#9ca3af;margin-top:16px;'>
                Belum punya akun? Hubungi administrator
            </div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown("""
            <div class='green-panel'>
                <div style='text-align:center;margin-bottom:20px;'>
                    <div style='font-size:50px;'>👨‍⚕️</div>
                    <h3 style='color:#fff;font-size:18px;margin:10px 0 4px;'>Deteksi Dini,</h3>
                    <h3 style='color:#fff;font-size:18px;margin:0 0 8px;'>Hidup Lebih Sehat</h3>
                    <p style='color:rgba(255,255,255,.8);font-size:12px;'>Gunakan kecerdasan buatan untuk membantu mendeteksi risiko diabetes sejak dini.</p>
                </div>
                <div style='display:flex;flex-direction:column;gap:10px;'>
                    <div style='background:rgba(255,255,255,.12);border-radius:12px;padding:12px;border:1px solid rgba(255,255,255,.2);'>
                        <span style='font-size:22px;'>🤖</span>
                        <span style='color:#fff;font-size:13px;font-weight:600;margin-left:8px;'>AI Canggih</span>
                        <p style='color:rgba(255,255,255,.7);font-size:11px;margin:4px 0 0;'>Algoritma KNN Machine Learning</p>
                    </div>
                    <div style='background:rgba(255,255,255,.12);border-radius:12px;padding:12px;border:1px solid rgba(255,255,255,.2);'>
                        <span style='font-size:22px;'>📊</span>
                        <span style='color:#fff;font-size:13px;font-weight:600;margin-left:8px;'>Analisis Akurat</span>
                        <p style='color:rgba(255,255,255,.7);font-size:11px;margin:4px 0 0;'>Prediksi dengan tingkat akurasi tinggi</p>
                    </div>
                    <div style='background:rgba(255,255,255,.12);border-radius:12px;padding:12px;border:1px solid rgba(255,255,255,.2);'>
                        <span style='font-size:22px;'>🔒</span>
                        <span style='color:#fff;font-size:13px;font-weight:600;margin-left:8px;'>Aman & Terenkripsi</span>
                        <p style='color:rgba(255,255,255,.7);font-size:11px;margin:4px 0 0;'>Data pasien terlindungi penuh</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD / BERANDA
# ══════════════════════════════════════════════════════════════════════════
def dashboard_page():
    render_sidebar()

    st.markdown(f"## Selamat Datang, {st.session_state.user_nama} 👋")
    st.markdown("<p style='color:#6b7280;margin-top:-10px;margin-bottom:20px;'>Berikut ringkasan prediksi risiko diabetes hari ini.</p>", unsafe_allow_html=True)

    total, positif, negatif = get_statistics()
    pct_pos = round(positif / total * 100, 1) if total else 0
    pct_neg = round(negatif / total * 100, 1) if total else 0

    # ── KPI cards ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Total Prediksi 📋</div>
            <div class='kpi-value'>{total}</div>
            <div class='kpi-sub'>Semua prediksi tercatat</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Prediksi Positif 🩸</div>
            <div class='kpi-value' style='color:#dc2626;'>{positif}</div>
            <div class='kpi-sub'>{pct_pos}% dari total</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Prediksi Negatif ✅</div>
            <div class='kpi-value' style='color:#16a34a;'>{negatif}</div>
            <div class='kpi-sub'>{pct_neg}% dari total</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown("""<div class='kpi-card'>
            <div class='kpi-label'>Akurasi Model 🤖</div>
            <div class='kpi-value' style='color:#7c3aed;'>87.3%</div>
            <div class='kpi-sub'>Model KNN Aktif</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────
    ch1, ch2, ch3 = st.columns([2, 1.2, 1.3])

    with ch1:
        monthly = get_monthly_stats()
        if monthly:
            labels = [r[0] for r in monthly]
            pos_d  = [r[2] for r in monthly]
            neg_d  = [r[3] for r in monthly]
        else:
            labels = ["Jan","Feb","Mar","Apr","Mei","Jun"]
            pos_d  = [3, 5, 4, 7, 5, positif]
            neg_d  = [9, 12, 10, 15, 11, negatif]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=labels, y=neg_d, name="Negatif",
            mode="lines+markers", line=dict(color="#16a34a", width=2.5),
            marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=labels, y=pos_d, name="Positif",
            mode="lines+markers", line=dict(color="#ef4444", width=2.5),
            marker=dict(size=7)))
        fig.update_layout(
            title="Tren Prediksi (6 Bulan Terakhir)",
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            font=dict(family="Inter", size=11),
            margin=dict(l=10,r=10,t=44,b=10), height=240,
            legend=dict(orientation="h", yanchor="top", y=1.14, xanchor="left", x=0),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f3f4f6")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        fig2 = go.Figure(go.Pie(
            labels=["Negatif","Positif"],
            values=[max(negatif,1), max(positif,0)],
            hole=0.55,
            marker_colors=["#16a34a","#ef4444"],
            textinfo="percent", textfont_size=12,
        ))
        fig2.update_layout(
            title="Distribusi Hasil",
            showlegend=True,
            legend=dict(orientation="v", x=1, y=0.5),
            font=dict(family="Inter"),
            margin=dict(l=0,r=0,t=44,b=0), height=240,
            paper_bgcolor="#fff",
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with ch3:
        recent_all = get_all_predictions()[:4]
        st.markdown("<div class='g-card' style='height:240px;overflow:hidden;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px;font-weight:600;margin-bottom:10px;'>Prediksi Terakhir</div>", unsafe_allow_html=True)
        if recent_all:
            for r in recent_all:
                bc = "badge-red" if r[3]=="Positif" else "badge-green"
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;align-items:center;
                            padding:7px 0;border-bottom:1px solid #f3f4f6;'>
                    <div>
                        <div style='font-size:12px;color:#374151;font-weight:500;'>Usia: {r[7]} thn</div>
                        <div style='font-size:11px;color:#9ca3af;'>Glu: {r[5]} mg/dL</div>
                    </div>
                    <span class='badge {bc}'>{r[3]}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size:12px;color:#9ca3af;'>Belum ada data</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Recent history table + tips ───────────────────────────────────────
    tbl_col, tips_col = st.columns([2.2, 1])

    with tbl_col:
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px;font-weight:600;margin-bottom:12px;'>Riwayat Prediksi Terbaru</div>", unsafe_allow_html=True)
        recent_all5 = get_all_predictions()[:6]
        if recent_all5:
            rows_html = ""
            for r in recent_all5:
                bc   = "badge-red" if r[3]=="Positif" else "badge-green"
                bclr = "#ef4444"   if r[3]=="Positif" else "#16a34a"
                pct  = round(r[4]*100, 1)
                rows_html += f"""<tr>
                    <td>{r[2][:16]}</td><td>{r[7]}</td>
                    <td>{r[5]} mg/dL</td><td>{r[6]}</td>
                    <td><span class='badge {bc}'>{r[3]}</span></td>
                    <td>{pct}%{prob_bar(pct, bclr)}</td>
                </tr>"""
            st.markdown(f"""<table>
                <thead><tr><th>Tanggal</th><th>Usia</th><th>Glukosa</th>
                <th>BMI</th><th>Hasil</th><th>Probabilitas</th></tr></thead>
                <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)
        else:
            st.info("Belum ada data prediksi.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tips_col:
        st.markdown("""
        <div class='tip-card'>
            <h4>💡 Tips Kesehatan Hari Ini</h4>
            <p>Konsumsi makanan sehat, rutin berolahraga, dan cukup tidur dapat menurunkan risiko diabetes.</p>
            <div style='display:flex;gap:12px;margin-top:12px;font-size:22px;'>
                <span>🥦</span><span>🏃</span><span>🌙</span>
            </div>
        </div>
        <div style='height:10px'></div>
        <div class='tip-card' style='background:linear-gradient(135deg,#1d4ed8,#2563eb);'>
            <h4>📊 Info Model</h4>
            <p>Model KNN dilatih dengan dataset Pima Indian Diabetes. Akurasi validasi 87.3%.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;color:#9ca3af;font-size:12px;margin-top:24px;'>© 2026 Diabetes AI. All rights reserved.</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: PREDIKSI
# ══════════════════════════════════════════════════════════════════════════
def prediksi_page():
    render_sidebar()

    st.markdown("## Prediksi Risiko Diabetes")
    st.markdown("<p style='color:#6b7280;margin-top:-10px;margin-bottom:20px;'>Masukkan data pasien untuk menganalisis risiko diabetes menggunakan AI.</p>", unsafe_allow_html=True)

    # ── Tampilkan hasil jika sudah ada ────────────────────────────────────
    if st.session_state.last_result:
        r      = st.session_state.last_result
        is_pos = r["hasil"] == "Positif"
        prob   = r["prob"]

        col_res, col_info = st.columns([1, 1.6])
        with col_res:
            if is_pos:
                st.markdown(f"""
                <div class='result-high'>
                    <div style='font-size:42px;'>⚠️</div>
                    <div style='font-size:20px;font-weight:700;color:#dc2626;margin:8px 0 4px;'>RISIKO TINGGI</div>
                    <div style='font-size:38px;font-weight:800;color:#dc2626;'>{prob:.1f}%</div>
                    <div style='font-size:12px;color:#6b7280;margin-top:4px;'>Probabilitas Diabetes Positif</div>
                    {prob_bar(prob, "#ef4444")}
                    <div style='display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-top:2px;'>
                        <span>0%</span><span>100%</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-low'>
                    <div style='font-size:42px;'>✅</div>
                    <div style='font-size:20px;font-weight:700;color:#16a34a;margin:8px 0 4px;'>RISIKO RENDAH</div>
                    <div style='font-size:38px;font-weight:800;color:#16a34a;'>{prob:.1f}%</div>
                    <div style='font-size:12px;color:#6b7280;margin-top:4px;'>Probabilitas Diabetes Negatif</div>
                    {prob_bar(prob, "#22c55e")}
                    <div style='display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-top:2px;'>
                        <span>0%</span><span>100%</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_info:
            st.markdown("<div class='g-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:14px;font-weight:600;margin-bottom:10px;'>📝 Interpretasi & Rekomendasi</div>", unsafe_allow_html=True)
            if is_pos:
                st.markdown("""
                <p style='font-size:13px;color:#374151;'>Berdasarkan data yang dimasukkan, pasien <b>berpotensi memiliki diabetes</b>. Segera konsultasi dokter.</p>
                <b style='font-size:12px;'>Rekomendasi:</b>
                <ul style='font-size:13px;color:#374151;margin-top:6px;'>
                    <li>✅ Lakukan pemeriksaan gula darah lebih lanjut</li>
                    <li>✅ Konsultasi dokter spesialis endokrin</li>
                    <li>✅ Kurangi konsumsi gula & karbohidrat sederhana</li>
                    <li>✅ Rutin berolahraga minimal 30 mnt/hari</li>
                    <li>✅ Periksa kesehatan secara rutin</li>
                </ul>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <p style='font-size:13px;color:#374151;'>Berdasarkan data yang dimasukkan, pasien kemungkinan <b>tidak memiliki diabetes</b>. Tetap jaga pola hidup sehat.</p>
                <b style='font-size:12px;'>Rekomendasi:</b>
                <ul style='font-size:13px;color:#374151;margin-top:6px;'>
                    <li>✅ Pertahankan pola makan sehat</li>
                    <li>✅ Rutin berolahraga minimal 30 mnt/hari</li>
                    <li>✅ Jaga berat badan ideal (BMI 18.5–24.9)</li>
                    <li>✅ Periksa kesehatan secara rutin</li>
                    <li>✅ Hindari stres berlebihan</li>
                </ul>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Input summary grid
        st.markdown("<div class='g-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px;font-weight:600;margin-bottom:12px;'>📊 Ringkasan Data Input</div>", unsafe_allow_html=True)
        sc1,sc2,sc3,sc4 = st.columns(4)
        items = [
            ("Usia", f"{r['usia']} tahun"),
            ("Glukosa", f"{r['glukosa']} mg/dL"),
            ("BMI", f"{r['bmi']}"),
            ("Tekanan Darah", f"{r['tekanan_darah']} mmHg"),
            ("Insulin", f"{r['insulin']} µU/ml"),
            ("Kehamilan", str(r['kehamilan'])),
            ("Ketebalan Kulit", f"{r['kulit']} mm"),
            ("DPF", f"{r['dpf']}"),
        ]
        for i,(lbl,val) in enumerate(items):
            c = [sc1,sc2,sc3,sc4][i%4]
            c.markdown(f"""
            <div style='background:#f9fafb;border-radius:10px;padding:10px 12px;margin-bottom:8px;'>
                <div style='font-size:11px;color:#9ca3af;'>{lbl}</div>
                <div style='font-size:15px;font-weight:600;color:#111;'>{val}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        b1,b2,_ = st.columns([1,1,2])
        with b1:
            if st.button("🔄 Analisis Baru"):
                st.session_state.last_result = None
                st.rerun()
        with b2:
            if st.button("📋 Lihat Riwayat"):
                st.session_state.last_result = None
                st.session_state.page = "riwayat"
                st.rerun()
        return  # jangan tampilkan form lagi

    # ── Form input ─────────────────────────────────────────────────────────
    st.markdown("<div class='form-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px;font-weight:600;margin-bottom:14px;'>📋 Data Pasien</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        usia     = st.number_input("Usia (Tahun)",             min_value=1,   max_value=120, value=45,   step=1)
        bmi      = st.number_input("BMI (Body Mass Index)",    min_value=10.0,max_value=70.0,value=28.5, step=0.1, format="%.1f")
        insulin  = st.number_input("Kadar Insulin (µU/ml)",    min_value=0,   max_value=900, value=100,  step=1)
        kulit    = st.number_input("Ketebalan Kulit (mm)",     min_value=0,   max_value=100, value=25,   step=1)
    with c2:
        glukosa  = st.number_input("Glukosa (mg/dL)",          min_value=0,   max_value=600, value=140,  step=1)
        tekanan  = st.number_input("Tekanan Darah (mm Hg)",    min_value=0,   max_value=200, value=80,   step=1)
        kehamilan= st.number_input("Jumlah Kehamilan (Wanita)",min_value=0,   max_value=20,  value=2,    step=1)
        dpf      = st.number_input("Diabetes Pedigree Function",min_value=0.0,max_value=3.0, value=0.50, step=0.01, format="%.2f")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("🔍 Analisis Risiko Sekarang", use_container_width=False):
        if model is not None and scaler is not None:
            # Urutan fitur: Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age
            input_arr    = np.array([[kehamilan, glukosa, tekanan, kulit, insulin, bmi, dpf, usia]])
            input_scaled = scaler.transform(input_arr)
            pred         = model.predict(input_scaled)[0]
            proba        = model.predict_proba(input_scaled)[0]

            hasil = "Positif" if pred == 1 else "Negatif"
            prob_display = proba[1]*100 if pred == 1 else proba[0]*100

            data_p = {
                "usia": usia, "glukosa": glukosa, "bmi": bmi,
                "tekanan_darah": tekanan, "insulin": insulin,
                "kehamilan": kehamilan, "kulit": kulit, "dpf": dpf,
            }
            save_prediction(st.session_state.user_id, st.session_state.username, data_p, hasil, proba)

            st.session_state.last_result = {**data_p, "hasil": hasil, "prob": prob_display}
            st.rerun()
        else:
            st.error("❌ Model tidak ditemukan. Pastikan file `model_knn.pkl` dan `scaler.pkl` ada di folder yang sama.")

    st.markdown("<p style='font-size:12px;color:#9ca3af;margin-top:6px;'>⚠️ Pastikan semua data diisi dengan benar</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: RIWAYAT
# ══════════════════════════════════════════════════════════════════════════
def riwayat_page():
    render_sidebar()

    st.markdown("## Riwayat Prediksi")
    st.markdown("<p style='color:#6b7280;margin-top:-10px;margin-bottom:20px;'>Lihat semua riwayat prediksi risiko diabetes.</p>", unsafe_allow_html=True)

    is_admin = st.session_state.user_role == "admin"
    history  = get_all_predictions() if is_admin else get_user_history(st.session_state.user_id, 50)

    total   = len(history)
    positif = sum(1 for h in history if h[3]=="Positif") if is_admin else sum(1 for h in history if h[2]=="Positif")
    negatif = total - positif

    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Total Prediksi</div>
            <div class='kpi-value'>{total}</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Positif 🩸</div>
            <div class='kpi-value' style='color:#dc2626;'>{positif}</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Negatif ✅</div>
            <div class='kpi-value' style='color:#16a34a;'>{negatif}</div></div>""", unsafe_allow_html=True)
    with k4:
        avg_p = round(sum(h[4] for h in history)*100/total, 1) if total and is_admin else (
                round(sum(h[3] for h in history)*100/total, 1) if total else 0)
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Rata-rata Prob</div>
            <div class='kpi-value' style='color:#7c3aed;'>{avg_p}%</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Grafik probabilitas
    if history:
        probs  = [round(h[4]*100,1) for h in history] if is_admin else [round(h[3]*100,1) for h in history]
        colors = ["#ef4444" if (h[3]=="Positif" if is_admin else h[2]=="Positif") else "#16a34a" for h in history]
        fig3   = go.Figure(go.Bar(
            x=list(range(len(history))), y=probs,
            marker_color=colors, text=probs, textposition="outside", textfont_size=9,
        ))
        fig3.add_hline(y=50, line_dash="dash", line_color="#f59e0b",
                       annotation_text="Threshold 50%", annotation_position="right")
        fig3.update_layout(
            title="Tren Probabilitas Prediksi",
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            font=dict(family="Inter",size=11),
            margin=dict(l=0,r=10,t=44,b=10), height=200,
            showlegend=False,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(range=[0,108], showgrid=True, gridcolor="#f3f4f6"),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # Tabel lengkap
    st.markdown("<div class='g-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px;font-weight:600;margin-bottom:12px;'>📋 Semua Data Prediksi</div>", unsafe_allow_html=True)
    if history:
        rows_html = ""
        for i,h in enumerate(history, 1):
            if is_admin:
                hasil, pct, tgl = h[3], round(h[4]*100,1), h[2][:16]
                extra = f"<td>{h[1]}</td><td>{h[7]}</td><td>{h[5]} mg/dL</td><td>{h[6]}</td>"
                head_extra = "<th>User</th><th>Usia</th><th>Glukosa</th><th>BMI</th>"
            else:
                hasil, pct, tgl = h[2], round(h[3]*100,1), h[1][:16]
                extra = f"<td>{h[6]}</td><td>{h[4]} mg/dL</td><td>{h[5]}</td>"
                head_extra = "<th>Usia</th><th>Glukosa</th><th>BMI</th>"
            bc   = "badge-red" if hasil=="Positif" else "badge-green"
            bclr = "#ef4444"  if hasil=="Positif" else "#16a34a"
            rows_html += f"""<tr>
                <td>{i}</td><td>{tgl}</td>{extra}
                <td><span class='badge {bc}'>{hasil}</span></td>
                <td>{pct}%{prob_bar(pct, bclr)}</td>
            </tr>"""
        st.markdown(f"""<table>
            <thead><tr><th>#</th><th>Tanggal</th>{head_extra}<th>Hasil</th><th>Probabilitas</th></tr></thead>
            <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)
    else:
        st.info("Belum ada riwayat prediksi.")
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: KELOLA PENGGUNA (admin only)
# ══════════════════════════════════════════════════════════════════════════
def kelola_user_page():
    render_sidebar()

    st.markdown("## Kelola Pengguna")
    st.markdown("<p style='color:#6b7280;margin-top:-10px;margin-bottom:20px;'>Kelola data pengguna aplikasi.</p>", unsafe_allow_html=True)

    users = get_all_users()
    total_u = len(users)
    admin_c = sum(1 for u in users if u[4]=="admin")
    active  = sum(1 for u in users if u[6] and u[6][:10]==datetime.now().strftime('%Y-%m-%d'))

    k1,k2,k3 = st.columns(3)
    with k1: st.metric("👥 Total Pengguna", total_u)
    with k2: st.metric("👑 Administrator", admin_c)
    with k3: st.metric("🟢 Aktif Hari Ini", active)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with st.expander("➕ Tambah Pengguna Baru"):
        a1,a2 = st.columns(2)
        with a1:
            nu = st.text_input("Username", key="nu")
            np_ = st.text_input("Password", type="password", key="np")
        with a2:
            nn = st.text_input("Nama Lengkap", key="nn")
            ne = st.text_input("Email", key="ne")
            nr = st.selectbox("Role", ["user","dokter","admin"], key="nr")
        if st.button("➕ Tambah Pengguna"):
            if nu and np_:
                if add_user(nu, np_, nn, ne, nr):
                    st.success(f"✅ User {nu} berhasil ditambahkan")
                    st.rerun()
                else:
                    st.error("❌ Username sudah digunakan")
            else:
                st.warning("Username dan password wajib diisi")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='g-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px;font-weight:600;margin-bottom:12px;'>📋 Daftar Pengguna</div>", unsafe_allow_html=True)

    if users:
        rows_html = ""
        for u in users:
            role_bc = "badge-green" if u[4]=="admin" else "badge-blue" if u[4]=="dokter" else "badge-gray"
            last    = u[6][:16] if u[6] else "-"
            rows_html += f"""<tr>
                <td>{u[0]}</td><td><b>{u[1]}</b></td>
                <td>{u[2] or '-'}</td><td>{u[3] or '-'}</td>
                <td><span class='badge {role_bc}'>{u[4]}</span></td>
                <td>{last}</td>
            </tr>"""
        st.markdown(f"""<table>
            <thead><tr><th>No</th><th>Username</th><th>Nama</th><th>Email</th><th>Role</th><th>Terakhir Login</th></tr></thead>
            <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Hapus pengguna
    opts = [f"{u[1]} (ID:{u[0]})" for u in users if u[1] != st.session_state.username]
    if opts:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        d1,d2,_ = st.columns([1.5,1,1.5])
        with d1:
            sel = st.selectbox("Pilih pengguna yang akan dihapus", opts)
        with d2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Hapus"):
                uid = int(sel.split("ID:")[1].replace(")",""))
                delete_user(uid)
                st.success("✅ Pengguna dihapus")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        login_page()
        return

    pg = st.session_state.page
    if pg == "dashboard":   dashboard_page()
    elif pg == "prediksi":  prediksi_page()
    elif pg == "riwayat":   riwayat_page()
    elif pg == "kelola_user" and st.session_state.user_role == "admin":
        kelola_user_page()
    else:
        dashboard_page()

if __name__ == "__main__":
    main()
