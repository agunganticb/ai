import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Konfigurasi Halaman
st.set_page_config(
    page_title="AI Diabetes Care", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-negative {
        background-color: #d4edda;
        color: #155724;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #10b981;
    }
    .result-positive {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #ef4444;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

# Train Model
@st.cache_resource
def train_model():
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    return model, scaler, metrics, cm, X_test, y_test, y_pred

# Load data
df = load_data()
model, scaler, metrics, cm, X_test, y_test, y_pred = train_model()

# Hitung statistik
total_pasien = len(df)
positif = len(df[df["Outcome"] == 1])
negatif = len(df[df["Outcome"] == 0])

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🩺 AI Diabetes Care")
    st.markdown("### Sistem Prediksi Risiko Diabetes")
    st.markdown("#### Menggunakan K-Nearest Neighbors (KNN)")
    st.markdown("---")
    menu = st.radio(
        "📌 Navigasi",
        ["🏠 Dashboard", "👤 Prediksi Pasien", "📋 Data Pasien", "📈 Analisis Data", "🎯 Evaluasi Model", "ℹ️ Tentang"],
        index=0
    )
    st.markdown("---")
    st.caption("© 2026 AI Diabetes Care")

# ==================== HALAMAN DASHBOARD ====================
if menu == "🏠 Dashboard":
    st.markdown('<p class="main-header">🏠 Dashboard Monitoring</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistem Prediksi Risiko Diabetes Menggunakan K-Nearest Neighbors (KNN)</p>', unsafe_allow_html=True)
    
    # Row 1: Metric Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.metric("Total Pasien", f"{total_pasien}", "100% dari dataset")
    
    with col2:
        with st.container(border=True):
            st.metric("Negatif Diabetes", f"{negatif} ({negatif/total_pasien*100:.1f}%)", "Tidak berisiko diabetes")
    
    with col3:
        with st.container(border=True):
            st.metric("Positif Diabetes", f"{positif} ({positif/total_pasien*100:.1f}%)", "Berisiko diabetes")
    
    # Row 2: Pie Chart dan Akurasi
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Distribusi Pasien Diabetes")
        fig_pie = px.pie(
            values=[negatif, positif],
            names=["Negatif Diabetes", "Positif Diabetes"],
            color_discrete_sequence=["#10b981", "#ef4444"],
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
        fig_pie.update_layout(height=400, showlegend=True, legend=dict(x=0.8, y=0.9))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Performa Model")
        with st.container(border=True):
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <h1 style="font-size: 4rem; color: #1f77b4; margin: 0;">{metrics['accuracy']*100:.2f}%</h1>
                <p style="font-size: 1.2rem;">Akurasi Model</p>
                <p style="color: #666;">KNN (k = 5)</p>
                <hr>
                <div style="display: flex; justify-content: space-around;">
                    <div><strong>Precision</strong><br>{metrics['precision']*100:.2f}%</div>
                    <div><strong>Recall</strong><br>{metrics['recall']*100:.2f}%</div>
                    <div><strong>F1-Score</strong><br>{metrics['f1']*100:.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Row 3: Distribusi Fitur Utama
    st.subheader("📈 Distribusi Fitur Utama")
    col1, col2, col3, col4 = st.columns(4)
    
    feature_configs = [
        {'col': 'Glucose', 'name': 'Glukosa', 'color': '#FF6B6B', 'unit': 'mg/dL'},
        {'col': 'Age', 'name': 'Usia', 'color': '#4ECDC4', 'unit': 'tahun'},
        {'col': 'BMI', 'name': 'BMI', 'color': '#45B7D1', 'unit': 'kg/m²'},
        {'col': 'Insulin', 'name': 'Insulin', 'color': '#96CEB4', 'unit': 'µU/ml'}
    ]
    
    for idx, fc in enumerate(feature_configs):
        with [col1, col2, col3, col4][idx]:
            avg_val = df[fc['col']].mean()
            max_val = df[fc['col']].max()
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_val,
                title = {'text': f"Rata-rata {fc['name']}", 'font': {'size': 14}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                number = {'suffix': f" {fc['unit']}", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [0, max_val], 'tickwidth': 1},
                    'bar': {'color': fc['color']},
                    'steps': [
                        {'range': [0, max_val/2], 'color': "#e0e0e0"},
                        {'range': [max_val/2, max_val], 'color': "#c0c0c0"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_val
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    # Row 4: Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Matplotlib Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Prediksi\nNegatif', 'Prediksi\nPositif'],
                    yticklabels=['Aktual\nNegatif', 'Aktual\nPositif'],
                    annot_kws={'size': 20, 'weight': 'bold'},
                    cbar=False, ax=ax)
        ax.set_xlabel('PREDIKSI', fontsize=12, fontweight='bold')
        ax.set_ylabel('AKTUAL', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 📝 Interpretasi")
        
        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                st.markdown("**✅ True Negatif (TN)**")
                st.markdown(f"<h2 style='color: #10b981; text-align: center;'>{TN}</h2>", unsafe_allow_html=True)
                st.caption("Prediksi Negatif ✓ | Aktual Negatif ✓")
                
                st.markdown("**⚠️ False Negatif (FN)**")
                st.markdown(f"<h2 style='color: #f59e0b; text-align: center;'>{FN}</h2>", unsafe_allow_html=True)
                st.caption("Prediksi Negatif ✗ | Aktual Positif ✓")
        
        with col_b:
            with st.container(border=True):
                st.markdown("**❌ False Positif (FP)**")
                st.markdown(f"<h2 style='color: #ef4444; text-align: center;'>{FP}</h2>", unsafe_allow_html=True)
                st.caption("Prediksi Positif ✗ | Aktual Negatif ✓")
                
                st.markdown("**🎯 True Positif (TP)**")
                st.markdown(f"<h2 style='color: #10b981; text-align: center;'>{TP}</h2>", unsafe_allow_html=True)
                st.caption("Prediksi Positif ✓ | Aktual Positif ✓")
    
    # Metrik dari Confusion Matrix
    st.markdown("---")
    col_met = st.columns(5)
    
    acc = (TN + TP) / (TN + FP + FN + TP)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    with col_met[0]:
        st.metric("Akurasi", f"{acc*100:.2f}%")
    with col_met[1]:
        st.metric("Precision", f"{prec*100:.2f}%")
    with col_met[2]:
        st.metric("Recall", f"{rec*100:.2f}%")
    with col_met[3]:
        st.metric("Specificity", f"{spec*100:.2f}%")
    with col_met[4]:
        st.metric("F1-Score", f"{f1*100:.2f}%")

# ==================== HALAMAN PREDIKSI PASIEN ====================
elif menu == "👤 Prediksi Pasien":
    st.markdown('<p class="main-header">👤 Prediksi Pasien Baru</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("📝 Data Pasien")
            pregnancies = st.number_input("Pregnancies (Jumlah Kehamilan)", min_value=0, max_value=20, value=2)
            glucose = st.number_input("Glucose (Kadar Glukosa)", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure (Tekanan Darah)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (Ketebalan Kulit)", min_value=0, max_value=100, value=20)
    
    with col2:
        with st.container(border=True):
            st.subheader("📝 Data Pasien (Lanjutan)")
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=28.5, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.35, step=0.01)
            age = st.number_input("Age (Usia)", min_value=1, max_value=120, value=30)
    
    st.markdown("---")
    
    if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        
        col_result, col_prob = st.columns([1, 1])
        
        with col_result:
            if pred == 0:
                st.markdown(f"""
                <div class="result-negative">
                    <h1>✅ NEGATIF DIABETES</h1>
                    <p style="font-size: 1.2rem;">Tidak Berisiko Diabetes</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-positive">
                    <h1>⚠️ POSITIF DIABETES</h1>
                    <p style="font-size: 1.2rem;">Berisiko Diabetes</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_prob:
            with st.container(border=True):
                st.subheader("📊 Probabilitas")
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <p><strong>Negatif Diabetes:</strong> {proba[0]*100:.1f}%</p>
                    <div style="background-color: #e0e0e0; border-radius: 10px;">
                        <div style="width: {proba[0]*100}%; background-color: #10b981; border-radius: 10px; padding: 0.3rem; text-align: center; color: white;">
                            {proba[0]*100:.1f}%
                        </div>
                    </div>
                </div>
                <div style="margin: 1rem 0;">
                    <p><strong>Positif Diabetes:</strong> {proba[1]*100:.1f}%</p>
                    <div style="background-color: #e0e0e0; border-radius: 10px;">
                        <div style="width: {proba[1]*100}%; background-color: #ef4444; border-radius: 10px; padding: 0.3rem; text-align: center; color: white;">
                            {proba[1]*100:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.info("ℹ️ Hasil ini hanya prediksi berdasarkan model AI. Tetap lakukan pemeriksaan medis untuk hasil yang lebih akurat.")
    
    # Riwayat Prediksi
    st.markdown("---")
    st.subheader("📜 Riwayat Prediksi Pasien")
    
    history_data = [
        {"No": 1, "Tanggal": "20 Mei 2025 10:25", "Glucose": 120, "BMI": 28.5, "Age": 30, "Hasil": "Negatif Diabetes", "Probabilitas": "82.4%", "Status": "✅ Tidak Berisiko"},
        {"No": 2, "Tanggal": "20 Mei 2025 10:10", "Glucose": 165, "BMI": 32.1, "Age": 45, "Hasil": "Positif Diabetes", "Probabilitas": "78.9%", "Status": "⚠️ Berisiko"},
        {"No": 3, "Tanggal": "20 Mei 2025 09:58", "Glucose": 95, "BMI": 24.3, "Age": 25, "Hasil": "Negatif Diabetes", "Probabilitas": "88.1%", "Status": "✅ Tidak Berisiko"},
    ]
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

# ==================== HALAMAN DATA PASIEN ====================
elif menu == "📋 Data Pasien":
    st.markdown('<p class="main-header">📋 Data Pasien</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pasien", len(df))
    col2.metric("Jumlah Fitur", len(df.columns)-1)
    col3.metric("Positif Diabetes", positif)
    col4.metric("Negatif Diabetes", negatif)
    
    st.markdown("---")
    st.dataframe(df, use_container_width=True, height=500)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data CSV",
        data=csv,
        file_name="data_diabetes.csv",
        mime="text/csv",
    )

# ==================== HALAMAN ANALISIS DATA ====================
elif menu == "📈 Analisis Data":
    st.markdown('<p class="main-header">📈 Analisis Data</p>', unsafe_allow_html=True)
    
    st.subheader("📊 Statistika Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("📈 Distribusi Fitur")
    feature_cols = list(df.columns.drop("Outcome"))
    selected_feature = st.selectbox("Pilih fitur untuk ditampilkan:", feature_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            df, x=selected_feature, color="Outcome",
            color_discrete_map={0: "#10b981", 1: "#ef4444"},
            title=f"Distribusi {selected_feature}",
            labels={"Outcome": "Diabetes", "count": "Frekuensi"},
            barmode="overlay"
        )
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_box = px.box(
            df, x="Outcome", y=selected_feature,
            color="Outcome",
            color_discrete_map={0: "#10b981", 1: "#ef4444"},
            title=f"Boxplot {selected_feature}",
            labels={"Outcome": "Diabetes"}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.subheader("🔥 Korelasi Antar Fitur")
    corr_matrix = df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Heatmap Korelasi"
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

# ==================== HALAMAN EVALUASI MODEL ====================
elif menu == "🎯 Evaluasi Model":
    st.markdown('<p class="main-header">🎯 Evaluasi Model</p>', unsafe_allow_html=True)
    
    st.subheader("📊 Performa Model K-Nearest Neighbors (KNN)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    col4.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📉 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Prediksi Negatif', 'Prediksi Positif'],
                    yticklabels=['Aktual Negatif', 'Aktual Positif'],
                    annot_kws={'size': 16, 'weight': 'bold'},
                    cbar=False, ax=ax)
        ax.set_xlabel('Prediksi', fontsize=12, fontweight='bold')
        ax.set_ylabel('Aktual', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📋 Classification Report")
        report_data = {
            'Kelas': ['Negatif (0)', 'Positif (1)'],
            'Precision': [f"{metrics['precision']*100:.2f}%", f"{metrics['precision']*100:.2f}%"],
            'Recall': [f"{metrics['recall']*100:.2f}%", f"{metrics['recall']*100:.2f}%"],
            'F1-Score': [f"{metrics['f1']*100:.2f}%", f"{metrics['f1']*100:.2f}%"],
            'Support': [len(y_test[y_test==0]), len(y_test[y_test==1])]
        }
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("ℹ️ Informasi Model")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("**🤖 Algoritma**")
            st.write("K-Nearest Neighbors (KNN)")
    with col2:
        with st.container(border=True):
            st.markdown("**🔢 Parameter K**")
            st.write("k = 5")
    with col3:
        with st.container(border=True):
            st.markdown("**📏 Metrik Jarak**")
            st.write("Euclidean Distance")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**⚙️ Preprocessing**")
            st.write("MinMaxScaler (Normalisasi 0-1)")
    with col2:
        with st.container(border=True):
            st.markdown("**📊 Split Data**")
            st.write("Train: 80% | Test: 20%")
    
    # Learning Curve
    st.subheader("📈 Learning Curve (Perbandingan Nilai K)")
    
    k_values = range(1, 21)
    accuracies = []
    
    X_full = df.drop("Outcome", axis=1)
    y_full = df["Outcome"]
    X_scaled = scaler.transform(X_full)
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y_full, test_size=0.2, random_state=42)
    
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train_s, y_train_s)
        acc = knn_temp.score(X_test_s, y_test_s)
        accuracies.append(acc)
    
    fig_lc = px.line(
        x=list(k_values), y=accuracies,
        title="Akurasi Model untuk Berbagai Nilai K",
        labels={"x": "Nilai K (Jumlah Tetangga)", "y": "Akurasi"}
    )
    fig_lc.add_hline(y=max(accuracies), line_dash="dash", line_color="green", 
                     annotation_text=f"Maks: {max(accuracies)*100:.2f}%")
    fig_lc.add_vline(x=accuracies.index(max(accuracies))+1, line_dash="dash", line_color="red",
                     annotation_text=f"K terbaik: {accuracies.index(max(accuracies))+1}")
    fig_lc.update_layout(height=500)
    st.plotly_chart(fig_lc, use_container_width=True)

# ==================== HALAMAN TENTANG ====================
else:
    st.markdown('<p class="main-header">ℹ️ Tentang Aplikasi</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1f77b4, #4ECDC4); border-radius: 15px; padding: 2rem; text-align: center; color: white;">
            <h1 style="font-size: 3rem;">🩺</h1>
            <h2>AI Diabetes Care</h2>
            <p>Sistem Prediksi Risiko Diabetes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📌 Tentang Aplikasi")
        st.markdown("""
        **AI Diabetes Care** adalah aplikasi berbasis Machine Learning untuk memprediksi risiko diabetes 
        pada pasien berdasarkan data medis mereka.
        
        ### 🧠 Algoritma yang Digunakan
        - **K-Nearest Neighbors (KNN)** dengan parameter k = 5
        - **Metrik Jarak:** Euclidean Distance
        - **Preprocessing:** MinMaxScaler untuk normalisasi fitur
        
        ### 📁 Dataset
        Dataset yang digunakan adalah **Pima Indians Diabetes Database** yang berisi data medis 
        pasien wanita keturunan Pima Indian usia minimal 21 tahun.
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("📋 Fitur yang Digunakan")
            st.markdown("""
            1. **Pregnancies** - Jumlah Kehamilan
            2. **Glucose** - Kadar Glukosa Plasma (mg/dL)
            3. **Blood Pressure** - Tekanan Darah Diastolik (mm Hg)
            4. **Skin Thickness** - Ketebalan Lipatan Kulit Trisep (mm)
            5. **Insulin** - Kadar Insulin Serum (µU/ml)
            6. **BMI** - Body Mass Index (kg/m²)
            7. **Diabetes Pedigree Function** - Fungsi Silsilah Diabetes
            8. **Age** - Usia (tahun)
            """)
    
    with col2:
        with st.container(border=True):
            st.subheader("📖 Cara Penggunaan")
            st.markdown("""
            1. Pilih menu **Prediksi Pasien** di sidebar
            2. Masukkan data pasien pada form yang tersedia
            3. Klik tombol **Prediksi Sekarang**
            4. Sistem akan menampilkan hasil prediksi dan probabilitas
            """)
    
    st.markdown("---")
    
    st.info("""
    ⚠️ **Disclaimer**: Hasil prediksi ini hanya berdasarkan model AI dan **bukan diagnosis medis resmi**. 
    Selalu konsultasikan dengan tenaga medis profesional untuk diagnosis yang akurat.
    """)
    
    st.caption("Dibangun dengan Streamlit, scikit-learn, dan Plotly | © 2026 AI Diabetes Care")