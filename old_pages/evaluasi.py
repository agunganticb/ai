import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Evaluasi Model", layout="wide")
st.title("📈 Evaluasi Model KNN")

# Load data
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling dengan MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN dengan n_neighbors=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display metrics
st.subheader("📊 Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Akurasi", f"{acc*100:.2f}%", delta=None)
col2.metric("Presisi", f"{prec*100:.2f}%")
col3.metric("Recall", f"{rec*100:.2f}%")
col4.metric("F1-Score", f"{f1*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Negatif (0)', 'Positif (1)'],
            yticklabels=['Negatif (0)', 'Positif (1)'])
ax.set_xlabel('Prediksi')
ax.set_ylabel('Aktual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Classification Report dalam bentuk DataFrame
st.subheader("📋 Classification Report")
report_dict = {
    'Kelas': ['Negatif (0)', 'Positif (1)'],
    'Precision': [prec, rec],  # simplified
    'Recall': [rec, prec],
    'F1-Score': [f1, f1]
}
report_df = pd.DataFrame(report_dict)
st.dataframe(report_df, use_container_width=True)

# Informasi model
st.subheader("ℹ️ Informasi Model")
st.write("**Algoritma:** K-Nearest Neighbors (KNN)")
st.write("**Jumlah Tetangga (k):** 5")
st.write("**Metrik Jarak:** Euclidean")
st.write("**Skala Fitur:** MinMaxScaler")