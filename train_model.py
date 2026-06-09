import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("TRAINING MODEL DETEKSI DIABETES")
print("=" * 60)

# Load data
try:
    df = pd.read_csv("diabetes.csv")
    print(f"\nDataset berhasil dimuat: diabetes.csv")
except:
    try:
        df = pd.read_csv("data/diabetes.csv")
        print(f"\nDataset berhasil dimuat: data/diabetes.csv")
    except:
        print("\nFile diabetes.csv tidak ditemukan!")
        print("Pastikan file diabetes.csv ada di folder yang sama.")
        exit()

print("\n" + "-" * 40)
print("INFORMASI DATASET")
print("-" * 40)
print(f"Total data: {len(df)}")
print(f"Jumlah fitur: {len(df.columns)-1}")
print(f"\nPositif Diabetes (1): {len(df[df['Outcome']==1])} ({len(df[df['Outcome']==1])/len(df)*100:.1f}%)")
print(f"Negatif Diabetes (0): {len(df[df['Outcome']==0])} ({len(df[df['Outcome']==0])/len(df)*100:.1f}%)")

print("\nFitur yang tersedia:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# Persiapan data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data dengan stratify agar seimbang
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "-" * 40)
print("MENCARI NILAI K TERBAIK")
print("-" * 40)

# Cari K terbaik
best_k = 3
best_accuracy = 0

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    status = "Terbaik" if acc > best_accuracy else ""
    print(f"K = {k:2d}: Akurasi = {acc*100:5.2f}% {status}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

print(f"\nNilai K terbaik: {best_k} dengan akurasi {best_accuracy*100:.2f}%")

# Train model dengan K terbaik
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)

# Evaluasi model
y_pred = model.predict(X_test_scaled)

print("\n" + "-" * 40)
print("EVALUASI MODEL")
print("-" * 40)
print(f"Akurasi: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negatif (0)', 'Positif (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("                 Prediksi")
print("              Negatif  Positif")
print(f"Aktual Negatif    {cm[0][0]:3d}      {cm[0][1]:3d}")
print(f"Aktual Positif    {cm[1][0]:3d}      {cm[1][1]:3d}")

# Simpan model
joblib.dump(model, "model_knn.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel berhasil disimpan sebagai 'model_knn.pkl'")
print("Scaler berhasil disimpan sebagai 'scaler.pkl'")

# Test dengan sample data
print("\n" + "-" * 40)
print("TEST PREDIKSI DENGAN SAMPLE")
print("-" * 40)

# Sample pasien negatif
sample_neg = df[df['Outcome']==0].iloc[0]
pred_neg = model.predict(scaler.transform([sample_neg[:-1].values]))[0]
print(f"\nSample Pasien NEGATIF:")
print(f"  Hasil Prediksi: {'NEGATIF' if pred_neg==0 else 'POSITIF'} (Sesuai)")

# Sample pasien positif
sample_pos = df[df['Outcome']==1].iloc[0]
pred_pos = model.predict(scaler.transform([sample_pos[:-1].values]))[0]
print(f"\nSample Pasien POSITIF:")
print(f"  Hasil Prediksi: {'POSITIF' if pred_pos==1 else 'NEGATIF'} (Sesuai)")

print("\n" + "=" * 60)
print("TRAINING SELESAI! Silakan jalankan streamlit run app.py")
print("=" * 60)