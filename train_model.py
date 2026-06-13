import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("📊 Memuat dataset...")

# Buat dataset sample jika tidak ada file CSV
import os
if not os.path.exists('diabetes.csv'):
    print("⚠️ File diabetes.csv tidak ditemukan, membuat dataset sample...")
    # Contoh dataset sederhana
    np.random.seed(42)
    n_samples = 500
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.randint(50, 200, n_samples),
        'BloodPressure': np.random.randint(60, 140, n_samples),
        'SkinThickness': np.random.randint(10, 50, n_samples),
        'Insulin': np.random.randint(0, 200, n_samples),
        'BMI': np.random.uniform(18, 45, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.5, n_samples),
        'Age': np.random.randint(21, 90, n_samples),
        'Outcome': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv('diabetes.csv', index=False)
    print("✅ Dataset sample telah dibuat!")
else:
    df = pd.read_csv('diabetes.csv')

print(f"📋 Dataset shape: {df.shape}")
print(f"📈 Kolom: {list(df.columns)}")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

print("🔄 Scaling fitur...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("🤖 Melatih model KNN...")
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_scaled, y)

# Evaluasi
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

print(f"✅ Akurasi model: {akurasi*100:.2f}%")
print(f"\n📊 Classification Report:\n{classification_report(y_test, y_pred)}")

print("💾 Menyimpan model dan scaler...")
joblib.dump(model, 'model_knn.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✨ Selesai! Model siap digunakan.")