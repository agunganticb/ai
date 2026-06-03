import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data TANPA Pregnancies
df = pd.read_csv("diabetes_tanpa_pregnancies.csv")

# Pisahkan fitur dan target (semua kolom kecuali Outcome)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("=" * 50)
print("TRAINING MODEL KNN (TANPA PREGNANCIES)")
print("=" * 50)
print(f"Jumlah fitur: {X.shape[1]}")
print(f"Fitur: {X.columns.tolist()}")
print(f"Jumlah data: {len(df)}")

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Train model KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
print(f"\n✅ Akurasi: {akurasi*100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Simpan model dan scaler
joblib.dump(model, "model_knn.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model berhasil disimpan!")
print("   File: model_knn.pkl dan scaler.pkl")
print(f"   Jumlah fitur input: {X.shape[1]}")