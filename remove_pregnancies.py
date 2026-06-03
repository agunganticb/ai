import pandas as pd

# Baca file CSV asli (yang masih ada Pregnancies)
df = pd.read_csv("diabetes.csv")

print("=" * 50)
print("MEMBUANG KOLOM PREGNANCIES")
print("=" * 50)

# Lihat kolom sebelum dihapus
print("\n📋 Kolom SEBELUM dihapus:")
print(df.columns.tolist())
print(f"Jumlah kolom: {len(df.columns)}")
print(f"Jumlah baris: {len(df)}")

# Hapus kolom Pregnancies (kolom pertama)
df = df.drop("Pregnancies", axis=1)

# Lihat kolom setelah dihapus
print("\n📋 Kolom SETELAH dihapus:")
print(df.columns.tolist())
print(f"Jumlah kolom: {len(df.columns)}")

# Simpan ke file baru
df.to_csv("diabetes_tanpa_pregnancies.csv", index=False)

print("\n" + "=" * 50)
print("✅ BERHASIL!")
print("=" * 50)
print(f"File baru: diabetes_tanpa_pregnancies.csv")
print(f"Jumlah fitur: 7 (tanpa Pregnancies)")
print("\nFitur yang tersisa:")
for i, col in enumerate(df.columns[:-1], 1):
    print(f"  {i}. {col}")
print(f"  {len(df.columns)}. Outcome (target)")