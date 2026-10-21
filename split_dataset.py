
import pandas as pd
from pathlib import Path

# 1. Baca dataset RetailRocket
file_input = Path("events.csv")

df = pd.read_csv(file_input)

print("Jumlah data awal:", len(df))
print("Kolom dataset:", df.columns.tolist())

# 2. Konversi timestamp dari milidetik Unix ke datetime
df["tanggal"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    utc=True
)

# 3. Pisahkan data training dan testing
# Tanggal akhir dibuat eksklusif dengan menambahkan satu hari.
training = df[
    (df["tanggal"] >= "2015-05-09") &
    (df["tanggal"] < "2015-07-10")
].copy()

testing = df[
    (df["tanggal"] >= "2015-07-18") &
    (df["tanggal"] < "2015-08-03")
].copy()

# 4. Simpan hasil ke CSV
training.to_csv("training.csv", index=False)
testing.to_csv("testing.csv", index=False)

# 5. Tampilkan ringkasan
print("\n=== HASIL PEMBAGIAN DATA ===")
print("Jumlah data training:", len(training))
print("Jumlah data testing:", len(testing))

if not training.empty:
    print(
        "Rentang training:",
        training["tanggal"].min(),
        "sampai",
        training["tanggal"].max()
    )

if not testing.empty:
    print(
        "Rentang testing:",
        testing["tanggal"].min(),
        "sampai",
        testing["tanggal"].max()
    )

print("\nFile training.csv dan testing.csv berhasil disimpan.")
